#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_train_gnn.py
===============
GNN hyperparameter optimisation, training, evaluation, and artifact generation.
Parametric on ``--architecture`` so the same script can be re-run for different
GNN types against the same dataset.

Usage
-----
    python 03_train_gnn.py --run_dir runs/P00533_scaffold --architecture sage
    python 03_train_gnn.py --run_dir runs/P00533_scaffold --architecture gat --hpo_trials 30

Workflow
--------
    1. HPO via W&B Bayes sweep (expanded search space, arch-specific knobs)
    2. Final training with best hyperparameters
    3. Evaluation on val + test splits
    4. Comparison table with baselines (reads baseline_metrics.json)
    5. Top-5 molecules by predicted affinity
    6. Inference on custom SMILES
    7. Embedding extraction + PCA scatter plots
    8. Runtime / memory profiling summary
    9. LaTeX manifest

Outputs (all under <run_dir>/paper_artifacts/)
"""

import argparse
import json
import os
import platform
import random
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from dta_gnn.models.gnn import (
    GnnTrainConfig,
    _get_device,
    extract_gnn_embeddings_on_run,
    train_gnn_on_run,
)
from dta_gnn.models.hyperopt import HyperoptConfig, optimize_gnn_wandb
from dta_gnn.models.predict import predict_with_gnn


_VALID_ARCHITECTURES = (
    "gin", "gcn", "gat", "sage", "pna", "transformer",
    "tag", "arma", "cheb", "supergat",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GNN HPO + train + evaluate + artifacts (parametric on architecture)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Core --
    p.add_argument("--run_dir", type=str, required=True,
                    help="Scaffold-split run directory (from 01_build_dataset.py).")
    p.add_argument("--architecture", type=str, required=True,
                    choices=_VALID_ARCHITECTURES,
                    help="GNN architecture to optimise and train.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument("--epochs", type=int, default=100,
                    help="Training epochs for final model.")

    # -- HPO ranges (common, searched) --
    p.add_argument("--hpo_trials", type=int, default=50)
    p.add_argument("--lr_min", type=float, default=1e-4)
    p.add_argument("--lr_max", type=float, default=5e-3)
    p.add_argument("--num_layers_min", type=int, default=2)
    p.add_argument("--num_layers_max", type=int, default=5)
    p.add_argument("--dropout_min", type=float, default=0.0)
    p.add_argument("--dropout_max", type=float, default=0.4)
    p.add_argument("--weight_decay_min", type=float, default=1e-6)
    p.add_argument("--weight_decay_max", type=float, default=1e-3)
    p.add_argument("--pooling_choices", nargs="+",
                    default=["add", "mean", "max", "attention"])

    # -- Fixed during HPO (override via CLI if needed) --
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--head_mlp_layers", type=int, default=2)

    # -- W&B --
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    # -- Skip HPO --
    p.add_argument("--skip_hpo", action="store_true",
                    help="Skip W&B HPO and use fallback defaults.")
    p.add_argument("--fallback_lr", type=float, default=1e-3)
    p.add_argument("--fallback_num_layers", type=int, default=3)
    p.add_argument("--fallback_hidden_dim", type=int, default=256)
    p.add_argument("--fallback_embedding_dim", type=int, default=128)
    p.add_argument("--fallback_dropout", type=float, default=0.1)
    p.add_argument("--fallback_batch_size", type=int, default=64)
    p.add_argument("--fallback_pooling", type=str, default="add")
    p.add_argument("--fallback_head_mlp_layers", type=int, default=2)
    p.add_argument("--fallback_weight_decay", type=float, default=1e-4)

    # -- Inference --
    p.add_argument("--inference_smiles", nargs="+", default=[
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
        "CC(C)NCC(O)COc1cccc2ccccc12",
        "CN1CCN(CC1)C2=NC3=CC=CC=C3N2",
    ])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def _mem_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _gpu_mem_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        pass
    return 0.0


def _reset_gpu_stats() -> None:
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HPO
# ---------------------------------------------------------------------------

def _build_hpo_config(args: argparse.Namespace, seed: int) -> HyperoptConfig:
    """Build an expanded HyperoptConfig with architecture-specific knobs."""
    arch = args.architecture

    kwargs: dict = dict(
        model_type="GNN",
        architecture=arch,
        n_trials=args.hpo_trials,
        sampler_seed=seed,
        epochs_default=args.epochs,
        device="auto",
        # Searched knobs (~6 dims + 1 arch-specific)
        optimize_lr=True,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        optimize_num_layers=True,
        num_layers_min=args.num_layers_min,
        num_layers_max=args.num_layers_max,
        optimize_dropout=True,
        dropout_min=args.dropout_min,
        dropout_max=args.dropout_max,
        optimize_weight_decay=True,
        weight_decay_min=args.weight_decay_min,
        weight_decay_max=args.weight_decay_max,
        optimize_pooling=True,
        pooling_choices=args.pooling_choices,
        optimize_residual=True,
        # Fixed dimensions (not searched)
        optimize_hidden_dim=False,
        hidden_dim_default=args.hidden_dim,
        optimize_embedding_dim=False,
        embedding_dim_default=args.embedding_dim,
        optimize_batch_size=False,
        batch_size_default=args.batch_size,
        optimize_head_mlp_layers=False,
        head_mlp_layers_default=args.head_mlp_layers,
    )

    # Architecture-specific knobs
    if arch == "gin":
        kwargs.update(
            optimize_gin_conv_mlp_layers=True,
            optimize_gin_train_eps=True,
            optimize_gin_eps=True,
        )
    elif arch == "gat":
        kwargs.update(optimize_gat_heads=True)
    elif arch == "sage":
        kwargs.update(
            optimize_sage_aggr=True,
            sage_aggr_choices=["mean", "max", "pool"],
        )
    elif arch == "transformer":
        kwargs.update(optimize_transformer_heads=True)
    elif arch == "tag":
        kwargs.update(optimize_tag_k=True)
    elif arch == "arma":
        kwargs.update(
            optimize_arma_stacks=True,
            optimize_arma_layers=True,
        )
    elif arch == "cheb":
        kwargs.update(optimize_cheb_k=True)
    elif arch == "supergat":
        kwargs.update(
            optimize_supergat_heads=True,
            optimize_supergat_attention_type=True,
        )

    return HyperoptConfig(**kwargs)


def run_hpo(
    run_dir: Path,
    paper_artifacts: Path,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    """Run HPO and return the full best_params dict."""
    arch = args.architecture

    if args.skip_hpo:
        best_params: dict = {
            "architecture": arch,
            "lr": args.fallback_lr,
            "num_layers": args.fallback_num_layers,
            "hidden_dim": args.fallback_hidden_dim,
            "embedding_dim": args.fallback_embedding_dim,
            "dropout": args.fallback_dropout,
            "batch_size": args.fallback_batch_size,
            "pooling": args.fallback_pooling,
            "head_mlp_layers": args.fallback_head_mlp_layers,
            "weight_decay": args.fallback_weight_decay,
            "residual": False,
        }
        print(f"[HPO skipped] Using fallback params: {best_params}")
        return best_params

    old_notebook_name = os.environ.get("WANDB_NOTEBOOK_NAME")
    os.environ["WANDB_NOTEBOOK_NAME"] = ""

    hpo_config = _build_hpo_config(args, seed)

    # Derive project name from metadata if available
    meta_path = run_dir / "metadata.json"
    uid = "unknown"
    if meta_path.exists():
        try:
            uid = json.loads(meta_path.read_text()).get("uniprot_id", "unknown")
        except Exception:
            pass
    wandb_project = args.wandb_project or f"dta_gnn_{uid.lower()}"

    try:
        result_hpo = optimize_gnn_wandb(
            run_dir,
            config=hpo_config,
            project=wandb_project,
            entity=args.wandb_entity,
            sweep_name=f"{uid.lower()}_{arch}_hpo",
        )
    finally:
        if old_notebook_name is not None:
            os.environ["WANDB_NOTEBOOK_NAME"] = old_notebook_name
        elif "WANDB_NOTEBOOK_NAME" in os.environ:
            del os.environ["WANDB_NOTEBOOK_NAME"]

    best_params = dict(result_hpo.best_params)
    best_params.setdefault("architecture", arch)

    print(f"\nBest params (val_score={result_hpo.best_value:.4f}):")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    # -- Save HPO artifacts --
    best_params_out = {**best_params, "val_score": float(result_hpo.best_value)}
    (paper_artifacts / "best_hyperparameters.json").write_text(
        json.dumps(best_params_out, indent=2)
    )

    # TeX table of all best hyperparameters
    lines = [
        f"%% Best hyperparameters ({arch.upper()} HPO, W&B Bayes sweep)\n",
        "\\begin{tabular}{ll}\n",
        "  \\hline\n",
    ]
    for k, v in sorted(best_params_out.items()):
        if k == "val_score":
            label = "Val.\\ score"
        else:
            label = k.replace("_", " ").title()
        if isinstance(v, float):
            lines.append(f"  {label} & {v:.4g} \\\\\n")
        else:
            lines.append(f"  {label} & {v} \\\\\n")
    lines += ["  \\hline\n", "\\end{tabular}\n"]
    (paper_artifacts / "best_hyperparameters.tex").write_text("".join(lines))

    df_trials = pd.DataFrame([best_params_out])
    df_trials.insert(0, "trial", result_hpo.best_trial_number)
    df_trials.to_csv(paper_artifacts / "table_hpo_best.csv", index=False)
    with open(paper_artifacts / "table_hpo_best.tex", "w") as fh:
        fh.write(df_trials.to_latex(index=False, float_format="%.4f", na_rep="--"))

    return best_params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    run_dir: Path,
    best_params: dict,
    args: argparse.Namespace,
    seed: int,
):
    """Train the GNN with best hyperparameters from HPO."""
    arch = args.architecture
    config_final = GnnTrainConfig(
        architecture=arch,
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("weight_decay", 0.0)),
        num_layers=int(best_params.get("num_layers", 3)),
        hidden_dim=int(best_params.get("hidden_dim", 256)),
        embedding_dim=int(best_params.get("embedding_dim", 128)),
        dropout=float(best_params.get("dropout", 0.1)),
        batch_size=int(best_params.get("batch_size", 64)),
        pooling=str(best_params.get("pooling", "add")),
        residual=bool(best_params.get("residual", False)),
        head_mlp_layers=int(best_params.get("head_mlp_layers", 2)),
        # GIN-specific
        gin_conv_mlp_layers=int(best_params.get("gin_conv_mlp_layers", 2)),
        gin_train_eps=bool(best_params.get("gin_train_eps", False)),
        gin_eps=float(best_params.get("gin_eps", 0.0)),
        # GAT-specific
        gat_heads=int(best_params.get("gat_heads", 4)),
        # SAGE-specific
        sage_aggr=str(best_params.get("sage_aggr", "mean")),
        # Transformer-specific
        transformer_heads=int(best_params.get("transformer_heads", 4)),
        # TAG-specific
        tag_k=int(best_params.get("tag_k", 2)),
        # ARMA-specific
        arma_num_stacks=int(best_params.get("arma_num_stacks", 1)),
        arma_num_layers=int(best_params.get("arma_num_layers", 1)),
        # Cheb-specific
        cheb_k=int(best_params.get("cheb_k", 2)),
        # SuperGAT-specific
        supergat_heads=int(best_params.get("supergat_heads", 4)),
        supergat_attention_type=str(best_params.get("supergat_attention_type", "MX")),
        epochs=args.epochs,
        random_seed=seed,
        device="auto",
    )
    result = train_gnn_on_run(run_dir, config=config_final)
    shutil.copy2(
        run_dir / f"model_gnn_{arch}.pt",
        run_dir / "best_model.pt",
    )
    print(f"Best model saved to {run_dir / 'best_model.pt'}")
    return result, config_final


# ---------------------------------------------------------------------------
# Evaluation + comparison table
# ---------------------------------------------------------------------------

def evaluate_model(
    run_dir: Path,
    paper_artifacts: Path,
    result_final,
    arch: str,
) -> None:
    """Compute val/test metrics, build comparison table with baselines, save top-5."""
    pred_path = run_dir / f"model_predictions_gnn_{arch}.csv"
    df_pred = pd.read_csv(pred_path)

    splits_metrics = result_final.metrics["splits"]

    # -- GNN performance table (all metrics) --
    gnn_rows = []
    for split_name in ("val", "test"):
        m = splits_metrics.get(split_name, {})
        if m:
            gnn_rows.append({
                "Split": split_name,
                "RMSE": m.get("rmse"),
                "MAE": m.get("mae"),
                "R2": m.get("r2"),
                "Pearson": m.get("pearson_r"),
                "Spearman": m.get("spearman_r"),
            })
    perf_df = pd.DataFrame(gnn_rows)
    print(f"\n-- GNN ({arch}) Performance --")
    print(perf_df.to_string(index=False, float_format="%.4f", na_rep="--"))

    perf_df.to_csv(paper_artifacts / "table_performance.csv", index=False)
    with open(paper_artifacts / "table_performance.tex", "w") as fh:
        fh.write(perf_df.to_latex(index=False, float_format="%.4f", na_rep="--"))

    # -- Comparison table with baselines --
    comparison_rows: list[dict] = []

    baseline_path = run_dir / "baseline_metrics.json"
    if baseline_path.exists():
        baselines = json.loads(baseline_path.read_text())
        for model_name, splits in baselines.items():
            for split_name in ("val", "test"):
                m = splits.get(split_name)
                if m:
                    comparison_rows.append({
                        "Model": model_name,
                        "Split": split_name,
                        "RMSE": m.get("rmse"),
                        "MAE": m.get("mae"),
                        "R2": m.get("r2"),
                        "Pearson": m.get("pearson_r"),
                        "Spearman": m.get("spearman_r"),
                    })

    gnn_label = f"GNN ({arch})"
    for split_name in ("val", "test"):
        m = splits_metrics.get(split_name, {})
        if m:
            comparison_rows.append({
                "Model": gnn_label,
                "Split": split_name,
                "RMSE": m.get("rmse"),
                "MAE": m.get("mae"),
                "R2": m.get("r2"),
                "Pearson": m.get("pearson_r"),
                "Spearman": m.get("spearman_r"),
            })

    comp_df = pd.DataFrame(comparison_rows)
    print(f"\n-- Model Comparison --")
    print(comp_df.to_string(index=False, float_format="%.4f", na_rep="--"))

    comp_df.to_csv(paper_artifacts / "table_model_comparison.csv", index=False)
    with open(paper_artifacts / "table_model_comparison.tex", "w") as fh:
        fh.write(comp_df.to_latex(index=False, float_format="%.4f", na_rep="--"))

    # -- Top-5 molecules --
    test_pred = df_pred[df_pred["split"] == "test"].copy()
    test_pred = test_pred.sort_values("y_pred", ascending=False).head(5)
    compounds = pd.read_csv(run_dir / "compounds.csv")
    test_pred = test_pred.merge(
        compounds[["molecule_chembl_id", "smiles"]],
        on="molecule_chembl_id",
        how="left",
    )
    n_top = len(test_pred)
    top5 = pd.DataFrame({
        "rank": range(1, n_top + 1),
        "SMILES": test_pred["smiles"].values,
        "predicted_affinity": test_pred["y_pred"].values,
        "true_affinity": test_pred["label"].values,
    })
    print(f"\n-- Top-{n_top} test molecules by predicted affinity --")
    print(top5.to_string(index=False))

    top5.to_csv(paper_artifacts / "table_top5_molecules.csv", index=False)
    with open(paper_artifacts / "table_top5_molecules.tex", "w") as fh:
        fh.write(top5.to_latex(index=False, float_format="%.4f", na_rep="--"))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    run_dir: Path,
    paper_artifacts: Path,
    smiles_list: list[str],
    arch: str,
) -> None:
    """Run the saved model on custom SMILES and save results."""
    result_inf = predict_with_gnn(run_dir, smiles_list, architecture=arch)
    inf_table = result_inf.predictions[["smiles", "prediction"]].copy()
    inf_table.columns = ["SMILES", "predicted_affinity"]
    print("\n-- Inference results --")
    print(inf_table.to_string(index=False))

    inf_table.to_csv(paper_artifacts / "table_inference.csv", index=False)
    with open(paper_artifacts / "table_inference.tex", "w") as fh:
        fh.write(inf_table.to_latex(index=False, float_format="%.4f", na_rep="--"))


# ---------------------------------------------------------------------------
# Embeddings + PCA
# ---------------------------------------------------------------------------

def run_embedding_pca(
    run_dir: Path,
    paper_artifacts: Path,
    arch: str,
    seed: int,
) -> None:
    """Extract test-set embeddings, reduce to 2D via PCA, save scatter plots."""
    emb_result = extract_gnn_embeddings_on_run(
        run_dir, batch_size=256, device=None
    )
    npz = np.load(emb_result.embeddings_path, allow_pickle=True)
    all_ids = npz["molecule_chembl_id"]
    all_emb = npz["embeddings"]

    dataset = pd.read_csv(run_dir / "dataset.csv")
    test_ids = set(
        dataset[dataset["split"] == "test"]["molecule_chembl_id"].astype(str)
    )
    idx_test = [i for i, mid in enumerate(all_ids) if str(mid) in test_ids]
    emb_test = all_emb[idx_test]
    ids_test = all_ids[idx_test]

    pred_df = pd.read_csv(run_dir / f"model_predictions_gnn_{arch}.csv")
    pred_test = pred_df[pred_df["split"] == "test"].set_index("molecule_chembl_id")

    def _first(x):
        return np.atleast_1d(x).flat[0]

    y_pred = np.array([_first(pred_test.loc[str(mid), "y_pred"]) for mid in ids_test])
    y_true = np.array([_first(pred_test.loc[str(mid), "label"]) for mid in ids_test])

    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(emb_test)

    # Plot 1: coloured by predicted affinity
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=y_pred, cmap="viridis", s=20)
    plt.colorbar(sc, ax=ax, label="Predicted affinity")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Test embeddings (PCA) - predicted ({arch})")
    plt.tight_layout()
    plt.savefig(run_dir / "embeddings_pca_predicted.png", dpi=120)
    plt.savefig(paper_artifacts / "fig_embeddings_pca_predicted.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 2: coloured by ground truth
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=y_true, cmap="viridis", s=20)
    plt.colorbar(sc, ax=ax, label="Ground truth affinity")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Test embeddings (PCA) - ground truth ({arch})")
    plt.tight_layout()
    plt.savefig(run_dir / "embeddings_pca_ground_truth.png", dpi=120)
    plt.savefig(paper_artifacts / "fig_embeddings_pca_ground_truth.pdf", bbox_inches="tight")
    plt.close(fig)

    # Plot 3: predicted vs true with diagonal
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", label="y = x")
    ax.set_xlabel("Ground truth affinity")
    ax.set_ylabel("Predicted affinity")
    ax.set_title(f"Predicted vs ground truth - test ({arch})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "predicted_vs_true.png", dpi=120)
    plt.savefig(paper_artifacts / "fig_predicted_vs_true.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Embedding PCA plots saved.")


# ---------------------------------------------------------------------------
# Runtime / memory summary
# ---------------------------------------------------------------------------

def print_and_save_profile(profile: dict, paper_artifacts: Path) -> None:
    """Print consolidated profiling table and save CSV / TeX / JSON."""
    try:
        import torch
        device_str = (
            f"GPU ({torch.cuda.get_device_name(0)})"
            if torch.cuda.is_available()
            else "CPU"
        )
    except Exception:
        device_str = "CPU"

    cpu_info = platform.processor() or platform.machine()
    print(f"\nHardware: {device_str}  |  CPU: {cpu_info}")
    print(f"Python:   {platform.python_version()}  |  OS: {platform.system()} {platform.release()}\n")

    rows = []
    for phase, d in profile.items():
        row = {"Phase": phase.replace("_", " ").title()}
        row["Wall-clock (s)"] = d.get("wall_s", float("nan"))
        if "wall_per_epoch_s" in d:
            row["Per-epoch (s)"] = d.get("wall_per_epoch_s", float("nan"))
            row["Epochs"] = int(d.get("n_epochs", 0))
        else:
            row["Per-epoch (s)"] = "---"
            row["Epochs"] = "---"
        row["dRAM (MiB)"] = d.get("delta_ram_mb", float("nan"))
        row["Peak GPU (MiB)"] = d.get("peak_gpu_mb", 0.0)
        rows.append(row)

    df_rt = pd.DataFrame(rows)
    print(df_rt.to_string(index=False))

    df_rt.to_csv(paper_artifacts / "table_runtime_profile.csv", index=False)

    latex_lines = [
        "\\begin{table}[htbp]\n",
        "  \\centering\n",
        f"  \\caption{{Wall-clock runtimes and memory footprints on {device_str}.}}\n",
        "  \\label{tab:runtime}\n",
        df_rt.to_latex(index=False, na_rep="---"),
        "\\end{table}\n",
    ]
    with open(paper_artifacts / "table_runtime_profile.tex", "w") as fh:
        fh.writelines(latex_lines)

    profile_out = {
        "device": device_str,
        "cpu": cpu_info,
        "python": platform.python_version(),
        "phases": profile,
    }
    with open(paper_artifacts / "runtime_profile.json", "w") as fh:
        json.dump(profile_out, fh, indent=2)

    print(f"\nProfiling artifacts saved to: {paper_artifacts}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(paper_artifacts: Path, uid: str, arch: str) -> None:
    manifest = f"""# Paper artifacts for LaTeX ({uid} DTA-GNN run, {arch} architecture)

## Tables (use \\input{{path/to/file.tex}} or paste content)
- table_leakage_ratio.csv / .tex    -- Leakage ratio: Random vs Scaffold
- table_hpo_best.csv / .tex         -- Best HPO trial (all hyperparameters)
- table_performance.csv / .tex      -- Val/Test RMSE, MAE, R2, Pearson, Spearman
- table_model_comparison.csv / .tex -- Baselines (RF, SVR) vs GNN ({arch})
- table_top5_molecules.csv / .tex   -- Top 5 test molecules by predicted affinity
- table_inference.csv / .tex        -- Inference on provided SMILES
- table_runtime_profile.csv / .tex  -- Runtime and memory profiling

## Best hyperparameters
- best_hyperparameters.json         -- All tuned hyperparameters + val_score
- best_hyperparameters.tex          -- LaTeX tabular (ready to \\input)

## Figures (use \\includegraphics[width=...]{{path/to/file.pdf}})
- fig_leakage_ratio.pdf
- fig_embeddings_pca_predicted.pdf
- fig_embeddings_pca_ground_truth.pdf
- fig_predicted_vs_true.pdf

## Profiling
- runtime_profile.json              -- Structured runtime data

## Example LaTeX
# % \\input{{paper_artifacts/table_model_comparison.tex}}
# % \\includegraphics[width=0.45\\textwidth]{{paper_artifacts/fig_predicted_vs_true.pdf}}
"""
    (paper_artifacts / "LATEX_MANIFEST.txt").write_text(manifest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    arch = args.architecture
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    run_dir = Path(args.run_dir).resolve()
    paper_artifacts = run_dir / f"paper_artifacts_{arch}"
    paper_artifacts.mkdir(parents=True, exist_ok=True)

    if not (run_dir / "dataset.csv").exists():
        raise FileNotFoundError(
            f"No dataset.csv in {run_dir}. Run 01_build_dataset.py first."
        )

    device = _get_device(None)
    print(f"Architecture: {arch}")
    print(f"Run dir:      {run_dir}")
    print(f"Seed: {seed}  |  Device: {device}\n")

    # Read UID from metadata
    uid = "unknown"
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        try:
            uid = json.loads(meta_path.read_text()).get("uniprot_id", "unknown")
        except Exception:
            pass

    profile: dict = {}

    # -- 1. HPO --
    print("=" * 60)
    print(f"HPO for {arch.upper()}")
    print("=" * 60)
    _reset_gpu_stats()
    _mem_before = _mem_mb()
    _t0 = time.perf_counter()

    best_params = run_hpo(run_dir, paper_artifacts, args, seed)

    _t = time.perf_counter() - _t0
    profile["hpo"] = {
        "wall_s": round(_t, 2),
        "delta_ram_mb": round(_mem_mb() - _mem_before, 1),
        "peak_gpu_mb": round(_gpu_mem_mb(), 1),
    }
    print(f"\nHPO wall-clock: {_t:.1f}s\n")

    # -- 2. Training --
    print("=" * 60)
    print(f"Training {arch.upper()} with best hyperparameters")
    print("=" * 60)
    _reset_gpu_stats()
    _mem_before = _mem_mb()
    _t0 = time.perf_counter()

    result_final, config_final = train_model(run_dir, best_params, args, seed)

    _t = time.perf_counter() - _t0
    _n_epochs = config_final.epochs
    profile["gnn_training"] = {
        "wall_s": round(_t, 2),
        "wall_per_epoch_s": round(_t / _n_epochs, 3),
        "n_epochs": _n_epochs,
        "delta_ram_mb": round(_mem_mb() - _mem_before, 1),
        "peak_gpu_mb": round(_gpu_mem_mb(), 1),
    }
    print(f"\nTraining wall-clock: {_t:.1f}s  ({_t / _n_epochs:.2f}s/epoch)\n")

    # -- 3. Evaluation + comparison --
    print("=" * 60)
    print("Evaluation + Comparison Table")
    print("=" * 60)
    evaluate_model(run_dir, paper_artifacts, result_final, arch)

    # -- 4. Inference --
    if args.inference_smiles:
        print("\n" + "=" * 60)
        print("Inference")
        print("=" * 60)
        run_inference(run_dir, paper_artifacts, args.inference_smiles, arch)

    # -- 5. Embeddings + PCA --
    print("\n" + "=" * 60)
    print("Embedding Extraction + PCA")
    print("=" * 60)
    _reset_gpu_stats()
    _mem_before = _mem_mb()
    _t0 = time.perf_counter()

    run_embedding_pca(run_dir, paper_artifacts, arch, seed)

    _t = time.perf_counter() - _t0
    profile["embedding_extraction"] = {
        "wall_s": round(_t, 2),
        "delta_ram_mb": round(_mem_mb() - _mem_before, 1),
        "peak_gpu_mb": round(_gpu_mem_mb(), 1),
    }
    print(f"\nEmbedding extraction wall-clock: {_t:.1f}s\n")

    # -- 6. Profile summary --
    print("=" * 60)
    print("Runtime / Memory Profile")
    print("=" * 60)
    print_and_save_profile(profile, paper_artifacts)

    # -- 7. Manifest --
    write_manifest(paper_artifacts, uid, arch)

    print(f"\nPipeline complete for {uid} / {arch}. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
