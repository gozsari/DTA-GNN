"""Hyperparameter optimization for model training."""

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

# Build Morgan fingerprints
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from dta_gnn.features import calculate_morgan_fingerprints


def _resolve_stratified_folds(y: np.ndarray, requested_folds: int) -> int:
    """Resolve a safe number of CV folds for StratifiedKFold.

    Stratified K-fold requires at least `n_splits` samples in every class.
    """
    if requested_folds < 2:
        raise ValueError("cv_folds must be >= 2")

    y = np.asarray(y)
    if y.size < 2:
        raise ValueError(
            "Not enough samples for cross-validation. "
            "Build a larger dataset (train+val must have at least 2 rows)."
        )

    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        raise ValueError(
            "Hyperparameter optimization needs at least two label classes. "
            "Your current run has only one class in train/val."
        )

    min_class = int(counts.min())
    folds = min(int(requested_folds), int(y.size), min_class)
    if folds < 2:
        raise ValueError(
            "Not enough samples per class for cross-validation. "
            f"Minimum class count is {min_class}; need at least 2 per class."
        )
    return folds




def _require_wandb():
    """Ensure Weights & Biases is installed."""
    try:
        import wandb

        return wandb
    except ImportError as e:
        raise ImportError(
            "wandb is not installed. Install with: `pip install 'dta_gnn[wandb]'`"
        ) from e


@dataclass
class HyperoptConfig:
    """Configuration for hyperparameter optimization."""

    model_type: Literal["RandomForest", "SVR", "GNN"]
    n_trials: int = 20
    n_jobs: int = 1
    sampler_seed: int = 42

    # RF parameters to optimize
    rf_optimize_n_estimators: bool = False
    rf_n_estimators_min: int = 50
    rf_n_estimators_max: int = 500

    rf_optimize_max_depth: bool = False
    rf_max_depth_min: int = 5
    rf_max_depth_max: int = 50

    rf_optimize_min_samples_split: bool = False
    rf_min_samples_split_min: int = 2
    rf_min_samples_split_max: int = 20

    # SVR parameters to optimize (regression only)
    svr_optimize_C: bool = False
    svr_C_min: float = 0.1
    svr_C_max: float = 100.0
    svr_C_default: float = 10.0

    svr_optimize_epsilon: bool = False
    svr_epsilon_min: float = 0.01
    svr_epsilon_max: float = 0.2
    svr_epsilon_default: float = 0.1

    svr_optimize_kernel: bool = False
    svr_kernel_choices: list[str] = None  # type: ignore[assignment]
    svr_kernel_default: str = "rbf"

    # GNN parameters to optimize
    architecture: Literal[
        "gin",
        "gcn",
        "gat",
        "sage",
        "pna",
        "transformer",
        "tag",
        "arma",
        "cheb",
        "supergat",
    ] = "gin"

    optimize_epochs: bool = False
    epochs_min: int = 5
    epochs_max: int = 50
    epochs_default: int = 20

    optimize_lr: bool = False
    lr_min: float = 0.00001
    lr_max: float = 0.01

    optimize_batch_size: bool = False
    batch_size_min: int = 16
    batch_size_max: int = 256
    batch_size_default: int = 64

    optimize_embedding_dim: bool = False
    embedding_dim_min: int = 32
    embedding_dim_max: int = 512
    embedding_dim_default: int = 128

    optimize_hidden_dim: bool = False
    hidden_dim_min: int = 32
    hidden_dim_max: int = 512
    hidden_dim_default: int = 128

    optimize_num_layers: bool = False
    num_layers_min: int = 1
    num_layers_max: int = 5
    num_layers_default: int = 3

    # Extra GNN architecture knobs (GIN-only today)
    optimize_dropout: bool = False
    dropout_min: float = 0.0
    dropout_max: float = 0.6
    dropout_default: float = 0.1

    optimize_pooling: bool = False
    pooling_choices: list[str] = None  # type: ignore[assignment]
    pooling_default: str = "add"

    optimize_residual: bool = False
    residual_default: bool = False

    optimize_head_mlp_layers: bool = False
    head_mlp_layers_min: int = 1
    head_mlp_layers_max: int = 4
    head_mlp_layers_default: int = 2

    optimize_gin_conv_mlp_layers: bool = False
    gin_conv_mlp_layers_min: int = 1
    gin_conv_mlp_layers_max: int = 4
    gin_conv_mlp_layers_default: int = 2

    # GIN-specific epsilon parameters
    optimize_gin_train_eps: bool = False
    gin_train_eps_default: bool = False
    optimize_gin_eps: bool = False
    gin_eps_min: float = 0.0
    gin_eps_max: float = 1.0
    gin_eps_default: float = 0.0

    # GAT-specific
    optimize_gat_heads: bool = False
    gat_heads_min: int = 1
    gat_heads_max: int = 8
    gat_heads_default: int = 4

    # GraphSAGE-specific
    optimize_sage_aggr: bool = False
    sage_aggr_choices: list[str] = None  # type: ignore[assignment]
    sage_aggr_default: str = "mean"

    # Transformer-specific
    optimize_transformer_heads: bool = False
    transformer_heads_min: int = 1
    transformer_heads_max: int = 8
    transformer_heads_default: int = 4

    # TAG-specific
    optimize_tag_k: bool = False
    tag_k_min: int = 1
    tag_k_max: int = 5
    tag_k_default: int = 2  # Reduced from 3 for better performance (K=2 is often sufficient)

    # ARMA-specific
    optimize_arma_stacks: bool = False
    arma_num_stacks_min: int = 1
    arma_num_stacks_max: int = 3
    arma_num_stacks_default: int = 1
    optimize_arma_layers: bool = False
    arma_num_layers_min: int = 1
    arma_num_layers_max: int = 3
    arma_num_layers_default: int = 1

    # Cheb-specific
    optimize_cheb_k: bool = False
    cheb_k_min: int = 1
    cheb_k_max: int = 5
    cheb_k_default: int = 2

    # SuperGAT-specific
    optimize_supergat_heads: bool = False
    supergat_heads_min: int = 1
    supergat_heads_max: int = 8
    supergat_heads_default: int = 4
    optimize_supergat_attention_type: bool = False
    supergat_attention_type_choices: list[str] = None  # type: ignore[assignment]
    supergat_attention_type_default: str = "MX"  # "MX" (Mixed) or "SD" (Self-Distillation)

    # Device for GNN training
    device: str | None = None  # "auto", "mps", "cuda", "cpu", or None (auto)


@dataclass
class HyperoptResult:
    """Result from hyperparameter optimization."""

    run_dir: Path
    best_params: dict
    best_value: float
    best_trial_number: int
    n_trials: int
    study_path: str
    best_params_path: str
    strategy: Literal["holdout-val", "cv"]
    cv_folds_used: Optional[int]


def _score_from_gnn_metrics(task_type: str, metrics: dict) -> float:
    """Single scalar score to maximize for regression."""
    val = (metrics or {}).get("splits", {}).get("val", {}) or {}
    # Regression: prefer r2, otherwise maximize -rmse.
    r2 = val.get("r2")
    if r2 is not None:
        try:
            return float(r2)
        except Exception:
            pass
    rmse = val.get("rmse")
    return -float(rmse) if rmse is not None else 0.0


def _infer_task_type_from_metadata_or_labels(
    df: pd.DataFrame,
    meta: dict | None,
) -> Literal["regression"]:
    return "regression"


def _bitstrings_to_numpy(bitstrings: list[str], *, n_bits: int) -> np.ndarray:
    out = np.empty((len(bitstrings), n_bits), dtype=np.uint8)
    for i, s in enumerate(bitstrings):
        b = np.frombuffer(str(s).encode("ascii"), dtype=np.uint8)
        if b.size != n_bits:
            raise ValueError(
                f"Fingerprint length mismatch: expected {n_bits}, got {b.size}."
            )
        out[i, :] = b - 48
    return out


def optimize_random_forest_wandb(
    run_dir: Path,
    *,
    config: HyperoptConfig,
    project: str,
    entity: str | None = None,
    api_key: str | None = None,
    sweep_name: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> HyperoptResult:
    """Optimize RandomForest hyperparameters using a W&B Bayes sweep.

    Uses:
    - Holdout validation if a val split exists.
    - Otherwise CV (KFold for regression).
    """

    wandb = _require_wandb()

    run_dir = Path(run_dir).resolve()
    dataset_path = run_dir / "dataset.csv"
    compounds_path = run_dir / "compounds.csv"
    metadata_path = run_dir / "metadata.json"

    if not dataset_path.exists() or not compounds_path.exists():
        raise FileNotFoundError(f"Expected {dataset_path} and {compounds_path}")

    df = pd.read_csv(dataset_path)
    compounds = pd.read_csv(compounds_path)

    if "molecule_chembl_id" not in df.columns or "label" not in df.columns:
        raise ValueError("dataset.csv must contain 'molecule_chembl_id' and 'label'.")
    if (
        "molecule_chembl_id" not in compounds.columns
        or "smiles" not in compounds.columns
    ):
        raise ValueError(
            "compounds.csv must contain 'molecule_chembl_id' and 'smiles'."
        )

    meta = None
    try:
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
    except Exception:
        meta = None

    task_type = _infer_task_type_from_metadata_or_labels(df, meta)

    # Compute fingerprints once.


    df_comp = (
        compounds[["molecule_chembl_id", "smiles"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df_feat = calculate_morgan_fingerprints(
        df_comp,
        smiles_col="smiles",
        radius=int(radius),
        n_bits=int(n_bits),
        out_col="morgan_fingerprint",
        drop_failures=True,
    )
    df_feat = (
        df_feat[["molecule_chembl_id", "morgan_fingerprint"]].dropna().drop_duplicates()
    )
    fp_map = dict(
        zip(
            df_feat["molecule_chembl_id"].astype(str),
            df_feat["morgan_fingerprint"].astype(str),
        )
    )

    df2 = df.copy()
    df2["_fp"] = df2["molecule_chembl_id"].astype(str).map(fp_map)
    df2 = df2.dropna(subset=["_fp"]).reset_index(drop=True)
    if df2.empty:
        raise ValueError("No rows left after joining fingerprints to dataset.")

    X = _bitstrings_to_numpy(df2["_fp"].astype(str).tolist(), n_bits=int(n_bits))
    y = df2["label"].to_numpy()

    # Exclude test split for tuning when present.
    if "split" in df2.columns:
        base_mask = df2["split"].astype(str).ne("test").to_numpy()
        has_val = bool((df2.loc[base_mask, "split"].astype(str) == "val").any())
    else:
        base_mask = np.ones(len(df2), dtype=bool)
        has_val = False

    project = (project or "").strip() or "dta_gnn"
    entity = (entity or "").strip() or None
    if api_key and str(api_key).strip():
        wandb.login(key=str(api_key).strip(), relogin=True)

    # Build sweep parameter space only from enabled knobs.
    parameters: dict[str, dict] = {}
    if config.rf_optimize_n_estimators:
        parameters["n_estimators"] = {
            "distribution": "int_uniform",
            "min": int(config.rf_n_estimators_min),
            "max": int(config.rf_n_estimators_max),
        }
    if config.rf_optimize_max_depth:
        parameters["max_depth"] = {
            "distribution": "int_uniform",
            "min": int(config.rf_max_depth_min),
            "max": int(config.rf_max_depth_max),
        }
    if config.rf_optimize_min_samples_split:
        parameters["min_samples_split"] = {
            "distribution": "int_uniform",
            "min": int(config.rf_min_samples_split_min),
            "max": int(config.rf_min_samples_split_max),
        }

    if not parameters:
        raise ValueError(
            "No parameters selected for optimization. "
            "Enable at least one 'Optimize ...' checkbox before running a sweep."
        )

    sweep_config: dict[str, object] = {
        "name": sweep_name or f"dta_gnn_rf_{task_type}",
        "method": "bayes",
        "metric": {"name": "val_score", "goal": "maximize"},
        "parameters": parameters,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)

    best_score = -math.inf
    best_params: dict[str, object] = {}
    best_trial_number = -1
    trial_counter = {"i": 0}

    def _trial_fn():
        nonlocal best_score, best_params, best_trial_number

        run = wandb.init(project=project, entity=entity, config={})
        trial_idx = int(trial_counter["i"])
        trial_counter["i"] = trial_idx + 1

        sampled = dict(getattr(wandb, "config", {}) or {})

        n_estimators = int(sampled.get("n_estimators", 500))
        max_depth = sampled.get("max_depth", None)
        max_depth = int(max_depth) if max_depth is not None else None
        min_samples_split = int(sampled.get("min_samples_split", 2))

        score: float
        extra_logs: dict[str, object] = {}
           
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        if has_val and "split" in df2.columns:
            train_mask = base_mask & (
                df2["split"].astype(str).eq("train").to_numpy()
            )
            val_mask = base_mask & (df2["split"].astype(str).eq("val").to_numpy())
            if int(train_mask.sum()) < 2 or int(val_mask.sum()) < 1:
                raise ValueError(
                    "Validation split exists but is too small for RF sweep."
                )

            model.fit(X[train_mask], y[train_mask].astype(float))
            y_true = y[val_mask].astype(float)
            y_pred = model.predict(X[val_mask]).astype(float)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            extra_logs["val/rmse"] = rmse
            if y_true.size >= 2:
                r2 = float(r2_score(y_true, y_pred))
                extra_logs["val/r2"] = r2
                score = r2
            else:
                score = -rmse
        else:
            y_base = y[base_mask].astype(float)
            X_base = X[base_mask]
            # Use simple 5-fold CV (non-stratified).
            n_splits = min(5, int(len(y_base)))
            n_splits = max(n_splits, 2)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_base, y_base, cv=cv, scoring="r2")
            score = float(np.mean(scores)) if len(scores) else 0.0
            extra_logs["cv/r2"] = score

        wandb.log({"val_score": float(score), **extra_logs})
        run.summary["val_score"] = float(score)
        run.summary["task_type"] = task_type
        run.finish()

        if float(score) > float(best_score):
            best_score = float(score)
            best_trial_number = int(trial_idx)
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "radius": int(radius),
                "n_bits": int(n_bits),
                "task_type": task_type,
            }

    wandb.agent(sweep_id, function=_trial_fn, count=int(config.n_trials))

    best_params_path = str(run_dir / f"hyperopt_best_params_wandb_rf_{task_type}.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    return HyperoptResult(
        run_dir=run_dir,
        best_params=best_params,
        best_value=float(best_score) if best_score != -math.inf else 0.0,
        best_trial_number=int(best_trial_number),
        n_trials=int(config.n_trials),
        study_path=str(sweep_id),
        best_params_path=str(best_params_path),
        strategy="holdout-val" if has_val else "cv",
        cv_folds_used=None,
    )


def optimize_svr_wandb(
    run_dir: Path,
    *,
    config: HyperoptConfig,
    project: str,
    entity: str | None = None,
    api_key: str | None = None,
    sweep_name: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> HyperoptResult:
    """Optimize SVR hyperparameters using a W&B Bayes sweep.

    Notes:
    - SVR sweep is intended for regression runs (DTA).
    - Uses holdout validation if a val split exists; otherwise uses KFold CV.
    - Logs a single scalar metric `val_score` (maximize).
    """

    wandb = _require_wandb()

    run_dir = Path(run_dir).resolve()
    dataset_path = run_dir / "dataset.csv"
    compounds_path = run_dir / "compounds.csv"
    metadata_path = run_dir / "metadata.json"

    if not dataset_path.exists() or not compounds_path.exists():
        raise FileNotFoundError(f"Expected {dataset_path} and {compounds_path}")

    df = pd.read_csv(dataset_path)
    compounds = pd.read_csv(compounds_path)

    if "molecule_chembl_id" not in df.columns or "label" not in df.columns:
        raise ValueError("dataset.csv must contain 'molecule_chembl_id' and 'label'.")
    if (
        "molecule_chembl_id" not in compounds.columns
        or "smiles" not in compounds.columns
    ):
        raise ValueError(
            "compounds.csv must contain 'molecule_chembl_id' and 'smiles'."
        )

    meta = None
    try:
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
    except Exception:
        meta = None

    task_type = _infer_task_type_from_metadata_or_labels(df, meta)
    if task_type != "regression":
        raise ValueError(
            "SVR sweeps are supported for regression runs only. "
            "Build a regression (DTA) dataset or choose a different model for HPO."
        )

    # Compute fingerprints once.
   
    df_comp = (
        compounds[["molecule_chembl_id", "smiles"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df_feat = calculate_morgan_fingerprints(
        df_comp,
        smiles_col="smiles",
        radius=int(radius),
        n_bits=int(n_bits),
        out_col="morgan_fingerprint",
        drop_failures=True,
    )
    df_feat = (
        df_feat[["molecule_chembl_id", "morgan_fingerprint"]].dropna().drop_duplicates()
    )
    fp_map = dict(
        zip(
            df_feat["molecule_chembl_id"].astype(str),
            df_feat["morgan_fingerprint"].astype(str),
        )
    )

    df2 = df.copy()
    df2["_fp"] = df2["molecule_chembl_id"].astype(str).map(fp_map)
    df2 = df2.dropna(subset=["_fp"]).reset_index(drop=True)
    if df2.empty:
        raise ValueError("No rows left after joining fingerprints to dataset.")

    X = _bitstrings_to_numpy(df2["_fp"].astype(str).tolist(), n_bits=int(n_bits))
    y = df2["label"].astype(float).to_numpy()

    # Exclude test split for tuning when present.
    if "split" in df2.columns:
        base_mask = df2["split"].astype(str).ne("test").to_numpy()
        has_val = bool((df2.loc[base_mask, "split"].astype(str) == "val").any())
    else:
        base_mask = np.ones(len(df2), dtype=bool)
        has_val = False

    project = (project or "").strip() or "dta_gnn"
    entity = (entity or "").strip() or None
    if api_key and str(api_key).strip():
        wandb.login(key=str(api_key).strip(), relogin=True)

    # Build sweep parameter space only from enabled knobs.
    parameters: dict[str, dict] = {}

    if config.svr_optimize_C:
        parameters["C"] = {
            "distribution": "log_uniform_values",
            "min": float(config.svr_C_min),
            "max": float(config.svr_C_max),
        }
    if config.svr_optimize_epsilon:
        parameters["epsilon"] = {
            "distribution": "log_uniform_values",
            "min": float(config.svr_epsilon_min),
            "max": float(config.svr_epsilon_max),
        }

    kernel_choices = getattr(config, "svr_kernel_choices", None) or ["rbf", "linear"]
    kernel_choices = [
        str(k).strip().lower()
        for k in kernel_choices
        if str(k).strip().lower() in {"rbf", "linear"}
    ]
    if not kernel_choices:
        kernel_choices = ["rbf", "linear"]
    if config.svr_optimize_kernel:
        parameters["kernel"] = {"values": kernel_choices}

    if not parameters:
        raise ValueError(
            "No parameters selected for optimization. "
            "Enable at least one 'Optimize ...' checkbox before running a sweep."
        )

    sweep_config: dict[str, object] = {
        "name": sweep_name or "dta_gnn_svr",
        "method": "bayes",
        "metric": {"name": "val_score", "goal": "maximize"},
        "parameters": parameters,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)

    best_score = -math.inf
    best_params: dict[str, object] = {}
    best_trial_number = -1
    trial_counter = {"i": 0}

    def _trial_fn():
        nonlocal best_score, best_params, best_trial_number

        run = wandb.init(project=project, entity=entity, config={})
        trial_idx = int(trial_counter["i"])
        trial_counter["i"] = trial_idx + 1

        sampled = dict(getattr(wandb, "config", {}) or {})



        C = float(sampled.get("C", float(getattr(config, "svr_C_default", 10.0))))
        epsilon = float(
            sampled.get("epsilon", float(getattr(config, "svr_epsilon_default", 0.1)))
        )

        k = (
            str(
                sampled.get("kernel", str(getattr(config, "svr_kernel_default", "rbf")))
            )
            .strip()
            .lower()
        )
        if k not in {"rbf", "linear"}:
            k = "rbf"

        model = SVR(kernel=k, C=C, epsilon=epsilon)

        score: float
        extra_logs: dict[str, object] = {}

        if has_val and "split" in df2.columns:
            train_mask = base_mask & (df2["split"].astype(str).eq("train").to_numpy())
            val_mask = base_mask & (df2["split"].astype(str).eq("val").to_numpy())
            if int(train_mask.sum()) < 2 or int(val_mask.sum()) < 1:
                raise ValueError(
                    "Validation split exists but is too small for SVR sweep."
                )

            model.fit(X[train_mask], y[train_mask])
            y_true = y[val_mask]
            y_pred = model.predict(X[val_mask]).astype(float)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            extra_logs["val/rmse"] = rmse
            if y_true.size >= 2:
                r2 = float(r2_score(y_true, y_pred))
                extra_logs["val/r2"] = r2
                score = r2
            else:
                score = -rmse
        else:
            y_base = y[base_mask]
            X_base = X[base_mask]
            n_splits = min(5, int(len(y_base)))
            n_splits = max(n_splits, 2)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_base, y_base, cv=cv, scoring="r2")
            score = float(np.mean(scores)) if len(scores) else 0.0
            extra_logs["cv/r2"] = score
            extra_logs["cv/folds"] = int(n_splits)

        wandb.log({"val_score": float(score), **extra_logs})
        run.summary["val_score"] = float(score)
        run.summary["task_type"] = task_type
        run.finish()

        if float(score) > float(best_score):
            best_score = float(score)
            best_trial_number = int(trial_idx)
            best_params = {
                "C": float(C),
                "epsilon": float(epsilon),
                "kernel": k,
                "radius": int(radius),
                "n_bits": int(n_bits),
                "task_type": task_type,
            }

    wandb.agent(sweep_id, function=_trial_fn, count=int(config.n_trials))

    best_params_path = str(run_dir / "hyperopt_best_params_wandb_svr.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    return HyperoptResult(
        run_dir=run_dir,
        best_params=best_params,
        best_value=float(best_score) if best_score != -math.inf else 0.0,
        best_trial_number=int(best_trial_number),
        n_trials=int(config.n_trials),
        study_path=str(sweep_id),
        best_params_path=str(best_params_path),
        strategy="holdout-val" if has_val else "cv",
        cv_folds_used=None,
    )


def optimize_gnn_wandb(
    run_dir: Path,
    *,
    config: HyperoptConfig,
    project: str,
    entity: str | None = None,
    api_key: str | None = None,
    sweep_name: str | None = None,
) -> HyperoptResult:
    """Optimize GNN hyperparameters using a W&B Bayes sweep.

    Notes:
    - Uses the existing train/val split (requires a non-empty val split).
    - Runs each trial in an isolated subdirectory so artifacts don't overwrite.
    - Logs a single scalar metric `val_score` (maximize).
    """

    wandb = _require_wandb()

    run_dir = Path(run_dir).resolve()
    dataset_path = run_dir / "dataset.csv"
    compounds_path = run_dir / "compounds.csv"
    metadata_path = run_dir / "metadata.json"

    if not dataset_path.exists() or not compounds_path.exists():
        raise FileNotFoundError(f"Expected {dataset_path} and {compounds_path}")

    df = pd.read_csv(dataset_path)
    if "split" not in df.columns or not bool((df["split"] == "val").any()):
        raise ValueError(
            "W&B sweeps currently require an explicit validation split. "
            "Rebuild the dataset with val_size > 0.0."
        )

    arch = str(getattr(config, "architecture", "gin") or "gin").strip().lower()
    if arch not in {"gin", "gcn", "gat", "sage", "pna"}:
        arch = "gin"

    project = (project or "").strip() or "dta_gnn"
    entity = (entity or "").strip() or None
    if api_key and str(api_key).strip():
        wandb.login(key=str(api_key).strip(), relogin=True)

    # Build sweep parameter space only from enabled knobs.
    parameters: dict[str, dict] = {}

    def _int_values(min_v: int, max_v: int, step: int = 1) -> list[int]:
        if step <= 1:
            return list(range(int(min_v), int(max_v) + 1))
        return list(range(int(min_v), int(max_v) + 1, int(step)))

    if config.optimize_epochs:
        parameters["epochs"] = {
            "distribution": "int_uniform",
            "min": int(config.epochs_min),
            "max": int(config.epochs_max),
        }
    if config.optimize_batch_size:
        parameters["batch_size"] = {
            "distribution": "int_uniform",
            "min": int(config.batch_size_min),
            "max": int(config.batch_size_max),
        }
    if config.optimize_lr:
        parameters["lr"] = {
            "distribution": "log_uniform_values",
            "min": float(config.lr_min),
            "max": float(config.lr_max),
        }
    if config.optimize_embedding_dim:
        parameters["embedding_dim"] = {
            "values": _int_values(
                int(config.embedding_dim_min),
                int(config.embedding_dim_max),
                step=16,
            )
        }
    if config.optimize_hidden_dim:
        parameters["hidden_dim"] = {
            "values": _int_values(
                int(config.hidden_dim_min), int(config.hidden_dim_max), step=16
            )
        }

    if config.optimize_num_layers:
        parameters["num_layers"] = {
            "distribution": "int_uniform",
            "min": int(config.num_layers_min),
            "max": int(config.num_layers_max),
        }

    if config.optimize_dropout:
        parameters["dropout"] = {
            "distribution": "uniform",
            "min": float(config.dropout_min),
            "max": float(config.dropout_max),
        }

    pooling_choices = getattr(config, "pooling_choices", None) or [
        "add",
        "mean",
        "max",
        "attention",
    ]
    pooling_choices = [
        str(x) for x in pooling_choices if str(x) in {"add", "mean", "max", "attention"}
    ]
    if not pooling_choices:
        pooling_choices = ["add", "mean", "max", "attention"]
    if config.optimize_pooling:
        parameters["pooling"] = {"values": pooling_choices}

    if config.optimize_residual:
        parameters["residual"] = {"values": [False, True]}

    if config.optimize_head_mlp_layers:
        parameters["head_mlp_layers"] = {
            "distribution": "int_uniform",
            "min": int(getattr(config, "head_mlp_layers_min", 1)),
            "max": int(getattr(config, "head_mlp_layers_max", 4)),
        }

    if arch == "gin":
        if config.optimize_gin_conv_mlp_layers:
            parameters["gin_conv_mlp_layers"] = {
                "distribution": "int_uniform",
                "min": int(getattr(config, "gin_conv_mlp_layers_min", 1)),
                "max": int(getattr(config, "gin_conv_mlp_layers_max", 4)),
            }
        if config.optimize_gin_train_eps:
            parameters["gin_train_eps"] = {"values": [False, True]}
        if config.optimize_gin_eps:
            parameters["gin_eps"] = {
                "distribution": "uniform",
                "min": float(getattr(config, "gin_eps_min", 0.0)),
                "max": float(getattr(config, "gin_eps_max", 1.0)),
            }

    if arch == "gat" and config.optimize_gat_heads:
        parameters["gat_heads"] = {
            "distribution": "int_uniform",
            "min": int(getattr(config, "gat_heads_min", 1)),
            "max": int(getattr(config, "gat_heads_max", 8)),
        }

    if arch == "sage":
        sage_aggr_choices = getattr(config, "sage_aggr_choices", None) or [
            "mean",
            "max",
            "lstm",
            "pool",
        ]
        sage_aggr_choices = [
            str(x) for x in sage_aggr_choices if str(x) in {"mean", "max", "lstm", "pool"}
        ]
        if not sage_aggr_choices:
            sage_aggr_choices = ["mean", "max", "lstm", "pool"]
        if config.optimize_sage_aggr:
            parameters["sage_aggr"] = {"values": sage_aggr_choices}

    if arch == "transformer" and config.optimize_transformer_heads:
        parameters["transformer_heads"] = {
            "distribution": "int_uniform",
            "min": int(getattr(config, "transformer_heads_min", 1)),
            "max": int(getattr(config, "transformer_heads_max", 8)),
        }

    if arch == "tag" and config.optimize_tag_k:
        parameters["tag_k"] = {
            "distribution": "int_uniform",
            "min": int(getattr(config, "tag_k_min", 1)),
            "max": int(getattr(config, "tag_k_max", 5)),
        }

    if arch == "arma":
        if config.optimize_arma_stacks:
            parameters["arma_num_stacks"] = {
                "distribution": "int_uniform",
                "min": int(getattr(config, "arma_num_stacks_min", 1)),
                "max": int(getattr(config, "arma_num_stacks_max", 3)),
            }
        if config.optimize_arma_layers:
            parameters["arma_num_layers"] = {
                "distribution": "int_uniform",
                "min": int(getattr(config, "arma_num_layers_min", 1)),
                "max": int(getattr(config, "arma_num_layers_max", 3)),
            }

    if arch == "cheb" and config.optimize_cheb_k:
        parameters["cheb_k"] = {
            "distribution": "int_uniform",
            "min": int(getattr(config, "cheb_k_min", 1)),
            "max": int(getattr(config, "cheb_k_max", 5)),
        }

    if arch == "supergat":
        if config.optimize_supergat_heads:
            parameters["supergat_heads"] = {
                "distribution": "int_uniform",
                "min": int(getattr(config, "supergat_heads_min", 1)),
                "max": int(getattr(config, "supergat_heads_max", 8)),
            }
        supergat_attention_choices = getattr(config, "supergat_attention_type_choices", None) or [
            "MX",
            "SD",
        ]
        supergat_attention_choices = [
            str(x) for x in supergat_attention_choices if str(x) in {"MX", "SD"}
        ]
        if not supergat_attention_choices:
            supergat_attention_choices = ["MX", "SD"]
        if config.optimize_supergat_attention_type:
            parameters["supergat_attention_type"] = {"values": supergat_attention_choices}

    if not parameters:
        raise ValueError(
            "No parameters selected for optimization. "
            "Enable at least one 'Optimize ...' checkbox before running a sweep."
        )

    sweep_config: dict[str, object] = {
        "name": sweep_name or f"dta_gnn_gnn_{arch}",
        "method": "bayes",
        "metric": {"name": "val_score", "goal": "maximize"},
        "parameters": parameters,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)

    best_score = -math.inf
    best_params: dict[str, object] = {}
    best_trial_number = -1
    trial_counter = {"i": 0}

    from dta_gnn.models.gnn import GnnTrainConfig, train_gnn_on_run

    def _trial_fn():
        nonlocal best_score, best_params, best_trial_number

        run = wandb.init(project=project, entity=entity, config={})
        trial_idx = int(trial_counter["i"])
        trial_counter["i"] = trial_idx + 1
        
        print(f"\n[Trial {trial_idx}] Starting GNN training...")
        print(f"[Trial {trial_idx}] Run ID: {run.id}")

        # Pull optimized params from wandb.config; fill the rest from defaults.
        sampled = dict(getattr(wandb, "config", {}) or {})
        
        # Log all hyperparameters to wandb
        wandb.config.update({
            "trial_number": trial_idx,
            "architecture": arch,
        })

        # Architecture-specific parameters
        gin_conv_mlp_layers = int(
            sampled.get(
                "gin_conv_mlp_layers",
                int(getattr(config, "gin_conv_mlp_layers_default", 2)),
            )
        )
        gin_train_eps = bool(
            sampled.get(
                "gin_train_eps",
                bool(getattr(config, "gin_train_eps_default", False)),
            )
        )
        gin_eps = float(
            sampled.get(
                "gin_eps",
                float(getattr(config, "gin_eps_default", 0.0)),
            )
        )
        gat_heads = int(
            sampled.get(
                "gat_heads",
                int(getattr(config, "gat_heads_default", 4)),
            )
        )
        sage_aggr = str(
            sampled.get(
                "sage_aggr",
                str(getattr(config, "sage_aggr_default", "mean")),
            )
        )
        transformer_heads = int(
            sampled.get(
                "transformer_heads",
                int(getattr(config, "transformer_heads_default", 4)),
            )
        )
        tag_k = int(
            sampled.get(
                "tag_k",
                int(getattr(config, "tag_k_default", 2)),
            )
        )
        arma_num_stacks = int(
            sampled.get(
                "arma_num_stacks",
                int(getattr(config, "arma_num_stacks_default", 1)),
            )
        )
        arma_num_layers = int(
            sampled.get(
                "arma_num_layers",
                int(getattr(config, "arma_num_layers_default", 1)),
            )
        )
        cheb_k = int(
            sampled.get(
                "cheb_k",
                int(getattr(config, "cheb_k_default", 2)),
            )
        )
        supergat_heads = int(
            sampled.get(
                "supergat_heads",
                int(getattr(config, "supergat_heads_default", 4)),
            )
        )
        supergat_attention_type = str(
            sampled.get(
                "supergat_attention_type",
                str(getattr(config, "supergat_attention_type_default", "MX")),
            )
        )

        cfg = GnnTrainConfig(
            architecture=arch,
            epochs=int(sampled.get("epochs", int(getattr(config, "epochs_default", 20)))),
            batch_size=int(sampled.get("batch_size", int(getattr(config, "batch_size_default", 64)))),
            lr=float(sampled.get("lr", 1e-3)),
            embedding_dim=int(sampled.get("embedding_dim", int(getattr(config, "embedding_dim_default", 128)))),
            device=getattr(config, "device", None),
            hidden_dim=int(sampled.get("hidden_dim", int(getattr(config, "hidden_dim_default", 128)))),
            dropout=float(
                sampled.get(
                    "dropout", float(getattr(config, "dropout_default", 0.1))
                )
            ),
            pooling=str(
                sampled.get(
                    "pooling", str(getattr(config, "pooling_default", "add"))
                )
            ),
            residual=bool(
                sampled.get(
                    "residual", bool(getattr(config, "residual_default", False))
                )
            ),
            head_mlp_layers=int(
                sampled.get(
                    "head_mlp_layers",
                    int(getattr(config, "head_mlp_layers_default", 2)),
                )
            ),
            gin_conv_mlp_layers=gin_conv_mlp_layers,
            gin_train_eps=gin_train_eps,
            gin_eps=gin_eps,
            gat_heads=gat_heads,
            sage_aggr=sage_aggr,
            transformer_heads=transformer_heads,
            tag_k=tag_k,
            arma_num_stacks=arma_num_stacks,
            arma_num_layers=arma_num_layers,
            cheb_k=cheb_k,
            supergat_heads=supergat_heads,
            supergat_attention_type=supergat_attention_type,
            num_layers=int(
                sampled.get(
                    "num_layers",
                    int(getattr(config, "num_layers_default", 3)),
                )
            ),
        )

        trial_dir = run_dir / f"_wandb_{arch}_{trial_idx:03d}_{run.id}"
        trial_dir.mkdir(exist_ok=True)
        shutil.copy2(dataset_path, trial_dir / "dataset.csv")
        shutil.copy2(compounds_path, trial_dir / "compounds.csv")
        if metadata_path.exists():
            shutil.copy2(metadata_path, trial_dir / "metadata.json")

        print(f"[Trial {trial_idx}] Training GNN with config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, lr={cfg.lr:.6f}, embedding_dim={cfg.embedding_dim}")
        res = train_gnn_on_run(trial_dir, config=cfg, wandb_run=run)
        score = _score_from_gnn_metrics(res.task_type, res.metrics)
        print(f"[Trial {trial_idx}] Training complete. Score: {score:.4f}")

        # Log all metrics from all splits
        all_metrics = {}
        splits_metrics = (res.metrics or {}).get("splits", {}) or {}
        
        # Log validation metrics
        val_metrics = splits_metrics.get("val", {}) or {}
        all_metrics.update({
            "val_score": float(score),
            "val/roc_auc": val_metrics.get("roc_auc"),
            "val/accuracy": val_metrics.get("accuracy"),
            "val/rmse": val_metrics.get("rmse"),
            "val/mae": val_metrics.get("mae"),
            "val/r2": val_metrics.get("r2"),
        })
        
        # Log training metrics
        train_metrics = splits_metrics.get("train", {}) or {}
        all_metrics.update({
            "train/roc_auc": train_metrics.get("roc_auc"),
            "train/accuracy": train_metrics.get("accuracy"),
            "train/rmse": train_metrics.get("rmse"),
            "train/mae": train_metrics.get("mae"),
            "train/r2": train_metrics.get("r2"),
        })
        
        # Log test metrics if available
        test_metrics = splits_metrics.get("test", {}) or {}
        if test_metrics:
            all_metrics.update({
                "test/roc_auc": test_metrics.get("roc_auc"),
                "test/accuracy": test_metrics.get("accuracy"),
                "test/rmse": test_metrics.get("rmse"),
                "test/mae": test_metrics.get("mae"),
                "test/r2": test_metrics.get("r2"),
            })
        
        # Remove None values before logging
        all_metrics = {k: v for k, v in all_metrics.items() if v is not None}
        wandb.log(all_metrics)

        run.summary["val_score"] = float(score)
        run.summary["run_dir"] = str(trial_dir)
        run.summary["architecture"] = arch
        run.summary["task_type"] = res.task_type
        run.summary["trial_number"] = trial_idx
        run.summary["is_best"] = float(score) > float(best_score)
        run.finish()

        if float(score) > float(best_score):
            best_score = float(score)
            best_trial_number = int(trial_idx)
            # Return the *sampled* params plus fixed architecture for reproducibility.
            best_params = {"architecture": arch, **{k: v for k, v in sampled.items()}}
            print(f"[Trial {trial_idx}] New best score: {best_score:.4f} (trial #{best_trial_number})")
        else:
            print(f"[Trial {trial_idx}] Score: {score:.4f} (best so far: {best_score:.4f})")

    print(f"\n{'='*60}")
    print(f"Starting W&B sweep with {config.n_trials} trials")
    print(f"Sweep ID: {sweep_id}")
    print(f"Architecture: {arch}")
    print(f"{'='*60}\n")
    
    wandb.agent(sweep_id, function=_trial_fn, count=int(config.n_trials))

    best_params_path = str(run_dir / f"hyperopt_best_params_wandb_{arch}.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep completed!")
    print(f"Best score: {best_score:.4f} (trial #{best_trial_number})")
    print(f"Best params saved to: {best_params_path}")
    print(f"{'='*60}\n")

    return HyperoptResult(
        run_dir=run_dir,
        best_params=best_params,
        best_value=float(best_score) if best_score != -math.inf else 0.0,
        best_trial_number=int(best_trial_number),
        n_trials=int(config.n_trials),
        study_path=str(sweep_id),
        best_params_path=str(best_params_path),
        strategy="holdout-val",
        cv_folds_used=None,
    )


# Backwards-compatible aliases: redirect to W&B Bayes versions
# All optimization now uses W&B Bayes sweeps instead of Optuna
optimize_random_forest = optimize_random_forest_wandb
optimize_gnn = optimize_gnn_wandb
optimize_gin = optimize_gnn_wandb  # Historical alias
