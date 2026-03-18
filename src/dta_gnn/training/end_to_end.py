"""End-to-end GNN training pipeline from a UniProt protein ID.

Orchestrates: UniProt→ChEMBL mapping → dataset build (scaffold split) →
W&B hyperparameter search → final model training → test evaluation.
Each step is wall-clock timed and the timings are returned in the result.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Literal

from loguru import logger

from dta_gnn import __version__
from dta_gnn.io.runs import create_run_dir
from dta_gnn.io.target_mapping import (
    map_uniprot_to_chembl_targets_sqlite,
    map_uniprot_to_chembl_targets_web,
    parse_uniprot_accessions,
)
from dta_gnn.models.gnn import GnnTrainConfig, GnnTrainResult, train_gnn_on_run
from dta_gnn.models.hyperopt import HyperoptConfig, HyperoptResult, optimize_gnn_wandb
from dta_gnn.pipeline import Pipeline


@dataclass
class EndToEndConfig:
    """Configuration for the end-to-end GNN training pipeline.

    Args:
        uniprot_ids: One or more UniProt accessions (comma/space/semicolon-separated).
            Example: ``"P00533"`` or ``"P00533, P04637"``.
        architecture: GNN architecture to train and tune.
        sqlite_path: Path to a local ChEMBL SQLite database. When provided,
            all ChEMBL data is fetched from this file. When ``None``, the
            ChEMBL web API is used as a fallback.
        standard_types: Activity standard types to include (e.g.
            ``["IC50", "Ki"]``). ``None`` keeps all types.
        test_size: Fraction of data reserved for the test split.
        val_size: Fraction of data reserved for the validation split.
        wandb_project: W&B project name used for both the HPO sweep and the
            final training run.
        wandb_entity: W&B entity (username or team). ``None`` uses the default.
        wandb_api_key: W&B API key. ``None`` relies on ``WANDB_API_KEY`` env
            variable or interactive login.
        n_trials: Number of W&B Bayes sweep trials for hyperparameter search.
        lr_min: Lower bound for learning-rate search (log-uniform).
        lr_max: Upper bound for learning-rate search (log-uniform).
        embedding_dim_min: Lower bound for embedding dimension search.
        embedding_dim_max: Upper bound for embedding dimension search.
        hidden_dim_min: Lower bound for hidden dimension search.
        hidden_dim_max: Upper bound for hidden dimension search.
        num_layers_min: Lower bound for number of GNN layers search.
        num_layers_max: Upper bound for number of GNN layers search.
        dropout_min: Lower bound for dropout rate search.
        dropout_max: Upper bound for dropout rate search.
        epochs: Number of training epochs for the *final* model (HPO trials use
            fewer epochs internally).
        batch_size: Mini-batch size for both HPO and final training.
        runs_root: Root directory under which timestamped run directories are
            created (default: ``"runs"``).
        device: PyTorch device string. ``None`` auto-detects (MPS > CUDA > CPU).
    """

    uniprot_ids: str
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
    sqlite_path: str | None = None
    standard_types: list[str] | None = None
    test_size: float = 0.2
    val_size: float = 0.1
    # W&B
    wandb_project: str = "dta_gnn"
    wandb_entity: str | None = None
    wandb_api_key: str | None = None
    # HPO search space
    n_trials: int = 20
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    embedding_dim_min: int = 32
    embedding_dim_max: int = 256
    hidden_dim_min: int = 32
    hidden_dim_max: int = 256
    num_layers_min: int = 1
    num_layers_max: int = 5
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    # Final training
    epochs: int = 30
    batch_size: int = 64
    # Infrastructure
    runs_root: str = "runs"
    device: str | None = None


@dataclass
class EndToEndResult:
    """Result of a complete end-to-end GNN training run.

    Attributes:
        run_dir: Path to the timestamped run directory holding all artifacts.
        uniprot_ids: Validated UniProt accessions used as input.
        target_chembl_ids: Resolved ChEMBL target IDs.
        architecture: GNN architecture that was trained.
        dataset_size: Total number of rows in the built dataset.
        train_size: Number of training rows.
        val_size_actual: Number of validation rows.
        test_size_actual: Number of test rows.
        hyperopt_result: Full result from ``optimize_gnn_wandb``.
        train_result: Full result from ``train_gnn_on_run``.
        test_metrics: Test-split metrics dict (``r2``, ``rmse``, ``mae``, …).
        timings: Wall-clock time in seconds for each pipeline step.
    """

    run_dir: Path
    uniprot_ids: list[str]
    target_chembl_ids: list[str]
    architecture: str
    dataset_size: int
    train_size: int
    val_size_actual: int
    test_size_actual: int
    hyperopt_result: HyperoptResult
    train_result: GnnTrainResult
    test_metrics: dict
    timings: dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextmanager
def _timed(name: str, timings: dict) -> Generator[None, None, None]:
    """Context manager that records wall-clock time for a named step."""
    t0 = time.perf_counter()
    logger.info("[{}] starting...", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        logger.info("[{}] done in {:.1f}s", name, elapsed)


def _best_params_to_gnn_config(best_params: dict, config: EndToEndConfig) -> GnnTrainConfig:
    """Build a GnnTrainConfig from hyperopt best_params + EndToEndConfig defaults."""
    return GnnTrainConfig(
        architecture=config.architecture,
        # Common searched params
        embedding_dim=int(best_params.get("embedding_dim", 128)),
        hidden_dim=int(best_params.get("hidden_dim", 128)),
        num_layers=int(best_params.get("num_layers", 3)),
        dropout=float(best_params.get("dropout", 0.1)),
        pooling=str(best_params.get("pooling", "add")),
        residual=bool(best_params.get("residual", False)),
        head_mlp_layers=int(best_params.get("head_mlp_layers", 2)),
        # Architecture-specific
        gin_conv_mlp_layers=int(best_params.get("gin_conv_mlp_layers", 2)),
        gin_train_eps=bool(best_params.get("gin_train_eps", False)),
        gin_eps=float(best_params.get("gin_eps", 0.0)),
        gat_heads=int(best_params.get("gat_heads", 4)),
        sage_aggr=str(best_params.get("sage_aggr", "mean")),
        transformer_heads=int(best_params.get("transformer_heads", 4)),
        tag_k=int(best_params.get("tag_k", 2)),
        arma_num_stacks=int(best_params.get("arma_num_stacks", 1)),
        arma_num_layers=int(best_params.get("arma_num_layers", 1)),
        cheb_k=int(best_params.get("cheb_k", 2)),
        supergat_heads=int(best_params.get("supergat_heads", 4)),
        supergat_attention_type=str(best_params.get("supergat_attention_type", "MX")),
        # Training knobs — final run uses full epochs; lr comes from HPO
        lr=float(best_params.get("lr", 1e-3)),
        batch_size=int(best_params.get("batch_size", config.batch_size)),
        epochs=int(config.epochs),
        device=config.device,
    )


def _require_wandb():
    try:
        import wandb
        return wandb
    except ImportError as e:
        raise ImportError(
            "wandb is required for the end-to-end pipeline. "
            "Install with: pip install wandb"
        ) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_gnn_end_to_end(config: EndToEndConfig) -> EndToEndResult:
    """Run the complete GNN training pipeline end-to-end.

    Steps
    -----
    1. Parse UniProt accessions and map them to ChEMBL target IDs.
    2. Build a DTA dataset from ChEMBL using a scaffold split; save all
       required files (``dataset.csv``, ``compounds.csv``, ``metadata.json``)
       to a new timestamped run directory.
    3. Run a W&B Bayes hyperparameter sweep (validation set used for scoring).
    4. Train the final model with the best hyperparameters and log the run to
       the same W&B project.
    5. Return an :class:`EndToEndResult` with test metrics and per-step timings.

    Args:
        config: Pipeline configuration.

    Returns:
        :class:`EndToEndResult` with all artifacts, metrics, and timing.

    Raises:
        ValueError: If no ChEMBL targets can be resolved or the dataset is empty.
    """
    timings: dict = {}
    arch = config.architecture

    # ------------------------------------------------------------------
    # Step 1: UniProt → ChEMBL mapping
    # ------------------------------------------------------------------
    with _timed("uniprot_mapping", timings):
        accessions = parse_uniprot_accessions(config.uniprot_ids)
        logger.info("Parsed {} UniProt accession(s): {}", len(accessions), accessions)

        if config.sqlite_path:
            logger.info("Using SQLite source: {}", config.sqlite_path)
            mapping = map_uniprot_to_chembl_targets_sqlite(config.sqlite_path, accessions)
        else:
            logger.info("Using ChEMBL web API for UniProt→ChEMBL mapping")
            mapping = map_uniprot_to_chembl_targets_web(accessions)

        if mapping.unmapped:
            logger.warning(
                "No ChEMBL targets found for {} accession(s): {}",
                len(mapping.unmapped),
                mapping.unmapped,
            )

        if not mapping.resolved_target_chembl_ids:
            raise ValueError(
                f"No ChEMBL target IDs could be resolved from UniProt "
                f"accession(s): {accessions}. "
                "Check that the accessions are valid and present in ChEMBL."
            )

        target_chembl_ids = mapping.resolved_target_chembl_ids
        logger.info(
            "Resolved {} ChEMBL target(s): {}",
            len(target_chembl_ids),
            target_chembl_ids,
        )

    # Create run directory (not timed — instantaneous)
    run_dir = create_run_dir(runs_root=config.runs_root)
    logger.info("Run directory: {}", run_dir)

    # ------------------------------------------------------------------
    # Step 2: Dataset building with scaffold split
    # ------------------------------------------------------------------
    with _timed("dataset_build", timings):
        source_type: str = "sqlite" if config.sqlite_path else "web"
        pipeline = Pipeline(
            source_type=source_type,
            sqlite_path=config.sqlite_path,
        )

        dataset_path = run_dir / "dataset.csv"
        dataset_df = pipeline.build_dta(
            target_ids=target_chembl_ids,
            standard_types=config.standard_types,
            split_method="scaffold",
            test_size=config.test_size,
            val_size=config.val_size,
            output_path=str(dataset_path),
        )

        if dataset_df is None or dataset_df.empty:
            raise ValueError(
                "Dataset is empty after building. "
                "Verify that the target IDs have associated activity data "
                "in ChEMBL and that standard_types (if set) match available data."
            )

        # Save compounds.csv (required by train_gnn_on_run and optimize_gnn_wandb)
        compounds_df = (
            dataset_df[["molecule_chembl_id", "smiles"]]
            .drop_duplicates()
            .dropna(subset=["smiles"])
            .reset_index(drop=True)
        )
        compounds_path = run_dir / "compounds.csv"
        try:
            compounds_df.to_csv(compounds_path, index=False)
        except OSError as e:
            logger.error("Failed to write compounds.csv to {}: {}", compounds_path, e)
            raise

        # Compute split counts
        split_counts: dict = {}
        if "split" in dataset_df.columns:
            split_counts = {
                str(k): int(v)
                for k, v in dataset_df["split"].value_counts().items()
            }

        # Save metadata.json
        metadata = {
            "uniprot_ids": accessions,
            "target_chembl_ids": target_chembl_ids,
            "architecture": arch,
            "source_type": source_type,
            "split_method": "scaffold",
            "test_size": config.test_size,
            "val_size": config.val_size,
            "dataset_size": len(dataset_df),
            "split_counts": split_counts,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dta_gnn_version": __version__,
        }
        metadata_path = run_dir / "metadata.json"
        try:
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except OSError as e:
            logger.error("Failed to write metadata.json to {}: {}", metadata_path, e)
            raise

        logger.info(
            "Dataset built: {} rows total, splits: {}",
            len(dataset_df),
            split_counts,
        )

    # ------------------------------------------------------------------
    # Step 3: Hyperparameter search via W&B Bayes sweep
    # ------------------------------------------------------------------
    with _timed("hyperparameter_search", timings):
        hpo_config = HyperoptConfig(
            model_type="GNN",
            architecture=arch,
            n_trials=config.n_trials,
            device=config.device,
            # Defaults: search lr, embedding_dim, hidden_dim, num_layers, dropout
            optimize_lr=True,
            lr_min=config.lr_min,
            lr_max=config.lr_max,
            optimize_embedding_dim=True,
            embedding_dim_min=config.embedding_dim_min,
            embedding_dim_max=config.embedding_dim_max,
            optimize_hidden_dim=True,
            hidden_dim_min=config.hidden_dim_min,
            hidden_dim_max=config.hidden_dim_max,
            optimize_num_layers=True,
            num_layers_min=config.num_layers_min,
            num_layers_max=config.num_layers_max,
            optimize_dropout=True,
            dropout_min=config.dropout_min,
            dropout_max=config.dropout_max,
        )

        hyperopt_result = optimize_gnn_wandb(
            run_dir,
            config=hpo_config,
            project=config.wandb_project,
            entity=config.wandb_entity,
            api_key=config.wandb_api_key,
            sweep_name=f"gnn_{arch}_hpo",
        )

        logger.info(
            "HPO complete — best val R²: {:.4f} (trial {}), best params: {}",
            hyperopt_result.best_value,
            hyperopt_result.best_trial_number,
            hyperopt_result.best_params,
        )

    # ------------------------------------------------------------------
    # Step 4: Final model training with best hyperparameters
    # ------------------------------------------------------------------
    with _timed("final_training", timings):
        final_config = _best_params_to_gnn_config(hyperopt_result.best_params, config)

        wandb = _require_wandb()
        wandb_run = None
        try:
            if config.wandb_api_key and str(config.wandb_api_key).strip():
                wandb.login(key=str(config.wandb_api_key).strip(), relogin=True)

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity or None,
                name=f"gnn_{arch}_final",
                config={
                    "uniprot_ids": accessions,
                    "target_chembl_ids": target_chembl_ids,
                    "architecture": arch,
                    "dataset_size": len(dataset_df),
                    "split_counts": split_counts,
                    **{f: getattr(final_config, f) for f in final_config.__dataclass_fields__},
                },
                tags=["final_training", arch],
            )
            logger.info("W&B final training run: {}", wandb_run.url if wandb_run else "n/a")

            train_result = train_gnn_on_run(
                run_dir,
                config=final_config,
                wandb_run=wandb_run,
            )
        finally:
            if wandb_run is not None:
                wandb_run.finish()

        logger.info(
            "Final training complete — best epoch: {}, model: {}",
            train_result.best_epoch,
            train_result.model_path,
        )

    # ------------------------------------------------------------------
    # Step 5: Extract test metrics and report timing
    # ------------------------------------------------------------------
    test_metrics: dict = {}
    if train_result.metrics and isinstance(train_result.metrics.get("splits"), dict):
        test_metrics = train_result.metrics["splits"].get("test") or {}

    total_time = sum(timings.values())
    logger.info("=" * 60)
    logger.info(
        "End-to-end pipeline completed in {:.1f}s ({:.1f} min)",
        total_time,
        total_time / 60.0,
    )
    for step_name, step_time in timings.items():
        pct = 100.0 * step_time / total_time if total_time > 0 else 0.0
        logger.info("  {:<30s}  {:6.1f}s  ({:.0f}%)", step_name, step_time, pct)
    logger.info("Test metrics: {}", test_metrics)
    logger.info("Run directory: {}", run_dir)
    logger.info("=" * 60)

    return EndToEndResult(
        run_dir=run_dir,
        uniprot_ids=accessions,
        target_chembl_ids=target_chembl_ids,
        architecture=arch,
        dataset_size=len(dataset_df),
        train_size=split_counts.get("train", 0),
        val_size_actual=split_counts.get("val", 0),
        test_size_actual=split_counts.get("test", 0),
        hyperopt_result=hyperopt_result,
        train_result=train_result,
        test_metrics=test_metrics,
        timings=timings,
    )


