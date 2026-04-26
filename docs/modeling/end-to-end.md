# End-to-End GNN Training Pipeline

DTA-GNN provides a single-call pipeline that takes a UniProt accession and produces a trained, evaluated GNN — handling everything in between automatically.

## Overview

Four sequential steps are executed and individually timed:

| Step | What happens |
|------|-------------|
| **uniprot_mapping** | Resolves UniProt accession(s) to ChEMBL target IDs via SQLite or web API |
| **dataset_build** | Fetches activities, cleans data, applies scaffold split, saves `dataset.csv` and `compounds.csv` |
| **hyperparameter_search** | Runs a W&B Bayesian sweep over `n_trials` trials; selects params by best validation R² |
| **final_training** | Trains the GNN with the best params, logs to W&B, evaluates on the held-out test set |

All artifacts (dataset, model weights, metrics, W&B run IDs) are saved in a timestamped run directory under `runs/`.

---

## Quick Start

=== "Python"

    ```python
    from dta_gnn.training import run_gnn_end_to_end, EndToEndConfig

    result = run_gnn_end_to_end(EndToEndConfig(
        uniprot_ids="P00533",        # EGFR
        architecture="gin",
        sqlite_path="chembl_36.db",  # omit to use the web API
        wandb_project="my_project",
        n_trials=20,
        epochs=30,
    ))

    print(result.test_metrics)   # {"rmse": ..., "r2": ..., "mae": ...}
    print(result.timings)        # {"uniprot_mapping": 1.3, "dataset_build": 182.4, ...}
    print(result.run_dir)        # runs/20260309_142301
    ```

=== "CLI"

    ```bash
    # Minimal — uses web API
    dta_gnn train-gnn P00533

    # With local SQLite DB (much faster for large datasets)
    dta_gnn train-gnn P00533 \
        --architecture gin \
        --sqlite-path ./chembl_dbs/chembl_36.db \
        --wandb-project my_project \
        --n-trials 20 --epochs 30
    ```

---

## EndToEndConfig Reference

```python
from dta_gnn.training import EndToEndConfig

config = EndToEndConfig(
    uniprot_ids="P00533",   # required
    # ... all other fields have defaults
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uniprot_ids` | `str` | _(required)_ | Comma/space/semicolon-separated UniProt accession(s) |
| `architecture` | `str` | `"gin"` | GNN architecture: `gin\|gcn\|gat\|sage\|pna\|transformer\|tag\|arma\|cheb\|supergat` |
| `sqlite_path` | `str \| None` | `None` | Path to ChEMBL SQLite DB; `None` falls back to web API |
| `standard_types` | `list[str] \| None` | `None` | Activity types to include (e.g. `["IC50","Ki"]`); `None` = all |
| `test_size` | `float` | `0.2` | Fraction of data held out for testing |
| `val_size` | `float` | `0.1` | Fraction of data used for validation during HPO |
| `wandb_project` | `str` | `"dta_gnn"` | W&B project for HPO sweep and final run |
| `wandb_entity` | `str \| None` | `None` | W&B entity / team name |
| `wandb_api_key` | `str \| None` | `None` | W&B API key; falls back to `WANDB_API_KEY` env variable |
| `n_trials` | `int` | `20` | Number of Bayesian HPO sweep trials |
| `lr_min` / `lr_max` | `float` | `1e-5` / `1e-2` | Learning rate search bounds |
| `embedding_dim_min` / `embedding_dim_max` | `int` | `32` / `256` | Embedding dimension search bounds |
| `hidden_dim_min` / `hidden_dim_max` | `int` | `32` / `256` | Hidden dimension search bounds |
| `num_layers_min` / `num_layers_max` | `int` | `1` / `5` | Message-passing layer count search bounds |
| `dropout_min` / `dropout_max` | `float` | `0.0` / `0.5` | Dropout search bounds |
| `epochs` | `int` | `30` | Epochs for the final training run |
| `batch_size` | `int` | `64` | Mini-batch size |
| `runs_root` | `str` | `"runs"` | Root directory for timestamped run folders |
| `device` | `str \| None` | `None` | `"mps"`, `"cuda"`, `"cpu"`, or `None` (auto-detect) |

---

## EndToEndResult Reference

| Field | Type | Description |
|-------|------|-------------|
| `run_dir` | `Path` | Timestamped run directory, e.g. `runs/20260309_142301` |
| `uniprot_ids` | `list[str]` | Parsed and validated UniProt accessions |
| `target_chembl_ids` | `list[str]` | Resolved ChEMBL target IDs |
| `architecture` | `str` | GNN architecture that was trained |
| `dataset_size` | `int` | Total number of rows in the dataset |
| `train_size` | `int` | Training set size |
| `val_size_actual` | `int` | Validation set size |
| `test_size_actual` | `int` | Test set size |
| `hyperopt_result` | `HyperoptResult` | Best hyperparameters and best validation R² |
| `train_result` | `GnnTrainResult` | Final training artifacts (model path, metrics) |
| `test_metrics` | `dict` | `{"rmse": float, "mae": float, "r2": float}` |
| `timings` | `dict` | Per-step wall-clock seconds, e.g. `{"dataset_build": 182.4, ...}` |

---

## Pipeline Steps

### 1. UniProt Mapping

```
UniProt accession(s) → ChEMBL target ID(s)
```

- Accessions are parsed and validated against the UniProt format regex.
- If `sqlite_path` is provided, mapping uses a SQL join on `component_sequences → target_components → target_dictionary`.
- Otherwise, the ChEMBL web API is queried per accession.
- Raises `ValueError` if no ChEMBL targets are found.

### 2. Dataset Build

```
ChEMBL target IDs → filtered activities → pChEMBL values → scaffold split → dataset.csv
```

- Fetches all activities for the resolved targets (filtered by `standard_types` if set).
- Standardises units to pChEMBL, removes censored values, and deduplicates by median.
- Applies a Murcko scaffold cold-drug split (train / val / test).
- Saves `dataset.csv` and `compounds.csv` to the run directory.
- Raises `ValueError` if the resulting dataset is empty.

### 3. Hyperparameter Search

```
dataset.csv + val split → W&B Bayesian sweep (n_trials) → best params
```

- Launches a W&B sweep in `wandb_project` using Bayesian optimisation.
- Each trial trains for up to `epochs` (with early stopping internally).
- Objective: maximise validation R².
- The five tuned parameters are: `lr`, `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`.

### 4. Final Training

```
best params → train on the train split → evaluate on val and test → model saved
```

- Reconstructs a `GnnTrainConfig` from the best hyperparameters.
- Trains on the `train` split for `epochs` epochs (the `val` split is used
  for best-checkpoint selection during training).
- After training, the best-checkpoint model is reloaded and evaluated on the
  held-out `test` split, recording RMSE, MAE, R², Pearson r, and Spearman r.
- Saves model weights, encoder weights, and an encoder config JSON to the
  run directory (`model_gnn_<arch>.pt`, `encoder_<arch>.pt`,
  `encoder_<arch>_config.json`).
- Logs the final run to W&B (same project as the HPO sweep).

---

## Run Directory Structure

After a successful run, the run directory contains:

```
runs/20260309_142301/
├── dataset.csv          # Full dataset with split column
├── compounds.csv        # molecule_chembl_id + smiles
├── metadata.json        # Pipeline metadata (targets, split, etc.)
├── model_gnn_gin.pt     # Trained GNN weights
├── encoder_gin.pt       # GNN encoder weights (for embedding extraction)
├── model_metrics_gnn.json
└── model_predictions_gnn.csv
```

---

## W&B Integration

Both the HPO sweep and the final training run are logged to the same W&B project.

- **HPO sweep**: Each trial is a separate W&B run with the trial's hyperparameters and validation R².
- **Final run**: A single W&B run with the best params, full training curves, and test metrics.

```python
# Access the W&B project from Python
result = run_gnn_end_to_end(EndToEndConfig(
    uniprot_ids="P00533",
    wandb_project="egfr_study",
    wandb_entity="my_team",
    wandb_api_key="...",   # or set WANDB_API_KEY env variable
))
```

---

## Timing and Reproducibility

`result.timings` is a dict of wall-clock seconds per step:

```python
{
    "uniprot_mapping": 1.3,
    "dataset_build": 182.4,
    "hyperparameter_search": 201.7,
    "final_training": 18.2,
}
```

The CLI prints a formatted timing table at the end of every run:

```
Timings
  uniprot_mapping                     1.3s  (0%)
  dataset_build                     182.4s  (45%)
  hyperparameter_search             201.7s  (50%)
  final_training                     18.2s  (5%)
  ──────────────────────────────────────────
  Total                             403.6s  (6.7 min)
```

Each run gets a unique timestamped directory, so reruns never overwrite previous results.

---

## Common Recipes

### Use local SQLite (recommended for large HPO budgets)

```python
result = run_gnn_end_to_end(EndToEndConfig(
    uniprot_ids="P00533",
    sqlite_path="./chembl_dbs/chembl_36.db",
    n_trials=50,
    epochs=100,
))
```

### Filter by activity type

```python
result = run_gnn_end_to_end(EndToEndConfig(
    uniprot_ids="P00533",
    standard_types=["IC50", "Ki"],
))
```

### Multiple targets

```python
result = run_gnn_end_to_end(EndToEndConfig(
    uniprot_ids="P00533,P04637",   # EGFR + TP53
    architecture="gat",
))
```

### Use a different GNN architecture

```python
result = run_gnn_end_to_end(EndToEndConfig(
    uniprot_ids="P00533",
    architecture="sage",  # gin, gcn, gat, sage, pna, transformer, tag, arma, cheb, supergat
))
```

### Run without W&B (offline mode)

Set the `WANDB_MODE=offline` environment variable before running:

```bash
WANDB_MODE=offline dta_gnn train-gnn P00533
```

Or in Python:

```python
import os
os.environ["WANDB_MODE"] = "offline"
result = run_gnn_end_to_end(EndToEndConfig(uniprot_ids="P00533"))
```
