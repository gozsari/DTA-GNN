# Quick Start

Get up and running with DTA-GNN in minutes. This guide walks you through
building your first target-specific binding affinity (DTA) dataset and
training a model on it.

## Prerequisites

Ensure DTA-GNN is installed:

```bash
pip install dta-gnn
```

(Or install from source — see [Installation](installation.md).)

## Option 0: One-call end-to-end (CLI)

If you just want a trained, evaluated GNN for a UniProt accession, the
fastest path is the `train-gnn` command:

```bash
# Optionally download a local ChEMBL SQLite database first (much faster):
dta_gnn setup --version 36 --dir ./chembl_dbs

# Run the full pipeline (UniProt → ChEMBL → scaffold split → W&B HPO → final GNN)
dta_gnn train-gnn P00533 \
  --architecture gin \
  --sqlite-path ./chembl_dbs/chembl_36.db \
  --n-trials 20 --epochs 30
```

The CLI prints a per-step timing summary and the test-set RMSE / MAE / R²
when the run completes. See [End-to-End Pipeline](../modeling/end-to-end.md).

## Option 1: Web Interface (Recommended for Beginners)

The easiest way to get started is with the interactive web UI:

```bash
dta_gnn ui
```

This launches a Gradio-based web application at `http://127.0.0.1:7860`.

### Building a Dataset via UI

1. **Select Data Source**: Choose "SQLite" and provide the path to your ChEMBL database, or use "Web API" (slower but no setup required)

2. **Specify Targets**: Enter ChEMBL target IDs (e.g., `CHEMBL204, CHEMBL205`) or leave empty to fetch all

3. **Configure Settings**:
   - Task Type: Regression (DTA) for continuous binding affinity prediction
   - Split Method: Scaffold, Random, or Temporal
   - Test/Validation sizes

4. **Build Dataset**: Click "Build Dataset" and wait for processing

5. **Review Results**: Examine the dataset preview, visualizations, and statistics

6. **Download**: Export your dataset as CSV or ZIP archive

## Option 2: Python API (scripting)

For scripting and automation, use the Python API:

### Build a DTA Dataset

```python
from dta_gnn.pipeline import Pipeline

pipeline = Pipeline(source_type="sqlite", sqlite_path="./chembl_dbs/chembl_36.db")
df = pipeline.build_dta(
    target_ids=["CHEMBL204", "CHEMBL205"],
    split_method="scaffold",
    test_size=0.2,
    val_size=0.1,
    output_path="dataset_dta.csv"
)
```

**Parameters explained:**

| Parameter | Description |
|-----------|-------------|
| `target_ids` | List of ChEMBL target IDs (e.g. `["CHEMBL204", "CHEMBL205"]`) |
| `split_method` | Split strategy: `"random"`, `"scaffold"`, or `"temporal"` |
| `test_size` | Fraction for test set (default: 0.2) |
| `val_size` | Fraction for validation set (default: 0.1) |
| `output_path` | Optional path to save dataset CSV |
| `source_type` / `sqlite_path` | Pipeline init: `"web"` or `"sqlite"` and path to `.db` |

## Option 3: Python API (full workflow)

For maximum flexibility and integration into your workflows:

```python
from dta_gnn.pipeline import Pipeline

# Initialize with SQLite database
pipeline = Pipeline(
    source_type="sqlite",
    sqlite_path="./chembl_dbs/chembl_36.db"
)

# Build a DTA regression dataset
df = pipeline.build_dta(
    target_ids=["CHEMBL204", "CHEMBL205", "CHEMBL206"],
    split_method="scaffold",
    test_size=0.2,
    val_size=0.1,
    output_path="dataset_dta.csv"
)

# Examine the result
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSplit distribution:")
print(df["split"].value_counts())
print(f"\nLabel statistics:")
print(df["label"].describe())
```

### Output Dataset Columns

The generated dataset contains these key columns:

| Column | Description |
|--------|-------------|
| `molecule_chembl_id` | ChEMBL molecule identifier |
| `target_chembl_id` | ChEMBL target identifier |
| `smiles` | Canonical SMILES string |
| `pchembl_value` | Standardized activity value |
| `label` | Continuous pChEMBL value for binding affinity |
| `split` | Dataset split: train, val, or test |

## Example Workflow

A complete example: build a dataset, then train both a Random Forest baseline
and a GNN on it.

```python
from dta_gnn.io.runs import create_run_dir
from dta_gnn.pipeline import Pipeline
from dta_gnn.models import (
    train_random_forest_on_run,
    train_gnn_on_run,
    GnnTrainConfig,
)

# 1. Create a timestamped run directory (also updates runs/current)
run_dir = create_run_dir()
print(f"Run directory: {run_dir}")

# 2. Build the dataset (writes dataset.csv to the run directory)
pipeline = Pipeline(source_type="sqlite", sqlite_path="./chembl_dbs/chembl_36.db")
df = pipeline.build_dta(
    target_ids=["CHEMBL204"],
    split_method="scaffold",
    test_size=0.2,
    val_size=0.1,
    output_path=str(run_dir / "dataset.csv"),
)

# 3. Save compounds.csv — required by all on-run trainers
df[["molecule_chembl_id", "smiles"]].drop_duplicates().to_csv(
    run_dir / "compounds.csv", index=False,
)

# 4. Train a Random Forest baseline (Morgan ECFP4 fingerprints)
rf = train_random_forest_on_run(run_dir, n_estimators=500)
print("RF test RMSE:", rf.metrics["splits"]["test"]["rmse"])

# 5. Train a GNN (PyTorch Geometric)
gnn = train_gnn_on_run(run_dir, config=GnnTrainConfig(
    architecture="gin",
    hidden_dim=128,
    num_layers=3,
    epochs=20,
))
print("GNN test RMSE:", gnn.metrics["splits"]["test"]["rmse"])
```

> **Note** — `Pipeline.build_dta` returns the DataFrame and writes
> `dataset.csv`, but it does **not** write `compounds.csv`. The on-run
> trainers (`train_random_forest_on_run`, `train_svr_on_run`,
> `train_gnn_on_run`) require both files in the run directory.

## Dataset Splits Explained

DTA-GNN offers several splitting strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Random** | Random assignment | Baseline, general ML |
| **Scaffold** | Based on molecular scaffolds | Drug discovery, prevents scaffold leakage |
| **Temporal** | Based on publication year | Simulates prospective prediction |

!!! warning "Data Leakage"
    Always use scaffold split for drug discovery applications. Random splits can lead to overly optimistic performance estimates due to scaffold leakage.

## Next Steps

Now that you've built your first dataset:

- [Data Sources](../user-guide/data-sources.md) - Learn about web vs SQLite sources
- [Cleaning](../user-guide/cleaning.md) - Understand data standardization
- [Splits](../user-guide/splits.md) - Deep dive into splitting strategies
- [Models](../modeling/models.md) - Train baseline models
- [Hyperopt](../hpo/hyperopt.md) - Optimize hyperparameters

## Common Issues

### No activities found

```python
# Check if your target IDs are valid
from dta_gnn.io.sqlite_source import ChemblSQLiteSource
source = ChemblSQLiteSource("chembl_36.db")
targets = source.fetch_targets(["CHEMBL204"])
print(targets)
```

### Dataset is empty after cleaning

This usually means all activities were filtered out. Try:

1. Using more targets
2. Including more standard types (IC50, Ki, Kd)
3. Checking that your target IDs are valid

### Memory errors with large datasets

For datasets with millions of activities:

```python
# Use chunked processing
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")

# Limit to specific activity types
df = pipeline.build_dta(
    target_ids=None,  # All targets
    standard_types=["IC50"],  # Only IC50
    split_method="random"
)
```
