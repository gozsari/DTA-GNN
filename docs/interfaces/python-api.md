# Python API

DTA-GNN provides a comprehensive Python API for programmatic dataset building and model training.

## Core Classes

### Pipeline

The main entry point for building datasets.

```python
from dta_gnn.pipeline import Pipeline

# Initialize with SQLite
pipeline = Pipeline(
    source_type="sqlite",
    sqlite_path="chembl_36.db"
)

# Or with Web API
pipeline = Pipeline(source_type="web")
```

#### build_dta

Build a Drug-Target Affinity (regression) dataset.

```python
df = pipeline.build_dta(
    target_ids=["CHEMBL204"],
    molecule_ids=None,                      # Optional: filter by molecules
    standard_types=["IC50", "Ki"],          # Activity types
    split_method="scaffold",
    output_path="dataset.csv",
    test_size=0.2,
    val_size=0.1,
    split_year=2022,                        # Year threshold for temporal split (only used when split_method='temporal')
    featurize=False
)
```

**Parameters:**
- `target_ids`: Optional list of ChEMBL target IDs
- `molecule_ids`: Optional list of ChEMBL molecule IDs to filter by
- `standard_types`: Optional list of activity standard types (e.g., ["IC50", "Ki"])
- `split_method`: Split strategy - "random", "scaffold", or "temporal" (default: "random")
- `output_path`: Optional path to save dataset CSV
- `test_size`: Fraction of data for test set (default: 0.2)
- `val_size`: Fraction of data for validation set (default: 0.1)
- `split_year`: Year threshold for temporal split, only used when `split_method="temporal"` (default: 2022)
- `featurize`: Whether to calculate Morgan fingerprints (default: False)

**Returns:** `pd.DataFrame` with `label` as continuous pChEMBL value.

## Data Sources

### ChemblSQLiteSource

Direct access to SQLite database.

```python
from dta_gnn.io.sqlite_source import ChemblSQLiteSource

source = ChemblSQLiteSource("chembl_36.db")
```

#### fetch_activities

```python
activities = source.fetch_activities(
    target_ids=["CHEMBL204"],       # Optional
    molecule_ids=["CHEMBL25"],      # Optional
    standard_types=["IC50", "Ki"]   # Optional
)
```

**Returns:** `pd.DataFrame` with raw activity data.

#### fetch_molecules

```python
molecules = source.fetch_molecules(
    molecule_ids=["CHEMBL25", "CHEMBL192"]
)
```

**Returns:** `pd.DataFrame` with `molecule_chembl_id` and `smiles`.

#### fetch_targets

```python
targets = source.fetch_targets(
    target_ids=["CHEMBL204"]
)
```

**Returns:** `pd.DataFrame` with `target_chembl_id`, `sequence`, `organism`.

### ChemblWebSource

Access via ChEMBL REST API.

```python
from dta_gnn.io.web_source import ChemblWebSource

source = ChemblWebSource()
activities = source.fetch_activities(target_ids=["CHEMBL204"])
```

## Cleaning Functions

### standardize_activities

```python
from dta_gnn.cleaning import standardize_activities

df_clean = standardize_activities(
    df_activities,
    convert_to_pchembl=True,  # Calculate pChEMBL if missing
    drop_censored=False       # Keep >, < relations
)
```

### aggregate_duplicates

```python
from dta_gnn.cleaning import aggregate_duplicates

df_agg = aggregate_duplicates(
    df_clean,
    group_cols=["molecule_chembl_id", "target_chembl_id"],
    agg_method="median"  # median, mean, max, min
)
```

### canonicalize_smiles

```python
from dta_gnn.cleaning import canonicalize_smiles

canonical = canonicalize_smiles("c1ccccc1O")
# Returns: "Oc1ccccc1"
```

## Splitting Functions

### split_random

```python
from dta_gnn.splits import split_random

df_split, train, val, test = split_random(
    df,
    test_size=0.2,
    val_size=0.1,
    seed=42
)
```

### split_cold_drug_scaffold

```python
from dta_gnn.splits import split_cold_drug_scaffold

df_split = split_cold_drug_scaffold(
    df,
    smiles_col="smiles",
    test_size=0.2,
    val_size=0.1,
    seed=42
)
```

### split_temporal

```python
from dta_gnn.splits import split_temporal

df_split = split_temporal(
    df,
    year_col="year",
    split_year=2022,
    val_size=0.1
)
```

## Audit Functions

### audit_scaffold_leakage

```python
from dta_gnn.audits import audit_scaffold_leakage

result = audit_scaffold_leakage(
    train_df,
    test_df,
    smiles_col="smiles"
)
# Returns: {'train_scaffolds': 1000, 'test_scaffolds': 200, 
#           'overlap_count': 0, 'leakage_ratio': 0.0}
```

### audit_target_leakage

```python
from dta_gnn.audits import audit_target_leakage

result = audit_target_leakage(
    train_df,
    test_df,
    target_col="target_chembl_id"
)
```

## Featurization

### Morgan Fingerprints

```python
from dta_gnn.features import calculate_morgan_fingerprints

df_feat = calculate_morgan_fingerprints(
    df,
    smiles_col="smiles",
    radius=2,              # ECFP4
    n_bits=2048,
    out_col="morgan_fingerprint",
    drop_failures=True
)
```

### Molecule Graphs (2D)

```python
from dta_gnn.features.molecule_graphs import build_graphs_2d

graphs = build_graphs_2d(
    molecules=[("CHEMBL25", "CCO"), ("CHEMBL192", "CC(=O)O")],
    drop_failures=True
)

for g in graphs:
    print(f"Molecule: {g.molecule_chembl_id}")
    print(f"  Atoms: {g.atom_type.shape[0]}")
    print(f"  Edges: {g.edge_index.shape[1]}")
```

## Model Training

### Random Forest

```python
from dta_gnn.models import train_random_forest_on_run

result = train_random_forest_on_run(
    run_dir="runs/current",
    n_estimators=500,
    random_seed=42
)

print(f"Task type: {result.task_type}")
print(f"Metrics: {result.metrics}")
print(f"Model saved to: {result.model_path}")
```

**Returns:** `RandomForestTrainResult` dataclass with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `run_dir` | Path | Run directory |
| `task_type` | str | regression (always) |
| `model_path` | Path | Saved model file |
| `metrics_path` | Path | Metrics JSON file |
| `predictions_path` | Path | Predictions CSV |
| `metrics` | dict | Evaluation metrics |

### GNN (Graph Neural Network)

```python
from dta_gnn.models.gnn import GnnTrainConfig, train_gnn_on_run

config = GnnTrainConfig(
    architecture="gin",     # gin, gcn, gat, sage, pna, transformer, tag, arma, cheb, supergat
    embedding_dim=128,
    hidden_dim=128,
    num_layers=5,
    dropout=0.1,
    pooling="add",          # add, mean, max, attention
    # Architecture-specific parameters (optional)
    gin_conv_mlp_layers=2,  # GIN: MLP depth in convolution
    gin_train_eps=False,    # GIN: Whether to learn epsilon
    gin_eps=0.0,            # GIN: Initial epsilon value
    gat_heads=4,            # GAT: Number of attention heads
    sage_aggr="mean",       # GraphSAGE: Aggregation method
    transformer_heads=4,    # Transformer: Number of attention heads
    tag_k=2,                # TAG: K-hop message passing
    arma_num_stacks=1,      # ARMA: Number of stacks
    arma_num_layers=1,      # ARMA: Number of layers per stack
    cheb_k=2,               # Cheb: K-hop spectral filtering
    supergat_heads=4,       # SuperGAT: Number of attention heads
    supergat_attention_type="MX",  # SuperGAT: Attention type
    lr=1e-3,
    batch_size=64,
    epochs=10
)

result = train_gnn_on_run("runs/current", config=config)

print(f"Metrics: {result.metrics}")
```

!!! note
    GNN support is included in the default install.

### GNN Embedding Extraction

```python
from dta_gnn.models.gnn import extract_gnn_embeddings_on_run

result = extract_gnn_embeddings_on_run(
    "runs/current",
    batch_size=256
)

print(f"Extracted {result.n_molecules} embeddings")
print(f"Dimension: {result.embedding_dim}")
```

## Hyperparameter Optimization

### Random Forest HPO

```python
from dta_gnn.models.hyperopt import (
    HyperoptConfig,
    optimize_random_forest_wandb
)

config = HyperoptConfig(
    model_type="RandomForest",
    n_trials=20,
    rf_optimize_n_estimators=True,
    rf_n_estimators_min=50,
    rf_n_estimators_max=500,
    rf_optimize_max_depth=True,
    rf_max_depth_min=5,
    rf_max_depth_max=50
)

result = optimize_random_forest_wandb(
    "runs/current",
    config=config,
    project="my-project",
    api_key="your-wandb-key"  # Optional
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_value}")
```

### GNN HPO

```python
from dta_gnn.models.hyperopt import optimize_gnn_wandb

config = HyperoptConfig(
    model_type="GNN",
    n_trials=20,
    gnn_architecture="gin",
    gnn_optimize_epochs=True,
    gnn_optimize_lr=True,
    gnn_optimize_batch_size=True
)

result = optimize_gnn_wandb(
    "runs/current",
    config=config,
    project="my-project"
)
```

!!! note
    W&B is included in the default install.

## Model Prediction

Make predictions on new molecules using trained models.

### Random Forest Prediction

```python
from dta_gnn.models import predict_with_random_forest
from pathlib import Path

# Single prediction
result = predict_with_random_forest(
    run_dir="runs/current",
    smiles_list=["CCO", "CC(=O)O", "c1ccccc1"],
    molecule_ids=["mol_1", "mol_2", "mol_3"]  # Optional
)

print(result.predictions)
#   molecule_id    smiles  prediction
# 0       mol_1       CCO         6.2
# 1       mol_2   CC(=O)O         5.8
# 2       mol_3   c1ccccc1         4.5

# Batch prediction from file
import pandas as pd
df_new = pd.read_csv("new_compounds.csv")
result = predict_with_random_forest(
    run_dir="runs/current",
    smiles_list=df_new["smiles"].tolist(),
    molecule_ids=df_new["id"].tolist() if "id" in df_new.columns else None
)

# Save predictions
result.predictions.to_csv("predictions_rf.csv", index=False)
```

**Returns:** `PredictionResult` with:
- `predictions`: DataFrame with molecule_id, smiles, prediction
- `model_type`: "RandomForest"
- `model_path`: Path to model file
- `run_dir`: Run directory path

### SVR Prediction

```python
from dta_gnn.models import predict_with_svr

result = predict_with_svr(
    run_dir="runs/current",
    smiles_list=["CCO", "CC(=O)O", "c1ccccc1"],
    molecule_ids=None  # Will auto-generate mol_0, mol_1, ...
)

print(result.predictions)
print(f"Model: {result.model_type}")
print(f"Model path: {result.model_path}")
```

**Returns:** `PredictionResult` with:
- `predictions`: DataFrame with molecule_id, smiles, prediction
- `model_type`: "SVR"
- `model_path`: Path to model file
- `run_dir`: Run directory path

### GNN Prediction

```python
from dta_gnn.models import predict_with_gnn

# Auto-detect architecture
result = predict_with_gnn(
    run_dir="runs/current",
    smiles_list=["CCO", "CC(=O)O", "c1ccccc1"],
    molecule_ids=["mol_1", "mol_2", "mol_3"],
    batch_size=64,
    device="cuda",  # Optional: "cuda", "mps", "cpu", or None (auto)
    architecture=None  # Auto-detect from model files
)

# Explicit architecture
result = predict_with_gnn(
    run_dir="runs/current",
    smiles_list=["CCO", "CC(=O)O"],
    batch_size=128,
    device="cpu",
    architecture="gin"  # Explicitly specify: gin, gcn, gat, sage, pna, transformer, tag, arma, cheb, supergat
)

print(result.predictions)
print(f"Model: {result.model_type}")
print(f"Architecture: {result.model_path}")  # Path contains architecture name
```

**Device Selection:**
- `"cuda"`: Use GPU (CUDA)
- `"mps"`: Use Apple Silicon GPU (macOS)
- `"cpu"`: Use CPU
- `None`: Auto-detect (prefers GPU if available)

**Returns:** `PredictionResult` with:
- `predictions`: DataFrame with molecule_id, smiles, prediction
- `model_type`: "GNN"
- `model_path`: Path to model file (contains architecture name)
- `run_dir`: Run directory path

!!! note
    GNN prediction uses PyTorch Geometric, included in the default install.

!!! warning "Transformer Architecture on MPS"
    TransformerConv doesn't support MPS. If using transformer architecture on Apple Silicon, it will automatically fall back to CPU.

### Error Handling

```python
from dta_gnn.models import predict_with_gnn

try:
    result = predict_with_gnn(
        run_dir="runs/current",
        smiles_list=["CCO", "INVALID_SMILES", "c1ccccc1"]
    )
    # Invalid SMILES are marked as None in predictions
    valid = result.predictions[result.predictions["prediction"].notna()]
    print(f"Valid predictions: {len(valid)}/{len(result.predictions)}")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Artifact Export and Collection

DTA-GNN provides utilities to collect and export all artifacts from a run directory.

### Collect Artifacts

```python
from dta_gnn.exporters import collect_artifacts

# Collect all artifacts from a run directory
artifacts = collect_artifacts(run_dir="runs/current")

print(artifacts)
# {
#     "dataset": "runs/current/dataset.csv",
#     "targets": "runs/current/targets.csv",
#     "compounds": "runs/current/compounds.csv",
#     "metadata": "runs/current/metadata.json",
#     "model": "runs/current/model_rf.pkl",
#     "model_metrics": "runs/current/model_metrics.json",
#     "model_predictions": "runs/current/model_predictions.csv",
#     "model_gnn": "runs/current/model_gnn_gin.pt",
#     "encoder_gnn": "runs/current/encoder_gin.pt",
#     "molecule_embeddings": "runs/current/molecule_embeddings.npz",
#     "zip": "runs/current/artifacts.zip"
# }

# Check which artifacts exist
for key, path in artifacts.items():
    if path:
        print(f"{key}: {path}")
```

**Artifact Keys:**
- `dataset`: Main dataset CSV
- `targets`: Target information CSV
- `compounds`: Molecules CSV
- `metadata`: Run metadata JSON
- `model`: RandomForest model (model_rf.pkl)
- `model_metrics`: Model metrics JSON
- `model_predictions`: Predictions CSV
- `model_gnn`: GNN model file (model_gnn_<arch>.pt)
- `model_metrics_gnn`: GNN metrics JSON
- `model_predictions_gnn`: GNN predictions CSV
- `encoder_gnn`: GNN encoder weights (encoder_<arch>.pt)
- `encoder_gnn_config`: Encoder configuration JSON
- `molecule_embeddings`: Extracted embeddings NPZ
- `molecule_features`: Molecule features CSV
- `protein_features`: Protein features CSV
- `zip`: Artifacts ZIP archive

### Create Artifacts ZIP

```python
from dta_gnn.exporters import (
    collect_artifacts,
    write_artifacts_zip_from_manifest
)

# Collect artifacts
artifacts = collect_artifacts(run_dir="runs/current")

# Create ZIP archive
zip_path = write_artifacts_zip_from_manifest(artifacts=artifacts)

print(f"Created ZIP: {zip_path}")
# Output: runs/current/artifacts.zip
```

### Artifacts Table

Create a DataFrame table for UI display:

```python
from dta_gnn.exporters import collect_artifacts, artifacts_table

artifacts = collect_artifacts(run_dir="runs/current")
table = artifacts_table(artifacts)

print(table)
#        artifact                    path
# 0    dataset.csv  runs/current/dataset.csv
# 1    targets.csv  runs/current/targets.csv
# 2  compounds.csv  runs/current/compounds.csv
# ...
```

### Custom Artifact Collection

```python
from dta_gnn.exporters import collect_artifacts

# Provide explicit paths
artifacts = collect_artifacts(
    run_dir=None,  # Don't use run_dir
    dataset_path="custom/dataset.csv",
    targets_path="custom/targets.csv",
    compounds_path="custom/compounds.csv"
)

# Or mix run_dir with explicit paths
artifacts = collect_artifacts(
    run_dir="runs/current",
    dataset_path="custom/dataset.csv"  # Override dataset path
)
```

## Dataset Export

### Generate Dataset Card

Create a markdown dataset card documenting your dataset:

```python
from dta_gnn.exporters.card import generate_dataset_card
import json
import pandas as pd

# Load metadata
with open("runs/current/metadata.json") as f:
    metadata = json.load(f)

# Load dataset
df = pd.read_csv("runs/current/dataset.csv")

# Generate card
generate_dataset_card(
    df=df,
    metadata=metadata,
    output_path="runs/current/dataset_card.md"
)
```

**Parameters:**
- `df`: DataFrame containing the dataset (must have `label` column for statistics)
- `metadata`: Dictionary containing dataset metadata (see structure below)
- `output_path`: Path where the markdown card will be written

**Metadata Structure:**

The `metadata` dictionary should contain the following fields:

```python
metadata = {
    # Required fields
    "targets": ["CHEMBL204", "CHEMBL205"],  # List of target ChEMBL IDs
    "source": "sqlite",                       # Data source: "web" or "sqlite"
    "split_method": "scaffold",               # Split strategy used
    
    # Optional fields
    "audit": {                                # Leakage audit results (JSON-serializable)
        "train_scaffolds": 1000,
        "test_scaffolds": 200,
        "overlap_count": 0,
        "leakage_ratio": 0.0
    }
}
```

**Required Metadata Fields:**
- `targets`: List of target ChEMBL IDs used in the dataset
- `source`: Data source type ("web" or "sqlite")
- `split_method`: Split strategy used ("random", "scaffold", or "temporal")

**Optional Metadata Fields:**
- `audit`: Dictionary or JSON-serializable object containing leakage audit results

**Output:**

The dataset card includes:
- **Metadata**: Target IDs, data source, creation date
- **Statistics**: Total samples, label range (min-max), mean affinity
- **Split Information**: Split strategy and counts for train/val/test splits
- **Preprocessing**: Details about deduplication and standardization
- **Leakage Audit**: JSON-formatted audit results (if provided in metadata)

## Run Management

DTA-GNN organizes datasets and model artifacts in timestamped run directories for reproducibility.

### Create Run Directory

```python
from dta_gnn.io.runs import create_run_dir

run_dir = create_run_dir()
# Returns: Path("runs/20260111_143025")
# Also creates/updates runs/current symlink
```

### Resolve Run Directory

```python
from dta_gnn.io.runs import resolve_run_dir, resolve_current_run_dir

# Resolve any run directory path
run_path = resolve_run_dir("runs/20260111_143025")
# Returns: Path object or None if invalid

# Resolve the current run (runs/current)
try:
    current_run = resolve_current_run_dir()
    print(f"Current run: {current_run}")
except FileNotFoundError:
    print("No current run found. Build a dataset first.")
```

### Save Metadata

```python
import json

metadata = {
    "inputs": {
        "target_ids": ["CHEMBL204"],
        "split_method": "scaffold",
        "task_type": "regression"
    },
    "created_at": "2026-01-11T14:30:25"
}

(run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
```

## Model Utilities

### List Available Models

Check which models are trained in a run directory:

```python
from dta_gnn.models.utils import list_available_models
from dta_gnn.io.runs import resolve_current_run_dir

run_dir = resolve_current_run_dir()
models = list_available_models(run_dir)

print(models)
# {
#     "rf": ["RandomForest"],
#     "svr": ["SVR"],
#     "gnn": ["GNN (GIN)", "GNN (GAT)"]
# }

# Check if a specific model exists
if models["gnn"]:
    print(f"Available GNN architectures: {models['gnn']}")
```

## IO Utilities

### Find ChEMBL Databases

Automatically discover ChEMBL SQLite databases:

```python
from dta_gnn.io.utils import find_chembl_sqlite_dbs

dbs = find_chembl_sqlite_dbs()
print(f"Found {len(dbs)} databases:")
for db in dbs:
    print(f"  - {db}")
```

### Preview CSV Files

Preview CSV files with error handling:

```python
from dta_gnn.io.utils import preview_csv, preview_csv_with_error

# Simple preview (returns DataFrame or None)
df = preview_csv("dataset.csv", n=50)
if df is not None:
    print(df.head())

# Preview with error details
result = preview_csv_with_error("dataset.csv", n=50)
if result.error:
    print(f"Error: {result.error}")
else:
    print(result.df.head())
```

### Normalize Paths

```python
from dta_gnn.io.utils import normalize_csv_path

path = normalize_csv_path("  ./data/dataset.csv  ")
# Returns: "./data/dataset.csv" (normalized, or None if empty)
```

## Database Downloader

Download ChEMBL SQLite databases programmatically:

```python
from dta_gnn.io.downloader import download_chembl_db

# Download ChEMBL 36 to current directory
db_path = download_chembl_db(version="36", output_dir=".")

print(f"Database downloaded to: {db_path}")
# Output: ./chembl_36/chembl_36_sqlite/chembl_36.db
```

**Parameters:**
- `version`: ChEMBL version (default: "36")
- `output_dir`: Directory to download to (default: ".")

**Returns:** Path to the extracted `.db` file

## Target Mapping

Map UniProt accessions to ChEMBL target IDs for easier target selection.

### Parse UniProt Accessions

```python
from dta_gnn.io.target_mapping import parse_uniprot_accessions

# Parse from text (comma, newline, or space separated)
accessions = parse_uniprot_accessions("P00533, P04626\nP15056")
# Returns: ["P00533", "P04626", "P15056"]
```

### Parse ChEMBL Target IDs

```python
from dta_gnn.io.target_mapping import parse_chembl_target_ids

targets = parse_chembl_target_ids("CHEMBL204, CHEMBL205")
# Returns: ["CHEMBL204", "CHEMBL205"]
```

### Map UniProt to ChEMBL (SQLite)

```python
from dta_gnn.io.target_mapping import map_uniprot_to_chembl_targets_sqlite

result = map_uniprot_to_chembl_targets_sqlite(
    sqlite_path="chembl_36.db",
    accessions=["P00533", "P04626"]
)

print(f"Resolved targets: {result.resolved_target_chembl_ids}")
# Output: ["CHEMBL1862", "CHEMBL203"]

print(f"Per-input mapping: {result.per_input}")
# Output: {"P00533": ["CHEMBL1862"], "P04626": ["CHEMBL203"]}

print(f"Unmapped: {result.unmapped}")
# Output: [] (empty if all mapped)
```

### Map UniProt to ChEMBL (Web API)

```python
from dta_gnn.io.target_mapping import map_uniprot_to_chembl_targets_web

result = map_uniprot_to_chembl_targets_web(
    accessions=["P00533", "P04626"]
)

print(f"Resolved: {result.resolved_target_chembl_ids}")
```

**Use Case:** When you have UniProt IDs from literature or databases but need ChEMBL target IDs for dataset building.

## UI Helper Functions

!!! note "Internal Use"
    The `app_features` module contains utility functions primarily designed for the Gradio web UI. These functions are available for advanced users who want to programmatically build SMILES or protein sequence DataFrames and compute features, but they are not part of the main public API.

### Compound Features

Build SMILES DataFrames and compute Morgan fingerprints:

```python
from dta_gnn.app_features.compound import build_smiles_frame, featurize_smiles_morgan

# Build SMILES DataFrame from text input
smiles_text = "CCO\nCC(=O)O\nc1ccccc1"
df_smiles = build_smiles_frame(
    smiles_text=smiles_text,
    df_state=None,
    source_mode="text"
)

# Or from existing dataset
df_smiles = build_smiles_frame(
    smiles_text="",
    df_state=dataset_df,  # DataFrame with 'smiles' or 'canonical_smiles' column
    source_mode="dataset"
)

# Compute Morgan fingerprints
df_feat = featurize_smiles_morgan(
    df_smiles,
    smiles_col="smiles",
    radius=2,
    n_bits=2048
)
```

**Parameters:**
- `build_smiles_frame`:
  - `smiles_text`: Text input with SMILES (comma, newline, or space separated)
  - `df_state`: Optional DataFrame with SMILES column
  - `source_mode`: "text" or "dataset" mode
- `featurize_smiles_morgan`:
  - `df`: DataFrame with SMILES column
  - `smiles_col`: Name of SMILES column (default: "smiles")
  - `radius`: Morgan fingerprint radius (default: 2)
  - `n_bits`: Fingerprint bit size (default: 2048)

### Protein Sequence Features

Build protein sequence DataFrames:

```python
from dta_gnn.app_features.proteins import build_sequence_frame

# Build sequence DataFrame from text input
seqs_text = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"

df_seqs = build_sequence_frame(
    seqs_text=seqs_text,
    df_state=None,
    source_mode="text"
)
```

**Parameters:**
- `seqs_text`: Text input with protein sequences (comma, newline, or space separated)
- `df_state`: Optional DataFrame with 'sequence' column
- `source_mode`: "text" or "dataset" mode

!!! tip
    For most use cases, use the main API functions (`Pipeline.build_dta()`, `calculate_morgan_fingerprints()`) instead of these UI helper functions. These are primarily intended for internal UI workflows.

## Complete Example

```python
from dta_gnn.pipeline import Pipeline
from dta_gnn.models import train_random_forest_on_run
from dta_gnn.io.runs import create_run_dir
from dta_gnn.audits import audit_scaffold_leakage

# 1. Create run directory
run_dir = create_run_dir()
print(f"Run: {run_dir}")

# 2. Build dataset
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")

df = pipeline.build_dta(
    target_ids=["CHEMBL204", "CHEMBL205"],
    split_method="scaffold",
    test_size=0.2,
    val_size=0.1,
    output_path=str(run_dir / "dataset.csv")
)

# 3. Save compounds
compounds = df[["molecule_chembl_id", "smiles"]].drop_duplicates()
compounds.to_csv(run_dir / "compounds.csv", index=False)

# 4. Audit for leakage
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
audit = audit_scaffold_leakage(train, test)
print(f"Scaffold leakage: {audit['leakage_ratio']:.2%}")

# 5. Train model
result = train_random_forest_on_run(run_dir)

# 6. Review results
print(f"\nModel Results:")
print(f"  Task: {result.task_type}")
for split, metrics in result.metrics.get("splits", {}).items():
    print(f"  {split}: {metrics}")
```

## Error Handling

```python
from dta_gnn.pipeline import Pipeline

try:
    pipeline = Pipeline(source_type="sqlite", sqlite_path="missing.db")
except FileNotFoundError as e:
    print(f"Database not found: {e}")

try:
    df = pipeline.build_dta(target_ids=["INVALID"])
except ValueError as e:
    print(f"Invalid target: {e}")
```

## Type Hints

DTA-GNN uses type hints throughout. Example:

```python
from typing import List, Optional
from pathlib import Path
import pandas as pd

def build_custom_dataset(
    pipeline: "Pipeline",
    targets: List[str],
    output: Optional[Path] = None
) -> pd.DataFrame:
    return pipeline.build_dta(
        target_ids=targets,
        output_path=str(output) if output else None
    )
```
