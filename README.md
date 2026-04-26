<div align="center">

<img src="https://raw.githubusercontent.com/gozsari/DTA-GNN/main/assets/logo3.png" alt="DTA-GNN Logo" width="400"/>

# DTA-GNN: Target-Specific Binding Affinity Dataset Builder and GNN Trainer

**Build leakage-free Drug–Target Affinity datasets from ChEMBL and train Graph Neural Networks for any target of interest.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🧬 *From a UniProt accession → curated ChEMBL dataset → trained GNN with test-set metrics, in a single call.*

[Documentation](docs/index.md) · [Quick Start](#-quick-start) · [Examples](examples/) · [Python API](docs/interfaces/python-api.md) · [CLI](docs/interfaces/cli.md)

</div>

---

## 🎯 Overview

DTA-GNN is an end-to-end toolkit for Drug–Target Affinity (DTA) prediction. It:

1. **Curates** clean, leakage-free regression datasets from ChEMBL (web API or local SQLite).
2. **Featurises** molecules either as Morgan fingerprints (ECFP4) or 2D molecular graphs.
3. **Trains** baseline models (Random Forest, SVR) and 10 Graph Neural Network architectures (GIN, GCN, GAT, GraphSAGE, PNA, Transformer, TAG, ARMA, Cheb, SuperGAT).
4. **Evaluates** with proper scaffold-aware splitting and built-in leakage audits.
5. **Tracks** hyperparameter search and final training in Weights & Biases.

The regression label is always **pChEMBL**.

<div align="center">

![DTA-GNN Overview](https://raw.githubusercontent.com/gozsari/DTA-GNN/main/assets/overview.png)
*DTA-GNN workflow: clean data from ChEMBL, molecular graph conversion, scaffold-aware splitting, and GNN training.*

</div>

---

## 📦 Installation

DTA-GNN supports Python 3.10+ on Linux, macOS, and Windows.

### From source (recommended while iterating)

```bash
git clone https://github.com/gozsari/DTA-GNN.git
cd DTA-GNN
pip install -e .
# With developer tools (pytest, ruff, black, build, twine):
pip install -e ".[dev]"
```

### From PyPI

```bash
pip install dta-gnn
```

PyTorch, PyTorch Geometric, RDKit, and Weights & Biases are installed by default. See the [Installation Guide](docs/getting-started/installation.md) for GPU/CUDA notes and RDKit troubleshooting.

### Docker

```bash
# Pre-built image
docker pull ghcr.io/gozsari/dta-gnn:latest

# Web UI on http://localhost:7860 (web-API mode, no local database)
docker run --rm -p 7860:7860 ghcr.io/gozsari/dta-gnn:latest \
  dta_gnn ui --host 0.0.0.0

# Web UI with a local ChEMBL SQLite database mounted in
docker run --rm -p 7860:7860 \
  -v $(pwd)/chembl_dbs:/home/dtagnn/app/chembl_dbs \
  -v $(pwd)/runs:/home/dtagnn/app/runs \
  ghcr.io/gozsari/dta-gnn:latest dta_gnn ui --host 0.0.0.0

# Or with Docker Compose
docker-compose up ui        # Web UI
docker-compose up jupyter   # JupyterLab
```

---

## 🚀 Quick Start

### One command, end-to-end (CLI)

Resolve a UniProt accession, build a scaffold-split dataset, run a W&B Bayesian hyperparameter sweep, train the final GNN, and report test metrics:

```bash
# Web API (no local DB — slower but zero setup)
dta_gnn train-gnn P00533 --architecture gin --n-trials 20 --epochs 30

# Or with a local ChEMBL SQLite database (much faster)
dta_gnn setup --version 36 --dir ./chembl_dbs
dta_gnn train-gnn P00533 \
  --architecture gin \
  --sqlite-path ./chembl_dbs/chembl_36.db \
  --wandb-project my_project \
  --n-trials 20 --epochs 30
```

The CLI prints a per-step timing breakdown and the test-set RMSE / MAE / R² when the run finishes. All artifacts land in a timestamped `runs/<TIMESTAMP>/` folder.

### One call, end-to-end (Python)

```python
from dta_gnn.training import run_gnn_end_to_end, EndToEndConfig

result = run_gnn_end_to_end(EndToEndConfig(
    uniprot_ids="P00533",            # EGFR — any UniProt accession
    architecture="gin",              # gin | gcn | gat | sage | pna | transformer | tag | arma | cheb | supergat
    sqlite_path="./chembl_dbs/chembl_36.db",  # omit to use the ChEMBL web API
    wandb_project="my_project",
    n_trials=20,
    epochs=30,
))

print(result.test_metrics)   # {"rmse": ..., "mae": ..., "r2": ...}
print(result.timings)        # {"uniprot_mapping": ..., "dataset_build": ..., ...}
print(result.run_dir)        # Path("runs/20260309_142301")
```

### Step-by-step (Python — full control)

`Pipeline.build_dta` returns a DataFrame; baseline / GNN trainers expect a *run directory* containing `dataset.csv` and `compounds.csv`. Set those up explicitly:

```python
from dta_gnn.io.runs import create_run_dir
from dta_gnn.pipeline import Pipeline
from dta_gnn.models import (
    train_random_forest_on_run,
    train_svr_on_run,
    train_gnn_on_run,
    GnnTrainConfig,
)

# 1. Create a fresh timestamped run directory and update runs/current
run_dir = create_run_dir()

# 2. Build the dataset (writes dataset.csv to the run directory)
pipeline = Pipeline(source_type="sqlite", sqlite_path="./chembl_dbs/chembl_36.db")
df = pipeline.build_dta(
    target_ids=["CHEMBL1862"],   # any ChEMBL target ID(s)
    split_method="scaffold",     # "random" | "scaffold" | "temporal"
    test_size=0.2,
    val_size=0.1,
    output_path=str(run_dir / "dataset.csv"),
)

# 3. Save compounds.csv (required by the trainers below)
df[["molecule_chembl_id", "smiles"]].drop_duplicates().to_csv(
    run_dir / "compounds.csv", index=False
)

# 4. Train baselines on Morgan fingerprints (ECFP4)
rf  = train_random_forest_on_run(run_dir, n_estimators=500)
svr = train_svr_on_run(run_dir, C=10.0, epsilon=0.1, kernel="rbf")
print("RF  test RMSE:", rf.metrics["splits"]["test"]["rmse"])
print("SVR test RMSE:", svr.metrics["splits"]["test"]["rmse"])

# 5. Train a Graph Neural Network on 2D molecular graphs
gnn = train_gnn_on_run(run_dir, config=GnnTrainConfig(
    architecture="gin",
    hidden_dim=256,
    num_layers=5,
    epochs=100,
))
print("GNN test RMSE:", gnn.metrics["splits"]["test"]["rmse"])
```

---

## 🖥️ Web UI

DTA-GNN ships with an interactive Gradio interface for users who prefer not to script.

### Live demos (limited compute, may be slow)

| Platform | URL |
|----------|-----|
| Hugging Face Spaces | <https://huggingface.co/spaces/gozsari/dta-gnn> |
| SciLifeLab Serve | <https://dta-gnn.serve.scilifelab.se/> |

### Launch locally

```bash
dta_gnn ui                  # http://127.0.0.1:7860
dta_gnn ui --host 0.0.0.0   # bind to all interfaces (Docker / remote)
dta_gnn ui --share          # temporary public Gradio link
```

The UI covers dataset building, leakage audits, baseline / GNN training, hyperparameter optimisation, prediction on new molecules, embedding extraction, and 2-D embedding visualisation. See [Web UI guide](docs/interfaces/ui.md).

---

## 🔑 Key Features

- **End-to-end pipeline.** `dta_gnn train-gnn <UNIPROT>` (CLI) or `run_gnn_end_to_end(...)` (Python) handles target mapping, data fetching, cleaning, scaffold split, W&B Bayesian HPO, final training, and test evaluation in one call.
- **Two ChEMBL data sources.** Local SQLite (fast, offline) or the official ChEMBL web client (no setup) — same interface either way.
- **Three splitting strategies.** Random, Murcko-scaffold (cold-drug), and temporal (year-based) splits.
- **Leakage audits.** `audit_scaffold_leakage` and `audit_target_leakage` quantify train/test contamination.
- **10 GNN architectures.** GIN, GCN, GAT, GraphSAGE, PNA, Transformer, TAG, ARMA, Cheb, SuperGAT — all with configurable depth, pooling, residual connections, and architecture-specific hyperparameters.
- **Two baselines.** Random Forest and SVR over Morgan ECFP4 fingerprints.
- **Hyperparameter optimisation.** W&B Bayesian sweeps for RF, SVR, and GNNs (`optimize_random_forest_wandb`, `optimize_svr_wandb`, `optimize_gnn_wandb`); a non-W&B Optuna fallback is also available.
- **Reproducible run directories.** Every run is written to `runs/<TIMESTAMP>/` with `dataset.csv`, `compounds.csv`, `metadata.json`, model weights, predictions, and metrics.
- **Three interfaces.** CLI, Python API, and Gradio Web UI.

---

## 🧠 Supported GNN Architectures

| Architecture | Key idea |
|--------------|----------|
| **GIN** | Graph Isomorphism Network — sum aggregation with learnable ε; MLP-based updates with strong discriminative power |
| **GCN** | Graph Convolutional Network — symmetric normalised adjacency; efficient spectral convolution |
| **GAT** | Graph Attention Network — learnable multi-head neighbour attention |
| **GraphSAGE** | Sample-and-aggregate inductive learning with mean / max / lstm / pool aggregators |
| **PNA** | Principal Neighbourhood Aggregation — multiple aggregators with degree-aware scalers |
| **Transformer** | Graph Transformer with multi-head self-attention and optional edge features |
| **TAG** | Topology Adaptive Graph Convolution — explicit K-hop message passing |
| **ARMA** | Auto-Regressive Moving Average filters with residual stacks |
| **Cheb** | Chebyshev spectral graph convolution (K-hop polynomial filtering) |
| **SuperGAT** | Self-supervised attention via link prediction |

Architecture configuration:

```python
from dta_gnn.models import GnnTrainConfig

config = GnnTrainConfig(
    architecture="gin",        # see table above
    embedding_dim=128,         # atom embedding dimension
    hidden_dim=256,            # hidden layer dimension
    num_layers=5,              # number of message-passing layers
    dropout=0.1,
    pooling="attention",       # add | mean | max | attention
    residual=True,
    head_mlp_layers=2,
    # Architecture-specific (only the relevant fields are used at runtime):
    gin_conv_mlp_layers=2, gin_train_eps=False, gin_eps=0.0,
    gat_heads=4,
    sage_aggr="mean",
    transformer_heads=4,
    tag_k=2,
    arma_num_stacks=1, arma_num_layers=1,
    cheb_k=2,
    supergat_heads=4, supergat_attention_type="MX",
    lr=1e-3,
    batch_size=64,
    epochs=100,
)
```

> **Note** — `TransformerConv` does not currently support Apple-Silicon MPS; DTA-GNN automatically falls back to CPU for the `transformer` architecture on MPS.

---

## 🔬 Molecular Graph Representation

```
SMILES → atoms (nodes) + bonds (edges) → GNN → pChEMBL prediction
```

**Atom features (6-D):** atomic number, total degree, formal charge, total H count, aromaticity, atomic mass.

**Bond features (6-D):** is single / double / triple / aromatic, conjugation, in-ring.

```python
from dta_gnn.features.molecule_graphs import smiles_to_graph_2d

g = smiles_to_graph_2d(
    molecule_chembl_id="aspirin",
    smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
)
print(f"Atoms: {len(g.atom_type)}, Bonds: {g.edge_index.shape[1] // 2}")
```

---

## 📂 Project Structure

```
DTA-GNN/
├── src/dta_gnn/
│   ├── cli.py                # CLI entry point (audit, setup, ui, train-gnn)
│   ├── pipeline.py           # Pipeline.build_dta — dataset assembly
│   ├── training/             # End-to-end orchestration (run_gnn_end_to_end)
│   ├── io/                   # ChEMBL sources (web / sqlite), downloader,
│   │                         # UniProt→ChEMBL target mapping, run dirs
│   ├── cleaning/             # Activity standardisation, deduplication
│   ├── splits/               # Random / scaffold / temporal split strategies
│   ├── features/             # Morgan fingerprints, 2-D molecule graphs
│   ├── models/               # RF, SVR, GNN training + W&B/Optuna HPO + predict
│   ├── audits/               # Scaffold and target leakage audits
│   ├── exporters/            # Run-dir artifact collection, ZIP, dataset cards
│   ├── visualization.py      # Activity dist, split sizes, chemical-space plots
│   ├── app/                  # Gradio Web UI
│   └── app_features/         # Helpers used by the Web UI
├── docs/                     # MkDocs-Material site (see mkdocs.yml)
├── examples/                 # Notebooks + setup scripts
├── tests/                    # pytest test suite
├── pyproject.toml            # Package metadata and dependencies
└── docker-compose.yml        # ui + jupyter services
```

A typical run directory looks like:

```
runs/20260309_142301/
├── dataset.csv                       # Full dataset with split column
├── compounds.csv                     # molecule_chembl_id, smiles
├── metadata.json                     # Pipeline metadata
├── model_rf.pkl                      # RF baseline (joblib)
├── model_svr.pkl                     # SVR baseline (joblib)
├── model_gnn_<arch>.pt               # Trained GNN weights
├── encoder_<arch>.pt                 # GNN encoder (for embedding extraction)
├── encoder_<arch>_config.json
├── model_metrics_gnn_<arch>.json
├── model_predictions_gnn_<arch>.csv
└── molecule_embeddings.npz           # (after extract_gnn_embeddings_on_run)
```

---

## 📊 Example: full workflow with leakage audit

```python
from dta_gnn.io.runs import create_run_dir
from dta_gnn.pipeline import Pipeline
from dta_gnn.audits import audit_scaffold_leakage
from dta_gnn.models import (
    train_gnn_on_run,
    extract_gnn_embeddings_on_run,
    GnnTrainConfig,
)

# 1. Create run directory
run_dir = create_run_dir()

# 2. Build dataset for several kinase targets
pipeline = Pipeline(source_type="sqlite", sqlite_path="./chembl_dbs/chembl_36.db")
df = pipeline.build_dta(
    target_ids=["CHEMBL1862", "CHEMBL2111", "CHEMBL3778"],
    split_method="scaffold",
    output_path=str(run_dir / "dataset.csv"),
)
df[["molecule_chembl_id", "smiles"]].drop_duplicates().to_csv(
    run_dir / "compounds.csv", index=False
)
print(f"Dataset: {len(df)} drug–target pairs")

# 3. Verify there is no scaffold leakage between train and test
train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]
audit = audit_scaffold_leakage(train, test)
print(f"Scaffold leakage: {audit['leakage_ratio']:.1%}")  # expect 0.0%

# 4. Train a GNN
gnn = train_gnn_on_run(run_dir, config=GnnTrainConfig(
    architecture="gin",
    hidden_dim=256,
    num_layers=5,
    pooling="attention",
    epochs=100,
))
for split in ("train", "val", "test"):
    print(f"{split:5s} RMSE: {gnn.metrics['splits'][split]['rmse']:.3f}")

# 5. Extract molecular embeddings for downstream tasks
emb = extract_gnn_embeddings_on_run(run_dir)
print(f"Extracted {emb.n_molecules} embeddings of dim {emb.embedding_dim}")
```

---

## 👥 Who is this for?

| You are… | You want to… | DTA-GNN gives you… |
|------------|----------------|---------------------|
| Drug-discovery researcher | Predict affinity for your target | End-to-end pipeline with baselines and GNNs |
| ML researcher | Benchmark GNN architectures | Leakage-free datasets + 2 baselines + 10 GNNs |
| Computational chemist | Screen compounds virtually | Trained models, predictions, and embeddings |

---

## 📖 Documentation

- [Installation](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quickstart.md)
- [Data Sources (web vs. SQLite)](docs/user-guide/data-sources.md)
- [Target Mapping (UniProt → ChEMBL)](docs/user-guide/target-mapping.md)
- [Cleaning](docs/user-guide/cleaning.md) · [Splits](docs/user-guide/splits.md) · [Leakage Audits](docs/user-guide/audits.md) · [Visualization](docs/user-guide/visualization.md)
- [Featurisation](docs/modeling/features.md) · [Training Models](docs/modeling/models.md) · [End-to-End Pipeline](docs/modeling/end-to-end.md)
- [Hyperparameter Optimisation](docs/hpo/hyperopt.md)
- Interfaces: [CLI](docs/interfaces/cli.md) · [Python API](docs/interfaces/python-api.md) · [Web UI](docs/interfaces/ui.md)
- [Contributing](docs/development/contributing.md)

Build the docs locally:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve   # http://127.0.0.1:8000
```

---

## 🧪 Testing

```bash
pytest                       # full test suite
pytest -k "splits"           # run a subset
pytest --cov=dta_gnn         # with coverage (requires pytest-cov)
```

---

## 🛠️ Troubleshooting (top hits)

| Symptom | Likely cause / fix |
|---------|--------------------|
| `FileNotFoundError: Missing dataset.csv` when training a baseline | You called `train_*_on_run("runs/current", …)` without first writing `dataset.csv`. Build a dataset with `pipeline.build_dta(..., output_path=str(run_dir / "dataset.csv"))` and save `compounds.csv` before training. |
| `ValueError: SQLite DB not found` | Pass `sqlite_path` to `Pipeline` (or `--sqlite-path` to the CLI) pointing at the `.db` file extracted by `dta_gnn setup`. |
| `ValueError: Invalid UniProt accession(s)` | UniProt regex-validates input. Check that you're passing accessions like `P00533`, not gene symbols. |
| ChEMBL web API HTTP 500s | Transient EBI outage — retry, or switch to a local SQLite database. |
| `TransformerConv` warning falling back to CPU on Mac | `transformer` architecture is not yet supported on MPS; CPU fallback is automatic. |
| W&B prompts for login during HPO | `export WANDB_API_KEY=…` or pass `wandb_api_key=...`; or run with `WANDB_MODE=offline` to skip cloud logging. |

More detail in each subsection of [the docs](docs/index.md).

---

## 📄 License

Released under the [MIT License](LICENSE).

---

## 📚 Citation

If you use DTA-GNN in your research, please cite:

```bibtex
@article{ozsari2026dta,
  title   = {DTA-GNN: a toolkit for constructing target-specific drug--target affinity datasets and training graph neural networks},
  author  = {Özsari, Gökhan and Rifaioğlu, Ahmet Süreyya and Acar, Aybar Can and Doğan, Tunca and Atalay, M Volkan},
  journal = {SoftwareX},
  volume  = {34},
  pages   = {102671},
  year    = {2026},
  publisher = {Elsevier}
}
```
