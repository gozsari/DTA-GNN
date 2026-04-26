<div align="center">

<img src="https://raw.githubusercontent.com/gozsari/DTA-GNN/main/assets/logo3.png" alt="DTA-GNN Logo" width="400"/>

# DTA-GNN: Target-Specific Binding Affinity Dataset Builder and GNN Trainer

**Build leakage-free Drug-Target Affinity datasets from ChEMBL and train Graph Neural Networks for any target of interest.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/dta-gnn/badge/?version=latest)](https://dta-gnn.readthedocs.io/en/latest/)

[Documentation](https://dta-gnn.readthedocs.io/en/latest/) · [Quick Start](#-quick-start) · [Python API](https://dta-gnn.readthedocs.io/en/latest/interfaces/python-api/) · [CLI](https://dta-gnn.readthedocs.io/en/latest/interfaces/cli/)

</div>

---

## Overview

DTA-GNN is an end-to-end toolkit for Drug-Target Affinity (DTA) prediction:

- **Curates** clean, leakage-free regression datasets from ChEMBL (web API or local SQLite)
- **Featurises** molecules as Morgan fingerprints (ECFP4) or 2D molecular graphs ([details](https://dta-gnn.readthedocs.io/en/latest/modeling/features/))
- **Trains** baseline models (Random Forest, SVR) and [10 GNN architectures](https://dta-gnn.readthedocs.io/en/latest/modeling/models/) (GIN, GCN, GAT, GraphSAGE, PNA, Transformer, TAG, ARMA, Cheb, SuperGAT)
- **Evaluates** with scaffold-aware splitting and built-in [leakage audits](https://dta-gnn.readthedocs.io/en/latest/user-guide/audits/)
- **Tracks** hyperparameter search and training in [Weights & Biases](https://dta-gnn.readthedocs.io/en/latest/hpo/hyperopt/)
- **Offers** three interfaces: [CLI](https://dta-gnn.readthedocs.io/en/latest/interfaces/cli/), [Python API](https://dta-gnn.readthedocs.io/en/latest/interfaces/python-api/), and [Web UI](https://dta-gnn.readthedocs.io/en/latest/interfaces/ui/)

<div align="center">

![DTA-GNN Overview](https://raw.githubusercontent.com/gozsari/DTA-GNN/main/assets/overview.png)

</div>

---

## Installation

```bash
pip install dta-gnn
```

Or from source:

```bash
git clone https://github.com/gozsari/DTA-GNN.git
cd DTA-GNN
pip install -e .          # editable install
pip install -e ".[dev]"   # with dev tools (pytest, ruff, black)
```

Docker images are also available. See the [Installation Guide](https://dta-gnn.readthedocs.io/en/latest/getting-started/installation/) for Docker, GPU/CUDA, and troubleshooting details.

---

## Quick Start

### CLI (one command, end-to-end)

```bash
dta_gnn train-gnn P00533 --architecture gin --n-trials 20 --epochs 30
```

Resolves the UniProt accession, builds a scaffold-split dataset, runs a W&B hyperparameter sweep, trains the final GNN, and reports test metrics. All artifacts land in `runs/<TIMESTAMP>/`.

### Python API (one call)

```python
from dta_gnn.training import run_gnn_end_to_end, EndToEndConfig

result = run_gnn_end_to_end(EndToEndConfig(
    uniprot_ids="P00533",
    architecture="gin",
    n_trials=20,
    epochs=30,
))

print(result.test_metrics)   # {"rmse": ..., "mae": ..., "r2": ...}
print(result.run_dir)        # Path("runs/20260309_142301")
```

For step-by-step control (build dataset, train baselines, train GNN separately), see the [Quick Start guide](https://dta-gnn.readthedocs.io/en/latest/getting-started/quickstart/).

---

## Web UI

DTA-GNN ships with an interactive Gradio interface.

### Live demos (limited compute, may be slow)

| Platform | URL |
|----------|-----|
| Hugging Face Spaces | <https://huggingface.co/spaces/gozsari/dta-gnn> |
| SciLifeLab Serve | <https://dta-gnn.serve.scilifelab.se/> |

### Launch locally

```bash
dta_gnn ui                  # http://127.0.0.1:7860
dta_gnn ui --host 0.0.0.0   # bind to all interfaces (Docker / remote)
```

See the [Web UI guide](https://dta-gnn.readthedocs.io/en/latest/interfaces/ui/) for full details.

---

## Who is this for?

| You are... | You want to... | DTA-GNN gives you... |
|------------|----------------|---------------------|
| Drug-discovery researcher | Predict affinity for your target | End-to-end pipeline with baselines and GNNs |
| ML researcher | Benchmark GNN architectures | Leakage-free datasets + 2 baselines + 10 GNNs |
| Computational chemist | Screen compounds virtually | Trained models, predictions, and embeddings |

---

## Documentation

Full documentation is hosted at **[dta-gnn.readthedocs.io](https://dta-gnn.readthedocs.io/en/latest/)**.

- [Installation](https://dta-gnn.readthedocs.io/en/latest/getting-started/installation/) · [Quick Start](https://dta-gnn.readthedocs.io/en/latest/getting-started/quickstart/)
- [Data Sources](https://dta-gnn.readthedocs.io/en/latest/user-guide/data-sources/) · [Target Mapping](https://dta-gnn.readthedocs.io/en/latest/user-guide/target-mapping/) · [Cleaning](https://dta-gnn.readthedocs.io/en/latest/user-guide/cleaning/) · [Splits](https://dta-gnn.readthedocs.io/en/latest/user-guide/splits/) · [Audits](https://dta-gnn.readthedocs.io/en/latest/user-guide/audits/)
- [Featurisation](https://dta-gnn.readthedocs.io/en/latest/modeling/features/) · [Training Models](https://dta-gnn.readthedocs.io/en/latest/modeling/models/) · [End-to-End Pipeline](https://dta-gnn.readthedocs.io/en/latest/modeling/end-to-end/) · [Hyperparameter Optimisation](https://dta-gnn.readthedocs.io/en/latest/hpo/hyperopt/)
- [CLI](https://dta-gnn.readthedocs.io/en/latest/interfaces/cli/) · [Python API](https://dta-gnn.readthedocs.io/en/latest/interfaces/python-api/) · [Web UI](https://dta-gnn.readthedocs.io/en/latest/interfaces/ui/)
- [API Reference](https://dta-gnn.readthedocs.io/en/latest/reference/python-api/) · [Contributing](https://dta-gnn.readthedocs.io/en/latest/development/contributing/)

---

## Testing

```bash
pytest                       # full test suite
pytest -k "splits"           # run a subset
pytest --cov=dta_gnn         # with coverage
```

---

## License

Released under the [MIT License](LICENSE).

---

## Citation

If you use DTA-GNN in your research, please cite:

```bibtex
@article{ozsari2026dta,
  title={DTA-GNN: a toolkit for constructing target-specific drug--target affinity datasets and training graph neural networks},
  author={Özsari, Gökhan and Rifaioğlu, Ahmet Süreyya and Acar, Aybar Can and Doğan, Tunca and Atalay, M Volkan},
  journal={SoftwareX},
  volume={34},
  pages={102671},
  year={2026},
  publisher={Elsevier}
}
```
