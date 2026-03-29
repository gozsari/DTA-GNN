# Supplementary Material

This directory contains all supplementary material for the DTA-GNN multi-target
benchmarking study: the written supplement, the codes, and
pre-computed artifacts for both classical baselines and GNN experiments.

## Directory structure

```
supplementary/
├── Supplementary_Material_Multi_Target_Benchmarking.md
├── codes/
│   ├── 01_build_dataset.py
│   ├── 02_train_baselines.py
│   └── 03_train_gnn.py
├── P10275/                          # Androgen Receptor (AR)
│   ├── dataset/
│   │   ├── compounds.csv
│   │   ├── dataset.csv
│   │   └── metadata.json
│   ├── baseline_artifacts/
│   │   ├── baseline_metrics.json
│   │   ├── baseline_runtime_profile.json
│   │   ├── model_metrics.json
│   │   ├── model_metrics_svr.json
│   │   ├── model_predictions.csv
│   │   └── model_predictions_svr.csv
│   └── gnn_artifacts/
│       ├── paper_artifacts_gcn/
│       ├── paper_artifacts_gin/
│       ├── paper_artifacts_gat/
│       ├── paper_artifacts_sage/
│       └── paper_artifacts_pna/
├── P23303/                          # Acetylcholinesterase (AChE)
│   └── ... (same structure)
└── P35354/                          # COX-2
    └── ... (same structure)
```

---

## 1. Supplementary manuscript

**`Supplementary_Material_Multi_Target_Benchmarking.md`**

Full supplementary text including:

- **S0** -- Experimental settings (dataset construction, baseline configuration, GNN HPO setup)
- **S1** -- Multi-target benchmarking: per-target performance tables (S2--S4), cross-target summary (S5), optimized hyperparameter tables (S6--S8)
- **S2** -- Wall-clock runtimes and memory footprints for GNN training (S9--S11) and classical baselines (S12)

---

## 2. `codes/`

The scripts were used to produce the paper results.
The active (potentially updated) versions live at the project root.

| File | Description |
|:-----|:------------|
| `01_build_dataset.py` | Builds DTA datasets from ChEMBL (random + scaffold splits), runs scaffold leakage audit, saves dataset artifacts. |
| `02_train_baselines.py` | Trains Random Forest and SVR baselines on Morgan fingerprints, writes metrics and predictions. |
| `03_train_gnn.py` | Runs W&B Bayesian HPO, trains the best GNN model, generates evaluation tables and figures. |

These copies are provided so that results can be reproduced exactly as reported,
independent of any subsequent changes to the main codebase.

---

## 3. Per-target directories

Each target (`P10275/`, `P23303/`, `P35354/`) has three subdirectories
containing the full dataset, baseline results, and GNN experiment outputs.

### Targets

| Directory | UniProt ID | Target |
|:----------|:----------:|:-------|
| `P10275/` | P10275 | Androgen Receptor (AR) |
| `P23303/` | P23303 | Acetylcholinesterase (AChE) |
| `P35354/` | P35354 | Prostaglandin-Endoperoxide Synthase 2 (COX-2) |

### `dataset/`

| File | Description |
|:-----|:------------|
| `dataset.csv` | Full dataset with `molecule_chembl_id`, `label`, and `split` columns. |
| `compounds.csv` | Compound metadata including SMILES strings. |
| `metadata.json` | Run metadata (UniProt ID, ChEMBL target IDs, split method, sample counts). |

### `baseline_artifacts/`

| File | Description |
|:-----|:------------|
| `baseline_metrics.json` | Consolidated RF + SVR metrics (RMSE, MAE, R², Pearson r, Spearman rho) per split. |
| `baseline_runtime_profile.json` | Wall-clock time and RAM delta for each model. |
| `model_metrics.json` | RF-only metrics per split. |
| `model_metrics_svr.json` | SVR-only metrics per split. |
| `model_predictions.csv` | RF predictions on val + test splits (includes `y_pred` column). |
| `model_predictions_svr.csv` | SVR predictions on val + test splits. |

### `gnn_artifacts/`

Five architecture subdirectories: `paper_artifacts_gcn/`, `paper_artifacts_gin/`,
`paper_artifacts_gat/`, `paper_artifacts_sage/`, `paper_artifacts_pna/`.

Each contains:

| File | Description |
|:-----|:------------|
| `best_hyperparameters.json` | Optimized hyperparameters from W&B Bayesian sweeps. |
| `table_performance.csv` | Val and test metrics (RMSE, MAE, R², Pearson, Spearman). |
| `table_model_comparison.csv` | Side-by-side GNN vs. baseline comparison. |
| `table_hpo_best.csv` | Best HPO trial details (all hyperparameters). |
| `table_top5_molecules.csv` | Top-5 test compounds ranked by predicted affinity. |
| `table_inference.csv` | Inference results on provided SMILES. |
| `table_runtime_profile.csv` | Training time, per-epoch time, embedding time, GPU memory. |
| `runtime_profile.json` | Structured runtime and memory data. |
| `fig_predicted_vs_true.pdf` | Scatter plot of predicted vs. ground-truth affinity. |
| `fig_embeddings_pca_predicted.pdf` | PCA of GNN embeddings coloured by predicted value. |
| `fig_embeddings_pca_ground_truth.pdf` | PCA of GNN embeddings coloured by ground truth. |

---

## 4. Usage examples

### Loading baseline metrics

```python
import json

with open("supplementary/P10275/baseline_artifacts/baseline_metrics.json") as f:
    metrics = json.load(f)

print(metrics["RandomForest"]["test"]["pearson_r"])
print(metrics["SVR"]["test"]["rmse"])
```

### Loading predictions

```python
import pandas as pd

preds = pd.read_csv("supplementary/P10275/baseline_artifacts/model_predictions.csv")
print(preds[["molecule_chembl_id", "label", "y_pred", "split"]].head())
```

### Loading GNN hyperparameters

```python
import json

path = "supplementary/P10275/gnn_artifacts/paper_artifacts_pna/best_hyperparameters.json"
with open(path) as f:
    hparams = json.load(f)

print(hparams["architecture"])  # "pna"
print(hparams["num_layers"])    # 3
print(hparams["val_score"])     # -0.7098 (negated RMSE)
```

### Loading GNN performance table

```python
import pandas as pd

perf = pd.read_csv(
    "supplementary/P10275/gnn_artifacts/paper_artifacts_pna/table_performance.csv"
)
print(perf)
```
