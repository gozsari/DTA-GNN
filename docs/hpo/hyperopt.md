# Hyperparameter Optimization

DTA-GNN includes integrated hyperparameter optimisation (HPO) for all of its
models. The backend is **Weights & Biases Bayesian sweeps**. To run without
sending data to the W&B cloud, set `WANDB_MODE=offline`.

## Overview

HPO automatically searches for optimal hyperparameters:

| Model | Tunable parameters |
|-------|-------------------|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| SVR (regression only) | `C`, `epsilon`, `kernel` |
| GNN | learning rate, `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`, `pooling`, `residual`, `head_mlp_layers`, `epochs`, `batch_size`, plus architecture-specific knobs |

The validation strategy is automatic:

- If the run's `dataset.csv` contains a `val` split, **holdout** validation is
  used (train on `train`, score on `val`).
- Otherwise the optimiser falls back to **5-fold KFold** cross-validation on
  the non-test rows, using R² as the score (regression).

The `test` split is always reserved and is **never** used during HPO.

## Installation

W&B is included in the default install; no extra step is required. Optional
Optuna fallback also ships with the package.

## Quick Start

### Via Web UI

1. Build a dataset in the **Dataset Builder** tab.
2. Open the **HPO** tab.
3. Pick the model type (RandomForest, SVR, or GNN) and the parameters to optimise.
4. Set the number of trials and your W&B project name (and API key if not logged in).
5. Click **Run Hyperparameter Optimization**, then download `hyperopt_best_params_…json`.

### Via Python API

```python
from dta_gnn.models.hyperopt import HyperoptConfig, optimize_random_forest_wandb

config = HyperoptConfig(
    model_type="RandomForest",
    n_trials=20,
    rf_optimize_n_estimators=True, rf_n_estimators_min=50,  rf_n_estimators_max=500,
    rf_optimize_max_depth=True,    rf_max_depth_min=5,      rf_max_depth_max=50,
)

result = optimize_random_forest_wandb(
    "runs/current",
    config=config,
    project="my-chembl-project",
)

print("Best params:", result.best_params)
print("Best score :", result.best_value)
```

## HyperoptConfig Reference

`HyperoptConfig` (in `dta_gnn.models.hyperopt`) is a single dataclass
covering all three model types. Only the fields relevant to your `model_type`
are read; **unused fields are ignored**.

> Each tunable parameter has an **`*_optimize_*` boolean** flag. The sweep
> only varies parameters whose flag is `True`. If no flag is `True`, HPO
> raises `ValueError("No parameters selected for optimization.")`.

```python
from dta_gnn.models.hyperopt import HyperoptConfig

config = HyperoptConfig(
    # General
    model_type="GNN",          # "RandomForest" | "SVR" | "GNN"
    n_trials=20,
    n_jobs=1,                  # parallel jobs (RF only)
    sampler_seed=42,

    # ---------- RandomForest knobs ----------
    rf_optimize_n_estimators=True, rf_n_estimators_min=50,  rf_n_estimators_max=500,
    rf_optimize_max_depth=True,    rf_max_depth_min=5,      rf_max_depth_max=50,
    rf_optimize_min_samples_split=True,
        rf_min_samples_split_min=2, rf_min_samples_split_max=20,

    # ---------- SVR knobs (regression only) ----------
    svr_optimize_C=True,        svr_C_min=0.1,  svr_C_max=100.0,  svr_C_default=10.0,
    svr_optimize_epsilon=True,  svr_epsilon_min=0.01, svr_epsilon_max=0.2, svr_epsilon_default=0.1,
    svr_optimize_kernel=True,   svr_kernel_choices=["rbf", "linear"], svr_kernel_default="rbf",

    # ---------- GNN knobs ----------
    architecture="gin",         # gin | gcn | gat | sage | pna | transformer | tag | arma | cheb | supergat

    optimize_lr=True,           lr_min=1e-5,  lr_max=1e-2,
    optimize_epochs=True,       epochs_min=5,  epochs_max=50,  epochs_default=20,
    optimize_batch_size=True,   batch_size_min=16, batch_size_max=256, batch_size_default=64,
    optimize_embedding_dim=True, embedding_dim_min=32, embedding_dim_max=512, embedding_dim_default=128,
    optimize_hidden_dim=True,    hidden_dim_min=32,    hidden_dim_max=512,    hidden_dim_default=128,
    optimize_num_layers=True,    num_layers_min=1,     num_layers_max=5,      num_layers_default=3,
    optimize_dropout=True,       dropout_min=0.0,      dropout_max=0.6,       dropout_default=0.1,
    optimize_pooling=True,       pooling_choices=["add","mean","max","attention"], pooling_default="add",
    optimize_residual=True,      residual_default=False,
    optimize_head_mlp_layers=True, head_mlp_layers_min=1, head_mlp_layers_max=4, head_mlp_layers_default=2,

    # GIN-specific
    optimize_gin_conv_mlp_layers=True,
        gin_conv_mlp_layers_min=1, gin_conv_mlp_layers_max=4, gin_conv_mlp_layers_default=2,
    optimize_gin_train_eps=True, gin_train_eps_default=False,
    optimize_gin_eps=True,       gin_eps_min=0.0, gin_eps_max=1.0, gin_eps_default=0.0,

    # GAT-specific
    optimize_gat_heads=True, gat_heads_min=1, gat_heads_max=8, gat_heads_default=4,

    # GraphSAGE-specific
    optimize_sage_aggr=True, sage_aggr_choices=["mean","max","lstm","pool"], sage_aggr_default="mean",

    # Transformer-specific
    optimize_transformer_heads=True,
        transformer_heads_min=1, transformer_heads_max=8, transformer_heads_default=4,

    # TAG-specific
    optimize_tag_k=True, tag_k_min=1, tag_k_max=5, tag_k_default=2,

    # ARMA-specific
    optimize_arma_stacks=True,
        arma_num_stacks_min=1, arma_num_stacks_max=3, arma_num_stacks_default=1,
    optimize_arma_layers=True,
        arma_num_layers_min=1, arma_num_layers_max=3, arma_num_layers_default=1,

    # Cheb-specific
    optimize_cheb_k=True, cheb_k_min=1, cheb_k_max=5, cheb_k_default=2,

    # SuperGAT-specific
    optimize_supergat_heads=True,
        supergat_heads_min=1, supergat_heads_max=8, supergat_heads_default=4,
    optimize_supergat_attention_type=True,
        supergat_attention_type_choices=["MX","SD"],
        supergat_attention_type_default="MX",
)
```

## Model-Specific Sweeps

### Random Forest

```python
from dta_gnn.models.hyperopt import optimize_random_forest_wandb, HyperoptConfig

result = optimize_random_forest_wandb(
    run_dir="runs/current",
    config=HyperoptConfig(
        model_type="RandomForest", n_trials=20,
        rf_optimize_n_estimators=True,
        rf_optimize_max_depth=True,
    ),
    project="chembl-rf-sweep",
    entity=None,         # optional W&B team
    api_key=None,        # optional; uses logged-in session by default
    sweep_name="rf_sweep_1",
    radius=2,            # Morgan FP radius
    n_bits=2048,         # Morgan FP bits
)
```

| Parameter | Default range | Description |
|-----------|--------------|-------------|
| `n_estimators` | 50–500 | Number of trees |
| `max_depth` | 5–50 | Maximum tree depth |
| `min_samples_split` | 2–20 | Minimum samples per split |

### SVR (regression only)

```python
from dta_gnn.models.hyperopt import optimize_svr_wandb, HyperoptConfig

config = HyperoptConfig(
    model_type="SVR",
    n_trials=20,
    svr_optimize_C=True,        svr_C_min=0.1,  svr_C_max=100.0,
    svr_optimize_epsilon=True,  svr_epsilon_min=0.01, svr_epsilon_max=0.2,
    svr_optimize_kernel=True,   svr_kernel_choices=["rbf", "linear"],
)

result = optimize_svr_wandb("runs/current", config=config, project="chembl-svr-sweep")
print("Best:", result.best_params, "score:", result.best_value)
```

| Parameter | Default range | Description |
|-----------|---------------|-------------|
| `C` | 0.1–100 (log) | Regularisation |
| `epsilon` | 0.01–0.2 (log) | Epsilon-tube width |
| `kernel` | `rbf`, `linear` | Kernel function |

> SVR sweeps require a regression dataset; classification runs raise
> `ValueError`.

### GNN

```python
from dta_gnn.models.hyperopt import optimize_gnn_wandb, HyperoptConfig

config = HyperoptConfig(
    model_type="GNN",
    n_trials=20,
    architecture="gin",
    optimize_lr=True,
    optimize_dropout=True,
    optimize_pooling=True,
)

result = optimize_gnn_wandb("runs/current", config=config, project="chembl-gnn-sweep")
print("Best:", result.best_params, "score:", result.best_value)
```

Architecture-specific tunables:

| Architecture | Parameter | Range |
|--------------|-----------|-------|
| GIN | `gin_conv_mlp_layers` / `gin_train_eps` / `gin_eps` | 1–4 / bool / 0.0–1.0 |
| GAT | `gat_heads` | 1–8 |
| GraphSAGE | `sage_aggr` | `mean`/`max`/`lstm`/`pool` |
| Transformer | `transformer_heads` | 1–8 |
| TAG | `tag_k` | 1–5 |
| ARMA | `arma_num_stacks` / `arma_num_layers` | 1–3 / 1–3 |
| Cheb | `cheb_k` | 1–5 |
| SuperGAT | `supergat_heads` / `supergat_attention_type` | 1–8 / `MX` or `SD` |

## Validation Strategy

If `dataset.csv` contains a `val` split:

```text
train  → fit
val    → score (R²)
test   → reserved, untouched
```

Otherwise:

```text
non-test rows → 5-fold KFold (R²)
test          → reserved, untouched
```

(For regression. The CV is plain `KFold`, **not** stratified.)

## HyperoptResult

```python
@dataclass
class HyperoptResult:
    run_dir: Path
    best_params: dict       # winning hyperparameters
    best_value: float       # best metric (validation R² for GNN/SVR/RF)
    best_trial_number: int  # 0-indexed winning trial
    n_trials: int
    study_path: str         # W&B sweep ID (or Optuna study path)
    best_params_path: str   # JSON file with best_params
    strategy: str           # "holdout-val" or "cv"
    cv_folds_used: int | None  # only set when strategy == "cv"
```

A `hyperopt_best_params_*.json` file is written into the run directory:

```json
{
  "n_estimators": 342,
  "max_depth": 28,
  "min_samples_split": 5,
  "radius": 2,
  "n_bits": 2048,
  "task_type": "regression"
}
```

## Weights & Biases Integration

```python
import wandb

# Option 1: log in interactively
wandb.login()

# Option 2: environment variable
# export WANDB_API_KEY=<your-key>

# Option 3: pass to the sweep function
optimize_random_forest_wandb(..., api_key="<your-key>")
```

Open the corresponding W&B project to inspect parameter importance,
parallel-coordinates plots, and per-trial training curves.

To run **without** W&B (offline):

```bash
WANDB_MODE=offline dta_gnn train-gnn P00533
```

## Aliases

For backward compatibility, the package exposes the following aliases:

| Alias | Resolves to |
|-------|-------------|
| `optimize_random_forest` | `optimize_random_forest_wandb` |
| `optimize_gnn` | `optimize_gnn_wandb` |
| `optimize_gin` | `optimize_gnn_wandb` |

They behave identically to the canonical names — there is no separate Optuna
backend at this time.

## Running offline (no W&B cloud)

```bash
WANDB_MODE=offline dta_gnn train-gnn P00533
```

Or in Python:

```python
import os
os.environ["WANDB_MODE"] = "offline"

from dta_gnn.models.hyperopt import optimize_gnn_wandb, HyperoptConfig
result = optimize_gnn_wandb(
    "runs/current",
    config=HyperoptConfig(model_type="GNN", n_trials=10, optimize_lr=True),
    project="local-only",
)
```

## Best Practices

| Dataset size | Recommended trials |
|--------------|--------------------|
| < 1 000 | 10–20 |
| 1 000–10 000 | 20–50 |
| > 10 000 | 50–100 |

- Start broad (enable many `*_optimize_*` flags), then narrow.
- Widen ranges if the best value lands at a boundary.
- Smaller `epochs_max` and a larger `batch_size_min` make GNN sweeps cheaper.

## Troubleshooting

**`ValueError: No parameters selected for optimization.`**
At least one `*_optimize_*` flag must be `True`.

**All trials report the same score.**
Widen the search ranges, increase the dataset, or sanity-check the labels.

**W&B connection issues.**
`wandb.login(verify=True)` to test, or run with `WANDB_MODE=offline`.

**CUDA OOM during GNN sweeps.**
Reduce `batch_size_max`, `hidden_dim_max`, and/or `embedding_dim_max`.

## Complete Example

```python
from dta_gnn.io.runs import create_run_dir
from dta_gnn.pipeline import Pipeline
from dta_gnn.models.hyperopt import HyperoptConfig, optimize_gnn_wandb
from dta_gnn.models import GnnTrainConfig, train_gnn_on_run

# 1. Build a dataset that includes a val split
run_dir = create_run_dir()
pipeline = Pipeline(source_type="sqlite", sqlite_path="./chembl_dbs/chembl_36.db")
df = pipeline.build_dta(
    target_ids=["CHEMBL204"],
    split_method="scaffold",
    val_size=0.1,
    output_path=str(run_dir / "dataset.csv"),
)
df[["molecule_chembl_id", "smiles"]].drop_duplicates().to_csv(
    run_dir / "compounds.csv", index=False,
)

# 2. Sweep
hpo = optimize_gnn_wandb(
    run_dir,
    config=HyperoptConfig(
        model_type="GNN", n_trials=20,
        architecture="gin",
        optimize_lr=True,
        optimize_dropout=True,
    ),
    project="chembl-hpo",
)
print("Best params:", hpo.best_params)

# 3. Final training with the best params
final_cfg = GnnTrainConfig(
    architecture="gin",
    lr=float(hpo.best_params.get("lr", 1e-3)),
    dropout=float(hpo.best_params.get("dropout", 0.1)),
    epochs=int(hpo.best_params.get("epochs", 30)),
)
final = train_gnn_on_run(run_dir, config=final_cfg)
print("Test:", final.metrics["splits"]["test"])
```
