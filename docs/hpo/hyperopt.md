# Hyperparameter Optimization

DTA-GNN includes integrated hyperparameter optimization (HPO) using Weights & Biases Bayesian sweeps.

## Overview

HPO automatically searches for optimal model hyperparameters:

- **RandomForest**: n_estimators, max_depth, min_samples_split
- **SVR**: C, epsilon, kernel
- **GNN**: epochs, learning rate, batch size, architecture parameters

## Installation

W&B (Weights & Biases) is included in the default install; no extra install step is required for HPO.

## Quick Start

### Via Web UI

1. Build a dataset in the "Dataset Builder" tab
2. Go to "Model" tab → "Hyperparameter Optimization" section
3. Select model type (RandomForest, SVR, or GNN)
4. Check parameters to optimize
5. Set number of trials
6. Enter W&B project name (and API key if needed)
7. Click "Run Hyperparameter Optimization"
8. View results and download `best_params.json`

### Via Python API

```python
from dta_gnn.models.hyperopt import HyperoptConfig, optimize_random_forest_wandb

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
    project="my-chembl-project"
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_value}")
```

## Configuration

### HyperoptConfig

```python
from dta_gnn.models.hyperopt import HyperoptConfig

config = HyperoptConfig(
    # General
    model_type="RandomForest",  # RandomForest, SVR, GNN
    n_trials=20,                # Number of trials
    n_jobs=1,                   # Parallel jobs (RF only)
    sampler_seed=42,            # Reproducibility
    
    # RandomForest parameters
    rf_optimize_n_estimators=True,
    rf_n_estimators_min=50,
    rf_n_estimators_max=500,
    
    rf_optimize_max_depth=True,
    rf_max_depth_min=5,
    rf_max_depth_max=50,
    
    rf_optimize_min_samples_split=True,
    rf_min_samples_split_min=2,
    rf_min_samples_split_max=20,
    
    # SVR parameters (regression only)
    svr_optimize_C=True,
    svr_C_min=0.1,
    svr_C_max=100.0,
    
    svr_optimize_epsilon=True,
    svr_epsilon_min=0.01,
    svr_epsilon_max=0.2,
    
    svr_optimize_kernel=True,
    svr_kernel_choices=["rbf", "linear"],
    
    # GNN parameters (see src/dta_gnn/models/hyperopt.py HyperoptConfig)
    architecture="gin",  # gin, gcn, gat, sage, pna, transformer, tag, arma, cheb, supergat
    
    optimize_epochs=True,
    epochs_min=5,
    epochs_max=50,
    epochs_default=20,
    
    optimize_lr=True,
    lr_min=1e-5,
    lr_max=1e-2,
    
    optimize_batch_size=True,
    batch_size_min=16,
    batch_size_max=256,
    batch_size_default=64,
    
    optimize_embedding_dim=True,
    embedding_dim_min=32,
    embedding_dim_max=512,
    embedding_dim_default=128,
    
    optimize_hidden_dim=True,
    hidden_dim_min=32,
    hidden_dim_max=512,
    hidden_dim_default=128,
    
    optimize_num_layers=True,
    num_layers_min=1,
    num_layers_max=5,
    num_layers_default=3,
    
    optimize_dropout=True,
    dropout_min=0.0,
    dropout_max=0.6,
    dropout_default=0.1,
    
    optimize_pooling=True,
    pooling_choices=["add", "mean", "max", "attention"],
    pooling_default="add",
    
    optimize_residual=True,
    residual_default=False,
    
    optimize_head_mlp_layers=True,
    head_mlp_layers_min=1,
    head_mlp_layers_max=4,
    head_mlp_layers_default=2,
    
    # GIN-specific
    optimize_gin_conv_mlp_layers=True,
    gin_conv_mlp_layers_min=1,
    gin_conv_mlp_layers_max=4,
    gin_conv_mlp_layers_default=2,
    optimize_gin_train_eps=True,
    gin_train_eps_default=False,
    optimize_gin_eps=True,
    gin_eps_min=0.0,
    gin_eps_max=1.0,
    gin_eps_default=0.0,
    
    # GAT-specific
    optimize_gat_heads=True,
    gat_heads_min=1,
    gat_heads_max=8,
    gat_heads_default=4,
    
    # GraphSAGE-specific
    optimize_sage_aggr=True,
    sage_aggr_choices=["mean", "max", "lstm", "pool"],
    sage_aggr_default="mean",
    
    # Transformer-specific
    optimize_transformer_heads=True,
    transformer_heads_min=1,
    transformer_heads_max=8,
    transformer_heads_default=4,
    
    # TAG-specific
    optimize_tag_k=True,
    tag_k_min=1,
    tag_k_max=5,
    tag_k_default=2,
    
    # ARMA-specific
    optimize_arma_stacks=True,
    arma_num_stacks_min=1,
    arma_num_stacks_max=3,
    arma_num_stacks_default=1,
    optimize_arma_layers=True,
    arma_num_layers_min=1,
    arma_num_layers_max=3,
    arma_num_layers_default=1,
    
    # Cheb-specific
    optimize_cheb_k=True,
    cheb_k_min=1,
    cheb_k_max=5,
    cheb_k_default=2,
    
    # SuperGAT-specific
    optimize_supergat_heads=True,
    supergat_heads_min=1,
    supergat_heads_max=8,
    supergat_heads_default=4,
    optimize_supergat_attention_type=True,
    supergat_attention_type_choices=["MX", "SD"],
    supergat_attention_type_default="MX",
)
```

## Model-Specific Optimization

### Random Forest

```python
from dta_gnn.models.hyperopt import optimize_random_forest_wandb

result = optimize_random_forest_wandb(
    run_dir="runs/current",
    config=config,
    project="chembl-rf-sweep",
    entity="my-team",        # Optional W&B team
    api_key="xxx",           # Optional, uses logged-in session
    sweep_name="rf_sweep_1", # Optional sweep name
    radius=2,                # Morgan FP radius
    n_bits=2048              # Morgan FP bits
)
```

**Optimized parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `n_estimators` | 50-500 | Number of trees |
| `max_depth` | 5-50 | Maximum tree depth |
| `min_samples_split` | 2-20 | Min samples to split |

### SVR (Regression Only)

```python
from dta_gnn.models.hyperopt import optimize_svr_wandb, HyperoptConfig

config = HyperoptConfig(
    model_type="SVR",
    n_trials=20,
    svr_optimize_C=True,
    svr_C_min=0.1,
    svr_C_max=100.0,
    svr_optimize_epsilon=True,
    svr_epsilon_min=0.01,
    svr_epsilon_max=0.2,
    svr_optimize_kernel=True,
    svr_kernel_choices=["rbf", "linear"]
)

result = optimize_svr_wandb(
    run_dir="runs/current",
    config=config,
    project="chembl-svr-sweep"
)

print(f"Best params: {result.best_params}")
print(f"Best score: {result.best_value}")
```

**Optimized parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `C` | 0.1-100 | Regularization (log scale) |
| `epsilon` | 0.01-0.2 | Epsilon-tube width |
| `kernel` | rbf, linear | Kernel function |

**Configuration options:**

```python
config = HyperoptConfig(
    model_type="SVR",
    n_trials=20,
    
    # C parameter (regularization)
    svr_optimize_C=True,        # Enable C optimization
    svr_C_min=0.1,              # Minimum C value
    svr_C_max=100.0,            # Maximum C value
    
    # Epsilon parameter (margin width)
    svr_optimize_epsilon=True,  # Enable epsilon optimization
    svr_epsilon_min=0.01,       # Minimum epsilon
    svr_epsilon_max=0.2,        # Maximum epsilon
    
    # Kernel selection
    svr_optimize_kernel=True,   # Enable kernel optimization
    svr_kernel_choices=["rbf", "linear"]  # Kernel options
)
```

**Best practices for SVR HPO:**

- **C range**: Start with 0.1-100, widen if best value is at boundary
- **Epsilon**: Smaller values (0.01-0.1) for tighter fit, larger (0.1-0.5) for more tolerance
- **Kernel**: Include both "rbf" and "linear" to compare
- **Trials**: 20-50 trials recommended for SVR

### GNN

```python
from dta_gnn.models.hyperopt import optimize_gnn_wandb

config = HyperoptConfig(
    model_type="GNN",
    n_trials=20,
    gnn_architecture="gin",
    gnn_optimize_epochs=True,
    gnn_optimize_lr=True,
    gnn_optimize_dropout=True,
    gnn_optimize_pooling=True
)

result = optimize_gnn_wandb(
    run_dir="runs/current",
    config=config,
    project="chembl-gnn-sweep"
)
```

**Optimized parameters:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `epochs` | 5-50 | Training iterations |
| `lr` | 1e-5 to 1e-2 | Learning rate (log scale) |
| `batch_size` | 16-256 | Batch size |
| `embedding_dim` | 32-512 | Output dimension |
| `hidden_dim` | 32-512 | Hidden layer size |
| `dropout` | 0-0.6 | Dropout rate |
| `pooling` | add/mean/max/attention | Graph pooling |
| `residual` | True/False | Skip connections |
| `head_mlp_layers` | 1-4 | Prediction head depth |

**Architecture-specific parameters:**

| Architecture | Parameter | Range | Description |
|--------------|-----------|-------|-------------|
| **GIN** | `gin_conv_mlp_layers` | 1-4 | MLP depth in GIN convolution |
| **GIN** | `gin_train_eps` | True/False | Whether to learn epsilon parameter |
| **GIN** | `gin_eps` | 0.0-1.0 | Initial epsilon value |
| **GAT** | `gat_heads` | 1-8 | Number of attention heads |
| **GraphSAGE** | `sage_aggr` | mean/max/lstm/pool | Aggregation method |
| **Transformer** | `transformer_heads` | 1-8 | Number of attention heads |
| **TAG** | `tag_k` | 1-5 | K-hop message passing |
| **ARMA** | `arma_num_stacks` | 1-3 | Number of ARMA stacks |
| **ARMA** | `arma_num_layers` | 1-3 | Number of layers per stack |
| **Cheb** | `cheb_k` | 1-5 | K-hop spectral filtering |
| **SuperGAT** | `supergat_heads` | 1-8 | Number of attention heads |
| **SuperGAT** | `supergat_attention_type` | MX/SD | Attention type (Mixed/Self-Distillation) |

## Validation Strategy

HPO uses two validation strategies:

### Holdout Validation

If your dataset has an explicit validation split:

```python
# Dataset with train/val/test splits
df["split"].value_counts()
# train    7000
# val      1000
# test     2000
```

HPO trains on `train` and evaluates on `val`.

### Cross-Validation

If no validation split exists, HPO uses stratified K-fold CV:

```python
# RF: 5-fold stratified CV
# Metric: ROC-AUC (classification) or R² (regression)
```

## Results

### HyperoptResult

```python
@dataclass
class HyperoptResult:
    run_dir: Path          # Run directory
    best_params: dict      # Optimal parameters
    best_value: float      # Best metric value
    best_trial_number: int # Winning trial index
    n_trials: int          # Total trials run
    study_path: str        # W&B sweep ID
    best_params_path: str  # Path to JSON file
    strategy: str          # "holdout-val" or "cv"
    cv_folds_used: int     # Folds used (if CV)
```

### Output Files

```
runs/current/
├── hyperopt_best_params_wandb_rf_regression.json
├── hyperopt_best_params_wandb_gin.json
└── _wandb_gin_001_abc123/  # Trial subdirectories
    ├── dataset.csv
    ├── compounds.csv
    └── ...
```

### Best Params JSON

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

### Authentication

```python
import wandb

# Option 1: Login interactively
wandb.login()

# Option 2: Environment variable
# export WANDB_API_KEY=your-key

# Option 3: Pass to function
optimize_random_forest_wandb(..., api_key="your-key")
```

### Viewing Results

1. Go to [wandb.ai](https://wandb.ai)
2. Find your project
3. View sweep dashboard
4. Analyze parameter importance
5. Compare trials

### Sweep Dashboard Features

- Parameter importance plots
- Parallel coordinates plot
- Trial history
- Best run details

## Local Optimization (Optuna)

For offline optimization without W&B:

```python
from dta_gnn.models.hyperopt import optimize_random_forest

result = optimize_random_forest(
    run_dir="runs/current",
    config=config,
    cv_folds=5
)
```

Output: `hyperopt_study_rf.pkl` (Optuna study object)

```python
import pickle

with open("runs/current/hyperopt_study_rf.pkl", "rb") as f:
    study = pickle.load(f)

# Visualize with Optuna
import optuna.visualization as vis
vis.plot_optimization_history(study)
vis.plot_param_importances(study)
```

## Best Practices

### Number of Trials

| Dataset Size | Recommended Trials |
|--------------|-------------------|
| < 1000 | 10-20 |
| 1000-10000 | 20-50 |
| > 10000 | 50-100 |

### Parameter Selection

1. **Start broad**: Enable all relevant parameters
2. **Focus later**: Disable low-importance parameters
3. **Widen ranges**: If best values are at boundaries

### Speed vs Accuracy

```python
# Fast exploration (fewer trials, smaller ranges)
config = HyperoptConfig(
    n_trials=10,
    gnn_epochs_max=20,
    gnn_batch_size_min=64
)

# Thorough search (more trials, wider ranges)
config = HyperoptConfig(
    n_trials=50,
    gnn_epochs_max=100,
    gnn_batch_size_min=16
)
```

## Troubleshooting

### No parameters selected

```
ValueError: No parameters selected for optimization.
Enable at least one 'Optimize ...' checkbox before running a sweep.
```

Enable at least one parameter:

```python
config = HyperoptConfig(
    rf_optimize_n_estimators=True  # At least one!
)
```

### All trials have same score

- Widen parameter ranges
- Increase dataset size
- Check for data issues

### W&B connection issues

```python
# Check connection
import wandb
wandb.login(verify=True)

# Offline mode fallback
wandb.init(mode="offline")
```

### Out of memory (GNN)

- Reduce `gnn_batch_size_max`
- Reduce `gnn_hidden_dim_max`
- Use fewer trials

## Example: Complete HPO Workflow

```python
from dta_gnn.pipeline import Pipeline
from dta_gnn.models.hyperopt import HyperoptConfig, optimize_gnn_wandb
from dta_gnn.models.gnn import GnnTrainConfig, train_gnn_on_run
from dta_gnn.io.runs import create_run_dir

# 1. Build dataset
run_dir = create_run_dir()
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")
df = pipeline.build_dta(
    target_ids=["CHEMBL204"],
    split_method="scaffold",
    val_size=0.1,  # Important for HPO!
    output_path=str(run_dir / "dataset.csv")
)
df[["molecule_chembl_id", "smiles"]].drop_duplicates().to_csv(
    run_dir / "compounds.csv", index=False
)

# 2. Run HPO
hpo_config = HyperoptConfig(
    model_type="GNN",
    n_trials=20,
    gnn_architecture="gin",
    gnn_optimize_lr=True,
    gnn_optimize_epochs=True,
    gnn_optimize_dropout=True
)

hpo_result = optimize_gnn_wandb(
    run_dir,
    config=hpo_config,
    project="chembl-hpo"
)

print(f"Best params: {hpo_result.best_params}")

# 3. Train final model with best params
final_config = GnnTrainConfig(
    architecture="gin",
    lr=hpo_result.best_params.get("lr", 1e-3),
    epochs=hpo_result.best_params.get("epochs", 20),
    dropout=hpo_result.best_params.get("dropout", 0.1)
)

final_result = train_gnn_on_run(run_dir, config=final_config)
print(f"Final test metrics: {final_result.metrics['splits']['test']}")
```
