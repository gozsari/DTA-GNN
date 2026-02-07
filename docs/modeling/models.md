# Training Models

DTA-GNN provides built-in model training for both classical machine learning and deep learning approaches.

## Overview

| Model | Features | Task Type | Speed | Accuracy |
|-------|----------|-----------|-------|----------|
| Random Forest | Morgan FP | Regression | Fast | Good |
| SVR | Morgan FP | Regression | Medium | Good |
| GNN | 2D Graphs | Regression | Slow | Best |

## Random Forest

A strong baseline using Morgan fingerprints (ECFP4).

### Training

```python
from dta_gnn.models import train_random_forest_on_run

result = train_random_forest_on_run(
    run_dir="runs/current",
    n_estimators=500,
    random_seed=42
)

print(f"Task: {result.task_type}")
print(f"Model: {result.model_path}")
print(f"Metrics: {result.metrics}")
```

### Required Files

The run directory must contain:

```
runs/current/
├── dataset.csv      # Main dataset with labels and splits
└── compounds.csv    # Molecules with SMILES
```

### Output Files

```
runs/current/
├── model_rf.pkl           # Trained model (joblib)
├── model_metrics.json     # Evaluation metrics
└── model_predictions.csv  # Predictions on val/test
```

### Metrics

**Regression:**

```json
{
  "model_type": "RandomForest",
  "task_type": "regression",
  "splits": {
    "train": {"rmse": 0.45, "mae": 0.32, "r2": 0.92},
    "val": {"rmse": 0.78, "mae": 0.58, "r2": 0.75},
    "test": {"rmse": 0.82, "mae": 0.61, "r2": 0.72}
  }
}
```

### Loading Trained Model

```python
import joblib

model = joblib.load("runs/current/model_rf.pkl")

# Make predictions
X_new = ...  # Morgan fingerprints (numpy array)
predictions = model.predict(X_new)
```

## SVR (Support Vector Regression)

Support Vector Regression using Morgan fingerprints (ECFP4). Suitable for regression tasks with non-linear relationships.

### Training

```python
from dta_gnn.models import train_svr_on_run

result = train_svr_on_run(
    run_dir="runs/current",
    C=10.0,              # Regularization parameter
    epsilon=0.1,         # Epsilon-tube width
    kernel="rbf",        # rbf or linear
    random_seed=42
)

print(f"Task: {result.task_type}")
print(f"Model: {result.model_path}")
print(f"Metrics: {result.metrics}")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | 10.0 | Regularization parameter (higher = less regularization) |
| `epsilon` | 0.1 | Margin of tolerance for errors |
| `kernel` | "rbf" | Kernel type: "rbf" or "linear" |

### Required Files

The run directory must contain:

```
runs/current/
├── dataset.csv      # Main dataset with labels and splits
└── compounds.csv    # Molecules with SMILES
```

### Output Files

```
runs/current/
├── model_svr.pkl              # Trained model (joblib)
├── model_metrics_svr.json     # Evaluation metrics
└── model_predictions_svr.csv  # Predictions on val/test
```

### Metrics

**Regression:**

```json
{
  "model_type": "SVR",
  "task_type": "regression",
  "params": {
    "C": 10.0,
    "epsilon": 0.1,
    "kernel": "rbf",
    "random_seed": 42
  },
  "splits": {
    "train": {"rmse": 0.52, "mae": 0.38, "r2": 0.89},
    "val": {"rmse": 0.81, "mae": 0.62, "r2": 0.73},
    "test": {"rmse": 0.85, "mae": 0.65, "r2": 0.71}
  }
}
```

### Loading Trained Model

```python
import joblib

model = joblib.load("runs/current/model_svr.pkl")

# Make predictions
X_new = ...  # Morgan fingerprints (numpy array)
predictions = model.predict(X_new)
```

### When to Use SVR

- **Non-linear relationships**: RBF kernel captures complex patterns
- **Memory efficiency**: More memory-efficient than Random Forest for large datasets
- **Regression tasks**: Designed specifically for continuous value prediction
- **High-dimensional features**: Works well with 2048-bit fingerprints

## Graph Neural Networks

Deep learning on molecular graphs for state-of-the-art performance. GNN support (PyTorch, PyTorch Geometric) is included in the default install.

### Configuration

```python
from dta_gnn.models.gnn import GnnTrainConfig

config = GnnTrainConfig(
    architecture="gin",      # gin, gcn, gat, sage, pna, transformer, tag, arma, cheb, supergat
    embedding_dim=128,       # Output embedding size
    hidden_dim=128,          # Hidden layer size
    num_layers=5,            # GNN depth
    dropout=0.1,             # Dropout rate
    pooling="add",           # add, mean, max, attention
    residual=False,          # Skip connections
    head_mlp_layers=2,       # Prediction head depth
    # Architecture-specific parameters (optional)
    gin_conv_mlp_layers=2,   # GIN: MLP depth in convolution
    gin_train_eps=False,     # GIN: Whether to learn epsilon
    gin_eps=0.0,             # GIN: Initial epsilon value
    gat_heads=4,             # GAT: Number of attention heads
    sage_aggr="mean",        # GraphSAGE: Aggregation (mean, max, lstm, pool)
    transformer_heads=4,     # Transformer: Number of attention heads
    tag_k=2,                 # TAG: K-hop message passing
    arma_num_stacks=1,       # ARMA: Number of stacks
    arma_num_layers=1,       # ARMA: Number of layers per stack
    cheb_k=2,                # Cheb: K-hop spectral filtering
    supergat_heads=4,        # SuperGAT: Number of attention heads
    supergat_attention_type="MX",  # SuperGAT: Attention type (MX, SD)
    lr=1e-3,                 # Learning rate
    weight_decay=0.0,        # L2 regularization
    batch_size=64,           # Batch size
    epochs=10,               # Training epochs
    random_seed=42           # Reproducibility
)
```

### Training

```python
from dta_gnn.models.gnn import train_gnn_on_run

result = train_gnn_on_run(
    run_dir="runs/current",
    config=config
)

print(f"Task: {result.task_type}")
print(f"Metrics: {result.metrics}")
```

### Architectures

DTA-GNN supports 10 GNN architectures:

| Architecture | Description | Key Characteristics |
|--------------|-------------|---------------------|
| **GIN** | Graph Isomorphism Network | Highly expressive; sum aggregation with learnable ε; MLP-based updates with strong theoretical discriminative power |
| **GCN** | Graph Convolutional Network | Symmetric normalized adjacency; efficient and stable spectral convolution; strong baseline for semi-supervised learning |
| **GAT** | Graph Attention Network | Learnable neighbor attention; multi-head attention for stability; supports edge features and residual connections |
| **GraphSAGE** | Sample and Aggregate | Inductive learning; neighborhood sampling for scalability; flexible aggregators (mean, max, LSTM, pool) |
| **PNA** | Principal Neighbourhood Aggregation | Multiple aggregators and degree-aware scalers; adapts to varying node degree distributions; robust on heterogeneous graphs |
| **Transformer** | Graph Transformer with multi-head attention | Dot-product self-attention; optional edge features; gated skip connections for stable deep learning |
| **TAG** | Topology Adaptive Graph Convolution | Explicit K-hop message passing; adapts filters to local topology; polynomial-style convolution |
| **ARMA** | Auto-Regressive Moving Average | Recursive stacked filters with residual connections; stable deep propagation; efficient spectral approximation |
| **Cheb** | Chebyshev Spectral Graph Convolution | K-hop localized spectral filtering; Chebyshev polynomial approximation; avoids eigen-decomposition |
| **SuperGAT** | Supervised Graph Attention Network | Self-supervised attention via link prediction; combines structural and feature-based attention; robust attention learning |

### GNN Output Files

```
runs/current/
├── model_gnn_gin.pt           # Full model state
├── encoder_gin.pt             # Encoder weights only
├── encoder_gin_config.json    # Encoder configuration
├── model_metrics_gnn_gin.json # Evaluation metrics
└── model_predictions_gnn_gin.csv  # Predictions
```

### Extracting Embeddings

Use the trained encoder to extract molecular embeddings:

```python
from dta_gnn.models.gnn import extract_gnn_embeddings_on_run

result = extract_gnn_embeddings_on_run(
    run_dir="runs/current",
    batch_size=256
)

print(f"Molecules: {result.n_molecules}")
print(f"Embedding dim: {result.embedding_dim}")
print(f"Saved to: {result.embeddings_path}")
```

Output: `molecule_embeddings.npz` containing:

- `molecule_chembl_id`: Array of IDs
- `embeddings`: (N, embedding_dim) array

### Loading Embeddings

```python
import numpy as np

data = np.load("runs/current/molecule_embeddings.npz", allow_pickle=True)
ids = data["molecule_chembl_id"]
embeddings = data["embeddings"]

print(f"Shape: {embeddings.shape}")
# (5000, 128) for 5000 molecules with dim=128
```

## Model Comparison

### Quick Benchmark

```python
from dta_gnn.models import train_random_forest_on_run, train_svr_on_run
from dta_gnn.models.gnn import train_gnn_on_run, GnnTrainConfig

# Random Forest
rf_result = train_random_forest_on_run("runs/current")
print(f"RF Test: {rf_result.metrics['splits']['test']}")

# SVR (regression only)
svr_result = train_svr_on_run("runs/current", C=10.0, epsilon=0.1, kernel="rbf")
print(f"SVR Test: {svr_result.metrics['splits']['test']}")

# GNN
gnn_config = GnnTrainConfig(epochs=20)
gnn_result = train_gnn_on_run("runs/current", config=gnn_config)
print(f"GNN Test: {gnn_result.metrics['splits']['test']}")
```

### Model Selection Guide

| Model | Best For | Speed | Memory | Accuracy |
|-------|----------|-------|--------|----------|
| **Random Forest** | Quick baselines, regression tasks | Fast | Medium | Good |
| **SVR** | Regression with non-linear relationships, memory-efficient | Medium | Low | Good |
| **GNN** | Best accuracy, molecular structure learning | Slow | High | Best |



## Custom Model Training

### Using sklearn

```python
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from dta_gnn.features import calculate_morgan_fingerprints

# Load data
df = pd.read_csv("runs/current/dataset.csv")
compounds = pd.read_csv("runs/current/compounds.csv")

# Merge SMILES
df = df.merge(compounds[["molecule_chembl_id", "smiles"]], on="molecule_chembl_id")

# Featurize
df = calculate_morgan_fingerprints(df)

# Convert to arrays
def fp_to_array(fp_str):
    return np.array([int(b) for b in fp_str])

X = np.vstack(df["morgan_fingerprint"].apply(fp_to_array))
y = df["label"].values

# Split
train_mask = df["split"] == "train"
X_train, y_train = X[train_mask], y[train_mask]

# Train
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

### Using PyTorch

```python
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dta_gnn.features.molecule_graphs import build_graphs_2d

# Build graphs
molecules = list(zip(df["molecule_chembl_id"], df["smiles"]))
graphs = build_graphs_2d(molecules)

# Convert to PyTorch Geometric
data_list = [graph_to_pyg(g) for g in graphs]

# Create DataLoader
train_loader = DataLoader(
    [d for d in data_list if d.split == "train"],
    batch_size=64,
    shuffle=True
)

# Define model
class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
        pass

# Training loop
model = SimpleGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = compute_loss(out, batch.y)
        loss.backward()
        optimizer.step()
```

## Best Practices

### Model Selection

1. **Start with Random Forest**: Fast baseline, often surprisingly good for regression tasks
2. **Try SVR for regression**: Memory-efficient alternative to RF, good for non-linear relationships
3. **Try GNN for better accuracy**: Worth the extra compute, learns molecular structure

### Hyperparameter Tuning

Use the built-in HPO:

```python
from dta_gnn.models.hyperopt import HyperoptConfig, optimize_gnn_wandb

config = HyperoptConfig(
    model_type="GNN",
    n_trials=20,
    gnn_optimize_lr=True,
    gnn_optimize_epochs=True
)

result = optimize_gnn_wandb("runs/current", config=config, project="my-project")
print(f"Best: {result.best_params}")
```

### Avoiding Overfitting

- Use dropout (0.1-0.3)
- Early stopping (monitor val loss)
- Weight decay (1e-4 to 1e-2)
- Proper train/val/test splits

### Reproducibility

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

## Troubleshooting

### Out of memory (GNN)

- Reduce batch size
- Reduce hidden_dim and embedding_dim
- Use gradient accumulation

### Poor performance

- Check for data leakage (use scaffold split)
- Increase model capacity
- Train longer (more epochs)
- Try different architectures

### Slow training

- Use GPU (`export CUDA_VISIBLE_DEVICES=0`)
- Reduce num_layers
- Use smaller model (GCN vs GIN)
