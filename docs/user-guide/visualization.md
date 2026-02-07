# Visualization

DTA-GNN provides visualization functions to explore datasets, analyze activity distributions, and visualize chemical space.

## Available Functions

| Function | Purpose | Output |
|----------|---------|--------|
| `plot_activity_distribution` | Histogram of pChEMBL values | Bar chart |
| `plot_split_sizes` | Train/val/test distribution | Bar chart |
| `plot_chemical_space` | 2D projection of molecular diversity | Scatter plot |

## Activity Distribution

Visualize the distribution of binding affinity values (pChEMBL) in your dataset.

```python
from dta_gnn.visualization import plot_activity_distribution
import matplotlib.pyplot as plt

fig = plot_activity_distribution(
    df,
    title="Binding Affinity Distribution"
)

plt.savefig("activity_dist.png")
plt.show()
```

**Features:**

- Bins pChEMBL values into 0.5-unit intervals
- Automatically handles missing values
- Customizable title

**Example output:**

```
pChEMBL Interval | Count
-----------------|------
5.0              | 150
5.5              | 200
6.0              | 300
6.5              | 250
...
```

## Split Sizes

Visualize the distribution of samples across train/validation/test splits.

```python
from dta_gnn.visualization import plot_split_sizes
import matplotlib.pyplot as plt

fig = plot_split_sizes(df)

plt.savefig("split_sizes.png")
plt.show()
```

**Features:**

- Shows count for each split
- Color-coded bars
- Automatic labeling

## Chemical Space Visualization

Project molecular structures into 2D space using dimensionality reduction techniques.

```python
from dta_gnn.visualization import plot_chemical_space
import matplotlib.pyplot as plt

# Option 1: Dictionary with groups
smiles_dict = {
    "train": train_df["smiles"].tolist(),
    "test": test_df["smiles"].tolist()
}

fig = plot_chemical_space(
    smiles_dict,
    method="t-SNE",      # or "PCA"
    radius=2,            # Morgan FP radius
    n_bits=1024,         # Fingerprint size
    perplexity=30,       # t-SNE perplexity
    random_state=42
)

plt.savefig("chemical_space.png")
plt.show()

# Option 2: Flat list
smiles_list = df["smiles"].tolist()
fig = plot_chemical_space(smiles_list, method="PCA")
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | "t-SNE" | Dimensionality reduction: "t-SNE" or "PCA" |
| `radius` | 2 | Morgan fingerprint radius |
| `n_bits` | 1024 | Fingerprint bit size |
| `perplexity` | 30 | t-SNE perplexity (5-50 recommended) |
| `learning_rate` | 200.0 | t-SNE learning rate |
| `random_state` | 42 | Random seed for reproducibility |

### When to Use Each Method

**t-SNE:**
- ✅ Better for visualizing clusters
- ✅ Non-linear relationships
- ❌ Slower for large datasets (>10k molecules)
- ❌ Perplexity tuning required

**PCA:**
- ✅ Fast and deterministic
- ✅ Good for large datasets
- ✅ Preserves global structure
- ❌ Linear projections only

### Example: Comparing Splits

```python
from dta_gnn.visualization import plot_chemical_space

# Group by split
train_smiles = df[df["split"] == "train"]["smiles"].tolist()
val_smiles = df[df["split"] == "val"]["smiles"].tolist()
test_smiles = df[df["split"] == "test"]["smiles"].tolist()

smiles_by_split = {
    "Train": train_smiles,
    "Validation": val_smiles,
    "Test": test_smiles
}

fig = plot_chemical_space(
    smiles_by_split,
    method="t-SNE",
    perplexity=30
)
```

This helps verify that:
- Test set covers different chemical space (scaffold split)
- No obvious clustering by split (random split)
- Chemical diversity is maintained

## Top-K Visualization

When visualizing model predictions, you can filter to show only the top-K highest binding affinity predictions from the test set. This is useful for identifying the most promising compounds.

### Usage in UI

1. Go to the **Visualization** tab
2. Select **"Model Predictions"** as the color scheme
3. Select a trained model
4. Enable **"Show Top-K Test Predictions"**
5. Set **K value** (10-1000, default: 100)
6. Click **"Generate Visualization"**

The visualization will show only the top-K molecules with highest predicted binding affinity from the test set, making it easier to identify the most promising compounds.

### Use Cases

- **Hit Identification**: Focus on the most promising compounds for further analysis
- **Model Validation**: Verify that top predictions align with chemical intuition
- **Scaffold Analysis**: Identify common scaffolds among top predictions
- **Diversity Analysis**: Check if top-K compounds are diverse or clustered

### Example Workflow

```python
# 1. Train a model and generate predictions
from dta_gnn.models import train_gnn_on_run, GnnTrainConfig

config = GnnTrainConfig(architecture="gin", epochs=50)
result = train_gnn_on_run("runs/current", config=config)

# 2. Extract embeddings
from dta_gnn.models import extract_gnn_embeddings_on_run
extract_gnn_embeddings_on_run("runs/current")

# 3. In UI: Visualize with top-K filtering
# - Select "Model Predictions" color scheme
# - Enable "Show Top-K Test Predictions"
# - Set K=50 to see top 50 predictions
# - Analyze the chemical space of top compounds
```

### Interpreting Top-K Results

- **Clustering**: If top-K compounds cluster together, they may share similar scaffolds
- **Diversity**: If top-K compounds are spread out, the model identifies diverse promising compounds
- **Ground Truth Alignment**: Compare top-K predictions with actual high-affinity compounds in the dataset

## Complete Visualization Workflow

```python
import matplotlib.pyplot as plt
from dta_gnn.visualization import (
    plot_activity_distribution,
    plot_split_sizes,
    plot_chemical_space
)

# 1. Activity distribution
fig1 = plot_activity_distribution(df, title="Dataset Activity Distribution")
plt.savefig("activity_dist.png", dpi=150, bbox_inches="tight")
plt.close(fig1)

# 2. Split sizes
fig2 = plot_split_sizes(df)
plt.savefig("split_sizes.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# 3. Chemical space
smiles_dict = {
    "Train": df[df["split"] == "train"]["smiles"].tolist(),
    "Test": df[df["split"] == "test"]["smiles"].tolist()
}
fig3 = plot_chemical_space(smiles_dict, method="t-SNE")
plt.savefig("chemical_space.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
```

## Integration with UI

The web UI automatically generates these visualizations when building datasets:

1. **Activity Distribution**: Shown in Dataset Builder tab
2. **Split Sizes**: Displayed after dataset creation
3. **Chemical Space**: Available in the visualization panel

## Performance Tips

### Large Datasets

For datasets with >10,000 molecules:

```python
# Use PCA instead of t-SNE
fig = plot_chemical_space(
    smiles_list,
    method="PCA",
    n_bits=512  # Smaller fingerprint for speed
)

# Or sample before visualization
import random
sampled = random.sample(smiles_list, 5000)
fig = plot_chemical_space(sampled, method="t-SNE")
```

### Memory Optimization

```python
# Process in batches for very large datasets
def visualize_large_dataset(df, batch_size=5000):
    all_figs = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        fig = plot_activity_distribution(batch)
        all_figs.append(fig)
        plt.close(fig)  # Free memory
    return all_figs
```

## Troubleshooting

### Empty plots

If plots appear empty:

```python
# Check for missing data
print(df["pchembl_value"].notna().sum())
print(df["split"].value_counts())

# Filter before plotting
df_clean = df.dropna(subset=["pchembl_value", "split"])
fig = plot_activity_distribution(df_clean)
```

### t-SNE errors

If t-SNE fails:

```python
# Reduce perplexity for small datasets
n_samples = len(smiles_list)
perplexity = min(30, max(5, n_samples // 4))

fig = plot_chemical_space(
    smiles_list,
    method="t-SNE",
    perplexity=perplexity
)
```

### Invalid SMILES

The visualization functions automatically skip invalid SMILES, but you can check:

```python
from rdkit import Chem

valid = df["smiles"].apply(
    lambda s: Chem.MolFromSmiles(s) is not None
)
print(f"Valid SMILES: {valid.sum()}/{len(df)}")
```
