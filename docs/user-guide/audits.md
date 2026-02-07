# Leakage Audits

DTA-GNN includes built-in audit functions to detect data leakage between train and test sets. These audits are essential for ensuring valid model evaluation.

## Why Audit for Leakage?

Data leakage occurs when information from the test set influences training, leading to:

- Overly optimistic performance metrics
- Poor real-world generalization
- Invalid scientific conclusions

Common sources of leakage in drug discovery:

| Leakage Type | Description | Detection |
|--------------|-------------|-----------|
| **Scaffold leakage** | Same molecular scaffold in train and test | Scaffold audit |
| **Temporal leakage** | Future data used for training | Temporal audit |

## Available Audits

### Scaffold Leakage Audit

Checks if molecular scaffolds from the test set appear in the training set:

```python
from dta_gnn.audits import audit_scaffold_leakage

train_df = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

result = audit_scaffold_leakage(
    train_df,
    test_df,
    smiles_col="smiles"
)

print(result)
# {
#     'train_scaffolds': 1523,
#     'test_scaffolds': 412,
#     'overlap_count': 45,
#     'leakage_ratio': 0.109
# }
```

#### Interpreting Results

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `train_scaffolds` | Unique scaffolds in train | N/A |
| `test_scaffolds` | Unique scaffolds in test | N/A |
| `overlap_count` | Scaffolds in both sets | 0 |
| `leakage_ratio` | overlap / test_scaffolds | 0.0 |

!!! success "Good Result"
    A `leakage_ratio` of **0.0** means no scaffold leakage (perfect scaffold split).

!!! warning "Concerning Result"
    A `leakage_ratio` > **0.1** indicates significant leakage. Consider using scaffold split.

### Target Leakage Audit

Checks if the same targets appear in both train and test sets:

```python
from dta_gnn.audits import audit_target_leakage

result = audit_target_leakage(
    train_df,
    test_df,
    target_col="target_chembl_id"
)

print(result)
# {
#     'train_targets': 15,
#     'test_targets': 5,
#     'overlap_count': 0,
#     'leakage_ratio': 0.0
# }
```

#### When to Use

- For multi-target datasets where target leakage could occur
- When working with datasets containing multiple targets

## Running Audits

The `Pipeline` class does **not** run audits automatically. After building a dataset, run audits manually using the Python API (`audit_scaffold_leakage`, `audit_target_leakage`) or the CLI (`dta_gnn audit <file>`). See the workflow below.

## Manual Audit Workflow

```python
from dta_gnn.audits import audit_scaffold_leakage, audit_target_leakage
import pandas as pd

# Load your dataset
df = pd.read_csv("dataset.csv")

# Split by partition
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
test = df[df["split"] == "test"]

# Audit train vs test
scaffold_audit = audit_scaffold_leakage(train, test)
target_audit = audit_target_leakage(train, test)

# Also audit train vs val (often overlooked!)
scaffold_audit_val = audit_scaffold_leakage(train, val)

print("Train vs Test:")
print(f"  Scaffold leakage: {scaffold_audit['leakage_ratio']:.2%}")
print(f"  Target leakage: {target_audit['leakage_ratio']:.2%}")

print("\nTrain vs Val:")
print(f"  Scaffold leakage: {scaffold_audit_val['leakage_ratio']:.2%}")
```

## Audit Thresholds

| Leakage Ratio | Interpretation | Action |
|---------------|----------------|--------|
| 0% | Perfect | None needed |
| 0-5% | Minor | Acceptable for most cases |
| 5-15% | Moderate | Consider re-splitting |
| >15% | Severe | Must re-split |

## Comparing Split Strategies

Use audits to compare splitting strategies:

```python
from dta_gnn.splits import split_random, split_cold_drug_scaffold

# Random split
df_random, *_ = split_random(df.copy(), test_size=0.2)
train_r = df_random[df_random["split"] == "train"]
test_r = df_random[df_random["split"] == "test"]

# Scaffold split
df_scaffold = split_cold_drug_scaffold(df.copy(), test_size=0.2)
train_s = df_scaffold[df_scaffold["split"] == "train"]
test_s = df_scaffold[df_scaffold["split"] == "test"]

# Compare leakage
leak_random = audit_scaffold_leakage(train_r, test_r)
leak_scaffold = audit_scaffold_leakage(train_s, test_s)

print(f"Random split leakage: {leak_random['leakage_ratio']:.2%}")
print(f"Scaffold split leakage: {leak_scaffold['leakage_ratio']:.2%}")
```

Typical results:

```
Random split leakage: 78.50%
Scaffold split leakage: 0.00%
```

## Visualization

Visualize scaffold overlap:

```python
import matplotlib.pyplot as plt
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffolds(df, smiles_col="smiles"):
    scaffolds = set()
    for smi in df[smiles_col].dropna():
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(smi))
        except:
            pass
    return scaffolds

train_scaffolds = get_scaffolds(train)
test_scaffolds = get_scaffolds(test)

# Venn diagram
from matplotlib_venn import venn2

plt.figure(figsize=(8, 6))
venn2([train_scaffolds, test_scaffolds], ('Train', 'Test'))
plt.title('Scaffold Overlap')
plt.savefig('scaffold_venn.png')
```

## Best Practices

!!! tip "Audit Guidelines"
    
    1. **Always audit after splitting** - Make it part of your workflow
    2. **Audit all pairs** - Train vs val, train vs test, val vs test
    3. **Document audit results** - Include in papers/reports
    4. **Set thresholds upfront** - Define acceptable leakage levels
    5. **Re-audit after data updates** - Leakage can change with new data

## Common Issues

### High leakage despite scaffold split

Check for:

1. Empty or invalid SMILES causing fallback behavior
2. Very small scaffolds (single ring) being common
3. Bug in splitting code

```python
# Debug: Check scaffold distribution
scaffolds = df["smiles"].apply(
    lambda s: MurckoScaffold.MurckoScaffoldSmiles(s) if pd.notna(s) else None
)
print(f"Unique scaffolds: {scaffolds.nunique()}")
print(f"Most common: {scaffolds.value_counts().head()}")
```

### Zero leakage with random split

This is unusual and may indicate:

1. All molecules have unique scaffolds
2. Very small dataset
3. Incorrect audit configuration

### Audit fails with missing data

Ensure SMILES column is populated:

```python
# Check for missing SMILES
missing = df["smiles"].isna().sum()
print(f"Missing SMILES: {missing}/{len(df)}")

# Filter before audit
df_valid = df.dropna(subset=["smiles"])
```

## API Reference

### `audit_scaffold_leakage`

```python
def audit_scaffold_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smiles_col: str = "smiles"
) -> Dict[str, Any]:
    """
    Check if scaffolds from test set appear in train set.
    
    Args:
        train_df: Training set DataFrame
        test_df: Test set DataFrame
        smiles_col: Column containing SMILES strings
    
    Returns:
        Dictionary with audit results
    """
```

### `audit_target_leakage`

```python
def audit_target_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target_chembl_id"
) -> Dict[str, Any]:
    """
    Check exact target ID overlap.
    
    Args:
        train_df: Training set DataFrame
        test_df: Test set DataFrame
        target_col: Column containing target identifiers
    
    Returns:
        Dictionary with audit results
    """
```
