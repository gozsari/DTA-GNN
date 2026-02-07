# Data Cleaning

DTA-GNN applies rigorous data cleaning to ensure high-quality, standardized datasets. This page describes the cleaning pipeline and available options.

## Overview

The cleaning pipeline handles:

1. **Activity standardization** - Unit conversion and pChEMBL calculation
2. **Censored value handling** - Dealing with `>` and `<` relations
3. **Duplicate aggregation** - Merging repeated measurements
4. **SMILES canonicalization** - Standardizing molecular representations

## Activity Standardization

### pChEMBL Conversion

All activity values are converted to pChEMBL scale for consistency:

$$\text{pChEMBL} = -\log_{10}(\text{value in molar})$$

For values in nanomolar (nM):

$$\text{pChEMBL} = -\log_{10}(\text{value} \times 10^{-9})$$

**Example:**

- IC50 = 100 nM → pChEMBL = 7.0
- IC50 = 1 µM → pChEMBL = 6.0
- IC50 = 10 nM → pChEMBL = 8.0

### API

```python
from dta_gnn.cleaning import standardize_activities

df_clean = standardize_activities(
    df_activities,
    convert_to_pchembl=True,  # Calculate pChEMBL if missing
    drop_censored=False        # Keep censored values (>, <)
)
```

### What it does

1. Converts `standard_value` to numeric type
2. Converts `pchembl_value` to numeric type  
3. Drops rows with missing `standard_value` or `standard_units`
4. Calculates pChEMBL for rows where it's missing (nM units only)

## Censored Value Handling

ChEMBL contains censored measurements where the exact value wasn't determined:

| Relation | Meaning | Example |
|----------|---------|---------|
| `=` | Exact measurement | IC50 = 100 nM |
| `>` | Greater than | IC50 > 10000 nM (inactive) |
| `<` | Less than | IC50 < 1 nM (very active) |

### Options

=== "Keep censored (default)"

    ```python
    df_clean = standardize_activities(
        df_activities,
        drop_censored=False  # Keep all values
    )
    ```
    
    Useful for:
    - Preserving maximum data
    - Including all available measurements

=== "Drop censored"

    ```python
    df_clean = standardize_activities(
        df_activities,
        drop_censored=True  # Only exact measurements
    )
    ```
    
    Useful for:
    - Regression tasks requiring exact values
    - Strict data quality requirements

## Duplicate Aggregation

Multiple measurements often exist for the same drug-target pair. DTA-GNN aggregates these:

### Aggregation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `median` | Median value (default) | Robust to outliers |
| `mean` | Arithmetic mean | Standard aggregation |
| `max` | Maximum value | Conservative activity estimate |
| `min` | Minimum value | Liberal activity estimate |

### API

```python
from dta_gnn.cleaning import aggregate_duplicates

df_agg = aggregate_duplicates(
    df_clean,
    group_cols=["molecule_chembl_id", "target_chembl_id"],
    agg_method="median"
)
```

### How it works

1. Groups by molecule and target IDs
2. Aggregates pChEMBL (or standard_value) using the specified method
3. For temporal data, preserves the earliest year (conservative for temporal splits)

### Example

```python
# Before aggregation
#   molecule  target   pchembl
#   CHEMBL25  CHEMBL204  6.5
#   CHEMBL25  CHEMBL204  7.0
#   CHEMBL25  CHEMBL204  6.8

# After aggregation (median)
#   molecule  target   pchembl
#   CHEMBL25  CHEMBL204  6.8
```

## SMILES Canonicalization

SMILES strings are canonicalized using RDKit for consistency:

```python
from dta_gnn.cleaning import canonicalize_smiles

canonical = canonicalize_smiles("c1ccccc1O")  # Returns "Oc1ccccc1"
```

### What it does

1. Parses SMILES into RDKit molecule object
2. Generates canonical SMILES (non-isomeric by default)
3. Returns `None` for invalid SMILES

## Complete Cleaning Pipeline

The `Pipeline` class applies all cleaning steps automatically:

```python
from dta_gnn.pipeline import Pipeline

pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")

# The build_dta method applies:
# 1. standardize_activities(convert_to_pchembl=True)
# 2. aggregate_duplicates(agg_method="median")
# 3. SMILES validation and filtering

df = pipeline.build_dta(
    target_ids=["CHEMBL204"],
    split_method="scaffold"
)
```

## Data Quality Metrics

After cleaning, review these metrics:

```python
# Check for missing values
print(df.isnull().sum())

# Check pChEMBL distribution
print(df["pchembl_value"].describe())

# Check SMILES validity
valid_smiles = df["smiles"].notna().sum()
print(f"Valid SMILES: {valid_smiles}/{len(df)}")
```

## Visualization

The UI provides visualizations for cleaned data:

```python
from dta_gnn.visualization import plot_activity_distribution

# Plot pChEMBL distribution
fig = plot_activity_distribution(df, value_col="pchembl_value")
```

## Best Practices

!!! tip "Cleaning Recommendations"
    
    1. **Always use pChEMBL scale** - Enables comparison across activity types
    2. **Use median aggregation** - Robust to experimental outliers
    3. **Drop censored for regression** - Ensures accurate continuous labels
    4. **Check data distribution** - Look for unexpected patterns after cleaning
    5. **Validate SMILES** - Ensure all molecules can be processed

## Troubleshooting

### All data removed after cleaning

Check if activities have valid units:

```python
print(df_activities["standard_units"].value_counts())
# Should see 'nM', 'uM', etc.
```

### pChEMBL values are NaN

Verify that standard values are positive:

```python
print((df_activities["standard_value"] <= 0).sum())
# pChEMBL can't be computed for non-positive values
```

### Unexpected aggregation results

Check for duplicate group columns:

```python
duplicates = df_clean.groupby(
    ["molecule_chembl_id", "target_chembl_id"]
).size()
print(f"Max duplicates: {duplicates.max()}")
```
