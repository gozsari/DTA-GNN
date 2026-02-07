# Featurization

DTA-GNN provides multiple featurization options for molecules and proteins, enabling flexible representation learning for drug-target prediction.

## Overview

| Feature Type | Input | Output | Use Case |
|--------------|-------|--------|----------|
| Morgan Fingerprints | SMILES | 2048-bit vector | Classical ML (RF, SVM) |
| Molecule Graphs | SMILES | Node/edge features | Graph Neural Networks |

## Molecular Features

### Morgan Fingerprints (ECFP)

Extended Connectivity Fingerprints capture molecular substructures.

```python
from dta_gnn.features import calculate_morgan_fingerprints

df_feat = calculate_morgan_fingerprints(
    df,
    smiles_col="smiles",
    radius=2,              # 2 = ECFP4, 3 = ECFP6
    n_bits=2048,           # Fingerprint length
    out_col="morgan_fingerprint",
    drop_failures=True     # Remove invalid SMILES
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius` | 2 | Circular radius (2 → ECFP4) |
| `n_bits` | 2048 | Output vector length |
| `drop_failures` | True | Remove rows with invalid SMILES |

**Output format:**

```python
# Fingerprint stored as bitstring
fp = df_feat.loc[0, "morgan_fingerprint"]
print(type(fp))  # str
print(len(fp))   # 2048 (e.g., "0110001010...")
```

**Converting to numpy:**

```python
import numpy as np

def fp_to_array(fp_string: str) -> np.ndarray:
    return np.array([int(b) for b in fp_string], dtype=np.uint8)

# Vectorized conversion
X = np.vstack(df_feat["morgan_fingerprint"].apply(fp_to_array))
print(X.shape)  # (n_samples, 2048)
```

### Molecule Graphs (2D)

For Graph Neural Networks, molecules are represented as graphs.

```python
from dta_gnn.features.molecule_graphs import (
    build_graphs_2d,
    smiles_to_graph_2d
)

# Single molecule
graph = smiles_to_graph_2d(
    molecule_chembl_id="CHEMBL25",
    smiles="CCO"
)

print(f"Atoms: {graph.atom_type.shape[0]}")
print(f"Bonds: {graph.edge_index.shape[1] // 2}")

# Batch processing
molecules = [
    ("CHEMBL25", "CCO"),
    ("CHEMBL192", "CC(=O)O"),
    ("CHEMBL545", "c1ccccc1")
]

graphs = build_graphs_2d(
    molecules=molecules,
    drop_failures=True
)
```

**Graph structure:**

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `molecule_chembl_id` | str | Identifier |
| `atom_type` | (N,) | Atomic numbers |
| `atom_feat` | (N, 6) | Atom features |
| `edge_index` | (2, E) | Bond connectivity |
| `edge_attr` | (E, 6) | Bond features |

**Atom features (6 dimensions):**

1. Atomic number
2. Total degree
3. Formal charge
4. Total H count
5. Aromaticity (0/1)
6. Atomic mass

**Bond features (6 dimensions):**

1. Bond type one-hot (single)
2. Bond type one-hot (double)
3. Bond type one-hot (triple)
4. Bond type one-hot (aromatic)
5. Is conjugated (0/1)
6. Is in ring (0/1)

### Converting to PyTorch Geometric

```python
import torch
from torch_geometric.data import Data

def graph_to_pyg(g):
    return Data(
        atom_type=torch.from_numpy(g.atom_type.astype(np.int64)),
        atom_feat=torch.from_numpy(g.atom_feat.astype(np.float32)),
        edge_index=torch.from_numpy(g.edge_index.astype(np.int64)),
        edge_attr=torch.from_numpy(g.edge_attr.astype(np.float32)),
        molecule_chembl_id=g.molecule_chembl_id
    )

data_list = [graph_to_pyg(g) for g in graphs]
```

## Feature Pipelines

### Classification Pipeline

```python
from dta_gnn.pipeline import Pipeline
from dta_gnn.features import calculate_morgan_fingerprints

# Build dataset
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")
df = pipeline.build_dta(
    target_ids=["CHEMBL204"],
    split_method="scaffold"
)

# Add fingerprints
df = calculate_morgan_fingerprints(df)

# Convert for sklearn
import numpy as np

def fps_to_matrix(df):
    fps = df["morgan_fingerprint"].values
    X = np.array([[int(b) for b in fp] for fp in fps], dtype=np.uint8)
    return X

X = fps_to_matrix(df)
y = df["label"].values

# Train/test split already done
X_train = X[df["split"] == "train"]
y_train = y[df["split"] == "train"]
```

### GNN Pipeline

```python
from dta_gnn.models.gnn import train_gnn_on_run, GnnTrainConfig

# Prepare run directory with dataset.csv and compounds.csv
# The GNN training automatically builds graphs from SMILES

config = GnnTrainConfig(
    architecture="gin",
    epochs=20,
    lr=1e-3
)

result = train_gnn_on_run("runs/current", config=config)
```

### Multi-Modal Pipeline

Combine molecule and protein features:

```python
import numpy as np
from dta_gnn.features import calculate_morgan_fingerprints

# Molecule features
df = calculate_morgan_fingerprints(df)

# Convert fingerprints to numpy array for ML models
def build_features(row):
    mol_fp = np.array([int(b) for b in row["morgan_fingerprint"]])
    return mol_fp

df["features"] = df.apply(build_features, axis=1)
```

## Feature Quality Checks

### Check for Invalid SMILES

```python
from rdkit import Chem

def check_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

df["valid_smiles"] = df["smiles"].apply(check_smiles)
invalid = (~df["valid_smiles"]).sum()
print(f"Invalid SMILES: {invalid}/{len(df)}")
```

### Check Fingerprint Distribution

```python
import matplotlib.pyplot as plt

# Convert fingerprints to matrix
X = fps_to_matrix(df)

# Check bit activation
bit_sums = X.sum(axis=0)
plt.hist(bit_sums, bins=50)
plt.xlabel("Number of molecules with bit ON")
plt.ylabel("Count")
plt.title("Fingerprint Bit Distribution")
```

## Performance Considerations

### Memory Usage

| Feature Type | Memory per Sample |
|--------------|-------------------|
| Morgan FP (2048) | ~2 KB |
| Molecule Graph | ~1-10 KB (varies) |

### Computation Time

| Feature Type | Time per 1000 Samples |
|--------------|----------------------|
| Morgan FP | ~1s |
| Molecule Graphs | ~5s |
