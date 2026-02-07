# Target Mapping

DTA-GNN provides utilities to map UniProt protein accessions to ChEMBL target IDs, making it easier to work with targets identified in literature or other databases.

## Overview

When building datasets, you may have:
- **UniProt IDs**: From literature, databases, or protein sequences (e.g., `P00533`, `P04626`)
- **ChEMBL IDs**: Required for dataset building (e.g., `CHEMBL1862`, `CHEMBL203`)

Target mapping bridges this gap by querying the ChEMBL database to find corresponding target IDs.

## Quick Start

```python
from dta_gnn.io.target_mapping import (
    parse_uniprot_accessions,
    map_uniprot_to_chembl_targets_sqlite
)

# Parse UniProt IDs from text
accessions = parse_uniprot_accessions("P00533, P04626, P15056")

# Map to ChEMBL targets
result = map_uniprot_to_chembl_targets_sqlite(
    sqlite_path="chembl_36.db",
    accessions=accessions
)

# Use in dataset building
from dta_gnn.pipeline import Pipeline
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")
df = pipeline.build_dta(
    target_ids=result.resolved_target_chembl_ids,
    split_method="scaffold"
)
```

## Parsing Functions

### Parse UniProt Accessions

Parse UniProt accession strings from various formats:

```python
from dta_gnn.io.target_mapping import parse_uniprot_accessions

# Comma-separated
ids1 = parse_uniprot_accessions("P00533, P04626, P15056")
# Returns: ["P00533", "P04626", "P15056"]

# Newline-separated
ids2 = parse_uniprot_accessions("P00533\nP04626\nP15056")
# Returns: ["P00533", "P04626", "P15056"]

# Space-separated
ids3 = parse_uniprot_accessions("P00533 P04626 P15056")
# Returns: ["P00533", "P04626", "P15056"]

# Mixed separators
ids4 = parse_uniprot_accessions("P00533, P04626\nP15056")
# Returns: ["P00533", "P04626", "P15056"]
```

**Validation:**
- Validates UniProt accession format
- Raises `ValueError` for invalid accessions
- Preserves order and removes duplicates

### Parse ChEMBL Target IDs

Parse ChEMBL target ID strings:

```python
from dta_gnn.io.target_mapping import parse_chembl_target_ids

targets = parse_chembl_target_ids("CHEMBL204, CHEMBL205, CHEMBL206")
# Returns: ["CHEMBL204", "CHEMBL205", "CHEMBL206"]
```

**Validation:**
- Validates ChEMBL ID format (`CHEMBL` + digits)
- Raises `ValueError` for invalid IDs
- Preserves order and removes duplicates

## Mapping Functions

### SQLite Mapping (Recommended)

Map UniProt accessions to ChEMBL targets using a local SQLite database:

```python
from dta_gnn.io.target_mapping import map_uniprot_to_chembl_targets_sqlite

result = map_uniprot_to_chembl_targets_sqlite(
    sqlite_path="chembl_36.db",
    accessions=["P00533", "P04626", "P15056"]
)
```

**Returns:** `UniProtToChEMBLResult` dataclass with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `resolved_target_chembl_ids` | `list[str]` | All unique ChEMBL target IDs found |
| `per_input` | `dict[str, list[str]]` | Mapping from each UniProt ID to its ChEMBL targets |
| `unmapped` | `list[str]` | UniProt IDs that couldn't be mapped |

**Example:**

```python
result = map_uniprot_to_chembl_targets_sqlite(
    sqlite_path="chembl_36.db",
    accessions=["P00533", "P04626", "INVALID"]
)

print(result.resolved_target_chembl_ids)
# ["CHEMBL1862", "CHEMBL203"]

print(result.per_input)
# {
#     "P00533": ["CHEMBL1862"],
#     "P04626": ["CHEMBL203"],
#     "INVALID": []
# }

print(result.unmapped)
# ["INVALID"]
```

### Web API Mapping

Map UniProt accessions using the ChEMBL Web API (slower, no database required):

```python
from dta_gnn.io.target_mapping import map_uniprot_to_chembl_targets_web

result = map_uniprot_to_chembl_targets_web(
    accessions=["P00533", "P04626"]
)

print(result.resolved_target_chembl_ids)
# ["CHEMBL1862", "CHEMBL203"]
```

**Note:** Web API mapping is slower and subject to rate limits. Use SQLite mapping for production.

## Integration with Pipeline

### Using in Dataset Building

```python
from dta_gnn.io.target_mapping import (
    parse_uniprot_accessions,
    map_uniprot_to_chembl_targets_sqlite
)
from dta_gnn.pipeline import Pipeline

# Step 1: Parse UniProt IDs from your source
uniprot_text = """
P00533  # EGFR
P04626  # ERBB2
P15056  # BRAF
"""
accessions = parse_uniprot_accessions(uniprot_text)

# Step 2: Map to ChEMBL
result = map_uniprot_to_chembl_targets_sqlite(
    sqlite_path="chembl_36.db",
    accessions=accessions
)

# Step 3: Check for unmapped
if result.unmapped:
    print(f"Warning: Could not map: {result.unmapped}")

# Step 4: Build dataset with mapped targets
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")
df = pipeline.build_dta(
    target_ids=result.resolved_target_chembl_ids,
    split_method="scaffold"
)
```

### Using in UI

The Web UI supports UniProt input directly:

1. Select **Target ID Type**: "UniProt"
2. Enter UniProt accessions: `P00533, P04626, P15056`
3. The UI automatically maps to ChEMBL targets
4. Unmapped accessions are logged as warnings

## One-to-Many Mapping

A single UniProt accession may map to multiple ChEMBL targets (e.g., different isoforms, species variants):

```python
result = map_uniprot_to_chembl_targets_sqlite(
    sqlite_path="chembl_36.db",
    accessions=["P00533"]
)

print(result.per_input["P00533"])
# ["CHEMBL1862", "CHEMBL203"]  # Multiple targets for same protein
```

**Handling:**
- All mapped targets are included in `resolved_target_chembl_ids`
- Dataset building will fetch activities for all targets
- Consider filtering by organism if needed

## Troubleshooting

### No targets found

If `unmapped` contains your UniProt IDs:

1. **Verify accession format**: Must match `P[0-9A-Z]{5}` pattern
2. **Check ChEMBL version**: Newer proteins may not be in older ChEMBL releases
3. **Try alternative accessions**: Some proteins have multiple UniProt entries

### Multiple targets per accession

This is normal for:
- Different species (human vs mouse)
- Different isoforms
- Protein complexes

Filter by organism if needed:

```python
from dta_gnn.io.sqlite_source import ChemblSQLiteSource

source = ChemblSQLiteSource("chembl_36.db")
targets = source.fetch_targets(result.resolved_target_chembl_ids)

# Filter to human only
human_targets = targets[targets["organism"] == "Homo sapiens"]
```

### Slow Web API mapping

Switch to SQLite for faster mapping:

```python
# Slow (Web API)
result = map_uniprot_to_chembl_targets_web(accessions)

# Fast (SQLite)
result = map_uniprot_to_chembl_targets_sqlite("chembl_36.db", accessions)
```

## Best Practices

1. **Use SQLite mapping** for production (much faster)
2. **Check unmapped IDs** before building datasets
3. **Handle one-to-many mappings** by reviewing target details
4. **Cache mappings** if processing the same accessions repeatedly
5. **Validate accessions** before mapping (use `parse_uniprot_accessions`)

## Example: Literature-Based Dataset

Build a dataset from UniProt IDs found in a research paper:

```python
from dta_gnn.io.target_mapping import (
    parse_uniprot_accessions,
    map_uniprot_to_chembl_targets_sqlite
)
from dta_gnn.pipeline import Pipeline

# UniProt IDs from paper
paper_targets = """
P00533  # EGFR - mentioned in paper
P04626  # ERBB2 - mentioned in paper
P15056  # BRAF - mentioned in paper
"""

# Parse and map
accessions = parse_uniprot_accessions(paper_targets)
result = map_uniprot_to_chembl_targets_sqlite(
    sqlite_path="chembl_36.db",
    accessions=accessions
)

# Build dataset
pipeline = Pipeline(source_type="sqlite", sqlite_path="chembl_36.db")
df = pipeline.build_dta(
    target_ids=result.resolved_target_chembl_ids,
    split_method="scaffold"
)

print(f"Dataset: {len(df)} samples across {len(result.resolved_target_chembl_ids)} targets")
```
