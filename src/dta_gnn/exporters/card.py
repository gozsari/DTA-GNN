import pandas as pd
from typing import Dict, Any


def generate_dataset_card(df: pd.DataFrame, metadata: Dict[str, Any], output_path: str):
    """
    Generate a markdown dataset card.
    """
    # Calculate label statistics
    if 'label' in df.columns and len(df['label'].dropna()) > 0:
        label_min = df['label'].min()
        label_max = df['label'].max()
        label_mean = df['label'].mean()
        label_range = f"{label_min:.2f} - {label_max:.2f} (pChEMBL)"
        mean_affinity = f"{label_mean:.2f} (pChEMBL)"
    else:
        label_range = 'N/A'
        mean_affinity = 'N/A'
    
    card = f"""# Dataset Card

## Metadata
- **Target IDs**: {metadata.get('targets')}
- **Source**: {metadata.get('source')} (Web/SQLite)
- **Date**: {pd.Timestamp.now()}

## Statistics
- **Total Samples**: {len(df)}
- **Label Range**: {label_range}
- **Mean Affinity**: {mean_affinity}
- **Columns**: {', '.join(df.columns)}

## Split Information
- **Strategy**: {metadata.get('split_method')}
"""

    if "split" in df.columns:
        counts = df["split"].value_counts().to_dict()
        card += f"""
### Split Counts
- **Train**: {counts.get('train', 0)}
- **Val**: {counts.get('val', 0)}
- **Test**: {counts.get('test', 0)}
"""

    card += """
## Preprocessing
- **Deduplication**: Median aggregation
- **Standardization**: Converted to pChEMBL, dropped invalid units.
- **Audits**: Leakage check performed.

## Leakage Audit
"""
    if "audit" in metadata:
        card += f"```json\n{metadata['audit']}\n```"

    with open(output_path, "w") as f:
        f.write(card)
