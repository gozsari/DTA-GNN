import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict, Any


def audit_scaffold_leakage(
    train_df: pd.DataFrame, test_df: pd.DataFrame, smiles_col: str = "smiles"
) -> Dict[str, Any]:
    """
    Check if scaffolds from test set appear in train set.
    """

    def get_scaffolds(df):
        scaffs = set()
        for s in df[smiles_col].dropna():
            try:
                scaffs.add(MurckoScaffold.MurckoScaffoldSmiles(s))
            except Exception:
                pass
        return scaffs

    train_scaffolds = get_scaffolds(train_df)
    test_scaffolds = get_scaffolds(test_df)

    overlap = train_scaffolds.intersection(test_scaffolds)
    return {
        "train_scaffolds": len(train_scaffolds),
        "test_scaffolds": len(test_scaffolds),
        "overlap_count": len(overlap),
        "leakage_ratio": len(overlap) / len(test_scaffolds) if test_scaffolds else 0.0,
    }


def audit_target_leakage(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "target_chembl_id"
) -> Dict[str, Any]:
    """
    Check exact target ID overlap.
    """
    train_targets = set(train_df[target_col].dropna())
    test_targets = set(test_df[target_col].dropna())

    overlap = train_targets.intersection(test_targets)
    return {
        "train_targets": len(train_targets),
        "test_targets": len(test_targets),
        "overlap_count": len(overlap),
        "leakage_ratio": len(overlap) / len(test_targets) if test_targets else 0.0,
    }
