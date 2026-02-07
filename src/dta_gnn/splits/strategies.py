import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Tuple


def split_random(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Random split into Train/Val/Test.
    """
    if len(df) == 0:
        # Create empty DataFrames with same dtypes
        col_list = df.columns
        dt = df.dtypes
        return (
            pd.DataFrame(columns=col_list).astype(dt),
            pd.DataFrame(columns=col_list).astype(dt),
            pd.DataFrame(columns=col_list).astype(dt),
            pd.DataFrame(columns=col_list).astype(dt),
        )
        
    train, test = train_test_split(df, test_size=test_size, random_state=seed)

    # Further split train to get val
    # val_size is relative to TOTAL, so we need to calculate fraction of REMAINING (Train)
    # remaining = 1 - test_size
    # fraction = val_size / remaining
    if val_size > 0:
        relative_val_size = val_size / (1 - test_size)
        train, val = train_test_split(
            train, test_size=relative_val_size, random_state=seed
        )
        val["split"] = "val"
    else:
        # Create empty DataFrame with same dtypes to avoid FutureWarning
        val = pd.DataFrame(columns=df.columns).astype(df.dtypes)

    # Add split column
    train["split"] = "train"
    test["split"] = "test"

    # Filter out empty DataFrames before concat to avoid FutureWarning
    dfs_to_concat = [d for d in [train, val, test] if len(d) > 0]
    if not dfs_to_concat:
        result = pd.DataFrame(columns=df.columns)
    else:
        result = pd.concat(dfs_to_concat, ignore_index=True)
    
    return result, train, val, test


def split_cold_drug_scaffold(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Scaffold split (Cold Drug).
    """
    # 1. Generate scaffolds
    scaffolds = {}
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        try:
            scaff = MurckoScaffold.MurckoScaffoldSmiles(smi)
            if scaff not in scaffolds:
                scaffolds[scaff] = []
            scaffolds[scaff].append(idx)
        except Exception:
            # Fallback for invalid/empty
            pass

    # 2. Sort scaffolds by size (to balance) or shuffle?
    # Standard practice: sort by num molecules to put rare scaffolds in test?
    # Or random shuffle scaffolds? DeepChem does random sort.

    scaffold_sets = list(scaffolds.values())

    # Deterministic shuffle of scaffold groups
    rng = np.random.default_rng(seed)
    rng.shuffle(scaffold_sets)

    train_idxs, val_idxs, test_idxs = [], [], []
    train_cutoff = len(df) * (1 - test_size - val_size)
    val_cutoff = len(df) * (1 - test_size)

    current_count = 0
    for group in scaffold_sets:
        if current_count < train_cutoff:
            train_idxs.extend(group)
        elif current_count < val_cutoff:
            val_idxs.extend(group)
        else:
            test_idxs.extend(group)
        current_count += len(group)

    df.loc[train_idxs, "split"] = "train"
    df.loc[val_idxs, "split"] = "val"
    df.loc[test_idxs, "split"] = "test"

    return df


def split_temporal(
    df: pd.DataFrame,
    year_col: str = "year",
    split_year: int = 2022,
    val_size: float = 0.1,
) -> pd.DataFrame:
    """
    Temporal split based on year.
    Train: year < split_year
    Test: year >= split_year
    Val: random subset of Train (or could be time-based if requested, but simple random of past is standard)
    """
    # Ensure year is numeric
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    train_mask = df[year_col] < split_year
    test_mask = df[year_col] >= split_year

    # Split Train further into Train/Val if val_size > 0
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if val_size > 0 and len(train_df) > 0:
        try:
            tr, va = train_test_split(train_df, test_size=val_size, random_state=42)
            train_df = tr
            val_df = va
            val_df["split"] = "val"
        except ValueError:
            # Create empty DataFrame with same dtypes to avoid FutureWarning
            val_df = pd.DataFrame(columns=df.columns).astype(df.dtypes)
    else:
        # Create empty DataFrame with same dtypes to avoid FutureWarning
        val_df = pd.DataFrame(columns=df.columns).astype(df.dtypes)

    train_df["split"] = "train"
    test_df["split"] = "test"

    # Filter out empty DataFrames before concat to avoid FutureWarning
    dfs_to_concat = [d for d in [train_df, val_df, test_df] if len(d) > 0]
    if not dfs_to_concat:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(dfs_to_concat, ignore_index=True)
