
import pytest
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from dta_gnn.splits.strategies import split_random, split_cold_drug_scaffold, split_temporal

def test_split_random():
    """Test standard random splitting."""
    df = pd.DataFrame({"smiles": ["C"] * 100, "y": range(100)})
    result, train, val, test = split_random(df, test_size=0.2, val_size=0.1, seed=42)
    
    assert len(train) == 70
    assert len(val) == 10
    assert len(test) == 20
    assert len(result) == 100
    assert "split" in result.columns
    assert set(result["split"].unique()) == {"train", "val", "test"}
    
    # Test with no validation
    result, train, val, test = split_random(df, test_size=0.2, val_size=0.0)
    assert len(train) == 80
    assert len(val) == 0
    assert len(test) == 20

def test_split_random_empty():
    """Test random splitting with empty input."""
    df = pd.DataFrame(columns=["smiles", "y"])
    result, _, _, _ = split_random(df)
    assert len(result) == 0
    assert list(result.columns) == ["smiles", "y"]

def test_split_cold_drug_scaffold_basic():
    """Test scaffold splitting basics."""
    # Create molecules with distinct scaffolds
    # Benzene vs Cyclohexane vs something else
    smiles = ["c1ccccc1", "C1CCCCC1", "CCO", "CCN", "CCC"]
    df = pd.DataFrame({"smiles": smiles, "id": range(5)})
    
    # We can mock MurckoScaffold to force specific scaffold groupings if we want to be robust against RDKit behavior
    # mapping: smi -> scaffold
    scaffold_map = {
        "c1ccccc1": "S1",
        "C1CCCCC1": "S2",
        "CCO": "S3",
        "CCN": "S3", # Same scaffold
        "CCC": "S4"  # Distinct
    }
    
    with patch("dta_gnn.splits.strategies.MurckoScaffold.MurckoScaffoldSmiles", side_effect=lambda x: scaffold_map.get(x, "")):
        # 5 mols. S1(1), S2(1), S3(2), S4(1).
        # Test size 0.2 (1 mol). Val size 0.2 (1 mol).
        # Should likely put S3 in train?
        df_split = split_cold_drug_scaffold(df, test_size=0.2, val_size=0.2, seed=42)
        
        assert "split" in df_split.columns
        counts = df_split["split"].value_counts()
        # With small N, exact counts depend on shuffle and greedy allocation
        assert "train" in counts.index
        assert "test" in counts.index

def test_split_cold_drug_scaffold_invalid():
    """Test scaffold splitting with invalid SMILES triggering exception handler."""
    df = pd.DataFrame({"smiles": ["INVALID_SMILES", "C"], "id": [1, 2]})
    
    # Mock Murcko to raise Exception for one
    def side_effect(x):
        if x == "INVALID_SMILES":
            raise ValueError("Bad SMILES")
        return "C"
        
    with patch("dta_gnn.splits.strategies.MurckoScaffold.MurckoScaffoldSmiles", side_effect=side_effect):
        df_split = split_cold_drug_scaffold(df)
        # Invalid SMILES gets skipped in scaffold dict?
        # Function catches Exception (line 63) and does `pass`.
        # So it's not added to `scaffolds`.
        # So it's not in `scaffold_sets`.
        # So it's not assigned a split?
        # Code: `df.loc[idxs, "split"] = ...`.
        # The invalid row will have NaN split.
        
        assert "split" in df_split.columns
        # One valid row gets split (likely train)
        # Invalid row remains untouched (NaN split) or whatever default was
        # The function initializes split column? No.
        # So invalid row will have NaN in "split".
        invalid_row = df_split[df_split["smiles"] == "INVALID_SMILES"]
        if "split" in invalid_row.columns:
             assert pd.isna(invalid_row.iloc[0]["split"])

def test_split_temporal():
    """Test temporal splitting."""
    df = pd.DataFrame({
        "smiles": ["C"]*10,
        "year": [2000, 2010, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]
    })
    
    # Split year 2022. 
    # < 2022: 2000, 2010, 2020, 2021 (4 items) -> Train (split into Train/Val)
    # >= 2022: 6 items -> Test
    
    result = split_temporal(df, year_col="year", split_year=2022, val_size=0.0)
    
    train = result[result["split"] == "train"]
    test = result[result["split"] == "test"]
    
    assert len(train) == 4
    assert len(test) == 6
    assert all(train["year"] < 2022)
    assert all(test["year"] >= 2022)

def test_split_temporal_with_val():
    """Test temporal splitting with validation set."""
    df = pd.DataFrame({
        "smiles": ["C"]*100,
        "year": [2000]*50 + [2025]*50
    })
    
    # Val size 0.1 of remaining train? Code: `train_test_split(train_df, test_size=val_size)`
    # Inputs: test_size is not param to split_temporal! Only val_size.
    # split_year determines test set.
    
    result = split_temporal(df, split_year=2022, val_size=0.1)
    
    test_len = len(result[result["split"] == "test"]) # 50
    train_val_len = 50
    
    val_len = len(result[result["split"] == "val"])
    train_len = len(result[result["split"] == "train"])
    
    assert test_len == 50
    assert val_len == 5 # 0.1 * 50
    assert train_len == 45

def test_split_temporal_edge_cases():
    """Test temporal splitting edge cases (bad year, empty result)."""
    df = pd.DataFrame({
        "smiles": ["C"],
        "year": ["not_a_year"]
    })
    
    # Should coerce to NaN. NaN < 2022 is False?
    result = split_temporal(df, split_year=2022)
    # NaN >= 2022 False.
    # Both masks False?
    # Then result is empty?
    # Code: `dfs_to_concat = [d for d in [train, val, test] if len(d) > 0]`.
    # If all empty, returns valid empty DF with cols.
    assert len(result) == 0
