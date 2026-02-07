"""Tests for dataset splitting strategies."""

import pandas as pd

from dta_gnn.splits import split_random, split_cold_drug_scaffold


def test_split_random():
    df = pd.DataFrame({"id": range(100), "smiles": ["C"] * 100})
    df_split, _, _, _ = split_random(df)
    counts = df_split["split"].value_counts()
    assert "train" in counts
    assert "test" in counts
    assert len(df_split) == 100


def test_split_scaffold():
    # C1, C2 same scaffold (benzene). C3 different (cyclohexane).
    df = pd.DataFrame(
        {"smiles": ["c1ccccc1", "c1ccccc1C", "C1CCCCC1"], "id": [1, 2, 3]}
    )
    res = split_cold_drug_scaffold(df, seed=42)
    # Check logic runs. Hard to assert exact splits with randomized shuffle on small N
    assert "split" in res.columns
    assert len(res) == 3


class TestSplitRandomAdvanced:
    """Additional tests for random splitting."""

    def test_reproducibility(self):
        """Test that same seed produces same splits."""
        df = pd.DataFrame({"id": range(100), "smiles": ["C"] * 100})

        df1, _, _, _ = split_random(df, seed=42)
        df2, _, _, _ = split_random(df, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds(self):
        """Test that different seeds produce different splits."""
        df = pd.DataFrame({"id": range(100), "smiles": ["C"] * 100})

        df1, _, _, _ = split_random(df, seed=42)
        df2, _, _, _ = split_random(df, seed=123)

        # After sorting by index, compare original indices (via reset_index before split)
        # The split column values may match but the rows should be shuffled differently
        train1 = set(df1[df1["split"] == "train"]["id"].tolist())
        train2 = set(df2[df2["split"] == "train"]["id"].tolist())
        # With 100 items, different seeds should give different train sets
        assert train1 != train2

    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        df = pd.DataFrame({"id": range(1000), "smiles": ["C"] * 1000})

        # Use test_size and val_size (not train_frac)
        df_split, _, _, _ = split_random(df, test_size=0.2, val_size=0.1, seed=42)
        counts = df_split["split"].value_counts()

        # Should be approximately 70/10/20
        assert 650 < counts.get("train", 0) < 750
        assert 80 < counts.get("val", 0) < 120
        assert 180 < counts.get("test", 0) < 220


class TestSplitScaffoldAdvanced:
    """Additional tests for scaffold-based splitting."""

    def test_larger_dataset(self):
        """Test scaffold split with larger dataset."""
        # Create diverse scaffolds
        smiles_list = [
            "c1ccccc1",  # benzene
            "c1ccc(C)cc1",  # toluene -> benzene scaffold
            "C1CCCCC1",  # cyclohexane
            "C1CCCC1",  # cyclopentane
            "c1ccncc1",  # pyridine
            "c1ccc2ccccc2c1",  # naphthalene
        ]

        df = pd.DataFrame({"smiles": smiles_list * 10, "id": range(60)})

        res = split_cold_drug_scaffold(df, seed=42)

        assert "split" in res.columns
        assert len(res) == 60
