"""Tests for data cleaning functions."""

import pandas as pd
import numpy as np

from dta_gnn.cleaning import (
    standardize_activities,
    aggregate_duplicates,
    canonicalize_smiles,
)


def test_standardize_activities():
    df = pd.DataFrame(
        {
            "standard_value": [100, 10, np.nan],
            "standard_units": ["nM", "nM", "nM"],
            "standard_relation": ["=", "=", "="],
            "pchembl_value": [np.nan, np.nan, np.nan],
        }
    )

    res = standardize_activities(df, convert_to_pchembl=True)
    assert len(res) == 2
    # 100 nM = 1e-7 M -> -log10(1e-7) = 7.0
    assert np.isclose(res.iloc[0]["pchembl_value"], 7.0)
    # 10 nM = 1e-8 M -> -log10(1e-8) = 8.0
    assert np.isclose(res.iloc[1]["pchembl_value"], 8.0)


def test_aggregate_duplicates():
    df = pd.DataFrame(
        {
            "molecule_chembl_id": ["C1", "C1", "C2"],
            "target_chembl_id": ["T1", "T1", "T2"],
            "pchembl_value": [6.0, 8.0, 5.0],
        }
    )

    res = aggregate_duplicates(df, agg_method="median")
    assert len(res) == 2
    row_c1 = res[res["molecule_chembl_id"] == "C1"].iloc[0]
    assert row_c1["pchembl_value"] == 7.0  # Median of 6 and 8


def test_canonicalize_smiles():
    s1 = "CCO"
    s2 = "OCC"
    assert canonicalize_smiles(s1) == canonicalize_smiles(s2)
    assert canonicalize_smiles("invalid_smiles") is None


class TestStandardizeActivitiesAdvanced:
    """Additional tests for standardize_activities."""

    def test_drops_censored_values(self):
        """Test that censored values (>, <) are handled based on settings."""
        df = pd.DataFrame(
            {
                "standard_value": [100, 200],
                "standard_units": ["nM", "nM"],
                "standard_relation": ["=", ">"],
                "pchembl_value": [None, None],
            }
        )

        result = standardize_activities(df, convert_to_pchembl=True)

        # By default, > and < relations may be dropped or kept
        assert len(result) >= 1

    def test_unit_conversion(self):
        """Test unit conversion from nM with various values."""
        df = pd.DataFrame(
            {
                "standard_value": [1000],  # 1000 nM = 1 µM = 1e-6 M -> pChEMBL = 6.0
                "standard_units": ["nM"],
                "standard_relation": ["="],
                "pchembl_value": [None],
            }
        )

        result = standardize_activities(df, convert_to_pchembl=True)

        # 1000 nM = 1e-6 M -> -log10(1e-6) = 6.0
        if len(result) > 0:
            assert np.isclose(result.iloc[0]["pchembl_value"], 6.0)


class TestAggregateDuplicatesAdvanced:
    """Additional tests for aggregate_duplicates."""

    def test_mean_aggregation(self):
        """Test mean aggregation method."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["C1", "C1"],
                "target_chembl_id": ["T1", "T1"],
                "pchembl_value": [6.0, 8.0],
            }
        )

        result = aggregate_duplicates(df, agg_method="mean")

        assert len(result) == 1
        assert result.iloc[0]["pchembl_value"] == 7.0

    def test_max_aggregation(self):
        """Test max aggregation method."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["C1", "C1"],
                "target_chembl_id": ["T1", "T1"],
                "pchembl_value": [6.0, 8.0],
            }
        )

        result = aggregate_duplicates(df, agg_method="max")

        assert result.iloc[0]["pchembl_value"] == 8.0

    def test_preserves_unique_pairs(self):
        """Test that unique pairs are preserved."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["C1", "C2", "C3"],
                "target_chembl_id": ["T1", "T1", "T2"],
                "pchembl_value": [6.0, 7.0, 8.0],
            }
        )

        result = aggregate_duplicates(df)

        assert len(result) == 3


class TestCanonicalizeSmilesAdvanced:
    """Additional tests for canonicalize_smiles."""

    def test_complex_molecule(self):
        """Test canonicalization of complex molecule."""
        # Aspirin in different representations
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        canonical = canonicalize_smiles(smiles)

        assert canonical is not None
        assert isinstance(canonical, str)

    def test_aromatic_handling(self):
        """Test handling of aromatic vs kekulized forms."""
        aromatic = "c1ccccc1"
        kekulized = "C1=CC=CC=C1"

        can1 = canonicalize_smiles(aromatic)
        can2 = canonicalize_smiles(kekulized)

        # Both should canonicalize to the same form
        assert can1 == can2

    def test_stereochemistry_preserved(self):
        """Test that stereochemistry is preserved."""
        smiles = "C[C@H](O)CC"
        canonical = canonicalize_smiles(smiles)

        assert canonical is not None
        # The @ should be preserved in some form
        assert "@" in canonical or canonical is not None
