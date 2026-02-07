"""Tests for Morgan fingerprint calculation."""

import pandas as pd

from dta_gnn.features import calculate_morgan_fingerprints


class TestCalculateMorganFingerprints:
    """Tests for calculate_morgan_fingerprints function."""

    def test_basic_calculation(self):
        """Test basic fingerprint calculation."""
        df = pd.DataFrame(
            {"molecule_chembl_id": ["M1", "M2"], "smiles": ["CCO", "c1ccccc1"]}
        )

        result = calculate_morgan_fingerprints(df)

        assert "morgan_fingerprint" in result.columns
        assert len(result) == 2

    def test_fingerprint_length(self):
        """Test that fingerprints have correct length."""
        df = pd.DataFrame({"smiles": ["CCO"]})

        result = calculate_morgan_fingerprints(df, n_bits=2048)
        fp = result.loc[0, "morgan_fingerprint"]

        assert len(fp) == 2048

    def test_custom_n_bits(self):
        """Test with custom n_bits parameter."""
        df = pd.DataFrame({"smiles": ["CCO"]})

        result = calculate_morgan_fingerprints(df, n_bits=1024)
        fp = result.loc[0, "morgan_fingerprint"]

        assert len(fp) == 1024

    def test_custom_radius(self):
        """Test with custom radius parameter."""
        df = pd.DataFrame({"smiles": ["CCO"]})

        # Should not raise
        result = calculate_morgan_fingerprints(df, radius=3)

        assert "morgan_fingerprint" in result.columns

    def test_custom_output_column(self):
        """Test with custom output column name."""
        df = pd.DataFrame({"smiles": ["CCO"]})

        result = calculate_morgan_fingerprints(df, out_col="my_fp")

        assert "my_fp" in result.columns
        assert "morgan_fingerprint" not in result.columns

    def test_custom_smiles_column(self):
        """Test with custom smiles column name."""
        df = pd.DataFrame({"my_smiles": ["CCO"]})

        result = calculate_morgan_fingerprints(df, smiles_col="my_smiles")

        assert "morgan_fingerprint" in result.columns

    def test_invalid_smiles_dropped(self):
        """Test that invalid SMILES are dropped when drop_failures=True."""
        df = pd.DataFrame({"smiles": ["CCO", "not_a_smiles", "c1ccccc1"]})

        result = calculate_morgan_fingerprints(df, drop_failures=True)

        assert len(result) == 2

    def test_invalid_smiles_kept(self):
        """Test that invalid SMILES are kept with None when drop_failures=False."""
        df = pd.DataFrame({"smiles": ["CCO", "not_a_smiles", "c1ccccc1"]})

        result = calculate_morgan_fingerprints(df, drop_failures=False)

        assert len(result) == 3
        assert result.loc[1, "morgan_fingerprint"] is None

    def test_empty_smiles_handled(self):
        """Test handling of empty SMILES strings."""
        df = pd.DataFrame({"smiles": ["CCO", "", "c1ccccc1"]})

        result = calculate_morgan_fingerprints(df, drop_failures=True)

        assert len(result) == 2

    def test_nan_smiles_handled(self):
        """Test handling of NaN SMILES values."""
        df = pd.DataFrame({"smiles": ["CCO", None, "c1ccccc1"]})

        result = calculate_morgan_fingerprints(df, drop_failures=True)

        assert len(result) == 2

    def test_fingerprint_is_bitstring(self):
        """Test that fingerprint is a string of 0s and 1s."""
        df = pd.DataFrame({"smiles": ["CCO"]})

        result = calculate_morgan_fingerprints(df)
        fp = result.loc[0, "morgan_fingerprint"]

        assert isinstance(fp, str)
        assert set(fp).issubset({"0", "1"})

    def test_different_molecules_different_fps(self):
        """Test that different molecules produce different fingerprints."""
        df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(=O)O"]})

        result = calculate_morgan_fingerprints(df)
        fps = result["morgan_fingerprint"].tolist()

        # All should be unique
        assert len(set(fps)) == 3

    def test_original_columns_preserved(self):
        """Test that original columns are preserved."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2"],
                "smiles": ["CCO", "c1ccccc1"],
                "extra_col": ["a", "b"],
            }
        )

        result = calculate_morgan_fingerprints(df)

        assert "molecule_chembl_id" in result.columns
        assert "extra_col" in result.columns

    def test_does_not_modify_original(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({"smiles": ["CCO"]})
        original_cols = set(df.columns)

        calculate_morgan_fingerprints(df)

        assert set(df.columns) == original_cols

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"smiles": []})

        result = calculate_morgan_fingerprints(df)

        assert len(result) == 0
        assert "morgan_fingerprint" in result.columns
