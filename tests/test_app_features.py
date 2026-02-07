"""Tests for app feature helpers (compound and protein processing)."""

import pandas as pd
import pytest

from dta_gnn.app_features.compound import (
    build_smiles_frame,
    featurize_smiles_morgan,
    _parse_smiles_text,
)
from dta_gnn.app_features.proteins import (
    build_sequence_frame,
    _parse_sequences_text,
)


class TestParseSmilesText:
    """Tests for SMILES text parsing."""

    def test_comma_separated(self):
        """Test parsing comma-separated SMILES."""
        result = _parse_smiles_text("CCO, CC(=O)O, c1ccccc1")
        assert result == ["CCO", "CC(=O)O", "c1ccccc1"]

    def test_newline_separated(self):
        """Test parsing newline-separated SMILES."""
        result = _parse_smiles_text("CCO\nCC(=O)O\nc1ccccc1")
        assert result == ["CCO", "CC(=O)O", "c1ccccc1"]

    def test_mixed_separators(self):
        """Test parsing with mixed separators."""
        result = _parse_smiles_text("CCO, CC(=O)O\nc1ccccc1")
        assert len(result) == 3

    def test_empty_string(self):
        """Test parsing empty string."""
        result = _parse_smiles_text("")
        assert result == []

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        result = _parse_smiles_text("  CCO  ,  CC(=O)O  ")
        assert result == ["CCO", "CC(=O)O"]


class TestBuildSmilesFrame:
    """Tests for build_smiles_frame function."""

    def test_from_text_input(self):
        """Test building frame from text input."""
        result = build_smiles_frame(
            smiles_text="CCO, CC(=O)O", df_state=None, source_mode="text"
        )

        assert len(result) == 2
        assert "smiles" in result.columns
        assert "molecule_chembl_id" in result.columns

    def test_from_dataset_state(self):
        """Test building frame from dataset state."""
        df_state = pd.DataFrame(
            {"molecule_chembl_id": ["M1", "M2"], "smiles": ["CCO", "CC(=O)O"]}
        )

        result = build_smiles_frame(
            smiles_text="", df_state=df_state, source_mode="dataset"
        )

        assert len(result) == 2
        assert result.loc[0, "smiles"] == "CCO"

    def test_canonical_smiles_column(self):
        """Test handling of canonical_smiles column."""
        df_state = pd.DataFrame(
            {"molecule_chembl_id": ["M1"], "canonical_smiles": ["CCO"]}
        )

        result = build_smiles_frame(
            smiles_text="", df_state=df_state, source_mode="dataset"
        )

        assert "smiles" in result.columns

    def test_empty_text_and_no_state_raises(self):
        """Test that empty text with no state raises error."""
        with pytest.raises(ValueError):
            build_smiles_frame(smiles_text="", df_state=None, source_mode="text")

    def test_generates_custom_ids(self):
        """Test that custom IDs are generated for text input."""
        result = build_smiles_frame(
            smiles_text="CCO, CC(=O)O", df_state=None, source_mode="text"
        )

        assert result.loc[0, "molecule_chembl_id"] == "custom_1"
        assert result.loc[1, "molecule_chembl_id"] == "custom_2"


class TestFeaturizeSmilessMorgan:
    """Tests for Morgan fingerprint featurization."""

    def test_basic_featurization(self):
        """Test basic Morgan fingerprint calculation."""
        df = pd.DataFrame(
            {"molecule_chembl_id": ["M1", "M2"], "smiles": ["CCO", "c1ccccc1"]}
        )

        result = featurize_smiles_morgan(df)

        assert "morgan_fingerprint" in result.columns
        assert len(result.loc[0, "morgan_fingerprint"]) == 2048

    def test_custom_parameters(self):
        """Test with custom radius and n_bits."""
        df = pd.DataFrame({"smiles": ["CCO"]})

        result = featurize_smiles_morgan(df, radius=3, n_bits=1024)

        assert len(result.loc[0, "morgan_fingerprint"]) == 1024

    def test_missing_smiles_column_raises(self):
        """Test that missing smiles column raises error."""
        df = pd.DataFrame({"other": ["value"]})

        with pytest.raises(ValueError, match="smiles"):
            featurize_smiles_morgan(df)


class TestParseSequencesText:
    """Tests for sequence text parsing."""

    def test_comma_separated(self):
        """Test parsing comma-separated sequences."""
        result = _parse_sequences_text("ACDE, FGHI, KLMN")
        assert result == ["ACDE", "FGHI", "KLMN"]

    def test_converts_to_uppercase(self):
        """Test that sequences are converted to uppercase."""
        result = _parse_sequences_text("acde, FgHi")
        assert result == ["ACDE", "FGHI"]


class TestBuildSequenceFrame:
    """Tests for build_sequence_frame function."""

    def test_from_text_input(self):
        """Test building frame from text input."""
        result = build_sequence_frame(
            seqs_text="ACDE, FGHI", df_state=None, source_mode="text"
        )

        assert len(result) == 2
        assert "sequence" in result.columns
        assert "target_chembl_id" in result.columns

    def test_from_dataset_state(self):
        """Test building frame from dataset state."""
        df_state = pd.DataFrame(
            {"target_chembl_id": ["T1", "T2"], "sequence": ["ACDE", "FGHI"]}
        )

        result = build_sequence_frame(
            seqs_text="", df_state=df_state, source_mode="dataset"
        )

        assert len(result) == 2

    def test_empty_raises(self):
        """Test that empty input raises error."""
        with pytest.raises(ValueError):
            build_sequence_frame(seqs_text="", df_state=None, source_mode="text")
