"""Tests for dataset exporters."""

import pandas as pd

from dta_gnn.exporters.card import generate_dataset_card


class TestGenerateDatasetCard:
    """Tests for dataset card generation."""

    def test_basic_card_generation(self, tmp_path):
        """Test basic dataset card generation."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2", "M3"],
                "target_chembl_id": ["T1", "T1", "T1"],
                "label": [1, 0, 1],
                "split": ["train", "train", "test"],
            }
        )

        metadata = {"targets": ["T1"], "source": "SQLite", "split_method": "scaffold"}

        output_path = tmp_path / "DATASET_CARD.md"
        generate_dataset_card(df, metadata, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()

        assert "# Dataset Card" in content
        assert "T1" in content
        assert "SQLite" in content

    def test_card_contains_statistics(self, tmp_path):
        """Test that card contains dataset statistics."""
        df = pd.DataFrame(
            {
                "label": [1, 1, 0, 0, 0],
                "split": ["train", "train", "train", "val", "test"],
            }
        )

        metadata = {"targets": ["T1"], "source": "Web", "split_method": "random"}
        output_path = tmp_path / "card.md"

        generate_dataset_card(df, metadata, str(output_path))

        content = output_path.read_text()
        assert "Total Samples" in content
        assert "5" in content  # Total samples
        assert "Label Range" in content or "Mean Affinity" in content

    def test_card_contains_split_counts(self, tmp_path):
        """Test that card contains split information."""
        df = pd.DataFrame(
            {
                "label": [1, 0, 1, 0, 1],
                "split": ["train", "train", "train", "val", "test"],
            }
        )

        metadata = {"targets": [], "source": "SQLite", "split_method": "scaffold"}
        output_path = tmp_path / "card.md"

        generate_dataset_card(df, metadata, str(output_path))

        content = output_path.read_text()
        assert "Split Counts" in content
        assert "Train" in content
        assert "Val" in content
        assert "Test" in content

    def test_card_with_audit_info(self, tmp_path):
        """Test card generation with audit metadata."""
        df = pd.DataFrame({"label": [1, 0], "split": ["train", "test"]})

        metadata = {
            "targets": ["T1"],
            "source": "SQLite",
            "split_method": "scaffold",
            "audit": '{"leakage_ratio": 0.0}',
        }

        output_path = tmp_path / "card.md"
        generate_dataset_card(df, metadata, str(output_path))

        content = output_path.read_text()
        assert "Leakage Audit" in content
        assert "leakage_ratio" in content

    def test_card_without_split_column(self, tmp_path):
        """Test card generation without split column."""
        df = pd.DataFrame(
            {"label": [1, 0, 1], "molecule_chembl_id": ["M1", "M2", "M3"]}
        )

        metadata = {"targets": [], "source": "Web", "split_method": "none"}
        output_path = tmp_path / "card.md"

        generate_dataset_card(df, metadata, str(output_path))

        content = output_path.read_text()
        assert "# Dataset Card" in content

    def test_card_columns_listed(self, tmp_path):
        """Test that column names are listed in card."""
        df = pd.DataFrame(
            {
                "molecule_chembl_id": ["M1"],
                "target_chembl_id": ["T1"],
                "smiles": ["CCO"],
                "label": [1],
                "split": ["train"],
            }
        )

        metadata = {"targets": [], "source": "SQLite", "split_method": "random"}
        output_path = tmp_path / "card.md"

        generate_dataset_card(df, metadata, str(output_path))

        content = output_path.read_text()
        assert "Columns" in content
        assert "molecule_chembl_id" in content
        assert "smiles" in content
