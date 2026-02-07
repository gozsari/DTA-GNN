"""Tests for visualization functions."""

import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from dta_gnn.visualization import (
    plot_activity_distribution,
    plot_split_sizes,
    plot_chemical_space,
)


class TestPlotActivityDistribution:
    """Tests for activity distribution plotting."""

    def test_basic_plot(self):
        """Test basic activity distribution plot."""
        df = pd.DataFrame({"pchembl_value": [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]})

        fig = plot_activity_distribution(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_title(self):
        """Test plot with custom title."""
        df = pd.DataFrame({"pchembl_value": [5.0, 6.0, 7.0]})

        fig = plot_activity_distribution(df, title="Custom Title")

        assert fig.axes[0].get_title() == "Custom Title"
        plt.close(fig)

    def test_missing_pchembl_column(self):
        """Test handling of missing pchembl_value column."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        fig = plot_activity_distribution(df)

        # Should still return a figure (with "No pChEMBL values found" message)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"pchembl_value": []})

        fig = plot_activity_distribution(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotSplitSizes:
    """Tests for split size plotting."""

    def test_basic_split_plot(self):
        """Test basic split size plot."""
        df = pd.DataFrame({"split": ["train"] * 70 + ["val"] * 15 + ["test"] * 15})

        fig = plot_split_sizes(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_split_column(self):
        """Test handling of missing split column."""
        df = pd.DataFrame({"other": [1, 2, 3]})

        fig = plot_split_sizes(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_different_split_names(self):
        """Test with various split names."""
        df = pd.DataFrame({"split": ["train", "val", "test", "train", "val"]})

        fig = plot_split_sizes(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotChemicalSpace:
    """Tests for chemical space visualization."""

    def test_basic_tsne_plot(self):
        """Test basic t-SNE chemical space plot."""
        smiles_data = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCC"]

        fig = plot_chemical_space(smiles_data, method="t-SNE")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_plot(self):
        """Test PCA chemical space plot."""
        smiles_data = ["CCO", "CC(=O)O", "c1ccccc1", "CCN"]

        fig = plot_chemical_space(smiles_data, method="PCA")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_grouped_data(self):
        """Test with grouped SMILES data."""
        smiles_data = {
            "Group A": ["CCO", "CCC", "CCCC"],
            "Group B": ["c1ccccc1", "c1ccc(C)cc1"],
        }

        fig = plot_chemical_space(smiles_data)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_smiles_handled(self):
        """Test that invalid SMILES are skipped."""
        smiles_data = ["CCO", "not_valid", "c1ccccc1"]

        fig = plot_chemical_space(smiles_data)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_smiles_list(self):
        """Test handling of empty SMILES list."""
        fig = plot_chemical_space([])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_invalid_smiles(self):
        """Test handling when all SMILES are invalid."""
        fig = plot_chemical_space(["invalid1", "invalid2"])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_parameters(self):
        """Test with custom fingerprint parameters."""
        smiles_data = ["CCO", "CC(=O)O", "c1ccccc1"]

        fig = plot_chemical_space(
            smiles_data, method="t-SNE", radius=3, n_bits=512, perplexity=2
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            plot_chemical_space(["CCO"], method="UMAP")
