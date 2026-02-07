"""Tests for molecule graph construction."""

import numpy as np

from dta_gnn.features.molecule_graphs import build_graphs_2d, smiles_to_graph_2d


def test_smiles_to_graph_2d_shapes():
    g = smiles_to_graph_2d(molecule_chembl_id="CHEMBL_TEST", smiles="CCO")
    assert g is not None

    assert g.atom_type.shape == (3,)
    assert g.atom_type.dtype == np.int64

    assert g.atom_feat.shape == (3, 6)
    assert g.atom_feat.dtype == np.float32

    # C-C and C-O bonds => 2 bonds => 4 directed edges
    assert g.edge_index.shape == (2, 4)
    assert g.edge_index.dtype == np.int64

    assert g.edge_attr.shape == (4, 6)
    assert g.edge_attr.dtype == np.float32


def test_build_graphs_2d_drops_invalid_smiles_by_default():
    graphs = build_graphs_2d(
        molecules=[("A", "CCO"), ("B", "not_a_smiles")],
        drop_failures=True,
    )
    assert len(graphs) == 1
    assert graphs[0].molecule_chembl_id == "A"


class TestSmilesToGraph2dAdvanced:
    """Additional tests for SMILES to graph conversion."""

    def test_benzene_ring(self):
        """Test graph construction for benzene."""
        g = smiles_to_graph_2d(molecule_chembl_id="benzene", smiles="c1ccccc1")

        assert g is not None
        assert g.atom_type.shape == (6,)
        # Benzene has 6 bonds => 12 directed edges
        assert g.edge_index.shape[1] == 12

    def test_molecule_with_heteroatoms(self):
        """Test graph construction for molecules with heteroatoms."""
        g = smiles_to_graph_2d(molecule_chembl_id="test", smiles="CCN")

        assert g is not None
        assert len(g.atom_type) == 3

    def test_complex_molecule(self):
        """Test graph construction for complex molecule (caffeine)."""
        caffeine = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
        g = smiles_to_graph_2d(molecule_chembl_id="caffeine", smiles=caffeine)

        assert g is not None
        assert len(g.atom_type) > 0

    def test_invalid_smiles_returns_none(self):
        """Test that invalid SMILES returns None."""
        g = smiles_to_graph_2d(molecule_chembl_id="invalid", smiles="not_valid")

        assert g is None

    def test_edge_features_dimensions(self):
        """Test edge feature dimensions."""
        g = smiles_to_graph_2d(molecule_chembl_id="test", smiles="CCO")

        assert g.edge_attr.shape[1] == 6
        assert g.edge_attr.dtype == np.float32


class TestBuildGraphs2dAdvanced:
    """Additional tests for batch graph construction."""

    def test_all_valid(self):
        """Test with all valid SMILES."""
        molecules = [("M1", "CCO"), ("M2", "c1ccccc1"), ("M3", "CC(=O)O")]

        graphs = build_graphs_2d(molecules=molecules)

        assert len(graphs) == 3

    def test_all_invalid(self):
        """Test with all invalid SMILES."""
        molecules = [("M1", "invalid1"), ("M2", "invalid2")]

        graphs = build_graphs_2d(molecules=molecules, drop_failures=True)

        assert len(graphs) == 0

    def test_empty_list(self):
        """Test with empty molecule list."""
        graphs = build_graphs_2d(molecules=[])

        assert len(graphs) == 0

    def test_graph_ids_preserved(self):
        """Test that molecule IDs are preserved."""
        molecules = [("CHEMBL123", "CCO"), ("CHEMBL456", "CCC")]

        graphs = build_graphs_2d(molecules=molecules)

        ids = [g.molecule_chembl_id for g in graphs]
        assert "CHEMBL123" in ids
        assert "CHEMBL456" in ids
