"""Tests for package initialization."""

from dta_gnn import __version__


def test_version():
    """Test that version is set correctly."""
    assert __version__ == "0.1.4"


def test_package_imports():
    """Test that main package components can be imported."""
    from dta_gnn.pipeline import Pipeline
    from dta_gnn.features import calculate_morgan_fingerprints

    # All should import without error
    assert Pipeline is not None
    assert calculate_morgan_fingerprints is not None
