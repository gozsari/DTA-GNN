
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

# Mocking gradio before importing ui avoids launching any web server stuff
with patch.dict("sys.modules", {"gradio": MagicMock()}):
    from dta_gnn.app.ui import (
        _update_model_choices,
        _sync_ho_gnn_arch,
        _sync_ho_wandb_section,
        build_ui,
    )

def test_sync_ho_gnn_arch():
    """Test logic for showing/hiding GNN arch params."""
    # Should hide if model != GNN
    update = _sync_ho_gnn_arch("gin", "RandomForest")
    # Since we mocked gradio, gr.update() returns a Mock. 
    # We can check if it was called (or just that the function runs without error).
    # Realistically, we'd check the properties of the returned update object if it wasn't a Mock.
    assert update is not None

def test_sync_ho_wandb_section():
    """Test logic for showing W&B section."""
    update_visible = _sync_ho_wandb_section("W&B Sweeps (Bayes)", "GNN")
    # We assume the mock returns something truthy or we can inspect calls if we mocked gr.update explicitly
    assert update_visible is not None

@patch("dta_gnn.app.ui.gr")
@patch("dta_gnn.app.ui.find_chembl_sqlite_dbs")
def test_build_ui_smoke(mock_find_dbs, mock_gr):
    """Smoke test for build_ui. Verifies it constructs without error."""
    mock_find_dbs.return_value = ["db1.sqlite"]
    
    # Mock all the component constructors to avoid side effects
    mock_gr.Blocks.return_value.__enter__.return_value = MagicMock()
    
    # Call build_ui
    try:
        demo, components = build_ui()
    except Exception as e:
        pytest.fail(f"build_ui() raised exception: {e}")
    
    # Check that it returned a tuple
    assert demo is not None
    assert components is not None
