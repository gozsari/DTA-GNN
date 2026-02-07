
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Need to mock PyG dependencies before import
with patch.dict("sys.modules", {
    "torch": MagicMock(),
    "torch_geometric": MagicMock(),
    "torch_geometric.data": MagicMock(),
    "torch_geometric.loader": MagicMock(),
}):
    from dta_gnn.models.gnn import train_gnn_on_run, GnnTrainConfig

@pytest.fixture
def mock_pyg_modules():
    # Helper to setup mostly-working mocks for the train function
    mock_torch = MagicMock()
    mock_torch.from_numpy.side_effect = lambda x: MagicMock()
    # Mock tensor conversion for labels
    mock_torch.tensor.side_effect = lambda x, dtype=None: MagicMock()
    
    mock_loader = MagicMock()
    
    # Mock Data object constructor
    mock_data = MagicMock()
    
    with patch.dict("sys.modules", {
        "torch": mock_torch,
        "torch_geometric": MagicMock(),
        "torch_geometric.data": MagicMock(Data=mock_data),
        "torch_geometric.loader": MagicMock(DataLoader=mock_loader),
        "dta_gnn.features.molecule_graphs": MagicMock(),
    }):
        yield {
            "torch": mock_torch,
            "data": mock_data, 
            "loader": mock_loader
        }

def test_train_gnn_on_run_basic(tmp_path, mock_pyg_modules):
    """Test basic execution path of train_gnn_on_run."""
    
    # Setup File System
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    
    # dataset.csv
    pd.DataFrame({
        "molecule_chembl_id": ["mol1", "mol2", "mol3"],
        "label": [1.0, 0.0, 1.0],
        "split": ["train", "val", "test"]
    }).to_csv(run_dir / "dataset.csv", index=False)
    
    # compounds.csv
    pd.DataFrame({
        "molecule_chembl_id": ["mol1", "mol2", "mol3"],
        "smiles": ["C", "CC", "CCC"]
    }).to_csv(run_dir / "compounds.csv", index=False)
    
    # metadata.json
    (run_dir / "metadata.json").write_text('{"task_type": "classification"}')
    
    # Mock graph conversion
    mock_graph = MagicMock()
    # Need to mimic pyg graph object attributes
    mock_graph.atom_type = np.array([0, 0])
    mock_graph.atom_feat = np.array([[0.1], [0.1]])
    mock_graph.edge_index = np.array([[0], [1]])
    mock_graph.edge_attr = np.array([[0.1], [0.1]])
    # Attributes needed for Data object creation
    # Real numpy arrays have astype, so we don't need to mock it unless we want to spy on it.
    # We just let it execute naturally.

    # We patch modules via fixture, but explicit patch here helps clarity
    with patch("dta_gnn.models.gnn.smiles_to_graph_2d", return_value=mock_graph) as mock_s2g, \
         patch("dta_gnn.models.gnn._make_encoder_and_model") as mock_make, \
         patch("dta_gnn.models.gnn._get_device"), \
         patch("dta_gnn.models.gnn.set_seed"):
            
            # Setup mock model classes
            MockEncoderClass = MagicMock()
            MockPredictorClass = MagicMock()
            mock_make.return_value = (MockEncoderClass, MockPredictorClass)
            
            # Setup instances
            mock_encoder_instance = MagicMock()
            mock_predictor_instance = MagicMock()
            MockEncoderClass.return_value = mock_encoder_instance
            MockPredictorClass.return_value = mock_predictor_instance
            
            # Ensure mocked Data structure matches expectation
            # In train_gnn_on_run:
            # data_list.append(Data(...))
            # split = getattr(d, 'split')
            
            # The mocked Data class constructor
            # We need instances of Data to have 'split' attribute
            def side_effect_data(**kwargs):
                m = MagicMock()
                for k, v in kwargs.items():
                    setattr(m, k, v)
                    # Also need to support getattr(m, k)
                return m
            
            mock_pyg_modules["data"].side_effect = side_effect_data
            
            # Mock optimizer
            mock_pyg_modules["torch"].optim.Adam.return_value = MagicMock()
            
            # Mock loader to return a batch
            mock_batch = MagicMock()
            mock_batch.to.return_value = mock_batch
            mock_batch.y = MagicMock()
            mock_batch.molecule_chembl_id = ["mol1"]
            mock_pyg_modules["loader"].return_value = [mock_batch]
            
            # Mock forward pass
            # mock_predictor_instance is what is returned by the class
            mock_logits = MagicMock()
            mock_emb = MagicMock()
            mock_predictor_instance.return_value = (mock_logits, mock_emb) 
            
            # Mock loss
            # criterion is usually MSELoss or similar called in train loop
            # We mock torch.nn.functional or the criterion object
            # In train_gnn_on_run, it uses criterion(out, y)
            mock_loss = MagicMock()
            mock_loss.item.return_value = 0.5
            # If criterion is instantiated, we need to find where.
            # Usually strict mocking of modules requires mocking the class constructor.
            # But here we mocked 'torch' module.
            # So `torch.nn.MSELoss()(pred, target)` -> returns mock_loss
            mock_pyg_modules["torch"].nn.MSELoss.return_value.return_value = mock_loss
            mock_pyg_modules["torch"].nn.L1Loss.return_value.return_value = mock_loss
            mock_pyg_modules["torch"].nn.CrossEntropyLoss.return_value.return_value = mock_loss
            # Also mock F.mse_loss if used
            mock_pyg_modules["torch"].nn.functional.mse_loss.return_value = mock_loss
            
            cfg = GnnTrainConfig(epochs=1, batch_size=2)
            
            # Execute
            result = train_gnn_on_run(run_dir, config=cfg)
            
            assert result.run_dir == run_dir
            # The real code calls torch.save()
            # Since torch is mocked globally in sys.modules for this test via fixture,
            # capture torch.save call.
            # We must access the mocked torch from the fixture dictionary
            mock_pyg_modules["torch"].save.assert_called()

def test_train_gnn_missing_files(tmp_path):
    """Test error handling for missing files."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    
    with pytest.raises(ValueError, match="Missing dataset.csv"):
        train_gnn_on_run(run_dir)
