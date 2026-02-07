
import pytest
import sys
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
from dta_gnn.models.predict import predict_with_random_forest, predict_with_svr, predict_with_gnn

@pytest.fixture
def mock_joblib():
    return MagicMock()

@pytest.fixture
def mock_rdkit():
    mock = MagicMock()
    mock.Chem.MolFromSmiles.return_value = MagicMock()
    mock.AllChem.GetMorganFingerprintAsBitVect.return_value = [0]*2048
    return mock

@pytest.fixture
def mock_torch_geom():
    mock = MagicMock()
    return mock

def test_predict_with_random_forest(mock_joblib, mock_rdkit):
    """Test predict_with_random_forest logic."""
    
    # We patch sys.modules via patch.dict wrapper
    with patch.dict("sys.modules", {
        "joblib": mock_joblib,
        "rdkit": mock_rdkit,
        "rdkit.Chem": mock_rdkit.Chem,
        "rdkit.Chem.AllChem": mock_rdkit.AllChem,
    }):
        # Mock Path to return true for model existence
        with patch("dta_gnn.models.predict.Path") as mock_path:
            mock_run_dir = MagicMock()
            mock_path.return_value.resolve.return_value = mock_run_dir
            mock_model_path = mock_run_dir / "model_rf.pkl"
            mock_model_path.exists.return_value = True
            
            # Mock Model
            mock_model = MagicMock()
            mock_joblib.load.return_value = mock_model
            # Set up prediction returns
            mock_model.predict.return_value = np.array([0.5, 0.8])
            mock_model.predict_proba.return_value = np.array([[0.5, 0.5], [0.2, 0.8]])
            
            smiles_list = ["C", "CC"]
            
            result = predict_with_random_forest(
                run_dir="output/run_1",
                smiles_list=smiles_list
            )
            
            assert result.model_type == "RandomForest"
            assert len(result.predictions) == 2
            assert "prediction" in result.predictions.columns
            assert "molecule_id" in result.predictions.columns
            assert result.predictions.iloc[0]["molecule_id"] == "mol_0"

def test_predict_with_random_forest_file_not_found():
    """Test error when model file is missing."""
    with patch("dta_gnn.models.predict.Path") as mock_path:
        mock_run_dir = MagicMock()
        mock_path.return_value.resolve.return_value = mock_run_dir
        (mock_run_dir / "model_rf.pkl").exists.return_value = False
        
        # Don't strictly need to patch libraries if it fails earlier
        with pytest.raises(FileNotFoundError):
            predict_with_random_forest("output/run_1", ["C"])

def test_predict_with_svr(mock_joblib, mock_rdkit):
    """Test predict_with_svr logic."""
    
    with patch.dict("sys.modules", {
        "joblib": mock_joblib,
        "rdkit": mock_rdkit,
        "rdkit.Chem": mock_rdkit.Chem,
        "rdkit.Chem.AllChem": mock_rdkit.AllChem,
        "rdkit.DataStructs": mock_rdkit.DataStructs,
    }):
        with patch("dta_gnn.models.predict.Path") as mock_path:
            mock_run_dir = MagicMock()
            mock_path.return_value.resolve.return_value = mock_run_dir
            mock_model_path = mock_run_dir / "model_svr.pkl"
            mock_model_path.exists.return_value = True
            
            mock_model = MagicMock()
            mock_joblib.load.return_value = mock_model
            # SVR predict returns array directly
            mock_model.predict.return_value = np.array([5.5, 6.0])
            
            smiles_list = ["C", "CC"]
            result = predict_with_svr(
                run_dir="output/run_1",
                smiles_list=smiles_list
            )
            
            assert result.model_type == "SVR"
            assert len(result.predictions) == 2
            assert result.predictions.iloc[0]["prediction"] == 5.5

def test_predict_with_gnn(mock_joblib, mock_rdkit, mock_torch_geom):
    """Test predict_with_gnn logic, including GNN architecture detection."""
    
    # Mock torch
    mock_torch = MagicMock()
    mock_torch.load.return_value = {} # Empty state dict
    mock_torch.no_grad.return_value.__enter__.return_value = None
    
    # Mock PyG
    mock_data = MagicMock()
    mock_loader = MagicMock()
    
    # Setup mock loader iteration
    # Iterate once, returning a batch
    mock_batch = MagicMock()
    mock_batch.to.return_value = mock_batch # .to(device)
    mock_batch.molecule_chembl_id = ["mol1", "mol2"]
    mock_loader.return_value = [mock_batch]
    
    with patch.dict("sys.modules", {
        "torch": mock_torch,
        "torch_geometric": mock_torch_geom,
        "torch_geometric.data": MagicMock(Data=mock_data),
        "torch_geometric.loader": MagicMock(DataLoader=mock_loader),
    }):
        # Mock internal dependencies explicitly
        # We patch where they are IMPORTED in predict.py scope? 
        # predict.py: from dta_gnn.features.molecule_graphs import smiles_to_graph_2d
        # If we didn't patch dta_gnn.features..., it imports the REAL one.
        # The real one imports torch. Torch is in sys.modules as MOCK.
        # So real module loads with Mock torch. That should be OK.
        
        # We patch the function on the real module (or whatever is loaded)
        # But we need to make sure the import in predict.py sees our mock, or we patch the name in place.
        # Since imports are inside the function, patching the source module works.
        
        with patch("dta_gnn.features.molecule_graphs.smiles_to_graph_2d") as mock_s2g, \
             patch("dta_gnn.models.gnn._make_encoder_and_model") as mock_make_model, \
             patch("dta_gnn.models.gnn._get_device") as mock_get_device, \
             patch("dta_gnn.models.gnn._load_json") as mock_load_json, \
             patch("dta_gnn.models.predict.Path") as mock_path:
                
                # Setup Paths
                mock_run_dir = MagicMock()
                mock_path.return_value.resolve.return_value = mock_run_dir
                
                # Case: Explicit architecture provided
                mock_model_path = mock_run_dir / "model_gnn_gin.pt"
                mock_config_path = mock_run_dir / "encoder_gin_config.json"
                mock_model_path.exists.return_value = True
                mock_config_path.exists.return_value = True
                
                # Mock config loading
                mock_json_file = MagicMock()
                mock_json_file.__enter__.return_value = mock_json_file
                # We need to mock open() as well since json.load reads from file object
                # But internal logic calls open() on Path object if possible, or python open()
                # predict.py uses `with open(config_path, "r") as f: config = json.load(f)`
                
                # Easier to mock json.load if we patch open globally
                with patch("builtins.open", MagicMock()) as mock_open, \
                     patch("json.load") as mock_json_load:
                    
                    mock_json_load.return_value = {"encoder": {"architecture": "gin"}}
                    
                    # Mock Model Construction
                    mock_predictor_cls = MagicMock()
                    mock_predictor_instance = MagicMock()
                    mock_make_model.return_value = (None, mock_predictor_cls)
                    mock_predictor_cls.return_value = mock_predictor_instance
                    
                    # Mock Model Prediction output: logits, embeddings
                    # Forward returns (logits, embeddings)
                    mock_logits = MagicMock()
                    # trace: logits.view(-1).float().cpu().numpy().tolist()
                    mock_logits.view.return_value.float.return_value.cpu.return_value.numpy.return_value.tolist.return_value = [0.5, 0.6]
                    
                    mock_predictor_instance.return_value = (
                        mock_logits, 
                        None
                    )
                    
                    # Mock Graph Conversion
                    mock_graph = MagicMock()
                    mock_graph.atom_type = np.array([0])
                    mock_graph.atom_feat = np.array([[0.1]])
                    mock_graph.edge_index = np.array([[0], [0]])
                    mock_graph.edge_attr = np.array([[0.1]])
                    mock_s2g.return_value = mock_graph
                    
                    # Run Test
                    result = predict_with_gnn(
                        run_dir="output/run_1",
                        smiles_list=["C", "CC"], 
                        architecture="gin"
                    )
                    
                    assert result.model_type == "GNN"
                    assert len(result.predictions) == 2


