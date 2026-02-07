
import pytest
import sys
from unittest.mock import MagicMock, patch, ANY

import pandas as pd
import numpy as np
import math
from dta_gnn.models.hyperopt import (
    HyperoptConfig,
    optimize_gnn_wandb,
    optimize_random_forest_wandb,
    optimize_svr_wandb,
)

def test_hyperopt_config_init():
    """Test HyperoptConfig initialization and default values."""
    config = HyperoptConfig(model_type="GNN")
    assert config.model_type == "GNN"
    assert config.architecture == "gin"
    assert config.optimize_lr is False
    assert config.lr_min == 0.00001
    
    # Test custom values
    config = HyperoptConfig(
        model_type="GNN",
        optimize_lr=True,
        lr_min=0.01,
        lr_max=0.1
    )
    assert config.optimize_lr is True
    assert config.lr_min == 0.01
    assert config.lr_max == 0.1

def test_hyperopt_config_validation():
    """Test HyperoptConfig validation (if any)."""
    config = HyperoptConfig(
        model_type="GNN",
        epochs_min=10,
        epochs_max=20
    )
    assert config.epochs_min == 10
    assert config.epochs_max == 20

@pytest.fixture
def mock_wandb():
    mock = MagicMock()
    mock.sweep.return_value = "dummy_sweep_id"
    return mock

@pytest.fixture
def run_dir_setup(tmp_path):
    # Create required files
    # Create with 2 classes for classification/stratified checks, though test uses regression mostly
    (tmp_path / "dataset.csv").write_text("split,smiles,label,molecule_chembl_id\nval,C,1.0,mol1\ntrain,O,0.5,mol2\ntrain,N,0.2,mol3")
    (tmp_path / "compounds.csv").write_text("molecule_chembl_id,smiles\nmol1,C\nmol2,O\nmol3,N")
    return tmp_path

def test_optimize_gnn_wandb_parameter_generation(mock_wandb, run_dir_setup):
    """Verify that optimize_gnn_wandb generates correct parameters for W&B."""
    
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        config = HyperoptConfig(
            model_type="GNN",
            architecture="gin",
            optimize_lr=True,
            lr_min=0.001,
            lr_max=0.01,
            optimize_epochs=True,
            epochs_min=5,
            epochs_max=10
        )
        
        # Don't run agent execution here, just setup
        mock_wandb.agent.return_value = None
        
        optimize_gnn_wandb(
            run_dir=run_dir_setup,
            config=config,
            project="test_project"
        )
        
        # Check the arguments passed to wandb.sweep
        assert mock_wandb.sweep.called
        call_args = mock_wandb.sweep.call_args
        sweep_config = call_args[1]['sweep']
        
        assert sweep_config['method'] == 'bayes'
        assert 'lr' in sweep_config['parameters']
        assert sweep_config['parameters']['lr']['min'] == 0.001
        assert sweep_config['parameters']['lr']['max'] == 0.01
        
        assert 'epochs' in sweep_config['parameters']
        assert sweep_config['parameters']['epochs']['min'] == 5
        assert sweep_config['parameters']['epochs']['max'] == 10
        
        assert 'batch_size' not in sweep_config['parameters']

def test_optimize_rf_wandb_parameter_generation(mock_wandb, run_dir_setup):
    """Verify optimize_random_forest_wandb parameter generation."""
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        config = HyperoptConfig(
            model_type="RandomForest",
            rf_optimize_n_estimators=True,
            rf_n_estimators_min=100,
            rf_n_estimators_max=200
        )
        
        mock_wandb.agent.return_value = None
        
        try:
            # Need to mock calculate_morgan_fingerprints as it is called BEFORE sweep setup
            with patch("dta_gnn.models.hyperopt.calculate_morgan_fingerprints") as mock_fp:
                # Mock return DF
                mock_fp.return_value = pd.DataFrame({
                    "molecule_chembl_id": ["mol1", "mol2", "mol3"],
                    "morgan_fingerprint": ["1010", "0101", "1100"] 
                })
                
                optimize_random_forest_wandb(
                    run_dir=run_dir_setup,
                    config=config,
                    project="test_project",
                    n_bits=4 # Small n_bits matching mock string length
                )
        except Exception as e:
            # Fail gracefully if logic inside optimize_rf changed
             pytest.fail(f"Execution failed: {e}")
        
        assert mock_wandb.sweep.called
        sweep_config = mock_wandb.sweep.call_args[1]['sweep']
        
        assert 'n_estimators' in sweep_config['parameters']
        assert sweep_config['parameters']['n_estimators']['min'] == 100
        assert sweep_config['parameters']['n_estimators']['max'] == 200

def test_optimize_svr_wandb_parameter_generation(mock_wandb, run_dir_setup):
    """Verify optimize_svr_wandb parameter generation."""
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        config = HyperoptConfig(
            model_type="SVR",
            svr_optimize_C=True,
            svr_C_min=1.0,
            svr_C_max=10.0
        )
        
        mock_wandb.agent.return_value = None
        
        # Need to mock calculate_morgan_fingerprints
        with patch("dta_gnn.models.hyperopt.calculate_morgan_fingerprints") as mock_fp:
            mock_fp.return_value = pd.DataFrame({
                "molecule_chembl_id": ["mol1", "mol2", "mol3"],
                "morgan_fingerprint": ["1010", "0101", "1100"]
            })
            
            optimize_svr_wandb(
                run_dir=run_dir_setup,
                config=config,
                project="test_project",
                n_bits=4
            )
        
        assert mock_wandb.sweep.called
        sweep_config = mock_wandb.sweep.call_args[1]['sweep']
        
        assert 'C' in sweep_config['parameters']
        assert sweep_config['parameters']['C']['min'] == 1.0
        assert sweep_config['parameters']['C']['max'] == 10.0

def test_optimize_no_params_error(mock_wandb, run_dir_setup):
    """Test that ValueError is raised when no optimization flags are enabled."""
    config = HyperoptConfig(model_type="GNN")
    
    with patch.dict("sys.modules", {"wandb": mock_wandb}):
        with pytest.raises(ValueError, match="No parameters selected"):
            optimize_gnn_wandb(
                run_dir=run_dir_setup,
                config=config,
                project="test_project"
            )

# --- Execution Tests ---

def test_optimize_gnn_wandb_execution(mock_wandb, run_dir_setup):
    """Test execution of GNN hyperopt trial."""
    
    # 1. Setup Mock Agent to run the callback
    def side_effect_agent(sweep_id, function, count):
        function() # Run trial once
    mock_wandb.agent.side_effect = side_effect_agent
    
    # 2. Setup Mock Config for the trial
    mock_wandb.config = {
        "lr": 0.005,
        "epochs": 7,
        "batch_size": 32 # Should default if not in sweep but accessed? No, it accesses attributes.
    }
    # optimize_gnn uses getattr(wandb, "config", {})
    
    config = HyperoptConfig(
        model_type="GNN",
        optimize_lr=True,
        lr_min=0.001,
        lr_max=0.01
    )
    
    mock_res = MagicMock()
    mock_res.task_type = "regression"
    mock_res.metrics = {"splits": {"val": {"r2": 0.8, "rmse": 0.2}}}
    
    with patch.dict("sys.modules", {"wandb": mock_wandb}), \
         patch("dta_gnn.models.gnn.train_gnn_on_run", return_value=mock_res) as mock_train:
        
        result = optimize_gnn_wandb(
            run_dir=run_dir_setup,
            config=config,
            project="test_project"
        )
        
        # Verify train_gnn_on_run called
        assert mock_train.called
        # Verify params usage
        _, kwargs = mock_train.call_args
        assert kwargs['config'].lr == 0.005
        assert kwargs['config'].epochs == 7 
        # Wait, I put 'epochs' in mock_wandb.config above.
        # But 'optimize_epochs' is False in config, so it might not be in the sweep params logic of `_trial_fn`?
        # `GnnTrainConfig` creation: `epochs=int(sampled.get("epochs", ...))`
        # So it should be 7.
        assert kwargs['config'].epochs == 7

        assert result.best_value == 0.8

def test_optimize_rf_wandb_execution(mock_wandb, run_dir_setup):
    """Test execution of RF hyperopt trial."""
    
    def side_effect_agent(sweep_id, function, count):
        function()
    mock_wandb.agent.side_effect = side_effect_agent
    
    # Needs explicit config
    mock_wandb.config = {"n_estimators": 150}
    
    config = HyperoptConfig(
        model_type="RandomForest",
        rf_optimize_n_estimators=True
    )
    
    with patch.dict("sys.modules", {"wandb": mock_wandb}), \
         patch("dta_gnn.models.hyperopt.calculate_morgan_fingerprints") as mock_fp, \
         patch("dta_gnn.models.hyperopt.RandomForestRegressor") as mock_rf:
         
        mock_fp.return_value = pd.DataFrame({
            "molecule_chembl_id": ["mol1", "mol2", "mol3"],
            "morgan_fingerprint": ["1010", "0101", "1100"] 
        })
        
        # Mock fit/predict
        mock_inst = mock_rf.return_value
        mock_inst.predict.return_value = np.array([0.5]) # 1 val sample?
        # Our dataset setup: 3 samples. 1 val (mol1). 2 train (mol2, mol3).
        # mol1 is val.
        
        # Because we have split col, it uses holdout validation.
        # It calculates RMSE/R2.
        
        result = optimize_random_forest_wandb(
            run_dir=run_dir_setup,
            config=config,
            project="test_project",
            n_bits=4
        )
        
        assert mock_rf.called
        assert mock_inst.fit.called
        assert mock_inst.predict.called    

def test_optimize_svr_wandb_execution(mock_wandb, run_dir_setup):
    """Test execution of SVR hyperopt trial."""
    
    def side_effect_agent(sweep_id, function, count):
        function()
    mock_wandb.agent.side_effect = side_effect_agent
    
    mock_wandb.config = {"C": 5.0}
    
    config = HyperoptConfig(
        model_type="SVR",
        svr_optimize_C=True
    )
    
    with patch.dict("sys.modules", {"wandb": mock_wandb}), \
         patch("dta_gnn.models.hyperopt.calculate_morgan_fingerprints") as mock_fp, \
         patch("dta_gnn.models.hyperopt.SVR") as mock_svr:
         
        mock_fp.return_value = pd.DataFrame({
            "molecule_chembl_id": ["mol1", "mol2", "mol3"],
            "morgan_fingerprint": ["1010", "0101", "1100"] 
        })
        
        mock_inst = mock_svr.return_value
        mock_inst.predict.return_value = np.array([0.5])
        
        result = optimize_svr_wandb(
            run_dir=run_dir_setup,
            config=config,
            project="test_project",
            n_bits=4
        )
        
        assert mock_svr.called
        assert mock_inst.fit.called
