"""Tests for SVR model training."""

import json

import pandas as pd
import pytest

from dta_gnn.models.svr import SvrTrainResult, train_svr_on_run


class TestTrainSvrOnRun:
    """Tests for SVR training pipeline."""

    @pytest.fixture
    def run_dir_regression(self, tmp_path):
        """Create a run directory with regression data."""
        run_dir = tmp_path / "runs" / "svr_test"
        run_dir.mkdir(parents=True)

        # Create dataset.csv with continuous labels
        pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
                "target_chembl_id": ["T1"] * 8,
                "label": [5.5, 6.2, 7.1, 4.8, 6.5, 5.9, 7.3, 6.0],
                "split": [
                    "train",
                    "train",
                    "train",
                    "train",
                    "train",
                    "val",
                    "val",
                    "test",
                ],
            }
        ).to_csv(run_dir / "dataset.csv", index=False)

        pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
                "smiles": [
                    "CCO",
                    "CC(=O)O",
                    "c1ccccc1",
                    "CCN",
                    "CCCO",
                    "CCC",
                    "CCCC",
                    "C1CCCCC1",
                ],
            }
        ).to_csv(run_dir / "compounds.csv", index=False)

        return run_dir

    def test_svr_training(self, run_dir_regression):
        """Test basic SVR training."""
        result = train_svr_on_run(run_dir_regression, C=1.0, epsilon=0.1, kernel="rbf")

        assert isinstance(result, SvrTrainResult)
        assert result.task_type == "regression"
        assert result.model_path.exists()
        assert result.metrics_path.exists()
        assert result.predictions_path.exists()

    def test_svr_metrics(self, run_dir_regression):
        """Test that SVR produces regression metrics."""
        result = train_svr_on_run(run_dir_regression)

        assert "rmse" in result.metrics["splits"]["train"]
        assert "mae" in result.metrics["splits"]["train"]
        assert "r2" in result.metrics["splits"]["train"]

    def test_svr_linear_kernel(self, run_dir_regression):
        """Test SVR with linear kernel."""
        result = train_svr_on_run(run_dir_regression, kernel="linear")

        assert result.metrics["params"]["kernel"] == "linear"

    def test_svr_hyperparams_saved(self, run_dir_regression):
        """Test that hyperparameters are saved in metrics."""
        result = train_svr_on_run(run_dir_regression, C=5.0, epsilon=0.2, kernel="rbf")

        with open(result.metrics_path) as f:
            metrics = json.load(f)

        assert metrics["params"]["C"] == 5.0
        assert metrics["params"]["epsilon"] == 0.2
        assert metrics["params"]["kernel"] == "rbf"

    def test_svr_predictions_saved(self, run_dir_regression):
        """Test that predictions are saved correctly."""
        result = train_svr_on_run(run_dir_regression)

        preds = pd.read_csv(result.predictions_path)
        assert "y_pred" in preds.columns
        # Val and test rows should be present
        assert len(preds) >= 2

    def test_invalid_kernel_defaults_to_rbf(self, run_dir_regression):
        """Test that invalid kernel falls back to rbf."""
        result = train_svr_on_run(run_dir_regression, kernel="invalid_kernel")

        assert result.metrics["params"]["kernel"] == "rbf"
