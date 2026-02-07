"""Tests for Random Forest model training."""

import json

import pandas as pd
import pytest

from dta_gnn.models.random_forest import (
    RandomForestTrainResult,
    train_random_forest_on_run,
    _infer_task_type_from_labels,
    _morgan_fingerprints,
)


class TestInferTaskType:
    """Tests for task type inference from labels."""

    def test_infers_classification_for_binary(self):
        """Test that binary 0/1 labels are classified as classification."""
        df = pd.DataFrame({"label": [0, 1, 0, 1, 1]})
        assert _infer_task_type_from_labels(df) == "classification"

    def test_infers_classification_for_all_zeros(self):
        """Test that all zeros is still classification."""
        df = pd.DataFrame({"label": [0, 0, 0]})
        assert _infer_task_type_from_labels(df) == "classification"

    def test_infers_classification_for_all_ones(self):
        """Test that all ones is still classification."""
        df = pd.DataFrame({"label": [1, 1, 1]})
        assert _infer_task_type_from_labels(df) == "classification"

    def test_infers_regression_for_continuous(self):
        """Test that continuous values are classified as regression."""
        df = pd.DataFrame({"label": [5.5, 6.2, 7.8, 4.3]})
        assert _infer_task_type_from_labels(df) == "regression"

    def test_infers_regression_for_mixed(self):
        """Test that non-binary values trigger regression."""
        df = pd.DataFrame({"label": [0.0, 0.5, 1.0]})
        assert _infer_task_type_from_labels(df) == "regression"

    def test_missing_label_column(self):
        """Test handling of missing label column."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        assert _infer_task_type_from_labels(df) == "regression"


class TestMorganFingerprints:
    """Tests for Morgan fingerprint calculation."""

    def test_valid_smiles(self):
        """Test fingerprint generation for valid SMILES."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        X, valid_mask = _morgan_fingerprints(smiles)

        assert X.shape == (3, 2048)
        assert all(valid_mask)

    def test_invalid_smiles_excluded(self):
        """Test that invalid SMILES are marked as invalid."""
        smiles = ["CCO", "not_a_smiles", "c1ccccc1"]
        X, valid_mask = _morgan_fingerprints(smiles)

        assert X.shape == (2, 2048)
        assert valid_mask[0] == True  # noqa: E712
        assert valid_mask[1] == False  # noqa: E712
        assert valid_mask[2] == True  # noqa: E712

    def test_empty_list(self):
        """Test handling of empty SMILES list."""
        X, valid_mask = _morgan_fingerprints([])

        assert X.shape == (0, 2048)
        assert len(valid_mask) == 0

    def test_custom_radius_and_bits(self):
        """Test custom fingerprint parameters."""
        smiles = ["CCO"]
        X, valid_mask = _morgan_fingerprints(smiles, radius=3, n_bits=1024)

        assert X.shape == (1, 1024)


class TestTrainRandomForestOnRun:
    """Tests for the full Random Forest training pipeline."""

    @pytest.fixture
    def run_dir_classification(self, tmp_path):
        """Create a run directory with classification data."""
        run_dir = tmp_path / "runs" / "test_run"
        run_dir.mkdir(parents=True)

        # Create dataset.csv with binary labels
        pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2", "M3", "M4", "M5", "M6"],
                "target_chembl_id": ["T1", "T1", "T1", "T1", "T1", "T1"],
                "label": [1, 0, 1, 0, 1, 0],
                "split": ["train", "train", "train", "train", "val", "test"],
            }
        ).to_csv(run_dir / "dataset.csv", index=False)

        # Create compounds.csv
        pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2", "M3", "M4", "M5", "M6"],
                "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCCO", "CCC"],
            }
        ).to_csv(run_dir / "compounds.csv", index=False)

        return run_dir

    @pytest.fixture
    def run_dir_regression(self, tmp_path):
        """Create a run directory with regression data."""
        run_dir = tmp_path / "runs" / "test_run_reg"
        run_dir.mkdir(parents=True)

        # Create dataset.csv with continuous labels
        pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2", "M3", "M4", "M5", "M6"],
                "target_chembl_id": ["T1", "T1", "T1", "T1", "T1", "T1"],
                "label": [5.5, 6.2, 7.1, 4.8, 6.5, 5.9],
                "split": ["train", "train", "train", "train", "val", "test"],
            }
        ).to_csv(run_dir / "dataset.csv", index=False)

        pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M2", "M3", "M4", "M5", "M6"],
                "smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCCO", "CCC"],
            }
        ).to_csv(run_dir / "compounds.csv", index=False)

        return run_dir

    def test_classification_training(self, run_dir_classification):
        """Test training on classification task."""
        result = train_random_forest_on_run(
            run_dir_classification, n_estimators=10, random_seed=42
        )

        assert isinstance(result, RandomForestTrainResult)
        assert result.task_type == "classification"
        assert result.model_path.exists()
        assert result.metrics_path.exists()
        assert result.predictions_path.exists()
        assert "accuracy" in result.metrics["splits"]["train"]

    def test_regression_training(self, run_dir_regression):
        """Test training on regression task."""
        result = train_random_forest_on_run(
            run_dir_regression, n_estimators=10, random_seed=42
        )

        assert result.task_type == "regression"
        assert "rmse" in result.metrics["splits"]["train"]
        assert "mae" in result.metrics["splits"]["train"]
        assert "r2" in result.metrics["splits"]["train"]

    def test_model_artifacts_saved(self, run_dir_classification):
        """Test that all artifacts are properly saved."""
        result = train_random_forest_on_run(run_dir_classification, n_estimators=5)

        # Check model file
        import joblib

        model = joblib.load(result.model_path)
        assert hasattr(model, "predict")

        # Check metrics JSON
        with open(result.metrics_path) as f:
            metrics = json.load(f)
        assert metrics["model_type"] == "RandomForest"

        # Check predictions CSV
        preds = pd.read_csv(result.predictions_path)
        assert "y_pred" in preds.columns

    def test_missing_dataset_raises(self, tmp_path):
        """Test that missing dataset.csv raises error."""
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        (run_dir / "compounds.csv").write_text("molecule_chembl_id,smiles\nM1,CCO")

        with pytest.raises(FileNotFoundError, match="dataset.csv"):
            train_random_forest_on_run(run_dir)

    def test_missing_compounds_raises(self, tmp_path):
        """Test that missing compounds.csv raises error."""
        run_dir = tmp_path / "no_compounds"
        run_dir.mkdir()
        pd.DataFrame(
            {"molecule_chembl_id": ["M1"], "label": [1], "split": ["train"]}
        ).to_csv(run_dir / "dataset.csv", index=False)

        with pytest.raises(FileNotFoundError, match="compounds.csv"):
            train_random_forest_on_run(run_dir)
