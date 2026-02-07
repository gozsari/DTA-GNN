import pandas as pd
import pytest

from dta_gnn.pipeline import Pipeline


class _DummySource:
    """Mock source for testing that implements the required interface."""

    def __init__(self, activities: pd.DataFrame, molecules: pd.DataFrame):
        self._activities = activities
        self._molecules = molecules

    def fetch_activities(
        self, target_ids=None, molecule_ids=None, standard_types=None, progress_callback=None
    ):
        df = self._activities.copy()
        if target_ids:
            df = df[df["target_chembl_id"].isin(target_ids)]
        if molecule_ids:
            df = df[df["molecule_chembl_id"].isin(molecule_ids)]
        if standard_types:
            df = df[df["standard_type"].isin(standard_types)]
        return df.reset_index(drop=True)

    def fetch_molecules(self, molecule_ids):
        return self._molecules[
            self._molecules["molecule_chembl_id"].isin(molecule_ids)
        ].reset_index(drop=True)

    def fetch_targets(self, target_ids):
        return pd.DataFrame(
            {
                "target_chembl_id": target_ids,
                "sequence": ["ACDE" for _ in target_ids],
                "organism": ["Human" for _ in target_ids],
            }
        )


def test_pipeline_dta_regression_label_and_standard_type():
    activities = pd.DataFrame(
        {
            "molecule_chembl_id": ["M1", "M2"],
            "target_chembl_id": ["T1", "T1"],
            "standard_type": ["Ki", "Ki"],
            "standard_value": [10.0, 100.0],
            "standard_units": ["nM", "nM"],
            "standard_relation": ["=", "="],
            "pchembl_value": [8.0, 7.0],
            "year": [2020, 2021],
        }
    )
    molecules = pd.DataFrame(
        {
            "molecule_chembl_id": ["M1", "M2"],
            "smiles": ["CCO", "CCN"],
        }
    )

    pipeline = Pipeline(source_type="web")
    pipeline.source = _DummySource(activities, molecules)  # type: ignore

    df = pipeline.build_dta(
        # Optional measurement type filtering; label always comes from pchembl_value.
        standard_types=["IC50", "Ki"],
        target_ids=["T1"],
        split_method="random",
        test_size=0.5,
        val_size=0.0,
    )

    assert not df.empty
    assert df["label"].tolist() == df["pchembl_value"].tolist()
    # Sequences are saved separately (avoids repeating per row)
    assert getattr(pipeline, "last_targets_csv", None)


class TestPipelineAdvanced:
    """Advanced tests for Pipeline class."""

    @pytest.fixture
    def mock_activities(self):
        """Sample activity data for testing."""
        return pd.DataFrame(
            {
                "molecule_chembl_id": [
                    "M1",
                    "M2",
                    "M3",
                    "M4",
                    "M5",
                    "M6",
                    "M7",
                    "M8",
                    "M9",
                    "M10",
                ],
                "target_chembl_id": [
                    "T1",
                    "T1",
                    "T1",
                    "T2",
                    "T2",
                    "T2",
                    "T3",
                    "T3",
                    "T3",
                    "T3",
                ],
                "standard_type": ["IC50"] * 10,
                "standard_value": [10, 100, 1000, 50, 500, 5000, 25, 250, 2500, 10],
                "standard_units": ["nM"] * 10,
                "standard_relation": ["="] * 10,
                "pchembl_value": [8.0, 7.0, 6.0, 7.3, 6.3, 5.3, 7.6, 6.6, 5.6, 8.0],
                "year": [2018, 2019, 2020, 2019, 2020, 2021, 2020, 2021, 2022, 2023],
            }
        )

    @pytest.fixture
    def mock_molecules(self):
        """Sample molecule data for testing."""
        return pd.DataFrame(
            {
                "molecule_chembl_id": [f"M{i}" for i in range(1, 11)],
                "smiles": [
                    "CCO",
                    "CC(=O)O",
                    "c1ccccc1",
                    "CCN",
                    "CCC",
                    "CCCC",
                    "CCCCC",
                    "c1ccc(C)cc1",
                    "C1CCCCC1",
                    "CCO",
                ],
            }
        )

    def test_pipeline_scaffold_split(self, mock_activities, mock_molecules):
        """Test pipeline with scaffold splitting."""
        pipeline = Pipeline(source_type="web")
        pipeline.source = _DummySource(mock_activities, mock_molecules)

        df = pipeline.build_dta(target_ids=["T1", "T2", "T3"], split_method="scaffold")

        assert not df.empty
        assert "split" in df.columns
        assert set(df["split"].unique()).issubset({"train", "val", "test"})

    def test_pipeline_with_featurization(self, mock_activities, mock_molecules):
        """Test pipeline with Morgan fingerprint featurization."""
        pipeline = Pipeline(source_type="web")
        pipeline.source = _DummySource(mock_activities, mock_molecules)

        df = pipeline.build_dta(
            target_ids=["T1"], split_method="random", featurize=True
        )

        assert not df.empty
        assert "morgan_fingerprint" in df.columns

    def test_pipeline_empty_result(self, mock_molecules):
        """Test pipeline when no activities are found."""
        empty_activities = pd.DataFrame(
            columns=[
                "molecule_chembl_id",
                "target_chembl_id",
                "standard_type",
                "standard_value",
                "standard_units",
                "standard_relation",
                "pchembl_value",
            ]
        )

        pipeline = Pipeline(source_type="web")
        pipeline.source = _DummySource(empty_activities, mock_molecules)

        df = pipeline.build_dta(target_ids=["NONEXISTENT"])

        assert df.empty

    def test_pipeline_targets_csv_created(self, mock_activities, mock_molecules):
        """Test that targets CSV is created."""
        import os

        pipeline = Pipeline(source_type="web")
        pipeline.source = _DummySource(mock_activities, mock_molecules)

        pipeline.build_dta(target_ids=["T1"])

        assert hasattr(pipeline, "last_targets_csv")
        assert os.path.exists(pipeline.last_targets_csv)


class TestPipelineInit:
    """Tests for Pipeline initialization."""

    def test_web_source_default(self):
        """Test default web source initialization."""
        pipeline = Pipeline()

        assert pipeline.source is not None

    def test_sqlite_source_requires_path(self):
        """Test that SQLite source requires path."""
        with pytest.raises(ValueError, match="sqlite_path"):
            Pipeline(source_type="sqlite")
