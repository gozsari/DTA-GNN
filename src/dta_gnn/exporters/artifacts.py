"""Artifact collection and export utilities for run directories."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd


def artifact_keys_in_zip() -> list[str]:
    """Stable list of artifact keys included in artifacts.zip.

    Keep this centralized so handlers don't duplicate long lists.
    """
    return [
        "dataset",
        "targets",
        "compounds",
        "metadata",
        "molecule_features",
        "protein_features",
        "model",
        "model_metrics",
        "model_predictions",
        "model_gnn",
        "model_metrics_gnn",
        "model_predictions_gnn",
        "encoder_gnn",
        "encoder_gnn_config",
        "molecule_embeddings",
    ]


def collect_artifacts(
    *,
    run_dir: str | None,
    dataset_path: str | None = None,
    targets_path: str | None = None,
    compounds_path: str | None = None,
) -> dict[str, str | None]:
    """Collect artifact file paths from a run directory.

    Args:
        run_dir: Path to the run directory
        dataset_path: Optional explicit path to dataset.csv
        targets_path: Optional explicit path to targets.csv
        compounds_path: Optional explicit path to compounds.csv

    Returns:
        Dictionary mapping artifact keys to file paths (or None if not found)
    """
    run_path = Path(run_dir).resolve() if run_dir else None

    def _maybe(p: Path) -> str | None:
        try:
            return str(p) if p.exists() else None
        except Exception:
            return None

    # If caller didn't provide explicit paths, fall back to conventional names in run_dir.
    if run_path is not None:
        if dataset_path is None:
            dataset_path = _maybe(run_path / "dataset.csv")
        if targets_path is None:
            targets_path = _maybe(run_path / "targets.csv")
        if compounds_path is None:
            compounds_path = _maybe(run_path / "compounds.csv")

        metadata_path = _maybe(run_path / "metadata.json")

        # Model artifacts (RandomForest baseline)
        model_path = _maybe(run_path / "model_rf.pkl")
        model_metrics_path = _maybe(run_path / "model_metrics.json")
        model_predictions_path = _maybe(run_path / "model_predictions.csv")

        # Model artifacts (GNN)
        gnn_model_path = _maybe(run_path / "model_gnn.pt")
        gnn_metrics_path = _maybe(run_path / "model_metrics_gnn.json")
        gnn_predictions_path = _maybe(run_path / "model_predictions_gnn.csv")

        encoder_path = _maybe(run_path / "encoder_gnn.pt")
        encoder_config_path = _maybe(run_path / "encoder_gnn_config.json")
        molecule_embeddings_path = _maybe(run_path / "molecule_embeddings.npz")

        molecule_features_path = _maybe(run_path / "molecule_features.csv")
        protein_features_path = _maybe(run_path / "protein_features.csv")

        zip_path = str(run_path / "artifacts.zip")
    else:
        metadata_path = None
        model_path = None
        model_metrics_path = None
        model_predictions_path = None
        gnn_model_path = None
        gnn_metrics_path = None
        gnn_predictions_path = None
        encoder_path = None
        encoder_config_path = None
        molecule_embeddings_path = None
        molecule_features_path = None
        protein_features_path = None

        zip_path = None

    return {
        "dataset": dataset_path,
        "targets": targets_path,
        "compounds": compounds_path,
        "metadata": metadata_path,
        "model": model_path,
        "model_metrics": model_metrics_path,
        "model_predictions": model_predictions_path,
        "model_gnn": gnn_model_path,
        "model_metrics_gnn": gnn_metrics_path,
        "model_predictions_gnn": gnn_predictions_path,
        "encoder_gnn": encoder_path,
        "encoder_gnn_config": encoder_config_path,
        "molecule_embeddings": molecule_embeddings_path,
        "molecule_features": molecule_features_path,
        "protein_features": protein_features_path,
        "zip": zip_path,
    }


def write_artifacts_zip(
    *, zip_path: str | None, paths: list[str | None]
) -> str | None:
    """Create a zip file from a list of artifact paths.

    Args:
        zip_path: Path where the zip file should be created
        paths: List of file paths to include in the zip

    Returns:
        Path to the created zip file, or None if creation failed
    """
    if not zip_path:
        return None
    try:
        zpath = Path(zip_path)
        zpath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                if p and os.path.exists(p):
                    zf.write(p, arcname=os.path.basename(p))
        return str(zpath)
    except Exception:
        return None


def write_artifacts_zip_from_manifest(
    *, artifacts: dict[str, str | None]
) -> str | None:
    """Create a zip file from an artifacts manifest dictionary.

    Args:
        artifacts: Dictionary mapping artifact keys to file paths

    Returns:
        Path to the created zip file, or None if creation failed
    """
    zip_path = artifacts.get("zip")
    keys = artifact_keys_in_zip()
    paths = [artifacts.get(k) for k in keys]
    return write_artifacts_zip(zip_path=zip_path, paths=paths)


def artifacts_table(artifacts: dict[str, str | None]) -> pd.DataFrame:
    """Create a DataFrame table of artifacts for UI display.

    Args:
        artifacts: Dictionary mapping artifact keys to file paths

    Returns:
        DataFrame with columns 'artifact' and 'path'
    """
    return pd.DataFrame(
        [
            {"artifact": "dataset.csv", "path": artifacts.get("dataset") or ""},
            {"artifact": "targets.csv", "path": artifacts.get("targets") or ""},
            {"artifact": "compounds.csv", "path": artifacts.get("compounds") or ""},
            {"artifact": "metadata.json", "path": artifacts.get("metadata") or ""},
            {
                "artifact": "molecule_features.csv",
                "path": artifacts.get("molecule_features") or "",
            },
            {"artifact": "model_rf.pkl", "path": artifacts.get("model") or ""},
            {
                "artifact": "model_metrics.json",
                "path": artifacts.get("model_metrics") or "",
            },
            {
                "artifact": "model_predictions.csv",
                "path": artifacts.get("model_predictions") or "",
            },
            {"artifact": "model_gnn.pt", "path": artifacts.get("model_gnn") or ""},
            {
                "artifact": "model_metrics_gnn.json",
                "path": artifacts.get("model_metrics_gnn") or "",
            },
            {
                "artifact": "model_predictions_gnn.csv",
                "path": artifacts.get("model_predictions_gnn") or "",
            },
            {"artifact": "encoder_gnn.pt", "path": artifacts.get("encoder_gnn") or ""},
            {
                "artifact": "encoder_gnn_config.json",
                "path": artifacts.get("encoder_gnn_config") or "",
            },
            {
                "artifact": "molecule_embeddings.npz",
                "path": artifacts.get("molecule_embeddings") or "",
            },
            {"artifact": "artifacts.zip", "path": artifacts.get("zip") or ""},
        ]
    )
