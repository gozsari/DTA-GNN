from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

def _infer_task_type_from_labels(df: pd.DataFrame) -> Literal["classification", "regression"]:
    """Infer task type from label column values.
    
    Returns "classification" if labels are binary (only 0 and 1).
    Returns "regression" otherwise (continuous values or missing label column).
    """
    if "label" not in df.columns:
        return "regression"
    
    labels = df["label"].dropna()
    if len(labels) == 0:
        return "regression"
    
    unique_values = set(labels.unique())
    # Check if all values are binary (0 and/or 1)
    if unique_values.issubset({0, 1, 0.0, 1.0}):
        return "classification"
    
    return "regression"


@dataclass(frozen=True)
class RandomForestTrainResult:
    run_dir: Path
    task_type: Literal["classification", "regression"]
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    metrics: dict[str, Any]


def _morgan_fingerprints(
    smiles_list: list[str], *, radius: int = 2, n_bits: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, valid_mask) where X is (n_valid, n_bits)."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    try:
        from rdkit.Chem import rdFingerprintGenerator
    except Exception:  # pragma: no cover
        rdFingerprintGenerator = None  # type: ignore[assignment]

    morgan_gen = None
    if rdFingerprintGenerator is not None:
        try:
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius, fpSize=n_bits
            )
        except Exception:
            morgan_gen = None

    fps: list[np.ndarray] = []
    valid = np.zeros((len(smiles_list),), dtype=bool)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            if morgan_gen is not None:
                fp = morgan_gen.GetFingerprint(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, int(radius), nBits=int(n_bits)
                )
            arr = np.zeros((int(n_bits),), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid[i] = True
        except Exception:
            continue

    if not fps:
        return np.empty((0, int(n_bits)), dtype=np.int8), valid
    return np.asarray(fps, dtype=np.int8), valid


def train_random_forest_on_run(
    run_dir: str | Path,
    *,
    n_estimators: int = 500,
    random_seed: int = 42,
) -> RandomForestTrainResult:
    """Train a RandomForest baseline on `runs/<run>/dataset.csv` + `compounds.csv`.

    Writes:
    - model_rf.pkl
    - model_metrics.json
    - model_predictions.csv
    """

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from scipy.stats import pearsonr, spearmanr

    import joblib

    run_dir = Path(run_dir).resolve()
    dataset_path = run_dir / "dataset.csv"
    compounds_path = run_dir / "compounds.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset.csv in run folder: {dataset_path}")
    if not compounds_path.exists():
        raise FileNotFoundError(
            f"Missing compounds.csv in run folder: {compounds_path}"
        )

    df = pd.read_csv(dataset_path)
    compounds = pd.read_csv(compounds_path)

    required = {"molecule_chembl_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv missing columns: {sorted(missing)}")

    if "split" not in df.columns:
        raise ValueError(
            "dataset.csv missing 'split' column (expected train/val/test)."
        )

    # If dataset already contains SMILES, don't force a merge that can create
    # suffixes (smiles_x/smiles_y). Otherwise join from compounds.csv.
    smiles_candidates = [c for c in ["smiles", "canonical_smiles"] if c in df.columns]

    if not smiles_candidates:
        if "molecule_chembl_id" not in compounds.columns:
            raise ValueError("compounds.csv must contain 'molecule_chembl_id'.")
        comp_smiles_col = (
            "smiles"
            if "smiles" in compounds.columns
            else (
                "canonical_smiles" if "canonical_smiles" in compounds.columns else None
            )
        )
        if comp_smiles_col is None:
            raise ValueError(
                "compounds.csv must contain a SMILES column ('smiles' or 'canonical_smiles')."
            )

        df = df.merge(
            compounds[["molecule_chembl_id", comp_smiles_col]].rename(
                columns={comp_smiles_col: "smiles"}
            ),
            on="molecule_chembl_id",
            how="left",
        )
    else:
        # Normalize to a single column name.
        if "smiles" not in df.columns and "canonical_smiles" in df.columns:
            df = df.rename(columns={"canonical_smiles": "smiles"})

    # If we still ended up with suffixes (e.g., user-provided dataset.csv includes
    # smiles and we merged anyway in a previous run), coalesce them.
    if "smiles" not in df.columns:
        if "smiles_x" in df.columns and "smiles_y" in df.columns:
            df["smiles"] = df["smiles_x"].combine_first(df["smiles_y"])
        elif "smiles_x" in df.columns:
            df["smiles"] = df["smiles_x"]
        elif "smiles_y" in df.columns:
            df["smiles"] = df["smiles_y"]

    if "smiles" not in df.columns:
        raise ValueError(
            "Could not locate a SMILES column after merging dataset.csv and compounds.csv."
        )

    df = df.dropna(subset=["smiles", "label", "split"]).copy()
    if df.empty:
        raise ValueError(
            "No rows left after joining SMILES and dropping missing values."
        )

    task_type = _infer_task_type_from_labels(df)
    y_all = df["label"].astype(float).to_numpy()
    X_all, valid_mask = _morgan_fingerprints(
        df["smiles"].astype(str).tolist(), radius=2, n_bits=2048
    )
    if X_all.shape[0] == 0:
        raise ValueError("No valid SMILES to build fingerprints.")

    df_valid = df.loc[valid_mask].reset_index(drop=True)
    y_valid = y_all[valid_mask]
    X_valid = X_all

    train_mask = df_valid["split"].astype(str) == "train"
    val_mask = df_valid["split"].astype(str) == "val"
    test_mask = df_valid["split"].astype(str) == "test"
    if int(train_mask.sum()) < 2:
        raise ValueError("Not enough training rows after featurization.")

    X_train, y_train = X_valid[train_mask.values], y_valid[train_mask.values]

    from sklearn.ensemble import RandomForestClassifier

    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_seed),
            n_jobs=-1,
        )
        y_train = y_train.astype(int)
    else:
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(random_seed),
            n_jobs=-1,
        )

    model.fit(X_train, y_train)

    def _eval(split_name: str, mask: np.ndarray) -> dict[str, float] | None:
        if int(mask.sum()) == 0:
            return None
        Xs, ys = X_valid[mask], y_valid[mask]
        yhat = model.predict(Xs)
        
        if task_type == "classification":
            from sklearn.metrics import accuracy_score
            ys_int = ys.astype(int)
            accuracy = float(accuracy_score(ys_int, yhat))
            return {"accuracy": accuracy}
        else:
            rmse = float(np.sqrt(mean_squared_error(ys, yhat)))
            mae = float(mean_absolute_error(ys, yhat))
            r2 = float(r2_score(ys, yhat)) if len(np.unique(ys)) > 1 else 0.0
            pearson_r = float(pearsonr(ys, yhat)[0]) if len(np.unique(ys)) > 1 else None
            spearman_r = float(spearmanr(ys, yhat)[0]) if len(np.unique(ys)) > 1 else None
            return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson_r, "spearman_r": spearman_r}

    splits: dict[str, dict[str, float]] = {}
    for name, mask in (
        ("train", train_mask.values),
        ("val", val_mask.values),
        ("test", test_mask.values),
    ):
        m = _eval(name, mask)
        if m is not None:
            splits[name] = m

    metrics = {
        "model_type": "RandomForest",
        "task_type": task_type,
        "splits": splits,
    }

    # Save artifacts
    model_path = run_dir / "model_rf.pkl"
    metrics_path = run_dir / "model_metrics.json"
    predictions_path = run_dir / "model_predictions.csv"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Predictions file (val/test rows; train is usually too large for UI preview)
    pred_rows = df_valid.loc[val_mask | test_mask].copy()
    if not pred_rows.empty:
        X_pred = X_valid[(val_mask | test_mask).values]
        y_pred = model.predict(X_pred)
        pred_rows["y_pred"] = y_pred
    pred_rows.to_csv(predictions_path, index=False)

    return RandomForestTrainResult(
        run_dir=run_dir,
        task_type=task_type,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        metrics=metrics,
    )
