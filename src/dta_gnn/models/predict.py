"""Model inference/prediction on new molecules."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    """Result from model prediction."""

    predictions: pd.DataFrame
    model_type: Literal["RandomForest", "SVR", "GNN"]
    model_path: str
    run_dir: Path


def predict_with_svr(
    run_dir: Path,
    smiles_list: list[str],
    molecule_ids: list[str] | None = None,
) -> PredictionResult:
    """Predict using a trained SVR model.

    Expects `model_svr.pkl` in the run directory.
    Uses Morgan (ECFP4) fingerprints with radius=2, nBits=2048.
    """

    import joblib
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    run_dir = Path(run_dir).resolve()
    model_path = run_dir / "model_svr.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"SVR model not found: {model_path}")

    model = joblib.load(model_path)

    if molecule_ids is None:
        molecule_ids = [f"mol_{i}" for i in range(len(smiles_list))]
    if len(molecule_ids) != len(smiles_list):
        raise ValueError(
            f"Length mismatch: {len(smiles_list)} SMILES but {len(molecule_ids)} IDs"
        )

    def _fp(smi: str) -> np.ndarray | None:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except Exception:
            return None

    fps: list[np.ndarray] = []
    valid_idx: list[int] = []
    for idx, smi in enumerate(smiles_list):
        arr = _fp(smi)
        if arr is not None:
            fps.append(arr)
            valid_idx.append(idx)

    if not fps:
        raise ValueError("No valid SMILES to generate predictions.")

    X = np.asarray(fps, dtype=np.float32)
    y_pred = model.predict(X).astype(float)

    results: list[dict[str, object]] = []
    for i, orig_idx in enumerate(valid_idx):
        results.append(
            {
                "molecule_id": molecule_ids[orig_idx],
                "smiles": smiles_list[orig_idx],
                "prediction": float(y_pred[i]),
            }
        )

    for idx in range(len(smiles_list)):
        if idx not in valid_idx:
            results.append(
                {
                    "molecule_id": molecule_ids[idx],
                    "smiles": smiles_list[idx],
                    "prediction": None,
                }
            )

    df = pd.DataFrame(results)
    return PredictionResult(
        predictions=df,
        model_type="SVR",
        model_path=str(model_path),
        run_dir=run_dir,
    )


def predict_with_random_forest(
    run_dir: Path,
    smiles_list: list[str],
    molecule_ids: list[str] | None = None,
) -> PredictionResult:
    """Predict using a trained RandomForest model.

    Args:
        run_dir: Directory containing the trained model (model_rf.pkl)
        smiles_list: List of SMILES strings to predict
        molecule_ids: Optional list of molecule IDs (defaults to mol_0, mol_1, ...)

    Returns:
        PredictionResult with predictions DataFrame
    """
    import joblib
    from rdkit import Chem
    from rdkit.Chem import AllChem

    run_dir = Path(run_dir).resolve()
    model_path = run_dir / "model_rf.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"RandomForest model not found: {model_path}")

    # Load model
    model = joblib.load(model_path)

    # Generate molecule IDs if not provided
    if molecule_ids is None:
        molecule_ids = [f"mol_{i}" for i in range(len(smiles_list))]

    if len(molecule_ids) != len(smiles_list):
        raise ValueError(
            f"Length mismatch: {len(smiles_list)} SMILES but {len(molecule_ids)} IDs"
        )

    # Build fingerprints
    def get_fp(smi: str) -> np.ndarray | None:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            return np.array(fp)
        except Exception:
            return None

    fps = []
    valid_idx = []
    for idx, smi in enumerate(smiles_list):
        fp = get_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(idx)

    if not fps:
        raise ValueError("No valid SMILES to generate predictions.")

    X = np.array(fps)

    # Predict
    try:
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
    except Exception:
        # Regression case
        preds = model.predict(X)
        proba = preds  # For regression, use prediction as score

    # Build results DataFrame
    results = []
    for i, orig_idx in enumerate(valid_idx):
        results.append(
            {
                "molecule_id": molecule_ids[orig_idx],
                "smiles": smiles_list[orig_idx],
                "prediction": float(preds[i]),
            }
        )

    # Add failed molecules
    for idx in range(len(smiles_list)):
        if idx not in valid_idx:
            results.append(
                {
                    "molecule_id": molecule_ids[idx],
                    "smiles": smiles_list[idx],
                    "prediction": None,
                }
            )

    df = pd.DataFrame(results)

    return PredictionResult(
        predictions=df,
        model_type="RandomForest",
        model_path=str(model_path),
        run_dir=run_dir,
    )


def predict_with_gnn(
    run_dir: Path,
    smiles_list: list[str],
    molecule_ids: list[str] | None = None,
    batch_size: int = 64,
    device: str | None = None,
    architecture: str | None = None,
) -> PredictionResult:
    """Predict using a trained GNN model.

    Args:
        run_dir: Directory containing the trained model (model_gnn_<architecture>.pt)
        smiles_list: List of SMILES strings to predict
        molecule_ids: Optional list of molecule IDs (defaults to mol_0, mol_1, ...)
        batch_size: Batch size for inference

    Returns:
        PredictionResult with predictions DataFrame
    """
    from dta_gnn.features.molecule_graphs import smiles_to_graph_2d
    from dta_gnn.models.gnn import _make_encoder_and_model, _get_device, _load_json

    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ImportError:
        raise ImportError(
            "PyTorch Geometric not installed. Install with: pip install 'dta_gnn[molecule-gnn]'"
        )

    run_dir = Path(run_dir).resolve()
    
    # Try to find model file by checking common architecture names
    model_path = None
    config_path = None
    arch_name = None
    
    # If architecture is explicitly provided, use it
    if architecture:
        arch_name = str(architecture).strip().lower()
        model_path = run_dir / f"model_gnn_{arch_name}.pt"
        config_path = run_dir / f"encoder_{arch_name}_config.json"
    
    # If not provided or files don't exist, try to detect
    if not arch_name or not model_path.exists():
        # First, try to get architecture from metadata.json if it exists
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            try:
                meta = _load_json(metadata_path) or {}
                arch_name = str(
                    (meta.get("model", {}) or {}).get("architecture", "gin")
                ).strip().lower()
            except Exception:
                pass
        
        # If not found, try common architectures
        if not arch_name:
            for arch in ["gin", "gcn", "gat", "sage", "pna", "transformer", "tag", "arma", "cheb", "supergat"]:
                test_model = run_dir / f"model_gnn_{arch}.pt"
                test_config = run_dir / f"encoder_{arch}_config.json"
                if test_model.exists() and test_config.exists():
                    model_path = test_model
                    config_path = test_config
                    arch_name = arch
                    break
        
        # Fallback to gin if nothing found
        if not model_path or not model_path.exists():
            arch_name = "gin"
            model_path = run_dir / f"model_gnn_{arch_name}.pt"
            config_path = run_dir / f"encoder_{arch_name}_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"GNN model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"GNN config not found: {config_path}")

    # Load model config
    with open(config_path, "r") as f:
        config = json.load(f)

    encoder_cfg = config.get("encoder") or {}
    feat_cfg = config.get("featurizer") or {}

    enc_type = str((encoder_cfg or {}).get("type") or "")
    architecture = str((encoder_cfg or {}).get("architecture") or "")
    if not architecture and enc_type in {"gin_pyg", "gin"}:
        architecture = "gin"
    if not architecture:
        architecture = "gin"
    
    # Use architecture from config if arch_name wasn't found earlier
    if not arch_name:
        arch_name = architecture.strip().lower()
        # Update paths if we found architecture from config
        if not model_path.exists():
            model_path = run_dir / f"model_gnn_{arch_name}.pt"
            config_path = run_dir / f"encoder_{arch_name}_config.json"

    pooling = str((encoder_cfg or {}).get("pooling") or "add")
    if pooling not in {"add", "mean", "max", "attention"}:
        pooling = "add"
    residual = bool((encoder_cfg or {}).get("residual") or False)
    head_mlp_layers = int((encoder_cfg or {}).get("head_mlp_layers") or 2)
    gin_conv_mlp_layers = int((encoder_cfg or {}).get("gin_conv_mlp_layers") or 2)
    gin_train_eps = bool((encoder_cfg or {}).get("gin_train_eps") or False)
    gin_eps = float((encoder_cfg or {}).get("gin_eps") or 0.0)
    gat_heads = int((encoder_cfg or {}).get("gat_heads") or 4)
    sage_aggr = str((encoder_cfg or {}).get("sage_aggr") or "mean")
    pna_deg_hist = (encoder_cfg or {}).get("pna_deg_hist")

    pna_deg = None
    if architecture.strip().lower() == "pna":
        if isinstance(pna_deg_hist, list) and pna_deg_hist:
            pna_deg = torch.tensor([int(x) for x in pna_deg_hist], dtype=torch.long)
        else:
            raise ValueError(
                "encoder_config.json missing pna_deg_hist needed for PNA inference."
            )

    edge_dim = int((feat_cfg or {}).get("bond_feat_dim") or 6)
    num_atom_types = int((feat_cfg or {}).get("atom_type_vocab") or 101)

    # Reconstruct model architecture
    _, GinPredictor = _make_encoder_and_model(
        embedding_dim=int((encoder_cfg or {}).get("embedding_dim") or 128),
        hidden_dim=int((encoder_cfg or {}).get("hidden_dim") or 128),
        num_layers=int((encoder_cfg or {}).get("num_layers") or 5),
        dropout=float((encoder_cfg or {}).get("dropout") or 0.0),
        edge_dim=edge_dim,
        num_atom_types=num_atom_types,
        architecture=architecture,
        pooling=pooling,
        residual=residual,
        head_mlp_layers=head_mlp_layers,
        gin_conv_mlp_layers=gin_conv_mlp_layers,
        gin_train_eps=gin_train_eps,
        gin_eps=gin_eps,
        gat_heads=gat_heads,
        sage_aggr=sage_aggr,
        pna_deg=pna_deg,
    )

    model = GinPredictor()
    # Load state dict and move to device
    device_obj = _get_device(device)
    
    # TransformerConv doesn't support MPS (scatter_reduce not implemented)
    # Fall back to CPU if transformer architecture is used on MPS
    if architecture.strip().lower() == "transformer" and str(device_obj) == "mps":
        import warnings
        import sys
        import os
        
        # Only show warning if not running in pytest (to reduce test noise)
        is_pytest = (
            "pytest" in sys.modules
            or os.environ.get("PYTEST_CURRENT_TEST") is not None
            or any("pytest" in arg for arg in sys.argv)
        )
        
        if not is_pytest:
            warnings.warn(
                "TransformerConv doesn't support MPS. Falling back to CPU. "
                "For better performance, use device='cpu' explicitly.",
                UserWarning,
            )
        device_obj = torch.device("cpu")
    
    state_dict = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(state_dict)
    model.to(device_obj)
    model.eval()

    # Generate molecule IDs if not provided
    if molecule_ids is None:
        molecule_ids = [f"mol_{i}" for i in range(len(smiles_list))]

    if len(molecule_ids) != len(smiles_list):
        raise ValueError(
            f"Length mismatch: {len(smiles_list)} SMILES but {len(molecule_ids)} IDs"
        )

    # Build graphs
    graphs = []
    valid_idx = []
    for idx, smi in enumerate(smiles_list):
        try:
            g = smiles_to_graph_2d(molecule_chembl_id=molecule_ids[idx], smiles=smi)
            if g is not None:
                data = Data(
                    atom_type=torch.from_numpy(g.atom_type.astype(np.int64)),
                    atom_feat=torch.from_numpy(g.atom_feat.astype(np.float32)),
                    edge_index=torch.from_numpy(g.edge_index.astype(np.int64)),
                    edge_attr=torch.from_numpy(g.edge_attr.astype(np.float32)),
                )
                graphs.append(data)
                valid_idx.append(idx)
        except Exception:
            continue

    if not graphs:
        raise ValueError("No valid SMILES to generate predictions.")

    # Predict
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_probs = []

    # Task type is always regression for DTA
    task_type = "regression"

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device_obj)
            logits, _z = model(
                batch.atom_type,
                batch.atom_feat,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            y_score = logits.view(-1).float().cpu().numpy()

            
            all_preds.extend(y_score.tolist())
            all_probs.extend(y_score.tolist())

    # Build results DataFrame
    results = []
    for i, orig_idx in enumerate(valid_idx):
        results.append(
            {
                "molecule_id": molecule_ids[orig_idx],
                "smiles": smiles_list[orig_idx],
                "prediction": float(all_preds[i]),
            }
        )

    # Add failed molecules
    for idx in range(len(smiles_list)):
        if idx not in valid_idx:
            results.append(
                {
                    "molecule_id": molecule_ids[idx],
                    "smiles": smiles_list[idx],
                    "prediction": None,
                }
            )

    df = pd.DataFrame(results)

    return PredictionResult(
        predictions=df,
        model_type="GNN",
        model_path=str(model_path),
        run_dir=run_dir,
    )


# Backwards-compatible alias (historical name)
predict_with_gin = predict_with_gnn
