"""Model utility functions for listing and managing trained models."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from dta_gnn.io.runs import resolve_current_run_dir


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Seeds:
    - random
    - numpy
    - torch (if available)
    - torch.cuda (if available)
    - torch.backends.cudnn (deterministic)
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def list_available_models(run_dir: Path | None = None) -> dict[str, list[str]]:
    """List all available trained models in the run directory.

    Args:
        run_dir: Path to run directory. If None, attempts to resolve current run directory.

    Returns:
        Dictionary with keys: 'rf', 'svr', 'gnn'
        Each value is a list of model identifiers (for GNN: architecture names)
    """
    if run_dir is None:
        try:
            run_dir = resolve_current_run_dir()
        except FileNotFoundError:
            return {"rf": [], "svr": [], "gnn": []}

    if not run_dir or not run_dir.exists():
        return {"rf": [], "svr": [], "gnn": []}

    models = {"rf": [], "svr": [], "gnn": []}

    # Check for RandomForest model
    rf_model = run_dir / "model_rf.pkl"
    if rf_model.exists():
        models["rf"].append("RandomForest")

    # Check for SVR model
    svr_model = run_dir / "model_svr.pkl"
    if svr_model.exists():
        models["svr"].append("SVR")

    # Check for GNN models (model_gnn_<architecture>.pt)
    gnn_architectures = [
        "gin",
        "gcn",
        "gat",
        "sage",
        "pna",
        "transformer",
        "tag",
        "arma",
        "cheb",
        "supergat",
    ]
    for arch in gnn_architectures:
        gnn_model = run_dir / f"model_gnn_{arch}.pt"
        config_file = run_dir / f"encoder_{arch}_config.json"
        
        # Correction in previous write attempt: variable name typo fix
        if gnn_model.exists() and config_file.exists():
            # Format: "GNN (GIN)", "GNN (Transformer)", etc.
            arch_display = (
                arch.upper() if arch in ["gin", "gat", "pna"] else arch.capitalize()
            )
            models["gnn"].append(f"GNN ({arch_display})")

    return models
