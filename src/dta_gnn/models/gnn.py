"""Graph Neural Network models for Drug-Target Affinity prediction.

Supports multiple GNN architectures: GIN, GCN, GAT, GraphSAGE, and PNA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from dta_gnn import __version__
from dta_gnn.features.molecule_graphs import build_graphs_2d, smiles_to_graph_2d
from dta_gnn.models.utils import set_seed


def _require_pyg() -> None:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "GNN features require optional dependencies. "
            "Install with: `pip install 'dta_gnn[molecule-gnn]'`"
        ) from e


def _get_device(device: str | None = None) -> Any:
    """Get the best available device for PyTorch.

    Priority: MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU

    Args:
        device: Device string ("mps", "cuda", "cpu", or "auto"/None for auto-detection)

    Returns:
        torch.device object
    """
    _require_pyg()
    import torch

    if device is None or device == "auto":
        # Auto-detect: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        device_str = str(device).lower()
        if device_str == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS (Apple Silicon GPU) is not available. "
                    "Falling back to CPU. Use device='cpu' or device='cuda'."
                )
            return torch.device("mps")
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. "
                    "Falling back to CPU. Use device='cpu' or device='mps'."
                )
            return torch.device("cuda")
        elif device_str == "cpu":
            return torch.device("cpu")
        else:
            raise ValueError(
                f"Invalid device: {device}. Must be 'auto', 'mps', 'cuda', or 'cpu'."
            )


def _resolve_run_dir(run_dir: str | Path) -> Path:
    p = Path(run_dir)
    if p.name == "current":
        try:
            p = p.resolve()
        except Exception:
            pass
    return p


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@dataclass(frozen=True)
class GnnTrainConfig:
    architecture: Literal[
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
    ] = "gin"
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 5
    dropout: float = 0.1
    pooling: Literal["add", "mean", "max", "attention"] = "add"
    residual: bool = False
    head_mlp_layers: int = 2
    # GIN-specific
    gin_conv_mlp_layers: int = 2
    gin_train_eps: bool = False
    gin_eps: float = 0.0
    # GAT-specific
    gat_heads: int = 4
    # GraphSAGE-specific
    sage_aggr: str = "mean"
    # Transformer-specific
    transformer_heads: int = 4
    transformer_edge_dim: int | None = None
    # TAG-specific
    tag_k: int = 2
    # ARMA-specific
    arma_num_stacks: int = 1
    arma_num_layers: int = 1
    # Cheb-specific
    cheb_k: int = 2
    # SuperGAT-specific
    supergat_heads: int = 4
    supergat_attention_type: str = "MX"  # "MX" (Mixed) or "SD" (Self-Distillation)
    # Training
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 10
    random_seed: int = 42
    device: str | None = None  # "auto", "mps", "cuda", "cpu", or None (auto)


# Backwards-compatible alias (historical name)
GinTrainConfig = GnnTrainConfig


@dataclass(frozen=True)
class GnnTrainResult:
    run_dir: Path
    task_type: Literal["regression"]
    model_path: Path
    encoder_path: Path
    encoder_config_path: Path
    metrics_path: Path
    predictions_path: Path
    metrics: dict[str, Any]
    best_epoch: int | None = None  # Epoch number where best model was saved


# Backwards-compatible alias (historical name)
GinTrainResult = GnnTrainResult


@dataclass(frozen=True)
class GnnEmbeddingExtractResult:
    run_dir: Path
    embeddings_path: Path
    n_molecules: int
    embedding_dim: int


# Backwards-compatible alias (historical name)
GinEmbeddingExtractResult = GnnEmbeddingExtractResult


def _infer_task_type(
    df: pd.DataFrame, meta: dict[str, Any] | None
) -> Literal["regression"]:
    return "regression"


def _graphs_to_pyg_data(
    *,
    mol_ids: list[str],
    smiles_list: list[str],
    labels: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    _require_pyg()
    import torch
    from torch_geometric.data import Data

    graphs = build_graphs_2d(
        molecules=list(zip(mol_ids, smiles_list)), drop_failures=True
    )
    by_id: dict[str, Data] = {}
    kept_ids: list[str] = []

    # Pre-encode labels map
    y_by_id = {str(mid): float(y) for mid, y in zip(mol_ids, labels)}

    for g in graphs:
        mid = str(g.molecule_chembl_id)
        if mid not in y_by_id:
            continue

        # x = [atom_type_embedding_input, numeric atom_feat]
        atom_type = torch.from_numpy(g.atom_type.astype(np.int64))
        atom_feat = torch.from_numpy(g.atom_feat.astype(np.float32))
        edge_index = torch.from_numpy(g.edge_index.astype(np.int64))
        edge_attr = torch.from_numpy(g.edge_attr.astype(np.float32))

        data = Data(
            atom_type=atom_type,
            atom_feat=atom_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([y_by_id[mid]], dtype=torch.float32),
            molecule_chembl_id=mid,
        )
        by_id[mid] = data
        kept_ids.append(mid)

    return by_id, kept_ids


def _make_encoder_and_model(
    *,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    edge_dim: int,
    num_atom_types: int = 101,
    architecture: str = "gin",
    pooling: str = "add",
    residual: bool = False,
    head_mlp_layers: int = 2,
    gin_conv_mlp_layers: int = 2,
    gin_train_eps: bool = False,
    gin_eps: float = 0.0,
    gat_heads: int = 4,
    sage_aggr: str = "mean",
    pna_deg: Any | None = None,
    transformer_heads: int = 4,
    transformer_edge_dim: int | None = None,
    tag_k: int = 2, 
    arma_num_stacks: int = 1,
    arma_num_layers: int = 1,
    cheb_k: int = 2,
    supergat_heads: int = 4,
    supergat_attention_type: str = "MX",
):
    _require_pyg()
    import torch
    import torch.nn as nn
    from torch_geometric.nn import (
        GATConv,
        GCNConv,
        GINEConv,
        PNAConv,
        SAGEConv,
        TransformerConv,
        TAGConv,
        ARMAConv,
        ChebConv,
        SuperGATConv,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )
    # Use AttentionalAggregation instead of deprecated GlobalAttention
    try:
        from torch_geometric.nn.aggr import AttentionalAggregation
        _use_attentional_agg = True
    except ImportError:
        # Fallback for older PyG versions
        from torch_geometric.nn import GlobalAttention
        _use_attentional_agg = False

    arch = (architecture or "gin").strip().lower()
    pool = (pooling or "add").strip().lower()
    if pool not in {"add", "mean", "max", "attention"}:
        pool = "add"

    residual = bool(residual)
    head_mlp_layers = max(1, int(head_mlp_layers))
    gin_conv_mlp_layers = max(1, int(gin_conv_mlp_layers))

    class GnnEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)
            self.in_proj = nn.Linear(hidden_dim + 6, hidden_dim)

            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                if arch == "gin":
                    layers: list[nn.Module] = []
                    for i in range(gin_conv_mlp_layers):
                        layers.append(nn.Linear(hidden_dim, hidden_dim))
                        if i < gin_conv_mlp_layers - 1:
                            layers.append(nn.ReLU())
                    mlp = nn.Sequential(*layers)
                    self.convs.append(
                        GINEConv(
                            nn=mlp,
                            edge_dim=edge_dim,
                            eps=float(gin_eps),
                            train_eps=bool(gin_train_eps),
                        )
                    )
                elif arch == "gcn":
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                elif arch == "gat":
                    self.convs.append(
                        GATConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            heads=int(gat_heads),
                            concat=False,
                            dropout=float(dropout),
                            edge_dim=edge_dim,
                        )
                    )
                elif arch == "sage":
                    self.convs.append(
                        SAGEConv(hidden_dim, hidden_dim, aggr=str(sage_aggr))
                    )
                elif arch == "pna":
                    if pna_deg is None:
                        raise ValueError(
                            "PNA requires a degree histogram tensor (pna_deg)."
                        )
                    self.convs.append(
                        PNAConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            aggregators=["mean", "min", "max", "std"],
                            scalers=["identity", "amplification", "attenuation"],
                            deg=pna_deg,
                            edge_dim=edge_dim,
                            towers=1,
                            pre_layers=1,
                            post_layers=1,
                            divide_input=False,
                        )
                    )
                elif arch == "transformer":
                    self.convs.append(
                        TransformerConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            heads=int(transformer_heads),
                            concat=False,  # Keep output dimension as hidden_dim
                            edge_dim=edge_dim
                            if transformer_edge_dim is None
                            else transformer_edge_dim,
                            dropout=float(dropout),
                        )
                    )
                elif arch == "tag":
                    self.convs.append(
                        TAGConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            K=int(tag_k),
                            bias=True,
                        )
                    )
                elif arch == "arma":
                    self.convs.append(
                        ARMAConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            num_stacks=int(arma_num_stacks),
                            num_layers=int(arma_num_layers),
                            dropout=float(dropout),
                        )
                    )
                elif arch == "cheb":
                    self.convs.append(
                        ChebConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            K=int(cheb_k),
                            bias=True,
                        )
                    )
                elif arch == "supergat":
                    self.convs.append(
                        SuperGATConv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            heads=int(supergat_heads),
                            concat=False,  # Keep output dimension consistent
                            attention_type=str(supergat_attention_type),
                            dropout=float(dropout),
                        )
                    )
                else:
                    raise ValueError(f"Unknown GNN architecture: {architecture}")
                self.norms.append(nn.BatchNorm1d(hidden_dim))

            self.attn_pool = None
            if pool == "attention":
                if _use_attentional_agg:
                    # Use new AttentionalAggregation API
                    self.attn_pool = AttentionalAggregation(
                        gate_nn=nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1),
                        )
                    )
                else:
                    # Fallback to deprecated GlobalAttention for older PyG versions
                    self.attn_pool = GlobalAttention(
                        gate_nn=nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1),
                        )
                    )

            self.dropout = float(dropout)
            self.out_proj = nn.Linear(hidden_dim, embedding_dim)

        def forward(self, atom_type, atom_feat, edge_index, edge_attr, batch):
            # atom_type: [N], atom_feat: [N,6]
            atom_type = atom_type.clamp(min=0, max=num_atom_types - 1)
            h = torch.cat([self.atom_emb(atom_type), atom_feat], dim=-1)
            h = self.in_proj(h)
            h = torch.relu(h)

            for conv, norm in zip(self.convs, self.norms):
                h_in = h
                if arch == "gin":
                    h = conv(h, edge_index, edge_attr)
                elif arch == "gat":
                    h = conv(h, edge_index, edge_attr)
                elif arch == "pna":
                    h = conv(h, edge_index, edge_attr)
                elif arch == "transformer":
                    h = conv(h, edge_index, edge_attr)
                else:
                    h = conv(h, edge_index)

                if residual:
                    h = h + h_in

                h = norm(h)
                h = torch.relu(h)
                h = torch.dropout(h, p=self.dropout, train=self.training)

            if pool == "attention" and self.attn_pool is not None:
                g = self.attn_pool(h, batch)
            elif pool == "mean":
                g = global_mean_pool(h, batch)
            elif pool == "max":
                g = global_max_pool(h, batch)
            else:
                g = global_add_pool(h, batch)
            g = self.out_proj(g)
            return g

    class GnnPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GnnEncoder()

            head: list[nn.Module] = []
            if head_mlp_layers <= 1:
                head.append(nn.Linear(embedding_dim, 1))
            else:
                for i in range(head_mlp_layers - 1):
                    head.append(nn.Linear(embedding_dim, embedding_dim))
                    head.append(nn.ReLU())
                    head.append(nn.Dropout(p=float(dropout)))
                head.append(nn.Linear(embedding_dim, 1))
            self.head = nn.Sequential(*head)

        def forward(self, atom_type, atom_feat, edge_index, edge_attr, batch):
            z = self.encoder(atom_type, atom_feat, edge_index, edge_attr, batch)
            out = self.head(z).squeeze(-1)
            return out, z

    return GnnEncoder, GnnPredictor


def train_gnn_on_run(
    run_dir: str | Path,
    *,
    config: GnnTrainConfig | None = None,
    wandb_run=None,
) -> GnnTrainResult:
    """Train a 2D GNN model on the run's dataset.csv, using molecule graphs from compounds.csv.

    Writes:
      - model_gnn_<architecture>.pt
      - encoder_<architecture>.pt
      - encoder_<architecture>_config.json
      - model_metrics_gnn_<architecture>.json
      - model_predictions_gnn_<architecture>.csv
    """

    _require_pyg()
    import torch
    from torch_geometric.loader import DataLoader

    cfg = config or GnnTrainConfig()
    set_seed(cfg.random_seed)

    run_path = _resolve_run_dir(run_dir)
    dataset_path = run_path / "dataset.csv"
    compounds_path = run_path / "compounds.csv"

    if not dataset_path.exists():
        raise ValueError(f"Missing dataset.csv in run folder: {dataset_path}")
    if not compounds_path.exists():
        raise ValueError(f"Missing compounds.csv in run folder: {compounds_path}")

    df = pd.read_csv(dataset_path)
    required = {"molecule_chembl_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset.csv missing columns: {sorted(missing)}")

    df_comp = pd.read_csv(compounds_path)
    if "molecule_chembl_id" not in df_comp.columns or "smiles" not in df_comp.columns:
        raise ValueError(
            "compounds.csv must contain 'molecule_chembl_id' and 'smiles'."
        )

    meta = _load_json(run_path / "metadata.json")
    task_type = _infer_task_type(df, meta)

    # Map molecule id -> smiles
    df_comp_clean = (
        df_comp.dropna(subset=["molecule_chembl_id", "smiles"])
        .drop_duplicates(subset=["molecule_chembl_id"], keep="first")
    )
    smiles_map = (
        df_comp_clean.set_index("molecule_chembl_id")["smiles"]
        .astype(str)
        .to_dict()
    )

    df = df.copy()
    df["_mid"] = df["molecule_chembl_id"].astype(str)
    df["_smiles"] = df["_mid"].map(smiles_map)
    df = df.dropna(subset=["_smiles"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after joining SMILES from compounds.csv")

    # Build one graph per dataset row (simple MVP, matches current RF 'molecule-only' baseline).
    # Keep df aligned by building graphs row-by-row.
    from torch_geometric.data import Data

    data_list: list[Data] = []

    for _, row in df.iterrows():
        mid = str(row["_mid"])
        smi = str(row["_smiles"])
        g = smiles_to_graph_2d(molecule_chembl_id=mid, smiles=smi)
        if g is None:
            continue
        data_list.append(
            Data(
                atom_type=torch.from_numpy(g.atom_type.astype(np.int64)),
                atom_feat=torch.from_numpy(g.atom_feat.astype(np.float32)),
                edge_index=torch.from_numpy(g.edge_index.astype(np.int64)),
                edge_attr=torch.from_numpy(g.edge_attr.astype(np.float32)),
                y=torch.tensor(float(row["label"]), dtype=torch.float32),
                split=str(row["split"]),
                molecule_chembl_id=str(row.get("molecule_chembl_id", "")),
                target_chembl_id=str(row.get("target_chembl_id", "")),
                num_nodes=int(g.atom_feat.shape[0]),
            )
        )

    if not data_list:
        raise ValueError("No valid SMILES could be converted to graphs.")

    train_data = [d for d in data_list if getattr(d, "split") == "train"]
    val_data = [d for d in data_list if getattr(d, "split") == "val"]
    test_data = [d for d in data_list if getattr(d, "split") == "test"]

    if not train_data:
        raise ValueError("No training rows found (split == 'train').")

    edge_dim = (
        int(train_data[0].edge_attr.shape[1]) if train_data[0].edge_attr.numel() else 6
    )

    pna_deg = None
    pna_deg_list: list[int] | None = None
    if str(cfg.architecture).strip().lower() == "pna":
        from torch_geometric.utils import degree

        # Build a degree histogram across the training graphs.
        # PNAConv expects a histogram tensor where index i holds the count of nodes with degree i.
        hist: torch.Tensor | None = None
        for d in train_data:
            deg_vec = degree(
                d.edge_index[1], num_nodes=int(d.num_nodes), dtype=torch.long
            )
            deg_hist = torch.bincount(deg_vec, minlength=int(deg_vec.max().item()) + 1)
            hist = (
                deg_hist
                if hist is None
                else (
                    torch.nn.functional.pad(
                        hist, (0, max(0, deg_hist.numel() - hist.numel()))
                    )
                    + torch.nn.functional.pad(
                        deg_hist, (0, max(0, hist.numel() - deg_hist.numel()))
                    )
                )
            )

        pna_deg = (
            hist if hist is not None else torch.tensor([0], dtype=torch.long)
        ).cpu()
        pna_deg_list = [int(x) for x in pna_deg.tolist()]

    GnnEncoder, GnnPredictor = _make_encoder_and_model(
        embedding_dim=int(cfg.embedding_dim),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
        edge_dim=edge_dim,
        architecture=str(cfg.architecture),
        pooling=str(cfg.pooling),
        residual=bool(cfg.residual),
        head_mlp_layers=int(cfg.head_mlp_layers),
        gin_conv_mlp_layers=int(cfg.gin_conv_mlp_layers),
        gin_train_eps=bool(cfg.gin_train_eps),
        gin_eps=float(cfg.gin_eps),
        gat_heads=int(cfg.gat_heads),
        sage_aggr=str(cfg.sage_aggr),
        pna_deg=pna_deg,
        transformer_heads=int(cfg.transformer_heads),
        transformer_edge_dim=cfg.transformer_edge_dim,
        tag_k=int(cfg.tag_k),
        arma_num_stacks=int(cfg.arma_num_stacks),
        arma_num_layers=int(cfg.arma_num_layers),
        cheb_k=int(cfg.cheb_k),
        supergat_heads=int(cfg.supergat_heads),
        supergat_attention_type=str(cfg.supergat_attention_type),
    )

    model = GnnPredictor()
    device = _get_device(cfg.device)
    
    # TransformerConv doesn't support MPS (scatter_reduce not implemented)
    # Fall back to CPU if transformer architecture is used on MPS
    if str(cfg.architecture).strip().lower() == "transformer" and str(device) == "mps":
        import torch
        import warnings
        import sys
        import os
        
        # Only show warning if not running in pytest (to reduce test noise)
        # Check both sys.modules and environment variable
        is_pytest = (
            "pytest" in sys.modules
            or os.environ.get("PYTEST_CURRENT_TEST") is not None
            or any("pytest" in arg for arg in sys.argv)
        )
        
        if not is_pytest:
            warnings.warn(
                "TransformerConv doesn't support MPS. Falling back to CPU. "
                "For better performance, use device='cpu' explicitly or use a different architecture.",
                UserWarning,
            )
        device = torch.device("cpu")
    
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(train_data, batch_size=int(cfg.batch_size), shuffle=True)
    val_loader = (
        DataLoader(val_data, batch_size=int(cfg.batch_size), shuffle=False)
        if val_data
        else None
    )
    test_loader = (
        DataLoader(test_data, batch_size=int(cfg.batch_size), shuffle=False)
        if test_data
        else None
    )

    torch.manual_seed(int(cfg.random_seed))

    def _compute_val_metrics(model, val_loader, device, criterion):
        """Compute validation metrics during training."""
        model.eval()
        val_losses = []
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(
                    batch.atom_type,
                    batch.atom_feat,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                y = batch.y.view(-1).float()
                loss = criterion(logits.view(-1), y)
                val_losses.append(loss.item())
                
                y_true_list.extend(y.cpu().numpy().tolist())
                y_pred_list.extend(logits.view(-1).cpu().numpy().tolist())
        
        model.train()
        
        if not y_true_list:
            return None
        
        yt = np.asarray(y_true_list, dtype=float)
        yp = np.asarray(y_pred_list, dtype=float)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import pearsonr, spearmanr
        
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp)) if yt.size >= 2 else None
        pearson_r = float(pearsonr(yt, yp)[0]) if yt.size >= 2 else None
        spearman_r = float(spearmanr(yt, yp)[0]) if yt.size >= 2 else None
        
        return {
            "loss": sum(val_losses) / len(val_losses) if val_losses else 0.0,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
        }

    # Initialize best model tracking
    best_val_score = float('-inf')
    best_epoch = -1
    best_model_state = None
    best_encoder_state = None
    best_val_metrics = None  # Store best validation metrics for logging

    print(f"Training GNN for {cfg.epochs} epochs (batch_size={cfg.batch_size}, lr={cfg.lr:.6f})...")
    for epoch in range(int(cfg.epochs)):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _z = model(
                batch.atom_type,
                batch.atom_feat,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            y = batch.y.view(-1).float()
            loss = criterion(logits.view(-1), y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        
        # Evaluate on validation set if available
        val_loss = None
        val_metrics = None
        if val_loader is not None:
            val_metrics = _compute_val_metrics(model, val_loader, device, criterion)
            if val_metrics:
                val_loss = val_metrics["loss"]
                
                # Use R² if available, otherwise use -RMSE (so higher is better)
                current_score = val_metrics["r2"] if val_metrics["r2"] is not None else -val_metrics["rmse"]
                
                if current_score > best_val_score:
                    best_val_score = current_score
                    best_epoch = epoch + 1
                    # Deep-copy state so later epochs don't overwrite it (state_dict().copy() is shallow)
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_encoder_state = {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()}
                    best_val_metrics = val_metrics.copy()
                    score_str = f"R²={val_metrics['r2']:.4f}" if val_metrics["r2"] is not None else f"RMSE={val_metrics['rmse']:.4f}"
                    print(f"  ✓ New best model at epoch {best_epoch} (val_{score_str})")
        
        # Log to wandb if available
        if wandb_run is not None:
            log_dict = {
                "train/loss": avg_loss,
                "epoch": epoch + 1,
            }
            if val_loss is not None:
                log_dict["val/loss"] = val_loss
            if val_metrics is not None:
                log_dict["val/rmse"] = val_metrics["rmse"]
                log_dict["val/mae"] = val_metrics["mae"]
                log_dict["val/pearson_r"] = val_metrics["pearson_r"]
                log_dict["val/spearman_r"] = val_metrics["spearman_r"]
                if val_metrics["r2"] is not None:
                    log_dict["val/r2"] = val_metrics["r2"]
            wandb_run.log(log_dict)
        
        if (epoch + 1) % max(1, int(cfg.epochs) // 5) == 0 or epoch == 0:
            val_str = f", val_loss: {val_loss:.4f}" if val_loss is not None else ""
            print(f"  Epoch {epoch + 1}/{cfg.epochs} completed (train_loss: {avg_loss:.4f}{val_str})")

    # Load best model if available
    if best_model_state is not None and val_loader is not None and best_val_metrics is not None:
        score_str = f"R²={best_val_metrics['r2']:.4f}" if best_val_metrics.get("r2") is not None else f"RMSE={best_val_metrics['rmse']:.4f}"
        print(f"Loading best model from epoch {best_epoch} (val_{score_str})...")
        model.load_state_dict(best_model_state)
        model.encoder.load_state_dict(best_encoder_state)
    else:
        print("Using final epoch model (no validation set for checkpointing)")

    print("Training complete. Evaluating...")
    # Evaluation + predictions
    model.eval()

    def _predict(loader):
        ys: list[float] = []
        preds: list[float] = []
        probs: list[float] = []
        mols: list[str] = []
        targs: list[str] = []
        splits: list[str] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits, _z = model(
                    batch.atom_type,
                    batch.atom_feat,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                )
                y_true = batch.y.view(-1).float().cpu().numpy()
                y_score = logits.view(-1).float().cpu().numpy()

                ys.extend(y_true.tolist())
                splits.extend([str(s) for s in batch.split])
                mols.extend([str(m) for m in batch.molecule_chembl_id])
                targs.extend([str(t) for t in batch.target_chembl_id])

                preds.extend(y_score.astype(float).tolist())
        return ys, preds, [], mols, targs, splits

    metrics: dict[str, Any] = {
        "task_type": task_type,
        "model": {
            "type": "gnn_2d_pyg",
            "architecture": str(cfg.architecture),
            "embedding_dim": int(cfg.embedding_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "num_layers": int(cfg.num_layers),
            "dropout": float(cfg.dropout),
            "pooling": str(cfg.pooling),
            "residual": bool(cfg.residual),
            "head_mlp_layers": int(cfg.head_mlp_layers),
            "gin_conv_mlp_layers": int(cfg.gin_conv_mlp_layers),
            "gin_train_eps": bool(cfg.gin_train_eps),
            "gin_eps": float(cfg.gin_eps),
            "gat_heads": int(cfg.gat_heads),
            "sage_aggr": str(cfg.sage_aggr),
            "pna_deg_hist": pna_deg_list,
            "transformer_heads": int(cfg.transformer_heads),
            "transformer_edge_dim": cfg.transformer_edge_dim,
            "tag_k": int(cfg.tag_k),
            "arma_num_stacks": int(cfg.arma_num_stacks),
            "arma_num_layers": int(cfg.arma_num_layers),
            "cheb_k": int(cfg.cheb_k),
            "supergat_heads": int(cfg.supergat_heads),
            "supergat_attention_type": str(cfg.supergat_attention_type),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "random_seed": int(cfg.random_seed),
            "best_epoch": best_epoch if best_epoch > 0 else None,  # Epoch number where best model was saved
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dta_gnn_version": __version__,
        "splits": {},
    }

    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )
    from scipy.stats import pearsonr, spearmanr
    def _split_metrics(
        y_true: list[float], y_pred: list[float], y_prob: list[float] | None
    ):
        if not y_true:
            return None
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        return {
            "n": int(yt.size),
            "rmse": rmse,
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": float(r2_score(yt, yp)) if yt.size >= 2 else None,
            "pearson_r": float(pearsonr(yt, yp)[0]) if yt.size >= 2 else None,
            "spearman_r": float(spearmanr(yt, yp)[0]) if yt.size >= 2 else None,
        }

    # metrics for train/val/test
    for name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        if loader is None:
            continue
        y_true, y_pred, y_prob, *_rest = _predict(loader)
        metrics["splits"][name] = _split_metrics(
            y_true, y_pred, None
        )

    # Predictions (val+test)
    pred_rows: list[dict[str, Any]] = []
    for name, loader in [("val", val_loader), ("test", test_loader)]:
        if loader is None:
            continue
        y_true, y_pred, y_prob, mols, targs, splits = _predict(loader)
        for i in range(len(y_true)):
            row: dict[str, Any] = {
                "molecule_chembl_id": mols[i],
                "target_chembl_id": targs[i],
                "label": float(y_true[i]),
                "split": splits[i],
                "y_pred": float(y_pred[i]),
            }
            pred_rows.append(row)

    df_preds = pd.DataFrame(pred_rows)

    # Write artifacts - use architecture name in filenames
    arch_name = str(cfg.architecture).strip().lower()
    model_path = run_path / f"model_gnn_{arch_name}.pt"
    encoder_path = run_path / f"encoder_{arch_name}.pt"
    encoder_config_path = run_path / f"encoder_{arch_name}_config.json"
    metrics_path = run_path / f"model_metrics_gnn_{arch_name}.json"
    predictions_path = run_path / f"model_predictions_gnn_{arch_name}.csv"

    torch.save(model.state_dict(), model_path)
    torch.save(model.encoder.state_dict(), encoder_path)

    encoder_cfg = {
        "created_at": metrics["created_at"],
        "dta_gnn_version": __version__,
        "featurizer": {
            "type": "rdkit_2d_graph",
            "atom_feat_dim": 6,
            "bond_feat_dim": 6,
            "atom_type_vocab": 101,
        },
        "encoder": {
            "type": "gnn_2d_pyg",
            "architecture": str(cfg.architecture),
            "embedding_dim": int(cfg.embedding_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "num_layers": int(cfg.num_layers),
            "dropout": float(cfg.dropout),
            "pooling": str(cfg.pooling),
            "residual": bool(cfg.residual),
            "head_mlp_layers": int(cfg.head_mlp_layers),
            "gin_conv_mlp_layers": int(cfg.gin_conv_mlp_layers),
            "gin_train_eps": bool(cfg.gin_train_eps),
            "gin_eps": float(cfg.gin_eps),
            "gat_heads": int(cfg.gat_heads),
            "sage_aggr": str(cfg.sage_aggr),
            "pna_deg_hist": pna_deg_list,
            "transformer_heads": int(cfg.transformer_heads),
            "transformer_edge_dim": cfg.transformer_edge_dim,
            "tag_k": int(cfg.tag_k),
            "arma_num_stacks": int(cfg.arma_num_stacks),
            "arma_num_layers": int(cfg.arma_num_layers),
            "cheb_k": int(cfg.cheb_k),
            "supergat_heads": int(cfg.supergat_heads),
            "supergat_attention_type": str(cfg.supergat_attention_type),
        },
        "reproducibility": {"random_seed": int(cfg.random_seed)},
    }
    encoder_config_path.write_text(
        json.dumps(encoder_cfg, indent=2, sort_keys=True) + "\n"
    )

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    df_preds.to_csv(predictions_path, index=False)

    return GnnTrainResult(
        run_dir=run_path,
        task_type=task_type,
        model_path=model_path,
        encoder_path=encoder_path,
        encoder_config_path=encoder_config_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        metrics=metrics,
        best_epoch=best_epoch if best_epoch > 0 else None,  # Epoch number where best model was saved
    )


def extract_gnn_embeddings_on_run(
    run_dir: str | Path,
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> GnnEmbeddingExtractResult:
    """Use a saved GNN encoder to generate embeddings for molecules in compounds.csv.

    Requires:
      - compounds.csv
      - encoder_<architecture>.pt
      - encoder_<architecture>_config.json

    Writes:
      - molecule_embeddings.npz (molecule_chembl_id, embeddings)
    """

    _require_pyg()
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    run_path = _resolve_run_dir(run_dir)
    compounds_path = run_path / "compounds.csv"

    if not compounds_path.exists():
        raise ValueError(f"Missing compounds.csv in run folder: {compounds_path}")

    # Try to find encoder files by checking common architecture names
    # or by reading metadata.json if available
    encoder_path = None
    encoder_config_path = None
    arch_name = None

    # First, try to get architecture from metadata.json if it was set by training
    metadata_path = run_path / "metadata.json"
    if metadata_path.exists():
        try:
            meta = _load_json(metadata_path) or {}
            model_meta = meta.get("model") if isinstance(meta.get("model"), dict) else None
            if model_meta and "architecture" in model_meta:
                arch_name = str(model_meta.get("architecture", "")).strip().lower()
                if arch_name:
                    encoder_path = run_path / f"encoder_{arch_name}.pt"
                    encoder_config_path = run_path / f"encoder_{arch_name}_config.json"
        except Exception:
            pass

    # If not found from metadata (e.g. metadata from dataset build has no model.architecture),
    # probe for existing encoder_<arch>.pt / encoder_<arch>_config.json
    if not encoder_path or not encoder_config_path or not encoder_path.exists() or not encoder_config_path.exists():
        arch_name = None
        encoder_path = None
        encoder_config_path = None
        for arch in ["gin", "gcn", "gat", "sage", "pna", "transformer", "tag", "arma", "cheb", "supergat"]:
            test_encoder = run_path / f"encoder_{arch}.pt"
            test_config = run_path / f"encoder_{arch}_config.json"
            if test_encoder.exists() and test_config.exists():
                encoder_path = test_encoder
                encoder_config_path = test_config
                arch_name = arch
                break

    # Fallback to gin if nothing found
    if not encoder_path:
        arch_name = "gin"
        encoder_path = run_path / f"encoder_{arch_name}.pt"
        encoder_config_path = run_path / f"encoder_{arch_name}_config.json"

    if not encoder_path.exists() or not encoder_config_path.exists():
        raise ValueError(
            f"Missing encoder artifacts. Train the GNN model first to create "
            f"encoder_{arch_name}.pt and encoder_{arch_name}_config.json."
        )

    cfg = _load_json(encoder_config_path) or {}
    enc_cfg = cfg.get("encoder") if isinstance(cfg.get("encoder"), dict) else {}
    feat_cfg = cfg.get("featurizer") if isinstance(cfg.get("featurizer"), dict) else {}

    enc_type = str(enc_cfg.get("type") or "")
    architecture = str(enc_cfg.get("architecture") or "")
    if not architecture and enc_type in {"gin_pyg", "gin"}:
        architecture = "gin"
    if not architecture:
        architecture = "gin"

    pooling = str(enc_cfg.get("pooling") or "add")
    if pooling not in {"add", "mean", "max", "attention"}:
        pooling = "add"
    residual = bool(enc_cfg.get("residual") or False)
    head_mlp_layers = int(enc_cfg.get("head_mlp_layers") or 2)
    gin_conv_mlp_layers = int(enc_cfg.get("gin_conv_mlp_layers") or 2)
    gin_train_eps = bool(enc_cfg.get("gin_train_eps") or False)
    gin_eps = float(enc_cfg.get("gin_eps") or 0.0)
    gat_heads = int(enc_cfg.get("gat_heads") or 4)
    sage_aggr = str(enc_cfg.get("sage_aggr") or "mean")
    pna_deg_hist = enc_cfg.get("pna_deg_hist")
    transformer_heads = int(enc_cfg.get("transformer_heads") or 4)
    transformer_edge_dim = enc_cfg.get("transformer_edge_dim")
    tag_k = int(enc_cfg.get("tag_k") or 2)  # Default reduced from 3 for better performance
    arma_num_stacks = int(enc_cfg.get("arma_num_stacks") or 1)
    arma_num_layers = int(enc_cfg.get("arma_num_layers") or 1)
    cheb_k = int(enc_cfg.get("cheb_k") or 2)
    supergat_heads = int(enc_cfg.get("supergat_heads") or 4)
    supergat_attention_type = str(enc_cfg.get("supergat_attention_type") or "MX")
    pna_deg = None
    if architecture.strip().lower() == "pna":
        if isinstance(pna_deg_hist, list) and pna_deg_hist:
            pna_deg = torch.tensor([int(x) for x in pna_deg_hist], dtype=torch.long)
        else:
            raise ValueError(
                "encoder_config.json missing pna_deg_hist needed for PNA embeddings."
            )

    embedding_dim = int(enc_cfg.get("embedding_dim") or 128)
    hidden_dim = int(enc_cfg.get("hidden_dim") or 128)
    num_layers = int(enc_cfg.get("num_layers") or 5)
    dropout = float(enc_cfg.get("dropout") or 0.0)

    num_atom_types = int(feat_cfg.get("atom_type_vocab") or 101)

    df_comp = pd.read_csv(compounds_path)
    if "molecule_chembl_id" not in df_comp.columns or "smiles" not in df_comp.columns:
        raise ValueError(
            "compounds.csv must contain 'molecule_chembl_id' and 'smiles'."
        )

    df_comp = (
        df_comp[["molecule_chembl_id", "smiles"]]
        .dropna()
        .drop_duplicates(subset=["molecule_chembl_id"], keep="first")
        .reset_index(drop=True)
    )
    if df_comp.empty:
        raise ValueError("compounds.csv has no molecules with SMILES.")

    graphs = build_graphs_2d(
        molecules=list(
            zip(
                df_comp["molecule_chembl_id"].astype(str), df_comp["smiles"].astype(str)
            )
        ),
        drop_failures=True,
    )
    if not graphs:
        raise ValueError("No valid SMILES could be converted to graphs.")

    edge_dim = int(graphs[0].edge_attr.shape[1]) if graphs[0].edge_attr.size else 6

    GnnEncoder, _GnnPredictor = _make_encoder_and_model(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
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
        transformer_heads=transformer_heads,
        transformer_edge_dim=transformer_edge_dim,
        tag_k=tag_k,
        arma_num_stacks=arma_num_stacks,
        arma_num_layers=arma_num_layers,
        cheb_k=cheb_k,
        supergat_heads=supergat_heads,
        supergat_attention_type=supergat_attention_type,
    )
    encoder = GnnEncoder()
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
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=device_obj))
    encoder.to(device_obj)
    encoder.eval()

    data_list: list[Data] = []
    ids: list[str] = []
    for g in graphs:
        ids.append(str(g.molecule_chembl_id))
        data_list.append(
            Data(
                atom_type=torch.from_numpy(g.atom_type.astype(np.int64)),
                atom_feat=torch.from_numpy(g.atom_feat.astype(np.float32)),
                edge_index=torch.from_numpy(g.edge_index.astype(np.int64)),
                edge_attr=torch.from_numpy(g.edge_attr.astype(np.float32)),
                molecule_chembl_id=str(g.molecule_chembl_id),
                num_nodes=int(g.atom_feat.shape[0]),
            )
        )

    loader = DataLoader(data_list, batch_size=int(batch_size), shuffle=False)

    embs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device_obj)
            z = encoder(
                batch.atom_type,
                batch.atom_feat,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )
            embs.append(z.detach().cpu().numpy().astype(np.float32))

    embeddings = (
        np.concatenate(embs, axis=0)
        if embs
        else np.zeros((0, embedding_dim), dtype=np.float32)
    )

    out_path = run_path / "molecule_embeddings.npz"
    np.savez_compressed(
        out_path,
        molecule_chembl_id=np.asarray(ids, dtype=object),
        embeddings=embeddings,
    )

    return GnnEmbeddingExtractResult(
        run_dir=run_path,
        embeddings_path=out_path,
        n_molecules=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1])
        if embeddings.ndim == 2
        else int(embedding_dim),
    )


# Backwards-compatible function aliases (historical names)
train_gin_on_run = train_gnn_on_run
extract_gin_embeddings_on_run = extract_gnn_embeddings_on_run

# Backwards-compatible class aliases (historical names)
GinTrainConfig = GnnTrainConfig
GinTrainResult = GnnTrainResult
GinEmbeddingExtractResult = GnnEmbeddingExtractResult

__all__ = [
    "GnnTrainConfig",
    "GnnTrainResult",
    "GnnEmbeddingExtractResult",
    "train_gnn_on_run",
    "extract_gnn_embeddings_on_run",
    "_get_device",
    "_make_encoder_and_model",
    # Backwards-compatible aliases
    "GinTrainConfig",
    "GinTrainResult",
    "GinEmbeddingExtractResult",
    "train_gin_on_run",
    "extract_gin_embeddings_on_run",
]
