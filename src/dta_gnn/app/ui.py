import gradio as gr
import pandas as pd
from dta_gnn.pipeline import Pipeline
from dta_gnn.visualization import (
    plot_activity_distribution,
    plot_split_sizes,
    plot_chemical_space,
)
from dta_gnn.app_features.compound import build_smiles_frame, featurize_smiles_morgan
import tempfile
import os
from pathlib import Path
import io
import threading
import time
import zipfile
import json
from dataclasses import dataclass
from typing import Iterable

from loguru import logger
from dta_gnn.cleaning.validation import validate_split_sizes, validate_sqlite_path
from dta_gnn.exporters.artifacts import (
    artifact_keys_in_zip,
    artifacts_table,
    collect_artifacts,
    write_artifacts_zip,
    write_artifacts_zip_from_manifest,
)
from dta_gnn.io.runs import create_run_dir, resolve_current_run_dir, resolve_run_dir
from dta_gnn.io.target_mapping import parse_chembl_target_ids
from dta_gnn.io.utils import (
    find_chembl_sqlite_dbs,
    iter_existing_files,
    normalize_csv_path,
    preview_csv,
    preview_csv_with_error,
)
from dta_gnn.models import train_random_forest_on_run
from dta_gnn.models.utils import list_available_models


APP_THEME = gr.themes.Soft()

APP_CSS = """
/* Layout polish (no custom colors; rely on theme tokens) */
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
}
#app-header { align-items: center; gap: 16px; }
#app-title h1 { margin-bottom: 0.25rem; }
#app-subtitle { opacity: 0.9; }
.section-card {
    border-radius: 12px;
    padding: 12px;
}
.compact-table .table-wrap { max-height: 420px; overflow: auto; }
.citation-box textarea {
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
    line-height: 1.6;
    background-color: var(--input-background-fill);
    border: 1px solid var(--border-color-primary);
}
"""


def _resolve_current_run_dir(*, hint: str = "Build a dataset first.") -> Path:
    """Resolve the current run folder.

    Prefers `runs/current` if it exists (dir or symlink). If missing, raises a
    Gradio-friendly error.
    """
    try:
        return resolve_current_run_dir(hint=hint)
    except FileNotFoundError as e:
        raise gr.Error(str(e))




def _update_model_choices(model_type: str) -> tuple[list[str], str]:
    """Update available model choices based on model type selection.
    
    Returns:
        Tuple of (choices list, default value)
    """
    try:
        run_dir = _resolve_current_run_dir()
        available = list_available_models(run_dir)
    except Exception:
        available = {"rf": [], "svr": [], "gnn": []}
    
    if (model_type or "").startswith("RandomForest"):
        choices = [f"RandomForest (ECFP4) - {m}" for m in available["rf"]] if available["rf"] else ["RandomForest (ECFP4)"]
        return choices, choices[0] if choices else "RandomForest (ECFP4)"
    elif (model_type or "").startswith("SVR"):
        choices = [f"SVR (ECFP4) - {m}" for m in available["svr"]] if available["svr"] else ["SVR (ECFP4)"]
        return choices, choices[0] if choices else "SVR (ECFP4)"
    else:  # GNN
        if available["gnn"]:
            # Extract architecture from choices like "GNN (GIN)" -> "gin"
            choices = available["gnn"]
            return choices, choices[0]
        else:
            return ["GNN (2D)"], "GNN (2D)"






def build_dataset(
    dataset_kind,
    target_id_type,
    targets_str,
    source,
    sqlite_path,
    split_method,
    test_size,
    val_size,
    split_year,
    progress=gr.Progress(),
):
    try:
        validate_sqlite_path(source, sqlite_path)
    except ValueError as e:
        raise gr.Error(str(e))
    try:
        validate_split_sizes(test_size, val_size)
    except ValueError as e:
        raise gr.Error(str(e))

    log_stream = io.StringIO()
    sink_id = logger.add(
        log_stream,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        level="INFO",
    )

    try:
        from dta_gnn.io.target_mapping import (
            map_uniprot_to_chembl_targets_sqlite,
            map_uniprot_to_chembl_targets_web,
            parse_uniprot_accessions,
        )

        ttype = (target_id_type or "ChEMBL").strip().lower()
        if ttype == "uniprot":
            accessions = parse_uniprot_accessions(targets_str)
            if source == "sqlite":
                res = map_uniprot_to_chembl_targets_sqlite(sqlite_path, accessions)
            else:
                res = map_uniprot_to_chembl_targets_web(accessions)

            if res.unmapped:
                logger.warning(
                    "No ChEMBL targets found for UniProt: " + ", ".join(res.unmapped)
                )

            target_ids_param = res.resolved_target_chembl_ids
            if not target_ids_param:
                raise gr.Error(
                    "Could not map any UniProt accessions to ChEMBL target IDs. "
                    "Try a different accession, or use 'ChEMBL' ID type."
                )
            logger.info(
                f"Resolved UniProt→ChEMBL targets: {accessions} -> {target_ids_param}"
            )
        else:
            target_ids_param = parse_chembl_target_ids(targets_str)
    except ValueError as e:
        raise gr.Error(str(e))

    pipeline = Pipeline(
        source_type=source, sqlite_path=sqlite_path if source == "sqlite" else None
    )
    run_dir = create_run_dir()
    pipeline.last_run_dir = str(run_dir)

    dataset_csv_path = Path(run_dir) / "dataset.csv"
    targets_csv_path = Path(run_dir) / "targets.csv"
    compounds_csv_path = Path(run_dir) / "compounds.csv"

    result: dict[str, object] = {}
    
    # Shared state for progress tracking across threads
    prog_state = {"value": 0, "total": None, "desc": "Starting..."}
    
    def _prog_cb(value, total, desc):
        prog_state["value"] = value
        prog_state["total"] = total
        prog_state["desc"] = desc

    def _runner():
        try:
            df_local = pipeline.build_dta(
                target_ids=target_ids_param,
                standard_types=["IC50", "Ki", "Kd"],
                split_method=split_method,
                output_path=str(dataset_csv_path),
                test_size=test_size,
                val_size=val_size,
                split_year=int(split_year) if split_year else 2022,
                featurize=False,
                progress_callback=_prog_cb,
            )

            # Persist common artifacts for the UI.
            if isinstance(df_local, pd.DataFrame) and not df_local.empty:
                pipeline.last_dataset_csv = str(dataset_csv_path)

                # Targets
                try:
                    unique_targets = sorted(
                        set(df_local["target_chembl_id"].dropna().astype(str).tolist())
                    )
                    targets_df = pipeline.source.fetch_targets(unique_targets)
                    targets_df.to_csv(targets_csv_path, index=False)
                    pipeline.last_targets_csv = str(targets_csv_path)
                except Exception:
                    pipeline.last_targets_csv = getattr(
                        pipeline, "last_targets_csv", None
                    )

                # Compounds
                try:
                    unique_mols = sorted(
                        set(
                            df_local["molecule_chembl_id"].dropna().astype(str).tolist()
                        )
                    )
                    compounds_df = pipeline.source.fetch_molecules(unique_mols)
                    compounds_df.to_csv(compounds_csv_path, index=False)
                    pipeline.last_compounds_csv = str(compounds_csv_path)
                except Exception:
                    pipeline.last_compounds_csv = getattr(
                        pipeline, "last_compounds_csv", None
                    )

            result["df"] = df_local
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    # Update progress bar while running; do not yield in loop to avoid UI blinking
    try:
        yield (
            None,
            "Building dataset...",
            None,
            None,
            None,
            None,
            log_stream.getvalue(),
            "",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        while thread.is_alive():
            curr = prog_state["value"]
            tot = prog_state["total"]
            desc = prog_state["desc"]
            if tot and tot > 0:
                progress(curr / tot, desc=desc)
            else:
                progress((0, None), desc=desc)
            time.sleep(0.4)
        thread.join()

        if "error" in result:
            raise gr.Error(f"Error building dataset: {result['error']}")

        df = result.get("df")
        if not isinstance(df, pd.DataFrame):
            raise gr.Error("Unexpected error: dataset build produced no DataFrame.")
    finally:
        logger.remove(sink_id)

    if df.empty:
        yield (
            None,
            "No data found.",
            None,
            None,
            None,
            None,
            log_stream.getvalue(),
            getattr(pipeline, "last_run_dir", "") or "",
            None,
            None,
            None,
            getattr(pipeline, "last_targets_csv", None),
            None,
            getattr(pipeline, "last_compounds_csv", None),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        return

    # Prefer the run-saved artifact for download (runs/<timestamp>/dataset.csv)
    download_path = getattr(pipeline, "last_dataset_csv", None)
    if not download_path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False)
        download_path = tmp.name

    summary = f"""
    ### Dataset Summary
    - **Total Rows**: {len(df)}
    - **Label Range**: {df['label'].min():.2f} - {df['label'].max():.2f}
    - **Mean Affinity**: {df['label'].mean():.2f}
    - **Split Sizes**:
      - Train: {len(df[df['split']=='train'])}
      - Val: {len(df[df['split']=='val'])}
      - Test: {len(df[df['split']=='test'])}
    """

    fig_activity = plot_activity_distribution(df)
    fig_splits = plot_split_sizes(df)

    run_dir = getattr(pipeline, "last_run_dir", "") or ""
    targets_path = getattr(pipeline, "last_targets_csv", None)
    compounds_path = getattr(pipeline, "last_compounds_csv", None)

    warnings: list[str] = []
    dataset_preview_info = preview_csv_with_error(download_path)
    dataset_preview = dataset_preview_info.df
    if dataset_preview_info.error:
        warnings.append(f"dataset preview: {dataset_preview_info.error}")

    targets_preview_info = preview_csv_with_error(targets_path)
    targets_preview = targets_preview_info.df
    if targets_preview_info.error:
        warnings.append(f"targets preview: {targets_preview_info.error}")

    compounds_preview_info = preview_csv_with_error(compounds_path)
    compounds_preview = compounds_preview_info.df
    if compounds_preview_info.error:
        warnings.append(f"compounds preview: {compounds_preview_info.error}")

    artifacts = collect_artifacts(
        run_dir=run_dir,
        dataset_path=download_path,
        targets_path=targets_path,
        compounds_path=compounds_path,
    )

    zip_path = write_artifacts_zip(
        zip_path=artifacts.get("zip"),
        paths=[artifacts.get(k) for k in artifact_keys_in_zip()],
    )
    artifacts["zip"] = zip_path
    artifacts_table_df = artifacts_table(artifacts)

    logs_out = log_stream.getvalue()
    if warnings:
        logs_out = logs_out + "\n\n[UI] Preview warnings:\n- " + "\n- ".join(warnings)

    yield (
        df,
        summary,
        download_path,
        fig_activity,
        fig_splits,
        df,
        logs_out,
        run_dir,
        artifacts_table_df,
        artifacts.get("dataset"),
        dataset_preview,
        artifacts.get("targets"),
        targets_preview,
        artifacts.get("compounds"),
        compounds_preview,
        artifacts.get("molecule_features"),
        artifacts.get("protein_features"),
        artifacts.get("model"),
        artifacts.get("model_metrics"),
        artifacts.get("model_predictions"),
        zip_path,
    )


def wrapper_plot_space(smiles_text, method, df_state, show_train, show_val, show_test):
    smiles_data = {}

    # 1. Add Custom SMILES
    if smiles_text:
        import re

        tokens = re.split(r"[,\n\s]+", smiles_text)
        custom_list = [t.strip() for t in tokens if t.strip()]
        if custom_list:
            smiles_data["Custom"] = custom_list

    # 2. Add Dataset SMILES from State
    if df_state is not None and not df_state.empty:
        # Check which column to use
        smiles_col = "smiles" if "smiles" in df_state.columns else "canonical_smiles"

        if show_train:
            train_smiles = (
                df_state[df_state["split"] == "train"][smiles_col].dropna().tolist()
            )
            if train_smiles:
                smiles_data["Train"] = train_smiles
        if show_val:
            val_smiles = (
                df_state[df_state["split"] == "val"][smiles_col].dropna().tolist()
            )
            if val_smiles:
                smiles_data["Validation"] = val_smiles
        if show_test:
            test_smiles = (
                df_state[df_state["split"] == "test"][smiles_col].dropna().tolist()
            )
            if test_smiles:
                smiles_data["Test"] = test_smiles

    if not smiles_data:
        return None

    return plot_chemical_space(smiles_data, method=method)


def visualize_embeddings(
    method: str,
    color_by: str,
    model_selector: str | None,
    perplexity: int,
    show_top_k: bool = False,
    top_k: int = 100,
):
    """Visualize embeddings with t-SNE or PCA, colored by split, ground truth, or predictions."""
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    
    try:
        run_dir = _resolve_current_run_dir(
            hint="Build a dataset and extract embeddings first."
        )
    except gr.Error as e:
        return None, str(e)
    
    # Try to load GNN embeddings first
    embeddings_path = run_dir / "molecule_embeddings.npz"
    use_ecfp_fallback = False
    warning_msg = ""
    
    if embeddings_path.exists():
        try:
            npz = np.load(str(embeddings_path), allow_pickle=True)
            molecule_ids = npz["molecule_chembl_id"]
            embeddings = npz["embeddings"].astype(np.float32)
        except Exception as e:
            return None, f"Error loading embeddings: {str(e)}"
    else:
        # Fallback to ECFP4 features
        use_ecfp_fallback = True
        warning_msg = "⚠️ Warning: GNN embeddings not found. Using ECFP4 (Morgan fingerprints) features instead. For better visualization, extract GNN embeddings first (Train tab → Extract Molecule Embeddings).\n\n"
        
        # Load compounds to get SMILES
        compounds_path = run_dir / "compounds.csv"
        if not compounds_path.exists():
            return None, "No embeddings found and compounds.csv missing. Please extract GNN embeddings or ensure compounds.csv exists."
        
        try:
            df_compounds = pd.read_csv(compounds_path)
            if "smiles" not in df_compounds.columns and "canonical_smiles" not in df_compounds.columns:
                return None, "No embeddings found and compounds.csv missing SMILES column."
            
            smiles_col = "smiles" if "smiles" in df_compounds.columns else "canonical_smiles"
            if "molecule_chembl_id" not in df_compounds.columns:
                return None, "No embeddings found and compounds.csv missing molecule_chembl_id column."
            
            # Generate ECFP4 features
            molecule_ids = []
            embeddings_list = []
            
            for _, row in df_compounds.iterrows():
                mol_id = str(row["molecule_chembl_id"])
                smiles = str(row[smiles_col])
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    
                    # Generate ECFP4 (radius=2, nBits=2048)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    arr = np.zeros((2048,), dtype=np.int8)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    
                    molecule_ids.append(mol_id)
                    embeddings_list.append(arr.astype(np.float32))
                except Exception:
                    continue
            
            if not embeddings_list:
                return None, "No valid SMILES found in compounds.csv to generate ECFP4 features."
            
            embeddings = np.array(embeddings_list, dtype=np.float32)
            molecule_ids = np.array(molecule_ids, dtype=object)
            
        except Exception as e:
            return None, f"Error generating ECFP4 features: {str(e)}"
    
    # Load dataset to get splits and labels
    dataset_path = run_dir / "dataset.csv"
    if not dataset_path.exists():
        return None, "Missing dataset.csv in run directory."
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"
    
    # Match embeddings to dataset rows
    # Create a mapping from molecule_chembl_id to index in embeddings
    mol_id_to_emb_idx = {str(mid): i for i, mid in enumerate(molecule_ids)}
    
    # Get unique molecules that have embeddings
    unique_mol_ids = list(mol_id_to_emb_idx.keys())
    
    # Filter dataset to only include molecules with embeddings
    df_filtered = df[df["molecule_chembl_id"].isin(unique_mol_ids)].copy()
    if df_filtered.empty:
        return None, "No matching molecules between embeddings and dataset."
    
    # Helper function to find predictions file
    def find_predictions_file(model_selector_val):
        """Find predictions file for given model selector."""
        model_name = str(model_selector_val).strip()
        if model_name.startswith("GNN ("):
            arch_name = model_name.replace("GNN (", "").replace(")", "").strip().lower()
            model_type = "gnn"
        elif model_name == "RandomForest":
            arch_name = "rf"
            model_type = "rf"
        elif model_name == "SVR":
            arch_name = "svr"
            model_type = "svr"
        else:
            arch_name = model_name.lower()
            model_type = "unknown"
        
        if model_type == "gnn":
            pred_paths = [
                run_dir / f"model_predictions_gnn_{arch_name}.csv",
                run_dir / f"model_predictions_{arch_name}.csv",
            ]
        elif model_type == "rf":
            pred_paths = [
                run_dir / "model_predictions.csv",
                run_dir / "model_predictions_rf.csv",
            ]
        elif model_type == "svr":
            pred_paths = [
                run_dir / "model_predictions_svr.csv",
                run_dir / "model_predictions.csv",
            ]
        else:
            pred_paths = [
                run_dir / f"model_predictions_{arch_name}.csv",
                run_dir / "model_predictions.csv",
            ]
        
        for pp in pred_paths:
            if pp.exists():
                return pp, model_type, arch_name
        return None, model_type, arch_name
    
    # If show_top_k is enabled and we're using predictions, filter to top-k test molecules first
    df_preds_loaded = None
    if show_top_k and color_by == "Model Predictions":
        if not model_selector:
            return None, "Please select a model for predictions to use top-k filtering."
        
        pred_path, model_type, arch_name = find_predictions_file(model_selector)
        if not pred_path:
            return None, f"Predictions file not found for model: {model_selector}. Please train the model first."
        
        try:
            df_preds_loaded = pd.read_csv(pred_path)
            if "split" not in df_preds_loaded.columns:
                return None, "Predictions file must contain 'split' column for top-k filtering."
            if "y_pred" not in df_preds_loaded.columns:
                return None, "Predictions file must contain 'y_pred' column for top-k filtering."
            
            # Filter to test set only
            test_preds = df_preds_loaded[df_preds_loaded["split"] == "test"].copy()
            if test_preds.empty:
                return None, "No test set predictions found. Cannot filter top-k."
            
            # Get top-k by highest predicted binding affinity (y_pred)
            # For regression, higher y_pred = higher binding affinity
            test_preds_sorted = test_preds.sort_values("y_pred", ascending=False)
            top_k_molecules = test_preds_sorted.head(int(top_k))["molecule_chembl_id"].unique().tolist()
            
            # Filter unique_mol_ids to only include top-k molecules
            unique_mol_ids = [mid for mid in unique_mol_ids if str(mid) in [str(m) for m in top_k_molecules]]
            
            if not unique_mol_ids:
                return None, f"No top-k molecules found in embeddings. Try a larger k value or ensure test set molecules have embeddings."
            
            # Re-filter dataset to only include top-k molecules
            df_filtered = df_filtered[df_filtered["molecule_chembl_id"].isin(unique_mol_ids)].copy()
        except Exception as e:
            return None, f"Error filtering top-k predictions: {str(e)}"
    
    # For each unique molecule, get its representative value
    # (since dataset can have multiple rows per molecule, we'll aggregate)
    if color_by == "Split (Train/Val/Test)":
        # For split, take the first split value per molecule
        mol_to_split = df_filtered.groupby("molecule_chembl_id")["split"].first().to_dict()
        color_data = np.array([mol_to_split.get(str(mid), "unknown") for mid in unique_mol_ids])
        color_label = "Split"
        cmap = "Set1"
        is_continuous = False
    elif color_by == "Ground Truth (Affinity)":
        if "label" not in df_filtered.columns:
            return None, "Dataset missing 'label' column for ground truth values."
        # For labels, take the mean per molecule (or could use median)
        mol_to_label = df_filtered.groupby("molecule_chembl_id")["label"].mean().to_dict()
        color_data = np.array([mol_to_label.get(str(mid), np.nan) for mid in unique_mol_ids], dtype=float)
        # Remove NaN values
        valid_mask = ~np.isnan(color_data)
        if not valid_mask.any():
            return None, "No valid label values found for molecules."
        color_label = "pChEMBL Value"
        cmap = "viridis"
        is_continuous = True
    elif color_by == "Model Predictions":
        if not model_selector:
            return None, "Please select a model for predictions."
        
        # Use already loaded predictions if available (from top-k filtering), otherwise load
        if df_preds_loaded is not None:
            df_preds = df_preds_loaded
        else:
            pred_path, model_type, arch_name = find_predictions_file(model_selector)
            if not pred_path:
                return None, f"Predictions file not found for model: {model_selector}. Please train the model first."
            
            try:
                df_preds = pd.read_csv(pred_path)
            except Exception as e:
                return None, f"Error loading predictions: {str(e)}"
        
        try:
            # Merge predictions with dataset to get molecule-level predictions
            if "molecule_chembl_id" not in df_preds.columns:
                return None, "Predictions file must contain 'molecule_chembl_id' column."
            if "y_pred" not in df_preds.columns:
                return None, "Predictions file must contain 'y_pred' column."
            
            # Aggregate predictions per molecule (mean if multiple)
            mol_to_pred = df_preds.groupby("molecule_chembl_id")["y_pred"].mean().to_dict()
            
            # Map predictions to unique molecules (already filtered to top-k if enabled)
            color_data = np.array([mol_to_pred.get(str(mid), np.nan) for mid in unique_mol_ids], dtype=float)
            # Remove NaN values
            valid_mask = ~np.isnan(color_data)
            if not valid_mask.any():
                return None, "No matching predictions found for molecules with embeddings."
            color_label = "Predicted pChEMBL"
            cmap = "plasma"
            is_continuous = True
        except Exception as e:
            return None, f"Error processing predictions: {str(e)}"
    else:
        return None, f"Unknown color_by option: {color_by}"
    
    # Get embeddings for unique molecules (in the same order as unique_mol_ids)
    emb_indices = [mol_id_to_emb_idx[mid] for mid in unique_mol_ids]
    embeddings_filtered = embeddings[emb_indices]
    
    # Filter out NaN values for continuous coloring
    if is_continuous:
        valid_mask = ~np.isnan(color_data)
        if not valid_mask.any():
            return None, f"No valid {color_label.lower()} values found."
        embeddings_filtered = embeddings_filtered[valid_mask]
        color_data = color_data[valid_mask]
        unique_mol_ids_filtered = [mid for i, mid in enumerate(unique_mol_ids) if valid_mask[i]]
    else:
        valid_mask = np.ones(len(color_data), dtype=bool)
        unique_mol_ids_filtered = unique_mol_ids
    
    # Dimensionality reduction
    if method == "t-SNE":
        n_samples = embeddings_filtered.shape[0]
        eff_perplexity = min(perplexity, n_samples - 1) if n_samples > 1 else 1
        init_method = "pca" if n_samples > 2 else "random"
        
        reducer = TSNE(
            n_components=2,
            perplexity=eff_perplexity,
            learning_rate=200.0,
            random_state=42,
            init=init_method,
        )
        X_2d = reducer.fit_transform(embeddings_filtered)
    elif method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(embeddings_filtered)
    else:
        return None, f"Unknown method: {method}"
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if is_continuous:
        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=color_data,
            cmap=cmap,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_label, rotation=270, labelpad=20)
    else:
        # Categorical coloring
        unique_values = sorted(set(color_data))
        colors = sns.color_palette("Set1", n_colors=len(unique_values))
        for i, val in enumerate(unique_values):
            mask = color_data == val
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[colors[i]],
                label=str(val),
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        ax.legend(title=color_label, loc='best')
    
    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    ax.set_title(f"Embedding Visualization ({method}) - Colored by {color_label}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    status_msg = warning_msg + f"Visualized {len(embeddings_filtered)} molecules using {method}"
    if use_ecfp_fallback:
        status_msg += " (ECFP4 features)"
    else:
        status_msg += " (GNN embeddings)"
    if color_by == "Model Predictions":
        status_msg += f" with predictions from {model_selector}"
        if show_top_k:
            status_msg += f" (Top-{top_k} from test set)"
    if is_continuous and valid_mask.sum() < len(valid_mask):
        status_msg += f" ({len(embeddings_filtered)}/{len(unique_mol_ids)} with valid values)"
    
    return fig, status_msg


def featurize_smiles(source_mode: str, smiles_text: str, df_state: pd.DataFrame | None):
    try:
        df = build_smiles_frame(
            smiles_text=smiles_text, df_state=df_state, source_mode=source_mode
        )
        df_feat = featurize_smiles_morgan(
            df, smiles_col="smiles", radius=2, n_bits=2048
        )
    except Exception as e:
        raise gr.Error(str(e))

    try:
        run_dir = _resolve_current_run_dir(
            hint="Build a dataset first (or create a run)."
        )
    except gr.Error:
        run_dir = create_run_dir()

    out_path = Path(run_dir) / "molecule_features.csv"
    # Save IDs + features (avoid repeating smiles)
    cols = [
        c for c in ["molecule_chembl_id", "morgan_fingerprint"] if c in df_feat.columns
    ]
    df_to_save = df_feat[cols] if cols else df_feat
    df_to_save.to_csv(out_path, index=False)
    artifacts = collect_artifacts(
        run_dir=str(run_dir),
        dataset_path=None,
        targets_path=None,
        compounds_path=None,
    )
    artifacts["molecule_features"] = str(out_path)
    zip_path = write_artifacts_zip_from_manifest(artifacts=artifacts)
    artifacts["zip"] = zip_path
    return (
        df_to_save,
        str(out_path),
        str(run_dir),
        artifacts_table(artifacts),
        artifacts.get("molecule_features"),
        artifacts.get("protein_features"),
        zip_path,
    )


def train_model(
    model_type: str,
    rf_n_estimators: int,
    svr_kernel: str,
    svr_c: float,
    svr_epsilon: float,
    gnn_arch: str,
    gnn_epochs: int,
    gnn_batch_size: int,
    gnn_embedding_dim: int,
    gnn_hidden_dim: int,
    gnn_num_layers: int,
    gnn_lr: float,
    gnn_dropout: float,
    gin_pooling: str,
    gnn_residual: bool,
    gnn_head_mlp_layers: int,
    gin_conv_mlp_layers: int,
    gin_train_eps: bool,
    gin_eps: float,
):
    run_dir = _resolve_current_run_dir(hint="Build a dataset first.")

    try:
        mt = (model_type or "").strip()
        if mt.startswith("RF") or mt.lower().startswith("randomforest"):
            result = train_random_forest_on_run(
                run_dir, n_estimators=int(rf_n_estimators)
            )

            artifacts = collect_artifacts(run_dir=str(result.run_dir))
            # Make the generic model download point at the selected model.
            artifacts["model"] = str(result.model_path)
            artifacts["model_metrics"] = str(result.metrics_path)
            artifacts["model_predictions"] = str(result.predictions_path)

            metrics_df = pd.DataFrame(
                [
                    {"split": k, **(v if isinstance(v, dict) else {})}
                    for k, v in (result.metrics.get("splits") or {}).items()
                ]
            )
            preds_preview = preview_csv(str(result.predictions_path), n=50)
            task_type = str(result.task_type)

            model_path = str(result.model_path)
            metrics_path = str(result.metrics_path)
            predictions_path = str(result.predictions_path)

        elif mt.startswith("SVR"):
            from dta_gnn.models.svr import train_svr_on_run

            res = train_svr_on_run(
                run_dir,
                kernel=str(svr_kernel),
                C=float(svr_c),
                epsilon=float(svr_epsilon),
            )

            artifacts = collect_artifacts(run_dir=str(res.run_dir))
            artifacts["model"] = str(res.model_path)
            artifacts["model_metrics"] = str(res.metrics_path)
            artifacts["model_predictions"] = str(res.predictions_path)

            metrics_df = pd.DataFrame(
                [
                    {"split": k, **(v if isinstance(v, dict) else {})}
                    for k, v in (res.metrics.get("splits") or {}).items()
                ]
            )
            preds_preview = preview_csv(str(res.predictions_path), n=50)
            task_type = str(res.task_type)

            model_path = str(res.model_path)
            metrics_path = str(res.metrics_path)
            predictions_path = str(res.predictions_path)

        else:
            from dta_gnn.models.gnn import GnnTrainConfig, train_gnn_on_run

            # Device is automatically detected (MPS > CUDA > CPU)
            # When device=None, _get_device() will auto-detect the best available device
            cfg = GnnTrainConfig(
                architecture=str(gnn_arch),
                epochs=int(gnn_epochs),
                batch_size=int(gnn_batch_size),
                embedding_dim=int(gnn_embedding_dim),
                hidden_dim=int(gnn_hidden_dim),
                num_layers=int(gnn_num_layers),
                lr=float(gnn_lr),
                dropout=float(gnn_dropout),
                pooling=str(gin_pooling),
                residual=bool(gnn_residual),
                head_mlp_layers=int(gnn_head_mlp_layers),
                gin_conv_mlp_layers=int(gin_conv_mlp_layers),
                gin_train_eps=bool(gin_train_eps),
                gin_eps=float(gin_eps),
                # device=None by default - auto-detects: MPS > CUDA > CPU
            )
            gnn_res = train_gnn_on_run(run_dir, config=cfg)

            artifacts = collect_artifacts(run_dir=str(gnn_res.run_dir))
            # Make the generic model download point at the selected model.
            artifacts["model"] = str(gnn_res.model_path)
            artifacts["model_metrics"] = str(gnn_res.metrics_path)
            artifacts["model_predictions"] = str(gnn_res.predictions_path)

            metrics_df = pd.DataFrame(
                [
                    {"split": k, **(v if isinstance(v, dict) else {})}
                    for k, v in (gnn_res.metrics.get("splits") or {}).items()
                ]
            )
            preds_preview = preview_csv(str(gnn_res.predictions_path), n=50)
            task_type = str(gnn_res.task_type)

            model_path = str(gnn_res.model_path)
            metrics_path = str(gnn_res.metrics_path)
            predictions_path = str(gnn_res.predictions_path)
    except Exception as e:
        raise gr.Error(str(e))

    zip_path = write_artifacts_zip_from_manifest(artifacts=artifacts)
    artifacts["zip"] = zip_path

    return (
        task_type,
        metrics_df,
        preds_preview,
        model_path,
        metrics_path,
        predictions_path,
        str(run_dir.resolve()),
        artifacts_table(artifacts),
        zip_path,
    )


def extract_gin_embeddings(batch_size: int):
    run_dir = _resolve_current_run_dir(hint="Train a GNN first.")

    try:
        from dta_gnn.models.gnn import extract_gnn_embeddings_on_run

        res = extract_gnn_embeddings_on_run(run_dir, batch_size=int(batch_size))
    except Exception as e:
        raise gr.Error(str(e))

    artifacts = collect_artifacts(run_dir=str(res.run_dir))
    zip_path = write_artifacts_zip_from_manifest(artifacts=artifacts)
    artifacts["zip"] = zip_path

    preview_df: pd.DataFrame | None = None
    try:
        import numpy as np

        npz = np.load(str(res.embeddings_path), allow_pickle=True)
        ids = npz["molecule_chembl_id"].astype(object)
        emb = npz["embeddings"].astype(float)
        cols: dict[str, object] = {"molecule_chembl_id": ids}
        if emb.ndim == 2:
            for j in range(min(8, emb.shape[1])):
                cols[f"e{j}"] = emb[:, j]
        preview_df = pd.DataFrame(cols).head(50)
    except Exception:
        preview_df = None

    return (
        preview_df,
        str(res.embeddings_path),
        str(res.run_dir),
        artifacts_table(artifacts),
        artifacts.get("molecule_embeddings"),
        zip_path,
    )


def run_hyperopt(
    model_type: str,
    optimizer_backend: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_api_key: str,
    wandb_sweep_name: str,
    n_trials: int,
    n_jobs: int,
    # RF params
    rf_opt_n_est: bool,
    rf_n_est_min: int,
    rf_n_est_max: int,
    rf_opt_depth: bool,
    rf_depth_min: int,
    rf_depth_max: int,
    rf_opt_min_samp: bool,
    rf_min_samp_min: int,
    rf_min_samp_max: int,
    # SVR params
    svr_opt_c: bool,
    svr_c_min: float,
    svr_c_max: float,
    svr_opt_epsilon: bool,
    svr_epsilon_min: float,
    svr_epsilon_max: float,
    svr_opt_kernel: bool,
    svr_kernel_choices: list[str],
    svr_kernel_default: str,
    # GNN params
    gnn_arch: str,
    gnn_opt_epochs: bool,
    gnn_epochs_min: int,
    gnn_epochs_max: int,
    gnn_opt_lr: bool,
    gnn_lr_min: float,
    gnn_lr_max: float,
    gnn_opt_batch: bool,
    gnn_batch_min: int,
    gnn_batch_max: int,
    gnn_opt_emb: bool,
    gnn_emb_min: int,
    gnn_emb_max: int,
    gnn_opt_hidden: bool,
    gnn_hidden_min: int,
    gnn_hidden_max: int,
    gnn_opt_dropout: bool,
    gnn_dropout_min: float,
    gnn_dropout_max: float,
    gnn_dropout_default: float,
    gnn_opt_pooling: bool,
    gnn_pooling_choices: list[str],
    gnn_pooling_default: str,
    gnn_opt_residual: bool,
    gnn_residual_default: bool,
    gnn_opt_head_mlp_layers: bool,
    gnn_head_mlp_layers_min: int,
    gnn_head_mlp_layers_max: int,
    gnn_head_mlp_layers_default: int,
    gnn_opt_gin_conv_mlp_layers: bool,
    gnn_gin_conv_mlp_layers_min: int,
    gnn_gin_conv_mlp_layers_max: int,
    gnn_gin_conv_mlp_layers_default: int,
):
    """Run hyperparameter optimization."""

    run_dir = _resolve_current_run_dir(hint="Build a dataset first.")

    try:
        from dta_gnn.models.hyperopt import (
            HyperoptConfig,
            optimize_gnn_wandb,
            optimize_random_forest_wandb,
            optimize_svr_wandb,
        )

        project = str(wandb_project or "dta_gnn").strip() or "dta_gnn"
        entity = str(wandb_entity).strip() or None
        api_key = str(wandb_api_key).strip() or None
        sweep_name = str(wandb_sweep_name).strip() or None

        if (model_type or "").startswith("RandomForest"):
            cfg = HyperoptConfig(
                model_type="RandomForest",
                n_trials=int(n_trials),
                n_jobs=1,
                rf_optimize_n_estimators=bool(rf_opt_n_est),
                rf_n_estimators_min=int(rf_n_est_min),
                rf_n_estimators_max=int(rf_n_est_max),
                rf_optimize_max_depth=bool(rf_opt_depth),
                rf_max_depth_min=int(rf_depth_min),
                rf_max_depth_max=int(rf_depth_max),
                rf_optimize_min_samples_split=bool(rf_opt_min_samp),
                rf_min_samples_split_min=int(rf_min_samp_min),
                rf_min_samples_split_max=int(rf_min_samp_max),
            )

            result = optimize_random_forest_wandb(
                run_dir,
                config=cfg,
                project=project,
                entity=entity,
                api_key=api_key,
                sweep_name=sweep_name,
            )

        elif (model_type or "").startswith("SVR"):
            cfg = HyperoptConfig(
                model_type="SVR",
                n_trials=int(n_trials),
                n_jobs=1,
                svr_optimize_C=bool(svr_opt_c),
                svr_C_min=float(svr_c_min),
                svr_C_max=float(svr_c_max),
                svr_optimize_epsilon=bool(svr_opt_epsilon),
                svr_epsilon_min=float(svr_epsilon_min),
                svr_epsilon_max=float(svr_epsilon_max),
                svr_optimize_kernel=bool(svr_opt_kernel),
                svr_kernel_choices=list(svr_kernel_choices or []),
                svr_kernel_default=str(svr_kernel_default or "rbf"),
            )

            result = optimize_svr_wandb(
                run_dir,
                config=cfg,
                project=project,
                entity=entity,
                api_key=api_key,
                sweep_name=sweep_name,
            )

        else:
            cfg = HyperoptConfig(
                model_type="GNN",
                n_trials=int(n_trials),
                n_jobs=1,
                architecture=str(gnn_arch or "gin").strip().lower(),
                optimize_epochs=bool(gnn_opt_epochs),
                epochs_min=int(gnn_epochs_min),
                epochs_max=int(gnn_epochs_max),
                optimize_lr=bool(gnn_opt_lr),
                lr_min=float(gnn_lr_min),
                lr_max=float(gnn_lr_max),
                optimize_batch_size=bool(gnn_opt_batch),
                batch_size_min=int(gnn_batch_min),
                batch_size_max=int(gnn_batch_max),
                optimize_embedding_dim=bool(gnn_opt_emb),
                embedding_dim_min=int(gnn_emb_min),
                embedding_dim_max=int(gnn_emb_max),
                optimize_hidden_dim=bool(gnn_opt_hidden),
                hidden_dim_min=int(gnn_hidden_min),
                hidden_dim_max=int(gnn_hidden_max),
                optimize_dropout=bool(gnn_opt_dropout),
                dropout_min=float(gnn_dropout_min),
                dropout_max=float(gnn_dropout_max),
                dropout_default=float(gnn_dropout_default),
                optimize_pooling=bool(gnn_opt_pooling),
                pooling_choices=list(gnn_pooling_choices or []),
                pooling_default=str(gnn_pooling_default or "add"),
                optimize_residual=bool(gnn_opt_residual),
                residual_default=bool(gnn_residual_default),
                optimize_head_mlp_layers=bool(gnn_opt_head_mlp_layers),
                head_mlp_layers_min=int(gnn_head_mlp_layers_min),
                head_mlp_layers_max=int(gnn_head_mlp_layers_max),
                head_mlp_layers_default=int(gnn_head_mlp_layers_default),
                optimize_gin_conv_mlp_layers=bool(gnn_opt_gin_conv_mlp_layers),
                gin_conv_mlp_layers_min=int(gnn_gin_conv_mlp_layers_min),
                gin_conv_mlp_layers_max=int(gnn_gin_conv_mlp_layers_max),
                gin_conv_mlp_layers_default=int(gnn_gin_conv_mlp_layers_default),
                # New parameters for other GNN architectures
                # GIN specific
                optimize_gin_train_eps=False, # Assuming these are not yet exposed in UI
                gin_train_eps_default=False,
                optimize_gin_eps=False,
                gin_eps_min=0.0,
                gin_eps_max=1.0,
                gin_eps_default=0.0,
                # GAT specific
                optimize_gat_heads=False,
                gat_heads_min=1,
                gat_heads_max=8,
                gat_heads_default=1,
                # SAGE specific
                optimize_sage_aggr=False,
                sage_aggr_choices=["mean", "max", "add"],
                sage_aggr_default="mean",
                # Transformer specific
                optimize_transformer_heads=False,
                transformer_heads_min=1,
                transformer_heads_max=8,
                transformer_heads_default=1,
                # TAG specific
                optimize_tag_k=False,
                tag_k_min=1,
                tag_k_max=3,
                tag_k_default=1,
                # ARMA specific
                optimize_arma_stacks=False,
                arma_num_stacks_min=1,
                arma_num_stacks_max=3,
                arma_num_stacks_default=1,
                optimize_arma_layers=False,
                arma_num_layers_min=1,
                arma_num_layers_max=3,
                arma_num_layers_default=1,
                # ChebNet specific
                optimize_cheb_k=False,
                cheb_k_min=1,
                cheb_k_max=3,
                cheb_k_default=1,
                # SuperGAT specific
                optimize_supergat_heads=False,
                supergat_heads_min=1,
                supergat_heads_max=8,
                supergat_heads_default=1,
                optimize_supergat_attention_type=False,
                supergat_attention_type_choices=["MX", "dot", "mlp"],
                supergat_attention_type_default="MX",
            )

            result = optimize_gnn_wandb(
                run_dir,
                config=cfg,
                project=project,
                entity=entity,
                api_key=api_key,
                sweep_name=sweep_name,
            )

    except Exception as e:
        raise gr.Error(str(e))

    best_params_df = pd.DataFrame([result.best_params])
    summary = (
        "### Hyperparameter Optimization Results\n"
        f"- **Best Trial**: #{result.best_trial_number}\n"
        f"- **Total Trials**: {result.n_trials}\n"
        f"- **Best Score**: {result.best_value:.4f}\n"
        "- **Best Parameters**:\n"
        "```json\n"
        f"{json.dumps(result.best_params, indent=2)}\n"
        "```\n"
    )

    artifacts = collect_artifacts(run_dir=str(result.run_dir))
    return (
        best_params_df,
        summary,
        str(result.best_params_path),
        str(result.run_dir),
        artifacts_table(artifacts),
    )


def run_prediction(
    model_type: str,
    model_selector: str | None,
    smiles_input: str,
    csv_file: str | None,
    batch_size: int,
):
    """Run prediction on uploaded SMILES."""

    run_dir = _resolve_current_run_dir(hint="Train a model first.")

    smiles_list: list[str] = []
    mol_ids: list[str] = []

    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            if "smiles" not in df.columns:
                raise gr.Error("CSV must have a 'smiles' column")
            smiles_list = df["smiles"].dropna().astype(str).tolist()
            if "molecule_id" in df.columns:
                mol_ids = df["molecule_id"].astype(str).tolist()
            elif "id" in df.columns:
                mol_ids = df["id"].astype(str).tolist()
        except Exception as e:
            raise gr.Error(f"Error reading CSV: {e}")
    elif smiles_input:
        import re

        tokens = re.split(r"[,\n\s]+", str(smiles_input).strip())
        smiles_list = [t.strip() for t in tokens if t.strip()]
    else:
        raise gr.Error("Please provide SMILES (text or CSV file)")

    if not smiles_list:
        raise gr.Error("No SMILES found in input")

    if not mol_ids or len(mol_ids) != len(smiles_list):
        mol_ids = [f"mol_{i}" for i in range(len(smiles_list))]

    try:
        if (model_type or "").startswith("RandomForest"):
            from dta_gnn.models.predict import predict_with_random_forest

            result = predict_with_random_forest(run_dir, smiles_list, mol_ids)
        elif (model_type or "").startswith("SVR"):
            from dta_gnn.models.predict import predict_with_svr

            result = predict_with_svr(run_dir, smiles_list, mol_ids)
        else:
            from dta_gnn.models.predict import predict_with_gnn
            
            # Extract architecture from model selector if provided
            # Format: "GNN (GIN)" -> "gin", "GNN (Transformer)" -> "transformer"
            architecture = None
            if model_selector and model_selector.startswith("GNN ("):
                # Extract architecture name from "GNN (ArchName)"
                arch_match = model_selector.replace("GNN (", "").replace(")", "").strip().lower()
                # Map display names to actual architecture names
                arch_map = {
                    "gin": "gin",
                    "gcn": "gcn",
                    "gat": "gat",
                    "sage": "sage",
                    "pna": "pna",
                    "transformer": "transformer",
                    "tag": "tag",
                    "arma": "arma",
                    "cheb": "cheb",
                    "supergat": "supergat",
                }
                architecture = arch_map.get(arch_match)
            
            result = predict_with_gnn(
                run_dir, smiles_list, mol_ids, batch_size=int(batch_size), architecture=architecture
            )
    except Exception as e:
        raise gr.Error(str(e))

    pred_path = run_dir / "predictions_new.csv"
    result.predictions.to_csv(pred_path, index=False)

    summary = (
        "### Prediction Results\n"
        f"- **Model**: {result.model_type}\n"
        f"- **Total Molecules**: {len(smiles_list)}\n"
        f"- **Successful**: {int(result.predictions['prediction'].notna().sum())}\n"
        f"- **Failed**: {int(result.predictions['prediction'].isna().sum())}\n"
    )

    return (
        result.predictions,
        summary,
        str(pred_path),
    )


@dataclass
class UIComponents:
    # Shared state
    dataset_state: gr.State

    # Dataset builder inputs
    target_id_type_input: gr.Radio
    targets_input: gr.Textbox
    source_input: gr.Radio
    sqlite_path_input: gr.components.Component
    split_method_input: gr.Dropdown
    test_size_input: gr.Slider
    val_size_input: gr.Slider
    split_year_input: gr.Slider
    temporal_controls: gr.Row
    build_btn: gr.Button

    # Dataset builder outputs
    output_df: gr.Dataframe
    output_summary: gr.Markdown
    plot_act: gr.Plot
    plot_split: gr.Plot
    download_btn: gr.File

    # Model training inputs/buttons
    baseline_model: gr.Dropdown
    rf_n_estimators: gr.Slider
    svr_kernel: gr.Dropdown
    svr_c: gr.Number
    svr_epsilon: gr.Number
    train_rf_btn: gr.Button
    train_gnn_btn: gr.Button
    gnn_arch: gr.Dropdown
    gnn_residual: gr.Checkbox
    gnn_epochs: gr.Slider
    gnn_batch_size: gr.Slider
    gnn_lr: gr.Number
    gnn_num_layers: gr.Slider
    gnn_dropout: gr.Slider
    gin_options: gr.Group
    gin_pooling: gr.Dropdown
    gnn_head_mlp_layers: gr.Slider
    gin_conv_mlp_layers: gr.Slider
    gin_train_eps: gr.Checkbox
    gin_eps: gr.Number
    gnn_embedding_dim: gr.Slider
    gnn_hidden_dim: gr.Slider

    # Model training outputs
    rf_task_text: gr.Textbox
    rf_metrics_df: gr.Dataframe
    rf_preds_preview: gr.Dataframe
    gnn_task_text: gr.Textbox
    gnn_metrics_df: gr.Dataframe
    gnn_preds_preview: gr.Dataframe

    # Embeddings
    gnn_embed_batch_size: gr.Slider
    extract_embeddings_btn: gr.Button
    gnn_embeddings_path: gr.Textbox
    gnn_embeddings_preview: gr.Dataframe
    gnn_embeddings_file: gr.File

    # Prediction
    pred_model_type: gr.Dropdown
    pred_model_selector: gr.Dropdown
    pred_batch_size: gr.Slider
    pred_smiles_text: gr.Textbox
    pred_csv_file: gr.File
    pred_run_btn: gr.Button
    pred_summary: gr.Markdown
    pred_results_df: gr.Dataframe
    pred_download: gr.File

    # Hyperopt
    ho_model_type: gr.Dropdown
    ho_optimizer: gr.Dropdown
    ho_wandb_section: gr.Group
    ho_wandb_project: gr.Textbox
    ho_wandb_entity: gr.Textbox
    ho_wandb_api_key: gr.Textbox
    ho_wandb_sweep_name: gr.Textbox
    ho_n_trials: gr.Slider
    ho_n_jobs: gr.Slider
    ho_rf_opt_n_est: gr.Checkbox
    ho_rf_n_est_min: gr.Slider
    ho_rf_n_est_max: gr.Slider
    ho_rf_opt_depth: gr.Checkbox
    ho_rf_depth_min: gr.Slider
    ho_rf_depth_max: gr.Slider
    ho_rf_opt_min_samp: gr.Checkbox
    ho_rf_min_samp_min: gr.Slider
    ho_rf_min_samp_max: gr.Slider

    ho_svr_opt_c: gr.Checkbox
    ho_svr_c_min: gr.Number
    ho_svr_c_max: gr.Number
    ho_svr_opt_epsilon: gr.Checkbox
    ho_svr_epsilon_min: gr.Number
    ho_svr_epsilon_max: gr.Number
    ho_svr_opt_kernel: gr.Checkbox
    ho_svr_kernel_choices: gr.CheckboxGroup
    ho_svr_kernel_default: gr.Dropdown

    ho_gnn_arch: gr.Dropdown
    ho_gnn_opt_epochs: gr.Checkbox
    ho_gnn_epochs_min: gr.Slider
    ho_gnn_epochs_max: gr.Slider
    ho_gnn_opt_lr: gr.Checkbox
    ho_gnn_lr_min: gr.Number
    ho_gnn_lr_max: gr.Number
    ho_gnn_opt_batch: gr.Checkbox
    ho_gnn_batch_min: gr.Slider
    ho_gnn_batch_max: gr.Slider
    ho_gnn_opt_emb: gr.Checkbox
    ho_gnn_emb_min: gr.Slider
    ho_gnn_emb_max: gr.Slider
    ho_gnn_opt_hidden: gr.Checkbox
    ho_gnn_hidden_min: gr.Slider
    ho_gnn_hidden_max: gr.Slider

    ho_gnn_opt_dropout: gr.Checkbox
    ho_gnn_dropout_min: gr.Slider
    ho_gnn_dropout_max: gr.Slider
    ho_gnn_dropout_default: gr.Slider

    ho_gnn_opt_pooling: gr.Checkbox
    ho_gnn_pooling_choices: gr.CheckboxGroup
    ho_gnn_pooling_default: gr.Dropdown

    ho_gnn_opt_residual: gr.Checkbox
    ho_gnn_residual_default: gr.Checkbox

    ho_gnn_opt_head_mlp_layers: gr.Checkbox
    ho_gnn_head_mlp_layers_min: gr.Slider
    ho_gnn_head_mlp_layers_max: gr.Slider
    ho_gnn_head_mlp_layers_default: gr.Slider

    ho_gnn_opt_gin_conv_mlp_layers: gr.Checkbox
    ho_gnn_gin_conv_mlp_layers_min: gr.Slider
    ho_gnn_gin_conv_mlp_layers_max: gr.Slider
    ho_gnn_gin_conv_mlp_layers_default: gr.Slider
    ho_rf_section: gr.Row
    ho_svr_section: gr.Row
    ho_gnn_section: gr.Row
    ho_gnn_section_2: gr.Row
    ho_gnn_section_3: gr.Row
    ho_gnn_section_4: gr.Row
    ho_gnn_section_gin: gr.Row
    ho_run_btn: gr.Button
    ho_best_params_df: gr.Dataframe
    ho_summary: gr.Markdown
    ho_best_params_file: gr.File

    # Visualization
    viz_method: gr.Dropdown
    color_by: gr.Radio
    model_for_pred: gr.Dropdown
    viz_perplexity: gr.Slider
    show_top_k: gr.Checkbox
    top_k_value: gr.Slider
    viz_button: gr.Button
    viz_plot: gr.Plot
    viz_status: gr.Markdown

    # Logs + artifacts
    logs_box: gr.Textbox
    run_dir_text: gr.Textbox
    artifacts_df: gr.Dataframe
    dataset_file: gr.File
    targets_file: gr.File
    compounds_file: gr.File
    molecule_features_file: gr.File
    protein_features_file: gr.File
    model_file: gr.File
    model_metrics_file: gr.File
    model_predictions_file: gr.File
    all_artifacts_zip: gr.File
    dataset_preview_df: gr.Dataframe
    targets_preview_df: gr.Dataframe
    compounds_preview_df: gr.Dataframe


def _sync_split_method(method: str):
    return gr.update(visible=(method == "temporal"))


def _sync_ho_model_type(t: str, gnn_arch: str):
    is_rf = (t or "").startswith("RandomForest")
    is_svr = (t or "").startswith("SVR")
    is_gnn = (t or "").startswith("GNN")
    is_gin = (gnn_arch or "").strip().lower() == "gin"
    return (
        gr.update(visible=is_rf),
        gr.update(visible=is_svr),
        gr.update(visible=is_gnn),
        gr.update(visible=is_gnn),
        gr.update(visible=is_gnn),
        gr.update(visible=is_gnn),
        gr.update(visible=is_gnn and is_gin),
    )


def _sync_ho_gnn_arch(gnn_arch: str, model_type: str):
    is_gnn = (model_type or "").startswith("GNN")
    is_gin = (gnn_arch or "").strip().lower() == "gin"
    return gr.update(visible=is_gnn and is_gin)


def _sync_ho_wandb_section(optimizer: str, model_type: str):
    opt = (optimizer or "").strip().lower()
    is_wandb = ("wandb" in opt) or ("w&b" in opt)
    return gr.update(visible=is_wandb)


def _sync_gnn_arch(arch: str):
    is_gin = (arch or "").strip().lower() == "gin"
    return gr.update(visible=is_gin)


def build_ui() -> tuple[gr.Blocks, UIComponents]:
    sqlite_db_choices = find_chembl_sqlite_dbs()

    with gr.Blocks(
        title="DTA-GNN: Target-Specific Binding Affinity Dataset Builder and GNN Trainer"
    ) as demo:
        with gr.Row(elem_id="app-header"):
            with gr.Column(scale=0, min_width=320):
                gr.Image(
                    value="assets/logo3.png",
                    show_label=False,
                    interactive=False,
                    container=False,
                    height=72,
                )
            with gr.Column(scale=4):
                gr.Markdown(
                    "# DTA-GNN: Target-Specific Binding Affinity Dataset Builder and GNN Trainer",
                    elem_id="app-title",
                )
                gr.Markdown(
                    "Build target-specific DTA datasets from ChEMBL and optimize/train/deploy GNN models. ",
                    elem_id="app-subtitle",
                )

        dataset_state = gr.State()

        with gr.Tabs():
            with gr.Tab("Home"):
                db_list_md = "\n".join([f"- `{p}`" for p in (sqlite_db_choices or [])])
                if not db_list_md:
                    db_list_md = "_None detected_"

                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown(
                            """
                            **Welcome to your research workspace.**  
                            This platform streamlines the pipeline for Drug-Target Affinity (DTA) deep learning: 
                            from curating high-quality ChEMBL datasets to training and optimizing state-of-the-art GNN models.
                            """
                        )

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown(
                                "#### 📂 Active Workspace\n"
                                "Runs are auto-saved to `runs/<timestamp>/`.\n"
                                "The app automatically tracks the latest run context.",
                            )

                with gr.Row():
                    with gr.Column():
                        gr.Image(
                            value="assets/overview.png",
                            show_label=False,
                            interactive=False,
                            container=False,
                        )

                gr.Markdown("---")

                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown("#### 1. Build Dataset")
                            gr.Markdown(
                                "Create high-quality benchmarks.\n\n"
                                "- **Inputs**: ChEMBL IDs or UniProt Accessions\n"
                                "- **Strategies**: Temporal, Scaffold, or Random splits\n"
                                "- **Source**: Web API or Local SQLite"
                            )

                    with gr.Column():
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown("#### 2. Train Models")
                            gr.Markdown(
                                "Train robust baselines and GNNs.\n\n"
                                "- **Baselines**: RandomForest & SVR (ECFP4)\n"
                                "- **GNNs**: GIN, GCN, GAT, SAGE, PNA, Transformer, TAG, ARMA, Cheb, SuperGAT\n"
                                "- **Features**: 2D Molecular Graphs (PyG)"
                            )

                    with gr.Column():
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown("#### 3. Optimize & Deploy")
                            gr.Markdown(
                                "Tune performance and inference.\n\n"
                                "- **HPO**: Bayesian Optimization via W&B Sweeps\n"
                                "- **Predict**: Score new SMILES candidates\n"
                                "- **Export**: Download full artifact packages"
                            )

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown("#### 💾 Local Data Sources")
                            gr.Markdown(
                                "For fastest build performance, place ChEMBL SQLite databases associated with your version in `chembl_dbs/`."
                            )
                            gr.Markdown(f"**Detected DBs:**\n{db_list_md}")

                    with gr.Column(scale=2):
                        with gr.Accordion("💡 Troubleshooting & Tips", open=False):
                            gr.Markdown(
                                """
                                - **Start Fresh**: If you encounter issues, try building a new dataset first to ensure the run context is fresh.
                                - **HPO Requirements**: GNN sweeps require a `val_size > 0`. SVR requires a regression dataset.
                                - **Performance**: GNN training on CPU can be slow; ensure PyTorch detects CUDA/MPS if available.
                                - **Imports**: Always launch via the project virtual environment to ensure all dependencies are loaded.
                                """
                            )

            with gr.Tab("Build"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            with gr.Accordion("Dataset Inputs", open=True):
                                target_id_type_input = gr.Radio(
                                    choices=["ChEMBL", "UniProt"],
                                    value="ChEMBL",
                                    label="Target ID Type",
                                    info="ChEMBL: CHEMBL203. UniProt: P00533, Q9Y6K9 (will be mapped to ChEMBL targets).",
                                )

                                targets_input = gr.Textbox(
                                    label="Target IDs",
                                    placeholder="CHEMBL203, CHEMBL220  (or UniProt: P00533)",
                                    value="CHEMBL203",
                                    info="Comma-separated target IDs (format depends on Target ID Type above).",
                                )

                                source_input = gr.Radio(
                                    ["web", "sqlite"],
                                    label="Data Source",
                                    value="sqlite",
                                    info="Use 'sqlite' for local ChEMBL DBs under chembl_dbs/.",
                                )
                                if sqlite_db_choices:
                                    sqlite_path_input: gr.components.Component = (
                                        gr.Dropdown(
                                            choices=sqlite_db_choices,
                                            value=sqlite_db_choices[0],
                                            label="SQLite DB",
                                            allow_custom_value=True,
                                            info="Pick an existing DB or paste a path.",
                                        )
                                    )
                                else:
                                    sqlite_path_input = gr.Textbox(
                                        value="",
                                        label="SQLite Path",
                                        placeholder="/path/to/chembl_<version>.db",
                                        info="Required when Data Source is 'sqlite'.",
                                    )

                            with gr.Accordion("Splitting", open=True):
                                split_method_input = gr.Dropdown(
                                    choices=["random", "scaffold", "temporal"],
                                    value="scaffold",
                                    label="Split Strategy",
                                    info="Scaffold split is recommended for molecular generalization.",
                                )
                                with gr.Row():
                                    test_size_input = gr.Slider(
                                        minimum=0.1,
                                        maximum=0.5,
                                        value=0.2,
                                        step=0.05,
                                        label="Test Size",
                                    )
                                    val_size_input = gr.Slider(
                                        minimum=0.0,
                                        maximum=0.5,
                                        value=0.1,
                                        step=0.05,
                                        label="Validation Size",
                                    )

                            with gr.Row(visible=False) as temporal_controls:
                                split_year_input = gr.Slider(
                                    minimum=1990,
                                    maximum=2024,
                                    value=2022,
                                    step=1,
                                    label="Split Year (Temporal Only)",
                                )

                            build_btn = gr.Button("Build Dataset", variant="primary")
                            download_btn = gr.File(label="Download dataset.csv")

                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["section-card"]):
                            output_summary = gr.Markdown(label="Summary")
                            output_df = gr.Dataframe(
                                label="Dataset Preview",
                                interactive=False,
                                elem_classes=["compact-table"],
                            )
                            with gr.Row():
                                plot_act = gr.Plot(label="Activity Distribution")
                                plot_split = gr.Plot(label="Split Sizes")

            with gr.Tab("Train"):
                with gr.Tabs():
                    with gr.Tab("Baseline"):
                        gr.Markdown(
                            "Train a baseline model using ECFP4 (Morgan) fingerprints."
                        )

                        baseline_model = gr.Dropdown(
                            choices=["RF (ECFP4)", "SVR (ECFP4)"],
                            value="RF (ECFP4)",
                            label="Baseline Model",
                        )

                        rf_n_estimators = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=500,
                            step=50,
                            label="RF: n_estimators",
                            info="Used only when Baseline Model is RF.",
                        )

                        with gr.Row():
                            svr_kernel = gr.Dropdown(
                                choices=["rbf", "linear"],
                                value="rbf",
                                label="SVR: kernel",
                                info="Used only when Baseline Model is SVR.",
                            )
                            svr_c = gr.Number(value=10.0, label="SVR: C")
                            svr_epsilon = gr.Number(value=0.1, label="SVR: epsilon")

                        train_rf_btn = gr.Button("Train Baseline", variant="primary")

                        rf_task_text = gr.Textbox(
                            label="Task Type", interactive=False, value=""
                        )
                        rf_metrics_df = gr.Dataframe(label="Metrics", interactive=False)
                        with gr.Accordion("Preview predictions (val/test)", open=False):
                            rf_preds_preview = gr.Dataframe(interactive=False)

                    with gr.Tab("GNN (2D)"):
                        gr.Markdown(
                            "Train a 2D molecular graph neural network (select architecture below)."
                        )

                        gnn_arch = gr.Dropdown(
                            choices=[
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
                            ],
                            value="gin",
                            label="architecture",
                            info="GNN architecture: GIN, GCN, GAT, GraphSAGE, PNA, Transformer, TAG, ARMA, Cheb, SuperGAT.",
                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gnn_epochs = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    value=10,
                                    step=1,
                                    label="epochs",
                                )
                            with gr.Column(scale=1):
                                gnn_batch_size = gr.Slider(
                                    minimum=1,
                                    maximum=256,
                                    value=64,
                                    step=1,
                                    label="batch_size",
                                )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gnn_lr = gr.Number(
                                    value=0.001,
                                    label="learning_rate",
                                )
                            with gr.Column(scale=1):
                                gnn_num_layers = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=5,
                                    step=1,
                                    label="num_layers",
                                )

                        gnn_dropout = gr.Slider(
                            minimum=0.0,
                            maximum=0.6,
                            value=0.1,
                            step=0.05,
                            label="dropout",
                        )

                        with gr.Row():
                            with gr.Column(scale=1):
                                gin_pooling = gr.Dropdown(
                                    choices=["add", "mean", "max", "attention"],
                                    value="add",
                                    label="pooling",
                                )
                            with gr.Column(scale=1):
                                gnn_residual = gr.Checkbox(
                                    value=False,
                                    label="residual (skip connections)",
                                )

                        gin_options = gr.Group(visible=True)
                        with gin_options:
                            gr.Markdown("### GIN-specific options")
                            with gr.Row():
                                gin_conv_mlp_layers = gr.Slider(
                                    minimum=1,
                                    maximum=4,
                                    value=2,
                                    step=1,
                                    label="conv_mlp_layers",
                                    info="Applies to GIN only (MLP inside each message passing layer).",
                                )
                                gin_train_eps = gr.Checkbox(
                                    value=False,
                                    label="train_eps",
                                    info="Whether to learn the GIN epsilon parameter.",
                                )
                            gin_eps = gr.Number(
                                value=0.0,
                                label="eps",
                                info="Initial epsilon for GIN (used when train_eps is enabled).",
                            )

                        with gr.Accordion("Architecture", open=False):
                            gnn_embedding_dim = gr.Slider(
                                minimum=16,
                                maximum=512,
                                value=128,
                                step=16,
                                label="embedding_dim",
                            )
                            gnn_hidden_dim = gr.Slider(
                                minimum=16,
                                maximum=512,
                                value=128,
                                step=16,
                                label="hidden_dim",
                            )
                            gnn_head_mlp_layers = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=2,
                                step=1,
                                label="head_mlp_layers",
                                info="MLP depth of the prediction head (applies to all architectures).",
                            )

                        train_gnn_btn = gr.Button("Train GNN", variant="primary")

                        gnn_task_text = gr.Textbox(
                            label="Task Type", interactive=False, value=""
                        )
                        gnn_metrics_df = gr.Dataframe(
                            label="Metrics", interactive=False
                        )
                        with gr.Accordion("Preview predictions (val/test)", open=False):
                            gnn_preds_preview = gr.Dataframe(interactive=False)

                        with gr.Accordion("Extract Molecule Embeddings", open=False):
                            gnn_embed_batch_size = gr.Slider(
                                minimum=1,
                                maximum=1024,
                                value=256,
                                step=1,
                                label="Batch Size",
                            )
                            extract_embeddings_btn = gr.Button(
                                "Extract GNN Embeddings", variant="secondary"
                            )
                            gnn_embeddings_path = gr.Textbox(
                                label="Embeddings path (molecule_embeddings.npz)",
                                interactive=False,
                                value="",
                            )
                            gnn_embeddings_preview = gr.Dataframe(
                                label="Embeddings preview (first 8 dims)",
                                interactive=False,
                            )
                            gnn_embeddings_file = gr.File(
                                label="Download molecule_embeddings.npz",
                                interactive=False,
                            )

            with gr.Tab("Predict"):
                gr.Markdown(
                    "Make predictions on new molecules using the current run's trained model artifacts."
                )

                with gr.Row():
                    pred_model_type = gr.Dropdown(
                        choices=["RandomForest (ECFP4)", "SVR (ECFP4)", "GNN (2D)"],
                        value="RandomForest (ECFP4)",
                        label="Model Type",
                    )
                    # Initialize with available models
                    try:
                        initial_choices, initial_default = _update_model_choices("RandomForest (ECFP4)")
                    except Exception:
                        initial_choices, initial_default = ["RandomForest (ECFP4)"], "RandomForest (ECFP4)"
                    
                    pred_model_selector = gr.Dropdown(
                        choices=initial_choices,
                        value=initial_default,
                        label="Select Trained Model",
                        info="Choose from available trained models in the current run",
                        interactive=True,
                    )
                    pred_batch_size = gr.Slider(
                        minimum=1,
                        maximum=512,
                        value=64,
                        step=1,
                        label="Batch Size (GNN only)",
                    )

                gr.Markdown("### Input SMILES")
                with gr.Row():
                    with gr.Column():
                        pred_smiles_text = gr.Textbox(
                            lines=8,
                            label="SMILES (one per line or comma-separated)",
                            placeholder="CCO\nCCN\nC1CCCCC1\n...",
                        )
                    with gr.Column():
                        pred_csv_file = gr.File(
                            label="Or upload CSV (must have 'smiles' column)",
                            type="filepath",
                        )

                pred_run_btn = gr.Button("Run Prediction", variant="primary")

                pred_summary = gr.Markdown(label="Prediction Summary")
                pred_results_df = gr.Dataframe(
                    label="Predictions",
                    interactive=False,
                )
                pred_download = gr.File(
                    label="Download predictions CSV",
                    interactive=False,
                )

            with gr.Tab("HPO"):
                gr.Markdown("## Hyperparameter Optimization (HPO)")
                gr.Markdown(
                    "Optimize hyperparameters using cross-validation. Select which parameters to tune and specify their search ranges."
                )

                with gr.Row():
                    ho_model_type = gr.Dropdown(
                        choices=["RandomForest (ECFP4)", "SVR (ECFP4)", "GNN (2D)"],
                        value="GNN (2D)",
                        label="Model Type for Optimization",
                    )
                    with gr.Column(scale=1):
                        ho_optimizer = gr.Dropdown(
                            choices=["W&B Sweeps (Bayes)"],
                            value="W&B Sweeps (Bayes)",
                            label="Optimizer Backend",
                            info="W&B Sweeps (Bayes) is supported for RF, SVR, and GNN.",
                        )
                        ho_n_trials = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=20,
                            step=5,
                            label="Number of Trials",
                        )
                        ho_n_jobs = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=1,
                            step=1,
                            label="Parallel Jobs",
                        )

                with gr.Group(visible=True) as ho_wandb_section:
                    gr.Markdown("### Weights & Biases")
                    with gr.Row():
                        ho_wandb_project = gr.Textbox(
                            value="dta_gnn",
                            label="W&B Project",
                        )
                        ho_wandb_entity = gr.Textbox(
                            value="",
                            label="W&B Entity (optional)",
                        )
                    with gr.Row():
                        ho_wandb_api_key = gr.Textbox(
                            value="",
                            label="W&B API Key",
                            type="password",
                            placeholder="wandb_...",
                            info="Required for W&B Sweeps unless you have a persistent local login (wandb login).",
                        )
                        ho_wandb_sweep_name = gr.Textbox(
                            value="",
                            label="Sweep Name (optional)",
                            placeholder="dta_gnn_gnn_gin",
                        )

                gr.Markdown("### RandomForest parameters")
                with gr.Row(visible=False) as ho_rf_section:
                    with gr.Column():
                        ho_rf_opt_n_est = gr.Checkbox(
                            label="Optimize n_estimators",
                            value=False,
                        )
                        with gr.Row():
                            ho_rf_n_est_min = gr.Slider(
                                minimum=10,
                                maximum=500,
                                value=50,
                                step=10,
                                label="n_estimators min",
                            )
                            ho_rf_n_est_max = gr.Slider(
                                minimum=100,
                                maximum=1000,
                                value=500,
                                step=50,
                                label="n_estimators max",
                            )
                    with gr.Column():
                        ho_rf_opt_depth = gr.Checkbox(
                            label="Optimize max_depth",
                            value=False,
                        )
                        with gr.Row():
                            ho_rf_depth_min = gr.Slider(
                                minimum=2,
                                maximum=30,
                                value=5,
                                step=1,
                                label="max_depth min",
                            )
                            ho_rf_depth_max = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=50,
                                step=1,
                                label="max_depth max",
                            )
                    with gr.Column():
                        ho_rf_opt_min_samp = gr.Checkbox(
                            label="Optimize min_samples_split",
                            value=False,
                        )
                        with gr.Row():
                            ho_rf_min_samp_min = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=2,
                                step=1,
                                label="min_samples_split min",
                            )
                            ho_rf_min_samp_max = gr.Slider(
                                minimum=5,
                                maximum=30,
                                value=20,
                                step=1,
                                label="min_samples_split max",
                            )

                gr.Markdown("### SVR parameters")
                with gr.Row(visible=False) as ho_svr_section:
                    with gr.Column():
                        ho_svr_opt_c = gr.Checkbox(label="Optimize C", value=False)
                        with gr.Row():
                            ho_svr_c_min = gr.Number(value=0.1, label="C min")
                            ho_svr_c_max = gr.Number(value=100.0, label="C max")

                    with gr.Column():
                        ho_svr_opt_epsilon = gr.Checkbox(
                            label="Optimize epsilon", value=False
                        )
                        with gr.Row():
                            ho_svr_epsilon_min = gr.Number(
                                value=0.01, label="epsilon min"
                            )
                            ho_svr_epsilon_max = gr.Number(
                                value=0.2, label="epsilon max"
                            )

                    with gr.Column():
                        ho_svr_opt_kernel = gr.Checkbox(
                            label="Optimize kernel", value=False
                        )
                        ho_svr_kernel_choices = gr.CheckboxGroup(
                            choices=["rbf", "linear"],
                            value=["rbf", "linear"],
                            label="Allowed kernel choices",
                        )
                        ho_svr_kernel_default = gr.Dropdown(
                            choices=["rbf", "linear"],
                            value="rbf",
                            label="kernel default (used when not optimizing)",
                        )

                gr.Markdown("### GNN parameters")
                with gr.Row(visible=True) as ho_gnn_section:
                    with gr.Column():
                        ho_gnn_arch = gr.Dropdown(
                            choices=[
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
                            ],
                            value="gin",
                            label="architecture",
                            info="Select which GNN backbone to use during HPO.",
                        )

                    with gr.Column():
                        ho_gnn_opt_epochs = gr.Checkbox(
                            label="Optimize epochs",
                            value=False,
                        )
                        with gr.Row():
                            ho_gnn_epochs_min = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=5,
                                step=1,
                                label="epochs min",
                            )
                            ho_gnn_epochs_max = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=50,
                                step=1,
                                label="epochs max",
                            )
                    with gr.Column():
                        ho_gnn_opt_lr = gr.Checkbox(
                            label="Optimize learning_rate",
                            value=False,
                        )
                        with gr.Row():
                            ho_gnn_lr_min = gr.Number(
                                value=0.00001,
                                label="lr min",
                            )
                            ho_gnn_lr_max = gr.Number(
                                value=0.01,
                                label="lr max",
                            )
                    with gr.Column():
                        ho_gnn_opt_batch = gr.Checkbox(
                            label="Optimize batch_size",
                            value=False,
                        )
                        with gr.Row():
                            ho_gnn_batch_min = gr.Slider(
                                minimum=8,
                                maximum=128,
                                value=16,
                                step=8,
                                label="batch_size min",
                            )
                            ho_gnn_batch_max = gr.Slider(
                                minimum=32,
                                maximum=512,
                                value=256,
                                step=32,
                                label="batch_size max",
                            )

                with gr.Row(visible=True) as ho_gnn_section_2:
                    with gr.Column():
                        ho_gnn_opt_emb = gr.Checkbox(
                            label="Optimize embedding_dim",
                            value=False,
                        )
                        with gr.Row():
                            ho_gnn_emb_min = gr.Slider(
                                minimum=16,
                                maximum=256,
                                value=32,
                                step=16,
                                label="embedding_dim min",
                            )
                            ho_gnn_emb_max = gr.Slider(
                                minimum=64,
                                maximum=512,
                                value=512,
                                step=16,
                                label="embedding_dim max",
                            )
                    with gr.Column():
                        ho_gnn_opt_hidden = gr.Checkbox(
                            label="Optimize hidden_dim",
                            value=False,
                        )
                        with gr.Row():
                            ho_gnn_hidden_min = gr.Slider(
                                minimum=16,
                                maximum=256,
                                value=32,
                                step=16,
                                label="hidden_dim min",
                            )
                            ho_gnn_hidden_max = gr.Slider(
                                minimum=64,
                                maximum=512,
                                value=512,
                                step=16,
                                label="hidden_dim max",
                            )

                gr.Markdown("### GNN architecture knobs")
                with gr.Row(visible=True) as ho_gnn_section_3:
                    with gr.Column():
                        ho_gnn_opt_dropout = gr.Checkbox(
                            label="Optimize dropout", value=False
                        )
                        with gr.Row():
                            ho_gnn_dropout_min = gr.Slider(
                                minimum=0.0,
                                maximum=0.9,
                                value=0.0,
                                step=0.05,
                                label="dropout min",
                            )
                            ho_gnn_dropout_max = gr.Slider(
                                minimum=0.0,
                                maximum=0.9,
                                value=0.6,
                                step=0.05,
                                label="dropout max",
                            )
                        ho_gnn_dropout_default = gr.Slider(
                            minimum=0.0,
                            maximum=0.9,
                            value=0.1,
                            step=0.05,
                            label="dropout default (used when not optimizing)",
                        )

                    with gr.Column():
                        ho_gnn_opt_pooling = gr.Checkbox(
                            label="Optimize pooling", value=False
                        )
                        ho_gnn_pooling_choices = gr.CheckboxGroup(
                            choices=["add", "mean", "max", "attention"],
                            value=["add", "mean", "max", "attention"],
                            label="Allowed pooling choices",
                        )
                        ho_gnn_pooling_default = gr.Dropdown(
                            choices=["add", "mean", "max", "attention"],
                            value="add",
                            label="pooling default (used when not optimizing)",
                        )

                    with gr.Column():
                        ho_gnn_opt_residual = gr.Checkbox(
                            label="Optimize residual (skip)", value=False
                        )
                        ho_gnn_residual_default = gr.Checkbox(
                            label="residual default (used when not optimizing)",
                            value=False,
                        )

                with gr.Row(visible=True) as ho_gnn_section_4:
                    with gr.Column():
                        ho_gnn_opt_head_mlp_layers = gr.Checkbox(
                            label="Optimize head_mlp_layers",
                            value=False,
                        )
                        with gr.Row():
                            ho_gnn_head_mlp_layers_min = gr.Slider(
                                minimum=1,
                                maximum=6,
                                value=1,
                                step=1,
                                label="head_mlp_layers min",
                            )
                            ho_gnn_head_mlp_layers_max = gr.Slider(
                                minimum=1,
                                maximum=6,
                                value=4,
                                step=1,
                                label="head_mlp_layers max",
                            )
                        ho_gnn_head_mlp_layers_default = gr.Slider(
                            minimum=1,
                            maximum=6,
                            value=2,
                            step=1,
                            label="head_mlp_layers default (used when not optimizing)",
                        )

                with gr.Row(visible=False) as ho_gnn_section_gin:
                    with gr.Column():
                        ho_gnn_opt_gin_conv_mlp_layers = gr.Checkbox(
                            label="Optimize gin_conv_mlp_layers (GIN only)",
                            value=False,
                        )
                        with gr.Row():
                            ho_gnn_gin_conv_mlp_layers_min = gr.Slider(
                                minimum=1,
                                maximum=6,
                                value=1,
                                step=1,
                                label="gin_conv_mlp_layers min",
                            )
                            ho_gnn_gin_conv_mlp_layers_max = gr.Slider(
                                minimum=1,
                                maximum=6,
                                value=4,
                                step=1,
                                label="gin_conv_mlp_layers max",
                            )
                        ho_gnn_gin_conv_mlp_layers_default = gr.Slider(
                            minimum=1,
                            maximum=6,
                            value=2,
                            step=1,
                            label="gin_conv_mlp_layers default (used when not optimizing)",
                        )

                ho_run_btn = gr.Button(
                    "Run Hyperparameter Optimization", variant="secondary"
                )

                ho_best_params_df = gr.Dataframe(
                    label="Best Parameters",
                    interactive=False,
                )
                ho_summary = gr.Markdown(label="Optimization Results")
                ho_best_params_file = gr.File(
                    label="Download best_params.json",
                    interactive=False,
                )

            with gr.Tab("Visualization"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown("### Visualization Settings")
                            
                            viz_method = gr.Dropdown(
                                choices=["t-SNE", "PCA"],
                                value="t-SNE",
                                label="Dimensionality Reduction Method",
                                info="t-SNE for non-linear, PCA for linear projections"
                            )
                            
                            color_by = gr.Radio(
                                choices=["Split (Train/Val/Test)", "Ground Truth (Affinity)", "Model Predictions"],
                                value="Split (Train/Val/Test)",
                                label="Color By",
                                info="Choose how to color the visualization"
                            )
                            
                            model_for_pred = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Model for Predictions",
                                visible=False,
                                info="Select a trained model to color by predictions"
                            )
                            
                            viz_perplexity = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=30,
                                step=1,
                                label="t-SNE Perplexity",
                                info="Only used for t-SNE method",
                                visible=True
                            )
                            
                            show_top_k = gr.Checkbox(
                                value=False,
                                label="Show Top-K Test Predictions",
                                info="Filter to show only top-k highest binding affinity predictions from test set"
                            )
                            
                            top_k_value = gr.Slider(
                                minimum=10,
                                maximum=1000,
                                value=100,
                                step=10,
                                label="K (Number of Top Predictions)",
                                info="Only used when 'Show Top-K Test Predictions' is enabled",
                                visible=False
                            )
                            
                            viz_button = gr.Button(
                                "Generate Visualization",
                                variant="primary"
                            )
                            
                            def update_top_k_visibility(show_top_k_val):
                                return gr.update(visible=show_top_k_val)
                            
                            show_top_k.change(
                                update_top_k_visibility,
                                inputs=[show_top_k],
                                outputs=[top_k_value]
                            )
                            
                            def update_viz_controls(method, color_by_val, show_top_k_val):
                                return [
                                    gr.update(visible=(method == "t-SNE")),
                                    gr.update(visible=(color_by_val == "Model Predictions")),
                                    gr.update(visible=(color_by_val == "Model Predictions" and show_top_k_val))
                                ]
                            
                            def update_top_k_visibility(color_by_val, show_top_k_val):
                                return gr.update(visible=(color_by_val == "Model Predictions" and show_top_k_val))
                            
                            viz_method.change(
                                update_viz_controls,
                                inputs=[viz_method, color_by, show_top_k],
                                outputs=[viz_perplexity, model_for_pred, top_k_value]
                            )
                            
                            color_by.change(
                                update_viz_controls,
                                inputs=[viz_method, color_by, show_top_k],
                                outputs=[viz_perplexity, model_for_pred, top_k_value]
                            )
                            
                            show_top_k.change(
                                update_top_k_visibility,
                                inputs=[color_by, show_top_k],
                                outputs=[top_k_value]
                            )
                    
                    with gr.Column(scale=2):
                        viz_plot = gr.Plot(
                            label="Embedding Visualization",
                            show_label=True
                        )
                        viz_status = gr.Markdown("")

            with gr.Tab("Artifacts & Logs"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            logs_box = gr.Textbox(
                                label="Run Logs",
                                lines=26,
                                interactive=False,
                                value="",
                                info="Live logs for long-running operations like dataset build.",
                            )
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            run_dir_text = gr.Textbox(
                                label="Current Run Folder",
                                interactive=False,
                                value="",
                            )
                            artifacts_df = gr.Dataframe(
                                label="Artifacts",
                                interactive=False,
                                headers=["artifact", "path"],
                                elem_classes=["compact-table"],
                            )

                            with gr.Accordion("Downloads", open=True):
                                dataset_file = gr.File(
                                    label="dataset.csv", interactive=False
                                )
                                targets_file = gr.File(label="targets.csv")
                                compounds_file = gr.File(label="compounds.csv")
                                molecule_features_file = gr.File(
                                    label="molecule_features.csv"
                                )
                                protein_features_file = gr.File(
                                    label="protein_features.csv"
                                )
                                model_file = gr.File(label="model file")
                                model_metrics_file = gr.File(label="model_metrics.json")
                                model_predictions_file = gr.File(
                                    label="model_predictions.csv"
                                )
                                all_artifacts_zip = gr.File(label="artifacts.zip")

                            with gr.Accordion("Preview dataset.csv", open=False):
                                dataset_preview_df = gr.Dataframe(interactive=False)
                            with gr.Accordion("Preview targets.csv", open=False):
                                targets_preview_df = gr.Dataframe(interactive=False)
                            with gr.Accordion("Preview compounds.csv", open=False):
                                compounds_preview_df = gr.Dataframe(interactive=False)

            with gr.Tab("Contact & Citation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown(
                                """
                                ## 👤 Contact Information
                                
                                **Gökhan Özsari**  
                                *Developer & Maintainer*
                                
                                ### 📧 Email
                                [gokhan.ozsari@chalmers.se](mailto:gokhan.ozsari@chalmers.se)
                                
                                ---
                                
                                ### 💬 Get in Touch
                                - **Questions**: Feel free to reach out via email
                                - **Feature Requests**: Submit your ideas on GitHub
                                - **Contributions**: Pull requests are welcome!
                                """
                            )
                    
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown(
                                """
                                ## 📚 Citation
                                
                                If you use **DTA-GNN** in your research, please cite:
                                """
                            )
                            
                            citation_bibtex = """@software{dta_gnn,
  title = {DTA-GNN: Target-Specific Binding Affinity Dataset Builder and GNN Trainer},
  author = {Özsari, Gökhan},
  year = {2026},
  url = {https://github.com/gozsari/DTA-GNN}
}"""
                            
                            citation_textbox = gr.Textbox(
                                value=citation_bibtex,
                                label="BibTeX Citation",
                                lines=7,
                                interactive=True,
                                info="Copy the citation text above for use in your publications.",
                                elem_classes=["citation-box"]
                            )
                            
                            gr.Markdown("---")
                            
                            gr.Markdown(
                                """
                                ### 📄 License
                                
                                This project is licensed under the **MIT License**.
                                
                                See the [LICENSE](https://github.com/gozsari/DTA-GNN/blob/main/LICENSE) file for details.
                                """
                            )
                
  
                
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes=["section-card"]):
                            gr.Markdown("### 🔗 Quick Links")
                            gr.Markdown(
                                """
                                - **📦 GitHub Repository**: [github.com/gozsari/DTA-GNN](https://github.com/gozsari/DTA-GNN)
                                - **📖 Documentation**: Available in the repository's `docs/` directory
                                - **🐛 Report Issues**: [GitHub Issues](https://github.com/gozsari/DTA-GNN/issues)
                                - **💡 Feature Requests**: [GitHub Discussions](https://github.com/gozsari/DTA-GNN/discussions)
                                """
                            )
                    
                    

        ui = UIComponents(
            dataset_state=dataset_state,
            target_id_type_input=target_id_type_input,
            targets_input=targets_input,
            source_input=source_input,
            sqlite_path_input=sqlite_path_input,
            split_method_input=split_method_input,
            test_size_input=test_size_input,
            val_size_input=val_size_input,
            split_year_input=split_year_input,
            temporal_controls=temporal_controls,
            build_btn=build_btn,
            output_df=output_df,
            output_summary=output_summary,
            plot_act=plot_act,
            plot_split=plot_split,
            download_btn=download_btn,
            baseline_model=baseline_model,
            rf_n_estimators=rf_n_estimators,
            svr_kernel=svr_kernel,
            svr_c=svr_c,
            svr_epsilon=svr_epsilon,
            train_rf_btn=train_rf_btn,
            train_gnn_btn=train_gnn_btn,
            gnn_arch=gnn_arch,
            gnn_residual=gnn_residual,
            gnn_epochs=gnn_epochs,
            gnn_batch_size=gnn_batch_size,
            gnn_lr=gnn_lr,
            gnn_num_layers=gnn_num_layers,
            gnn_dropout=gnn_dropout,
            gin_options=gin_options,
            gin_pooling=gin_pooling,
            gnn_head_mlp_layers=gnn_head_mlp_layers,
            gin_conv_mlp_layers=gin_conv_mlp_layers,
            gin_train_eps=gin_train_eps,
            gin_eps=gin_eps,
            gnn_embedding_dim=gnn_embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            rf_task_text=rf_task_text,
            rf_metrics_df=rf_metrics_df,
            rf_preds_preview=rf_preds_preview,
            gnn_task_text=gnn_task_text,
            gnn_metrics_df=gnn_metrics_df,
            gnn_preds_preview=gnn_preds_preview,
            gnn_embed_batch_size=gnn_embed_batch_size,
            extract_embeddings_btn=extract_embeddings_btn,
            gnn_embeddings_path=gnn_embeddings_path,
            gnn_embeddings_preview=gnn_embeddings_preview,
            gnn_embeddings_file=gnn_embeddings_file,
            pred_model_type=pred_model_type,
            pred_model_selector=pred_model_selector,
            pred_batch_size=pred_batch_size,
            pred_smiles_text=pred_smiles_text,
            pred_csv_file=pred_csv_file,
            pred_run_btn=pred_run_btn,
            pred_summary=pred_summary,
            pred_results_df=pred_results_df,
            pred_download=pred_download,
            ho_model_type=ho_model_type,
            ho_optimizer=ho_optimizer,
            ho_wandb_section=ho_wandb_section,
            ho_wandb_project=ho_wandb_project,
            ho_wandb_entity=ho_wandb_entity,
            ho_wandb_api_key=ho_wandb_api_key,
            ho_wandb_sweep_name=ho_wandb_sweep_name,
            ho_n_trials=ho_n_trials,
            ho_n_jobs=ho_n_jobs,
            ho_rf_opt_n_est=ho_rf_opt_n_est,
            ho_rf_n_est_min=ho_rf_n_est_min,
            ho_rf_n_est_max=ho_rf_n_est_max,
            ho_rf_opt_depth=ho_rf_opt_depth,
            ho_rf_depth_min=ho_rf_depth_min,
            ho_rf_depth_max=ho_rf_depth_max,
            ho_rf_opt_min_samp=ho_rf_opt_min_samp,
            ho_rf_min_samp_min=ho_rf_min_samp_min,
            ho_rf_min_samp_max=ho_rf_min_samp_max,
            ho_svr_opt_c=ho_svr_opt_c,
            ho_svr_c_min=ho_svr_c_min,
            ho_svr_c_max=ho_svr_c_max,
            ho_svr_opt_epsilon=ho_svr_opt_epsilon,
            ho_svr_epsilon_min=ho_svr_epsilon_min,
            ho_svr_epsilon_max=ho_svr_epsilon_max,
            ho_svr_opt_kernel=ho_svr_opt_kernel,
            ho_svr_kernel_choices=ho_svr_kernel_choices,
            ho_svr_kernel_default=ho_svr_kernel_default,
            ho_gnn_arch=ho_gnn_arch,
            ho_gnn_opt_epochs=ho_gnn_opt_epochs,
            ho_gnn_epochs_min=ho_gnn_epochs_min,
            ho_gnn_epochs_max=ho_gnn_epochs_max,
            ho_gnn_opt_lr=ho_gnn_opt_lr,
            ho_gnn_lr_min=ho_gnn_lr_min,
            ho_gnn_lr_max=ho_gnn_lr_max,
            ho_gnn_opt_batch=ho_gnn_opt_batch,
            ho_gnn_batch_min=ho_gnn_batch_min,
            ho_gnn_batch_max=ho_gnn_batch_max,
            ho_gnn_opt_emb=ho_gnn_opt_emb,
            ho_gnn_emb_min=ho_gnn_emb_min,
            ho_gnn_emb_max=ho_gnn_emb_max,
            ho_gnn_opt_hidden=ho_gnn_opt_hidden,
            ho_gnn_hidden_min=ho_gnn_hidden_min,
            ho_gnn_hidden_max=ho_gnn_hidden_max,
            ho_gnn_opt_dropout=ho_gnn_opt_dropout,
            ho_gnn_dropout_min=ho_gnn_dropout_min,
            ho_gnn_dropout_max=ho_gnn_dropout_max,
            ho_gnn_dropout_default=ho_gnn_dropout_default,
            ho_gnn_opt_pooling=ho_gnn_opt_pooling,
            ho_gnn_pooling_choices=ho_gnn_pooling_choices,
            ho_gnn_pooling_default=ho_gnn_pooling_default,
            ho_gnn_opt_residual=ho_gnn_opt_residual,
            ho_gnn_residual_default=ho_gnn_residual_default,
            ho_gnn_opt_head_mlp_layers=ho_gnn_opt_head_mlp_layers,
            ho_gnn_head_mlp_layers_min=ho_gnn_head_mlp_layers_min,
            ho_gnn_head_mlp_layers_max=ho_gnn_head_mlp_layers_max,
            ho_gnn_head_mlp_layers_default=ho_gnn_head_mlp_layers_default,
            ho_gnn_opt_gin_conv_mlp_layers=ho_gnn_opt_gin_conv_mlp_layers,
            ho_gnn_gin_conv_mlp_layers_min=ho_gnn_gin_conv_mlp_layers_min,
            ho_gnn_gin_conv_mlp_layers_max=ho_gnn_gin_conv_mlp_layers_max,
            ho_gnn_gin_conv_mlp_layers_default=ho_gnn_gin_conv_mlp_layers_default,
            ho_rf_section=ho_rf_section,
            ho_svr_section=ho_svr_section,
            ho_gnn_section=ho_gnn_section,
            ho_gnn_section_2=ho_gnn_section_2,
            ho_gnn_section_3=ho_gnn_section_3,
            ho_gnn_section_4=ho_gnn_section_4,
            ho_gnn_section_gin=ho_gnn_section_gin,
            ho_run_btn=ho_run_btn,
            ho_best_params_df=ho_best_params_df,
            ho_summary=ho_summary,
            ho_best_params_file=ho_best_params_file,
            viz_method=viz_method,
            color_by=color_by,
            model_for_pred=model_for_pred,
            viz_perplexity=viz_perplexity,
            show_top_k=show_top_k,
            top_k_value=top_k_value,
            viz_button=viz_button,
            viz_plot=viz_plot,
            viz_status=viz_status,
            logs_box=logs_box,
            run_dir_text=run_dir_text,
            artifacts_df=artifacts_df,
            dataset_file=dataset_file,
            targets_file=targets_file,
            compounds_file=compounds_file,
            molecule_features_file=molecule_features_file,
            protein_features_file=protein_features_file,
            model_file=model_file,
            model_metrics_file=model_metrics_file,
            model_predictions_file=model_predictions_file,
            all_artifacts_zip=all_artifacts_zip,
            dataset_preview_df=dataset_preview_df,
            targets_preview_df=targets_preview_df,
            compounds_preview_df=compounds_preview_df,
        )

        # IMPORTANT: event wiring must happen inside the Blocks context.
        bind_events(demo, ui)

    return demo, ui


def bind_events(demo: gr.Blocks, ui: UIComponents) -> None:
    ui.split_method_input.change(
        _sync_split_method,
        inputs=[ui.split_method_input],
        outputs=[ui.temporal_controls],
    )

    ui.gnn_arch.change(
        _sync_gnn_arch,
        inputs=[ui.gnn_arch],
        outputs=[ui.gin_options],
    )

    ui.ho_model_type.change(
        _sync_ho_model_type,
        inputs=[ui.ho_model_type, ui.ho_gnn_arch],
        outputs=[
            ui.ho_rf_section,
            ui.ho_svr_section,
            ui.ho_gnn_section,
            ui.ho_gnn_section_2,
            ui.ho_gnn_section_3,
            ui.ho_gnn_section_4,
            ui.ho_gnn_section_gin,
        ],
    )

    ui.ho_optimizer.change(
        _sync_ho_wandb_section,
        inputs=[ui.ho_optimizer, ui.ho_model_type],
        outputs=[ui.ho_wandb_section],
    )

    ui.ho_model_type.change(
        _sync_ho_wandb_section,
        inputs=[ui.ho_optimizer, ui.ho_model_type],
        outputs=[ui.ho_wandb_section],
    )

    ui.ho_gnn_arch.change(
        _sync_ho_gnn_arch,
        inputs=[ui.ho_gnn_arch, ui.ho_model_type],
        outputs=[ui.ho_gnn_section_gin],
    )

    ui.train_rf_btn.click(
        train_model,
        inputs=[
            ui.baseline_model,
            ui.rf_n_estimators,
            ui.svr_kernel,
            ui.svr_c,
            ui.svr_epsilon,
            gr.State("gin"),
            gr.State(10),
            gr.State(64),
            gr.State(128),
            gr.State(128),
            gr.State(5),
            gr.State(0.001),
            gr.State(0.1),
            gr.State("add"),
            gr.State(False),
            gr.State(2),
            gr.State(2),
            gr.State(False),
            gr.State(0.0),
        ],
        outputs=[
            ui.rf_task_text,
            ui.rf_metrics_df,
            ui.rf_preds_preview,
            ui.model_file,
            ui.model_metrics_file,
            ui.model_predictions_file,
            ui.run_dir_text,
            ui.artifacts_df,
            ui.all_artifacts_zip,
        ],
    )

    ui.train_gnn_btn.click(
        train_model,
        inputs=[
            gr.State("GNN (2D)"),
            gr.State(500),
            gr.State("rbf"),
            gr.State(10.0),
            gr.State(0.1),
            ui.gnn_arch,
            ui.gnn_epochs,
            ui.gnn_batch_size,
            ui.gnn_embedding_dim,
            ui.gnn_hidden_dim,
            ui.gnn_num_layers,
            ui.gnn_lr,
            ui.gnn_dropout,
            ui.gin_pooling,
            ui.gnn_residual,
            ui.gnn_head_mlp_layers,
            ui.gin_conv_mlp_layers,
            ui.gin_train_eps,
            ui.gin_eps,
        ],
        outputs=[
            ui.gnn_task_text,
            ui.gnn_metrics_df,
            ui.gnn_preds_preview,
            ui.model_file,
            ui.model_metrics_file,
            ui.model_predictions_file,
            ui.run_dir_text,
            ui.artifacts_df,
            ui.all_artifacts_zip,
        ],
    )

    def update_viz_model_list_and_visualize(method, color_by_val, model_sel, perplexity, show_top_k_val, top_k_val):
        """Update the model dropdown and generate visualization."""
        # First update the model list
        try:
            run_dir = _resolve_current_run_dir()
            models = list_available_models(run_dir)
            choices = []
            
            # Add RF models
            if models["rf"]:
                choices.append("RandomForest")
            
            # Add SVR models
            if models["svr"]:
                choices.append("SVR")
            
            # Add GNN models
            for arch in models["gnn"]:
                choices.append(f"GNN ({arch.upper()})")
            
            # If current model_sel is still valid, keep it; otherwise use first choice
            if model_sel and model_sel in choices:
                updated_model_sel = model_sel
            else:
                updated_model_sel = choices[0] if choices else None
        except Exception:
            choices = []
            updated_model_sel = model_sel  # Keep current selection if update fails
        
        # Generate visualization with the (possibly updated) model selector
        # Use the updated_model_sel if color_by requires it, otherwise use original
        if color_by_val == "Model Predictions":
            model_to_use = updated_model_sel if updated_model_sel else model_sel
        else:
            model_to_use = model_sel
        
        fig, status = visualize_embeddings(
            method, 
            color_by_val, 
            model_to_use, 
            perplexity,
            show_top_k_val,
            top_k_val
        )
        
        return (
            gr.update(choices=choices, value=updated_model_sel),
            fig,
            status
        )
    
    # Update model list and generate visualization in one call
    ui.viz_button.click(
        update_viz_model_list_and_visualize,
        inputs=[
            ui.viz_method,
            ui.color_by,
            ui.model_for_pred,
            ui.viz_perplexity,
            ui.show_top_k,
            ui.top_k_value,
        ],
        outputs=[
            ui.model_for_pred,
            ui.viz_plot,
            ui.viz_status
        ]
    )
    
    ui.extract_embeddings_btn.click(
        extract_gin_embeddings,
        inputs=[ui.gnn_embed_batch_size],
        outputs=[
            ui.gnn_embeddings_preview,
            ui.gnn_embeddings_path,
            ui.run_dir_text,
            ui.artifacts_df,
            ui.gnn_embeddings_file,
            ui.all_artifacts_zip,
        ],
    )

    ui.ho_run_btn.click(
        run_hyperopt,
        inputs=[
            ui.ho_model_type,
            ui.ho_optimizer,
            ui.ho_wandb_project,
            ui.ho_wandb_entity,
            ui.ho_wandb_api_key,
            ui.ho_wandb_sweep_name,
            ui.ho_n_trials,
            ui.ho_n_jobs,
            ui.ho_rf_opt_n_est,
            ui.ho_rf_n_est_min,
            ui.ho_rf_n_est_max,
            ui.ho_rf_opt_depth,
            ui.ho_rf_depth_min,
            ui.ho_rf_depth_max,
            ui.ho_rf_opt_min_samp,
            ui.ho_rf_min_samp_min,
            ui.ho_rf_min_samp_max,
            ui.ho_svr_opt_c,
            ui.ho_svr_c_min,
            ui.ho_svr_c_max,
            ui.ho_svr_opt_epsilon,
            ui.ho_svr_epsilon_min,
            ui.ho_svr_epsilon_max,
            ui.ho_svr_opt_kernel,
            ui.ho_svr_kernel_choices,
            ui.ho_svr_kernel_default,
            ui.ho_gnn_arch,
            ui.ho_gnn_opt_epochs,
            ui.ho_gnn_epochs_min,
            ui.ho_gnn_epochs_max,
            ui.ho_gnn_opt_lr,
            ui.ho_gnn_lr_min,
            ui.ho_gnn_lr_max,
            ui.ho_gnn_opt_batch,
            ui.ho_gnn_batch_min,
            ui.ho_gnn_batch_max,
            ui.ho_gnn_opt_emb,
            ui.ho_gnn_emb_min,
            ui.ho_gnn_emb_max,
            ui.ho_gnn_opt_hidden,
            ui.ho_gnn_hidden_min,
            ui.ho_gnn_hidden_max,
            ui.ho_gnn_opt_dropout,
            ui.ho_gnn_dropout_min,
            ui.ho_gnn_dropout_max,
            ui.ho_gnn_dropout_default,
            ui.ho_gnn_opt_pooling,
            ui.ho_gnn_pooling_choices,
            ui.ho_gnn_pooling_default,
            ui.ho_gnn_opt_residual,
            ui.ho_gnn_residual_default,
            ui.ho_gnn_opt_head_mlp_layers,
            ui.ho_gnn_head_mlp_layers_min,
            ui.ho_gnn_head_mlp_layers_max,
            ui.ho_gnn_head_mlp_layers_default,
            ui.ho_gnn_opt_gin_conv_mlp_layers,
            ui.ho_gnn_gin_conv_mlp_layers_min,
            ui.ho_gnn_gin_conv_mlp_layers_max,
            ui.ho_gnn_gin_conv_mlp_layers_default,
        ],
        outputs=[
            ui.ho_best_params_df,
            ui.ho_summary,
            ui.ho_best_params_file,
            ui.run_dir_text,
            ui.artifacts_df,
        ],
    )

    # Update model selector when model type changes
    def update_model_selector(model_type: str):
        choices, default = _update_model_choices(model_type)
        return gr.update(choices=choices, value=default)
    
    ui.pred_model_type.change(
        update_model_selector,
        inputs=[ui.pred_model_type],
        outputs=[ui.pred_model_selector],
    )
    
    ui.pred_run_btn.click(
        run_prediction,
        inputs=[
            ui.pred_model_type,
            ui.pred_model_selector,
            ui.pred_smiles_text,
            ui.pred_csv_file,
            ui.pred_batch_size,
        ],
        outputs=[
            ui.pred_results_df,
            ui.pred_summary,
            ui.pred_download,
        ],
    )

    ui.build_btn.click(
        build_dataset,
        inputs=[
            gr.State("dta"),
            ui.target_id_type_input,
            ui.targets_input,
            ui.source_input,
            ui.sqlite_path_input,
            ui.split_method_input,
            ui.test_size_input,
            ui.val_size_input,
            ui.split_year_input,
        ],
        outputs=[
            ui.output_df,
            ui.output_summary,
            ui.download_btn,
            ui.plot_act,
            ui.plot_split,
            ui.dataset_state,
            ui.logs_box,
            ui.run_dir_text,
            ui.artifacts_df,
            ui.dataset_file,
            ui.dataset_preview_df,
            ui.targets_file,
            ui.targets_preview_df,
            ui.compounds_file,
            ui.compounds_preview_df,
            ui.molecule_features_file,
            ui.protein_features_file,
            ui.model_file,
            ui.model_metrics_file,
            ui.model_predictions_file,
            ui.all_artifacts_zip,
        ],
    )


def main() -> gr.Blocks:
    demo, _ui = build_ui()
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """Launch the Gradio web UI.

    Args:
        host: Host to bind to. Use "0.0.0.0" for Docker/external access.
        port: Port to run the server on.
        share: Whether to create a public Gradio link.
    """
    demo = main()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        quiet=True,
        theme=APP_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    launch()
