#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_baselines.py
=====================
Train classical baseline models (Random Forest, SVR) on a previously built
DTA dataset and evaluate on validation + test splits.

Usage
-----
    python 02_train_baselines.py --run_dir runs/P35354_scaffold
    python 02_train_baselines.py --run_dir runs/P00533_scaffold --rf_n_estimators 1000 --svr_C 1.0

Outputs
-------
    <run_dir>/model_rf.pkl                 -- trained RF model
    <run_dir>/model_metrics.json           -- RF metrics per split
    <run_dir>/model_predictions.csv        -- RF val+test predictions
    <run_dir>/model_svr.pkl                -- trained SVR model
    <run_dir>/model_metrics_svr.json       -- SVR metrics per split
    <run_dir>/model_predictions_svr.csv    -- SVR val+test predictions
    <run_dir>/baseline_metrics.json        -- consolidated metrics for both models
    <run_dir>/baseline_runtime_profile.json -- wall-clock and memory profiling
"""

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from dta_gnn.models.random_forest import train_random_forest_on_run
from dta_gnn.models.svr import train_svr_on_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train baseline models (RF + SVR) on a DTA run directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_dir", type=str, required=True,
                    help="Path to the scaffold-split run directory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Random Forest
    p.add_argument("--rf_n_estimators", type=int, default=10,
                    help="Number of trees for Random Forest.")
    p.add_argument("--rf_max_depth", type=int, default=5,
                    help="Maximum tree depth for Random Forest (None for unlimited).")

    # SVR
    p.add_argument("--svr_C", type=float, default=0.1,
                    help="Regularisation parameter C for SVR.")
    p.add_argument("--svr_epsilon", type=float, default=0.01,
                    help="Epsilon-tube width for SVR.")
    p.add_argument("--svr_kernel", type=str, default="rbf",
                    choices=["rbf", "linear"],
                    help="SVR kernel function.")
    return p.parse_args()


def _mem_mb() -> float:
    """Current process RSS in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _format_metrics_row(model_name: str, split: str, m: dict) -> dict:
    return {
        "Model": model_name,
        "Split": split,
        "RMSE": m.get("rmse"),
        "MAE": m.get("mae"),
        "R2": m.get("r2"),
        "Pearson": m.get("pearson_r"),
        "Spearman": m.get("spearman_r"),
    }


def main() -> None:
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "dataset.csv").exists():
        raise FileNotFoundError(
            f"No dataset.csv in {run_dir}. Run 01_build_dataset.py first."
        )

    print(f"Run dir: {run_dir}")
    print(f"Seed:    {seed}\n")

    consolidated: dict = {}
    table_rows: list[dict] = []
    profile: dict = {}

    # -- Random Forest --
    print("=" * 60)
    print("Training Random Forest baseline")
    print("=" * 60)
    _mem_before = _mem_mb()
    _t0 = time.perf_counter()

    rf_result = train_random_forest_on_run(
        run_dir,
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        random_seed=seed,
    )

    _t_rf = time.perf_counter() - _t0
    _delta_ram_rf = _mem_mb() - _mem_before
    profile["random_forest"] = {
        "wall_s": round(_t_rf, 2),
        "delta_ram_mb": round(_delta_ram_rf, 1),
    }

    rf_splits = rf_result.metrics.get("splits", {})
    consolidated["RandomForest"] = {}
    for split_name in ("train", "val", "test"):
        m = rf_splits.get(split_name)
        if m is not None:
            consolidated["RandomForest"][split_name] = m
            table_rows.append(_format_metrics_row("RandomForest", split_name, m))

    print(f"  Model saved to {rf_result.model_path}")
    print(f"  Metrics saved to {rf_result.metrics_path}")
    print(f"  Wall-clock: {_t_rf:.1f}s  |  dRAM: {_delta_ram_rf:.1f} MiB\n")

    # -- SVR --
    print("=" * 60)
    print("Training SVR baseline")
    print("=" * 60)
    _mem_before = _mem_mb()
    _t0 = time.perf_counter()

    svr_result = train_svr_on_run(
        run_dir,
        C=args.svr_C,
        epsilon=args.svr_epsilon,
        kernel=args.svr_kernel,
        random_seed=seed,
    )

    _t_svr = time.perf_counter() - _t0
    _delta_ram_svr = _mem_mb() - _mem_before
    profile["svr"] = {
        "wall_s": round(_t_svr, 2),
        "delta_ram_mb": round(_delta_ram_svr, 1),
    }

    svr_splits = svr_result.metrics.get("splits", {})
    consolidated["SVR"] = {}
    for split_name in ("train", "val", "test"):
        m = svr_splits.get(split_name)
        if m is not None:
            consolidated["SVR"][split_name] = m
            table_rows.append(_format_metrics_row("SVR", split_name, m))

    print(f"  Model saved to {svr_result.model_path}")
    print(f"  Metrics saved to {svr_result.metrics_path}")
    print(f"  Wall-clock: {_t_svr:.1f}s  |  dRAM: {_delta_ram_svr:.1f} MiB\n")

    # -- Consolidated output --
    consolidated_path = run_dir / "baseline_metrics.json"
    consolidated_path.write_text(json.dumps(consolidated, indent=2))
    print(f"Consolidated baseline metrics saved to {consolidated_path}\n")

    # -- Runtime profile --
    cpu_info = platform.processor() or platform.machine()
    profile_out = {
        "device": "CPU",
        "cpu": cpu_info,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "phases": profile,
    }
    profile_path = run_dir / "baseline_runtime_profile.json"
    profile_path.write_text(json.dumps(profile_out, indent=2))
    print(f"Runtime profile saved to {profile_path}\n")

    # -- Summary table --
    df_summary = pd.DataFrame(table_rows)
    print("=" * 60)
    print("Baseline Results")
    print("=" * 60)
    print(df_summary.to_string(index=False, float_format="%.4f", na_rep="--"))

    print(f"\n{'=' * 60}")
    print("Runtime / Memory Profile")
    print("=" * 60)
    print(f"Hardware: CPU  |  {cpu_info}")
    print(f"Python:   {platform.python_version()}  |  OS: {platform.system()} {platform.release()}")
    rt_rows = []
    for phase, d in profile.items():
        rt_rows.append({
            "Phase": phase.replace("_", " ").title(),
            "Wall-clock (s)": d["wall_s"],
            "dRAM (MiB)": d["delta_ram_mb"],
        })
    df_rt = pd.DataFrame(rt_rows)
    print(df_rt.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
