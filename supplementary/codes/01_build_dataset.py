#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_dataset.py
====================
Build Drug-Target Affinity datasets (random + scaffold splits) from ChEMBL,
run scaffold leakage audit, and save all dataset artifacts.

Usage
-----
    python 01_build_dataset.py --uniprot_id P00533
    python 01_build_dataset.py --uniprot_id P14416 --output_dir runs --seed 42

Outputs
-------
    <output_dir>/<UID>_random/   dataset.csv, compounds.csv, metadata.json
    <output_dir>/<UID>_scaffold/ dataset.csv, compounds.csv, metadata.json
        paper_artifacts/         table_leakage_ratio.csv/.tex, fig_leakage_ratio.pdf
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dta_gnn.pipeline import Pipeline
from dta_gnn.io.target_mapping import (
    map_uniprot_to_chembl_targets_web,
    parse_uniprot_accessions,
)
from dta_gnn.audits.leakage import audit_scaffold_leakage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build DTA datasets from ChEMBL (parametric on UniProt ID)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--uniprot_id", type=str, required=True,
                    help="UniProt accession (e.g. P00533, P14416).")
    p.add_argument("--standard_types", nargs="+", default=["IC50", "Ki", "Kd"],
                    help="ChEMBL bioactivity types to include.")
    p.add_argument("--output_dir", type=str, default="runs",
                    help="Root directory for all run outputs.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.1)
    return p.parse_args()


def build_datasets(
    uniprot_id: str,
    standard_types: list[str],
    run_dir_random: Path,
    run_dir_scaffold: Path,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch bioactivity data and build two datasets (random + scaffold split)."""
    accessions = parse_uniprot_accessions(uniprot_id)
    mapping_result = map_uniprot_to_chembl_targets_web(accessions)
    target_ids = mapping_result.resolved_target_chembl_ids

    if not target_ids:
        raise ValueError(
            f"Could not map UniProt {uniprot_id} to any ChEMBL targets. "
            "Check network access or the UniProt ID."
        )
    print(f"Mapped {uniprot_id} -> ChEMBL targets: {target_ids}")

    shared_kwargs = dict(
        target_ids=target_ids,
        standard_types=standard_types,
        test_size=test_size,
        val_size=val_size,
        featurize=False,
    )

    # -- Random split --
    pipeline_random = Pipeline(source_type="web")
    df_random = pipeline_random.build_dta(split_method="random", **shared_kwargs)
    if df_random.empty:
        raise ValueError("Dataset is empty after building with random split.")

    df_random.to_csv(run_dir_random / "dataset.csv", index=False)
    compounds_random = df_random[["molecule_chembl_id", "smiles"]].drop_duplicates()
    compounds_random.to_csv(run_dir_random / "compounds.csv", index=False)
    (run_dir_random / "metadata.json").write_text(
        json.dumps(
            {
                "uniprot_id": uniprot_id,
                "target_chembl_ids": target_ids,
                "split_method": "random",
                "n_samples": len(df_random),
                "n_compounds": len(compounds_random),
            },
            indent=2,
        )
    )
    print(f"\nRandom split: {len(df_random)} samples")
    print(df_random["split"].value_counts().to_string())

    # -- Scaffold split --
    pipeline_scaffold = Pipeline(source_type="web")
    df_scaffold = pipeline_scaffold.build_dta(split_method="scaffold", **shared_kwargs)
    if df_scaffold.empty:
        raise ValueError("Dataset is empty after building with scaffold split.")

    df_scaffold.to_csv(run_dir_scaffold / "dataset.csv", index=False)
    compounds_scaffold = df_scaffold[["molecule_chembl_id", "smiles"]].drop_duplicates()
    compounds_scaffold.to_csv(run_dir_scaffold / "compounds.csv", index=False)
    (run_dir_scaffold / "metadata.json").write_text(
        json.dumps(
            {
                "uniprot_id": uniprot_id,
                "target_chembl_ids": target_ids,
                "split_method": "scaffold",
                "n_samples": len(df_scaffold),
                "n_compounds": len(compounds_scaffold),
            },
            indent=2,
        )
    )
    print(f"\nScaffold split: {len(df_scaffold)} samples")
    print(df_scaffold["split"].value_counts().to_string())

    return df_random, df_scaffold


def run_leakage_audit(
    df_random: pd.DataFrame,
    df_scaffold: pd.DataFrame,
    run_dir_scaffold: Path,
    paper_artifacts: Path,
) -> None:
    """Compute scaffold leakage ratios and save bar chart + CSV/TeX tables."""
    train_r = df_random[df_random["split"] == "train"]
    test_r = df_random[df_random["split"] == "test"]
    leak_random = audit_scaffold_leakage(train_r, test_r, smiles_col="smiles")

    train_s = df_scaffold[df_scaffold["split"] == "train"]
    test_s = df_scaffold[df_scaffold["split"] == "test"]
    leak_scaffold = audit_scaffold_leakage(train_s, test_s, smiles_col="smiles")

    print("Leakage ratio (Random):  ", leak_random["leakage_ratio"])
    print("Leakage ratio (Scaffold):", leak_scaffold["leakage_ratio"])

    leak_df = pd.DataFrame(
        {
            "Split": ["Random", "Scaffold"],
            "Leakage ratio": [
                leak_random["leakage_ratio"],
                leak_scaffold["leakage_ratio"],
            ],
        }
    )
    leak_df.to_csv(paper_artifacts / "table_leakage_ratio.csv", index=False)
    with open(paper_artifacts / "table_leakage_ratio.tex", "w") as fh:
        fh.write(leak_df.to_latex(index=False, float_format="%.4f", na_rep="--"))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Random", "Scaffold"],
        [leak_random["leakage_ratio"], leak_scaffold["leakage_ratio"]],
        color=["#1f77b4", "#ff7f0e"],
    )
    ax.set_ylabel("Leakage ratio")
    ax.set_title("Scaffold leakage: test scaffolds also in train")
    plt.tight_layout()
    plt.savefig(run_dir_scaffold / "leakage_ratio.png", dpi=120)
    plt.savefig(paper_artifacts / "fig_leakage_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Leakage plot saved.")


def main() -> None:
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    uid = args.uniprot_id.upper()
    run_dir_random = Path(args.output_dir) / f"{uid}_random"
    run_dir_scaffold = Path(args.output_dir) / f"{uid}_scaffold"
    paper_artifacts = run_dir_scaffold / "paper_artifacts"

    for d in (run_dir_random, run_dir_scaffold, paper_artifacts):
        d.mkdir(parents=True, exist_ok=True)

    print(f"UniProt ID:         {uid}")
    print(f"Run dir (random):   {run_dir_random}")
    print(f"Run dir (scaffold): {run_dir_scaffold}")
    print(f"Paper artifacts:    {paper_artifacts}")

    df_random, df_scaffold = build_datasets(
        uniprot_id=uid,
        standard_types=args.standard_types,
        run_dir_random=run_dir_random,
        run_dir_scaffold=run_dir_scaffold,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    run_leakage_audit(df_random, df_scaffold, run_dir_scaffold, paper_artifacts)

    print(f"\nDataset build complete for {uid}.")
    print(f"  Scaffold run dir: {run_dir_scaffold}")
    print(f"  Random run dir:   {run_dir_random}")


if __name__ == "__main__":
    main()
