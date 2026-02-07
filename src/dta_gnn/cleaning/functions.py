import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Optional, Literal


def standardize_activities(
    df: pd.DataFrame, convert_to_pchembl: bool = True, drop_censored: bool = False
) -> pd.DataFrame:
    """
    Standardize activity values.
    - Filters rows with missing standard_value.
    - Converts nanomolar units to molar if needed (though ChEMBL standard_value is usually nM).
    - Calculates pChEMBL if missing and requested.
    """
    df = df.copy()

    # Ensure numeric columns are actually numeric
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    if "pchembl_value" in df.columns:
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")

    # Drop missing values
    df = df.dropna(subset=["standard_value", "standard_units"])

    # Handle censored values (>, <)
    # If standard_relation exists, we can use it.
    if "standard_relation" in df.columns:
        if drop_censored:
            # Keep only exact measurements
            df = df[
                df["standard_relation"].isin(["=", ""])
            ]  # Some might be empty string?
        else:
            pass  # Keep them for now, might be used for binary labeling depending on direction

    # ChEMBL standard_value is typically in nM.
    # pChEMBL = -log10(value_molar)
    # value_molar = value_nM * 1e-9

    if convert_to_pchembl and "pchembl_value" in df.columns:
        # If pchembl_value column exists but has NaNs, try to fill them
        mask = df["pchembl_value"].isna()
        # Only fill if units are nM (most common)
        nm_mask = df["standard_units"] == "nM"

        # Calculate for those missing pchembl but having nM values
        to_calc = mask & nm_mask
        if to_calc.any():
            # Avoid log(0)
            valid_val = df.loc[to_calc, "standard_value"] > 0
            df.loc[to_calc & valid_val, "pchembl_value"] = -np.log10(
                df.loc[to_calc & valid_val, "standard_value"] * 1e-9
            )

    return df


def aggregate_duplicates(
    df: pd.DataFrame,
    group_cols: list = ["molecule_chembl_id", "target_chembl_id"],
    agg_method: Literal["median", "mean", "max", "min"] = "median",
) -> pd.DataFrame:
    """
    Deduplicate measurements for the same drug-target pair.
    """
    if df.empty:
        return df

    # We aggregate pchembl_value if available, else standard_value
    target_col = "pchembl_value" if "pchembl_value" in df.columns else "standard_value"

    # Drop rows where the target column is NaN before aggregating
    df_clean = df.dropna(subset=[target_col])

    # Groupby and aggregate
    grouped = df_clean.groupby(group_cols)[target_col].agg(agg_method).reset_index()

    # If 'year' is present, we need to preserve it.
    # Since year is per-activity, and we are aggregating activities for the same Mol-Target pair,
    # we should take the MIN key (earliest year)? Or MAX?
    # In 'cold drug', we just need 'a' year. For temporal split, using the earliest year
    # ensures we don't leak future info (if a drug was published in 2018 and 2022, treating it as 2018 is safe for train).
    if "year" in df.columns:
        # We need to aggregate year as well.
        # Let's perform aggregation on multiple columns.
        agg_dict = {target_col: agg_method, "year": "min"}
        grouped = df_clean.groupby(group_cols).agg(agg_dict).reset_index()

    return grouped


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a single SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        pass
    return None
