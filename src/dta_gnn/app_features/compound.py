from __future__ import annotations

import re

import pandas as pd

from dta_gnn.features import calculate_morgan_fingerprints


def _parse_smiles_text(smiles_text: str) -> list[str]:
    tokens = re.split(r"[,\n\s]+", smiles_text or "")
    smiles_list = [t.strip() for t in tokens if t and t.strip()]
    return smiles_list


def build_smiles_frame(
    *,
    smiles_text: str,
    df_state: pd.DataFrame | None,
    source_mode: str,
) -> pd.DataFrame:
    """Build a minimal SMILES DataFrame for UI featurization.

    The UI supports two rough modes:
    - custom text input (one SMILES per line / comma-separated)
    - using the current dataset state (expects a smiles-like column)
    """

    mode = (source_mode or "").strip().lower()
    want_dataset = ("dataset" in mode) or ("state" in mode)

    if (not want_dataset) and smiles_text:
        smiles_list = _parse_smiles_text(smiles_text)
        if not smiles_list:
            raise ValueError("No SMILES found in the provided text.")
        return pd.DataFrame(
            {
                "molecule_chembl_id": [
                    f"custom_{i+1}" for i in range(len(smiles_list))
                ],
                "smiles": smiles_list,
            }
        )

    if df_state is None or df_state.empty:
        if smiles_text:
            smiles_list = _parse_smiles_text(smiles_text)
            if not smiles_list:
                raise ValueError("No SMILES found in the provided text.")
            return pd.DataFrame(
                {
                    "molecule_chembl_id": [
                        f"custom_{i+1}" for i in range(len(smiles_list))
                    ],
                    "smiles": smiles_list,
                }
            )
        raise ValueError("No dataset state available and no SMILES text provided.")

    df = df_state.copy()
    if "smiles" in df.columns:
        smiles_col = "smiles"
    elif "canonical_smiles" in df.columns:
        smiles_col = "canonical_smiles"
    else:
        raise ValueError(
            "Dataset state is missing a 'smiles' (or 'canonical_smiles') column."
        )

    id_col = None
    for candidate in ("molecule_chembl_id", "compound_chembl_id"):
        if candidate in df.columns:
            id_col = candidate
            break

    out = pd.DataFrame({"smiles": df[smiles_col].astype(str)})
    if id_col:
        out["molecule_chembl_id"] = df[id_col].astype(str)
    else:
        out["molecule_chembl_id"] = [f"row_{i+1}" for i in range(len(out))]

    out = out.replace({"smiles": {"nan": None, "None": None}})
    out = out.dropna(subset=["smiles"])
    out = out.drop_duplicates(subset=["molecule_chembl_id", "smiles"])
    return out


def featurize_smiles_morgan(
    df: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Compute Morgan fingerprints for a SMILES DataFrame.

    Produces a `morgan_fingerprint` column containing bitstrings.
    """

    if smiles_col not in df.columns:
        raise ValueError(f"Missing required column: {smiles_col}")
    df_feat = calculate_morgan_fingerprints(
        df.copy(), smiles_col=smiles_col, radius=radius, n_bits=n_bits
    )

    # Keep deterministic ordering for UI previews.
    if "molecule_chembl_id" in df_feat.columns:
        df_feat = df_feat.sort_values(["molecule_chembl_id"], kind="stable")
    return df_feat
