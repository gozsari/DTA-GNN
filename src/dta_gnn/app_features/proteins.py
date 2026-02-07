from __future__ import annotations

import re

import pandas as pd


def _parse_sequences_text(seqs_text: str) -> list[str]:
    # Allow newline/comma/whitespace separated sequences.
    tokens = re.split(r"[,\n\s]+", seqs_text or "")
    seqs = [t.strip().upper() for t in tokens if t and t.strip()]
    return seqs


def build_sequence_frame(
    *,
    seqs_text: str,
    df_state: pd.DataFrame | None,
    source_mode: str,
) -> pd.DataFrame:
    """Build a minimal protein-sequence DataFrame for UI featurization."""

    mode = (source_mode or "").strip().lower()
    want_dataset = ("dataset" in mode) or ("state" in mode)

    if (not want_dataset) and seqs_text:
        seqs = _parse_sequences_text(seqs_text)
        if not seqs:
            raise ValueError("No sequences found in the provided text.")
        return pd.DataFrame(
            {
                "target_chembl_id": [f"custom_{i+1}" for i in range(len(seqs))],
                "sequence": seqs,
            }
        )

    if df_state is None or df_state.empty:
        if seqs_text:
            seqs = _parse_sequences_text(seqs_text)
            if not seqs:
                raise ValueError("No sequences found in the provided text.")
            return pd.DataFrame(
                {
                    "target_chembl_id": [f"custom_{i+1}" for i in range(len(seqs))],
                    "sequence": seqs,
                }
            )
        raise ValueError("No dataset state available and no sequences text provided.")

    df = df_state.copy()
    if "sequence" not in df.columns:
        raise ValueError("Dataset state is missing a 'sequence' column.")

    id_col = "target_chembl_id" if "target_chembl_id" in df.columns else None

    out = pd.DataFrame({"sequence": df["sequence"].astype(str).str.upper()})
    if id_col:
        out["target_chembl_id"] = df[id_col].astype(str)
    else:
        out["target_chembl_id"] = [f"row_{i+1}" for i in range(len(out))]

    out = out.replace({"sequence": {"nan": None, "None": None}})
    out = out.dropna(subset=["sequence"])
    out = out.drop_duplicates(subset=["target_chembl_id", "sequence"])
    return out
