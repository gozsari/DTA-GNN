from __future__ import annotations

import pandas as pd
from loguru import logger


def calculate_morgan_fingerprints(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    radius: int = 2,
    n_bits: int = 2048,
    *,
    out_col: str = "morgan_fingerprint",
    drop_failures: bool = True,
) -> pd.DataFrame:
    """Calculate Morgan fingerprints for molecules in the DataFrame.

    Returns a copy of `df` with an added fingerprint column containing bitstrings.

    Args:
        df: Input dataframe
        smiles_col: Column containing SMILES strings
        radius: Morgan radius (2 => ECFP4)
        n_bits: Fingerprint bit length
        out_col: Output column name
        drop_failures: Whether to drop rows that fail featurization
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    logger.info(f"Calculating Morgan fingerprints (r={radius}, n={n_bits})...")

    fps: list[str | None] = []
    indices_to_drop: list[int] = []

    # Prefer the new generator API if available.
    mfgen = None
    try:
        mfgen = AllChem.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    except Exception:
        mfgen = None

    for idx, row in df.iterrows():
        smi = row.get(smiles_col)
        if smi is None or pd.isna(smi) or not str(smi).strip():
            fps.append(None)
            indices_to_drop.append(int(idx))
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            fps.append(None)
            indices_to_drop.append(int(idx))
            continue

        try:
            if mfgen is not None:
                fp = mfgen.GetFingerprint(mol)
                fps.append(fp.ToBitString())
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, int(radius), nBits=int(n_bits)
                )
                fps.append(fp.ToBitString())
        except Exception:
            fps.append(None)
            indices_to_drop.append(int(idx))

    out = df.copy()
    out[out_col] = fps

    if drop_failures and indices_to_drop:
        logger.warning(f"Failed to featurize {len(indices_to_drop)} molecules.")
        out = out.drop(indices_to_drop)

    return out


__all__ = [
    "calculate_morgan_fingerprints",
]
