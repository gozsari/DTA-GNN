from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


_UNIPROT_RE = re.compile(
    r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
)
_CHEMBL_TARGET_RE = re.compile(r"^CHEMBL[0-9]+$")


@dataclass(frozen=True)
class UniProtToChEMBLResult:
    resolved_target_chembl_ids: list[str]
    per_input: Mapping[str, list[str]]
    unmapped: list[str]


def parse_uniprot_accessions(text: str) -> list[str]:
    if not (text or "").strip():
        raise ValueError("No UniProt accessions provided")

    raw = re.split(r"[\s,;]+", text.strip())
    accessions = [t.upper() for t in raw if t]

    bad = [a for a in accessions if not _UNIPROT_RE.match(a)]
    if bad:
        raise ValueError(f"Invalid UniProt accession(s): {', '.join(bad)}")

    # preserve order, unique
    seen: set[str] = set()
    out: list[str] = []
    for a in accessions:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def parse_chembl_target_ids(text: str) -> list[str]:
    if not (text or "").strip():
        raise ValueError("No ChEMBL target IDs provided")

    raw = re.split(r"[\s,;]+", text.strip())
    ids = [t.upper() for t in raw if t]

    bad = [t for t in ids if not _CHEMBL_TARGET_RE.match(t)]
    if bad:
        raise ValueError(f"Invalid ChEMBL target ID(s): {', '.join(bad)}")

    seen: set[str] = set()
    out: list[str] = []
    for t in ids:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def map_uniprot_to_chembl_targets_sqlite(
    sqlite_path: str | Path,
    accessions: Iterable[str],
) -> UniProtToChEMBLResult:
    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite DB not found: {path}")

    input_accessions = [a.upper() for a in accessions]
    per_input: dict[str, list[str]] = {a: [] for a in input_accessions}

    if not input_accessions:
        return UniProtToChEMBLResult(
            resolved_target_chembl_ids=[], per_input=per_input, unmapped=[]
        )

    placeholders = ",".join(["?"] * len(input_accessions))
    query = f"""
        SELECT cs.accession, td.chembl_id
        FROM component_sequences cs
        JOIN target_components tc ON tc.component_id = cs.component_id
        JOIN target_dictionary td ON td.tid = tc.tid
        WHERE cs.accession IN ({placeholders})
    """

    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute(query, input_accessions).fetchall()

    for accession, chembl_id in rows:
        if accession is None or chembl_id is None:
            continue
        a = str(accession).upper()
        t = str(chembl_id).upper()
        if a in per_input and t not in per_input[a]:
            per_input[a].append(t)

    resolved = sorted({tid for tids in per_input.values() for tid in tids})
    unmapped = [a for a, tids in per_input.items() if not tids]

    return UniProtToChEMBLResult(
        resolved_target_chembl_ids=resolved,
        per_input=per_input,
        unmapped=unmapped,
    )


def map_uniprot_to_chembl_targets_web(
    accessions: Iterable[str],
) -> UniProtToChEMBLResult:
    """Web-based UniProt→ChEMBL mapping.

    This is implemented as a thin fallback to the existing ChEMBL web client
    logic in the app. It keeps the API stable for the UI.
    """

    from dta_gnn.io.web_source import ChemblWebSource

    input_accessions = [a.upper() for a in accessions]
    per_input: dict[str, list[str]] = {a: [] for a in input_accessions}

    src = ChemblWebSource()
    for a in input_accessions:
        try:
            # Best-effort: match targets whose component accession matches.
            targets = src.get_targets(accession=a)
            per_input[a] = sorted(
                {t["target_chembl_id"] for t in targets if "target_chembl_id" in t}
            )
        except Exception:
            per_input[a] = []

    resolved = sorted({tid for tids in per_input.values() for tid in tids})
    unmapped = [a for a, tids in per_input.items() if not tids]

    return UniProtToChEMBLResult(
        resolved_target_chembl_ids=resolved,
        per_input=per_input,
        unmapped=unmapped,
    )
