"""File I/O utilities for CSV preview, path normalization, and database discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CsvPreview:
    """Result of CSV preview operation."""

    df: pd.DataFrame | None
    error: str | None = None


def normalize_csv_path(path: str | None) -> str | None:
    """Normalize a CSV file path string.

    Args:
        path: Path string to normalize

    Returns:
        Normalized path string, or None if input is empty/None
    """
    if not path:
        return None
    p = str(path).strip()
    return p or None


def preview_csv_with_error(path: str | None, n: int = 50) -> CsvPreview:
    """Preview a CSV file with error handling.

    Args:
        path: Path to CSV file
        n: Number of rows to read (default: 50)

    Returns:
        CsvPreview object with DataFrame and optional error message
    """
    path = normalize_csv_path(path)
    if not path:
        return CsvPreview(df=None, error=None)
    try:
        return CsvPreview(df=pd.read_csv(path, nrows=n), error=None)
    except Exception as e:
        return CsvPreview(df=None, error=f"Could not read CSV: {e}")


def preview_csv(path: str | None, n: int = 50) -> pd.DataFrame | None:
    """Preview a CSV file (wrapper that returns only DataFrame).

    Args:
        path: Path to CSV file
        n: Number of rows to read (default: 50)

    Returns:
        DataFrame with first n rows, or None if error
    """
    return preview_csv_with_error(path, n=n).df


def iter_existing_files(paths: Iterable[str | None]) -> list[str]:
    """Filter a list of paths to only those that exist.

    Args:
        paths: Iterable of file paths (may include None)

    Returns:
        List of existing file paths
    """
    existing: list[str] = []
    for p in paths:
        if not p:
            continue
        try:
            if Path(p).exists():
                existing.append(p)
        except Exception:
            continue
    return existing


def find_chembl_sqlite_dbs() -> list[str]:
    """Find available ChEMBL SQLite DB files under a `chembl_dbs/` folder.

    Searches both the current working directory and the repo root (when running
    from source). Returns absolute paths.

    Returns:
        Sorted list of absolute paths to SQLite database files
    """
    candidates: list[Path] = []
    cwd_dir = Path.cwd() / "chembl_dbs"
    if cwd_dir.exists() and cwd_dir.is_dir():
        candidates.append(cwd_dir)

    # When running from source, try to find repo root
    # This function may be called from various locations, so we try multiple approaches
    try:
        # Try to find repo root by looking for common markers
        current_file = Path(__file__).resolve()
        # io/utils.py is at src/dta_gnn/io/utils.py, so repo root is 3 levels up
        repo_dir = current_file.parents[3]
        repo_candidate = repo_dir / "chembl_dbs"
        if repo_candidate.exists() and repo_candidate.is_dir():
            candidates.append(repo_candidate)
    except Exception:
        pass

    exts = {".db", ".sqlite", ".sqlite3"}
    found: list[Path] = []
    for base in candidates:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                found.append(p.resolve())

    # Stable ordering for a nicer UX
    return sorted({str(p) for p in found})
