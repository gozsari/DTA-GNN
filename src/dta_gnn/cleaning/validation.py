"""Validation functions for dataset building parameters."""

from __future__ import annotations

from dta_gnn.io.utils import normalize_csv_path


def validate_sqlite_path(source: str, sqlite_path: str | None) -> None:
    """Validate SQLite database path if source is 'sqlite'.

    Args:
        source: Data source type ('sqlite' or 'web')
        sqlite_path: Path to SQLite database file

    Raises:
        ValueError: If source is 'sqlite' and path is missing or invalid
    """
    if (source or "").strip() != "sqlite":
        return
    p = normalize_csv_path(sqlite_path)
    if not p:
        raise ValueError("SQLite DB path is required when Data Source is 'sqlite'.")
    from pathlib import Path

    if not Path(p).exists():
        raise ValueError(f"SQLite DB not found: {p}")


def validate_split_sizes(test_size: float, val_size: float) -> None:
    """Validate test and validation split sizes.

    Args:
        test_size: Test set size (fraction)
        val_size: Validation set size (fraction)

    Raises:
        ValueError: If sizes are invalid (not numbers, negative, or sum >= 1.0)
    """
    try:
        ts = float(test_size)
        vs = float(val_size)
    except (ValueError, TypeError):
        raise ValueError("Test/Validation sizes must be numbers.")
    if ts < 0 or vs < 0:
        raise ValueError("Test/Validation sizes must be non-negative.")
    if ts + vs >= 1.0:
        raise ValueError("Test size + validation size must be < 1.0.")
