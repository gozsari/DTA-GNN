from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path


@dataclass(frozen=True)
class RunDirResult:
    run_dir: Path
    current_link: Path


def resolve_run_dir(run_dir: str | Path | None) -> Path | None:
    """Resolve and normalize a run directory path.

    Args:
        run_dir: Path to run directory (string or Path object)

    Returns:
        Resolved Path object, or None if input is None or resolution fails
    """
    if run_dir is None:
        return None
    try:
        return Path(run_dir).expanduser().resolve()
    except Exception:
        return None


def resolve_current_run_dir(*, hint: str = "Build a dataset first.") -> Path:
    """Resolve the current run folder.

    Prefers `runs/current` if it exists (dir or symlink). If missing, raises
    FileNotFoundError.

    Args:
        hint: Optional hint message to include in error

    Returns:
        Resolved Path to current run directory

    Raises:
        FileNotFoundError: If runs/current does not exist
    """
    run_dir = Path("runs") / "current"
    if run_dir.exists():
        try:
            return run_dir.resolve()
        except Exception:
            return run_dir
    raise FileNotFoundError(f"No current run found. Looked for 'runs/current'. {hint}")


def create_run_dir(*, runs_root: str | Path = "runs") -> Path:
    """Create a new timestamped run directory and update `runs/current`.

    Returns the created run directory path.
    """

    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / ts

    # Ensure uniqueness if called multiple times within the same second.
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = runs_root / f"{ts}_{suffix}"

    run_dir.mkdir(parents=True, exist_ok=False)

    current = runs_root / "current"
    try:
        if current.is_symlink() or current.exists():
            if current.is_dir() and not current.is_symlink():
                # Avoid deleting a real directory with contents unexpectedly.
                # If it's a directory, we keep it and just return the run_dir.
                return run_dir
            current.unlink()

        # Prefer a relative symlink (matches existing repo structure).
        os.symlink(run_dir.name, str(current))
    except OSError:
        # Symlinks can fail on some platforms; fall back to writing a pointer.
        # The UI also resolves runs/current; if it's a file, it won't work.
        # So in that case, create a directory and store a marker.
        current.mkdir(parents=True, exist_ok=True)
        (current / "RUN_DIR.txt").write_text(str(run_dir.resolve()), encoding="utf-8")

    return run_dir
