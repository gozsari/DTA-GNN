"""Tests for run directory management."""

from pathlib import Path

from dta_gnn.io.runs import create_run_dir


class TestCreateRunDir:
    """Tests for create_run_dir function."""

    def test_creates_timestamped_directory(self, tmp_path):
        """Test that a timestamped directory is created."""
        runs_root = tmp_path / "runs"

        run_dir = create_run_dir(runs_root=runs_root)

        assert run_dir.exists()
        assert run_dir.is_dir()
        # Directory name should be a timestamp
        assert len(run_dir.name) >= 15  # YYYYMMDD_HHMMSS format

    def test_creates_runs_root_if_missing(self, tmp_path):
        """Test that runs root is created if it doesn't exist."""
        runs_root = tmp_path / "nonexistent" / "runs"

        run_dir = create_run_dir(runs_root=runs_root)

        assert runs_root.exists()
        assert run_dir.exists()

    def test_creates_current_symlink(self, tmp_path):
        """Test that 'current' symlink is created."""
        runs_root = tmp_path / "runs"

        run_dir = create_run_dir(runs_root=runs_root)
        current = runs_root / "current"

        # On platforms that support symlinks
        if current.is_symlink():
            assert current.resolve() == run_dir.resolve()
        elif current.is_dir():
            # Fallback behavior
            assert current.exists()

    def test_updates_current_on_subsequent_calls(self, tmp_path):
        """Test that 'current' is updated on each call."""
        runs_root = tmp_path / "runs"

        run_dir_1 = create_run_dir(runs_root=runs_root)
        # Small delay to ensure different timestamp
        import time

        time.sleep(0.1)
        run_dir_2 = create_run_dir(runs_root=runs_root)

        assert run_dir_1 != run_dir_2

        current = runs_root / "current"
        if current.is_symlink():
            # Current should point to the latest run
            assert current.resolve() == run_dir_2.resolve()

    def test_handles_duplicate_timestamps(self, tmp_path):
        """Test handling of duplicate timestamps (same second)."""
        runs_root = tmp_path / "runs"

        # Create multiple runs quickly
        run_dirs = []
        for _ in range(3):
            run_dirs.append(create_run_dir(runs_root=runs_root))

        # All should be unique
        assert len(set(run_dirs)) == 3
        for rd in run_dirs:
            assert rd.exists()

    def test_returns_path_object(self, tmp_path):
        """Test that the return value is a Path object."""
        runs_root = tmp_path / "runs"

        run_dir = create_run_dir(runs_root=runs_root)

        assert isinstance(run_dir, Path)

    def test_default_runs_root(self, monkeypatch, tmp_path):
        """Test with default runs_root parameter."""
        # Change to temp directory to avoid creating in project root
        monkeypatch.chdir(tmp_path)

        run_dir = create_run_dir()

        assert run_dir.exists()
        assert "runs" in str(run_dir.parent)
