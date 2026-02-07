"""Tests for CLI commands."""

from typer.testing import CliRunner

from dta_gnn.cli import app


runner = CliRunner()


class TestCli:
    """Tests for the CLI application."""

    def test_help(self):
        """Test that help command works."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "DTA-GNN" in result.stdout

    def test_audit_help(self):
        """Test audit subcommand help."""
        result = runner.invoke(app, ["audit", "--help"])

        assert result.exit_code == 0

    def test_ui_help(self):
        """Test ui subcommand help."""
        result = runner.invoke(app, ["ui", "--help"])

        assert result.exit_code == 0
