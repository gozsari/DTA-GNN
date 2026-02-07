import typer
from pathlib import Path

app = typer.Typer(help="DTA-GNN CLI")


@app.command()
def audit(
    file: Path = typer.Argument(..., help="Path to dataset CSV"),
):
    """
    Run audit on an existing dataset file.
    """
    # todo: load and audit
    typer.echo("Audit completed (Mock).")


@app.command()
def setup(
    version: str = typer.Option("36", help="ChEMBL version to download"),
    dir: Path = typer.Option(".", help="Directory to download to"),
):
    """
    Download and set up the ChEMBL SQLite database.
    """
    from dta_gnn.io.downloader import download_chembl_db

    typer.echo(f"Downloading ChEMBL {version} to {dir}...")
    try:
        db_path = download_chembl_db(version, str(dir))
        typer.echo(f"Successfully set up database at: {db_path}")
        typer.echo("You can now use this path with --db-path or in the UI.")
    except Exception as e:
        typer.echo(f"Setup failed: {e}", err=True)


@app.command()
def ui(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="Host to bind to. Use 0.0.0.0 for Docker."
    ),
    port: int = typer.Option(7860, "--port", "-p", help="Port to run the server on."),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link."),
):
    """
    Launch the Gradio UI.
    """
    from dta_gnn.app.ui import launch

    launch(host=host, port=port, share=share)


if __name__ == "__main__":
    app()
