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


@app.command("train-gnn")
def train_gnn(
    uniprot_ids: str = typer.Argument(
        ...,
        help=(
            "UniProt accession(s) for the target protein(s), "
            "comma/space/semicolon-separated. E.g. 'P00533' or 'P00533,P04637'."
        ),
    ),
    architecture: str = typer.Option(
        "gin",
        help="GNN architecture: gin|gcn|gat|sage|pna|transformer|tag|arma|cheb|supergat",
    ),
    sqlite_path: str = typer.Option(
        None, "--sqlite-path", help="Path to ChEMBL SQLite DB. Omit to use the web API."
    ),
    standard_types: str = typer.Option(
        None,
        "--standard-types",
        help="Comma-separated activity types to include, e.g. 'IC50,Ki'. Omit for all.",
    ),
    test_size: float = typer.Option(0.2, "--test-size", help="Test split fraction."),
    val_size: float = typer.Option(0.1, "--val-size", help="Validation split fraction."),
    wandb_project: str = typer.Option("dta_gnn", "--wandb-project", help="W&B project name."),
    wandb_entity: str = typer.Option(None, "--wandb-entity", help="W&B entity/team name."),
    wandb_api_key: str = typer.Option(None, "--wandb-api-key", help="W&B API key."),
    n_trials: int = typer.Option(20, "--n-trials", help="Number of HPO sweep trials."),
    epochs: int = typer.Option(30, "--epochs", help="Epochs for final model training."),
    batch_size: int = typer.Option(64, "--batch-size", help="Mini-batch size."),
    device: str = typer.Option(
        None, "--device", help="Device: auto|mps|cuda|cpu. Default: auto-detect."
    ),
    runs_root: str = typer.Option("runs", "--runs-root", help="Root directory for run folders."),
):
    """
    Run the end-to-end GNN training pipeline for a target protein.

    Steps: UniProt→ChEMBL mapping, scaffold-split dataset build, W&B
    hyperparameter search, final model training, test evaluation.
    All steps are timed and results printed on completion.

    Example usage:
    dta-gnn train-gnn P00533 --standard-types IC50,Ki --n-trials 50 --epochs 100 --batch-size 128  
    """
    from dta_gnn.training import EndToEndConfig, run_gnn_end_to_end

    standard_types_list = (
        [s.strip() for s in standard_types.split(",") if s.strip()]
        if standard_types
        else None
    )

    config = EndToEndConfig(
        uniprot_ids=uniprot_ids,
        architecture=architecture,
        sqlite_path=sqlite_path or None,
        standard_types=standard_types_list,
        test_size=test_size,
        val_size=val_size,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity or None,
        wandb_api_key=wandb_api_key or None,
        n_trials=n_trials,
        epochs=epochs,
        batch_size=batch_size,
        device=device or None,
        runs_root=runs_root,
    )

    try:
        result = run_gnn_end_to_end(config)
    except Exception as e:
        typer.echo(f"train-gnn failed: {e}", err=True)
        raise typer.Exit(1)

    # Summary output
    total = sum(result.timings.values())
    tm = result.test_metrics
    typer.echo("")
    typer.echo(f"Run directory : {result.run_dir}")
    typer.echo(
        f"Dataset size  : {result.dataset_size} rows"
        f"  (train={result.train_size},"
        f" val={result.val_size_actual},"
        f" test={result.test_size_actual})"
    )
    typer.echo(f"Best val R\u00b2   : {result.hyperopt_result.best_value:.4f}")
    typer.echo(
        f"Test metrics  :"
        f"  rmse={tm.get('rmse', 'n/a')}"
        f"  mae={tm.get('mae', 'n/a')}"
        f"  r2={tm.get('r2', 'n/a')}"
    )
    typer.echo("")
    typer.echo("Timings")
    for step, t in result.timings.items():
        pct = 100.0 * t / total if total > 0 else 0.0
        typer.echo(f"  {step:<30s}  {t:6.1f}s  ({pct:.0f}%)")
    typer.echo(f"  {'─' * 42}")
    typer.echo(f"  {'Total':<30s}  {total:6.1f}s  ({total / 60:.1f} min)")


if __name__ == "__main__":
    app()
