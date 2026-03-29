# Command Line Interface

DTA-GNN provides a command-line interface for setup, UI launch, lightweight auditing, and end-to-end GNN training from a UniProt accession.

## Installation

The CLI is installed automatically with the package:

```bash
pip install dta-gnn
```

Verify installation:

```bash
dta_gnn --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `audit` | Run audit on an existing dataset *(coming soon)* |
| `setup` | Download and set up ChEMBL database |
| `ui` | Launch the Gradio web interface |
| `train-gnn` | Run end-to-end GNN training pipeline from a UniProt ID |

Dataset building is also available via the Python API (`Pipeline.build_dta()`) or the Web UI (`dta_gnn ui`) for more control over individual steps.

## setup

Download and configure the ChEMBL SQLite database.

### Usage

```bash
dta_gnn setup --version 36 --dir ./chembl_dbs
```

### Options

```
Options:
  --version TEXT  ChEMBL version to download  [default: 36]
  --dir PATH      Directory to download to  [default: .]
  --help          Show this message and exit.
```

### Example

```bash
# Download ChEMBL 36 to a dedicated folder
mkdir -p ./chembl_dbs
dta_gnn setup --version 36 --dir ./chembl_dbs

# Output:
# Downloading ChEMBL 36 to ./chembl_dbs...
# Successfully set up database at: ./chembl_dbs/chembl_36.db
# You can now use this path in the UI or with `Pipeline(source_type="sqlite", sqlite_path=...)`.
```

## audit

!!! warning "Coming Soon"
    The `audit` CLI command is a placeholder. For now, use the Python API
    (`audit_scaffold_leakage`, `audit_target_leakage`) or the Web UI instead.

Run leakage audits on an existing dataset file.

### Usage

```bash
dta_gnn audit dataset.csv
```

### Options

```
Usage: dta_gnn audit [OPTIONS] FILE

  Run audit on an existing dataset file.

Arguments:
  FILE  Path to dataset CSV  [required]

Options:
  --help  Show this message and exit.
```

!!! note
    The audit command currently provides basic functionality. For comprehensive audit features including scaffold leakage detection and detailed reporting, use the Python API (`dta_gnn.audits` module).

## ui

Launch the interactive Gradio web interface.

### Usage

```bash
dta_gnn ui
```

This starts a web server at `http://127.0.0.1:7860`.

### Options

```
Options:
  --host TEXT     Host to bind to. Use 0.0.0.0 for Docker.  [default: 127.0.0.1]
  --port INTEGER  Port to run the server on.  [default: 7860]
  --share         Create a public Gradio link.
  --help          Show this message and exit.
```

### Examples

```bash
# Default (localhost only)
dta_gnn ui

# Custom port
dta_gnn ui --port 8080

# Accessible from network (for Docker/remote access)
dta_gnn ui --host 0.0.0.0

# Create public Gradio share link (temporary)
dta_gnn ui --share
```

### Features Available in UI

- Dataset building with visual feedback
- Interactive parameter configuration
- Activity distribution plots
- Split size visualization
- Chemical space analysis
- Model training (RF, GNN)
- Hyperparameter optimization
- Download results as CSV/ZIP

## train-gnn

Run the complete end-to-end GNN training pipeline from a UniProt protein accession. The command handles UniProt→ChEMBL mapping, dataset building with scaffold split, W&B Bayesian hyperparameter search, final model training, and test evaluation — all timed per step.

### Usage

```bash
dta_gnn train-gnn UNIPROT_IDS [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `UNIPROT_IDS` | UniProt accession(s), comma/space/semicolon-separated. E.g. `P00533` or `P00533,P04637` |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--architecture` | `gin` | GNN architecture: `gin\|gcn\|gat\|sage\|pna\|transformer\|tag\|arma\|cheb\|supergat` |
| `--sqlite-path` | _(web API)_ | Path to ChEMBL SQLite DB. Omit to use the web API. |
| `--standard-types` | _(all)_ | Comma-separated activity types, e.g. `IC50,Ki`. Omit for all types. |
| `--test-size` | `0.2` | Fraction of data for the test split. |
| `--val-size` | `0.1` | Fraction of data for the validation split. |
| `--wandb-project` | `dta_gnn` | W&B project name for HPO and final training runs. |
| `--wandb-entity` | _(none)_ | W&B entity / team name. |
| `--wandb-api-key` | _(env)_ | W&B API key. Falls back to `WANDB_API_KEY` env variable. |
| `--n-trials` | `20` | Number of Bayesian HPO sweep trials. |
| `--epochs` | `30` | Epochs for final model training. |
| `--batch-size` | `64` | Mini-batch size for GNN training. |
| `--device` | _(auto)_ | Device: `auto\|mps\|cuda\|cpu`. Defaults to auto-detect. |
| `--runs-root` | `runs` | Root directory for timestamped run folders. |

### Examples

```bash
# Minimal — uses web API for ChEMBL data
dta_gnn train-gnn P00533

# With local SQLite database (faster)
dta_gnn train-gnn P00533 --architecture gin --sqlite-path ./chembl_dbs/chembl_36.db

# Full options — multiple targets, filtered activity types, custom HPO budget
dta_gnn train-gnn "P00533,P04637" \
    --architecture gat \
    --sqlite-path ./chembl_dbs/chembl_36.db \
    --standard-types IC50,Ki \
    --n-trials 40 --epochs 50 \
    --wandb-project egfr_study \
    --wandb-entity my_team
```

### Terminal Output

```
Run directory : runs/20260309_142301
Dataset size  : 4821 rows  (train=3735, val=469, test=617)
Best val R²   : 0.7842
Test metrics  :  rmse=0.6231  mae=0.4812  r2=0.7614

Timings
  uniprot_mapping                     1.3s  (0%)
  dataset_build                     182.4s  (45%)
  hyperparameter_search             201.7s  (50%)
  final_training                     18.2s  (5%)
  ──────────────────────────────────────────
  Total                             403.6s  (6.7 min)
```

## Shell Completion

Enable shell completion for a better CLI experience:

=== "Bash"

    ```bash
    # Add to ~/.bashrc
    eval "$(_DTA_GNN_COMPLETE=bash_source dta_gnn)"
    ```

=== "Zsh"

    ```bash
    # Add to ~/.zshrc
    eval "$(_DTA_GNN_COMPLETE=zsh_source dta_gnn)"
    ```

=== "Fish"

    ```bash
    # Add to ~/.config/fish/completions/dta_gnn.fish
    _DTA_GNN_COMPLETE=fish_source dta_gnn | source
    ```

## Scripting Examples

For a quick end-to-end run from the terminal, use `train-gnn`. For fine-grained control over individual pipeline steps, use the [Python API](python-api.md).

## Troubleshooting

### Command not found

Ensure the package is installed in your active environment:

```bash
pip show dta-gnn
```

### Permission denied

On Unix systems, ensure the script is executable:

```bash
which dta_gnn
# Should show path to the script
```

### Database errors

Verify database path and permissions:

```bash
ls -la ./chembl_dbs/chembl_36.db
sqlite3 ./chembl_dbs/chembl_36.db "SELECT COUNT(*) FROM activities;"
```
