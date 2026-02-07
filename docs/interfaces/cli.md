# Command Line Interface

DTA-GNN provides a command-line interface for setup, UI launch, and lightweight auditing.

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
| `audit` | Run audit on an existing dataset |
| `setup` | Download and set up ChEMBL database |
| `ui` | Launch the Gradio web interface |

**Note**: Dataset building is done via the Python API (`Pipeline.build_dta()`) or the Web UI (`dta_gnn ui`). The CLI no longer includes dataset building commands.

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

**Note**: Dataset building is done via the [Python API](python-api.md#build_dta) or the [Web UI](ui.md). The CLI is primarily for database setup, UI access, and auditing.

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
