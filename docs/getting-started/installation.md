# Installation

This guide covers all installation options for DTA-GNN.

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB+ RAM recommended (8GB+ for large datasets)

## Basic Installation

Install the core package from PyPI:

```bash
pip install dta-gnn
```

This includes all essential dependencies:

| Package | Purpose |
|---------|---------|
| `typer` | CLI framework |
| `pandas` | Data manipulation |
| `rdkit` | Cheminformatics |
| `scikit-learn` | Machine learning |
| `gradio` | Web UI |
| `loguru` | Logging |
| `pydantic` | Data validation |

GNN support (PyTorch, PyTorch Geometric) and Weights & Biases (W&B) are included in the default install; no extra install step is required.

## Development Installation

For contributing to DTA-GNN:

```bash
# Clone the repository
git clone https://github.com/gozsari/DTA-GNN.git
cd DTA-GNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

Development dependencies include:

- `pytest` - Testing framework
- `ruff` - Fast linter
- `black` - Code formatter
- `build` - Package building
- `twine` - Package publishing

## ChEMBL Database Setup

For optimal performance, we recommend using a local ChEMBL SQLite database instead of the web API.

### Automatic Download

Use the built-in setup command:

```bash
dta_gnn setup --version 36 --dir ./chembl_dbs
```

This will:

1. Download the ChEMBL SQLite database (~5GB compressed)
2. Extract it to the specified directory
3. Verify the database schema

### Manual Download

1. Visit the [ChEMBL Downloads](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/) page
2. Download `chembl_XX_sqlite.tar.gz`
3. Extract: `tar -xzf chembl_XX_sqlite.tar.gz`

## Verification

Verify your installation:

```bash
# Check CLI is available
dta_gnn --help

# Check version
python -c "import dta_gnn; print(dta_gnn.__version__)"

# Test imports
python -c "from dta_gnn.pipeline import Pipeline; print('OK')"
```

## Troubleshooting

### RDKit Installation Issues

RDKit can sometimes be tricky to install. If you encounter issues:

=== "Conda (Recommended)"

    ```bash
    conda create -n chembl python=3.10
    conda activate chembl
    conda install -c conda-forge rdkit
    pip install dta-gnn
    ```

=== "pip (macOS)"

    ```bash
    # Ensure you have the latest pip
    pip install --upgrade pip
    pip install rdkit
    ```

### PyTorch/CUDA Issues

For GPU support with GNNs:

```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

### Memory Issues

For large datasets, increase available memory:

```bash
# Set environment variable for larger datasets
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Docker

To run the Gradio UI in Docker. See the main repository `Dockerfile` for build/run commands.

## Next Steps

- [Quick Start Guide](quickstart.md) - Build your first dataset
- [Data Sources](../user-guide/data-sources.md) - Configure data ingestion
- [Web UI](../interfaces/ui.md) - Launch the interactive interface
