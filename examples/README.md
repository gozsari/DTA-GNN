# DTA-GNN Examples

This directory contains step-by-step Jupyter notebook tutorials for using DTA-GNN.

## Notebooks

### 1. Baseline Models (`01_baseline_models.ipynb`)

Learn how to:
- Build DTA datasets from ChEMBL
- Train Random Forest models using Morgan fingerprints
- Train SVR (Support Vector Regression) models
- Evaluate model performance
- Make predictions on new molecules

**Best for**: Getting started quickly, understanding the pipeline, baseline comparisons

### 2. GNN Models (`02_gnn_models.ipynb`)

Learn how to:
- Build datasets for GNN training
- Understand molecular graph representation
- Train Graph Neural Networks (10 architectures supported)
- Compare different GNN architectures
- Extract molecule embeddings
- Make predictions with GNN models

**Best for**: Deep learning workflows, advanced modeling, embedding extraction

## Getting Started

### Option 1: Using Virtual Environment (Recommended)

1. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install DTA-GNN and Jupyter**:
   ```bash
   pip install dta-gnn jupyter ipykernel
   ```

3. **Register the virtual environment as a Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name=dta-gnn --display-name "Python (dta-gnn)"
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

5. **Select the kernel**: In Jupyter, go to Kernel → Change Kernel → Select "Python (dta-gnn)"

### Option 2: Using System Python

1. **Install DTA-GNN**:
   ```bash
   pip install dta-gnn
   ```

2. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter
   ```

3. **Launch Jupyter**:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

4. **Open a notebook** and follow along step-by-step!

## Quick Setup

**For Linux/macOS:**
```bash
cd examples
./setup_kernel.sh
```

**For Windows:**
```cmd
cd examples
setup_kernel.bat
```

This script will:
- Create a virtual environment (if it doesn't exist)
- Install DTA-GNN and Jupyter
- Register the virtual environment as a Jupyter kernel

## Requirements

- Python 3.10+
- DTA-GNN installed
- Jupyter Lab or Jupyter Notebook
- (Optional) ChEMBL SQLite database for faster data access
- (Optional) GPU/MPS for faster GNN training

## Data Sources

The notebooks support two data sources:

1. **Web API** (default in notebooks)
   - No setup required
   - Slower but convenient
   - Requires internet connection

2. **SQLite Database** (recommended)
   - Much faster
   - Works offline
   - Download: `dta_gnn setup --version 36 --dir ./chembl_dbs`

To use SQLite in the notebooks, uncomment the SQLite initialization lines and adjust the path.

## Tips

- Start with the baseline models notebook to understand the workflow
- Use SQLite for faster dataset building
- Enable GPU/MPS for GNN training (automatic detection)
- Adjust batch sizes based on your hardware
- Experiment with different targets and architectures

## Troubleshooting

### Notebook Can't Find Kernel / Wrong Environment

**Problem**: Notebook runs but can't import `dta_gnn` or uses wrong Python environment.

**Solution 1** (Recommended): Use the setup script
```bash
cd examples
./setup_kernel.sh  # Linux/macOS
# or
setup_kernel.bat   # Windows
```
Then in Jupyter: Kernel → Change Kernel → Python (dta-gnn)

**Solution 2**: Manually register the kernel
```bash
# Activate your virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install ipykernel if not already installed
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=dta-gnn --display-name "Python (dta-gnn)"
```

**Solution 3**: Use the venv's kernel directly
```bash
# Activate virtual environment
source .venv/bin/activate

# Install jupyter in the venv
pip install jupyter

# Launch Jupyter from within the venv
jupyter lab
```
The notebooks should automatically use the venv's Python.

### Other Issues

- **Import errors**: Ensure DTA-GNN is installed (`pip install -e .` from source)
- **Slow dataset building**: Switch to SQLite database
- **Out of memory**: Reduce batch size or model dimensions
- **Device not detected**: Check PyTorch installation and GPU drivers

For more help, see the [documentation](../docs/).
