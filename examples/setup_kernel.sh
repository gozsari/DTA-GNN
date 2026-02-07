#!/bin/bash
# Setup script to register .venv as a Jupyter kernel

set -e

echo "Setting up Jupyter kernel for DTA-GNN..."

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install required packages
echo "Installing DTA-GNN and Jupyter..."
pip install -q --upgrade pip
pip install -q dta-gnn jupyter ipykernel

# Register kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=dta-gnn --display-name "Python (dta-gnn)"

echo ""
echo "✓ Setup complete!"
echo ""
echo "To use the notebooks:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Launch Jupyter: jupyter lab"
echo "  3. Select kernel: Kernel → Change Kernel → Python (dta-gnn)"
echo ""
