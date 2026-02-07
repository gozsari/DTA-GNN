@echo off
REM Setup script to register .venv as a Jupyter kernel (Windows)

echo Setting up Jupyter kernel for DTA-GNN...

REM Check if .venv exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install required packages
echo Installing DTA-GNN and Jupyter...
pip install -q --upgrade pip
pip install -q dta-gnn jupyter ipykernel

REM Register kernel
echo Registering Jupyter kernel...
python -m ipykernel install --user --name=dta-gnn --display-name "Python (dta-gnn)"

echo.
echo Setup complete!
echo.
echo To use the notebooks:
echo   1. Activate the virtual environment: .venv\Scripts\activate
echo   2. Launch Jupyter: jupyter lab
echo   3. Select kernel: Kernel → Change Kernel → Python (dta-gnn)
echo.
