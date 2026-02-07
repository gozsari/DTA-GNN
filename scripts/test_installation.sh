#!/bin/bash
# Quick test script for TestPyPI installation

set -e

echo "🧪 Testing DTA-GNN from TestPyPI"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  No virtual environment detected. Creating one..."
    python3 -m venv test_env
    source test_env/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "✅ Virtual environment: $VIRTUAL_ENV"
fi

echo ""
echo "📦 Installing/upgrading package from TestPyPI..."
pip install --upgrade pip
pip install --force-reinstall --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dta-gnn

echo ""
echo "🔍 Checking installation..."
echo "CLI location: $(which dta_gnn)"
echo "Python path: $(python -c 'import dta_gnn; print(dta_gnn.__file__)')"

echo ""
echo "✅ Testing CLI..."
dta_gnn --help

echo ""
echo "✅ Testing Python import..."
python -c "
import dta_gnn
print(f'✅ Import successful')
print(f'📦 Version: {dta_gnn.__version__}')
"

echo ""
echo "✅ All tests passed!"
