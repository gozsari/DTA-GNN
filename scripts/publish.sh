#!/bin/bash
# Quick script to build and publish DTA-GNN to PyPI
# Usage: ./scripts/publish.sh [testpypi|pypi]

set -e

REPO=${1:-testpypi}  # Default to testpypi for safety

echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

echo "📦 Building package..."
python -m build

echo "✅ Checking package..."
twine check dist/*

if [ "$REPO" = "testpypi" ]; then
    echo "🚀 Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo ""
    echo "✅ Uploaded to TestPyPI!"
    echo "📦 Test installation with:"
    echo "   pip install --index-url https://test.pypi.org/simple/ dta-gnn"
elif [ "$REPO" = "pypi" ]; then
    echo "🚀 Uploading to PyPI..."
    twine upload --repository pypi dist/*
    echo ""
    echo "✅ Uploaded to PyPI!"
    echo "📦 Install with: pip install dta-gnn"
else
    echo "❌ Invalid repository. Use 'testpypi' or 'pypi'"
    exit 1
fi
