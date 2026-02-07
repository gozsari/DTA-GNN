# Contributing

Thank you for your interest in contributing to DTA-GNN! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) CUDA-capable GPU for GNN development

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/gozsari/DTA-GNN.git
cd dta_gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install all optional dependencies for full testing
pip install -e ".[dev,molecule-gnn,wandb]"
```

### Verify Setup

```bash
# Run tests
pytest

# Check linting
ruff check src/

# Format code
black src/ tests/
```

## Project Structure

```
dta_gnn/
├── src/
│   └── dta_gnn/
│       ├── __init__.py
│       ├── cli.py              # CLI commands
│       ├── pipeline.py         # Main pipeline
│       ├── app/
│       │   └── ui.py           # Gradio web interface
│       ├── app_features/       # UI feature helpers
│       ├── audits/             # Leakage detection
│       ├── cleaning/           # Data standardization
│       ├── exporters/          # Export utilities
│       ├── features/           # Featurization
│       ├── io/                 # Data sources
│       ├── labeling/           # Label generation
│       ├── models/             # Model training
│       ├── negatives/          # Negative sampling
│       ├── splits/             # Splitting strategies
│       └── visualization.py    # Plotting
├── tests/
│   ├── test_audits.py
│   ├── test_cleaning.py
│   └── ...
├── docs/                       # Documentation
├── pyproject.toml             # Package configuration
└── README.md
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow the coding standards (below).

### 3. Write Tests

```python
# tests/test_your_feature.py
import pytest
from dta_gnn.your_module import your_function

def test_your_function_basic():
    result = your_function(input_data)
    assert result == expected_output

def test_your_function_edge_case():
    with pytest.raises(ValueError):
        your_function(invalid_input)
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_feature.py

# Run with coverage
pytest --cov=dta_gnn

# Run only fast tests (skip slow ones)
pytest -m "not slow"
```

### 5. Format and Lint

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat: add new feature description"
git push origin feature/your-feature-name
```

### 7. Create Pull Request

- Go to GitHub
- Create PR from your branch to `main`
- Fill out the PR template
- Request review

## Coding Standards

### Style Guide

- **Black** for formatting (line length 88)
- **Ruff** for linting
- **Type hints** for all public functions
- **Docstrings** for all public functions/classes

### Example Function

```python
from typing import Optional, List
import pandas as pd

def process_data(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.5,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Process data with specified columns and threshold.
    
    Args:
        df: Input DataFrame with activity data.
        columns: Column names to process.
        threshold: Minimum value threshold. Defaults to 0.5.
        drop_missing: Whether to drop rows with missing values.
            Defaults to True.
    
    Returns:
        Processed DataFrame with filtered data.
    
    Raises:
        ValueError: If columns are not in DataFrame.
    
    Example:
        >>> df = pd.DataFrame({"value": [0.3, 0.7, 0.9]})
        >>> result = process_data(df, ["value"], threshold=0.5)
        >>> len(result)
        2
    """
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    
    result = df.copy()
    if drop_missing:
        result = result.dropna(subset=columns)
    
    for col in columns:
        result = result[result[col] >= threshold]
    
    return result
```

### Import Order

```python
# Standard library
import json
from pathlib import Path
from typing import Optional, List

# Third-party
import numpy as np
import pandas as pd
from loguru import logger

# Local
from dta_gnn.cleaning import standardize_activities
from dta_gnn.splits import split_random
```

## Testing Guidelines

### Test Organization

```python
# Group tests by functionality
class TestDataCleaning:
    """Tests for data cleaning functions."""
    
    def test_standardize_basic(self):
        """Test basic standardization."""
        ...
    
    def test_standardize_with_missing(self):
        """Test handling of missing values."""
        ...

class TestDataCleaningEdgeCases:
    """Edge case tests for data cleaning."""
    
    def test_empty_dataframe(self):
        """Test with empty input."""
        ...
```

### Fixtures

```python
# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_activities():
    """Sample activity DataFrame for testing."""
    return pd.DataFrame({
        "molecule_chembl_id": ["CHEMBL25", "CHEMBL192"],
        "target_chembl_id": ["CHEMBL204", "CHEMBL204"],
        "standard_value": [100.0, 50.0],
        "standard_units": ["nM", "nM"],
        "pchembl_value": [7.0, 7.3]
    })

@pytest.fixture
def sample_smiles():
    """Sample SMILES for testing."""
    return ["CCO", "CC(=O)O", "c1ccccc1"]
```

### Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_full_pipeline():
    """Test complete pipeline (slow)."""
    ...

# Mark tests requiring database
@pytest.mark.requires_db
def test_sqlite_source():
    """Test SQLite source."""
    ...

# Mark tests requiring optional deps
@pytest.mark.requires_torch
def test_gnn_training():
    """Test GNN training."""
    ...
```

## Documentation

### Building Docs

```bash
# Install MkDocs
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

### Writing Docs

- Use clear, concise language
- Include code examples
- Add cross-references with `[link text](page.md)`
- Use admonitions for tips/warnings:

```markdown
!!! tip "Performance Tip"
    Use SQLite for faster data loading.

!!! warning
    Random splits may cause scaffold leakage.
```

## Pull Request Guidelines

### PR Title Format

```
type: brief description

Examples:
feat: add temporal split strategy
fix: handle empty SMILES in fingerprint calculation
docs: update installation guide
test: add tests for scaffold splitting
refactor: simplify pipeline data flow
```

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Linting passes (`ruff check`)
- [ ] Code is formatted (`black`)
- [ ] Documentation updated (if applicable)
- [ ] Changelog updated (for significant changes)
- [ ] Type hints added for new functions

### Review Process

1. Automated checks run
2. Reviewer assigned
3. Address feedback
4. Approval required
5. Merge to main

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Creating a Release

1. Update version in `src/dta_gnn/__init__.py`
2. Update `CHANGELOG.md`
3. Create PR with version bump
4. After merge, create GitHub release
5. CI automatically publishes to PyPI

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Pull Requests**: Code contributions

### Issue Templates

When reporting bugs:

```markdown
## Description
Brief description of the bug.

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: macOS 14.0
- Python: 3.10
- dta_gnn: 0.1.0
```

### Feature Requests

```markdown
## Feature Description
What would you like to add?

## Use Case
Why is this needed?

## Proposed Implementation
Any ideas on how to implement?

## Alternatives Considered
Other approaches you've thought about.
```

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/):

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers
- Prioritize the project's best interests

## Recognition

Contributors are recognized in:

- Release notes
- README acknowledgments
- Annual contributor reports

Thank you for contributing to DTA-GNN! 🎉
