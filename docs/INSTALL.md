# Installation Guide

## Prerequisites

- Python 3.12 or higher
- uv package manager (install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Installation Options

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/kylejones200/geostats.git
cd geostats

# Install in development mode with all dependencies
uv sync --dev
```

### Option 2: Install Core Dependencies Only

```bash
uv sync
```

### Option 3: Install with All Optional Features

```bash
uv sync --all-extras
```

**Note**: The project now uses `uv` and `pyproject.toml` as the single source of truth. Legacy `requirements.txt` files are kept for reference but `uv.lock` is authoritative.

## Verify Installation

```python
# Test the installation
python3 -c "
import sys
sys.path.insert(0, 'src')

from geostats import variogram, kriging
print('GeoStats successfully installed!')
print(f'Available modules: variogram, kriging, models, validation, utils')
"
```

## Running Tests

```bash
# Install test dependencies (included with --dev)
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=geostats --cov-report=html
```

## Running Examples

```bash
cd examples

# Run basic variogram example
python example_1_basic_variogram.py

# Run kriging interpolation example
python example_2_kriging_interpolation.py

# Run comparison example
python example_3_comparison_kriging_methods.py
```

## Development Setup

For development, install additional tools:

```bash
# Install all dev dependencies
uv sync --dev

# Format code
uv run black src/geostats tests examples

# Check style with ruff
uv run ruff check .

# Or with flake8
uv run flake8 src/geostats tests

# Type checking
uv run mypy src/geostats
```

## Quick Start After Installation

```python
import numpy as np
from geostats import variogram, kriging

# Generate sample data
x = np.random.rand(50) * 100
y = np.random.rand(50) * 100
z = np.sin(x/10) + np.cos(y/10)

# Calculate variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)

# Fit model
model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

# Perform kriging
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)
x_new, y_new = np.array([25, 50]), np.array([25, 50])
predictions, variances = ok.predict(x_new, y_new)

print(f"Predictions: {predictions}")
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
```bash
# Ensure you're in the project root
cd /Users/k.jones/Desktop/geostats

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Or in Python:
```python
import sys
sys.path.insert(0, '/Users/k.jones/Desktop/geostats/src')
```

### Dependency Issues

If scipy installation fails:
```bash
# macOS
brew install openblas
uv sync

# Linux
sudo apt-get install libopenblas-dev liblapack-dev
uv sync
```

### Memory Issues with Large Datasets

For large datasets (>10,000 points):
- Use spatial subsampling
- Implement moving window kriging
- Consider using sparse matrices

## Next Steps

1. Read `QUICKSTART.md` for usage guide
2. Check `examples/` directory for complete examples
3. Review `ARCHITECTURE.md` to understand the design
4. See `CONTRIBUTING.md` if you want to contribute

## Support

- Documentation: See README.md and docstrings
- Examples: Check the `examples/` directory
- Issues: Open an issue on GitHub
- Architecture: See ARCHITECTURE.md
