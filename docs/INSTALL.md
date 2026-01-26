# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Options

### Option 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
cd /Users/k.jones/Desktop/geostats

# Install dependencies
pip install numpy scipy matplotlib pandas

# Install in development mode
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

### Option 2: Install Dependencies Only

```bash
pip install -r requirements.txt
```

### Option 3: Create requirements.txt

Create a `requirements.txt` file:

```txt
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
```

Then install:
```bash
pip install -r requirements.txt
```

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
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=geostats --cov-report=html
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
pip install black flake8 mypy sphinx sphinx-rtd-theme

# Format code
black src/geostats tests examples

# Check style
flake8 src/geostats tests

# Type checking
mypy src/geostats
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
pip install numpy scipy

# Linux
sudo apt-get install libopenblas-dev liblapack-dev
pip install numpy scipy
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
