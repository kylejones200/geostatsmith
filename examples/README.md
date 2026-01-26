# GeoStats Examples

This directory contains example scripts demonstrating the usage of the GeoStats library.

## Examples

### Example 1: Basic Variogram Analysis
**File:** `example_1_basic_variogram.py`

Demonstrates:
- Generating synthetic spatial data
- Calculating experimental variogram
- Fitting various variogram models (Spherical, Exponential, Gaussian)
- Automatic model selection
- Visualizing variogram models

**Run:**
```bash
python example_1_basic_variogram.py
```

### Example 2: Kriging Interpolation
**File:** `example_2_kriging_interpolation.py`

Demonstrates:
- Ordinary Kriging interpolation
- Prediction on a regular grid
- Kriging variance (uncertainty quantification)
- Leave-one-out cross-validation
- Visualization of results

**Run:**
```bash
python example_2_kriging_interpolation.py
```

### Example 3: Comparison of Kriging Methods
**File:** `example_3_comparison_kriging_methods.py`

Demonstrates:
- Simple Kriging
- Ordinary Kriging
- Universal Kriging with linear drift
- Comparison of methods on data with trend
- Cross-validation metrics comparison

**Run:**
```bash
python example_3_comparison_kriging_methods.py
```

## Requirements

All examples require:
- numpy
- matplotlib
- geostats (this library)

Install with:
```bash
pip install numpy matplotlib
pip install -e .. # Install geostats in development mode
```

## Expected Output

Each example will:
1. Print progress information to the console
2. Display performance metrics
3. Generate and save publication-quality plots
4. Show interactive plots (if running in an interactive environment)

## Notes

- Examples use synthetic data for reproducibility
- Random seeds are set for consistent results
- Plots are saved as PNG files in the examples directory
- You can modify parameters to experiment with different scenarios
