# GeoStats User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Basic Workflows](#basic-workflows)
5. [Advanced Topics](#advanced-topics)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

GeoStats is a comprehensive Python library for geostatistical analysis. This guide will help you understand how to use it effectively for your spatial data analysis needs.

### What is Geostatistics?

Geostatistics is a branch of statistics that deals with spatial or spatiotemporal datasets. It provides methods for:
- **Interpolation**: Estimating values at unsampled locations
- **Uncertainty Quantification**: Assessing prediction uncertainty
- **Simulation**: Generating multiple realizations of spatial fields
- **Optimization**: Designing optimal sampling strategies

### When to Use GeoStats

GeoStats is ideal for:
- **Mining & Exploration**: Ore grade estimation, resource modeling
- **Environmental Science**: Pollution mapping, soil property estimation
- **Hydrology**: Groundwater level mapping, contaminant transport
- **Agriculture**: Soil nutrient mapping, yield prediction
- **Climate Science**: Temperature/precipitation interpolation
- **Any spatial interpolation problem** where you need uncertainty estimates

---

## Installation

### System Requirements

- **Python**: 3.12 or newer
- **Operating System**: Ubuntu (Linux) recommended
- **Memory**: 4GB+ RAM recommended for large datasets
- **Disk Space**: ~500MB for installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/kylejones200/geostatsmith.git
cd geostatsmith

# Install with pip
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Optional Dependencies

For enhanced functionality, install optional dependencies:

```bash
# For raster I/O (GeoTIFF support)
pip install rasterio

# For NetCDF support
pip install netCDF4

# For GeoJSON support
pip install geopandas

# For interactive visualizations
pip install plotly

# For ML-enhanced kriging
pip install scikit-learn xgboost
```

### Verify Installation

```python
import geostats
print(geostats.__version__)  # Should print 0.3.0

# Test basic functionality
from geostats import variogram, kriging
print("Installation successful!")
```

---

## Core Concepts

### 1. Variograms

A **variogram** describes the spatial correlation structure of your data. It shows how similar values are at different distances.

**Key Components:**
- **Nugget**: Variance at zero distance (measurement error + micro-scale variation)
- **Sill**: Maximum variance (total variance of the process)
- **Range**: Distance at which correlation becomes negligible

**Types:**
- **Experimental Variogram**: Calculated from your data
- **Theoretical Variogram Model**: Mathematical function fitted to experimental data

### 2. Kriging

**Kriging** is a geostatistical interpolation method that:
- Provides **best linear unbiased predictions** (BLUP)
- Quantifies **prediction uncertainty** (kriging variance)
- Accounts for **spatial correlation** structure

**Types of Kriging:**
- **Simple Kriging**: Assumes known mean
- **Ordinary Kriging**: Estimates mean from data (most common)
- **Universal Kriging**: Accounts for trends/drift
- **Indicator Kriging**: For probability estimation
- **Cokriging**: Uses multiple correlated variables

### 3. Simulation

**Geostatistical simulation** generates multiple equally-probable realizations of a spatial field, useful for:
- Uncertainty quantification
- Risk assessment
- Resource estimation

---

## Basic Workflows

### Workflow 1: Basic Interpolation

**Goal**: Interpolate values at unsampled locations

```python
import numpy as np
from geostats import variogram, kriging

# 1. Prepare your data
x = np.array([0, 10, 20, 30, 40, 50])  # X coordinates
y = np.array([0, 10, 20, 30, 40, 50])  # Y coordinates
z = np.array([1.2, 2.1, 1.8, 3.2, 2.9, 3.5])  # Values

# 2. Calculate experimental variogram
lags, gamma, n_pairs = variogram.experimental_variogram(
    x, y, z, 
    n_lags=10,
    maxlag=50
)

# 3. Fit theoretical model
model = variogram.fit_model(
    'spherical',  # Model type
    lags, gamma,
    weights=n_pairs
)

print(f"Fitted model: {model.parameters}")

# 4. Perform kriging
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)

# 5. Predict at new locations
x_new = np.array([5, 15, 25, 35, 45])
y_new = np.array([5, 15, 25, 35, 45])
predictions, variances = ok.predict(x_new, y_new, return_variance=True)

print(f"Predictions: {predictions}")
print(f"Standard deviations: {np.sqrt(variances)}")
```

### Workflow 2: Grid Interpolation

**Goal**: Create a continuous map (grid) from point data

```python
import numpy as np
from geostats import variogram, kriging

# 1. Your point data
x = np.random.uniform(0, 100, 50)
y = np.random.uniform(0, 100, 50)
z = np.sin(x/20) + np.cos(y/20) + np.random.randn(50) * 0.2

# 2. Create prediction grid
x_grid = np.linspace(0, 100, 100)
y_grid = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# 3. Fit variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)
model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

# 4. Krige on grid
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)
z_pred, z_var = ok.predict(X.flatten(), Y.flatten(), return_variance=True)

# 5. Reshape for plotting
Z_pred = z_pred.reshape(X.shape)
Z_var = z_var.reshape(X.shape)

# 6. Visualize
import matplotlib.pyplot as plt
plt.contourf(X, Y, Z_pred, levels=20)
plt.colorbar()
plt.scatter(x, y, c=z, edgecolors='k', s=50)
plt.title('Kriging Interpolation')
plt.show()
```

### Workflow 3: Cross-Validation

**Goal**: Validate your model's predictive performance

```python
from geostats import variogram, kriging
from geostats.validation.cross_validation import leave_one_out

# Your data
x, y, z = ...  # Your data

# Fit variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)
model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

# Create kriging object
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)

# Cross-validate
predictions, metrics = leave_one_out(ok)

print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"R²: {metrics['r2']:.3f}")

# Plot predictions vs actual
import matplotlib.pyplot as plt
plt.scatter(z, predictions)
plt.plot([z.min(), z.max()], [z.min(), z.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Cross-Validation Results')
plt.show()
```

---

## Advanced Topics

### 1. Handling Trends

If your data has a trend (non-stationary mean), use **Universal Kriging**:

```python
from geostats.algorithms.universal_kriging import UniversalKriging

# Fit variogram to residuals (detrended data)
# ... (detrending code) ...

uk = UniversalKriging(
    x, y, z,
    variogram_model=model,
    drift_terms='linear'  # or 'quadratic'
)

predictions, variances = uk.predict(x_new, y_new)
```

### 2. Multiple Variables (Cokriging)

When you have correlated secondary variables:

```python
from geostats.algorithms.cokriging import Cokriging

# Primary variable
x1, y1, z1 = ...  # Gold grades

# Secondary variable (correlated)
x2, y2, z2 = ...  # Silver grades

# Fit variograms
model1 = ...  # Primary variogram
model2 = ...  # Secondary variogram

# Cokriging
ck = Cokriging(
    x1, y1, z1,  # Primary
    x2, y2, z2,  # Secondary
    variogram_primary=model1,
    variogram_secondary=model2
)

predictions, variances = ck.predict(x_new, y_new)
```

### 3. Probability Estimation (Indicator Kriging)

For probability maps (e.g., probability of exceeding a threshold):

```python
from geostats.algorithms.indicator_kriging import IndicatorKriging

# Define threshold
threshold = 2.0

# Create indicator variable
z_indicator = (z > threshold).astype(float)

# Fit indicator variogram
lags, gamma, n_pairs = variogram.experimental_variogram(
    x, y, z_indicator
)
model = variogram.fit_model('spherical', lags, gamma)

# Indicator kriging
ik = IndicatorKriging(
    x, y, z_indicator,
    variogram_model=model
)

probabilities, variances = ik.predict(x_new, y_new)
```

### 4. Simulation

Generate multiple realizations for uncertainty:

```python
from geostats.simulation.gaussian_simulation import sequential_gaussian_simulation

# Your data
x, y, z = ...

# Fit variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)
model = variogram.fit_model('spherical', lags, gamma)

# Generate realizations
realizations = sequential_gaussian_simulation(
    x, y, z,
    x_grid, y_grid,  # Grid to simulate on
    variogram_model=model,
    n_realizations=100
)

# Calculate statistics
mean_map = np.mean(realizations, axis=0)
std_map = np.std(realizations, axis=0)
```

---

## Best Practices

### 1. Data Quality

- **Check for outliers**: Use `geostats.diagnostics.outlier_detection`
- **Handle missing data**: Remove or impute missing values
- **Check coordinate system**: Ensure consistent units
- **Validate data**: Use `geostats.core.validators`

### 2. Variogram Fitting

- **Use enough lags**: At least 10-15 lags
- **Check experimental variogram**: Look for patterns
- **Try multiple models**: Compare fit quality
- **Use weights**: Weight by number of pairs
- **Validate fit**: Check residuals

### 3. Kriging Parameters

- **Neighborhood size**: Balance accuracy vs. speed
- **Maximum neighbors**: 15-25 typically sufficient
- **Search radius**: 2-3 times the variogram range
- **Regularization**: Use for stability if needed

### 4. Validation

- **Always cross-validate**: Use leave-one-out or k-fold
- **Check metrics**: RMSE, MAE, R²
- **Visual inspection**: Plot predictions vs. actual
- **Spatial validation**: Check for spatial patterns in residuals

### 5. Performance

- **Use parallel processing**: For large datasets
- **Cache results**: If re-predicting on same grid
- **Chunk large grids**: Process in blocks
- **Use approximate methods**: For very large datasets

---

## Troubleshooting

### Common Issues

#### 1. "Singular matrix" error

**Cause**: Collocated points or numerical instability

**Solutions**:
- Remove duplicate coordinates
- Use regularization: `regularize_matrix()`
- Increase `epsilon` parameter
- Check for collinear points

#### 2. Poor variogram fit

**Cause**: Insufficient data or wrong model

**Solutions**:
- Increase number of lags
- Try different models (spherical, exponential, Gaussian)
- Check for trends (use Universal Kriging)
- Use more data points

#### 3. Unrealistic predictions

**Cause**: Extrapolation beyond data range

**Solutions**:
- Limit search radius
- Use fewer neighbors
- Check variogram range
- Consider using simulation instead

#### 4. Slow performance

**Cause**: Large datasets or fine grids

**Solutions**:
- Use parallel processing
- Reduce grid resolution
- Limit maximum neighbors
- Use approximate methods
- Enable caching

#### 5. Memory errors

**Cause**: Very large datasets

**Solutions**:
- Process in chunks
- Use chunked kriging
- Reduce grid size
- Use approximate methods
- Increase system memory

### Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: Report on GitHub
- **Questions**: Open a discussion

---

## Next Steps

1. **Try the examples**: Run scripts in `examples/` directory
2. **Read API docs**: See `docs/QUICK_REFERENCE.md`
3. **Explore config-driven workflows**: See `docs/CONFIG_DRIVEN.md`
4. **Check advanced topics**: See other guides in `docs/`

---

**Happy geostatistical analysis!**
