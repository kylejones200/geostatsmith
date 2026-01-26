## GeoStats Quick Start Guide

### Installation

```bash
# From source (development)
git clone https://github.com/yourusername/geostats.git
cd geostats
pip install -e ".[dev]"

# Or directly
pip install geostats
```

### Basic Workflow

The typical geostatistical workflow consists of three steps:

1. **Calculate experimental variogram**
2. **Fit a theoretical variogram model**
3. **Perform kriging interpolation**

### Example: Complete Workflow

```python
import numpy as np
from geostats import variogram, kriging

# 1. Your data
x = np.array([0, 10, 20, 30, 40, 50]) # X coordinates
y = np.array([0, 10, 20, 30, 40, 50]) # Y coordinates
z = np.array([1.2, 2.1, 1.8, 3.2, 2.9, 3.5]) # Values

# 2. Calculate experimental variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=10)

# 3. Fit variogram model
model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)
print(f"Model parameters: {model.parameters}")

# 4. Perform Ordinary Kriging
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)

# 5. Predict at new locations
x_new = np.array([5, 15, 25])
y_new = np.array([5, 15, 25])
predictions, variances = ok.predict(x_new, y_new, return_variance=True)

print(f"Predictions: {predictions}")
print(f"Uncertainties (std dev): {np.sqrt(variances)}")
```

### Variogram Models

The library provides several theoretical variogram models:

```python
from geostats.models.variogram_models import (
 SphericalModel,
 ExponentialModel,
 GaussianModel,
 LinearModel,
 PowerModel,
 MaternModel,
)

# Create a model with specific parameters
model = SphericalModel(nugget=0.1, sill=1.0, range_param=20.0)

# Evaluate at distance h
h = np.linspace(0, 30, 100)
gamma = model(h)
```

### Automatic Model Selection

Let the library choose the best model:

```python
result = variogram.auto_fit(lags, gamma, weights=n_pairs, criterion='rmse')
best_model = result['model']
print(f"Best model: {best_model.__class__.__name__}")
print(f"Score: {result['score']}")
```

### Kriging Methods

#### Ordinary Kriging (Most Common)

```python
from geostats.kriging import OrdinaryKriging

ok = OrdinaryKriging(x, y, z, variogram_model=model)
predictions, variances = ok.predict(x_new, y_new)
```

#### Simple Kriging

```python
from geostats.kriging import SimpleKriging

# Requires known mean
sk = SimpleKriging(x, y, z, variogram_model=model, mean=2.5)
predictions, variances = sk.predict(x_new, y_new)
```

#### Universal Kriging (For Data with Trend)

```python
from geostats.kriging import UniversalKriging

# Handles linear or quadratic trends
uk = UniversalKriging(
 x, y, z,
 variogram_model=model,
 drift_terms='linear' # or 'quadratic'
)
predictions, variances = uk.predict(x_new, y_new)
```

### Cross-Validation

Validate your model:

```python
# Leave-one-out cross-validation
cv_predictions, metrics = ok.cross_validate()

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

### Interpolation on a Grid

```python
from geostats.utils import create_grid, interpolate_to_grid

# Create prediction grid
X, Y = create_grid(
 x_min=0, x_max=100,
 y_min=0, y_max=100,
 resolution=50
)

# Interpolate
X, Y, Z, V = interpolate_to_grid(
 ok, 0, 100, 0, 100,
 resolution=50,
 return_variance=True
)

# Visualize
import matplotlib.pyplot as plt
plt.contourf(X, Y, Z, levels=15, cmap='viridis')
plt.colorbar(label='Predicted Value')
plt.scatter(x, y, c=z, edgecolors='white', s=50, zorder=5)
plt.title('Kriging Interpolation')
plt.show()
```

### Advanced: Anisotropy

Handle directional spatial correlation:

```python
from geostats.models.anisotropy import AnisotropicModel, DirectionalVariogram

# Detect anisotropy
dir_vario = DirectionalVariogram(x, y, z)
aniso_params = dir_vario.fit_anisotropy(angles=[0, 45, 90, 135])

print(f"Major direction: {aniso_params['major_angle']}°")
print(f"Anisotropy ratio: {aniso_params['ratio']:.3f}")

# Use anisotropic model
from geostats.models.variogram_models import SphericalModel
base_model = SphericalModel(nugget=0.1, sill=1.0, range_param=20.0)
aniso_model = AnisotropicModel(
 base_model,
 angle=aniso_params['major_angle'],
 ratio=aniso_params['ratio']
)
```

### Robust Variogram

For data with outliers:

```python
lags, gamma, n_pairs = variogram.robust_variogram(
 x, y, z,
 n_lags=15,
 estimator='cressie' # or 'dowd'
)
```

### Synthetic Data Generation

For testing and learning:

```python
from geostats.utils import generate_synthetic_data

x, y, z = generate_synthetic_data(
 n_points=100,
 spatial_structure='spherical',
 nugget=0.1,
 sill=1.0,
 range_param=20.0,
 seed=42 # For reproducibility
)
```

### Tips for Production Use

1. **Always validate your variogram model** using cross-validation
2. **Check for anisotropy** in your data using directional variograms
3. **Handle outliers** using robust variogram estimators
4. **Use appropriate kriging method**:
 - Ordinary Kriging: General purpose, most common
 - Simple Kriging: When mean is well-known
 - Universal Kriging: When data has large-scale trends
5. **Monitor kriging variance** to assess prediction uncertainty
6. **Use enough sample points**: Generally 50+ for reliable variogram

### Common Issues and Solutions

#### Issue: Negative kriging variance
```python
# Solution: Regularize the covariance matrix
# This is handled automatically, but you can adjust:
from geostats.math.matrices import regularize_matrix
# The library uses epsilon=1e-10 by default
```

#### Issue: Poor variogram fit
```python
# Solution 1: Try automatic fitting
result = variogram.auto_fit(lags, gamma)

# Solution 2: Use robust estimator
lags, gamma, _ = variogram.robust_variogram(x, y, z, estimator='cressie')

# Solution 3: Adjust lag parameters
lags, gamma, _ = variogram.experimental_variogram(
 x, y, z,
 n_lags=20, # More lags
 maxlag=max_distance/3 # Shorter maximum distance
)
```

#### Issue: Singular kriging matrix
```python
# Solution: Check for duplicate locations
unique_coords = np.unique(np.column_stack([x, y]), axis=0)
if len(unique_coords) < len(x):
 print("Warning: Duplicate locations detected")
# Remove duplicates or add small random noise
```

### Further Reading

- Full documentation: [https://geostats.readthedocs.io](https://geostats.readthedocs.io)
- Examples directory: `examples/`
- API reference: See module docstrings
- Theoretical background: See `ARCHITECTURE.md`

### Getting Help

- GitHub Issues: [https://github.com/yourusername/geostats/issues](https://github.com/yourusername/geostats/issues)
- Documentation: [https://geostats.readthedocs.io](https://geostats.readthedocs.io)
