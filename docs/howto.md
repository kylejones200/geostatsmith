# How-To Guides

Step-by-step guides for common tasks in GeoStats.

## Basic Interpolation

### Simple Kriging

```python
from geostats import OrdinaryKriging
from geostats.models import SphericalModel

# Create kriging object
ok = OrdinaryKriging(x, y, z, variogram_model=SphericalModel())

# Predict at new locations
predictions, variance = ok.predict(x_new, y_new)
```

### Variogram Analysis

```python
from geostats import variogram

# Calculate experimental variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)

# Fit model
model = variogram.fit_model("spherical", lags, gamma)
```

## Advanced Topics

### Config-Driven Analysis

See [Config-Driven Analysis](CONFIG_DRIVEN.md) for complete workflows from YAML configuration.

### AutoML

```python
from geostats.automl import auto_method

# Automatically select best method
best_method = auto_method(x, y, z)
```

### Performance Optimization

```python
from geostats.performance import parallel_kriging

# Parallel processing for large datasets
results = parallel_kriging(x, y, z, x_pred, y_pred, n_jobs=-1)
```

## Common Workflows

See [Tutorials](TUTORIALS.md) for complete workflow examples.
