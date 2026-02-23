# GeoStats API Reference

Complete API reference for all GeoStats modules and functions.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Variogram Analysis](#variogram-analysis)
3. [Kriging Methods](#kriging-methods)
4. [Simulation](#simulation)
5. [Transformations](#transformations)
6. [I/O Operations](#io-operations)
7. [Validation](#validation)
8. [Visualization](#visualization)
9. [Performance](#performance)
10. [Configuration](#configuration)

---

## Core Modules

### Constants

```python
from geostats.core.constants import (
    EPSILON,                    # Numerical stability (1e-10)
    DEFAULT_MAX_NEIGHBORS,      # Default max neighbors (25)
    DEFAULT_N_REALIZATIONS,     # Default simulations (100)
    get_constants,              # Get all constants
    set_constants_config,       # Set config path
    reload_constants            # Reload constants
)
```

### Validators

```python
from geostats.core.validators import (
    validate_coordinates,       # Validate coordinate arrays
    validate_values,            # Validate value arrays
    validate_positive,         # Ensure positive values
    validate_in_range,          # Check value ranges
    validate_array_shapes_match # Check array compatibility
)
```

### Exceptions

```python
from geostats.core.exceptions import (
    GeoStatsError,             # Base exception
    ValidationError,           # Validation errors
    KrigingError,              # Kriging errors
    VariogramError,            # Variogram errors
    SimulationError            # Simulation errors
)
```

### Logging

```python
from geostats.core.logging_config import (
    setup_logging,             # Configure logging
    get_logger                 # Get logger instance
)
```

---

## Variogram Analysis

### Experimental Variogram

```python
from geostats import variogram

lags, gamma, n_pairs = variogram.experimental_variogram(
    x,                          # X coordinates (array)
    y,                          # Y coordinates (array)
    z,                          # Values (array)
    n_lags=15,                 # Number of lag bins
    maxlag=None,                # Maximum lag distance
    estimator='matheron',       # 'matheron', 'cressie', 'dowd'
    direction=None,             # Direction for directional variogram
    tolerance=22.5,             # Angular tolerance (degrees)
    bandwidth=None              # Bandwidth for directional variogram
)
```

**Returns:**
- `lags`: Lag distances (array)
- `gamma`: Semivariances (array)
- `n_pairs`: Number of pairs per lag (array)

### Variogram Fitting

```python
model = variogram.fit_model(
    'spherical',                # Model type or model instance
    lags,                       # Lag distances
    gamma,                      # Semivariances
    weights=None,               # Weights (typically n_pairs)
    method='weighted_least_squares'  # Fitting method
)
```

**Model Types:**
- `'spherical'`: Spherical model
- `'exponential'`: Exponential model
- `'gaussian'`: Gaussian model
- `'matern'`: Matérn model
- `'cubic'`: Cubic model
- `'stable'`: Stable model
- `'power'`: Power model
- `'linear'`: Linear model
- `'hole_effect'`: Hole-effect model

### Variogram Models

```python
from geostats.models.variogram_models import (
    SphericalModel,
    ExponentialModel,
    GaussianModel,
    MaternModel,
    CubicModel,
    StableModel,
    PowerModel,
    LinearModel,
    HoleEffectModel
)

# Create model
model = SphericalModel(
    nugget=0.1,                 # Nugget effect
    sill=1.0,                   # Sill
    range_param=30.0            # Range
)

# Evaluate model
gamma = model(h)                # h: distance array

# Get parameters
params = model.parameters        # Dictionary of parameters
```

---

## Kriging Methods

### Ordinary Kriging

```python
from geostats import kriging

ok = kriging.OrdinaryKriging(
    x,                          # X coordinates
    y,                          # Y coordinates
    z,                          # Values
    variogram_model=None,       # Fitted variogram model
    max_neighbors=25,           # Maximum neighbors
    search_radius=None,         # Search radius
    min_neighbors=1             # Minimum neighbors
)

# Predict
predictions, variances = ok.predict(
    x_pred,                     # Prediction X coordinates
    y_pred,                     # Prediction Y coordinates
    return_variance=True        # Return variance
)

# Cross-validate
predictions_cv, metrics = ok.cross_validate()
```

### Simple Kriging

```python
from geostats.algorithms.simple_kriging import SimpleKriging

sk = SimpleKriging(
    x, y, z,
    variogram_model=model,
    mean=2.0                    # Known mean
)

predictions, variances = sk.predict(x_pred, y_pred)
```

### Universal Kriging

```python
from geostats.algorithms.universal_kriging import UniversalKriging

uk = UniversalKriging(
    x, y, z,
    variogram_model=model,
    drift_terms='linear'         # 'linear' or 'quadratic'
)

predictions, variances = uk.predict(x_pred, y_pred)
```

### Indicator Kriging

```python
from geostats.algorithms.indicator_kriging import IndicatorKriging

# Create indicator variable
z_indicator = (z > threshold).astype(float)

ik = IndicatorKriging(
    x, y, z_indicator,
    variogram_model=model
)

probabilities, variances = ik.predict(x_pred, y_pred)
```

### Cokriging

```python
from geostats.algorithms.cokriging import Cokriging

ck = Cokriging(
    x1, y1, z1,                 # Primary variable
    x2, y2, z2,                 # Secondary variable
    variogram_primary=model1,
    variogram_secondary=model2,
    cross_variogram=None        # Optional cross-variogram
)

predictions, variances = ck.predict(x_pred, y_pred)
```

### 3D Kriging

```python
from geostats.algorithms.kriging_3d import Kriging3D

k3d = Kriging3D(
    x, y, z_coords, values,     # 3D coordinates and values
    variogram_model=model
)

predictions, variances = k3d.predict(x_pred, y_pred, z_pred)
```

---

## Simulation

### Sequential Gaussian Simulation

```python
from geostats.simulation.gaussian_simulation import sequential_gaussian_simulation

realizations = sequential_gaussian_simulation(
    x, y, z,                    # Conditioning data
    x_grid, y_grid,             # Grid to simulate on
    variogram_model=model,
    n_realizations=100,          # Number of realizations
    seed=None                    # Random seed
)
```

### Sequential Indicator Simulation

```python
from geostats.simulation.sequential_indicator import sequential_indicator_simulation

realizations = sequential_indicator_simulation(
    x, y, z,
    x_grid, y_grid,
    thresholds=[1.0, 2.0, 3.0], # Threshold values
    variogram_model=model,
    n_realizations=100
)
```

### Unconditional Simulation

```python
from geostats.simulation.unconditional import unconditional_simulation

realization = unconditional_simulation(
    x_grid, y_grid,
    variogram_model=model,
    method='cholesky'           # 'cholesky' or 'turning_bands'
)
```

---

## Transformations

### Normal Score Transform

```python
from geostats.transformations.normal_score import NormalScoreTransform

nst = NormalScoreTransform()
z_transformed = nst.transform(z)
z_back = nst.back_transform(z_transformed)
```

### Log Transform

```python
from geostats.transformations.log_transform import LogTransform

log_transform = LogTransform()
z_log = log_transform.transform(z)
z_back = log_transform.back_transform(z_log)
```

### Box-Cox Transform

```python
from geostats.transformations.boxcox import BoxCoxTransform

bc = BoxCoxTransform(lambda_param=0.5)
z_transformed = bc.transform(z)
z_back = bc.back_transform(z_transformed)
```

### Declustering

```python
from geostats.transformations.declustering import cell_declustering

weights = cell_declustering(
    x, y, z,
    cell_size=10.0,             # Cell size
    method='polygonal'           # 'polygonal' or 'cell'
)
```

---

## I/O Operations

### Raster I/O

```python
from geostats.io.raster import (
    read_geotiff,
    write_geotiff,
    read_ascii_grid,
    write_ascii_grid
)

# Read GeoTIFF
x, y, z, metadata = read_geotiff(
    'file.tif',
    band=1,
    as_grid=True                # Return as grid or points
)

# Write GeoTIFF
write_geotiff(
    'output.tif',
    x, y, z,
    crs='EPSG:4326',
    nodata=-9999.0
)

# Read ASCII Grid
x, y, z, metadata = read_ascii_grid('file.asc', as_grid=True)

# Write ASCII Grid
write_ascii_grid('output.asc', x, y, z, nodata=-9999.0)
```

### Tabular I/O

```python
from geostats.io.tabular import (
    read_csv_spatial,
    write_csv_spatial,
    read_excel_spatial
)

# Read CSV
x, y, z, extra = read_csv_spatial(
    'data.csv',
    x_column='X',
    y_column='Y',
    z_column='Value'
)

# Write CSV
write_csv_spatial(
    'output.csv',
    x, y, z,
    x_column='X',
    y_column='Y',
    z_column='Value'
)
```

### Format Conversions

```python
from geostats.io.formats import (
    to_dataframe,
    to_geopandas,
    read_netcdf,
    write_netcdf,
    read_geojson,
    write_geojson
)

# To DataFrame
df = to_dataframe(x, y, z)

# To GeoDataFrame
gdf = to_geopandas(x, y, z, crs='EPSG:4326')
```

---

## Validation

### Cross-Validation

```python
from geostats.validation.cross_validation import (
    leave_one_out,
    k_fold_cross_validation,
    spatial_cross_validation
)

# Leave-one-out
predictions, metrics = leave_one_out(kriging_object)

# K-fold
predictions, metrics = k_fold_cross_validation(
    kriging_object,
    k=5
)

# Spatial
predictions, metrics = spatial_cross_validation(
    kriging_object,
    n_folds=5
)
```

### Metrics

```python
from geostats.validation.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared,
    mean_error,
    mean_standardized_error
)

mse = mean_squared_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r_squared(y_true, y_pred)
```

---

## Visualization

### Variogram Plots

```python
from geostats.visualization.variogram_plots import (
    plot_variogram,
    plot_variogram_cloud,
    plot_directional_variogram
)

fig, ax = plot_variogram(x, y, z, variogram_model)
fig, ax = plot_variogram_cloud(x, y, z)
fig = plot_directional_variogram(x, y, z, directions=[0, 45, 90, 135])
```

### Spatial Plots

```python
from geostats.visualization.spatial_plots import (
    plot_prediction_map,
    plot_variance_map,
    plot_uncertainty_map
)

fig, ax = plot_prediction_map(x_grid, y_grid, z_pred, samples=(x, y, z))
fig, ax = plot_variance_map(x_grid, y_grid, variance)
fig, ax = plot_uncertainty_map(x_grid, y_grid, predictions, variance)
```

### Interactive Plots

```python
from geostats.interactive.variogram_plots import interactive_variogram
from geostats.interactive.prediction_maps import interactive_prediction_map

fig = interactive_variogram(x, y, z, variogram_model)
fig.show()

fig = interactive_prediction_map(x_grid, y_grid, z_pred, samples=(x, y, z))
fig.show()
```

---

## Performance

### Parallel Processing

```python
from geostats.performance.parallel import parallel_kriging

predictions, variances = parallel_kriging(
    kriging_object,
    x_pred, y_pred,
    n_jobs=-1                   # Use all cores
)
```

### Caching

```python
from geostats.performance.caching import CachedKriging

cached_ok = CachedKriging(kriging_object)
predictions, variances = cached_ok.predict(x_pred, y_pred, use_cache=True)
```

### Chunked Processing

```python
from geostats.performance.chunked import ChunkedKriging

chunked_ok = ChunkedKriging(kriging_object, chunk_size=1000)
predictions, variances = chunked_ok.predict(x_pred, y_pred)
```

---

## Configuration

### Load Configuration

```python
from geostats.config import (
    load_config,
    validate_config,
    load_config_dict
)

# From file
config = load_config('analysis.yaml')

# From dictionary
config = load_config_dict({'project': {'name': 'Test'}})

# Validate
is_valid, message = validate_config('analysis.yaml')
```

### Run Pipeline

```python
from geostats.workflows import AnalysisPipeline

pipeline = AnalysisPipeline(config)
pipeline.run()
```

---

## AutoML

### Auto Variogram

```python
from geostats.automl.auto_variogram import auto_fit_variogram

result = auto_fit_variogram(x, y, z)
model = result['model']
```

### Auto Method

```python
from geostats.automl.auto_method import auto_select_method

result = auto_select_method(x, y, z)
method = result['method']
```

### Auto Interpolate

```python
from geostats.automl.auto_interpolate import auto_interpolate

predictions, variances = auto_interpolate(x, y, z, x_pred, y_pred)
```

---

For more details on specific functions, see the docstrings or run `help(function_name)` in Python.
