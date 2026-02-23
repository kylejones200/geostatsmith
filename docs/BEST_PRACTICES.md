# Best Practices for GeoStats

This guide covers best practices for using GeoStats effectively and avoiding common pitfalls.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Variogram Analysis](#variogram-analysis)
3. [Kriging Interpolation](#kriging-interpolation)
4. [Performance Optimization](#performance-optimization)
5. [Validation and Quality Control](#validation-and-quality-control)
6. [Common Pitfalls](#common-pitfalls)

---

## Data Preparation

### 1. Data Quality Checks

**Always validate your data before analysis:**

```python
from geostats.core.validators import validate_coordinates, validate_values

# Validate coordinates
x, y, z = validate_coordinates(x, y, z)

# Validate values
z = validate_values(z, allow_nan=False)

# Check for duplicates
from scipy.spatial.distance import cdist
distances = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
duplicates = np.sum(distances < 1e-6) - len(x)  # Subtract diagonal
if duplicates > 0:
    print(f"Warning: {duplicates} duplicate locations found")
```

### 2. Outlier Detection

**Detect and handle outliers before variogram analysis:**

```python
from geostats.diagnostics.outlier_detection import outlier_analysis

# Detect outliers
outliers = outlier_analysis(x, y, z)

# Review outliers
print(f"Found {len(outliers['outlier_indices'])} outliers")
print(f"Indices: {outliers['outlier_indices']}")

# Option 1: Remove outliers
mask = np.ones(len(z), dtype=bool)
mask[outliers['outlier_indices']] = False
x_clean, y_clean, z_clean = x[mask], y[mask], z[mask]

# Option 2: Use robust variogram estimator
lags, gamma, n_pairs = variogram.experimental_variogram(
    x, y, z,
    estimator='cressie'  # More robust to outliers
)
```

### 3. Coordinate System

**Ensure consistent coordinate system:**

```python
# Check coordinate ranges
print(f"X range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")

# Check units (should be consistent)
# If mixing meters and kilometers, convert:
# x_km = x_m / 1000

# Check for coordinate system issues
if x.max() - x.min() < 1e-6 or y.max() - y.min() < 1e-6:
    raise ValueError("Coordinates appear to be constant")
```

### 4. Data Transformations

**Transform data if needed for normality:**

```python
from geostats.transformations.normal_score import NormalScoreTransform

# Check distribution
import matplotlib.pyplot as plt
plt.hist(z, bins=30)
plt.title('Data Distribution')
plt.show()

# If skewed, use normal score transform
if np.abs(z.mean() - np.median(z)) / z.std() > 0.5:
    print("Data appears skewed, applying normal score transform")
    nst = NormalScoreTransform()
    z_transformed = nst.transform(z)
    # Work with transformed data, then back-transform predictions
```

---

## Variogram Analysis

### 1. Lag Selection

**Choose appropriate lag parameters:**

```python
# Calculate data extent
x_range = x.max() - x.min()
y_range = y.max() - y.min()
max_distance = np.sqrt(x_range**2 + y_range**2)

# Rule of thumb: maxlag = 1/2 to 1/3 of maximum distance
maxlag = max_distance / 2

# Number of lags: enough to see structure, not too many
n_lags = min(15, int(maxlag / (max_distance / 30)))  # ~30 points per lag

lags, gamma, n_pairs = variogram.experimental_variogram(
    x, y, z,
    n_lags=n_lags,
    maxlag=maxlag
)

# Check: each lag should have sufficient pairs
min_pairs = 30  # Minimum pairs per lag
valid_lags = n_pairs >= min_pairs
if np.sum(valid_lags) < len(lags) * 0.7:
    print("Warning: Many lags have insufficient pairs")
```

### 2. Model Selection

**Try multiple models and compare:**

```python
from geostats.models.variogram_models import (
    SphericalModel, ExponentialModel, GaussianModel, MaternModel
)

models_to_try = {
    'Spherical': SphericalModel(),
    'Exponential': ExponentialModel(),
    'Gaussian': GaussianModel(),
    'Matern': MaternModel()
}

results = {}
for name, model_class in models_to_try.items():
    try:
        fitted = variogram.fit_model(model_class, lags, gamma, weights=n_pairs)
        
        # Calculate fit quality
        predicted = fitted(lags[valid_lags])
        observed = gamma[valid_lags]
        weights = n_pairs[valid_lags]
        
        # Weighted sum of squared errors
        wss = np.sum(weights * (observed - predicted)**2)
        
        # R-squared
        ss_res = np.sum(weights * (observed - predicted)**2)
        ss_tot = np.sum(weights * (observed - np.mean(observed))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        results[name] = {
            'model': fitted,
            'wss': wss,
            'r2': r2,
            'parameters': fitted.parameters
        }
    except Exception as e:
        print(f"Failed to fit {name}: {e}")

# Select best model
best_name = min(results.keys(), key=lambda k: results[k]['wss'])
best_model = results[best_name]['model']

print(f"Best model: {best_name}")
print(f"R²: {results[best_name]['r2']:.3f}")
```

### 3. Visual Inspection

**Always visualize variogram fit:**

```python
import matplotlib.pyplot as plt

h_plot = np.linspace(0, maxlag, 100)
gamma_plot = best_model(h_plot)

plt.figure(figsize=(10, 6))
plt.scatter(lags, gamma, s=n_pairs*2, alpha=0.6, label='Experimental')
plt.plot(h_plot, gamma_plot, 'r-', linewidth=2, label=f'Fitted ({best_name})')
plt.axhline(y=best_model.parameters['sill'], color='g', linestyle='--', 
            label=f"Sill: {best_model.parameters['sill']:.3f}")
plt.axvline(x=best_model.parameters['range'], color='b', linestyle='--',
            label=f"Range: {best_model.parameters['range']:.3f}")
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Variogram Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check residuals
predicted_at_lags = best_model(lags[valid_lags])
residuals = gamma[valid_lags] - predicted_at_lags

plt.figure(figsize=(10, 4))
plt.scatter(lags[valid_lags], residuals, s=n_pairs[valid_lags]*2, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Lag Distance')
plt.ylabel('Residuals')
plt.title('Variogram Fit Residuals')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Kriging Interpolation

### 1. Neighborhood Selection

**Choose appropriate neighborhood parameters:**

```python
# Rule of thumb: search radius = 2-3 × variogram range
range_param = best_model.parameters['range']
search_radius = 2.5 * range_param

# Maximum neighbors: balance accuracy vs speed
# More neighbors = more accurate but slower
# Fewer neighbors = faster but potentially less accurate
max_neighbors = min(25, len(x) // 2)  # Use at most half of data

ok = kriging.OrdinaryKriging(
    x, y, z,
    variogram_model=best_model,
    max_neighbors=max_neighbors,
    search_radius=search_radius
)
```

### 2. Grid Resolution

**Choose appropriate grid resolution:**

```python
# Rule of thumb: resolution = range / 10 to range / 20
range_param = best_model.parameters['range']
resolution = range_param / 15  # Good balance

# But also consider:
# - Data density: finer resolution if you have many samples
# - Computational cost: coarser resolution for large areas
# - Application needs: match resolution to your use case

x_grid = np.arange(x.min(), x.max() + resolution, resolution)
y_grid = np.arange(y.min(), y.max() + resolution, resolution)
```

### 3. Handling Edge Effects

**Be aware of edge effects near data boundaries:**

```python
# Predictions near edges have higher uncertainty
predictions, variances = ok.predict(x_grid, y_grid, return_variance=True)

# Identify edge regions (high variance)
variance_threshold = np.percentile(variances, 75)  # Top 25%
edge_mask = variances > variance_threshold

# Option 1: Mask edge predictions
predictions_masked = np.where(edge_mask, np.nan, predictions)

# Option 2: Use larger search radius near edges
# (requires custom implementation)

# Option 3: Accept higher uncertainty at edges
# (document in your report)
```

---

## Performance Optimization

### 1. Parallel Processing

**Use parallel processing for large datasets:**

```python
from geostats.performance.parallel import parallel_kriging

# For large prediction grids
predictions, variances = parallel_kriging(
    ok,
    x_grid, y_grid,
    n_jobs=-1  # Use all available cores
)
```

### 2. Caching

**Cache results for repeated predictions:**

```python
from geostats.performance.caching import CachedKriging

# Create cached kriging object
cached_ok = CachedKriging(ok)

# First prediction (computes and caches)
pred1, var1 = cached_ok.predict(x_grid, y_grid, use_cache=True)

# Subsequent predictions (uses cache if same grid)
pred2, var2 = cached_ok.predict(x_grid, y_grid, use_cache=True)
```

### 3. Chunked Processing

**Process large grids in chunks:**

```python
from geostats.performance.chunked import ChunkedKriging

chunked_ok = ChunkedKriging(ok, chunk_size=1000)

# Process in chunks automatically
predictions, variances = chunked_ok.predict(x_grid, y_grid)
```

### 4. Approximate Methods

**Use approximate methods for very large datasets:**

```python
from geostats.performance.approximate import approximate_kriging

# KNN-based approximate kriging (faster)
predictions, variances = approximate_kriging(
    ok,
    x_grid, y_grid,
    n_neighbors=15  # Use fewer neighbors for speed
)
```

---

## Validation and Quality Control

### 1. Cross-Validation

**Always perform cross-validation:**

```python
from geostats.validation.cross_validation import leave_one_out

# Leave-one-out cross-validation
predictions_cv, metrics = leave_one_out(ok)

print(f"Cross-Validation Metrics:")
print(f"  RMSE: {metrics['rmse']:.3f}")
print(f"  MAE: {metrics['mae']:.3f}")
print(f"  R²: {metrics['r2']:.3f}")
print(f"  Mean Error: {metrics['mean_error']:.3f}")

# Check for bias
if abs(metrics['mean_error']) > 0.1 * metrics['rmse']:
    print("Warning: Significant bias detected")
```

### 2. Spatial Validation

**Check for spatial patterns in residuals:**

```python
# Calculate residuals
residuals = z - predictions_cv

# Plot residuals spatially
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x, y, c=residuals, cmap='RdBu_r', s=100, edgecolors='k')
plt.colorbar(scatter, label='Residuals')
plt.title('Spatial Distribution of Residuals')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Check for spatial autocorrelation in residuals
from geostats.spatial_stats.spatial_autocorrelation import morans_i

I = morans_i(x, y, residuals)
print(f"Moran's I for residuals: {I:.3f}")

if abs(I) > 0.2:
    print("Warning: Residuals show spatial autocorrelation")
    print("  This suggests the model may not capture all spatial structure")
```

### 3. Uncertainty Validation

**Validate uncertainty estimates:**

```python
# Check if kriging variance is reasonable
# (should be higher where data is sparse)

# Calculate distance to nearest sample
from scipy.spatial.distance import cdist

distances_to_samples = cdist(
    np.column_stack([x_grid, y_grid]),
    np.column_stack([x, y])
)
min_distances = distances_to_samples.min(axis=1)

# Variance should correlate with distance
correlation = np.corrcoef(min_distances, variances)[0, 1]
print(f"Correlation between distance and variance: {correlation:.3f}")

if correlation < 0.3:
    print("Warning: Variance doesn't correlate well with distance")
    print("  This may indicate issues with variogram model")
```

---

## Common Pitfalls

### 1. Ignoring Trends

**Problem**: Using Ordinary Kriging when data has a trend

**Solution**: Use Universal Kriging or detrend first

```python
# Check for trends
from scipy import stats

# Test for linear trend in X
slope_x, intercept_x, r_x, p_x, _ = stats.linregress(x, z)
# Test for linear trend in Y
slope_y, intercept_y, r_y, p_y, _ = stats.linregress(y, z)

if p_x < 0.05 or p_y < 0.05:
    print("Significant trend detected, use Universal Kriging")
    from geostats.algorithms.universal_kriging import UniversalKriging
    uk = UniversalKriging(x, y, z, variogram_model=model, drift_terms='linear')
else:
    ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)
```

### 2. Overfitting Variogram

**Problem**: Using too many parameters or complex models

**Solution**: Prefer simpler models, validate fit

```python
# Prefer simpler models (spherical, exponential) over complex ones
# unless you have strong evidence for complexity

# Use cross-validation to compare models
models_to_compare = ['spherical', 'exponential', 'gaussian']
cv_results = {}

for model_name in models_to_compare:
    model = variogram.fit_model(model_name, lags, gamma, weights=n_pairs)
    ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)
    _, metrics = leave_one_out(ok)
    cv_results[model_name] = metrics['rmse']

best_model_name = min(cv_results.keys(), key=lambda k: cv_results[k])
print(f"Best model by CV: {best_model_name} (RMSE: {cv_results[best_model_name]:.3f})")
```

### 3. Extrapolation

**Problem**: Predicting far from data

**Solution**: Limit predictions to data extent or use larger uncertainty

```python
# Define prediction extent (buffer around data)
buffer = best_model.parameters['range']  # Use range as buffer

x_min_pred = x.min() - buffer
x_max_pred = x.max() + buffer
y_min_pred = y.min() - buffer
y_max_pred = y.max() + buffer

# Only predict within this extent
mask = (x_grid >= x_min_pred) & (x_grid <= x_max_pred) & \
       (y_grid >= y_min_pred) & (y_grid <= y_max_pred)

predictions_masked = np.where(mask, predictions, np.nan)
```

### 4. Ignoring Uncertainty

**Problem**: Only using predictions, ignoring variance

**Solution**: Always consider uncertainty in decisions

```python
# Use uncertainty in decision-making
# Example: Risk-averse decision (use lower confidence bound)
confidence_level = 0.05  # 5th percentile
z_lower = predictions - 1.96 * np.sqrt(variances)

# Or use probability maps
from geostats.uncertainty.probability import probability_map
prob_exceed = probability_map(predictions, variances, threshold=2.0)
```

---

## Summary

1. **Always validate data** before analysis
2. **Try multiple variogram models** and compare
3. **Perform cross-validation** to assess performance
4. **Consider uncertainty** in your decisions
5. **Use appropriate methods** for your data (trends, multiple variables, etc.)
6. **Optimize performance** for large datasets
7. **Document your choices** and assumptions

For more details, see the [User Guide](USER_GUIDE.md) and [Tutorials](TUTORIALS.md).
