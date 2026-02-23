# GeoStats Tutorials

A collection of step-by-step tutorials for common geostatistical tasks.

## Table of Contents

1. [Tutorial 1: Your First Variogram](#tutorial-1-your-first-variogram)
2. [Tutorial 2: Creating Interpolation Maps](#tutorial-2-creating-interpolation-maps)
3. [Tutorial 3: Uncertainty Quantification](#tutorial-3-uncertainty-quantification)
4. [Tutorial 4: Multi-Element Analysis](#tutorial-4-multi-element-analysis)
5. [Tutorial 5: Optimal Sampling Design](#tutorial-5-optimal-sampling-design)
6. [Tutorial 6: Config-Driven Workflows](#tutorial-6-config-driven-workflows)

---

## Tutorial 1: Your First Variogram

**Goal**: Calculate and fit a variogram to understand spatial correlation.

### Step 1: Load Your Data

```python
import numpy as np
import pandas as pd
from geostats import variogram

# Load data (example: CSV file)
data = pd.read_csv('your_data.csv')
x = data['X'].values
y = data['Y'].values
z = data['Value'].values

# Or create synthetic data for testing
np.random.seed(42)
x = np.random.uniform(0, 100, 50)
y = np.random.uniform(0, 100, 50)
z = np.sin(x/20) + np.cos(y/20) + np.random.randn(50) * 0.2
```

### Step 2: Calculate Experimental Variogram

```python
# Calculate experimental variogram
lags, gamma, n_pairs = variogram.experimental_variogram(
    x, y, z,
    n_lags=15,        # Number of lag bins
    maxlag=50,        # Maximum lag distance
    estimator='matheron'  # or 'cressie', 'dowd'
)

print(f"Number of lags: {len(lags)}")
print(f"Lag distances: {lags}")
print(f"Semivariances: {gamma}")
print(f"Number of pairs: {n_pairs}")
```

### Step 3: Visualize Experimental Variogram

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(lags, gamma, s=n_pairs*2, alpha=0.6, label='Experimental')
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Experimental Variogram')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

### Step 4: Fit Theoretical Model

```python
# Fit spherical model
model = variogram.fit_model(
    'spherical',
    lags, gamma,
    weights=n_pairs  # Weight by number of pairs
)

print(f"Fitted parameters: {model.parameters}")
print(f"Nugget: {model.parameters['nugget']:.3f}")
print(f"Sill: {model.parameters['sill']:.3f}")
print(f"Range: {model.parameters['range']:.3f}")
```

### Step 5: Compare Models

```python
from geostats.models.variogram_models import (
    SphericalModel, ExponentialModel, GaussianModel
)

# Try multiple models
models = {
    'Spherical': SphericalModel(),
    'Exponential': ExponentialModel(),
    'Gaussian': GaussianModel()
}

best_model = None
best_score = np.inf

for name, model in models.items():
    fitted = variogram.fit_model(model, lags, gamma, weights=n_pairs)
    
    # Calculate fit quality (weighted sum of squared errors)
    predicted = fitted(lags)
    wss = np.sum(n_pairs * (gamma - predicted)**2)
    
    print(f"{name}: WSS = {wss:.3f}")
    
    if wss < best_score:
        best_score = wss
        best_model = fitted

print(f"\nBest model: {best_model.__class__.__name__}")
```

### Step 6: Visualize Fitted Model

```python
# Plot experimental and fitted
h_plot = np.linspace(0, max(lags), 100)
gamma_plot = best_model(h_plot)

plt.figure(figsize=(10, 6))
plt.scatter(lags, gamma, s=n_pairs*2, alpha=0.6, label='Experimental')
plt.plot(h_plot, gamma_plot, 'r-', linewidth=2, label='Fitted Model')
plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Variogram: Experimental vs Fitted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Tutorial 2: Creating Interpolation Maps

**Goal**: Create a continuous map from point data using kriging.

### Step 1: Prepare Data

```python
import numpy as np
from geostats import variogram, kriging

# Your point data
x = np.random.uniform(0, 100, 50)
y = np.random.uniform(0, 100, 50)
z = np.sin(x/20) + np.cos(y/20) + np.random.randn(50) * 0.2
```

### Step 2: Create Prediction Grid

```python
# Define grid extent and resolution
x_min, x_max = 0, 100
y_min, y_max = 0, 100
resolution = 2.0  # Grid cell size

# Create grid
x_grid = np.arange(x_min, x_max + resolution, resolution)
y_grid = np.arange(y_min, y_max + resolution, resolution)
X, Y = np.meshgrid(x_grid, y_grid)

print(f"Grid size: {X.shape}")
print(f"Number of prediction points: {X.size}")
```

### Step 3: Fit Variogram

```python
# Calculate and fit variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)
model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)
```

### Step 4: Perform Kriging

```python
# Create kriging object
ok = kriging.OrdinaryKriging(
    x, y, z,
    variogram_model=model,
    max_neighbors=25,  # Limit neighborhood size
    search_radius=60  # Search radius
)

# Predict on grid
z_pred, z_var = ok.predict(
    X.flatten(), Y.flatten(),
    return_variance=True
)

# Reshape for plotting
Z_pred = z_pred.reshape(X.shape)
Z_var = z_var.reshape(X.shape)
Z_std = np.sqrt(Z_var)
```

### Step 5: Visualize Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Prediction map
im1 = axes[0].contourf(X, Y, Z_pred, levels=20, cmap='viridis')
axes[0].scatter(x, y, c=z, edgecolors='k', s=50, cmap='viridis')
axes[0].set_title('Kriging Predictions')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0])

# Uncertainty map
im2 = axes[1].contourf(X, Y, Z_std, levels=20, cmap='Reds')
axes[1].scatter(x, y, c='k', s=30, alpha=0.5)
axes[1].set_title('Prediction Uncertainty (Std Dev)')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1])

# Sample locations
axes[2].scatter(x, y, c=z, s=100, cmap='viridis', edgecolors='k')
axes[2].set_title('Sample Locations')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Step 6: Export Results

```python
# Save as GeoTIFF (if rasterio available)
try:
    from geostats.io.raster import write_geotiff
    write_geotiff(
        'prediction_map.tif',
        x_grid, y_grid, Z_pred,
        crs='EPSG:4326'
    )
    print("Saved prediction map to prediction_map.tif")
except ImportError:
    print("rasterio not available, skipping GeoTIFF export")

# Save as CSV
import pandas as pd
results = pd.DataFrame({
    'X': X.flatten(),
    'Y': Y.flatten(),
    'Prediction': z_pred,
    'Variance': z_var,
    'StdDev': np.sqrt(z_var)
})
results.to_csv('kriging_results.csv', index=False)
print("Saved results to kriging_results.csv")
```

---

## Tutorial 3: Uncertainty Quantification

**Goal**: Quantify uncertainty in predictions using multiple methods.

### Method 1: Kriging Variance

```python
from geostats import variogram, kriging
import numpy as np

# Your data and grid
x, y, z = ...  # Your data
x_grid, y_grid = ...  # Your grid

# Fit variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)
model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

# Kriging with variance
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=model)
predictions, variances = ok.predict(x_grid, y_grid, return_variance=True)

# Calculate confidence intervals (95%)
std_dev = np.sqrt(variances)
ci_lower = predictions - 1.96 * std_dev
ci_upper = predictions + 1.96 * std_dev
```

### Method 2: Conditional Simulation

```python
from geostats.simulation.gaussian_simulation import sequential_gaussian_simulation

# Generate multiple realizations
n_realizations = 100
realizations = sequential_gaussian_simulation(
    x, y, z,
    x_grid, y_grid,
    variogram_model=model,
    n_realizations=n_realizations
)

# Calculate statistics
mean_map = np.mean(realizations, axis=0)
std_map = np.std(realizations, axis=0)
p5_map = np.percentile(realizations, 5, axis=0)  # 5th percentile
p95_map = np.percentile(realizations, 95, axis=0)  # 95th percentile
```

### Method 3: Bootstrap

```python
from geostats.uncertainty.bootstrap import bootstrap_uncertainty

# Bootstrap uncertainty
results = bootstrap_uncertainty(
    x, y, z,
    x_grid, y_grid,
    variogram_model=model,
    n_bootstrap=100
)

mean_bootstrap = results['mean']
std_bootstrap = results['std']
ci_lower_bootstrap = results['ci_lower']
ci_upper_bootstrap = results['ci_upper']
```

### Visualize Uncertainty

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Mean prediction
im1 = axes[0, 0].contourf(X, Y, mean_map, levels=20)
axes[0, 0].set_title('Mean Prediction')
plt.colorbar(im1, ax=axes[0, 0])

# Standard deviation
im2 = axes[0, 1].contourf(X, Y, std_map, levels=20, cmap='Reds')
axes[0, 1].set_title('Uncertainty (Std Dev)')
plt.colorbar(im2, ax=axes[0, 1])

# 5th percentile
im3 = axes[1, 0].contourf(X, Y, p5_map, levels=20)
axes[1, 0].set_title('5th Percentile (Conservative)')
plt.colorbar(im3, ax=axes[1, 0])

# 95th percentile
im4 = axes[1, 1].contourf(X, Y, p95_map, levels=20)
axes[1, 1].set_title('95th Percentile (Optimistic)')
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

---

## Tutorial 4: Multi-Element Analysis

**Goal**: Use multiple correlated variables (cokriging).

### Step 1: Prepare Multi-Element Data

```python
import numpy as np

# Primary variable (e.g., Gold)
x1, y1, z1 = ...  # Gold grades

# Secondary variable (e.g., Silver, correlated with Gold)
x2, y2, z2 = ...  # Silver grades

# Check correlation
correlation = np.corrcoef(z1, z2)[0, 1]
print(f"Correlation between variables: {correlation:.3f}")
```

### Step 2: Fit Individual Variograms

```python
from geostats import variogram

# Fit variogram for primary variable
lags1, gamma1, n_pairs1 = variogram.experimental_variogram(x1, y1, z1)
model1 = variogram.fit_model('spherical', lags1, gamma1, weights=n_pairs1)

# Fit variogram for secondary variable
lags2, gamma2, n_pairs2 = variogram.experimental_variogram(x2, y2, z2)
model2 = variogram.fit_model('spherical', lags2, gamma2, weights=n_pairs2)
```

### Step 3: Perform Cokriging

```python
from geostats.algorithms.cokriging import Cokriging

# Create cokriging object
ck = Cokriging(
    x1, y1, z1,  # Primary variable
    x2, y2, z2,  # Secondary variable
    variogram_primary=model1,
    variogram_secondary=model2
)

# Predict primary variable
predictions, variances = ck.predict(x_new, y_new, return_variance=True)
```

### Step 4: Compare with Ordinary Kriging

```python
from geostats import kriging

# Ordinary kriging (using only primary variable)
ok = kriging.OrdinaryKriging(x1, y1, z1, variogram_model=model1)
ok_predictions, ok_variances = ok.predict(x_new, y_new, return_variance=True)

# Compare
print(f"Cokriging RMSE: {np.sqrt(np.mean((z1_test - predictions)**2)):.3f}")
print(f"Ordinary Kriging RMSE: {np.sqrt(np.mean((z1_test - ok_predictions)**2)):.3f}")
print(f"Improvement: {((ok_variances.mean() - variances.mean()) / ok_variances.mean() * 100):.1f}% variance reduction")
```

---

## Tutorial 5: Optimal Sampling Design

**Goal**: Determine optimal locations for additional sampling.

### Step 1: Existing Data

```python
import numpy as np
from geostats import variogram, kriging
from geostats.optimization.sampling_design import optimal_sampling_design

# Your existing samples
x_existing = np.random.uniform(0, 100, 20)
y_existing = np.random.uniform(0, 100, 20)
z_existing = np.sin(x_existing/20) + np.cos(y_existing/20) + np.random.randn(20) * 0.2

# Fit variogram
lags, gamma, n_pairs = variogram.experimental_variogram(x_existing, y_existing, z_existing)
model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)
```

### Step 2: Optimal Sampling Design

```python
# Find optimal locations for 5 additional samples
x_new, y_new = optimal_sampling_design(
    x_existing, y_existing, z_existing,
    n_new_samples=5,
    variogram_model=model,
    strategy='variance_reduction',  # or 'space_filling', 'hybrid'
    n_candidates=1000
)

print(f"Optimal new sample locations:")
for i, (xi, yi) in enumerate(zip(x_new, y_new)):
    print(f"  Sample {i+1}: ({xi:.2f}, {yi:.2f})")
```

### Step 3: Evaluate Improvement

```python
# Create kriging with existing data
ok_before = kriging.OrdinaryKriging(
    x_existing, y_existing, z_existing,
    variogram_model=model
)

# Create prediction grid
x_grid = np.linspace(0, 100, 50)
y_grid = np.linspace(0, 100, 50)
X, Y = np.meshgrid(x_grid, y_grid)

# Predict variance before
_, var_before = ok_before.predict(X.flatten(), Y.flatten(), return_variance=True)

# Simulate new samples (in practice, you'd collect these)
z_new = np.sin(x_new/20) + np.cos(y_new/20) + np.random.randn(5) * 0.2

# Create kriging with new samples
x_all = np.concatenate([x_existing, x_new])
y_all = np.concatenate([y_existing, y_new])
z_all = np.concatenate([z_existing, z_new])

ok_after = kriging.OrdinaryKriging(x_all, y_all, z_all, variogram_model=model)
_, var_after = ok_after.predict(X.flatten(), Y.flatten(), return_variance=True)

# Calculate improvement
improvement = (var_before.mean() - var_after.mean()) / var_before.mean() * 100
print(f"Variance reduction: {improvement:.1f}%")
```

---

## Tutorial 6: Config-Driven Workflows

**Goal**: Run complete workflows from YAML configuration files.

### Step 1: Create Configuration File

Create `my_analysis.yaml`:

```yaml
project:
  name: "My Geostatistical Analysis"
  output_dir: "./results/my_analysis"

data:
  input_file: "data.csv"
  x_column: "X"
  y_column: "Y"
  z_column: "Value"
  filter_column: null
  filter_value: null

variogram:
  n_lags: 15
  maxlag: 50
  estimator: "matheron"
  model: "spherical"
  fit_method: "weighted_least_squares"

kriging:
  method: "ordinary"
  max_neighbors: 25
  search_radius: 60
  grid:
    x_min: 0
    x_max: 100
    y_min: 0
    y_max: 100
    resolution: 2.0

validation:
  cross_validation: true
  k_folds: 5

output:
  save_predictions: true
  save_variance: true
  save_report: true
  format: "geotiff"
```

### Step 2: Run Analysis

```python
from geostats.config import load_config
from geostats.workflows import AnalysisPipeline

# Load configuration
config = load_config('my_analysis.yaml')

# Create and run pipeline
pipeline = AnalysisPipeline(config)
pipeline.run()

print("Analysis complete!")
print(f"Results saved to: {config.project.output_dir}")
```

### Step 3: Command Line

```bash
# Validate config
geostats-validate my_analysis.yaml

# Run analysis
geostats-run my_analysis.yaml

# With overrides
geostats-run my_analysis.yaml --override project.name="Test Run"
```

---

## Next Steps

- Explore more examples in `examples/` directory
- Read the [User Guide](USER_GUIDE.md) for detailed explanations
- Check [API Reference](QUICK_REFERENCE.md) for function details
- See [Best Practices](BEST_PRACTICES.md) for optimization tips
