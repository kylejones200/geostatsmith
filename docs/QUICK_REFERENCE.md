# GeoStats Library - Quick Reference Card

## 🆕 New Advanced Features

### 1. Normal Score Transform
```python
from geostats.transformations import NormalScoreTransform

# Transform to normal distribution
nst = NormalScoreTransform()
normal_scores = nst.fit_transform(skewed_data)

# Back-transform
original = nst.inverse_transform(normal_scores)
```

### 2. External Drift Kriging
```python
from geostats.algorithms import ExternalDriftKriging

# Kriging with external covariate (e.g., elevation)
edk = ExternalDriftKriging(x, y, temperature, 
                           covariates_data=elevation,
                           variogram_model=model)
pred, var = edk.predict(x_new, y_new, 
                        covariates_new=elevation_new)
```

### 3. Neighborhood Search
```python
from geostats.algorithms import NeighborhoodSearch, NeighborhoodConfig

# Configure search with octant distribution
config = NeighborhoodConfig(
    max_neighbors=25,
    search_radius=50,
    use_octants=True,
    max_per_octant=3
)

ns = NeighborhoodSearch(x, y, config)
indices, distances = ns.find_neighbors(x0, y0)
```

### 4. Nested Variogram
```python
from geostats.algorithms import fit_nested_variogram

# Fit multi-scale variogram (e.g., nugget + short + long range)
nested = fit_nested_variogram(
    lags, semivariance,
    n_structures=2,
    model_types=['spherical', 'exponential']
)

# Evaluate
gamma = nested(h)  # At distance h
print(nested)  # Shows nugget + structures
```

### 5. Declustering
```python
from geostats.transformations import cell_declustering

# Correct for clustered sampling
weights, info = cell_declustering(x, y, z)

# Calculate unbiased mean
unbiased_mean = np.average(z, weights=weights)
print(f"Bias correction: {info['mean_difference']}")
```

### 6. Lognormal Kriging
```python
from geostats.algorithms import LognormalKriging

# For lognormally distributed data (e.g., ore grades)
lnk = LognormalKriging(x, y, z_lognormal, 
                       variogram_model=model,
                       kriging_type='ordinary')

# Predict with proper back-transform
pred = lnk.predict(x_new, y_new, 
                   back_transform_method='unbiased')
```

### 7. 3D Kriging
```python
from geostats.algorithms import OrdinaryKriging3D

# Full 3D spatial interpolation
ok3d = OrdinaryKriging3D(x, y, z_coord, values, 
                         variogram_model_3d)

# Predict in 3D space
pred, var = ok3d.predict(x_new, y_new, z_new)
```

### 8. Sequential Indicator Simulation
```python
from geostats.simulation import SequentialIndicatorSimulation, SISConfig

# Non-parametric simulation
config = SISConfig(
    n_realizations=100,
    n_thresholds=5,
    random_seed=42
)

sis = SequentialIndicatorSimulation(x, y, z, config)
sis.fit_indicator_variograms(variogram_models)

# Generate realizations
realizations = sis.simulate(x_grid, y_grid)

# Get statistics
stats = sis.get_statistics(realizations)
e_type = stats['e_type']  # Mean
uncertainty = stats['uncertainty']  # Std
```

### 9. Block Kriging (Support Change)
```python
from geostats.algorithms import BlockKriging

# Estimate block averages (e.g., mining blocks)
bk = BlockKriging(x, y, z, 
                  variogram_model=model,
                  block_size=(10.0, 10.0),  # 10x10 blocks
                  n_disc=5)  # Discretization

# Predict block values
pred_block, var_block = bk.predict(x_block_centers, 
                                   y_block_centers)

# Note: var_block << var_point (variance reduction)
```

### 10. Octant/Quadrant Search
```python
# Integrated in NeighborhoodConfig

# Octant search (8 sectors, 45° each)
config_octant = NeighborhoodConfig(
    use_octants=True,
    max_per_octant=2  # Max 2 per sector = 16 total
)

# Quadrant search (4 sectors, 90° each)
config_quadrant = NeighborhoodConfig(
    use_quadrants=True,
    max_per_quadrant=6  # Max 6 per sector = 24 total
)
```

---

## Classic Features (Already Available)

### Variogram Analysis
```python
from geostats.algorithms import experimental_variogram, fit_variogram_model

# Calculate experimental variogram
lags, gamma = experimental_variogram(x, y, z, n_lags=15)

# Fit model
model = fit_variogram_model(lags, gamma, model_type='spherical')
```

### Ordinary Kriging
```python
from geostats.algorithms import OrdinaryKriging

ok = OrdinaryKriging(x, y, z, variogram_model=model)
pred, var = ok.predict(x_new, y_new)
```

### Sequential Gaussian Simulation
```python
from geostats.simulation import SequentialGaussianSimulation

sgs = SequentialGaussianSimulation(x, y, z, variogram_model=model)
realizations = sgs.simulate(x_grid, y_grid, n_realizations=100)
```

### Indicator Kriging
```python
from geostats.algorithms import IndicatorKriging

ik = IndicatorKriging(x, y, z, threshold=cutoff, 
                      variogram_model=model)
prob, var = ik.predict(x_new, y_new)
```

### Cokriging
```python
from geostats.algorithms import Cokriging

ck = Cokriging(x, y, primary=z1, secondary=z2,
               variogram_models=[model1, model2, cross_model])
pred = ck.predict(x_new, y_new)
```

---

## Complete Workflow Example

```python
import numpy as np
from geostats.transformations import NormalScoreTransform, cell_declustering
from geostats.algorithms import (
    experimental_variogram,
    fit_nested_variogram,
    OrdinaryKriging,
    NeighborhoodSearch,
    NeighborhoodConfig
)

# 1. Load data
x, y, z = load_sample_data()

# 2. Check for clustering
from geostats.transformations.declustering import detect_clustering
cluster_stats = detect_clustering(x, y)
if cluster_stats['is_likely_clustered']:
    # Apply declustering
    weights, _ = cell_declustering(x, y, z)
    z_mean = np.average(z, weights=weights)
else:
    z_mean = np.mean(z)

# 3. Check distribution
from scipy import stats
_, p_value = stats.normaltest(z)
if p_value < 0.05:
    # Apply normal score transform
    nst = NormalScoreTransform()
    z_transformed = nst.fit_transform(z)
else:
    z_transformed = z

# 4. Variogram analysis
lags, gamma = experimental_variogram(x, y, z_transformed, n_lags=15)

# 5. Fit nested variogram for multi-scale structure
nested_model = fit_nested_variogram(lags, gamma, n_structures=2)

# 6. Set up neighborhood search for efficiency
config = NeighborhoodConfig(
    max_neighbors=25,
    use_octants=True,
    max_per_octant=3
)

# 7. Ordinary Kriging
ok = OrdinaryKriging(x, y, z_transformed, variogram_model=nested_model)

# 8. Predict
x_grid = np.linspace(x.min(), x.max(), 50)
y_grid = np.linspace(y.min(), y.max(), 50)
xx, yy = np.meshgrid(x_grid, y_grid)
pred, var = ok.predict(xx.flatten(), yy.flatten())

# 9. Back-transform if needed
if p_value < 0.05:
    pred = nst.inverse_transform(pred)

# 10. Reshape and plot
pred = pred.reshape(xx.shape)
```

---

## Testing Your Installation

```python
# Test all new features
from geostats.transformations import (
    NormalScoreTransform, 
    cell_declustering
)
from geostats.algorithms import (
    ExternalDriftKriging,
    LognormalKriging,
    SimpleKriging3D,
    OrdinaryKriging3D,
    BlockKriging,
    NeighborhoodSearch,
    fit_nested_variogram
)
from geostats.simulation import SequentialIndicatorSimulation

print("✅ All imports successful!")

# Run comprehensive example
import subprocess
subprocess.run(["python", "examples/example_7_advanced_features.py"])
```

---

## Documentation

- **Comprehensive Guide**: `TOP_10_FEATURES_SUMMARY.md`
- **Implementation Details**: `IMPLEMENTATION_COMPLETE.md`
- **Visual Checklist**: `FEATURES_CHECKLIST.txt`
- **Examples**: `examples/example_7_advanced_features.py`
- **API Reference**: See docstrings in source files

---

## Common Use Cases

### Mining & Resource Estimation
```python
# Block kriging for ore grade estimation
bk = BlockKriging(x, y, z, model, block_size=(10, 10))
block_grades, block_var = bk.predict(block_centers_x, block_centers_y)
```

### Environmental Monitoring
```python
# Lognormal kriging for contaminant concentrations
lnk = LognormalKriging(x, y, concentrations, model)
pred = lnk.predict(x_new, y_new, back_transform_method='unbiased')
```

### Hydrogeology
```python
# 3D kriging for aquifer properties
ok3d = OrdinaryKriging3D(x, y, depth, permeability, model_3d)
perm_3d, var_3d = ok3d.predict(x_grid, y_grid, depth_grid)
```

### Large Datasets
```python
# Efficient neighborhood search
config = NeighborhoodConfig(max_neighbors=25, use_octants=True)
ns = NeighborhoodSearch(x, y, config)
# Use in kriging loop
```

---

## Tips & Best Practices

1. **Always check data distribution** before choosing kriging method
2. **Use declustering** for preferentially sampled data
3. **Apply normal score transform** for SGS and skewed data
4. **Fit nested variograms** when multi-scale structure is evident
5. **Use neighborhood search** for datasets > 1000 points
6. **Choose lognormal kriging** for positively skewed mining/environmental data
7. **Use block kriging** when estimating area averages (not point values)
8. **Apply 3D methods** when vertical correlation is important
9. **Use SIS** for non-Gaussian or categorical variables
10. **Always cross-validate** your kriging model

---

## Performance Optimization

```python
# For large datasets (>10,000 points)
config = NeighborhoodConfig(
    max_neighbors=25,           # Limit search
    search_radius=max_range*3,  # Limit by distance
    use_octants=True,           # Balanced distribution
    max_per_octant=3            # Control per sector
)

# For block kriging, adjust discretization
bk = BlockKriging(..., n_disc=5)  # Lower = faster, less accurate

# For simulation, reduce realizations
sgs = SequentialGaussianSimulation(...)
realizations = sgs.simulate(..., n_realizations=50)  # Instead of 100+
```

---

## Support & References

**Implementation based on:**
- Olea (2009) - A Practical Primer on Geostatistics (USGS)
- Deutsch & Journel (1998) - GSLIB
- Chilès & Delfiner (2012) - Geostatistics: Modeling Spatial Uncertainty
- Wackernagel (2003) - Multivariate Geostatistics

**All PDF references cited in source code docstrings**
