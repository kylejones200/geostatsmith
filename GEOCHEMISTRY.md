# GeoStats for Geochemistry

## Overview

The GeoStats library is **ideally suited for geochemical data analysis**. Geostatistics and geochemistry have a long history together - kriging was originally developed for ore grade estimation in mining, and modern geochemistry relies heavily on spatial statistics.

## Why GeoStats for Geochemistry?

### Core Capabilities

✅ **Multi-element Analysis** - Cokriging for correlated elements  
✅ **Compositional Data** - Log-ratio transformations for constrained data  
✅ **Probability Mapping** - Exceedance probability for contamination thresholds  
✅ **Anomaly Detection** - Outlier detection and robust variograms  
✅ **Sampling Optimization** - Design optimal geochemical surveys  
✅ **Uncertainty Quantification** - Bootstrap and simulation for confidence intervals  
✅ **3D Modeling** - Subsurface element distribution  
✅ **Data Fusion** - Integrate geophysics, geology, and geochemistry  

### Common Geochemistry Applications

1. **Mineral Exploration**
   - Ore grade estimation (Au, Cu, Ag, etc.)
   - Resource modeling and reserve estimation
   - Pathfinder element mapping
   - Geochemical anomaly detection

2. **Environmental Geochemistry**
   - Soil contamination mapping (Pb, As, Hg, etc.)
   - Risk assessment and remediation planning
   - Background vs. anthropogenic sources
   - Exceedance probability for regulatory limits

3. **Petroleum Geochemistry**
   - Source rock quality mapping
   - Maturity indices (TOC, Ro, S1/S2)
   - Basin-scale geochemical modeling

4. **Marine Geochemistry**
   - Sediment composition mapping
   - Nutrient distribution
   - Trace element cycling

5. **Agricultural Geochemistry**
   - Soil nutrient mapping (N, P, K, pH)
   - Micronutrient deficiency identification
   - Precision agriculture applications

## Key Features for Geochemistry

### 1. Multivariate Analysis (Cokriging)

Handle correlated elements together for improved estimation:

```python
from geostats.algorithms.cokriging import Cokriging

# Example: Cu and Mo (often correlated in porphyry deposits)
cokriging = Cokriging(
    x_primary=x, y_primary=y, z_primary=cu_values,
    x_secondary=x, y_secondary=y, z_secondary=mo_values,
    primary_variogram=cu_model,
    secondary_variogram=mo_model,
    cross_variogram=cu_mo_model
)

cu_pred, variance = cokriging.predict(x_new, y_new)
```

### 2. Compositional Data Analysis

Properly handle closed compositional data (percentages that sum to 100%):

```python
from geostats.transformations import log_ratio_transform, inverse_log_ratio

# For major elements (SiO2, Al2O3, etc.) or mineralogy
# Use centered log-ratio (CLR) or additive log-ratio (ALR)
data_clr = log_ratio_transform(compositions, method='clr')

# Perform kriging on transformed data
kriging = OrdinaryKriging(x, y, data_clr, model)
predictions_clr = kriging.predict(x_new, y_new)

# Back-transform to compositional space
predictions = inverse_log_ratio(predictions_clr, method='clr')
```

### 3. Probability Mapping for Contamination

Calculate probability of exceeding regulatory thresholds:

```python
from geostats.uncertainty import probability_of_exceedance

# Example: Pb contamination risk assessment
# EPA residential soil screening level for Pb = 400 mg/kg
threshold = 400  # mg/kg

prob_map = probability_of_exceedance(
    x_samples, y_samples, pb_samples,
    x_grid, y_grid,
    threshold=threshold,
    n_realizations=100,
    model=pb_variogram
)

# prob_map now contains probability that Pb > 400 mg/kg at each location
# Values > 0.95 indicate high-risk areas requiring remediation
```

### 4. Lognormal Data (Common in Geochemistry)

Many geochemical elements follow lognormal distributions:

```python
from geostats.algorithms.lognormal_kriging import LognormalKriging

# For highly skewed data (Au, Pt, rare earth elements)
# Lognormal kriging handles skewness and back-transformation bias
lk = LognormalKriging(x, y, au_values, variogram_model=model)
au_pred, variance = lk.predict(x_new, y_new, return_variance=True)

# Predictions are in original units, bias-corrected
```

### 5. Indicator Kriging for Multi-class Mapping

Create probability maps for geochemical provinces or contamination classes:

```python
from geostats.algorithms.indicator_kriging import IndicatorKriging

# Example: Map probability of "high", "medium", "low" Cu zones
thresholds = [50, 200]  # ppm Cu cutoffs

ik = IndicatorKriging(
    x, y, cu_values,
    threshold=thresholds[0],  # For first threshold
    variogram_model=indicator_model
)

prob_high_cu = ik.predict(x_grid, y_grid)
```

### 6. 3D Subsurface Modeling

Model element distribution with depth:

```python
from geostats.algorithms.kriging_3d import Kriging3D

# Borehole data with x, y, z coordinates and element values
kriging_3d = Kriging3D(
    x_samples, y_samples, z_samples, element_values,
    variogram_model=model_3d
)

# Predict on 3D grid
element_3d = kriging_3d.predict(x_grid, y_grid, z_grid)
```

### 7. Optimal Sampling Design

Design cost-effective geochemical surveys:

```python
from geostats.optimization import optimal_sampling, infill_sampling

# Initial survey design
optimal_locations = optimal_sampling(
    x_bounds=(0, 1000),
    y_bounds=(0, 1000),
    n_samples=50,
    method='space_filling'  # or 'variance_reduction', 'hybrid'
)

# After initial results, identify where to collect additional samples
infill_locations = infill_sampling(
    x_existing, y_existing, z_existing,
    variogram_model=model,
    n_new_samples=10,
    bounds=(0, 1000, 0, 1000)
)
```

### 8. Outlier Detection (Anomalies)

Distinguish true anomalies from errors:

```python
from geostats.diagnostics.outlier_detection import detect_outliers

# Detect geochemical anomalies
outlier_results = detect_outliers(
    x, y, element_values,
    methods=['iqr', 'zscore', 'spatial'],
    return_details=True
)

# Separate background from anomalous values
background_mask = ~outlier_results['outlier_mask']
anomaly_mask = outlier_results['outlier_mask']

# Use robust variogram for background estimation
from geostats.algorithms.variogram import experimental_variogram
lags, gamma = experimental_variogram(
    x[background_mask], 
    y[background_mask], 
    element_values[background_mask],
    estimator='cressie_hawkins'  # Robust to outliers
)
```

## Complete Geochemical Workflow Example

```python
import numpy as np
import pandas as pd
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.uncertainty import bootstrap_confidence_intervals, probability_of_exceedance
from geostats.diagnostics import comprehensive_validation
from geostats.visualization import plot_variogram, plot_prediction_surface
from geostats.reporting import generate_report

# 1. Load geochemical data
data = pd.read_csv('soil_samples.csv')
x = data['easting'].values
y = data['northing'].values
pb = data['pb_ppm'].values  # Lead concentration

# 2. Exploratory analysis
print(f"Pb statistics: mean={pb.mean():.1f}, median={pb.median():.1f}, max={pb.max():.1f} ppm")
print(f"Samples > 400 ppm (EPA limit): {(pb > 400).sum()}/{len(pb)}")

# 3. Variogram analysis
lags, gamma = experimental_variogram(x, y, pb, maxlag=500)
model = fit_variogram(lags, gamma, model_type='spherical')

print(f"Variogram: range={model['range']:.1f}m, sill={model['sill']:.1f}, nugget={model['nugget']:.1f}")

# 4. Cross-validation
validation_results = comprehensive_validation(x, y, pb, model)
print(f"CV RMSE: {validation_results['cv_rmse']:.2f} ppm")
print(f"Quality score: {validation_results['overall_score']:.0f}/100")

# 5. Kriging prediction
kriging = OrdinaryKriging(x, y, pb, variogram_model=model)

# Create prediction grid
x_grid = np.linspace(x.min(), x.max(), 200)
y_grid = np.linspace(y.min(), y.max(), 200)
X, Y = np.meshgrid(x_grid, y_grid)

pb_pred, pb_var = kriging.predict(X.flatten(), Y.flatten(), return_variance=True)
pb_pred = pb_pred.reshape(X.shape)
pb_var = pb_var.reshape(X.shape)

# 6. Uncertainty quantification
# Bootstrap confidence intervals
ci_lower, ci_upper = bootstrap_confidence_intervals(
    x, y, pb, x_grid, y_grid, model, n_bootstrap=100
)

# Exceedance probability for EPA limit
prob_exceed = probability_of_exceedance(
    x, y, pb, X.flatten(), Y.flatten(),
    threshold=400,  # EPA residential soil screening level
    n_realizations=100,
    model=model
).reshape(X.shape)

# 7. Risk classification
risk = np.zeros_like(prob_exceed)
risk[prob_exceed < 0.05] = 0  # Low risk
risk[(prob_exceed >= 0.05) & (prob_exceed < 0.5)] = 1  # Medium risk
risk[prob_exceed >= 0.5] = 2  # High risk

print(f"\nRisk Assessment:")
print(f"  Low risk: {(risk==0).sum() / risk.size * 100:.1f}% of area")
print(f"  Medium risk: {(risk==1).sum() / risk.size * 100:.1f}% of area")
print(f"  High risk: {(risk==2).sum() / risk.size * 100:.1f}% of area")

# 8. Generate professional report
generate_report(
    x, y, pb,
    output='pb_contamination_report.html',
    title='Lead Contamination Risk Assessment',
    author='Your Name',
    include_cv=True,
    include_variogram=True,
    threshold=400,
    threshold_name='EPA Residential Limit'
)

print("\nReport generated: pb_contamination_report.html")
```

## Geochemistry-Specific Tips

### 1. Check Data Distribution

```python
import matplotlib.pyplot as plt

# Histogram and Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(element_values, bins=30)
ax1.set_xlabel('Element Concentration')
ax1.set_title('Distribution')

# If heavily right-skewed, consider log transform
from scipy import stats
stats.probplot(element_values, dist="lognorm", plot=ax2)
ax2.set_title('Lognormal Q-Q Plot')
```

### 2. Handle Detection Limits

```python
# For censored data (below detection limit)
bdl_mask = element_values < detection_limit

# Option 1: Use indicator kriging at detection limit
# Option 2: Impute as detection_limit / 2
# Option 3: Use specialized methods from geostats.transformations
```

### 3. Anisotropy in Geology

```python
from geostats.algorithms.variogram import directional_variogram

# Check for directional trends (e.g., along strike of ore body)
directions = [0, 45, 90, 135]  # degrees
for angle in directions:
    lags, gamma = directional_variogram(
        x, y, element,
        angle=angle,
        tolerance=22.5
    )
    plt.plot(lags, gamma, label=f'{angle}°')
    
plt.legend()
plt.title('Directional Variograms')
```

### 4. Quality Control

```python
# Duplicate samples for QC
duplicates = data[data['sample_id'].str.contains('DUP')]

# Blank samples for contamination check
blanks = data[data['sample_id'].str.contains('BLANK')]

# Standard reference materials
standards = data[data['sample_id'].str.contains('STD')]
```

## Example Datasets

The library includes geochemistry-friendly synthetic datasets:

```python
from geostats.datasets import generate_random_field

# Simulate geochemical data with spatial structure
x, y, element = generate_random_field(
    n_points=100,
    trend_type='none',  # or 'linear' for regional gradient
    correlation_range=200,  # meters
    nugget_effect=0.2,  # Analytical + micro-scale variation
    noise_level=0.1
)

# Add lognormal distribution (typical for trace elements)
element = np.exp(element)
```

## References for Geochemical Applications

- Goovaerts, P. (1997). "Geostatistics for Natural Resources Evaluation"
- Reimann, C., et al. (2008). "Statistical Data Analysis Explained"
- Pawlowsky-Glahn, V., & Buccianti, A. (2011). "Compositional Data Analysis"
- Webster, R., & Oliver, M. A. (2007). "Geostatistics for Environmental Scientists"

## Getting Help

For geochemistry-specific questions:
1. Check the examples: `examples/workflow_03_uncertainty.py`
2. See the API documentation
3. Open an issue on GitHub
4. The validation suite (`comprehensive_validation`) can help diagnose data quality issues

---

**GeoStats + Geochemistry = Powerful Spatial Analysis** 🗺️⚗️
