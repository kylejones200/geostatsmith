# GeoStats Demos - Real Alaska Data Analysis

## Overview

These demos use the GeoStats library with data from the Alaska Geochemical Database (AGDB4). Each demo is a workflow that can be run with geochemical data.

## Prerequisites

### 1. Install GeoStats

```bash
cd /Users/k.jones/Desktop/geostats
pip install -e ".[all]" # Install with all optional dependencies
```

### 2. Download Alaska AGDB4 Database

The demos use the Alaska Geochemical Database:
- **Source**: USGS Data Release
- **DOI**: https://doi.org/10.5066/F7445KBJ
- **Size**: ~400MB (text files)
- **Samples**: 375,000+ geochemical analyses

Download and extract to: `/Users/k.jones/Downloads/AGDB4_text`

## The Demos

### Demo 1: Gold Rush Alaska

**File**: `demo_01_gold_exploration.py`

**What it does**:
- Loads real gold data from Fairbanks district
- Anisotropy detection (directional variograms)
- Compares 3 kriging methods (Ordinary, Lognormal, Indicator)
- Uncertainty quantification (Bootstrap + variance)
- Optimal sampling design (where to drill next?)
- Performance comparison
- Validation

**Features showcased**:
- Multiple kriging algorithms
- Directional variograms
- Bootstrap confidence intervals
- Probability mapping
- Infill sampling optimization
- Parallel processing
- Cross-validation

**Outputs**:
1. `alaska_gold_anisotropy.png` - 4 directional variograms
2. `alaska_gold_methods_comparison.png` - OK vs Lognormal vs Indicator
3. `alaska_gold_uncertainty.png` - 3 uncertainty maps
4. `alaska_gold_sampling_design.png` - Optimal new sample locations

**Run time**: ~2-5 minutes

**Output**: Gold anomaly maps for the mining district.

---

### Demo 2: Multi-Element Arsenal

**File**: `demo_02_multi_element_cokriging.py`

**What it does**:
- Analyzes Cu-Mo-Au associations (porphyry signature)
- Element correlation analysis
- Geochemical anomaly detection (3 methods)
- Cokriging vs Ordinary Kriging comparison
- Porphyry fertility index calculation

**Features showcased**:
- Multi-element geochemistry
- Correlation analysis
- Cokriging (use correlation to improve predictions)
- Outlier detection (IQR, Z-score, spatial)
- Composite indices
- Advanced visualization

**Outputs**:
1. `alaska_element_correlations.png` - Cu-Mo-Au scatter plots
2. `alaska_anomaly_detection.png` - Multi-element anomaly maps
3. `alaska_cokriging_comparison.png` - Variance reduction demo
4. `alaska_porphyry_index.png` - Integrated fertility map

**Run time**: ~3-6 minutes

**Output**: Comparison of cokriging and ordinary kriging results.

---

### Demo 3: Environmental Guardian

**File**: `demo_03_environmental_assessment.py`

**What it does**:
- Analyzes As, Pb, Hg contamination
- Compares to EPA regulatory thresholds
- Probability of exceedance mapping
- Multi-threshold risk classification
- Hotspot identification
- HTML report generation

**Features showcased**:
- Environmental thresholds (EPA standards)
- Indicator kriging
- Probability mapping
- Multi-threshold classification
- Risk assessment
- Reporting

**Outputs**:
1. `alaska_threshold_analysis.png` - Regulatory comparison
2. `alaska_exceedance_probability.png` - Probability maps (3 elements)
3. `alaska_as_risk_classification.png` - Multi-level risk map
4. `alaska_contamination_hotspots.png` - Priority locations
5. `alaska_environmental_report.html` - HTML report

**Run time**: ~4-7 minutes

---

## Quick Start

### Run All Demos

```bash
cd /Users/k.jones/Desktop/geostats/examples

# Demo 1: Gold exploration
python demo_01_gold_exploration.py

# Demo 2: Multi-element
python demo_02_multi_element_cokriging.py

# Demo 3: Environmental
python demo_03_environmental_assessment.py
```

### Customize for Your Region

Each demo can focus on specific Alaska regions:

```python
# Edit the demo file to change region:

# Fairbanks (gold mining)
data = load_fairbanks_gold_data(AGDB_PATH)

# Iliamna (Pebble deposit - Cu-Mo-Au)
data = load_multi_element_data(AGDB_PATH, region='Iliamna')

# Juneau (gold belt)
data = load_fairbanks_gold_data(AGDB_PATH, region='Juneau')
```

## What Makes These Demos Special?

### 1. Real Data
- **Not synthetic!** Uses actual USGS database
- 375,000+ samples from Alaska
- Published, peer-reviewed data source
- Real geologic complexity

### 2. Complete Workflows
- Full workflows, not code snippets
- Error handling
- Visualization outputs
- Structured outputs

### 3. Multiple Applications
- **Exploration**: Find mineral deposits
- **Environmental**: Assess contamination
- **Research**: Test methods, validate algorithms

### 4. Library Features
Each demo demonstrates different GeoStats capabilities:
- Kriging methods
- Uncertainty quantification
- Performance optimization
- Interactive visualization
- Reporting

### 5. Output Quality
- High-resolution figures (150 DPI)
- Structured formatting
- Axis labels, colorbars, legends
- Suitable for reports/presentations

## Expected Results

### Demo 1 Output Example:
```
 GOLD RUSH ALASKA - EXPLORATION WORKFLOW

 Loading Alaska gold data...
 Total samples in database: 375,279
 Gold analyses: 2,858,374
 Fairbanks district samples: 8,347

 Gold Statistics:
 Mean: 0.023 ppm
 Median: 0.005 ppm
 Max: 89.5 ppm
 >0.1 ppm: 687 samples (8.2%)
 >1.0 ppm: 23 samples (economic grade!)

 Directional Variogram Analysis (Anisotropy)...
 Anisotropy detected! Range ratio: 1.8
 Consider using anisotropic kriging

 Comparing Kriging Methods...
 Ordinary Kriging (log-transformed)... Time: 2.3s
 Lognormal Kriging (handles skewness)... Time: 2.5s
 Indicator Kriging (probability >0.1 ppm)... Time: 1.8s

 Performance Showcase...
 Standard Kriging: 2.30s
 Parallel Kriging: 0.35s
 Parallel processing enabled

 Model Validation...
 RMSE: 0.234
 R²: 0.82
 Overall Quality Score: 85/100
```

### Visual Output

All demos generate figures:
- High resolution (150 DPI)
- Standard colormaps
- Clear labeling
- Sample locations shown
- Contour lines for important thresholds
- Multiple panels for comparison

## Troubleshooting

### "AGDB4 not found"
```bash
# Check path:
ls /Users/k.jones/Downloads/AGDB4_text

# Should contain:
# Geol_DeDuped.txt, Chem_A_Br.txt, etc.
```

### "Module not found"
```bash
# Install all dependencies:
pip install -e ".[all]"

# Or individually:
pip install numpy scipy pandas matplotlib scikit-learn
pip install plotly fastapi rasterio geopandas # Optional
```

### "Not enough data"
```python
# Some regions have sparse data
# Try a different region or expand search area
data = load_data(AGDB_PATH, region=None) # Use all Alaska
```

### Performance Issues
```python
# Reduce grid resolution:
x_grid = np.linspace(x.min(), x.max(), 50) # Instead of 100

# Use fewer bootstrap iterations:
bootstrap_confidence_intervals(..., n_bootstrap=50) # Instead of 100
```

## Advanced Usage

### Add Your Own Analysis

```python
# All demos are modular - add custom analysis:

def my_custom_analysis(data):
 """Your custom geochemical analysis"""
 x, y, z = data['x'], data['y'], data['values']

 # Your GeoStats code here...
 from geostats.algorithms.universal_kriging import UniversalKriging

 uk = UniversalKriging(x, y, z, trend='linear')
 predictions = uk.predict(x_new, y_new)

 return predictions

# Add to demo:
results = my_custom_analysis(data)
```

### Export Results

```python
# Save predictions to GeoTIFF:
from geostats.io import write_geotiff

write_geotiff(
 'alaska_gold_predictions.tif',
 predictions,
 extent=(x.min(), x.max(), y.min(), y.max()),
 crs='EPSG:4326'
)

# Save to CSV:
output = pd.DataFrame({
 'longitude': X.flatten(),
 'latitude': Y.flatten(),
 'prediction': Z.flatten(),
 'variance': variance.flatten()
})
output.to_csv('alaska_predictions.csv', index=False)
```

### Interactive Maps

```python
# Create interactive web maps:
from geostats.interactive import interactive_prediction_map

fig = interactive_prediction_map(X, Y, predictions)
fig.write_html('alaska_interactive.html')
# Open in browser!
```

## Performance Notes

**Timing on typical laptop**:
- Demo 1: ~3 minutes (8,000 samples, 80x80 grid)
- Demo 2: ~5 minutes (multi-element merging)
- Demo 3: ~6 minutes (3 elements analyzed)

**Speed up tips**:
1. Use parallel processing (`n_jobs=-1`)
2. Reduce grid resolution
3. Focus on specific regions
4. Cache results for repeated runs

## Scientific References

These demos implement methods from:
- Goovaerts, P. (1997). "Geostatistics for Natural Resources Evaluation"
- Cressie, N. (1993). "Statistics for Spatial Data"
- Webster & Oliver (2007). "Geostatistics for Environmental Scientists"

AGDB4 Database:
- Granitto, M., et al. (2019). "Alaska Geochemical Database (AGDB)—Geochemical data for rock, sediment, soil, mineral, and concentrate samples" USGS Data Release. https://doi.org/10.5066/F7445KBJ

## Next Steps

1. **Run the demos** with real Alaska data
2. **Customize** for your region of interest
3. **Combine** methods for your specific application
4. Use results as needed

## Support

- **For demo issues**: Check this README
- **For GeoStats bugs**: Open issue on GitHub
- **For AGDB4 questions**: Contact USGS Alaska Science Center
- **For geochemical interpretation**: Consult a professional geochemist

---

**Built with GeoStats v0.3.0** - geostatistics for Python

*These demos demonstrate spatial analysis workflows using GeoStats.*
