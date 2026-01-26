# Alaska Geochemical Database (AGDB4) with GeoStats

## Overview

The **Alaska Geochemical Database (AGDB4)** is one of the most comprehensive geochemical datasets available, containing over **375,000 deduplicated samples** from across Alaska. Combined with GeoStats, it provides a powerful platform for:

- 🏔️ **Mineral Exploration** - Gold, copper, silver, REE deposits
- 🌊 **Environmental Assessment** - Arsenic, lead, mercury contamination
- 🗺️ **Geochemical Mapping** - Regional element distribution
- 📊 **Research** - Spatial geochemistry, baseline studies

## Database Contents

### Samples
- **375,279 deduplicated samples** (Geol_DeDuped.txt)
- **416,347 total samples** including duplicates (Geol_AllSpls.txt)
- Sample types: stream sediments, rocks, soils, concentrates

### Chemistry Data
- **70+ elements** analyzed
- **15+ million** individual chemical determinations
- Multiple analytical methods ranked by quality
- "Best Value" tables with optimal method selection

### Spatial Coverage
- All of Alaska
- Lat/Long coordinates for each sample
- Quadrangle-based organization
- Known deposits and mineral districts

### Files Structure

**Location/Geology:**
- `Geol_DeDuped.txt` - Primary sample table (375K samples)
- `Geol_AllSpls.txt` - All samples including duplicates

**Chemistry ("Best Value" Tables):**
- `BV_Ag_Br.txt` - Silver through Bromine
- `BV_C_Gd.txt` - Carbon through Gadolinium  
- `BV_Ge_Os.txt` - Germanium through Osmium
- `BV_P_Te.txt` - Phosphorus through Tellurium
- `BV_Th_Zr.txt` - Thorium through Zirconium
- `BV_WholeRock_Majors.txt` - Major element oxides
- `BV_NonElement.txt` - Non-element parameters

**Raw Chemistry Tables:**
- `Chem_A_Br.txt` - All As, Au, Ag analyses (2.8M records)
- `Chem_C_Gd.txt` - Cu, Co, Cr, etc. (3.4M records)
- `Chem_Ge_Os.txt` - Fe, Hg, K, Mg, etc. (3.4M records)
- `Chem_P_Te.txt` - Pb, Zn, S, etc. (3.1M records)
- `Chem_Th_Zr.txt` - U, V, W, etc. (3.0M records)

**Reference:**
- `DataDictionary.txt` - Complete field descriptions
- `Parameter.txt` - Analytical parameters
- `AnalyticMethod.txt` - Methods used

## Quick Start with GeoStats

### Installation

```bash
# Install GeoStats
cd /path/to/geostats
pip install -e .

# Download AGDB4 from USGS
# https://doi.org/10.5066/F7445KBJ
```

### Basic Example

```python
import pandas as pd
import numpy as np
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram

# Load location data
geol = pd.read_csv('AGDB4_text/Geol_DeDuped.txt', low_memory=False)

# Load gold data
au_data = pd.read_csv('AGDB4_text/Chem_A_Br.txt', low_memory=False)

# Merge and filter
merged = geol.merge(au_data[au_data['PARAMETER'] == 'Au'], on='AGDB_ID')
merged = merged[merged['VALUE'] > 0]  # Remove non-detects

# Extract coordinates and values
x = merged['LONGITUDE'].values
y = merged['LATITUDE'].values
au = merged['VALUE'].values  # in ppm

# Log-transform (gold is lognormal)
au_log = np.log10(au + 1)

# Variogram and kriging
lags, gamma = experimental_variogram(x, y, au_log)
model = fit_variogram(lags, gamma, model_type='spherical')

kriging = OrdinaryKriging(x, y, au_log, variogram_model=model)
predictions = kriging.predict(x_new, y_new)
```

## Complete Example

See `examples/alaska_geochemical_analysis.py` for a comprehensive workflow that includes:

1. **Gold Exploration Analysis**
   - Load and filter AGDB4 data
   - Variogram modeling
   - Kriging interpolation
   - Identify high-potential zones
   - Uncertainty quantification

2. **Multi-element Cokriging** 
   - Analyze correlated elements (Cu-Mo, Pb-Zn)
   - Cross-variogram modeling
   - Improved predictions using secondary variables

3. **Environmental Assessment**
   - Arsenic/lead contamination mapping
   - Probability of exceedance
   - Risk classification (low/medium/high)
   - Regulatory threshold analysis

## Geochemical Applications

### Mineral Exploration

**Gold (Au):**
```python
# Typical threshold: 100 ppb (0.1 ppm) for anomalies
# Economic grade: >1 ppm for placer, >5 ppm for lode
from geostats.uncertainty import probability_of_exceedance

prob_anomaly = probability_of_exceedance(
    x, y, au_values, x_grid, y_grid,
    threshold=0.1,  # 100 ppb
    n_realizations=100
)
```

**Copper (Cu):**
```python
# Background: ~50 ppm in sediments
# Anomalous: >200 ppm
# Economic: >5000 ppm (0.5%)
```

**Pathfinder Elements:**
- Au-As-Sb (epithermal gold)
- Cu-Mo (porphyry)
- Pb-Zn-Ag (VMS deposits)
- Ni-Cr-Co (ultramafic)

### Environmental Geochemistry

**Arsenic (As):**
```python
# EPA drinking water standard: 10 ppb (0.01 ppm)
# Soil screening level: 0.39 ppm (residential)
# Natural background in Alaska: 5-20 ppm in sediments

env_assessment = environmental_assessment(
    agdb_path,
    element='As',
    threshold=20  # ppm
)
```

**Lead (Pb):**
```python
# EPA soil screening level: 400 ppm (residential)
# Background: 10-30 ppm
```

**Mercury (Hg):**
```python
# Naturally elevated in Alaska (placer mining)
# Background: 0.05-0.5 ppm
# Contaminated: >1 ppm
```

## Data Quality Considerations

### Non-Detects
- Negative values represent non-detects in AGDB4
- Filter: `data[data['VALUE'] > 0]`
- Consider indicator kriging for heavily censored data

### Analytical Methods
- "Best Value" tables use ranked methods
- Check `BESTVALUE_RANK` for quality
- Multiple methods available in `_ALL` columns

### Spatial Coverage
- Highly variable sample density
- Dense near known deposits/roads
- Sparse in remote areas
- Consider neighborhood search for kriging

### Detection Limits
- Vary by method and element
- Check metadata for actual limits
- LOD issues for Au (typically 2-10 ppb)

## Tips for Working with AGDB4

### 1. File Size Management

The chemistry files are LARGE (2-3GB each). Loading strategies:

```python
# Use chunksize for large files
chunks = pd.read_csv('Chem_A_Br.txt', chunksize=100000, low_memory=False)

# Filter while reading
for chunk in chunks:
    chunk = chunk[chunk['PARAMETER'] == 'Au']
    # Process chunk...

# Or use only BV (Best Value) tables - much smaller!
bv_data = pd.read_csv('BV_Ag_Br.txt', low_memory=False)
```

### 2. Coordinate Systems

- AGDB4 uses decimal degrees (WGS84)
- Longitude is negative (western hemisphere)
- For large-scale mapping, consider projecting to Alaska Albers (EPSG:3338)

```python
import geopandas as gpd

gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['LONGITUDE'], data['LATITUDE']),
    crs='EPSG:4326'
)

# Project to Alaska Albers
gdf_proj = gdf.to_crs('EPSG:3338')
```

### 3. Regional Analysis

Focus on specific quadrangles to reduce data size:

```python
# Iliamna Quadrangle (contains Pebble deposit)
iliamna = geol[geol['QUAD'] == 'Iliamna']

# Fairbanks District
fairbanks = geol[geol['DISTRICT_NAME'].str.contains('Fairbanks', na=False)]
```

### 4. Sample Type Selection

```python
# Stream sediments only (most common for regional surveys)
stream_sed = geol[geol['PRIMARY_CLASS'] == 'sediment']

# Rock samples
rocks = geol[geol['PRIMARY_CLASS'] == 'rock']

# Heavy mineral concentrates
concentrates = geol[geol['PRIMARY_CLASS'] == 'concentrate']
```

## Example Results

Running `alaska_geochemical_analysis.py`:

```
ALASKA GEOCHEMICAL DATABASE (AGDB4) ANALYSIS
Using GeoStats Library

Gold Statistics:
  Mean: 0.023 ppm
  Median: 0.005 ppm  
  Max: 89.5 ppm
  >100 ppb: 1,247 samples (8.3%)

Variogram Analysis...
  Model: spherical
  Range: 0.52 degrees (~50 km)
  Sill: 0.85

Exploration Targets:
  High potential area: 12.4% of region
  Max predicted Au: 2.3 ppm

Multi-Element Correlation:
  Cu-Mo correlation: 0.67
  Strong correlation detected - good for cokriging!

Environmental Assessment:
  As > 20 ppm: 3,456 samples (9.2%)
  High risk areas: 5.8% of region
```

## Further Reading

### AGDB4 Resources
- USGS Data Release: https://doi.org/10.5066/F7445KBJ
- Documentation: Included with download
- Analytical method rankings: Crock et al. (2018)

### Alaska Geochemistry
- Alaska Division of Geological & Geophysical Surveys (DGGS)
- USGS Mineral Resources Program
- Alaska Geochemical Database publications

### GeoStats Documentation
- See `GEOCHEMISTRY.md` for general geochemical workflows
- `examples/` directory for more examples
- API documentation for detailed method descriptions

## Support

For questions about:
- **AGDB4 data**: Contact USGS Alaska Science Center
- **GeoStats usage**: Open an issue on GitHub
- **Geochemical interpretation**: Consult with a geochemist!

---

**Alaska + GeoStats = World-class Geochemical Analysis** 🗻⚗️
