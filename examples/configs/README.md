# Configuration Templates

This directory contains YAML configuration templates for common geostatistical workflows.

## Available Templates

### `basic_template.yaml`

**Use Case**: Quick exploratory analysis with minimal configuration

**Features**:
- Ordinary kriging with auto-fitted variogram
- Default preprocessing (no outlier removal, no transformation)
- Basic cross-validation
- Standard visualizations (variogram, kriging map, CV)

**Best For**:
- Initial data exploration
- Quick interpolation needs
- Learning the config system

**Example**:
```bash
geostats-init quick_test --template basic
geostats-run quick_test.yaml
```

---

### `advanced_template.yaml`

**Use Case**: Complete geostatistical workflow with all features

**Features**:
- Full preprocessing pipeline (outliers, Box-Cox transform, declustering)
- Anisotropy analysis with directional variograms
- Multiple model comparison
- Comprehensive validation metrics
- All plot types
- Multiple output formats (NPY, CSV, GeoTIFF)

**Best For**:
- Production analyses
- Research projects
- Complex spatial patterns
- Publication-quality results

**Example**:
```bash
geostats-init full_analysis --template advanced
# Edit full_analysis.yaml with your data
geostats-run full_analysis.yaml
```

---

### `gold_exploration_template.yaml`

**Use Case**: Mineral exploration and resource estimation

**Features**:
- Log transformation (typical for metal grades)
- Declustering for clustered drill holes
- Cressie-Hawkins estimator (robust to outliers)
- Octant search for directional deposits
- Anisotropy analysis (important for mineralization direction)
- Kriging variance for resource classification
- Grade-appropriate color maps (YlOrRd)

**Best For**:
- Gold, copper, silver exploration
- Resource/reserve estimation
- Grade control
- Deposit modeling

**Example**:
```bash
geostats-init gold_prospect --template gold_exploration
# Edit gold_prospect.yaml with drill data
geostats-run gold_prospect.yaml
```

**Notes**:
- Adjust `grid.resolution` to match drill spacing / 2
- Set `neighborhood.min_neighbors` based on data density
- Use `variance_map` output for Measured/Indicated/Inferred classification

---

## Example: Alaska Gold Analysis

The `alaska_gold_example.yaml` demonstrates a real-world analysis using the Alaska Geochemical Database (AGDB4).

**To run**:
```bash
# Ensure AGDB4 data is available
cd /Users/k.jones/Desktop/geostats
geostats-run examples/configs/alaska_gold_example.yaml
```

**Features**:
- Filters gold (Au) geochemistry data
- Log transformation for positively skewed data
- Outlier removal (IQR method)
- Anisotropy check for regional trends
- Creates maps suitable for exploration targeting

---

## Customizing Templates

### 1. Copy Template

```bash
cp examples/configs/basic_template.yaml my_custom.yaml
```

### 2. Edit Parameters

Open `my_custom.yaml` and modify:

```yaml
data:
  input_file: "my_data.csv"  # Your data file
  x_column: "Easting"        # Your X column
  y_column: "Northing"       # Your Y column
  z_column: "Grade"          # Your value column

kriging:
  grid:
    resolution: 10.0         # Match your data spacing
```

### 3. Validate

```bash
geostats-validate my_custom.yaml
```

### 4. Run

```bash
geostats-run my_custom.yaml
```

---

## Common Customizations

### Change Variogram Model

```yaml
variogram:
  models: ["spherical"]      # Force spherical model
  auto_fit: false
  manual_model: "spherical"
  manual_range: 100.0
  manual_sill: 1.5
  manual_nugget: 0.1
```

### Grid Resolution

```yaml
kriging:
  grid:
    resolution: 0.5          # Finer grid
    # OR
    nx: 200                  # Explicit grid size
    ny: 150
```

### Transformation Options

```yaml
preprocessing:
  # Log transform
  transform: "log"
  
  # Box-Cox (auto-finds best lambda)
  transform: "boxcox"
  
  # Normal score (for non-Gaussian data)
  transform: "normal_score"
  
  # Square root
  transform: "sqrt"
  
  # None
  transform: null
```

### Kriging Methods

```yaml
kriging:
  # Ordinary kriging (most common)
  method: "ordinary"
  
  # Simple kriging (known mean)
  method: "simple"
  mean: 10.5
  
  # Universal kriging (with trend)
  method: "universal"
  drift_terms: "linear"      # or "quadratic"
```

### Cross-Validation Methods

```yaml
validation:
  # Leave-one-out (most thorough)
  cv_method: "loo"
  
  # K-fold (faster)
  cv_method: "kfold"
  n_folds: 10
  
  # Spatial CV (for spatially correlated errors)
  cv_method: "spatial"
```

---

## Template Inheritance (Future)

*Note: Template inheritance is planned for a future release*

```yaml
# my_analysis.yaml
extends: basic_template.yaml

# Only override what changes
data:
  input_file: "my_data.csv"

kriging:
  grid:
    resolution: 0.5
```

---

## Best Practices

### 1. Start Simple

Begin with `basic_template.yaml`, then add complexity as needed.

### 2. Document Choices

```yaml
variogram:
  estimator: "cressie"       # Comment: Robust to outliers
  n_lags: 20                 # Comment: Fine lag spacing for short-range structure
```

### 3. Version Control

```bash
git add configs/*.yaml
git commit -m "Add analysis configurations"
```

### 4. Descriptive Names

Use names that describe the analysis:
- `gold_preliminary_2024q1.yaml` ✓
- `test.yaml` ✗

### 5. Test First

```bash
# Validate before running
geostats-validate my_config.yaml

# Test on subset if dataset is large
# (add filter in config)
data:
  filter_column: "Year"
  filter_value: 2024
```

---

## See Also

- **Full Documentation**: `docs/CONFIG_DRIVEN.md`
- **Traditional API**: `docs/QUICKSTART.md`
- **Plotting Styles**: `PLOTTING_STYLE_GUIDE.md`
- **Mathematical Details**: `MATHEMATICAL_REVIEW.md`
