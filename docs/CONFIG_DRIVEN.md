# Config-Driven Geostatistical Analysis

## Overview

The `geostats` library now supports **fully config-driven analysis** using YAML or JSON configuration files. This enables:

- **Reproducible workflows**: All parameters in version-controlled config files
- **Batch processing**: Run multiple analyses with different configs
- **Parameter exploration**: Easy parameter sweeps and sensitivity analysis
- **Collaboration**: Share analyses via config files
- **Documentation**: Self-documenting analysis parameters

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

This will install `pydantic`, `pyyaml`, and `click` for config-driven analysis.

### 2. Create a Project

```bash
geostats-init my_analysis --template basic
```

This creates `my_analysis.yaml` with a template configuration.

### 3. Edit Configuration

Open `my_analysis.yaml` and update the data paths and parameters:

```yaml
project:
  name: "My Analysis"
  output_dir: "./results/my_analysis"

data:
  input_file: "my_data.csv"
  x_column: "X"
  y_column: "Y"
  z_column: "Value"

kriging:
  method: "ordinary"
  grid:
    resolution: 1.0
```

### 4. Run Analysis

```bash
geostats-run my_analysis.yaml
```

Results are saved to `./results/my_analysis/`.

## Command-Line Interface

### `geostats-run`

Run a complete geostatistical analysis from a config file.

```bash
# Basic usage
geostats-run analysis.yaml

# Validate only (don't run)
geostats-run analysis.yaml --validate-only

# Verbose output
geostats-run analysis.yaml --verbose

# Override config values
geostats-run analysis.yaml -o project.name="Test Run"
```

### `geostats-validate`

Validate a configuration file without running.

```bash
geostats-validate analysis.yaml
```

### `geostats-init`

Initialize a new project with a template configuration.

```bash
# Create basic config
geostats-init my_project

# Create from template
geostats-init gold_analysis --template gold_exploration

# Save to specific directory
geostats-init my_project --output-dir ./configs
```

Available templates:
- `basic`: Minimal configuration for quick analysis
- `advanced`: Complete workflow with all features
- `gold_exploration`: Mineral exploration template

### `geostats-templates`

List available configuration templates.

```bash
geostats-templates
```

## Configuration Structure

A complete configuration has the following sections:

### Project Metadata

```yaml
project:
  name: "Analysis Name"
  output_dir: "./results"
  description: "Optional description"
  author: "Your Name"
```

### Data Loading

```yaml
data:
  input_file: "data.csv"
  x_column: "X"
  y_column: "Y"
  z_column: "Value"
  z_secondary: "Secondary"  # For cokriging
  filter_column: "Type"      # Optional filtering
  filter_value: "Surface"
```

### Preprocessing

```yaml
preprocessing:
  # Outlier removal
  remove_outliers: true
  outlier_method: "iqr"      # iqr, zscore, isolation_forest
  outlier_threshold: 3.0
  
  # Data transformation
  transform: "log"           # log, boxcox, normal_score, sqrt, null
  handle_negatives: "shift"  # shift, remove, absolute
  
  # Declustering
  declustering: true
  declustering_method: "cell"  # cell, polygonal
```

### Variogram Modeling

```yaml
variogram:
  # Experimental variogram
  n_lags: 15
  max_lag: null              # Auto if null
  estimator: "matheron"      # matheron, cressie, dowd, madogram
  
  # Model fitting
  models: ["spherical", "exponential", "gaussian"]
  auto_fit: true
  fit_method: "wls"          # ols, wls
  fit_criterion: "rmse"      # rmse, mae, r2, aic
  
  # Anisotropy
  check_anisotropy: true
  anisotropy_angles: [0, 45, 90, 135]
```

### Kriging Configuration

```yaml
kriging:
  method: "ordinary"  # ordinary, simple, universal, indicator, cokriging
  
  neighborhood:
    max_neighbors: 25
    min_neighbors: 3
    search_radius: null      # Auto if null
    use_octant_search: false
  
  grid:
    resolution: 1.0
    buffer: 0.0
    # Or specify grid size:
    # nx: 100
    # ny: 100
  
  return_variance: true
  parallel: false
```

### Validation

```yaml
validation:
  cross_validation: true
  cv_method: "loo"           # loo, kfold, spatial
  n_folds: 5
  metrics: ["rmse", "mae", "r2"]
  save_predictions: true
```

### Visualization

```yaml
visualization:
  style: "minimalist"        # minimalist, default, seaborn
  
  plots:
    - "variogram"
    - "kriging_map"
    - "variance_map"
    - "cross_validation"
    - "histogram"
  
  dpi: 300
  colormap: "viridis"
  save_format: ["png", "pdf"]
```

### Output

```yaml
output:
  save_predictions: true
  save_variance: true
  save_weights: true
  save_model: true
  save_report: true
  
  formats: ["npy", "csv", "geotiff"]
  compression: true
  precision: "float32"
```

### Advanced Options

```yaml
random_seed: 42            # For reproducibility
verbose: true
log_file: "./analysis.log"
```

## Configuration Validation

All configurations are validated using [Pydantic](https://docs.pydantic.dev/):

- **Type checking**: Ensures correct data types
- **Value constraints**: Validates ranges (e.g., `n_lags >= 5`)
- **Cross-field validation**: Checks dependencies (e.g., cokriging requires secondary variable)
- **Helpful error messages**: Clear descriptions of what's wrong

Example validation error:

```
Configuration validation failed:
  • data -> input_file: Input file not found: data.csv
  • variogram -> n_lags: Input should be greater than or equal to 5
  • kriging -> neighborhood -> max_neighbors: Input should be greater than or equal to min_neighbors
```

## Programmatic Usage

You can also use the config system programmatically:

```python
from geostats.config import load_config
from geostats.workflows import AnalysisPipeline

# Load config
config = load_config('analysis.yaml')

# Modify programmatically
config.kriging.method = 'universal'
config.visualization.plots.append('residuals')

# Run pipeline
pipeline = AnalysisPipeline(config)
pipeline.run()
```

### Creating Configs in Code

```python
from geostats.config import AnalysisConfig, load_config_dict

config_dict = {
    'project': {'name': 'Test', 'output_dir': './results'},
    'data': {
        'input_file': 'data.csv',
        'x_column': 'X',
        'y_column': 'Y',
        'z_column': 'Value'
    },
    # ... other sections with defaults
}

config = load_config_dict(config_dict)
```

### Merging Configs

```python
from geostats.config import load_config, merge_configs

base = load_config('base.yaml')
overrides = {
    'project': {'name': 'Modified'},
    'kriging': {'grid': {'resolution': 0.5}}
}

config = merge_configs(base, overrides)
```

## Examples

### Example 1: Basic Analysis

```yaml
# basic_analysis.yaml
project:
  name: "Quick Kriging"
  output_dir: "./results"

data:
  input_file: "data.csv"
  x_column: "X"
  y_column: "Y"
  z_column: "Value"

kriging:
  method: "ordinary"
  grid:
    resolution: 1.0

visualization:
  plots: ["variogram", "kriging_map"]
```

Run: `geostats-run basic_analysis.yaml`

### Example 2: Gold Exploration

```yaml
# gold_exploration.yaml
project:
  name: "Gold Grade Estimation"
  output_dir: "./results/gold"

data:
  input_file: "assays.csv"
  x_column: "Easting"
  y_column: "Northing"
  z_column: "Au_ppm"

preprocessing:
  remove_outliers: true
  transform: "log"
  declustering: true

variogram:
  estimator: "cressie"       # Robust
  check_anisotropy: true     # Important for mineralization

kriging:
  method: "ordinary"
  neighborhood:
    use_octant_search: true  # Better for deposits

visualization:
  colormap: "YlOrRd"
  plots:
    - "variogram"
    - "directional_variograms"
    - "kriging_map"
    - "variance_map"
```

Run: `geostats-run gold_exploration.yaml`

### Example 3: Parameter Sweep

Create multiple configs with different parameters:

```bash
# Generate configs
for resolution in 0.5 1.0 2.0; do
  sed "s/resolution: 1.0/resolution: $resolution/" base.yaml > analysis_res_${resolution}.yaml
done

# Run all
for config in analysis_res_*.yaml; do
  geostats-run $config
done
```

## Pipeline Architecture

The config-driven pipeline follows this workflow:

```
1. Load Config (YAML/JSON)
   ↓
2. Validate (Pydantic schemas)
   ↓
3. Load Data
   ↓
4. Preprocess
   - Outlier removal
   - Transformation
   - Declustering
   ↓
5. Variogram Modeling
   - Experimental variogram
   - Model fitting
   - Anisotropy check
   ↓
6. Kriging
   - Grid creation
   - Prediction
   - Variance estimation
   ↓
7. Validation
   - Cross-validation
   - Metric computation
   ↓
8. Visualization
   - Plot generation
   - Style application
   ↓
9. Output
   - Save predictions
   - Save report
   - Export formats
```

## Best Practices

### 1. Use Version Control

Keep configs in Git:

```bash
git add analysis.yaml
git commit -m "Add kriging analysis config"
```

### 2. Template Reuse

Create base templates for common workflows:

```yaml
# base_template.yaml
project:
  output_dir: "./results/${ANALYSIS_NAME}"

variogram:
  n_lags: 15
  models: ["spherical", "exponential"]
```

Then extend:

```yaml
# specific_analysis.yaml
extends: base_template.yaml

project:
  name: "Specific Analysis"

data:
  input_file: "specific_data.csv"
```

### 3. Document Configs

Add comments explaining choices:

```yaml
variogram:
  n_lags: 20  # Increased for better resolution at small lags
  estimator: "cressie"  # Robust to outliers in this dataset
```

### 4. Validate Before Long Runs

```bash
# Quick validation
geostats-validate analysis.yaml

# Then run
geostats-run analysis.yaml
```

### 5. Use Descriptive Names

```yaml
project:
  name: "Alaska_Gold_2024Q1_Preliminary"
  output_dir: "./results/alaska_gold_2024q1_preliminary"
```

## Troubleshooting

### Config Validation Fails

```bash
# Get detailed error messages
geostats-validate analysis.yaml
```

Common issues:
- File paths: Ensure `input_file` exists and is correct
- Constraints: Check value ranges (e.g., `n_lags >= 5`)
- Dependencies: Cokriging needs `z_secondary`, indicator needs `thresholds`

### Pipeline Fails

```bash
# Run with verbose output
geostats-run analysis.yaml --verbose

# Check log file
cat ./results/analysis.log
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

## Next Steps

- See `examples/configs/` for complete examples
- Check `docs/QUICKSTART.md` for traditional API usage
- Read `PLOTTING_STYLE_GUIDE.md` for visualization customization
- Review `MATHEMATICAL_REVIEW.md` for algorithm details

## Support

- GitHub Issues: https://github.com/kylejones200/geostats/issues
- Documentation: https://geostats.readthedocs.io
