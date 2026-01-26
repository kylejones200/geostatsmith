# Config-Driven GeoStats - Quick Reference Card

## Installation
```bash
cd /Users/k.jones/Desktop/geostats
pip install -e .
```

## CLI Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `geostats-init` | Create new project | `geostats-init my_project --template basic` |
| `geostats-validate` | Validate config | `geostats-validate config.yaml` |
| `geostats-run` | Run analysis | `geostats-run config.yaml --verbose` |
| `geostats-templates` | List templates | `geostats-templates` |

## Quick Start (30 seconds)
```bash
geostats-init quicktest          # Create config
nano quicktest.yaml              # Edit data paths
geostats-run quicktest.yaml      # Run!
```

## Available Templates

| Template | Use Case | Key Features |
|----------|----------|--------------|
| `basic` | Quick analysis | Minimal config, auto everything |
| `advanced` | Complete workflow | All preprocessing, anisotropy |
| `gold_exploration` | Mineral exploration | Log transform, octant search |

## Config Structure (YAML)

```yaml
project:
  name: "My Analysis"
  output_dir: "./results"

data:
  input_file: "data.csv"
  x_column: "X"
  y_column: "Y"
  z_column: "Value"

preprocessing:
  remove_outliers: true
  transform: "log"        # log, boxcox, normal_score, sqrt
  declustering: false

variogram:
  n_lags: 15
  estimator: "matheron"   # matheron, cressie, dowd, madogram
  models: ["spherical", "exponential", "gaussian"]
  auto_fit: true
  check_anisotropy: false

kriging:
  method: "ordinary"      # ordinary, simple, universal
  neighborhood:
    max_neighbors: 25
    min_neighbors: 3
  grid:
    resolution: 1.0

validation:
  cross_validation: true
  cv_method: "loo"        # loo, kfold, spatial
  metrics: ["rmse", "mae", "r2"]

visualization:
  style: "minimalist"
  plots: ["variogram", "kriging_map", "cross_validation"]

output:
  save_predictions: true
  save_variance: true
  save_report: true
  formats: ["npy", "csv"]
```

## Common Customizations

### Change Grid Resolution
```yaml
kriging:
  grid:
    resolution: 0.5     # Finer grid
```

### Add Outlier Removal
```yaml
preprocessing:
  remove_outliers: true
  outlier_method: "iqr"  # or zscore, isolation_forest
  outlier_threshold: 3.0
```

### Enable Anisotropy Check
```yaml
variogram:
  check_anisotropy: true
  anisotropy_angles: [0, 45, 90, 135]
```

### Change Kriging Method
```yaml
kriging:
  method: "universal"    # or simple, indicator
  drift_terms: "linear"  # for universal
```

### Add More Plots
```yaml
visualization:
  plots:
    - "variogram"
    - "variogram_cloud"
    - "directional_variograms"
    - "kriging_map"
    - "variance_map"
    - "cross_validation"
    - "histogram"
    - "qq_plot"
```

## Validation Errors

Common errors and fixes:

| Error | Cause | Fix |
|-------|-------|-----|
| `Input file not found` | Wrong path | Check `data.input_file` |
| `n_lags must be >= 5` | Too few lags | Increase `variogram.n_lags` |
| `max < min neighbors` | Invalid constraint | Fix `neighborhood` settings |
| `cokriging requires z_secondary` | Missing data | Add `data.z_secondary` |

## Programmatic Usage

```python
from geostats.config import load_config
from geostats.workflows import AnalysisPipeline

# Load config
config = load_config('analysis.yaml')

# Modify if needed
config.kriging.grid.resolution = 0.5
config.preprocessing.transform = 'boxcox'

# Run
pipeline = AnalysisPipeline(config)
pipeline.run()

# Access results
predictions = pipeline.predictions
variance = pipeline.variance
cv_results = pipeline.cv_results
```

## Config Merging

```python
from geostats.config import load_config, merge_configs

base = load_config('base.yaml')
overrides = {
    'project': {'name': 'Modified'},
    'kriging': {'grid': {'resolution': 0.5}}
}
config = merge_configs(base, overrides)
```

## Batch Processing

### Parameter Sweep
```bash
for res in 0.5 1.0 2.0; do
  sed "s/resolution: 1.0/resolution: $res/" base.yaml > run_${res}.yaml
  geostats-run run_${res}.yaml
done
```

### Multiple Datasets
```bash
for file in data/*.csv; do
  name=$(basename "$file" .csv)
  sed "s|INPUT|$file|" template.yaml > ${name}.yaml
  geostats-run ${name}.yaml
done
```

## Output Structure

After running, find results in `output_dir/`:
```
results/
├── predictions.npy          # Kriging predictions
├── predictions.csv          # CSV format
├── variance.npy             # Kriging variance
├── variogram_model.pkl      # Fitted model
├── cv_predictions.csv       # Cross-validation results
├── analysis_report.txt      # Text report
└── plots/
    ├── variogram.png
    ├── kriging_map.png
    ├── variance_map.png
    └── cross_validation.png
```

## Config Validation Checklist

Before running:
- [ ] Input file exists
- [ ] Column names match data
- [ ] `n_lags >= 5`
- [ ] `max_neighbors >= min_neighbors`
- [ ] Output directory writable
- [ ] Cokriging has `z_secondary` (if used)
- [ ] Indicator has `thresholds` (if used)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Config won't validate | Run `geostats-validate config.yaml` for details |
| Pipeline crashes | Add `--verbose` flag, check log file |
| Import errors | Run `pip install -e .` again |
| Slow performance | Set `kriging.parallel: true` |
| Memory issues | Reduce `grid.resolution` or use chunking |

## Best Practices

1. **Version control**: `git add *.yaml && git commit`
2. **Start simple**: Use `basic` template first
3. **Validate first**: Always run `geostats-validate`
4. **Document**: Add comments to YAML explaining choices
5. **Use templates**: Reuse successful configs
6. **Test small**: Start with subset of data
7. **Backup configs**: Keep master copies

## Examples

### Minimal Working Example
```yaml
project: {name: "Test", output_dir: "./results"}
data: {input_file: "data.csv", x_column: "X", y_column: "Y", z_column: "Z"}
```

### Production Example
```yaml
project:
  name: "Gold Grade Estimation - Q1 2026"
  output_dir: "./results/gold_q1_2026"
  author: "Exploration Team"

data:
  input_file: "../data/drill_assays_2026q1.csv"
  x_column: "Easting"
  y_column: "Northing"
  z_column: "Au_ppm"
  filter_column: "QC_Status"
  filter_value: "Passed"

preprocessing:
  remove_outliers: true
  outlier_method: "iqr"
  outlier_threshold: 3.0
  transform: "log"
  handle_negatives: "shift"
  declustering: true
  declustering_method: "cell"

variogram:
  n_lags: 20
  estimator: "cressie"
  models: ["spherical", "exponential"]
  auto_fit: true
  check_anisotropy: true
  anisotropy_angles: [0, 30, 60, 90, 120, 150]

kriging:
  method: "ordinary"
  neighborhood:
    max_neighbors: 25
    min_neighbors: 8
    use_octant_search: true
  grid:
    resolution: 10.0
    buffer: 50.0

validation:
  cross_validation: true
  cv_method: "loo"
  metrics: ["rmse", "mae", "r2", "bias"]
  save_predictions: true

visualization:
  style: "minimalist"
  plots: ["variogram", "directional_variograms", "kriging_map", "variance_map"]
  colormap: "YlOrRd"

output:
  save_predictions: true
  save_variance: true
  save_report: true
  formats: ["npy", "csv", "geotiff"]

random_seed: 42
verbose: true
log_file: "./results/gold_q1_2026/analysis.log"
```

## Documentation Links

- **Complete Guide**: `docs/CONFIG_DRIVEN.md`
- **Template Guide**: `examples/configs/README.md`
- **Architecture**: `CONFIG_ARCHITECTURE.md`
- **Traditional API**: `docs/QUICKSTART.md`
- **Plotting Style**: `PLOTTING_STYLE_GUIDE.md`

## Support

- **Examples**: `examples/configs/`
- **Tests**: `tests/test_config.py`
- **Issues**: GitHub Issues
- **Docs**: https://geostats.readthedocs.io

---

**Print this card for quick reference while working!**
