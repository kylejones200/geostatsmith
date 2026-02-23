# Constants Configuration for Config-Driven Workloads

The geostats library supports **config-driven constants** through YAML files, enabling different constant values for different workloads without code changes.

## Quick Start

### 1. Environment Variable (Easiest)

```bash
export GEOSTATS_CONSTANTS_CONFIG=config/my_constants.yaml
python my_analysis.py
```

All constants automatically load from the config file.

### 2. Programmatic Setup

```python
from geostats.core.constants import set_constants_config

# Set config path before importing constants
set_constants_config('config/my_constants.yaml')

# Now all imports use config values
from geostats.core.constants import EPSILON, DEFAULT_MAX_NEIGHBORS
print(EPSILON)  # Uses value from config
```

### 3. In Analysis Config

```yaml
# analysis.yaml
project:
  name: "My Analysis"

constants:
  epsilon: 1e-12
  default_max_neighbors: 30
```

Then in your code:

```python
from geostats.config import load_config
from geostats.core.constants import set_constants_config

config = load_config('analysis.yaml')
if config.constants:
    # Apply constants from config
    # (Implementation details in constants_loader)
    pass
```

## Architecture

**Hybrid Approach:**
- **Python constants** (`src/geostats/core/constants.py`) serve as defaults
- **YAML configs** can override specific constants
- **Backward compatible**: Code using Python constants directly still works
- **Automatic loading**: Environment variable or programmatic setup

## Usage Patterns

### Pattern 1: Standard (Python Defaults)

```python
from geostats.core.constants import EPSILON, DEFAULT_MAX_NEIGHBORS

# Uses Python defaults
print(EPSILON)  # 1e-10
```

### Pattern 2: Environment Variable

```bash
# Set once, applies to all scripts
export GEOSTATS_CONSTANTS_CONFIG=config/production_constants.yaml
```

```python
# All imports automatically use config
from geostats.core.constants import EPSILON
print(EPSILON)  # Uses config value
```

### Pattern 3: Programmatic (Per-Script)

```python
from geostats.core.constants import set_constants_config, EPSILON

# Set config at start of script
set_constants_config('config/high_precision.yaml')

# All subsequent imports use config
from geostats.core.constants import DEFAULT_MAX_NEIGHBORS
```

### Pattern 4: Explicit Loading

```python
from geostats.core.constants_loader import get_constants, get_constant

# Load specific config
constants = get_constants('config/my_constants.yaml')
EPSILON = constants['EPSILON']

# Or get single constant
EPSILON = get_constant('epsilon', 'config/my_constants.yaml')
```

## YAML Format

Constants in YAML use **lowercase_with_underscores** and are automatically converted to **UPPERCASE**:

```yaml
# YAML (lowercase)
epsilon: 1e-12
default_max_neighbors: 30
default_n_realizations: 200

# Becomes Python constants (uppercase)
EPSILON = 1e-12
DEFAULT_MAX_NEIGHBORS = 30
DEFAULT_N_REALIZATIONS = 200
```

## Available Constants

See `src/geostats/core/constants.py` for the full list. Common overrides:

### Numerical Stability
- `epsilon`: Numerical epsilon (default: 1e-10)
- `small_number`: Small number threshold (default: 1e-6)
- `regularization_factor`: Matrix regularization (default: 1e-8)

### Kriging
- `default_max_neighbors`: Max neighbors (default: 25)
- `default_min_neighbors`: Min neighbors (default: 3)
- `default_search_radius_multiplier`: Search radius multiplier (default: 3.0)

### Variogram
- `default_n_lags`: Number of lag bins (default: 15)
- `default_max_distance`: Max distance (default: 100.0)

### Simulation
- `default_n_realizations`: Number of realizations (default: 100)
- `default_n_thresholds`: Number of thresholds (default: 5)

### Outlier Detection
- `z_score_threshold`: Z-score threshold (default: 3.0)
- `iqr_multiplier`: IQR multiplier (default: 1.5)
- `spatial_threshold_factor`: Spatial threshold (default: 3.0)

### Optimization
- `max_iterations_global`: Max global iterations (default: 500)
- `max_iterations_local`: Max local iterations (default: 1000)
- `convergence_tolerance`: Convergence tolerance (default: 1e-6)

## Example Config Files

### High-Precision Analysis

```yaml
# config/high_precision.yaml
epsilon: 1e-12
convergence_tolerance: 1e-8
max_iterations_global: 2000
default_n_realizations: 500
```

### Fast Exploratory Analysis

```yaml
# config/fast_exploration.yaml
default_n_lags: 10
default_n_realizations: 50
max_iterations_global: 100
default_max_neighbors: 15
```

### Sensitive Outlier Detection

```yaml
# config/sensitive_outliers.yaml
z_score_threshold: 2.5
iqr_multiplier: 1.2
spatial_threshold_factor: 2.0
```

## Reloading Constants

If you need to reload constants (e.g., config file changed):

```python
from geostats.core.constants import reload_constants

# Reload from current config
reload_constants()

# Or reload from new config
reload_constants('config/new_constants.yaml')
```

## Best Practices

1. **Use environment variables for production**: Set once, applies everywhere
2. **Use programmatic setup for scripts**: Explicit control per script
3. **Version control configs**: Track constant changes over time
4. **Document overrides**: Explain why constants differ from defaults
5. **Test with defaults**: Ensure code works with Python defaults
6. **Partial overrides**: Only specify what you need to change

## Migration Guide

**Existing code continues to work:**
```python
# Still works - uses Python defaults
from geostats.core.constants import EPSILON
```

**New config-driven code:**
```python
# Option 1: Environment variable (automatic)
export GEOSTATS_CONSTANTS_CONFIG=config/constants.yaml
# Then use normally:
from geostats.core.constants import EPSILON

# Option 2: Programmatic (explicit)
from geostats.core.constants import set_constants_config
set_constants_config('config/constants.yaml')
from geostats.core.constants import EPSILON
```

## Integration with AnalysisConfig

Constants can be included in your main analysis config:

```yaml
# analysis.yaml
project:
  name: "My Analysis"

data:
  input_file: "data.csv"

constants:
  epsilon: 1e-12
  default_max_neighbors: 30
```

The constants section is automatically available via `config.constants`.

## See Also

- `src/geostats/core/constants.py` - Python default constants
- `src/geostats/core/constants_loader.py` - Constants loader implementation
- `src/geostats/config/constants_config.py` - Pydantic schema for constants
- `examples/configs/constants_example.yaml` - Example constants config
