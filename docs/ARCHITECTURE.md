# GeoStats Architecture

This document describes the internal architecture and design decisions of the GeoStats library.

## Table of Contents

1. [Overview](#overview)
2. [Module Structure](#module-structure)
3. [Core Design Principles](#core-design-principles)
4. [Data Flow](#data-flow)
5. [Extension Points](#extension-points)

---

## Overview

GeoStats is designed with a modular architecture that separates concerns and enables extensibility. The library follows object-oriented principles while maintaining a functional API for common operations.

### Key Design Goals

1. **Modularity**: Clear separation between algorithms, models, I/O, and utilities
2. **Extensibility**: Easy to add new kriging methods, variogram models, etc.
3. **Performance**: Support for parallel processing and optimization
4. **Usability**: Simple API for common tasks, advanced options for experts
5. **Configurability**: Support for config-driven workflows

---

## Module Structure

```
geostats/
├── core/              # Core functionality (constants, validators, exceptions)
├── algorithms/        # Kriging algorithms
├── models/            # Variogram and covariance models
├── math/              # Mathematical utilities (distance, matrices)
├── simulation/        # Geostatistical simulation
├── transformations/   # Data transformations
├── validation/        # Validation and metrics
├── visualization/     # Plotting functions
├── io/                # Input/output operations
├── optimization/      # Sampling design optimization
├── uncertainty/       # Uncertainty quantification
├── performance/       # Performance optimizations
├── ml/                # Machine learning integration
├── automl/            # AutoML functionality
├── interactive/       # Interactive visualizations
├── workflows/         # Config-driven workflows
├── config/            # Configuration management
├── api/               # Web API
├── cli/               # Command-line interface
├── reporting/         # Report generation
└── diagnostics/       # Diagnostic tools
```

---

## Core Design Principles

### 1. Base Classes

**BaseModel**: Abstract base class for variogram/covariance models

```python
class BaseModel(ABC):
    @abstractmethod
    def __call__(self, h: np.ndarray) -> np.ndarray:
        """Evaluate model at distance h"""
        pass
    
    @abstractmethod
    def fit(self, lags, values, **kwargs) -> "BaseModel":
        """Fit model to data"""
        pass
```

**BaseKriging**: Abstract base class for kriging methods

```python
class BaseKriging(ABC):
    @abstractmethod
    def predict(self, x, y, return_variance=True):
        """Perform kriging prediction"""
        pass
    
    @abstractmethod
    def cross_validate(self):
        """Perform cross-validation"""
        pass
```

### 2. Strategy Pattern

Different algorithms implement the same interface:

- All kriging methods implement `BaseKriging`
- All variogram models implement `BaseModel`
- All simulation methods follow the same pattern

### 3. Dependency Injection

Variogram models are injected into kriging objects:

```python
model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)
ok = OrdinaryKriging(x, y, z, variogram_model=model)
```

This allows:
- Easy model swapping
- Testing with mock models
- Custom model implementations

### 4. Immutability

Kriging objects are immutable after creation:

```python
ok = OrdinaryKriging(x, y, z, variogram_model=model)
# ok.x, ok.y, ok.z cannot be changed
# New predictions don't modify the object
```

### 5. Functional API

High-level functions for common operations:

```python
# Instead of:
model = SphericalModel()
fitted = model.fit(lags, gamma)
ok = OrdinaryKriging(x, y, z, variogram_model=fitted)
predictions, variances = ok.predict(x_new, y_new)

# Use:
predictions, variances = auto_interpolate(x, y, z, x_new, y_new)
```

---

## Data Flow

### Typical Workflow

```
1. Data Input
   └─> I/O Module (read_csv, read_geotiff, etc.)
       └─> Validators (validate_coordinates, validate_values)
           └─> Core Data Structures (numpy arrays)

2. Variogram Analysis
   └─> Experimental Variogram Calculation
       └─> Model Fitting
           └─> Fitted Model Object

3. Kriging
   └─> Kriging Object Creation (with variogram model)
       └─> Neighborhood Search
           └─> Kriging System Solution
               └─> Predictions + Variances

4. Output
   └─> Visualization
   └─> I/O (write results)
   └─> Reporting
```

### Config-Driven Workflow

```
1. YAML/JSON Config File
   └─> Config Parser (Pydantic validation)
       └─> AnalysisConfig Object
           └─> AnalysisPipeline
               ├─> Data Loading
               ├─> Preprocessing
               ├─> Variogram Fitting
               ├─> Kriging
               ├─> Validation
               ├─> Visualization
               └─> Output Generation
```

---

## Extension Points

### Adding a New Variogram Model

1. Create model class inheriting from `BaseModel`:

```python
from geostats.core.base import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, param1=1.0, param2=2.0):
        super().__init__()
        self.set_parameters(param1=param1, param2=param2)
    
    def __call__(self, h):
        # Implement model equation
        param1 = self._parameters['param1']
        param2 = self._parameters['param2']
        return param1 * (1 - np.exp(-h / param2))
    
    def fit(self, lags, values, **kwargs):
        # Implement fitting logic
        # ...
        return self
```

2. Register in `variogram.fit_model()`:

```python
# In variogram.py
MODELS = {
    'spherical': SphericalModel,
    'exponential': ExponentialModel,
    'my_custom': MyCustomModel,  # Add here
}
```

### Adding a New Kriging Method

1. Create kriging class inheriting from `BaseKriging`:

```python
from geostats.core.base import BaseKriging

class MyCustomKriging(BaseKriging):
    def predict(self, x, y, return_variance=True):
        # Implement prediction logic
        # ...
        return predictions, variances
    
    def cross_validate(self):
        # Implement cross-validation
        # ...
        return predictions, metrics
```

2. Add to appropriate module or create new module

### Adding a New I/O Format

1. Create read/write functions:

```python
# In io/custom_format.py
def read_custom_format(filename):
    # Read logic
    return x, y, z, metadata

def write_custom_format(filename, x, y, z, **kwargs):
    # Write logic
    pass
```

2. Export in `io/__init__.py`

---

## Performance Considerations

### Parallel Processing

- Uses `joblib` for parallel kriging
- Chunked processing for large grids
- Caching for repeated operations

### Memory Management

- Lazy evaluation where possible
- Chunked I/O for large files
- Efficient array operations with NumPy

### Numerical Stability

- Regularization for singular matrices
- Epsilon constants for numerical comparisons
- Careful handling of edge cases

---

## Testing Strategy

### Unit Tests

- Test individual functions/classes
- Mock dependencies where appropriate
- Test edge cases and error conditions

### Integration Tests

- Test complete workflows
- Test config-driven pipelines
- Test I/O operations

### Performance Tests

- Benchmark key operations
- Test with large datasets
- Monitor memory usage

---

## Configuration System

### Pydantic Schemas

All configuration uses Pydantic for:
- Type validation
- Default values
- Documentation
- Serialization

### Constants System

- Python defaults in `constants.py`
- YAML overrides via `constants_loader.py`
- Environment variable support
- Runtime reloading

---

## Future Extensibility

### Plugin System (Planned)

Allow external plugins for:
- Custom variogram models
- Custom kriging methods
- Custom I/O formats
- Custom visualizations

### GPU Acceleration (Planned)

- CuPy integration for large datasets
- GPU-accelerated matrix operations
- Parallel simulation on GPU

---

For implementation details, see the source code and docstrings.
