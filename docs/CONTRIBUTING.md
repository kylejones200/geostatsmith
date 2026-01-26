# Contributing to GeoStats

We welcome contributions to the GeoStats library! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/yourusername/geostats.git
cd geostats
```

3. Install development dependencies:
```bash
uv sync --dev
```

4. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

## Code Standards

### Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 88 characters (Black formatter default)
- Use descriptive variable names

### Code Formatting

We use Black for code formatting:

```bash
uv run black src/geostats tests examples
```

### Type Checking

Run mypy for type checking:

```bash
uv run mypy src/geostats
```

### Linting

Run ruff for linting (recommended):

```bash
uv run ruff check .
```

Or flake8:

```bash
uv run flake8 src/geostats tests
```

## Testing

### Running Tests

Run all tests:

```bash
uv run pytest tests/
```

Run with coverage:

```bash
uv run pytest tests/ --cov=geostats --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names
- Include docstrings explaining what each test does

Example test:

```python
def test_spherical_model_at_origin():
 """Test that spherical model equals nugget at h=0"""
 model = SphericalModel(nugget=0.5, sill=2.0, range_param=10.0)
 result = model(np.array([0.0]))
 assert np.isclose(result[0], 0.5)
```

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1: int, param2: str) -> float:
 """
 Brief description of function.

 Longer description if needed.

 Parameters
 ----------
 param1 : int
 Description of param1
 param2 : str
 Description of param2

 Returns
 -------
 float
 Description of return value

 Examples
 --------
 >>> result = my_function(5, "test")
 >>> print(result)
 42.0
 """
 return 42.0
```

### Building Documentation

```bash
cd docs
make html
```

## Pull Request Process

1. **Create descriptive PR title**: Use conventional commits format
 - `feat: Add new kriging method`
 - `fix: Fix variogram calculation bug`
 - `docs: Update README`
 - `test: Add tests for kriging`

2. **Write clear description**: Explain what changes you made and why

3. **Add tests**: Ensure new features have test coverage

4. **Update documentation**: Update relevant docstrings and docs

5. **Run checks**: Ensure all tests pass and code is formatted

6. **Request review**: Request review from maintainers

## Architecture Guidelines

Follow the layered architecture (see `ARCHITECTURE.md`):

1. **Core Layer**: Base classes, types, exceptions, validators
2. **Math Layer**: Distance calculations, matrix operations
3. **Models Layer**: Variogram and covariance models
4. **Algorithms Layer**: Variogram calculation, kriging implementations
5. **API Layer**: User-facing interfaces
6. **Visualization Layer**: Plotting functions

### Adding a New Variogram Model

1. Create model class in `src/geostats/models/variogram_models.py`:

```python
class MyModel(VariogramModelBase):
 """
 My custom variogram model

 Formula:
 γ(h) = ...
 """

 def _model_function(self, h):
 """Implement model formula"""
 # Your implementation
 return gamma_values
```

2. Add tests in `tests/test_variogram.py`

3. Update `__all__` in model module

4. Add documentation and example

### Adding a New Kriging Method

1. Create class in `src/geostats/algorithms/`:

```python
class MyKriging(BaseKriging):
 """My custom kriging method"""

 def predict(self, x, y, return_variance=True):
 """Implement prediction logic"""
 # Your implementation
 return predictions, variances

 def cross_validate(self):
 """Implement cross-validation"""
 # Your implementation
 return predictions, metrics
```

2. Add tests in `tests/test_kriging.py`

3. Export in appropriate `__init__.py` files

4. Add example in `examples/`

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, library version
6. **Error messages**: Full error traceback if applicable

Example:

```python
# Minimal reproducible example
import numpy as np
from geostats import variogram

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
z = np.array([1, 2, 3])

# This causes an error
lags, gamma, n = variogram.experimental_variogram(x, y, z)
# Error: ...
```

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the best outcome for the project
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions or discussions!
