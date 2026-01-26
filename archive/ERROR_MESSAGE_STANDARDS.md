# Error Message Standards

## Principles

1. **Be Specific**: State what went wrong and why
2. **Be Actionable**: Tell the user what they can do to fix it
3. **Be Consistent**: Use the same format and terminology throughout
4. **Be Clear**: Avoid jargon when possible

## Error Message Format

```python
# Good format: What went wrong + specific details + (optional) suggestion
raise ValidationError(f"X and Y coordinates must have same length: got {len(x)} and {len(y)}")

# Bad format: Vague or unhelpful
raise ValueError("bad input")
```

## Standard Patterns

### 1. Validation Errors

**Shape mismatches:**
```python
raise ValidationError(
    f"X and Y coordinates must have same shape: got {x.shape} vs {y.shape}"
)
```

**Missing required values:**
```python
raise ValidationError(f"Parameter '{name}' is required but was not provided")
```

**Invalid values:**
```python
raise ValidationError(f"{name} must be positive, got {value}")
raise ValidationError(f"{name} must be in range [{min_val}, {max_val}], got {value}")
```

**NaN/Inf:**
```python
raise ValidationError(f"Values contain {np.sum(np.isnan(values))} NaN entries")
raise ValidationError("Values contain infinity")
```

### 2. Fitting Errors

**All methods failed:**
```python
raise FittingError(
    f"All {len(methods)} variogram models failed to fit. "
    f"Check data quality and try different models."
)
```

**Convergence:**
```python
raise ConvergenceError(
    f"Optimization failed to converge after {max_iter} iterations. "
    f"Try increasing max_iter or adjusting initial parameters."
)
```

### 3. Missing Dependencies

**Standard format with installation instruction:**
```python
raise ImportError(
    "plotly is required for interactive plots. "
    "Install with: pip install plotly"
)
```

### 4. Runtime Errors

**Method not available:**
```python
raise RuntimeError(
    f"Method '{method}' is not available. "
    f"Available methods: {', '.join(available_methods)}"
)
```

**Model not fitted:**
```python
raise ModelError(
    "Model must be fitted before prediction. Call fit() first."
)
```

### 5. Logger Warnings

**Non-fatal issues:**
```python
logger.warning(f"Method '{method}' failed with error: {str(e)}. Skipping.")
logger.warning(f"Optional dependency '{package}' not available. Feature disabled.")
logger.warning(f"Parameter '{param}' = {value} is outside typical range [{min_val}, {max_val}]")
```

### 6. Logger Errors

**Fatal issues before raising:**
```python
logger.error(f"All {len(methods)} interpolation methods failed")
logger.error(f"Data validation failed: {error_message}")
```

## Exception Type Guidelines

- **ValidationError**: Invalid input parameters, shapes, values
- **FittingError**: Model fitting failures
- **KrigingError**: Kriging calculation failures  
- **ConvergenceError**: Optimization/iteration failures
- **ModelError**: Model state issues (not fitted, invalid configuration)
- **DataError**: Data quality issues
- **SimulationError**: Simulation-specific failures
- **ImportError**: Missing dependencies (built-in)
- **ValueError**: Invalid choices/enums (built-in, use sparingly)
- **RuntimeError**: Generic runtime issues (use custom exceptions when possible)

## Examples

### Before (Inconsistent)
```python
raise RuntimeError("All methods failed")
raise ValueError(f"Unknown method: {method}")
logger.warning("IDW not available (comparison module needed)")
```

### After (Consistent)
```python
raise FittingError(
    f"All {len(methods)} methods failed. Check data quality and try different approaches."
)
raise ValidationError(
    f"Unknown method '{method}'. Available methods: {', '.join(valid_methods)}"
)
logger.warning(
    "IDW interpolation not available: comparison module not installed. "
    "Install with: pip install geostats[comparison]"
)
```
