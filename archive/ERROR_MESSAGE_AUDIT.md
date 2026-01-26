# Error Message Standardization Audit

## Changes Applied

### 1. AutoML Module

**auto_method.py:**
- ✅ Changed: `RuntimeError("All methods failed")` 
- ✅ To: `FittingError("All {len(methods)} interpolation methods failed. Check data quality...")`
- ✅ Changed: `logger.warning("IDW not available (comparison module needed)")`
- ✅ To: `logger.warning("IDW interpolation not available: comparison module not installed. Install with: pip install geostats[comparison]")`
- ✅ Changed: `logger.error(f"FAILED: {e}")`
- ✅ To: `logger.error(f"Method '{method}' failed: {str(e)}")`

**auto_variogram.py:**
- ✅ Changed: `RuntimeError("All variogram fits failed")`
- ✅ To: `FittingError("All {len(model_types)} variogram models failed to fit. Check data quality...")`
- ✅ Changed: `logger.warning(f"- {model_type}: FAILED ({e})")`
- ✅ To: `logger.warning(f"Model '{model_type}' failed to fit: {str(e)}")`

### 2. CLI Module

**cli.py:**
- ✅ Combined two logger.error statements into one clear message
- ✅ Changed: Two separate error lines
- ✅ To: `logger.error("FastAPI and uvicorn are required for API server. Install with: pip install fastapi uvicorn")`

### 3. Interactive Module

**All files (variogram_plots.py, prediction_maps.py, comparison.py):**
- ✅ Changed: `raise ImportError("plotly is required")`
- ✅ To: `raise ImportError("plotly is required for interactive plots. Install with: pip install plotly")`
- ✅ Applied to 6 locations across 3 files

### 4. Comparison Module

**interpolation_comparison.py:**
- ✅ Changed: `ValueError("Need at least {MIN_TRAINING_SAMPLES} points for cross-validation")`
- ✅ To: `ValidationError("Insufficient data for cross-validation: need at least {MIN_TRAINING_SAMPLES} points, got {n}. Add more sample points...")`
- ✅ Improved fold error message to include fold number
- ✅ Improved method failure messages to include method name in quotes

## Standards Established

### Error Message Template
```
[Problem] + [Specific Details] + [Action/Solution]
```

### Examples

**Good:**
- `"All 3 interpolation methods failed. Check data quality (sufficient points, no duplicates, valid values)."`
- `"Insufficient data for cross-validation: need at least 10 points, got 7. Add more sample points or use a simpler validation approach."`
- `"plotly is required for interactive plots. Install with: pip install plotly"`

**Bad (old style):**
- `"All methods failed"`
- `"Need at least 10 points"`
- `"plotly is required"`

### Exception Types Used

| Type | Use Case | Example |
|------|----------|---------|
| `ValidationError` | Invalid input | Shape mismatches, out of range values |
| `FittingError` | Model fitting failures | All models failed, convergence issues |
| `KrigingError` | Kriging computation | Matrix singularity, system solve failures |
| `ImportError` | Missing dependencies | plotly, sklearn not available |
| `ModelError` | Model state issues | Not fitted, invalid configuration |

## Remaining Work

### Low Priority (docstring examples only)
- Docstring example outputs (45 print statements in examples)
- These are documentation and should remain

### Already Correct
- `core/validators.py` - Excellent, consistent messages
- `core/exceptions.py` - Well-defined exception hierarchy  
- Most algorithm modules - Use specific exception types correctly

## Verification

Run these to check for remaining issues:
```bash
# Check for generic RuntimeError/ValueError
grep -r "raise RuntimeError\|raise ValueError" src/geostats --include="*.py" | grep -v "# "

# Check for vague error messages  
grep -r 'raise.*Error("' src/geostats --include="*.py" | grep -E '"(fail|error|bad|invalid)"'

# Check for warnings without context
grep -r 'logger.warning(' src/geostats --include="*.py" | grep -v "not available"
```

## Benefits

1. **User-friendly**: Clear explanations of what went wrong
2. **Actionable**: Users know how to fix the problem
3. **Consistent**: Same format and terminology throughout
4. **Debuggable**: Specific details help identify root causes
5. **Professional**: Matches industry best practices
