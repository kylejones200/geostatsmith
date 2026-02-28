# Frequently Asked Questions

## General

### What Python version is required?

GeoStats requires Python 3.12 or newer.

### What platforms are supported?

Currently supported on Ubuntu Linux. Windows and macOS support may be added in the future.

## Usage

### Which kriging method should I use?

- **Ordinary Kriging**: Most common, good default choice
- **Simple Kriging**: Use when mean is well-known
- **Universal Kriging**: Use when trend is present
- **Indicator Kriging**: Use for probability estimation

### How do I choose a variogram model?

Use `auto_fit_variogram()` to automatically select the best model, or try multiple models and compare cross-validation metrics.

### How do I handle large datasets?

Use chunked processing or parallel processing options. See [Performance](BEST_PRACTICES.md#performance) section.

## Troubleshooting

### "Singular matrix" errors

This usually means:
- Duplicate sample locations
- Too few neighbors
- Numerical issues

Solutions:
- Remove duplicate locations
- Increase search radius
- Use regularization

### Slow performance

- Enable parallel processing
- Use chunked processing for large grids
- Enable caching for repeated predictions

## Configuration

### How do I configure constants?

See [Constants Configuration](CONSTANTS_CONFIG.md) for details.

### Can I run analyses from config files?

Yes! See [Config-Driven Analysis](CONFIG_DRIVEN.md) for complete workflows.
