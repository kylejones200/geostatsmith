# GeoStats Overview

GeoStats is a Python library for geostatistics and spatial data analysis, providing comprehensive tools for variogram analysis, kriging interpolation, and spatial simulation.

## What is GeoStats?

GeoStats is a modern Python library that implements classical geostatistical methods with modern software engineering practices. It supports:

- **15+ Kriging Methods**: Simple, Ordinary, Universal, Block, 3D, Cokriging, Indicator, and more
- **9 Variogram Models**: Spherical, Exponential, Gaussian, Matérn, and others
- **AutoML Integration**: Automatic model selection and hyperparameter tuning
- **Config-Driven Workflows**: Run complete analyses from YAML/JSON configuration files
- **High Performance**: Parallel processing, caching, and optimization

## Key Features

### Geostatistics
- Experimental variogram calculation with multiple estimators
- Theoretical variogram model fitting
- Multiple kriging interpolation methods
- Spatial simulation techniques

### Machine Learning Integration
- Regression Kriging with scikit-learn models
- Gaussian Process Regression
- Random Forest and XGBoost Kriging
- Ensemble methods

### Performance
- Parallel processing for large datasets
- Caching for repeated operations
- Chunked processing for memory efficiency
- Optimized numerical routines

## Use Cases

GeoStats is used in:
- Environmental science (soil contamination, pollution mapping)
- Geology (ore grade estimation, resource modeling)
- Hydrogeology (aquifer characterization)
- Agriculture (precision agriculture, yield mapping)
- Meteorology (spatial interpolation of weather data)
- Public health (disease mapping, exposure assessment)

## Getting Started

See [Quickstart](QUICKSTART.md) for a quick introduction, or [Installation](INSTALL.md) for installation instructions.
