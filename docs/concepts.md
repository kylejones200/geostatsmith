# Core Concepts

This document explains the core geostatistical concepts used in GeoStats.

## Variograms

A variogram describes the spatial correlation structure of a random field. It measures how dissimilarity between values increases with distance.

### Experimental Variogram

The experimental variogram is calculated from observed data:

```
gamma(h) = 1/(2*N(h)) * sum[z(xi) - z(xi+h)]^2
```

where:
- `h` is the lag distance
- `N(h)` is the number of pairs at distance h
- `z(xi)` is the value at location xi

### Theoretical Variogram Models

Theoretical models fit the experimental variogram:
- **Spherical**: Reaches sill at exact range
- **Exponential**: Approaches sill asymptotically
- **Gaussian**: Very smooth near origin
- **Matérn**: Flexible family with smoothness parameter

## Kriging

Kriging is a geostatistical interpolation method that provides:
- **Best Linear Unbiased Predictions** (BLUP)
- **Uncertainty quantification** (kriging variance)

### Types of Kriging

- **Simple Kriging**: Assumes known mean
- **Ordinary Kriging**: Estimates unknown mean
- **Universal Kriging**: Accounts for trend/drift
- **Indicator Kriging**: For probability estimation
- **Cokriging**: Uses secondary variables

## Spatial Simulation

Spatial simulation generates multiple realizations of a random field:
- **Sequential Gaussian Simulation**: Most common method
- **Sequential Indicator Simulation**: For categorical data
- **Truncated Gaussian Simulation**: For facies modeling

## Cross-Validation

Cross-validation assesses model performance:
- **Leave-One-Out**: Predict each point using all others
- **K-Fold**: Divide data into k folds
- **Spatial**: Respect spatial structure in folds
