"""
Machine Learning Integration for Geostatistics

This module provides hybrid methods that combine classical geostatistics
with modern machine learning techniques.

Key Approaches:
1. Regression Kriging with ML
 - Use ML models (RF, XGBoost, Neural Nets) for trend
 - Krige the residuals
 - More flexible than polynomial trends

2. Gaussian Process Regression
 - ML interpretation of kriging
 - sklearn-compatible interface
 - Hyperparameter optimization

3. Ensemble Methods
 - Combine multiple models
 - Bootstrap aggregating
 - Model stacking

Modules
-------
regression_kriging : ML-based regression kriging
gaussian_process : Gaussian Process interface
ensemble : Ensemble geostatistical methods
"""

from .regression_kriging import (
 RegressionKriging,
 RandomForestKriging,
 XGBoostKriging,
)
from .gaussian_process import GaussianProcessGeostat
from .ensemble import EnsembleKriging

__all__ = [
 'RegressionKriging',
 'RandomForestKriging',
 'XGBoostKriging',
 'GaussianProcessGeostat',
 'EnsembleKriging',
]
