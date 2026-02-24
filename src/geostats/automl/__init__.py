"""
    AutoML for Geostatistics
=========================

Automated model selection and hyperparameter tuning.

Key Features:
    pass
- Automatic variogram model selection
- Automatic method selection (kriging vs alternatives)
- Hyperparameter optimization
- One-function workflows

Examples
--------
>>> from geostats.automl import auto_interpolate, auto_variogram
>>>
>>> # Automatic everything
>>> results = auto_interpolate(x, y, z, x_pred, y_pred)
>>> # returns best method, fitted model, predictions
>>>
>>> # Just variogram
>>> model = auto_variogram(x, y, z)
>>> logger.info(f"Best model: {model}")
"""

import logging

from .auto_method import (
    auto_interpolate,
    auto_select_method,  # Alias for backward compatibility
    suggest_method,
)
from .auto_variogram import (
    auto_fit,
    auto_fit_variogram,  # Alias for backward compatibility
    auto_variogram,
)
from .hyperparameter_tuning import (
    optimize_neighborhood,
    tune_kriging,
    tune_variogram_hyperparameters,  # Alias for backward compatibility
)

__all__ = [
    # Auto variogram
    "auto_variogram",
    "auto_fit_variogram",  # Alias for backward compatibility
    "auto_fit",
    # Auto method
    "auto_interpolate",
    "auto_select_method",  # Alias for backward compatibility
    "suggest_method",
    # Hyperparameter tuning
    "tune_kriging",
    "tune_variogram_hyperparameters",  # Alias for backward compatibility
    "optimize_neighborhood",
]
