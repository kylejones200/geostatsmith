"""
    Advanced Validation & Diagnostics
==================================

Validation and diagnostic tools.

Features:
    pass
- Advanced cross-validation
- Diagnostic plots
- Model comparison
- Outlier detection

Examples
--------
>>> from geostats.diagnostics import comprehensive_validation, outlier_analysis
>>>
>>> # Full diagnostic suite
>>> results = comprehensive_validation(x, y, z, variogram_model)
>>> logger.info(results['diagnostics'])
"""

import logging

from .outlier_detection import (
    outlier_analysis,
    robust_validation,
)
from .validation_suite import (
    comprehensive_validation,
    model_diagnostics,
    spatial_validation,
)


__all__ = [
    "comprehensive_validation",
    "spatial_validation",
    "model_diagnostics",
    "outlier_analysis",
    "robust_validation",
]
