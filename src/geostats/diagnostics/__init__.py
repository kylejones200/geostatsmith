"""
Advanced Validation & Diagnostics
==================================

Validation and diagnostic tools.

Features:
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

from .validation_suite import (
 comprehensive_validation,
 spatial_validation,
 model_diagnostics,
)

from .outlier_detection import (
import logging

logger = logging.getLogger(__name__)
 outlier_analysis,
 robust_validation,
)

__all__ = [
 'comprehensive_validation',
 'spatial_validation',
 'model_diagnostics',
 'outlier_analysis',
 'robust_validation',
]
