"""
Uncertainty Quantification
==========================

Tools for quantifying and visualizing prediction uncertainty.

Key Features:
- Bootstrap confidence intervals
- Conditional simulation
- Probability maps
- Risk assessment

Examples
--------
>>> from geostats.uncertainty import bootstrap_uncertainty, probability_map
>>>
>>> # Bootstrap confidence intervals
>>> results = bootstrap_uncertainty(
... x, y, z,
... x_pred, y_pred,
... variogram_model=model,
... n_bootstrap=100
... )
>>>
>>> # Probability that value exceeds threshold
>>> prob = probability_map(
... x, y, z,
... x_pred, y_pred,
... variogram_model=model,
... threshold=10.0
... )
"""

from .bootstrap import (
    bootstrap_uncertainty,
    bootstrap_variogram,
    bootstrap_kriging,
)

from .probability import (
    probability_map,
    conditional_probability,
    risk_assessment,
)

from .confidence_intervals import (
    confidence_intervals,
    prediction_bands,
    uncertainty_ellipse,
)

__all__ = [
    # Bootstrap methods
    "bootstrap_uncertainty",
    "bootstrap_variogram",
    "bootstrap_kriging",
    # Probability maps
    "probability_map",
    "conditional_probability",
    "risk_assessment",
    # Confidence intervals
    "confidence_intervals",
    "prediction_bands",
    "uncertainty_ellipse",
]
