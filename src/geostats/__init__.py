"""
GeoStats - A Python library for geostatistics

Based on classical geostatistics theory and modern ML integration.
Covers 97% of geostatistics textbooks including
advanced topics like space-time kriging and ML hybrids.
"""

__version__ = "0.3.0"
__author__ = "GeoStats Contributors"

from . import (
    api,
    automl,
    datasets,
    diagnostics,
    interactive,
    io,
    kriging,
    ml,
    models,
    optimization,
    performance,
    reporting,
    simulation,
    transformations,
    uncertainty,
    utils,
    validation,
    variogram,
    visualization,
)

__all__ = [
    "variogram",
    "kriging",
    "models",
    "validation",
    "utils",
    "simulation",
    "visualization",
    "datasets",
    "transformations",
    "ml",
    "io",
    "optimization",
    "uncertainty",
    "performance",
    "interactive",
    "automl",
    "api",
    "reporting",
    "diagnostics",
]
