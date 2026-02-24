"""
Spatial statistics and point pattern analysis tools.

Provides methods for analyzing spatial point distributions:
- Nearest neighbor analysis
- Ripley's K function
- Quadrat analysis
- Clustering metrics

Reference: Python Recipes for Earth Sciences (Trauth 2024), Section 7.8
"""

from .point_patterns import (
    clustering_index,
    nearest_neighbor_analysis,
    quadrat_analysis,
    ripley_k_function,
    spatial_randomness_test,
)
from .spatial_autocorrelation import (
    gearys_c,
    morans_i,
)

__all__ = [
    "nearest_neighbor_analysis",
    "ripley_k_function",
    "quadrat_analysis",
    "spatial_randomness_test",
    "clustering_index",
    "morans_i",
    "gearys_c",
]
