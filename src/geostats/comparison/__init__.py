"""
Comparison and benchmarking tools for spatial interpolation methods.

This module provides utilities to compare different interpolation methods
including kriging, IDW, RBF, and natural neighbor interpolation.

Based on concepts from:
- Python Recipes for Earth Sciences (Trauth 2024), Sections 7.6-7.7
- Comparison of gridding and interpolation methods
"""

from .interpolation_comparison import (
    compare_interpolation_methods,
    cross_validate_interpolation,
    benchmark_interpolation_speed,
    interpolation_error_metrics,
)
from .method_implementations import (
    inverse_distance_weighting,
    radial_basis_function_interpolation,
    natural_neighbor_interpolation,
)

__all__ = [
    "compare_interpolation_methods",
    "cross_validate_interpolation",
    "benchmark_interpolation_speed",
    "interpolation_error_metrics",
    "inverse_distance_weighting",
    "radial_basis_function_interpolation",
    "natural_neighbor_interpolation",
]
