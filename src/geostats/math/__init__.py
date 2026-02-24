"""
Mathematical operations layer - distance calculations, matrix operations, numerical methods
"""

from .distance import (
    anisotropic_distance,
    euclidean_distance,
    euclidean_distance_matrix,
    manhattan_distance,
    pairwise_distances,
)
from .matrices import (
    build_covariance_matrix,
    build_variogram_matrix,
    is_positive_definite,
    regularize_matrix,
    solve_kriging_system,
)
from .numerical import (
    optimize_parameters,
    ordinary_least_squares,
    weighted_least_squares,
)

__all__ = [
    # Distance functions
    "euclidean_distance",
    "euclidean_distance_matrix",
    "manhattan_distance",
    "anisotropic_distance",
    "pairwise_distances",
    # Matrix operations
    "build_covariance_matrix",
    "build_variogram_matrix",
    "solve_kriging_system",
    "is_positive_definite",
    "regularize_matrix",
    # Numerical methods
    "weighted_least_squares",
    "ordinary_least_squares",
    "optimize_parameters",
]
