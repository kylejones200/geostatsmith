"""
Mathematical operations layer - distance calculations, matrix operations, numerical methods
"""

from .distance import (
 euclidean_distance,
 euclidean_distance_matrix,
 anisotropic_distance,
 pairwise_distances,
)
from .matrices import (
 build_covariance_matrix,
 build_variogram_matrix,
 solve_kriging_system,
 is_positive_definite,
 regularize_matrix,
)
from .numerical import (
 weighted_least_squares,
 ordinary_least_squares,
 optimize_parameters,
)

__all__ = [
 # Distance functions
 "euclidean_distance",
 "euclidean_distance_matrix",
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
