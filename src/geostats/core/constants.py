"""
Geostatistics Constants

Centralized constants to avoid magic numbers throughout the codebase.
"""

import numpy as np

# Numerical stability and tolerances
EPSILON = 1e-10
SMALL_NUMBER = 1e-6
REGULARIZATION_FACTOR = 1e-8

# Normal score transform
RANK_OFFSET = 0.5  # (i - 0.5) / n for empirical CDF calculation

# Kriging defaults
DEFAULT_MAX_NEIGHBORS = 25  # From Olea (2009): "25 are more than adequate"
DEFAULT_MIN_NEIGHBORS = 3  # From Olea (2009): "3 are a reasonable bare minimum"
DEFAULT_SEARCH_RADIUS_MULTIPLIER = 3.0  # Search radius = range * multiplier

# Octant/Quadrant search
N_OCTANTS = 8
N_QUADRANTS = 4
OCTANT_ANGLE = np.pi / 4  # 45 degrees
QUADRANT_ANGLE = np.pi / 2  # 90 degrees

# Optimization
MAX_ITERATIONS_GLOBAL = 500
MAX_ITERATIONS_LOCAL = 1000
CONVERGENCE_TOLERANCE = 1e-6
OPTIMIZATION_ATOL = 1e-6
OPTIMIZATION_SEED = 42  # Default random seed for reproducibility

# Lognormal kriging
LOG_EPSILON_RATIO = 0.01  # 1% of minimum positive value for zeros

# Block kriging
DEFAULT_N_DISCRETIZATION = 5  # Points per dimension for block discretization
DISCRETIZATION_MIN = 3
DISCRETIZATION_MAX = 20

# Sequential simulation
DEFAULT_N_REALIZATIONS = 100
DEFAULT_N_THRESHOLDS = 5
PROBABILITY_BOUNDS = (0.0, 1.0)

# Variogram fitting
MIN_SEMIVARIANCE_RATIO = 0.01  # Minimum sill as fraction of max semivariance
MAX_SEMIVARIANCE_RATIO = 2.0  # Maximum sill as fraction of max semivariance
MIN_RANGE_RATIO = 0.05  # Minimum range as fraction of max lag
MAX_RANGE_RATIO = 3.0  # Maximum range as fraction of max lag

# Declustering
MIN_CELL_SIZE_RATIO = 0.05  # 1/20 of data range
MAX_CELL_SIZE_RATIO = 2.0  # 2x data range
DEFAULT_N_CELL_SIZES = 10
CLUSTERING_CV_THRESHOLD = 0.5  # CV > 0.5 suggests clustering

# Cross-validation
MIN_TRAIN_FRACTION = 0.5
MAX_TRAIN_FRACTION = 0.95

# Angle conversions
DEGREES_TO_RADIANS = np.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / np.pi

# Array operations
ARRAY_DIMENSION_2D = 2
ARRAY_DIMENSION_3D = 3

# Statistical tests
NORMALITY_P_VALUE_THRESHOLD = 0.05
CORRELATION_THRESHOLD = 0.7

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
