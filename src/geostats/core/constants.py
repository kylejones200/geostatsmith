"""
Geostatistics Constants

Centralized constants to avoid magic numbers throughout the codebase.

This module supports config-driven workloads:
- Python constants serve as defaults
- YAML configs can override via constants_loader
- Environment variable GEOSTATS_CONSTANTS_CONFIG can specify config path
- AnalysisConfig can include constants section

Usage:
    # Standard usage (Python defaults)
    from geostats.core.constants import EPSILON

    # Config-driven (automatic if GEOSTATS_CONSTANTS_CONFIG is set)
    from geostats.core.constants import EPSILON  # Auto-loads from config if available

    # Explicit config-driven
    from geostats.core.constants_loader import get_constants
    constants = get_constants('config/constants.yaml')
"""

import os
from typing import Any

# Import loader for config-driven support
from .constants_loader import ConstantsLoader
from .constants_loader import get_constants as _get_constants

# Default constants (used if no config override)
_DEFAULT_CONSTANTS = {
    # Numerical stability and tolerances
    "EPSILON": 1e-10,
    "SMALL_NUMBER": 1e-6,
    "REGULARIZATION_FACTOR": 1e-8,
    # Normal score transform
    "RANK_OFFSET": 0.5,  # (i - 0.5) / n for empirical CDF calculation
    # Kriging defaults
    "DEFAULT_MAX_NEIGHBORS": 25,  # From Olea (2009): "25 are more than adequate"
    "DEFAULT_MIN_NEIGHBORS": 3,  # From Olea (2009): "3 are a reasonable bare minimum"
    "DEFAULT_SEARCH_RADIUS_MULTIPLIER": 3.0,  # Search radius = range * multiplier
    # Octant/Quadrant search
    "N_OCTANTS": 8,
    "N_QUADRANTS": 4,
    "OCTANT_ANGLE": 0.7853981633974483,  # np.pi / 4 (45 degrees)
    "QUADRANT_ANGLE": 1.5707963267948966,  # np.pi / 2 (90 degrees)
    # Optimization
    "MAX_ITERATIONS_GLOBAL": 500,
    "MAX_ITERATIONS_LOCAL": 1000,
    "CONVERGENCE_TOLERANCE": 1e-6,
    "OPTIMIZATION_ATOL": 1e-6,
    "OPTIMIZATION_SEED": 42,  # Default random seed for reproducibility
    # Lognormal kriging
    "LOG_EPSILON_RATIO": 0.01,  # 1% of minimum positive value for zeros
    # Block kriging
    "DEFAULT_N_DISCRETIZATION": 5,  # Points per dimension for block discretization
    "DISCRETIZATION_MIN": 3,
    "DISCRETIZATION_MAX": 20,
    # Sequential simulation
    "DEFAULT_N_REALIZATIONS": 100,
    "DEFAULT_N_THRESHOLDS": 5,
    "PROBABILITY_BOUNDS": (0.0, 1.0),
    # Variogram fitting
    "MIN_SEMIVARIANCE_RATIO": 0.01,  # Minimum sill as fraction of max semivariance
    "MAX_SEMIVARIANCE_RATIO": 2.0,  # Maximum sill as fraction of max semivariance
    "MIN_RANGE_RATIO": 0.05,  # Minimum range as fraction of max lag
    "MAX_RANGE_RATIO": 3.0,  # Maximum range as fraction of max lag
    # Declustering
    "MIN_CELL_SIZE_RATIO": 0.05,  # 1/20 of data range
    "MAX_CELL_SIZE_RATIO": 2.0,  # 2x data range
    "DEFAULT_N_CELL_SIZES": 10,
    "CLUSTERING_CV_THRESHOLD": 0.5,  # CV > 0.5 suggests clustering
    # Cross-validation
    "MIN_TRAIN_FRACTION": 0.5,
    "MAX_TRAIN_FRACTION": 0.95,
    # Angle conversions
    "DEGREES_TO_RADIANS": 0.017453292519943295,  # np.pi / 180.0
    "RADIANS_TO_DEGREES": 57.29577951308232,  # 180.0 / np.pi
    # Array operations
    "ARRAY_DIMENSION_2D": 2,
    "ARRAY_DIMENSION_3D": 3,
    # Statistical tests
    "NORMALITY_P_VALUE_THRESHOLD": 0.05,
    "CORRELATION_THRESHOLD": 0.7,
    # Logging
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_DATE_FORMAT": "%Y-%m-%d %H:%M:%S",
    # Outlier detection
    "Z_SCORE_THRESHOLD": 3.0,
    "MODIFIED_Z_THRESHOLD": 3.5,
    "IQR_MULTIPLIER": 1.5,
    "IQR_EXTREME_MULTIPLIER": 3.0,  # For extreme outliers (typically 3.0)
    "SPATIAL_NEIGHBORS_MIN": 5,
    "SPATIAL_THRESHOLD_FACTOR": 3.0,
    "SPATIAL_NEIGHBORS_DEFAULT_K": 6,  # Includes self
    "OUTLIER_IMPROVEMENT_THRESHOLD": 100.0,  # Percentage improvement
    # Sampling Design
    "DEFAULT_N_CANDIDATES": 1000,
    "DEFAULT_N_NEW_SAMPLES": 5,
    "DEFAULT_X_MARGIN_RATIO": 0.1,
    "DEFAULT_Y_MARGIN_RATIO": 0.1,
    "DEFAULT_VARIANCE_THRESHOLD": 0.5,
    "DEFAULT_MAX_SAMPLES": 100,
    "DEFAULT_N_EVAL": 50,
    "DEFAULT_STRATIFICATION_GRID": (10, 10),
    # Disjunctive Kriging
    "CDF_CLIP_EPSILON": 1e-10,
    "DZDY_THRESHOLD": 1e-6,
    # Nested Variogram
    "NUGGET_MAX_FRACTION_DEFAULT": 0.5,
    "MIN_SEMIVARIANCE_RATIO_DEFAULT": 0.1,
    "MAX_SEMIVARIANCE_RATIO_DEFAULT": 1.5,
    "MIN_RANGE_RATIO_DEFAULT": 0.1,
    "MAX_RANGE_RATIO_DEFAULT": 2.0,
    "MAX_ITERATIONS_GLOBAL_DEFAULT": 100,
    "OPTIMIZATION_ATOL_DEFAULT": 1e-6,
    "CONVERGENCE_TOLERANCE_DEFAULT": 1e-4,
    # Visualization
    "SCATTER_SIZE_MULTIPLIER": 100,
    "SCATTER_SIZE_MIN": 20,
    "SCATTER_SIZE_BASE": 20,
    "SCATTER_SIZE_SCALE": 200,
    "SCATTER_ALPHA_DEFAULT": 0.6,
    "SCATTER_ALPHA_CLOUD": 0.3,
    "SCATTER_SIZE_CLOUD": 10,
    "LINEWIDTH_DEFAULT": 2,
    "LINEWIDTH_THIN": 1,
    "FONTSIZE_SMALL": 9,
    "FONTSIZE_MEDIUM": 11,
    "FONTSIZE_LARGE": 12,
    "FONTSIZE_XLARGE": 14,
    "TEXT_X_POS": 0.65,
    "TEXT_Y_POS": 0.05,
    "ALPHA_HALF": 0.5,
    "COLOR_EXPERIMENTAL": "#1f77b4",
    "COLOR_MODEL": "#d62728",
    "COLOR_WHITE": "white",
    "LINESTYLE_DASHED": "--",
    # H-scatterplot
    "H_SCATTER_TOLERANCE_DEFAULT": 0.5,
    "H_SCATTER_ANGLE_TOLERANCE_DEFAULT": 45.0,
    "H_SCATTER_POINT_SIZE": 30,
    "H_SCATTER_LINEWIDTH": 0.5,
    # Directional Variogram
    "DEFAULT_DIRECTIONS": [0, 45, 90, 135],
    "DEFAULT_ANGLE_TOLERANCE": 22.5,
    # Variogram Map
    "SEMIVARIANCE_FACTOR": 0.5,
    # Variogram calculation
    "MAXLAG_FRACTION": 0.5,  # Default maxlag is half the maximum distance
    "LAG_TOL_FRACTION": 0.5,  # Default lag tolerance is half the lag width
    "SEMIVARIANCE_DIVISOR": 2.0,  # Semivariance = sum(diff^2) / (2 * n_pairs)
    "DEFAULT_N_LAGS": 15,  # Default number of lag bins for variogram calculation
    "DEFAULT_MAX_DISTANCE": 100.0,  # Default maximum distance for variogram
    "VARIANCE_THRESHOLD_DEFAULT": 0.5,  # Default variance threshold
    # Variogram model coefficients
    "SPHERICAL_COEFFICIENT_1": 1.5,  # Coefficient for spherical model: 1.5*(h/a)
    "SPHERICAL_COEFFICIENT_2": 0.5,  # Coefficient for spherical model: 0.5*(h/a)^3
    "MATERN_POWER_BASE": 2.0,  # Base for Matern model power calculation
    "MATERN_NU_DEFAULT": 0.5,  # Default nu parameter for Matern model (exponential when nu=0.5)
    # Matrix operations
    "UNBIASEDNESS_CONSTRAINT": 1.0,  # Constraint value for unbiasedness in kriging
    "ZERO_VALUE": 0.0,  # Zero value for matrix operations
    # Default sill value
    "DEFAULT_SILL_VALUE": 1.0,
    # Probability bounds
    "PROBABILITY_CLIP_MIN": 1e-10,
    "PROBABILITY_CLIP_MAX": 1.0 - 1e-10,
    # Percentages
    "PERCENTAGE_MULTIPLIER": 100.0,  # Multiply by 100 to convert to percentage
    "PERCENTAGE_DIVISOR": 100.0,  # Divide by 100 to convert from percentage
    # Variogram map
    "VARIOMAP_GRID_MULTIPLIER": 2,  # 2*n_lags for variogram map grid
}

# Global config path (can be set programmatically or via environment variable)
_CONFIG_PATH: str | None = None


def set_constants_config(config_path: str | None) -> None:
    """
    Set the global constants config path for config-driven workloads.

    Parameters
    ----------
    config_path : str or None
        Path to YAML constants config file. If None, uses Python defaults.

    Examples
    --------
    >>> from geostats.core.constants import set_constants_config
    >>> set_constants_config('config/my_constants.yaml')
    >>> from geostats.core.constants import EPSILON  # Now uses config value
    """
    global _CONFIG_PATH
    _CONFIG_PATH = config_path
    # Clear cache to force reload
    ConstantsLoader.clear_cache()


def get_constants_config() -> str | None:
    """Get the current global constants config path."""
    return _CONFIG_PATH or os.getenv("GEOSTATS_CONSTANTS_CONFIG")


def _load_constants() -> dict[str, Any]:
    """Load constants (config-driven if available, else defaults)."""
    config_path = get_constants_config()
    if config_path:
        try:
            return _get_constants(config_path)
        except (FileNotFoundError, Exception):
            # Fall back to defaults if config fails to load
            return _DEFAULT_CONSTANTS.copy()
    return _DEFAULT_CONSTANTS.copy()


# Load constants (config-driven if available)
_CONSTANTS = _load_constants()

# Export all constants as module-level variables for backward compatibility
# This allows: from geostats.core.constants import EPSILON
for name, value in _CONSTANTS.items():
    globals()[name] = value


def reload_constants(config_path: str | None = None) -> None:
    """
    Reload constants from config (useful when config changes).

    Parameters
    ----------
    config_path : str, optional
        Config path to use. If None, uses current global config.

    Examples
    --------
    >>> from geostats.core.constants import reload_constants, EPSILON
    >>> reload_constants('config/new_constants.yaml')
    >>> print(EPSILON)  # Now uses new config
    """
    global _CONSTANTS
    if config_path:
        set_constants_config(config_path)
    _CONSTANTS = _load_constants()
    # Update module-level variables
    for name, value in _CONSTANTS.items():
        globals()[name] = value


# Make constants available as dict for programmatic access
def get_all_constants() -> dict[str, Any]:
    """Get all constants as a dictionary."""
    return _CONSTANTS.copy()


# Alias for backward compatibility with constants_loader
def get_constants(config_path: str | None = None) -> dict[str, Any]:
    """
    Get all constants as a dictionary (alias for get_all_constants).

    This function provides backward compatibility. For config-driven workloads,
    use geostats.core.constants_loader.get_constants() instead.

    Parameters
    ----------
    config_path : str, optional
        Ignored for backward compatibility. Use constants_loader for config support.

    Returns
    -------
    dict
        Dictionary of all constants
    """
    return get_all_constants()


__all__ = list(_DEFAULT_CONSTANTS.keys()) + [
    "set_constants_config",
    "get_constants_config",
    "reload_constants",
    "get_all_constants",
    "get_constants",
]
