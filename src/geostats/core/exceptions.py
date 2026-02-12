"""
Custom exception classes for the GeoStats library
"""


class GeoStatsError(Exception):
    """Base exception class for all GeoStats errors"""

    pass


class ValidationError(GeoStatsError):
    """Raised when input validation fails"""

    pass


class FittingError(GeoStatsError):
    """Raised when model fitting fails"""

    pass


class KrigingError(GeoStatsError):
    """Raised when kriging calculation fails"""

    pass


class ConvergenceError(GeoStatsError):
    """Raised when iterative algorithm fails to converge"""

    pass


class ModelError(GeoStatsError):
    """Raised when there are issues with model configuration or usage"""

    pass


class DataError(GeoStatsError):
    """Raised when there are issues with input data"""

    pass


class SimulationError(GeoStatsError):
    """Raised when simulation fails"""

    pass
