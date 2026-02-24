"""
Core foundation layer - base classes, types, exceptions, validators
"""

from .base import BaseKriging, BaseModel
from .exceptions import (
    ConvergenceError,
    FittingError,
    GeoStatsError,
    KrigingError,
    ValidationError,
)
from .types import (
    ArrayLike,
    CoordinatesType,
    ValuesType,
)
from .validators import (
    validate_coordinates,
    validate_in_range,
    validate_positive,
    validate_values,
)

__all__ = [
    # Base classes
    "BaseModel",
    "BaseKriging",
    # Exceptions
    "GeoStatsError",
    "ValidationError",
    "FittingError",
    "KrigingError",
    "ConvergenceError",
    # Types
    "ArrayLike",
    "CoordinatesType",
    "ValuesType",
    # Validators
    "validate_coordinates",
    "validate_values",
    "validate_positive",
    "validate_in_range",
]
