"""
Core foundation layer - base classes, types, exceptions, validators
"""

from .base import BaseModel, BaseKriging
from .exceptions import (
 GeoStatsError,
 ValidationError,
 FittingError,
 KrigingError,
 ConvergenceError,
)
from .types import (
 ArrayLike,
 CoordinatesType,
 ValuesType,
)
from .validators import (
 validate_coordinates,
 validate_values,
 validate_positive,
 validate_in_range,
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
