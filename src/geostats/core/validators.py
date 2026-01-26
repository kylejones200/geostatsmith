"""
Input validation utilities
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

from .exceptions import ValidationError

def validate_coordinates(
 x: npt.NDArray[np.float64],
 y: Optional[npt.NDArray[np.float64]] = None,
 z: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[npt.NDArray[np.float64], ...]:
 """
 Validate and convert coordinate arrays

 Parameters
 ----------
 x : array-like
 X coordinates or combined coordinates
 y : array-like, optional
 Y coordinates
 z : array-like, optional
 Z coordinates

 Returns
 -------
 tuple of np.ndarray
 Validated coordinate arrays

 Raises
 ------
 ValidationError
 If coordinates are invalid
 """
 x = np.asarray(x, dtype=np.float64)

 if x.ndim == 0:
 raise ValidationError("Coordinates must be at least 1-dimensional")

 if y is None:
 # x contains all coordinates
 if x.ndim == 1:
 return (x,)
 else:
 return tuple(x[:, i] for i in range(x.shape[1]))

 y = np.asarray(y, dtype=np.float64)

 if x.shape != y.shape:
 raise ValidationError(
 f"X and Y coordinates must have same shape, got {x.shape} and {y.shape}"
 )

 if z is not None:
 z = np.asarray(z, dtype=np.float64)
 if x.shape != z.shape:
 raise ValidationError(
 f"All coordinates must have same shape, got {x.shape}, {y.shape}, and {z.shape}"
 )
 return x, y, z

 return x, y

def validate_values(
 values: npt.NDArray[np.float64],
 n_expected: Optional[int] = None,
 allow_nan: bool = False,
) -> npt.NDArray[np.float64]:
 """
 Validate values array

 Parameters
 ----------
 values : array-like
 Values to validate
 n_expected : int, optional
 Expected number of values
 allow_nan : bool
 Whether to allow NaN values

 Returns
 -------
 np.ndarray
 Validated values array

 Raises
 ------
 ValidationError
 If values are invalid
 """
 values = np.asarray(values, dtype=np.float64)

 if values.ndim != 1:
 raise ValidationError(f"Values must be 1-dimensional, got shape {values.shape}")

 if n_expected is not None and len(values) != n_expected:
 raise ValidationError(
 f"Expected {n_expected} values, got {len(values)}"
 )

 if not allow_nan and np.any(np.isnan(values)):
 raise ValidationError("Values contain NaN")

 if not allow_nan and np.any(np.isinf(values)):
 raise ValidationError("Values contain infinity")

 return values

def validate_positive(value: float, name: str = "value") -> float:
 """
 Validate that a value is positive

 Parameters
 ----------
 value : float
 Value to validate
 name : str
 Name of the parameter for error messages

 Returns
 -------
 float
 Validated value

 Raises
 ------
 ValidationError
 If value is not positive
 """
 if value <= 0:
 raise ValidationError(f"{name} must be positive, got {value}")
 return value

def validate_in_range(
 value: float,
 min_val: Optional[float] = None,
 max_val: Optional[float] = None,
 name: str = "value",
) -> float:
 """
 Validate that a value is within a range

 Parameters
 ----------
 value : float
 Value to validate
 min_val : float, optional
 Minimum allowed value (inclusive)
 max_val : float, optional
 Maximum allowed value (inclusive)
 name : str
 Name of the parameter for error messages

 Returns
 -------
 float
 Validated value

 Raises
 ------
 ValidationError
 If value is out of range
 """
 if min_val is not None and value < min_val:
 raise ValidationError(f"{name} must be >= {min_val}, got {value}")

 if max_val is not None and value > max_val:
 raise ValidationError(f"{name} must be <= {max_val}, got {value}")

 return value

def validate_array_shapes_match(
 *arrays: npt.NDArray,
 names: Optional[Tuple[str, ...]] = None,
) -> None:
 """
 Validate that multiple arrays have matching shapes

 Parameters
 ----------
 *arrays : np.ndarray
 Arrays to validate
 names : tuple of str, optional
 Names of arrays for error messages

 Raises
 ------
 ValidationError
 If array shapes don't match
 """
 if len(arrays) < 2:
 return

 shapes = [arr.shape for arr in arrays]
 first_shape = shapes[0]

 for i, shape in enumerate(shapes[1:], 1):
 if shape != first_shape:
 if names:
 msg = f"{names[0]} and {names[i]} have different shapes: {first_shape} vs {shape}"
 else:
 msg = f"Arrays have different shapes: {first_shape} vs {shape}"
 raise ValidationError(msg)
