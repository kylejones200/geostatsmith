"""
Grid utility functions for spatial interpolation
"""

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt

def create_grid(
def create_grid(
 x_max: float,
 y_min: float,
 y_max: float,
 nx: int = 100,
 ny: Optional[int] = None,
 resolution: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Create a regular grid for interpolation

 Parameters
 ----------
 x_min, x_max : float
 X coordinate bounds
 y_min, y_max : float
 Y coordinate bounds
 nx : int
 Number of grid points in x direction (default: 100)
 ny : int, optional
 Number of grid points in y direction (defaults to nx)
 resolution : int, optional
 Alias for nx (for backward compatibility)

 Returns
 -------
 X, Y : np.ndarray
 Meshgrid arrays of shape (ny, nx)
 """
 # Handle multiple parameter names
 if resolution is not None:
 if resolution is not None:

 if ny is None:
 if ny is None:

 x_grid = np.linspace(x_min, x_max, nx)
 y_grid = np.linspace(y_min, y_max, ny)
 X, Y = np.meshgrid(x_grid, y_grid)

 return X, Y

def interpolate_to_grid(
def interpolate_to_grid(
 x_min: float,
 x_max: float,
 y_min: float,
 y_max: float,
 resolution: int = 100,
 return_variance: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
 """
 Interpolate kriging predictions to a regular grid

 Parameters
 ----------
 kriging_obj : BaseKriging
 Fitted kriging object (SimpleKriging, OrdinaryKriging, etc.)
 x_min, x_max : float
 X coordinate bounds
 y_min, y_max : float
 Y coordinate bounds
 resolution : int
 Number of grid points per dimension
 return_variance : bool
 Whether to return kriging variance

 Returns
 -------
 X, Y : np.ndarray
 Meshgrid coordinate arrays
 Z : np.ndarray
 Interpolated values on grid
 V : np.ndarray or None
 Kriging variance on grid (if return_variance=True)
 """
 # Create grid
 X, Y = create_grid(x_min, x_max, y_min, y_max, resolution)

 # Flatten for prediction
 x_flat = X.flatten()
 y_flat = Y.flatten()

 # Predict
 z_flat, v_flat = kriging_obj.predict(x_flat, y_flat, return_variance=return_variance)

 # Reshape to grid
 Z = z_flat.reshape(X.shape)
 V = v_flat.reshape(X.shape) if return_variance else None

 return X, Y, Z, V
