"""
Approximate Methods for Speed
==============================

Fast approximate kriging methods for large datasets.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
from scipy.spatial import cKDTree

from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..models.base_model import VariogramModelBase

def approximate_kriging(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 max_neighbors: int = 50,
 search_radius: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Approximate kriging using local neighborhoods.

 Instead of using all samples for each prediction, uses only
 the nearest neighbors. Much faster for large datasets.

 Parameters
 ----------
 x : ndarray
 Sample X coordinates
 y : ndarray
 Sample Y coordinates
 z : ndarray
 Sample values
 x_pred : ndarray
 Prediction X coordinates
 y_pred : ndarray
 Prediction Y coordinates
 variogram_model : VariogramModelBase
 Variogram model
 max_neighbors : int, default=50
 Maximum number of neighbors to use
 search_radius : float, optional
 Search radius for neighbors. If None, uses all max_neighbors.

 Returns
 -------
 predictions : ndarray
 Predicted values
 variance : ndarray
 Kriging variance

 Examples
 --------
 >>> from geostats.performance import approximate_kriging
 >>>
 >>> # Fast approximate kriging with 30 nearest neighbors
 >>> z_pred, var = approximate_kriging(
 ... x, y, z,
 ... x_pred, y_pred,
 ... variogram_model=model,
 ... max_neighbors=30
 ... )

 Notes
 -----
 Speedup can be 10-100x for large datasets with minimal accuracy loss.
 Recommended for datasets with >1000 samples and >10000 predictions.
 """
 # Build spatial index
 sample_coords = np.column_stack([x, y])
 tree = cKDTree(sample_coords)

 n_pred = len(x_pred)
 predictions = np.zeros(n_pred)
 variance = np.zeros(n_pred)

 for i in range(n_pred):
 pred_point = np.array([x_pred[i], y_pred[i]])

 if search_radius is not None:
 indices = tree.query_ball_point(pred_point, search_radius)
 if len(indices) > max_neighbors:
 distances = np.linalg.norm(
 sample_coords[indices] - pred_point, axis=1
 )
 closest = np.argsort(distances)[:max_neighbors]
 indices = [indices[j] for j in closest]
 else:
 else:
 distances, indices = tree.query(
 pred_point, k=min(max_neighbors, len(x))
 )
 indices = indices.flatten()

 if len(indices) == 0:
 predictions[i] = np.nan
 variance[i] = np.inf
 continue

 # Local kriging
 x_local = x[indices]
 y_local = y[indices]
 z_local = z[indices]

 krig_local = OrdinaryKriging(
 x=x_local,
 y=y_local,
 z=z_local,
 variogram_model=variogram_model,
 )

 pred, var = krig_local.predict(
 np.array([x_pred[i]]),
 np.array([y_pred[i]]),
 return_variance=True
 )

 predictions[i] = pred[0]
 variance[i] = var[0]

 return predictions, variance

def coarse_to_fine(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_grid: npt.NDArray[np.float64],
 y_grid: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 coarse_factor: int = 4,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Coarse-to-fine kriging for large grids.

 First interpolates on coarse grid, then refines to full resolution.

 Parameters
 ----------
 x : ndarray
 Sample X coordinates
 y : ndarray
 Sample Y coordinates
 z : ndarray
 Sample values
 x_grid : ndarray
 Fine grid X coordinates (1D)
 y_grid : ndarray
 Fine grid Y coordinates (1D)
 variogram_model : VariogramModelBase
 Variogram model
 coarse_factor : int, default=4
 Coarsening factor (e.g., 4 = 1/4 resolution)

 Returns
 -------
 z_grid : ndarray
 2D grid of predictions
 var_grid : ndarray
 2D grid of variance

 Examples
 --------
 >>> # Fast interpolation on 1000x1000 grid
 >>> z_grid, var_grid = coarse_to_fine(
 ... x, y, z,
 ... x_grid, y_grid,
 ... variogram_model=model,
 ... coarse_factor=4 # First interpolate at 250x250
 ... )

 Notes
 -----
 Provides significant speedup (4-16x) with good accuracy for smooth fields.
 """
 # Create coarse grid
 x_coarse = x_grid[::coarse_factor]
 y_coarse = y_grid[::coarse_factor]

 x_coarse_2d, y_coarse_2d = np.meshgrid(x_coarse, y_coarse)
 x_coarse_flat = x_coarse_2d.ravel()
 y_coarse_flat = y_coarse_2d.ravel()

 # Interpolate on coarse grid
 krig = OrdinaryKriging(x=x, y=y, z=z, variogram_model=variogram_model)
 z_coarse_flat, var_coarse_flat = krig.predict(
 x_coarse_flat, y_coarse_flat, return_variance=True
 )

 z_coarse = z_coarse_flat.reshape(x_coarse_2d.shape)
 var_coarse = var_coarse_flat.reshape(x_coarse_2d.shape)

 # Upsample to fine grid using bilinear interpolation
 from scipy.interpolate import RectBivariateSpline

 interp_z = RectBivariateSpline(y_coarse, x_coarse, z_coarse)
 interp_var = RectBivariateSpline(y_coarse, x_coarse, var_coarse)

 z_grid = interp_z(y_grid, x_grid)
 var_grid = interp_var(y_grid, x_grid)

 return z_grid, var_grid
