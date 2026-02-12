"""
3D Kriging

Extension of kriging methods to three-dimensional space.
Common applications:
- Mining (ore grade in x, y, z coordinates)
- Hydrogeology (groundwater properties in 3D aquifers)
- Atmospheric science (temperature, pollutants at different altitudes)
- Oceanography (salinity, temperature at depth)

The mathematics are identical to 2D kriging, but distances are computed
in 3D space. Anisotropy becomes more complex with different ranges in
vertical vs horizontal directions.

References:
- Deutsch & Journel (1998) - GSLIB
- Wackernagel (2003) - Multivariate Geostatistics (Chapter 10)
- Chilès & Delfiner (2012) - Geostatistics (Chapter 3)
"""

from typing import Optional, Tuple, Union, Dict
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_values
from ..math.distance import euclidean_distance_3d, euclidean_distance_matrix_3d, euclidean_distance
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..core.constants import REGULARIZATION_FACTOR
from ..core.logging_config import get_logger

logger = get_logger(__name__)

def validate_coordinates_3d(
def validate_coordinates_3d(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """Validate 3D coordinates"""
 x = np.asarray(x, dtype=np.float64).flatten()
 y = np.asarray(y, dtype=np.float64).flatten()
 z = np.asarray(z, dtype=np.float64).flatten()

 if not (len(x) == len(y) == len(z)):
 if not (len(x) == len(y) == len(z)):

 return x, y, z

class SimpleKriging3D(BaseKriging):
class SimpleKriging3D(BaseKriging):
 Simple Kriging in 3D space

 Assumes known constant mean. Kriging equations are identical to 2D,
 but distances are computed in 3D Euclidean space.
 """

 def __init__(
 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     values: npt.NDArray[np.float64],
     variogram_model: Optional[object] = None,
     known_mean: Optional[float] = None,
     ):
     """
     Initialize 3D Simple Kriging

     Parameters
     ----------
     x, y, z : np.ndarray
     3D coordinates of sample points
     values : np.ndarray
     Values at sample points
     variogram_model : VariogramModelBase, optional
     Fitted variogram model (3D variogram)
     known_mean : float, optional
     Known mean of the random field
     """
     # Note: BaseKriging expects x, y, z as data, not coordinates
     # We'll override to handle 3D properly
     self.x, self.y, self.z = validate_coordinates_3d(x, y, z)
     self.values = validate_values(values, n_expected=len(self.x))
     self.variogram_model = variogram_model

     # Estimate mean if not provided
     if known_mean is not None:
     if known_mean is not None:
     else:
     else:

     # Center the data
     self.residuals = self.values - self.mean

     if self.variogram_model is not None:
     if self.variogram_model is not None:

 def _build_kriging_matrix(self):
 def _build_kriging_matrix(self):
     n = len(self.x)

 # Build covariance matrix (vectorized distance calculation)
 # For simple kriging: C(h) = sill - γ(h)
 sill = self.variogram_model.sill if hasattr(self.variogram_model, 'sill') else 1.0

 # Vectorized 3D distance matrix
 dist_matrix = euclidean_distance_matrix_3d(self.x, self.y, self.z)

 # Vectorized variogram evaluation
 gamma_matrix = self.variogram_model(dist_matrix)

 # Covariance = sill - variogram
 K = sill - gamma_matrix

 # Regularize if needed
 K = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)

 self.kriging_matrix = K
 logger.debug(f"3D Simple Kriging matrix built (vectorized): {n}x{n}")

 def predict(
 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     z_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
     """
     Predict at new 3D locations

     Parameters
     ----------
     x_new, y_new, z_new : np.ndarray
     3D coordinates of prediction points
     return_variance : bool
     If True, return both predictions and kriging variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values
     variance : np.ndarray, optional
     Kriging variance at each point
     """
     if self.variogram_model is None:
     if self.variogram_model is None:

     x_new, y_new, z_new = validate_coordinates_3d(x_new, y_new, z_new)
     n_pred = len(x_new)
     n_data = len(self.x)

     sill = self.variogram_model.sill if hasattr(self.variogram_model, 'sill') else 1.0

     # Vectorized distance calculation from data to prediction points
     dist_to_pred = euclidean_distance(self.x, self.y, x_new, y_new, self.z, z_new)

     # Vectorized variogram evaluation
     gamma_to_pred = self.variogram_model(dist_to_pred)

     # Covariance vectors (vectorized)
     cov_to_pred = sill - gamma_to_pred # shape: (n_data, n_pred)

     predictions = np.zeros(n_pred, dtype=np.float64)
     variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

     # Still need loop over prediction points for solving (inherent to kriging)
     for i in range(n_pred):
     for i in range(n_pred):

     # Solve for weights
     try:
     try:
     except np.linalg.LinAlgError as e:
     logger.error(f"Failed to solve 3D kriging system at point {i}: {e}")
     raise KrigingError(f"Failed to solve kriging system: {e}")

     # Prediction: m + Σλᵢ(zᵢ - m)
     predictions[i] = self.mean + np.dot(lambdas, self.residuals)

     # Variance: σ²(x₀) = C(0) - Σλᵢ·C(xᵢ-x₀)
     if return_variance:
     if return_variance:

     logger.info(f"3D Simple Kriging completed for {n_pred} prediction points (vectorized)")
     if return_variance:
     if return_variance:
     return predictions

class OrdinaryKriging3D(BaseKriging):
class OrdinaryKriging3D(BaseKriging):
 Ordinary Kriging in 3D space

 Accounts for unknown mean through Lagrange multiplier.
 Most common kriging variant for 3D applications.
 """

 def __init__(
 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     values: npt.NDArray[np.float64],
     variogram_model: Optional[object] = None,
     ):
     """
     Initialize 3D Ordinary Kriging

     Parameters
     ----------
     x, y, z : np.ndarray
     3D coordinates of sample points
     values : np.ndarray
     Values at sample points
     variogram_model : VariogramModelBase, optional
     Fitted variogram model
     """
     self.x, self.y, self.z = validate_coordinates_3d(x, y, z)
     self.values = validate_values(values, n_expected=len(self.x))
     self.variogram_model = variogram_model

     if self.variogram_model is not None:
     if self.variogram_model is not None:

 def _build_kriging_matrix(self):
 def _build_kriging_matrix(self):
     n = len(self.x)

 # Build variogram matrix with Lagrange constraint
 K = np.zeros((n + 1, n + 1), dtype=np.float64)

 # Vectorized 3D distance matrix
 dist_matrix = euclidean_distance_matrix_3d(self.x, self.y, self.z)

 # Vectorized variogram evaluation
 K[:n, :n] = self.variogram_model(dist_matrix)

 # Unbiasedness constraint
 K[:n, n] = 1.0
 K[n, :n] = 1.0
 K[n, n] = 0.0

 # Regularize
 K = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)

 self.kriging_matrix = K
 logger.debug(f"3D Ordinary Kriging matrix built (vectorized): {n+1}x{n+1}")

 def predict(
 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     z_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
     """
     Predict at new 3D locations using ordinary kriging

     Parameters
     ----------
     x_new, y_new, z_new : np.ndarray
     3D coordinates of prediction points
     return_variance : bool
     If True, return both predictions and kriging variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values
     variance : np.ndarray, optional
     Kriging variance
     """
     if self.variogram_model is None:
     if self.variogram_model is None:

     x_new, y_new, z_new = validate_coordinates_3d(x_new, y_new, z_new)
     n_pred = len(x_new)
     n_data = len(self.x)

     # Vectorized distance calculation from data to prediction points
     dist_to_pred = euclidean_distance(self.x, self.y, x_new, y_new, self.z, z_new)

     # Vectorized variogram evaluation
     gamma_to_pred = self.variogram_model(dist_to_pred) # shape: (n_data, n_pred)

     predictions = np.zeros(n_pred, dtype=np.float64)
     variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

     # Still need loop over prediction points for solving (inherent to kriging)
     for i in range(n_pred):
     for i in range(n_pred):
     rhs = np.zeros(n_data + 1, dtype=np.float64)
     rhs[:n_data] = gamma_to_pred[:, i]
     rhs[n_data] = 1.0 # Unbiasedness constraint

     # Solve
     try:
     try:
     except np.linalg.LinAlgError as e:
     logger.error(f"Failed to solve 3D OK system at point {i}: {e}")
     raise KrigingError(f"Failed to solve kriging system: {e}")

     # Extract lambda weights
     lambdas = weights[:n_data]
     mu = weights[n_data] # Lagrange multiplier

     # Prediction
     predictions[i] = np.dot(lambdas, self.values)

     # Variance
     if return_variance:
     if return_variance:

     logger.info(f"3D Ordinary Kriging completed for {n_pred} prediction points (vectorized)")
     if return_variance:
     if return_variance:
     return predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     n = len(self.x)
     predictions = np.zeros(n)

 for i in range(n):
     mask = np.ones(n, dtype=bool)
 mask[i] = False

 # Temporary kriging without point i
 ok3d_temp = OrdinaryKriging3D(
 self.x[mask],
 self.y[mask],
 self.z[mask],
 self.values[mask],
 self.variogram_model
 )

 # Predict at left-out point
 pred = ok3d_temp.predict(
 np.array([self.x[i]]),
 np.array([self.y[i]]),
 np.array([self.z[i]]),
 return_variance=False
 )
 predictions[i] = pred[0]

 # Calculate errors
 errors = self.values - predictions

 metrics = {
 'MSE': np.mean(errors**2),
 'RMSE': np.sqrt(np.mean(errors**2)),
 'MAE': np.mean(np.abs(errors)),
 'R2': 1 - np.sum(errors**2) / np.sum((self.values - np.mean(self.values))**2),
 'bias': np.mean(errors),
 'predictions': predictions,
 }

 return errors, metrics
