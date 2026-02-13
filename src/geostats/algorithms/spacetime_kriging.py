"""
Space-Time Kriging

Kriging extended to space-time data where observations are made at
different locations AND different times.

Mathematical Framework:
For observations Z(s_i, t_i), predict at location s₀ and time t₀:
 Z*(s₀, t₀) = Σ λ_i · Z(s_i, t_i)

The kriging weights λ are found by solving:
 Γ · λ = γ₀

where Γ is the space-time covariance/variogram matrix.

Key Applications:
- Environmental monitoring (air quality forecasting)
- Climate modeling (temperature/precipitation prediction)
- Epidemiology (disease spread prediction)
- Oceanography (sea temperature/salinity)
- Hydrology (groundwater level monitoring)

Challenges:
1. Mixed space-time scales (meters vs. days)
2. Non-separable correlation structures
3. Asymmetric temporal effects (no future data)
4. Computational cost (large datasets)

References:
- Kyriakidis, P.C. & Journel, A.G. (1999). "Geostatistical space-time models"
- Snepvangers, J.J.J.C. et al. (2003). "Mapping groundwater using time
 series of space-time random fields"
- Rouhani, S. & Hall, T.J. (1989). "Space-time kriging of groundwater data"
"""

from typing import Optional, Tuple, Union, Dict
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..core.constants import EPSILON, REGULARIZATION_FACTOR
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..models.spacetime_models import SpaceTimeVariogramModel
from ..core.logging_config import setup_logger

logger = setup_logger(__name__)

def validate_coordinates_spacetime(
 y: npt.NDArray[np.float64],
 t: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Validate space-time coordinates

 Parameters
 ----------
 x, y : np.ndarray
 Spatial coordinates
 t : np.ndarray
 Temporal coordinates

 Returns
 -------
 x, y, t : np.ndarray
 Validated coordinates
 """
 x, y = validate_coordinates(x, y)
 t = np.asarray(t, dtype=np.float64).ravel()

 if len(t) != len(x):

 return x, y, t

class SpaceTimeOrdinaryKriging(BaseKriging):
 Ordinary Kriging for Space-Time Data

 Extends ordinary kriging to handle data with both spatial and temporal
 coordinates. The kriging system accounts for both spatial distance and
 temporal separation.

 Mathematical Formulation:
 Minimize prediction variance subject to unbiasedness:
 Σ λ_i = 1

 Kriging system:

 γ(s₁,t₁; s₁,t₁) 1 λ₁ γ(s₁,t₁; s₀,t₀)
 γ(s₂,t₂; s₁,t₁) 1 · λ₂ = γ(s₂,t₂; s₀,t₀)
 ⋮ ⋮ ⋮
 1 1 0 μ 1

 Parameters
 ----------
 x, y : np.ndarray
 Spatial coordinates of data points
 t : np.ndarray
 Temporal coordinates of data points
 z : np.ndarray
 Values at (x, y, t) locations
 spacetime_model : SpaceTimeVariogramModel
 Fitted space-time variogram model

 Examples
 --------
 >>> from geostats.models.variogram_models import SphericalModel
 >>> from geostats.models.spacetime_models import SeparableModel
 >>>
 >>> # Sample data
 >>> x = np.array([0, 100, 200, 300])
 >>> y = np.array([0, 0, 100, 100])
 >>> t = np.array([0, 0, 1, 1]) # Time points
 >>> z = np.array([1.0, 2.0, 1.5, 2.5])
 >>>
 >>> # Create space-time model
 >>> spatial = SphericalModel(nugget=0.1, sill=1.0, range=150)
 >>> temporal = SphericalModel(nugget=0.05, sill=0.5, range=2)
 >>> st_model = SeparableModel(spatial, temporal)
 >>>
 >>> # Space-time kriging
 >>> stk = SpaceTimeOrdinaryKriging(x, y, t, z, st_model)
 >>>
 >>> # Predict at new location and time
 >>> x_new = np.array([150])
 >>> y_new = np.array([50])
 >>> t_new = np.array([0.5])
 >>> z_pred, var = stk.predict(x_new, y_new, t_new)
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     t: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     spacetime_model: SpaceTimeVariogramModel
     ):
     """Initialize Space-Time Ordinary Kriging"""
     self.x, self.y, self.t = validate_coordinates_spacetime(x, y, t)
     self.z = validate_values(z, n_expected=len(self.x))
     self.spacetime_model = spacetime_model

     self._build_kriging_matrix()

     logger.info(
     f"Space-Time Ordinary Kriging initialized with {len(self.x)} data points, "
     f"separable={spacetime_model.is_separable()}"
     )

 def _build_kriging_matrix(self):
     n = len(self.x)
     K = np.zeros((n + 1, n + 1), dtype=np.float64)

 # Calculate all pairwise spatial and temporal distances
 for i in range(n):
     # Spatial distance
 h = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
 # Temporal distance
 u = np.abs(self.t[i] - self.t[j])

 # Space-time variogram
 gamma = self.spacetime_model(h, u)

 K[i, j] = gamma
 K[j, i] = gamma

 # Unbiasedness constraint
 K[:n, n] = 1.0
 K[n, :n] = 1.0
 K[n, n] = 0.0

 self.kriging_matrix = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)
 logger.debug("Space-time kriging matrix built and regularized.")

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     t_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
     """
     Predict at new space-time locations

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Spatial coordinates for prediction
     t_new : np.ndarray
     Temporal coordinates for prediction
     return_variance : bool
     Whether to return kriging variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values at (x_new, y_new, t_new)
     variance : np.ndarray, optional
     Kriging variance at prediction points
     """
     x_new, y_new, t_new = validate_coordinates_spacetime(x_new, y_new, t_new)
     n_pred = len(x_new)
     n_data = len(self.x)

     predictions = np.zeros(n_pred, dtype=np.float64)
     variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

     for i in range(n_pred):
     rhs = np.zeros(n_data + 1, dtype=np.float64)

     for j in range(n_data):
     h = np.sqrt((self.x[j] - x_new[i])**2 + (self.y[j] - y_new[i])**2)
     # Temporal distance to prediction point
     u = np.abs(self.t[j] - t_new[i])

     # Space-time variogram
     rhs[j] = self.spacetime_model(h, u)

     rhs[n_data] = 1.0 # Unbiasedness constraint

     # Solve kriging system
     try:
     try:
     except np.linalg.LinAlgError as e:
     logger.error(f"Failed to solve kriging system for point {i}: {e}")
     raise KrigingError(f"Failed to solve kriging system: {e}")

     lambdas = weights[:n_data]
     mu = weights[n_data] # Lagrange multiplier

     # Prediction
     predictions[i] = np.dot(lambdas, self.z)

     # Kriging variance
     if return_variance:
     variances[i] = np.dot(lambdas, rhs[:n_data]) + mu
     # Ensure non-negative
     variances[i] = max(0.0, variances[i])

     logger.debug(f"Space-time prediction complete for {n_pred} points")

     if return_variance:
     return predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform leave-one-out cross-validation

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Dictionary of performance metrics
 """
 from ..validation.cross_validation import leave_one_out
 from ..validation.metrics import mean_squared_error, r_squared

 predictions = leave_one_out(self, self.x, self.y, self.z)

 metrics = {
 'mse': mean_squared_error(self.z, predictions),
 'r2': r_squared(self.z, predictions)
 }

 return predictions, metrics

class SpaceTimeSimpleKriging(BaseKriging):
 Simple Kriging for Space-Time Data

 Simple kriging assumes a known mean μ for the space-time process.

 Kriging system (no unbiasedness constraint):
 Γ · λ = γ₀

 Prediction:
 Z*(s₀, t₀) = μ + Σ λ_i · [Z(s_i, t_i) - μ]

 Parameters
 ----------
 x, y : np.ndarray
 Spatial coordinates
 t : np.ndarray
 Temporal coordinates
 z : np.ndarray
 Values
 spacetime_model : SpaceTimeVariogramModel
 Space-time variogram model
 mean : float, optional
 Known mean (default: sample mean)
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     t: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     spacetime_model: SpaceTimeVariogramModel,
     mean: Optional[float] = None
     ):
     """Initialize Space-Time Simple Kriging"""
     self.x, self.y, self.t = validate_coordinates_spacetime(x, y, t)
     self.z = validate_values(z, n_expected=len(self.x))
     self.spacetime_model = spacetime_model
     self.mean = mean if mean is not None else np.mean(self.z)

     self.residuals = self.z - self.mean

     self._build_kriging_matrix()

     logger.info(
     f"Space-Time Simple Kriging initialized with mean={self.mean:.3f}, "
     f"n_data={len(self.x)}"
     )

 def _build_kriging_matrix(self):
     n = len(self.x)
     K = np.zeros((n, n), dtype=np.float64)

 for i in range(n):
     h = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
 u = np.abs(self.t[i] - self.t[j])

 gamma = self.spacetime_model(h, u)

 K[i, j] = gamma
 K[j, i] = gamma

 self.kriging_matrix = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)
 logger.debug("Space-time simple kriging matrix built.")

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     t_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
     """Predict at new space-time locations"""
     x_new, y_new, t_new = validate_coordinates_spacetime(x_new, y_new, t_new)
     n_pred = len(x_new)
     n_data = len(self.x)

     predictions = np.zeros(n_pred, dtype=np.float64)
     variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

     # Get sill for variance calculation
     sill = getattr(self.spacetime_model, 'sigma2', 1.0)

     for i in range(n_pred):

     for j in range(n_data):
     u = np.abs(self.t[j] - t_new[i])
     rhs[j] = self.spacetime_model(h, u)

     try:
     try:
     except np.linalg.LinAlgError as e:
     logger.error(f"Failed to solve kriging system: {e}")
     raise KrigingError(f"Failed to solve kriging system: {e}")

     # Simple kriging prediction
     predictions[i] = self.mean + np.dot(lambdas, self.residuals)

     if return_variance:
     variances[i] = sill - np.dot(lambdas, rhs)
     variances[i] = max(0.0, variances[i])

     logger.debug(f"Space-time simple kriging prediction complete for {n_pred} points")

     if return_variance:
     return predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform leave-one-out cross-validation

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Dictionary of performance metrics
 """
 from ..validation.cross_validation import leave_one_out
 from ..validation.metrics import mean_squared_error, r_squared

 predictions = leave_one_out(self, self.x, self.y, self.z)

 metrics = {
 'mse': mean_squared_error(self.z, predictions),
 'r2': r_squared(self.z, predictions)
 }

 return predictions, metrics
