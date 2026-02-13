"""
External Drift Kriging (Regression Kriging)

Implementation based on:
- geokniga-introductiontogeostatistics.txt §2935-2994, §5920-6027
- imet131-i-chapitre-5.txt (External Drift examples)

External Drift Kriging allows using external covariates (like elevation, rainfall,
temperature) that are correlated with the target variable.

From geokniga (§5974-5994):
"The equations of external drift kriging are as follows:
 Σ λj γ(ui - uj) + μ1 + μ2*Y(ui) = γ(ui - u) for i=1,...,I
 Σ λj = 1
 Σ λj*Y(uj) = Y(u)

where Y is the external covariate."

This extends Universal Kriging to handle arbitrary spatial covariates beyond
just polynomial functions of coordinates.
"""

from typing import Optional, Tuple, Union, Dict
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance_matrix, euclidean_distance
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..core.logging_config import get_logger

logger = get_logger(__name__)

class ExternalDriftKriging(BaseKriging):
 External Drift Kriging (also known as Regression Kriging)

 Uses external covariate(s) to improve estimation. The mean is modeled as
 a linear function of external variables:

 m(x) = β₀ + β₁*Y₁(x) + β₂*Y₂(x) + ...

 where Y₁, Y₂, ... are external covariates (elevation, temperature, etc.)

 Advantages over Universal Kriging:
 - Can use actual measured covariates (not just coordinates)
 - Better for variables with clear relationships to auxiliary data
 - More flexible trend modeling
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     covariates_data: npt.NDArray[np.float64],
     variogram_model: Optional[object] = None,
     ):
     """
     Initialize External Drift Kriging

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of sample points
     z : np.ndarray
     Values at sample points
     covariates_data : np.ndarray
     External covariate values at sample points
     Shape: (n_samples, n_covariates) or (n_samples,) for single covariate
     variogram_model : VariogramModelBase, optional
     Fitted variogram model (should be fitted to residuals ideally)
     """
     super().__init__(x, y, z, variogram_model)

     # Validate
     self.x, self.y = validate_coordinates(x, y)
     self.z = validate_values(z, n_expected=len(self.x))

     # Handle covariates
     covariates_data = np.asarray(covariates_data, dtype=np.float64)
     if covariates_data.ndim == 1:

     if len(covariates_data) != len(self.x):
     f"covariates_data must have same length as x,y,z. "
     f"Got {len(covariates_data)} vs {len(self.x)}"
     )

     self.covariates_data = covariates_data
     self.n_covariates = covariates_data.shape[1]

     # Build kriging matrix
     if self.variogram_model is not None:

 def _build_kriging_matrix(self) -> None:
     Build the external drift kriging system matrix

 System (from geokniga §5974-5994):
 [ Γ 1 Y ] [ λ ] [ γ₀ ]
 [ 1ᵀ 0 0 ] [ μ₁] = [ 1 ]
 [ Yᵀ 0 0 ] [ μ₂] [ y₀ ]

 where:
 - Γ is the n×n variogram matrix
 - 1 is vector of ones
 - Y is the n×n_covariates matrix of external covariate values
 - y₀ is the covariate value(s) at prediction location
 """
 n = len(self.x)
 n_cov = self.n_covariates

 logger.debug(f"Building EDK matrix: {n} points, {n_cov} covariates")

 # Build variogram matrix (vectorized)
 # Compute all pairwise distances at once
 dist_matrix = euclidean_distance_matrix(self.x, self.y)

 # Apply variogram function to all distances (vectorized)
 gamma_matrix = self.variogram_model(dist_matrix)

 # Build full kriging matrix
 # Size: (n + 1 + n_cov) × (n + 1 + n_cov)
 size = n + 1 + n_cov
 K = np.zeros((size, size))

 # Top-left: variogram matrix
 K[:n, :n] = gamma_matrix

 # Unbiasedness constraint (sum of weights = 1)
 K[:n, n] = 1.0
 K[n, :n] = 1.0

 # External drift constraints (Σ λj*Yk(uj) = Yk(u))
 for k in range(n_cov):
     K[n + 1 + k, :n] = self.covariates_data[:, k]

 # Regularize if needed
 K = regularize_matrix(K)

 self.kriging_matrix = K

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     covariates_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
     """
     Predict at new locations using external drift kriging

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Coordinates of prediction points
     covariates_new : np.ndarray
     External covariate values at prediction points
     Shape: (n_pred, n_covariates) or (n_pred,) for single covariate
     return_variance : bool
     If True, return both predictions and kriging variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values
     variance : np.ndarray, optional
     Kriging variance at each prediction point
     """
     if self.variogram_model is None:

     x_new, y_new = validate_coordinates(x_new, y_new)
     n_pred = len(x_new)
     n_data = len(self.x)
     n_cov = self.n_covariates

     # Handle covariates
     covariates_new = np.asarray(covariates_new, dtype=np.float64)
     if covariates_new.ndim == 1:

     if len(covariates_new) != n_pred:
     f"covariates_new must match length of x_new, y_new. "
     f"Got {len(covariates_new)} vs {n_pred}"
     )

     if covariates_new.shape[1] != n_cov:
     f"covariates_new must have {n_cov} columns (covariates). "
     f"Got {covariates_new.shape[1]}"
     )

     logger.debug(f"Predicting at {n_pred} locations with EDK")

     # Vectorized distance calculation for all prediction points
     # Shape: (n_pred, n_data)
     dist_to_data = euclidean_distance(
     x_new.reshape(-1, 1), y_new.reshape(-1, 1),
     self.x.reshape(1, -1), self.y.reshape(1, -1)
     )

     # Apply variogram to all distances (vectorized)
     gamma_to_data = self.variogram_model(dist_to_data)

     predictions = np.zeros(n_pred, dtype=np.float64)
     variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

     # Build and solve kriging system for each prediction point
     # Note: This loop is necessary as each point has different RHS
     for i in range(n_pred):
     rhs = np.zeros(n_data + 1 + n_cov, dtype=np.float64)
     rhs[:n_data] = gamma_to_data[i, :] # Variogram values (vectorized)
     rhs[n_data] = 1.0 # Unbiasedness constraint
     rhs[n_data + 1:] = covariates_new[i, :] # External drift constraints

     # Solve kriging system
     try:
     try:
     except np.linalg.LinAlgError as e:
     raise KrigingError(f"Failed to solve kriging system at point {i}: {e}")

     # Prediction: Σλᵢzᵢ
     predictions[i] = np.dot(weights[:n_data], self.z)

     # Variance: σ²(x₀) = Σλᵢγ(xᵢ-x₀) + μ₁ + Σμₖ₊₁Yₖ(x₀)
     if return_variance:
     variances[i] = np.dot(weights, rhs)

     if return_variance:
     return predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Leave-one-out cross-validation for external drift kriging

 Returns
 -------
 errors : np.ndarray
 Prediction errors at each sample location
 metrics : Dict[str, float]
 Error statistics (MSE, RMSE, MAE, R², bias)
 """
 n = len(self.x)
 predictions = np.zeros(n)

 for i in range(n):
     mask = np.ones(n, dtype=bool)
 mask[i] = False

 # Create temporary kriging object without point i
 edk_temp = ExternalDriftKriging(
 self.x[mask],
 self.y[mask],
 self.z[mask],
 self.covariates_data[mask],
 self.variogram_model
 )

 # Predict at left-out point
 pred = edk_temp.predict(
 np.array([self.x[i]]),
 np.array([self.y[i]]),
 self.covariates_data[i:i+1],
 return_variance=False
 )
 predictions[i] = pred[0]

 # Calculate errors
 errors = self.z - predictions

 metrics = {
 'MSE': np.mean(errors**2),
 'RMSE': np.sqrt(np.mean(errors**2)),
 'MAE': np.mean(np.abs(errors)),
 'R2': 1 - np.sum(errors**2) / np.sum((self.z - np.mean(self.z))**2),
 'bias': np.mean(errors),
 'predictions': predictions,
 }

 return errors, metrics
