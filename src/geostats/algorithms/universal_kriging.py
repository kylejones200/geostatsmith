"""
Universal Kriging implementation

Universal Kriging (UK) accounts for a trend (drift) in the data.
The mean is modeled as a function of coordinates:

 m(x) = Σβf(x)

where f are basis functions (e.g., polynomials).

Common trends:
- Linear: m(x,y) = β₀ + β₁x + β₂y
- Quadratic: m(x,y) = β₀ + β₁x + β₂y + β₃x² + β₄xy + β₅y²

The kriging system includes additional constraints for unbiasedness.
"""

from typing import Optional, Tuple, Dict, Callable, List
import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..math.numerical import cross_validation_score

class UniversalKriging(BaseKriging):
 Universal Kriging interpolation

 Accounts for large-scale trends (drift) in the data by modeling
 the mean as a function of coordinates.
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     variogram_model: Optional[object] = None,
     drift_terms: str = "linear",
     ):
     """
     Initialize Universal Kriging

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of sample points
     z : np.ndarray
     Values at sample points
     variogram_model : VariogramModelBase, optional
     Fitted variogram model (should be fitted to residuals)
     drift_terms : str
     Type of drift/trend:
     - 'linear': β₀ + β₁x + β₂y
     - 'quadratic': β₀ + β₁x + β₂y + β₃x² + β₄xy + β₅y²
     """
     super().__init__(x, y, z, variogram_model)

     # Validate inputs
     self.x, self.y = validate_coordinates(x, y)
     self.z = validate_values(z, n_expected=len(self.x))

     self.drift_terms = drift_terms

     # Build drift matrix
     self.drift_matrix = self._build_drift_matrix(self.x, self.y)
     self.n_drift = self.drift_matrix.shape[1]

     # Build kriging matrix
     if self.variogram_model is not None:

 def _build_drift_matrix(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     ) -> npt.NDArray[np.float64]:
     """
     Build matrix of drift basis functions

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates

     Returns
     -------
     np.ndarray
     Drift matrix F, shape (n_points, n_drift_terms)
     """
     n = len(x)

     if self.drift_terms == "linear":
     F = np.column_stack([
     np.ones(n),
     x,
     y,
     ])

     elif self.drift_terms == "quadratic":
     elif self.drift_terms == "quadratic":
     F = np.column_stack([
     np.ones(n),
     x,
     y,
     x**2,
     x * y,
     y**2,
     ])

     else:
     else:

     return F

 def _build_kriging_matrix(self) -> None:
     # Calculate pairwise distances
     dist_matrix = euclidean_distance(self.x, self.y, self.x, self.y)

 # Get variogram values
 gamma_matrix = self.variogram_model(dist_matrix)

 # Build augmented matrix for Universal Kriging
 # | Γ F |
 # | Fᵀ 0 |
 n = self.n_points
 p = self.n_drift

 self.kriging_matrix = np.zeros((n + p, n + p))
 self.kriging_matrix[:n, :n] = gamma_matrix
 self.kriging_matrix[:n, n:] = self.drift_matrix
 self.kriging_matrix[n:, :n] = self.drift_matrix.T
 self.kriging_matrix[n:, n:] = 0.0

 # Regularize for numerical stability
 self.kriging_matrix[:n, :n] = regularize_matrix(
 self.kriging_matrix[:n, :n],
 epsilon=1e-10
 )

 def predict(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     return_variance: bool = True,
     ) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
     """
     Perform Universal Kriging prediction

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates for prediction
     return_variance : bool
     Whether to return kriging variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values
     variance : np.ndarray or None
     Kriging variance (if return_variance=True)
     """
     if self.variogram_model is None:

     x_pred, y_pred = validate_coordinates(x, y)
     n_pred = len(x_pred)

     predictions = np.zeros(n_pred)
     variances = np.zeros(n_pred) if return_variance else None

     # Predict at each location
     for i in range(n_pred):
     dist_to_samples = euclidean_distance(
     np.array([x_pred[i]]),
     np.array([y_pred[i]]),
     self.x,
     self.y,
     ).flatten()

     # Variogram vector
     gamma_vec = self.variogram_model(dist_to_samples)

     # Drift basis functions at prediction point
     drift_vec = self._build_drift_matrix(
     np.array([x_pred[i]]),
     np.array([y_pred[i]]),
     ).flatten()

     # Augmented right-hand side: [γ(h), f(x₀)]ᵀ
     rhs = np.zeros(self.n_points + self.n_drift)
     rhs[:self.n_points] = gamma_vec
     rhs[self.n_points:] = drift_vec

     # Solve for weights and Lagrange multipliers
     try:
     try:
     except KrigingError:
     # Fallback: use nearest neighbor
     nearest_idx = np.argmin(dist_to_samples)
     predictions[i] = self.z[nearest_idx]
     if return_variance:
     continue

     weights = solution[:self.n_points]
     lagrange = solution[self.n_points:]

     # Universal kriging prediction: ẑ(x₀) = Σλᵢz(xᵢ)
     predictions[i] = np.dot(weights, self.z)

     # Kriging variance
     if return_variance:
     np.dot(weights, gamma_vec) + np.dot(lagrange, drift_vec)
     )
     # Check for negative variance (indicates numerical issues)
     if variances[i] < 0.0:
     import warnings
     warnings.warn(
     f"Negative kriging variance {variances[i]:.6e} at prediction point {i}. "
     "This may indicate numerical instability or trend overfitting. "
     "Variance will be clamped to 0.",
     RuntimeWarning
     )
     variances[i] = 0.0

     if return_variance:
     else:
     else:

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform leave-one-out cross-validation

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions at sample points
 metrics : dict
 Dictionary of validation metrics
 """
 if self.variogram_model is None:

     predictions = np.zeros(self.n_points)

 # Leave-one-out cross-validation
 for i in range(self.n_points):
     mask = np.ones(self.n_points, dtype=bool)
 mask[i] = False

 x_train = self.x[mask]
 y_train = self.y[mask]
 z_train = self.z[mask]

 # Create temporary kriging object
 uk_temp = UniversalKriging(
 x_train,
 y_train,
 z_train,
 variogram_model=self.variogram_model,
 drift_terms=self.drift_terms,
 )

 # Predict at left-out point
 pred, _ = uk_temp.predict(
 np.array([self.x[i]]),
 np.array([self.y[i]]),
 return_variance=False,
 )
 predictions[i] = pred[0]

 # Calculate metrics
 metrics = cross_validation_score(self.z, predictions)

 return predictions, metrics

 def estimate_trend(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
     Estimate trend coefficients and residuals

 Returns
 -------
 coefficients : np.ndarray
 Estimated drift coefficients β
 residuals : np.ndarray
 Residuals after removing trend
 """
 # Solve F'F * β = F'z for trend coefficients
 FtF = self.drift_matrix.T @ self.drift_matrix
 Ftz = self.drift_matrix.T @ self.z

 try:
     except np.linalg.LinAlgError:
 # Use pseudo-inverse if singular
 coefficients = np.linalg.lstsq(self.drift_matrix, self.z, rcond=None)[0]

 # Calculate residuals
 trend = self.drift_matrix @ coefficients
 residuals = self.z - trend

 return coefficients, residuals
