"""
Indicator Kriging implementation

Indicator Kriging transforms continuous data into indicator (binary) variables
and uses kriging to estimate the probability of exceeding a threshold.

Based on:
- Zhang, Y. (2010). Introduction to Geostatistics - Course Notes, Chapter 6
- Journel, A.G. (1983). Nonparametric estimation of spatial distributions
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..math.numerical import cross_validation_score

class IndicatorKriging(BaseKriging):
 Indicator Kriging for probability estimation

 Transforms data into indicators and estimates the probability
 that a value exceeds a given threshold.

 I(x; z_c) = 1 if z(x) > z_c else 0

 The kriging estimate provides P{Z(x₀) > z_c}
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     threshold: float,
     variogram_model: Optional[object] = None,
     ):
     """
     Initialize Indicator Kriging

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of sample points
     z : np.ndarray
     Values at sample points
     threshold : float
     Threshold value for indicator transformation
     variogram_model : VariogramModelBase, optional
     Fitted variogram model for indicators
     """
     # Transform to indicators
     indicators = (z > threshold).astype(np.float64)

     super().__init__(x, y, indicators, variogram_model)

     # Validate inputs
     self.x, self.y = validate_coordinates(x, y)
     self.z_original = validate_values(z, n_expected=len(self.x))
     self.z = indicators # Indicator values (0 or 1)
     self.threshold = threshold

     # Build kriging matrix
     if self.variogram_model is not None:
     if self.variogram_model is not None:

 def _build_kriging_matrix(self) -> None:
     # Calculate pairwise distances
     dist_matrix = euclidean_distance(self.x, self.y, self.x, self.y)

 # Get variogram values
 gamma_matrix = self.variogram_model(dist_matrix)

 # Build augmented matrix
 n = self.n_points
 self.kriging_matrix = np.zeros((n + 1, n + 1))
 self.kriging_matrix[:n, :n] = gamma_matrix
 self.kriging_matrix[:n, n] = 1.0
 self.kriging_matrix[n, :n] = 1.0
 self.kriging_matrix[n, n] = 0.0

 # Regularize
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
     Perform Indicator Kriging prediction

     Returns probability of exceeding the threshold.

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates for prediction
     return_variance : bool
     Whether to return kriging variance

     Returns
     -------
     probabilities : np.ndarray
     Estimated probabilities P{Z(x₀) > threshold}
     variance : np.ndarray or None
     Kriging variance (if return_variance=True)
     """
     if self.variogram_model is None:
     if self.variogram_model is None:

     x_pred, y_pred = validate_coordinates(x, y)
     n_pred = len(x_pred)

     probabilities = np.zeros(n_pred)
     variances = np.zeros(n_pred) if return_variance else None

     # Predict at each location
     for i in range(n_pred):
     for i in range(n_pred):
     dist_to_samples = euclidean_distance(
     np.array([x_pred[i]]),
     np.array([y_pred[i]]),
     self.x,
     self.y,
     ).flatten()

     # Variogram vector
     gamma_vec = self.variogram_model(dist_to_samples)

     # Augmented right-hand side
     rhs = np.zeros(self.n_points + 1)
     rhs[:self.n_points] = gamma_vec
     rhs[self.n_points] = 1.0

     # Solve for weights
     try:
     try:
     except KrigingError:
     # Fallback: use nearest neighbor
     nearest_idx = np.argmin(dist_to_samples)
     probabilities[i] = self.z[nearest_idx]
     if return_variance:
     if return_variance:
     continue

     weights = solution[:self.n_points]
     lagrange = solution[self.n_points]

     # Indicator kriging prediction (probability)
     probabilities[i] = np.dot(weights, self.z)

     # Ensure probability is in [0, 1]
     probabilities[i] = np.clip(probabilities[i], 0.0, 1.0)

     # Kriging variance
     if return_variance:
     if return_variance:
     variances[i] = max(0.0, variances[i])

     if return_variance:
     if return_variance:
     else:
     else:

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform leave-one-out cross-validation

 Returns
 -------
 predictions : np.ndarray
 Cross-validated probability predictions
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
 z_train = self.z_original[mask]

 # Create temporary kriging object
 ik_temp = IndicatorKriging(
 x_train,
 y_train,
 z_train,
 threshold=self.threshold,
 variogram_model=self.variogram_model,
 )

 # Predict at left-out point
 pred, _ = ik_temp.predict(
 np.array([self.x[i]]),
 np.array([self.y[i]]),
 return_variance=False,
 )
 predictions[i] = pred[0]

 # Calculate metrics (comparing probabilities to indicators)
 metrics = cross_validation_score(self.z, predictions)

 return predictions, metrics

class MultiThresholdIndicatorKriging:
 Multiple Indicator Kriging for full CDF estimation

 Estimates the complete cumulative distribution function (CDF)
 by performing indicator kriging at multiple threshold values.
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     thresholds: Optional[List[float]] = None,
     n_thresholds: int = 5,
     ):
     """
     Initialize Multiple Indicator Kriging

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of sample points
     z : np.ndarray
     Values at sample points
     thresholds : list of float, optional
     Threshold values. If None, uses quantiles.
     n_thresholds : int
     Number of thresholds if not explicitly provided
     """
     self.x = np.asarray(x, dtype=np.float64)
     self.y = np.asarray(y, dtype=np.float64)
     self.z = np.asarray(z, dtype=np.float64)

     # Define thresholds
     if thresholds is None:
     if thresholds is None:
     quantiles = np.linspace(0.1, 0.9, n_thresholds)
     self.thresholds = np.quantile(z, quantiles)
     else:
     else:

     self.kriging_objects = []
     self.variogram_models = []

 def fit(self) -> None:
     Fit indicator variogram models for each threshold
     """
     from ..algorithms.variogram import experimental_variogram
     from ..algorithms.fitting import fit_variogram_model
     from ..models.variogram_models import SphericalModel

 for threshold in self.thresholds:
     indicators = (self.z > threshold).astype(np.float64)

 # Calculate experimental variogram for indicators
 try:
     self.x, self.y, indicators, n_lags=10
 )

 # Fit variogram model
 model = SphericalModel()
 model = fit_variogram_model(model, lags, gamma, weights=n_pairs)

 except Exception:
 # Fallback: use simple model
 model = SphericalModel(nugget=0.1, sill=0.25, range_param=50.0)

 # Create indicator kriging object
 ik = IndicatorKriging(
 self.x, self.y, self.z,
 threshold=threshold,
 variogram_model=model
 )

 self.kriging_objects.append(ik)
 self.variogram_models.append(model)

 def predict_cdf(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     ) -> Tuple[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
     """
     Predict full CDF at locations

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates for prediction

     Returns
     -------
     cdfs : list of np.ndarray
     CDF values at each threshold for each location
     thresholds : np.ndarray
     Threshold values
     """
     if not self.kriging_objects:
     if not self.kriging_objects:

     n_pred = len(x)
     cdfs = []

     for ik in self.kriging_objects:
     for ik in self.kriging_objects:
     cdfs.append(probs)

     return cdfs, self.thresholds

 def predict_quantile(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     quantile: float = 0.5,
     ) -> npt.NDArray[np.float64]:
     """
     Predict specific quantile (e.g., median) at locations

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates for prediction
     quantile : float
     Quantile to predict (0-1), e.g., 0.5 for median

     Returns
     -------
     np.ndarray
     Predicted quantile values
     """
     cdfs, thresholds = self.predict_cdf(x, y)

     n_pred = len(x)
     quantile_values = np.zeros(n_pred)

     # Interpolate CDF to find quantile
     for i in range(n_pred):
     for i in range(n_pred):
     # Ensure monotonicity
     cdf_values = np.maximum.accumulate(cdf_values)
     quantile_values[i] = np.interp(quantile, cdf_values, thresholds)

     return quantile_values
