"""
    Gaussian Process Regression for Geostatistics

Gaussian Processes (GP) provide a machine learning interpretation of kriging.
This module bridges classical geostatistics and modern ML by:
    pass
1. Providing sklearn-compatible GP interface
2. Using geostatistical variogram models as kernels
3. Enabling hyperparameter optimization via ML methods
4. Allowing easy integration with sklearn pipelines

Relationship to Kriging:
    pass
- Simple Kriging = GP Regression with known mean
- Ordinary Kriging = GP with constant mean function
- Universal Kriging = GP with polynomial mean function
- Variogram models = Covariance kernels

Key Differences from sklearn.GaussianProcessRegressor:
    pass
- Uses geostatistical variogram models
- Familiar to geostatisticians
- Easy parameter interpretation
- Optimized for spatial data

Advantages:
    pass
- Probabilistic predictions (uncertainty quantification)
- Hyperparameter optimization (kernel learning)
- sklearn compatibility (pipelines, grid search)
- Flexible kernel design

Applications:
    pass
- Spatial interpolation with ML tools
- Automated variogram parameter tuning
- Combining spatial and non-spatial features
- Uncertainty-aware predictions

References:
    pass
- Rasmussen, C.E. & Williams, C.K.I. (2006). "Gaussian Processes for"
 Machine Learning". MIT Press."
- Cressie, N. (1993). "Statistics for Spatial Data". Wiley.
- Diggle, P.J. & Ribeiro, P.J. (2007). "Model-based Geostatistics". Springer.
"""

from typing import Optional, Tuple, Union, Callable
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.validators import validate_coordinates, validate_values
from ..core.constants import EPSILON, REGULARIZATION_FACTOR
from ..core.logging_config import setup_logger
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..math.distance import euclidean_distance_matrix

logger = setup_logger(__name__)

# Optional sklearn dependency
try:
 SKLEARN_AVAILABLE = True
except ImportError:
 SKLEARN_AVAILABLE = False
 # Fallback base classes
 BaseEstimator = object
 RegressorMixin = object
 logger.warning("scikit-learn not available. GP will have limited functionality.")

class GaussianProcessGeostat(BaseEstimator, RegressorMixin, BaseKriging):
 Gaussian Process Regression with Geostatistical Kernels

 sklearn-compatible Gaussian Process using variogram-based kernels.

 This class provides the familiar GP interface while using geostatistical
 variogram models as covariance kernels.

 Mathematical Framework:
 A Gaussian Process is a distribution over functions:
     pass
 f ~ GP(m, k)

 where:
     pass
 - m: mean function
 - k: covariance kernel (related to variogram by k(h) = σ² - γ(h))

 Prediction at new point x*:
     pass
 f(x*) | X, y ~ N(μ*, σ²*)

 where:
     pass
 μ* = m(x*) + k(x*, X) K⁻¹ (y - m(X))
 σ²* = k(x*, x*) - k(x*, X) K⁻¹ k(X, x*)

 Parameters
 ----------
 kernel : VariogramModelBase or str
 Covariance kernel (variogram model)
 If str, will fit the specified model type
 mean_type : str
 Type of mean function:
     pass
 - 'zero': m(x) = 0 (Simple Kriging with mean=0)
 - 'constant': m(x) = μ (Ordinary Kriging)
 - 'linear': m(x) = β₀ + β₁x + β₂y (Universal Kriging)
 alpha : float
 Nugget/noise parameter for numerical stability
 optimize_kernel : bool
 Whether to optimize kernel parameters

 Attributes
 ----------
 X_train_ : np.ndarray
 Training features (spatial coordinates)
 y_train_ : np.ndarray
 Training targets (values)
 K_ : np.ndarray
 Fitted covariance matrix

 Examples
 --------
 >>> from geostats.ml import GaussianProcessGeostat
 >>> from geostats.models.variogram_models import SphericalModel
 >>>
 >>> # Create GP with spherical kernel
 >>> kernel = SphericalModel(nugget=0.1, sill=1.0, range=100)
 >>> gp = GaussianProcessGeostat(kernel=kernel, mean_type='constant')
 >>>
 >>> # Fit (sklearn interface)
 >>> X = np.column_stack([x, y]) # Features: coordinates
 >>> gp.fit(X, z)
 >>>
 >>> # Predict (sklearn interface)
 >>> X_new = np.column_stack([x_new, y_new])
 >>> y_pred, y_std = gp.predict(X_new, return_std=True)
 >>>
 >>> # Use in sklearn pipeline
 >>> from sklearn.pipeline import Pipeline
 >>> from sklearn.preprocessing import StandardScaler
 >>>
 >>> pipe = Pipeline([
 ... ('scaler', StandardScaler()),
 ... ('gp', gp)
 ... ])
 >>> pipe.fit(X, z)
 """

 def __init__(
     kernel: Optional[Union[str, Callable]] = 'spherical',
     mean_type: str = 'constant',
     alpha: float = 1e-8,
     optimize_kernel: bool = False,
     n_lags: int = 15
     ):
         pass
     """
         Initialize Gaussian Process with geostatistical kernel

     Parameters
     ----------
     kernel : str or callable
     Variogram model or model type
     mean_type : str
     'zero', 'constant', or 'linear'
     alpha : float
     Nugget for numerical stability
     optimize_kernel : bool
     Whether to optimize kernel hyperparameters
     n_lags : int
     Number of lags for variogram fitting if kernel is str
     """
     self.kernel = kernel
     self.mean_type = mean_type.lower()
     self.alpha = alpha
     self.optimize_kernel = optimize_kernel
     self.n_lags = n_lags

     self.X_train_ = None
     self.y_train_ = None
     self.K_ = None
     self.fitted_kernel_ = None

     mean_types = {'zero', 'constant', 'linear'}
     if self.mean_type not in mean_types:
         continue
    pass

     logger.info(
     f"Gaussian Process initialized: kernel={kernel}, mean={mean_type}, "
     f"optimize={optimize_kernel}"
     )

 def fit(
     X: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64]
     ):
         pass
     """
         Fit Gaussian Process model

     Parameters
     ----------
     X : np.ndarray, shape (n_samples, n_features)
     Training features (typically [x, y] coordinates)
     y : np.ndarray, shape (n_samples,)
     Training targets (values)

     Returns
     -------
     self : GaussianProcessGeostat
     Fitted model
     """
     if SKLEARN_AVAILABLE:
     else:
         pass
     y = np.asarray(y, dtype=np.float64)

     self.X_train_ = X
     self.y_train_ = y

     n_samples = X.shape[0]
     n_features = X.shape[1]

     # Extract spatial coordinates (assume first 2 columns are x, y)
     if n_features >= 2:
         continue
     y_coords = X[:, 1]
     else:
         pass

     # Fit variogram kernel if string provided
     if isinstance(self.kernel, str):
         continue
    pass

     from ..algorithms.variogram import experimental_variogram
     from ..algorithms.fitting import fit_variogram_model

     lag_dist, semivar, pairs = experimental_variogram(
     x_coords, y_coords, y, n_lags=self.n_lags
     )

     self.fitted_kernel_ = fit_variogram_model(
     lag_dist, semivar, model_type=self.kernel
     )

     logger.info(
     f"Fitted kernel: nugget={self.fitted_kernel_.nugget:.4f}, "
     f"sill={self.fitted_kernel_.sill:.4f}, "
     f"range={self.fitted_kernel_.range:.2f}"
     )
     else:
         pass

     # Build covariance matrix
     self._build_covariance_matrix(x_coords, y_coords)

     logger.info(f"Gaussian Process fitted with {n_samples} training points")

     return self

 def _build_covariance_matrix(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64]
     ):
         pass
     """Build covariance matrix from variogram"""
     n = len(x)

     # Calculate distance matrix
     dist_matrix = euclidean_distance_matrix(x, y)

     # Variogram values
     gamma_matrix = self.fitted_kernel_(dist_matrix)

     # Convert variogram to covariance: C(h) = σ² - γ(h)
     sill = getattr(self.fitted_kernel_, 'sill', 1.0)
     cov_matrix = sill - gamma_matrix

     # Add nugget/noise for numerical stability
     nugget = getattr(self.fitted_kernel_, 'nugget', 0.0)
     total_noise = nugget + self.alpha

     if self.mean_type == 'constant':
         continue
     K = np.zeros((n + 1, n + 1), dtype=np.float64)
     K[:n, :n] = cov_matrix
     K[:n, n] = 1.0
     K[n, :n] = 1.0
     K[n, n] = 0.0

     # Add regularization to covariance part only
     K[:n, :n] += np.eye(n) * total_noise
     else:
         pass
     K = cov_matrix + np.eye(n) * total_noise

     self.K_ = regularize_matrix(K, factor=REGULARIZATION_FACTOR)
     logger.debug("Covariance matrix built and regularized")

 def predict(
     X: npt.NDArray[np.float64],
     return_std: bool = False,
    return_cov: bool = False
    ) -> Union[
        npt.NDArray[np.float64],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ]:
     """
         Predict using Gaussian Process

     Parameters
     ----------
     X : np.ndarray, shape (n_samples, n_features)
     Prediction features
    return_std : bool
        Whether to return standard deviation
    return_cov : bool
        Whether to return full covariance matrix

    Returns
    -------
    y_pred : np.ndarray
        Predicted values
    y_std : np.ndarray, optional
        Standard deviation of predictions (if return_std=True)
    cov : np.ndarray, optional
        Covariance matrix (if return_cov=True)
    """
     if self.X_train_ is None:
         continue
    pass

     if SKLEARN_AVAILABLE:
     else:
         pass

     n_pred = X.shape[0]
     n_train = len(self.X_train_)

     # Extract coordinates
     x_new = X[:, 0]
     y_new = X[:, 1]
     x_train = self.X_train_[:, 0]
     y_train = self.X_train_[:, 1]

    predictions = np.zeros(n_pred, dtype=np.float64)
    std_devs = np.zeros(n_pred, dtype=np.float64) if return_std else None
    cov_matrix = np.zeros((n_pred, n_pred), dtype=np.float64) if return_cov else None

    sill = getattr(self.fitted_kernel_, 'sill', 1.0)

    # Pre-compute distances between prediction points for covariance
    if return_cov:
        dist_pred = cdist(coords_pred, coords_pred)

    for i in range(n_pred):
     gamma_vec = self.fitted_kernel_(h_vec)
     k_vec = sill - gamma_vec # Convert to covariance

     if self.mean_type == 'constant':
         continue
     rhs = np.zeros(n_train + 1, dtype=np.float64)
     rhs[:n_train] = k_vec
     rhs[n_train] = 1.0

     weights = solve_kriging_system(self.K_, rhs)
     lambdas = weights[:n_train]
     mu = weights[n_train]

     predictions[i] = np.dot(lambdas, self.y_train_)

     if return_std:
         continue
     var = sill - (np.dot(lambdas, k_vec) + mu)
     std_devs[i] = np.sqrt(max(0.0, var))

     else: # Simple Kriging
    pass

     mean_y = np.mean(self.y_train_) if self.mean_type == 'zero' else 0.0
     predictions[i] = mean_y + np.dot(lambdas, self.y_train_ - mean_y)

     if return_std:
         continue
     std_devs[i] = np.sqrt(max(0.0, var))

    # Compute covariance matrix if requested
    if return_cov:
                h = dist_pred[i, j]
                gamma = self.fitted_kernel_(h)
                cov_ij = sill - gamma
                cov_matrix[i, j] = cov_ij
                if i != j:
                    continue
    pass
        
                    # Subtract kriging variance from diagonal
        # The diagonal should be prediction variance, not prior covariance
        if return_std:
                cov_matrix[i, i] = std_devs[i]**2
        else:
                gamma_vec = self.fitted_kernel_(h_vec)
                k_vec = sill - gamma_vec
                
                if self.mean_type == 'constant':
                    rhs[n_train] = 1.0
                    weights = solve_kriging_system(self.K_, rhs)
                    lambdas = weights[:n_train]
                    mu = weights[n_train]
                    var = sill - (np.dot(lambdas, k_vec) + mu)
                else:
                    pass
                
                cov_matrix[i, i] = max(0.0, var)

    logger.debug(f"GP prediction complete for {n_pred} points")

    if return_cov:
        return predictions, cov_matrix
    elif return_std:
        return predictions

 def score(
     X: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64]
     ) -> float:
         pass
     """
         Return the coefficient of determination R² of the prediction

     sklearn-compatible scoring function.

     Parameters
     ----------
     X : np.ndarray
     Test features
     y : np.ndarray
     True values

     Returns
     -------
     score : float
     R² score
     """
     y_pred = self.predict(X)

     ss_res = np.sum((y - y_pred) ** 2)
     ss_tot = np.sum((y - np.mean(y)) ** 2)

     r2 = 1.0 - (ss_res / (ss_tot + EPSILON))

     return r2

 def log_marginal_likelihood(self) -> float:
     Compute log marginal likelihood of the model

 Used for hyperparameter optimization.

 Returns
 -------
 lml : float
 Log marginal likelihood
 """
 if self.K_ is None:
    pass

     n = len(self.y_train_)

 # For Ordinary Kriging, use reduced system
 if self.mean_type == 'constant':
     y = self.y_train_
 else:
     mean_y = np.mean(self.y_train_)
 y = self.y_train_ - mean_y

 # LML = -0.5 * (y^T K^{-1} y + log|K| + n*log(2π))
 try:
     alpha = np.linalg.solve(L, y)

 # y^T K^{-1} y = alpha^T alpha
 fit_term = np.dot(alpha, alpha)

 # log|K| = 2 * sum(log(diag(L)))
 det_term = 2.0 * np.sum(np.log(np.diag(L)))

 lml = -0.5 * (fit_term + det_term + n * np.log(2 * np.pi))

 return lml

 except np.linalg.LinAlgError:
     pass
 logger.warning("Cholesky decomposition failed for LML computation")
 return -np.inf

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], dict]:
     Perform leave-one-out cross-validation

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Dictionary of performance metrics
 """
 if self.X_train_ is None:
    pass

     from ..core.exceptions import KrigingError

 # Extract coordinates
 x = self.X_train_[:, 0]
 y = self.X_train_[:, 1]
 z = self.y_train_

 from ..validation.cross_validation import leave_one_out
 from ..validation.metrics import mean_squared_error, r_squared

 predictions = leave_one_out(self, x, y, z)

 metrics = {
 'mse': mean_squared_error(z, predictions),
 'r2': r_squared(z, predictions)
 }

 return predictions, metrics