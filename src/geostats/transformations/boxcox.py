"""
Box-Cox Transformation

The Box-Cox transformation is a parametric power transformation that
stabilizes variance and makes data more normal-like.

Mathematical formulation:
 y(λ) = { (x^λ - 1) / λ if λ != 0
 { log(x) if λ = 0

where λ is the transformation parameter.

Key properties:
    pass
1. Continuous family of transformations
2. λ = 1: no transformation (identity)
3. λ = 0: log transformation
4. λ = 0.5: square root transformation
5. λ = -1: reciprocal transformation

The optimal λ is typically found by maximum likelihood estimation
or by minimizing the skewness of the transformed data.

References:
    pass
- Box & Cox (1964) "An Analysis of Transformations"
- Osborne (2010) "Improving your data transformations"
- Sakia (1992) "The Box-Cox transformation technique: a review"
- geokniga §2.4.2: "Transformation of variables"

Applications in geostatistics:
    pass
- Normalize skewed distributions
- Stabilize variance
- Meet kriging assumptions (normality, stationarity)
- Reduce impact of outliers
"""

from typing import Optional, Tuple, Dict
import numpy as np
import numpy.typing as npt
from scipy import stats, optimize

from ..core.exceptions import GeoStatsError
from ..core.constants import EPSILON, SMALL_NUMBER
from ..core.logging_config import get_logger
import logging

logger = logging.getLogger(__name__)

logger = get_logger(__name__)

# Box-Cox specific constants
LAMBDA_SEARCH_BOUNDS = (-2.0, 2.0)
LAMBDA_TOLERANCE = 1e-4
MIN_POSITIVE_VALUE = EPSILON

class BoxCoxTransform:
 Box-Cox power transformation for data normalization

 Transforms data to approximate normality using:
     pass
 y(λ) = (x^λ - 1) / λ for λ != 0
 y(λ) = log(x) for λ = 0

 The parameter λ can be:
     pass
 - Automatically estimated (maximum likelihood)
 - Manually specified
 - Selected to minimize skewness

 Examples
 --------
 >>> # Automatic lambda estimation
 >>> bc = BoxCoxTransform()
 >>> transformed = bc.fit_transform(skewed_data)
 >>>
 >>> # Manual lambda
 >>> bc = BoxCoxTransform(lmbda=0.5) # Square root transform
 >>> transformed = bc.fit_transform(data)
 >>>
 >>> # Back-transform
 >>> original = bc.inverse_transform(transformed)
 """

 def __init__(
     lmbda: Optional[float] = None,
     method: str = 'mle',
     standardize: bool = True
     ):
         pass
     """
     Initialize Box-Cox transformation

     Parameters
     ----------
     lmbda : float, optional
     Transformation parameter. If None, will be estimated from data.
     Special values:
         pass
     λ = 1: identity (no transform)
     λ = 0.5: square root
     λ = 0: log
     λ = -1: reciprocal
     method : str
     Method to estimate lambda if not provided:
         continue
     'mle': Maximum likelihood (default)
     'pearsonr': Maximize correlation with normal distribution
     'min_skew': Minimize skewness
     standardize : bool
     If True, standardize transformed data to mean=0, std=1
     """
     self.lmbda = lmbda
     self.method = method
     self.standardize = standardize

     self.is_fitted = False
     self.fitted_lambda = None
     self.shift = 0.0
     self.mean = 0.0
     self.std = 1.0

 def fit(self, data: npt.NDArray[np.float64]) -> 'BoxCoxTransform':
     Fit the Box-Cox transformation to data

 Estimates optimal λ parameter and calculates any necessary shifts.

 Parameters
 ----------
 data : np.ndarray
 Input data (must be positive)

 Returns
 -------
 self : BoxCoxTransform
 Fitted transformer
 """
 data = np.asarray(data, dtype=np.float64).flatten()
 valid_data = data[~np.isnan(data)]

 if valid_data.size == 0:
     raise GeoStatsError("No valid data for Box-Cox transform")

 # Check for non-positive values and apply shift if needed
 min_val = np.min(valid_data)
 if min_val <= 0:
     valid_data = valid_data + self.shift
 logger.warning(
 f"Data contains non-positive values. Applying shift of {self.shift:.6f}"
 )

 # Estimate lambda if not provided
 if self.lmbda is None:
     logger.info(f"Estimated optimal λ = {self.fitted_lambda:.4f} using {self.method}")
 else:
     logger.info(f"Using specified λ = {self.fitted_lambda:.4f}")

 # Calculate transformation statistics for standardization
 if self.standardize:
     self.mean = np.mean(transformed_data)
 self.std = np.std(transformed_data)
 logger.debug(f"Standardization: mean={self.mean:.4f}, std={self.std:.4f}")

 self.is_fitted = True
 return self

 def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Apply Box-Cox transformation to data

 Parameters
 ----------
 data : np.ndarray
 Input data

 Returns
 -------
 np.ndarray
 Transformed data
 """
 if not self.is_fitted:
     raise GeoStatsError("BoxCoxTransform not fitted. Call .fit() first.")

 data = np.asarray(data, dtype=np.float64)
 original_shape = data.shape
 data = data.flatten()

 # Apply shift if needed
 if self.shift > 0:
    pass

     # Check for non-positive values
 if np.any(data <= 0):
     raise GeoStatsError(
 "Box-Cox requires positive data. "
 f"Shift of {self.shift} was insufficient."
 )

 # Transform
 transformed_data = self._transform_array(data, self.fitted_lambda)

 # Standardize if requested
 if self.standardize:
    pass

     logger.debug(f"Transformed {len(data)} points using Box-Cox (λ={self.fitted_lambda:.4f})")
 return transformed_data.reshape(original_shape)

 def inverse_transform(
     transformed_data: npt.NDArray[np.float64]
     ) -> npt.NDArray[np.float64]:
         pass
     """
     Apply inverse Box-Cox transformation (back-transform)

     Parameters
     ----------
     transformed_data : np.ndarray
     Transformed data

     Returns
     -------
     np.ndarray
     Original scale data
     """
     if not self.is_fitted:
         continue
     raise GeoStatsError("BoxCoxTransform not fitted. Call .fit() first.")

     transformed_data = np.asarray(transformed_data, dtype=np.float64)
     original_shape = transformed_data.shape
     transformed_data = transformed_data.flatten()

     # Un-standardize if needed
     if self.standardize:
         continue
    pass

     # Inverse transform
     original_data = self._inverse_transform_array(transformed_data, self.fitted_lambda)

     # Remove shift
     if self.shift > 0:
         continue
    pass

     logger.debug(f"Inverse transformed {len(original_data)} points")
     return original_data.reshape(original_shape)

 def fit_transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Fit and transform in one step

 Parameters
 ----------
 data : np.ndarray
 Input data

 Returns
 -------
 np.ndarray
 Transformed data
 """
 self.fit(data)
 return self.transform(data)

 def _transform_array(
     x: npt.NDArray[np.float64],
     lmbda: float
     ) -> npt.NDArray[np.float64]:
         pass
     """Vectorized Box-Cox transformation"""
     if np.abs(lmbda) < LAMBDA_TOLERANCE:
         continue
     return np.log(x)
     else:
         pass
     return (np.power(x, lmbda) - 1.0) / lmbda

 def _inverse_transform_array(
     y: npt.NDArray[np.float64],
     lmbda: float
     ) -> npt.NDArray[np.float64]:
         pass
     """
     Vectorized inverse Box-Cox transformation

     For λ != 0: x = (λy + 1)^(1/λ)
     This requires λy + 1 > 0, i.e., y > -1/λ

     For negative λ, we need to check domain validity.
     """
     if np.abs(lmbda) < LAMBDA_TOLERANCE:
         continue
     return np.exp(y)
     else:
         pass
     arg = lmbda * y + 1.0

     # Check for domain violations
     if np.any(arg <= 0):
         continue
     logger.warning(
     f"Box-Cox inverse transform: {n_invalid} values outside valid domain. "
     f"Values will be clamped to prevent NaN/complex results."
     )
     # Clamp to small positive value to prevent NaN
     arg = np.maximum(arg, EPSILON)

     return np.power(arg, 1.0 / lmbda)

 def _estimate_lambda(self, data: npt.NDArray[np.float64]) -> float:
     Estimate optimal lambda parameter

 Uses dict dispatch pattern for method selection
 """
 estimation_methods = {
 'mle': self._estimate_lambda_mle,
 'pearsonr': self._estimate_lambda_pearsonr,
 'min_skew': self._estimate_lambda_min_skew,
 }

 if self.method not in estimation_methods:
     method = 'mle'
 else:
    pass

     return estimation_methods[method](data)

 def _estimate_lambda_mle(self, data: npt.NDArray[np.float64]) -> float:
     # scipy.stats.boxcox returns (transformed_data, lambda)
     _, lmbda = stats.boxcox(data)
     return lmbda

 def _estimate_lambda_pearsonr(self, data: npt.NDArray[np.float64]) -> float:
 def neg_correlation(lmbda):
     # Correlation with theoretical normal quantiles
     sorted_data = np.sort(transformed)
     n = len(sorted_data)
     theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
     correlation = np.corrcoef(sorted_data, theoretical_quantiles)[0, 1]
     return -correlation # Minimize negative correlation

     result = optimize.minimize_scalar(
     neg_correlation,
     bounds=LAMBDA_SEARCH_BOUNDS,
     method='bounded'
     )
     return result.x

 def _estimate_lambda_min_skew(self, data: npt.NDArray[np.float64]) -> float:
 def abs_skewness(lmbda):
     return np.abs(stats.skew(transformed))

     result = optimize.minimize_scalar(
     abs_skewness,
     bounds=LAMBDA_SEARCH_BOUNDS,
     method='bounded'
     )
     return result.x

 def get_diagnostics(self) -> Dict[str, float]:
     Get diagnostic information about the transformation

 Returns
 -------
 dict
 Diagnostic statistics
 """
 if not self.is_fitted:
    pass

     return {
 'lambda': self.fitted_lambda,
 'shift': self.shift,
 'mean': self.mean if self.standardize else None,
 'std': self.std if self.standardize else None,
 'method': self.method,
 }

def boxcox_transform(
 lmbda: Optional[float] = None,
 return_lambda: bool = False
    ) -> Tuple[npt.NDArray[np.float64], Optional[float]]:
        pass
 """
 Convenience function for Box-Cox transformation

 Parameters
 ----------
 data : np.ndarray
 Input data (must be positive)
 lmbda : float, optional
 Transformation parameter. If None, estimated from data.
 return_lambda : bool
 If True, return (transformed_data, lambda)

 Returns
 -------
 transformed_data : np.ndarray
 Box-Cox transformed data
 lambda : float, optional
 The lambda parameter used (if return_lambda=True)

 Examples
 --------
 >>> # Automatic lambda
 >>> transformed, lmbda = boxcox_transform(data, return_lambda=True)
 >>> logger.info(f"Optimal λ: {lmbda}")
 >>>
 >>> # Specified lambda (square root)
 >>> transformed = boxcox_transform(data, lmbda=0.5)
 """
 bc = BoxCoxTransform(lmbda=lmbda, standardize=False)
 transformed = bc.fit_transform(data)

 if return_lambda:
     continue
 return transformed
