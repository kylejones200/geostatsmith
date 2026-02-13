"""
Numerical methods for optimization and fitting
"""

from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit, minimize, OptimizeResult

from ..core.exceptions import FittingError, ConvergenceError

def weighted_least_squares(
 xdata: npt.NDArray[np.float64],
 ydata: npt.NDArray[np.float64],
 weights: Optional[npt.NDArray[np.float64]] = None,
 p0: Optional[npt.NDArray[np.float64]] = None,
 bounds: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (-np.inf, np.inf),
 **kwargs: Any,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Perform weighted least squares fitting

 Parameters
 ----------
 func : callable
 Model function to fit
 xdata : np.ndarray
 Independent variable data
 ydata : np.ndarray
 Dependent variable data
 weights : np.ndarray, optional
 Weights for each data point
 p0 : np.ndarray, optional
 Initial parameter guess
 bounds : tuple
 Lower and upper bounds for parameters
 **kwargs
 Additional arguments passed to scipy.optimize.curve_fit

 Returns
 -------
 params : np.ndarray
 Optimal parameters
 cov : np.ndarray
 Covariance matrix of parameters

 Raises
 ------
 FittingError
 If fitting fails
 """
 try:
 try:
 sigma = 1.0 / np.sqrt(weights)
 else:
 else:

 params, cov = curve_fit(
 func,
 xdata,
 ydata,
 p0=p0,
 sigma=sigma,
 absolute_sigma=True,
 bounds=bounds,
 **kwargs,
 )

 return params, cov

 except Exception as e:
 raise FittingError(f"Weighted least squares fitting failed: {e}")

def ordinary_least_squares(
 xdata: npt.NDArray[np.float64],
 ydata: npt.NDArray[np.float64],
 p0: Optional[npt.NDArray[np.float64]] = None,
 bounds: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (-np.inf, np.inf),
 **kwargs: Any,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Perform ordinary least squares fitting

 Parameters
 ----------
 func : callable
 Model function to fit
 xdata : np.ndarray
 Independent variable data
 ydata : np.ndarray
 Dependent variable data
 p0 : np.ndarray, optional
 Initial parameter guess
 bounds : tuple
 Lower and upper bounds for parameters
 **kwargs
 Additional arguments passed to scipy.optimize.curve_fit

 Returns
 -------
 params : np.ndarray
 Optimal parameters
 cov : np.ndarray
 Covariance matrix of parameters
 """
 return weighted_least_squares(func, xdata, ydata, weights=None, p0=p0, bounds=bounds, **kwargs)

def optimize_parameters(
 x0: npt.NDArray[np.float64],
 bounds: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None,
 method: str = "L-BFGS-B",
 **kwargs: Any,
    ) -> OptimizeResult:
 """
 Optimize parameters using scipy.optimize.minimize

 Parameters
 ----------
 objective : callable
 Objective function to minimize
 x0 : np.ndarray
 Initial parameter values
 bounds : tuple, optional
 Parameter bounds as (lower, upper)
 method : str
 Optimization method
 **kwargs
 Additional arguments passed to scipy.optimize.minimize

 Returns
 -------
 OptimizeResult
 Optimization result object

 Raises
 ------
 ConvergenceError
 If optimization fails to converge
 """
 result = minimize(
 objective,
 x0,
 bounds=bounds,
 method=method,
 **kwargs,
 )

 if not result.success:

 return result

def cross_validation_score(
 y_pred: npt.NDArray[np.float64],
    ) -> Dict[str, float]:
 """
 Calculate cross-validation metrics

 Parameters
 ----------
 y_true : np.ndarray
 True values
 y_pred : np.ndarray
 Predicted values

 Returns
 -------
 dict
 Dictionary containing:
 - 'mse': Mean squared error
 - 'rmse': Root mean squared error
 - 'mae': Mean absolute error
 - 'r2': R-squared coefficient
 - 'bias': Mean error (bias)
 """
 errors = y_true - y_pred

 mse = np.mean(errors**2)
 rmse = np.sqrt(mse)
 mae = np.mean(np.abs(errors))
 bias = np.mean(errors)

 # R-squared
 ss_res = np.sum(errors**2)
 ss_tot = np.sum((y_true - np.mean(y_true))**2)
 r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

 return {
 "mse": mse,
 "rmse": rmse,
 "mae": mae,
 "r2": r2,
 "bias": bias,
 }

def compute_weights(
 n_lags: int,
    ) -> npt.NDArray[np.float64]:
 """
 Compute weights for variogram fitting based on number of pairs per lag

 Parameters
 ----------
 distances : np.ndarray
 All pairwise distances
 n_lags : int
 Number of lag bins

 Returns
 -------
 np.ndarray
 Weights for each lag
 """
 max_dist = np.max(distances)
 lag_bins = np.linspace(0, max_dist, n_lags + 1)

 weights = np.zeros(n_lags)
 for i in range(n_lags):
 weights[i] = np.sum(mask)

 # Normalize weights
 weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights

 return weights
