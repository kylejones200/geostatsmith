"""
Bootstrap Uncertainty Estimation
=================================

Bootstrap methods for uncertainty quantification.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Optional
from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..models.base_model import VariogramModelBase
from ..algorithms.variogram import experimental_variogram
from ..algorithms.fitting import fit_variogram_model as fit_variogram
import logging

logger = logging.getLogger(__name__)

def bootstrap_uncertainty(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 n_bootstrap: int = 100,
 confidence_level: float = 0.95,
 method: str = 'residual',
    ) -> Dict[str, npt.NDArray[np.float64]]:
 """
 Estimate uncertainty using bootstrap resampling.

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
 Fitted variogram model
 n_bootstrap : int, default=100
 Number of bootstrap iterations
 confidence_level : float, default=0.95
 Confidence level for intervals
 method : str, default='residual'
 Bootstrap method: 'residual' or 'pairs'
 - 'residual': Bootstrap residuals (recommended)
 - 'pairs': Bootstrap data pairs

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'mean': Mean prediction across bootstrap samples
 - 'std': Standard deviation
 - 'lower_bound': Lower confidence bound
 - 'upper_bound': Upper confidence bound
 - 'percentile_2.5': 2.5th percentile
 - 'percentile_97.5': 97.5th percentile
 - 'all_predictions': All bootstrap predictions (n_bootstrap, n_pred)

 Examples
 --------
 >>> from geostats.uncertainty import bootstrap_uncertainty
 >>> from geostats.models.variogram_models import SphericalModel
 >>>
 >>> # Fit model
 >>> model = SphericalModel(nugget=0.1, sill=1.0, range_param=50)
 >>>
 >>> # Bootstrap uncertainty
 >>> results = bootstrap_uncertainty(
 ... x, y, z,
 ... x_pred, y_pred,
 ... variogram_model=model,
 ... n_bootstrap=200
 ... )
 >>>
 >>> # Plot with confidence bands
 >>> plt.plot(x_pred, results['mean'], 'b-', label='Mean')
 >>> plt.fill_between(
 ... x_pred,
 ... results['lower_bound'],
 ... results['upper_bound'],
 ... alpha=0.3,
 ... label='95% CI'
 ... )

 Notes
 -----
 Bootstrap provides non-parametric confidence intervals that don't
 assume Gaussian errors.

 References
 ----------
 Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
 Chapman and Hall/CRC.
 """
 n = len(x)
 n_pred = len(x_pred)
 all_predictions = np.zeros((n_bootstrap, n_pred))

 # Fit initial model
 krig = OrdinaryKriging(
 x=x,
 y=y,
 z=z,
 variogram_model=variogram_model,
 )
 initial_pred, _ = krig.predict(x_pred, y_pred, return_variance=True)

 if method == 'residual':
 if method == 'residual':
 z_fitted, _ = krig.predict(x, y, return_variance=True)
 residuals = z - z_fitted

 # Bootstrap residuals
 for i in range(n_bootstrap):
 for i in range(n_bootstrap):
 resampled_residuals = np.random.choice(residuals, size=n, replace=True)
 z_bootstrap = z_fitted + resampled_residuals

 # Fit kriging to bootstrap sample
 krig_boot = OrdinaryKriging(
 x=x,
 y=y,
 z=z_bootstrap,
 variogram_model=variogram_model,
 )

 # Predict
 pred, _ = krig_boot.predict(x_pred, y_pred, return_variance=True)
 all_predictions[i, :] = pred

 elif method == 'pairs':
 elif method == 'pairs':
 for i in range(n_bootstrap):
 for i in range(n_bootstrap):
 indices = np.random.choice(n, size=n, replace=True)
 x_boot = x[indices]
 y_boot = y[indices]
 z_boot = z[indices]

 # Fit kriging to bootstrap sample
 krig_boot = OrdinaryKriging(
 x=x_boot,
 y=y_boot,
 z=z_boot,
 variogram_model=variogram_model,
 )

 # Predict
 pred, _ = krig_boot.predict(x_pred, y_pred, return_variance=True)
 all_predictions[i, :] = pred

 else:
 else:

 # Compute statistics
 mean_pred = np.mean(all_predictions, axis=0)
 std_pred = np.std(all_predictions, axis=0)

 # Confidence intervals (percentile method)
 alpha = 1 - confidence_level
 lower_percentile = 100 * (alpha / 2)
 upper_percentile = 100 * (1 - alpha / 2)

 lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
 upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)

 return {
 'mean': mean_pred,
 'std': std_pred,
 'lower_bound': lower_bound,
 'upper_bound': upper_bound,
 'percentile_2.5': np.percentile(all_predictions, 2.5, axis=0),
 'percentile_97.5': np.percentile(all_predictions, 97.5, axis=0),
 'all_predictions': all_predictions,
 'confidence_level': confidence_level,
 }

def bootstrap_variogram(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 model_type: str,
 n_bootstrap: int = 100,
 n_lags: int = 15,
    ) -> Dict[str, any]:
 """
 Bootstrap confidence intervals for variogram parameters.

 Parameters
 ----------
 x : ndarray
 X coordinates
 y : ndarray
 Y coordinates
 z : ndarray
 Values
 model_type : str
 Variogram model type ('spherical', 'exponential', etc.)
 n_bootstrap : int, default=100
 Number of bootstrap iterations
 n_lags : int, default=15
 Number of lags for variogram

 Returns
 -------
 results : dict
 Dictionary containing bootstrap statistics for variogram parameters

 Examples
 --------
 >>> results = bootstrap_variogram(
 ... x, y, z,
 ... model_type='spherical',
 ... n_bootstrap=200
 ... )
 >>> logger.info(f"Range: {results['range_mean']:.2f} ± {results['range_std']:.2f}")
 """
 n = len(x)

 # Storage for parameters
 nuggets = []
 sills = []
 ranges = []

 for i in range(n_bootstrap):
 for i in range(n_bootstrap):
 indices = np.random.choice(n, size=n, replace=True)
 x_boot = x[indices]
 y_boot = y[indices]
 z_boot = z[indices]

 try:
 try:
 lags, gamma = experimental_variogram(
 x_boot, y_boot, z_boot,
 n_lags=n_lags
 )

 # Fit model
 model = fit_variogram(lags, gamma, model_type=model_type)

 # Extract parameters
 params = model.get_parameters()
 nuggets.append(params['nugget'])
 sills.append(params['sill'])
 ranges.append(params['range'])
 except Exception:
 # Skip failed fits
 continue

 nuggets = np.array(nuggets)
 sills = np.array(sills)
 ranges = np.array(ranges)

 return {
 'nugget_mean': np.mean(nuggets),
 'nugget_std': np.std(nuggets),
 'nugget_ci': (np.percentile(nuggets, 2.5), np.percentile(nuggets, 97.5)),
 'sill_mean': np.mean(sills),
 'sill_std': np.std(sills),
 'sill_ci': (np.percentile(sills, 2.5), np.percentile(sills, 97.5)),
 'range_mean': np.mean(ranges),
 'range_std': np.std(ranges),
 'range_ci': (np.percentile(ranges, 2.5), np.percentile(ranges, 97.5)),
 'n_successful': len(nuggets),
 'n_bootstrap': n_bootstrap,
 }

def bootstrap_kriging(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 model_type: str = 'spherical',
 n_bootstrap: int = 100,
    ) -> Dict[str, npt.NDArray[np.float64]]:
 """
 Bootstrap kriging with variogram uncertainty.

 Resamples data, refits variogram each time, and performs kriging.
 Accounts for both data and model uncertainty.

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
 model_type : str, default='spherical'
 Variogram model type
 n_bootstrap : int, default=100
 Number of bootstrap iterations

 Returns
 -------
 results : dict
 Bootstrap statistics for predictions

 Examples
 --------
 >>> results = bootstrap_kriging(
 ... x, y, z,
 ... x_pred, y_pred,
 ... model_type='exponential',
 ... n_bootstrap=200
 ... )
 """
 n = len(x)
 n_pred = len(x_pred)
 all_predictions = np.zeros((n_bootstrap, n_pred))

 for i in range(n_bootstrap):
 for i in range(n_bootstrap):
 indices = np.random.choice(n, size=n, replace=True)
 x_boot = x[indices]
 y_boot = y[indices]
 z_boot = z[indices]

 try:
 try:
 lags, gamma = experimental_variogram(x_boot, y_boot, z_boot)
 model = fit_variogram(lags, gamma, model_type=model_type)

 # Kriging
 krig = OrdinaryKriging(
 x=x_boot,
 y=y_boot,
 z=z_boot,
 variogram_model=model,
 )

 pred, _ = krig.predict(x_pred, y_pred, return_variance=True)
 all_predictions[i, :] = pred
 except Exception:
 # Use NaN for failed iterations
 all_predictions[i, :] = np.nan

 # Remove failed iterations
 valid_mask = ~np.isnan(all_predictions).any(axis=1)
 all_predictions = all_predictions[valid_mask, :]

 # Compute statistics
 return {
 'mean': np.mean(all_predictions, axis=0),
 'std': np.std(all_predictions, axis=0),
 'lower_bound': np.percentile(all_predictions, 2.5, axis=0),
 'upper_bound': np.percentile(all_predictions, 97.5, axis=0),
 'all_predictions': all_predictions,
 'n_successful': all_predictions.shape[0],
 }
