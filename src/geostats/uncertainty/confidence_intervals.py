"""
Confidence Intervals and Prediction Bands
==========================================

Functions for computing confidence intervals and visualization utilities.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple, Optional
from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..models.base_model import VariogramModelBase

def confidence_intervals(
def confidence_intervals(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 confidence_level: float = 0.95,
    ) -> Dict[str, npt.NDArray[np.float64]]:
 """
 Compute confidence intervals for kriging predictions.

 Uses kriging variance to construct confidence intervals
 assuming Gaussian errors.

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
 confidence_level : float, default=0.95
 Confidence level (0-1)

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'predictions': Kriging predictions
 - 'std_errors': Standard errors
 - 'lower_bound': Lower confidence bound
 - 'upper_bound': Upper confidence bound
 - 'confidence_level': Confidence level used

 Examples
 --------
 >>> from geostats.uncertainty import confidence_intervals
 >>>
 >>> ci = confidence_intervals(
 ... x, y, z,
 ... x_pred, y_pred,
 ... variogram_model=model,
 ... confidence_level=0.95
 ... )
 >>>
 >>> # Plot
 >>> plt.plot(x_pred, ci['predictions'], 'b-')
 >>> plt.fill_between(
 ... x_pred,
 ... ci['lower_bound'],
 ... ci['upper_bound'],
 ... alpha=0.3
 ... )
 """
 # Perform kriging
 krig = OrdinaryKriging(
 x=x,
 y=y,
 z=z,
 variogram_model=variogram_model,
 )

 predictions, variance = krig.predict(x_pred, y_pred, return_variance=True)
 std_errors = np.sqrt(variance)

 # Confidence interval multiplier
 from scipy.stats import norm
 z_score = norm.ppf((1 + confidence_level) / 2)

 # Confidence bounds
 margin = z_score * std_errors
 lower_bound = predictions - margin
 upper_bound = predictions + margin

 return {
 'predictions': predictions,
 'variance': variance,
 'std_errors': std_errors,
 'lower_bound': lower_bound,
 'upper_bound': upper_bound,
 'confidence_level': confidence_level,
 'z_score': z_score,
 }

def prediction_bands(
def prediction_bands(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 confidence_levels: Optional[npt.NDArray[np.float64]] = None,
    ) -> Dict[str, any]:
 """
 Compute multiple prediction bands at different confidence levels.

 Useful for visualizing uncertainty with multiple bands.

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
 confidence_levels : ndarray, optional
 Array of confidence levels (e.g., [0.68, 0.95, 0.99])
 Default: [0.68, 0.95, 0.99]

 Returns
 -------
 results : dict
 Dictionary with predictions and bands for each level

 Examples
 --------
 >>> bands = prediction_bands(
 ... x, y, z,
 ... x_pred, y_pred,
 ... variogram_model=model,
 ... confidence_levels=[0.68, 0.95, 0.99]
 ... )
 >>>
 >>> # Plot multiple bands
 >>> plt.plot(x_pred, bands['predictions'], 'b-', linewidth=2)
 >>> colors = ['lightblue', 'blue', 'darkblue']
 >>> for i, level in enumerate([0.68, 0.95, 0.99]):
 ... plt.fill_between(
 ... x_pred,
 ... bands[f'lower_{level}'],
 ... bands[f'upper_{level}'],
 ... alpha=0.2,
 ... color=colors[i],
 ... label=f'{level*100:.0f}% CI'
 ... )
 """
 if confidence_levels is None:
 if confidence_levels is None:

 # Perform kriging
 krig = OrdinaryKriging(
 x=x,
 y=y,
 z=z,
 variogram_model=variogram_model,
 )

 predictions, variance = krig.predict(x_pred, y_pred, return_variance=True)
 std_errors = np.sqrt(variance)

 # Compute bands for each level
 from scipy.stats import norm

 results = {
 'predictions': predictions,
 'std_errors': std_errors,
 'confidence_levels': confidence_levels,
 }

 for level in confidence_levels:
 for level in confidence_levels:
 margin = z_score * std_errors

 results[f'lower_{level}'] = predictions - margin
 results[f'upper_{level}'] = predictions + margin
 results[f'z_score_{level}'] = z_score

 return results

def uncertainty_ellipse(
def uncertainty_ellipse(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 x_center: float,
 y_center: float,
 confidence_level: float = 0.95,
 n_points: int = 100,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Compute uncertainty ellipse for a prediction location.

 The ellipse represents regions of equal kriging variance
 around a point.

 Parameters
 ----------
 x : ndarray
 Sample X coordinates
 y : ndarray
 Sample Y coordinates
 z : ndarray
 Sample values
 variogram_model : VariogramModelBase
 Fitted variogram model
 x_center : float
 X coordinate of center point
 y_center : float
 Y coordinate of center point
 confidence_level : float, default=0.95
 Confidence level
 n_points : int, default=100
 Number of points on ellipse

 Returns
 -------
 x_ellipse : ndarray
 X coordinates of ellipse
 y_ellipse : ndarray
 Y coordinates of ellipse

 Examples
 --------
 >>> # Plot uncertainty ellipse
 >>> x_ell, y_ell = uncertainty_ellipse(
 ... x, y, z,
 ... variogram_model=model,
 ... x_center=50,
 ... y_center=50
 ... )
 >>> plt.plot(x_ell, y_ell, 'r-', linewidth=2)

 Notes
 -----
 This is a simplified version that assumes isotropic variogram.
 For anisotropic variograms, the ellipse would need to be adjusted.
 """
 # Perform kriging at center
 krig = OrdinaryKriging(
 x=x,
 y=y,
 z=z,
 variogram_model=variogram_model,
 )

 _, variance = krig.predict(
 np.array([x_center]),
 np.array([y_center]),
 return_variance=True
 )

 std = np.sqrt(variance[0])

 # Confidence multiplier
 from scipy.stats import norm
 z_score = norm.ppf((1 + confidence_level) / 2)

 # Radius of ellipse
 radius = z_score * std

 # Generate circle (assuming isotropy)
 theta = np.linspace(0, 2*np.pi, n_points)
 x_ellipse = x_center + radius * np.cos(theta)
 y_ellipse = y_center + radius * np.sin(theta)

 return x_ellipse, y_ellipse
