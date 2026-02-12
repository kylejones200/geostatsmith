"""
High-level variogram API

This module provides user-friendly functions for variogram analysis.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import numpy.typing as npt

from .algorithms.variogram import (
    experimental_variogram as _experimental_variogram,
    experimental_variogram_directional as _experimental_variogram_directional,
    variogram_cloud as _variogram_cloud,
    robust_variogram as _robust_variogram,
)
from .algorithms.fitting import (
    automatic_fit,
    fit_variogram_model as fit_variogram_model,
)
from .models.variogram_models import (
    SphericalModel,
    ExponentialModel,
    GaussianModel,
    LinearModel,
    PowerModel,
    MaternModel,
)
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "experimental_variogram",
    "experimental_variogram_directional",
    "variogram_cloud",
    "robust_variogram",
    "fit_model",
    "auto_fit",
]


def experimental_variogram(
def experimental_variogram(
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Calculate experimental variogram

    Parameters
    ----------
    x, y : array-like
    Coordinates of sample points
    z : array-like
    Values at sample points
    n_lags : int, default=15
    Number of lag bins
    maxlag : float, optional
    Maximum lag distance. If None, uses half the maximum distance.

    Returns
    -------
    lags : np.ndarray
    Lag distances (bin centers)
    gamma : np.ndarray
    Semivariance values
    n_pairs : np.ndarray
    Number of pairs in each lag bin

    Examples
    --------
    >>> import numpy as np
    >>> from geostats import variogram
    >>> x = np.random.rand(100) * 100
    >>> y = np.random.rand(100) * 100
    >>> z = np.sin(x/10) + np.cos(y/10)
    >>> lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)
    """
    return _experimental_variogram(x, y, z, n_lags=n_lags, maxlag=maxlag)


def experimental_variogram_directional(
def experimental_variogram_directional(
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    angle: float = 0.0,
    tolerance: float = 22.5,
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Calculate directional experimental variogram

    Parameters
    ----------
    x, y : array-like
    Coordinates of sample points
    z : array-like
    Values at sample points
    angle : float, default=0.0
    Direction angle in degrees (0=East, 90=North)
    tolerance : float, default=22.5
    Angular tolerance in degrees
    n_lags : int, default=15
    Number of lag bins
    maxlag : float, optional
    Maximum lag distance

    Returns
    -------
    lags : np.ndarray
    Lag distances
    gamma : np.ndarray
    Semivariance values
    n_pairs : np.ndarray
    Number of pairs in each lag bin
    """
    return _experimental_variogram_directional(
        x, y, z, angle=angle, tolerance=tolerance, n_lags=n_lags, maxlag=maxlag
    )


def variogram_cloud(
def variogram_cloud(
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    maxlag: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate variogram cloud

    Shows all individual squared differences vs distances.

    Parameters
    ----------
    x, y : array-like
    Coordinates of sample points
    z : array-like
    Values at sample points
    maxlag : float, optional
    Maximum lag distance to include

    Returns
    -------
    distances : np.ndarray
    All pairwise distances
    semivariances : np.ndarray
    Squared differences / 2 for each pair
    """
    return _variogram_cloud(x, y, z, maxlag=maxlag)


def robust_variogram(
def robust_variogram(
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    estimator: str = "cressie",
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Calculate robust experimental variogram

    Uses robust estimators less sensitive to outliers.

    Parameters
    ----------
    x, y : array-like
    Coordinates
    z : array-like
    Values
    n_lags : int, default=15
    Number of lag bins
    maxlag : float, optional
    Maximum lag distance
    estimator : str, default='cressie'
    Robust estimator: 'cressie' or 'dowd'

    Returns
    -------
    lags : np.ndarray
    Lag distances
    gamma : np.ndarray
    Robust semivariance values
    n_pairs : np.ndarray
    Number of pairs in each lag
    """
    return _robust_variogram(x, y, z, n_lags=n_lags, maxlag=maxlag, estimator=estimator)


def fit_model(
def fit_model(
    lags: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    weights: Optional[npt.NDArray[np.float64]] = None,
    **kwargs: Any,
    ):
    """
    Fit a variogram model to experimental data

    Parameters
    ----------
    model_type : str
    Type of model: 'spherical', 'exponential', 'gaussian',
    'linear', 'power', 'matern'
    lags : array-like
    Lag distances
    gamma : array-like
    Experimental semivariance values
    weights : array-like, optional
    Weights for fitting (e.g., number of pairs)
    **kwargs
    Additional model parameters

    Returns
    -------
    model : VariogramModelBase
    Fitted variogram model

    Examples
    --------
    >>> lags, gamma, n_pairs = experimental_variogram(x, y, z)
    >>> model = fit_model('spherical', lags, gamma, weights=n_pairs)
    >>> logger.info(model.parameters)
    """
    # Select model class
    model_map = {
        "spherical": SphericalModel,
        "exponential": ExponentialModel,
        "gaussian": GaussianModel,
        "linear": LinearModel,
        "power": PowerModel,
        "matern": MaternModel,
    }

    model_type_lower = model_type.lower()
    if model_type_lower not in model_map:
    if model_type_lower not in model_map:
        )

    # Create and fit model
    model = model_map[model_type_lower](**kwargs)
    return fit_variogram_model(model, lags, gamma, weights=weights)


def auto_fit(
def auto_fit(
    gamma: npt.NDArray[np.float64],
    weights: Optional[npt.NDArray[np.float64]] = None,
    criterion: str = "rmse",
    ) -> Dict[str, Any]:
    """
    Automatically select and fit the best variogram model

    Parameters
    ----------
    lags : array-like
    Lag distances
    gamma : array-like
    Experimental semivariance values
    weights : array-like, optional
    Weights for fitting
    criterion : str, default='rmse'
    Selection criterion: 'rmse', 'mae', 'r2', 'aic'

    Returns
    -------
    dict
    Dictionary containing:
    - 'model': Best fitted model
    - 'score': Best score value
    - 'all_results': Results for all models tried

    Examples
    --------
    >>> lags, gamma, n_pairs = experimental_variogram(x, y, z)
    >>> result = auto_fit(lags, gamma, weights=n_pairs)
    >>> best_model = result['model']
    """
    return automatic_fit(lags, gamma, weights=weights, criterion=criterion)
