"""
Base classes for variogram and covariance models
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import numpy.typing as npt

from ..core.base import BaseModel
from ..core.validators import validate_positive
from ..math.numerical import weighted_least_squares

class VariogramModelBase(BaseModel):
 Base class for all variogram models

 A variogram model describes spatial correlation as a function of distance.
 The semivariance γ(h) typically increases with distance h.
 """

 def __init__(
     nugget: float = 0.0,
     sill: Optional[float] = None,
     range_param: Optional[float] = None,
     ) -> None:
     """
     Initialize variogram model

     Parameters
     ----------
     nugget : float
     Nugget effect (variance at distance 0)
     sill : float, optional
     Sill (maximum variance)
     range_param : float, optional
     Range parameter (distance at which correlation becomes negligible)
     """
     super().__init__()
     self._parameters = {
     "nugget": max(0.0, nugget),
     "sill": sill if sill is not None else 1.0,
     "range": range_param if range_param is not None else 1.0,
     }

     @abstractmethod
 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Core model function (without nugget)

 Parameters
 ----------
 h : np.ndarray
 Distance values

 Returns
 -------
 np.ndarray
 Model values
 """
 pass

 def __call__(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Evaluate variogram at distance h

 Parameters
 ----------
 h : np.ndarray
 Distance values

 Returns
 -------
 np.ndarray
 Semivariance values
 """
 h = np.asarray(h, dtype=np.float64)
 nugget = self._parameters["nugget"]

 # Add nugget effect for h > 0
 result = self._model_function(h)

 # Nugget only applies at h = 0
 if np.isscalar(h):
     return nugget
 else:
     result[h == 0] = nugget

 return result

 def fit(
     lags: npt.NDArray[np.float64],
     gamma: npt.NDArray[np.float64],
     weights: Optional[npt.NDArray[np.float64]] = None,
     fit_nugget: bool = True,
     **kwargs: Any,
     ) -> "VariogramModelBase":
     """
     Fit variogram model to experimental data

     Parameters
     ----------
     lags : np.ndarray
     Lag distances
     gamma : np.ndarray
     Experimental semivariance values
     weights : np.ndarray, optional
     Weights for fitting
     fit_nugget : bool
     Whether to fit nugget effect
     **kwargs
     Additional fitting parameters

     Returns
     -------
     self
     Fitted model
     """
     lags = np.asarray(lags, dtype=np.float64)
     gamma = np.asarray(gamma, dtype=np.float64)

     # Remove zero lags for fitting
     mask = lags > 0
     lags_fit = lags[mask]
     gamma_fit = gamma[mask]

     if weights is not None:

     # Initial parameter estimates
     sill_init = np.max(gamma_fit) if len(gamma_fit) > 0 else 1.0
     range_init = lags_fit[len(lags_fit) // 2] if len(lags_fit) > 0 else 1.0
     nugget_init = gamma[0] if len(gamma) > 0 and lags[0] == 0 else 0.0

     # Prepare fitting
     p0, bounds_lower, bounds_upper = self._prepare_fitting(
     nugget_init, sill_init, range_init, fit_nugget
     )

     try:
     try:
     self._fitting_function,
     lags_fit,
     gamma_fit,
     weights=weights,
     p0=p0,
     bounds=(bounds_lower, bounds_upper),
     maxfev=10000,
     )

     self._update_parameters_from_fit(params, fit_nugget)
     self._is_fitted = True

     except Exception as e:
     # Use initial estimates if fitting fails
     self._parameters["nugget"] = nugget_init
     self._parameters["sill"] = sill_init
     self._parameters["range"] = range_init
     self._is_fitted = True

     return self

 def _prepare_fitting(
     nugget_init: float,
     sill_init: float,
     range_init: float,
     fit_nugget: bool,
     ) -> tuple:
     """Prepare initial parameters and bounds for fitting"""
     if fit_nugget:
     bounds_lower = [0.0, 0.0, 0.0]
     bounds_upper = [np.inf, np.inf, np.inf]
     else:
     else:
     bounds_lower = [0.0, 0.0]
     bounds_upper = [np.inf, np.inf]

     return p0, bounds_lower, bounds_upper

 def _fitting_function(self, h: npt.NDArray[np.float64], *params: float) -> npt.NDArray[np.float64]:
     # Temporarily set parameters
     old_params = self._parameters.copy()

 if len(params) == 3: # Fitting nugget
     self._parameters["sill"] = params[1]
 self._parameters["range"] = params[2]
 else: # Not fitting nugget
     self._parameters["range"] = params[1]

 result = self._model_function(h)

 self._parameters = old_params
 return result

 def _update_parameters_from_fit(self, params: npt.NDArray[np.float64], fit_nugget: bool) -> None:
     if fit_nugget:
     self._parameters["sill"] = max(0.0, params[1])
     self._parameters["range"] = max(0.0, params[2])
     else:
     else:
     self._parameters["range"] = max(0.0, params[1])

class CovarianceModelBase(BaseModel):
 Base class for covariance models

 A covariance model describes spatial correlation as a function of distance.
 The covariance C(h) typically decreases with distance h.
 """

 def __init__(
     sill: float = 1.0,
     range_param: float = 1.0,
     ) -> None:
     """
     Initialize covariance model

     Parameters
     ----------
     sill : float
     Sill (maximum covariance at distance 0)
     range_param : float
     Range parameter
     """
     super().__init__()
     self._parameters = {
     "sill": validate_positive(sill, "sill"),
     "range": validate_positive(range_param, "range"),
     }

     @abstractmethod
 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     pass

 def __call__(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Evaluate covariance at distance h

 Parameters
 ----------
 h : np.ndarray
 Distance values

 Returns
 -------
 np.ndarray
 Covariance values
 """
 h = np.asarray(h, dtype=np.float64)
 return self._model_function(h)

 def fit(
     lags: npt.NDArray[np.float64],
     cov: npt.NDArray[np.float64],
     **kwargs: Any,
     ) -> "CovarianceModelBase":
     """
     Fit covariance model to data

     Parameters
     ----------
     lags : np.ndarray
     Lag distances
     cov : np.ndarray
     Covariance values
     **kwargs
     Additional fitting parameters

     Returns
     -------
     self
     Fitted model
     """
     # Simple implementation - can be overridden
     self._parameters["sill"] = np.max(cov)
     self._parameters["range"] = lags[len(lags) // 2] if len(lags) > 0 else 1.0
     self._is_fitted = True
     return self
