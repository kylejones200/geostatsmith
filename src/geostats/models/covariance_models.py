"""
    Covariance function models

Covariance C(h) and variogram γ(h) are related by:
 C(h) = C(0) - γ(h)

where C(0) is the sill variance.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import gamma as gamma_func, kv

from .base_model import CovarianceModelBase

class SphericalCovariance(CovarianceModelBase):
 Spherical covariance model

 Formula:
     pass
 C(h) = sill * [1 - 1.5*(h/a) + 0.5*(h/a)^3] for 0 < h <= a
 C(h) = 0 for h > a
 C(0) = sill
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 h_norm = h / range_param

 result = np.where(
 h_norm <= 1.0,
 sill * (1.0 - 1.5 * h_norm + 0.5 * h_norm**3),
 0.0,
 )

 # Set C(0) = sill
 if np.isscalar(h):
     return sill
 else:
    pass

     return result

class ExponentialCovariance(CovarianceModelBase):
 Exponential covariance model

 Formula:
     pass
 C(h) = sill * exp(-h/a)

 where a is the range parameter.
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 return sill * np.exp(-h / range_param)

class GaussianCovariance(CovarianceModelBase):
 Gaussian covariance model

 Formula:
     pass
 C(h) = sill * exp(-(h/a)^2)

 where a is the range parameter.
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 h_norm = h / range_param
 return sill * np.exp(-(h_norm**2))

class MaternCovariance(CovarianceModelBase):
 Matérn covariance model

 Formula:
     pass
 C(h) = sill * (2^(1-ν)/Γ(ν)) * (h/a)^ν * K_ν(h/a) for h > 0
 C(0) = sill
 """

 def __init__(self, sill: float = 1.0, range_param: float = 1.0, nu: float = 0.5):
     Initialize Matérn covariance

 Parameters
 ----------
 sill : float
 Sill (variance at distance 0)
 range_param : float
 Range parameter
 nu : float
 Smoothness parameter
 """
 super().__init__(sill=sill, range_param=range_param)
 if nu <= 0:
     self._parameters["nu"] = nu

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]
     nu = self._parameters["nu"]

 result = np.zeros_like(h)
 mask = h > 0

 if np.any(mask):
    pass

     const = 2.0 ** (1.0 - nu) / gamma_func(nu)
 bessel_part = kv(nu, h_scaled)

 with np.errstate(over='ignore', invalid='ignore'):
     spatial_part = np.nan_to_num(spatial_part, nan=0.0, posinf=0.0)

 result[mask] = sill * spatial_part

 # C(0) = sill
 if np.isscalar(h):
     return sill
 else:
    pass

     return result

class LinearCovariance(CovarianceModelBase):
 Linear covariance model (decreasing)

 Formula:
     pass
 C(h) = sill * (1 - h/a) for 0 <= h <= a
 C(h) = 0 for h > a
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 h_norm = h / range_param

 return np.where(
 h_norm <= 1.0,
 sill * (1.0 - h_norm),
 0.0,
 )
