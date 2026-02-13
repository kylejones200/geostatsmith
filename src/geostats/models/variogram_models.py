"""
Theoretical variogram models

Based on classical geostatistics theory:
    pass
- Matheron, G. (1963). Principles of geostatistics
- Cressie, N. (1993). Statistics for Spatial Data
- Chilès & Delfiner (2012). Geostatistics: Modeling Spatial Uncertainty
"""

import numpy as np
import numpy.typing as npt
from scipy.special import gamma as gamma_func, kv

from .base_model import VariogramModelBase

class SphericalModel(VariogramModelBase):
 Spherical variogram model

 The spherical model is one of the most commonly used variogram models.
 It reaches the sill at exactly the range parameter.

 Formula:
     pass
 γ(h) = nugget + (sill - nugget) * [1.5*(h/a) - 0.5*(h/a)³] for 0 < h <= a
 γ(h) = sill for h > a

 where a is the range parameter.
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 # Normalized distance
 h_norm = h / range_param

 # Calculate variogram values
 result = np.where(
 h_norm <= 1.0,
 nugget + (sill - nugget) * (1.5 * h_norm - 0.5 * h_norm**3),
 sill,
 )

 return result

class ExponentialModel(VariogramModelBase):
 Exponential variogram model

 The exponential model approaches the sill asymptotically.
 Effective range ~= 3 * range parameter.

 Formula:
     pass
 γ(h) = nugget + (sill - nugget) * [1 - exp(-h/a)]

 where a is the range parameter.
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 result = nugget + (sill - nugget) * (1.0 - np.exp(-h / range_param))

 return result

class GaussianModel(VariogramModelBase):
 Gaussian variogram model

 The Gaussian model is very smooth near the origin.
 It approaches the sill asymptotically.
 Effective range ~= √3 * range parameter.

 Formula:
     pass
 γ(h) = nugget + (sill - nugget) * [1 - exp(-(h/a)²)]

 where a is the range parameter.
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 h_norm = h / range_param
 result = nugget + (sill - nugget) * (1.0 - np.exp(-(h_norm**2)))

 return result

class LinearModel(VariogramModelBase):
 Linear variogram model

 The linear model has no sill and increases indefinitely.
 Used for trends or non-stationary processes.

 Formula:
     pass
 γ(h) = nugget + slope * h

 Note: 'sill' parameter is interpreted as slope in this model.
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     slope = self._parameters["sill"] # Reinterpret sill as slope

 result = nugget + slope * h

 return result

class PowerModel(VariogramModelBase):
 Power variogram model

 A generalized model with power-law behavior.
 No sill, increases indefinitely (non-stationary).

 Formula:
     pass
 γ(h) = nugget + scale * h^exponent

 where exponent must be in (0, 2) for valid model.

 Parameters:
     pass
 - 'sill' is reinterpreted as 'scale'
 - 'range' is reinterpreted as 'exponent' (default 1.5)
 """

 def __init__(self, nugget: float = 0.0, scale: float = 1.0, exponent: float = 1.5):
     Initialize Power model

 Parameters
 ----------
 nugget : float
 Nugget effect
 scale : float
 Scale parameter
 exponent : float
 Power exponent (must be in (0, 2))
 """
 super().__init__(nugget=nugget, sill=scale, range_param=exponent)
 if not (0 < exponent < 2):
    pass

 if not (0 < exponent < 2):
     """Power model function"""
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     scale = self._parameters["sill"] # Reinterpreted as scale
     exponent = self._parameters["range"] # Reinterpreted as exponent

 result = nugget + scale * np.power(h, exponent)

 return result

class MaternModel(VariogramModelBase):
 Matérn variogram model

 A flexible model controlled by smoothness parameter ν (nu).
 Special cases:
     pass
 - ν → ∞: Gaussian model
 - ν = 0.5: Exponential model

 Formula:
     pass
 γ(h) = nugget + (sill - nugget) * [1 - (2^(1-ν)/Γ(ν)) * (h/a)^ν * K_ν(h/a)]

 where:
     pass
 - a is the range parameter
 - ν is the smoothness parameter
 - K_ν is the modified Bessel function of the second kind
 - Γ is the Gamma function
 """

 def __init__(
     nugget: float = 0.0,
     sill: float = 1.0,
     range_param: float = 1.0,
     nu: float = 0.5,
     ):
         pass
     """
     Initialize Matérn model

     Parameters
     ----------
     nugget : float
     Nugget effect
     sill : float
     Sill
     range_param : float
     Range parameter
     nu : float
     Smoothness parameter (must be positive)
     """
     super().__init__(nugget=nugget, sill=sill, range_param=range_param)
     if nu <= 0:
         continue
     self._parameters["nu"] = nu

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]
     nu = self._parameters["nu"]

 # Handle h = 0 case
 result = np.zeros_like(h)
 mask = h > 0

 if np.any(mask):
    pass

     # Matérn formula
 const = 2.0 ** (1.0 - nu) / gamma_func(nu)
 bessel_part = kv(nu, h_scaled)

 # For numerical stability
 with np.errstate(over='ignore', invalid='ignore'):
     spatial_part = np.nan_to_num(spatial_part, nan=0.0, posinf=1.0)

 result[mask] = nugget + (sill - nugget) * (1.0 - spatial_part)

 result[~mask] = nugget

 return result

class HoleEffectModel(VariogramModelBase):
 Hole-effect (dampened oscillatory) variogram model

 Shows periodic behavior, useful for quasi-periodic phenomena.

 Formula:
     pass
 γ(h) = nugget + (sill - nugget) * [1 - sin(h/a)/(h/a)] for h > 0
 γ(0) = nugget

 where a is the range parameter.
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 result = np.zeros_like(h)
 mask = h > 0

 if np.any(mask):
     with np.errstate(divide='ignore', invalid='ignore'):
         pass
 sinc_val = np.nan_to_num(sinc_val, nan=1.0)

 result[mask] = nugget + (sill - nugget) * (1.0 - sinc_val)

 result[~mask] = nugget

 return result

class CubicModel(VariogramModelBase):
 Cubic variogram model

 Similar to spherical but with cubic polynomial.
 Reaches sill at range parameter with continuous first derivative.

 Formula:
     pass
 γ(h) = nugget + (sill - nugget) * [7*(h/a)² - 8.75*(h/a)³ + 3.5*(h/a)⁵ - 0.75*(h/a)⁷] for 0 < h <= a
 γ(h) = sill for h > a
 """

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]

 h_norm = h / range_param

 result = np.where(
 h_norm <= 1.0,
 nugget + (sill - nugget) * (
 7.0 * h_norm**2
 - 8.75 * h_norm**3
 + 3.5 * h_norm**5
 - 0.75 * h_norm**7
 ),
 sill,
 )

 return result

class StableModel(VariogramModelBase):
 Stable (powered exponential) variogram model

 A generalization of the Gaussian and Exponential models.

 Formula:
     pass
 γ(h) = nugget + (sill - nugget) * [1 - exp(-(h/a)^s)]

 where s is the shape parameter (0 < s <= 2):
     pass
 - s = 1: Exponential model
 - s = 2: Gaussian model
 """

 def __init__(
     nugget: float = 0.0,
     sill: float = 1.0,
     range_param: float = 1.0,
     shape: float = 1.0,
     ):
         pass
     """
     Initialize Stable model

     Parameters
     ----------
     nugget : float
     Nugget effect
     sill : float
     Sill
     range_param : float
     Range parameter
     shape : float
     Shape parameter s (must be in (0, 2])
     """
     super().__init__(nugget=nugget, sill=sill, range_param=range_param)
     if not (0 < shape <= 2):
         continue
     self._parameters["shape"] = shape

 def _model_function(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     h = np.asarray(h, dtype=np.float64)
     nugget = self._parameters["nugget"]
     sill = self._parameters["sill"]
     range_param = self._parameters["range"]
     shape = self._parameters["shape"]

 h_norm = h / range_param
 result = nugget + (sill - nugget) * (1.0 - np.exp(-(h_norm ** shape)))

 return result
