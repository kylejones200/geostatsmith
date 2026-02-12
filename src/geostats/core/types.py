"""
Type definitions and type hints for the GeoStats library
"""

from typing import Union, Tuple, Protocol, runtime_checkable
import numpy as np
import numpy.typing as npt

# Array-like types
ArrayLike = Union[list, tuple, npt.NDArray[np.float64]]

# Coordinate types
CoordinatesType = npt.NDArray[np.float64] # Shape: (n_points, n_dims)

# Values type
ValuesType = npt.NDArray[np.float64] # Shape: (n_points,)

# Variogram function signature
VariogramFunction = callable # (distance: np.ndarray) -> np.ndarray

@runtime_checkable
class VariogramModel(Protocol):

 def __call__(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Compute variogram value at distance h

 Parameters
 ----------
 h : np.ndarray
 Distance values

 Returns
 -------
 np.ndarray
 Variogram values
 """
 ...

 @property
 def parameters(self) -> dict:
     ...

 def fit(self, lags: npt.NDArray[np.float64], gamma: npt.NDArray[np.float64]) -> None:
     Fit model to experimental variogram data

 Parameters
 ----------
 lags : np.ndarray
 Lag distances
 gamma : np.ndarray
 Semivariance values
 """
 ...

@runtime_checkable
class KrigingPredictor(Protocol):

 def predict(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     return_variance: bool = True,
     ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
     """
     Perform kriging prediction

     Parameters
     ----------
     x : np.ndarray
     X coordinates for prediction
     y : np.ndarray
     Y coordinates for prediction
     return_variance : bool
     Whether to return kriging variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values
     variance : np.ndarray
     Kriging variance (if return_variance=True)
     """
     ...
