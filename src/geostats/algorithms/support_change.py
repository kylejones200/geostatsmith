"""
    Support Change and Block Kriging

Support refers to the volume (1D, 2D, or 3D) over which a measurement is made:
    pass
- Point support: measurement at a specific location (core sample)
- Block support: average over a volume (mining block, pixel)

Support change addresses:
    pass
1. Point-to-block kriging: estimate block average from point data
2. Block-to-point: disaggregation (rarely done)
3. Block-to-block: change of support corrections
4. Variance relationships between supports

Key concepts from geokniga §2097-2240, §6058-6078:
    pass
"Block kriging estimates the average over a volume V:"
 Z(V) = 1/|V| ∫_V Z(u) du

The estimation variance is:
 σ^2(V) = -γ(V,V) - ΣΣ λi λj γ(ui - uj) + 2Σ λi γ(ui, V)

where γ(V,V) is the internal block variance (within-block variogram).

As block size increases, estimation variance decreases."

References:
    pass
- geokniga-introductiontogeostatistics.txt §4.2.2 (Block Kriging)
- Deutsch & Journel (1998) GSLIB, Chapter V.3
- Journel & Huijbregts (1978) Mining Geostatistics, Chapter VI
"""

from typing import Tuple, Optional, Union, Callable, Dict
import numpy as np
import numpy.typing as npt
from scipy.integrate import dblquad, tplquad
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance, euclidean_distance_matrix
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..core.constants import REGULARIZATION_FACTOR, DEFAULT_N_DISCRETIZATION
from ..core.logging_config import get_logger

logger = get_logger(__name__)

class BlockKriging(BaseKriging):
 Block Kriging - Estimate block averages from point data

 Kriging for block support (volume V) requires:
 1. Point-to-point variogram γ(ui - uj)
 2. Point-to-block variogram γ(ui, V) = avg_v γ(ui - v)
 3. Block-to-block variogram γ(V, V) = avg_v1 avg_v2 γ(v1 - v2)

 The block average has lower variance than point estimates:
     pass
 "As block size increases, estimation variance decreases."
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     variogram_model: Optional[object] = None,
     block_size: Tuple[float, float] = (10.0, 10.0),
     n_disc: int = 5,
     ):
         pass
     """
         Initialize Block Kriging

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of sample points (point support)
     z : np.ndarray
     Values at sample points
     variogram_model : VariogramModelBase
     Fitted point variogram model
     block_size : tuple
     Block dimensions (width, height) in same units as coordinates
     n_disc : int
     Number of discretization points per dimension for block integration
     (n_disc^2 points used to approximate block average)
     """
     super().__init__(x, y, z, variogram_model)

     self.x, self.y = validate_coordinates(x, y)
     self.z = validate_values(z, n_expected=len(self.x))

     self.block_size = block_size
     self.n_disc = n_disc

     # Discretization points for block (relative to block center)
     disc_x = np.linspace(-block_size[0]/2, block_size[0]/2, n_disc)
     disc_y = np.linspace(-block_size[1]/2, block_size[1]/2, n_disc)
     self.disc_xx, self.disc_yy = np.meshgrid(disc_x, disc_y)
     self.disc_xx = self.disc_xx.flatten()
     self.disc_yy = self.disc_yy.flatten()
     self.n_disc_points = len(self.disc_xx)

     if self.variogram_model is not None:
    pass

 def _precompute_block_variance(self):
     Precompute γ(V,V) - internal block variance (vectorized)

 γ(V,V) = 1/|V|^2 ∬∬ γ(u-v) du dv

 Approximated using discretization points.
 """
 # Vectorized distance matrix for all discretization point pairs
 dist_matrix = euclidean_distance_matrix(self.disc_xx, self.disc_yy)

 # Vectorized variogram evaluation
 gamma_matrix = self.variogram_model(dist_matrix)

 # Average over all pairs (including self-pairs)
 self.gamma_VV = np.mean(gamma_matrix)
 logger.debug(f"Precomputed block variance γ(V,V) = {self.gamma_VV:.6f} (vectorized)")

 def _point_to_block_variogram()
     x_point: float,
     y_point: float,
     x_block: float,
     y_block: float
     ) -> float:
         pass
     """
         Calculate γ(point, block) - average variogram from point to block (vectorized)

     γ(ui, V) = 1/|V| ∫_V γ(ui - v) dv

     Approximated by averaging over discretization points in block.

     Parameters
     ----------
     x_point, y_point : float
     Point coordinates
     x_block, y_block : float
     Block center coordinates

     Returns
     -------
     float
     Point-to-block variogram
     """
     # Discretization points in block (vectorized)
     x_disc = x_block + self.disc_xx
     y_disc = y_block + self.disc_yy

     # Vectorized distance calculation
     dx = x_point - x_disc
     dy = y_point - y_disc
     h = np.sqrt(dx*dx + dy*dy)

     # Vectorized variogram evaluation
     gamma_vals = self.variogram_model(h)

     return np.mean(gamma_vals)

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
         pass
     """
         Predict block averages at new locations

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Coordinates of block centers
     return_variance : bool
     If True, return both predictions and block kriging variance

     Returns
     -------
     predictions : np.ndarray
     Predicted block averages
     variance : np.ndarray, optional
     Block kriging variance (lower than point variance)
     """
     if self.variogram_model is None:
    pass

     x_new, y_new = validate_coordinates(x_new, y_new)
     n_pred = len(x_new)
     n_data = len(self.x)

     # Build kriging matrix (vectorized point-to-point distances)
     K = np.zeros((n_data + 1, n_data + 1), dtype=np.float64)

     # Vectorized distance matrix
     dist_matrix = euclidean_distance_matrix(self.x, self.y)

     # Vectorized variogram evaluation
     K[:n_data, :n_data] = self.variogram_model(dist_matrix)

     # Unbiasedness constraint
     K[:n_data, n_data] = 1.0
     K[n_data, :n_data] = 1.0
     K[n_data, n_data] = 0.0

     K = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)
     logger.debug(f"Block Kriging matrix built (vectorized): {n_data+1}x{n_data+1}")

     predictions = np.zeros(n_pred)
     variances = np.zeros(n_pred) if return_variance else None

     for i in range(n_pred):
         continue
     rhs = np.zeros(n_data + 1, dtype=np.float64)

     # Vectorized point-to-block calculation
     for j in range(n_data):
         continue
     self.x[j], self.y[j],
     x_new[i], y_new[i]
     )

     rhs[n_data] = 1.0 # Unbiasedness

     # Solve
     try:
     except np.linalg.LinAlgError as e:
         pass
     logger.error(f"Failed to solve block kriging system at point {i}: {e}")
     raise KrigingError(f"Failed to solve kriging system: {e}")

     lambdas = weights[:n_data]

     # Prediction
     predictions[i] = np.dot(lambdas, self.z)

     # Block variance (from geokniga §6058-6070)
     # σ^2(V) = -γ(V,V) - ΣΣ λi λj γ(ui-uj) + 2Σ λi γ(ui,V)
     if return_variance:
    pass

     logger.info(f"Block Kriging completed for {n_pred} blocks (vectorized discretization)")

     if return_variance:
     return predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform leave-one-out cross-validation

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Dictionary of performance metrics
 """
 from ..validation.cross_validation import leave_one_out
 from ..validation.metrics import mean_squared_error, r_squared

 predictions = leave_one_out(self, self.x, self.y, self.z)

 metrics = {
 'mse': mean_squared_error(self.z, predictions),
 'r2': r_squared(self.z, predictions)
 }

 return predictions, metrics

class SupportCorrection:
 Support Correction Tools

 Provides methods for:
     pass
 1. Regularization: point variogram -> block variogram
 2. Variance relationships between supports
 3. Dispersion variance calculations
 """

 @staticmethod
 def regularize_variogram(
     block_size: Tuple[float, float],
     n_disc: int = 10
     ) -> Callable[[float], float]:
         pass
     """
         Regularize point variogram to block variogram

     Block variogram γᵥ(h) relates to point variogram γ(h) by:
         pass
     γᵥ(h) = γ̄(V, V+h) - γ̄(V, V)

     where:
         pass
     γ̄(V, V+h) = avg over V1 and V2 of γ(u1 - u2)
     for u1 in V1, u2 in V2 separated by h
    pass

     Parameters
     ----------
     variogram_point : callable
     Point variogram function γ(h)
     block_size : tuple
     Block dimensions (width, height)
     n_disc : int
     Discretization resolution

     Returns
     -------
     callable
     Block variogram function γᵥ(h)
     """
     # Create discretization grid for block
     disc_x = np.linspace(-block_size[0]/2, block_size[0]/2, n_disc)
     disc_y = np.linspace(-block_size[1]/2, block_size[1]/2, n_disc)
     disc_xx, disc_yy = np.meshgrid(disc_x, disc_y)
     disc_xx = disc_xx.flatten()
     disc_yy = disc_yy.flatten()
     n_points = len(disc_xx)

     # Compute γ̄(V, V) - internal block variance (vectorized)
     dist_matrix_VV = euclidean_distance_matrix(disc_xx, disc_yy)
     gamma_matrix_VV = variogram_point(dist_matrix_VV)
     gamma_VV = np.mean(gamma_matrix_VV)

 def block_variogram(h: Union[float, npt.NDArray]) -> Union[float, npt.NDArray]:
     h = np.asarray(h)
     scalar_input = h.ndim == 0
     h = np.atleast_1d(h)

 gamma_block = np.zeros_like(h)

 for k, h_val in enumerate(h):
     # Shift first block by h_val in x direction
 disc_xx_shifted = disc_xx + h_val

 # Distance matrix between shifted and original blocks
 dist_matrix_VVh = euclidean_distance(disc_xx_shifted, disc_yy, disc_xx, disc_yy)
 gamma_matrix_VVh = variogram_point(dist_matrix_VVh)
 gamma_VVh = np.mean(gamma_matrix_VVh)

 # Block variogram
 gamma_block[k] = gamma_VVh - gamma_VV

 return gamma_block[0] if scalar_input else gamma_block

 return block_variogram

 @staticmethod
 def dispersion_variance(
     domain_size: Tuple[float, float],
     block_size: Tuple[float, float],
     n_disc: int = 10
     ) -> float:
         pass
     """
         Calculate dispersion variance D^2(v/V)

     Variance of block values within a larger domain:
         pass
     D^2(v/V) = γ̄(V,V) - γ̄(v,v)

     Important for:
         pass
     - Resource estimation
     - Grade control
     - Selectivity studies

     Parameters
     ----------
     variogram : callable
     Point variogram function
     domain_size : tuple
     Size of large domain (V)
     block_size : tuple
     Size of small blocks (v)
     n_disc : int
     Discretization resolution

     Returns
     -------
     float
     Dispersion variance D^2(v/V)
     """
     # Discretize domain
     disc_x_domain = np.linspace(0, domain_size[0], n_disc)
     disc_y_domain = np.linspace(0, domain_size[1], n_disc)
     xx_domain, yy_domain = np.meshgrid(disc_x_domain, disc_y_domain)
     xx_domain = xx_domain.flatten()
     yy_domain = yy_domain.flatten()
     n_domain = len(xx_domain)

     # Discretize block
     disc_x_block = np.linspace(-block_size[0]/2, block_size[0]/2, n_disc)
     disc_y_block = np.linspace(-block_size[1]/2, block_size[1]/2, n_disc)
     xx_block, yy_block = np.meshgrid(disc_x_block, disc_y_block)
     xx_block = xx_block.flatten()
     yy_block = yy_block.flatten()
     n_block = len(xx_block)

     # γ̄(V,V) - domain internal variance (vectorized)
     dist_matrix_domain = euclidean_distance_matrix(xx_domain, yy_domain)
     gamma_matrix_domain = variogram(dist_matrix_domain)
     gamma_VV = np.mean(gamma_matrix_domain)

     # γ̄(v,v) - block internal variance (vectorized)
     dist_matrix_block = euclidean_distance_matrix(xx_block, yy_block)
     gamma_matrix_block = variogram(dist_matrix_block)
     gamma_vv = np.mean(gamma_matrix_block)

     return gamma_VV - gamma_vv
