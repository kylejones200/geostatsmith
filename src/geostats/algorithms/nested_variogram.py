"""
Nested Variogram Models

A nested variogram is the sum of multiple variogram structures at different scales:

 γ(h) = C₀ + Σᵢ Cᵢ·γᵢ(h/aᵢ)

where:
- C₀ = nugget effect (micro-scale variation)
- Cᵢ = partial sill of structure i
- γᵢ = base variogram model (spherical, exponential, etc.)
- aᵢ = range of structure i

Nested models capture multi-scale spatial variation common in natural phenomena:
- Short-range: local variability
- Medium-range: deposit-scale structures
- Long-range: regional trends

References:
- Deutsch & Journel (1998) - GSLIB: Multiple structures
- Chilès & Delfiner (2012) - Geostatistics: Modeling Spatial Uncertainty
- Wackernagel (2003) - Multivariate Geostatistics
"""

from typing import List, Tuple, Dict, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution, minimize
import logging

logger = logging.getLogger(__name__)

from ..models.variogram_models import (
 SphericalModel, ExponentialModel, GaussianModel,
 LinearModel, PowerModel
)
from ..core.constants import (
 MIN_SEMIVARIANCE_RATIO,
 MAX_SEMIVARIANCE_RATIO,
 MIN_RANGE_RATIO,
 MAX_RANGE_RATIO,
 MAX_ITERATIONS_GLOBAL,
 CONVERGENCE_TOLERANCE,
 OPTIMIZATION_ATOL,
)
from ..core.logging_config import get_logger

logger = get_logger(__name__)

# Optimization constants
NUGGET_MAX_FRACTION = 0.5
PENALTY_VALUE = 1e10
RANDOM_SEED = 42

@dataclass
class VariogramStructure:
 """Single structure in a nested variogram"""
 model_type: str # 'spherical', 'exponential', 'gaussian', etc.
 sill: float # Partial sill (Cᵢ)
 range: float # Range parameter (aᵢ)
 nugget: float = 0.0 # Only used for first structure

 def __str__(self):
 return f"{self.model_type}(sill={self.sill:.3f}, range={self.range:.1f})"

class NestedVariogram:
 """
 Nested Variogram Model

 Combines multiple variogram structures at different scales:
 γ(h) = nugget + Σᵢ sillᵢ · modelᵢ(h/rangeᵢ)

 Examples:
 - Nugget + Spherical: micro-scale + deposit-scale
 - Nugget + Exponential + Gaussian: 3-scale structure
 - Spherical + Spherical: dual-scale nested model

 All structures must be positive definite variogram models.
 """

 def __init__(self, nugget: float = 0.0):
 """
 Initialize nested variogram

 Parameters
 ----------
 nugget : float
 Nugget effect (micro-scale discontinuity at h=0)
 """
 self.nugget = nugget
 self.structures: List[VariogramStructure] = []

 # Map model names to model classes
 self.model_classes = {
 'spherical': SphericalModel,
 'exponential': ExponentialModel,
 'gaussian': GaussianModel,
 'linear': LinearModel,
 'power': PowerModel,
 }

 def add_structure(
 self,
 model_type: str,
 sill: float,
 range: float
 ):
 """
 Add a structure to the nested model

 Parameters
 ----------
 model_type : str
 Type of variogram model ('spherical', 'exponential', etc.)
 sill : float
 Partial sill (contribution to total variance)
 range : float
 Range parameter (correlation distance)
 """
 if model_type not in self.model_classes:
 raise ValueError(
 f"Unknown model type: {model_type}. "
 f"Available: {list(self.model_classes.keys())}"
 )

 if sill <= 0:
 raise ValueError("Sill must be positive")

 if range <= 0:
 raise ValueError("Range must be positive")

 structure = VariogramStructure(
 model_type=model_type,
 sill=sill,
 range=range
 )
 self.structures.append(structure)

 def __call__(self, h: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
 """
 Evaluate nested variogram at distance h

 Parameters
 ----------
 h : float or np.ndarray
 Separation distance(s)

 Returns
 -------
 float or np.ndarray
 Variogram value(s)
 """
 h = np.asarray(h, dtype=np.float64)
 gamma = np.full_like(h, self.nugget, dtype=np.float64)

 for struct in self.structures:
 model_class = self.model_classes[struct.model_type]
 # Create model instance with parameters
 model = model_class(nugget=0.0, sill=1.0, range_param=1.0)
 # Evaluate model at h and scale by sill and range
 gamma += struct.sill * model._model_function(h / struct.range)

 return gamma

 def total_sill(self) -> float:
 """Total sill = nugget + sum of partial sills"""
 return self.nugget + sum(s.sill for s in self.structures)

 def effective_range(self) -> float:
 """Effective range (largest range in the model)"""
 if not self.structures:
 return 0.0
 return max(s.range for s in self.structures)

 def get_parameters(self) -> Dict:
 """Get all model parameters"""
 return {
 'nugget': self.nugget,
 'total_sill': self.total_sill(),
 'structures': [
 {
 'model': s.model_type,
 'sill': s.sill,
 'range': s.range,
 }
 for s in self.structures
 ]
 }

 def __str__(self):
 parts = [f"Nugget: {self.nugget:.3f}"]
 for i, struct in enumerate(self.structures, 1):
 parts.append(f"Structure {i}: {struct}")
 return "\n".join(parts)

def fit_nested_variogram(
 lags: npt.NDArray[np.float64],
 semivariance: npt.NDArray[np.float64],
 n_structures: int = 2,
 model_types: Optional[List[str]] = None,
 nugget_bounds: Tuple[float, float] = (0.0, None),
 weights: Optional[npt.NDArray[np.float64]] = None,
) -> NestedVariogram:
 """
 Fit nested variogram model to experimental variogram

 Uses global optimization (differential evolution) followed by
 local refinement to find optimal parameters.

 Parameters
 ----------
 lags : np.ndarray
 Lag distances (experimental variogram x-axis)
 semivariance : np.ndarray
 Semivariance values (experimental variogram y-axis)
 n_structures : int
 Number of structures to fit (default 2)
 model_types : list of str, optional
 Model type for each structure
 If None, uses 'spherical' for all
 nugget_bounds : tuple
 (min, max) for nugget. If max is None, uses max(semivariance)
 weights : np.ndarray, optional
 Weights for each lag (e.g., number of pairs)

 Returns
 -------
 NestedVariogram
 Fitted nested variogram model

 Examples
 --------
 >>> # Fit 2-structure nested model (nugget + 2 spherical)
 >>> lags = np.array([10, 20, 30, 40, 50])
 >>> gamma = np.array([0.1, 0.4, 0.7, 0.85, 0.92])
 >>> model = fit_nested_variogram(lags, gamma, n_structures=2)
 """
 lags = np.asarray(lags, dtype=np.float64)
 semivariance = np.asarray(semivariance, dtype=np.float64)

 if len(lags) != len(semivariance):
 raise ValueError("lags and semivariance must have same length")

 # Default model types
 if model_types is None:
 model_types = ['spherical'] * n_structures

 if len(model_types) != n_structures:
 raise ValueError(f"model_types must have length {n_structures}")

 logger.debug(f"Fitting nested variogram with {n_structures} structures")

 # Set up bounds
 max_lag = float(np.max(lags))
 max_semivar = float(np.max(semivariance))

 nugget_max = nugget_bounds[1] if nugget_bounds[1] is not None else max_semivar * NUGGET_MAX_FRACTION

 # Bounds: [nugget, sill1, range1, sill2, range2, ...]
 bounds: List[Tuple[float, float]] = [(nugget_bounds[0], nugget_max)]
 for _ in range(n_structures):
 # Sill bounds
 bounds.append((MIN_SEMIVARIANCE_RATIO * max_semivar, MAX_SEMIVARIANCE_RATIO * max_semivar))
 # Range bounds
 bounds.append((MIN_RANGE_RATIO * max_lag, MAX_RANGE_RATIO * max_lag))

 # Weights
 if weights is None:
 weights = np.ones_like(lags)
 else:
 weights = np.asarray(weights, dtype=np.float64)

 # Objective function (weighted sum of squared residuals)
 def objective(params: npt.NDArray[np.float64]) -> float:
 nugget = params[0]

 # Penalty for invalid parameters
 if nugget < 0:
 return PENALTY_VALUE

 nested_model = NestedVariogram(nugget=nugget)

 for i in range(n_structures):
 sill = params[1 + i * 2]
 range_param = params[2 + i * 2]

 # Penalty for invalid parameters
 if sill < 0 or range_param < 0:
 return PENALTY_VALUE

 nested_model.add_structure(
 model_type=model_types[i],
 sill=sill,
 range=range_param
 )

 # Predicted semivariance (vectorized)
 gamma_pred = nested_model(lags)

 # Weighted sum of squared residuals
 residuals = semivariance - gamma_pred
 wss = float(np.sum(weights * residuals**2))

 return wss

 logger.debug("Starting global optimization (differential evolution)")

 # Global optimization with differential evolution
 result = differential_evolution(
 objective,
 bounds,
 seed=RANDOM_SEED,
 maxiter=MAX_ITERATIONS_GLOBAL,
 atol=OPTIMIZATION_ATOL,
 tol=CONVERGENCE_TOLERANCE,
 workers=1
 )

 logger.debug("Refining with local optimization (L-BFGS-B)")

 # Local refinement
 result_local = minimize(
 objective,
 result.x,
 method='L-BFGS-B',
 bounds=bounds
 )

 # Use local result if successful, otherwise global result
 best_params = result_local.x if result_local.success else result.x
 final_wss = objective(best_params)

 logger.debug(f"Optimization complete. Final WSS: {final_wss:.6f}")

 # Build final model
 nugget = float(best_params[0])
 final_model = NestedVariogram(nugget=nugget)

 for i in range(n_structures):
 sill = float(best_params[1 + i * 2])
 range_param = float(best_params[2 + i * 2])

 final_model.add_structure(
 model_type=model_types[i],
 sill=sill,
 range=range_param
 )

 logger.info(f"Fitted nested variogram: {final_model.get_parameters()}")

 return final_model

def auto_fit_nested_variogram(
 lags: npt.NDArray[np.float64],
 semivariance: npt.NDArray[np.float64],
 max_structures: int = 3,
 weights: Optional[npt.NDArray[np.float64]] = None,
) -> NestedVariogram:
 """
 Automatically determine optimal number of structures

 Fits models with 1, 2, ..., max_structures and selects the best
 based on AIC (Akaike Information Criterion).

 Parameters
 ----------
 lags : np.ndarray
 Lag distances
 semivariance : np.ndarray
 Semivariance values
 max_structures : int
 Maximum number of structures to try
 weights : np.ndarray, optional
 Weights for fitting

 Returns
 -------
 NestedVariogram
 Best model selected by AIC
 """
 n = len(lags)
 best_aic = np.inf
 best_model = None

 for n_struct in range(1, max_structures + 1):
 try:
 model = fit_nested_variogram(
 lags, semivariance,
 n_structures=n_struct,
 weights=weights
 )

 # Calculate residuals
 gamma_pred = model(lags)
 residuals = semivariance - gamma_pred

 # Calculate AIC
 # AIC = n·ln(RSS/n) + 2k
 # where k = number of parameters
 k = 1 + 2 * n_struct # nugget + (sill, range) per structure
 rss = np.sum(residuals**2)
 aic = n * np.log(rss / n) + 2 * k

 if aic < best_aic:
 best_aic = aic
 best_model = model

 except Exception:
 continue

 if best_model is None:
 raise RuntimeError("Failed to fit any nested variogram model")

 return best_model
