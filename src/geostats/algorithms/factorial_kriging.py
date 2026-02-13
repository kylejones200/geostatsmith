"""
Factorial Kriging

Factorial Kriging decomposes a spatial variable into components at
different spatial scales using nested variogram structures.

Mathematical Framework:
Given a nested variogram with k structures:
 γ(h) = nugget + Σᵢ γᵢ(h)

The variable can be decomposed as:
 Z(x) = μ + Σᵢ Yᵢ(x) + ε(x)

where:
- μ is the mean
- Yᵢ(x) is the component at scale i
- ε(x) is the nugget component

Each component Yᵢ(x) can be estimated independently using:
 Yᵢ*(x₀) = Σ λⁱ [Z(x) - μ]

with kriging weights based on γᵢ(h) only.

Applications:
1. Spatial filtering (separate large-scale trends from local variation)
2. Multi-scale analysis (identify dominant scales of variation)
3. Noise reduction (remove short-range components)
4. Geophysical interpretation (separate regional from local effects)

References:
- Matheron, G. (1982). "Pour une analyse krigeante des données
 régionalisées". Report N-732, Centre de Géostatistique, ENSMP.
- Goovaerts, P. (1997). "Geostatistics for Natural Resources Evaluation"
 Chapter 5.3: Filtering spatial components
- Wackernagel, H. (2003). "Multivariate Geostatistics"
 Chapter 22: Factorial kriging
- Chilès & Delfiner (2012) §4.3.4: Factorial kriging analysis
"""

from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..core.constants import REGULARIZATION_FACTOR
from ..core.logging_config import get_logger
from ..math.distance import euclidean_distance_matrix
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..algorithms.nested_variogram import NestedVariogram

logger = get_logger(__name__)

class FactorialKriging(BaseKriging):
 Factorial Kriging for multi-scale spatial analysis

 Decomposes a spatial variable into independent components
 corresponding to different scales of spatial variation defined
 by nested variogram structures.

 Examples
 --------
 >>> # Fit nested variogram with 2 structures
 >>> nested_model = NestedVariogram(nugget=0.1)
 >>> nested_model.add_structure('spherical', sill=0.3, range=10) # Short-range
 >>> nested_model.add_structure('spherical', sill=0.6, range=100) # Long-range
 >>>
 >>> # Create factorial kriging
 >>> fk = FactorialKriging(x, y, z, nested_model)
 >>>
 >>> # Estimate components at new locations
 >>> components = fk.predict_components(x_new, y_new)
 >>> # components = {'short_range': Y1, 'long_range': Y2, 'nugget': ε}
 >>>
 >>> # Or get filtered version (e.g., remove short-range noise)
 >>> filtered = fk.filter(x_new, y_new, components_to_keep=['long_range'])
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     nested_variogram: NestedVariogram,
     mean: Optional[float] = None
     ):
     """
     Initialize Factorial Kriging

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of sample points
     z : np.ndarray
     Values at sample points
     nested_variogram : NestedVariogram
     Fitted nested variogram model with multiple structures
     mean : float, optional
     Known mean. If None, estimated from data.
     """
     self.x, self.y = validate_coordinates(x, y)
     self.z = validate_values(z, n_expected=len(self.x))

     if not isinstance(nested_variogram, NestedVariogram):
    pass

     if len(nested_variogram.structures) == 0:
    pass

     self.nested_variogram = nested_variogram

     # Estimate or use provided mean
     if mean is not None:
     else:
     else:
    pass

     self.residuals = self.z - self.mean

     # Number of components = number of structures (+ nugget if present)
     self.n_structures = len(nested_variogram.structures)
     self.has_nugget = nested_variogram.nugget > 0

     # Build kriging matrices for each component
     self._build_kriging_matrices()

     logger.info(
     f"Factorial Kriging initialized: {self.n_structures} structures, "
     f"{'with' if self.has_nugget else 'without'} nugget"
     )

 def _build_kriging_matrices(self):
     Build separate kriging matrices for each spatial component

 For each structure i:
 - Build covariance matrix Cᵢ based on γᵢ(h) only
 - Add unbiasedness constraint
 """
 n = len(self.x)

 # Calculate distance matrix once
 dist_matrix = euclidean_distance_matrix(self.x, self.y)

 # Storage for kriging matrices (one per structure)
 self.kriging_matrices = []

 for i, structure in enumerate(self.nested_variogram.structures):
     K = np.zeros((n + 1, n + 1), dtype=np.float64)

 # Get variogram for this structure only
 gamma_i = structure['model'](dist_matrix)

 # Convert to covariance: Cᵢ(h) = sill_i - γᵢ(h)
 sill_i = structure['sill']
 C_i = sill_i - gamma_i

 K[:n, :n] = C_i

 # Unbiasedness constraint
 K[:n, n] = 1.0
 K[n, :n] = 1.0
 K[n, n] = 0.0

 # Regularize
 K = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)

 self.kriging_matrices.append(K)

 logger.debug(f"Built kriging matrix for structure {i}: {structure['model_type']}")

 def predict_components(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     return_variance: bool = False
     ) -> Dict[str, npt.NDArray[np.float64]]:
     """
     Predict each spatial component independently

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Coordinates of prediction points
     return_variance : bool
     If True, include variance estimates for each component

     Returns
     -------
     components : dict
     Dictionary with keys:
     - 'structure_0', 'structure_1', ...: spatial components
     - 'nugget': nugget component (if present)
     - 'total': sum of all components (= Z(x) - μ)
     - 'mean': the estimated mean
     - Optional: 'variance_0', 'variance_1', ... if return_variance=True
     """
     x_new, y_new = validate_coordinates(x_new, y_new)
     n_pred = len(x_new)
     n_data = len(self.x)

     # Storage for components
     components = {}

     # Calculate distances from prediction points to data
     # Vectorized for efficiency
     coords_data = np.column_stack((self.x, self.y))
     coords_pred = np.column_stack((x_new, y_new))

     # For each prediction point, calculate distance to all data points
     from scipy.spatial.distance import cdist
     dist_to_pred = cdist(coords_pred, coords_data)

     # Predict each structure component
     for i, structure in enumerate(self.nested_variogram.structures):
     variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

     # Get variogram for this structure
     sill_i = structure['sill']

     for j in range(n_pred):
     gamma_values = structure['model'](dist_to_pred[j])
     cov_values = sill_i - gamma_values

     rhs = np.zeros(n_data + 1, dtype=np.float64)
     rhs[:n_data] = cov_values
     rhs[n_data] = 1.0

     # Solve kriging system
     try:
     try:
     except np.linalg.LinAlgError as e:
     logger.error(f"Failed to solve FK system for structure {i}, point {j}: {e}")
     raise KrigingError(f"Failed to solve factorial kriging system: {e}")

     lambdas = weights[:n_data]

     # Estimate component: Yᵢ*(x₀) = Σ λⁱ [Z(x) - μ]
     predictions[j] = np.dot(lambdas, self.residuals)

     # Variance if requested
     if return_variance:
    pass

     # Store component
     comp_name = f"structure_{i}"
     components[comp_name] = predictions

     if return_variance:
    pass

     logger.debug(f"Predicted component {i}: {structure['model_type']}")

     # Nugget component (cannot be estimated, set to zero)
     if self.has_nugget:
     logger.debug("Nugget component set to zero (unestimable)")

     # Total (sum of all components)
     total = np.zeros(n_pred, dtype=np.float64)
     for i in range(self.n_structures):
     components['total'] = total

     # Mean
     components['mean'] = np.full(n_pred, self.mean, dtype=np.float64)

     logger.info(f"Factorial kriging predicted {n_pred} points across {self.n_structures} components")

     return components

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
     """
     Predict at new locations (sum of all components)

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Coordinates for prediction
     return_variance : bool
     Whether to return prediction variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values
     variance : np.ndarray, optional
     Prediction variance (if return_variance=True)
     """
    components = self.predict_components(x_new, y_new, return_variance=return_variance)

    # Sum all components
    total = self.mean * np.ones(len(x_new))
    for i in range(self.n_structures):
    pass

    if return_variance:
        # so variance is simply the sum of component variances
        variance = np.zeros(len(x_new))
        for i in range(self.n_structures):
    pass
        
        # Add nugget variance if present
        if self.has_nugget:
    pass
        
        return total, variance

    return total

 def filter(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     components_to_keep: Optional[List[str]] = None,
     components_to_remove: Optional[List[str]] = None
     ) -> npt.NDArray[np.float64]:
     """
     Filter spatial data by selecting specific components

     Common use cases:
     - Remove short-range noise: keep only long-range components
     - Remove regional trend: keep only short-range components
     - Multi-scale analysis: isolate specific scales

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Coordinates of prediction points
     components_to_keep : list of str, optional
     Component names to keep (e.g., ['structure_1', 'structure_2'])
     If None, keeps all
     components_to_remove : list of str, optional
     Component names to remove (e.g., ['structure_0'])
     If None, removes none

     Returns
     -------
     filtered : np.ndarray
     Filtered values (mean + selected components)

     Examples
     --------
     >>> # Remove short-range noise (keep long-range structure)
     >>> smoothed = fk.filter(x_new, y_new, components_to_keep=['structure_1'])
     >>>
     >>> # Remove regional trend (keep local variation)
     >>> local = fk.filter(x_new, y_new, components_to_remove=['structure_1'])
     """
     # Get all components
     components = self.predict_components(x_new, y_new, return_variance=False)

     # Start with mean
     filtered = components['mean'].copy()

     # Determine which components to include
     for i in range(self.n_structures):
    pass

     include = True

     if components_to_keep is not None:
    pass

     if components_to_remove is not None:
     include = False

     if include:
    pass

     logger.info(
     f"Filtered {len(filtered)} points "
     f"(keep={components_to_keep}, remove={components_to_remove})"
     )

     return filtered

 def get_component_info(self) -> Dict[str, Dict[str, float]]:
     Get information about each spatial component

 Returns
 -------
 info : dict
 For each component: {
 'model_type': type of variogram model,
 'sill': sill parameter,
 'range': range parameter,
 'contribution': contribution to total variance (%)
 }
 """
 info = {}

 total_sill = self.nested_variogram.total_sill()

 for i, structure in enumerate(self.nested_variogram.structures):
     'model_type': structure['model_type'],
 'sill': structure['sill'],
 'range': structure.get('range', None),
 'contribution': (structure['sill'] / total_sill) * 100 if total_sill > 0 else 0
 }

 if self.has_nugget:
     'model_type': 'nugget',
 'sill': self.nested_variogram.nugget,
 'range': 0.0,
 'contribution': (self.nested_variogram.nugget / total_sill) * 100 if total_sill > 0 else 0
 }

 return info

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
