"""
Plurigaussian Simulation (PGS)

Plurigaussian simulation extends Truncated Gaussian Simulation to handle
complex categorical relationships using multiple independent Gaussian
random fields.

Key concepts:
- Uses 2+ independent Gaussian fields: Y₁(x), Y₂(x), ...
- Categories defined by regions in multi-dimensional Gaussian space
- More flexible than TGS for modeling complex transitions
- Can honor asymmetric relationships between categories

Mathematical Framework:
For 2 Gaussian fields Y₁ and Y₂:
 Category = f(Y₁(x), Y₂(x))

where f is a rule function (lithotype rule) that partitions the
(Y₁, Y₂) plane into regions, each corresponding to a category.

Example lithotype rule for 3 facies:
 - Sand: Y₁ < t₁
 - Silt: Y₁ ≥ t₁ AND Y₂ < t₂
 - Clay: Y₁ ≥ t₁ AND Y₂ ≥ t₂

Applications:
1. Sedimentary facies modeling (complex transitions)
2. Ore type classification (multiple controlling factors)
3. Soil type mapping (multiple soil-forming processes)
4. Geological domains (structural + stratigraphic controls)

Advantages over Truncated Gaussian:
- More flexible category relationships
- Can model forbidden transitions
- Better for asymmetric relationships
- Multiple controlling processes

References:
- Le Loc'h, G. & Galli, A. (1997). "Truncated plurigaussian method:
 Theoretical and practical points of view". In Geostatistics Wollongong '96.
- Armstrong, M. et al. (2011). "Plurigaussian Simulations in Geosciences"
- Emery, X. (2007). "Simulation of geological domains using the plurigaussian
 model: New developments and computer programs"
"""

from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

from ..core.exceptions import SimulationError
from ..core.constants import OPTIMIZATION_SEED, DEFAULT_N_REALIZATIONS
from ..core.logging_config import get_logger
from .gaussian_simulation import sequential_gaussian_simulation

logger = get_logger(__name__)

@dataclass
class PlurigaussianConfig:
class PlurigaussianConfig:
 n_realizations: int = DEFAULT_N_REALIZATIONS
 n_gaussian_fields: int = 2 # Number of independent Gaussian fields
 categories: List[int] = None # Category labels
 rule_function: Optional[Callable] = None # Lithotype rule function
 random_seed: Optional[int] = OPTIMIZATION_SEED
 max_neighbors: int = 12
 search_radius: Optional[float] = None

class PlurigaussianSimulation:
class PlurigaussianSimulation:
 Plurigaussian Simulation for complex categorical variables

 Uses multiple independent Gaussian fields with a rule function
 to simulate complex categorical relationships.

 Examples
 --------
 >>> # Define lithotype rule for 3 facies using 2 Gaussian fields
 >>> def lithotype_rule(y1, y2):
 ... '''
 ... Sand: Y1 < -0.5
 ... Silt: Y1 >= -0.5 AND Y2 < 0
 ... Clay: Y1 >= -0.5 AND Y2 >= 0
 ... '''
 ... facies = np.zeros(len(y1), dtype=np.int32)
 ... sand_mask = y1 < -0.5
 ... silt_mask = (y1 >= -0.5) & (y2 < 0)
 ... clay_mask = (y1 >= -0.5) & (y2 >= 0)
 ... facies[sand_mask] = 1 # Sand
 ... facies[silt_mask] = 2 # Silt
 ... facies[clay_mask] = 3 # Clay
 ... return facies
 >>>
 >>> # Create configuration
 >>> config = PlurigaussianConfig(
 ... n_realizations=100,
 ... n_gaussian_fields=2,
 ... categories=[1, 2, 3],
 ... rule_function=lithotype_rule
 ... )
 >>>
 >>> # Initialize simulator
 >>> pgs = PlurigaussianSimulation(
 ... x, y, categories_data,
 ... variogram_models=[variogram1, variogram2],
 ... config=config
 ... )
 >>>
 >>> # Generate realizations
 >>> realizations = pgs.simulate(x_grid, y_grid)
 """

 def __init__(
 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     categories: npt.NDArray[np.int32],
     variogram_models: List[object],
     config: Optional[PlurigaussianConfig] = None
     ):
     """
     Initialize Plurigaussian Simulation

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of conditioning data
     categories : np.ndarray (int)
     Category labels at conditioning points
     variogram_models : list of VariogramModelBase
     Variogram models for each Gaussian field
     config : PlurigaussianConfig, optional
     Simulation configuration
     """
     self.x = np.asarray(x, dtype=np.float64).flatten()
     self.y = np.asarray(y, dtype=np.float64).flatten()
     self.categories = np.asarray(categories, dtype=np.int32).flatten()

     if len(self.x) != len(self.y) or len(self.x) != len(self.categories):
     if len(self.x) != len(self.y) or len(self.x) != len(self.categories):

     # Setup configuration
     if config is None:
     if config is None:
     self.config = config

     # Validate variogram models
     if len(variogram_models) != config.n_gaussian_fields:
     if len(variogram_models) != config.n_gaussian_fields:
     f"Number of variogram models ({len(variogram_models)}) must match "
     f"number of Gaussian fields ({config.n_gaussian_fields})"
     )
     self.variogram_models = variogram_models

     # Validate rule function
     if config.rule_function is None:
     if config.rule_function is None:
     self.rule_function = config.rule_function

     # Setup categories
     self._setup_categories()

     # Transform conditioning data to Gaussian fields
     self._transform_conditioning_data()

     logger.info(
     f"Plurigaussian Simulation initialized: "
     f"{len(self.unique_categories)} categories, "
     f"{config.n_gaussian_fields} Gaussian fields, "
     f"{len(self.x)} conditioning points"
     )

 def _setup_categories(self):
 def _setup_categories(self):
     unique_cats = np.unique(self.categories)

 if self.config.categories is not None:
     else:

 self.n_categories = len(self.unique_categories)

 logger.debug(f"Categories: {self.unique_categories}")

 def _transform_conditioning_data(self):
 def _transform_conditioning_data(self):
     Transform categorical conditioning data to Gaussian field values

 For each conditioning point with category c:
 1. Find region in Gaussian space corresponding to category c
 2. Sample from that region (using rejection sampling or truncated normal)

 This is simplified - in practice, would use gibbs sampling or MCMC
 """
 n_cond = len(self.categories)
 n_fields = self.config.n_gaussian_fields

 # Storage for Gaussian values (one array per field)
 self.gaussian_fields_cond = [
 np.zeros(n_cond, dtype=np.float64) for _ in range(n_fields)
 ]

 # For each conditioning point, find valid Gaussian values
 # that satisfy the rule function
 for i in range(n_cond):

     # Use rejection sampling to find Gaussian values
 # that produce the correct category
 max_attempts = 1000
 found = False

 for attempt in range(max_attempts):
     if self.config.random_seed is not None:

 y_samples = np.random.randn(n_fields)

 # Check if this produces the correct category
 test_cat = self.rule_function(*y_samples)[0] if n_fields > 1 else self.rule_function(y_samples[0])[0]

 if test_cat == cat:
     for j in range(n_fields):
 found = True
 break

 if not found:
     f"Could not find valid Gaussian values for category {cat} "
 f"at point {i} after {max_attempts} attempts. Using approximate values."
 )
 # Fallback: use mean of region (simplified)
 for j in range(n_fields):

     logger.debug(f"Transformed {n_cond} conditioning points to {n_fields} Gaussian fields")

 def simulate(
 def simulate(
     x_grid: npt.NDArray[np.float64],
     y_grid: npt.NDArray[np.float64]
     ) -> npt.NDArray[np.int32]:
     """
     Generate Plurigaussian realizations

     Parameters
     ----------
     x_grid, y_grid : np.ndarray
     Grid coordinates for simulation

     Returns
     -------
     realizations : np.ndarray (int)
     Simulated category labels
     Shape: (n_realizations, n_grid_points) or (n_realizations, *grid_shape)
     """
     # Flatten grid if needed
     original_shape = x_grid.shape
     x_flat = x_grid.flatten()
     y_flat = y_grid.flatten()
     n_nodes = len(x_flat)

     # Storage for realizations
     realizations_categorical = np.zeros(
     (self.config.n_realizations, n_nodes),
     dtype=np.int32
     )

     logger.info(
     f"Starting Plurigaussian Simulation: {self.config.n_realizations} realizations, "
     f"{n_nodes} grid nodes, {self.config.n_gaussian_fields} Gaussian fields"
     )

     # Generate realizations
     for r in range(self.config.n_realizations):
     for r in range(self.config.n_realizations):
     gaussian_realizations = []

     for field_idx in range(self.config.n_gaussian_fields):
     for field_idx in range(self.config.n_gaussian_fields):
     seed = self.config.random_seed + r * self.config.n_gaussian_fields + field_idx
     else:
     else:

     # Use SGS for each independent Gaussian field
     gaussian_field = sequential_gaussian_simulation(
     x_cond=self.x,
     y_cond=self.y,
     z_cond=self.gaussian_fields_cond[field_idx],
     x_grid=x_flat,
     y_grid=y_flat,
     variogram_model=self.variogram_models[field_idx],
     mean=0.0,
     random_seed=seed
     )

     gaussian_realizations.append(gaussian_field)

     # Apply rule function to convert Gaussian fields to categories
     categorical_realization = self.rule_function(*gaussian_realizations)
     realizations_categorical[r, :] = categorical_realization

     if (r + 1) % 10 == 0:
     if (r + 1) % 10 == 0:

     # Reshape to original grid shape if needed
     if len(original_shape) > 1:
     if len(original_shape) > 1:
     self.config.n_realizations, *original_shape
     )

     logger.info("Plurigaussian simulation completed successfully")
     return realizations_categorical

 def get_proportions_summary(
 def get_proportions_summary(
     realizations: npt.NDArray[np.int32]
     ) -> Dict[int, Dict[str, float]]:
     """
     Calculate realized proportions for each category

     Parameters
     ----------
     realizations : np.ndarray
     Simulated realizations

     Returns
     -------
     summary : dict
     For each category: {
     'mean': mean realized proportion,
     'std': std of realized proportions,
     'min': min realized proportion,
     'max': max realized proportion
     }
     """
     n_realizations = realizations.shape[0]
     n_nodes = realizations.shape[1] if realizations.ndim == 2 else np.prod(realizations.shape[1:])

     # Flatten if needed
     if realizations.ndim > 2:
     if realizations.ndim > 2:

     summary = {}

     for cat in self.unique_categories:
     for cat in self.unique_categories:
     props_per_realization = np.zeros(n_realizations)
     for r in range(n_realizations):
     for r in range(n_realizations):

     summary[int(cat)] = {
     'mean': float(np.mean(props_per_realization)),
     'std': float(np.std(props_per_realization)),
     'min': float(np.min(props_per_realization)),
     'max': float(np.max(props_per_realization)),
     }

     return summary

def create_rectangular_rule(
def create_rectangular_rule(
 thresholds_y1: List[float],
 thresholds_y2: List[float],
 layout: str = 'grid'
    ) -> Callable:
 """
 Create a rectangular lithotype rule function

 Partitions the (Y₁, Y₂) plane into rectangular regions.

 Parameters
 ----------
 categories : list of int
 Category labels
 thresholds_y1 : list of float
 Thresholds for Y₁ axis
 thresholds_y2 : list of float
 Thresholds for Y₂ axis
 layout : str
 'grid': rectangular grid layout
 'hierarchical': hierarchical partitioning

 Returns
 -------
 rule_function : callable
 Function that takes (y1, y2) and returns categories

 Examples
 --------
 >>> # 3x2 grid of categories
 >>> rule = create_rectangular_rule(
 ... categories=[1, 2, 3, 4, 5, 6],
 ... thresholds_y1=[-1.0, 0.0, 1.0],
 ... thresholds_y2=[0.0]
 ... )
 """
 if layout == 'grid':
 if layout == 'grid':
     y2 = np.asarray(y2).flatten()
     n = len(y1)

     result = np.zeros(n, dtype=np.int32)

     # Digitize into bins
     bins_y1 = np.digitize(y1, thresholds_y1)
     bins_y2 = np.digitize(y2, thresholds_y2)

     # Map to categories
     n_bins_y2 = len(thresholds_y2) + 1

     for i in range(n):
     for i in range(n):
     if cat_idx < len(categories):
     if cat_idx < len(categories):
     else:
     else:

     return result

     return grid_rule

     else:
     else:
