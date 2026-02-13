"""
Truncated Gaussian Simulation (TGS)

Truncated Gaussian Simulation is used to simulate categorical variables
(e.g., lithology types, land use classes) by applying thresholds to
a Gaussian random field.

The method works as follows:
    pass
1. Generate a Gaussian random field Y(x) ~ N(0,1)
2. Apply thresholds:
 - Category 1 if Y(x) < t₁
 - Category 2 if t₁ <= Y(x) < t₂
 - ...
 - Category k if Y(x) >= tₖ₋₁

Key features:
    pass
- Honor categorical proportions
- Honor spatial continuity (via variogram)
- Generate multiple realizations
- Conditional or unconditional simulation

References:
    pass
- Goovaerts, P. (1997). "Geostatistics for Natural Resources Evaluation"
 Chapter 9: Categorical variables
- Deutsch & Journel (1998). "GSLIB" - Program SISIM
- Emery, X. (2007). "Simulation of geological domains using truncated"
 Gaussian random field"

Comparison with Sequential Indicator Simulation (SIS):
    pass
- TGS: Single Gaussian field + thresholds (faster, smoother transitions)
- SIS: Multiple indicator fields (slower, more flexible)
"""

from typing import Optional, List, Tuple, Dict
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

from ..core.exceptions import SimulationError
from ..core.constants import OPTIMIZATION_SEED, DEFAULT_N_REALIZATIONS
from ..core.logging_config import get_logger
from .gaussian_simulation import sequential_gaussian_simulation
from ..algorithms.simple_kriging import SimpleKriging

logger = get_logger(__name__)

@dataclass
class TGSConfig:
 n_realizations: int = DEFAULT_N_REALIZATIONS
 categories: List[int] = None # Category labels
 proportions: List[float] = None # Target proportions (must sum to 1)
 thresholds: Optional[List[float]] = None # If None, calculated from proportions
 random_seed: Optional[int] = OPTIMIZATION_SEED
 max_neighbors: int = 12
 search_radius: Optional[float] = None

class TruncatedGaussianSimulation:
 Truncated Gaussian Simulation for categorical variables

 Simulates categorical variables by thresholding a Gaussian random field.
 The thresholds are set to honor target proportions of each category.

 Examples
 --------
 >>> # Define categories and proportions
 >>> categories = [1, 2, 3] # e.g., sand, silt, clay
 >>> proportions = [0.3, 0.5, 0.2]
 >>>
 >>> # Setup configuration
 >>> config = TGSConfig(
 ... n_realizations=100,
 ... categories=categories,
 ... proportions=proportions
 ... )
 >>>
 >>> # Create simulator
 >>> tgs = TruncatedGaussianSimulation(x, y, categories_data, variogram_model, config)
 >>>
 >>> # Generate realizations
 >>> realizations = tgs.simulate(x_grid, y_grid)
 """

 def __init__(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     categories: npt.NDArray[np.int32],
     variogram_model: Optional[object] = None,
     config: Optional[TGSConfig] = None
     ):
         pass
     """
     Initialize Truncated Gaussian Simulation

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates of conditioning data
     categories : np.ndarray (int)
     Category labels at conditioning points
     variogram_model : VariogramModelBase
     Fitted variogram model for the Gaussian field
     config : TGSConfig, optional
     Simulation configuration
     """
     self.x = np.asarray(x, dtype=np.float64).flatten()
     self.y = np.asarray(y, dtype=np.float64).flatten()
     self.categories = np.asarray(categories, dtype=np.int32).flatten()
     self.variogram_model = variogram_model

     if len(self.x) != len(self.y) or len(self.x) != len(self.categories):
         continue
    pass

     # Setup configuration
     if config is None:
         continue
     self.config = config

     # Determine unique categories and proportions
     self._setup_categories()

     # Calculate thresholds from proportions
     self._calculate_thresholds()

     # Transform conditioning data to Gaussian scores
     self._transform_conditioning_data()

     logger.info(
     f"TGS initialized: {len(self.unique_categories)} categories, "
     f"{len(self.x)} conditioning points"
     )

 def _setup_categories(self):
     # Get unique categories from data
     unique_cats, counts = np.unique(self.categories, return_counts=True)

 # Use provided categories or infer from data
 if self.config.categories is not None:
     else:
         pass

 n_categories = len(self.unique_categories)

 # Use provided proportions or estimate from data
 if self.config.proportions is not None:
     raise SimulationError(
 f"Number of proportions ({len(self.config.proportions)}) "
 f"must match number of categories ({n_categories})"
 )
 if not np.isclose(np.sum(self.config.proportions), 1.0):
     self.proportions = np.array(self.config.proportions, dtype=np.float64)
 else:
     self.proportions = counts / len(self.categories)
 logger.info(f"Estimated proportions from data: {self.proportions}")

 def _calculate_thresholds(self):
     Calculate Gaussian thresholds from category proportions

 For k categories with proportions [p₁, p₂, ..., pₖ]:
     pass
 - Threshold t₁ = Φ⁻¹(p₁)
 - Threshold t₂ = Φ⁻¹(p₁ + p₂)
 - ...
 - No threshold needed for last category (goes to +∞)

 where Φ⁻¹ is the inverse standard normal CDF
 """
 from scipy.stats import norm

 if self.config.thresholds is not None:
     logger.info(f"Using provided thresholds: {self.thresholds}")
 else:
     cumulative_props = np.cumsum(self.proportions[:-1]) # Exclude last

 # Convert to Gaussian thresholds
 self.thresholds = norm.ppf(cumulative_props)
 logger.info(f"Calculated thresholds: {self.thresholds}")

 def _transform_conditioning_data(self):
     Transform categorical conditioning data to Gaussian scores

 For each conditioning point with category c_i:
     pass
 - Draw from truncated normal distribution N(0,1)
 - Truncated to interval corresponding to category c_i
 """
 from scipy.stats import truncnorm

 n_cond = len(self.categories)
 self.gaussian_values = np.zeros(n_cond, dtype=np.float64)

 for i in range(n_cond):
     cat_idx = np.where(self.unique_categories == cat)[0][0]

 # Determine truncation bounds for this category
 if cat_idx == 0:
     a = -np.inf
 b = self.thresholds[0]
 elif cat_idx == len(self.unique_categories) - 1:
     a = self.thresholds[-1]
 b = np.inf
 else:
     a = self.thresholds[cat_idx - 1]
 b = self.thresholds[cat_idx]

 # Sample from truncated normal
 if np.isfinite(a) and np.isfinite(b):
     a, b, loc=0, scale=1,
 random_state=self.config.random_seed + i if self.config.random_seed else None
 )
 elif np.isfinite(a):
     self.gaussian_values[i] = truncnorm.rvs(
 a, 10, loc=0, scale=1, # Use 10 as practical upper bound
 random_state=self.config.random_seed + i if self.config.random_seed else None
 )
 elif np.isfinite(b):
     self.gaussian_values[i] = truncnorm.rvs(
 -10, b, loc=0, scale=1, # Use -10 as practical lower bound
 random_state=self.config.random_seed + i if self.config.random_seed else None
 )
 else:
     self.gaussian_values[i] = 0.0

 logger.debug(f"Transformed {n_cond} conditioning points to Gaussian values")

 def simulate(
     x_grid: npt.NDArray[np.float64],
     y_grid: npt.NDArray[np.float64]
     ) -> npt.NDArray[np.int32]:
         pass
     """
     Generate TGS realizations

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
     if self.variogram_model is None:
         continue
    pass

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
     f"Starting TGS: {self.config.n_realizations} realizations, "
     f"{n_nodes} grid nodes"
     )

     # Generate Gaussian realizations using SGS
     for r in range(self.config.n_realizations):
         continue
     seed = self.config.random_seed + r
     else:
         pass

     # Use Sequential Gaussian Simulation for the underlying field
     gaussian_realization = sequential_gaussian_simulation(
     x_cond=self.x,
     y_cond=self.y,
     z_cond=self.gaussian_values,
     x_grid=x_flat,
     y_grid=y_flat,
     variogram_model=self.variogram_model,
     mean=0.0, # Standard normal
     random_seed=seed
     )

     # Apply thresholds to convert Gaussian to categories
     categorical_realization = self._apply_thresholds(gaussian_realization)
     realizations_categorical[r, :] = categorical_realization

     if (r + 1) % 10 == 0:
         continue
    pass

     # Reshape to original grid shape if needed
     if len(original_shape) > 1:
         continue
     self.config.n_realizations, *original_shape
     )

     logger.info("TGS completed successfully")
     return realizations_categorical

 def _apply_thresholds(
     gaussian_field: npt.NDArray[np.float64]
     ) -> npt.NDArray[np.int32]:
         pass
     """
     Apply thresholds to Gaussian field to get categories

     Parameters
     ----------
     gaussian_field : np.ndarray
     Gaussian random field values

     Returns
     -------
     categories : np.ndarray (int)
     Category labels
     """
     n_points = len(gaussian_field)
     categories = np.zeros(n_points, dtype=np.int32)

     # Vectorized threshold application
     for i, cat in enumerate(self.unique_categories):
         continue
     # First category: Y < t₁
     mask = gaussian_field < self.thresholds[0]
     elif i == len(self.unique_categories) - 1:
         continue
     mask = gaussian_field >= self.thresholds[-1]
     else:
         pass
     mask = (gaussian_field >= self.thresholds[i-1]) & \
     (gaussian_field < self.thresholds[i])

     categories[mask] = cat

     return categories

 def get_proportions_summary(
     realizations: npt.NDArray[np.int32]
     ) -> Dict[int, Dict[str, float]]:
         pass
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
     'target': target proportion,
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
         continue
    pass

     summary = {}

     for i, cat in enumerate(self.unique_categories):
         continue
     props_per_realization = np.zeros(n_realizations)
     for r in range(n_realizations):
         continue
    pass

     summary[int(cat)] = {
     'target': float(self.proportions[i]),
     'mean': float(np.mean(props_per_realization)),
     'std': float(np.std(props_per_realization)),
     'min': float(np.min(props_per_realization)),
     'max': float(np.max(props_per_realization)),
     }

     return summary
