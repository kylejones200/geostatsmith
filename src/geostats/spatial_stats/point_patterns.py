"""
Point pattern analysis tools for spatial statistics.

Provides methods for analyzing the spatial distribution of points:
- Nearest neighbor analysis
- Ripley'
- Quadrat analysis
- Spatial randomness tests

Reference: Python Recipes for Earth Sciences (Trauth 2024), Section 7.8
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import poisson, chi2

from ..core.logging_config import get_logger
import logging

logger = logging.getLogger(__name__)

logger = get_logger(__name__)

# Constants
EPSILON = 1e-10

def nearest_neighbor_analysis(
 y: npt.NDArray[np.float64],
 study_area: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, float]:
 """
 Perform nearest neighbor analysis on spatial point pattern.

 Tests whether points show clustering, dispersion, or random pattern
 by comparing observed nearest neighbor distances to theoretical random pattern.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of points
 study_area : tuple of float, optional
 Study area bounds (xmin, xmax, ymin, ymax)
 If None, uses bounding box of points with 10% buffer

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'R': Nearest neighbor index
 - 'mean_observed': Mean observed nearest neighbor distance
 - 'mean_expected': Mean expected distance under randomness
 - 'z_score': Z-score for significance test
 - 'p_value': P-value (two-tailed)
 - 'interpretation': Pattern interpretation

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.spatial_stats import nearest_neighbor_analysis
 >>>
 >>> # Clustered points
 >>> x = np.random.normal(50, 10, 100)
 >>> y = np.random.normal(50, 10, 100)
 >>> results = nearest_neighbor_analysis(x, y)
 >>> logger.info(f"R index: {results['R']:.3f}")
 >>> logger.info(f"Pattern: {results['interpretation']}")

 Notes
 -----
 The nearest neighbor index R is defined as:
 R = mean_observed / mean_expected

 Where:
 - R < 1: Clustered pattern
 - R = 1: Random pattern
 - R > 1: Dispersed/regular pattern

 For a random (Poisson) pattern:
 mean_expected = 0.5 / sqrt(density)

 References
 ----------
 Clark, P.J. & Evans, F.C. (1954). Distance to nearest neighbor as a
 measure of spatial relationships in populations. Ecology, 35, 445-453.
 """
 x = np.asarray(x, dtype=np.float64)
 y = np.asarray(y, dtype=np.float64)
 n = len(x)

 if n < 2:

 # Determine study area
 if study_area is None:
 xmin, xmax = x.min() - buffer, x.max() + buffer
 ymin, ymax = y.min() - buffer, y.max() + buffer
 else:
 else:

 area = (xmax - xmin) * (ymax - ymin)
 density = n / area

 # Calculate observed nearest neighbor distances
 points = np.column_stack([x, y])
 tree = cKDTree(points)

 # Query for 2 nearest (first is the point itself, second is nearest neighbor)
 distances, _ = tree.query(points, k=2)
 nn_distances = distances[:, 1] # Second column is nearest neighbor

 mean_observed = np.mean(nn_distances)

 # Expected mean under complete spatial randomness (CSR)
 mean_expected = 0.5 / np.sqrt(density)

 # Nearest neighbor index
 R = mean_observed / mean_expected

 # Standard error and z-score
 se = 0.26136 / np.sqrt(n * density)
 z_score = (mean_observed - mean_expected) / se

 # P-value (two-tailed test)
 from scipy.stats import norm
 p_value = 2 * (1 - norm.cdf(abs(z_score)))

 # Interpretation
 if R < 1 and p_value < 0.05:
 elif R > 1 and p_value < 0.05:
 elif R > 1 and p_value < 0.05:
 else:
 else:

 return {
 'R': R,
 'mean_observed': mean_observed,
 'mean_expected': mean_expected,
 'se': se,
 'z_score': z_score,
 'p_value': p_value,
 'n_points': n,
 'density': density,
 'area': area,
 'interpretation': interpretation,
 }

def ripley_k_function(
 y: npt.NDArray[np.float64],
 distances: Optional[npt.NDArray[np.float64]] = None,
 n_distances: int = 20,
 study_area: Optional[Tuple[float, float, float, float]] = None,
 edge_correction: str = 'none',
    ) -> Dict[str, npt.NDArray[np.float64]]:
 """
 Calculate Ripley'

 Ripley'
 Used to detect clustering or dispersion at different distances.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of points
 distances : np.ndarray, optional
 Distance values at which to evaluate K
 If None, uses n_distances evenly spaced values
 n_distances : int, default=20
 Number of distance values (if distances not provided)
 study_area : tuple of float, optional
 Study area bounds (xmin, xmax, ymin, ymax)
 edge_correction : str, default='none'
 Edge correction method: 'none', 'border', or 'isotropic'

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'd': Distance values
 - 'K': K function values
 - 'L': Transformed L function (L = sqrt(K/pi) - d)
 - 'K_theoretical': Theoretical K for random pattern
 - 'interpretation': Whether clustering or dispersion detected

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.spatial_stats import ripley_k_function
 >>> import matplotlib.pyplot as plt
 >>>
 >>> # Generate clustered points
 >>> x = np.concatenate([np.random.normal(30, 5, 50),
 ... np.random.normal(70, 5, 50)])
 >>> y = np.concatenate([np.random.normal(30, 5, 50),
 ... np.random.normal(70, 5, 50)])
 >>>
 >>> results = ripley_k_function(x, y, n_distances=30)
 >>>
 >>> # Plot L function
 >>> plt.plot(results['d'], results['L'], label='Observed')
 >>> plt.axhline(0, color='red', linestyle='--', label='Random')
 >>> plt.xlabel('Distance')
 >>> plt.ylabel('L(d)')
 >>> plt.legend()
 >>> plt.show()

 Notes
 -----
 Ripley'
 K(d) = (Area / n²) * Σ Σ I(dij < d)

 Where I(dij < d) is an indicator function that equals 1 if the distance
 between points i and j is less than d.

 For a random pattern: K(d) = π * d²

 The L transformation (L = sqrt(K/π) - d) is often used because:
 - For random pattern: L(d) = 0
 - For clustered pattern: L(d) > 0
 - For dispersed pattern: L(d) < 0

 References
 ----------
 Ripley, B.D. (1977). Modelling spatial patterns. Journal of the
 Royal Statistical Society, Series B, 39, 172-212.
 """
 x = np.asarray(x, dtype=np.float64)
 y = np.asarray(y, dtype=np.float64)
 n = len(x)

 if n < 3:

 # Determine study area
 if study_area is None:
 ymin, ymax = y.min(), y.max()
 else:
 else:

 area = (xmax - xmin) * (ymax - ymin)

 # Distance values
 if distances is None:
 distances = np.linspace(0, max_distance, n_distances + 1)[1:] # Exclude 0

 distances = np.asarray(distances)

 # Calculate pairwise distances
 points = np.column_stack([x, y])
 dist_matrix = distance_matrix(points, points)

 # Calculate K function
 K = np.zeros(len(distances))

 for i, d in enumerate(distances):
 count = np.sum(dist_matrix < d) - n # Subtract diagonal

 # Edge correction
 if edge_correction == 'none':
 elif edge_correction == 'border':
 elif edge_correction == 'border':
 buffer = d
 interior_mask = (
 (x > xmin + buffer) & (x < xmax - buffer) &
 (y > ymin + buffer) & (y < ymax - buffer)
 )
 n_interior = np.sum(interior_mask)
 weight = n / max(n_interior, 1)
 else:
 else:

 K[i] = (area * count * weight) / (n * (n - 1))

 # Theoretical K for random pattern
 K_theoretical = np.pi * distances**2

 # L transformation
 L = np.sqrt(K / np.pi) - distances

 # Interpretation
 # Check if observed K is consistently above or below theoretical
 above_threshold = np.sum(K > K_theoretical * 1.1) # 10% above
 below_threshold = np.sum(K < K_theoretical * 0.9) # 10% below

 if above_threshold > len(distances) * 0.5:
 elif below_threshold > len(distances) * 0.5:
 elif below_threshold > len(distances) * 0.5:
 else:
 else:

 return {
 'd': distances,
 'K': K,
 'L': L,
 'K_theoretical': K_theoretical,
 'interpretation': interpretation,
 'area': area,
 'n_points': n,
 }

def quadrat_analysis(
 y: npt.NDArray[np.float64],
 n_quadrats_x: int = 5,
 n_quadrats_y: int = 5,
 study_area: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
 """
 Perform quadrat analysis to test for spatial randomness.

 Divides study area into quadrats and tests if point counts follow
 a Poisson distribution (expected for random pattern).

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of points
 n_quadrats_x, n_quadrats_y : int
 Number of quadrats in x and y directions
 study_area : tuple of float, optional
 Study area bounds (xmin, xmax, ymin, ymax)

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'counts': Array of point counts per quadrat
 - 'mean': Mean count per quadrat
 - 'variance': Variance of counts
 - 'vmr': Variance-to-mean ratio
 - 'chi2_statistic': Chi-squared test statistic
 - 'chi2_p_value': P-value from chi-squared test
 - 'interpretation': Pattern interpretation

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.spatial_stats import quadrat_analysis
 >>>
 >>> # Random points
 >>> x = np.random.uniform(0, 100, 200)
 >>> y = np.random.uniform(0, 100, 200)
 >>>
 >>> results = quadrat_analysis(x, y, n_quadrats_x=10, n_quadrats_y=10)
 >>> logger.info(f"VMR: {results['vmr']:.3f}")
 >>> logger.info(f"Pattern: {results['interpretation']}")

 Notes
 -----
 Variance-to-Mean Ratio (VMR):
 - VMR < 1: Regular/dispersed pattern
 - VMR = 1: Random pattern (Poisson)
 - VMR > 1: Clustered pattern

 The chi-squared test compares observed frequency distribution
 to expected Poisson distribution.

 References
 ----------
 Greig-Smith, P. (1952). The use of random and contiguous quadrats
 in the study of the structure of plant communities. Annals of Botany,
 16, 293-316.
 """
 x = np.asarray(x, dtype=np.float64)
 y = np.asarray(y, dtype=np.float64)
 n = len(x)

 if n < 10:

 # Determine study area
 if study_area is None:
 ymin, ymax = y.min(), y.max()
 else:
 else:

 # Create quadrats
 x_edges = np.linspace(xmin, xmax, n_quadrats_x + 1)
 y_edges = np.linspace(ymin, ymax, n_quadrats_y + 1)

 # Count points in each quadrat
 counts = np.zeros((n_quadrats_y, n_quadrats_x))

 for i in range(n_quadrats_y):
 in_quadrat = (
 (x >= x_edges[j]) & (x < x_edges[j + 1]) &
 (y >= y_edges[i]) & (y < y_edges[i + 1])
 )
 counts[i, j] = np.sum(in_quadrat)

 # Handle edge case: points exactly on upper boundary
 if n_quadrats_x > 0 and n_quadrats_y > 0:
 on_top_edge = (y == ymax)
 for i in range(n):
 j_idx = n_quadrats_x - 1
 if on_top_edge[i]:
 else:
 else:
 counts[i_idx, j_idx] += 1
 elif on_top_edge[i]:
 elif on_top_edge[i]:
 j_idx = np.searchsorted(x_edges[:-1], x[i], side='right') - 1
 counts[i_idx, j_idx] += 1

 counts_flat = counts.flatten()

 # Statistics
 mean_count = np.mean(counts_flat)
 var_count = np.var(counts_flat, ddof=1)
 vmr = var_count / mean_count if mean_count > 0 else np.nan

 # Chi-squared test for Poisson distribution
 # Observed frequency distribution
 unique_counts = np.arange(0, int(np.max(counts_flat)) + 1)
 observed_freq = np.array([np.sum(counts_flat == c) for c in unique_counts])

 # Expected frequencies under Poisson
 expected_freq = len(counts_flat) * poisson.pmf(unique_counts, mean_count)

 # Combine categories with low expected frequencies
 min_expected = 5
 mask = expected_freq >= min_expected
 if np.sum(mask) < 2:
 chi2_stat = np.nan
 chi2_p = np.nan
 else:
 else:
 expected_combined = expected_freq[mask]

 # Chi-squared statistic
 chi2_stat = np.sum((observed_combined - expected_combined)**2 / expected_combined)

 # Degrees of freedom = number of categories - 1 - number of estimated parameters (1 for lambda)
 df = len(observed_combined) - 2
 if df > 0:
 else:
 else:

 # Interpretation
 if not np.isnan(vmr):
 pattern = "Regular/Dispersed"
 elif vmr > 1.1:
 elif vmr > 1.1:
 else:
 else:

 if not np.isnan(chi2_p):
 significance = "significant"
 else:
 else:
 interpretation = f"{pattern} ({significance}, p={chi2_p:.4f})"
 else:
 else:
 else:
 else:

 return {
 'counts': counts,
 'counts_flat': counts_flat,
 'mean': mean_count,
 'variance': var_count,
 'vmr': vmr,
 'chi2_statistic': chi2_stat if not np.isnan(chi2_stat) else None,
 'chi2_p_value': chi2_p if not np.isnan(chi2_p) else None,
 'n_quadrats': (n_quadrats_y, n_quadrats_x),
 'total_points': n,
 'interpretation': interpretation,
 }

def spatial_randomness_test(
 y: npt.NDArray[np.float64],
 method: str = 'all',
 **kwargs,
    ) -> Dict[str, Any]:
 """
 Test for spatial randomness using multiple methods.

 Combines nearest neighbor, Ripley'
 provide robust assessment of spatial pattern.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of points
 method : str, default='all'
 Which tests to perform: 'all', 'nearest_neighbor', 'ripley_k', 'quadrat'
 **kwargs
 Additional arguments passed to individual test functions

 Returns
 -------
 results : dict
 Dictionary containing results from all requested tests

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.spatial_stats import spatial_randomness_test
 >>>
 >>> # Generate points
 >>> x = np.random.uniform(0, 100, 150)
 >>> y = np.random.uniform(0, 100, 150)
 >>>
 >>> results = spatial_randomness_test(x, y)
 >>> logger.info("Nearest Neighbor:", results['nearest_neighbor']['interpretation'])
 >>> logger.info("Ripley's K:", results['ripley_k']['interpretation'
 >>> logger.info("Quadrat:", results['quadrat']['interpretation'])

 Notes
 -----
 Using multiple tests provides more robust conclusions:
 - Nearest neighbor: Good for overall clustering/dispersion
 - Ripley'
 - Quadrat: Tests departure from Poisson distribution

 Agreement among tests strengthens conclusions.
 """
 results = {}

 if method in ['all', 'nearest_neighbor']:
 results['nearest_neighbor'] = nearest_neighbor_analysis(x, y, **kwargs)
 except Exception as e:
 logger.error(f"Nearest neighbor analysis failed: {e}")
 results['nearest_neighbor'] = {'error': str(e)}

 if method in ['all', 'ripley_k']:
 results['ripley_k'] = ripley_k_function(x, y, **kwargs)
 except Exception as e:
 logger.error(f"Ripley's K function failed: {e}"
 results['ripley_k'] = {'error': str(e)}

 if method in ['all', 'quadrat']:
 results['quadrat'] = quadrat_analysis(x, y, **kwargs)
 except Exception as e:
 logger.error(f"Quadrat analysis failed: {e}")
 results['quadrat'] = {'error': str(e)}

 return results

def clustering_index(
 y: npt.NDArray[np.float64],
 method: str = 'nearest_neighbor',
    ) -> float:
 """
 Calculate a single clustering index value.

 Provides a simple numeric measure of clustering/dispersion.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of points
 method : str, default='nearest_neighbor'
 Method to use: 'nearest_neighbor', 'vmr'

 Returns
 -------
 index : float
 Clustering index value
 - < 1: Clustered
 - = 1: Random
 - > 1: Dispersed

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.spatial_stats import clustering_index
 >>>
 >>> x = np.random.uniform(0, 100, 100)
 >>> y = np.random.uniform(0, 100, 100)
 >>>
 >>> index = clustering_index(x, y)
 >>> logger.info(f"Clustering index: {index:.3f}")
 """
 if method == 'nearest_neighbor':
 return results['R']
 elif method == 'vmr':
 elif method == 'vmr':
 return results['vmr']
 else:
 else:
