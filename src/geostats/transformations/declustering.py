"""
Declustering Methods

Implementation based on Olea (2009) §1996-2066:
"Declustering is important for the inference of global parameters, such as any
of those associated with the histogram."

Handles preferential sampling where data points are clustered in certain areas,
leading to biased global statistics.

Reference:
- ofr20091103.txt (USGS Practical Primer)
- Declustering (pages 91-97)
- Cell declustering and distance-based methods
"""

from typing import Tuple, Optional, Dict
import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix, KDTree
import logging

logger = logging.getLogger(__name__)

def cell_declustering(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 cell_sizes: Optional[npt.NDArray[np.float64]] = None,
 n_sizes: int = 10
) -> Tuple[npt.NDArray[np.float64], Dict]:
 """
 Cell Declustering

 Assigns weights to samples based on spatial distribution to correct for
 preferential clustering. Samples in densely sampled areas get lower weights.

 Method:
 1. Overlay grid of cells with varying cell sizes
 2. Count samples per cell
 3. Assign weight = 1/n_samples_in_cell to each sample
 4. Choose cell size that minimizes variance of weights

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of sample points
 z : np.ndarray
 Values at sample points
 cell_sizes : np.ndarray, optional
 Array of cell sizes to try
 If None, automatically generates sizes
 n_sizes : int
 Number of cell sizes to try (if cell_sizes not provided)

 Returns
 -------
 weights : np.ndarray
 Declustering weights (sum to n_samples)
 info : dict
 Information about declustering including:
 - optimal_cell_size
 - weights_by_cell_size
 - statistics (weighted vs unweighted)

 Examples
 --------
 >>> import numpy as np
 >>> # Clustered data (more samples in one area)
 >>> x = np.array([1, 1.1, 1.2, 5, 5.1, 10])
 >>> y = np.array([1, 1.1, 1.2, 5, 5.1, 10])
 >>> z = np.array([10, 11, 12, 50, 51, 100])
 >>> weights, info = cell_declustering(x, y, z)
 >>> weighted_mean = np.average(z, weights=weights)
 >>> unweighted_mean = np.mean(z)
 >>> logger.info(f"Unweighted: {unweighted_mean:.2f}, Weighted: {weighted_mean:.2f}")

 Notes
 -----
 From Olea (2009): "Declustering is important for the inference of global
 parameters" when data has preferential clustering.
 """
 x = np.asarray(x, dtype=np.float64)
 y = np.asarray(y, dtype=np.float64)
 z = np.asarray(z, dtype=np.float64)

 if len(x) != len(y) or len(x) != len(z):
 raise ValueError("x, y, z must have same length")

 n = len(x)

 # Define spatial extent
 x_min, x_max = np.min(x), np.max(x)
 y_min, y_max = np.min(y), np.max(y)
 x_range = x_max - x_min
 y_range = y_max - y_min
 max_range = max(x_range, y_range)

 # Generate cell sizes to try if not provided
 if cell_sizes is None:
 # Try sizes from 1/20 to 2x the data range
 min_size = max_range / 20
 max_size = max_range * 2
 cell_sizes = np.linspace(min_size, max_size, n_sizes)

 # Try each cell size
 weights_all = []
 variances = []

 for cell_size in cell_sizes:
 # Compute cell indices for each sample
 ix = np.floor((x - x_min) / cell_size).astype(int)
 iy = np.floor((y - y_min) / cell_size).astype(int)

 # Create cell IDs
 cell_ids = ix * 10000 + iy # Simple hash

 # Count samples per cell
 unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)

 # Create mapping from cell_id to count
 cell_count_map = dict(zip(unique_cells, cell_counts))

 # Assign weights (1 / samples_in_cell)
 weights = np.array([1.0 / cell_count_map[cell_id] for cell_id in cell_ids])

 # Normalize weights to sum to n (conventional)
 weights = weights * n / np.sum(weights)

 weights_all.append(weights)
 variances.append(np.var(weights))

 # Choose cell size with minimum variance
 optimal_idx = np.argmin(variances)
 optimal_weights = weights_all[optimal_idx]
 optimal_cell_size = cell_sizes[optimal_idx]

 # Calculate statistics
 unweighted_mean = np.mean(z)
 unweighted_var = np.var(z)
 weighted_mean = np.average(z, weights=optimal_weights)
 weighted_var = np.average((z - weighted_mean)**2, weights=optimal_weights)

 info = {
 'optimal_cell_size': optimal_cell_size,
 'cell_sizes_tried': cell_sizes,
 'variances': np.array(variances),
 'weights_by_cell_size': weights_all,
 'unweighted_mean': unweighted_mean,
 'unweighted_var': unweighted_var,
 'weighted_mean': weighted_mean,
 'weighted_var': weighted_var,
 'mean_difference': weighted_mean - unweighted_mean,
 }

 return optimal_weights, info

def polygonal_declustering(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 power: float = 2.0
) -> Tuple[npt.NDArray[np.float64], Dict]:
 """
 Polygonal (Voronoi-based) Declustering

 Assigns weights based on the area of influence (Voronoi polygon) around
 each sample. Samples in dense areas have smaller polygons and lower weights.

 This is an approximate method using distance to nearest neighbors as a proxy
 for polygon area.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of sample points
 z : np.ndarray
 Values at sample points
 power : float
 Power to raise distances to (default 2 for area approximation)

 Returns
 -------
 weights : np.ndarray
 Declustering weights (normalized to sum to n_samples)
 info : dict
 Information including statistics

 Notes
 -----
 This method approximates Voronoi polygon areas using the squared distance
 to the nearest neighbor, which is computationally simpler than computing
 actual Voronoi diagrams.
 """
 x = np.asarray(x, dtype=np.float64)
 y = np.asarray(y, dtype=np.float64)
 z = np.asarray(z, dtype=np.float64)

 if len(x) != len(y) or len(x) != len(z):
 raise ValueError("x, y, z must have same length")

 n = len(x)

 if n < 2:
 # Single point, weight = n
 return np.array([float(n)]), {'method': 'single_point'}

 # Build KD-tree for efficient nearest neighbor search
 points = np.column_stack([x, y])
 tree = KDTree(points)

 # Find distance to nearest neighbor for each point
 distances, indices = tree.query(points, k=2) # k=2 includes self
 nearest_distances = distances[:, 1] # Second closest is nearest neighbor

 # Weight proportional to distance^power (proxy for area)
 weights = nearest_distances ** power

 # Normalize to sum to n
 weights = weights * n / np.sum(weights)

 # Calculate statistics
 unweighted_mean = np.mean(z)
 weighted_mean = np.average(z, weights=weights)

 info = {
 'method': 'polygonal',
 'power': power,
 'nearest_distances': nearest_distances,
 'unweighted_mean': unweighted_mean,
 'weighted_mean': weighted_mean,
 'mean_difference': weighted_mean - unweighted_mean,
 }

 return weights, info

def detect_clustering(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64]
) -> Dict[str, float]:
 """
 Detect presence of spatial clustering

 Uses nearest neighbor distances to assess if data is clustered.
 From Olea (2009) §2009: "Detect presence of clusters by preparing a
 cumulative distribution of distance to nearest neighbor."

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of sample points

 Returns
 -------
 dict
 Statistics about clustering:
 - mean_nn_dist: Mean nearest neighbor distance
 - std_nn_dist: Standard deviation of nearest neighbor distances
 - cv_nn_dist: Coefficient of variation (std/mean)
 - clustering_index: Ratio of std to mean (>0.5 suggests clustering)
 """
 points = np.column_stack([x, y])
 tree = KDTree(points)

 # Find nearest neighbor distances
 distances, _ = tree.query(points, k=2)
 nn_distances = distances[:, 1]

 mean_dist = np.mean(nn_distances)
 std_dist = np.std(nn_distances)
 cv = std_dist / mean_dist if mean_dist > 0 else 0

 return {
 'mean_nn_dist': mean_dist,
 'std_nn_dist': std_dist,
 'cv_nn_dist': cv,
 'clustering_index': cv,
 'is_likely_clustered': cv > 0.5, # Rule of thumb
 }
