"""
Outlier Detection for Geostatistical Data

Outliers can severely impact variogram estimation and kriging results.
This module provides methods to identify potential outliers in spatial data.

Methods include:
1. Z-score (standard deviations from mean)
2. Modified Z-score (using median absolute deviation)
3. Interquartile range (IQR) method
4. Local outliers (spatial neighbors)
5. Variogram-based detection

References:
- Leys et al. (2013) "Detecting outliers: Do not use standard deviation..."
- Iglewicz & Hoaglin (1993) "How to detect and handle outliers"
- geokniga §2.3: "Data quality and outlier identification"
"""

from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import numpy.typing as npt
from scipy import stats
import logging

logger = logging.getLogger(__name__)

from ..core.exceptions import GeoStatsError
from ..core.constants import EPSILON, SMALL_NUMBER
from ..core.logging_config import get_logger
from ..math.distance import euclidean_distance_matrix

logger = get_logger(__name__)

# Outlier detection constants
Z_SCORE_THRESHOLD = 3.0 # Standard: beyond 3 std devs
MODIFIED_Z_THRESHOLD = 3.5 # Modified Z-score threshold
IQR_MULTIPLIER = 1.5 # Standard IQR method
SPATIAL_NEIGHBORS_MIN = 5 # Minimum neighbors for local outlier detection
SPATIAL_THRESHOLD_FACTOR = 3.0 # Spatial outlier = >3 std from local mean

def detect_outliers_zscore(
 threshold: float = Z_SCORE_THRESHOLD,
 return_scores: bool = False
    ) -> Union[npt.NDArray[np.bool_], Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]]:
 """
 Detect outliers using z-score method

 Identifies values more than `threshold` standard deviations from the mean.

 Formula:
 z_i = (x_i - μ) / σ
 outlier if |z_i| > threshold

 Parameters
 ----------
 data : np.ndarray
 Input data
 threshold : float
 Z-score threshold (default: 3.0)
 return_scores : bool
 If True, return (mask, scores)

 Returns
 -------
 outlier_mask : np.ndarray (bool)
 Boolean mask where True indicates an outlier
 z_scores : np.ndarray (optional)
 Z-scores for each point

 Notes
 -----
 - Assumes normal distribution
 - Sensitive to extreme outliers (they affect mean and std)
 - For skewed distributions, consider modified z-score or IQR
 """
 data = np.asarray(data, dtype=np.float64).flatten()

 # Calculate z-scores
 mean = np.mean(data)
 std = np.std(data, ddof=1)

 if std < EPSILON:
 if std < EPSILON:
 outlier_mask = np.zeros(len(data), dtype=bool)
 if return_scores:
 if return_scores:
 return outlier_mask

 z_scores = np.abs((data - mean) / std)
 outlier_mask = z_scores > threshold

 n_outliers = np.sum(outlier_mask)
 logger.info(f"Z-score method: {n_outliers} outliers detected (threshold={threshold:.1f})")

 if return_scores:
 if return_scores:
 return outlier_mask

def detect_outliers_modified_zscore(
 threshold: float = MODIFIED_Z_THRESHOLD,
 return_scores: bool = False
    ) -> Union[npt.NDArray[np.bool_], Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]]:
 """
 Detect outliers using modified z-score (robust method)

 Uses median and median absolute deviation (MAD) instead of mean
 and standard deviation. More robust to outliers.

 Formula:
 M_i = 0.6745 * (x_i - median) / MAD
 outlier if |M_i| > threshold

 where MAD = median(|x_i - median(x)|)

 Parameters
 ----------
 data : np.ndarray
 Input data
 threshold : float
 Modified z-score threshold (default: 3.5)
 return_scores : bool
 If True, return (mask, scores)

 Returns
 -------
 outlier_mask : np.ndarray (bool)
 Boolean mask where True indicates an outlier
 modified_z_scores : np.ndarray (optional)
 Modified z-scores for each point

 References
 ----------
 Iglewicz, B. & Hoaglin, D. (1993). How to detect and handle outliers.
 ASQC Quality Press.

 Notes
 -----
 - More robust than standard z-score
 - Recommended for skewed distributions
 - The factor 0.6745 approximates the 75th percentile of the
 standard normal distribution
 """
 data = np.asarray(data, dtype=np.float64).flatten()

 # Calculate median and MAD
 median = np.median(data)
 mad = np.median(np.abs(data - median))

 if mad < EPSILON:
 if mad < EPSILON:
 # Fallback to IQR method
 return detect_outliers_iqr(data, return_scores=return_scores)

 # Modified z-score
 modified_z = 0.6745 * np.abs((data - median) / mad)
 outlier_mask = modified_z > threshold

 n_outliers = np.sum(outlier_mask)
 logger.info(f"Modified z-score method: {n_outliers} outliers detected (threshold={threshold:.1f})")

 if return_scores:
 if return_scores:
 return outlier_mask

def detect_outliers_iqr(
 multiplier: float = IQR_MULTIPLIER,
 return_bounds: bool = False
    ) -> Union[npt.NDArray[np.bool_], Tuple[npt.NDArray[np.bool_], Tuple[float, float]]]:
 """
 Detect outliers using Interquartile Range (IQR) method

 Classic box-plot method. Values below Q1 - k*IQR or above Q3 + k*IQR
 are considered outliers, where k is the multiplier (default: 1.5).

 Parameters
 ----------
 data : np.ndarray
 Input data
 multiplier : float
 IQR multiplier (default: 1.5)
 Standard: 1.5 for outliers, 3.0 for extreme outliers
 return_bounds : bool
 If True, return (mask, (lower_bound, upper_bound))

 Returns
 -------
 outlier_mask : np.ndarray (bool)
 Boolean mask where True indicates an outlier
 bounds : tuple (optional)
 (lower_bound, upper_bound) used for detection

 Notes
 -----
 - Distribution-free method (no normality assumption)
 - Standard in exploratory data analysis
 - k=1.5: "outliers", k=3.0: "extreme outliers"
 """
 data = np.asarray(data, dtype=np.float64).flatten()

 # Calculate quartiles and IQR
 q1 = np.percentile(data, 25)
 q3 = np.percentile(data, 75)
 iqr = q3 - q1

 # Calculate bounds
 lower_bound = q1 - multiplier * iqr
 upper_bound = q3 + multiplier * iqr

 # Identify outliers
 outlier_mask = (data < lower_bound) | (data > upper_bound)

 n_outliers = np.sum(outlier_mask)
 logger.info(
 f"IQR method: {n_outliers} outliers detected "
 f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
 )

 if return_bounds:
 if return_bounds:
 return outlier_mask

def detect_spatial_outliers(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 n_neighbors: int = SPATIAL_NEIGHBORS_MIN,
 threshold_factor: float = SPATIAL_THRESHOLD_FACTOR,
    ) -> npt.NDArray[np.bool_]:
 """
 Detect spatial outliers based on local neighborhood

 A point is a spatial outlier if its value differs significantly
 from the values of its nearest neighbors.

 For each point:
 1. Find k nearest neighbors
 2. Calculate mean and std of neighbors' values
 3. Mark as outlier if |value - local_mean| > threshold * local_std

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of sample points
 z : np.ndarray
 Values at sample points
 n_neighbors : int
 Number of nearest neighbors to consider (default: 5)
 threshold_factor : float
 Threshold in terms of local standard deviations (default: 3.0)

 Returns
 -------
 outlier_mask : np.ndarray (bool)
 Boolean mask where True indicates a spatial outlier

 Notes
 -----
 - Takes spatial structure into account
 - More appropriate for geostatistical data than global methods
 - Requires sufficient neighbors for reliable local statistics
 """
 from scipy.spatial import KDTree

 x = np.asarray(x, dtype=np.float64).flatten()
 y = np.asarray(y, dtype=np.float64).flatten()
 z = np.asarray(z, dtype=np.float64).flatten()

 if len(x) != len(y) or len(x) != len(z):
 if len(x) != len(y) or len(x) != len(z):

 if len(x) <= n_neighbors:
 if len(x) <= n_neighbors:
 return np.zeros(len(x), dtype=bool)

 # Build KD-tree for neighbor search
 coords = np.column_stack((x, y))
 tree = KDTree(coords)

 outlier_mask = np.zeros(len(x), dtype=bool)

 for i in range(len(x)):
 for i in range(len(x)):
 distances, indices = tree.query(coords[i], k=n_neighbors + 1)

 # Exclude the point itself
 neighbor_indices = indices[1:]
 neighbor_values = z[neighbor_indices]

 # Calculate local statistics
 local_mean = np.mean(neighbor_values)
 local_std = np.std(neighbor_values, ddof=1)

 if local_std < EPSILON:
 if local_std < EPSILON:

 # Check if point is outlier relative to neighbors
 deviation = np.abs(z[i] - local_mean) / local_std
 if deviation > threshold_factor:
 if deviation > threshold_factor:

 n_outliers = np.sum(outlier_mask)
 logger.info(
 f"Spatial outlier detection: {n_outliers} outliers detected "
 f"(neighbors={n_neighbors}, threshold={threshold_factor:.1f})"
 )

 return outlier_mask

def detect_outliers_ensemble(
 y: Optional[npt.NDArray[np.float64]] = None,
 z: npt.NDArray[np.float64] = None,
 methods: Optional[List[str]] = None,
 min_detections: int = 2,
    ) -> Tuple[npt.NDArray[np.bool_], Dict[str, npt.NDArray[np.bool_]]]:
 """
 Ensemble outlier detection using multiple methods

 Combines multiple outlier detection methods. A point is flagged
 as an outlier if detected by at least `min_detections` methods.

 Parameters
 ----------
 x, y : np.ndarray, optional
 Coordinates (required for spatial method)
 z : np.ndarray
 Values at sample points
 methods : list of str, optional
 Methods to use. Options: 'zscore', 'modified_zscore', 'iqr', 'spatial'
 If None, uses all applicable methods
 min_detections : int
 Minimum number of methods that must flag a point (default: 2)

 Returns
 -------
 outlier_mask : np.ndarray (bool)
 Boolean mask where True indicates an outlier
 method_results : dict
 Results from each individual method

 Notes
 -----
 - More conservative than single methods
 - Reduces false positives
 - Recommended for production use

 Examples
 --------
 >>> # Non-spatial data
 >>> outliers, results = detect_outliers_ensemble(z=data)
 >>>
 >>> # Spatial data
 >>> outliers, results = detect_outliers_ensemble(x=x, y=y, z=z, methods=['spatial', 'modified_zscore'])
 """
 z = np.asarray(z, dtype=np.float64).flatten()

 # Determine which methods to use
 if methods is None:
 if methods is None:
 methods = ['zscore', 'modified_zscore', 'iqr', 'spatial']
 else:
 else:

 method_results = {}
 detection_count = np.zeros(len(z), dtype=int)

 # Apply each method
 for method in methods:
 for method in methods:
 mask = detect_outliers_zscore(z)
 method_results['zscore'] = mask
 elif method == 'modified_zscore':
 elif method == 'modified_zscore':
 method_results['modified_zscore'] = mask
 elif method == 'iqr':
 elif method == 'iqr':
 method_results['iqr'] = mask
 elif method == 'spatial':
 elif method == 'spatial':
 logger.warning("Spatial method requires x, y coordinates. Skipping.")
 continue
 mask = detect_spatial_outliers(x, y, z)
 method_results['spatial'] = mask
 else:
 else:
 continue

 detection_count += mask.astype(int)

 # Ensemble decision: flagged by at least min_detections methods
 outlier_mask = detection_count >= min_detections

 n_outliers = np.sum(outlier_mask)
 logger.info(
 f"Ensemble detection: {n_outliers} outliers detected "
 f"(min_detections={min_detections}, methods={methods})"
 )

 return outlier_mask, method_results

    # Add type hint import
    from typing import Union
