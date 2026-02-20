"""
    Data utility functions
"""

from typing import Tuple, Optional, Dict
import numpy as np
import numpy.typing as npt

def generate_synthetic_data(
 spatial_structure: str = "spherical",
 nugget: float = 0.1,
 sill: float = 1.0,
 range_param: float = 20.0,
 seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pass
 """
     Generate synthetic spatial data with known variogram structure

 Parameters
 ----------
 n_points : int
 Number of sample points
 spatial_structure : str
 Type of spatial correlation structure
 nugget : float
 Nugget effect
 sill : float
 Sill variance
 range_param : float
 Range parameter
 seed : int, optional
 Random seed for reproducibility

 Returns
 -------
 x, y, z : np.ndarray
 Coordinates and values
 """
 if seed is not None:
    pass

 # Generate random locations in [0, 100] x [0, 100]
 x = np.random.uniform(0, 100, n_points)
 y = np.random.uniform(0, 100, n_points)

 # Generate spatial field with trend
 trend = 0.05 * x + 0.02 * y

 # Add spatial correlation (simplified)
 # In production, this would use actual spatial simulation
 z = trend + np.random.randn(n_points) * np.sqrt(sill)

 return x, y, z

def load_sample_data(dataset: str = "walker_lake") -> Dict:
 Load sample geostatistical dataset

 Parameters
 ----------
 dataset : str
 Name of dataset to load
 Options: 'walker_lake', 'walker_lake_v', 'walker_lake_u'

 Returns
 -------
 dict
 Dictionary with keys: 'x', 'y', 'z', 'description'
 """
 if dataset == "walker_lake" or dataset == "walker_lake_v":
 from ..datasets import load_walker_lake
 data = load_walker_lake()
 return {
 'x': data['x'],
 'y': data['y'],
 'z': data['V'],
 'description': 'Walker Lake V (arsenious contaminant)',
 }

 elif dataset == "walker_lake_u":
 from ..datasets import load_walker_lake
 data = load_walker_lake()
 return {
 'x': data['x'],
 'y': data['y'],
 'z': data['U'],
 'description': 'Walker Lake U (PCE concentration)',
 }

 else:
    pass

def split_train_test(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 test_fraction: float = 0.2,
 test_size: Optional[float] = None,
 random_state: Optional[int] = None,
 seed: Optional[int] = None,
    ) -> Tuple:
        pass
 """
     Split data into training and test sets

 Parameters
 ----------
 x, y, z : np.ndarray
 Coordinates and values
 test_fraction : float
 Fraction of data to use for testing (default: 0.2)
 test_size : float, optional
 Alias for test_fraction (for sklearn compatibility)
 random_state : int, optional
 Random seed (sklearn-style)
 seed : int, optional
 Random seed (alternative name)

 Returns
 -------
 x_train, y_train, z_train, x_test, y_test, z_test : np.ndarray
 Split data
 """
 # Handle multiple parameter names
 if test_size is not None:
    pass

 random_seed = random_state if random_state is not None else seed

 if random_seed is not None:
    pass

 n = len(x)
 n_test = int(n * test_fraction)

 indices = np.arange(n)
 np.random.shuffle(indices)

 test_idx = indices[:n_test]
 train_idx = indices[n_test:]

 return (
 x[train_idx], y[train_idx], z[train_idx],
 x[test_idx], y[test_idx], z[test_idx],
 )

    # Additional utility functions
def find_duplicate_locations(
 y: npt.NDArray[np.float64],
 tolerance: float = 1e-6,
    ) -> list:
        pass
 """
     Find duplicate spatial locations

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 tolerance : float
 Distance tolerance for considering points duplicates

 Returns
 -------
 list
 List of tuples (i, j) indicating duplicate indices
 """
 duplicates = []
 n = len(x)

 for i in range(n):
     continue
 dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
 if dist < tolerance:
    pass

 return duplicates

def check_collinearity(
 y: npt.NDArray[np.float64],
 tolerance: float = 0.01,
    ) -> bool:
        pass
 """
     Check if points are approximately collinear

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 tolerance : float
 Tolerance for collinearity check

 Returns
 -------
 bool
 True if points are collinear
 """
 if len(x) < 3:
    pass

 # Center the data
 x_centered = x - np.mean(x)
 y_centered = y - np.mean(y)

 # Compute covariance matrix
 cov = np.cov(x_centered, y_centered)

 # Check if determinant is near zero
 det = np.linalg.det(cov)

 return abs(det) < tolerance

def compute_data_spacing(
 y: npt.NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        pass
 """
     Compute statistics about data spacing

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates

 Returns
 -------
 min_spacing : float
 Minimum distance between any two points
 mean_spacing : float
 Mean distance to nearest neighbor
 max_spacing : float
 Maximum distance between any two points
 """
 n = len(x)

 # Compute all pairwise distances
 from scipy.spatial.distance import pdist, squareform

 coords = np.column_stack([x, y])
 distances = squareform(pdist(coords))

 # Set diagonal to inf to ignore self-distances
 np.fill_diagonal(distances, np.inf)

 # Minimum distance
 min_spacing = np.min(distances)

 # Mean nearest neighbor distance
 nearest_neighbor_dists = np.min(distances, axis=1)
 mean_spacing = np.mean(nearest_neighbor_dists)

 # Maximum distance
 max_spacing = np.max(distances[distances < np.inf])

 return min_spacing, mean_spacing, max_spacing
