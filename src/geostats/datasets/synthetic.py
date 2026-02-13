"""
Synthetic data generators for testing and demonstrations.

Provides various synthetic spatial datasets with known properties
for testing interpolation methods and demonstrating concepts.

Reference: Python Recipes for Earth Sciences (Trauth 2024)
"""

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt

from ..core.logging_config import get_logger
import logging

logger = logging.getLogger(__name__)

logger = get_logger(__name__)

# Constants
DEFAULT_N_POINTS = 100
DEFAULT_SPATIAL_RANGE = 100.0
DEFAULT_NOISE_LEVEL = 0.1
MIN_POINTS = 10
MAX_POINTS = 100000

def generate_random_field(
 x_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 y_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 trend_type: str = 'linear',
 noise_level: float = DEFAULT_NOISE_LEVEL,
 seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Generate a synthetic random field with specified trend and noise.

 Parameters
 ----------
 n_points : int, default=100
 Number of sample points
 x_range, y_range : tuple of float
 Spatial extent (min, max)
 trend_type : str, default='linear'
 Type of spatial trend:
 - 'linear': z = a*x + b*y
 - 'quadratic': z = a*x² + b*y²
 - 'saddle': z = x² - y²
 - 'wave': z = sin(x) * cos(y)
 - 'none': No trend (pure noise)
 noise_level : float, default=0.1
 Standard deviation of Gaussian noise
 seed : int, optional
 Random seed for reproducibility

 Returns
 -------
 x, y, z : np.ndarray
 Coordinates and values

 Examples
 --------
 >>> from geostats.datasets.synthetic import generate_random_field
 >>> x, y, z = generate_random_field(
 ... n_points=50,
 ... trend_type='linear',
 ... noise_level=0.2,
 ... seed=42
 ... )
 >>> logger.info(f"Generated {len(x)} points")
 Generated 50 points

 Notes
 -----
 Useful for:
 - Testing interpolation methods with known ground truth
 - Demonstrating spatial interpolation concepts
 - Comparing method performance under different conditions
 """
 if not MIN_POINTS <= n_points <= MAX_POINTS:

 if seed is not None:

 # Generate random coordinates
 x = np.random.uniform(x_range[0], x_range[1], n_points)
 y = np.random.uniform(y_range[0], y_range[1], n_points)

 # Normalize coordinates to [0, 1] for trend calculation
 x_norm = (x - x_range[0]) / (x_range[1] - x_range[0])
 y_norm = (y - y_range[0]) / (y_range[1] - y_range[0])

 # Generate trend using dispatch pattern
 trend_functions = {
 'linear': lambda xn, yn: 2.0 * xn + 3.0 * yn,
 'quadratic': lambda xn, yn: xn**2 + 2.0 * yn**2,
 'saddle': lambda xn, yn: (xn - 0.5)**2 - (yn - 0.5)**2,
 'wave': lambda xn, yn: np.sin(4 * np.pi * xn) * np.cos(4 * np.pi * yn),
 'none': lambda xn, yn: np.zeros(len(xn)),
 }

 if trend_type not in trend_functions:
 raise ValueError(
 f"Unknown trend_type '{trend_type}'. "
 f"Valid types: {valid_types}"
 )

 z = trend_functions[trend_type](x_norm, y_norm)

 # Add noise
 if noise_level > 0:
 z = z + noise

 return x, y, z

def generate_clustered_samples(
 points_per_cluster: int = 20,
 cluster_std: float = 5.0,
 x_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 y_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 value_by_cluster: bool = True,
 seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Generate clustered spatial samples.

 Useful for testing declustering methods and demonstrating
 the effect of preferential sampling.

 Parameters
 ----------
 n_clusters : int, default=3
 Number of spatial clusters
 points_per_cluster : int, default=20
 Number of points in each cluster
 cluster_std : float, default=5.0
 Standard deviation of clusters (spatial spread)
 x_range, y_range : tuple of float
 Spatial extent
 value_by_cluster : bool, default=True
 If True, values vary by cluster
 If False, values are spatially continuous
 seed : int, optional
 Random seed

 Returns
 -------
 x, y, z : np.ndarray
 Coordinates and values

 Examples
 --------
 >>> from geostats.datasets.synthetic import generate_clustered_samples
 >>> x, y, z = generate_clustered_samples(
 ... n_clusters=5,
 ... points_per_cluster=30,
 ... seed=42
 ... )

 Notes
 -----
 This dataset type is useful for demonstrating:
 - Declustering techniques
 - Effect of preferential sampling on global statistics
 - Spatial bias in sampling
 """
 if seed is not None:

 n_total = n_clusters * points_per_cluster
 x = np.zeros(n_total)
 y = np.zeros(n_total)
 z = np.zeros(n_total)

 # Generate cluster centers
 cluster_x = np.random.uniform(x_range[0], x_range[1], n_clusters)
 cluster_y = np.random.uniform(y_range[0], y_range[1], n_clusters)
 cluster_values = np.random.uniform(0, 10, n_clusters)

 # Generate points around each cluster
 for i in range(n_clusters):
 end_idx = (i + 1) * points_per_cluster

 # Cluster coordinates
 x[start_idx:end_idx] = cluster_x[i] + np.random.normal(0, cluster_std, points_per_cluster)
 y[start_idx:end_idx] = cluster_y[i] + np.random.normal(0, cluster_std, points_per_cluster)

 # Clip to bounds
 x[start_idx:end_idx] = np.clip(x[start_idx:end_idx], x_range[0], x_range[1])
 y[start_idx:end_idx] = np.clip(y[start_idx:end_idx], y_range[0], y_range[1])

 # Values
 if value_by_cluster:
 z[start_idx:end_idx] = cluster_values[i] + np.random.normal(0, 0.5, points_per_cluster)
 else:
 else:
 x_norm = (x[start_idx:end_idx] - x_range[0]) / (x_range[1] - x_range[0])
 y_norm = (y[start_idx:end_idx] - y_range[0]) / (y_range[1] - y_range[0])
 z[start_idx:end_idx] = x_norm + 2*y_norm + np.random.normal(0, 0.3, points_per_cluster)

 return x, y, z

def generate_elevation_like_data(
 x_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 y_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 n_hills: int = 3,
 roughness: float = 0.1,
 seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Generate synthetic elevation-like data with hills and valleys.

 Mimics digital elevation models (DEMs) for testing interpolation
 of topographic data.

 Parameters
 ----------
 n_points : int, default=100
 Number of sample points
 x_range, y_range : tuple of float
 Spatial extent
 n_hills : int, default=3
 Number of hill/valley features
 roughness : float, default=0.1
 Terrain roughness (higher = more noise)
 seed : int, optional
 Random seed

 Returns
 -------
 x, y, z : np.ndarray
 Coordinates and elevation values

 Examples
 --------
 >>> from geostats.datasets.synthetic import generate_elevation_like_data
 >>> x, y, z = generate_elevation_like_data(
 ... n_points=200,
 ... n_hills=5,
 ... roughness=0.2,
 ... seed=42
 ... )

 Notes
 -----
 This generates data similar to a DEM with:
 - Multiple peaks/hills
 - Smooth transitions
 - Realistic terrain-like features
 - Optional roughness/noise

 Useful for demonstrating kriging of elevation data as discussed
 in Python Recipes for Earth Sciences (Trauth 2024), Chapter 7.
 """
 if seed is not None:

 # Generate sample points
 x = np.random.uniform(x_range[0], x_range[1], n_points)
 y = np.random.uniform(y_range[0], y_range[1], n_points)

 # Generate hill centers and heights
 hill_x = np.random.uniform(x_range[0], x_range[1], n_hills)
 hill_y = np.random.uniform(y_range[0], y_range[1], n_hills)
 hill_heights = np.random.uniform(50, 200, n_hills)
 hill_widths = np.random.uniform(20, 40, n_hills)

 # Calculate elevation as sum of Gaussian hills
 z = np.zeros(n_points)

 for i in range(n_hills):
 dy = y - hill_y[i]
 dist_sq = dx**2 + dy**2
 z += hill_heights[i] * np.exp(-dist_sq / (2 * hill_widths[i]**2))

 # Add baseline elevation
 base_elevation = 100.0
 z += base_elevation

 # Add roughness
 if roughness > 0:

 return x, y, z

def generate_anisotropic_field(
 x_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 y_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 anisotropy_ratio: float = 3.0,
 anisotropy_angle: float = 45.0,
 correlation_length: float = 20.0,
 seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Generate synthetic data with anisotropic spatial correlation.

 Useful for testing anisotropic kriging and demonstrating
 directional correlation structures.

 Parameters
 ----------
 n_points : int, default=100
 Number of sample points
 x_range, y_range : tuple of float
 Spatial extent
 anisotropy_ratio : float, default=3.0
 Ratio of correlation lengths (major/minor axis)
 anisotropy_angle : float, default=45.0
 Angle of anisotropy in degrees
 correlation_length : float, default=20.0
 Base correlation length
 seed : int, optional
 Random seed

 Returns
 -------
 x, y, z : np.ndarray
 Coordinates and values with anisotropic correlation

 Examples
 --------
 >>> from geostats.datasets.synthetic import generate_anisotropic_field
 >>> x, y, z = generate_anisotropic_field(
 ... n_points=150,
 ... anisotropy_ratio=4.0,
 ... anisotropy_angle=30.0,
 ... seed=42
 ... )

 Notes
 -----
 Anisotropy is common in earth sciences:
 - Geological layers (horizontal > vertical correlation)
 - River valleys (along-valley > across-valley)
 - Glacial deposits (flow direction dependent)

 This dataset helps demonstrate the importance of accounting for
 directional effects in spatial interpolation.
 """
 if seed is not None:

 # Generate sample points
 x = np.random.uniform(x_range[0], x_range[1], n_points)
 y = np.random.uniform(y_range[0], y_range[1], n_points)

 # Convert angle to radians
 angle_rad = np.deg2rad(anisotropy_angle)

 # Rotation matrix
 cos_theta = np.cos(angle_rad)
 sin_theta = np.sin(angle_rad)

 # Generate values with anisotropic correlation
 # Use sum of sinusoids with different wavelengths along major/minor axes
 z = np.zeros(n_points)

 n_waves = 5
 for _ in range(n_waves):
 freq_major = 2 * np.pi / (correlation_length * anisotropy_ratio)
 freq_minor = 2 * np.pi / correlation_length
 phase = np.random.uniform(0, 2 * np.pi)
 amplitude = np.random.uniform(0.5, 1.5)

 # Rotate coordinates
 x_rot = cos_theta * x + sin_theta * y
 y_rot = -sin_theta * x + cos_theta * y

 # Add wave component
 z += amplitude * np.sin(freq_major * x_rot + freq_minor * y_rot + phase)

 # Normalize and add noise
 z = z / np.std(z)
 z += np.random.normal(0, 0.2, n_points)

 return x, y, z

def generate_sparse_dense_mix(
 n_dense: int = 100,
 dense_region_center: Tuple[float, float] = (50.0, 50.0),
 dense_region_radius: float = 20.0,
 x_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 y_range: Tuple[float, float] = (0.0, DEFAULT_SPATIAL_RANGE),
 seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
 """
 Generate dataset with both sparse and densely sampled regions.

 Demonstrates challenges in interpolation with uneven sampling
 and the importance of appropriate neighbor selection.

 Parameters
 ----------
 n_sparse : int, default=30
 Number of points in sparse region
 n_dense : int, default=100
 Number of points in dense region
 dense_region_center : tuple of float
 Center of dense sampling region
 dense_region_radius : float
 Radius of dense sampling region
 x_range, y_range : tuple of float
 Overall spatial extent
 seed : int, optional
 Random seed

 Returns
 -------
 x, y, z : np.ndarray
 Coordinates and values

 Examples
 --------
 >>> from geostats.datasets.synthetic import generate_sparse_dense_mix
 >>> x, y, z = generate_sparse_dense_mix(
 ... n_sparse=50,
 ... n_dense=200,
 ... seed=42
 ... )

 Notes
 -----
 This sampling pattern is realistic in earth sciences where:
 - Detailed surveys are done in areas of interest
 - Sparse regional coverage elsewhere
 - Need to interpolate across both regions
 """
 if seed is not None:

 # Sparse points (uniform over entire region)
 x_sparse = np.random.uniform(x_range[0], x_range[1], n_sparse)
 y_sparse = np.random.uniform(y_range[0], y_range[1], n_sparse)

 # Dense points (concentrated in one region)
 angles = np.random.uniform(0, 2*np.pi, n_dense)
 radii = np.random.uniform(0, dense_region_radius, n_dense)
 x_dense = dense_region_center[0] + radii * np.cos(angles)
 y_dense = dense_region_center[1] + radii * np.sin(angles)

 # Clip dense points to bounds
 x_dense = np.clip(x_dense, x_range[0], x_range[1])
 y_dense = np.clip(y_dense, y_range[0], y_range[1])

 # Combine
 x = np.concatenate([x_sparse, x_dense])
 y = np.concatenate([y_sparse, y_dense])

 # Generate values with spatial trend
 x_norm = (x - x_range[0]) / (x_range[1] - x_range[0])
 y_norm = (y - y_range[0]) / (y_range[1] - y_range[0])
 z = 5.0 * x_norm + 3.0 * y_norm + np.random.normal(0, 0.5, len(x))

 return x, y, z
