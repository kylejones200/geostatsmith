"""
Sample elevation datasets for demonstration and testing.

Provides access to small sample DEM datasets similar to those
discussed in Python Recipes for Earth Sciences (Trauth 2024).

Note: These are small sample datasets for demonstration.
For full ETOPO2022, SRTM, or GTOPO30 data, users should download
from the official sources.
"""

from typing import Tuple, Dict, Any
import numpy as np
import numpy.typing as npt

from ..core.logging_config import get_logger
import logging

logger = logging.getLogger(__name__)

logger = get_logger(__name__)

def load_synthetic_dem_sample() -> Dict[str, Any]:
 """
 Load a synthetic DEM sample for testing and demonstrations.

 Creates a small synthetic elevation dataset that mimics real DEMs
 with hills, valleys, and realistic terrain features.

 Returns
 -------
 data : dict
 Dictionary containing:
 - 'x': X coordinates (1D array)
 - 'y': Y coordinates (1D array)
 - 'z': Elevation values (1D array, sparse sampling)
 - 'X_grid': X grid coordinates (2D array)
 - 'Y_grid': Y grid coordinates (2D array)
 - 'Z_true': True elevation on grid (2D array)
 - 'metadata': Dataset information

 Examples
 --------
 >>> from geostats.datasets import load_synthetic_dem_sample
 >>> data = load_synthetic_dem_sample()
 >>> logger.info(f"Sample has {len(data['x'])} sparse points")
 >>> logger.info(f"Grid size: {data['X_grid'].shape}")

 Notes
 -----
 This synthetic dataset is useful for:
 - Testing interpolation methods
 - Comparing results to ground truth
 - Demonstrating DEM interpolation workflows
 - Educational purposes

 The dataset mimics characteristics of real DEMs:
 - Multiple peaks and valleys
 - Smooth terrain transitions
 - Realistic elevation ranges
 - Sparse sample points for interpolation
 """
 # Create fine grid for "true" DEM
 x_grid = np.linspace(0, 100, 100)
 y_grid = np.linspace(0, 100, 100)
 X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

 # Generate synthetic terrain with multiple hills
 Z_true = _generate_terrain(X_grid, Y_grid)

 # Create sparse sampling (what we would "measure")
 np.random.seed(42)
 n_samples = 150
 sample_idx_x = np.random.choice(len(x_grid), n_samples)
 sample_idx_y = np.random.choice(len(y_grid), n_samples)

 x = x_grid[sample_idx_x]
 y = y_grid[sample_idx_y]
 z = Z_true[sample_idx_y, sample_idx_x] # Note: matrix indexing

 # Add small measurement noise
 z = z + np.random.normal(0, 2.0, len(z))

 metadata = {
 'name': 'Synthetic DEM Sample',
 'description': 'Synthetic elevation data with hills and valleys',
 'n_samples': n_samples,
 'grid_size': (100, 100),
 'x_range': (0, 100),
 'y_range': (0, 100),
 'z_range': (float(np.min(Z_true)), float(np.max(Z_true))),
 'units': 'arbitrary',
 'reference': 'Generated for geostats library',
 }

 return {
 'x': x,
 'y': y,
 'z': z,
 'X_grid': X_grid,
 'Y_grid': Y_grid,
 'Z_true': Z_true,
 'metadata': metadata,
 }

def load_volcano_sample() -> Dict[str, Any]:
 """
 Load a synthetic volcano elevation dataset.

 Simulates elevation data around a volcanic cone, useful for
 demonstrating interpolation of steep terrain.

 Returns
 -------
 data : dict
 Dictionary containing sparse samples and grid information

 Examples
 --------
 >>> from geostats.datasets import load_volcano_sample
 >>> data = load_volcano_sample()
 >>> logger.info(f"Peak elevation: {data['metadata']['peak_elevation']:.1f}")

 Notes
 -----
 Volcanic terrain is challenging for interpolation due to:
 - Steep slopes near summit
 - Radial symmetry
 - Sharp elevation gradients

 This dataset is useful for testing interpolation methods on
 terrain with strong gradients.
 """
 # Create grid
 x_grid = np.linspace(-50, 50, 80)
 y_grid = np.linspace(-50, 50, 80)
 X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

 # Generate volcano shape
 distance = np.sqrt(X_grid**2 + Y_grid**2)

 # Volcanic cone with crater
 Z_true = np.maximum(0, 1000 - 15 * distance)

 # Add crater at top
 crater_mask = distance < 5
 Z_true[crater_mask] = 1000 - 50 * distance[crater_mask]

 # Add some roughness
 np.random.seed(123)
 roughness = np.random.normal(0, 10, Z_true.shape)
 Z_true = Z_true + roughness

 # Sparse sampling with more points near summit
 n_samples_base = 80
 n_samples_summit = 40

 # Base samples
 angles_base = np.random.uniform(0, 2*np.pi, n_samples_base)
 radii_base = np.random.uniform(10, 50, n_samples_base)
 x_base = radii_base * np.cos(angles_base)
 y_base = radii_base * np.sin(angles_base)

 # Summit samples
 angles_summit = np.random.uniform(0, 2*np.pi, n_samples_summit)
 radii_summit = np.random.uniform(0, 15, n_samples_summit)
 x_summit = radii_summit * np.cos(angles_summit)
 y_summit = radii_summit * np.sin(angles_summit)

 x = np.concatenate([x_base, x_summit])
 y = np.concatenate([y_base, y_summit])

 # Interpolate to get z values
 from scipy.interpolate import RegularGridInterpolator
 interpolator = RegularGridInterpolator(
 (y_grid, x_grid), Z_true, method='linear', bounds_error=False, fill_value=0
 )
 z = interpolator(np.column_stack([y, x]))

 # Add measurement noise
 z = z + np.random.normal(0, 5.0, len(z))

 metadata = {
 'name': 'Synthetic Volcano',
 'description': 'Synthetic volcanic cone elevation data',
 'n_samples': len(x),
 'grid_size': (80, 80),
 'x_range': (-50, 50),
 'y_range': (-50, 50),
 'peak_elevation': float(np.max(Z_true)),
 'units': 'meters (simulated)',
 'reference': 'Generated for geostats library',
 }

 return {
 'x': x,
 'y': y,
 'z': z,
 'X_grid': X_grid,
 'Y_grid': Y_grid,
 'Z_true': Z_true,
 'metadata': metadata,
 }

def load_valley_sample() -> Dict[str, Any]:
 """
 Load a synthetic valley elevation dataset.

 Simulates a U-shaped valley with ridges on both sides,
 useful for demonstrating interpolation of linear features.

 Returns
 -------
 data : dict
 Dictionary containing sparse samples and grid information

 Examples
 --------
 >>> from geostats.datasets import load_valley_sample
 >>> data = load_valley_sample()
 >>> logger.info(f"Valley floor elevation: {data['metadata']['valley_floor']:.1f}")

 Notes
 -----
 Valley terrain demonstrates:
 - Anisotropic spatial correlation
 - Linear features
 - Different correlation lengths in different directions
 """
 # Create grid
 x_grid = np.linspace(0, 100, 100)
 y_grid = np.linspace(0, 100, 100)
 X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

 # Valley runs diagonally
 # Rotate coordinates
 angle = np.pi / 4
 X_rot = X_grid * np.cos(angle) + Y_grid * np.sin(angle)
 Y_rot = -X_grid * np.sin(angle) + Y_grid * np.cos(angle)

 # U-shaped valley profile
 Z_true = 500 + (Y_rot / 15)**2 * 200

 # Add ridges with some noise
 np.random.seed(234)
 Z_true = Z_true + np.random.normal(0, 10, Z_true.shape)

 # Sparse sampling - more along valley floor
 n_samples = 120

 # Random samples
 x = np.random.uniform(0, 100, n_samples)
 y = np.random.uniform(0, 100, n_samples)

 # Interpolate z values
 from scipy.interpolate import RegularGridInterpolator
 interpolator = RegularGridInterpolator(
 (y_grid, x_grid), Z_true, method='linear'
 )
 z = interpolator(np.column_stack([y, x]))

 # Add noise
 z = z + np.random.normal(0, 15.0, len(z))

 metadata = {
 'name': 'Synthetic Valley',
 'description': 'U-shaped valley with ridges',
 'n_samples': n_samples,
 'grid_size': (100, 100),
 'x_range': (0, 100),
 'y_range': (0, 100),
 'valley_floor': 500.0,
 'valley_orientation': '45 degrees',
 'units': 'meters (simulated)',
 'reference': 'Generated for geostats library',
 }

 return {
 'x': x,
 'y': y,
 'z': z,
 'X_grid': X_grid,
 'Y_grid': Y_grid,
 'Z_true': Z_true,
 'metadata': metadata,
 }

def _generate_terrain(X: npt.NDArray[np.float64], Y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
 """
 Generate synthetic terrain with multiple hills.

 Helper function for creating realistic-looking elevation data.
 """
 # Multiple Gaussian hills
 peaks = [
 {'x': 30, 'y': 30, 'height': 150, 'width': 15},
 {'x': 70, 'y': 60, 'height': 200, 'width': 20},
 {'x': 50, 'y': 80, 'height': 120, 'width': 12},
 {'x': 20, 'y': 70, 'height': 100, 'width': 10},
 {'x': 80, 'y': 20, 'height': 180, 'width': 18},
 ]

 Z = np.zeros_like(X)

 for peak in peaks:
 dx = X - peak['x']
 dy = Y - peak['y']
 dist_sq = dx**2 + dy**2
 Z += peak['height'] * np.exp(-dist_sq / (2 * peak['width']**2))

 # Add base elevation
 Z += 50

 # Add gentle regional slope
 Z += 0.3 * X + 0.2 * Y

 return Z
