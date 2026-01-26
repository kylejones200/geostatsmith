"""
Conditional Simulation using Cholesky Decomposition

Generates Gaussian random fields conditioned to data.
Uses Cholesky decomposition of the covariance matrix.

Based on Zhang, Y. (2010). Course Notes, Section 6.3.1 and 6.3.2
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from scipy import linalg

from ..math.distance import euclidean_distance
from ..math.matrices import regularize_matrix

def cholesky_simulation(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 covariance_model,
 n_realizations: int = 1,
 mean: float = 0.0,
 seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
 """
 Unconditional simulation using Cholesky decomposition

 Generates Gaussian random fields by decomposing the covariance matrix.

 Algorithm:
 1. Build covariance matrix C from locations and covariance model
 2. Perform Cholesky decomposition: C = L * L'
 3. Generate independent N(0,1) random values: w
 4. Compute z = mean + L * w

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates where to simulate
 covariance_model
 Covariance function model
 n_realizations : int
 Number of realizations
 mean : float
 Mean of the field
 seed : int, optional
 Random seed

 Returns
 -------
 realizations : np.ndarray
 Shape (n_realizations, n_points)
 """
 if seed is not None:
 np.random.seed(seed)

 n = len(x)

 # Build covariance matrix
 dist_matrix = euclidean_distance(x, y, x, y)

 # Get covariance from model
 # If it's a variogram model, convert to covariance
 if hasattr(covariance_model, 'parameters'):
 if 'sill' in covariance_model.parameters:
 # It's a variogram model: C(h) = sill - γ(h)
 sill = covariance_model.parameters['sill']
 gamma = covariance_model(dist_matrix)
 cov_matrix = sill - gamma
 else:
 # It's a covariance model
 cov_matrix = covariance_model(dist_matrix)
 else:
 # Assume it's a callable covariance function
 cov_matrix = covariance_model(dist_matrix)

 # Regularize for numerical stability
 cov_matrix = regularize_matrix(cov_matrix, epsilon=1e-6)

 # Cholesky decomposition: C = L * L'
 try:
 L = linalg.cholesky(cov_matrix, lower=True)
 except linalg.LinAlgError:
 # If Cholesky fails, try eigenvalue decomposition
 eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
 eigenvalues = np.maximum(eigenvalues, 0) # Ensure non-negative
 L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

 # Generate realizations
 realizations = np.zeros((n_realizations, n))

 for r in range(n_realizations):
 # Generate independent N(0,1) random values
 w = np.random.randn(n)

 # Compute z = mean + L * w
 realizations[r, :] = mean + L @ w

 return realizations

def conditional_simulation(
 x_data: npt.NDArray[np.float64],
 y_data: npt.NDArray[np.float64],
 z_data: npt.NDArray[np.float64],
 x_sim: npt.NDArray[np.float64],
 y_sim: npt.NDArray[np.float64],
 covariance_model,
 n_realizations: int = 1,
 mean: float = 0.0,
 seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
 """
 Conditional simulation using conditioning by kriging

 Algorithm:
 1. Generate unconditional simulation at data and simulation locations
 2. Krige the simulated values at data locations
 3. Condition: z_cond = z_uncond + (z_data - z_kriged)

 Parameters
 ----------
 x_data, y_data : np.ndarray
 Coordinates of conditioning data
 z_data : np.ndarray
 Values at conditioning locations
 x_sim, y_sim : np.ndarray
 Coordinates where to simulate
 covariance_model
 Covariance model
 n_realizations : int
 Number of realizations
 mean : float
 Mean of the field
 seed : int, optional
 Random seed

 Returns
 -------
 realizations : np.ndarray
 Conditional realizations, shape (n_realizations, n_sim)
 """
 if seed is not None:
 np.random.seed(seed)

 n_data = len(x_data)
 n_sim = len(x_sim)

 # Combine data and simulation locations
 x_all = np.concatenate([x_data, x_sim])
 y_all = np.concatenate([y_data, y_sim])
 n_all = len(x_all)

 # Generate unconditional simulations at all locations
 uncond_sims = cholesky_simulation(
 x_all, y_all, covariance_model,
 n_realizations=n_realizations,
 mean=mean,
 seed=seed
 )

 # Condition each realization
 conditional_sims = np.zeros((n_realizations, n_sim))

 from ..algorithms.simple_kriging import SimpleKriging

 for r in range(n_realizations):
 # Extract simulated values at data locations
 z_sim_at_data = uncond_sims[r, :n_data]

 # Krige the simulated values to get smooth field
 sk = SimpleKriging(
 x_data, y_data, z_sim_at_data,
 variogram_model=None, # Will use covariance directly
 mean=mean
 )

 # Build covariance matrix manually
 dist_matrix = euclidean_distance(x_data, y_data, x_data, y_data)
 if hasattr(covariance_model, 'parameters'):
 if 'sill' in covariance_model.parameters:
 sill = covariance_model.parameters['sill']
 gamma = covariance_model(dist_matrix)
 sk.cov_matrix = sill - gamma
 else:
 sk.cov_matrix = covariance_model(dist_matrix)
 else:
 sk.cov_matrix = covariance_model(dist_matrix)

 sk.cov_matrix = regularize_matrix(sk.cov_matrix)

 try:
 # Krige simulated values at simulation locations
 z_kriged_sim, _ = sk.predict(x_sim, y_sim, return_variance=False)
 except Exception:
 # If kriging fails, use unconditional simulation
 z_kriged_sim = uncond_sims[r, n_data:]

 # Get unconditional simulation at simulation locations
 z_uncond_sim = uncond_sims[r, n_data:]

 # Krige actual data values
 sk_data = SimpleKriging(
 x_data, y_data, z_data,
 variogram_model=None,
 mean=mean
 )
 sk_data.cov_matrix = sk.cov_matrix

 try:
 z_kriged_data, _ = sk_data.predict(x_sim, y_sim, return_variance=False)
 except Exception:
 z_kriged_data = np.full(n_sim, mean)

 # Conditioning: z_cond = z_uncond + (z_data_kriged - z_sim_kriged)
 conditional_sims[r, :] = z_uncond_sim + (z_kriged_data - z_kriged_sim)

 return conditional_sims

def turning_bands_simulation(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 variogram_model,
 n_bands: int = 100,
 n_realizations: int = 1,
 seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
 """
 Turning Bands Method for unconditional simulation

 Faster alternative to Cholesky for large grids.
 Approximates 2D simulation by combining 1D simulations along random lines.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 variogram_model
 Variogram model
 n_bands : int
 Number of bands (lines)
 n_realizations : int
 Number of realizations
 seed : int, optional
 Random seed

 Returns
 -------
 realizations : np.ndarray
 Shape (n_realizations, n_points)
 """
 if seed is not None:
 np.random.seed(seed)

 n_points = len(x)
 realizations = np.zeros((n_realizations, n_points))

 for r in range(n_realizations):
 # Generate random directions
 angles = np.random.uniform(0, 2*np.pi, n_bands)

 sim = np.zeros(n_points)

 for angle in angles:
 # Project points onto line at this angle
 u = x * np.cos(angle) + y * np.sin(angle)

 # Sort by projection
 sorted_idx = np.argsort(u)
 u_sorted = u[sorted_idx]

 # Generate 1D simulation along this line
 n = len(u_sorted)

 # Simple 1D correlation structure
 h = np.abs(u_sorted[:, np.newaxis] - u_sorted[np.newaxis, :])
 gamma = variogram_model(h)
 sill = variogram_model.parameters.get('sill', 1.0)
 cov = sill - gamma
 cov = regularize_matrix(cov, epsilon=1e-6)

 try:
 L = linalg.cholesky(cov, lower=True)
 w = np.random.randn(n)
 sim_1d = L @ w
 except:
 sim_1d = np.random.randn(n)

 # Unsort and add to simulation
 sim_1d_unsorted = np.zeros(n)
 sim_1d_unsorted[sorted_idx] = sim_1d
 sim += sim_1d_unsorted

 # Normalize by number of bands
 realizations[r, :] = sim / np.sqrt(n_bands)

 return realizations
