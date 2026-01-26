"""
Sequential Gaussian Simulation (SGS)

SGS is the most widely used geostatistical simulation algorithm.
It generates multiple equiprobable realizations that honor:
1. Data values at conditioning points
2. Spatial continuity model (variogram)
3. Target histogram (Gaussian after normal score transform)

Algorithm:
1. Transform data to Gaussian space (normal score transform)
2. Define a random path through all grid nodes
3. For each node in the path:
   - Use simple kriging to estimate mean and variance
   - Draw a random value from N(mean, variance)
   - Add to conditioning data set
4. Back-transform to original space

Reference:
- Zhang, Y. (2010). Course Notes, Section 6.3.3
- Deutsch & Journel (1998). GSLIB, Chapter 6
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from scipy import stats

from ..algorithms.simple_kriging import SimpleKriging
from ..core.exceptions import KrigingError


def normal_score_transform(
    data: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], callable]:
    """
    Transform data to standard normal distribution
    
    Uses the normal score transform (NST) which ranks data
    and assigns Gaussian quantiles.
    
    Parameters
    ----------
    data : np.ndarray
        Original data values
        
    Returns
    -------
    transformed_data : np.ndarray
        Data in Gaussian space
    back_transform_func : callable
        Function to back-transform from Gaussian to original space
    """
    n = len(data)
    
    # Rank the data
    ranks = stats.rankdata(data, method='average')
    
    # Convert ranks to probabilities (avoid 0 and 1)
    probs = (ranks - 0.5) / n
    
    # Transform to standard normal
    transformed = stats.norm.ppf(probs)
    
    # Create back-transform function using interpolation
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_transformed = transformed[sorted_indices]
    
    def back_transform(y_gaussian):
        """Back-transform from Gaussian to original space"""
        return np.interp(y_gaussian, sorted_transformed, sorted_data,
                        left=sorted_data[0], right=sorted_data[-1])
    
    return transformed, back_transform


def sequential_gaussian_simulation(
    x_data: npt.NDArray[np.float64],
    y_data: npt.NDArray[np.float64],
    z_data: npt.NDArray[np.float64],
    x_grid: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    variogram_model,
    n_realizations: int = 1,
    seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """
    Perform Sequential Gaussian Simulation
    
    Parameters
    ----------
    x_data, y_data : np.ndarray
        Coordinates of conditioning data
    z_data : np.ndarray
        Values at conditioning locations
    x_grid, y_grid : np.ndarray
        Grid coordinates for simulation
    variogram_model : VariogramModelBase
        Fitted variogram model
    n_realizations : int
        Number of realizations to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    realizations : np.ndarray
        Shape (n_realizations, n_grid_points)
        Multiple realizations of the spatial field
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_grid = len(x_grid)
    realizations = np.zeros((n_realizations, n_grid))
    
    # Normal score transform
    z_gaussian, back_transform = normal_score_transform(z_data)
    
    for r in range(n_realizations):
        # Create a random path through grid nodes
        path = np.random.permutation(n_grid)
        
        # Initialize simulation with conditioning data
        x_sim = np.copy(x_data)
        y_sim = np.copy(y_data)
        z_sim = np.copy(z_gaussian)
        
        sim_values = np.zeros(n_grid)
        
        # Sequential simulation along random path
        for i in path:
            # Current location
            x0 = x_grid[i]
            y0 = y_grid[i]
            
            # Simple kriging to get mean and variance
            try:
                sk = SimpleKriging(
                    x_sim, y_sim, z_sim,
                    variogram_model=variogram_model,
                    mean=0.0  # Standard normal has mean 0
                )
                
                mean, variance = sk.predict(
                    np.array([x0]),
                    np.array([y0]),
                    return_variance=True
                )
                
                mean = mean[0]
                variance = variance[0]
                
            except Exception:
                # If kriging fails, use unconditional simulation
                mean = 0.0
                variance = 1.0
            
            # Ensure variance is positive
            std_dev = np.sqrt(max(variance, 0.01))
            
            # Draw random value from conditional distribution
            simulated_value = np.random.normal(mean, std_dev)
            
            # Store simulated value
            sim_values[i] = simulated_value
            
            # Add to conditioning data for next iteration
            x_sim = np.append(x_sim, x0)
            y_sim = np.append(y_sim, y0)
            z_sim = np.append(z_sim, simulated_value)
        
        # Back-transform to original space
        realizations[r, :] = back_transform(sim_values)
    
    return realizations


class SequentialGaussianSimulation:
    """
    Sequential Gaussian Simulation class
    
    Provides a class interface for SGS with additional features.
    """
    
    def __init__(
        self,
        x_data: npt.NDArray[np.float64],
        y_data: npt.NDArray[np.float64],
        z_data: npt.NDArray[np.float64],
        variogram_model,
    ):
        """
        Initialize SGS
        
        Parameters
        ----------
        x_data, y_data : np.ndarray
            Conditioning data coordinates
        z_data : np.ndarray
            Conditioning data values
        variogram_model : VariogramModelBase
            Spatial continuity model
        """
        self.x_data = np.asarray(x_data, dtype=np.float64)
        self.y_data = np.asarray(y_data, dtype=np.float64)
        self.z_data = np.asarray(z_data, dtype=np.float64)
        self.variogram_model = variogram_model
        
        # Perform normal score transform
        self.z_gaussian, self.back_transform = normal_score_transform(z_data)
    
    def simulate(
        self,
        x_grid: npt.NDArray[np.float64],
        y_grid: npt.NDArray[np.float64],
        n_realizations: int = 1,
        seed: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Generate realizations
        
        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            Grid coordinates
        n_realizations : int
            Number of realizations
        seed : int, optional
            Random seed
            
        Returns
        -------
        np.ndarray
            Realizations, shape (n_realizations, n_grid)
        """
        return sequential_gaussian_simulation(
            self.x_data,
            self.y_data,
            self.z_data,
            x_grid,
            y_grid,
            self.variogram_model,
            n_realizations=n_realizations,
            seed=seed,
        )
    
    def simulate_grid(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int = 50,
        ny: int = 50,
        n_realizations: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Simulate on a regular grid
        
        Parameters
        ----------
        x_min, x_max, y_min, y_max : float
            Grid bounds
        nx, ny : int
            Grid dimensions
        n_realizations : int
            Number of realizations
        seed : int, optional
            Random seed
            
        Returns
        -------
        X, Y : np.ndarray
            Meshgrid coordinates
        realizations : np.ndarray
            Simulated values, shape (n_realizations, ny, nx)
        """
        # Create grid
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        x_grid = X.flatten()
        y_grid = Y.flatten()
        
        # Simulate
        realizations_flat = self.simulate(
            x_grid, y_grid,
            n_realizations=n_realizations,
            seed=seed
        )
        
        # Reshape to grid
        realizations = realizations_flat.reshape(n_realizations, ny, nx)
        
        return X, Y, realizations
    
    def get_statistics(
        self,
        realizations: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Calculate statistics from multiple realizations
        
        Parameters
        ----------
        realizations : np.ndarray
            Multiple realizations, shape (n_realizations, ...)
            
        Returns
        -------
        mean : np.ndarray
            E-type estimate (mean of realizations)
        std : np.ndarray
            Standard deviation
        p10 : np.ndarray
            10th percentile
        p90 : np.ndarray
            90th percentile
        """
        mean = np.mean(realizations, axis=0)
        std = np.std(realizations, axis=0)
        p10 = np.percentile(realizations, 10, axis=0)
        p90 = np.percentile(realizations, 90, axis=0)
        
        return mean, std, p10, p90
