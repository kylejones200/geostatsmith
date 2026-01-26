"""
Unconditional Gaussian simulation

Generates random fields without conditioning to data.
Useful for understanding spatial patterns and testing.
"""

from typing import Optional
import numpy as np
import numpy.typing as npt

from .conditional_simulation import cholesky_simulation, turning_bands_simulation


def unconditional_gaussian_simulation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    covariance_model,
    n_realizations: int = 1,
    mean: float = 0.0,
    method: str = "cholesky",
    seed: Optional[int] = None,
    **kwargs,
) -> npt.NDArray[np.float64]:
    """
    Generate unconditional Gaussian simulations
    
    Parameters
    ----------
    x, y : np.ndarray
        Simulation coordinates
    covariance_model
        Spatial covariance model
    n_realizations : int
        Number of realizations
    mean : float
        Mean of the field
    method : str
        Simulation method: 'cholesky' or 'turning_bands'
    seed : int, optional
        Random seed
    **kwargs
        Additional arguments for the method
        
    Returns
    -------
    realizations : np.ndarray
        Shape (n_realizations, n_points)
    """
    if method == "cholesky":
        return cholesky_simulation(
            x, y, covariance_model,
            n_realizations=n_realizations,
            mean=mean,
            seed=seed
        )
    
    elif method == "turning_bands":
        # Convert covariance to variogram if needed
        n_bands = kwargs.get('n_bands', 100)
        return turning_bands_simulation(
            x, y, covariance_model,
            n_bands=n_bands,
            n_realizations=n_realizations,
            seed=seed
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
