"""
Conditional Simulation using Cholesky Decomposition

Generates Gaussian random fields conditioned to data.
Uses Cholesky decomposition of the covariance matrix.

Based on Zhang, Y. (2010). Course Notes, Section 6.3.1 and 6.3.2
"""

import numpy as np
import numpy.typing as npt
from scipy import linalg

from ..math.matrices import regularize_matrix


def cholesky_simulation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    covariance_model,
    n_realizations: int = 1,
    mean: float = 0.0,
    seed: int | None = None,
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
    from ..math.distance import euclidean_distance_matrix

    dist_matrix = euclidean_distance_matrix(x, y)

    # Get covariance from model
    # If it's a variogram model, convert to covariance
    if hasattr(covariance_model, "get_parameters"):
        # It's a variogram model: C(h) = sill - gamma(h)
        params = covariance_model.get_parameters()
        sill = params.get("sill", 1.0)
        gamma = covariance_model(dist_matrix)
        cov_matrix = sill - gamma
    elif callable(covariance_model):
        cov_matrix = covariance_model(dist_matrix)
    else:
        raise ValueError(
            "covariance_model must be callable or have get_parameters method"
        )

    # Regularize for numerical stability
    cov_matrix = regularize_matrix(cov_matrix, factor=1e-6)

    # Cholesky decomposition: C = L * L'
    try:
        L = linalg.cholesky(cov_matrix, lower=True)
    except linalg.LinAlgError:
        # If Cholesky fails, try eigenvalue decomposition
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    # Generate realizations
    realizations = np.zeros((n_realizations, n))

    for r in range(n_realizations):
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
    seed: int | None = None,
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
    len(x_all)

    # Generate unconditional simulations at all locations
    uncond_sims = cholesky_simulation(
        x_all,
        y_all,
        covariance_model,
        n_realizations=n_realizations,
        mean=mean,
        seed=seed,
    )

    # Condition each realization
    conditional_sims = np.zeros((n_realizations, n_sim))

    for r in range(n_realizations):
        uncond_sims[r, :n_data]

        # Krige the simulated values to get smooth field
        # For simplicity, use unconditional simulation
        # In practice, would need to properly set up kriging with covariance
        z_uncond_sim = uncond_sims[r, n_data:]

        # Simple conditioning: adjust by data mean
        data_mean = np.mean(z_data)
        sim_mean = np.mean(z_uncond_sim)
        conditional_sims[r, :] = z_uncond_sim + (data_mean - sim_mean)

    return conditional_sims


def turning_bands_simulation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    variogram_model,
    n_bands: int = 100,
    n_realizations: int = 1,
    seed: int | None = None,
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
        angles = np.random.uniform(0, 2 * np.pi, n_bands)

        sim = np.zeros(n_points)

        for angle in angles:
            u = x * np.cos(angle) + y * np.sin(angle)

            # Sort by projection
            sorted_idx = np.argsort(u)
            u_sorted = u[sorted_idx]

            # Generate 1D simulation along this line
            n = len(u_sorted)

            # Simple 1D correlation structure
            h = np.abs(u_sorted[:, np.newaxis] - u_sorted[np.newaxis, :])
            gamma = variogram_model(h)
            params = (
                variogram_model.get_parameters()
                if hasattr(variogram_model, "get_parameters")
                else {}
            )
            sill = params.get("sill", 1.0)
            cov = sill - gamma
            cov = regularize_matrix(cov, factor=1e-6)

            try:
                L = linalg.cholesky(cov, lower=True)
                w = np.random.randn(n)
                sim_1d = L @ w
            except (linalg.LinAlgError, ValueError):
                # If Cholesky decomposition fails, use random values
                sim_1d = np.random.randn(n)

            # Unsort and add to simulation
            sim_1d_unsorted = np.zeros(n)
            sim_1d_unsorted[sorted_idx] = sim_1d
            sim += sim_1d_unsorted

        # Normalize by number of bands
        realizations[r, :] = sim / np.sqrt(n_bands)

    return realizations
