"""
Matrix operations and utilities for kriging
"""

from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from scipy import linalg

from ..core.exceptions import KrigingError


def build_covariance_matrix(
    distances: npt.NDArray[np.float64],
    covariance_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """
    Build covariance matrix from distances and covariance function
    
    Parameters
    ----------
    distances : np.ndarray
        Distance matrix
    covariance_func : callable
        Covariance function that takes distances and returns covariances
        
    Returns
    -------
    np.ndarray
        Covariance matrix
    """
    return covariance_func(distances)


def build_variogram_matrix(
    distances: npt.NDArray[np.float64],
    variogram_func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """
    Build variogram matrix from distances and variogram function
    
    Parameters
    ----------
    distances : np.ndarray
        Distance matrix
    variogram_func : callable
        Variogram function that takes distances and returns semivariances
        
    Returns
    -------
    np.ndarray
        Variogram matrix
    """
    return variogram_func(distances)


def solve_kriging_system(
    A: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    method: str = "auto",
) -> npt.NDArray[np.float64]:
    """
    Solve kriging system A * weights = b
    
    Uses appropriate solver based on matrix properties and size.
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (typically covariance or variogram matrix)
    b : np.ndarray
        Right-hand side vector
    method : str
        Solution method: 'auto', 'cholesky', 'lu', 'lstsq'
        
    Returns
    -------
    np.ndarray
        Solution vector (kriging weights)
        
    Raises
    ------
    KrigingError
        If system cannot be solved
    """
    try:
        # Solution methods using dispatch
        solution_methods = {
            'cholesky': lambda: linalg.cho_solve(linalg.cho_factor(A), b),
            'lu': lambda: linalg.solve(A, b),
            'lstsq': lambda: linalg.lstsq(A, b)[0],
        }
        
        if method == "auto":
            # Try Cholesky first (fastest for symmetric positive definite)
            try:
                return linalg.cho_solve(linalg.cho_factor(A), b)
            except linalg.LinAlgError:
                # Fall back to LU decomposition
                method = "lu"
        
        if method not in solution_methods:
            valid_methods = ', '.join(['auto'] + list(solution_methods.keys()))
            raise ValueError(
                f"Unknown solution method '{method}'. "
                f"Valid methods: {valid_methods}"
            )
        
        return solution_methods[method]()
            
    except linalg.LinAlgError as e:
        raise KrigingError(f"Failed to solve kriging system: {e}")


def is_positive_definite(
    matrix: npt.NDArray[np.float64],
    tol: float = 1e-10,
) -> bool:
    """
    Check if a matrix is positive definite
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to check
    tol : float
        Tolerance for eigenvalue positivity
        
    Returns
    -------
    bool
        True if matrix is positive definite
    """
    try:
        # Try Cholesky decomposition
        linalg.cholesky(matrix)
        return True
    except linalg.LinAlgError:
        # Check eigenvalues as fallback
        eigenvalues = linalg.eigvalsh(matrix)
        return np.all(eigenvalues > tol)


def regularize_matrix(
    matrix: npt.NDArray[np.float64],
    epsilon: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """
    Regularize a matrix by adding small value to diagonal
    
    This improves numerical stability for near-singular matrices.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to regularize
    epsilon : float
        Regularization parameter
        
    Returns
    -------
    np.ndarray
        Regularized matrix
    """
    result = matrix.copy()
    n = min(result.shape)
    result.flat[::n + 1] += epsilon
    return result


def condition_number(matrix: npt.NDArray[np.float64]) -> float:
    """
    Calculate condition number of a matrix
    
    High condition numbers indicate ill-conditioned matrices
    that may lead to numerical instability.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to analyze
        
    Returns
    -------
    float
        Condition number
    """
    return np.linalg.cond(matrix)


def make_symmetric(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Make a matrix symmetric by averaging with its transpose
    
    Parameters
    ----------
    matrix : np.ndarray
        Matrix to symmetrize
        
    Returns
    -------
    np.ndarray
        Symmetric matrix
    """
    return (matrix + matrix.T) / 2


def add_nugget_effect(
    matrix: npt.NDArray[np.float64],
    nugget: float,
) -> npt.NDArray[np.float64]:
    """
    Add nugget effect to variogram/covariance matrix
    
    Parameters
    ----------
    matrix : np.ndarray
        Original matrix
    nugget : float
        Nugget variance
        
    Returns
    -------
    np.ndarray
        Matrix with nugget effect added to diagonal
    """
    result = matrix.copy()
    n = min(result.shape)
    result.flat[::n + 1] += nugget
    return result
