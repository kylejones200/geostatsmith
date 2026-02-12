"""
Experimental variogram calculation algorithms

The experimental variogram (or semivariogram) estimates spatial correlation
from sample data using Matheron's estimator:

 γ(h) = 1/(2*N(h)) * Σ[z(xi) - z(xi+h)]²

where N(h) is the number of pairs separated by distance h.
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance_matrix, directional_distance

def experimental_variogram(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    lag_tol: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Calculate experimental (empirical) variogram

    Uses Matheron's classical estimator for the semivariogram.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates of sample points
    z : np.ndarray
        Values at sample points
    n_lags : int
        Number of lag bins
    maxlag : float, optional
        Maximum lag distance to consider
        If None, uses half the maximum distance
    lag_tol : float, optional
        Tolerance for lag binning
        If None, uses lag_width / 2

    Returns
    -------
    lags : np.ndarray
        Lag distances (bin centers)
    gamma : np.ndarray
        Semivariance values
    n_pairs : np.ndarray
        Number of pairs in each lag bin

    References
    ----------
    Matheron, G. (1963). Principles of geostatistics.
    """
    # Validate inputs
    x, y = validate_coordinates(x, y)
    z = validate_values(z, n_expected=len(x))

    # Calculate all pairwise distances
    dist = euclidean_distance_matrix(x, y)

    # Calculate squared differences
    z_diff_sq = (z[:, np.newaxis] - z[np.newaxis, :]) ** 2

    # Determine lag bins
    if maxlag is None:
        maxlag = np.max(dist) / 2.0

    lag_width = maxlag / n_lags
    lag_bins = np.linspace(0, maxlag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2

    if lag_tol is None:
        lag_tol = lag_width / 2.0

    # Compute variogram for each lag
    gamma = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=np.int64)

    for i in range(n_lags):
        lag_min = lag_bins[i]
        lag_max = lag_bins[i + 1]

        # Use upper triangle only (avoid double counting)
        mask = (dist >= lag_min) & (dist < lag_max)
        mask = np.triu(mask, k=1)  # Upper triangle, excluding diagonal

        n_pairs_lag = np.sum(mask)

        if n_pairs_lag > 0:
            gamma[i] = np.sum(z_diff_sq[mask]) / (2.0 * n_pairs_lag)
            n_pairs[i] = n_pairs_lag
        else:
            n_pairs[i] = 0

    return lag_centers, gamma, n_pairs

def experimental_variogram_directional(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    angle: float = 0.0,
    tolerance: float = 22.5,
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Calculate directional experimental variogram

    Computes variogram along a specific direction, useful for
    detecting and analyzing anisotropy.

    Parameters
    ----------
    x, y : np.ndarray
    Coordinates of sample points
    z : np.ndarray
    Values at sample points
    angle : float
    Direction angle in degrees (0-360)
    0° = East, 90° = North
    tolerance : float
    Angular tolerance in degrees
    n_lags : int
    Number of lag bins
    maxlag : float, optional
    Maximum lag distance

    Returns
    -------
    lags : np.ndarray
    Lag distances
    gamma : np.ndarray
    Semivariance values
    n_pairs : np.ndarray
    Number of pairs in each lag bin
    """
    # Validate inputs
    x, y = validate_coordinates(x, y)
    z = validate_values(z, n_expected=len(x))

    # Calculate distances and directional mask
    dist, dir_mask = directional_distance(x, y, x, y, angle, tolerance)

    # Calculate squared differences
    z_diff_sq = (z[:, np.newaxis] - z[np.newaxis, :]) ** 2

    # Determine lag bins
    if maxlag is None:
        maxlag = np.max(dist) / 2.0

    lag_width = maxlag / n_lags
    lag_bins = np.linspace(0, maxlag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2

    # Compute variogram for each lag
    gamma = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=np.int64)

    for i in range(n_lags):
        lag_min = lag_bins[i]
        lag_max = lag_bins[i + 1]

        # Find pairs in this lag bin and direction
        mask = (dist >= lag_min) & (dist < lag_max) & dir_mask
        mask = np.triu(mask, k=1)  # Upper triangle only

        n_pairs_lag = np.sum(mask)

        if n_pairs_lag > 0:
            gamma[i] = np.sum(z_diff_sq[mask]) / (2.0 * n_pairs_lag)
            n_pairs[i] = n_pairs_lag
        else:
            n_pairs[i] = 0

    return lag_centers, gamma, n_pairs

def variogram_cloud(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    maxlag: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate variogram cloud (all pairwise points)

    The variogram cloud shows all individual squared differences
    vs. distances, useful for detecting outliers and spatial patterns.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates of sample points
    z : np.ndarray
        Values at sample points
    maxlag : float, optional
        Maximum lag distance to include

    Returns
    -------
    distances : np.ndarray
        All pairwise distances
    semivariances : np.ndarray
        Squared differences / 2 for each pair
    """
    # Validate inputs
    x, y = validate_coordinates(x, y)
    z = validate_values(z, n_expected=len(x))

    # Calculate all pairwise distances
    dist = euclidean_distance_matrix(x, y)

    # Calculate semivariances (squared differences / 2)
    z_diff_sq = (z[:, np.newaxis] - z[np.newaxis, :]) ** 2
    semivar = z_diff_sq / 2.0

    # Extract upper triangle (avoid duplicates and self-pairs)
    mask = np.triu(np.ones_like(dist, dtype=bool), k=1)

    if maxlag is not None:
        mask = mask & (dist <= maxlag)

    distances = dist[mask]
    semivariances = semivar[mask]

    return distances, semivariances

def robust_variogram(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    n_lags: int = 15,
    maxlag: Optional[float] = None,
    estimator: str = "cressie",
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Calculate robust experimental variogram

    Uses robust estimators less sensitive to outliers.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinates
    z : np.ndarray
        Values
    n_lags : int
        Number of lag bins
    maxlag : float, optional
        Maximum lag distance
    estimator : str
        Robust estimator to use:
        - 'cressie': Cressie-Hawkins estimator
        - 'dowd': Dowd's estimator

    Returns
    -------
    lags : np.ndarray
        Lag distances
    gamma : np.ndarray
        Robust semivariance values
    n_pairs : np.ndarray
        Number of pairs in each lag

    References
    ----------
    Cressie, N. & Hawkins, D.M. (1980). Robust estimation of the variogram.
    """
    # Validate inputs
    x, y = validate_coordinates(x, y)
    z = validate_values(z, n_expected=len(x))

    # Calculate distances
    dist = euclidean_distance_matrix(x, y)

    # Determine lag bins
    if maxlag is None:
        maxlag = np.max(dist) / 2.0

    lag_width = maxlag / n_lags
    lag_bins = np.linspace(0, maxlag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2

    # Calculate absolute differences
    z_diff_abs = np.abs(z[:, np.newaxis] - z[np.newaxis, :])

    # Compute robust variogram for each lag
    gamma = np.zeros(n_lags)
    n_pairs = np.zeros(n_lags, dtype=np.int64)

    for i in range(n_lags):
        lag_min = lag_bins[i]
        lag_max = lag_bins[i + 1]

        mask = (dist >= lag_min) & (dist < lag_max)
        mask = np.triu(mask, k=1)

        n_pairs_lag = np.sum(mask)

        if n_pairs_lag > 0:
            z_diff = z_diff_abs[mask]

            if estimator == "cressie":
                # γ(h) = [(1/N(h) * Σ|z_i - z_j|^0.5)^4] / [0.457 + 0.494/N(h)]
                mean_fourth_root = np.mean(z_diff ** 0.5)
                gamma[i] = (mean_fourth_root ** 4) / (0.457 + 0.494 / n_pairs_lag)
            elif estimator == "dowd":
                # γ(h) = 2.198 * median(|z_i - z_j|)²
                median_diff = np.median(z_diff)
                gamma[i] = 2.198 * (median_diff ** 2)
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

            n_pairs[i] = n_pairs_lag
        else:
            n_pairs[i] = 0

    return lag_centers, gamma, n_pairs

def madogram(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
 n_lags: int = 15,
 maxlag: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
 """
 Calculate Madogram (median-based variogram)

 The Madogram is a robust variogram estimator using the median
 instead of the mean. It's less sensitive to outliers than Matheron's
 classical estimator.

 Formula:
 γ(h) = 0.5 * [median(|Z(xi) - Z(xi+h)|)]²

 The factor of 0.5 makes it comparable to the classical variogram
 under normality assumptions.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of sample points
 z : np.ndarray
 Values at sample points
 n_lags : int
 Number of lag bins
 maxlag : float, optional
 Maximum lag distance to consider

 Returns
 -------
 lags : np.ndarray
 Lag distances (bin centers)
 gamma : np.ndarray
 Madogram values
 n_pairs : np.ndarray
 Number of pairs in each lag bin

 References
 ----------
 - Cressie, N. & Hawkins, D.M. (1980). Robust estimation of the
 variogram: I. Mathematical Geology, 12:115-125.
 - Genton, M.G. (1998). Highly robust variogram estimation.
 Mathematical Geology, 30:213-221.

 Notes
 -----
 The Madogram is particularly useful when:
 - Data contains outliers
 - Distribution is heavy-tailed
 - Classical variogram shows erratic behavior
 """
 # Validate inputs
 x, y = validate_coordinates(x, y)
 z = validate_values(z, n_expected=len(x))

 # Calculate all pairwise distances
 dist = euclidean_distance_matrix(x, y)

 # Calculate absolute differences
 z_diff_abs = np.abs(z[:, np.newaxis] - z[np.newaxis, :])

    # Determine lag bins
    if maxlag is None:
        maxlag = np.max(dist) / 2.0

    lag_width = maxlag / n_lags
    lag_bins = np.linspace(0, maxlag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2

    # Compute madogram for each lag
    gamma = np.zeros(n_lags, dtype=np.float64)
    n_pairs = np.zeros(n_lags, dtype=np.int64)

    for i in range(n_lags):
        lag_min = lag_bins[i]
        lag_max = lag_bins[i + 1]

        # Use upper triangle only (avoid double counting)
        mask = (dist >= lag_min) & (dist < lag_max)
        mask = np.triu(mask, k=1)

        n_pairs_lag = np.sum(mask)

        if n_pairs_lag > 0:
            # Extract differences for this lag
            diffs = z_diff_abs[mask]

            # Madogram: 0.5 * [median(|differences|)]²
            median_diff = np.median(diffs)
            gamma[i] = 0.5 * (median_diff ** 2)
            n_pairs[i] = n_pairs_lag
        else:
            n_pairs[i] = 0

 return lag_centers, gamma, n_pairs

def rodogram(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
 n_lags: int = 15,
 maxlag: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
 """
 Calculate Rodogram (robust variogram estimator)

 The Rodogram is an alternative robust variogram estimator that uses
 a weighted combination of fourth-root transformed differences.
 It's particularly resistant to outliers while maintaining efficiency.

 Formula:
 γ(h) = [1/(2*N(h)) * Σ|Z(xi) - Z(xi+h)|^(1/2)]^4 / [0.457 + 0.494/N(h)]

 This is essentially the Cressie-Hawkins estimator, but presented
 as the "Rodogram" in some geostatistics literature.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of sample points
 z : np.ndarray
 Values at sample points
 n_lags : int
 Number of lag bins
 maxlag : float, optional
 Maximum lag distance to consider

 Returns
 -------
 lags : np.ndarray
 Lag distances (bin centers)
 gamma : np.ndarray
 Rodogram values
 n_pairs : np.ndarray
 Number of pairs in each lag bin

 References
 ----------
 - Cressie, N. & Hawkins, D.M. (1980). Robust estimation of the
 variogram: I. Mathematical Geology, 12:115-125.
 - Dowd, P.A. (1984). The variogram and kriging: Robust and resistant
 estimators. In Geostatistics for Natural Resources Characterization,
 Part 1, pp. 91-106.

 Notes
 -----
 The Rodogram provides a good balance between:
 - Robustness to outliers (better than Matheron)
 - Efficiency under normality (better than median-based)
 - Computational simplicity

 The normalization factor (0.457 + 0.494/N) adjusts for bias
 and depends on the number of pairs.
 """
 # Validate inputs
 x, y = validate_coordinates(x, y)
 z = validate_values(z, n_expected=len(x))

 # Calculate all pairwise distances
 dist = euclidean_distance_matrix(x, y)

 # Calculate absolute differences
 z_diff_abs = np.abs(z[:, np.newaxis] - z[np.newaxis, :])

    # Determine lag bins
    if maxlag is None:
        maxlag = np.max(dist) / 2.0

    lag_width = maxlag / n_lags
    lag_bins = np.linspace(0, maxlag, n_lags + 1)
    lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2

    # Compute rodogram for each lag
    gamma = np.zeros(n_lags, dtype=np.float64)
    n_pairs = np.zeros(n_lags, dtype=np.int64)

    for i in range(n_lags):
        lag_min = lag_bins[i]
        lag_max = lag_bins[i + 1]

        # Use upper triangle only (avoid double counting)
        mask = (dist >= lag_min) & (dist < lag_max)
        mask = np.triu(mask, k=1)

        n_pairs_lag = np.sum(mask)

        if n_pairs_lag > 0:
            # Extract differences for this lag
            diffs = z_diff_abs[mask]

            # Rodogram (Cressie-Hawkins):
            # γ(h) = [mean(|diff|^0.5)]^4 / [0.457 + 0.494/N(h)]
            mean_fourth_root = np.mean(diffs ** 0.5)
            normalization = 0.457 + 0.494 / n_pairs_lag
            gamma[i] = (mean_fourth_root ** 4) / normalization
            n_pairs[i] = n_pairs_lag
        else:
            n_pairs[i] = 0

 return lag_centers, gamma, n_pairs
