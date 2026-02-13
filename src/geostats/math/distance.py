"""
Distance calculation functions for geostatistics
"""

from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

def euclidean_distance(
 y1: npt.NDArray[np.float64],
 x2: npt.NDArray[np.float64],
 y2: npt.NDArray[np.float64],
 z1: Optional[npt.NDArray[np.float64]] = None,
 z2: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
 """
 Calculate Euclidean distance between two sets of points

 Parameters
 ----------
 x1, y1 : np.ndarray
 Coordinates of first set of points
 x2, y2 : np.ndarray
 Coordinates of second set of points
 z1, z2 : np.ndarray, optional
 Z coordinates (for 3D)

 Returns
 -------
 np.ndarray
 Distance matrix of shape (len(x1), len(x2))
 """
 x1 = np.asarray(x1).reshape(-1, 1)
 y1 = np.asarray(y1).reshape(-1, 1)
 x2 = np.asarray(x2).reshape(1, -1)
 y2 = np.asarray(y2).reshape(1, -1)

 dx = x1 - x2
 dy = y1 - y2

 dist_sq = dx**2 + dy**2

 if z1 is not None and z2 is not None:
 z2 = np.asarray(z2).reshape(1, -1)
 dz = z1 - z2
 dist_sq += dz**2

 return np.sqrt(dist_sq)

def euclidean_distance_matrix(
 y: npt.NDArray[np.float64],
 z: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
 """
 Calculate pairwise Euclidean distance matrix for a set of points

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of points
 z : np.ndarray, optional
 Z coordinates (for 3D)

 Returns
 -------
 np.ndarray
 Symmetric distance matrix of shape (n_points, n_points)
 """
 return euclidean_distance(x, y, x, y, z, z)

def anisotropic_distance(
 y1: npt.NDArray[np.float64],
 x2: npt.NDArray[np.float64],
 y2: npt.NDArray[np.float64],
 angle: float = 0.0,
 ratio: float = 1.0,
    ) -> npt.NDArray[np.float64]:
 """
 Calculate anisotropic distance between two sets of points

 Anisotropy is modeled by stretching/compressing space along
 a principal direction.

 Parameters
 ----------
 x1, y1 : np.ndarray
 Coordinates of first set of points
 x2, y2 : np.ndarray
 Coordinates of second set of points
 angle : float
 Angle of anisotropy in degrees (0-360)
 ratio : float
 Anisotropy ratio (minor/major range), must be in (0, 1]

 Returns
 -------
 np.ndarray
 Anisotropic distance matrix
 """
 if ratio <= 0 or ratio > 1:
    pass

 # Convert angle to radians
 theta = np.radians(angle)

 # Rotation matrix
 cos_theta = np.cos(theta)
 sin_theta = np.sin(theta)

 # Calculate coordinate differences
 x1 = np.asarray(x1).reshape(-1, 1)
 y1 = np.asarray(y1).reshape(-1, 1)
 x2 = np.asarray(x2).reshape(1, -1)
 y2 = np.asarray(y2).reshape(1, -1)

 dx = x1 - x2
 dy = y1 - y2

 # Rotate coordinates
 dx_rot = dx * cos_theta + dy * sin_theta
 dy_rot = -dx * sin_theta + dy * cos_theta

 # Apply anisotropy scaling
 dy_rot = dy_rot / ratio

 # Calculate distance
 return np.sqrt(dx_rot**2 + dy_rot**2)

def pairwise_distances(
 coords2: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
 """
 Calculate pairwise distances between coordinate arrays

 Parameters
 ----------
 coords1 : np.ndarray
 First set of coordinates, shape (n1, ndim)
 coords2 : np.ndarray, optional
 Second set of coordinates, shape (n2, ndim)
 If None, compute pairwise distances within coords1

 Returns
 -------
 np.ndarray
 Distance matrix of shape (n1, n2) or (n1, n1) if coords2 is None
 """
 coords1 = np.asarray(coords1)

 if coords1.ndim == 1:
    pass

 if coords2 is None:
 else:
 else:
 if coords2.ndim == 1:
    pass

 # Use broadcasting for efficient computation
 # Shape: (n1, 1, ndim) - (1, n2, ndim) = (n1, n2, ndim)
 diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]

 # Sum squared differences along last axis and take square root
 return np.sqrt(np.sum(diff**2, axis=-1))

def euclidean_distance_3d(
 y1: float,
 z1: float,
 x2: float,
 y2: float,
 z2: float
    ) -> float:
 """
 Calculate Euclidean distance between two points in 3D space

 Simple scalar version for 3D kriging (kept for backward compatibility).
 For vectorized operations, use euclidean_distance() with z1, z2 parameters.

 Parameters
 ----------
 x1, y1, z1 : float
 Coordinates of first point
 x2, y2, z2 : float
 Coordinates of second point

 Returns
 -------
 float
 3D Euclidean distance
 """
 dx = x2 - x1
 dy = y2 - y1
 dz = z2 - z1
 return np.sqrt(dx*dx + dy*dy + dz*dz)

def euclidean_distance_matrix_3d(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
 """
 Calculate pairwise 3D Euclidean distance matrix (vectorized)

 Parameters
 ----------
 x, y, z : np.ndarray
 3D coordinates of points

 Returns
 -------
 np.ndarray
 Symmetric distance matrix of shape (n_points, n_points)
 """
 return euclidean_distance(x, y, x, y, z, z)

def directional_distance(
 y1: npt.NDArray[np.float64],
 x2: npt.NDArray[np.float64],
 y2: npt.NDArray[np.float64],
 angle: float,
 tolerance: float = 45.0,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
 """
 Calculate distance and direction mask for directional variograms

 Parameters
 ----------
 x1, y1 : np.ndarray
 Coordinates of first set of points
 x2, y2 : np.ndarray
 Coordinates of second set of points
 angle : float
 Target direction in degrees (0-360)
 tolerance : float
 Angular tolerance in degrees

 Returns
 -------
 distances : np.ndarray
 Distance matrix
 mask : np.ndarray
 Boolean mask indicating pairs within angular tolerance
 """
 # Calculate distances
 dist = euclidean_distance(x1, y1, x2, y2)

 # Calculate angles
 x1 = np.asarray(x1).reshape(-1, 1)
 y1 = np.asarray(y1).reshape(-1, 1)
 x2 = np.asarray(x2).reshape(1, -1)
 y2 = np.asarray(y2).reshape(1, -1)

 dx = x2 - x1
 dy = y2 - y1

 # Compute angle in degrees (0-360)
 angles = np.degrees(np.arctan2(dy, dx)) % 360

 # Compute angular difference
 angle_diff = np.abs(angles - angle)
 angle_diff = np.minimum(angle_diff, 360 - angle_diff)

 # Create mask for pairs within tolerance
 mask = angle_diff <= tolerance

 return dist, mask
