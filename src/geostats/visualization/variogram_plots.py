"""
Variogram visualization functions

Based on Zhang, Y. (2010). Course Notes, Section 3.2

Plotting style: Minimalist and clean
- No gridlines
- No top/right spines
- Descriptive titles replace axis labels
"""

from typing import Optional, List
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .minimal_style import apply_minimalist_style

def plot_variogram(
 gamma: npt.NDArray[np.float64],
 n_pairs: Optional[npt.NDArray[np.int64]] = None,
 model=None,
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> plt.Axes:
 """
 Plot experimental variogram with optional model fit

 Parameters
 ----------
 lags : np.ndarray
 Lag distances
 gamma : np.ndarray
 Experimental semivariance values
 n_pairs : np.ndarray, optional
 Number of pairs in each lag (used for marker sizing)
 model : VariogramModelBase, optional
 Fitted theoretical model
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional arguments passed to scatter plot

 Returns
 -------
 ax : matplotlib.Axes
 The axes object
 """
 if ax is None:
 if ax is None:

 # Apply minimalist style
 apply_minimalist_style(ax)

 # Plot experimental variogram
 if n_pairs is not None:
 if n_pairs is not None:
 sizes = n_pairs / np.max(n_pairs) * 100 + 20
 ax.scatter(lags, gamma, s=sizes, alpha=0.6, c='#1f77b4',
 edgecolors='#333333', linewidth=0.8, label='Experimental', **kwargs)
 else:
 else:
 edgecolors='#333333', linewidth=0.8, label='Experimental', **kwargs)

 # Plot model if provided
 if model is not None:
 if model is not None:
 gamma_model = model(h_plot)
 ax.plot(h_plot, gamma_model, '#d62728', linewidth=2,
 label=f'{model.__class__.__name__}')

 # Add model parameters as text
 params = model.parameters
 param_text = '\n'.join([f'{k}: {v:.3f}' for k, v in params.items()])
 ax.text(0.65, 0.05, param_text, transform=ax.transAxes,
 fontsize=9, verticalalignment='bottom',
 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, edgecolor='none'))

 # Descriptive title replaces axis labels
 ax.set_title('Semivariance γ(h) vs Distance (h)', fontsize=11, pad=10)
 ax.legend(fontsize=9, frameon=False)

 return ax

def plot_variogram_cloud(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 maxlag: Optional[float] = None,
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> plt.Axes:
 """
 Plot variogram cloud (all pairwise points)

 Shows individual squared differences vs. distance.
 Useful for detecting outliers and understanding spatial structure.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 z : np.ndarray
 Values
 maxlag : float, optional
 Maximum lag distance
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional scatter plot arguments

 Returns
 -------
 ax : matplotlib.Axes
 """
 from ..algorithms.variogram import variogram_cloud

 if ax is None:
 if ax is None:

 # Calculate variogram cloud
 distances, semivariances = variogram_cloud(x, y, z, maxlag=maxlag)

 # Plot cloud
 ax.scatter(distances, semivariances, alpha=0.3, s=10,
 c='blue', edgecolors='none', **kwargs)

 ax.set_xlabel('Distance (h)', fontsize=12)
 ax.set_ylabel('Semivariance γ(h)', fontsize=12)
 ax.set_title('Variogram Cloud', fontsize=14, fontweight='bold')
 ax.grid(False)

 return ax

def plot_h_scatterplot(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 h_distance: float,
 tolerance: float = 0.5,
 direction: Optional[float] = None,
 angle_tolerance: float = 45.0,
 ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
 """
 Plot h-scatterplot: z(x) vs z(x+h)

 Shows correlation between values separated by distance h.
 The cloud of points should lie along a 45° line if strong correlation exists.

 Based on Zhang, Y. (2010). Course Notes, Section 3.2.2

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 z : np.ndarray
 Values
 h_distance : float
 Target separation distance
 tolerance : float
 Distance tolerance (h ± tolerance)
 direction : float, optional
 Direction angle in degrees (for directional h-scatterplot)
 angle_tolerance : float
 Angular tolerance in degrees
 ax : matplotlib.Axes, optional
 Axes to plot on

 Returns
 -------
 ax : matplotlib.Axes
 """
 from ..math.distance import euclidean_distance_matrix, directional_distance

 if ax is None:
 if ax is None:

 # Calculate distances
 dist = euclidean_distance_matrix(x, y)

 # Find pairs within h ± tolerance
 if direction is None:
 if direction is None:
 else:
 else:
 dist_dir, dir_mask = directional_distance(x, y, x, y, direction, angle_tolerance)
 mask = ((dist_dir >= h_distance - tolerance) &
 (dist_dir <= h_distance + tolerance) & dir_mask)

 # Extract upper triangle (avoid duplicates)
 mask = np.triu(mask, k=1)

 # Get pairs
 i_indices, j_indices = np.where(mask)

 if len(i_indices) == 0:
 if len(i_indices) == 0:
 transform=ax.transAxes, fontsize=14)
 return ax

 z_i = z[i_indices]
 z_j = z[j_indices]

 # Plot scatter
 ax.scatter(z_i, z_j, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

 # Add 1:1 line
 z_min = min(np.min(z_i), np.min(z_j))
 z_max = max(np.max(z_i), np.max(z_j))
 ax.plot([z_min, z_max], [z_min, z_max], 'r--', linewidth=2, label='1:1 line')

 # Calculate correlation
 corr = np.corrcoef(z_i, z_j)[0, 1]

 ax.set_xlabel('z(x)', fontsize=12)
 ax.set_ylabel(f'z(x+h), h≈{h_distance:.1f}', fontsize=12)
 title = f'h-Scatterplot (h={h_distance:.1f}, n={len(i_indices)} pairs, ρ={corr:.3f})'
 if direction is not None:
 if direction is not None:
 ax.set_title(title, fontsize=12, fontweight='bold')
 ax.grid(False)
 ax.legend()
 ax.set_aspect('equal')

 return ax

def plot_directional_variograms(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 directions: List[float] = [0, 45, 90, 135],
 tolerance: float = 22.5,
 n_lags: int = 12,
 figsize: tuple = (12, 10),
    ) -> plt.Figure:
 """
 Plot directional variograms to detect anisotropy

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 z : np.ndarray
 Values
 directions : list of float
 Direction angles to plot (degrees)
 tolerance : float
 Angular tolerance (degrees)
 n_lags : int
 Number of lag bins
 figsize : tuple
 Figure size

 Returns
 -------
 fig : matplotlib.Figure
 """
 from ..algorithms.variogram import experimental_variogram_directional

 n_dir = len(directions)
 fig, axes = plt.subplots(2, 2, figsize=figsize)
 axes = axes.flatten()

 for i, direction in enumerate(directions):
 for i, direction in enumerate(directions):

 # Calculate directional variogram
 lags, gamma, n_pairs = experimental_variogram_directional(
 x, y, z,
 angle=direction,
 tolerance=tolerance,
 n_lags=n_lags
 )

 # Plot
 valid = ~np.isnan(gamma)
 if np.any(valid):
 if np.any(valid):
 ax.scatter(lags[valid], gamma[valid], s=sizes, alpha=0.6,
 edgecolors='black', linewidth=1)

 ax.set_xlabel('Distance (h)', fontsize=11)
 ax.set_ylabel('γ(h)', fontsize=11)
 ax.set_title(f'Direction: {direction}° (±{tolerance}°)',
 fontsize=12, fontweight='bold')
 ax.grid(False)

 plt.suptitle('Directional Variograms', fontsize=14, fontweight='bold', y=0.995)
 plt.tight_layout()

 return fig

def plot_variogram_map(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 n_lags: int = 10,
 lag_size: Optional[float] = None,
 figsize: tuple = (10, 8),
    ) -> plt.Figure:
 """
 Create a variogram map (2D variogram surface)

 Shows how semivariance varies with direction and distance.

 Parameters
 ----------
 x, y, z : np.ndarray
 Data coordinates and values
 n_lags : int
 Number of lag bins in each direction
 lag_size : float, optional
 Size of each lag
 figsize : tuple
 Figure size

 Returns
 -------
 fig : matplotlib.Figure
 """
 from ..math.distance import euclidean_distance_matrix

 fig, ax = plt.subplots(figsize=figsize)

 # Calculate all pairwise differences
 dist_matrix = euclidean_distance_matrix(x, y)
 dx = x[:, np.newaxis] - x[np.newaxis, :]
 dy = y[:, np.newaxis] - y[np.newaxis, :]
 dz = z[:, np.newaxis] - z[np.newaxis, :]
 semivar = 0.5 * dz**2

 # Determine lag size
 if lag_size is None:
 if lag_size is None:
 lag_size = max_dist / n_lags

 # Create 2D bins
 max_dx = np.max(np.abs(dx))
 max_dy = np.max(np.abs(dy))
 x_bins = np.linspace(-max_dx, max_dx, 2*n_lags + 1)
 y_bins = np.linspace(-max_dy, max_dy, 2*n_lags + 1)

 # Compute variogram map
 vario_map = np.zeros((2*n_lags, 2*n_lags))
 counts = np.zeros((2*n_lags, 2*n_lags))

 # Upper triangle only
 mask = np.triu(np.ones_like(dx, dtype=bool), k=1)
 dx_flat = dx[mask]
 dy_flat = dy[mask]
 semivar_flat = semivar[mask]

 for i in range(len(dx_flat)):
 for i in range(len(dx_flat)):
 yi = np.digitize(dy_flat[i], y_bins) - 1

 if 0 <= xi < 2*n_lags and 0 <= yi < 2*n_lags:
 if 0 <= xi < 2*n_lags and 0 <= yi < 2*n_lags:
 counts[yi, xi] += 1

 # Average
 with np.errstate(divide='ignore', invalid='ignore'):
 with np.errstate(divide='ignore', invalid='ignore'):

 # Plot
 extent = [-max_dx, max_dx, -max_dy, max_dy]
 im = ax.imshow(vario_map, extent=extent, origin='lower',
 cmap='viridis', interpolation='nearest')

 plt.colorbar(im, ax=ax, label='Semivariance')
 ax.set_xlabel('Δx', fontsize=12)
 ax.set_ylabel('Δy', fontsize=12)
 ax.set_title('Variogram Map', fontsize=14, fontweight='bold')
 ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
 ax.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

 return fig

    # API compatibility functions
def plot_experimental_variogram(
 gamma: npt.NDArray[np.float64],
 n_pairs: Optional[npt.NDArray[np.int64]] = None,
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> tuple:
 """
 Plot experimental variogram points

 Parameters
 ----------
 lags : np.ndarray
 Lag distances
 gamma : np.ndarray
 Experimental semivariance values
 n_pairs : np.ndarray, optional
 Number of pairs in each lag
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional scatter arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 if ax is None:
 else:
 else:

 # Plot experimental points
 if n_pairs is not None:
 if n_pairs is not None:
 ax.scatter(lags, gamma, s=sizes, alpha=0.7, c='blue',
 edgecolors='black', linewidth=1, label='Experimental', **kwargs)
 else:
 else:
 edgecolors='black', linewidth=1, label='Experimental', **kwargs)

 ax.set_xlabel('Distance (h)', fontsize=12)
 ax.set_ylabel('Semivariance γ(h)', fontsize=12)
 ax.set_title('Experimental Variogram', fontsize=14, fontweight='bold')
 ax.grid(False)
 ax.legend(fontsize=10)

 return fig, ax

def plot_variogram_model(
 max_distance: float = 100.0,
 ax: Optional[plt.Axes] = None,
 n_points: int = 200,
 **kwargs,
    ) -> tuple:
 """
 Plot theoretical variogram model

 Parameters
 ----------
 model : VariogramModelBase
 Fitted variogram model
 max_distance : float
 Maximum distance to plot
 ax : matplotlib.Axes, optional
 Axes to plot on
 n_points : int
 Number of points for smooth curve
 **kwargs
 Additional plot arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 if ax is None:
 else:
 else:

 # Generate model curve
 h = np.linspace(0, max_distance, n_points)
 gamma_model = model(h)

 ax.plot(h, gamma_model, 'r-', linewidth=2,
 label=f'{model.__class__.__name__}', **kwargs)

 # Add model parameters
 params = model.parameters
 param_text = '\n'.join([f'{k}: {v:.3f}' for k, v in params.items()])
 ax.text(0.65, 0.05, param_text, transform=ax.transAxes,
 fontsize=9, verticalalignment='bottom',
 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

 ax.set_xlabel('Distance (h)', fontsize=12)
 ax.set_ylabel('Semivariance γ(h)', fontsize=12)
 ax.set_title('Theoretical Variogram Model', fontsize=14, fontweight='bold')
 ax.grid(False)
 ax.legend(fontsize=10)

 return fig, ax

def plot_variogram_with_model(
 gamma: npt.NDArray[np.float64],
 model,
 n_pairs: Optional[npt.NDArray[np.int64]] = None,
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> tuple:
 """
 Plot experimental variogram with fitted model

 Parameters
 ----------
 lags : np.ndarray
 Lag distances
 gamma : np.ndarray
 Experimental semivariance values
 model : VariogramModelBase
 Fitted theoretical model
 n_pairs : np.ndarray, optional
 Number of pairs in each lag
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 if ax is None:
 else:
 else:

 # Plot experimental points
 if n_pairs is not None:
 if n_pairs is not None:
 ax.scatter(lags, gamma, s=sizes, alpha=0.7, c='blue',
 edgecolors='black', linewidth=1, label='Experimental', zorder=5)
 else:
 else:
 edgecolors='black', linewidth=1, label='Experimental', zorder=5)

 # Plot model
 h_plot = np.linspace(0, np.max(lags) * 1.1, 200)
 gamma_model = model(h_plot)
 ax.plot(h_plot, gamma_model, 'r-', linewidth=2,
 label=f'{model.__class__.__name__}', **kwargs)

 # Add model parameters
 params = model.parameters
 param_text = '\n'.join([f'{k}: {v:.3f}' for k, v in params.items()])
 ax.text(0.65, 0.05, param_text, transform=ax.transAxes,
 fontsize=9, verticalalignment='bottom',
 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

 ax.set_xlabel('Distance (h)', fontsize=12)
 ax.set_ylabel('Semivariance γ(h)', fontsize=12)
 ax.set_title('Variogram with Fitted Model', fontsize=14, fontweight='bold')
 ax.grid(False)
 ax.legend(fontsize=10)

 return fig, ax

def plot_variogram_map(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 n_lags: int = 10,
 ax: Optional[plt.Axes] = None,
    ) -> tuple:
 """
 Create variogram map (directional variography)

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 z : np.ndarray
 Values
 n_lags : int
 Number of lag bins in each direction
 ax : matplotlib.Axes, optional
 Axes to plot on

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 if ax is None:
 else:
 else:

 # Compute pairwise differences
 n = len(x)
 dx = x[:, np.newaxis] - x[np.newaxis, :]
 dy = y[:, np.newaxis] - y[np.newaxis, :]
 dz = z[:, np.newaxis] - z[np.newaxis, :]
 semivar = 0.5 * dz**2

 # Mask diagonal
 mask = ~np.eye(n, dtype=bool)
 dx_flat = dx[mask]
 dy_flat = dy[mask]
 semivar_flat = semivar[mask]

 # Create bins
 max_dx = np.max(np.abs(dx_flat))
 max_dy = np.max(np.abs(dy_flat))

 x_bins = np.linspace(-max_dx, max_dx, 2*n_lags+1)
 y_bins = np.linspace(-max_dy, max_dy, 2*n_lags+1)

 # Create map
 vario_map = np.zeros((2*n_lags, 2*n_lags))
 counts = np.zeros((2*n_lags, 2*n_lags))

 for i in range(len(dx_flat)):
 for i in range(len(dx_flat)):
 yi = np.digitize(dy_flat[i], y_bins) - 1

 if 0 <= xi < 2*n_lags and 0 <= yi < 2*n_lags:
 if 0 <= xi < 2*n_lags and 0 <= yi < 2*n_lags:
 counts[yi, xi] += 1

 # Average
 with np.errstate(divide='ignore', invalid='ignore'):
 with np.errstate(divide='ignore', invalid='ignore'):

 # Plot
 extent = [-max_dx, max_dx, -max_dy, max_dy]
 im = ax.imshow(vario_map, extent=extent, origin='lower',
 cmap='viridis', interpolation='nearest')

 plt.colorbar(im, ax=ax, label='Semivariance')
 ax.set_xlabel('Δx', fontsize=12)
 ax.set_ylabel('Δy', fontsize=12)
 ax.set_title('Variogram Map', fontsize=14, fontweight='bold')
 ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)
 ax.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.5)

 return fig, ax
