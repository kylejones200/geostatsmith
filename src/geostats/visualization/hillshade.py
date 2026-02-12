"""
Hillshading and terrain visualization tools.

Provides functions for creating hillshaded relief maps from elevation data,
a common technique in geosciences for visualizing topography.

Reference: Python Recipes for Earth Sciences (Trauth 2024)
"""

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from ..core.logging_config import get_logger
import logging

logger = logging.getLogger(__name__)

logger = get_logger(__name__)

def hillshade(
def hillshade(
 azimuth: float = 315.0,
 altitude: float = 45.0,
 z_factor: float = 1.0,
 dx: float = 1.0,
 dy: float = 1.0,
    ) -> npt.NDArray[np.float64]:
 """
 Calculate hillshade from elevation data.

 Creates a shaded relief representation of terrain by simulating
 illumination from a specified sun position.

 Parameters
 ----------
 elevation : np.ndarray
 2D array of elevation values
 azimuth : float, default=315.0
 Sun azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)
 altitude : float, default=45.0
 Sun altitude angle in degrees (0=horizon, 90=overhead)
 z_factor : float, default=1.0
 Vertical exaggeration factor
 dx, dy : float
 Grid spacing in x and y directions

 Returns
 -------
 hillshade : np.ndarray
 2D array of hillshade values (0-255)

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.visualization.enhanced import hillshade
 >>> import matplotlib.pyplot as plt
 >>>
 >>> # Create simple elevation data
 >>> x = np.linspace(-5, 5, 100)
 >>> y = np.linspace(-5, 5, 100)
 >>> X, Y = np.meshgrid(x, y)
 >>> Z = np.exp(-(X**2 + Y**2)/5) # Gaussian hill
 >>>
 >>> # Calculate hillshade
 >>> hs = hillshade(Z, azimuth=315, altitude=45)
 >>>
 >>> # Plot
 >>> plt.imshow(hs, cmap='gray')
 >>> plt.title('Hillshaded Relief')
 >>> plt.show()

 Notes
 -----
 Hillshading enhances terrain visualization by showing:
 - Slopes facing the sun appear bright
 - Slopes away from the sun appear dark
 - Creates a 3D appearance on 2D maps

 The algorithm calculates:
 1. Slope (rate of change in elevation)
 2. Aspect (direction of slope)
 3. Illumination based on sun position

 References
 ----------
 Horn, B.K.P. (1981). Hill shading and the reflectance map.
 Proceedings of the IEEE, 69(1), 14-47.
 """
 elevation = np.asarray(elevation, dtype=np.float64)

 if elevation.ndim != 2:
 if elevation.ndim != 2:

 # Convert angles to radians
 azimuth_rad = np.deg2rad(azimuth)
 altitude_rad = np.deg2rad(altitude)

 # Calculate gradient (slope components)
 # Using numpy gradient which handles edges properly
 dz_dy, dz_dx = np.gradient(elevation * z_factor, dy, dx)

 # Calculate slope
 slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

 # Calculate aspect (direction of slope)
 aspect = np.arctan2(-dz_dy, dz_dx)

 # Calculate hillshade
 # Formula: cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(sun_azimuth - aspect)
 zenith_rad = np.pi/2 - altitude_rad

 hillshade_value = (
 np.cos(zenith_rad) * np.cos(slope) +
 np.sin(zenith_rad) * np.sin(slope) *
 np.cos(azimuth_rad - np.pi/2 - aspect)
 )

 # Normalize to 0-255
 hillshade_value = np.clip(hillshade_value, 0, 1)
 hillshade_8bit = (hillshade_value * 255).astype(np.uint8)

 return hillshade_8bit

def plot_hillshaded_dem(
def plot_hillshaded_dem(
 y: npt.NDArray[np.float64],
 elevation: npt.NDArray[np.float64],
 azimuth: float = 315.0,
 altitude: float = 45.0,
 cmap: str = 'terrain',
 alpha: float = 0.6,
 figsize: Tuple[int, int] = (12, 10),
    ) -> Tuple[plt.Figure, plt.Axes]:
 """
 Create a hillshaded DEM visualization.

 Combines hillshading with color elevation map for enhanced visualization.

 Parameters
 ----------
 x, y : np.ndarray
 1D arrays of x and y coordinates (for grid)
 elevation : np.ndarray
 2D array of elevation values
 azimuth, altitude : float
 Sun position for hillshading
 cmap : str, default='terrain'
 Colormap for elevation
 alpha : float, default=0.6
 Transparency of color layer (0=transparent, 1=opaque)
 figsize : tuple, default=(12, 10)
 Figure size

 Returns
 -------
 fig, ax : matplotlib Figure and Axes

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.visualization.enhanced import plot_hillshaded_dem
 >>>
 >>> # Create elevation grid
 >>> x = np.linspace(0, 100, 200)
 >>> y = np.linspace(0, 100, 200)
 >>> X, Y = np.meshgrid(x, y)
 >>> Z = 100 * np.exp(-((X-50)**2 + (Y-50)**2)/500)
 >>>
 >>> fig, ax = plot_hillshaded_dem(x, y, Z)
 >>> plt.show()

 Notes
 -----
 This function creates a professional-looking terrain visualization
 by overlaying colored elevation data on top of hillshading.

 The hillshade provides depth perception while the colors show
 actual elevation values.
 """
 # Calculate hillshade
 hs = hillshade(elevation, azimuth=azimuth, altitude=altitude)

 # Create figure
 fig, ax = plt.subplots(figsize=figsize)

 # Create meshgrid if needed
 if x.ndim == 1 and y.ndim == 1:
 if x.ndim == 1 and y.ndim == 1:
 else:
 else:

 # Plot hillshade (grayscale base)
 ax.imshow(hs, extent=[X.min(), X.max(), Y.min(), Y.max()],
 cmap='gray', origin='lower', aspect='auto')

 # Overlay colored elevation with transparency
 im = ax.contourf(X, Y, elevation, levels=20, cmap=cmap, alpha=alpha)

 # Add colorbar
 cbar = plt.colorbar(im, ax=ax, label='Elevation', fraction=0.046, pad=0.04)

 # Labels and title
 ax.set_xlabel('X', fontsize=12)
 ax.set_ylabel('Y', fontsize=12)
 ax.set_title('Hillshaded Digital Elevation Model', fontsize=14, fontweight='bold')
 ax.set_aspect('equal')

 return fig, ax

def create_multi_azimuth_hillshade(
def create_multi_azimuth_hillshade(
 azimuths: Optional[list] = None,
 altitude: float = 45.0,
    ) -> npt.NDArray[np.float64]:
 """
 Create combined hillshade from multiple sun azimuths.

 Averages hillshades from different sun positions to reduce
 directional bias and enhance terrain features.

 Parameters
 ----------
 elevation : np.ndarray
 2D elevation array
 azimuths : list of float, optional
 Sun azimuth angles. If None, uses [315, 45, 135, 225]
 altitude : float
 Sun altitude angle

 Returns
 -------
 hillshade : np.ndarray
 Combined hillshade (0-255)

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.visualization.enhanced import create_multi_azimuth_hillshade
 >>>
 >>> # Create terrain
 >>> x = np.linspace(-10, 10, 200)
 >>> y = np.linspace(-10, 10, 200)
 >>> X, Y = np.meshgrid(x, y)
 >>> Z = np.sin(X) * np.cos(Y) * 50
 >>>
 >>> # Multi-directional hillshade
 >>> hs = create_multi_azimuth_hillshade(Z)

 Notes
 -----
 Using multiple sun positions helps:
 - Reduce directional bias
 - Enhance features regardless of orientation
 - Create more balanced visualization

 Common in professional cartography and GIS.
 """
 if azimuths is None:
 if azimuths is None:

 # Calculate hillshade for each azimuth
 hillshades = []
 for az in azimuths:
 for az in azimuths:
 hillshades.append(hs.astype(float))

 # Average
 combined = np.mean(hillshades, axis=0)

 # Convert back to uint8
 return combined.astype(np.uint8)

def slope_map(
def slope_map(
 dx: float = 1.0,
 dy: float = 1.0,
 units: str = 'degrees',
    ) -> npt.NDArray[np.float64]:
 """
 Calculate slope from elevation data.

 Parameters
 ----------
 elevation : np.ndarray
 2D elevation array
 dx, dy : float
 Grid spacing
 units : str, default='degrees'
 Output units: 'degrees', 'radians', or 'percent'

 Returns
 -------
 slope : np.ndarray
 2D array of slope values

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.visualization.enhanced import slope_map
 >>>
 >>> # Create elevation
 >>> x = np.linspace(0, 100, 100)
 >>> y = np.linspace(0, 100, 100)
 >>> X, Y = np.meshgrid(x, y)
 >>> Z = 0.5 * X + 0.3 * Y # Tilted plane
 >>>
 >>> slope_deg = slope_map(Z, units='degrees')
 >>> logger.info(f"Mean slope: {np.mean(slope_deg):.2f} degrees")

 Notes
 -----
 Slope is the rate of change in elevation - important for:
 - Erosion modeling
 - Landslide hazard assessment
 - Habitat suitability
 - Hydrological modeling
 """
 # Calculate gradients
 dz_dy, dz_dx = np.gradient(elevation, dy, dx)

 # Calculate slope (rise/run)
 slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

 # Unit conversions using dispatch
 unit_conversions = {
 'degrees': lambda s: np.rad2deg(s),
 'radians': lambda s: s,
 'percent': lambda s: np.tan(s) * 100,
 }

 if units not in unit_conversions:
 if units not in unit_conversions:
 raise ValueError(
 f"Unknown units '{units}'. "
 f"Valid units: {valid_units}"
 )

 return unit_conversions[units](slope_rad)

def aspect_map(
def aspect_map(
 dx: float = 1.0,
 dy: float = 1.0,
    ) -> npt.NDArray[np.float64]:
 """
 Calculate aspect (slope direction) from elevation data.

 Parameters
 ----------
 elevation : np.ndarray
 2D elevation array
 dx, dy : float
 Grid spacing

 Returns
 -------
 aspect : np.ndarray
 2D array of aspect values in degrees (0-360)
 0/360 = North, 90 = East, 180 = South, 270 = West

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.visualization.enhanced import aspect_map
 >>>
 >>> # Create elevation
 >>> x = np.linspace(0, 100, 100)
 >>> y = np.linspace(0, 100, 100)
 >>> X, Y = np.meshgrid(x, y)
 >>> Z = X # Slopes to the east
 >>>
 >>> aspect_deg = aspect_map(Z)
 >>> logger.info(f"Mean aspect: {np.mean(aspect_deg):.1f} degrees")

 Notes
 -----
 Aspect indicates the direction a slope faces - important for:
 - Solar radiation modeling
 - Snow accumulation
 - Vegetation patterns
 - Microclimate analysis
 """
 # Calculate gradients
 dz_dy, dz_dx = np.gradient(elevation, dy, dx)

 # Calculate aspect
 # atan2 gives angle from -π to π, we convert to 0-360 degrees
 aspect_rad = np.arctan2(-dz_dy, dz_dx)
 aspect_deg = np.rad2deg(aspect_rad)

 # Convert to 0-360 range (0=North, clockwise)
 aspect_deg = 90 - aspect_deg
 aspect_deg[aspect_deg < 0] += 360

 return aspect_deg
