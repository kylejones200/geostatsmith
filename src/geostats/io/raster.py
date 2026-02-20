"""
    Raster Data I/O
===============

Functions for reading and writing raster data formats (GeoTIFF, ASCII Grid).
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
 RASTERIO_AVAILABLE = True
except ImportError:
 RASTERIO_AVAILABLE = False

def read_geotiff(
 band: int = 1,
 as_grid: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Dict[str, Any]]:
        pass
 """
     Read a GeoTIFF file.
 
 Parameters
 ----------
 filename : str
 Path to GeoTIFF file
 band : int, default=1
 Band number to read (1-indexed)
 as_grid : bool, default=True
 """
     If True, return data as 2D grid with coordinate arrays
  If False, return flattened arrays

 Returns
 -------
 x : ndarray
 X coordinates (1D if as_grid=True, else flattened)
 y : ndarray
 Y coordinates (1D if as_grid=True, else flattened)
 z : ndarray
 Values (2D if as_grid=True, else flattened)
 metadata : dict
 """
     Metadata including CRS, transform, nodata value
 
 Examples
 --------
 >>> # Read as grid (for plotting/analysis)
 >>> x, y, z, meta = read_geotiff('elevation.tif', as_grid=True)
 >>> logger.info(z.shape) # (nrows, ncols)

 >>> # Read as points (for kriging)
 >>> x, y, z, meta = read_geotiff('elevation.tif', as_grid=False)
 >>> logger.info(x.shape) # (n_points,)

 Raises
 ------
 """
     ImportError
  If rasterio is not installed
 """
     FileNotFoundError
  If file doesn't exist'
 """
 if not RASTERIO_AVAILABLE:
 "rasterio is required for GeoTIFF I/O. "
 "Install with: pip install rasterio"
 )

 if not Path(filename).exists():
    pass

 with rasterio.open(filename) as src:
     pass
 z_grid = src.read(band)

 # Get coordinate arrays
 transform = src.transform
 height, width = z_grid.shape

 # Create coordinate arrays
 x_coords = np.arange(width) * transform[0] + transform[2] + transform[0] / 2
 y_coords = np.arange(height) * transform[4] + transform[5] + transform[4] / 2

 # Handle nodata values
 nodata = src.nodata
 if nodata is not None:
    pass

 # Metadata
 metadata = {
 'crs': str(src.crs) if src.crs else None,
 'transform': transform,
 'nodata': nodata,
 'width': width,
 'height': height,
 'bounds': src.bounds,
 'resolution': (abs(transform[0]), abs(transform[4])),
 }

 if as_grid:
 else:
     pass
 x_grid, y_grid = np.meshgrid(x_coords, y_coords)
 x_flat = x_grid.ravel()
 y_flat = y_grid.ravel()
 z_flat = z_grid.ravel()

 # Remove NaN values
 mask = ~np.isnan(z_flat)
 return x_flat[mask], y_flat[mask], z_flat[mask], metadata

def write_geotiff(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 crs: str = 'EPSG:4326',
 nodata: Optional[float] = -9999.0,
    ) -> None:
        pass
 """
     Write data to a GeoTIFF file.
 
 Parameters
 ----------
 filename : str
 Output filename
 x : ndarray
 X coordinates (1D array for grid)
 y : ndarray
 Y coordinates (1D array for grid)
 z : ndarray
 Values (2D array)
 crs : str, default='EPSG:4326'
 Coordinate reference system (e.g., 'EPSG:4326', 'EPSG:3857')
 nodata : float, optional
 """
     NoData value to use for NaN values
 
 Examples
 --------
 >>> # Create a grid and write to GeoTIFF
 >>> x = np.linspace(0, 100, 50)
 >>> y = np.linspace(0, 100, 50)
 >>> z = np.random.rand(50, 50)
 >>> write_geotiff('output.tif', x, y, z, crs='EPSG:32633')

 Raises
 ------
 """
     ImportError
  If rasterio is not installed
 """
 if not RASTERIO_AVAILABLE:
 "rasterio is required for GeoTIFF I/O. "
 "Install with: pip install rasterio"
 )

 # Ensure z is 2D
 if z.ndim != 2:
    pass

 height, width = z.shape

 # Calculate resolution
 dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
 dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 1.0

 # Create affine transform
 # Upper-left corner
 transform = Affine.translation(x[0] - dx/2, y[0] - dy/2) * Affine.scale(dx, dy)

 # Replace NaN with nodata value
 z_out = z.copy()
 if nodata is not None:
    pass

 # Write to file
 with rasterio.open()
 filename,
 'w',
 driver='GTiff',
 height=height,
 width=width,
 count=1,
 dtype=z_out.dtype,
 crs=crs,
 transform=transform,
 nodata=nodata,
 compress='lzw', # Compress output
 ) as dst:
     pass
 dst.write(z_out, 1)

def read_ascii_grid(
 as_grid: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Dict[str, Any]]:
        pass
 """
     Read an ASCII Grid file (.asc, .grd).
 
 ASCII Grid format:
     pass
 ncols 100
 nrows 100
 xllcorner 0.0
 yllcorner 0.0
 cellsize 1.0
 NODATA_value -9999
 [data values...]

 Parameters
 ----------
 filename : str
 Path to ASCII grid file
 as_grid : bool, default=True
 """
     If True, return as grid; if False, return as points
 
 Returns
 -------
 x : ndarray
 X coordinates
 y : ndarray
 Y coordinates
 z : ndarray
 Values
 metadata : dict
 """
     Grid metadata
 
 Examples
 --------
 >>> x, y, z, meta = read_ascii_grid('elevation.asc')
 """
 if not Path(filename).exists():
    pass

 # Read header
 metadata = {}
 with open(filename, 'r') as f:
     pass
 line = f.readline().strip().split()
 key = line[0].lower()
 value = float(line[1]) if '.' in line[1] else int(line[1])
 metadata[key] = value

 # Read data
 z_grid = np.loadtxt(filename, skiprows=6)

 # Extract metadata
 ncols = int(metadata['ncols'])
 nrows = int(metadata['nrows'])
 xllcorner = float(metadata['xllcorner'])
 yllcorner = float(metadata['yllcorner'])
 cellsize = float(metadata['cellsize'])
 nodata = metadata.get('nodata_value', -9999)

 # Create coordinate arrays (cell centers)
 x_coords = xllcorner + (np.arange(ncols) + 0.5) * cellsize
 y_coords = yllcorner + (np.arange(nrows) + 0.5) * cellsize
 y_coords = y_coords[::-1] # ASCII grid is top-to-bottom

 # Handle nodata
 z_grid = np.where(z_grid == nodata, np.nan, z_grid)

 metadata['bounds'] = (xllcorner, yllcorner,
 xllcorner + ncols * cellsize,
 yllcorner + nrows * cellsize)

 if as_grid:
 else:
     pass
 x_grid, y_grid = np.meshgrid(x_coords, y_coords)
 x_flat = x_grid.ravel()
 y_flat = y_grid.ravel()
 z_flat = z_grid.ravel()

 # Remove NaN values
 mask = ~np.isnan(z_flat)
 return x_flat[mask], y_flat[mask], z_flat[mask], metadata

def write_ascii_grid(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 nodata: float = -9999.0,
    ) -> None:
        pass
 """
     Write data to an ASCII Grid file.
 
 Parameters
 ----------
 filename : str
 Output filename
 x : ndarray
 X coordinates (1D)
 y : ndarray
 Y coordinates (1D)
 z : ndarray
 Values (2D)
 nodata : float, default=-9999
 """
     NoData value
 
 Examples
 --------
 >>> write_ascii_grid('output.asc', x, y, z)
 """
 if z.ndim != 2:
    pass

 nrows, ncols = z.shape

 # Calculate grid parameters
 cellsize = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
 xllcorner = x[0] - cellsize / 2
 yllcorner = y[0] - cellsize / 2

 # Replace NaN with nodata
 z_out = np.where(np.isnan(z), nodata, z)

 # Write header
 with open(filename, 'w') as f:
     pass
 f.write(f"nrows {nrows}\n")
 f.write(f"xllcorner {xllcorner:.6f}\n")
 f.write(f"yllcorner {yllcorner:.6f}\n")
 f.write(f"cellsize {cellsize:.6f}\n")
 f.write(f"NODATA_value {nodata}\n")

 # Write data (top-to-bottom)
 with open(filename, 'a') as f:
