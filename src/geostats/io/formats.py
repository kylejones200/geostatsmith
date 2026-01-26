"""
Other Format I/O
================

Functions for reading/writing NetCDF, GeoJSON, and conversion utilities.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, Dict, Optional, Any, List
from pathlib import Path
import json

try:
    import netCDF4 as nc
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


def read_netcdf(
    filename: str,
    z_var: str,
    x_var: str = 'x',
    y_var: str = 'y',
    time_index: Optional[int] = None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Dict[str, Any]]:
    """
    Read spatial data from NetCDF file.
    
    Parameters
    ----------
    filename : str
        Path to NetCDF file
    x_var : str, default='x'
        Name of X coordinate variable (e.g., 'lon', 'longitude')
    y_var : str, default='y'
        Name of Y coordinate variable (e.g., 'lat', 'latitude')
    z_var : str
        Name of data variable (e.g., 'temperature', 'precipitation')
    time_index : int, optional
        If data has time dimension, specify which time slice to read
    
    Returns
    -------
    x : ndarray
        X coordinates (1D)
    y : ndarray
        Y coordinates (1D)
    z : ndarray
        Values (2D or flattened)
    metadata : dict
        Variable metadata and attributes
    
    Examples
    --------
    >>> # Read temperature data
    >>> x, y, temp, meta = read_netcdf(
    ...     'climate.nc',
    ...     x_var='longitude',
    ...     y_var='latitude',
    ...     z_var='temperature',
    ...     time_index=0
    ... )
    
    Raises
    ------
    ImportError
        If netCDF4 is not installed
    FileNotFoundError
        If file doesn't exist
    """
    if not NETCDF_AVAILABLE:
        raise ImportError(
            "netCDF4 is required for NetCDF I/O. "
            "Install with: pip install netCDF4"
        )
    
    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Open NetCDF file
    with nc.Dataset(filename, 'r') as dataset:
        # Check variables exist
        if x_var not in dataset.variables:
            raise KeyError(f"Variable '{x_var}' not found. Available: {list(dataset.variables.keys())}")
        if y_var not in dataset.variables:
            raise KeyError(f"Variable '{y_var}' not found")
        if z_var not in dataset.variables:
            raise KeyError(f"Variable '{z_var}' not found")
        
        # Read coordinates
        x = dataset.variables[x_var][:].data
        y = dataset.variables[y_var][:].data
        
        # Read data
        z_data = dataset.variables[z_var]
        
        # Handle time dimension if present
        if time_index is not None and 'time' in z_data.dimensions:
            z = z_data[time_index, :, :].data
        elif len(z_data.shape) == 3:
            # If 3D but no time_index specified, take first slice
            z = z_data[0, :, :].data
        else:
            z = z_data[:].data
        
        # Handle masked arrays
        if hasattr(z, 'mask'):
            z = np.ma.filled(z, np.nan)
        
        # Collect metadata
        metadata = {
            'dimensions': dict(dataset.dimensions),
            'attributes': {attr: dataset.getncattr(attr) for attr in dataset.ncattrs()},
            'z_attributes': {attr: z_data.getncattr(attr) for attr in z_data.ncattrs()},
        }
    
    return x, y, z, metadata


def write_netcdf(
    filename: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    z_varname: str = 'data',
    x_varname: str = 'x',
    y_varname: str = 'y',
    units: Optional[str] = None,
    long_name: Optional[str] = None,
) -> None:
    """
    Write spatial data to NetCDF file.
    
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
    z_varname : str, default='data'
        Name for data variable
    x_varname : str, default='x'
        Name for X coordinate variable
    y_varname : str, default='y'
        Name for Y coordinate variable
    units : str, optional
        Units for data variable
    long_name : str, optional
        Long descriptive name for data variable
    
    Examples
    --------
    >>> write_netcdf(
    ...     'output.nc',
    ...     x, y, z,
    ...     z_varname='temperature',
    ...     units='degrees_C',
    ...     long_name='Surface Temperature'
    ... )
    
    Raises
    ------
    ImportError
        If netCDF4 is not installed
    """
    if not NETCDF_AVAILABLE:
        raise ImportError(
            "netCDF4 is required for NetCDF I/O. "
            "Install with: pip install netCDF4"
        )
    
    if z.ndim != 2:
        raise ValueError("z must be 2D array")
    
    # Create NetCDF file
    with nc.Dataset(filename, 'w', format='NETCDF4') as dataset:
        # Create dimensions
        dataset.createDimension(y_varname, len(y))
        dataset.createDimension(x_varname, len(x))
        
        # Create coordinate variables
        x_var = dataset.createVariable(x_varname, 'f8', (x_varname,))
        y_var = dataset.createVariable(y_varname, 'f8', (y_varname,))
        z_var = dataset.createVariable(z_varname, 'f8', (y_varname, x_varname))
        
        # Write data
        x_var[:] = x
        y_var[:] = y
        z_var[:] = z
        
        # Add attributes
        if units:
            z_var.units = units
        if long_name:
            z_var.long_name = long_name
        
        # Global attributes
        dataset.description = 'Geostatistics output'
        dataset.source = 'geostats library'


def read_geojson(
    filename: str,
    z_property: str,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Read point data from GeoJSON file.
    
    Parameters
    ----------
    filename : str
        Path to GeoJSON file
    z_property : str
        Name of property to use as z values
    
    Returns
    -------
    x : ndarray
        X coordinates (longitude)
    y : ndarray
        Y coordinates (latitude)
    z : ndarray
        Values from specified property
    
    Examples
    --------
    >>> x, y, z = read_geojson('samples.geojson', z_property='elevation')
    
    Raises
    ------
    ImportError
        If geopandas is not installed
    FileNotFoundError
        If file doesn't exist
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError(
            "geopandas is required for GeoJSON I/O. "
            "Install with: pip install geopandas"
        )
    
    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Read GeoJSON
    gdf = gpd.read_file(filename)
    
    # Check property exists
    if z_property not in gdf.columns:
        raise KeyError(f"Property '{z_property}' not found. Available: {list(gdf.columns)}")
    
    # Extract coordinates
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    z = gdf[z_property].values
    
    return x, y, z


def write_geojson(
    filename: str,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    z_property: str = 'value',
    crs: str = 'EPSG:4326',
) -> None:
    """
    Write point data to GeoJSON file.
    
    Parameters
    ----------
    filename : str
        Output filename
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    z : ndarray
        Values
    z_property : str, default='value'
        Name for value property
    crs : str, default='EPSG:4326'
        Coordinate reference system
    
    Examples
    --------
    >>> write_geojson('output.geojson', x, y, z, z_property='elevation')
    
    Raises
    ------
    ImportError
        If geopandas is not installed
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError(
            "geopandas is required for GeoJSON I/O. "
            "Install with: pip install geopandas"
        )
    
    # Create GeoDataFrame
    geometry = [Point(xi, yi) for xi, yi in zip(x, y)]
    gdf = gpd.GeoDataFrame(
        {z_property: z},
        geometry=geometry,
        crs=crs
    )
    
    # Write to file
    gdf.to_file(filename, driver='GeoJSON')


def to_dataframe(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    x_col: str = 'x',
    y_col: str = 'y',
    z_col: str = 'z',
    **extra_cols,
) -> pd.DataFrame:
    """
    Convert spatial data to pandas DataFrame.
    
    Parameters
    ----------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    z : ndarray
        Values
    x_col : str, default='x'
        Column name for X
    y_col : str, default='y'
        Column name for Y
    z_col : str, default='z'
        Column name for values
    **extra_cols
        Additional columns as keyword arguments
    
    Returns
    -------
    df : DataFrame
        Spatial data as DataFrame
    
    Examples
    --------
    >>> df = to_dataframe(x, y, z)
    >>> df = to_dataframe(x, y, z, variance=var, std=std)
    """
    df = pd.DataFrame({
        x_col: x,
        y_col: y,
        z_col: z,
    })
    
    # Add extra columns
    for col_name, col_data in extra_cols.items():
        df[col_name] = col_data
    
    return df


def to_geopandas(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    z_col: str = 'value',
    crs: str = 'EPSG:4326',
    **extra_cols,
) -> 'gpd.GeoDataFrame':
    """
    Convert spatial data to GeoPandas GeoDataFrame.
    
    Parameters
    ----------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    z : ndarray
        Values
    z_col : str, default='value'
        Column name for values
    crs : str, default='EPSG:4326'
        Coordinate reference system
    **extra_cols
        Additional columns as keyword arguments
    
    Returns
    -------
    gdf : GeoDataFrame
        Spatial data as GeoDataFrame
    
    Examples
    --------
    >>> gdf = to_geopandas(x, y, z, crs='EPSG:32633')
    >>> gdf = to_geopandas(x, y, z, variance=var, std=std)
    
    Raises
    ------
    ImportError
        If geopandas is not installed
    """
    if not GEOPANDAS_AVAILABLE:
        raise ImportError(
            "geopandas is required. "
            "Install with: pip install geopandas"
        )
    
    # Create geometry
    geometry = [Point(xi, yi) for xi, yi in zip(x, y)]
    
    # Create data dictionary
    data = {z_col: z}
    data.update(extra_cols)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    
    return gdf
