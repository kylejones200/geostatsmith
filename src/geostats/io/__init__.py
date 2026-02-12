"""
Data I/O Module
================

Read and write spatial data in various formats.

Supported Formats:
- GeoTIFF (raster data, DEMs)
- CSV (tabular spatial data)
- NetCDF (climate, oceanographic data)
- Shapefiles (vector data)
- GeoJSON (vector data)

Examples
--------
>>> from geostats.io import read_geotiff, read_csv_spatial, write_geotiff
>>>
>>> # Read a GeoTIFF DEM
>>> x, y, z, metadata = read_geotiff('elevation.tif')
>>>
>>> # Read CSV with spatial coordinates
>>> data = read_csv_spatial('samples.csv', x_col='lon', y_col='lat', z_col='value')
>>>
>>> # Write predictions to GeoTIFF
>>> write_geotiff('predictions.tif', x_grid, y_grid, z_pred, crs='EPSG:4326')
"""

from .raster import (
    read_geotiff,
    write_geotiff,
    read_ascii_grid,
    write_ascii_grid,
)

from .tabular import (
    read_csv_spatial,
    write_csv_spatial,
    read_excel_spatial,
)

from .formats import (
    read_netcdf,
    write_netcdf,
    read_geojson,
    write_geojson,
    to_dataframe,
    to_geopandas,
)

__all__ = [
    # Raster I/O
    "read_geotiff",
    "write_geotiff",
    "read_ascii_grid",
    "write_ascii_grid",
    # Tabular I/O
    "read_csv_spatial",
    "write_csv_spatial",
    "read_excel_spatial",
    # Other formats
    "read_netcdf",
    "write_netcdf",
    "read_geojson",
    "write_geojson",
    # Conversion utilities
    "to_dataframe",
    "to_geopandas",
]
