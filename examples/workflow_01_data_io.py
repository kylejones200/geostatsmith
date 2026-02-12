"""
Example Workflow: Data I/O and Format Conversion
=================================================

Demonstrates how to:
1. Read data from various formats (GeoTIFF, CSV, NetCDF)
2. Perform kriging interpolation
3. Export results in multiple formats
4. Convert between formats

Author: geostats development team
Date: January 2026
"""

import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
 level=logging.INFO,
 format='%(message)s'
)
logger = logging.getLogger(__name__)

try:
try:
 read_csv_spatial, write_csv_spatial,
 to_dataframe, to_geopandas
 )
 from geostats.algorithms.ordinary_kriging import OrdinaryKriging
 from geostats.models.variogram_models import SphericalModel
 from geostats.algorithms.variogram import experimental_variogram
 from geostats.algorithms.fitting import fit_variogram_model as fit_variogram
except ImportError:
 logger.error("Please install geostats: pip install -e .")
 exit(1)

def example_1_read_csv_and_interpolate():
def example_1_read_csv_and_interpolate():
 
 logger.info("Example 1: CSV → Kriging → GeoTIFF")
 

 logger.info("Creating sample elevation data...")
 np.random.seed(42)
 n_samples = 50
 x = np.random.uniform(0, 100, n_samples)
 y = np.random.uniform(0, 100, n_samples)
 z = 50 + 0.3*x + 0.2*y + 10*np.sin(x/20) + np.random.normal(0, 2, n_samples)

 write_csv_spatial(
 'sample_elevation.csv',
 x, y, z,
 x_col='easting',
 y_col='northing',
 z_col='elevation'
 )
 logger.info(" Saved sample_elevation.csv")

 x_read, y_read, z_read, _ = read_csv_spatial(
 'sample_elevation.csv',
 x_col='easting',
 y_col='northing',
 z_col='elevation'
 )
 logger.info(f" Read {len(x_read)} samples from CSV")

 logger.info("\nFitting variogram...")
 lags, gamma, n_pairs = experimental_variogram(x_read, y_read, z_read)
 variogram_model = SphericalModel()
 variogram_model = fit_variogram(variogram_model, lags, gamma)
 logger.info(f" Nugget: {variogram_model._parameters['nugget']:.2f}")
 logger.info(f" Sill: {variogram_model._parameters['sill']:.2f}")
 logger.info(f" Range: {variogram_model._parameters['range']:.2f}")

 logger.info("\nPerforming kriging interpolation...")
 nx, ny = 100, 100
 x_grid = np.linspace(0, 100, nx)
 y_grid = np.linspace(0, 100, ny)
 x_pred_2d, y_pred_2d = np.meshgrid(x_grid, y_grid)
 x_pred = x_pred_2d.ravel()
 y_pred = y_pred_2d.ravel()

 krig = OrdinaryKriging(
 x=x_read,
 y=y_read,
 z=z_read,
 variogram_model=variogram_model,
 )
 z_pred, var_pred = krig.predict(x_pred, y_pred, return_variance=True)
 z_grid = z_pred.reshape((ny, nx))
 logger.info(f" Interpolated to {nx}x{ny} grid")

 write_geotiff(
 'elevation_kriging.tif',
 x_grid, y_grid, z_grid,
 crs='EPSG:32633',
 )
 logger.info(" Saved elevation_kriging.tif")

 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

 scatter = ax1.scatter(x_read, y_read, c=z_read, cmap='terrain', s=50, edgecolor='k')
 ax1.set_xlabel('Easting (m)')
 ax1.set_ylabel('Northing (m)')
 ax1.set_title('Original Sample Points')
 ax1.set_aspect('equal')
 plt.colorbar(scatter, ax=ax1, label='Elevation (m)')

 im = ax2.contourf(x_grid, y_grid, z_grid, levels=15, cmap='terrain')
 ax2.scatter(x_read, y_read, c='k', s=10, alpha=0.5, label='Samples')
 ax2.set_xlabel('Easting (m)')
 ax2.set_ylabel('Northing (m)')
 ax2.set_title('Kriging Interpolation')
 ax2.set_aspect('equal')
 ax2.legend()
 plt.colorbar(im, ax=ax2, label='Elevation (m)')

 plt.tight_layout()
 plt.savefig('example_workflow_01_io.png', dpi=150, bbox_inches='tight')
 logger.info(" Saved example_workflow_01_io.png")
 plt.close()

def example_2_geotiff_workflow():
def example_2_geotiff_workflow():
 
 logger.info("Example 2: GeoTIFF Validation Workflow")
 

 try:
 try:
 x_grid, y_grid, z_grid, metadata = read_geotiff(
 'elevation_kriging.tif',
 as_grid=True
 )
 logger.info(f" Read GeoTIFF: {z_grid.shape[1]}x{z_grid.shape[0]} grid")
 logger.info(f" CRS: {metadata['crs']}")
 logger.info(f" Resolution: {metadata['resolution']}")

 logger.info("\nExtracting validation points...")
 n_valid = 20
 i_indices = np.random.randint(0, len(y_grid), n_valid)
 j_indices = np.random.randint(0, len(x_grid), n_valid)

 x_valid = x_grid[j_indices]
 y_valid = y_grid[i_indices]
 z_valid = z_grid[i_indices, j_indices]

 write_csv_spatial(
 'validation_points.csv',
 x_valid, y_valid, z_valid,
 x_col='x', y_col='y', z_col='elevation'
 )
 logger.info(f" Saved {n_valid} validation points to CSV")

 df = to_dataframe(x_valid, y_valid, z_valid,
 x_col='easting', y_col='northing', z_col='elevation')
 logger.info(f"Converted to DataFrame:")
 logger.info(f"\n{df.head()}")

 try:
 try:
 logger.info(f"Converted to GeoDataFrame with CRS: {gdf.crs}")
 gdf.to_file('validation_points.geojson', driver='GeoJSON')
 logger.info(" Saved validation_points.geojson")
 except ImportError:
 logger.warning("geopandas not available, skipping GeoDataFrame conversion")

 except FileNotFoundError:
 logger.warning(" elevation_kriging.tif not found. Run Example 1 first.")

def example_3_format_comparison():
def example_3_format_comparison():
 
 logger.info("Example 3: Format Comparison")
 

 n = 100
 x = np.linspace(0, 100, n)
 y = np.linspace(0, 100, n)
 z = np.random.rand(n)

 formats = []

 start = time.time()
 write_csv_spatial('test.csv', x, y, z)
 formats.append(('CSV', time.time() - start, os.path.getsize('test.csv') / 1024))

 x_grid, y_grid = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 100, 50))
 z_grid = np.random.rand(50, 50)
 try:
 try:
 write_geotiff('test.tif', np.linspace(0, 100, 50), np.linspace(0, 100, 50), z_grid)
 formats.append(('GeoTIFF', time.time() - start, os.path.getsize('test.tif') / 1024))
 except ImportError:
 logger.warning(" rasterio not available for GeoTIFF")

 logger.info("\nFormat Performance:")
 logger.info(f"{'Format':<15} {'Write Time (s)':<15} {'File Size (KB)':<15}")
 logger.info("-" * 45)
 for fmt, t, s in formats:
 for fmt, t, s in formats:

def main():
def main():
 logger.info("GEOSTATS DATA I/O WORKFLOW EXAMPLES")

 example_1_read_csv_and_interpolate()
 example_2_geotiff_workflow()
 example_3_format_comparison()

 logger.info("ALL EXAMPLES COMPLETE!")
 logger.info("\nFiles created:")
 logger.info(" - sample_elevation.csv")
 logger.info(" - elevation_kriging.tif")
 logger.info(" - validation_points.csv")
 logger.info(" - validation_points.geojson (if geopandas available)")
 logger.info(" - example_workflow_01_io.png")

if __name__ == '__main__':
if __name__ == '__main__':
