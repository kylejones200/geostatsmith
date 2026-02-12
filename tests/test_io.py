"""
Tests for I/O Module

Tests:
- CSV reading/writing
- Excel reading
- GeoTIFF reading/writing (if rasterio available)
- ASCII Grid reading/writing
- NetCDF reading/writing (if netCDF4 available)
- GeoJSON reading/writing (if geopandas available)
- Data conversion utilities
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from geostats.io.tabular import (
    read_csv_spatial,
    write_csv_spatial,
    read_excel_spatial,
)
from geostats.io.formats import (
    to_dataframe,
    to_geopandas,
)

# Check for optional dependencies
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import netCDF4
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Import raster functions if available
if RASTERIO_AVAILABLE:
    from geostats.io.raster import (
        read_geotiff,
        write_geotiff,
        read_ascii_grid,
        write_ascii_grid,
    )

# Import NetCDF functions if available
if NETCDF_AVAILABLE:
    from geostats.io.formats import (
        read_netcdf,
        write_netcdf,
    )

# Import GeoJSON functions if available
if GEOPANDAS_AVAILABLE:
    from geostats.io.formats import (
        read_geojson,
        write_geojson,
    )


class TestCSVIO:
    """Tests for CSV I/O"""

    def setup_method(self):
        """Set up test data and temporary directory"""
        np.random.seed(42)
        self.n = 50
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)
        self.z = 50 + 0.3 * self.x + np.random.normal(0, 3, self.n)
        
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_read_csv_spatial_basic(self):
        """Test basic CSV reading"""
        # Create CSV file
        df = pd.DataFrame({'x': self.x, 'y': self.y, 'z': self.z})
        csv_file = self.temp_dir / 'test.csv'
        df.to_csv(csv_file, index=False)

        # Read it
        x_read, y_read, z_read, extra = read_csv_spatial(str(csv_file))

        np.testing.assert_allclose(x_read, self.x)
        np.testing.assert_allclose(y_read, self.y)
        np.testing.assert_allclose(z_read, self.z)
        assert extra is None

    def test_read_csv_spatial_custom_columns(self):
        """Test CSV reading with custom column names"""
        df = pd.DataFrame({
            'lon': self.x,
            'lat': self.y,
            'value': self.z
        })
        csv_file = self.temp_dir / 'test.csv'
        df.to_csv(csv_file, index=False)

        x_read, y_read, z_read, _ = read_csv_spatial(
            str(csv_file),
            x_col='lon',
            y_col='lat',
            z_col='value'
        )

        np.testing.assert_allclose(x_read, self.x)
        np.testing.assert_allclose(y_read, self.y)
        np.testing.assert_allclose(z_read, self.z)

    def test_read_csv_spatial_with_additional_cols(self):
        """Test CSV reading with additional columns"""
        extra_data = np.random.randn(self.n)
        df = pd.DataFrame({
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'covariate': extra_data
        })
        csv_file = self.temp_dir / 'test.csv'
        df.to_csv(csv_file, index=False)

        x_read, y_read, z_read, extra = read_csv_spatial(
            str(csv_file),
            additional_cols=['covariate']
        )

        assert extra is not None
        assert 'covariate' in extra.columns
        np.testing.assert_allclose(extra['covariate'].values, extra_data)

    def test_read_csv_spatial_handles_nan(self):
        """Test that CSV reading handles NaN values"""
        df = pd.DataFrame({
            'x': self.x,
            'y': self.y,
            'z': self.z
        })
        # Add some NaN values
        df.loc[0, 'z'] = np.nan
        df.loc[5, 'x'] = np.nan

        csv_file = self.temp_dir / 'test.csv'
        df.to_csv(csv_file, index=False)

        x_read, y_read, z_read, _ = read_csv_spatial(str(csv_file))

        # Should have fewer points (NaN rows removed)
        assert len(x_read) < self.n
        assert len(x_read) == len(y_read) == len(z_read)
        assert not np.any(np.isnan(x_read))
        assert not np.any(np.isnan(y_read))
        assert not np.any(np.isnan(z_read))

    def test_read_csv_spatial_file_not_found(self):
        """Test that missing file raises error"""
        with pytest.raises(FileNotFoundError):
            read_csv_spatial('nonexistent_file.csv')

    def test_read_csv_spatial_missing_columns(self):
        """Test that missing columns raise error"""
        df = pd.DataFrame({'x': self.x, 'y': self.y})  # Missing 'z'
        csv_file = self.temp_dir / 'test.csv'
        df.to_csv(csv_file, index=False)

        with pytest.raises(KeyError, match="Missing columns"):
            read_csv_spatial(str(csv_file))

    def test_write_csv_spatial_basic(self):
        """Test basic CSV writing"""
        csv_file = self.temp_dir / 'output.csv'
        
        write_csv_spatial(
            str(csv_file),
            self.x, self.y, self.z
        )

        assert csv_file.exists()
        
        # Read it back
        x_read, y_read, z_read, _ = read_csv_spatial(str(csv_file))
        np.testing.assert_allclose(x_read, self.x)
        np.testing.assert_allclose(y_read, self.y)
        np.testing.assert_allclose(z_read, self.z)

    def test_write_csv_spatial_custom_columns(self):
        """Test CSV writing with custom column names"""
        csv_file = self.temp_dir / 'output.csv'
        
        write_csv_spatial(
            str(csv_file),
            self.x, self.y, self.z,
            x_col='longitude',
            y_col='latitude',
            z_col='temperature'
        )

        # Read back and verify column names
        df = pd.read_csv(csv_file)
        assert 'longitude' in df.columns
        assert 'latitude' in df.columns
        assert 'temperature' in df.columns

    def test_write_csv_spatial_with_extra(self):
        """Test CSV writing with extra columns"""
        csv_file = self.temp_dir / 'output.csv'
        extra = pd.DataFrame({
            'variance': np.random.rand(self.n),
            'std': np.random.rand(self.n)
        })
        
        write_csv_spatial(
            str(csv_file),
            self.x, self.y, self.z,
            extra=extra
        )

        # Read back
        df = pd.read_csv(csv_file)
        assert 'variance' in df.columns
        assert 'std' in df.columns
        assert len(df) == self.n

    def test_read_excel_spatial(self):
        """Test Excel reading (if openpyxl available)"""
        try:
            excel_file = self.temp_dir / 'test.xlsx'
            df = pd.DataFrame({
                'x': self.x,
                'y': self.y,
                'z': self.z
            })
            df.to_excel(excel_file, index=False)

            x_read, y_read, z_read, _ = read_excel_spatial(str(excel_file))

            np.testing.assert_allclose(x_read, self.x)
            np.testing.assert_allclose(y_read, self.y)
            np.testing.assert_allclose(z_read, self.z)
        except ImportError:
            pytest.skip("openpyxl not available")


class TestRasterIO:
    """Tests for Raster I/O (requires rasterio)"""

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test grid
        self.x_grid = np.linspace(0, 100, 50)
        self.y_grid = np.linspace(0, 100, 50)
        x_2d, y_2d = np.meshgrid(self.x_grid, self.y_grid)
        self.z_grid = 50 + 0.1 * x_2d + 0.05 * y_2d + np.random.randn(50, 50) * 2

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def teardown_method(self):
        """Clean up"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_write_read_geotiff(self):
        """Test writing and reading GeoTIFF"""
        tif_file = self.temp_dir / 'test.tif'
        
        # Write
        write_geotiff(
            str(tif_file),
            self.x_grid, self.y_grid, self.z_grid,
            crs='EPSG:4326'
        )

        assert tif_file.exists()

        # Read back
        x_read, y_read, z_read, meta = read_geotiff(str(tif_file), as_grid=True)

        # Should be similar (within tolerance for coordinate rounding)
        np.testing.assert_allclose(z_read, self.z_grid, rtol=1e-5)
        assert meta['crs'] is not None

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_read_geotiff_as_points(self):
        """Test reading GeoTIFF as point arrays"""
        tif_file = self.temp_dir / 'test.tif'
        write_geotiff(str(tif_file), self.x_grid, self.y_grid, self.z_grid)

        x_read, y_read, z_read, meta = read_geotiff(str(tif_file), as_grid=False)

        assert x_read.ndim == 1
        assert y_read.ndim == 1
        assert z_read.ndim == 1
        assert len(x_read) == len(y_read) == len(z_read)

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_write_geotiff_handles_nan(self):
        """Test that GeoTIFF writing handles NaN values"""
        z_with_nan = self.z_grid.copy()
        z_with_nan[10:15, 20:25] = np.nan
        
        tif_file = self.temp_dir / 'test_nan.tif'
        write_geotiff(
            str(tif_file),
            self.x_grid, self.y_grid, z_with_nan,
            nodata=-9999.0
        )

        # Should write without error
        assert tif_file.exists()

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_read_geotiff_file_not_found(self):
        """Test that missing GeoTIFF raises error"""
        with pytest.raises(FileNotFoundError):
            read_geotiff('nonexistent.tif')

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_write_ascii_grid(self):
        """Test writing ASCII Grid"""
        asc_file = self.temp_dir / 'test.asc'
        
        write_ascii_grid(
            str(asc_file),
            self.x_grid, self.y_grid, self.z_grid
        )

        assert asc_file.exists()

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_read_ascii_grid(self):
        """Test reading ASCII Grid"""
        asc_file = self.temp_dir / 'test.asc'
        write_ascii_grid(str(asc_file), self.x_grid, self.y_grid, self.z_grid)

        x_read, y_read, z_read, meta = read_ascii_grid(str(asc_file), as_grid=True)

        assert z_read.shape == self.z_grid.shape
        assert 'ncols' in meta
        assert 'nrows' in meta

    @pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not available")
    def test_read_ascii_grid_as_points(self):
        """Test reading ASCII Grid as points"""
        asc_file = self.temp_dir / 'test.asc'
        write_ascii_grid(str(asc_file), self.x_grid, self.y_grid, self.z_grid)

        x_read, y_read, z_read, meta = read_ascii_grid(str(asc_file), as_grid=False)

        assert x_read.ndim == 1
        assert y_read.ndim == 1
        assert z_read.ndim == 1


class TestNetCDFIO:
    """Tests for NetCDF I/O (requires netCDF4)"""

    @pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.x = np.linspace(0, 100, 50)
        self.y = np.linspace(0, 100, 50)
        x_2d, y_2d = np.meshgrid(self.x, self.y)
        self.z = 50 + 0.1 * x_2d + 0.05 * y_2d + np.random.randn(50, 50) * 2

    @pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
    def teardown_method(self):
        """Clean up"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
    def test_write_read_netcdf(self):
        """Test writing and reading NetCDF"""
        nc_file = self.temp_dir / 'test.nc'
        
        # Write
        write_netcdf(
            str(nc_file),
            self.x, self.y, self.z,
            z_var='temperature',
            x_var='longitude',
            y_var='latitude'
        )

        assert nc_file.exists()

        # Read back
        x_read, y_read, z_read, meta = read_netcdf(
            str(nc_file),
            z_var='temperature',
            x_var='longitude',
            y_var='latitude'
        )

        np.testing.assert_allclose(z_read, self.z, rtol=1e-5)

    @pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
    def test_read_netcdf_file_not_found(self):
        """Test that missing NetCDF raises error"""
        with pytest.raises(FileNotFoundError):
            read_netcdf('nonexistent.nc', z_var='temp')

    @pytest.mark.skipif(not NETCDF_AVAILABLE, reason="netCDF4 not available")
    def test_read_netcdf_missing_variable(self):
        """Test that missing variable raises error"""
        nc_file = self.temp_dir / 'test.nc'
        write_netcdf(str(nc_file), self.x, self.y, self.z, z_var='temp')

        with pytest.raises(KeyError, match="Variable"):
            read_netcdf(str(nc_file), z_var='nonexistent_var')


class TestDataConversion:
    """Tests for data conversion utilities"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n = 50
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)
        self.z = 50 + 0.3 * self.x + np.random.normal(0, 3, self.n)

    def test_to_dataframe(self):
        """Test conversion to DataFrame"""
        df = to_dataframe(self.x, self.y, self.z)

        assert isinstance(df, pd.DataFrame)
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'z' in df.columns
        assert len(df) == self.n

    def test_to_dataframe_custom_columns(self):
        """Test conversion with custom column names"""
        df = to_dataframe(
            self.x, self.y, self.z,
            x_col='lon',
            y_col='lat',
            z_col='value'
        )

        assert 'lon' in df.columns
        assert 'lat' in df.columns
        assert 'value' in df.columns

    @pytest.mark.skipif(not GEOPANDAS_AVAILABLE, reason="geopandas not available")
    def test_to_geopandas(self):
        """Test conversion to GeoDataFrame"""
        gdf = to_geopandas(self.x, self.y, self.z)

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert 'geometry' in gdf.columns
        assert len(gdf) == self.n

    @pytest.mark.skipif(not GEOPANDAS_AVAILABLE, reason="geopandas not available")
    def test_read_write_geojson(self):
        """Test reading and writing GeoJSON"""
        temp_dir = Path(tempfile.mkdtemp())
        geojson_file = temp_dir / 'test.geojson'
        
        try:
            # Write
            write_geojson(
                str(geojson_file),
                self.x, self.y, self.z
            )

            assert geojson_file.exists()

            # Read back
            x_read, y_read, z_read, meta = read_geojson(str(geojson_file))

            np.testing.assert_allclose(x_read, self.x)
            np.testing.assert_allclose(y_read, self.y)
            np.testing.assert_allclose(z_read, self.z)
        finally:
            shutil.rmtree(temp_dir)
