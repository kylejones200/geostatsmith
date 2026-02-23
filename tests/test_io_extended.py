"""
Extended tests for I/O operations
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import pandas as pd


class TestRasterIO:
    """Test raster I/O operations"""
    
    def test_read_ascii_grid(self):
        """Test reading ASCII grid"""
        # Create a simple ASCII grid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
            f.write("ncols 3\n")
            f.write("nrows 3\n")
            f.write("xllcorner 0.0\n")
            f.write("yllcorner 0.0\n")
            f.write("cellsize 1.0\n")
            f.write("NODATA_value -9999\n")
            f.write("1.0 2.0 3.0\n")
            f.write("4.0 5.0 6.0\n")
            f.write("7.0 8.0 9.0\n")
            grid_path = f.name
        
        try:
            from geostats.io.raster import read_ascii_grid
            x, y, z, metadata = read_ascii_grid(grid_path, as_grid=True)
            
            assert z.shape == (3, 3)
            assert z[0, 0] == 1.0
            assert z[2, 2] == 9.0
            assert 'ncols' in metadata
            assert 'nrows' in metadata
        finally:
            Path(grid_path).unlink()
    
    def test_write_ascii_grid(self):
        """Test writing ASCII grid"""
        x = np.array([0.5, 1.5, 2.5])
        y = np.array([0.5, 1.5, 2.5])
        z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
            grid_path = f.name
        
        try:
            from geostats.io.raster import write_ascii_grid, read_ascii_grid
            write_ascii_grid(grid_path, x, y, z)
            
            # Read it back
            x_read, y_read, z_read, _ = read_ascii_grid(grid_path, as_grid=True)
            assert np.allclose(z_read, z, atol=0.01)
        finally:
            Path(grid_path).unlink()


class TestTabularIO:
    """Test tabular I/O operations"""
    
    def test_read_csv_spatial(self):
        """Test reading CSV with spatial data"""
        # Create a test CSV
        data = pd.DataFrame({
            'x': [0.0, 10.0, 20.0],
            'y': [0.0, 10.0, 20.0],
            'z': [1.0, 2.0, 1.5],
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            from geostats.io.tabular import read_csv_spatial
            x, y, z = read_csv_spatial(
                csv_path,
                x_column='x',
                y_column='y',
                z_column='z',
            )
            
            assert len(x) == 3
            assert len(y) == 3
            assert len(z) == 3
            assert np.allclose(x, [0.0, 10.0, 20.0])
        finally:
            Path(csv_path).unlink()


class TestFormatConversions:
    """Test format conversion utilities"""
    
    def test_to_dataframe(self):
        """Test converting to DataFrame"""
        from geostats.io.formats import to_dataframe
        
        x = np.array([0.0, 10.0, 20.0])
        y = np.array([0.0, 10.0, 20.0])
        z = np.array([1.0, 2.0, 1.5])
        
        df = to_dataframe(x, y, z)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'z' in df.columns
