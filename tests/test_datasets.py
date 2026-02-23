"""
Tests for dataset loading and generation
"""

import pytest
import numpy as np
from geostats.datasets.synthetic import generate_random_field
from geostats.datasets.walker_lake import load_walker_lake
from geostats.datasets.elevation_samples import load_synthetic_dem_sample


class TestSyntheticData:
    """Test synthetic data generation"""
    
    def test_generate_random_field(self):
        """Test random field generation"""
        x, y, z = generate_random_field(
            n_points=50,
            x_range=(0, 100),
            y_range=(0, 100),
            seed=42,
        )
        
        assert len(x) == 50
        assert len(y) == 50
        assert len(z) == 50
        assert np.all(x >= 0)
        assert np.all(x <= 100)
        assert np.all(y >= 0)
        assert np.all(y <= 100)
    
    def test_generate_with_trend(self):
        """Test random field with trend"""
        x, y, z = generate_random_field(
            n_points=30,
            trend_type='linear',
            seed=42,
        )
        
        assert len(z) == 30
        # Should have some variation
        assert np.std(z) > 0


class TestWalkerLake:
    """Test Walker Lake dataset"""
    
    def test_load_walker_lake(self):
        """Test loading Walker Lake dataset"""
        try:
            data = load_walker_lake()
            assert 'x' in data
            assert 'y' in data
            assert 'z' in data
            assert len(data['x']) > 0
        except FileNotFoundError:
            pytest.skip("Walker Lake dataset not available")


class TestElevationSamples:
    """Test elevation samples dataset"""
    
    def test_load_elevation_samples(self):
        """Test loading elevation samples"""
        data = load_synthetic_dem_sample()
        assert 'x' in data
        assert 'y' in data
        assert 'z' in data
        assert len(data['x']) > 0
        assert 'X_grid' in data
        assert 'Y_grid' in data
        assert 'Z_true' in data
        assert 'metadata' in data
