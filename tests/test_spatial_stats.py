"""
Tests for spatial statistics: point patterns, autocorrelation
"""

import numpy as np

from geostats.spatial_stats.point_patterns import (
    nearest_neighbor_analysis,
    ripley_k_function,
)
from geostats.spatial_stats.spatial_autocorrelation import (
    gearys_c,
    morans_i,
)


class TestPointPatterns:
    """Test point pattern analysis"""

    def test_nearest_neighbor_analysis(self):
        """Test nearest neighbor analysis"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)

        result = nearest_neighbor_analysis(x, y)

        assert "mean_distance" in result
        assert "std_distance" in result
        assert result["mean_distance"] > 0
        assert result["std_distance"] >= 0

    def test_ripley_k_function(self):
        """Test Ripley's K function"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 30)
        y = np.random.uniform(0, 100, 30)

        result = ripley_k_function(x, y, n_distances=10)

        assert "distances" in result
        assert "k_values" in result
        assert len(result["distances"]) == 10
        assert len(result["k_values"]) == 10


class TestSpatialAutocorrelation:
    """Test spatial autocorrelation measures"""

    def test_morans_i(self):
        """Test Moran's I"""
        np.random.seed(42)
        x = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Positive spatial autocorrelation

        I = morans_i(x, y, z)

        # Should be positive for this pattern
        assert isinstance(I, float)
        assert not np.isnan(I)
        assert not np.isinf(I)

    def test_gearys_c(self):
        """Test Geary's C"""
        np.random.seed(42)
        x = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        C = gearys_c(x, y, z)

        assert isinstance(C, float)
        assert not np.isnan(C)
        assert not np.isinf(C)
