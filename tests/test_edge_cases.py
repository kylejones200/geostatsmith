"""
Tests for edge cases and error handling
"""

import numpy as np
import pytest

from geostats.core.exceptions import ValidationError
from geostats.core.validators import validate_coordinates, validate_values


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_arrays(self):
        """Test handling of empty arrays"""
        x = np.array([])
        y = np.array([])
        z = np.array([])

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValidationError, ValueError)):
            validate_coordinates(x, y, z)

    def test_single_point(self):
        """Test handling of single point"""
        x = np.array([1.0])
        y = np.array([2.0])
        z = np.array([3.0])

        x_val, y_val, z_val = validate_coordinates(x, y, z)
        assert len(x_val) == 1
        assert len(y_val) == 1
        assert len(z_val) == 1

    def test_all_same_values(self):
        """Test handling of constant values"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([5.0, 5.0, 5.0])  # Constant

        # Should handle constant values
        x_val, y_val, z_val = validate_coordinates(x, y, z)
        assert np.all(z_val == 5.0)

    def test_very_large_values(self):
        """Test handling of very large values"""
        x = np.array([1e10, 2e10, 3e10])
        y = np.array([1e10, 2e10, 3e10])
        z = np.array([1.0, 2.0, 3.0])

        x_val, y_val, z_val = validate_coordinates(x, y, z)
        assert len(x_val) == 3

    def test_very_small_values(self):
        """Test handling of very small values"""
        x = np.array([1e-10, 2e-10, 3e-10])
        y = np.array([1e-10, 2e-10, 3e-10])
        z = np.array([1.0, 2.0, 3.0])

        x_val, y_val, z_val = validate_coordinates(x, y, z)
        assert len(x_val) == 3

    def test_duplicate_locations(self):
        """Test handling of duplicate locations"""
        x = np.array([1.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 2.0])
        z = np.array([1.0, 2.0, 3.0])

        # Should handle duplicates (may warn or handle gracefully)
        x_val, y_val, z_val = validate_coordinates(x, y, z)
        assert len(x_val) == 3

    def test_inf_values(self):
        """Test handling of infinity values"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, np.inf, 3.0])

        with pytest.raises(ValidationError):
            validate_values(z, allow_nan=False)

    def test_mismatched_shapes(self):
        """Test handling of shape mismatches"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])  # Different length

        with pytest.raises(ValidationError):
            validate_coordinates(x, y)
