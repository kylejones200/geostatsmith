"""
Tests for math utilities: distance, matrices, numerical
"""

import pytest
import numpy as np
from geostats.math.distance import (
    euclidean_distance,
    euclidean_distance_matrix,
    manhattan_distance,
)
from geostats.math.matrices import (
    solve_kriging_system,
    regularize_matrix,
)
from geostats.math.numerical import (
    weighted_least_squares,
    ordinary_least_squares,
)


class TestDistance:
    """Test distance calculations"""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation"""
        x1, y1 = 0.0, 0.0
        x2, y2 = 3.0, 4.0
        dist = euclidean_distance(x1, y1, x2, y2)
        assert abs(dist - 5.0) < 1e-10
    
    def test_euclidean_distance_matrix(self):
        """Test Euclidean distance matrix"""
        x = np.array([0.0, 3.0, 0.0])
        y = np.array([0.0, 0.0, 4.0])
        dist_matrix = euclidean_distance_matrix(x, y)
        
        assert dist_matrix.shape == (3, 3)
        assert dist_matrix[0, 0] == 0.0
        assert abs(dist_matrix[0, 1] - 3.0) < 1e-10
        assert abs(dist_matrix[0, 2] - 4.0) < 1e-10
        # Symmetric
        assert np.allclose(dist_matrix, dist_matrix.T)
    
    def test_manhattan_distance(self):
        """Test Manhattan distance"""
        x1, y1 = 0.0, 0.0
        x2, y2 = 3.0, 4.0
        dist = manhattan_distance(x1, y1, x2, y2)
        # Returns array, extract scalar value
        assert abs(float(dist) - 7.0) < 1e-10


class TestMatrices:
    """Test matrix operations"""
    
    def test_regularize_matrix(self):
        """Test matrix regularization"""
        # Create a singular matrix
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        A_reg = regularize_matrix(A, epsilon=1e-8)
        
        # Should be invertible now
        assert np.linalg.cond(A_reg) < 1e10
    
    def test_solve_kriging_system(self):
        """Test kriging system solver"""
        # Simple 2x2 system: Ax = b
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([5.0, 4.0])
        
        x = solve_kriging_system(A, b)
        # Solution: x = [2, 1]
        assert np.allclose(x, [2.0, 1.0], atol=1e-10)


class TestNumerical:
    """Test numerical methods"""
    
    def test_ordinary_least_squares(self):
        """Test OLS fitting"""
        # Simple linear fit: y = 2x + 1
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0 + np.random.randn(5) * 0.01
        
        # Fit with OLS
        X = np.column_stack([np.ones_like(x), x])
        params, _ = ordinary_least_squares(X, y)
        
        # Should recover approximately [1, 2]
        assert abs(params[0] - 1.0) < 0.1
        assert abs(params[1] - 2.0) < 0.1
    
    def test_weighted_least_squares(self):
        """Test WLS fitting"""
        # Simple linear fit with weights
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0 + np.random.randn(5) * 0.01
        weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0])  # Higher weight on middle point
        
        X = np.column_stack([np.ones_like(x), x])
        params, _ = weighted_least_squares(X, y, weights=weights)
        
        # Should recover approximately [1, 2]
        assert abs(params[0] - 1.0) < 0.1
        assert abs(params[1] - 2.0) < 0.1
