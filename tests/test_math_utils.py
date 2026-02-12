"""
Tests for math utilities and helper functions

Tests:
- Distance calculations
- Matrix operations
- Numerical methods
- Grid utilities
- Data utilities
"""

import pytest
import numpy as np
from geostats.math import distance, matrices, numerical
from geostats.utils import data_utils, grid_utils

class TestDistanceCalculations:

    def test_euclidean_distance_2d(self):
        x1 = np.array([0, 1, 2])
        y1 = np.array([0, 0, 0])
        x2 = np.array([0, 0, 0])
        y2 = np.array([0, 1, 2])

        dist = distance.euclidean_distance(x1, y1, x2, y2)

        expected = np.array([
        [0.0, 1.0, 2.0],
        [1.0, np.sqrt(2), np.sqrt(5)],
        [2.0, np.sqrt(5), np.sqrt(8)]
        ])

        np.testing.assert_array_almost_equal(dist, expected)

    def test_euclidean_distance_single_point(self):
        x1 = np.array([0])
        y1 = np.array([0])
        x2 = np.array([3])
        y2 = np.array([4])

        dist = distance.euclidean_distance(x1, y1, x2, y2)

        assert dist.shape == (1, 1)
        assert abs(dist[0, 0] - 5.0) < 1e-10

    def test_euclidean_distance_symmetry(self):
        np.random.seed(42)
        x = np.random.rand(10)
        y = np.random.rand(10)

        dist = distance.euclidean_distance(x, y, x, y)

        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_euclidean_distance_diagonal_zero(self):
        np.random.seed(42)
        x = np.random.rand(10)
        y = np.random.rand(10)

        dist = distance.euclidean_distance(x, y, x, y)

        np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(10))

    def test_distance_positive(self):
        np.random.seed(42)
        x1 = np.random.rand(5)
        y1 = np.random.rand(5)
        x2 = np.random.rand(8)
        y2 = np.random.rand(8)

        dist = distance.euclidean_distance(x1, y1, x2, y2)

        assert np.all(dist >= 0)

class TestMatrixOperations:

    def test_solve_kriging_system_basic(self):
        # Create simple symmetric positive definite matrix
        A = np.array([
        [2.0, 1.0],
        [1.0, 2.0]
        ])
        b = np.array([3.0, 3.0])

        x = matrices.solve_kriging_system(A, b)

        assert len(x) == 2
        assert all(np.isfinite(x))

        # Check solution
        np.testing.assert_array_almost_equal(A @ x, b)

    def test_regularize_matrix(self):
        # Create matrix that might be ill-conditioned
        A = np.array([
        [1.0, 0.999],
        [0.999, 1.0]
        ])

        A_reg = matrices.regularize_matrix(A, epsilon=0.01)

        # Diagonal should be increased
        assert A_reg[0, 0] > A[0, 0]
        assert A_reg[1, 1] > A[1, 1]

        # Off-diagonal should be unchanged
        assert A_reg[0, 1] == A[0, 1]

    def test_matrix_positive_definite_check(self):
        # Positive definite matrix
        A_pos = np.array([
        [2.0, 1.0],
        [1.0, 2.0]
        ])

        # Check eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(A_pos)
        assert all(eigenvalues > 0)

    def test_covariance_matrix_properties(self):
        # Create covariance matrix
        np.random.seed(42)
        data = np.random.randn(10, 3)
        cov = np.cov(data, rowvar=False)

        # Should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)

        # Should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(cov)
        assert all(eigenvalues >= -1e-10) # Allow small numerical errors

class TestNumericalMethods:

    def test_cross_validation_score_rmse(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = numerical.cross_validation_score(y_true, y_pred)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

        # RMSE should be reasonable
        assert 0 < metrics['rmse'] < 1.0

    def test_cross_validation_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        metrics = numerical.cross_validation_score(y_true, y_pred)

        assert metrics['rmse'] < 1e-10
        assert metrics['mae'] < 1e-10
        assert abs(metrics['r2'] - 1.0) < 1e-10

    def test_numerical_stability_small_values(self):
        y_true = np.array([1e-8, 2e-8, 3e-8])
        y_pred = np.array([1.1e-8, 2.1e-8, 2.9e-8])

        metrics = numerical.cross_validation_score(y_true, y_pred)

        # Should handle small values without overflow/underflow
        assert all(np.isfinite([metrics['rmse'], metrics['mae']]))

class TestGridUtilities:

    def test_create_regular_grid(self):
        x_grid, y_grid = grid_utils.create_grid(
        x_min=0, x_max=100,
        y_min=0, y_max=100,
        nx=11, ny=11
        )

        assert x_grid.shape == (11, 11)
        assert y_grid.shape == (11, 11)

        # Check corners
        assert x_grid[0, 0] == 0
        assert x_grid[-1, -1] == 100
        assert y_grid[0, 0] == 0
        assert y_grid[-1, -1] == 100

    def test_grid_spacing(self):
        x_grid, y_grid = grid_utils.create_grid(
        x_min=0, x_max=10,
        y_min=0, y_max=10,
        nx=11, ny=11
        )

        # Spacing should be 1.0
        x_spacing = x_grid[0, 1] - x_grid[0, 0]
        y_spacing = y_grid[1, 0] - y_grid[0, 0]

        assert abs(x_spacing - 1.0) < 1e-10
        assert abs(y_spacing - 1.0) < 1e-10

    def test_grid_to_points(self):
        x_grid, y_grid = grid_utils.create_grid(
        x_min=0, x_max=10,
        y_min=0, y_max=10,
        nx=3, ny=3
        )

        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        assert len(x_flat) == 9
        assert len(y_flat) == 9

        # Check that all combinations are present
        assert 0 in x_flat
        assert 5 in x_flat
        assert 10 in x_flat

class TestDataUtilities:

    def test_generate_synthetic_data(self):
        np.random.seed(42)

        x, y, z = data_utils.generate_synthetic_data(
        n_points=100,
        spatial_structure='spherical',
        nugget=0.1,
        sill=1.0,
        range_param=30.0
        )

        assert len(x) == 100
        assert len(y) == 100
        assert len(z) == 100

        assert all(np.isfinite(x))
        assert all(np.isfinite(y))
        assert all(np.isfinite(z))

    def test_split_train_test(self):
        np.random.seed(42)
        n = 100
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)

        x_train, y_train, z_train, x_test, y_test, z_test = \
        data_utils.split_train_test(x, y, z, test_fraction=0.2)

        assert len(x_train) == 80
        assert len(x_test) == 20

        # All data should be used
        assert len(x_train) + len(x_test) == n

    def test_data_statistics(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        mean = np.mean(data)
        std = np.std(data)

        assert abs(mean - 5.5) < 1e-10
        assert abs(std - np.sqrt(8.25)) < 1e-10

class TestEdgeCases:

    def test_distance_empty_arrays(self):
        x1 = np.array([])
        y1 = np.array([])
        x2 = np.array([1, 2])
        y2 = np.array([1, 2])

        dist = distance.euclidean_distance(x1, y1, x2, y2)

        assert dist.shape == (0, 2)

    def test_matrix_solve_singular(self):
        # Singular matrix (rank deficient)
        A = np.array([
        [1.0, 1.0],
        [1.0, 1.0]
        ])
        b = np.array([1.0, 1.0])

        # Should either regularize or raise error
        try:
            # If it succeeds, solution should satisfy equation approximately
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1.0
        except np.linalg.LinAlgError:
            # Acceptable to raise error for singular matrix
            pass

    def test_grid_single_point(self):
        x_grid, y_grid = grid_utils.create_grid(
        x_min=5, x_max=5,
        y_min=10, y_max=10,
        nx=1, ny=1
        )

        assert x_grid.shape == (1, 1)
        assert x_grid[0, 0] == 5
        assert y_grid[0, 0] == 10

    def test_very_large_distances(self):
        x1 = np.array([0, 1e6])
        y1 = np.array([0, 1e6])
        x2 = np.array([1e6, 0])
        y2 = np.array([1e6, 0])

        dist = distance.euclidean_distance(x1, y1, x2, y2)

        # Should handle large values without overflow
        assert all(np.isfinite(dist.flatten()))

class TestComputationalEfficiency:

    def test_distance_vectorized(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.rand(n)
        y1 = np.random.rand(n)
        x2 = np.random.rand(n)
        y2 = np.random.rand(n)

        # Should complete quickly (vectorized operation)
        import time
        start = time.time()
        dist = distance.euclidean_distance(x1, y1, x2, y2)
        elapsed = time.time() - start

        # Should be very fast (<0.1 seconds)
        assert elapsed < 0.1
        assert dist.shape == (n, n)

    def test_matrix_operations_efficient(self):
        np.random.seed(42)
        n = 50
        A = np.random.rand(n, n)
        A = A @ A.T # Make symmetric positive definite
        b = np.random.rand(n)

        import time
        start = time.time()
        x = matrices.solve_kriging_system(A, b)
        elapsed = time.time() - start

        # Should use efficient solver (<0.1 seconds for n=50)
        assert elapsed < 0.1
        assert len(x) == n

if __name__ == "__main__":
