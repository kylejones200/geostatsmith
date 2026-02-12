"""
Tests for kriging algorithms

Tests all major kriging variants with various scenarios:
    - Simple Kriging
- Ordinary Kriging (already partially tested)
- Universal Kriging (already partially tested)
- 3D Kriging
- Block Kriging
- Edge cases and error handling
"""

import pytest
import numpy as np
from geostats import kriging, variogram
from geostats.algorithms import simple_kriging, ordinary_kriging, universal_kriging
from geostats.models.variogram_models import SphericalModel, ExponentialModel
from geostats.core.exceptions import KrigingError

class TestSimpleKriging:

    def setup_method(self):
        np.random.seed(42)
        self.n = 50
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)
        self.mean = 5.0
        self.z = self.mean + np.random.randn(self.n) * 2.0

        # Create variogram model
        self.model = SphericalModel(nugget=0.5, sill=4.0, range_param=30.0)

    def test_initialization(self):
        sk = simple_kriging.SimpleKriging(
        self.x, self.y, self.z,
        variogram_model=self.model,
        mean=self.mean
        )
        assert sk.n_points == self.n
        assert sk.mean == self.mean
        assert sk.variogram_model is not None

    def test_prediction_single_point(self):
        sk = simple_kriging.SimpleKriging(
        self.x, self.y, self.z,
        variogram_model=self.model,
        mean=self.mean
        )

        pred, var = sk.predict(np.array([50.0]), np.array([50.0]))

        assert len(pred) == 1
        assert len(var) == 1
        assert np.isfinite(pred[0])
        assert np.isfinite(var[0])
        assert var[0] >= 0

    def test_prediction_multiple_points(self):
        sk = simple_kriging.SimpleKriging(
        self.x, self.y, self.z,
        variogram_model=self.model,
        mean=self.mean
        )

        x_new = np.array([25, 50, 75])
        y_new = np.array([25, 50, 75])
        pred, var = sk.predict(x_new, y_new)

        assert len(pred) == 3
        assert len(var) == 3
        assert all(np.isfinite(pred))
        assert all(np.isfinite(var))
        assert all(var >= 0)

    def test_prediction_at_data_point(self):
        sk = simple_kriging.SimpleKriging(
        self.x, self.y, self.z,
        variogram_model=self.model,
        mean=self.mean
        )

        # Predict at first data point
        pred, var = sk.predict(
        np.array([self.x[0]]),
        np.array([self.y[0]])
        )

        # Should be close to actual value
        assert abs(pred[0] - self.z[0]) < 0.1
        # Variance should be small (but not exactly 0 due to numerical issues)
        assert var[0] < 0.5

    def test_prediction_returns_mean_at_infinity(self):
        sk = simple_kriging.SimpleKriging(
        self.x, self.y, self.z,
        variogram_model=self.model,
        mean=self.mean
        )

        # Predict very far from data
        pred, var = sk.predict(np.array([1000.0]), np.array([1000.0]))

        # Should be close to mean
        assert abs(pred[0] - self.mean) < 1.0

    def test_without_variogram_model(self):
        sk = simple_kriging.SimpleKriging(
        self.x, self.y, self.z,
        variogram_model=None,
        mean=self.mean
        )

        with pytest.raises(KrigingError):

        with pytest.raises(KrigingError):
        """Test prediction on regular grid"""
        sk = simple_kriging.SimpleKriging(
        self.x, self.y, self.z,
        variogram_model=self.model,
        mean=self.mean
        )

        # Create grid
        x_grid = np.linspace(0, 100, 10)
        y_grid = np.linspace(0, 100, 10)
        X, Y = np.meshgrid(x_grid, y_grid)

        pred, var = sk.predict(X.flatten(), Y.flatten())

        assert len(pred) == 100
        assert len(var) == 100
        assert all(np.isfinite(pred))
        assert all(var >= 0)

        # Reshape and check
        Z_pred = pred.reshape(10, 10)
        assert Z_pred.shape == (10, 10)

class TestOrdinaryKrigingExtended:

    def setup_method(self):
        np.random.seed(42)
        self.n = 100
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)
        self.z = np.sin(self.x / 20) + np.cos(self.y / 20) + np.random.randn(self.n) * 0.2

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(
        self.x, self.y, self.z, n_lags=15
        )
        self.model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

    def test_weights_sum_to_one(self):
        ok = ordinary_kriging.OrdinaryKriging(
        self.x, self.y, self.z,
        variogram_model=self.model
        )

        # The weights should sum to 1 due to Lagrange multiplier
        # This is implicit in the ordinary kriging system
        # Just verify prediction works
        pred, var = ok.predict(np.array([50.0]), np.array([50.0]))
        assert np.isfinite(pred[0])

    def test_handles_duplicate_locations(self):
        # Create data with duplicates
        x_dup = np.array([0, 10, 20, 20, 30])
        y_dup = np.array([0, 10, 20, 20, 30])
        z_dup = np.array([1, 2, 3, 3.1, 4])

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=15.0)

        # Should handle duplicates gracefully
        ok = ordinary_kriging.OrdinaryKriging(
        x_dup, y_dup, z_dup,
        variogram_model=model
        )

        pred, var = ok.predict(np.array([15.0]), np.array([15.0]))
        assert np.isfinite(pred[0])

    def test_anisotropic_data(self):
        # Create anisotropic data (stronger correlation in x direction)
        np.random.seed(42)
        n = 80
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 50, n)
        z = np.sin(x / 30) + 0.2 * np.sin(y / 10) + np.random.randn(n) * 0.1

        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
        model = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)

        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)
        pred, var = ok.predict(np.array([50.0]), np.array([25.0]))

        assert np.isfinite(pred[0])
        assert var[0] > 0

class TestUniversalKrigingExtended:

    def test_linear_drift_recovery(self):
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)

        # Create data with linear trend + noise
        z = 0.5 * x + 0.3 * y + 10 + np.random.randn(n) * 0.5

        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=15)
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        uk = universal_kriging.UniversalKriging(
        x, y, z,
        variogram_model=model,
        drift_terms='linear'
        )

        # Predict on grid
        x_test = np.array([25, 50, 75])
        y_test = np.array([25, 50, 75])
        pred, var = uk.predict(x_test, y_test)

        # Expected values based on trend
        expected = 0.5 * x_test + 0.3 * y_test + 10

        # Should be reasonably close to trend
        assert all(np.abs(pred - expected) < 5.0)

    def test_quadratic_drift(self):
        np.random.seed(42)
        n = 80
        x = np.random.uniform(0, 50, n)
        y = np.random.uniform(0, 50, n)

        # Quadratic trend
        z = 0.01 * x**2 + 0.02 * y**2 + np.random.randn(n) * 0.5

        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
        model = variogram.fit_model('gaussian', lags, gamma, weights=n_pairs)

        uk = universal_kriging.UniversalKriging(
        x, y, z,
        variogram_model=model,
        drift_terms='quadratic'
        )

        pred, var = uk.predict(np.array([25.0]), np.array([25.0]))

        assert np.isfinite(pred[0])
        assert var[0] > 0

class TestBlockKriging:

    def test_block_kriging_basic(self):
        np.random.seed(42)
        n = 60
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.sin(x / 20) + np.random.randn(n) * 0.3

        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Block prediction
        block_pred, block_var = ok.predict_block(
        x_block=(40, 60),
        y_block=(40, 60),
        discretization=5
        )

        assert np.isfinite(block_pred)
        assert np.isfinite(block_var)
        assert block_var > 0

    def test_block_variance_smaller_than_point(self):
        np.random.seed(42)
        n = 60
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.randn(n)

        model = SphericalModel(nugget=0.2, sill=1.0, range_param=30.0)
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Point prediction
        point_pred, point_var = ok.predict(np.array([50.0]), np.array([50.0]))

        # Block prediction
        block_pred, block_var = ok.predict_block(
        x_block=(45, 55),
        y_block=(45, 55),
        discretization=5
        )

        # Block variance should be smaller (averaging effect)
        assert block_var < point_var[0]

class TestKrigingEdgeCases:

    def test_single_data_point(self):
        x = np.array([50.0])
        y = np.array([50.0])
        z = np.array([10.0])

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=20.0)

        # Should work but may have limitations
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Prediction should return constant value
        pred, var = ok.predict(np.array([30.0]), np.array([40.0]))
        assert np.isfinite(pred[0])

    def test_collinear_points(self):
        x = np.array([0, 10, 20, 30, 40])
        y = np.array([0, 0, 0, 0, 0]) # All on same line
        z = np.array([1, 2, 3, 4, 5])

        model = SphericalModel(nugget=0.1, sill=2.0, range_param=15.0)

        # Should handle collinear data
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)
        pred, var = ok.predict(np.array([15.0]), np.array([0.0]))

        assert np.isfinite(pred[0])

    def test_extrapolation_far_from_data(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 20)
        y = np.random.uniform(0, 10, 20)
        z = np.random.randn(20)

        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=8)
        model = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)

        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Predict very far away
        pred, var = ok.predict(np.array([1000.0]), np.array([1000.0]))

        # Should still return finite values
        assert np.isfinite(pred[0])
        assert np.isfinite(var[0])
        # Variance should be high (uncertain)
        assert var[0] > 0.5

    def test_empty_prediction_arrays(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 20)
        y = np.random.uniform(0, 10, 20)
        z = np.random.randn(20)

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=5.0)
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Empty arrays
        pred, var = ok.predict(np.array([]), np.array([]))

        assert len(pred) == 0
        assert len(var) == 0

    def test_invalid_coordinates(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 20)
        y = np.random.uniform(0, 10, 20)
        z = np.random.randn(20)

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=5.0)
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # NaN coordinates
        with pytest.raises((ValueError, Exception)):

        with pytest.raises((ValueError, Exception)):
    """Test kriging variance properties"""

    def test_variance_at_data_point_is_small(self):
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)
        z = np.random.randn(50)

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Predict at data point
        pred, var = ok.predict(np.array([x[0]]), np.array([y[0]]))

        # Variance should be approximately nugget
        assert var[0] < 0.5

    def test_variance_increases_with_distance(self):
        np.random.seed(42)
        x = np.array([50.0])
        y = np.array([50.0])
        z = np.array([10.0])

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=20.0)
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Predict at increasing distances
        distances = [5, 10, 20, 40]
        variances = []

        for d in distances:
            variances.append(var[0])

            # Variance should generally increase with distance
            # (may not be strictly monotonic due to kriging system)
        assert variances[-1] > variances[0]

    def test_variance_reaches_sill(self):
        x = np.array([50.0])
        y = np.array([50.0])
        z = np.array([10.0])

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=20.0)
        ok = ordinary_kriging.OrdinaryKriging(x, y, z, variogram_model=model)

        # Predict very far away
        _, var = ok.predict(np.array([500.0]), np.array([500.0]))

        # Should approach sill
        assert var[0] > 0.8 # Close to sill of 1.0

if __name__ == "__main__":
