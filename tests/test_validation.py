"""
Tests for validation and cross-validation methods
"""

import pytest
import numpy as np
from geostats.validation import metrics

class TestMetrics:
    """Test validation metrics"""

    def test_rmse_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        assert abs(rmse) < 1e-10

    def test_rmse_calculation(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        expected = 0.5
        assert abs(rmse - expected) < 1e-10

    def test_mae_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        mae = metrics.mean_absolute_error(y_true, y_pred)
        assert abs(mae) < 1e-10

    def test_mae_calculation(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        mae = metrics.mean_absolute_error(y_true, y_pred)
        assert abs(mae - 1.0) < 1e-10

    def test_r2_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        r2 = metrics.r_squared(y_true, y_pred)
        assert abs(r2 - 1.0) < 1e-10

    def test_r2_poor(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3]) # Just predict mean
        r2 = metrics.r_squared(y_true, y_pred)
        assert abs(r2) < 1e-10 # R² should be 0 for mean prediction

    def test_bias_zero(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        # Bias is just mean(y_pred - y_true)
        bias = np.mean(y_pred - y_true)
        assert abs(bias) < 1e-10

    def test_bias_calculation(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        bias = np.mean(y_pred - y_true)
        assert abs(bias - 1.0) < 1e-10 # Consistently 1 unit too high

    def test_metrics_with_nan(self):
        y_true = np.array([1, 2, np.nan, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        # NaN will propagate
        rmse = metrics.root_mean_squared_error(y_true, y_pred)
        assert np.isnan(rmse)

    def test_metrics_different_lengths(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3, 4])

        with pytest.raises((ValueError, IndexError)):
            metrics.root_mean_squared_error(y_true, y_pred)

class TestCrossValidation:
    """Test cross-validation methods"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.x = np.random.rand(50) * 100
        self.y = np.random.rand(50) * 100
        self.z = np.sin(self.x / 20) + np.cos(self.y / 20) + np.random.randn(50) * 0.1

    def test_leave_one_out_returns_correct_length(self):
        """Test that LOO-CV returns predictions for all points"""
        from geostats import kriging, variogram

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(self.x, self.y, self.z)
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Create kriging object
        ok = kriging.OrdinaryKriging(self.x, self.y, self.z, variogram_model=model)

        # Cross-validate
        predictions, cv_metrics = ok.cross_validate()

        assert len(predictions) == len(self.z)
        assert all(np.isfinite(predictions))
        assert 'rmse' in cv_metrics
        assert 'mae' in cv_metrics
        assert 'r2' in cv_metrics

        if __name__ == "__main__":
        pytest.main([__file__, "-v"])
