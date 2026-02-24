"""
Tests for comparison and benchmarking
"""

import numpy as np

from geostats.comparison.interpolation_comparison import (
    cross_validate_interpolation,
    interpolation_error_metrics,
)


class TestInterpolationComparison:
    """Test interpolation method comparison"""

    def test_error_metrics(self):
        """Test error metrics calculation"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.0, 3.1, 4.0, 4.9])

        metrics = interpolation_error_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_cross_validate_interpolation(self, sample_data_2d):
        """Test cross-validation comparison"""
        x, y, z = sample_data_2d

        result = cross_validate_interpolation(
            x,
            y,
            z,
            methods=["ordinary_kriging", "idw"],
            n_folds=3,
        )

        assert isinstance(result, dict)
        assert "methods" in result or "results" in result
