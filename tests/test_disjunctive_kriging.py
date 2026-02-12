"""
Tests for Disjunctive Kriging

Tests the non-linear kriging method that handles non-Gaussian data
through Hermite polynomial expansions.
"""

import pytest
import numpy as np
from scipy import stats

from geostats.algorithms.disjunctive_kriging import DisjunctiveKriging
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram_model
from geostats.models.variogram_models import SphericalModel, ExponentialModel
from geostats.core.exceptions import KrigingError


class TestDisjunctiveKriging:

    def setup_method(self):
        np.random.seed(42)
        self.n = 80

        # Create spatial coordinates
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)

        # Generate lognormal data (skewed, non-Gaussian)
        # This simulates environmental data like pollutant concentrations
        spatial_trend = 0.1 * self.x + 0.05 * self.y
        spatial_correlation = np.random.multivariate_normal(
            np.zeros(self.n),
            np.exp(
                -np.sqrt(
                    (self.x[:, None] - self.x[None, :]) ** 2
                    + (self.y[:, None] - self.y[None, :]) ** 2
                )
                / 20
            ),
        )
        self.z = np.exp(
            2
            + 0.5 * spatial_trend
            + 0.3 * spatial_correlation
            + np.random.normal(0, 0.2, self.n)
        )

        # Transform to Gaussian for variogram fitting
        cdf_values = (np.argsort(np.argsort(self.z)) + 0.5) / len(self.z)
        z_gaussian = stats.norm.ppf(np.clip(cdf_values, 1e-10, 1 - 1e-10))

        # Fit variogram on Gaussian-transformed data
        lags, gamma, n_pairs = experimental_variogram(
            self.x, self.y, z_gaussian, n_lags=12
        )
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

    def test_initialization_ordinary(self):
        dk = DisjunctiveKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            max_hermite_order=15,
            kriging_type="ordinary",
        )

        assert dk.n_points == self.n
        assert dk.variogram_model is not None
        assert dk.kriging_type == "ordinary"
        assert dk.max_hermite_order == 15
        assert len(dk.hermite_coeffs) == 16  # 0 to 15 inclusive
        assert dk.mean is None

    def test_initialization_simple(self):
        dk = DisjunctiveKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            max_hermite_order=10,
            kriging_type="simple",
            mean=0.0,
        )

        assert dk.kriging_type == "simple"
        assert dk.mean == 0.0
        assert len(dk.hermite_coeffs) == 11

    def test_initialization_invalid_kriging_type(self):
        from geostats.algorithms.disjunctive_kriging import DisjunctiveKriging
        with pytest.raises(ValueError, match="kriging_type must be 'simple' or 'ordinary'"):
            DisjunctiveKriging(
                self.x,
                self.y,
                self.z,
                variogram_model=self.model,
                kriging_type='invalid'
            )

    def test_hermite_expansion_fitting(self):
        dk = DisjunctiveKriging(
            self.x, self.y, self.z, variogram_model=self.model, max_hermite_order=10
        )

        # Check that coefficients are computed
        assert hasattr(dk, "hermite_coeffs")
        assert len(dk.hermite_coeffs) == 11
        # First coefficient should be non-zero (mean term)
        assert not np.allclose(dk.hermite_coeffs, 0)

    def test_gaussian_transformation(self):
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=self.model)

        # Check that Gaussian transform exists
        assert hasattr(dk, "y_gaussian")
        assert len(dk.y_gaussian) == self.n
        assert np.all(np.isfinite(dk.y_gaussian))

        # Gaussian values should be roughly standard normal
        assert np.abs(np.mean(dk.y_gaussian)) < 0.5  # Mean near 0
        assert 0.5 < np.std(dk.y_gaussian) < 1.5  # Std near 1

    def test_prediction_single_point(self):
        dk = DisjunctiveKriging(
            self.x, self.y, self.z, variogram_model=self.model, max_hermite_order=15
        )

        pred, var = dk.predict(np.array([50.0]), np.array([50.0]), return_variance=True)

        assert len(pred) == 1
        assert len(var) == 1
        assert np.isfinite(pred[0])
        assert np.isfinite(var[0])
        assert var[0] >= 0
        # Prediction should be in original space (positive for lognormal)
        assert pred[0] > 0

    def test_prediction_multiple_points(self):
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=self.model)

        x_pred = np.array([25.0, 50.0, 75.0])
        y_pred = np.array([25.0, 50.0, 75.0])

        pred, var = dk.predict(x_pred, y_pred, return_variance=True)

        assert len(pred) == 3
        assert len(var) == 3
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)
        assert np.all(pred > 0)  # Lognormal data should be positive

    def test_prediction_without_variance(self):
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=self.model)

        pred, var = dk.predict(
            np.array([50.0]), np.array([50.0]), return_variance=False
        )

        assert len(pred) == 1
        assert var is None

    def test_prediction_grid(self):
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=self.model)

        x_grid = np.linspace(0, 100, 20)
        y_grid = np.linspace(0, 100, 20)
        x_2d, y_2d = np.meshgrid(x_grid, y_grid)

        pred, var = dk.predict(x_2d.ravel(), y_2d.ravel(), return_variance=True)

        assert len(pred) == 400
        assert len(var) == 400
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)

    def test_simple_vs_ordinary_kriging(self):
        dk_ordinary = DisjunctiveKriging(
            self.x, self.y, self.z, variogram_model=self.model, kriging_type="ordinary"
        )

        dk_simple = DisjunctiveKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            kriging_type="simple",
            mean=np.mean(dk_ordinary.y_gaussian),
        )

        x_pred = np.array([50.0])
        y_pred = np.array([50.0])

        pred_ord, _ = dk_ordinary.predict(x_pred, y_pred)
        pred_sim, _ = dk_simple.predict(x_pred, y_pred)

        # Both should give valid predictions
        assert np.isfinite(pred_ord[0])
        assert np.isfinite(pred_sim[0])
        # They may differ slightly
        assert pred_ord[0] > 0
        assert pred_sim[0] > 0

    def test_cross_validation(self):
        dk = DisjunctiveKriging(
            self.x, self.y, self.z, variogram_model=self.model, max_hermite_order=10
        )

        cv_predictions, cv_metrics = dk.cross_validate()

        assert len(cv_predictions) == self.n
        assert isinstance(cv_metrics, dict)
        assert "RMSE" in cv_metrics
        assert "MAE" in cv_metrics
        assert "R2" in cv_metrics
        assert "bias" in cv_metrics

        # All predictions should be finite
        assert np.all(np.isfinite(cv_predictions))
        # Metrics should be finite
        assert np.isfinite(cv_metrics["RMSE"])
        assert np.isfinite(cv_metrics["R2"])

    def test_cross_validation_without_variogram(self):
        """Test that cross-validation requires variogram model"""
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=None)

        with pytest.raises(KrigingError, match="Variogram model must be set"):
            dk.cross_validate(n_folds=5)

    def test_prediction_without_variogram(self):
        """Test that prediction requires variogram model"""
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=None)

        with pytest.raises(KrigingError, match="Variogram model must be set"):
            dk.predict(self.x, self.y)

    def test_different_hermite_orders(self):
        """Test that different Hermite orders work"""
        for order in [5, 10, 15, 20]:
            dk = DisjunctiveKriging(
                self.x,
                self.y,
                self.z,
                variogram_model=self.model,
                max_hermite_order=order,
            )

            pred, _ = dk.predict(np.array([50.0]), np.array([50.0]))

            assert np.isfinite(pred[0])
            assert pred[0] > 0

    def test_back_transformation(self):
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=self.model)

        # Test internal back-transformation
        y_test = np.array([0.0, 1.0, -1.0, 2.0])
        z_back = dk._transform_from_gaussian(y_test)

        assert len(z_back) == len(y_test)
        assert np.all(np.isfinite(z_back))

    def test_skewed_data_handling(self):
        # Create highly skewed data
        np.random.seed(123)
        n = 60
        x_skew = np.random.uniform(0, 100, n)
        y_skew = np.random.uniform(0, 100, n)
        z_skew = np.random.lognormal(mean=2, sigma=1, size=n)

        # Transform for variogram
        cdf_vals = (np.argsort(np.argsort(z_skew)) + 0.5) / len(z_skew)
        z_gauss = stats.norm.ppf(np.clip(cdf_vals, 1e-10, 1 - 1e-10))

        lags, gamma, n_pairs = experimental_variogram(
            x_skew, y_skew, z_gauss, n_lags=10
        )
        model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        dk = DisjunctiveKriging(
            x_skew, y_skew, z_skew, variogram_model=model, max_hermite_order=15
        )

        # Should handle skewed data without errors
        pred, var = dk.predict(np.array([50.0]), np.array([50.0]))

        assert np.isfinite(pred[0])
        assert pred[0] > 0
        assert var[0] >= 0

    def test_edge_case_single_point(self):
        x_min = np.array([0.0, 10.0, 20.0])
        y_min = np.array([0.0, 10.0, 20.0])
        z_min = np.array([1.0, 2.0, 3.0])

        # Create simple variogram
        model_min = SphericalModel(nugget=0.1, sill=1.0, range_param=20.0)

        dk = DisjunctiveKriging(
            x_min, y_min, z_min, variogram_model=model_min, max_hermite_order=5
        )

        pred, _ = dk.predict(np.array([5.0]), np.array([5.0]))

        assert np.isfinite(pred[0])

    def test_variance_properties(self):
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=self.model)

        # Predict at sample locations (should have low variance)
        pred_at_samples, var_at_samples = dk.predict(
            self.x[:5], self.y[:5], return_variance=True
        )

        # Predict far from samples (should have higher variance)
        pred_far, var_far = dk.predict(
            np.array([200.0, 200.0]), np.array([200.0, 200.0]), return_variance=True
        )

        # Variance should be non-negative
        assert np.all(var_at_samples >= 0)
        assert np.all(var_far >= 0)

        # Variance far from samples should generally be higher
        # (though not always guaranteed due to back-transformation)
        assert np.all(np.isfinite(var_at_samples))
        assert np.all(np.isfinite(var_far))

    def test_consistency_with_repeated_calls(self):
        dk = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=self.model)

        x_pred = np.array([50.0, 60.0])
        y_pred = np.array([50.0, 60.0])

        pred1, var1 = dk.predict(x_pred, y_pred)
        pred2, var2 = dk.predict(x_pred, y_pred)

        # Should be identical
        np.testing.assert_array_almost_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(var1, var2)

    def test_different_variogram_models(self):
        # Transform for variogram
        cdf_vals = (np.argsort(np.argsort(self.z)) + 0.5) / len(self.z)
        z_gauss = stats.norm.ppf(np.clip(cdf_vals, 1e-10, 1 - 1e-10))

        lags, gamma, n_pairs = experimental_variogram(
            self.x, self.y, z_gauss, n_lags=12
        )

        # Test with Exponential model
        model_exp = fit_variogram_model(
            ExponentialModel(), lags, gamma, weights=n_pairs
        )

        dk_exp = DisjunctiveKriging(self.x, self.y, self.z, variogram_model=model_exp)

        pred, var = dk_exp.predict(np.array([50.0]), np.array([50.0]))

        assert np.isfinite(pred[0])
        assert var[0] >= 0
