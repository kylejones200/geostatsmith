"""
Tests for Uncertainty Quantification Module

Tests:
    pass
- Bootstrap uncertainty estimation
- Confidence intervals
- Probability maps
"""

import pytest
import numpy as np

from geostats.uncertainty.bootstrap import bootstrap_uncertainty
from geostats.uncertainty.confidence_intervals import confidence_intervals
from geostats.uncertainty.probability import probability_map
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram_model
from geostats.models.variogram_models import SphericalModel


class TestBootstrapUncertainty:

    def setup_method(self):
        np.random.seed(42)
        self.n_samples = 50
        self.x = np.random.uniform(0, 100, self.n_samples)
        self.y = np.random.uniform(0, 100, self.n_samples)
        self.z = (
            50 + 0.3 * self.x + 0.2 * self.y + np.random.normal(0, 3, self.n_samples)
        )

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(self.x, self.y, self.z, n_lags=10)
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        # Prediction points
        self.n_pred = 20
        self.x_pred = np.random.uniform(0, 100, self.n_pred)
        self.y_pred = np.random.uniform(0, 100, self.n_pred)

    def test_bootstrap_uncertainty_basic(self):
        results = bootstrap_uncertainty(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_bootstrap=50,  # Reduced for speed
            confidence_level=0.95,
        )

        assert "mean" in results
        assert "std" in results
        assert "lower_bound" in results
        assert "upper_bound" in results
        assert "all_predictions" in results

        assert len(results["mean"]) == self.n_pred
        assert len(results["std"]) == self.n_pred
        assert len(results["lower_bound"]) == self.n_pred
        assert len(results["upper_bound"]) == self.n_pred
        assert results["all_predictions"].shape == (50, self.n_pred)

        # Check bounds are reasonable
        assert np.all(results["lower_bound"] <= results["mean"])
        assert np.all(results["upper_bound"] >= results["mean"])
        assert np.all(results["std"] >= 0)

    def test_bootstrap_uncertainty_residual_method(self):
        results = bootstrap_uncertainty(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_bootstrap=30,
            method="residual",
        )

        assert len(results["mean"]) == self.n_pred
        assert np.all(np.isfinite(results["mean"]))

    def test_bootstrap_uncertainty_pairs_method(self):
        results = bootstrap_uncertainty(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_bootstrap=30,
            method="pairs",
        )

        assert len(results["mean"]) == self.n_pred
        assert np.all(np.isfinite(results["mean"]))

    def test_bootstrap_uncertainty_different_confidence(self):
        results_90 = bootstrap_uncertainty(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_bootstrap=30,
            confidence_level=0.90,
        )

        results_95 = bootstrap_uncertainty(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_bootstrap=30,
            confidence_level=0.95,
        )

        # 95% CI should be wider than 90% CI
        width_90 = results_90["upper_bound"] - results_90["lower_bound"]
        width_95 = results_95["upper_bound"] - results_95["lower_bound"]
        assert np.all(width_95 >= width_90)

    def test_bootstrap_uncertainty_invalid_method(self):
        from geostats.uncertainty.bootstrap import bootstrap_uncertainty
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_uncertainty(
                self.x,
                self.y,
                self.z,
                self.x_pred,
                self.y_pred,
                variogram_model=self.model,
                method="invalid_method"
            )


class TestConfidenceIntervals:

    def setup_method(self):
        np.random.seed(42)
        self.n_samples = 50
        self.x = np.random.uniform(0, 100, self.n_samples)
        self.y = np.random.uniform(0, 100, self.n_samples)
        self.z = (
            50 + 0.3 * self.x + 0.2 * self.y + np.random.normal(0, 3, self.n_samples)
        )

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(self.x, self.y, self.z, n_lags=10)
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        # Prediction points
        self.n_pred = 20
        self.x_pred = np.random.uniform(0, 100, self.n_pred)
        self.y_pred = np.random.uniform(0, 100, self.n_pred)

    def test_confidence_intervals_basic(self):
        results = confidence_intervals(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            confidence_level=0.95,
        )

        assert "predictions" in results
        assert "std_errors" in results
        assert "lower_bound" in results
        assert "upper_bound" in results
        assert "confidence_level" in results

        assert len(results["predictions"]) == self.n_pred
        assert len(results["std_errors"]) == self.n_pred
        assert len(results["lower_bound"]) == self.n_pred
        assert len(results["upper_bound"]) == self.n_pred
        assert results["confidence_level"] == 0.95

        # Check bounds
        assert np.all(results["lower_bound"] <= results["predictions"])
        assert np.all(results["upper_bound"] >= results["predictions"])
        assert np.all(results["std_errors"] >= 0)

    def test_confidence_intervals_different_levels(self):
        results_90 = confidence_intervals(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            confidence_level=0.90,
        )

        results_95 = confidence_intervals(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            confidence_level=0.95,
        )

        # 95% CI should be wider
        width_90 = results_90["upper_bound"] - results_90["lower_bound"]
        width_95 = results_95["upper_bound"] - results_95["lower_bound"]
        assert np.all(width_95 >= width_90)

    def test_confidence_intervals_properties(self):
        results = confidence_intervals(
            self.x, self.y, self.z, self.x_pred, self.y_pred, variogram_model=self.model
        )

        # Standard errors should be square root of variance
        std_from_var = np.sqrt(results["variance"])
        np.testing.assert_allclose(results["std_errors"], std_from_var, rtol=1e-10)

        # Margin should be symmetric
        margin_lower = results["predictions"] - results["lower_bound"]
        margin_upper = results["upper_bound"] - results["predictions"]
        np.testing.assert_allclose(margin_lower, margin_upper, rtol=1e-10)


class TestProbabilityMaps:

    def setup_method(self):
        np.random.seed(42)
        self.n_samples = 40
        self.x = np.random.uniform(0, 100, self.n_samples)
        self.y = np.random.uniform(0, 100, self.n_samples)
        self.z = 50 + 0.3 * self.x + np.random.normal(0, 3, self.n_samples)

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(self.x, self.y, self.z, n_lags=10)
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        # Prediction points
        self.n_pred = 15
        self.x_pred = np.random.uniform(0, 100, self.n_pred)
        self.y_pred = np.random.uniform(0, 100, self.n_pred)

    def test_probability_map_basic(self):
        threshold = 60.0
        prob = probability_map(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            threshold=threshold,
            operator=">",
            n_realizations=20,  # Reduced for speed
        )

        assert len(prob) == self.n_pred
        assert np.all(prob >= 0)
        assert np.all(prob <= 1)
        assert np.all(np.isfinite(prob))

    def test_probability_map_different_operators(self):
        threshold = 60.0

        prob_gt = probability_map(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            threshold=threshold,
            operator=">",
            n_realizations=15,
        )

        prob_lt = probability_map(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            threshold=threshold,
            operator="<",
            n_realizations=15,
        )

        # P(Z > threshold) + P(Z < threshold) should be approximately 1
        # (allowing for P(Z == threshold))
        assert np.all(prob_gt + prob_lt <= 1.1)  # Allow some tolerance

    def test_probability_map_different_thresholds(self):
        prob_low = probability_map(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            threshold=40.0,
            n_realizations=15,
        )

        prob_high = probability_map(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            threshold=70.0,
            n_realizations=15,
        )

        # Lower threshold should have higher probability of exceedance
        assert np.mean(prob_low) >= np.mean(prob_high)

    def test_probability_map_invalid_operator(self):
        from geostats.uncertainty.probability import probability_map
        with pytest.raises(ValueError, match="Unknown operator"):
            probability_map(
                self.x,
                self.y,
                self.z,
                self.x_pred,
                self.y_pred,
                variogram_model=self.model,
                operator="invalid_operator"
            )
