"""
Tests for Optimization Module

Tests:
- Sampling design optimization
- Cost-benefit analysis
- Sample size calculation
"""

import pytest
import numpy as np

from geostats.optimization.sampling_design import optimal_sampling_design
from geostats.optimization.cost_benefit import (
    sample_size_calculator,
    cost_benefit_analysis,
)
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram_model
from geostats.models.variogram_models import SphericalModel


class TestSamplingDesign:

    def setup_method(self):
        np.random.seed(42)
        self.n_existing = 20
        self.x_existing = np.random.uniform(0, 100, self.n_existing)
        self.y_existing = np.random.uniform(0, 100, self.n_existing)
        self.z_existing = (
            50 + 0.3 * self.x_existing + np.random.normal(0, 3, self.n_existing)
        )

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(
            self.x_existing, self.y_existing, self.z_existing, n_lags=10
        )
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

    def test_optimal_sampling_design_variance_reduction(self):
        x_new, y_new = optimal_sampling_design(
            self.x_existing,
            self.y_existing,
            self.z_existing,
            n_new_samples=5,
            variogram_model=self.model,
            strategy="variance_reduction",
            n_candidates=100,  # Reduced for speed
        )

        assert len(x_new) == 5
        assert len(y_new) == 5
        assert np.all(np.isfinite(x_new))
        assert np.all(np.isfinite(y_new))

    def test_optimal_sampling_design_space_filling(self):
        x_new, y_new = optimal_sampling_design(
            self.x_existing,
            self.y_existing,
            self.z_existing,
            n_new_samples=5,
            variogram_model=self.model,
            strategy="space_filling",
            n_candidates=100,
        )

        assert len(x_new) == 5
        assert len(y_new) == 5

    def test_optimal_sampling_design_hybrid(self):
        x_new, y_new = optimal_sampling_design(
            self.x_existing,
            self.y_existing,
            self.z_existing,
            n_new_samples=5,
            variogram_model=self.model,
            strategy="hybrid",
            n_candidates=100,
        )

        assert len(x_new) == 5
        assert len(y_new) == 5

    def test_optimal_sampling_design_with_bounds(self):
        x_new, y_new = optimal_sampling_design(
            self.x_existing,
            self.y_existing,
            self.z_existing,
            n_new_samples=5,
            variogram_model=self.model,
            x_bounds=(10, 90),
            y_bounds=(10, 90),
            n_candidates=100,
        )

        assert np.all(x_new >= 10)
        assert np.all(x_new <= 90)
        assert np.all(y_new >= 10)
        assert np.all(y_new <= 90)

    def test_optimal_sampling_design_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
        with pytest.raises(ValueError, match="Unknown strategy"):
                self.z_existing,
                n_new_samples=5,
                variogram_model=self.model,
                strategy="invalid",
            )


class TestCostBenefit:

    def setup_method(self):
        np.random.seed(42)
        self.n_initial = 30
        self.x_initial = np.random.uniform(0, 100, self.n_initial)
        self.y_initial = np.random.uniform(0, 100, self.n_initial)
        self.z_initial = (
            50 + 0.3 * self.x_initial + np.random.normal(0, 3, self.n_initial)
        )

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(
            self.x_initial, self.y_initial, self.z_initial, n_lags=10
        )
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

    def test_sample_size_calculator_basic(self):
        results = sample_size_calculator(
            self.x_initial,
            self.y_initial,
            self.z_initial,
            variogram_model=self.model,
            target_rmse=2.0,
            max_samples=100,
            n_simulations=5,  # Reduced for speed
        )

        assert "required_samples" in results
        assert "current_rmse" in results
        assert "target_rmse" in results
        assert "sample_sizes" in results
        assert "rmse_values" in results

        assert results["target_rmse"] == 2.0
        assert np.isfinite(results["current_rmse"])
        assert results["required_samples"] > 0
        assert len(results["sample_sizes"]) > 0
        assert len(results["rmse_values"]) == len(results["sample_sizes"])

    def test_sample_size_calculator_with_bounds(self):
        results = sample_size_calculator(
            self.x_initial,
            self.y_initial,
            self.z_initial,
            variogram_model=self.model,
            target_rmse=2.0,
            x_bounds=(10, 90),
            y_bounds=(10, 90),
            n_simulations=5,
        )

        assert results["required_samples"] > 0
        assert np.isfinite(results["current_rmse"])

    def test_sample_size_calculator_achievable_target(self):
        results = sample_size_calculator(
            self.x_initial,
            self.y_initial,
            self.z_initial,
            variogram_model=self.model,
            target_rmse=10.0,  # High target (should be achievable)
            n_simulations=5,
        )

        # Should find a solution
        assert results["required_samples"] >= self.n_initial
        assert results["current_rmse"] >= results["target_rmse"]

    def test_cost_benefit_analysis_basic(self):
        results = cost_benefit_analysis(
            self.x_initial,
            self.y_initial,
            self.z_initial,
            variogram_model=self.model,
            cost_per_sample=100.0,
            benefit_per_rmse_reduction=50.0,
            max_samples=100,
            n_simulations=5,
        )

        assert "optimal_samples" in results
        assert "optimal_cost" in results
        assert "optimal_benefit" in results
        assert "net_benefit" in results
        assert "sample_sizes" in results
        assert "costs" in results
        assert "benefits" in results

        assert results["optimal_samples"] > 0
        assert results["optimal_cost"] >= 0
        assert np.isfinite(results["net_benefit"])

    def test_cost_benefit_analysis_different_costs(self):
        results1 = cost_benefit_analysis(
            self.x_initial,
            self.y_initial,
            self.z_initial,
            variogram_model=self.model,
            cost_per_sample=50.0,
            benefit_per_rmse_reduction=100.0,
            n_simulations=5,
        )

        results2 = cost_benefit_analysis(
            self.x_initial,
            self.y_initial,
            self.z_initial,
            variogram_model=self.model,
            cost_per_sample=200.0,
            benefit_per_rmse_reduction=100.0,
            n_simulations=5,
        )

        # Higher cost should lead to fewer optimal samples
        # (or at least different optimal solution)
        assert (
            results1["optimal_samples"] != results2["optimal_samples"]
            or results1["net_benefit"] != results2["net_benefit"]
        )

    def test_cost_benefit_analysis_net_benefit_properties(self):
        results = cost_benefit_analysis(
            self.x_initial,
            self.y_initial,
            self.z_initial,
            variogram_model=self.model,
            cost_per_sample=100.0,
            benefit_per_rmse_reduction=50.0,
            n_simulations=5,
        )

        # Net benefit should be benefit - cost
        net_manual = results["optimal_benefit"] - results["optimal_cost"]
        assert np.isclose(results["net_benefit"], net_manual, rtol=1e-5)

        # Optimal samples should maximize net benefit
        # Check that net benefit at optimal is >= net benefit at other points
        net_benefits = results["benefits"] - results["costs"]
        optimal_idx = np.argmin(
            np.abs(results["sample_sizes"] - results["optimal_samples"])
        )
        assert (
            net_benefits[optimal_idx] >= np.max(net_benefits) - 0.1
        )  # Allow small tolerance
