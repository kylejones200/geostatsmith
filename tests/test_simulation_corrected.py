"""
Tests for geostatistical simulation methods (CORRECTED APIS)

Tests all simulation approaches with actual library APIs:
    - Unconditional Gaussian Simulation
- Sequential Gaussian Simulation (SGS)
- Conditional vs Unconditional
- Multiple realizations
- Statistical validation
"""

import pytest
import numpy as np
from geostats import variogram
from geostats.simulation.gaussian_simulation import (
 sequential_gaussian_simulation,
 SequentialGaussianSimulation
)
from geostats.simulation.unconditional import unconditional_gaussian_simulation
from geostats.transformations.normal_score import NormalScoreTransform
from geostats.models.variogram_models import SphericalModel, ExponentialModel

class TestUnconditionalSimulation:
    """Tests for unconditional Gaussian simulation"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        # Variogram model (convert to covariance)
        self.model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)

    def test_unconditional_basic(self):
        """Test basic unconditional simulation"""
        # Create simulation grid
        x_sim = np.linspace(0, 100, 20)
        y_sim = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x_sim, y_sim)

        # Run unconditional simulation
        realizations = unconditional_gaussian_simulation(
        X.flatten(),
        Y.flatten(),
        covariance_model=self.model,
        n_realizations=1,
        mean=0.0,
        method="cholesky",
        seed=42
        )

        assert realizations.shape == (1, 400)
        assert all(np.isfinite(realizations.flatten()))

    def test_unconditional_multiple_realizations(self):
        """Test generating multiple realizations"""
        x_sim = np.linspace(0, 100, 10)
        y_sim = np.linspace(0, 100, 10)
        X, Y = np.meshgrid(x_sim, y_sim)

        n_realizations = 5
        realizations = unconditional_gaussian_simulation(
        X.flatten(),
        Y.flatten(),
        covariance_model=self.model,
        n_realizations=n_realizations,
        mean=0.0,
        seed=42
        )

        assert realizations.shape == (5, 100)

        # Realizations should be different
        assert not np.allclose(realizations[0], realizations[1])

    def test_unconditional_statistics(self):
        """Test that unconditional simulation has correct statistics"""
        x_sim = np.linspace(0, 100, 30)
        y_sim = np.linspace(0, 100, 30)
        X, Y = np.meshgrid(x_sim, y_sim)

        # Generate many realizations
        realizations = unconditional_gaussian_simulation(
        X.flatten(),
        Y.flatten(),
        covariance_model=self.model,
        n_realizations=20,
        mean=5.0,
        seed=42
        )

        # Check mean
        overall_mean = np.mean(realizations)
        assert abs(overall_mean - 5.0) < 1.0

    def test_unconditional_reproducibility(self):
        """Test that simulation is reproducible with same seed"""
        x_sim = np.linspace(0, 100, 15)
        y_sim = np.linspace(0, 100, 15)
        X, Y = np.meshgrid(x_sim, y_sim)

        real1 = unconditional_gaussian_simulation(
        X.flatten(), Y.flatten(),
        covariance_model=self.model,
        n_realizations=1,
        seed=123
        )

        real2 = unconditional_gaussian_simulation(
        X.flatten(), Y.flatten(),
        covariance_model=self.model,
        n_realizations=1,
        seed=123
        )

        np.testing.assert_array_equal(real1, real2)

    def test_unconditional_turning_bands(self):
        """Test unconditional simulation with turning bands method"""
        x = np.linspace(0, 50, 20)
        y = np.linspace(0, 50, 20)
        X, Y = np.meshgrid(x, y)

        # Use turning bands method
        realization = unconditional_gaussian_simulation(
        X.flatten(), Y.flatten(),
        covariance_model=self.model,
        n_realizations=1,
        method="turning_bands",
        n_bands=50,
        seed=42
        )

        assert realization.shape == (1, 400)
        assert not np.any(np.isnan(realization))

    def test_unconditional_invalid_method(self):
        """Test that invalid method raises error"""
        x = np.array([0, 10, 20])
        y = np.array([0, 10, 20])

        with pytest.raises(ValueError, match="Unknown method"):
        with pytest.raises(ValueError, match="Unknown method"):
        x, y,
        covariance_model=self.model,
        method="invalid_method"
        )

class TestSequentialGaussianSimulation:
    """Tests for Sequential Gaussian Simulation (conditional)"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_data = 30
        self.x_data = np.random.uniform(0, 100, self.n_data)
        self.y_data = np.random.uniform(0, 100, self.n_data)
        self.z_data = np.random.randn(self.n_data) * 2 + 10

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(
        self.x_data, self.y_data, self.z_data, n_lags=10
        )
        self.model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

    def test_sgs_basic(self):
        """Test basic Sequential Gaussian Simulation"""
        # Simulation grid
        x_sim = np.linspace(0, 100, 15)
        y_sim = np.linspace(0, 100, 15)
        X, Y = np.meshgrid(x_sim, y_sim)

        # Run SGS
        realizations = sequential_gaussian_simulation(
        x_data=self.x_data,
        y_data=self.y_data,
        z_data=self.z_data,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=self.model,
        n_realizations=1,
        seed=42
        )

        assert realizations.shape == (1, 225)
        assert all(np.isfinite(realizations.flatten()))

    def test_sgs_multiple_realizations(self):
        """Test multiple realizations"""
        x_sim = np.linspace(0, 100, 10)
        y_sim = np.linspace(0, 100, 10)
        X, Y = np.meshgrid(x_sim, y_sim)

        realizations = sequential_gaussian_simulation(
        x_data=self.x_data,
        y_data=self.y_data,
        z_data=self.z_data,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=self.model,
        n_realizations=5,
        seed=42
        )

        assert realizations.shape == (5, 100)

        # Different realizations
        assert not np.allclose(realizations[0], realizations[1])

    def test_sgs_class_interface(self):
        """Test SGS class interface"""
        x_sim = np.linspace(0, 100, 12)
        y_sim = np.linspace(0, 100, 12)
        X, Y = np.meshgrid(x_sim, y_sim)

        sgs = SequentialGaussianSimulation(
        x_data=self.x_data,
        y_data=self.y_data,
        z_data=self.z_data,
        variogram_model=self.model
        )

        realizations = sgs.simulate(
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        n_realizations=2,
        seed=42
        )

        assert realizations.shape == (2, 144)
        assert all(np.isfinite(realizations.flatten()))

    def test_sgs_etype(self):
        """Test E-type (expected value) from multiple realizations"""
        x_sim = np.linspace(0, 100, 10)
        y_sim = np.linspace(0, 100, 10)
        X, Y = np.meshgrid(x_sim, y_sim)

        # Generate many realizations
        realizations = sequential_gaussian_simulation(
        x_data=self.x_data,
        y_data=self.y_data,
        z_data=self.z_data,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=self.model,
        n_realizations=10,
        seed=42
        )

        # E-type is mean of realizations
        etype = np.mean(realizations, axis=0)

        assert len(etype) == 100
        assert all(np.isfinite(etype))

        # E-type should be smoother (lower variance)
        assert np.std(etype) < np.mean([np.std(r) for r in realizations])

    def test_sgs_reproducibility(self):
        """Test reproducibility with same seed"""
        x_sim = np.linspace(0, 100, 10)
        y_sim = np.linspace(0, 100, 10)
        X, Y = np.meshgrid(x_sim, y_sim)

        real1 = sequential_gaussian_simulation(
        x_data=self.x_data,
        y_data=self.y_data,
        z_data=self.z_data,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=self.model,
        n_realizations=1,
        seed=123
        )

        real2 = sequential_gaussian_simulation(
        x_data=self.x_data,
        y_data=self.y_data,
        z_data=self.z_data,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=self.model,
        n_realizations=1,
        seed=123
        )

        np.testing.assert_array_almost_equal(real1, real2)

class TestSimulationStatistics:
    """Tests for statistical properties of simulations"""

    def test_histogram_reproduction(self):
        """Test that simulation reproduces data histogram"""
        np.random.seed(42)

        # Create data
        n = 50
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.randn(n) * 2 + 10

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=10)
        model = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)

        # Simulate many points
        x_sim = np.linspace(0, 100, 20)
        y_sim = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x_sim, y_sim)

        realizations = sequential_gaussian_simulation(
        x_data=x,
        y_data=y,
        z_data=z,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=model,
        n_realizations=5,
        seed=42
        )

        all_values = realizations.flatten()

        # Compare statistics (approximately)
        assert abs(np.mean(all_values) - 10.0) < 2.0
        assert abs(np.std(all_values) - 2.0) < 1.0

class TestSimulationEdgeCases:
    """Test edge cases for simulation"""

    def test_simulation_single_conditioning_point(self):
        """Test simulation with minimal conditioning data"""
        x_data = np.array([50.0])
        y_data = np.array([50.0])
        z_data = np.array([10.0])

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)

        x_sim = np.linspace(0, 100, 10)
        y_sim = np.linspace(0, 100, 10)
        X, Y = np.meshgrid(x_sim, y_sim)

        # Should work with single point
        realizations = sequential_gaussian_simulation(
        x_data=x_data,
        y_data=y_data,
        z_data=z_data,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=model,
        n_realizations=1,
        seed=42
        )

        assert realizations.shape == (1, 100)
        assert all(np.isfinite(realizations.flatten()))

    def test_unconditional_small_grid(self):
        """Test unconditional simulation on small grid"""
        x = np.array([0, 10, 20])
        y = np.array([0, 10, 20])
        X, Y = np.meshgrid(x, y)

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=15.0)

        realizations = unconditional_gaussian_simulation(
        X.flatten(),
        Y.flatten(),
        covariance_model=model,
        n_realizations=1,
        seed=42
        )

        assert realizations.shape == (1, 9)
        assert all(np.isfinite(realizations.flatten()))

if __name__ == "__main__":
if __name__ == "__main__":
