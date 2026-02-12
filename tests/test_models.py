"""
Tests for variogram models
"""

import pytest
import numpy as np
from geostats.models import variogram_models

class TestSphericalModel:

    def test_initialization(self):
        assert model.parameters['nugget'] == 0.1
        assert model.parameters['sill'] == 1.0
        assert model.parameters['range'] == 10.0

    def test_at_zero(self):
        result = model(np.array([0.0]))
        assert abs(result[0] - 0.1) < 1e-6

    def test_at_range(self):
        result = model(np.array([10.0]))
        assert abs(result[0] - 1.0) < 1e-6

    def test_beyond_range(self):
        result = model(np.array([20.0]))
        assert abs(result[0] - 1.0) < 1e-6

    def test_monotonic_increasing(self):
        h_values = np.array([0, 2, 4, 6, 8, 10])
        gamma_values = model(h_values)
        assert all(np.diff(gamma_values) >= -1e-10) # Allow small numerical errors

    def test_vectorized(self):
        h_values = np.array([0, 5, 10, 15, 20])
        result = model(h_values)
        assert len(result) == 5
        assert all(np.isfinite(result))

class TestExponentialModel:

    def test_initialization(self):
        assert model.parameters['nugget'] == 0.0
        assert model.parameters['sill'] == 1.0
        assert model.parameters['range'] == 10.0

    def test_at_zero(self):
        result = model(np.array([0.0]))
        assert abs(result[0]) < 1e-6

    def test_monotonic_increasing(self):
        h_values = np.array([0, 5, 10, 15, 20])
        gamma_values = model(h_values)
        assert all(np.diff(gamma_values) >= -1e-10)

    def test_approaches_sill(self):
        # At 3*range, should be close to sill
        result = model(np.array([30.0]))
        assert result[0] > 0.95 # Should be close to sill

class TestGaussianModel:

    def test_initialization(self):
        assert model.parameters['nugget'] == 0.0
        assert model.parameters['sill'] == 1.0
        assert model.parameters['range'] == 10.0

    def test_at_zero(self):
        result = model(np.array([0.0]))
        assert abs(result[0]) < 1e-6

    def test_smooth_near_origin(self):
        h_small = np.array([0.1, 0.2])
        gamma_small = model(h_small)
        # Gaussian is very smooth near origin
        assert all(gamma_small < 0.01)

class TestLinearModel:

    def test_initialization(self):
        model = variogram_models.LinearModel(nugget=0.0, sill=0.1, range_param=1.0)
        assert model.parameters['nugget'] == 0.0
        assert model.parameters['sill'] == 0.1 # This is actually the slope

    def test_linearity(self):
        model = variogram_models.LinearModel(nugget=0.0, sill=0.1, range_param=1.0)
        h_values = np.array([0, 10, 20, 30])
        expected = np.array([0, 1.0, 2.0, 3.0])
        result = model(h_values)
        np.testing.assert_array_almost_equal(result, expected)

class TestPowerModel:

    def test_initialization(self):
        assert model.parameters['nugget'] == 0.0
        # Internally stored as 'sill' and 'range'
        assert model.parameters['sill'] == 1.0 # This is scale
        assert model.parameters['range'] == 1.5 # This is exponent

    def test_exponent_constraint(self):
        with pytest.raises(ValueError):
            pass

        with pytest.raises(ValueError):
            pass
    """Test Matérn variogram model"""

    def test_initialization(self):
        assert model.parameters['nugget'] == 0.0
        assert model.parameters['sill'] == 1.0
        assert model.parameters['range'] == 10.0
        assert model.parameters['nu'] == 1.5

    def test_at_zero(self):
        result = model(np.array([0.0]))
        assert abs(result[0] - 0.1) < 1e-6

class TestModelFitting:

    def test_fit_to_data(self):
        np.random.seed(42)
        lags = np.linspace(1, 50, 20)

        # True model: spherical with nugget=0.1, sill=1.0, range=30
        true_model = variogram_models.SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)
        gamma_true = true_model(lags)
        gamma_noisy = gamma_true + np.random.randn(len(lags)) * 0.05

        # Fit model
        fitted_model = variogram_models.SphericalModel()
        fitted_model.fit(lags, gamma_noisy)

        assert fitted_model.is_fitted
        assert 'nugget' in fitted_model.parameters
        assert 'sill' in fitted_model.parameters
        assert 'range' in fitted_model.parameters

        # Parameters should be close to true values
        assert abs(fitted_model.parameters['nugget'] - 0.1) < 0.2
        assert abs(fitted_model.parameters['sill'] - 1.0) < 0.3
        assert abs(fitted_model.parameters['range'] - 30.0) < 10.0

if __name__ == "__main__":
    pass
