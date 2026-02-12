"""
Tests for variogram calculation and models
"""

import pytest
import numpy as np
from geostats import variogram
from geostats.models import variogram_models


def test_experimental_variogram():
    # Create simple test data
    np.random.seed(42)
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 0, 0, 0, 0, 0])
    z = np.array([1, 2, 1.5, 3, 2.5, 4])

    lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=3)

    assert len(lags) == 3
    assert len(gamma) == 3
    assert len(n_pairs) == 3
    assert all(lags >= 0)
    assert all(gamma >= 0)


def test_spherical_model():
    model = variogram_models.SphericalModel(nugget=0.1, sill=1.0, range_param=10.0)

    # Test at h=0
    assert abs(model(np.array([0.0]))[0] - 0.1) < 1e-6

    # Test at h=range
    assert abs(model(np.array([10.0]))[0] - 1.0) < 1e-6

    # Test beyond range
    assert abs(model(np.array([20.0]))[0] - 1.0) < 1e-6


def test_exponential_model():
    model = variogram_models.ExponentialModel(nugget=0.0, sill=1.0, range_param=10.0)

    # Test at h=0
    assert abs(model(np.array([0.0]))[0]) < 1e-6

    # Test monotonically increasing
    h_values = np.array([0, 5, 10, 15, 20])
    gamma_values = model(h_values)
    assert all(np.diff(gamma_values) >= 0)


def test_gaussian_model():
    model = variogram_models.GaussianModel(nugget=0.0, sill=1.0, range_param=10.0)

    # Test at h=0
    assert abs(model(np.array([0.0]))[0]) < 1e-6

    # Test smooth near origin
    h_small = np.array([0.1, 0.2])
    gamma_small = model(h_small)
    assert all(gamma_small < 0.1)  # Should be very small near origin


def test_variogram_fitting():
    # Generate synthetic data
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = np.random.randn(n)

    # Calculate experimental variogram
    lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)

    # Fit spherical model
    model = variogram.fit_model("spherical", lags, gamma, weights=n_pairs)

    assert model.is_fitted
    assert "nugget" in model.parameters
    assert "sill" in model.parameters
    assert "range" in model.parameters


def test_auto_fit():
    # Generate synthetic data
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = np.random.randn(n)

    # Calculate experimental variogram
    lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z)

    # Auto-fit
    result = variogram.auto_fit(lags, gamma, weights=n_pairs)

    assert "model" in result
    assert "score" in result
    assert "all_results" in result
    assert result["model"].is_fitted


if __name__ == "__main__":
    pass
