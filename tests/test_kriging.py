"""
Tests for kriging methods
"""

import pytest
import numpy as np
from geostats import kriging, variogram


def test_ordinary_kriging_basic():
    # Create simple test data
    np.random.seed(42)
    x = np.array([0, 10, 20, 30, 40])
    y = np.array([0, 10, 20, 30, 40])
    z = np.array([1.0, 2.0, 1.5, 3.0, 2.5])

    # Create and fit variogram
    lags, gamma, _ = variogram.experimental_variogram(x, y, z, n_lags=3)
    vario_model = variogram.fit_model("spherical", lags, gamma)

    # Create kriging object
    ok = kriging.OrdinaryKriging(x, y, z, variogram_model=vario_model)

    # Predict at known locations
    pred, var = ok.predict(x[:2], y[:2], return_variance=True)

    assert len(pred) == 2
    assert len(var) == 2
    assert all(var >= 0)  # Variance should be non-negative


def test_simple_kriging_basic():
    np.random.seed(42)
    x = np.array([0, 10, 20, 30, 40])
    y = np.array([0, 10, 20, 30, 40])
    z = np.array([1.0, 2.0, 1.5, 3.0, 2.5])

    # Create and fit variogram
    lags, gamma, _ = variogram.experimental_variogram(x, y, z, n_lags=3)
    vario_model = variogram.fit_model("exponential", lags, gamma)

    # Create kriging object
    sk = kriging.SimpleKriging(x, y, z, variogram_model=vario_model, mean=2.0)

    # Predict at new locations
    x_new = np.array([5, 15, 25])
    y_new = np.array([5, 15, 25])
    pred, var = sk.predict(x_new, y_new, return_variance=True)

    assert len(pred) == 3
    assert len(var) == 3
    assert all(np.isfinite(pred))
    assert all(np.isfinite(var))


def test_universal_kriging_basic():
    np.random.seed(42)
    n = 30
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    # Add linear trend
    z = 0.05 * x + 0.03 * y + np.random.randn(n) * 0.5

    # Create and fit variogram
    lags, gamma, _ = variogram.experimental_variogram(x, y, z)
    vario_model = variogram.fit_model("gaussian", lags, gamma)

    # Create Universal Kriging with linear drift
    uk = kriging.UniversalKriging(
        x, y, z, variogram_model=vario_model, drift_terms="linear"
    )

    # Predict
    x_new = np.array([25, 50, 75])
    y_new = np.array([25, 50, 75])
    pred, var = uk.predict(x_new, y_new, return_variance=True)

    assert len(pred) == 3
    assert all(np.isfinite(pred))


def test_cross_validation():
    np.random.seed(42)
    x = np.array([0, 10, 20, 30, 40, 50])
    y = np.array([0, 10, 20, 30, 40, 50])
    z = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 2.0])

    # Create and fit variogram
    lags, gamma, _ = variogram.experimental_variogram(x, y, z)
    vario_model = variogram.fit_model("spherical", lags, gamma)

    # Create kriging and cross-validate
    ok = kriging.OrdinaryKriging(x, y, z, variogram_model=vario_model)
    predictions, metrics = ok.cross_validate()

    assert len(predictions) == len(z)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0


def test_block_kriging():
    np.random.seed(42)
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 100, 20)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = (
        np.sin(x_flat / 20) + np.cos(y_flat / 20) + np.random.randn(len(x_flat)) * 0.1
    )

    # Sample subset
    indices = np.random.choice(len(x_flat), 50, replace=False)
    x_sample = x_flat[indices]
    y_sample = y_flat[indices]
    z_sample = z_flat[indices]

    # Create variogram and kriging
    lags, gamma, _ = variogram.experimental_variogram(x_sample, y_sample, z_sample)
    vario_model = variogram.fit_model("exponential", lags, gamma)
    ok = kriging.OrdinaryKriging(
        x_sample, y_sample, z_sample, variogram_model=vario_model
    )

    # Block kriging
    block_pred, block_var = ok.predict_block((20, 40), (20, 40), discretization=5)

    assert np.isfinite(block_pred)
    assert np.isfinite(block_var)
    assert block_var >= 0


if __name__ == "__main__":
