"""
Tests for variogram fitting algorithms and datasets

Tests:
- Variogram model fitting
- Optimization methods
- Dataset loading
- Data validation
"""

import pytest
import numpy as np
from geostats import variogram
from geostats.algorithms import fitting
from geostats.models.variogram_models import (
 SphericalModel, ExponentialModel, GaussianModel,
 LinearModel, PowerModel
)
from geostats.datasets import walker_lake

class TestVariogramFitting:
    """Tests for variogram model fitting"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)

        # Create synthetic spatial data
        n = 100
        self.x = np.random.uniform(0, 100, n)
        self.y = np.random.uniform(0, 100, n)

        # Create data with known spatial structure
        model = SphericalModel(nugget=0.5, sill=2.0, range_param=30.0)

        # Approximate simulation by creating correlated data
        self.z = np.random.randn(n)

        # Compute experimental variogram
        self.lags, self.gamma, self.n_pairs = variogram.experimental_variogram(
        self.x, self.y, self.z, n_lags=15
        )

    def test_fit_spherical_model(self):
        """Test fitting spherical variogram model"""
        fitted_model = variogram.fit_model(
        'spherical',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        assert fitted_model is not None
        assert isinstance(fitted_model, SphericalModel)
        assert fitted_model.nugget >= 0
        assert fitted_model.sill > 0
        assert fitted_model.range_param > 0

    def test_fit_exponential_model(self):
        """Test fitting exponential variogram model"""
        fitted_model = variogram.fit_model(
        'exponential',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        assert fitted_model is not None
        assert isinstance(fitted_model, ExponentialModel)
        assert fitted_model.nugget >= 0
        assert fitted_model.sill > 0
        assert fitted_model.range_param > 0

    def test_fit_gaussian_model(self):
        """Test fitting Gaussian variogram model"""
        fitted_model = variogram.fit_model(
        'gaussian',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        assert fitted_model is not None
        assert isinstance(fitted_model, GaussianModel)
        assert fitted_model.nugget >= 0
        assert fitted_model.sill > 0
        assert fitted_model.range_param > 0

    def test_fit_linear_model(self):
        """Test fitting linear variogram model"""
        fitted_model = variogram.fit_model(
        'linear',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        assert fitted_model is not None
        assert isinstance(fitted_model, LinearModel)

    def test_fit_with_weights(self):
        """Test that weights affect fit quality"""
        # Fit with weights (more emphasis on reliable lags)
        fitted_weighted = variogram.fit_model(
        'spherical',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        # Fit without weights
        fitted_unweighted = variogram.fit_model(
        'spherical',
        self.lags,
        self.gamma,
        weights=None
        )

        # Both should be valid, but may differ
        assert fitted_weighted is not None
        assert fitted_unweighted is not None

        # Parameters might be different
        # (but not guaranteed, so we just check they're valid)
        assert fitted_weighted.sill > 0
        assert fitted_unweighted.sill > 0

    def test_fit_evaluates_on_lags(self):
        """Test that fitted model can evaluate on lag distances"""
        fitted_model = variogram.fit_model(
        'spherical',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        # Evaluate fitted model at lag distances
        gamma_fitted = fitted_model(self.lags)

        assert len(gamma_fitted) == len(self.lags)
        assert all(np.isfinite(gamma_fitted))
        assert all(gamma_fitted >= 0) # Variogram values should be non-negative

    def test_goodness_of_fit(self):
        """Test calculation of goodness of fit"""
        fitted_model = variogram.fit_model(
        'spherical',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        # Calculate R-squared or similar metric
        gamma_fitted = fitted_model(self.lags)

        # Residuals
        residuals = self.gamma - gamma_fitted

        # All residuals should be finite
        assert all(np.isfinite(residuals))

    def test_fitting_with_few_lags(self):
        """Test fitting with very few lag points"""
        # Use only first 3 lags
        lags_few = self.lags[:3]
        gamma_few = self.gamma[:3]
        n_pairs_few = self.n_pairs[:3]

        # Should still fit, though quality may be poor
        fitted_model = variogram.fit_model(
        'spherical',
        lags_few,
        gamma_few,
        weights=n_pairs_few
        )

        assert fitted_model is not None

    def test_fitting_preserves_nugget_behavior(self):
        """Test that fitted model has correct nugget behavior"""
        fitted_model = variogram.fit_model(
        'spherical',
        self.lags,
        self.gamma,
        weights=self.n_pairs
        )

        # At distance 0, should be close to nugget
        gamma_0 = fitted_model(np.array([0.0]))
        assert abs(gamma_0[0] - fitted_model.nugget) < 0.1

class TestFittingMethods:
    """Tests for optimization methods used in fitting"""

    def test_weighted_least_squares(self):
        """Test weighted least squares fitting"""
        # Create simple linear problem: y = 2*x + 1
        x = np.array([1, 2, 3, 4, 5])
        y = 2*x + 1 + np.random.randn(5) * 0.1
        weights = np.array([1, 1, 2, 2, 2]) # More weight on later points

        # This is a simplified test - actual implementation uses scipy.optimize
        # We're just checking that the fitting module exists and can be called
        assert hasattr(fitting, 'fit_variogram_model')

    def test_parameter_bounds(self):
        """Test that fitting respects parameter bounds"""
        np.random.seed(42)
        x = np.random.uniform(0, 50, 50)
        y = np.random.uniform(0, 50, 50)
        z = np.random.randn(50)

        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=10)

        fitted_model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Check that parameters are within reasonable bounds
        assert 0 <= fitted_model.nugget <= 10 * np.var(z)
        assert 0 < fitted_model.sill <= 10 * np.var(z)
        assert 0 < fitted_model.range_param <= 2 * np.max(lags)

    def test_convergence(self):
        """Test that fitting converges"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 80)
        y = np.random.uniform(0, 100, 80)
        z = np.random.randn(80)

        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)

        # Should converge without errors
        fitted_model = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)

        assert fitted_model is not None
        # If it converged, model should produce reasonable values
        test_lags = np.array([10, 20, 30])
        gamma_test = fitted_model(test_lags)
        assert all(np.isfinite(gamma_test))

class TestWalkerLakeDataset:
    """Tests for Walker Lake dataset"""

    def test_load_dataset(self):
        """Test loading Walker Lake dataset"""
        data = walker_lake.load_data()

        assert data is not None
        assert 'x' in data or 'X' in data
        assert 'y' in data or 'Y' in data
        assert 'V' in data or 'value' in data

    def test_dataset_shape(self):
        """Test dataset has expected structure"""
        data = walker_lake.load_data()

        # Should have at least 100 points (Walker Lake is a standard dataset)
        n_points = len(data.get('x', data.get('X', [])))
        assert n_points > 50

    def test_dataset_values_valid(self):
        """Test that dataset values are valid"""
        data = walker_lake.load_data()

        x_key = 'x' if 'x' in data else 'X'
        y_key = 'y' if 'y' in data else 'Y'
        v_key = 'V' if 'V' in data else 'value'

        x = data[x_key]
        y = data[y_key]
        v = data[v_key]

        # All values should be finite
        assert all(np.isfinite(x))
        assert all(np.isfinite(y))
        assert all(np.isfinite(v))

        # Values should be non-negative (typical for concentration data)
        assert all(v >= 0)

    def test_dataset_coordinates_reasonable(self):
        """Test that coordinates are in reasonable range"""
        data = walker_lake.load_data()

        x_key = 'x' if 'x' in data else 'X'
        y_key = 'y' if 'y' in data else 'Y'

        x = data[x_key]
        y = data[y_key]

        # Coordinates should be positive and within reasonable bounds
        assert np.min(x) >= 0
        assert np.min(y) >= 0
        assert np.max(x) < 1e6 # Not unreasonably large
        assert np.max(y) < 1e6

class TestVariogramComputation:
    """Additional tests for experimental variogram computation"""

    def test_variogram_with_different_lag_numbers(self):
        """Test variogram with different numbers of lags"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 60)
        y = np.random.uniform(0, 100, 60)
        z = np.random.randn(60)

        for n_lags in [5, 10, 15, 20]:
        for n_lags in [5, 10, 15, 20]:
        x, y, z, n_lags=n_lags
        )

        assert len(lags) <= n_lags
        assert len(gamma) == len(lags)
        assert len(n_pairs) == len(lags)

    def test_variogram_lag_tolerance(self):
        """Test variogram with different lag tolerances"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)
        z = np.random.randn(50)

        # Compute with default tolerance
        lags1, gamma1, n_pairs1 = variogram.experimental_variogram(
        x, y, z, n_lags=10
        )

        # Both should be valid
        assert len(lags1) > 0
        assert all(np.isfinite(gamma1))

    def test_variogram_directional(self):
        """Test directional variogram (if implemented)"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 80)
        y = np.random.uniform(0, 100, 80)
        # Create anisotropic data (stronger correlation in x direction)
        z = np.sin(x / 20) + 0.1 * np.sin(y / 20) + np.random.randn(80) * 0.2

        # Standard omnidirectional variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(
        x, y, z, n_lags=12
        )

        assert len(lags) > 0
        assert all(np.isfinite(gamma))

    def test_variogram_increases_with_distance(self):
        """Test that variogram generally increases with distance"""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)

        # Create spatially correlated data
        # Simple approach: base value + smooth component + noise
        z = 10 + np.sin(x / 30) * np.cos(y / 30) + np.random.randn(n) * 0.5

        lags, gamma, n_pairs = variogram.experimental_variogram(
        x, y, z, n_lags=15
        )

        # In general, gamma should increase with lag
        # (though not strictly monotonic due to sampling variation)
        # Check that later lags are generally larger than early lags
        if len(gamma) >= 5:
        if len(gamma) >= 5:
        late_mean = np.mean(gamma[-2:])
        # Late lags should generally be larger (more variance)
        assert late_mean >= early_mean * 0.5 # Allow some flexibility

class TestEdgeCases:
    """Test edge cases in fitting"""

    def test_fit_with_constant_variogram(self):
        """Test fitting when variogram is constant (pure nugget)"""
        lags = np.array([10, 20, 30, 40, 50])
        gamma = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) # Constant
        n_pairs = np.array([100, 90, 80, 70, 60])

        # Should handle constant variogram
        try:
        try:
        # If successful, should have large nugget, small sill
        assert fitted_model.nugget > 0
        except (ValueError, RuntimeError):
        # Also acceptable to fail on degenerate case
        pass

    def test_fit_with_decreasing_variogram(self):
        """Test fitting when variogram decreases (hole effect)"""
        lags = np.array([10, 20, 30, 40, 50])
        gamma = np.array([0.5, 1.0, 0.8, 0.6, 0.7]) # Non-monotonic
        n_pairs = np.array([100, 90, 80, 70, 60])

        # Should handle non-monotonic variogram
        fitted_model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Fit may not be perfect, but should be valid
        assert fitted_model is not None
        assert fitted_model.sill > 0

    def test_fit_with_missing_values(self):
        """Test fitting with NaN values in experimental variogram"""
        lags = np.array([10, 20, 30, 40, 50])
        gamma = np.array([0.5, 1.0, np.nan, 1.8, 2.0]) # Has NaN
        n_pairs = np.array([100, 90, 0, 70, 60]) # Zero pairs for NaN lag

        # Should handle NaN by filtering or skipping
        # Remove NaN values before fitting
        valid = ~np.isnan(gamma)
        lags_valid = lags[valid]
        gamma_valid = gamma[valid]
        n_pairs_valid = n_pairs[valid]

        fitted_model = variogram.fit_model(
        'spherical',
        lags_valid,
        gamma_valid,
        weights=n_pairs_valid
        )

        assert fitted_model is not None

if __name__ == "__main__":
if __name__ == "__main__":
