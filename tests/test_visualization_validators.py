"""
Tests for visualization and validation utilities

Tests:
- Spatial plots
- Variogram plots
- Diagnostic plots
- Input validators
- Data validators
"""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from geostats import variogram
from geostats.visualization import spatial_plots, variogram_plots, diagnostic_plots
from geostats.core.validators import (
    validate_coordinates,
    validate_values,
    validate_positive,
    validate_array_shapes_match,
)
from geostats.models.variogram_models import SphericalModel


class TestSpatialPlots:
    """Tests for spatial plotting functions"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n = 50
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)
        self.z = np.random.randn(self.n)

    def teardown_method(self):
        """Close all plots after each test"""
        plt.close("all")

    def test_scatter_plot(self):
        """Test basic scatter plot creation"""
        fig, ax = spatial_plots.plot_data_points(self.x, self.y, self.z)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_scatter_plot_with_colorbar(self):
        """Test scatter plot with colorbar"""
        fig, ax = spatial_plots.plot_data_points(self.x, self.y, self.z, colorbar=True)

        assert fig is not None
        plt.close(fig)

    def test_contour_plot(self):
        """Test contour plot creation"""
        # Create grid data
        x_grid = np.linspace(0, 100, 20)
        y_grid = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.sin(X / 20) + np.cos(Y / 20)

        fig, ax = spatial_plots.plot_contour(X, Y, Z)

        assert fig is not None
        plt.close(fig)

    def test_contourf_plot(self):
        """Test filled contour plot"""
        x_grid = np.linspace(0, 100, 20)
        y_grid = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.sin(X / 20) + np.cos(Y / 20)

        fig, ax = spatial_plots.plot_filled_contour(X, Y, Z)

        assert fig is not None
        plt.close(fig)

    def test_plot_with_title(self):
        """Test plot with custom title"""
        fig, ax = spatial_plots.plot_data_points(
            self.x, self.y, self.z, title="Test Plot"
        )

        assert ax.get_title() == "Test Plot"
        plt.close(fig)

    def test_plot_with_labels(self):
        """Test plot with axis labels"""
        fig, ax = spatial_plots.plot_data_points(
            self.x, self.y, self.z, xlabel="X Coordinate", ylabel="Y Coordinate"
        )

        assert "X Coordinate" in ax.get_xlabel()
        assert "Y Coordinate" in ax.get_ylabel()
        plt.close(fig)


class TestVariogramPlots:
    """Tests for variogram plotting functions"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.randn(n)

        # Compute experimental variogram
        self.lags, self.gamma, self.n_pairs = variogram.experimental_variogram(
            x, y, z, n_lags=15
        )

        # Fit model
        self.model = variogram.fit_model(
            "spherical", self.lags, self.gamma, weights=self.n_pairs
        )

    def teardown_method(self):
        """Close all plots"""
        plt.close("all")

    def test_plot_experimental_variogram(self):
        """Test plotting experimental variogram"""
        fig, ax = variogram_plots.plot_experimental_variogram(
            self.lags, self.gamma, n_pairs=self.n_pairs
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_variogram_model(self):
        """Test plotting variogram model"""
        fig, ax = variogram_plots.plot_variogram_model(self.model, max_distance=100)

        assert fig is not None
        plt.close(fig)

    def test_plot_experimental_and_model(self):
        """Test plotting experimental variogram with fitted model"""
        fig, ax = variogram_plots.plot_variogram_with_model(
            self.lags, self.gamma, self.model, n_pairs=self.n_pairs
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_variogram_map(self):
        """Test variogram map (directional variography)"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 80)
        y = np.random.uniform(0, 100, 80)
        z = np.random.randn(80)

        fig, ax = variogram_plots.plot_variogram_map(x, y, z)

        assert fig is not None
        plt.close(fig)


class TestDiagnosticPlots:
    """Tests for diagnostic plotting functions"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n = 50
        self.observed = np.random.randn(self.n) * 2 + 10
        self.predicted = self.observed + np.random.randn(self.n) * 0.5
        self.residuals = self.observed - self.predicted

    def teardown_method(self):
        """Close all plots"""
        plt.close("all")

    def test_qq_plot(self):
        """Test Q-Q plot"""
        fig, ax = diagnostic_plots.qq_plot(self.residuals)

        assert fig is not None
        plt.close(fig)

    def test_histogram(self):
        """Test histogram plot"""
        fig, ax = diagnostic_plots.plot_histogram(self.observed)

        assert fig is not None
        plt.close(fig)

    def test_scatterplot_observed_vs_predicted(self):
        """Test observed vs predicted scatter plot"""
        fig, ax = diagnostic_plots.plot_obs_vs_pred(self.observed, self.predicted)

        assert fig is not None
        plt.close(fig)

    def test_residual_plot(self):
        """Test residual plot"""
        fig, ax = diagnostic_plots.plot_residuals(self.predicted, self.residuals)

        assert fig is not None
        plt.close(fig)

    def test_residual_histogram(self):
        """Test residual histogram"""
        fig, ax = diagnostic_plots.plot_residual_histogram(self.residuals)

        assert fig is not None
        plt.close(fig)


class TestValidators:
    """Tests for input validation functions"""

    def test_validate_coordinates_valid(self):
        """Test validation of valid coordinates"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        # Should not raise error
        validate_coordinates(x, y)

    def test_validate_coordinates_mismatched_length(self):
        """Test validation catches mismatched lengths"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3])

        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test validation catches NaN values"""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test validation catches infinite values"""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, np.inf, 4, 5])

        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test validation of valid values"""
        z = np.array([10, 20, 30, 40, 50])

        # Should not raise error
        validate_values(z, name="z")

    def test_validate_values_with_nan(self):
        """Test validation catches NaN in values"""
        z = np.array([10, 20, np.nan, 40, 50])

        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test validation catches infinite values"""
        z = np.array([10, 20, np.inf, 40, 50])

        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test validation of positive values"""
        value = 5.0

        # Should not raise error
        result = validate_positive(value, name="test_value")
        assert result == 5.0

    def test_validate_positive_rejects_negative(self):
        """Test that negative values are rejected"""
        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test that zero is rejected"""
        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test array shape validation"""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([6, 7, 8, 9, 10])

        # Should not raise error
        validate_array_shapes_match(a, b)

    def test_validate_array_shapes_mismatch(self):
        """Test detection of shape mismatch"""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([6, 7, 8])

        with pytest.raises((ValueError, AssertionError)):


        with pytest.raises((ValueError, AssertionError)):
    """Tests for parameter validation"""

    def test_negative_nugget_rejected(self):
        """Test that negative nugget is rejected"""
        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test that negative sill is rejected"""
        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test that negative range is rejected"""
        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test that zero range is rejected"""
        with pytest.raises((ValueError, AssertionError)):

        with pytest.raises((ValueError, AssertionError)):
        """Test that valid parameters are accepted"""
        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)

        assert model.nugget == 0.1
        assert model.sill == 1.0
        assert model.range_param == 30.0


class TestDataQuality:
    """Tests for data quality checks"""

    def test_data_coverage_check(self):
        """Test checking data coverage"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)

        # Check coverage
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)

        assert x_range > 50  # Should cover reasonable area
        assert y_range > 50


class TestPlotSaving:
    """Tests for saving plots"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.x = np.random.uniform(0, 100, 50)
        self.y = np.random.uniform(0, 100, 50)
        self.z = np.random.randn(50)

    def teardown_method(self):
        """Close all plots"""
        plt.close("all")

    def test_save_plot_to_file(self):
        """Test saving plot to file"""
        import os
        import tempfile

        fig, ax = spatial_plots.plot_data_points(self.x, self.y, self.z)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:

             with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:

                # Check file was created
                assert os.path.exists(tmp_path)
                assert os.path.getsize(tmp_path) > 0
            finally:
                # Clean up
                if os.path.exists(tmp_path):
                if os.path.exists(tmp_path):


if __name__ == "__main__":
     if __name__ == "__main__":
