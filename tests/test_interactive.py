"""
Tests for interactive visualization
"""

import numpy as np
import pytest


class TestInteractiveVisualization:
    """Test interactive visualization functions"""

    def test_interactive_variogram(self, sample_data_2d, variogram_model):
        """Test interactive variogram plot"""
        x, y, z = sample_data_2d

        try:
            from geostats.interactive.variogram_plots import interactive_variogram

            fig = interactive_variogram(x, y, z, variogram_model)
            # Should return a Plotly figure
            assert fig is not None
            # Check if it has the expected structure (Plotly figure)
            assert hasattr(fig, "to_dict") or hasattr(fig, "show")
        except ImportError:
            pytest.skip("Plotly not available")

    def test_interactive_prediction_map(self, sample_data_2d, prediction_grid):
        """Test interactive prediction map"""
        x, y, z = sample_data_2d
        x_pred, y_pred = prediction_grid
        z_pred = np.random.randn(len(x_pred))

        try:
            from geostats.interactive.prediction_maps import interactive_prediction_map

            fig = interactive_prediction_map(x_pred, y_pred, z_pred, samples=(x, y, z))
            assert fig is not None
        except ImportError:
            pytest.skip("Plotly not available")
