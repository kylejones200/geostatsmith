"""
Tests for visualization functions
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class TestVariogramPlots:
    """Test variogram plotting functions"""
    
    def test_plot_variogram(self, sample_data_2d, variogram_model):
        """Test basic variogram plot"""
        x, y, z = sample_data_2d
        
        from geostats.visualization.variogram_plots import plot_variogram
        
        fig, ax = plot_variogram(x, y, z, variogram_model)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_variogram_cloud(self, sample_data_2d):
        """Test variogram cloud plot"""
        x, y, z = sample_data_2d
        
        from geostats.visualization.variogram_plots import plot_variogram_cloud
        
        fig, ax = plot_variogram_cloud(x, y, z)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_directional_variogram(self, sample_data_2d):
        """Test directional variogram plot"""
        x, y, z = sample_data_2d
        
        from geostats.visualization.variogram_plots import plot_directional_variogram
        
        fig = plot_directional_variogram(x, y, z, directions=[0, 45, 90])
        
        assert fig is not None
        plt.close(fig)


class TestSpatialPlots:
    """Test spatial plotting functions"""
    
    def test_plot_prediction_map(self, sample_data_2d, prediction_grid):
        """Test prediction map plot"""
        x, y, z = sample_data_2d
        x_pred, y_pred = prediction_grid
        z_pred = np.random.randn(len(x_pred))
        
        from geostats.visualization.spatial_plots import plot_prediction_map
        
        fig, ax = plot_prediction_map(x_pred, y_pred, z_pred, samples=(x, y, z))
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_plot_variance_map(self, prediction_grid):
        """Test variance map plot"""
        x_pred, y_pred = prediction_grid
        variance = np.random.rand(len(x_pred))
        
        from geostats.visualization.spatial_plots import plot_variance_map
        
        fig, ax = plot_variance_map(x_pred, y_pred, variance)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
