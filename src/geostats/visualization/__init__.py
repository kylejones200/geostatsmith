"""
Visualization tools for geostatistics

Based on Zhang, Y. (2010). Course Notes, Chapter 3
"""

from .variogram_plots import (
    plot_variogram,
    plot_variogram_cloud,
    plot_h_scatterplot,
    plot_directional_variograms,
)
from .spatial_plots import (
    plot_data_locations,
    plot_contour_map,
    plot_symbol_map,
    plot_kriging_results,
)
from .diagnostic_plots import (
    plot_cross_validation,
    plot_histogram,
    plot_qq_plot,
)

__all__ = [
    # Variogram plots
    "plot_variogram",
    "plot_variogram_cloud",
    "plot_h_scatterplot",
    "plot_directional_variograms",
    # Spatial plots
    "plot_data_locations",
    "plot_contour_map",
    "plot_symbol_map",
    "plot_kriging_results",
    # Diagnostic plots
    "plot_cross_validation",
    "plot_histogram",
    "plot_qq_plot",
]
