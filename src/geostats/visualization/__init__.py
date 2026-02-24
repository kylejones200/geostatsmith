"""
Visualization tools for geostatistics

Based on Zhang, Y. (2010). Course Notes, Chapter 3
"""

from .diagnostic_plots import (
    plot_cross_validation,
    plot_histogram,
    plot_qq_plot,
)
from .spatial_plots import (
    plot_contour_map,
    plot_data_locations,
    plot_kriging_results,
    plot_symbol_map,
)
from .variogram_plots import (
    plot_directional_variograms,
    plot_h_scatterplot,
    plot_variogram,
    plot_variogram_cloud,
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
