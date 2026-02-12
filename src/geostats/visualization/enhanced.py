"""
Enhanced visualization tools for spatial data.

Provides additional visualization capabilities including:
- Hillshading for elevation data
- Comparison plots for interpolation
- 3D terrain visualization

Reference: Python Recipes for Earth Sciences (Trauth 2024), Chapter 7
"""

from .hillshade import hillshade, plot_hillshaded_dem
from .comparison_plots import plot_method_comparison, plot_error_analysis

__all__ = [
    "hillshade",
    "plot_hillshaded_dem",
    "plot_method_comparison",
    "plot_error_analysis",
]
