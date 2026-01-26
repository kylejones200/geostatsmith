"""
Interactive Visualization Module
=================================

Interactive plots using Plotly for web-based visualization.

Key Features:
- Interactive variogram plots
- Interactive prediction maps with hover info
- Comparison dashboards
- 3D visualizations

Examples
--------
>>> from geostats.interactive import (
... interactive_variogram,
... interactive_prediction_map,
... interactive_comparison
... )
>>>
>>> # Interactive variogram plot
>>> fig = interactive_variogram(x, y, z, model)
>>> fig.show() # Opens in browser
>>>
>>> # Interactive prediction map
>>> fig = interactive_prediction_map(x_grid, y_grid, z_pred, samples=(x, y, z))
>>> fig.show()
"""

from .variogram_plots import (
 interactive_variogram,
 interactive_variogram_cloud,
)

from .prediction_maps import (
 interactive_prediction_map,
 interactive_uncertainty_map,
 interactive_3d_surface,
)

from .comparison import (
 interactive_comparison,
 interactive_cross_validation,
)

__all__ = [
 # Variogram plots
 'interactive_variogram',
 'interactive_variogram_cloud',
 # Prediction maps
 'interactive_prediction_map',
 'interactive_uncertainty_map',
 'interactive_3d_surface',
 # Comparison
 'interactive_comparison',
 'interactive_cross_validation',
]
