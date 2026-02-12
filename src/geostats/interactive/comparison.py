"""
Interactive Comparison Tools
=============================

Interactive comparison of methods and cross-validation results.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List

try:
try:
 PLOTLY_AVAILABLE = True
except ImportError:
 PLOTLY_AVAILABLE = False

def interactive_comparison(
 title: str = 'Method Comparison',
    ):
 """
 Interactive comparison of interpolation methods.

 Parameters
 ----------
 comparison_results : dict
 Results from geostats.comparison.compare_interpolation_methods
 title : str
 Plot title

 Returns
 -------
 fig : plotly Figure

 Examples
 --------
 >>> from geostats.comparison import compare_interpolation_methods
 >>> results = compare_interpolation_methods(...)
 >>>
 >>> from geostats.interactive import interactive_comparison
 >>> fig = interactive_comparison(results)
 >>> fig.show()
 """
 if not PLOTLY_AVAILABLE:
 if not PLOTLY_AVAILABLE:
 "plotly is required for interactive plots. "
 "Install with: pip install plotly"
 )

 methods = list(comparison_results.keys())

 # Extract metrics
 rmse_vals = [comparison_results[m]['rmse'] for m in methods]
 mae_vals = [comparison_results[m]['mae'] for m in methods]
 r2_vals = [comparison_results[m]['r2'] for m in methods]
 times = [comparison_results[m].get('time', 0) for m in methods]

 # Create subplots
 fig = make_subplots(
 rows=2, cols=2,
 subplot_titles=('RMSE (lower is better)', 'R² (higher is better)',
 'MAE (lower is better)', 'Computation Time'),
 specs=[[{'type': 'bar'}, {'type': 'bar'}],
 [{'type': 'bar'}, {'type': 'bar'}]]
 )

 # RMSE
 fig.add_trace(
 go.Bar(x=methods, y=rmse_vals, name='RMSE', marker_color='indianred'),
 row=1, col=1
 )

 # R²
 fig.add_trace(
 go.Bar(x=methods, y=r2_vals, name='R²', marker_color='lightseagreen'),
 row=1, col=2
 )

 # MAE
 fig.add_trace(
 go.Bar(x=methods, y=mae_vals, name='MAE', marker_color='lightsalmon'),
 row=2, col=1
 )

 # Time
 fig.add_trace(
 go.Bar(x=methods, y=times, name='Time (s)', marker_color='cornflowerblue'),
 row=2, col=2
 )

 fig.update_layout(
 title_text=title,
 showlegend=False,
 height=800,
 width=1200
 )

 return fig

def interactive_cross_validation(
 predicted: npt.NDArray[np.float64],
 method_name: str = 'Kriging',
    ):
 """
 Interactive cross-validation diagnostic plots.

 Parameters
 ----------
 observed : ndarray
 Observed values
 predicted : ndarray
 Predicted values from CV
 method_name : str
 Name of method

 Returns
 -------
 fig : plotly Figure
 """
 if not PLOTLY_AVAILABLE:
 if not PLOTLY_AVAILABLE:
 "plotly is required for interactive plots. "
 "Install with: pip install plotly"
 )

 errors = observed - predicted

 fig = make_subplots(
 rows=1, cols=2,
 subplot_titles=('Observed vs. Predicted', 'Residual Distribution')
 )

 # Observed vs Predicted
 fig.add_trace(
 go.Scatter(
 x=observed,
 y=predicted,
 mode='markers',
 name='Data',
 marker=dict(size=6, opacity=0.6),
 hovertemplate='Observed: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>'
 ),
 row=1, col=1
 )

 # 1:1 line
 min_val = min(observed.min(), predicted.min())
 max_val = max(observed.max(), predicted.max())
 fig.add_trace(
 go.Scatter(
 x=[min_val, max_val],
 y=[min_val, max_val],
 mode='lines',
 name='1:1 line',
 line=dict(color='red', dash='dash')
 ),
 row=1, col=1
 )

 # Histogram of errors
 fig.add_trace(
 go.Histogram(
 x=errors,
 name='Residuals',
 nbinsx=30,
 marker_color='lightblue'
 ),
 row=1, col=2
 )

 # Compute metrics
 rmse = np.sqrt(np.mean(errors**2))
 mae = np.mean(np.abs(errors))
 r2 = 1 - np.sum(errors**2) / np.sum((observed - observed.mean())**2)

 # Add metrics annotation
 metrics_text = f"<b>Metrics:</b><br>"
 metrics_text += f"RMSE: {rmse:.3f}<br>"
 metrics_text += f"MAE: {mae:.3f}<br>"
 metrics_text += f"R²: {r2:.3f}"

 fig.add_annotation(
 xref="paper", yref="paper",
 x=0.45, y=0.95,
 text=metrics_text,
 showarrow=False,
 bgcolor="white",
 bordercolor="black",
 borderwidth=1
 )

 fig.update_xaxes(title_text="Observed", row=1, col=1)
 fig.update_yaxes(title_text="Predicted", row=1, col=1)
 fig.update_xaxes(title_text="Residuals", row=1, col=2)
 fig.update_yaxes(title_text="Count", row=1, col=2)

 fig.update_layout(
 title_text=f'{method_name} Cross-Validation Results',
 height=500,
 width=1200,
 showlegend=True
 )

 return fig
