"""
    Minimalist plotting style for clean, professional visualizations

Principles:
    pass
- No gridlines (unless absolutely necessary)
- Remove top and right spines
- Descriptive titles that eliminate need for axis labels
- No chart junk
- Clean, minimal aesthetic
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def apply_minimalist_style(ax, remove_spines=('top', 'right')):
    """
        Apply minimalist style to matplotlib axes
 
 Parameters
 ----------
 ax : matplotlib.Axes
 Axes to style
 remove_spines : tuple
Which spines to remove ('top', 'right', 'left', 'bottom')
"""
    # Remove gridlines
    ax.grid(False)

    # Remove specified spines
    for spine in remove_spines:
        ax.spines[spine].set_visible(False)

    # Make remaining spines thinner
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    # Clean up ticks
    ax.tick_params(axis='both', which='both', length=4, width=0.8,
    color='#333333', labelsize=10)

    # Remove tick marks from hidden spines
    if 'top' in remove_spines:
        ax.tick_params(axis='x', top=False)
    if 'right' in remove_spines:
        ax.tick_params(axis='y', right=False)

def set_minimalist_rcparams():
    mpl.rcParams.update({
        # Figure
        'figure.facecolor': 'white',
    'figure.edgecolor': 'none',
    'figure.dpi': 100,

    # Axes
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.titlesize': 12,
    'axes.titleweight': 'normal',
    'axes.labelsize': 10,
    'axes.labelcolor': '#333333',

    # Grid
    'grid.alpha': 0.3,
    'grid.color': '#CCCCCC',
    'grid.linewidth': 0.5,

    # Ticks
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.top': False,
    'ytick.right': False,

    # Legend
    'legend.frameon': False,
    'legend.fontsize': 9,
    'legend.title_fontsize': 10,

    # Lines
    'lines.linewidth': 1.5,
    'lines.markersize': 6,

    # Fonts
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    })

def create_minimalist_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create figure with minimalist style applied
 
 Parameters
 ----------
 nrows, ncols : int
 Number of subplot rows and columns
 figsize : tuple, optional
 Figure size
**kwargs
    Additional arguments passed to plt.subplots

Returns
-------
    fig, ax or axes
    Figure and axes objects
"""
    if figsize is None:
        figsize = (8, 6)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Apply style to all axes
    import numpy as np
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            apply_minimalist_style(ax)
    else:
        apply_minimalist_style(axes)

    return fig, axes
