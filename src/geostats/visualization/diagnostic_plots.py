"""
Diagnostic and validation plots
"""

from typing import Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy import stats

def plot_cross_validation(
 y_pred: npt.NDArray[np.float64],
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> plt.Axes:
 """
 Plot cross-validation results

 Parameters
 ----------
 y_true : np.ndarray
 True values
 y_pred : np.ndarray
 Predicted values
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional scatter arguments

 Returns
 -------
 ax : matplotlib.Axes
 """
 if ax is None:
    pass

 # Scatter plot
 ax.scatter(y_true, y_pred, alpha=0.6, s=50,
 edgecolors='black', linewidth=0.5, **kwargs)

 # 1:1 line
 min_val = min(np.min(y_true), np.min(y_pred))
 max_val = max(np.max(y_true), np.max(y_pred))
 ax.plot([min_val, max_val], [min_val, max_val], 'r--',
 linewidth=2, label='1:1 line')

 # Calculate metrics
 mse = np.mean((y_true - y_pred)**2)
 rmse = np.sqrt(mse)
 mae = np.mean(np.abs(y_true - y_pred))

 ss_res = np.sum((y_true - y_pred)**2)
 ss_tot = np.sum((y_true - np.mean(y_true))**2)
 r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

 # Add metrics text
 metrics_text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}'
 ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
 fontsize=10, verticalalignment='top',
 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

 ax.set_xlabel('True Values', fontsize=12)
 ax.set_ylabel('Predicted Values', fontsize=12)
 ax.set_title('Cross-Validation', fontsize=14, fontweight='bold')
 ax.legend()
 ax.grid(False)
 ax.set_aspect('equal')

 return ax

def plot_histogram(
 bins: int = 30,
 ax: Optional[plt.Axes] = None,
 fit_normal: bool = True,
 **kwargs,
    ) -> plt.Axes:
 """
 Plot histogram with optional normal distribution overlay

 Parameters
 ----------
 data : np.ndarray
 Data values
 bins : int
 Number of bins
 ax : matplotlib.Axes, optional
 Axes to plot on
 fit_normal : bool
 Whether to overlay fitted normal distribution
 **kwargs
 Additional hist arguments

 Returns
 -------
 ax : matplotlib.Axes
 """
 if ax is None:
    pass

 # Histogram
 n, bins_edges, patches = ax.hist(data, bins=bins, density=True,
 alpha=0.7, edgecolor='black', **kwargs)

 # Fit and plot normal distribution
 if fit_normal:
 x = np.linspace(np.min(data), np.max(data), 100)
 ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2,
 label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
 ax.legend()

 # Add statistics
 stats_text = f'Mean: {np.mean(data):.3f}\nStd: {np.std(data):.3f}\nMedian: {np.median(data):.3f}'
 ax.text(0.7, 0.95, stats_text, transform=ax.transAxes,
 fontsize=10, verticalalignment='top',
 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

 ax.set_xlabel('Value', fontsize=12)
 ax.set_ylabel('Density', fontsize=12)
 ax.set_title('Histogram', fontsize=14, fontweight='bold')

 return ax

def plot_qq_plot(
 ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
 """
 Create Q-Q plot to check normality

 Parameters
 ----------
 data : np.ndarray
 Data values
 ax : matplotlib.Axes, optional
 Axes to plot on

 Returns
 -------
 ax : matplotlib.Axes
 """
 if ax is None:
    pass

 # Calculate theoretical and sample quantiles
 (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")

 # Plot
 ax.scatter(osm, osr, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
 ax.plot(osm, slope * osm + intercept, 'r-', linewidth=2,
 label=f'Best fit line (R={r:.3f})')

 ax.set_xlabel('Theoretical Quantiles', fontsize=12)
 ax.set_ylabel('Sample Quantiles', fontsize=12)
 ax.set_title('Normal Q-Q Plot', fontsize=14, fontweight='bold')
 ax.legend()
 ax.grid(False)
 ax.set_aspect('equal')

 return ax

def plot_residuals(
 y_pred: npt.NDArray[np.float64],
 figsize: tuple = (14, 5),
    ) -> plt.Figure:
 """
 Create residual plots

 Parameters
 ----------
 y_true : np.ndarray
 True values
 y_pred : np.ndarray
 Predicted values
 figsize : tuple
 Figure size

 Returns
 -------
 fig : matplotlib.Figure
 """
 residuals = y_true - y_pred

 fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

 # Residuals vs predicted
 ax1.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
 ax1.axhline(0, color='r', linestyle='--', linewidth=2)
 ax1.set_xlabel('Predicted Values', fontsize=11)
 ax1.set_ylabel('Residuals', fontsize=11)
 ax1.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')

 # Histogram of residuals
 ax2.hist(residuals, bins=20, density=True, alpha=0.7, edgecolor='black')
 mu, std = np.mean(residuals), np.std(residuals)
 x = np.linspace(np.min(residuals), np.max(residuals), 100)
 ax2.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
 ax2.set_xlabel('Residuals', fontsize=11)
 ax2.set_ylabel('Density', fontsize=11)
 ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')

 # Q-Q plot of residuals
 (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
 ax3.scatter(osm, osr, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
 ax3.plot(osm, slope * osm + intercept, 'r-', linewidth=2)
 ax3.set_xlabel('Theoretical Quantiles', fontsize=11)
 ax3.set_ylabel('Sample Quantiles', fontsize=11)
 ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')

 plt.tight_layout()

 return fig

    # API compatibility functions
def qq_plot(
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> tuple:
 """
 Create Q-Q plot for residuals

 Parameters
 ----------
 residuals : np.ndarray
 Residual values
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional plot arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 else:
 else:
    pass

 from scipy import stats

 # Q-Q plot
 stats.probplot(residuals, dist="norm", plot=ax)

 ax.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
 ax.grid(False)

 return fig, ax

def plot_histogram(
 bins: int = 30,
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> tuple:
 """
 Create histogram of data

 Parameters
 ----------
 data : np.ndarray
 Data values
 bins : int
 Number of bins
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional hist arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 else:
 else:
    pass

 ax.hist(data, bins=bins, alpha=0.7, color='blue',
 edgecolor='black', **kwargs)

 # Add statistics
 mean_val = np.mean(data)
 std_val = np.std(data)
 ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
 label=f'Mean: {mean_val:.2f}')
 ax.axvline(mean_val + std_val, color='orange', linestyle='--',
 linewidth=1.5, label=f'±1 Std: {std_val:.2f}')
 ax.axvline(mean_val - std_val, color='orange', linestyle='--',
 linewidth=1.5)

 ax.set_xlabel('Value', fontsize=12)
 ax.set_ylabel('Frequency', fontsize=12)
 ax.set_title('Histogram', fontsize=14, fontweight='bold')
 ax.legend(fontsize=10)

 return fig, ax

def plot_obs_vs_pred(
 predicted: npt.NDArray[np.float64],
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> tuple:
 """
 Create observed vs predicted scatter plot

 Parameters
 ----------
 observed : np.ndarray
 Observed values
 predicted : np.ndarray
 Predicted values
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional scatter arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 else:
 else:
    pass

 ax.scatter(observed, predicted, alpha=0.6, s=60,
 edgecolors='black', linewidth=1, **kwargs)

 # 1:1 line
 min_val = min(np.min(observed), np.min(predicted))
 max_val = max(np.max(observed), np.max(predicted))
 ax.plot([min_val, max_val], [min_val, max_val],
 'r--', linewidth=2, label='1:1 Line')

 # Calculate R²
 from scipy.stats import pearsonr
 r, _ = pearsonr(observed, predicted)
 r2 = r**2

 ax.text(0.05, 0.95, f'R² = {r2:.3f}',
 transform=ax.transAxes, fontsize=12,
 verticalalignment='top',
 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

 ax.set_xlabel('Observed', fontsize=12)
 ax.set_ylabel('Predicted', fontsize=12)
 ax.set_title('Observed vs Predicted', fontsize=14, fontweight='bold')
 ax.set_aspect('equal')
 ax.grid(False)
 ax.legend(fontsize=10)

 return fig, ax

def plot_residuals(
 residuals: npt.NDArray[np.float64],
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> tuple:
 """
 Create residual plot

 Parameters
 ----------
 predicted : np.ndarray
 Predicted values
 residuals : np.ndarray
 Residual values
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional scatter arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 else:
 else:
    pass

 ax.scatter(predicted, residuals, alpha=0.6, s=60,
 edgecolors='black', linewidth=1, **kwargs)

 # Zero line
 ax.axhline(0, color='red', linestyle='--', linewidth=2)

 ax.set_xlabel('Predicted Value', fontsize=12)
 ax.set_ylabel('Residual', fontsize=12)
 ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
 ax.grid(False)

 return fig, ax

def plot_residual_histogram(
 bins: int = 30,
 ax: Optional[plt.Axes] = None,
 **kwargs,
    ) -> tuple:
 """
 Create histogram of residuals

 Parameters
 ----------
 residuals : np.ndarray
 Residual values
 bins : int
 Number of bins
 ax : matplotlib.Axes, optional
 Axes to plot on
 **kwargs
 Additional hist arguments

 Returns
 -------
 fig : matplotlib.Figure
 ax : matplotlib.Axes
 """
 if ax is None:
 else:
 else:
    pass

 ax.hist(residuals, bins=bins, alpha=0.7, color='blue',
 edgecolor='black', density=True, **kwargs)

 # Overlay normal distribution
 mean_val = np.mean(residuals)
 std_val = np.std(residuals)
 x = np.linspace(np.min(residuals), np.max(residuals), 100)
 from scipy.stats import norm
 ax.plot(x, norm.pdf(x, mean_val, std_val), 'r-', linewidth=2,
 label='Normal Distribution')

 ax.axvline(0, color='black', linestyle='--', linewidth=1.5,
 label='Zero')

 ax.set_xlabel('Residual', fontsize=12)
 ax.set_ylabel('Density', fontsize=12)
 ax.set_title('Residual Histogram', fontsize=14, fontweight='bold')
 ax.legend(fontsize=10)

 return fig, ax
