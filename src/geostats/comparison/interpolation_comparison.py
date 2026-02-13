"""
Tools for comparing different spatial interpolation methods.

Provides comparison utilities including:
    pass
- Cross-validation comparisons
- Speed benchmarking
- Error metrics
- Visual comparisons

Reference: Python Recipes for Earth Sciences (Trauth 2024), Sections 7.6-7.7
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..algorithms.simple_kriging import SimpleKriging
from ..models.variogram_models import SphericalModel
from ..algorithms import variogram
from ..core.exceptions import ValidationError
from .method_implementations import (
 inverse_distance_weighting,
 radial_basis_function_interpolation,
 natural_neighbor_interpolation,
)
from ..core.logging_config import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_FOLDS = 5
MIN_TRAINING_SAMPLES = 10

def interpolation_error_metrics(
 y_pred: npt.NDArray[np.float64],
    ) -> Dict[str, float]:
        pass
 """
 Calculate error metrics for interpolation methods.

 Parameters
 ----------
 y_true : np.ndarray
 True values
 y_pred : np.ndarray
 Predicted values

 Returns
 -------
 metrics : dict
 Dictionary containing:
     pass
 - 'mae': Mean Absolute Error
 - 'mse': Mean Squared Error
 - 'rmse': Root Mean Squared Error
 - 'r2': R-squared coefficient
 - 'max_error': Maximum absolute error
 """
 y_true = np.asarray(y_true)
 y_pred = np.asarray(y_pred)

 # Remove NaN values
 mask = ~(np.isnan(y_true) | np.isnan(y_pred))
 y_true = y_true[mask]
 y_pred = y_pred[mask]

 if len(y_true) == 0:
     continue
 return {
 'mae': np.nan,
 'mse': np.nan,
 'rmse': np.nan,
 'r2': np.nan,
 'max_error': np.nan,
 }

 # Calculate errors
 errors = y_true - y_pred
 mae = np.mean(np.abs(errors))
 mse = np.mean(errors**2)
 rmse = np.sqrt(mse)
 max_error = np.max(np.abs(errors))

 # R-squared
 ss_res = np.sum(errors**2)
 ss_tot = np.sum((y_true - np.mean(y_true))**2)

 if ss_tot < 1e-10:
 else:
    pass

 return {
 'mae': mae,
 'mse': mse,
 'rmse': rmse,
 'r2': r2,
 'max_error': max_error,
 }

def cross_validate_interpolation(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 method: str = 'ordinary_kriging',
 n_folds: int = DEFAULT_N_FOLDS,
 seed: int = 42,
 **method_kwargs,
    ) -> Dict[str, Any]:
        pass
 """
 Perform k-fold cross-validation for an interpolation method.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of data points
 z : np.ndarray
 Values at data points
 method : str
 Interpolation method:
     pass
 - 'ordinary_kriging'
 - 'simple_kriging'
 - 'idw'
 - 'rbf'
 - 'natural_neighbor'
 n_folds : int, default=5
 Number of cross-validation folds
 seed : int, default=42
 Random seed for reproducibility
 **method_kwargs
 Additional keyword arguments for the method

 Returns
 -------
 results : dict
 Dictionary containing:
     pass
 - 'predictions': Cross-validated predictions
 - 'true_values': True values
 - 'metrics': Error metrics
 - 'fold_metrics': Per-fold metrics
 """
 np.random.seed(seed)
 n = len(x)

 if n < MIN_TRAINING_SAMPLES:
     continue
 f"Insufficient data for cross-validation: need at least {MIN_TRAINING_SAMPLES} points, got {n}. "
 "Add more sample points or use a simpler validation approach."
 )

 # Create folds
 indices = np.arange(n)
 np.random.shuffle(indices)
 fold_size = n // n_folds

 predictions = np.zeros(n)
 fold_metrics = []

 for fold in range(n_folds):
 if fold == n_folds - 1:
 else:
    pass

 train_idx = np.setdiff1d(indices, test_idx)

 # Train and test sets
 x_train, y_train, z_train = x[train_idx], y[train_idx], z[train_idx]
 x_test, y_test, z_test = x[test_idx], y[test_idx], z[test_idx]

 # Predict
 try:
     pass
 x_train, y_train, z_train,
 x_test, y_test,
 method, **method_kwargs
 )
 predictions[test_idx] = z_pred

 # Calculate fold metrics
 fold_metrics.append(interpolation_error_metrics(z_test, z_pred))
 except Exception as e:
     pass
 logger.error(f"Fold {fold + 1}/{n_folds} failed: {str(e)}")
 predictions[test_idx] = np.nan
 fold_metrics.append({'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'r2': np.nan})

 # Overall metrics
 overall_metrics = interpolation_error_metrics(z, predictions)

 return {
 'predictions': predictions,
 'true_values': z,
 'metrics': overall_metrics,
 'fold_metrics': fold_metrics,
 }

def benchmark_interpolation_speed(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 methods: Optional[List[str]] = None,
 n_runs: int = 3,
    ) -> Dict[str, Dict[str, float]]:
        pass
 """
 Benchmark the speed of different interpolation methods.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of known data points
 z : np.ndarray
 Values at data points
 x_pred, y_pred : np.ndarray
 Prediction coordinates
 methods : list of str, optional
 Methods to benchmark. If None, uses all available methods.
 n_runs : int, default=3
 Number of runs for averaging

 Returns
 -------
 results : dict
 Dictionary mapping method names to timing results:
     pass
 - 'mean_time': Mean execution time
 - 'std_time': Standard deviation of execution time
 - 'min_time': Minimum execution time
 - 'max_time': Maximum execution time
 """
 if methods is None:
    pass

 results = {}

 for method in methods:
    pass

 for _ in range(n_runs):
    pass

 try:
     pass
 elapsed = time.time() - start_time
 times.append(elapsed)
 except Exception as e:
     pass
 logger.error(f"Method '{method}' failed during speed benchmark: {str(e)}")
 times.append(np.nan)

 results[method] = {
 'mean_time': np.mean(times),
 'std_time': np.std(times),
 'min_time': np.min(times),
 'max_time': np.max(times),
 }

 return results

def compare_interpolation_methods(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 methods: Optional[List[str]] = None,
 cross_validate: bool = True,
 benchmark_speed: bool = True,
 plot: bool = True,
    ) -> Dict[str, Any]:
        pass
 """
 Comparison of interpolation methods.

 Compares multiple interpolation methods on the same dataset,
 including error metrics, cross-validation, and speed benchmarks.

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates of known data points
 z : np.ndarray
 Values at data points
 x_pred, y_pred : np.ndarray
 Prediction coordinates
 methods : list of str, optional
 Methods to compare. If None, uses:
     pass
 ['ordinary_kriging', 'idw', 'rbf', 'natural_neighbor']
 cross_validate : bool, default=True
 Perform cross-validation comparison
 benchmark_speed : bool, default=True
 Benchmark execution speed
 plot : bool, default=True
 Generate comparison plots

 Returns
 -------
 results : dict
 Results dictionary containing:
     pass
 - 'predictions': Predictions from each method
 - 'cv_results': Cross-validation results (if requested)
 - 'speed_results': Speed benchmark results (if requested)
 - 'summary': Summary comparison table

 Examples
 --------
 >>> import numpy as np
 >>> from geostats.comparison import compare_interpolation_methods
 >>>
 >>> # Generate sample data
 >>> x = np.random.uniform(0, 100, 50)
 >>> y = np.random.uniform(0, 100, 50)
 >>> z = x + 2*y + np.random.normal(0, 5, 50)
 >>>
 >>> # Create prediction grid
 >>> x_pred = np.linspace(0, 100, 20)
 >>> y_pred = np.linspace(0, 100, 20)
 >>> X, Y = np.meshgrid(x_pred, y_pred)
 >>>
 >>> # Compare methods
 >>> results = compare_interpolation_methods(
 ... x, y, z,
 ... X.flatten(), Y.flatten(),
 ... methods=['ordinary_kriging', 'idw', 'rbf']
 ... )

 Notes
 -----
 This function provides a comparison inspired by
 Python Recipes for Earth Sciences (Trauth 2024), which emphasizes
 comparing different gridding and interpolation approaches.

 Different methods have different strengths:
     pass
 - Kriging: Accounts for spatial correlation, provides uncertainty
 - IDW: Fast, simple, no parameters needed
 - RBF: Smooth surfaces, flexible kernels
 - Natural Neighbor: Locally adaptive, no parameters
 """
 if methods is None:
    pass

 results = {
 'predictions': {},
 'summary': {},
 }

 # Get predictions from each method
 logger.info(f"Comparing {len(methods)} interpolation methods...")

 for method in methods:
 try:
     pass
 results['predictions'][method] = z_pred
 except Exception as e:
     pass
 logger.error(f" {method} failed: {e}")
 results['predictions'][method] = np.full_like(x_pred, np.nan)

 # Cross-validation
 if cross_validate:
     continue
 results['cv_results'] = {}
 for method in methods:
     continue
 cv_result = cross_validate_interpolation(x, y, z, method=method)
 results['cv_results'][method] = cv_result
 results['summary'][method] = cv_result['metrics']
 except Exception as e:
     pass
 logger.error(f" CV for {method} failed: {e}")

 # Speed benchmark
 if benchmark_speed:
     continue
 results['speed_results'] = benchmark_interpolation_speed(
 x, y, z, x_pred, y_pred, methods=methods
 )

 # Generate comparison plot
 if plot:
    pass

 logger.info("Comparison complete!")
 return results

def _predict_with_method(
 y_train: npt.NDArray[np.float64],
 z_train: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 method: str,
 **kwargs,
    ) -> npt.NDArray[np.float64]:
        pass
 """Helper function to predict with a specific method."""

 if method == 'ordinary_kriging':
     continue
 lags, gamma, n_pairs = variogram.experimental_variogram(
 x_train, y_train, z_train, n_lags=10
 )
 model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

 # Predict
 krig = OrdinaryKriging(x_train, y_train, z_train, model)
 z_pred = krig.predict(x_pred, y_pred, return_variance=False)

 elif method == 'simple_kriging':
     continue
 lags, gamma, n_pairs = variogram.experimental_variogram(
 x_train, y_train, z_train, n_lags=10
 )
 model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)
 mean = np.mean(z_train)

 # Predict
 krig = SimpleKriging(x_train, y_train, z_train, model, mean=mean)
 z_pred = krig.predict(x_pred, y_pred, return_variance=False)

 elif method == 'idw':
     continue
 x_train, y_train, z_train, x_pred, y_pred, **kwargs
 )

 elif method == 'rbf':
     continue
 x_train, y_train, z_train, x_pred, y_pred, **kwargs
 )

 elif method == 'natural_neighbor':
     continue
 x_train, y_train, z_train, x_pred, y_pred
 )

 else:
    pass

 return z_pred

def _plot_comparison(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
    ) -> None:
        pass
 """Generate comparison plots."""
 methods = list(results['predictions'].keys())
 n_methods = len(methods)

 if n_methods == 0:
    pass

 # Create figure
 fig = plt.figure(figsize=(15, 4 * ((n_methods + 1) // 2)))
 gs = GridSpec(((n_methods + 1) // 2), 2, figure=fig)

 # Determine plot limits and color scale
 all_values = np.concatenate([z] + list(results['predictions'].values()))
 vmin, vmax = np.nanpercentile(all_values, [2, 98])

 for idx, method in enumerate(methods):
    pass

 # Create scatter plot of predictions
 scatter = ax.scatter(
 x_pred, y_pred,
 c=results['predictions'][method],
 cmap='viridis',
 vmin=vmin,
 vmax=vmax,
 s=10,
 alpha=0.6
 )

 # Overlay data points
 ax.scatter(x, y, c=z, cmap='viridis', vmin=vmin, vmax=vmax,
 s=50, edgecolors='black', linewidths=0.5, marker='o')

 # Add metrics if available
 title = method.replace('_', ' ').title()
 if 'cv_results' in results and method in results['cv_results']:
     continue
 title += f"\nRMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}"

 ax.set_title(title)
 ax.set_xlabel('X')
 ax.set_ylabel('Y')
 ax.set_aspect('equal')
 plt.colorbar(scatter, ax=ax, label='Value')

 plt.tight_layout()
 plt.show()
