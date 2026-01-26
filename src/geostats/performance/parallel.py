"""
Parallel Processing for Kriging
================================

Multi-core parallel implementations of kriging and related operations.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List
from joblib import Parallel, delayed
import multiprocessing

from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..models.base_model import VariogramModelBase
from ..algorithms.variogram import experimental_variogram
from ..algorithms.fitting import fit_variogram_model as fit_variogram


def parallel_kriging(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    x_pred: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    variogram_model: VariogramModelBase,
    n_jobs: int = -1,
    batch_size: int = 1000,
    return_variance: bool = True,
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
    """
    Perform kriging in parallel across multiple CPU cores.
    
    Splits prediction locations into batches and processes them in parallel.
    
    Parameters
    ----------
    x : ndarray
        Sample X coordinates
    y : ndarray
        Sample Y coordinates
    z : ndarray
        Sample values
    x_pred : ndarray
        Prediction X coordinates
    y_pred : ndarray
        Prediction Y coordinates
    variogram_model : VariogramModelBase
        Fitted variogram model
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all CPU cores.
    batch_size : int, default=1000
        Number of prediction points per batch
    return_variance : bool, default=True
        Whether to return kriging variance
    
    Returns
    -------
    predictions : ndarray
        Predicted values
    variance : ndarray, optional
        Kriging variance (if return_variance=True)
    
    Examples
    --------
    >>> from geostats.performance import parallel_kriging
    >>> 
    >>> # Process 100k prediction points in parallel
    >>> z_pred, var = parallel_kriging(
    ...     x, y, z,
    ...     x_pred, y_pred,
    ...     variogram_model=model,
    ...     n_jobs=-1  # Use all CPU cores
    ... )
    
    Notes
    -----
    Speedup is typically 2-8x depending on number of cores and dataset size.
    Most beneficial for large prediction grids (>10,000 points).
    """
    n_pred = len(x_pred)
    
    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Create batches
    n_batches = int(np.ceil(n_pred / batch_size))
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_pred)
        batches.append((x_pred[start:end], y_pred[start:end], start, end))
    
    # Process batches in parallel
    def process_batch(x_batch, y_batch, start, end):
        krig = OrdinaryKriging(
            x=x,
            y=y,
            z=z,
            variogram_model=variogram_model,
        )
        return krig.predict(x_batch, y_batch, return_variance=return_variance)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(x_b, y_b, s, e) 
        for x_b, y_b, s, e in batches
    )
    
    # Combine results
    if return_variance:
        predictions = np.concatenate([r[0] for r in results])
        variance = np.concatenate([r[1] for r in results])
        return predictions, variance
    else:
        predictions = np.concatenate([r for r in results])
        return predictions, None


def parallel_cross_validation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    variogram_model: VariogramModelBase,
    method: str = 'leave_one_out',
    n_folds: int = 5,
    n_jobs: int = -1,
) -> dict:
    """
    Perform cross-validation in parallel.
    
    Parameters
    ----------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    z : ndarray
        Values
    variogram_model : VariogramModelBase
        Variogram model
    method : str, default='leave_one_out'
        Cross-validation method: 'leave_one_out' or 'k_fold'
    n_folds : int, default=5
        Number of folds for k-fold CV
    n_jobs : int, default=-1
        Number of parallel jobs
    
    Returns
    -------
    results : dict
        Cross-validation results with predictions and errors
    
    Examples
    --------
    >>> results = parallel_cross_validation(
    ...     x, y, z,
    ...     variogram_model=model,
    ...     method='k_fold',
    ...     n_folds=10,
    ...     n_jobs=-1
    ... )
    >>> print(f"RMSE: {results['rmse']:.3f}")
    """
    n = len(x)
    
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    def predict_single(i, train_indices):
        """Predict for a single test point."""
        x_train = x[train_indices]
        y_train = y[train_indices]
        z_train = z[train_indices]
        
        krig = OrdinaryKriging(
            x=x_train,
            y=y_train,
            z=z_train,
            variogram_model=variogram_model,
        )
        
        z_pred, var = krig.predict(
            np.array([x[i]]),
            np.array([y[i]]),
            return_variance=True
        )
        
        return z_pred[0], var[0]
    
    if method == 'leave_one_out':
        # Leave-one-out CV
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(predict_single)(i, np.delete(np.arange(n), i))
            for i in range(n)
        )
        
        predictions = np.array([r[0] for r in results_list])
        variances = np.array([r[1] for r in results_list])
    
    elif method == 'k_fold':
        # K-fold CV
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        predictions = np.zeros(n)
        variances = np.zeros(n)
        
        for train_idx, test_idx in kf.split(x):
            # Process fold in parallel
            fold_results = Parallel(n_jobs=n_jobs)(
                delayed(predict_single)(i, train_idx)
                for i in test_idx
            )
            
            predictions[test_idx] = [r[0] for r in fold_results]
            variances[test_idx] = [r[1] for r in fold_results]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute metrics
    errors = z - predictions
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    r2 = 1 - np.sum(errors**2) / np.sum((z - z.mean())**2)
    
    return {
        'predictions': predictions,
        'variances': variances,
        'errors': errors,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'observed': z,
    }


def parallel_variogram_fit(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    model_types: List[str] = None,
    n_lags: int = 15,
    n_jobs: int = -1,
) -> dict:
    """
    Fit multiple variogram models in parallel and select best.
    
    Parameters
    ----------
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    z : ndarray
        Values
    model_types : list of str, optional
        Variogram models to try. Default: ['spherical', 'exponential', 'gaussian']
    n_lags : int, default=15
        Number of lags
    n_jobs : int, default=-1
        Number of parallel jobs
    
    Returns
    -------
    results : dict
        Dictionary with best model and all fitted models
    
    Examples
    --------
    >>> results = parallel_variogram_fit(
    ...     x, y, z,
    ...     model_types=['spherical', 'exponential', 'gaussian', 'linear'],
    ...     n_jobs=-1
    ... )
    >>> best_model = results['best_model']
    >>> print(f"Best: {results['best_type']} (R² = {results['best_r2']:.3f})")
    """
    if model_types is None:
        model_types = ['spherical', 'exponential', 'gaussian']
    
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Compute experimental variogram once
    lags, gamma = experimental_variogram(x, y, z, n_lags=n_lags)
    
    # Fit models in parallel
    def fit_single_model(model_type):
        try:
            model = fit_variogram(lags, gamma, model_type=model_type)
            
            # Compute R²
            gamma_fitted = model(lags)
            ss_res = np.sum((gamma - gamma_fitted)**2)
            ss_tot = np.sum((gamma - gamma.mean())**2)
            r2 = 1 - ss_res / ss_tot
            
            return {
                'model': model,
                'type': model_type,
                'r2': r2,
                'success': True
            }
        except Exception as e:
            return {
                'model': None,
                'type': model_type,
                'r2': -np.inf,
                'success': False,
                'error': str(e)
            }
    
    fit_results = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_model)(mt) for mt in model_types
    )
    
    # Find best model
    successful = [r for r in fit_results if r['success']]
    
    if not successful:
        raise RuntimeError("All variogram fits failed")
    
    best = max(successful, key=lambda x: x['r2'])
    
    return {
        'best_model': best['model'],
        'best_type': best['type'],
        'best_r2': best['r2'],
        'all_results': fit_results,
        'lags': lags,
        'gamma': gamma,
    }
