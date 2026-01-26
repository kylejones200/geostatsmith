"""
Automatic Method Selection
===========================

Automatically select best interpolation method.
"""

import logging
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Optional, Tuple

from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..core.exceptions import FittingError
from .auto_variogram import auto_variogram

logger = logging.getLogger(__name__)


def auto_interpolate(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    x_pred: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    methods: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Automatically select best interpolation method and predict.
    
    Tries multiple methods via cross-validation and selects the best.
    
    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    x_pred, y_pred : ndarray
        Prediction locations
    methods : list of str, optional
        Methods to try. Default: ['ordinary_kriging', 'idw', 'rbf']
    verbose : bool, default=True
        Print selection process
    
    Returns
    -------
    results : dict
        Dictionary with best method, model, predictions, CV scores
    
    Examples
    --------
    >>> from geostats.automl import auto_interpolate
    >>> 
    >>> # One function does everything!
    >>> results = auto_interpolate(x, y, z, x_pred, y_pred)
    >>> 
    >>> print(f"Best method: {results['best_method']}")
    >>> print(f"CV RMSE: {results['cv_rmse']:.3f}")
    >>> z_pred = results['predictions']
    
    Notes
    -----
    This is the ultimate convenience function - automatically:
    1. Fits variogram (if kriging)
    2. Tries multiple methods
    3. Cross-validates each
    4. Selects best
    5. Makes final predictions
    """
    if methods is None:
        methods = ['ordinary_kriging']
    
    if verbose:
        logger.info("=" * 60)
        logger.info("AUTO-INTERPOLATE: Automatic Method Selection")
        logger.info("=" * 60)
    
    best_method = None
    best_rmse = np.inf
    best_predictions = None
    best_model = None
    all_results = {}
    
    for method in methods:
        if verbose:
            logger.info(f"\nTesting: {method}")
        
        try:
            if method == 'ordinary_kriging':
                model = auto_variogram(x, y, z, verbose=False)
                
                n = len(x)
                cv_preds = np.zeros(n)
                
                for i in range(n):
                    train_idx = np.delete(np.arange(n), i)
                    krig = OrdinaryKriging(x[train_idx], y[train_idx], z[train_idx], model)
                    pred, _ = krig.predict(np.array([x[i]]), np.array([y[i]]), return_variance=True)
                    cv_preds[i] = pred[0]
                
                errors = z - cv_preds
                rmse = np.sqrt(np.mean(errors**2))
                
                krig_final = OrdinaryKriging(x, y, z, model)
                predictions, variance = krig_final.predict(x_pred, y_pred, return_variance=True)
                
                all_results[method] = {
                    'rmse': rmse,
                    'predictions': predictions,
                    'variance': variance,
                    'model': model
                }
                
                if verbose:
                    logger.info(f"  CV RMSE: {rmse:.4f}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_method = method
                    best_predictions = predictions
                    best_model = model
            
            elif method == 'idw':
                try:
                    from ..comparison.method_implementations import InverseDistanceWeighting
                    idw = InverseDistanceWeighting(x, y, z, power=2)
                    
                    n = len(x)
                    cv_preds = np.zeros(n)
                    for i in range(n):
                        mask = np.ones(n, dtype=bool)
                        mask[i] = False
                        idw_cv = InverseDistanceWeighting(x[mask], y[mask], z[mask], power=2)
                        cv_preds[i] = idw_cv.predict(np.array([x[i]]), np.array([y[i]]))[0]
                    
                    errors = z - cv_preds
                    rmse = np.sqrt(np.mean(errors**2))
                    
                    predictions = idw.predict(x_pred, y_pred)
                    
                    all_results[method] = {
                        'rmse': rmse,
                        'predictions': predictions,
                    }
                    
                    if verbose:
                        logger.info(f"  CV RMSE: {rmse:.4f}")
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_method = method
                        best_predictions = predictions
                        best_model = None
                
                except ImportError:
                    if verbose:
                        logger.warning(
                            "IDW interpolation not available: comparison module not installed. "
                            "Install with: pip install geostats[comparison]"
                        )
        
        except Exception as e:
            if verbose:
                logger.error(f"Method '{method}' failed: {str(e)}")
            continue
    
    if best_method is None:
        raise FittingError(
            f"All {len(methods)} interpolation methods failed. "
            "Check data quality (sufficient points, no duplicates, valid values)."
        )
    
    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info(f"BEST METHOD: {best_method}")
        logger.info(f"   CV RMSE: {best_rmse:.4f}")
        logger.info("=" * 60)
    
    return {
        'best_method': best_method,
        'cv_rmse': best_rmse,
        'predictions': best_predictions,
        'model': best_model,
        'all_results': all_results,
    }


def suggest_method(
    n_samples: int,
    n_predictions: int,
    data_characteristics: Optional[Dict] = None,
) -> str:
    """
    Suggest best interpolation method based on data characteristics.
    
    Parameters
    ----------
    n_samples : int
        Number of sample points
    n_predictions : int
        Number of prediction points
    data_characteristics : dict, optional
        Dictionary with keys like 'has_trend', 'is_clustered', etc.
    
    Returns
    -------
    suggestion : str
        Suggested method name
    
    Examples
    --------
    >>> method = suggest_method(n_samples=100, n_predictions=10000)
    >>> print(f"Suggested: {method}")
    """
    # Simple heuristics
    if n_samples < 20:
        return 'idw'  # Too few for good variogram
    elif n_samples < 100:
        return 'ordinary_kriging'  # Good default
    elif n_predictions > 50000:
        return 'approximate_kriging'  # Need speed
    else:
        return 'ordinary_kriging'  # Default to best method
