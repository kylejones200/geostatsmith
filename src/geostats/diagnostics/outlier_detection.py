"""
Outlier Detection and Robust Validation
========================================

Tools for detecting outliers and robust validation.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Tuple


def outlier_analysis(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    method: str = 'iqr',
    threshold: float = 3.0,
) -> Dict:
    """
    Detect potential outliers in spatial data.
    
    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    method : str, default='iqr'
        Outlier detection method: 'iqr', 'zscore', or 'spatial'
    threshold : float
        Threshold for outlier detection
    
    Returns
    -------
    results : dict
        Outlier analysis results including:
        - outlier_indices: Indices of potential outliers
        - outlier_scores: Outlier scores
        - n_outliers: Number of outliers detected
    
    Examples
    --------
    >>> from geostats.diagnostics import outlier_analysis
    >>> results = outlier_analysis(x, y, z, method='zscore', threshold=3.0)
    >>> print(f"Found {results['n_outliers']} potential outliers")
    """
    if method == 'iqr':
        # Interquartile range method
        q1 = np.percentile(z, 25)
        q3 = np.percentile(z, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = (z < lower) | (z > upper)
        scores = np.abs(z - np.median(z)) / iqr
    
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs((z - z.mean()) / z.std())
        outliers = z_scores > threshold
        scores = z_scores
    
    elif method == 'spatial':
        # Spatial outliers (points very different from neighbors)
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([x, y]))
        
        scores = np.zeros(len(z))
        for i in range(len(z)):
            # Find 5 nearest neighbors
            distances, indices = tree.query([x[i], y[i]], k=6)  # 6 because includes self
            neighbor_indices = indices[1:]  # Exclude self
            
            # Compare to neighbor mean
            neighbor_mean = z[neighbor_indices].mean()
            neighbor_std = z[neighbor_indices].std()
            if neighbor_std > 0:
                scores[i] = abs(z[i] - neighbor_mean) / neighbor_std
            else:
                scores[i] = 0
        
        outliers = scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'outlier_indices': np.where(outliers)[0].tolist(),
        'outlier_scores': scores.tolist(),
        'n_outliers': int(outliers.sum()),
        'method': method,
        'threshold': threshold,
    }


def robust_validation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    variogram_model,
    outlier_method: str = 'iqr',
) -> Dict:
    """
    Validation with outlier detection and removal.
    
    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    variogram_model : VariogramModelBase
        Variogram model
    outlier_method : str
        Outlier detection method
    
    Returns
    -------
    results : dict
        Validation results with and without outliers
    """
    from ..algorithms.ordinary_kriging import OrdinaryKriging
    
    # Standard validation
    n = len(x)
    predictions_all = np.zeros(n)
    for i in range(n):
        train_idx = np.delete(np.arange(n), i)
        krig = OrdinaryKriging(x[train_idx], y[train_idx], z[train_idx], variogram_model)
        pred, _ = krig.predict(np.array([x[i]]), np.array([y[i]]), return_variance=True)
        predictions_all[i] = pred[0]
    
    errors_all = z - predictions_all
    rmse_all = np.sqrt(np.mean(errors_all**2))
    
    # Detect outliers
    outliers = outlier_analysis(x, y, z, method=outlier_method)
    outlier_idx = outliers['outlier_indices']
    
    # Validation without outliers
    if len(outlier_idx) > 0:
        mask = np.ones(n, dtype=bool)
        mask[outlier_idx] = False
        x_clean = x[mask]
        y_clean = y[mask]
        z_clean = z[mask]
        
        n_clean = len(x_clean)
        predictions_clean = np.zeros(n_clean)
        for i in range(n_clean):
            train_idx = np.delete(np.arange(n_clean), i)
            krig = OrdinaryKriging(x_clean[train_idx], y_clean[train_idx], z_clean[train_idx], variogram_model)
            pred, _ = krig.predict(np.array([x_clean[i]]), np.array([y_clean[i]]), return_variance=True)
            predictions_clean[i] = pred[0]
        
        errors_clean = z_clean - predictions_clean
        rmse_clean = np.sqrt(np.mean(errors_clean**2))
    else:
        rmse_clean = rmse_all
    
    return {
        'rmse_with_outliers': float(rmse_all),
        'rmse_without_outliers': float(rmse_clean),
        'n_outliers': len(outlier_idx),
        'outlier_indices': outlier_idx,
        'improvement': float((rmse_all - rmse_clean) / rmse_all * 100) if rmse_all > 0 else 0.0,
    }
