"""
    Automatic Variogram Model Selection
====================================

Automatically select best variogram model and parameters.
"""

import logging
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Optional

from ..algorithms.variogram import experimental_variogram
from ..algorithms.fitting import fit_variogram_model as fit_variogram
from ..models.base_model import VariogramModelBase
from ..core.exceptions import FittingError

logger = logging.getLogger(__name__)

def auto_variogram(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    model_types: Optional[List[str]] = None,
    n_lags: int = 15,
    verbose: bool = True,
) -> VariogramModelBase:
    """
    Automatically select best variogram model.

    Tries multiple models and selects based on R^2 fit to experimental variogram.

    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    model_types : list of str, optional
        Models to try. Default: ['spherical', 'exponential', 'gaussian', 'linear']
    n_lags : int, default=15
        Number of lags
    verbose : bool, default=True
        Print selection process

    Returns
    -------
    best_model : VariogramModelBase
        Best fitted variogram model

    Examples
    --------
    >>> from geostats.automl import auto_variogram
    >>>
    >>> # Automatically select best model
    >>> model = auto_variogram(x, y, z)
    >>> logger.info(f"Selected: {model.__class__.__name__}")

    Notes
    -----
    This uses parallel_variogram_fit under the hood for speed.
    """
    if model_types is None:
        model_types = ['spherical', 'exponential', 'gaussian', 'linear']

    try:
        from ..algorithms.fitting import automatic_fit
        from ..algorithms.variogram import experimental_variogram

        lags, gamma, _ = experimental_variogram(x, y, z, n_lags=n_lags)
        results = automatic_fit(lags, gamma, criterion='r2')

        if verbose:
            logger.info(f"✓ Best model: {results['model'].__class__.__name__}")
            logger.info(f"  R^2: {results['score']:.4f}")

        return results['model']

    except ImportError:
        # Fallback if automatic_fit not available
        from ..algorithms.variogram import experimental_variogram
        from ..algorithms.fitting import fit_variogram_model
        from ..models.variogram_models import SphericalModel, ExponentialModel, GaussianModel, LinearModel

        lags, gamma, _ = experimental_variogram(x, y, z, n_lags=n_lags)

        best_model = None
        best_r2 = -np.inf
        best_type = None

        model_classes = {
            'spherical': SphericalModel,
            'exponential': ExponentialModel,
            'gaussian': GaussianModel,
            'linear': LinearModel,
        }

        for model_type in model_types:
            try:
                model_class = model_classes.get(model_type)
                if model_class is None:
                    continue
                model = model_class()
                model = fit_variogram_model(model, lags, gamma)

                gamma_fitted = model(lags)
                ss_res = np.sum((gamma - gamma_fitted)**2)
                ss_tot = np.sum((gamma - gamma.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                if r2 > best_r2:
                    best_model = model
                    best_r2 = r2
                    best_type = model_type

            except Exception:
                continue

        if verbose:
            logger.info(f"✓ Best model: {best_type}")
            logger.info(f"  R^2: {best_r2:.4f}")

        if best_model is None:
            raise RuntimeError(
                f"All {len(model_types)} variogram models failed to fit. "
                "Check data quality (sufficient points, spatial structure, no duplicates)."
            )

        return best_model

    except Exception as e:
        if verbose:
            logger.error(f"Error in auto_variogram: {e}")
        raise

# Alias for backward compatibility - returns dict like auto_fit
def auto_fit_variogram(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    model_types: Optional[List[str]] = None,
    models: Optional[List] = None,  # Accept 'models' parameter for backward compatibility
    n_lags: int = 15,
    verbose: bool = True,
    cross_validate: bool = False,
) -> Dict:
    """
    Alias for auto_fit that returns a dictionary with model and parameters.
    
    This function provides backward compatibility for tests and code expecting
    a dictionary return type instead of just the model.
    """
    # Convert models (classes) to model_types (strings) if provided
    if models is not None and model_types is None:
        # Extract model type names from model classes
        # SphericalModel -> 'spherical', ExponentialModel -> 'exponential', etc.
        model_type_map = {
            'SphericalModel': 'spherical',
            'ExponentialModel': 'exponential',
            'GaussianModel': 'gaussian',
            'LinearModel': 'linear',
            'PowerModel': 'power',
        }
        model_types = [model_type_map.get(m.__name__, m.__name__.replace('Model', '').lower()) for m in models]
    
    model = auto_variogram(x, y, z, model_types=model_types, n_lags=n_lags, verbose=verbose)
    
    # Get fit quality if available
    try:
        from ..algorithms.variogram import experimental_variogram
        lags, gamma, _ = experimental_variogram(x, y, z, n_lags=n_lags)
        gamma_fitted = model(lags)
        ss_res = np.sum((gamma - gamma_fitted)**2)
        ss_tot = np.sum((gamma - gamma.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        criterion = r2
    except Exception:
        criterion = None
    
    result = {
        'model': model,
        'parameters': getattr(model, '_parameters', {}),
        'criterion': criterion,
    }
    
    if cross_validate:
        # Use auto_fit for CV
        return auto_fit(x, y, z, cross_validate=True, verbose=verbose)
    
    return result


def auto_fit(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    cross_validate: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Automatic model fitting with cross-validation.

    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    cross_validate : bool, default=True
        Perform cross-validation to assess fit
    verbose : bool, default=True
        Print results

    Returns
    -------
    results : dict
        Dictionary with model, CV scores, etc.

    Examples
    --------
    >>> results = auto_fit(x, y, z)
    >>> model = results['model']
    >>> logger.info(f"CV RMSE: {results['cv_rmse']:.3f}")
    """
    model = auto_variogram(x, y, z, verbose=verbose)

    results = {
        'model': model,
        'model_type': model.__class__.__name__,
        'parameters': getattr(model, '_parameters', {}),
    }

    if cross_validate:
        from ..algorithms.ordinary_kriging import OrdinaryKriging
        n = len(x)
        predictions = np.zeros(n)

        for i in range(n):
            train_idx = np.arange(n) != i
            x_train = x[train_idx]
            y_train = y[train_idx]
            z_train = z[train_idx]

            krig = OrdinaryKriging(x_train, y_train, z_train, model)
            pred, _ = krig.predict(np.array([x[i]]), np.array([y[i]]), return_variance=True)
            predictions[i] = pred[0]

        errors = z - predictions
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        r2 = 1 - np.sum(errors**2) / np.sum((z - z.mean())**2)

        results['cv_predictions'] = predictions
        results['cv_rmse'] = rmse
        results['cv_mae'] = mae
        results['cv_r2'] = r2

        if verbose:
            logger.info(f"  CV RMSE: {rmse:.4f}")
            logger.info(f"  CV MAE: {mae:.4f}")
            logger.info(f"  CV R^2: {r2:.4f}")

    return results
