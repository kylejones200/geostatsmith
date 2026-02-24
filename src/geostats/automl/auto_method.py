"""
    Automatic Method Selection
===========================

Automatically select best interpolation method.
"""

import logging

import numpy as np
import numpy.typing as npt

from ..algorithms.ordinary_kriging import OrdinaryKriging
from .auto_variogram import auto_variogram


def auto_interpolate(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    x_pred: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    methods: list[str] | None = None,
    verbose: bool = True,
) -> dict:
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
    >>> logger.info(f"Best method: {results['best_method']}")
    >>> logger.info(f"CV RMSE: {results['cv_rmse']:.3f}")
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
        methods = ["ordinary_kriging", "idw", "rbf"]

    if verbose:
        logger.info(f"Auto-selecting best interpolation method from {methods}")

    best_method = None
    best_rmse = np.inf
    best_predictions = None
    best_model = None
    all_results = {}

    for method in methods:
        if verbose:
            logger.info(f"\nTesting: {method}")

        try:
            if method == "ordinary_kriging":
                from ..algorithms.ordinary_kriging import OrdinaryKriging
                from .auto_variogram import auto_variogram

                model = auto_variogram(x, y, z, verbose=False)

                n = len(x)
                cv_preds = np.zeros(n)

                for i in range(n):
                    train_idx = np.arange(n) != i
                    krig = OrdinaryKriging(
                        x[train_idx], y[train_idx], z[train_idx], model
                    )
                    pred, _ = krig.predict(
                        np.array([x[i]]), np.array([y[i]]), return_variance=True
                    )
                    cv_preds[i] = pred[0]

                errors = z - cv_preds
                rmse = np.sqrt(np.mean(errors**2))

                krig_final = OrdinaryKriging(x, y, z, model)
                predictions, variance = krig_final.predict(
                    x_pred, y_pred, return_variance=True
                )

                all_results[method] = {
                    "rmse": rmse,
                    "predictions": predictions,
                    "variance": variance,
                    "model": model,
                }

                if verbose:
                    logger.info(f"  RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_method = method
                    best_rmse = rmse
                    best_predictions = predictions
                    best_model = model

            elif method == "idw":
                from ..comparison.method_implementations import InverseDistanceWeighting

                idw = InverseDistanceWeighting(x, y, z, power=2)

                n = len(x)
                cv_preds = np.zeros(n)
                for i in range(n):
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    idw_cv = InverseDistanceWeighting(
                        x[mask], y[mask], z[mask], power=2
                    )
                    cv_preds[i] = idw_cv.predict(np.array([x[i]]), np.array([y[i]]))[0]

                errors = z - cv_preds
                rmse = np.sqrt(np.mean(errors**2))

                predictions = idw.predict(x_pred, y_pred)

                all_results[method] = {
                    "rmse": rmse,
                    "predictions": predictions,
                }

                if verbose:
                    logger.info(f"  RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_method = method
                    best_rmse = rmse
                    best_predictions = predictions
                    best_model = None

        except ImportError:
            if verbose:
                logger.warning(
                    "IDW interpolation not available: comparison module not installed. "
                    "Install with: pip install geostats[comparison]"
                )
            continue

        except Exception as e:
            if verbose:
                logger.warning(f"  Method {method} failed: {e}")
            continue

    if best_method is None:
        raise RuntimeError(
            f"All {len(methods)} interpolation methods failed. "
            "Check data quality (sufficient points, no duplicates, valid values)."
        )

    if verbose:
        logger.info(f"\n✓ Best method: {best_method} (CV RMSE: {best_rmse:.4f})")

    return {
        "best_method": best_method,
        "cv_rmse": best_rmse,
        "predictions": best_predictions,
        "model": best_model,
        "all_results": all_results,
    }


def auto_select_method(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    methods: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Automatically select best interpolation method via cross-validation.

    This is a convenience wrapper that performs cross-validation to select
    the best method without making predictions.

    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    methods : list of str, optional
        Methods to try. Default: ['ordinary_kriging', 'idw', 'rbf']
    verbose : bool, default=False
        Print selection process

    Returns
    -------
    results : dict
        Dictionary with 'method' (best method name) and 'score' (CV RMSE)
    """
    if methods is None:
        methods = ["ordinary_kriging", "simple_kriging", "idw"]

    best_method = None
    best_rmse = np.inf

    for method in methods:
        try:
            if method in ["ordinary_kriging", "simple_kriging"]:
                model = auto_variogram(x, y, z, verbose=False)

                n = len(x)
                cv_preds = np.zeros(n)

                for i in range(n):
                    train_idx = np.arange(n) != i
                    if method == "ordinary_kriging":
                        krig = OrdinaryKriging(
                            x[train_idx], y[train_idx], z[train_idx], model
                        )
                    else:
                        from ..algorithms.simple_kriging import SimpleKriging

                        krig = SimpleKriging(
                            x[train_idx], y[train_idx], z[train_idx], model
                        )
                    pred, _ = krig.predict(
                        np.array([x[i]]), np.array([y[i]]), return_variance=True
                    )
                    cv_preds[i] = pred[0]

                errors = z - cv_preds
                rmse = np.sqrt(np.mean(errors**2))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_method = method

            elif method == "idw":
                from ..algorithms.idw import IDW

                n = len(x)
                cv_preds = np.zeros(n)

                for i in range(n):
                    train_idx = np.arange(n) != i
                    idw = IDW(x[train_idx], y[train_idx], z[train_idx])
                    pred = idw.predict(np.array([x[i]]), np.array([y[i]]))
                    cv_preds[i] = pred[0]

                errors = z - cv_preds
                rmse = np.sqrt(np.mean(errors**2))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_method = method
        except Exception:
            continue

    if best_method is None:
        # Fallback
        best_method = "ordinary_kriging"
        best_rmse = 0.0

    return {
        "method": best_method,
        "score": best_rmse,
    }


def suggest_method(
    n_predictions: int,
    data_characteristics: dict | None = None,
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
    >>> logger.info(f"Suggested: {method}")
    """
    # Simple heuristics
    if n_predictions > 50000:
        return "idw"  # Fast for large grids
    else:
        return "ordinary_kriging"  # Best quality
