"""
    Hyperparameter Tuning
======================

Optimize hyperparameters for kriging and related methods.
"""

import logging
from itertools import product

import numpy as np
import numpy.typing as npt

from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..models.base_model import VariogramModelBase

logger = logging.getLogger(__name__)


def tune_kriging(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    variogram_model: VariogramModelBase,
    param_ranges: dict,
    n_iterations: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Tune kriging hyperparameters via grid search.

    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    variogram_model : VariogramModelBase
        Initial variogram model
    param_ranges : dict
        Parameter ranges to search
    n_iterations : int, default=20
        Number of iterations
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Best parameters and CV score

    Examples
    --------
    >>> results = tune_kriging()
    ...     x, y, z,
    ...     variogram_model=model,
    ...     param_ranges={'nugget': (0, 1), 'sill': (0.5, 2.0)}
    ... )
    """
    logger.info(f"Tuning hyperparameters with {n_iterations} iterations")

    # Get current parameters as baseline
    current_params = variogram_model.get_parameters()
    model_class = variogram_model.__class__

    # Generate parameter grid
    param_grid = {}
    for param_name, param_range in param_ranges.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            param_grid[param_name] = np.linspace(
                param_range[0], param_range[1], n_iterations
            )
        elif isinstance(param_range, list):
            param_grid[param_name] = param_range
        else:
            param_grid[param_name] = [param_range]

    # Generate all combinations

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    # Limit to n_iterations if too many combinations
    if len(param_combinations) > n_iterations:
        indices = np.random.choice(
            len(param_combinations), size=n_iterations, replace=False
        )
        param_combinations = [param_combinations[i] for i in indices]

    best_score = np.inf
    best_params = current_params.copy()
    best_model = variogram_model

    # Test each parameter combination
    for i, param_combo in enumerate(param_combinations):
        try:
            test_params = current_params.copy()
            for param_name, param_value in zip(param_names, param_combo):
                test_params[param_name] = param_value

            # Create model instance
            test_model = model_class(**test_params)

            # Perform cross-validation
            from ..algorithms.ordinary_kriging import OrdinaryKriging

            ok = OrdinaryKriging(x, y, z, variogram_model=test_model)
            predictions, metrics = ok.cross_validate()

            # Use RMSE as score (lower is better)
            score = metrics.get("RMSE", np.inf)

            if score < best_score:
                best_score = score
                best_params = test_params.copy()
                best_model = test_model

            if verbose and (i + 1) % max(1, len(param_combinations) // 10) == 0:
                logger.info(f"  Progress: {i + 1}/{len(param_combinations)}")

        except Exception as e:
            logger.debug(f"Failed to test parameter combination {i + 1}: {e}")

    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best CV RMSE: {best_score:.4f}")

    return {
        "best_params": best_params,
        "best_model": best_model,
        "cv_score": best_score,
        "cv_rmse": best_score,
        "n_tested": len(param_combinations),
    }


def optimize_neighborhood(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    variogram_model: VariogramModelBase,
    max_neighbors_range: tuple[int, int] = (10, 100),
    verbose: bool = True,
) -> int:
    """
    Optimize neighborhood size for approximate kriging.

    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    variogram_model : VariogramModelBase
        Variogram model
    max_neighbors_range : tuple
        (min, max) neighbors to test
    verbose : bool
        Print progress

    Returns
    -------
    optimal_neighbors : int
        Optimal number of neighbors
    """
    logger.info(f"Optimizing neighborhood size in range {max_neighbors_range}")

    # Test different neighborhood sizes via CV
    best_neighbors = max_neighbors_range[0]
    best_rmse = np.inf

    test_sizes = np.linspace(
        max_neighbors_range[0], max_neighbors_range[1], 5, dtype=int
    )
    test_sizes = np.unique(test_sizes)  # Remove duplicates

    from ..algorithms.neighborhood_search import NeighborhoodConfig

    for n_neighbors in test_sizes:
        try:
            neighborhood_config = NeighborhoodConfig(
                max_neighbors=int(n_neighbors),
                min_neighbors=3,
                search_radius=None,  # Use full range
            )

            # Create kriging model with this neighborhood size
            ok = OrdinaryKriging(
                x,
                y,
                z,
                variogram_model=variogram_model,
                neighborhood_config=neighborhood_config,
            )

            # Perform cross-validation

            predictions, metrics = ok.cross_validate()
            rmse = metrics.get("RMSE", np.inf)

            if rmse < best_rmse:
                best_rmse = rmse
                best_neighbors = n_neighbors

            if verbose:
                logger.info(f"  {n_neighbors} neighbors: RMSE = {rmse:.4f}")

        except Exception as e:
            logger.debug(f"Failed to test {n_neighbors} neighbors: {e}")

    if verbose:
        logger.info(
            f"Optimal neighborhood size: {best_neighbors} (RMSE: {best_rmse:.4f})"
        )

    return best_neighbors


def tune_variogram_hyperparameters(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    model_class: type,
    n_trials: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Tune variogram hyperparameters for a given model class.

    This is a convenience function that automatically determines parameter
    ranges and tunes the variogram model via cross-validation.

    Parameters
    ----------
    x, y, z : ndarray
        Sample data
    model_class : type
        Variogram model class (e.g., SphericalModel)
    n_trials : int, default=20
        Number of parameter combinations to try
    verbose : bool, default=True
        Print progress

    Returns
    -------
    results : dict
        Dictionary with 'best_params' (containing 'nugget', 'sill', 'range')
        and 'best_score' (CV RMSE)

    Examples
    --------
    >>> from geostats.models.variogram_models import SphericalModel
    >>> results = tune_variogram_hyperparameters(x, y, z, SphericalModel, n_trials=10)
    >>> print(f"Best nugget: {results['best_params']['nugget']}")
    """
    from ..algorithms.variogram import experimental_variogram

    # Get experimental variogram to determine reasonable parameter ranges
    lags, gamma, _ = experimental_variogram(x, y, z)

    # Estimate reasonable parameter ranges from data
    nugget_max = gamma[0] * 0.5  # Nugget should be less than first semivariance
    sill_min = gamma[-1] * 0.5
    sill_max = gamma[-1] * 2.0
    range_min = lags[1] if len(lags) > 1 else lags[0] * 0.1
    range_max = lags[-1] * 2.0

    # Create initial model to get default parameters
    initial_model = model_class()
    current_params = getattr(initial_model, "_parameters", {})

    # Define parameter ranges
    param_ranges = {
        "nugget": (0.0, nugget_max),
        "sill": (sill_min, sill_max),
        "range": (range_min, range_max),
    }

    # Use grid search to find best parameters
    best_score = np.inf
    best_params = current_params.copy()

    # Generate parameter grid
    nugget_vals = np.linspace(
        param_ranges["nugget"][0], param_ranges["nugget"][1], max(3, n_trials // 4)
    )
    sill_vals = np.linspace(
        param_ranges["sill"][0], param_ranges["sill"][1], max(3, n_trials // 4)
    )
    range_vals = np.linspace(
        param_ranges["range"][0], param_ranges["range"][1], max(3, n_trials // 4)
    )

    # Limit total combinations
    if len(nugget_vals) * len(sill_vals) * len(range_vals) > n_trials:
        # Randomly sample
        from itertools import product

        all_combos = list(product(nugget_vals, sill_vals, range_vals))
        indices = np.random.choice(len(all_combos), size=n_trials, replace=False)
        combos = [all_combos[i] for i in indices]
    else:
        from itertools import product

        combos = list(product(nugget_vals, sill_vals, range_vals))

    if verbose:
        logger.info(
            f"Tuning {model_class.__name__} with {len(combos)} parameter combinations"
        )

    # Test each combination via cross-validation
    for nugget, sill, range_param in combos:
        try:
            # Create model with these parameters
            test_model = model_class(nugget=nugget, sill=sill, range_param=range_param)

            # Perform simple cross-validation
            from ..algorithms.ordinary_kriging import OrdinaryKriging

            n = len(x)
            cv_preds = np.zeros(n)

            for i in range(n):
                train_idx = np.arange(n) != i
                try:
                    krig = OrdinaryKriging(
                        x[train_idx], y[train_idx], z[train_idx], test_model
                    )
                    pred, _ = krig.predict(
                        np.array([x[i]]), np.array([y[i]]), return_variance=True
                    )
                    cv_preds[i] = pred[0]
                except Exception:
                    cv_preds[i] = z[train_idx].mean()  # Fallback

            errors = z - cv_preds
            rmse = np.sqrt(np.mean(errors**2))

            if rmse < best_score:
                best_score = rmse
                best_params = {
                    "nugget": float(nugget),
                    "sill": float(sill),
                    "range": float(range_param),
                }
        except Exception:
            continue

    if verbose:
        logger.info(
            f"Best parameters: nugget={best_params['nugget']:.4f}, "
            f"sill={best_params['sill']:.4f}, range={best_params['range']:.4f}"
        )
        logger.info(f"Best CV RMSE: {best_score:.4f}")

    return {
        "best_params": best_params,
        "best_score": float(best_score),
    }
