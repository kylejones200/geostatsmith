"""
    Cross-validation methods for geostatistical models

Supports parallel processing for k-fold and spatial cross-validation
using Python's multiprocessing module.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..core.logging_config import get_logger

logger = get_logger(__name__)


def leave_one_out(kriging_obj) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """
    Perform leave-one-out cross-validation

    Parameters
    ----------
    kriging_obj : BaseKriging
    Fitted kriging object

    Returns
    -------
    predictions : np.ndarray
    Cross-validated predictions
    metrics : dict
    Validation metrics
    """
    return kriging_obj.cross_validate()


def _process_fold(args):
    fold_idx, train_idx, test_idx, x, y, z, kriging_class, variogram_model = args

    # Train kriging model
    krig = kriging_class(
        x[train_idx],
        y[train_idx],
        z[train_idx],
        variogram_model=variogram_model,
    )

    # Predict on test set
    pred = krig.predict(x[test_idx], y[test_idx], return_variance=False)

    return fold_idx, test_idx, pred, z[test_idx]


def k_fold_cross_validation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    kriging_class,
    variogram_model,
    n_folds: int = 5,
    seed: int = 42,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """
    Perform k-fold cross-validation (with optional parallelization)

        Parameters
        ----------
        x, y, z : np.ndarray
        Coordinates and values
        kriging_class : class
        Kriging class to use
        variogram_model
        Variogram model
        n_folds : int
        Number of folds
        seed : int
        Random seed
        n_jobs : int, optional
        Number of parallel jobs. If None, uses serial execution.
        Use -1 to use all available CPU cores.

        Returns
        -------
        dict
        Cross-validation results with predictions, true values, and metrics
    """
    np.random.seed(seed)
    n = len(x)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // n_folds
    all_predictions = np.zeros(n, dtype=np.float64)
    all_true = np.zeros(n, dtype=np.float64)

    # Prepare fold arguments
    fold_args = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        fold_args.append(
            (i, train_idx, test_idx, x, y, z, kriging_class, variogram_model)
        )

    # Execute folds (parallel or serial)
    if n_jobs is not None and n_jobs != 0:
        from multiprocessing import Pool, cpu_count

        n_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        logger.info(
            f"Running {n_folds}-fold cross-validation in parallel ({n_workers} workers)"
        )

        with Pool(processes=n_workers) as pool:
            results = pool.starmap(_process_fold, fold_args)
    else:
        logger.info(f"Running {n_folds}-fold cross-validation (serial)")
        results = [_process_fold(*args) for args in fold_args]

    # Collect results
    for _fold_idx, test_idx, pred, true in results:
        all_predictions[test_idx] = pred
        all_true[test_idx] = true

    from ..validation.metrics import calculate_metrics

    metrics = calculate_metrics(all_true, all_predictions)
    logger.info(
        f"K-fold CV complete. RMSE: {metrics.get('RMSE', 0):.4f}, R^2: {metrics.get('R2', 0):.4f}"
    )

    return {
        "predictions": all_predictions,
        "true_values": all_true,
        "metrics": metrics,
    }


def _process_fold(
    fold_idx, train_idx, test_idx, x, y, z, kriging_class, variogram_model
):
    """Helper function for parallel fold processing."""
    krig = kriging_class(x[train_idx], y[train_idx], z[train_idx], variogram_model)
    pred = krig.predict(x[test_idx], y[test_idx], return_variance=False)
    return fold_idx, test_idx, pred, z[test_idx]


def spatial_cross_validation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    kriging_class,
    variogram_model,
    n_blocks: int = 4,
    seed: int = 42,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """
    Perform spatial cross-validation using spatial blocks (with optional parallelization)

    Divides the spatial domain into blocks to preserve spatial structure.
    More appropriate for spatial data than random k-fold CV.

    Parameters
    ----------
    x, y, z : np.ndarray
    Coordinates and values
    kriging_class : class
    Kriging class to use
    variogram_model
    Variogram model
    n_blocks : int
    Number of spatial blocks (per dimension)
    seed : int
    Random seed
    n_jobs : int, optional
    Number of parallel jobs. If None, uses serial execution.
    Use -1 to use all available CPU cores.

    Returns
    -------
    dict
    Cross-validation results with predictions, true values, and metrics
    """
    # Create spatial blocks
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_bins = np.linspace(x_min, x_max, n_blocks + 1)
    y_bins = np.linspace(y_min, y_max, n_blocks + 1)

    # Assign points to blocks
    x_block_idx = np.digitize(x, x_bins) - 1
    y_block_idx = np.digitize(y, y_bins) - 1
    block_ids = x_block_idx * n_blocks + y_block_idx

    # Prepare block arguments
    block_args = []
    for block_id in range(n_blocks * n_blocks):
        test_mask = block_ids == block_id
        train_mask = ~test_mask
        block_args.append(
            (block_id, test_mask, train_mask, x, y, z, kriging_class, variogram_model)
        )

    # Execute blocks (parallel or serial)
    if n_jobs is not None and n_jobs != 0:
        from multiprocessing import Pool, cpu_count

        n_workers = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())
        logger.info(
            f"Running spatial CV with {n_blocks}x{n_blocks} blocks in parallel ({n_workers} workers)"
        )

        with Pool(processes=n_workers) as pool:
            results = pool.starmap(_cv_block, block_args)
    else:
        logger.info(f"Running spatial CV with {n_blocks}x{n_blocks} blocks (serial)")
        results = [_cv_block(*args) for args in block_args]

    # Collect results
    all_predictions = []
    all_true = []
    for _block_id, pred, true in results:
        all_predictions.extend(pred)
        all_true.extend(true)

    all_predictions = np.array(all_predictions, dtype=np.float64)
    all_true = np.array(all_true, dtype=np.float64)

    from ..validation.metrics import calculate_metrics

    metrics = calculate_metrics(all_true, all_predictions)
    logger.info(
        f"Spatial CV complete. RMSE: {metrics.get('RMSE', 0):.4f}, R^2: {metrics.get('R2', 0):.4f}"
    )

    return {
        "predictions": all_predictions,
        "true_values": all_true,
        "metrics": metrics,
    }


def _cv_block(block_id, test_mask, train_mask, x, y, z, kriging_class, variogram_model):
    """Helper function for parallel block cross-validation."""
    x_test = x[test_mask]
    y_test = y[test_mask]
    z_test = z[test_mask]

    x_train = x[train_mask]
    y_train = y[train_mask]
    z_train = z[train_mask]

    if len(x_train) == 0:
        return block_id, [], []

    try:
        krig = kriging_class(x_train, y_train, z_train, variogram_model)
        pred, _ = krig.predict(x_test, y_test, return_variance=False)
        return block_id, pred.tolist(), z_test.tolist()
    except Exception:
        return block_id, [], []
