"""
Variogram model fitting algorithms
"""

from typing import Optional, Dict, Any, List, Type
import numpy as np
import numpy.typing as npt

from ..models.base_model import VariogramModelBase
from ..models.variogram_models import (
 SphericalModel,
 ExponentialModel,
 GaussianModel,
 MaternModel,
)
from ..core.exceptions import FittingError

def fit_variogram_model(
 lags: npt.NDArray[np.float64],
 gamma: npt.NDArray[np.float64],
 weights: Optional[npt.NDArray[np.float64]] = None,
 fit_nugget: bool = True,
 **kwargs: Any,
    ) -> VariogramModelBase:
 """
 Fit a variogram model to experimental data

 Parameters
 ----------
 model : VariogramModelBase
 Variogram model to fit
 lags : np.ndarray
 Lag distances
 gamma : np.ndarray
 Experimental semivariance values
 weights : np.ndarray, optional
 Weights for each lag (e.g., number of pairs)
 fit_nugget : bool
 Whether to fit the nugget effect
 **kwargs
 Additional parameters passed to model.fit()

 Returns
 -------
 VariogramModelBase
 Fitted model
 """
 # Remove NaN values
 valid_mask = ~np.isnan(gamma) & ~np.isnan(lags)
 lags_clean = lags[valid_mask]
 gamma_clean = gamma[valid_mask]

 if weights is not None:
    pass

 if len(lags_clean) == 0:
    pass

 # Fit the model
 model.fit(lags_clean, gamma_clean, weights=weights, fit_nugget=fit_nugget, **kwargs)

 return model

def automatic_fit(
 gamma: npt.NDArray[np.float64],
 models: Optional[List[Type[VariogramModelBase]]] = None,
 weights: Optional[npt.NDArray[np.float64]] = None,
 criterion: str = "rmse",
    ) -> Dict[str, Any]:
 """
 Automatically select and fit the best variogram model

 Tries multiple models and selects the best based on a criterion.

 Parameters
 ----------
 lags : np.ndarray
 Lag distances
 gamma : np.ndarray
 Experimental semivariance values
 models : list of model classes, optional
 Models to try. If None, tries standard models.
 weights : np.ndarray, optional
 Weights for fitting
 criterion : str
 Selection criterion: 'rmse', 'mae', 'r2', 'aic'

 Returns
 -------
 dict
 Dictionary containing:
 - 'model': Best fitted model
 - 'score': Best score value
 - 'all_results': Results for all models tried

 Examples
 --------
 >>> lags, gamma, _ = experimental_variogram(x, y, z)
 >>> result = automatic_fit(lags, gamma)
 >>> best_model = result['model']
 """
 # Default models to try
 if models is None:
 SphericalModel,
 ExponentialModel,
 GaussianModel,
 MaternModel,
 ]

 # Remove NaN values
 valid_mask = ~np.isnan(gamma) & ~np.isnan(lags)
 lags_clean = lags[valid_mask]
 gamma_clean = gamma[valid_mask]

 if weights is not None:
 else:
 else:
    pass

 if len(lags_clean) == 0:
    pass

 # Try each model
 results = []

 for model_class in models:
 # Initialize and fit model
 model = model_class()
 model.fit(lags_clean, gamma_clean, weights=weights_clean)

 # Predict with fitted model
 gamma_pred = model(lags_clean)

 # Calculate goodness-of-fit metrics
 residuals = gamma_clean - gamma_pred

 rmse = np.sqrt(np.mean(residuals ** 2))
 mae = np.mean(np.abs(residuals))

 # R²
 ss_res = np.sum(residuals ** 2)
 ss_tot = np.sum((gamma_clean - np.mean(gamma_clean)) ** 2)
 r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

 # AIC (Akaike Information Criterion)
 n = len(gamma_clean)
 k = len(model.parameters) # Number of parameters
 aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else np.inf

 results.append({
 'model': model,
 'model_name': model_class.__name__,
 'rmse': rmse,
 'mae': mae,
 'r2': r2,
 'aic': aic,
 })

 except Exception as e:
 # Skip models that fail to fit
 results.append({
 'model': None,
 'model_name': model_class.__name__,
 'error': str(e),
 'rmse': np.inf,
 'mae': np.inf,
 'r2': -np.inf,
 'aic': np.inf,
 })

 # Select best model based on criterion using dictionary dispatch
 criterion_functions = {
 'rmse': (np.argmin, 'rmse'),
 'mae': (np.argmin, 'mae'),
 'r2': (np.argmax, 'r2'),
 'aic': (np.argmin, 'aic'),
 }

 if criterion not in criterion_functions:
 raise ValueError(
 f"Unknown criterion '{criterion}'. "
 f"Valid criteria: {valid_criteria}"
 )

 select_fn, metric = criterion_functions[criterion]
 best_idx = select_fn([r[metric] for r in results])
 best_result = results[best_idx]

 if best_result['model'] is None:
    pass

 return {
 'model': best_result['model'],
 'score': best_result[criterion],
 'all_results': results,
 }

def cross_validation_fit(
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 model_class: Type[VariogramModelBase],
 n_folds: int = 5,
    ) -> Dict[str, float]:
 """
 Cross-validate variogram model fitting

 Parameters
 ----------
 x, y : np.ndarray
 Coordinates
 z : np.ndarray
 Values
 model_class : VariogramModelBase class
 Model class to use
 n_folds : int
 Number of cross-validation folds

 Returns
 -------
 dict
 Cross-validation metrics
 """
 from .variogram import experimental_variogram

 n = len(x)
 indices = np.arange(n)
 np.random.shuffle(indices)
 fold_size = n // n_folds

 scores = []

 for i in range(n_folds):
 test_start = i * fold_size
 test_end = (i + 1) * fold_size if i < n_folds - 1 else n
 test_idx = indices[test_start:test_end]
 train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

 # Training data
 x_train, y_train, z_train = x[train_idx], y[train_idx], z[train_idx]
 x_test, y_test, z_test = x[test_idx], y[test_idx], z[test_idx]

 # Fit model on training data
 lags, gamma, _ = experimental_variogram(x_train, y_train, z_train)
 model = model_class()
 model.fit(lags, gamma)

 # Evaluate on test data (using the variogram)
 test_lags, test_gamma, _ = experimental_variogram(x_test, y_test, z_test)
 pred_gamma = model(test_lags)

 # Remove NaN
 valid = ~np.isnan(test_gamma) & ~np.isnan(pred_gamma)
 if np.sum(valid) > 0:
 scores.append(rmse)

 return {
 'mean_rmse': np.mean(scores),
 'std_rmse': np.std(scores),
 'scores': scores,
 }
