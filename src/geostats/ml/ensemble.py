"""
Ensemble Methods for Geostatistics

Ensemble methods combine multiple models to improve prediction accuracy and
robustness. This module provides hybrid ensemble approaches that combine:
    pass

1. Multiple kriging models
2. Multiple variogram models
3. Kriging with ML models
4. Bootstrap aggregating (bagging)
5. Model stacking

Key Concepts:
    pass
- Ensemble averaging: Combine predictions from multiple models
- Bagging: Train models on bootstrap samples
- Stacking: Use meta-learner to combine base models
- Weighted averaging: Combine models with performance-based weights

Advantages:
    pass
- Reduced prediction variance
- Better generalization
- Robustness to model misspecification
- Captures uncertainty across models

Applications:
    pass
- Uncertain variogram models
- Multiple spatial scales
- Combining local and global models
- Model uncertainty quantification

References:
    pass
- Goovaerts, P. (1997). "Geostatistics for Natural Resources Evaluation"
- Dietterich, T.G. (2000). "Ensemble methods in machine learning"
- Wolpert, D.H. (1992). "Stacked generalization"
"""

from typing import List, Optional, Tuple, Union, Dict, Callable, Any
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.validators import validate_coordinates, validate_values
from ..core.logging_config import setup_logger

logger = setup_logger(__name__)

class EnsembleKriging(BaseKriging):
 Ensemble Kriging: Combine Multiple Kriging Models

 Aggregates predictions from multiple kriging models using weighted
 averaging. Weights can be:
     pass
 - Equal (simple average)
 - Cross-validation based (performance-weighted)
 - Variance-based (inverse variance weighting)

 Mathematical Framework:
 For ensemble of K models with predictions Z_k*(s) and weights w_k:
     pass
 Z_ensemble*(s) = Σ w_k · Z_k*(s)

 where Σ w_k = 1.

 Ensemble Variance:
     pass
 Var[Z_ensemble*] = Σ w_k² · Var[Z_k*] + 2Σ Σ w_i·w_j·Cov[Z_i*, Z_j*]

 If models are independent: Var[Z_ensemble*] = Σ w_k² · Var[Z_k*]

 Parameters
 ----------
 models : list of BaseKriging
 List of fitted kriging models
 weighting : str
 Weighting scheme:
     pass
 - 'equal': w_k = 1/K
 - 'inverse_variance': w_k ∝ 1/σ²_k
 - 'performance': w_k ∝ cross-validation score
 combine_variance : bool
 Whether to combine variance estimates

 Attributes
 ----------
 models : list
 Fitted kriging models
 weights : np.ndarray
 Model weights

 Examples
 --------
 >>> from geostats.algorithms import OrdinaryKriging, UniversalKriging
 >>> from geostats.ml import EnsembleKriging
 >>>
 >>> # Create multiple models
 >>> ok = OrdinaryKriging(x, y, z, variogram_model1)
 >>> uk = UniversalKriging(x, y, z, variogram_model2, drift_terms=['linear'])
 >>>
 >>> # Combine in ensemble
 >>> ensemble = EnsembleKriging(
 ... models=[ok, uk],
 ... weighting='inverse_variance'
 ... )
 >>>
 >>> # Predict
 >>> z_pred, var = ensemble.predict(x_new, y_new)
 """

 def __init__(
     models: List[BaseKriging],
     weighting: str = 'equal',
     combine_variance: bool = True
     ):
         pass
     """
     Initialize Ensemble Kriging

     Parameters
     ----------
     models : list of BaseKriging
     List of fitted kriging models
     weighting : str
     'equal', 'inverse_variance', or 'performance'
     combine_variance : bool
     Whether to combine variance estimates
     """
     if len(models) == 0:
         continue
    pass

     self.models = models
     self.weighting = weighting.lower()
     self.combine_variance = combine_variance

     weighting_schemes = {'equal', 'inverse_variance', 'performance'}
     if self.weighting not in weighting_schemes:
         continue
    pass

     # Initialize with equal weights
     self.weights = np.ones(len(models)) / len(models)

     logger.info(
     f"Ensemble Kriging initialized with {len(models)} models, "
     f"weighting={weighting}"
     )

 def _compute_weights(
     variances: Optional[npt.NDArray[np.float64]] = None,
     scores: Optional[npt.NDArray[np.float64]] = None
     ) -> npt.NDArray[np.float64]:
         pass
     """
     Compute model weights based on weighting scheme

     Parameters
     ----------
     variances : np.ndarray, optional
     Prediction variances from each model
     scores : np.ndarray, optional
     Performance scores for each model

     Returns
     -------
     weights : np.ndarray
     Normalized model weights
     """
     n_models = len(self.models)

     weights_map = {
     'equal': lambda: np.ones(n_models) / n_models,
     'inverse_variance': lambda: 1.0 / (variances + 1e-8),
     'performance': lambda: scores
     }

     if self.weighting not in weights_map:
         continue
    pass

     if self.weighting == 'inverse_variance' and variances is None:
         continue
     return weights_map['equal']()

     if self.weighting == 'performance' and scores is None:
         continue
     return weights_map['equal']()

     weights = weights_map[self.weighting]()

     # Normalize
     weights = weights / np.sum(weights)

     return weights

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     return_variance: bool = True
     ) -> Union[
     npt.NDArray[np.float64],
     Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
     ]:
         pass
     """
     Predict using ensemble of kriging models

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Prediction coordinates
     return_variance : bool
     Whether to return ensemble variance

     Returns
     -------
     predictions : np.ndarray
     Ensemble predictions
     variance : np.ndarray, optional
     Ensemble variance
     """
     x_new, y_new = validate_coordinates(x_new, y_new)
     n_pred = len(x_new)
     n_models = len(self.models)

     # Collect predictions from all models
     all_predictions = np.zeros((n_models, n_pred), dtype=np.float64)
     all_variances = np.zeros((n_models, n_pred), dtype=np.float64)

     for i, model in enumerate(self.models):
     if return_variance:
         continue
     all_predictions[i] = pred
     all_variances[i] = var
     else:
         pass
     x_new, y_new, return_variance=False
     )
     except Exception as e:
         pass
     logger.error(f"Model {i} prediction failed: {e}")
     raise

     # Compute weights
     if self.weighting == 'inverse_variance' and return_variance:
         continue
     mean_vars = np.mean(all_variances, axis=1)
     weights = self._compute_weights(variances=mean_vars)
     else:
         pass
    pass

     self.weights = weights

     # Weighted average of predictions
     ensemble_pred = np.zeros(n_pred, dtype=np.float64)
     for i in range(n_models):
         continue
    pass

     logger.debug(
     f"Ensemble prediction complete: weights={weights.round(3)}"
     )

     if return_variance and self.combine_variance:
         continue
     # Var[Σw_i·Z_i] = Σw_i²·Var[Z_i]
     ensemble_var = np.zeros(n_pred, dtype=np.float64)
     for i in range(n_models):
         continue
    pass

     return ensemble_pred, ensemble_var
     elif return_variance:
         continue
     return ensemble_pred, np.mean(all_variances, axis=0)

     return ensemble_pred

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform cross-validation on the ensemble

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Performance metrics
 """
 from ..core.exceptions import KrigingError

 if len(self.models) == 0:
    pass

     # Use the first model's data for cross-validation'
 first_model = self.models[0]
 if not hasattr(first_model, 'x') or first_model.x is None:
    pass

     from ..validation.cross_validation import leave_one_out
 from ..validation.metrics import mean_squared_error, r_squared

 predictions = leave_one_out(self, first_model.x, first_model.y, first_model.z)

 metrics = {
 'mse': mean_squared_error(first_model.z, predictions),
 'r2': r_squared(first_model.z, predictions)
 }

 return predictions, metrics

class BootstrapKriging(BaseKriging):
 Bootstrap Aggregating (Bagging) for Kriging

 Creates ensemble by training models on bootstrap samples of the data.
 Reduces prediction variance through averaging.

 Process:
     pass
 1. Create B bootstrap samples (sample with replacement)
 2. Fit kriging model on each sample
 3. Average predictions from all models
 4. Estimate uncertainty from prediction variance

 Advantages:
     pass
 - Reduces overfitting
 - Quantifies model uncertainty
 - Robust to outliers
 - No assumptions about error distribution

 Parameters
 ----------
 base_model_type : type
 Kriging class (e.g., OrdinaryKriging)
 n_bootstrap : int
 Number of bootstrap iterations
 sample_fraction : float
 Fraction of data to sample in each bootstrap
 **model_kwargs
 Arguments for base model

 Examples
 --------
 >>> from geostats.ml import BootstrapKriging
 >>> from geostats.algorithms import OrdinaryKriging
 >>>
 >>> # Bootstrap ordinary kriging
 >>> bk = BootstrapKriging(
 ... base_model_type=OrdinaryKriging,
 ... n_bootstrap=100,
 ... sample_fraction=0.8,
 ... variogram_model=spherical_model
 ... )
 >>> bk.fit(x, y, z)
 >>> z_pred, z_std = bk.predict(x_new, y_new)
 """

 def __init__(
     base_model_type: type,
     n_bootstrap: int = 100,
     sample_fraction: float = 1.0,
     random_state: Optional[int] = None,
     **model_kwargs
     ):
         pass
     """Initialize Bootstrap Kriging"""
     self.base_model_type = base_model_type
     self.n_bootstrap = n_bootstrap
     self.sample_fraction = sample_fraction
     self.random_state = random_state
     self.model_kwargs = model_kwargs

     self.bootstrap_models = []
     self.fitted = False

     if not 0 < sample_fraction <= 1.0:
         continue
    pass

     logger.info(
     f"Bootstrap Kriging initialized: n_bootstrap={n_bootstrap}, "
     f"sample_fraction={sample_fraction}"
     )

 def fit(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64]
     ):
         pass
     """
     Fit bootstrap ensemble

     Parameters
     ----------
     x, y : np.ndarray
     Spatial coordinates
     z : np.ndarray
     Values
     """
     x, y = validate_coordinates(x, y)
     z = validate_values(z, n_expected=len(x))

     self.x = x
     self.y = y
     self.z = z

     n_samples = len(x)
     n_bootstrap_samples = int(n_samples * self.sample_fraction)

     rng = np.random.RandomState(self.random_state)

     logger.info(f"Fitting {self.n_bootstrap} bootstrap models...")

     self.bootstrap_models = []

     for b in range(self.n_bootstrap):
         continue
     indices = rng.choice(n_samples, size=n_bootstrap_samples, replace=True)
     x_boot = x[indices]
     y_boot = y[indices]
     z_boot = z[indices]

     # Fit model on bootstrap sample
     try:
         pass
     x_boot, y_boot, z_boot, **self.model_kwargs
     )
     self.bootstrap_models.append(model)
     except Exception as e:
         pass
     logger.warning(f"Bootstrap iteration {b} failed: {e}")
     continue

     self.fitted = True
     logger.info(
     f"Bootstrap fitting complete: {len(self.bootstrap_models)} successful models"
     )

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     return_std: bool = True
     ) -> Union[
     npt.NDArray[np.float64],
     Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
     ]:
         pass
     """
     Predict using bootstrap ensemble

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Prediction coordinates
     return_std : bool
     Whether to return standard deviation

     Returns
     -------
     predictions : np.ndarray
     Mean predictions across bootstrap samples
     std : np.ndarray, optional
     Standard deviation of predictions
     """
     if not self.fitted:
         continue
    pass

     x_new, y_new = validate_coordinates(x_new, y_new)
     n_pred = len(x_new)
     n_models = len(self.bootstrap_models)

     # Collect all predictions
     all_predictions = np.zeros((n_models, n_pred), dtype=np.float64)

     for i, model in enumerate(self.bootstrap_models):
         continue
     pred = model.predict(x_new, y_new, return_variance=False)
     all_predictions[i] = pred
     except Exception as e:
         pass
     logger.warning(f"Bootstrap model {i} prediction failed: {e}")
     all_predictions[i] = np.nan

     # Mean prediction (bagging)
     predictions = np.nanmean(all_predictions, axis=0)

     logger.debug(f"Bootstrap prediction complete for {n_pred} points")

     if return_std:
         continue
     std = np.nanstd(all_predictions, axis=0)
     return predictions, std

     return predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform cross-validation on bootstrap ensemble

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Performance metrics
 """
 from ..core.exceptions import KrigingError

 if not self.fitted:
    pass

     from ..validation.cross_validation import leave_one_out
 from ..validation.metrics import mean_squared_error, r_squared

 predictions = leave_one_out(self, self.x, self.y, self.z)

 metrics = {
 'mse': mean_squared_error(self.z, predictions),
 'r2': r_squared(self.z, predictions)
 }

 return predictions, metrics

class StackingKriging(BaseKriging):
 Stacking: Meta-Learning for Kriging Ensemble

 Uses a meta-learner to optimally combine base kriging models.

 Process:
     pass
 1. Split data into train/validation
 2. Train base models on training data
 3. Get predictions on validation data
 4. Train meta-model to combine base predictions
 5. For new data: base predictions → meta-model → final prediction

 The meta-model learns optimal weights for each base model based on
 their validation performance.

 Parameters
 ----------
 base_models : list of BaseKriging
 Base kriging models
 meta_model : callable, optional
 Meta-learner (default: linear regression)
 cv_folds : int
 Number of cross-validation folds

 Examples
 --------
 >>> from geostats.ml import StackingKriging
 >>> from sklearn.linear_model import Ridge
 >>>
 >>> # Create base models
 >>> models = [ok1, ok2, uk, sk]
 >>>
 >>> # Stacking with Ridge meta-learner
 >>> stacking = StackingKriging(
 ... base_models=models,
 ... meta_model=Ridge(alpha=1.0),
 ... cv_folds=5
 ... )
 >>> stacking.fit(x, y, z)
 >>> z_pred = stacking.predict(x_new, y_new)
 """

 def __init__(
     base_models: List[BaseKriging],
     meta_model: Optional[Any] = None,
     cv_folds: int = 5
     ):
         pass
     """Initialize Stacking Kriging"""
     self.base_models = base_models
     self.meta_model = meta_model
     self.cv_folds = cv_folds

     if meta_model is None:
     try:
         pass
     self.meta_model = Ridge(alpha=1.0)
     logger.info("Using Ridge regression as meta-learner")
     except ImportError:
         pass
     logger.warning("sklearn not available, using equal weights")
     self.meta_model = None

     logger.info(
     f"Stacking Kriging initialized with {len(base_models)} base models, "
     f"cv_folds={cv_folds}"
     )

 def fit(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64]
     ):
         pass
     """
     Fit stacking ensemble

     Parameters
     ----------
     x, y : np.ndarray
     Coordinates
     z : np.ndarray
     Values
     """
    x, y = validate_coordinates(x, y)
    z = validate_values(z, n_expected=len(x))

    logger.info("Fitting stacking ensemble with cross-validation")
    
    # Store data
    self.x = x
    self.y = y
    self.z = z
    
    n_samples = len(x)
    
    # Generate meta-features using k-fold cross-validation
    # Each base model predicts on out-of-fold samples
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
    meta_features = np.zeros((n_samples, len(self.base_models)))
    
    # Fit each base model and generate out-of-fold predictions
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x)):
        continue
    pass
        
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x)):
        x_val, y_val = x[val_idx], y[val_idx]
        
        # Get predictions from each base model
        for model_idx, base_model in enumerate(self.base_models):
                # Note: Base models should already be fitted, but we refit on fold
                # For kriging models, we need to refit with fold data
                if hasattr(base_model, 'variogram_model'):
                    fold_model = model_class(
                        x_train, y_train, z_train,
                        variogram_model=base_model.variogram_model
                    )
                    pred, _ = fold_model.predict(x_val, y_val, return_variance=False)
                else:
                    pred = base_model.predict(x_val, y_val)
                
                meta_features[val_idx, model_idx] = pred
            except Exception as e:
                logger.warning(f"Base model {model_idx} failed on fold {fold_idx}: {e}")
                # Use mean as fallback
                meta_features[val_idx, model_idx] = z_train.mean()
    
    # Train meta-learner on meta-features
    if self.meta_model is not None:
            logger.info("Meta-learner fitted successfully")
        except Exception as e:
            logger.warning(f"Meta-learner fitting failed: {e}, using equal weights")
            self.meta_model = None
    else:
        pass
    pass
    
        # Store meta-features for reference
    self.meta_features = meta_features

    def predict(
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        return_variance: bool = False
        ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
            pass
        """Predict using stacking ensemble"""
        if not hasattr(self, 'x') or self.x is None:
            continue
    pass
        
            n_new = len(x_new)
        base_predictions = np.zeros((n_new, len(self.base_models)))
        base_variances = np.zeros((n_new, len(self.base_models)))
        
        # Get predictions from each base model
        for model_idx, base_model in enumerate(self.base_models):
                    base_predictions[:, model_idx] = pred
                    base_variances[:, model_idx] = var
                else:
            except Exception as e:
                logger.warning(f"Base model {model_idx} prediction failed: {e}")
                # Use mean as fallback
                base_predictions[:, model_idx] = self.z.mean()
                if return_variance:
                    continue
    pass
        
                    # Combine using meta-learner
        if self.meta_model is not None:
                final_predictions = self.meta_model.predict(base_predictions)
            except Exception as e:
                logger.warning(f"Meta-learner prediction failed: {e}, using equal weights")
                final_predictions = base_predictions.mean(axis=1)
        else:
            pass
    pass
        
        if return_variance:
                weights = np.abs(self.meta_model.coef_)
                weights = weights / weights.sum()  # Normalize
            else:
                pass
    pass
            
                final_variance = np.sum(weights * base_variances, axis=1)
            return final_predictions, final_variance
        
        return final_predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform cross-validation on stacking ensemble

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Performance metrics
 """
 from ..core.exceptions import KrigingError

 if not hasattr(self, 'x') or self.x is None:
    pass

     from ..validation.cross_validation import leave_one_out
 from ..validation.metrics import mean_squared_error, r_squared

 predictions = leave_one_out(self, self.x, self.y, self.z)

 metrics = {
 'mse': mean_squared_error(self.z, predictions),
 'r2': r_squared(self.z, predictions)
 }

 return predictions, metrics
