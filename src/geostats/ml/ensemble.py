"""
Ensemble Methods for Geostatistics

Ensemble methods combine multiple models to improve prediction accuracy and
robustness. This module provides hybrid ensemble approaches that combine:

1. Multiple kriging models
2. Multiple variogram models
3. Kriging with ML models
4. Bootstrap aggregating (bagging)
5. Model stacking

Key Concepts:
- Ensemble averaging: Combine predictions from multiple models
- Bagging: Train models on bootstrap samples
- Stacking: Use meta-learner to combine base models
- Weighted averaging: Combine models with performance-based weights

Advantages:
- Reduced prediction variance
- Better generalization
- Robustness to model misspecification
- Captures uncertainty across models

Applications:
- Uncertain variogram models
- Multiple spatial scales
- Combining local and global models
- Model uncertainty quantification

References:
- Goovaerts, P. (1997). "Geostatistics for Natural Resources Evaluation"
- Dietterich, T.G. (2000). "Ensemble methods in machine learning"
- Wolpert, D.H. (1992). "Stacked generalization"
"""

from typing import List, Optional, Tuple, Union, Dict, Callable, Any
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from ..core.base import BaseKriging
from ..core.validators import validate_coordinates, validate_values
from ..core.logging_config import setup_logger

logger = setup_logger(__name__)


class EnsembleKriging(BaseKriging):
    """
    Ensemble Kriging: Combine Multiple Kriging Models
    
    Aggregates predictions from multiple kriging models using weighted
    averaging. Weights can be:
    - Equal (simple average)
    - Cross-validation based (performance-weighted)
    - Variance-based (inverse variance weighting)
    
    Mathematical Framework:
    For ensemble of K models with predictions Z_k*(s) and weights w_k:
        Z_ensemble*(s) = Σ w_k · Z_k*(s)
        
    where Σ w_k = 1.
    
    Ensemble Variance:
        Var[Z_ensemble*] = Σ w_k² · Var[Z_k*] + 2Σ Σ w_i·w_j·Cov[Z_i*, Z_j*]
        
    If models are independent: Var[Z_ensemble*] = Σ w_k² · Var[Z_k*]
    
    Parameters
    ----------
    models : list of BaseKriging
        List of fitted kriging models
    weighting : str
        Weighting scheme:
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
    ...     models=[ok, uk],
    ...     weighting='inverse_variance'
    ... )
    >>> 
    >>> # Predict
    >>> z_pred, var = ensemble.predict(x_new, y_new)
    """
    
    def __init__(
        self,
        models: List[BaseKriging],
        weighting: str = 'equal',
        combine_variance: bool = True
    ):
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
            raise ValueError("Must provide at least one model")
        
        self.models = models
        self.weighting = weighting.lower()
        self.combine_variance = combine_variance
        
        weighting_schemes = {'equal', 'inverse_variance', 'performance'}
        if self.weighting not in weighting_schemes:
            raise ValueError(f"weighting must be one of {weighting_schemes}")
        
        # Initialize with equal weights
        self.weights = np.ones(len(models)) / len(models)
        
        logger.info(
            f"Ensemble Kriging initialized with {len(models)} models, "
            f"weighting={weighting}"
        )
    
    def _compute_weights(
        self,
        variances: Optional[npt.NDArray[np.float64]] = None,
        scores: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
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
            raise ValueError(f"Unknown weighting: {self.weighting}")
        
        if self.weighting == 'inverse_variance' and variances is None:
            logger.warning("No variances provided, using equal weights")
            return weights_map['equal']()
        
        if self.weighting == 'performance' and scores is None:
            logger.warning("No scores provided, using equal weights")
            return weights_map['equal']()
        
        weights = weights_map[self.weighting]()
        
        # Normalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def predict(
        self,
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        return_variance: bool = True
    ) -> Union[
        npt.NDArray[np.float64],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ]:
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
            try:
                if return_variance:
                    pred, var = model.predict(x_new, y_new, return_variance=True)
                    all_predictions[i] = pred
                    all_variances[i] = var
                else:
                    all_predictions[i] = model.predict(
                        x_new, y_new, return_variance=False
                    )
            except Exception as e:
                logger.error(f"Model {i} prediction failed: {e}")
                raise
        
        # Compute weights
        if self.weighting == 'inverse_variance' and return_variance:
            # Use mean variance for each model
            mean_vars = np.mean(all_variances, axis=1)
            weights = self._compute_weights(variances=mean_vars)
        else:
            weights = self._compute_weights()
        
        self.weights = weights
        
        # Weighted average of predictions
        ensemble_pred = np.zeros(n_pred, dtype=np.float64)
        for i in range(n_models):
            ensemble_pred += weights[i] * all_predictions[i]
        
        logger.debug(
            f"Ensemble prediction complete: weights={weights.round(3)}"
        )
        
        if return_variance and self.combine_variance:
            # Ensemble variance (assuming independence)
            # Var[Σw_i·Z_i] = Σw_i²·Var[Z_i]
            ensemble_var = np.zeros(n_pred, dtype=np.float64)
            for i in range(n_models):
                ensemble_var += (weights[i] ** 2) * all_variances[i]
            
            return ensemble_pred, ensemble_var
        elif return_variance:
            # Return mean variance
            return ensemble_pred, np.mean(all_variances, axis=0)
        
        return ensemble_pred
    
    def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
        """
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
            raise KrigingError("No models in ensemble")
        
        # Use the first model's data for cross-validation
        first_model = self.models[0]
        if not hasattr(first_model, 'x') or first_model.x is None:
            raise KrigingError("Models must be fitted before cross-validation")
        
        from ..validation.cross_validation import leave_one_out
        from ..validation.metrics import mean_squared_error, r_squared
        
        predictions = leave_one_out(self, first_model.x, first_model.y, first_model.z)
        
        metrics = {
            'mse': mean_squared_error(first_model.z, predictions),
            'r2': r_squared(first_model.z, predictions)
        }
        
        return predictions, metrics


class BootstrapKriging(BaseKriging):
    """
    Bootstrap Aggregating (Bagging) for Kriging
    
    Creates ensemble by training models on bootstrap samples of the data.
    Reduces prediction variance through averaging.
    
    Process:
    1. Create B bootstrap samples (sample with replacement)
    2. Fit kriging model on each sample
    3. Average predictions from all models
    4. Estimate uncertainty from prediction variance
    
    Advantages:
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
    ...     base_model_type=OrdinaryKriging,
    ...     n_bootstrap=100,
    ...     sample_fraction=0.8,
    ...     variogram_model=spherical_model
    ... )
    >>> bk.fit(x, y, z)
    >>> z_pred, z_std = bk.predict(x_new, y_new)
    """
    
    def __init__(
        self,
        base_model_type: type,
        n_bootstrap: int = 100,
        sample_fraction: float = 1.0,
        random_state: Optional[int] = None,
        **model_kwargs
    ):
        """Initialize Bootstrap Kriging"""
        self.base_model_type = base_model_type
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        
        self.bootstrap_models = []
        self.fitted = False
        
        if not 0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        
        logger.info(
            f"Bootstrap Kriging initialized: n_bootstrap={n_bootstrap}, "
            f"sample_fraction={sample_fraction}"
        )
    
    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64]
    ):
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
            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_bootstrap_samples, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            z_boot = z[indices]
            
            # Fit model on bootstrap sample
            try:
                model = self.base_model_type(
                    x_boot, y_boot, z_boot, **self.model_kwargs
                )
                self.bootstrap_models.append(model)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {b} failed: {e}")
                continue
        
        self.fitted = True
        logger.info(
            f"Bootstrap fitting complete: {len(self.bootstrap_models)} successful models"
        )
    
    def predict(
        self,
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        return_std: bool = True
    ) -> Union[
        npt.NDArray[np.float64],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ]:
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
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        x_new, y_new = validate_coordinates(x_new, y_new)
        n_pred = len(x_new)
        n_models = len(self.bootstrap_models)
        
        # Collect all predictions
        all_predictions = np.zeros((n_models, n_pred), dtype=np.float64)
        
        for i, model in enumerate(self.bootstrap_models):
            try:
                pred = model.predict(x_new, y_new, return_variance=False)
                all_predictions[i] = pred
            except Exception as e:
                logger.warning(f"Bootstrap model {i} prediction failed: {e}")
                all_predictions[i] = np.nan
        
        # Mean prediction (bagging)
        predictions = np.nanmean(all_predictions, axis=0)
        
        logger.debug(f"Bootstrap prediction complete for {n_pred} points")
        
        if return_std:
            # Standard deviation across bootstrap samples
            std = np.nanstd(all_predictions, axis=0)
            return predictions, std
        
        return predictions
    
    def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
        """
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
            raise KrigingError("Model must be fitted before cross-validation")
        
        from ..validation.cross_validation import leave_one_out
        from ..validation.metrics import mean_squared_error, r_squared
        
        predictions = leave_one_out(self, self.x, self.y, self.z)
        
        metrics = {
            'mse': mean_squared_error(self.z, predictions),
            'r2': r_squared(self.z, predictions)
        }
        
        return predictions, metrics


class StackingKriging(BaseKriging):
    """
    Stacking: Meta-Learning for Kriging Ensemble
    
    Uses a meta-learner to optimally combine base kriging models.
    
    Process:
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
    ...     base_models=models,
    ...     meta_model=Ridge(alpha=1.0),
    ...     cv_folds=5
    ... )
    >>> stacking.fit(x, y, z)
    >>> z_pred = stacking.predict(x_new, y_new)
    """
    
    def __init__(
        self,
        base_models: List[BaseKriging],
        meta_model: Optional[Any] = None,
        cv_folds: int = 5
    ):
        """Initialize Stacking Kriging"""
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        
        if meta_model is None:
            # Default: simple linear combination
            try:
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0)
                logger.info("Using Ridge regression as meta-learner")
            except ImportError:
                logger.warning("sklearn not available, using equal weights")
                self.meta_model = None
        
        logger.info(
            f"Stacking Kriging initialized with {len(base_models)} base models, "
            f"cv_folds={cv_folds}"
        )
    
    def fit(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64]
    ):
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
        
        # Placeholder: Full implementation would require cross-validation
        # to generate meta-features
        
        logger.info("Stacking ensemble fitted")
        logger.warning("Stacking implementation is placeholder - uses simple averaging")
        
        # For now, just store data
        self.x = x
        self.y = y
        self.z = z
    
    def predict(
        self,
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        return_variance: bool = False
    ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Predict using stacking ensemble"""
        # Placeholder: Use simple averaging
        ensemble = EnsembleKriging(self.base_models, weighting='equal')
        return ensemble.predict(x_new, y_new, return_variance=return_variance)
    
    def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
        """
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
            raise KrigingError("Model must be fitted before cross-validation")
        
        from ..validation.cross_validation import leave_one_out
        from ..validation.metrics import mean_squared_error, r_squared
        
        predictions = leave_one_out(self, self.x, self.y, self.z)
        
        metrics = {
            'mse': mean_squared_error(self.z, predictions),
            'r2': r_squared(self.z, predictions)
        }
        
        return predictions, metrics
