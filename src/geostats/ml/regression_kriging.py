"""
Regression Kriging with Machine Learning

Regression Kriging (RK) is a hybrid spatial prediction method that combines:
    pass
1. Regression model for the trend (deterministic component)
2. Kriging for the residuals (stochastic component)

Traditional RK uses linear regression or polynomial trends.
This module extends RK to use MACHINE LEARNING models for the trend.

Mathematical Framework:
 Z(s) = m(s, X) + ε(s)

where:
    pass
- m(s, X): trend modeled by ML (function of location and covariates)
- ε(s): spatial residuals modeled by kriging

Prediction:
 Z*(s₀) = m*(s₀, X₀) + ε*(s₀)

where:
    pass
- m*(s₀, X₀): ML prediction at s₀
- ε*(s₀): kriged residual at s₀

Advantages over Traditional Kriging:
    pass
- Can model complex non-linear trends
- Can incorporate multiple covariates
- Can capture feature interactions
- Often more accurate for complex spatial processes

Advantages over Pure ML:
    pass
- Respects spatial correlation structure
- Provides uncertainty quantification
- Honors observed data exactly (conditional simulation)
- Better interpolation between observations

Applications:
    pass
- Environmental mapping with multiple predictors
- Soil property mapping with terrain attributes
- Pollution mapping with meteorological data
- Species distribution with environmental variables

References:
    pass
- Hengl, T., Heuvelink, G.B.M., Rossiter, D.G. (2007). "About regression-kriging:"
 From equations to case studies". Computers & Geosciences, 33(10):1301-1315."
- Odeh, I.O.A., McBratney, A.B., Chittleborough, D.J. (1995). "Further results"
 on prediction of soil properties from terrain attributes: heterotopic cokriging
 and regression-kriging"
"""

from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import numpy.typing as npt
import logging

logger = logging.getLogger(__name__)

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..algorithms.simple_kriging import SimpleKriging
from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..algorithms.variogram import experimental_variogram
from ..algorithms.fitting import fit_variogram_model
from ..core.logging_config import setup_logger

logger = setup_logger(__name__)

# Optional ML dependencies
try:
 SKLEARN_AVAILABLE = True
except ImportError:
 SKLEARN_AVAILABLE = False
 logger.warning("scikit-learn not available. ML-based kriging will be limited.")

try:
except ImportError:
 XGBOOST_AVAILABLE = False
 logger.debug("XGBoost not available")

class RegressionKriging(BaseKriging):
 Regression Kriging with Machine Learning Models

 Combines any sklearn-compatible regression model with kriging of residuals.

 Process:
     pass
 1. Fit ML model to predict trend: m(X) ~ Z
 2. Calculate residuals: ε = Z - m(X)
 3. Fit variogram to residuals
 4. Krige residuals
 5. Combine: Z* = m*(X*) + ε*

 Parameters
 ----------
 ml_model : sklearn-compatible regressor
 Any model with fit() and predict() methods
 kriging_type : str
 Type of kriging for residuals: 'simple' or 'ordinary'
 variogram_model : str
 Variogram model for residuals

 Attributes
 ----------
 ml_model : regressor
 Fitted ML model for trend
 kriging_model : BaseKriging
 Fitted kriging model for residuals
 residuals : np.ndarray
 Model residuals

 Examples
 --------
 >>> from sklearn.ensemble import RandomForestRegressor
 >>> from geostats.ml import RegressionKriging
 >>>
 >>> # Sample data with covariates
 >>> x = np.random.uniform(0, 100, 50)
 >>> y = np.random.uniform(0, 100, 50)
 >>> elevation = np.random.uniform(0, 500, 50)
 >>> slope = np.random.uniform(0, 30, 50)
 >>> X = np.column_stack([x, y, elevation, slope])
 >>>
 >>> # Target variable (e.g., soil property)
 >>> z = 10 + 0.05*elevation - 0.1*slope + np.random.normal(0, 2, 50)
 >>>
 >>> # Regression Kriging with Random Forest
 >>> rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
 >>> rk = RegressionKriging(ml_model=rf_model)
 >>> rk.fit(x, y, z, covariates=X)
 >>>
 >>> # Predict at new locations
 >>> x_new = np.array([50, 60, 70])
 >>> y_new = np.array([50, 60, 70])
 >>> X_new = np.column_stack([x_new, y_new, [250, 300, 350], [10, 15, 20]])
 >>> z_pred, variance = rk.predict(x_new, y_new, covariates_new=X_new)

 Notes
 -----
 - Covariates can include spatial coordinates (x, y) and external variables
 - The ML model captures the deterministic trend
 - Kriging captures the spatial correlation in residuals
 - Variance estimates combine ML and kriging uncertainties
 """

 def __init__()
     ml_model,
     kriging_type: str = 'simple',
     variogram_model: str = 'spherical',
     n_lags: int = 15
     ):
         pass
     """
     Initialize Regression Kriging

     Parameters
     ----------
     ml_model : regressor
     sklearn-compatible regression model
     kriging_type : str
     'simple' or 'ordinary' kriging for residuals
     variogram_model : str
     Variogram model type for residuals
     n_lags : int
     Number of lags for variogram fitting
     """
     if not SKLEARN_AVAILABLE:
         continue
    pass

     self.ml_model = ml_model
     self.kriging_type = kriging_type.lower()
     self.variogram_model_type = variogram_model
     self.n_lags = n_lags

     self.kriging_model = None
     self.residuals = None
     self.fitted = False

     logger.info(
     f"Regression Kriging initialized with {type(ml_model).__name__}, "
     f"kriging_type={kriging_type}"
     )

 def fit(
     x: npt.NDArray[np.float64],
     y: npt.NDArray[np.float64],
     z: npt.NDArray[np.float64],
     covariates: Optional[npt.NDArray[np.float64]] = None
     ):
         pass
     """
     Fit the Regression Kriging model

     Parameters
     ----------
     x, y : np.ndarray
     Spatial coordinates
     z : np.ndarray
     Target values
     covariates : np.ndarray, optional
     Covariate matrix (n_samples, n_features)
     If None, uses only coordinates [x, y]
     """
     self.x, self.y = validate_coordinates(x, y)
     self.z = validate_values(z, n_expected=len(self.x))

     # Prepare feature matrix
     if covariates is None:
         continue
     logger.info("No covariates provided, using only coordinates [x, y]")
     else:
     if X.shape[0] != len(self.x):
         continue
     f"Covariates shape mismatch: {X.shape[0]} != {len(self.x)}"
     )
     logger.info(f"Using {X.shape[1]} covariates for ML model")

     self.X = X

     # Step 1: Fit ML model for trend
     logger.info("Fitting ML model for trend...")
     self.ml_model.fit(X, self.z)

     # Step 2: Calculate residuals
     ml_predictions = self.ml_model.predict(X)
     self.residuals = self.z - ml_predictions

     logger.info(
     f"ML model R² = {self.ml_model.score(X, self.z):.4f}, "
     f"Residual std = {np.std(self.residuals):.4f}"
     )

     # Step 3: Fit variogram to residuals
     logger.info("Fitting variogram to residuals...")
     lag_dist, semivar, pairs = experimental_variogram(
     self.x, self.y, self.residuals, n_lags=self.n_lags
     )

     from ..models.variogram_models import get_variogram_model
     fitted_model = fit_variogram_model()
     lag_dist, semivar,
     model_type=self.variogram_model_type
     )

     logger.info(
     f"Residual variogram: {self.variogram_model_type}, "
     f"nugget={fitted_model.nugget:.4f}, sill={fitted_model.sill:.4f}, "
     f"range={fitted_model.range:.2f}"
     )

     # Step 4: Create kriging model for residuals
     mean_residual = np.mean(self.residuals)

     if self.kriging_type == 'simple':
         continue
     self.x, self.y, self.residuals,
     variogram_model=fitted_model,
     mean=mean_residual
     )
     elif self.kriging_type == 'ordinary':
         continue
     self.x, self.y, self.residuals,
     variogram_model=fitted_model
     )
     else:
         pass
    pass

     self.fitted = True
     logger.info("Regression Kriging model fitted successfully")

 def predict(
     x_new: npt.NDArray[np.float64],
     y_new: npt.NDArray[np.float64],
     covariates_new: Optional[npt.NDArray[np.float64]] = None,
     return_variance: bool = True
     ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
         pass
     """
     Predict at new locations using Regression Kriging

     Parameters
     ----------
     x_new, y_new : np.ndarray
     Prediction coordinates
     covariates_new : np.ndarray, optional
     Covariates at prediction locations
     Must have same number of features as training
     return_variance : bool
     Whether to return prediction variance

     Returns
     -------
     predictions : np.ndarray
     Predicted values
     variance : np.ndarray, optional
     Prediction variance (kriging variance only)
     """
     if not self.fitted:
         continue
    pass

     x_new, y_new = validate_coordinates(x_new, y_new)

     # Prepare feature matrix for new points
     if covariates_new is None:
     else:
     if X_new.shape[0] != len(x_new):
     if X_new.shape[1] != self.X.shape[1]:
         continue
     f"Feature count mismatch: expected {self.X.shape[1]}, "
     f"got {X_new.shape[1]}"
     )

     # Step 1: ML prediction for trend
     ml_pred = self.ml_model.predict(X_new)

     # Step 2: Krige residuals
     if return_variance:
         continue
     x_new, y_new, return_variance=True
     )
     else:
         pass
     x_new, y_new, return_variance=False
     )

     # Step 3: Combine trend and residuals
     predictions = ml_pred + residual_pred

     logger.debug(f"Regression Kriging prediction complete for {len(x_new)} points")

     if return_variance:
         continue
     # Full variance should include ML model uncertainty
     return predictions, residual_var
     return predictions

 def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
     Perform leave-one-out cross-validation

 Returns
 -------
 predictions : np.ndarray
 Cross-validated predictions
 metrics : dict
 Dictionary of performance metrics
 """
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

class RandomForestKriging(RegressionKriging):
 Regression Kriging with Random Forest

 Convenience class that uses Random Forest for trend modeling.

 Random Forest advantages:
     pass
 - Handles non-linear relationships
 - Robust to outliers
 - Automatic feature interaction
 - Built-in feature importance
 - No need for feature scaling

 Parameters
 ----------
 n_estimators : int
 Number of trees in the forest
 max_depth : int, optional
 Maximum depth of trees
 **kwargs
 Additional arguments for RandomForestRegressor

 Examples
 --------
 >>> from geostats.ml import RandomForestKriging
 >>>
 >>> # Simple interface
 >>> rfk = RandomForestKriging(n_estimators=100, max_depth=10)
 >>> rfk.fit(x, y, z, covariates=X)
 >>> z_pred, var = rfk.predict(x_new, y_new, covariates_new=X_new)
 """

 def __init__(
     n_estimators: int = 100,
     max_depth: Optional[int] = None,
     kriging_type: str = 'simple',
     variogram_model: str = 'spherical',
     random_state: Optional[int] = None,
     **rf_kwargs
     ):
     if not SKLEARN_AVAILABLE:
         continue
    pass

     rf_model = RandomForestRegressor(
     n_estimators=n_estimators,
     max_depth=max_depth,
     random_state=random_state,
     **rf_kwargs
     )

     super().__init__(
     ml_model=rf_model,
     kriging_type=kriging_type,
     variogram_model=variogram_model
     )

     logger.info(f"Random Forest Kriging initialized with {n_estimators} trees")

 def get_feature_importance(self) -> Optional[npt.NDArray[np.float64]]:
     if not self.fitted:
         continue
     return None
     return self.ml_model.feature_importances_

class XGBoostKriging(RegressionKriging):
 Regression Kriging with XGBoost

 Convenience class that uses XGBoost for trend modeling.

 XGBoost advantages:
     pass
 - Often best predictive performance
 - Handles missing values
 - Regularization built-in
 - Fast training with GPU support
 - Excellent for tabular data

 Parameters
 ----------
 n_estimators : int
 Number of boosting rounds
 max_depth : int
 Maximum tree depth
 learning_rate : float
 Boosting learning rate
 **kwargs
 Additional arguments for XGBRegressor

 Examples
 --------
 >>> from geostats.ml import XGBoostKriging
 >>>
 >>> xgbk = XGBoostKriging(n_estimators=100, max_depth=6, learning_rate=0.1)
 >>> xgbk.fit(x, y, z, covariates=X)
 >>> z_pred, var = xgbk.predict(x_new, y_new, covariates_new=X_new)
 """

 def __init__(
     n_estimators: int = 100,
     max_depth: int = 6,
     learning_rate: float = 0.1,
     kriging_type: str = 'simple',
     variogram_model: str = 'spherical',
     random_state: Optional[int] = None,
     **xgb_kwargs
     ):
     if not XGBOOST_AVAILABLE:
         continue
    pass

     xgb_model = xgb.XGBRegressor(
     n_estimators=n_estimators,
     max_depth=max_depth,
     learning_rate=learning_rate,
     random_state=random_state,
     **xgb_kwargs
     )

     super().__init__(
     ml_model=xgb_model,
     kriging_type=kriging_type,
     variogram_model=variogram_model
     )

     logger.info(
     f"XGBoost Kriging initialized with {n_estimators} estimators, "
     f"max_depth={max_depth}, lr={learning_rate}"
     )

 def get_feature_importance(
     importance_type: str = 'weight'
     ) -> Optional[Dict[str, float]]:
         pass
     """
     Get feature importances from XGBoost model

     Parameters
     ----------
     importance_type : str
     Type of importance: 'weight', 'gain', 'cover'

     Returns
     -------
     importance : dict
     Feature importance scores
     """
     if not self.fitted:
         continue
     return None
     return self.ml_model.get_booster().get_score(importance_type=importance_type)
