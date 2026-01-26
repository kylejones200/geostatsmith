"""
Lognormal Kriging

Implementation based on ofr20091103.txt §2805-2810, §3006-3031:
"Lognormal kriging [is] simply any of the first three forms of kriging
applied to transformed data."

For lognormal data (common in mining, environmental science), kriging is
performed in log-space and back-transformed with proper variance correction.

The back-transformation must account for the bias introduced by Jensen's
inequality: E[exp(Y)] ≠ exp(E[Y]) for lognormal Y.

Reference:
- ofr20091103.txt (USGS Practical Primer)
- Lognormal transformation and kriging (§2162-2176, §2805-2810)
- Back-transformation considerations (§3622-3624)
"""

from typing import Tuple, Optional, Union
import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from .simple_kriging import SimpleKriging
from .ordinary_kriging import OrdinaryKriging


class LognormalKriging(BaseKriging):
    """
    Lognormal Kriging
    
    Performs kriging in log-space and back-transforms to original space
    with proper variance correction for the lognormal distribution.
    
    For lognormal data Z = exp(Y) where Y ~ Normal:
    1. Transform: Y = log(Z)
    2. Krige Y to get Ŷ and σ²ʏ
    3. Back-transform: Ẑ = exp(Ŷ + σ²ʏ/2)  # Variance correction
    
    This is critical for unbiased estimates in original space.
    """
    
    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        variogram_model: Optional[object] = None,
        kriging_type: str = 'ordinary',
        mean_log: Optional[float] = None,
    ):
        """
        Initialize Lognormal Kriging
        
        Parameters
        ----------
        x, y : np.ndarray
            Coordinates of sample points
        z : np.ndarray
            Values at sample points (must be > 0)
        variogram_model : VariogramModelBase, optional
            Fitted variogram model (fit to log-transformed data)
        kriging_type : str
            'simple' or 'ordinary' kriging in log-space
        mean_log : float, optional
            Mean of log-transformed data (required for simple kriging)
        """
        super().__init__(x, y, z, variogram_model)
        
        # Validate
        self.x, self.y = validate_coordinates(x, y)
        self.z = validate_values(z, n_expected=len(self.x))
        
        # Check for non-positive values
        if np.any(self.z <= 0):
            raise ValueError(
                "Lognormal kriging requires all values > 0. "
                "Consider adding a constant or using a different method."
            )
        
        # Transform to log-space
        self.log_z = np.log(self.z)
        self.mean_log = mean_log if mean_log is not None else np.mean(self.log_z)
        
        # Select kriging method
        self.kriging_type = kriging_type.lower()
        
        if self.kriging_type == 'simple':
            self.kriging_engine = SimpleKriging(
                self.x, self.y, self.log_z,
                variogram_model=variogram_model,
                known_mean=self.mean_log
            )
        elif self.kriging_type == 'ordinary':
            self.kriging_engine = OrdinaryKriging(
                self.x, self.y, self.log_z,
                variogram_model=variogram_model
            )
        else:
            raise ValueError(f"kriging_type must be 'simple' or 'ordinary', got {kriging_type}")
    
    def predict(
        self,
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        return_variance: bool = True,
        back_transform_method: str = 'unbiased'
    ) -> Union[npt.NDArray[np.float64], Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """
        Predict at new locations using lognormal kriging
        
        Parameters
        ----------
        x_new, y_new : np.ndarray
            Coordinates of prediction points
        return_variance : bool
            If True, return both predictions and variance
        back_transform_method : str
            Method for back-transformation:
            - 'unbiased': exp(Ŷ + σ²/2) - proper unbiased estimator
            - 'median': exp(Ŷ) - returns median of lognormal
            - 'simple': exp(Ŷ) - simple back-transform (biased low)
            
        Returns
        -------
        predictions : np.ndarray
            Predicted values in original space
        variance : np.ndarray, optional
            Variance in original space (approximate)
        """
        if self.variogram_model is None:
            raise KrigingError("Variogram model required for prediction")
        
        # Krige in log-space
        if return_variance:
            log_predictions, log_variances = self.kriging_engine.predict(
                x_new, y_new, return_variance=True
            )
        else:
            log_predictions = self.kriging_engine.predict(
                x_new, y_new, return_variance=False
            )
            log_variances = None
        
        # Back-transform to original space using dispatch
        def unbiased_transform(log_pred, log_var):
            if log_var is not None:
                return np.exp(log_pred + log_var / 2)
            return np.exp(log_pred)
        
        back_transform_methods = {
            'unbiased': unbiased_transform,
            'median': lambda lp, lv: np.exp(lp),
            'simple': lambda lp, lv: np.exp(lp),
        }
        
        if back_transform_method not in back_transform_methods:
            valid_methods = ', '.join(back_transform_methods.keys())
            raise ValueError(
                f"back_transform_method must be one of {valid_methods}, "
                f"got '{back_transform_method}'"
            )
        
        predictions = back_transform_methods[back_transform_method](
            log_predictions, log_variances
        )
        
        # Transform variance to original space (approximate)
        if return_variance and log_variances is not None:
            # Approximation: Var[Z] ≈ Z² * (exp(σ²ʏ) - 1)
            # where Z = exp(Ŷ)
            original_variances = predictions**2 * (np.exp(log_variances) - 1)
            return predictions, original_variances
        
        return predictions
    
    def cross_validate(
        self,
        back_transform_method: str = 'unbiased'
    ) -> Tuple[npt.NDArray[np.float64], dict]:
        """
        Leave-one-out cross-validation
        
        Parameters
        ----------
        back_transform_method : str
            Method for back-transformation
            
        Returns
        -------
        errors : np.ndarray
            Prediction errors in original space
        metrics : dict
            Error statistics
        """
        n = len(self.x)
        predictions = np.zeros(n)
        
        for i in range(n):
            # Leave out point i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            # Create temporary kriging object
            lnk_temp = LognormalKriging(
                self.x[mask],
                self.y[mask],
                self.z[mask],
                self.variogram_model,
                kriging_type=self.kriging_type,
                mean_log=self.mean_log if self.kriging_type == 'simple' else None
            )
            
            # Predict at left-out point
            pred = lnk_temp.predict(
                np.array([self.x[i]]),
                np.array([self.y[i]]),
                return_variance=False,
                back_transform_method=back_transform_method
            )
            predictions[i] = pred[0]
        
        # Calculate errors in original space
        errors = self.z - predictions
        
        metrics = {
            'MSE': np.mean(errors**2),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAE': np.mean(np.abs(errors)),
            'R2': 1 - np.sum(errors**2) / np.sum((self.z - np.mean(self.z))**2),
            'bias': np.mean(errors),
            'predictions': predictions,
        }
        
        return errors, metrics
    
    def get_log_statistics(self) -> dict:
        """
        Get statistics of the log-transformed data
        
        Returns
        -------
        dict
            Statistics in both original and log space
        """
        return {
            'original_mean': np.mean(self.z),
            'original_std': np.std(self.z),
            'original_min': np.min(self.z),
            'original_max': np.max(self.z),
            'log_mean': np.mean(self.log_z),
            'log_std': np.std(self.log_z),
            'log_min': np.min(self.log_z),
            'log_max': np.max(self.log_z),
        }
