"""
Abstract base classes and interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import numpy.typing as npt

class BaseModel(ABC):
    """
    Abstract base class for all geostatistical models
    
    This provides a common interface for variogram models,
    covariance models, and other spatial models.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._parameters: Dict[str, float] = {}
        self._is_fitted: bool = False

    @abstractmethod
    def __call__(self, h: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Evaluate the model at distance h
        
        Parameters
        ----------
        h : np.ndarray
            Distance values
        
        Returns
        -------
        np.ndarray
            Model values at distance h
        """
        pass

    @property
    def parameters(self) -> Dict[str, float]:
        return self._parameters.copy()

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        lags: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> "BaseModel":
        """
        Fit the model to data
        
        Parameters
        ----------
        lags : np.ndarray
            Lag distances
        values : np.ndarray
            Values at each lag
        **kwargs
            Additional fitting parameters
        
        Returns
        -------
        self
            Fitted model
        """
        pass

    def set_parameters(self, **params: float) -> None:
        """
        Set model parameters manually
        
        Parameters
        ----------
        **params
            Model parameters to set
        """
        self._parameters.update(params)
        self._is_fitted = True

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v:.4f}" for k, v in self._parameters.items())
        return f"{self.__class__.__name__}({params_str})"

class BaseKriging(ABC):
    """
    Abstract base class for all kriging methods
    
    This provides a common interface for simple kriging,
    ordinary kriging, universal kriging, etc.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        variogram_model: Optional["VariogramModelBase"] = None,
    ) -> None:
        """
        Initialize kriging with data
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates of sample points
        y : np.ndarray
            Y coordinates of sample points
        z : np.ndarray
            Values at sample points
        variogram_model : VariogramModelBase, optional
            Fitted variogram model
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        self.variogram_model = variogram_model
        self.n_points = len(self.x)

    @abstractmethod
    def predict(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        return_variance: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        """
        Perform kriging prediction
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates for prediction
        y : np.ndarray
            Y coordinates for prediction
        return_variance : bool
            Whether to return kriging variance
        
        Returns
        -------
        predictions : np.ndarray
            Predicted values
        variance : np.ndarray or None
            Kriging variance (if return_variance=True)
        """
        pass

    @abstractmethod
    def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
        """
        Perform leave-one-out cross-validation
        
        Returns
        -------
        predictions : np.ndarray
            Cross-validated predictions at sample points
        metrics : dict
            Dictionary of validation metrics
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_points={self.n_points}, "
            f"model={self.variogram_model.__class__.__name__ if self.variogram_model else None}"
            f")"
        )
