"""
Validation metrics for geostatistical models
"""

from typing import Dict
import numpy as np
import numpy.typing as npt

def mean_squared_error(
 y_pred: npt.NDArray[np.float64],
    ) -> float:
 """Calculate Mean Squared Error"""
 return float(np.mean((y_true - y_pred) ** 2))

def root_mean_squared_error(
 y_pred: npt.NDArray[np.float64],
    ) -> float:
 """Calculate Root Mean Squared Error"""
 return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_error(
 y_pred: npt.NDArray[np.float64],
    ) -> float:
 """Calculate Mean Absolute Error"""
 return float(np.mean(np.abs(y_true - y_pred)))

def r_squared(
 y_pred: npt.NDArray[np.float64],
    ) -> float:
 """Calculate R-squared (coefficient of determination)"""
 ss_res = np.sum((y_true - y_pred) ** 2)
 ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

 if ss_tot == 0:
 if ss_tot == 0:

 return float(1.0 - (ss_res / ss_tot))

def calculate_metrics(
 y_pred: npt.NDArray[np.float64],
    ) -> Dict[str, float]:
 """
 Calculate all validation metrics

 Parameters
 ----------
 y_true : np.ndarray
 True values
 y_pred : np.ndarray
 Predicted values

 Returns
 -------
 dict
 Dictionary containing:
 - 'mse': Mean Squared Error
 - 'rmse': Root Mean Squared Error
 - 'mae': Mean Absolute Error
 - 'r2': R-squared
 - 'bias': Mean Error (bias)
 """
 return {
 'mse': mean_squared_error(y_true, y_pred),
 'rmse': root_mean_squared_error(y_true, y_pred),
 'mae': mean_absolute_error(y_true, y_pred),
 'r2': r_squared(y_true, y_pred),
 'bias': float(np.mean(y_true - y_pred)),
 }
