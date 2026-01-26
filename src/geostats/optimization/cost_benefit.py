"""
Cost-Benefit Analysis
=====================

Functions for sample size estimation and cost-benefit analysis.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Dict, Any
from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..models.base_model import VariogramModelBase
import logging

logger = logging.getLogger(__name__)

def _rmse(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
 """Calculate root mean squared error."""
 return np.sqrt(np.mean((y_true - y_pred) ** 2))

def sample_size_calculator(
 x_initial: npt.NDArray[np.float64],
 y_initial: npt.NDArray[np.float64],
 z_initial: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 target_rmse: float,
 x_bounds: Optional[Tuple[float, float]] = None,
 y_bounds: Optional[Tuple[float, float]] = None,
 max_samples: int = 500,
 n_simulations: int = 10,
) -> Dict[str, Any]:
 """
 Estimate the number of samples needed to achieve target accuracy.

 Uses cross-validation to estimate how prediction error decreases
 with increasing sample size.

 Parameters
 ----------
 x_initial : ndarray
 Initial X coordinates
 y_initial : ndarray
 Initial Y coordinates
 z_initial : ndarray
 Initial values
 variogram_model : VariogramModelBase
 Fitted variogram model
 target_rmse : float
 Target RMSE for predictions
 x_bounds : tuple, optional
 Domain bounds for X
 y_bounds : tuple, optional
 Domain bounds for Y
 max_samples : int, default=500
 Maximum samples to consider
 n_simulations : int, default=10
 Number of Monte Carlo simulations

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'required_samples': Estimated number of samples needed
 - 'current_rmse': RMSE with current samples
 - 'target_rmse': Target RMSE
 - 'sample_sizes': Array of sample sizes evaluated
 - 'rmse_values': Corresponding RMSE values
 - 'confidence_90': 90% confidence interval

 Examples
 --------
 >>> from geostats.optimization import sample_size_calculator
 >>>
 >>> results = sample_size_calculator(
 ... x, y, z,
 ... variogram_model=model,
 ... target_rmse=0.5
 ... )
 >>> logger.info(f"Need approximately {results['required_samples']} samples")
 >>> logger.info(f"Current RMSE: {results['current_rmse']:.3f}")

 Notes
 -----
 This uses a power law relationship: RMSE = a * n^b
 where n is sample size. Parameters are estimated empirically.

 References
 ----------
 Webster, R., & Oliver, M. A. (2007). Geostatistics for Environmental
 Scientists. Wiley.
 """
 n_initial = len(x_initial)

 # Set bounds
 if x_bounds is None:
 x_bounds = (x_initial.min(), x_initial.max())
 if y_bounds is None:
 y_bounds = (y_initial.min(), y_initial.max())

 # Evaluate different sample sizes
 sample_sizes = np.linspace(n_initial, min(max_samples, n_initial * 10), 20, dtype=int)
 rmse_values = []
 rmse_std = []

 for n in sample_sizes:
 rmse_sim = []

 for _ in range(n_simulations):
 # Randomly select n samples
 if n <= n_initial:
 indices = np.random.choice(n_initial, n, replace=False)
 x_train = x_initial[indices]
 y_train = y_initial[indices]
 z_train = z_initial[indices]

 # Test on remaining points
 mask = np.ones(n_initial, dtype=bool)
 mask[indices] = False
 x_test = x_initial[mask]
 y_test = y_initial[mask]
 z_test = z_initial[mask]
 else:
 # Need to simulate additional samples
 # Use initial data plus synthetic samples
 n_synthetic = n - n_initial
 x_syn = np.random.uniform(x_bounds[0], x_bounds[1], n_synthetic)
 y_syn = np.random.uniform(y_bounds[0], y_bounds[1], n_synthetic)

 # Predict values at synthetic locations
 krig = OrdinaryKriging(
 x=x_initial,
 y=y_initial,
 z=z_initial,
 variogram_model=variogram_model,
 )
 z_syn, _ = krig.predict(x_syn, y_syn, return_variance=True)

 x_train = np.concatenate([x_initial, x_syn])
 y_train = np.concatenate([y_initial, y_syn])
 z_train = np.concatenate([z_initial, z_syn])

 # Test on random points in domain
 n_test = max(50, n_initial)
 x_test = np.random.uniform(x_bounds[0], x_bounds[1], n_test)
 y_test = np.random.uniform(y_bounds[0], y_bounds[1], n_test)
 z_test, _ = krig.predict(x_test, y_test, return_variance=True)

 # Train and test
 if len(x_test) > 0:
 krig_train = OrdinaryKriging(
 x=x_train,
 y=y_train,
 z=z_train,
 variogram_model=variogram_model,
 )
 z_pred, _ = krig_train.predict(x_test, y_test, return_variance=True)
 rmse_sim.append(_rmse(z_test, z_pred))

 rmse_values.append(np.mean(rmse_sim))
 rmse_std.append(np.std(rmse_sim))

 rmse_values = np.array(rmse_values)
 rmse_std = np.array(rmse_std)

 # Fit power law: RMSE = a * n^b
 # log(RMSE) = log(a) + b * log(n)
 log_n = np.log(sample_sizes)
 log_rmse = np.log(rmse_values + 1e-10)

 # Linear regression in log space
 coeffs = np.polyfit(log_n, log_rmse, 1)
 b = coeffs[0]
 log_a = coeffs[1]
 a = np.exp(log_a)

 # Estimate required sample size
 # target_rmse = a * n^b => n = (target_rmse / a)^(1/b)
 if b < 0: # Power law should have negative exponent
 required_samples = int((target_rmse / a) ** (1 / b))
 else:
 required_samples = max_samples # Can't achieve target

 # Current RMSE
 current_rmse = rmse_values[0]

 # Confidence interval (90%)
 confidence_90 = 1.645 * rmse_std # 90% CI

 return {
 'required_samples': required_samples,
 'current_rmse': current_rmse,
 'target_rmse': target_rmse,
 'sample_sizes': sample_sizes,
 'rmse_values': rmse_values,
 'confidence_90': confidence_90,
 'power_law_params': {'a': a, 'b': b},
 'achievable': required_samples <= max_samples,
 }

def cost_benefit_analysis(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 cost_per_sample: float,
 benefit_per_rmse_reduction: float,
 max_budget: float,
) -> Dict[str, Any]:
 """
 Perform cost-benefit analysis for sampling.

 Determines optimal number of samples by balancing cost of sampling
 against benefit of improved predictions.

 Parameters
 ----------
 x : ndarray
 Current X coordinates
 y : ndarray
 Current Y coordinates
 z : ndarray
 Current values
 variogram_model : VariogramModelBase
 Fitted variogram model
 cost_per_sample : float
 Cost to collect one additional sample
 benefit_per_rmse_reduction : float
 Benefit (in same units as cost) per unit RMSE reduction
 max_budget : float
 Maximum budget available

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'optimal_n_samples': Optimal number of samples
 - 'optimal_total_cost': Total cost at optimum
 - 'optimal_net_benefit': Net benefit at optimum
 - 'sample_sizes': Array of sample sizes evaluated
 - 'net_benefits': Net benefit for each sample size

 Examples
 --------
 >>> results = cost_benefit_analysis(
 ... x, y, z,
 ... variogram_model=model,
 ... cost_per_sample=100, # $100 per sample
 ... benefit_per_rmse_reduction=1000, # $1000 per RMSE unit reduced
 ... max_budget=10000 # $10,000 budget
 ... )
 >>> logger.info(f"Optimal: {results['optimal_n_samples']} samples")
 >>> logger.info(f"Net benefit: ${results['optimal_net_benefit']:.2f}")

 Notes
 -----
 Net benefit = (RMSE_reduction * benefit_per_rmse) - (n_new * cost_per_sample)

 Optimal sample size maximizes net benefit subject to budget constraint.
 """
 n_current = len(x)
 max_samples = int(max_budget / cost_per_sample) + n_current

 # Get RMSE curve
 size_results = sample_size_calculator(
 x, y, z,
 variogram_model=variogram_model,
 target_rmse=0.1, # Arbitrary target
 max_samples=max_samples,
 )

 sample_sizes = size_results['sample_sizes']
 rmse_values = size_results['rmse_values']
 baseline_rmse = rmse_values[0]

 # Calculate costs and benefits
 n_additional = sample_sizes - n_current
 costs = n_additional * cost_per_sample

 # Benefits from RMSE reduction
 rmse_reductions = baseline_rmse - rmse_values
 benefits = rmse_reductions * benefit_per_rmse_reduction

 # Net benefit
 net_benefits = benefits - costs

 # Apply budget constraint
 feasible = costs <= max_budget
 net_benefits[~feasible] = -np.inf

 # Find optimum
 optimal_idx = np.argmax(net_benefits)
 optimal_n_samples = int(sample_sizes[optimal_idx])
 optimal_total_cost = costs[optimal_idx]
 optimal_net_benefit = net_benefits[optimal_idx]

 return {
 'optimal_n_samples': optimal_n_samples,
 'optimal_n_additional': int(n_additional[optimal_idx]),
 'optimal_total_cost': optimal_total_cost,
 'optimal_net_benefit': optimal_net_benefit,
 'optimal_rmse': rmse_values[optimal_idx],
 'baseline_rmse': baseline_rmse,
 'sample_sizes': sample_sizes,
 'costs': costs,
 'benefits': benefits,
 'net_benefits': net_benefits,
 }

def estimate_interpolation_error(
 x: npt.NDArray[np.float64],
 y: npt.NDArray[np.float64],
 z: npt.NDArray[np.float64],
 variogram_model: VariogramModelBase,
 x_pred: npt.NDArray[np.float64],
 y_pred: npt.NDArray[np.float64],
 confidence_level: float = 0.95,
) -> Dict[str, npt.NDArray[np.float64]]:
 """
 Estimate interpolation error and confidence intervals.

 Parameters
 ----------
 x : ndarray
 Sample X coordinates
 y : ndarray
 Sample Y coordinates
 z : ndarray
 Sample values
 variogram_model : VariogramModelBase
 Fitted variogram model
 x_pred : ndarray
 Prediction X coordinates
 y_pred : ndarray
 Prediction Y coordinates
 confidence_level : float, default=0.95
 Confidence level for intervals (0-1)

 Returns
 -------
 results : dict
 Dictionary containing:
 - 'predictions': Predicted values
 - 'std_errors': Standard errors
 - 'lower_bound': Lower confidence bound
 - 'upper_bound': Upper confidence bound
 - 'relative_error': Relative error (%)

 Examples
 --------
 >>> results = estimate_interpolation_error(
 ... x, y, z,
 ... variogram_model=model,
 ... x_pred=x_grid,
 ... y_pred=y_grid,
 ... confidence_level=0.95
 ... )
 >>>
 >>> # Plot with confidence bands
 >>> plt.plot(x_pred, results['predictions'], 'b-')
 >>> plt.fill_between(
 ... x_pred,
 ... results['lower_bound'],
 ... results['upper_bound'],
 ... alpha=0.3
 ... )

 Notes
 -----
 Uses kriging variance to construct confidence intervals.
 Assumes Gaussian errors.
 """
 # Perform kriging
 krig = OrdinaryKriging(
 x=x,
 y=y,
 z=z,
 variogram_model=variogram_model,
 )

 predictions, variance = krig.predict(x_pred, y_pred, return_variance=True)

 # Standard errors
 std_errors = np.sqrt(variance)

 # Confidence interval multiplier (z-score)
 from scipy.stats import norm
 z_score = norm.ppf((1 + confidence_level) / 2)

 # Confidence bounds
 margin = z_score * std_errors
 lower_bound = predictions - margin
 upper_bound = predictions + margin

 # Relative error (as percentage)
 relative_error = 100 * std_errors / (np.abs(predictions) + 1e-10)

 return {
 'predictions': predictions,
 'variance': variance,
 'std_errors': std_errors,
 'lower_bound': lower_bound,
 'upper_bound': upper_bound,
 'relative_error': relative_error,
 'confidence_level': confidence_level,
 }
