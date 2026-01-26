"""
Optimization Tools
==================

Tools for optimizing sampling strategies and resource allocation.

Key Features:
- Optimal sampling design
- Sample size estimation
- Infill sampling strategies
- Cost-benefit analysis

Examples
--------
>>> from geostats.optimization import optimal_sampling_design, sample_size_calculator
>>>
>>> # Design optimal sampling network
>>> x_new, y_new = optimal_sampling_design(
... x_existing, y_existing, z_existing,
... n_new_samples=20,
... variogram_model=model,
... strategy='variance_reduction'
... )
>>>
>>> # Calculate required sample size
>>> n_required = sample_size_calculator(
... x, y, z,
... target_rmse=0.5,
... variogram_model=model
... )
"""

from .sampling_design import (
 optimal_sampling_design,
 infill_sampling,
 stratified_sampling,
 adaptive_sampling,
)

from .cost_benefit import (
 sample_size_calculator,
 cost_benefit_analysis,
 estimate_interpolation_error,
)

__all__ = [
 # Sampling design
 'optimal_sampling_design',
 'infill_sampling',
 'stratified_sampling',
 'adaptive_sampling',
 # Cost-benefit analysis
 'sample_size_calculator',
 'cost_benefit_analysis',
 'estimate_interpolation_error',
]
