"""
Example Workflow: Optimal Sampling Design
==========================================

Demonstrates how to:
1. Design optimal sampling networks
2. Perform cost-benefit analysis
3. Plan adaptive sampling campaigns
4. Estimate required sample sizes

Author: geostats development team
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# Try importing geostats
try:
try:
 infill_sampling,
 stratified_sampling,
 sample_size_calculator,
 cost_benefit_analysis,
 )
 from geostats.algorithms.ordinary_kriging import OrdinaryKriging
 from geostats.models.variogram_models import SphericalModel
 from geostats.algorithms.variogram import experimental_variogram
 from geostats.algorithms.fitting import fit_variogram
except ImportError:
 logger.info("Please install geostats: pip install -e .")
 exit(1)

def example_1_optimal_sampling():
 
 logger.info("Example 1: Optimal Sampling Design")
 

 # Initial sparse sampling
 logger.info("Creating initial sparse sample network...")
 np.random.seed(42)
 n_initial = 20
 x_init = np.random.uniform(0, 100, n_initial)
 y_init = np.random.uniform(0, 100, n_initial)
 z_init = 50 + 0.3*x_init + 0.2*y_init + 10*np.sin(x_init/20) + np.random.normal(0, 2, n_initial)

 # Fit variogram
 logger.info("Fitting variogram to initial data...")
 lags, gamma = experimental_variogram(x_init, y_init, z_init)
 variogram_model = fit_variogram(lags, gamma, model_type='spherical')
 params = variogram_model.get_parameters()
 logger.info(f" Range: {params['range']:.2f} m")

 # Design optimal locations for 10 new samples
 logger.info("\nDesigning optimal locations for 10 new samples...")

 # Strategy 1: Variance reduction
 x_var, y_var = optimal_sampling_design(
 x_init, y_init, z_init,
 n_new_samples=10,
 variogram_model=variogram_model,
 strategy='variance_reduction'
 )

 # Strategy 2: Space-filling
 x_space, y_space = optimal_sampling_design(
 x_init, y_init, z_init,
 n_new_samples=10,
 variogram_model=variogram_model,
 strategy='space_filling'
 )

 # Strategy 3: Hybrid
 x_hybrid, y_hybrid = optimal_sampling_design(
 x_init, y_init, z_init,
 n_new_samples=10,
 variogram_model=variogram_model,
 strategy='hybrid'
 )

 logger.info(" Optimal locations computed for 3 strategies")

 # Visualize
 fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 strategies = [
 ('Variance Reduction', x_var, y_var),
 ('Space-Filling', x_space, y_space),
 ('Hybrid', x_hybrid, y_hybrid)
 ]

 for ax, (name, x_new, y_new) in zip(axes, strategies):
 ax.scatter(x_init, y_init, c='blue', s=100, marker='o',
 label='Existing', edgecolor='k', linewidth=1.5)
 # Proposed new samples
 ax.scatter(x_new, y_new, c='red', s=150, marker='*',
 label='Proposed', edgecolor='k', linewidth=1.5)
 ax.set_xlabel('X (m)')
 ax.set_ylabel('Y (m)')
 ax.set_title(f'{name} Strategy')
 ax.set_xlim(0, 100)
 ax.set_ylim(0, 100)
 ax.legend()
 ax.set_aspect('equal')

 plt.tight_layout()
 plt.savefig('example_workflow_02_optimal_sampling.png', dpi=150, bbox_inches='tight')
 logger.info(" Saved example_workflow_02_optimal_sampling.png")
 plt.close()

def example_2_infill_sampling():
 
 logger.info("Example 2: Infill Sampling")
 

 # Initial samples
 np.random.seed(42)
 n_initial = 15
 x_init = np.random.uniform(10, 90, n_initial)
 y_init = np.random.uniform(10, 90, n_initial)
 z_init = 50 + 0.3*x_init + np.random.normal(0, 3, n_initial)

 # Fit variogram
 lags, gamma = experimental_variogram(x_init, y_init, z_init)
 variogram_model = fit_variogram(lags, gamma, model_type='exponential')

 # Find infill locations (variance < 2.0)
 logger.info("Finding infill locations to reduce variance below 2.0...")
 x_infill, y_infill = infill_sampling(
 x_init, y_init, z_init,
 variogram_model=variogram_model,
 variance_threshold=2.0,
 max_samples=20
 )

 logger.info(f" Need {len(x_infill)} additional samples")

 # Visualize
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 # Before infill - show kriging variance
 nx, ny = 50, 50
 x_grid = np.linspace(0, 100, nx)
 y_grid = np.linspace(0, 100, ny)
 x_pred_2d, y_pred_2d = np.meshgrid(x_grid, y_grid)

 krig_before = OrdinaryKriging(x_init, y_init, z_init, variogram_model)
 _, var_before = krig_before.predict(x_pred_2d.ravel(), y_pred_2d.ravel(), return_variance=True)
 var_before_grid = var_before.reshape((ny, nx))

 im1 = ax1.contourf(x_grid, y_grid, var_before_grid, levels=15, cmap='YlOrRd')
 ax1.scatter(x_init, y_init, c='blue', s=100, marker='o', label='Initial', edgecolor='k')
 # Remove top and right spines
 ax1.spines['right'].set_visible(False)
 # Remove top and right spines
 ax1.scatter(x_init, y_init, c
 c.spines['right'].set_visible(False)
 ax1.set_xlabel('X (m)')
 ax1.set_ylabel('Y (m)')
 ax1.set_title(f'Kriging Variance - Before Infill\nMax var: {var_before.max():.2f}')
 ax1.legend()
 ax1.set_aspect('equal')
 plt.colorbar(im1, ax=ax1, label='Variance')
 # Remove top and right spines
 ax1.set_aspect('equal')

 # After infill
 x_all = np.concatenate([x_init, x_infill])
 y_all = np.concatenate([y_init, y_infill])
 # Simulate values at infill locations
 z_infill_sim = 50 + 0.3*x_infill + np.random.normal(0, 3, len(x_infill))
 z_all = np.concatenate([z_init, z_infill_sim])

 krig_after = OrdinaryKriging(x_all, y_all, z_all, variogram_model)
 _, var_after = krig_after.predict(x_pred_2d.ravel(), y_pred_2d.ravel(), return_variance=True)
 var_after_grid = var_after.reshape((ny, nx))

 im2 = ax2.contourf(x_grid, y_grid, var_after_grid, levels=15, cmap='YlOrRd')
 ax2.scatter(x_init, y_init, c='blue', s=100, marker='o', label='Initial', edgecolor='k')
 # Remove top and right spines
 ax2.spines['right'].set_visible(False)
 # Remove top and right spines
 ax2.scatter(x_init, y_init, c
 c.spines['right'].set_visible(False)
 ax2.scatter(x_infill, y_infill, c='red', s=150, marker='*', label='Infill', edgecolor='k')
 # Remove top and right spines
 ax2.scatter(x_infill, y_infill, c
 c.spines['right'].set_visible(False)
 ax2.set_xlabel('X (m)')
 ax2.set_ylabel('Y (m)')
 ax2.set_title(f'Kriging Variance - After Infill\nMax var: {var_after.max():.2f}')
 ax2.legend()
 ax2.set_aspect('equal')
 plt.colorbar(im2, ax=ax2, label='Variance')
 # Remove top and right spines
 ax2.set_aspect('equal')

 plt.tight_layout()
 plt.savefig('example_workflow_02_infill.png', dpi=150, bbox_inches='tight')
 logger.info(" Saved example_workflow_02_infill.png")
 plt.close()

def example_3_sample_size_calculator():
 
 logger.info("Example 3: Sample Size Calculator")
 

 # Initial samples
 np.random.seed(42)
 n_initial = 30
 x = np.random.uniform(0, 100, n_initial)
 y = np.random.uniform(0, 100, n_initial)
 z = 50 + 0.3*x + 0.2*y + np.random.normal(0, 3, n_initial)

 # Fit variogram
 lags, gamma = experimental_variogram(x, y, z)
 variogram_model = fit_variogram(lags, gamma, model_type='spherical')

 # Calculate required samples for target RMSE = 1.5
 logger.info("Calculating required samples for target RMSE = 1.5...")
 results = sample_size_calculator(
 x, y, z,
 variogram_model=variogram_model,
 target_rmse=1.5,
 max_samples=200,
 n_simulations=10
 )

 logger.info(f"Results:")
 logger.info(f" Current samples: {n_initial}")
 logger.info(f" Current RMSE: {results['current_rmse']:.3f}")
 logger.info(f" Target RMSE: {results['target_rmse']:.3f}")
 logger.info(f" Required samples: {results['required_samples']}")
 logger.info(f" Additional needed: {results['required_samples'] - n_initial}")
 logger.info(f" Achievable: {'Yes' if results['achievable'] else 'No'}")

 # Visualize RMSE vs sample size
 fig, ax = plt.subplots(figsize=(10, 6))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 ax.plot(results['sample_sizes'], results['rmse_values'], 'bo-', linewidth=2, markersize=8)
 ax.axhline(results['target_rmse'], color='red', linestyle='--', linewidth=2, label='Target RMSE')
 ax.axvline(results['required_samples'], color='green', linestyle='--', linewidth=2,
 label=f'Required: {results["required_samples"]} samples')

 # Confidence bands
 ci_upper = results['rmse_values'] + results['confidence_90']
 ci_lower = results['rmse_values'] - results['confidence_90']
 ax.fill_between(results['sample_sizes'], ci_lower, ci_upper, alpha=0.3, color='blue')

 ax.set_xlabel('Number of Samples', fontsize=12)
 ax.set_ylabel('RMSE', fontsize=12)
 ax.set_title('RMSE vs Sample Size (with 90% confidence bands)', fontsize=14)
 ax.legend(fontsize=10)

 plt.tight_layout()
 plt.savefig('example_workflow_02_sample_size.png', dpi=150, bbox_inches='tight')
 logger.info(" Saved example_workflow_02_sample_size.png")
 plt.close()

def example_4_cost_benefit():
 logger.info("Example 4: Cost-Benefit Analysis")

 # Initial samples
 np.random.seed(42)
 n_initial = 25
 x = np.random.uniform(0, 100, n_initial)
 y = np.random.uniform(0, 100, n_initial)
 z = 50 + 0.3*x + 0.2*y + np.random.normal(0, 3, n_initial)

 # Fit variogram
 lags, gamma = experimental_variogram(x, y, z)
 variogram_model = fit_variogram(lags, gamma, model_type='spherical')

 # Cost-benefit analysis
 logger.info("Performing cost-benefit analysis...")
 logger.info(" Cost per sample: $500")
 logger.info(" Benefit per RMSE reduction: $2000")
 logger.info(" Maximum budget: $20,000")

 results = cost_benefit_analysis(
 x, y, z,
 variogram_model=variogram_model,
 cost_per_sample=500,
 benefit_per_rmse_reduction=2000,
 max_budget=20000
 )

 logger.info(f"Optimal Strategy:")
 logger.info(f" Current samples: {n_initial}")
 logger.info(f" Optimal total samples: {results['optimal_n_samples']}")
 logger.info(f" Additional samples needed: {results['optimal_n_additional']}")
 logger.info(f" Total cost: ${results['optimal_total_cost']:,.2f}")
 logger.info(f" Net benefit: ${results['optimal_net_benefit']:,.2f}")
 logger.info(f" Baseline RMSE: {results['baseline_rmse']:.3f}")
 logger.info(f" Optimal RMSE: {results['optimal_rmse']:.3f}")

 # Visualize
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 # Costs and benefits
 ax1.plot(results['sample_sizes'], results['costs'], 'r-', linewidth=2, label='Cost')
 # Remove top and right spines
 ax1.plot(results['sample_sizes'], results['costs'], 'r-', linewidth
 linewidth.spines['right'].set_visible(False)
 ax1.plot(results['sample_sizes'], results['benefits'], 'g-', linewidth=2, label='Benefit')
 # Remove top and right spines
 ax1.plot(results['sample_sizes'], results['benefits'], 'g-', linewidth
 linewidth.spines['right'].set_visible(False)
 ax1.axvline(results['optimal_n_samples'], color='blue', linestyle='--',
 # Remove top and right spines
 ax1.axvline(results['optimal_n_samples'], color
 color.spines['right'].set_visible(False)
 linewidth=2, label='Optimal')
 ax1.set_xlabel('Total Number of Samples')
 ax1.set_ylabel('Cost / Benefit ($)')
 ax1.set_title('Cost-Benefit Analysis')
 ax1.legend()

 # Net benefit
 ax2.plot(results['sample_sizes'], results['net_benefits'], 'b-', linewidth=2)
 # Remove top and right spines
 ax2.plot(results['sample_sizes'], results['net_benefits'], 'b-', linewidth
 linewidth.spines['right'].set_visible(False)
 ax2.axvline(results['optimal_n_samples'], color='green', linestyle='--',
 # Remove top and right spines
 ax2.axvline(results['optimal_n_samples'], color
 color.spines['right'].set_visible(False)
 linewidth=2, label=f'Optimal: {results["optimal_n_samples"]} samples')
 ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
 # Remove top and right spines
 ax2.axhline(0, color
 color.spines['right'].set_visible(False)
 ax2.fill_between(results['sample_sizes'], 0, results['net_benefits'],
 where=(results['net_benefits'] > 0), alpha=0.3, color='green')
 ax2.set_xlabel('Total Number of Samples')
 ax2.set_ylabel('Net Benefit ($)')
 ax2.set_title('Net Benefit vs Sample Size')
 ax2.legend()

 plt.tight_layout()
 plt.savefig('example_workflow_02_cost_benefit.png', dpi=150, bbox_inches='tight')
 logger.info(" Saved example_workflow_02_cost_benefit.png")
 plt.close()

def main():
 logger.info("GEOSTATS OPTIMIZATION WORKFLOW EXAMPLES")

 example_1_optimal_sampling()
 example_2_infill_sampling()
 example_3_sample_size_calculator()
 example_4_cost_benefit()

 logger.info("ALL EXAMPLES COMPLETE!")
 logger.info("\nFiles created:")
 logger.info(" - example_workflow_02_optimal_sampling.png")
 logger.info(" - example_workflow_02_infill.png")
 logger.info(" - example_workflow_02_sample_size.png")
 logger.info(" - example_workflow_02_cost_benefit.png")

if __name__ == '__main__':
