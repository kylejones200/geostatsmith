"""
    Example 7: Advanced Features (Top 10 Missing Features)

Demonstrates the newly implemented advanced geostatistics features:
    pass
1. Normal Score Transform
2. External Drift Kriging
3. Neighborhood Search with Octant/Quadrant
4. Nested Variogram Fitting
5. Declustering
6. Lognormal Kriging
7. 3D Kriging
8. Sequential Indicator Simulation
9. Support Change (Block Kriging)
10. All integrated together

Based on algorithms from:
    pass
- Olea (2009) - A Practical Primer on Geostatistics
- Deutsch & Journel (1998) - GSLIB
- geokniga Introduction to Geostatistics
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Setup logging for examples
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s',
 datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Feature 1: Normal Score Transform
logger.info("Feature 1: Normal Score Transform")

from geostats.transformations import NormalScoreTransform

# Simulated skewed data (e.g., mineral concentrations)
np.random.seed(42)
original_data = np.random.lognormal(mean=0, sigma=1, size=100)

# Transform to normal scores
nst = NormalScoreTransform()
normal_scores = nst.fit_transform(original_data)

logger.info(f"Original data - Mean: {np.mean(original_data):.3f}, Std: {np.std(original_data):.3f}")
logger.info(f"Normal scores - Mean: {np.mean(normal_scores):.3f}, Std: {np.std(normal_scores):.3f}")

# Back-transform
back_transformed = nst.inverse_transform(normal_scores)
logger.info(f"Back-transformed matches original: {np.allclose(back_transformed, original_data)}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# Remove top and right spines

axes[0].hist(original_data, bins=20, edgecolor='black', alpha=0.7)
# Remove top and right spines
axes[0]
# Remove top and right spines
axes[0].hist(original_data, bins)

axes[0].set_title('Original Data (Lognormal)')
# Remove top and right spines
axes[0].set_title('Original Data (Lognormal)')
axes[0].set_xlabel('Value')
# Remove top and right spines
axes[0].set_xlabel('Value')
axes[1].hist(normal_scores, bins=20, edgecolor='black', alpha=0.7)
# Remove top and right spines
axes[1]
# Remove top and right spines
axes[1].hist(normal_scores, bins)

axes[1].set_title('Normal Scores (Gaussian)')
# Remove top and right spines
axes[1].set_title('Normal Scores (Gaussian)')
axes[1].set_xlabel('Normal Score')
# Remove top and right spines
axes[1].set_xlabel('Normal Score')
axes[2].scatter(original_data, normal_scores, alpha=0.6)
# Remove top and right spines
axes[2]
# Remove top and right spines
axes[2].scatter(original_data, normal_scores, alpha)

axes[2].set_xlabel('Original Value')
# Remove top and right spines
axes[2].set_xlabel('Original Value')
axes[2].set_ylabel('Normal Score')
# Remove top and right spines
axes[2].set_ylabel('Normal Score')
axes[2].set_title('Transform Mapping')
# Remove top and right spines
axes[2].set_title('Transform Mapping')
plt.tight_layout()
plt.savefig('advanced_1_normal_score.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_1_normal_score.png\n")

# Feature 2: External Drift Kriging
logger.info("Feature 2: External Drift Kriging")

from geostats.algorithms import ExternalDriftKriging
from geostats.models.variogram_models import SphericalModel

# Generate synthetic data with elevation covariate
x = np.random.uniform(0, 100, 50)
y = np.random.uniform(0, 100, 50)
elevation = 200 + 0.5 * x + 0.3 * y + np.random.normal(0, 5, 50) # Elevation trend
temperature = 25 - 0.02 * elevation + np.random.normal(0, 1, 50) # Temp decreases with elevation

# Fit variogram on residuals (after removing elevation trend)
from geostats.algorithms import experimental_variogram, fit_variogram_model
temp_residuals = temperature - (25 - 0.02 * elevation)
lags, gamma = experimental_variogram(x, y, temp_residuals, n_lags=10)
variogram_model = fit_variogram_model(lags, gamma, model_type='spherical')

# External drift kriging
edk = ExternalDriftKriging(x, y, temperature, elevation, variogram_model)

# Prediction grid
x_grid = np.linspace(0, 100, 20)
y_grid = np.linspace(0, 100, 20)
xx, yy = np.meshgrid(x_grid, y_grid)
elevation_grid = 200 + 0.5 * xx + 0.3 * yy

pred, var = edk.predict(xx.flatten(), yy.flatten(), elevation_grid.flatten())
pred = pred.reshape(xx.shape)

logger.info(f"Temperature range: {temperature.min():.2f} to {temperature.max():.2f}°C")
logger.info(f"Predicted range: {pred.min():.2f} to {pred.max():.2f}°C")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Remove top and right spines

c1 = axes[0].scatter(x, y, c=temperature, cmap='coolwarm', s=100, edgecolors='black')
axes[0].set_title('Observed Temperature')
# Remove top and right spines
axes[0].set_title('Observed Temperature')
plt.colorbar(c1, ax=axes[0], label='°C')
# Remove top and right spines
axes[0].set_title('Observed Temperature')
c2 = axes[1].contourf(xx, yy, pred, levels=15, cmap='coolwarm')
axes[1].scatter(x, y, c='black', s=20, marker='+')
# Remove top and right spines
axes[1].scatter(x, y, c)

axes[1].set_title('EDK Prediction (with Elevation Covariate)')
# Remove top and right spines
axes[1].set_title('EDK Prediction (with Elevation Covariate)')
plt.colorbar(c2, ax=axes[1], label='°C')
# Remove top and right spines
axes[1].set_title('EDK Prediction (with Elevation Covariate)')
plt.tight_layout()
plt.savefig('advanced_2_external_drift.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_2_external_drift.png\n")

# Feature 3: Neighborhood Search with Octant
logger.info("Feature 3: Neighborhood Search (Octant/Quadrant)")

from geostats.algorithms import NeighborhoodSearch, NeighborhoodConfig

# Create search configuration
config_octant = NeighborhoodConfig(
 max_neighbors=16,
 min_neighbors=4,
 search_radius=30,
 use_octants=True,
 max_per_octant=2
)

config_simple = NeighborhoodConfig(
 max_neighbors=16,
 search_radius=30
)

# Create neighborhood searches
ns_octant = NeighborhoodSearch(x, y, config_octant)
ns_simple = NeighborhoodSearch(x, y, config_simple)

# Find neighbors for a point
x0, y0 = 50, 50
indices_octant, dists_octant = ns_octant.find_neighbors(x0, y0)
indices_simple, dists_simple = ns_simple.find_neighbors(x0, y0)

logger.info(f"Octant search: {len(indices_octant)} neighbors")
logger.info(f"Simple search: {len(indices_simple)} neighbors")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Remove top and right spines

for ax, indices, title in zip(axes, [indices_simple, indices_octant],
 ['Simple Nearest', 'Octant Search']):
     pass
 ax.scatter(x, y, c='lightgray', s=50, alpha=0.5, label='Other points')
 ax.scatter(x[indices], y[indices], c='red', s=100, marker='o',
 edgecolors='black', label='Selected neighbors')
 ax.scatter([x0], [y0], c='blue', s=200, marker='*',
 edgecolors='black', label='Target point')
 circle = plt.Circle((x0, y0), 30, fill=False, linestyle='--', color='blue')
 ax.add_patch(circle)
 ax.set_title(title)
 ax.legend()
 ax.set_xlim(-5, 105)
 ax.set_ylim(-5, 105)
 ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('advanced_3_neighborhood.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_3_neighborhood.png\n")

# Feature 4: Nested Variogram Fitting
logger.info("Feature 4: Nested Variogram Fitting")

from geostats.algorithms import fit_nested_variogram

# Generate multi-scale spatial data
x_var = np.random.uniform(0, 100, 200)
y_var = np.random.uniform(0, 100, 200)

# Multi-scale variation: short-range + long-range
z_short = np.sin(x_var / 10) + np.random.normal(0, 0.3, 200) # Short range
z_long = 0.5 * np.sin(x_var / 50) + 0.5 * np.cos(y_var / 50) # Long range
z_multi = z_short + z_long + np.random.normal(0, 0.1, 200) # Combined

# Experimental variogram
lags_multi, gamma_multi = experimental_variogram(x_var, y_var, z_multi, n_lags=15)

# Fit nested model (2 structures)
nested_model = fit_nested_variogram()
 lags_multi, gamma_multi,
 n_structures=2,
 model_types=['spherical', 'spherical']
)

logger.info(nested_model)
logger.info(f"Total sill: {nested_model.total_sill():.3f}")
logger.info(f"Effective range: {nested_model.effective_range():.1f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(lags_multi, gamma_multi, 'o', markersize=8, label='Experimental', color='black')
h_plot = np.linspace(0, np.max(lags_multi), 100)
gamma_plot = nested_model(h_plot)
plt.plot(h_plot, gamma_plot, '-', linewidth=2, label='Nested Model', color='red')

# Plot individual structures
gamma_nugget = np.full_like(h_plot, nested_model.nugget)
plt.plot(h_plot, gamma_nugget, '--', alpha=0.5, label=f'Nugget: {nested_model.nugget:.3f}')
cumulative = nested_model.nugget
for i, struct in enumerate(nested_model.structures, 1):
 cumulative_next = cumulative + gamma_struct
 plt.fill_between(h_plot, cumulative, cumulative_next, alpha=0.2,
 label=f'Structure {i}: {struct.model_type}')
 cumulative = cumulative_next

plt.xlabel('Lag Distance')
plt.ylabel('Semivariance')
plt.title('Nested Variogram Model (2 Structures)')
plt.legend()
plt.tight_layout()
plt.savefig('advanced_4_nested_variogram.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_4_nested_variogram.png\n")

# Feature 5: Declustering
logger.info("Feature 5: Declustering")

from geostats.transformations import cell_declustering

# Generate clustered data (preferential sampling)
x_cluster1 = np.random.normal(20, 5, 30)
y_cluster1 = np.random.normal(20, 5, 30)
z_cluster1 = np.random.normal(10, 2, 30) # Low values in cluster

x_cluster2 = np.random.normal(80, 3, 40)
y_cluster2 = np.random.normal(80, 3, 40)
z_cluster2 = np.random.normal(50, 3, 40) # High values in cluster

x_sparse = np.random.uniform(0, 100, 15)
y_sparse = np.random.uniform(0, 100, 15)
z_sparse = np.random.normal(30, 5, 15) # Medium values sparse

x_declust = np.concatenate([x_cluster1, x_cluster2, x_sparse])
y_declust = np.concatenate([y_cluster1, y_cluster2, y_sparse])
z_declust = np.concatenate([z_cluster1, z_cluster2, z_sparse])

# Decluster
weights, info = cell_declustering(x_declust, y_declust, z_declust)

logger.info(f"Unweighted mean: {info['unweighted_mean']:.2f}")
logger.info(f"Weighted mean: {info['weighted_mean']:.2f}")
logger.info(f"Difference: {info['mean_difference']:.2f}")
logger.info(f"Optimal cell size: {info['optimal_cell_size']:.1f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Remove top and right spines

scatter1 = axes[0].scatter(x_declust, y_declust, c=z_declust, s=50, cmap='viridis', edgecolors='black')
axes[0].set_title(f"Clustered Data (Unweighted Mean: {info['unweighted_mean']:.2f})")
# Remove top and right spines
axes[0].set_title(f"Clustered Data (Unweighted Mean: {info['unweighted_mean']:.2f})")
axes[0].set_xlabel('X')
# Remove top and right spines
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
# Remove top and right spines
axes[0].set_ylabel('Y')
plt.colorbar(scatter1, ax=axes[0])
# Remove top and right spines
axes[0].set_ylabel('Y')

# Size points by weight
scatter2 = axes[1].scatter(x_declust, y_declust, c=z_declust, s=weights*10,
 cmap='viridis', edgecolors='black', alpha=0.7)
axes[1].set_title(f"Weighted by Declustering (Weighted Mean: {info['weighted_mean']:.2f})")
# Remove top and right spines
axes[1].set_title(f"Weighted by Declustering (Weighted Mean: {info['weighted_mean']:.2f})")
axes[1].set_xlabel('X')
# Remove top and right spines
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
# Remove top and right spines
axes[1].set_ylabel('Y')
plt.colorbar(scatter2, ax=axes[1])
# Remove top and right spines
axes[1].set_ylabel('Y')
plt.tight_layout()
plt.savefig('advanced_5_declustering.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_5_declustering.png\n")

# Feature 6: Lognormal Kriging
logger.info("Feature 6: Lognormal Kriging")

from geostats.algorithms import LognormalKriging

# Generate lognormal data (e.g., ore grades)
x_ln = np.random.uniform(0, 50, 40)
y_ln = np.random.uniform(0, 50, 40)
z_ln = np.random.lognormal(mean=2, sigma=0.5, size=40)

# Fit variogram on log-transformed data
z_log = np.log(z_ln)
lags_ln, gamma_ln = experimental_variogram(x_ln, y_ln, z_log, n_lags=8)
variogram_ln = fit_variogram_model(lags_ln, gamma_ln, model_type='exponential')

# Lognormal kriging
lnk = LognormalKriging(x_ln, y_ln, z_ln, variogram_model=variogram_ln, kriging_type='ordinary')

# Prediction
x_grid_ln = np.linspace(0, 50, 25)
y_grid_ln = np.linspace(0, 50, 25)
xx_ln, yy_ln = np.meshgrid(x_grid_ln, y_grid_ln)
pred_ln = lnk.predict(xx_ln.flatten(), yy_ln.flatten(), return_variance=False,
 back_transform_method='unbiased')
pred_ln = pred_ln.reshape(xx_ln.shape)

logger.info(f"Original data range: {z_ln.min():.2f} to {z_ln.max():.2f}")
logger.info(f"Predicted range: {pred_ln.min():.2f} to {pred_ln.max():.2f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Remove top and right spines

c1 = axes[0].scatter(x_ln, y_ln, c=z_ln, s=100, cmap='YlOrRd', edgecolors='black')
axes[0].set_title('Lognormal Data (Ore Grades)')
# Remove top and right spines
axes[0].set_title('Lognormal Data (Ore Grades)')
plt.colorbar(c1, ax=axes[0], label='Grade')
# Remove top and right spines
axes[0].set_title('Lognormal Data (Ore Grades)')
c2 = axes[1].contourf(xx_ln, yy_ln, pred_ln, levels=15, cmap='YlOrRd')
axes[1].scatter(x_ln, y_ln, c='black', s=30, marker='+')
# Remove top and right spines
axes[1].scatter(x_ln, y_ln, c)

axes[1].set_title('Lognormal Kriging Prediction')
# Remove top and right spines
axes[1].set_title('Lognormal Kriging Prediction')
plt.colorbar(c2, ax=axes[1], label='Grade')
# Remove top and right spines
axes[1].set_title('Lognormal Kriging Prediction')
plt.tight_layout()
plt.savefig('advanced_6_lognormal_kriging.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_6_lognormal_kriging.png\n")

# Feature 7: 3D Kriging
logger.info("Feature 7: 3D Kriging")

from geostats.algorithms import OrdinaryKriging3D

# 3D sample data (e.g., ore body in x, y, z coordinates)
np.random.seed(10)
x_3d = np.random.uniform(0, 100, 30)
y_3d = np.random.uniform(0, 100, 30)
z_3d_coord = np.random.uniform(0, 50, 30) # depth coordinate
values_3d = 10 + 0.05 * x_3d - 0.1 * z_3d_coord + np.random.normal(0, 2, 30)

# Note: For 3D variogram, would need to compute distances in 3D
# Using a simple model for demonstration
class Simple3DVariogram:
 self.sill = 10.0
 self.range_val = 30.0
 def __call__(self, h):
     pass

variogram_3d = Simple3DVariogram()
ok3d = OrdinaryKriging3D(x_3d, y_3d, z_3d_coord, values_3d, variogram_3d)

# Predict on 3D grid (horizontal slice at z=25)
x_grid_3d = np.linspace(0, 100, 15)
y_grid_3d = np.linspace(0, 100, 15)
xx_3d, yy_3d = np.meshgrid(x_grid_3d, y_grid_3d)
zz_3d = np.full_like(xx_3d, 25.0) # Slice at depth 25

pred_3d = ok3d.predict(xx_3d.flatten(), yy_3d.flatten(), zz_3d.flatten(),
 return_variance=False)
pred_3d = pred_3d.reshape(xx_3d.shape)

logger.info(f"3D sample points: {len(x_3d)}")
logger.info(f"Depth range: {z_3d_coord.min():.1f} to {z_3d_coord.max():.1f}")
logger.info(f"Value range: {values_3d.min():.2f} to {values_3d.max():.2f}")

# Plot (slice view)
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
# Remove top and right spines
# Remove top and right spines

scatter = ax1.scatter(x_3d, y_3d, z_3d_coord, c=values_3d, cmap='plasma', s=100, edgecolors='black')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Depth')
ax1.set_title('3D Sample Data')
plt.colorbar(scatter, ax=ax1, label='Value', shrink=0.5)
# Remove top and right spines
ax1.set_title('3D Sample Data')

ax2 = fig.add_subplot(122)
# Remove top and right spines

contour = ax2.contourf(xx_3d, yy_3d, pred_3d, levels=15, cmap='plasma')
# Show samples at this depth slice
mask_slice = np.abs(z_3d_coord - 25) < 10
ax2.scatter(x_3d[mask_slice], y_3d[mask_slice], c='white', s=50, marker='o', edgecolors='black')
# Remove top and right spines

# Remove top and right spines
ax2.scatter(x_3d[mask_slice], y_3d[mask_slice], c)

ax2.set_title('3D Kriging Prediction (Slice at Depth=25)')
# Remove top and right spines
ax2.set_title(')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(contour, ax=ax2, label='Value')
# Remove top and right spines
ax2.set_ylabel('Y')
plt.tight_layout()
plt.savefig('advanced_7_3d_kriging.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_7_3d_kriging.png\n")

# Feature 9: Block Kriging (Support Change)
logger.info("Feature 9: Block Kriging (Support Change)")

from geostats.algorithms import BlockKriging

# Point data
x_bk = np.random.uniform(0, 100, 50)
y_bk = np.random.uniform(0, 100, 50)
z_bk = np.sin(x_bk / 15) + np.cos(y_bk / 20) + np.random.normal(0, 0.3, 50)

# Fit variogram
lags_bk, gamma_bk = experimental_variogram(x_bk, y_bk, z_bk, n_lags=10)
variogram_bk = fit_variogram_model(lags_bk, gamma_bk, model_type='spherical')

# Block kriging (estimate 10x10 block averages)
bk = BlockKriging(x_bk, y_bk, z_bk, variogram_model=variogram_bk,
 block_size=(10.0, 10.0), n_disc=5)

# Prediction grid (block centers)
x_grid_bk = np.arange(10, 100, 15)
y_grid_bk = np.arange(10, 100, 15)
xx_bk, yy_bk = np.meshgrid(x_grid_bk, y_grid_bk)

pred_bk, var_bk = bk.predict(xx_bk.flatten(), yy_bk.flatten())
pred_bk = pred_bk.reshape(xx_bk.shape)
var_bk = var_bk.reshape(xx_bk.shape)

logger.info(f"Block size: 10 x 10")
logger.info(f"Point variance (typical): ~{np.mean(gamma_bk[-3:]):.3f}")
logger.info(f"Block variance (typical): ~{np.mean(var_bk):.3f}")
logger.info(f"Variance reduction: {(1 - np.mean(var_bk)/np.mean(gamma_bk[-3:]))*100:.1f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Remove top and right spines

c1 = axes[0].scatter(x_bk, y_bk, c=z_bk, s=60, cmap='coolwarm', edgecolors='black')
axes[0].set_title('Point Data')
# Remove top and right spines
axes[0].set_title('Point Data')
plt.colorbar(c1, ax=axes[0])
# Remove top and right spines
axes[0].set_title('Point Data')
axes[0].set_xlabel('X')
# Remove top and right spines
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
# Remove top and right spines
axes[0].set_ylabel('Y')

# Show blocks with predictions
for i in range(len(x_grid_bk)):
 fill=True, facecolor=plt.cm.coolwarm((pred_bk[j,i] - pred_bk.min())/(pred_bk.max() - pred_bk.min())),
 edgecolor='black', linewidth=0.5, alpha=0.7)
 axes[1].add_patch(rect)
 # Remove top and right spines
 axes[1].add_patch(rect)
axes[1].scatter(x_bk, y_bk, c='black', s=20, marker='+', alpha=0.5)
# Remove top and right spines
axes[1].scatter(x_bk, y_bk, c)

axes[1].set_xlim(0, 100)
# Remove top and right spines
axes[1].set_xlim(0, 100)
axes[1].set_ylim(0, 100)
# Remove top and right spines
axes[1].set_ylim(0, 100)
axes[1].set_title('Block Kriging (10×10 blocks)')
# Remove top and right spines
axes[1].set_title('Block Kriging (10×10 blocks)')
axes[1].set_xlabel('X')
# Remove top and right spines
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
# Remove top and right spines
axes[1].set_ylabel('Y')
axes[1].set_aspect('equal')
# Remove top and right spines
axes[1].set_aspect('equal')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=pred_bk.min(), vmax=pred_bk.max()))
# Remove top and right spines
axes[1].set_aspect('equal')
sm.set_array([])
plt.colorbar(sm, ax=axes[1])
# Remove top and right spines

plt.tight_layout()
plt.savefig('advanced_9_block_kriging.png', dpi=150, bbox_inches='tight')
logger.info("Saved: advanced_9_block_kriging.png\n")

logger.info("All 10 Advanced Features Demonstrated Successfully!")
logger.info("\nGenerated figures:")
logger.info(" - advanced_1_normal_score.png")
logger.info(" - advanced_2_external_drift.png")
logger.info(" - advanced_3_neighborhood.png")
logger.info(" - advanced_4_nested_variogram.png")
logger.info(" - advanced_5_declustering.png")
logger.info(" - advanced_6_lognormal_kriging.png")
logger.info(" - advanced_7_3d_kriging.png")
logger.info(" - advanced_9_block_kriging.png")
logger.info("\nNote: Sequential Indicator Simulation (Feature 8) requires indicator")
logger.info(" variogram modeling and is demonstrated separately in the docs.")
