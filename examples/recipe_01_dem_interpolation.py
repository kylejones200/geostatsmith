"""
Recipe 1: DEM Interpolation Workflow

This example demonstrates a complete workflow for interpolating digital
elevation model (DEM) data using different methods and comparing results.

Inspired by: Python Recipes for Earth Sciences (Trauth 2024), Chapter 7

Steps:
1. Load sample elevation data
2. Explore the data distribution
3. Fit variogram model
4. Compare interpolation methods
5. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import geostats modules
from geostats.datasets import load_synthetic_dem_sample
from geostats.algorithms import variogram
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.comparison import (
    compare_interpolation_methods,
    inverse_distance_weighting,
    radial_basis_function_interpolation,
)

print("=" * 70)
print("RECIPE 1: DEM INTERPOLATION WORKFLOW")
print("=" * 70)

# Step 1: Load sample DEM data
print("\nStep 1: Loading sample DEM data...")
data = load_synthetic_dem_sample()

x = data['x']
y = data['y']
z = data['z']
X_grid = data['X_grid']
Y_grid = data['Y_grid']
Z_true = data['Z_true']

print(f"  Loaded {len(x)} sample points")
print(f"  Elevation range: {z.min():.1f} to {z.max():.1f} m")
print(f"  Grid size: {X_grid.shape}")

# Step 2: Explore data distribution
print("\nStep 2: Exploring data distribution...")
print(f"  Mean elevation: {np.mean(z):.2f} m")
print(f"  Std deviation: {np.std(z):.2f} m")
print(f"  Spatial extent: X=[{x.min():.1f}, {x.max():.1f}], "
      f"Y=[{y.min():.1f}, {y.max():.1f}]")

# Step 3: Fit variogram model
print("\nStep 3: Fitting variogram model...")
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=15)

# Try different models
models = {}
for model_type in ['spherical', 'exponential', 'gaussian']:
    try:
        model = variogram.fit_model(model_type, lags, gamma, weights=n_pairs)
        models[model_type] = model
        print(f"  {model_type.capitalize()}: "
              f"sill={model.sill:.2f}, range={model.range_param:.2f}")
    except Exception as e:
        print(f"  {model_type.capitalize()}: Failed to fit")

# Use best model (spherical is often good for DEMs)
best_model = models.get('spherical', list(models.values())[0])
print(f"\n  Selected model: Spherical")

# Step 4: Compare interpolation methods
print("\nStep 4: Comparing interpolation methods...")
print("  This may take a moment...")

# Create prediction points
x_pred = X_grid.flatten()
y_pred = Y_grid.flatten()

# Run comparison
results = compare_interpolation_methods(
    x, y, z,
    x_pred, y_pred,
    methods=['ordinary_kriging', 'idw', 'rbf'],
    cross_validate=True,
    benchmark_speed=True,
    plot=False,  # We'll make custom plots
)

# Print cross-validation results
print("\n  Cross-validation results:")
for method, cv_result in results['cv_results'].items():
    metrics = cv_result['metrics']
    print(f"    {method:20s}: RMSE={metrics['rmse']:6.2f}, "
          f"R²={metrics['r2']:5.3f}, MAE={metrics['mae']:6.2f}")

# Print speed results
print("\n  Speed benchmark:")
for method, timing in results['speed_results'].items():
    print(f"    {method:20s}: {timing['mean_time']:6.3f}s "
          f"(±{timing['std_time']:.3f}s)")

# Step 5: Visualize results
print("\nStep 5: Visualizing results...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Sample points
ax1 = fig.add_subplot(gs[0, 0])
scatter = ax1.scatter(x, y, c=z, cmap='terrain', s=30, edgecolors='black', linewidths=0.5)
ax1.set_title('Sample Points (Measured)', fontsize=12, fontweight='bold')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_aspect('equal')
plt.colorbar(scatter, ax=ax1, label='Elevation (m)')

# Plot 2: True DEM (ground truth)
ax2 = fig.add_subplot(gs[0, 1])
im = ax2.contourf(X_grid, Y_grid, Z_true, levels=15, cmap='terrain')
ax2.scatter(x, y, c='red', s=10, alpha=0.5, marker='x')
ax2.set_title('True DEM (Ground Truth)', fontsize=12, fontweight='bold')
ax1.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_aspect('equal')
plt.colorbar(im, ax=ax2, label='Elevation (m)')

# Plot 3: Experimental variogram
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(lags, gamma, 'o-', label='Experimental', linewidth=2, markersize=8)
h_model = np.linspace(0, np.max(lags), 100)
gamma_model = best_model(h_model)
ax3.plot(h_model, gamma_model, 'r--', label='Fitted Model', linewidth=2)
ax3.set_title('Variogram Analysis', fontsize=12, fontweight='bold')
ax3.set_xlabel('Distance (h)')
ax3.set_ylabel('Semivariance γ(h)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4-6: Interpolation results
methods_to_plot = ['ordinary_kriging', 'idw', 'rbf']
titles = ['Ordinary Kriging', 'Inverse Distance Weighting', 'Radial Basis Function']

for idx, (method, title) in enumerate(zip(methods_to_plot, titles)):
    ax = fig.add_subplot(gs[1, idx])
    
    Z_pred = results['predictions'][method].reshape(X_grid.shape)
    im = ax.contourf(X_grid, Y_grid, Z_pred, levels=15, cmap='terrain')
    ax.scatter(x, y, c='red', s=5, alpha=0.5, marker='x')
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # Add metrics
    if method in results['cv_results']:
        metrics = results['cv_results'][method]['metrics']
        textstr = f"RMSE={metrics['rmse']:.2f}\nR²={metrics['r2']:.3f}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 7-9: Error maps
for idx, (method, title) in enumerate(zip(methods_to_plot, titles)):
    ax = fig.add_subplot(gs[2, idx])
    
    Z_pred = results['predictions'][method].reshape(X_grid.shape)
    error = Z_pred - Z_true
    
    vmax = np.max(np.abs(error))
    im = ax.contourf(X_grid, Y_grid, error, levels=15, cmap='RdBu_r', 
                     vmin=-vmax, vmax=vmax)
    ax.scatter(x, y, c='black', s=5, alpha=0.3, marker='x')
    ax.set_title(f'{title} Error', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Error (m)')
    
    # Add error statistics
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    textstr = f"MAE={mae:.2f}\nRMSE={rmse:.2f}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('DEM Interpolation Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('recipe_01_dem_interpolation.png', dpi=150, bbox_inches='tight')
print("  Figure saved as 'recipe_01_dem_interpolation.png'")
plt.show()

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nBest method for this DEM:")
best_method = min(results['cv_results'].items(), 
                  key=lambda x: x[1]['metrics']['rmse'])
print(f"  {best_method[0]} (RMSE: {best_method[1]['metrics']['rmse']:.2f})")

print("\nKey findings:")
print("  • Ordinary Kriging accounts for spatial correlation")
print("  • IDW is fastest but may create bull's-eye artifacts")
print("  • RBF produces smooth surfaces")
print("  • Error maps show spatial patterns of uncertainty")

print("\n" + "=" * 70)
print("Recipe complete! See output figure for visual comparison.")
print("=" * 70)
