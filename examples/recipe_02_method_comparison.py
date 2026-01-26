"""
Recipe 2: Method Comparison for Spatial Data

This recipe demonstrates how to compare different interpolation methods
systematically using cross-validation and performance metrics.

Inspired by: Python Recipes for Earth Sciences (Trauth 2024), Sections 7.6-7.7

Key Concepts:
- Cross-validation for method selection
- Speed vs accuracy tradeoffs
- Visualization of interpolation quality
"""

import numpy as np
import matplotlib.pyplot as plt

from geostats.datasets import generate_elevation_like_data
from geostats.comparison import (
    compare_interpolation_methods,
    cross_validate_interpolation,
)

print("=" * 70)
print("RECIPE 2: SYSTEMATIC METHOD COMPARISON")
print("=" * 70)

# Generate test data with known properties
print("\nGenerating synthetic terrain data...")
np.random.seed(42)
x, y, z = generate_elevation_like_data(
    n_points=120,
    n_hills=4,
    roughness=0.15,
    seed=42
)

print(f"  Created {len(x)} sample points")
print(f"  Elevation range: {z.min():.1f} to {z.max():.1f} m")

# Create prediction grid
x_min, x_max = 0, 100
y_min, y_max = 0, 100
grid_res = 40

x_pred = np.linspace(x_min, x_max, grid_res)
y_pred = np.linspace(y_min, y_max, grid_res)
X_grid, Y_grid = np.meshgrid(x_pred, y_pred)

# Compare all methods
print("\nComparing interpolation methods...")
methods = ['ordinary_kriging', 'simple_kriging', 'idw', 'rbf', 'natural_neighbor']

results = compare_interpolation_methods(
    x, y, z,
    X_grid.flatten(), Y_grid.flatten(),
    methods=methods,
    cross_validate=True,
    benchmark_speed=True,
    plot=False
)

# Create comprehensive comparison plot
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Method Comparison: Accuracy vs Speed', fontsize=16, fontweight='bold')

# Plot interpolation results (top row and middle row)
plot_idx = 0
for i, method in enumerate(methods):
    row = plot_idx // 3
    col = plot_idx % 3
    ax = axes[row, col]
    
    Z_pred = results['predictions'][method].reshape(X_grid.shape)
    
    # Contour plot
    levels = 15
    im = ax.contourf(X_grid, Y_grid, Z_pred, levels=levels, cmap='terrain')
    ax.scatter(x, y, c=z, s=20, edgecolors='black', linewidths=0.5, 
               cmap='terrain', vmin=Z_pred.min(), vmax=Z_pred.max())
    
    # Title with metrics
    title = method.replace('_', ' ').title()
    if method in results['cv_results']:
        metrics = results['cv_results'][method]['metrics']
        title += f"\nRMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}"
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Elevation')
    
    plot_idx += 1

# Bar chart: RMSE comparison (bottom left)
ax_rmse = axes[2, 0]
method_names = []
rmse_values = []
for method in methods:
    if method in results['cv_results']:
        method_names.append(method.replace('_', '\n'))
        rmse_values.append(results['cv_results'][method]['metrics']['rmse'])

bars = ax_rmse.bar(range(len(method_names)), rmse_values, color='steelblue', alpha=0.7)
ax_rmse.set_xlabel('Method')
ax_rmse.set_ylabel('RMSE (Cross-Validation)')
ax_rmse.set_title('Accuracy Comparison', fontweight='bold')
ax_rmse.set_xticks(range(len(method_names)))
ax_rmse.set_xticklabels(method_names, rotation=0, ha='center', fontsize=9)
ax_rmse.grid(axis='y', alpha=0.3)

# Highlight best method
best_idx = np.argmin(rmse_values)
bars[best_idx].set_color('green')
bars[best_idx].set_alpha(0.9)

# Bar chart: Speed comparison (bottom middle)
ax_speed = axes[2, 1]
speed_values = [results['speed_results'][m]['mean_time'] for m in methods]
bars_speed = ax_speed.bar(range(len(method_names)), speed_values, 
                          color='coral', alpha=0.7)
ax_speed.set_xlabel('Method')
ax_speed.set_ylabel('Time (seconds)')
ax_speed.set_title('Speed Comparison', fontweight='bold')
ax_speed.set_xticks(range(len(method_names)))
ax_speed.set_xticklabels(method_names, rotation=0, ha='center', fontsize=9)
ax_speed.grid(axis='y', alpha=0.3)

# Highlight fastest method
fastest_idx = np.argmin(speed_values)
bars_speed[fastest_idx].set_color('darkgreen')
bars_speed[fastest_idx].set_alpha(0.9)

# Scatter plot: Accuracy vs Speed tradeoff (bottom right)
ax_tradeoff = axes[2, 2]
colors = ['green' if i == best_idx else 'steelblue' for i in range(len(methods))]
ax_tradeoff.scatter(speed_values, rmse_values, s=200, c=colors, alpha=0.6, 
                   edgecolors='black', linewidths=2)

# Add method labels
for i, method in enumerate(methods):
    ax_tradeoff.annotate(method.replace('_', '\n'), 
                        (speed_values[i], rmse_values[i]),
                        fontsize=8, ha='center', va='center')

ax_tradeoff.set_xlabel('Computation Time (seconds)')
ax_tradeoff.set_ylabel('RMSE')
ax_tradeoff.set_title('Accuracy vs Speed Tradeoff', fontweight='bold')
ax_tradeoff.grid(True, alpha=0.3)

# Add "optimal" region annotation
ax_tradeoff.axhline(min(rmse_values) * 1.1, color='green', linestyle='--', 
                   alpha=0.5, label='Target accuracy')
ax_tradeoff.axvline(min(speed_values) * 2, color='blue', linestyle='--', 
                   alpha=0.5, label='Target speed')
ax_tradeoff.legend(fontsize=8)

plt.tight_layout()
plt.savefig('recipe_02_method_comparison.png', dpi=150, bbox_inches='tight')
print("\nFigure saved as 'recipe_02_method_comparison.png'")
plt.show()

# Print summary table
print("\n" + "=" * 70)
print("DETAILED COMPARISON TABLE")
print("=" * 70)
print(f"{'Method':<25} {'RMSE':>10} {'R²':>8} {'MAE':>10} {'Time (s)':>12}")
print("-" * 70)

for method in methods:
    if method in results['cv_results']:
        metrics = results['cv_results'][method]['metrics']
        timing = results['speed_results'][method]['mean_time']
        
        # Highlight best
        marker = " ★" if method == methods[best_idx] else "  "
        
        print(f"{method:<25} {metrics['rmse']:>10.3f} {metrics['r2']:>8.3f} "
              f"{metrics['mae']:>10.3f} {timing:>12.4f}{marker}")

print("=" * 70)

# Recommendations
print("\nRECOMMENDATIONS")
print("=" * 70)
print(f"\nMost Accurate: {methods[best_idx]}")
print(f"  RMSE: {rmse_values[best_idx]:.3f}")

print(f"\nFastest: {methods[fastest_idx]}")
print(f"  Time: {speed_values[fastest_idx]:.4f}s")

# Find best balance
# Normalize scores (lower is better)
norm_rmse = np.array(rmse_values) / max(rmse_values)
norm_time = np.array(speed_values) / max(speed_values)
combined_score = norm_rmse + norm_time  # Simple additive score
balanced_idx = np.argmin(combined_score)

print(f"\nBest Balance (Accuracy + Speed): {methods[balanced_idx]}")
print(f"  RMSE: {rmse_values[balanced_idx]:.3f}, Time: {speed_values[balanced_idx]:.4f}s")

print("\nUse Case Recommendations:")
print("  • Production/Real-time: Use fastest method")
print("  • Research/Publication: Use most accurate method")
print("  • General purpose: Use balanced method")

print("\n" + "=" * 70)
print("Recipe complete!")
print("=" * 70)
