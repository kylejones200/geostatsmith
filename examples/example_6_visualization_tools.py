"""
    Example 6: Visualization Tools for Geostatistics

This example demonstrates:
    pass
- h-scatterplots for exploring spatial correlation
- Variogram cloud visualization
- Directional variograms for anisotropy detection
- Variogram map (2D surface)
- Diagnostic plots

Based on Zhang, Y. (2010). Course Notes, Chapter 3
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')

from geostats.datasets import load_walker_lake
from geostats import variogram, visualization

# Load Walker Lake dataset
logger.info("Loading Walker Lake dataset...")
data = load_walker_lake()
x, y, V, U = data['x'], data['y'], data['V'], data['U']

logger.info(f"Dataset: {len(x)} samples on a 10x10 grid")
logger.info(f"V (Arsenious): [{np.min(V)}, {np.max(V)}] ppm")
logger.info(f"U (PCE): [{np.min(U)}, {np.max(U)}] ppm")

# Calculate experimental variogram
logger.info("\nCalculating experimental variogram...")
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, V, n_lags=8)
vario_model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

# Create visualization
logger.info("\nGenerating visualizations...")

# Figure 1: h-Scatterplots at different lags
fig1 = plt.figure(figsize=(16, 12))

# h-scatterplots for V at different distances
distances = [10, 20, 30, 40]
for i, h in enumerate(distances, 1):
    ax = plt.subplot(3, 4, i)
    visualization.plot_h_scatterplot(x, y, V, h, ax=ax)
    ax.set_title(f'h-scatterplot (h={h})')

# Directional h-scatterplots for V
directions = [0, 45, 90, 135]
for i, direction in enumerate(directions, 5):
    ax = plt.subplot(3, 4, i)
    visualization.plot_h_scatterplot(x, y, V, h=20,
                                      direction=direction, angle_tolerance=30, ax=ax)
    ax.set_title(f'Directional h-scatterplot (θ={direction}°)')
 
# h-scatterplot for U vs V (cross-correlation)
ax9 = plt.subplot(3, 4, 9)
# Remove top and right spines

ax9.scatter(V, U, alpha=0.7, s=80, edgecolors='black', linewidth=1)

corr = np.corrcoef(V, U)[0, 1]
ax9.set_xlabel('V (ppm)', fontsize=11)
# Remove top and right spines
ax9.set_ylabel('U (ppm)', fontsize=11)
# Remove top and right spines
ax9.set_title(f'V vs U Cross-plot (ρ={corr:.3f})', fontweight='bold', fontsize=11)
# Histograms
ax10 = plt.subplot(3, 4, 10)
# Remove top and right spines
# Remove top and right spines

visualization.plot_histogram(V, bins=15, ax=ax10, fit_normal=True)
# Remove top and right spines

ax10.set_title('V Distribution', fontweight='bold', fontsize=11)
# Remove top and right spines

# Remove top and right spines
ax10.set_title('V Distribution', fontweight)

ax11 = plt.subplot(3, 4, 11)
# Remove top and right spines
# Remove top and right spines

visualization.plot_histogram(U, bins=15, ax=ax11, fit_normal=True)
# Remove top and right spines

ax11.set_title('U Distribution', fontweight='bold', fontsize=11)
# Remove top and right spines

# Remove top and right spines
ax11.set_title('U Distribution', fontweight)

# Q-Q plot
ax12 = plt.subplot(3, 4, 12)
# Remove top and right spines
# Remove top and right spines

visualization.plot_qq_plot(V, ax=ax12)
# Remove top and right spines

plt.tight_layout()
plt.savefig('example_6_hscatterplots.png', dpi=300, bbox_inches='tight')
logger.info("Saved h-scatterplots to: example_6_hscatterplots.png")

# Figure 2: Variogram Analyses
fig2 = plt.figure(figsize=(16, 10))

# Variogram cloud
ax1 = plt.subplot(2, 3, 1)
# Remove top and right spines
# Remove top and right spines

visualization.plot_variogram_cloud(x, y, V, maxlag=50, ax=ax1)
# Remove top and right spines

# Standard variogram
ax2 = plt.subplot(2, 3, 2)
# Remove top and right spines
# Remove top and right spines

visualization.plot_variogram(lags, gamma, n_pairs, model=vario_model, ax=ax2)
# Remove top and right spines

# Directional variograms
ax3 = plt.subplot(2, 3, 3)
# Remove top and right spines
# Remove top and right spines

logger.info("Calculating directional variograms...")
dirs = [0, 45, 90, 135]
for direction in dirs:
    lags_dir, gamma_dir, n_pairs_dir = variogram.experimental_variogram(
        x, y, V, n_lags=8, direction=direction, angle_tolerance=30
    )
    valid = ~np.isnan(gamma_dir)
    ax3.plot(lags_dir[valid], gamma_dir[valid], 'o-', label=f'{direction}°', linewidth=2, markersize=6)
 
ax3.set_xlabel('Distance (h)', fontsize=11)
# Remove top and right spines
ax3.set_ylabel('γ(h)', fontsize=11)
# Remove top and right spines
ax3.set_title('Directional Variograms', fontweight='bold', fontsize=12)
# Remove top and right spines
ax3.set_title('Directional Variograms', fontweight)

ax3.legend()

# Variogram for U
ax4 = plt.subplot(2, 3, 4)
# Remove top and right spines
# Remove top and right spines

lags_u, gamma_u, n_pairs_u = variogram.experimental_variogram(x, y, U, n_lags=8)
model_u = variogram.fit_model('exponential', lags_u, gamma_u, weights=n_pairs_u)
visualization.plot_variogram(lags_u, gamma_u, n_pairs_u, model=model_u, ax=ax4)
# Remove top and right spines

ax4.set_title('Variogram for U', fontweight='bold', fontsize=12)
# Remove top and right spines

# Remove top and right spines
ax4.set_title('Variogram for U', fontweight)

# Comparison of V and U variograms
ax5 = plt.subplot(2, 3, 5)
# Remove top and right spines

ax5.plot(lags, gamma, 'o-', label='V (Arsenious)', linewidth=2, markersize=7)
# Remove top and right spines

# Remove top and right spines
ax5.plot(lags, gamma, 'o-', label)

ax5.plot(lags_u, gamma_u, 's-', label='U (PCE)', linewidth=2, markersize=7)
# Remove top and right spines
ax5.plot(lags_u, gamma_u, 's-', label)

ax5.set_xlabel('Distance (h)', fontsize=11)
# Remove top and right spines
ax5.set_ylabel('γ(h)', fontsize=11)
# Remove top and right spines
ax5.set_title('Variogram Comparison', fontweight='bold', fontsize=12)
# Remove top and right spines
ax5.set_title('Variogram Comparison', fontweight)

ax5.legend()

# Anisotropy analysis
ax6 = plt.subplot(2, 3, 6)
# Remove top and right spines
# Remove top and right spines

from geostats.models.anisotropy import DirectionalVariogram
import logging

logger = logging.getLogger(__name__)
dir_vario = DirectionalVariogram(x, y, V)
aniso_params = dir_vario.fit_anisotropy(angles=[0, 45, 90, 135], n_lags=6)

aniso_text = (
    f"Anisotropy Analysis:\n"
    f"Major direction: {aniso_params['major_angle']:.0f}°\n"
    f"Major range: {aniso_params['major_range']:.1f}\n"
    f"Minor range: {aniso_params['minor_range']:.1f}\n"
    f"Ratio: {aniso_params['ratio']:.3f}"
)
ax6.text(0.1, 0.5, aniso_text, transform=ax6.transAxes,
         fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax6.axis('off')
ax6.set_title('Anisotropy Parameters', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('example_6_variogram_analysis.png', dpi=300, bbox_inches='tight')
logger.info("Saved variogram analysis to: example_6_variogram_analysis.png")

# Figure 3: Spatial Maps
fig3, axes = plt.subplots(2, 2, figsize=(14, 12))
# Remove top and right spines
ax6.set_title('Anisotropy Parameters', fontweight)

# Data posting for V
visualization.plot_data_locations(x, y, V, ax=axes[0, 0], cmap='RdYlGn_r')
# Remove top and right spines

axes[0, 0].set_title('V - Data Locations', fontweight='bold', fontsize=12)
# Remove top and right spines
axes[0, 0]
# Remove top and right spines
axes[0, 0].set_title('V - Data Locations', fontweight)

# Data posting for U
visualization.plot_data_locations(x, y, U, ax=axes[0, 1], cmap='viridis')
# Remove top and right spines
axes[0, 0].set_title('V - Data Locations', fontweight)

axes[0, 1].set_title('U - Data Locations', fontweight='bold', fontsize=12)
# Remove top and right spines
axes[0, 1]
# Remove top and right spines
axes[0, 1].set_title('U - Data Locations', fontweight)

# Symbol map for V
visualization.plot_symbol_map(x, y, V, ax=axes[1, 0])
# Remove top and right spines
axes[0, 1].set_title('U - Data Locations', fontweight)

axes[1, 0].set_title('V - Symbol Map (size ∝ value)', fontweight='bold', fontsize=12)
# Remove top and right spines
axes[1, 0]
# Remove top and right spines
axes[1, 0].set_title('V - Symbol Map (size ∝ value)', fontweight)

# Symbol map for U
visualization.plot_symbol_map(x, y, U, ax=axes[1, 1])
# Remove top and right spines
axes[1, 0].set_title('V - Symbol Map (size ∝ value)', fontweight)

axes[1, 1].set_title('U - Symbol Map (size ∝ value)', fontweight='bold', fontsize=12)
# Remove top and right spines
axes[1, 1]
# Remove top and right spines
axes[1, 1].set_title('U - Symbol Map (size ∝ value)', fontweight)

plt.tight_layout()
plt.savefig('example_6_spatial_maps.png', dpi=300, bbox_inches='tight')
logger.info("Saved spatial maps to: example_6_spatial_maps.png")

# Display plots
plt.show()

# Print summary statistics
logger.info("SPATIAL ANALYSIS SUMMARY")
logger.info("\nVariable V (Arsenious):")
logger.info(f" Mean: {np.mean(V):.2f} ppm")
logger.info(f" Std Dev: {np.std(V):.2f} ppm")
logger.info(f" Range: [{np.min(V)}, {np.max(V)}] ppm")
logger.info(f" Variogram model: {vario_model.__class__.__name__}")
logger.info(f" Sill: {vario_model.parameters['sill']:.2f}")
logger.info(f" Range: {vario_model.parameters['range']:.2f}")
logger.info(f" Nugget: {vario_model.parameters['nugget']:.2f}")

logger.info("\nVariable U (PCE):")
logger.info(f" Mean: {np.mean(U):.2f} ppm")
logger.info(f" Std Dev: {np.std(U):.2f} ppm")
logger.info(f" Range: [{np.min(U)}, {np.max(U)}] ppm")
logger.info(f" Variogram model: {model_u.__class__.__name__}")
logger.info(f" Sill: {model_u.parameters['sill']:.2f}")
logger.info(f" Range: {model_u.parameters['range']:.2f}")

logger.info("\nCross-correlation:")
logger.info(f" Correlation(V, U): {np.corrcoef(V, U)[0, 1]:.3f}")

logger.info("\nAnisotropy (V):")
for key, value in aniso_params.items():
    pass

logger.info("\nAll visualizations demonstrate key geostatistical concepts:")
logger.info("- h-scatterplots show spatial correlation at different lags")
logger.info("- Variogram cloud reveals individual pair contributions")
logger.info("- Directional variograms detect anisotropy")
logger.info("- Symbol maps and data posting aid spatial understanding")

logger.info("\nExample completed successfully!")
