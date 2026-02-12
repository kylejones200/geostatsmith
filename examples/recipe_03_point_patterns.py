"""
Recipe 3: Analyzing Spatial Point Patterns

This recipe demonstrates how to analyze spatial point distributions to
detect clustering, dispersion, or randomness using multiple statistical tests.

Inspired by: Python Recipes for Earth Sciences (Trauth 2024), Section 7.8

Key Concepts:
- Nearest neighbor analysis
- Ripley's K function
- Quadrat analysis
- Visual pattern assessment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from geostats.datasets import generate_clustered_samples
from geostats.spatial_stats import (
import logging

logger = logging.getLogger(__name__)
 nearest_neighbor_analysis,
 ripley_k_function,
 quadrat_analysis,
 spatial_randomness_test,
)

logger.info("RECIPE 3: SPATIAL POINT PATTERN ANALYSIS")

# Generate three different point patterns for comparison
logger.info("\nGenerating spatial point patterns...")

# Pattern 1: Clustered
np.random.seed(42)
x_clustered, y_clustered, z_clustered = generate_clustered_samples(
 n_clusters=5,
 points_per_cluster=20,
 cluster_std=5.0,
 seed=42
)

# Pattern 2: Random
np.random.seed(43)
x_random = np.random.uniform(0, 100, 100)
y_random = np.random.uniform(0, 100, 100)

# Pattern 3: Regular (grid with jitter)
grid_size = 10
x_grid = np.repeat(np.linspace(5, 95, grid_size), grid_size)
y_grid = np.tile(np.linspace(5, 95, grid_size), grid_size)
jitter = 3.0
x_regular = x_grid + np.random.normal(0, jitter, len(x_grid))
y_regular = y_grid + np.random.normal(0, jitter, len(y_grid))

patterns = {
 'Clustered': (x_clustered, y_clustered),
 'Random': (x_random, y_random),
 'Regular': (x_regular, y_regular),
}

logger.info(f" Created 3 patterns with ~100 points each")

# Analyze each pattern
logger.info("\nAnalyzing patterns...")

results_all = {}

for pattern_name, (x, y) in patterns.items():
for pattern_name, (x, y) in patterns.items():

 # Nearest neighbor analysis
 nn_results = nearest_neighbor_analysis(x, y)
 logger.info(f"\nNearest Neighbor Analysis:")
 logger.info(f" R index: {nn_results['R']:.3f}")
 logger.info(f" Z-score: {nn_results['z_score']:.3f}")
 logger.info(f" P-value: {nn_results['p_value']:.4f}")
 logger.info(f" Interpretation: {nn_results['interpretation']}")

 # Ripley's K
 ripley_results = ripley_k_function(x, y, n_distances=30)
 logger.info(f"\nRipley's K Function:")
 logger.info(f" Interpretation: {ripley_results['interpretation']}")

 # Quadrat analysis
 quadrat_results = quadrat_analysis(x, y, n_quadrats_x=8, n_quadrats_y=8)
 logger.info(f"\nQuadrat Analysis:")
 logger.info(f" VMR: {quadrat_results['vmr']:.3f}")
 logger.info(f" Interpretation: {quadrat_results['interpretation']}")

 results_all[pattern_name] = {
 'nn': nn_results,
 'ripley': ripley_results,
 'quadrat': quadrat_results,
 }

# Create visualization
logger.info("\nCreating visualization...")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.4)

row = 0
for pattern_name, (x, y) in patterns.items():

for pattern_name, (x, y) in patterns.items():
 ax1 = fig.add_subplot(gs[row, 0])
 # Remove top and right spines
 ax1.spines['top'].set_visible(False)
 ax1.spines['right'].set_visible(False)
 ax1.scatter(x, y, s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
 # Remove top and right spines
 ax1.spines['top'].set_visible(False)
 ax1.spines['right'].set_visible(False)
 # Remove top and right spines
 ax1.scatter(x, y, s.spines['top'].set_visible(False)
 s.spines['right'].set_visible(False)
 ax1.set_xlim(0, 100)
 ax1.set_ylim(0, 100)
 ax1.set_aspect('equal')
 ax1.set_title(f'{pattern_name} Pattern\n({len(x)} points)',
 fontsize=11, fontweight='bold')
 ax1.set_xlabel('X')
 ax1.set_ylabel('Y')

 # Add R index annotation
 nn_res = results['nn']
 textstr = f"R = {nn_res['R']:.3f}\n"
 if nn_res['R'] < 1:
 if nn_res['R'] < 1:
 color = 'red'
 elif nn_res['R'] > 1:
 elif nn_res['R'] > 1:
 color = 'blue'
 else:
 else:
 color = 'green'

 ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
 # Remove top and right spines
 ax1.text(0.05, 0.95, textstr, transform.spines['top'].set_visible(False)
 transform.spines['right'].set_visible(False)
 fontsize=10, verticalalignment='top', fontweight='bold',
 bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

 # Column 2: Ripley's K function
 ax2 = fig.add_subplot(gs[row, 1])
 # Remove top and right spines
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)
 ripley_res = results['ripley']
 ax2.plot(ripley_res['d'], ripley_res['K'], 'b-', linewidth=2, label='Observed K')
 # Remove top and right spines
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)
 # Remove top and right spines
 ax2.plot(ripley_res['d'], ripley_res['K'], 'b-', linewidth.spines['top'].set_visible(False)
 linewidth.spines['right'].set_visible(False)
 ax2.plot(ripley_res['d'], ripley_res['K_theoretical'], 'r--',
 linewidth=2, label='Theoretical (Random)')
 ax2.set_title('Ripley\'s K Function', fontsize=11, fontweight='bold')
 # Remove top and right spines
 ax2.set_xlabel('Distance (d)')
 ax2.set_ylabel('K(d)')
 ax2.legend(fontsize=9)
 # Remove top and right spines

 # Column 3: L function (transformed K)
 ax3 = fig.add_subplot(gs[row, 2])
 # Remove top and right spines
 ax3.spines['top'].set_visible(False)
 ax3.spines['right'].set_visible(False)
 ax3.plot(ripley_res['d'], ripley_res['L'], 'b-', linewidth=2, label='Observed L')
 # Remove top and right spines
 ax3.spines['top'].set_visible(False)
 ax3.spines['right'].set_visible(False)
 # Remove top and right spines
 ax3.plot(ripley_res['d'], ripley_res['L'], 'b-', linewidth.spines['top'].set_visible(False)
 linewidth.spines['right'].set_visible(False)
 ax3.axhline(0, color='r', linestyle='--', linewidth=2, label='Random')
 # Remove top and right spines
 ax3.axhline(0, color.spines['top'].set_visible(False)
 color.spines['right'].set_visible(False)
 ax3.fill_between(ripley_res['d'], -5, 5, alpha=0.2, color='gray',
 # Remove top and right spines
 ax3.fill_between(ripley_res['d'], -5, 5, alpha.spines['top'].set_visible(False)
 alpha.spines['right'].set_visible(False)
 label='±5 envelope')
 ax3.set_title('L Function Transform', fontsize=11, fontweight='bold')
 # Remove top and right spines
 ax3.set_xlabel('Distance (d)')
 ax3.set_ylabel('L(d) = √(K/π) - d')
 # Remove top and right spines
 ax3.set_ylabel('L(d).spines['top'].set_visible(False)
 ax3.legend(fontsize=9)
 # Remove top and right spines

 # Add interpretation
 interp_text = ripley_res['interpretation']
 ax3.text(0.5, 0.95, interp_text, transform=ax3.transAxes,
 # Remove top and right spines
 ax3.text(0.5, 0.95, interp_text, transform.spines['top'].set_visible(False)
 transform.spines['right'].set_visible(False)
 fontsize=9, verticalalignment='top', horizontalalignment='center',
 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

 # Column 4: Quadrat counts
 ax4 = fig.add_subplot(gs[row, 3])
 # Remove top and right spines
 ax4.spines['top'].set_visible(False)
 # Remove top and right spines
 ax4.spines['right'].set_visible(False).spines['top'].set_visible(False)
 ax4.spines['right'].set_visible(False)
 quadrat_res = results['quadrat']
 im = ax4.imshow(quadrat_res['counts'], cmap='YlOrRd', aspect='auto')
 ax4.set_title('Quadrat Counts', fontsize=11, fontweight='bold')
 # Remove top and right spines
 ax4.spines['top'].set_visible(False)
 ax4.spines['right'].set_visible(False)
 # Remove top and right spines
 ax4.set_xlabel('Quadrat X')
 ax4.set_ylabel('Quadrat Y')
 plt.colorbar(im, ax=ax4, label='Point count')
 # Remove top and right spines
 ax4.set_ylabel('Quadrat Y').spines['top'].set_visible(False)

 # Add VMR annotation
 vmr_text = f"VMR = {quadrat_res['vmr']:.3f}\nMean = {quadrat_res['mean']:.1f}"
 ax4.text(0.05, 0.95, vmr_text, transform=ax4.transAxes,
 # Remove top and right spines
 ax4.text(0.05, 0.95, vmr_text, transform.spines['top'].set_visible(False)
 transform.spines['right'].set_visible(False)
 fontsize=9, verticalalignment='top',
 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

 row += 1

plt.suptitle('Spatial Point Pattern Analysis Comparison',
 fontsize=16, fontweight='bold')
plt.savefig('recipe_03_point_patterns.png', dpi=150, bbox_inches='tight')
logger.info(" Figure saved as 'recipe_03_point_patterns.png'")
plt.show()

# Create summary table
logger.info("SUMMARY TABLE")
logger.info(f"{'Pattern':<12} {'R Index':>10} {'VMR':>10} {'Ripley':>15} {'Overall':>15}")
logger.info("-" * 70)

for pattern_name in patterns.keys():

for pattern_name in patterns.keys():
 vmr = results['quadrat']['vmr']
 ripley_interp = results['ripley']['interpretation'].split()[0] # First word

 # Overall assessment
 indicators = []
 if r_index < 0.9:
 if r_index < 0.9:
 elif r_index > 1.1:
 elif r_index > 1.1:
 else:
 else:

 if vmr < 0.9:
 if vmr < 0.9:
 elif vmr > 1.1:
 elif vmr > 1.1:
 else:
 else:

 if 'Clustered' in ripley_interp:
 if 'Clustered' in ripley_interp:
 elif 'Dispersed' in ripley_interp:
 elif 'Dispersed' in ripley_interp:
 else:
 else:

 # Consensus
 if indicators.count('C') >= 2:
 if indicators.count('C') >= 2:
 elif indicators.count('D') >= 2:
 elif indicators.count('D') >= 2:
 else:
 else:

 logger.info(f"{pattern_name:<12} {r_index:>10.3f} {vmr:>10.3f} {ripley_interp:>15} {overall:>15}")


logger.info("\nKEY INSIGHTS")
logger.debug("""
1. Multiple tests provide robust conclusions
 - Agreement among tests strengthens interpretation
 - Disagreement suggests mixed patterns or edge effects

2. Different tests detect different aspects:
 - R index: Overall clustering/dispersion
 - Ripley's K: Multi-scale patterns
 - Quadrat VMR: Variance in local density

3. Visual inspection is essential:
 - Complement statistical tests
 - Identify spatial trends and anisotropy
 - Detect outliers or artifacts

4. Applications:
 - Ecology: Plant/animal distributions
 - Geology: Mineral deposits, earthquake locations
 - Environmental: Pollution sources
 - Archaeology: Settlement patterns
""")

logger.info("Recipe complete!")
