"""
Example 4: Indicator Kriging for Probability Estimation

This example demonstrates:
- Indicator transformation
- Probability estimation with indicator kriging
- Multiple threshold indicator kriging for CDF estimation
- Risk assessment applications

Based on Zhang, Y. (2010). Course Notes, Section 6.2.2
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')

from geostats.datasets import load_walker_lake
from geostats import variogram, kriging
from geostats.utils import create_grid

# Load Walker Lake data
logger.info("Loading Walker Lake dataset...")
data = load_walker_lake()
x, y, V = data['x'], data['y'], data['V']

logger.info(f"Samples: {len(x)}")
logger.info(f"V range: [{np.min(V)}, {np.max(V)}] ppm")

# Define threshold (e.g., regulatory limit)
threshold = 100.0 # ppm
logger.info(f"\nThreshold: {threshold} ppm")

# Create indicator variable
indicators = (V > threshold).astype(float)
exceedance_rate = np.mean(indicators)
logger.info(f"Exceedance rate in samples: {exceedance_rate*100:.1f}%")

# Calculate indicator variogram
logger.info("\nFitting indicator variogram...")
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, indicators, n_lags=8)
indicator_model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)
logger.info(f"Indicator variogram parameters: {indicator_model.parameters}")

# Perform indicator kriging
logger.info("\nPerforming indicator kriging...")
ik = kriging.IndicatorKriging(x, y, V, threshold=threshold, variogram_model=indicator_model)

# Create prediction grid
X, Y = create_grid(x_min=-5, x_max=95, y_min=-5, y_max=95, resolution=40)
x_grid, y_grid = X.flatten(), Y.flatten()

# Predict probabilities
probabilities, variance = ik.predict(x_grid, y_grid, return_variance=True)

# Reshape to grid
P_grid = probabilities.reshape(X.shape)
V_grid = variance.reshape(X.shape)

# Cross-validation
logger.info("\nCross-validation...")
cv_pred, metrics = ik.cross_validate()
logger.info(f" RMSE: {metrics['rmse']:.4f}")
logger.info(f" MAE: {metrics['mae']:.4f}")

# Multiple threshold analysis
logger.info("\nMultiple threshold indicator kriging...")
thresholds = [70, 85, 100, 115, 130]
multi_ik = kriging.MultiThresholdIndicatorKriging(x, y, V, thresholds=thresholds)
multi_ik.fit()

# Predict median (50th percentile)
median_pred = multi_ik.predict_quantile(x_grid, y_grid, quantile=0.5)
Median_grid = median_pred.reshape(X.shape)

# Visualize
logger.info("\nGenerating plots...")
fig = plt.figure(figsize=(16, 10))

# Plot 1: Original data
ax1 = plt.subplot(2, 3, 1)
scatter1 = ax1.scatter(x, y, c=V, cmap='RdYlGn_r', s=100, edgecolors='black', linewidth=1)
ax1.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.5)
plt.colorbar(scatter1, ax=ax1, label='V (ppm)')
ax1.set_title(f'Sample Data (threshold={threshold} ppm)', fontweight='bold', fontsize=12)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Plot 2: Indicator values
ax2 = plt.subplot(2, 3, 2)
colors = ['green' if i == 0 else 'red' for i in indicators]
ax2.scatter(x, y, c=colors, s=100, edgecolors='black', linewidth=1, alpha=0.7)
ax2.set_title(f'Indicators (V > {threshold})', fontweight='bold', fontsize=12)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
# Add legend
from matplotlib.patches import Patch
import logging

logger = logging.getLogger(__name__)
legend_elements = [Patch(facecolor='green', label='Below threshold'),
 Patch(facecolor='red', label='Above threshold')]
ax2.legend(handles=legend_elements, loc='upper right')

# Plot 3: Indicator variogram
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(lags, gamma, s=n_pairs/np.max(n_pairs)*150+30, alpha=0.6, c='black', edgecolors='black')
h_plot = np.linspace(0, np.max(lags)*1.1, 100)
gamma_plot = indicator_model(h_plot)
ax3.plot(h_plot, gamma_plot, 'r-', linewidth=2, label='Spherical Model')
ax3.set_xlabel('Distance (h)')
ax3.set_ylabel('γ(h)')
ax3.set_title('Indicator Variogram', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Probability map
ax4 = plt.subplot(2, 3, 4)
contour4 = ax4.contourf(X, Y, P_grid, levels=15, cmap='RdYlGn_r', alpha=0.9)
ax4.scatter(x, y, c=indicators, cmap='RdYlGn_r', s=60,
 edgecolors='white', linewidth=1.5, zorder=5)
plt.colorbar(contour4, ax=ax4, label='P(V > 100 ppm)')
ax4.set_title('Probability of Exceedance', fontweight='bold', fontsize=12)
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_aspect('equal')

# Plot 5: Uncertainty
ax5 = plt.subplot(2, 3, 5)
contour5 = ax5.contourf(X, Y, V_grid, levels=15, cmap='YlOrRd', alpha=0.9)
ax5.scatter(x, y, s=40, c='blue', marker='x', linewidth=2, zorder=5)
plt.colorbar(contour5, ax=ax5, label='Variance')
ax5.set_title('Indicator Kriging Variance', fontweight='bold', fontsize=12)
ax5.set_xlabel('X (m)')
ax5.set_ylabel('Y (m)')
ax5.set_aspect('equal')

# Plot 6: Predicted median from multi-threshold IK
ax6 = plt.subplot(2, 3, 6)
contour6 = ax6.contourf(X, Y, Median_grid, levels=15, cmap='viridis', alpha=0.9)
ax6.scatter(x, y, c=V, cmap='viridis', s=60,
 edgecolors='white', linewidth=1.5, zorder=5)
plt.colorbar(contour6, ax=ax6, label='Median V (ppm)')
ax6.set_title('Predicted Median (Multi-threshold IK)', fontweight='bold', fontsize=12)
ax6.set_xlabel('X (m)')
ax6.set_ylabel('Y (m)')
ax6.set_aspect('equal')

plt.tight_layout()
plt.savefig('example_4_indicator_kriging.png', dpi=300, bbox_inches='tight')
logger.info("Saved plot to: example_4_indicator_kriging.png")
plt.show()

# Risk assessment summary

logger.info("RISK ASSESSMENT SUMMARY")

logger.info(f"Threshold: {threshold} ppm")
logger.info(f"Sample exceedance rate: {exceedance_rate*100:.1f}%")
logger.info(f"Mean predicted probability: {np.mean(probabilities)*100:.1f}%")
logger.info(f"Max predicted probability: {np.max(probabilities)*100:.1f}%")
logger.info(f"Min predicted probability: {np.min(probabilities)*100:.1f}%")

# Identify high-risk areas (P > 0.7)
high_risk = probabilities > 0.7
n_high_risk = np.sum(high_risk)
pct_high_risk = (n_high_risk / len(probabilities)) * 100
logger.info(f"\nHigh-risk locations (P>0.7): {n_high_risk} ({pct_high_risk:.1f}% of area)")

logger.info("\nExample completed successfully!")
