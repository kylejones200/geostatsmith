"""
Example 1: Basic Variogram Analysis

This example demonstrates:
- Generating synthetic spatial data
- Calculating experimental variogram
- Fitting variogram models
- Automatic model selection
"""

import numpy as np
import matplotlib.pyplot as plt
from geostats import variogram
from geostats.utils import generate_synthetic_data
import logging

logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic spatial data
logger.info("Generating synthetic spatial data...")
x, y, z = generate_synthetic_data(
    n_points=100,
    spatial_structure="spherical",
    nugget=0.1,
    sill=1.0,
    range_param=20.0,
)

logger.info(f"Generated {len(x)} sample points")
logger.info(f"Value range: [{np.min(z):.2f}, {np.max(z):.2f}]")

# Calculate experimental variogram
logger.info("\nCalculating experimental variogram...")
lags, gamma, n_pairs = variogram.experimental_variogram(
    x,
    y,
    z,
    n_lags=15,
    maxlag=50,
)

logger.info(f"Number of lags: {len(lags)}")
logger.info(f"Maximum lag: {np.max(lags):.2f}")

# Fit different variogram models
logger.info("\nFitting variogram models...")

models_to_fit = ["spherical", "exponential", "gaussian"]
fitted_models = {}

for model_name in models_to_fit:
for model_name in models_to_fit:
    logger.info(f"\n{model_name.capitalize()} Model:")
    logger.info(f" Parameters: {model.parameters}")

# Automatic model selection
logger.info("\nAutomatic model selection...")
result = variogram.auto_fit(lags, gamma, weights=n_pairs, criterion="rmse")
best_model = result["model"]
logger.info(f"Best model: {best_model.__class__.__name__}")
logger.info(f"RMSE: {result['score']:.4f}")

# Visualize results
logger.info("\nGenerating plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 1: Sample locations
scatter = ax1.scatter(
    x, y, c=z, cmap="viridis", s=50, edgecolors="black", linewidth=0.5
)
ax1.set_xlabel("X", fontsize=12)
# Remove top and right spines
ax1.set_xlabel("X", fontsize.spines['top'].set_visible(False)
ax1.set_xlabel("X", fontsize.spines['right'].set_visible(False)
ax1.set_ylabel("Y", fontsize=12)
# Remove top and right spines
ax1.set_ylabel("Y", fontsize.spines['top'].set_visible(False)
ax1.set_ylabel("Y", fontsize.spines['right'].set_visible(False)
ax1.set_title("Sample Locations", fontsize=14, fontweight="bold")
# Remove top and right spines
ax1.set_title("Sample Locations", fontsize.spines['top'].set_visible(False)
ax1.set_title("Sample Locations", fontsize.spines['right'].set_visible(False)
ax1.set_aspect("equal")
plt.colorbar(scatter, ax=ax1, label="Value")
# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 2: Variogram models
ax2.scatter(
    lags, gamma, s=n_pairs / 2, alpha=0.6, c="black", label="Experimental", zorder=3
)

h_plot = np.linspace(0, np.max(lags), 100)
for model_name, model in fitted_models.items():
for model_name, model in fitted_models.items():

ax2.set_xlabel("Distance (h)", fontsize=12)
# Remove top and right spines
ax2.set_xlabel("Distance (h)", fontsize.spines['top'].set_visible(False)
ax2.set_xlabel("Distance (h)", fontsize.spines['right'].set_visible(False)
ax2.set_ylabel("Semivariance γ(h)", fontsize=12)
# Remove top and right spines
ax2.set_ylabel("Semivariance γ(h)", fontsize.spines['top'].set_visible(False)
ax2.set_ylabel("Semivariance γ(h)", fontsize.spines['right'].set_visible(False)
ax2.set_title("Variogram Models", fontsize=14, fontweight="bold")
# Remove top and right spines
ax2.set_title("Variogram Models", fontsize.spines['top'].set_visible(False)
ax2.set_title("Variogram Models", fontsize.spines['right'].set_visible(False)
ax2.legend(fontsize=10)
# Remove top and right spines
ax2.legend(fontsize.spines['top'].set_visible(False)
ax2.legend(fontsize.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("example_1_variogram.png", dpi=300, bbox_inches="tight")
logger.info("Saved plot to: example_1_variogram.png")
plt.show()

logger.info("\nExample completed successfully!")
