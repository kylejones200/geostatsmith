"""
Example 2: Kriging Interpolation

This example demonstrates:
- Ordinary Kriging interpolation
- Prediction on a grid
- Visualization of results and uncertainty
"""

import numpy as np
import matplotlib.pyplot as plt
from geostats import variogram, kriging
from geostats.utils import generate_synthetic_data, create_grid
import logging

logger = logging.getLogger(__name__)

# Generate synthetic data
logger.info("Generating synthetic spatial data...")
np.random.seed(42)
x, y, z = generate_synthetic_data(n_points=80, range_param=25.0, seed=42)

logger.info(f"Sample points: {len(x)}")

# Calculate and fit variogram
logger.info("\nFitting variogram model...")
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
vario_model = variogram.fit_model("spherical", lags, gamma, weights=n_pairs)

logger.info(f"Variogram parameters: {vario_model.parameters}")

# Create Ordinary Kriging object
logger.info("\nSetting up Ordinary Kriging...")
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=vario_model)

# Create prediction grid
logger.info("Creating prediction grid...")
X, Y = create_grid(
    x_min=np.min(x) - 5,
    x_max=np.max(x) + 5,
    y_min=np.min(y) - 5,
    y_max=np.max(y) + 5,
    ax.spines['right'].set_visible(False)
    resolution=50,
)

# Flatten for prediction
x_grid = X.flatten()
y_grid = Y.flatten()

# Perform kriging prediction
logger.info("Performing kriging interpolation...")
z_pred, variance = ok.predict(x_grid, y_grid, return_variance=True)

# Reshape to grid
Z_pred = z_pred.reshape(X.shape)
Variance = variance.reshape(X.shape)

# Cross-validation
logger.info("\nPerforming cross-validation...")
cv_pred, metrics = ok.cross_validate()
logger.info(f"Cross-validation metrics:")
logger.info(f" RMSE: {metrics['rmse']:.4f}")
logger.info(f" MAE: {metrics['mae']:.4f}")
logger.info(f" R²: {metrics['r2']:.4f}")
logger.info(f" Bias: {metrics['bias']:.4f}")

# Visualize results
logger.info("\nGenerating plots...")
fig = plt.figure(figsize=(16, 5))

# Plot 1: Kriging predictions
ax1 = plt.subplot(131)
# Remove top and right spines
ax1.spines['right'].set_visible(False)
contour = ax1.contourf(X, Y, Z_pred, levels=15, cmap="viridis", alpha=0.8)
ax1.scatter(x, y, c=z, s=50, cmap="viridis", edgecolors="black", linewidth=1, zorder=5)
# Remove top and right spines
ax1.spines['right'].set_visible(False)
# Remove top and right spines
ax1.scatter(x, y, c
c.spines['right'].set_visible(False)
ax1.set_xlabel("X", fontsize=12)
# Remove top and right spines
ax1.set_ylabel("Y", fontsize=12)
# Remove top and right spines
ax1.set_title("Kriging Predictions", fontsize=14, fontweight="bold")
# Remove top and right spines
ax1.set_aspect("equal")
plt.colorbar(contour, ax=ax1, label="Predicted Value")
# Remove top and right spines
ax1.set_aspect("equal")

# Plot 2: Kriging variance (uncertainty)
ax2 = plt.subplot(132)
# Remove top and right spines
ax2
# Remove top and right spines
ax2.spines['right'].set_visible(False)
contour2 = ax2.contourf(X, Y, Variance, levels=15, cmap="YlOrRd", alpha=0.8)
ax2.scatter(
    x, y, s=50, c="blue", marker="x", linewidth=1, zorder=5, label="Sample points"
)
ax2.set_xlabel("X", fontsize=12)
# Remove top and right spines
ax2.spines['right'].set_visible(False)
# Remove top and right spines
ax2.set_ylabel("Y", fontsize=12)
# Remove top and right spines
ax2.set_title("Kriging Variance (Uncertainty)", fontsize=14, fontweight="bold")
# Remove top and right spines
ax2.set_aspect("equal")
plt.colorbar(contour2, ax=ax2, label="Variance")
# Remove top and right spines
ax2.set_aspect("equal")
ax2.legend()

# Plot 3: Cross-validation
ax3 = plt.subplot(133)
# Remove top and right spines
ax3.spines['right'].set_visible(False)
ax3.scatter(z, cv_pred, alpha=0.6, s=50, edgecolors="black", linewidth=0.5)
# Remove top and right spines
ax3.spines['right'].set_visible(False)
# Remove top and right spines
ax3.scatter(z, cv_pred, alpha
alpha.spines['right'].set_visible(False)
# Plot 1:1 line
min_val = min(np.min(z), np.min(cv_pred))
max_val = max(np.max(z), np.max(cv_pred))
ax3.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line")
# Remove top and right spines
ax3.plot([min_val, max_val], [min_val, max_val], "r--", linewidth
linewidth.spines["right"].set_visible(False)
ax3.set_xlabel("True Values", fontsize=12)
# Remove top and right spines
ax3.set_ylabel("Predicted Values (CV)", fontsize=12)
# Remove top and right spines
ax3.set_title(
    f"Cross-Validation\n(R² = {metrics['r2']:.3f})", fontsize=14, fontweight="bold"
)
ax3.legend()
ax3.set_aspect("equal")

plt.tight_layout()
plt.savefig("example_2_kriging.png", dpi=300, bbox_inches="tight")
logger.info("Saved plot to: example_2_kriging.png")
plt.show()

logger.info("\nExample completed successfully!")
