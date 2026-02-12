"""
Example 3: Comparison of Kriging Methods

This example compares:
- Simple Kriging
- Ordinary Kriging
- Universal Kriging
"""

import numpy as np
import matplotlib.pyplot as plt
from geostats import variogram, kriging
from geostats.utils import create_grid
import logging

logger = logging.getLogger(__name__)

# Generate data with trend
logger.info("Generating data with spatial trend...")
np.random.seed(123)
n = 100
x = np.random.uniform(0, 100, n)
y = np.random.uniform(0, 100, n)

# Add linear trend + spatial correlation
trend = 0.05 * x + 0.03 * y
spatial_var = 10 * np.sin(x / 15) * np.cos(y / 15)
noise = np.random.randn(n) * 2
z = trend + spatial_var + noise

logger.info(f"Generated {n} points with trend")

# Calculate variogram
logger.info("\nFitting variogram...")
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=15)
vario_model = variogram.fit_model("exponential", lags, gamma, weights=n_pairs)

# Create prediction grid
X, Y = create_grid(0, 100, 0, 100, resolution=40)
x_grid, y_grid = X.flatten(), Y.flatten()

# Simple Kriging
logger.info("\nPerforming Simple Kriging...")
sk = kriging.SimpleKriging(x, y, z, variogram_model=vario_model, mean=np.mean(z))
z_sk, var_sk = sk.predict(x_grid, y_grid, return_variance=True)
cv_sk, metrics_sk = sk.cross_validate()

# Ordinary Kriging
logger.info("Performing Ordinary Kriging...")
ok = kriging.OrdinaryKriging(x, y, z, variogram_model=vario_model)
z_ok, var_ok = ok.predict(x_grid, y_grid, return_variance=True)
cv_ok, metrics_ok = ok.cross_validate()

# Universal Kriging with linear drift
logger.info("Performing Universal Kriging...")
uk = kriging.UniversalKriging(
    x, y, z, variogram_model=vario_model, drift_terms="linear"
)
z_uk, var_uk = uk.predict(x_grid, y_grid, return_variance=True)
cv_uk, metrics_uk = uk.cross_validate()

# Print metrics
logger.info("\nCross-validation metrics:")
logger.info(f"{'Method':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
logger.info("-" * 50)
logger.info(
    f"{'Simple Kriging':<20} {metrics_sk['rmse']:<10.4f} {metrics_sk['mae']:<10.4f} {metrics_sk['r2']:<10.4f}"
)
logger.info(
    f"{'Ordinary Kriging':<20} {metrics_ok['rmse']:<10.4f} {metrics_ok['mae']:<10.4f} {metrics_ok['r2']:<10.4f}"
)
logger.info(
    f"{'Universal Kriging':<20} {metrics_uk['rmse']:<10.4f} {metrics_uk['mae']:<10.4f} {metrics_uk['r2']:<10.4f}"
)

# Visualize
logger.info("\nGenerating comparison plots...")
fig = plt.figure(figsize=(16, 10))

methods = [
    ("Simple Kriging", z_sk, var_sk, metrics_sk),
    ("Ordinary Kriging", z_ok, var_ok, metrics_ok),
    ("Universal Kriging", z_uk, var_uk, metrics_uk),
]

for idx, (method_name, z_pred, var_pred, metrics) in enumerate(methods, 1):
for idx, (method_name, z_pred, var_pred, metrics) in enumerate(methods, 1):
    Z_pred = z_pred.reshape(X.shape)
    contour = ax1.contourf(X, Y, Z_pred, levels=15, cmap="viridis", alpha=0.8)
    ax1.scatter(
        x, y, c=z, s=30, cmap="viridis", edgecolors="white", linewidth=0.5, zorder=5
    )
    ax1.set_title(
        f"{method_name}\n(R² = {metrics['r2']:.3f})", fontsize=12, fontweight="bold"
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal")
    plt.colorbar(contour, ax=ax1, label="Value")

    # Variance
    ax2 = plt.subplot(2, 3, idx + 3)
    Var_pred = var_pred.reshape(X.shape)
    contour2 = ax2.contourf(X, Y, Var_pred, levels=15, cmap="YlOrRd", alpha=0.8)
    ax2.scatter(x, y, s=20, c="blue", marker="x", linewidth=0.5, zorder=5)
    ax2.set_title(f"{method_name} - Variance", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")
    plt.colorbar(contour2, ax=ax2, label="Variance")

plt.tight_layout()
plt.savefig("example_3_comparison.png", dpi=300, bbox_inches="tight")
logger.info("Saved plot to: example_3_comparison.png")
plt.show()

logger.info("\nExample completed successfully!")
