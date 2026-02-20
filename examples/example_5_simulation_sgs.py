"""
    Example 5: Sequential Gaussian Simulation (SGS)

This example demonstrates:
    pass
- Unconditional and conditional simulation
- Multiple realizations for uncertainty quantification
- E-type estimates and probability maps
- Comparison with kriging

Based on Zhang, Y. (2010). Course Notes, Section 6.3.3
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "../src")

from geostats.datasets import get_walker_lake_subset
from geostats import variogram, kriging, simulation
from geostats.utils import create_grid
import logging

logger = logging.getLogger(__name__)

# Load subset of Walker Lake data
logger.info("Loading Walker Lake data subset...")
data = get_walker_lake_subset(n_samples=30, seed=42)
x, y, V = data["x"], data["y"], data["V"]

logger.info(f"Conditioning data: {len(x)} samples")
logger.info(f"V range: [{np.min(V):.1f}, {np.max(V):.1f}] ppm")

# Fit variogram
logger.info("\nFitting variogram...")
lags, gamma, n_pairs = variogram.experimental_variogram(x, y, V, n_lags=8)
vario_model = variogram.fit_model("spherical", lags, gamma, weights=n_pairs)
logger.info(f"Variogram parameters: {vario_model.parameters}")

# Create simulation grid
logger.info("\nCreating simulation grid...")
X, Y = create_grid(x_min=0, x_max=90, y_min=0, y_max=90, resolution=30)
# Remove top and right spines

x_grid, y_grid = X.flatten(), Y.flatten()

# Perform Sequential Gaussian Simulation
logger.info("\nPerforming Sequential Gaussian Simulation...")
sgs = simulation.SequentialGaussianSimulation(x, y, V, vario_model)

# Generate multiple realizations
n_realizations = 5
logger.info(f"Generating {n_realizations} realizations...")
realizations_flat = sgs.simulate(
    x_grid, y_grid, n_realizations=n_realizations, seed=123
)

# Reshape to grid
realizations = realizations_flat.reshape(n_realizations, X.shape[0], X.shape[1])

# Calculate statistics from realizations
logger.info("\nCalculating statistics...")
mean_sgs, std_sgs, p10, p90 = sgs.get_statistics(realizations_flat)

# Reshape statistics
Mean_sgs = mean_sgs.reshape(X.shape)
Std_sgs = std_sgs.reshape(X.shape)
P10 = p10.reshape(X.shape)
P90 = p90.reshape(X.shape)

# Compare with Ordinary Kriging
logger.info("\nPerforming Ordinary Kriging for comparison...")
ok = kriging.OrdinaryKriging(x, y, V, variogram_model=vario_model)
z_ok, var_ok = ok.predict(x_grid, y_grid, return_variance=True)
Z_ok = z_ok.reshape(X.shape)
Var_ok = var_ok.reshape(X.shape)

# Visualize
logger.info("\nGenerating plots...")
fig = plt.figure(figsize=(18, 12))

# Plot realizations
for i in range(min(4, n_realizations)):
    ax.scatter(
        x, y, c=V, cmap="viridis", s=60, edgecolors="white", linewidth=1.5, zorder=5
    )
    plt.colorbar(contour, ax=ax, label="V (ppm)")
    # Remove top and right spines
    
    ax.set_title(f"Realization {i + 1}", fontweight="bold", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

# E-type estimate (mean of realizations)
ax5 = plt.subplot(3, 4, 5)
# Remove top and right spines
# Remove top and right spines

contour5 = ax5.contourf(X, Y, Mean_sgs, levels=15, cmap="viridis", alpha=0.9)
ax5.scatter(
    x, y, c=V, cmap="viridis", s=60, edgecolors="white", linewidth=1.5, zorder=5
)
plt.colorbar(contour5, ax=ax5, label="V (ppm)")
# Remove top and right spines
ax5.scatter()
ax5.set_title("E-type (Mean of Realizations)", fontweight="bold", fontsize=11)
# Remove top and right spines

# Remove top and right spines
ax5.set_title("E-type (Mean of Realizations)", fontweight)

ax5.set_xlabel("X (m)")
ax5.set_ylabel("Y (m)")
ax5.set_aspect("equal")

# Standard deviation
ax6 = plt.subplot(3, 4, 6)
# Remove top and right spines
# Remove top and right spines

contour6 = ax6.contourf(X, Y, Std_sgs, levels=15, cmap="YlOrRd", alpha=0.9)
ax6.scatter(x, y, s=40, c="blue", marker="x", linewidth=2, zorder=5)
# Remove top and right spines

# Remove top and right spines
ax6.scatter(x, y, s)

plt.colorbar(contour6, ax=ax6, label="Std Dev")
# Remove top and right spines
ax6.scatter(x, y, s)

ax6.set_title("Standard Deviation (SGS)", fontweight="bold", fontsize=11)
# Remove top and right spines
ax6.set_title("Standard Deviation (SGS)", fontweight)

ax6.set_xlabel("X (m)")
ax6.set_ylabel("Y (m)")
ax6.set_aspect("equal")

# P10 and P90 (uncertainty bounds)
ax7 = plt.subplot(3, 4, 7)
# Remove top and right spines
# Remove top and right spines

contour7 = ax7.contourf(X, Y, P10, levels=15, cmap="viridis", alpha=0.9)
ax7.scatter(
    x, y, c=V, cmap="viridis", s=60, edgecolors="white", linewidth=1.5, zorder=5
)
plt.colorbar(contour7, ax=ax7, label="V (ppm)")
# Remove top and right spines
ax7.scatter()
ax7.set_title("P10 (Conservative Estimate)", fontweight="bold", fontsize=11)
# Remove top and right spines

# Remove top and right spines
ax7.set_title("P10 (Conservative Estimate)", fontweight)

ax7.set_xlabel("X (m)")
ax7.set_ylabel("Y (m)")
ax7.set_aspect("equal")

ax8 = plt.subplot(3, 4, 8)
# Remove top and right spines
# Remove top and right spines

contour8 = ax8.contourf(X, Y, P90, levels=15, cmap="viridis", alpha=0.9)
ax8.scatter(
    x, y, c=V, cmap="viridis", s=60, edgecolors="white", linewidth=1.5, zorder=5
)
plt.colorbar(contour8, ax=ax8, label="V (ppm)")
# Remove top and right spines
ax8.scatter()
ax8.set_title("P90 (Optimistic Estimate)", fontweight="bold", fontsize=11)
# Remove top and right spines

# Remove top and right spines
ax8.set_title("P90 (Optimistic Estimate)", fontweight)

ax8.set_xlabel("X (m)")
ax8.set_ylabel("Y (m)")
ax8.set_aspect("equal")

# Kriging prediction for comparison
ax9 = plt.subplot(3, 4, 9)
# Remove top and right spines
# Remove top and right spines

contour9 = ax9.contourf(X, Y, Z_ok, levels=15, cmap="viridis", alpha=0.9)
ax9.scatter(
    x, y, c=V, cmap="viridis", s=60, edgecolors="white", linewidth=1.5, zorder=5
)
plt.colorbar(contour9, ax=ax9, label="V (ppm)")
# Remove top and right spines
ax9.scatter()
ax9.set_title("Ordinary Kriging", fontweight="bold", fontsize=11)
# Remove top and right spines

# Remove top and right spines
ax9.set_title("Ordinary Kriging", fontweight)

ax9.set_xlabel("X (m)")
ax9.set_ylabel("Y (m)")
ax9.set_aspect("equal")

# Kriging variance
ax10 = plt.subplot(3, 4, 10)
# Remove top and right spines
# Remove top and right spines

contour10 = ax10.contourf(X, Y, Var_ok, levels=15, cmap="YlOrRd", alpha=0.9)
ax10.scatter(x, y, s=40, c="blue", marker="x", linewidth=2, zorder=5)
# Remove top and right spines

# Remove top and right spines
ax10.scatter(x, y, s)

plt.colorbar(contour10, ax=ax10, label="Variance")
# Remove top and right spines
ax10.scatter(x, y, s)

ax10.set_title("Kriging Variance", fontweight="bold", fontsize=11)
# Remove top and right spines
ax10.set_title("Kriging Variance", fontweight)

ax10.set_xlabel("X (m)")
ax10.set_ylabel("Y (m)")
ax10.set_aspect("equal")

# Histogram comparison
ax11 = plt.subplot(3, 4, 11)
# Remove top and right spines

ax11.hist(V, bins=15, alpha=0.5, label="Original Data", density=True, edgecolor="black")
# Remove top and right spines

# Remove top and right spines
ax11.hist(V, bins)

ax11.hist(
    Mean_sgs.flatten(),
    bins=15,
    alpha=0.5,
    label="SGS E-type",
    density=True,
    edgecolor="black",
)
ax11.hist(
    Z_ok.flatten(), bins=15, alpha=0.5, label="OK", density=True, edgecolor="black"
)
ax11.set_xlabel("V (ppm)")
ax11.set_ylabel("Density")
ax11.set_title("Histogram Comparison", fontweight="bold", fontsize=11)
# Remove top and right spines

ax11.legend()

# Comparison scatter: SGS mean vs Kriging
ax12 = plt.subplot(3, 4, 12)
# Remove top and right spines

ax12.scatter(Z_ok.flatten(), Mean_sgs.flatten(), alpha=0.3, s=10)
# Remove top and right spines

# Remove top and right spines
ax12.scatter(Z_ok.flatten(), Mean_sgs.flatten(), alpha)

min_val = min(np.min(Z_ok), np.min(Mean_sgs))
max_val = max(np.max(Z_ok), np.max(Mean_sgs))
ax12.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
# Remove top and right spines
ax12.plot([min_val, max_val], [min_val, max_val], "r--", linewidth)

ax12.set_xlabel("Kriging Prediction")
ax12.set_ylabel("SGS E-type")
ax12.set_title("SGS vs Kriging", fontweight="bold", fontsize=11)
# Remove top and right spines
ax12.set_title("SGS vs Kriging", fontweight)

ax12.set_aspect("equal")

plt.tight_layout()
plt.savefig("example_5_simulation_sgs.png", dpi=300, bbox_inches="tight")
logger.info("Saved plot to: example_5_simulation_sgs.png")
plt.show()

# Summary statistics

logger.info("SIMULATION SUMMARY")

logger.info(f"Number of realizations: {n_realizations}")
logger.info(f"\nOriginal data:")
logger.info(f" Mean: {np.mean(V):.2f}")
logger.info(f" Std: {np.std(V):.2f}")
logger.info(f"\nSGS E-type estimate:")
logger.info(f" Mean: {np.mean(Mean_sgs):.2f}")
logger.info(f" Mean Std Dev: {np.mean(Std_sgs):.2f}")
logger.info(f"\nKriging:")
logger.info(f" Mean prediction: {np.mean(Z_ok):.2f}")
logger.info(f" Mean variance: {np.mean(Var_ok):.2f}")

logger.info("\nSimulation captures spatial variability and uncertainty!")
logger.info("Multiple realizations provide full uncertainty quantification.")
logger.info("\nExample completed successfully!")
