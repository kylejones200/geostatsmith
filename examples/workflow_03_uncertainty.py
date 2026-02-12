"""
Example Workflow: Uncertainty Quantification
=============================================

Demonstrates how to:
1. Compute bootstrap confidence intervals
2. Create probability maps
3. Perform risk assessment
4. Quantify prediction uncertainty

Author: geostats development team
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# Try importing geostats
try:
try:
        probability_map,
        risk_assessment,
        confidence_intervals,
    )
    from geostats.algorithms.ordinary_kriging import OrdinaryKriging
    from geostats.models.variogram_models import SphericalModel
    from geostats.algorithms.variogram import experimental_variogram
    from geostats.algorithms.fitting import fit_variogram
except ImportError:
    logger.info("Please install geostats: pip install -e .")
    exit(1)


def example_1_bootstrap_confidence():
def example_1_bootstrap_confidence():
    logger.info("Example 1: Bootstrap Confidence Intervals")

    # Create sample data
    logger.info("Creating sample data...")
    np.random.seed(42)
    n = 40
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = 50 + 0.3 * x + 0.2 * y + 10 * np.sin(x / 20) + np.random.normal(0, 3, n)

    # Fit variogram
    logger.info("Fitting variogram...")
    lags, gamma = experimental_variogram(x, y, z)
    variogram_model = fit_variogram(lags, gamma, model_type="spherical")

    # Create prediction transect
    x_pred = np.linspace(0, 100, 100)
    y_pred = np.ones(100) * 50

    # Bootstrap uncertainty
    logger.info("Computing bootstrap uncertainty (100 iterations)...")
    results = bootstrap_uncertainty(
        x,
        y,
        z,
        x_pred,
        y_pred,
        variogram_model=variogram_model,
        n_bootstrap=100,
        confidence_level=0.95,
        method="residual",
    )

    logger.info(f" Bootstrap complete")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))

    # Mean prediction
    ax.plot(x_pred, results["mean"], "b-", linewidth=2, label="Mean Prediction")

    # Confidence bands
    ax.fill_between(
        x_pred,
        results["lower_bound"],
        results["upper_bound"],
        alpha=0.3,
        color="blue",
        label="95% CI",
    )
    ax.fill_between(
        x_pred,
        results["percentile_2.5"],
        results["percentile_97.5"],
        alpha=0.2,
        color="lightblue",
        label="95% Percentile",
    )

    # Original samples (only those near the transect)
    mask = np.abs(y - 50) < 10
    ax.scatter(
        x[mask],
        z[mask],
        c="red",
        s=100,
        marker="o",
        label="Samples",
        edgecolor="k",
        zorder=5,
    )

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Bootstrap Confidence Intervals (Transect at Y=50m)", fontsize=14)
    ax.legend()
    ax.grid(False)

    plt.tight_layout()
    plt.savefig("example_workflow_03_bootstrap.png", dpi=150, bbox_inches="tight")
    logger.info(" Saved example_workflow_03_bootstrap.png")
    plt.close()


def example_2_probability_map():
def example_2_probability_map():
    logger.info("Example 2: Probability Maps")

    # Create contamination data
    logger.info("Creating contamination scenario...")
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)

    # Contamination hotspot at (70, 70)
    distance_to_source = np.sqrt((x - 70) ** 2 + (y - 70) ** 2)
    z = 5 + 15 * np.exp(-distance_to_source / 20) + np.random.normal(0, 2, n)

    # Fit variogram
    logger.info("Fitting variogram...")
    lags, gamma = experimental_variogram(x, y, z)
    variogram_model = fit_variogram(lags, gamma, model_type="exponential")

    # Create prediction grid
    nx, ny = 50, 50
    x_grid = np.linspace(0, 100, nx)
    y_grid = np.linspace(0, 100, ny)
    x_pred_2d, y_pred_2d = np.meshgrid(x_grid, y_grid)
    x_pred = x_pred_2d.ravel()
    y_pred = y_pred_2d.ravel()

    # Probability of exceeding regulatory limit (threshold = 12)
    logger.info("Computing probability map (50 realizations)...")
    prob = probability_map(
        x,
        y,
        z,
        x_pred,
        y_pred,
        variogram_model=variogram_model,
        threshold=12.0,
        operator=">",
        n_realizations=50,
    )

    prob_grid = prob.reshape((ny, nx))

    logger.info(f" Probability map complete")
    logger.info(f" Max probability: {prob.max():.2%}")
    logger.info(f" Area with P>50%: {(prob > 0.5).sum() / len(prob):.1%}")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Kriging prediction
    krig = OrdinaryKriging(x, y, z, variogram_model)
    z_pred, _ = krig.predict(x_pred, y_pred, return_variance=True)
    z_pred_grid = z_pred.reshape((ny, nx))

    im1 = ax1.contourf(x_grid, y_grid, z_pred_grid, levels=15, cmap="YlOrRd")
    ax1.contour(
        x_grid,
        y_grid,
        z_pred_grid,
        levels=[12],
        colors="black",
        linewidths=2,
        linestyles="--",
    )
    ax1.scatter(x, y, c="blue", s=30, edgecolor="k", alpha=0.7, label="Samples")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Kriging Prediction\n(dashed = regulatory limit)")
    ax1.legend()
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1, label="Concentration")

    # Probability map
    im2 = ax2.contourf(
        x_grid, y_grid, prob_grid, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r"
    )
    ax2.contour(
        x_grid,
        y_grid,
        prob_grid,
        levels=[0.5],
        colors="black",
        linewidths=3,
        linestyles="-",
    )
    ax2.scatter(x, y, c="blue", s=30, edgecolor="k", alpha=0.7)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("P(Concentration > 12)\n(solid line = 50% probability)")
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2, label="Probability", ticks=np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.savefig("example_workflow_03_probability.png", dpi=150, bbox_inches="tight")
    logger.info(" Saved example_workflow_03_probability.png")
    plt.close()


def example_3_risk_assessment():
def example_3_risk_assessment():

    logger.info("Example 3: Risk Assessment")

    # Use same contamination data from example 2
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    distance_to_source = np.sqrt((x - 70) ** 2 + (y - 70) ** 2)
    z = 5 + 15 * np.exp(-distance_to_source / 20) + np.random.normal(0, 2, n)

    # Fit variogram
    lags, gamma = experimental_variogram(x, y, z)
    variogram_model = fit_variogram(lags, gamma, model_type="exponential")

    # Create prediction grid
    nx, ny = 30, 30
    x_grid = np.linspace(0, 100, nx)
    y_grid = np.linspace(0, 100, ny)
    x_pred_2d, y_pred_2d = np.meshgrid(x_grid, y_grid)
    x_pred = x_pred_2d.ravel()
    y_pred = y_pred_2d.ravel()

    # Risk assessment
    logger.info("Performing risk assessment...")
    logger.info(" Threshold (regulatory limit): 12")
    logger.info(" Cost of false positive (unnecessary remediation): $10,000")
    logger.info(" Cost of false negative (health risk): $100,000")

    results = risk_assessment(
        x,
        y,
        z,
        x_pred,
        y_pred,
        variogram_model=variogram_model,
        threshold=12.0,
        cost_false_positive=10000,
        cost_false_negative=100000,
        n_realizations=50,
    )

    # Count decisions
    n_remediate = np.sum(results["optimal_decision"] == "positive")
    n_no_action = np.sum(results["optimal_decision"] == "negative")
    total_cost = results["total_expected_cost"].sum()

    logger.info(f"Risk assessment complete:")
    logger.info(
        f" Recommend remediation: {n_remediate} cells ({n_remediate / len(x_pred) * 100:.1f}%)"
    )
    logger.info(
        f" Recommend no action: {n_no_action} cells ({n_no_action / len(x_pred) * 100:.1f}%)"
    )
    logger.info(f" Total expected cost: ${total_cost:,.2f}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Probability map
    prob_grid = results["probability_exceed"].reshape((ny, nx))
    im1 = axes[0].contourf(
        x_grid, y_grid, prob_grid, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r"
    )
    axes[0].scatter(x, y, c="blue", s=30, edgecolor="k", alpha=0.7)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("P(Exceed Threshold)")
    axes[0].set_aspect("equal")
    plt.colorbar(im1, ax=axes[0], label="Probability")

    # Expected cost map
    cost_grid = results["total_expected_cost"].reshape((ny, nx))
    im2 = axes[1].contourf(x_grid, y_grid, cost_grid, levels=15, cmap="YlOrRd")
    axes[1].scatter(x, y, c="blue", s=30, edgecolor="k", alpha=0.7)
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].set_title("Expected Cost")
    axes[1].set_aspect("equal")
    plt.colorbar(im2, ax=axes[1], label="Cost ($)")

    # Optimal decision map
    decision_numeric = (
        (results["optimal_decision"] == "positive").astype(int).reshape((ny, nx))
    )
    im3 = axes[2].contourf(
        x_grid,
        y_grid,
        decision_numeric,
        levels=[0, 0.5, 1],
        colors=["lightgreen", "salmon"],
        alpha=0.7,
    )
    axes[2].scatter(x, y, c="blue", s=30, edgecolor="k", alpha=0.7)
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")
    axes[2].set_title("Optimal Decision\n(Red=Remediate, Green=No Action)")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("example_workflow_03_risk.png", dpi=150, bbox_inches="tight")
    logger.info(" Saved example_workflow_03_risk.png")
    plt.close()


def main():
def main():
    logger.info("GEOSTATS UNCERTAINTY QUANTIFICATION EXAMPLES")

    example_1_bootstrap_confidence()
    example_2_probability_map()
    example_3_risk_assessment()

    logger.info("ALL EXAMPLES COMPLETE!")
    logger.info("\nFiles created:")
    logger.info(" - example_workflow_03_bootstrap.png")
    logger.info(" - example_workflow_03_probability.png")
    logger.info(" - example_workflow_03_risk.png")


if __name__ == "__main__":
if __name__ == "__main__":
