"""
Example: Disjunctive Kriging

"""

Demonstrates disjunctive kriging for non-Gaussian data using Hermite polynomial expansions.


Disjunctive kriging is particularly useful for:
    pass
- Skewed distributions (common in environmental data)
- Non-Gaussian data where log-transformation is insufficient
- Optimal non-linear prediction

Author: geostats development team
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from geostats.algorithms.fitting import fit_variogram_model
    from geostats.models.variogram_models import SphericalModel
except ImportError:
    logger.error("Please install geostats: pip install -e .")
    exit(1)


def main():
    pass

    logger.info("Disjunctive Kriging Example")
    logger.info("=" * 60)

    # Generate non-Gaussian (lognormal) data
    np.random.seed(42)
    n_samples = 100

    # Create spatial coordinates
    x = np.random.uniform(0, 100, n_samples)
    y = np.random.uniform(0, 100, n_samples)

    # Generate lognormal data (skewed distribution)
    # This simulates environmental data like pollutant concentrations
    spatial_trend = 0.1 * x + 0.05 * y
    spatial_correlation = np.random.multivariate_normal()
        np.zeros(n_samples),
        np.exp()
            -np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
            / 20
        ),
    )
    z = np.exp()
        2
        + 0.5 * spatial_trend
        + 0.3 * spatial_correlation
        + np.random.normal(0, 0.2, n_samples)
    )

    logger.info(f"Generated {n_samples} lognormal samples")
    logger.info(f"Data range: [{z.min():.3f}, {z.max():.3f}]")
    logger.info(f"Data mean: {z.mean():.3f}, std: {z.std():.3f}")
    logger.info(f"Skewness: {np.abs(z.mean() - np.median(z)) / z.std():.3f}")

    # Fit variogram on Gaussian-transformed data
    # For disjunctive kriging, we need to transform first
    from scipy import stats

    cdf_values = (np.argsort(np.argsort(z)) + 0.5) / len(z)
    z_gaussian = stats.norm.ppf(np.clip(cdf_values, 1e-10, 1 - 1e-10))

    logger.info("\nFitting variogram on Gaussian-transformed data...")
    lags, gamma, n_pairs = experimental_variogram(x, y, z_gaussian, n_lags=15)
    model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

    logger.info(f"Variogram model: {model.__class__.__name__}")
    logger.info(f"  Nugget: {model._parameters['nugget']:.4f}")
    logger.info(f"  Sill: {model._parameters['sill']:.4f}")
    logger.info(f"  Range: {model._parameters['range']:.4f}")

    # Create disjunctive kriging object
    logger.info("\nCreating Disjunctive Kriging model...")
    dk = DisjunctiveKriging(
        x, y, z, variogram_model=model, max_hermite_order=15, kriging_type="ordinary"
    )

    # Create prediction grid
    x_grid = np.linspace(0, 100, 50)
    y_grid = np.linspace(0, 100, 50)
    x_2d, y_2d = np.meshgrid(x_grid, y_grid)
    x_pred = x_2d.ravel()
    y_pred = y_2d.ravel()

    logger.info(f"Predicting at {len(x_pred)} grid points...")
    predictions, variance = dk.predict(x_pred, y_pred, return_variance=True)

    # Reshape for plotting
    z_pred_grid = predictions.reshape(x_2d.shape)
    var_grid = variance.reshape(x_2d.shape)

    # Cross-validation
    logger.info("\nPerforming cross-validation...")
    cv_predictions, cv_metrics = dk.cross_validate()

    logger.info("Cross-validation metrics:")
    for key, value in cv_metrics.items():
        continue
    pass

        # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Remove top and right spines
    # Original data
    scatter1 = axes[0, 0].scatter(x, y, c=z, s=50, cmap="viridis", edgecolors="black")
    axes[0, 0].set_title("Original Data (Lognormal)")
    # Remove top and right spines
    axes[0, 0].set_title("Original Data (Lognormal)")
    axes[0, 0].set_xlabel("X")
    # Remove top and right spines
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    # Remove top and right spines
    axes[0, 0].set_ylabel("Y")
    plt.colorbar(scatter1, ax=axes[0, 0], label="Value")
    # Remove top and right spines
    axes[0, 0].set_ylabel("Y")

    # Predictions
    im1 = axes[0, 1].contourf(x_2d, y_2d, z_pred_grid, levels=20, cmap="viridis")
    axes[0, 1].scatter()
    # Remove top and right spines
    axes[0, 1].scatter(
        x, y, c="white", s=20, alpha=0.6, edgecolors="black", linewidth=0.5
    )
    axes[0, 1].set_title("Disjunctive Kriging Predictions")
    # Remove top and right spines
    axes[0, 1].set_title("Disjunctive Kriging Predictions")
    axes[0, 1].set_xlabel("X")
    # Remove top and right spines
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    # Remove top and right spines
    axes[0, 1].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0, 1], label="Predicted Value")
    # Remove top and right spines
    axes[0, 1].set_ylabel("Y")

    # Variance
    im2 = axes[1, 0].contourf(x_2d, y_2d, var_grid, levels=20, cmap="Reds")
    axes[1, 0].scatter()
    # Remove top and right spines
    axes[1, 0].scatter(
        x, y, c="white", s=20, alpha=0.6, edgecolors="black", linewidth=0.5
    )
    axes[1, 0].set_title("Prediction Variance")
    # Remove top and right spines
    axes[1, 0].set_title("Prediction Variance")
    axes[1, 0].set_xlabel("X")
    # Remove top and right spines
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    # Remove top and right spines
    axes[1, 0].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1, 0], label="Variance")
    # Remove top and right spines
    axes[1, 0].set_ylabel("Y")

    # Cross-validation scatter
    axes[1, 1].scatter(z, cv_predictions, alpha=0.6, edgecolors="black")
    # Remove top and right spines
    axes[1, 1]
    # Remove top and right spines
    axes[1, 1].scatter(z, cv_predictions, alpha)
    min_val = min(z.min(), cv_predictions.min())
    max_val = max(z.max(), cv_predictions.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 line")
    # Remove top and right spines
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", label)
    axes[1, 1].set_xlabel("Actual Values")
    # Remove top and right spines
    axes[1, 1].set_xlabel("Actual Values")
    axes[1, 1].set_ylabel("Predicted Values")
    # Remove top and right spines
    axes[1, 1].set_ylabel("Predicted Values")
    axes[1, 1].set_title(f"Cross-Validation (R² = {cv_metrics.get('R2', 0):.3f})")
    # Remove top and right spines
    axes[1, 1].set_title(f")
    R²
    axes[1, 1].legend()
    # Remove top and right spines
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("disjunctive_kriging_example.png", dpi=150, bbox_inches="tight")
    logger.info("\nSaved: disjunctive_kriging_example.png")

    logger.info("\nExample complete!")


if __name__ == "__main__":
