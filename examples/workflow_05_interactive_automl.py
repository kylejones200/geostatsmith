"""
Example Workflow: Interactive Visualization & AutoML
=====================================================

Demonstrates Phase 2 interactive and automatic features.

Shows:
1. Interactive variogram plots (Plotly)
2. Interactive prediction maps
3. AutoML - automatic model selection
4. One-function workflows

Author: geostats development team
Date: January 2026
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
 from geostats.algorithms.fitting import fit_variogram
except ImportError:
 logger.info("Please install geostats: pip install -e .")
 exit(1)

# Check for optional dependencies
try:
    from geostats.visualization import (
        interactive_prediction_map,
        interactive_uncertainty_map,
    )
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info(" Plotly not available. Install with: pip install plotly")

def example_1_interactive_variogram():
    if not PLOTLY_AVAILABLE:
        return

    logger.info("\n" + "="*60)
    logger.info("Example 1: Interactive Variogram")
    logger.info("="*60)

    # Create data
    np.random.seed(42)
    x = np.random.uniform(0, 100, 100)
    y = np.random.uniform(0, 100, 100)
    z = 50 + 0.3*x + 10*np.sin(x/20) + np.random.normal(0, 3, 100)

    # Fit variogram
    lags, gamma = experimental_variogram(x, y, z)
    model = fit_variogram(lags, gamma, model_type='spherical')

    # Create interactive plot
    logger.info("\nCreating interactive variogram plot...")
    fig = interactive_variogram(x, y, z, fitted_model=model)

    # Save to HTML
    fig.write_html('interactive_variogram.html')
    logger.info(" Saved to: interactive_variogram.html")
    logger.info(" Open in browser to explore!")

def example_2_interactive_prediction_map():
    if not PLOTLY_AVAILABLE:
        return

    logger.info("\n" + "="*60)
    logger.info("Example 2: Interactive Prediction Map")
    logger.info("="*60)

    # Data
    np.random.seed(42)
    x = np.random.uniform(0, 100, 50)
    y = np.random.uniform(0, 100, 50)
    z = 50 + 0.3*x + 0.2*y + np.random.normal(0, 3, 50)

    # Auto interpolate
    x_grid = np.linspace(0, 100, 50)
    y_grid = np.linspace(0, 100, 50)
    x_2d, y_2d = np.meshgrid(x_grid, y_grid)

    results = auto_interpolate(
        x, y, z,
        x_2d.ravel(), y_2d.ravel(),
        verbose=False
    )

z_grid = results['predictions'].reshape(x_2d.shape)

# Interactive map
logger.info("\nCreating interactive prediction map...")
fig = interactive_prediction_map(
    x_grid, y_grid, z_grid,
    samples=(x, y, z),
    colorscale='Viridis'
)

fig.write_html('interactive_map.html')
logger.info(" Saved to: interactive_map.html")

def example_3_auto_variogram():
    logger.info("Example 3: Automatic Variogram Selection")

    # Data
    np.random.seed(42)
    x = np.random.uniform(0, 100, 80)
    y = np.random.uniform(0, 100, 80)
    z = 50 + 0.3*x + np.random.normal(0, 3, 80)

    # Auto select best variogram model
    logger.info("\nTrying multiple variogram models...")
    model = auto_variogram(x, y, z, verbose=True)

    logger.info(f"Best model selected and ready to use!")

def example_4_auto_interpolate():
    logger.info("\n" + "="*60)
    logger.info("Example 4: Automatic Interpolation (One Function!)")
    logger.info("="*60)

    # Data
    np.random.seed(42)
    x = np.random.uniform(0, 100, 60)
    y = np.random.uniform(0, 100, 60)
    z = 50 + 0.3*x + 0.2*y + np.random.normal(0, 3, 60)

    # Prediction locations
    x_pred = np.linspace(0, 100, 100)
    y_pred = np.linspace(50, 50, 100)  # Transect at y=50

    # ONE FUNCTION DOES EVERYTHING!
    logger.info("\nAutomatic interpolation (fits model, selects method, predicts)...")
    results = auto_interpolate(
        x, y, z,
        x_pred, y_pred,
        verbose=True
    )

    logger.info(f"Complete! Got {len(results['predictions'])} predictions")
    logger.info(f" Best method: {results['best_method']}")
    logger.info(f" CV RMSE: {results['cv_rmse']:.3f}")

def main():
    logger.info("\n" + "="*70)
    logger.info("GEOSTATS INTERACTIVE & AUTOML EXAMPLES")
    logger.info("="*70)

    if PLOTLY_AVAILABLE:
        example_1_interactive_variogram()
        example_2_interactive_prediction_map()

    example_3_auto_variogram()
    example_4_auto_interpolate()

    logger.info("\n" + "="*70)
    logger.info("ALL EXAMPLES COMPLETE!")
    logger.info("="*70)

    if PLOTLY_AVAILABLE:
        logger.info(" • interactive_variogram.html - Open in browser!")
        logger.info(" • interactive_map.html - Hover for values!")

    logger.info("\nKey Features:")
    logger.info(" • Auto-select best variogram model")
    logger.info(" • One-function automatic interpolation")
    logger.info(" • Interactive plots (if plotly installed)")

if __name__ == '__main__':
    main()
