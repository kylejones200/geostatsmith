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

try:
    from geostats.automl import auto_variogram, auto_interpolate
    from geostats.algorithms.variogram import experimental_variogram
    from geostats.algorithms.fitting import fit_variogram
except ImportError:
    print("Please install geostats: pip install -e .")
    exit(1)

# Check for optional dependencies
try:
    from geostats.interactive import (
        interactive_variogram,
        interactive_prediction_map,
        interactive_uncertainty_map,
    )
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠ Plotly not available. Install with: pip install plotly")


def example_1_interactive_variogram():
    """Example 1: Interactive variogram plot."""
    if not PLOTLY_AVAILABLE:
        print("\n⚠ Skipping interactive examples (plotly not installed)")
        return
    
    print("\n" + "="*60)
    print("Example 1: Interactive Variogram")
    print("="*60)
    
    # Create data
    np.random.seed(42)
    x = np.random.uniform(0, 100, 100)
    y = np.random.uniform(0, 100, 100)
    z = 50 + 0.3*x + 10*np.sin(x/20) + np.random.normal(0, 3, 100)
    
    # Fit variogram
    lags, gamma = experimental_variogram(x, y, z)
    model = fit_variogram(lags, gamma, model_type='spherical')
    
    # Create interactive plot
    print("\nCreating interactive variogram plot...")
    fig = interactive_variogram(x, y, z, fitted_model=model)
    
    # Save to HTML
    fig.write_html('interactive_variogram.html')
    print("✓ Saved to: interactive_variogram.html")
    print("  Open in browser to explore!")


def example_2_interactive_prediction_map():
    """Example 2: Interactive prediction map."""
    if not PLOTLY_AVAILABLE:
        return
    
    print("\n" + "="*60)
    print("Example 2: Interactive Prediction Map")
    print("="*60)
    
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
    print("\nCreating interactive prediction map...")
    fig = interactive_prediction_map(
        x_grid, y_grid, z_grid,
        samples=(x, y, z),
        colorscale='Viridis'
    )
    
    fig.write_html('interactive_map.html')
    print("✓ Saved to: interactive_map.html")


def example_3_auto_variogram():
    """Example 3: Automatic variogram selection."""
    print("\n" + "="*60)
    print("Example 3: Automatic Variogram Selection")
    print("="*60)
    
    # Data
    np.random.seed(42)
    x = np.random.uniform(0, 100, 80)
    y = np.random.uniform(0, 100, 80)
    z = 50 + 0.3*x + np.random.normal(0, 3, 80)
    
    # Auto select best variogram model
    print("\nTrying multiple variogram models...")
    model = auto_variogram(x, y, z, verbose=True)
    
    print(f"\n✓ Best model selected and ready to use!")


def example_4_auto_interpolate():
    """Example 4: One-function automatic interpolation."""
    print("\n" + "="*60)
    print("Example 4: Automatic Interpolation (One Function!)")
    print("="*60)
    
    # Data
    np.random.seed(42)
    x = np.random.uniform(0, 100, 60)
    y = np.random.uniform(0, 100, 60)
    z = 50 + 0.3*x + 0.2*y + np.random.normal(0, 3, 60)
    
    # Prediction locations
    x_pred = np.linspace(0, 100, 100)
    y_pred = np.linspace(50, 50, 100)  # Transect at y=50
    
    # ONE FUNCTION DOES EVERYTHING!
    print("\nAutomatic interpolation (fits model, selects method, predicts)...")
    results = auto_interpolate(
        x, y, z,
        x_pred, y_pred,
        verbose=True
    )
    
    print(f"\n✓ Complete! Got {len(results['predictions'])} predictions")
    print(f"✓ Best method: {results['best_method']}")
    print(f"✓ CV RMSE: {results['cv_rmse']:.3f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GEOSTATS INTERACTIVE & AUTOML EXAMPLES")
    print("="*70)
    
    if PLOTLY_AVAILABLE:
        example_1_interactive_variogram()
        example_2_interactive_prediction_map()
    
    example_3_auto_variogram()
    example_4_auto_interpolate()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)
    
    if PLOTLY_AVAILABLE:
        print("\nInteractive HTML files created:")
        print("  • interactive_variogram.html - Open in browser!")
        print("  • interactive_map.html - Hover for values!")
    
    print("\nKey Features:")
    print("  • Auto-select best variogram model")
    print("  • One-function automatic interpolation")
    print("  • Interactive plots (if plotly installed)")
    print("\n")


if __name__ == '__main__':
    main()
