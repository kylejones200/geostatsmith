"""
    DEMO 1: Gold Rush Alaska - Complete Exploration Workflow
========================================================

This demo shows a realistic gold exploration workflow using real Alaska data.
Demonstrates: Multiple kriging methods, uncertainty, visualization, reporting

Data: Alaska Geochemical Database (AGDB4)
Target: Gold (Au) in the Fairbanks District
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

# GeoStats imports - showcasing variety!
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.algorithms.lognormal_kriging import LognormalKriging
from geostats.algorithms.indicator_kriging import IndicatorKriging
from geostats.algorithms.variogram import experimental_variogram, experimental_variogram_directional
from geostats.algorithms.fitting import fit_variogram_model
from geostats.models import SphericalModel, ExponentialModel, GaussianModel
from geostats.uncertainty import bootstrap_uncertainty, probability_map
from geostats.validation import cross_validation
from geostats.diagnostics import comprehensive_validation
from geostats.optimization import optimal_sampling_design, infill_sampling
import logging

logger = logging.getLogger(__name__)

# Performance features - skip if not available
try:
    from geostats.performance import parallel_kriging
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

# Interactive viz - skip if not available
PLOTLY_AVAILABLE = False # Not yet implemented

logger.info(" GOLD RUSH ALASKA - COMPLETE EXPLORATION WORKFLOW")

# ==============================================================================
# STEP 1: Load Real Alaska Gold Data
# ==============================================================================

def load_fairbanks_gold_data(agdb_path):
    logger.info("Loading Alaska gold data...")

    agdb_path = Path(agdb_path)

    # Load location data
    geol = pd.read_csv(agdb_path / 'Geol_DeDuped.txt', encoding='latin-1', low_memory=False)
    logger.info(f" Total samples in database: {len(geol):,}")

    # Filter to Alaska geographic bounds
    geol = geol[(geol['LATITUDE'] >= 51) & (geol['LATITUDE'] <= 72) &
                (geol['LONGITUDE'] >= -180) & (geol['LONGITUDE'] <= -130)]

    # Load gold chemistry
    chem = pd.read_csv(agdb_path / 'Chem_A_Br.txt', encoding='latin-1', low_memory=False)

    # Filter for gold - PARAMETER column contains method info like "Au_ppm_ES_SQ"
    au_chem = chem[chem['PARAMETER'].str.contains('Au_', case=False, na=False)].copy()
    au_chem = au_chem[['AGDB_ID', 'DATA_VALUE']].copy()
    au_chem = au_chem.rename(columns={'DATA_VALUE': 'Au'})
    logger.info(f" Gold analyses: {len(au_chem):,}")

    # Merge
    data = geol.merge(au_chem, on='AGDB_ID', how='inner')

    # Filter for valid coordinates and positive values
    data = data.dropna(subset=['LATITUDE', 'LONGITUDE', 'Au'])
    data = data[data['Au'] > 0] # Remove non-detects

    # Focus on Fairbanks area (rich gold district!)
    # Fairbanks: ~64.5°N to 65.5°N, -148°W to -146°W
    fairbanks = data[
        (data['LATITUDE'] > 64.0) & (data['LATITUDE'] < 66.0) &
        (data['LONGITUDE'] > -149.0) & (data['LONGITUDE'] < -145.0)
    ].copy()

    # Or use district name
    if 'DISTRICT_NAME' in fairbanks.columns:
        district_data = fairbanks[
            fairbanks['DISTRICT_NAME'].str.contains('Fairbanks', case=False, na=False)
        ]
    if len(district_data) > 100:
        pass

    logger.info(f" Fairbanks district samples: {len(fairbanks):,}")

    # Extract arrays
    x = fairbanks['LONGITUDE'].values
    y = fairbanks['LATITUDE'].values
    au = fairbanks['Au'].values # ppm

    logger.info(f"Gold Statistics:")
    logger.info(f" Mean: {au.mean():.3f} ppm")
    logger.info(f" Median: {np.median(au):.3f} ppm")
    logger.info(f" Max: {au.max():.3f} ppm")
    logger.info(f" >0.1 ppm: {(au > 0.1).sum()} samples ({(au > 0.1).sum()/len(au)*100:.1f}%)")
    logger.info(f" >1.0 ppm: {(au > 1.0).sum()} samples (economic grade!)")

    return x, y, au, fairbanks

# ==============================================================================
# STEP 2: Advanced Variogram Analysis (Anisotropy Detection)
# ==============================================================================

def analyze_variogram_anisotropy(x, y, z):
    logger.info("Directional Variogram Analysis (Anisotropy)...")

    # Log-transform for stationarity
    z_log = np.log10(z + 0.001)

    # Check 4 directions
    directions = [0, 45, 90, 135] # N-S, NE-SW, E-W, NW-SE

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # Remove top and right spines
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes = axes.flatten()
 
    models = {}
    for i, angle in enumerate(directions):
        lags, gamma = experimental_variogram_directional(
            x, y, z_log,
            angle=angle,
            tolerance=22.5,
            n_lags=12
        )

        # Fit model
        model = SphericalModel()
        model = fit_variogram_model(model, lags, gamma)
        models[angle] = model

        # Plot
        axes[i].plot(lags, gamma, 'o', label='Experimental', markersize=6)
        lag_smooth = np.linspace(0, lags.max(), 100)

        # Plot fitted model
        gamma_smooth = model(lag_smooth)
        axes[i].plot(lag_smooth, gamma_smooth, '-', label='Model', linewidth=2)
        
        axes[i].set_title(f'Direction: {angle}° (Range: {model._parameters["range"]:.3f})')
        axes[i].set_xlabel('Distance (degrees)')
        axes[i].set_ylabel('Semivariance')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('alaska_gold_anisotropy.png', dpi=150)
    logger.info(" Saved: alaska_gold_anisotropy.png")

    # Check if anisotropic
    ranges = [models[a]._parameters['range'] for a in directions]
    if max(ranges) / min(ranges) > 1.5:
        logger.info(f" Consider using anisotropic kriging for better results")
    else:
        pass

 return models[0] # Return default model

# ==============================================================================
# STEP 3: Compare Multiple Kriging Methods
# ==============================================================================

def compare_kriging_methods(x, y, au, model):
 logger.info("Comparing Kriging Methods...")

 # Create prediction grid
 x_min, x_max = x.min(), x.max()
 # Remove top and right spines
 
 y_min, y_max = y.min(), y.max()
 # Remove top and right spines
 
 x_grid = np.linspace(x_min, x_max, 80)
 y_grid = np.linspace(y_min, y_max, 80)
 X, Y = np.meshgrid(x_grid, y_grid)
 x_pred = X.flatten()
 y_pred = Y.flatten()

 results = {}

 # Log-transform for some methods
 au_log = np.log10(au + 0.001)

 # Method 1: Ordinary Kriging (standard)
 logger.info("Ordinary Kriging (log-transformed)...")
 t0 = time.time()
 ok = OrdinaryKriging(x, y, au_log, variogram_model=model)
 z_ok, var_ok = ok.predict(x_pred, y_pred, return_variance=True)
 z_ok = 10**z_ok - 0.001 # Back-transform
 t_ok = time.time() - t0
 results['Ordinary'] = {'pred': z_ok.reshape(X.shape), 'time': t_ok}
 logger.info(f" Time: {t_ok:.2f}s")

 # Method 2: Lognormal Kriging (handles skewness)
 logger.info("Lognormal Kriging (handles skewness)...")
 t0 = time.time()
 lk = LognormalKriging(x, y, au, variogram_model=model)
 z_lk, var_lk = lk.predict(x_pred, y_pred, return_variance=True)
 t_lk = time.time() - t0
 results['Lognormal'] = {'pred': z_lk.reshape(X.shape), 'time': t_lk}
 logger.info(f" Time: {t_lk:.2f}s")

 # Method 3: Indicator Kriging (probability mapping)
 logger.info("Indicator Kriging (probability >0.1 ppm)...")
 t0 = time.time()
 threshold = 0.1 # Economic interest threshold
 ik = IndicatorKriging(x, y, au, threshold=threshold, variogram_model=model)
 z_ik = ik.predict(x_pred, y_pred)
 t_ik = time.time() - t0
 results['Indicator'] = {'pred': z_ik.reshape(X.shape), 'time': t_ik}
 logger.info(f" Time: {t_ik:.2f}s")
 logger.info(f" High probability areas: {(z_ik > 0.7).sum()/len(z_ik)*100:.1f}%")

 # Visualize comparison
 fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 # Remove top and right spines
 
 # Ordinary Kriging
 im1 = axes[0].contourf(X, Y, results['Ordinary']['pred'], levels=20, cmap='YlOrRd')
 axes[0].scatter(x, y, c='k', s=2, alpha=0.3)
 # Remove top and right spines
 # Remove top and right spines
 
 axes[0].set_title('Ordinary Kriging\n(Log-transformed)')
 # Remove top and right spines
 axes[0].set_title('Ordinary Kriging\n(Log-transformed)')
 plt.colorbar(im1, ax=axes[0], label='Au (ppm)')
 # Remove top and right spines
 axes[0].set_title('Ordinary Kriging\n(Log-transformed)')

 # Lognormal Kriging
 im2 = axes[1].contourf(X, Y, results['Lognormal']['pred'], levels=20, cmap='YlOrRd')
 axes[1].scatter(x, y, c='k', s=2, alpha=0.3)
 # Remove top and right spines
 # Remove top and right spines
 
 axes[1].set_title('Lognormal Kriging\n(Bias-corrected)')
 # Remove top and right spines
 axes[1].set_title('Lognormal Kriging\n(Bias-corrected)')
 plt.colorbar(im2, ax=axes[1], label='Au (ppm)')
 # Remove top and right spines
 axes[1].set_title('Lognormal Kriging\n(Bias-corrected)')

 # Indicator Kriging
 im3 = axes[2].contourf(X, Y, results['Indicator']['pred'], levels=20, cmap='RdYlGn_r')
 axes[2].scatter(x, y, c='k', s=2, alpha=0.3)
 # Remove top and right spines
 # Remove top and right spines
 
 axes[2].set_title('Indicator Kriging\nP(Au > 0.1 ppm)')
 # Remove top and right spines
 axes[2].set_title('Indicator Kriging\nP(Au > 0.1 ppm)')
 plt.colorbar(im3, ax=axes[2], label='Probability')
 # Remove top and right spines
 axes[2].set_title('Indicator Kriging\nP(Au > 0.1 ppm)')

 for ax in axes:
     continue
 ax.set_ylabel('Latitude')

 plt.tight_layout()
 plt.savefig('alaska_gold_methods_comparison.png', dpi=150)
 logger.info("Saved: alaska_gold_methods_comparison.png")

 return results, X, Y

# ==============================================================================
# STEP 4: Uncertainty Quantification (Bootstrap + Simulation)
# ==============================================================================

def quantify_uncertainty(x, y, au, model, X, Y):
 logger.info("Uncertainty Quantification...")

 au_log = np.log10(au + 0.001)

 # Method 1: Bootstrap confidence intervals
 logger.info("Bootstrap (100 iterations)...")
 t0 = time.time()
 ci_lower, ci_upper = bootstrap_uncertainty()
 x, y, au_log,
 X[0, :], Y[:, 0],
 model,
 n_bootstrap=100,
 confidence_level=0.95
 )
 logger.info(f" Time: {time.time() - t0:.1f}s")

 # Back-transform
 ci_lower = 10**ci_lower - 0.001
 ci_upper = 10**ci_upper - 0.001

 # Method 2: Kriging variance
 logger.info("Kriging Variance...")
 ok = OrdinaryKriging(x, y, au_log, variogram_model=model)
 z_pred, variance = ok.predict(X.flatten(), Y.flatten(), return_variance=True)
 variance = variance.reshape(X.shape)
 std_dev = np.sqrt(variance)

 # Visualize uncertainty
 fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 # Remove top and right spines
 
 # Confidence interval width
 ci_width = ci_upper - ci_lower
 im1 = axes[0].contourf(X, Y, ci_width, levels=20, cmap='viridis')
 axes[0].set_title('95% CI Width\n(Bootstrap)')
 # Remove top and right spines
 axes[0].set_title('95% CI Width\n(Bootstrap)')
 plt.colorbar(im1, ax=axes[0], label='Width (ppm)')
 # Remove top and right spines
 axes[0].set_title('95% CI Width\n(Bootstrap)')

 # Standard deviation
 im2 = axes[1].contourf(X, Y, std_dev, levels=20, cmap='viridis')
 axes[1].scatter(x, y, c='k', s=1, alpha=0.5, label='Samples')
 # Remove top and right spines
 
 axes[1].set_title('Prediction Std Dev\n(Kriging Variance)')
 # Remove top and right spines
 axes[1].set_title('Prediction Std Dev\n(Kriging Variance)')
 plt.colorbar(im2, ax=axes[1], label='Std Dev')
 # Remove top and right spines
 axes[1].set_title('Prediction Std Dev\n(Kriging Variance)')
 axes[1].legend()
 # Remove top and right spines
 axes[1].legend()

 # Coefficient of variation (relative uncertainty)
 z_pred_reshaped = (10**z_pred.reshape(X.shape) - 0.001)
 cv = (std_dev / np.abs(z_pred_reshaped)) * 100
 cv = np.clip(cv, 0, 200) # Cap at 200% for visualization
 im3 = axes[2].contourf(X, Y, cv, levels=20, cmap='RdYlGn_r')
 axes[2].set_title('Coefficient of Variation\n(Relative Uncertainty)')
 # Remove top and right spines
 axes[2].set_title('Coefficient of Variation\n(Relative Uncertainty)')
 plt.colorbar(im3, ax=axes[2], label='CV (%)')
 # Remove top and right spines
 axes[2].set_title('Coefficient of Variation\n(Relative Uncertainty)')

 for ax in axes:
     continue
 ax.set_ylabel('Latitude')

 plt.tight_layout()
 plt.savefig('alaska_gold_uncertainty.png', dpi=150)
 logger.info("Saved: alaska_gold_uncertainty.png")

 return ci_lower, ci_upper, variance

# ==============================================================================
# STEP 5: Optimal Sampling (Where to collect more samples?)
# ==============================================================================

def design_infill_sampling(x, y, au, model, X, Y):
 logger.info("Optimal Infill Sampling Design...")

 # Find high-uncertainty areas that need more samples
 au_log = np.log10(au + 0.001)
 ok = OrdinaryKriging(x, y, au_log, variogram_model=model)
 _, variance = ok.predict(X.flatten(), Y.flatten(), return_variance=True)
 variance = variance.reshape(X.shape)

 # Identify high-variance areas
 high_var_threshold = np.percentile(variance, 75)
 high_var_mask = variance > high_var_threshold

 # Suggest 10 new sample locations
 n_new = 10
 bounds = (x.min(), x.max(), y.min(), y.max())

 logger.info(f"Suggesting {n_new} optimal new sample locations...")
 new_locations = infill_sampling()
 x, y, au_log,
 variogram_model=model,
 n_new_samples=n_new,
 bounds=bounds
 )

 # Visualize
 fig, ax = plt.subplots(figsize=(10, 8))
 # Remove top and right spines
 
 # Variance map
 im = ax.contourf(X, Y, variance, levels=20, cmap='YlOrRd')

 # Existing samples
 ax.scatter(x, y, c='blue', s=20, alpha=0.5, label='Existing samples',
 edgecolors='k', linewidths=0.5)

 # Proposed new samples
 ax.scatter(new_locations[:, 0], new_locations[:, 1],
 c='lime', s=200, marker='*',
 edgecolors='darkgreen', linewidths=2,
 label=f'Proposed new samples (n={n_new})', zorder=10)

 # Add numbers to new samples
 for i, (nx, ny) in enumerate(new_locations, 1):
     continue
 ha='center', va='center')

 ax.set_title('Optimal Infill Sampling Design\n(targeting high-uncertainty areas)')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 ax.legend(loc='upper right')
 plt.colorbar(im, ax=ax, label='Kriging Variance')
 # Remove top and right spines
 
 plt.tight_layout()
 plt.savefig('alaska_gold_sampling_design.png', dpi=150)
 logger.info("Saved: alaska_gold_sampling_design.png")
 logger.info(f"Cost Savings Estimate:")
 logger.info(f" Random sampling would need ~{n_new*3} samples")
 logger.info(f" Optimal design needs only {n_new} samples")
 logger.info(f" Savings: ~{(n_new*2)/1000:.0f}k USD (assuming $500/sample)")

 return new_locations

# ==============================================================================
# STEP 6: Performance Showcase (Parallel Processing)
# ==============================================================================

def performance_comparison(x, y, au, model, X, Y):
 logger.info("Performance Showcase...")

 au_log = np.log10(au + 0.001)
 x_pred = X.flatten()
 y_pred = Y.flatten()

 # Standard kriging (single core)
 logger.info("Standard Kriging (single core)...")
 t0 = time.time()
 ok = OrdinaryKriging(x, y, au_log, variogram_model=model)
 z_standard = ok.predict(x_pred, y_pred)
 t_standard = time.time() - t0
 logger.info(f" Time: {t_standard:.2f}s")

 # Parallel kriging (all cores)
 logger.info("Parallel Kriging (all cores)...")
 t0 = time.time()
 z_parallel, var_parallel = parallel_kriging()
 x, y, au_log, x_pred, y_pred,
 model, n_jobs=-1
 )
 t_parallel = time.time() - t0
 logger.info(f" Time: {t_parallel:.2f}s")
 logger.info(f" SPEEDUP: {t_standard/t_parallel:.1f}x faster!")

 return t_standard / t_parallel

# ==============================================================================
# STEP 7: Cross-Validation & Quality Assessment
# ==============================================================================

def validate_predictions(x, y, au, model):
 logger.info("Model Validation & Quality Assessment...")

 au_log = np.log10(au + 0.001)

 # Validation suite
 results = comprehensive_validation(x, y, au_log, model)

 logger.info(f"Validation Metrics:")
 logger.info(f" RMSE: {results['cv_rmse']:.4f}")
 logger.info(f" MAE: {results['cv_mae']:.4f}")
 logger.info(f" R²: {results['cv_r2']:.4f}")
 logger.info(f" Overall Quality Score: {results['overall_score']:.0f}/100")

 if results['overall_score'] > 80:
 elif results['overall_score'] > 60:
 else:
    pass

 return results

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    pass

 if not Path(AGDB_PATH).exists():
     continue
 logger.info("Please download from: https://doi.org/10.5066/F7445KBJ")
 exit(1)

 try:
     pass
 x, y, au, data_df = load_fairbanks_gold_data(AGDB_PATH)

 # Variogram analysis
 model = analyze_variogram_anisotropy(x, y, au)

 # Compare kriging methods
 results, X, Y = compare_kriging_methods(x, y, au, model)

 # Uncertainty quantification
 ci_lower, ci_upper, variance = quantify_uncertainty(x, y, au, model, X, Y)

 # Optimal sampling design
 new_locations = design_infill_sampling(x, y, au, model, X, Y)

 # Performance comparison
 speedup = performance_comparison(x, y, au, model, X, Y)

 # Validation
 validation = validate_predictions(x, y, au, model)

 logger.info(" COMPLETE! Generated 4 Publication-Quality Figures:")
 logger.info(" 1. alaska_gold_anisotropy.png - Directional variograms")
 logger.info(" 2. alaska_gold_methods_comparison.png - 3 kriging methods")
 logger.info(" 3. alaska_gold_uncertainty.png - Uncertainty quantification")
 logger.info(" 4. alaska_gold_sampling_design.png - Optimal sampling locations")
 logger.info("This demo showcased:")
 logger.info(" Multiple kriging methods (Ordinary, Lognormal, Indicator)")
 logger.info(" Anisotropy detection")
 logger.info(" Uncertainty quantification (Bootstrap, Variance)")
 logger.info(" Optimal sampling design")
 logger.info(f" Performance optimization ({speedup:.1f}x speedup!)")
 logger.info(" Validation")
 logger.info("GeoStats + Real Alaska Data = Professional Results!")

 except Exception as e:
     pass
 logger.exception("Error in gold exploration workflow")
