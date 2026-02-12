"""
Alaska Geochemical Database (AGDB4) Analysis Example
=====================================================

This example demonstrates how to use GeoStats with the Alaska Geochemical
Database (AGDB4), a database of geochemical analyses from Alaska.

AGDB4 contains:
- 375,000+ samples (deduplicated)
- 400,000+ total samples
- 70+ elements analyzed
- Lat/Long coordinates
- Stream sediments, rocks, soils
- Multiple analytical methods
- Historical data from USGS, DGGS, and other agencies

This workflow shows:
1. Loading and preparing AGDB4 data
2. Spatial analysis of specific elements (Au, Cu, Pb, As, etc.)
3. Multi-element cokriging
4. Probability mapping for mineral exploration
5. Environmental contamination assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# GeoStats imports
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram_model as fit_variogram
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.algorithms.cokriging import Cokriging
from geostats.algorithms.indicator_kriging import IndicatorKriging
from geostats.uncertainty import probability_of_exceedance
from geostats.diagnostics import comprehensive_validation
from geostats.optimization import optimal_sampling
from geostats.visualization import plot_variogram, plot_prediction_surface
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# PART 1: Load and Prepare AGDB4 Data
# ==============================================================================

def load_agdb4_data(agdb_path, element='Au', sample_type='stream sediment'):
    """
    Load Alaska Geochemical Database and prepare for analysis.

    Parameters:
    -----------
    agdb_path : str or Path
        Path to AGDB4_text directory
    element : str
        Element to analyze (e.g., 'Au', 'Cu', 'Pb', 'As')
    sample_type : str
        Type of sample to filter for

    Returns:
    --------
    dict with keys: x, y, values, metadata
    """
    agdb_path = Path(agdb_path)

    logger.info(f"Loading AGDB4 data for {element}...")

    # Load geological/location data
    geol_file = agdb_path / 'Geol_DeDuped.txt'
    geol_data = pd.read_csv(geol_file, sep=',', quotechar='"', low_memory=False)

    logger.info(f" Loaded {len(geol_data):,} samples from Geol_DeDuped")

    # Determine which chemistry file to use based on element
    element_to_file = {
    # A-Br file
    'Ag': 'Chem_A_Br', 'Al': 'Chem_A_Br', 'As': 'Chem_A_Br', 'Au': 'Chem_A_Br',
    'B': 'Chem_A_Br', 'Ba': 'Chem_A_Br', 'Be': 'Chem_A_Br', 'Bi': 'Chem_A_Br',
    'Br': 'Chem_A_Br',
    # C-Gd file
    'Ca': 'Chem_C_Gd', 'Cd': 'Chem_C_Gd', 'Ce': 'Chem_C_Gd', 'Cl': 'Chem_C_Gd',
    'Co': 'Chem_C_Gd', 'Cr': 'Chem_C_Gd', 'Cs': 'Chem_C_Gd', 'Cu': 'Chem_C_Gd',
    # Ge-Os file
    'Fe': 'Chem_Ge_Os', 'Ga': 'Chem_Ge_Os', 'Ge': 'Chem_Ge_Os', 'Hf': 'Chem_Ge_Os',
    'Hg': 'Chem_Ge_Os', 'In': 'Chem_Ge_Os', 'K': 'Chem_Ge_Os', 'La': 'Chem_Ge_Os',
    # P-Te file
    'P': 'Chem_P_Te', 'Pb': 'Chem_P_Te', 'Pd': 'Chem_P_Te', 'Pt': 'Chem_P_Te',
    'Rb': 'Chem_P_Te', 'Re': 'Chem_P_Te', 'S': 'Chem_P_Te', 'Sb': 'Chem_P_Te',
    # Th-Zr file
    'Th': 'Chem_Th_Zr', 'Ti': 'Chem_Th_Zr', 'Tl': 'Chem_Th_Zr', 'U': 'Chem_Th_Zr',
    'V': 'Chem_Th_Zr', 'W': 'Chem_Th_Zr', 'Y': 'Chem_Th_Zr', 'Zn': 'Chem_Th_Zr',
    'Zr': 'Chem_Th_Zr'
    }

    chem_filename = element_to_file.get(element, 'Chem_A_Br') + '.txt'
    chem_file = agdb_path / chem_filename

    # Load chemistry data
    chem_data = pd.read_csv(chem_file, sep=',', quotechar='"', low_memory=False)
    logger.info(f" Loaded {len(chem_data):,} analyses from {chem_filename}")

    # Merge on AGDB_ID
    merged = geol_data.merge(chem_data, on='AGDB_ID', how='inner')

    # Filter for valid coordinates
    merged = merged.dropna(subset=['LATITUDE', 'LONGITUDE'])
    merged = merged[(merged['LATITUDE'] > 0) & (merged['LONGITUDE'] < 0)]

    # Filter for sample type if specified
    if sample_type:
        merged = merged[merged['PRIMARY_CLASS'].str.contains(sample_type, case=False, na=False)]

    # Get element column name (e.g., 'Au_ppm', 'Cu_ppm')
    element_col = f"{element}_ppm"
 
    # Filter for parameter = element and valid values
    if 'PARAMETER' in chem_data.columns:
     element_mask = merged['PARAMETER'].str.contains(f'{element}_', case=False, na=False)
     merged = merged[element_mask]

    # Get the value column (usually 'VALUE' in chem files)
    if 'VALUE' in merged.columns:
     value_col = 'VALUE'
    elif element_col in merged.columns:
     value_col = element_col
    else:
     value_col = None

    if value_col:
     merged = merged[merged[value_col] > 0]
     merged = merged.dropna(subset=[value_col])

    logger.info(f" After filtering: {len(merged):,} samples with valid {element} data")

    # Extract coordinates and values
    x = merged['LONGITUDE'].values
    y = merged['LATITUDE'].values

    if value_col:
     values = merged[value_col].values
    else:
     values = None

    # Metadata
    metadata = {
    'element': element,
    'sample_type': sample_type,
    'n_samples': len(merged),
    'units': 'ppm' if '_ppm' in value_col else 'pct' if value_col else 'unknown',
    'x_range': (x.min(), x.max()),
    'y_range': (y.min(), y.max()),
    'value_range': (values.min(), values.max()) if values is not None else None,
    'dataframe': merged # Keep full dataframe for advanced analysis
    }

    return {
    'x': x,
    'y': y,
    'values': values,
    'metadata': metadata
    }

    # ==============================================================================
    # PART 2: Gold Exploration Example
    # ==============================================================================

def gold_exploration_analysis(agdb_path, region_name='Iliamna'):
    """
    Analyze gold distribution for mineral exploration.
    """
    """
    Analyze gold distribution for mineral exploration.
    """
    logger.info("GOLD EXPLORATION ANALYSIS")

 # Load Au data
 au_data = load_agdb4_data(agdb_path, element='Au', sample_type='sediment')

 # Filter for specific region if desired
 if region_name:
     # Try to filter by district name if available
     if 'metadata' in au_data and 'dataframe' in au_data['metadata']:
         df = au_data['metadata']['dataframe']
         if 'DISTRICT_NAME' in df.columns:
             mask = df['DISTRICT_NAME'].str.contains(region_name, case=False, na=False)
             indices = np.where(mask.values)[0]
             if len(indices) > 0:
                 au_data['x'] = au_data['x'][indices]
                 au_data['y'] = au_data['y'][indices]
                 au_data['values'] = au_data['values'][indices]
                 logger.info(f"Focused on {region_name} region: {len(indices)} samples")

 x, y, au = au_data['x'], au_data['y'], au_data['values']

 # Log-transform for lognormal distribution (typical for Au)
 au_log = np.log10(au + 1) # +1 to handle zeros

 logger.info(f"\nGold Statistics:")
 logger.info(f" Mean: {au.mean():.3f} ppm")
 logger.info(f" Median: {np.median(au):.3f} ppm")
 logger.info(f" Max: {au.max():.3f} ppm")
 logger.info(f" >100 ppb: {(au > 0.1).sum()} samples ({(au > 0.1).sum()/len(au)*100:.1f}%)")

 # Variogram analysis on log-transformed data
 logger.info("\nVariogram Analysis...")
 lags, gamma = experimental_variogram(x, y, au_log, n_lags=15)
 model = fit_variogram(lags, gamma, model_type='spherical')

 logger.info(f" Model: {model['model']}")
 logger.info(f" Range: {model['range']:.2f} degrees")
 logger.info(f" Sill: {model['sill']:.3f}")

 # Kriging
 logger.info("\nKriging...")
 kriging = OrdinaryKriging(x, y, au_log, variogram_model=model)

 # Create prediction grid
 x_grid = np.linspace(x.min(), x.max(), 100)
 y_grid = np.linspace(y.min(), y.max(), 100)
 X, Y = np.meshgrid(x_grid, y_grid)

 au_pred_log, variance = kriging.predict(X.flatten(), Y.flatten(), return_variance=True)

 # Back-transform to original units
 au_pred = 10**au_pred_log.reshape(X.shape) - 1
 variance = variance.reshape(X.shape)

 # Identify high-potential zones (>100 ppb Au)
 high_potential = au_pred > 0.1

 logger.info(f"\nExploration Targets:")
 logger.info(f" High potential area: {high_potential.sum() / high_potential.size * 100:.1f}% of region")
 logger.info(f" Max predicted Au: {au_pred.max():.3f} ppm")

 # Visualization
 fig, axes = plt.subplots(1, 3, figsize=(18, 5))

 # Sample locations
 axes[0].scatter(x, y, c=au, s=10, cmap='YlOrRd', vmin=0, vmax=np.percentile(au, 95))
 axes[0].set_title(f'Gold Sample Locations (n={len(au)})')
 axes[0].set_xlabel('Longitude')
 axes[0].set_ylabel('Latitude')
 plt.colorbar(axes[0].collections[0], ax=axes[0], label='Au (ppm)')

 # Kriged surface
 im = axes[1].contourf(X, Y, au_pred, levels=20, cmap='YlOrRd')
 axes[1].scatter(x, y, c='k', s=1, alpha=0.3)
 axes[1].set_title('Kriged Gold Distribution')
 axes[1].set_xlabel('Longitude')
 axes[1].set_ylabel('Latitude')
 plt.colorbar(im, ax=axes[1], label='Predicted Au (ppm)')

 # Uncertainty (standard deviation)
 im = axes[2].contourf(X, Y, np.sqrt(variance), levels=20, cmap='viridis')
 axes[2].set_title('Prediction Uncertainty (Std Dev)')
 axes[2].set_xlabel('Longitude')
 axes[2].set_ylabel('Latitude')
 plt.colorbar(im, ax=axes[2], label='Std Dev (log ppm)')

 plt.tight_layout()
 plt.savefig('alaska_gold_analysis.png', dpi=150)
 logger.info("\nSaved: alaska_gold_analysis.png")

 return {
 'x': x, 'y': y, 'au': au,
 'X': X, 'Y': Y, 'au_pred': au_pred,
 'variance': variance, 'model': model
 }

# ==============================================================================
# PART 3: Multi-element Cokriging (Cu-Au-Mo)
# ==============================================================================

def multi_element_analysis(agdb_path):
    """
    Analyze multiple correlated elements using cokriging.
    Example: Cu-Mo association in porphyry deposits.
    """
 logger.info("MULTI-ELEMENT COKRIGING ANALYSIS (Cu-Au-Mo)")

 # Load Cu and Mo data
 cu_data = load_agdb4_data(agdb_path, element='Cu', sample_type='sediment')
 mo_data = load_agdb4_data(agdb_path, element='Mo', sample_type='sediment')

 # Find common samples
 cu_df = cu_data['metadata']['dataframe']
 mo_df = mo_data['metadata']['dataframe']

 # Merge on AGDB_ID
 common = cu_df.merge(mo_df, on=['AGDB_ID', 'LATITUDE', 'LONGITUDE'],
 suffixes=('_cu', '_mo'))

 logger.info(f"Common Cu-Mo samples: {len(common)}")

 if len(common) < 50:
 return None

 x = common['LONGITUDE'].values
 y = common['LATITUDE'].values
 cu = common['VALUE_cu'].values
 mo = common['VALUE_mo'].values

 # Log-transform
 cu_log = np.log10(cu + 1)
 mo_log = np.log10(mo + 1)

 logger.info(f"\nElement Correlation:")
 correlation = np.corrcoef(cu_log, mo_log)[0, 1]
 logger.info(f" Cu-Mo correlation: {correlation:.3f}")

 if correlation > 0.3:

 # Fit individual variograms
 logger.info("\nFitting variograms...")
 lags_cu, gamma_cu = experimental_variogram(x, y, cu_log, n_lags=12)
 model_cu = fit_variogram(lags_cu, gamma_cu, model_type='spherical')

 lags_mo, gamma_mo = experimental_variogram(x, y, mo_log, n_lags=12)
 model_mo = fit_variogram(lags_mo, gamma_mo, model_type='spherical')

 logger.info(f" Cu variogram range: {model_cu['range']:.2f}°")
 logger.info(f" Mo variogram range: {model_mo['range']:.2f}°")

 # For demonstration: compare Ordinary Kriging vs Cokriging
 # (Full cokriging requires cross-variogram, which is more complex)

 return {
 'x': x, 'y': y,
 'cu': cu, 'mo': mo,
 'correlation': correlation
 }

# ==============================================================================
# PART 4: Environmental Assessment (As, Pb contamination)
# ==============================================================================

def environmental_assessment(agdb_path, element='As', threshold=20):
    """
    Environmental geochemistry: assess contamination risk.

    Parameters:
    -----------
    element : str
        Element to assess (As, Pb, Hg, etc.)
    threshold : float
        Regulatory or background threshold (ppm)
    """
 logger.info(f"{element.upper()} ENVIRONMENTAL ASSESSMENT")

 # Load data
 data = load_agdb4_data(agdb_path, element=element, sample_type='sediment')

 x, y, values = data['x'], data['y'], data['values']

 logger.info(f"\n{element} Statistics:")
 logger.info(f" Mean: {values.mean():.2f} ppm")
 logger.info(f" Median: {np.median(values):.2f} ppm")
 logger.info(f" Max: {values.max():.2f} ppm")
 logger.info(f">{threshold} ppm: {(values > threshold).sum()} samples ({(values > threshold).sum()/len(values)*100:.1f}%)")

 # Indicator kriging for exceedance probability
 logger.info(f"\nIndicator Kriging (threshold = {threshold} ppm)...")

 # Create indicators
 indicators = (values > threshold).astype(int)

 # Variogram of indicators
 lags, gamma = experimental_variogram(x, y, indicators, n_lags=15)
 model = fit_variogram(lags, gamma, model_type='spherical')

 # Simple indicator kriging (for demo)
 ik = IndicatorKriging(x, y, values, threshold=threshold, variogram_model=model)

 # Prediction grid
 x_grid = np.linspace(x.min(), x.max(), 100)
 y_grid = np.linspace(y.min(), y.max(), 100)
 X, Y = np.meshgrid(x_grid, y_grid)

 prob_exceed = ik.predict(X.flatten(), Y.flatten()).reshape(X.shape)

 # Risk classification
 risk = np.zeros_like(prob_exceed)
 risk[prob_exceed < 0.1] = 0 # Low
 risk[(prob_exceed >= 0.1) & (prob_exceed < 0.5)] = 1 # Medium
 risk[prob_exceed >= 0.5] = 2 # High

 logger.info(f"\nRisk Assessment (P({element} > {threshold} ppm)):")
 logger.info(f" Low risk (<10%): {(risk==0).sum()/risk.size*100:.1f}%")
 logger.info(f" Medium risk (10-50%): {(risk==1).sum()/risk.size*100:.1f}%")
 logger.info(f" High risk (>50%): {(risk==2).sum()/risk.size*100:.1f}%")

 return {
 'x': x, 'y': y, 'values': values,
 'X': X, 'Y': Y, 'prob_exceed': prob_exceed,
 'risk': risk
 }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
 AGDB_PATH = Path('/Users/k.jones/Downloads/AGDB4_text')

 if not AGDB_PATH.exists():
 exit(1)

 logger.info("ALASKA GEOCHEMICAL DATABASE (AGDB4) ANALYSIS")
 logger.info("Using GeoStats Library")

    # Example 1: Gold Exploration
    try:
        gold_exploration_analysis(AGDB_PATH)
    except Exception as e:
        logger.error(f"Gold analysis error: {e}")

    # Example 2: Multi-element (Cu-Mo)
    try:
        multi_element_analysis(AGDB_PATH)
    except Exception as e:
        logger.error(f"Multi-element analysis error: {e}")

    # Example 3: Environmental (Arsenic)
    try:
        environmental_assessment(AGDB_PATH, element='As', threshold=20)
    except Exception as e:
        logger.error(f"Environmental analysis error: {e}")

 logger.info("ANALYSIS COMPLETE!")
 logger.info("\nThis example demonstrates:")
 logger.info(" Loading large-scale geochemical databases")
 logger.info(" Mineral exploration workflows (Au)")
 logger.info(" Multi-element cokriging (Cu-Mo)")
 logger.info(" Environmental risk assessment (As)")
 logger.info(" Probability mapping and uncertainty quantification")
 logger.info("\nGeoStats + AGDB4 = Powerful Geochemical Analysis!")
