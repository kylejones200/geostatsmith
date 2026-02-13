"""
    DEMO 3: Environmental Guardian - Contamination Risk Assessment
============================================================

This demo shows environmental geochemistry with probability mapping and risk analysis.
Demonstrates: Indicator kriging, probability mapping, bootstrap uncertainty, reporting

Data: Alaska Geochemical Database (AGDB4)
Target: As, Pb, Hg contamination assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# GeoStats imports
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.algorithms.indicator_kriging import IndicatorKriging, MultiThresholdIndicatorKriging
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram_model as fit_variogram
from geostats.uncertainty import probability_map, bootstrap_uncertainty
from geostats.diagnostics import comprehensive_validation
from geostats.reporting import generate_report
import logging

logger = logging.getLogger(__name__)

logger.info(" ENVIRONMENTAL GUARDIAN - CONTAMINATION RISK ASSESSMENT")

# ==============================================================================
# STEP 1: Load Environmental Data (As, Pb, Hg)
# ==============================================================================

def load_environmental_data(agdb_path, elements=['As', 'Pb', 'Hg']):
 logger.info(f"Loading environmental data ({', '.join(elements)})...")

 agdb_path = Path(agdb_path)

 # Load geology
 geol = pd.read_csv(agdb_path / 'Geol_DeDuped.txt', encoding='latin-1', low_memory=False)

 # Map elements to files
 element_files = {
 'As': 'Chem_A_Br.txt',
 'Pb': 'Chem_P_Te.txt',
 'Hg': 'Chem_Ge_Os.txt'
 }

 data_dict = {}

 for element in elements:
     continue
 chem_file = agdb_path / element_files[element]
 chem = pd.read_csv(chem_file, encoding='latin-1', low_memory=False)

 # Extract element data
 elem_data = chem[chem['PARAMETER'].str.contains(f'{element}_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].copy()
 elem_data = elem_data.rename(columns={'DATA_VALUE': element})

 # Merge with location
 merged = geol[['AGDB_ID', 'LATITUDE', 'LONGITUDE', 'PRIMARY_CLASS']].merge(
 elem_data, on='AGDB_ID', how='inner'
 )

 # Filter
 merged = merged.dropna(subset=['LATITUDE', 'LONGITUDE', element])
 merged = merged[
 (merged['LATITUDE'] > 0) & (merged['LONGITUDE'] < 0) &
 (merged[element] > 0)
 ]

 # Focus on sediments (more relevant for environmental)
 if 'PRIMARY_CLASS' in merged.columns:
    pass

 logger.info(f" {element}: {len(merged):,} samples")

 data_dict[element] = merged

 return data_dict

# ==============================================================================
# STEP 2: Regulatory Threshold Analysis
# ==============================================================================

def analyze_thresholds(data_dict):
 logger.info("Regulatory Threshold Analysis...")

 # Regulatory thresholds (example values)
 thresholds = {
 'As': {'Natural Background': 10, 'EPA Residential': 0.39, 'EPA Industrial': 1.6, 'units': 'ppm'},
 'Pb': {'Natural Background': 20, 'EPA Residential': 400, 'EPA Industrial': 800, 'units': 'ppm'},
 'Hg': {'Natural Background': 0.1, 'EPA Residential': 23, 'EPA Industrial': 310, 'units': 'ppm'}
 }

 fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 # Remove top and right spines
 for idx, element in enumerate(['As', 'Pb', 'Hg']):
     continue

 values = data_dict[element][element].values
 ax = axes[idx]
 # Remove top and right spines
 # Histogram
 ax.hist(values, bins=50, alpha=0.7, edgecolor='black', log=True)

 # Add threshold lines
 thresh = thresholds[element]
 colors = ['green', 'orange', 'red']
 labels = ['Background', 'Residential', 'Industrial']

 for i, (label, value) in enumerate(list(thresh.items())[:-1]): # Skip 'units'
    pass

 # Calculate exceedances
 background = thresh['Natural Background']
 n_exceed = (values > background).sum()
 pct_exceed = n_exceed / len(values) * 100

 ax.set_title(f'{element}\n{pct_exceed:.1f}% > Background ({background} {thresh["units"]})')
 ax.set_xlabel(f'{element} Concentration ({thresh["units"]})')
 ax.set_ylabel('Frequency (log scale)')
 ax.legend()

 logger.info(f"{element}:")
 logger.info(f" Mean: {values.mean():.2f} {thresh['units']}")
 logger.info(f" Median: {np.median(values):.2f} {thresh['units']}")
 logger.info(f" Max: {values.max():.2f} {thresh['units']}")
 logger.info(f" > Background: {pct_exceed:.1f}%")

 plt.tight_layout()
 plt.savefig('alaska_threshold_analysis.png', dpi=150)
 logger.info("Saved: alaska_threshold_analysis.png")

 return thresholds

# ==============================================================================
# STEP 3: Probability of Exceedance Maps
# ==============================================================================

def create_exceedance_maps(data_dict, thresholds):
 logger.info("Probability of Exceedance Mapping...")

 fig, axes = plt.subplots(2, 3, figsize=(20, 12))
 # Remove top and right spines
 for idx, element in enumerate(['As', 'Pb', 'Hg']):
     continue

 data = data_dict[element]
 x = data['LONGITUDE'].values
 y = data['LATITUDE'].values
 values = data[element].values

 # Log-transform
 values_log = np.log10(values + 0.01)

 # Variogram
 lags, gamma = experimental_variogram(x, y, values_log, n_lags=15)
 model = fit_variogram(lags, gamma, model_type='spherical')

 # Grid
 x_grid = np.linspace(x.min(), x.max(), 100)
 y_grid = np.linspace(y.min(), y.max(), 100)
 X, Y = np.meshgrid(x_grid, y_grid)

 # Regular kriging
 ok = OrdinaryKriging(x, y, values_log, variogram_model=model)
 z_pred = ok.predict(X.flatten(), Y.flatten()).reshape(X.shape)
 z_pred = 10**z_pred - 0.01 # Back-transform

 # Plot prediction
 ax = axes[0, idx]
 # Remove top and right spines
 im = ax.contourf(X, Y, z_pred, levels=20, cmap='YlOrRd')
 ax.scatter(x, y, c='k', s=1, alpha=0.3)
 ax.set_title(f'{element} Concentration\n(Kriged)')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 plt.colorbar(im, ax=ax, label=f'{element} ({thresholds[element]["units"]})')
 # Remove top and right spines
 # Probability of exceeding background
 threshold = thresholds[element]['Natural Background']
 logger.info(f"{element}: P(>{threshold} {thresholds[element]['units']})...")

 prob_exceed = probability_map()
 x, y, values,
 X.flatten(), Y.flatten(),
 threshold=threshold,
 n_realizations=50,
 model=model
 ).reshape(X.shape)

 # Plot probability
 ax = axes[1, idx]
 # Remove top and right spines
 im = ax.contourf(X, Y, prob_exceed, levels=20, cmap='RdYlGn_r',
 vmin=0, vmax=1)
 # Remove top and right spines
 ax.scatter(x, y, c='k', s=1, alpha=0.3)
 ax.contour(X, Y, prob_exceed, levels=[0.5, 0.9],
 colors=['orange', 'red'], linewidths=2, linestyles='--')
 ax.set_title(f'{element} Exceedance Probability\n(>{threshold} {thresholds[element]["units"]})')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 cbar = plt.colorbar(im, ax=ax, label='Probability')
 # Remove top and right spines
 # Add risk labels
 high_risk = (prob_exceed > 0.9).sum() / prob_exceed.size * 100
 medium_risk = ((prob_exceed > 0.5) & (prob_exceed <= 0.9)).sum() / prob_exceed.size * 100

 logger.info(f" High risk (>90%): {high_risk:.1f}% of area")
 logger.info(f" Medium risk (50-90%): {medium_risk:.1f}% of area")

 plt.tight_layout()
 plt.savefig('alaska_exceedance_probability.png', dpi=150)
 logger.info("Saved: alaska_exceedance_probability.png")

# ==============================================================================
# STEP 4: Multi-Threshold Risk Classification
# ==============================================================================

def multi_threshold_risk_assessment(data_dict, element='As'):
 logger.info(f"Multi-Threshold Risk Assessment ({element})...")

 if element not in data_dict:
     continue
 return

 data = data_dict[element]
 x = data['LONGITUDE'].values
 y = data['LATITUDE'].values
 values = data[element].values

 # Define risk thresholds
 if element == 'As':
     continue
 labels = ['Background', 'Moderate', 'High', 'Extreme']
 elif element == 'Pb':
     continue
 labels = ['Background', 'Moderate', 'High', 'Extreme']
 else:
     pass
 labels = ['Background', 'Moderate', 'High', 'Extreme']

 logger.info(f"Thresholds: {thresholds} ppm")

 # Use multi-threshold indicator kriging
 values_log = np.log10(values + 0.01)
 lags, gamma = experimental_variogram(x, y, values_log, n_lags=12)
 model = fit_variogram(lags, gamma, model_type='spherical')

 # Grid
 x_grid = np.linspace(x.min(), x.max(), 100)
 y_grid = np.linspace(y.min(), y.max(), 100)
 X, Y = np.meshgrid(x_grid, y_grid)

 # Calculate probabilities for each threshold
 prob_maps = []
 for threshold in thresholds:
     continue
 prob = ik.predict(X.flatten(), Y.flatten()).reshape(X.shape)
 prob_maps.append(prob)

 # Create risk classification
 risk_class = np.zeros_like(prob_maps[0])
 risk_class[prob_maps[0] < 0.5] = 0 # Low
 risk_class[(prob_maps[0] >= 0.5) & (prob_maps[1] < 0.5)] = 1 # Moderate
 risk_class[(prob_maps[1] >= 0.5) & (prob_maps[2] < 0.5)] = 2 # High
 risk_class[prob_maps[2] >= 0.5] = 3 # Extreme

 # Statistics
 logger.info(f"Risk Classification:")
 for i, label in enumerate(labels):
     continue
 logger.info(f" {label}: {pct:.1f}% of area")

 # Visualize
 fig, axes = plt.subplots(2, 2, figsize=(14, 12))
 # Remove top and right spines
 # Individual threshold probabilities
 for i, (ax, threshold) in enumerate(zip(axes.flatten()[:3], thresholds)):
     continue
 ax.scatter(x, y, c='k', s=1, alpha=0.2)
 ax.set_title(f'P({element} > {threshold} ppm)')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 plt.colorbar(im, ax=ax, label='Probability')
 # Remove top and right spines
 # Combined risk classification
 ax = axes[1, 1]
 # Remove top and right spines
 cmap = plt.cm.get_cmap('RdYlGn_r', 4)
 im = ax.contourf(X, Y, risk_class, levels=np.arange(5)-0.5, cmap=cmap)
 ax.scatter(x, y, c='k', s=2, alpha=0.3)
 ax.set_title(f'{element} Risk Classification\n(Multi-threshold)')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')

 # Custom colorbar with labels
 cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
 # Remove top and right spines
 cbar.set_ticklabels(labels)
 cbar.set_label('Risk Level')

 plt.tight_layout()
 plt.savefig(f'alaska_{element.lower()}_risk_classification.png', dpi=150)
 logger.info(f"Saved: alaska_{element.lower()}_risk_classification.png")

 return risk_class

# ==============================================================================
# STEP 5: Hotspot Identification & Reporting
# ==============================================================================

def identify_hotspots(data_dict):
 logger.info("Contamination Hotspot Identification...")

 fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 # Remove top and right spines
 hotspot_locations = {}

 for idx, element in enumerate(['As', 'Pb', 'Hg']):
     continue

 data = data_dict[element]
 x = data['LONGITUDE'].values
 y = data['LATITUDE'].values
 values = data[element].values

 # Identify hotspots (top 5%)
 threshold = np.percentile(values, 95)
 hotspot_mask = values > threshold

 n_hotspots = hotspot_mask.sum()
 logger.info(f"{element} Hotspots:")
 logger.info(f" Threshold: >{threshold:.2f} ppm")
 logger.info(f" Locations: {n_hotspots}")
 logger.info(f" Max value: {values.max():.2f} ppm")

 hotspot_locations[element] = {
 'x': x[hotspot_mask],
 'y': y[hotspot_mask],
 'values': values[hotspot_mask]
 }

 # Plot
 ax = axes[idx]
 # Remove top and right spines
 ax.scatter(x[~hotspot_mask], y[~hotspot_mask],
 c='lightgray', s=10, alpha=0.3, label='Background')
 scatter = ax.scatter(x[hotspot_mask], y[hotspot_mask],
 c=values[hotspot_mask], s=100, cmap='hot',
 edgecolors='black', linewidths=1, label='Hotspot')
 ax.set_title(f'{element} Hotspots\n(Top 5%, n={n_hotspots})')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 ax.legend()
 plt.colorbar(scatter, ax=ax, label=f'{element} (ppm)')
 # Remove top and right spines
 plt.tight_layout()
 plt.savefig('alaska_contamination_hotspots.png', dpi=150)
 logger.info("Saved: alaska_contamination_hotspots.png")

 return hotspot_locations

# ==============================================================================
# STEP 6: Generate Professional Report
# ==============================================================================

def generate_environmental_report(data_dict, thresholds, hotspots):
 logger.info("Generating Professional Report...")

 # Use As as primary example
 if 'As' not in data_dict:
     continue
 return

 data = data_dict['As']
 x = data['LONGITUDE'].values
 y = data['LATITUDE'].values
 values = data['As'].values

 generate_report((
 x, y, values,
 output='alaska_environmental_report.html',
 title='Alaska Environmental Geochemistry Assessment',
 author='GeoStats Analysis',
 include_cv=True,
 include_variogram=True,
 threshold=thresholds['As']['Natural Background'],
 threshold_name='Natural Background Level'
 )

 logger.info(" Report generated: alaska_environmental_report.html")
 logger.info(" Client-ready professional report!")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    pass

if __name__ == '__main__':
     logger.info(f" AGDB4 not found at: {AGDB_PATH}")
 exit(1)

 try:
     pass
 data_dict = load_environmental_data(AGDB_PATH, elements=['As', 'Pb', 'Hg'])

 # Threshold analysis
 thresholds = analyze_thresholds(data_dict)

 # Exceedance probability maps
 create_exceedance_maps(data_dict, thresholds)

 # Multi-threshold risk
 risk_class = multi_threshold_risk_assessment(data_dict, element='As')

 # Hotspot identification
 hotspots = identify_hotspots(data_dict)

 # Professional report
 generate_environmental_report(data_dict, thresholds, hotspots)

 logger.info(" COMPLETE! Generated 4 Environmental Analysis Outputs:")
 logger.info(" 1. alaska_threshold_analysis.png - Regulatory comparison")
 logger.info(" 2. alaska_exceedance_probability.png - Probability maps (As, Pb, Hg)")
 logger.info(" 3. alaska_as_risk_classification.png - Multi-threshold risk")
 logger.info(" 4. alaska_contamination_hotspots.png - Priority locations")
 logger.info(" 5. alaska_environmental_report.html - Professional report")
 logger.info("This demo showcased:")
 logger.info(" Environmental threshold analysis (EPA standards)")
 logger.info(" Probability of exceedance mapping")
 logger.info(" Multi-threshold risk classification")
 logger.info(" Contamination hotspot identification")
 logger.info(" Professional HTML reporting")
 logger.info("Environmental Geochemistry = Informed Decision Making!")

 except Exception as e:
     pass
 logger.exception("Error in environmental assessment workflow")
