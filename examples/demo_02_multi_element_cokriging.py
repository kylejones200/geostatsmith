"""
DEMO 2: Multi-Element Arsenal - Cu-Mo-Au Porphyry Analysis
==========================================================

This demo shows multi-element geochemical analysis for porphyry deposit exploration.
Demonstrates: Cokriging, correlation analysis, 3D visualization, anomaly detection

Data: Alaska Geochemical Database (AGDB4)
Target: Cu-Mo-Au associations (porphyry signatures)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from scipy.stats import pearsonr

# GeoStats imports
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.algorithms.cokriging import Cokriging
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram_model as fit_variogram
import logging

logger = logging.getLogger(__name__)
# from geostats.diagnostics.outlier_detection import outlier_analysis # Not available
# from geostats.transformations import normal_score_transform, back_transform as inverse_normal_score # Not available

logger.info(" MULTI-ELEMENT ARSENAL - PORPHYRY DEPOSIT ANALYSIS")

# ==============================================================================
# STEP 1: Load Multi-Element Data
# ==============================================================================

def load_multi_element_data(agdb_path, region='Pebble'):
 Load Cu, Mo, Au data - the classic porphyry association
 """
 logger.info("Loading multi-element data (Cu-Mo-Au)...")

 agdb_path = Path(agdb_path)

 # Load geology
 geol = pd.read_csv(agdb_path / 'Geol_DeDuped.txt', encoding='latin-1', low_memory=False)

 # Load chemistry files
 chem_c_gd = pd.read_csv(agdb_path / 'Chem_C_Gd.txt', encoding='latin-1', low_memory=False) # Cu
 chem_ge_os = pd.read_csv(agdb_path / 'Chem_Ge_Os.txt', encoding='latin-1', low_memory=False) # Mo
 chem_a_br = pd.read_csv(agdb_path / 'Chem_A_Br.txt', encoding='latin-1', low_memory=False) # Au

 # Extract element-specific data
 cu_data = chem_c_gd[chem_c_gd['PARAMETER'].str.contains('Cu_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].rename(columns={'DATA_VALUE': 'Cu'})
 mo_data = chem_ge_os[chem_ge_os['PARAMETER'].str.contains('Mo_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].rename(columns={'DATA_VALUE': 'Mo'})
 au_data = chem_a_br[chem_a_br['PARAMETER'].str.contains('Au_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].rename(columns={'DATA_VALUE': 'Au'})

 # Merge all three elements
 merged = geol[['AGDB_ID', 'LATITUDE', 'LONGITUDE', 'QUAD', 'DISTRICT_NAME']].copy()
 merged = merged.merge(cu_data, on='AGDB_ID', how='inner')
 merged = merged.merge(mo_data, on='AGDB_ID', how='inner')
 merged = merged.merge(au_data, on='AGDB_ID', how='inner')

 # Filter valid coordinates and positive values
 merged = merged.dropna(subset=['LATITUDE', 'LONGITUDE', 'Cu', 'Mo', 'Au'])
 merged = merged[
 (merged['LATITUDE'] > 0) & (merged['LONGITUDE'] < 0) &
 (merged['Cu'] > 0) & (merged['Mo'] > 0) & (merged['Au'] > 0)
 ]

 # Focus on specific region if desired (e.g., Iliamna - contains Pebble deposit)
 if region and 'QUAD' in merged.columns:
 if len(region_data) > 50:
 logger.info(f" Focused on {region} region")

 logger.info(f" Samples with Cu+Mo+Au: {len(merged):,}")

 return merged

# ==============================================================================
# STEP 2: Correlation & Association Analysis
# ==============================================================================

def analyze_element_correlations(data):
 logger.info("Element Correlation Analysis...")

 # Extract values
 cu = data['Cu'].values
 mo = data['Mo'].values
 au = data['Au'].values

 # Log-transform (typical for geochemistry)
 cu_log = np.log10(cu + 1)
 mo_log = np.log10(mo + 0.1)
 au_log = np.log10(au + 0.001)

 # Calculate correlations
 corr_cu_mo, p_cu_mo = pearsonr(cu_log, mo_log)
 corr_cu_au, p_cu_au = pearsonr(cu_log, au_log)
 corr_mo_au, p_mo_au = pearsonr(mo_log, au_log)

 logger.info(f"Element Correlations (log-transformed):")
 logger.info(f" Cu-Mo: r={corr_cu_mo:.3f} (p={p_cu_mo:.4f})")
 logger.info(f" Cu-Au: r={corr_cu_au:.3f} (p={p_cu_au:.4f})")
 logger.info(f" Mo-Au: r={corr_mo_au:.3f} (p={p_mo_au:.4f})")

 if corr_cu_mo > 0.5:
    pass

 # Scatter plots with density
 fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 # Cu vs Mo
 axes[0].hexbin(cu_log, mo_log, gridsize=30, cmap='YlOrRd', mincnt=1)
 # Remove top and right spines
 axes[0]
 axes[0].spines['right'].set_visible(False)
 # Remove top and right spines
 axes[0].hexbin(cu_log, mo_log, gridsize
 gridsize.spines['right'].set_visible(False)
 axes[0].set_xlabel('log10(Cu, ppm)')
 # Remove top and right spines
 axes[0].set_xlabel('log10(Cu, ppm)')
 axes[0].set_ylabel('log10(Mo, ppm)')
 # Remove top and right spines
 axes[0].set_ylabel('log10(Mo, ppm)')
 axes[0].set_title(f'Cu vs Mo\nr = {corr_cu_mo:.3f}')
 # Remove top and right spines
 axes[0].set_title(f'
 nr.spines['right'].set_visible(False)
 # Cu vs Au
 axes[1].hexbin(cu_log, au_log, gridsize=30, cmap='YlOrRd', mincnt=1)
 # Remove top and right spines
 axes[1]
 axes[1].spines['right'].set_visible(False)
 # Remove top and right spines
 axes[1].hexbin(cu_log, au_log, gridsize
 gridsize.spines['right'].set_visible(False)
 axes[1].set_xlabel('log10(Cu, ppm)')
 # Remove top and right spines
 axes[1].set_xlabel('log10(Cu, ppm)')
 axes[1].set_ylabel('log10(Au, ppm)')
 # Remove top and right spines
 axes[1].set_ylabel('log10(Au, ppm)')
 axes[1].set_title(f'Cu vs Au\nr = {corr_cu_au:.3f}')
 # Remove top and right spines
 axes[1].set_title(f'
 nr.spines['right'].set_visible(False)
 # Mo vs Au
 axes[2].hexbin(mo_log, au_log, gridsize=30, cmap='YlOrRd', mincnt=1)
 # Remove top and right spines
 axes[2]
 axes[2].spines['right'].set_visible(False)
 # Remove top and right spines
 axes[2].hexbin(mo_log, au_log, gridsize
 gridsize.spines['right'].set_visible(False)
 axes[2].set_xlabel('log10(Mo, ppm)')
 # Remove top and right spines
 axes[2].set_xlabel('log10(Mo, ppm)')
 axes[2].set_ylabel('log10(Au, ppm)')
 # Remove top and right spines
 axes[2].set_ylabel('log10(Au, ppm)')
 axes[2].set_title(f'Mo vs Au\nr = {corr_mo_au:.3f}')
 # Remove top and right spines
 axes[2].set_title(f'
 nr.spines['right'].set_visible(False)
 plt.tight_layout()
 plt.savefig('alaska_element_correlations.png', dpi=150)
 logger.info("Saved: alaska_element_correlations.png")

 return {'Cu': cu, 'Mo': mo, 'Au': au,
 'Cu_log': cu_log, 'Mo_log': mo_log, 'Au_log': au_log,
 'correlations': {'Cu-Mo': corr_cu_mo, 'Cu-Au': corr_cu_au, 'Mo-Au': corr_mo_au}}

# ==============================================================================
# STEP 3: Geochemical Anomaly Detection
# ==============================================================================

def detect_geochemical_anomalies(data, elements_dict):
 logger.info("Geochemical Anomaly Detection...")

 x = data['LONGITUDE'].values
 y = data['LATITUDE'].values

 anomalies_dict = {}

 for element in ['Cu', 'Mo', 'Au']:
    pass

 # Detect outliers using multiple methods
 outlier_results = detect_outliers(
 x, y, values,
 methods=['iqr', 'zscore', 'spatial'],
 return_details=True
 )

 n_anomalies = outlier_results['outlier_mask'].sum()
 logger.info(f"{element} Anomalies:")
 logger.info(f" Detected: {n_anomalies} ({n_anomalies/len(values)*100:.1f}%)")
 logger.info(f" Threshold: >{outlier_results['threshold']:.2f} ppm")
 logger.info(f" Max value: {values.max():.2f} ppm")

 anomalies_dict[element] = outlier_results

 # Combined anomaly index (Cu + Mo + Au all elevated)
 combined_anomaly = (
 anomalies_dict['Cu']['outlier_mask'] &
 anomalies_dict['Mo']['outlier_mask'] &
 anomalies_dict['Au']['outlier_mask']
 )

 n_combined = combined_anomaly.sum()
 logger.info(f"MULTI-ELEMENT ANOMALIES (Cu+Mo+Au):")
 logger.info(f" Samples: {n_combined} ({n_combined/len(combined_anomaly)*100:.2f}%)")
 logger.info(f" These are high-priority exploration targets!")

 # Map anomalies
 fig, axes = plt.subplots(2, 2, figsize=(16, 14))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 # Individual elements
 for idx, element in enumerate(['Cu', 'Mo', 'Au']):
 mask = anomalies_dict[element]['outlier_mask']

 # Background samples
 ax.scatter(x[~mask], y[~mask], c='lightgray', s=10, alpha=0.5, label='Background')

 # Anomalous samples
 ax.scatter(x[mask], y[mask], c=elements_dict[element][mask],
 s=50, cmap='hot', edgecolors='black', linewidths=0.5, label='Anomaly')

 ax.set_title(f'{element} Anomalies (n={mask.sum()})')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 ax.legend()

 # Combined
 ax = axes[1, 1]
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 ax.scatter(x[~combined_anomaly], y[~combined_anomaly],
 c='lightgray', s=10, alpha=0.5, label='Background')
 ax.scatter(x[combined_anomaly], y[combined_anomaly],
 c='red', s=150, marker='*', edgecolors='darkred',
 linewidths=2, label='Multi-element target', zorder=10)
 ax.set_title(f'COMBINED Anomalies\n(Cu+Mo+Au, n={n_combined})')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 ax.legend()

 plt.tight_layout()
 plt.savefig('alaska_anomaly_detection.png', dpi=150)
 logger.info("Saved: alaska_anomaly_detection.png")

 return anomalies_dict, combined_anomaly

# ==============================================================================
# STEP 4: Cokriging (Use correlation to improve predictions)
# ==============================================================================

def compare_kriging_vs_cokriging(data, elements_dict):
 logger.info("Cokriging vs Ordinary Kriging Comparison...")

 x = data['LONGITUDE'].values
 y = data['LATITUDE'].values

 cu_log = elements_dict['Cu_log']
 mo_log = elements_dict['Mo_log']

 # Create grid
 x_grid = np.linspace(x.min(), x.max(), 60)
 y_grid = np.linspace(y.min(), y.max(), 60)
 X, Y = np.meshgrid(x_grid, y_grid)

 # Method 1: Ordinary Kriging (Cu only)
 logger.info("Ordinary Kriging (Cu alone)...")
 lags, gamma = experimental_variogram(x, y, cu_log, n_lags=12)
 model_cu = fit_variogram(lags, gamma, model_type='spherical')

 ok_cu = OrdinaryKriging(x, y, cu_log, variogram_model=model_cu)
 cu_ok, var_ok = ok_cu.predict(X.flatten(), Y.flatten(), return_variance=True)
 cu_ok = cu_ok.reshape(X.shape)
 var_ok = var_ok.reshape(X.shape)

 # Method 2: Cokriging (Cu with Mo as secondary)
 logger.info("Cokriging (Cu using Mo as secondary variable)...")

 # Fit variograms
 lags, gamma_mo = experimental_variogram(x, y, mo_log, n_lags=12)
 model_mo = fit_variogram(lags, gamma_mo, model_type='spherical')

 # For simplicity, use same model for cross-variogram
 # (In practice, would calculate actual cross-variogram)
 model_cross = model_cu.copy()
 model_cross['sill'] = (model_cu['sill'] + model_mo['sill']) / 2

 cok = Cokriging(
 x_primary=x, y_primary=y, z_primary=cu_log,
 x_secondary=x, y_secondary=y, z_secondary=mo_log,
 primary_variogram=model_cu,
 secondary_variogram=model_mo,
 cross_variogram=model_cross
 )

 cu_cok, var_cok = cok.predict(X.flatten(), Y.flatten())
 cu_cok = cu_cok.reshape(X.shape)
 var_cok = var_cok.reshape(X.shape)

 # Compare variances
 variance_reduction = ((var_ok - var_cok) / var_ok * 100).mean()
 logger.info(f"Average variance reduction: {variance_reduction:.1f}%")
 logger.info(f" Cokriging provides {variance_reduction:.0f}% more certain predictions!")

 # Visualize
 fig, axes = plt.subplots(2, 2, figsize=(14, 12))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 # OK prediction
 im1 = axes[0, 0].contourf(X, Y, cu_ok, levels=20, cmap='YlOrRd')
 axes[0, 0].scatter(x, y, c='k', s=2, alpha=0.3)
 # Remove top and right spines
 axes[0, 0]
 axes[0, 0].spines['right'].set_visible(False)
 # Remove top and right spines
 axes[0, 0].scatter(x, y, c
 c.spines['right'].set_visible(False)
 axes[0, 0].set_title('Ordinary Kriging\n(Cu alone)')
 # Remove top and right spines
 axes[0, 0].set_title('Ordinary Kriging\n(Cu alone)')
 plt.colorbar(im1, ax=axes[0, 0], label='log10(Cu)')
 # Remove top and right spines
 axes[0, 0].set_title('Ordinary Kriging\n(Cu alone)')

 # OK variance
 im2 = axes[0, 1].contourf(X, Y, var_ok, levels=20, cmap='viridis')
 axes[0, 1].set_title('OK Variance')
 # Remove top and right spines
 axes[0, 1].set_title('OK Variance')
 plt.colorbar(im2, ax=axes[0, 1], label='Variance')
 # Remove top and right spines
 axes[0, 1].set_title('OK Variance')

 # Cokriging prediction
 im3 = axes[1, 0].contourf(X, Y, cu_cok, levels=20, cmap='YlOrRd')
 axes[1, 0].scatter(x, y, c='k', s=2, alpha=0.3)
 # Remove top and right spines
 axes[1, 0]
 axes[1, 0].spines['right'].set_visible(False)
 # Remove top and right spines
 axes[1, 0].scatter(x, y, c
 c.spines['right'].set_visible(False)
 axes[1, 0].set_title('Cokriging\n(Cu with Mo)')
 # Remove top and right spines
 axes[1, 0].set_title('Cokriging\n(Cu with Mo)')
 plt.colorbar(im3, ax=axes[1, 0], label='log10(Cu)')
 # Remove top and right spines
 axes[1, 0].set_title('Cokriging\n(Cu with Mo)')

 # Cokriging variance
 im4 = axes[1, 1].contourf(X, Y, var_cok, levels=20, cmap='viridis')
 axes[1, 1].set_title(f'Cokriging Variance\n({variance_reduction:.0f}% reduction!)')
 # Remove top and right spines
 axes[1, 1].set_title(f'Cokriging Variance\n({variance_reduction:.0f}% reduction!)')
 plt.colorbar(im4, ax=axes[1, 1], label='Variance')
 # Remove top and right spines
 axes[1, 1].set_title(f'Cokriging Variance\n({variance_reduction:.0f}% reduction!)')

 for ax in axes.flatten():
 ax.set_ylabel('Latitude')

 plt.tight_layout()
 plt.savefig('alaska_cokriging_comparison.png', dpi=150)
 logger.info("Saved: alaska_cokriging_comparison.png")

 return variance_reduction

# ==============================================================================
# STEP 5: Porphyry Index (Combined Geochemical Signature)
# ==============================================================================

def calculate_porphyry_index(data, elements_dict):
 logger.info("Porphyry Fertility Index...")

 x = data['LONGITUDE'].values
 y = data['LATITUDE'].values

 cu_log = elements_dict['Cu_log']
 mo_log = elements_dict['Mo_log']
 au_log = elements_dict['Au_log']

 # Normalize each element to 0-1
 cu_norm = (cu_log - cu_log.min()) / (cu_log.max() - cu_log.min())
 mo_norm = (mo_log - mo_log.min()) / (mo_log.max() - mo_log.min())
 au_norm = (au_log - au_log.min()) / (au_log.max() - au_log.min())

 # Porphyry index: weighted combination
 # (Cu and Mo more important for porphyry)
 porphyry_index = 0.4 * cu_norm + 0.4 * mo_norm + 0.2 * au_norm

 logger.info(f" Index range: {porphyry_index.min():.3f} to {porphyry_index.max():.3f}")

 # Krige the index
 lags, gamma = experimental_variogram(x, y, porphyry_index, n_lags=12)
 model = fit_variogram(lags, gamma, model_type='spherical')

 x_grid = np.linspace(x.min(), x.max(), 80)
 y_grid = np.linspace(y.min(), y.max(), 80)
 X, Y = np.meshgrid(x_grid, y_grid)

 ok = OrdinaryKriging(x, y, porphyry_index, variogram_model=model)
 index_pred = ok.predict(X.flatten(), Y.flatten()).reshape(X.shape)

 # Identify high-potential zones
 high_potential = index_pred > np.percentile(index_pred, 90)

 logger.info(f"High-potential zones: {high_potential.sum() / high_potential.size * 100:.1f}% of area")
 logger.info(f" Top 10% porphyry fertility")

 # Visualize
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
 # Remove top and right spines
 ax.spines['right'].set_visible(False)
 # Porphyry index map
 im1 = ax1.contourf(X, Y, index_pred, levels=20, cmap='RdYlGn')
 ax1.scatter(x, y, c=porphyry_index, s=30, cmap='RdYlGn',
 # Remove top and right spines
 ax1.spines['right'].set_visible(False)
 # Remove top and right spines
 ax1.scatter(x, y, c
 c.spines['right'].set_visible(False)
 edgecolors='k', linewidths=0.5)
 ax1.contour(X, Y, index_pred, levels=[np.percentile(index_pred, 90)],
 # Remove top and right spines
 ax1.contour(X, Y, index_pred, levels
 levels.spines['right'].set_visible(False)
 colors='red', linewidths=3, linestyles='--')
 ax1.set_title('Porphyry Fertility Index\n(Combined Cu-Mo-Au signature)')
 ax1.set_xlabel('Longitude')
 ax1.set_ylabel('Latitude')
 plt.colorbar(im1, ax=ax1, label='Fertility Index')
 # Remove top and right spines
 ax1.set_ylabel('Latitude')

 # High-potential zones
 im2 = ax2.contourf(X, Y, high_potential.astype(int), levels=1, colors=['white', 'red'], alpha=0.5)
 ax2.scatter(x, y, c=porphyry_index, s=20, cmap='RdYlGn', alpha=0.6)
 # Remove top and right spines
 ax2.spines['right'].set_visible(False)
 # Remove top and right spines
 ax2.scatter(x, y, c
 c.spines['right'].set_visible(False)
 ax2.set_title('High-Potential Zones\n(Top 10% fertility)')
 ax2.set_xlabel('Longitude')
 ax2.set_ylabel('Latitude')

 plt.tight_layout()
 plt.savefig('alaska_porphyry_index.png', dpi=150)
 logger.info("Saved: alaska_porphyry_index.png")

 return porphyry_index, index_pred

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    pass

if __name__ == '__main__':
     logger.info(f" AGDB4 not found at: {AGDB_PATH}")
 exit(1)

 try:
 try:
 data = load_multi_element_data(AGDB_PATH, region='Iliamna') # or 'Pebble'

 # Correlation analysis
 elements_dict = analyze_element_correlations(data)

 # Anomaly detection
 anomalies, combined = detect_geochemical_anomalies(data, elements_dict)

 # Cokriging comparison
 var_reduction = compare_kriging_vs_cokriging(data, elements_dict)

 # Porphyry fertility index
 index, index_pred = calculate_porphyry_index(data, elements_dict)

 logger.info(" COMPLETE! Generated 4 Advanced Analysis Figures:")
 logger.info(" 1. alaska_element_correlations.png - Cu-Mo-Au associations")
 logger.info(" 2. alaska_anomaly_detection.png - Multi-element anomaly maps")
 logger.info(" 3. alaska_cokriging_comparison.png - OK vs Cokriging")
 logger.info(" 4. alaska_porphyry_index.png - Integrated fertility index")
 logger.info("This demo showcased:")
 logger.info(" Multi-element geochemistry")
 logger.info(" Element correlation analysis")
 logger.info(" Anomaly detection (3 methods)")
 logger.info(" Cokriging for improved predictions")
 logger.info(f" {var_reduction:.0f}% variance reduction with cokriging!")
 logger.info(" Porphyry fertility indexing")
 logger.info("Multi-Element Analysis = Powerful Exploration Tool!")

 except Exception as e:
 logger.exception("Multi-element analysis error")
