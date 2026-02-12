"""
Alaska Geochemical Database (AGDB4) - Full Analysis
====================================================

This script runs geostatistical analysis on AGDB4 data
and saves all results and figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from datetime import datetime

# Ensure output directory exists
OUTPUT_DIR = Path('/Users/k.jones/Desktop/geostats/alaska_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = OUTPUT_DIR / 'alaska_full_analysis_results.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("ALASKA GEOCHEMICAL DATABASE (AGDB4) - FULL ANALYSIS")
logger.info(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# ==============================================================================
# PART 1: Data Loading and Exploration
# ==============================================================================

logger.info("PART 1: DATA LOADING AND EXPLORATION")

AGDB_PATH = Path('/Users/k.jones/Downloads/AGDB4_text')

if not AGDB_PATH.exists():
if not AGDB_PATH.exists():
 sys.exit(1)

logger.info(f"AGDB4 data found at: {AGDB_PATH}")

# List data files
logger.debug("Available data files:")
data_files = sorted(AGDB_PATH.glob('*.txt'))
for file in data_files[:15]:
for file in data_files[:15]:

# Load location data
logger.info("Loading sample location data...")
try:
try:
 logger.debug(f"Columns: {list(geol.columns[:15])}")

 # Geographic extent
 if 'LATITUDE' in geol.columns and 'LONGITUDE' in geol.columns:
 if 'LATITUDE' in geol.columns and 'LONGITUDE' in geol.columns:
 logger.info(f"Geographic coverage: {len(valid_coords):,} samples with coordinates")
 logger.info(f"Latitude: {valid_coords['LATITUDE'].min():.2f}° to {valid_coords['LATITUDE'].max():.2f}°N")
 logger.info(f"Longitude: {valid_coords['LONGITUDE'].min():.2f}° to {valid_coords['LONGITUDE'].max():.2f}°W")

except Exception as e:
 logger.exception("Error loading location data")
 sys.exit(1)

# ==============================================================================
# PART 2: Gold Exploration Analysis
# ==============================================================================

logger.info("PART 2: GOLD EXPLORATION ANALYSIS - FAIRBANKS DISTRICT")

try:
try:
 chem_file = AGDB_PATH / 'Chem_A_Br.txt'
 chem = pd.read_csv(chem_file, low_memory=False, encoding='latin-1')

 # Filter for gold (PARAMETER contains 'Au')
 au_chem = chem[chem['PARAMETER'].str.contains('Au_', case=False, na=False)].copy()
 au_chem = au_chem.rename(columns={'DATA_VALUE': 'Au'})
 logger.info(f"Found {len(au_chem):,} gold analyses")

 # Merge with locations
 au_data = geol.merge(au_chem[['AGDB_ID', 'Au']], on='AGDB_ID', how='inner')
 au_data = au_data.dropna(subset=['LATITUDE', 'LONGITUDE', 'Au'])
 au_data = au_data[au_data['Au'] > 0] # Remove non-detects

 # Focus on Fairbanks area (64-66°N, -149 to -145°W)
 fairbanks = au_data[
 (au_data['LATITUDE'] > 64.0) & (au_data['LATITUDE'] < 66.0) &
 (au_data['LONGITUDE'] > -149.0) & (au_data['LONGITUDE'] < -145.0)
 ].copy()

 logger.info(f"Fairbanks region: {len(fairbanks):,} samples")

 # Extract arrays
 x = fairbanks['LONGITUDE'].values
 y = fairbanks['LATITUDE'].values
 au = fairbanks['Au'].values

 # Statistics
 logger.info(f"Gold Statistics (Fairbanks): {len(au):,} samples")
 logger.info(f"  Mean: {au.mean():.4f} ppm, Median: {np.median(au):.4f} ppm")
 logger.info(f"  Range: {au.min():.4f} - {au.max():.4f} ppm")
 logger.info(f"  >0.1 ppm: {(au > 0.1).sum():,} samples ({(au > 0.1).mean()*100:.1f}%)")
 logger.info(f"  >1.0 ppm: {(au > 1.0).sum():,} samples ({(au > 1.0).mean()*100:.1f}%)")

 # Create visualization
 logger.info("Creating gold distribution map...")
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

 # Linear scale
 scatter1 = ax1.scatter(x, y, c=au, s=30, cmap='YlOrRd',
 vmin=0, vmax=np.percentile(au, 95),
 alpha=0.6, edgecolors='k', linewidths=0.5)
 ax1.set_title('Gold Distribution - Fairbanks\n(Linear Scale)', fontsize=14, fontweight='bold')
 ax1.set_xlabel('Longitude')
 ax1.set_ylabel('Latitude')
 plt.colorbar(scatter1, ax=ax1, label='Au (ppm)')

 # Log scale
 au_log = np.log10(au + 0.001)
 scatter2 = ax2.scatter(x, y, c=au_log, s=30, cmap='YlOrRd',
 alpha=0.6, edgecolors='k', linewidths=0.5)
 ax2.set_title('Gold Distribution - Fairbanks\n(Log Scale)', fontsize=14, fontweight='bold')
 ax2.set_xlabel('Longitude')
 plt.colorbar(scatter2, ax=ax2, label='log₁₀(Au + 0.001)')

 plt.tight_layout()
 output_file = OUTPUT_DIR / 'figure_01_gold_distribution.png'
 plt.savefig(output_file, dpi=150, bbox_inches='tight')
 logger.info(f"Saved: {output_file.name}")
 plt.close()

except Exception as e:
 logger.exception("Gold analysis error")

# ==============================================================================
# PART 3: Multi-Element Analysis
# ==============================================================================

logger.info("PART 3: MULTI-ELEMENT ANALYSIS - Cu, Mo, Au")

try:

try:
 chem_c = pd.read_csv(AGDB_PATH / 'Chem_C_Gd.txt', low_memory=False, encoding='latin-1')
 cu_chem = chem_c[chem_c['PARAMETER'].str.contains('Cu_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].copy()
 cu_chem = cu_chem.rename(columns={'DATA_VALUE': 'Cu'})
 logger.info(f"Cu: {len(cu_chem):,} analyses")

 # Load gold data
 au_chem = chem[chem['PARAMETER'].str.contains('Au_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].copy()
 au_chem = au_chem.rename(columns={'DATA_VALUE': 'Au'})
 logger.info(f"Au: {len(au_chem):,} analyses")

 # Merge all
 multi_data = geol.merge(cu_chem, on='AGDB_ID', how='inner')
 multi_data = multi_data.merge(au_chem, on='AGDB_ID', how='inner')
 multi_data = multi_data.dropna(subset=['LATITUDE', 'LONGITUDE', 'Cu', 'Au'])
 multi_data = multi_data[(multi_data['Cu'] > 0) & (multi_data['Au'] > 0)]

 logger.info(f"Combined dataset: {len(multi_data):,} samples with both Cu and Au")

 # Calculate correlation
 from scipy import stats
 cu_vals = multi_data['Cu'].values
 au_vals = multi_data['Au'].values

 # Log-transform for better correlation
 cu_log = np.log10(cu_vals + 1)
 au_log = np.log10(au_vals + 0.001)

 corr_raw = stats.pearsonr(cu_vals, au_vals)[0]
 corr_log = stats.pearsonr(cu_log, au_log)[0]

 logger.info(f"Element Correlations: Cu-Au (raw) r={corr_raw:.3f}, (log-trans) r={corr_log:.3f}")

 # Create correlation plot
 logger.info("Creating multi-element correlation plot...")
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

 # Raw correlation
 ax1.scatter(cu_vals, au_vals, alpha=0.4, s=20, c='steelblue', edgecolors='k', linewidths=0.3)
 ax1.set_xlabel('Cu (ppm)', fontsize=12)
 ax1.set_ylabel('Au (ppm)', fontsize=12)
 ax1.set_title(f'Cu vs Au Correlation\n(r = {corr_raw:.3f})', fontsize=14, fontweight='bold')
 ax1.grid(True, alpha=0.3)

 # Log-transformed
 ax2.scatter(cu_log, au_log, alpha=0.4, s=20, c='coral', edgecolors='k', linewidths=0.3)

 # Trend line
 z = np.polyfit(cu_log, au_log, 1)
 p = np.poly1d(z)
 x_trend = np.linspace(cu_log.min(), cu_log.max(), 100)
 ax2.plot(x_trend, p(x_trend), 'r--', linewidth=2, label='Trend line')

 ax2.set_xlabel('log₁₀(Cu + 1)', fontsize=12)
 ax2.set_ylabel('log₁₀(Au + 0.001)', fontsize=12)
 ax2.set_title(f'Log-Transformed Correlation\n(r = {corr_log:.3f})', fontsize=14, fontweight='bold')
 ax2.legend()
 ax2.grid(True, alpha=0.3)

 plt.tight_layout()
 output_file = OUTPUT_DIR / 'figure_02_multi_element_correlation.png'
 plt.savefig(output_file, dpi=150, bbox_inches='tight')
 logger.info(f"Saved: {output_file.name}")
 plt.close()

except Exception as e:
 logger.exception("Multi-element analysis error")

# ==============================================================================
# PART 4: Environmental Assessment - Arsenic
# ==============================================================================

logger.info("PART 4: ENVIRONMENTAL ASSESSMENT - ARSENIC (As)")

try:

try:
 as_chem = chem[chem['PARAMETER'].str.contains('As_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].copy()
 as_chem = as_chem.rename(columns={'DATA_VALUE': 'As'})
 logger.info(f"As: {len(as_chem):,} analyses")

 # Merge with locations
 as_data = geol.merge(as_chem, on='AGDB_ID', how='inner')
 as_data = as_data.dropna(subset=['LATITUDE', 'LONGITUDE', 'As'])
 as_data = as_data[as_data['As'] > 0]

 # Filter for Alaska coordinates only (exclude outliers)
 # Alaska proper: 51-72°N, -180 to -130°W
 as_data = as_data[
 (as_data['LATITUDE'] >= 51.0) & (as_data['LATITUDE'] <= 72.0) &
 (as_data['LONGITUDE'] >= -180.0) & (as_data['LONGITUDE'] <= -130.0)
 ]

 logger.info(f"Valid samples (Alaska only): {len(as_data):,}")

 # Extract values
 as_vals = as_data['As'].values

 # EPA threshold
 EPA_AS_THRESHOLD = 0.39 # ppm (very strict - residential)

 logger.info(f"Arsenic Statistics: {len(as_vals):,} samples")
 logger.info(f"  Mean: {as_vals.mean():.3f} ppm, Median: {np.median(as_vals):.3f} ppm, Max: {as_vals.max():.3f} ppm")
 logger.info(f"  EPA Threshold (residential): {EPA_AS_THRESHOLD} ppm")
 logger.info(f"  Exceeding EPA: {(as_vals > EPA_AS_THRESHOLD).sum():,} samples ({(as_vals > EPA_AS_THRESHOLD).mean()*100:.1f}%)")

 # Create arsenic distribution map
 logger.info("Creating arsenic distribution map...")
 fig, ax = plt.subplots(figsize=(12, 8))

 x_as = as_data['LONGITUDE'].values
 y_as = as_data['LATITUDE'].values

 scatter = ax.scatter(x_as, y_as, c=as_vals, s=15, cmap='RdYlGn_r',
 vmin=0, vmax=np.percentile(as_vals, 95),
 alpha=0.5, edgecolors='none')

 ax.set_title('Arsenic Distribution - Alaska\n(EPA Residential Threshold: 0.39 ppm)',
 fontsize=14, fontweight='bold')
 ax.set_xlabel('Longitude')
 ax.set_ylabel('Latitude')
 cbar = plt.colorbar(scatter, ax=ax, label='As (ppm)')

 plt.tight_layout()
 output_file = OUTPUT_DIR / 'figure_03_arsenic_distribution.png'
 plt.savefig(output_file, dpi=150, bbox_inches='tight')
 logger.info(f"Saved: {output_file.name}")
 plt.close()

except Exception as e:
 logger.exception("Arsenic analysis error")

# ==============================================================================
# SUMMARY
# ==============================================================================

logger.info("ANALYSIS SUMMARY")
logger.info("Completed Analyses: Gold exploration (Fairbanks), Multi-element correlation (Cu-Au), Environmental assessment (Arsenic)")

logger.info("Output Files Created:")
output_files = sorted(OUTPUT_DIR.glob('*'))
for f in output_files:
for f in output_files:

logger.info(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Full analysis complete! Results saved to: {OUTPUT_DIR}")
