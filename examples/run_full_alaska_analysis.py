"""
Alaska Geochemical Database (AGDB4) - Full Analysis
====================================================

This script runs comprehensive geostatistical analysis on AGDB4 data
and saves all results and figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Ensure output directory exists
OUTPUT_DIR = Path('/Users/k.jones/Desktop/geostats/alaska_outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Redirect stdout to file AND console
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open results file
results_file = open(OUTPUT_DIR / 'alaska_full_analysis_results.txt', 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, results_file)

print("=" * 80)
print("ALASKA GEOCHEMICAL DATABASE (AGDB4) - FULL ANALYSIS")
print("=" * 80)
print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# ==============================================================================
# PART 1: Data Loading and Exploration
# ==============================================================================

print("=" * 80)
print("PART 1: DATA LOADING AND EXPLORATION")
print("=" * 80)
print()

AGDB_PATH = Path('/Users/k.jones/Downloads/AGDB4_text')

if not AGDB_PATH.exists():
    print(f"ERROR: AGDB4 data not found at {AGDB_PATH}")
    print("Please ensure the Alaska Geochemical Database is available.")
    sys.exit(1)

print(f"✅ AGDB4 data found at: {AGDB_PATH}")
print()

# List data files
print("Available data files:")
data_files = sorted(AGDB_PATH.glob('*.txt'))
for file in data_files[:15]:
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"  {file.name:35s} {size_mb:>8.1f} MB")
print()

# Load location data
print("Loading sample location data...")
try:
    geol = pd.read_csv(AGDB_PATH / 'Geol_DeDuped.txt', low_memory=False, encoding='latin-1')
    print(f"✅ Loaded {len(geol):,} deduplicated samples")
    print(f"   Columns: {list(geol.columns[:15])}")
    
    # Geographic extent
    if 'LATITUDE' in geol.columns and 'LONGITUDE' in geol.columns:
        valid_coords = geol[(geol['LATITUDE'].notna()) & (geol['LONGITUDE'].notna())]
        print(f"\n   Geographic coverage:")
        print(f"     Samples with coordinates: {len(valid_coords):,}")
        print(f"     Latitude:  {valid_coords['LATITUDE'].min():.2f}° to {valid_coords['LATITUDE'].max():.2f}°N")
        print(f"     Longitude: {valid_coords['LONGITUDE'].min():.2f}° to {valid_coords['LONGITUDE'].max():.2f}°W")
    print()
    
except Exception as e:
    print(f"❌ Error loading location data: {e}")
    sys.exit(1)

# ==============================================================================
# PART 2: Gold Exploration Analysis
# ==============================================================================

print("=" * 80)
print("PART 2: GOLD EXPLORATION ANALYSIS - FAIRBANKS DISTRICT")
print("=" * 80)
print()

try:
    # Load gold chemistry
    print("Loading gold (Au) data...")
    chem_file = AGDB_PATH / 'Chem_A_Br.txt'
    chem = pd.read_csv(chem_file, low_memory=False, encoding='latin-1')
    
    # Filter for gold (PARAMETER contains 'Au')
    au_chem = chem[chem['PARAMETER'].str.contains('Au_', case=False, na=False)].copy()
    au_chem = au_chem.rename(columns={'DATA_VALUE': 'Au'})
    print(f"✅ Found {len(au_chem):,} gold analyses")
    
    # Merge with locations
    au_data = geol.merge(au_chem[['AGDB_ID', 'Au']], on='AGDB_ID', how='inner')
    au_data = au_data.dropna(subset=['LATITUDE', 'LONGITUDE', 'Au'])
    au_data = au_data[au_data['Au'] > 0]  # Remove non-detects
    
    # Focus on Fairbanks area (64-66°N, -149 to -145°W)
    fairbanks = au_data[
        (au_data['LATITUDE'] > 64.0) & (au_data['LATITUDE'] < 66.0) &
        (au_data['LONGITUDE'] > -149.0) & (au_data['LONGITUDE'] < -145.0)
    ].copy()
    
    print(f"   Fairbanks region: {len(fairbanks):,} samples")
    print()
    
    # Extract arrays
    x = fairbanks['LONGITUDE'].values
    y = fairbanks['LATITUDE'].values
    au = fairbanks['Au'].values
    
    # Statistics
    print("Gold Statistics (Fairbanks):")
    print(f"  Samples: {len(au):,}")
    print(f"  Mean: {au.mean():.4f} ppm")
    print(f"  Median: {np.median(au):.4f} ppm")
    print(f"  Std Dev: {au.std():.4f} ppm")
    print(f"  Min: {au.min():.4f} ppm")
    print(f"  Max: {au.max():.4f} ppm")
    print(f"  >0.1 ppm (100 ppb): {(au > 0.1).sum():,} samples ({(au > 0.1).mean()*100:.1f}%)")
    print(f"  >1.0 ppm: {(au > 1.0).sum():,} samples ({(au > 1.0).mean()*100:.1f}%)")
    print()
    
    # Create visualization
    print("Creating gold distribution map...")
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
    print(f"✅ Saved: {output_file.name}")
    plt.close()
    
except Exception as e:
    print(f"❌ Gold analysis error: {e}")
    import traceback
    traceback.print_exc()

print()

# ==============================================================================
# PART 3: Multi-Element Analysis
# ==============================================================================

print("=" * 80)
print("PART 3: MULTI-ELEMENT ANALYSIS - Cu, Mo, Au")
print("=" * 80)
print()

try:
    print("Loading multi-element data...")
    
    # Load copper data
    chem_c = pd.read_csv(AGDB_PATH / 'Chem_C_Gd.txt', low_memory=False, encoding='latin-1')
    cu_chem = chem_c[chem_c['PARAMETER'].str.contains('Cu_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].copy()
    cu_chem = cu_chem.rename(columns={'DATA_VALUE': 'Cu'})
    print(f"  Cu: {len(cu_chem):,} analyses")
    
    # Load gold data
    au_chem = chem[chem['PARAMETER'].str.contains('Au_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].copy()
    au_chem = au_chem.rename(columns={'DATA_VALUE': 'Au'})
    print(f"  Au: {len(au_chem):,} analyses")
    
    # Merge all
    multi_data = geol.merge(cu_chem, on='AGDB_ID', how='inner')
    multi_data = multi_data.merge(au_chem, on='AGDB_ID', how='inner')
    multi_data = multi_data.dropna(subset=['LATITUDE', 'LONGITUDE', 'Cu', 'Au'])
    multi_data = multi_data[(multi_data['Cu'] > 0) & (multi_data['Au'] > 0)]
    
    print(f"  Combined dataset: {len(multi_data):,} samples with both Cu and Au")
    print()
    
    # Calculate correlation
    from scipy import stats
    cu_vals = multi_data['Cu'].values
    au_vals = multi_data['Au'].values
    
    # Log-transform for better correlation
    cu_log = np.log10(cu_vals + 1)
    au_log = np.log10(au_vals + 0.001)
    
    corr_raw = stats.pearsonr(cu_vals, au_vals)[0]
    corr_log = stats.pearsonr(cu_log, au_log)[0]
    
    print("Element Correlations:")
    print(f"  Cu-Au (raw):        r = {corr_raw:.3f}")
    print(f"  Cu-Au (log-trans):  r = {corr_log:.3f}")
    print()
    
    # Create correlation plot
    print("Creating multi-element correlation plot...")
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
    print(f"✅ Saved: {output_file.name}")
    plt.close()
    
except Exception as e:
    print(f"❌ Multi-element analysis error: {e}")
    import traceback
    traceback.print_exc()

print()

# ==============================================================================
# PART 4: Environmental Assessment - Arsenic
# ==============================================================================

print("=" * 80)
print("PART 4: ENVIRONMENTAL ASSESSMENT - ARSENIC (As)")
print("=" * 80)
print()

try:
    print("Loading arsenic data...")
    
    # Load arsenic from Chem_A_Br
    as_chem = chem[chem['PARAMETER'].str.contains('As_', case=False, na=False)][['AGDB_ID', 'DATA_VALUE']].copy()
    as_chem = as_chem.rename(columns={'DATA_VALUE': 'As'})
    print(f"  As: {len(as_chem):,} analyses")
    
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
    
    print(f"  Valid samples (Alaska only): {len(as_data):,}")
    print()
    
    # Extract values
    as_vals = as_data['As'].values
    
    # EPA threshold
    EPA_AS_THRESHOLD = 0.39  # ppm (very strict - residential)
    
    print("Arsenic Statistics:")
    print(f"  Samples: {len(as_vals):,}")
    print(f"  Mean: {as_vals.mean():.3f} ppm")
    print(f"  Median: {np.median(as_vals):.3f} ppm")
    print(f"  Max: {as_vals.max():.3f} ppm")
    print(f"  EPA Threshold (residential): {EPA_AS_THRESHOLD} ppm")
    print(f"  Exceeding EPA: {(as_vals > EPA_AS_THRESHOLD).sum():,} samples ({(as_vals > EPA_AS_THRESHOLD).mean()*100:.1f}%)")
    print()
    
    # Create arsenic distribution map
    print("Creating arsenic distribution map...")
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
    print(f"✅ Saved: {output_file.name}")
    plt.close()
    
except Exception as e:
    print(f"❌ Arsenic analysis error: {e}")
    import traceback
    traceback.print_exc()

print()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print()

print("Completed Analyses:")
print("  ✅ Gold exploration (Fairbanks district)")
print("  ✅ Multi-element correlation (Cu-Au)")
print("  ✅ Environmental assessment (Arsenic)")
print()

print("Output Files Created:")
output_files = sorted(OUTPUT_DIR.glob('*'))
for f in output_files:
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:50s} {size_kb:>8.1f} KB")
print()

print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("=" * 80)
print("ALASKA GEOCHEMICAL ANALYSIS COMPLETE!")
print("=" * 80)
print()

print("Next Steps:")
print("  • Review figures in:", OUTPUT_DIR)
print("  • Use notebooks for interactive analysis")
print("  • Run advanced geostatistical workflows (kriging, cokriging)")
print()

# Restore stdout and close file
sys.stdout = original_stdout
results_file.close()

print(f"\n✅ Full analysis complete! Results saved to:")
print(f"   {OUTPUT_DIR}")
