"""
Simple Alaska Geochemical Analysis - Test Run
==============================================

This is a simplified version that demonstrates the core functionality
without requiring the full AGDB4 dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Check if AGDB4 data exists
AGDB_PATH = Path('/Users/k.jones/Downloads/AGDB4_text')

print("=" * 70)
print("ALASKA GEOCHEMICAL ANALYSIS - TEST RUN")
print("=" * 70)
print()

if not AGDB_PATH.exists():
    print("⚠️  AGDB4 data not found at:", AGDB_PATH)
    print()
    print("Expected location: /Users/k.jones/Downloads/AGDB4_text")
    print()
    print("The Alaska Geochemical Database (AGDB4) contains:")
    print("  - 375,000+ geochemical samples")
    print("  - 70+ elements analyzed") 
    print("  - Stream sediments, rocks, soils")
    print("  - Lat/Long coordinates")
    print()
    print("To run this analysis, please ensure AGDB4 data is available.")
    print()
else:
    print("✅ AGDB4 data found at:", AGDB_PATH)
    print()
    
    # List available files
    print("Available data files:")
    for file in sorted(AGDB_PATH.glob('*.txt'))[:10]:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name:30s} ({size_mb:>6.1f} MB)")
    print()
    
    # Try to load location data
    try:
        geol_file = AGDB_PATH / 'Geol_DeDuped.txt'
        if geol_file.exists():
            print("Loading sample locations...")
            geol = pd.read_csv(geol_file, low_memory=False, nrows=1000, encoding='latin-1')
            print(f"  ✅ Loaded {len(geol):,} samples (preview)")
            print(f"  Columns: {list(geol.columns[:10])}")
            print()
            
            # Show coverage
            if 'LATITUDE' in geol.columns and 'LONGITUDE' in geol.columns:
                print("Geographic coverage:")
                print(f"  Latitude:  {geol['LATITUDE'].min():.2f} to {geol['LATITUDE'].max():.2f}°N")
                print(f"  Longitude: {geol['LONGITUDE'].min():.2f} to {geol['LONGITUDE'].max():.2f}°W")
                print()
        
        # Try to load chemistry data
        chem_file = AGDB_PATH / 'Chem_A_Br.txt'
        if chem_file.exists():
            print("Loading chemistry data...")
            chem = pd.read_csv(chem_file, low_memory=False, nrows=1000, encoding='latin-1')
            print(f"  ✅ Loaded {len(chem):,} analyses (preview)")
            
            if 'PARAMETER' in chem.columns:
                elements = chem['PARAMETER'].unique()
                print(f"  Elements available: {', '.join(elements[:20])}")
                print()
        
        print("=" * 70)
        print("DATA CHECK COMPLETE")
        print("=" * 70)
        print()
        print("The full analysis workflow includes:")
        print("  1. Loading and filtering AGDB4 data")
        print("  2. Variogram analysis and modeling")
        print("  3. Kriging predictions")
        print("  4. Multi-element cokriging")
        print("  5. Probability mapping")
        print("  6. Visualization")
        print()
        print("To run the full analysis, use alaska_geochemical_analysis.py")
        print()
        
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        print()

print("=" * 70)
print("TEST RUN COMPLETE")
print("=" * 70)
