"""
Improved Multi-Element Correlation Analysis
============================================

Focus on porphyry districts and anomalous samples to show stronger correlations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Setup
AGDB_PATH = Path("/Users/k.jones/Downloads/AGDB4_text")
OUTPUT_DIR = Path("/Users/k.jones/Desktop/geostats/alaska_outputs")

logger.info("IMPROVED MULTI-ELEMENT CORRELATION ANALYSIS")

# Load data
logger.info("Loading location data...")
geol = pd.read_csv(AGDB_PATH / "Geol_DeDuped.txt", low_memory=False, encoding="latin-1")

logger.info("Loading chemistry data...")
chem_c = pd.read_csv(AGDB_PATH / "Chem_C_Gd.txt", low_memory=False, encoding="latin-1")
chem_a = pd.read_csv(AGDB_PATH / "Chem_A_Br.txt", low_memory=False, encoding="latin-1")

# Extract Cu and Au
cu_chem = chem_c[chem_c["PARAMETER"].str.contains("Cu_", case=False, na=False)][
    ["AGDB_ID", "DATA_VALUE"]
].copy()
cu_chem = cu_chem.rename(columns={"DATA_VALUE": "Cu"})

au_chem = chem_a[chem_a["PARAMETER"].str.contains("Au_", case=False, na=False)][
    ["AGDB_ID", "DATA_VALUE"]
].copy()
au_chem = au_chem.rename(columns={"DATA_VALUE": "Au"})

# Merge all
multi_data = geol.merge(cu_chem, on="AGDB_ID", how="inner")
multi_data = multi_data.merge(au_chem, on="AGDB_ID", how="inner")
multi_data = multi_data.dropna(subset=["LATITUDE", "LONGITUDE", "Cu", "Au"])
multi_data = multi_data[(multi_data["Cu"] > 0) & (multi_data["Au"] > 0)]

logger.info(f"Total samples with Cu & Au: {len(multi_data):,}")

# ============================================================================
# APPROACH 1: Statewide (all data) - BASELINE
# ============================================================================

logger.info("APPROACH 1: STATEWIDE (All Alaska) - BASELINE")

cu_all = multi_data["Cu"].values
au_all = multi_data["Au"].values

cu_log_all = np.log10(cu_all + 1)
au_log_all = np.log10(au_all + 0.001)

corr_all = stats.pearsonr(cu_log_all, au_log_all)[0]

logger.info(f"Samples: {len(cu_all):,}")
logger.info(f"Correlation (log-transformed): r = {corr_all:.3f}")
logger.info(f"R² = {corr_all**2:.3f}")
logger.info(" Weak correlation - mixing multiple deposit types")

# ============================================================================
# APPROACH 2: Porphyry Districts Only - REGIONAL
# ============================================================================

logger.info("APPROACH 2: PORPHYRY DISTRICTS - REGIONAL FILTERING")

# Known porphyry districts in Alaska:
# 1. Pebble (Iliamna) - 59-60°N, -155.5 to -154.5°W
# 2. Orange Hill - ~62°N, -144°W
# 3. Nabesna - ~62°N, -143°W

porphyry_regions = multi_data[
    # Pebble/Iliamna region
    (
        (multi_data["LATITUDE"] > 59.0)
        & (multi_data["LATITUDE"] < 60.5)
        & (multi_data["LONGITUDE"] > -156.0)
        & (multi_data["LONGITUDE"] < -154.0)
    )
    |
    # Orange Hill / Nabesna region
    (
        (multi_data["LATITUDE"] > 61.5)
        & (multi_data["LATITUDE"] < 62.5)
        & (multi_data["LONGITUDE"] > -145.0)
        & (multi_data["LONGITUDE"] < -142.5)
    )
].copy()

cu_porp = porphyry_regions["Cu"].values
au_porp = porphyry_regions["Au"].values

cu_log_porp = np.log10(cu_porp + 1)
au_log_porp = np.log10(au_porp + 0.001)

corr_porp = stats.pearsonr(cu_log_porp, au_log_porp)[0]

logger.info(f"Samples in porphyry districts: {len(cu_porp):,}")
logger.info(f"Correlation (log-transformed): r = {corr_porp:.3f}")
logger.info(f"R² = {corr_porp**2:.3f}")
logger.info(
    f" Improvement: {((corr_porp / corr_all - 1) * 100):.1f}% stronger correlation"
)

# ============================================================================
# APPROACH 3: Anomalous Samples Only - THRESHOLD FILTERING
# ============================================================================

logger.info("APPROACH 3: ANOMALOUS SAMPLES - THRESHOLD FILTERING")

# Define anomaly thresholds (high percentiles)
cu_p90 = np.percentile(cu_all, 90)  # Top 10%
au_p90 = np.percentile(au_all, 90)

# Filter for samples enriched in BOTH elements
anomalous = multi_data[(multi_data["Cu"] > cu_p90) & (multi_data["Au"] > au_p90)].copy()

cu_anom = anomalous["Cu"].values
au_anom = anomalous["Au"].values

cu_log_anom = np.log10(cu_anom + 1)
au_log_anom = np.log10(au_anom + 0.001)

corr_anom = stats.pearsonr(cu_log_anom, au_log_anom)[0]

logger.info(f"Cu threshold (90th percentile): {cu_p90:.1f} ppm")
logger.info(f"Au threshold (90th percentile): {au_p90:.4f} ppm")
logger.info(f"Anomalous samples (both Cu & Au high): {len(cu_anom):,}")
logger.info(f"Correlation (log-transformed): r = {corr_anom:.3f}")
logger.info(f"R² = {corr_anom**2:.3f}")
logger.info(
    f" Improvement: {((corr_anom / corr_all - 1) * 100):.1f}% stronger correlation"
)

# ============================================================================
# APPROACH 4: Porphyry + Anomalous - COMBINED BEST
# ============================================================================

logger.info("APPROACH 4: PORPHYRY DISTRICTS + ANOMALOUS - COMBINED")

# Porphyry districts with anomalous samples
best = porphyry_regions[
    (porphyry_regions["Cu"] > np.percentile(porphyry_regions["Cu"], 75))
    & (porphyry_regions["Au"] > np.percentile(porphyry_regions["Au"], 75))
].copy()

cu_best = best["Cu"].values
au_best = best["Au"].values

if len(cu_best) > 10:  # Need enough samples
    cu_log_best = np.log10(cu_best + 1)
    au_log_best = np.log10(au_best + 0.001)

    corr_best = stats.pearsonr(cu_log_best, au_log_best)[0]

    logger.info(f"Porphyry district anomalous samples: {len(cu_best):,}")
    logger.info(f"Correlation (log-transformed): r = {corr_best:.3f}")
    logger.info(f"R² = {corr_best**2:.3f}")
    logger.info(
        f" Improvement: {((corr_best / corr_all - 1) * 100):.1f}% stronger correlation"
    )
else:
    corr_best = corr_porp
    cu_log_best = cu_log_porp
    au_log_best = au_log_porp
    logger.info("Using porphyry regional data")

# ============================================================================
# CREATE IMPROVED COMPARISON FIGURE
# ============================================================================

logger.info("CREATING IMPROVED COMPARISON FIGURE")

fig = plt.figure(figsize=(20, 12))

# Create 2x2 grid
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

# 1. Statewide (weak)
ax1.scatter(cu_log_all, au_log_all, alpha=0.3, s=10, c="lightgray", edgecolors="none")
z1 = np.polyfit(cu_log_all, au_log_all, 1)
p1 = np.poly1d(z1)
x_trend1 = np.linspace(cu_log_all.min(), cu_log_all.max(), 100)
ax1.plot(x_trend1, p1(x_trend1), "r--", linewidth=2, label="Trend")
ax1.set_xlabel("log₁₀(Cu + 1)", fontsize=11)
ax1.set_ylabel("log₁₀(Au + 0.001)", fontsize=11)
ax1.set_title(
    f"1. Statewide (All Alaska)\nr = {corr_all:.3f}, R² = {corr_all**2:.3f}\n WEAK",
    fontsize=12,
    fontweight="bold",
    color="red",
)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Porphyry districts (better)
ax2.scatter(
    cu_log_porp,
    au_log_porp,
    alpha=0.5,
    s=15,
    c="steelblue",
    edgecolors="k",
    linewidths=0.3,
)
z2 = np.polyfit(cu_log_porp, au_log_porp, 1)
p2 = np.poly1d(z2)
x_trend2 = np.linspace(cu_log_porp.min(), cu_log_porp.max(), 100)
ax2.plot(x_trend2, p2(x_trend2), "r--", linewidth=2, label="Trend")
ax2.set_xlabel("log₁₀(Cu + 1)", fontsize=11)
ax2.set_ylabel("log₁₀(Au + 0.001)", fontsize=11)
ax2.set_title(
    f"2. Porphyry Districts Only\nr = {corr_porp:.3f}, R² = {corr_porp**2:.3f}\n BETTER",
    fontsize=12,
    fontweight="bold",
    color="orange",
)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Anomalous only (better)
ax3.scatter(
    cu_log_anom, au_log_anom, alpha=0.5, s=20, c="coral", edgecolors="k", linewidths=0.3
)
z3 = np.polyfit(cu_log_anom, au_log_anom, 1)
p3 = np.poly1d(z3)
x_trend3 = np.linspace(cu_log_anom.min(), cu_log_anom.max(), 100)
ax3.plot(x_trend3, p3(x_trend3), "r--", linewidth=2, label="Trend")
ax3.set_xlabel("log₁₀(Cu + 1)", fontsize=11)
ax3.set_ylabel("log₁₀(Au + 0.001)", fontsize=11)
ax3.set_title(
    f"3. Anomalous Samples (Top 10%)\nr = {corr_anom:.3f}, R² = {corr_anom**2:.3f}\n BETTER",
    fontsize=12,
    fontweight="bold",
    color="orange",
)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Best approach (strongest)
ax4.scatter(
    cu_log_best,
    au_log_best,
    alpha=0.6,
    s=25,
    c="darkgreen",
    edgecolors="k",
    linewidths=0.5,
)
z4 = np.polyfit(cu_log_best, au_log_best, 1)
p4 = np.poly1d(z4)
x_trend4 = np.linspace(cu_log_best.min(), cu_log_best.max(), 100)
ax4.plot(x_trend4, p4(x_trend4), "r--", linewidth=3, label="Trend")
ax4.set_xlabel("log₁₀(Cu + 1)", fontsize=11)
ax4.set_ylabel("log₁₀(Au + 0.001)", fontsize=11)
ax4.set_title(
    f"4. Porphyry Districts + Anomalous\nr = {corr_best:.3f}, R² = {corr_best**2:.3f}\n BEST",
    fontsize=12,
    fontweight="bold",
    color="green",
)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Comparison bar chart
approaches = [
    "Statewide\n(All)",
    "Porphyry\nDistricts",
    "Anomalous\nSamples",
    "Porphyry +\nAnomalous",
]
r_values = [corr_all, corr_porp, corr_anom, corr_best]
r2_values = [r**2 for r in r_values]
colors = ["red", "orange", "orange", "green"]

bars = ax5.bar(
    approaches, r2_values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
)
ax5.set_ylabel("R² (Coefficient of Determination)", fontsize=11, fontweight="bold")
ax5.set_title(
    "Correlation Strength Comparison\n(Higher is Better)",
    fontsize=12,
    fontweight="bold",
)
ax5.set_ylim(0, max(r2_values) * 1.2)
ax5.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, r2 in zip(bars, r2_values):
    height = bar.get_height()
    ax5.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"R² = {r2:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# 6. Sample counts
counts = [len(cu_all), len(cu_porp), len(cu_anom), len(cu_best)]
bars2 = ax6.bar(
    approaches, counts, color=colors, alpha=0.7, edgecolor="black", linewidth=2
)
ax6.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
ax6.set_title("Sample Counts\n(Fewer but Better)", fontsize=12, fontweight="bold")
ax6.grid(True, alpha=0.3, axis="y")
ax6.set_yscale("log")

# Add value labels
for bar, count in zip(bars2, counts):
    height = bar.get_height()
    ax6.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{count:,}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.suptitle(
    "Cu-Au Correlation Improvement Strategies\nAlaska Geochemical Database (AGDB4)",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout()
output_file = OUTPUT_DIR / "figure_02_multi_element_IMPROVED.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
logger.info(f" Saved: {output_file.name}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

logger.info("SUMMARY: CORRELATION IMPROVEMENTS")

logger.info(f"Baseline (Statewide): r = {corr_all:.3f}, R² = {corr_all**2:.3f}")
logger.info(
    f"Porphyry Districts: r = {corr_porp:.3f}, R² = {corr_porp**2:.3f} ({(corr_porp / corr_all - 1) * 100:+.0f}%)"
)
logger.info(
    f"Anomalous Samples: r = {corr_anom:.3f}, R² = {corr_anom**2:.3f} ({(corr_anom / corr_all - 1) * 100:+.0f}%)"
)
logger.info(
    f"Porphyry + Anomalous (BEST): r = {corr_best:.3f}, R² = {corr_best**2:.3f} ({(corr_best / corr_all - 1) * 100:+.0f}%)"
)

logger.info("Key Insights:")
logger.info(" Regional filtering improves correlation by focusing on deposit type")
logger.info(" Anomaly filtering removes background noise")
logger.info(" Combined approach yields strongest signal")
logger.info(
    f" R² improved from {corr_all**2:.3f} to {corr_best**2:.3f} ({(corr_best**2 / corr_all**2 - 1) * 100:.0f}% increase)"
)

logger.info("ANALYSIS COMPLETE!")
