#!/usr/bin/env python3
"""
Demonstration of Minimalist Plotting Style

Before and After examples showing the transformation from cluttered to clean plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/k.jones/Desktop/geostats/src')

from geostats.visualization.minimal_style import apply_minimalist_style, set_minimalist_rcparams

# Generate sample data
np.random.seed(42)
lags = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
gamma = np.array([0.1, 0.3, 0.6, 0.85, 0.95, 1.0, 1.02, 1.0, 1.01, 1.0, 0.99])
n_pairs = np.array([100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 15])

# Model line
h_model = np.linspace(0, 50, 200)
gamma_model = 1.0 * (1 - np.exp(-3 * h_model / 25))

print("=" * 70)
print("MINIMALIST PLOTTING STYLE DEMONSTRATION")
print("=" * 70)
print()

# EXAMPLE 1: Variogram Plot - Before and After
print("Example 1: Variogram Plot")
print("-" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# BEFORE: Cluttered style
sizes = n_pairs / np.max(n_pairs) * 100 + 20
ax1.scatter(lags, gamma, s=sizes, c='black', edgecolors='black', linewidth=2, label='Experimental')
ax1.plot(h_model, gamma_model, 'r-', linewidth=3, label='Exponential Model')
ax1.set_xlabel('Distance (h)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Semivariance γ(h)', fontsize=14, fontweight='bold')
ax1.set_title('Experimental Variogram', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.5, linestyle='--', linewidth=1)
ax1.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='black', shadow=True)

ax1.text(0.5, -0.15, 'BEFORE: Cluttered', transform=ax1.transAxes,
         fontsize=12, ha='center', fontweight='bold', color='red')

# AFTER: Minimalist style
sizes = n_pairs / np.max(n_pairs) * 100 + 20
ax2.scatter(lags, gamma, s=sizes, c='#1f77b4', alpha=0.6,
           edgecolors='#333333', linewidth=0.8, label='Experimental')
ax2.plot(h_model, gamma_model, '#d62728', linewidth=2, label='Exponential')
apply_minimalist_style(ax2)
ax2.set_title('Semivariance γ(h) vs Distance (h)', fontsize=11)
ax2.legend(fontsize=9, frameon=False)

ax2.text(0.5, -0.15, 'AFTER: Clean & Minimalist', transform=ax2.transAxes,
         fontsize=12, ha='center', fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('/Users/k.jones/Desktop/geostats/demo_variogram_comparison.png', 
            dpi=150, bbox_inches='tight')
print("✓ Saved: demo_variogram_comparison.png")
plt.close()

# EXAMPLE 2: Cross-validation Plot
print("\nExample 2: Cross-Validation Plot")
print("-" * 70)

np.random.seed(123)
y_true = np.random.uniform(0, 100, 50)
y_pred = y_true + np.random.normal(0, 10, 50)
r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# BEFORE
ax1.scatter(y_true, y_pred, s=80, c='blue', edgecolors='black', linewidth=2, alpha=0.7)
ax1.plot([0, 100], [0, 100], 'r-', linewidth=3, label='1:1 Line')
ax1.set_xlabel('True Values', fontsize=14, fontweight='bold')
ax1.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
ax1.set_title('Cross-Validation Results', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.5, which='both', linestyle='--')
ax1.legend(fontsize=12, frameon=True, shadow=True, loc='upper left')
ax1.set_aspect('equal')
ax1.text(0.5, -0.12, 'BEFORE: Cluttered', transform=ax1.transAxes,
         fontsize=12, ha='center', fontweight='bold', color='red')

# AFTER
ax2.scatter(y_true, y_pred, s=50, c='#1f77b4', alpha=0.5,
           edgecolors='#333333', linewidth=0.5)
ax2.plot([0, 100], [0, 100], '#d62728', linestyle='--', linewidth=1.5, label='1:1')
apply_minimalist_style(ax2)
ax2.set_title(f'Predicted vs Observed (ppm) — R² = {r2:.3f}', fontsize=11)
ax2.set_aspect('equal')
ax2.legend(fontsize=9, frameon=False, loc='upper left')
ax2.text(0.5, -0.12, 'AFTER: Clean & Minimalist', transform=ax2.transAxes,
         fontsize=12, ha='center', fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('/Users/k.jones/Desktop/geostats/demo_crossval_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: demo_crossval_comparison.png")
plt.close()

# EXAMPLE 3: Histogram
print("\nExample 3: Histogram Plot")
print("-" * 70)

np.random.seed(456)
data = np.random.lognormal(3, 0.5, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# BEFORE
ax1.hist(data, bins=25, color='blue', edgecolor='black', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Gold Concentration (ppm)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax1.set_title('Histogram of Gold Concentrations', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.5, axis='y', linestyle='--')
ax1.axvline(np.mean(data), color='red', linestyle='--', linewidth=3, label=f'Mean: {np.mean(data):.1f}')
ax1.legend(fontsize=12, frameon=True, shadow=True)
ax1.text(0.5, -0.15, 'BEFORE: Cluttered', transform=ax1.transAxes,
         fontsize=12, ha='center', fontweight='bold', color='red')

# AFTER
ax2.hist(data, bins=25, color='#1f77b4', edgecolor='#333333', linewidth=0.8, alpha=0.7)
apply_minimalist_style(ax2)
ax2.axvline(np.mean(data), color='#d62728', linestyle='--', linewidth=1.5, 
           label=f'Mean: {np.mean(data):.1f} ppm')
ax2.set_title('Distribution of Gold Concentration (ppm) — n=200', fontsize=11)
ax2.legend(fontsize=9, frameon=False)
ax2.text(0.5, -0.15, 'AFTER: Clean & Minimalist', transform=ax2.transAxes,
         fontsize=12, ha='center', fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('/Users/k.jones/Desktop/geostats/demo_histogram_comparison.png',
            dpi=150, bbox_inches='tight')
print("✓ Saved: demo_histogram_comparison.png")
plt.close()

print()
print("=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
print()
print("Key Improvements:")
print("  ✓ No gridlines (cleaner look)")
print("  ✓ Removed top and right spines (less clutter)")
print("  ✓ Descriptive titles (no need for axis labels)")
print("  ✓ Muted colors (#1f77b4 blue, #d62728 red)")
print("  ✓ Thinner lines and edges (more elegant)")
print("  ✓ Frameless legends (integrated, not boxed)")
print("  ✓ Smaller, normal-weight fonts (easier to read)")
print()
print("All plots saved to:")
print("  - demo_variogram_comparison.png")
print("  - demo_crossval_comparison.png")
print("  - demo_histogram_comparison.png")
print()
