# Minimalist Plotting Implementation Summary

**Date**: January 22, 2026  
**Purpose**: Clean, professional visualization style

---

## What Was Done

### 1. Created Minimalist Style Module ✓

**File**: `src/geostats/visualization/minimal_style.py`

Provides:
- `apply_minimalist_style(ax)` - Apply clean style to any matplotlib axes
- `set_minimalist_rcparams()` - Set as default for all plots
- `create_minimalist_figure()` - Create pre-styled figures

### 2. Updated Visualization Module ✓

**File**: `src/geostats/visualization/variogram_plots.py`

Updated `plot_variogram()` function to use:
- No gridlines
- No top/right spines
- Descriptive titles
- Muted color palette (#1f77b4 blue, #d62728 red)
- Clean legends (no frames)

### 3. Created Style Guide ✓

**File**: `PLOTTING_STYLE_GUIDE.md`

Comprehensive 400+ line guide covering:
- Philosophy and principles
- Color palette
- Typography guidelines
- Title best practices
- Complete examples
- Migration checklist

### 4. Generated Demos ✓

**Files**: 
- `demo_minimalist_plots.py` - Demonstration script
- `demo_variogram_comparison.png` - Before/after variogram
- `demo_crossval_comparison.png` - Before/after cross-validation
- `demo_histogram_comparison.png` - Before/after histogram

---

## Key Style Changes

| Element | Old Style | New Style |
|---------|-----------|-----------|
| **Gridlines** | `grid(True, alpha=0.5)` | `grid(False)` |
| **Spines** | All 4 visible | Only bottom & left |
| **Title** | "Experimental Variogram" (size 14, bold) | "Semivariance γ(h) vs Distance (h)" (size 11, normal) |
| **Axis Labels** | Large, bold labels | Omitted (info in title) |
| **Colors** | Black, red | #1f77b4 (blue), #d62728 (red) |
| **Legend** | Framed box with shadow | Frameless, integrated |
| **Line Widths** | 2-3 | 0.8-2.0 (thinner) |

---

## Usage

### Quick Start

```python
from geostats.visualization.minimal_style import apply_minimalist_style

# Your existing plot code
fig, ax = plt.subplots()
ax.scatter(x, y, c='#1f77b4', edgecolors='#333333', linewidth=0.8)

# Apply minimalist style (one line!)
apply_minimalist_style(ax)

# Descriptive title
ax.set_title('Gold Concentration (ppm) vs Distance from Source (km)', fontsize=11)
```

### Set as Default

```python
from geostats.visualization.minimal_style import set_minimalist_rcparams

# At the start of your script
set_minimalist_rcparams()

# Now ALL plots will use minimalist style automatically
fig, ax = plt.subplots()
ax.plot(x, y)  # Already has clean style!
```

---

## Color Palette

```python
# Use these consistently
BLUE = '#1f77b4'      # Primary data points
RED = '#d62728'       # Model fits, highlights
GRAY = '#333333'      # Edges, text
LIGHT_GRAY = '#f0f0f0'  # Backgrounds

# Example
ax.scatter(x, y, c=BLUE, edgecolors=GRAY, linewidth=0.8)
ax.plot(x, model, color=RED, linewidth=1.5)
```

---

## Before & After Examples

### Variogram Plot

**Before** (Cluttered):
- Heavy gridlines
- All 4 spines visible
- Bold, large fonts
- Generic title: "Experimental Variogram"
- Boxed legend with shadow
- Heavy line weights

**After** (Clean):
- No gridlines
- Only bottom/left spines  
- Clean, readable fonts
- Descriptive title: "Semivariance γ(h) vs Distance (h)"
- Frameless legend
- Elegant line weights

*See `demo_variogram_comparison.png` for visual comparison*

---

## Migration Guide

To update existing visualization code:

### Step 1: Add Import
```python
from geostats.visualization.minimal_style import apply_minimalist_style
```

### Step 2: Update Colors
```python
# Old
ax.scatter(x, y, c='black', edgecolors='black', linewidth=2)

# New  
ax.scatter(x, y, c='#1f77b4', edgecolors='#333333', linewidth=0.8)
```

### Step 3: Apply Style
```python
# Add this line
apply_minimalist_style(ax)
```

### Step 4: Update Title
```python
# Old
ax.set_xlabel('Distance (h)', fontsize=12)
ax.set_ylabel('Semivariance γ(h)', fontsize=12)
ax.set_title('Experimental Variogram', fontsize=14, fontweight='bold')

# New (descriptive title, no labels)
ax.set_title('Semivariance γ(h) vs Distance (h)', fontsize=11)
```

### Step 5: Clean Legend
```python
# Old
ax.legend(fontsize=10, frameon=True, shadow=True)

# New
ax.legend(fontsize=9, frameon=False)
```

---

## Files to Update

Still need to apply minimalist style to:

1. `spatial_plots.py` - Spatial visualization functions
2. `diagnostic_plots.py` - Diagnostic plots  
3. `enhanced.py` - Enhanced visualization functions
4. Examples in `examples/` directory
5. Notebook plots in `notebooks/`

**Pattern to follow**: Use `demo_minimalist_plots.py` as template

---

## Benefits

### Visual
- ✓ Cleaner, more professional appearance
- ✓ Reduced visual clutter
- ✓ Better focus on data
- ✓ Journal/publication ready

### Practical
- ✓ Easier to read
- ✓ Works better in presentations
- ✓ Prints clearly in black & white
- ✓ Consistent across all plots

### Scientific
- ✓ Follows Tufte's principles (minimize data-ink ratio)
- ✓ Aligns with scientific visualization best practices
- ✓ Professional standard in statistics

---

## Testing

Run the demo to see before/after:

```bash
cd /Users/k.jones/Desktop/geostats
python demo_minimalist_plots.py
```

This generates three comparison images showing the transformation.

---

## Quick Reference

### Essential Functions

```python
# Apply to existing axes
apply_minimalist_style(ax)

# Set as global default
set_minimalist_rcparams()

# Create pre-styled figure
fig, ax = create_minimalist_figure(figsize=(8, 6))
```

### Color Constants

```python
BLUE = '#1f77b4'
RED = '#d62728'
GRAY = '#333333'
LIGHT_GRAY = '#f0f0f0'
```

### Font Sizes

```python
TITLE_SIZE = 11
LABEL_SIZE = 9  # if needed
TICK_SIZE = 9
LEGEND_SIZE = 9
```

### Line Widths

```python
SPINE_WIDTH = 0.8
DATA_LINE_WIDTH = 1.5-2.0
EDGE_WIDTH = 0.8
```

---

## Examples in Action

### Variogram
```python
from geostats.visualization.minimal_style import apply_minimalist_style

fig, ax = plt.subplots()
ax.scatter(lags, gamma, c='#1f77b4', edgecolors='#333333', linewidth=0.8)
apply_minimalist_style(ax)
ax.set_title('Semivariance γ(h) vs Distance (h)', fontsize=11)
```

### Cross-Validation
```python
ax.scatter(y_true, y_pred, c='#1f77b4', alpha=0.5, edgecolors='#333333', linewidth=0.5)
ax.plot([0, 100], [0, 100], '#d62728', linestyle='--', linewidth=1.5)
apply_minimalist_style(ax)
ax.set_title(f'Predicted vs Observed — R² = {r2:.3f}', fontsize=11)
```

### Spatial Map
```python
contour = ax.contourf(X, Y, Z, cmap='viridis')
apply_minimalist_style(ax, remove_spines=('all',))  # No spines for maps
ax.set_title('Gold Concentration (ppm) Across Study Area', fontsize=11)
```

---

## Summary

Your plotting style has been transformed from cluttered to clean, professional visualizations that:

1. **Remove chart junk** - No unnecessary gridlines or spines
2. **Use descriptive titles** - Title explains the plot completely
3. **Apply muted colors** - Professional blue/red palette
4. **Maintain readability** - Clean fonts and appropriate sizes
5. **Follow best practices** - Aligned with Tufte and scientific standards

**Result**: Publication-ready, professional visualizations! ✓

---

**Documentation**:
- Full guide: `PLOTTING_STYLE_GUIDE.md`
- Demo script: `demo_minimalist_plots.py`
- Module: `src/geostats/visualization/minimal_style.py`

**Status**: ✓ Ready to use
