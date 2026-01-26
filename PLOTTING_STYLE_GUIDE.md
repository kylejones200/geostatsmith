# Minimalist Plotting Style Guide

## Philosophy

Clean, professional visualizations that communicate clearly without visual clutter.

### Core Principles

1. **No chart junk** - Remove anything that doesn't add information
2. **No gridlines** - Unless absolutely necessary for reading values
3. **Remove top and right spines** - Only keep necessary axes
4. **Descriptive titles** - Title should explain what's being shown, eliminating need for axis labels
5. **Muted color palette** - Professional, not garish
6. **Clean typography** - Consistent, readable fonts

---

## Implementation

### Automatic Styling

Use the `minimal_style` module:

```python
from geostats.visualization.minimal_style import apply_minimalist_style, set_minimalist_rcparams

# Option 1: Apply to existing axes
fig, ax = plt.subplots()
ax.plot(x, y)
apply_minimalist_style(ax)  # Removes gridlines, top/right spines

# Option 2: Set as default for all plots
set_minimalist_rcparams()  # All future plots will use minimalist style
```

### Manual Styling

For custom plots:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# Your plot code
ax.scatter(x, y, c='#1f77b4', edgecolors='#333333', linewidth=0.8)

# Apply minimalist style
ax.grid(False)  # No gridlines
ax.spines['top'].set_visible(False)  # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['left'].set_linewidth(0.8)  # Thin remaining spines
ax.spines['bottom'].set_linewidth(0.8)

# Descriptive title instead of axis labels
ax.set_title('Gold Concentration (ppm) vs Distance from Source (km)', fontsize=11)

# If axis labels are truly needed (rare), use small font
# ax.set_xlabel('Distance (km)', fontsize=9)  # Only if not obvious from title
```

---

## Examples

### Before (Cluttered)

```python
fig, ax = plt.subplots()
ax.scatter(lags, gamma, s=80, c='black', edgecolors='black', linewidth=2)
ax.set_xlabel('Distance (h)', fontsize=14, fontweight='bold')
ax.set_ylabel('Semivariance γ(h)', fontsize=14, fontweight='bold')
ax.set_title('Experimental Variogram', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.5, linestyle='--')
ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='black')
```

### After (Clean)

```python
from geostats.visualization.minimal_style import apply_minimalist_style

fig, ax = plt.subplots()
ax.scatter(lags, gamma, s=80, c='#1f77b4', edgecolors='#333333', linewidth=0.8, alpha=0.6)
apply_minimalist_style(ax)
ax.set_title('Semivariance γ(h) vs Distance (h)', fontsize=11)
ax.legend(fontsize=9, frameon=False)
```

---

## Color Palette

Use muted, professional colors:

```python
# Primary colors
BLUE = '#1f77b4'      # Data points, main line
RED = '#d62728'       # Model fits, important highlights
GRAY = '#333333'      # Edges, text, spines
LIGHT_GRAY = '#f0f0f0'  # Backgrounds, boxes

# Secondary colors (use sparingly)
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'
PURPLE = '#9467bd'

# Example usage
ax.scatter(x, y, c=BLUE, edgecolors=GRAY, linewidth=0.8)
ax.plot(x, model, color=RED, linewidth=1.5)
```

---

## Title Guidelines

### Bad Titles (Too Generic)
- ❌ "Variogram"
- ❌ "Data Plot"
- ❌ "Results"

### Good Titles (Descriptive)
- ✅ "Semivariance γ(h) vs Distance (h)"
- ✅ "Gold Concentration Across Study Area"
- ✅ "Predicted vs Observed Au (ppm) - R² = 0.85"
- ✅ "Kriging Variance (km²) Showing Uncertainty"

The title should answer: **What is being shown?**

---

## When to Use Axis Labels

**Rarely!** Only when:

1. Units are not obvious from title
2. Multiple subplots share axes
3. Presenting to audience unfamiliar with the domain

If you must use axis labels:
- Keep them small (fontsize=9)
- Use standard symbols (γ, σ², h)
- No bold fonts

---

## Gridlines: When Are They OK?

Gridlines are acceptable ONLY when:

1. **Quantitative reading is critical** - e.g., reading specific values from plot
2. **Log scales** - Helps track orders of magnitude
3. **Complex multi-panel plots** - Where alignment matters

If using gridlines:
```python
ax.grid(True, alpha=0.2, color='#CCCCCC', linewidth=0.5, zorder=0)
# Always: subtle (alpha=0.2), behind data (zorder=0)
```

---

## Legend Best Practices

```python
# Minimalist legend
ax.legend(
    fontsize=9,
    frameon=False,  # No box
    loc='best',     # Auto position
    markerscale=0.8  # Slightly smaller markers
)

# If you need a box (rare):
ax.legend(
    fontsize=9,
    frameon=True,
    edgecolor='none',  # No border
    facecolor='#f8f8f8',  # Subtle background
    framealpha=0.9
)
```

---

## Common Patterns

### Variogram Plot

```python
from geostats.visualization.minimal_style import apply_minimalist_style

fig, ax = plt.subplots(figsize=(8, 6))

# Experimental points
ax.scatter(lags, gamma, s=sizes, c='#1f77b4', alpha=0.6,
           edgecolors='#333333', linewidth=0.8, label='Experimental')

# Model fit
ax.plot(h_model, gamma_model, '#d62728', linewidth=2, label='Spherical')

apply_minimalist_style(ax)
ax.set_title('Semivariance γ(h) vs Distance (h)', fontsize=11)
ax.legend(fontsize=9, frameon=False)
```

### Cross-Validation Plot

```python
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(y_true, y_pred, c='#1f77b4', alpha=0.5, s=50,
           edgecolors='#333333', linewidth=0.5)
ax.plot([ymin, ymax], [ymin, ymax], '#d62728', linestyle='--', 
        linewidth=1.5, label='1:1 Line')

apply_minimalist_style(ax)
ax.set_title(f'Predicted vs Observed Au (ppm) - R² = {r2:.3f}', fontsize=11)
ax.set_aspect('equal')
ax.legend(fontsize=9, frameon=False)
```

### Spatial Map

```python
fig, ax = plt.subplots(figsize=(10, 8))

contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.9)
ax.scatter(x_data, y_data, c='white', s=30, edgecolors='#333333',
           linewidth=0.8, zorder=5)

apply_minimalist_style(ax, remove_spines=('all',))  # Remove all spines for maps
ax.set_title('Gold Concentration (ppm) Across Study Area', fontsize=11)
ax.set_aspect('equal')

cbar = plt.colorbar(contour, ax=ax)
cbar.outline.set_visible(False)  # Remove colorbar box
```

---

## Typography

```python
# Sizes
TITLE_SIZE = 11
LABEL_SIZE = 9
TICK_SIZE = 9
LEGEND_SIZE = 9

# Weights (avoid bold)
TITLE_WEIGHT = 'normal'  # Not 'bold'

# Family
FONT_FAMILY = ['Arial', 'DejaVu Sans', 'Liberation Sans']  # Sans-serif
```

---

## Quick Reference

| Element | Style |
|---------|-------|
| Gridlines | `False` (or very subtle if needed) |
| Top spine | `False` |
| Right spine | `False` |
| Title font size | `11` |
| Title font weight | `normal` (not bold) |
| Axis label font size | `9` (if used) |
| Tick label size | `9` |
| Legend font size | `9` |
| Legend frame | `False` |
| Data point color | `#1f77b4` (blue) |
| Model line color | `#d62728` (red) |
| Edge color | `#333333` (dark gray) |
| Line width | `0.8` (spines), `1.5-2.0` (data lines) |

---

## Testing Your Plot

Ask yourself:

1. ✓ Can I understand what's plotted without axis labels?
2. ✓ Are the spines minimal (only bottom and left)?
3. ✓ Are there no gridlines (or very subtle ones)?
4. ✓ Is the title descriptive and informative?
5. ✓ Are the colors professional and muted?
6. ✓ Is text readable but not overwhelming?
7. ✓ Could this appear in a scientific journal?

If yes to all: **Good to go!** ✓

---

## Migration Checklist

For existing code:

- [ ] Add `from geostats.visualization.minimal_style import apply_minimalist_style`
- [ ] Replace `ax.grid(True)` with `ax.grid(False)` or remove
- [ ] Remove `ax.spines['top'].set_visible(True)` calls
- [ ] Remove `ax.spines['right'].set_visible(True)` calls
- [ ] Update titles to be descriptive
- [ ] Remove or minimize axis labels
- [ ] Change `fontweight='bold'` to `fontweight='normal'`
- [ ] Update colors to muted palette
- [ ] Set `legend(frameon=False)`
- [ ] Add `apply_minimalist_style(ax)` before returning

---

## Complete Example

```python
from geostats.visualization.minimal_style import apply_minimalist_style
import matplotlib.pyplot as plt
import numpy as np

# Generate example data
x = np.linspace(0, 100, 50)
y = 10 + 2*x + np.random.normal(0, 10, 50)

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot data
ax.scatter(x, y, c='#1f77b4', alpha=0.6, s=60, 
           edgecolors='#333333', linewidth=0.8, label='Data')

# Fit line
coeffs = np.polyfit(x, y, 1)
y_fit = coeffs[0] * x + coeffs[1]
ax.plot(x, y_fit, '#d62728', linewidth=2, label=f'Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.1f}')

# Apply minimalist style
apply_minimalist_style(ax)

# Descriptive title (no axis labels needed)
ax.set_title('Gold Grade (ppm) vs Distance from Mineralized Zone (m)', fontsize=11)

# Clean legend
ax.legend(fontsize=9, frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig('example_minimalist.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

**Result**: Clean, professional plot ready for publication or presentation!
