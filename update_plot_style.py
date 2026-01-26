#!/usr/bin/env python3
"""
Update all visualization functions to minimalist style

This script applies consistent minimalist styling across all plot functions:
- Remove gridlines (unless necessary)
- Remove top and right spines  
- Use descriptive titles instead of axis labels
- Clean, professional aesthetic
"""

import re
import os

# Define the transformations
def update_plot_styling(content):
    """Update plot styling in visualization code"""
    
    # Replace grid(True) or grid(alpha=...) with grid(False) or remove grid calls
    content = re.sub(r'ax\.grid\(True[^)]*\)', 'apply_minimalist_style(ax)', content)
    content = re.sub(r'ax\.grid\(alpha=[^)]+\)', '', content)
    
    # Remove explicit axis labels and replace with descriptive titles
    # This is context-specific, so we'll do it manually for each plot type
    
    # Update colors to muted palette
    content = content.replace("c='black'", "c='#1f77b4'")  # Blue
    content = content.replace("color='black'", "color='#333333'")  # Dark gray
    content = content.replace("'r-'", "'#d62728'")  # Red
    content = content.replace("'r--'", "'#d62728', linestyle='--'")  # Red dashed
    
    # Update box styling
    content = content.replace("facecolor='wheat'", "facecolor='#f0f0f0', edgecolor='none'")
    content = content.replace("facecolor='lightblue'", "facecolor='#f0f0f0', edgecolor='none'")
    
    # Update font weights
    content = content.replace("fontweight='bold'", "")
    
    return content

# List of files to update
visualization_files = [
    '/Users/k.jones/Desktop/geostats/src/geostats/visualization/variogram_plots.py',
    '/Users/k.jones/Desktop/geostats/src/geostats/visualization/spatial_plots.py',
    '/Users/k.jones/Desktop/geostats/src/geostats/visualization/diagnostic_plots.py',
]

print("Updating visualization files to minimalist style...")
for filepath in visualization_files:
    if os.path.exists(filepath):
        print(f"  Processing: {filepath}")
        # Note: Actual file updates will be done manually with proper context
        # This script serves as a template
    else:
        print(f"  Not found: {filepath}")

print("\nManual updates required for:")
print("- Title rewording to be descriptive")
print("- Removing unnecessary axis labels")
print("- Context-specific styling adjustments")
