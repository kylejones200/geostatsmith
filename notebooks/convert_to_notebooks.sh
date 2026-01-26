#!/bin/bash
#
# Convert Python scripts to Jupyter notebooks using jupytext
# 
# Usage: ./convert_to_notebooks.sh
#

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Converting Python scripts to Jupyter notebooks...           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if jupytext is installed
if ! command -v jupytext &> /dev/null; then
    echo "⚠️  jupytext not found. Installing..."
    pip install jupytext
    echo ""
fi

# Convert each notebook
echo "📓 Converting Notebook 1: Gold Exploration..."
jupytext --to ipynb 01_gold_exploration_insights.py
echo "✅ Created: 01_gold_exploration_insights.ipynb"
echo ""

echo "📓 Converting Notebook 2: Multi-Element Detective..."
jupytext --to ipynb 02_multi_element_detective.py
echo "✅ Created: 02_multi_element_detective.ipynb"
echo ""

echo "📓 Converting Notebook 3: Environmental Risk..."
jupytext --to ipynb 03_environmental_risk.py
echo "✅ Created: 03_environmental_risk.ipynb"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ All notebooks converted successfully!                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Launch Jupyter with:"
echo "   jupyter notebook"
echo ""
echo "   or"
echo ""
echo "   jupyter lab"
echo ""
