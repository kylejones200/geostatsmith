"""
Sphinx configuration for GeoStats documentation.
"""

import os
import sys
from pathlib import Path

# Add the source directory to the path so we can import geostats
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Project information
project = "GeoStats"
copyright = "2026, Kyle Jones"
author = "Kyle Jones"
release = "0.3.0"
version = "0.3.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",  # For Markdown support
]

# Napoleon settings for NumPy/Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}
autodoc_mock_imports = [
    "rasterio",
    "netCDF4",
    "geopandas",
    "shapely",
    "fiona",
    "openpyxl",
    "xgboost",
    "plotly",
    "fastapi",
    "uvicorn",
    "pydantic",
    "python-multipart",
    "jinja2",
]
# Continue building even if there are import errors
autodoc_continue_on_import_error = True

# Suppress warnings for missing imports
suppress_warnings = ['autodoc.import_object']
autodoc_mock_imports = [
    "rasterio",
    "netCDF4",
    "geopandas",
    "shapely",
    "fiona",
    "openpyxl",
    "xgboost",
    "plotly",
    "fastapi",
    "uvicorn",
    "pydantic",
    "python-multipart",
    "jinja2",
]

# Templates
templates_path = ["_templates"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
