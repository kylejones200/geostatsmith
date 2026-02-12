# GeoStats Documentation

This directory contains the documentation source files for GeoStats.

## Building Documentation Locally

To build the documentation locally:

```bash
# Install documentation dependencies
uv sync --dev

# Build HTML documentation
cd docs/source
sphinx-build -b html . _build/html

# View the documentation
open _build/html/index.html
```

## Documentation Structure

- `source/` - Sphinx source files (reStructuredText and Markdown)
- `source/api/` - API reference documentation (auto-generated from docstrings)
- `QUICKSTART.md`, `INSTALL.md`, etc. - User guide markdown files (included via MyST parser)

## Read the Docs

The documentation is configured for Read the Docs via `.readthedocs.yaml`.

To enable on Read the Docs:
1. Import the repository on readthedocs.org
2. The configuration will be automatically detected
3. Documentation will build on each push to main
