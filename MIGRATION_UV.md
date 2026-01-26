# Migration to uv

This document describes the migration from pip/requirements.txt to uv for the GeoStats library.

## Overview

The project has been modernized to use `uv` as the default package manager and `pyproject.toml` as the single source of truth. The `uv.lock` file ensures reproducible builds.

## What Changed

- **Package Manager**: `pip` → `uv`
- **Dependency Management**: `requirements.txt` → `pyproject.toml` + `uv.lock`
- **Dev Dependencies**: Now managed via `[dependency-groups]` in `pyproject.toml`
- **CI/CD**: Updated to use `uv` for all operations

## Command Migration

### Installation

| Old Command | New Command |
|------------|-------------|
| `pip install -e .` | `uv sync` |
| `pip install -e ".[dev]"` | `uv sync --dev` |
| `pip install -e ".[all]"` | `uv sync --all-extras` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip install -r requirements-dev.txt` | `uv sync --dev` |

### Running Commands

| Old Command | New Command |
|------------|-------------|
| `python script.py` | `uv run python script.py` |
| `pytest tests/` | `uv run pytest tests/` |
| `black src/` | `uv run black src/` |
| `ruff check .` | `uv run ruff check .` |
| `mypy src/` | `uv run mypy src/` |
| `flake8 src/` | `uv run flake8 src/` |

### Building and Publishing

| Old Command | New Command |
|------------|-------------|
| `python -m build` | `uv run python -m build` |
| `twine check dist/*` | `uv run twine check dist/*` |
| `twine upload dist/*` | `uv run twine upload dist/*` |

## Installation of uv

Install uv using the official installer:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (if you have pip)
pip install uv
```

After installation, add `~/.local/bin` to your PATH (or restart your terminal).

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kylejones200/geostats.git
   cd geostats
   ```

2. **Install dependencies**:
   ```bash
   uv sync --dev
   ```

3. **Run tests**:
   ```bash
   uv run pytest tests/
   ```

4. **Run linting**:
   ```bash
   uv run ruff check .
   ```

## Benefits of uv

- **Faster**: 10-100x faster than pip for dependency resolution and installation
- **Reproducible**: `uv.lock` ensures exact dependency versions
- **Single Source of Truth**: `pyproject.toml` contains all dependency information
- **Better Error Messages**: Clearer error messages when dependencies conflict
- **Virtual Environment Management**: Automatically manages `.venv` directory

## Legacy Files

The following files are kept for reference but are no longer the source of truth:

- `requirements.txt` - Legacy runtime dependencies
- `requirements-dev.txt` - Legacy development dependencies

**Do not use these files for installation.** Use `uv sync` instead.

## CI/CD Changes

The GitHub Actions workflow (`.github/workflows/ci.yml`) has been updated to:

1. Install `uv` using `astral-sh/setup-uv@v4`
2. Use `uv sync --frozen --dev` to install dependencies
3. Run all commands via `uv run`

## Troubleshooting

### uv command not found

Make sure uv is installed and in your PATH:
```bash
# Check installation
~/.local/bin/uv --version

# Add to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"
```

### Lock file out of sync

If `uv.lock` is out of sync with `pyproject.toml`:
```bash
uv lock
```

### Virtual environment issues

uv automatically manages `.venv`. If you need to recreate it:
```bash
rm -rf .venv
uv sync --dev
```

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 621 - Project metadata](https://peps.python.org/pep-0621/)
- [PEP 735 - Dependency groups](https://peps.python.org/pep-0735/)

## Questions?

If you encounter issues during migration, please:
1. Check this migration guide
2. Review the updated documentation in `docs/INSTALL.md`
3. Open an issue on GitHub

