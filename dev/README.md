# Development Scripts

This directory contains the development workflow scripts for the three-layer check system.

## Scripts

- **`./dev/fmt`** - Format code with ruff (Layer 1)
- **`./dev/lint`** - Lint code with ruff (Layer 1)
- **`./dev/type`** - Type check with pyright/mypy (Layer 1/2)
- **`./dev/test`** - Run fast tests (Layer 2)
- **`./dev/check`** - Run all Layer 2 checks (pre-push)

## Usage

**Before every push, run:**
```bash
./dev/check
```

This runs all pre-push checks in under 1 minute.

## Three-Layer System

### Layer 1: Pre-commit (<5s)
- Automatically runs on save/pre-commit
- Formats and lints only staged files
- Configured in `.pre-commit-config.yaml`

### Layer 2: Pre-push (<1min)
- **Run `./dev/check` before every push**
- Runs format, lint, type check, and fast tests
- This is what developers and Cursor should run

### Layer 3: CI
- Full test suite with coverage
- Matrix builds across Python versions
- Runs after Layer 2 passes

## Individual Commands

You can also run individual checks:

```bash
./dev/fmt      # Format code
./dev/lint     # Lint code
./dev/type     # Type check
./dev/test     # Run fast tests
```
