# Development Setup

## Virtual Environment

A virtual environment has been created in `venv/` with the following tools installed:

- **ruff** (0.15.0) - Fast Python linter and formatter
- **black** (26.1.0) - Python code formatter
- **flake8** (7.3.0) - Python style guide checker

## Activating the Virtual Environment

```bash
source venv/bin/activate
```

After activation, you can use the tools directly:
```bash
ruff check .
ruff format .
black .
flake8 .
```

## Pre-Push Hook

The pre-push git hook (`.git/hooks/pre-push`) has been configured to:
1. Automatically activate the venv if it exists
2. Run `ruff check --fix` and `ruff format` on changed Python files
3. Run `black` formatter on changed Python files
4. Check for Python syntax errors
5. Block push if any checks fail

The hook will automatically use the venv if available, or fall back to system commands.

## Running Linters Manually

### Ruff
```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Black
```bash
# Format code
black .

# Check formatting (don't modify)
black --check .
```

### Flake8
```bash
# Run flake8
flake8 src/ tests/
```

## Note

The `venv/` directory is already in `.gitignore` and will not be committed to git. Each developer should create their own venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install ruff black flake8
```
