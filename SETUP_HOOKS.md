# Pre-Push Hooks Setup

This repository now has pre-push hooks configured to run `ruff` and `black` before pushing to git.

## What's Configured

1. **Pre-push Git Hook** (`.git/hooks/pre-push`)
   - Runs `ruff check --fix` and `ruff format` on changed Python files
   - Runs `black` formatter on changed Python files
   - Checks for Python syntax errors
   - Blocks push if any checks fail

2. **Pre-commit Configuration** (`.pre-commit-config.yaml`)
   - Updated to include `ruff` alongside `black` and `flake8`
   - Configured to run on both `commit` and `push` stages

## Installation

### Option 1: Use Pre-commit Framework (Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
pre-commit install --hook-type pre-push
```

This will use the `.pre-commit-config.yaml` configuration.

### Option 2: Use Direct Git Hook (Already Installed)

The `.git/hooks/pre-push` hook is already installed and will run automatically.

## Manual Testing

You can test the hook manually:

```bash
# Test the pre-push hook
.git/hooks/pre-push
```

## Requirements

Make sure you have the required tools installed:

```bash
pip install ruff black
```

## How It Works

1. When you run `git push`, the pre-push hook automatically runs
2. It checks all changed Python files (staged and unstaged)
3. Runs `ruff check --fix` to find and auto-fix linting issues
4. Runs `ruff format` to format code
5. Runs `black` to ensure consistent formatting
6. Checks for Python syntax errors
7. If any check fails, the push is blocked and you'll see error messages
8. If all checks pass, the push proceeds normally

## Bypassing Hooks (Not Recommended)

If you absolutely need to bypass the hooks (not recommended):

```bash
git push --no-verify
```

## Troubleshooting

### Hook not running
- Make sure the hook is executable: `chmod +x .git/hooks/pre-push`
- Check that you're pushing to a branch (hooks don't run on initial clone)

### Ruff/Black not found
- Install them: `pip install ruff black`
- Or use the pre-commit framework which handles dependencies automatically

### Hook fails but you want to push anyway
- Fix the issues first (recommended)
- Or use `git push --no-verify` (not recommended)
