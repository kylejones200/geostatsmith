# Development loop and CI discipline

## Goal
Run fast checks on the laptop.
Use CI for integration, packaging, and release gates.
Do not use CI as the first place to discover formatter, linter, or type errors.

## Required commands
This repo defines a single source of truth for checks.

Run this before any commit or push
./dev/check

Run this to auto-fix formatting and lint issues
./dev/fmt

Run this to run unit tests fast
./dev/test

Do not invent new ad hoc commands.
Do not bypass these scripts.

## Layer model
Layer 1 runs on save or pre-commit
Format and lint
Fix errors before commit

Layer 2 runs before push and before opening a PR
Run ./dev/check
Fix errors locally until it passes

Layer 3 runs in CI
CI runs ./dev/check plus full integration tasks
CI failures should indicate integration drift, dependency drift, platform drift, or flaky tests
CI failures should not indicate basic formatting, lint, or type errors

## Cursor behavior rules
When you change code, you must run ./dev/check locally in the workspace
When ./dev/check fails, you must fix the failure before any new feature work
When you propose a PR, you must state that ./dev/check passes
When you see CI failures, you must map each failure to the correct layer
If the failure belongs to Layer 1 or Layer 2, move the check earlier in the loop

## Failure handling
If formatting fails
Run ./dev/fmt
Re-run ./dev/check

If lint fails
Fix the lint issue in code
Prefer small, local fixes
Re-run ./dev/check

If type check fails
Fix types at the source
Add types to public functions and module boundaries
Avoid broad ignores
Re-run ./dev/check

If tests fail
Reproduce locally with ./dev/test
Fix or quarantine flaky tests with a ticket and a clear reason
Re-run ./dev/check

## Scope discipline
Keep changes small
Keep commits focused
Do not mix refactors with feature changes unless required by the fix

## Output discipline
When reporting results, paste the exact command and the first failing error block
Do not paste full CI logs unless asked