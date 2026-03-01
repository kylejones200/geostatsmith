Follow dev-loop.md when developing

follow ci-gates.md when doing CI


# Repository Development Standard

This repository follows a layered development model.

You must follow the rules in dev-loop.md when writing or modifying code.
You must follow the rules in ci-gates.md when creating or modifying CI workflows.

These rules are mandatory.

## Authority

dev-loop.md defines the local development contract.
ci-gates.md defines the CI contract.

If a change violates either document, the change is invalid and must be corrected.

## Local development requirement

Before committing, pushing, or proposing a PR:

Run
./dev/check

If it fails:
Fix the failure.
Re-run ./dev/check.
Repeat until it passes.

Do not rely on CI to discover formatter, lint, type, or basic test failures.

## CI requirement

CI must execute checks in the order defined in ci-gates.md.

CI must not introduce checks that are missing from ./dev/check if they belong in Layer 1 or Layer 2.

If CI fails due to formatting, lint, or type errors:
Move that check earlier into ./dev/check.

## Change discipline

Keep changes scoped.
Do not mix refactors and features unless required.
Keep commits small and coherent.

## Reporting discipline

When reporting status:
State whether ./dev/check passes.
If it fails, show the first failing error block.
Do not paste full logs unless requested.



# Cursor Rules

Follow .cursor/rules/ci-gates.md. No exceptions.

## Workflow

You must pass local checks before any push.
You must run scripts/check.sh before you claim success.

You must keep diffs small.
You must keep modules focused.
You must keep separation of concerns.

## Dependencies and schemas

You must ask before you add any dependency.
You must show a migration plan before any schema change.

## Python rules

You must write idiomatic Python.
You must prefer dict lookup and vectorized logic over long if/elif chains.
You must avoid a trash util.py file.
You must put helpers in domain modules.

## Web rules

You must use Tailwind only.
You must not add CSS modules.
You must not add styled-components.

## Completion standard

You must include tests for new behavior.
You must make CI pass.
You must not leave TODO placeholders.