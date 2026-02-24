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