# CI gates

CI must run fast checks first, then deeper checks.
Order matters.

1. ./dev/check
2. Full unit tests
3. Integration tests
4. Build and packaging
5. Release validation

Do not add new CI checks that duplicate Layer 1 or Layer 2 without also adding them to ./dev/check.
Do not accept PRs that fail ./dev/check.