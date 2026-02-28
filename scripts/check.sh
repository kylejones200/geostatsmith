
#!/usr/bin/env bash
set -euo pipefail

if [ -f pyproject.toml ]; then
  echo "Python checks"
  python -m ruff format .
  python -m ruff check .
  python -m mypy .
  python -m pytest -q
fi

if [ -f package.json ]; then
  echo "Web checks"
  npm run -s lint
  npm run -s typecheck
  npm run -s test --if-present
  npm run -s build --if-present
fi

echo "OK"