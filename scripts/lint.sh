#!/usr/bin/env bash
# Run static analysis (ruff + black check + mypy).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> ruff check"
ruff check iad tests scripts

echo "==> ruff format --check"
ruff format --check iad tests scripts

echo "==> black --check"
black --check iad tests scripts

echo "==> mypy"
mypy iad

echo "All lint checks passed."
