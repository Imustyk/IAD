#!/usr/bin/env bash
# Local CI mirror — same gates as .github/workflows/ci.yml
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

./scripts/lint.sh
pytest -m "not slow"
echo "Local CI passed."
