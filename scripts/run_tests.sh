#!/usr/bin/env bash
# Run the full IAD test suite with coverage gate (Phase 9).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
exec pytest "$@"
