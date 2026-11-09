#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$REPO_ROOT/../../envs/starvla/bin/python}"
cd "$REPO_ROOT"
exec "$STARVLA_PYTHON" "$HERE/run_groundingpi_pi05_pipeline.py" "$@"
