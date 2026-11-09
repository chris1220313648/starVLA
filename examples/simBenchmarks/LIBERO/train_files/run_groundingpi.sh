#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/starvla/bin/python}"
exec "$STARVLA_PYTHON" "$HERE/groundingpi_libero.py" "$@"
