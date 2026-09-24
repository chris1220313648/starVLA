#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-$SCRIPT_DIR/u0_fast_libero_all.yaml}"
export RUN_ID="${RUN_ID:-u0_fast_libero_all_full}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25000}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

exec bash "$SCRIPT_DIR/run_u0_fast_libero.sh" "$@"
