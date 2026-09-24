#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-$SCRIPT_DIR/u0_fast_libero_all.yaml}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$SCRIPT_DIR/u0_fast_zero2_accelerate.yaml}"
export RUN_ID="${RUN_ID:-u0_fast_libero_all_zero2_full}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

# Default global batch: 8 examples/GPU * 8 GPUs * 4 accumulation steps = 256.
exec bash "$SCRIPT_DIR/run_u0_fast_libero.sh" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-8}" "$@"
