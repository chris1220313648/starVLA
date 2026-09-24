#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."
export STARVLA_PYTHON="${STARVLA_PYTHON:-/root/nas/envs/starvla-u0/bin/python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ID="${RUN_ID:-u0_fast_libero_all_zero2_h2_20k}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
# Real h=2 batch 8 OOMed; batch 4 keeps global batch 256 with accumulation 8.
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
export NUM_PROCESSES=8
H="${H:-2}"
if [[ "$H" != 2 ]]; then echo 'This stateful preset currently requires H=2' >&2; exit 2; fi
SOURCE_CONFIG="$SCRIPT_DIR/u0_fast_libero_all_h2.yaml"
PREPARED_DIR="$PWD/playground/Checkpoints/$RUN_ID/preparation"
"$STARVLA_PYTHON" -m starVLA.dataloader.u0_sequence --config "$SOURCE_CONFIG" --output "$PREPARED_DIR/config.yaml" --h "$H"
export CONFIG="$PREPARED_DIR/config.yaml"
exec bash "$SCRIPT_DIR/run_u0_fast_libero_all_zero2.sh" \
  --trainer.eval_interval "$((MAX_TRAIN_STEPS + 1))" "$@"
