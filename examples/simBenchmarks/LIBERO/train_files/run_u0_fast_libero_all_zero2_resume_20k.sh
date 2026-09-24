#!/usr/bin/env bash
# Weight-only continuation: optimizer/RNG/data progress were not saved by starVLA.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
SOURCE_RUN="$STARVLA_DIR/playground/Checkpoints/u0_fast_libero_all_zero2_full"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$SOURCE_RUN/checkpoints/steps_7500_model.safetensors}"
RESUME_CHECKPOINT=$(realpath -e "$RESUME_CHECKPOINT")
export CONFIG="$SOURCE_RUN/config.full.yaml"
export STARVLA_PYTHON="${STARVLA_PYTHON:-/root/nas/envs/starvla-u0/bin/python}"
export RUN_ID="${RUN_ID:-u0_fast_libero_all_zero2_20k}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$STARVLA_DIR/playground/Checkpoints}"
OUTPUT_DIR="$RUN_ROOT_DIR/$RUN_ID"
export MAX_TRAIN_STEPS=20000 SAVE_INTERVAL=5000
export NUM_PROCESSES=8 PER_DEVICE_BATCH_SIZE=8 GRAD_ACCUM_STEPS=4

name=$(basename "$RESUME_CHECKPOINT")
if [[ ! -f "$RESUME_CHECKPOINT" || ! "$name" =~ ^steps_([0-9]+)_model\.safetensors$ ]]; then
    echo "Expected steps_N_model.safetensors: $RESUME_CHECKPOINT" >&2
    exit 2
fi
if (( 10#${BASH_REMATCH[1]} >= MAX_TRAIN_STEPS )); then
    echo "Source checkpoint must be earlier than step $MAX_TRAIN_STEPS" >&2
    exit 2
fi

# Existing is_resume discovers the highest step in the new run's checkpoints.
# A link seeds that lookup without copying 9.5 GiB or changing the source run.
mkdir -p "$OUTPUT_DIR/checkpoints"
if [[ ! -e "$OUTPUT_DIR/checkpoints/$name" && ! -L "$OUTPUT_DIR/checkpoints/$name" ]]; then
    ln -s "$RESUME_CHECKPOINT" "$OUTPUT_DIR/checkpoints/$name"
elif [[ "$(realpath -e "$OUTPUT_DIR/checkpoints/$name")" != "$RESUME_CHECKPOINT" ]]; then
    echo "Conflicting checkpoint already exists: $OUTPUT_DIR/checkpoints/$name" >&2
    exit 2
fi

echo "Weight-only continuation to 20000; optimizer and data progress restart."
echo "Batch: 8/GPU x 8 GPUs x accumulation 4 = 256; save every 5000 steps."
echo "Output: $OUTPUT_DIR (resume uses the highest checkpoint step here)."
exec bash "$SCRIPT_DIR/run_u0_fast_libero_all_zero2.sh" \
    --run_root_dir "$RUN_ROOT_DIR" \
    --trainer.is_resume true \
    --trainer.pretrained_checkpoint null \
    "$@"
