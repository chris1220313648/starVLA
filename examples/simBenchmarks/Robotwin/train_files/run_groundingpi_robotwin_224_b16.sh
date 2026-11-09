#!/usr/bin/env bash
# RoboTwin Clean v3: 224x224, 8 GPUs, batch 16/GPU, 10,000 steps.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$REPO_ROOT/../../envs/starvla/bin/python}"
RUNNER="$SCRIPT_DIR/groundingpi_robotwin.py"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/playground/Checkpoints/groundingpipi_v3_robotwin_clean_224_8gpu_b16_10k_$(date -u +%Y%m%d_%H%M%S)}"

cd "$REPO_ROOT"
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=4

train_args=(
  --image-size 224 --batch-size 16 --gpus 8 --steps 10000
  --gradient-accumulation-steps 1
  --train-port "${TRAIN_PORT:-29527}"
  --output "$OUTPUT_DIR"
)

"$STARVLA_PYTHON" "$RUNNER" validate --image-size 224
"$STARVLA_PYTHON" "$RUNNER" train "${train_args[@]}" --dry-run
"$STARVLA_PYTHON" "$RUNNER" train "${train_args[@]}"
