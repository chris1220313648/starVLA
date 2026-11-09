#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$REPO_ROOT/../../envs/starvla/bin/python}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_separate_cot_240x320_$(date -u +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
cd "$REPO_ROOT"
exec "$STARVLA_PYTHON" "$HERE/run_groundingpi_pi05_pipeline.py" \
  --eef --steps 50000 --batch-size "$BATCH_SIZE" --gpus "$GPUS" \
  --dataset-root /share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_Lerobotv3 \
  --with-cot --image-input-mode separate --train-only --output "$OUTPUT_PATH" \
  "$@"
