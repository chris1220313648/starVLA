#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
WORKSPACE="$(cd -- "$REPO_ROOT/../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$WORKSPACE/envs/starvla/bin/python}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_shared_cot01_seen100_rgb_w8_$(date -u +%Y%m%d_%H%M%S)}"
cd "$REPO_ROOT"
exec "$STARVLA_PYTHON" "$HERE/run_groundingpi_pi05_pipeline.py" \
  --eef --steps 50000 --batch-size "${BATCH_SIZE:-16}" --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --workers 8 --train-port "${TRAIN_PORT:-29645}" \
  --dataset-root /share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_RGB_Lerobotv3 \
  --data-root "$REPO_ROOT/playground/Datasets/robotwin_eef20_rgb_lerobot_cot_canvas_seen100_v1" \
  --with-cot --image-input-mode composite --color-jitter 0 \
  --cot-loss-weight 0.1 --instruction-source seen --train-only --output "$OUTPUT_PATH" "$@"
