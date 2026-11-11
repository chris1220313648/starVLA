#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
WORKSPACE="$(cd -- "$REPO_ROOT/../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$WORKSPACE/envs/starvla/bin/python}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_history5_stride40_cot01_seen100_rgb_w8_$(date -u +%Y%m%d_%H%M%S)}"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=, read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
ARGS=( --framework pi05_eef --steps 50000 --batch-size "${BATCH_SIZE:-4}" --gpus "${#GPU_IDS[@]}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" --save-interval 5000 \
  --workers 8 --train-port "${TRAIN_PORT:-29655}" \
  --dataset-root /share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_RGB_Lerobotv3 \
  --data-root "$REPO_ROOT/playground/Datasets/robotwin_eef20_rgb_lerobot_cot_canvas_seen100_v1" \
  --with-cot --image-input-mode composite --color-jitter 0.2 \
  --config-yaml "$HERE/groundingpipi05_eef_history5_stride40.yaml" \
  --cot-loss-weight 0.1 --instruction-source seen --output "$OUTPUT_PATH" "$@" )
"$STARVLA_PYTHON" "$HERE/groundingpi_robotwin.py" validate "${ARGS[@]}"
exec "$STARVLA_PYTHON" "$HERE/groundingpi_robotwin.py" train "${ARGS[@]}"
