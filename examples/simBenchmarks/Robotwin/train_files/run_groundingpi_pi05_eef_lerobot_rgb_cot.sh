#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
WORKSPACE="$(cd -- "$REPO_ROOT/../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$WORKSPACE/envs/starvla/bin/python}"
SDK_PYTHON="${SDK_PYTHON:-$WORKSPACE/envs/roboscope-kit/bin/python}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_shared_cot_rgb_$(date -u +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
cd "$REPO_ROOT"
# Fail before training if the existing benchmark scenes cannot be reused.
"$SDK_PYTHON" "$HERE/../eval_files/evaluate_groundingpi_cached.py" --check-cache
"$STARVLA_PYTHON" "$HERE/run_groundingpi_pi05_pipeline.py" "$@" \
  --eef --steps 50000 --batch-size "$BATCH_SIZE" --gpus "$GPUS" \
  --dataset-root /share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_RGB_Lerobotv3 \
  --data-root "$REPO_ROOT/playground/Datasets/robotwin_eef20_rgb_lerobot_cot_canvas_v1" \
  --with-cot --image-input-mode composite --color-jitter 0 --cot-loss-weight 1 \
  --train-only --output "$OUTPUT_PATH"
CHECKPOINT="$OUTPUT_PATH/$("$STARVLA_PYTHON" -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["run_dir"]).name)' "$OUTPUT_PATH/status.json")/final_model/pytorch_model.pt"
exec "$SDK_PYTHON" "$HERE/../eval_files/evaluate_groundingpi_cached.py" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT_PATH/eval" --gpus "$GPUS"
