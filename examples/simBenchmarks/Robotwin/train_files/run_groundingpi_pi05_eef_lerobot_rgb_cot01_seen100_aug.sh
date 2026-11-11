#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
WORKSPACE="$(cd -- "$REPO_ROOT/../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$WORKSPACE/envs/starvla/bin/python}"
SDK_PYTHON="${SDK_PYTHON:-$WORKSPACE/envs/roboscope-kit/bin/python}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_shared_cot01_seen100_rgb_aug_$(date -u +%Y%m%d_%H%M%S)}"
CACHE_SOURCE="${CACHE_SOURCE:-$WORKSPACE/experiments/roboscope-groundingpi/cache_20261110_170840}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
cd "$REPO_ROOT"
"$SDK_PYTHON" "$HERE/../eval_files/evaluate_groundingpi_cached.py" \
  --check-cache --cache-source "$CACHE_SOURCE"
"$STARVLA_PYTHON" "$HERE/run_groundingpi_pi05_pipeline.py" \
  --eef --steps 50000 --batch-size "${BATCH_SIZE:-16}" --gpus "$GPUS" \
  --workers 8 --train-port "${TRAIN_PORT:-29645}" --save-interval 5000 \
  --dataset-root /share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_RGB_Lerobotv3 \
  --data-root "$REPO_ROOT/playground/Datasets/robotwin_eef20_rgb_lerobot_cot_canvas_seen100_v1" \
  --with-cot --image-input-mode composite --color-jitter 0.2 --filter-static-segments \
  --cot-loss-weight 0.1 --instruction-source seen --train-only --output "$OUTPUT_PATH" "$@"
CHECKPOINT="$OUTPUT_PATH/$("$STARVLA_PYTHON" -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["run_dir"]).name)' "$OUTPUT_PATH/status.json")/final_model/pytorch_model.pt"
DRIVER_ROOT="$CACHE_SOURCE/driver-libs/usr"
if [[ -f "$DRIVER_ROOT/share/vulkan/icd.d/nvidia_icd.json" ]]; then
  export VK_ICD_FILENAMES="$DRIVER_ROOT/share/vulkan/icd.d/nvidia_icd.json"
  export LD_LIBRARY_PATH="$DRIVER_ROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
exec "$SDK_PYTHON" "$HERE/../eval_files/evaluate_groundingpi_cached.py" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT_PATH/eval" --gpus "$GPUS" \
  --cache-source "$CACHE_SOURCE" --suites demo_clean demo_randomized
