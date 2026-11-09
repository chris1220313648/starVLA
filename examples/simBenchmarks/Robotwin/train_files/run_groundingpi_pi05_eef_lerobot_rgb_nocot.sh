#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
WORKSPACE="$(cd -- "$REPO_ROOT/../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$WORKSPACE/envs/starvla/bin/python}"
SDK_PYTHON="${SDK_PYTHON:-$WORKSPACE/envs/roboscope-kit/bin/python}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_nocot_rgb_w8_$(date -u +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
cd "$REPO_ROOT"
"$SDK_PYTHON" "$HERE/evaluate_groundingpi_cached.py" --check-cache --suites demo_clean
"$STARVLA_PYTHON" "$HERE/run_groundingpi_pi05_pipeline.py" \
  --eef --steps 50000 --batch-size 16 --strict-batch --workers 8 --gpus "$GPUS" \
  --dataset-root /share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_RGB_Lerobotv3 \
  --data-root "$REPO_ROOT/playground/Datasets/robotwin_eef20_rgb_lerobot_nocot_canvas_v1" \
  --image-input-mode composite --color-jitter 0 --train-port 29645 \
  --train-only --output "$OUTPUT_PATH"
CHECKPOINT="$OUTPUT_PATH/train_b16/final_model/pytorch_model.pt"
"$SDK_PYTHON" "$HERE/evaluate_groundingpi_cached.py" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT_PATH/eval" --gpus "$GPUS" --suites demo_clean
"$STARVLA_PYTHON" - "$OUTPUT_PATH" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
reference = output.parent / 'groundingpipi05_eef20_h32_shared_cot_rgb_w8_20261106_112125/eval/demo_clean/result.json'
before = json.loads(reference.read_text())
after = json.loads((output / 'eval/demo_clean/result.json').read_text())
old = {row['task']: row for row in before['official_result']['tasks']}
new = {row['task']: row for row in after['official_result']['tasks']}
assert set(old) == set(new) and len(new) == 50
assert all(row['episodes'] == 20 for row in [*old.values(), *new.values()])
rows = [dict(task=task, cot_rate=old[task]['success_rate'], nocot_rate=new[task]['success_rate'],
             delta_pp=100 * (new[task]['success_rate'] - old[task]['success_rate'])) for task in sorted(new)]
comparison = dict(reference=str(reference), cot=before['result'], nocot=after['result'], tasks=rows)
(output / 'eval/clean_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
lines = ['# Clean: no CoT vs CoT', '', '| Task | CoT | No CoT | Delta (pp) |', '|---|---:|---:|---:|']
for row in rows + [dict(task='**Total**', cot_rate=before['result']['success_rate'],
                       nocot_rate=after['result']['success_rate'],
                       delta_pp=100 * (after['result']['success_rate'] - before['result']['success_rate']))]:
    lines.append(f"| {row['task']} | {row['cot_rate']:.1%} | {row['nocot_rate']:.1%} | {row['delta_pp']:+.1f} |")
(output / 'eval/CLEAN_COMPARISON.md').write_text('\n'.join(lines) + '\n')
PY
