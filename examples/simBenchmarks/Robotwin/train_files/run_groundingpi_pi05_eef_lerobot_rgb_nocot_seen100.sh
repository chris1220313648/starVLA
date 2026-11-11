#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$HERE/../../../.." && pwd)"
WORKSPACE="$(cd -- "$REPO_ROOT/../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-$WORKSPACE/envs/starvla/bin/python}"
SDK_PYTHON="${SDK_PYTHON:-$WORKSPACE/envs/roboscope-kit/bin/python}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_nocot_rgb_seen100_w8_$(date -u +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
DATA_ROOT=/share/project/robot-rfm/robot-processed-data/robotwin_eef20_rgb_lerobot_nocot_canvas_seen100_v1
DATASET_ROOT=/share/project/robot-rfm/robot-processed-data/lerobot_v3_data/robotwin2.0_EEF20_RGB_Lerobotv3
REFERENCE="$REPO_ROOT/playground/Checkpoints/groundingpipi05_eef20_h32_nocot_rgb_w8_20261107_185715"
cd "$REPO_ROOT"
test ! -e "$OUTPUT_PATH"
"$SDK_PYTHON" "$HERE/../eval_files/evaluate_groundingpi_cached.py" --check-cache --suites demo_clean
"$STARVLA_PYTHON" - "$DATASET_ROOT" "$DATA_ROOT" "$REFERENCE" "$OUTPUT_PATH.preflight.json" <<'PY'
import json
from pathlib import Path
import sys
from examples.simBenchmarks.Robotwin.train_files.robotwin_eef_lerobot_dataset import prepare

dataset, cache, reference, report = map(Path, sys.argv[1:])
index = prepare(dataset, cache, instruction_source='seen')
from omegaconf import OmegaConf
baseline_cfg = OmegaConf.load(reference / 'train_b16/config.full.yaml')
baseline_cache = Path(baseline_cfg.datasets.vla_data.data_root_dir)
baseline = json.loads((baseline_cache / 'index.json').read_text())
assert len(index['episodes']) == len(baseline['episodes']) == 2500
assert index['total_samples'] == baseline['total_samples'] == 549787
assert index['quaternion_order'] == baseline['quaternion_order'] == 'wxyz'
assert index['cot_root'] is None and index['image_input_mode'] == 'composite'
for new, old in zip(index['episodes'], baseline['episodes'], strict=True):
    assert len(new['seen_instructions']) == 100
    assert {k: v for k, v in new.items() if k != 'seen_instructions'} == old
assert json.loads((cache / 'dataset_statistics.json').read_text()) == json.loads(
    (reference / 'train_b16/dataset_statistics.json').read_text())
result = dict(passed=True, episodes=2500, total_samples=549787,
              instructions_per_episode=100, instruction_source='seen', reference=str(reference),
              episode_metadata_identical=True, statistics_identical=True)
report.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
PY
"$STARVLA_PYTHON" "$HERE/run_groundingpi_pi05_pipeline.py" \
  --eef --steps 50000 --batch-size 16 --strict-batch --workers 8 --gpus "$GPUS" \
  --dataset-root "$DATASET_ROOT" --data-root "$DATA_ROOT" \
  --instruction-source seen --image-input-mode composite --color-jitter 0 \
  --train-port "${TRAIN_PORT:-29645}" --train-only --output "$OUTPUT_PATH"
mv "$OUTPUT_PATH.preflight.json" "$OUTPUT_PATH/preflight.json"
"$SDK_PYTHON" "$HERE/../eval_files/evaluate_groundingpi_cached.py" \
  --checkpoint "$OUTPUT_PATH/train_b16/final_model/pytorch_model.pt" \
  --output "$OUTPUT_PATH/eval" --gpus "$GPUS" --suites demo_clean
"$STARVLA_PYTHON" - "$OUTPUT_PATH" "$REFERENCE" <<'PY'
import json
from pathlib import Path
import sys

output, reference = map(Path, sys.argv[1:])
before = json.loads((reference / 'eval/demo_clean/result.json').read_text())
after = json.loads((output / 'eval/demo_clean/result.json').read_text())
old = {r['task']: r for r in before['official_result']['tasks']}
new = {r['task']: r for r in after['official_result']['tasks']}
assert set(old) == set(new) and len(new) == 50
assert all(r['episodes'] == 20 for r in [*old.values(), *new.values()])
rows = [dict(task=task, fixed_rate=old[task]['success_rate'],
             seen100_rate=new[task]['success_rate'],
             delta_pp=100 * (new[task]['success_rate'] - old[task]['success_rate']))
        for task in sorted(new)]
comparison = dict(reference=str(reference), fixed=before['result'], seen100=after['result'], tasks=rows)
(output / 'eval/clean_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
lines = ['# Clean: random seen100 vs fixed instruction', '',
         '| Task | Fixed | Random seen100 | Delta (pp) |', '|---|---:|---:|---:|']
for row in rows + [dict(task='**Total**', fixed_rate=before['result']['success_rate'],
                       seen100_rate=after['result']['success_rate'],
                       delta_pp=100*(after['result']['success_rate']-before['result']['success_rate']))]:
    lines.append(f"| {row['task']} | {row['fixed_rate']:.1%} | {row['seen100_rate']:.1%} | {row['delta_pp']:+.1f} |")
(output / 'eval/CLEAN_COMPARISON.md').write_text('\n'.join(lines) + '\n')
PY
