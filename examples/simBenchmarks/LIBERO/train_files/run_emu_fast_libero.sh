#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${STARVLA_PYTHON:-/opt/conda/envs/starVLA/bin/python}"
CONFIG="${CONFIG:-examples/simBenchmarks/LIBERO/train_files/emu_fast_libero_goal.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
RUN_ID="${RUN_ID:-emu_fast_libero_goal_text_tail}"

cd "$STARVLA_DIR"
export PYTHONPATH="$STARVLA_DIR:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export STARVLA_GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS"

"$PYTHON" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --num_processes "$NUM_PROCESSES" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --run_id "$RUN_ID" \
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" \
  --trainer.save_interval "$SAVE_INTERVAL" \
  --trainer.eval_interval 1000000 \
  --trainer.gradient_accumulation_steps "$GRAD_ACCUM_STEPS"
