#!/usr/bin/env bash
set -euo pipefail
cd /root/nas/code/starVLA
STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
PYTHON="${STARVLA_PYTHON:-/opt/conda/envs/starvla/bin/python}"
CONFIG="${CONFIG:-examples/simBenchmarks/LIBERO/train_files/u0_fast_libero_goal.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
RUN_ID="${RUN_ID:-u0_fast_libero_goal_full}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero3.yaml}"

cd "$STARVLA_DIR"
LOG_DIR="${LOG_DIR:-$STARVLA_DIR/playground/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Training log: $LOG_FILE"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$STARVLA_DIR:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export STARVLA_GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS"

"$PYTHON" -m starVLA.dataloader.u0_vision_cache \
  --config "$CONFIG" --workers "$NUM_PROCESSES" 2>&1 | tee "$LOG_FILE"

"$PYTHON" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --run_id "$RUN_ID" \
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" \
  --trainer.save_interval "$SAVE_INTERVAL" \
  --trainer.eval_interval 10000 \
  --trainer.gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  "$@" 2>&1 | tee -a "$LOG_FILE"
