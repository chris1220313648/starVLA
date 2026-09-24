#!/usr/bin/env bash
set -euo pipefail

cd /root/nas/code/starVLA
PYTHON="${STARVLA_PYTHON:-/opt/conda/envs/starvla/bin/python}"
CONFIG="${CONFIG:-examples/simBenchmarks/LIBERO/train_files/u0_pi05_libero.yaml}"
RUN_ID="${RUN_ID:-u0_pi05_libero_smoke}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5}"
LOG_DIR="${LOG_DIR:-playground/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_ID}_$(date +%Y%m%d_%H%M%S)_$$.log"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export STARVLA_DISABLE_DEEPSPEED=1
export STARVLA_GRAD_ACCUM_STEPS=1

echo "Training log: $LOG_FILE"

"$PYTHON" -m starVLA.dataloader.u0_vision_cache \
  --config "$CONFIG" --workers 1 2>&1 | tee "$LOG_FILE"

"$PYTHON" -m starVLA.training.train_starvla \
  --config_yaml "$CONFIG" \
  --run_id "$RUN_ID" \
  --trainer.max_train_steps "$MAX_TRAIN_STEPS" \
  --trainer.save_interval "$MAX_TRAIN_STEPS" \
  --trainer.eval_interval 100000000 2>&1 | tee -a "$LOG_FILE"

echo "Completed U0PI05 single-GPU smoke: $RUN_ID"
