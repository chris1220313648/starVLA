#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/root/nas/envs/starvla-u0/bin/python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/u0_fast_libero_goal_full/checkpoints/steps_5000_model.safetensors}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-6694}"

cd "$STARVLA_DIR"
export PYTHONPATH="$STARVLA_DIR:${PYTHONPATH:-}"
echo "Policy GPUs: $GPU_IDS"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$STARVLA_PYTHON" deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" \
  --port "$PORT" \
  --config_override framework.u0.gradient_checkpointing=false \
  --config_override framework.u0.max_new_tokens=64
