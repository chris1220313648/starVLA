#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/opt/conda/envs/starVLA/bin/python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/emu_fast_libero_goal_text_tail/checkpoints/steps_5000_model.safetensors}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}}"
PORT="${PORT:-6694}"

cd "$STARVLA_DIR"
export PYTHONPATH="$STARVLA_DIR:${PYTHONPATH:-}"
echo "Policy GPUs: $GPU_IDS"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$STARVLA_PYTHON" deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" \
  --port "$PORT" \
  --config_override framework.emu.device_map=auto \
  --config_override framework.emu.vq_device=cuda:0 \
  --config_override framework.emu.gradient_checkpointing=false \
  --config_override framework.emu.max_new_tokens=64
