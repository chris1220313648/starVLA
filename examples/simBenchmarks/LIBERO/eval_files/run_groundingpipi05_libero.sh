#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/playground/Checkpoints/groundingpipi05_libero_4gpu_b16_20261023_131723}"

CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_DIR}/checkpoints/steps_10000_pytorch_model.pt}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/starvla/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/libero/bin/python}"
LIBERO_ROOT="${LIBERO_ROOT:-/share/project/robot-rfm/robot-exp/usr/cjy/simulation/LIBERO}"
RAW_DATA="${RAW_DATA:-${REPO_ROOT}/playground/Datasets/LIBERO_LeRobot_v3_groundingpi}"
BASE_PORT="${BASE_PORT:-12093}"
EVAL_TRIALS="${EVAL_TRIALS:-50}"
EVAL_SEED="${EVAL_SEED:-7}"
EVAL_MAX_TASKS="${EVAL_MAX_TASKS:--1}"
EVAL_SUITES="${EVAL_SUITES:-libero_spatial libero_object libero_goal libero_10}"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)_$$"
EVAL_OUTPUT="${EVAL_OUTPUT:-${CHECKPOINT_DIR}/eval/run_${RUN_STAMP}}"

for required_path in "${CHECKPOINT}" "${STARVLA_PYTHON}" "${LIBERO_PYTHON}" \
                     "${LIBERO_ROOT}/libero/libero/assets" \
                     "${LIBERO_ROOT}/libero/libero/bddl_files" \
                     "${LIBERO_ROOT}/libero/libero/init_files" "${RAW_DATA}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

DETECTED_GPUS="$("${STARVLA_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ ! "${DETECTED_GPUS}" =~ ^[0-9]+$ ]] || (( DETECTED_GPUS < 1 )); then
  echo "No CUDA GPUs are visible to ${STARVLA_PYTHON}. Set CUDA_VISIBLE_DEVICES or check the environment." >&2
  exit 1
fi

if [[ -n "${EVAL_GPUS:-}" ]]; then
  if [[ ! "${EVAL_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_GPUS must be a positive integer; got '${EVAL_GPUS}'." >&2
    exit 2
  fi
  GPU_COUNT="${EVAL_GPUS}"
else
  GPU_COUNT="${DETECTED_GPUS}"
fi
if (( GPU_COUNT > DETECTED_GPUS )); then
  echo "EVAL_GPUS=${GPU_COUNT} exceeds the ${DETECTED_GPUS} visible GPU(s)." >&2
  exit 2
fi

if [[ -e "${EVAL_OUTPUT}" ]]; then
  echo "Evaluation output already exists; refusing to overwrite: ${EVAL_OUTPUT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${EVAL_OUTPUT}")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export TOKENIZERS_PARALLELISM="false"

read -r -a SUITES <<< "${EVAL_SUITES}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Visible GPUs: ${DETECTED_GPUS}; evaluation workers: ${GPU_COUNT}"
echo "Suites: ${SUITES[*]} | trials/task: ${EVAL_TRIALS} | seed: ${EVAL_SEED}"
echo "Output: ${EVAL_OUTPUT}"

exec "${STARVLA_PYTHON}" "${REPO_ROOT}/examples/simBenchmarks/LIBERO/train_files/groundingpi_libero.py" eval \
  --checkpoint "${CHECKPOINT}" \
  --eval-gpus "${GPU_COUNT}" \
  --libero-python "${LIBERO_PYTHON}" \
  --libero-root "${LIBERO_ROOT}" \
  --raw-data "${RAW_DATA}" \
  --port "${BASE_PORT}" \
  --suites "${SUITES[@]}" \
  --trials "${EVAL_TRIALS}" \
  --max-tasks "${EVAL_MAX_TASKS}" \
  --seed "${EVAL_SEED}" \
  --eval-output "${EVAL_OUTPUT}"
