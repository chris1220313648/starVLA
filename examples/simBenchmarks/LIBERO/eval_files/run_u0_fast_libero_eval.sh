#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/u0_fast_libero_goal_full/checkpoints/steps_5000_model.safetensors}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
PORT="${PORT:-6694}"
LIBERO_HOME="${LIBERO_HOME:-/root/nas/code/LIBERO-original}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/nas/envs/starvla-libero/bin/python}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-900}"

[[ -f "$CKPT" ]] || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }
[[ -d "$LIBERO_HOME/libero" ]] || { echo "LIBERO not found: $LIBERO_HOME" >&2; exit 1; }
[[ -x "$LIBERO_PYTHON" ]] || { echo "LIBERO Python not executable: $LIBERO_PYTHON" >&2; exit 1; }

RUN_DIR="${CKPT%%/checkpoints/*}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/eval_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="${LOG_DIR}/${STAMP}_server.log"
EVAL_LOG="${LOG_DIR}/${STAMP}_${TASK_SUITE_NAME}_eval.log"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$STARVLA_DIR"
echo "Starting policy server; log: $SERVER_LOG"
setsid env CKPT="$CKPT" GPU_IDS="$GPU_IDS" PORT="$PORT" \
  bash examples/simBenchmarks/LIBERO/eval_files/run_u0_fast_policy_server.sh \
  > >(tee "$SERVER_LOG") 2>&1 &
server_pid=$!

deadline=$((SECONDS + SERVER_WAIT_SECONDS))
until (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
  kill -0 "$server_pid" 2>/dev/null || { wait "$server_pid"; exit 1; }
  (( SECONDS < deadline )) || { echo "Server did not open port $PORT within ${SERVER_WAIT_SECONDS}s" >&2; exit 1; }
  sleep 2
done

echo "Policy server ready. Starting LIBERO eval; log: $EVAL_LOG"
# LIBERO's bundled initial states contain NumPy arrays (PyTorch >= 2.6).
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 LIBERO_HOME="$LIBERO_HOME" LIBERO_PYTHON="$LIBERO_PYTHON" CKPT="$CKPT" \
TASK_SUITE_NAME="$TASK_SUITE_NAME" NUM_TRIALS_PER_TASK="$NUM_TRIALS_PER_TASK" \
MAX_TASKS="$MAX_TASKS" PORT="$PORT" \
  bash examples/simBenchmarks/LIBERO/eval_files/eval_libero.sh 2>&1 | tee "$EVAL_LOG"
