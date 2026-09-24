#!/usr/bin/env bash
# Run the five-episode smoke, then a separate complete Goal evaluation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=4
CKPT=${CKPT:-$PWD/playground/Checkpoints/u0_fast_libero_all_zero2_full/checkpoints/steps_7500_model.safetensors}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
TASK_START=${TASK_START:-0} TASK_END=${TASK_END:-9} NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
BASE_PORT=${BASE_PORT:-6800} OUTPUT_DIR=${OUTPUT_DIR:-}
export STARVLA_PYTHON=${STARVLA_PYTHON:-/root/nas/envs/starvla-u0/bin/python}
export LIBERO_PYTHON=${LIBERO_PYTHON:-/root/nas/envs/starvla-libero/bin/python}
export LIBERO_HOME=${LIBERO_HOME:-/root/nas/code/LIBERO-original}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$PWD/playground/LIBERO_CONFIG}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$LIBERO_HOME:$PYTHONPATH"
while (( $# )); do
    case $1 in
        --help|-h) echo 'Options: --checkpoint PATH --gpus 0,1,2,3,4,5,6,7 --task-start 0 --task-end 9 --trials 50 --base-port 6800 --output DIR'; exit 0 ;;
        --checkpoint|--gpus|--task-start|--task-end|--trials|--base-port|--output)
            (( $# >= 2 )) || { echo "Missing value: $1" >&2; exit 2; } ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    case $1 in
        --checkpoint) CKPT=$2 ;; --gpus) GPU_IDS=$2 ;; --task-start) TASK_START=$2 ;;
        --task-end) TASK_END=$2 ;; --trials) NUM_TRIALS_PER_TASK=$2 ;;
        --base-port) BASE_PORT=$2 ;; --output) OUTPUT_DIR=$2 ;;
    esac
    shift 2
done
CKPT=$(realpath "$CKPT")
OUTPUT_DIR=${OUTPUT_DIR:-$(dirname "$CKPT")/eval/$(basename "$CKPT" .safetensors)/libero_goal_$(date +%Y%m%d_%H%M%S_%N)}
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")
export CKPT OUTPUT_DIR BASE_PORT
export EVAL_DIR=$PWD/examples/simBenchmarks/LIBERO/eval_files
export CONTROL=$EVAL_DIR/u0_fast_eval_control.py
"$STARVLA_PYTHON" "$CONTROL" prepare --checkpoint "$CKPT" --output "$OUTPUT_DIR" \
    --gpus "$GPU_IDS" --base-port "$BASE_PORT" --task-start "$TASK_START" --task-end "$TASK_END" --trials "$NUM_TRIALS_PER_TASK"
exec > >(tee -a "$OUTPUT_DIR/pipeline.log") 2>&1
stage=smoke
trap 'code=$?; printf "%s exit=%s\n" "$stage" "$code" > "$OUTPUT_DIR/status"' EXIT

worker() {
    local rank=$1 gpu=$2 folder="$PHASE_DIR/worker_$1" port=$((BASE_PORT+$1))
    # Each policy is independent; do not inherit the training platform's rendezvous.
    unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
    unset MASTER_ADDR MASTER_PORT
    # EXIT runs after this function returns, so cleanup PIDs must outlive locals.
    server_pid='' client_pid=''
    mkdir -p "$folder"
    trap 'kill ${server_pid:-} ${client_pid:-} 2>/dev/null || true; wait ${server_pid:-} ${client_pid:-} 2>/dev/null || true' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    CUDA_VISIBLE_DEVICES='' "$LIBERO_PYTHON" -u "$EVAL_DIR/eval_libero.py" --args.render-check \
        --args.video-out-path "$folder/render" > "$folder/render.log" 2>&1
    echo "RENDER_READY worker=$rank"
    CUDA_VISIBLE_DEVICES="$gpu" "$STARVLA_PYTHON" -u deployment/model_server/server_policy.py \
        --ckpt_path "$CKPT" --port "$port" \
        --config_override framework.u0.gradient_checkpointing=false \
        --config_override framework.u0.use_cached_vision=false \
        --config_override framework.u0.max_new_tokens=64 > "$folder/server.log" 2>&1 &
    server_pid=$!
    if ! "$STARVLA_PYTHON" "$CONTROL" health --checkpoint "$CKPT" --base-port "$port" --pid "$server_pid" > "$folder/health.json"; then
        echo "Policy startup failed: $folder/server.log" >&2
        tail -n 80 "$folder/server.log" >&2
        return 1
    fi
    echo "POLICY_READY worker=$rank GPU=$gpu port=$port"
    CUDA_VISIBLE_DEVICES='' "$LIBERO_PYTHON" -u "$EVAL_DIR/eval_libero.py" --args.u0-fast \
        --args.pretrained-path "$CKPT" --args.port "$port" --args.task-suite-name libero_goal \
        --args.task-start "$PHASE_START" --args.task-end "$PHASE_END" --args.num-trials-per-task "$PHASE_TRIALS" \
        --args.worker-id "$rank" --args.num-workers 8 --args.unnorm-key franka \
        --args.video-out-path "$folder" > "$folder/client.log" 2>&1 &
    client_pid=$!
    while kill -0 "$client_pid" 2>/dev/null; do
        kill -0 "$server_pid" 2>/dev/null || { echo "Policy died: $folder/server.log" >&2; return 1; }
        sleep 1
    done
    wait "$client_pid"
    client_pid=''
    echo "WORKER_DONE $rank"
}
export -f worker

run_phase() (
    export PHASE_DIR="$OUTPUT_DIR/$1" PHASE_START=$2 PHASE_END=$3 PHASE_TRIALS=$4
    mkdir -p "$PHASE_DIR"
    pids=()
    started=$SECONDS
    finish() {
        code=$?
        trap - EXIT INT TERM
        for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
        sleep 2
        for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
        "$STARVLA_PYTHON" "$CONTROL" aggregate --output "$PHASE_DIR" --task-start "$PHASE_START" \
            --task-end "$PHASE_END" --trials "$PHASE_TRIALS" --failure-code "$code" || code=1
        printf '%s\n' "$((SECONDS-started))" > "$PHASE_DIR/elapsed_seconds"
        exit "$code"
    }
    trap finish EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    IFS=, read -r -a gpus <<< "$GPU_IDS"
    for rank in {0..7}; do
        count=0
        for ((task=PHASE_START; task<=PHASE_END; task++)); do
            for ((trial=0; trial<PHASE_TRIALS; trial++)); do
                if (( (task*PHASE_TRIALS+trial)%8 == rank )); then count=$((count+1)); fi
            done
        done
        (( count > 0 )) || continue
        setsid bash -euo pipefail -c 'worker "$@"' bash "$rank" "${gpus[$rank]}" &
        pids+=("$!")
    done
    pending=("${pids[@]}")
    while (( ${#pending[@]} )); do
        wait -n -p finished "${pending[@]}" || exit $?
        remaining=()
        for pid in "${pending[@]}"; do [[ "$pid" == "$finished" ]] || remaining+=("$pid"); done
        pending=("${remaining[@]}")
    done
)
echo "OUTPUT $OUTPUT_DIR"
printf '%s\n' "$stage" > "$OUTPUT_DIR/status"
run_phase smoke 0 0 5
stage=formal
printf '%s\n' "$stage" > "$OUTPUT_DIR/status"
run_phase formal "$TASK_START" "$TASK_END" "$NUM_TRIALS_PER_TASK"
stage=complete
