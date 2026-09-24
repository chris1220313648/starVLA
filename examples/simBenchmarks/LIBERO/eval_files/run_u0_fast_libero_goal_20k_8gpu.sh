#!/usr/bin/env bash
# Eight-GPU LIBERO Goal evaluation for the zero2_20k step-20000 checkpoint.
# Override with CKPT=/path/to/model.safetensors or --checkpoint PATH.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
export CKPT=${CKPT:-$REPO_ROOT/playground/Checkpoints/u0_fast_libero_all_zero2_20k/checkpoints/steps_20000_model.safetensors}
export GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
exec bash "$SCRIPT_DIR/run_u0_fast_libero_goal_8gpu.sh" "$@"
