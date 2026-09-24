#!/usr/bin/env bash
# Evaluate the latest saved h=2 checkpoint as of 2026-09-16 (step 15000).
# Override with CKPT=/path/to/model.safetensors or --checkpoint PATH.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
export CKPT=${CKPT:-$REPO_ROOT/playground/Checkpoints/u0_fast_libero_all_zero2_h2_20k/checkpoints/steps_15000_model.safetensors}
export GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
exec bash "$SCRIPT_DIR/run_u0_fast_libero_goal_8gpu.sh" "$@"
