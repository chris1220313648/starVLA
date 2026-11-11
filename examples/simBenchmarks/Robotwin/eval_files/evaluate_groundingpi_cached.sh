#!/usr/bin/env bash
set -euo pipefail

# Usage: bash evaluate_groundingpi_cached.sh --checkpoint /path/to/model.pt --output /path/to/new/eval --gpus 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
WORKSPACE="$(cd -- "$REPO_ROOT/../.." && pwd)"
SDK_PYTHON="${SDK_PYTHON:-$WORKSPACE/envs/roboscope-kit/bin/python}"
CACHE_SOURCE="${CACHE_SOURCE:-$WORKSPACE/experiments/roboscope-groundingpi/cache_20261110_170840}"

# Use the same cache for scene seeds and its matching graphics libraries.
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --cache-source)
            if ((i + 1 < ${#args[@]})); then
                CACHE_SOURCE="${args[i+1]}"
            fi
            ;;
        --cache-source=*) CACHE_SOURCE="${args[i]#*=}" ;;
    esac
done
# Resolve relative cache paths for the graphics library environment.
CACHE_SOURCE="$(realpath -m -- "$CACHE_SOURCE")"
DRIVER_ROOT="$CACHE_SOURCE/driver-libs/usr"
if [[ -f "$DRIVER_ROOT/share/vulkan/icd.d/nvidia_icd.json" ]]; then
    export VK_ICD_FILENAMES="$DRIVER_ROOT/share/vulkan/icd.d/nvidia_icd.json"
    export LD_LIBRARY_PATH="$DRIVER_ROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

exec "$SDK_PYTHON" "$SCRIPT_DIR/evaluate_groundingpi_cached.py" \
    --cache-source "$CACHE_SOURCE" "$@"
