#!/usr/bin/env bash
set -euo pipefail
ENV_DIR=${ENV_DIR:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/starvla}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
mkdir -p playground/groundingpi-smoke
for framework in GroundingPIOFT GroundingPIPI GroundingPIFast; do
    "$ENV_DIR/bin/python" -m examples.modelExtensions.GroundingPI.smoke --framework "$framework" "$@" \
        2>&1 | tee "playground/groundingpi-smoke/${framework}.log"
done
