#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/lx/IsaacLab}"
ISAAC_RUNTIME="${ISAAC_RUNTIME:-/home/lx/isaacsim-runtime/5.1}"
ISAAC_ASSETS="${ISAAC_ASSETS:-/home/lx/isaac-assets/5.1}"
GPU="${GPU:-0}"
export PYTHONPATH="$PROJECT_ROOT"
source "$ISAACLAB_ROOT/lab/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export TMPDIR="${TMPDIR:-/home/lx/tmp}"
cd "$PROJECT_ROOT"
exec python -u record.py --headless --device "cuda:$GPU" \
  --kit_args "--portable-root $ISAAC_RUNTIME --/renderer/activeGpu=$GPU --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/persistent/isaac/asset_root/cloud=$ISAAC_ASSETS" "$@"
