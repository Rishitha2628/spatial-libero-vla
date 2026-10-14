#!/bin/bash
# Native environment for spatial-vla — no Docker.
#   source env.sh
#
# Verified working 2026-10-13: robosuite 1.4.1 + mujoco 3.9.0 + gym 0.25.2,
# EGL offscreen rendering on the RTX 3060, LIBERO-Spatial envs building and
# stepping. Docker is deliberately not used here.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LIBERO_DIR="${LIBERO_DIR:-$HOME/LIBERO}"

source "$HERE/.venv/bin/activate"

# scripts/ first so libero_config_seed.py is importable (LIBERO's first import
# calls input() and would block forever otherwise).
#
# LeRobot is the 0.4.4 wheel installed in .venv, NOT the ~/lerobot checkout: that
# checkout is 0.5.x, which needs Python 3.12 and torch>=2.7, while this venv is
# 3.10 / torch 2.6 to keep robosuite 1.4.1 + numpy 1.26 working. Likewise
# transformers 4.57 lives in .venv; ~/lerobot_data/pydeps (transformers 5) must
# stay off the path.
export PYTHONPATH="$HERE/scripts:$LIBERO_DIR"
export HF_HOME="${HF_HOME:-$HERE/data/hf_home}"   # copy of ~/hf_home (SmolVLA base), project-local so it is writable
export MUJOCO_GL=egl          # offscreen on the NVIDIA GPU; no X needed
export PYOPENGL_PLATFORM=egl

# robosuite wants a private macro file; ours disables its GPU-rendering path,
# which matters only for on-screen windows but is harmless headless.
export ROBOSUITE_MACROS="$HERE/scripts/macros_private.py"

echo "spatial-vla env: python $(python -V 2>&1 | awk '{print $2}'), mujoco $(python -c 'import mujoco;print(mujoco.__version__)' 2>/dev/null), MUJOCO_GL=$MUJOCO_GL"
