#!/usr/bin/env bash
# Create .venv (Python 3.11) with the pinned MuJoCo / JAX / brax stack.
#
#   bash scripts/setup_env.sh              # CPU only: enough to run and test the policy
#   CUDA=1 bash scripts/setup_env.sh       # + CUDA jax, needed to TRAIN on an NVIDIA GPU
#   EXPORT=1 bash scripts/setup_env.sh     # + CPU torch, needed to export new ONNX files
#   PY=/path/to/python3.11 bash scripts/setup_env.sh   # use a specific interpreter
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3.11}
command -v "$PY" >/dev/null || { echo "Python 3.11 not found. Install it or set PY=/path/to/python3.11"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -q "pip==26.2"   # 26.2.1 fails to unzip some wheels on Python 3.11
pip install -q -r requirements.txt
if [ "${CUDA:-0}" = "1" ]; then pip install -q "jax[cuda12]==0.9.2"; fi
if [ "${EXPORT:-0}" = "1" ]; then pip install -q "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu; fi
python -c "import mujoco, jax, brax, onnxruntime; print('ok: mujoco', mujoco.__version__, '| jax', jax.__version__, jax.devices(), '| brax', brax.__version__)"
