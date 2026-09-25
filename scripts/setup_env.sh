#!/usr/bin/env bash
# Create .venv (Python 3.11) with the pinned stack, CUDA jax and CPU torch.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/rakshith/miniconda3/envs/py311tmp/bin/python3.11}
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -q "pip==26.2"   # 26.2.1 breaks wheel unzip on this python
pip install -q -r requirements.txt
pip install -q "jax[cuda12]==0.9.2"
pip install -q "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu
python -c "import mujoco, jax, brax, torch; print('mujoco', mujoco.__version__, 'jax', jax.__version__, 'brax', brax.__version__, 'torch', torch.__version__)"
