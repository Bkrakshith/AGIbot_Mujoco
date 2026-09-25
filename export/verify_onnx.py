#!/usr/bin/env python
"""Step 3 of the export contract (spec section 8): onnxruntime vs torch parity.

  python export/verify_onnx.py --onnx outputs/onnx/student.onnx \
                               --torch outputs/onnx/student.pt [--tol 1e-4]

After this passes, the FINAL gate before hand-off is the battery on the ONNX:
  python eval/run_battery.py --onnx outputs/onnx/student.onnx --gate 0.9
"""

import argparse
import os

# CPU by design (eval/export path). setdefault: an explicit env override
# still wins; without this, jax auto-discovers the CUDA plugin, which on this
# box trips the stale-nvJitLink LD_LIBRARY_PATH shadow (see backend.py).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from params_to_torch import StudentPolicyNet  # noqa: E402
from x1_locomotion.env import STUDENT_OBS_SIZE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--torch", dest="torch_path", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    bundle = torch.load(args.torch_path, weights_only=False)
    model = StudentPolicyNet(bundle["meta"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    H = bundle["meta"]["history_length"]

    # sample in RAW obs space (mean + 3σ·noise, from the stats baked into the
    # torch module) — see params_to_torch.verify_parity for why unit-normal
    # inputs mis-measure parity when some obs dims were near-constant in training
    mean = model.obs_mean.numpy()
    std = model.obs_std.numpy()
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(args.n):
        noise = rng.standard_normal((1, H, STUDENT_OBS_SIZE)).astype(np.float32)
        x = (mean + 3.0 * std * noise).astype(np.float32)
        with torch.no_grad():
            t = model(torch.from_numpy(x)).numpy()
        o = sess.run(None, {input_name: x})[0]
        worst = max(worst, float(np.max(np.abs(t - o))))
    assert worst < args.tol, f"parity FAILED: max abs diff {worst:.2e} >= {args.tol}"
    print(f"onnxruntime vs torch parity OK: max abs diff {worst:.2e} "
          f"over {args.n} inputs")


if __name__ == "__main__":
    main()
