#!/usr/bin/env python
"""Step 1 of the export contract (spec section 8): flax -> torch (CPU).

Loads the student orbax checkpoint, reconstructs the deployment policy
(obs normalisation + adaptation 1D-CNN + trunk MLP) as a torch module layer by
layer, copies weights, and asserts forward-pass parity with the JAX model on
1000 random inputs (max abs diff < 1e-4, float32).

  python export/params_to_torch.py --checkpoint outputs/checkpoints/student_v1/ckpt_final \
                                   [--out outputs/onnx/student.pt]

torch here is CPU-only and used exclusively for conversion — never training.
"""

import argparse
import json
import os

# CPU by design (eval/export path). setdefault: an explicit env override
# still wins; without this, jax auto-discovers the CUDA plugin, which on this
# box trips the stale-nvJitLink LD_LIBRARY_PATH shadow (see backend.py).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from x1_locomotion.env import STUDENT_OBS_SIZE  # noqa: E402

from x1_locomotion.layout import NU  # noqa: E402

ACTION_SIZE = NU


class StudentPolicyNet(nn.Module):
    """Deployment policy: raw obs history (B, H, obs) -> action (B, NU) in [-1, 1].
    Mirrors cpu_eval.StudentJaxPolicy exactly (normalisation baked in)."""

    def __init__(self, meta: dict):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(STUDENT_OBS_SIZE))
        self.register_buffer("obs_std", torch.ones(STUDENT_OBS_SIZE))
        a = meta["adaptation"]
        chans = [STUDENT_OBS_SIZE] + list(a["conv_channels"])
        self.convs = nn.ModuleList([
            nn.Conv1d(chans[i], chans[i + 1], a["conv_kernels"][i],
                      stride=a["conv_strides"][i])  # VALID padding == padding 0
            for i in range(len(a["conv_channels"]))
        ])
        self.adapt_dense0 = nn.Linear(meta["conv_flat_dim"], a["dense"])
        self.adapt_dense1 = nn.Linear(a["dense"], meta["latent_size"])
        trunk_sizes = ([STUDENT_OBS_SIZE + meta["latent_size"]]
                       + list(meta["policy_hidden"]) + [2 * ACTION_SIZE])
        self.trunk = nn.ModuleList([
            nn.Linear(trunk_sizes[i], trunk_sizes[i + 1])
            for i in range(len(trunk_sizes) - 1)
        ])

    def forward(self, hist):
        x = (hist - self.obs_mean) / self.obs_std
        y = x.transpose(1, 2)  # (B, C, L) for Conv1d
        for conv in self.convs:
            y = torch.nn.functional.silu(conv(y))
        # flatten in (L, C) order to match flax reshape on (..., L, C)
        y = y.transpose(1, 2).flatten(1)
        z = torch.nn.functional.silu(self.adapt_dense0(y))
        z = self.adapt_dense1(z)
        a = torch.cat([x[:, -1], z], dim=-1)
        for lin in self.trunk[:-1]:
            a = torch.nn.functional.silu(lin(a))
        logits = self.trunk[-1](a)
        return torch.tanh(logits[:, :ACTION_SIZE])


def _linear_from_flax(linear: nn.Linear, p):
    linear.weight.data = torch.from_numpy(np.asarray(p["kernel"]).T.copy())
    linear.bias.data = torch.from_numpy(np.asarray(p["bias"]).copy())


def _conv_from_flax(conv: nn.Conv1d, p):
    # flax Conv kernel (k, in, out) -> torch (out, in, k)
    conv.weight.data = torch.from_numpy(
        np.asarray(p["kernel"]).transpose(2, 1, 0).copy())
    conv.bias.data = torch.from_numpy(np.asarray(p["bias"]).copy())


def build_torch_policy(checkpoint: str):
    from x1_locomotion.config import load_config
    from x1_locomotion.ppo_teacher import load_checkpoint, rehydrate_normalizer

    cfg = load_config()
    net = cfg.train.networks
    params, _ = load_checkpoint(checkpoint)
    if not (isinstance(params, dict) and "adaptation" in params):
        raise SystemExit("expected a STUDENT checkpoint (with 'adaptation'); "
                         "teachers are not exported — distill first.")
    norm = rehydrate_normalizer(params["teacher"][0])
    trunk_p = params["teacher"][1]["params"]["trunk"]
    adapt_p = params["adaptation"]["params"]

    a = net.adaptation
    # VALID conv output length after the stack
    L = a.history_length
    for k, s in zip(a.conv_kernels, a.conv_strides):
        L = (L - k) // s + 1
    meta = {
        "latent_size": net.latent_size,
        "policy_hidden": list(net.policy_hidden),
        "adaptation": {"conv_channels": list(a.conv_channels),
                       "conv_kernels": list(a.conv_kernels),
                       "conv_strides": list(a.conv_strides),
                       "dense": a.dense},
        "history_length": a.history_length,
        "conv_flat_dim": L * a.conv_channels[-1],
    }
    model = StudentPolicyNet(meta)
    model.obs_mean.data = torch.from_numpy(np.asarray(norm.mean["state"]).copy())
    model.obs_std.data = torch.from_numpy(np.asarray(norm.std["state"]).copy())
    for i, conv in enumerate(model.convs):
        _conv_from_flax(conv, adapt_p[f"conv_{i}"])
    _linear_from_flax(model.adapt_dense0, adapt_p["dense_0"])
    _linear_from_flax(model.adapt_dense1, adapt_p["dense_1"])
    for i, lin in enumerate(model.trunk):
        _linear_from_flax(lin, trunk_p[f"hidden_{i}"])
    model.eval()
    return model, meta, cfg


def verify_parity(model, meta, cfg, checkpoint, n: int = 1000, tol: float = 1e-4):
    """JAX vs torch forward-pass parity on plausible observations.

    Inputs are sampled as mean + 3σ·noise in RAW obs space (the checkpoint's
    own normalizer statistics), not unit-normal: dims that were near-constant
    in training have tiny σ, and unit-normal raw inputs on those dims would be
    amplified ~1e6× by the in-graph normalisation — measuring float32 noise on
    inputs the network can never see, instead of agreement where it matters."""
    from x1_locomotion.cpu_eval import StudentJaxPolicy
    import jax.numpy as jnp

    jax_pol = StudentJaxPolicy(checkpoint, cfg)
    rng = np.random.default_rng(0)
    H = meta["history_length"]
    worst = 0.0
    noise = rng.standard_normal((n, H, STUDENT_OBS_SIZE)).astype(np.float32)
    batch = (jax_pol._mean + 3.0 * jax_pol._std * noise).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(batch)).numpy()
    # jax path, batched, mirroring StudentJaxPolicy.__call__
    h_n = (batch - jax_pol._mean) / jax_pol._std
    z = jax_pol._adapt.apply(jax_pol._adapt_params, jnp.asarray(h_n))
    logits = jax_pol._module.apply(jax_pol._policy_params, jnp.asarray(h_n[:, -1]), z,
                                   method="act_with_latent")
    jax_out = np.asarray(jnp.tanh(logits[:, :ACTION_SIZE]))
    worst = float(np.max(np.abs(torch_out - jax_out)))
    assert worst < tol, f"parity FAILED: max abs diff {worst:.2e} >= {tol}"
    print(f"parity OK: max abs diff {worst:.2e} over {n} random inputs")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from x1_locomotion.config import REPO_ROOT
    model, meta, cfg = build_torch_policy(args.checkpoint)
    worst = verify_parity(model, meta, cfg, args.checkpoint)

    out = args.out or os.path.join(REPO_ROOT, "outputs", "onnx", "student.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta,
                "source_checkpoint": os.path.abspath(args.checkpoint),
                "parity_max_abs_diff": worst}, out)
    print(f"torch policy -> {out}")


if __name__ == "__main__":
    main()
