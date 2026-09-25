#!/usr/bin/env python
"""Phase 0 gate: MJX throughput benchmark on the reduced X1 model.

Measures REAL env.step throughput (full task env: PD substeps, DR field
replacement, rewards, contacts) — the number that predicts training time.

  python scripts/bench.py                         # 256..2048 envs, CPU
  python scripts/bench.py --num_envs 256,512      # subset
  python scripts/bench.py --backend metal         # experimental, falls back
  python scripts/bench.py --quick                 # setup-time smoke (fast)

Results print as a table and land in outputs/bench.json. Pick the knee point
(env steps/s stops scaling) and set it as ppo.num_envs in configs/train.yaml.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from x1_locomotion.backend import select_backend  # noqa: E402


def bench(num_envs_list, steps, backend):
    import jax
    import jax.numpy as jnp
    from x1_locomotion.config import REPO_ROOT, load_config
    from x1_locomotion.env import X1LocomotionEnv
    from x1_locomotion.wrappers import wrap_for_training

    cfg = load_config()
    env = X1LocomotionEnv(cfg, cfg.stages[-1])  # hardest stage = honest cost
    results = {}
    for n in num_envs_list:
        wrapped = wrap_for_training(env, episode_length=cfg.train.ppo.episode_length)
        reset = jax.jit(wrapped.reset)
        step = jax.jit(wrapped.step)
        rng = jax.random.split(jax.random.PRNGKey(0), n)
        state = reset(rng)
        act = jnp.zeros((n, env.action_size))
        state = step(state, act)  # compile
        jax.block_until_ready(state.obs["state"])

        t0 = time.perf_counter()
        for _ in range(steps):
            state = step(state, act)
        jax.block_until_ready(state.obs["state"])
        dt = time.perf_counter() - t0
        sps = n * steps / dt
        results[n] = sps
        print(f"num_envs={n:5d}  {sps:12,.0f} env steps/s  "
              f"({steps} steps in {dt:.1f}s)", flush=True)

        if bool(jnp.any(jnp.isnan(state.obs["state"]))):
            print(f"WARNING: NaNs in observations on backend={backend}! "
                  "Do not train on this backend.")
            results[n] = float("nan")

    out = os.path.join(REPO_ROOT, "outputs", "bench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {}
    if os.path.exists(out):
        with open(out) as f:
            payload = json.load(f)
    payload[backend] = {str(k): v for k, v in results.items()}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"results -> {out}")

    best = max(results, key=lambda k: results[k])
    print(f"\nsuggested ppo.num_envs (knee point candidate): {best} "
          "— set it in configs/train.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", default="256,512,1024,2048")
    ap.add_argument("--steps", type=int, default=50,
                    help="control steps per measurement (x10 physics substeps)")
    ap.add_argument("--backend", choices=["cpu", "gpu", "metal"], default="cpu")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke: 64 envs, 10 steps")
    args = ap.parse_args()

    backend = select_backend(args.backend)
    if args.quick:
        bench([64], 10, backend)
    else:
        bench([int(x) for x in args.num_envs.split(",")], args.steps, backend)


if __name__ == "__main__":
    main()
