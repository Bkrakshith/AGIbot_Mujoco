#!/usr/bin/env python
"""Phase A CLI: train the privileged teacher through the curriculum.

  python train/train_teacher.py --run_name teacher_v1 [--num_envs 8192]
                                [--backend cpu|gpu] [--resume]
  python train/train_teacher.py --run_name smoke --dry_run   # pipeline check

Long runs: tmux, and on this Linux box ALWAYS prefix with
`taskset -c 0-7,10-31` (logical CPUs 8/9 segfault under load).
Checkpoints land in outputs/checkpoints/<run_name>/ at every eval (default
cadence keeps this well under 5 minutes of wall time); --resume picks up the
latest one (restores params + normalizer + curriculum stage + seed; see
ppo_teacher.py for the optimizer-state caveat).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from x1_locomotion.backend import select_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--num_envs", type=int, default=None,
                    help="override configs/train.yaml ppo.num_envs (bench knee point)")
    ap.add_argument("--backend", choices=["cpu", "gpu", "metal"], default="cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="stage 0 only, 200k steps, gates off: runs the REAL "
                         "brax training scan to catch structural mismatches "
                         "(metric keys, pytrees) that unit tests cannot see")
    args = ap.parse_args()

    select_backend(args.backend)
    from x1_locomotion.config import _deep_merge, load_config
    from x1_locomotion.ppo_teacher import train_teacher
    cfg = load_config()
    if args.dry_run:
        s0 = dict(cfg.train.curriculum[0], num_timesteps=200_000,
                  gate_tracking=0.0, gate_air_time=0.0)
        cfg.train = _deep_merge(cfg.train, {
            "curriculum": [s0],
            "checkpoint": {"eval_every_steps": 100_000}})
        print("[teacher] DRY RUN: stage 0 only, 200k steps, gates off", flush=True)
    train_teacher(args.run_name, num_envs=args.num_envs, resume=args.resume,
                  cfg=cfg)


if __name__ == "__main__":
    main()
