#!/usr/bin/env python
"""Phase A CLI: train the privileged teacher through the curriculum.

  python train/train_teacher.py --run_name teacher_v1 [--num_envs 8192]
                                [--backend cpu|gpu] [--resume]
  python train/train_teacher.py --run_name smoke --dry_run   # pipeline check

Long runs: use tmux. If the machine has unstable CPU cores, pin the run to
the good ones with taskset (see README, Troubleshooting).
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
    ap.add_argument("--init_from", default=None,
                    help="checkpoint to warm-start from (use with --start_stage)")
    ap.add_argument("--start_stage", default=None,
                    help="curriculum stage name to start at with --init_from")
    ap.add_argument("--overlay", default=None,
                    help="config overlay under configs/ (e.g. ablations/nodr.yaml)")
    ap.add_argument("--dry_run", action="store_true",
                    help="stage 0 only, 200k steps, gates off: runs the REAL "
                         "brax training scan to catch structural mismatches "
                         "(metric keys, pytrees) that unit tests cannot see")
    args = ap.parse_args()

    select_backend(args.backend)
    from x1_locomotion.config import _deep_merge, load_config
    from x1_locomotion.ppo_teacher import train_teacher
    cfg = load_config(args.overlay)
    if args.overlay:
        print(f"[teacher] overlay: {args.overlay}", flush=True)
    if args.dry_run:
        s0 = dict(cfg.train.curriculum[0], num_timesteps=200_000,
                  gate_tracking=0.0, gate_air_time=0.0, gate_survival=0.0)
        cfg.train = _deep_merge(cfg.train, {
            "curriculum": [s0],
            "checkpoint": {"eval_every_steps": 100_000}})
        print("[teacher] DRY RUN: stage 0 only, 200k steps, gates off", flush=True)
    if bool(args.init_from) != bool(args.start_stage):
        ap.error("--init_from and --start_stage go together")
    train_teacher(args.run_name, num_envs=args.num_envs, resume=args.resume,
                  cfg=cfg, init_from=args.init_from, start_stage=args.start_stage)


if __name__ == "__main__":
    main()
