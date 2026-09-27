#!/usr/bin/env python
"""Evaluation battery for the X1 stand/walk policy — the only number that counts.

13 scenarios x N episodes (default 50) on the FULL-FIDELITY model in plain CPU
MuJoCo: stand, stand + pushes, walk fwd/back, rotate, sidestep, turn-while-
walking, walk + pushes, stop-and-go, and walking at 0.5 m/s with the arms held
forward / sideways / overhead or moving through task poses. Report survival with Wilson CIs.
With --onnx this is the final export gate (--gate 0.9).
A survival regression against outputs/eval/best.json fails the script (exit 1).

Usage:
  python eval/run_battery.py --checkpoint outputs/checkpoints/run/ckpt_N
  python eval/run_battery.py --onnx outputs/onnx/student.onnx [--gate 0.9]
"""

import argparse
import json
import os

# CPU by design (eval/export path). setdefault: an explicit env override
# still wins; without this, jax auto-discovers the CUDA plugin, which on this
# box trips the stale-nvJitLink LD_LIBRARY_PATH shadow (see backend.py).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from x1_locomotion.config import REPO_ROOT, load_config  # noqa: E402
from x1_locomotion.cpu_eval import CpuRollout, Scenario, make_policy  # noqa: E402
from metrics import compare, summarize  # noqa: E402

EVAL_DIR = os.path.join(REPO_ROOT, "outputs", "eval")
NO_LOAD = [(0.0, [0.0, 0.0], [[0.0] * 3, [0.0] * 3])]


def _in_envelope(cfg, cmd):
    """Battery commands must lie inside the TRAINING envelope (failure M2)."""
    c = cfg.train.commands
    for v, (lo, hi), name in zip(cmd, (c.vx_range, c.vy_range, c.yaw_range,
                                       c.height_range),
                                 ("vx", "vy", "yaw", "height")):
        if not lo - 1e-9 <= v <= hi + 1e-9:
            raise ValueError(f"battery {name}={v} outside training range [{lo}, {hi}]")
    return list(cmd)


def battery(cfg=None) -> list[Scenario]:
    """X1 stand/walk battery, 13 scenarios (9 locomotion + 4 task-arm). Commands are DERIVED from the
    config and checked against the training envelope. Push magnitudes stay
    below the final stage's push_max_n (60 N)."""
    cfg = cfg or load_config()
    h = float(cfg.train.commands.height_range[0])
    cmd = lambda vx, vy, wz: _in_envelope(cfg, [vx, vy, wz, h])
    stand = cmd(0.0, 0.0, 0.0)
    pushes = [(t, [40.0 if i % 2 == 0 else -40.0, 20.0 if i % 3 == 0 else -20.0, 0.0])
              for i, t in enumerate([5.0, 10.0, 15.0])]
    walk_pushes = [(6.0, [0.0, 35.0, 0.0]), (12.0, [-35.0, 0.0, 0.0])]
    lib = [list(p) for p in cfg.domain_rand.arm_poses]   # both arms: tile per arm
    arms = lambda i: lib[i] + lib[i]
    hold_arms = lambda i: [(0.0, arms(i))]
    # arms move through poses while walking (1 s moves, as in training)
    moving = [(0.0, arms(0)), (2.0, arms(0)), (3.0, arms(1)), (6.0, arms(1)),
              (7.0, arms(3)), (10.0, arms(3)), (11.0, arms(4)), (14.0, arms(4)),
              (15.0, arms(6)), (18.0, arms(6)), (19.0, arms(0))]
    walk_arms = cmd(0.5, 0.0, 0.0)
    return [
        Scenario("a_stand_still", stand, payload_schedule=NO_LOAD),
        Scenario("b_stand_pushes", stand, payload_schedule=NO_LOAD, pushes=pushes),
        Scenario("c_walk_forward", cmd(0.8, 0.0, 0.0), payload_schedule=NO_LOAD),
        Scenario("d_walk_backward", cmd(-0.3, 0.0, 0.0), payload_schedule=NO_LOAD),
        Scenario("e_rotate_in_place", cmd(0.0, 0.0, 0.5), payload_schedule=NO_LOAD),
        Scenario("f_sidestep", cmd(0.0, 0.3, 0.0), payload_schedule=NO_LOAD),
        Scenario("g_turn_while_walking", cmd(0.5, 0.0, 0.4), payload_schedule=NO_LOAD),
        Scenario("h_walk_pushes", cmd(0.5, 0.0, 0.0), payload_schedule=NO_LOAD,
                 pushes=walk_pushes),
        # task arm poses while walking (library indices, configs/domain_rand.yaml)
        Scenario("j_walk_arms_forward", walk_arms, payload_schedule=NO_LOAD,
                 arm_schedule=hold_arms(1)),
        Scenario("k_walk_arms_sideways", walk_arms, payload_schedule=NO_LOAD,
                 arm_schedule=hold_arms(3)),
        Scenario("l_walk_arms_overhead", walk_arms, payload_schedule=NO_LOAD,
                 arm_schedule=hold_arms(4)),
        Scenario("m_walk_arms_moving", walk_arms, payload_schedule=NO_LOAD,
                 arm_schedule=moving),
        # stand -> walk -> stand, ramped over 0.5 s
        Scenario("i_stop_and_go", stand, payload_schedule=NO_LOAD,
                 cmd_schedule=[(3.0, stand), (3.5, cmd(0.6, 0.0, 0.0)),
                               (11.0, cmd(0.6, 0.0, 0.0)), (11.5, stand)]),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="teacher or student orbax checkpoint")
    ap.add_argument("--onnx", help="exported ONNX policy (final gate mode)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--gate", type=float, default=None,
                    help="fail if any scenario survival is below this (e.g. 0.9)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--obs_noise", action="store_true",
                    help="add the full X1 sensor noise to observations (grade a "
                         "student / ONNX policy under deployment conditions)")
    ap.add_argument("--no-best-update", action="store_true")
    args = ap.parse_args()
    if not args.checkpoint and not args.onnx:
        ap.error("one of --checkpoint / --onnx is required")

    cfg = load_config()
    runner = CpuRollout(cfg, obs_noise=args.obs_noise)
    policy = make_policy(cfg, checkpoint=args.checkpoint, onnx=args.onnx)

    name = os.path.basename((args.onnx or args.checkpoint).rstrip("/"))
    report = {}
    scenarios = battery(cfg)
    print(f"[battery] obs_noise={args.obs_noise}, {len(scenarios)} scenarios, "
          f"{args.episodes} episodes each", flush=True)
    for sc in scenarios:
        results = []
        for ep in range(args.episodes):
            if hasattr(policy, "reset"):
                policy.reset()
            results.append(runner.run_episode(sc, policy, seed=ep))
        report[sc.name] = summarize(results)
        print(f"[{sc.name}] survival={report[sc.name]['survival']:.2%} "
              f"ori_err={report[sc.name]['mean_orientation_err']:.3f} "
              f"trk_err={report[sc.name]['mean_tracking_err']:.3f} "
              f"recovery={report[sc.name]['median_recovery_s']}", flush=True)

    os.makedirs(EVAL_DIR, exist_ok=True)
    out_path = args.out or os.path.join(EVAL_DIR, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump({"policy": args.onnx or args.checkpoint, "obs_noise": args.obs_noise,
                   "episodes": args.episodes, "scenarios": report}, f, indent=2)
    print(f"results -> {out_path}")

    failed = False
    if args.gate is not None:
        for sc_name, r in report.items():
            if r["survival"] < args.gate:
                print(f"GATE FAIL: {sc_name} survival {r['survival']:.2%} < {args.gate:.0%}")
                failed = True

    best_path = os.path.join(EVAL_DIR, "best.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)["scenarios"]
        regressions = compare(report, best)
        for r in regressions:
            print(f"REGRESSION vs best: {r}")
        failed = failed or bool(regressions)
    if not failed and not args.no_best_update:
        shutil.copy(out_path, best_path)
        print(f"new best -> {best_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
