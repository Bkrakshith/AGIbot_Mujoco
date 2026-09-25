#!/usr/bin/env python
"""Phase B/C CLI: distill the student, optionally fine-tune (gated).

Phase B:
  python train/train_student.py --teacher_ckpt outputs/checkpoints/teacher_v1/ckpt_N \
                                --run_name student_v1

Phase C (only if Phase B student shows >2% survival drop vs teacher — pass the
two battery JSONs so the gate is checked, or --force to override):
  python train/train_student.py --finetune --student_ckpt .../student_v1/ckpt_final \
      --run_name student_v1_ft --teacher_eval outputs/eval/teacher.json \
      --student_eval outputs/eval/student.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from x1_locomotion.backend import select_backend  # noqa: E402


def _finetune_gate(teacher_eval: str, student_eval: str, gate: float) -> bool:
    """True if the student's survival drop vs the teacher exceeds the gate on
    any battery scenario (i.e. Phase C is warranted)."""
    with open(teacher_eval) as f:
        t = json.load(f)["scenarios"]
    with open(student_eval) as f:
        s = json.load(f)["scenarios"]
    worst = 0.0
    for name in t:
        if name in s:
            worst = max(worst, t[name]["survival"] - s[name]["survival"])
    print(f"[gate] worst survival drop student vs teacher: {worst:.2%} "
          f"(threshold {gate:.2%})")
    return worst > gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--backend", choices=["cpu", "gpu", "metal"], default="cpu")
    ap.add_argument("--num_envs", type=int, default=None)
    # Phase B
    ap.add_argument("--teacher_ckpt", help="teacher checkpoint (Phase B)")
    # Phase C
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--student_ckpt", help="Phase B checkpoint (Phase C)")
    ap.add_argument("--teacher_eval", help="battery JSON of the teacher")
    ap.add_argument("--student_eval", help="battery JSON of the Phase B student")
    ap.add_argument("--force", action="store_true",
                    help="run Phase C without the survival-drop gate")
    args = ap.parse_args()

    select_backend(args.backend)
    from x1_locomotion.config import load_config
    from x1_locomotion.distill_student import distill, finetune

    cfg = load_config()
    distill_env_fn = finetune_env_fn = None

    if args.finetune:
        if not args.student_ckpt:
            ap.error("--finetune requires --student_ckpt")
        if not args.force:
            if not (args.teacher_eval and args.student_eval):
                ap.error("--finetune requires --teacher_eval and --student_eval "
                         "(or --force). Run eval/run_battery.py on both first.")
            if not _finetune_gate(args.teacher_eval, args.student_eval,
                                  cfg.train.finetune.survival_drop_gate):
                print("[gate] student within tolerance of teacher — Phase C "
                      "skipped per spec. Use --force to run anyway.")
                return
        finetune(args.student_ckpt, args.run_name, cfg=cfg,
                 num_envs=args.num_envs, env_fn=finetune_env_fn)
    else:
        if not args.teacher_ckpt:
            ap.error("Phase B requires --teacher_ckpt")
        distill(args.teacher_ckpt, args.run_name, cfg=cfg,
                num_envs=args.num_envs, env_fn=distill_env_fn)


if __name__ == "__main__":
    main()
