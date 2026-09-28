#!/usr/bin/env python
"""Does the walking policy tolerate the two 0.43 kg grippers?

ONNX policy on the full floating-base model, full obs noise, 20 seeds per
case. Arm poses: trained library poses "carry", "reach", "down" (both arms,
exact library values) and the squeeze study's chest-hold pose (outside the
library). Gripper mass at each payload body: 0 (as trained, 0.05 kg payload
body) or 0.43 kg at the gripper centre. Scenarios: stand 20 s, walk 0.3 and
0.5 m/s 15 s, turn in place 0.4 rad/s 15 s (commands ramped over 1 s).

  python manip/gripper_mass.py      # -> manip/out/gripper_mass.json
"""

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import Scene  # noqa: E402
from runner import CTRL_DT, PolicyRunner  # noqa: E402

OUT = os.path.join(HERE, "out")
POSES = {
    "down": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "carry": [-0.5, -0.1, 0.0, 1.4, 0.0, 0.0],
    "reach": [-1.2, -0.1, 0.0, 1.0, 0.0, 0.0],
    # squeeze-study chest hold (commanded), per arm L / R
    "chest_hold": [-0.764, 0.141, -0.047, 1.335, -0.132, -0.071,
                   -0.779, 0.052, -0.105, 1.327, -0.034, -0.076],
}
SCEN = {"stand": ((0.0, 0.0, 0.0), 20.0), "walk0.3": ((0.3, 0.0, 0.0), 15.0),
        "walk0.5": ((0.5, 0.0, 0.0), 15.0), "turn0.4": ((0.0, 0.0, 0.4), 15.0)}
GRIPPER_MASS = 0.43
GRIPPER_COM = (0.0, -0.09, 0.0)      # payload frame


def arm_of(name):
    p = POSES[name]
    return np.array(p * 2 if len(p) == 6 else p, float)


def set_gripper_mass(s, mass):
    for side in ("left", "right"):
        b = s.m.body(f"{side}_payload").id
        s.m.body_mass[b] = 0.05 + mass
        s.m.body_ipos[b] = GRIPPER_COM if mass > 0 else (0.0, 0.0, 0.0)
        s.m.body_inertia[b] = (1.2e-3, 6e-4, 1.2e-3) if mass > 0 else (2e-4, 2e-4, 2e-4)


def episode(args):
    pose, mass, scen, seed = args
    s = Scene(fixed=False)
    set_gripper_mass(s, mass)
    arm = arm_of(pose)
    s.reset(arm)
    r = PolicyRunner(s, seed)
    r.perturb_legs()
    cmd, dur = SCEN[scen]
    pitch, roll, vx, wz = [], [], [], []
    n = int(dur / CTRL_DT)
    for k in range(n):
        u = min(1.0, k * CTRL_DT / 1.0)
        if not r.step(arm, [c * u for c in cmd]):
            break
        p, rl = r.pitch_roll()
        pitch.append(p)
        roll.append(rl)
        if k * CTRL_DT > 3.0:
            R = s.d.xmat[s.pelvis_bid].reshape(3, 3)
            vx.append(float((R.T @ s.d.qvel[:3])[0]))
            wz.append(float(s.d.qvel[5]))
    P = np.array(pitch)
    return {"pose": pose, "mass": mass, "scen": scen, "seed": seed, "fell": r.fell,
            "t": r.t, "pitch_mean": float(P.mean()), "pitch_absmax": float(np.abs(P).max()),
            "roll_p2p": float(np.ptp(roll)), "vx": float(np.mean(vx)) if vx else None,
            "wz": float(np.mean(wz)) if wz else None}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(p, m, sc, sd) for p in POSES for m in (0.0, GRIPPER_MASS) for sc in SCEN
            for sd in range(20)]
    with Pool(24) as pool:
        res = pool.map(episode, jobs, chunksize=4)
    with open(os.path.join(OUT, "gripper_mass.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("pose        grip  " + "  ".join(f"{sc:>28}" for sc in SCEN))
    for p in POSES:
        for m in (0.0, GRIPPER_MASS):
            cells = []
            for sc in SCEN:
                rr = [x for x in res if x["pose"] == p and x["mass"] == m and x["scen"] == sc]
                surv = sum(not x["fell"] for x in rr)
                v = np.mean([x["vx"] for x in rr if x["vx"] is not None]) if sc.startswith("walk") \
                    else np.mean([x["wz"] for x in rr if x["wz"] is not None]) if sc.startswith("turn") \
                    else 0.0
                cells.append(f"{surv:2d}/20 p{np.mean([x['pitch_mean'] for x in rr]):+5.1f}"
                             f" max{np.mean([x['pitch_absmax'] for x in rr]):5.1f} v{v:+.2f}")
            print(f"{p:11} {m:4.2f}  " + "  ".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
