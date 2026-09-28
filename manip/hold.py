#!/usr/bin/env python
"""Static squeeze-hold test, pelvis fixed at 0.613 m, arm PD at the sidecar
gains and torque limits.

Sequence: arms at a pre-grasp pose (squeeze pose, shoulder roll 0.25 rad out) ->
move to the squeeze pose commanded d inside the faces (1 s) -> settle 0.5 s
-> lift the box (command LIFT_DZ, 1 s) -> remove the table -> hold 5 s.
Measured: achieved arm lift (table still there), sag in the first second
without the table, then over the last 4 s: box drop (slip), tilt, table and
floor force (must be 0), normal force per side, arm joint torques vs limits.
Held = no support, drop < 1 cm over 4 s, tilt < 15 deg.

Two contact sets: full (forearm and upper-arm capsules collide with the box)
and pad_only (only the gripper pads do, i.e. a gripper standing proud of the
forearm). Gravity feedforward on the arms is optional (grav_comp).

  python manip/hold.py      # sweep -> manip/out/hold_sweep.json
"""

import json
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARM, ARM_NAMES, BOX_HALF, DEFAULT_POSE, TAU_MAX, Scene  # noqa: E402
from ik import ArmIK, solve_squeeze  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DT = 0.002
PRE_ROLL = 0.25             # pre-grasp: squeeze pose with the shoulders abducted
LIFT_DZ = 0.08              # commanded; the arms sag under load (reported)
TABLE_EDGE_MIN = 0.15       # table front edge never closer to the robot than this


def table_for(scene, xc, top):
    edge = max(TABLE_EDGE_MIN, xc - 0.15)
    scene.set_table(top, x_centre=edge + scene.m.geom_size[scene.table_gid, 0])


def pregrasp(q_sq):
    q = np.array(q_sq, float).copy()
    q[[1, 7]] -= PRE_ROLL
    return q


class Plan:
    """Arm waypoints from IK (on its own fixed-base scene), with continuation
    seeding."""

    def __init__(self, mode="forearm"):
        self.ik = ArmIK(Scene(fixed=True), mode)
        self.seed = None

    def pose(self, centre, depth):
        q, ok, info = solve_squeeze(self.ik, centre, depth, seed=self.seed)
        if ok:
            self.seed = q
        return q, ok, info


GRAV_COMP = False           # arm gravity feedforward (set by the caller)


def run_segments(scene, segments, log_every=10, on_step=None):
    """segments: [(duration_s, q_from(12), q_to(12), tag)], linear joint
    interpolation of the arm command; legs held at the default pose (fixed
    base). Returns a log of dicts."""
    s = scene
    log = []
    t = 0.0
    for dur, qa, qb, tag in segments:
        n = int(round(dur / DT))
        for i in range(n):
            u = (i + 1) / n if tag.startswith("move") else 1.0
            arm = qa + u * (qb - qa)
            tgt = DEFAULT_POSE.copy()
            tgt[ARM] = arm
            tau = s.pd(tgt, GRAV_COMP)
            mujoco.mj_step(s.m, s.d)
            t += DT
            if on_step is not None:
                on_step(s, t, tag)
            if i % log_every == 0 or i == n - 1:
                c = s.box_contacts()
                log.append({"t": t, "tag": tag, "box": s.box_pos().tolist(),
                            "tilt": s.box_tilt_deg(), "tau": tau[ARM].tolist(),
                            "q": s.q()[ARM].tolist(), **c})
    return log


def hand_frame_pos(scene):
    """Box centre in the left wrist frame (slip measure)."""
    b = scene.m.body("left_wrist_roll").id
    R = scene.d.xmat[b].reshape(3, 3)
    return R.T @ (scene.box_pos() - scene.d.xpos[b])


DEFAULTS = dict(hb=0.75, xc=0.37, depth=0.10, mu=0.5, pull=0.0, torsion=None,
                mode="pad", grav_comp=True, pad_only=False)


def hold_trial(cfg):
    """cfg: dict over DEFAULTS. hb box bottom, xc box centre x, depth squeeze
    depth, pull (hold pose solved for the box this much further back),
    torsion (contact patch, m), mode (IK contact), grav_comp (arm feedforward),
    pad_only (only the pads collide with the box)."""
    import ik as ikmod
    c = dict(DEFAULTS, **cfg)
    global GRAV_COMP
    GRAV_COMP = c["grav_comp"]
    ikmod.BOX_CLEAR = -1.0 if c["pad_only"] else 0.01
    s = Scene(fixed=True)
    s.set_friction(c["mu"], c["torsion"])
    if c["pad_only"]:
        for g in s.arm_gids:
            if g not in s.pad_gid:
                s.m.geom_contype[g] = 0
    plan = Plan(c["mode"])
    hb, xc = c["hb"], c["xc"]
    centre = np.array([xc, 0.0, hb + BOX_HALF[2]])
    q_sq, ok2, info = plan.pose(centre, c["depth"])
    q_pre = pregrasp(q_sq)
    plan.seed = q_sq
    q_lift, ok3, _ = plan.pose(centre + [-c["pull"], 0, LIFT_DZ], c["depth"])
    jump = float(np.abs(q_lift - q_sq).max())
    res = dict(c, max_joint_jump=jump, ik_ok=bool(ok2 and ok3 and jump < 0.6),
               q_squeeze=q_sq.tolist(), q_lift=q_lift.tolist())
    if not res["ik_ok"]:
        return res
    s.reset(q_pre)
    table_for(s, xc, hb)
    s.place_box(centre + [0, 0, 0.002])
    mujoco.mj_forward(s.m, s.d)
    segs = [(0.5, q_pre, q_pre, "settle"), (1.0, q_pre, q_sq, "move_in"),
            (0.5, q_sq, q_sq, "squeeze"), (1.0, q_sq, q_lift, "move_lift")]
    log = run_segments(s, segs)
    arm_lift = float(s.box_pos()[2] - centre[2])
    lift_table_force = log[-1]["table"]
    # take the table away so the arms carry the whole box, then hold 5 s
    s.set_table(None)
    z_rel = s.box_pos()[2]
    hold = run_segments(s, [(1.0, q_lift, q_lift, "hold")])
    p0, z0 = hand_frame_pos(s), s.box_pos()[2]
    hold += run_segments(s, [(4.0, q_lift, q_lift, "hold")])
    p1, z1 = hand_frame_pos(s), s.box_pos()[2]
    H = hold[len(hold) // 5:]           # last 4 s
    tau = np.array([h["tau"] for h in H])
    res.update({
        "arm_lift": arm_lift, "lift_table_force": lift_table_force,
        "sag_first_1s": float(z_rel - z0), "drop_1_5s": float(z0 - z1),
        "slip_1_5s": float(np.linalg.norm(p1 - p0)),
        "tilt_end": s.box_tilt_deg(),
        "table_force_end": hold[-1]["table"], "floor_force_end": hold[-1]["floor"],
        "torso_force": float(np.mean([h["torso"] for h in H])),
        "normal_left": float(np.mean([h["left"] for h in H])),
        "normal_right": float(np.mean([h["right"] for h in H])),
        "pad_share": float(np.mean([h["left_pad"] for h in H])
                           / max(1e-6, np.mean([h["left"] for h in H]))),
        "tau_mean": np.abs(tau).mean(0).tolist(), "tau_max": np.abs(tau).max(0).tolist(),
        "tau_ratio_max": (np.abs(tau).max(0) / TAU_MAX[ARM]).tolist(),
    })
    res["held"] = bool(res["table_force_end"] < 1e-6 and res["floor_force_end"] < 1e-6
                       and res["drop_1_5s"] < 0.01 and res["tilt_end"] < 15.0)
    return res


def fmt(r):
    head = (f"{r['mode']}{'/pad-only' if r['pad_only'] else ''} gc={int(r['grav_comp'])} "
            f"hb={r['hb']:.2f} mu={r['mu']:.1f} d={r['depth']:.3f} tors={r['torsion']}")
    if not r.get("ik_ok"):
        return head + ": IK failed"
    worst = int(np.argmax(r["tau_ratio_max"]))
    return (head + f" held={r['held']!s:5} lift={r['arm_lift'] * 100:4.1f}cm "
            f"sag={r['sag_first_1s'] * 1000:5.1f}mm drop1-5s={r['drop_1_5s'] * 1000:6.1f}mm "
            f"tilt={r['tilt_end']:5.1f} N={r['normal_left']:5.1f}/{r['normal_right']:5.1f} "
            f"pad%={100 * r['pad_share']:3.0f} torso={r['torso_force']:4.1f} "
            f"worst={ARM_NAMES[worst % 6]} {r['tau_ratio_max'][worst] * 100:3.0f}%")


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for pad_only in (False, True):
        for mu in (0.3, 0.5, 0.8):
            for d in (0.04, 0.08, 0.10, 0.12, 0.14, 0.16):
                jobs.append(dict(mu=mu, depth=d, pad_only=pad_only))
    for d in (0.10, 0.14):
        jobs.append(dict(depth=d, pad_only=True, grav_comp=False))
        jobs.append(dict(depth=d, pad_only=True, torsion=0.02))
        for hb in (0.6, 0.9):
            jobs.append(dict(depth=d, pad_only=True, hb=hb))
    with Pool(24) as p:
        res = p.map(hold_trial, jobs)
    with open(os.path.join(OUT, "hold_sweep.json"), "w") as f:
        json.dump(res, f, indent=1)
    for r in res:
        print(fmt(r))


if __name__ == "__main__":
    main()
