#!/usr/bin/env python
"""Place test: pick the box at the chest-hold height (box bottom 0.75 m),
carry it, then lower it onto a table of another height and release.

Arm path: squeeze IK waypoints every 2.5 cm of box height (continuation
seeded), 0.5 s each; the box is first raised 6 cm above the new table top if
needed, the table appears, and the box is lowered until it rests 2 cm below the table
top in the command (the arms sag, so the box lands before that), then the
arms open to a pose with the pads 5 cm outside the faces (release). Success = after 2 s the box rests on the table (table force within
20 % of m*g, no arm contact, tilt < 10 deg, box centre over the table).

Two bases: fixed pelvis (0.613 m) and the floating base with the ONNX policy
standing (same loop as balance.py).

  python manip/place.py        # -> manip/out/place.json
"""

import json
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from balance import CTRL_DT, DECIM, HEIGHT_CMD, NOISE, Policy  # noqa: E402
from common import (ARM, BOX_HALF, BOX_MASS, DEFAULT_POSE, G, KD, KP, SC, TAU_MAX,  # noqa: E402
                    Scene, quat_rotate_inv)
from hold import LIFT_DZ, Plan, pregrasp, table_for  # noqa: E402

OUT = os.path.join(HERE, "out")
PICK_HB, XC, DEPTH, MU = 0.75, 0.37, 0.14, 0.5
STEP_DZ = 0.025
RAISE_CLEAR = 0.03          # box bottom above the new table top before it appears
Z_CMD_MAX = 1.10 + BOX_HALF[2]   # top of the squeeze IK band (box bottom 1.10)


def waypoints(plan, z_from, z_to):
    """Squeeze poses for box-centre heights z_from -> z_to."""
    n = max(1, int(np.ceil(abs(z_to - z_from) / STEP_DZ)))
    out = []
    for z in np.linspace(z_from, z_to, n + 1)[1:]:
        q, ok, _ = plan.pose(np.array([XC, 0.0, z]), DEPTH)
        if not ok:
            return out, False
        plan.seed = q
        out.append(q)
    return out, True


class Runner:
    def __init__(self, floating, seed=0):
        self.s = Scene(fixed=not floating)
        self.floating = floating
        s = self.s
        s.set_friction(MU)
        for g in s.arm_gids:
            if g not in s.pad_gid:
                s.m.geom_contype[g] = 0
        self.rng = np.random.default_rng(seed)
        if floating:
            self.pol = Policy()
            self.phase = self.rng.uniform(0, 2 * np.pi)
            self.last = np.zeros(12, np.float32)
        self.fell = False

    def step(self, arm):
        """Advance 0.01 s with the arm command."""
        s, d = self.s, self.s.d
        if self.floating:
            rng = self.rng
            g = quat_rotate_inv(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
            obs = np.concatenate([
                g + rng.normal(0, NOISE["gravity"], 3),
                d.qvel[3:6] + rng.normal(0, NOISE["gyro"], 3),
                s.q() - DEFAULT_POSE + rng.normal(0, NOISE["joint_pos"], 24),
                s.qd() + rng.normal(0, NOISE["joint_vel"], 24),
                self.last, [0.0, 0.0, 0.0, HEIGHT_CMD], arm,
                [np.sin(self.phase), np.cos(self.phase)]]).astype(np.float32)
            act = np.clip(self.pol(obs), -1, 1)
            self.last = act.astype(np.float32)
            self.phase = (self.phase + 2 * np.pi * CTRL_DT * SC["gait_clock_hz"]) % (2 * np.pi)
            legs = DEFAULT_POSE[:12] + 0.5 * act
        else:
            legs = DEFAULT_POSE[:12]
        tgt = np.concatenate([legs, arm])
        for _ in range(DECIM):
            tau = KP * (tgt - s.q()) - KD * s.qd()
            d.ctrl[s.act] = np.clip(tau, -TAU_MAX, TAU_MAX)
            mujoco.mj_step(s.m, d)
        if self.floating:
            gt = quat_rotate_inv(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
            if d.qpos[2] < 0.35 or -gt[2] < np.cos(1.0):
                self.fell = True

    def move(self, qa, qb, dur):
        n = max(1, int(round(dur / CTRL_DT)))
        for i in range(n):
            self.step(qa + (i + 1) / n * (qb - qa))
            if self.fell:
                return


def place_trial(args):
    place_hb, floating, seed = args
    import ik as ikmod
    ikmod.BOX_CLEAR = -1.0
    plan = Plan("pad")
    zc0 = PICK_HB + BOX_HALF[2]
    q_sq, ok, _ = plan.pose(np.array([XC, 0.0, zc0]), DEPTH)
    plan.seed = q_sq
    carry, ok2 = waypoints(plan, zc0, zc0 + LIFT_DZ)
    # the arms lag under the load: when the new table is higher than the
    # carry height, the command keeps rising (up to the IK band) until the
    # box bottom is RAISE_CLEAR above the new table top
    up = []
    if place_hb + RAISE_CLEAR > PICK_HB:
        up, _ = waypoints(plan, zc0 + LIFT_DZ, Z_CMD_MAX)
    res = {"place_hb": place_hb, "floating": floating, "seed": seed,
           "ik_ok": bool(ok and ok2)}
    if not res["ik_ok"]:
        return res
    r = Runner(floating, seed)
    s = r.s
    q_pre = pregrasp(q_sq)
    s.reset(q_pre)
    table_for(s, XC, PICK_HB)
    s.place_box([XC, 0.0, zc0 + 0.002])
    mujoco.mj_forward(s.m, s.d)
    r.move(q_pre, q_pre, 1.5)
    r.move(q_pre, q_sq, 1.0)
    r.move(q_sq, q_sq, 0.5)
    q = q_sq
    for w in carry:
        r.move(q, w, 0.5)
        q = w
    s.set_table(None)                    # pick table away
    r.move(q, q, 1.0)
    z_cmd = zc0 + LIFT_DZ
    for w in up:
        if s.box_pos()[2] - BOX_HALF[2] > place_hb + RAISE_CLEAR:
            break
        r.move(q, w, 0.5)
        q = w
        z_cmd += STEP_DZ
    r.move(q, q, 0.5)
    res["cmd_bottom"] = float(z_cmd - BOX_HALF[2])
    res["reached_bottom"] = float(s.box_pos()[2] - BOX_HALF[2])
    res["tilt_carry"] = s.box_tilt_deg()
    if res["reached_bottom"] < place_hb + 0.005:
        res.update(ok=False, why="arms could not raise the box above the table top")
        return res
    table_for(s, XC, place_hb)
    plan.seed = q
    down, ok3 = waypoints(plan, z_cmd, place_hb + BOX_HALF[2] - 0.02)
    res["lower_ik_ok"] = bool(ok3)
    for w in down:
        r.move(q, w, 0.5)
        q = w
        if r.fell:
            break
    r.move(q, q, 0.5)
    bp = s.box_pos()
    q_rel, okr, _ = plan.pose(np.array([XC, 0.0, bp[2]]), -0.05)   # pads 5 cm outside
    if not okr:
        q_rel = pregrasp(q) - np.tile([0, 0.35, 0, 0, 0, 0], 2)
    r.move(q, q_rel, 1.0)
    r.move(q_rel, q_rel, 2.0)
    c = s.box_contacts()
    bp = s.box_pos()
    res.update({
        "fell": r.fell, "table_force": c["table"], "arm_force": c["left"] + c["right"],
        "tilt": s.box_tilt_deg(), "box_bottom": float(bp[2] - BOX_HALF[2]),
        "box_x": float(bp[0]),
    })
    res["ok"] = bool(not r.fell and abs(c["table"] - BOX_MASS * G) < 0.2 * BOX_MASS * G
                     and c["left"] + c["right"] < 0.5 and res["tilt"] < 10.0
                     and abs(res["box_bottom"] - place_hb) < 0.02)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    heights = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    jobs = [(h, False, 0) for h in heights] + [(h, True, sd) for h in heights for sd in range(5)]
    with Pool(24) as p:
        res = p.map(place_trial, jobs)
    with open(os.path.join(OUT, "place.json"), "w") as f:
        json.dump(res, f, indent=1)
    for fl in (False, True):
        print("== floating base + policy (5 seeds)" if fl else "== fixed pelvis")
        for h in heights:
            rr = [x for x in res if x["place_hb"] == h and x["floating"] == fl]
            if not rr[0]["ik_ok"]:
                print(f"  table {h:.2f}: IK path failed ({rr[0]['ik_note']})")
                continue
            okn = sum(bool(x.get("ok")) for x in rr)
            print(f"  table {h:.2f}: success {okn}/{len(rr)} "
                  f"tilt {np.mean([x.get('tilt', np.nan) for x in rr]):.1f} deg "
                  f"table force {np.mean([x.get('table_force', np.nan) for x in rr]):.1f} N "
                  f"box bottom {np.mean([x.get('box_bottom', np.nan) for x in rr]):.3f} "
                  f"fell {sum(bool(x.get('fell')) for x in rr)} "
                  f"carry tilt {np.mean([x.get('tilt_carry', np.nan) for x in rr]):.1f} "
                  f"reached bottom {np.mean([x.get('reached_bottom', np.nan) for x in rr]):.3f} "
                  f"(cmd {rr[0].get('cmd_bottom', np.nan):.3f}) "
                  f"{[x.get('why') for x in rr if x.get('why')]}")


if __name__ == "__main__":
    main()
