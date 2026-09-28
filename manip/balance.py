#!/usr/bin/env python
"""Balance check: the ONNX walking policy (policy/student.onnx) on the full
floating-base scene, picking the box off a table and carrying it with the
arms in the squeeze pose. Same control loop as the evaluation battery: 100 Hz
policy, 500 Hz PD at the sidecar gains, full X1 observation noise, free-running
gait clock. Arms follow the task command through PD only (no feedforward),
exactly as in training.

Per episode: stand 1.5 s (arms at pre-grasp) -> squeeze + lift (2.5 s) ->
table removed -> stand STAND_S (or walk WALK_S at 0.3 m/s after a 1 s ramp).
Baselines: same arm poses without a box.

  python manip/balance.py      # -> manip/out/balance.json
"""

import json
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from common import (ARM, BOX_HALF, DEFAULT_POSE, KD, KP, ONNX, SC, TAU_MAX,  # noqa: E402
                    Scene, quat_rotate_inv)
from hold import LIFT_DZ, Plan, pregrasp, table_for  # noqa: E402

OUT = os.path.join(HERE, "out")
DECIM = 5
CTRL_DT = 0.01
HEIGHT_CMD = 0.613
MAX_TILT = 1.0             # configs/rewards.yaml termination
HEIGHT_MIN = 0.35
NOISE = SC["training_obs_noise_std"]
STAND_S, WALK_S = 20.0, 10.0

HOLD = dict(hb=0.75, xc=0.37, depth=0.14, mu=0.5)


class Policy:
    def __init__(self):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.H = SC["policy_input"]["shape"][1]
        self.mean = np.asarray(SC["observation_normalization"]["mean"], np.float32)
        self.hist = np.tile(self.mean, (1, self.H, 1)).astype(np.float32)

    def __call__(self, obs):
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[0, -1] = obs
        return self.sess.run(None, {self.inp: self.hist})[0][0]


def arm_poses():
    plan = Plan("pad")
    c = np.array([HOLD["xc"], 0.0, HOLD["hb"] + BOX_HALF[2]])
    q_sq, ok1, _ = plan.pose(c, HOLD["depth"])
    plan.seed = q_sq
    q_lift, ok2, _ = plan.pose(c + [0, 0, LIFT_DZ], HOLD["depth"])
    assert ok1 and ok2
    return pregrasp(q_sq), q_sq, q_lift


def episode(args):
    with_box, walk, seed = args
    import ik as ikmod
    ikmod.BOX_CLEAR = -1.0
    q_pre, q_sq, q_lift = arm_poses()
    rng = np.random.default_rng(seed)
    s = Scene(fixed=False)
    s.set_friction(HOLD["mu"])
    for g in s.arm_gids:                 # gripper pads only (hold.py pad_only)
        if g not in s.pad_gid:
            s.m.geom_contype[g] = 0
    m, d = s.m, s.d
    s.reset(q_pre)
    d.qpos[s.qadr[:12]] += rng.uniform(-0.03, 0.03, 12)
    centre = np.array([HOLD["xc"], 0.0, HOLD["hb"] + BOX_HALF[2]])
    if with_box:
        table_for(s, HOLD["xc"], HOLD["hb"])
        s.place_box(centre + [0, 0, 0.002])
    mujoco.mj_forward(m, d)
    pol = Policy()
    phase = rng.uniform(0, 2 * np.pi)
    dphase = 2 * np.pi * CTRL_DT * SC["gait_clock_hz"]
    last = np.zeros(12, np.float32)
    # arm schedule: (t0, t1, q_from, q_to)
    sched = [(0.0, 1.5, q_pre, q_pre), (1.5, 2.5, q_pre, q_sq), (2.5, 3.0, q_sq, q_sq),
             (3.0, 4.0, q_sq, q_lift)]
    t_table_off = 4.0
    t_test = 4.5
    dur = t_test + (WALK_S if walk else STAND_S)
    log = {"t": [], "pitch": [], "roll": [], "z": [], "box_rel": [], "vx": []}
    fell, t_fell = False, None
    n = int(dur / CTRL_DT)
    for k in range(n):
        t = k * CTRL_DT
        arm = q_lift
        for t0, t1, qa, qb in sched:
            if t0 <= t < t1:
                arm = qa + (t - t0) / (t1 - t0) * (qb - qa)
                break
        if with_box and abs(t - t_table_off) < CTRL_DT / 2:
            s.set_table(None)
        vx = 0.0
        if walk and t >= t_test:
            vx = 0.3 * min(1.0, (t - t_test) / 1.0)
        cmd = np.array([vx, 0.0, 0.0, HEIGHT_CMD])
        quat = d.qpos[3:7]
        g = quat_rotate_inv(quat, np.array([0.0, 0.0, -1.0]))
        gyro = d.qvel[3:6].copy()
        qj = s.q() - DEFAULT_POSE
        vj = s.qd()
        g = g + rng.normal(0, NOISE["gravity"], 3)
        gyro = gyro + rng.normal(0, NOISE["gyro"], 3)
        qj = qj + rng.normal(0, NOISE["joint_pos"], 24)
        vj = vj + rng.normal(0, NOISE["joint_vel"], 24)
        obs = np.concatenate([g, gyro, qj, vj, last, cmd, arm,
                              [np.sin(phase), np.cos(phase)]]).astype(np.float32)
        act = np.clip(pol(obs), -1, 1)
        last = act.astype(np.float32)
        phase = (phase + dphase) % (2 * np.pi)
        tgt = np.concatenate([DEFAULT_POSE[:12] + 0.5 * act, arm])
        for _ in range(DECIM):
            tau = KP * (tgt - s.q()) - KD * s.qd()
            d.ctrl[s.act] = np.clip(tau, -TAU_MAX, TAU_MAX)
            mujoco.mj_step(m, d)
        gt = quat_rotate_inv(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        pitch = float(np.degrees(np.arctan2(gt[0], -gt[2])))    # + = forward lean
        roll = float(np.degrees(np.arctan2(gt[1], -gt[2])))
        if t >= t_test:
            R = d.xmat[s.pelvis_bid].reshape(3, 3)
            log["t"].append(t)
            log["pitch"].append(pitch)
            log["roll"].append(roll)
            log["z"].append(float(d.qpos[2]))
            log["box_rel"].append((R.T @ (s.box_pos() - d.qpos[:3])).tolist())
            log["vx"].append(float((R.T @ d.qvel[:3])[0]))
        tilt = np.degrees(np.arccos(np.clip(-gt[2], -1, 1)))
        if d.qpos[2] < HEIGHT_MIN or tilt > np.degrees(MAX_TILT):
            fell, t_fell = True, t
            break
    P = np.array(log["pitch"]) if log["pitch"] else np.array([np.nan])
    Rl = np.array(log["roll"]) if log["roll"] else np.array([np.nan])
    B = np.array(log["box_rel"]) if log["box_rel"] else np.zeros((1, 3))
    box_ok = None
    if with_box:
        box_ok = bool(not fell and B[-1][2] > B[0][2] - 0.10)
    return {"with_box": with_box, "walk": walk, "seed": seed, "fell": fell,
            "t_fell": t_fell, "survived_s": (t_fell - t_test) if fell else dur - t_test,
            "pitch_mean": float(np.nanmean(P)), "pitch_min": float(np.nanmin(P)),
            "pitch_max": float(np.nanmax(P)), "roll_p2p": float(np.nanmax(Rl) - np.nanmin(Rl)),
            "vx_mean": float(np.mean(log["vx"][len(log["vx"]) // 3:])) if log["vx"] else None,
            "box_held": box_ok, "box_drop": float(B[0][2] - B[-1][2]),
            "box_rel_end": B[-1].tolist()}


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = range(10)
    jobs = [(wb, wk, sd) for wb in (False, True) for wk in (False, True) for sd in seeds]
    with Pool(20) as p:
        res = p.map(episode, jobs)
    with open(os.path.join(OUT, "balance.json"), "w") as f:
        json.dump(res, f, indent=1)
    for wb in (False, True):
        for wk in (False, True):
            rr = [r for r in res if r["with_box"] == wb and r["walk"] == wk]
            surv = sum(not r["fell"] for r in rr)
            held = sum(bool(r["box_held"]) for r in rr) if wb else None
            print(f"box={wb!s:5} {'walk 0.3' if wk else 'stand   '}: survived {surv}/{len(rr)} "
                  f"held {held} pitch mean {np.mean([r['pitch_mean'] for r in rr]):+.1f} "
                  f"[{min(r['pitch_min'] for r in rr):+.1f}, {max(r['pitch_max'] for r in rr):+.1f}] deg "
                  f"roll p2p {np.mean([r['roll_p2p'] for r in rr]):.1f} deg "
                  f"box drop {np.mean([r['box_drop'] for r in rr]) * 1000:.0f} mm "
                  f"vx {np.mean([r['vx_mean'] or 0 for r in rr]):.2f} "
                  f"surv_s {[round(r['survived_s'], 1) for r in rr]}")


if __name__ == "__main__":
    main()
