#!/usr/bin/env python
"""Pick, carry and place the handle box with the existing walking policy (no
retraining). Floating base, ONNX student, full obs noise, arms by PD only.

Per episode (one place pose per episode, 20 seeds each):
  stand 2 s, arms "down", fingers open -> table + box appear in front
  (handles at the measured carry-pose TCPs) -> arms down -> pre-grasp (carry
  with shoulder yaw +0.25, hands outside the handle ends, 1.5 s) -> carry
  (1 s, the open jaws slide onto the handles) -> close fingers (0.7 s, 30 N) -> lift: the table drops away ->
  hold 2 s -> walk 0.3 m/s (--vx) 10 s -> turn 0.4 rad/s 5 s -> stop 2 s ->
  arms to a pre-place pose above the place table (1.5 s) -> the place table
  appears under the box once the robot has stopped moving -> arms to the
  place pose (1.5 s) -> open fingers ->
  step back (-0.2 m/s, 2 s) -> arms down (1.5 s) -> stand 2 s.
All arm poses are the trained "carry" (and "reach" as a pre-pose) or carry
plus offsets inside arm_pose_noise.

  python manip/handle_task.py [--calibrate | --calibrate-pick] [--vx 0.2]
                                  # -> manip/out/handle_task_vx<vx>.json
"""

import json
import os
import sys
from multiprocessing import Pool

import mujoco
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_handle_scene import SPEC, STROKE  # noqa: E402
from common import Scene  # noqa: E402
from runner import CTRL_DT, PolicyRunner  # noqa: E402

OUT = os.path.join(HERE, "out")
SCENE = os.path.join(HERE, "models", "x1_handle_scene.xml")
NOISE_AMP = np.array([0.3, 0.25, 0.4, 0.3, 0.4, 0.3])      # configs/domain_rand.yaml
LIB = {"down": [0, 0, 0, 0, 0, 0], "carry": [-0.5, -0.1, 0, 1.4, 0, 0],
       "reach": [-1.2, -0.1, 0, 1.0, 0, 0]}
# level-box place poses: carry + (shoulder pitch, wrist pitch) offsets
PLACE = {
    "low":     [-0.20, -0.1, 0, 1.4, 0, -0.30],
    "midlow":  [-0.35, -0.1, 0, 1.4, 0, -0.15],
    "carry":   [-0.50, -0.1, 0, 1.4, 0, 0.00],
    "midhigh": [-0.65, -0.1, 0, 1.4, 0, 0.15],
    "high":    [-0.80, -0.1, 0, 1.4, 0, 0.30],
}
ORDER = list(PLACE)
# pre-grasp: carry with shoulder yaw +0.25 (inside the 0.4 noise): each TCP
# sits ~9 cm further out, beyond the outer end of its handle; moving back to
# "carry" slides the open jaws onto the handle along its axis
PREGRASP = [-0.5, -0.1, 0.25, 1.4, 0, 0]
FINGER_SPEED = STROKE / 0.7
WALK_VX = 0.3                      # --vx to change
# the box can be placed this far behind the mean TCP (deeper in the jaws);
# unused now that the approach is corrected
GRASP_DEPTH_BIAS = 0.0
# approach correction from the handle position seen at the pre-grasp
# (stands in for the wrist D405); False = open loop
CORRECT_APPROACH = True
PRE_TCP_X_OFFSET = 0.009         # carry TCP is 9 mm further along the approach than pre-grasp        # m/s per finger (0.12 m opening in 0.7 s)


def in_envelope(pose):
    """Smallest per-joint excess over a library pose + arm_pose_noise."""
    best = None
    for name, lib in LIB.items():
        ex = np.maximum(np.abs(np.array(pose) - lib) - NOISE_AMP, 0.0)
        if best is None or ex.max() < best[1]:
            best = (name, float(ex.max()))
    return best


class Task:
    def __init__(self, seed, spec):
        self.spec = spec
        self.s = Scene(False, SCENE)
        m = self.s.m
        self.fa = [m.actuator(f"{sd}_finger_{k}_servo").id for sd in ("left", "right")
                   for k in "ab"]
        self.finger_g = {sd: [m.geom(f"{sd}_finger_{k}_geom").id for k in "ab"]
                         for sd in ("left", "right")}
        self.handle_g = {sd: m.geom(f"{sd}_handle").id for sd in ("left", "right")}
        self.box_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.s.box_bid]
        self.ptable_gid = m.geom("place_table").id
        self.ptable_mocap = m.body_mocapid[m.body("place_table").id]
        self.grip_target = STROKE
        self.grip_cmd = STROKE
        self.s.reset(np.zeros(12))
        self.r = PolicyRunner(self.s, seed)
        self.r.perturb_legs()
        self.r.pre_substep = self._fingers
        self.arm = np.zeros(12)
        self.log = {"pitch": [], "stage": []}
        self.stage = "stand"

    def _fingers(self, s):
        step = FINGER_SPEED * s.m.opt.timestep
        self.grip_cmd += np.clip(self.grip_target - self.grip_cmd, -step, step)
        s.d.ctrl[self.fa] = self.grip_cmd

    def pelvis_frame(self):
        d = self.s.d
        p = d.xpos[self.s.pelvis_bid].copy()
        R = d.xmat[self.s.pelvis_bid].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s_ = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        return p, yaw, Ry

    def set_table(self, xy, yaw, top):
        d, s = self.s.d, self.s
        hz = s.m.geom_size[s.table_gid, 2]
        d.mocap_pos[s.table_mocap] = [xy[0], xy[1], top - hz]
        d.mocap_quat[s.table_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def set_place_table(self, xy, yaw, top):
        d = self.s.d
        hz = self.s.m.geom_size[self.ptable_gid, 2]
        d.mocap_pos[self.ptable_mocap] = [xy[0], xy[1], top - hz]
        d.mocap_quat[self.ptable_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def settle(self, max_s=4.0, v_tol=0.05, w_tol=0.1, hold_s=0.5):
        """Stand (arms fixed) until the pelvis is still for hold_s; returns
        the time waited, or None if it never settled / fell."""
        still, n = 0, 0
        while n * CTRL_DT < max_s:
            if not self.r.step(self.arm):
                return None
            self.log["pitch"].append(self.r.pitch_roll()[0])
            self.log["stage"].append(self.stage)
            n += 1
            d = self.s.d
            if np.linalg.norm(d.qvel[:2]) < v_tol and abs(d.qvel[5]) < w_tol:
                still += 1
                if still * CTRL_DT >= hold_s:
                    return n * CTRL_DT
            else:
                still = 0
        return None

    def hide_table(self):
        self.s.d.mocap_pos[self.s.table_mocap] = [50.0, 50.0, -2.0]

    def run(self, dur, arm_to=None, cmd=(0.0, 0.0, 0.0), ramp=1.0):
        """Advance dur s; arm moves linearly to arm_to over the whole dur."""
        a0 = self.arm.copy()
        n = int(round(dur / CTRL_DT))
        for k in range(n):
            if arm_to is not None:
                self.arm = a0 + (k + 1) / n * (np.asarray(arm_to) - a0)
            u = min(1.0, (k + 1) * CTRL_DT / ramp) if ramp > 0 else 1.0
            ok = self.r.step(self.arm, [c * u for c in cmd])
            self.log["pitch"].append(self.r.pitch_roll()[0])
            self.log["stage"].append(self.stage)
            if not ok:
                return False
        return True

    def contacts(self):
        """Finger-handle normal force per side, box-table force, and other
        robot-box contact force."""
        m, d = self.s.m, self.s.d
        f6 = np.zeros(6)
        out = {"left": 0.0, "right": 0.0, "table": 0.0, "robot_other": 0.0, "floor": 0.0}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = g1 in self.box_geoms, g2 in self.box_geoms
            if b1 == b2:
                continue
            other = g2 if b1 else g1
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(f6[0])
            if other in (self.s.table_gid, self.ptable_gid):
                out["table"] += fn
            elif m.geom(other).name == "floor":
                out["floor"] += fn
            elif other in self.finger_g["left"]:
                out["left"] += fn
            elif other in self.finger_g["right"]:
                out["right"] += fn
            else:
                out["robot_other"] += fn
        return out

    def box_in_hands(self):
        """Handle centres vs TCPs (m)."""
        m, d = self.s.m, self.s.d
        e = []
        for sd in ("left", "right"):
            e.append(np.linalg.norm(d.site_xpos[m.site(f"{sd}_handle_site").id]
                                    - d.site_xpos[m.site(f"{sd}_tcp").id]))
        return max(e)


def approach_correction(t, carry):
    """Per arm: along-approach error of the handle vs the TCP at the
    pre-grasp (whose TCP x matches carry to ~1 cm), turned into equal
    shoulder-pitch and elbow-pitch offsets (measured sensitivity: +0.1 rad on
    both moves the TCP 0.0147 m back), clipped to stay inside
    carry +- arm_pose_noise."""
    m, d = t.s.m, t.s.d
    out = carry.copy()
    corr = []
    for k, sd in enumerate(("left", "right")):
        tcp = m.site(f"{sd}_tcp").id
        R = d.site_xmat[tcp].reshape(3, 3)
        e = float((-R[:, 1]) @ (d.site_xpos[m.site(f"{sd}_handle_site").id] - d.site_xpos[tcp]))
        delta = float(np.clip(-0.1 * (e + PRE_TCP_X_OFFSET) / 0.0147, -0.25, 0.25))
        out[6 * k + 0] += delta
        out[6 * k + 3] += delta
        corr.append(round(delta, 3))
    return out, corr


def episode(args):
    place, seed = args[:2]
    vx = args[2] if len(args) > 2 else WALK_VX
    with open(SPEC) as f:
        spec = yaml.safe_load(f)
    t = Task(seed, spec)
    s = t.s
    res = {"place": place, "seed": seed}
    stages = {}
    carry = np.array(LIB["carry"] * 2, float)

    def done(name, ok, **kw):
        stages[name] = bool(ok)
        res.update(kw)
        return ok

    # 1. stand, then the table and box appear in front of the robot
    t.stage = "stand"
    if not t.run(2.0):
        return dict(res, stages=stages, fell_at="stand")
    p, yaw, Ry = t.pelvis_frame()
    centre = p + Ry @ np.array(spec["pick"]["box_centre_in_spawn_frame"])
    bottom = centre[2] - spec["box"]["size"][2] / 2
    t.set_table(centre[:2], yaw, bottom)
    s.place_box(centre + [0, 0, 0.001], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    mujoco.mj_forward(s.m, s.d)
    res["pick_table_top"] = float(bottom)
    # 2. arms down -> carry, grippers open
    t.stage = "reach"
    pre = np.array(PREGRASP * 2, float)
    ok = t.run(1.5, pre) and t.run(0.5)
    if CORRECT_APPROACH:
        # the wrist camera would see the handle here: shift each arm's grasp
        # pose along the approach (shoulder pitch + elbow, inside the noise
        # envelope) so the handle ends up at the TCP
        carry, corr = approach_correction(t, carry)
        res["approach_correction_rad"] = corr
    ok = ok and t.run(1.0, carry) and t.run(0.5)
    c = t.contacts()
    disturbed = float(np.linalg.norm(s.box_pos() - centre))
    if not done("approach", ok and disturbed < 0.02, approach_box_shift=disturbed,
                approach_robot_contact=c["robot_other"] + c["left"] + c["right"]):
        return dict(res, stages=stages, fell=t.r.fell)
    # 3. close
    t.stage = "grasp"
    t.grip_target = 0.0
    ok = t.run(1.2)
    c = t.contacts()
    if not done("grasp", ok and c["left"] > 5 and c["right"] > 5,
                grip_force=[c["left"], c["right"]]):
        return dict(res, stages=stages, fell=t.r.fell)
    # 4. lift: the table drops away
    t.stage = "lift"
    z0 = s.box_pos()[2]
    for k in range(10):
        t.set_table(centre[:2], yaw, bottom - 0.01 * (k + 1))
        if not t.run(0.05):
            break
    t.hide_table()
    ok = t.run(2.0)
    c = t.contacts()
    if not done("lift", ok and c["left"] > 2 and c["right"] > 2 and t.box_in_hands() < 0.04,
                lift_sag=float(z0 - s.box_pos()[2])):
        return dict(res, stages=stages, fell=t.r.fell)
    # 5. walk 0.3 m/s 10 s, turn 0.4 rad/s 5 s, stop
    t.stage = "walk"
    x0 = s.d.qpos[:2].copy()
    ok = t.run(10.0, cmd=(vx, 0.0, 0.0))
    walked = float(np.linalg.norm(s.d.qpos[:2] - x0))
    held = t.box_in_hands() < 0.04 and t.contacts()["floor"] == 0
    if not done("walk", ok and held, walked_m=walked):
        return dict(res, stages=stages, fell=t.r.fell, box_lost=not held)
    t.stage = "turn"
    ok = t.run(5.0, cmd=(0.0, 0.0, 0.4)) and t.run(2.0) and t.settle() is not None
    held = t.box_in_hands() < 0.04
    if not done("turn", ok and held):
        return dict(res, stages=stages, fell=t.r.fell, box_lost=not held)
    # 6. place
    t.stage = "place"
    target = np.array(PLACE[place] * 2, float)
    i = ORDER.index(place)
    pre = np.array((PLACE[ORDER[i + 1]] if i + 1 < len(ORDER) else LIB["reach"]) * 2, float)
    ok = t.run(1.5, pre)
    waited = t.settle()
    res["place_settle_s"] = waited
    ok = ok and waited is not None
    p, yaw, Ry = t.pelvis_frame()
    top = p[2] + spec["place_tables"][place]["top_above_pelvis"]
    bp = s.box_pos()
    box_bottom = bp[2] - spec["box"]["size"][2] / 2
    if box_bottom < top + 0.005:
        done("place", False, place_note="box not above the place table at the pre-pose",
             box_bottom_pre=float(box_bottom), table_top=float(top))
        return dict(res, stages=stages)
    # full-size place table, centred 0.1 m beyond the box (the robot side
    # edge stays clear of the knees)
    t.set_place_table(bp[:2] + Ry[:2, 0] * 0.10, yaw, top)
    trace = {}

    def mark(tag):
        b = s.box_pos()
        trace[tag] = {"box_off_table_xy": np.round(Ry.T @ (b - [bp[0], bp[1], b[2]]), 3).tolist(),
                      "box_bottom_above_top": round(float(b[2] - spec["box"]["size"][2] / 2
                                                          - top), 3),
                      **{k: round(v, 2) for k, v in t.contacts().items()}}
    ok = ok and t.run(1.5, target) and t.run(0.5)
    mark("lowered")
    t.grip_target = STROKE
    ok = ok and t.run(1.0)
    mark("opened")
    t.stage = "step_back"
    ok = ok and t.run(2.0, cmd=(-0.2, 0.0, 0.0), ramp=0.5)
    mark("stepped_back")
    ok = ok and t.run(1.5, np.zeros(12)) and t.run(2.0)
    mark("arms_down")
    res["place_trace"] = trace
    c = t.contacts()
    R = s.d.xmat[s.box_bid].reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    on_table = abs(c["table"] - 0.1 * 9.81) < 0.5 and c["left"] + c["right"] + c["robot_other"] < 0.05
    done("place", ok and on_table and tilt < 10.0, place_tilt=tilt,
         place_table_force=c["table"], place_table_top=float(top))
    return dict(res, stages=stages, fell=t.r.fell,
                pitch_mean=float(np.mean(t.log["pitch"])),
                pitch_absmax=float(np.max(np.abs(t.log["pitch"]))),
                pitch_walk=float(np.mean([pp for pp, st in zip(t.log["pitch"], t.log["stage"])
                                          if st == "walk"])))


def calibrate_pick(spec, seeds=5):
    """The pelvis shifts when the arms rise to "carry", so the box is placed
    where the TCPs END UP, expressed in the pelvis frame at the moment the box
    appears (after 2 s standing with the arms down). No box, 5 seeds."""
    rows = []
    for seed in range(seeds):
        t = Task(200 + seed, spec)
        s = t.s
        t.run(2.0)
        p, yaw, Ry = t.pelvis_frame()
        t.run(1.5, np.array(PREGRASP * 2, float))
        t.run(0.5)
        t.run(1.0, np.array(LIB["carry"] * 2, float))
        t.run(0.5)
        mid = np.mean([s.d.site_xpos[s.m.site(f"{sd}_tcp").id] for sd in ("left", "right")], 0)
        rows.append(Ry.T @ (mid - p))
    tcp_mid = np.mean(rows, 0)
    h = spec["handles"]["left"]["centre"]
    centre = tcp_mid - np.array([h[0], 0.0, h[2]]) - [GRASP_DEPTH_BIAS, 0.0, 0.0]
    out = {"box_centre_in_spawn_frame": np.round(centre, 4).tolist(),
           "grasp_depth_bias": GRASP_DEPTH_BIAS,
           "tcp_mid_in_spawn_frame": np.round(tcp_mid, 4).tolist(),
           "tcp_mid_std_mm": np.round(np.std(rows, 0) * 1000, 1).tolist(),
           "note": "spawn frame = pelvis (yaw only) after 2 s standing with the arms down; "
                   "the pelvis moves while the arms rise, so this differs from "
                   "hold.box_centre_in_pelvis_frame"}
    print("pick", out)
    return out


def calibrate_place_tables(spec):
    """Box-bottom height (above the pelvis) in each place pose, from the
    measured TCPs: policy standing, box held, 3 seeds."""
    out = {}
    for name, pose in PLACE.items():
        vals = []
        for seed in range(3):
            t = Task(100 + seed, spec)
            s = t.s
            carry = np.array(pose * 2, float)
            t.run(1.0)
            p, yaw, Ry = t.pelvis_frame()
            centre = p + Ry @ np.array(spec["pick"]["box_centre_in_spawn_frame"])
            t.set_table(centre[:2], yaw, centre[2] - spec["box"]["size"][2] / 2)
            s.place_box(centre + [0, 0, 0.001], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            t.run(1.5, np.array(PREGRASP * 2, float))
            t.run(0.5)
            t.run(1.0, np.array(LIB["carry"] * 2, float))
            t.run(0.5)
            t.grip_target = 0.0
            t.run(1.2)
            t.hide_table()
            t.run(1.5, carry)
            t.run(2.0)
            p, _, _ = t.pelvis_frame()
            vals.append(s.box_pos()[2] - spec["box"]["size"][2] / 2 - p[2])
        top = float(np.mean(vals)) - 0.005      # box lands 5 mm before the pose is reached
        env = in_envelope(pose)
        out[name] = {"arm_pose_rad_per_arm": pose, "top_above_pelvis": round(top, 4),
                     "top_world_at_standing": round(top + 0.61, 3),
                     "envelope": {"nearest_library_pose": env[0],
                                  "max_excess_over_noise_rad": round(env[1], 3)}}
        print(name, out[name])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SPEC) as f:
        spec = yaml.safe_load(f)
    full = "place_tables" not in spec or "--calibrate" in sys.argv
    if full or "--calibrate-pick" in sys.argv:
        spec["pick"] = calibrate_pick(spec)
        if full:
            spec["place_tables"] = calibrate_place_tables(spec)
        with open(SPEC) as f:
            head = "".join(line for line in f if line.startswith("#"))
        with open(SPEC, "w") as f:
            f.write(head)
            yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None)
    vx = float(sys.argv[sys.argv.index("--vx") + 1]) if "--vx" in sys.argv else WALK_VX
    jobs = [(pl, 1000 * i + sd, vx) for i, pl in enumerate(PLACE) for sd in range(20)]
    with Pool(24) as p:
        res = p.map(episode, jobs, chunksize=2)
    print(f"walk command {vx} m/s")
    with open(os.path.join(OUT, f"handle_task_vx{vx:.1f}.json"), "w") as f:
        json.dump(res, f, indent=1)
    names = ["approach", "grasp", "lift", "walk", "turn", "place"]
    print("place    " + " ".join(f"{n:>8}" for n in names) + "   fell  lost  pitch(walk)  |pitch|max")
    for pl in PLACE:
        rr = [x for x in res if x["place"] == pl]
        cells = [sum(x["stages"].get(n, False) for x in rr) for n in names]
        pw = [x["pitch_walk"] for x in rr if "pitch_walk" in x]
        pm = [x["pitch_absmax"] for x in rr if "pitch_absmax" in x]
        print(f"{pl:8} " + " ".join(f"{c:>6}/20" for c in cells)
              + f"   {sum(bool(x.get('fell')) for x in rr):3d}  lost {sum(bool(x.get('box_lost')) for x in rr)}  "
              + (f"{np.mean(pw):+5.1f}       {np.mean(pm):5.1f}" if pw else ""))
    allr = res
    print(f"all {len(allr)} episodes: fell {sum(bool(x.get('fell')) for x in allr)}, "
          f"box lost without a fall {sum(bool(x.get('box_lost')) and not x.get('fell') for x in allr)}, "
          f"full success {sum(x['stages'].get('place', False) for x in allr)}")
    for x in allr:
        if x.get("fell") or not x["stages"].get("place", False):
            last = [k for k, v in x["stages"].items() if not v] or ["fell"]
            print(f"  fail: place={x['place']} seed={x['seed']} stage={last[0]} "
                  f"fell={x.get('fell')} lost={x.get('box_lost')} "
                  f"tilt={x.get('place_tilt')} table_f={x.get('place_table_force')} "
                  f"note={x.get('place_note', '')}")


if __name__ == "__main__":
    main()
