#!/usr/bin/env python
"""Design the handle box around the TRAINED "carry" arm pose.

1. Measure where the gripper TCPs and jaw axes really are in the carry pose:
   floating base, the ONNX policy standing, fingers open, full obs noise,
   5 seeds x 5 s. The arms sag under the gripper mass (PD only), so the
   measured, not the kinematic, TCP is used.
2. Put a rubber handle at each TCP with its axis along the payload z axis
   (perpendicular to both the jaw closing axis and the approach), size the
   box to sit between the handles in front of the chest, and write
   manip/handle_box.yaml. Then run build_handle_scene.py.

  python manip/handle_design.py
"""

import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_handle_scene import SPEC  # noqa: E402
from common import ARM, Scene  # noqa: E402
from runner import PolicyRunner  # noqa: E402

SCENE = os.path.join(HERE, "models", "x1_handle_scene.xml")
CARRY = [-0.5, -0.1, 0.0, 1.4, 0.0, 0.0]         # configs/domain_rand.yaml arm_poses
BOX_SIZE = (0.20, 0.26, 0.14)                    # depth x, width y, height z
TOP_ABOVE_HANDLE = 0.025                         # box top above the handle axis
HANDLE = dict(diameter=0.031, length=0.11, mass=0.01, friction=1.0,
              standoff_radius=0.010, standoff_mass=0.005)
BOX_MASS = 0.10
CAM_PELVIS = (0.085, 0.018, 0.4185)              # chest D435, looking along +x


def finger_ids(s):
    return [s.m.actuator(f"{sd}_finger_{k}_servo").id for sd in ("left", "right") for k in "ab"]


def measure(seeds=5):
    arm = np.array(CARRY * 2)
    rows, qerr = [], []
    for seed in range(seeds):
        s = Scene(False, SCENE)
        s.reset(arm)
        fa = finger_ids(s)
        r = PolicyRunner(s, seed)
        r.perturb_legs()
        for k in range(800):
            s.d.ctrl[fa] = 0.06
            r.step(arm)
            if k >= 300:
                P = s.d.xmat[s.pelvis_bid].reshape(3, 3)
                p0 = s.d.xpos[s.pelvis_bid]
                row = []
                for sd in ("left", "right"):
                    site = s.m.site(f"{sd}_tcp").id
                    R = s.d.site_xmat[site].reshape(3, 3)
                    row.append(np.concatenate([P.T @ (s.d.site_xpos[site] - p0),
                                               P.T @ R[:, 0], P.T @ R[:, 2]]))
                rows.append(row)
                qerr.append(s.q()[ARM] - arm)
        assert not r.fell
    A = np.array(rows)
    return A.mean(0), A.std(0), np.array(qerr).mean(0), float(np.mean(
        [s.d.qpos[2] for _ in [0]]))


def design(mean):
    L, R = mean
    # symmetric design: average the two sides (mirror in y)
    tcp = np.array([(L[0] + R[0]) / 2, (L[1] - R[1]) / 2, (L[2] + R[2]) / 2])
    out_l = -L[6:9]                      # payload z points medially for the left arm
    out_r = -R[6:9]
    ax = np.array([(out_l[0] + out_r[0]) / 2, (out_l[1] - out_r[1]) / 2,
                   (out_l[2] + out_r[2]) / 2])
    ax /= np.linalg.norm(ax)
    D, W, H = BOX_SIZE
    # box frame: centre of the corrugated body, axes = pelvis axes at the hold
    centre = np.array([tcp[0], 0.0, tcp[2] + TOP_ABOVE_HANDLE - H / 2])
    spec = {"box": {}, "handles": dict(HANDLE)}
    handles = {}
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        c = np.array([tcp[0], sgn * tcp[1], tcp[2]]) - centre
        a = np.array([ax[0], sgn * ax[1], ax[2]])
        t = (abs(c[1]) - W / 2) / abs(a[1])       # back along the axis to the box side
        root = c - a * t
        inner_gap = t - HANDLE["length"] / 2
        handles[side] = {"centre": np.round(c, 4).tolist(), "axis_out": np.round(a, 4).tolist(),
                         "standoff_root": np.round(root, 4).tolist(),
                         "standoff_length": round(float(inner_gap), 4)}
    body_mass = BOX_MASS - 2 * (HANDLE["mass"] + HANDLE["standoff_mass"])
    spec["box"] = {"size": list(BOX_SIZE), "mass_total": BOX_MASS,
                   "body_mass": round(body_mass, 4), "material": "corrugated cardboard"}
    spec["handles"].update(handles)
    spec["hold"] = {
        "arm_pose": "carry (configs/domain_rand.yaml arm_poses), both arms, no deviation",
        "arm_pose_rad_per_arm": CARRY,
        "box_centre_in_pelvis_frame": np.round(centre, 4).tolist(),
        "tcp_in_pelvis_frame_left": np.round(L[:3], 4).tolist(),
        "tcp_in_pelvis_frame_right": np.round(R[:3], 4).tolist(),
        "note": "TCPs measured with the policy standing (PD arms sag ~0.12 m under the "
                "0.43 kg grippers); the kinematic carry TCP is ~0.12 m higher.",
    }
    return spec, centre


def camera_blockage(spec, centre, n=(87, 58)):
    """Fraction of the D435 depth field of view (87 x 58 deg) whose ray hits
    the box body or handles (analytic ray-box test on the body only)."""
    D, W, H = spec["box"]["size"]
    lo = centre - np.array([D, W, H]) / 2
    hi = centre + np.array([D, W, H]) / 2
    cam = np.array(CAM_PELVIS)
    hits = 0
    az = np.radians(np.linspace(-43.5, 43.5, n[0]))
    el = np.radians(np.linspace(-29.0, 29.0, n[1]))
    for a in az:
        for e in el:
            d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - cam) / d, (hi - cam) / d
            tmin = np.nanmax(np.minimum(t1, t2))
            tmax = np.nanmin(np.maximum(t1, t2))
            hits += bool(tmax >= max(tmin, 0.0))
    return hits / (n[0] * n[1])


def main():
    mean, std, qerr, _ = measure()
    spec, centre = design(mean)
    spec["hold"]["tcp_std_mm"] = np.round(std[:, :3] * 1000, 1).tolist()
    spec["hold"]["arm_tracking_error_rad"] = np.round(qerr, 3).tolist()
    spec["camera"] = {"d435_in_pelvis": list(CAM_PELVIS),
                      "fov_blocked_fraction": round(camera_blockage(spec, centre), 3)}
    with open(SPEC, "w") as f:
        f.write("# Handle box for the X1 pinch-carry task. Generated by manip/handle_design.py;\n"
                "# table heights and test results are added by manip/handle_task.py.\n"
                "# Units m, kg, rad. Box frame: centre of the corrugated body, x forward,\n"
                "# y left, z up (the pelvis axes at the hold). Handle entries are in the box frame.\n")
        yaml.safe_dump(spec, f, sort_keys=False, default_flow_style=None)
    print(yaml.safe_dump(spec, sort_keys=False, default_flow_style=None))


if __name__ == "__main__":
    main()
