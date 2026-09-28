#!/usr/bin/env python
"""Video + stills of the best squeeze: floating base with the ONNX policy
standing, pick at box bottom 0.75 m, carry 4 s, place on a 0.85 m table.

  MUJOCO_GL=egl python manip/render.py   # -> outputs/videos/manip/
"""

import os
import sys

import mediapy as media
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import BOX_HALF, VIDEO_DIR, renderer, snap  # noqa: E402
from hold import LIFT_DZ, Plan, pregrasp, table_for  # noqa: E402
from place import DEPTH, PICK_HB, STEP_DZ, XC, Runner, waypoints  # noqa: E402

FPS = 30
PLACE_HB = 0.85


def main():
    import ik as ikmod
    ikmod.BOX_CLEAR = -1.0
    os.makedirs(VIDEO_DIR, exist_ok=True)
    plan = Plan("pad")
    zc0 = PICK_HB + BOX_HALF[2]
    q_sq, _, _ = plan.pose(np.array([XC, 0.0, zc0]), DEPTH)
    plan.seed = q_sq
    up, _ = waypoints(plan, zc0, zc0 + 0.35)
    r = Runner(floating=True, seed=1)
    s = r.s
    q_pre = pregrasp(q_sq)
    s.reset(q_pre)
    table_for(s, XC, PICK_HB)
    s.place_box([XC, 0.0, zc0 + 0.002])
    mujoco.mj_forward(s.m, s.d)
    ren, opt, cam = renderer(s, 960, 540)
    cam.lookat[:] = [0.2, 0.0, 0.8]
    cam.distance, cam.elevation, cam.azimuth = 2.4, -12.0, 140.0
    frames, stills = [], {}
    every = int(round(1.0 / (FPS * 0.01)))
    k = [0]
    orig_step = r.step

    def step(arm):
        orig_step(arm)
        k[0] += 1
        if k[0] % every == 0:
            frames.append(snap(s, ren, opt, cam))
    r.step = step

    r.move(q_pre, q_pre, 1.0)
    r.move(q_pre, q_sq, 1.0)
    r.move(q_sq, q_sq, 0.5)
    stills["squeeze"] = snap(s, ren, opt, cam)
    q, z_cmd = q_sq, zc0
    for i, w in enumerate(up):
        if s.box_pos()[2] - BOX_HALF[2] > PLACE_HB + 0.03:
            break
        r.move(q, w, 0.5)
        q, z_cmd = w, z_cmd + STEP_DZ
        if i == 1:
            s.set_table(None)            # pick table away once lifted
    r.move(q, q, 3.0)
    stills["carry"] = snap(s, ren, opt, cam)
    table_for(s, XC, PLACE_HB)
    plan.seed = q
    down, _ = waypoints(plan, z_cmd, PLACE_HB + BOX_HALF[2] - 0.02)
    for w in down:
        r.move(q, w, 0.5)
        q = w
    q_rel, ok, _ = plan.pose(np.array([XC, 0.0, s.box_pos()[2]]), -0.05)
    r.move(q, q_rel, 1.0)
    r.move(q_rel, q_rel, 1.5)
    stills["placed"] = snap(s, ren, opt, cam)
    try:
        import imageio_ffmpeg
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    media.write_video(os.path.join(VIDEO_DIR, "squeeze_pick_carry_place.mp4"), frames, fps=FPS)
    media.write_image(os.path.join(VIDEO_DIR, "squeeze_stills.png"),
                      np.concatenate([stills["squeeze"], stills["carry"], stills["placed"]], 1))
    print(f"{len(frames)} frames; fell={r.fell}; box bottom {s.box_pos()[2] - BOX_HALF[2]:.3f}")


if __name__ == "__main__":
    main()
