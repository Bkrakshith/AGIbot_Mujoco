#!/usr/bin/env python
"""Rollout a policy on the full-fidelity model and render an MP4 via mediapy.

  python train/play.py --checkpoint outputs/checkpoints/run/ckpt_N \
      [--scenario a_stand_7kg_single_arm] [--out outputs/videos/run.mp4]
  python train/play.py --onnx outputs/onnx/student.onnx --scenario c_walk_0.5_5kg_asym
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

from x1_locomotion.backend import select_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--onnx")
    ap.add_argument("--scenario", default="c_walk_forward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()
    if not args.checkpoint and not args.onnx:
        ap.error("one of --checkpoint / --onnx is required")

    select_backend("cpu")
    import mediapy
    try:  # no system ffmpeg on this box; imageio-ffmpeg ships a static binary
        import imageio_ffmpeg
        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    import mujoco
    from x1_locomotion.config import REPO_ROOT, load_config
    from x1_locomotion.cpu_eval import CpuRollout, make_policy
    from run_battery import battery

    cfg = load_config()
    source = battery(cfg)
    scenarios = {s.name: s for s in source}
    if args.scenario not in scenarios:
        ap.error(f"unknown scenario {args.scenario!r}; choose from {list(scenarios)}")
    sc = scenarios[args.scenario]

    runner = CpuRollout(cfg)
    policy = make_policy(cfg, checkpoint=args.checkpoint, onnx=args.onnx)
    if hasattr(policy, "reset"):
        policy.reset()
    result, qpos_trace = runner.run_episode(sc, policy, seed=args.seed,
                                            record_qpos=True)
    print(f"[{sc.name}] survived={result.survived} "
          f"t={result.survived_time_s:.1f}s ori_err={result.orientation_err:.3f}")

    import numpy as np
    from x1_locomotion.cpu_eval import _interp_cmd

    model = runner.model
    data = mujoco.MjData(model)
    pelvis = model.body("pelvis").id
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = pelvis
    cam.distance, cam.elevation, cam.azimuth = 2.2, -12.0, 135.0
    base_cmd = np.asarray(sc.cmd, dtype=float)

    frames = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for i, qpos in enumerate(qpos_trace):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            # red marker at the COMMANDED pelvis height, beside the robot
            h_cmd = (_interp_cmd(sorted(sc.cmd_schedule), i * runner.ctrl_dt,
                                 base_cmd)[3] if sc.cmd_schedule
                     else float(base_cmd[3]))
            scn = renderer.scene
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.05, 0.0, 0.0]),
                    pos=np.array([data.xpos[pelvis][0],
                                  data.xpos[pelvis][1] + 0.40, h_cmd]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1.0, 0.15, 0.15, 0.8]))
                scn.ngeom += 1
            frames.append(renderer.render())

    name = os.path.basename((args.onnx or args.checkpoint).rstrip("/"))
    out = args.out or os.path.join(REPO_ROOT, "outputs", "videos",
                                   f"{name}_{sc.name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mediapy.write_video(out, frames, fps=args.fps)
    print(f"video -> {out}")


if __name__ == "__main__":
    main()
