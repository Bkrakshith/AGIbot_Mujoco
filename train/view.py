#!/usr/bin/env python
"""Interactive viewer: run a teacher/student checkpoint (or ONNX) live on the
FULL-FIDELITY X1 model and steer it from the keyboard.

  python train/view.py --checkpoint outputs/checkpoints/teacher_v6/ckpt_422259200
  python train/view.py --onnx outputs/onnx/student.onnx

Keys (in the MuJoCo window; commands are clipped to the training envelope):
  Up / Down          forward speed  vx  +/- 0.1 m/s
  PageUp / PageDown  lateral speed  vy  +/- 0.1 m/s
  Left / Right       yaw rate       wz  +/- 0.1 rad/s
  Home               stop (zero command -> stand still)
  End                push the torso: 40 N for 0.2 s, random horizontal direction
  Insert             reset the robot to the home pose
  1-7                arm pose (both arms, moved over 1 s): 1 down, 2 forward,
                     3 forward-reach, 4 sideways, 5 overhead, 6 back, 7 carry
  8                  random arms (each arm its own pose + noise, as in training)

Same control loop as the evaluation battery (cpu_eval.CpuRollout): 100 Hz
policy, 500 Hz PD, identical observation layout and free-running gait clock.
Terminations are reported in the terminal and the robot is reset.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from x1_locomotion import rewards  # noqa: E402
from x1_locomotion.config import load_config  # noqa: E402
from x1_locomotion.cpu_eval import CpuRollout, make_policy, quat_rotate_inv_np  # noqa: E402
from x1_locomotion.layout import ARM_QPOS, N_ACT, N_LT  # noqa: E402

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PGUP, KEY_PGDN, KEY_HOME, KEY_END, KEY_INSERT = 266, 267, 268, 269, 260


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--onnx")
    ap.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"), help="initial command")
    ap.add_argument("--push_n", type=float, default=40.0)
    ap.add_argument("--obs_noise", action="store_true",
                    help="full X1 sensor noise on observations (deployment condition)")
    args = ap.parse_args()
    if not args.checkpoint and not args.onnx:
        ap.error("one of --checkpoint / --onnx is required")

    cfg = load_config()
    r = CpuRollout(cfg, obs_noise=args.obs_noise)
    m = r.model
    d = mujoco.MjData(m)
    policy = make_policy(cfg, checkpoint=args.checkpoint, onnx=args.onnx)
    c = cfg.train.commands
    lo = np.array([c.vx_range[0], c.vy_range[0], c.yaw_range[0]])
    hi = np.array([c.vx_range[1], c.vy_range[1], c.yaw_range[1]])
    height = float(c.height_range[0])
    phase_inc = 2.0 * np.pi * r.ctrl_dt / cfg.rewards.cpg.cycle_time_s
    push_steps = r.push_dur_steps
    rng = np.random.default_rng(0)

    lib = np.asarray(cfg.domain_rand.arm_poses, float)
    noise = np.asarray(cfg.domain_rand.arm_pose_noise, float)
    arm_lo, arm_hi = r.joint_lo[N_LT:], r.joint_hi[N_LT:]
    move_steps = int(1.0 / r.ctrl_dt)
    st = {"cmd": np.clip(np.array(args.cmd, float), lo, hi), "push": np.zeros(3),
          "push_left": 0, "reset": True, "arm": r.nominal_carry.copy(),
          "arm_from": r.nominal_carry.copy(), "arm_to": r.nominal_carry.copy(),
          "arm_k": move_steps}

    def key_callback(key):
        step = {KEY_UP: (0, 0.1), KEY_DOWN: (0, -0.1), KEY_PGUP: (1, 0.1),
                KEY_PGDN: (1, -0.1), KEY_LEFT: (2, 0.1), KEY_RIGHT: (2, -0.1)}
        if key in step:
            i, dv = step[key]
            st["cmd"][i] = np.clip(round(st["cmd"][i] + dv, 2), lo[i], hi[i])
        elif key == KEY_HOME:
            st["cmd"][:] = 0.0
        elif key == KEY_END:
            a = rng.uniform(0, 2 * np.pi)
            st["push"] = args.push_n * np.array([np.cos(a), np.sin(a), 0.0])
            st["push_left"] = push_steps
            print(f"push {args.push_n:.0f} N at {np.degrees(a):.0f} deg", flush=True)
            return
        elif key == KEY_INSERT:
            st["reset"] = True
        elif ord("1") <= key <= ord("8"):
            if key == ord("8"):
                pose = np.concatenate([lib[rng.integers(len(lib))] + noise * rng.uniform(-1, 1, 6)
                                       for _ in range(2)])
            else:
                pose = np.tile(lib[key - ord("1")], 2)
            st["arm_from"], st["arm_to"], st["arm_k"] = st["arm"].copy(), np.clip(pose, arm_lo, arm_hi), 0
            print(f"arm pose -> {np.round(st['arm_to'], 2).tolist()}", flush=True)
            return
        else:
            return
        print(f"cmd vx {st['cmd'][0]:+.1f} m/s  vy {st['cmd'][1]:+.1f} m/s  "
              f"wz {st['cmd'][2]:+.1f} rad/s", flush=True)

    def reset():
        mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
        mujoco.mj_forward(m, d)
        if hasattr(policy, "reset"):
            policy.reset()
        d.qpos[ARM_QPOS] = st["arm"]
        mujoco.mj_forward(m, d)
        st.update(last_action=np.zeros(N_ACT, np.float32),
                  phase=float(rng.uniform(0, 2 * np.pi)), reset=False, t0=d.time)

    print(__doc__.split("Keys")[1].split("Same control")[0], flush=True)
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = r.pelvis_id
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.2, -15.0, 135.0
        friction = float(m.geom_friction[m.geom("floor").id, 0])
        zeros2, zeros23 = np.zeros(2), np.zeros((2, 3))
        while viewer.is_running():
            tick = time.perf_counter()
            with viewer.lock():
                if st["reset"]:
                    reset()
                if st["arm_k"] < move_steps:
                    st["arm_k"] += 1
                    st["arm"] = st["arm_from"] + (st["arm_k"] / move_steps) * (st["arm_to"] - st["arm_from"])
                cmd = np.concatenate([st["cmd"], [height]])
                push = st["push"] if st["push_left"] > 0 else np.zeros(3)
                obs = {"state": r.build_state_obs(d, st["last_action"], cmd,
                                                  st["arm"], st["phase"], rng=rng),
                       "privileged_state": r.build_priv_obs(d, zeros2, zeros23,
                                                            friction, push)}
                action = np.clip(np.asarray(policy(obs)), -1.0, 1.0)
                st["last_action"] = action.astype(np.float32)
                st["phase"] = (st["phase"] + phase_inc) % (2 * np.pi)
                legs = r.default_pose[:N_LT] + action[:N_LT] * r.action_scale_lt
                target = np.clip(np.concatenate([legs, st["arm"]]), r.joint_lo, r.joint_hi)
                d.xfrc_applied[r.torso_id, :3] = push
                for _ in range(r.decimation):
                    tau = r.kp * (target - d.qpos[7:]) - r.kd * d.qvel[6:]
                    d.ctrl[:] = np.clip(tau, -r.tau_max, r.tau_max)
                    mujoco.mj_step(m, d)
                d.xfrc_applied[r.torso_id, :3] = 0.0
                st["push_left"] = max(0, st["push_left"] - 1)

                g = quat_rotate_inv_np(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
                if d.qpos[2] < r.height_min or rewards.tilt_exceeded(
                        g, r.max_tilt, r.max_pitch_forward, np):
                    print(f"FELL after {d.time - st['t0']:.1f} s — resetting", flush=True)
                    st["reset"] = True
            viewer.sync()
            time.sleep(max(0.0, r.ctrl_dt - (time.perf_counter() - tick)))


if __name__ == "__main__":
    main()
