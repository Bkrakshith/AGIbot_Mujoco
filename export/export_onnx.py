#!/usr/bin/env python
"""Step 2 of the export contract (spec section 8): torch -> ONNX + JSON sidecar.

  python export/export_onnx.py --torch outputs/onnx/student.pt \
                               [--out outputs/onnx/student.onnx]

ONNX opset 17, static shapes: obs_history (1, H, STUDENT_OBS_SIZE) float32 -> action (1, NU).
(X1: 48 -> 12.)
Obs normalisation is baked into the graph. The JSON sidecar is the ONLY
contract with the downstream Isaac Sim / ROS2 deployment build: obs ordering,
action scaling, default joint angles, PD gains, control rate, command
envelope.
"""

import argparse
import json
import os

# CPU by design (eval/export path). setdefault: an explicit env override
# still wins; without this, jax auto-discovers the CUDA plugin, which on this
# box trips the stale-nvJitLink LD_LIBRARY_PATH shadow (see backend.py).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from params_to_torch import StudentPolicyNet  # noqa: E402
from x1_locomotion.env import STUDENT_OBS_SIZE  # noqa: E402
from x1_locomotion.layout import N_ARM, N_LT, NU  # noqa: E402


def _payload_assumptions(cfg) -> dict:
    """What payload the policy actually saw, read from config."""
    peak = max(float(st["payload_max_total"]) for st in cfg.train.curriculum)
    return {
        "total_kg": [0.0, peak],
        "note": ("stand/walk policy: no payload was ever sampled in training "
                 "(every curriculum stage has payload_max_total 0)."
                 if peak == 0.0 else "sampled per episode, curriculum-scaled"),
    }


def _command_envelope(cfg) -> dict:
    """The command ranges the policy was trained on, as published limits."""
    c = cfg.train.commands
    return {
        "command_slot_3": "target_base_height_m",
        "vx_range": list(c.vx_range),
        "vy_range": list(c.vy_range),
        "yaw_rate_range": list(c.yaw_range),
        "base_height_range": list(c.height_range),
        "note": "ranges sampled during training; never command outside them.",
    }


def build_sidecar(cfg, meta) -> dict:
    import mujoco
    from x1_locomotion.config import MJX_XML
    m = mujoco.MjModel.from_xml_path(MJX_XML)
    joint_names = [m.joint(i + 1).name for i in range(NU)]  # skip free joint
    act = cfg.actuators
    H = meta["history_length"]
    return {
        "model": "agibot_x1 (agibot_x1_train xyber_x1_serial, 12 leg DoF)",
        "policy_input": {
            "name": "obs_history",
            "shape": [1, H, STUDENT_OBS_SIZE],
            "dtype": "float32",
            "note": ("ring buffer of the last "
                     f"{H} control steps of the observation vector, oldest "
                     "first, newest at index -1; "
                     "normalisation is baked into the graph — feed RAW values. "
                     "PRIME THE BUFFER WITH observation_normalization.mean ON "
                     "RESET, NOT ZEROS — the adaptation CNN is trained on "
                     "normalised history whose unfilled slots are exactly 0, "
                     "and only mean-priming reproduces that after in-graph "
                     "normalisation. Zero-priming feeds -mean/std and destroys "
                     f"the latent for the first {H} steps of every episode."),
        },
        # Published so deployment can prime the ring buffer correctly. These are
        # the SAME constants already baked into the graph; they are exposed here
        # for buffer initialisation, not for the caller to apply.
        "observation_normalization": {
            "mean": [float(v) for v in meta["obs_mean"]],
            "std": [float(v) for v in meta["obs_std"]],
            "applied": "inside the ONNX graph",
            "reset_fill": "mean",
        },
        "observation_layout": [
            {"name": "projected_gravity", "slice": [0, 3],
             "note": "world gravity unit vector expressed in pelvis frame"},
            {"name": "base_angular_velocity", "slice": [3, 6],
             "note": "rad/s, pelvis body frame (IMU gyro)"},
            {"name": "joint_positions_rel", "slice": [6, 6 + NU],
             "note": "q - default_joint_angles, joint_order below"},
            {"name": "joint_velocities", "slice": [6 + NU, 6 + 2 * NU],
             "note": "rad/s"},
            {"name": "previous_action", "slice": [6 + 2 * NU, 6 + 3 * NU],
             "note": "policy output of the previous control step, in [-1, 1]"},
            {"name": "command", "slice": [6 + 3 * NU, 10 + 3 * NU],
             "note": "[vx m/s, vy m/s, yaw_rate rad/s, base_height m]"},
            {"name": "arm_pose_command", "slice": [10 + 3 * NU, 10 + 3 * NU + N_ARM],
             "note": f"commanded arm pose (rad), {N_ARM} arm joints"},
            {"name": "gait_clock", "slice": [10 + 3 * NU + N_ARM, 12 + 3 * NU + N_ARM],
             "note": ("[sin(theta), cos(theta)] of a free-running gait phase "
                      "the DEPLOYMENT SIDE must maintain: "
                      "theta += 2*pi*gait_clock_hz*dt each control tick, "
                      "modulo 2*pi. Initial value is arbitrary (training "
                      "randomises it); keep the clock running while standing "
                      "— the policy learns to ignore it at zero command.")},
        ],
        "gait_clock_hz": 1.0 / cfg.rewards.cpg.cycle_time_s,
        "policy_output": {
            "name": "action", "shape": [1, NU], "dtype": "float32",
            "range": [-1.0, 1.0],
        },
        "action_scaling": {
            "joint_order": joint_names,
            "legs_torso": {
                "indices": [0, N_LT],
                "rule": "target = default_joint_angles[i] + action[i] * scale[i]",
                "scale": list(act.action_scale_legs_torso),
            },
            "arms": {
                "indices": [N_LT, NU],
                "rule": f"target = arm_pose_command[i-{N_LT}] + action[i] * residual_clip",
                "residual_clip": act.control.arm_residual_clip,
            },
            "clip": "targets clipped to joint limits",
        },
        "default_joint_angles": list(act.default_pose),
        "nominal_carry_pose": list(act.nominal_carry_pose),
        "pd_gains": {"kp": list(act.kp), "kd": list(act.kd),
                     "note": "torque = kp*(target-q) - kd*qd at the low-level rate"},
        "control_rate_hz": 1.0 / (act.control.physics_dt * act.control.decimation),
        "lowlevel_rate_hz": 1.0 / act.control.physics_dt,
        "payload_assumptions": _payload_assumptions(cfg),
        # What values the policy was trained on; deployment must not
        # command outside them.
        "command_envelope": _command_envelope(cfg),
        "training_commit_meta": meta.get("source_checkpoint", ""),
        "opset": 17,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", dest="torch_path", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from x1_locomotion.config import REPO_ROOT, load_config
    cfg = load_config()
    bundle = torch.load(args.torch_path, weights_only=False)
    meta = bundle["meta"]
    meta.setdefault("source_checkpoint", bundle.get("source_checkpoint", ""))
    model = StudentPolicyNet(meta)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    out = args.out or os.path.join(REPO_ROOT, "outputs", "onnx", "student.onnx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dummy = torch.zeros(1, meta["history_length"], STUDENT_OBS_SIZE)
    torch.onnx.export(model, (dummy,), out, opset_version=17,
                      input_names=["obs_history"], output_names=["action"],
                      dynamo=False)
    print(f"onnx -> {out}")

    # The normalisation constants are already inside the graph; publish them so
    # the deployment side can prime its ring buffer with the mean (see the
    # policy_input note — zero-priming silently breaks the adaptation CNN).
    meta["obs_mean"] = model.obs_mean.detach().cpu().numpy().tolist()
    meta["obs_std"] = model.obs_std.detach().cpu().numpy().tolist()

    sidecar = build_sidecar(cfg, meta)
    sidecar_path = os.path.splitext(out)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"sidecar -> {sidecar_path}")


if __name__ == "__main__":
    main()
