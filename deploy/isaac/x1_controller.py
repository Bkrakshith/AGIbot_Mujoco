"""X1 ONNX policy + joint controller for Isaac Sim, driven entirely by the sidecar.

Shared by run_x1_policy.py (headless battery / demo) and x1_office_teleop.py
(keyboard control inside a running Isaac Sim window).

Control loop (identical to training): physics 500 Hz; every physics step the
joint effort is  clip(kp*(q* - q) - kd*qd, +-torque_limit) - passive_damping*qd
(simulator drives must have zero stiffness/damping); the policy runs every 5th
step (100 Hz) on a 50-step raw observation history primed with the
normalisation mean. Legs: q* = default + 0.5*action. Arms: q* = task pose.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_ONNX = os.path.join(REPO, "policy", "student.onnx")
if os.path.join(HERE, "_pydeps") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "_pydeps"))      # local onnxruntime

import onnxruntime as ort  # noqa: E402

# arm pose library (same order as configs/domain_rand.yaml arm_poses), one arm
ARM_POSES = {
    "down": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "forward": [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0],
    "reach": [-1.2, -0.1, 0.0, 1.0, 0.0, 0.0],
    "sideways": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
    "overhead": [-2.9, -0.25, 0.0, 0.0, 0.0, 0.0],
    "back": [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
    "carry": [-0.5, -0.1, 0.0, 1.4, 0.0, 0.0],
}


def arms(name):
    return np.array(ARM_POSES[name] * 2, float)


def quat_to_rot(quat):
    w, x, y, z = quat
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class X1Policy:
    """The sidecar contract + ONNX session: observation in, joint targets out."""

    def __init__(self, onnx_path=DEFAULT_ONNX, obs_noise=False, seed=0):
        side = json.load(open(os.path.splitext(onnx_path)[0] + ".json"))
        rm = side["robot_model"]
        self.joints = rm["joint_order"]                       # 24, legs first then arms
        self.nu = len(self.joints)
        self.n_act = side["policy_output"]["shape"][1]         # 12 leg actions
        self.h = side["policy_input"]["shape"][1]              # 50
        self.q0 = np.array(side["default_joint_angles"])
        self.kp = np.array(side["pd_gains"]["kp"], float)
        self.kd = np.array(side["pd_gains"]["kd"], float)
        self.tau = np.array(rm["torque_limits_nm"], float)
        self.damp = np.array(rm["passive_damping_nms_per_rad"], float)
        self.armature = np.array(rm["armature_kgm2"], float)
        self.lim = np.array(rm["joint_limits_rad"], float)
        self.scale = np.array(side["action_scaling"]["legs_torso"]["scale"], float)
        self.dt = rm["physics_dt_s"]
        self.decim = int(round(rm["policy_dt_s"] / self.dt))
        self.clock_inc = 2 * math.pi * side["gait_clock_hz"] * rm["policy_dt_s"]
        self.mean = np.array(side["observation_normalization"]["mean"], np.float32)
        self.noise = side["training_obs_noise_std"]
        self.height = float(side["command_envelope"]["base_height_range"][0])
        self.base_z = float(rm["initial_state"]["base_height_m"])
        self.envelope = side["command_envelope"]
        self.obs_noise = obs_noise
        self.rng = np.random.default_rng(seed)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.reset()

    def reset(self):
        self.phase = float(self.rng.uniform(0, 2 * math.pi))
        self.last_action = np.zeros(self.n_act, np.float32)
        self.hist = np.tile(self.mean, (1, self.h, 1)).astype(np.float32)

    def clip_cmd(self, cmd):
        e = self.envelope
        return np.array([np.clip(cmd[0], *e["vx_range"]), np.clip(cmd[1], *e["vy_range"]),
                         np.clip(cmd[2], *e["yaw_rate_range"])])

    def act(self, g, gyro, q, qd, cmd, arm_cmd):
        """One 100 Hz policy step -> 24 joint position targets (sidecar order)."""
        qj, vj = q - self.q0, qd
        if self.obs_noise:
            n, r = self.noise, self.rng
            g = g + r.normal(0, n["gravity"], 3)
            gyro = gyro + r.normal(0, n["gyro"], 3)
            qj = qj + r.normal(0, n["joint_pos"], self.nu)
            vj = vj + r.normal(0, n["joint_vel"], self.nu)
        obs = np.concatenate([g, gyro, qj, vj, self.last_action, cmd, [self.height], arm_cmd,
                              [math.sin(self.phase), math.cos(self.phase)]]).astype(np.float32)
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[0, -1] = obs
        a = np.clip(self.sess.run(None, {self.inp: self.hist})[0][0], -1.0, 1.0)
        self.last_action = a.astype(np.float32)
        self.phase = (self.phase + self.clock_inc) % (2 * math.pi)
        legs = self.q0[:self.n_act] + a * self.scale
        return np.clip(np.concatenate([legs, arm_cmd]), self.lim[:, 0], self.lim[:, 1])


class X1Controller:
    """Runs X1Policy on an isaacsim.core.experimental.prims.Articulation.

    Call setup() once physics is running, reset() to (re)place the robot, and
    physics_step(cmd, arm_cmd) from a POST_PHYSICS_STEP callback.
    """

    def __init__(self, robot, policy):
        self.robot, self.p = robot, policy
        self.ready = False
        self.alive = False

    def setup(self):
        r, p = self.robot, self.p
        names = list(r.dof_names)
        missing = [j for j in p.joints if j not in names]
        assert not missing, f"USD is missing joints {missing}"
        self.idx = np.array([names.index(j) for j in p.joints])   # sidecar order -> dof index
        r.set_dof_drive_types("force")
        r.switch_dof_control_mode("effort")
        r.set_dof_gains(np.zeros((1, p.nu)), np.zeros((1, p.nu)))
        r.set_dof_armatures(self.to_dof(p.armature)[None])
        r.set_dof_max_efforts(self.to_dof(p.tau)[None])
        self.ready = True

    def to_dof(self, v):          # sidecar order -> articulation dof order
        out = np.zeros(self.p.nu)
        out[self.idx] = v
        return out

    def reset(self, xy=(0.0, 0.0), yaw=0.0, arm_cmd=None, rng=None):
        p, r = self.p, self.robot
        rng = rng if rng is not None else p.rng
        p.reset()
        self.k, self.t = 0, 0.0
        self.alive, self.t_fall = True, None
        arm_cmd = arms("down") if arm_cmd is None else np.asarray(arm_cmd, float)
        q = p.q0.copy()
        q[:p.n_act] += rng.uniform(-0.03, 0.03, p.n_act)
        q[p.n_act:] = arm_cmd
        self.target = q.copy()
        r.set_world_poses(positions=[[xy[0], xy[1], p.base_z]],
                          orientations=[[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
        r.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        r.set_dof_positions(self.to_dof(q)[None])
        r.set_dof_velocities(np.zeros((1, p.nu)))
        r.set_dof_efforts(np.zeros((1, p.nu)))

    def state(self):
        r = self.robot
        q = r.get_dof_positions().numpy()[0][self.idx]
        qd = r.get_dof_velocities().numpy()[0][self.idx]
        pos, quat = r.get_world_poses()
        lin, ang = r.get_velocities()
        R = quat_to_rot(quat.numpy()[0])
        g = R.T @ np.array([0.0, 0.0, -1.0])           # gravity in the pelvis frame
        gyro = R.T @ ang.numpy()[0]                     # body-frame angular velocity
        self.pos, self.R = pos.numpy()[0], R
        return q, qd, g, gyro, self.pos, R.T @ lin.numpy()[0]

    def physics_step(self, cmd, arm_cmd):
        """One physics step. Returns False once the robot has fallen."""
        if not self.alive:
            return False
        p = self.p
        q, qd, g, gyro, pos, _ = self.state()
        if self.k % p.decim == 0:
            self.target = p.act(g, gyro, q, qd, np.asarray(cmd, float), np.asarray(arm_cmd, float))
            if pos[2] < 0.35 or -g[2] < math.cos(1.0):        # training termination
                self.alive, self.t_fall = False, self.t
                return False
        tau = np.clip(p.kp * (self.target - q) - p.kd * qd, -p.tau, p.tau) - p.damp * qd
        self.robot.set_dof_efforts(self.to_dof(tau)[None])
        self.k += 1
        self.t = self.k * p.dt
        return True
