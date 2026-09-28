"""Walking-policy loop for the manip scenes: the ONNX student at 100 Hz, joint
PD at 500 Hz with the sidecar gains and torque limits, full X1 observation
noise, free-running gait clock. Arms follow the task command by PD only, as in
training. Same observation layout as src/x1_locomotion/cpu_eval.py."""

import mujoco
import numpy as np

from balance import Policy
from common import DEFAULT_POSE, KD, KP, NOISE_STD, SC, TAU_MAX, quat_rotate_inv

CTRL_DT = 0.01
DECIM = 5
HEIGHT_CMD = 0.613
MAX_TILT = 1.0              # configs/rewards.yaml termination
HEIGHT_MIN = 0.35


class PolicyRunner:
    def __init__(self, scene, seed=0, noise=True):
        self.s = scene
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.pol = Policy()
        self.phase = self.rng.uniform(0.0, 2.0 * np.pi)
        self.last = np.zeros(12, np.float32)
        self.fell = False
        self.t = 0.0
        self.pre_substep = None     # optional hook(scene) run before every mj_step

    def perturb_legs(self, amp=0.03):
        self.s.d.qpos[self.s.qadr[:12]] += self.rng.uniform(-amp, amp, 12)
        mujoco.mj_forward(self.s.m, self.s.d)

    def pitch_roll(self):
        g = quat_rotate_inv(self.s.d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        return (float(np.degrees(np.arctan2(g[0], -g[2]))),      # + = forward
                float(np.degrees(np.arctan2(g[1], -g[2]))))

    def step(self, arm, cmd=(0.0, 0.0, 0.0)):
        s, d, rng = self.s, self.s.d, self.rng
        g = quat_rotate_inv(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        gyro = d.qvel[3:6].copy()
        qj = s.q() - DEFAULT_POSE
        vj = s.qd()
        if self.noise:
            g = g + rng.normal(0, NOISE_STD["gravity"], 3)
            gyro = gyro + rng.normal(0, NOISE_STD["gyro"], 3)
            qj = qj + rng.normal(0, NOISE_STD["joint_pos"], 24)
            vj = vj + rng.normal(0, NOISE_STD["joint_vel"], 24)
        obs = np.concatenate([g, gyro, qj, vj, self.last, list(cmd) + [HEIGHT_CMD], arm,
                              [np.sin(self.phase), np.cos(self.phase)]]).astype(np.float32)
        act = np.clip(self.pol(obs), -1.0, 1.0)
        self.last = act.astype(np.float32)
        self.phase = (self.phase + 2 * np.pi * CTRL_DT * SC["gait_clock_hz"]) % (2 * np.pi)
        tgt = np.concatenate([DEFAULT_POSE[:12] + 0.5 * act, arm])
        for _ in range(DECIM):
            tau = KP * (tgt - s.q()) - KD * s.qd()
            d.ctrl[s.act] = np.clip(tau, -TAU_MAX, TAU_MAX)
            if self.pre_substep is not None:
                self.pre_substep(s)
            mujoco.mj_step(s.m, d)
        self.t += CTRL_DT
        gt = quat_rotate_inv(d.qpos[3:7], np.array([0.0, 0.0, -1.0]))
        if d.qpos[2] < HEIGHT_MIN or -gt[2] < np.cos(MAX_TILT):
            self.fell = True
        return not self.fell
