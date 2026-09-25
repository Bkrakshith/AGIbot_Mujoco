"""Plain CPU MuJoCo rollout engine for evaluation (spec section 7/8).

Runs the FULL-FIDELITY MJCF (assets/x1_full.xml) — mesh collisions, default
solver — with the same 50 Hz PD control convention as the MJX training env,
and mirrors env._get_obs ordering exactly (noise-free for deterministic eval;
observation-noise robustness is a training-time property).

Termination here is height/tilt only: the full model's mesh self-contacts are
not classified (training's reduced model owns the self-collision term).

Policy adapters: teacher (JAX, privileged), student (JAX, obs-history CNN),
ONNX (onnxruntime, the deployment artifact). All return actions in [-1, 1].
"""

import dataclasses
import json
import os
from typing import Callable, Sequence

import mujoco
import numpy as np

from . import rewards
from .config import FULL_XML, Config
from .env import PRIV_OBS_SIZE, STUDENT_OBS_SIZE
from .layout import LT_QPOS, N_LT, NU


def quat_rotate_inv_np(q, v):
    w, u = q[0], q[1:4]
    return v + 2.0 * np.cross(-u, np.cross(-u, v) + w * v)


def _interp_cmd(schedule, t, before_first):
    """Command at time t from [(t, cmd(4)), ...] waypoints.

    Linear between waypoints, held flat outside them; `before_first` applies
    until the first waypoint's time. Interpolated rather than stepped because
    the training env ramps the height command over ~1.5 s — a stepped command
    is a disturbance the policy never saw."""
    if t <= schedule[0][0]:
        return np.asarray(before_first, dtype=np.float64)
    if t >= schedule[-1][0]:
        return np.asarray(schedule[-1][1], dtype=np.float64)
    for (t0, c0), (t1, c1) in zip(schedule, schedule[1:]):
        if t0 <= t <= t1:
            span = max(t1 - t0, 1e-9)
            u = (t - t0) / span
            return ((1.0 - u) * np.asarray(c0, dtype=np.float64)
                    + u * np.asarray(c1, dtype=np.float64))
    return np.asarray(schedule[-1][1], dtype=np.float64)


@dataclasses.dataclass
class Scenario:
    """Deterministic eval scenario. Schedules are lists of (time_s, value)
    applied at the first control step at/after time_s."""

    name: str
    cmd: Sequence[float]                       # (4,) vx, vy, wyaw, height
    payload_schedule: Sequence[tuple]          # [(t, masses(2), com(2,3))]
    pushes: Sequence[tuple] = ()               # [(t, force(3))], 0.2 s each
    duration_s: float = 20.0
    # Optional [(t, cmd(4)), ...] waypoints, linearly interpolated between —
    # `cmd` is then only the value before the first waypoint. Squat scenarios
    # need this: the motion IS a height trajectory, and a fixed command cannot
    # express stand -> crouch -> stand. Interpolation (rather than stepping at
    # each waypoint, as the payload schedule does) matches training, where the
    # height command is ramped over ~1.5 s and never stepped.
    cmd_schedule: Sequence[tuple] = ()


@dataclasses.dataclass
class EpisodeResult:
    survived: bool
    survived_time_s: float
    orientation_err: float      # mean ||proj_gravity_xy||
    tracking_err: float         # mean ||v_xy_local - cmd_xy|| + |wyaw err|
    height_err: float           # mean |height - cmd_height|
    recovery_times_s: list      # per push/swap event (nan if never recovered)


class CpuRollout:
    def __init__(self, cfg: Config, xml_path: str = FULL_XML):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        act = cfg.actuators
        self.kp = np.asarray(act.kp, dtype=np.float64)
        self.kd = np.asarray(act.kd, dtype=np.float64)
        self.default_pose = np.asarray(act.default_pose, dtype=np.float64)
        self.nominal_carry = np.asarray(act.nominal_carry_pose, dtype=np.float64)
        self.action_scale_lt = np.asarray(act.action_scale_legs_torso, dtype=np.float64)
        self.arm_clip = act.control.arm_residual_clip
        self.decimation = act.control.decimation
        self.ctrl_dt = act.control.physics_dt * self.decimation
        self.tau_max = self.model.actuator_ctrlrange[:, 1].copy()
        self.joint_lo = self.model.jnt_range[1:, 0].copy()
        self.joint_hi = self.model.jnt_range[1:, 1].copy()
        m = self.model
        self.pelvis_id = m.body("pelvis").id
        self.torso_id = m.body("torso_link").id
        self.payload_ids = [m.body("left_payload").id, m.body("right_payload").id]
        self.payload_base_ipos = m.body_ipos[self.payload_ids].copy()
        self.push_dur_steps = int(cfg.domain_rand.push.duration_s / self.ctrl_dt)
        self.inertia_r2 = cfg.domain_rand.payload.inertia_radius ** 2
        term = cfg.rewards.termination
        self.height_min = term.base_height_min
        # Must mirror env._tilted, or the battery grades a policy against a
        # rule it was never trained under. A squat policy holding the ~57 deg
        # posture its task needs would read as falling under the old combined
        # 57.3 deg limit, and the battery is the independent check.
        self.max_tilt = term.max_tilt
        self.max_pitch_forward = getattr(term, "max_pitch_forward",
                                         term.max_tilt)

    def _set_payload(self, masses, com):
        m = self.model
        for i, bid in enumerate(self.payload_ids):
            m.body_mass[bid] = masses[i] + 1e-3
            m.body_ipos[bid] = self.payload_base_ipos[i] + com[i]
            m.body_inertia[bid] = 0.4 * (masses[i] + 1e-3) * self.inertia_r2

    def build_state_obs(self, data, last_action, cmd, arm_cmd, gait_phase):
        """Must match env._get_obs 'state' ordering exactly (noise-free).
        gait_phase: the deployment-side CPG clock, advanced by the caller at
        2π·omega_hz per second (any initial value — training randomises it)."""
        quat = data.qpos[3:7]
        proj_g = quat_rotate_inv_np(quat, np.array([0.0, 0.0, -1.0]))
        gyro = data.qvel[3:6]
        qj = data.qpos[7:] - self.default_pose
        vj = data.qvel[6:]
        clock = np.array([np.sin(gait_phase), np.cos(gait_phase)])
        return np.concatenate([proj_g, gyro, qj, vj, last_action, cmd, arm_cmd,
                               clock]).astype(np.float32)

    def build_priv_obs(self, data, masses, com, friction, push_force):
        quat = data.qpos[3:7]
        lin_vel_local = quat_rotate_inv_np(quat, data.qvel[0:3])
        return np.concatenate([
            masses, np.asarray(com).ravel(), lin_vel_local, [friction],
            np.asarray(push_force) / 100.0, [1.0], [1.0],  # kp/kd scale nominal
        ]).astype(np.float32)

    def run_episode(self, scenario: Scenario, policy: Callable, seed: int = 0,
                    record_qpos: bool = False) -> EpisodeResult:
        m, cfg = self.model, self.cfg
        rng = np.random.default_rng(seed)
        data = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, data, m.keyframe("home").id)
        # The keyframe IS default_pose (scripts/build_assets.py); a mismatch
        # means the assets are stale and the policy frame would be wrong.
        if not np.allclose(data.qpos[7:], self.default_pose):
            raise ValueError("home keyframe != actuators.yaml default_pose; "
                             "re-run scripts/build_assets.py")
        data.qpos[LT_QPOS] += rng.uniform(-0.03, 0.03, N_LT)

        payload_sched = sorted(scenario.payload_schedule, key=lambda x: x[0])
        pushes = sorted(scenario.pushes, key=lambda x: x[0])
        masses, com = np.zeros(2), np.zeros((2, 3))
        friction = float(m.geom_friction[m.geom("floor").id, 0])
        cmd = np.asarray(scenario.cmd, dtype=np.float64)
        cmd_sched = sorted(scenario.cmd_schedule, key=lambda x: x[0])
        arm_cmd = self.nominal_carry
        last_action = np.zeros(NU, dtype=np.float32)
        # deployment-convention gait clock: free-running, seeded per episode
        gait_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_inc = 2.0 * np.pi * self.ctrl_dt / self.cfg.rewards.cpg.cycle_time_s

        n_steps = int(scenario.duration_s / self.ctrl_dt)
        events = []          # control-step indices of pushes and swaps
        ori_err, trk_err, hgt_err, stable = [], [], [], []
        qpos_trace = [] if record_qpos else None
        survived, t_survived = True, scenario.duration_s

        pay_i = push_i = 0
        push_end, push_force = -1, np.zeros(3)
        mujoco.mj_forward(m, data)

        for step in range(n_steps):
            t = step * self.ctrl_dt
            if cmd_sched:
                cmd = _interp_cmd(cmd_sched, t, np.asarray(scenario.cmd, float))
            while pay_i < len(payload_sched) and t >= payload_sched[pay_i][0]:
                _, masses, com = payload_sched[pay_i]
                masses, com = np.asarray(masses, float), np.asarray(com, float)
                self._set_payload(masses, com)
                if payload_sched[pay_i][0] > 0:
                    events.append(step)
                pay_i += 1
            if push_i < len(pushes) and t >= pushes[push_i][0]:
                push_force = np.asarray(pushes[push_i][1], float)
                push_end = step + self.push_dur_steps
                events.append(step)
                push_i += 1
            active_push = push_force if step < push_end else np.zeros(3)

            obs = {
                "state": self.build_state_obs(data, last_action, cmd, arm_cmd,
                                              gait_phase),
                "privileged_state": self.build_priv_obs(data, masses, com,
                                                        friction, active_push),
            }
            action = np.clip(np.asarray(policy(obs)), -1.0, 1.0)
            last_action = action.astype(np.float32)
            gait_phase = (gait_phase + phase_inc) % (2.0 * np.pi)

            lt_target = self.default_pose[:N_LT] + action[:N_LT] * self.action_scale_lt
            arm_target = arm_cmd + action[N_LT:] * self.arm_clip
            targets = np.clip(np.concatenate([lt_target, arm_target]),
                              self.joint_lo, self.joint_hi)

            data.xfrc_applied[self.torso_id, :3] = active_push
            for _ in range(self.decimation):
                tau = self.kp * (targets - data.qpos[7:]) - self.kd * data.qvel[6:]
                data.ctrl[:] = np.clip(tau, -self.tau_max, self.tau_max)
                mujoco.mj_step(m, data)
            data.xfrc_applied[self.torso_id, :3] = 0.0

            quat = data.qpos[3:7]
            proj_g = quat_rotate_inv_np(quat, np.array([0.0, 0.0, -1.0]))
            v_local = quat_rotate_inv_np(quat, data.qvel[0:3])
            ori = float(np.linalg.norm(proj_g[:2]))
            ori_err.append(ori)
            trk_err.append(float(np.linalg.norm(v_local[:2] - cmd[:2])
                                 + abs(data.qvel[5] - cmd[2])))
            hgt_err.append(abs(float(data.qpos[2]) - cmd[3]))
            stable.append(ori < 0.1 and np.linalg.norm(data.qvel[3:5]) < 0.5)
            if record_qpos:
                qpos_trace.append(data.qpos.copy())

            tilted = rewards.tilt_exceeded(proj_g, self.max_tilt,
                                           self.max_pitch_forward, np)
            if data.qpos[2] < self.height_min or tilted:
                survived, t_survived = False, t
                break

        recovery = _recovery_times(events, stable, self.ctrl_dt,
                                   window=int(0.5 / self.ctrl_dt))
        result = EpisodeResult(
            survived=survived,
            survived_time_s=t_survived,
            orientation_err=float(np.mean(ori_err)),
            tracking_err=float(np.mean(trk_err)),
            height_err=float(np.mean(hgt_err)),
            recovery_times_s=recovery,
        )
        return (result, np.array(qpos_trace)) if record_qpos else result


def _recovery_times(events, stable, dt, window):
    """Time from each event until `stable` holds for `window` consecutive
    steps. NaN if the episode ends first."""
    out = []
    for ev in events:
        t_rec = float("nan")
        run = 0
        for i in range(ev, len(stable)):
            run = run + 1 if stable[i] else 0
            if run >= window:
                t_rec = (i - window + 1 - ev) * dt
                break
        out.append(t_rec)
    return out


# ------------------------------------------------------------ policy adapters
class TeacherJaxPolicy:
    """Frozen teacher: encoder(priv) -> z, trunk([state, z]) -> action mode."""

    def __init__(self, ckpt_path: str, cfg: Config):
        import jax.numpy as jnp
        from .networks import make_policy_module, normalize_obs
        from .ppo_teacher import load_checkpoint, rehydrate_normalizer
        params, _ = load_checkpoint(ckpt_path)
        if isinstance(params, dict):  # student ckpt carries the teacher tuple
            params = params["teacher"]
        self._norm = rehydrate_normalizer(params[0])
        self._policy_params = params[1]
        self._module = make_policy_module(cfg.train.networks)
        self._normalize = normalize_obs
        self._jnp = jnp

    def __call__(self, obs):
        jnp = self._jnp
        o = self._normalize({"state": jnp.asarray(obs["state"]),
                             "privileged_state": jnp.asarray(obs["privileged_state"])},
                            self._norm)
        logits = self._module.apply(self._policy_params, o["state"],
                                    o["privileged_state"])
        return np.asarray(jnp.tanh(logits[: logits.shape[-1] // 2]))


class StudentJaxPolicy:
    """Distilled student: adaptation CNN over raw obs history + teacher trunk.
    Exactly the computation the ONNX export freezes."""

    def __init__(self, ckpt_path: str, cfg: Config):
        import jax.numpy as jnp
        from .networks import make_adaptation_module, make_policy_module
        from .ppo_teacher import load_checkpoint, rehydrate_normalizer
        params, _ = load_checkpoint(ckpt_path)
        norm = rehydrate_normalizer(params["teacher"][0])
        self._mean = np.asarray(norm.mean["state"])
        self._std = np.asarray(norm.std["state"])
        self._policy_params = params["teacher"][1]
        self._adapt_params = params["adaptation"]
        self._module = make_policy_module(cfg.train.networks)
        self._adapt = make_adaptation_module(cfg.train.networks)
        self._H = cfg.train.networks.adaptation.history_length
        # An unfilled history slot MUST normalise to exactly zero, because that
        # is what distillation trains against: distill_student.py stores already
        # -normalised observations and zeroes the buffer at episode boundaries.
        # Here the buffer holds RAW observations and is normalised on the way
        # into the CNN, so it must be primed with obs_mean — (mean-mean)/std = 0.
        # Priming with 0.0 instead feeds the CNN -mean/std (norm ~14), which is
        # far outside its training distribution and destroys z_hat for the first
        # `history_length` steps of every episode.
        self._hist = np.tile(self._mean, (self._H, 1)).astype(np.float32)
        self._jnp = jnp

    def reset(self):
        self._hist[:] = self._mean

    def __call__(self, obs):
        jnp = self._jnp
        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = obs["state"]
        h_n = (self._hist - self._mean) / self._std
        z_hat = self._adapt.apply(self._adapt_params, jnp.asarray(h_n))
        s_n = jnp.asarray(h_n[-1])
        logits = self._module.apply(self._policy_params, s_n, z_hat,
                                    method="act_with_latent")
        return np.asarray(jnp.tanh(logits[: logits.shape[-1] // 2]))


class OnnxPolicy:
    """The deployment artifact: raw obs history in, action out. Normalisation
    is baked into the graph."""

    def __init__(self, onnx_path: str, cfg: Config):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self._H = cfg.train.networks.adaptation.history_length
        # The ring buffer is primed with the observation mean, NOT zeros — see
        # StudentJaxPolicy. Normalisation lives inside the graph, so priming with
        # the mean is what makes an unfilled slot read as zero to the CNN. The
        # mean is published in the sidecar precisely so deployment can do this.
        sidecar_path = os.path.splitext(onnx_path)[0] + ".json"
        with open(sidecar_path) as fh:
            sidecar = json.load(fh)
        self._mean = np.asarray(
            sidecar["observation_normalization"]["mean"], dtype=np.float32)
        self._hist = np.tile(self._mean, (1, self._H, 1)).astype(np.float32)

    def reset(self):
        self._hist[:] = self._mean

    def __call__(self, obs):
        self._hist = np.roll(self._hist, -1, axis=1)
        self._hist[0, -1] = obs["state"]
        return self._sess.run(None, {self._input: self._hist})[0][0]


def make_policy(cfg: Config, checkpoint: str | None = None, onnx: str | None = None):
    """Pick the right adapter from what the checkpoint contains."""
    if onnx:
        return OnnxPolicy(onnx, cfg)
    from .ppo_teacher import load_checkpoint
    params, _ = load_checkpoint(checkpoint)
    if isinstance(params, dict) and "adaptation" in params:
        return StudentJaxPolicy(checkpoint, cfg)
    return TeacherJaxPolicy(checkpoint, cfg)
