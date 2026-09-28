"""AgiBot X1 stand + walk (CPG-guided) — MJX environment.

Port of the H1-2 stability env (UnitreeH1_2_Factory/Mujoco_unitree) to the
AgiBot X1. Sizes come from layout.py (configs/actuators.yaml `layout:`).

Control split (X1: N_LT = 12 legs, N_ARM = 12 arms, NU = 24, N_ACT = 12):
  actions[0:N_LT]  legs — fully learned position targets around defaults
  arms             follow the TASK arm-pose command (info["arm_cmd"]) through
                   their own PD; the policy observes arm state + command but
                   does not actuate the arms (policy_controls_arms: false).
                   The command holds a pose, then moves linearly to a newly
                   sampled one (forward / sideways / overhead / back / carry /
                   down + noise, curriculum-scaled by stage.arm_range_scale).
PD torque is recomputed at every physics substep (500 Hz), mirroring the
on-robot low-level loop tracking 50 Hz policy targets.

Observations (dict):
  'state' (84 on X1, deployment-safe): proj gravity 3, gyro 3, qpos-rel NU,
      qvel NU, prev action N_ACT, command 4 (vx, vy, wyaw, height), arm pose
      command N_ARM,
      gait clock 2 (sin θ, cos θ — the CPG phase; deployment runs the same
      trivial clock, see the ONNX sidecar). NO base linear velocity.
      Noise applied per configs/domain_rand.yaml (clock is noise-free).
  'privileged_state' (23, teacher only): payload masses 2, payload CoM offsets
      6, base lin vel (local) 3, friction 1, push force 3, PD gain multipliers
      2, motor strength 1, action delay 1, base mass delta 1, base CoM shift 3.
"""

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.base import Env, State
from mujoco import mjx

from . import cpg as gait_cpg
from . import randomize, rewards
from .config import (MJX_XML, Config, Stage, apply_torque_limits, stage_commands,
                     stage_domain_rand)
from .layout import (ARM_QPOS, KNEE_QPOS, LEG_QPOS, LT_QPOS, N_ACT, N_ARM, N_LEG,
                     N_LT, NU, POLICY_CONTROLS_ARMS)

# mjx.Model fields rebuilt from state.info each step (per-episode DR + swaps)
_FIELD_KEYS = (
    "body_mass", "body_ipos", "body_inertia",
    "dof_damping", "dof_armature", "geom_friction", "geom_solref",
)

STUDENT_OBS_SIZE = 3 + 3 + NU + NU + N_ACT + 4 + N_ARM + 2  # 84 on X1 (incl. gait clock)
# 23: payload 8, base lin vel 3, friction 1, push 3, kp/kd scale 2, and
# (teacher_v10, 2026-09-26) motor strength 1, action delay 1, base mass
# delta 1, base CoM shift 3. The last six were sampled per episode but NOT
# shown to the teacher, so under full X1 DR it had to hedge blindly against
# latency / weak motors / a shifted CoM and stalled at ~25 % survival. RMA:
# the teacher sees every randomised factor; the student infers them.
PRIV_OBS_SIZE = 2 + 6 + 3 + 1 + 3 + 2 + 1 + 1 + 1 + 3  # 23
_NOISY_PREFIX = 3 + 3 + NU + NU  # obs entries that carry sensor noise


def quat_rotate_inv(q, v):
    """Express world-frame vector v in the body frame of quaternion q (wxyz)."""
    w, u = q[0], q[1:4]
    return v + 2.0 * jnp.cross(-u, jnp.cross(-u, v) + w * v)


class X1LocomotionEnv(Env):
    """One curriculum stage of the X1 stand/walk task."""

    def __init__(self, cfg: Config, stage: Stage, xml_path: str = MJX_XML,
                 include_history: bool = False, student_noise: bool = False,
                 emit_clean_state: bool = False):
        """student_noise: full domain_rand.obs_noise regardless of
        train.teacher_obs_noise (distillation / fine-tune data).
        emit_clean_state: also return obs['state_clean'] (same layout, no
        sensor noise) so a noise-free teacher can label the student's states."""
        self._cfg = cfg
        self._stage = stage
        self._emit_clean = emit_clean_state
        self._priv_push = 1.0 if cfg.train.get("privileged_push", True) else 0.0
        # 'history' obs entry (raw student-obs ring buffer) is only materialised
        # for Phase C student fine-tuning; teacher training keeps obs small.
        self._history_len = (
            cfg.train.networks.adaptation.history_length if include_history else 0)
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        apply_torque_limits(self._mj_model, cfg.actuators)
        self._model = mjx.put_model(self._mj_model)

        m = self._mj_model
        act = cfg.actuators
        self._physics_dt = act.control.physics_dt
        self._decimation = act.control.decimation
        self._ctrl_dt = self._physics_dt * self._decimation
        self._episode_steps = int(cfg.train.env.episode_length_s / self._ctrl_dt)
        self._push_dur_steps = int(cfg.domain_rand.push.duration_s / self._ctrl_dt)
        # ring buffer long enough for the largest sampled action latency
        self._latency_buf = int(cfg.domain_rand.actuation.action_latency_steps[1]) + 1

        self._default_qpos = jnp.asarray(m.keyframe("home").qpos)
        self._default_pose = jnp.asarray(act.default_pose)  # (NU,)
        self._nominal_carry = jnp.asarray(act.nominal_carry_pose)  # (N_ARM,)
        self._kp = jnp.asarray(act.kp)
        self._kd = jnp.asarray(act.kd)
        self._action_scale_lt = jnp.asarray(act.action_scale_legs_torso)  # (N_LT,)
        self._arm_clip = act.control.arm_residual_clip
        self._tau_max = jnp.asarray(m.actuator_ctrlrange[:, 1])  # symmetric limits
        self._joint_lo = jnp.asarray(m.jnt_range[1:, 0])  # skip free joint
        self._joint_hi = jnp.asarray(m.jnt_range[1:, 1])

        self._pelvis_id = m.body("pelvis").id
        self._torso_id = m.body("torso_link").id
        self._floor_gid = m.geom("floor").id
        self._feet_gids = jnp.array([m.geom("left_foot").id, m.geom("right_foot").id])
        # contact geoms: the vendor sole spheres (the reference box does not collide)
        self._sole_gids = jnp.array([
            [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(f"{s}_sole_")]
            for s in ("left", "right")])
        # Foot box half-extents (x, y), for the support-polygon corners.
        self._foot_half = jnp.array(
            [m.geom("left_foot").size[:2], m.geom("right_foot").size[:2]])
        payload_ids = jnp.array([m.body("left_payload").id, m.body("right_payload").id])
        # the hands, for reach tracking: the payload bodies ride at the wrists
        self._hand_ids = payload_ids

        self._base = randomize.ModelBase(
            body_mass=jnp.asarray(m.body_mass),
            body_ipos=jnp.asarray(m.body_ipos),
            body_inertia=jnp.asarray(m.body_inertia),
            dof_damping=jnp.asarray(m.dof_damping),
            dof_armature=jnp.asarray(m.dof_armature),
            geom_friction=jnp.asarray(m.geom_friction),
            geom_solref=jnp.asarray(m.geom_solref),
            payload_ids=payload_ids,
            pelvis_id=self._pelvis_id,
            floor_geom_id=self._floor_gid,
        )

        # this stage's DR (curriculum-scaled); payload/push fields unchanged
        self._dr = stage_domain_rand(cfg.domain_rand, stage)
        noise = self._dr.obs_noise
        self._noise_scale = jnp.concatenate([
            jnp.full(3, noise.gravity),
            jnp.full(3, noise.gyro),
            jnp.full(NU, noise.joint_pos),
            jnp.full(NU, noise.joint_vel),
        ])
        # Teacher vs student noise. The teacher (single frame + privileged
        # obs) cannot filter the X1's 1.3 rad/s joint-velocity noise: under
        # full DR, removing noise alone lifted its walk survival 8 % -> 42 %
        # (teacher_v11 s2d, 2026-09-26). Noise robustness belongs to the
        # student (50-step history, like the vendor's 66-frame actor), which
        # sees the FULL noise: history envs (include_history=True) always get
        # it; teacher training scales it by train.teacher_obs_noise.
        if not (include_history or student_noise):
            self._noise_scale = self._noise_scale * float(
                cfg.train.get("teacher_obs_noise", 1.0))

    # ------------------------------------------------------------- brax API
    @property
    def observation_size(self):
        sizes = {"state": STUDENT_OBS_SIZE, "privileged_state": PRIV_OBS_SIZE}
        if self._emit_clean:
            sizes["state_clean"] = STUDENT_OBS_SIZE
        if self._history_len:
            sizes["history"] = (self._history_len, STUDENT_OBS_SIZE)
        return sizes

    @property
    def action_size(self) -> int:
        return N_ACT

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def dt(self) -> float:
        return self._ctrl_dt

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @property
    def stage(self) -> Stage:
        return self._stage

    # ------------------------------------------------------------- helpers
    def _model_with(self, fields: dict) -> mjx.Model:
        return self._model.replace(**{k: fields[k] for k in _FIELD_KEYS})

    def _physics(self, model, data, motor_targets, kp, kd, tau_max):
        """Run `decimation` substeps with PD recomputed each substep."""

        def substep(d, _):
            tau = kp * (motor_targets - d.qpos[7:]) - kd * d.qvel[6:]
            tau = jnp.clip(tau, -tau_max, tau_max)
            d = mjx.step(model, d.replace(ctrl=tau))
            return d, tau

        data, taus = jax.lax.scan(substep, data, None, self._decimation)
        return data, taus[-1]

    def _feet_contact(self, data):
        """(2,) bool: any penetrating contact on any of each foot's sole
        spheres (the vendor 4-point sole)."""
        g = data.contact.geom
        pen = data.contact.dist < 0.0

        def touching(gids):
            hit = jnp.isin(g[:, 0], gids) | jnp.isin(g[:, 1], gids)
            return jnp.any(hit & pen)

        return jax.vmap(touching)(self._sole_gids)

    def _self_collision(self, data):
        """Any penetrating contact not involving the floor (foot-foot or
        hand-torso — the only enabled self pairs in the reduced model)."""
        g = data.contact.geom
        pen = data.contact.dist < 0.0
        no_floor = (g[:, 0] != self._floor_gid) & (g[:, 1] != self._floor_gid)
        return jnp.any(pen & no_floor)

    def _foot_corners(self, data):
        """The eight support-polygon corners in world xy.

        Each foot is a box, so its four ground corners are the centre plus the
        rotated half-extents. Used by the balance-margin rewards, which need
        the polygon EDGE rather than its centroid.
        """
        c = data.geom_xpos[self._feet_gids][:, :2]           # (2, 2)
        r = data.geom_xmat[self._feet_gids].reshape(2, 3, 3)[:, :2, :2]
        signs = jnp.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        # half-extents are PER FOOT, so they need their own axis before the
        # corner signs broadcast against them
        scaled = self._foot_half[:, None, :] * signs[None, :, :]      # (2,4,2)
        offs = jnp.einsum("fij,fcj->fci", r, scaled)                  # (2,4,2)
        return (c[:, None, :] + offs).reshape(8, 2)

    @staticmethod
    def _tilted(term, proj_g):
        """Tilt termination, with a separate allowance for FORWARD pitch.

        A single combined limit treats a deliberate forward bend and a
        sideways topple as the same event. That blocked the squat task
        outright: reaching a 0.50 m hand target costs 54 Nm at 57 deg of
        pitch versus 90 Nm squatting deep and upright
        (eval/squat_envelope.py), but `max_tilt: 1.0` terminated at 57.3 deg
        with a -100 penalty, putting the cheap posture 0.3 deg from a cliff.
        Measured 2026-08-06: the policy settled at 23 deg and left the hands
        stranded at 0.78 m, 0.28 m short, for all 32M steps of q5.

        The limit is an ellipse in (forward pitch, roll) stretched forward by
        max_pitch_forward / max_tilt, so roll and BACKWARD pitch keep the
        original threshold and only the intended direction opens up. When
        max_pitch_forward == max_tilt this is algebraically identical to the
        old `-proj_g[2] < cos(max_tilt)` (proj_g is a unit vector, so
        proj_g[0]^2 + proj_g[1]^2 == sin^2(tilt)) — locomotion is unaffected.
        """
        return rewards.tilt_exceeded(
            proj_g, term.max_tilt,
            getattr(term, "max_pitch_forward", term.max_tilt), jnp)

    # -- task hooks -------------------------------------------------------
    # Task subclasses override these to script payload and
    # command over the episode. The base implementations are the original
    # behaviour: one instantaneous payload swap, one command switch.

    def _active_command(self, info, step_i):
        """(4,) [vx, vy, wyaw, height] command at control step `step_i`."""
        return jnp.where(step_i >= info["switch_step"], info["cmd2"], info["cmd"])

    def _payload_masses(self, info, step_i):
        """(2,) left/right payload mass (kg) at control step `step_i`."""
        return jnp.where(step_i == info["swap_step"], info["swap_post"],
                         info["payload_mass"])

    def _initial_qpos(self, rng, qpos):
        """Override to start an episode from a non-standing posture.

        Returns (qpos, info) — the info is merged before _extra_reset_info
        runs, so a schedule hook can stay consistent with the posture chosen
        here instead of re-drawing it.
        """
        del rng
        return qpos, {}

    def _extra_reset_info(self, rng, info) -> dict:
        """Per-episode task state, merged into `info` BEFORE the first
        observation is built — the command and payload hooks above may depend
        on it. Returns {} for the locomotion task."""
        del rng, info
        return {}

    def _advance_arm_schedule(self, info, step_i, rng):
        """Task arm-pose command at this step (hold, then linear move to a
        freshly sampled pose). Pure function of info; see reset()."""
        k_pose, k_time = jax.random.split(rng)
        start = step_i >= info["arm_next"]
        new_b = randomize.sample_arm_pose(
            k_pose, self._cfg.domain_rand, self._nominal_carry,
            self._joint_lo[N_LT:], self._joint_hi[N_LT:], self._stage.arm_range_scale)
        hold, move = randomize.sample_arm_timing(k_time, self._cfg.domain_rand, self._ctrl_dt)
        info["arm_a"] = jnp.where(start, info["arm_cmd"], info["arm_a"])
        info["arm_b"] = jnp.where(start, new_b, info["arm_b"])
        info["arm_t0"] = jnp.where(start, step_i, info["arm_t0"])
        info["arm_move"] = jnp.where(start, jnp.maximum(move, 1), info["arm_move"])
        info["arm_next"] = jnp.where(start, step_i + move + hold, info["arm_next"])
        frac = jnp.clip((step_i - info["arm_t0"]) / info["arm_move"], 0.0, 1.0)
        info["arm_cmd"] = info["arm_a"] + frac * (info["arm_b"] - info["arm_a"])
        return info

    def _get_obs(self, data, info, rng):
        quat = data.qpos[3:7]
        proj_g = quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        gyro = data.qvel[3:6]  # free-joint angular velocity is body-frame
        qj = data.qpos[7:] - self._default_pose
        vj = data.qvel[6:]

        cmd = self._active_command(info, info["step"])
        clean = jnp.concatenate([proj_g, gyro, qj, vj])
        noisy = clean + self._noise_scale * jax.random.normal(rng, (_NOISY_PREFIX,))
        tail = jnp.concatenate([info["last_action"], cmd, info["arm_cmd"],
                                gait_cpg.clock(info["gait_phase"])])
        state_obs = jnp.concatenate([noisy, tail])

        lin_vel_local = quat_rotate_inv(quat, data.qvel[0:3])
        priv_obs = jnp.concatenate([
            info["payload_mass"],
            info["payload_com"].ravel(),
            lin_vel_local,
            info["friction"][None],
            # Instantaneous push force: only when train.privileged_push is on.
            # A teacher that SEES the push reacts before a history-based
            # student can sense it, so the student learns un-causable actions
            # (student_v2: standing-push survival 72 % -> 20 % with more push
            # data). The slot stays (zeros) so obs sizes do not change.
            info["push_force_active"] / 100.0 * self._priv_push,
            info["kp_scale"][None],
            info["kd_scale"][None],
            info["motor_scale"][None],
            info["action_delay"][None].astype(jnp.float32) / 4.0,
            info["base_mass_delta"][None] / 3.0,
            info["base_com_shift"] / 0.05,
        ])
        out = {"state": state_obs, "privileged_state": priv_obs}
        if self._emit_clean:
            out["state_clean"] = jnp.concatenate([clean, tail])
        return out

    # ------------------------------------------------------------- reset
    def reset(self, rng: jax.Array) -> State:
        dr = self._cfg.domain_rand
        stage = self._stage
        keys = jax.random.split(rng, 11)

        fields, priv = randomize.sample_model_fields(keys[0], self._base, self._dr)
        masses, com = randomize.sample_payload(
            keys[1], dr, stage.payload_max_total, stage.payload_symmetric)
        fields = randomize.payload_fields(self._base, fields, masses, com, dr)
        swap_step, swap_post = randomize.sample_swap(
            keys[2], dr, masses, stage.payload_max_total, stage.swap_enabled, self._ctrl_dt)
        cmd, cmd2, switch_step = randomize.sample_command(
            keys[3], stage_commands(self._cfg.train.commands, stage), stage.cmd_switch_enabled, self._ctrl_dt)
        k_arm_pose, k_arm_time = jax.random.split(keys[4])
        arm_cmd = randomize.sample_arm_pose(
            k_arm_pose, dr, self._nominal_carry,
            self._joint_lo[N_LT:], self._joint_hi[N_LT:], stage.arm_range_scale)
        arm_hold, _ = randomize.sample_arm_timing(k_arm_time, dr, self._ctrl_dt)
        push_start, push_force = randomize.sample_push(
            keys[5], dr, stage.push_max_n, self._ctrl_dt, jnp.int32(0))

        qpos = self._default_qpos
        qpos = qpos.at[LT_QPOS].add(
            jax.random.uniform(keys[6], (N_LT,), minval=-0.05, maxval=0.05))
        qpos = qpos.at[ARM_QPOS].set(arm_cmd)
        # Task hook: lets a subclass start the episode somewhere other than
        # standing. Shares keys[10] with _extra_reset_info, which reads the
        # returned info rather than re-drawing, so posture and schedule agree.
        qpos, init_info = self._initial_qpos(keys[10], qpos)

        model = self._model_with(fields)
        data = mjx.make_data(model)
        data = data.replace(qpos=qpos, qvel=jnp.zeros_like(data.qvel))
        data = mjx.forward(model, data)

        info = {
            "rng": keys[7],
            "step": jnp.int32(0),
            "cmd": cmd,
            "cmd2": cmd2,
            "switch_step": switch_step,
            "arm_cmd": arm_cmd,
            # arm-pose schedule: hold arm_cmd until arm_next, then move from
            # arm_a to a new arm_b over arm_move steps starting at arm_t0
            "arm_a": arm_cmd,
            "arm_b": arm_cmd,
            "arm_t0": jnp.int32(0),
            "arm_move": jnp.int32(1),
            "arm_next": arm_hold,
            "last_base_vel": jnp.zeros(3),
            "model_fields": fields,
            "payload_mass": masses,
            "payload_com": com,
            "swap_step": swap_step,
            "swap_post": swap_post,
            "friction": priv["friction"],
            "kp_scale": priv["kp_scale"],
            "kd_scale": priv["kd_scale"],
            "motor_scale": priv["motor_scale"],
            "action_delay": priv["action_delay"],
            "base_mass_delta": priv["base_mass_delta"],
            "base_com_shift": priv["base_com_shift"],
            "action_buffer": jnp.zeros((self._latency_buf, N_ACT)),
            "last_action": jnp.zeros(N_ACT),
            "last_qvel": jnp.zeros(NU),
            "push_start": push_start,
            "push_force": push_force,
            "push_force_active": jnp.zeros(3),
            "feet_air_time": jnp.zeros(2),
            "last_foot_pos": data.geom_xpos[self._feet_gids],
            "gait_phase": jax.random.uniform(keys[9], minval=0.0, maxval=2.0 * jnp.pi),
        }
        info.update(init_info)
        info.update(self._extra_reset_info(keys[10], info))

        obs = self._get_obs(data, info, keys[8])
        if self._history_len:
            hist = jnp.zeros((self._history_len, STUDENT_OBS_SIZE)).at[-1].set(obs["state"])
            info["obs_hist"] = hist
            obs["history"] = hist
        metrics = {f"reward/{k}": jnp.float32(0.0) for k in self._reward_keys()}
        metrics["payload_total"] = jnp.sum(info["payload_mass"])
        metrics["termination"] = jnp.float32(0.0)
        return State(pipeline_state=data, obs=obs, reward=jnp.float32(0.0),
                     done=jnp.float32(0.0), metrics=metrics, info=info)

    def _reward_keys(self):
        """Metric names for the reward terms, DERIVED from the weights.

        This was a hardcoded tuple until 2026-08-07 and adding two reward
        terms without extending it crashed training with a pytree mismatch —
        reset() pre-creates these metrics and step() emits one per term, so
        the two must agree exactly. The unit tests never caught it because
        they call compute_reward directly and never run brax's training scan.
        Deriving it means the list cannot drift from the terms again.

        `termination` is a weight but not a term: it is applied once on the
        terminating step, not accumulated per-step.
        """
        return tuple(k for k in self._cfg.rewards.weights if k != "termination")

    # ------------------------------------------------------------- step
    def step(self, state: State, action: jax.Array) -> State:
        cfg = self._cfg
        dr = cfg.domain_rand
        info = dict(state.info)
        rng, k_noise, k_push, k_arm = jax.random.split(info["rng"], 4)
        step_i = info["step"]
        info = self._advance_arm_schedule(info, step_i, k_arm)

        # action latency: apply the action from `action_delay` control steps ago
        action = jnp.clip(action, -1.0, 1.0)
        buf = jnp.concatenate([action[None], info["action_buffer"][:-1]], axis=0)
        applied = buf[info["action_delay"]]

        lt_target = self._default_pose[:N_LT] + applied[:N_LT] * self._action_scale_lt
        if POLICY_CONTROLS_ARMS:
            arm_delta = applied[N_LT:] * self._arm_clip
        else:
            arm_delta = jnp.zeros(N_ARM)
        arm_target = info["arm_cmd"] + arm_delta
        motor_targets = jnp.clip(
            jnp.concatenate([lt_target, arm_target]), self._joint_lo, self._joint_hi)

        # payload event: rewrite payload model fields (no-op when unchanged)
        masses = self._payload_masses(info, step_i)
        fields = randomize.payload_fields(
            self._base, info["model_fields"], masses, info["payload_com"], dr)

        # external push window on the torso
        push_end = info["push_start"] + self._push_dur_steps
        push_on = (step_i >= info["push_start"]) & (step_i < push_end)
        force = info["push_force"] * push_on
        xfrc = jnp.zeros((self._model.nbody, 6)).at[self._torso_id, :3].set(force)
        next_start, next_force = randomize.sample_push(
            k_push, dr, self._stage.push_max_n, self._ctrl_dt, step_i)
        resample = step_i == push_end
        info["push_start"] = jnp.where(resample, next_start, info["push_start"])
        info["push_force"] = jnp.where(resample, next_force, info["push_force"])
        info["push_force_active"] = force

        model = self._model_with(fields)
        data0 = state.pipeline_state.replace(xfrc_applied=xfrc)
        data, tau = self._physics(
            model, data0, motor_targets,
            self._kp * info["kp_scale"], self._kd * info["kd_scale"],
            self._tau_max * info["motor_scale"])

        # ---------------- state extraction for rewards/termination
        quat = data.qpos[3:7]
        proj_g = quat_rotate_inv(quat, jnp.array([0.0, 0.0, -1.0]))
        lin_vel_local = quat_rotate_inv(quat, data.qvel[0:3])
        ang_vel_local = data.qvel[3:6]
        height = data.qpos[2]
        cmd = self._active_command(info, step_i)

        contact = self._feet_contact(data)
        first_contact = contact & (info["feet_air_time"] > 0.0)
        foot_pos = data.geom_xpos[self._feet_gids]
        foot_vel = (foot_pos - info["last_foot_pos"]) / self._ctrl_dt
        joint_acc = (data.qvel[6:] - info["last_qvel"]) / self._ctrl_dt
        # base (pelvis) linear acceleration, world frame: the vibration signal
        base_acc = (data.qvel[0:3] - info["last_base_vel"]) / self._ctrl_dt

        # CPG: command-scaled reference for the current phase, then advance.
        # The phase itself is observable (gait clock in _get_obs) — without
        # that the randomised phase is an unobservable latent and no policy
        # can synchronise with the reference.
        cpg_ref = gait_cpg.cpg_reference(info["gait_phase"], cmd, cfg.rewards,
                                         self._default_pose[:N_LEG])
        cpg_stance = gait_cpg.stance_windows(info["gait_phase"], cfg.rewards)
        info["gait_phase"] = gait_cpg.advance_phase(
            info["gait_phase"], self._ctrl_dt, cfg.rewards)

        x = rewards.RewardInputs(
            cmd=cmd,
            lin_vel_local=lin_vel_local,
            ang_vel_local=ang_vel_local,
            proj_gravity=proj_g,
            base_height=height,
            hand_height=jnp.mean(data.xpos[self._hand_ids][:, 2]),
            feet_contact=contact,
            first_contact=first_contact,
            feet_air_time=info["feet_air_time"],
            feet_vel_xy=jnp.linalg.norm(foot_vel[:, :2], axis=-1),
            feet_vel_z=foot_vel[:, 2],
            com_xy=data.subtree_com[self._pelvis_id][:2],
            support_centroid_xy=jnp.mean(foot_pos[:, :2], axis=0),
            com_vel_xy=data.qvel[:2],   # free-joint linear vel is world-frame
            com_height=data.subtree_com[self._pelvis_id][2],
            foot_corners=self._foot_corners(data),
            arm_residual=arm_delta,
            # effort terms on the LEGS only: the arm torques come from the task
            # command's PD loop, which the policy cannot change
            torque=tau[:N_LT],
            joint_acc=joint_acc[:N_LT],
            base_acc=base_acc,
            action=action,
            prev_action=info["last_action"],
            knee_angles=data.qpos[jnp.array(KNEE_QPOS)],  # left/right knee (absolute rad)
            feet_lateral=quat_rotate_inv(quat, foot_pos[0] - foot_pos[1])[1],
            leg_joint_pos=data.qpos[LEG_QPOS],             # left+right leg joints (12,)
            cpg_ref=cpg_ref,
            cpg_stance=cpg_stance,
        )
        reward, terms = rewards.compute_reward(x, cfg.rewards)

        term = cfg.rewards.termination
        fell = height < term.base_height_min
        tilted = self._tilted(term, proj_g)
        done = fell | tilted | self._self_collision(data)
        reward = reward + cfg.rewards.weights.termination * done

        # ---------------- info/metrics bookkeeping
        info["rng"] = rng
        info["step"] = step_i + 1
        info["action_buffer"] = buf
        info["last_action"] = action
        info["last_qvel"] = data.qvel[6:]
        info["last_base_vel"] = data.qvel[0:3]
        info["payload_mass"] = masses
        info["feet_air_time"] = jnp.where(contact, 0.0, info["feet_air_time"] + self._ctrl_dt)
        info["last_foot_pos"] = foot_pos

        obs = self._get_obs(data, info, k_noise)
        if self._history_len:
            hist = jnp.roll(info["obs_hist"], -1, axis=0).at[-1].set(obs["state"])
            info["obs_hist"] = hist
            obs["history"] = hist
        metrics = dict(state.metrics)
        for k, v in terms.items():
            metrics[f"reward/{k}"] = v
        metrics["payload_total"] = jnp.sum(masses)
        metrics["termination"] = done.astype(jnp.float32)

        return state.replace(pipeline_state=data, obs=obs,
                             reward=jnp.float32(reward),
                             done=done.astype(jnp.float32), metrics=metrics, info=info)
