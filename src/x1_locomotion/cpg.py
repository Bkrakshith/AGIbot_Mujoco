"""Central Pattern Generator for the AgiBot X1 walking gait reference.

The shape is the one the vendor validated for the X1 (agibot_x1_train
x1_dh_stand_env.compute_ref_state): the STANCE leg tracks the default pose,
the SWING leg adds a per-joint delta scaled by a half-sine, and a small dead
band around the sine's zero crossings is double support (both legs at
default). The X1 hip-pitch axis is mounted at 45 deg, so a clean
"hip pitch = stride" mapping does not exist; per-joint deltas from the vendor
(rewards.yaml cpg.swing_delta) capture the coupled motion instead.

Phase convention (one cycle = cpg.cycle_time_s, X1 cfg 0.7 s):
  s = sin θ;  left swings while s < -dead_band, right swings while s > dead_band.
  ref_left  = q0_left  + gate * max(0, -s) * delta_left
  ref_right = q0_right + gate * max(0,  s) * delta_right
`gate` in [0, 1] rises with the commanded speed (yaw folded in), so at
cmd = 0 the reference IS the default pose — standing is never asked to step.

THE PHASE IS OBSERVABLE: env appends (sin θ, cos θ) to the state obs — the
"gait clock" (the X1 policy gets the same two numbers). Without it the phase
is an unobservable latent and no policy can synchronise with the reference.
Deployment runs the same clock: θ += 2π·dt/cycle_time_s per control tick.

Joint order per leg (X1): hip_pitch, hip_roll, hip_yaw, knee_pitch,
ankle_pitch, ankle_roll. Output: left (6) + right (6) = qpos[7:19].
"""

import jax.numpy as jnp


def swing_gate(cmd, cfg):
    """[0, 1] reference amplitude from the command's effective speed."""
    c = cfg.cpg
    v_eff = jnp.sqrt(cmd[0] ** 2 + cmd[1] ** 2
                     + jnp.square(c.yaw_equiv_radius * cmd[2]))
    return jnp.clip(v_eff / c.v_clear_ref, 0.0, 1.0)


def swing_profiles(theta, cfg):
    """(left, right) swing amplitudes in [0, 1] at phase θ."""
    s = jnp.sin(theta)
    db = cfg.cpg.dead_band
    left = jnp.where(s < -db, -s, 0.0)
    right = jnp.where(s > db, s, 0.0)
    return left, right


def cpg_reference(theta, cmd, cfg, q0):
    """(12,) leg reference [left(6), right(6)] at phase θ under command cmd.
    q0 is the (12,) default leg pose (actuators.yaml default_pose[:12])."""
    left, right = swing_profiles(theta, cfg)
    amp = jnp.concatenate([jnp.full(6, left), jnp.full(6, right)])
    return q0 + swing_gate(cmd, cfg) * amp * jnp.asarray(cfg.cpg.swing_delta)


def omega_hz(cfg):
    return 1.0 / cfg.cpg.cycle_time_s


def advance_phase(theta, ctrl_dt, cfg):
    """Advance gait phase by one control step."""
    return (theta + 2.0 * jnp.pi * omega_hz(cfg) * ctrl_dt) % (2.0 * jnp.pi)


def stance_windows(theta, cfg):
    """(2,) bool: is each foot's clock in its STANCE window at phase θ?

    Consistent with the reference: a foot is in swing exactly when its swing
    amplitude is non-zero. The dead band makes both windows overlap around
    the zero crossings (double support, ~6 % of the cycle at dead_band 0.1).

    Drives the gait_contact rewards: joint-angle imitation alone is
    satisfiable by wiggling in place (H1-2 teacher_v3), contact TIMING is the
    signal that forces weight transfer."""
    left, right = swing_profiles(theta, cfg)
    return jnp.array([left == 0.0, right == 0.0])


def clock(theta):
    """The observable gait clock: (sin θ, cos θ), appended to the state obs."""
    return jnp.array([jnp.sin(theta), jnp.cos(theta)])
