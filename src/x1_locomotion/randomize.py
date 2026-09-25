"""Per-episode domain randomisation and mid-episode events (spec section 6).

Unlike the stock Playground pattern (model fields randomised once per env
instance via brax's `randomization_fn`), every field here is sampled at every
reset and carried in `state.info`. The env rebuilds the mjx.Model view each
step with `model.replace(**fields)`, so payload swaps and true per-episode DR
work under auto-reset. All ranges come from configs/domain_rand.yaml.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ModelBase(NamedTuple):
    """Nominal (unrandomised) model fields + ids, captured once at env init."""

    body_mass: jax.Array          # (nbody,)
    body_ipos: jax.Array          # (nbody, 3)
    body_inertia: jax.Array       # (nbody, 3)
    dof_damping: jax.Array        # (nv,)
    dof_armature: jax.Array       # (nv,)
    geom_friction: jax.Array      # (ngeom, 3)
    geom_solref: jax.Array        # (ngeom, 2)
    payload_ids: jax.Array        # (2,) left, right payload body ids
    pelvis_id: int
    floor_geom_id: int


def _u(rng, lo, hi, shape=()):
    return jax.random.uniform(rng, shape, minval=lo, maxval=hi)


def sample_payload(rng, dr, stage_max_total: float, symmetric: bool):
    """Total mass U[0, stage_max], split U[0,1] between hands (deliberately
    covering fully asymmetric 7/0), CoM offset per hand per axis."""
    k1, k2, k3 = jax.random.split(rng, 3)
    lo, hi = dr.payload.total_mass_range
    total = _u(k1, lo, jnp.minimum(hi, stage_max_total))
    frac = jnp.where(
        symmetric, 0.5, _u(k2, dr.payload.split_fraction_range[0], dr.payload.split_fraction_range[1])
    )
    masses = jnp.array([total * frac, total * (1.0 - frac)])
    off_lo, off_hi = dr.payload.com_offset_range
    com = _u(k3, off_lo, off_hi, (2, 3))
    return masses, com


def payload_fields(base: ModelBase, fields: dict, masses, com, dr) -> dict:
    """Write payload masses / CoM offsets / point-mass inertia into the model
    field dict. `masses` is (2,), `com` is (2, 3) offsets from the nominal
    attachment point."""
    r2 = dr.payload.inertia_radius ** 2
    body_mass = fields["body_mass"].at[base.payload_ids].set(masses + 1e-3)
    ipos = base.body_ipos[base.payload_ids] + com
    body_ipos = fields["body_ipos"].at[base.payload_ids].set(ipos)
    # sphere-ish point mass: I = 2/5 m r^2 per axis
    inertia = jnp.broadcast_to((0.4 * (masses + 1e-3) * r2)[:, None], (2, 3))
    body_inertia = fields["body_inertia"].at[base.payload_ids].set(inertia)
    return {**fields, "body_mass": body_mass, "body_ipos": body_ipos, "body_inertia": body_inertia}


def sample_model_fields(rng, base: ModelBase, dr) -> tuple[dict, dict]:
    """Sample the per-episode mjx.Model field overrides (payload excluded —
    that's layered on with payload_fields so swaps can rewrite it alone).

    Returns (fields, priv) where priv holds scalars that feed the privileged
    observation (friction) and actuation randomisation applied in the env
    (kp/kd scale, motor strength, latency)."""
    keys = jax.random.split(rng, 8)

    # link masses ±10 %, base (pelvis) mass ±3 kg with CoM shift ±5 cm
    scale = _u(keys[0], dr.body.link_mass_scale_range[0], dr.body.link_mass_scale_range[1],
               base.body_mass.shape)
    body_mass = base.body_mass * scale
    body_inertia = base.body_inertia * scale[:, None]
    base_delta = _u(keys[1], dr.body.base_mass_delta_range[0], dr.body.base_mass_delta_range[1])
    body_mass = body_mass.at[base.pelvis_id].add(base_delta)
    com_shift = _u(keys[2], dr.body.base_com_shift_range[0], dr.body.base_com_shift_range[1], (3,))
    body_ipos = base.body_ipos.at[base.pelvis_id].add(com_shift)

    # ground friction + restitution (via floor solref damping ratio)
    friction = _u(keys[3], dr.ground.friction_range[0], dr.ground.friction_range[1])
    geom_friction = base.geom_friction.at[base.floor_geom_id, 0].set(friction)
    restitution = _u(keys[4], dr.ground.restitution_range[0], dr.ground.restitution_range[1])
    geom_solref = base.geom_solref.at[base.floor_geom_id, 1].set(1.0 - restitution)

    # joint damping / armature ±30 %
    damping = base.dof_damping * _u(keys[5], dr.actuation.joint_damping_scale_range[0],
                                    dr.actuation.joint_damping_scale_range[1], base.dof_damping.shape)
    armature = base.dof_armature * _u(keys[6], dr.actuation.armature_scale_range[0],
                                      dr.actuation.armature_scale_range[1], base.dof_armature.shape)

    k_kp, k_kd, k_motor, k_lat = jax.random.split(keys[7], 4)
    priv = {
        "friction": friction,
        "kp_scale": _u(k_kp, dr.actuation.pd_gain_scale_range[0], dr.actuation.pd_gain_scale_range[1]),
        "kd_scale": _u(k_kd, dr.actuation.pd_gain_scale_range[0], dr.actuation.pd_gain_scale_range[1]),
        "motor_scale": _u(k_motor, dr.actuation.motor_strength_range[0], dr.actuation.motor_strength_range[1]),
        "action_delay": jax.random.randint(
            k_lat, (), dr.actuation.action_latency_steps[0], dr.actuation.action_latency_steps[1] + 1),
    }
    fields = {
        "body_mass": body_mass,
        "body_ipos": body_ipos,
        "body_inertia": body_inertia,
        "dof_damping": damping,
        "dof_armature": armature,
        "geom_friction": geom_friction,
        "geom_solref": geom_solref,
    }
    return fields, priv


def sample_swap(rng, dr, masses, stage_max_total: float, enabled: bool, ctrl_dt: float):
    """Payload swap event: with probability p, at a random step, one hand's
    mass changes instantly (box picked up / set down). Returns
    (swap_step, post_swap_masses); swap_step is a huge sentinel when disabled."""
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    fires = enabled & (jax.random.uniform(k1) < dr.payload.swap_probability)
    lo_s, hi_s = dr.payload.swap_window_s
    step = jax.random.randint(k2, (), int(lo_s / ctrl_dt), int(hi_s / ctrl_dt))
    swap_step = jnp.where(fires, step, jnp.iinfo(jnp.int32).max)
    hand = jax.random.bernoulli(k3)  # True = left
    other = jnp.where(hand, masses[1], masses[0])
    new_mass = _u(k4, 0.0, jnp.maximum(stage_max_total - other, 0.0))
    post = jnp.where(hand,
                     jnp.array([new_mass, masses[1]]),
                     jnp.array([masses[0], new_mass]))
    return swap_step, post


def sample_push(rng, dr, push_max_n: float, ctrl_dt: float, cur_step):
    """Sample the next push window: start step and horizontal force vector."""
    k1, k2, k3 = jax.random.split(rng, 3)
    lo_s, hi_s = dr.push.interval_range_s
    start = cur_step + jax.random.randint(k1, (), int(lo_s / ctrl_dt), int(hi_s / ctrl_dt))
    angle = _u(k2, 0.0, 2.0 * jnp.pi)
    mag = _u(k3, 0.0, push_max_n)
    force = jnp.array([mag * jnp.cos(angle), mag * jnp.sin(angle), 0.0])
    return start, force


def sample_arm_command(rng, dr, nominal_pose, joint_lo, joint_hi, scale: float):
    """Carry-pose command: nominal ± per-joint max offsets * curriculum scale,
    mirrored across arms, clipped to joint limits. nominal_pose is (N_ARM,);
    a robot without arm joints (X1) gets the empty command back."""
    n = nominal_pose.shape[0]
    if n == 0:
        return nominal_pose
    offsets_max = jnp.asarray(dr.arm_pose_max_offsets)  # (n/2,)
    off = jax.random.uniform(rng, (n,), minval=-1.0, maxval=1.0)
    off = off * jnp.tile(offsets_max, 2) * scale
    return jnp.clip(nominal_pose + off, joint_lo, joint_hi)


def sample_command(rng, cmds, stage_switch_enabled: bool, ctrl_dt: float):
    """Velocity + height command per spec section 4: 35 % stand-still, 45 %
    slow locomotion, 20 % command switching. Returns (cmd, cmd2, switch_step)
    where cmd = [vx, vy, wyaw, height]."""
    k_mode, k1, k2, k3, k4, k5, k6, k7, k8, kh = jax.random.split(rng, 10)

    def walk_cmd(ka, kb, kc):
        return jnp.array([
            _u(ka, cmds.vx_range[0], cmds.vx_range[1]),
            _u(kb, cmds.vy_range[0], cmds.vy_range[1]),
            _u(kc, cmds.yaw_range[0], cmds.yaw_range[1]),
        ])

    height = _u(kh, cmds.height_range[0], cmds.height_range[1])
    u = jax.random.uniform(k_mode)
    p_stand, p_walk = cmds.p_stand, cmds.p_walk
    # mode 0: stand, 1: walk, 2: switch (walk -> stop or stop -> walk)
    mode = jnp.where(u < p_stand, 0, jnp.where(u < p_stand + p_walk, 1, 2))
    mode = jnp.where(stage_switch_enabled, mode, jnp.minimum(mode, 1))

    zero3 = jnp.zeros(3)
    walk1, walk2 = walk_cmd(k1, k2, k3), walk_cmd(k4, k5, k6)
    vel1 = jnp.where(mode == 0, zero3, walk1)
    # switch target: walking episodes stop, standing-start switch episodes walk
    to_stand = jax.random.bernoulli(k7)
    vel2 = jnp.where(mode == 2, jnp.where(to_stand, zero3, walk2), vel1)

    lo_s, hi_s = cmds.switch_window_s
    sw = jax.random.randint(k8, (), int(lo_s / ctrl_dt), int(hi_s / ctrl_dt))
    switch_step = jnp.where(mode == 2, sw, jnp.iinfo(jnp.int32).max)

    cmd = jnp.concatenate([vel1, height[None]])
    cmd2 = jnp.concatenate([vel2, height[None]])
    return cmd, cmd2, switch_step
