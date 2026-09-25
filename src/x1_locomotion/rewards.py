"""Reward terms (spec section 5). Stability-dominant shaping; every weight and
kernel width comes from configs/rewards.yaml. Each term function is pure and
takes plain arrays so it can be unit-tested with hand-constructed states."""

from typing import NamedTuple

import jax.numpy as jnp


class RewardInputs(NamedTuple):
    """Everything the reward needs, extracted from the physics state once."""

    cmd: jnp.ndarray              # (4,) vx, vy, wyaw, height (active command)
    lin_vel_local: jnp.ndarray    # (3,) base linear velocity, base frame
    ang_vel_local: jnp.ndarray    # (3,) base angular velocity, base frame
    proj_gravity: jnp.ndarray     # (3,) gravity direction in base frame
    base_height: jnp.ndarray      # ()
    hand_height: jnp.ndarray      # () mean z of the two payload/hand bodies
    feet_contact: jnp.ndarray     # (2,) bool
    first_contact: jnp.ndarray    # (2,) bool, touched down this step
    feet_air_time: jnp.ndarray    # (2,) s, air time at touchdown
    feet_vel_xy: jnp.ndarray      # (2,) horizontal foot speed (m/s)
    feet_vel_z: jnp.ndarray       # (2,) vertical foot speed (m/s)
    com_xy: jnp.ndarray           # (2,) whole-body CoM, world xy
    support_centroid_xy: jnp.ndarray  # (2,) mean foot position, world xy
    com_vel_xy: jnp.ndarray       # (2,) CoM horizontal velocity, world frame
    com_height: jnp.ndarray       # () CoM height, for the capture point
    foot_corners: jnp.ndarray     # (8, 2) support-polygon corners, world xy
    arm_residual: jnp.ndarray     # (14,) applied arm delta (rad)
    torque: jnp.ndarray           # (27,) last applied joint torques
    joint_acc: jnp.ndarray        # (27,) finite-difference joint acceleration
    action: jnp.ndarray           # (27,)
    prev_action: jnp.ndarray      # (27,)
    knee_angles: jnp.ndarray      # (2,) absolute left/right knee joint angles (rad)
    leg_joint_pos: jnp.ndarray    # (12,) absolute left+right leg joints (rad)
    cpg_ref: jnp.ndarray          # (12,) CPG reference for current gait phase
    cpg_stance: jnp.ndarray       # (2,) bool: clock says foot should be in stance


def is_standing(cmd, threshold: float):
    return jnp.linalg.norm(cmd[:3]) < threshold


def tracking_lin_vel(cmd, lin_vel_local, sigma: float):
    """Exp kernel on (vx, vy). At zero command this IS the stand-still reward."""
    err = jnp.sum(jnp.square(cmd[:2] - lin_vel_local[:2]))
    return jnp.exp(-err / sigma)


def tracking_yaw(cmd, ang_vel_local, sigma: float):
    return jnp.exp(-jnp.square(cmd[2] - ang_vel_local[2]) / sigma)


def tracking_height(cmd, base_height, sigma: float):
    return jnp.exp(-jnp.square(cmd[3] - base_height) / sigma)


def tracking_reach(cmd, hand_height, sigma: float):
    """Track a commanded HAND height rather than a pelvis height.

    The palletizing task needs the hands at a target, and many (squat depth,
    torso pitch) pairs put them there at very different stability costs —
    measured 2026-08-06: gripping at 0.50 m needs either 0.40 m of squat with
    an upright torso, or only 0.20 m of squat with 45 deg of pitch. Rewarding
    PELVIS height silently picks the upright-and-deep option, which is the
    worst of them for survival. Rewarding the outcome instead lets the policy
    find its own trade-off, which is also what it did unprompted when two
    different viable squat strategies appeared during the curriculum.

    cmd[3] carries the target for whichever convention the task uses; a squat
    policy's exported sidecar documents it as a reach height, not a base
    height."""
    return jnp.exp(-jnp.square(cmd[3] - hand_height) / sigma)


def tilt_exceeded(proj_g, max_tilt, max_pitch_forward, xp=jnp):
    """Shared by env.step (jnp) and cpu_eval (numpy) — they MUST agree.

    An ellipse in (forward pitch, roll) stretched forward by
    max_pitch_forward / max_tilt: forward pitch is the squat posture, roll and
    BACKWARD pitch are falls. With the two limits equal this is algebraically
    the old `-proj_g[2] < cos(max_tilt)`, since proj_g is a unit vector.
    Pass xp=np from the eval path.
    """
    s, fwd = xp.sin(max_tilt), xp.sin(max_pitch_forward)
    px = xp.where(proj_g[0] > 0.0, proj_g[0] * (s / fwd), proj_g[0])
    past_horizontal = -proj_g[2] < 0.0
    return (xp.sqrt(px * px + proj_g[1] * proj_g[1]) > s) | past_horizontal


def orientation(proj_gravity, free_forward_pitch: bool = False):
    """Keep the torso upright — optionally excluding FORWARD pitch.

    With a strictly negative weight this term structurally UNDER-REACHES a
    hand target. At the posture that exactly hits it the reach kernel is at
    its peak, so its gradient in pitch is zero, while this penalty's gradient
    is not; the optimum therefore always sits short. Measured 2026-08-06 at
    the 0.50 m reach, which needs ~57 deg: weight -3.0 settled at 23 deg,
    -1.5 at 36 deg. Shrinking the weight only slides the equilibrium — the
    bias is structural and goes away only by dropping the forward component.

    Roll and BACKWARD pitch keep the penalty: nothing else rewards them, and
    backward is where this robot topples in a crouch. Forward pitch stays
    bounded without it — past the target the reach error grows again, and
    com_support and the tilt termination both still apply.
    """
    fwd = jnp.minimum(proj_gravity[0], 0.0) if free_forward_pitch \
        else proj_gravity[0]
    return jnp.square(fwd) + jnp.square(proj_gravity[1])



def ang_vel_roll(ang_vel_local):
    """SPLIT from ang_vel_xy (payload-study recipe, 2026-08-04): the combined
    roll+pitch rate penalty let teacher_v6 trade roll quiet for pitch rock
    (student pitch-rate RMS doubled). Separate weights let pitch be damped
    harder than roll, which the gait needs for lateral weight transfer."""
    return jnp.square(ang_vel_local[0])


def ang_vel_pitch(ang_vel_local):
    return jnp.square(ang_vel_local[1])


def lin_vel_z(lin_vel_local):
    return jnp.square(lin_vel_local[2])


def feet_air_time(air_time, first_contact, standing, target: float):
    """Gait shaping: reward air time (clipped at target) on touchdown, only
    when a locomotion command is active."""
    rew = jnp.sum(jnp.clip(air_time, 0.0, target) * first_contact)
    return jnp.where(standing, 0.0, rew)


def feet_stance(feet_contact, standing):
    """BOTH-feet-planted bonus when command is zero."""
    return jnp.where(standing & jnp.all(feet_contact), 1.0, 0.0)


def feet_slip(feet_contact, feet_vel_xy):
    return jnp.sum(jnp.square(feet_vel_xy) * feet_contact)


def feet_impact(first_contact, feet_vel_z):
    """Touchdown vertical-speed penalty — cheap contact-force smoothness proxy."""
    return jnp.sum(jnp.square(feet_vel_z) * first_contact)


def com_support(com_xy, centroid_xy, feet_contact, standing):
    """Horizontal CoM to support-polygon-CENTROID distance when standing.

    SUPERSEDED by support_margin — kept because the locomotion teacher was
    trained with it and its weight is still live in rewards.yaml. The flaw:
    distance to the centroid punishes ANY CoM offset, including the
    hips-back shift that a deep forward reach requires. It is a stability
    proxy that fights the posture the squat task needs.
    """
    d2 = jnp.sum(jnp.square(com_xy - centroid_xy))
    return jnp.where(standing & jnp.all(feet_contact), d2, 0.0)


def support_margin(point_xy, foot_corners):
    """Distance from `point_xy` to the edge of the support polygon.

    Positive inside, negative outside. This is the quantity classical balance
    analysis actually uses, and what eval/squat_envelope.py optimises
    offline — unlike com_support, it does not care WHERE inside the polygon
    the point sits, so it leaves the robot free to shift its hips back for a
    deep reach while still penalising leaving the footprint.

    Uses the axis-aligned bounding box of the eight foot corners rather than
    their true convex hull: a hull is awkward under jit and the box is
    conservative (never reports more margin than the hull has). Fore-aft is
    the binding axis for this task anyway — the polygon is far wider laterally
    than it is long, and the deep-reach failures are fore-aft topples.
    """
    lo = jnp.min(foot_corners, axis=0)
    hi = jnp.max(foot_corners, axis=0)
    return jnp.min(jnp.minimum(point_xy - lo, hi - point_xy))


def capture_point(com_xy, com_vel_xy, com_height):
    """Linear-inverted-pendulum capture point: where the CoM would have to be
    caught to come to rest.

        xi = com + com_vel * sqrt(h / g)

    The DYNAMIC counterpart to the static CoM margin. A crouch that is
    statically fine can still be unrecoverable if the CoM is already moving,
    which is exactly the loaded-descent failure mode (measured 2026-08-07:
    deaths at 3.6-7.1 s, after the descent completes, when the robot arrives
    at the bottom with momentum it cannot arrest). Pratt 2006 / Koolen 2012,
    and the H1-2 recovery result in arXiv 2603.08619 feeds it to the critic.
    """
    h = jnp.maximum(com_height, 0.05)
    return com_xy + com_vel_xy * jnp.sqrt(h / 9.81)


def margin_deficit(margin, target):
    """Squared shortfall of `margin` below `target`; zero once safe.

    A hinge rather than a kernel: past the target there is nothing more to
    gain, so it does not keep pulling the CoM toward the middle of the foot
    the way com_support does.
    """
    return jnp.square(jnp.minimum(margin - target, 0.0))


def arm_residual(delta):
    return jnp.sum(jnp.square(delta))


def torque(tau):
    return jnp.sum(jnp.square(tau))


def joint_acc(acc):
    return jnp.sum(jnp.square(acc))


def action_rate(action, prev_action):
    return jnp.sum(jnp.square(action - prev_action))


def knee_bend(knee_angles, min_angle: float):
    """Penalise knees straightening below min_angle — forces spring-like preload.
    Only fires on the deficit; zero penalty when both knees are sufficiently bent."""
    deficit = jnp.maximum(0.0, min_angle - knee_angles)
    return jnp.sum(jnp.square(deficit))


def gait_imitation(leg_joint_pos, cpg_ref, cmd, cpg_cfg):
    """Mean-square deviation of leg joints from the CPG reference trajectory.

    Gated by the EFFECTIVE locomotion speed — |v_xy| with commanded yaw rate
    folded in via an equivalent foot-arc radius — so rotate-on-the-spot counts
    as locomotion, while standing episodes incur exactly zero penalty."""
    err = jnp.mean(jnp.square(leg_joint_pos - cpg_ref))
    v_eff = jnp.sqrt(cmd[0] ** 2 + cmd[1] ** 2
                     + jnp.square(cpg_cfg.yaw_equiv_radius * cmd[2]))
    gate = jnp.tanh(cpg_cfg.gate_speed_gain * v_eff)
    return err * gate


def _cmd_speed_gate(cmd, cpg_cfg):
    v_eff = jnp.sqrt(cmd[0] ** 2 + cmd[1] ** 2
                     + jnp.square(cpg_cfg.yaw_equiv_radius * cmd[2]))
    return jnp.tanh(cpg_cfg.gate_speed_gain * v_eff)


def gait_contact_stance(feet_contact, cpg_stance, cmd, cpg_cfg):
    """Foot IN contact during its stance window (clock-based, Cassie-style).
    A stander earns this for free (~stance_duty on average, same as a correct
    walker), so it carries deliberately little weight — it only shapes
    against hover-stepping once a gait exists. Speed-gated like
    gait_imitation.

    SPLIT from the old single `gait_contact` equality-match (teacher_v4 s1
    post-mortem, 2026-08-03): the combined term paid a stander ~stance_duty
    (0.6) per step for free, so raising its weight raised drift income in
    lockstep with walker income and could never break the drift optimum."""
    match = jnp.mean(jnp.where(cpg_stance, feet_contact.astype(jnp.float32), 0.0))
    return match * _cmd_speed_gate(cmd, cpg_cfg)


def gait_contact_swing(feet_contact, cpg_stance, cmd, cpg_cfg):
    """Foot OUT of contact during its swing window — the half of the periodic
    contact reward that a stander CANNOT earn (a planted foot in its swing
    window scores exactly 0). Dense, pays every step; this is the term that
    makes lifting a foot at the right time immediately profitable, before
    tracking improves. A correct walker averages ~(1 - stance_duty) = 0.4;
    a stander exactly 0. Speed-gated like gait_imitation."""
    lifted = jnp.mean(jnp.where(cpg_stance, 0.0,
                                1.0 - feet_contact.astype(jnp.float32)))
    return lifted * _cmd_speed_gate(cmd, cpg_cfg)


def compute_reward(x: RewardInputs, cfg) -> tuple[jnp.ndarray, dict]:
    """Weighted sum of all terms. Returns (total, per-term dict of the RAW
    unweighted values) — the raw tracking kernel is also the curriculum gate
    metric."""
    w = cfg.weights
    k = cfg.kernels
    standing = is_standing(x.cmd, cfg.standing_cmd_threshold)

    terms = {
        "tracking_lin_vel": tracking_lin_vel(x.cmd, x.lin_vel_local, k.tracking_sigma),
        "tracking_yaw": tracking_yaw(x.cmd, x.ang_vel_local, k.tracking_yaw_sigma),
        "tracking_height": tracking_height(x.cmd, x.base_height, k.tracking_height_sigma),
        "tracking_reach": tracking_reach(x.cmd, x.hand_height,
                                         k.tracking_reach_sigma),
        "orientation": orientation(
            x.proj_gravity,
            getattr(cfg, "orientation_free_forward_pitch", False)),
        "ang_vel_roll": ang_vel_roll(x.ang_vel_local),
        "ang_vel_pitch": ang_vel_pitch(x.ang_vel_local),
        "lin_vel_z": lin_vel_z(x.lin_vel_local),
        "feet_air_time": feet_air_time(x.feet_air_time, x.first_contact, standing, cfg.air_time_target),
        "feet_stance": feet_stance(x.feet_contact, standing),
        "feet_slip": feet_slip(x.feet_contact, x.feet_vel_xy),
        "feet_impact": feet_impact(x.first_contact, x.feet_vel_z),
        "com_support": com_support(x.com_xy, x.support_centroid_xy, x.feet_contact, standing),
        # Balance-informed terms (2026-08-07). Both are zero unless the robot
        # is standing on both feet, matching com_support's gating: a margin is
        # only meaningful over a real support polygon.
        "com_margin": jnp.where(
            standing & jnp.all(x.feet_contact),
            margin_deficit(support_margin(x.com_xy, x.foot_corners),
                           cfg.support_margin_target), 0.0),
        "capture_margin": jnp.where(
            standing & jnp.all(x.feet_contact),
            margin_deficit(
                support_margin(capture_point(x.com_xy, x.com_vel_xy,
                                             x.com_height),
                               x.foot_corners),
                cfg.capture_margin_target), 0.0),
        "arm_residual": arm_residual(x.arm_residual),
        "torque": torque(x.torque),
        "joint_acc": joint_acc(x.joint_acc),
        "action_rate": action_rate(x.action, x.prev_action),
        "knee_bend": knee_bend(x.knee_angles, cfg.min_knee_angle),
        "gait_imitation": gait_imitation(x.leg_joint_pos, x.cpg_ref, x.cmd, cfg.cpg),
        "gait_contact_stance": gait_contact_stance(x.feet_contact, x.cpg_stance, x.cmd, cfg.cpg),
        "gait_contact_swing": gait_contact_swing(x.feet_contact, x.cpg_stance, x.cmd, cfg.cpg),
    }
    total = sum(w[name] * value for name, value in terms.items())
    return total, terms
