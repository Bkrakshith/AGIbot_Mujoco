"""Unit tests for reward terms with hand-constructed states (Phase 1 gate)."""

import jax.numpy as jnp
import numpy as np
import pytest

from x1_locomotion import rewards
from x1_locomotion.config import load_config
from x1_locomotion.layout import N_ARM, N_LT, NU  # noqa: E402,F401


@pytest.fixture(scope="module")
def cfg():
    return load_config().rewards


def _inputs(**overrides):
    base = rewards.RewardInputs(
        cmd=jnp.array([0.0, 0.0, 0.0, 0.98]),
        lin_vel_local=jnp.zeros(3),
        ang_vel_local=jnp.zeros(3),
        proj_gravity=jnp.array([0.0, 0.0, -1.0]),
        base_height=jnp.array(0.98),
        hand_height=0.96,
        feet_contact=jnp.array([True, True]),
        first_contact=jnp.array([False, False]),
        feet_air_time=jnp.zeros(2),
        feet_vel_xy=jnp.zeros(2),
        feet_vel_z=jnp.zeros(2),
        com_xy=jnp.zeros(2),
        support_centroid_xy=jnp.zeros(2),
        com_vel_xy=jnp.zeros(2),
        com_height=jnp.array(0.95),
        # two 0.2x0.2 m feet 0.3 m apart, centred on the origin
        foot_corners=jnp.asarray(
            [[ox, sy * 0.15 + oy] for sy in (-1, 1)
             for ox in (-0.1, 0.1) for oy in (-0.1, 0.1)]),
        arm_residual=jnp.zeros(N_ARM),
        torque=jnp.zeros(NU),
        joint_acc=jnp.zeros(NU),
        action=jnp.zeros(NU),
        prev_action=jnp.zeros(NU),
        knee_angles=jnp.array([0.9, 0.9]),  # bent knees, no penalty
        leg_joint_pos=jnp.zeros(12),
        cpg_ref=jnp.zeros(12),              # identical → zero gait_imitation error
        cpg_stance=jnp.array([True, True]), # matches both-feet-planted default
    )
    return base._replace(**overrides)


def test_perfect_standstill_maximises_tracking(cfg):
    total, terms = rewards.compute_reward(_inputs(), cfg)
    assert float(terms["tracking_lin_vel"]) == pytest.approx(1.0)
    assert float(terms["tracking_yaw"]) == pytest.approx(1.0)
    assert float(terms["tracking_height"]) == pytest.approx(1.0)
    assert float(terms["feet_stance"]) == 1.0  # both planted, standing cmd
    assert float(total) > 0


def test_velocity_error_decays_kernel(cfg):
    x = _inputs(cmd=jnp.array([0.5, 0.0, 0.0, 0.98]))
    _, terms = rewards.compute_reward(x, cfg)
    expected = np.exp(-0.25 / cfg.kernels.tracking_sigma)
    assert float(terms["tracking_lin_vel"]) == pytest.approx(expected, rel=1e-5)
    # walking command => no stance bonus
    assert float(terms["feet_stance"]) == 0.0


def test_orientation_and_angvel_penalties(cfg):
    x = _inputs(proj_gravity=jnp.array([0.3, -0.2, -0.93]),
                ang_vel_local=jnp.array([1.0, -2.0, 0.0]))
    _, terms = rewards.compute_reward(x, cfg)
    assert float(terms["orientation"]) == pytest.approx(0.3**2 + 0.2**2, rel=1e-5)
    assert float(terms["ang_vel_roll"]) == pytest.approx(1.0, rel=1e-5)
    assert float(terms["ang_vel_pitch"]) == pytest.approx(4.0, rel=1e-5)
    assert cfg.weights.orientation < 0
    assert cfg.weights.ang_vel_roll < 0 and cfg.weights.ang_vel_pitch < 0


def _pg(pitch_deg=0.0, roll_deg=0.0):
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    return jnp.array([np.sin(p), -np.cos(p) * np.sin(r), -np.cos(p) * np.cos(r)])


def test_orientation_is_symmetric_in_pitch_by_default():
    """Locomotion behaviour: forward and backward pitch cost the same."""
    for d in (10, 30, 54):
        assert float(rewards.orientation(_pg(d))) == pytest.approx(
            float(rewards.orientation(_pg(-d))), rel=1e-5)


def test_free_forward_pitch_exempts_only_forward():
    """Backward pitch and roll must keep their penalty — nothing else
    discourages them, and backward is where the robot topples in a crouch."""
    for d in (20, 36, 54, 70):
        assert float(rewards.orientation(_pg(d), True)) == pytest.approx(0.0, abs=1e-7)
        assert float(rewards.orientation(_pg(-d), True)) > 0.01
        assert float(rewards.orientation(_pg(roll_deg=d), True)) > 0.01
        assert float(rewards.orientation(_pg(roll_deg=-d), True)) > 0.01


def test_free_forward_pitch_removes_the_structural_under_reach():
    """THE POINT. With a penalty on forward pitch the best posture is always
    SHORT of the hand target: at the posture that hits it the reach kernel is
    at its peak (zero gradient) while the penalty's gradient is not. Measured:
    weight -3.0 settled at 23 deg, -1.5 at 36 deg, both short of the ~57 deg
    the 0.50 m reach needs. Exempting forward pitch must move the argmax onto
    the target itself."""
    L, cmd, sigma, w = 0.461, 0.50, 0.10, -1.5
    pitches = np.arange(0.0, 75.0, 0.5)

    def best(free):
        hand = 0.961 - 0.02 - L * np.sin(np.radians(pitches))
        reach = 4.0 * np.exp(-(hand - cmd) ** 2 / sigma)
        pen = np.array([float(rewards.orientation(_pg(p), free)) for p in pitches])
        return pitches[np.argmax(reach + w * pen)]

    on_target = pitches[np.argmin(np.abs(0.941 - L * np.sin(np.radians(pitches)) - cmd))]
    assert best(False) < on_target - 10.0, "penalised: must under-reach"
    assert best(True) == pytest.approx(on_target, abs=1.0), "exempt: must hit it"


def test_air_time_only_rewarded_when_moving(cfg):
    below_clip = cfg.air_time_target * 0.8   # stay under the clip on purpose
    stand = _inputs(first_contact=jnp.array([True, False]),
                    feet_air_time=jnp.array([below_clip, 0.0]))
    _, terms = rewards.compute_reward(stand, cfg)
    assert float(terms["feet_air_time"]) == 0.0
    walk = stand._replace(cmd=jnp.array([0.5, 0.0, 0.0, 0.98]))
    _, terms = rewards.compute_reward(walk, cfg)
    assert float(terms["feet_air_time"]) == pytest.approx(below_clip, rel=1e-5)


def test_air_time_clipped_at_target(cfg):
    x = _inputs(cmd=jnp.array([0.5, 0.0, 0.0, 0.98]),
                first_contact=jnp.array([True, False]),
                feet_air_time=jnp.array([5.0, 0.0]))
    _, terms = rewards.compute_reward(x, cfg)
    assert float(terms["feet_air_time"]) == pytest.approx(cfg.air_time_target)


def test_slip_penalty_needs_contact(cfg):
    moving_air = _inputs(feet_contact=jnp.array([False, False]),
                         feet_vel_xy=jnp.array([1.0, 1.0]))
    _, terms = rewards.compute_reward(moving_air, cfg)
    assert float(terms["feet_slip"]) == 0.0
    moving_contact = _inputs(feet_vel_xy=jnp.array([1.0, 0.0]))
    _, terms = rewards.compute_reward(moving_contact, cfg)
    assert float(terms["feet_slip"]) == pytest.approx(1.0)


def test_com_support_only_standing_both_feet(cfg):
    off = _inputs(com_xy=jnp.array([0.1, 0.0]))
    _, terms = rewards.compute_reward(off, cfg)
    assert float(terms["com_support"]) == pytest.approx(0.01, rel=1e-5)
    one_foot = off._replace(feet_contact=jnp.array([True, False]))
    _, terms = rewards.compute_reward(one_foot, cfg)
    assert float(terms["com_support"]) == 0.0
    walking = off._replace(cmd=jnp.array([0.5, 0.0, 0.0, 0.98]))
    _, terms = rewards.compute_reward(walking, cfg)
    assert float(terms["com_support"]) == 0.0


def test_effort_terms(cfg):
    x = _inputs(arm_residual=jnp.full(N_ARM, 0.1), torque=jnp.full(NU, 10.0),
                action=jnp.full(NU, 0.5), prev_action=jnp.zeros(NU))
    _, terms = rewards.compute_reward(x, cfg)
    assert float(terms["arm_residual"]) == pytest.approx(N_ARM * 0.01, rel=1e-5)
    assert float(terms["torque"]) == pytest.approx(NU * 100.0, rel=1e-5)
    assert float(terms["action_rate"]) == pytest.approx(NU * 0.25, rel=1e-5)


def test_all_weights_present(cfg):
    _, terms = rewards.compute_reward(_inputs(), cfg)
    for name in terms:
        assert name in cfg.weights, f"missing weight for {name}"


# ---- balance-informed margins (2026-08-07) --------------------------------

def _square_feet(half=0.1, sep=0.3, cx=0.0, cy=0.0):
    """Two axis-aligned foot boxes -> the 8 support-polygon corners."""
    out = []
    for sy in (-1, 1):
        for ox in (-half, half):
            for oy in (-half, half):
                out.append([cx + ox, cy + sy * sep / 2 + oy])
    return jnp.asarray(out)


def test_support_margin_is_positive_inside_and_negative_outside():
    c = _square_feet(half=0.1, sep=0.3)
    # polygon spans x in [-0.1, 0.1], y in [-0.25, 0.25]
    assert float(rewards.support_margin(jnp.array([0.0, 0.0]), c)) == \
        pytest.approx(0.1)
    assert float(rewards.support_margin(jnp.array([0.05, 0.0]), c)) == \
        pytest.approx(0.05)
    assert float(rewards.support_margin(jnp.array([0.15, 0.0]), c)) < 0.0
    assert float(rewards.support_margin(jnp.array([0.0, 0.30]), c)) < 0.0


def test_support_margin_ignores_where_inside_the_foot_the_com_sits():
    """THE POINT vs com_support: shifting the CoM back within the footprint
    is what a deep forward reach requires, and must not be penalised until it
    actually approaches the edge."""
    c = _square_feet(half=0.1, sep=0.3)
    near_edge = float(rewards.support_margin(jnp.array([0.08, 0.0]), c))
    centre = float(rewards.support_margin(jnp.array([0.0, 0.0]), c))
    assert centre > near_edge
    # but the DEFICIT is zero for both while they clear the target
    assert float(rewards.margin_deficit(centre, 0.02)) == 0.0
    assert float(rewards.margin_deficit(near_edge, 0.02)) == 0.0
    # and only bites near the edge
    assert float(rewards.margin_deficit(near_edge, 0.06)) > 0.0


def test_capture_point_shifts_with_velocity():
    com = jnp.array([0.0, 0.0])
    still = rewards.capture_point(com, jnp.zeros(2), jnp.float32(1.0))
    assert np.allclose(np.asarray(still), [0.0, 0.0])
    moving = rewards.capture_point(com, jnp.array([1.0, 0.0]), jnp.float32(0.98))
    assert float(moving[0]) == pytest.approx(np.sqrt(0.98 / 9.81), rel=1e-4)
    # a lower CoM is easier to catch: the capture point moves less
    low = rewards.capture_point(com, jnp.array([1.0, 0.0]), jnp.float32(0.5))
    assert float(low[0]) < float(moving[0])


def test_capture_point_catches_what_a_static_margin_cannot():
    """The loaded-descent failure: the CoM is inside the polygon but moving,
    and arrives at the bottom with momentum it cannot arrest. A static margin
    reports it as safe; the capture point does not."""
    c = _square_feet(half=0.1, sep=0.3)
    com = jnp.array([0.05, 0.0])
    assert float(rewards.support_margin(com, c)) > 0.0, "statically inside"
    xi = rewards.capture_point(com, jnp.array([0.8, 0.0]), jnp.float32(0.9))
    assert float(rewards.support_margin(xi, c)) < 0.0, "dynamically lost"


def test_margin_terms_are_off_for_locomotion(cfg):
    """The locomotion teacher's reward must be bit-identical."""
    assert cfg.weights.com_margin == 0.0
    assert cfg.weights.capture_margin == 0.0
