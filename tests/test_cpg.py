"""CPG invariants for the X1 swing reference and the observable gait clock.

Pinned here:
  1. the reference is the vendor X1 shape: stance leg at default, swing leg
     at default + |sin θ|·delta, dead band = double support;
  2. it collapses to the default pose when standing is commanded;
  3. the clock is in the observation (an unobservable phase cannot be
     followed — H1-2 teacher_v2);
  4. stance windows agree with the reference, and the contact rewards pay a
     walker, not a stander (H1-2 failure L4).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from x1_locomotion import cpg, rewards
from x1_locomotion.config import load_config
from x1_locomotion.env import STUDENT_OBS_SIZE, X1LocomotionEnv
from x1_locomotion.layout import N_LEG, NU  # noqa: E402,F401


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def q0(cfg):
    return jnp.asarray(cfg.actuators.default_pose[:N_LEG])


H = 0.613  # X1 stand height command
WALK = jnp.array([0.5, 0.0, 0.0, H])
STAND = jnp.array([0.0, 0.0, 0.0, H])
ROTATE = jnp.array([0.0, 0.0, 0.5, H])
SIDESTEP = jnp.array([0.0, 0.3, 0.0, H])
THETAS = jnp.linspace(0.0, 2 * np.pi, 1000, endpoint=False)


def refs(cfg, q0, cmd):
    return np.asarray(jax.vmap(lambda th: cpg.cpg_reference(th, cmd, cfg.rewards, q0))(THETAS))


def test_matches_the_vendor_reference_at_mid_swing(cfg, q0):
    """θ = 3π/2 (sin = -1): left at full swing, right at default."""
    r = np.asarray(cpg.cpg_reference(jnp.float32(1.5 * np.pi), WALK, cfg.rewards, q0))
    delta = np.asarray(cfg.rewards.cpg.swing_delta)
    np.testing.assert_allclose(r[:6], np.asarray(q0)[:6] + delta[:6], atol=1e-5)
    np.testing.assert_allclose(r[6:], np.asarray(q0)[6:], atol=1e-6)
    r = np.asarray(cpg.cpg_reference(jnp.float32(0.5 * np.pi), WALK, cfg.rewards, q0))
    np.testing.assert_allclose(r[6:], np.asarray(q0)[6:] + delta[6:], atol=1e-5)
    np.testing.assert_allclose(r[:6], np.asarray(q0)[:6], atol=1e-6)


def test_legs_never_swing_together(cfg, q0):
    r = refs(cfg, q0, WALK) - np.asarray(q0)
    left_moving = np.abs(r[:, :6]).max(axis=1) > 0
    right_moving = np.abs(r[:, 6:]).max(axis=1) > 0
    assert not np.any(left_moving & right_moving)
    assert left_moving.any() and right_moving.any()


def test_right_delta_mirrors_left(cfg):
    """Pitch-like knee/ankle-pitch deltas share a sign; the mirrored axes
    (hip pitch/roll/yaw on the X1) flip — as in the vendor config."""
    d = np.asarray(cfg.rewards.cpg.swing_delta)
    np.testing.assert_allclose(d[[3, 4, 5]], d[[9, 10, 11]])
    np.testing.assert_allclose(d[[0, 1, 2]], -d[[6, 7, 8]])


def test_zero_command_reference_is_static_default_pose(cfg, q0):
    r = refs(cfg, q0, STAND)
    np.testing.assert_allclose(r, np.broadcast_to(np.asarray(q0), r.shape), atol=1e-6)


@pytest.mark.parametrize("cmd", [WALK, SIDESTEP, ROTATE], ids=["walk", "side", "rotate"])
def test_every_locomotion_command_gets_swing(cfg, q0, cmd):
    """Sidestep and rotate-in-place must still lift feet (yaw folded in)."""
    r = refs(cfg, q0, cmd)
    knee_span = r[:, 3].max() - r[:, 3].min()
    assert knee_span > 0.2


def test_gait_imitation_gate_covers_the_envelope(cfg, q0):
    """Gate must be 0 standing, >0 for walk, sidestep AND pure rotation."""
    ref = cpg.cpg_reference(jnp.float32(0.3), WALK, cfg.rewards, q0)
    off = ref + 0.3
    g = lambda cmd: float(rewards.gait_imitation(off, ref, cmd, cfg.rewards.cpg))
    assert g(STAND) == 0.0
    assert g(WALK) > 0.01
    assert g(SIDESTEP) > 0.01
    assert g(ROTATE) > 0.01, "rotation must count as locomotion in the gate"


def test_clock_is_in_the_observation(cfg):
    """The last two obs dims are (sin θ, cos θ) of info['gait_phase']."""
    env = X1LocomotionEnv(cfg, cfg.stages[0])
    state = jax.jit(env.reset)(jax.random.PRNGKey(11))
    assert state.obs["state"].shape == (STUDENT_OBS_SIZE,)
    theta = float(state.info["gait_phase"])
    np.testing.assert_allclose(np.asarray(state.obs["state"])[-2:],
                               [np.sin(theta), np.cos(theta)], atol=1e-5)

    state2 = jax.jit(env.step)(state, jnp.zeros(NU))
    theta2 = float(state2.info["gait_phase"])
    np.testing.assert_allclose(np.asarray(state2.obs["state"])[-2:],
                               [np.sin(theta2), np.cos(theta2)], atol=1e-5)
    assert theta2 - theta == pytest.approx(
        2 * np.pi * env.dt / cfg.rewards.cpg.cycle_time_s, abs=1e-5)


def test_stance_windows_agree_with_reference_and_double_support(cfg, q0):
    """A foot is in swing exactly when its reference deviates from default;
    the dead band gives 2·asin(db)/π double support and no flight phase."""
    db = cfg.rewards.cpg.dead_band
    w = np.asarray(jax.vmap(lambda th: cpg.stance_windows(th, cfg.rewards))(THETAS))
    r = refs(cfg, q0, WALK) - np.asarray(q0)
    np.testing.assert_array_equal(~w[:, 0], np.abs(r[:, :6]).max(axis=1) > 0)
    np.testing.assert_array_equal(~w[:, 1], np.abs(r[:, 6:]).max(axis=1) > 0)
    duty = 0.5 + np.arcsin(db) / np.pi
    assert abs(w[:, 0].mean() - duty) < 0.01 and abs(w[:, 1].mean() - duty) < 0.01
    both = (w[:, 0] & w[:, 1]).mean()
    assert abs(both - 2 * np.arcsin(db) / np.pi) < 0.01
    assert ((~w[:, 0]) & (~w[:, 1])).mean() == 0.0


def test_air_time_target_is_the_swing_duration(cfg):
    c = cfg.rewards.cpg
    swing = c.cycle_time_s * (np.pi - 2 * np.arcsin(c.dead_band)) / (2 * np.pi)
    assert cfg.rewards.air_time_target == pytest.approx(swing, abs=0.005)


def test_gait_contact_rewards_correct_alternation(cfg):
    """Swing half: only actual foot lifts in the swing window pay — a stander
    earns exactly zero. Stance half pays walker and stander alike. Weighted
    per configs/rewards.yaml: walker > stander > anti-phase."""
    ccfg = cfg.rewards.cpg
    w = cfg.rewards.weights
    thetas = jnp.linspace(0.0, 2 * np.pi, 200, endpoint=False)

    def mean_reward(term_fn, contact_fn):
        vals = [float(term_fn(
            contact_fn(cpg.stance_windows(th, cfg.rewards)),
            cpg.stance_windows(th, cfg.rewards), WALK, ccfg)) for th in thetas]
        return np.mean(vals)

    cases = {
        "walker": lambda stance: stance,
        "stander": lambda stance: jnp.array([True, True]),
        "inverse": lambda stance: ~stance,
    }
    swing = {k: mean_reward(rewards.gait_contact_swing, f) for k, f in cases.items()}
    stance = {k: mean_reward(rewards.gait_contact_stance, f) for k, f in cases.items()}

    gate = np.tanh(ccfg.gate_speed_gain * 0.5)
    duty = 0.5 + np.arcsin(ccfg.dead_band) / np.pi
    assert swing["stander"] == 0.0 and swing["inverse"] == 0.0
    assert swing["walker"] == pytest.approx((1 - duty) * gate, abs=2e-2)
    assert stance["walker"] == pytest.approx(duty * gate, abs=2e-2)
    assert stance["stander"] == pytest.approx(duty * gate, abs=2e-2)
    assert stance["inverse"] == 0.0
    total = {k: w["gait_contact_swing"] * swing[k]
             + w["gait_contact_stance"] * stance[k] for k in cases}
    assert total["walker"] > total["stander"] > total["inverse"]
    for term in (rewards.gait_contact_swing, rewards.gait_contact_stance):
        z = term(jnp.array([True, False]), jnp.array([False, True]),
                 jnp.zeros(4), ccfg)
        assert float(z) == 0.0


def test_reference_stays_within_action_authority(cfg, q0):
    """At full swing the reference must be reachable: |ref - q0| <= action
    scale (target = q0 + a·s, |a| <= 1)."""
    scale = np.asarray(cfg.actuators.action_scale_legs_torso[:N_LEG])
    r = refs(cfg, q0, WALK)
    assert np.all(np.abs(r - np.asarray(q0)) <= scale + 1e-6)
