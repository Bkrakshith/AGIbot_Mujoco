"""Phase 1 gates: env resets/steps, zero-action stand under PD, per-episode DR
verifiably rewrites model fields, payload swap fires and shows in the mass
trace, obs layout parity with the CPU eval mirror."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from x1_locomotion.config import load_config
from x1_locomotion.env import (PRIV_OBS_SIZE, STUDENT_OBS_SIZE, X1LocomotionEnv)
from x1_locomotion.layout import N_ACT, N_ARM, N_LT, NU  # noqa: E402,F401


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env0(cfg):
    return X1LocomotionEnv(cfg, cfg.stages[0])


@pytest.fixture(scope="module")
def env_full(cfg):
    return X1LocomotionEnv(cfg, cfg.stages[-1])


def test_reset_step_shapes(env0):
    state = jax.jit(env0.reset)(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (STUDENT_OBS_SIZE,)
    assert state.obs["privileged_state"].shape == (PRIV_OBS_SIZE,)
    state = jax.jit(env0.step)(state, jnp.zeros(N_ACT))
    assert jnp.all(jnp.isfinite(state.obs["state"]))
    assert jnp.all(jnp.isfinite(state.reward))


def test_zero_action_stands_briefly(env0):
    """Pure PD hold on defaults must keep the robot up >= 1 s. (It WILL fall
    eventually: ankle stiffness << mgh for a 36 kg humanoid — active balance
    is the policy's job.)"""
    state = jax.jit(env0.reset)(jax.random.PRNGKey(3))
    step = jax.jit(env0.step)
    steps_alive = 0
    for _ in range(75):  # 1.5 s
        state = step(state, jnp.zeros(N_ACT))
        if float(state.done):
            break
        steps_alive += 1
    assert steps_alive >= 50, f"fell after {steps_alive} steps (<1 s)"


def test_per_episode_dr_changes_model_fields(env_full):
    reset = jax.jit(env_full.reset)
    s1, s2 = reset(jax.random.PRNGKey(1)), reset(jax.random.PRNGKey(2))
    f1, f2 = s1.info["model_fields"], s2.info["model_fields"]
    assert bool(jnp.any(f1["body_mass"] != f2["body_mass"]))
    assert bool(jnp.any(f1["geom_friction"] != f2["geom_friction"]))
    assert bool(jnp.any(f1["dof_damping"] != f2["dof_damping"]))


def test_payload_within_stage_bounds(cfg, env_full):
    for seed in range(20):
        s = jax.jit(env_full.reset)(jax.random.PRNGKey(seed))
        total = float(jnp.sum(s.info["payload_mass"]))
        assert 0.0 <= total <= cfg.stages[-1].payload_max_total + 1e-6


def test_stage1_payload_zero_and_symmetric(cfg):
    env = X1LocomotionEnv(cfg, cfg.stages[0])
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert float(jnp.sum(s.info["payload_mass"])) == pytest.approx(0.0, abs=1e-6)
    assert int(s.info["swap_step"]) == np.iinfo(np.int32).max  # swaps disabled
    assert int(s.info["switch_step"]) == np.iinfo(np.int32).max


def test_payload_swap_fires_and_shows_in_mass_trace(env_full):
    """Machinery check (stand/walk stages carry no load; the swap path is
    kept for a future carry task and must still work)."""
    state = jax.jit(env_full.reset)(jax.random.PRNGKey(0))
    post = jnp.array([1.0, 6.0])
    info = dict(state.info)
    info["swap_step"] = jnp.int32(2)
    info["swap_post"] = post
    state = state.replace(info=info)
    step = jax.jit(env_full.step)
    trace = []
    for _ in range(4):
        state = step(state, jnp.zeros(N_ACT))
        trace.append(float(state.metrics["payload_total"]))
    assert trace[0] != pytest.approx(7.0)          # pre-swap
    assert trace[2] == pytest.approx(7.0)          # post-swap: 1 + 6
    assert np.allclose(np.asarray(state.info["payload_mass"]), [1.0, 6.0])


def test_action_latency_buffer(env0):
    state = jax.jit(env0.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env0.step)
    a = jnp.ones(N_ACT) * 0.5
    state = step(state, a)
    buf = state.info["action_buffer"]
    assert jnp.allclose(buf[0], a)
    assert jnp.allclose(buf[1], 0.0)
    assert int(state.info["action_delay"]) in (0, 1, 2)


def test_obs_layout_parity_with_cpu_eval(cfg):
    """The MJX obs builder and the CPU eval mirror must produce identical
    'state' vectors for identical physical states (noise off)."""
    import mujoco
    from x1_locomotion.cpu_eval import CpuRollout

    env = X1LocomotionEnv(cfg, cfg.stages[0])
    env._noise_scale = jnp.zeros_like(env._noise_scale)
    state = jax.jit(env.reset)(jax.random.PRNGKey(7))
    state = jax.jit(env.step)(state, jnp.full(N_ACT, 0.1))

    runner = CpuRollout(cfg)  # full-fidelity model, same layout
    d = mujoco.MjData(runner.model)
    d.qpos[:] = np.asarray(state.pipeline_state.qpos)
    d.qvel[:] = np.asarray(state.pipeline_state.qvel)
    mujoco.mj_forward(runner.model, d)
    cmd = np.asarray(jnp.where(state.info["step"] >= state.info["switch_step"],
                               state.info["cmd2"], state.info["cmd"]))
    cpu_obs = runner.build_state_obs(
        d, np.asarray(state.info["last_action"], dtype=np.float32),
        cmd, np.asarray(state.info["arm_cmd"]),
        float(state.info["gait_phase"]))
    np.testing.assert_allclose(np.asarray(state.obs["state"]), cpu_obs,
                               atol=1e-5)


def test_reward_metric_keys_match_the_terms_exactly():
    """THE BUG THIS GUARDS (2026-08-07): _reward_keys was a hardcoded tuple.
    Adding com_margin/capture_margin without extending it crashed training
    with 'scan body function carry input and carry output must have the same
    pytree structure' — reset() pre-creates one metric per key, step() emits
    one per TERM, and brax carries metrics through a scan. Unit tests missed
    it because they call compute_reward directly.

    """
    import jax
    from x1_locomotion import rewards
    from x1_locomotion.config import load_config
    from x1_locomotion.env import X1LocomotionEnv
    loco = load_config()
    for env in (X1LocomotionEnv(loco, loco.stages[0]),):
        st = env.reset(jax.random.PRNGKey(0))
        st = env.step(st, jnp.zeros(N_ACT))
        from_reset = {k for k in st.metrics if k.startswith("reward/")}
        expected = {f"reward/{k}" for k in env._reward_keys()}
        assert from_reset == expected
        # and every key must be a term compute_reward actually produces
        _, terms = rewards.compute_reward(
            _reward_inputs_probe(env, st), env._cfg.rewards)
        assert set(env._reward_keys()) == set(terms), (
            set(env._reward_keys()) ^ set(terms))


def _reward_inputs_probe(env, state):
    """Rebuild RewardInputs the way env.step does, for the key check."""
    import jax.numpy as jnp2
    d = state.pipeline_state
    from x1_locomotion import rewards
    return rewards.RewardInputs(
        cmd=state.info["cmd"], lin_vel_local=jnp2.zeros(3),
        ang_vel_local=jnp2.zeros(3),
        proj_gravity=jnp2.array([0.0, 0.0, -1.0]),
        base_height=d.qpos[2], hand_height=jnp2.float32(0.9),
        feet_contact=jnp2.array([True, True]),
        first_contact=jnp2.array([False, False]),
        feet_air_time=jnp2.zeros(2), feet_vel_xy=jnp2.zeros(2),
        feet_vel_z=jnp2.zeros(2),
        com_xy=d.subtree_com[env._pelvis_id][:2],
        support_centroid_xy=jnp2.zeros(2),
        com_vel_xy=jnp2.zeros(2),
        com_height=d.subtree_com[env._pelvis_id][2],
        foot_corners=env._foot_corners(d),
        arm_residual=jnp2.zeros(N_ARM), torque=jnp2.zeros(N_LT),
        joint_acc=jnp2.zeros(N_LT), base_acc=jnp2.zeros(3), action=jnp2.zeros(N_ACT),
        prev_action=jnp2.zeros(N_ACT), knee_angles=jnp2.array([0.9, 0.9]),
        leg_joint_pos=jnp2.zeros(12), cpg_ref=jnp2.zeros(12),  # 2 legs x 6
        cpg_stance=jnp2.array([True, True]))


def test_privileged_obs_carries_every_randomised_latent(cfg):
    """RMA: the teacher must see the DR factors it is randomised over
    (motor strength, latency, base mass/CoM), or it can only hedge."""
    env = X1LocomotionEnv(cfg, cfg.stages[-1])
    for seed in range(3):
        s = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        p = np.asarray(s.obs["privileged_state"])
        assert p.shape == (23,)
        np.testing.assert_allclose(p[17], float(s.info["motor_scale"]), rtol=1e-6)
        np.testing.assert_allclose(p[18], int(s.info["action_delay"]) / 4.0, rtol=1e-6)
        np.testing.assert_allclose(p[19], float(s.info["base_mass_delta"]) / 3.0, rtol=1e-5)
        np.testing.assert_allclose(p[20:23], np.asarray(s.info["base_com_shift"]) / 0.05, rtol=1e-5)


def test_teacher_is_noise_free_student_is_not(cfg):
    """teacher_obs_noise scales the teacher env's sensor noise; history
    (student) envs always carry the full X1 noise."""
    from x1_locomotion.config import _deep_merge
    c = load_config()
    c.train = _deep_merge(c.train, {"teacher_obs_noise": 0.0})
    teacher = X1LocomotionEnv(c, c.stages[-1])
    student = X1LocomotionEnv(c, c.stages[-1], include_history=True)
    assert float(jnp.max(teacher._noise_scale)) == 0.0
    assert float(jnp.max(student._noise_scale)) > 1.0      # joint_vel 1.3 rad/s


def test_clean_state_for_distillation(cfg):
    """student_noise + emit_clean_state: 'state' carries the full noise,
    'state_clean' is the same vector without it (identical tail)."""
    from x1_locomotion.env import _NOISY_PREFIX
    env = X1LocomotionEnv(cfg, cfg.stages[-1], student_noise=True, emit_clean_state=True)
    s = jax.jit(env.reset)(jax.random.PRNGKey(5))
    noisy, clean = np.asarray(s.obs["state"]), np.asarray(s.obs["state_clean"])
    assert noisy.shape == clean.shape == (STUDENT_OBS_SIZE,)
    np.testing.assert_allclose(noisy[_NOISY_PREFIX:], clean[_NOISY_PREFIX:])
    assert np.abs(noisy[:_NOISY_PREFIX] - clean[:_NOISY_PREFIX]).max() > 0.1


def test_cpu_eval_obs_noise_switch(cfg):
    import mujoco
    from x1_locomotion.cpu_eval import CpuRollout
    quiet, noisy = CpuRollout(cfg), CpuRollout(cfg, obs_noise=True)
    d = mujoco.MjData(quiet.model)
    mujoco.mj_resetDataKeyframe(quiet.model, d, 0)
    mujoco.mj_forward(quiet.model, d)
    args = (d, np.zeros(N_ACT, np.float32), np.zeros(4), quiet.nominal_carry, 0.0)
    a = quiet.build_state_obs(*args, rng=np.random.default_rng(0))
    b = noisy.build_state_obs(*args, rng=np.random.default_rng(0))
    assert np.array_equal(a, quiet.build_state_obs(*args))      # off by default
    assert np.abs(a - b).max() > 0.1


def test_privileged_push_can_be_hidden(cfg):
    """train.privileged_push false: the push slots of the privileged obs are 0
    even while a push is active (slot kept, so obs sizes do not change)."""
    import dataclasses
    from x1_locomotion.config import _deep_merge
    c = load_config()
    c.train = _deep_merge(c.train, {"privileged_push": False})
    st = dataclasses.replace(c.stages[-1], push_max_n=60.0)
    env = X1LocomotionEnv(c, st)
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    info = dict(s.info)
    info["push_start"] = jnp.int32(0)
    info["push_force"] = jnp.array([50.0, 0.0, 0.0])
    s = s.replace(info=info)
    s = jax.jit(env.step)(s, jnp.zeros(N_ACT))
    assert float(jnp.abs(s.info["push_force_active"]).max()) > 0     # push applied
    np.testing.assert_allclose(np.asarray(s.obs["privileged_state"])[12:15], 0.0)  # push slots
