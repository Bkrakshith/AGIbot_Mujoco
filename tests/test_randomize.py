"""DR sampling ranges and event logic (configs/domain_rand.yaml)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from x1_locomotion import randomize
from x1_locomotion.config import load_config
from x1_locomotion.env import X1LocomotionEnv
from x1_locomotion.layout import N_ARM, N_LT, NU  # noqa: E402,F401


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def base(cfg):
    return X1LocomotionEnv(cfg, cfg.stages[0])._base


def _dr_with_payload(cfg, hi=7.0):
    """The stand/walk task samples 0 kg; exercise the sampler on a real range."""
    from x1_locomotion.config import _deep_merge
    return _deep_merge(cfg.domain_rand, {"payload": {"total_mass_range": [0.0, hi]}})


def test_payload_symmetric_split(cfg):
    for seed in range(10):
        m, _ = randomize.sample_payload(jax.random.PRNGKey(seed),
                                        _dr_with_payload(cfg), 6.0, True)
        assert float(m[0]) == pytest.approx(float(m[1]))


def test_payload_asymmetric_covers_range(cfg):
    fracs = []
    for seed in range(200):
        m, com = randomize.sample_payload(jax.random.PRNGKey(seed),
                                          _dr_with_payload(cfg), 7.0, False)
        total = float(m[0] + m[1])
        assert 0.0 <= total <= 7.0
        assert np.all(np.abs(np.asarray(com)) <= 0.10 + 1e-6)
        if total > 0.5:
            fracs.append(float(m[0]) / total)
    assert min(fracs) < 0.1 and max(fracs) > 0.9  # fully asymmetric covered


def test_model_field_ranges(cfg, base):
    dr = cfg.domain_rand
    for seed in range(5):
        fields, priv = randomize.sample_model_fields(
            jax.random.PRNGKey(seed), base, dr)
        ratio = np.asarray(fields["body_mass"] / np.maximum(base.body_mass, 1e-9))
        ratio = np.delete(ratio, base.pelvis_id)  # pelvis gets an extra delta
        ok = (ratio > 0.9 - 1e-6) & (ratio < 1.1 + 1e-6)
        assert ok[1:].all()  # world body 0 has zero mass
        assert dr.ground.friction_range[0] <= float(priv["friction"]) \
            <= dr.ground.friction_range[1]
        assert 0 <= int(priv["action_delay"]) <= 2
        assert dr.actuation.pd_gain_scale_range[0] <= float(priv["kp_scale"]) \
            <= dr.actuation.pd_gain_scale_range[1]


def test_swap_disabled_gives_sentinel(cfg):
    step, post = randomize.sample_swap(jax.random.PRNGKey(0), cfg.domain_rand,
                                       jnp.array([2.0, 3.0]), 7.0, False, 0.02)
    assert int(step) == np.iinfo(np.int32).max


def test_swap_respects_total_budget(cfg):
    masses = jnp.array([2.0, 3.0])
    for seed in range(50):
        step, post = randomize.sample_swap(jax.random.PRNGKey(seed),
                                           cfg.domain_rand, masses, 7.0, True, 0.02)
        assert float(jnp.sum(post)) <= 7.0 + 1e-5


def test_command_modes(cfg):
    cmds = cfg.train.commands
    stand = walk = 0
    for seed in range(200):
        cmd, cmd2, sw = randomize.sample_command(jax.random.PRNGKey(seed),
                                                 cmds, False, 0.02)
        assert int(sw) == np.iinfo(np.int32).max  # switching disabled
        assert cmds.height_range[0] - 1e-6 <= float(cmd[3]) <= cmds.height_range[1] + 1e-6
        if float(jnp.linalg.norm(cmd[:3])) < 1e-6:
            stand += 1
        else:
            walk += 1
            assert abs(float(cmd[0])) <= cmds.vx_range[1] + 1e-6
    # 35 % stand / 65 % walk when switching is folded into walk
    assert 0.2 < stand / 200 < 0.5


def test_arm_command_scale_zero_is_nominal(cfg, base):
    env = X1LocomotionEnv(cfg, cfg.stages[0])
    nominal = env._nominal_carry
    cmd = randomize.sample_arm_command(jax.random.PRNGKey(0), cfg.domain_rand,
                                       nominal, env._joint_lo[N_LT:],
                                       env._joint_hi[N_LT:], 0.0)
    np.testing.assert_allclose(np.asarray(cmd), np.asarray(nominal), atol=1e-6)


def test_arm_command_respects_limits(cfg):
    env = X1LocomotionEnv(cfg, cfg.stages[-1])
    for seed in range(20):
        cmd = randomize.sample_arm_command(
            jax.random.PRNGKey(seed), cfg.domain_rand, env._nominal_carry,
            env._joint_lo[N_LT:], env._joint_hi[N_LT:], 1.0)
        assert np.all(np.asarray(cmd) >= np.asarray(env._joint_lo[N_LT:]) - 1e-6)
        assert np.all(np.asarray(cmd) <= np.asarray(env._joint_hi[N_LT:]) + 1e-6)


def test_task_samples_no_payload(cfg):
    """Stand/walk: the configured range and every stage are 0 kg."""
    assert list(cfg.domain_rand.payload.total_mass_range) == [0.0, 0.0]
    assert all(s.payload_max_total == 0.0 for s in cfg.stages)
