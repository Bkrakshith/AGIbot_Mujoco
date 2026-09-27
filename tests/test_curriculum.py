"""Curriculum gate logic — the two-criterion gate and the drift loophole.

teacher_v3's s1 passed a 0.78 tracking gate with zero foot lift (drift-leaning
toward small commands). These tests pin the fix: tracking alone must never
pass a stage whose air-time gate is set, because air time cannot be earned
without stepping.
"""

import dataclasses

import pytest

from x1_locomotion import curriculum
from x1_locomotion.config import load_config


@pytest.fixture(scope="module")
def stage():
    # a real stage from config, with known gates
    s = load_config().stages[0]
    assert s.gate_tracking > 0 and s.gate_air_time > 0
    # isolate the tracking/air-time logic; the survival gate has its own test
    return dataclasses.replace(s, gate_survival=0.0)


def _metrics(tracking_per_step, air_per_step, episode_len=1000.0):
    return {
        "eval/episode_reward/tracking_lin_vel": tracking_per_step * episode_len,
        "eval/episode_reward/feet_air_time": air_per_step * episode_len,
        "eval/avg_episode_length": episode_len,
    }


def test_walking_policy_passes(stage):
    passed, trk, air = curriculum.gate_passed(
        stage, _metrics(stage.gate_tracking + 0.05, stage.gate_air_time * 2))
    assert passed and trk > stage.gate_tracking and air > stage.gate_air_time


def test_drift_tracking_fails_air_gate(stage):
    """The teacher_v3 s1 exploit: high tracking, zero air time -> must fail."""
    passed, trk, air = curriculum.gate_passed(
        stage, _metrics(stage.gate_tracking + 0.05, 0.0))
    assert trk > stage.gate_tracking, "tracking criterion alone would pass"
    assert air == 0.0
    assert not passed, "drift-tracking must not clear a stage"


def test_stepping_but_sloppy_fails_tracking_gate(stage):
    """The teacher_v3 s2 end state: real gait, tracking not yet there."""
    passed, trk, air = curriculum.gate_passed(
        stage, _metrics(stage.gate_tracking - 0.05, stage.gate_air_time * 2))
    assert not passed and air > stage.gate_air_time


def test_zero_air_gate_disables_criterion(stage):
    legacy = dataclasses.replace(stage, gate_air_time=0.0)
    passed, _, _ = curriculum.gate_passed(
        legacy, _metrics(stage.gate_tracking + 0.05, 0.0))
    assert passed, "gate_air_time=0 must reduce to the tracking-only gate"


def test_stall_report_names_both_criteria(stage):
    rep = curriculum.stall_report(stage, 0.738, 0.0021)
    assert "0.738" in rep and "0.0021" in rep
    assert "air" in rep.lower()


@pytest.mark.parametrize("idx", range(len(load_config().stages)))
def test_gate_is_halfway_between_floor_and_perfect(idx):
    """Theory §7 / failure M8: a do-nothing policy scores the floor; the gate
    must sit at (floor + 1) / 2. Recompute the floor if commands change."""
    cfg = load_config()
    st = cfg.stages[idx]
    floor = curriculum.do_nothing_floor(cfg, st)
    assert st.gate_tracking > floor, "gate at/below the floor tests nothing"
    assert st.gate_tracking == pytest.approx((floor + 1.0) / 2.0, abs=0.005)


def test_final_stage_trains_the_full_command_envelope():
    """The command curriculum may start slow, but the last stage (and so the
    exported policy and the battery) must see the full envelope."""
    assert load_config().stages[-1].cmd_scale == 1.0
    scales = [s.cmd_scale for s in load_config().stages]
    assert scales == sorted(scales), "command envelope must never shrink"


def test_env_samples_the_stage_envelope():
    import jax
    from x1_locomotion.env import X1LocomotionEnv
    cfg = load_config()
    st = cfg.stages[0]
    env = X1LocomotionEnv(cfg, st)
    hi = st.cmd_scale * cfg.train.commands.vx_range[1]
    for seed in range(20):
        s = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        assert float(s.info["cmd"][0]) <= hi + 1e-6


def test_final_stage_trains_the_full_domain_randomisation():
    """DR may be ramped in, but never shrunk, and the last stage (the one the
    battery and export see) must use the full configs/domain_rand.yaml."""
    scales = [s.dr_scale for s in load_config().stages]
    assert scales[-1] == 1.0 and scales == sorted(scales)


def test_stage_domain_rand_interpolates_to_nominal():
    import dataclasses
    from x1_locomotion.config import stage_domain_rand
    cfg = load_config()
    st = cfg.stages[-1]
    full = stage_domain_rand(cfg.domain_rand, dataclasses.replace(st, dr_scale=1.0))
    assert full.ground.friction_range == pytest.approx(list(cfg.domain_rand.ground.friction_range))
    assert full.actuation.action_latency_steps == list(cfg.domain_rand.actuation.action_latency_steps)
    off = stage_domain_rand(cfg.domain_rand, dataclasses.replace(st, dr_scale=0.0))
    assert off.ground.friction_range == [1.0, 1.0]
    assert off.actuation.motor_strength_range == [1.0, 1.0]
    assert off.actuation.action_latency_steps == [0, 0]
    assert off.obs_noise.joint_vel == 0.0
    half = stage_domain_rand(cfg.domain_rand, dataclasses.replace(st, dr_scale=0.5))
    lo, hi = cfg.domain_rand.ground.friction_range
    assert half.ground.friction_range == pytest.approx([1 + 0.5 * (lo - 1), 1 + 0.5 * (hi - 1)])


def test_stage_passes_on_its_best_eval_not_its_last(tmp_path, stage):
    """Failure L5: PPO degrades late. A stage whose MID-stage eval cleared
    both gates passes, and the best such checkpoint (longest episodes) is
    the one carried forward — even if the final eval fell below the gate."""
    import json, os
    from x1_locomotion.ppo_teacher import best_passing_checkpoint
    rows = [(100, stage.gate_tracking + 0.02, stage.gate_air_time * 2, 900.0),
            (200, stage.gate_tracking + 0.01, stage.gate_air_time * 2, 1500.0),
            (300, stage.gate_tracking - 0.05, 0.0, 1900.0)]   # final: fails
    with open(tmp_path / "evals.jsonl", "w") as fh:
        for step, trk, air, L in rows:
            os.makedirs(tmp_path / f"ckpt_{step}")
            fh.write(json.dumps({"stage": stage.name, "global_step": step,
                                 **_metrics(trk, air, L)}) + "\n")
    ckpt, row = best_passing_checkpoint(str(tmp_path), stage)
    assert ckpt.endswith("ckpt_200") and row["global_step"] == 200
    other = dataclasses.replace(stage, name="another_stage")
    assert best_passing_checkpoint(str(tmp_path), other) == (None, None)


def test_warm_start_unfreezes_never_varied_obs_dims():
    """teacher_v4 s3 bug: the privileged push obs was 0 through s1-s2, its
    std sat at 1e-6, and a 30 N push normalised to 300,000. Warm-starting
    must give such dims unit std and leave varied dims alone."""
    import jax.numpy as jnp
    import numpy as np
    from brax.training.acme import running_statistics
    from x1_locomotion.networks import normalize_obs
    from x1_locomotion.ppo_teacher import unfreeze_constant_dims
    spec = {"privileged_state": jnp.zeros(3)}
    st = running_statistics.init_state(spec)
    batch = {"privileged_state": jnp.stack([jnp.array([0.0, float(i), 0.0])
                                            for i in range(100)])}
    st = running_statistics.update(st, batch)
    assert float(st.std["privileged_state"][0]) < 1e-3          # frozen
    fixed = unfreeze_constant_dims(st)
    std = np.asarray(fixed.std["privileged_state"])
    assert std[0] == 1.0 and std[2] == 1.0
    assert std[1] == float(st.std["privileged_state"][1])       # untouched
    x = normalize_obs({"privileged_state": jnp.array([0.3, 50.0, 0.0])}, fixed)
    assert abs(float(x["privileged_state"][0])) < 1.0
    # and the fix survives the next running update (not reset to 1e-6)
    st2 = running_statistics.update(fixed, {"privileged_state": jnp.zeros((10, 3))})
    assert float(st2.std["privileged_state"][0]) > 0.5


def test_final_stage_uses_full_task_arm_poses():
    scales = [s.arm_range_scale for s in load_config().stages]
    assert scales[-1] == 1.0 and scales == sorted(scales)


def test_survival_gate_blocks_a_falling_policy(stage):
    """A stage with gate_survival fails when episodes end early even if
    tracking and air time pass (the reliability criterion)."""
    st = dataclasses.replace(stage, gate_survival=0.8)
    curriculum.EPISODE_STEPS = 2000
    ok, _, _ = curriculum.gate_passed(st, _metrics(st.gate_tracking + 0.05, st.gate_air_time * 2, 1900.0))
    bad, _, _ = curriculum.gate_passed(st, _metrics(st.gate_tracking + 0.05, st.gate_air_time * 2, 900.0))
    assert ok and not bad


def test_final_stage_has_a_strict_survival_gate():
    assert load_config().stages[-1].gate_survival >= 0.8
