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
    return s


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
