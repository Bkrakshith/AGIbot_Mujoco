"""Sidecar contract (export_onnx.build_sidecar) for the X1 stand/walk policy.

The sidecar is the ONLY machine-readable contract with the deployment side.
These tests pin the obs layout, action split and published envelopes, because
getting them wrong is silent: deployment would feed the policy a vector it
has never seen and the failure would look like a policy bug.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "export"))

from export_onnx import (  # noqa: E402
    _command_envelope, _payload_assumptions, build_sidecar,
)
from x1_locomotion.config import load_config  # noqa: E402
from x1_locomotion.env import STUDENT_OBS_SIZE  # noqa: E402
from x1_locomotion.layout import N_ARM, N_LT, NU  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def sidecar(cfg):
    meta = {"history_length": cfg.train.networks.adaptation.history_length,
            "obs_mean": [0.0] * STUDENT_OBS_SIZE,
            "obs_std": [1.0] * STUDENT_OBS_SIZE}
    return build_sidecar(cfg, meta)


def test_obs_layout_is_contiguous_and_complete(sidecar):
    slices = [e["slice"] for e in sidecar["observation_layout"]]
    assert slices[0][0] == 0
    for (_, end), (start, _) in zip(slices, slices[1:]):
        assert end == start
    assert slices[-1][1] == STUDENT_OBS_SIZE == 48


def test_action_split_and_joint_order(sidecar, cfg):
    sc = sidecar["action_scaling"]
    assert sidecar["policy_output"]["shape"] == [1, NU]
    assert sc["legs_torso"]["indices"] == [0, N_LT]
    assert sc["arms"]["indices"] == [N_LT, NU]
    assert len(sc["joint_order"]) == NU
    assert sc["joint_order"][:6] == [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
        "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll"]
    assert len(sidecar["default_joint_angles"]) == NU
    assert len(sidecar["nominal_carry_pose"]) == N_ARM


def test_gait_clock_is_published(sidecar, cfg):
    assert sidecar["gait_clock_hz"] == pytest.approx(1.0 / cfg.rewards.cpg.cycle_time_s)
    assert sidecar["control_rate_hz"] == pytest.approx(100.0)
    assert sidecar["observation_layout"][-1]["name"] == "gait_clock"


def test_envelopes_come_from_config(cfg):
    env = _command_envelope(cfg)
    c = cfg.train.commands
    assert env["vx_range"] == list(c.vx_range)
    assert env["base_height_range"] == list(c.height_range)
    assert _payload_assumptions(cfg)["total_kg"] == [0.0, 0.0]


def test_sidecar_is_json_serialisable(sidecar):
    json.dumps(sidecar)


def test_command_interpolation_ramps_and_holds():
    """Scheduled commands (stop-and-go) ramp between waypoints."""
    from x1_locomotion.cpu_eval import _interp_cmd
    sched = [(3.0, [0, 0, 0, 0.613]), (3.5, [0.5, 0, 0, 0.613])]
    before = np.array([0, 0, 0, 0.613])
    f = lambda t: _interp_cmd(sched, t, before)[0]
    assert f(0.0) == pytest.approx(0.0)
    assert f(3.25) == pytest.approx(0.25)
    assert f(10.0) == pytest.approx(0.5)


def test_battery_stays_inside_the_training_envelope(cfg):
    """Failure M2: every battery command lies inside the training ranges."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
    from run_battery import battery
    c = cfg.train.commands
    for sc in battery(cfg):
        cmds = [sc.cmd] + [w[1] for w in sc.cmd_schedule]
        for cmd in cmds:
            for v, (lo, hi) in zip(cmd, (c.vx_range, c.vy_range, c.yaw_range,
                                         c.height_range)):
                assert lo - 1e-9 <= v <= hi + 1e-9, sc.name
        for _, f in sc.pushes:
            assert np.linalg.norm(f) <= cfg.stages[-1].push_max_n
