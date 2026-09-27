"""Directional tilt termination (env._tilted), shared by env and evaluator.

The anisotropic rule (a separate forward-pitch limit) exists for posture
tasks; for stand/walk both limits are equal and the rule must reduce exactly
to the old combined limit. Training and evaluation must agree (failure M1),
and the task posture must keep >= 15-20 deg of headroom (failure L1).
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax.numpy as jnp  # noqa: E402

from x1_locomotion.config import DotDict, load_config  # noqa: E402
from x1_locomotion.env import X1LocomotionEnv  # noqa: E402


def pg(pitch_deg=0.0, roll_deg=0.0):
    """Projected gravity for a torso pitched forward and/or rolled.

    Body-frame gravity for a pitch-then-roll rotation; +x is forward pitch.
    """
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    return jnp.array([np.sin(p), -np.cos(p) * np.sin(r), -np.cos(p) * np.cos(r)])


def term(max_tilt=1.0, max_fwd=None):
    d = {"max_tilt": max_tilt}
    if max_fwd is not None:
        d["max_pitch_forward"] = max_fwd
    return DotDict(d)


def tilted(t, g):
    return bool(X1LocomotionEnv._tilted(t, g))


# ---- locomotion must be unchanged -----------------------------------------

def test_matches_the_old_combined_limit_when_forward_is_not_raised():
    """The old rule was `-proj_g[2] < cos(max_tilt)`. With max_pitch_forward
    equal to max_tilt the new rule must agree on EVERY orientation, or the
    locomotion teacher's termination silently moved."""
    t = term(1.0, 1.0)
    for pitch in range(-170, 171, 5):
        for roll in range(-170, 171, 5):
            g = pg(pitch, roll)
            old = bool(-g[2] < jnp.cos(1.0))
            assert tilted(t, g) == old, f"pitch={pitch} roll={roll}"


def test_absent_key_falls_back_to_max_tilt():
    """Configs predating max_pitch_forward must keep their exact behaviour."""
    for pitch in (-80, -40, 0, 40, 80):
        assert tilted(term(1.0), pg(pitch)) == tilted(term(1.0, 1.0), pg(pitch))


def test_locomotion_config_does_not_move_its_limit(  ):
    cfg = load_config()
    t = cfg.rewards.termination
    assert t.max_pitch_forward == t.max_tilt


def test_training_and_eval_paths_agree_exactly():
    """The battery is the independent check; it must use the same contract
    as training for every orientation (failure M1)."""
    import numpy as _np

    from x1_locomotion.rewards import tilt_exceeded
    for pitch in range(-100, 101, 5):
        for roll in range(-100, 101, 5):
            g = pg(pitch, roll)
            train = bool(tilt_exceeded(g, 1.0, 1.30, jnp))
            ev = bool(tilt_exceeded(_np.asarray(g, dtype=float), 1.0, 1.30, _np))
            assert train == ev, f"pitch={pitch} roll={roll}"


def test_walking_posture_has_termination_headroom():
    """Walking pitches/rolls the torso by ~10 deg at most; the limit must sit
    >= 20 deg beyond that (skill Phase 2.4)."""
    t = load_config().rewards.termination
    assert np.degrees(min(t.max_tilt, t.max_pitch_forward)) >= 10.0 + 20.0
