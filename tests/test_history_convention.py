"""The observation-history ring buffer must mean exactly the same thing at
training time and at deployment time.

Phase B (distill_student.py) stores ALREADY-NORMALISED observations in the
buffer and zeroes it at episode boundaries, so an unfilled slot is exactly 0.0
in the CNN's input space.

The eval/deployment adapters (cpu_eval.StudentJaxPolicy, cpu_eval.OnnxPolicy,
and the exported graph) store RAW observations and normalise on the way in.
For those to agree with training, the buffer must be primed with the
observation MEAN — (mean - mean)/std == 0.

Priming with zeros instead presents the CNN with -mean/std, which measured
~14 in norm on a trained checkpoint and drove battery survival to 0 % on every
scenario, including one the teacher passed at 100 %. Nothing else in the suite
catches this: the export parity gates feed dense random buffers and so never
exercise a partially-filled one.
"""

import numpy as np
import pytest

from x1_locomotion.config import load_config
from x1_locomotion.cpu_eval import StudentJaxPolicy
from x1_locomotion.env import STUDENT_OBS_SIZE


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class _Buffer:
    """The raw-buffer + normalise-on-read convention, isolated from any policy."""

    def __init__(self, mean, std, length, fill):
        self._mean, self._std, self._H = mean, std, length
        self._hist = np.tile(fill, (length, 1)).astype(np.float32)

    def push(self, obs):
        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = obs

    def normalised(self):
        return (self._hist - self._mean) / self._std


@pytest.fixture(scope="module")
def stats():
    rng = np.random.default_rng(0)
    mean = rng.normal(0.0, 3.0, STUDENT_OBS_SIZE).astype(np.float32)
    std = rng.uniform(0.5, 2.0, STUDENT_OBS_SIZE).astype(np.float32)
    return mean, std


def test_mean_primed_buffer_reads_as_zero(stats):
    """An unfilled slot must be 0 after normalisation — what training produces."""
    mean, std = stats
    buf = _Buffer(mean, std, 50, fill=mean)
    assert np.allclose(buf.normalised(), 0.0, atol=1e-5)


def test_zero_primed_buffer_is_out_of_distribution(stats):
    """The regression this guards: zero-priming is NOT neutral, it is -mean/std."""
    mean, std = stats
    buf = _Buffer(mean, std, 50, fill=np.zeros(STUDENT_OBS_SIZE, np.float32))
    ghost = buf.normalised()
    assert not np.allclose(ghost, 0.0, atol=1e-2)
    # and it is large, not a rounding-level nuisance
    assert np.linalg.norm(ghost[0]) > 1.0


def test_only_pushed_rows_are_nonzero(stats):
    """After k pushes, exactly the newest k rows carry signal; the rest are 0."""
    mean, std = stats
    buf = _Buffer(mean, std, 50, fill=mean)
    rng = np.random.default_rng(1)
    k = 7
    for _ in range(k):
        buf.push(rng.normal(0.0, 1.0, STUDENT_OBS_SIZE).astype(np.float32))
    n = buf.normalised()
    assert np.allclose(n[:-k], 0.0, atol=1e-5), "unfilled slots must stay neutral"
    assert not np.allclose(n[-k:], 0.0, atol=1e-5), "pushed slots must carry signal"


def test_student_adapter_primes_with_mean(cfg):
    """StudentJaxPolicy must prime its buffer with obs_mean, not zeros.

    Constructed without a checkpoint: we only assert the buffer convention.
    """
    H = cfg.train.networks.adaptation.history_length
    rng = np.random.default_rng(2)
    mean = rng.normal(0.0, 3.0, STUDENT_OBS_SIZE).astype(np.float32)
    std = rng.uniform(0.5, 2.0, STUDENT_OBS_SIZE).astype(np.float32)

    pol = object.__new__(StudentJaxPolicy)      # bypass checkpoint loading
    pol._mean, pol._std, pol._H = mean, std, H
    pol._hist = np.tile(mean, (H, 1)).astype(np.float32)

    StudentJaxPolicy.reset(pol)
    assert np.allclose((pol._hist - mean) / std, 0.0, atol=1e-5), (
        "reset() must leave the buffer normalising to zero")
    assert not np.allclose(pol._hist, 0.0), (
        "buffer holds RAW values, so a correct reset is NOT all-zeros")
