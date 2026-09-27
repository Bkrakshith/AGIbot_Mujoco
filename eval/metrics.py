"""Aggregation of eval-battery episode results into scenario metrics."""

import math

import numpy as np


def summarize(results) -> dict:
    """Aggregate a list of cpu_eval.EpisodeResult into scenario metrics."""
    recov = [t for r in results for t in r.recovery_times_s if not math.isnan(t)]
    n_events = sum(len(r.recovery_times_s) for r in results)
    return {
        "episodes": len(results),
        "survival": float(np.mean([r.survived for r in results])),
        "mean_survived_time_s": float(np.mean([r.survived_time_s for r in results])),
        "mean_orientation_err": float(np.mean([r.orientation_err for r in results])),
        "mean_tracking_err": float(np.mean([r.tracking_err for r in results])),
        "mean_height_err": float(np.mean([r.height_err for r in results])),
        "median_recovery_s": float(np.median(recov)) if recov else None,
        "recovered_events": f"{len(recov)}/{n_events}",
    }


def compare(current: dict, best: dict, tolerance: float = 0.0) -> list[str]:
    """Survival regressions of `current` vs the previous `best`, per scenario.
    Returns a list of human-readable regression strings (empty = no regression)."""
    regressions = []
    for name, cur in current.items():
        prev = best.get(name)
        if prev is None:
            continue
        if cur["survival"] < prev["survival"] - tolerance:
            regressions.append(
                f"{name}: survival {cur['survival']:.2%} < previous best "
                f"{prev['survival']:.2%}")
    return regressions
