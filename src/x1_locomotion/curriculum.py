"""Curriculum staging (spec section 7 Phase A). Stages are defined in
configs/train.yaml; each stage is a frozen env configuration and training
proceeds stage by stage, gated on the tracking metric. Per the hard
boundaries: a stalled stage is surfaced loudly, never papered over by
shrinking the task spec."""

from .config import Config, Stage


def get_stages(cfg: Config) -> list[Stage]:
    return cfg.stages


def _per_step(eval_metrics: dict, key: str) -> float:
    per_episode = float(eval_metrics.get(key, 0.0))
    steps = float(eval_metrics.get("eval/avg_episode_length", 1.0)) or 1.0
    return per_episode / steps


def gate_passed(stage: Stage, eval_metrics: dict) -> tuple[bool, float, float]:
    """Two-criterion gate from the final eval (brax logs env metrics as
    eval/episode_<name>, summed over the episode — divide by episode length
    for per-step values):

      1. tracking: mean per-step lin-vel tracking kernel >= gate_tracking.
      2. gait activity: mean per-step RAW feet_air_time reward >= gate_air_time.

    The second criterion exists because the first is gameable: teacher_v3's s1
    scored 0.78 tracking by drift-leaning toward small commands WITHOUT EVER
    LIFTING A FOOT (measured: vx 0.000, zero foot lift under a forced 0.5 m/s
    command). Air time is only credited on touchdown after a swing — drift
    scores exactly 0. gate_air_time=0 disables the criterion.

    Returns (passed, per_step_tracking, per_step_air_time)."""
    tracking = _per_step(eval_metrics, "eval/episode_reward/tracking_lin_vel")
    air = _per_step(eval_metrics, "eval/episode_reward/feet_air_time")
    passed = tracking >= stage.gate_tracking and air >= stage.gate_air_time
    return passed, tracking, air


def stall_report(stage: Stage, per_step_tracking: float,
                 per_step_air: float = float("nan")) -> str:
    return (
        f"\n{'=' * 72}\n"
        f"CURRICULUM STALL: stage '{stage.name}' finished below its gate.\n"
        f"  mean per-step tracking kernel: {per_step_tracking:.3f} "
        f"(gate: {stage.gate_tracking:.3f})\n"
        f"  mean per-step air-time reward: {per_step_air:.4f} "
        f"(gate: {stage.gate_air_time:.4f})\n\n"
        "Per the project spec the task (speeds, pushes) is NOT shrunk silently.\n"
        "Suggested knobs, in order of past usefulness on similar tasks:\n"
        "  0. If TRACKING passed but AIR TIME failed: the policy is tracking\n"
        "     without stepping (drift-leaning). More budget rarely fixes this\n"
        "     alone — check gait_imitation / feet_air_time weights first.\n"
        "     If AIR TIME passed but tracking failed: a young gait is being\n"
        "     refined — more budget for this stage is usually enough.\n"
        "  1. More timesteps for this stage (configs/train.yaml curriculum entry).\n"
        "  2. Soften effort penalties (torque/action_rate in configs/rewards.yaml)\n"
        "     or widen tracking_sigma — over-tight kernels stall early learning.\n"
        "  3. Raise entropy_cost slightly (exploration collapse shows up as\n"
        "     high survival but poor tracking).\n"
        "  4. Check termination rate in W&B: if >30 % of episodes terminate,\n"
        "     lower push_max_n FOR THIS STAGE ONLY and add\n"
        "     an intermediate stage — do not touch the final stage-4 spec.\n"
        f"{'=' * 72}\n"
    )


def do_nothing_floor(cfg, stage: Stage, n: int = 200_000, seed: int = 0) -> float:
    """Per-step tracking kernel a policy that never moves would score.

    Stand-still steps pay exp(0) = 1 for free, so a tracking gate at or below
    this floor tests nothing (theory §7, failure M8). Monte Carlo over the
    SAME command distribution randomize.sample_command draws from, including
    walk <-> stand switches when the stage enables them. Assumes the stander
    survives the episode (an upper bound on what doing nothing earns).
    Gates are set at (floor + 1) / 2; tests/test_curriculum.py pins that.
    """
    import numpy as np
    c = cfg.train.commands
    sigma = cfg.rewards.kernels.tracking_sigma
    ep_s = cfg.train.env.episode_length_s
    rng = np.random.default_rng(seed)

    def kernel(k):
        vx = rng.uniform(*c.vx_range, k)
        vy = rng.uniform(*c.vy_range, k)
        return np.exp(-(vx ** 2 + vy ** 2) / sigma)

    u = rng.uniform(size=n)
    mode = np.where(u < c.p_stand, 0, np.where(u < c.p_stand + c.p_walk, 1, 2))
    if not stage.cmd_switch_enabled:
        mode = np.minimum(mode, 1)
    k1, k2 = kernel(n), kernel(n)
    frac = rng.uniform(*c.switch_window_s, n) / ep_s   # fraction before switch
    to_stand = rng.uniform(size=n) < 0.5
    # switch episodes: walk->stand (k1 then 1) or stand->walk (1 then k2)
    sw = np.where(to_stand, frac * k1 + (1 - frac) * 1.0, frac * 1.0 + (1 - frac) * k2)
    per_ep = np.where(mode == 0, 1.0, np.where(mode == 1, k1, sw))
    return float(per_ep.mean())
