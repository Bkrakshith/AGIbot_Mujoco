"""Phase A: privileged teacher via brax PPO, staged curriculum (spec section 7).

Each curriculum stage is a separate brax `ppo.train` run (fresh jit — stage
parameters are static in the env), with network params carried across stages
via `restore_params`. Checkpoints are written from the eval callback; with the
default eval cadence (configs/train.yaml checkpoint.eval_every_steps) this
lands well inside the 5-minute wall-clock requirement at expected throughput.

Resume semantics (--resume): restores network + normalizer params, curriculum
stage, and the run seed from the latest checkpoint, then restarts the
in-progress stage. KNOWN LIMITATION: brax's `restore_params` does not carry
Adam optimizer state across process restarts — a resumed stage begins with a
fresh optimizer on restored weights. Adam re-warms within a few updates; this
is the accepted trade-off for using stock brax PPO rather than a fork.
"""

import dataclasses
import functools
import json
import os
import time

import jax
import numpy as np
from brax.training.agents.ppo import train as ppo

from . import curriculum
from .config import REPO_ROOT, Config, Stage
from .env import X1LocomotionEnv
from .networks import make_networks_factory
from .wrappers import wrap_for_training


def _ckpt_dir(cfg: Config, run_name: str) -> str:
    return os.path.join(REPO_ROOT, cfg.train.checkpoint.root, run_name)


def save_checkpoint(path: str, params, meta: dict):
    """Orbax save of (normalizer, policy, value) + sidecar meta.json."""
    import orbax.checkpoint as ocp
    path = os.path.abspath(path)
    ocp.PyTreeCheckpointer().save(path, jax.tree.map(np.asarray, params), force=True)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(path: str):
    import orbax.checkpoint as ocp
    ckptr = ocp.PyTreeCheckpointer()
    abspath = os.path.abspath(path)
    # Restore every leaf as a plain numpy array (they were saved as numpy).
    # The default restore path rebuilds jax.Arrays from the sharding stored
    # at save time, which names cpu:0 — absent from jax.local_devices() when
    # the default backend is CUDA, so a CPU-saved checkpoint would fail to
    # load under --backend gpu.
    meta_tree = ckptr.metadata(abspath)
    meta_tree = getattr(meta_tree, "item_metadata", meta_tree)
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta_tree)
    params = ckptr.restore(abspath, restore_args=restore_args)
    with open(path + ".meta.json") as f:
        meta = json.load(f)
    return params, meta


def latest_checkpoint(run_dir: str) -> str | None:
    if not os.path.isdir(run_dir):
        return None
    cands = [d for d in os.listdir(run_dir) if d.startswith("ckpt_")
             and d.split("_")[-1].isdigit()
             and os.path.exists(os.path.join(run_dir, d + ".meta.json"))]
    if not cands:
        return None
    cands.sort(key=lambda d: int(d.split("_")[-1]))
    return os.path.join(run_dir, cands[-1])


def _init_wandb(cfg: Config, run_name: str, stage_idx: int = 0):
    """One wandb run PER STAGE. Offline mode cannot append to an existing run
    id: re-initialising with the same id creates a second run dir whose
    datastore never spools — teacher_v3's s2 metrics were silently lost this
    way. A per-stage id (and a repeat counter for --resume re-runs of the same
    stage) guarantees every stage logs to a fresh, working datastore."""
    try:
        import glob
        import wandb
        base = f"{run_name}-s{stage_idx}"
        # REPO_ROOT-anchored: a cwd-relative glob undercounts prior runs when
        # training is launched from outside the repo root, recreating the
        # id-collision data loss this scheme exists to prevent.
        wandb_dir = os.path.join(REPO_ROOT, "wandb")
        n_prior = len(glob.glob(os.path.join(wandb_dir, f"offline-run-*-{base}*")))
        run_id = base if n_prior == 0 else f"{base}-r{n_prior}"
        wandb.init(project=cfg.train.logging.wandb_project, name=run_id,
                   mode=cfg.train.logging.wandb_mode, id=run_id, dir=REPO_ROOT)
        return wandb
    except Exception as e:  # wandb must never kill a training run
        print(f"[wandb disabled: {e}]")
        return None


def train_stage(cfg: Config, stage: Stage, stage_idx: int, run_name: str,
                num_envs: int | None = None, restore_params=None, seed: int = 0,
                env_fn=None, all_stages: list | None = None, on_eval=None):
    """Train one curriculum stage. Returns (params, final_eval_metrics).

    `env_fn(cfg, stage) -> Env` lets a different task supply its own
    environment (a task subclass of X1LocomotionEnv); it defaults to the locomotion env.
    `all_stages` is the full stage list this stage belongs to, used only to
    compute the wandb step offset; it defaults to the locomotion curriculum.
    `on_eval(current_step, metrics)` fires at every evaluation, so a caller can
    grade checkpoints as they are produced instead of only grading the last
    one (PPO often degrades late in a stage; select the best, not the last).
    """
    p = cfg.train.ppo
    num_envs = num_envs or p.num_envs
    env = (env_fn or (lambda c, s: X1LocomotionEnv(c, s)))(cfg, stage)

    run_dir = _ckpt_dir(cfg, run_name)
    os.makedirs(run_dir, exist_ok=True)
    wandb = _init_wandb(cfg, run_name, stage_idx)

    num_evals = max(2, int(stage.num_timesteps // cfg.train.checkpoint.eval_every_steps))
    last_metrics = {}
    t_start = time.time()
    prior = cfg.stages if all_stages is None else all_stages
    steps_offset = sum(s.num_timesteps for s in prior[:stage_idx])

    def progress_fn(num_steps, metrics):
        nonlocal last_metrics
        if metrics:
            last_metrics = dict(metrics)
            if on_eval is not None:
                # brax calls progress_fn then policy_params_fn with the SAME
                # current_step, so the checkpoint for these metrics is written
                # immediately after this returns.
                on_eval(int(num_steps), dict(metrics))
        sps = num_steps / max(time.time() - t_start, 1e-6)
        rew = metrics.get("eval/episode_reward", float("nan"))
        print(f"[{stage.name}] steps={num_steps:,} eval_reward={rew:.1f} "
              f"({sps:,.0f} steps/s)", flush=True)
        if wandb and metrics:
            wandb.log({"stage": stage_idx, **metrics}, step=steps_offset + num_steps)

    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        save_checkpoint(
            os.path.join(run_dir, f"ckpt_{steps_offset + current_step}"),
            params,
            meta={"stage": stage_idx, "stage_name": stage.name,
                  "stage_step": int(current_step), "seed": seed,
                  "run_name": run_name, "stage_cfg": dataclasses.asdict(stage),
                  "wall_time": time.time() - t_start},
        )

    make_policy, params, metrics = ppo.train(
        environment=env,
        wrap_env_fn=wrap_for_training,
        num_timesteps=stage.num_timesteps,
        episode_length=p.episode_length,
        num_envs=num_envs,
        num_eval_envs=p.num_eval_envs,
        num_evals=num_evals,
        unroll_length=p.unroll_length,
        batch_size=p.batch_size,
        num_minibatches=p.num_minibatches,
        num_updates_per_batch=p.num_updates_per_batch,
        learning_rate=p.learning_rate,
        entropy_cost=p.entropy_cost,
        discounting=p.discounting,
        gae_lambda=p.gae_lambda,
        reward_scaling=p.reward_scaling,
        clipping_epsilon=p.clipping_epsilon,
        max_grad_norm=p.max_grad_norm,
        normalize_observations=p.normalize_observations,
        network_factory=make_networks_factory(cfg.train.networks),
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        restore_params=restore_params,
        # deterministic per-stage seeding so --resume replays the same stream
        seed=seed + stage_idx,
    )
    del make_policy
    last_metrics.update(metrics or {})
    return params, last_metrics


def train_teacher(run_name: str, num_envs: int | None = None,
                  resume: bool = False, cfg: Config | None = None):
    from .config import load_config
    cfg = cfg or load_config()
    stages = cfg.stages
    seed = int(cfg.train.ppo.seed)
    run_dir = _ckpt_dir(cfg, run_name)

    start_stage, restore = 0, None
    if resume:
        latest = latest_checkpoint(run_dir)
        if latest is None:
            print(f"--resume: no checkpoint under {run_dir}, starting fresh")
        else:
            params, meta = load_checkpoint(latest)
            start_stage, seed = int(meta["stage"]), int(meta["seed"])
            # Stage indices renumber when the curriculum list is edited (e.g.
            # s1b_gait was inserted mid-run of teacher_v3). Trust the recorded
            # name, not the index, before restoring into a stage.
            ckpt_stage_name = meta.get("stage_name")
            if ckpt_stage_name is not None and (
                    start_stage >= len(stages)
                    or stages[start_stage].name != ckpt_stage_name):
                by_name = [i for i, s in enumerate(stages)
                           if s.name == ckpt_stage_name]
                if by_name:
                    print(f"--resume: curriculum changed; stage "
                          f"'{ckpt_stage_name}' moved from index "
                          f"{start_stage} to {by_name[0]}")
                    start_stage = by_name[0]
                else:
                    raise SystemExit(
                        f"--resume: checkpoint {latest} was written for stage "
                        f"'{ckpt_stage_name}', which no longer exists in "
                        f"configs/train.yaml — pick a checkpoint explicitly "
                        f"or restore the curriculum entry")
            restore = _as_restore_tuple(params)
            print(f"--resume: restored {latest} (stage {start_stage} "
                  f"'{meta['stage_name']}', stage_step {meta['stage_step']:,})")

    for i in range(start_stage, len(stages)):
        stage = stages[i]
        print(f"\n=== curriculum stage {i + 1}/{len(stages)}: {stage.name} "
              f"({stage.num_timesteps:,} steps) ===")
        params, metrics = train_stage(cfg, stage, i, run_name,
                                      num_envs=num_envs, restore_params=restore, seed=seed)
        restore = _as_restore_tuple(params)

        passed, tracking, air = curriculum.gate_passed(stage, metrics)
        if not passed:
            print(curriculum.stall_report(stage, tracking, air))
            raise SystemExit(2)
        print(f"[{stage.name}] gate passed: per-step tracking {tracking:.3f} "
              f">= {stage.gate_tracking}, air-time {air:.4f} "
              f">= {stage.gate_air_time}")

    print(f"\nTeacher training complete. Checkpoints: {run_dir}")
    return restore


def rehydrate_normalizer(d):
    """Orbax deserialises brax's RunningStatisticsState (and its UInt64 count)
    as plain dicts; rebuild the flax structs so running_statistics
    normalize/update work."""
    from brax.training.acme.running_statistics import RunningStatisticsState
    from brax.training.types import UInt64
    if isinstance(d, RunningStatisticsState):
        return d
    count = d["count"]
    if isinstance(count, dict):
        count = UInt64(hi=count["hi"], lo=count["lo"])
    return RunningStatisticsState(count=count, mean=d["mean"],
                                  summed_variance=d["summed_variance"], std=d["std"],
                                  std_eps=d.get("std_eps", 0.0))


def _as_restore_tuple(params):
    """brax returns/accepts (normalizer_params, policy_params, value_params);
    orbax round-trips it as a list of plain pytrees — normalise the container
    and rehydrate the normalizer struct."""
    params = tuple(params)
    return (rehydrate_normalizer(params[0]),) + params[1:]
