"""Phase B/C: student distillation (spec section 7).

Phase B (DAgger-style, on-policy): freeze the teacher; roll out the FULL-event
env (stage 4) where actions come from the frozen teacher trunk acting on z_hat
from the adaptation module (so the state distribution is the student's), and
regress   loss = MSE(z_hat, z) + w * MSE(a(z_hat), a(z))
training only the adaptation CNN. Gradients flow through the per-step
supervised loss, never through the physics.

Phase C (optional, gated): short PPO fine-tune of trunk + adaptation with
privileged obs removed from the policy path entirely (critic keeps them).
Observation normalisation is FROZEN at the teacher's statistics and baked into
the network factory — identical to what the exported ONNX will do.
"""

import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import distribution, networks, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen

from .config import Config, load_config
from .env import PRIV_OBS_SIZE, STUDENT_OBS_SIZE, X1LocomotionEnv
from .networks import (make_adaptation_module, make_policy_module,
                       make_teacher_networks, normalize_obs)
from .ppo_teacher import (_ckpt_dir, load_checkpoint, rehydrate_normalizer,
                          save_checkpoint)
from .wrappers import wrap_for_training


def _det_action(logits):
    """Deterministic (mode) action of the tanh-normal head, in [-1, 1]."""
    return jnp.tanh(logits[..., : logits.shape[-1] // 2])


# --------------------------------------------------------------- Phase B
def _with_trunk(policy_params, trunk):
    """Teacher policy params with the trunk subtree replaced (encoder kept):
    the student checkpoint stays shape-compatible with the teacher, so the
    CPU evaluator and the ONNX export read it unchanged."""
    p = jax.tree.map(lambda x: x, policy_params)
    p = dict(p)
    p["params"] = dict(p["params"])
    p["params"]["trunk"] = trunk
    return p


def distill(teacher_ckpt: str, run_name: str, cfg: Config | None = None,
            num_envs: int | None = None, env_fn=None,
            init_student: str | None = None):
    """Phase B: DAgger-distil a privileged, NOISE-FREE teacher into the
    deployable student = history CNN (-> z_hat) + its own action trunk.

    Why the trunk trains too (2026-09-27): the teacher was trained without
    sensor noise (train.teacher_obs_noise = 0; a single-frame teacher cannot
    filter the X1's 1.3 rad/s joint-vel noise). The student receives the FULL
    noise on both its current state and its history, so a frozen teacher
    trunk would face inputs it never saw. The student trunk starts as a copy
    of the teacher's.

      student action  a_s = trunk_s(noisy_state, CNN(noisy_history))
      teacher label   a_t = trunk_t(clean_state, encoder(privileged))
      loss = MSE(z_hat, z) + w * MSE(a_s, a_t)

    Rollouts are DAgger: each env acts with the teacher with prob beta
    (1 -> 0 over distill.beta_decay_iters) else with the student, so the
    data covers the states the student itself visits.
    """
    cfg = cfg or load_config()
    d = cfg.train.distill
    net = cfg.train.networks
    H = net.adaptation.history_length
    B = num_envs or d.num_envs
    T = d.unroll_length

    teacher_params, _meta = load_checkpoint(teacher_ckpt)
    normalizer = rehydrate_normalizer(teacher_params[0])
    teacher_policy = teacher_params[1]

    policy_module = make_policy_module(net)
    adapt_module = make_adaptation_module(net)

    env = (env_fn or (lambda c: X1LocomotionEnv(
        c, c.stages[-1], student_noise=True, emit_clean_state=True)))(cfg)
    wrapped = wrap_for_training(env, cfg.train.ppo.episode_length, 1)

    key = jax.random.PRNGKey(int(d.seed))
    key, k_init, k_reset = jax.random.split(key, 3)
    params = {"adapt": adapt_module.init(k_init, jnp.zeros((1, H, STUDENT_OBS_SIZE))),
              "trunk": teacher_policy["params"]["trunk"]}
    if init_student is not None:
        # continue an existing student (e.g. extra DAgger on a data mix
        # weighted to its weak scenario), same teacher labels
        sp, _ = load_checkpoint(init_student)
        params = {"adapt": sp["adaptation"], "trunk": sp["teacher"][1]["params"]["trunk"]}
        print(f"[distill] continuing student {init_student}", flush=True)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(d.learning_rate))
    opt_state = tx.init(params)

    env_state = jax.jit(wrapped.reset)(jax.random.split(k_reset, B))
    hist = jnp.zeros((B, H, STUDENT_OBS_SIZE))

    def norm(state, priv):
        return normalize_obs({"state": state, "privileged_state": priv}, normalizer)

    def teacher_act(s_clean_n, p_n):
        z = policy_module.apply(teacher_policy, p_n, method="encode")
        return _det_action(policy_module.apply(teacher_policy, s_clean_n, z,
                                               method="act_with_latent")), z

    def student_act(params, s_n, h):
        z_hat = adapt_module.apply(params["adapt"], h)
        pp = _with_trunk(teacher_policy, params["trunk"])
        return _det_action(policy_module.apply(pp, s_n, z_hat,
                                               method="act_with_latent")), z_hat

    @functools.partial(jax.jit, static_argnums=())
    def rollout(params, env_state, hist, beta, rng):
        def step_fn(carry, rng_t):
            env_state, hist = carry
            o = env_state.obs
            n = norm(o["state"], o["privileged_state"])
            s_n, p_n = n["state"], n["privileged_state"]
            s_clean_n = norm(o["state_clean"], o["privileged_state"])["state"]
            hist = jnp.roll(hist, -1, axis=1).at[:, -1].set(s_n)
            a_t, _ = teacher_act(s_clean_n, p_n)
            a_s, _ = student_act(params, s_n, hist)
            use_t = jax.random.uniform(rng_t, (a_t.shape[0], 1)) < beta
            nxt = wrapped.step(env_state, jnp.where(use_t, a_t, a_s))
            sample = (s_n, s_clean_n, p_n, hist, nxt.done)
            hist = hist * (1.0 - nxt.done)[:, None, None]   # mean-primed = 0
            return (nxt, hist), sample

        (env_state, hist), samples = jax.lax.scan(
            step_fn, (env_state, hist), jax.random.split(rng, T))
        dones = samples[-1]
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), samples[:-1])
        return env_state, hist, flat, jnp.mean(dones)

    def loss_fn(params, s_n, s_clean_n, p_n, h):
        a_t, z = teacher_act(s_clean_n, p_n)
        a_s, z_hat = student_act(params, s_n, h)
        z_loss = jnp.mean(jnp.square(z_hat - jax.lax.stop_gradient(z)))
        a_loss = jnp.mean(jnp.square(a_s - jax.lax.stop_gradient(a_t)))
        return z_loss + d.action_loss_weight * a_loss, (z_loss, a_loss)

    n_mb = int(d.get("num_minibatches", 8))
    n_ep = int(d.get("num_epochs", 2))

    @jax.jit
    def update(params, opt_state, batch, rng):
        n = batch[0].shape[0]
        mb = n // n_mb

        def epoch(carry, rng_e):
            params, opt_state = carry
            perm = jax.random.permutation(rng_e, n)[: mb * n_mb].reshape(n_mb, mb)

            def mb_step(carry, idx):
                params, opt_state = carry
                b = jax.tree.map(lambda x: x[idx], batch)
                (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, *b)
                upd, opt_state = tx.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(mb_step, (params, opt_state), perm)
            return (params, opt_state), jax.tree.map(jnp.mean, aux)

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state),
                                                jax.random.split(rng, n_ep))
        return params, opt_state, aux[0][-1], aux[1][-1]

    run_dir = _ckpt_dir(cfg, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_a, best_it, t_save = float("inf"), 0, time.time()
    decay = max(1, int(d.get("beta_decay_iters", 200)))

    def save(it, a_loss, path_suffix=None):
        student_policy = _with_trunk(teacher_policy, params["trunk"])
        save_checkpoint(
            os.path.join(run_dir, path_suffix or f"ckpt_{it}"),
            {"teacher": (teacher_params[0], student_policy, teacher_params[2]),
             "adaptation": params["adapt"]},
            meta={"phase": "B", "iteration": it, "a_loss": float(a_loss),
                  "teacher_ckpt": os.path.abspath(teacher_ckpt),
                  "student_trunk_trained": True, "run_name": run_name},
        )

    for it in range(int(d.num_iterations)):
        beta = max(0.0, 1.0 - it / decay)
        key, k_roll, k_upd = jax.random.split(key, 3)
        env_state, hist, batch, done_rate = rollout(params, env_state, hist, beta, k_roll)
        params, opt_state, z_loss, a_loss = update(params, opt_state, batch, k_upd)
        a_loss = float(a_loss)
        if it % 10 == 0:
            # done_rate: terminations per env step in these rollouts; with
            # beta = 0 it is the student's own fall rate (x episode length ->
            # falls per episode)
            print(f"[distill] it={it} beta={beta:.2f} z_loss={float(z_loss):.5f} "
                  f"a_loss={a_loss:.5f} done_rate={float(done_rate)*1e3:.2f}/1000 steps",
                  flush=True)
        if beta == 0.0 and a_loss < best_a - 1e-5:
            best_a, best_it = a_loss, it
            save(it, a_loss, path_suffix="ckpt_best")
        if time.time() - t_save > cfg.train.checkpoint.interval_s:
            save(it, a_loss)
            t_save = time.time()
        if beta == 0.0 and it - max(best_it, decay) > int(d.plateau_patience):
            print(f"[distill] a-loss plateaued (best {best_a:.5f} at it {best_it}); stopping")
            break

    save(it, a_loss, path_suffix="ckpt_final")
    print(f"[distill] done. Student checkpoints: {run_dir}/ckpt_best (lowest action "
          f"loss after beta=0) and ckpt_final")
    return os.path.join(run_dir, "ckpt_final")


# --------------------------------------------------------------- Phase C
class StudentPolicy(linen.Module):
    """Trunk + adaptation as one module — the deployment policy."""

    param_size: int
    latent_size: int
    policy_hidden: tuple
    adaptation: linen.Module

    @linen.compact
    def __call__(self, obs_state, obs_hist):
        from .networks import MLP
        z_hat = self.adaptation(obs_hist)
        trunk = MLP(tuple(self.policy_hidden) + (self.param_size,), name="trunk")
        return trunk(jnp.concatenate([obs_state, z_hat], axis=-1))


def make_student_networks(observation_size, action_size,
                          preprocess_observations_fn=types.identity_observation_preprocessor,
                          *, net_cfg, frozen_normalizer):
    """Phase C factory: policy sees ONLY student obs (+history); normalisation
    is frozen at the teacher's statistics (pass normalize_observations=False
    to ppo.train). Critic keeps privileged obs — it never ships."""
    del preprocess_observations_fn  # frozen stats baked in below
    param_dist = distribution.NormalTanhDistribution(event_size=action_size)
    adapt = make_adaptation_module(net_cfg)
    module = StudentPolicy(param_size=param_dist.param_size,
                           latent_size=net_cfg.latent_size,
                           policy_hidden=tuple(net_cfg.policy_hidden),
                           adaptation=adapt)
    from .networks import MLP
    value_module = MLP(tuple(net_cfg.value_hidden) + (1,))

    H = net_cfg.adaptation.history_length
    s_stats = frozen_normalizer

    def _norm(obs):
        n = normalize_obs({"state": obs["state"],
                           "privileged_state": obs["privileged_state"]},
                          _sub_stats(s_stats, ("state", "privileged_state")))
        # history normalised with the 'state' statistics broadcast over time
        hist_stats = _sub_stats(s_stats, ("state",))
        h = running_statistics.normalize({"state": obs["history"]}, hist_stats)["state"]
        return n["state"], n["privileged_state"], h

    def policy_init(key):
        return module.init(key, jnp.zeros((1, STUDENT_OBS_SIZE)),
                           jnp.zeros((1, H, STUDENT_OBS_SIZE)))

    def policy_apply(processor_params, params, obs):
        del processor_params
        s, _, h = _norm(obs)
        return module.apply(params, s, h)

    def value_init(key):
        return value_module.init(key, jnp.zeros((1, STUDENT_OBS_SIZE + PRIV_OBS_SIZE)))

    def value_apply(processor_params, params, obs):
        del processor_params
        s, p, _ = _norm(obs)
        return jnp.squeeze(value_module.apply(params, jnp.concatenate([s, p], -1)), -1)

    return ppo_networks.PPONetworks(
        policy_network=networks.FeedForwardNetwork(init=policy_init, apply=policy_apply),
        value_network=networks.FeedForwardNetwork(init=value_init, apply=value_apply),
        parametric_action_distribution=param_dist,
    )


def _sub_stats(stats, keys):
    """Slice a running-statistics pytree whose mean/std/count are dicts down to
    the given obs keys."""
    return jax.tree.map(
        lambda leaf: {k: leaf[k] for k in keys} if isinstance(leaf, dict) else leaf,
        stats, is_leaf=lambda l: isinstance(l, dict))


def finetune(student_ckpt: str, run_name: str, cfg: Config | None = None,
             env_fn=None,
             num_envs: int | None = None):
    """Phase C: PPO fine-tune of the full student. Caller is responsible for
    the >2 % survival-drop gate (train/train_student.py enforces it)."""
    from brax.training.agents.ppo import train as ppo

    cfg = cfg or load_config()
    net = cfg.train.networks
    fcfg = cfg.train.finetune
    p = cfg.train.ppo

    params, _meta = load_checkpoint(student_ckpt)
    normalizer = rehydrate_normalizer(params["teacher"][0])
    teacher_policy = params["teacher"][1]
    teacher_value = params["teacher"][2]
    adapt_params = params["adaptation"]

    # transplant: student policy = frozen-teacher trunk + distilled adaptation
    student_policy_params = {"params": {
        "adaptation": adapt_params["params"],
        "trunk": teacher_policy["params"]["trunk"],
    }}

    env = (env_fn or (lambda c: X1LocomotionEnv(c, c.stages[-1],
                                                include_history=True)))(cfg)
    obs_spec = jax.tree.map(
        lambda s: specs.Array(s if isinstance(s, tuple) else (s,), jnp.dtype("float32")),
        env.observation_size, is_leaf=lambda x: isinstance(x, (int, tuple)))
    norm_init = running_statistics.init_state(obs_spec)

    factory = functools.partial(make_student_networks, net_cfg=net,
                                frozen_normalizer=normalizer)

    run_dir = _ckpt_dir(cfg, run_name)
    os.makedirs(run_dir, exist_ok=True)
    t_start = time.time()

    def progress_fn(num_steps, metrics):
        rew = metrics.get("eval/episode_reward", float("nan"))
        L = float(metrics.get("eval/avg_episode_length", float("nan")))
        trk = float(metrics.get("eval/episode_reward/tracking_lin_vel", float("nan"))) / max(L, 1.0)
        print(f"[finetune] steps={num_steps:,} eval_reward={rew:.1f} ep_len={L:.0f} "
              f"tracking={trk:.3f}", flush=True)
        if metrics:
            with open(os.path.join(run_dir, "evals.jsonl"), "a") as fh:
                fh.write(json.dumps({"step": int(num_steps),
                                     **{k: float(v) for k, v in metrics.items()}}) + "\n")

    def policy_params_fn(current_step, make_policy, ppo_params):
        del make_policy
        _save_student(run_dir, f"ckpt_ft_{current_step}", ppo_params, normalizer,
                      teacher_policy, student_ckpt, current_step, t_start)

    _, ppo_params, _metrics = ppo.train(
        environment=env,
        wrap_env_fn=wrap_for_training,
        num_timesteps=int(fcfg.num_timesteps),
        episode_length=p.episode_length,
        # history obs (H x obs per env) make the rollout buffer ~H times
        # larger than the teacher's: finetune may override the PPO geometry
        num_envs=num_envs or int(fcfg.get("num_envs", p.num_envs)),
        num_eval_envs=p.num_eval_envs,
        num_evals=max(2, int(fcfg.num_timesteps // cfg.train.checkpoint.eval_every_steps)),
        unroll_length=p.unroll_length,
        batch_size=int(fcfg.get("batch_size", p.batch_size)),
        num_minibatches=int(fcfg.get("num_minibatches", p.num_minibatches)),
        num_updates_per_batch=p.num_updates_per_batch,
        learning_rate=float(fcfg.learning_rate),
        entropy_cost=p.entropy_cost,
        discounting=p.discounting,
        gae_lambda=p.gae_lambda,
        reward_scaling=p.reward_scaling,
        clipping_epsilon=p.clipping_epsilon,
        max_grad_norm=p.max_grad_norm,
        normalize_observations=False,  # frozen stats baked into the factory
        network_factory=factory,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        restore_params=(norm_init, student_policy_params, teacher_value),
        seed=int(p.seed) + 100,
    )
    out = _save_student(run_dir, "ckpt_final", ppo_params, normalizer,
                        teacher_policy, student_ckpt, int(fcfg.num_timesteps), t_start)
    print(f"[finetune] done. Checkpoint: {out}")
    return out


def _save_student(run_dir, name, ppo_params, normalizer, teacher_policy,
                  src_ckpt, step, t_start):
    """Re-pack fine-tuned params into the standard student checkpoint layout
    so eval/export code paths are identical for Phase B and C outputs."""
    student_policy = ppo_params[1]
    policy = {"params": {
        "encoder": teacher_policy["params"]["encoder"],  # unused downstream, kept for shape
        "trunk": student_policy["params"]["trunk"],
    }}
    adaptation = {"params": student_policy["params"]["adaptation"]}
    path = os.path.join(run_dir, name)
    save_checkpoint(
        path,
        {"teacher": (normalizer, policy, ppo_params[2]), "adaptation": adaptation},
        meta={"phase": "C", "step": int(step), "source_ckpt": os.path.abspath(src_ckpt),
              "wall_time": time.time() - t_start},
    )
    return path
