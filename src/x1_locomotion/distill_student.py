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
def distill(teacher_ckpt: str, run_name: str, cfg: Config | None = None,
            num_envs: int | None = None, env_fn=None):
    """Phase B: DAgger-distil a privileged teacher into the history CNN.

    `env_fn(cfg) -> Env` supplies the environment the adaptation data is
    collected in; it defaults to the locomotion env at its final curriculum
    stage. A different task's teacher needs its own env — the student can only
    learn to infer a latent from history it has actually seen.
    """
    cfg = cfg or load_config()
    d = cfg.train.distill
    net = cfg.train.networks
    H = net.adaptation.history_length
    B = num_envs or d.num_envs
    T = d.unroll_length

    teacher_params, _meta = load_checkpoint(teacher_ckpt)
    normalizer = rehydrate_normalizer(teacher_params[0])
    policy_params = teacher_params[1]

    policy_module = make_policy_module(net)
    adapt_module = make_adaptation_module(net)

    # full events for adaptation data
    env = (env_fn or (lambda c: X1LocomotionEnv(c, c.stages[-1])))(cfg)
    wrapped = wrap_for_training(env, cfg.train.ppo.episode_length, 1)

    key = jax.random.PRNGKey(int(d.seed))
    key, k_init, k_reset = jax.random.split(key, 3)
    adapt_params = adapt_module.init(k_init, jnp.zeros((1, H, STUDENT_OBS_SIZE)))
    tx = optax.adam(d.learning_rate)
    opt_state = tx.init(adapt_params)

    env_state = jax.jit(wrapped.reset)(jax.random.split(k_reset, B))
    hist = jnp.zeros((B, H, STUDENT_OBS_SIZE))

    def act_with_latent(s, z):
        return policy_module.apply(policy_params, s, z, method="act_with_latent")

    def encode(p):
        return policy_module.apply(policy_params, p, method="encode")

    @jax.jit
    def rollout(adapt_params, env_state, hist):
        """T steps driven by the student's z_hat; returns supervised samples."""

        def step_fn(carry, _):
            env_state, hist = carry
            obs_n = normalize_obs(env_state.obs, normalizer)
            hist = jnp.roll(hist, -1, axis=1).at[:, -1].set(obs_n["state"])
            z_hat = adapt_module.apply(adapt_params, hist)
            act = _det_action(act_with_latent(obs_n["state"], z_hat))
            next_state = wrapped.step(env_state, act)
            sample = (obs_n["state"], obs_n["privileged_state"], hist)
            # zero the history across episode boundaries — no leakage of the
            # previous episode's dynamics into z_hat
            hist = hist * (1.0 - next_state.done)[:, None, None]
            return (next_state, hist), sample

        (env_state, hist), samples = jax.lax.scan(step_fn, (env_state, hist), None, T)
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), samples)
        return env_state, hist, flat

    def loss_fn(adapt_params, s, p, h):
        z = encode(p)
        z_hat = adapt_module.apply(adapt_params, h)
        a_teacher = _det_action(act_with_latent(s, z))
        a_student = _det_action(act_with_latent(s, z_hat))
        z_loss = jnp.mean(jnp.square(z_hat - z))
        a_loss = jnp.mean(jnp.square(a_student - a_teacher))
        return z_loss + d.action_loss_weight * a_loss, (z_loss, a_loss)

    @jax.jit
    def update(adapt_params, opt_state, s, p, h):
        (_, (z_loss, a_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            adapt_params, s, p, h)
        updates, opt_state = tx.update(grads, opt_state)
        return optax.apply_updates(adapt_params, updates), opt_state, z_loss, a_loss

    run_dir = _ckpt_dir(cfg, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_z, best_it, t_save = float("inf"), 0, time.time()

    def save(it, z_loss, path_suffix=None):
        save_checkpoint(
            os.path.join(run_dir, path_suffix or f"ckpt_{it}"),
            {"teacher": tuple(teacher_params), "adaptation": adapt_params},
            meta={"phase": "B", "iteration": it, "z_loss": float(z_loss),
                  "teacher_ckpt": os.path.abspath(teacher_ckpt),
                  "run_name": run_name},
        )

    for it in range(int(d.num_iterations)):
        env_state, hist, (s, p, h) = rollout(adapt_params, env_state, hist)
        adapt_params, opt_state, z_loss, a_loss = update(adapt_params, opt_state, s, p, h)
        z_loss = float(z_loss)
        if it % 10 == 0:
            print(f"[distill] it={it} z_loss={z_loss:.5f} a_loss={float(a_loss):.5f}",
                  flush=True)
        if z_loss < best_z - 1e-5:
            best_z, best_it = z_loss, it
        if time.time() - t_save > cfg.train.checkpoint.interval_s:
            save(it, z_loss)
            t_save = time.time()
        if it - best_it > int(d.plateau_patience):
            print(f"[distill] z-loss plateaued (best {best_z:.5f} at it {best_it}); stopping")
            break

    save(it, z_loss, path_suffix="ckpt_final")
    print(f"[distill] done. Student checkpoint: {os.path.join(run_dir, 'ckpt_final')}")
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
        print(f"[finetune] steps={num_steps:,} eval_reward={rew:.1f}", flush=True)

    def policy_params_fn(current_step, make_policy, ppo_params):
        del make_policy
        _save_student(run_dir, f"ckpt_ft_{current_step}", ppo_params, normalizer,
                      teacher_policy, student_ckpt, current_step, t_start)

    _, ppo_params, _metrics = ppo.train(
        environment=env,
        wrap_env_fn=wrap_for_training,
        num_timesteps=int(fcfg.num_timesteps),
        episode_length=p.episode_length,
        num_envs=num_envs or p.num_envs,
        num_eval_envs=p.num_eval_envs,
        num_evals=max(2, int(fcfg.num_timesteps // cfg.train.checkpoint.eval_every_steps)),
        unroll_length=p.unroll_length,
        batch_size=p.batch_size,
        num_minibatches=p.num_minibatches,
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
