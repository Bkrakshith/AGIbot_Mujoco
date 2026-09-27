"""Training wrappers for brax PPO (passed via `wrap_env_fn`).

brax's stock AutoResetWrapper restores the episode's FIRST state on reset,
which would freeze domain randomisation per env for the whole run. The spec
requires per-episode DR (payloads, friction, gains, ...), so this AutoReset
calls env.reset with a fresh key every step and splices the new state in where
done — a real re-randomised reset at every episode boundary. Cost: one extra
mjx.forward per env per control step (~10 % of the physics budget).
"""

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

# bookkeeping keys owned by outer wrappers / PPO — never overwritten on reset
_PRESERVED_INFO = ("truncation", "steps", "autoreset_rng")


class PerEpisodeAutoResetWrapper(Wrapper):
    """Auto-reset with fresh per-episode randomisation (batched envs only)."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["autoreset_rng"] = jax.vmap(
            lambda k: jax.random.fold_in(k, 0x5EED))(rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # fresh episode bookkeeping for envs that finished last step
        if "steps" in state.info:
            steps = jnp.where(state.done, jnp.zeros_like(state.info["steps"]),
                              state.info["steps"])
            state = state.replace(info={**state.info, "steps": steps})
        state = state.replace(done=jnp.zeros_like(state.done))

        carry_rng = state.info["autoreset_rng"]
        state = self.env.step(state, action)

        keys = jax.vmap(jax.random.split)(carry_rng)  # (B, 2, 2)
        reset_state = self.env.reset(keys[:, 0])
        done = state.done

        def where_done(new, old):
            d = jnp.reshape(done, done.shape + (1,) * (new.ndim - 1))
            return jnp.where(d, new, old)

        pipeline_state = jax.tree.map(where_done, reset_state.pipeline_state,
                                      state.pipeline_state)
        obs = jax.tree.map(where_done, reset_state.obs, state.obs)
        info = {}
        for k, v in state.info.items():
            if k in _PRESERVED_INFO or k not in reset_state.info:
                info[k] = v
            else:
                info[k] = jax.tree.map(where_done, reset_state.info[k], v)
        info["autoreset_rng"] = keys[:, 1]
        return state.replace(pipeline_state=pipeline_state, obs=obs, info=info)


def wrap_for_training(env: Env, episode_length: int = 1000, action_repeat: int = 1,
                      randomization_fn=None) -> Wrapper:
    """Drop-in for brax.envs.training.wrap (same signature, so it can be passed
    as ppo.train(wrap_env_fn=...)). randomization_fn is ignored — DR here is
    per-episode inside the env, not per-instance via vmapped model axes."""
    del randomization_fn
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = VmapWrapper(env)
    return PerEpisodeAutoResetWrapper(env)
