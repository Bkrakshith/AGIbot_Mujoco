"""Actor, critic, privileged encoder, and adaptation module (flax), wired as a
brax-compatible PPO network factory.

RMA-style structure (spec section 5/7):
  teacher policy  pi(a | obs_state, z),  z = mu(obs_priv)   [encoder + trunk]
  student         pi(a | obs_state, z_hat), z_hat = phi(obs_state history)
The trunk is shared verbatim between teacher and student; distillation only
trains phi (the 1D-CNN adaptation module, RMA-faithful and ONNX-friendly —
VALID padding so the torch/ONNX ports are exact).
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen

from .layout import NU


class MLP(linen.Module):
    layer_sizes: Sequence[int]
    activation: Callable = linen.swish
    activate_final: bool = False

    @linen.compact
    def __call__(self, x):
        for i, size in enumerate(self.layer_sizes):
            x = linen.Dense(size, name=f"hidden_{i}")(x)
            if i < len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
        return x


class TeacherPolicy(linen.Module):
    """Privileged encoder mu(x_priv) -> z, plus action trunk on [state, z]."""

    param_size: int
    latent_size: int
    encoder_hidden: Sequence[int]
    policy_hidden: Sequence[int]
    activation: Callable = linen.swish

    def setup(self):
        self.encoder = MLP(tuple(self.encoder_hidden) + (self.latent_size,),
                           self.activation, name="encoder")
        self.trunk = MLP(tuple(self.policy_hidden) + (self.param_size,),
                         self.activation, name="trunk")

    def __call__(self, obs_state, obs_priv):
        return self.act_with_latent(obs_state, self.encode(obs_priv))

    def encode(self, obs_priv):
        return self.encoder(obs_priv)

    def act_with_latent(self, obs_state, z):
        return self.trunk(jnp.concatenate([obs_state, z], axis=-1))


class AdaptationModule(linen.Module):
    """RMA adaptation: 1D CNN over the last H control steps of student obs,
    regressed to the frozen teacher's z. Stateless (no GRU carry), so
    deployment just keeps a ring buffer."""

    latent_size: int
    conv_channels: Sequence[int]
    conv_kernels: Sequence[int]
    conv_strides: Sequence[int]
    dense: int
    activation: Callable = linen.swish

    @linen.compact
    def __call__(self, hist):  # (..., H, obs_state_dim)
        x = hist
        for i, (c, k, s) in enumerate(zip(self.conv_channels, self.conv_kernels,
                                          self.conv_strides)):
            x = linen.Conv(features=c, kernel_size=(k,), strides=(s,),
                           padding="VALID", name=f"conv_{i}")(x)
            x = self.activation(x)
        x = x.reshape(x.shape[:-2] + (-1,))
        x = self.activation(linen.Dense(self.dense, name="dense_0")(x))
        return linen.Dense(self.latent_size, name="dense_1")(x)


def _obs_dim(observation_size, key):
    size = observation_size[key]
    return size[-1] if isinstance(size, (tuple, list)) else int(size)


def make_teacher_networks(
    observation_size,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    *,
    latent_size: int = 16,
    encoder_hidden: Sequence[int] = (64, 64),
    policy_hidden: Sequence[int] = (512, 256, 128),
    value_hidden: Sequence[int] = (512, 256, 128),
    activation: Callable = linen.swish,
) -> ppo_networks.PPONetworks:
    """brax PPO network factory (pass via functools.partial to ppo.train).

    The policy net consumes the dict obs {'state', 'privileged_state'}; the
    critic sees both parts concatenated (asymmetric actor-critic is free here
    since the critic never ships)."""
    state_dim = _obs_dim(observation_size, "state")
    priv_dim = _obs_dim(observation_size, "privileged_state")

    param_dist = distribution.NormalTanhDistribution(event_size=action_size)
    policy_module = TeacherPolicy(
        param_size=param_dist.param_size,
        latent_size=latent_size,
        encoder_hidden=encoder_hidden,
        policy_hidden=policy_hidden,
        activation=activation,
    )
    value_module = MLP(tuple(value_hidden) + (1,), activation)

    def policy_init(key):
        return policy_module.init(key, jnp.zeros((1, state_dim)), jnp.zeros((1, priv_dim)))

    def policy_apply(processor_params, params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(params, obs["state"], obs["privileged_state"])

    def value_init(key):
        return value_module.init(key, jnp.zeros((1, state_dim + priv_dim)))

    def value_apply(processor_params, params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        x = jnp.concatenate([obs["state"], obs["privileged_state"]], axis=-1)
        return jnp.squeeze(value_module.apply(params, x), axis=-1)

    return ppo_networks.PPONetworks(
        policy_network=networks.FeedForwardNetwork(init=policy_init, apply=policy_apply),
        value_network=networks.FeedForwardNetwork(init=value_init, apply=value_apply),
        parametric_action_distribution=param_dist,
    )


def make_networks_factory(net_cfg):
    """Bind configs/train.yaml network sizes into a brax network_factory."""
    import functools
    return functools.partial(
        make_teacher_networks,
        latent_size=net_cfg.latent_size,
        encoder_hidden=tuple(net_cfg.encoder_hidden),
        policy_hidden=tuple(net_cfg.policy_hidden),
        value_hidden=tuple(net_cfg.value_hidden),
    )


def make_adaptation_module(net_cfg) -> AdaptationModule:
    a = net_cfg.adaptation
    return AdaptationModule(
        latent_size=net_cfg.latent_size,
        conv_channels=tuple(a.conv_channels),
        conv_kernels=tuple(a.conv_kernels),
        conv_strides=tuple(a.conv_strides),
        dense=a.dense,
    )


def make_policy_module(net_cfg, action_size: int = NU) -> TeacherPolicy:
    param_dist = distribution.NormalTanhDistribution(event_size=action_size)
    return TeacherPolicy(
        param_size=param_dist.param_size,
        latent_size=net_cfg.latent_size,
        encoder_hidden=tuple(net_cfg.encoder_hidden),
        policy_hidden=tuple(net_cfg.policy_hidden),
    )


def deterministic_action(param_dist: distribution.NormalTanhDistribution, logits):
    """Mode of the tanh-normal — the deployment-time action in [-1, 1]."""
    return param_dist.postprocess(logits[..., : logits.shape[-1] // 2])


def normalize_obs(obs, normalizer_params):
    """Running-statistics normalisation identical to brax's preprocessing."""
    from brax.training.acme import running_statistics
    return running_statistics.normalize(obs, normalizer_params)
