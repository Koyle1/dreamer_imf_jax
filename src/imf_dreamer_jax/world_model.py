"""Functional recurrent world model with Gaussian or conditional iMF prior."""

from __future__ import annotations

from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .imf import improved_meanflow_loss, init_imf, sample_imf_one_step
from .nn import Params, gru, init_gru, init_mlp, mlp
from .types import DiagonalNormal, RSSMState, SequenceStates, WorldModelLoss


Batch = Mapping[str, Array]


def init_world_model(config: DreamerConfig, key: Array) -> Params:
    """Initialize all recurrent world-model parameters."""

    keys = iter(jax.random.split(key, 8))
    params: Params = {
        "encoder": init_mlp(
            next(keys), config.observation_dim, config.hidden_dim, config.embedding_dim
        ),
        "decoder": init_mlp(
            next(keys), config.feature_dim, config.hidden_dim, config.observation_dim
        ),
        "recurrence": init_gru(
            next(keys), config.stochastic_dim + config.action_dim, config.deterministic_dim
        ),
        "posterior": init_mlp(
            next(keys),
            config.deterministic_dim + config.embedding_dim,
            config.hidden_dim,
            2 * config.stochastic_dim,
        ),
        "reward": init_mlp(next(keys), config.feature_dim, config.hidden_dim, 1),
        "continuation": init_mlp(next(keys), config.feature_dim, config.hidden_dim, 1),
    }
    prior_key = next(keys)
    if config.prior == "gaussian":
        params["prior"] = init_mlp(
            prior_key,
            config.deterministic_dim,
            config.hidden_dim,
            2 * config.stochastic_dim,
        )
    else:
        params["prior"] = init_imf(
            prior_key,
            config.stochastic_dim,
            config.deterministic_dim,
            config.hidden_dim,
            depth=2,
        )
    return params


def initial_state(config: DreamerConfig, batch_size: int, *, dtype=jnp.float32) -> RSSMState:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return RSSMState(
        jnp.zeros((batch_size, config.deterministic_dim), dtype=dtype),
        jnp.zeros((batch_size, config.stochastic_dim), dtype=dtype),
    )


def preprocess_observation(observation: Array) -> Array:
    value = jnp.asarray(observation)
    if value.dtype == jnp.uint8:
        value = value.astype(jnp.float32) / 255.0
    elif not jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.float32)
    return value


def encode(params: Params, observation: Array, config: DreamerConfig) -> Array:
    observation = preprocess_observation(observation)
    if observation.shape[-len(config.observation_shape) :] != config.observation_shape:
        raise ValueError("observation has the wrong trailing shape")
    flat = observation.reshape((*observation.shape[: -len(config.observation_shape)], -1))
    return mlp(params["encoder"], flat)


def decode(params: Params, feature: Array, config: DreamerConfig) -> Array:
    flat = mlp(params["decoder"], feature)
    return flat.reshape((*feature.shape[:-1], *config.observation_shape))


def _normal_stats(raw: Array, config: DreamerConfig) -> DiagonalNormal:
    mean, raw_std = jnp.split(raw, 2, axis=-1)
    std = config.min_std + (config.max_std - config.min_std) * jax.nn.sigmoid(raw_std)
    return DiagonalNormal(mean, std)


def posterior(
    params: Params,
    deterministic: Array,
    embedding: Array,
    config: DreamerConfig,
) -> DiagonalNormal:
    raw = mlp(params["posterior"], jnp.concatenate((deterministic, embedding), axis=-1))
    return _normal_stats(raw, config)


def prior_distribution(
    params: Params,
    deterministic: Array,
    config: DreamerConfig,
) -> DiagonalNormal:
    if config.prior != "gaussian":
        raise ValueError("prior_distribution is defined only for the Gaussian prior")
    return _normal_stats(mlp(params["prior"], deterministic), config)


def transition_deterministic(
    params: Params,
    state: RSSMState,
    action: Array,
    config: DreamerConfig,
) -> Array:
    if action.shape != (*state.stochastic.shape[:-1], config.action_dim):
        raise ValueError("action has the wrong shape")
    inputs = jnp.concatenate((state.stochastic, action), axis=-1)
    return gru(params["recurrence"], inputs, state.deterministic)


def sample_prior(
    params: Params,
    deterministic: Array,
    key: Array | None,
    config: DreamerConfig,
    *,
    noise: Array | None = None,
) -> tuple[Array, DiagonalNormal | None]:
    if noise is None and key is None:
        raise ValueError("key is required when noise is omitted")
    if config.prior == "gaussian":
        distribution = prior_distribution(params, deterministic, config)
        if noise is None:
            noise = jax.random.normal(key, distribution.mean.shape, dtype=distribution.mean.dtype)
        return distribution.mean + distribution.std * noise, distribution
    return sample_imf_one_step(params["prior"], deterministic, key, noise=noise), None


def observe_step(
    params: Params,
    observation: Array,
    previous_action: Array,
    state: RSSMState,
    key: Array,
    config: DreamerConfig,
) -> tuple[RSSMState, DiagonalNormal, DiagonalNormal | None]:
    deterministic = transition_deterministic(params, state, previous_action, config)
    distribution = posterior(params, deterministic, encode(params, observation, config), config)
    stochastic = distribution.mean + distribution.std * jax.random.normal(
        key, distribution.mean.shape, dtype=distribution.mean.dtype
    )
    prior = prior_distribution(params, deterministic, config) if config.prior == "gaussian" else None
    return RSSMState(deterministic, stochastic), distribution, prior


def observe_sequence(
    params: Params,
    observations: Array,
    actions: Array,
    key: Array,
    config: DreamerConfig,
    *,
    start: RSSMState | None = None,
) -> SequenceStates:
    expected_ndim = 2 + len(config.observation_shape)
    if observations.ndim != expected_ndim or observations.shape[2:] != config.observation_shape:
        raise ValueError("observations must have shape [batch, time, *observation_shape]")
    if actions.shape != (*observations.shape[:2], config.action_dim):
        raise ValueError("actions must have shape [batch, time, action_dim]")
    batch, time = actions.shape[:2]
    state = initial_state(config, batch) if start is None else start
    keys = jax.random.split(key, time)

    def scan_step(carry: RSSMState, inputs: tuple[Array, Array, Array]):
        observation, action, step_key = inputs
        next_state, post, prior = observe_step(
            params, observation, action, carry, step_key, config
        )
        if prior is None:
            prior_mean = jnp.zeros_like(post.mean)
            prior_std = jnp.zeros_like(post.std)
        else:
            prior_mean, prior_std = prior
        outputs = (
            next_state.deterministic,
            next_state.stochastic,
            post.mean,
            post.std,
            prior_mean,
            prior_std,
        )
        return next_state, outputs

    _, outputs = jax.lax.scan(
        scan_step,
        state,
        (
            jnp.swapaxes(observations, 0, 1),
            jnp.swapaxes(actions, 0, 1),
            keys,
        ),
    )
    deterministic, stochastic, post_mean, post_std, prior_mean, prior_std = (
        jnp.swapaxes(value, 0, 1) for value in outputs
    )
    return SequenceStates(
        RSSMState(deterministic, stochastic),
        DiagonalNormal(post_mean, post_std),
        prior_mean if config.prior == "gaussian" else None,
        prior_std if config.prior == "gaussian" else None,
    )


def predict_reward(params: Params, feature: Array) -> Array:
    return mlp(params["reward"], feature)[..., 0]


def predict_continuation_logits(params: Params, feature: Array) -> Array:
    return mlp(params["continuation"], feature)[..., 0]


def gaussian_kl(q: DiagonalNormal, p: DiagonalNormal, *, stop_q: bool = False) -> Array:
    q_mean = jax.lax.stop_gradient(q.mean) if stop_q else q.mean
    q_std = jax.lax.stop_gradient(q.std) if stop_q else q.std
    return jnp.sum(
        jnp.log(p.std / q_std)
        + (jnp.square(q_std) + jnp.square(q_mean - p.mean)) / (2.0 * jnp.square(p.std))
        - 0.5,
        axis=-1,
    )


def representation_kl(distribution: DiagonalNormal) -> Array:
    variance = jnp.square(distribution.std)
    return 0.5 * jnp.sum(
        jnp.square(distribution.mean) + variance - jnp.log(variance) - 1.0,
        axis=-1,
    )


def _binary_cross_entropy_with_logits(logits: Array, labels: Array) -> Array:
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def overshooting_loss(
    params: Params,
    sequence: SequenceStates,
    actions: Array,
    key: Array,
    config: DreamerConfig,
) -> Array:
    horizon = min(config.overshooting_horizon, actions.shape[1] - 1)
    if horizon < 2:
        return jnp.asarray(0.0, dtype=actions.dtype)
    terms: list[Array] = []
    time = actions.shape[1]
    for start_index in range(time - 2):
        imagined = RSSMState(
            sequence.states.deterministic[:, start_index],
            sequence.states.stochastic[:, start_index],
        )
        maximum = min(horizon, time - 1 - start_index)
        for distance in range(1, maximum + 1):
            step_key = jax.random.fold_in(key, start_index * (horizon + 1) + distance)
            prior_sample_key, target_noise_key, loss_key = jax.random.split(step_key, 3)
            deterministic = transition_deterministic(
                params, imagined, actions[:, start_index + distance], config
            )
            stochastic, prior = sample_prior(
                params, deterministic, prior_sample_key, config
            )
            imagined = RSSMState(deterministic, stochastic)
            if distance < 2:
                continue
            target = DiagonalNormal(
                sequence.posterior.mean[:, start_index + distance],
                sequence.posterior.std[:, start_index + distance],
            )
            if prior is not None:
                terms.append(jnp.mean(jnp.maximum(gaussian_kl(target, prior, stop_q=True), config.kl_free_nats)))
            else:
                target_sample = target.mean + target.std * jax.random.normal(
                    target_noise_key, target.mean.shape, dtype=target.mean.dtype
                )
                terms.append(
                    improved_meanflow_loss(
                        params["prior"],
                        jax.lax.stop_gradient(target_sample),
                        deterministic,
                        loss_key,
                        boundary_fraction=config.imf_boundary_fraction,
                    )
                )
    return jnp.mean(jnp.stack(terms)) if terms else jnp.asarray(0.0, dtype=actions.dtype)


def world_model_loss(
    params: Params,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
) -> WorldModelLoss:
    required = ("observations", "actions", "rewards", "continuations")
    if any(name not in batch for name in required):
        raise KeyError(f"batch must contain {required}")
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    continuations = batch["continuations"]
    if rewards.shape != actions.shape[:2] or continuations.shape != actions.shape[:2]:
        raise ValueError("rewards and continuations must have shape [batch, time]")
    sequence_key, prior_key, overshooting_key = jax.random.split(key, 3)
    sequence = observe_sequence(params, observations, actions, sequence_key, config)
    feature = sequence.states.feature
    targets = preprocess_observation(observations)
    reconstruction = jnp.mean(jnp.square(decode(params, feature, config) - targets))
    reward = jnp.mean(jnp.square(predict_reward(params, feature) - rewards))
    continuation = jnp.mean(
        _binary_cross_entropy_with_logits(
            predict_continuation_logits(params, feature), continuations
        )
    )
    representation = jnp.mean(representation_kl(sequence.posterior))
    if config.prior == "gaussian":
        prior_distribution_value = DiagonalNormal(sequence.prior_mean, sequence.prior_std)
        prior = jnp.mean(
            jnp.maximum(
                gaussian_kl(sequence.posterior, prior_distribution_value), config.kl_free_nats
            )
        )
    else:
        prior = improved_meanflow_loss(
            params["prior"],
            jax.lax.stop_gradient(
                sequence.states.stochastic.reshape((-1, config.stochastic_dim))
            ),
            sequence.states.deterministic.reshape((-1, config.deterministic_dim)),
            prior_key,
            boundary_fraction=config.imf_boundary_fraction,
        )
    overshooting = overshooting_loss(params, sequence, actions, overshooting_key, config)
    total = (
        config.reconstruction_scale * reconstruction
        + config.reward_scale * reward
        + config.continuation_scale * continuation
        + config.prior_scale * prior
        + config.representation_scale * representation
        + config.overshooting_scale * overshooting
    )
    return WorldModelLoss(
        total, reconstruction, reward, continuation, prior, representation, overshooting
    )


jit_observe_sequence = partial(jax.jit, static_argnames=("config",))(observe_sequence)
jit_world_model_loss = partial(jax.jit, static_argnames=("config",))(world_model_loss)
