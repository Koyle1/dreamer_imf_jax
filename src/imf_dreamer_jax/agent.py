"""Actor, critic, acting, and functional training state."""

from __future__ import annotations

from functools import partial
import math
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .nn import Params, clip_by_global_norm, init_mlp, mlp
from .optim import adam_update, init_adam
from .types import (
    ActorCriticMetrics,
    AgentParams,
    AgentState,
    Imagination,
    RSSMState,
    WorldModelLoss,
)
from .world_model import (
    initial_state,
    init_world_model,
    observe_step,
    predict_continuation_logits,
    predict_reward,
    sample_prior,
    transition_deterministic,
    world_model_loss,
)


def init_actor(config: DreamerConfig, key: Array) -> Params:
    return init_mlp(key, config.feature_dim, config.hidden_dim, 2 * config.action_dim)


def init_critic(config: DreamerConfig, key: Array) -> Params:
    return init_mlp(key, config.feature_dim, config.hidden_dim, 1)


def actor(
    params: Params,
    feature: Array,
    key: Array | None,
    config: DreamerConfig,
    *,
    deterministic: bool = False,
) -> tuple[Array, Array]:
    raw = mlp(params, feature)
    mean, raw_std = jnp.split(raw, 2, axis=-1)
    std = 0.05 + 0.95 * jax.nn.sigmoid(raw_std)
    if deterministic:
        pre_tanh = mean
    else:
        if key is None:
            raise ValueError("key is required for stochastic actions")
        pre_tanh = mean + std * jax.random.normal(key, mean.shape, dtype=mean.dtype)
    action = jnp.tanh(pre_tanh)
    entropy = jnp.sum(
        0.5 + 0.5 * math.log(2.0 * math.pi) + jnp.log(std), axis=-1
    )
    return action, entropy


def critic(params: Params, feature: Array) -> Array:
    return mlp(params, feature)[..., 0]


def create_agent(config: DreamerConfig, key: Array) -> AgentState:
    """Create parameters and independent Adam states from one PRNG key."""

    model_key, actor_key, critic_key = jax.random.split(key, 3)
    params = AgentParams(
        init_world_model(config, model_key),
        init_actor(config, actor_key),
        init_critic(config, critic_key),
    )
    return AgentState(
        params,
        init_adam(params.world_model),
        init_adam(params.actor),
        init_adam(params.critic),
    )


def act(
    params: AgentParams,
    observation: Array,
    previous_action: Array,
    state: RSSMState,
    key: Array,
    config: DreamerConfig,
    *,
    deterministic: bool = False,
) -> tuple[Array, RSSMState]:
    posterior_key, actor_key = jax.random.split(key)
    next_state, _, _ = observe_step(
        params.world_model,
        observation,
        previous_action,
        state,
        posterior_key,
        config,
    )
    action, _ = actor(
        params.actor, next_state.feature, actor_key, config, deterministic=deterministic
    )
    return action, next_state


def lambda_returns(
    rewards: Array,
    values: Array,
    continuations: Array,
    *,
    discount: float,
    lambda_: float,
) -> Array:
    if rewards.shape != continuations.shape:
        raise ValueError("rewards and continuations must have identical shape")
    if values.shape != (*rewards.shape[:-1], rewards.shape[-1] + 1):
        raise ValueError("values must have one additional horizon element")
    accumulator = values[..., -1]
    reversed_returns: list[Array] = []
    for index in range(rewards.shape[-1] - 1, -1, -1):
        bootstrap = (1.0 - lambda_) * values[..., index + 1] + lambda_ * accumulator
        accumulator = rewards[..., index] + discount * continuations[..., index] * bootstrap
        reversed_returns.append(accumulator)
    return jnp.stack(tuple(reversed(reversed_returns)), axis=-1)


def imagine(
    params: AgentParams,
    start: RSSMState,
    key: Array,
    config: DreamerConfig,
    *,
    horizon: int | None = None,
) -> Imagination:
    horizon = config.imagination_horizon if horizon is None else horizon
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    step_keys = jax.random.split(key, horizon)

    def step(state: RSSMState, step_key: Array):
        actor_key, prior_key = jax.random.split(step_key)
        action_value, entropy = actor(params.actor, state.feature, actor_key, config)
        deterministic = transition_deterministic(
            params.world_model, state, action_value, config
        )
        stochastic, _ = sample_prior(
            params.world_model, deterministic, prior_key, config
        )
        next_state = RSSMState(deterministic, stochastic)
        output = (
            next_state.feature,
            action_value,
            predict_reward(params.world_model, next_state.feature),
            jax.nn.sigmoid(
                predict_continuation_logits(params.world_model, next_state.feature)
            ),
            entropy,
        )
        return next_state, output

    _, outputs = jax.lax.scan(step, start, step_keys)
    features, actions, rewards, continuations, entropies = (
        jnp.swapaxes(value, 0, 1) for value in outputs
    )
    features = jnp.concatenate((start.feature[:, None], features), axis=1)
    return Imagination(features, actions, rewards, continuations, entropies)


def train_world_model(
    state: AgentState,
    batch: Mapping[str, Array],
    key: Array,
    config: DreamerConfig,
) -> tuple[AgentState, WorldModelLoss]:
    """Run one clipped Adam update of the recurrent world model."""

    def objective(model_params: Params):
        losses = world_model_loss(model_params, batch, key, config)
        return losses.total, losses

    (_, losses), gradients = jax.value_and_grad(objective, has_aux=True)(
        state.params.world_model
    )
    gradients, _ = clip_by_global_norm(gradients, config.grad_clip)
    model_params, optimizer = adam_update(
        state.params.world_model,
        gradients,
        state.model_optimizer,
        learning_rate=config.model_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    params = AgentParams(model_params, state.params.actor, state.params.critic)
    return AgentState(params, optimizer, state.actor_optimizer, state.critic_optimizer), losses


def train_actor_critic(
    state: AgentState,
    start: RSSMState,
    key: Array,
    config: DreamerConfig,
) -> tuple[AgentState, ActorCriticMetrics]:
    """Update actor through imagined dynamics, then fit the critic to lambda returns."""

    start = jax.tree_util.tree_map(jax.lax.stop_gradient, start)
    actor_key, critic_rollout_key = jax.random.split(key)

    def actor_objective(actor_params: Params) -> Array:
        params = AgentParams(state.params.world_model, actor_params, state.params.critic)
        imagined = imagine(params, start, actor_key, config)
        values = critic(state.params.critic, imagined.features)
        returns = lambda_returns(
            imagined.rewards,
            values,
            imagined.continuations,
            discount=config.discount,
            lambda_=config.lambda_,
        )
        return -(
            jnp.mean(returns) + config.actor_entropy_scale * jnp.mean(imagined.entropies)
        )

    actor_loss, actor_gradients = jax.value_and_grad(actor_objective)(state.params.actor)
    actor_gradients, _ = clip_by_global_norm(actor_gradients, config.grad_clip)
    actor_params, actor_optimizer = adam_update(
        state.params.actor,
        actor_gradients,
        state.actor_optimizer,
        learning_rate=config.actor_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    rollout_params = AgentParams(state.params.world_model, actor_params, state.params.critic)
    imagined = imagine(rollout_params, start, critic_rollout_key, config)
    target_values = critic(state.params.critic, imagined.features)
    targets = jax.lax.stop_gradient(
        lambda_returns(
            imagined.rewards,
            target_values,
            imagined.continuations,
            discount=config.discount,
            lambda_=config.lambda_,
        )
    )
    critic_features = jax.lax.stop_gradient(imagined.features[:, :-1])

    def critic_objective(critic_params: Params) -> Array:
        predictions = critic(critic_params, critic_features)
        return jnp.mean(jnp.square(predictions - targets))

    critic_loss, critic_gradients = jax.value_and_grad(critic_objective)(
        state.params.critic
    )
    critic_gradients, _ = clip_by_global_norm(critic_gradients, config.grad_clip)
    critic_params, critic_optimizer = adam_update(
        state.params.critic,
        critic_gradients,
        state.critic_optimizer,
        learning_rate=config.critic_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    params = AgentParams(state.params.world_model, actor_params, critic_params)
    new_state = AgentState(
        params, state.model_optimizer, actor_optimizer, critic_optimizer
    )
    metrics = ActorCriticMetrics(actor_loss, critic_loss, jnp.mean(targets[:, 0]))
    return new_state, metrics


jit_act = partial(jax.jit, static_argnames=("config", "deterministic"))(act)
jit_imagine = partial(jax.jit, static_argnames=("config", "horizon"))(imagine)
jit_train_world_model = partial(jax.jit, static_argnames=("config",))(train_world_model)
jit_train_actor_critic = partial(jax.jit, static_argnames=("config",))(train_actor_critic)


__all__ = [
    "act",
    "actor",
    "create_agent",
    "critic",
    "imagine",
    "initial_state",
    "jit_act",
    "jit_imagine",
    "jit_train_actor_critic",
    "jit_train_world_model",
    "lambda_returns",
    "train_actor_critic",
    "train_world_model",
]

