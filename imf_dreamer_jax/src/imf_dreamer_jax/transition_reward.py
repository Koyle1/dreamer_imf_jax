"""Direct transition-conditioned reward fitting with frozen iMF dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .nn import Params, clip_by_global_norm
from .optim import adam_update, init_adam
from .types import AdamState, AgentParams, AgentState
from .world_model import (
    init_transition_reward_head,
    observe_sequence,
    predict_direct_transition_reward,
)


Batch = Mapping[str, Array]


@dataclass(frozen=True)
class TransitionRewardConfig:
    """Optimizer settings for the isolated scalar reward MLP."""

    learning_rate: float = 3e-4
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be finite and positive")


class TransitionRewardState(NamedTuple):
    """A reward head with a fresh, independent Adam time base."""

    params: Params
    optimizer: AdamState


class TransitionRewardLoss(NamedTuple):
    """Masked scalar-MSE details for one posterior sequence batch."""

    loss: Array
    predictions: Array
    targets: Array
    mask: Array


class TransitionRewardMetrics(NamedTuple):
    """Training diagnostics returned by one isolated head update."""

    loss: Array
    grad_norm: Array


def init_transition_reward_state(
    config: DreamerConfig, key: Array
) -> TransitionRewardState:
    """Create a direct reward head and Adam state starting at step zero."""

    _validate_world_config(config)
    params = init_transition_reward_head(config, key)
    return TransitionRewardState(params, init_adam(params))


def _validate_world_config(config: DreamerConfig) -> None:
    if config.prior != "imf" or not config.imf_trajectory_enabled:
        raise ValueError("direct transition reward fitting requires trajectory iMF")
    if config.reward_loss != "mse" or config.reward_prediction_horizon != 0:
        raise ValueError("direct transition reward fitting requires scalar one-step MSE")


def _aligned_transition_inputs(
    feature: Array,
    actions: Array,
    is_first: Array,
) -> tuple[Array, Array, Array]:
    """Return reset-safe ``(previous, action, next)`` aligned to reward tokens."""

    previous = jnp.concatenate(
        (jnp.zeros_like(feature[:, :1]), feature[:, :-1]), axis=1
    )
    reset = jnp.asarray(is_first, dtype=jnp.bool_)[..., None]
    previous = jnp.where(reset, jnp.zeros_like(previous), previous)
    aligned_actions = jnp.where(reset, jnp.zeros_like(actions), actions)
    return previous, aligned_actions, feature


def _loss_mask(batch: Batch, config: DreamerConfig) -> Array:
    actions = batch["actions"]
    if "loss_mask" in batch:
        mask = jnp.asarray(batch["loss_mask"], dtype=actions.dtype)
    else:
        mask = jnp.ones(actions.shape[:2], dtype=actions.dtype)
        if config.burn_in:
            mask = mask.at[:, : config.burn_in].set(0.0)
    if mask.shape != actions.shape[:2]:
        raise ValueError("loss_mask must have shape [batch, time]")
    return mask


def transition_reward_loss(
    reward_params: Params,
    frozen_world_model: Params,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
) -> TransitionRewardLoss:
    """Fit only the direct head to observed one-step rewards with scalar MSE."""

    _validate_world_config(config)
    for name in ("observations", "actions", "rewards"):
        if name not in batch:
            raise KeyError(f"reward batch is missing {name!r}")
    if "reward_transition" in frozen_world_model:
        raise ValueError("frozen_world_model must not contain a transition reward head")
    actions = batch["actions"]
    targets = jnp.asarray(batch["rewards"], dtype=actions.dtype)
    if targets.shape != actions.shape[:2]:
        raise ValueError("rewards must have shape [batch, time]")
    is_first = jnp.asarray(
        batch.get("is_first", jnp.zeros(actions.shape[:2], dtype=jnp.bool_)),
        dtype=jnp.bool_,
    )
    if is_first.shape != actions.shape[:2]:
        raise ValueError("is_first must have shape [batch, time]")
    mask = _loss_mask(batch, config)
    sequence = observe_sequence(
        frozen_world_model,
        batch["observations"],
        actions,
        key,
        config,
        is_first=is_first,
    )
    feature = jax.lax.stop_gradient(sequence.states.feature)
    previous, aligned_actions, next_feature = _aligned_transition_inputs(
        feature, actions, is_first
    )
    previous, aligned_actions, next_feature = (
        jax.lax.stop_gradient(value)
        for value in (previous, aligned_actions, next_feature)
    )
    model = {**frozen_world_model, "reward_transition": reward_params}
    predictions = predict_direct_transition_reward(
        model, previous, aligned_actions, next_feature, config
    )
    squared_error = jnp.square(predictions - targets)
    denominator = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, mask.dtype))
    loss = jnp.sum(squared_error * mask) / denominator
    return TransitionRewardLoss(loss, predictions, targets, mask)


def train_transition_reward_step(
    state: TransitionRewardState,
    frozen_world_model: Params,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: TransitionRewardConfig,
) -> tuple[TransitionRewardState, TransitionRewardMetrics]:
    """Apply one Adam update to the reward head and no source parameter."""

    def loss_fn(params: Params) -> tuple[Array, TransitionRewardLoss]:
        details = transition_reward_loss(
            params, frozen_world_model, batch, key, config
        )
        return details.loss, details

    (_, details), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    gradients, grad_norm = clip_by_global_norm(gradients, objective.grad_clip)
    params, optimizer = adam_update(
        state.params,
        gradients,
        state.optimizer,
        learning_rate=objective.learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    return TransitionRewardState(params, optimizer), TransitionRewardMetrics(
        details.loss, grad_norm
    )


def attach_transition_reward_head(
    state: AgentState,
    reward_state: TransitionRewardState,
) -> AgentState:
    """Install a trained direct head without changing any source subtree."""

    if "reward_transition" in state.params.world_model:
        raise ValueError("transition reward head is already attached")
    head = reward_state.params
    world = {**state.params.world_model, "reward_transition": head}
    first = {
        **state.model_optimizer.first_moment,
        "reward_transition": jax.tree_util.tree_map(jnp.zeros_like, head),
    }
    second = {
        **state.model_optimizer.second_moment,
        "reward_transition": jax.tree_util.tree_map(jnp.zeros_like, head),
    }
    model_optimizer = AdamState(
        state.model_optimizer.step, first, second
    )
    return AgentState(
        AgentParams(world, state.params.actor, state.params.critic),
        model_optimizer,
        state.actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        state.world_model_teacher,
    )


jit_train_transition_reward_step = partial(
    jax.jit, static_argnames=("config", "objective")
)(train_transition_reward_step)


__all__ = [
    "TransitionRewardConfig",
    "TransitionRewardLoss",
    "TransitionRewardMetrics",
    "TransitionRewardState",
    "attach_transition_reward_head",
    "init_transition_reward_state",
    "jit_train_transition_reward_step",
    "train_transition_reward_step",
    "transition_reward_loss",
]
