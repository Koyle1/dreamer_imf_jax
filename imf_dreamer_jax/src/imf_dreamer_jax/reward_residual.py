"""Action-sensitive reward correction for frozen trajectory world models."""

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
from .optim import adam_update
from .policy_consistency import (
    model_candidate_returns,
    running_rms_value,
    update_running_rms,
)
from .types import AdamState, AgentParams, AgentState, RunningRMSState
from .world_model import init_action_reward_residual, observe_sequence


Batch = Mapping[str, Array]


@dataclass(frozen=True)
class ActionRewardResidualConfig:
    """Exact uncentered return objective for the causal reward residual."""

    horizons: tuple[int, ...] = (1, 3, 5)
    huber_delta: float = 1.0
    normalization_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        horizons = tuple(self.horizons)
        object.__setattr__(self, "horizons", horizons)
        if (
            not horizons
            or tuple(sorted(set(horizons))) != horizons
            or any(value <= 0 for value in horizons)
        ):
            raise ValueError("horizons must be unique increasing positive integers")
        for name in ("huber_delta", "normalization_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class ActionRewardResidualLoss(NamedTuple):
    total: Array
    uncentered_return: Array


def attach_action_reward_residual(
    state: AgentState,
    config: DreamerConfig,
    key: Array,
) -> AgentState:
    """Attach an exact-zero correction and matching zero optimizer moments."""

    if "reward_action_residual" in state.params.world_model:
        raise ValueError("action reward residual is already attached")
    residual = init_action_reward_residual(config, key)
    world = {**state.params.world_model, "reward_action_residual": residual}
    first = {
        **state.model_optimizer.first_moment,
        "reward_action_residual": jax.tree_util.tree_map(jnp.zeros_like, residual),
    }
    second = {
        **state.model_optimizer.second_moment,
        "reward_action_residual": jax.tree_util.tree_map(jnp.zeros_like, residual),
    }
    optimizer = AdamState(state.model_optimizer.step, first, second)
    teacher = state.world_model_teacher
    if teacher is not None:
        teacher = {**teacher, "reward_action_residual": residual}
    return AgentState(
        AgentParams(world, state.params.actor, state.params.critic),
        optimizer,
        state.actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        teacher,
    )


def _pseudo_huber(value: Array, delta: float) -> Array:
    scaled = value / delta
    return delta**2 * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)


def residual_return_regression_loss(
    predicted_returns: Array,
    target_returns: Array,
    *,
    mask: Array | None = None,
    normalization_scale: Array | float = 1.0,
    huber_delta: float = 1.0,
) -> Array:
    """Regress absolute candidate returns with one robust loss.

    Unlike a candidate-centered objective, this loss has no translation
    nullspace.  It is exactly the centered pseudo-Huber objective plus the
    algebraic difference between uncentered and centered pseudo-Huber losses;
    evaluating the simplified expression avoids two cancelling terms.
    """

    predicted = jnp.asarray(predicted_returns)
    target = jax.lax.stop_gradient(jnp.asarray(target_returns))
    if predicted.shape != target.shape or predicted.ndim < 3:
        raise ValueError(
            "predicted_returns and target_returns must share [..., candidate, horizon]"
        )
    if predicted.shape[-2] < 2:
        raise ValueError("residual return regression requires at least two candidates")
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    scale = jax.lax.stop_gradient(
        jnp.maximum(jnp.asarray(normalization_scale), 1e-8)
    )
    per_horizon = jnp.mean(
        _pseudo_huber((predicted - target) / scale, huber_delta), axis=-2
    )
    horizon_shape = predicted.shape[:-2] + (predicted.shape[-1],)
    if mask is None:
        horizon_mask = jnp.ones(horizon_shape, dtype=predicted.dtype)
    else:
        horizon_mask = jnp.broadcast_to(
            jnp.asarray(mask, dtype=predicted.dtype), horizon_shape
        )
    denominator = jnp.maximum(jnp.sum(horizon_mask), 1.0)
    return jnp.sum(per_horizon * horizon_mask) / denominator


def action_reward_residual_loss(
    residual: Params,
    frozen_world_model: Params,
    decision_batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: ActionRewardResidualConfig,
    *,
    normalization_scale: Array | float,
) -> ActionRewardResidualLoss:
    """Regress simulator returns through only the residual subtree."""

    if "reward_action_residual" in frozen_world_model:
        raise ValueError("frozen_world_model must not already contain a residual")
    model = {**frozen_world_model, "reward_action_residual": residual}
    sequence = observe_sequence(
        frozen_world_model,
        decision_batch["observations"],
        decision_batch["actions"],
        jax.random.fold_in(key, 1),
        config,
        is_first=decision_batch.get("is_first"),
    )
    sequence = jax.tree_util.tree_map(jax.lax.stop_gradient, sequence)
    predicted = model_candidate_returns(
        model,
        sequence,
        decision_batch["advantage_action_sequences"],
        jax.random.fold_in(key, 2),
        config,
        horizons=objective.horizons,
    )
    targets = decision_batch["advantage_target_returns"]
    if targets.shape != predicted.shape:
        raise ValueError("advantage_target_returns have the wrong shape")
    mask = decision_batch["advantage_mask"]
    if mask.shape == predicted.shape[:2]:
        mask = mask[..., None]
    if mask.shape != predicted.shape[:2] + (predicted.shape[-1],):
        raise ValueError("advantage_mask must be [batch, time] or [batch, time, horizon]")
    return_loss = residual_return_regression_loss(
        predicted,
        targets,
        mask=mask,
        normalization_scale=normalization_scale,
        huber_delta=objective.huber_delta,
    )
    return ActionRewardResidualLoss(return_loss, return_loss)


def train_action_reward_residual(
    state: AgentState,
    decision_batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: ActionRewardResidualConfig,
    advantage_rms: RunningRMSState,
) -> tuple[AgentState, ActionRewardResidualLoss, RunningRMSState]:
    """Update only the action-reward residual; every source subtree is frozen."""

    if "reward_action_residual" not in state.params.world_model:
        raise ValueError("action reward residual has not been attached")
    targets = jax.lax.stop_gradient(decision_batch["advantage_target_returns"])
    # Preserve the completed centered study's candidate-independent scale so
    # the only experimental change is removal of candidate centering.
    centered = targets - jnp.mean(targets, axis=-2, keepdims=True)
    mask = decision_batch["advantage_mask"]
    if mask.shape == targets.shape[:2]:
        mask = mask[..., None]
    next_rms = update_running_rms(
        advantage_rms,
        centered,
        jnp.asarray(mask)[..., None, :],
    )
    scale = running_rms_value(next_rms, epsilon=objective.normalization_epsilon)
    residual = state.params.world_model["reward_action_residual"]
    frozen_world = {
        name: value
        for name, value in state.params.world_model.items()
        if name != "reward_action_residual"
    }

    def loss_fn(candidate: Params):
        details = action_reward_residual_loss(
            candidate,
            frozen_world,
            decision_batch,
            key,
            config,
            objective,
            normalization_scale=scale,
        )
        return details.total, details

    (_, details), gradient = jax.value_and_grad(loss_fn, has_aux=True)(residual)
    gradient, _ = clip_by_global_norm(gradient, config.grad_clip)
    updated_residual, residual_optimizer = adam_update(
        residual,
        gradient,
        AdamState(
            state.model_optimizer.step,
            state.model_optimizer.first_moment["reward_action_residual"],
            state.model_optimizer.second_moment["reward_action_residual"],
        ),
        learning_rate=config.model_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    world = {**state.params.world_model, "reward_action_residual": updated_residual}
    first = {
        **state.model_optimizer.first_moment,
        "reward_action_residual": residual_optimizer.first_moment,
    }
    second = {
        **state.model_optimizer.second_moment,
        "reward_action_residual": residual_optimizer.second_moment,
    }
    updated = AgentState(
        AgentParams(world, state.params.actor, state.params.critic),
        AdamState(residual_optimizer.step, first, second),
        state.actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        state.world_model_teacher,
    )
    return updated, details, next_rms


jit_train_action_reward_residual = partial(
    jax.jit, static_argnames=("config", "objective")
)(train_action_reward_residual)


__all__ = [
    "ActionRewardResidualConfig",
    "ActionRewardResidualLoss",
    "action_reward_residual_loss",
    "attach_action_reward_residual",
    "jit_train_action_reward_residual",
    "residual_return_regression_loss",
    "train_action_reward_residual",
]
