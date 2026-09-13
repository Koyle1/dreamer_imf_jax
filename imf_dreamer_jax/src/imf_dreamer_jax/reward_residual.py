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
    """Objective settings for the frozen-world-model reward residual."""

    horizons: tuple[int, ...] = (1, 3, 5)
    huber_delta: float = 1.0
    normalization_epsilon: float = 1e-3
    objective: str = "uncentered_pseudo_huber"
    control_scale: float = 1.0

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
        if self.objective not in {
            "uncentered_pseudo_huber",
            "dense_quadratic_control",
        }:
            raise ValueError(
                "objective must be 'uncentered_pseudo_huber' or "
                "'dense_quadratic_control'"
            )
        if not math.isfinite(self.control_scale) or self.control_scale < 0.0:
            raise ValueError("control_scale must be finite and nonnegative")
        if self.objective == "dense_quadratic_control" and horizons != tuple(
            range(1, horizons[-1] + 1)
        ):
            raise ValueError(
                "dense_quadratic_control requires every prefix horizon from 1 to H"
            )


class ActionRewardResidualLoss(NamedTuple):
    total: Array
    uncentered_return: Array
    control: Array


class DenseRewardObjectiveLoss(NamedTuple):
    """Components of the exact dense-prefix and control-alignment objective."""

    total: Array
    dense_prefix: Array
    control: Array
    terminal_return_errors: Array


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


def discounted_prefix_quadratic_matrix(
    horizon: int,
    discount: float,
    *,
    horizon_weights: Array | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Return ``Q = C.T @ W @ C`` for all discounted reward prefixes.

    ``C[h, t] = discount**t`` when ``t <= h`` and zero otherwise.  Therefore
    ``d.T @ Q @ d`` is exactly the weighted sum of squared discounted-prefix
    errors for a per-step reward-error vector ``d``.
    """

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not math.isfinite(discount) or discount <= 0.0:
        raise ValueError("discount must be finite and positive")
    powers = jnp.power(jnp.asarray(discount, dtype=dtype), jnp.arange(horizon))
    cumulative = jnp.tril(jnp.ones((horizon, horizon), dtype=dtype)) * powers[None]
    if horizon_weights is None:
        weights = jnp.ones((horizon,), dtype=dtype)
    else:
        weights = jnp.asarray(horizon_weights, dtype=dtype)
        if weights.shape != (horizon,):
            raise ValueError("horizon_weights must have shape [horizon]")
    return cumulative.T @ (weights[:, None] * cumulative)


def dense_prefix_quadratic_form(
    reward_errors: Array,
    discount: float,
    *,
    horizon_weights: Array | None = None,
) -> Array:
    """Evaluate the trace-form dense-prefix loss without forming prefixes."""

    errors = jnp.asarray(reward_errors)
    if errors.ndim < 1 or errors.shape[-1] == 0:
        raise ValueError("reward_errors must end in a nonempty step dimension")
    matrix = discounted_prefix_quadratic_matrix(
        errors.shape[-1],
        discount,
        horizon_weights=horizon_weights,
        dtype=errors.dtype,
    )
    return jnp.einsum("...t,tu,...u->...", errors, matrix, errors)


def best_candidate_relative_error_penalties(
    predicted_returns: Array,
    target_returns: Array,
) -> Array:
    """Return the exactly simplified best-versus-rest hinge penalties.

    For each leading row, the target-best candidate ``i*`` is selected at the
    final horizon.  The result is ``[E_i - E_i*]_+`` with a structural zero at
    ``i*``.  This equals
    ``[Delta_i - (predicted_gap_i)]_+`` algebraically; no approximation or
    statistical assumption is used.
    """

    predicted = jnp.asarray(predicted_returns)
    target = jax.lax.stop_gradient(jnp.asarray(target_returns))
    if predicted.shape != target.shape or predicted.ndim < 2:
        raise ValueError(
            "predicted_returns and target_returns must share [..., candidate]"
        )
    if predicted.shape[-1] < 2:
        raise ValueError("control alignment requires at least two candidates")
    errors = predicted - target
    best = jnp.argmax(target, axis=-1)
    best_error = jnp.take_along_axis(errors, best[..., None], axis=-1)
    penalties = jax.nn.relu(errors - best_error)
    best_mask = jax.nn.one_hot(best, predicted.shape[-1], dtype=predicted.dtype)
    return penalties * (1.0 - best_mask)


def dense_reward_objective(
    predicted_prefix_returns: Array,
    target_prefix_returns: Array,
    *,
    mask: Array | None = None,
    normalization_scale: Array | float = 1.0,
    control_scale: float = 1.0,
) -> DenseRewardObjectiveLoss:
    """Exact dense-prefix quadratic plus simplified terminal control loss.

    Inputs end in ``[candidate, horizon]`` and must contain every prefix from
    one through ``H``.  Since these prefix returns are already materialized,
    their squared residual is the direct evaluation of ``tr(D Q D.T)``.  The
    control component uses only terminal return errors and target-best indices;
    explicit target gaps cancel exactly.
    """

    predicted = jnp.asarray(predicted_prefix_returns)
    target = jax.lax.stop_gradient(jnp.asarray(target_prefix_returns))
    if predicted.shape != target.shape or predicted.ndim < 3:
        raise ValueError(
            "predicted_prefix_returns and target_prefix_returns must share "
            "[..., candidate, horizon]"
        )
    if predicted.shape[-2] < 2 or predicted.shape[-1] < 1:
        raise ValueError("dense reward objective requires candidates and horizons")
    if not math.isfinite(control_scale) or control_scale < 0.0:
        raise ValueError("control_scale must be finite and nonnegative")
    scale = jax.lax.stop_gradient(
        jnp.maximum(jnp.asarray(normalization_scale), 1e-8)
    )
    errors = (predicted - target) / scale
    horizon_shape = predicted.shape[:-2] + (predicted.shape[-1],)
    if mask is None:
        horizon_mask = jnp.ones(horizon_shape, dtype=predicted.dtype)
    else:
        horizon_mask = jnp.broadcast_to(
            jnp.asarray(mask, dtype=predicted.dtype), horizon_shape
        )
    dense_per_horizon = jnp.mean(jnp.square(errors), axis=-2)
    dense_denominator = jnp.maximum(jnp.sum(horizon_mask), 1.0)
    dense_prefix = jnp.sum(dense_per_horizon * horizon_mask) / dense_denominator

    terminal_errors = errors[..., -1]
    terminal_targets = target[..., -1]
    penalties = best_candidate_relative_error_penalties(
        terminal_errors + terminal_targets,
        terminal_targets,
    )
    terminal_mask = horizon_mask[..., -1]
    candidate_count = predicted.shape[-2]
    control_denominator = jnp.maximum(
        jnp.sum(terminal_mask) * (candidate_count - 1), 1.0
    )
    control = jnp.sum(penalties * terminal_mask[..., None]) / control_denominator
    total = dense_prefix + control_scale * control
    return DenseRewardObjectiveLoss(total, dense_prefix, control, terminal_errors)


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
    if objective.objective == "dense_quadratic_control":
        dense = dense_reward_objective(
            predicted,
            targets,
            mask=mask,
            normalization_scale=normalization_scale,
            control_scale=objective.control_scale,
        )
        return ActionRewardResidualLoss(dense.total, dense.dense_prefix, dense.control)
    return_loss = residual_return_regression_loss(
        predicted,
        targets,
        mask=mask,
        normalization_scale=normalization_scale,
        huber_delta=objective.huber_delta,
    )
    return ActionRewardResidualLoss(return_loss, return_loss, jnp.zeros_like(return_loss))


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
    # Keep the candidate-independent normalization used by the earlier
    # residual studies; dense-horizon experiments change only label coverage.
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
    "DenseRewardObjectiveLoss",
    "action_reward_residual_loss",
    "attach_action_reward_residual",
    "best_candidate_relative_error_penalties",
    "dense_prefix_quadratic_form",
    "dense_reward_objective",
    "discounted_prefix_quadratic_matrix",
    "jit_train_action_reward_residual",
    "residual_return_regression_loss",
    "train_action_reward_residual",
]
