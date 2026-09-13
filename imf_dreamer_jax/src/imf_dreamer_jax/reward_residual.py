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
    PolicyConsistencyConfig,
    running_rms_value,
    trajectory_advantage_consistency_loss,
    update_running_rms,
)
from .types import AdamState, AdvantageConsistencyLoss, AgentParams, AgentState, RunningRMSState
from .world_model import init_action_reward_residual, observe_sequence


Batch = Mapping[str, Array]


@dataclass(frozen=True)
class ActionRewardResidualConfig:
    """Minimal centered-return objective for the causal reward residual."""

    horizons: tuple[int, ...] = (1, 3, 5)
    huber_delta: float = 1.0
    tie_tolerance: float = 1e-4
    normalization_epsilon: float = 1e-3
    output_l2_scale: float = 1e-4

    def __post_init__(self) -> None:
        if (
            not self.horizons
            or tuple(sorted(set(self.horizons))) != self.horizons
            or any(value <= 0 for value in self.horizons)
        ):
            raise ValueError("horizons must be unique increasing positive integers")
        for name in ("huber_delta", "normalization_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("tie_tolerance", "output_l2_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def advantage_objective(self) -> PolicyConsistencyConfig:
        # Centered magnitude regression already penalizes wrong ordering and
        # flat targets. Avoid redundant rank and flat loss terms.
        return PolicyConsistencyConfig(
            advantage_consistency_scale=1.0,
            advantage_horizons=self.horizons,
            advantage_magnitude_scale=1.0,
            advantage_ranking_scale=0.0,
            advantage_flat_scale=0.0,
            advantage_huber_delta=self.huber_delta,
            advantage_tie_tolerance=self.tie_tolerance,
            advantage_normalization_epsilon=self.normalization_epsilon,
        )


class ActionRewardResidualLoss(NamedTuple):
    total: Array
    centered_return: Array
    output_l2: Array
    advantage: AdvantageConsistencyLoss


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


def _output_projection_l2(residual: Params) -> Array:
    output = residual["layers"][-1]
    numerator = jnp.sum(jnp.square(output["weight"])) + jnp.sum(
        jnp.square(output["bias"])
    )
    count = output["weight"].size + output["bias"].size
    return numerator / count


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
    """Regress centered simulator returns through only the residual subtree."""

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
    advantage = trajectory_advantage_consistency_loss(
        model,
        sequence,
        decision_batch,
        jax.random.fold_in(key, 2),
        config,
        objective.advantage_objective(),
        normalization_scale=normalization_scale,
    )
    output_l2 = _output_projection_l2(residual)
    total = advantage.total + objective.output_l2_scale * output_l2
    return ActionRewardResidualLoss(total, advantage.magnitude, output_l2, advantage)


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
    "train_action_reward_residual",
]
