"""Frozen-dynamics reward adaptation with multi-token prediction.

The module isolates the failure mode found by the Reacher audit: dynamics are
never updated here.  A reward head is trained on posterior, lightly corrupted,
and lightly generated latent contexts, optionally with simulator-grounded
counterfactual advantage supervision.
"""

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
    generated_trajectory_history,
    running_rms_value,
    trajectory_advantage_consistency_loss,
    update_running_rms,
)
from .types import (
    AdvantageConsistencyLoss,
    AdamState,
    AgentParams,
    AgentState,
    RSSMState,
    RunningRMSState,
    SequenceStates,
)
from .world_model import (
    init_reward_head,
    observe_sequence,
    predict_action_reward_residual,
    predict_reward_offsets,
    reward_mtp_targets,
    reward_prediction_loss,
    trajectory_deterministic_conditions,
)


Batch = Mapping[str, Array]


@dataclass(frozen=True)
class RewardHeadObjectiveConfig:
    """Matched reward-only objective and latent-context exposure settings."""

    posterior_scale: float = 1.0
    corrupted_scale: float = 0.5
    generated_scale: float = 0.5
    context_corruption_max: float = 0.1
    generated_context_mix: float = 0.25
    advantage_scale: float = 0.0
    advantage_horizons: tuple[int, ...] = (1, 3, 5)
    advantage_magnitude_scale: float = 0.25
    advantage_ranking_scale: float = 1.0
    advantage_flat_scale: float = 0.1
    advantage_huber_delta: float = 1.0
    advantage_tie_tolerance: float = 1e-4
    advantage_normalization_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        for name in (
            "posterior_scale",
            "corrupted_scale",
            "generated_scale",
            "advantage_scale",
            "advantage_magnitude_scale",
            "advantage_ranking_scale",
            "advantage_flat_scale",
            "advantage_tie_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "context_corruption_max",
            "generated_context_mix",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("advantage_huber_delta", "advantage_normalization_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not self.advantage_horizons
            or tuple(sorted(set(self.advantage_horizons))) != self.advantage_horizons
            or any(value <= 0 for value in self.advantage_horizons)
        ):
            raise ValueError("advantage_horizons must be unique increasing positive integers")

    def advantage_objective(self) -> PolicyConsistencyConfig:
        return PolicyConsistencyConfig(
            advantage_consistency_scale=self.advantage_scale,
            advantage_horizons=self.advantage_horizons,
            advantage_magnitude_scale=self.advantage_magnitude_scale,
            advantage_ranking_scale=self.advantage_ranking_scale,
            advantage_flat_scale=self.advantage_flat_scale,
            advantage_huber_delta=self.advantage_huber_delta,
            advantage_tie_tolerance=self.advantage_tie_tolerance,
            advantage_normalization_epsilon=self.advantage_normalization_epsilon,
        )


class RewardHeadLoss(NamedTuple):
    total: Array
    posterior: Array
    corrupted: Array
    generated: Array
    advantage: AdvantageConsistencyLoss


class RewardContextPredictions(NamedTuple):
    posterior: Array
    corrupted: Array
    generated: Array
    targets: Array
    mask: Array


def _zero_advantage(dtype: jnp.dtype) -> AdvantageConsistencyLoss:
    zero = jnp.asarray(0.0, dtype=dtype)
    empty = jnp.zeros((1, 2, 1), dtype=dtype)
    return AdvantageConsistencyLoss(zero, zero, zero, zero, empty, empty, zero)


def _copy_compatible_hidden_layers(source: Params, target: Params) -> Params:
    source_layers = source["layers"]
    target_layers = list(target["layers"])
    for index in range(min(len(source_layers), len(target_layers)) - 1):
        old = source_layers[index]
        new = target_layers[index]
        if all(old[name].shape == new[name].shape for name in ("weight", "bias")):
            target_layers[index] = old
    return {"layers": tuple(target_layers)}


def reconfigure_reward_head(
    state: AgentState,
    old_config: DreamerConfig,
    new_config: DreamerConfig,
    key: Array,
    *,
    reset_output: bool = False,
) -> AgentState:
    """Resize only the reward output and preserve all other params/moments exactly."""

    if old_config.feature_dim != new_config.feature_dim or old_config.hidden_dim != new_config.hidden_dim:
        raise ValueError("reward-head reconfiguration cannot alter feature or hidden dimensions")
    old_reward = state.params.world_model["reward"]
    output = old_reward["layers"][-1]["bias"]
    if output.shape == (new_config.reward_head_output_dim,) and not reset_output:
        return state
    reward = _copy_compatible_hidden_layers(
        old_reward, init_reward_head(new_config, key)
    )
    world_model = {**state.params.world_model, "reward": reward}
    first = {
        **state.model_optimizer.first_moment,
        "reward": jax.tree_util.tree_map(jnp.zeros_like, reward),
    }
    second = {
        **state.model_optimizer.second_moment,
        "reward": jax.tree_util.tree_map(jnp.zeros_like, reward),
    }
    optimizer = AdamState(state.model_optimizer.step, first, second)
    teacher = state.world_model_teacher
    if teacher is not None:
        teacher = {**teacher, "reward": reward}
    return AgentState(
        AgentParams(world_model, state.params.actor, state.params.critic),
        optimizer,
        state.actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        teacher,
    )


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


def _context_features(
    params: Params,
    sequence: SequenceStates,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: RewardHeadObjectiveConfig,
) -> tuple[Array, Array, Array]:
    if config.prior != "imf" or not config.imf_trajectory_enabled:
        raise ValueError("reward context exposure currently requires trajectory iMF")
    actions = batch["actions"]
    is_first = jnp.asarray(
        batch.get("is_first", jnp.zeros(actions.shape[:2], dtype=jnp.bool_)),
        dtype=jnp.bool_,
    )
    posterior = jax.tree_util.tree_map(jax.lax.stop_gradient, sequence.states)
    noise_key, level_key, generated_key, mix_key = jax.random.split(key, 4)
    levels = jax.random.uniform(
        level_key,
        (*actions.shape[:2], 1),
        minval=0.0,
        maxval=objective.context_corruption_max,
        dtype=posterior.stochastic.dtype,
    )
    corrupted_z = posterior.stochastic + levels * jax.random.normal(
        noise_key, posterior.stochastic.shape, dtype=posterior.stochastic.dtype
    )
    corrupted_h = trajectory_deterministic_conditions(
        params, corrupted_z, levels, actions, is_first, config
    )
    generated_z = generated_trajectory_history(
        params,
        actions,
        is_first,
        jax.random.normal(
            generated_key,
            posterior.stochastic.shape,
            dtype=posterior.stochastic.dtype,
        ),
        config,
    )
    mix = jax.random.uniform(
        mix_key,
        (*actions.shape[:2], 1),
        minval=0.0,
        maxval=objective.generated_context_mix,
        dtype=posterior.stochastic.dtype,
    )
    mixed_z = posterior.stochastic + mix * (generated_z - posterior.stochastic)
    mixed_h = trajectory_deterministic_conditions(
        params,
        mixed_z,
        jnp.zeros_like(mix),
        actions,
        is_first,
        config,
    )
    return tuple(
        jax.lax.stop_gradient(value)
        for value in (
            posterior.feature,
            RSSMState(corrupted_h, corrupted_z).feature,
            RSSMState(mixed_h, mixed_z).feature,
        )
    )


def reward_head_loss(
    params: Params,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: RewardHeadObjectiveConfig,
    *,
    advantage_normalization: Array | float = 1.0,
    advantage_batch: Batch | None = None,
) -> RewardHeadLoss:
    """Compute the matched reward-only objective on three context regimes."""

    for name in ("observations", "actions", "rewards", "continuations"):
        if name not in batch:
            raise KeyError(f"reward batch is missing {name!r}")
    is_first = jnp.asarray(
        batch.get("is_first", jnp.zeros(batch["actions"].shape[:2], dtype=jnp.bool_)),
        dtype=jnp.bool_,
    )
    mask = _loss_mask(batch, config)
    sequence = observe_sequence(
        params,
        batch["observations"],
        batch["actions"],
        jax.random.fold_in(key, 1),
        config,
        is_first=is_first,
    )
    posterior_feature, corrupted_feature, generated_feature = _context_features(
        params,
        sequence,
        batch,
        jax.random.fold_in(key, 2),
        config,
        objective,
    )

    def supervised(feature: Array) -> Array:
        return reward_prediction_loss(
            params,
            feature,
            batch["rewards"],
            batch["continuations"],
            is_first,
            mask,
            config,
        )

    posterior_loss = supervised(posterior_feature)
    corrupted_loss = supervised(corrupted_feature)
    generated_loss = supervised(generated_feature)
    advantage = _zero_advantage(posterior_loss.dtype)
    if objective.advantage_scale > 0.0:
        decision = batch if advantage_batch is None else advantage_batch
        decision_first = decision.get("is_first")
        decision_sequence = observe_sequence(
            params,
            decision["observations"],
            decision["actions"],
            jax.random.fold_in(key, 3),
            config,
            is_first=decision_first,
        )
        decision_sequence = jax.tree_util.tree_map(
            jax.lax.stop_gradient, decision_sequence
        )
        advantage = trajectory_advantage_consistency_loss(
            params,
            decision_sequence,
            decision,
            jax.random.fold_in(key, 4),
            config,
            objective.advantage_objective(),
            normalization_scale=advantage_normalization,
        )
    total = (
        objective.posterior_scale * posterior_loss
        + objective.corrupted_scale * corrupted_loss
        + objective.generated_scale * generated_loss
        + objective.advantage_scale * advantage.total
    )
    return RewardHeadLoss(
        total, posterior_loss, corrupted_loss, generated_loss, advantage
    )


def reward_context_predictions(
    params: Params,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: RewardHeadObjectiveConfig,
) -> RewardContextPredictions:
    """Return common scalar diagnostics for the three training context regimes."""

    is_first = jnp.asarray(
        batch.get("is_first", jnp.zeros(batch["actions"].shape[:2], dtype=jnp.bool_)),
        dtype=jnp.bool_,
    )
    mask = _loss_mask(batch, config)
    sequence = observe_sequence(
        params,
        batch["observations"],
        batch["actions"],
        jax.random.fold_in(key, 1),
        config,
        is_first=is_first,
    )
    features = _context_features(
        params,
        sequence,
        batch,
        jax.random.fold_in(key, 2),
        config,
        objective,
    )
    targets, target_mask = reward_mtp_targets(
        batch["rewards"],
        batch["continuations"],
        is_first,
        mask,
        config.reward_prediction_horizon,
    )
    def predict(value: Array) -> Array:
        offsets = predict_reward_offsets(params, value, config)
        previous = jnp.concatenate(
            (jnp.zeros_like(value[:, :1]), value[:, :-1]), axis=1
        )
        reset = is_first[..., None]
        previous = jnp.where(reset, jnp.zeros_like(previous), previous)
        actions = jnp.where(reset, jnp.zeros_like(batch["actions"]), batch["actions"])
        correction = predict_action_reward_residual(
            params, previous, actions, value, config
        )
        return offsets.at[..., 0].add(correction)

    predictions = tuple(predict(value) for value in features)
    return RewardContextPredictions(*predictions, targets, target_mask)


def train_reward_head(
    state: AgentState,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: RewardHeadObjectiveConfig,
    advantage_rms: RunningRMSState,
    *,
    advantage_batch: Batch | None = None,
) -> tuple[AgentState, RewardHeadLoss, RunningRMSState]:
    """Update only the reward subtree; all other parameters/moments are restored."""

    next_rms = advantage_rms
    if objective.advantage_scale > 0.0:
        decision = batch if advantage_batch is None else advantage_batch
        targets = jax.lax.stop_gradient(decision["advantage_target_returns"])
        centered = targets - jnp.mean(targets, axis=-2, keepdims=True)
        mask = decision["advantage_mask"]
        if mask.shape == targets.shape[:2]:
            mask = mask[..., None]
        next_rms = update_running_rms(
            advantage_rms, centered, jnp.asarray(mask)[..., None, :]
        )
    scale = running_rms_value(
        next_rms, epsilon=objective.advantage_normalization_epsilon
    )

    def loss_fn(reward_params: Params):
        model = {**state.params.world_model, "reward": reward_params}
        details = reward_head_loss(
            model,
            batch,
            key,
            config,
            objective,
            advantage_normalization=scale,
            advantage_batch=advantage_batch,
        )
        return details.total, details

    (_, details), gradient = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params.world_model["reward"]
    )
    gradient, _ = clip_by_global_norm(gradient, config.grad_clip)
    reward, reward_optimizer = adam_update(
        state.params.world_model["reward"],
        gradient,
        AdamState(
            state.model_optimizer.step,
            state.model_optimizer.first_moment["reward"],
            state.model_optimizer.second_moment["reward"],
        ),
        learning_rate=config.model_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    world = {**state.params.world_model, "reward": reward}
    first = {**state.model_optimizer.first_moment, "reward": reward_optimizer.first_moment}
    second = {**state.model_optimizer.second_moment, "reward": reward_optimizer.second_moment}
    optimizer = AdamState(reward_optimizer.step, first, second)
    updated = AgentState(
        AgentParams(world, state.params.actor, state.params.critic),
        optimizer,
        state.actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        state.world_model_teacher,
    )
    return updated, details, next_rms


jit_train_reward_head = partial(
    jax.jit, static_argnames=("config", "objective")
)(train_reward_head)


__all__ = [
    "RewardContextPredictions",
    "RewardHeadLoss",
    "RewardHeadObjectiveConfig",
    "jit_train_reward_head",
    "reconfigure_reward_head",
    "reward_head_loss",
    "reward_context_predictions",
    "train_reward_head",
]
