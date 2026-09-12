"""Decision-aligned training primitives for trajectory world models.

This module keeps the interventions discovered by the Reacher audit explicit:
short-horizon advantage consistency, generated/corrupted context exposure, and
endpoint prediction.  All additions are opt-in and leave ``world_model_loss``
unchanged, which preserves old checkpoints and baseline objective semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .imf import imf_outputs
from .nn import Params, clip_by_global_norm
from .optim import adam_update
from .trajectory import (
    corrupt_trajectory,
    sample_trajectory_schedule,
    trajectory_imf_loss,
)
from .types import (
    AdvantageConsistencyLoss,
    AgentParams,
    AgentState,
    PolicyConsistencyLoss,
    RSSMState,
    RunningRMSState,
    SequenceStates,
)
from .world_model import (
    observe_sequence,
    predict_continuation_logits,
    predict_reward,
    sample_prior,
    trajectory_deterministic_conditions,
    transition_deterministic,
    world_model_loss,
)


Batch = Mapping[str, Array]


@dataclass(frozen=True)
class PolicyConsistencyConfig:
    """Opt-in decision and exposure objective settings.

    ``training_context_length`` is a contract check rather than padding: the
    caller must actually supply sequences at least this long.
    """

    advantage_consistency_scale: float = 0.0
    advantage_horizons: tuple[int, ...] = (1, 3, 5)
    advantage_magnitude_scale: float = 1.0
    advantage_ranking_scale: float = 1.0
    advantage_flat_scale: float = 0.1
    advantage_huber_delta: float = 1.0
    advantage_tie_tolerance: float = 1e-4
    advantage_normalization_epsilon: float = 1e-3
    exposure_meanflow_scale: float = 0.0
    endpoint_scale: float = 0.0
    generated_context_probability: float = 0.0
    context_corruption_max: float = 0.0
    training_context_length: int = 1

    def __post_init__(self) -> None:
        nonnegative = (
            "advantage_consistency_scale",
            "advantage_magnitude_scale",
            "advantage_ranking_scale",
            "advantage_flat_scale",
            "advantage_tie_tolerance",
            "exposure_meanflow_scale",
            "endpoint_scale",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("advantage_huber_delta", "advantage_normalization_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.advantage_horizons:
            raise ValueError("advantage_horizons must be nonempty")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.advantage_horizons
        ):
            raise ValueError("advantage_horizons must contain positive integers")
        if tuple(sorted(set(self.advantage_horizons))) != self.advantage_horizons:
            raise ValueError("advantage_horizons must be unique and increasing")
        for name in ("generated_context_probability", "context_corruption_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if (
            not isinstance(self.training_context_length, int)
            or isinstance(self.training_context_length, bool)
            or self.training_context_length <= 0
        ):
            raise ValueError("training_context_length must be a positive integer")


def init_running_rms(*, dtype: jnp.dtype = jnp.float32) -> RunningRMSState:
    return RunningRMSState(
        jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
    )


def update_running_rms(
    state: RunningRMSState,
    values: Array,
    mask: Array | None = None,
) -> RunningRMSState:
    """Accumulate a stopped RMS without per-minibatch renormalization."""

    values = jax.lax.stop_gradient(jnp.asarray(values))
    if mask is None:
        mask = jnp.ones_like(values)
    else:
        mask = jnp.broadcast_to(jnp.asarray(mask, dtype=values.dtype), values.shape)
    count = jnp.sum(mask)
    square_sum = jnp.sum(jnp.square(values) * mask)
    total_count = state.count + count
    mean_square = jnp.where(
        total_count > 0.0,
        (state.mean_square * state.count + square_sum) / total_count,
        state.mean_square,
    )
    return RunningRMSState(mean_square, total_count)


def running_rms_value(state: RunningRMSState, *, epsilon: float) -> Array:
    return jax.lax.stop_gradient(jnp.sqrt(state.mean_square + epsilon**2))


def _pseudo_huber(value: Array, delta: float) -> Array:
    scaled = value / delta
    return delta**2 * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)


def advantage_consistency_loss(
    predicted_returns: Array,
    target_returns: Array,
    *,
    mask: Array | None = None,
    normalization_scale: Array | float = 1.0,
    magnitude_scale: float = 1.0,
    ranking_scale: float = 1.0,
    flat_scale: float = 0.1,
    huber_delta: float = 1.0,
    tie_tolerance: float = 1e-4,
) -> AdvantageConsistencyLoss:
    """Match action advantages, ordering, and flat regions across horizons.

    Inputs end in ``[candidate, horizon]``.  Centering across candidates makes
    this objective insensitive to action-independent value offsets.  Ranking
    is applied only to simulator-informative pairs, while flat simulator rows
    explicitly suppress hallucinated action dependence.
    """

    predicted = jnp.asarray(predicted_returns)
    target = jax.lax.stop_gradient(jnp.asarray(target_returns))
    if predicted.shape != target.shape or predicted.ndim < 3:
        raise ValueError(
            "predicted_returns and target_returns must share [..., candidate, horizon]"
        )
    if predicted.shape[-2] < 2:
        raise ValueError("advantage consistency requires at least two candidates")
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.asarray(normalization_scale), 1e-8))
    target_advantage = target - jnp.mean(target, axis=-2, keepdims=True)
    predicted_advantage = predicted - jnp.mean(predicted, axis=-2, keepdims=True)
    residual = (predicted_advantage - target_advantage) / scale
    magnitude_per = jnp.mean(_pseudo_huber(residual, huber_delta), axis=-2)

    leading_horizon_shape = predicted.shape[:-2] + (predicted.shape[-1],)
    if mask is None:
        horizon_mask = jnp.ones(leading_horizon_shape, dtype=predicted.dtype)
    else:
        horizon_mask = jnp.broadcast_to(
            jnp.asarray(mask, dtype=predicted.dtype), leading_horizon_shape
        )
    denominator = jnp.maximum(jnp.sum(horizon_mask), 1.0)
    magnitude = jnp.sum(magnitude_per * horizon_mask) / denominator

    predicted_delta = (
        predicted_advantage[..., :, None, :]
        - predicted_advantage[..., None, :, :]
    )
    target_delta = (
        target_advantage[..., :, None, :] - target_advantage[..., None, :, :]
    )
    candidate_count = predicted.shape[-2]
    upper = jnp.triu(
        jnp.ones((candidate_count, candidate_count), dtype=predicted.dtype), k=1
    )
    pair_mask = (
        (jnp.abs(target_delta) > tie_tolerance).astype(predicted.dtype)
        * upper.reshape((1,) * (predicted.ndim - 2) + upper.shape + (1,))
        * horizon_mask[..., None, None, :]
    )
    signed_margin = jnp.sign(target_delta) * predicted_delta / scale
    ranking_per = jax.nn.softplus(-signed_margin)
    pair_count = jnp.sum(pair_mask)
    ranking = jnp.sum(ranking_per * pair_mask) / jnp.maximum(pair_count, 1.0)

    target_range = jnp.max(target, axis=-2) - jnp.min(target, axis=-2)
    flat_mask = (target_range <= tie_tolerance).astype(predicted.dtype) * horizon_mask
    flat_per = jnp.mean(
        _pseudo_huber(predicted_advantage / scale, huber_delta), axis=-2
    )
    flat = jnp.sum(flat_per * flat_mask) / jnp.maximum(jnp.sum(flat_mask), 1.0)
    total = magnitude_scale * magnitude + ranking_scale * ranking + flat_scale * flat
    possible_pairs = jnp.maximum(
        jnp.sum(horizon_mask) * candidate_count * (candidate_count - 1) / 2.0,
        1.0,
    )
    return AdvantageConsistencyLoss(
        total,
        magnitude,
        ranking,
        flat,
        predicted_advantage,
        target_advantage,
        pair_count / possible_pairs,
    )


def _previous_states(sequence: SequenceStates) -> RSSMState:
    return RSSMState(
        jnp.concatenate(
            (
                jnp.zeros_like(sequence.states.deterministic[:, :1]),
                sequence.states.deterministic[:, :-1],
            ),
            axis=1,
        ),
        jnp.concatenate(
            (
                jnp.zeros_like(sequence.states.stochastic[:, :1]),
                sequence.states.stochastic[:, :-1],
            ),
            axis=1,
        ),
    )


def model_candidate_returns(
    params: Params,
    sequence: SequenceStates,
    action_sequences: Array,
    key: Array,
    config: DreamerConfig,
    *,
    horizons: tuple[int, ...],
) -> Array:
    """Roll out fixed candidate suffixes with common noise across candidates."""

    actions = jnp.asarray(action_sequences)
    if actions.ndim != 5:
        raise ValueError(
            "action_sequences must have [batch, time, candidate, step, action] shape"
        )
    batch, time, candidates, maximum_horizon, action_dim = actions.shape
    if action_dim != config.action_dim:
        raise ValueError("action_sequences have the wrong action dimension")
    if not horizons or max(horizons) > maximum_horizon:
        raise ValueError("requested horizons exceed the action suffix")
    previous = jax.tree_util.tree_map(jax.lax.stop_gradient, _previous_states(sequence))
    base_count = batch * time

    def repeat_candidates(value: Array) -> Array:
        flat = value.reshape((base_count, value.shape[-1]))
        return jnp.repeat(flat, candidates, axis=0)

    state = RSSMState(
        repeat_candidates(previous.deterministic),
        repeat_candidates(previous.stochastic),
    )
    cumulative = jnp.zeros((base_count * candidates,), dtype=actions.dtype)
    survival = jnp.ones_like(cumulative)
    selected: list[Array] = []
    noise_keys = jax.random.split(key, maximum_horizon)
    for offset in range(maximum_horizon):
        action = actions[..., offset, :].reshape((base_count * candidates, action_dim))
        deterministic = transition_deterministic(params, state, action, config)
        base_noise = jax.random.normal(
            noise_keys[offset],
            (base_count, config.stochastic_dim),
            dtype=actions.dtype,
        )
        common_noise = jnp.repeat(base_noise, candidates, axis=0)
        stochastic, _ = sample_prior(
            params, deterministic, None, config, noise=common_noise
        )
        state = RSSMState(deterministic, stochastic)
        reward = predict_reward(params, state.feature, config)
        continuation = jax.nn.sigmoid(
            predict_continuation_logits(params, state.feature)
        )
        cumulative = cumulative + (config.discount**offset) * survival * reward
        survival = survival * continuation
        if offset + 1 in horizons:
            selected.append(cumulative)
    stacked = jnp.stack(selected, axis=-1)
    return stacked.reshape((batch, time, candidates, len(horizons)))


def trajectory_advantage_consistency_loss(
    params: Params,
    sequence: SequenceStates,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: PolicyConsistencyConfig,
    *,
    normalization_scale: Array | float,
) -> AdvantageConsistencyLoss:
    required = (
        "advantage_action_sequences",
        "advantage_target_returns",
        "advantage_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"advantage consistency batch is missing {missing}")
    predicted = model_candidate_returns(
        params,
        sequence,
        batch["advantage_action_sequences"],
        key,
        config,
        horizons=objective.advantage_horizons,
    )
    targets = batch["advantage_target_returns"]
    if targets.shape != predicted.shape:
        raise ValueError("advantage_target_returns have the wrong shape")
    mask = batch["advantage_mask"]
    if mask.shape == predicted.shape[:2]:
        mask = mask[..., None]
    if mask.shape != predicted.shape[:2] + (predicted.shape[-1],):
        raise ValueError("advantage_mask must be [batch, time] or [batch, time, horizon]")
    return advantage_consistency_loss(
        predicted,
        targets,
        mask=mask,
        normalization_scale=normalization_scale,
        magnitude_scale=objective.advantage_magnitude_scale,
        ranking_scale=objective.advantage_ranking_scale,
        flat_scale=objective.advantage_flat_scale,
        huber_delta=objective.advantage_huber_delta,
        tie_tolerance=objective.advantage_tie_tolerance,
    )


def generated_trajectory_history(
    params: Params,
    actions: Array,
    is_first: Array,
    noise: Array,
    config: DreamerConfig,
) -> Array:
    """Generate a reset-safe latent history under replay action suffixes."""

    if actions.ndim != 3 or actions.shape[-1] != config.action_dim:
        raise ValueError("actions must have [batch, time, action_dim] shape")
    if noise.shape != (*actions.shape[:2], config.stochastic_dim):
        raise ValueError("noise has the wrong generated-history shape")
    if is_first.shape != actions.shape[:2]:
        raise ValueError("is_first must match action leading axes")
    batch = actions.shape[0]
    state = RSSMState(
        jnp.zeros((batch, config.deterministic_dim), dtype=actions.dtype),
        jnp.zeros((batch, config.stochastic_dim), dtype=actions.dtype),
    )

    def step(carry: RSSMState, inputs: tuple[Array, Array, Array]):
        action, first, step_noise = inputs
        reset = first[:, None]
        carry = RSSMState(
            jnp.where(reset, jnp.zeros_like(carry.deterministic), carry.deterministic),
            jnp.where(reset, jnp.zeros_like(carry.stochastic), carry.stochastic),
        )
        action = jnp.where(reset, jnp.zeros_like(action), action)
        deterministic = transition_deterministic(params, carry, action, config)
        stochastic, _ = sample_prior(
            params, deterministic, None, config, noise=step_noise
        )
        next_state = RSSMState(deterministic, stochastic)
        return next_state, stochastic

    _, history = jax.lax.scan(
        step,
        state,
        (
            jnp.swapaxes(actions, 0, 1),
            jnp.swapaxes(is_first, 0, 1),
            jnp.swapaxes(noise, 0, 1),
        ),
    )
    return jnp.swapaxes(history, 0, 1)


def trajectory_exposure_loss(
    params: Params,
    sequence: SequenceStates,
    batch: Batch,
    loss_mask: Array,
    key: Array,
    config: DreamerConfig,
    objective: PolicyConsistencyConfig,
) -> tuple[Array, Array]:
    """Train on generated/noisy contexts and direct clean endpoint targets."""

    if not config.imf_trajectory_enabled or config.prior != "imf":
        raise ValueError("trajectory exposure controls require trajectory iMF")
    actions = batch["actions"]
    is_first = jnp.asarray(
        batch.get("is_first", jnp.zeros(actions.shape[:2], dtype=jnp.bool_)),
        dtype=jnp.bool_,
    )
    if actions.shape[1] < objective.training_context_length:
        raise ValueError("batch is shorter than training_context_length")
    target = jax.lax.stop_gradient(sequence.states.stochastic)
    query_key, schedule_key, generation_key, mixture_key = jax.random.split(key, 4)
    query_noise = jax.random.normal(query_key, target.shape, dtype=target.dtype)
    generation_noise = jax.random.normal(
        generation_key, target.shape, dtype=target.dtype
    )
    schedule = sample_trajectory_schedule(
        schedule_key,
        target.shape[0],
        target.shape[1],
        mode="corrupted_context",
        token_mask=loss_mask,
        dtype=target.dtype,
        boundary_fraction=config.imf_boundary_fraction,
        time_mean=config.imf_time_mean,
        time_std=config.imf_time_std,
        history_noise_max=objective.context_corruption_max,
    )
    corrupted = corrupt_trajectory(target, query_noise, schedule.history_t)
    generated = jax.lax.stop_gradient(
        generated_trajectory_history(
            params, actions, is_first, generation_noise, config
        )
    )
    use_generated = jax.random.bernoulli(
        mixture_key,
        objective.generated_context_probability,
        (target.shape[0], 1, 1),
    )
    history = jnp.where(use_generated, generated, corrupted)
    history_times = jnp.where(use_generated, jnp.zeros_like(schedule.history_t), schedule.history_t)
    conditions = trajectory_deterministic_conditions(
        params, history, history_times, actions, is_first, config
    )
    meanflow = trajectory_imf_loss(
        params["prior"],
        target,
        conditions,
        jax.random.fold_in(query_key, 1),
        noise=query_noise,
        r=schedule.r,
        t=schedule.t,
        token_mask=schedule.loss_mask,
        adaptive_power=config.imf_adaptive_power,
        adaptive_epsilon=config.imf_adaptive_epsilon,
        meanflow_scale=config.imf_meanflow_scale,
        velocity_scale=config.imf_velocity_scale,
        signal_weight_floor=config.imf_signal_weight_floor,
        signal_weight_scale=config.imf_signal_weight_scale,
        condition_gradient_scale=config.imf_condition_gradient_scale,
        boundary_velocity_supervision=True,
    )
    batch_size, steps, stochastic_dim = target.shape
    flat_noise = query_noise.reshape((batch_size * steps, stochastic_dim))
    flat_conditions = conditions.reshape((batch_size * steps, conditions.shape[-1]))
    zeros = jnp.zeros((batch_size * steps, 1), dtype=target.dtype)
    ones = jnp.ones_like(zeros)
    velocity, _ = imf_outputs(
        params["prior"], flat_noise, flat_conditions, zeros, ones
    )
    endpoint = flat_noise - velocity
    target_flat = target.reshape(endpoint.shape)
    target_scale = jax.lax.stop_gradient(
        jnp.sqrt(jnp.mean(jnp.square(target_flat)) + objective.advantage_normalization_epsilon**2)
    )
    endpoint_per_token = jnp.mean(
        _pseudo_huber(
            (endpoint - target_flat) / jnp.maximum(target_scale, 1e-8),
            objective.advantage_huber_delta,
        ),
        axis=-1,
    ).reshape(loss_mask.shape)
    endpoint_loss = jnp.sum(endpoint_per_token * loss_mask) / jnp.maximum(
        jnp.sum(loss_mask), 1.0
    )
    return meanflow, endpoint_loss


def _zero_advantage(dtype: jnp.dtype) -> AdvantageConsistencyLoss:
    zero = jnp.asarray(0.0, dtype=dtype)
    empty = jnp.zeros((1, 2, 1), dtype=dtype)
    return AdvantageConsistencyLoss(zero, zero, zero, zero, empty, empty, zero)


def policy_consistent_world_model_loss(
    params: Params,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: PolicyConsistencyConfig,
    *,
    advantage_normalization: Array | float = 1.0,
    shortcut_teacher_params: Params | None = None,
) -> PolicyConsistencyLoss:
    """Base world-model loss plus independently ablatable policy terms."""

    base = world_model_loss(
        params,
        batch,
        key,
        config,
        shortcut_teacher_params=shortcut_teacher_params,
    )
    zero = jnp.asarray(0.0, dtype=base.total.dtype)
    if (
        objective.advantage_consistency_scale == 0.0
        and objective.exposure_meanflow_scale == 0.0
        and objective.endpoint_scale == 0.0
    ):
        return PolicyConsistencyLoss(base.total, base, _zero_advantage(base.total.dtype), zero, zero)
    sequence = observe_sequence(
        params,
        batch["observations"],
        batch["actions"],
        jax.random.fold_in(key, 0xA11CE),
        config,
        is_first=batch.get("is_first"),
    )
    loss_mask = jnp.asarray(
        batch.get("loss_mask", jnp.ones(batch["actions"].shape[:2])),
        dtype=base.total.dtype,
    )
    advantage = _zero_advantage(base.total.dtype)
    if objective.advantage_consistency_scale > 0.0:
        advantage = trajectory_advantage_consistency_loss(
            params,
            sequence,
            batch,
            jax.random.fold_in(key, 0xAD0),
            config,
            objective,
            normalization_scale=advantage_normalization,
        )
    exposure = zero
    endpoint = zero
    if objective.exposure_meanflow_scale > 0.0 or objective.endpoint_scale > 0.0:
        exposure, endpoint = trajectory_exposure_loss(
            params,
            sequence,
            batch,
            loss_mask,
            jax.random.fold_in(key, 0xE0),
            config,
            objective,
        )
    total = (
        base.total
        + objective.advantage_consistency_scale * advantage.total
        + objective.exposure_meanflow_scale * exposure
        + objective.endpoint_scale * endpoint
    )
    return PolicyConsistencyLoss(total, base, advantage, exposure, endpoint)


def train_policy_consistent_world_model(
    state: AgentState,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    objective: PolicyConsistencyConfig,
    advantage_rms: RunningRMSState,
) -> tuple[AgentState, PolicyConsistencyLoss, RunningRMSState]:
    """One clipped Adam update with a persistent advantage RMS."""

    next_rms = advantage_rms
    if objective.advantage_consistency_scale > 0.0:
        targets = jax.lax.stop_gradient(batch["advantage_target_returns"])
        target_advantages = targets - jnp.mean(targets, axis=-2, keepdims=True)
        mask = batch["advantage_mask"]
        if mask.shape == targets.shape[:2]:
            mask = mask[..., None]
        next_rms = update_running_rms(
            advantage_rms,
            target_advantages,
            jnp.asarray(mask)[..., None, :],
        )
    scale = running_rms_value(
        next_rms, epsilon=objective.advantage_normalization_epsilon
    )

    def loss_fn(model_params: Params):
        details = policy_consistent_world_model_loss(
            model_params,
            batch,
            key,
            config,
            objective,
            advantage_normalization=scale,
            shortcut_teacher_params=state.world_model_teacher,
        )
        return details.total, details

    (_, details), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
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
    updated = AgentState(
        params,
        optimizer,
        state.actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        state.world_model_teacher,
    )
    return updated, details, next_rms


jit_train_policy_consistent_world_model = partial(
    jax.jit, static_argnames=("config", "objective")
)(train_policy_consistent_world_model)


__all__ = [
    "PolicyConsistencyConfig",
    "advantage_consistency_loss",
    "generated_trajectory_history",
    "init_running_rms",
    "jit_train_policy_consistent_world_model",
    "model_candidate_returns",
    "policy_consistent_world_model_loss",
    "running_rms_value",
    "train_policy_consistent_world_model",
    "trajectory_advantage_consistency_loss",
    "trajectory_exposure_loss",
    "update_running_rms",
]
