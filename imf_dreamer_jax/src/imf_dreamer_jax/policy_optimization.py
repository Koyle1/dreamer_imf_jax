"""Continuous-policy optimization primitives used by actor diagnostics.

The main Dreamer actor uses a likelihood-ratio gradient.  This module keeps
its robust return scale explicit (rather than extending ``AgentState`` and
breaking old checkpoints) and provides a separate, conservative multi-action
MPO update for experiments that require ranked actions from the same state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .nn import Params
from .types import DiagonalNormal, PyTree


class ReturnScaleState(NamedTuple):
    """EMA state for the 5th--95th percentile return range."""

    percentile_range: Array
    initialized: Array


class SafeMPOState(NamedTuple):
    """Frozen reference actor and its refresh counter."""

    reference_actor: PyTree
    updates_since_refresh: Array


class SafeMPOObjective(NamedTuple):
    """Decomposition of a positive-weight, decoupled-KL MPO objective."""

    loss: Array
    policy_objective: Array
    mean_kl: Array
    std_kl: Array
    elite_fraction: Array
    effective_sample_size: Array
    normalized_weights: Array


class SafeMPOCandidates(NamedTuple):
    """Four actions sampled from the same reference-policy state."""

    actions: Array
    pre_tanh: Array


@dataclass(frozen=True)
class SafeMPOConfig:
    """Small continuous-action MPO variant used as a diagnostic fallback.

    Four actions are sampled from one frozen reference policy for every state.
    Only the best two receive a positive softmax weight.  The old-to-new
    Gaussian KL is split into location and scale terms so variance collapse
    cannot hide behind a small joint average.
    """

    samples_per_state: int = 4
    elite_count: int = 2
    temperature: float = 1.0
    mean_kl_limit: float = 0.01
    std_kl_limit: float = 0.001
    mean_kl_penalty: float = 10.0
    std_kl_penalty: float = 10.0
    reference_refresh_interval: int = 100

    def __post_init__(self) -> None:
        if self.samples_per_state != 4:
            raise ValueError("safe MPO requires exactly four actions per state")
        if self.elite_count != 2:
            raise ValueError("safe MPO requires the best two of four actions")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        for name in (
            "mean_kl_limit",
            "std_kl_limit",
            "mean_kl_penalty",
            "std_kl_penalty",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not isinstance(self.reference_refresh_interval, int)
            or isinstance(self.reference_refresh_interval, bool)
            or self.reference_refresh_interval <= 0
        ):
            raise ValueError("reference_refresh_interval must be a positive integer")


def init_return_scale_state(
    *, dtype: jnp.dtype = jnp.float32
) -> ReturnScaleState:
    """Create an uninitialized robust return-scale accumulator."""

    return ReturnScaleState(
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(False),
    )


def update_return_scale(
    state: ReturnScaleState,
    returns: Array,
    *,
    decay: float = 0.99,
) -> tuple[ReturnScaleState, Array]:
    """Update and return ``max(1, EMA(P95(returns)-P5(returns)))``.

    The statistic is gradient-isolated because it conditions optimization but
    is not itself part of the policy objective.  The first update initializes
    the EMA from data instead of biasing it toward zero.
    """

    if returns.size == 0:
        raise ValueError("returns must be nonempty")
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("decay must be finite and in [0, 1)")
    batch_range = jax.lax.stop_gradient(
        jnp.percentile(returns, 95.0) - jnp.percentile(returns, 5.0)
    )
    percentile_range = jnp.where(
        state.initialized,
        decay * state.percentile_range + (1.0 - decay) * batch_range,
        batch_range,
    )
    updated = ReturnScaleState(
        percentile_range,
        jnp.asarray(True),
    )
    return updated, jax.lax.stop_gradient(jnp.maximum(1.0, percentile_range))


def normalized_reinforce_objective(
    log_probs: Array,
    advantages: Array,
    weights: Array,
    return_scale: Array,
) -> Array:
    """Fixed-denominator, normalized likelihood-ratio objective."""

    if log_probs.shape != advantages.shape or log_probs.shape != weights.shape:
        raise ValueError("log_probs, advantages, and weights must match")
    if return_scale.ndim != 0:
        raise ValueError("return_scale must be scalar")
    normalized = jax.lax.stop_gradient(advantages / return_scale)
    return jnp.mean(log_probs * normalized * jax.lax.stop_gradient(weights))


def decoupled_reference_kl(
    reference: DiagonalNormal,
    current: DiagonalNormal,
) -> tuple[Array, Array]:
    """Return location and scale parts of ``KL(reference || current)``."""

    if reference.mean.shape != reference.std.shape:
        raise ValueError("reference mean and std shapes must match")
    if current.mean.shape != current.std.shape:
        raise ValueError("current mean and std shapes must match")
    if reference.mean.shape != current.mean.shape:
        raise ValueError("reference and current distributions must match")
    mean_kl = 0.5 * jnp.sum(
        jnp.square((reference.mean - current.mean) / current.std), axis=-1
    )
    std_kl = jnp.sum(
        jnp.log(current.std / reference.std)
        + 0.5 * (jnp.square(reference.std / current.std) - 1.0),
        axis=-1,
    )
    return mean_kl, jnp.maximum(std_kl, 0.0)


def positive_elite_weights(
    scores: Array,
    *,
    elite_count: int = 2,
    temperature: float = 1.0,
) -> Array:
    """Positive normalized weights on each state's highest-scoring actions."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, candidates]")
    if elite_count <= 0 or elite_count > scores.shape[-1]:
        raise ValueError("elite_count must select a nonempty candidate subset")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    order = jnp.argsort(scores, axis=-1)
    elite_indices = order[:, -elite_count:]
    elite_mask = jnp.sum(
        jax.nn.one_hot(elite_indices, scores.shape[-1], dtype=scores.dtype),
        axis=1,
    )
    centered = (scores - jnp.max(scores, axis=-1, keepdims=True)) / temperature
    masked = jnp.where(elite_mask > 0.0, centered, -jnp.inf)
    return jax.lax.stop_gradient(jax.nn.softmax(masked, axis=-1))


def sample_safe_mpo_candidates(
    reference_distribution: DiagonalNormal,
    key: Array,
    config: SafeMPOConfig,
) -> SafeMPOCandidates:
    """Draw exactly four candidates per state from one reference policy."""

    if reference_distribution.mean.ndim != 2:
        raise ValueError("reference distribution must have shape [batch, action]")
    if reference_distribution.mean.shape != reference_distribution.std.shape:
        raise ValueError("reference mean and std shapes must match")
    noise = jax.random.normal(
        key,
        (
            reference_distribution.mean.shape[0],
            config.samples_per_state,
            reference_distribution.mean.shape[1],
        ),
        dtype=reference_distribution.mean.dtype,
    )
    pre_tanh = (
        reference_distribution.mean[:, None, :]
        + reference_distribution.std[:, None, :] * noise
    )
    return SafeMPOCandidates(
        jax.lax.stop_gradient(jnp.tanh(pre_tanh)),
        jax.lax.stop_gradient(pre_tanh),
    )


def safe_multi_action_mpo_objective(
    new_log_probs: Array,
    scores: Array,
    reference_distribution: DiagonalNormal,
    current_distribution: DiagonalNormal,
    config: SafeMPOConfig,
) -> SafeMPOObjective:
    """Positive weighted regression with separate old-to-new KL constraints."""

    if new_log_probs.shape != scores.shape:
        raise ValueError("new_log_probs and scores must match")
    if scores.shape[-1] != config.samples_per_state:
        raise ValueError("candidate dimension does not match samples_per_state")
    normalized_weights = positive_elite_weights(
        scores,
        elite_count=config.elite_count,
        temperature=config.temperature,
    )
    policy_objective = jnp.mean(
        jnp.sum(normalized_weights * new_log_probs, axis=-1)
    )
    mean_kl_per_state, std_kl_per_state = decoupled_reference_kl(
        reference_distribution, current_distribution
    )
    mean_kl = jnp.mean(mean_kl_per_state)
    std_kl = jnp.mean(std_kl_per_state)
    mean_violation = jax.nn.relu(mean_kl - config.mean_kl_limit)
    std_violation = jax.nn.relu(std_kl - config.std_kl_limit)
    loss = (
        -policy_objective
        + config.mean_kl_penalty * mean_violation
        + config.std_kl_penalty * std_violation
    )
    effective_sample_size = jnp.mean(
        1.0 / jnp.sum(jnp.square(normalized_weights), axis=-1)
    )
    return SafeMPOObjective(
        loss,
        policy_objective,
        mean_kl,
        std_kl,
        jnp.asarray(config.elite_count / config.samples_per_state),
        effective_sample_size,
        normalized_weights,
    )


def init_safe_mpo_state(actor_params: Params) -> SafeMPOState:
    """Snapshot the reference policy for conservative action regression."""

    reference = jax.tree_util.tree_map(
        lambda value: jax.lax.stop_gradient(jnp.array(value, copy=True)),
        actor_params,
    )
    return SafeMPOState(reference, jnp.asarray(0, dtype=jnp.int32))


def update_safe_mpo_reference(
    state: SafeMPOState,
    actor_params: Params,
    *,
    refresh_interval: int,
) -> SafeMPOState:
    """Refresh the frozen reference after exactly ``refresh_interval`` updates."""

    if (
        not isinstance(refresh_interval, int)
        or isinstance(refresh_interval, bool)
        or refresh_interval <= 0
    ):
        raise ValueError("refresh_interval must be a positive integer")
    next_count = state.updates_since_refresh + 1
    should_refresh = next_count >= refresh_interval
    reference = jax.tree_util.tree_map(
        lambda old, new: jax.lax.stop_gradient(
            jnp.where(should_refresh, new, old)
        ),
        state.reference_actor,
        actor_params,
    )
    return SafeMPOState(
        reference,
        jnp.where(should_refresh, 0, next_count),
    )


def policy_gradient_cosine(left: PyTree, right: PyTree) -> Array:
    """Cosine similarity between two policy-gradient PyTrees."""

    left_leaves = [jnp.ravel(value) for value in jax.tree_util.tree_leaves(left)]
    right_leaves = [jnp.ravel(value) for value in jax.tree_util.tree_leaves(right)]
    if len(left_leaves) != len(right_leaves):
        raise ValueError("gradient trees must have the same structure")
    left_flat = jnp.concatenate(left_leaves)
    right_flat = jnp.concatenate(right_leaves)
    denominator = jnp.linalg.norm(left_flat) * jnp.linalg.norm(right_flat)
    return jnp.where(
        denominator > 0.0,
        jnp.vdot(left_flat, right_flat) / denominator,
        jnp.asarray(0.0, dtype=left_flat.dtype),
    )


__all__ = [
    "ReturnScaleState",
    "SafeMPOConfig",
    "SafeMPOCandidates",
    "SafeMPOObjective",
    "SafeMPOState",
    "decoupled_reference_kl",
    "init_return_scale_state",
    "init_safe_mpo_state",
    "normalized_reinforce_objective",
    "policy_gradient_cosine",
    "positive_elite_weights",
    "safe_multi_action_mpo_objective",
    "sample_safe_mpo_candidates",
    "update_return_scale",
    "update_safe_mpo_reference",
]
