"""Control-aware objectives for Improved MeanFlow transition models.

The primitives in this module deliberately keep three different ideas apart:

* policy-density tilting reweights *observed* transitions by a tractable
  stochastic policy density;
* planner-matched value equivalence compares Bellman backups, with a separate
  squared multi-sample estimator that applies the CVAML variance subtraction;
* direct action-chunk training predicts an observed endpoint from a start
  condition and a padded action sequence.

None of these objectives establishes actor-occupancy coverage.  In particular,
the fixed-variance Gaussian around a deterministic action is explicitly an
approximation (a deterministic policy has no Lebesgue action density), and the
behavior-density/distance fields below are empirical rejection diagnostics,
not mathematical support sets.  The CVAML-compatible label is restricted to
the stochastic squared estimator; no arbitrary robust loss is called
calibrated here.

All numerical functions are pure JAX functions.  Configuration objects contain
only immutable Python values and can be made static arguments to :func:`jax.jit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .imf import sample_imf_one_step
from .nn import Params

Reduction = Literal["none", "mean", "sum"]

EXPLICIT_STOCHASTIC_DENSITY_LABEL = "explicit_stochastic_log_probability"
APPROXIMATE_DETERMINISTIC_DENSITY_LABEL = (
    "approximate_fixed_variance_gaussian_around_deterministic_action"
)
EMPIRICAL_BEHAVIOR_REJECTION_LABEL = (
    "empirical_behavior_density_or_distance_rejection_not_a_support_set"
)
CVAML_COMPATIBLE_SQUARED_LABEL = (
    "cvaml_compatible_squared_multi_sample_variance_subtraction"
)
DIRECT_ACTION_CHUNK_ENDPOINT_LABEL = (
    "direct_action_chunk_any_step_imf_endpoint_objective"
)


@dataclass(frozen=True)
class PolicyDensityTiltConfig:
    """Settings for the normalized policy-density tilt.

    For dataset log probabilities ``l_i = log pi(a_i | s_i)``, the unclipped
    ITPO-style minibatch weight is

    ``w_i = exp(eta * l_i) / mean_j(exp(eta * l_j))``.

    Log scores are clipped before exponentiation for numerical stability.  A
    bounded-simplex projection then keeps valid weights in
    ``[minimum_weight, maximum_weight]`` while preserving mean one to machine
    precision.  Setting ``eta=0`` therefore recovers uniform weighting exactly.
    """

    eta: float = 0.0
    log_weight_clip: float = 30.0
    minimum_weight: float = 0.0
    maximum_weight: float = 20.0

    def __post_init__(self) -> None:
        for name in ("eta", "log_weight_clip", "minimum_weight", "maximum_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.eta < 0.0:
            raise ValueError("eta must be nonnegative")
        if self.log_weight_clip <= 0.0:
            raise ValueError("log_weight_clip must be positive")
        if not 0.0 <= self.minimum_weight <= 1.0:
            raise ValueError("minimum_weight must lie in [0, 1]")
        if self.maximum_weight < 1.0:
            raise ValueError("maximum_weight must be at least one")
        if self.minimum_weight > self.maximum_weight:
            raise ValueError("minimum_weight cannot exceed maximum_weight")


@dataclass(frozen=True)
class PlannerValueEquivalenceConfig:
    """Planner Bellman-backup convention used by the value-aware losses."""

    discount: float = 0.99
    normalization_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if (
            not isinstance(self.discount, (int, float))
            or isinstance(self.discount, bool)
            or not math.isfinite(float(self.discount))
            or not 0.0 <= self.discount <= 1.0
        ):
            raise ValueError("discount must be finite and lie in [0, 1]")
        if (
            not isinstance(self.normalization_epsilon, (int, float))
            or isinstance(self.normalization_epsilon, bool)
            or not math.isfinite(float(self.normalization_epsilon))
            or self.normalization_epsilon <= 0.0
        ):
            raise ValueError("normalization_epsilon must be finite and positive")


@dataclass(frozen=True)
class ActionChunkEndpointConfig:
    """Static layout and empirical rejection thresholds for endpoint batches.

    ``horizons`` declares every direct prediction length.  Actions are padded
    to ``max(horizons)`` and, when that maximum exceeds one, a binary chunk mask
    is appended to the model condition.  For ``horizons=(1,)`` no constant mask
    is appended, so the condition is exactly ``concat(start_condition, action)``.

    A mean log-density threshold is comparable across different chunk lengths;
    the joint (summed) log density is still retained as metadata.  Distance is
    aggregated by a maximum so any distant step can reject a chunk.  These are
    empirical filters only.
    """

    horizons: tuple[int, ...] = (1,)
    minimum_mean_behavior_log_density: float | None = None
    maximum_behavior_distance: float | None = None
    behavior_density_is_approximate: bool = False

    def __post_init__(self) -> None:
        try:
            horizons = tuple(self.horizons)
        except TypeError as error:
            raise ValueError("horizons must be an iterable of integers") from error
        object.__setattr__(self, "horizons", horizons)
        if (
            not horizons
            or tuple(sorted(set(horizons))) != horizons
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in horizons
            )
        ):
            raise ValueError("horizons must be unique increasing positive integers")
        threshold = self.minimum_mean_behavior_log_density
        if threshold is not None and (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
        ):
            raise ValueError(
                "minimum_mean_behavior_log_density must be finite when supplied"
            )
        distance = self.maximum_behavior_distance
        if distance is not None and (
            not isinstance(distance, (int, float))
            or isinstance(distance, bool)
            or not math.isfinite(float(distance))
            or distance < 0.0
        ):
            raise ValueError("maximum_behavior_distance must be finite and nonnegative")
        if not isinstance(self.behavior_density_is_approximate, bool):
            raise ValueError("behavior_density_is_approximate must be boolean")


class PolicyDensityTiltDetails(NamedTuple):
    """Diagnostics for one stopped-gradient density-tilt computation."""

    weights: Array
    log_weights: Array
    clipped_log_weights: Array
    effective_sample_size: Array
    clipped_fraction: Array
    valid_count: Array
    invalid_log_probability_count: Array


class PlannerValueEquivalenceLoss(NamedTuple):
    """Decomposition of a deterministic or stochastic Bellman loss."""

    loss: Array
    uncorrected_loss: Array
    variance_correction: Array
    per_example_loss: Array
    uncorrected_per_example_loss: Array
    normalized_residual: Array
    model_backup: Array
    real_target: Array
    normalization_scale: Array
    weights: Array


class BehaviorRejectionMetadata(NamedTuple):
    """Empirical chunk diagnostics; no field certifies distributional support."""

    mean_log_density: Array
    joint_log_density: Array
    maximum_distance: Array
    density_available: Array
    distance_available: Array
    density_is_approximate: Array
    rejected_by_invalid_transition: Array
    rejected_by_density: Array
    rejected_by_distance: Array
    accepted: Array


class ActionChunkEndpointBatch(NamedTuple):
    """Flat direct-endpoint examples constructed from sequential replay."""

    targets: Array
    conditions: Array
    start_conditions: Array
    action_chunks: Array
    action_mask: Array
    source_batch_index: Array
    start_index: Array
    horizon: Array
    rejection: BehaviorRejectionMetadata


class ActionChunkEndpointLoss(NamedTuple):
    """Raw one-NFE endpoint regression and its empirical sample weights."""

    loss: Array
    per_example_loss: Array
    weighted_per_example_loss: Array
    prediction: Array
    target: Array
    weights: Array
    accepted_fraction: Array
    rejection: BehaviorRejectionMetadata


def _validate_reduction(reduction: Reduction) -> None:
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _require_floating(name: str, value: Array) -> None:
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError(f"{name} must have a floating dtype")


def _masked_reduction(
    values: Array,
    weights: Array,
    reduction: Reduction,
) -> Array:
    weighted = values * weights
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return jnp.sum(weighted)
    denominator = jnp.maximum(
        jnp.sum(weights), jnp.asarray(jnp.finfo(values.dtype).tiny, values.dtype)
    )
    return jnp.sum(weighted) / denominator


def diagonal_gaussian_log_probability(
    actions: Array,
    mean: Array,
    log_std: Array | float,
) -> Array:
    """Return an exact diagonal-Gaussian log probability over the last axis.

    This is a tractable stochastic-density path on unconstrained Euclidean
    actions.  A squashed or otherwise transformed policy must provide its own
    Jacobian-corrected log probability instead of using this helper.
    """

    actions = jnp.asarray(actions)
    mean = jnp.asarray(mean, dtype=actions.dtype)
    if actions.ndim < 1 or actions.shape[-1] <= 0:
        raise ValueError("actions must end in a nonempty action dimension")
    if mean.shape != actions.shape:
        raise ValueError("mean must have the same shape as actions")
    _require_floating("actions", actions)
    log_std_array = jnp.asarray(log_std, dtype=actions.dtype)
    try:
        log_std_array = jnp.broadcast_to(log_std_array, actions.shape)
    except ValueError as error:
        raise ValueError("log_std must broadcast to the action shape") from error
    standardized = (actions - mean) * jnp.exp(-log_std_array)
    return -0.5 * jnp.sum(
        jnp.square(standardized) + 2.0 * log_std_array + math.log(2.0 * math.pi),
        axis=-1,
    )


def approximate_fixed_variance_gaussian_log_probability(
    actions: Array,
    deterministic_actions: Array,
    *,
    fixed_std: float,
) -> Array:
    """Approximate a deterministic policy by a fixed-variance Gaussian kernel.

    This wrapper is **not** the density of the deterministic policy: a Dirac
    action has no finite Lebesgue log density.  It merely evaluates
    ``Normal(deterministic_action, fixed_std**2 I)`` and must be reported with
    :data:`APPROXIMATE_DETERMINISTIC_DENSITY_LABEL`.
    """

    if (
        not isinstance(fixed_std, (int, float))
        or isinstance(fixed_std, bool)
        or not math.isfinite(float(fixed_std))
        or fixed_std <= 0.0
    ):
        raise ValueError("fixed_std must be a finite positive scalar")
    actions = jnp.asarray(actions)
    deterministic_actions = jnp.asarray(deterministic_actions, dtype=actions.dtype)
    if actions.shape != deterministic_actions.shape:
        raise ValueError("deterministic_actions must have the action shape")
    return diagonal_gaussian_log_probability(
        actions,
        deterministic_actions,
        math.log(float(fixed_std)),
    )


def _bounded_mean_one_weights(
    base_weights: Array,
    valid: Array,
    config: PolicyDensityTiltConfig,
) -> Array:
    """Project positive scores to bounded weights with valid mean exactly one."""

    if config.minimum_weight == 1.0 or config.maximum_weight == 1.0:
        return valid.astype(base_weights.dtype)

    count = jnp.sum(valid, dtype=base_weights.dtype)
    tiny = jnp.finfo(base_weights.dtype).tiny
    positive = jnp.where(valid, jnp.maximum(base_weights, tiny), 1.0)
    minimum = jnp.where(
        count > 0.0,
        jnp.min(jnp.where(valid, positive, jnp.inf)),
        1.0,
    )
    maximum = jnp.where(
        count > 0.0,
        jnp.max(jnp.where(valid, positive, 0.0)),
        1.0,
    )
    # Form these bounds in log space: ``tiny / maximum`` can itself
    # underflow in float32 before the logarithm is evaluated.
    lower_log_scale = jnp.log(tiny) - jnp.log(maximum)
    upper_log_scale = -jnp.log(minimum)

    def bisect(_, bounds):
        lower, upper = bounds
        midpoint = 0.5 * (lower + upper)
        scale = jnp.exp(midpoint)
        candidate = jnp.where(
            valid,
            jnp.clip(
                scale * base_weights,
                config.minimum_weight,
                config.maximum_weight,
            ),
            0.0,
        )
        candidate_mean = jnp.sum(candidate) / jnp.maximum(count, 1.0)
        return jnp.where(candidate_mean < 1.0, midpoint, lower), jnp.where(
            candidate_mean < 1.0, upper, midpoint
        )

    lower_log_scale, upper_log_scale = jax.lax.fori_loop(
        0,
        64,
        bisect,
        (lower_log_scale, upper_log_scale),
    )
    scale = jnp.exp(0.5 * (lower_log_scale + upper_log_scale))
    projected = jnp.where(
        valid,
        jnp.clip(
            scale * base_weights,
            config.minimum_weight,
            config.maximum_weight,
        ),
        0.0,
    )
    already_feasible = jnp.all(
        (~valid)
        | (
            (base_weights >= config.minimum_weight)
            & (base_weights <= config.maximum_weight)
        )
    )
    return jnp.where(
        already_feasible,
        jnp.where(valid, base_weights, 0.0),
        projected,
    )


def policy_density_tilt_weights(
    policy_log_probabilities: Array,
    config: PolicyDensityTiltConfig = PolicyDensityTiltConfig(),
    *,
    mask: Array | None = None,
    return_details: bool = False,
) -> Array | PolicyDensityTiltDetails:
    """Compute stable stopped weights proportional to ``pi(a|s)**eta``.

    ``policy_log_probabilities`` must come from an explicit tractable
    stochastic policy, or from the separately labeled deterministic Gaussian
    approximation above.  The normalization is over all valid elements (one
    minibatch).  Non-finite log probabilities are excluded when ``eta>0`` and
    reported in the details; ``eta=0`` recovers uniform weights on ``mask``.
    """

    log_probabilities = jnp.asarray(policy_log_probabilities)
    if log_probabilities.ndim < 1 or log_probabilities.size == 0:
        raise ValueError("policy_log_probabilities must be a nonempty array")
    _require_floating("policy_log_probabilities", log_probabilities)
    if mask is None:
        requested = jnp.ones(log_probabilities.shape, dtype=jnp.bool_)
    else:
        requested = jnp.asarray(mask, dtype=jnp.bool_)
        if requested.shape != log_probabilities.shape:
            raise ValueError("mask must have the log-probability shape")

    finite = jnp.isfinite(log_probabilities)
    invalid_count = jnp.sum(requested & ~finite, dtype=log_probabilities.dtype)
    if config.eta == 0.0:
        valid = requested
        log_weights = jnp.zeros_like(log_probabilities)
    else:
        valid = requested & finite
        log_weights = config.eta * log_probabilities
    clipped_log_weights = jnp.clip(
        log_weights, -config.log_weight_clip, config.log_weight_clip
    )
    valid_count = jnp.sum(valid, dtype=log_probabilities.dtype)
    maximum = jnp.max(jnp.where(valid, clipped_log_weights, -jnp.inf))
    maximum = jnp.where(valid_count > 0.0, maximum, 0.0)
    relative = clipped_log_weights - maximum
    relative = jnp.maximum(relative, jnp.log(jnp.finfo(relative.dtype).tiny))
    unnormalized = jnp.where(valid, jnp.exp(relative), 0.0)
    base_weights = (
        unnormalized
        * valid_count
        / jnp.maximum(jnp.sum(unnormalized), jnp.finfo(unnormalized.dtype).tiny)
    )
    if config.eta == 0.0:
        # This is the exact algebraic reduction, not a limiting numerical case.
        # Avoid routing uniform weights through logarithms and bisection.
        weights = valid.astype(log_probabilities.dtype)
    else:
        weights = _bounded_mean_one_weights(base_weights, valid, config)
    weights = jax.lax.stop_gradient(weights)

    if not return_details:
        return weights
    weight_sum = jnp.sum(weights)
    effective_sample_size = jnp.where(
        weight_sum > 0.0,
        jnp.square(weight_sum)
        / jnp.maximum(jnp.sum(jnp.square(weights)), jnp.finfo(weights.dtype).tiny),
        0.0,
    )
    at_weight_bound = valid & (
        (weights <= config.minimum_weight) | (weights >= config.maximum_weight)
    )
    score_clipped = valid & (clipped_log_weights != log_weights)
    clipped_fraction = jnp.sum(
        at_weight_bound | score_clipped, dtype=weights.dtype
    ) / jnp.maximum(valid_count, 1.0)
    return PolicyDensityTiltDetails(
        weights=weights,
        log_weights=log_weights,
        clipped_log_weights=clipped_log_weights,
        effective_sample_size=effective_sample_size,
        clipped_fraction=clipped_fraction,
        valid_count=valid_count,
        invalid_log_probability_count=invalid_count,
    )


jit_policy_density_tilt_weights = partial(
    jax.jit,
    static_argnames=("config", "return_details"),
)(policy_density_tilt_weights)


def planner_bellman_backup(
    rewards: Array,
    continuations: Array,
    next_values: Array,
    *,
    discount: float,
) -> Array:
    """Return the planner backup ``reward + discount*continuation*value``."""

    rewards = jnp.asarray(rewards)
    continuations = jnp.asarray(continuations, dtype=rewards.dtype)
    next_values = jnp.asarray(next_values, dtype=rewards.dtype)
    if rewards.shape != continuations.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards, continuations, and next_values must match")
    if not jnp.issubdtype(rewards.dtype, jnp.floating):
        raise ValueError("Bellman backup arrays must have a floating dtype")
    if (
        not isinstance(discount, (int, float))
        or isinstance(discount, bool)
        or not math.isfinite(float(discount))
        or not 0.0 <= discount <= 1.0
    ):
        raise ValueError("discount must be finite and lie in [0, 1]")
    return rewards + discount * continuations * next_values


def _loss_weights(mask: Array | None, reference: Array) -> Array:
    if mask is None:
        return jnp.ones(reference.shape, dtype=reference.dtype)
    weights = jnp.asarray(mask, dtype=reference.dtype)
    if weights.shape != reference.shape:
        raise ValueError("mask must have the Bellman-target shape")
    return jax.lax.stop_gradient(weights)


def _normalization(
    normalization_scale: Array | float,
    reference: Array,
    epsilon: float,
) -> Array:
    if isinstance(normalization_scale, (int, float)) and not isinstance(
        normalization_scale, bool
    ):
        if not math.isfinite(float(normalization_scale)) or normalization_scale <= 0.0:
            raise ValueError("normalization_scale must be finite and positive")
    scale = jnp.asarray(normalization_scale, dtype=reference.dtype)
    try:
        scale = jnp.broadcast_to(scale, reference.shape)
    except ValueError as error:
        raise ValueError(
            "normalization_scale must broadcast to the Bellman-target shape"
        ) from error
    return jax.lax.stop_gradient(jnp.maximum(scale, epsilon))


def planner_matched_bellman_residual_loss(
    model_rewards: Array,
    model_continuations: Array,
    model_next_values: Array,
    real_rewards: Array,
    real_continuations: Array,
    real_next_values: Array,
    config: PlannerValueEquivalenceConfig = PlannerValueEquivalenceConfig(),
    *,
    normalization_scale: Array | float = 1.0,
    mask: Array | None = None,
    reduction: Reduction = "mean",
    return_details: bool = False,
) -> Array | PlannerValueEquivalenceLoss:
    """Match deterministic planner Bellman backups with a stopped real target.

    Let ``B_hat = r_hat + gamma*c_hat*V_hat`` and
    ``B = r + gamma*c*V``.  The per-example objective is exactly
    ``((B_hat - stop_gradient(B)) / scale)**2``.  It is a deterministic
    squared value-equivalence auxiliary, not a CVAML claim and not a generic
    robust surrogate.
    """

    _validate_reduction(reduction)
    model_backup = planner_bellman_backup(
        model_rewards,
        model_continuations,
        model_next_values,
        discount=config.discount,
    )
    real_backup = planner_bellman_backup(
        real_rewards,
        real_continuations,
        real_next_values,
        discount=config.discount,
    )
    if model_backup.shape != real_backup.shape or model_backup.ndim < 1:
        raise ValueError("model and real Bellman backups must have one shared shape")
    real_target = jax.lax.stop_gradient(real_backup)
    scale = _normalization(
        normalization_scale, real_target, config.normalization_epsilon
    )
    normalized_residual = (model_backup - real_target) / scale
    per_example = jnp.square(normalized_residual)
    weights = _loss_weights(mask, real_target)
    loss = _masked_reduction(per_example, weights, reduction)
    if not return_details:
        return loss
    zero_correction = _masked_reduction(jnp.zeros_like(per_example), weights, reduction)
    return PlannerValueEquivalenceLoss(
        loss=loss,
        uncorrected_loss=loss,
        variance_correction=zero_correction,
        per_example_loss=per_example,
        uncorrected_per_example_loss=per_example,
        normalized_residual=normalized_residual,
        model_backup=model_backup,
        real_target=real_target,
        normalization_scale=scale,
        weights=weights,
    )


jit_planner_matched_bellman_residual_loss = partial(
    jax.jit,
    static_argnames=("config", "reduction", "return_details"),
)(planner_matched_bellman_residual_loss)


def cvaml_compatible_bellman_residual_loss(
    model_reward_samples: Array,
    model_continuation_samples: Array,
    model_next_value_samples: Array,
    real_rewards: Array,
    real_continuations: Array,
    real_next_values: Array,
    config: PlannerValueEquivalenceConfig = PlannerValueEquivalenceConfig(),
    *,
    sample_axis: int = 0,
    normalization_scale: Array | float = 1.0,
    mask: Array | None = None,
    reduction: Reduction = "mean",
    return_details: bool = False,
) -> Array | PlannerValueEquivalenceLoss:
    """Apply the CVAML-compatible correction to a squared Bellman residual.

    For ``K >= 2`` conditionally independent model backups ``Y_i`` and a
    stopped real target ``T``, this computes the appendix-consistent estimator

    ``(mean_i(Y_i) - T)**2 - sample_var(Y_i, ddof=1) / K``.

    The second term unbiasedly estimates the ``Var(Y)/K`` finite-sample bias
    of the squared sample-mean residual.  It is not clipped, so an individual
    contribution may be negative.  With one sampled real target, its variance
    remains as a model-independent additive term; this preserves model-side
    calibrated minimizers.  Callers must keep masks and normalization
    independent of the learned model distribution.  Applying the construction
    to a full stochastic Bellman backup is a mathematical extension of the
    paper's value-only ``(m, 0)`` derivation, not an exact reproduction.
    """

    _validate_reduction(reduction)
    sample_backup = planner_bellman_backup(
        model_reward_samples,
        model_continuation_samples,
        model_next_value_samples,
        discount=config.discount,
    )
    if sample_backup.ndim < 1:
        raise ValueError("model samples must include a sample axis")
    if not isinstance(sample_axis, int) or isinstance(sample_axis, bool):
        raise ValueError("sample_axis must be an integer")
    if not -sample_backup.ndim <= sample_axis < sample_backup.ndim:
        raise ValueError("sample_axis is out of range for the model samples")
    axis = sample_axis % sample_backup.ndim
    sample_count = sample_backup.shape[axis]
    if sample_count < 2:
        raise ValueError("CVAML-compatible correction requires at least two samples")

    real_backup = planner_bellman_backup(
        real_rewards,
        real_continuations,
        real_next_values,
        discount=config.discount,
    )
    expected_shape = sample_backup.shape[:axis] + sample_backup.shape[axis + 1 :]
    if real_backup.shape != expected_shape or real_backup.ndim < 1:
        raise ValueError(
            "real Bellman target must match the model shape without sample_axis"
        )
    real_target = jax.lax.stop_gradient(real_backup)
    scale = _normalization(
        normalization_scale, real_target, config.normalization_epsilon
    )
    expanded_scale = jnp.expand_dims(scale, axis=axis)
    normalized_samples = sample_backup / expanded_scale
    normalized_mean = jnp.mean(normalized_samples, axis=axis)
    normalized_target = real_target / scale
    normalized_residual = normalized_mean - normalized_target
    uncorrected_per_example = jnp.square(normalized_residual)
    variance_of_mean = jnp.var(normalized_samples, axis=axis, ddof=1) / float(
        sample_count
    )
    per_example = uncorrected_per_example - variance_of_mean
    weights = _loss_weights(mask, real_target)
    loss = _masked_reduction(per_example, weights, reduction)
    if not return_details:
        return loss
    uncorrected = _masked_reduction(uncorrected_per_example, weights, reduction)
    correction = _masked_reduction(variance_of_mean, weights, reduction)
    return PlannerValueEquivalenceLoss(
        loss=loss,
        uncorrected_loss=uncorrected,
        variance_correction=correction,
        per_example_loss=per_example,
        uncorrected_per_example_loss=uncorrected_per_example,
        normalized_residual=normalized_residual,
        model_backup=jnp.mean(sample_backup, axis=axis),
        real_target=real_target,
        normalization_scale=scale,
        weights=weights,
    )


jit_cvaml_compatible_bellman_residual_loss = partial(
    jax.jit,
    static_argnames=("config", "sample_axis", "reduction", "return_details"),
)(cvaml_compatible_bellman_residual_loss)


def build_action_chunk_endpoint_batch(
    states: Array,
    actions: Array,
    config: ActionChunkEndpointConfig = ActionChunkEndpointConfig(),
    *,
    start_conditions: Array | None = None,
    valid_transitions: Array | None = None,
    behavior_log_densities: Array | None = None,
    behavior_distances: Array | None = None,
) -> ActionChunkEndpointBatch:
    """Build every requested direct ``(start, action chunk) -> endpoint`` pair.

    ``states`` has shape ``[batch, steps+1, state_features]`` and ``actions``
    has shape ``[batch, steps, action_features]``.  For each requested horizon
    ``h`` and valid start index ``k``, the target is ``states[:, k+h]`` and the
    action chunk is ``actions[:, k:k+h]`` padded to the largest requested
    horizon.  Returned examples are flat, ordered batch-major after a
    horizon-major list of starts.

    When supplied, per-transition log densities are summed and averaged over
    the chunk.  Per-transition empirical distances are reduced by a maximum.
    Threshold failures and invalid replay transitions set ``accepted=False``;
    the observations are retained in metadata rather than silently discarded.
    """

    states = jnp.asarray(states)
    actions = jnp.asarray(actions)
    if states.ndim != 3 or actions.ndim != 3:
        raise ValueError("states and actions must be rank-three sequence arrays")
    if states.shape[0] != actions.shape[0]:
        raise ValueError("states and actions must have matching batches")
    if states.shape[1] != actions.shape[1] + 1:
        raise ValueError("states must contain exactly one more step than actions")
    if min(states.shape) <= 0 or min(actions.shape) <= 0:
        raise ValueError("state and action dimensions must be positive")
    _require_floating("states", states)
    _require_floating("actions", actions)
    batch_size, step_count, _ = actions.shape
    maximum_horizon = config.horizons[-1]
    if maximum_horizon > step_count:
        raise ValueError("requested horizon exceeds the action sequence length")

    if start_conditions is None:
        condition_sequence = states[:, :-1]
    else:
        condition_sequence = jnp.asarray(start_conditions)
        if (
            condition_sequence.ndim != 3
            or condition_sequence.shape[:2] != (batch_size, step_count)
            or condition_sequence.shape[-1] <= 0
        ):
            raise ValueError(
                "start_conditions must have shape [batch, steps, features]"
            )
        _require_floating("start_conditions", condition_sequence)

    starts_tuple = tuple(
        start
        for horizon in config.horizons
        for start in range(step_count - horizon + 1)
    )
    horizons_tuple = tuple(
        horizon for horizon in config.horizons for _ in range(step_count - horizon + 1)
    )
    start_index = jnp.asarray(starts_tuple, dtype=jnp.int32)
    horizon = jnp.asarray(horizons_tuple, dtype=jnp.int32)
    pair_count = start_index.shape[0]
    endpoint_index = start_index + horizon
    offsets = jnp.arange(maximum_horizon, dtype=jnp.int32)[None, :]
    step_index = start_index[:, None] + offsets
    action_mask = offsets < horizon[:, None]
    safe_step_index = jnp.minimum(step_index, step_count - 1)

    action_chunks = actions[:, safe_step_index, :]
    action_chunks = jnp.where(action_mask[None, :, :, None], action_chunks, 0.0)
    targets = states[:, endpoint_index, :]
    selected_start_conditions = condition_sequence[:, start_index, :]
    condition_parts = [
        selected_start_conditions,
        action_chunks.reshape((batch_size, pair_count, -1)),
    ]
    if maximum_horizon > 1:
        condition_parts.append(
            jnp.broadcast_to(
                action_mask[None], (batch_size, pair_count, maximum_horizon)
            ).astype(selected_start_conditions.dtype)
        )
    conditions = jnp.concatenate(condition_parts, axis=-1)

    if valid_transitions is None:
        transition_validity = jnp.ones((batch_size, step_count), dtype=jnp.bool_)
    else:
        transition_validity = jnp.asarray(valid_transitions)
        if transition_validity.shape != (batch_size, step_count):
            raise ValueError("valid_transitions must have shape [batch, steps]")
        transition_validity = transition_validity > 0
    gathered_validity = transition_validity[:, safe_step_index]
    chunk_valid = jnp.all(
        jnp.where(action_mask[None], gathered_validity, True), axis=-1
    )

    if behavior_log_densities is None:
        if config.minimum_mean_behavior_log_density is not None:
            raise ValueError(
                "behavior_log_densities are required by the density threshold"
            )
        if config.behavior_density_is_approximate:
            raise ValueError(
                "approximate density labeling requires behavior_log_densities"
            )
        joint_log_density = jnp.zeros((batch_size, pair_count), dtype=targets.dtype)
        mean_log_density = jnp.zeros_like(joint_log_density)
        density_available = jnp.zeros_like(joint_log_density, dtype=jnp.bool_)
    else:
        log_density = jnp.asarray(behavior_log_densities, dtype=targets.dtype)
        if log_density.shape != (batch_size, step_count):
            raise ValueError("behavior_log_densities must have shape [batch, steps]")
        gathered_density = log_density[:, safe_step_index]
        joint_log_density = jnp.sum(
            jnp.where(action_mask[None], gathered_density, 0.0), axis=-1
        )
        mean_log_density = joint_log_density / horizon[None].astype(targets.dtype)
        density_available = jnp.ones_like(joint_log_density, dtype=jnp.bool_)

    invalid_density = density_available & ~jnp.isfinite(mean_log_density)

    if behavior_distances is None:
        if config.maximum_behavior_distance is not None:
            raise ValueError(
                "behavior_distances are required by the distance threshold"
            )
        maximum_distance = jnp.zeros((batch_size, pair_count), dtype=targets.dtype)
        distance_available = jnp.zeros_like(maximum_distance, dtype=jnp.bool_)
        invalid_distance = jnp.zeros_like(maximum_distance, dtype=jnp.bool_)
    else:
        distances = jnp.asarray(behavior_distances, dtype=targets.dtype)
        if distances.shape != (batch_size, step_count):
            raise ValueError("behavior_distances must have shape [batch, steps]")
        gathered_distance = distances[:, safe_step_index]
        maximum_distance = jnp.max(
            jnp.where(action_mask[None], gathered_distance, -jnp.inf), axis=-1
        )
        distance_available = jnp.ones_like(maximum_distance, dtype=jnp.bool_)
        invalid_distance = jnp.any(
            action_mask[None]
            & (~jnp.isfinite(gathered_distance) | (gathered_distance < 0.0)),
            axis=-1,
        )

    rejected_by_density = invalid_density
    if config.minimum_mean_behavior_log_density is not None:
        rejected_by_density = rejected_by_density | (
            mean_log_density < config.minimum_mean_behavior_log_density
        )
    rejected_by_distance = invalid_distance
    if config.maximum_behavior_distance is not None:
        rejected_by_distance = rejected_by_distance | (
            maximum_distance > config.maximum_behavior_distance
        )
    rejected_by_invalid_transition = ~chunk_valid
    accepted = ~(
        rejected_by_invalid_transition | rejected_by_density | rejected_by_distance
    )

    def flatten_leading(value: Array) -> Array:
        return value.reshape((batch_size * pair_count,) + value.shape[2:])

    flat_count = batch_size * pair_count
    source_batch_index = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32), pair_count)
    flat_start_index = jnp.tile(start_index, batch_size)
    flat_horizon = jnp.tile(horizon, batch_size)
    rejection = BehaviorRejectionMetadata(
        mean_log_density=mean_log_density.reshape((flat_count,)),
        joint_log_density=joint_log_density.reshape((flat_count,)),
        maximum_distance=maximum_distance.reshape((flat_count,)),
        density_available=density_available.reshape((flat_count,)),
        distance_available=distance_available.reshape((flat_count,)),
        density_is_approximate=jnp.full(
            (flat_count,),
            config.behavior_density_is_approximate
            and behavior_log_densities is not None,
            dtype=jnp.bool_,
        ),
        rejected_by_invalid_transition=rejected_by_invalid_transition.reshape(
            (flat_count,)
        ),
        rejected_by_density=rejected_by_density.reshape((flat_count,)),
        rejected_by_distance=rejected_by_distance.reshape((flat_count,)),
        accepted=accepted.reshape((flat_count,)),
    )
    return ActionChunkEndpointBatch(
        targets=flatten_leading(targets),
        conditions=flatten_leading(conditions),
        start_conditions=flatten_leading(selected_start_conditions),
        action_chunks=flatten_leading(action_chunks),
        action_mask=jnp.broadcast_to(
            action_mask[None], (batch_size, pair_count, maximum_horizon)
        ).reshape((flat_count, maximum_horizon)),
        source_batch_index=source_batch_index,
        start_index=flat_start_index,
        horizon=flat_horizon,
        rejection=rejection,
    )


jit_build_action_chunk_endpoint_batch = partial(
    jax.jit,
    static_argnames=("config",),
)(build_action_chunk_endpoint_batch)


def action_chunk_endpoint_imf_loss(
    params: Params,
    batch: ActionChunkEndpointBatch,
    key: Array,
    *,
    noise: Array | None = None,
    tilt_weights: Array | None = None,
    reduction: Reduction = "mean",
    return_details: bool = False,
) -> Array | ActionChunkEndpointLoss:
    """Train the exact one-NFE endpoint of a direct action-chunk iMF model.

    The prediction is ``epsilon - u(epsilon, condition, r=0, t=1)`` and the
    target is the stopped observed endpoint.  The loss is raw feature-mean
    squared error, without the adaptive iMF normalization that can saturate a
    certificate-facing endpoint residual.  Empirical rejection and optional
    stopped policy-tilt weights multiply each example.

    This objective is a direct any-step construction, not a claim that action
    chunks or behavior filters provide a novel support or control guarantee.
    """

    _validate_reduction(reduction)
    targets = jnp.asarray(batch.targets)
    conditions = jnp.asarray(batch.conditions, dtype=targets.dtype)
    if targets.ndim != 2 or conditions.ndim != 2:
        raise ValueError("endpoint targets and conditions must be matrices")
    if targets.shape[0] != conditions.shape[0] or targets.shape[0] == 0:
        raise ValueError("endpoint targets and conditions need one shared batch")
    _require_floating("targets", targets)
    accepted = jnp.asarray(batch.rejection.accepted, dtype=targets.dtype)
    if accepted.shape != (targets.shape[0],):
        raise ValueError("rejection.accepted must have one value per endpoint")
    if tilt_weights is None:
        tilt = jnp.ones_like(accepted)
    else:
        tilt = jnp.asarray(tilt_weights, dtype=targets.dtype)
        if tilt.shape != accepted.shape:
            raise ValueError("tilt_weights must have one value per endpoint")
        tilt = jax.lax.stop_gradient(tilt)
    weights = jax.lax.stop_gradient(accepted * tilt)

    if noise is not None:
        noise = jnp.asarray(noise, dtype=targets.dtype)
        if noise.shape != targets.shape:
            raise ValueError("noise must have the endpoint-target shape")
    prediction = sample_imf_one_step(
        params,
        conditions,
        key,
        noise=noise,
    )
    target = jax.lax.stop_gradient(targets)
    per_example = jnp.mean(jnp.square(prediction - target), axis=-1)
    weighted = per_example * weights
    loss = _masked_reduction(per_example, weights, reduction)
    if not return_details:
        return loss
    return ActionChunkEndpointLoss(
        loss=loss,
        per_example_loss=per_example,
        weighted_per_example_loss=weighted,
        prediction=prediction,
        target=target,
        weights=weights,
        accepted_fraction=jnp.mean(accepted > 0.0),
        rejection=batch.rejection,
    )


jit_action_chunk_endpoint_imf_loss = partial(
    jax.jit,
    static_argnames=("reduction", "return_details"),
)(action_chunk_endpoint_imf_loss)


__all__ = [
    "APPROXIMATE_DETERMINISTIC_DENSITY_LABEL",
    "ActionChunkEndpointBatch",
    "ActionChunkEndpointConfig",
    "ActionChunkEndpointLoss",
    "BehaviorRejectionMetadata",
    "CVAML_COMPATIBLE_SQUARED_LABEL",
    "DIRECT_ACTION_CHUNK_ENDPOINT_LABEL",
    "EMPIRICAL_BEHAVIOR_REJECTION_LABEL",
    "EXPLICIT_STOCHASTIC_DENSITY_LABEL",
    "PlannerValueEquivalenceConfig",
    "PlannerValueEquivalenceLoss",
    "PolicyDensityTiltConfig",
    "PolicyDensityTiltDetails",
    "action_chunk_endpoint_imf_loss",
    "approximate_fixed_variance_gaussian_log_probability",
    "build_action_chunk_endpoint_batch",
    "cvaml_compatible_bellman_residual_loss",
    "diagonal_gaussian_log_probability",
    "jit_action_chunk_endpoint_imf_loss",
    "jit_build_action_chunk_endpoint_batch",
    "jit_cvaml_compatible_bellman_residual_loss",
    "jit_planner_matched_bellman_residual_loss",
    "jit_policy_density_tilt_weights",
    "planner_bellman_backup",
    "planner_matched_bellman_residual_loss",
    "policy_density_tilt_weights",
]
