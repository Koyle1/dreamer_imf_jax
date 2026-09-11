"""Multi-time conditional Improved MeanFlow for causal trajectories.

The public loss treats each realized causal context as a conditioning value.
Consequently, the material derivative for token ``k`` differentiates only the
current corrupted token ``z_k`` and its query time ``t_k``.  Earlier corrupted
tokens and their noise times are *held fixed*.  This distinction matters when
contexts are built from a corrupted prefix: differentiating the context inside
the JVP follows a different, diagonal path through the whole sequence.

All functions in this module are pure JAX.  Targets, conditions, noise, and
times use leading ``[batch, time]`` axes; no Python loop is used over either
axis in the loss.  In the derived two-view objective, callers use one
independent Gaussian tensor per trajectory and evaluate it at both the query
times ``t`` and the history exposure times ``history_t``.  The low-level API
keeps context construction explicit, so a caller intentionally studying a
different noise coupling can still provide precomputed conditions.
"""

from __future__ import annotations

from functools import partial
import math
from typing import Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from .fidelity import scale_condition_gradient
from .imf import imf_outputs, imf_velocity
from .nn import Params


Reduction = Literal["none", "mean", "sum"]
ScheduleMode = Literal["clean_context", "corrupted_context", "future_suffix", "mixed"]


class TrajectoryIMFSchedule(NamedTuple):
    """Per-token query and history corruption sampled by one schedule.

    ``r`` and ``t`` parameterize the iMF query. ``history_t`` is separate on
    purpose: a token can be queried at a nonzero time while the copy exposed to
    a later token is clean. ``loss_mask`` selects tokens supervised by this
    draw. ``pattern`` is 0 (clean context), 1 (corrupted context), or 2
    (future suffix), and ``suffix_start`` is meaningful only for pattern 2.
    """

    r: Array
    t: Array
    history_t: Array
    loss_mask: Array
    pattern: Array
    suffix_start: Array


class TrajectoryJVPDetails(NamedTuple):
    """Components of a trajectory iMF material derivative."""

    field: Array
    jvp: Array
    marginal_velocity: Array
    velocity_prediction: Array
    conditions: Array


class TrajectoryIMFLossDetails(NamedTuple):
    """Structured, JAX-PyTree-compatible trajectory loss diagnostics."""

    loss: Array
    per_token_loss: Array
    weighted_per_token_loss: Array
    prediction: Array
    regression_target: Array
    field: Array
    jvp: Array
    marginal_velocity: Array
    velocity_prediction: Array
    supervised_velocity: Array
    interpolated: Array
    conditions: Array
    noise: Array
    r: Array
    t: Array
    token_mask: Array
    token_weights: Array
    signal_weights: Array
    raw_loss_u: Array
    raw_loss_v: Array
    adaptive_weight_u: Array
    adaptive_weight_v: Array


def _validate_trajectory(name: str, value: Array) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, time, features]")
    if value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError(f"{name} dimensions must be positive")


def _token_column(value: Array | float, reference: Array, name: str) -> Array:
    """Broadcast a scalar or per-token value to ``[batch, time, 1]``."""

    result = jnp.asarray(value, dtype=reference.dtype)
    batch, steps = reference.shape[:2]
    if result.ndim == 0:
        return jnp.broadcast_to(result, (batch, steps, 1))
    if result.shape == (batch, steps):
        return result[..., None]
    if result.shape == (batch, steps, 1):
        return result
    if result.shape == (batch, 1):
        return jnp.broadcast_to(result[:, None, :], (batch, steps, 1))
    if result.shape == (batch, 1, 1):
        return jnp.broadcast_to(result, (batch, steps, 1))
    raise ValueError(
        f"{name} must be scalar, [batch, time], [batch, time, 1], "
        f"[batch, 1], or [batch, 1, 1], got {result.shape}"
    )


def _token_values(
    value: Array | float | None,
    reference: Array,
    name: str,
    *,
    default: float,
) -> Array:
    if value is None:
        return jnp.full(reference.shape[:2], default, dtype=reference.dtype)
    result = jnp.asarray(value, dtype=reference.dtype)
    batch, steps = reference.shape[:2]
    if result.ndim == 0:
        return jnp.broadcast_to(result, (batch, steps))
    if result.shape == (batch, steps):
        return result
    if result.shape == (batch, steps, 1):
        return result[..., 0]
    if result.shape == (batch, 1):
        return jnp.broadcast_to(result, (batch, steps))
    if result.shape == (batch, 1, 1):
        return jnp.broadcast_to(result[..., 0], (batch, steps))
    raise ValueError(f"{name} must have one scalar per token, got {result.shape}")


def corrupt_trajectory(targets: Array, noise: Array, times: Array | float) -> Array:
    """Apply ``z_k=(1-t_k)x_k+t_k*epsilon_k`` independently per token."""

    _validate_trajectory("targets", targets)
    if noise.shape != targets.shape:
        raise ValueError("noise must have the same shape as targets")
    time_columns = _token_column(times, targets, "times")
    return (1.0 - time_columns) * targets + time_columns * noise


def sample_trajectory_time_pairs(
    key: Array,
    batch_size: int,
    sequence_length: int,
    *,
    dtype: jnp.dtype = jnp.float32,
    boundary_fraction: float = 0.5,
    time_mean: float = -0.4,
    time_std: float = 1.0,
) -> tuple[Array, Array]:
    """Sample independent ordered ``r_k <= t_k`` pairs for every token."""

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if not 0.0 <= boundary_fraction <= 1.0:
        raise ValueError("boundary_fraction must lie in [0, 1]")
    if not math.isfinite(time_mean) or not math.isfinite(time_std) or time_std <= 0:
        raise ValueError("logit-normal parameters must be finite with positive std")
    sample_key, boundary_key = jax.random.split(key)
    logits = jax.random.normal(
        sample_key, (batch_size, sequence_length, 2), dtype=dtype
    )
    samples = jax.nn.sigmoid(logits * time_std + time_mean)
    r = jnp.min(samples, axis=-1, keepdims=True)
    t = jnp.max(samples, axis=-1, keepdims=True)
    on_boundary = jax.random.bernoulli(
        boundary_key,
        boundary_fraction,
        shape=(batch_size, sequence_length, 1),
    )
    return jnp.where(on_boundary, t, r), t


def sample_trajectory_schedule(
    key: Array,
    batch_size: int,
    sequence_length: int,
    *,
    mode: ScheduleMode = "mixed",
    token_mask: Array | None = None,
    dtype: jnp.dtype = jnp.float32,
    boundary_fraction: float = 0.5,
    time_mean: float = -0.4,
    time_std: float = 1.0,
    history_noise_max: float = 1.0,
    pattern_probabilities: Sequence[float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    canonical_uniforms: Array | None = None,
) -> TrajectoryIMFSchedule:
    """Sample clean-context, corrupted-context, or future-suffix supervision.

    The three cases differ only in sampled corruption and masking; downstream
    training always uses :func:`trajectory_imf_loss`. ``t`` is the query time
    and ``history_t`` is the exposure time for the copy seen by later tokens;
    the derived objective evaluates both with the same per-token Gaussian
    noise. In ``future_suffix`` mode, a clean prefix is followed by
    independently timed corrupted future tokens, and only that suffix
    contributes to the loss.
    """

    if mode not in ("clean_context", "corrupted_context", "future_suffix", "mixed"):
        raise ValueError(f"unknown schedule mode: {mode}")
    if not math.isfinite(history_noise_max) or not 0.0 <= history_noise_max <= 1.0:
        raise ValueError("history_noise_max must lie in [0, 1]")
    probabilities = tuple(float(value) for value in pattern_probabilities)
    if len(probabilities) != 3 or any(
        not math.isfinite(value) or value < 0.0 for value in probabilities
    ):
        raise ValueError("pattern_probabilities must contain three nonnegative values")
    probability_sum = sum(probabilities)
    if probability_sum <= 0.0:
        raise ValueError("pattern_probabilities must have positive mass")

    query_key, history_key, pattern_key, suffix_key = jax.random.split(key, 4)
    if canonical_uniforms is None:
        r, t = sample_trajectory_time_pairs(
            query_key,
            batch_size,
            sequence_length,
            dtype=dtype,
            boundary_fraction=boundary_fraction,
            time_mean=time_mean,
            time_std=time_std,
        )
        history_t_random = history_noise_max * jax.random.uniform(
            history_key, (batch_size, sequence_length, 1), dtype=dtype
        )
        pattern_uniform = None
        suffix_uniform = None
    else:
        uniforms = jnp.asarray(canonical_uniforms, dtype=dtype)
        if uniforms.shape != (batch_size, sequence_length, 6):
            raise ValueError(
                "canonical_uniforms must have shape [batch, time, 6]"
            )
        epsilon = jnp.finfo(dtype).eps
        query_uniforms = jnp.clip(uniforms[..., :2], epsilon, 1.0 - epsilon)
        logits = jax.scipy.special.ndtri(query_uniforms) * time_std + time_mean
        samples = jax.nn.sigmoid(logits)
        lower = jnp.min(samples, axis=-1, keepdims=True)
        t = jnp.max(samples, axis=-1, keepdims=True)
        r = jnp.where(
            uniforms[..., 2:3] < boundary_fraction,
            t,
            lower,
        )
        history_t_random = history_noise_max * uniforms[..., 3:4]
        pattern_uniform = uniforms[:, 0, 4]
        suffix_uniform = uniforms[:, 0, 5]
    if mode == "mixed":
        if pattern_uniform is None:
            logits = jnp.log(jnp.asarray(probabilities, dtype=dtype) / probability_sum)
            pattern = jax.random.categorical(pattern_key, logits, shape=(batch_size,))
        else:
            cumulative = jnp.cumsum(
                jnp.asarray(probabilities, dtype=dtype) / probability_sum
            )
            pattern = jnp.sum(
                pattern_uniform[:, None] >= cumulative[None, :], axis=-1
            ).astype(jnp.int32)
    else:
        code = {
            "clean_context": 0,
            "corrupted_context": 1,
            "future_suffix": 2,
        }[mode]
        pattern = jnp.full((batch_size,), code, dtype=jnp.int32)

    if token_mask is None:
        base_mask = jnp.ones((batch_size, sequence_length), dtype=dtype)
    else:
        dummy = jnp.empty((batch_size, sequence_length, 1), dtype=dtype)
        base_mask = _token_values(token_mask, dummy, "token_mask", default=1.0)

    # Choose a valid token, including for padded or non-contiguous masks.  A
    # uniform valid-token rank avoids the empty suffixes produced by drawing a
    # raw position beyond a short episode.  An entirely empty row remains
    # empty and uses the harmless sentinel start 0.
    valid = base_mask > 0.0
    valid_count = jnp.sum(valid, axis=1, dtype=jnp.int32)
    if suffix_uniform is None:
        suffix_uniform = jax.random.uniform(suffix_key, (batch_size,), dtype=dtype)
    suffix_rank = jnp.floor(
        suffix_uniform * jnp.maximum(valid_count, 1).astype(dtype)
    ).astype(jnp.int32)
    cumulative_valid = jnp.cumsum(valid.astype(jnp.int32), axis=1)
    candidate = valid & (cumulative_valid > suffix_rank[:, None])
    positions = jnp.broadcast_to(
        jnp.arange(sequence_length, dtype=jnp.int32)[None, :], candidate.shape
    )
    suffix_start = jnp.min(
        jnp.where(candidate, positions, sequence_length), axis=1
    )
    suffix_start = jnp.where(valid_count > 0, suffix_start, 0)
    position = jnp.arange(sequence_length)[None, :]
    in_suffix = position >= suffix_start[:, None]
    clean_history_t = jnp.zeros_like(history_t_random)
    suffix_history_t = jnp.where(
        in_suffix[..., None], history_t_random, clean_history_t
    )
    history_t = jnp.where(
        (pattern == 0)[:, None, None],
        clean_history_t,
        jnp.where((pattern == 1)[:, None, None], history_t_random, suffix_history_t),
    )

    suffix_loss_mask = base_mask * in_suffix.astype(dtype)
    loss_mask = jnp.where((pattern == 2)[:, None], suffix_loss_mask, base_mask)
    return TrajectoryIMFSchedule(
        r=r,
        t=t,
        history_t=history_t,
        loss_mask=loss_mask,
        pattern=pattern,
        suffix_start=suffix_start,
    )


def causal_corrupted_history_contexts(
    corrupted_history: Array,
    history_times: Array | float,
    *,
    base_conditions: Array | None = None,
    token_mask: Array | None = None,
    initial_history: Array | None = None,
) -> Array:
    """Construct strictly causal recurrent contexts from a corrupted history.

    At token ``k`` the returned context contains the caller-provided causal
    condition at ``k`` followed by the latest valid corrupted token strictly
    before ``k``, its noise time, and a validity bit.  A scan carries the latest
    valid token across padding, so masked values can never leak into later
    contexts.  Actions through ``k-1`` should be encoded in ``base_conditions``.
    """

    _validate_trajectory("corrupted_history", corrupted_history)
    batch, steps, features = corrupted_history.shape
    times = _token_column(history_times, corrupted_history, "history_times")
    mask = _token_values(
        token_mask, corrupted_history, "token_mask", default=1.0
    )
    if base_conditions is not None:
        if base_conditions.ndim != 3 or base_conditions.shape[:2] != (batch, steps):
            raise ValueError(
                "base_conditions must have shape [batch, time, condition_features]"
            )
    if initial_history is None:
        initial_value = jnp.zeros((batch, features), dtype=corrupted_history.dtype)
        initial_valid = jnp.zeros((batch, 1), dtype=corrupted_history.dtype)
    else:
        initial_value = jnp.asarray(initial_history, dtype=corrupted_history.dtype)
        if initial_value.shape != (batch, features):
            raise ValueError("initial_history must have shape [batch, target_features]")
        initial_valid = jnp.ones((batch, 1), dtype=corrupted_history.dtype)
    initial_time = jnp.zeros((batch, 1), dtype=corrupted_history.dtype)

    def step(carry, inputs):
        previous_value, previous_time, previous_valid = carry
        current_value, current_time, current_valid = inputs
        context = jnp.concatenate(
            (previous_value, previous_time, previous_valid), axis=-1
        )
        valid = current_valid[:, None]
        next_value = jnp.where(valid > 0.0, current_value, previous_value)
        next_time = jnp.where(valid > 0.0, current_time, previous_time)
        next_valid = jnp.maximum(previous_valid, valid.astype(previous_valid.dtype))
        return (next_value, next_time, next_valid), context

    scan_inputs = (
        jnp.swapaxes(corrupted_history, 0, 1),
        jnp.swapaxes(times, 0, 1),
        jnp.swapaxes(mask, 0, 1),
    )
    _, history_context = jax.lax.scan(
        step,
        (initial_value, initial_time, initial_valid),
        scan_inputs,
    )
    history_context = jnp.swapaxes(history_context, 0, 1)
    if base_conditions is None:
        return history_context
    return jnp.concatenate((base_conditions, history_context), axis=-1)


def trajectory_partial_jvp(
    params: Params,
    interpolated: Array,
    conditions: Array,
    r: Array | float,
    t: Array | float,
) -> TrajectoryJVPDetails:
    """Compute the correct per-token iMF JVP with contexts held fixed."""

    _validate_trajectory("interpolated", interpolated)
    if conditions.ndim != 3 or conditions.shape[:2] != interpolated.shape[:2]:
        raise ValueError("conditions must have shape [batch, time, condition_features]")
    r_columns = _token_column(r, interpolated, "r")
    t_columns = _token_column(t, interpolated, "t")
    batch, steps, features = interpolated.shape
    flat_z = interpolated.reshape((batch * steps, features))
    flat_condition = conditions.reshape((batch * steps, conditions.shape[-1]))
    flat_r = r_columns.reshape((batch * steps, 1))
    flat_t = t_columns.reshape((batch * steps, 1))
    marginal_velocity = imf_velocity(
        params, flat_z, flat_condition, flat_t, flat_t
    )

    # flat_condition is deliberately captured rather than passed as a primal.
    # The JVP therefore moves each (z_k, t_k) while holding c_k fixed.
    def fixed_context_field(z_value: Array, r_value: Array, t_value: Array):
        average_velocity, velocity_prediction = imf_outputs(
            params, z_value, flat_condition, r_value, t_value
        )
        return average_velocity, velocity_prediction

    field, tangent, velocity_prediction = jax.jvp(
        fixed_context_field,
        (flat_z, flat_r, flat_t),
        (marginal_velocity, jnp.zeros_like(flat_r), jnp.ones_like(flat_t)),
        has_aux=True,
    )
    trajectory_shape = (batch, steps, features)
    return TrajectoryJVPDetails(
        field=field.reshape(trajectory_shape),
        jvp=tangent.reshape(trajectory_shape),
        marginal_velocity=marginal_velocity.reshape(trajectory_shape),
        velocity_prediction=velocity_prediction.reshape(trajectory_shape),
        conditions=conditions,
    )


def naive_joint_context_jvp(
    params: Params,
    interpolated: Array,
    base_conditions: Array | None,
    r: Array | float,
    t: Array | float,
    *,
    token_mask: Array | None = None,
    initial_history: Array | None = None,
) -> TrajectoryJVPDetails:
    """Diagnostic diagonal-path JVP that incorrectly moves causal context.

    This function is a named positive control, not a training primitive.  It
    rebuilds ``c_k(z_{<k},t_{<k})`` inside a whole-sequence JVP, so token ``k``
    receives spurious derivatives through all represented prefix variables.
    """

    _validate_trajectory("interpolated", interpolated)
    if base_conditions is not None and (
        base_conditions.ndim != 3
        or base_conditions.shape[:2] != interpolated.shape[:2]
    ):
        raise ValueError("base_conditions must share the trajectory leading axes")
    r_columns = _token_column(r, interpolated, "r")
    t_columns = _token_column(t, interpolated, "t")

    contexts = causal_corrupted_history_contexts(
        interpolated,
        t_columns,
        base_conditions=base_conditions,
        token_mask=token_mask,
        initial_history=initial_history,
    )
    batch, steps, features = interpolated.shape
    marginal_velocity = imf_velocity(
        params,
        interpolated.reshape((batch * steps, features)),
        contexts.reshape((batch * steps, contexts.shape[-1])),
        t_columns.reshape((batch * steps, 1)),
        t_columns.reshape((batch * steps, 1)),
    ).reshape(interpolated.shape)

    def moving_context_field(z_value: Array, r_value: Array, t_value: Array):
        moving_context = causal_corrupted_history_contexts(
            z_value,
            t_value,
            base_conditions=base_conditions,
            token_mask=token_mask,
            initial_history=initial_history,
        )
        output, velocity = imf_outputs(
            params,
            z_value.reshape((batch * steps, features)),
            moving_context.reshape((batch * steps, moving_context.shape[-1])),
            r_value.reshape((batch * steps, 1)),
            t_value.reshape((batch * steps, 1)),
        )
        return output.reshape(interpolated.shape), velocity.reshape(interpolated.shape)

    field, tangent, velocity_prediction = jax.jvp(
        moving_context_field,
        (interpolated, r_columns, t_columns),
        (
            marginal_velocity,
            jnp.zeros_like(r_columns),
            jnp.ones_like(t_columns),
        ),
        has_aux=True,
    )
    return TrajectoryJVPDetails(
        field=field,
        jvp=tangent,
        marginal_velocity=marginal_velocity,
        velocity_prediction=velocity_prediction,
        conditions=contexts,
    )


def trajectory_imf_loss(
    params: Params,
    targets: Array,
    conditions: Array,
    key: Array,
    *,
    noise: Array | None = None,
    r: Array | float | None = None,
    t: Array | float | None = None,
    token_mask: Array | None = None,
    weights: Array | None = None,
    boundary_fraction: float = 0.5,
    time_mean: float = -0.4,
    time_std: float = 1.0,
    adaptive_power: float = 1.0,
    adaptive_epsilon: float = 0.01,
    meanflow_scale: float = 1.0,
    velocity_scale: float = 1.0,
    signal_weight_floor: float = 1.0,
    signal_weight_scale: float = 0.0,
    condition_gradient_scale: float = 1.0,
    boundary_velocity_supervision: bool = True,
    reduction: Reduction = "mean",
    return_details: bool = False,
) -> Array | TrajectoryIMFLossDetails:
    """Compute one mask-aware multi-time conditional iMF objective.

    No shortcut-consistency, context-reconstruction, endpoint, or overshooting
    penalty is added.  Clean-context, corrupted-context, and suffix behavior is
    selected by the inputs produced by :func:`sample_trajectory_schedule` and
    :func:`causal_corrupted_history_contexts`, while this objective is unchanged.
    """

    _validate_trajectory("targets", targets)
    if conditions.ndim != 3 or conditions.shape[:2] != targets.shape[:2]:
        raise ValueError("conditions must have shape [batch, time, condition_features]")
    if conditions.shape[-1] <= 0:
        raise ValueError("conditions must have a positive feature dimension")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("invalid reduction")
    if (r is None) != (t is None):
        raise ValueError("r and t must both be supplied or both omitted")
    scalar_arguments = (
        adaptive_power,
        adaptive_epsilon,
        meanflow_scale,
        velocity_scale,
        signal_weight_floor,
        signal_weight_scale,
        condition_gradient_scale,
    )
    if (
        any(not math.isfinite(value) for value in scalar_arguments)
        or adaptive_power < 0.0
        or adaptive_epsilon <= 0.0
        or meanflow_scale < 0.0
        or velocity_scale < 0.0
        or signal_weight_floor < 0.0
        or signal_weight_scale < 0.0
        or signal_weight_floor + signal_weight_scale <= 0.0
        or not 0.0 <= condition_gradient_scale <= 1.0
    ):
        raise ValueError("loss scales and weights must be finite and valid")
    if not isinstance(boundary_velocity_supervision, bool):
        raise ValueError("boundary_velocity_supervision must be boolean")

    noise_key, time_key = jax.random.split(key)
    if noise is None:
        noise = jax.random.normal(noise_key, targets.shape, dtype=targets.dtype)
    elif noise.shape != targets.shape:
        raise ValueError("noise must have the same shape as targets")
    if r is None:
        r_columns, t_columns = sample_trajectory_time_pairs(
            time_key,
            targets.shape[0],
            targets.shape[1],
            dtype=targets.dtype,
            boundary_fraction=boundary_fraction,
            time_mean=time_mean,
            time_std=time_std,
        )
    else:
        r_columns = _token_column(r, targets, "r")
        t_columns = _token_column(t, targets, "t")

    model_conditions = scale_condition_gradient(conditions, condition_gradient_scale)
    interpolated = corrupt_trajectory(targets, noise, t_columns)
    derivative = trajectory_partial_jvp(
        params, interpolated, model_conditions, r_columns, t_columns
    )
    prediction = derivative.field + (t_columns - r_columns) * jax.lax.stop_gradient(
        derivative.jvp
    )
    regression_target = jax.lax.stop_gradient(noise - targets)
    squared_u = jnp.square(prediction - regression_target)
    supervised_velocity = (
        derivative.marginal_velocity
        if boundary_velocity_supervision
        else derivative.velocity_prediction
    )
    squared_v = jnp.square(supervised_velocity - regression_target)
    raw_loss_u = jnp.mean(squared_u, axis=-1)
    raw_loss_v = jnp.mean(squared_v, axis=-1)
    summed_u = jnp.sum(squared_u, axis=-1)
    summed_v = jnp.sum(squared_v, axis=-1)
    adaptive_weight_u = jnp.power(summed_u + adaptive_epsilon, adaptive_power)
    adaptive_weight_v = jnp.power(summed_v + adaptive_epsilon, adaptive_power)
    normalized_u = summed_u / jax.lax.stop_gradient(adaptive_weight_u)
    normalized_v = summed_v / jax.lax.stop_gradient(adaptive_weight_v)
    per_token_loss = meanflow_scale * normalized_u + velocity_scale * normalized_v

    mask = _token_values(token_mask, targets, "token_mask", default=1.0)
    token_weights = _token_values(weights, targets, "weights", default=1.0)
    signal_weights = signal_weight_floor + signal_weight_scale * (
        1.0 - t_columns[..., 0]
    )
    combined_weights = mask * token_weights * signal_weights
    weighted_per_token_loss = per_token_loss * combined_weights
    if reduction == "none":
        loss = weighted_per_token_loss
    elif reduction == "sum":
        loss = jnp.sum(weighted_per_token_loss)
    else:
        denominator = jnp.maximum(
            jnp.sum(combined_weights), jnp.finfo(targets.dtype).tiny
        )
        loss = jnp.sum(weighted_per_token_loss) / denominator

    if not return_details:
        return loss
    return TrajectoryIMFLossDetails(
        loss=loss,
        per_token_loss=per_token_loss,
        weighted_per_token_loss=weighted_per_token_loss,
        prediction=prediction,
        regression_target=regression_target,
        field=derivative.field,
        jvp=derivative.jvp,
        marginal_velocity=derivative.marginal_velocity,
        velocity_prediction=derivative.velocity_prediction,
        supervised_velocity=supervised_velocity,
        interpolated=interpolated,
        conditions=model_conditions,
        noise=noise,
        r=r_columns,
        t=t_columns,
        token_mask=mask,
        token_weights=token_weights,
        signal_weights=signal_weights,
        raw_loss_u=raw_loss_u,
        raw_loss_v=raw_loss_v,
        adaptive_weight_u=adaptive_weight_u,
        adaptive_weight_v=adaptive_weight_v,
    )


jit_trajectory_imf_loss = partial(
    jax.jit,
    static_argnames=(
        "boundary_fraction",
        "time_mean",
        "time_std",
        "adaptive_power",
        "adaptive_epsilon",
        "meanflow_scale",
        "velocity_scale",
        "signal_weight_floor",
        "signal_weight_scale",
        "condition_gradient_scale",
        "boundary_velocity_supervision",
        "reduction",
        "return_details",
    ),
)(trajectory_imf_loss)


__all__ = [
    "ScheduleMode",
    "TrajectoryIMFLossDetails",
    "TrajectoryIMFSchedule",
    "TrajectoryJVPDetails",
    "causal_corrupted_history_contexts",
    "corrupt_trajectory",
    "jit_trajectory_imf_loss",
    "naive_joint_context_jvp",
    "sample_trajectory_schedule",
    "sample_trajectory_time_pairs",
    "trajectory_imf_loss",
    "trajectory_partial_jvp",
]
