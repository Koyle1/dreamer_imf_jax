"""Conditional Improved MeanFlow with a one-network-evaluation sampler."""

from __future__ import annotations

from functools import partial
import math
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from .fidelity import scale_condition_gradient
from .nn import Params, init_mlp, mlp
from .types import IMFLossDetails


Reduction = Literal["none", "mean", "sum"]


def init_imf(
    key: Array,
    sample_dim: int,
    condition_dim: int,
    hidden_dim: int = 256,
    depth: int = 3,
) -> Params:
    """Initialize one shared network with average- and instant-velocity heads.

    The final axis is ``(u, v)``.  Keeping both heads in the same MLP matches
    the official iMF recipe and lets sampling obtain ``u`` with one network
    evaluation even though training also supervises the auxiliary ``v`` head.
    """

    if min(sample_dim, condition_dim, hidden_dim, depth) <= 0:
        raise ValueError("iMF dimensions and depth must be positive")
    return init_mlp(
        key,
        sample_dim + condition_dim + 2,
        hidden_dim,
        2 * sample_dim,
        depth=depth,
    )


def _time_column(value: Array | float, reference: Array) -> Array:
    value = jnp.asarray(value, dtype=reference.dtype)
    batch = reference.shape[0]
    if value.ndim == 0:
        return jnp.broadcast_to(value, (batch, 1))
    if value.ndim == 1 and value.shape == (batch,):
        return value[:, None]
    if value.ndim == 2 and value.shape == (batch, 1):
        return value
    raise ValueError(f"time must be scalar, [batch], or [batch, 1], got {value.shape}")


def imf_outputs(
    params: Params,
    z: Array,
    condition: Array,
    r: Array | float,
    t: Array | float,
) -> tuple[Array, Array]:
    """Evaluate the shared network and split its output into ``(u, v)``."""

    if z.ndim != 2 or condition.ndim != 2 or z.shape[0] != condition.shape[0]:
        raise ValueError("z and condition must be rank-two arrays with matching batches")
    r_column = _time_column(r, z)
    t_column = _time_column(t, z)
    inputs = jnp.concatenate((z, condition, r_column, t_column), axis=-1)
    output = mlp(params, inputs, activation=jax.nn.silu)
    if output.shape[-1] != 2 * z.shape[-1]:
        raise ValueError("iMF output dimension must be twice the sample dimension")
    return tuple(jnp.split(output, 2, axis=-1))


def imf_field(
    params: Params,
    z: Array,
    condition: Array,
    r: Array | float,
    t: Array | float,
) -> Array:
    """Evaluate the conditional average-velocity field ``u``."""

    return imf_outputs(params, z, condition, r, t)[0]


def imf_velocity(
    params: Params,
    z: Array,
    condition: Array,
    r: Array | float,
    t: Array | float,
) -> Array:
    """Evaluate the auxiliary instantaneous-velocity head ``v``."""

    return imf_outputs(params, z, condition, r, t)[1]


def sample_imf_one_step(
    params: Params,
    condition: Array,
    key: Array | None = None,
    *,
    noise: Array | None = None,
) -> Array:
    """Map standard-normal noise to a sample with exactly one field call."""

    if condition.ndim != 2:
        raise ValueError("condition must have shape [batch, features]")
    output_dim = params["layers"][-1]["bias"].shape[0]
    if output_dim % 2:
        raise ValueError("iMF network output dimension must be even")
    sample_dim = output_dim // 2
    if noise is None:
        if key is None:
            raise ValueError("key is required when noise is omitted")
        noise = jax.random.normal(key, (condition.shape[0], sample_dim), dtype=condition.dtype)
    elif noise.shape != (condition.shape[0], sample_dim):
        raise ValueError("noise has the wrong shape")
    zeros = jnp.zeros((condition.shape[0], 1), dtype=condition.dtype)
    ones = jnp.ones_like(zeros)
    average_velocity, _ = imf_outputs(params, noise, condition, zeros, ones)
    return noise - average_velocity


def transport_imf_interval(
    params: Params,
    state_at_t: Array,
    condition: Array,
    r: Array | float,
    t: Array | float,
) -> Array:
    """Transport a state backward from noise time ``t`` to ``r`` in one call."""

    if state_at_t.ndim != 2 or condition.ndim != 2:
        raise ValueError("state_at_t and condition must be rank-two arrays")
    if state_at_t.shape[0] != condition.shape[0]:
        raise ValueError("state_at_t and condition batches must match")
    r_column = _time_column(r, state_at_t)
    t_column = _time_column(t, state_at_t)
    if isinstance(r, (int, float)) and isinstance(t, (int, float)) and r > t:
        raise ValueError("transport requires r <= t")
    return state_at_t - (t_column - r_column) * imf_field(
        params, state_at_t, condition, r_column, t_column
    )


def sample_imf_steps(
    params: Params,
    condition: Array,
    key: Array | None = None,
    *,
    noise: Array | None = None,
    steps: int = 1,
) -> Array:
    """Sample with a requested number of equal MeanFlow transport intervals."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if condition.ndim != 2:
        raise ValueError("condition must have shape [batch, features]")
    output_dim = params["layers"][-1]["bias"].shape[0]
    if output_dim % 2:
        raise ValueError("iMF network output dimension must be even")
    sample_dim = output_dim // 2
    if noise is None:
        if key is None:
            raise ValueError("key is required when noise is omitted")
        state = jax.random.normal(
            key, (condition.shape[0], sample_dim), dtype=condition.dtype
        )
    else:
        if noise.shape != (condition.shape[0], sample_dim):
            raise ValueError("noise has the wrong shape")
        state = noise
    for index in range(steps):
        t = 1.0 - index / steps
        r = 1.0 - (index + 1) / steps
        state = transport_imf_interval(params, state, condition, r, t)
    return state


def sample_time_pairs(
    key: Array,
    batch_size: int,
    *,
    dtype: jnp.dtype = jnp.float32,
    boundary_fraction: float = 0.5,
    time_mean: float = -0.4,
    time_std: float = 1.0,
) -> tuple[Array, Array]:
    """Sample ordered logit-normal times with exact flow-matching mass.

    Two independent logit-normal values are ordered to obtain ``r <= t``.
    The first ``floor(batch_size * boundary_fraction)`` examples are then set
    to ``r=t``.  This deterministic mask gives the requested mass exactly for
    divisible batch sizes and follows the official iMF implementation.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= boundary_fraction <= 1:
        raise ValueError("boundary_fraction must lie in [0, 1]")
    if not math.isfinite(time_mean) or not math.isfinite(time_std) or time_std <= 0:
        raise ValueError("logit-normal parameters must be finite with positive std")
    logits = jax.random.normal(key, (batch_size, 2), dtype=dtype)
    samples = jax.nn.sigmoid(logits * time_std + time_mean)
    t = jnp.max(samples, axis=-1, keepdims=True)
    r = jnp.min(samples, axis=-1, keepdims=True)
    boundary_count = int(batch_size * boundary_fraction)
    on_boundary = jnp.arange(batch_size)[:, None] < boundary_count
    r = jnp.where(on_boundary, t, r)
    return r, t


def improved_meanflow_loss(
    params: Params,
    target: Array,
    condition: Array,
    key: Array,
    *,
    noise: Array | None = None,
    r: Array | float | None = None,
    t: Array | float | None = None,
    boundary_fraction: float = 0.5,
    time_mean: float = -0.4,
    time_std: float = 1.0,
    adaptive_power: float = 1.0,
    adaptive_epsilon: float = 0.01,
    meanflow_scale: float = 1.0,
    velocity_scale: float = 1.0,
    endpoint_scale: float = 0.0,
    shortcut_scale: float = 0.0,
    signal_weight_floor: float = 1.0,
    signal_weight_scale: float = 0.0,
    condition_gradient_scale: float = 1.0,
    boundary_velocity_supervision: bool = False,
    weights: Array | None = None,
    reduction: Reduction = "mean",
    return_details: bool = False,
) -> Array | IMFLossDetails:
    """Compute the improved MeanFlow compound regression objective.

    For ``z = (1-t)x + t e``, the implementation is

    ``v = v_head(z, c, t, t)``

    ``(u, dudt) = jvp(u, (z, r, t), (v, 0, 1))``

    ``V = u + (t-r) stop_gradient(dudt)``.

    Both ``V`` and the selected auxiliary velocity prediction are regressed to
    ``e - x``.  With boundary supervision enabled, that auxiliary prediction
    is exactly ``v(z, c, t, t)`` used as the JVP state tangent; the legacy
    default instead supervises ``v(z, c, r, t)``.  A positive endpoint scale
    additionally supervises the direct one-step sample ``e-u(e,c,0,1)``
    against ``x`` using the same ``e``.  The main u/v squared-error sums use
    the stopped adaptive normalization from official iMF; the endpoint term
    deliberately remains a raw per-sample MSE so its gradient cannot collapse
    under the default power-one normalization.
    """

    if target.ndim != 2 or condition.ndim != 2 or target.shape[0] != condition.shape[0]:
        raise ValueError("target and condition must be matrices with matching batches")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("invalid reduction")
    if (r is None) != (t is None):
        raise ValueError("r and t must both be supplied or both omitted")
    if (
        not math.isfinite(adaptive_power)
        or not math.isfinite(adaptive_epsilon)
        or not math.isfinite(velocity_scale)
        or not math.isfinite(meanflow_scale)
        or not math.isfinite(endpoint_scale)
        or not math.isfinite(shortcut_scale)
        or not math.isfinite(signal_weight_floor)
        or not math.isfinite(signal_weight_scale)
        or not math.isfinite(condition_gradient_scale)
        or adaptive_power < 0
        or adaptive_epsilon <= 0
        or velocity_scale < 0
        or meanflow_scale < 0
        or endpoint_scale < 0
        or shortcut_scale < 0
        or signal_weight_floor < 0
        or signal_weight_scale < 0
        or signal_weight_floor + signal_weight_scale <= 0
        or not 0 <= condition_gradient_scale <= 1
    ):
        raise ValueError(
            "adaptive power, signal weights, and loss scales must be finite and nonnegative, "
            "epsilon finite and positive, and condition gradient scale in [0, 1]"
        )
    if not isinstance(boundary_velocity_supervision, bool):
        raise ValueError("boundary_velocity_supervision must be boolean")
    noise_key, time_key = jax.random.split(key)
    if noise is None:
        noise = jax.random.normal(noise_key, target.shape, dtype=target.dtype)
    if noise.shape != target.shape:
        raise ValueError("noise must have the target shape")
    if r is None:
        r_column, t_column = sample_time_pairs(
            time_key,
            target.shape[0],
            dtype=target.dtype,
            boundary_fraction=boundary_fraction,
            time_mean=time_mean,
            time_std=time_std,
        )
    else:
        r_column = _time_column(r, target)
        t_column = _time_column(t, target)

    model_condition = scale_condition_gradient(condition, condition_gradient_scale)
    interpolated = (1.0 - t_column) * target + t_column * noise
    # The state tangent is the auxiliary instantaneous-velocity prediction at
    # the flow-matching boundary.  Using u here is the original MeanFlow target
    # and is intentionally not equivalent.
    marginal_velocity = imf_velocity(
        params, interpolated, model_condition, t_column, t_column
    )

    def field(z_value: Array, r_value: Array, t_value: Array):
        average_velocity, velocity = imf_outputs(
            params, z_value, model_condition, r_value, t_value
        )
        return average_velocity, velocity

    value, tangent, velocity_prediction = jax.jvp(
        field,
        (interpolated, r_column, t_column),
        (marginal_velocity, jnp.zeros_like(r_column), jnp.ones_like(t_column)),
        has_aux=True,
    )
    prediction = value + (t_column - r_column) * jax.lax.stop_gradient(tangent)
    regression_target = jax.lax.stop_gradient(noise - target)
    squared_u = jnp.square(prediction - regression_target)
    supervised_velocity = (
        marginal_velocity if boundary_velocity_supervision else velocity_prediction
    )
    squared_v = jnp.square(supervised_velocity - regression_target)
    if endpoint_scale > 0:
        endpoint_r = jnp.zeros_like(r_column)
        endpoint_t = jnp.ones_like(t_column)
        endpoint_field = imf_field(
            params, noise, model_condition, endpoint_r, endpoint_t
        )
        endpoint_prediction = noise - endpoint_field
        squared_endpoint = jnp.square(
            endpoint_prediction - jax.lax.stop_gradient(target)
        )
    else:
        endpoint_prediction = jnp.zeros_like(target)
        squared_endpoint = jnp.zeros_like(target)
    if shortcut_scale > 0:
        midpoint = 0.5 * (r_column + t_column)
        shortcut_prediction = transport_imf_interval(
            params, interpolated, model_condition, r_column, t_column
        )
        midpoint_prediction = transport_imf_interval(
            params, interpolated, model_condition, midpoint, t_column
        )
        shortcut_target = transport_imf_interval(
            params, midpoint_prediction, model_condition, r_column, midpoint
        )
        squared_shortcut = jnp.square(
            shortcut_prediction - jax.lax.stop_gradient(shortcut_target)
        )
    else:
        shortcut_prediction = jnp.zeros_like(target)
        shortcut_target = jnp.zeros_like(target)
        squared_shortcut = jnp.zeros_like(target)
    raw_loss_u = jnp.mean(squared_u, axis=-1)
    raw_loss_v = jnp.mean(squared_v, axis=-1)
    summed_u = jnp.sum(squared_u, axis=-1)
    summed_v = jnp.sum(squared_v, axis=-1)
    adaptive_weight_u = jnp.power(summed_u + adaptive_epsilon, adaptive_power)
    adaptive_weight_v = jnp.power(summed_v + adaptive_epsilon, adaptive_power)
    adaptive_u = summed_u / jax.lax.stop_gradient(adaptive_weight_u)
    adaptive_v = summed_v / jax.lax.stop_gradient(adaptive_weight_v)
    raw_loss_endpoint = jnp.mean(squared_endpoint, axis=-1)
    raw_loss_shortcut = jnp.mean(squared_shortcut, axis=-1)
    per_sample = (
        meanflow_scale * adaptive_u
        + velocity_scale * adaptive_v
        + endpoint_scale * raw_loss_endpoint
        + shortcut_scale * raw_loss_shortcut
    )
    signal_weights = (
        signal_weight_floor + signal_weight_scale * (1.0 - t_column[:, 0])
    )
    if weights is not None:
        external_weights = jnp.asarray(weights, dtype=target.dtype).reshape((-1,))
        if external_weights.shape != (target.shape[0],):
            raise ValueError("weights must have one value per batch item")
        combined_weights = signal_weights * external_weights
    else:
        combined_weights = signal_weights
    weighted = per_sample * combined_weights
    if reduction == "none":
        loss = weighted
    elif reduction == "sum":
        loss = jnp.sum(weighted)
    else:
        loss = jnp.sum(weighted) / jnp.maximum(
            jnp.sum(combined_weights), jnp.finfo(target.dtype).tiny
        )

    if not return_details:
        return loss
    return IMFLossDetails(
        loss=loss,
        per_sample_loss=per_sample,
        prediction=prediction,
        regression_target=regression_target,
        field=value,
        jvp=tangent,
        marginal_velocity=marginal_velocity,
        interpolated=interpolated,
        noise=noise,
        r=r_column,
        t=t_column,
        raw_loss_u=raw_loss_u,
        raw_loss_v=raw_loss_v,
        adaptive_weight_u=adaptive_weight_u,
        adaptive_weight_v=adaptive_weight_v,
        velocity_prediction=velocity_prediction,
        endpoint_prediction=endpoint_prediction,
        raw_loss_endpoint=raw_loss_endpoint,
        shortcut_prediction=shortcut_prediction,
        shortcut_target=shortcut_target,
        raw_loss_shortcut=raw_loss_shortcut,
        signal_weights=signal_weights,
    )


jit_improved_meanflow_loss = partial(
    jax.jit,
    static_argnames=(
        "boundary_fraction",
        "time_mean",
        "time_std",
        "adaptive_power",
        "adaptive_epsilon",
        "meanflow_scale",
        "velocity_scale",
        "endpoint_scale",
        "shortcut_scale",
        "signal_weight_floor",
        "signal_weight_scale",
        "condition_gradient_scale",
        "boundary_velocity_supervision",
        "reduction",
        "return_details",
    ),
)(improved_meanflow_loss)

jit_sample_imf_steps = partial(jax.jit, static_argnames=("steps",))(
    sample_imf_steps
)
