"""Conditional Improved MeanFlow with a one-network-evaluation sampler."""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

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
    """Initialize ``u(z, condition, r, t)`` parameters."""

    if min(sample_dim, condition_dim, hidden_dim, depth) <= 0:
        raise ValueError("iMF dimensions and depth must be positive")
    return init_mlp(
        key,
        sample_dim + condition_dim + 2,
        hidden_dim,
        sample_dim,
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


def imf_field(
    params: Params,
    z: Array,
    condition: Array,
    r: Array | float,
    t: Array | float,
) -> Array:
    """Evaluate the conditional average-velocity field."""

    if z.ndim != 2 or condition.ndim != 2 or z.shape[0] != condition.shape[0]:
        raise ValueError("z and condition must be rank-two arrays with matching batches")
    r_column = _time_column(r, z)
    t_column = _time_column(t, z)
    inputs = jnp.concatenate((z, condition, r_column, t_column), axis=-1)
    return mlp(params, inputs, activation=jax.nn.silu)


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
    if noise is None:
        if key is None:
            raise ValueError("key is required when noise is omitted")
        noise = jax.random.normal(key, (condition.shape[0], output_dim), dtype=condition.dtype)
    elif noise.shape != (condition.shape[0], output_dim):
        raise ValueError("noise has the wrong shape")
    zeros = jnp.zeros((condition.shape[0], 1), dtype=condition.dtype)
    ones = jnp.ones_like(zeros)
    return noise - imf_field(params, noise, condition, zeros, ones)


def sample_time_pairs(
    key: Array,
    batch_size: int,
    *,
    dtype: jnp.dtype = jnp.float32,
    boundary_fraction: float = 0.0,
) -> tuple[Array, Array]:
    """Sample ``0 <= r <= t <= 1`` with optional boundary mass."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 <= boundary_fraction <= 1:
        raise ValueError("boundary_fraction must lie in [0, 1]")
    t_key, r_key, boundary_key = jax.random.split(key, 3)
    t = jax.random.uniform(t_key, (batch_size, 1), dtype=dtype)
    r = jax.random.uniform(r_key, (batch_size, 1), dtype=dtype) * t
    if boundary_fraction > 0:
        on_boundary = jax.random.uniform(boundary_key, (batch_size, 1)) < boundary_fraction
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
    boundary_fraction: float = 0.0,
    weights: Array | None = None,
    reduction: Reduction = "mean",
    return_details: bool = False,
) -> Array | IMFLossDetails:
    """Compute the improved MeanFlow compound regression objective.

    For ``z = (1-t)x + t e``, the implementation is

    ``v = u(z, c, t, t)``

    ``(u, dudt) = jvp(u, (z, r, t), (v, 0, 1))``

    ``V = u + (t-r) stop_gradient(dudt)``.

    ``V`` is regressed to ``e - x``. The stopped tangent matches the current
    PyTorch prototype while remaining differentiable in the field parameters.
    """

    if target.ndim != 2 or condition.ndim != 2 or target.shape[0] != condition.shape[0]:
        raise ValueError("target and condition must be matrices with matching batches")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("invalid reduction")
    if (r is None) != (t is None):
        raise ValueError("r and t must both be supplied or both omitted")
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
        )
    else:
        r_column = _time_column(r, target)
        t_column = _time_column(t, target)

    interpolated = (1.0 - t_column) * target + t_column * noise
    marginal_velocity = imf_field(params, interpolated, condition, t_column, t_column)

    def field(z_value: Array, r_value: Array, t_value: Array) -> Array:
        return imf_field(params, z_value, condition, r_value, t_value)

    value, tangent = jax.jvp(
        field,
        (interpolated, r_column, t_column),
        (marginal_velocity, jnp.zeros_like(r_column), jnp.ones_like(t_column)),
    )
    prediction = value + (t_column - r_column) * jax.lax.stop_gradient(tangent)
    regression_target = noise - target
    per_sample = jnp.mean(jnp.square(prediction - regression_target), axis=-1)
    if weights is not None:
        weights = jnp.asarray(weights, dtype=target.dtype).reshape((-1,))
        if weights.shape != (target.shape[0],):
            raise ValueError("weights must have one value per batch item")
        weighted = per_sample * weights
    else:
        weighted = per_sample
    if reduction == "none":
        loss = weighted
    elif reduction == "sum":
        loss = jnp.sum(weighted)
    elif weights is None:
        loss = jnp.mean(weighted)
    else:
        loss = jnp.sum(weighted) / jnp.maximum(jnp.sum(weights), jnp.finfo(target.dtype).tiny)

    if not return_details:
        return loss
    return IMFLossDetails(
        loss,
        per_sample,
        prediction,
        regression_target,
        value,
        tangent,
        marginal_velocity,
        interpolated,
        noise,
        r_column,
        t_column,
    )


jit_improved_meanflow_loss = partial(
    jax.jit,
    static_argnames=("boundary_fraction", "reduction", "return_details"),
)(improved_meanflow_loss)

