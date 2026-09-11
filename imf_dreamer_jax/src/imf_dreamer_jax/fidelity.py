"""Gradient and randomness controls for faithful posterior-to-prior matching."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array


IMFNoiseCoupling = Literal["independent", "posterior"]


def scale_condition_gradient(condition: Array, scale: float) -> Array:
    """Scale only the backward signal through a condition, preserving its value."""

    stopped = jax.lax.stop_gradient(condition)
    return stopped + scale * (condition - stopped)


def transport_base_noise(
    posterior_epsilon: Array,
    coupling: IMFNoiseCoupling,
) -> Array | None:
    """Select the transport base noise without leaking posterior gradients.

    ``None`` retains the historical independent draw inside the iMF loss.
    Repaired training passes the exact stopped posterior reparameterization
    noise so the transport endpoint and posterior target use one coupling.
    """

    if coupling == "independent":
        return None
    if coupling == "posterior":
        return jax.lax.stop_gradient(posterior_epsilon)
    raise ValueError("coupling must be 'independent' or 'posterior'")


def posterior_sequence_noise(
    key: Array,
    batch_size: int,
    time_steps: int,
    stochastic_dim: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Reproduce the exact epsilon sequence consumed by ``observe_sequence``."""

    if min(batch_size, time_steps, stochastic_dim) <= 0:
        raise ValueError("posterior noise dimensions must be positive")
    step_keys = jax.random.split(key, time_steps)
    step_major = jax.vmap(
        lambda step_key: jax.random.normal(
            step_key, (batch_size, stochastic_dim), dtype=dtype
        )
    )(step_keys)
    return jnp.swapaxes(step_major, 0, 1)
