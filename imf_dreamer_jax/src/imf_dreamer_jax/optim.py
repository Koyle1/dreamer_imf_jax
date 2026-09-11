"""Minimal JAX Adam implementation used to keep the package dependency-free."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .types import AdamState, PyTree


def init_adam(params: PyTree) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def adam_update(
    params: PyTree,
    gradients: PyTree,
    state: AdamState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[PyTree, AdamState]:
    step = state.step + 1
    first = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state.first_moment,
        gradients,
    )
    second = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        state.second_moment,
        gradients,
    )
    correction1 = 1.0 - beta1**step.astype(jnp.float32)
    correction2 = 1.0 - beta2**step.astype(jnp.float32)

    def apply(parameter: Array, first_value: Array, second_value: Array) -> Array:
        first_hat = first_value / correction1
        second_hat = second_value / correction2
        return parameter - learning_rate * first_hat / (jnp.sqrt(second_hat) + epsilon)

    updated = jax.tree_util.tree_map(apply, params, first, second)
    return updated, AdamState(step, first, second)

