"""Small dependency-free neural-network primitives implemented in JAX."""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


Params = dict[str, object]


def init_linear(
    key: Array,
    input_dim: int,
    output_dim: int,
    *,
    gain: float = math.sqrt(2.0),
) -> dict[str, Array]:
    weight = jax.random.normal(key, (input_dim, output_dim), dtype=jnp.float32)
    weight = weight * (gain / math.sqrt(input_dim))
    return {"weight": weight, "bias": jnp.zeros((output_dim,), dtype=jnp.float32)}


def linear(params: dict[str, Array], value: Array) -> Array:
    return value @ params["weight"] + params["bias"]


def init_mlp(
    key: Array,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    depth: int = 2,
    output_gain: float = 1.0,
) -> Params:
    keys = jax.random.split(key, depth + 1)
    layers: list[dict[str, Array]] = []
    width = input_dim
    for index in range(depth):
        layers.append(init_linear(keys[index], width, hidden_dim))
        width = hidden_dim
    layers.append(init_linear(keys[-1], width, output_dim, gain=output_gain))
    return {"layers": tuple(layers)}


def mlp(
    params: Params,
    value: Array,
    *,
    activation: Callable[[Array], Array] = jax.nn.elu,
) -> Array:
    layers = params["layers"]
    for layer in layers[:-1]:
        value = activation(linear(layer, value))
    return linear(layers[-1], value)


def init_gru(key: Array, input_dim: int, hidden_dim: int) -> Params:
    input_key, hidden_key = jax.random.split(key)
    return {
        "input": init_linear(input_key, input_dim, 3 * hidden_dim),
        "hidden": init_linear(hidden_key, hidden_dim, 3 * hidden_dim),
    }


def gru(params: Params, value: Array, hidden: Array) -> Array:
    input_gates = linear(params["input"], value)
    hidden_gates = linear(params["hidden"], hidden)
    input_reset, input_update, input_new = jnp.split(input_gates, 3, axis=-1)
    hidden_reset, hidden_update, hidden_new = jnp.split(hidden_gates, 3, axis=-1)
    reset = jax.nn.sigmoid(input_reset + hidden_reset)
    update = jax.nn.sigmoid(input_update + hidden_update)
    candidate = jnp.tanh(input_new + reset * hidden_new)
    return (1.0 - update) * candidate + update * hidden


def tree_global_norm(tree: object) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def clip_by_global_norm(tree: object, maximum: float) -> tuple[object, Array]:
    norm = tree_global_norm(tree)
    scale = jnp.minimum(1.0, maximum / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda value: value * scale, tree), norm

