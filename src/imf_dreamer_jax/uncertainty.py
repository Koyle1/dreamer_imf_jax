"""Ensemble sampling and proper distributional diagnostics."""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .types import PredictiveMoments, PyTree
from .world_model import sample_prior


def stack_ensemble(members: Sequence[PyTree]) -> PyTree:
    """Stack identically structured parameter PyTrees along a member axis."""

    if not members:
        raise ValueError("at least one ensemble member is required")
    reference = jax.tree_util.tree_structure(members[0])
    if any(jax.tree_util.tree_structure(member) != reference for member in members[1:]):
        raise ValueError("ensemble members must have identical parameter structure")
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *members)


def sample_prior_ensemble(
    stacked_params: PyTree,
    condition: Array,
    key: Array,
    config: DreamerConfig,
    *,
    noise_samples: int = 8,
) -> Array:
    """Return samples shaped ``[member, noise, batch, stochastic]``."""

    if condition.ndim != 2 or noise_samples <= 0:
        raise ValueError("condition must be rank two and noise_samples must be positive")
    member_count = jax.tree_util.tree_leaves(stacked_params)[0].shape[0]
    member_keys = jax.random.split(key, member_count)

    def sample_member(params: PyTree, member_key: Array) -> Array:
        keys = jax.random.split(member_key, noise_samples)
        return jax.vmap(lambda sample_key: sample_prior(params, condition, sample_key, config)[0])(
            keys
        )

    return jax.vmap(sample_member)(stacked_params, member_keys)


def predictive_moments(samples: Array) -> PredictiveMoments:
    if samples.ndim != 4:
        raise ValueError("samples must have shape [member, noise, batch, target]")
    member_means = jnp.mean(samples, axis=1)
    mean = jnp.mean(member_means, axis=0)
    predictive_variance = jnp.var(samples.reshape((-1, *samples.shape[2:])), axis=0)
    aleatoric = jnp.mean(jnp.var(samples, axis=1), axis=0)
    epistemic = jnp.var(member_means, axis=0)
    return PredictiveMoments(
        samples, member_means, mean, predictive_variance, aleatoric, epistemic
    )


def energy_score(samples: Array, target: Array) -> Array:
    """Multivariate energy score using all independent off-diagonal draw pairs."""

    if samples.ndim == 4:
        samples = samples.reshape((-1, *samples.shape[2:]))
    if samples.ndim != 3 or target.shape != samples.shape[1:]:
        raise ValueError("samples must be [draw, batch, target] and target [batch, target]")
    draw_count = samples.shape[0]
    if draw_count < 2:
        raise ValueError("energy score requires at least two draws")
    observation_term = jnp.mean(jnp.linalg.norm(samples - target[None], axis=-1), axis=0)
    pair_distances = jnp.linalg.norm(
        samples[:, None] - samples[None, :], axis=-1
    )
    pair_term = jnp.sum(pair_distances, axis=(0, 1)) / (draw_count * (draw_count - 1))
    return observation_term - 0.5 * pair_term


jit_sample_prior_ensemble = partial(
    jax.jit, static_argnames=("config", "noise_samples")
)(sample_prior_ensemble)

