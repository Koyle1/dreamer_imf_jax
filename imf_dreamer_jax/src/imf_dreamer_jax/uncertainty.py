"""Ensemble sampling and proper distributional diagnostics."""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .nn import clip_by_global_norm, init_mlp, mlp
from .optim import adam_update, init_adam
from .types import BootstrapRewardEnsembleState, PredictiveMoments, PyTree
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


def init_bootstrap_reward_ensemble(
    key: Array,
    feature_dim: int,
    hidden_dim: int,
    *,
    members: int = 5,
) -> BootstrapRewardEnsembleState:
    """Initialize independent reward heads over one shared latent space.

    The heads are deterministic conditional predictors.  Their between-head
    disagreement is therefore epistemic; no process-noise samples enter the
    statistic.
    """

    if min(feature_dim, hidden_dim, members) <= 0:
        raise ValueError("feature_dim, hidden_dim, and members must be positive")
    params = stack_ensemble(
        [
            init_mlp(member_key, feature_dim, hidden_dim, 1)
            for member_key in jax.random.split(key, members)
        ]
    )
    return BootstrapRewardEnsembleState(params, init_adam(params))


def reward_ensemble_predictions(stacked_params: PyTree, features: Array) -> Array:
    """Return deterministic predictions shaped ``[member, *batch]``."""

    if features.ndim < 2:
        raise ValueError("features must have a final feature dimension")
    leaves = jax.tree_util.tree_leaves(stacked_params)
    if not leaves or leaves[0].ndim == 0:
        raise ValueError("stacked_params must contain a leading member axis")
    predictions = jax.vmap(lambda params: mlp(params, features)[..., 0])(
        stacked_params
    )
    return predictions


def epistemic_reward_std(stacked_params: PyTree, features: Array) -> Array:
    """Between-bootstrap reward standard deviation, excluding aleatoric noise."""

    predictions = reward_ensemble_predictions(stacked_params, features)
    return jnp.sqrt(jnp.maximum(jnp.var(predictions, axis=0), 0.0))


def pessimistic_rewards(
    rewards: Array,
    features: Array,
    stacked_params: PyTree,
    *,
    penalty_scale: float,
) -> tuple[Array, Array]:
    """Subtract an epistemic lower-confidence penalty from imagined rewards."""

    if not isinstance(penalty_scale, (int, float)) or penalty_scale < 0.0:
        raise ValueError("penalty_scale must be a nonnegative scalar")
    disagreement = epistemic_reward_std(stacked_params, features)
    if disagreement.shape != rewards.shape:
        raise ValueError("reward ensemble output must match rewards")
    return rewards - penalty_scale * disagreement, disagreement


def train_bootstrap_reward_ensemble(
    state: BootstrapRewardEnsembleState,
    features: Array,
    rewards: Array,
    key: Array,
    *,
    bootstrap_mask: Array | None = None,
    learning_rate: float = 3e-4,
    grad_clip: float = 100.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[BootstrapRewardEnsembleState, Array]:
    """Fit bootstrap reward heads without modifying the shared world model."""

    if features.ndim < 2 or rewards.shape != features.shape[:-1]:
        raise ValueError("rewards must match the leading feature axes")
    member_count = jax.tree_util.tree_leaves(state.params)[0].shape[0]
    flat_features = jax.lax.stop_gradient(features.reshape((-1, features.shape[-1])))
    flat_rewards = jax.lax.stop_gradient(rewards.reshape((-1,)))
    if bootstrap_mask is None:
        bootstrap_mask = jax.random.bernoulli(
            key, 0.632, (member_count, flat_rewards.shape[0])
        ).astype(flat_rewards.dtype)
    else:
        bootstrap_mask = jnp.asarray(bootstrap_mask, dtype=flat_rewards.dtype)
        if bootstrap_mask.shape != (member_count, flat_rewards.shape[0]):
            raise ValueError("bootstrap_mask must have shape [member, sample]")

    def objective(params: PyTree) -> Array:
        predictions = reward_ensemble_predictions(params, flat_features)
        squared = jnp.square(predictions - flat_rewards[None])
        per_member = jnp.sum(squared * bootstrap_mask, axis=1) / jnp.maximum(
            jnp.sum(bootstrap_mask, axis=1), 1.0
        )
        return jnp.mean(per_member)

    loss, gradients = jax.value_and_grad(objective)(state.params)
    gradients, _ = clip_by_global_norm(gradients, grad_clip)
    params, optimizer = adam_update(
        state.params,
        gradients,
        state.optimizer,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    return BootstrapRewardEnsembleState(params, optimizer), loss


jit_sample_prior_ensemble = partial(
    jax.jit, static_argnames=("config", "noise_samples")
)(sample_prior_ensemble)

jit_train_bootstrap_reward_ensemble = jax.jit(train_bootstrap_reward_ensemble)
