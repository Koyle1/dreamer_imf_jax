"""JAX PyTree-compatible value types used by the public API."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import Array


PyTree = Any


class RSSMState(NamedTuple):
    deterministic: Array
    stochastic: Array

    @property
    def feature(self) -> Array:
        return jnp.concatenate((self.deterministic, self.stochastic), axis=-1)


class DiagonalNormal(NamedTuple):
    mean: Array
    std: Array


class SequenceStates(NamedTuple):
    states: RSSMState
    posterior: DiagonalNormal
    prior_mean: Array | None
    prior_std: Array | None


class IMFLossDetails(NamedTuple):
    loss: Array
    per_sample_loss: Array
    prediction: Array
    regression_target: Array
    field: Array
    jvp: Array
    marginal_velocity: Array
    interpolated: Array
    noise: Array
    r: Array
    t: Array


class WorldModelLoss(NamedTuple):
    total: Array
    reconstruction: Array
    reward: Array
    continuation: Array
    prior: Array
    representation: Array
    overshooting: Array


class Imagination(NamedTuple):
    features: Array
    actions: Array
    rewards: Array
    continuations: Array
    entropies: Array


class ActorCriticMetrics(NamedTuple):
    actor_loss: Array
    critic_loss: Array
    mean_imagined_return: Array


class AgentParams(NamedTuple):
    world_model: PyTree
    actor: PyTree
    critic: PyTree


class AdamState(NamedTuple):
    step: Array
    first_moment: PyTree
    second_moment: PyTree


class AgentState(NamedTuple):
    params: AgentParams
    model_optimizer: AdamState
    actor_optimizer: AdamState
    critic_optimizer: AdamState


class PredictiveMoments(NamedTuple):
    samples: Array
    member_means: Array
    mean: Array
    predictive_variance: Array
    aleatoric_variance: Array
    epistemic_variance: Array


class PlanResult(NamedTuple):
    action: Array
    mean_sequence: Array
    std_sequence: Array
    score: Array
    expected_return: Array
    epistemic_std: Array
    effective_horizon: Array

