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


class ActorSample(NamedTuple):
    action: Array
    pre_tanh: Array
    log_prob: Array
    entropy: Array
    distribution: DiagonalNormal


class SequenceStates(NamedTuple):
    states: RSSMState
    posterior: DiagonalNormal
    prior_mean: Array | None
    prior_std: Array | None


class PriorSample(NamedTuple):
    stochastic: Array
    distribution: DiagonalNormal | None
    nfe: Array


class ParameterCounts(NamedTuple):
    active: int
    total: int
    prior_active: int
    prior_total: int


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
    raw_loss_u: Array
    raw_loss_v: Array
    adaptive_weight_u: Array
    adaptive_weight_v: Array
    velocity_prediction: Array
    endpoint_prediction: Array
    raw_loss_endpoint: Array
    shortcut_prediction: Array
    shortcut_target: Array
    raw_loss_shortcut: Array
    signal_weights: Array


class WorldModelLoss(NamedTuple):
    total: Array
    reconstruction: Array
    reward: Array
    continuation: Array
    prior: Array
    representation: Array
    overshooting: Array
    imf_loss_u: Array
    imf_loss_v: Array
    imf_endpoint: Array
    imf_shortcut: Array
    overshooting_distance_5: Array
    overshooting_distance_15: Array
    causal_consistency: Array
    causal_observation: Array
    causal_reward: Array


class Imagination(NamedTuple):
    features: Array
    actions: Array
    rewards: Array
    continuations: Array
    entropies: Array
    log_probs: Array
    pre_tanh_means: Array
    pre_tanh_values: Array


class ActorCriticMetrics(NamedTuple):
    actor_loss: Array
    critic_loss: Array
    mean_imagined_return: Array
    actor_grad_norm: Array
    critic_grad_norm: Array
    action_saturation: Array
    mean_pre_tanh_abs: Array
    squashed_entropy: Array
    return_scale: Array
    slow_critic_delta: Array


class RunningRMSState(NamedTuple):
    """Online root-mean-square state used by policy-aligned objectives."""

    mean_square: Array
    count: Array


class AdvantageConsistencyLoss(NamedTuple):
    """Decomposed short-horizon, action-advantage consistency loss."""

    total: Array
    magnitude: Array
    ranking: Array
    flat: Array
    predicted_advantages: Array
    target_advantages: Array
    informative_pair_fraction: Array


class BootstrapRewardEnsembleState(NamedTuple):
    """Independent bootstrap reward heads and their joint optimizer state."""

    params: PyTree
    optimizer: PyTree


class PolicyConsistencyLoss(NamedTuple):
    """World-model objective with optional decision and exposure terms."""

    total: Array
    base: WorldModelLoss
    advantage: AdvantageConsistencyLoss
    exposure_meanflow: Array
    endpoint: Array


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
    slow_critic: PyTree | None = None
    world_model_teacher: PyTree | None = None


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
