"""Risk-sensitive CEM planner with shared aleatoric samples across candidates."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from .agent import critic
from .config import DreamerConfig, PlannerConfig
from .types import PlanResult, PyTree, RSSMState
from .world_model import (
    predict_continuation_logits,
    predict_reward,
    sample_prior,
    transition_deterministic,
)


def _rollout_single(
    params: PyTree,
    initial_state_value: RSSMState,
    action_sequence: Array,
    noise_sequence: Array,
    config: DreamerConfig,
) -> tuple[Array, Array, Array]:
    state = RSSMState(
        initial_state_value.deterministic[None], initial_state_value.stochastic[None]
    )

    def step(carry: RSSMState, inputs: tuple[Array, Array]):
        action_value, noise = inputs
        deterministic = transition_deterministic(
            params, carry, action_value[None], config
        )
        stochastic, _ = sample_prior(
            params, deterministic, None, config, noise=noise[None]
        )
        next_state = RSSMState(deterministic, stochastic)
        feature = next_state.feature
        output = (
            predict_reward(params, feature)[0],
            jax.nn.sigmoid(predict_continuation_logits(params, feature))[0],
            feature[0],
        )
        return next_state, output

    _, outputs = jax.lax.scan(step, state, (action_sequence, noise_sequence))
    return outputs


def evaluate_action_sequences(
    stacked_world_models: PyTree,
    critic_params: PyTree,
    initial_states: RSSMState,
    action_sequences: Array,
    key: Array,
    config: DreamerConfig,
    planner: PlannerConfig,
    *,
    behavior_mean: Array | None = None,
    behavior_std: Array | None = None,
    epistemic_thresholds: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Score candidates with common random numbers and ensemble disagreement."""

    if action_sequences.shape[1:] != (planner.horizon, config.action_dim):
        raise ValueError("action_sequences have the wrong shape")
    member_count = jax.tree_util.tree_leaves(stacked_world_models)[0].shape[0]
    if initial_states.deterministic.shape != (member_count, config.deterministic_dim):
        raise ValueError("initial deterministic states must have one row per ensemble member")
    if initial_states.stochastic.shape != (member_count, config.stochastic_dim):
        raise ValueError("initial stochastic states must have one row per ensemble member")
    noises = jax.random.normal(
        key,
        (member_count, planner.noise_samples, planner.horizon, config.stochastic_dim),
        dtype=action_sequences.dtype,
    )

    def rollout_member(params: PyTree, member_state: RSSMState, member_noises: Array):
        def rollout_noise(noise_sequence: Array):
            return jax.vmap(
                lambda actions: _rollout_single(
                    params, member_state, actions, noise_sequence, config
                )
            )(action_sequences)

        return jax.vmap(rollout_noise)(member_noises)

    rewards, continuations, features = jax.vmap(rollout_member)(
        stacked_world_models, initial_states, noises
    )
    # Arrays are [member, noise, candidate, horizon, ...]. Disagreement uses
    # member means so aleatoric variability is not mislabeled epistemic.
    member_feature_means = jnp.mean(features, axis=1)
    disagreement = jnp.mean(jnp.var(member_feature_means, axis=0), axis=-1)
    if epistemic_thresholds is None:
        epistemic_thresholds = jnp.full(
            (planner.horizon,), planner.epistemic_threshold, dtype=action_sequences.dtype
        )
    if epistemic_thresholds.shape != (planner.horizon,):
        raise ValueError("epistemic_thresholds must have shape [horizon]")
    safe = disagreement <= epistemic_thresholds[None]
    active = jnp.cumprod(safe.astype(action_sequences.dtype), axis=-1)
    effective_horizon = jnp.sum(active, axis=-1)

    discounts = planner.discount ** jnp.arange(planner.horizon, dtype=action_sequences.dtype)
    survival = jnp.cumprod(
        jnp.concatenate((jnp.ones_like(continuations[..., :1]), continuations[..., :-1]), axis=-1),
        axis=-1,
    )
    path_returns = jnp.sum(
        rewards * survival * discounts[None, None, None] * active[None, None], axis=-1
    )
    member_returns = jnp.mean(path_returns, axis=1)
    expected_return = jnp.mean(member_returns, axis=0)
    epistemic_std = jnp.sqrt(jnp.maximum(jnp.var(member_returns, axis=0), 0.0))

    if planner.terminal_value:
        initial_feature = jnp.mean(initial_states.feature, axis=0)
        mean_features = jnp.mean(features, axis=(0, 1))
        candidate_count = action_sequences.shape[0]
        feature_path = jnp.concatenate(
            (
                jnp.broadcast_to(initial_feature, (candidate_count, 1, config.feature_dim)),
                mean_features,
            ),
            axis=1,
        )
        gather_index = effective_horizon.astype(jnp.int32)
        terminal_feature = feature_path[jnp.arange(candidate_count), gather_index]
        terminal_discount = planner.discount**effective_horizon
        survival_after = jnp.cumprod(continuations, axis=-1)
        mean_survival_after = jnp.mean(survival_after, axis=(0, 1))
        survival_path = jnp.concatenate(
            (jnp.ones((candidate_count, 1), dtype=action_sequences.dtype), mean_survival_after),
            axis=1,
        )
        terminal_survival = survival_path[jnp.arange(candidate_count), gather_index]
        expected_return = expected_return + terminal_discount * terminal_survival * critic(
            critic_params, terminal_feature
        )

    if (behavior_mean is None) != (behavior_std is None):
        raise ValueError("behavior_mean and behavior_std must be supplied together")
    if behavior_mean is None:
        behavior_penalty = jnp.zeros_like(expected_return)
    else:
        if behavior_mean.shape != (planner.horizon, config.action_dim):
            raise ValueError("behavior prior has the wrong shape")
        standardized = (action_sequences - behavior_mean[None]) / jnp.maximum(
            behavior_std[None], 1e-4
        )
        behavior_penalty = jnp.mean(jnp.square(standardized), axis=(1, 2))
    score = (
        expected_return
        - planner.risk_beta * epistemic_std
        - planner.behavior_weight * behavior_penalty
    )
    return score, expected_return, epistemic_std, effective_horizon


def cem_plan(
    stacked_world_models: PyTree,
    critic_params: PyTree,
    initial_states: RSSMState,
    key: Array,
    config: DreamerConfig,
    planner: PlannerConfig,
    *,
    behavior_mean: Array | None = None,
    behavior_std: Array | None = None,
    epistemic_thresholds: Array | None = None,
) -> PlanResult:
    """Optimize a bounded action sequence and return its first action."""

    mean = jnp.zeros((planner.horizon, config.action_dim), dtype=jnp.float32)
    std = jnp.full_like(mean, planner.max_std)
    iteration_keys = jax.random.split(key, planner.iterations + 1)
    for index in range(planner.iterations):
        sample_key, evaluation_key = jax.random.split(iteration_keys[index])
        candidates = jnp.clip(
            mean[None]
            + std[None]
            * jax.random.normal(
                sample_key,
                (planner.population, planner.horizon, config.action_dim),
                dtype=mean.dtype,
            ),
            -1.0,
            1.0,
        )
        scores, _, _, _ = evaluate_action_sequences(
            stacked_world_models,
            critic_params,
            initial_states,
            candidates,
            evaluation_key,
            config,
            planner,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            epistemic_thresholds=epistemic_thresholds,
        )
        _, elite_indices = jax.lax.top_k(scores, planner.elite_count)
        elites = candidates[elite_indices]
        mean = jnp.mean(elites, axis=0)
        std = jnp.clip(jnp.std(elites, axis=0), planner.min_std, planner.max_std)

    score, expected, epistemic, effective = evaluate_action_sequences(
        stacked_world_models,
        critic_params,
        initial_states,
        mean[None],
        iteration_keys[-1],
        config,
        planner,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        epistemic_thresholds=epistemic_thresholds,
    )
    return PlanResult(
        jnp.clip(mean[0], -1.0, 1.0),
        mean,
        std,
        score[0],
        expected[0],
        epistemic[0],
        effective[0],
    )


jit_evaluate_action_sequences = partial(
    jax.jit, static_argnames=("config", "planner")
)(evaluate_action_sequences)
jit_cem_plan = partial(jax.jit, static_argnames=("config", "planner"))(cem_plan)
