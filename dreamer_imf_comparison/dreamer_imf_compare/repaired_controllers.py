"""Frozen-checkpoint endpoint control with explicit value and execution semantics.

The default latent reference preserves the original intervention. Selecting an
endpoint reference changes the search domain and is a separate experiment.
Neither choice fits parameters or changes the frozen actor, reward head or Q.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from imf_dreamer_jax import RSSMState, decode, rebrac_actor, rebrac_critics
from imf_dreamer_jax.robust_flowmpc import (
    ActionSequenceDomainConfig,
    ActionSequenceProposal,
    CEMActionSequenceConfig,
    feasible_first_cem_action_sequence_search,
    make_action_sequence_domain,
)
from imf_dreamer_jax.world_model import (
    predict_state_action_reward_from_observation,
    sample_prior,
    transition_deterministic,
)

from .actor_gap_model_training import sample_endpoint_observation


def _configuration(model, actor_seed, direct_any_step, controller):
    # Changes to these settings require a new protocol, not silent defaults.
    expected = {
        "horizon": 5,
        "action_sequence_particles": 64,
        "action_sequence_objective_evaluations": 10,
        "action_sequence_residual_limit": 0.1,
    }
    for name, value in expected.items():
        actual = controller.get(name)
        if isinstance(actual, bool) or actual != value:
            raise ValueError(f"unsupported endpoint controller {name}: {actual!r}")
    cfg, rc = model["config"], model["rebrac_config"]
    if cfg.action_dim != rc.action_dim or cfg.observation_dim != rc.state_dim:
        raise ValueError("endpoint and ReBRAC dimensions differ")
    family = "endpoint_anystep_imf" if direct_any_step else "endpoint_h1_imf"
    checkpoint = model["endpoints"][family]
    if int(checkpoint["maximum_chunk_horizon"]) < (5 if direct_any_step else 1):
        raise ValueError("endpoint checkpoint cannot represent the controller horizon")
    threshold = float(model["thresholds"][actor_seed])
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("behavior threshold must be finite and nonnegative")
    return cfg, rc, checkpoint, model["policies"][actor_seed], threshold


def _endpoint_predictor(checkpoint, cfg):
    maximum_horizon = int(checkpoint["maximum_chunk_horizon"])
    mean = jnp.asarray(checkpoint["observation_mean"], jnp.float32)
    std = jnp.asarray(checkpoint["observation_std"], jnp.float32)

    def predict(start, actions, horizon, noise):
        padded = jnp.zeros((maximum_horizon, cfg.action_dim), actions.dtype)
        padded = padded.at[: actions.shape[0]].set(actions)
        particles = start.shape[0]
        return sample_endpoint_observation(
            checkpoint["params"],
            start,
            jnp.broadcast_to(padded, (particles, maximum_horizon, cfg.action_dim)),
            horizon,
            cfg,
            observation_mean=mean,
            observation_std=std,
            noise=noise,
        )

    return predict


def make_endpoint_scorer(
    model: Mapping[str, Any],
    actor_seed: int,
    *,
    direct_any_step: bool,
    controller: Mapping[str, Any],
):
    """Return JIT ``score(current, sequence, noises)`` for a fixed action plan.

    ``current`` is [1, observation_dim], ``sequence`` is [5, action_dim], and
    noises are [64, 5, observation_dim]. Stage reward is the discounted sum of
    predicted rewards, terminal_value is gamma**5 times the minimum critic.
    It is a behavior-regularized critic surrogate, not pure environment return.
    reward_mean/reward_second_moment are unweighted per-step particle moments.
    Feasibility uses the maximum over time of particle-mean RMS policy drift.
    """

    cfg, rc, checkpoint, policy, threshold = _configuration(
        model, actor_seed, direct_any_step, controller
    )
    horizon = int(controller["horizon"])
    particles = int(controller["action_sequence_particles"])
    predict = _endpoint_predictor(checkpoint, cfg)
    reward_head = model["reward_world"]["reward_transition"]

    def score(current, sequence, noises):
        current = jnp.asarray(current, jnp.float32)
        sequence = jnp.asarray(sequence, jnp.float32)
        noises = jnp.asarray(noises, jnp.float32)
        if current.shape != (1, *cfg.observation_shape):
            raise ValueError("endpoint current observation has the wrong shape")
        if sequence.shape != (horizon, cfg.action_dim):
            raise ValueError("endpoint action sequence has the wrong shape")
        if noises.shape != (particles, horizon, cfg.observation_dim):
            raise ValueError("endpoint noise bank has the wrong shape")
        start = jnp.broadcast_to(current, (particles, *cfg.observation_shape))
        observation = start
        rewards, reward_second_moments, observations, distances = [], [], [], []
        for step in range(horizon):
            action = jnp.broadcast_to(sequence[step], (particles, cfg.action_dim))
            policy_action = rebrac_actor(policy.actor, observation)
            distances.append(
                jnp.mean(jnp.sqrt(jnp.mean((action - policy_action) ** 2, -1)))
            )
            reward = predict_state_action_reward_from_observation(
                reward_head, observation, action, cfg
            )
            rewards.append(jnp.mean(reward))
            reward_second_moments.append(jnp.mean(reward**2))
            observations.append(jnp.mean(observation, axis=0))
            if direct_any_step:
                observation = predict(
                    start, sequence[: step + 1], step + 1, noises[:, step]
                )
            else:
                observation = predict(
                    observation, sequence[step : step + 1], 1, noises[:, step]
                )
        terminal_action = rebrac_actor(policy.actor, observation)
        terminal_q = jnp.mean(
            jnp.min(
                rebrac_critics(policy.critics, observation, terminal_action), axis=0
            )
        )
        reward_mean = jnp.stack(rewards)
        stage_reward = jnp.sum(
            reward_mean
            * jnp.asarray(
                [float(rc.discount) ** t for t in range(horizon)], jnp.float32
            )
        )
        terminal_value = (float(rc.discount) ** horizon) * terminal_q
        objective = stage_reward + terminal_value
        distance = jnp.max(jnp.stack(distances))
        violation = jnp.maximum(distance - threshold, 0.0)
        finite = (
            jnp.isfinite(objective)
            & jnp.isfinite(distance)
            & jnp.all(jnp.isfinite(sequence))
        )
        action_bounds = jnp.all(jnp.abs(sequence) <= 1.0)
        return {
            "stage_reward": stage_reward,
            "terminal_value": terminal_value,
            "terminal_q_raw": terminal_q,
            "objective": objective,
            "behavior_distance": distance,
            "feasible": finite & action_bounds & (distance <= threshold),
            "violation": violation,
            "reward_mean": reward_mean,
            "reward_by_step": reward_mean,
            "reward_second_moment": jnp.stack(reward_second_moments),
            "observations_mean": jnp.stack(observations),
        }

    return jax.jit(score)


def make_endpoint_planner(
    model: Mapping[str, Any],
    actor_seed: int,
    *,
    direct_any_step: bool,
    controller: Mapping[str, Any],
    reference_mode: str = "latent",
):
    """Return a pure JIT planner with separate proposal/reference/executed scores.

    The callable accepts (belief, current, noises, proposals), where proposals
    are a supplied [2, 4, 5, action_dim] standard-normal bank. Ten plan scores
    cover initial reference, two four-member populations, and final CEM mean.
    No post-selection scoring is needed: execution reuses the selected metrics.

    action_sequence is the search proposal; executed_sequence is the entire
    reference on rejection. actions is its first action. A reference can itself
    be infeasible on the stochastic endpoint bank; this is reported faithfully.
    The optional endpoint reference uses this same endpoint model with zero
    noise; it does not ensure stochastic reference feasibility.
    """

    if reference_mode not in ("latent", "endpoint"):
        raise ValueError("reference_mode must be 'latent' or 'endpoint'")
    cfg, _, checkpoint, policy, _ = _configuration(
        model, actor_seed, direct_any_step, controller
    )
    horizon = int(controller["horizon"])
    world = model["reward_world"]
    predict = _endpoint_predictor(checkpoint, cfg)
    score = make_endpoint_scorer(
        model, actor_seed, direct_any_step=direct_any_step, controller=controller
    )
    domain_config = ActionSequenceDomainConfig(
        horizon, cfg.action_dim, float(controller["action_sequence_residual_limit"])
    )
    cem_config = CEMActionSequenceConfig(
        iterations=2,
        population=4,
        elite_count=2,
        initial_std=0.1,
        minimum_std=0.01,
        maximum_std=0.1,
    )

    def reference_sequence(belief, current):
        state, observation, actions = belief, current, []
        for step in range(horizon):
            action = rebrac_actor(policy.actor, observation)[0]
            actions.append(action)
            # The final observation is never queried by the reference policy.
            if step == horizon - 1:
                continue
            if reference_mode == "latent":
                deterministic = transition_deterministic(
                    world, state, action[None], cfg
                )
                stochastic, _ = sample_prior(
                    world,
                    deterministic,
                    None,
                    cfg,
                    noise=jnp.zeros((1, cfg.stochastic_dim), jnp.float32),
                )
                state = RSSMState(deterministic, stochastic)
                observation = decode(world, state.feature, cfg)
            else:
                sequence = jnp.stack(actions)
                observation = predict(
                    current if direct_any_step else observation,
                    sequence if direct_any_step else sequence[-1:],
                    step + 1 if direct_any_step else 1,
                    jnp.zeros((1, cfg.observation_dim), jnp.float32),
                )
        return jnp.stack(actions)

    def plan(belief, current, noises, proposals):
        reference = reference_sequence(belief, current)
        domain = make_action_sequence_domain(reference, domain_config)
        result = feasible_first_cem_action_sequence_search(
            lambda sequence: score(current, sequence, noises),
            domain,
            ActionSequenceProposal(jnp.zeros_like(reference)),
            proposals,
            cem_config,
        )
        proposal_metrics, reference_metrics = (
            result["final_metrics"],
            result["initial_metrics"],
        )
        used_fallback = ~proposal_metrics["feasible"]
        executed = jnp.where(used_fallback, reference, result["action_sequence"])
        executed_metrics = jax.tree_util.tree_map(
            lambda proposed, baseline: jnp.where(used_fallback, baseline, proposed),
            proposal_metrics,
            reference_metrics,
        )
        first_population = (
            jnp.clip(
                cem_config.initial_std * proposals[0],
                domain.residual_minimum[None],
                domain.residual_maximum[None],
            )
            .at[0]
            .set(jnp.zeros_like(reference))
        )
        output = {
            "action_sequence": result["action_sequence"],
            "reference_sequence": reference,
            "executed_sequence": executed,
            "proposal_sequences": reference[None] + first_population,
            "actions": executed[0],
            "used_fallback": used_fallback,
            "evaluations": result["evaluations"],
            "residual_norm": jnp.linalg.norm(result["residuals"]),
            "proposal": proposal_metrics,
            "reference": reference_metrics,
            "executed": executed_metrics,
        }
        for prefix, metrics in (
            ("proposal", proposal_metrics),
            ("reference", reference_metrics),
            ("executed", executed_metrics),
        ):
            output.update(
                {f"{prefix}_{name}": value for name, value in metrics.items()}
            )
        return output

    return jax.jit(plan)
