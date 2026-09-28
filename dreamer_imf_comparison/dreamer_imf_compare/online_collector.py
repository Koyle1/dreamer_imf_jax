"""Episode-block A1 collection with dynamic checkpoint parameters."""

from __future__ import annotations

import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import initial_state, observe_step, rebrac_actor
from imf_dreamer_jax.flowmpc import (
    FlowMPCConfig,
    jit_flowmpc_adapt_actor,
    jit_flowmpc_objective,
)
from imf_dreamer_jax.nn import tree_global_norm
from imf_dreamer_jax.robust_flowmpc import (
    ActionTrustRegionConfig,
    PersistenceConfig,
    init_reference_actor_state,
    jit_backtrack_actor_proposal,
    jit_finish_actor_adaptation_step,
    measure_actor_reference_action_drift,
)
from .dmc import DMCAdapter
from .matched_objective_benchmark import _tree_digest, derive_jax_key


class OnlineCollector:
    """One reusable compiled controller; beliefs and carried actors are episode-local.

    Only architecture/configuration is closed over by JIT. World, critics,
    reference actor, and fixed anchors are explicit dynamic array arguments.
    """

    def __init__(self, cfg: Any, rc: Any, controller: dict[str, Any]):
        self.cfg = cfg
        self.flow = FlowMPCConfig(
            horizon=controller["horizon"],
            particles=controller["flowmpc_particles"],
            step_size=controller["flowmpc_step_size"],
            discount=rc.discount,
        )
        trust = ActionTrustRegionConfig(
            anchor_mean_squared_budget=controller["trust_anchor_mse_sum_budget"],
            current_max_absolute_budget=controller["trust_current_linf_budget"],
        )
        persistence = PersistenceConfig(mode="persistent")
        flow = self.flow

        @jax.jit
        def observe(world, current, previous, belief, key):
            return observe_step(world, current, previous, belief, key, cfg)[0]

        @jax.jit
        def select(world, critics, anchors, state, belief, current, noises):
            start = state.carried_actor
            update = jit_flowmpc_adapt_actor(
                start, critics, world, belief, current, noises, cfg, rc, flow
            )
            first = jit_backtrack_actor_proposal(
                state.frozen_reference_actor,
                start,
                update.actor,
                anchors,
                current,
                trust,
            )
            # Match the repaired A1 implementation's executed-actor projection.
            second = jit_backtrack_actor_proposal(
                state.frozen_reference_actor,
                start,
                first.actor,
                anchors,
                current,
                trust,
            )
            selected = second.actor
            drift = measure_actor_reference_action_drift(
                state.frozen_reference_actor, selected, anchors, current, trust
            )
            objective = jit_flowmpc_objective(
                selected, critics, world, belief, current, noises, cfg, rc, flow
            ).objective
            telemetry = dict(
                objective_before=update.objective_before,
                objective_after=objective,
                gradient_norm=update.gradient_norm,
                parameter_delta=tree_global_norm(
                    jax.tree_util.tree_map(lambda a, b: a - b, selected, start)
                ),
                anchor_drift=drift.anchor_mean_squared_action_drift,
                current_drift=drift.current_max_absolute_action_drift,
                within_reference_budgets=drift.feasible,
                backtracks=first.backtracks + second.backtracks,
                used_reference_fallback=jnp.logical_or(
                    first.used_reference_fallback, second.used_reference_fallback
                ),
            )
            return (
                jit_finish_actor_adaptation_step(state, selected, persistence),
                rebrac_actor(selected, current),
                telemetry,
            )

        self.observe = observe
        self.select = select
        self._warmed = False

    def rollout(
        self,
        model,
        actor_seed,
        world_seed,
        env_seed,
        *,
        maximum_steps=1000,
        exploration_std=0.0,
        training=False,
    ):
        if (
            isinstance(maximum_steps, bool)
            or not isinstance(maximum_steps, int)
            or maximum_steps <= 0
        ):
            raise ValueError("maximum_steps must be a positive integer")
        if not np.isfinite(exploration_std) or exploration_std < 0:
            raise ValueError("exploration_std must be finite and nonnegative")
        if not training and exploration_std != 0:
            raise ValueError("evaluation exploration is forbidden")
        cfg, flow = self.cfg, self.flow
        world, policy = model["reward_world"], model["policies"][actor_seed]
        anchors = jnp.asarray(model["anchors"], jnp.float32)
        before = _tree_digest((world, policy, anchors))
        task = model.get("task", "dmc_reacher_hard")
        posterior_key = derive_jax_key(
            "flowmpc-posterior", task, world_seed, actor_seed, env_seed
        )
        noise_key = derive_jax_key(
            "flowmpc-noise", task, world_seed, actor_seed, env_seed
        )
        exploration_key = derive_jax_key(
            "online-exploration", task, world_seed, actor_seed, env_seed
        )
        environment = DMCAdapter(task, seed=int(env_seed), action_repeat=1)
        trace_lists: dict[str, list] = {}
        elapsed, discarded = 0.0, 0
        try:
            observation = np.asarray(environment.reset(), np.float32)
            observations = [observation.copy()]
            belief = initial_state(cfg, 1)
            previous = jnp.zeros((1, cfg.action_dim), jnp.float32)
            state = init_reference_actor_state(policy.actor)
            for step in range(maximum_steps):
                current = jnp.asarray(observation[None], jnp.float32)
                key = jax.random.fold_in(posterior_key, step)
                noises = jax.random.normal(
                    jax.random.fold_in(noise_key, step),
                    (flow.particles, flow.horizon, cfg.stochastic_dim),
                )
                if not self._warmed:
                    warm_belief = self.observe(world, current, previous, belief, key)
                    warm_result = self.select(
                        world,
                        policy.critics,
                        anchors,
                        state,
                        warm_belief,
                        current,
                        noises,
                    )
                    jax.block_until_ready(warm_result)
                    self._warmed = True
                    discarded = 1
                started = time.perf_counter()
                belief = self.observe(world, current, previous, belief, key)
                state, clean, telemetry = self.select(
                    world, policy.critics, anchors, state, belief, current, noises
                )
                clean_action = np.asarray(jax.device_get(clean[0]), np.float32)
                action = clean_action.copy()
                if exploration_std:
                    noise = np.asarray(
                        jax.random.normal(
                            jax.random.fold_in(exploration_key, step), action.shape
                        )
                    )
                    action = np.clip(action + exploration_std * noise, -1, 1).astype(
                        np.float32
                    )
                host_telemetry = {
                    k: np.asarray(v) for k, v in jax.device_get(telemetry).items()
                }
                if not np.isfinite(action).all() or not all(
                    np.isfinite(v).all() for v in host_telemetry.values()
                ):
                    raise ValueError("nonfinite online controller output")
                elapsed += time.perf_counter() - started
                transition = environment.step(action)
                values = dict(
                    actions=action.copy(),
                    clean_actions=clean_action.copy(),
                    observations=observation.copy(),
                    rewards=float(transition.reward),
                    continuations=float(transition.continuation),
                    is_last=bool(transition.is_last),
                    **host_telemetry,
                )
                for name, value in values.items():
                    trace_lists.setdefault(name, []).append(value)
                observation = np.asarray(transition.observation, np.float32)
                observations.append(observation.copy())
                # Posterior inference must condition on the actually executed action.
                previous = jnp.asarray(action[None], jnp.float32)
                if transition.is_last:
                    break
        finally:
            environment.close()
        trace = {k: np.asarray(v)[None] for k, v in trace_lists.items()}
        steps = len(trace_lists["rewards"])
        trace["lengths"] = np.asarray([steps], np.int32)
        trace["evaluation_seeds"] = np.asarray([env_seed], np.uint32)
        episode = dict(
            observations=np.asarray(observations, np.float32)[None],
            actions=np.concatenate(
                [np.zeros((1, 1, cfg.action_dim), np.float32), trace["actions"]], axis=1
            ),
            rewards=np.concatenate(
                [np.zeros((1, 1), np.float64), trace["rewards"]], axis=1
            ),
            continuations=np.concatenate(
                [np.ones((1, 1), np.float64), trace["continuations"]], axis=1
            ),
            is_first=np.asarray([[True] + [False] * steps]),
            is_last=np.concatenate([np.zeros((1, 1), bool), trace["is_last"]], axis=1),
        )
        if not all(np.isfinite(v).all() for v in (*episode.values(), *trace.values())):
            raise ValueError("nonfinite collected episode")
        if _tree_digest((world, policy, anchors)) != before:
            raise RuntimeError("collector changed frozen checkpoint parameters")
        metrics = dict(
            native_steps=steps,
            timed_steps=steps,
            episode_return=float(trace["rewards"].sum(dtype=np.float64)),
            native_boundary=bool(trace["is_last"][0, -1]),
            truncated=not bool(trace["is_last"][0, -1]),
            training=bool(training),
            exploration_std=float(exploration_std),
            total_timed_seconds=elapsed,
            mean_milliseconds_per_step=elapsed * 1000 / steps,
            discarded_compile_warmup_steps=discarded,
            parameter_digest_before=before,
            parameter_digest_after=before,
        )
        json.dumps(metrics, allow_nan=False)
        return episode, trace, metrics


def make_collector(cfg, rc, controller):
    return OnlineCollector(cfg, rc, controller)
