"""Frozen-checkpoint, matched-real-start endpoint decision diagnostics.

Publication and checkpoint authentication belong to the study runner. This
module only reads checkpoints and returns reproducible arrays and JSON data.
The continuation target is a finite, native-episode Monte Carlo return under
the frozen ReBRAC actor. It is deliberately not labelled an infinite-horizon Q.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from . import matched_objective_benchmark as benchmark

SCHEMA = "controller-repair-matched-diagnostics-v1"
SPLITS = ("calibration", "validation", "preflight")
FAMILIES = ("recursive", "direct")
REFERENCE_MODES = ("latent", "endpoint")
PLAN_KINDS = ("reference", "proposal", "executed", "directional")


def _model_digest(model: Mapping[str, Any]) -> str:
    # Dataclasses are static JAX leaves; hashing np.asarray(config) would hash
    # an object address and break cross-process replay of otherwise equal data.
    return benchmark._tree_digest(
        dict(
            model,
            config=asdict(model["config"]),
            rebrac_config=asdict(model["rebrac_config"]),
        )
    )


def _finite_array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ValueError(f"nonfinite or nonnumeric {name}")
    return result


def resolve_settings(protocol: Mapping[str, Any], *, preflight: bool) -> dict[str, Any]:
    """Read every realized diagnostic parameter; reject split/config drift."""
    source = protocol["diagnostics"]
    settings = dict(source)
    if settings["horizon"] != protocol["controller"]["horizon"]:
        raise ValueError("diagnostic horizon differs from controller")
    if settings["families"] != list(FAMILIES):
        raise ValueError("diagnostic family set differs")
    if (
        not settings["reference_modes"]
        or any(mode not in REFERENCE_MODES for mode in settings["reference_modes"])
        or len(set(settings["reference_modes"])) != len(settings["reference_modes"])
    ):
        raise ValueError("invalid reference modes")
    calibration = list(settings["calibration_environment_seeds"])
    validation = list(settings["validation_environment_seeds"])
    final = list(protocol["evaluation_environment_seeds"])
    for name, seeds in (("calibration", calibration), ("validation", validation)):
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError(f"invalid {name} environment seeds")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate {name} environment seeds")
    if set(calibration) & set(validation) or (set(calibration) | set(validation)) & set(
        final
    ):
        raise ValueError("diagnostic/evaluation seed overlap")
    settings["splits"] = {"calibration": calibration, "validation": validation}
    if preflight:
        pf = protocol["preflight"]
        settings["snapshot_steps"] = list(pf["diagnostic_snapshot_steps"])
        settings["maximum_episode_steps"] = pf["diagnostic_continuation_steps"]
        seeds = list(pf["diagnostic_environment_seeds"])
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or set(seeds) & (set(calibration) | set(validation) | set(final))
        ):
            raise ValueError("invalid/overlapping preflight seeds")
        settings["splits"] = {"preflight": seeds}
    horizon, maximum = settings["horizon"], settings["maximum_episode_steps"]
    if (
        type(horizon) is not int
        or horizon < 1
        or type(maximum) is not int
        or maximum < horizon
    ):
        raise ValueError("invalid diagnostic horizon/episode length")
    steps = settings["snapshot_steps"]
    if (
        not steps
        or list(steps) != sorted(set(steps))
        or any(
            type(step) is not int or not 0 <= step <= maximum - horizon
            for step in steps
        )
    ):
        raise ValueError("invalid diagnostic snapshot steps")
    scales = _finite_array(settings["direction_scales"], "direction scales")
    if (
        scales.ndim != 1
        or not len(scales)
        or np.any(scales <= 0)
        or len(set(scales)) != len(scales)
    ):
        raise ValueError("invalid direction scales")
    if np.max(scales) > protocol["controller"]["action_sequence_residual_limit"]:
        raise ValueError("direction scales exceed the registered residual box")
    return settings


def directional_ladder(
    reference: np.ndarray, scales: list[float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded finite action changes; retain actual clipping, never infer gradients.

    A single seeded unit-L2 direction is shared across both signs and all step
    sizes. At a boundary, clipping can make the two actual changes asymmetric;
    callers retain the full actual deltas and their norms.
    """
    reference = _finite_array(reference, "reference").astype(np.float32)
    if reference.ndim != 2 or np.any(np.abs(reference) > 1):
        raise ValueError("reference must be a bounded [horizon,action] array")
    direction = np.random.default_rng(seed).normal(size=reference.shape)
    direction /= np.linalg.norm(direction)
    direction = direction.astype(np.float32)
    signed = np.asarray(
        [sign * scale for scale in scales for sign in (-1, 1)], np.float32
    )
    plans = np.clip(reference[None] + signed[:, None, None] * direction, -1, 1)
    return plans.astype(np.float32), signed, direction


def rollout_fixed_plan(
    environment: Any,
    snapshot: Any,
    observation: np.ndarray,
    action_sequence: np.ndarray,
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    discount: float,
    maximum_steps: int,
) -> dict[str, Any]:
    """Restore exactly, execute H fixed actions, then the frozen observation actor.

    Rewards at index t belong to (observation[t], action[t]). The terminal
    continuation starts with reward[H], weighted by gamma**H. Restoring in a
    finally block also preserves baseline occupancy after a failed rollout.
    """
    actions = _finite_array(action_sequence, "fixed actions").astype(np.float32)
    current = (
        _finite_array(observation, "initial observation").astype(np.float32).copy()
    )
    if actions.ndim != 2 or not len(actions) or np.any(np.abs(actions) > 1):
        raise ValueError("fixed plan must be bounded and nonempty")
    if maximum_steps < len(actions) or not 0 < discount <= 1:
        raise ValueError("invalid rollout length/discount")
    horizon = len(actions)
    rewards = np.zeros(maximum_steps, np.float64)
    continuations = np.zeros(maximum_steps, np.float64)
    executed = np.zeros((maximum_steps, actions.shape[-1]), np.float32)
    stage_observations = np.zeros((horizon + 1, *current.shape), np.float32)
    stage_observations[0] = current
    terminal = False
    environment.restore(snapshot)
    try:
        for step in range(maximum_steps):
            action = (
                actions[step]
                if step < horizon
                else np.asarray(policy(current), np.float32)
            )
            if (
                action.shape != actions.shape[1:]
                or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1)
            ):
                raise ValueError("frozen actor produced invalid action")
            transition = environment.step(action)
            rewards[step] = transition.reward
            continuations[step] = transition.continuation
            executed[step] = action
            current = np.asarray(transition.observation, np.float32)
            if step < horizon:
                stage_observations[step + 1] = current
            if transition.is_last:
                terminal = True
                break
        length = step + 1
    finally:
        environment.restore(snapshot)
    weights = discount ** np.arange(maximum_steps, dtype=np.float64)
    weights[1:] *= np.cumprod(continuations[:-1])
    stage = float(np.dot(weights[:horizon], rewards[:horizon]))
    continuation = float(np.dot(weights[horizon:], rewards[horizon:]))
    result = dict(
        stage_reward=stage,
        terminal_value=continuation,
        objective=stage + continuation,
        rewards=rewards,
        continuations=continuations,
        actions=executed,
        stage_observations=stage_observations,
        length=length,
        stage_length=min(length, horizon),
        native_episode_end=terminal,
    )
    for name, value in result.items():
        _finite_array(value, name)
    return result


def training_support(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Describe all valid training transitions, excluding held-out episodes/reset slots."""
    ids = np.asarray(arrays["train_episode_ids"], np.int64)
    if (
        not len(ids)
        or len(set(ids)) != len(ids)
        or set(ids) & set(arrays["test_episode_ids"])
    ):
        raise ValueError("invalid training episode split")
    rewards = np.asarray(arrays["rewards"])[ids, 1:]
    valid = np.asarray(arrays["continuations"])[ids, :-1] > 0
    if "is_first" in arrays:
        valid &= ~np.asarray(arrays["is_first"])[ids, 1:].astype(bool)
    selected = _finite_array(rewards[valid], "training rewards")
    if not len(selected):
        raise ValueError("no training transitions")
    return dict(
        transitions=int(len(selected)),
        positive_rewards=int(np.sum(selected > 0)),
        positive_reward_fraction=float(np.mean(selected > 0)),
        mean_reward=float(np.mean(selected)),
        episode_return_mean=float(np.mean(np.sum(np.where(valid, rewards, 0), axis=1))),
        includes_last_valid_transition=True,
    )


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(np.mean(values[mask])) if np.any(mask) else None


def _ranking_agreement(
    predicted: np.ndarray, actual: np.ndarray
) -> tuple[float | None, int]:
    left, right = np.triu_indices(len(actual), 1)
    true_delta = actual[left] - actual[right]
    predicted_delta = predicted[left] - predicted[right]
    informative = np.abs(true_delta) > 1e-8
    return _masked_mean(
        (np.sign(true_delta) == np.sign(predicted_delta)).astype(float), informative
    ), int(np.sum(informative))


def validate_diagnostic_trace(
    trace: Mapping[str, np.ndarray],
    settings: Mapping[str, Any],
    *,
    discount: float = 0.99,
) -> None:
    """Check matched occupancy, executed fallback and return arithmetic from arrays."""
    for name, array in trace.items():
        _finite_array(array, name)
    horizon, maximum = settings["horizon"], settings["maximum_episode_steps"]
    plan_count = 3 + 2 * len(settings["direction_scales"])
    cursor, snapshot_id, baseline_id = 0, 0, 0
    for split, seeds in settings["splits"].items():
        for seed in seeds:
            if (
                trace["baseline_split_id"][baseline_id] != SPLITS.index(split)
                or trace["baseline_environment_seed"][baseline_id] != seed
            ):
                raise ValueError("baseline split/seed differs")
            for step in settings["snapshot_steps"]:
                for family in settings["families"]:
                    for mode in settings["reference_modes"]:
                        indices = np.arange(cursor, cursor + plan_count)
                        if indices[-1] >= len(trace["plan_kind"]):
                            raise ValueError("missing diagnostic plan rows")
                        for name, expected in (
                            ("split_id", SPLITS.index(split)),
                            ("environment_seed", seed),
                            ("episode_step", step),
                            ("snapshot_id", snapshot_id),
                            ("family_id", FAMILIES.index(family)),
                            ("reference_mode_id", REFERENCE_MODES.index(mode)),
                        ):
                            if not np.all(trace[name][indices] == expected):
                                raise ValueError(
                                    f"matched trace identity differs: {name}"
                                )
                        if not np.array_equal(
                            trace["plan_kind"][indices],
                            [0, 1, 2] + [3] * (plan_count - 3),
                        ):
                            raise ValueError("diagnostic plan kinds differ")
                        start = trace["baseline_observations"][baseline_id, step]
                        if not np.all(trace["initial_observation"][indices] == start):
                            raise ValueError(
                                "diagnostic starts differ from frozen actor occupancy"
                            )
                        ref, proposed, executed = indices[:3]
                        expected_sequence = trace["action_sequence"][
                            ref if trace["used_fallback"][executed] else proposed
                        ]
                        if not np.array_equal(
                            trace["action_sequence"][executed], expected_sequence
                        ):
                            raise ValueError(
                                "executed sequence differs from fallback decision"
                            )
                        for source in ("model", "real"):
                            for component in ("stage", "terminal", "objective"):
                                values = trace[f"{source}_{component}"][indices]
                                if not np.array_equal(
                                    trace[f"{source}_{component}_gain"][indices],
                                    values - values[0],
                                ):
                                    raise ValueError(
                                        "candidate/reference gain arithmetic differs"
                                    )
                        if not np.array_equal(
                            trace["actual_delta"][indices],
                            trace["action_sequence"][indices]
                            - trace["action_sequence"][ref],
                        ):
                            raise ValueError("actual finite changes differ")
                        cursor += plan_count
                snapshot_id += 1
            baseline_id += 1
    if cursor != len(trace["plan_kind"]) or baseline_id != len(
        trace["baseline_length"]
    ):
        raise ValueError("unexpected extra diagnostic/baseline rows")
    if np.any(np.abs(trace["action_sequence"]) > 1) or np.any(
        np.abs(trace["real_actions"]) > 1
    ):
        raise ValueError("out-of-bounds diagnostic action")
    if not np.array_equal(trace["first_action"], trace["action_sequence"][:, 0]):
        raise ValueError("recorded first actions differ")
    for index in range(cursor):
        length = int(trace["real_length"][index])
        if not 1 <= length <= maximum - trace["episode_step"][index]:
            raise ValueError("invalid diagnostic trajectory length")
        if not np.array_equal(
            trace["stage_mask"][index], np.arange(horizon) < min(length, horizon)
        ):
            raise ValueError("stage mask differs")
        if not np.array_equal(
            trace["real_actions"][index, : min(length, horizon)],
            trace["action_sequence"][index, : min(length, horizon)],
        ):
            raise ValueError("simulator did not execute the fixed action plan")
        for name in ("real_rewards", "real_continuations", "real_actions"):
            if np.any(trace[name][index, length:]):
                raise ValueError("nonzero diagnostic trajectory padding")
        weights = discount ** np.arange(maximum, dtype=np.float64)
        weights[1:] *= np.cumprod(trace["real_continuations"][index, :-1])
        for name, section in (
            ("stage", slice(None, horizon)),
            ("terminal", slice(horizon, None)),
        ):
            computed = np.dot(weights[section], trace["real_rewards"][index, section])
            if not np.isclose(
                trace[f"real_{name}"][index], computed, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("real stage/continuation arithmetic differs")
    for source in ("model", "real"):
        if not np.allclose(
            trace[f"{source}_objective"],
            trace[f"{source}_stage"] + trace[f"{source}_terminal"],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("stage/terminal objective decomposition differs")


def summarize_trace(trace: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Recompute decision summaries exclusively from numeric trace evidence."""
    summary: dict[str, Any] = {}
    for split_id in sorted(set(trace["split_id"].tolist())):
        split = SPLITS[split_id]
        summary[split] = {}
        for family_id in sorted(set(trace["family_id"].tolist())):
            family = FAMILIES[family_id]
            summary[split][family] = {}
            for mode_id in sorted(set(trace["reference_mode_id"].tolist())):
                group = (
                    (trace["split_id"] == split_id)
                    & (trace["family_id"] == family_id)
                    & (trace["reference_mode_id"] == mode_id)
                )
                if not np.any(group):
                    continue
                masks = {
                    name: group & (trace["plan_kind"] == index)
                    for index, name in enumerate(PLAN_KINDS)
                }
                ref, proposal, executed = (masks[name] for name in PLAN_KINDS[:3])
                result: dict[str, Any] = dict(
                    records=int(np.sum(group)),
                    snapshots=int(np.sum(ref)),
                    reference_feasible_fraction=float(
                        np.mean(trace["model_feasible"][ref])
                    ),
                    proposal_feasible_fraction=float(
                        np.mean(trace["model_feasible"][proposal])
                    ),
                    fallback_fraction=float(np.mean(trace["used_fallback"][executed])),
                    executed_first_action_change_fraction=float(
                        np.mean(trace["first_action_changed"][executed])
                    ),
                )
                # Across reference modes the baselines differ. The selector
                # must compare these matched levels, not differences of gains.
                for kind in PLAN_KINDS:
                    for source in ("model", "real"):
                        for component in ("stage", "terminal", "objective"):
                            result[f"{source}_{kind}_{component}_mean"] = float(
                                np.mean(trace[f"{source}_{component}"][masks[kind]])
                            )
                for kind in ("proposal", "executed"):
                    mask = masks[kind]
                    for component in ("stage", "terminal", "objective"):
                        result[f"{kind}_{component}_gain_mean"] = float(
                            np.mean(trace[f"model_{component}_gain"][mask])
                        )
                        result[f"real_{kind}_{component}_gain_mean"] = float(
                            np.mean(trace[f"real_{component}_gain"][mask])
                        )
                        result[f"{kind}_{component}_gain_error_mae"] = float(
                            np.mean(
                                np.abs(
                                    trace[f"model_{component}_gain"][mask]
                                    - trace[f"real_{component}_gain"][mask]
                                )
                            )
                        )
                    error = (
                        trace["model_objective_gain"][mask]
                        - trace["real_objective_gain"][mask]
                    )
                    result[f"{kind}_gain_error_mae"] = float(np.mean(np.abs(error)))
                    actual, predicted = (
                        trace["real_objective_gain"][mask],
                        trace["model_objective_gain"][mask],
                    )
                    informative = np.abs(actual) > 1e-8
                    result[f"{kind}_sign_agreement"] = _masked_mean(
                        (np.sign(actual) == np.sign(predicted)).astype(float),
                        informative,
                    )
                    result[f"{kind}_informative_sign_count"] = int(np.sum(informative))
                directional = masks["directional"] & (trace["actual_delta_l2"] > 0)
                informative = directional & (
                    np.abs(trace["real_objective_gain"]) > 1e-8
                )
                signs = np.sign(trace["model_objective_gain"]) == np.sign(
                    trace["real_objective_gain"]
                )
                result["directional_sign_agreement"] = _masked_mean(
                    signs.astype(float), informative
                )
                result["informative_direction_count"] = int(np.sum(informative))
                ranking_numerator, ranking_count = 0.0, 0
                for snapshot_id in np.unique(trace["snapshot_id"][group]):
                    candidate = (
                        group
                        & (trace["snapshot_id"] == snapshot_id)
                        & (trace["plan_kind"] != PLAN_KINDS.index("executed"))
                    )
                    agreement, count = _ranking_agreement(
                        trace["model_objective"][candidate],
                        trace["real_objective"][candidate],
                    )
                    ranking_numerator += (agreement or 0) * count
                    ranking_count += count
                result["candidate_pairwise_ranking_agreement"] = (
                    ranking_numerator / ranking_count if ranking_count else None
                )
                result["informative_ranking_pairs"] = ranking_count
                result["terminal_transition_error_mae"] = float(
                    np.mean(np.abs(trace["terminal_transition_error"][group]))
                )
                result["terminal_critic_mc_error_mae"] = float(
                    np.mean(np.abs(trace["terminal_critic_mc_error"][group]))
                )
                result["reward_errors"] = {}
                for kind, mask in masks.items():
                    valid = mask[:, None] & trace["stage_mask"].astype(bool)
                    positive = valid & (
                        trace["real_rewards"][:, : trace["stage_mask"].shape[1]] > 0
                    )
                    zero = valid & ~positive
                    error_result = dict(
                        transitions=int(np.sum(valid)),
                        positive_events=int(np.sum(positive)),
                    )
                    for source in ("real_state", "generated_state", "generated_mean"):
                        errors = trace[f"{source}_reward_squared_error"]
                        error_result[f"{source}_mse"] = _masked_mean(errors, valid)
                        error_result[f"{source}_positive_event_mse"] = _masked_mean(
                            errors, positive
                        )
                        error_result[f"{source}_zero_event_mse"] = _masked_mean(
                            errors, zero
                        )
                    result["reward_errors"][kind] = error_result
                summary[split][family][REFERENCE_MODES[mode_id]] = result
    return summary


def evaluate_diagnostics(
    protocol: Mapping[str, Any],
    cell: Mapping[str, Any],
    training_dir: str | Path,
    *,
    preflight: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float]]:
    """Compare frozen K controllers at identical frozen-actor occupancy snapshots."""
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        initial_state,
        observe_step,
        rebrac_actor,
        rebrac_critics,
    )
    from imf_dreamer_jax.world_model import predict_state_action_reward_from_observation
    from .dmc import DMCAdapter
    from .mechanism_replication import load_checkpoint
    from .repaired_controllers import make_endpoint_planner, make_endpoint_scorer

    started = time.perf_counter()
    settings = resolve_settings(protocol, preflight=preflight)
    directory = Path(training_dir)
    checkpoint_sha = benchmark.file_sha256(directory / "checkpoint.pkl")
    replay_sha = benchmark.file_sha256(directory / "replay.npz")
    model = load_checkpoint(directory)
    frozen_digest = _model_digest(model)
    cfg, rc = model["config"], model["rebrac_config"]
    actor_seed = int(settings["actor_seed"])
    if "actor_seed" in cell and cell["actor_seed"] != actor_seed:
        raise ValueError("diagnostic cell actor differs from protocol")
    if float(rc.discount) != 0.99:
        raise ValueError("diagnostic frozen critic discount differs from .99")
    actor = model["policies"][actor_seed].actor
    critics = model["policies"][actor_seed].critics
    horizon, maximum = settings["horizon"], settings["maximum_episode_steps"]
    controller = protocol["controller"]
    actor_jit = jax.jit(lambda obs: rebrac_actor(actor, obs))
    reward_jit = jax.jit(
        lambda obs, actions: predict_state_action_reward_from_observation(
            model["reward_world"]["reward_transition"], obs, actions, cfg
        )
    )
    critic_jit = jax.jit(
        lambda obs: jnp.min(
            rebrac_critics(critics, obs, rebrac_actor(actor, obs)), axis=0
        )
    )
    observe_jit = jax.jit(
        lambda obs, previous, belief, key: observe_step(
            model["reward_world"], obs, previous, belief, key, cfg
        )[0]
    )

    def policy(observation: np.ndarray) -> np.ndarray:
        return np.asarray(
            actor_jit(jnp.asarray(observation[None], jnp.float32))[0], np.float32
        )

    planners = {
        (family, mode): make_endpoint_planner(
            model,
            actor_seed,
            direct_any_step=family == "direct",
            controller=controller,
            reference_mode=mode,
        )
        for family in settings["families"]
        for mode in settings["reference_modes"]
    }
    scorers = {
        family: make_endpoint_scorer(
            model,
            actor_seed,
            direct_any_step=family == "direct",
            controller=controller,
        )
        for family in settings["families"]
    }
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    planner_seconds, rollout_seconds, planner_calls, rollout_calls = 0.0, 0.0, 0, 0
    snapshot_id = 0
    for split, seeds in settings["splits"].items():
        for environment_seed in seeds:
            env = DMCAdapter(cell["task"], seed=environment_seed, action_repeat=1)
            baseline_rewards = np.zeros(maximum, np.float64)
            baseline_actions = np.zeros((maximum, cfg.action_dim), np.float32)
            baseline_observations = np.zeros(
                (maximum + 1, cfg.observation_dim), np.float32
            )
            baseline_continuations = np.zeros(maximum, np.float64)
            baseline_native_end = False
            captured = 0
            try:
                observation = env.reset()
                baseline_observations[0] = observation
                belief = initial_state(cfg, 1)
                previous_action = jnp.zeros((1, cfg.action_dim), jnp.float32)
                for episode_step in range(maximum):
                    current = jnp.asarray(observation[None], jnp.float32)
                    posterior_key = benchmark.derive_jax_key(
                        "repair-diagnostic-posterior",
                        cell["task"],
                        cell["world_model_seed"],
                        actor_seed,
                        environment_seed,
                        episode_step,
                    )
                    belief = observe_jit(
                        current, previous_action, belief, posterior_key
                    )
                    if episode_step in settings["snapshot_steps"]:
                        snapshot = env.snapshot()
                        captured += 1
                        seed_parts = (
                            cell["task"],
                            cell["world_model_seed"],
                            actor_seed,
                            environment_seed,
                            episode_step,
                        )
                        # Exactly the same noise, CEM perturbations and finite-change
                        # direction are used across both endpoint families/references.
                        noises = jax.random.normal(
                            benchmark.derive_jax_key(
                                "repair-diagnostic-noise", *seed_parts
                            ),
                            (
                                controller["action_sequence_particles"],
                                horizon,
                                cfg.observation_dim,
                            ),
                        )
                        proposals = jax.random.normal(
                            benchmark.derive_jax_key(
                                "repair-diagnostic-proposals", *seed_parts
                            ),
                            (2, 4, horizon, cfg.action_dim),
                        )
                        direction_seed = benchmark.derive_seed(
                            "repair-diagnostic-direction", *seed_parts
                        )
                        rollout_cache: dict[bytes, dict[str, Any]] = {}
                        for family in settings["families"]:
                            for mode in settings["reference_modes"]:
                                before = time.perf_counter()
                                planned = jax.device_get(
                                    planners[family, mode](
                                        belief, current, noises, proposals
                                    )
                                )
                                planner_seconds += time.perf_counter() - before
                                planner_calls += 1
                                reference = np.asarray(
                                    planned["reference_sequence"], np.float32
                                )
                                proposal = np.asarray(
                                    planned["action_sequence"], np.float32
                                )
                                executed = np.asarray(
                                    planned["executed_sequence"], np.float32
                                )
                                ladder, signed_scales, direction = directional_ladder(
                                    reference,
                                    settings["direction_scales"],
                                    direction_seed,
                                )
                                plans = [
                                    (kind, sequence, 0.0)
                                    for kind, sequence in (
                                        ("reference", reference),
                                        ("proposal", proposal),
                                        ("executed", executed),
                                    )
                                ] + [
                                    ("directional", sequence, float(scale))
                                    for sequence, scale in zip(
                                        ladder, signed_scales, strict=True
                                    )
                                ]
                                group_rows = []
                                for kind, sequence, scale in plans:
                                    # Preserve the exact metrics used by the compiled
                                    # controller's decision. Re-evaluation under a
                                    # different fusion boundary can differ in low bits.
                                    model_score = (
                                        planned[kind]
                                        if kind != "directional"
                                        else jax.device_get(
                                            scorers[family](
                                                current, jnp.asarray(sequence), noises
                                            )
                                        )
                                    )
                                    for name, value in model_score.items():
                                        _finite_array(value, f"model score {name}")
                                    key = sequence.tobytes()
                                    if key not in rollout_cache:
                                        before = time.perf_counter()
                                        rollout_cache[key] = rollout_fixed_plan(
                                            env,
                                            snapshot,
                                            observation,
                                            sequence,
                                            policy,
                                            discount=float(rc.discount),
                                            maximum_steps=maximum - episode_step,
                                        )
                                        rollout_seconds += time.perf_counter() - before
                                        rollout_calls += 1
                                    real = rollout_cache[key]
                                    stage_length = real["stage_length"]
                                    mask = np.arange(horizon) < stage_length
                                    actual_rewards = real["rewards"][:horizon]
                                    real_head = np.asarray(
                                        reward_jit(
                                            jnp.asarray(
                                                real["stage_observations"][:horizon]
                                            ),
                                            jnp.asarray(sequence),
                                        ),
                                        np.float64,
                                    ).reshape(horizon)
                                    reward_mean = np.asarray(
                                        model_score["reward_mean"], np.float64
                                    ).reshape(horizon)
                                    reward_second = np.asarray(
                                        model_score["reward_second_moment"], np.float64
                                    ).reshape(horizon)
                                    actual_delta = sequence - reference
                                    real_terminal_critic = 0.0
                                    if stage_length == horizon:
                                        real_terminal_critic = (
                                            float(rc.discount) ** horizon
                                        ) * float(
                                            np.asarray(
                                                critic_jit(
                                                    jnp.asarray(
                                                        real["stage_observations"][
                                                            horizon
                                                        ][None]
                                                    )
                                                )
                                            ).reshape(-1)[0]
                                        )
                                    row = dict(
                                        split_id=SPLITS.index(split),
                                        family_id=FAMILIES.index(family),
                                        reference_mode_id=REFERENCE_MODES.index(mode),
                                        snapshot_id=snapshot_id,
                                        environment_seed=environment_seed,
                                        episode_step=episode_step,
                                        plan_kind=PLAN_KINDS.index(kind),
                                        direction_scale=scale,
                                        common_direction=direction,
                                        actual_delta=actual_delta,
                                        actual_delta_l2=float(
                                            np.linalg.norm(actual_delta)
                                        ),
                                        actual_delta_linf=float(
                                            np.max(np.abs(actual_delta))
                                        ),
                                        initial_observation=observation.copy(),
                                        action_sequence=sequence,
                                        first_action=sequence[0],
                                        first_action_changed=bool(
                                            np.any(sequence[0] != reference[0])
                                        ),
                                        used_fallback=bool(planned["used_fallback"]),
                                        stage_mask=mask,
                                        real_stage_observations=real[
                                            "stage_observations"
                                        ],
                                        real_state_reward=real_head,
                                        model_reward_mean=reward_mean,
                                        model_reward_second_moment=reward_second,
                                        real_state_reward_squared_error=(
                                            real_head - actual_rewards
                                        )
                                        ** 2,
                                        generated_mean_reward_squared_error=(
                                            reward_mean - actual_rewards
                                        )
                                        ** 2,
                                        generated_state_reward_squared_error=np.maximum(
                                            0,
                                            reward_second
                                            - 2 * reward_mean * actual_rewards
                                            + actual_rewards**2,
                                        ),
                                        real_stage=real["stage_reward"],
                                        real_terminal=real["terminal_value"],
                                        real_objective=real["objective"],
                                        model_stage=float(model_score["stage_reward"]),
                                        model_terminal=float(
                                            model_score["terminal_value"]
                                        ),
                                        model_objective=float(model_score["objective"]),
                                        model_distance=float(
                                            model_score["behavior_distance"]
                                        ),
                                        model_violation=float(model_score["violation"]),
                                        model_feasible=bool(model_score["feasible"]),
                                        real_terminal_critic=real_terminal_critic,
                                        terminal_transition_error=float(
                                            model_score["terminal_value"]
                                        )
                                        - real_terminal_critic,
                                        terminal_critic_mc_error=real_terminal_critic
                                        - real["terminal_value"],
                                        real_length=real["length"],
                                        native_episode_end=real["native_episode_end"],
                                    )
                                    for name in ("rewards", "continuations", "actions"):
                                        array = real[name]
                                        row[f"real_{name}"] = np.pad(
                                            array,
                                            [(0, maximum - len(array))]
                                            + [(0, 0)] * (array.ndim - 1),
                                        )
                                    group_rows.append(row)
                                base = group_rows[0]
                                for row in group_rows:
                                    for source in ("model", "real"):
                                        for component in (
                                            "stage",
                                            "terminal",
                                            "objective",
                                        ):
                                            row[f"{source}_{component}_gain"] = (
                                                row[f"{source}_{component}"]
                                                - base[f"{source}_{component}"]
                                            )
                                rows.extend(group_rows)
                        snapshot_id += 1
                    action = policy(observation)
                    transition = env.step(action)
                    baseline_actions[episode_step] = action
                    baseline_rewards[episode_step] = transition.reward
                    baseline_continuations[episode_step] = transition.continuation
                    baseline_observations[episode_step + 1] = transition.observation
                    observation = transition.observation
                    previous_action = jnp.asarray(action[None], jnp.float32)
                    if transition.is_last:
                        baseline_native_end = True
                        break
                if captured != len(settings["snapshot_steps"]):
                    raise ValueError(
                        "frozen actor ended before all registered snapshots"
                    )
                baseline_rows.append(
                    dict(
                        split_id=SPLITS.index(split),
                        environment_seed=environment_seed,
                        rewards=baseline_rewards,
                        actions=baseline_actions,
                        observations=baseline_observations,
                        continuations=baseline_continuations,
                        length=episode_step + 1,
                        native_episode_end=baseline_native_end,
                    )
                )
            finally:
                env.close()
    if not rows:
        raise ValueError("diagnostics collected no matched starts")
    trace = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    trace.update(
        {
            f"baseline_{name}": np.stack([row[name] for row in baseline_rows])
            for name in baseline_rows[0]
        }
    )
    for name, value in trace.items():
        _finite_array(value, name)
    validate_diagnostic_trace(trace, settings, discount=float(rc.discount))
    if not preflight and (
        not np.all(trace["native_episode_end"])
        or not np.all(trace["baseline_native_episode_end"])
    ):
        raise ValueError(
            "full diagnostic continuation was truncated before native episode end"
        )
    if _model_digest(model) != frozen_digest:
        raise RuntimeError("diagnostics mutated the frozen model")
    if (
        benchmark.file_sha256(directory / "checkpoint.pkl") != checkpoint_sha
        or benchmark.file_sha256(directory / "replay.npz") != replay_sha
    ):
        raise RuntimeError("diagnostics changed checkpoint/replay bytes")
    baseline_summary = [
        dict(
            split=SPLITS[row["split_id"]],
            environment_seed=row["environment_seed"],
            episode_return=float(np.sum(row["rewards"])),
            length=row["length"],
            native_episode_end=row["native_episode_end"],
        )
        for row in baseline_rows
    ]
    core = dict(cell)
    core.update(
        schema=SCHEMA,
        actor_seed=actor_seed,
        preflight=bool(preflight),
        realized_settings=settings,
        realized_controller=dict(controller),
        discount=float(rc.discount),
        checkpoint_sha256=checkpoint_sha,
        replay_file_sha256=replay_sha,
        frozen_model_sha256=frozen_digest,
        trace_sha256=benchmark.array_sha256(trace),
        continuation_semantics=(
            "discounted_frozen_ReBRAC_return_to_native_episode_end"
            if not preflight
            else "preflight_truncated_discounted_frozen_ReBRAC_return"
        ),
        model_terminal_semantics="gamma_to_horizon_times_mean_minimum_frozen_regularized_ReBRAC_critics",
        diagnostic_scope="matched_fixed_open_loop_plans_then_frozen_actor_continuation; finite_changes_are_not_derivatives",
        trace_codes=dict(
            split=list(SPLITS),
            family=list(FAMILIES),
            reference_mode=list(REFERENCE_MODES),
            plan_kind=list(PLAN_KINDS),
        ),
        training_support=training_support(benchmark.load_npz(directory / "replay.npz")),
        baseline=baseline_summary,
        summary=summarize_trace(trace),
    )
    timing = dict(
        wall_seconds=time.perf_counter() - started,
        planner_seconds_including_compilation=planner_seconds,
        simulator_rollout_seconds=rollout_seconds,
        planner_calls=planner_calls,
        unique_simulator_rollout_calls=rollout_calls,
    )
    return core, trace, timing
