"""Immutable policy-independent probe banks for paired world-model evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .dmc import DMCAdapter
from .matched_objective_benchmark import (
    array_sha256,
    derive_seed,
    posterior_mean_filter,
)
from .policy_alignment_diagnostics import action_ranking_metrics


SCHEMA = "trajectory-imf-shared-probe-bank-v2"


def _json_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_probe_plan(
    arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    world_model_seed: int,
    probes: int = 16,
    horizons: Sequence[int] = (1, 3, 5, 15),
    action_delta: float = 0.5,
    intervention_steps: int = 5,
    stochastic_dim: int = 16,
    draws: int = 8,
    episode_ids_key: str = "test_episode_ids",
    minimum_anchor: int = 1,
) -> dict[str, np.ndarray]:
    """Choose states, candidates, suffixes, and model noise before evaluation."""

    ordered_horizons = tuple(int(value) for value in horizons)
    if not ordered_horizons or tuple(sorted(set(ordered_horizons))) != ordered_horizons:
        raise ValueError("horizons must be unique, positive, and increasing")
    if ordered_horizons[0] <= 0 or probes <= 0 or draws <= 0 or stochastic_dim <= 0:
        raise ValueError("probe counts, horizons, draws, and stochastic_dim must be positive")
    if not 0.0 < action_delta <= 1.0:
        raise ValueError("action_delta must lie in (0, 1]")
    if not 0 < intervention_steps <= ordered_horizons[-1]:
        raise ValueError("intervention_steps must lie in [1, maximum_horizon]")
    if minimum_anchor < 1:
        raise ValueError("minimum_anchor must be positive")
    observations = np.asarray(arrays["observations"])
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    if episode_ids_key not in ("train_episode_ids", "test_episode_ids"):
        raise ValueError("episode_ids_key must select the train or test split")
    test_ids = np.asarray(arrays[episode_ids_key], dtype=np.int32)
    if observations.shape[:2] != actions.shape[:2] or actions.ndim != 3:
        raise ValueError("dataset observations and actions have incompatible shapes")
    maximum_horizon = ordered_horizons[-1]
    candidates = [
        (int(episode), int(anchor))
        for episode in test_ids
        for anchor in range(minimum_anchor, actions.shape[1] - maximum_horizon + 1)
        if np.all(np.asarray(arrays["continuations"])[episode, anchor : anchor + maximum_horizon] > 0.0)
    ]
    if len(candidates) < probes:
        raise ValueError("dataset does not contain enough full-horizon test probes")
    random = np.random.default_rng(
        derive_seed("shared-probe-plan", episode_ids_key, task, world_model_seed)
    )
    chosen = sorted(
        candidates[index]
        for index in random.choice(len(candidates), size=probes, replace=False)
    )
    action_dim = actions.shape[-1]
    candidate_count = 1 + 2 * action_dim
    action_sequences = np.empty(
        (probes, candidate_count, maximum_horizon, action_dim), np.float32
    )
    for probe_index, (episode, anchor) in enumerate(chosen):
        suffix = actions[episode, anchor : anchor + maximum_horizon]
        action_sequences[probe_index] = suffix[None]
        for dimension in range(action_dim):
            for offset in range(intervention_steps):
                lower = suffix[offset].copy()
                upper = suffix[offset].copy()
                lower[dimension] = np.clip(
                    lower[dimension] - action_delta, -1.0, 1.0
                )
                upper[dimension] = np.clip(
                    upper[dimension] + action_delta, -1.0, 1.0
                )
                action_sequences[probe_index, 1 + 2 * dimension, offset] = lower
                action_sequences[probe_index, 2 + 2 * dimension, offset] = upper
    noise = np.random.default_rng(
        derive_seed(
            "shared-probe-model-noise", episode_ids_key, task, world_model_seed
        )
    ).standard_normal(
        (draws, probes, maximum_horizon, stochastic_dim)
    ).astype(np.float32)
    return {
        "episode_ids": np.asarray([value[0] for value in chosen], np.int32),
        "anchors": np.asarray([value[1] for value in chosen], np.int32),
        "horizons": np.asarray(ordered_horizons, np.int32),
        "action_sequences": action_sequences,
        "model_noise": noise,
    }


def generate_shared_probe_bank(
    arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    world_model_seed: int,
    action_repeat: int,
    discount: float = 0.99,
    probes: int = 32,
    horizons: Sequence[int] = (1, 3, 5, 15),
    action_delta: float = 0.5,
    intervention_steps: int = 5,
    selection_pool_multiplier: int = 8,
    minimum_informative_fraction: float = 0.5,
    stochastic_dim: int = 16,
    draws: int = 8,
    episode_ids_key: str = "test_episode_ids",
    minimum_anchor: int = 1,
) -> dict[str, np.ndarray]:
    """Replay exact simulator states and label every fixed candidate suffix."""

    if selection_pool_multiplier <= 0:
        raise ValueError("selection_pool_multiplier must be positive")
    if not 0.0 <= minimum_informative_fraction <= 1.0:
        raise ValueError("minimum_informative_fraction must lie in [0, 1]")
    pool_probes = probes * selection_pool_multiplier
    plan = build_probe_plan(
        arrays,
        task=task,
        world_model_seed=world_model_seed,
        probes=pool_probes,
        horizons=horizons,
        action_delta=action_delta,
        intervention_steps=intervention_steps,
        stochastic_dim=stochastic_dim,
        draws=draws,
        episode_ids_key=episode_ids_key,
        minimum_anchor=minimum_anchor,
    )
    episodes = np.asarray(plan["episode_ids"], dtype=np.int32)
    anchors = np.asarray(plan["anchors"], dtype=np.int32)
    action_sequences = np.asarray(plan["action_sequences"], dtype=np.float32)
    ordered_horizons = tuple(int(value) for value in plan["horizons"])
    targets = np.zeros(
        (pool_probes, action_sequences.shape[1], len(ordered_horizons)), np.float32
    )
    valid = np.zeros((pool_probes, len(ordered_horizons)), np.float32)
    replay_error = 0.0
    by_location = {
        (int(episode), int(anchor)): index
        for index, (episode, anchor) in enumerate(zip(episodes, anchors, strict=True))
    }
    environment = DMCAdapter(
        task,
        seed=derive_seed("dataset-environment", task, world_model_seed),
        action_repeat=action_repeat,
    )
    try:
        maximum_episode = int(np.max(episodes))
        for episode in range(maximum_episode + 1):
            observation = environment.reset()
            replay_error = max(
                replay_error,
                float(np.max(np.abs(observation - arrays["observations"][episode, 0]))),
            )
            for anchor in range(1, arrays["actions"].shape[1]):
                snapshot = environment.snapshot()
                probe_index = by_location.get((episode, anchor))
                if probe_index is not None:
                    for candidate in range(action_sequences.shape[1]):
                        environment.restore(snapshot)
                        cumulative = 0.0
                        survival = 1.0
                        for offset in range(action_sequences.shape[2]):
                            transition = environment.step(
                                action_sequences[probe_index, candidate, offset]
                            )
                            cumulative += (discount**offset) * survival * transition.reward
                            survival *= transition.continuation
                            if offset + 1 in ordered_horizons:
                                horizon_index = ordered_horizons.index(offset + 1)
                                targets[probe_index, candidate, horizon_index] = cumulative
                                valid[probe_index, horizon_index] = 1.0
                            if transition.is_last:
                                break
                environment.restore(snapshot)
                actual = environment.step(
                    np.asarray(arrays["actions"][episode, anchor], np.float32)
                )
                replay_error = max(
                    replay_error,
                    float(
                        np.max(
                            np.abs(
                                actual.observation
                                - arrays["observations"][episode, anchor]
                            )
                        )
                    ),
                    abs(float(actual.reward) - float(arrays["rewards"][episode, anchor])),
                    abs(
                        float(actual.continuation)
                        - float(arrays["continuations"][episode, anchor])
                    ),
                )
                if actual.is_last:
                    break
    finally:
        environment.close()
    if replay_error > 1e-6:
        raise ValueError(f"shared probe replay differs from dataset by {replay_error}")
    longest_ranges = np.ptp(targets[:, :, -1], axis=1)
    ranked = np.argsort(-longest_ranges, kind="stable")[:probes]
    informative = int(np.sum(longest_ranges[ranked] > 1e-8))
    required_informative = int(np.ceil(probes * minimum_informative_fraction))
    if informative < required_informative:
        raise ValueError(
            "shared probe bank is not informative enough at the longest horizon: "
            f"{informative}/{probes}, required {required_informative}"
        )
    # Put retained probes back into location order. Selection uses simulator
    # labels only and is therefore independent of every learned model arm.
    retained = ranked[
        np.lexsort((anchors[ranked], episodes[ranked]))
    ]
    result = {
        **{name: np.asarray(value)[retained] for name, value in plan.items() if name not in ("horizons", "model_noise")},
        "horizons": np.asarray(plan["horizons"]),
        "model_noise": np.asarray(plan["model_noise"])[:, retained],
        "simulator_returns": targets[retained],
        "horizon_mask": valid[retained],
        "maximum_replay_error": np.asarray([replay_error], np.float64),
        "candidate_pool_size": np.asarray([pool_probes], np.int32),
        "selection_horizon": np.asarray([ordered_horizons[-1]], np.int32),
        "selection_return_ranges": longest_ranges[retained].astype(np.float32),
        "action_delta": np.asarray([action_delta], np.float32),
        "intervention_steps": np.asarray([intervention_steps], np.int32),
    }
    validate_shared_probe_bank(result)
    return result


def relabel_shared_probe_bank_horizons(
    arrays: Mapping[str, np.ndarray],
    source_bank: Mapping[str, np.ndarray],
    *,
    task: str,
    world_model_seed: int,
    action_repeat: int,
    horizons: Sequence[int],
    discount: float = 0.99,
) -> dict[str, np.ndarray]:
    """Replay an existing fixed probe design at a denser set of horizons.

    States and the complete source intervention prefix are copied exactly.
    When a longer horizon is requested, every candidate receives the same
    recorded behavior-action tail, and source model-noise draws are preserved
    as a prefix. Simulator return labels and masks are then regenerated. This
    keeps the causal intervention fixed while extending its evaluation window.
    """

    validate_shared_probe_bank(source_bank)
    ordered_horizons = tuple(int(value) for value in horizons)
    source_horizon = int(np.asarray(source_bank["action_sequences"]).shape[2])
    maximum_horizon = ordered_horizons[-1] if ordered_horizons else 0
    if (
        not ordered_horizons
        or tuple(sorted(set(ordered_horizons))) != ordered_horizons
        or ordered_horizons[0] <= 0
        or maximum_horizon < source_horizon
    ):
        raise ValueError(
            "relabel horizons must be unique increasing positive integers "
            "covering the fixed source suffix"
        )
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    episodes = np.asarray(source_bank["episode_ids"], dtype=np.int32)
    anchors = np.asarray(source_bank["anchors"], dtype=np.int32)
    source_actions = np.asarray(source_bank["action_sequences"], dtype=np.float32)
    behavior_suffixes = []
    continuations = np.asarray(arrays["continuations"])
    for episode, anchor in zip(episodes, anchors, strict=True):
        episode = int(episode)
        anchor = int(anchor)
        if (
            anchor + maximum_horizon > arrays["actions"].shape[1]
            or not np.all(
                continuations[episode, anchor : anchor + maximum_horizon] > 0.0
            )
        ):
            raise ValueError("fixed probe state lacks a valid dense-horizon suffix")
        behavior_suffixes.append(
            np.asarray(
                arrays["actions"][episode, anchor : anchor + maximum_horizon],
                np.float32,
            )
        )
    behavior_actions = np.stack(behavior_suffixes)
    action_sequences = np.repeat(
        behavior_actions[:, None], source_actions.shape[1], axis=1
    )
    action_sequences[:, :, :source_horizon] = source_actions
    source_noise = np.asarray(source_bank["model_noise"])
    noise = np.empty(
        (
            source_noise.shape[0],
            source_noise.shape[1],
            maximum_horizon,
            source_noise.shape[3],
        ),
        dtype=source_noise.dtype,
    )
    noise[:, :, :source_horizon] = source_noise
    if maximum_horizon > source_horizon:
        noise[:, :, source_horizon:] = np.random.default_rng(
            derive_seed("dense-probe-model-noise-tail", task, world_model_seed)
        ).standard_normal(noise[:, :, source_horizon:].shape).astype(noise.dtype)
    targets = np.zeros(
        (len(episodes), action_sequences.shape[1], len(ordered_horizons)),
        np.float32,
    )
    valid = np.zeros((len(episodes), len(ordered_horizons)), np.float32)
    horizon_indices = {
        horizon: index for index, horizon in enumerate(ordered_horizons)
    }
    by_location = {
        (int(episode), int(anchor)): index
        for index, (episode, anchor) in enumerate(zip(episodes, anchors, strict=True))
    }
    if len(by_location) != len(episodes):
        raise ValueError("source probe locations must be unique")
    replay_error = 0.0
    environment = DMCAdapter(
        task,
        seed=derive_seed("dataset-environment", task, world_model_seed),
        action_repeat=action_repeat,
    )
    try:
        maximum_episode = int(np.max(episodes))
        for episode in range(maximum_episode + 1):
            observation = environment.reset()
            replay_error = max(
                replay_error,
                float(
                    np.max(
                        np.abs(observation - arrays["observations"][episode, 0])
                    )
                ),
            )
            for anchor in range(1, arrays["actions"].shape[1]):
                snapshot = environment.snapshot()
                probe_index = by_location.get((episode, anchor))
                if probe_index is not None:
                    for candidate in range(action_sequences.shape[1]):
                        environment.restore(snapshot)
                        cumulative = 0.0
                        survival = 1.0
                        for offset in range(maximum_horizon):
                            transition = environment.step(
                                action_sequences[probe_index, candidate, offset]
                            )
                            cumulative += (
                                (discount**offset) * survival * transition.reward
                            )
                            survival *= transition.continuation
                            horizon = offset + 1
                            if horizon in horizon_indices:
                                horizon_index = horizon_indices[horizon]
                                targets[probe_index, candidate, horizon_index] = cumulative
                                valid[probe_index, horizon_index] = 1.0
                            if transition.is_last:
                                break
                environment.restore(snapshot)
                actual = environment.step(
                    np.asarray(arrays["actions"][episode, anchor], np.float32)
                )
                replay_error = max(
                    replay_error,
                    float(
                        np.max(
                            np.abs(
                                actual.observation
                                - arrays["observations"][episode, anchor]
                            )
                        )
                    ),
                    abs(float(actual.reward) - float(arrays["rewards"][episode, anchor])),
                    abs(
                        float(actual.continuation)
                        - float(arrays["continuations"][episode, anchor])
                    ),
                )
                if actual.is_last:
                    break
    finally:
        environment.close()
    if replay_error > 1e-6:
        raise ValueError(f"shared probe replay differs from dataset by {replay_error}")
    result = {
        "episode_ids": np.asarray(source_bank["episode_ids"]).copy(),
        "anchors": np.asarray(source_bank["anchors"]).copy(),
        "horizons": np.asarray(ordered_horizons, np.int32),
        "action_sequences": action_sequences.copy(),
        "model_noise": noise,
        "simulator_returns": targets,
        "horizon_mask": valid,
        "maximum_replay_error": np.asarray([replay_error], np.float64),
        "candidate_pool_size": np.asarray(source_bank["candidate_pool_size"]).copy(),
        "selection_horizon": np.asarray(source_bank["selection_horizon"]).copy(),
        "selection_return_ranges": np.asarray(
            source_bank["selection_return_ranges"]
        ).copy(),
        "action_delta": np.asarray(source_bank["action_delta"]).copy(),
        "intervention_steps": np.asarray(source_bank["intervention_steps"]).copy(),
    }
    validate_shared_probe_bank(result)
    return result


def validate_shared_probe_bank(bank: Mapping[str, np.ndarray]) -> None:
    required = {
        "episode_ids",
        "anchors",
        "horizons",
        "action_sequences",
        "model_noise",
        "simulator_returns",
        "horizon_mask",
        "maximum_replay_error",
        "candidate_pool_size",
        "selection_horizon",
        "selection_return_ranges",
        "action_delta",
        "intervention_steps",
    }
    if set(bank) != required:
        raise ValueError("shared probe bank fields differ from schema")
    probes = len(bank["episode_ids"])
    horizons = np.asarray(bank["horizons"])
    actions = np.asarray(bank["action_sequences"])
    noise = np.asarray(bank["model_noise"])
    targets = np.asarray(bank["simulator_returns"])
    mask = np.asarray(bank["horizon_mask"])
    if (
        probes <= 0
        or bank["anchors"].shape != (probes,)
        or horizons.ndim != 1
        or actions.ndim != 4
        or actions.shape[0] != probes
        or actions.shape[2] != int(horizons[-1])
        or noise.ndim != 4
        or noise.shape[1] != probes
        or noise.shape[2] != actions.shape[2]
        or targets.shape != (probes, actions.shape[1], len(horizons))
        or mask.shape != (probes, len(horizons))
        or np.asarray(bank["candidate_pool_size"]).shape != (1,)
        or int(np.asarray(bank["candidate_pool_size"])[0]) < probes
        or np.asarray(bank["selection_horizon"]).shape != (1,)
        or int(np.asarray(bank["selection_horizon"])[0])
        not in {int(value) for value in horizons}
        or np.asarray(bank["selection_return_ranges"]).shape != (probes,)
        or np.asarray(bank["action_delta"]).shape != (1,)
        or np.asarray(bank["intervention_steps"]).shape != (1,)
        or not 0 < int(np.asarray(bank["intervention_steps"])[0]) <= int(horizons[-1])
        or not np.all((mask == 0.0) | (mask == 1.0))
    ):
        raise ValueError("shared probe bank shapes are invalid")
    for value in bank.values():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("shared probe bank contains non-finite values")


def shared_probe_manifest(
    bank: Mapping[str, np.ndarray],
    *,
    task: str,
    world_model_seed: int,
    dataset_sha256: str,
    action_delta: float,
) -> dict[str, Any]:
    validate_shared_probe_bank(bank)
    retained_delta = float(np.asarray(bank["action_delta"])[0])
    if not np.isclose(retained_delta, action_delta):
        raise ValueError("manifest action_delta differs from retained bank")
    manifest = {
        "schema_version": SCHEMA,
        "task": task,
        "world_model_seed": int(world_model_seed),
        "dataset_sha256": str(dataset_sha256),
        "probe_bank_sha256": array_sha256(bank),
        "probe_count": int(len(bank["episode_ids"])),
        "candidate_count": int(bank["action_sequences"].shape[1]),
        "horizons": [int(value) for value in bank["horizons"]],
        "draws": int(bank["model_noise"].shape[0]),
        "action_delta": float(action_delta),
        "intervention_steps": int(np.asarray(bank["intervention_steps"])[0]),
        "candidate_pool_size": int(np.asarray(bank["candidate_pool_size"])[0]),
        "selection_horizon": int(np.asarray(bank["selection_horizon"])[0]),
        "informative_states_at_selection_horizon": int(
            np.sum(np.asarray(bank["selection_return_ranges"]) > 1e-8)
        ),
        "policy_independent": True,
        "selection_rule": "largest_simulator_return_range_at_longest_horizon_before_model_evaluation",
        "candidate_rule": "replay_suffix_with_coordinatewise_minus_plus_delta_for_fixed_prefix",
        "noise_rule": "same_standard_normal_draw_per_state_step_across_candidates_and_model_arms",
    }
    return {**manifest, "manifest_sha256": _json_digest(manifest)}


def advantage_training_arrays(
    bank: Mapping[str, np.ndarray],
    *,
    episodes: int,
    steps: int,
    horizons: Sequence[int] = (1, 3, 5),
) -> dict[str, np.ndarray]:
    """Scatter a train-split bank into fields consumed by the JAX objective."""

    validate_shared_probe_bank(bank)
    selected_horizons = tuple(int(value) for value in horizons)
    available = [int(value) for value in bank["horizons"]]
    if any(value not in available for value in selected_horizons):
        raise ValueError("requested training horizon is absent from the bank")
    action_sequences = np.asarray(bank["action_sequences"], dtype=np.float32)
    candidates, maximum_horizon, action_dim = action_sequences.shape[1:]
    if maximum_horizon < max(selected_horizons):
        raise ValueError("probe suffix is shorter than a training horizon")
    actions = np.zeros(
        (episodes, steps, candidates, max(selected_horizons), action_dim),
        np.float32,
    )
    returns = np.zeros(
        (episodes, steps, candidates, len(selected_horizons)), np.float32
    )
    mask = np.zeros((episodes, steps, len(selected_horizons)), np.float32)
    indices = [available.index(value) for value in selected_horizons]
    for probe, (episode, anchor) in enumerate(
        zip(bank["episode_ids"], bank["anchors"], strict=True)
    ):
        episode = int(episode)
        anchor = int(anchor)
        if not 0 <= episode < episodes or not 0 <= anchor < steps:
            raise ValueError("probe location lies outside requested training tensor")
        actions[episode, anchor] = action_sequences[
            probe, :, : max(selected_horizons)
        ]
        returns[episode, anchor] = bank["simulator_returns"][probe][:, indices]
        mask[episode, anchor] = bank["horizon_mask"][probe][indices]
    return {
        "advantage_action_sequences": actions,
        "advantage_target_returns": returns,
        "advantage_mask": mask,
    }


def _model_probe_returns(
    params: Any,
    initial: Any,
    action_sequences: Any,
    noise: Any,
    config: Any,
    horizons: tuple[int, ...],
) -> Any:
    """JAX implementation; arrays are draw x probe x candidate x step."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        RSSMState,
        predict_continuation_logits,
        predict_transition_reward,
        sample_prior,
        transition_deterministic,
    )

    draws, probes = noise.shape[:2]
    candidates = action_sequences.shape[1]
    state = RSSMState(
        jnp.broadcast_to(
            initial.deterministic[None, :, None, :],
            (draws, probes, candidates, config.deterministic_dim),
        ),
        jnp.broadcast_to(
            initial.stochastic[None, :, None, :],
            (draws, probes, candidates, config.stochastic_dim),
        ),
    )
    cumulative = jnp.zeros((draws, probes, candidates), action_sequences.dtype)
    survival = jnp.ones_like(cumulative)
    selected = []
    for offset in range(action_sequences.shape[2]):
        previous_feature = state.feature
        action = jnp.broadcast_to(
            action_sequences[None, :, :, offset, :],
            (draws, probes, candidates, config.action_dim),
        )
        deterministic = transition_deterministic(params, state, action, config)
        step_noise = jnp.broadcast_to(
            noise[:, :, None, offset, :],
            (draws, probes, candidates, config.stochastic_dim),
        )
        stochastic, _ = sample_prior(
            params,
            deterministic.reshape((-1, config.deterministic_dim)),
            None,
            config,
            noise=step_noise.reshape((-1, config.stochastic_dim)),
        )
        state = RSSMState(
            deterministic,
            stochastic.reshape((draws, probes, candidates, config.stochastic_dim)),
        )
        reward = predict_transition_reward(
            params, previous_feature, action, state.feature, config
        )
        continuation = jax.nn.sigmoid(
            predict_continuation_logits(params, state.feature)
        )
        cumulative = cumulative + (config.discount**offset) * survival * reward
        survival = survival * continuation
        if offset + 1 in horizons:
            selected.append(cumulative)
    return jnp.stack(selected, axis=-1)


def evaluate_shared_probe_bank(
    params: Any,
    config: Any,
    arrays: Mapping[str, np.ndarray],
    bank: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate a checkpoint on the bank without consulting a learned actor."""

    import jax.numpy as jnp
    from imf_dreamer_jax import RSSMState

    validate_shared_probe_bank(bank)
    deterministic = []
    stochastic = []
    for episode, anchor in zip(bank["episode_ids"], bank["anchors"], strict=True):
        state = posterior_mean_filter(
            params,
            jnp.asarray(arrays["observations"][int(episode), : int(anchor)][None]),
            jnp.asarray(arrays["actions"][int(episode), : int(anchor)][None]),
            config,
        )
        deterministic.append(np.asarray(state.deterministic[0]))
        stochastic.append(np.asarray(state.stochastic[0]))
    initial = RSSMState(jnp.asarray(np.stack(deterministic)), jnp.asarray(np.stack(stochastic)))
    predicted_draws = _model_probe_returns(
        params,
        initial,
        jnp.asarray(bank["action_sequences"]),
        jnp.asarray(bank["model_noise"]),
        config,
        tuple(int(value) for value in bank["horizons"]),
    )
    predicted = np.asarray(predicted_draws).mean(axis=0)
    targets = np.asarray(bank["simulator_returns"])
    metrics = {}
    for index, horizon in enumerate(bank["horizons"]):
        active = np.asarray(bank["horizon_mask"][:, index] > 0.0)
        metrics[str(int(horizon))] = action_ranking_metrics(
            predicted[active, :, index], targets[active, :, index]
        )
    return {
        "probe_bank_sha256": array_sha256(bank),
        "predicted_returns": predicted,
        "predicted_return_draws": np.asarray(predicted_draws),
        "metrics_by_horizon": metrics,
    }


__all__ = [
    "SCHEMA",
    "advantage_training_arrays",
    "build_probe_plan",
    "evaluate_shared_probe_bank",
    "generate_shared_probe_bank",
    "relabel_shared_probe_bank_horizons",
    "shared_probe_manifest",
    "validate_shared_probe_bank",
]
