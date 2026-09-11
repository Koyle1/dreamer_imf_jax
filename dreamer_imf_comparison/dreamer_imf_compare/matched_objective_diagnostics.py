"""Fail-closed registered scalar-transition diagnostics.

This module is deliberately separate from the primary matched-objective
runner.  A registered run consumes its frozen confirmatory matrix and selected
per-arm configurations, trains through the shared ``imf_dreamer_jax`` world
model API, and writes the exact ``diagnostic_summary.json`` schema required by
``validate_confirmatory_auxiliary_evidence``.

The JSON summary is never treated as primary evidence.  Every estimand and
threshold decision is recomputed from a canonical numeric NPZ payload.  A
smoke run uses reduced counts, is written below ``diagnostics/smoke``, and uses
the deliberately validator-ineligible status ``smoke_nonclaim``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator, Mapping

import numpy as np

from .artifacts import read_json
from .matched_objective_benchmark import (
    array_sha256,
    canonical_bytes,
    derive_jax_key,
    derive_seed,
    file_sha256,
    make_config,
    matrix_cells,
    object_sha256,
    open_loop_samples_with_continuation,
    posterior_mean_filter,
    runtime_fingerprint,
    runtime_homogeneity_identity,
    stage_directory,
    validate_compute_plan_cell,
    validate_matrix,
    validate_source_manifest,
)
from .matched_objective_protocol import (
    ARM_ORDER,
    protocol_digest,
    validate_matched_objective_protocol,
)


DIAGNOSTIC_SCHEMA = "matched-objective-diagnostics-v1"
TRAINING_MANIFEST_SCHEMA = "matched-objective-diagnostic-training-v1"
ARM_RESULT_SCHEMA = "matched-objective-diagnostic-arm-v1"
DETERMINISTIC_NAME = "deterministic_transition_diagnostic"
STOCHASTIC_NAME = "genuinely_stochastic_transition_diagnostic"
DIAGNOSTIC_NAMES = (DETERMINISTIC_NAME, STOCHASTIC_NAME)
DETERMINISTIC_RAW_FIELDS = frozenset(
    {
        "horizons",
        "initial_states",
        "future_actions",
        "target_futures",
        "model_futures",
        "training_observation_std",
        "test_seed",
        "model_noise_key",
        "ratio_epsilon",
    }
)
STOCHASTIC_RAW_FIELDS = frozenset(
    {
        "horizons",
        "score_horizons",
        "rollout_horizons",
        "initial_states",
        "future_actions",
        "oracle_reference_futures",
        "evaluation_futures",
        "evaluation_branches",
        "model_futures",
        "training_observation_std",
        "ece_bin_edges",
        "test_seed",
        "model_noise_key",
    }
)
ENERGY_HORIZONS = (1, 8, 30)
ECE_BINS = 10
RATIO_EPSILON = 1e-12
REGISTERED_EPISODE_LENGTH = 64
REGISTERED_DETERMINISTIC_CONDITIONS = 512
REGISTERED_STOCHASTIC_CONDITIONS = 512
REGISTERED_STOCHASTIC_DRAWS = 64
REGISTERED_RETAINED_HORIZONS = tuple(range(1, 31))
REGISTERED_ROLLOUT_HORIZONS = (1, 2, 4, 8, 15, 30)
TRAINING_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "diagnostic_name",
        "arm",
        "protocol_sha256",
        "source_sha256",
        "matrix_sha256",
        "selection_sha256",
        "smoke_nonclaim",
        "effective_updates",
        "effective_config_sha256",
        "dataset_file_sha256",
        "schedule_file_sha256",
        "identity_sha256",
        "selected_candidate",
        "effective_config",
        "checkpoint_trace",
        "checkpoint_sha256",
        "wall_seconds",
        "runtime",
        "training_manifest_sha256",
    }
)
CHECKPOINT_METADATA_FIELDS = frozenset(
    {
        "stage",
        "identity_sha256",
        "completed_updates",
        "wall_seconds",
        "checkpoint_trace",
    }
)
CHECKPOINT_STAGE = "matched_objective_diagnostic"
CHECKPOINT_TRACE_METRIC_FIELDS = (
    "total",
    "reconstruction",
    "reward",
    "continuation",
    "prior",
    "representation",
    "overshooting",
    "imf_loss_u",
    "imf_loss_v",
    "imf_endpoint",
    "imf_shortcut",
    "overshooting_distance_5",
    "overshooting_distance_15",
)


@dataclass(frozen=True)
class DiagnosticPlan:
    """Effective execution counts, with an explicit claim boundary."""

    smoke_nonclaim: bool
    status: str
    episode_length: int
    deterministic_training_episodes: int
    deterministic_test_conditions: int
    deterministic_repeated_futures: int
    deterministic_updates: int
    stochastic_training_episodes: int
    stochastic_test_conditions: int
    stochastic_repeated_futures: int
    stochastic_predictive_draws: int
    stochastic_oracle_reference_futures: int
    stochastic_evaluation_futures: int
    stochastic_updates: int
    batch_size: int
    checkpoint_every: int


def _without_digest(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    function = getattr(np, "trapezoid", None)
    if function is None:  # NumPy 1.26 compatibility.
        function = np.trapz
    return float(function(values, coordinates) / (coordinates[-1] - coordinates[0]))


def resolve_diagnostic_plan(
    protocol: Mapping[str, Any],
    *,
    smoke: bool = False,
    smoke_updates: int = 2,
    smoke_training_episodes: int = 16,
    smoke_test_conditions: int = 8,
    smoke_repeated_futures: int = 8,
    smoke_batch_size: int = 4,
) -> DiagnosticPlan:
    """Resolve registered counts or a bounded, explicitly non-claim smoke plan."""

    validate_matched_objective_protocol(protocol)
    deterministic = protocol["diagnostics"][DETERMINISTIC_NAME]
    stochastic = protocol["diagnostics"][STOCHASTIC_NAME]
    if not smoke:
        plan = DiagnosticPlan(
            smoke_nonclaim=False,
            status="complete",
            episode_length=int(deterministic["episode_length"]),
            deterministic_training_episodes=int(deterministic["training_episodes"]),
            deterministic_test_conditions=int(deterministic["test_initial_conditions"]),
            deterministic_repeated_futures=int(
                deterministic["repeated_futures_per_identical_condition"]
            ),
            deterministic_updates=int(deterministic["world_model_updates"]),
            stochastic_training_episodes=int(stochastic["training_episodes"]),
            stochastic_test_conditions=int(stochastic["test_conditions"]),
            stochastic_repeated_futures=int(
                stochastic["repeated_futures_per_identical_condition"]
            ),
            stochastic_predictive_draws=int(
                stochastic["predictive_draws_per_condition"]
            ),
            stochastic_oracle_reference_futures=int(
                stochastic["oracle_reference_futures_per_condition"]
            ),
            stochastic_evaluation_futures=int(
                stochastic["evaluation_futures_per_condition"]
            ),
            stochastic_updates=int(stochastic["world_model_updates"]),
            batch_size=32,
            checkpoint_every=500,
        )
        registered_identity = (
            plan.episode_length,
            plan.deterministic_test_conditions,
            plan.deterministic_repeated_futures,
            plan.stochastic_test_conditions,
            plan.stochastic_repeated_futures,
            plan.stochastic_predictive_draws,
            plan.stochastic_oracle_reference_futures,
            plan.stochastic_evaluation_futures,
        )
        expected_identity = (
            REGISTERED_EPISODE_LENGTH,
            REGISTERED_DETERMINISTIC_CONDITIONS,
            1,
            REGISTERED_STOCHASTIC_CONDITIONS,
            REGISTERED_STOCHASTIC_DRAWS,
            REGISTERED_STOCHASTIC_DRAWS,
            REGISTERED_STOCHASTIC_DRAWS,
            REGISTERED_STOCHASTIC_DRAWS,
        )
        _require(
            registered_identity == expected_identity,
            "diagnostic protocol counts changed without a runner version update",
        )
        _require(
            tuple(int(value) for value in stochastic["rollout_horizons"])
            == REGISTERED_ROLLOUT_HORIZONS,
            "registered stochastic rollout horizons changed without a runner update",
        )
        _require(
            int(stochastic["branch_probability_calibration_bins"]) == ECE_BINS,
            "registered calibration bins changed without a runner update",
        )
        return plan
    values = {
        "smoke_updates": smoke_updates,
        "smoke_training_episodes": smoke_training_episodes,
        "smoke_test_conditions": smoke_test_conditions,
        "smoke_repeated_futures": smoke_repeated_futures,
        "smoke_batch_size": smoke_batch_size,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values.values()
    ):
        raise ValueError("all smoke counts must be positive integers")
    if smoke_repeated_futures < 4 or smoke_repeated_futures % 2:
        raise ValueError("smoke repeated futures must be an even integer >= 4")
    if smoke_updates > min(
        int(deterministic["world_model_updates"]),
        int(stochastic["world_model_updates"]),
    ):
        raise ValueError("smoke updates cannot exceed a registered update budget")
    if smoke_training_episodes > min(
        int(deterministic["training_episodes"]),
        int(stochastic["training_episodes"]),
    ):
        raise ValueError("smoke training episodes cannot exceed registered counts")
    if smoke_test_conditions > min(
        int(deterministic["test_initial_conditions"]),
        int(stochastic["test_conditions"]),
    ):
        raise ValueError("smoke test conditions cannot exceed registered counts")
    return DiagnosticPlan(
        smoke_nonclaim=True,
        status="smoke_nonclaim",
        episode_length=int(deterministic["episode_length"]),
        deterministic_training_episodes=smoke_training_episodes,
        deterministic_test_conditions=smoke_test_conditions,
        deterministic_repeated_futures=1,
        deterministic_updates=smoke_updates,
        stochastic_training_episodes=smoke_training_episodes,
        stochastic_test_conditions=smoke_test_conditions,
        stochastic_repeated_futures=smoke_repeated_futures,
        stochastic_predictive_draws=smoke_repeated_futures,
        stochastic_oracle_reference_futures=smoke_repeated_futures,
        stochastic_evaluation_futures=smoke_repeated_futures,
        stochastic_updates=smoke_updates,
        batch_size=min(smoke_batch_size, smoke_training_episodes),
        checkpoint_every=1,
    )


def _action_sequences(random: np.random.Generator, count: int, length: int) -> np.ndarray:
    phase = random.uniform(0.0, 2.0 * np.pi, size=(count, 1))
    innovation = random.uniform(-1.0, 1.0, size=(count, length))
    time_index = np.arange(length, dtype=np.float64)[None, :]
    actions = np.clip(
        0.6 * np.sin(0.17 * time_index + phase) + 0.2 * innovation,
        -1.0,
        1.0,
    )
    return actions.astype(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _simulate_deterministic(initial: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state = np.asarray(initial, dtype=np.float64).copy()
    futures = np.empty(actions.shape, dtype=np.float64)
    for step in range(actions.shape[1]):
        state = 0.85 * state + 0.10 * np.tanh(2.0 * state) + 0.20 * actions[:, step]
        futures[:, step] = state
    return futures.astype(np.float32)


def _simulate_stochastic(
    initial: np.ndarray,
    actions: np.ndarray,
    repeated_futures: int,
    random: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count, length = actions.shape
    state = np.broadcast_to(initial[:, None], (count, repeated_futures)).astype(
        np.float64
    ).copy()
    futures = np.empty((count, repeated_futures, length), dtype=np.float64)
    branches = np.empty((count, repeated_futures, length), dtype=np.uint8)
    innovations = np.empty((count, repeated_futures, length), dtype=np.float64)
    for step in range(length):
        action = actions[:, step, None]
        probability = _sigmoid(2.0 * state + 0.5 * action)
        branch = random.random((count, repeated_futures)) < probability
        innovation = random.normal(0.0, 0.15, size=(count, repeated_futures))
        state = 0.70 * state + 0.20 * action + 0.60 * (2.0 * branch - 1.0) + innovation
        futures[:, :, step] = state
        branches[:, :, step] = branch
        innovations[:, :, step] = innovation
    return (
        futures.astype(np.float32),
        branches,
        innovations.astype(np.float32),
    )


def generate_scalar_training_data(
    diagnostic_name: str,
    *,
    episodes: int,
    episode_length: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate the exact registered scalar process in world-model batch format."""

    if diagnostic_name not in DIAGNOSTIC_NAMES:
        raise ValueError(f"unknown diagnostic {diagnostic_name!r}")
    if episodes <= 0 or episode_length <= 0:
        raise ValueError("episodes and episode_length must be positive")
    random = np.random.default_rng(seed)
    initial = random.uniform(-1.0, 1.0, size=episodes).astype(np.float32)
    future_actions = _action_sequences(random, episodes, episode_length)
    if diagnostic_name == DETERMINISTIC_NAME:
        future = _simulate_deterministic(initial, future_actions)
        branches = np.zeros((episodes, episode_length), dtype=np.uint8)
        innovations = np.zeros((episodes, episode_length), dtype=np.float32)
    else:
        repeated, branch_values, innovation_values = _simulate_stochastic(
            initial, future_actions, 1, random
        )
        future = repeated[:, 0]
        branches = branch_values[:, 0]
        innovations = innovation_values[:, 0]
    observations = np.concatenate((initial[:, None], future), axis=1)[..., None]
    actions = np.zeros((episodes, episode_length + 1, 1), dtype=np.float32)
    actions[:, 1:, 0] = future_actions
    rewards = np.zeros((episodes, episode_length + 1), dtype=np.float32)
    continuations = np.ones_like(rewards)
    is_first = np.zeros((episodes, episode_length + 1), dtype=np.bool_)
    is_first[:, 0] = True
    return {
        "observations": observations.astype(np.float32),
        "actions": actions,
        "rewards": rewards,
        "continuations": continuations,
        "is_first": is_first,
        "branches": branches,
        "innovations": innovations,
        "generation_seed": np.asarray([seed], dtype=np.uint32),
    }


def generate_scalar_evaluation_data(
    diagnostic_name: str,
    *,
    conditions: int,
    episode_length: int,
    repeated_futures: int,
    seed: int,
    oracle_reference_futures: int | None = None,
    evaluation_futures: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate paired held-out conditions and true/oracle futures."""

    if diagnostic_name not in DIAGNOSTIC_NAMES:
        raise ValueError(f"unknown diagnostic {diagnostic_name!r}")
    random = np.random.default_rng(seed)
    initial = random.uniform(-1.0, 1.0, size=conditions).astype(np.float32)
    actions = _action_sequences(random, conditions, episode_length)
    if diagnostic_name == DETERMINISTIC_NAME:
        return {
            "initial_states": initial,
            "future_actions": actions,
            "target_futures": _simulate_deterministic(initial, actions),
        }
    reference_count = (
        repeated_futures
        if oracle_reference_futures is None
        else int(oracle_reference_futures)
    )
    evaluation_count = (
        repeated_futures if evaluation_futures is None else int(evaluation_futures)
    )
    if min(repeated_futures, reference_count, evaluation_count) <= 0:
        raise ValueError("all stochastic future counts must be positive")
    if repeated_futures != evaluation_count:
        raise ValueError(
            "repeated_futures must equal the independent evaluation-future count"
        )
    reference, _, _ = _simulate_stochastic(
        initial, actions, reference_count, random
    )
    evaluation, branches, _ = _simulate_stochastic(
        initial, actions, evaluation_count, random
    )
    return {
        "initial_states": initial,
        "future_actions": actions,
        "oracle_reference_futures": reference,
        "evaluation_futures": evaluation,
        "evaluation_branches": branches,
    }


def _validate_numeric_raw(
    raw: Mapping[str, np.ndarray], expected: frozenset[str], name: str
) -> None:
    if set(raw) != set(expected):
        raise ValueError(
            f"{name} raw fields differ; missing={sorted(expected - set(raw))}, "
            f"extra={sorted(set(raw) - expected)}"
        )
    for field, value in raw.items():
        array = np.asarray(value)
        if array.size == 0 or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name}.{field} must be a nonempty numeric array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}.{field} contains non-finite values")


def deterministic_estimands_from_raw(
    raw: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, bool], bool]:
    """Recompute every deterministic estimand and registered threshold."""

    _validate_numeric_raw(raw, DETERMINISTIC_RAW_FIELDS, DETERMINISTIC_NAME)
    horizons = np.asarray(raw["horizons"], dtype=np.int64)
    targets = np.asarray(raw["target_futures"], dtype=np.float64)
    predictions = np.asarray(raw["model_futures"], dtype=np.float64)
    _require(horizons.ndim == 1 and np.array_equal(horizons, np.arange(1, 31)),
             "deterministic horizons must be exactly 1..30")
    _require(targets.shape == predictions.shape, "deterministic prediction shape mismatch")
    _require(targets.ndim == 2 and targets.shape[1] == horizons.size,
             "deterministic futures must have shape [conditions, 30]")
    scale = max(_finite_scalar(np.asarray(raw["training_observation_std"]).reshape(-1)[0],
                               "training observation std"), 1e-6)
    normalized_rmse = np.sqrt(np.mean(np.square(predictions - targets), axis=0)) / scale
    epsilon = _finite_scalar(np.asarray(raw["ratio_epsilon"]).reshape(-1)[0], "ratio epsilon")
    _require(epsilon > 0.0, "ratio epsilon must be positive")
    ratio = float(normalized_rmse[29] / max(float(normalized_rmse[0]), epsilon))
    estimands = {
        "one_step_normalized_RMSE": float(normalized_rmse[0]),
        "horizon_30_normalized_RMSE": float(normalized_rmse[29]),
        "normalized_rollout_error_AUC": _trapezoid(
            normalized_rmse, horizons.astype(np.float64)
        ),
        "horizon_30_to_horizon_1_error_ratio": ratio,
    }
    thresholds = {
        "one_step_normalized_RMSE_max": estimands["one_step_normalized_RMSE"] <= 0.05,
        "horizon_30_normalized_RMSE_max": estimands["horizon_30_normalized_RMSE"] <= 0.25,
        "horizon_30_to_horizon_1_error_ratio_max": ratio <= 5.0,
    }
    _require(all(math.isfinite(value) for value in estimands.values()),
             "deterministic estimands are non-finite")
    return estimands, thresholds, all(thresholds.values())


def _energy_score(prediction: np.ndarray, observation: np.ndarray) -> float:
    """Empirical scalar energy score averaged over conditions."""

    cross = np.mean(np.abs(prediction[:, :, None] - observation[:, None, :]), axis=(1, 2))
    self_distance = np.mean(
        np.abs(prediction[:, :, None] - prediction[:, None, :]), axis=(1, 2)
    )
    return float(np.mean(cross - 0.5 * self_distance))


def _branch_ece(
    initial: np.ndarray,
    first_action: np.ndarray,
    first_model_futures: np.ndarray,
    bin_edges: np.ndarray,
) -> float:
    base = 0.70 * initial[:, None] + 0.20 * first_action[:, None]
    predicted = np.mean(first_model_futures - base >= 0.0, axis=1)
    truth = _sigmoid(2.0 * initial + 0.5 * first_action)
    bin_index = np.clip(np.searchsorted(bin_edges, predicted, side="right") - 1, 0,
                        bin_edges.size - 2)
    ece = 0.0
    for index in range(bin_edges.size - 1):
        selected = bin_index == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(predicted[selected])) - float(np.mean(truth[selected]))
            )
    return float(ece)


def stochastic_estimands_from_raw(
    raw: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, bool], bool]:
    """Recompute stochastic distributional estimands and decisions."""

    _validate_numeric_raw(raw, STOCHASTIC_RAW_FIELDS, STOCHASTIC_NAME)
    horizons = np.asarray(raw["horizons"], dtype=np.int64)
    score_horizons = np.asarray(raw["score_horizons"], dtype=np.int64)
    rollout_horizons = np.asarray(raw["rollout_horizons"], dtype=np.int64)
    oracle = np.asarray(raw["oracle_reference_futures"], dtype=np.float64)
    evaluation = np.asarray(raw["evaluation_futures"], dtype=np.float64)
    model = np.asarray(raw["model_futures"], dtype=np.float64)
    initial = np.asarray(raw["initial_states"], dtype=np.float64)
    actions = np.asarray(raw["future_actions"], dtype=np.float64)
    branches = np.asarray(raw["evaluation_branches"])
    edges = np.asarray(raw["ece_bin_edges"], dtype=np.float64)
    _require(np.array_equal(horizons, np.arange(1, 31)),
             "stochastic horizons must be exactly 1..30")
    _require(np.array_equal(score_horizons, np.asarray(ENERGY_HORIZONS)),
             "stochastic score horizons must be exactly 1, 8, 30")
    _require(np.array_equal(rollout_horizons, np.asarray((1, 2, 4, 8, 15, 30))),
             "stochastic rollout horizons differ from 1, 2, 4, 8, 15, 30")
    _require(oracle.ndim == model.ndim == evaluation.ndim == 3,
             "stochastic futures must have [conditions, draws, horizon]")
    _require(
        oracle.shape[0] == model.shape[0] == evaluation.shape[0]
        and oracle.shape[2] == model.shape[2] == evaluation.shape[2] == horizons.size,
        "stochastic future condition/horizon axes differ",
    )
    _require(min(oracle.shape[1], model.shape[1], evaluation.shape[1]) >= 4,
             "each stochastic sample set must retain at least four futures")
    _require(branches.shape == evaluation.shape, "evaluation branch shape mismatch")
    _require(initial.shape == (model.shape[0],), "initial-state shape mismatch")
    _require(actions.shape == (model.shape[0], horizons.size), "action shape mismatch")
    _require(np.array_equal(edges, np.linspace(0.0, 1.0, ECE_BINS + 1)),
             "ECE bin edges differ from the registered runner definition")
    ratios: dict[int, float] = {}
    coverages: dict[int, float] = {}
    for horizon in score_horizons:
        index = int(horizon) - 1
        model_score = _energy_score(model[:, :, index], evaluation[:, :, index])
        oracle_score = _energy_score(oracle[:, :, index], evaluation[:, :, index])
        _require(model_score >= -1e-12, "model energy score is materially negative")
        _require(oracle_score > 0.0, "oracle energy-score baseline must be positive")
        ratios[int(horizon)] = max(model_score, 0.0) / oracle_score
        lower = np.quantile(model[:, :, index], 0.05, axis=1)
        upper = np.quantile(model[:, :, index], 0.95, axis=1)
        coverage = np.mean(
            (evaluation[:, :, index] >= lower[:, None])
            & (evaluation[:, :, index] <= upper[:, None])
        )
        coverages[int(horizon)] = float(coverage)
    model_mean = np.mean(model, axis=1)
    evaluation_mean = np.mean(evaluation, axis=1)
    scale = max(_finite_scalar(np.asarray(raw["training_observation_std"]).reshape(-1)[0],
                               "training observation std"), 1e-6)
    normalized_rmse = np.sqrt(
        np.mean(np.square(model_mean - evaluation_mean), axis=0)
    ) / scale
    ece = _branch_ece(initial, actions[:, 0], model[:, :, 0], edges)
    estimands = {
        **{
            f"energy_score_over_oracle_ratio_horizon_{horizon}": float(ratios[horizon])
            for horizon in ENERGY_HORIZONS
        },
        **{
            f"central_90_percent_interval_coverage_horizon_{horizon}": float(
                coverages[horizon]
            )
            for horizon in ENERGY_HORIZONS
        },
        "branch_probability_expected_calibration_error": ece,
        "normalized_rollout_error_AUC": _trapezoid(
            normalized_rmse[rollout_horizons - 1], rollout_horizons.astype(np.float64)
        ),
    }
    thresholds = {
        **{
            f"energy_score_over_oracle_ratio_horizon_{horizon}_max": ratios[horizon]
            <= 1.25
            for horizon in ENERGY_HORIZONS
        },
        **{
            f"central_90_percent_interval_coverage_horizon_{horizon}_min": coverages[
                horizon
            ]
            >= 0.85
            for horizon in ENERGY_HORIZONS
        },
        **{
            f"central_90_percent_interval_coverage_horizon_{horizon}_max": coverages[
                horizon
            ]
            <= 0.95
            for horizon in ENERGY_HORIZONS
        },
        "branch_probability_expected_calibration_error_max": ece <= 0.05,
    }
    _require(all(math.isfinite(value) for value in estimands.values()),
             "stochastic estimands are non-finite")
    return estimands, thresholds, all(thresholds.values())


def recompute_diagnostic_raw(
    diagnostic_name: str, raw: Mapping[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, bool], bool]:
    if diagnostic_name == DETERMINISTIC_NAME:
        return deterministic_estimands_from_raw(raw)
    if diagnostic_name == STOCHASTIC_NAME:
        return stochastic_estimands_from_raw(raw)
    raise ValueError(f"unknown diagnostic {diagnostic_name!r}")


def load_numeric_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Create a canonical payload once; never replace completed evidence."""

    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if path.exists():
        existing = load_numeric_npz(path)
        if set(existing) != set(normalized) or any(
            not np.array_equal(existing[name], normalized[name], equal_nan=False)
            for name in normalized
        ):
            raise FileExistsError(f"existing NPZ differs and will not be overwritten: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **normalized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to race an existing artifact: {path}"
            ) from error
        os.unlink(temporary)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create immutable JSON, accepting only an identical completed rerun."""

    if path.exists():
        if canonical_bytes(read_json(path)) != canonical_bytes(payload):
            raise FileExistsError(f"existing JSON differs and will not be overwritten: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=path.name + ".", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to race an existing artifact: {path}"
            ) from error
        os.unlink(temporary)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _raw_file_entry(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("diagnostic artifact escapes output root") from error
    if ".." in relative.parts or path.suffix != ".npz" or not path.is_file():
        raise ValueError("diagnostic raw artifact path is invalid")
    return {
        "path": relative.as_posix(),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _safe_summary_raw_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("raw path must be a nonempty relative string")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("raw path traversal is forbidden")
    path = (root / value).resolve()
    diagnostic_root = (root / "diagnostics").resolve()
    try:
        path.relative_to(diagnostic_root)
    except ValueError as error:
        raise ValueError("raw path must remain below diagnostics") from error
    return path


def _require_exact_array(actual: Any, expected: Any, name: str) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    _require(
        actual_array.dtype == expected_array.dtype
        and actual_array.shape == expected_array.shape
        and np.array_equal(actual_array, expected_array),
        f"{name} differs from deterministic regeneration",
    )


def _require_exact_arrays(
    actual: Mapping[str, np.ndarray],
    expected: Mapping[str, np.ndarray],
    name: str,
) -> None:
    _require(set(actual) == set(expected), f"{name} fields differ from regeneration")
    for field in sorted(expected):
        _require_exact_array(actual[field], expected[field], f"{name}.{field}")


def _diagnostic_directory(
    root: Path, diagnostic_name: str, plan: DiagnosticPlan
) -> Path:
    base = root / "diagnostics"
    if plan.smoke_nonclaim:
        base = base / "smoke"
    return base / diagnostic_name


def _frozen_compute_runtime_identity(
    root: Path,
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and coalesce frozen compute-plan runtime identities."""

    cells = matrix_cells(matrix, "compute_plan")
    _require(bool(cells), "diagnostics require frozen compute-plan runtime evidence")
    identities: list[dict[str, Any]] = []
    for cell in cells:
        result_path = stage_directory(root, cell) / "result.json"
        _require(result_path.is_file(), "diagnostic compute-plan result is missing")
        result = read_json(result_path)
        validate_compute_plan_cell(result, protocol, matrix, cell, root)
        identities.append(runtime_homogeneity_identity(result.get("runtime", {})))
    reference = identities[0]
    _require(
        all(identity == reference for identity in identities[1:]),
        "frozen compute-plan cells have heterogeneous runtime identities",
    )
    return reference


def _validate_runtime_against_frozen(
    runtime: Mapping[str, Any],
    reference: Mapping[str, Any],
    name: str,
) -> None:
    """Compare stable runtime fields while retaining scheduler ordinals."""

    _require(
        runtime_homogeneity_identity(runtime) == dict(reference),
        f"{name} runtime differs from the frozen compute-plan runtime",
    )


def _regenerated_diagnostic_inputs(
    root: Path,
    diagnostic_name: str,
    matrix: Mapping[str, Any],
    plan: DiagnosticPlan,
) -> dict[str, Any]:
    """Authenticate shared generated data and return the held-out oracle inputs."""

    episodes, conditions, draws, updates = _plan_counts(plan, diagnostic_name)
    directory = _diagnostic_directory(root, diagnostic_name, plan)
    dataset_path = directory / "training_dataset.npz"
    schedule_path = directory / "minibatch_schedule.npz"
    _require(dataset_path.is_file(), f"{diagnostic_name} training dataset is missing")
    _require(schedule_path.is_file(), f"{diagnostic_name} minibatch schedule is missing")
    train_seed = derive_seed(
        "matched-objective-diagnostic-train",
        matrix["matrix_sha256"],
        diagnostic_name,
    )
    schedule_seed = derive_seed(
        "matched-objective-diagnostic-minibatches",
        matrix["matrix_sha256"],
        diagnostic_name,
    )
    expected_dataset = generate_scalar_training_data(
        diagnostic_name,
        episodes=episodes,
        episode_length=plan.episode_length,
        seed=train_seed,
    )
    dataset = load_numeric_npz(dataset_path)
    _require_exact_arrays(
        dataset, expected_dataset, f"{diagnostic_name} training dataset"
    )
    expected_schedule = _batch_schedule(
        schedule_seed, updates, plan.batch_size, episodes
    )
    schedule = load_numeric_npz(schedule_path)
    _require_exact_arrays(
        schedule, expected_schedule, f"{diagnostic_name} minibatch schedule"
    )
    test_seed = derive_seed(
        "matched-objective-diagnostic-test",
        matrix["matrix_sha256"],
        diagnostic_name,
    )
    evaluation = generate_scalar_evaluation_data(
        diagnostic_name,
        conditions=conditions,
        episode_length=plan.episode_length,
        repeated_futures=draws,
        seed=test_seed,
        oracle_reference_futures=(
            plan.stochastic_oracle_reference_futures
            if diagnostic_name == STOCHASTIC_NAME
            else None
        ),
        evaluation_futures=(
            plan.stochastic_evaluation_futures
            if diagnostic_name == STOCHASTIC_NAME
            else None
        ),
    )
    return {
        "dataset": dataset,
        "dataset_path": dataset_path,
        "schedule_path": schedule_path,
        "evaluation": evaluation,
        "test_seed": test_seed,
        "model_noise_key": derive_jax_key(
            "matched-objective-diagnostic-model-noise",
            matrix["matrix_sha256"],
            diagnostic_name,
        ),
        "training_observation_std": max(
            float(np.std(dataset["observations"], ddof=0)), 1e-6
        ),
    }


def _validate_raw_provenance(
    raw: Mapping[str, np.ndarray],
    diagnostic_name: str,
    regenerated: Mapping[str, Any],
    plan: DiagnosticPlan,
) -> None:
    """Validate counts, seeds, and all non-model arrays bit for bit."""

    import jax

    expected_fields = (
        DETERMINISTIC_RAW_FIELDS
        if diagnostic_name == DETERMINISTIC_NAME
        else STOCHASTIC_RAW_FIELDS
    )
    _validate_numeric_raw(raw, expected_fields, diagnostic_name)
    _, conditions, draws, _ = _plan_counts(plan, diagnostic_name)
    retained = len(REGISTERED_RETAINED_HORIZONS)
    evaluation = regenerated["evaluation"]
    _require_exact_array(
        raw["horizons"],
        np.asarray(REGISTERED_RETAINED_HORIZONS, dtype=np.int32),
        f"{diagnostic_name}.horizons",
    )
    _require_exact_array(
        raw["initial_states"],
        evaluation["initial_states"],
        f"{diagnostic_name}.initial_states",
    )
    _require_exact_array(
        raw["future_actions"],
        evaluation["future_actions"][:, :retained],
        f"{diagnostic_name}.future_actions",
    )
    _require_exact_array(
        raw["training_observation_std"],
        np.asarray([regenerated["training_observation_std"]], dtype=np.float64),
        f"{diagnostic_name}.training_observation_std",
    )
    _require_exact_array(
        raw["test_seed"],
        np.asarray([regenerated["test_seed"]], dtype=np.uint32),
        f"{diagnostic_name}.test_seed",
    )
    _require_exact_array(
        raw["model_noise_key"],
        np.asarray(
            jax.random.key_data(regenerated["model_noise_key"]), dtype=np.uint32
        ),
        f"{diagnostic_name}.model_noise_key",
    )
    model = np.asarray(raw["model_futures"])
    if diagnostic_name == DETERMINISTIC_NAME:
        _require_exact_array(
            raw["target_futures"],
            evaluation["target_futures"][:, :retained],
            f"{diagnostic_name}.target_futures",
        )
        _require_exact_array(
            raw["ratio_epsilon"],
            np.asarray([RATIO_EPSILON], dtype=np.float64),
            f"{diagnostic_name}.ratio_epsilon",
        )
        _require(
            model.dtype == np.dtype(np.float32)
            and model.shape == (conditions, retained),
            "deterministic model futures must be float32 [512, 30] in a registered run",
        )
        return
    _require_exact_array(
        raw["score_horizons"],
        np.asarray(ENERGY_HORIZONS, dtype=np.int32),
        f"{diagnostic_name}.score_horizons",
    )
    _require_exact_array(
        raw["rollout_horizons"],
        np.asarray(REGISTERED_ROLLOUT_HORIZONS, dtype=np.int32),
        f"{diagnostic_name}.rollout_horizons",
    )
    _require_exact_array(
        raw["oracle_reference_futures"],
        evaluation["oracle_reference_futures"][:, :, :retained],
        f"{diagnostic_name}.oracle_reference_futures",
    )
    _require_exact_array(
        raw["evaluation_futures"],
        evaluation["evaluation_futures"][:, :, :retained],
        f"{diagnostic_name}.evaluation_futures",
    )
    _require_exact_array(
        raw["evaluation_branches"],
        evaluation["evaluation_branches"][:, :, :retained],
        f"{diagnostic_name}.evaluation_branches",
    )
    _require_exact_array(
        raw["ece_bin_edges"],
        np.linspace(0.0, 1.0, ECE_BINS + 1, dtype=np.float64),
        f"{diagnostic_name}.ece_bin_edges",
    )
    _require(
        model.dtype == np.dtype(np.float32)
        and model.shape == (conditions, draws, retained),
        "stochastic model futures must be float32 [512, 64, 30] in a registered run",
    )


def _checkpoint_update_schedule(
    plan: DiagnosticPlan, total_updates: int
) -> list[int]:
    """Return every update at which the diagnostic runner may checkpoint."""

    _require(
        isinstance(total_updates, int)
        and not isinstance(total_updates, bool)
        and total_updates > 0,
        "diagnostic total updates must be a positive integer",
    )
    expected = list(
        range(plan.checkpoint_every, total_updates + 1, plan.checkpoint_every)
    )
    if not expected or expected[-1] != total_updates:
        expected.append(total_updates)
    return expected


def _validate_checkpoint_trace(
    trace: Any,
    *,
    completed_updates: int,
    total_updates: int,
    plan: DiagnosticPlan,
) -> list[Mapping[str, Any]]:
    """Require the exact scheduled trace prefix and fixed metric schema."""

    _require(isinstance(trace, list) and trace, "diagnostic checkpoint trace is empty")
    expected_updates = [
        update
        for update in _checkpoint_update_schedule(plan, total_updates)
        if update <= completed_updates
    ]
    _require(
        bool(expected_updates) and expected_updates[-1] == completed_updates,
        "diagnostic checkpoint completed update is not on the checkpoint schedule",
    )
    observed_updates: list[int] = []
    expected_fields = {"update", *CHECKPOINT_TRACE_METRIC_FIELDS}
    for row in trace:
        _require(isinstance(row, Mapping), "diagnostic checkpoint trace row is malformed")
        _require(
            set(row) == expected_fields,
            "diagnostic checkpoint trace metric schema mismatch",
        )
        update = row["update"]
        _require(
            isinstance(update, int) and not isinstance(update, bool),
            "diagnostic checkpoint update must be an integer",
        )
        observed_updates.append(update)
        for key in CHECKPOINT_TRACE_METRIC_FIELDS:
            _finite_scalar(row[key], f"diagnostic checkpoint metric {key}")
    _require(
        observed_updates == expected_updates,
        "diagnostic checkpoint trace is not the canonical update prefix",
    )
    return trace


def _optimizer_step(state: Any, optimizer_name: str) -> int:
    """Read one scalar integer optimizer counter from a loaded AgentState."""

    optimizer = getattr(state, optimizer_name, None)
    step = getattr(optimizer, "step", None)
    array = np.asarray(step)
    _require(
        array.shape == () and np.issubdtype(array.dtype, np.integer),
        f"diagnostic {optimizer_name} step must be a scalar integer",
    )
    return int(array.item())


def _validate_loaded_diagnostic_checkpoint(
    state: Any,
    metadata: Any,
    *,
    identity_sha256: str,
    expected_completed_updates: int | None,
    total_updates: int,
    plan: DiagnosticPlan,
    expected_trace: Any | None = None,
    expected_wall_seconds: Any | None = None,
) -> tuple[int, float, list[Mapping[str, Any]]]:
    """Bind optimizer state and canonical progress metadata to one checkpoint."""

    _require(isinstance(metadata, Mapping), "diagnostic checkpoint metadata is malformed")
    _require(
        set(metadata) == set(CHECKPOINT_METADATA_FIELDS),
        "diagnostic checkpoint metadata schema mismatch",
    )
    _require(
        metadata["stage"] == CHECKPOINT_STAGE,
        "diagnostic checkpoint stage mismatch",
    )
    _require(
        metadata["identity_sha256"] == identity_sha256,
        "diagnostic checkpoint identity mismatch",
    )
    completed = metadata["completed_updates"]
    _require(
        isinstance(completed, int) and not isinstance(completed, bool),
        "diagnostic checkpoint completed_updates must be an integer",
    )
    if expected_completed_updates is not None:
        _require(
            completed == expected_completed_updates,
            "diagnostic checkpoint completed update mismatch",
        )
    _require(
        0 < completed <= total_updates,
        "diagnostic checkpoint completed update is outside the training budget",
    )
    trace = _validate_checkpoint_trace(
        metadata["checkpoint_trace"],
        completed_updates=completed,
        total_updates=total_updates,
        plan=plan,
    )
    _require(
        _optimizer_step(state, "model_optimizer") == completed,
        "diagnostic model optimizer step does not equal completed_updates",
    )
    _require(
        _optimizer_step(state, "actor_optimizer") == 0,
        "diagnostic actor optimizer step must remain zero",
    )
    _require(
        _optimizer_step(state, "critic_optimizer") == 0,
        "diagnostic critic optimizer step must remain zero",
    )
    wall_seconds = _finite_scalar(
        metadata["wall_seconds"], "diagnostic checkpoint wall seconds"
    )
    _require(
        wall_seconds >= 0.0,
        "diagnostic checkpoint wall seconds must be nonnegative",
    )
    if expected_trace is not None:
        _require(
            canonical_bytes(trace) == canonical_bytes(expected_trace),
            "diagnostic checkpoint trace differs from the training manifest",
        )
    if expected_wall_seconds is not None:
        manifest_wall = _finite_scalar(
            expected_wall_seconds, "diagnostic manifest wall seconds"
        )
        _require(
            wall_seconds == manifest_wall,
            "diagnostic checkpoint wall time differs from the training manifest",
        )
    return completed, wall_seconds, trace


def _validate_training_manifest(
    root: Path,
    diagnostic_name: str,
    arm: str,
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    plan: DiagnosticPlan,
    regenerated: Mapping[str, Any],
    frozen_runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the selected config and the dataset/schedule/checkpoint chain."""

    _, _, _, updates = _plan_counts(plan, diagnostic_name)
    directory = _diagnostic_directory(root, diagnostic_name, plan) / arm
    manifest_path = directory / "training_manifest.json"
    checkpoint_path = directory / "checkpoint.pkl"
    _require(manifest_path.is_file(), f"{diagnostic_name}/{arm} training manifest missing")
    _require(checkpoint_path.is_file(), f"{diagnostic_name}/{arm} checkpoint missing")
    _require(checkpoint_path.stat().st_size > 0, "diagnostic checkpoint is empty")
    manifest = read_json(manifest_path)
    _require(
        set(manifest) == set(TRAINING_MANIFEST_FIELDS),
        f"{diagnostic_name}/{arm} training manifest schema mismatch",
    )
    candidate = _selected_candidate(matrix, arm)
    config = make_config(
        protocol,
        "confirmatory",
        arm,
        (1,),
        1,
        candidate=candidate,
    )
    config_payload = asdict(config)
    identity = {
        "diagnostic_name": diagnostic_name,
        "arm": arm,
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "selection_sha256": matrix["selection_sha256"],
        "smoke_nonclaim": plan.smoke_nonclaim,
        "effective_updates": updates,
        "effective_config_sha256": object_sha256(config_payload),
        "dataset_file_sha256": file_sha256(regenerated["dataset_path"]),
        "schedule_file_sha256": file_sha256(regenerated["schedule_path"]),
    }
    _require(manifest["schema_version"] == TRAINING_MANIFEST_SCHEMA,
             "diagnostic training schema mismatch")
    _require(manifest["status"] == plan.status, "diagnostic training status mismatch")
    for key, value in identity.items():
        _require(
            canonical_bytes(manifest[key]) == canonical_bytes(value),
            f"diagnostic training identity {key} mismatch",
        )
    _require(
        manifest["identity_sha256"] == object_sha256(identity),
        "diagnostic training identity digest mismatch",
    )
    _require(
        canonical_bytes(manifest["selected_candidate"]) == canonical_bytes(candidate),
        "diagnostic selected candidate mismatch",
    )
    _require(
        canonical_bytes(manifest["effective_config"])
        == canonical_bytes(config_payload),
        "diagnostic selected config mismatch",
    )
    _require(
        manifest["checkpoint_sha256"] == file_sha256(checkpoint_path),
        "diagnostic checkpoint digest mismatch",
    )
    _validate_runtime_against_frozen(
        manifest["runtime"],
        frozen_runtime_identity,
        f"{diagnostic_name}/{arm} recorded training",
    )
    wall_seconds = _finite_scalar(manifest["wall_seconds"], "diagnostic wall seconds")
    _require(wall_seconds >= 0.0, "diagnostic wall seconds must be nonnegative")
    _validate_checkpoint_trace(
        manifest["checkpoint_trace"],
        completed_updates=updates,
        total_updates=updates,
        plan=plan,
    )
    _require(
        manifest["training_manifest_sha256"]
        == object_sha256(_without_digest(manifest, "training_manifest_sha256")),
        "diagnostic training manifest digest mismatch",
    )
    return manifest


def _validate_model_futures_from_checkpoint(
    root: Path,
    diagnostic_name: str,
    arm: str,
    raw: Mapping[str, np.ndarray],
    training_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    plan: DiagnosticPlan,
    regenerated: Mapping[str, Any],
) -> None:
    """Re-run the canonical diagnostic inference from its bound checkpoint.

    This is intentionally part of claim verification rather than an optional
    audit.  The registered runtime identity has already been checked, so exact
    equality is the appropriate invariant: the runner itself also requires an
    interrupted evaluation to regenerate a bitwise-identical NPZ payload.
    """

    from imf_dreamer_jax import load_checkpoint

    checkpoint_path = (
        _diagnostic_directory(root, diagnostic_name, plan) / arm / "checkpoint.pkl"
    )
    candidate = _selected_candidate(matrix, arm)
    expected_config = make_config(
        protocol,
        "confirmatory",
        arm,
        (1,),
        1,
        candidate=candidate,
    )
    state, stored_config, metadata = load_checkpoint(checkpoint_path)
    _require(
        stored_config == expected_config,
        f"{diagnostic_name}/{arm} checkpoint config mismatch",
    )
    _validate_loaded_diagnostic_checkpoint(
        state,
        metadata,
        identity_sha256=training_manifest["identity_sha256"],
        expected_completed_updates=training_manifest["effective_updates"],
        total_updates=training_manifest["effective_updates"],
        plan=plan,
        expected_trace=training_manifest["checkpoint_trace"],
        expected_wall_seconds=training_manifest["wall_seconds"],
    )
    _, _, draws, _ = _plan_counts(plan, diagnostic_name)
    replayed = _evaluate_model(
        state,
        expected_config,
        diagnostic_name,
        regenerated["evaluation"],
        draws=draws,
        training_observation_std=regenerated["training_observation_std"],
        model_noise_key=regenerated["model_noise_key"],
        test_seed=regenerated["test_seed"],
    )
    _require_exact_arrays(
        raw,
        replayed,
        f"{diagnostic_name}/{arm}.checkpoint_replay",
    )


def _validate_arm_result_manifest(
    root: Path,
    diagnostic_name: str,
    arm: str,
    raw_path: Path,
    training_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: DiagnosticPlan,
    frozen_runtime_identity: Mapping[str, Any],
) -> None:
    result_path = _diagnostic_directory(root, diagnostic_name, plan) / arm / "result_manifest.json"
    _require(result_path.is_file(), f"{diagnostic_name}/{arm} result manifest missing")
    observed = read_json(result_path)
    runtime = observed.get("runtime", {})
    _validate_runtime_against_frozen(
        runtime,
        frozen_runtime_identity,
        f"{diagnostic_name}/{arm} recorded evaluation",
    )
    expected = _arm_result_manifest(
        root,
        diagnostic_name,
        arm,
        raw_path,
        training_manifest,
        protocol,
        plan,
        runtime,
    )
    _require(
        canonical_bytes(observed) == canonical_bytes(expected),
        f"{diagnostic_name}/{arm} result manifest does not recompute",
    )


def build_diagnostic_summary(
    root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    raw_paths: Mapping[str, Mapping[str, Path]],
    *,
    smoke_nonclaim: bool,
) -> dict[str, Any]:
    """Build the validator schema exclusively by recomputing retained NPZ."""

    output_root = Path(root)
    diagnostics: dict[str, Any] = {}
    for diagnostic_name in DIAGNOSTIC_NAMES:
        spec = protocol["diagnostics"][diagnostic_name]
        arms: dict[str, Any] = {}
        for arm in ARM_ORDER:
            raw_path = Path(raw_paths[diagnostic_name][arm])
            raw = load_numeric_npz(raw_path)
            estimands, _, passed = recompute_diagnostic_raw(diagnostic_name, raw)
            arms[arm] = {
                "estimands": estimands,
                "thresholds": dict(spec["interpretation_thresholds"]),
                "passed": bool(passed),
                "raw_files": [_raw_file_entry(output_root, raw_path)],
            }
        diagnostics[diagnostic_name] = {
            "status": "smoke_nonclaim" if smoke_nonclaim else "complete",
            "registered_spec_sha256": object_sha256(spec),
            "interpretation_only": True,
            "arms": arms,
        }
    summary = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": "smoke_nonclaim" if smoke_nonclaim else "complete",
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "selection_sha256": matrix["selection_sha256"],
        "diagnostics": diagnostics,
    }
    summary["diagnostic_summary_sha256"] = object_sha256(summary)
    return summary


def validate_diagnostic_summary_from_raw(
    summary: Mapping[str, Any],
    root: str | Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    smoke_nonclaim: bool = False,
    plan: DiagnosticPlan | None = None,
) -> None:
    """Recompute raw estimands and authenticate their complete artifact chain.

    Model futures are regenerated from the selected-config checkpoint and the
    frozen deterministic evaluation keys.  Exact equality is required in
    addition to recomputing every estimand and authenticating the artifact
    manifests.
    """

    required = {
        "schema_version",
        "status",
        "protocol_sha256",
        "source_sha256",
        "matrix_sha256",
        "selection_sha256",
        "diagnostics",
        "diagnostic_summary_sha256",
    }
    _require(set(summary) == required, "diagnostic summary keys differ from schema")
    effective_plan = plan or resolve_diagnostic_plan(protocol, smoke=smoke_nonclaim)
    _require(
        effective_plan.smoke_nonclaim is smoke_nonclaim,
        "diagnostic plan claim boundary mismatch",
    )
    expected_status = effective_plan.status
    _require(summary["schema_version"] == DIAGNOSTIC_SCHEMA, "diagnostic schema mismatch")
    _require(summary["status"] == expected_status, "diagnostic claim status mismatch")
    _require(
        matrix["protocol_sha256"] == protocol_digest(protocol),
        "diagnostic protocol identity mismatch",
    )
    if not smoke_nonclaim:
        _require(
            matrix.get("claim_eligible") is True,
            "registered diagnostic summary requires a claim-eligible matrix",
        )
    for key in ("protocol_sha256", "source_sha256", "matrix_sha256", "selection_sha256"):
        _require(summary[key] == matrix[key], f"diagnostic {key} identity mismatch")
    _require(
        summary["diagnostic_summary_sha256"]
        == object_sha256(_without_digest(summary, "diagnostic_summary_sha256")),
        "diagnostic summary digest mismatch",
    )
    _require(set(summary["diagnostics"]) == set(DIAGNOSTIC_NAMES),
             "diagnostic set is incomplete")
    output_root = Path(root).resolve()
    frozen_runtime_identity = _frozen_compute_runtime_identity(
        output_root, protocol, matrix
    )
    seen_raw_paths: set[Path] = set()
    for diagnostic_name in DIAGNOSTIC_NAMES:
        row = summary["diagnostics"][diagnostic_name]
        spec = protocol["diagnostics"][diagnostic_name]
        regenerated = _regenerated_diagnostic_inputs(
            output_root, diagnostic_name, matrix, effective_plan
        )
        _require(
            set(row) == {"status", "registered_spec_sha256", "interpretation_only", "arms"},
            f"{diagnostic_name} schema mismatch",
        )
        _require(row["status"] == expected_status, f"{diagnostic_name} status mismatch")
        _require(row["registered_spec_sha256"] == object_sha256(spec),
                 f"{diagnostic_name} specification digest mismatch")
        _require(row["interpretation_only"] is True,
                 f"{diagnostic_name} must remain interpretation-only")
        _require(set(row["arms"]) == set(ARM_ORDER), f"{diagnostic_name} arms incomplete")
        for arm in ARM_ORDER:
            arm_row = row["arms"][arm]
            _require(set(arm_row) == {"estimands", "thresholds", "passed", "raw_files"},
                     f"{diagnostic_name}/{arm} schema mismatch")
            _require(arm_row["thresholds"] == spec["interpretation_thresholds"],
                     f"{diagnostic_name}/{arm} thresholds differ from protocol")
            files = arm_row["raw_files"]
            _require(isinstance(files, list) and len(files) == 1,
                     f"{diagnostic_name}/{arm} must bind exactly one canonical raw file")
            entry = files[0]
            _require(set(entry) == {"path", "size", "sha256"}, "raw-file entry malformed")
            path = _safe_summary_raw_path(output_root, entry["path"])
            expected_path = (
                _diagnostic_directory(output_root, diagnostic_name, effective_plan)
                / arm
                / "evaluation_raw.npz"
            ).resolve()
            _require(
                path == expected_path,
                f"{diagnostic_name}/{arm} raw path is not canonical",
            )
            _require(path not in seen_raw_paths, "diagnostic raw paths must be unique")
            seen_raw_paths.add(path)
            _require(path.is_file() and path.suffix == ".npz", "raw NPZ is missing")
            _require(
                entry == _raw_file_entry(output_root, path),
                "raw NPZ file identity mismatch",
            )
            raw = load_numeric_npz(path)
            _validate_raw_provenance(
                raw, diagnostic_name, regenerated, effective_plan
            )
            estimands, _, passed = recompute_diagnostic_raw(
                diagnostic_name, raw
            )
            _require(canonical_bytes(arm_row["estimands"]) == canonical_bytes(estimands),
                     f"{diagnostic_name}/{arm} estimands do not recompute")
            _require(
                isinstance(arm_row["passed"], bool)
                and arm_row["passed"] == bool(passed),
                f"{diagnostic_name}/{arm} decision does not recompute",
            )
            training_manifest = _validate_training_manifest(
                output_root,
                diagnostic_name,
                arm,
                protocol,
                matrix,
                effective_plan,
                regenerated,
                frozen_runtime_identity,
            )
            _validate_model_futures_from_checkpoint(
                output_root,
                diagnostic_name,
                arm,
                raw,
                training_manifest,
                protocol,
                matrix,
                effective_plan,
                regenerated,
            )
            _validate_arm_result_manifest(
                output_root,
                diagnostic_name,
                arm,
                path,
                training_manifest,
                protocol,
                effective_plan,
                frozen_runtime_identity,
            )
    _require(
        len(seen_raw_paths) == len(DIAGNOSTIC_NAMES) * len(ARM_ORDER),
        "diagnostic raw evidence is incomplete",
    )


def _selected_candidate(matrix: Mapping[str, Any], arm: str) -> dict[str, Any]:
    selection = matrix.get("selection_manifest")
    if not isinstance(selection, Mapping) or not isinstance(selection.get("selected"), Mapping):
        raise ValueError("diagnostics require the frozen confirmatory HPO selection")
    row = selection["selected"].get(arm)
    if not isinstance(row, Mapping):
        raise ValueError(f"selected confirmatory candidate is missing for {arm}")
    return {"candidate_id": row.get("candidate_id"), "overrides": row.get("overrides")}


def _plan_counts(plan: DiagnosticPlan, diagnostic_name: str) -> tuple[int, int, int, int]:
    if diagnostic_name == DETERMINISTIC_NAME:
        return (
            plan.deterministic_training_episodes,
            plan.deterministic_test_conditions,
            plan.deterministic_repeated_futures,
            plan.deterministic_updates,
        )
    return (
        plan.stochastic_training_episodes,
        plan.stochastic_test_conditions,
        plan.stochastic_predictive_draws,
        plan.stochastic_updates,
    )


def _ensure_npz_generated(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    if path.exists():
        stored = load_numeric_npz(path)
        if array_sha256(stored) != array_sha256(arrays):
            raise ValueError(f"existing deterministic artifact differs: {path}")
        return path
    return _write_npz_once(path, arrays)


def _batch_schedule(
    seed: int, updates: int, batch_size: int, episodes: int
) -> dict[str, np.ndarray]:
    random = np.random.default_rng(seed)
    return {
        "episode_indices": random.integers(
            0, episodes, size=(updates, batch_size), dtype=np.int32
        ),
        "schedule_seed": np.asarray([seed], dtype=np.uint32),
    }


def _batch_from_dataset(dataset: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
    import jax.numpy as jnp

    return {
        name: jnp.asarray(dataset[name][indices])
        for name in ("observations", "actions", "rewards", "continuations", "is_first")
    }


def _metrics_dict(metrics: Any) -> dict[str, float]:
    values = metrics._asdict() if hasattr(metrics, "_asdict") else dict(metrics)
    result = {name: float(np.asarray(value)) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("world-model training produced non-finite metrics")
    return result


def _train_world_model_arm(
    root: Path,
    diagnostic_root: Path,
    diagnostic_name: str,
    arm: str,
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    plan: DiagnosticPlan,
    dataset_path: Path,
    schedule_path: Path,
    execution_runtime: Mapping[str, Any],
    frozen_runtime_identity: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Train or resume one selected arm without replacing completed evidence."""

    import jax
    from imf_dreamer_jax import (
        create_agent,
        jit_train_world_model,
        load_checkpoint,
        save_checkpoint,
    )

    _validate_runtime_against_frozen(
        execution_runtime,
        frozen_runtime_identity,
        f"{diagnostic_name}/{arm} worker",
    )
    _, _, _, updates = _plan_counts(plan, diagnostic_name)
    candidate = _selected_candidate(matrix, arm)
    config = make_config(
        protocol,
        "confirmatory",
        arm,
        (1,),
        1,
        candidate=candidate,
    )
    config_digest = object_sha256(asdict(config))
    arm_root = diagnostic_root / diagnostic_name / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_root / "checkpoint.pkl"
    training_manifest_path = arm_root / "training_manifest.json"
    dataset_digest = file_sha256(dataset_path)
    schedule_digest = file_sha256(schedule_path)
    identity = {
        "diagnostic_name": diagnostic_name,
        "arm": arm,
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "selection_sha256": matrix["selection_sha256"],
        "smoke_nonclaim": plan.smoke_nonclaim,
        "effective_updates": updates,
        "effective_config_sha256": config_digest,
        "dataset_file_sha256": dataset_digest,
        "schedule_file_sha256": schedule_digest,
    }
    identity_sha = object_sha256(identity)

    if training_manifest_path.exists():
        manifest = read_json(training_manifest_path)
        if (
            set(manifest) != set(TRAINING_MANIFEST_FIELDS)
            or manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA
            or manifest.get("status") != plan.status
            or manifest.get("identity_sha256") != identity_sha
            or any(
                canonical_bytes(manifest.get(key)) != canonical_bytes(value)
                for key, value in identity.items()
            )
            or canonical_bytes(manifest.get("selected_candidate"))
            != canonical_bytes(candidate)
            or canonical_bytes(manifest.get("effective_config"))
            != canonical_bytes(asdict(config))
            or manifest.get("checkpoint_sha256") != file_sha256(checkpoint_path)
            or manifest.get("training_manifest_sha256")
            != object_sha256(_without_digest(manifest, "training_manifest_sha256"))
        ):
            raise ValueError("completed diagnostic training manifest is invalid")
        _validate_runtime_against_frozen(
            manifest["runtime"],
            frozen_runtime_identity,
            f"{diagnostic_name}/{arm} recorded training",
        )
        state, stored_config, metadata = load_checkpoint(checkpoint_path)
        if stored_config != config:
            raise ValueError("completed diagnostic checkpoint identity mismatch")
        _validate_loaded_diagnostic_checkpoint(
            state,
            metadata,
            identity_sha256=identity_sha,
            expected_completed_updates=updates,
            total_updates=updates,
            plan=plan,
            expected_trace=manifest["checkpoint_trace"],
            expected_wall_seconds=manifest["wall_seconds"],
        )
        return state, config, manifest

    initialization_key = derive_jax_key(
        "diagnostic-world-init", matrix["matrix_sha256"], diagnostic_name
    )
    start_update = 0
    accumulated_wall = 0.0
    trace: list[dict[str, float | int]] = []
    if checkpoint_path.exists():
        state, stored_config, metadata = load_checkpoint(checkpoint_path)
        if stored_config != config:
            raise ValueError("partial diagnostic checkpoint identity mismatch")
        start_update, accumulated_wall, validated_trace = (
            _validate_loaded_diagnostic_checkpoint(
                state,
                metadata,
                identity_sha256=identity_sha,
                expected_completed_updates=None,
                total_updates=updates,
                plan=plan,
            )
        )
        trace = [dict(row) for row in validated_trace]
    else:
        state = create_agent(config, initialization_key)
    checkpoint_wall = accumulated_wall
    dataset = load_numeric_npz(dataset_path)
    schedule = load_numeric_npz(schedule_path)
    indices = np.asarray(schedule["episode_indices"], dtype=np.int32)
    if indices.shape != (updates, plan.batch_size):
        raise ValueError("diagnostic minibatch schedule shape mismatch")
    objective_key = derive_jax_key(
        "diagnostic-world-objective", matrix["matrix_sha256"], diagnostic_name
    )
    started = time.perf_counter()
    for update in range(start_update, updates):
        batch = _batch_from_dataset(dataset, indices[update])
        state, metrics = jit_train_world_model(
            state, batch, jax.random.fold_in(objective_key, update), config
        )
        completed = update + 1
        if completed % plan.checkpoint_every == 0 or completed == updates:
            jax.block_until_ready(state)
            row = {"update": completed, **_metrics_dict(metrics)}
            trace.append(row)
            wall = accumulated_wall + time.perf_counter() - started
            checkpoint_wall = wall
            save_checkpoint(
                checkpoint_path,
                state,
                config,
                metadata={
                    "stage": CHECKPOINT_STAGE,
                    "identity_sha256": identity_sha,
                    "completed_updates": completed,
                    "wall_seconds": wall,
                    "checkpoint_trace": trace,
                },
            )
    if updates == 0 or not checkpoint_path.is_file():
        raise RuntimeError("diagnostic training produced no checkpoint")
    manifest = {
        "schema_version": TRAINING_MANIFEST_SCHEMA,
        "status": plan.status,
        **identity,
        "identity_sha256": identity_sha,
        "selected_candidate": candidate,
        "effective_config": asdict(config),
        "checkpoint_trace": trace,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "wall_seconds": checkpoint_wall,
        "runtime": dict(execution_runtime),
    }
    manifest["training_manifest_sha256"] = object_sha256(manifest)
    _write_json_once(training_manifest_path, manifest)
    return state, config, manifest


def _evaluate_model(
    state: Any,
    config: Any,
    diagnostic_name: str,
    evaluation: Mapping[str, np.ndarray],
    *,
    draws: int,
    training_observation_std: float,
    model_noise_key: Any,
    test_seed: int,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    initial = np.asarray(evaluation["initial_states"], dtype=np.float32)
    future_actions = np.asarray(evaluation["future_actions"], dtype=np.float32)
    conditions, horizon = future_actions.shape
    context_observation = jnp.asarray(initial[:, None, None])
    context_action = jnp.zeros((conditions, 1, 1), dtype=jnp.float32)
    start = posterior_mean_filter(
        state.params.world_model, context_observation, context_action, config
    )
    noise = jax.random.normal(
        model_noise_key,
        (draws, horizon, conditions, config.stochastic_dim),
        dtype=jnp.float32,
    )
    sampler = jax.jit(open_loop_samples_with_continuation, static_argnames=("config",))
    model_observations, _, _ = sampler(
        state.params.world_model,
        start,
        jnp.asarray(future_actions[..., None]),
        noise,
        config,
    )
    model = np.asarray(model_observations[..., 0], dtype=np.float32)
    # Shared sampler layout is [draws, conditions, horizon].
    model = np.transpose(model, (1, 0, 2))
    if not np.isfinite(model).all():
        raise FloatingPointError("diagnostic model generated non-finite futures")
    key_data = np.asarray(jax.random.key_data(model_noise_key), dtype=np.uint32)
    retained_horizon = max(ENERGY_HORIZONS)
    _require(horizon >= retained_horizon, "evaluation trajectory is shorter than horizon 30")
    future_actions = future_actions[:, :retained_horizon]
    model = model[:, :, :retained_horizon]
    common = {
        "horizons": np.arange(1, retained_horizon + 1, dtype=np.int32),
        "initial_states": initial,
        "future_actions": future_actions,
        "training_observation_std": np.asarray(
            [max(float(training_observation_std), 1e-6)], dtype=np.float64
        ),
        "test_seed": np.asarray([test_seed], dtype=np.uint32),
        "model_noise_key": key_data,
    }
    if diagnostic_name == DETERMINISTIC_NAME:
        return {
            **common,
            "target_futures": np.asarray(
                evaluation["target_futures"][:, :retained_horizon], dtype=np.float32
            ),
            "model_futures": model[:, 0],
            "ratio_epsilon": np.asarray([RATIO_EPSILON], dtype=np.float64),
        }
    return {
        **common,
        "score_horizons": np.asarray(ENERGY_HORIZONS, dtype=np.int32),
        "rollout_horizons": np.asarray((1, 2, 4, 8, 15, 30), dtype=np.int32),
        "oracle_reference_futures": np.asarray(
            evaluation["oracle_reference_futures"][:, :, :retained_horizon],
            dtype=np.float32,
        ),
        "evaluation_futures": np.asarray(
            evaluation["evaluation_futures"][:, :, :retained_horizon],
            dtype=np.float32,
        ),
        "evaluation_branches": np.asarray(
            evaluation["evaluation_branches"][:, :, :retained_horizon], dtype=np.uint8
        ),
        "model_futures": model,
        "ece_bin_edges": np.linspace(0.0, 1.0, ECE_BINS + 1, dtype=np.float64),
    }


def _arm_result_manifest(
    root: Path,
    diagnostic_name: str,
    arm: str,
    raw_path: Path,
    training_manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: DiagnosticPlan,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    estimands, threshold_decisions, passed = recompute_diagnostic_raw(
        diagnostic_name, load_numeric_npz(raw_path)
    )
    result = {
        "schema_version": ARM_RESULT_SCHEMA,
        "status": plan.status,
        "diagnostic_name": diagnostic_name,
        "arm": arm,
        "smoke_nonclaim": plan.smoke_nonclaim,
        "interpretation_only": True,
        "registered_spec_sha256": object_sha256(protocol["diagnostics"][diagnostic_name]),
        "training_manifest_sha256": training_manifest["training_manifest_sha256"],
        "runtime": dict(runtime),
        "raw_array_sha256": array_sha256(load_numeric_npz(raw_path)),
        "raw_file": _raw_file_entry(root, raw_path),
        "estimands": estimands,
        "threshold_decisions": threshold_decisions,
        "passed": bool(passed),
        "aggregation_definitions": {
            "rollout_error": "per_horizon_predictive_mean_RMSE_divided_by_training_state_std",
            "rollout_auc": "normalized_trapezoid_over_registered_horizons_1_2_4_8_15_30",
            "energy": (
                "model_to_independent_oracle_reference_empirical_"
                "energy_score_ratio_per_horizon"
            ),
            "coverage": (
                "independent_evaluation_futures_inside_model_"
                "central_90_percent_interval_per_horizon"
            ),
            "branch_ece": "ten_equal_width_bins_of_first_transition_inferred_branch_probability",
        },
    }
    result["arm_result_sha256"] = object_sha256(result)
    return result


@contextmanager
def _exclusive_runner_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"diagnostic runner lock exists; inspect concurrent/stale run: {path}"
        ) from error
    try:
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    try:
        yield
    finally:
        if path.exists():
            path.unlink()


def _validate_frozen_root(
    root: Path, workspace: str | Path | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = read_json(root / "frozen_protocol.json")
    source = read_json(root / "source_manifest.json")
    matrix = read_json(root / "matrix.json")
    validate_matched_objective_protocol(protocol)
    validate_source_manifest(source, workspace)
    validate_matrix(matrix, protocol, source)
    _require(matrix.get("profile") == "confirmatory" and matrix.get("claim_eligible") is True,
             "registered diagnostics require a frozen claim-eligible confirmatory matrix")
    _require(matrix.get("protocol_sha256") == protocol_digest(protocol),
             "matrix protocol digest mismatch")
    _require(isinstance(matrix.get("selection_sha256"), str),
             "confirmatory selection identity is missing")
    for arm in ARM_ORDER:
        _selected_candidate(matrix, arm)
    return protocol, source, matrix


def run_registered_diagnostics(
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
    smoke: bool = False,
    smoke_updates: int = 2,
    smoke_training_episodes: int = 16,
    smoke_test_conditions: int = 8,
    smoke_repeated_futures: int = 8,
    smoke_batch_size: int = 4,
) -> dict[str, Any]:
    """Run/resume both registered diagnostics for both objective arms."""

    root = Path(output_root).resolve()
    protocol, _, matrix = _validate_frozen_root(root, workspace)
    plan = resolve_diagnostic_plan(
        protocol,
        smoke=smoke,
        smoke_updates=smoke_updates,
        smoke_training_episodes=smoke_training_episodes,
        smoke_test_conditions=smoke_test_conditions,
        smoke_repeated_futures=smoke_repeated_futures,
        smoke_batch_size=smoke_batch_size,
    )
    diagnostic_root = root / "diagnostics" / "smoke" if smoke else root / "diagnostics"
    summary_path = diagnostic_root / "diagnostic_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        validate_diagnostic_summary_from_raw(
            summary,
            root,
            matrix,
            protocol,
            smoke_nonclaim=smoke,
            plan=plan,
        )
        return summary

    frozen_runtime_identity = _frozen_compute_runtime_identity(root, protocol, matrix)
    execution_runtime = runtime_fingerprint()
    _validate_runtime_against_frozen(
        execution_runtime,
        frozen_runtime_identity,
        "diagnostic worker",
    )
    raw_paths: dict[str, dict[str, Path]] = {name: {} for name in DIAGNOSTIC_NAMES}
    with _exclusive_runner_lock(diagnostic_root / ".runner.lock"):
        for diagnostic_name in DIAGNOSTIC_NAMES:
            episodes, conditions, draws, updates = _plan_counts(plan, diagnostic_name)
            diagnostic_directory = diagnostic_root / diagnostic_name
            train_seed = derive_seed(
                "matched-objective-diagnostic-train",
                matrix["matrix_sha256"],
                diagnostic_name,
            )
            test_seed = derive_seed(
                "matched-objective-diagnostic-test",
                matrix["matrix_sha256"],
                diagnostic_name,
            )
            schedule_seed = derive_seed(
                "matched-objective-diagnostic-minibatches",
                matrix["matrix_sha256"],
                diagnostic_name,
            )
            dataset = generate_scalar_training_data(
                diagnostic_name,
                episodes=episodes,
                episode_length=plan.episode_length,
                seed=train_seed,
            )
            dataset_path = _ensure_npz_generated(
                diagnostic_directory / "training_dataset.npz", dataset
            )
            schedule = _batch_schedule(schedule_seed, updates, plan.batch_size, episodes)
            schedule_path = _ensure_npz_generated(
                diagnostic_directory / "minibatch_schedule.npz", schedule
            )
            evaluation = generate_scalar_evaluation_data(
                diagnostic_name,
                conditions=conditions,
                episode_length=plan.episode_length,
                repeated_futures=draws,
                seed=test_seed,
                oracle_reference_futures=(
                    plan.stochastic_oracle_reference_futures
                    if diagnostic_name == STOCHASTIC_NAME
                    else None
                ),
                evaluation_futures=(
                    plan.stochastic_evaluation_futures
                    if diagnostic_name == STOCHASTIC_NAME
                    else None
                ),
            )
            training_std = float(np.std(dataset["observations"], ddof=0))
            for arm in ARM_ORDER:
                arm_directory = diagnostic_directory / arm
                result_path = arm_directory / "result_manifest.json"
                raw_path = arm_directory / "evaluation_raw.npz"
                if result_path.exists():
                    result = read_json(result_path)
                    if (
                        result.get("schema_version") != ARM_RESULT_SCHEMA
                        or result.get("status") != plan.status
                        or result.get("diagnostic_name") != diagnostic_name
                        or result.get("arm") != arm
                        or result.get("arm_result_sha256")
                        != object_sha256(_without_digest(result, "arm_result_sha256"))
                    ):
                        raise ValueError("completed diagnostic arm manifest is invalid")
                    _, _, training_manifest = _train_world_model_arm(
                        root,
                        diagnostic_root,
                        diagnostic_name,
                        arm,
                        protocol,
                        matrix,
                        plan,
                        dataset_path,
                        schedule_path,
                        execution_runtime,
                        frozen_runtime_identity,
                    )
                    result_runtime = result.get("runtime", {})
                    _validate_runtime_against_frozen(
                        result_runtime,
                        frozen_runtime_identity,
                        f"{diagnostic_name}/{arm} recorded evaluation",
                    )
                    expected = _arm_result_manifest(
                        root,
                        diagnostic_name,
                        arm,
                        raw_path,
                        training_manifest,
                        protocol,
                        plan,
                        result_runtime,
                    )
                    if canonical_bytes(result) != canonical_bytes(expected):
                        raise ValueError("diagnostic arm result does not recompute")
                    raw_paths[diagnostic_name][arm] = raw_path
                    continue
                state, config, training_manifest = _train_world_model_arm(
                    root,
                    diagnostic_root,
                    diagnostic_name,
                    arm,
                    protocol,
                    matrix,
                    plan,
                    dataset_path,
                    schedule_path,
                    execution_runtime,
                    frozen_runtime_identity,
                )
                model_noise_key = derive_jax_key(
                    "matched-objective-diagnostic-model-noise",
                    matrix["matrix_sha256"],
                    diagnostic_name,
                )
                raw = _evaluate_model(
                    state,
                    config,
                    diagnostic_name,
                    evaluation,
                    draws=draws,
                    training_observation_std=training_std,
                    model_noise_key=model_noise_key,
                    test_seed=test_seed,
                )
                recompute_diagnostic_raw(diagnostic_name, raw)
                # A process can be preempted after the immutable raw file is
                # linked but before its result manifest is linked. Regenerate
                # from the authenticated checkpoint and deterministic keys;
                # the create-once writer accepts only a bitwise-equal payload.
                _write_npz_once(raw_path, raw)
                result = _arm_result_manifest(
                    root,
                    diagnostic_name,
                    arm,
                    raw_path,
                    training_manifest,
                    protocol,
                    plan,
                    execution_runtime,
                )
                _write_json_once(result_path, result)
                raw_paths[diagnostic_name][arm] = raw_path

        summary = build_diagnostic_summary(
            root, matrix, protocol, raw_paths, smoke_nonclaim=smoke
        )
        validate_diagnostic_summary_from_raw(
            summary,
            root,
            matrix,
            protocol,
            smoke_nonclaim=smoke,
            plan=plan,
        )
        _write_json_once(summary_path, summary)
    return summary


__all__ = [
    "DETERMINISTIC_NAME",
    "DIAGNOSTIC_NAMES",
    "DIAGNOSTIC_SCHEMA",
    "DiagnosticPlan",
    "STOCHASTIC_NAME",
    "build_diagnostic_summary",
    "deterministic_estimands_from_raw",
    "generate_scalar_evaluation_data",
    "generate_scalar_training_data",
    "load_numeric_npz",
    "recompute_diagnostic_raw",
    "resolve_diagnostic_plan",
    "run_registered_diagnostics",
    "stochastic_estimands_from_raw",
    "validate_diagnostic_summary_from_raw",
]
