"""Control-alignment diagnostics for frozen learned world models.

The numerical helpers in this module are deliberately independent of JAX and
dm_control.  The command-line runner owns checkpoint replay and simulator
interaction, then feeds retained arrays into these small, testable summaries.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPORT_SCHEMA = "trajectory-imf-policy-alignment-diagnostics-v1"
PRESERVATION_SCHEMA = "frozen-pilot-preservation-v1"
METRIC_NAMES = (
    "actor_visited_model_error",
    "imagined_real_alignment",
    "action_ranking",
    "gradient_fidelity",
)
ARM_NAMES = ("shortcut_forcing", "trajectory_imf")


def _array(value: Any, *, ndim: int | None = None, name: str = "array") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks for ties without SciPy."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(x: Any, y: Any, *, ranks: bool = False) -> float | None:
    first = _array(x, ndim=1, name="first correlation vector")
    second = _array(y, ndim=1, name="second correlation vector")
    if first.shape != second.shape or first.size < 2:
        raise ValueError("correlation vectors must have equal length of at least two")
    if ranks:
        first, second = _rankdata(first), _rankdata(second)
    first = first - first.mean()
    second = second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-15:
        return None
    return _finite(float(np.dot(first, second) / denominator), "correlation")


def actor_visited_model_error(
    observation_samples: Any,
    reward_samples: Any,
    continuation_samples: Any,
    target_observations: Any,
    target_rewards: Any,
    target_continuations: Any,
    observation_std: Any,
) -> dict[str, float | int]:
    """Summarize one-step predictive error at states visited by trained actors.

    Sample arrays use ``[draw, state, ...]`` and targets use ``[state, ...]``.
    Reporting both expected sample error and predictive-mean error separates
    distributional spread from mean dynamics fidelity.
    """

    observations = _array(observation_samples, name="observation samples")
    rewards = _array(reward_samples, name="reward samples")
    continuations = _array(continuation_samples, name="continuation samples")
    target_obs = _array(target_observations, name="target observations")
    target_reward = _array(target_rewards, ndim=1, name="target rewards")
    target_continuation = _array(
        target_continuations, ndim=1, name="target continuations"
    )
    std = _array(observation_std, name="observation standard deviation")
    if observations.ndim < 3 or observations.shape[1:] != target_obs.shape:
        raise ValueError("observation samples and targets have incompatible shapes")
    if rewards.shape != (observations.shape[0], observations.shape[1]):
        raise ValueError("reward samples have incompatible shape")
    if continuations.shape != rewards.shape:
        raise ValueError("continuation samples have incompatible shape")
    if target_reward.shape[0] != observations.shape[1]:
        raise ValueError("reward targets have incompatible shape")
    if target_continuation.shape != target_reward.shape:
        raise ValueError("continuation targets have incompatible shape")
    if std.shape != target_obs.shape[1:] or np.any(std <= 0.0):
        raise ValueError("observation standard deviation has incompatible shape")
    standardized = (observations - target_obs[None]) / std
    mean_standardized = (observations.mean(axis=0) - target_obs) / std
    return {
        "states": int(observations.shape[1]),
        "draws": int(observations.shape[0]),
        "expected_standardized_observation_mse": _finite(
            np.mean(np.square(standardized)), "expected observation error"
        ),
        "predictive_mean_standardized_observation_rmse": _finite(
            np.sqrt(np.mean(np.square(mean_standardized))),
            "predictive mean observation error",
        ),
        "reward_expected_mse": _finite(
            np.mean(np.square(rewards - target_reward[None])), "reward error"
        ),
        "reward_predictive_mean_mse": _finite(
            np.mean(np.square(rewards.mean(axis=0) - target_reward)),
            "mean reward error",
        ),
        "continuation_expected_brier": _finite(
            np.mean(np.square(continuations - target_continuation[None])),
            "continuation error",
        ),
    }


def action_ranking_metrics(
    predicted_returns: Any,
    simulator_returns: Any,
    *,
    tie_tolerance: float = 1e-8,
) -> dict[str, float | int | None]:
    """Measure whether the world model ranks local action alternatives correctly."""

    predicted = _array(predicted_returns, ndim=2, name="predicted returns")
    actual = _array(simulator_returns, ndim=2, name="simulator returns")
    if predicted.shape != actual.shape or predicted.shape[1] < 2:
        raise ValueError("ranking arrays must share [state, candidate>=2] shape")
    if tie_tolerance < 0.0:
        raise ValueError("tie tolerance must be nonnegative")
    correct = 0
    comparable = 0
    state_correlations: list[float] = []
    top1 = 0
    regrets: list[float] = []
    for model_row, true_row in zip(predicted, actual, strict=True):
        for left in range(true_row.size):
            for right in range(left + 1, true_row.size):
                truth = float(true_row[left] - true_row[right])
                if abs(truth) <= tie_tolerance:
                    continue
                estimate = float(model_row[left] - model_row[right])
                comparable += 1
                correct += int(np.sign(estimate) == np.sign(truth))
        value = correlation(model_row, true_row, ranks=True)
        if value is not None:
            state_correlations.append(value)
        predicted_best = int(np.argmax(model_row))
        true_best = int(np.argmax(true_row))
        top1 += int(predicted_best == true_best)
        regrets.append(float(true_row[true_best] - true_row[predicted_best]))
    return {
        "states": int(predicted.shape[0]),
        "candidates_per_state": int(predicted.shape[1]),
        "comparable_pairs": int(comparable),
        "pairwise_accuracy": None if comparable == 0 else float(correct / comparable),
        "mean_state_spearman": (
            None if not state_correlations else _finite(np.mean(state_correlations), "rank correlation")
        ),
        "top1_accuracy": float(top1 / predicted.shape[0]),
        "mean_simulator_regret": _finite(np.mean(regrets), "simulator regret"),
    }


def gradient_fidelity_metrics(
    model_gradients: Any,
    simulator_gradients: Any,
    *,
    nonzero_tolerance: float = 1e-10,
) -> dict[str, float | int | None]:
    """Compare differentiable-model gradients to simulator finite differences."""

    model = _array(model_gradients, ndim=2, name="model gradients")
    actual = _array(simulator_gradients, ndim=2, name="simulator gradients")
    if model.shape != actual.shape:
        raise ValueError("gradient arrays must share [state, action_dim] shape")
    cosines: list[float] = []
    sign_correct = 0
    sign_total = 0
    norm_ratios: list[float] = []
    for model_row, true_row in zip(model, actual, strict=True):
        model_norm = float(np.linalg.norm(model_row))
        true_norm = float(np.linalg.norm(true_row))
        if model_norm > nonzero_tolerance and true_norm > nonzero_tolerance:
            cosines.append(float(np.dot(model_row, true_row) / (model_norm * true_norm)))
            norm_ratios.append(model_norm / true_norm)
        mask = np.abs(true_row) > nonzero_tolerance
        sign_total += int(mask.sum())
        sign_correct += int(np.sum(np.sign(model_row[mask]) == np.sign(true_row[mask])))
    return {
        "states": int(model.shape[0]),
        "action_dimensions": int(model.shape[1]),
        "nonzero_vector_pairs": int(len(cosines)),
        "mean_cosine_similarity": None if not cosines else _finite(np.mean(cosines), "cosine"),
        "median_cosine_similarity": None if not cosines else _finite(np.median(cosines), "median cosine"),
        "component_sign_accuracy": None if sign_total == 0 else float(sign_correct / sign_total),
        "median_model_to_simulator_norm_ratio": (
            None if not norm_ratios else _finite(np.median(norm_ratios), "gradient norm ratio")
        ),
    }


def imagined_real_alignment(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("alignment requires at least two actor records")
    imagined = _array(
        [record["final_metrics"]["mean_imagined_return"] for record in records],
        ndim=1,
        name="imagined returns",
    )
    real = _array(
        [1000.0 * float(record["normalized_episode_return_mean"]) for record in records],
        ndim=1,
        name="real returns",
    )
    return {
        "cells": int(len(records)),
        "pearson": correlation(imagined, real),
        "spearman": correlation(imagined, real, ranks=True),
        "imagined_mean": _finite(imagined.mean(), "imagined mean"),
        "real_episode_return_mean": _finite(real.mean(), "real mean"),
        "real_episode_return_median": _finite(np.median(real), "real median"),
    }


def actor_number_summary(records: Sequence[Mapping[str, Any]], *, expected_cells: int) -> dict[str, Any]:
    if expected_cells <= 0 or not records:
        raise ValueError("actor summary requires records and a positive expected count")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    checkpoint_hashes: set[str] = set()
    trace_hashes: set[str] = set()
    for record in records:
        if record.get("status") != "complete":
            raise ValueError("actor summary accepts only complete records")
        if record.get("world_model_frozen") is not True or float(
            record.get("world_model_parameter_delta", math.nan)
        ) != 0.0:
            raise ValueError("actor result did not preserve its world model")
        checkpoint_hashes.add(str(record["checkpoint_sha256"]))
        trace_hashes.add(str(record["raw_action_traces_sha256"]))
        groups[(str(record["task"]), str(record["arm"]))].append(record)
    rows = []
    for (task, arm), values in sorted(groups.items()):
        cell_returns = _array(
            [value["normalized_episode_return_mean"] for value in values],
            ndim=1,
            name="cell returns",
        )
        episodes = _array(
            [episode for value in values for episode in value["normalized_episode_returns"]],
            ndim=1,
            name="episode returns",
        )
        rows.append(
            {
                "task": task,
                "arm": arm,
                "cells": len(values),
                "episodes": int(episodes.size),
                "normalized_return_mean": _finite(cell_returns.mean(), "return mean"),
                "normalized_return_median": _finite(np.median(cell_returns), "return median"),
                "normalized_return_q25": _finite(np.quantile(cell_returns, 0.25), "return q25"),
                "normalized_return_q75": _finite(np.quantile(cell_returns, 0.75), "return q75"),
                "nonzero_episodes": int(np.sum(episodes > 0.0)),
                "mean_wall_seconds": _finite(
                    np.mean([float(value["wall_seconds"]) for value in values]),
                    "wall time",
                ),
            }
        )
    if len(checkpoint_hashes) != len(records) or len(trace_hashes) != len(records):
        raise ValueError("completed actor records must have unique checkpoints and traces")
    return {
        "completed_cells": len(records),
        "expected_cells": int(expected_cells),
        "completion_fraction": float(len(records) / expected_cells),
        "world_models_frozen": True,
        "unique_checkpoints": len(checkpoint_hashes),
        "unique_action_traces": len(trace_hashes),
        "groups": rows,
    }


def aggregate_metric_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Pool retained state-level arrays and recompute a metric by task and arm."""

    if metric not in {"actor_visited_model_error", "action_ranking", "gradient_fidelity"}:
        raise ValueError(f"unsupported pooled metric {metric!r}")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["arm"]))].append(row)
    output = []
    for (task, arm), values in sorted(grouped.items()):
        raw = [value["raw"] for value in values]
        if metric == "actor_visited_model_error":
            # Each world-model seed has its own train-only normalization
            # statistics.  Standardize cellwise before pooling so one seed's
            # scale is never applied to another seed's observations.
            standardized_samples = []
            standardized_targets = []
            for item in raw:
                samples = np.asarray(item["observation_samples"])
                targets = np.asarray(item["target_observations"])
                std = np.asarray(item["observation_std"])
                standardized_samples.append((samples - targets[None]) / std)
                standardized_targets.append(np.zeros_like(targets))
            summary = actor_visited_model_error(
                np.concatenate(standardized_samples, axis=1),
                np.concatenate([np.asarray(item["reward_samples"]) for item in raw], axis=1),
                np.concatenate([np.asarray(item["continuation_samples"]) for item in raw], axis=1),
                np.concatenate(standardized_targets, axis=0),
                np.concatenate([np.asarray(item["target_rewards"]) for item in raw], axis=0),
                np.concatenate([np.asarray(item["target_continuations"]) for item in raw], axis=0),
                np.ones_like(np.asarray(raw[0]["observation_std"])),
            )
        elif metric == "action_ranking":
            summary = action_ranking_metrics(
                np.concatenate([np.asarray(item["predicted_returns"]) for item in raw]),
                np.concatenate([np.asarray(item["simulator_returns"]) for item in raw]),
            )
        else:
            summary = gradient_fidelity_metrics(
                np.concatenate([np.asarray(item["model_gradients"]) for item in raw]),
                np.concatenate([np.asarray(item["simulator_gradients"]) for item in raw]),
            )
        output.append(
            {
                "task": task,
                "arm": arm,
                "cells": len(values),
                **summary,
            }
        )
    return output


def validate_report(report: Mapping[str, Any], *, forbidden_root: str | None = None) -> None:
    required = {
        "schema_version",
        "status",
        "source_commit",
        "pilot_root",
        "diagnostic_output_root",
        "config",
        "sample_manifest",
        "actor_numbers",
        "metrics",
    }
    if set(report) != required or report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("diagnostic report schema is incomplete")
    if report.get("status") != "complete":
        raise ValueError("diagnostic report is incomplete")
    if not isinstance(report.get("source_commit"), str) or len(report["source_commit"]) != 40:
        raise ValueError("diagnostic source commit is invalid")
    output = Path(str(report["diagnostic_output_root"])).resolve()
    if forbidden_root is not None:
        forbidden = Path(forbidden_root).resolve()
        if output == forbidden or forbidden in output.parents:
            raise ValueError("diagnostic output is inside the frozen pilot root")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
        raise ValueError("diagnostic report does not contain all four metrics")
    manifest = report.get("sample_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("diagnostic sample manifest is empty")
    for row in manifest:
        if set(row) != {
            "cell_id",
            "task",
            "arm",
            "world_model_seed",
            "actor_seed",
            "candidate_id",
            "raw_path",
            "raw_sha256",
            "actor_checkpoint_sha256",
        }:
            raise ValueError("diagnostic sample manifest row is malformed")
    summary = report.get("actor_numbers")
    if not isinstance(summary, Mapping) or int(summary.get("completed_cells", 0)) <= 0:
        raise ValueError("actor number summary is missing")
    if int(summary.get("expected_cells", 0)) != 288:
        raise ValueError("actor number summary expected-cell count drifted")
    if int(summary.get("unique_checkpoints", -1)) != int(summary["completed_cells"]):
        raise ValueError("actor checkpoints are not unique")


def verify_metric_coverage(
    report: Mapping[str, Any],
    *,
    metrics: Sequence[str],
    tasks: Sequence[str],
    arms: Sequence[str],
) -> None:
    validate_report(report)
    expected = {(task, arm) for task in tasks for arm in arms}
    for metric in metrics:
        if metric not in METRIC_NAMES:
            raise ValueError(f"unknown metric {metric!r}")
        rows = report["metrics"][metric]
        if not isinstance(rows, list):
            raise ValueError(f"metric {metric} must contain grouped rows")
        observed = {(str(row.get("task")), str(row.get("arm"))) for row in rows}
        if observed != expected:
            raise ValueError(f"metric {metric} coverage differs from requested task-arm grid")
        for row in rows:
            if int(row.get("cells", 0)) <= 0:
                raise ValueError(f"metric {metric} has an empty task-arm group")
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"metric {metric} contains non-finite value {key}")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


__all__ = [
    "ARM_NAMES",
    "METRIC_NAMES",
    "PRESERVATION_SCHEMA",
    "REPORT_SCHEMA",
    "action_ranking_metrics",
    "actor_number_summary",
    "actor_visited_model_error",
    "aggregate_metric_rows",
    "canonical_sha256",
    "correlation",
    "gradient_fidelity_metrics",
    "imagined_real_alignment",
    "validate_report",
    "verify_metric_coverage",
]
