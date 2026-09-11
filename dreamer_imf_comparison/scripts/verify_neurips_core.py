#!/usr/bin/env python3
"""Run the fail-closed core checks required before NeurIPS GPU execution."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (WORKSPACE, PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json  # noqa: E402
import dreamer_imf_compare.matched_objective_benchmark as benchmark  # noqa: E402


COMPARISON_MODULES = (
    "dreamer_imf_comparison.tests.test_matched_objective_protocol",
    "dreamer_imf_comparison.tests.test_matched_objective_benchmark",
    "dreamer_imf_comparison.tests.test_matched_objective_diagnostics",
    "dreamer_imf_comparison.tests.test_matched_objective_artifact_manifest",
    "dreamer_imf_comparison.tests.test_matched_objective_validation_cache",
    "dreamer_imf_comparison.tests.test_rollout_replay_validation",
    "dreamer_imf_comparison.tests.test_neurips_verifiers",
    "dreamer_imf_comparison.tests.test_neurips_controls",
    "dreamer_imf_comparison.tests.test_neurips_report",
    "dreamer_imf_comparison.tests.test_pixel_benchmark",
    "dreamer_imf_comparison.tests.test_trajectory_imf_theory",
    "dreamer_imf_comparison.tests.test_cluster_workflow",
)

REGRESSION_TIMEOUT_SECONDS = 900

# This outcome-independent manifest binds the complete sorted identity list in
# each suite as newline-delimited UTF-8.  It is not a module allowlist: adding,
# removing, or renaming one test changes the count and/or SHA-256 and fails.
REGRESSION_TEST_MANIFEST: Mapping[str, Mapping[str, Any]] = {
    "library": {
        "count": 114,
        "identity_sha256": "ee2e8364732ac94eefe3f86c2a3fb838b1f77cb25025fe0e35410a7235ffdd32",
    },
    "comparison": {
        "count": 212,
        "identity_sha256": "72a32575991d26c6b3d4875c0bcd8adf1917ef63fc7af3617918e30303b1eeef",
    },
}

TAMPER_TESTS = (
    "dreamer_imf_comparison.tests.test_matched_objective_protocol."
    "MatchedObjectiveProtocolTests.test_zero_adverse_missing_and_nonconfirmatory_intervals_fail",
    "dreamer_imf_comparison.tests.test_matched_objective_benchmark."
    "MatchedObjectiveBenchmarkTest.test_strong_dataset_validation_rejects_tiny_relabelled_replay",
    "dreamer_imf_comparison.tests.test_matched_objective_benchmark."
    "MatchedObjectiveBenchmarkTest.test_actor_environment_replay_rejects_forged_rewards_after_rehash",
    "dreamer_imf_comparison.tests.test_matched_objective_diagnostics."
    "MatchedObjectiveDiagnosticTests.test_checkpoint_replay_rejects_model_future_tampering",
    "dreamer_imf_comparison.tests.test_matched_objective_diagnostics."
    "MatchedObjectiveDiagnosticTests.test_main_confirmatory_gate_recomputes_estimands_from_raw",
    "dreamer_imf_comparison.tests.test_matched_objective_artifact_manifest."
    "MatchedObjectiveArtifactManifestTests.test_manifest_file_set_must_equal_every_retained_file",
    "dreamer_imf_comparison.tests.test_rollout_replay_validation."
    "RolloutReplayValidationTests.test_favorable_rehashed_draws_fail_checkpoint_replay",
)


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    additions = [
        str(WORKSPACE / "imf_dreamer_jax" / "src"),
        str(WORKSPACE / "imf_dreamer_jax" / "tests"),
        str(PROJECT),
    ]
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        additions + ([previous] if previous else [])
    )
    return environment


def _run(arguments: list[str], *, timeout_seconds: int = REGRESSION_TIMEOUT_SECONDS) -> None:
    try:
        subprocess.run(
            arguments,
            cwd=WORKSPACE,
            env=_environment(),
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"verification command timed out after {timeout_seconds} seconds"
        ) from error


def _expect_rejection(
    function: Callable[[], Any], expected_error_substring: str, message: str
) -> None:
    if not expected_error_substring:
        raise ValueError("expected rejection substring must be nonempty")
    try:
        function()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        if expected_error_substring not in str(error):
            raise AssertionError(
                f"{message}; expected error containing {expected_error_substring!r}, "
                f"observed {type(error).__name__}: {error}"
            ) from error
        return
    raise AssertionError(message)


def _flatten_test_ids(suite: unittest.TestSuite) -> tuple[str, ...]:
    identities: list[str] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            identities.extend(_flatten_test_ids(item))
        else:
            identities.append(item.id())
    return tuple(sorted(identities))


def _discover_regression_test_identities() -> dict[str, tuple[str, ...]]:
    loader = unittest.TestLoader()
    library = loader.discover(
        str(WORKSPACE / "imf_dreamer_jax" / "tests"), pattern="test_*.py"
    )
    comparison = loader.loadTestsFromNames(list(COMPARISON_MODULES))
    if loader.errors:
        raise RuntimeError(
            "regression test discovery failed: " + " | ".join(loader.errors)
        )
    discovered = {
        "library": _flatten_test_ids(library),
        "comparison": _flatten_test_ids(comparison),
    }
    for label, identities in discovered.items():
        if not identities or len(identities) != len(set(identities)):
            raise AssertionError(f"{label} regression identities are empty or duplicated")
    return discovered


def _assert_frozen_test_identities(
    discovered: Mapping[str, Sequence[str]],
    expected: Mapping[str, Mapping[str, Any]] = REGRESSION_TEST_MANIFEST,
) -> None:
    if set(discovered) != set(expected):
        raise AssertionError("regression suite names differ from the frozen identity manifest")
    for label in sorted(expected):
        observed_ids = tuple(sorted(str(value) for value in discovered[label]))
        observed_manifest = {
            "count": len(observed_ids),
            "identity_sha256": hashlib.sha256(
                "\n".join(observed_ids).encode("utf-8")
            ).hexdigest(),
        }
        expected_manifest = dict(expected[label])
        if observed_manifest != expected_manifest:
            raise AssertionError(
                f"{label} regression identity manifest mismatch; "
                f"expected={expected_manifest}, observed={observed_manifest}"
            )


def _assert_close(label: str, observed: float, expected: float) -> None:
    if not math.isfinite(observed) or not math.isfinite(expected) or not math.isclose(
        observed, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise AssertionError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def _manual_iqm(values: Sequence[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0 or not np.isfinite(ordered).all():
        raise ValueError("manual IQM requires a nonempty finite vector")
    lower = 0.25 * ordered.size
    upper = 0.75 * ordered.size
    weighted_sum = 0.0
    for index, value in enumerate(ordered):
        overlap = max(0.0, min(float(index + 1), upper) - max(float(index), lower))
        weighted_sum += overlap * float(value)
    return weighted_sum / (upper - lower)


def _manual_rollout_metrics(
    arrays: Mapping[str, np.ndarray], horizons: Sequence[int]
) -> tuple[dict[str, float], float]:
    samples = np.asarray(arrays["observation_samples"], dtype=np.float64)
    targets = np.asarray(arrays["target_observations"], dtype=np.float64)
    standard_deviation = np.asarray(
        arrays["training_observation_std"], dtype=np.float64
    )
    if samples.ndim < 4 or samples.shape[1:3] != targets.shape[:2]:
        raise AssertionError("manual rollout audit found incompatible sample/target shapes")
    coordinates = [int(value) for value in horizons]
    if len(coordinates) < 2 or coordinates != sorted(set(coordinates)):
        raise AssertionError("manual rollout audit requires ordered distinct horizons")
    errors: dict[str, float] = {}
    for horizon in coordinates:
        index = horizon - 1
        if index < 0 or index >= targets.shape[1]:
            raise AssertionError("manual rollout horizon exceeds retained targets")
        residual = (
            samples[:, :, index] - targets[None, :, index]
        ) / standard_deviation
        errors[str(horizon)] = float(np.mean(np.square(residual), dtype=np.float64))
    area = 0.0
    for left, right in zip(coordinates[:-1], coordinates[1:]):
        area += 0.5 * (errors[str(left)] + errors[str(right)]) * (right - left)
    auc = area / float(coordinates[-1] - coordinates[0])
    return errors, float(auc)


def _manual_actor_returns(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    rewards = np.asarray(arrays["rewards"], dtype=np.float64)
    lengths = np.asarray(arrays["lengths"], dtype=np.int64)
    if rewards.ndim != 2 or lengths.shape != (rewards.shape[0],):
        raise AssertionError("manual actor audit found incompatible reward/length shapes")
    if np.any(lengths <= 0) or np.any(lengths > rewards.shape[1]):
        raise AssertionError("manual actor audit found invalid retained episode lengths")
    return np.asarray(
        [np.sum(rewards[index, : int(length)], dtype=np.float64) for index, length in enumerate(lengths)],
        dtype=np.float64,
    )


def _load_npz_independently(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _one_matrix_cell(
    matrix: Mapping[str, Any], *, label: str, **identity: Any
) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in matrix["cells"]
        if all(cell.get(key) == value for key, value in identity.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"manual audit expected one {label} cell, found {len(matches)}")
    return matches[0]


def _raw_unit_index(analysis: Mapping[str, Any]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for unit in analysis.get("raw_units", []):
        key = (
            str(unit["budget_track"]),
            str(unit["task"]),
            int(unit["world_model_seed"]),
        )
        if key in indexed:
            raise AssertionError("analysis contains duplicate raw statistical units")
        indexed[key] = unit
    return indexed


def independently_recompute_primary_estimands(
    root: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild primary point estimands directly from retained arrays.

    This deliberately does not call the production analysis validator, rollout
    statistics helper, IQM helper, or unit collector.  Only canonical cell-path
    routing and JSON loading are shared with the benchmark.
    """

    profile = str(matrix["profile"])
    selected = protocol["profiles"][profile]
    arms = tuple(str(value) for value in protocol["arm_order"])
    tracks = tuple(str(value) for value in protocol["budget_track_order"])
    if set(tracks) != set(protocol["budget_tracks"]):
        raise AssertionError("registered budget-track order and definitions differ")
    tasks = tuple(str(value) for value in selected["tasks"])
    world_seeds = tuple(int(value) for value in selected["world_model_seeds"])
    actor_seeds = tuple(
        int(value) for value in selected["actor_seeds_nested_within_world_model_seed"]
    )
    horizons = tuple(int(value) for value in protocol["evaluation"]["rollout_horizons"])
    if (
        analysis.get("profile") != profile
        or analysis.get("evidence_class") != selected["evidence_class"]
    ):
        raise AssertionError("analysis profile/evidence class differs from the frozen matrix")
    if arms != ("shortcut_forcing", "trajectory_imf"):
        raise AssertionError("manual core audit is frozen to the registered two-arm comparison")

    raw_index = _raw_unit_index(analysis)
    expected_unit_keys = {
        (track, task, seed)
        for track in tracks
        for task in tasks
        for seed in world_seeds
    }
    if set(raw_index) != expected_unit_keys:
        raise AssertionError("analysis raw-unit identities differ from the frozen matrix")

    reconstructed: dict[tuple[str, str, int], dict[str, Any]] = {}
    checked_rollouts = 0
    checked_actors = 0
    for key in sorted(expected_unit_keys):
        track, task, world_seed = key
        rollout_auc: dict[str, float] = {}
        actor_returns: dict[str, dict[str, list[float]]] = {}
        raw_unit = raw_index[key]
        for arm in arms:
            rollout_cell = _one_matrix_cell(
                matrix,
                label="primary rollout",
                stage="rollout",
                budget_track=track,
                task=task,
                world_model_seed=world_seed,
                arm=arm,
                nfe=int(protocol["evaluation"]["primary_nfe"][arm]),
            )
            rollout_directory = benchmark.stage_directory(root, rollout_cell)
            rollout_result = read_json(rollout_directory / "result.json")
            rollout_arrays = _load_npz_independently(
                rollout_directory / "predictive_draws.npz"
            )
            errors, auc = _manual_rollout_metrics(rollout_arrays, horizons)
            for horizon, error in errors.items():
                _assert_close(
                    f"rollout {rollout_cell['cell_id']} horizon {horizon}",
                    error,
                    float(
                        rollout_result["per_horizon"][horizon][
                            "expected_standardized_squared_error"
                        ]
                    ),
                )
            _assert_close(
                f"rollout {rollout_cell['cell_id']} AUC",
                auc,
                float(rollout_result["normalized_free_running_rollout_error_auc"]),
            )
            _assert_close(
                f"raw-unit rollout {key} {arm}",
                auc,
                float(raw_unit["rollout_auc"][arm]),
            )
            rollout_auc[arm] = auc
            checked_rollouts += 1

            actor_returns[arm] = {}
            for actor_seed in actor_seeds:
                actor_cell = _one_matrix_cell(
                    matrix,
                    label="actor",
                    stage="actor",
                    budget_track=track,
                    task=task,
                    world_model_seed=world_seed,
                    arm=arm,
                    actor_seed=actor_seed,
                )
                actor_directory = benchmark.stage_directory(root, actor_cell)
                actor_result = read_json(actor_directory / "result.json")
                trace_arrays = _load_npz_independently(
                    actor_directory / "action_traces.npz"
                )
                raw_returns = _manual_actor_returns(trace_arrays)
                saved_returns = np.asarray(
                    actor_result["episode_returns"], dtype=np.float64
                )
                if raw_returns.shape != saved_returns.shape:
                    raise AssertionError(
                        f"actor {actor_cell['cell_id']} episode-return shape mismatch"
                    )
                for episode, (observed, expected) in enumerate(
                    zip(raw_returns, saved_returns)
                ):
                    _assert_close(
                        f"actor {actor_cell['cell_id']} episode {episode}",
                        float(observed),
                        float(expected),
                    )
                normalized = raw_returns / 1000.0
                saved_normalized = np.asarray(
                    actor_result["normalized_episode_returns"], dtype=np.float64
                )
                if normalized.shape != saved_normalized.shape:
                    raise AssertionError(
                        f"actor {actor_cell['cell_id']} normalized-return shape mismatch"
                    )
                for episode, (observed, expected) in enumerate(
                    zip(normalized, saved_normalized)
                ):
                    _assert_close(
                        f"actor {actor_cell['cell_id']} normalized episode {episode}",
                        float(observed),
                        float(expected),
                    )
                _assert_close(
                    f"actor {actor_cell['cell_id']} normalized mean",
                    float(np.mean(normalized, dtype=np.float64)),
                    float(actor_result["normalized_episode_return_mean"]),
                )
                raw_unit_returns = np.asarray(
                    raw_unit["actor_episode_returns"][arm][str(actor_seed)],
                    dtype=np.float64,
                )
                if raw_unit_returns.shape != normalized.shape:
                    raise AssertionError(
                        f"raw-unit actor {key} {arm} seed {actor_seed} shape mismatch"
                    )
                for episode, (observed, expected) in enumerate(
                    zip(normalized, raw_unit_returns)
                ):
                    _assert_close(
                        f"raw-unit actor {key} {arm} seed {actor_seed} episode {episode}",
                        float(observed),
                        float(expected),
                    )
                actor_returns[arm][str(actor_seed)] = normalized.tolist()
                checked_actors += 1
        reconstructed[key] = {
            "rollout_auc": rollout_auc,
            "actor_episode_returns": actor_returns,
        }

    manual_tracks: dict[str, Any] = {}
    rollout_name = "normalized_free_running_rollout_error_auc"
    actor_name = "real_environment_normalized_return_iqm"
    for track in tracks:
        rollout_values = {
            arm: [
                float(reconstructed[(track, task, seed)]["rollout_auc"][arm])
                for task in tasks
                for seed in world_seeds
            ]
            for arm in arms
        }
        actor_values: dict[str, list[float]] = {arm: [] for arm in arms}
        for arm in arms:
            for task in tasks:
                for seed in world_seeds:
                    nested = reconstructed[(track, task, seed)][
                        "actor_episode_returns"
                    ][arm]
                    actor_seed_means = [
                        float(np.mean(np.asarray(nested[str(actor_seed)], dtype=np.float64)))
                        for actor_seed in actor_seeds
                    ]
                    actor_values[arm].append(
                        float(np.mean(np.asarray(actor_seed_means, dtype=np.float64)))
                    )
        rollout_iqm = {arm: _manual_iqm(rollout_values[arm]) for arm in arms}
        actor_iqm = {arm: _manual_iqm(actor_values[arm]) for arm in arms}
        contrasts = {
            rollout_name: rollout_iqm["shortcut_forcing"]
            - rollout_iqm["trajectory_imf"],
            actor_name: actor_iqm["trajectory_imf"]
            - actor_iqm["shortcut_forcing"],
        }
        manual_tracks[track] = {
            rollout_name: {"arm_iqm": rollout_iqm, "contrast": contrasts[rollout_name]},
            actor_name: {"arm_iqm": actor_iqm, "contrast": contrasts[actor_name]},
        }
        for metric_name in (rollout_name, actor_name):
            for arm in arms:
                _assert_close(
                    f"analysis {track} {metric_name} {arm} IQM",
                    float(manual_tracks[track][metric_name]["arm_iqm"][arm]),
                    float(analysis["tracks"][track][metric_name]["arm_iqm"][arm]),
                )
            _assert_close(
                f"analysis {track} {metric_name} contrast",
                float(manual_tracks[track][metric_name]["contrast"]),
                float(analysis["tracks"][track][metric_name]["contrast"]),
            )

    practical_spec = protocol["statistics"]["practical_significance"]
    practical_saved = analysis["practical_significance"]
    all_thresholds_met = True
    for track in tracks:
        rollout_row = manual_tracks[track][rollout_name]
        rollout_absolute = float(rollout_row["contrast"])
        rollout_relative = rollout_absolute / max(
            float(rollout_row["arm_iqm"]["shortcut_forcing"]), 1e-6
        )
        rollout_threshold = practical_spec["thresholds"][rollout_name]
        rollout_passed = (
            rollout_absolute
            >= float(rollout_threshold["minimum_absolute_iqm_contrast"])
            and rollout_relative
            >= float(rollout_threshold["minimum_relative_iqm_reduction"])
        )
        actor_absolute = float(manual_tracks[track][actor_name]["contrast"])
        actor_threshold = practical_spec["thresholds"][actor_name]
        actor_passed = actor_absolute >= float(
            actor_threshold["minimum_absolute_iqm_contrast"]
        )
        saved_rollout = practical_saved["tracks"][track][rollout_name]
        saved_actor = practical_saved["tracks"][track][actor_name]
        _assert_close(
            f"practical {track} rollout absolute", rollout_absolute,
            float(saved_rollout["absolute_iqm_contrast"]),
        )
        _assert_close(
            f"practical {track} rollout relative", rollout_relative,
            float(saved_rollout["relative_iqm_reduction"]),
        )
        _assert_close(
            f"practical {track} actor absolute", actor_absolute,
            float(saved_actor["absolute_iqm_contrast"]),
        )
        _assert_close(
            f"practical {track} rollout absolute threshold",
            float(rollout_threshold["minimum_absolute_iqm_contrast"]),
            float(saved_rollout["minimum_absolute_iqm_contrast"]),
        )
        _assert_close(
            f"practical {track} rollout relative threshold",
            float(rollout_threshold["minimum_relative_iqm_reduction"]),
            float(saved_rollout["minimum_relative_iqm_reduction"]),
        )
        _assert_close(
            f"practical {track} actor absolute threshold",
            float(actor_threshold["minimum_absolute_iqm_contrast"]),
            float(saved_actor["minimum_absolute_iqm_contrast"]),
        )
        _assert_close(
            f"practical {track} actor raw-return equivalent",
            float(actor_threshold["raw_dmc_return_equivalent"]),
            float(saved_actor["raw_dmc_return_equivalent"]),
        )
        if bool(saved_rollout["passed"]) is not rollout_passed:
            raise AssertionError(f"practical {track} rollout threshold decision mismatch")
        if bool(saved_actor["passed"]) is not actor_passed:
            raise AssertionError(f"practical {track} actor threshold decision mismatch")
        all_thresholds_met = all_thresholds_met and rollout_passed and actor_passed

    if bool(practical_saved["all_thresholds_met"]) is not all_thresholds_met:
        raise AssertionError("overall practical-threshold decision mismatch")
    superiority_passed = bool(analysis["superiority"]["passed"])
    if bool(practical_saved["statistical_superiority_passed"]) is not superiority_passed:
        raise AssertionError("practical gate statistical-superiority input mismatch")
    claim_allowed = (
        str(analysis["evidence_class"]) == "confirmatory"
        and all_thresholds_met
        and superiority_passed
    )
    if (
        bool(practical_saved["practically_meaningful_superiority_claim_allowed"])
        is not claim_allowed
    ):
        raise AssertionError("practical superiority wording decision mismatch")
    return {
        "checked_rollout_cells": checked_rollouts,
        "checked_actor_cells": checked_actors,
        "statistical_units": len(reconstructed),
        "tracks": manual_tracks,
        "all_practical_thresholds_met": bool(all_thresholds_met),
        "claim_allowed": bool(claim_allowed),
    }


def run_regressions() -> None:
    discovered = _discover_regression_test_identities()
    _assert_frozen_test_identities(discovered)
    _run([sys.executable, "-m", "unittest", *discovered["library"]])
    _run([sys.executable, "-m", "unittest", *discovered["comparison"]])
    identity_digest = hashlib.sha256(
        "\n".join(
            f"{suite}:{identity}"
            for suite in sorted(discovered)
            for identity in discovered[suite]
        ).encode("utf-8")
    ).hexdigest()
    print(
        "NEURIPS_CORE_REGRESSION_IDENTITIES "
        f"library={len(discovered['library'])} "
        f"comparison={len(discovered['comparison'])} sha256={identity_digest}"
    )
    print("NEURIPS_CORE_REGRESSIONS_VERIFIED")


def run_tamper_controls(output: Path) -> None:
    _run([sys.executable, "-m", "unittest", *TAMPER_TESTS])
    root = output.resolve()
    protocol = read_json(root / "frozen_protocol.json")
    source = read_json(root / "source_manifest.json")
    matrix = read_json(root / "matrix.json")
    benchmark.validate_source_manifest(source, WORKSPACE)
    benchmark.validate_matrix(matrix, protocol, source)

    compute_cell = benchmark.matrix_cells(matrix, "compute_plan")[0]
    compute_directory = benchmark.stage_directory(root, compute_cell)
    compute_plan = read_json(compute_directory / "result.json")
    benchmark.validate_compute_plan_cell(
        compute_plan,
        protocol,
        matrix,
        compute_cell,
        root,
        rederive_compiler_evidence=True,
    )
    benchmark._validate_compute_files(compute_plan, compute_directory)

    with tempfile.TemporaryDirectory(prefix="trajectory-imf-compute-tamper-") as directory:
        copied = Path(directory) / "compute"
        shutil.copytree(compute_directory, copied)
        target = copied / compute_plan["hlo_files"][0]["path"]
        target.write_text(target.read_text(encoding="utf-8") + "\nTAMPER\n", encoding="utf-8")
        _expect_rejection(
            lambda: benchmark._validate_compute_files(compute_plan, copied),
            "retained compiler artifact digest mismatch",
            "modified compiler IR was accepted",
        )

    altered_matrix = copy.deepcopy(matrix)
    altered_matrix["matrix_sha256"] = "0" * 64
    _expect_rejection(
        lambda: benchmark._authenticate_frozen_arguments(
            root,
            altered_matrix,
            protocol,
            workspace=WORKSPACE,
        ),
        "supplied matrix differs from the output root's frozen matrix",
        "a finalizer accepted a supplied matrix differing from the frozen root",
    )

    world_cell = benchmark.matrix_cells(matrix, "world_model")[0]
    world_directory = benchmark.stage_directory(root, world_cell)
    world_result = read_json(world_directory / "result.json")
    with tempfile.TemporaryDirectory(prefix="trajectory-imf-world-tamper-") as directory:
        copied_world = Path(directory) / "world"
        shutil.copytree(world_directory, copied_world)
        checkpoint_path = copied_world / "checkpoint.pkl"
        from imf_dreamer_jax import load_checkpoint, save_checkpoint

        state, config, metadata = load_checkpoint(checkpoint_path)
        forged_optimizer = state.model_optimizer._replace(
            step=state.model_optimizer.step + 1
        )
        forged_state = state._replace(model_optimizer=forged_optimizer)
        save_checkpoint(
            checkpoint_path,
            forged_state,
            config,
            metadata=metadata,
        )
        reloaded_state, reloaded_config, reloaded_metadata = load_checkpoint(
            checkpoint_path
        )
        original_step = int(np.asarray(state.model_optimizer.step))
        if (
            reloaded_config != config
            or reloaded_metadata != metadata
            or int(np.asarray(reloaded_state.model_optimizer.step)) != original_step + 1
        ):
            raise AssertionError("checkpoint tamper changed more than the intended step")
        restored_step_state = reloaded_state._replace(
            model_optimizer=reloaded_state.model_optimizer._replace(
                step=state.model_optimizer.step
            )
        )
        if benchmark._tree_digest(restored_step_state) != benchmark._tree_digest(state):
            raise AssertionError("checkpoint tamper changed state beyond model_optimizer.step")
        forged_world = copy.deepcopy(world_result)
        forged_world["checkpoint_sha256"] = benchmark.file_sha256(checkpoint_path)
        if {
            key
            for key in forged_world
            if forged_world[key] != world_result[key]
        } != {"checkpoint_sha256"}:
            raise AssertionError("forged world result changed beyond checkpoint_sha256")
        _expect_rejection(
            lambda: benchmark.validate_world_result(
                forged_world,
                world_cell,
                copied_world,
                protocol=protocol,
                matrix=matrix,
                output_root=root,
            ),
            "world-model checkpoint optimizer step mismatch",
            "a real checkpoint with a forged optimizer step was accepted",
        )

    analysis = read_json(root / "analysis.json")
    independent = independently_recompute_primary_estimands(
        root, matrix, protocol, analysis
    )
    print(
        "NEURIPS_CORE_INDEPENDENT_ESTIMANDS_VERIFIED "
        f"rollouts={independent['checked_rollout_cells']} "
        f"actors={independent['checked_actor_cells']} "
        f"units={independent['statistical_units']}"
    )
    print("NEURIPS_CORE_TAMPER_CONTROLS_VERIFIED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--regressions", action="store_true")
    mode.add_argument("--tamper-controls", action="store_true")
    parser.add_argument(
        "--output",
        default=str(PROJECT / "results" / "matched_objective_smoke_final"),
    )
    arguments = parser.parse_args()
    if arguments.regressions:
        run_regressions()
    else:
        run_tamper_controls(Path(arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
