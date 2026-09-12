from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "dreamer_imf_compare" / "policy_alignment_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("policy_alignment_diagnostics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)

REPORT_SCHEMA = diagnostics.REPORT_SCHEMA
action_ranking_metrics = diagnostics.action_ranking_metrics
actor_number_summary = diagnostics.actor_number_summary
actor_visited_model_error = diagnostics.actor_visited_model_error
correlation = diagnostics.correlation
gradient_fidelity_metrics = diagnostics.gradient_fidelity_metrics
imagined_real_alignment = diagnostics.imagined_real_alignment
validate_report = diagnostics.validate_report
verify_metric_coverage = diagnostics.verify_metric_coverage


TASKS = ("dmc_pendulum_swingup", "dmc_reacher_easy")
ARMS = ("shortcut_forcing", "trajectory_imf")


def actor_record(task: str, arm: str, index: int, real: float, imagined: float):
    return {
        "status": "complete",
        "task": task,
        "arm": arm,
        "normalized_episode_return_mean": real,
        "normalized_episode_returns": [real, real],
        "final_metrics": {"mean_imagined_return": imagined},
        "world_model_frozen": True,
        "world_model_parameter_delta": 0.0,
        "checkpoint_sha256": f"checkpoint-{index}",
        "raw_action_traces_sha256": f"trace-{index}",
        "wall_seconds": 10.0 + index,
    }


def complete_report():
    metric_rows = [
        {"task": task, "arm": arm, "cells": 1, "score": 0.5}
        for task in TASKS
        for arm in ARMS
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": "1" * 40,
        "pilot_root": "/work/pilot",
        "diagnostic_output_root": "/work/diagnostics",
        "config": {},
        "sample_manifest": [
            {
                "cell_id": "actor-1",
                "task": TASKS[0],
                "arm": ARMS[0],
                "world_model_seed": 1,
                "actor_seed": 2,
                "evaluation_episode": 0,
                "candidate_id": "candidate-1",
                "raw_path": "cells/actor-1/diagnostics.npz",
                "raw_sha256": "2" * 64,
                "actor_checkpoint_sha256": "3" * 64,
            }
        ],
        "actor_numbers": {
            "completed_cells": 1,
            "expected_cells": 288,
            "unique_checkpoints": 1,
        },
        "metrics": {
            "actor_visited_model_error": copy.deepcopy(metric_rows),
            "imagined_real_alignment": copy.deepcopy(metric_rows),
            "action_ranking": copy.deepcopy(metric_rows),
            "gradient_fidelity": copy.deepcopy(metric_rows),
        },
    }


def test_actor_visited_model_error_exact_prediction_and_bad_scale_control():
    targets = np.array([[1.0, 2.0], [2.0, 4.0]])
    samples = np.stack([targets, targets])
    summary = actor_visited_model_error(
        samples,
        np.zeros((2, 2)),
        np.ones((2, 2)),
        targets,
        np.zeros(2),
        np.ones(2),
        np.ones(2),
    )
    assert summary["expected_standardized_observation_mse"] == 0.0
    assert summary["reward_expected_mse"] == 0.0
    assert summary["continuation_expected_brier"] == 0.0
    with pytest.raises(ValueError, match="standard deviation"):
        actor_visited_model_error(
            samples,
            np.zeros((2, 2)),
            np.ones((2, 2)),
            targets,
            np.zeros(2),
            np.ones(2),
            np.array([1.0, 0.0]),
        )


def test_action_ranking_detects_perfect_and_reversed_controls():
    true = np.array([[0.0, 1.0, 3.0], [3.0, 2.0, 0.0]])
    perfect = action_ranking_metrics(true, true)
    reversed_result = action_ranking_metrics(-true, true)
    assert perfect["pairwise_accuracy"] == 1.0
    assert perfect["top1_accuracy"] == 1.0
    assert perfect["simulator_tie_fraction"] == 0.0
    assert perfect["mean_simulator_regret"] == 0.0
    assert reversed_result["pairwise_accuracy"] == 0.0
    assert reversed_result["top1_accuracy"] == 0.0
    assert reversed_result["mean_simulator_regret"] > 0.0


def test_gradient_fidelity_detects_aligned_and_opposed_controls():
    true = np.array([[1.0, 2.0], [-2.0, 1.0]])
    aligned = gradient_fidelity_metrics(2.0 * true, true)
    opposed = gradient_fidelity_metrics(-true, true)
    assert aligned["mean_cosine_similarity"] == pytest.approx(1.0)
    assert aligned["component_sign_accuracy"] == 1.0
    assert aligned["median_model_to_simulator_norm_ratio"] == pytest.approx(2.0)
    assert opposed["mean_cosine_similarity"] == pytest.approx(-1.0)
    assert opposed["component_sign_accuracy"] == 0.0
    assert aligned["mean_model_gradient_norm"] > aligned["mean_simulator_gradient_norm"]


def test_degenerate_action_and_gradient_controls_are_reported_not_scored():
    model_returns = np.array([[0.0, 1.0, 2.0], [3.0, 1.0, -1.0]])
    tied_returns = np.zeros_like(model_returns)
    ranking = action_ranking_metrics(model_returns, tied_returns)
    assert ranking["informative_states"] == 0
    assert ranking["simulator_tie_fraction"] == 1.0
    assert ranking["top1_accuracy"] is None
    assert ranking["mean_predicted_return_range"] > 0.0

    model_gradients = np.ones((2, 2))
    zero_gradients = np.zeros((2, 2))
    gradients = gradient_fidelity_metrics(model_gradients, zero_gradients)
    assert gradients["nonzero_vector_pairs"] == 0
    assert gradients["simulator_nonzero_states"] == 0
    assert gradients["hallucinated_nonzero_fraction"] == 1.0


def test_imagined_real_alignment_and_constant_negative_control():
    records = [
        actor_record(TASKS[0], ARMS[0], index, real, imagined)
        for index, (real, imagined) in enumerate(((0.1, 1.0), (0.2, 2.0), (0.4, 4.0)))
    ]
    summary = imagined_real_alignment(records)
    assert summary["pearson"] == pytest.approx(1.0)
    assert summary["spearman"] == pytest.approx(1.0)
    constant = copy.deepcopy(records)
    for record in constant:
        record["final_metrics"]["mean_imagined_return"] = 1.0
    assert imagined_real_alignment(constant)["pearson"] is None
    assert correlation([1.0, 1.0], [1.0, 2.0]) is None


def test_actor_number_summary_rejects_unfrozen_world_model():
    records = [
        actor_record(TASKS[0], ARMS[0], 0, 0.1, 1.0),
        actor_record(TASKS[1], ARMS[1], 1, 0.2, 2.0),
    ]
    summary = actor_number_summary(records, expected_cells=4)
    assert summary["completed_cells"] == 2
    assert summary["completion_fraction"] == 0.5
    broken = copy.deepcopy(records)
    broken[0]["world_model_parameter_delta"] = 1e-6
    with pytest.raises(ValueError, match="preserve"):
        actor_number_summary(broken, expected_cells=4)


def test_report_validation_and_coverage_negative_control():
    report = complete_report()
    validate_report(report, forbidden_root="/work/pilot")
    verify_metric_coverage(
        report,
        metrics=("action_ranking", "gradient_fidelity"),
        tasks=TASKS,
        arms=ARMS,
    )
    missing = copy.deepcopy(report)
    missing["metrics"]["gradient_fidelity"].pop()
    with pytest.raises(ValueError, match="coverage"):
        verify_metric_coverage(
            missing,
            metrics=("gradient_fidelity",),
            tasks=TASKS,
            arms=ARMS,
        )
    contaminated = copy.deepcopy(report)
    contaminated["diagnostic_output_root"] = "/work/pilot/diagnostics"
    with pytest.raises(ValueError, match="inside"):
        validate_report(contaminated, forbidden_root="/work/pilot")
