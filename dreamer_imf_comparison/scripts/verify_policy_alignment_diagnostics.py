#!/usr/bin/env python3
"""Dependency-light positive and negative controls for diagnostic summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "dreamer_imf_compare" / "policy_alignment_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("policy_alignment_diagnostics", MODULE)
assert SPEC is not None and SPEC.loader is not None
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_raises(function, message_fragment: str) -> None:
    try:
        function()
    except ValueError as error:
        require(message_fragment in str(error), f"wrong negative-control error: {error}")
    else:
        raise AssertionError("negative control unexpectedly passed")


def main() -> None:
    targets = np.array([[1.0, 2.0], [2.0, 4.0]])
    samples = np.stack([targets, targets])
    error = diagnostics.actor_visited_model_error(
        samples,
        np.zeros((2, 2)),
        np.ones((2, 2)),
        targets,
        np.zeros(2),
        np.ones(2),
        np.ones(2),
    )
    require(error["expected_standardized_observation_mse"] == 0.0, "exact model error")
    require_raises(
        lambda: diagnostics.actor_visited_model_error(
            samples,
            np.zeros((2, 2)),
            np.ones((2, 2)),
            targets,
            np.zeros(2),
            np.ones(2),
            np.array([1.0, 0.0]),
        ),
        "standard deviation",
    )

    truth = np.array([[0.0, 1.0, 3.0], [3.0, 2.0, 0.0]])
    perfect = diagnostics.action_ranking_metrics(truth, truth)
    reversed_result = diagnostics.action_ranking_metrics(-truth, truth)
    require(perfect["pairwise_accuracy"] == 1.0, "perfect ranking")
    require(reversed_result["pairwise_accuracy"] == 0.0, "reversed ranking control")

    true_gradient = np.array([[1.0, 2.0], [-2.0, 1.0]])
    aligned = diagnostics.gradient_fidelity_metrics(2.0 * true_gradient, true_gradient)
    opposed = diagnostics.gradient_fidelity_metrics(-true_gradient, true_gradient)
    require(abs(aligned["mean_cosine_similarity"] - 1.0) < 1e-12, "aligned gradient")
    require(abs(opposed["mean_cosine_similarity"] + 1.0) < 1e-12, "opposed gradient")
    require(opposed["component_sign_accuracy"] == 0.0, "gradient sign control")

    records = []
    for index, value in enumerate((0.1, 0.2, 0.4)):
        records.append(
            {
                "normalized_episode_return_mean": value,
                "final_metrics": {"mean_imagined_return": 10.0 * value},
            }
        )
    alignment = diagnostics.imagined_real_alignment(records)
    require(abs(alignment["pearson"] - 1.0) < 1e-12, "imagined-real correlation")
    for record in records:
        record["final_metrics"]["mean_imagined_return"] = 1.0
    require(diagnostics.imagined_real_alignment(records)["pearson"] is None, "constant control")
    print("POLICY_DIAGNOSTIC_CONTROLS_VERIFIED")


if __name__ == "__main__":
    main()
