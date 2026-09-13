#!/usr/bin/env python3
"""Fail-closed verification for the continuous actor-repair study."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dreamer_imf_compare import actor_repair_study as study
from dreamer_imf_compare import matched_objective_benchmark as benchmark


def verify_contract() -> None:
    cells = study.build_cell_matrix()
    study.validate_cell_matrix(cells)
    expected = {
        "random_policy": 1,
        "exact_bandit_h1": 6,
        "exact_finite_no_bootstrap": 12,
        "exact_reward_learned_critic": 12,
        "analytic_reward_learned_dynamics": 12,
        "learned_reward_learned_dynamics": 12,
        "safe_mpo_bandit": 2,
        "gradient_oracle": 2,
    }
    observed = {
        stage: sum(cell["stage"] == stage for cell in cells) for stage in expected
    }
    if len(cells) != 59 or observed != expected:
        raise ValueError("actor-repair factorial is incomplete")
    for stage in study.PURE_STAGES + study.WORLD_STAGES:
        if {
            cell["behavior_kl_scale"] for cell in cells if cell["stage"] == stage
        } != set(study.BETAS):
            raise ValueError(f"KL ablation is incomplete for {stage}")
    if study.PRIMARY_BETA != 0.1:
        raise ValueError("primary beta changed after registration")
    seeds = study.shared_evaluation_seeds(25)
    if seeds != study.shared_evaluation_seeds(25) or len(seeds) != len(set(seeds)):
        raise ValueError("shared evaluation scenarios are not deterministic and unique")
    print("ACTOR_REPAIR_STUDY_CONTRACT_VERIFIED")


def _fake_result(cell: dict, manifest: dict) -> dict:
    stage = cell["stage"]
    if stage == "random_policy":
        metrics = {
            "episode_returns": [10.0, 10.0],
            "normalized_return_mean": 0.01,
        }
    elif stage in study.PURE_STAGES or stage == "safe_mpo_bandit":
        metrics = {"action_target_mean_absolute_error": 0.1}
    elif stage == "gradient_oracle":
        metrics = {
            "gradient_cosine": 0.99,
            "gradient_sign_agreement": 0.95,
            "classification": "policy_gradient_estimator_verified",
        }
    else:
        metrics = {
            "episode_returns": [100.0, 100.0],
            "normalized_return_mean": 0.1,
        }
    result = {
        "schema_version": study.RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": cell["index"],
        "stage": stage,
        "world_model_seed": cell["world_model_seed"],
        "actor_seed": cell["actor_seed"],
        "horizon": cell["horizon"],
        "behavior_kl_scale": cell["behavior_kl_scale"],
        "actor_updates": 0 if stage in ("random_policy", "gradient_oracle") else 3,
        "preparation_updates": 2 if stage in study.WORLD_STAGES else 0,
        "metrics": metrics,
        "world_model_parameter_delta": 0.0 if stage in study.WORLD_STAGES else None,
        "raw_action_traces_sha256": None,
        "wall_seconds": 1.0,
        "slurm_job_id": "self-test",
        "runtime": {"backend": "cpu"},
    }
    if stage in study.WORLD_STAGES:
        result.update(
            {
                "source_checkpoint_sha256": "c" * 64,
                "final_behavior_cloning_loss": 1.0,
                "final_replay_critic_loss": 1.0,
            }
        )
    return result


def verify_self_test() -> None:
    cells = study.build_cell_matrix()
    forged_cells = copy.deepcopy(cells)
    forged_cells.pop()
    try:
        study.validate_cell_matrix(forged_cells)
    except ValueError:
        pass
    else:
        raise ValueError("incomplete actor-repair factorial was accepted")
    manifest = {
        "source_commit": "a" * 40,
        "manifest_sha256": "b" * 64,
        "primary_decision_beta": study.PRIMARY_BETA,
        "sensitivity_only_betas": [0.0, 0.3],
        "causal_order": list(study.PURE_STAGES + study.WORLD_STAGES),
        "decision_thresholds": {
            "exact_action_mean_absolute_error": 0.25,
            "normalized_return_over_random": 0.005,
        },
        "independent_unit": "world_model_seed",
        "actor_seed_role": "conditional_optimization_variance_only",
    }
    rows = [_fake_result(cell, manifest) for cell in cells]
    decision = study.causal_decision(rows, manifest)
    if (
        decision["classification"] != "all_preregistered_actor_rungs_passed"
        or decision["first_failing_rung"] is not None
    ):
        raise ValueError("positive causal-ladder fixture did not pass")
    incomplete = [
        row
        for row in rows
        if not (
            row["stage"] == "learned_reward_learned_dynamics"
            and row["behavior_kl_scale"] == study.PRIMARY_BETA
        )
    ]
    try:
        study.causal_decision(incomplete, manifest)
    except ValueError:
        pass
    else:
        raise ValueError("incomplete causal ladder was accepted")
    pure_cell = next(cell for cell in cells if cell["stage"] == "exact_bandit_h1")
    pure_result = _fake_result(pure_cell, manifest)
    study.validate_cell_result(pure_result, pure_cell, manifest)
    pure_result["metrics"]["action_target_mean_absolute_error"] = float("nan")
    try:
        study.validate_cell_result(pure_result, pure_cell, manifest)
    except ValueError:
        pass
    else:
        raise ValueError("non-finite actor evidence was accepted")
    print("ACTOR_REPAIR_STUDY_SELF_TEST_VERIFIED")


def verify_gradient(output_root: Path) -> None:
    report = json.loads((output_root / "smoke_report.json").read_text())
    gradient = report["gradient_oracle"]
    if (
        gradient["classification"] not in (
            "policy_gradient_estimator_verified",
            "policy_gradient_estimator_mismatch",
        )
        or not np.isfinite(gradient["gradient_cosine"])
        or not np.isfinite(gradient["gradient_sign_agreement"])
    ):
        raise ValueError("gradient diagnostic did not emit a valid decision")
    print(json.dumps(gradient, sort_keys=True))
    print("ACTOR_REPAIR_GRADIENT_DECIDED")


def _load_test(path: Path, name: str) -> unittest.TestSuite:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load test module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def verify_regression() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "imf_dreamer_jax/tests/test_actor_repairs.py",
        root / "imf_dreamer_jax/tests/test_dreamer4_actor_ablations.py",
        root / "imf_dreamer_jax/tests/test_actor_failure_diagnostic.py",
        root / "imf_dreamer_jax/tests/test_continuous_actor_repair.py",
        root / "dreamer_imf_comparison/tests/test_actor_failure_diagnostic.py",
        root / "dreamer_imf_comparison/tests/test_actor_repair_study.py",
    )
    suite = unittest.TestSuite(
        _load_test(path, f"_actor_repair_{index}")
        for index, path in enumerate(paths)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("ACTOR_REPAIR_REGRESSION_VERIFIED")


def verify_final(output_root: Path) -> None:
    report = study.validate_complete_evidence(output_root)
    print(json.dumps(report["decision"], sort_keys=True))
    print("ACTOR_REPAIR_STUDY_FINAL_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("contract", "self-test", "gradient", "regression", "final")
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.mode == "contract":
        verify_contract()
    elif args.mode == "self-test":
        verify_self_test()
    elif args.mode == "regression":
        verify_regression()
    else:
        if args.output_root is None:
            parser.error("--output-root is required")
        if args.mode == "gradient":
            verify_gradient(args.output_root)
        else:
            verify_final(args.output_root)


if __name__ == "__main__":
    main()
