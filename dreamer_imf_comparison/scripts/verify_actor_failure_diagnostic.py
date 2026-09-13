#!/usr/bin/env python3
"""Fail-closed verifier for the actor-failure causal diagnostic."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from dreamer_imf_compare import actor_failure_diagnostic as study
def verify_library() -> None:
    root = Path(__file__).resolve().parents[2]
    suites = []
    for name in ("test_actor_failure_diagnostic", "test_actor_repairs"):
        path = root / "imf_dreamer_jax" / "tests" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_actor_failure_{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load test module {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suites.append(unittest.defaultTestLoader.loadTestsFromModule(module))
    suite = unittest.TestSuite(suites)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("ACTOR_FAILURE_OVERRIDE_LIBRARY_VERIFIED")


def verify_contract() -> None:
    cells = study.build_cell_matrix()
    study.validate_cell_matrix(cells)
    if (
        len(cells) != 49
        or sum(cell["stage"] == "actor" for cell in cells) != 48
        or cells[0]["arm"] != "random_policy"
    ):
        raise ValueError("diagnostic matrix does not have exactly 49 cells")
    expected = {
        (arm, seed, actor_seed, horizon)
        for arm in study.ARMS
        for seed in study.WORLD_MODEL_SEEDS
        for actor_seed in study.ACTOR_SEEDS
        for horizon in study.HORIZONS
    }
    observed = {
        (cell["arm"], cell["world_model_seed"], cell["actor_seed"], cell["horizon"])
        for cell in cells if cell["stage"] == "actor"
    }
    if observed != expected:
        raise ValueError("diagnostic factorial is incomplete")
    seeds = study.shared_evaluation_seeds(50)
    if len(seeds) != len(set(seeds)) or seeds != study.shared_evaluation_seeds(50):
        raise ValueError("shared evaluation scenarios are not deterministic and unique")
    print("ACTOR_FAILURE_DIAGNOSTIC_CONTRACT_VERIFIED")


def _fake_manifest(cells: list[dict]) -> dict:
    return {
        "source_commit": "a" * 40,
        "manifest_sha256": "b" * 64,
        "actor_updates": 3,
        "preparation_updates": 2,
        "evaluation_episodes": 2,
        "shared_evaluation_seeds": [17, 19],
        "effect_thresholds": {
            "normalized_return_mean": 0.005,
            "world_model_seed_fraction": 1.0,
        },
        "synthetic_competence": {
            "required_mean_absolute_error": 0.25,
            "required_fraction_improved_over_bc": 0.75,
            "required_mean_improvement_over_bc": 0.05,
        },
        "independent_unit": "world_model_seed",
        "actor_seed_role": "conditional_optimization_variance_only",
        "cells": cells,
    }


def _fake_raw(action_value: float, returns: tuple[float, float]) -> dict[str, np.ndarray]:
    rewards = np.zeros((2, 3), dtype=np.float64)
    rewards[:, 0] = np.asarray(returns)
    return {
        "actions": np.full((2, 3, 2), action_value, dtype=np.float32),
        "rewards": rewards,
        "continuations": np.ones((2, 3), dtype=np.float64),
        "is_last": np.asarray([[False, False, True]] * 2, dtype=np.bool_),
        "lengths": np.asarray([3, 3], dtype=np.int32),
        "evaluation_seeds": np.asarray([17, 19], dtype=np.uint32),
    }


def _fake_result(cell: dict, manifest: dict, normalized: float) -> tuple[dict, dict]:
    raw = _fake_raw(0.5 if cell["arm"] == "synthetic_action_reward_pmpo" else 0.0,
                    (normalized * 1000.0, normalized * 1000.0))
    metrics = study._trace_metrics((normalized * 1000.0,) * 2, raw)
    result = {
        "schema_version": study.RESULT_SCHEMA,
        "stage": cell["stage"],
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": cell["index"],
        "arm": cell["arm"],
        "world_model_seed": cell["world_model_seed"],
        "actor_seed": cell["actor_seed"],
        "horizon": cell["horizon"],
        "actor_updates": 0 if cell["arm"] in ("random_policy", "bc_no_pmpo") else 3,
        "preparation_updates": 0 if cell["arm"] == "random_policy" else 2,
        "metrics": metrics,
        "raw_action_traces_sha256": "unused-in-pure-self-test",
        "wall_seconds": 1.0,
        "slurm_job_id": "self-test",
        "runtime": {"backend": "cpu"},
    }
    if cell["stage"] == "actor":
        result.update({
            "source_checkpoint_sha256": "c" * 64,
            "world_model_parameter_delta": 0.0,
            "behavior_prior_frozen": True,
            "final_behavior_cloning_loss": 1.0,
            "final_replay_critic_loss": 1.0,
            "final_metrics": None if cell["arm"] == "bc_no_pmpo" else {},
            "checkpoint_sha256": "d" * 64,
        })
    return result, raw


def verify_self_test() -> None:
    cells = study.build_cell_matrix()
    manifest = _fake_manifest(cells)
    rows = []
    for cell in cells:
        values = {
            "random_policy": 0.01,
            "bc_no_pmpo": 0.02,
            "base_mtp_pmpo": 0.03,
            "residual_pmpo": 0.01,
            "analytic_reward_pmpo": 0.06,
            "analytic_reward_unit_continuation_pmpo": 0.08,
            "synthetic_action_reward_pmpo": 0.02,
        }
        result, raw = _fake_result(cell, manifest, values[cell["arm"]])
        study.validate_cell_result(result, cell, manifest, raw)
        rows.append(result)
    decision = study.diagnostic_decision(rows, manifest)
    if not decision["actor_harness_competence_passed"]:
        raise ValueError("positive synthetic control did not pass")
    forged_cells = copy.deepcopy(cells)
    forged_cells.pop()
    try:
        study.validate_cell_matrix(forged_cells)
    except ValueError:
        pass
    else:
        raise ValueError("incomplete factorial was accepted")
    forged, raw = _fake_result(cells[1], manifest, 0.02)
    forged["world_model_parameter_delta"] = 1e-9
    try:
        study.validate_cell_result(forged, cells[1], manifest, raw)
    except ValueError:
        pass
    else:
        raise ValueError("mutated world model was accepted")
    forged, raw = _fake_result(cells[1], manifest, 0.02)
    raw["evaluation_seeds"][0] = 23
    try:
        study.validate_cell_result(forged, cells[1], manifest, raw)
    except ValueError:
        pass
    else:
        raise ValueError("non-shared scenario evidence was accepted")
    print("ACTOR_FAILURE_DIAGNOSTIC_SELF_TEST_VERIFIED")


def verify_final(root: Path) -> dict:
    report = study.validate_complete_evidence(root)
    print("ACTOR_FAILURE_DIAGNOSTIC_FINAL_VERIFIED")
    return report


def verify_decision(root: Path) -> None:
    report = study.validate_complete_evidence(root)
    decision = report["decision"]
    if (
        decision.get("independent_unit") != "world_model_seed"
        or decision.get("actor_seed_role") != "conditional_optimization_variance_only"
        or not decision.get("classification")
        or not decision.get("next_action")
    ):
        raise ValueError("diagnostic decision overstates or omits its replicate contract")
    print(json.dumps(decision, sort_keys=True))
    print("ACTOR_FAILURE_CAUSAL_LADDER_DECIDED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("library", "contract", "self-test", "final", "decision"))
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    if args.mode == "library":
        verify_library()
    elif args.mode == "contract":
        verify_contract()
    elif args.mode == "self-test":
        verify_self_test()
    else:
        if args.evidence_root is None:
            parser.error("--evidence-root is required")
        if args.mode == "final":
            verify_final(args.evidence_root)
        else:
            verify_decision(args.evidence_root)


if __name__ == "__main__":
    main()
