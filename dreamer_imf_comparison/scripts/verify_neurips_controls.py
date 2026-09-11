#!/usr/bin/env python3
"""Independent contract, test, and real-smoke oracles for NeurIPS controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json  # noqa: E402
from dreamer_imf_compare.matched_objective_protocol import (  # noqa: E402
    read_matched_objective_protocol,
)
from dreamer_imf_compare.neurips_controls import (  # noqa: E402
    _profile_arm_specs,
    build_controls_source_manifest,
    expected_arm_specs,
    read_controls_protocol,
    run_controls_all,
    validate_controls_protocol,
    verify_controls_output,
)


def verify_contract() -> dict:
    protocol = read_controls_protocol(PROJECT / "neurips_controls_protocol.json")
    parent_protocol = read_matched_objective_protocol(
        PROJECT / "matched_objective_protocol.json"
    )
    validate_controls_protocol(protocol, parent_protocol)
    arms = expected_arm_specs(protocol)
    factorial = [arm for arm in arms if arm["family"] == "trajectory_imf"]
    if len(arms) != 33 or len(factorial) != 28:
        raise AssertionError("controls protocol did not expand to 33/28 arms")
    if len([arm for arm in arms if arm["arm_id"] == "trajectory_imf"]) != 1:
        raise AssertionError("proposed factorial cell is absent or duplicated")
    executed_arm_counts = {
        profile: len(_profile_arm_specs(protocol, profile))
        for profile in ("smoke", "development", "confirmatory")
    }
    if executed_arm_counts != {"smoke": 33, "development": 33, "confirmatory": 6}:
        raise AssertionError("profile-specific controls arm scope drifted")
    run_counts = dict(executed_arm_counts)
    run_counts["development"] += sum(
        len(protocol["development_selection"]["candidate_grid"][family]) - 1
        for family in protocol["development_selection"]["selected_families"]
    )
    matrix_cell_counts = {}
    for profile, row in protocol["profiles"].items():
        matrix_cell_counts[profile] = len(row["tasks"]) + (
            len(row["tasks"])
            * len(row["world_model_seeds"])
            * len(row["run_budget_tracks"])
            * run_counts[profile]
            * (2 + len(row["actor_seeds_nested_within_world_model_seed"]))
        )
    if matrix_cell_counts != {
        "smoke": 100,
        "development": 938,
        "confirmatory": 3606,
    }:
        raise AssertionError("registered controls matrix size drifted")
    documentation = (PROJECT / "NEURIPS_CONTROLS.md").read_text(encoding="utf-8")
    for phrase in (
        "not a reproduction of MeLISA",
        "can change, replace, rescue, or fail",
        "2 × 2 × (2³ − 1)",
        "engineering test",
        "raw endpoint-certificate slice",
    ):
        if phrase not in documentation:
            raise AssertionError(f"controls documentation lacks boundary phrase: {phrase}")
    return {
        "arm_count": len(arms),
        "factorial_arm_count": len(factorial),
        "executed_arm_counts": executed_arm_counts,
        "registered_matrix_cell_counts": matrix_cell_counts,
        "primary_gate_accessed": False,
    }


def verify_tests() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_neurips_controls"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("NeurIPS controls tests failed")
    return {"tests_run": result.testsRun, "failures": 0, "errors": 0}


def verify_smoke(output: Path | None) -> dict:
    protocol = read_controls_protocol(PROJECT / "neurips_controls_protocol.json")
    parent_protocol = read_matched_objective_protocol(
        PROJECT / "matched_objective_protocol.json"
    )
    if output is None:
        # Include both protocol and live source identity in the default path so
        # a stale finalized smoke is never silently reused after source edits.
        source = build_controls_source_manifest(WORKSPACE)
        output = Path("/private/tmp") / (
            "trajectory-imf-neurips-controls-smoke-"
            + source["source_sha256"][:12]
        )
    summary = run_controls_all(
        protocol,
        parent_protocol,
        "smoke",
        output,
        workspace=WORKSPACE,
    )
    independently_verified = verify_controls_output(output, workspace=WORKSPACE)
    if summary != independently_verified:
        raise AssertionError("controls smoke completion and independent verification differ")
    if independently_verified["arm_count"] != 33:
        raise AssertionError("controls smoke did not execute every registered arm")
    return {"output": str(output), **independently_verified}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="store_true")
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    selected = sum((arguments.contract, arguments.tests, arguments.smoke))
    if selected != 1:
        parser.error("select exactly one of --contract, --tests, or --smoke")
    if arguments.contract:
        print(json.dumps(verify_contract(), indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_CONTRACT_VERIFIED")
    elif arguments.tests:
        print(json.dumps(verify_tests(), indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_TESTS_VERIFIED")
    else:
        output = Path(arguments.output).resolve() if arguments.output else None
        print(json.dumps(verify_smoke(output), indent=2, sort_keys=True))
        print("NEURIPS_CONTROLS_SMOKE_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
