#!/usr/bin/env python3
"""Fail-closed independent gates for the actor-gap roadmap study."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for path in (PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dreamer_imf_compare.artifacts import read_json  # noqa: E402
from dreamer_imf_compare import actor_gap_diagnostics  # noqa: E402
from dreamer_imf_compare import actor_gap_model_training  # noqa: E402
from dreamer_imf_compare import actor_gap_roadmap_study as study  # noqa: E402
from imf_dreamer_jax import control_aware_imf  # noqa: E402
from imf_dreamer_jax import robust_flowmpc  # noqa: E402

COMMANDS = (
    "self-test",
    "implementation",
    "gpu",
    "library-suite",
    "comparison-suite",
    "preflight",
    "calibration",
    "diagnostics",
    "models",
    "evaluations",
    "final",
)

REQUIRED_CORE_APIS = (
    "authenticate_dependency",
    "build_manifest",
    "validate_manifest",
    "write_manifest",
    "write_preflight_marker",
    "write_preflight_gate",
    "validate_preflight_marker",
    "write_submission_map",
    "validate_submission_map",
    "run_calibration",
    "verify_calibration",
    "prime_diagnostic_a0_cache",
    "prime_diagnostic_cell",
    "run_diagnostic_cell",
    "seal_evaluation_cache_reader",
    "verify_diagnostic_cell",
    "verify_diagnostic_stage",
    "train_model_cell",
    "verify_model_cell",
    "verify_model_stage",
    "run_evaluation_cell",
    "verify_evaluation_cell",
    "verify_evaluation_stage",
    "finalize",
    "validate_final",
    "self_test",
)

REQUIRED_IMPLEMENTATION_TOKENS = (
    "fit_coverage_calibration(",
    "component_residual_decomposition(",
    "independent_noise_diagnostics(",
    "CounterfactualProvenance.",
    "init_reference_actor_state(",
    "begin_actor_adaptation_step(",
    "finish_actor_adaptation_step(",
    "backtrack_actor_proposal(",
    "held_out_noise_acceptance(",
    "gradient_action_sequence_search(",
    "cem_action_sequence_search(",
    "ensemble_relative_risk_score(",
    "policy_density_tilt_weights(",
    "cvaml_compatible_bellman_residual_loss(",
    "action_chunk_endpoint_imf_loss(",
    "require_canonical_training_schedule(",
    "creation_submission_authorization",
    "verification_submission_authorization",
    "creation_submission_receipt",
    "verification_submission_receipt",
    "prime-diagnostic-cell",
    "prime-diagnostic-a0-cache",
    "seal-diagnostic-cache-reader",
    "AGR_DIAGNOSTIC_A0_CACHE_FINGERPRINT",
    "AGR_DIAGNOSTIC_CACHE_FINGERPRINT",
    "a0_only_writer_prepass_both_actor_seeds",
    "separate_process_discarded_complete_diagnostic_cache_reader",
    "verify-model-cell",
    "strict replay must execute in a fresh Python process",
    "held_until_receipt_persisted_before_release",
    "--kill-on-invalid-dep=yes",
)


def verify_implementation() -> None:
    missing_apis = [
        name for name in REQUIRED_CORE_APIS if not callable(getattr(study, name, None))
    ]
    source = "\n".join(
        inspect.getsource(module)
        for module in (
            study,
            actor_gap_model_training,
            actor_gap_diagnostics,
            control_aware_imf,
            robust_flowmpc,
        )
    )
    source += "\n" + (
        PROJECT / "cluster/actor_gap_roadmap/submit_actor_gap_roadmap.py"
    ).read_text(encoding="utf-8")
    source += "\n" + (
        PROJECT / "cluster/actor_gap_roadmap/actor_gap_roadmap.sbatch"
    ).read_text(encoding="utf-8")
    source += "\n" + (PROJECT / "scripts/run_actor_gap_roadmap_study.py").read_text(
        encoding="utf-8"
    )
    missing_tokens = [
        token for token in REQUIRED_IMPLEMENTATION_TOKENS if token not in source
    ]
    if missing_apis or missing_tokens:
        raise ValueError(
            "roadmap implementation is incomplete: "
            f"missing_apis={missing_apis}, missing_tokens={missing_tokens}"
        )
    if (
        len(study.WORLD_SEEDS) != 3
        or len(study.ACTOR_SEEDS) != 2
        or len(study.MODEL_FAMILIES) != 5
        or len(study.REFERENCE_ARMS) != 2
        or len(study.FRESH_ARMS) != 15
        or study.ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS != 10
    ):
        raise ValueError("roadmap factorial constants differ from the frozen design")
    print("ACTOR_GAP_ROADMAP_IMPLEMENTATION_VERIFIED")


def verify_preflight(output_root: Path) -> dict[str, Any]:
    manifest = read_json(output_root / "manifest.json")
    study.validate_manifest(manifest)
    dependency = study.authenticate_dependency(manifest["dependency_root"])
    marker = study.validate_preflight_marker(output_root)
    if (
        manifest.get("claim_eligible") is not False
        or manifest.get("evidence_class")
        != "exploratory_single_task_actor_gap_factorial"
        or manifest.get("dependency_source_commit") != dependency["source_commit"]
        or len(manifest.get("diagnostic_cells", ())) != 3
        or len(manifest.get("model_cells", ())) != 21
        or len(manifest.get("evaluation_cells", ())) != 90
        or manifest.get("calibration", {}).get("partition") != "training_replay_only"
        or marker.get("status") != "verified"
        or marker.get("manifest_sha256") != manifest.get("manifest_sha256")
    ):
        raise ValueError("roadmap preflight contract is incomplete")
    print(manifest["manifest_sha256"])
    print("ACTOR_GAP_ROADMAP_PREFLIGHT_CONTRACT_VERIFIED")
    return manifest


def _record_gate(output_root: Path | None, gate: str, evidence: dict[str, Any]) -> None:
    if output_root is not None:
        study.write_preflight_gate(output_root, gate, evidence)


def _run_unittest_suite(directory: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(directory),
        "-p",
        "test_*.py",
    ]
    completed = subprocess.run(
        command,
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"unit-test suite failed with exit {completed.returncode}")
    matches = re.findall(r"Ran ([0-9]+) tests?", completed.stdout + completed.stderr)
    if len(matches) != 1 or int(matches[0]) <= 0:
        raise ValueError("unit-test suite did not report one positive test count")
    return int(matches[0])


def _require_root(parser: argparse.ArgumentParser, arguments: Any) -> Path:
    if arguments.output_root is None:
        parser.error(f"{arguments.command} requires --output-root")
    return arguments.output_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--output-root", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "self-test":
        study.self_test()
        _record_gate(
            arguments.output_root,
            "self_test",
            {"contract_self_test_completed": True},
        )
        print("ACTOR_GAP_ROADMAP_CONTRACT_VERIFIED")
    elif arguments.command == "implementation":
        verify_implementation()
        _record_gate(
            arguments.output_root,
            "implementation",
            {
                "required_core_apis": len(REQUIRED_CORE_APIS),
                "required_implementation_tokens": len(REQUIRED_IMPLEMENTATION_TOKENS),
            },
        )
    elif arguments.command == "gpu":
        import jax

        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise ValueError("actor-gap roadmap cells require exactly one GPU")
        _record_gate(
            arguments.output_root,
            "single_gpu",
            {
                "device_count": 1,
                "platform": "gpu",
                "device_kind": str(devices[0].device_kind),
            },
        )
        print("ACTOR_GAP_ROADMAP_GPU_VERIFIED")
    elif arguments.command in {"library-suite", "comparison-suite"}:
        if arguments.command == "library-suite":
            gate = "library_suite"
            directory = WORKSPACE / "imf_dreamer_jax/tests"
            token = "ACTOR_GAP_ROADMAP_LIBRARY_SUITE_VERIFIED"
        else:
            gate = "comparison_suite"
            directory = PROJECT / "tests"
            token = "ACTOR_GAP_ROADMAP_COMPARISON_SUITE_VERIFIED"
        tests_run = _run_unittest_suite(directory)
        _record_gate(
            arguments.output_root,
            gate,
            {"tests_run": tests_run, "failures": 0, "errors": 0},
        )
        print(token)
    elif arguments.command == "preflight":
        verify_preflight(_require_root(parser, arguments))
    elif arguments.command == "calibration":
        marker = study.verify_calibration(_require_root(parser, arguments))
        print(marker["marker_sha256"])
        print("ACTOR_GAP_ROADMAP_CALIBRATION_VERIFIED")
    elif arguments.command == "diagnostics":
        marker = study.verify_diagnostic_stage(_require_root(parser, arguments))
        print(marker["marker_sha256"])
        print("ACTOR_GAP_ROADMAP_DIAGNOSTICS_VERIFIED")
    elif arguments.command == "models":
        marker = study.verify_model_stage(_require_root(parser, arguments))
        print(marker["marker_sha256"])
        print("ACTOR_GAP_ROADMAP_MODELS_VERIFIED")
    elif arguments.command == "evaluations":
        marker = study.verify_evaluation_stage(_require_root(parser, arguments))
        print(marker["marker_sha256"])
        print("ACTOR_GAP_ROADMAP_EVALUATIONS_VERIFIED")
    else:
        report = study.validate_final(_require_root(parser, arguments))
        print(report["report_sha256"])
        print("ACTOR_GAP_ROADMAP_STUDY_FINAL_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
