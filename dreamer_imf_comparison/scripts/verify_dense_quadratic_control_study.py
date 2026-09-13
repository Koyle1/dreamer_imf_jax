#!/usr/bin/env python3
"""Verify a locally mirrored dense quadratic-control benchmark evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SEEDS = (211, 223)
ACTOR_HORIZONS = (5, 15)
PROBE_HORIZONS = ("1", "3", "5", "15")
OBJECTIVE_IDENTITY = (
    "exact_dense_quadratic_plus_relative_error_control_h1_to_h15"
)


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _close(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _rows(root: Path, stage: str) -> list[dict[str, Any]]:
    return [_read(path) for path in sorted((root / stage).glob("*/result.json"))]


def _contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read(root / "contract.json")
    manifest = _read(root / "manifest.json")
    digest_payload = dict(manifest)
    recorded_digest = digest_payload.pop("manifest_sha256")
    if _canonical_sha256(digest_payload) != recorded_digest:
        raise ValueError("manifest object digest mismatch")
    if (
        manifest.get("source_commit") != contract.get("source_commit")
        or manifest.get("manifest_sha256") != contract.get("manifest_sha256")
        or manifest.get("objective_identity") != OBJECTIVE_IDENTITY
        or manifest.get("world_model_seeds") != list(SEEDS)
        or manifest.get("imagination_horizons") != list(ACTOR_HORIZONS)
        or manifest.get("trainable_world_subtrees") != ["reward_action_residual"]
    ):
        raise ValueError("contract and manifest do not identify the frozen study")
    objective = manifest.get("objective", {})
    if (
        objective.get("horizons") != list(range(1, 16))
        or objective.get("objective") != "dense_quadratic_control"
        or objective.get("control_scale") != 1.0
        or manifest.get("dense_residual_reference") is None
        or manifest.get("interpretation", {}).get(
            "exact_equivalence_simplification"
        )
        is not True
        or manifest.get("interpretation", {}).get("regret_upper_bound_term")
        is not True
    ):
        raise ValueError("objective freeze contract is invalid")
    return contract, manifest


def _accounting(root: Path, job_ids: Iterable[str]) -> None:
    accounting = _read(root / "accounting.json")
    rows = accounting.get("jobs", {})
    for job_id in job_ids:
        row = rows.get(str(job_id))
        if row is None or row.get("state") != "COMPLETED":
            raise ValueError(f"Slurm job {job_id} is not authenticated complete")


def verify_manifest(root: Path) -> None:
    contract, manifest = _contract(root)
    if (
        manifest.get("residual_updates") != 1000
        or manifest.get("actor_updates") != 3000
        or manifest.get("preparation_updates") != 500
        or manifest.get("evaluation_episodes") != 5
        or not contract.get("remote_source_root")
        or not contract.get("remote_result_root")
    ):
        raise ValueError("matched-compute schedule is not frozen as expected")
    print("DENSE_QUADRATIC_CONTROL_MANIFEST_VERIFIED")


def verify_preflight(root: Path) -> None:
    contract, manifest = _contract(root)
    preflight = _read(root / "preflight.json")
    if (
        preflight.get("status") != "complete"
        or preflight.get("source_commit") != manifest["source_commit"]
        or preflight.get("manifest_sha256") != manifest["manifest_sha256"]
        or preflight.get("full_library_suite_passed") is not True
        or preflight.get("full_comparison_suite_passed") is not True
        or preflight.get("runtime", {}).get("backend") != "gpu"
    ):
        raise ValueError("preflight evidence is incomplete")
    _accounting(root, contract["jobs"]["preflight"])
    print("DENSE_QUADRATIC_CONTROL_PREFLIGHT_VERIFIED")


def verify_residual(root: Path) -> None:
    contract, manifest = _contract(root)
    rows = _rows(root, "residual")
    if sorted(row.get("world_model_seed") for row in rows) != list(SEEDS):
        raise ValueError("residual cells are missing or duplicated")
    for row in rows:
        metrics = row.get("final_metrics", {})
        if (
            row.get("status") != "complete"
            or row.get("source_commit") != manifest["source_commit"]
            or row.get("manifest_sha256") != manifest["manifest_sha256"]
            or row.get("objective") != manifest["objective"]
            or row.get("residual_updates") != manifest["residual_updates"]
            or _finite(row.get("residual_parameter_delta"), "residual delta") <= 0.0
            or any(float(value) != 0.0 for value in row.get("source_parameter_deltas", {}).values())
            or any(float(value) != 0.0 for value in row.get("source_first_moment_deltas", {}).values())
            or any(float(value) != 0.0 for value in row.get("source_second_moment_deltas", {}).values())
        ):
            raise ValueError(f"invalid residual cell for seed {row.get('world_model_seed')}")
        for name in ("total", "uncentered_return", "control", "advantage_running_rms"):
            _finite(metrics.get(name), f"residual metric {name}")
    _accounting(root, contract["jobs"]["residual"])
    print("DENSE_QUADRATIC_CONTROL_RESIDUAL_VERIFIED")


def verify_rollout(root: Path) -> None:
    contract, manifest = _contract(root)
    rows = _rows(root, "evaluation")
    if sorted(row.get("world_model_seed") for row in rows) != list(SEEDS):
        raise ValueError("rollout cells are missing or duplicated")
    for row in rows:
        if (
            row.get("status") != "complete"
            or row.get("source_commit") != manifest["source_commit"]
            or row.get("manifest_sha256") != manifest["manifest_sha256"]
            or set(row.get("metrics_by_horizon", {})) != set(PROBE_HORIZONS)
        ):
            raise ValueError(f"invalid rollout cell for seed {row.get('world_model_seed')}")
        for horizon in PROBE_HORIZONS:
            metrics = row["metrics_by_horizon"][horizon]
            for name in (
                "pairwise_accuracy",
                "top1_accuracy",
                "mean_state_spearman",
                "mean_simulator_regret",
            ):
                _finite(metrics.get(name), f"rollout {horizon} {name}")
        for context in ("posterior", "corrupted", "generated"):
            metrics = row.get("reward_context_metrics", {}).get(context, {})
            _finite(metrics.get("mse"), f"reward {context} MSE")
            _finite(
                metrics.get("offset_zero_mean_calibration_error"),
                f"reward {context} calibration",
            )
    _accounting(root, contract["jobs"]["evaluation"])
    print("DENSE_QUADRATIC_CONTROL_ROLLOUT_VERIFIED")


def verify_actor(root: Path) -> None:
    contract, manifest = _contract(root)
    rows = _rows(root, "actor")
    keys = sorted(
        (row.get("world_model_seed"), row.get("imagination_horizon")) for row in rows
    )
    expected = sorted((seed, horizon) for seed in SEEDS for horizon in ACTOR_HORIZONS)
    if keys != expected:
        raise ValueError("actor cells are missing or duplicated")
    for row in rows:
        returns = row.get("normalized_episode_returns", [])
        if (
            row.get("status") != "complete"
            or row.get("source_commit") != manifest["source_commit"]
            or row.get("manifest_sha256") != manifest["manifest_sha256"]
            or row.get("world_model_frozen") is not True
            or row.get("world_model_parameter_delta") != 0.0
            or row.get("behavior_prior_frozen") is not True
            or len(returns) != manifest["evaluation_episodes"]
        ):
            raise ValueError(
                f"invalid actor cell {row.get('world_model_seed')}, "
                f"h{row.get('imagination_horizon')}"
            )
        measured = sum(_finite(value, "actor return") for value in returns) / len(returns)
        if not _close(measured, row.get("normalized_episode_return_mean")):
            raise ValueError("actor return mean does not match retained episodes")
    _accounting(root, contract["jobs"]["actor"])
    print("DENSE_QUADRATIC_CONTROL_ACTOR_VERIFIED")


def verify_final(root: Path) -> None:
    contract, manifest = _contract(root)
    verify_residual(root)
    verify_rollout(root)
    verify_actor(root)
    report = _read(root / "report.json")
    if (
        report.get("status") != "complete"
        or report.get("source_commit") != manifest["source_commit"]
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("completed_residual_cells") != len(SEEDS)
        or report.get("completed_evaluation_cells") != len(SEEDS)
        or report.get("completed_actor_cells") != len(SEEDS) * len(ACTOR_HORIZONS)
        or report.get("interpretation", {}).get("claim_eligible") is not False
    ):
        raise ValueError("final report is incomplete or overclaims evidence")
    actors = _rows(root, "actor")
    for horizon in ACTOR_HORIZONS:
        measured = sum(
            row["normalized_episode_return_mean"]
            for row in actors
            if row["imagination_horizon"] == horizon
        ) / len(SEEDS)
        recorded = next(
            row["mean_normalized_return"]
            for row in report["actor_groups"]
            if row["imagination_horizon"] == horizon
        )
        if not _close(measured, recorded):
            raise ValueError(f"actor aggregation mismatch at horizon {horizon}")
    evaluations = _rows(root, "evaluation")
    for horizon in PROBE_HORIZONS:
        pairwise = sum(
            row["metrics_by_horizon"][horizon]["pairwise_accuracy"]
            for row in evaluations
        ) / len(SEEDS)
        regret = sum(
            row["metrics_by_horizon"][horizon]["mean_simulator_regret"]
            for row in evaluations
        ) / len(SEEDS)
        if (
            not _close(pairwise, report["probe_groups"][horizon]["pairwise_accuracy_mean"])
            or not _close(regret, report["probe_groups"][horizon]["mean_simulator_regret"])
        ):
            raise ValueError(f"rollout aggregation mismatch at horizon {horizon}")
    transcript = (root / "remote_verification.txt").read_text(encoding="utf-8")
    if "ACTION_REWARD_RESIDUAL_RESULTS_VERIFIED" not in transcript:
        raise ValueError("independent remote verifier success marker is absent")
    all_jobs = [job for jobs in contract["jobs"].values() for job in jobs]
    _accounting(root, all_jobs)
    print("DENSE_QUADRATIC_CONTROL_BENCHMARK_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("manifest", "preflight", "residual", "rollout", "actor", "final")
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    functions = {
        "manifest": verify_manifest,
        "preflight": verify_preflight,
        "residual": verify_residual,
        "rollout": verify_rollout,
        "actor": verify_actor,
        "final": verify_final,
    }
    functions[args.stage](args.evidence_root)


if __name__ == "__main__":
    main()
