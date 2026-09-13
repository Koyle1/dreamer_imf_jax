#!/usr/bin/env python3
"""Independent verifier for the action-reward residual study."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess

import numpy as np

from dreamer_imf_compare.artifacts import read_json
from dreamer_imf_compare import action_reward_residual_study as study
from dreamer_imf_compare import matched_objective_benchmark as benchmark


def _settings(manifest: dict) -> dict:
    return {
        "residual_updates": manifest["residual_updates"],
        "actor_updates": manifest["actor_updates"],
        "preparation_updates": manifest["preparation_updates"],
        "evaluation_episodes": manifest["evaluation_episodes"],
    }


def verify_manifest(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    expected = study.build_manifest(
        manifest["mtp_root"], root, **_settings(manifest)
    )
    if manifest != expected:
        raise ValueError("manifest is not the canonical frozen design")
    if (
        manifest["status"] != "frozen_before_execution"
        or manifest["world_model_seeds"] != list(study.WORLD_MODEL_SEEDS)
        or manifest["imagination_horizons"] != list(study.HORIZONS)
        or manifest["trainable_world_subtrees"] != ["reward_action_residual"]
        or manifest["interpretation"]["trajectory_model_frozen"] is not True
        or manifest["interpretation"]["base_reward_head_frozen"] is not True
    ):
        raise ValueError("manifest design or freeze contract is invalid")
    for source in manifest["source_artifacts"]:
        checks = (
            (source["checkpoint"], source["checkpoint_sha256"]),
            (source["source_schedule"], source["source_schedule_sha256"]),
            (source["test_probe_bank"], source["test_probe_bank_file_sha256"]),
            (source["train_probe_bank"], source["train_probe_bank_file_sha256"]),
        )
        for path, digest in checks:
            if benchmark.file_sha256(path) != digest:
                raise ValueError(f"source artifact digest changed: {path}")
    print("ACTION_REWARD_RESIDUAL_MANIFEST_VERIFIED")
    return manifest


def _preflight_evidence_complete(preflight: dict, manifest: dict) -> bool:
    """Validate the fields emitted by ``benchmark.runtime_fingerprint``."""
    return bool(
        preflight.get("status") == "complete"
        and preflight.get("source_commit") == manifest["source_commit"]
        and preflight.get("manifest_sha256") == manifest["manifest_sha256"]
        and preflight.get("full_library_suite_passed") is True
        and preflight.get("full_comparison_suite_passed") is True
        and preflight.get("runtime", {}).get("backend") == "gpu"
    )


def verify_preflight(root: Path) -> dict:
    manifest = verify_manifest(root)
    preflight = read_json(root / "preflight.json")
    if not _preflight_evidence_complete(preflight, manifest):
        raise ValueError("preflight evidence is incomplete")
    job_id = str(preflight["slurm_job_id"])
    output = subprocess.check_output(
        ["sacct", "-X", "-n", "-j", job_id, "-o", "JobIDRaw,State", "-P"],
        text=True,
    )
    states = {
        row.split("|", 1)[0]: row.split("|", 2)[1]
        for row in output.splitlines()
        if "|" in row
    }
    if states.get(job_id) != "COMPLETED":
        raise ValueError(f"preflight Slurm job is not complete: {states}")
    print("ACTION_REWARD_RESIDUAL_PREFLIGHT_VERIFIED")
    return preflight


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def verify_results(root: Path) -> dict:
    manifest = verify_manifest(root)
    residual_rows = []
    evaluation_rows = []
    actor_rows = []
    for seed in study.WORLD_MODEL_SEEDS:
        residual_directory = root / "residual" / f"seed-{seed}"
        residual = read_json(residual_directory / "result.json")
        if (
            residual.get("status") != "complete"
            or residual.get("source_commit") != manifest["source_commit"]
            or residual.get("manifest_sha256") != manifest["manifest_sha256"]
            or residual.get("world_model_seed") != seed
            or residual.get("residual_parameter_delta", 0.0) <= 0.0
            or any(residual.get("source_parameter_deltas", {}).values())
            or any(residual.get("source_first_moment_deltas", {}).values())
            or any(residual.get("source_second_moment_deltas", {}).values())
            or benchmark.file_sha256(residual_directory / "checkpoint.pkl")
            != residual["checkpoint_sha256"]
            or benchmark.file_sha256(residual_directory / "schedule.npz")
            != residual["schedule_file_sha256"]
        ):
            raise ValueError(f"invalid residual result for seed {seed}")
        residual_rows.append(residual)
        evaluation_directory = root / "evaluation" / f"seed-{seed}"
        evaluation = read_json(evaluation_directory / "result.json")
        if (
            evaluation.get("status") != "complete"
            or evaluation.get("world_model_checkpoint_sha256") != residual["checkpoint_sha256"]
            or set(evaluation.get("metrics_by_horizon", {})) != {"1", "3", "5", "15"}
            or set(evaluation.get("reward_context_metrics", {}))
            != {"posterior", "corrupted", "generated"}
            or benchmark.file_sha256(evaluation_directory / "evaluation_arrays.npz")
            != evaluation["raw_sha256"]
        ):
            raise ValueError(f"invalid evaluation result for seed {seed}")
        evaluation_rows.append(evaluation)
        for horizon in study.HORIZONS:
            actor_directory = root / "actor" / f"seed-{seed}-h{horizon}"
            actor = read_json(actor_directory / "result.json")
            if (
                actor.get("status") != "complete"
                or actor.get("world_model_frozen") is not True
                or actor.get("world_model_parameter_delta") != 0.0
                or actor.get("world_model_checkpoint_sha256") != residual["checkpoint_sha256"]
                or benchmark.file_sha256(actor_directory / "checkpoint.pkl")
                != actor["checkpoint_sha256"]
                or benchmark.file_sha256(actor_directory / "action_traces.npz")
                != actor["raw_action_traces_sha256"]
            ):
                raise ValueError(f"invalid actor result for seed {seed}, horizon {horizon}")
            actor_rows.append(actor)
    report = read_json(root / "report.json")
    if (
        report.get("status") != "complete"
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("completed_residual_cells") != len(study.WORLD_MODEL_SEEDS)
        or report.get("completed_evaluation_cells") != len(study.WORLD_MODEL_SEEDS)
        or report.get("completed_actor_cells") != len(study.WORLD_MODEL_SEEDS) * len(study.HORIZONS)
        or set(report.get("matched_deltas", {})) != {"reward", "probe", "actor"}
    ):
        raise ValueError("final report is incomplete")
    for horizon in study.HORIZONS:
        rows = [row for row in actor_rows if row["imagination_horizon"] == horizon]
        measured = float(np.mean([row["normalized_episode_return_mean"] for row in rows]))
        recorded = next(
            row["mean_normalized_return"]
            for row in report["actor_groups"]
            if row["imagination_horizon"] == horizon
        )
        if not _close(measured, recorded):
            raise ValueError(f"actor aggregation mismatch at horizon {horizon}")
    for horizon in ("1", "3", "5", "15"):
        measured = float(np.mean([
            row["metrics_by_horizon"][horizon]["pairwise_accuracy"]
            for row in evaluation_rows
        ]))
        if not _close(measured, report["probe_groups"][horizon]["pairwise_accuracy_mean"]):
            raise ValueError(f"probe aggregation mismatch at horizon {horizon}")
    for context in ("posterior", "corrupted", "generated"):
        measured = float(np.mean([
            row["reward_context_metrics"][context]["mse"]
            for row in evaluation_rows
        ]))
        if not _close(measured, report["reward_groups"][context]["mean_mse"]):
            raise ValueError(f"reward aggregation mismatch for {context}")
    print("ACTION_REWARD_RESIDUAL_RESULTS_VERIFIED")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "manifest", "results"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        verify_preflight(args.root)
    elif args.mode == "manifest":
        verify_manifest(args.root)
    else:
        verify_results(args.root)


if __name__ == "__main__":
    main()
