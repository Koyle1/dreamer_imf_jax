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
        "centered_root": manifest["centered_residual_root"],
        "sparse_root": manifest["sparse_residual_root"],
        "residual_updates": manifest["residual_updates"],
        "actor_updates": manifest["actor_updates"],
        "preparation_updates": manifest["preparation_updates"],
        "evaluation_episodes": manifest["evaluation_episodes"],
        "reward_objective": manifest["objective"]["objective"],
        "control_scale": manifest["objective"]["control_scale"],
        "dense_reference_root": (
            None
            if manifest.get("dense_residual_reference") is None
            else manifest["dense_residual_reference"]["root"]
        ),
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
        or set(manifest["objective"]) != {
            "horizons",
            "huber_delta",
            "normalization_epsilon",
            "objective",
            "control_scale",
        }
        or manifest["objective"]["horizons"]
        != list(study.DENSE_TRAINING_HORIZONS)
        or manifest.get("objective_identity") not in {
            "uncentered_pseudo_huber_all_prefix_horizons_1_to_15",
            "exact_dense_quadratic_plus_relative_error_control_h1_to_h15",
        }
        or manifest.get("normalization_basis")
        != "centered_target_return_running_rms"
        or manifest.get("dense_probe_design", {}).get("training_horizons")
        != list(study.DENSE_TRAINING_HORIZONS)
        or manifest["dense_probe_design"].get("source_intervention_horizon") != 5
        or manifest["dense_probe_design"].get("legacy_horizon_agreement")
        != [1, 3, 5]
        or manifest["dense_probe_design"].get("tail_action_rule")
        != "recorded_behavior_suffix_common_across_candidates"
        or manifest["interpretation"]["trajectory_model_frozen"] is not True
        or manifest["interpretation"]["base_reward_head_frozen"] is not True
        or manifest["interpretation"].get("single_loss_term") is not True
        or manifest["interpretation"].get("dense_horizon_identification") is not True
    ):
        raise ValueError("manifest design or freeze contract is invalid")
    objective = manifest["objective"]
    if manifest["objective_identity"].startswith("exact_dense_quadratic"):
        if (
            objective["objective"] != "dense_quadratic_control"
            or objective["control_scale"] <= 0.0
            or manifest.get("dense_residual_reference") is None
            or manifest["interpretation"].get("exact_equivalence_simplification")
            is not True
            or manifest["interpretation"].get("regret_upper_bound_term") is not True
        ):
            raise ValueError("dense quadratic-control contract is incomplete")
    elif objective["objective"] != "uncentered_pseudo_huber":
        raise ValueError("pseudo-Huber objective identity mismatch")
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
    centered_report = Path(manifest["centered_residual_root"]) / "report.json"
    if (
        benchmark.file_sha256(centered_report)
        != manifest["centered_residual_report_file_sha256"]
    ):
        raise ValueError("centered residual reference digest changed")
    sparse_report = Path(manifest["sparse_residual_root"]) / "report.json"
    if (
        benchmark.file_sha256(sparse_report)
        != manifest["sparse_residual_report_file_sha256"]
    ):
        raise ValueError("sparse residual reference digest changed")
    dense_reference = manifest.get("dense_residual_reference")
    if dense_reference is not None and (
        benchmark.file_sha256(Path(dense_reference["root"]) / "report.json")
        != dense_reference["report_file_sha256"]
    ):
        raise ValueError("dense residual reference digest changed")
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


def verify_dense_probes(root: Path) -> dict[int, dict]:
    """Authenticate every dense relabeling before residual training starts."""

    manifest = verify_manifest(root)
    dense_probe_rows = {}
    for seed in study.WORLD_MODEL_SEEDS:
        source = study._source_row(manifest, seed)
        dense_probe_directory = root / "dense_probe" / f"seed-{seed}"
        dense_probe = read_json(dense_probe_directory / "result.json")
        dense_probe_path = dense_probe_directory / "probe_bank.npz"
        dense_bank = benchmark.load_npz(dense_probe_path)
        source_bank = benchmark.load_npz(source["train_probe_bank"])
        if (
            dense_probe.get("status") != "complete"
            or dense_probe.get("source_commit") != manifest["source_commit"]
            or dense_probe.get("manifest_sha256") != manifest["manifest_sha256"]
            or dense_probe.get("world_model_seed") != seed
            or dense_probe.get("horizons")
            != list(study.DENSE_TRAINING_HORIZONS)
            or dense_probe.get("legacy_horizon_max_abs_error", math.inf) > 1e-6
            or dense_probe.get("maximum_replay_error", math.inf) > 1e-6
            or benchmark.file_sha256(dense_probe_path)
            != dense_probe.get("probe_bank_file_sha256")
            or benchmark.array_sha256(dense_bank)
            != dense_probe.get("probe_bank_sha256")
            or benchmark.array_sha256(source_bank)
            != source["train_probe_bank_sha256"]
            or study.dense_probe_legacy_max_error(source_bank, dense_bank) > 1e-6
        ):
            raise ValueError(f"invalid dense probe result for seed {seed}")
        dense_probe_rows[seed] = dense_probe
    print("ACTION_REWARD_RESIDUAL_DENSE_PROBES_VERIFIED")
    return dense_probe_rows


def verify_results(root: Path) -> dict:
    manifest = verify_manifest(root)
    dense_probe_rows = verify_dense_probes(root)
    residual_rows = []
    evaluation_rows = []
    actor_rows = []
    for seed in study.WORLD_MODEL_SEEDS:
        dense_probe = dense_probe_rows[seed]
        residual_directory = root / "residual" / f"seed-{seed}"
        residual = read_json(residual_directory / "result.json")
        if (
            residual.get("status") != "complete"
            or residual.get("source_commit") != manifest["source_commit"]
            or residual.get("manifest_sha256") != manifest["manifest_sha256"]
            or residual.get("world_model_seed") != seed
            or residual.get("dense_probe_bank_sha256")
            != dense_probe["probe_bank_sha256"]
            or residual.get("objective") != manifest["objective"]
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
        or report.get("completed_dense_probe_cells") != len(study.WORLD_MODEL_SEEDS)
        or report.get("completed_residual_cells") != len(study.WORLD_MODEL_SEEDS)
        or report.get("completed_evaluation_cells") != len(study.WORLD_MODEL_SEEDS)
        or report.get("completed_actor_cells") != len(study.WORLD_MODEL_SEEDS) * len(study.HORIZONS)
        or set(report.get("matched_deltas", {})) != {"reward", "probe", "actor"}
        or report.get("centered_residual_reference")
        != manifest["centered_residual_reference"]
        or report.get("sparse_residual_reference")
        != manifest["sparse_residual_reference"]
        or report.get("dense_residual_reference")
        != manifest.get("dense_residual_reference")
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
        base = next(
            row for row in manifest["base_mtp_reference"]["actor_groups"]
            if row["imagination_horizon"] == horizon
        )["mean_normalized_return"]
        shortcut = next(
            row for row in manifest["shortcut_reference"]["actor_groups"]
            if row["imagination_horizon"] == horizon
        )["mean_normalized_return"]
        centered = next(
            row for row in manifest["centered_residual_reference"]["actor_groups"]
            if row["imagination_horizon"] == horizon
        )["mean_normalized_return"]
        sparse = next(
            row for row in manifest["sparse_residual_reference"]["actor_groups"]
            if row["imagination_horizon"] == horizon
        )["mean_normalized_return"]
        deltas = report["matched_deltas"]["actor"][str(horizon)]
        for name, reference in (
            ("versus_base_mtp", base),
            ("versus_shortcut", shortcut),
            ("versus_centered_residual", centered),
            ("versus_sparse_residual", sparse),
        ):
            if not _close(deltas[name], measured - reference):
                raise ValueError(f"actor delta mismatch for {name}, horizon {horizon}")
        dense_reference = manifest.get("dense_residual_reference")
        if dense_reference is not None:
            dense = next(
                row for row in dense_reference["actor_groups"]
                if row["imagination_horizon"] == horizon
            )["mean_normalized_return"]
            if not _close(deltas["versus_dense_residual"], measured - dense):
                raise ValueError(
                    f"actor delta mismatch versus dense residual at horizon {horizon}"
                )
    for horizon in ("1", "3", "5", "15"):
        measured = float(np.mean([
            row["metrics_by_horizon"][horizon]["pairwise_accuracy"]
            for row in evaluation_rows
        ]))
        if not _close(measured, report["probe_groups"][horizon]["pairwise_accuracy_mean"]):
            raise ValueError(f"probe aggregation mismatch at horizon {horizon}")
        measured_regret = float(np.mean([
            row["metrics_by_horizon"][horizon]["mean_simulator_regret"]
            for row in evaluation_rows
        ]))
        deltas = report["matched_deltas"]["probe"][horizon]
        base = manifest["base_mtp_reference"]["probe_groups"][horizon]
        centered = manifest["centered_residual_reference"]["probe_groups"][horizon]
        sparse = manifest["sparse_residual_reference"]["probe_groups"][horizon]
        if (
            not _close(
                deltas["pairwise_accuracy_versus_base_mtp"],
                measured - base["pairwise_accuracy_mean"],
            )
            or not _close(
                deltas["regret_versus_base_mtp"],
                measured_regret - base["mean_simulator_regret"],
            )
            or not _close(
                deltas["pairwise_accuracy_versus_centered_residual"],
                measured - centered["pairwise_accuracy_mean"],
            )
            or not _close(
                deltas["regret_versus_centered_residual"],
                measured_regret - centered["mean_simulator_regret"],
            )
            or not _close(
                deltas["pairwise_accuracy_versus_sparse_residual"],
                measured - sparse["pairwise_accuracy_mean"],
            )
            or not _close(
                deltas["regret_versus_sparse_residual"],
                measured_regret - sparse["mean_simulator_regret"],
            )
        ):
            raise ValueError(f"probe delta mismatch at horizon {horizon}")
        dense_reference = manifest.get("dense_residual_reference")
        if dense_reference is not None:
            dense = dense_reference["probe_groups"][horizon]
            if (
                not _close(
                    deltas["pairwise_accuracy_versus_dense_residual"],
                    measured - dense["pairwise_accuracy_mean"],
                )
                or not _close(
                    deltas["regret_versus_dense_residual"],
                    measured_regret - dense["mean_simulator_regret"],
                )
            ):
                raise ValueError(
                    f"probe delta mismatch versus dense residual at horizon {horizon}"
                )
    for context in ("posterior", "corrupted", "generated"):
        measured = float(np.mean([
            row["reward_context_metrics"][context]["mse"]
            for row in evaluation_rows
        ]))
        if not _close(measured, report["reward_groups"][context]["mean_mse"]):
            raise ValueError(f"reward aggregation mismatch for {context}")
        measured_calibration = float(np.mean([
            row["reward_context_metrics"][context][
                "offset_zero_mean_calibration_error"
            ]
            for row in evaluation_rows
        ]))
        deltas = report["matched_deltas"]["reward"][context]
        base = manifest["base_mtp_reference"]["reward_groups"][context]
        centered = manifest["centered_residual_reference"]["reward_groups"][context]
        sparse = manifest["sparse_residual_reference"]["reward_groups"][context]
        if (
            not _close(deltas["mse_versus_base_mtp"], measured - base["mean_mse"])
            or not _close(
                deltas["calibration_versus_base_mtp"],
                measured_calibration - base["mean_offset_zero_calibration_error"],
            )
            or not _close(
                deltas["mse_versus_centered_residual"],
                measured - centered["mean_mse"],
            )
            or not _close(
                deltas["calibration_versus_centered_residual"],
                measured_calibration
                - centered["mean_offset_zero_calibration_error"],
            )
            or not _close(
                deltas["mse_versus_sparse_residual"],
                measured - sparse["mean_mse"],
            )
            or not _close(
                deltas["calibration_versus_sparse_residual"],
                measured_calibration
                - sparse["mean_offset_zero_calibration_error"],
            )
        ):
            raise ValueError(f"reward delta mismatch for {context}")
        dense_reference = manifest.get("dense_residual_reference")
        if dense_reference is not None:
            dense = dense_reference["reward_groups"][context]
            if (
                not _close(
                    deltas["mse_versus_dense_residual"],
                    measured - dense["mean_mse"],
                )
                or not _close(
                    deltas["calibration_versus_dense_residual"],
                    measured_calibration
                    - dense["mean_offset_zero_calibration_error"],
                )
            ):
                raise ValueError(
                    f"reward delta mismatch versus dense residual for {context}"
                )
    print("ACTION_REWARD_RESIDUAL_RESULTS_VERIFIED")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("preflight", "manifest", "dense-probes", "results")
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        verify_preflight(args.root)
    elif args.mode == "manifest":
        verify_manifest(args.root)
    elif args.mode == "dense-probes":
        verify_dense_probes(args.root)
    else:
        verify_results(args.root)


if __name__ == "__main__":
    main()
