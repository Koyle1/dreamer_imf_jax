#!/usr/bin/env python3
"""Independent structural and artifact verifier for reward-head MTP study."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamer_imf_compare.artifacts import read_json
from dreamer_imf_compare import matched_objective_benchmark as benchmark
from dreamer_imf_compare import reward_head_mtp_study as study


def _settings(manifest: dict) -> dict:
    return {
        "reward_updates": manifest["reward_updates"],
        "actor_updates": manifest["actor_updates"],
        "preparation_updates": manifest["preparation_updates"],
        "reward_bins": manifest["reward_bins"],
        "evaluation_episodes": manifest["evaluation_episodes"],
    }


def verify_manifest(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    expected = study.build_manifest(
        manifest["baseline_root"], manifest["train_probe_root"], **_settings(manifest)
    )
    if manifest != expected:
        raise ValueError("manifest is not the canonical frozen design")
    if (
        manifest["status"] != "frozen_before_execution"
        or manifest["arms"] != list(study.ARMS)
        or manifest["world_model_seeds"] != list(study.WORLD_MODEL_SEEDS)
        or manifest["trainable_world_subtrees"] != ["reward"]
        or manifest["imagination_horizons"] != [5, 15]
    ):
        raise ValueError("manifest factorial or freeze contract is invalid")
    if set(manifest["objectives"]) != set(study.ARMS):
        raise ValueError("manifest objective arms are incomplete")
    for source in manifest["source_artifacts"]:
        checks = (
            (source["checkpoint"], source["checkpoint_sha256"]),
            (source["test_probe_bank"], source["test_probe_bank_file_sha256"]),
            (source["train_probe_bank"], source["train_probe_bank_file_sha256"]),
        )
        for path, digest in checks:
            if benchmark.file_sha256(path) != digest:
                raise ValueError(f"source artifact digest changed: {path}")
    print("REWARD_HEAD_MTP_MANIFEST_VERIFIED")
    return manifest


def verify_results(root: Path) -> dict:
    manifest = verify_manifest(root)
    expected_reward = len(study.WORLD_MODEL_SEEDS) * len(study.ARMS)
    expected_actor = expected_reward * 2
    schedules: dict[int, str] = {}
    for seed in study.WORLD_MODEL_SEEDS:
        for arm in study.ARMS:
            tag = f"seed-{seed}-{arm}"
            reward_directory = root / "reward" / tag
            reward = read_json(reward_directory / "result.json")
            if (
                reward.get("status") != "complete"
                or reward.get("source_commit") != manifest["source_commit"]
                or reward.get("manifest_sha256") != manifest["manifest_sha256"]
                or reward.get("nonreward_frozen") is not True
                or reward.get("arm") != arm
                or reward.get("world_model_seed") != seed
                or benchmark.file_sha256(reward_directory / "checkpoint.pkl")
                != reward["checkpoint_sha256"]
                or benchmark.file_sha256(reward_directory / "schedules.npz")
                != reward["schedule_file_sha256"]
            ):
                raise ValueError(f"invalid reward result: {tag}")
            for field in (
                "world_model_subtree_parameter_deltas",
                "world_model_subtree_first_moment_deltas",
                "world_model_subtree_second_moment_deltas",
            ):
                leaked = {
                    name: value for name, value in reward[field].items()
                    if name != "reward" and value != 0.0
                }
                if leaked:
                    raise ValueError(f"non-reward mutation in {tag}: {field}={leaked}")
            schedule_digest = benchmark.array_sha256(
                benchmark.load_npz(reward_directory / "schedules.npz")
            )
            if seed in schedules and schedules[seed] != schedule_digest:
                raise ValueError(f"arms use different schedules for seed {seed}")
            schedules[seed] = schedule_digest
            evaluation_directory = root / "evaluation" / tag
            evaluation = read_json(evaluation_directory / "result.json")
            if (
                evaluation.get("status") != "complete"
                or evaluation.get("world_model_checkpoint_sha256")
                != reward["checkpoint_sha256"]
                or benchmark.file_sha256(evaluation_directory / "evaluation_arrays.npz")
                != evaluation["raw_sha256"]
                or set(evaluation.get("reward_context_metrics", {}))
                != {"posterior", "corrupted", "generated"}
                or set(evaluation.get("metrics_by_horizon", {}))
                != {"1", "3", "5", "15"}
            ):
                raise ValueError(f"invalid evaluation result: {tag}")
            for horizon in (5, 15):
                actor_directory = root / "actor" / f"{tag}-h{horizon}"
                actor = read_json(actor_directory / "result.json")
                if (
                    actor.get("status") != "complete"
                    or actor.get("world_model_frozen") is not True
                    or actor.get("world_model_parameter_delta") != 0.0
                    or actor.get("world_model_checkpoint_sha256")
                    != reward["checkpoint_sha256"]
                    or benchmark.file_sha256(actor_directory / "checkpoint.pkl")
                    != actor["checkpoint_sha256"]
                    or benchmark.file_sha256(actor_directory / "action_traces.npz")
                    != actor["raw_action_traces_sha256"]
                ):
                    raise ValueError(f"invalid actor result: {tag}-h{horizon}")
    report = read_json(root / "report.json")
    if (
        report.get("status") != "complete"
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("completed_reward_cells") != expected_reward
        or report.get("completed_evaluation_cells") != expected_reward
        or report.get("completed_actor_cells") != expected_actor
        or set(report.get("reward_groups", {})) != set(study.ARMS)
        or set(report.get("probe_groups", {})) != set(study.ARMS)
    ):
        raise ValueError("final report is incomplete or inconsistent")
    print("REWARD_HEAD_MTP_RESULTS_VERIFIED")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("manifest", "results"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "manifest":
        verify_manifest(args.root)
    else:
        verify_results(args.root)


if __name__ == "__main__":
    main()
