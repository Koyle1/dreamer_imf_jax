#!/usr/bin/env python3
"""CLI for the reduced Reacher causal trajectory-iMF intervention study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dreamer_imf_compare import causal_reacher_study as study
from dreamer_imf_compare.artifacts import read_json, write_json_atomic
from dreamer_imf_compare import matched_objective_benchmark as benchmark


def _verify_manifest(arguments: argparse.Namespace) -> dict:
    output = Path(arguments.output).resolve()
    manifest = read_json(output / "manifest.json")
    expected = study.build_manifest(arguments.pilot_root)
    if manifest != expected:
        raise ValueError("small-study manifest differs from current source and pilot")
    if manifest.get("status") != "frozen_before_execution":
        raise ValueError("small-study manifest was not frozen before execution")
    if len(manifest.get("scheduled_cells", ())) != len(study.WORLD_MODEL_SEEDS):
        raise ValueError("small-study manifest has the wrong number of scheduled cells")
    if any(row.get("arm") != "trajectory_imf" for row in manifest["scheduled_cells"]):
        raise ValueError("small-study manifest schedules a non-iMF arm")
    if any(
        row.get("mode") != "reference_only_never_scheduled"
        for row in manifest["shortcut_baseline"]
    ):
        raise ValueError("shortcut baseline is not reference-only")
    if manifest["interpretation"].get("claim_eligible") is not False:
        raise ValueError("the exploratory intervention was incorrectly claim-enabled")
    return manifest


def _preflight(arguments: argparse.Namespace) -> None:
    import jax

    output = Path(arguments.output).resolve()
    manifest = _verify_manifest(arguments)
    source = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True
    ).strip()
    if commit != manifest["source_commit"]:
        raise ValueError("preflight checkout differs from the frozen manifest commit")
    if dirty:
        raise ValueError("preflight requires a clean source checkout")
    if jax.default_backend() != "gpu":
        raise ValueError(f"preflight requires a GPU backend, got {jax.default_backend()}")
    result = {
        "schema_version": study.SCHEMA,
        "stage": "preflight",
        "status": "complete",
        "source_commit": commit,
        "manifest_sha256": manifest["manifest_sha256"],
        "focused_and_full_tests_completed": True,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(output / "preflight.json", result)
    print("CAUSAL_REACHER_PREFLIGHT_VERIFIED")


def _diagnose(arguments: argparse.Namespace) -> None:
    import jax
    import run_policy_alignment_diagnostics as policy_runner

    output = Path(arguments.output).resolve()
    seed = int(arguments.seed)
    dataset_id = f"seed-{seed}"
    actor_id = f"seed-{seed}-actor-{study.ACTOR_SEED}"
    dataset_cell = {
        "stage": "dataset",
        "cell_id": dataset_id,
        "dependencies": [],
    }
    actor_cell = {
        "stage": "actor",
        "cell_id": actor_id,
        "dependencies": [dataset_id],
        "task": study.TASK,
        "arm": "trajectory_imf",
        "world_model_seed": seed,
        "actor_seed": study.ACTOR_SEED,
        "candidate_id": "trajectory_imf_causal_consistency",
    }
    matrix = {"cells": [dataset_cell, actor_cell]}
    sampler = jax.jit(
        benchmark.open_loop_samples_with_continuation,
        static_argnames=("config",),
    )
    row, manifest = policy_runner._diagnose_actor_cell(
        output,
        matrix,
        actor_cell,
        output_root=output / "diagnostic_raw" / f"seed-{seed}",
        episode=-1,
        states_per_cell=4,
        horizon=30,
        action_persistence=30,
        draws=8,
        action_delta=0.25,
        finite_difference_epsilon=0.05,
        discount=0.99,
        sampler=sampler,
        gradient_function=policy_runner._model_action_gradient_function(),
    )
    result = {
        "schema_version": study.SCHEMA,
        "stage": "policy_diagnostics",
        "status": "complete",
        "source_commit": read_json(output / "manifest.json")["source_commit"],
        "world_model_seed": seed,
        "row": row,
        "artifact": manifest,
    }
    write_json_atomic(output / "diagnostics" / f"seed-{seed}.json", result)
    print(f"CAUSAL_REACHER_DIAGNOSTIC_COMPLETE seed={seed}")


def _verify(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output).resolve()
    manifest = _verify_manifest(arguments)
    preflight = read_json(output / "preflight.json")
    if (
        preflight.get("status") != "complete"
        or preflight.get("source_commit") != manifest["source_commit"]
        or preflight.get("jax_backend") != "gpu"
        or preflight.get("focused_and_full_tests_completed") is not True
    ):
        raise ValueError("small-study preflight is absent or invalid")
    for seed in study.WORLD_MODEL_SEEDS:
        dataset = read_json(output / "dataset" / f"seed-{seed}" / "result.json")
        world = read_json(output / "world_model" / f"seed-{seed}" / "result.json")
        rollout = read_json(output / "rollout" / f"seed-{seed}" / "result.json")
        actor = read_json(
            output / "actor" / f"seed-{seed}-actor-{study.ACTOR_SEED}" / "result.json"
        )
        diagnostic = read_json(output / "diagnostics" / f"seed-{seed}.json")
        if any(
            value.get("status") != "complete"
            for value in (dataset, world, rollout, actor, diagnostic)
        ):
            raise ValueError(f"seed {seed} has an incomplete stage")
        if dataset["probe_file_sha256"] != benchmark.file_sha256(
            output / "dataset" / f"seed-{seed}" / "causal_probes.npz"
        ):
            raise ValueError(f"seed {seed} causal probe digest mismatch")
        if world["checkpoint_sha256"] != benchmark.file_sha256(
            output / "world_model" / f"seed-{seed}" / "checkpoint.pkl"
        ):
            raise ValueError(f"seed {seed} world checkpoint digest mismatch")
        if actor["checkpoint_sha256"] != benchmark.file_sha256(
            output
            / "actor"
            / f"seed-{seed}-actor-{study.ACTOR_SEED}"
            / "checkpoint.pkl"
        ):
            raise ValueError(f"seed {seed} actor checkpoint digest mismatch")
    comparison = read_json(output / "comparison.json")
    recomputed = study.finalize(
        arguments.pilot_root, output, arguments.baseline_diagnostics
    )
    if comparison != recomputed:
        raise ValueError("comparison does not match recomputation from retained artifacts")
    digest_payload = dict(comparison)
    digest = digest_payload.pop("comparison_sha256")
    if digest != benchmark.object_sha256(digest_payload):
        raise ValueError("comparison digest mismatch")
    if comparison.get("claim_eligible") is not False:
        raise ValueError("exploratory comparison was incorrectly promoted")
    print("CAUSAL_REACHER_SMALL_VERIFIED")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)
    for command in (
        "manifest",
        "verify-manifest",
        "preflight",
        "prepare",
        "train",
        "diagnose",
        "finalize",
        "verify",
    ):
        selected = subparsers.add_parser(command)
        selected.add_argument("--pilot-root", required=True)
        selected.add_argument("--output", required=True)
        if command in ("prepare", "train", "diagnose"):
            selected.add_argument("--seed", required=True, type=int)
        if command == "finalize":
            selected.add_argument("--baseline-diagnostics", required=True)
        if command == "verify":
            selected.add_argument("--baseline-diagnostics", required=True)
    return value


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "manifest":
        manifest = study.write_manifest(arguments.pilot_root, arguments.output)
        print(json.dumps(manifest, sort_keys=True, indent=2))
    elif arguments.command == "verify-manifest":
        _verify_manifest(arguments)
        print("REACHER_IMF_SMALL_MANIFEST_VERIFIED")
    elif arguments.command == "preflight":
        _preflight(arguments)
    elif arguments.command == "prepare":
        result = study.prepare_seed(arguments.pilot_root, arguments.output, arguments.seed)
        print(json.dumps(result, sort_keys=True, indent=2))
    elif arguments.command == "train":
        result = study.train_seed(arguments.pilot_root, arguments.output, arguments.seed)
        print(json.dumps(result, sort_keys=True, indent=2))
    elif arguments.command == "diagnose":
        _diagnose(arguments)
    elif arguments.command == "finalize":
        result = study.finalize(
            arguments.pilot_root,
            arguments.output,
            arguments.baseline_diagnostics,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        _verify(arguments)


if __name__ == "__main__":
    main()
