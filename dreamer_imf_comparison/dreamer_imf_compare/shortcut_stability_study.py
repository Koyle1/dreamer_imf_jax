"""Paired failed-seed diagnostic for shortcut world-model stability repairs.

This is deliberately an engineering diagnostic, not claim evidence.  It reuses
the exact archived pilot datasets, initialization keys, minibatch schedules,
and objective RNG streams while crossing three binary stabilizers: EMA teacher,
bounded bootstrap intermediates, and support-safe bootstrap composition.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import itertools
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from .artifacts import read_json, write_json_atomic
from .matched_objective_benchmark import (
    _batch_schedule,
    _materialize_batch,
    _metrics_dict,
    _tree_digest,
    derive_jax_key,
    file_sha256,
    load_npz,
    object_sha256,
    stage_directory,
)


MANIFEST_SCHEMA = "shortcut-stability-study-manifest-v1"
RESULT_SCHEMA = "shortcut-stability-study-result-v1"
SUMMARY_SCHEMA = "shortcut-stability-study-summary-v1"
VARIANTS = tuple(
    {
        "ema_teacher": ema,
        "bounded_intermediate": bounded,
        "support_safe_bootstrap": support,
    }
    for ema, bounded, support in itertools.product((False, True), repeat=3)
)
DEFAULT_CHECKPOINTS = (1, 100, 500, 1_000, 2_000, 5_000, 10_000)
LOSS_CEILING = 1_000.0
ABORT_LOSS = 1_000_000.0
PRIOR_SPECTRAL_NORM_CEILING = 256.0
ROLLOUT_ABS_CEILING = 100.0


def _git_commit(workspace: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _variant_id(variant: Mapping[str, bool]) -> str:
    return "-".join(
        (
            f"ema{int(variant['ema_teacher'])}",
            f"bound{int(variant['bounded_intermediate'])}",
            f"support{int(variant['support_safe_bootstrap'])}",
        )
    )


def _entry_id(task: str, seed: int, variant: Mapping[str, bool]) -> str:
    return f"{task}-seed{seed}-{_variant_id(variant)}"


def _config_from_json(value: Mapping[str, Any]) -> Any:
    from imf_dreamer_jax import DreamerConfig

    fields = dict(value)
    fields["observation_shape"] = tuple(fields["observation_shape"])
    fields["overshooting_distances"] = tuple(fields["overshooting_distances"])
    return DreamerConfig(**fields)


def _find_dataset_cell(matrix: Mapping[str, Any], world: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in matrix["cells"]
        if cell.get("stage") == "dataset" and cell.get("cell_id") in world["dependencies"]
    ]
    if len(matches) != 1:
        raise ValueError("archived world cell does not have one dataset dependency")
    return matches[0]


def prepare_manifest(
    archived_root: str | Path,
    output_root: str | Path,
    *,
    updates: int = 10_000,
) -> dict[str, Any]:
    """Select the worst archived shortcut config per task/seed and freeze 2^3 cells."""

    if not isinstance(updates, int) or isinstance(updates, bool) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    archived = Path(archived_root).resolve()
    output = Path(output_root).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("diagnostic manifest already exists; use a new output root")
    matrix_path = archived / "matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"archived matrix is missing: {matrix_path}")
    matrix = read_json(matrix_path)
    candidates: dict[tuple[str, int], list[tuple[float, Mapping[str, Any], dict[str, Any]]]] = {}
    for cell in matrix.get("cells", []):
        if (
            cell.get("stage") != "world_model"
            or cell.get("arm") != "shortcut_forcing"
            or cell.get("budget_track") != "equal_updates"
        ):
            continue
        result_path = stage_directory(archived, cell) / "result.json"
        if not result_path.is_file():
            continue
        result = read_json(result_path)
        if result.get("status") != "complete" or "runtime_config" not in result:
            continue
        prior_loss = float(result.get("final_metrics", {}).get("prior", math.nan))
        if math.isnan(prior_loss):
            continue
        key = (str(cell["task"]), int(cell["world_model_seed"]))
        candidates.setdefault(key, []).append((prior_loss, cell, result))
    if not candidates:
        raise ValueError("no completed equal-updates shortcut world cells were found")

    base_rows: list[dict[str, Any]] = []
    for (task, seed), rows in sorted(candidates.items()):
        prior_loss, world, result = max(rows, key=lambda row: row[0])
        dataset = _find_dataset_cell(matrix, world)
        dataset_path = stage_directory(archived, dataset) / "dataset.npz"
        dataset_result_path = stage_directory(archived, dataset) / "result.json"
        if not dataset_path.is_file() or not dataset_result_path.is_file():
            raise FileNotFoundError("selected archived dataset evidence is missing")
        base_rows.append(
            {
                "task": task,
                "world_model_seed": seed,
                "archived_world_cell_id": world["cell_id"],
                "archived_world_result_sha256": file_sha256(
                    stage_directory(archived, world) / "result.json"
                ),
                "archived_final_prior_loss": prior_loss,
                "archived_runtime_config": result["runtime_config"],
                "dataset_cell_id": dataset["cell_id"],
                "dataset_path": str(dataset_path),
                "dataset_sha256": file_sha256(dataset_path),
                "dataset_result_sha256": file_sha256(dataset_result_path),
            }
        )
    if len(base_rows) != 6:
        raise ValueError(f"expected six task/seed bases, found {len(base_rows)}")

    entries: list[dict[str, Any]] = []
    for base in base_rows:
        for variant in VARIANTS:
            entries.append(
                {
                    "index": len(entries),
                    "entry_id": _entry_id(
                        base["task"], base["world_model_seed"], variant
                    ),
                    "base": base,
                    "variant": variant,
                }
            )
    workspace = Path(__file__).resolve().parents[2]
    body = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "frozen_before_diagnostic_outcomes",
        "evidence_class": "engineering_diagnostic_not_claim_evidence",
        "source_commit": _git_commit(workspace),
        "archived_root": str(archived),
        "archived_matrix_sha256": file_sha256(matrix_path),
        "updates": updates,
        "batch_size": 32,
        "sequence_length": 32,
        "checkpoint_updates": [value for value in DEFAULT_CHECKPOINTS if value <= updates],
        "thresholds": {
            "acceptance_prior_loss": LOSS_CEILING,
            "abort_prior_loss": ABORT_LOSS,
            "prior_spectral_norm": PRIOR_SPECTRAL_NORM_CEILING,
            "rollout_absolute_value": ROLLOUT_ABS_CEILING,
        },
        "objective": {
            "stable_x_space_regression": True,
            "sampling_clip": None,
            "factorial_stabilizers": [
                "ema_teacher",
                "bounded_intermediate",
                "support_safe_bootstrap",
            ],
            "pairing": "same_dataset_initialization_minibatches_and_objective_rng_within_task_seed",
        },
        "entries": entries,
    }
    body["manifest_sha256"] = object_sha256(body)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(manifest_path, body)
    return body


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "status",
        "evidence_class",
        "source_commit",
        "archived_root",
        "archived_matrix_sha256",
        "updates",
        "batch_size",
        "sequence_length",
        "checkpoint_updates",
        "thresholds",
        "objective",
        "entries",
        "manifest_sha256",
    }
    if set(manifest) != required or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("shortcut stability manifest schema is invalid")
    body = dict(manifest)
    claimed = body.pop("manifest_sha256")
    if claimed != object_sha256(body):
        raise ValueError("shortcut stability manifest digest mismatch")
    entries = manifest["entries"]
    if len(entries) != 48 or [entry.get("index") for entry in entries] != list(range(48)):
        raise ValueError("shortcut stability manifest must contain indexed 6x8 cells")
    pairs = {
        (entry["base"]["task"], int(entry["base"]["world_model_seed"]))
        for entry in entries
    }
    if len(pairs) != 6:
        raise ValueError("shortcut stability manifest must contain six task/seed pairs")
    for entry in entries:
        if entry.get("entry_id") != _entry_id(
            entry["base"]["task"], int(entry["base"]["world_model_seed"]), entry["variant"]
        ):
            raise ValueError("shortcut stability entry identity mismatch")
        if entry["variant"] not in VARIANTS:
            raise ValueError("shortcut stability manifest contains an unknown variant")


def _tree_health(tree: Any) -> dict[str, Any]:
    import jax

    leaves = [np.asarray(value) for value in jax.tree_util.tree_leaves(tree)]
    finite = all(np.isfinite(value).all() for value in leaves)
    maximum = max((float(np.max(np.abs(value))) for value in leaves if value.size), default=0.0)
    return {"finite": finite, "max_abs": maximum, "sha256": _tree_digest(tree)}


def _prior_spectral_norms(prior: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for name, child in sorted(value.items()):
                visit(child, f"{path}.{name}" if path else str(name))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{path}.{index}")
        elif path.endswith("weight"):
            array = np.asarray(value)
            if array.ndim == 2:
                rows.append(
                    {
                        "path": path,
                        "shape": list(array.shape),
                        "spectral_norm": float(np.linalg.norm(array.astype(np.float64), ord=2)),
                    }
                )

    visit(prior, "prior")
    if not rows:
        raise ValueError("shortcut prior contains no matrix weights")
    return rows


def _rollout_health(state: Any, batch: Mapping[str, Any], config: Any, task: str, seed: int) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import diverse_imagination_starts, imagine, observe_sequence

    sequence = observe_sequence(
        state.params.world_model,
        batch["observations"],
        batch["actions"],
        derive_jax_key("stability-observe", task, seed),
        config,
        is_first=batch["is_first"],
    )
    start = diverse_imagination_starts(
        sequence.states,
        config.burn_in,
        derive_jax_key("stability-start", task, seed),
    )
    imagined = imagine(
        state.params,
        start,
        derive_jax_key("stability-imagine", task, seed),
        config,
        horizon=15,
    )
    jax.block_until_ready(imagined)
    arrays = [np.asarray(getattr(imagined, field)) for field in imagined._fields]
    return {
        "horizon": 15,
        "finite": all(np.isfinite(value).all() for value in arrays),
        "max_abs": max(float(np.max(np.abs(value))) for value in arrays if value.size),
        "feature_max_abs": float(np.max(np.abs(np.asarray(imagined.features)))),
    }


def run_entry(output_root: str | Path, index: int) -> dict[str, Any]:
    """Run one frozen factorial cell. Divergence is recorded, not hidden or clipped."""

    import jax
    from imf_dreamer_jax import create_agent, jit_train_world_model

    output = Path(output_root).resolve()
    manifest = read_json(output / "manifest.json")
    validate_manifest(manifest)
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(manifest["entries"]):
        raise ValueError("diagnostic index is out of range")
    entry = manifest["entries"][index]
    directory = output / "cells" / entry["entry_id"]
    result_path = directory / "result.json"
    if result_path.is_file():
        result = read_json(result_path)
        validate_result(result, entry, manifest)
        return result
    directory.mkdir(parents=True, exist_ok=False)
    base = entry["base"]
    dataset_path = Path(base["dataset_path"])
    if file_sha256(dataset_path) != base["dataset_sha256"]:
        raise ValueError("archived diagnostic dataset digest mismatch")
    arrays = load_npz(dataset_path)
    variant = entry["variant"]
    config = replace(
        _config_from_json(base["archived_runtime_config"]),
        shortcut_bootstrap_ema_decay=0.999 if variant["ema_teacher"] else None,
        shortcut_intermediate_clip=4.0 if variant["bounded_intermediate"] else None,
        shortcut_support_safe_bootstrap=variant["support_safe_bootstrap"],
        shortcut_sampling_clip=None,
    )
    updates = int(manifest["updates"])
    batch_size = int(manifest["batch_size"])
    sequence_length = int(manifest["sequence_length"])
    task = str(base["task"])
    seed = int(base["world_model_seed"])
    schedule = _batch_schedule(
        arrays,
        task=task,
        world_model_seed=seed,
        updates=updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    state = create_agent(config, derive_jax_key("world-init", task, seed))
    objective_key = derive_jax_key("world-objective", task, seed)
    checkpoints: list[dict[str, Any]] = []
    checkpoint_updates = set(int(value) for value in manifest["checkpoint_updates"])
    status = "complete"
    failure_reason: str | None = None
    latest: dict[str, float] | None = None
    last_batch: Mapping[str, Any] | None = None
    started = time.perf_counter()
    for update in range(updates):
        last_batch = _materialize_batch(
            arrays,
            schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        try:
            state, metrics = jit_train_world_model(
                state, last_batch, jax.random.fold_in(objective_key, update), config
            )
            completed = update + 1
            if completed in checkpoint_updates or completed % 100 == 0:
                jax.block_until_ready(metrics)
                latest = _metrics_dict(metrics)
                prior = float(latest["prior"])
                health = _tree_health(state.params.world_model["prior"])
                if completed in checkpoint_updates:
                    checkpoints.append(
                        {
                            "update": completed,
                            "metrics": latest,
                            "prior_parameter_max_abs": health["max_abs"],
                            "prior_parameters_finite": health["finite"],
                        }
                    )
                if not health["finite"] or prior > float(manifest["thresholds"]["abort_prior_loss"]):
                    status = "diverged"
                    failure_reason = "nonfinite_parameters_or_prior_loss_exceeded_abort_ceiling"
                    break
        except (FloatingPointError, ValueError) as error:
            status = "diverged"
            failure_reason = f"{type(error).__name__}: {error}"
            break
    completed_updates = update + 1
    prior_health = _tree_health(state.params.world_model["prior"])
    spectral = _prior_spectral_norms(state.params.world_model["prior"])
    rollout = None
    if status == "complete" and last_batch is not None:
        try:
            rollout = _rollout_health(state, last_batch, config, task, seed)
        except (FloatingPointError, ValueError) as error:
            status = "diverged"
            failure_reason = f"rollout_{type(error).__name__}: {error}"
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "failure_reason": failure_reason,
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": manifest["source_commit"],
        "index": index,
        "entry_id": entry["entry_id"],
        "task": task,
        "world_model_seed": seed,
        "variant": variant,
        "runtime_config": asdict(config),
        "archived_world_cell_id": base["archived_world_cell_id"],
        "archived_final_prior_loss": base["archived_final_prior_loss"],
        "updates_requested": updates,
        "updates_completed": completed_updates,
        "checkpoint_metrics": checkpoints,
        "final_metrics": latest,
        "prior_parameter_health": prior_health,
        "prior_spectral_norms": spectral,
        "max_prior_spectral_norm": max(row["spectral_norm"] for row in spectral),
        "rollout_health": rollout,
        "sampling_clip": None,
        "wall_seconds": time.perf_counter() - started,
    }
    result["result_sha256"] = object_sha256(result)
    validate_result(result, entry, manifest)
    write_json_atomic(result_path, result)
    return result


def validate_result(result: Mapping[str, Any], entry: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("shortcut stability result schema is invalid")
    body = dict(result)
    claimed = body.pop("result_sha256", None)
    if claimed != object_sha256(body):
        raise ValueError("shortcut stability result digest mismatch")
    if (
        result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("index") != entry["index"]
        or result.get("entry_id") != entry["entry_id"]
        or result.get("variant") != entry["variant"]
        or result.get("sampling_clip") is not None
    ):
        raise ValueError("shortcut stability result identity mismatch")
    if result.get("status") not in {"complete", "diverged"}:
        raise ValueError("shortcut stability result status is invalid")
    if not math.isfinite(float(result.get("wall_seconds", math.nan))):
        raise ValueError("shortcut stability result wall time is invalid")


def _passes(result: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    metrics = result.get("final_metrics") or {}
    rollout = result.get("rollout_health") or {}
    checkpoints = result.get("checkpoint_metrics") or []
    observed_prior = [float(row["metrics"]["prior"]) for row in checkpoints]
    return bool(
        result.get("status") == "complete"
        and int(result.get("updates_completed", -1)) == int(result.get("updates_requested", -2))
        and math.isfinite(float(metrics.get("prior", math.nan)))
        and float(metrics["prior"]) <= float(thresholds["acceptance_prior_loss"])
        and all(value <= float(thresholds["acceptance_prior_loss"]) for value in observed_prior)
        and result.get("prior_parameter_health", {}).get("finite") is True
        and float(result.get("max_prior_spectral_norm", math.inf))
        <= float(thresholds["prior_spectral_norm"])
        and rollout.get("finite") is True
        and float(rollout.get("max_abs", math.inf))
        <= float(thresholds["rollout_absolute_value"])
    )


def aggregate(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    manifest = read_json(output / "manifest.json")
    validate_manifest(manifest)
    results: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        path = output / "cells" / entry["entry_id"] / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"diagnostic result is missing: {entry['entry_id']}")
        result = read_json(path)
        validate_result(result, entry, manifest)
        results.append(result)
    thresholds = manifest["thresholds"]
    production = [
        result
        for result in results
        if all(result["variant"].values())
    ]
    archived_failures = sum(
        float(row["base"]["archived_final_prior_loss"])
        > float(thresholds["acceptance_prior_loss"])
        for row in manifest["entries"][::8]
    )
    variant_rows = []
    for variant in VARIANTS:
        selected = [result for result in results if result["variant"] == variant]
        passing = [_passes(result, thresholds) for result in selected]
        final_priors = [
            float(result["final_metrics"]["prior"])
            for result in selected
            if result.get("final_metrics") is not None
            and math.isfinite(float(result["final_metrics"].get("prior", math.nan)))
        ]
        variant_rows.append(
            {
                "variant": variant,
                "variant_id": _variant_id(variant),
                "passed": int(sum(passing)),
                "total": len(selected),
                "median_final_prior_loss": (
                    float(np.median(final_priors)) if final_priors else None
                ),
                "median_wall_seconds": float(
                    np.median([float(result["wall_seconds"]) for result in selected])
                ),
            }
        )
    minimal: list[dict[str, Any]] = []
    for task, seed in sorted({(r["task"], r["world_model_seed"]) for r in results}):
        selected = [r for r in results if r["task"] == task and r["world_model_seed"] == seed]
        passing = [r for r in selected if _passes(r, thresholds)]
        fewest = min((sum(r["variant"].values()) for r in passing), default=None)
        minimal.append(
            {
                "task": task,
                "world_model_seed": seed,
                "fewest_stabilizers": fewest,
                "minimal_passing_variants": [
                    _variant_id(r["variant"])
                    for r in passing
                    if sum(r["variant"].values()) == fewest
                ],
            }
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "evidence_class": manifest["evidence_class"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source_commit": manifest["source_commit"],
        "cells": len(results),
        "archived_baseline_failures": archived_failures,
        "production_variant_passed": sum(_passes(result, thresholds) for result in production),
        "production_variant_total": len(production),
        "accepted": (
            archived_failures >= 4
            and len(production) == 6
            and all(_passes(result, thresholds) for result in production)
        ),
        "thresholds": thresholds,
        "variant_summary": variant_rows,
        "minimal_passing_variants": minimal,
        "total_gpu_wall_seconds": float(sum(float(r["wall_seconds"]) for r in results)),
        "limitations": [
            "engineering diagnostic on two pilot tasks and three selected stress seeds",
            "selection used archived outcomes and is ineligible for confirmatory claims",
            "tests stability and unbounded rollout health, not actor return superiority",
        ],
    }
    summary["summary_sha256"] = object_sha256(summary)
    write_json_atomic(output / "summary.json", summary)
    return summary


def resolve_array_index(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None:
        raise ValueError("--index or SLURM_ARRAY_TASK_ID is required")
    return int(value)


__all__ = [
    "ABORT_LOSS",
    "LOSS_CEILING",
    "MANIFEST_SCHEMA",
    "PRIOR_SPECTRAL_NORM_CEILING",
    "RESULT_SCHEMA",
    "ROLLOUT_ABS_CEILING",
    "SUMMARY_SCHEMA",
    "VARIANTS",
    "aggregate",
    "prepare_manifest",
    "resolve_array_index",
    "run_entry",
    "validate_manifest",
    "validate_result",
]
