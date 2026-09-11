#!/usr/bin/env python3
"""Read-only post-pilot feasibility gate for the frozen GPU campaign.

The gate deliberately uses aggregate GPU seconds, not elapsed array time.  Four
concurrent one-GPU jobs therefore provide four GPU-seconds per wall-clock second.
Artifact storage is projected per frozen execution unit, while training time is
projected per registered update-equivalent.  A twofold reserve covers workload
mix and pilot-to-confirmatory scale uncertainty.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Mapping


CLUSTER_DIR = Path(__file__).resolve().parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import gpu_preflight
import workflow


FEASIBILITY_SCHEMA = "trajectory-imf-post-pilot-feasibility-v1"
FROZEN_TOTAL_UNITS = 606 + 1746 + 944 + 3666 + 30
FROZEN_PILOT_UNITS = 606
FROZEN_TOTAL_UPDATE_EQUIVALENTS = 396_540_000
FROZEN_PILOT_UPDATE_EQUIVALENTS = 4_320_000
SAFETY_FACTOR = 2.0


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def assess_post_pilot_feasibility(
    spec: Mapping[str, Any],
    *,
    measured_gpu_seconds: float,
    measured_artifact_bytes: int,
    measured_artifact_files: int,
    measured_result_updates: int,
    workspace_record: Mapping[str, Any],
    quota_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the remaining frozen campaign from authenticated pilot evidence."""

    gpu_seconds = _positive_number(measured_gpu_seconds, "pilot GPU seconds")
    artifact_bytes = _positive_number(measured_artifact_bytes, "pilot artifact bytes")
    artifact_files = _positive_number(measured_artifact_files, "pilot artifact files")
    result_updates = _positive_number(measured_result_updates, "pilot result updates")
    remaining_seconds = _positive_number(
        workspace_record.get("remaining_seconds_at_observation"),
        "workspace remaining seconds",
    )
    remaining_quota_bytes = _positive_number(
        quota_record.get("remaining_quota_kib"), "remaining quota KiB"
    ) * 1024.0
    remaining_inodes = _positive_number(
        quota_record.get("remaining_inodes"), "remaining inodes"
    )
    concurrency = spec["scheduler"]["array_concurrency"]
    if concurrency != 4:
        raise ValueError("feasibility oracle requires the frozen four-GPU concurrency")

    remaining_units = FROZEN_TOTAL_UNITS - FROZEN_PILOT_UNITS
    remaining_updates = (
        FROZEN_TOTAL_UPDATE_EQUIVALENTS - FROZEN_PILOT_UPDATE_EQUIVALENTS
    )
    projected_gpu_seconds = math.ceil(
        gpu_seconds
        * remaining_updates
        / FROZEN_PILOT_UPDATE_EQUIVALENTS
        * SAFETY_FACTOR
    )
    reserve_seconds = 3600 * int(
        spec["workspace_allocator"]["minimum_remaining_hours"]
    )
    usable_workspace_seconds = max(0.0, remaining_seconds - reserve_seconds)
    available_gpu_seconds = math.floor(usable_workspace_seconds * concurrency)
    projected_artifact_bytes = math.ceil(
        artifact_bytes * remaining_units / FROZEN_PILOT_UNITS * SAFETY_FACTOR
    )
    projected_artifact_files = math.ceil(
        artifact_files * remaining_units / FROZEN_PILOT_UNITS * SAFETY_FACTOR
    )
    time_feasible = projected_gpu_seconds <= available_gpu_seconds
    storage_feasible = projected_artifact_bytes <= remaining_quota_bytes
    inode_feasible = projected_artifact_files <= remaining_inodes
    feasible = time_feasible and storage_feasible and inode_feasible
    record: dict[str, Any] = {
        "schema_version": FEASIBILITY_SCHEMA,
        "status": "feasible_with_twofold_reserve" if feasible else "extension_required",
        "frozen_campaign": {
            "total_execution_units": FROZEN_TOTAL_UNITS,
            "pilot_execution_units": FROZEN_PILOT_UNITS,
            "remaining_execution_units": remaining_units,
            "total_update_equivalents": FROZEN_TOTAL_UPDATE_EQUIVALENTS,
            "pilot_update_equivalents": FROZEN_PILOT_UPDATE_EQUIVALENTS,
            "remaining_update_equivalents": remaining_updates,
            "array_concurrency": concurrency,
            "safety_factor": SAFETY_FACTOR,
        },
        "pilot_measurement": {
            "gpu_seconds": gpu_seconds,
            "artifact_bytes": int(artifact_bytes),
            "artifact_files": int(artifact_files),
            "result_reported_updates": int(result_updates),
        },
        "capacity": {
            "workspace_remaining_seconds": int(remaining_seconds),
            "workspace_reserve_seconds": reserve_seconds,
            "available_gpu_seconds": available_gpu_seconds,
            "remaining_quota_bytes": int(remaining_quota_bytes),
            "remaining_inodes": int(remaining_inodes),
        },
        "projection": {
            "gpu_seconds": projected_gpu_seconds,
            "artifact_bytes": projected_artifact_bytes,
            "artifact_files": projected_artifact_files,
            "time_feasible": time_feasible,
            "storage_feasible": storage_feasible,
            "inode_feasible": inode_feasible,
        },
        "workspace_evidence_sha256": workflow.object_sha256(workspace_record),
        "quota_evidence_sha256": workflow.object_sha256(quota_record),
    }
    record["feasibility_sha256"] = workflow.object_sha256(record)
    if not feasible:
        failed = [
            name
            for name, passed in (
                ("workspace lifetime/GPU time", time_feasible),
                ("Lustre byte quota", storage_feasible),
                ("Lustre inode quota", inode_feasible),
            )
            if not passed
        ]
        raise RuntimeError(
            "post-pilot feasibility blocks later launch: request a workspace/quota "
            f"extension before submission ({', '.join(failed)})"
        )
    return record


def _pilot_measurement(
    spec: Mapping[str, Any], *, checkout: Path, cluster_root: str | Path | None
) -> dict[str, Any]:
    workflow.require_profile_ledger_authenticated(
        spec, "pilot", cluster_root=cluster_root
    )
    output = workflow.profile_output(spec, "pilot", cluster_root=cluster_root)
    _, _, matrix = workflow.load_validated_main_run(
        output, checkout, "pilot", spec
    )
    workflow.require_all_stages_verified(output, "pilot", matrix)
    gpu_seconds = 0.0
    result_updates = 0
    for stage in ("world_model", "actor"):
        stage_map = workflow.read_json(workflow.map_path(output, stage))
        workflow.validate_stage_map(stage_map, matrix, "pilot", stage)
        for entry in stage_map["entries"]:
            cell = workflow.cell_for_array_index(
                stage_map, matrix, entry["array_index"]
            )
            result_path = output / stage / cell["cell_id"] / "result.json"
            result = workflow.read_json(result_path)
            gpu_seconds += _positive_number(
                result.get("wall_seconds"), f"{cell['cell_id']} wall seconds"
            )
            updates = result.get("updates")
            if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
                raise ValueError(f"{cell['cell_id']} result updates are invalid")
            result_updates += updates
    snapshot = workflow.tree_state(spec, output, cluster_root=cluster_root)
    if any(row["type"] in {"symlink", "other"} for row in snapshot["entries"]):
        raise ValueError("pilot artifact tree contains a symlink or special node")
    files = [row for row in snapshot["entries"] if row["type"] == "file"]
    return {
        "gpu_seconds": gpu_seconds,
        "artifact_bytes": sum(int(row["bytes"]) for row in files),
        "artifact_files": len(files),
        "result_updates": result_updates,
    }


def require_post_pilot_feasible(
    spec: Mapping[str, Any],
    *,
    checkout: str | Path | None = None,
    cluster_root: str | Path | None = None,
    workspace_evidence: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed unless live authoritative capacity covers the projected campaign."""

    checkout_path = (
        Path(checkout)
        if checkout is not None
        else workflow.source_root(spec, cluster_root=cluster_root)
    )
    measured = _pilot_measurement(
        spec, checkout=checkout_path, cluster_root=cluster_root
    )
    workspace, quota = (
        workspace_evidence
        if workspace_evidence is not None
        else gpu_preflight.authoritative_workspace_evidence(dict(spec))
    )
    return assess_post_pilot_feasibility(
        spec,
        measured_gpu_seconds=measured["gpu_seconds"],
        measured_artifact_bytes=measured["artifact_bytes"],
        measured_artifact_files=measured["artifact_files"],
        measured_result_updates=measured["result_updates"],
        workspace_record=workspace,
        quota_record=quota,
    )
