#!/usr/bin/env python3
"""Fail-closed Slurm primitives for controls and hard-pixel evidence.

Scientific result directories remain owned by their canonical runners.  Every
cluster-only map, receipt, retry audit, and profile marker is therefore stored
under ``cluster_state/supplementary`` so that runner artifact inventories stay
exact.  This module never submits jobs; :mod:`submit` is the only Slurm client.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


CLUSTER_DIR = Path(__file__).resolve().parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import workflow


CONTROLS_PROFILES = ("development", "confirmatory")
CONTROLS_STAGES = ("dataset", "compute", "world", "rollout", "actor")
CONTROLS_MAP_SCHEMA = "trajectory-imf-controls-slurm-map-v1"
CONTROLS_FREEZE_SCHEMA = "trajectory-imf-controls-cluster-freeze-v1"
CONTROLS_UNIT_SCHEMA = "trajectory-imf-controls-gpu-unit-verification-v1"
CONTROLS_STAGE_SCHEMA = "trajectory-imf-controls-stage-verification-v1"
CONTROLS_PROFILE_SCHEMA = "trajectory-imf-controls-profile-verification-v1"
PIXEL_MAP_SCHEMA = "trajectory-imf-pixel-slurm-map-v1"
PIXEL_FREEZE_SCHEMA = "trajectory-imf-pixel-cluster-freeze-v1"
PIXEL_UNIT_SCHEMA = "trajectory-imf-pixel-gpu-unit-verification-v1"
PIXEL_PROFILE_SCHEMA = "trajectory-imf-pixel-profile-verification-v1"
SUPPLEMENTARY_RETRY_SCHEMA = "trajectory-imf-supplementary-retry-map-v1"


def _add_source_paths(checkout: Path) -> None:
    workflow._add_source_paths(checkout)


def results_root(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    root = (
        Path(cluster_root) / "results"
        if cluster_root is not None
        else Path(spec["results_root"])
    )
    return workflow.require_safe_cluster_path(
        spec, root, cluster_root=cluster_root
    )


def controls_output(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    if profile not in CONTROLS_PROFILES:
        raise ValueError("controls profile must be development or confirmatory")
    output = results_root(spec, cluster_root=cluster_root) / spec["supplementary"][
        "controls"
    ][profile]["output_name"]
    return workflow.require_safe_cluster_path(
        spec, output, cluster_root=cluster_root
    )


def pixel_output(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    output = results_root(spec, cluster_root=cluster_root) / spec["supplementary"][
        "pixel"
    ]["output_name"]
    return workflow.require_safe_cluster_path(
        spec, output, cluster_root=cluster_root
    )


def supplementary_state(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    state = workflow.state_root(spec, cluster_root=cluster_root) / "supplementary"
    return workflow.require_safe_cluster_path(
        spec, state, cluster_root=cluster_root
    )


def controls_state(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    if profile not in CONTROLS_PROFILES:
        raise ValueError("controls profile must be development or confirmatory")
    state = supplementary_state(spec, cluster_root=cluster_root) / "controls" / profile
    return workflow.require_safe_cluster_path(
        spec, state, cluster_root=cluster_root
    )


def pixel_state(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    state = supplementary_state(spec, cluster_root=cluster_root) / "pixel" / "hard"
    return workflow.require_safe_cluster_path(
        spec, state, cluster_root=cluster_root
    )


def _slurm_id(name: str = "SLURM_JOB_ID") -> str:
    value = os.environ.get(name)
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdigit()
    ):
        raise RuntimeError(f"{name} must be a numeric Slurm job id")
    return value


def _numeric_slurm_id(value: Any) -> bool:
    return isinstance(value, str) and value.isascii() and value.isdigit()


def _slurm_array_identity(array_index: int) -> tuple[str, str]:
    job_id = _slurm_id("SLURM_ARRAY_JOB_ID")
    task_id = _slurm_id("SLURM_ARRAY_TASK_ID")
    if int(task_id) != array_index:
        raise RuntimeError("Slurm task id differs from the resolved array index")
    return job_id, task_id


def _register_event_once(path: Path, event: Mapping[str, Any]) -> None:
    """Append a deterministic event once, tolerating marker-write recovery."""

    workflow.append_ledger_once(path, event)


def _read_regular_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink JSON file")
    return workflow.read_json(path)


def _require_cluster_ready(
    spec: Mapping[str, Any],
    checkout: Path,
    *,
    cluster_root: str | Path | None,
) -> dict[str, Any]:
    contract = _read_regular_json(
        workflow.run_contract_path(spec, cluster_root=cluster_root),
        "cluster run contract",
    )
    workflow.validate_run_contract(contract, spec, checkout)
    preflight = _read_regular_json(
        workflow.preflight_path(spec, cluster_root=cluster_root),
        "GPU preflight",
    )
    workflow.validate_preflight_record(preflight, spec=spec, contract=contract)
    return contract


def require_authenticated_main_profile(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    """Require a main-profile marker and its unique hash-chain event."""

    return workflow.require_profile_ledger_authenticated(
        spec, profile, cluster_root=cluster_root
    )


def _controls_modules(checkout: Path) -> tuple[Any, Any]:
    _add_source_paths(checkout)
    from dreamer_imf_compare import matched_objective_benchmark as parent
    from dreamer_imf_compare import neurips_controls as controls

    return controls, parent


def load_controls_run(
    spec: Mapping[str, Any],
    profile: str,
    checkout: Path,
    *,
    cluster_root: str | Path | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    controls, parent = _controls_modules(checkout)
    output = controls_output(spec, profile, cluster_root=cluster_root)
    protocol, parent_protocol, source, matrix = controls.load_frozen_controls(output)
    controls.validate_controls_protocol(protocol, parent_protocol)
    controls.validate_controls_source_manifest(source, checkout)
    controls.validate_controls_matrix(matrix, protocol, parent_protocol, source)
    parent_matrix = workflow.read_json(output / "parent_dataset_matrix.json")
    parent_source = source["parent_source_manifest"]
    parent_profile = controls._parent_profile(protocol, profile)
    if parent_profile == "pilot":
        parent.validate_pilot_hpo_matrix(parent_matrix, parent_protocol, parent_source)
    else:
        parent.validate_matrix(parent_matrix, parent_protocol, parent_source)
    _validate_controls_counts(spec, profile, protocol, matrix)
    return controls, parent, protocol, parent_protocol, source, matrix, parent_matrix


def _validate_controls_counts(
    spec: Mapping[str, Any],
    profile: str,
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, int]:
    expected = dict(
        spec["supplementary"]["controls"][profile]["expected_stage_counts"]
    )
    counts = {stage: 0 for stage in CONTROLS_STAGES}
    counts["dataset"] = len(protocol["profiles"][profile]["tasks"]) * len(
        protocol["profiles"][profile]["world_model_seeds"]
    )
    for cell in matrix["cells"]:
        if cell.get("stage") not in counts or cell["stage"] == "dataset":
            raise ValueError("controls matrix contains an unknown cluster stage")
        counts[cell["stage"]] += 1
    if counts != expected:
        raise ValueError(f"controls {profile} stage counts differ: {counts} != {expected}")
    if sum(counts.values()) != spec["supplementary"]["controls"][profile][
        "expected_execution_units"
    ]:
        raise ValueError("controls execution-unit total differs from the specification")
    return counts


def _controls_dataset_cells(
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    parent_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected = protocol["profiles"][matrix["profile"]]
    identities = {
        (str(task), int(seed))
        for task in selected["tasks"]
        for seed in selected["world_model_seeds"]
    }
    cells = [
        dict(cell)
        for cell in parent_matrix["cells"]
        if cell.get("stage") == "dataset"
        and (str(cell.get("task")), int(cell.get("world_model_seed"))) in identities
    ]
    observed = {
        (str(cell["task"]), int(cell["world_model_seed"])) for cell in cells
    }
    if observed != identities or len(cells) != len(identities):
        raise ValueError("controls dataset map does not resolve one canonical cell per unit")
    return cells


def build_controls_map(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    parent_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in CONTROLS_STAGES:
        raise ValueError("unknown controls stage")
    if stage == "dataset":
        cells = _controls_dataset_cells(protocol, matrix, parent_matrix)
    else:
        cells = [dict(cell) for cell in matrix["cells"] if cell["stage"] == stage]
    cells.sort(key=lambda cell: cell["cell_id"])
    entries = []
    for index, cell in enumerate(cells):
        entry = {
            "array_index": index,
            "cell_id": cell["cell_id"],
            "identity_sha256": cell["identity_sha256"],
            "task": cell["task"],
            "world_model_seed": cell.get("world_model_seed"),
        }
        entries.append(entry)
    counts = _validate_controls_counts(spec, profile, protocol, matrix)
    if len(entries) != counts[stage]:
        raise ValueError("controls map count differs from its registered stage count")
    payload = {
        "schema_version": CONTROLS_MAP_SCHEMA,
        "status": "frozen_before_stage_execution",
        "kind": "controls",
        "profile": profile,
        "stage": stage,
        "previous_stage": (
            None
            if stage == CONTROLS_STAGES[0]
            else CONTROLS_STAGES[CONTROLS_STAGES.index(stage) - 1]
        ),
        "matrix_sha256": matrix["matrix_sha256"],
        "parent_dataset_matrix_sha256": parent_matrix["matrix_sha256"],
        "count": len(entries),
        "entries": entries,
    }
    payload["map_sha256"] = workflow.object_sha256(payload)
    return payload


def controls_map_path(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    return controls_state(spec, profile, cluster_root=cluster_root) / "maps" / f"{stage}.json"


def validate_controls_map(
    payload: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    parent_matrix: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "status",
        "kind",
        "profile",
        "stage",
        "previous_stage",
        "matrix_sha256",
        "parent_dataset_matrix_sha256",
        "count",
        "entries",
        "map_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != CONTROLS_MAP_SCHEMA:
        raise ValueError("controls stage-map schema mismatch")
    if payload.get("map_sha256") != workflow.object_sha256(
        workflow._without_digest(payload, "map_sha256")
    ):
        raise ValueError("controls stage-map digest mismatch")
    expected = build_controls_map(
        spec, profile, stage, protocol, matrix, parent_matrix
    )
    if dict(payload) != expected:
        raise ValueError("controls stage map differs from its frozen derivation")


def controls_freeze_path(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    return controls_state(spec, profile, cluster_root=cluster_root) / "freeze.json"


def _controls_prerequisites(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None,
) -> dict[str, str]:
    pilot = require_authenticated_main_profile(
        spec, "pilot", cluster_root=cluster_root
    )
    result = {
        "matched_pilot_profile_verification_sha256": pilot[
            "profile_verification_sha256"
        ]
    }
    if profile == "confirmatory":
        confirmatory = require_authenticated_main_profile(
            spec, "confirmatory", cluster_root=cluster_root
        )
        development = require_controls_profile_verified(
            spec, "development", cluster_root=cluster_root
        )
        result.update(
            {
                "matched_confirmatory_profile_verification_sha256": confirmatory[
                    "profile_verification_sha256"
                ],
                "development_controls_profile_verification_sha256": development[
                    "profile_verification_sha256"
                ],
                "development_selection_sha256": development[
                    "selection_sha256"
                ],
            }
        )
    return result


def freeze_controls_profile(
    spec: Mapping[str, Any],
    profile: str,
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    contract = _require_cluster_ready(
        spec, checkout_path, cluster_root=cluster_root
    )
    prerequisites = _controls_prerequisites(
        spec, profile, cluster_root=cluster_root
    )
    output = controls_output(spec, profile, cluster_root=cluster_root)
    executable = str(python or Path(spec["environment_root"]) / "bin" / "python")
    runner = checkout_path / "dreamer_imf_comparison" / "scripts" / "run_neurips_controls.py"
    if not (output / "matrix.json").is_file():
        command = [
            executable,
            str(runner),
            "freeze",
            "--profile",
            profile,
            "--output",
            str(output),
            "--workspace",
            str(checkout_path),
        ]
        if profile == "confirmatory":
            command.extend(
                (
                    "--parent-selection",
                    str(workflow.profile_output(spec, "pilot", cluster_root=cluster_root)),
                    "--controls-selection",
                    str(controls_output(spec, "development", cluster_root=cluster_root)),
                )
            )
        subprocess.run(command, check=True)
    controls, parent, protocol, parent_protocol, source, matrix, parent_matrix = load_controls_run(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    del controls, parent, protocol, parent_protocol
    source_commits = {
        source["git"].get("commit"),
        source["parent_source_manifest"]["git"].get("commit"),
    }
    if source_commits != {contract["source_commit"]}:
        raise ValueError("controls source commits differ from the cluster run contract")
    map_digests: dict[str, str] = {}
    protocol, _, _, matrix = _controls_modules(checkout_path)[0].load_frozen_controls(output)
    for stage in CONTROLS_STAGES:
        payload = build_controls_map(
            spec, profile, stage, protocol, matrix, parent_matrix
        )
        workflow.write_json_once(
            controls_map_path(spec, profile, stage, cluster_root=cluster_root), payload
        )
        map_digests[stage] = payload["map_sha256"]
    slurm_job_id = _slurm_id()
    payload = {
        "schema_version": CONTROLS_FREEZE_SCHEMA,
        "status": "frozen_before_control_execution",
        "profile": profile,
        "source_commit": contract["source_commit"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "runner_freeze_sha256": workflow.file_sha256(output / "freeze.json"),
        "map_sha256": map_digests,
        "prerequisites": prerequisites,
        "slurm_job_id": slurm_job_id,
    }
    payload["cluster_freeze_sha256"] = workflow.object_sha256(payload)
    event = {
        "event": "controls_profile_frozen",
        "profile": profile,
        "cluster_freeze_sha256": payload["cluster_freeze_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "slurm_job_id": slurm_job_id,
    }
    _register_event_once(
        workflow.ledger_path(spec, cluster_root=cluster_root), event
    )
    workflow.write_json_once(
        controls_freeze_path(spec, profile, cluster_root=cluster_root), payload
    )
    return require_controls_freeze(
        spec, profile, checkout_path, cluster_root=cluster_root
    )


def require_controls_freeze(
    spec: Mapping[str, Any],
    profile: str,
    checkout: Path,
    *,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    path = controls_freeze_path(spec, profile, cluster_root=cluster_root)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"controls {profile} cluster freeze is absent")
    payload = workflow.read_json(path)
    required = {
        "schema_version",
        "status",
        "profile",
        "source_commit",
        "run_contract_sha256",
        "matrix_sha256",
        "runner_freeze_sha256",
        "map_sha256",
        "prerequisites",
        "slurm_job_id",
        "cluster_freeze_sha256",
    }
    contract = _require_cluster_ready(spec, checkout, cluster_root=cluster_root)
    prerequisites = _controls_prerequisites(
        spec, profile, cluster_root=cluster_root
    )
    controls, parent, protocol, parent_protocol, source, matrix, parent_matrix = load_controls_run(
        spec, profile, checkout, cluster_root=cluster_root
    )
    del controls, parent, parent_protocol, source
    if (
        set(payload) != required
        or payload.get("schema_version") != CONTROLS_FREEZE_SCHEMA
        or payload.get("status") != "frozen_before_control_execution"
        or payload.get("profile") != profile
        or payload.get("source_commit") != contract["source_commit"]
        or payload.get("run_contract_sha256") != contract["run_contract_sha256"]
        or payload.get("matrix_sha256") != matrix["matrix_sha256"]
        or payload.get("runner_freeze_sha256")
        != workflow.file_sha256(
            controls_output(spec, profile, cluster_root=cluster_root) / "freeze.json"
        )
        or payload.get("prerequisites") != prerequisites
        or payload.get("cluster_freeze_sha256")
        != workflow.object_sha256(
            workflow._without_digest(payload, "cluster_freeze_sha256")
        )
        or not _numeric_slurm_id(payload.get("slurm_job_id"))
    ):
        raise ValueError("controls cluster freeze is invalid")
    digests = {}
    for stage in CONTROLS_STAGES:
        stage_map = _read_regular_json(
            controls_map_path(spec, profile, stage, cluster_root=cluster_root),
            f"controls {stage} map",
        )
        validate_controls_map(
            stage_map,
            spec=spec,
            profile=profile,
            stage=stage,
            protocol=protocol,
            matrix=matrix,
            parent_matrix=parent_matrix,
        )
        digests[stage] = stage_map["map_sha256"]
    if payload.get("map_sha256") != digests:
        raise ValueError("controls cluster freeze map set differs")
    rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    matches = [
        row
        for row in rows
        if row.get("event") == "controls_profile_frozen"
        and row.get("profile") == profile
        and row.get("cluster_freeze_sha256") == payload["cluster_freeze_sha256"]
        and row.get("matrix_sha256") == matrix["matrix_sha256"]
        and row.get("slurm_job_id") == payload["slurm_job_id"]
    ]
    if len(matches) != 1:
        raise ValueError("controls cluster freeze is not uniquely ledger-authenticated")
    return payload


def _controls_entry_cell(
    entry: Mapping[str, Any],
    stage: str,
    matrix: Mapping[str, Any],
    parent_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    source = parent_matrix if stage == "dataset" else matrix
    matches = [cell for cell in source["cells"] if cell["cell_id"] == entry["cell_id"]]
    if len(matches) != 1:
        raise ValueError("controls array entry does not resolve exactly one cell")
    cell = dict(matches[0])
    if cell.get("identity_sha256") != entry.get("identity_sha256"):
        raise ValueError("controls array entry cell identity differs")
    return cell


def _controls_unit_directory(
    spec: Mapping[str, Any],
    output: Path,
    stage: str,
    cell: Mapping[str, Any],
    parent: Any,
    *,
    cluster_root: str | Path | None,
) -> Path:
    directory = (
        parent.stage_directory(output / "parent_dataset", cell)
        if stage == "dataset"
        else output / "cells" / cell["cell_id"]
    )
    return workflow.require_safe_cluster_path(
        spec,
        directory,
        cluster_root=cluster_root,
        allow_leaf_symlink=True,
    )


def _lexical_path(path: Path) -> Path:
    """Return an absolute normalized path without resolving symlinks."""

    return workflow.lexical_path(path)


def _reject_symlink_chain(
    root: Path, target: Path, *, anchor: Path | None = None
) -> tuple[Path, Path]:
    """Reject symlinks at or below ``root`` before any path resolution.

    Calling ``resolve`` first would erase evidence that a registered artifact
    directory was replaced by a symlink.  Keep this lexical check separate,
    then also verify resolved containment to defend against path escapes.
    """

    lexical_root = _lexical_path(root)
    lexical_target = _lexical_path(target)
    lexical_anchor = _lexical_path(anchor or root)
    if lexical_root != lexical_anchor and lexical_anchor not in lexical_root.parents:
        raise ValueError("artifact root escapes its registered workspace")
    if lexical_target != lexical_root and lexical_root not in lexical_target.parents:
        raise ValueError("artifact directory escapes its registered root")
    cursor = lexical_anchor
    if cursor.is_symlink():
        raise ValueError(f"artifact path contains a symlink: {cursor}")
    relative = lexical_target.relative_to(lexical_anchor)
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {cursor}")
    resolved_root = lexical_root.resolve()
    resolved_target = lexical_target.resolve()
    if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
        raise ValueError("artifact directory resolves outside its registered root")
    return lexical_root, lexical_target


def _inventory(
    root: Path, directory: Path, *, anchor: Path | None = None
) -> list[dict[str, Any]]:
    root, directory = _reject_symlink_chain(root, directory, anchor=anchor)
    if not directory.is_dir():
        raise FileNotFoundError(f"artifact directory is absent: {directory}")
    rows = []
    for path in sorted(directory.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"artifact inventory contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": workflow.file_sha256(path),
                }
            )
    if not rows:
        raise ValueError("artifact unit contains no regular files")
    return rows


def _validate_inventory(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor: Path | None = None,
) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("artifact inventory is empty")
    paths = [row.get("path") for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("artifact inventory paths are not canonical")
    lexical_root, _ = _reject_symlink_chain(root, root, anchor=anchor)
    resolved_root = lexical_root.resolve()
    for row in rows:
        if set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("artifact inventory record schema mismatch")
        relative = row["path"]
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or any(part in ("", ".", "..") for part in Path(relative).parts)
        ):
            raise ValueError("artifact inventory path is unsafe")
        path = lexical_root / relative
        _reject_symlink_chain(lexical_root, path, anchor=anchor)
        if not path.is_file() or resolved_root not in path.resolve().parents:
            raise ValueError("artifact inventory target is absent, symlinked, or escaped")
        if path.stat().st_size != row["bytes"] or workflow.file_sha256(path) != row["sha256"]:
            raise ValueError("artifact inventory target changed after verification")


def _controls_artifact_inventory(
    spec: Mapping[str, Any],
    output: Path,
    directory: Path,
    stage: str,
    result: Mapping[str, Any],
    *,
    cluster_root: str | Path | None,
) -> list[dict[str, Any]]:
    """Bind the unit directory plus content-addressed compute IR dependencies."""

    rows = _inventory(
        output,
        directory,
        anchor=workflow.cluster_anchor(spec, cluster_root=cluster_root),
    )
    if stage == "compute":
        digests = sorted(
            {
                digest
                for arm in result.get("arms", {}).values()
                for digest in arm.get("compiler_ir_sha256", {}).values()
            }
        )
        if not digests:
            raise ValueError("controls compute unit has no compiler-IR dependencies")
        for digest in digests:
            if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
                raise ValueError("controls compiler-IR dependency digest is malformed")
            path = output / "compiler_ir" / f"{digest}.txt"
            workflow.require_safe_cluster_path(
                spec, path, cluster_root=cluster_root
            )
            if path.is_symlink() or not path.is_file() or workflow.file_sha256(path) != digest:
                raise ValueError("controls compiler-IR dependency changed")
            rows.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    by_path = {row["path"]: row for row in rows}
    if len(by_path) != len(rows):
        raise ValueError("controls unit artifact inventory contains duplicate paths")
    return [by_path[path] for path in sorted(by_path)]


def validate_controls_unit(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    entry: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controls, parent, protocol, parent_protocol, source, matrix, parent_matrix = load_controls_run(
        spec, profile, checkout, cluster_root=cluster_root
    )
    del source
    cell = _controls_entry_cell(entry, stage, matrix, parent_matrix)
    output = controls_output(spec, profile, cluster_root=cluster_root)
    directory = _controls_unit_directory(
        spec, output, stage, cell, parent, cluster_root=cluster_root
    )
    result = workflow.read_json(directory / "result.json")
    if stage == "dataset":
        parent.validate_dataset_result(
            result,
            cell,
            directory / "dataset.npz",
            protocol=parent_protocol,
            matrix=parent_matrix,
        )
    elif stage == "compute":
        controls.validate_controls_compute_plan(
            result, protocol, parent_protocol, matrix, cell, output
        )
    elif stage == "world":
        controls.validate_world_cell(
            result, cell, protocol, parent_protocol, matrix, output
        )
    elif stage == "rollout":
        controls.validate_rollout_cell(
            result, cell, protocol, parent_protocol, matrix, output
        )
    elif stage == "actor":
        controls.validate_actor_cell(
            result, cell, protocol, parent_protocol, matrix, output
        )
    else:  # pragma: no cover - callers validate earlier
        raise ValueError("unknown controls stage")
    return cell, _controls_artifact_inventory(
        spec,
        output,
        directory,
        stage,
        result,
        cluster_root=cluster_root,
    )


def _controls_unit_state(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    entry: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None,
) -> str:
    _, parent, _, _, _, matrix, parent_matrix = load_controls_run(
        spec, profile, checkout, cluster_root=cluster_root
    )
    cell = _controls_entry_cell(entry, stage, matrix, parent_matrix)
    output = controls_output(spec, profile, cluster_root=cluster_root)
    directory = _controls_unit_directory(
        spec, output, stage, cell, parent, cluster_root=cluster_root
    )
    artifact_state = workflow.tree_state(
        spec, directory, cluster_root=cluster_root
    )
    external = []
    result_path = directory / "result.json"
    if (
        stage == "compute"
        and directory.is_dir()
        and not directory.is_symlink()
        and result_path.is_file()
        and not result_path.is_symlink()
    ):
        try:
            result = workflow.read_json(result_path)
            digests = sorted(
                {
                    digest
                    for arm in result.get("arms", {}).values()
                    for digest in arm.get("compiler_ir_sha256", {}).values()
                    if isinstance(digest, str)
                }
            )
        except Exception:
            digests = []
        for digest in digests:
            path = output / "compiler_ir" / f"{digest}.txt"
            external.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "state": workflow.tree_state(
                        spec, path, cluster_root=cluster_root
                    ),
                }
            )
    receipt_path = controls_unit_marker_path(
        spec, profile, stage, entry["cell_id"], cluster_root=cluster_root
    )
    body: dict[str, Any] = {
        "artifact": artifact_state,
        "external": external,
        "receipt": workflow.tree_state(
            spec, receipt_path, cluster_root=cluster_root
        ),
    }
    return workflow.object_sha256(body)


def incomplete_controls_indices(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
    strong_validation: bool,
) -> list[int]:
    context = load_controls_run(
        spec, profile, checkout, cluster_root=cluster_root
    )
    _, _, protocol, _, _, matrix, parent_matrix = context
    freeze = require_controls_freeze(
        spec, profile, checkout, cluster_root=cluster_root
    )
    preflight = _read_regular_json(
        workflow.preflight_path(spec, cluster_root=cluster_root),
        "GPU preflight",
    )
    ledger_rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    stage_map = _read_regular_json(
        controls_map_path(spec, profile, stage, cluster_root=cluster_root),
        f"controls {stage} map",
    )
    validate_controls_map(
        stage_map,
        spec=spec,
        profile=profile,
        stage=stage,
        protocol=protocol,
        matrix=matrix,
        parent_matrix=parent_matrix,
    )
    incomplete = []
    for entry in stage_map["entries"]:
        receipt = controls_unit_marker_path(
            spec, profile, stage, entry["cell_id"], cluster_root=cluster_root
        )
        if not receipt.is_file() or receipt.is_symlink():
            incomplete.append(entry["array_index"])
            continue
        try:
            require_controls_unit_verified(
                spec,
                profile,
                stage,
                entry,
                checkout=checkout,
                cluster_root=cluster_root,
                authenticate_artifact=True,
                _freeze=freeze,
                _preflight=preflight,
                _ledger_rows=ledger_rows,
                _controls_context=context,
            )
            if strong_validation:
                validate_controls_unit(
                    spec,
                    profile,
                    stage,
                    entry,
                    checkout=checkout,
                    cluster_root=cluster_root,
                )
        except Exception:
            incomplete.append(entry["array_index"])
    return incomplete


def controls_unit_marker_path(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    cell_id: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    if stage not in CONTROLS_STAGES or re.fullmatch(
        r"(?:dataset|compute|world|rollout|actor)-[0-9a-f]{24}", cell_id
    ) is None:
        raise ValueError("controls receipt identity is not canonical")
    return (
        controls_state(spec, profile, cluster_root=cluster_root)
        / "units"
        / stage
        / f"{cell_id}.json"
    )


def _controls_worker_runtime() -> dict[str, Any]:
    import jax

    devices = jax.devices()
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "numpy": importlib.metadata.version("numpy"),
        "dm-control": importlib.metadata.version("dm-control"),
        "mujoco": importlib.metadata.version("mujoco"),
        "backend": jax.default_backend(),
        "visible_device_count": len(devices),
        "device_platforms": [str(device.platform) for device in devices],
        "device_kinds": [
            str(getattr(device, "device_kind", "unknown")) for device in devices
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {
            name: os.environ.get(name)
            for name in ("JAX_ENABLE_X64", "JAX_PLATFORM_NAME", "MUJOCO_GL")
        },
    }


def _validate_controls_worker_runtime(
    runtime: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    if not isinstance(runtime, Mapping):
        raise ValueError("controls worker runtime fingerprint is malformed")
    required = {
        "python",
        "jax",
        "jaxlib",
        "numpy",
        "dm-control",
        "mujoco",
        "backend",
        "visible_device_count",
        "device_platforms",
        "device_kinds",
        "cuda_visible_devices",
        "environment",
    }
    versions = {
        name: runtime.get(name)
        for name in ("python", "jax", "jaxlib", "numpy", "dm-control", "mujoco")
    }
    selector = runtime.get("cuda_visible_devices")
    if (
        set(runtime) != required
        or versions != preflight.get("versions")
        or versions != {name: spec["runtime"][name] for name in versions}
        or runtime.get("environment") != spec["runtime"]["environment"]
        or runtime.get("environment") != preflight.get("environment")
        or runtime.get("backend") != "gpu"
        or runtime.get("backend") != preflight.get("jax_runtime", {}).get("backend")
        or runtime.get("visible_device_count") != 1
        or runtime.get("visible_device_count")
        != preflight.get("jax_runtime", {}).get("visible_device_count")
        or runtime.get("device_platforms") != ["gpu"]
        or runtime.get("device_platforms")
        != preflight.get("jax_runtime", {}).get("device_platforms")
        or runtime.get("device_kinds") != preflight.get("jax_runtime", {}).get(
            "device_kinds"
        )
        or not isinstance(runtime.get("device_kinds"), list)
        or len(runtime["device_kinds"]) != 1
        or spec["scheduler"]["gpu_model_substring"] not in runtime["device_kinds"][0]
        or not isinstance(selector, str)
        or not selector
        or selector != selector.strip()
        or "," in selector
        or selector in {"-1", "NoDevFiles"}
        or selector != preflight.get("slurm", {}).get("cuda_visible_devices")
    ):
        raise ValueError("controls worker runtime is not the registered single L40S lock")


def _validate_controls_result_runtime(
    result: Mapping[str, Any], worker: Mapping[str, Any]
) -> None:
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("controls result lacks a runtime fingerprint")
    expected = {
        "python": worker["python"],
        "jax_version": worker["jax"],
        "jaxlib_version": worker["jaxlib"],
        "backend": worker["backend"],
        "device_platforms": worker["device_platforms"],
        "device_kinds": worker["device_kinds"],
        "visible_device_count": worker["visible_device_count"],
        "cuda_visible_devices": worker["cuda_visible_devices"],
    }
    if any(runtime.get(name) != value for name, value in expected.items()):
        raise ValueError("controls result runtime differs from its GPU worker receipt")


def _controls_receipt_body(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    entry: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]],
    *,
    cluster_root: str | Path | None,
    freeze: Mapping[str, Any],
    array_job_id: str,
    task_id: str,
) -> dict[str, Any]:
    preflight = _read_regular_json(
        workflow.preflight_path(spec, cluster_root=cluster_root),
        "GPU preflight",
    )
    worker_runtime = _controls_worker_runtime()
    _validate_controls_worker_runtime(
        worker_runtime, spec=spec, preflight=preflight
    )
    return {
        "schema_version": CONTROLS_UNIT_SCHEMA,
        "status": "canonical_stage_validator_passed_on_gpu",
        "profile": profile,
        "stage": stage,
        "cell_id": entry["cell_id"],
        "identity_sha256": entry["identity_sha256"],
        "map_sha256": freeze["map_sha256"][stage],
        "run_contract_sha256": freeze["run_contract_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "worker_runtime": worker_runtime,
        "worker_runtime_sha256": workflow.object_sha256(worker_runtime),
        "files": list(files),
        "slurm_array_job_id": array_job_id,
        "slurm_task_id": task_id,
    }


def require_controls_unit_verified(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    entry: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
    authenticate_artifact: bool,
    _freeze: Mapping[str, Any] | None = None,
    _preflight: Mapping[str, Any] | None = None,
    _ledger_rows: Sequence[Mapping[str, Any]] | None = None,
    _controls_context: tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    | None = None,
) -> dict[str, Any]:
    freeze = (
        _freeze
        if _freeze is not None
        else require_controls_freeze(
            spec, profile, checkout, cluster_root=cluster_root
        )
    )
    path = controls_unit_marker_path(
        spec, profile, stage, entry["cell_id"], cluster_root=cluster_root
    )
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"controls GPU receipt is absent: {entry['cell_id']}")
    marker = workflow.read_json(path)
    required = {
        "schema_version",
        "status",
        "profile",
        "stage",
        "cell_id",
        "identity_sha256",
        "map_sha256",
        "run_contract_sha256",
        "preflight_sha256",
        "worker_runtime",
        "worker_runtime_sha256",
        "files",
        "slurm_array_job_id",
        "slurm_task_id",
        "unit_verification_sha256",
    }
    preflight = (
        _preflight
        if _preflight is not None
        else _read_regular_json(
            workflow.preflight_path(spec, cluster_root=cluster_root),
            "GPU preflight",
        )
    )
    if (
        set(marker) != required
        or marker.get("schema_version") != CONTROLS_UNIT_SCHEMA
        or marker.get("status") != "canonical_stage_validator_passed_on_gpu"
        or marker.get("profile") != profile
        or marker.get("stage") != stage
        or marker.get("cell_id") != entry["cell_id"]
        or marker.get("identity_sha256") != entry["identity_sha256"]
        or marker.get("map_sha256") != freeze["map_sha256"][stage]
        or marker.get("run_contract_sha256") != freeze["run_contract_sha256"]
        or marker.get("preflight_sha256") != preflight["preflight_sha256"]
        or marker.get("worker_runtime_sha256")
        != workflow.object_sha256(marker.get("worker_runtime"))
        or not isinstance(marker.get("files"), list)
        or not marker["files"]
        or not _numeric_slurm_id(marker.get("slurm_array_job_id"))
        or not _numeric_slurm_id(marker.get("slurm_task_id"))
        or marker.get("unit_verification_sha256")
        != workflow.object_sha256(
            workflow._without_digest(marker, "unit_verification_sha256")
        )
    ):
        raise ValueError("controls GPU verification receipt is invalid")
    _validate_controls_worker_runtime(
        marker["worker_runtime"], spec=spec, preflight=preflight
    )
    if authenticate_artifact:
        context = (
            _controls_context
            if _controls_context is not None
            else load_controls_run(
                spec, profile, checkout, cluster_root=cluster_root
            )
        )
        _, parent, _, _, _, matrix, parent_matrix = context
        cell = _controls_entry_cell(entry, stage, matrix, parent_matrix)
        output = controls_output(spec, profile, cluster_root=cluster_root)
        directory = _controls_unit_directory(
            spec, output, stage, cell, parent, cluster_root=cluster_root
        )
        result = workflow.read_json(directory / "result.json")
        if stage != "dataset":
            _validate_controls_result_runtime(result, marker["worker_runtime"])
        expected_files = _controls_artifact_inventory(
            spec,
            output,
            directory,
            stage,
            result,
            cluster_root=cluster_root,
        )
        if marker.get("files") != expected_files:
            raise ValueError("controls GPU receipt artifact inventory changed")
        _validate_inventory(
            output,
            marker["files"],
            anchor=workflow.cluster_anchor(spec, cluster_root=cluster_root),
        )
    rows = (
        list(_ledger_rows)
        if _ledger_rows is not None
        else workflow.verify_ledger(
            workflow.ledger_path(spec, cluster_root=cluster_root)
        )
    )
    matches = [
        row
        for row in rows
        if row.get("event") == "controls_unit_gpu_verified"
        and row.get("profile") == profile
        and row.get("stage") == stage
        and row.get("cell_id") == entry["cell_id"]
        and row.get("unit_verification_sha256")
        == marker["unit_verification_sha256"]
        and row.get("slurm_array_job_id") == marker["slurm_array_job_id"]
        and row.get("slurm_task_id") == marker["slurm_task_id"]
    ]
    if len(matches) != 1:
        raise ValueError("controls GPU receipt is not uniquely ledger-authenticated")
    return marker


def controls_stage_marker_path(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    return controls_state(spec, profile, cluster_root=cluster_root) / "stages" / f"{stage}.json"


def _previous_controls_stage(stage: str) -> str | None:
    if stage not in CONTROLS_STAGES:
        raise ValueError("unknown controls stage")
    index = CONTROLS_STAGES.index(stage)
    return None if index == 0 else CONTROLS_STAGES[index - 1]


def require_controls_stage_verified(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    context = load_controls_run(
        spec, profile, checkout, cluster_root=cluster_root
    )
    _, _, protocol, _, _, matrix, parent_matrix = context
    freeze = require_controls_freeze(
        spec, profile, checkout, cluster_root=cluster_root
    )
    preflight = _read_regular_json(
        workflow.preflight_path(spec, cluster_root=cluster_root),
        "GPU preflight",
    )
    ledger_rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    stage_map = _read_regular_json(
        controls_map_path(spec, profile, stage, cluster_root=cluster_root),
        f"controls {stage} map",
    )
    validate_controls_map(
        stage_map,
        spec=spec,
        profile=profile,
        stage=stage,
        protocol=protocol,
        matrix=matrix,
        parent_matrix=parent_matrix,
    )
    path = controls_stage_marker_path(
        spec, profile, stage, cluster_root=cluster_root
    )
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"entire controls {stage} stage is not verified")
    marker = workflow.read_json(path)
    required = {
        "schema_version",
        "status",
        "profile",
        "stage",
        "matrix_sha256",
        "map_sha256",
        "run_contract_sha256",
        "verified_unit_count",
        "units",
        "slurm_job_id",
        "stage_verification_sha256",
    }
    contract = _read_regular_json(
        workflow.run_contract_path(spec, cluster_root=cluster_root),
        "cluster run contract",
    )
    if (
        set(marker) != required
        or marker.get("schema_version") != CONTROLS_STAGE_SCHEMA
        or marker.get("status") != "verified_complete_on_gpu"
        or marker.get("profile") != profile
        or marker.get("stage") != stage
        or marker.get("matrix_sha256") != matrix["matrix_sha256"]
        or marker.get("map_sha256") != stage_map["map_sha256"]
        or marker.get("run_contract_sha256") != contract["run_contract_sha256"]
        or marker.get("verified_unit_count") != stage_map["count"]
        or not _numeric_slurm_id(marker.get("slurm_job_id"))
        or marker.get("stage_verification_sha256")
        != workflow.object_sha256(
            workflow._without_digest(marker, "stage_verification_sha256")
        )
    ):
        raise ValueError("controls stage marker is invalid")
    units = marker.get("units")
    if not isinstance(units, list) or [row.get("cell_id") for row in units] != [
        entry["cell_id"] for entry in stage_map["entries"]
    ]:
        raise ValueError("controls stage marker unit order differs from its map")
    for row, entry in zip(units, stage_map["entries"], strict=True):
        if set(row) != {
            "cell_id",
            "identity_sha256",
            "unit_verification_sha256",
        }:
            raise ValueError("controls stage unit schema mismatch")
        if (
            row["cell_id"] != entry["cell_id"]
            or row["identity_sha256"] != entry["identity_sha256"]
        ):
            raise ValueError("controls stage unit identity mismatch")
        receipt = require_controls_unit_verified(
            spec,
            profile,
            stage,
            entry,
            checkout=checkout,
            cluster_root=cluster_root,
            authenticate_artifact=True,
            _freeze=freeze,
            _preflight=preflight,
            _ledger_rows=ledger_rows,
            _controls_context=context,
        )
        if (
            row["unit_verification_sha256"]
            != receipt["unit_verification_sha256"]
        ):
            raise ValueError("controls stage unit receipt changed")
    rows = ledger_rows
    matches = [
        row
        for row in rows
        if row.get("event") == "controls_stage_verified"
        and row.get("profile") == profile
        and row.get("stage") == stage
        and row.get("stage_verification_sha256")
        == marker["stage_verification_sha256"]
        and row.get("slurm_job_id") == marker["slurm_job_id"]
    ]
    if len(matches) != 1:
        raise ValueError("controls stage marker is not uniquely ledger-authenticated")
    return marker


def _require_previous_controls_stage(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    checkout: Path,
    cluster_root: str | Path | None,
) -> None:
    previous = _previous_controls_stage(stage)
    if previous is not None:
        require_controls_stage_verified(
            spec,
            profile,
            previous,
            checkout=checkout,
            cluster_root=cluster_root,
        )


def _quarantine_move(
    spec: Mapping[str, Any],
    source: Path,
    destination: Path,
    *,
    cluster_root: str | Path | None,
) -> bool:
    """Atomically preserve an invalid artifact so its canonical path can rerun."""

    if not source.exists() and not source.is_symlink():
        return False
    workflow.quarantine_cluster_node(
        spec,
        source,
        destination,
        cluster_root=cluster_root,
    )
    return True


def _retry_membership(
    spec: Mapping[str, Any],
    path: str | Path,
    *,
    kind: str,
    profile: str,
    stage: str,
    base_map: Mapping[str, Any],
    array_index: int,
    cluster_root: str | Path | None,
) -> dict[str, Any]:
    supplied = _lexical_path(Path(path))
    state = _lexical_path(
        (
        controls_state(spec, profile, cluster_root=cluster_root)
        if kind == "controls"
        else pixel_state(spec, cluster_root=cluster_root)
        )
    )
    _reject_symlink_chain(state, supplied)
    if not supplied.is_file():
        raise ValueError("retry map is absent from its registered state root")
    resolved = supplied.resolve()
    payload = _read_regular_json(resolved, "supplementary retry map")
    validate_supplementary_retry_map(
        payload,
        kind=kind,
        profile=profile,
        stage=stage,
        base_map=base_map,
    )
    _require_retry_registered(spec, payload, resolved, cluster_root=cluster_root)
    if array_index not in payload["indices"]:
        raise ValueError("array index is absent from the authenticated retry map")
    return payload


def _require_selected_retry_state(
    retry: Mapping[str, Any], array_index: int, current_state: str
) -> None:
    position = retry["indices"].index(array_index)
    if retry["selected_state_sha256"][position] != current_state:
        raise ValueError(
            "supplementary retry map became stale before array-cell execution"
        )


def _quarantine_controls_unit(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    entry: Mapping[str, Any],
    retry: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None,
) -> tuple[list[str], bool]:
    _, parent, _, _, _, matrix, parent_matrix = load_controls_run(
        spec, profile, checkout, cluster_root=cluster_root
    )
    output = controls_output(spec, profile, cluster_root=cluster_root)
    cell = _controls_entry_cell(entry, stage, matrix, parent_matrix)
    directory = _controls_unit_directory(
        spec, output, stage, cell, parent, cluster_root=cluster_root
    )
    quarantine = (
        controls_state(spec, profile, cluster_root=cluster_root)
        / "quarantine"
        / stage
        / retry["audit_id"]
        / entry["cell_id"]
    )
    moved = []
    receipt_path = controls_unit_marker_path(
        spec, profile, stage, entry["cell_id"], cluster_root=cluster_root
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        if _quarantine_move(
            spec,
            receipt_path,
            quarantine / "receipt.json",
            cluster_root=cluster_root,
        ):
            moved.append(receipt_path.as_posix())
    artifact_valid = False
    if directory.exists() and not directory.is_symlink():
        try:
            validate_controls_unit(
                spec,
                profile,
                stage,
                entry,
                checkout=checkout,
                cluster_root=cluster_root,
            )
            artifact_valid = True
        except Exception:
            artifact_valid = False
    if artifact_valid:
        return moved, True
    # Read external IR identities before moving result.json.  Compute arrays are
    # serialized.  Preserve valid content-addressed shared IR, but quarantine a
    # corrupt object so the canonical runner can recreate it.
    digests = []
    result_path = directory / "result.json"
    if (
        stage == "compute"
        and directory.is_dir()
        and not directory.is_symlink()
        and result_path.is_file()
        and not result_path.is_symlink()
    ):
        try:
            result = workflow.read_json(result_path)
            digests = sorted(
                {
                    digest
                    for arm in result.get("arms", {}).values()
                    for digest in arm.get("compiler_ir_sha256", {}).values()
                    if re.fullmatch(r"[0-9a-f]{64}", str(digest))
                }
            )
        except Exception:
            digests = []
    if _quarantine_move(
        spec,
        directory,
        quarantine / "unit",
        cluster_root=cluster_root,
    ):
        moved.append(directory.as_posix())
    for digest in digests:
        source = output / "compiler_ir" / f"{digest}.txt"
        if (
            (source.is_symlink() or (source.is_file() and workflow.file_sha256(source) != digest))
            and _quarantine_move(
                spec,
                source,
                quarantine / "compiler_ir" / source.name,
                cluster_root=cluster_root,
            )
        ):
            moved.append(source.as_posix())
    return moved, False


def run_controls_array_cell(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    array_index: int,
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    python: str | Path | None = None,
    retry_map_path: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    freeze = require_controls_freeze(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    _require_previous_controls_stage(
        spec,
        profile,
        stage,
        checkout=checkout_path,
        cluster_root=cluster_root,
    )
    _, _, protocol, _, _, matrix, parent_matrix = load_controls_run(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    stage_map = _read_regular_json(
        controls_map_path(spec, profile, stage, cluster_root=cluster_root),
        f"controls {stage} map",
    )
    validate_controls_map(
        stage_map,
        spec=spec,
        profile=profile,
        stage=stage,
        protocol=protocol,
        matrix=matrix,
        parent_matrix=parent_matrix,
    )
    if isinstance(array_index, bool) or not isinstance(array_index, int) or not 0 <= array_index < stage_map["count"]:
        raise IndexError("controls array index is outside its immutable map")
    entry = stage_map["entries"][array_index]
    if entry["array_index"] != array_index:
        raise ValueError("controls array map index is noncanonical")
    array_job_id, task_id = _slurm_array_identity(array_index)
    ledger = workflow.ledger_path(spec, cluster_root=cluster_root)
    if controls_stage_marker_path(
        spec, profile, stage, cluster_root=cluster_root
    ).exists():
        raise RuntimeError("a verified controls stage cannot be executed again")
    reuse_existing = False
    if retry_map_path is not None:
        retry = _retry_membership(
            spec,
            retry_map_path,
            kind="controls",
            profile=profile,
            stage=stage,
            base_map=stage_map,
            array_index=array_index,
            cluster_root=cluster_root,
        )
        _require_selected_retry_state(
            retry,
            array_index,
            _controls_unit_state(
                spec,
                profile,
                stage,
                entry,
                checkout=checkout_path,
                cluster_root=cluster_root,
            ),
        )
        moved, reuse_existing = _quarantine_controls_unit(
            spec,
            profile,
            stage,
            entry,
            retry,
            checkout=checkout_path,
            cluster_root=cluster_root,
        )
        if moved:
            workflow.append_ledger(
                ledger,
                {
                    "event": "controls_unit_quarantined_for_retry",
                    "profile": profile,
                    "stage": stage,
                    "array_index": array_index,
                    "cell_id": entry["cell_id"],
                    "retry_map_sha256": retry["retry_map_sha256"],
                    "moved_paths": moved,
                    "slurm_array_job_id": array_job_id,
                    "slurm_task_id": task_id,
                },
            )
    workflow.append_ledger(
        ledger,
        {
            "event": "controls_unit_started",
            "profile": profile,
            "stage": stage,
            "array_index": array_index,
            "cell_id": entry["cell_id"],
            "slurm_array_job_id": array_job_id,
            "slurm_task_id": task_id,
        },
    )
    try:
        if not reuse_existing:
            runner = (
                checkout_path
                / "dreamer_imf_comparison"
                / "scripts"
                / "run_neurips_controls.py"
            )
            executable = str(
                python or Path(spec["environment_root"]) / "bin" / "python"
            )
            command = [
                executable,
                str(runner),
                "run",
                "--profile",
                profile,
                "--output",
                str(controls_output(spec, profile, cluster_root=cluster_root)),
                "--workspace",
                str(checkout_path),
            ]
            command.extend(
                ("--dataset-cell-id", entry["cell_id"])
                if stage == "dataset"
                else ("--cell-id", entry["cell_id"])
            )
            subprocess.run(command, check=True)
        _, files = validate_controls_unit(
            spec,
            profile,
            stage,
            entry,
            checkout=checkout_path,
            cluster_root=cluster_root,
        )
        marker = _controls_receipt_body(
            spec,
            profile,
            stage,
            entry,
            files,
            cluster_root=cluster_root,
            freeze=freeze,
            array_job_id=array_job_id,
            task_id=task_id,
        )
        marker["unit_verification_sha256"] = workflow.object_sha256(marker)
        event = {
            "event": "controls_unit_gpu_verified",
            "profile": profile,
            "stage": stage,
            "cell_id": entry["cell_id"],
            "unit_verification_sha256": marker["unit_verification_sha256"],
            "slurm_array_job_id": array_job_id,
            "slurm_task_id": task_id,
        }
        _register_event_once(ledger, event)
        workflow.write_json_once(
            controls_unit_marker_path(
                spec,
                profile,
                stage,
                entry["cell_id"],
                cluster_root=cluster_root,
            ),
            marker,
        )
    except BaseException as error:
        workflow.append_ledger(
            ledger,
            {
                "event": "controls_unit_failed",
                "profile": profile,
                "stage": stage,
                "array_index": array_index,
                "cell_id": entry["cell_id"],
                "slurm_array_job_id": array_job_id,
                "slurm_task_id": task_id,
                "error_type": type(error).__name__,
            },
        )
        raise
    state_sha256 = marker["unit_verification_sha256"]
    workflow.append_ledger(
        ledger,
        {
            "event": "controls_unit_completed",
            "profile": profile,
            "stage": stage,
            "array_index": array_index,
            "cell_id": entry["cell_id"],
            "state_sha256": state_sha256,
            "slurm_array_job_id": array_job_id,
            "slurm_task_id": task_id,
        },
    )
    return require_controls_unit_verified(
        spec,
        profile,
        stage,
        entry,
        checkout=checkout_path,
        cluster_root=cluster_root,
        authenticate_artifact=True,
    )


def verify_controls_stage(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    freeze = require_controls_freeze(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    _require_previous_controls_stage(
        spec,
        profile,
        stage,
        checkout=checkout_path,
        cluster_root=cluster_root,
    )
    existing = controls_stage_marker_path(
        spec, profile, stage, cluster_root=cluster_root
    )
    if existing.is_file() and not existing.is_symlink():
        return require_controls_stage_verified(
            spec,
            profile,
            stage,
            checkout=checkout_path,
            cluster_root=cluster_root,
        )
    context = load_controls_run(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    _, _, protocol, _, _, matrix, parent_matrix = context
    preflight = _read_regular_json(
        workflow.preflight_path(spec, cluster_root=cluster_root),
        "GPU preflight",
    )
    ledger_rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    stage_map = _read_regular_json(
        controls_map_path(spec, profile, stage, cluster_root=cluster_root),
        f"controls {stage} map",
    )
    validate_controls_map(
        stage_map,
        spec=spec,
        profile=profile,
        stage=stage,
        protocol=protocol,
        matrix=matrix,
        parent_matrix=parent_matrix,
    )
    units = []
    for entry in stage_map["entries"]:
        receipt = require_controls_unit_verified(
            spec,
            profile,
            stage,
            entry,
            checkout=checkout_path,
            cluster_root=cluster_root,
            authenticate_artifact=True,
            _freeze=freeze,
            _preflight=preflight,
            _ledger_rows=ledger_rows,
            _controls_context=context,
        )
        units.append(
            {
                "cell_id": entry["cell_id"],
                "identity_sha256": entry["identity_sha256"],
                "unit_verification_sha256": receipt[
                    "unit_verification_sha256"
                ],
            }
        )
    slurm_job_id = _slurm_id()
    marker = {
        "schema_version": CONTROLS_STAGE_SCHEMA,
        "status": "verified_complete_on_gpu",
        "profile": profile,
        "stage": stage,
        "matrix_sha256": matrix["matrix_sha256"],
        "map_sha256": stage_map["map_sha256"],
        "run_contract_sha256": freeze["run_contract_sha256"],
        "verified_unit_count": stage_map["count"],
        "units": units,
        "slurm_job_id": slurm_job_id,
    }
    marker["stage_verification_sha256"] = workflow.object_sha256(marker)
    event = {
        "event": "controls_stage_verified",
        "profile": profile,
        "stage": stage,
        "stage_verification_sha256": marker["stage_verification_sha256"],
        "verified_unit_count": stage_map["count"],
        "slurm_job_id": slurm_job_id,
    }
    _register_event_once(
        workflow.ledger_path(spec, cluster_root=cluster_root), event
    )
    workflow.write_json_once(existing, marker)
    return require_controls_stage_verified(
        spec,
        profile,
        stage,
        checkout=checkout_path,
        cluster_root=cluster_root,
    )


def _safe_audit_id(value: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
        raise ValueError("retry audit id must be a lowercase filesystem-safe token")
    return value


def supplementary_retry_path(
    spec: Mapping[str, Any],
    kind: str,
    profile: str,
    stage: str,
    audit_id: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    audit_id = _safe_audit_id(audit_id)
    if kind == "controls":
        base = controls_state(spec, profile, cluster_root=cluster_root)
    elif kind == "pixel" and profile == "hard" and stage == "artifact":
        base = pixel_state(spec, cluster_root=cluster_root)
    else:
        raise ValueError("unsupported supplementary retry identity")
    return base / "retry_maps" / stage / f"retry-{audit_id}.json"


def _build_retry_map(
    *,
    kind: str,
    profile: str,
    stage: str,
    base_map: Mapping[str, Any],
    indices: Sequence[int],
    states: Sequence[str],
    audit_id: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    if not indices:
        raise ValueError("strong retry audit found no missing or invalid units")
    normalized = list(indices)
    if normalized != sorted(set(normalized)) or len(states) != len(normalized):
        raise ValueError("retry indices and state snapshots are noncanonical")
    entries = [base_map["entries"][index] for index in normalized]
    payload = {
        "schema_version": SUPPLEMENTARY_RETRY_SCHEMA,
        "status": "gpu_audited_missing_or_invalid_only",
        "kind": kind,
        "profile": profile,
        "stage": stage,
        "base_map_sha256": base_map["map_sha256"],
        "audit_id": _safe_audit_id(audit_id),
        "indices": normalized,
        "cell_ids": [entry["cell_id"] for entry in entries],
        "selected_state_sha256": list(states),
        "slurm_job_id": slurm_job_id,
    }
    payload["retry_map_sha256"] = workflow.object_sha256(payload)
    return payload


def validate_supplementary_retry_map(
    payload: Mapping[str, Any],
    *,
    kind: str,
    profile: str,
    stage: str,
    base_map: Mapping[str, Any],
    current_states: Sequence[str] | None = None,
) -> None:
    required = {
        "schema_version",
        "status",
        "kind",
        "profile",
        "stage",
        "base_map_sha256",
        "audit_id",
        "indices",
        "cell_ids",
        "selected_state_sha256",
        "slurm_job_id",
        "retry_map_sha256",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != SUPPLEMENTARY_RETRY_SCHEMA
        or payload.get("status") != "gpu_audited_missing_or_invalid_only"
        or payload.get("kind") != kind
        or payload.get("profile") != profile
        or payload.get("stage") != stage
        or payload.get("base_map_sha256") != base_map["map_sha256"]
        or payload.get("retry_map_sha256")
        != workflow.object_sha256(
            workflow._without_digest(payload, "retry_map_sha256")
        )
        or not _numeric_slurm_id(payload.get("slurm_job_id"))
    ):
        raise ValueError("supplementary retry-map identity or digest mismatch")
    _safe_audit_id(str(payload.get("audit_id")))
    indices = payload.get("indices")
    states = payload.get("selected_state_sha256")
    if (
        not isinstance(indices, list)
        or not indices
        or indices != sorted(set(indices))
        or not all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
        or any(index < 0 or index >= base_map["count"] for index in indices)
        or not isinstance(states, list)
        or len(states) != len(indices)
        or not all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in states)
    ):
        raise ValueError("supplementary retry-map indices or state snapshots are invalid")
    expected_ids = [base_map["entries"][index]["cell_id"] for index in indices]
    if payload.get("cell_ids") != expected_ids:
        raise ValueError("supplementary retry-map cell identities differ")
    if current_states is not None and list(current_states) != states:
        raise ValueError("supplementary retry audit is stale; selected unit bytes changed")


def _require_retry_registered(
    spec: Mapping[str, Any],
    payload: Mapping[str, Any],
    path: Path,
    *,
    cluster_root: str | Path | None,
) -> None:
    rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    matches = [
        row
        for row in rows
        if row.get("event") == "supplementary_retry_audited"
        and row.get("kind") == payload["kind"]
        and row.get("profile") == payload["profile"]
        and row.get("stage") == payload["stage"]
        and row.get("retry_map") == path.as_posix()
        and row.get("retry_map_sha256") == payload["retry_map_sha256"]
        and row.get("slurm_job_id") == payload["slurm_job_id"]
    ]
    if len(matches) != 1:
        raise ValueError("supplementary retry map is not uniquely ledger-authenticated")


def audit_controls_retry(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    audit_id: str,
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    require_controls_freeze(spec, profile, checkout_path, cluster_root=cluster_root)
    _require_previous_controls_stage(
        spec, profile, stage, checkout=checkout_path, cluster_root=cluster_root
    )
    base_map = _read_regular_json(
        controls_map_path(spec, profile, stage, cluster_root=cluster_root),
        f"controls {stage} map",
    )
    indices = incomplete_controls_indices(
        spec,
        profile,
        stage,
        checkout=checkout_path,
        cluster_root=cluster_root,
        strong_validation=True,
    )
    states = [
        _controls_unit_state(
            spec,
            profile,
            stage,
            base_map["entries"][index],
            checkout=checkout_path,
            cluster_root=cluster_root,
        )
        for index in indices
    ]
    slurm_job_id = _slurm_id()
    payload = _build_retry_map(
        kind="controls",
        profile=profile,
        stage=stage,
        base_map=base_map,
        indices=indices,
        states=states,
        audit_id=audit_id,
        slurm_job_id=slurm_job_id,
    )
    path = supplementary_retry_path(
        spec,
        "controls",
        profile,
        stage,
        audit_id,
        cluster_root=cluster_root,
    )
    event = {
        "event": "supplementary_retry_audited",
        "kind": "controls",
        "profile": profile,
        "stage": stage,
        "retry_map": path.as_posix(),
        "retry_map_sha256": payload["retry_map_sha256"],
        "slurm_job_id": slurm_job_id,
    }
    _register_event_once(workflow.ledger_path(spec, cluster_root=cluster_root), event)
    workflow.write_json_once(path, payload)
    _require_retry_registered(spec, payload, path, cluster_root=cluster_root)
    return payload


def validate_controls_retry_for_submission(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    path: str | Path,
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    supplied = _lexical_path(Path(path))
    expected_parent = _lexical_path(
        controls_state(spec, profile, cluster_root=cluster_root)
    )
    _reject_symlink_chain(expected_parent, supplied)
    if not supplied.is_file():
        raise ValueError("controls retry map is absent")
    resolved = supplied.resolve()
    payload = _read_regular_json(resolved, "controls retry map")
    base_map = _read_regular_json(
        controls_map_path(spec, profile, stage, cluster_root=cluster_root),
        f"controls {stage} map",
    )
    states = [
        _controls_unit_state(
            spec,
            profile,
            stage,
            base_map["entries"][index],
            checkout=checkout,
            cluster_root=cluster_root,
        )
        for index in payload.get("indices", [])
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < base_map["count"]
    ]
    validate_supplementary_retry_map(
        payload,
        kind="controls",
        profile=profile,
        stage=stage,
        base_map=base_map,
        current_states=states,
    )
    _require_retry_registered(spec, payload, resolved, cluster_root=cluster_root)
    return payload


def controls_profile_marker_path(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    return controls_state(spec, profile, cluster_root=cluster_root) / "PROFILE_VERIFIED.json"


def _controls_final_files(output: Path, profile: str) -> dict[str, str]:
    names = ["analysis.json", "REPORT.md", "artifact_manifest.json", "FINALIZED.json"]
    if profile == "development":
        names.append("controls_selection.json")
    result = {}
    for name in names:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"controls final artifact is absent: {name}")
        result[name] = workflow.file_sha256(path)
    return result


def finalize_controls_profile(
    spec: Mapping[str, Any],
    profile: str,
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    freeze = require_controls_freeze(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    stage_digests = {}
    for stage in CONTROLS_STAGES:
        marker = require_controls_stage_verified(
            spec,
            profile,
            stage,
            checkout=checkout_path,
            cluster_root=cluster_root,
        )
        stage_digests[stage] = marker["stage_verification_sha256"]
    prerequisites = _controls_prerequisites(
        spec, profile, cluster_root=cluster_root
    )
    output = controls_output(spec, profile, cluster_root=cluster_root)
    executable = str(python or Path(spec["environment_root"]) / "bin" / "python")
    runner = checkout_path / "dreamer_imf_comparison" / "scripts" / "run_neurips_controls.py"
    common = [
        "--profile",
        profile,
        "--output",
        str(output),
        "--workspace",
        str(checkout_path),
    ]
    subprocess.run([executable, str(runner), "finalize", *common], check=True)
    subprocess.run([executable, str(runner), "verify", *common], check=True)
    _, _, _, _, _, matrix, _ = load_controls_run(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    files = _controls_final_files(output, profile)
    selection_sha256 = (
        workflow.read_json(output / "controls_selection.json")["selection_sha256"]
        if profile == "development"
        else None
    )
    slurm_job_id = _slurm_id()
    marker = {
        "schema_version": CONTROLS_PROFILE_SCHEMA,
        "status": "verified_complete_on_gpu",
        "profile": profile,
        "source_commit": freeze["source_commit"],
        "run_contract_sha256": freeze["run_contract_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "cluster_freeze_sha256": freeze["cluster_freeze_sha256"],
        "stage_verification_sha256": stage_digests,
        "expected_execution_units": spec["supplementary"]["controls"][profile][
            "expected_execution_units"
        ],
        "result_files": files,
        "selection_sha256": selection_sha256,
        "prerequisites": prerequisites,
        "slurm_job_id": slurm_job_id,
    }
    marker["profile_verification_sha256"] = workflow.object_sha256(marker)
    event = {
        "event": "controls_profile_verified",
        "profile": profile,
        "profile_verification_sha256": marker["profile_verification_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "result_files": files,
        "slurm_job_id": slurm_job_id,
    }
    _register_event_once(workflow.ledger_path(spec, cluster_root=cluster_root), event)
    workflow.write_json_once(
        controls_profile_marker_path(spec, profile, cluster_root=cluster_root), marker
    )
    return require_controls_profile_verified(
        spec, profile, cluster_root=cluster_root, checkout=checkout_path
    )


def require_controls_profile_verified(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None = None,
    checkout: Path | None = None,
) -> dict[str, Any]:
    checkout_path = (
        Path(checkout).resolve()
        if checkout is not None
        else workflow.source_root(spec, cluster_root=cluster_root)
    )
    path = controls_profile_marker_path(
        spec, profile, cluster_root=cluster_root
    )
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"controls {profile} profile is not verified")
    marker = workflow.read_json(path)
    freeze = require_controls_freeze(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    _, _, _, _, _, matrix, _ = load_controls_run(
        spec, profile, checkout_path, cluster_root=cluster_root
    )
    stage_digests = {}
    for stage in CONTROLS_STAGES:
        stage_marker = require_controls_stage_verified(
            spec,
            profile,
            stage,
            checkout=checkout_path,
            cluster_root=cluster_root,
        )
        stage_digests[stage] = stage_marker["stage_verification_sha256"]
    output = controls_output(spec, profile, cluster_root=cluster_root)
    files = _controls_final_files(output, profile)
    prerequisites = _controls_prerequisites(
        spec, profile, cluster_root=cluster_root
    )
    required = {
        "schema_version",
        "status",
        "profile",
        "source_commit",
        "run_contract_sha256",
        "matrix_sha256",
        "cluster_freeze_sha256",
        "stage_verification_sha256",
        "expected_execution_units",
        "result_files",
        "selection_sha256",
        "prerequisites",
        "slurm_job_id",
        "profile_verification_sha256",
    }
    expected_selection = (
        workflow.read_json(output / "controls_selection.json")["selection_sha256"]
        if profile == "development"
        else None
    )
    if (
        set(marker) != required
        or marker.get("schema_version") != CONTROLS_PROFILE_SCHEMA
        or marker.get("status") != "verified_complete_on_gpu"
        or marker.get("profile") != profile
        or marker.get("source_commit") != freeze["source_commit"]
        or marker.get("run_contract_sha256") != freeze["run_contract_sha256"]
        or marker.get("matrix_sha256") != matrix["matrix_sha256"]
        or marker.get("cluster_freeze_sha256") != freeze["cluster_freeze_sha256"]
        or marker.get("stage_verification_sha256") != stage_digests
        or marker.get("expected_execution_units")
        != spec["supplementary"]["controls"][profile]["expected_execution_units"]
        or marker.get("result_files") != files
        or marker.get("selection_sha256") != expected_selection
        or marker.get("prerequisites") != prerequisites
        or not _numeric_slurm_id(marker.get("slurm_job_id"))
        or marker.get("profile_verification_sha256")
        != workflow.object_sha256(
            workflow._without_digest(marker, "profile_verification_sha256")
        )
    ):
        raise ValueError("controls profile marker is invalid")
    rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    matches = [
        row
        for row in rows
        if row.get("event") == "controls_profile_verified"
        and row.get("profile") == profile
        and row.get("profile_verification_sha256")
        == marker["profile_verification_sha256"]
        and row.get("matrix_sha256") == marker["matrix_sha256"]
        and row.get("result_files") == marker["result_files"]
        and row.get("slurm_job_id") == marker["slurm_job_id"]
    ]
    if len(matches) != 1:
        raise ValueError("controls profile marker is not uniquely ledger-authenticated")
    return marker


def _pixel_module(checkout: Path) -> Any:
    _add_source_paths(checkout)
    from dreamer_imf_compare import pixel_benchmark

    return pixel_benchmark


def _validate_pixel_protocol(spec: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    plan = spec["supplementary"]["pixel"]
    hard = protocol["profiles"]["hard"]
    if (
        protocol.get("schema_version") != "trajectory-imf-pixel-secondary-v1"
        or hard.get("tasks") != plan["tasks"]
        or hard.get("seeds") != plan["seeds"]
        or hard.get("tracks") != plan["tracks"]
        or len(hard["tasks"]) * len(hard["seeds"]) * len(hard["tracks"])
        != plan["expected_artifacts"]
    ):
        raise ValueError("hard pixel protocol differs from the cluster specification")


def build_pixel_map(
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    source_commit: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    _validate_pixel_protocol(spec, protocol)
    plan = spec["supplementary"]["pixel"]
    entries = []
    identities = []
    for track in plan["tracks"]:
        for task in plan["tasks"]:
            for seed in plan["seeds"]:
                identity = {
                    "profile": "hard",
                    "track": track,
                    "task": task,
                    "seed": int(seed),
                    "source_commit": source_commit,
                    "source_manifest_sha256": source_manifest_sha256,
                }
                digest = workflow.object_sha256(identity)
                identities.append(
                    {
                        **identity,
                        "identity_sha256": digest,
                        "cell_id": f"pixel-{digest[:24]}",
                        "relative_output": f"{track}/{task}/seed_{int(seed)}",
                    }
                )
    identities.sort(key=lambda row: row["cell_id"])
    for index, identity in enumerate(identities):
        entries.append({"array_index": index, **identity})
    payload = {
        "schema_version": PIXEL_MAP_SCHEMA,
        "status": "frozen_before_pixel_outcomes",
        "kind": "pixel",
        "profile": "hard",
        "protocol_sha256": workflow.file_sha256(
            workflow.PROJECT / "pixel_benchmark_protocol.json"
        ),
        "base_protocol_sha256": workflow.file_sha256(
            workflow.PROJECT / "matched_objective_protocol.json"
        ),
        "source_commit": source_commit,
        "source_manifest_sha256": source_manifest_sha256,
        "count": len(entries),
        "entries": entries,
    }
    if payload["count"] != plan["expected_artifacts"]:
        raise ValueError("hard pixel map does not contain exactly 30 units")
    payload["map_sha256"] = workflow.object_sha256(payload)
    return payload


def pixel_map_path(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return pixel_state(spec, cluster_root=cluster_root) / "map.json"


def pixel_freeze_path(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return pixel_state(spec, cluster_root=cluster_root) / "freeze.json"


def validate_pixel_map(
    payload: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    source_commit: str,
    source_manifest_sha256: str,
) -> None:
    required = {
        "schema_version",
        "status",
        "kind",
        "profile",
        "protocol_sha256",
        "base_protocol_sha256",
        "source_commit",
        "source_manifest_sha256",
        "count",
        "entries",
        "map_sha256",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != PIXEL_MAP_SCHEMA
        or payload.get("map_sha256")
        != workflow.object_sha256(workflow._without_digest(payload, "map_sha256"))
        or dict(payload)
        != build_pixel_map(
            spec,
            protocol,
            source_commit=source_commit,
            source_manifest_sha256=source_manifest_sha256,
        )
    ):
        raise ValueError("pixel map differs from its immutable derivation")


def freeze_pixel_profile(
    spec: Mapping[str, Any],
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    contract = _require_cluster_ready(
        spec, checkout_path, cluster_root=cluster_root
    )
    pilot = require_authenticated_main_profile(
        spec, "pilot", cluster_root=cluster_root
    )
    pixel = _pixel_module(checkout_path)
    protocol = pixel.load_protocol()
    source_manifest_sha256 = workflow.object_sha256(pixel.source_manifest())
    stage_map = build_pixel_map(
        spec,
        protocol,
        source_commit=contract["source_commit"],
        source_manifest_sha256=source_manifest_sha256,
    )
    workflow.write_json_once(
        pixel_map_path(spec, cluster_root=cluster_root), stage_map
    )
    slurm_job_id = _slurm_id()
    payload = {
        "schema_version": PIXEL_FREEZE_SCHEMA,
        "status": "frozen_before_pixel_outcomes",
        "source_commit": contract["source_commit"],
        "run_contract_sha256": contract["run_contract_sha256"],
        "map_sha256": stage_map["map_sha256"],
        "expected_artifacts": stage_map["count"],
        "matched_pilot_profile_verification_sha256": pilot[
            "profile_verification_sha256"
        ],
        "slurm_job_id": slurm_job_id,
    }
    payload["cluster_freeze_sha256"] = workflow.object_sha256(payload)
    event = {
        "event": "pixel_profile_frozen",
        "cluster_freeze_sha256": payload["cluster_freeze_sha256"],
        "map_sha256": stage_map["map_sha256"],
        "slurm_job_id": slurm_job_id,
    }
    _register_event_once(workflow.ledger_path(spec, cluster_root=cluster_root), event)
    workflow.write_json_once(pixel_freeze_path(spec, cluster_root=cluster_root), payload)
    return require_pixel_freeze(
        spec, checkout=checkout_path, cluster_root=cluster_root
    )


def require_pixel_freeze(
    spec: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    path = pixel_freeze_path(spec, cluster_root=cluster_root)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("hard pixel cluster freeze is absent")
    payload = workflow.read_json(path)
    contract = _require_cluster_ready(spec, checkout, cluster_root=cluster_root)
    pilot = require_authenticated_main_profile(
        spec, "pilot", cluster_root=cluster_root
    )
    pixel = _pixel_module(checkout)
    protocol = pixel.load_protocol()
    source_manifest_sha256 = workflow.object_sha256(pixel.source_manifest())
    stage_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    validate_pixel_map(
        stage_map,
        spec=spec,
        protocol=protocol,
        source_commit=contract["source_commit"],
        source_manifest_sha256=source_manifest_sha256,
    )
    required = {
        "schema_version",
        "status",
        "source_commit",
        "run_contract_sha256",
        "map_sha256",
        "expected_artifacts",
        "matched_pilot_profile_verification_sha256",
        "slurm_job_id",
        "cluster_freeze_sha256",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != PIXEL_FREEZE_SCHEMA
        or payload.get("status") != "frozen_before_pixel_outcomes"
        or payload.get("source_commit") != contract["source_commit"]
        or payload.get("run_contract_sha256") != contract["run_contract_sha256"]
        or payload.get("map_sha256") != stage_map["map_sha256"]
        or payload.get("expected_artifacts") != stage_map["count"]
        or payload.get("matched_pilot_profile_verification_sha256")
        != pilot["profile_verification_sha256"]
        or not _numeric_slurm_id(payload.get("slurm_job_id"))
        or payload.get("cluster_freeze_sha256")
        != workflow.object_sha256(
            workflow._without_digest(payload, "cluster_freeze_sha256")
        )
    ):
        raise ValueError("hard pixel cluster freeze is invalid")
    rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    matches = [
        row
        for row in rows
        if row.get("event") == "pixel_profile_frozen"
        and row.get("cluster_freeze_sha256") == payload["cluster_freeze_sha256"]
        and row.get("map_sha256") == payload["map_sha256"]
        and row.get("slurm_job_id") == payload["slurm_job_id"]
    ]
    if len(matches) != 1:
        raise ValueError("pixel cluster freeze is not uniquely ledger-authenticated")
    return payload


def pixel_unit_marker_path(
    spec: Mapping[str, Any],
    cell_id: str,
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    if re.fullmatch(r"pixel-[0-9a-f]{24}", cell_id) is None:
        raise ValueError("pixel cell id is not canonical")
    return pixel_state(spec, cluster_root=cluster_root) / "units" / f"{cell_id}.json"


def _pixel_artifact_root(
    spec: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    cluster_root: str | Path | None,
) -> Path:
    relative = str(entry["relative_output"])
    if Path(relative).is_absolute() or any(
        part in ("", ".", "..") for part in Path(relative).parts
    ):
        raise ValueError("pixel output path is not canonical")
    return workflow.require_safe_cluster_path(
        spec,
        pixel_output(spec, cluster_root=cluster_root) / relative,
        cluster_root=cluster_root,
        allow_leaf_symlink=True,
    )


def _pixel_receipt_body(
    spec: Mapping[str, Any],
    entry: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    cluster_root: str | Path | None,
    freeze: Mapping[str, Any],
    array_job_id: str,
    task_id: str,
) -> dict[str, Any]:
    root = _pixel_artifact_root(spec, entry, cluster_root=cluster_root)
    preflight = _read_regular_json(
        workflow.preflight_path(spec, cluster_root=cluster_root),
        "GPU preflight",
    )
    return {
        "schema_version": PIXEL_UNIT_SCHEMA,
        "status": "checkpoint_and_compute_verified_on_gpu",
        "cell_id": entry["cell_id"],
        "identity_sha256": entry["identity_sha256"],
        "map_sha256": freeze["map_sha256"],
        "run_contract_sha256": freeze["run_contract_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "profile": "hard",
        "track": entry["track"],
        "task": entry["task"],
        "seed": entry["seed"],
        "manifest_sha256": workflow.file_sha256(root / "manifest.json"),
        "seal_sha256": workflow.file_sha256(root / "seal.json"),
        "result_sha256": workflow.file_sha256(root / "result.json"),
        "runtime_sha256": workflow.object_sha256(result["runtime"]),
        "checkpoint_prediction_replay": True,
        "compiler_compute_rederived": True,
        "slurm_array_job_id": array_job_id,
        "slurm_task_id": task_id,
    }


def run_pixel_array_cell(
    spec: Mapping[str, Any],
    array_index: int,
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    python: str | Path | None = None,
    retry_map_path: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    freeze = require_pixel_freeze(
        spec, checkout=checkout_path, cluster_root=cluster_root
    )
    stage_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    if isinstance(array_index, bool) or not isinstance(array_index, int) or not 0 <= array_index < stage_map["count"]:
        raise IndexError("pixel array index is outside its immutable map")
    entry = stage_map["entries"][array_index]
    if entry["array_index"] != array_index:
        raise ValueError("pixel array map index is noncanonical")
    array_job_id, task_id = _slurm_array_identity(array_index)
    ledger = workflow.ledger_path(spec, cluster_root=cluster_root)
    if pixel_profile_marker_path(spec, cluster_root=cluster_root).exists():
        raise RuntimeError("a verified pixel profile cannot be executed again")
    if retry_map_path is not None:
        retry = _retry_membership(
            spec,
            retry_map_path,
            kind="pixel",
            profile="hard",
            stage="artifact",
            base_map=stage_map,
            array_index=array_index,
            cluster_root=cluster_root,
        )
        _require_selected_retry_state(
            retry,
            array_index,
            _pixel_unit_state(
                spec,
                entry,
                checkout=checkout_path,
                cluster_root=cluster_root,
            ),
        )
        root = _pixel_artifact_root(spec, entry, cluster_root=cluster_root)
        receipt_path = pixel_unit_marker_path(
            spec, entry["cell_id"], cluster_root=cluster_root
        )
        quarantine = (
            pixel_state(spec, cluster_root=cluster_root)
            / "quarantine"
            / retry["audit_id"]
            / entry["cell_id"]
        )
        moved = []
        if receipt_path.exists() or receipt_path.is_symlink():
            if _quarantine_move(
                spec,
                receipt_path,
                quarantine / "receipt.json",
                cluster_root=cluster_root,
            ):
                moved.append(receipt_path.as_posix())
        if root.exists() and not root.is_symlink():
            pixel = _pixel_module(checkout_path)
            try:
                pixel.verify_artifact(
                    root, recompute_predictions=False, rederive_compute=False
                )
                artifact_valid = True
            except Exception:
                artifact_valid = False
            if not artifact_valid and _quarantine_move(
                spec,
                root,
                quarantine / "artifact",
                cluster_root=cluster_root,
            ):
                moved.append(root.as_posix())
        elif root.is_symlink() and _quarantine_move(
            spec,
            root,
            quarantine / "artifact",
            cluster_root=cluster_root,
        ):
            moved.append(root.as_posix())
        if moved:
            workflow.append_ledger(
                ledger,
                {
                    "event": "pixel_unit_quarantined_for_retry",
                    "array_index": array_index,
                    "cell_id": entry["cell_id"],
                    "retry_map_sha256": retry["retry_map_sha256"],
                    "moved_paths": moved,
                    "slurm_array_job_id": array_job_id,
                    "slurm_task_id": task_id,
                },
            )
    workflow.append_ledger(
        ledger,
        {
            "event": "pixel_unit_started",
            "array_index": array_index,
            "cell_id": entry["cell_id"],
            "slurm_array_job_id": array_job_id,
            "slurm_task_id": task_id,
        },
    )
    root = _pixel_artifact_root(spec, entry, cluster_root=cluster_root)
    runner = checkout_path / "dreamer_imf_comparison" / "scripts" / "run_pixel_benchmark.py"
    executable = str(python or Path(spec["environment_root"]) / "bin" / "python")
    command = [
        executable,
        str(runner),
        "run",
        "--profile",
        "hard",
        "--track",
        entry["track"],
        "--task",
        entry["task"],
        "--seed",
        str(entry["seed"]),
        "--output",
        str(root),
    ]
    try:
        subprocess.run(command, check=True)
        pixel = _pixel_module(checkout_path)
        result = pixel.verify_artifact(
            root, recompute_predictions=True, rederive_compute=True
        )
        observed_identity = (
            result.get("profile"),
            result.get("track"),
            result.get("task"),
            result.get("seed"),
        )
        expected_identity = (
            "hard",
            entry["track"],
            entry["task"],
            entry["seed"],
        )
        if observed_identity != expected_identity:
            raise ValueError("pixel result identity differs from its immutable map entry")
        marker = _pixel_receipt_body(
            spec,
            entry,
            result,
            cluster_root=cluster_root,
            freeze=freeze,
            array_job_id=array_job_id,
            task_id=task_id,
        )
        marker["unit_verification_sha256"] = workflow.object_sha256(marker)
        event = {
            "event": "pixel_unit_gpu_verified",
            "cell_id": entry["cell_id"],
            "unit_verification_sha256": marker["unit_verification_sha256"],
            "slurm_array_job_id": array_job_id,
            "slurm_task_id": task_id,
        }
        _register_event_once(ledger, event)
        workflow.write_json_once(
            pixel_unit_marker_path(
                spec, entry["cell_id"], cluster_root=cluster_root
            ),
            marker,
        )
    except BaseException as error:
        workflow.append_ledger(
            ledger,
            {
                "event": "pixel_unit_failed",
                "array_index": array_index,
                "cell_id": entry["cell_id"],
                "slurm_array_job_id": array_job_id,
                "slurm_task_id": task_id,
                "error_type": type(error).__name__,
            },
        )
        raise
    return require_pixel_unit_verified(
        spec,
        entry,
        checkout=checkout_path,
        cluster_root=cluster_root,
        authenticate_artifact=True,
    )


def require_pixel_unit_verified(
    spec: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
    authenticate_artifact: bool,
) -> dict[str, Any]:
    freeze = require_pixel_freeze(
        spec, checkout=checkout, cluster_root=cluster_root
    )
    path = pixel_unit_marker_path(
        spec, entry["cell_id"], cluster_root=cluster_root
    )
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"pixel GPU verification receipt is absent: {entry['cell_id']}")
    marker = workflow.read_json(path)
    required = {
        "schema_version",
        "status",
        "cell_id",
        "identity_sha256",
        "map_sha256",
        "run_contract_sha256",
        "preflight_sha256",
        "profile",
        "track",
        "task",
        "seed",
        "manifest_sha256",
        "seal_sha256",
        "result_sha256",
        "runtime_sha256",
        "checkpoint_prediction_replay",
        "compiler_compute_rederived",
        "slurm_array_job_id",
        "slurm_task_id",
        "unit_verification_sha256",
    }
    root = _pixel_artifact_root(spec, entry, cluster_root=cluster_root)
    preflight = _read_regular_json(
        workflow.preflight_path(spec, cluster_root=cluster_root),
        "GPU preflight",
    )
    if (
        set(marker) != required
        or marker.get("schema_version") != PIXEL_UNIT_SCHEMA
        or marker.get("status") != "checkpoint_and_compute_verified_on_gpu"
        or marker.get("cell_id") != entry["cell_id"]
        or marker.get("identity_sha256") != entry["identity_sha256"]
        or marker.get("map_sha256") != freeze["map_sha256"]
        or marker.get("run_contract_sha256") != freeze["run_contract_sha256"]
        or marker.get("preflight_sha256") != preflight["preflight_sha256"]
        or (marker.get("profile"), marker.get("track"), marker.get("task"), marker.get("seed"))
        != ("hard", entry["track"], entry["task"], entry["seed"])
        or marker.get("manifest_sha256") != workflow.file_sha256(root / "manifest.json")
        or marker.get("seal_sha256") != workflow.file_sha256(root / "seal.json")
        or marker.get("result_sha256") != workflow.file_sha256(root / "result.json")
        or marker.get("checkpoint_prediction_replay") is not True
        or marker.get("compiler_compute_rederived") is not True
        or not _numeric_slurm_id(marker.get("slurm_array_job_id"))
        or not _numeric_slurm_id(marker.get("slurm_task_id"))
        or marker.get("unit_verification_sha256")
        != workflow.object_sha256(
            workflow._without_digest(marker, "unit_verification_sha256")
        )
    ):
        raise ValueError("pixel GPU verification receipt is invalid")
    if authenticate_artifact:
        pixel = _pixel_module(checkout)
        result = pixel.verify_artifact(
            root, recompute_predictions=False, rederive_compute=False
        )
        if (
            result.get("profile"),
            result.get("track"),
            result.get("task"),
            result.get("seed"),
        ) != ("hard", entry["track"], entry["task"], entry["seed"]):
            raise ValueError("authenticated pixel artifact identity differs from its map")
        if workflow.object_sha256(result["runtime"]) != marker["runtime_sha256"]:
            raise ValueError("pixel runtime differs from its GPU receipt")
    rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    matches = [
        row
        for row in rows
        if row.get("event") == "pixel_unit_gpu_verified"
        and row.get("cell_id") == entry["cell_id"]
        and row.get("unit_verification_sha256")
        == marker["unit_verification_sha256"]
        and row.get("slurm_array_job_id") == marker["slurm_array_job_id"]
        and row.get("slurm_task_id") == marker["slurm_task_id"]
    ]
    if len(matches) != 1:
        raise ValueError("pixel GPU receipt is not uniquely ledger-authenticated")
    return marker


def _pixel_unit_state(
    spec: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None,
) -> str:
    del checkout
    root = _pixel_artifact_root(spec, entry, cluster_root=cluster_root)
    artifact_state = workflow.tree_state(spec, root, cluster_root=cluster_root)
    receipt_path = pixel_unit_marker_path(
        spec, entry["cell_id"], cluster_root=cluster_root
    )
    body: dict[str, Any] = {
        "artifact": artifact_state,
        "receipt": workflow.tree_state(
            spec, receipt_path, cluster_root=cluster_root
        ),
    }
    return workflow.object_sha256(body)


def incomplete_pixel_indices(
    spec: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
    strong_validation: bool,
) -> list[int]:
    stage_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    incomplete = []
    for entry in stage_map["entries"]:
        marker = pixel_unit_marker_path(
            spec, entry["cell_id"], cluster_root=cluster_root
        )
        if not marker.is_file() or marker.is_symlink():
            incomplete.append(entry["array_index"])
        else:
            try:
                require_pixel_unit_verified(
                    spec,
                    entry,
                    checkout=checkout,
                    cluster_root=cluster_root,
                    authenticate_artifact=strong_validation,
                )
            except Exception:
                incomplete.append(entry["array_index"])
    return incomplete


def audit_pixel_retry(
    spec: Mapping[str, Any],
    audit_id: str,
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    require_pixel_freeze(spec, checkout=checkout_path, cluster_root=cluster_root)
    base_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    indices = incomplete_pixel_indices(
        spec,
        checkout=checkout_path,
        cluster_root=cluster_root,
        strong_validation=True,
    )
    states = [
        _pixel_unit_state(
            spec,
            base_map["entries"][index],
            checkout=checkout_path,
            cluster_root=cluster_root,
        )
        for index in indices
    ]
    slurm_job_id = _slurm_id()
    payload = _build_retry_map(
        kind="pixel",
        profile="hard",
        stage="artifact",
        base_map=base_map,
        indices=indices,
        states=states,
        audit_id=audit_id,
        slurm_job_id=slurm_job_id,
    )
    path = supplementary_retry_path(
        spec,
        "pixel",
        "hard",
        "artifact",
        audit_id,
        cluster_root=cluster_root,
    )
    event = {
        "event": "supplementary_retry_audited",
        "kind": "pixel",
        "profile": "hard",
        "stage": "artifact",
        "retry_map": path.as_posix(),
        "retry_map_sha256": payload["retry_map_sha256"],
        "slurm_job_id": slurm_job_id,
    }
    _register_event_once(workflow.ledger_path(spec, cluster_root=cluster_root), event)
    workflow.write_json_once(path, payload)
    _require_retry_registered(spec, payload, path, cluster_root=cluster_root)
    return payload


def validate_pixel_retry_for_submission(
    spec: Mapping[str, Any],
    path: str | Path,
    *,
    checkout: Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    supplied = _lexical_path(Path(path))
    expected_parent = _lexical_path(pixel_state(spec, cluster_root=cluster_root))
    _reject_symlink_chain(expected_parent, supplied)
    if not supplied.is_file():
        raise ValueError("pixel retry map is absent")
    resolved = supplied.resolve()
    payload = _read_regular_json(resolved, "pixel retry map")
    base_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    states = [
        _pixel_unit_state(
            spec,
            base_map["entries"][index],
            checkout=checkout,
            cluster_root=cluster_root,
        )
        for index in payload.get("indices", [])
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < base_map["count"]
    ]
    validate_supplementary_retry_map(
        payload,
        kind="pixel",
        profile="hard",
        stage="artifact",
        base_map=base_map,
        current_states=states,
    )
    _require_retry_registered(spec, payload, resolved, cluster_root=cluster_root)
    return payload


def pixel_profile_marker_path(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return pixel_state(spec, cluster_root=cluster_root) / "PROFILE_VERIFIED.json"


def pixel_aggregate_path(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return pixel_output(spec, cluster_root=cluster_root) / "aggregate.json"


def finalize_pixel_profile(
    spec: Mapping[str, Any],
    *,
    checkout: str | Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    freeze = require_pixel_freeze(
        spec, checkout=checkout_path, cluster_root=cluster_root
    )
    stage_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    receipts = []
    roots = []
    identities = set()
    for entry in stage_map["entries"]:
        marker = require_pixel_unit_verified(
            spec,
            entry,
            checkout=checkout_path,
            cluster_root=cluster_root,
            authenticate_artifact=True,
        )
        receipts.append(
            {
                "cell_id": entry["cell_id"],
                "unit_verification_sha256": marker["unit_verification_sha256"],
            }
        )
        roots.append(_pixel_artifact_root(spec, entry, cluster_root=cluster_root))
        identities.add((entry["track"], entry["task"], entry["seed"]))
    if len(receipts) != 30 or len(identities) != 30:
        raise ValueError("pixel finalization requires 30 unique frozen identities")
    pixel = _pixel_module(checkout_path)
    tracks = list(spec["supplementary"]["pixel"]["tracks"])
    aggregate = pixel_aggregate_path(spec, cluster_root=cluster_root)
    if aggregate.is_symlink():
        raise ValueError("pixel aggregate must not be a symlink")
    if aggregate.is_file():
        expected = pixel.verify_aggregate(aggregate, roots)
    else:
        expected = pixel.aggregate_artifacts(roots, aggregate, tracks=tracks)
    if (
        expected.get("requested_tracks") != tracks
        or expected.get("expected_input_count") != 30
        or set(expected.get("tracks", {})) != set(tracks)
    ):
        raise ValueError("pixel aggregate does not contain both registered tracks")
    pilot = require_authenticated_main_profile(
        spec, "pilot", cluster_root=cluster_root
    )
    slurm_job_id = _slurm_id()
    marker = {
        "schema_version": PIXEL_PROFILE_SCHEMA,
        "status": "verified_aggregated_complete_on_gpu",
        "source_commit": freeze["source_commit"],
        "run_contract_sha256": freeze["run_contract_sha256"],
        "cluster_freeze_sha256": freeze["cluster_freeze_sha256"],
        "map_sha256": freeze["map_sha256"],
        "expected_artifacts": 30,
        "unit_verifications": receipts,
        "aggregate_sha256": workflow.file_sha256(aggregate),
        "aggregate_analysis_sha256": expected["analysis_sha256"],
        "tracks": sorted(expected["tracks"]),
        "matched_pilot_profile_verification_sha256": pilot[
            "profile_verification_sha256"
        ],
        "slurm_job_id": slurm_job_id,
    }
    marker["profile_verification_sha256"] = workflow.object_sha256(marker)
    event = {
        "event": "pixel_profile_verified",
        "profile_verification_sha256": marker["profile_verification_sha256"],
        "map_sha256": marker["map_sha256"],
        "aggregate_sha256": marker["aggregate_sha256"],
        "slurm_job_id": slurm_job_id,
    }
    _register_event_once(workflow.ledger_path(spec, cluster_root=cluster_root), event)
    workflow.write_json_once(
        pixel_profile_marker_path(spec, cluster_root=cluster_root), marker
    )
    return require_pixel_profile_verified(
        spec, checkout=checkout_path, cluster_root=cluster_root
    )


def require_pixel_profile_verified(
    spec: Mapping[str, Any],
    *,
    checkout: Path | None = None,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = (
        Path(checkout).resolve()
        if checkout is not None
        else workflow.source_root(spec, cluster_root=cluster_root)
    )
    path = pixel_profile_marker_path(spec, cluster_root=cluster_root)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("hard pixel profile is not verified")
    marker = workflow.read_json(path)
    freeze = require_pixel_freeze(
        spec, checkout=checkout_path, cluster_root=cluster_root
    )
    stage_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    receipts = []
    roots = []
    for entry in stage_map["entries"]:
        receipt = require_pixel_unit_verified(
            spec,
            entry,
            checkout=checkout_path,
            cluster_root=cluster_root,
            authenticate_artifact=True,
        )
        receipts.append(
            {
                "cell_id": entry["cell_id"],
                "unit_verification_sha256": receipt["unit_verification_sha256"],
            }
        )
        roots.append(_pixel_artifact_root(spec, entry, cluster_root=cluster_root))
    aggregate_path = pixel_aggregate_path(spec, cluster_root=cluster_root)
    if aggregate_path.is_symlink() or not aggregate_path.is_file():
        raise RuntimeError("hard pixel aggregate is absent")
    pixel = _pixel_module(checkout_path)
    aggregate = pixel.verify_aggregate(aggregate_path, roots)
    expected_tracks = list(spec["supplementary"]["pixel"]["tracks"])
    if (
        aggregate.get("requested_tracks") != expected_tracks
        or aggregate.get("expected_input_count") != 30
        or set(aggregate.get("tracks", {})) != set(expected_tracks)
    ):
        raise ValueError("hard pixel aggregate matrix differs from the frozen map")
    pilot = require_authenticated_main_profile(
        spec, "pilot", cluster_root=cluster_root
    )
    required = {
        "schema_version",
        "status",
        "source_commit",
        "run_contract_sha256",
        "cluster_freeze_sha256",
        "map_sha256",
        "expected_artifacts",
        "unit_verifications",
        "aggregate_sha256",
        "aggregate_analysis_sha256",
        "tracks",
        "matched_pilot_profile_verification_sha256",
        "slurm_job_id",
        "profile_verification_sha256",
    }
    if (
        set(marker) != required
        or marker.get("schema_version") != PIXEL_PROFILE_SCHEMA
        or marker.get("status") != "verified_aggregated_complete_on_gpu"
        or marker.get("source_commit") != freeze["source_commit"]
        or marker.get("run_contract_sha256") != freeze["run_contract_sha256"]
        or marker.get("cluster_freeze_sha256") != freeze["cluster_freeze_sha256"]
        or marker.get("map_sha256") != stage_map["map_sha256"]
        or marker.get("expected_artifacts") != 30
        or marker.get("unit_verifications") != receipts
        or marker.get("aggregate_sha256") != workflow.file_sha256(aggregate_path)
        or marker.get("aggregate_analysis_sha256") != aggregate["analysis_sha256"]
        or marker.get("tracks") != sorted(spec["supplementary"]["pixel"]["tracks"])
        or marker.get("matched_pilot_profile_verification_sha256")
        != pilot["profile_verification_sha256"]
        or not _numeric_slurm_id(marker.get("slurm_job_id"))
        or marker.get("profile_verification_sha256")
        != workflow.object_sha256(
            workflow._without_digest(marker, "profile_verification_sha256")
        )
    ):
        raise ValueError("hard pixel profile marker is invalid")
    rows = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    matches = [
        row
        for row in rows
        if row.get("event") == "pixel_profile_verified"
        and row.get("profile_verification_sha256")
        == marker["profile_verification_sha256"]
        and row.get("map_sha256") == marker["map_sha256"]
        and row.get("aggregate_sha256") == marker["aggregate_sha256"]
        and row.get("slurm_job_id") == marker["slurm_job_id"]
    ]
    if len(matches) != 1:
        raise ValueError("hard pixel profile marker is not uniquely ledger-authenticated")
    return marker


def audit_pixel_retry_for_submission_state(
    spec: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    checkout: Path,
    cluster_root: str | Path | None,
) -> list[str]:
    base_map = _read_regular_json(
        pixel_map_path(spec, cluster_root=cluster_root), "hard pixel map"
    )
    return [
        _pixel_unit_state(
            spec,
            base_map["entries"][index],
            checkout=checkout,
            cluster_root=cluster_root,
        )
        for index in payload["indices"]
    ]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "freeze-controls",
            "run-controls-cell",
            "audit-controls-retry",
            "verify-controls-stage",
            "finalize-controls",
            "verify-controls-profile",
            "freeze-pixel",
            "run-pixel-cell",
            "audit-pixel-retry",
            "finalize-pixel",
            "verify-pixel-profile",
        ),
    )
    parser.add_argument("--controls-profile", choices=CONTROLS_PROFILES)
    parser.add_argument("--stage", choices=CONTROLS_STAGES)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--audit-id")
    parser.add_argument("--cluster-root")
    parser.add_argument("--checkout")
    parser.add_argument("--python")
    parser.add_argument("--retry-map")
    arguments = parser.parse_args()
    spec = workflow.load_spec()
    root = Path(arguments.cluster_root).resolve() if arguments.cluster_root else None
    checkout = (
        Path(arguments.checkout).resolve()
        if arguments.checkout
        else workflow.source_root(spec, cluster_root=root)
    )
    if "controls" in arguments.command and arguments.controls_profile is None:
        parser.error("controls commands require --controls-profile")
    if arguments.command in (
        "run-controls-cell",
        "audit-controls-retry",
        "verify-controls-stage",
    ) and arguments.stage is None:
        parser.error(f"{arguments.command} requires --stage")
    if arguments.command in ("run-controls-cell", "run-pixel-cell") and arguments.array_index is None:
        parser.error(f"{arguments.command} requires --array-index")
    if arguments.command in ("audit-controls-retry", "audit-pixel-retry"):
        audit_id = arguments.audit_id or _slurm_id()
    else:
        audit_id = None
    if arguments.command == "freeze-controls":
        result = freeze_controls_profile(
            spec,
            arguments.controls_profile,
            checkout=checkout,
            cluster_root=root,
            python=arguments.python,
        )
    elif arguments.command == "run-controls-cell":
        result = run_controls_array_cell(
            spec,
            arguments.controls_profile,
            arguments.stage,
            arguments.array_index,
            checkout=checkout,
            cluster_root=root,
            python=arguments.python,
            retry_map_path=arguments.retry_map,
        )
    elif arguments.command == "audit-controls-retry":
        result = audit_controls_retry(
            spec,
            arguments.controls_profile,
            arguments.stage,
            str(audit_id),
            checkout=checkout,
            cluster_root=root,
        )
    elif arguments.command == "verify-controls-stage":
        result = verify_controls_stage(
            spec,
            arguments.controls_profile,
            arguments.stage,
            checkout=checkout,
            cluster_root=root,
        )
    elif arguments.command == "finalize-controls":
        result = finalize_controls_profile(
            spec,
            arguments.controls_profile,
            checkout=checkout,
            cluster_root=root,
            python=arguments.python,
        )
    elif arguments.command == "verify-controls-profile":
        result = require_controls_profile_verified(
            spec,
            arguments.controls_profile,
            checkout=checkout,
            cluster_root=root,
        )
    elif arguments.command == "freeze-pixel":
        result = freeze_pixel_profile(
            spec, checkout=checkout, cluster_root=root
        )
    elif arguments.command == "run-pixel-cell":
        result = run_pixel_array_cell(
            spec,
            arguments.array_index,
            checkout=checkout,
            cluster_root=root,
            python=arguments.python,
            retry_map_path=arguments.retry_map,
        )
    elif arguments.command == "audit-pixel-retry":
        result = audit_pixel_retry(
            spec, str(audit_id), checkout=checkout, cluster_root=root
        )
    elif arguments.command == "finalize-pixel":
        result = finalize_pixel_profile(
            spec, checkout=checkout, cluster_root=root
        )
    else:
        result = require_pixel_profile_verified(
            spec, checkout=checkout, cluster_root=root
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
