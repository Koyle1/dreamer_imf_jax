#!/usr/bin/env python3
"""Fail-closed local orchestration primitives for the Leipzig GPU study.

This module does not submit jobs.  It freezes array-index maps, runs exactly
one matrix cell, verifies complete stages, and records a hash-chained ledger.
`submit.py` is the only component that talks to Slurm.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence


CLUSTER_DIR = Path(__file__).resolve().parent
PROJECT = CLUSTER_DIR.parents[1]
WORKSPACE = PROJECT.parent
SPEC_PATH = CLUSTER_DIR / "cluster_spec.json"
LOCK_PATH = PROJECT / "requirements-neurips-cuda12-lock.txt"
STAGES = ("dataset", "compute_plan", "world_model", "rollout", "actor")
MAP_SCHEMA = "trajectory-imf-slurm-cell-map-v1"
RUN_CONTRACT_SCHEMA = "trajectory-imf-cluster-run-contract-v1"
STAGE_MARKER_SCHEMA = "trajectory-imf-cluster-stage-verification-v1"
PROFILE_MARKER_SCHEMA = "trajectory-imf-cluster-profile-verification-v1"
LEDGER_SCHEMA = "trajectory-imf-cluster-ledger-event-v1"
PREFLIGHT_SCHEMA = "trajectory-imf-cluster-gpu-preflight-v1"
RETRY_MAP_SCHEMA = "trajectory-imf-cluster-retry-map-v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lexical_path(path: str | Path) -> Path:
    """Normalize an absolute path without erasing symlink evidence."""

    return Path(os.path.abspath(os.fspath(path)))


def cluster_anchor(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return lexical_path(cluster_root if cluster_root is not None else spec["workspace_root"])


def require_safe_cluster_path(
    spec: Mapping[str, Any],
    path: str | Path,
    *,
    cluster_root: str | Path | None = None,
    allow_leaf_symlink: bool = False,
) -> Path:
    """Require lexical containment and reject every symlink below the workspace.

    The optional leaf exception is used only to snapshot or atomically quarantine
    an invalid symlink node; all of its ancestors remain non-symlinks.
    """

    anchor = cluster_anchor(spec, cluster_root=cluster_root)
    target = lexical_path(path)
    if target != anchor and anchor not in target.parents:
        raise ValueError("cluster path escapes the registered workspace root")
    relative = target.relative_to(anchor)
    cursor = anchor
    components = relative.parts[:-1] if allow_leaf_symlink and relative.parts else relative.parts
    if cursor.is_symlink():
        raise ValueError(f"cluster path contains a symlink: {cursor}")
    for component in components:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"cluster path contains a symlink: {cursor}")
    return target


def tree_state(
    spec: Mapping[str, Any],
    target: str | Path,
    *,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    """Content snapshot that records symlinks without ever following them."""

    root = require_safe_cluster_path(
        spec,
        target,
        cluster_root=cluster_root,
        allow_leaf_symlink=True,
    )
    if not root.exists() and not root.is_symlink():
        return {"exists": False, "entries": []}
    records: list[dict[str, Any]] = []

    def visit(path: Path, relative: str) -> None:
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISLNK(metadata.st_mode):
            records.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "mode": mode,
                    "target": os.readlink(path),
                }
            )
            return
        if stat.S_ISREG(metadata.st_mode):
            records.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "bytes": metadata.st_size,
                    "sha256": file_sha256(path),
                }
            )
            return
        if stat.S_ISDIR(metadata.st_mode):
            records.append({"path": relative, "type": "directory", "mode": mode})
            for child in sorted(path.iterdir(), key=lambda value: value.name):
                child_relative = child.name if not relative else f"{relative}/{child.name}"
                visit(child, child_relative)
            return
        records.append(
            {
                "path": relative,
                "type": "other",
                "mode": mode,
                "kind": stat.S_IFMT(metadata.st_mode),
            }
        )

    visit(root, "")
    return {"exists": True, "entries": records}


def tree_state_sha256(
    spec: Mapping[str, Any],
    target: str | Path,
    *,
    cluster_root: str | Path | None = None,
) -> str:
    return object_sha256(tree_state(spec, target, cluster_root=cluster_root))


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json_once(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Create an immutable JSON artifact or verify its exact prior contents."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload)
    if destination.is_symlink():
        raise ValueError(f"immutable artifact must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"immutable artifact must be a regular file: {destination}")
        if read_json(destination) != normalized:
            raise FileExistsError(f"immutable artifact differs: {destination}")
        return destination
    data = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(
                f"immutable artifact must be a regular non-symlink file: {destination}"
            )
        if read_json(destination) != normalized:
            raise FileExistsError(f"immutable artifact differs: {destination}")
    return destination


def load_spec(path: str | Path = SPEC_PATH) -> dict[str, Any]:
    spec = read_json(path)
    validate_spec(spec)
    return spec


def validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != "trajectory-imf-neurips-cluster-spec-v1":
        raise ValueError("cluster specification schema mismatch")
    expected_paths = {
        "workspace_root": "/work2/ci72buri-dreamer_imf_neurips",
        "source_root": "/work2/ci72buri-dreamer_imf_neurips/source",
        "environment_root": "/work2/ci72buri-dreamer_imf_neurips/venv-cuda12",
        "state_root": "/work2/ci72buri-dreamer_imf_neurips/cluster_state",
        "results_root": "/work2/ci72buri-dreamer_imf_neurips/results",
    }
    if any(spec.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("cluster workspace paths differ from the registered contract")
    scheduler = spec.get("scheduler", {})
    expected_scheduler = {
        "account": "dep_inin_dat",
        "partition": "gpu-l40s",
        "gpu_resource": "gpu:1",
        "gpu_model_substring": "L40S",
        "cpus_per_task": 8,
        "memory": "64G",
        "time_limit": "2-00:00:00",
        "array_concurrency": 4,
        "dependency_type": "afterok",
        "forbidden_dependency_types": ["aftercorr"],
    }
    if scheduler != expected_scheduler:
        raise ValueError("scheduler resources differ from the registered contract")
    if spec.get("python_module") != "Python/3.12.3-GCCcore-13.3.0":
        raise ValueError("cluster Python module mismatch")
    expected_runtime = {
        "python": "3.12.3",
        "jax": "0.8.1",
        "jaxlib": "0.8.1",
        "numpy": "2.5.3",
        "dm-control": "1.0.46",
        "mujoco": "3.13.0",
        "environment": {
            "JAX_ENABLE_X64": "0",
            "JAX_PLATFORM_NAME": "gpu",
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "JAX_COMPILATION_CACHE_DIR": (
                "/work2/ci72buri-dreamer_imf_neurips/jax-compilation-cache"
            ),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
            "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": (
                "xla_gpu_per_fusion_autotune_cache_dir"
            ),
            "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
            "MUJOCO_GL": "egl",
        },
    }
    if spec.get("runtime") != expected_runtime:
        raise ValueError("cluster runtime differs from the registered contract")
    registered = spec.get("matched_objective", {})
    if tuple(registered.get("stage_order", ())) != STAGES:
        raise ValueError("matched-objective stage order mismatch")
    expected = {
        "pilot": {
            "total": 606,
            "counts": {
                "dataset": 6,
                "compute_plan": 24,
                "world_model": 144,
                "rollout": 144,
                "actor": 288,
            },
        },
        "confirmatory": {
            "total": 1746,
            "counts": {
                "dataset": 60,
                "compute_plan": 6,
                "world_model": 240,
                "rollout": 720,
                "actor": 720,
            },
        },
    }
    for profile, values in expected.items():
        row = registered.get(profile, {})
        if row.get("expected_total_cells") != values["total"]:
            raise ValueError(f"{profile} total cell contract mismatch")
        if row.get("expected_stage_counts") != values["counts"]:
            raise ValueError(f"{profile} stage-count contract mismatch")
    supplementary = spec.get("supplementary", {})
    controls = supplementary.get("controls", {})
    if tuple(controls.get("stage_order", ())) != (
        "dataset",
        "compute",
        "world",
        "rollout",
        "actor",
    ):
        raise ValueError("controls stage-order contract mismatch")
    expected_controls = {
        "development": {
            "output_name": "controls_development",
            "expected_execution_units": 944,
            "expected_stage_counts": {
                "dataset": 6,
                "compute": 2,
                "world": 234,
                "rollout": 234,
                "actor": 468,
            },
            "requires": ["matched_objective_pilot_verified"],
        },
        "confirmatory": {
            "output_name": "controls_confirmatory",
            "expected_execution_units": 3666,
            "expected_stage_counts": {
                "dataset": 60,
                "compute": 6,
                "world": 720,
                "rollout": 720,
                "actor": 2160,
            },
            "requires": [
                "matched_objective_pilot_verified",
                "matched_objective_confirmatory_verified",
                "controls_development_verified_and_selection_frozen",
            ],
        },
    }
    for profile, expected_control in expected_controls.items():
        if controls.get(profile) != expected_control:
            raise ValueError(f"controls {profile} launch-plan contract mismatch")
    if controls.get("stage_array_concurrency") != {
        "dataset": 4,
        "compute": 1,
        "world": 4,
        "rollout": 4,
        "actor": 4,
    }:
        raise ValueError("controls stage concurrency contract mismatch")
    pixel = supplementary.get("pixel", {})
    if (
        pixel.get("profile") != "hard"
        or pixel.get("output_name") != "pixel_hard"
        or pixel.get("expected_artifacts") != 30
        or pixel.get("requires") != ["matched_objective_pilot_verified"]
        or len(pixel.get("tasks", ())) * len(pixel.get("seeds", ()))
        * len(pixel.get("tracks", ()))
        != 30
    ):
        raise ValueError("pixel launch-plan contract mismatch")
    if spec.get("workspace_allocator") != {
        "id": "dreamer_imf_neurips",
        "comment": "Trajectory iMF NeurIPS matched-compute benchmark",
        "timezone": "Europe/Berlin",
        "minimum_remaining_hours": 168,
        "minimum_available_extensions": 0,
        "quota_filesystem": "/work2",
        "quota_user": "ci72buri",
        "minimum_block_limit_kib": 5368709120,
        "minimum_inode_limit": 10000000,
    }:
        raise ValueError("workspace allocator/quota contract mismatch")
    if spec.get("minimum_free_workspace_bytes") != 107374182400:
        raise ValueError("minimum free workspace capacity contract mismatch")


def expected_stage_counts(spec: Mapping[str, Any], profile: str) -> dict[str, int]:
    if profile not in ("pilot", "confirmatory"):
        raise ValueError(f"unsupported matched-objective profile: {profile}")
    return dict(
        spec["matched_objective"][profile]["expected_stage_counts"]
    )


def validate_main_matrix_shape(
    matrix: Mapping[str, Any], profile: str, spec: Mapping[str, Any]
) -> dict[str, int]:
    """Check the preregistered cell totals without trusting a map file."""

    if matrix.get("profile") != profile:
        raise ValueError("frozen matrix profile mismatch")
    cells = matrix.get("cells")
    if not isinstance(cells, list):
        raise ValueError("frozen matrix cells are malformed")
    identifiers = [cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)]
    if len(identifiers) != len(cells) or len(set(identifiers)) != len(identifiers):
        raise ValueError("frozen matrix cell identifiers are missing or duplicated")
    counts = {stage: 0 for stage in STAGES}
    by_id = {cell["cell_id"]: cell for cell in cells}
    rank = {stage: index for index, stage in enumerate(STAGES)}
    for cell in cells:
        stage = cell.get("stage")
        if stage not in counts:
            raise ValueError("frozen matrix contains an unknown stage")
        counts[stage] += 1
        dependencies = cell.get("dependencies")
        if not isinstance(dependencies, list):
            raise ValueError("frozen matrix dependencies are malformed")
        for dependency in dependencies:
            if dependency not in by_id:
                raise ValueError("frozen matrix dependency is absent")
            dependency_stage = by_id[dependency].get("stage")
            if dependency_stage not in rank:
                raise ValueError("frozen matrix dependency has an unknown stage")
            if rank[dependency_stage] >= rank[stage]:
                raise ValueError("frozen matrix dependency does not precede its cell")
    expected = expected_stage_counts(spec, profile)
    if counts != expected:
        raise ValueError(f"{profile} matrix stage counts differ: {counts} != {expected}")
    expected_total = spec["matched_objective"][profile]["expected_total_cells"]
    if len(cells) != expected_total or sum(counts.values()) != expected_total:
        raise ValueError(f"{profile} matrix total must be {expected_total}")
    return counts


def _without_digest(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def build_stage_map(
    matrix: Mapping[str, Any], profile: str, stage: str
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    cells = sorted(
        (cell for cell in matrix["cells"] if cell["stage"] == stage),
        key=lambda cell: cell["cell_id"],
    )
    entries = [
        {
            "array_index": index,
            "cell_id": cell["cell_id"],
            "identity_sha256": cell["identity_sha256"],
            "dependencies": list(cell["dependencies"]),
        }
        for index, cell in enumerate(cells)
    ]
    payload = {
        "schema_version": MAP_SCHEMA,
        "status": "frozen_before_stage_execution",
        "profile": profile,
        "stage": stage,
        "matrix_sha256": matrix["matrix_sha256"],
        "count": len(entries),
        "entries": entries,
    }
    payload["map_sha256"] = object_sha256(payload)
    return payload


def validate_stage_map(
    payload: Mapping[str, Any], matrix: Mapping[str, Any], profile: str, stage: str
) -> None:
    required = {
        "schema_version",
        "status",
        "profile",
        "stage",
        "matrix_sha256",
        "count",
        "entries",
        "map_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != MAP_SCHEMA:
        raise ValueError("stage map schema mismatch")
    if payload.get("map_sha256") != object_sha256(
        _without_digest(payload, "map_sha256")
    ):
        raise ValueError("stage map digest mismatch")
    expected = build_stage_map(matrix, profile, stage)
    if dict(payload) != expected:
        raise ValueError("stage map differs from the frozen matrix")


def maps_directory(output_root: str | Path) -> Path:
    return Path(output_root) / "cluster_maps"


def map_path(output_root: str | Path, stage: str) -> Path:
    return maps_directory(output_root) / f"{stage}.json"


def freeze_stage_maps(
    output_root: str | Path,
    matrix: Mapping[str, Any],
    profile: str,
    spec: Mapping[str, Any],
) -> dict[str, str]:
    validate_main_matrix_shape(matrix, profile, spec)
    digests: dict[str, str] = {}
    for stage in STAGES:
        payload = build_stage_map(matrix, profile, stage)
        write_json_once(map_path(output_root, stage), payload)
        digests[stage] = payload["map_sha256"]
    return digests


def _git(checkout: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return process.stdout.strip()


def require_clean_checkout(checkout: str | Path) -> str:
    root = Path(checkout).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("source checkout HEAD is not a full commit digest")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("cluster execution requires a clean committed checkout")
    return commit


def profile_output(
    spec: Mapping[str, Any], profile: str, *, cluster_root: str | Path | None = None
) -> Path:
    if profile not in ("pilot", "confirmatory"):
        raise ValueError("profile must be pilot or confirmatory")
    base = (
        Path(cluster_root) / "results"
        if cluster_root is not None
        else Path(spec["results_root"])
    )
    return require_safe_cluster_path(
        spec,
        base / spec["matched_objective"][profile]["output_name"],
        cluster_root=cluster_root,
    )


def state_root(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    root = (
        Path(cluster_root) / "cluster_state"
        if cluster_root is not None
        else Path(spec["state_root"])
    )
    return require_safe_cluster_path(spec, root, cluster_root=cluster_root)


def source_root(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    root = (
        Path(cluster_root) / "source"
        if cluster_root is not None
        else Path(spec["source_root"])
    )
    return require_safe_cluster_path(spec, root, cluster_root=cluster_root)


def run_contract_path(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return state_root(spec, cluster_root=cluster_root) / "run_contract.json"


def preflight_path(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return state_root(spec, cluster_root=cluster_root) / "preflight.json"


def ledger_path(
    spec: Mapping[str, Any], *, cluster_root: str | Path | None = None
) -> Path:
    return state_root(spec, cluster_root=cluster_root) / "provenance_job_ledger.jsonl"


def initialize_cluster_root(
    *,
    spec: Mapping[str, Any],
    checkout: str | Path,
    lock_path: str | Path = LOCK_PATH,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    commit = require_clean_checkout(checkout_path)
    lock = Path(lock_path).resolve()
    if not lock.is_file():
        raise FileNotFoundError("CUDA dependency lock is missing")
    payload = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "status": "initialized_no_jobs_submitted",
        "source_commit": commit,
        "source_root": str(checkout_path),
        "repository_url": spec["repository_url"],
        "cluster_spec_sha256": file_sha256(SPEC_PATH),
        "dependency_lock_sha256": file_sha256(lock),
        "dependency_lock_name": lock.name,
    }
    payload["run_contract_sha256"] = object_sha256(payload)
    write_json_once(run_contract_path(spec, cluster_root=cluster_root), payload)
    append_ledger(
        ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "cluster_root_initialized",
            "source_commit": commit,
            "run_contract_sha256": payload["run_contract_sha256"],
        },
    )
    return payload


def validate_run_contract(
    payload: Mapping[str, Any], spec: Mapping[str, Any], checkout: str | Path
) -> None:
    if payload.get("schema_version") != RUN_CONTRACT_SCHEMA:
        raise ValueError("cluster run contract schema mismatch")
    if payload.get("run_contract_sha256") != object_sha256(
        _without_digest(payload, "run_contract_sha256")
    ):
        raise ValueError("cluster run contract digest mismatch")
    if payload.get("cluster_spec_sha256") != file_sha256(SPEC_PATH):
        raise ValueError("cluster specification changed after initialization")
    if payload.get("dependency_lock_sha256") != file_sha256(LOCK_PATH):
        raise ValueError("dependency lock changed after initialization")
    if payload.get("source_commit") != require_clean_checkout(checkout):
        raise ValueError("source checkout changed after initialization")


def validate_preflight_record(
    payload: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != PREFLIGHT_SCHEMA:
        raise ValueError("GPU preflight schema mismatch")
    if payload.get("preflight_sha256") != object_sha256(
        _without_digest(payload, "preflight_sha256")
    ):
        raise ValueError("GPU preflight digest mismatch")
    if payload.get("status") != "verified_gpu_worker":
        raise ValueError("GPU preflight is not complete")
    if payload.get("cluster_spec_sha256") != contract["cluster_spec_sha256"]:
        raise ValueError("GPU preflight cluster specification mismatch")
    if payload.get("dependency_lock_sha256") != contract["dependency_lock_sha256"]:
        raise ValueError("GPU preflight dependency lock mismatch")
    if payload.get("source_commit") != contract["source_commit"]:
        raise ValueError("GPU preflight source commit mismatch")
    expected_versions = {
        name: spec["runtime"][name]
        for name in ("python", "jax", "jaxlib", "numpy", "dm-control", "mujoco")
    }
    if payload.get("versions") != expected_versions:
        raise ValueError("GPU preflight package versions mismatch")
    environment = payload.get("environment")
    if environment != spec["runtime"]["environment"]:
        raise ValueError("GPU preflight environment mismatch")
    jax_runtime = payload.get("jax_runtime", {})
    if (
        jax_runtime.get("backend") != "gpu"
        or jax_runtime.get("visible_device_count") != 1
        or jax_runtime.get("jax_enable_x64") is not False
        or len(jax_runtime.get("device_kinds", ())) != 1
        or spec["scheduler"]["gpu_model_substring"]
        not in jax_runtime["device_kinds"][0]
        or jax_runtime.get("tiny_device_check") != "passed"
    ):
        raise ValueError("GPU preflight JAX device contract mismatch")
    slurm = payload.get("slurm", {})
    if (
        not isinstance(slurm.get("job_id"), str)
        or not slurm["job_id"].isascii()
        or not slurm["job_id"].isdigit()
        or slurm.get("job_account") != spec["scheduler"]["account"]
        or slurm.get("job_partition") != spec["scheduler"]["partition"]
        or slurm.get("cpus_per_task") != str(spec["scheduler"]["cpus_per_task"])
        or not isinstance(slurm.get("cuda_visible_devices"), str)
        or slurm["cuda_visible_devices"] != slurm["cuda_visible_devices"].strip()
        or "," in slurm["cuda_visible_devices"]
    ):
        raise ValueError("GPU preflight Slurm allocation mismatch")
    render = payload.get("dmc_render", {})
    if (
        render.get("task") != "dmc_cartpole_swingup"
        or render.get("shape") != [16, 16, 3]
        or render.get("dtype") != "uint8"
        or render.get("finite") is not True
        or not isinstance(render.get("sha256"), str)
    ):
        raise ValueError("GPU preflight DMC EGL render mismatch")
    storage = payload.get("workspace_storage", {})
    if (
        storage.get("root") != spec["workspace_root"]
        or storage.get("measurement") != "shutil.disk_usage"
        or not isinstance(storage.get("free_bytes"), int)
        or storage["free_bytes"] < spec["minimum_free_workspace_bytes"]
        or not isinstance(storage.get("total_bytes"), int)
        or storage["total_bytes"] < storage["free_bytes"]
        or not isinstance(storage.get("owner_uid"), int)
        or storage.get("owner_uid") != storage.get("effective_uid")
        or storage.get("owned_by_effective_user") is not True
        or storage.get("read_write_search_access") is not True
        or storage.get("claim_scope") != "filesystem_capacity_not_user_quota"
    ):
        raise ValueError("GPU preflight workspace capacity is insufficient")
    allocator_contract = spec["workspace_allocator"]
    allocator = payload.get("workspace_allocator", {})
    quota = payload.get("workspace_quota", {})
    allocator_required = {
        "interface",
        "id",
        "workspace_directory",
        "remaining_days",
        "remaining_hours",
        "comment",
        "creation_time",
        "expiration_date",
        "filesystem_name",
        "available_extensions",
        "timezone",
        "creation_utc",
        "expiration_utc",
        "observed_at_utc",
        "remaining_seconds_at_observation",
        "ws_find_stdout_sha256",
        "ws_list_stdout_sha256",
    }
    quota_required = {
        "interface",
        "user",
        "kbytes",
        "bquota",
        "blimit",
        "files",
        "iquota",
        "ilimit",
        "filesystem",
        "bgrace",
        "igrace",
        "remaining_quota_kib",
        "remaining_inodes",
        "stdout_sha256",
    }
    try:
        observed = dt.datetime.fromisoformat(allocator["observed_at_utc"])
        expiration = dt.datetime.fromisoformat(allocator["expiration_utc"])
        creation = dt.datetime.fromisoformat(allocator["creation_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GPU preflight workspace lifetime timestamps are invalid") from error
    if any(value.tzinfo is None for value in (observed, expiration, creation)):
        raise ValueError("GPU preflight workspace timestamps lack time zones")
    remaining_seconds = int((expiration - observed).total_seconds())
    reported_seconds = 3600 * (
        24 * int(allocator.get("remaining_days", -1))
        + int(allocator.get("remaining_hours", -1))
    )
    if (
        set(allocator) != allocator_required
        or allocator.get("interface") != "ws_find_plus_ws_list"
        or allocator.get("id") != allocator_contract["id"]
        or allocator.get("workspace_directory") != spec["workspace_root"]
        or allocator.get("comment") != allocator_contract["comment"]
        or allocator.get("timezone") != allocator_contract["timezone"]
        or creation >= expiration
        or allocator.get("remaining_seconds_at_observation") != remaining_seconds
        or reported_seconds > remaining_seconds + 120
        or remaining_seconds >= reported_seconds + 3720
        or remaining_seconds < 3600 * allocator_contract["minimum_remaining_hours"]
        or expiration <= dt.datetime.now(dt.timezone.utc)
        or not isinstance(allocator.get("available_extensions"), int)
        or allocator["available_extensions"]
        < allocator_contract["minimum_available_extensions"]
        or any(
            not isinstance(allocator.get(name), str)
            or len(allocator[name]) != 64
            or any(character not in "0123456789abcdef" for character in allocator[name])
            for name in ("ws_find_stdout_sha256", "ws_list_stdout_sha256")
        )
    ):
        raise ValueError("GPU preflight authoritative workspace evidence is invalid")
    expected_remaining_kib = quota.get("blimit", -1) - quota.get("kbytes", 0)
    expected_remaining_inodes = quota.get("ilimit", -1) - quota.get("files", 0)
    if (
        set(quota) != quota_required
        or quota.get("interface") != "lfs_quota_user"
        or quota.get("user") != allocator_contract["quota_user"]
        or quota.get("filesystem") != allocator_contract["quota_filesystem"]
        or not all(
            isinstance(quota.get(name), int) and quota[name] >= 0
            for name in ("kbytes", "bquota", "blimit", "files", "iquota", "ilimit")
        )
        or quota.get("remaining_quota_kib") != expected_remaining_kib
        or quota.get("remaining_inodes") != expected_remaining_inodes
        or quota.get("blimit", 0) < allocator_contract["minimum_block_limit_kib"]
        or quota.get("ilimit", 0) < allocator_contract["minimum_inode_limit"]
        or expected_remaining_kib * 1024 < spec["minimum_free_workspace_bytes"]
        or expected_remaining_inodes <= 0
        or not isinstance(quota.get("stdout_sha256"), str)
        or len(quota["stdout_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in quota["stdout_sha256"])
    ):
        raise ValueError("GPU preflight authoritative quota evidence is invalid")


@contextmanager
def _locked_ledger(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("cluster ledger must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("cluster ledger must be a regular file")
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_ledger_handle(handle: Any) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            raise ValueError(f"blank ledger row at line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("ledger row is not a JSON object")
        rows.append(value)
    return rows


def validate_ledger_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    previous = "0" * 64
    for sequence, row in enumerate(rows):
        if row.get("schema_version") != LEDGER_SCHEMA:
            raise ValueError("ledger schema mismatch")
        if row.get("sequence") != sequence or row.get("previous_event_sha256") != previous:
            raise ValueError("ledger chain ordering mismatch")
        digest = row.get("event_sha256")
        if digest != object_sha256(_without_digest(row, "event_sha256")):
            raise ValueError("ledger event digest mismatch")
        previous = str(digest)


def append_ledger(path: str | Path, event: Mapping[str, Any]) -> dict[str, Any]:
    if "event" not in event or not isinstance(event["event"], str):
        raise ValueError("ledger event requires an event name")
    destination = Path(path)
    with _locked_ledger(destination) as handle:
        rows = _read_ledger_handle(handle)
        validate_ledger_rows(rows)
        previous = rows[-1]["event_sha256"] if rows else "0" * 64
        record = {
            "schema_version": LEDGER_SCHEMA,
            "sequence": len(rows),
            "previous_event_sha256": previous,
            "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **dict(event),
        }
        if set(event) & {
            "schema_version",
            "sequence",
            "previous_event_sha256",
            "recorded_at_utc",
            "event_sha256",
        }:
            raise ValueError("ledger event attempts to replace chain metadata")
        record["event_sha256"] = object_sha256(record)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return record


def append_ledger_once(path: str | Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically append one exact idempotent event, or return its prior row."""

    if "event" not in event or not isinstance(event["event"], str):
        raise ValueError("ledger event requires an event name")
    metadata = {
        "schema_version",
        "sequence",
        "previous_event_sha256",
        "recorded_at_utc",
        "event_sha256",
    }
    if set(event) & metadata:
        raise ValueError("ledger event attempts to replace chain metadata")
    expected = dict(event)
    destination = Path(path)
    with _locked_ledger(destination) as handle:
        rows = _read_ledger_handle(handle)
        validate_ledger_rows(rows)
        matches = [
            row
            for row in rows
            if {key: value for key, value in row.items() if key not in metadata}
            == expected
        ]
        if len(matches) > 1:
            raise ValueError("idempotent cluster event is duplicated in the ledger")
        if matches:
            return matches[0]
        previous = rows[-1]["event_sha256"] if rows else "0" * 64
        record = {
            "schema_version": LEDGER_SCHEMA,
            "sequence": len(rows),
            "previous_event_sha256": previous,
            "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            **expected,
        }
        record["event_sha256"] = object_sha256(record)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return record


def verify_ledger(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("cluster ledger must not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("cluster ledger must be a regular file")
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        rows = _read_ledger_handle(handle)
    validate_ledger_rows(rows)
    return rows


def _add_source_paths(checkout: Path) -> None:
    for path in (checkout / "dreamer_imf_comparison", checkout / "imf_dreamer_jax" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_validated_main_run(
    output_root: str | Path,
    checkout: str | Path,
    profile: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkout_path = Path(checkout).resolve()
    _add_source_paths(checkout_path)
    from dreamer_imf_compare.artifacts import read_json as benchmark_read_json
    from dreamer_imf_compare.matched_objective_benchmark import (
        HPO_MATRIX_SCHEMA,
        validate_matrix,
        validate_pilot_hpo_matrix,
        validate_source_manifest,
    )

    root = Path(output_root)
    protocol = benchmark_read_json(root / "frozen_protocol.json")
    source = benchmark_read_json(root / "source_manifest.json")
    matrix = benchmark_read_json(root / "matrix.json")
    validate_source_manifest(source, checkout_path)
    if matrix.get("schema_version") == HPO_MATRIX_SCHEMA:
        if profile != "pilot":
            raise ValueError("HPO matrix is only valid for the pilot")
        validate_pilot_hpo_matrix(matrix, protocol, source)
    else:
        validate_matrix(matrix, protocol, source)
    validate_main_matrix_shape(matrix, profile, spec)
    return protocol, source, matrix


def freeze_profile(
    profile: str,
    *,
    spec: Mapping[str, Any],
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    checkout_path = Path(checkout).resolve()
    contract = read_json(run_contract_path(spec, cluster_root=cluster_root))
    validate_run_contract(contract, spec, checkout_path)
    preflight = preflight_path(spec, cluster_root=cluster_root)
    if not preflight.is_file():
        raise RuntimeError("GPU preflight must complete before freezing a matrix")
    validate_preflight_record(read_json(preflight), spec=spec, contract=contract)
    output = profile_output(spec, profile, cluster_root=cluster_root)
    executable = str(python or Path(spec["environment_root"]) / "bin" / "python")
    runner = checkout_path / "dreamer_imf_comparison" / "scripts" / "run_matched_objective_benchmark.py"
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
            pilot = profile_output(spec, "pilot", cluster_root=cluster_root)
            require_profile_ledger_authenticated(
                spec, "pilot", cluster_root=cluster_root
            )
            command.extend(("--pilot-output", str(pilot)))
        subprocess.run(command, check=True)
    _, source, matrix = load_validated_main_run(output, checkout_path, profile, spec)
    if source["git"]["commit"] != contract["source_commit"]:
        raise ValueError("frozen source commit differs from the cluster run contract")
    map_digests = freeze_stage_maps(output, matrix, profile, spec)
    freeze_record = {
        "schema_version": "trajectory-imf-cluster-freeze-v1",
        "status": "frozen_before_execution",
        "profile": profile,
        "source_commit": contract["source_commit"],
        "matrix_sha256": matrix["matrix_sha256"],
        "cell_count": len(matrix["cells"]),
        "map_sha256": map_digests,
    }
    freeze_record["freeze_sha256"] = object_sha256(freeze_record)
    write_json_once(output / "cluster_freeze.json", freeze_record)
    append_ledger(
        ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "profile_frozen",
            "profile": profile,
            "matrix_sha256": matrix["matrix_sha256"],
            "freeze_sha256": freeze_record["freeze_sha256"],
        },
    )
    return freeze_record


def stage_marker_path(output_root: str | Path, stage: str) -> Path:
    return Path(output_root) / "cluster_stage_verification" / f"{stage}.json"


def previous_stage(stage: str) -> str | None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    index = STAGES.index(stage)
    return None if index == 0 else STAGES[index - 1]


def validate_stage_marker(
    marker: Mapping[str, Any],
    *,
    profile: str,
    stage: str,
    matrix: Mapping[str, Any],
    stage_map: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "status",
        "profile",
        "stage",
        "matrix_sha256",
        "map_sha256",
        "verified_cell_count",
        "result_files",
        "slurm_job_id",
        "stage_verification_sha256",
    }
    if set(marker) != required or marker.get("schema_version") != STAGE_MARKER_SCHEMA:
        raise ValueError("stage verification marker schema mismatch")
    if marker.get("stage_verification_sha256") != object_sha256(
        _without_digest(marker, "stage_verification_sha256")
    ):
        raise ValueError("stage verification marker digest mismatch")
    expected = {
        "status": "verified_complete",
        "profile": profile,
        "stage": stage,
        "matrix_sha256": matrix["matrix_sha256"],
        "map_sha256": stage_map["map_sha256"],
        "verified_cell_count": stage_map["count"],
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("stage verification marker identity mismatch")
    result_files = marker.get("result_files")
    if not isinstance(result_files, list) or len(result_files) != stage_map["count"]:
        raise ValueError("stage verification result manifest is incomplete")
    expected_ids = [entry["cell_id"] for entry in stage_map["entries"]]
    if [entry.get("cell_id") for entry in result_files] != expected_ids:
        raise ValueError("stage verification cell order differs from its map")
    if not isinstance(marker.get("slurm_job_id"), str) or not marker[
        "slurm_job_id"
    ].isascii() or not marker["slurm_job_id"].isdigit():
        raise ValueError("stage verification marker lacks a Slurm job id")
    for entry in result_files:
        if not isinstance(entry, Mapping) or set(entry) != {
            "cell_id",
            "path",
            "sha256",
        }:
            raise ValueError("stage verification result entry schema mismatch")
        relative = entry["path"]
        digest = entry["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or any(part in ("", ".", "..") for part in Path(relative).parts)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("stage verification result entry is not canonical")


def validate_stage_result_files(
    output_root: str | Path, marker: Mapping[str, Any]
) -> None:
    """Recheck every immutable stage result without trusting manifest paths."""

    output = Path(output_root).resolve()
    for entry in marker["result_files"]:
        path = (output / entry["path"]).resolve()
        if output not in path.parents:
            raise ValueError("stage verification result path escapes its output root")
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise ValueError("previously verified stage result changed")


def require_previous_stage_verified(
    output_root: str | Path,
    stage: str,
    profile: str,
    matrix: Mapping[str, Any],
) -> None:
    dependency = previous_stage(stage)
    if dependency is None:
        return
    dependency_map = read_json(map_path(output_root, dependency))
    validate_stage_map(dependency_map, matrix, profile, dependency)
    marker_path = stage_marker_path(output_root, dependency)
    if not marker_path.is_file():
        raise RuntimeError(
            f"stage {stage} is blocked until the entire {dependency} stage verifies"
        )
    marker = read_json(marker_path)
    validate_stage_marker(
        marker,
        profile=profile,
        stage=dependency,
        matrix=matrix,
        stage_map=dependency_map,
    )
    validate_stage_result_files(output_root, marker)


def cell_for_array_index(
    stage_map: Mapping[str, Any], matrix: Mapping[str, Any], array_index: int
) -> Mapping[str, Any]:
    if isinstance(array_index, bool) or not isinstance(array_index, int):
        raise ValueError("array index must be an integer")
    entries = stage_map["entries"]
    if not 0 <= array_index < len(entries):
        raise IndexError("array index is outside the frozen stage map")
    entry = entries[array_index]
    if entry["array_index"] != array_index:
        raise ValueError("stage map index is not canonical")
    matches = [cell for cell in matrix["cells"] if cell["cell_id"] == entry["cell_id"]]
    if len(matches) != 1:
        raise ValueError("stage map cell is absent from or duplicated in the matrix")
    return matches[0]


def run_array_cell(
    profile: str,
    stage: str,
    array_index: int,
    *,
    spec: Mapping[str, Any],
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    python: str | Path | None = None,
    retry_map_path: str | Path | None = None,
) -> str:
    output = profile_output(spec, profile, cluster_root=cluster_root)
    _, _, matrix = load_validated_main_run(output, checkout, profile, spec)
    stage_map = read_json(map_path(output, stage))
    validate_stage_map(stage_map, matrix, profile, stage)
    require_previous_stage_verified(output, stage, profile, matrix)
    cell = cell_for_array_index(stage_map, matrix, array_index)
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if (
        not isinstance(array_job_id, str)
        or not array_job_id.isascii()
        or not array_job_id.isdigit()
        or not isinstance(task_id, str)
        or not task_id.isascii()
        or not task_id.isdigit()
        or int(task_id) != array_index
    ):
        raise RuntimeError("main worker requires matching numeric Slurm array identity")
    retry_digest = None
    if retry_map_path is not None:
        retry_path = require_safe_cluster_path(
            spec,
            retry_map_path,
            cluster_root=cluster_root,
        )
        if retry_path.is_symlink() or not retry_path.is_file():
            raise ValueError("main retry map must be a regular non-symlink file")
        retry = read_json(retry_path)
        validate_retry_map(
            retry,
            profile=profile,
            stage=stage,
            matrix=matrix,
            stage_map=stage_map,
        )
        exported_retry_digest = os.environ.get("NEURIPS_RETRY_MAP_SHA256")
        if exported_retry_digest != retry["retry_map_sha256"]:
            raise ValueError("main retry map digest differs from its Slurm export")
        require_retry_map_registered(
            retry_path,
            retry,
            output_root=output,
            ledger=ledger_path(spec, cluster_root=cluster_root),
            spec=spec,
            cluster_root=cluster_root,
        )
        if array_index not in retry["indices"]:
            raise ValueError("main array index is absent from its authenticated retry map")
        position = retry["indices"].index(array_index)
        current_state = main_cell_state_sha256(
            spec,
            output,
            cell,
            cluster_root=cluster_root,
        )
        if retry["selected_state_sha256"][position] != current_state:
            raise ValueError("main retry map became stale before worker execution")
        retry_action = retry["actions"][position]
        observed_action = main_cell_retry_action(
            spec, output, cell, cluster_root=cluster_root
        )
        if retry_action != observed_action:
            raise ValueError("main retry action differs from current artifact state")
        artifact = main_cell_artifact_path(
            spec, output, cell, cluster_root=cluster_root
        )
        quarantine = (
            state_root(spec, cluster_root=cluster_root)
            / "main_quarantine"
            / profile
            / stage
            / retry["audit_id"]
            / cell["cell_id"]
        )
        if retry_action == "quarantine_invalid" and (
            artifact.exists() or artifact.is_symlink()
        ):
            quarantine_cluster_node(
                spec,
                artifact,
                quarantine,
                cluster_root=cluster_root,
            )
            append_ledger(
                ledger_path(spec, cluster_root=cluster_root),
                {
                    "event": "main_cell_quarantined_for_retry",
                    "profile": profile,
                    "stage": stage,
                    "array_index": array_index,
                    "cell_id": cell["cell_id"],
                    "retry_map_sha256": retry["retry_map_sha256"],
                    "retry_action": retry_action,
                    "slurm_array_job_id": array_job_id,
                    "slurm_task_id": task_id,
                },
            )
        retry_digest = retry["retry_map_sha256"]
    executable = str(python or Path(spec["environment_root"]) / "bin" / "python")
    runner = Path(checkout) / "dreamer_imf_comparison" / "scripts" / "run_matched_objective_benchmark.py"
    command = [
        executable,
        str(runner),
        "run",
        "--profile",
        profile,
        "--output",
        str(output),
        "--workspace",
        str(Path(checkout).resolve()),
        "--cell-id",
        cell["cell_id"],
    ]
    append_ledger(
        ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "cell_started",
            "profile": profile,
            "stage": stage,
            "array_index": array_index,
            "cell_id": cell["cell_id"],
            "retry_map_sha256": retry_digest,
            "slurm_job_id": array_job_id,
            "slurm_task_id": task_id,
        },
    )
    try:
        subprocess.run(command, check=True)
    except BaseException as error:
        append_ledger(
            ledger_path(spec, cluster_root=cluster_root),
            {
                "event": "cell_failed",
                "profile": profile,
                "stage": stage,
                "array_index": array_index,
                "cell_id": cell["cell_id"],
                "retry_map_sha256": retry_digest,
                "slurm_job_id": array_job_id,
                "slurm_task_id": task_id,
                "error_type": type(error).__name__,
            },
        )
        raise
    append_ledger(
        ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "cell_completed",
            "profile": profile,
            "stage": stage,
            "array_index": array_index,
            "cell_id": cell["cell_id"],
            "retry_map_sha256": retry_digest,
            "slurm_job_id": array_job_id,
            "slurm_task_id": task_id,
        },
    )
    return cell["cell_id"]


def _validated_cell(
    output: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cell: Mapping[str, Any],
    checkout: Path,
) -> Path:
    _add_source_paths(checkout)
    from dreamer_imf_compare.matched_objective_benchmark import (
        _completed_result_for_cell,
    )

    _, path = _completed_result_for_cell(output, matrix, protocol, cell)
    return path


def incomplete_indices(
    profile: str,
    stage: str,
    *,
    spec: Mapping[str, Any],
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    strong_validation: bool = True,
) -> list[int]:
    output = profile_output(spec, profile, cluster_root=cluster_root)
    protocol, _, matrix = load_validated_main_run(output, checkout, profile, spec)
    stage_map = read_json(map_path(output, stage))
    validate_stage_map(stage_map, matrix, profile, stage)
    incomplete: list[int] = []
    for entry in stage_map["entries"]:
        cell = cell_for_array_index(stage_map, matrix, entry["array_index"])
        result = output / stage / cell["cell_id"] / "result.json"
        if not result.is_file():
            incomplete.append(entry["array_index"])
            continue
        if strong_validation:
            try:
                _validated_cell(
                    output, matrix, protocol, cell, Path(checkout).resolve()
                )
            except Exception:
                incomplete.append(entry["array_index"])
    return incomplete


def _safe_retry_audit_id(audit_id: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", audit_id) is None:
        raise ValueError("retry audit id must be a filesystem-safe token")
    return audit_id


def retry_map_path(output_root: str | Path, stage: str, audit_id: str) -> Path:
    _safe_retry_audit_id(audit_id)
    return Path(output_root) / "cluster_retry_maps" / stage / f"retry-{audit_id}.json"


def main_cell_artifact_path(
    spec: Mapping[str, Any],
    output: str | Path,
    cell: Mapping[str, Any],
    *,
    cluster_root: str | Path | None = None,
) -> Path:
    stage = cell.get("stage")
    cell_id = cell.get("cell_id")
    if stage not in STAGES or not isinstance(cell_id, str) or re.fullmatch(
        rf"{re.escape(str(stage))}-[0-9a-f]{{24}}", cell_id
    ) is None:
        raise ValueError("main cell artifact identity is not canonical")
    path = Path(output) / str(stage) / cell_id
    return require_safe_cluster_path(
        spec,
        path,
        cluster_root=cluster_root,
        allow_leaf_symlink=True,
    )


def main_cell_state_sha256(
    spec: Mapping[str, Any],
    output: str | Path,
    cell: Mapping[str, Any],
    *,
    cluster_root: str | Path | None = None,
) -> str:
    return tree_state_sha256(
        spec,
        main_cell_artifact_path(
            spec, output, cell, cluster_root=cluster_root
        ),
        cluster_root=cluster_root,
    )


def main_cell_retry_action(
    spec: Mapping[str, Any],
    output: str | Path,
    cell: Mapping[str, Any],
    *,
    cluster_root: str | Path | None = None,
) -> str:
    artifact = main_cell_artifact_path(
        spec, output, cell, cluster_root=cluster_root
    )
    snapshot = tree_state(spec, artifact, cluster_root=cluster_root)
    if not snapshot["exists"]:
        return "run_missing"
    if any(row["type"] in {"symlink", "other"} for row in snapshot["entries"]):
        return "quarantine_invalid"
    result = artifact / "result.json"
    if result.is_file() or result.is_symlink():
        return "quarantine_invalid"
    return "resume_partial"


def quarantine_cluster_node(
    spec: Mapping[str, Any],
    source: str | Path,
    destination: str | Path,
    *,
    cluster_root: str | Path | None = None,
) -> None:
    source_path = require_safe_cluster_path(
        spec,
        source,
        cluster_root=cluster_root,
        allow_leaf_symlink=True,
    )
    destination_path = require_safe_cluster_path(
        spec,
        destination,
        cluster_root=cluster_root,
    )
    if not source_path.exists() and not source_path.is_symlink():
        raise FileNotFoundError("quarantine source is absent")
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(f"quarantine destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    require_safe_cluster_path(
        spec,
        destination_path.parent,
        cluster_root=cluster_root,
    )
    os.replace(source_path, destination_path)


def validate_retry_map(
    payload: Mapping[str, Any],
    *,
    profile: str,
    stage: str,
    matrix: Mapping[str, Any],
    stage_map: Mapping[str, Any],
    current_states: Sequence[str] | None = None,
    current_actions: Sequence[str] | None = None,
) -> None:
    required = {
        "schema_version",
        "status",
        "profile",
        "stage",
        "matrix_sha256",
        "stage_map_sha256",
        "audit_id",
        "indices",
        "cell_ids",
        "selected_state_sha256",
        "actions",
        "slurm_job_id",
        "retry_map_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != RETRY_MAP_SCHEMA:
        raise ValueError("retry map schema mismatch")
    if payload.get("retry_map_sha256") != object_sha256(
        _without_digest(payload, "retry_map_sha256")
    ):
        raise ValueError("retry map digest mismatch")
    expected = {
        "status": "strongly_audited_missing_or_invalid_only",
        "profile": profile,
        "stage": stage,
        "matrix_sha256": matrix["matrix_sha256"],
        "stage_map_sha256": stage_map["map_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("retry map identity mismatch")
    indices = payload.get("indices")
    cell_ids = payload.get("cell_ids")
    states = payload.get("selected_state_sha256")
    actions = payload.get("actions")
    if (
        not isinstance(indices, list)
        or not indices
        or len(set(indices)) != len(indices)
        or indices != sorted(indices)
        or not all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
        or not isinstance(cell_ids, list)
        or len(cell_ids) != len(indices)
        or not isinstance(states, list)
        or len(states) != len(indices)
        or not all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            for value in states
        )
        or not isinstance(actions, list)
        or len(actions) != len(indices)
        or any(
            action not in {"run_missing", "resume_partial", "quarantine_invalid"}
            for action in actions
        )
        or not isinstance(payload.get("slurm_job_id"), str)
        or not payload["slurm_job_id"].isascii()
        or not payload["slurm_job_id"].isdigit()
    ):
        raise ValueError("retry map index list is not canonical")
    _safe_retry_audit_id(str(payload.get("audit_id")))
    expected_ids = [
        cell_for_array_index(stage_map, matrix, index)["cell_id"]
        for index in indices
    ]
    if cell_ids != expected_ids:
        raise ValueError("retry map cell ids differ from its indices")
    if current_states is not None and list(current_states) != states:
        raise ValueError("main retry map is stale; selected artifact bytes changed")
    if current_actions is not None and list(current_actions) != actions:
        raise ValueError("main retry actions differ from the audited artifact state")


def audit_retry_map(
    profile: str,
    stage: str,
    audit_id: str,
    *,
    spec: Mapping[str, Any],
    checkout: str | Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    output = profile_output(spec, profile, cluster_root=cluster_root)
    _, _, matrix = load_validated_main_run(output, checkout, profile, spec)
    stage_map = read_json(map_path(output, stage))
    validate_stage_map(stage_map, matrix, profile, stage)
    indices = incomplete_indices(
        profile,
        stage,
        spec=spec,
        checkout=checkout,
        cluster_root=cluster_root,
        strong_validation=True,
    )
    if not indices:
        raise ValueError("strong retry audit found no missing or invalid cells")
    cells = [cell_for_array_index(stage_map, matrix, index) for index in indices]
    states = [
        main_cell_state_sha256(
            spec,
            output,
            cell,
            cluster_root=cluster_root,
        )
        for cell in cells
    ]
    actions = [
        main_cell_retry_action(
            spec,
            output,
            cell,
            cluster_root=cluster_root,
        )
        for cell in cells
    ]
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if (
        not isinstance(slurm_job_id, str)
        or not slurm_job_id.isascii()
        or not slurm_job_id.isdigit()
    ):
        raise RuntimeError("main retry audit requires a numeric Slurm job id")
    payload = {
        "schema_version": RETRY_MAP_SCHEMA,
        "status": "strongly_audited_missing_or_invalid_only",
        "profile": profile,
        "stage": stage,
        "matrix_sha256": matrix["matrix_sha256"],
        "stage_map_sha256": stage_map["map_sha256"],
        "audit_id": _safe_retry_audit_id(audit_id),
        "indices": indices,
        "cell_ids": [cell["cell_id"] for cell in cells],
        "selected_state_sha256": states,
        "actions": actions,
        "slurm_job_id": slurm_job_id,
    }
    payload["retry_map_sha256"] = object_sha256(payload)
    destination = retry_map_path(output, stage, audit_id)
    write_json_once(destination, payload)
    append_ledger_once(
        ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "retry_map_audited",
            "profile": profile,
            "stage": stage,
            "retry_map": destination.relative_to(output).as_posix(),
            "retry_map_sha256": payload["retry_map_sha256"],
            "retry_cell_count": len(indices),
            "slurm_job_id": slurm_job_id,
        },
    )
    return payload


def require_retry_map_registered(
    retry_path: str | Path,
    payload: Mapping[str, Any],
    *,
    output_root: str | Path,
    ledger: str | Path,
    spec: Mapping[str, Any] | None = None,
    cluster_root: str | Path | None = None,
) -> None:
    if spec is None:
        output = lexical_path(output_root)
        path = lexical_path(retry_path)
    else:
        output = require_safe_cluster_path(
            spec, output_root, cluster_root=cluster_root
        )
        path = require_safe_cluster_path(
            spec, retry_path, cluster_root=cluster_root
        )
    if path.is_symlink() or not path.is_file() or output not in path.parents:
        raise ValueError("retry map is outside its frozen output root")
    relative = path.relative_to(output).as_posix()
    matches = [
        row
        for row in verify_ledger(ledger)
        if row.get("event") == "retry_map_audited"
        and row.get("profile") == payload.get("profile")
        and row.get("stage") == payload.get("stage")
        and row.get("retry_map") == relative
        and row.get("retry_map_sha256") == payload.get("retry_map_sha256")
        and row.get("retry_cell_count") == len(payload.get("indices", ()))
        and row.get("slurm_job_id") == payload.get("slurm_job_id")
    ]
    if len(matches) != 1:
        raise ValueError("retry map is not uniquely registered by the audit ledger")


def verify_stage(
    profile: str,
    stage: str,
    *,
    spec: Mapping[str, Any],
    checkout: str | Path,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    output = profile_output(spec, profile, cluster_root=cluster_root)
    protocol, _, matrix = load_validated_main_run(output, checkout, profile, spec)
    stage_map = read_json(map_path(output, stage))
    validate_stage_map(stage_map, matrix, profile, stage)
    require_previous_stage_verified(output, stage, profile, matrix)
    existing_path = stage_marker_path(output, stage)
    if existing_path.is_file():
        existing = read_json(existing_path)
        validate_stage_marker(
            existing,
            profile=profile,
            stage=stage,
            matrix=matrix,
            stage_map=stage_map,
        )
        validate_stage_result_files(output, existing)
        return existing
    result_files = []
    for entry in stage_map["entries"]:
        cell = cell_for_array_index(stage_map, matrix, entry["array_index"])
        result_path = _validated_cell(
            output, matrix, protocol, cell, Path(checkout).resolve()
        )
        result_files.append(
            {
                "cell_id": cell["cell_id"],
                "path": result_path.relative_to(output).as_posix(),
                "sha256": file_sha256(result_path),
            }
        )
    marker = {
        "schema_version": STAGE_MARKER_SCHEMA,
        "status": "verified_complete",
        "profile": profile,
        "stage": stage,
        "matrix_sha256": matrix["matrix_sha256"],
        "map_sha256": stage_map["map_sha256"],
        "verified_cell_count": stage_map["count"],
        "result_files": result_files,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    marker["stage_verification_sha256"] = object_sha256(marker)
    write_json_once(existing_path, marker)
    append_ledger(
        ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "stage_verified",
            "profile": profile,
            "stage": stage,
            "verified_cell_count": stage_map["count"],
            "stage_verification_sha256": marker["stage_verification_sha256"],
        },
    )
    return marker


def require_all_stages_verified(
    output: Path, profile: str, matrix: Mapping[str, Any]
) -> None:
    for stage in STAGES:
        stage_map = read_json(map_path(output, stage))
        validate_stage_map(stage_map, matrix, profile, stage)
        marker_path = stage_marker_path(output, stage)
        if not marker_path.is_file():
            raise RuntimeError(f"profile finalization is blocked on stage {stage}")
        marker = read_json(marker_path)
        validate_stage_marker(
            marker,
            profile=profile,
            stage=stage,
            matrix=matrix,
            stage_map=stage_map,
        )
        validate_stage_result_files(output, marker)


def profile_marker_path(output: str | Path) -> Path:
    return Path(output) / "cluster_profile_verified.json"


def _profile_result_files(output: Path, profile: str) -> dict[str, str]:
    names = (
        ("hpo_trials.json", "hpo_selection.json")
        if profile == "pilot"
        else ("analysis.json", "REPORT.md")
    )
    missing = [
        name
        for name in names
        if not (output / name).is_file() or (output / name).is_symlink()
    ]
    if missing:
        raise FileNotFoundError(f"profile result artifacts are absent: {missing}")
    return {name: file_sha256(output / name) for name in names}


def require_profile_verified(output: str | Path, profile: str) -> dict[str, Any]:
    if profile not in ("pilot", "confirmatory"):
        raise ValueError("profile must be pilot or confirmatory")
    output_path = Path(output)
    path = profile_marker_path(output)
    if path.is_symlink():
        raise ValueError(f"{profile} profile marker must not be a symlink")
    if not path.is_file():
        raise RuntimeError(f"{profile} profile is not verified")
    marker = read_json(path)
    required = {
        "schema_version",
        "status",
        "profile",
        "matrix_sha256",
        "expected_cell_count",
        "result_files",
        "slurm_job_id",
        "profile_verification_sha256",
    }
    matrix = read_json(output_path / "matrix.json")
    expected_results = _profile_result_files(output_path, profile)
    if (
        set(marker) != required
        or marker.get("schema_version") != PROFILE_MARKER_SCHEMA
        or marker.get("status") != "verified_complete"
        or marker.get("profile") != profile
        or matrix.get("profile") != profile
        or marker.get("matrix_sha256") != matrix.get("matrix_sha256")
        or marker.get("expected_cell_count") != len(matrix.get("cells", []))
        or marker.get("result_files") != expected_results
        or not isinstance(marker.get("slurm_job_id"), str)
        or not marker["slurm_job_id"].isascii()
        or not marker["slurm_job_id"].isdigit()
        or marker.get("profile_verification_sha256")
        != object_sha256(_without_digest(marker, "profile_verification_sha256"))
    ):
        raise ValueError(f"{profile} profile marker is invalid")
    return marker


def require_profile_ledger_authenticated(
    spec: Mapping[str, Any],
    profile: str,
    *,
    cluster_root: str | Path | None = None,
) -> dict[str, Any]:
    """Require the profile marker's unique matching hash-chain event."""

    marker = require_profile_verified(
        profile_output(spec, profile, cluster_root=cluster_root), profile
    )
    rows = verify_ledger(ledger_path(spec, cluster_root=cluster_root))
    matches = [
        row
        for row in rows
        if row.get("event") == "profile_verified"
        and row.get("profile") == profile
        and row.get("profile_verification_sha256")
        == marker["profile_verification_sha256"]
        and row.get("slurm_job_id") == marker["slurm_job_id"]
        and row.get("result_files") == marker["result_files"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{profile} profile marker is not uniquely authenticated by the ledger"
        )
    return marker


def finalize_profile(
    profile: str,
    *,
    spec: Mapping[str, Any],
    checkout: str | Path,
    cluster_root: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    output = profile_output(spec, profile, cluster_root=cluster_root)
    _, _, matrix = load_validated_main_run(output, checkout, profile, spec)
    require_all_stages_verified(output, profile, matrix)
    if profile == "confirmatory":
        require_profile_ledger_authenticated(
            spec, "pilot", cluster_root=cluster_root
        )
        diagnostic_summary = output / "diagnostics" / "diagnostic_summary.json"
        if not diagnostic_summary.is_file():
            raise RuntimeError(
                "confirmatory finalization requires completed registered diagnostics"
            )
    executable = str(python or Path(spec["environment_root"]) / "bin" / "python")
    runner = Path(checkout) / "dreamer_imf_comparison" / "scripts" / "run_matched_objective_benchmark.py"
    common = [
        "--profile",
        profile,
        "--output",
        str(output),
        "--workspace",
        str(Path(checkout).resolve()),
    ]
    subprocess.run([executable, str(runner), "finalize", *common], check=True)
    subprocess.run([executable, str(runner), "verify", *common], check=True)
    result_files = _profile_result_files(output, profile)
    marker = {
        "schema_version": PROFILE_MARKER_SCHEMA,
        "status": "verified_complete",
        "profile": profile,
        "matrix_sha256": matrix["matrix_sha256"],
        "expected_cell_count": spec["matched_objective"][profile][
            "expected_total_cells"
        ],
        "result_files": result_files,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    marker["profile_verification_sha256"] = object_sha256(marker)
    write_json_once(profile_marker_path(output), marker)
    require_profile_verified(output, profile)
    append_ledger(
        ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "profile_verified",
            "profile": profile,
            "matrix_sha256": matrix["matrix_sha256"],
            "profile_verification_sha256": marker["profile_verification_sha256"],
            "slurm_job_id": marker["slurm_job_id"],
            "result_files": result_files,
        },
    )
    return marker


def format_array_indices(indices: Sequence[int], concurrency: int = 4) -> str:
    """Compress indices into a canonical Slurm expression capped at four.

    Controls compute cells use ``%1`` because their upstream compiler-IR writer
    has a shared content-addressed temporary filename.  Every other registered
    array uses ``%4``.
    """

    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or not 1 <= concurrency <= 4
    ):
        raise ValueError("Slurm array concurrency must be between one and four")
    if not indices or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indices
    ):
        raise ValueError("array indices must be integers")
    normalized = sorted(set(indices))
    if any(index < 0 for index in normalized):
        raise ValueError("at least one nonnegative array index is required")
    ranges: list[str] = []
    start = previous = normalized[0]
    for index in normalized[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return f"{','.join(ranges)}%{concurrency}"


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "contract",
            "initialize",
            "freeze",
            "run-cell",
            "missing",
            "audit-retry",
            "verify-stage",
            "finalize",
            "verify-ledger",
        ),
    )
    parser.add_argument("--profile", choices=("pilot", "confirmatory"))
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--cluster-root")
    parser.add_argument("--checkout")
    parser.add_argument("--python")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--audit-id")
    parser.add_argument("--retry-map")
    arguments = parser.parse_args()
    spec = load_spec()
    root_override = Path(arguments.cluster_root).resolve() if arguments.cluster_root else None
    checkout = Path(arguments.checkout).resolve() if arguments.checkout else source_root(
        spec, cluster_root=root_override
    )
    if arguments.command == "contract":
        print(json.dumps(spec, indent=2, sort_keys=True))
        print("NEURIPS_CLUSTER_CONTRACT_VALID")
        return 0
    if arguments.command == "initialize":
        result = initialize_cluster_root(
            spec=spec,
            checkout=checkout,
            cluster_root=root_override,
        )
    elif arguments.command == "verify-ledger":
        rows = verify_ledger(ledger_path(spec, cluster_root=root_override))
        result = {"events": len(rows), "last_event_sha256": rows[-1]["event_sha256"]}
    else:
        if arguments.profile is None:
            parser.error(f"{arguments.command} requires --profile")
        if arguments.command == "freeze":
            result = freeze_profile(
                arguments.profile,
                spec=spec,
                checkout=checkout,
                cluster_root=root_override,
                python=arguments.python,
            )
        elif arguments.command == "run-cell":
            if arguments.stage is None or arguments.array_index is None:
                parser.error("run-cell requires --stage and --array-index")
            result = {
                "cell_id": run_array_cell(
                    arguments.profile,
                    arguments.stage,
                    arguments.array_index,
                    spec=spec,
                    checkout=checkout,
                    cluster_root=root_override,
                    python=arguments.python,
                    retry_map_path=arguments.retry_map,
                )
            }
        elif arguments.command == "missing":
            if arguments.stage is None:
                parser.error("missing requires --stage")
            indices = incomplete_indices(
                arguments.profile,
                arguments.stage,
                spec=spec,
                checkout=checkout,
                cluster_root=root_override,
                strong_validation=not arguments.quick,
            )
            result = {
                "indices": indices,
                "slurm_array": (
                    format_array_indices(indices) if indices else None
                ),
            }
        elif arguments.command == "audit-retry":
            if arguments.stage is None:
                parser.error("audit-retry requires --stage")
            audit_id = arguments.audit_id or os.environ.get("SLURM_JOB_ID")
            if not audit_id:
                parser.error("audit-retry requires --audit-id outside Slurm")
            result = audit_retry_map(
                arguments.profile,
                arguments.stage,
                audit_id,
                spec=spec,
                checkout=checkout,
                cluster_root=root_override,
            )
        elif arguments.command == "verify-stage":
            if arguments.stage is None:
                parser.error("verify-stage requires --stage")
            result = verify_stage(
                arguments.profile,
                arguments.stage,
                spec=spec,
                checkout=checkout,
                cluster_root=root_override,
            )
        else:
            result = finalize_profile(
                arguments.profile,
                spec=spec,
                checkout=checkout,
                cluster_root=root_override,
                python=arguments.python,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
