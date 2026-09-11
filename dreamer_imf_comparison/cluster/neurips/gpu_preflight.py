#!/usr/bin/env python3
"""Strict single-L40S JAX and EGL/DMC preflight for claim-bearing jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo


CLUSTER_DIR = Path(__file__).resolve().parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import workflow


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run_read_only(command: list[str]) -> str:
    process = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.stderr.strip():
        raise RuntimeError(f"read-only site command wrote stderr: {command[0]}")
    return process.stdout


def parse_ws_list(text: str, *, workspace_id: str) -> dict:
    """Parse exactly one Leipzig ``ws_list`` block by immutable workspace id."""

    blocks = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("id:"):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    matches = []
    for block in blocks:
        identifier = block[0].split(":", maxsplit=1)[1].strip()
        if identifier != workspace_id:
            continue
        fields = {}
        for line in block[1:]:
            if not line.strip():
                continue
            if ":" not in line:
                raise ValueError("ws_list workspace field lacks a colon")
            name, value = line.split(":", maxsplit=1)
            key = " ".join(name.strip().split())
            if key in fields:
                raise ValueError("ws_list workspace field is duplicated")
            fields[key] = value.strip()
        matches.append(fields)
    if len(matches) != 1:
        raise ValueError("ws_list does not contain exactly one registered workspace")
    fields = matches[0]
    required = {
        "workspace directory",
        "remaining time",
        "comment",
        "creation time",
        "expiration date",
        "filesystem name",
        "available extensions",
    }
    if set(fields) != required:
        raise ValueError("ws_list registered workspace field set is unexpected")
    remaining_tokens = fields["remaining time"].split()
    if len(remaining_tokens) not in (2, 4):
        raise ValueError("ws_list remaining time is malformed")
    remaining_days = 0
    remaining_hours = 0
    seen_units: set[str] = set()
    for index in range(0, len(remaining_tokens), 2):
        value = remaining_tokens[index]
        unit = remaining_tokens[index + 1]
        if not value.isascii() or not value.isdigit():
            raise ValueError("ws_list remaining time is nonnumeric")
        canonical_unit = "days" if unit in ("day", "days") else (
            "hours" if unit in ("hour", "hours") else None
        )
        if canonical_unit is None or canonical_unit in seen_units:
            raise ValueError("ws_list remaining time uses an unsupported or duplicate unit")
        seen_units.add(canonical_unit)
        if canonical_unit == "days":
            remaining_days = int(value)
        elif canonical_unit == "hours":
            remaining_hours = int(value)
    if remaining_hours >= 24:
        raise ValueError("ws_list remaining hours are noncanonical")
    extensions = fields["available extensions"]
    if not extensions.isascii() or not extensions.isdigit():
        raise ValueError("ws_list extension count is malformed")
    return {
        "workspace_directory": fields["workspace directory"],
        "remaining_days": remaining_days,
        "remaining_hours": remaining_hours,
        "comment": fields["comment"],
        "creation_time": fields["creation time"],
        "expiration_date": fields["expiration date"],
        "filesystem_name": fields["filesystem name"],
        "available_extensions": int(extensions),
    }


def parse_lfs_quota(text: str, *, filesystem: str, user: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or not re.fullmatch(
        rf"Disk quotas for usr {re.escape(user)} \(uid [0-9]+\):", lines[0].strip()
    ):
        raise ValueError("lfs quota header does not identify the registered user")
    expected_header = (
        "Filesystem",
        "kbytes",
        "bquota",
        "blimit",
        "bgrace",
        "files",
        "iquota",
        "ilimit",
        "igrace",
    )
    header_matches = [tuple(line.split()) for line in lines[1:] if "Filesystem" in line]
    if header_matches != [expected_header]:
        raise ValueError("lfs quota column schema differs")
    rows = [line.split() for line in lines[1:] if line.split()[0] == filesystem]
    if len(rows) != 1 or len(rows[0]) != len(expected_header):
        raise ValueError("lfs quota lacks one exact registered-filesystem row")
    row = dict(zip(expected_header, rows[0], strict=True))
    numeric = ("kbytes", "bquota", "blimit", "files", "iquota", "ilimit")
    if any(not row[name].isascii() or not row[name].isdigit() for name in numeric):
        raise ValueError("lfs quota numeric column is malformed")
    result = {name: int(row[name]) for name in numeric}
    result.update(
        {
            "filesystem": row["Filesystem"],
            "bgrace": row["bgrace"],
            "igrace": row["igrace"],
        }
    )
    return result


def authoritative_workspace_evidence(spec: dict) -> tuple[dict, dict]:
    allocator = spec["workspace_allocator"]
    workspace_id = allocator["id"]
    ws_find = _run_read_only(["ws_find", workspace_id])
    ws_list = _run_read_only(["ws_list"])
    quota = _run_read_only(
        [
            "lfs",
            "quota",
            "-u",
            allocator["quota_user"],
            allocator["quota_filesystem"],
        ]
    )
    resolved_paths = [line.strip() for line in ws_find.splitlines() if line.strip()]
    _require(
        resolved_paths == [spec["workspace_root"]],
        "ws_find did not resolve exactly the registered workspace",
    )
    parsed_workspace = parse_ws_list(ws_list, workspace_id=workspace_id)
    parsed_quota = parse_lfs_quota(
        quota,
        filesystem=allocator["quota_filesystem"],
        user=allocator["quota_user"],
    )
    timezone = ZoneInfo(allocator["timezone"])
    expiration_local = dt.datetime.strptime(
        parsed_workspace["expiration_date"], "%a %b %d %H:%M:%S %Y"
    ).replace(tzinfo=timezone)
    creation_local = dt.datetime.strptime(
        parsed_workspace["creation_time"], "%a %b %d %H:%M:%S %Y"
    ).replace(tzinfo=timezone)
    observed = dt.datetime.now(dt.timezone.utc)
    remaining_seconds = int(
        (expiration_local.astimezone(dt.timezone.utc) - observed).total_seconds()
    )
    reported_seconds = 3600 * (
        24 * parsed_workspace["remaining_days"]
        + parsed_workspace["remaining_hours"]
    )
    _require(
        reported_seconds <= remaining_seconds + 120
        and remaining_seconds < reported_seconds + 3720,
        "ws_list remaining time disagrees with its expiration date",
    )
    _require(
        remaining_seconds >= 3600 * allocator["minimum_remaining_hours"],
        "registered workspace lifetime is too short for the campaign",
    )
    _require(
        parsed_workspace["workspace_directory"] == spec["workspace_root"]
        and parsed_workspace["comment"] == allocator["comment"]
        and parsed_workspace["available_extensions"]
        >= allocator["minimum_available_extensions"],
        "registered workspace identity or extension policy differs",
    )
    remaining_quota_kib = parsed_quota["blimit"] - parsed_quota["kbytes"]
    remaining_inodes = parsed_quota["ilimit"] - parsed_quota["files"]
    _require(
        parsed_quota["blimit"] >= allocator["minimum_block_limit_kib"]
        and parsed_quota["ilimit"] >= allocator["minimum_inode_limit"]
        and remaining_quota_kib * 1024 >= spec["minimum_free_workspace_bytes"]
        and remaining_inodes > 0,
        "authoritative Lustre user quota is insufficient",
    )
    workspace_record = {
        "interface": "ws_find_plus_ws_list",
        "id": workspace_id,
        **parsed_workspace,
        "timezone": allocator["timezone"],
        "creation_utc": creation_local.astimezone(dt.timezone.utc).isoformat(),
        "expiration_utc": expiration_local.astimezone(dt.timezone.utc).isoformat(),
        "observed_at_utc": observed.isoformat(),
        "remaining_seconds_at_observation": remaining_seconds,
        "ws_find_stdout_sha256": hashlib.sha256(ws_find.encode("utf-8")).hexdigest(),
        "ws_list_stdout_sha256": hashlib.sha256(ws_list.encode("utf-8")).hexdigest(),
    }
    quota_record = {
        "interface": "lfs_quota_user",
        "user": allocator["quota_user"],
        **parsed_quota,
        "remaining_quota_kib": remaining_quota_kib,
        "remaining_inodes": remaining_inodes,
        "stdout_sha256": hashlib.sha256(quota.encode("utf-8")).hexdigest(),
    }
    return workspace_record, quota_record


def run_preflight(
    *,
    spec: dict,
    checkout: Path,
    cluster_root: Path | None = None,
) -> dict:
    contract = workflow.read_json(
        workflow.run_contract_path(spec, cluster_root=cluster_root)
    )
    workflow.validate_run_contract(contract, spec, checkout)
    expected_environment = spec["runtime"]["environment"]
    observed_environment = {
        name: os.environ.get(name) for name in expected_environment
    }
    _require(
        observed_environment == expected_environment,
        "JAX/MuJoCo environment variables differ from the frozen contract",
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(
        isinstance(visible, str)
        and bool(visible.strip())
        and visible == visible.strip()
        and "," not in visible
        and visible.strip() not in {"-1", "NoDevFiles"},
        "exactly one CUDA device must be visible",
    )
    observed_versions = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "numpy": version("numpy"),
        "dm-control": version("dm-control"),
        "mujoco": version("mujoco"),
    }
    expected_versions = {
        name: spec["runtime"][name] for name in observed_versions
    }
    _require(
        observed_versions == expected_versions,
        f"installed runtime differs from lock: {observed_versions}",
    )

    import jax
    import jax.numpy as jnp
    import numpy as np
    from dm_control import suite

    devices = jax.devices()
    kinds = [str(getattr(device, "device_kind", "")) for device in devices]
    _require(jax.default_backend() == "gpu", "JAX did not initialize the GPU backend")
    _require(len(devices) == 1, "JAX must see exactly one device")
    _require(
        all(device.platform == "gpu" for device in devices),
        "the visible JAX device is not a GPU",
    )
    _require(
        spec["scheduler"]["gpu_model_substring"] in kinds[0],
        f"worker GPU is not the registered model: {kinds[0]!r}",
    )
    _require(jax.config.jax_enable_x64 is False, "JAX x64 must remain disabled")
    tiny = jnp.linalg.norm(jnp.arange(4096, dtype=jnp.float32)).block_until_ready()
    _require(bool(jnp.isfinite(tiny)), "tiny JAX GPU operation was non-finite")

    environment = suite.load("cartpole", "swingup", task_kwargs={"random": 0})
    environment.reset()
    pixels = np.asarray(
        environment.physics.render(height=16, width=16, camera_id=0)
    )
    _require(pixels.shape == (16, 16, 3), "tiny DMC render has the wrong shape")
    _require(pixels.dtype == np.uint8, "tiny DMC render has the wrong dtype")
    _require(np.isfinite(pixels).all(), "tiny DMC render is non-finite")

    workspace = Path(spec["workspace_root"])
    _require(workspace.is_dir(), "registered personal workspace is absent")
    workspace_stat = workspace.stat()
    effective_uid = os.geteuid()
    _require(
        workspace_stat.st_uid == effective_uid,
        "registered personal workspace is not owned by the effective user",
    )
    _require(
        os.access(workspace, os.R_OK | os.W_OK | os.X_OK),
        "registered personal workspace is not readable, writable, and searchable",
    )
    usage = shutil.disk_usage(workspace)
    _require(
        usage.free >= spec["minimum_free_workspace_bytes"],
        "registered workspace has insufficient free capacity",
    )
    allocator_record, quota_record = authoritative_workspace_evidence(spec)
    record = {
        "schema_version": workflow.PREFLIGHT_SCHEMA,
        "status": "verified_gpu_worker",
        "cluster_spec_sha256": contract["cluster_spec_sha256"],
        "dependency_lock_sha256": contract["dependency_lock_sha256"],
        "source_commit": contract["source_commit"],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "cuda_visible_devices": visible,
        },
        "versions": observed_versions,
        "environment": observed_environment,
        "jax_runtime": {
            "backend": jax.default_backend(),
            "visible_device_count": len(devices),
            "device_kinds": kinds,
            "device_platforms": [device.platform for device in devices],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "tiny_device_check": "passed",
        },
        "dmc_render": {
            "task": "dmc_cartpole_swingup",
            "shape": list(pixels.shape),
            "dtype": str(pixels.dtype),
            "finite": bool(np.isfinite(pixels).all()),
            "sha256": hashlib.sha256(pixels.tobytes(order="C")).hexdigest(),
        },
        "workspace_storage": {
            "root": spec["workspace_root"],
            "measurement": "shutil.disk_usage",
            "total_bytes": int(usage.total),
            "free_bytes": int(usage.free),
            "owner_uid": int(workspace_stat.st_uid),
            "effective_uid": int(effective_uid),
            "owned_by_effective_user": workspace_stat.st_uid == effective_uid,
            "read_write_search_access": os.access(
                workspace, os.R_OK | os.W_OK | os.X_OK
            ),
            "claim_scope": "filesystem_capacity_not_user_quota",
        },
        "workspace_allocator": allocator_record,
        "workspace_quota": quota_record,
    }
    record["preflight_sha256"] = workflow.object_sha256(record)
    workflow.write_json_once(
        workflow.preflight_path(spec, cluster_root=cluster_root), record
    )
    workflow.validate_preflight_record(record, spec=spec, contract=contract)
    workflow.append_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "gpu_preflight_verified",
            "preflight_sha256": record["preflight_sha256"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root")
    parser.add_argument("--checkout")
    arguments = parser.parse_args()
    spec = workflow.load_spec()
    root = Path(arguments.cluster_root).resolve() if arguments.cluster_root else None
    checkout = (
        Path(arguments.checkout).resolve()
        if arguments.checkout
        else workflow.source_root(spec, cluster_root=root)
    )
    record = run_preflight(spec=spec, checkout=checkout, cluster_root=root)
    print(json.dumps(record, indent=2, sort_keys=True))
    print("NEURIPS_GPU_PREFLIGHT_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
