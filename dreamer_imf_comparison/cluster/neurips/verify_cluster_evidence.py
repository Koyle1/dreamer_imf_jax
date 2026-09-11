#!/usr/bin/env python3
"""Recompute cluster evidence gates from immutable source and raw artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


CLUSTER_DIR = Path(__file__).resolve().parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import workflow
import supplementary


def _verify_lock(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    required = {
        "jax": "0.8.1",
        "jaxlib": "0.8.1",
        "numpy": "2.5.3",
        "dm-control": "1.0.46",
        "mujoco": "3.13.0",
        "jax-cuda12-plugin": "0.8.1",
        "jax-cuda12-pjrt": "0.8.1",
    }
    packages: dict[str, str] = {}
    current: str | None = None
    hashes: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        core = stripped[:-1].rstrip() if stripped.endswith("\\") else stripped
        match = re.fullmatch(r"([a-z0-9][a-z0-9._-]*)==([^ ]+)", core)
        if match:
            current = match.group(1)
            if current in packages:
                raise ValueError("dependency lock contains a duplicate package")
            packages[current] = match.group(2)
            hashes[current] = 0
        elif stripped.startswith("--hash=sha256:"):
            if current is None or not re.fullmatch(
                r"--hash=sha256:[0-9a-f]{64}", core
            ):
                raise ValueError("dependency lock hash is malformed")
            hashes[current] += 1
    if any(packages.get(name) != version for name, version in required.items()):
        raise ValueError("dependency lock lacks an exact registered runtime pin")
    if not packages or any(hashes[name] == 0 for name in packages):
        raise ValueError("every locked package must carry at least one wheel hash")
    if any(operator in text for operator in (">=", "~=", " @ ")):
        raise ValueError("dependency lock contains a non-exact requirement")
    return {
        "packages": len(packages),
        "sha256": workflow.file_sha256(path),
    }


def _verify_sbatch_contract(spec: Mapping[str, Any]) -> list[str]:
    scripts = sorted(CLUSTER_DIR.glob("*.sbatch"))
    if not scripts:
        raise ValueError("no Slurm scripts are registered")
    required = {
        "#SBATCH --account=dep_inin_dat",
        "#SBATCH --partition=gpu-l40s",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=2-00:00:00",
    }
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        if not required <= set(text.splitlines()):
            raise ValueError(f"Slurm resources differ in {path.name}")
        dependency_directives = [
            line.lower() for line in text.splitlines() if "--dependency" in line
        ]
        if any("aftercorr" in line for line in dependency_directives):
            raise ValueError("correlated Slurm array dependencies are forbidden")
    if spec["scheduler"]["array_concurrency"] != 4:
        raise ValueError("array concurrency contract changed")
    return [path.name for path in scripts]


def verify_preflight(
    spec: Mapping[str, Any], *, cluster_root: Path | None
) -> dict[str, Any]:
    checkout = workflow.source_root(spec, cluster_root=cluster_root)
    contract = workflow.read_json(
        workflow.run_contract_path(spec, cluster_root=cluster_root)
    )
    workflow.validate_run_contract(contract, spec, checkout)
    preflight = workflow.read_json(
        workflow.preflight_path(spec, cluster_root=cluster_root)
    )
    workflow.validate_preflight_record(preflight, spec=spec, contract=contract)
    lock = _verify_lock(checkout / "dreamer_imf_comparison" / workflow.LOCK_PATH.name)
    if lock["sha256"] != contract["dependency_lock_sha256"]:
        raise ValueError("checked-out lock differs from the run contract")
    scripts = _verify_sbatch_contract(spec)
    ledger = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    events = [row["event"] for row in ledger]
    if "cluster_root_initialized" not in events or "gpu_preflight_verified" not in events:
        raise ValueError("preflight provenance events are incomplete")
    return {
        "source_commit": contract["source_commit"],
        "dependency_lock": lock,
        "slurm_scripts": scripts,
        "preflight_sha256": preflight["preflight_sha256"],
        "workspace_expiration_utc": preflight["workspace_allocator"][
            "expiration_utc"
        ],
        "remaining_quota_kib": preflight["workspace_quota"][
            "remaining_quota_kib"
        ],
        "ledger_events": len(ledger),
    }


def _verify_main_profile(
    spec: Mapping[str, Any], profile: str, *, cluster_root: Path | None
) -> dict[str, Any]:
    checkout = workflow.source_root(spec, cluster_root=cluster_root)
    output = workflow.profile_output(spec, profile, cluster_root=cluster_root)
    protocol, _, matrix = workflow.load_validated_main_run(
        output, checkout, profile, spec
    )
    counts = workflow.validate_main_matrix_shape(matrix, profile, spec)
    for stage in workflow.STAGES:
        stage_map = workflow.read_json(workflow.map_path(output, stage))
        workflow.validate_stage_map(stage_map, matrix, profile, stage)
        marker = workflow.read_json(workflow.stage_marker_path(output, stage))
        workflow.validate_stage_marker(
            marker,
            profile=profile,
            stage=stage,
            matrix=matrix,
            stage_map=stage_map,
        )
        workflow.validate_stage_result_files(output, marker)
    profile_marker = workflow.require_profile_verified(output, profile)
    ledger = workflow.verify_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root)
    )
    matching_events = [
        row
        for row in ledger
        if row.get("event") == "profile_verified"
        and row.get("profile") == profile
        and row.get("profile_verification_sha256")
        == profile_marker["profile_verification_sha256"]
        and row.get("slurm_job_id") == profile_marker["slurm_job_id"]
        and row.get("result_files") == profile_marker["result_files"]
    ]
    if not matching_events:
        raise ValueError("profile marker is not authenticated by the cluster ledger")
    workflow._add_source_paths(checkout)
    if profile == "pilot":
        from dreamer_imf_compare.matched_objective_benchmark import (
            verify_pilot_hpo_output_root,
        )

        verified = verify_pilot_hpo_output_root(output, workspace=checkout)
        selection = workflow.read_json(output / "hpo_selection.json")
        if selection.get("confirmatory_outcomes_accessed") is not False:
            raise ValueError("pilot selection accessed confirmatory outcomes")
    else:
        from dreamer_imf_compare.matched_objective_benchmark import verify_output_root

        workflow.require_profile_verified(
            workflow.profile_output(spec, "pilot", cluster_root=cluster_root), "pilot"
        )
        verified = verify_output_root(output, workspace=checkout)
    if verified.get("cells") != spec["matched_objective"][profile]["expected_total_cells"]:
        raise ValueError("verified profile cell count mismatch")
    if profile_marker["matrix_sha256"] != matrix["matrix_sha256"]:
        raise ValueError("profile marker matrix mismatch")
    return {
        "profile": profile,
        "counts": counts,
        "cells": len(matrix["cells"]),
        "matrix_sha256": matrix["matrix_sha256"],
        "profile_verification_sha256": profile_marker[
            "profile_verification_sha256"
        ],
        "protocol_sha256": matrix["protocol_sha256"],
    }


def verify_controls(
    spec: Mapping[str, Any], *, cluster_root: Path | None
) -> dict[str, Any]:
    checkout = workflow.source_root(spec, cluster_root=cluster_root)
    development = supplementary.require_controls_profile_verified(
        spec,
        "development",
        checkout=checkout,
        cluster_root=cluster_root,
    )
    confirmatory = supplementary.require_controls_profile_verified(
        spec,
        "confirmatory",
        checkout=checkout,
        cluster_root=cluster_root,
    )
    if (
        development.get("selection_sha256") is None
        or confirmatory["prerequisites"].get(
            "development_controls_profile_verification_sha256"
        )
        != development["profile_verification_sha256"]
        or confirmatory["prerequisites"].get("development_selection_sha256")
        != development["selection_sha256"]
    ):
        raise ValueError("controls development-selection dependency is not exact")
    return {
        "development": {
            "execution_units": development["expected_execution_units"],
            "profile_verification_sha256": development[
                "profile_verification_sha256"
            ],
        },
        "confirmatory": {
            "execution_units": confirmatory["expected_execution_units"],
            "profile_verification_sha256": confirmatory[
                "profile_verification_sha256"
            ],
        },
    }


def verify_pixel(
    spec: Mapping[str, Any], *, cluster_root: Path | None
) -> dict[str, Any]:
    checkout = workflow.source_root(spec, cluster_root=cluster_root)
    marker = supplementary.require_pixel_profile_verified(
        spec, checkout=checkout, cluster_root=cluster_root
    )
    return {
        "artifacts": marker["expected_artifacts"],
        "tracks": marker["tracks"],
        "aggregate_sha256": marker["aggregate_sha256"],
        "profile_verification_sha256": marker[
            "profile_verification_sha256"
        ],
    }


def verify_superiority(
    spec: Mapping[str, Any], *, cluster_root: Path | None
) -> dict[str, Any]:
    profile = _verify_main_profile(spec, "confirmatory", cluster_root=cluster_root)
    output = workflow.profile_output(spec, "confirmatory", cluster_root=cluster_root)
    analysis = workflow.read_json(output / "analysis.json")
    decision = analysis.get("superiority", {})
    if decision.get("passed") is not True:
        raise RuntimeError("the frozen four-interval superiority gate did not pass")
    intervals = [
        row["interval"]
        for track in analysis["tracks"].values()
        for row in track.values()
    ]
    if len(intervals) != 4 or any(float(interval["lower"]) <= 0.0 for interval in intervals):
        raise RuntimeError("at least one adjusted superiority lower bound is not positive")
    return {"profile": profile, "superiority": decision}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--confirmatory", action="store_true")
    modes.add_argument("--controls", action="store_true")
    modes.add_argument("--pixel", action="store_true")
    modes.add_argument("--superiority", action="store_true")
    parser.add_argument("--cluster-root")
    arguments = parser.parse_args()
    spec = workflow.load_spec()
    root = Path(
        arguments.cluster_root
        or os.environ.get("NEURIPS_CLUSTER_ROOT", spec["workspace_root"])
    ).resolve()
    cluster_root = None if str(root) == spec["workspace_root"] else root
    if arguments.preflight:
        result = verify_preflight(spec, cluster_root=cluster_root)
        token = "NEURIPS_CLUSTER_PREFLIGHT_VERIFIED"
    elif arguments.pilot:
        result = _verify_main_profile(spec, "pilot", cluster_root=cluster_root)
        token = "NEURIPS_PILOT_VERIFIED"
    elif arguments.confirmatory:
        result = _verify_main_profile(spec, "confirmatory", cluster_root=cluster_root)
        token = "NEURIPS_CONFIRMATORY_VERIFIED"
    elif arguments.controls:
        result = verify_controls(spec, cluster_root=cluster_root)
        token = "NEURIPS_CLUSTER_CONTROLS_VERIFIED"
    elif arguments.pixel:
        result = verify_pixel(spec, cluster_root=cluster_root)
        token = "NEURIPS_CLUSTER_PIXEL_VERIFIED"
    else:
        result = verify_superiority(spec, cluster_root=cluster_root)
        token = "TRAJECTORY_IMF_SUPERIORITY_GATE_PASSED"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
