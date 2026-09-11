#!/usr/bin/env python3
"""Explicit Slurm submitter; dry-run by default and stage-marker gated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


CLUSTER_DIR = Path(__file__).resolve().parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import workflow
import supplementary
import feasibility


def build_sbatch_command(
    script: str | Path,
    *,
    exports: Mapping[str, str] | None = None,
    array: str | None = None,
    dependency_job_id: str | None = None,
) -> list[str]:
    command = ["sbatch", "--parsable"]
    if dependency_job_id is not None:
        if re.fullmatch(r"[0-9]+", dependency_job_id) is None:
            raise ValueError("Slurm dependency job id must be numeric")
        command.append(f"--dependency=afterok:{dependency_job_id}")
    if array is not None:
        match = re.fullmatch(
            r"([0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*)%([1-4])",
            array,
        )
        if match is None:
            raise ValueError("Slurm array concurrency must be between %1 and %4")
        previous_end = -2
        for token in match.group(1).split(","):
            endpoints = token.split("-", maxsplit=1)
            start = int(endpoints[0])
            end = int(endpoints[-1])
            if end < start or (len(endpoints) == 2 and end == start):
                raise ValueError("Slurm array contains a noncanonical range")
            if start <= previous_end + 1:
                raise ValueError("Slurm array ranges must be sorted, unique, and merged")
            previous_end = end
        command.append(f"--array={array}")
    if exports:
        if any(re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None for key in exports):
            raise ValueError("Slurm export names must be uppercase identifiers")
        if any(
            not isinstance(value, str)
            or not value
            or re.fullmatch(r"[A-Za-z0-9_.:/+-]+", value) is None
            for value in exports.values()
        ):
            raise ValueError("Slurm export values contain unsafe characters")
        values = ["ALL", *[f"{key}={value}" for key, value in sorted(exports.items())]]
        command.append("--export=" + ",".join(values))
    command.append(str(Path(script).resolve()))
    if any("aftercorr" in argument.lower() for argument in command):
        raise ValueError("correlated array dependencies are forbidden")
    return command


def _submit(command: Sequence[str], *, execute: bool) -> str | None:
    if not execute:
        return None
    process = subprocess.run(
        list(command), check=True, capture_output=True, text=True, timeout=30
    )
    token = process.stdout.strip().split(";", maxsplit=1)[0]
    if not token.isdigit():
        raise RuntimeError(f"sbatch returned a malformed job id: {process.stdout!r}")
    return token


def _record_submission(
    spec: Mapping[str, Any],
    *,
    kind: str,
    command: Sequence[str],
    job_id: str,
    profile: str | None = None,
    stage: str | None = None,
    cluster_root: str | Path | None = None,
) -> None:
    workflow.append_ledger(
        workflow.ledger_path(spec, cluster_root=cluster_root),
        {
            "event": "job_submitted",
            "kind": kind,
            "profile": profile,
            "stage": stage,
            "job_id": job_id,
            "command": list(command),
        },
    )


def _reject_duplicate_supplementary_array(
    spec: Mapping[str, Any],
    *,
    kind: str,
    profile: str,
    stage: str,
    retry_map_path: str | Path | None,
) -> None:
    ledger = workflow.ledger_path(spec)
    rows = workflow.verify_ledger(ledger) if ledger.is_file() else []
    prior = [
        row
        for row in rows
        if row.get("event") == "job_submitted"
        and row.get("kind") == kind
        and row.get("profile") == profile
        and row.get("stage") == stage
    ]
    active_cells: set[str] = set()
    started_events = {"cell_started", "controls_unit_started", "pixel_unit_started"}
    terminal_events = {
        "cell_completed",
        "cell_failed",
        "controls_unit_completed",
        "controls_unit_failed",
        "pixel_unit_gpu_verified",
        "pixel_unit_failed",
    }
    for row in rows:
        event = row.get("event")
        cell_id = row.get("cell_id")
        if event not in started_events | terminal_events or not isinstance(cell_id, str):
            continue
        if event.startswith("controls_") and (
            row.get("profile") != profile or row.get("stage") != stage
        ):
            continue
        if event.startswith("cell_") and (
            row.get("profile") != profile or row.get("stage") != stage
        ):
            continue
        if event in started_events:
            active_cells.add(cell_id)
        else:
            active_cells.discard(cell_id)
    if retry_map_path is None:
        if prior or active_cells:
            raise RuntimeError(
                "an initial array was already submitted or has in-flight cells; "
                "use a GPU-audited retry map"
            )
        return
    retry_path = Path(retry_map_path).resolve()
    retry_payload = workflow.read_json(retry_path)
    retry_digest = retry_payload.get("retry_map_sha256")
    selected = set(retry_payload.get("cell_ids", ()))
    if not isinstance(retry_digest, str) or not selected or not all(
        isinstance(cell_id, str) for cell_id in selected
    ):
        raise ValueError("retry map lacks a canonical digest-bound selected cell set")
    if active_cells & selected:
        raise RuntimeError("a selected retry cell is already in flight")
    token = f"NEURIPS_RETRY_MAP={retry_path}"
    if any(
        any(token in argument for argument in row.get("command", []))
        for row in prior
    ) or any(row.get("retry_map_sha256") == retry_digest for row in prior):
        raise RuntimeError("this immutable supplementary retry map was already submitted")


def submit_preflight(spec: Mapping[str, Any], *, execute: bool) -> dict[str, Any]:
    command = build_sbatch_command(CLUSTER_DIR / "preflight.sbatch")
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(spec, kind="preflight", command=command, job_id=job_id)
    return {"command": command, "job_id": job_id, "executed": execute}


def submit_controls_freeze(
    spec: Mapping[str, Any], profile: str, *, execute: bool
) -> dict[str, Any]:
    supplementary._controls_prerequisites(spec, profile, cluster_root=None)
    feasibility_record = feasibility.require_post_pilot_feasible(spec)
    command = build_sbatch_command(
        CLUSTER_DIR / "controls_freeze.sbatch",
        exports={
            "NEURIPS_CONTROLS_PROFILE": profile,
            "NEURIPS_FEASIBILITY_SHA256": feasibility_record["feasibility_sha256"],
        },
    )
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(
            spec,
            kind="controls_freeze",
            command=command,
            job_id=job_id,
            profile=profile,
        )
    return {
        "profile": profile,
        "command": command,
        "job_id": job_id,
        "executed": execute,
        "post_pilot_feasibility_sha256": feasibility_record["feasibility_sha256"],
    }


def submit_controls_stage(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    execute: bool,
    retry_map_path: str | Path | None = None,
) -> dict[str, Any]:
    checkout = Path(spec["source_root"])
    supplementary.require_controls_freeze(spec, profile, checkout)
    supplementary._require_previous_controls_stage(
        spec, profile, stage, checkout=checkout, cluster_root=None
    )
    if retry_map_path is not None:
        retry = supplementary.validate_controls_retry_for_submission(
            spec, profile, stage, retry_map_path, checkout=checkout
        )
        indices = list(retry["indices"])
    else:
        indices = supplementary.incomplete_controls_indices(
            spec,
            profile,
            stage,
            checkout=checkout,
            strong_validation=False,
        )
    exports = {
        "NEURIPS_CONTROLS_PROFILE": profile,
        "NEURIPS_CONTROLS_STAGE": stage,
    }
    if retry_map_path is not None:
        exports["NEURIPS_RETRY_MAP"] = str(Path(retry_map_path).resolve())
    array_command = None
    array_job_id = None
    if indices:
        concurrency = spec["supplementary"]["controls"][
            "stage_array_concurrency"
        ][stage]
        expression = workflow.format_array_indices(indices, concurrency)
        array_command = build_sbatch_command(
            CLUSTER_DIR / "controls_array.sbatch",
            exports=exports,
            array=expression,
        )
        if execute:
            _reject_duplicate_supplementary_array(
                spec,
                kind="controls_stage_array",
                profile=profile,
                stage=stage,
                retry_map_path=retry_map_path,
            )
        array_job_id = _submit(array_command, execute=execute)
        if array_job_id is not None:
            _record_submission(
                spec,
                kind="controls_stage_array",
                command=array_command,
                job_id=array_job_id,
                profile=profile,
                stage=stage,
            )
    verify_command = build_sbatch_command(
        CLUSTER_DIR / "controls_stage_verify.sbatch",
        exports=exports,
        dependency_job_id=array_job_id,
    )
    verify_job_id = _submit(verify_command, execute=execute)
    if verify_job_id is not None:
        _record_submission(
            spec,
            kind="controls_stage_verifier",
            command=verify_command,
            job_id=verify_job_id,
            profile=profile,
            stage=stage,
        )
    return {
        "profile": profile,
        "stage": stage,
        "missing_or_invalid_indices": indices,
        "array_command": array_command,
        "array_job_id": array_job_id,
        "verify_command": verify_command,
        "verify_job_id": verify_job_id,
        "whole_stage_dependency": (
            f"afterok:{array_job_id}" if array_job_id is not None else None
        ),
        "executed": execute,
    }


def submit_controls_retry_audit(
    spec: Mapping[str, Any], profile: str, stage: str, *, execute: bool
) -> dict[str, Any]:
    checkout = Path(spec["source_root"])
    supplementary.require_controls_freeze(spec, profile, checkout)
    supplementary._require_previous_controls_stage(
        spec, profile, stage, checkout=checkout, cluster_root=None
    )
    command = build_sbatch_command(
        CLUSTER_DIR / "controls_retry_audit.sbatch",
        exports={
            "NEURIPS_CONTROLS_PROFILE": profile,
            "NEURIPS_CONTROLS_STAGE": stage,
        },
    )
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(
            spec,
            kind="controls_retry_audit",
            command=command,
            job_id=job_id,
            profile=profile,
            stage=stage,
        )
    return {"command": command, "job_id": job_id, "executed": execute}


def submit_controls_finalize(
    spec: Mapping[str, Any], profile: str, *, execute: bool
) -> dict[str, Any]:
    checkout = Path(spec["source_root"])
    for stage in supplementary.CONTROLS_STAGES:
        supplementary.require_controls_stage_verified(
            spec, profile, stage, checkout=checkout
        )
    supplementary._controls_prerequisites(spec, profile, cluster_root=None)
    command = build_sbatch_command(
        CLUSTER_DIR / "controls_finalize.sbatch",
        exports={"NEURIPS_CONTROLS_PROFILE": profile},
    )
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(
            spec,
            kind="controls_profile_finalize",
            command=command,
            job_id=job_id,
            profile=profile,
        )
    return {
        "profile": profile,
        "command": command,
        "job_id": job_id,
        "executed": execute,
    }


def submit_pixel_freeze(
    spec: Mapping[str, Any], *, execute: bool
) -> dict[str, Any]:
    supplementary.require_authenticated_main_profile(spec, "pilot")
    feasibility_record = feasibility.require_post_pilot_feasible(spec)
    command = build_sbatch_command(
        CLUSTER_DIR / "pixel_freeze.sbatch",
        exports={
            "NEURIPS_FEASIBILITY_SHA256": feasibility_record["feasibility_sha256"]
        },
    )
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(
            spec, kind="pixel_freeze", command=command, job_id=job_id
        )
    return {
        "command": command,
        "job_id": job_id,
        "executed": execute,
        "post_pilot_feasibility_sha256": feasibility_record["feasibility_sha256"],
    }


def submit_pixel(
    spec: Mapping[str, Any],
    *,
    execute: bool,
    retry_map_path: str | Path | None = None,
) -> dict[str, Any]:
    checkout = Path(spec["source_root"])
    supplementary.require_pixel_freeze(spec, checkout=checkout)
    if retry_map_path is not None:
        retry = supplementary.validate_pixel_retry_for_submission(
            spec, retry_map_path, checkout=checkout
        )
        indices = list(retry["indices"])
    else:
        indices = supplementary.incomplete_pixel_indices(
            spec,
            checkout=checkout,
            strong_validation=False,
        )
    array_command = None
    array_job_id = None
    if indices:
        expression = workflow.format_array_indices(
            indices, spec["scheduler"]["array_concurrency"]
        )
        exports = (
            {"NEURIPS_RETRY_MAP": str(Path(retry_map_path).resolve())}
            if retry_map_path is not None
            else None
        )
        array_command = build_sbatch_command(
            CLUSTER_DIR / "pixel_array.sbatch", array=expression, exports=exports
        )
        if execute:
            _reject_duplicate_supplementary_array(
                spec,
                kind="pixel_array",
                profile="hard",
                stage="artifact",
                retry_map_path=retry_map_path,
            )
        array_job_id = _submit(array_command, execute=execute)
        if array_job_id is not None:
            _record_submission(
                spec,
                kind="pixel_array",
                command=array_command,
                job_id=array_job_id,
                profile="hard",
                stage="artifact",
            )
    finalize_command = build_sbatch_command(
        CLUSTER_DIR / "pixel_finalize.sbatch",
        dependency_job_id=array_job_id,
    )
    finalize_job_id = _submit(finalize_command, execute=execute)
    if finalize_job_id is not None:
        _record_submission(
            spec,
            kind="pixel_whole_stage_finalize",
            command=finalize_command,
            job_id=finalize_job_id,
            profile="hard",
            stage="artifact",
        )
    return {
        "profile": "hard",
        "missing_or_invalid_indices": indices,
        "array_command": array_command,
        "array_job_id": array_job_id,
        "finalize_command": finalize_command,
        "finalize_job_id": finalize_job_id,
        "whole_stage_dependency": (
            f"afterok:{array_job_id}" if array_job_id is not None else None
        ),
        "executed": execute,
    }


def submit_pixel_retry_audit(
    spec: Mapping[str, Any], *, execute: bool
) -> dict[str, Any]:
    checkout = Path(spec["source_root"])
    supplementary.require_pixel_freeze(spec, checkout=checkout)
    command = build_sbatch_command(CLUSTER_DIR / "pixel_retry_audit.sbatch")
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(
            spec,
            kind="pixel_retry_audit",
            command=command,
            job_id=job_id,
            profile="hard",
            stage="artifact",
        )
    return {"command": command, "job_id": job_id, "executed": execute}


def submit_confirmatory_freeze(
    spec: Mapping[str, Any], *, execute: bool
) -> dict[str, Any]:
    workflow.require_profile_ledger_authenticated(spec, "pilot")
    feasibility_record = feasibility.require_post_pilot_feasible(spec)
    command = build_sbatch_command(
        CLUSTER_DIR / "freeze_confirmatory.sbatch",
        exports={
            "NEURIPS_FEASIBILITY_SHA256": feasibility_record["feasibility_sha256"]
        },
    )
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(
            spec,
            kind="confirmatory_freeze",
            command=command,
            job_id=job_id,
            profile="confirmatory",
        )
    return {
        "command": command,
        "job_id": job_id,
        "executed": execute,
        "post_pilot_feasibility_sha256": feasibility_record["feasibility_sha256"],
    }


def submit_stage(
    spec: Mapping[str, Any],
    profile: str,
    stage: str,
    *,
    execute: bool,
    retry_map_path: str | Path | None = None,
) -> dict[str, Any]:
    checkout = Path(spec["source_root"])
    output = workflow.profile_output(spec, profile)
    _, _, matrix = workflow.load_validated_main_run(
        output, checkout, profile, spec
    )
    stage_map = workflow.read_json(workflow.map_path(output, stage))
    workflow.validate_stage_map(stage_map, matrix, profile, stage)
    workflow.require_previous_stage_verified(output, stage, profile, matrix)
    if retry_map_path is not None:
        resolved_retry_map = workflow.require_safe_cluster_path(spec, retry_map_path)
        if resolved_retry_map.is_symlink() or not resolved_retry_map.is_file():
            raise ValueError("main retry map must be a regular non-symlink file")
        retry_map = workflow.read_json(resolved_retry_map)
        candidate_indices = retry_map.get("indices", [])
        current_states = [
            workflow.main_cell_state_sha256(
                spec,
                output,
                workflow.cell_for_array_index(stage_map, matrix, index),
            )
            for index in candidate_indices
            if isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < stage_map["count"]
        ]
        current_actions = [
            workflow.main_cell_retry_action(
                spec,
                output,
                workflow.cell_for_array_index(stage_map, matrix, index),
            )
            for index in candidate_indices
            if isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < stage_map["count"]
        ]
        workflow.validate_retry_map(
            retry_map,
            profile=profile,
            stage=stage,
            matrix=matrix,
            stage_map=stage_map,
            current_states=current_states,
            current_actions=current_actions,
        )
        workflow.require_retry_map_registered(
            resolved_retry_map,
            retry_map,
            output_root=output,
            ledger=workflow.ledger_path(spec),
            spec=spec,
        )
        indices = list(retry_map["indices"])
    else:
        indices = workflow.incomplete_indices(
            profile,
            stage,
            spec=spec,
            checkout=checkout,
            strong_validation=False,
        )
    exports = {"NEURIPS_PROFILE": profile, "NEURIPS_STAGE": stage}
    if retry_map_path is not None:
        exports["NEURIPS_RETRY_MAP"] = str(resolved_retry_map)
        exports["NEURIPS_RETRY_MAP_SHA256"] = retry_map["retry_map_sha256"]
    array_command = None
    array_job_id = None
    if indices:
        concurrency = spec["matched_objective"]["stage_array_concurrency"][stage]
        expression = workflow.format_array_indices(
            indices, concurrency
        )
        array_command = build_sbatch_command(
            CLUSTER_DIR / "stage_array.sbatch",
            exports=exports,
            array=expression,
        )
        if execute:
            _reject_duplicate_supplementary_array(
                spec,
                kind="stage_array",
                profile=profile,
                stage=stage,
                retry_map_path=retry_map_path,
            )
        array_job_id = _submit(array_command, execute=execute)
        if array_job_id is not None:
            _record_submission(
                spec,
                kind="stage_array",
                command=array_command,
                job_id=array_job_id,
                profile=profile,
                stage=stage,
            )
    verify_command = build_sbatch_command(
        CLUSTER_DIR / "stage_verify.sbatch",
        exports=exports,
        dependency_job_id=array_job_id,
    )
    verify_job_id = _submit(verify_command, execute=execute)
    if verify_job_id is not None:
        _record_submission(
            spec,
            kind="stage_verifier",
            command=verify_command,
            job_id=verify_job_id,
            profile=profile,
            stage=stage,
        )
    return {
        "profile": profile,
        "stage": stage,
        "missing_or_failed_indices": indices,
        "array_command": array_command,
        "array_job_id": array_job_id,
        "verify_command": verify_command,
        "verify_job_id": verify_job_id,
        "executed": execute,
    }


def submit_finalize(
    spec: Mapping[str, Any], profile: str, *, execute: bool
) -> dict[str, Any]:
    output = workflow.profile_output(spec, profile)
    _, _, matrix = workflow.load_validated_main_run(
        output, spec["source_root"], profile, spec
    )
    workflow.require_all_stages_verified(output, profile, matrix)
    dependency = None
    diagnostic_command = None
    if profile == "confirmatory":
        workflow.require_profile_ledger_authenticated(spec, "pilot")
        diagnostic_command = build_sbatch_command(CLUSTER_DIR / "diagnostics.sbatch")
        dependency = _submit(diagnostic_command, execute=execute)
        if dependency is not None:
            _record_submission(
                spec,
                kind="registered_diagnostics",
                command=diagnostic_command,
                job_id=dependency,
                profile=profile,
            )
    finalize_command = build_sbatch_command(
        CLUSTER_DIR / "profile_finalize.sbatch",
        exports={"NEURIPS_PROFILE": profile},
        dependency_job_id=dependency,
    )
    finalize_job_id = _submit(finalize_command, execute=execute)
    if finalize_job_id is not None:
        _record_submission(
            spec,
            kind="profile_finalize",
            command=finalize_command,
            job_id=finalize_job_id,
            profile=profile,
        )
    return {
        "profile": profile,
        "diagnostic_command": diagnostic_command,
        "diagnostic_job_id": dependency,
        "finalize_command": finalize_command,
        "finalize_job_id": finalize_job_id,
        "executed": execute,
    }


def submit_retry_audit(
    spec: Mapping[str, Any], profile: str, stage: str, *, execute: bool
) -> dict[str, Any]:
    output = workflow.profile_output(spec, profile)
    _, _, matrix = workflow.load_validated_main_run(
        output, spec["source_root"], profile, spec
    )
    workflow.require_previous_stage_verified(output, stage, profile, matrix)
    command = build_sbatch_command(
        CLUSTER_DIR / "retry_audit.sbatch",
        exports={"NEURIPS_PROFILE": profile, "NEURIPS_STAGE": stage},
    )
    job_id = _submit(command, execute=execute)
    if job_id is not None:
        _record_submission(
            spec,
            kind="retry_audit",
            command=command,
            job_id=job_id,
            profile=profile,
            stage=stage,
        )
    return {"command": command, "job_id": job_id, "executed": execute}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "freeze-confirmatory",
            "stage",
            "audit-retry",
            "finalize",
            "freeze-controls",
            "controls-stage",
            "audit-controls-retry",
            "finalize-controls",
            "freeze-pixel",
            "pixel",
            "audit-pixel-retry",
        ),
    )
    parser.add_argument("--profile", choices=("pilot", "confirmatory"))
    parser.add_argument("--stage", choices=workflow.STAGES)
    parser.add_argument(
        "--controls-profile", choices=supplementary.CONTROLS_PROFILES
    )
    parser.add_argument("--controls-stage", choices=supplementary.CONTROLS_STAGES)
    parser.add_argument(
        "--retry-map",
        help="immutable GPU-audited retry map; submits exactly its invalid indices",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually invoke sbatch; omission prints a side-effect-free plan",
    )
    arguments = parser.parse_args()
    spec = workflow.load_spec()
    if arguments.command == "preflight":
        result = submit_preflight(spec, execute=arguments.execute)
    elif arguments.command == "freeze-confirmatory":
        result = submit_confirmatory_freeze(spec, execute=arguments.execute)
    elif arguments.command == "stage":
        if arguments.profile is None or arguments.stage is None:
            parser.error("stage requires --profile and --stage")
        result = submit_stage(
            spec,
            arguments.profile,
            arguments.stage,
            execute=arguments.execute,
            retry_map_path=arguments.retry_map,
        )
    elif arguments.command == "audit-retry":
        if arguments.profile is None or arguments.stage is None:
            parser.error("audit-retry requires --profile and --stage")
        result = submit_retry_audit(
            spec, arguments.profile, arguments.stage, execute=arguments.execute
        )
    elif arguments.command == "finalize":
        if arguments.profile is None:
            parser.error("finalize requires --profile")
        result = submit_finalize(spec, arguments.profile, execute=arguments.execute)
    elif arguments.command == "freeze-controls":
        if arguments.controls_profile is None:
            parser.error("freeze-controls requires --controls-profile")
        result = submit_controls_freeze(
            spec, arguments.controls_profile, execute=arguments.execute
        )
    elif arguments.command == "controls-stage":
        if arguments.controls_profile is None or arguments.controls_stage is None:
            parser.error(
                "controls-stage requires --controls-profile and --controls-stage"
            )
        result = submit_controls_stage(
            spec,
            arguments.controls_profile,
            arguments.controls_stage,
            execute=arguments.execute,
            retry_map_path=arguments.retry_map,
        )
    elif arguments.command == "audit-controls-retry":
        if arguments.controls_profile is None or arguments.controls_stage is None:
            parser.error(
                "audit-controls-retry requires --controls-profile and --controls-stage"
            )
        result = submit_controls_retry_audit(
            spec,
            arguments.controls_profile,
            arguments.controls_stage,
            execute=arguments.execute,
        )
    elif arguments.command == "finalize-controls":
        if arguments.controls_profile is None:
            parser.error("finalize-controls requires --controls-profile")
        result = submit_controls_finalize(
            spec, arguments.controls_profile, execute=arguments.execute
        )
    elif arguments.command == "freeze-pixel":
        result = submit_pixel_freeze(spec, execute=arguments.execute)
    elif arguments.command == "pixel":
        result = submit_pixel(
            spec,
            execute=arguments.execute,
            retry_map_path=arguments.retry_map,
        )
    else:
        result = submit_pixel_retry_audit(spec, execute=arguments.execute)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("NEURIPS_SLURM_SUBMISSION_EXECUTED" if arguments.execute else "NEURIPS_SLURM_DRY_RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
