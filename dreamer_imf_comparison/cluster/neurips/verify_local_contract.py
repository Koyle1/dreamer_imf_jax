#!/usr/bin/env python3
"""Verify the local, pre-result Slurm contract without submitting or training.

These checks prove orchestration structure and fail-closed negative controls.
They deliberately do not claim that any cluster result exists.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest import mock


CLUSTER_DIR = Path(__file__).resolve().parent
PROJECT = CLUSTER_DIR.parents[1]
WORKSPACE = PROJECT.parent
for path in (CLUSTER_DIR, PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gpu_preflight
import feasibility
import submit
import supplementary
import verify_cluster_evidence
import workflow


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _controls_counts(spec: dict) -> dict[str, dict[str, int]]:
    from dreamer_imf_compare import neurips_controls

    protocol = neurips_controls.read_controls_protocol(
        PROJECT / "neurips_controls_protocol.json"
    )
    result = {}
    for profile in supplementary.CONTROLS_PROFILES:
        row = protocol["profiles"][profile]
        expected = spec["supplementary"]["controls"][profile][
            "expected_stage_counts"
        ]
        divisor = (
            len(row["tasks"])
            * len(row["world_model_seeds"])
            * len(row["run_budget_tracks"])
        )
        _require(expected["world"] % divisor == 0, "controls run count is fractional")
        runs = expected["world"] // divisor
        derived = {
            "dataset": len(row["tasks"]) * len(row["world_model_seeds"]),
            "compute": len(row["tasks"]),
            "world": divisor * runs,
            "rollout": divisor * runs,
            "actor": divisor
            * runs
            * len(row["actor_seeds_nested_within_world_model_seed"]),
        }
        _require(derived == expected, f"controls {profile} stage counts do not rederive")
        _require(
            sum(derived.values())
            == spec["supplementary"]["controls"][profile][
                "expected_execution_units"
            ],
            f"controls {profile} execution-unit count does not rederive",
        )
        result[profile] = derived
    return result


def _verify_script_set(spec: dict) -> list[str]:
    scripts = verify_cluster_evidence._verify_sbatch_contract(spec)
    required = {
        "controls_freeze.sbatch",
        "controls_array.sbatch",
        "controls_stage_verify.sbatch",
        "controls_retry_audit.sbatch",
        "controls_finalize.sbatch",
        "pixel_freeze.sbatch",
        "pixel_array.sbatch",
        "pixel_retry_audit.sbatch",
        "pixel_finalize.sbatch",
    }
    _require(required <= set(scripts), "supplementary Slurm script set is incomplete")
    return scripts


def verify_controls_contract(spec: dict) -> dict:
    counts = _controls_counts(spec)
    scripts = _verify_script_set(spec)
    plan = workflow.read_json(CLUSTER_DIR / "supplementary_plan.json")
    development_command = plan["controls"]["development"]["freeze_command"]
    _require(
        "--parent-selection" not in development_command
        and "freeze-controls --controls-profile development" in development_command,
        "development controls freeze incorrectly consumes a selection",
    )
    _require(
        spec["supplementary"]["controls"]["stage_array_concurrency"]["compute"]
        == 1,
        "controls compute cells are not serialized",
    )
    _require(
        supplementary.controls_output(spec, "development")
        not in supplementary.controls_state(spec, "development").parents
        and supplementary.controls_state(spec, "development")
        not in supplementary.controls_output(spec, "development").parents,
        "controls cluster metadata overlaps a canonical result root",
    )
    source = inspect.getsource(supplementary.freeze_controls_profile)
    _require(
        source.count("--parent-selection") == 1
        and 'if profile == "confirmatory"' in source,
        "controls selection boundary is not encoded in the freezer",
    )
    stage_source = inspect.getsource(supplementary.verify_controls_stage)
    _require(
        "require_controls_unit_verified" in stage_source
        and "unit_verification_sha256" in stage_source
        and "controls_stage_verified" in stage_source,
        "controls whole-stage GPU-receipt validator or ledger event is absent",
    )
    retry_source = inspect.getsource(supplementary.audit_controls_retry)
    _require(
        "strong_validation=True" in retry_source
        and "selected_state" not in retry_source,
        "controls retry audit does not use strong validation",
    )
    # The selected-state binding is created centrally and then revalidated at
    # submission; check both sides rather than trusting prose.
    _require(
        "selected_state_sha256" in inspect.getsource(supplementary._build_retry_map)
        and "current_states" in inspect.getsource(
            supplementary.validate_controls_retry_for_submission
        ),
        "controls retry map lacks a stale-state binding",
    )
    main_audit = inspect.getsource(workflow.audit_retry_map)
    main_submit = inspect.getsource(submit.submit_stage)
    main_worker = inspect.getsource(workflow.run_array_cell)
    _require(
        "selected_state_sha256" in main_audit
        and '"actions"' in main_audit
        and "current_states" in main_submit
        and "NEURIPS_RETRY_MAP_SHA256" in main_submit
        and "main retry map digest differs from its Slurm export" in main_worker
        and "quarantine_cluster_node" in main_worker,
        "main retry path lacks byte-state, action, export, worker, or quarantine binding",
    )
    receipt_source = inspect.getsource(supplementary.require_controls_unit_verified)
    _require(
        "worker_runtime_sha256" in receipt_source
        and "_validate_controls_worker_runtime" in receipt_source,
        "controls receipts do not authenticate each worker runtime",
    )
    return {
        "profiles": counts,
        "slurm_scripts": len(scripts),
        "schemas": {
            "map": supplementary.CONTROLS_MAP_SCHEMA,
            "unit": supplementary.CONTROLS_UNIT_SCHEMA,
            "stage": supplementary.CONTROLS_STAGE_SCHEMA,
            "profile": supplementary.CONTROLS_PROFILE_SCHEMA,
        },
    }


def verify_pixel_contract(spec: dict) -> dict:
    from dreamer_imf_compare import pixel_benchmark

    protocol = pixel_benchmark.load_protocol()
    stage_map = supplementary.build_pixel_map(
        spec,
        protocol,
        source_commit="1" * 40,
        source_manifest_sha256="2" * 64,
    )
    supplementary.validate_pixel_map(
        stage_map,
        spec=spec,
        protocol=protocol,
        source_commit="1" * 40,
        source_manifest_sha256="2" * 64,
    )
    identities = [
        (entry["track"], entry["task"], entry["seed"])
        for entry in stage_map["entries"]
    ]
    _require(
        stage_map["count"] == 30
        and len(identities) == len(set(identities)) == 30
        and [entry["array_index"] for entry in stage_map["entries"]]
        == list(range(30)),
        "pixel map is not an exact unique 30-unit array",
    )
    expected_paths = {
        f"{track}/{task}/seed_{seed}"
        for track in spec["supplementary"]["pixel"]["tracks"]
        for task in spec["supplementary"]["pixel"]["tasks"]
        for seed in spec["supplementary"]["pixel"]["seeds"]
    }
    _require(
        {entry["relative_output"] for entry in stage_map["entries"]}
        == expected_paths,
        "pixel map paths differ from the registered Cartesian product",
    )
    run_source = inspect.getsource(supplementary.run_pixel_array_cell)
    _require(
        "recompute_predictions=True" in run_source
        and "rederive_compute=True" in run_source
        and "pixel_unit_gpu_verified" in run_source,
        "pixel cell does not perform and ledger-bind both GPU-side checks",
    )
    final_source = inspect.getsource(supplementary.finalize_pixel_profile)
    verifier_source = inspect.getsource(supplementary.require_pixel_profile_verified)
    _require(
        "len(receipts) != 30" in final_source
        and "aggregate_artifacts" in final_source
        and "verify_aggregate" in final_source
        and "pixel_profile_verified" in final_source
        and "pixel.verify_aggregate(aggregate_path, roots)" in verifier_source,
        "pixel finalization does not require, aggregate, and authenticate all units",
    )
    _require(
        supplementary.pixel_output(spec)
        not in supplementary.pixel_state(spec).parents
        and supplementary.pixel_state(spec)
        not in supplementary.pixel_output(spec).parents,
        "pixel cluster metadata overlaps the canonical artifact root",
    )
    _verify_script_set(spec)
    return {
        "artifacts": stage_map["count"],
        "tracks": sorted({entry["track"] for entry in stage_map["entries"]}),
        "map_sha256": stage_map["map_sha256"],
        "receipt_schema": supplementary.PIXEL_UNIT_SCHEMA,
        "profile_schema": supplementary.PIXEL_PROFILE_SCHEMA,
    }


def _assert_raises(exception: type[BaseException], function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except exception:
        return
    except Exception as error:  # pragma: no cover - decisive diagnostic
        raise RuntimeError(
            f"negative control raised {type(error).__name__}, expected {exception.__name__}"
        ) from error
    raise RuntimeError(f"negative control did not raise {exception.__name__}")


def verify_negative_controls(spec: dict) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _assert_raises(
            RuntimeError,
            supplementary.require_authenticated_main_profile,
            spec,
            "pilot",
            cluster_root=root,
        )
        _assert_raises(
            RuntimeError,
            supplementary.require_controls_profile_verified,
            spec,
            "development",
            checkout=WORKSPACE,
            cluster_root=root,
        )
        _assert_raises(
            RuntimeError,
            supplementary.require_pixel_profile_verified,
            spec,
            checkout=WORKSPACE,
            cluster_root=root,
        )
        dry_spec = copy.deepcopy(spec)
        absent_workspace = root / "dry-run-must-not-create"
        dry_spec["workspace_root"] = str(absent_workspace)
        preflight = submit.submit_preflight(dry_spec, execute=False)
        _require(
            preflight["job_id"] is None and not absent_workspace.exists(),
            "preflight dry run mutated the workspace",
        )
        before = tuple(root.rglob("*"))
        with mock.patch.object(
            supplementary, "_controls_prerequisites", return_value={}
        ), mock.patch.object(
            feasibility,
            "require_post_pilot_feasible",
            return_value={"feasibility_sha256": "a" * 64},
        ):
            submit.submit_controls_freeze(spec, "development", execute=False)
        with mock.patch.object(
            supplementary, "require_authenticated_main_profile", return_value={}
        ), mock.patch.object(
            feasibility,
            "require_post_pilot_feasible",
            return_value={"feasibility_sha256": "a" * 64},
        ):
            submit.submit_pixel_freeze(spec, execute=False)
        with mock.patch.object(
            supplementary, "require_controls_freeze", return_value={}
        ), mock.patch.object(
            supplementary, "_require_previous_controls_stage"
        ), mock.patch.object(
            supplementary, "incomplete_controls_indices", return_value=[0, 1]
        ):
            controls_plan = submit.submit_controls_stage(
                spec, "development", "compute", execute=False
            )
        with mock.patch.object(
            supplementary, "require_pixel_freeze", return_value={}
        ), mock.patch.object(
            supplementary, "incomplete_pixel_indices", return_value=list(range(30))
        ):
            pixel_plan = submit.submit_pixel(spec, execute=False)
        _require(before == tuple(root.rglob("*")), "supplementary dry run wrote files")
        _require(
            "--array=0-1%1" in controls_plan["array_command"]
            and "--array=0-29%4" in pixel_plan["array_command"],
            "dry-run array expressions differ from registered concurrency",
        )

    _assert_raises(
        ValueError,
        submit.build_sbatch_command,
        CLUSTER_DIR / "controls_array.sbatch",
        array="0-1%5",
    )
    command = submit.build_sbatch_command(
        CLUSTER_DIR / "controls_stage_verify.sbatch",
        dependency_job_id="123",
    )
    _require(
        "--dependency=afterok:123" in command
        and not any("aftercorr" in token.lower() for token in command),
        "whole-stage verifier does not use exact afterok dependency",
    )
    base_map = {
        "map_sha256": "a" * 64,
        "count": 2,
        "entries": [
            {"array_index": 0, "cell_id": "pixel-" + "1" * 24},
            {"array_index": 1, "cell_id": "pixel-" + "2" * 24},
        ],
    }
    retry = supplementary._build_retry_map(
        kind="pixel",
        profile="hard",
        stage="artifact",
        base_map=base_map,
        indices=[0],
        states=["3" * 64],
        audit_id="audit1",
        slurm_job_id="456",
    )
    supplementary.validate_supplementary_retry_map(
        retry,
        kind="pixel",
        profile="hard",
        stage="artifact",
        base_map=base_map,
        current_states=["3" * 64],
    )
    _assert_raises(
        ValueError,
        supplementary.validate_supplementary_retry_map,
        retry,
        kind="pixel",
        profile="hard",
        stage="artifact",
        base_map=base_map,
        current_states=["4" * 64],
    )
    versions = {
        name: spec["runtime"][name]
        for name in ("python", "jax", "jaxlib", "numpy", "dm-control", "mujoco")
    }
    preflight_runtime = {
        "versions": versions,
        "environment": spec["runtime"]["environment"],
        "jax_runtime": {
            "backend": "gpu",
            "visible_device_count": 1,
            "device_platforms": ["gpu"],
            "device_kinds": ["NVIDIA L40S"],
        },
        "slurm": {"cuda_visible_devices": "0"},
    }
    worker_runtime = {
        **versions,
        "backend": "cpu",
        "visible_device_count": 1,
        "device_platforms": ["cpu"],
        "device_kinds": ["CPU"],
        "cuda_visible_devices": "0",
        "environment": spec["runtime"]["environment"],
    }
    _assert_raises(
        ValueError,
        supplementary._validate_controls_worker_runtime,
        worker_runtime,
        spec=spec,
        preflight=preflight_runtime,
    )
    _assert_raises(
        RuntimeError,
        feasibility.assess_post_pilot_feasibility,
        spec,
        measured_gpu_seconds=1000.0,
        measured_artifact_bytes=1_000_000,
        measured_artifact_files=1000,
        measured_result_updates=4_320_000,
        workspace_record={"remaining_seconds_at_observation": 604801},
        quota_record={
            "remaining_quota_kib": 10_000_000_000,
            "remaining_inodes": 10_000_000,
        },
    )
    # The parsers must reject plausible-looking corruption, proving these are
    # genuine negative controls rather than mere absence tests.
    ws = (
        "id: dreamer_imf_neurips\n"
        " workspace directory: /work2/ci72buri-dreamer_imf_neurips\n"
        " remaining time: 29 days 21 hours\n"
        " comment: Trajectory iMF NeurIPS matched-compute benchmark\n"
        " creation time: Fri Sep 11 02:05:17 2026\n"
        " expiration date: Sun Oct 11 02:05:17 2026\n"
        " filesystem name: work_new\n"
        " available extensions: 6\n"
    )
    parsed = gpu_preflight.parse_ws_list(ws, workspace_id="dreamer_imf_neurips")
    _require(parsed["remaining_days"] == 29, "positive ws_list parser control failed")
    _assert_raises(
        ValueError,
        gpu_preflight.parse_ws_list,
        ws.replace("29 days 21 hours", "never"),
        workspace_id="dreamer_imf_neurips",
    )
    return {
        "missing_prerequisite_boundaries": 3,
        "dry_run_planners": 5,
        "retry_tamper_controls": 1,
        "runtime_and_feasibility_controls": 2,
        "dependency": "afterok:123",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--controls", action="store_true")
    modes.add_argument("--pixel", action="store_true")
    modes.add_argument("--negative-controls", action="store_true")
    arguments = parser.parse_args()
    spec = workflow.load_spec()
    if arguments.controls:
        result = verify_controls_contract(spec)
        token = "NEURIPS_CLUSTER_CONTROLS_CONTRACT_VERIFIED"
    elif arguments.pixel:
        result = verify_pixel_contract(spec)
        token = "NEURIPS_CLUSTER_PIXEL_CONTRACT_VERIFIED"
    else:
        result = verify_negative_controls(spec)
        token = "NEURIPS_CLUSTER_NEGATIVE_CONTROLS_VERIFIED"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
