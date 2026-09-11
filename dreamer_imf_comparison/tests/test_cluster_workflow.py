from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
CLUSTER = PROJECT / "cluster" / "neurips"
for path in (
    CLUSTER,
    PROJECT,
    WORKSPACE / "imf_dreamer_jax" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import submit
import supplementary
import verify_cluster_evidence
import workflow
import gpu_preflight
import feasibility


class ClusterWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = workflow.load_spec()

    @staticmethod
    def _cell(stage: str, suffix: str, dependencies=()):
        return {
            "stage": stage,
            "cell_id": f"{stage}-{suffix}",
            "identity_sha256": suffix.rjust(64, "0"),
            "dependencies": list(dependencies),
        }

    def test_registered_main_counts_rederive_from_frozen_protocol(self) -> None:
        from dreamer_imf_compare.artifacts import read_json
        from dreamer_imf_compare.matched_objective_benchmark import (
            expected_hpo_matrix_counts,
            expected_matrix_counts,
        )

        protocol = read_json(PROJECT / "matched_objective_protocol.json")
        pilot = expected_hpo_matrix_counts(protocol)
        confirmatory = expected_matrix_counts(protocol, "confirmatory")
        self.assertEqual(
            pilot,
            self.spec["matched_objective"]["pilot"]["expected_stage_counts"],
        )
        self.assertEqual(sum(pilot.values()), 606)
        self.assertEqual(
            confirmatory,
            self.spec["matched_objective"]["confirmatory"][
                "expected_stage_counts"
            ],
        )
        self.assertEqual(sum(confirmatory.values()), 1746)

    def test_stage_maps_are_deterministic_indexed_and_immutable(self) -> None:
        matrix = {
            "matrix_sha256": "a" * 64,
            "cells": [
                self._cell("dataset", "2"),
                self._cell("dataset", "1"),
            ],
        }
        payload = workflow.build_stage_map(matrix, "pilot", "dataset")
        self.assertEqual(
            [row["array_index"] for row in payload["entries"]], [0, 1]
        )
        self.assertEqual(
            [row["cell_id"] for row in payload["entries"]],
            ["dataset-1", "dataset-2"],
        )
        workflow.validate_stage_map(payload, matrix, "pilot", "dataset")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            workflow.write_json_once(path, payload)
            workflow.write_json_once(path, payload)
            changed = copy.deepcopy(payload)
            changed["count"] = 3
            with self.assertRaises(FileExistsError):
                workflow.write_json_once(path, changed)

    def test_next_stage_rejects_missing_or_changed_whole_stage_evidence(self) -> None:
        dataset = self._cell("dataset", "1")
        compute = self._cell("compute_plan", "2")
        matrix = {
            "matrix_sha256": "b" * 64,
            "cells": [dataset, compute],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            dataset_map = workflow.build_stage_map(matrix, "pilot", "dataset")
            workflow.write_json_once(
                workflow.map_path(output, "dataset"), dataset_map
            )
            with self.assertRaisesRegex(RuntimeError, "entire dataset stage"):
                workflow.require_previous_stage_verified(
                    output, "compute_plan", "pilot", matrix
                )

            result = output / "dataset" / dataset["cell_id"] / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text('{"status":"complete"}\n', encoding="utf-8")
            marker = {
                "schema_version": workflow.STAGE_MARKER_SCHEMA,
                "status": "verified_complete",
                "profile": "pilot",
                "stage": "dataset",
                "matrix_sha256": matrix["matrix_sha256"],
                "map_sha256": dataset_map["map_sha256"],
                "verified_cell_count": 1,
                "result_files": [
                    {
                        "cell_id": dataset["cell_id"],
                        "path": result.relative_to(output).as_posix(),
                        "sha256": workflow.file_sha256(result),
                    }
                ],
                "slurm_job_id": "100",
            }
            marker["stage_verification_sha256"] = workflow.object_sha256(marker)
            workflow.write_json_once(
                workflow.stage_marker_path(output, "dataset"), marker
            )
            workflow.require_previous_stage_verified(
                output, "compute_plan", "pilot", matrix
            )
            result.write_text('{"status":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                workflow.require_previous_stage_verified(
                    output, "compute_plan", "pilot", matrix
                )

    def test_slurm_commands_use_whole_job_afterok_and_never_aftercorr(self) -> None:
        command = submit.build_sbatch_command(
            CLUSTER / "stage_array.sbatch",
            exports={"NEURIPS_PROFILE": "pilot", "NEURIPS_STAGE": "world_model"},
            array="0-143%4",
            dependency_job_id="12345",
        )
        self.assertIn("--dependency=afterok:12345", command)
        self.assertFalse(any("aftercorr" in token.lower() for token in command))
        self.assertIn("--array=0-143%4", command)
        with self.assertRaises(ValueError):
            submit.build_sbatch_command(
                CLUSTER / "stage_array.sbatch",
                array="0-143%8",
            )
        serialized = submit.build_sbatch_command(
            CLUSTER / "controls_array.sbatch", array="0-5%1"
        )
        self.assertIn("--array=0-5%1", serialized)
        for invalid in ("0-143%40", "0-3,4-7%4", "3-1%4", "1,0%4"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                submit.build_sbatch_command(
                    CLUSTER / "stage_array.sbatch", array=invalid
                )
        for path in CLUSTER.glob("*.sbatch"):
            directives = [
                line.lower()
                for line in path.read_text(encoding="utf-8").splitlines()
                if "--dependency" in line
            ]
            self.assertFalse(any("aftercorr" in line for line in directives))

    def test_retry_expression_contains_only_selected_indices(self) -> None:
        self.assertEqual(workflow.format_array_indices([0, 1, 2, 7, 9, 10]), "0-2,7,9-10%4")
        self.assertEqual(workflow.format_array_indices([5, 5]), "5%4")
        with self.assertRaises(ValueError):
            workflow.format_array_indices([])
        with self.assertRaises(ValueError):
            workflow.format_array_indices([True])
        with self.assertRaises(ValueError):
            workflow.format_array_indices([0], concurrency=8)

    def test_stage_result_manifest_rejects_paths_outside_output(self) -> None:
        dataset = self._cell("dataset", "1")
        matrix = {"matrix_sha256": "e" * 64, "cells": [dataset]}
        stage_map = workflow.build_stage_map(matrix, "pilot", "dataset")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            outside = Path(directory) / "outside.json"
            outside.write_text('{"status":"complete"}\n', encoding="utf-8")
            marker = {
                "schema_version": workflow.STAGE_MARKER_SCHEMA,
                "status": "verified_complete",
                "profile": "pilot",
                "stage": "dataset",
                "matrix_sha256": matrix["matrix_sha256"],
                "map_sha256": stage_map["map_sha256"],
                "verified_cell_count": 1,
                "result_files": [
                    {
                        "cell_id": dataset["cell_id"],
                        "path": "../outside.json",
                        "sha256": workflow.file_sha256(outside),
                    }
                ],
                "slurm_job_id": "100",
            }
            marker["stage_verification_sha256"] = workflow.object_sha256(marker)
            with self.assertRaisesRegex(ValueError, "not canonical"):
                workflow.validate_stage_marker(
                    marker,
                    profile="pilot",
                    stage="dataset",
                    matrix=matrix,
                    stage_map=stage_map,
                )

    def test_retry_map_cannot_expand_or_remap_the_audited_subset(self) -> None:
        matrix = {
            "matrix_sha256": "d" * 64,
            "cells": [
                self._cell("world_model", "1"),
                self._cell("world_model", "2"),
                self._cell("world_model", "3"),
            ],
        }
        stage_map = workflow.build_stage_map(matrix, "pilot", "world_model")
        payload = {
            "schema_version": workflow.RETRY_MAP_SCHEMA,
            "status": "strongly_audited_missing_or_invalid_only",
            "profile": "pilot",
            "stage": "world_model",
            "matrix_sha256": matrix["matrix_sha256"],
            "stage_map_sha256": stage_map["map_sha256"],
            "audit_id": "123",
            "indices": [0, 2],
            "cell_ids": [
                stage_map["entries"][0]["cell_id"],
                stage_map["entries"][2]["cell_id"],
            ],
            "selected_state_sha256": ["a" * 64, "b" * 64],
            "actions": ["run_missing", "quarantine_invalid"],
            "slurm_job_id": "99",
        }
        payload["retry_map_sha256"] = workflow.object_sha256(payload)
        workflow.validate_retry_map(
            payload,
            profile="pilot",
            stage="world_model",
            matrix=matrix,
            stage_map=stage_map,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = workflow.retry_map_path(output, "world_model", "123")
            workflow.write_json_once(path, payload)
            ledger = output / "ledger.jsonl"
            workflow.append_ledger(
                ledger,
                {
                    "event": "retry_map_audited",
                    "profile": "pilot",
                    "stage": "world_model",
                    "retry_map": path.relative_to(output).as_posix(),
                    "retry_map_sha256": payload["retry_map_sha256"],
                    "retry_cell_count": 2,
                    "slurm_job_id": "99",
                },
            )
            workflow.require_retry_map_registered(
                path, payload, output_root=output, ledger=ledger
            )
            expanded = copy.deepcopy(payload)
            expanded["audit_id"] = "forged"
            expanded["indices"] = [0, 1, 2]
            expanded["cell_ids"] = [
                entry["cell_id"] for entry in stage_map["entries"]
            ]
            expanded["selected_state_sha256"] = ["a" * 64, "c" * 64, "b" * 64]
            expanded["actions"] = [
                "run_missing",
                "resume_partial",
                "quarantine_invalid",
            ]
            expanded["retry_map_sha256"] = workflow.object_sha256(
                {
                    key: value
                    for key, value in expanded.items()
                    if key != "retry_map_sha256"
                }
            )
            forged_path = workflow.retry_map_path(output, "world_model", "forged")
            workflow.write_json_once(forged_path, expanded)
            workflow.validate_retry_map(
                expanded,
                profile="pilot",
                stage="world_model",
                matrix=matrix,
                stage_map=stage_map,
            )
            with self.assertRaisesRegex(ValueError, "audit ledger"):
                workflow.require_retry_map_registered(
                    forged_path, expanded, output_root=output, ledger=ledger
                )

    def test_dependency_lock_is_complete_exact_and_hashed(self) -> None:
        result = verify_cluster_evidence._verify_lock(
            PROJECT / "requirements-neurips-cuda12-lock.txt"
        )
        self.assertEqual(result["packages"], 45)
        self.assertRegex(result["sha256"], r"^[0-9a-f]{64}$")

    def test_supplementary_plan_counts_controls_and_pixel_units(self) -> None:
        plan = workflow.read_json(CLUSTER / "supplementary_plan.json")
        development = plan["controls"]["development"]
        confirmatory = plan["controls"]["confirmatory"]
        self.assertEqual(
            development["matrix_cells"],
            sum(development["matrix_stage_counts"].values()),
        )
        self.assertEqual(
            development["execution_units_including_datasets"],
            development["matrix_cells"] + development["canonical_dataset_units"],
        )
        self.assertEqual(
            confirmatory["matrix_cells"],
            sum(confirmatory["matrix_stage_counts"].values()),
        )
        self.assertEqual(
            confirmatory["execution_units_including_datasets"],
            confirmatory["matrix_cells"] + confirmatory["canonical_dataset_units"],
        )
        pixel = plan["pixel"]
        self.assertEqual(
            len(pixel["tasks"]) * len(pixel["seeds"]) * len(pixel["tracks"]),
            pixel["artifact_units"],
        )
        self.assertEqual(pixel["artifact_units"], 30)

    def test_controls_registered_counts_rederive_from_protocol(self) -> None:
        from dreamer_imf_compare import neurips_controls

        protocol = neurips_controls.read_controls_protocol(
            PROJECT / "neurips_controls_protocol.json"
        )
        for profile in supplementary.CONTROLS_PROFILES:
            row = protocol["profiles"][profile]
            expected = self.spec["supplementary"]["controls"][profile][
                "expected_stage_counts"
            ]
            runs = expected["world"] // (
                len(row["tasks"])
                * len(row["world_model_seeds"])
                * len(row["run_budget_tracks"])
            )
            derived = {
                "dataset": len(row["tasks"]) * len(row["world_model_seeds"]),
                "compute": len(row["tasks"]),
                "world": len(row["tasks"])
                * len(row["world_model_seeds"])
                * len(row["run_budget_tracks"])
                * runs,
                "rollout": len(row["tasks"])
                * len(row["world_model_seeds"])
                * len(row["run_budget_tracks"])
                * runs,
                "actor": len(row["tasks"])
                * len(row["world_model_seeds"])
                * len(row["run_budget_tracks"])
                * runs
                * len(row["actor_seeds_nested_within_world_model_seed"]),
            }
            self.assertEqual(derived, expected)
            self.assertEqual(
                sum(derived.values()),
                self.spec["supplementary"]["controls"][profile][
                    "expected_execution_units"
                ],
            )

    def test_pixel_map_is_deterministic_unique_source_bound_and_exactly_30(self) -> None:
        from dreamer_imf_compare import pixel_benchmark

        protocol = pixel_benchmark.load_protocol()
        kwargs = {
            "source_commit": "a" * 40,
            "source_manifest_sha256": "b" * 64,
        }
        first = supplementary.build_pixel_map(self.spec, protocol, **kwargs)
        second = supplementary.build_pixel_map(self.spec, protocol, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["count"], 30)
        self.assertEqual(
            [entry["array_index"] for entry in first["entries"]], list(range(30))
        )
        identities = {
            (entry["track"], entry["task"], entry["seed"])
            for entry in first["entries"]
        }
        self.assertEqual(len(identities), 30)
        changed = supplementary.build_pixel_map(
            self.spec,
            protocol,
            source_commit="c" * 40,
            source_manifest_sha256="b" * 64,
        )
        self.assertNotEqual(first["map_sha256"], changed["map_sha256"])

    def test_supplementary_retry_map_rejects_expansion_remap_and_staleness(self) -> None:
        base_map = {
            "map_sha256": "a" * 64,
            "count": 3,
            "entries": [
                {"array_index": index, "cell_id": f"pixel-{'a' * 23}{index}"}
                for index in range(3)
            ],
        }
        states = ["1" * 64, "2" * 64]
        payload = supplementary._build_retry_map(
            kind="pixel",
            profile="hard",
            stage="artifact",
            base_map=base_map,
            indices=[0, 2],
            states=states,
            audit_id="123",
            slurm_job_id="456",
        )
        supplementary.validate_supplementary_retry_map(
            payload,
            kind="pixel",
            profile="hard",
            stage="artifact",
            base_map=base_map,
            current_states=states,
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            supplementary.validate_supplementary_retry_map(
                payload,
                kind="pixel",
                profile="hard",
                stage="artifact",
                base_map=base_map,
                current_states=["3" * 64, "2" * 64],
            )
        for key, value in (
            ("indices", [0, 1, 2]),
            ("cell_ids", ["forged", payload["cell_ids"][1]]),
        ):
            forged = copy.deepcopy(payload)
            forged[key] = value
            forged["retry_map_sha256"] = workflow.object_sha256(
                {
                    name: item
                    for name, item in forged.items()
                    if name != "retry_map_sha256"
                }
            )
            with self.subTest(key=key), self.assertRaises(ValueError):
                supplementary.validate_supplementary_retry_map(
                    forged,
                    kind="pixel",
                    profile="hard",
                    stage="artifact",
                    base_map=base_map,
                )

    def test_supplementary_cluster_metadata_never_enters_result_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in supplementary.CONTROLS_PROFILES:
                output = supplementary.controls_output(
                    self.spec, profile, cluster_root=root
                )
                for stage in supplementary.CONTROLS_STAGES:
                    self.assertNotIn(
                        output,
                        supplementary.controls_map_path(
                            self.spec,
                            profile,
                            stage,
                            cluster_root=root,
                        ).parents,
                    )
                self.assertNotIn(
                    output,
                    supplementary.controls_profile_marker_path(
                        self.spec, profile, cluster_root=root
                    ).parents,
                )
            pixel_output = supplementary.pixel_output(
                self.spec, cluster_root=root
            )
            self.assertNotIn(
                pixel_output,
                supplementary.pixel_map_path(
                    self.spec, cluster_root=root
                ).parents,
            )

    def test_preflight_dry_run_is_side_effect_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "must-not-be-created"
            spec = copy.deepcopy(self.spec)
            spec["workspace_root"] = str(workspace)
            result = submit.submit_preflight(spec, execute=False)
            self.assertFalse(result["executed"])
            self.assertIsNone(result["job_id"])
            self.assertFalse(workspace.exists())

    def test_all_supplementary_dry_run_planners_do_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = sorted(root.rglob("*"))
            with mock.patch.object(
                supplementary, "_controls_prerequisites", return_value={}
            ), mock.patch.object(
                feasibility,
                "require_post_pilot_feasible",
                return_value={"feasibility_sha256": "a" * 64},
            ):
                submit.submit_controls_freeze(
                    self.spec, "development", execute=False
                )
            with mock.patch.object(
                supplementary, "require_authenticated_main_profile", return_value={}
            ), mock.patch.object(
                feasibility,
                "require_post_pilot_feasible",
                return_value={"feasibility_sha256": "a" * 64},
            ):
                submit.submit_pixel_freeze(self.spec, execute=False)
            common = [
                mock.patch.object(
                    supplementary, "require_controls_freeze", return_value={}
                ),
                mock.patch.object(
                    supplementary, "_require_previous_controls_stage"
                ),
                mock.patch.object(
                    supplementary, "incomplete_controls_indices", return_value=[0, 1]
                ),
            ]
            with common[0], common[1], common[2]:
                plan = submit.submit_controls_stage(
                    self.spec,
                    "development",
                    "compute",
                    execute=False,
                )
            self.assertIn("--array=0-1%1", plan["array_command"])
            with mock.patch.object(
                supplementary, "require_pixel_freeze", return_value={}
            ), mock.patch.object(
                supplementary, "incomplete_pixel_indices", return_value=list(range(30))
            ):
                pixel_plan = submit.submit_pixel(self.spec, execute=False)
            self.assertIn("--array=0-29%4", pixel_plan["array_command"])
            self.assertEqual(before, sorted(root.rglob("*")))

    def test_executed_supplementary_verifiers_depend_on_whole_array(self) -> None:
        with mock.patch.object(
            supplementary, "require_controls_freeze", return_value={}
        ), mock.patch.object(
            supplementary, "_require_previous_controls_stage"
        ), mock.patch.object(
            supplementary, "incomplete_controls_indices", return_value=[0, 1]
        ), mock.patch.object(
            submit, "_submit", side_effect=["101", "102"]
        ), mock.patch.object(submit, "_record_submission"):
            controls = submit.submit_controls_stage(
                self.spec, "development", "compute", execute=True
            )
        self.assertEqual(controls["whole_stage_dependency"], "afterok:101")
        self.assertIn("--dependency=afterok:101", controls["verify_command"])
        with mock.patch.object(
            supplementary, "require_pixel_freeze", return_value={}
        ), mock.patch.object(
            supplementary, "incomplete_pixel_indices", return_value=[0, 1]
        ), mock.patch.object(
            submit, "_submit", side_effect=["201", "202"]
        ), mock.patch.object(submit, "_record_submission"):
            pixel = submit.submit_pixel(self.spec, execute=True)
        self.assertEqual(pixel["whole_stage_dependency"], "afterok:201")
        self.assertIn("--dependency=afterok:201", pixel["finalize_command"])

    def test_missing_prerequisite_markers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "pilot profile is not verified"):
                supplementary.require_authenticated_main_profile(
                    self.spec, "pilot", cluster_root=root
                )
            with self.assertRaisesRegex(RuntimeError, "controls development profile"):
                supplementary.require_controls_profile_verified(
                    self.spec,
                    "development",
                    checkout=WORKSPACE,
                    cluster_root=root,
                )
            with self.assertRaisesRegex(RuntimeError, "hard pixel profile"):
                supplementary.require_pixel_profile_verified(
                    self.spec, checkout=WORKSPACE, cluster_root=root
                )

    def test_authoritative_workspace_and_lustre_quota_parsers(self) -> None:
        ws_list = (
            "id: dreamer_imf_neurips\n"
            "     workspace directory  : /work2/ci72buri-dreamer_imf_neurips\n"
            "     remaining time       : 29 days 21 hours\n"
            "     comment              : Trajectory iMF NeurIPS matched-compute benchmark\n"
            "     creation time        : Fri Sep 11 02:05:17 2026\n"
            "     expiration date      : Sun Oct 11 02:05:17 2026\n"
            "     filesystem name      : work_new\n"
            "     available extensions : 6\n"
        )
        parsed = gpu_preflight.parse_ws_list(
            ws_list, workspace_id="dreamer_imf_neurips"
        )
        self.assertEqual(parsed["remaining_days"], 29)
        self.assertEqual(parsed["available_extensions"], 6)
        quota = (
            "Disk quotas for usr ci72buri (uid 2083):\n"
            "     Filesystem  kbytes  bquota  blimit  bgrace   files  iquota  ilimit  igrace \n"
            "         /work2 2230585692       0 5368709120       - 1796132       0 10000000       - \n"
        )
        parsed_quota = gpu_preflight.parse_lfs_quota(
            quota, filesystem="/work2", user="ci72buri"
        )
        self.assertEqual(parsed_quota["blimit"], 5368709120)
        self.assertEqual(parsed_quota["ilimit"], 10000000)
        with self.assertRaises(ValueError):
            gpu_preflight.parse_ws_list(
                ws_list.replace("29 days 21 hours", "forever"),
                workspace_id="dreamer_imf_neurips",
            )
        with self.assertRaises(ValueError):
            gpu_preflight.parse_lfs_quota(
                quota.replace("5368709120", "unlimited"),
                filesystem="/work2",
                user="ci72buri",
            )

    def test_preflight_validator_binds_authoritative_lifetime_and_quota(self) -> None:
        observed = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        expiration = observed + dt.timedelta(days=10)
        creation = observed - dt.timedelta(days=20)
        contract = {
            "cluster_spec_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "source_commit": "c" * 40,
        }
        payload = {
            "schema_version": workflow.PREFLIGHT_SCHEMA,
            "status": "verified_gpu_worker",
            **contract,
            "host": "worker",
            "platform": "linux",
            "slurm": {
                "job_id": "123",
                "job_account": "dep_inin_dat",
                "job_partition": "gpu-l40s",
                "cpus_per_task": "8",
                "cuda_visible_devices": "0",
            },
            "versions": {
                name: self.spec["runtime"][name]
                for name in ("python", "jax", "jaxlib", "numpy", "dm-control", "mujoco")
            },
            "environment": self.spec["runtime"]["environment"],
            "jax_runtime": {
                "backend": "gpu",
                "visible_device_count": 1,
                "device_kinds": ["NVIDIA L40S"],
                "device_platforms": ["gpu"],
                "jax_enable_x64": False,
                "tiny_device_check": "passed",
            },
            "dmc_render": {
                "task": "dmc_cartpole_swingup",
                "shape": [16, 16, 3],
                "dtype": "uint8",
                "finite": True,
                "sha256": "d" * 64,
            },
            "workspace_storage": {
                "root": self.spec["workspace_root"],
                "measurement": "shutil.disk_usage",
                "total_bytes": 10**13,
                "free_bytes": 10**12,
                "owner_uid": 1000,
                "effective_uid": 1000,
                "owned_by_effective_user": True,
                "read_write_search_access": True,
                "claim_scope": "filesystem_capacity_not_user_quota",
            },
            "workspace_allocator": {
                "interface": "ws_find_plus_ws_list",
                "id": "dreamer_imf_neurips",
                "workspace_directory": self.spec["workspace_root"],
                "remaining_days": 10,
                "remaining_hours": 0,
                "comment": self.spec["workspace_allocator"]["comment"],
                "creation_time": creation.strftime("%a %b %d %H:%M:%S %Y"),
                "expiration_date": expiration.strftime("%a %b %d %H:%M:%S %Y"),
                "filesystem_name": "work_new",
                "available_extensions": 6,
                "timezone": "Europe/Berlin",
                "creation_utc": creation.isoformat(),
                "expiration_utc": expiration.isoformat(),
                "observed_at_utc": observed.isoformat(),
                "remaining_seconds_at_observation": 10 * 24 * 3600,
                "ws_find_stdout_sha256": "e" * 64,
                "ws_list_stdout_sha256": "f" * 64,
            },
            "workspace_quota": {
                "interface": "lfs_quota_user",
                "user": "ci72buri",
                "kbytes": 2230585692,
                "bquota": 0,
                "blimit": 5368709120,
                "files": 1796132,
                "iquota": 0,
                "ilimit": 10000000,
                "filesystem": "/work2",
                "bgrace": "-",
                "igrace": "-",
                "remaining_quota_kib": 3138123428,
                "remaining_inodes": 8203868,
                "stdout_sha256": "1" * 64,
            },
        }
        payload["preflight_sha256"] = workflow.object_sha256(payload)
        workflow.validate_preflight_record(
            payload, spec=self.spec, contract=contract
        )
        forged = copy.deepcopy(payload)
        forged["workspace_quota"]["remaining_quota_kib"] += 1
        forged["preflight_sha256"] = workflow.object_sha256(
            {
                key: value
                for key, value in forged.items()
                if key != "preflight_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "quota"):
            workflow.validate_preflight_record(
                forged, spec=self.spec, contract=contract
            )

    def test_main_profile_marker_without_ledger_authentication_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = workflow.profile_output(
                self.spec, "pilot", cluster_root=root
            )
            output.mkdir(parents=True)
            matrix = {
                "profile": "pilot",
                "matrix_sha256": "a" * 64,
                "cells": [{"cell_id": "one"}],
            }
            workflow.write_json_once(output / "matrix.json", matrix)
            (output / "hpo_trials.json").write_text("trials\n", encoding="utf-8")
            (output / "hpo_selection.json").write_text("selection\n", encoding="utf-8")
            marker = {
                "schema_version": workflow.PROFILE_MARKER_SCHEMA,
                "status": "verified_complete",
                "profile": "pilot",
                "matrix_sha256": matrix["matrix_sha256"],
                "expected_cell_count": 1,
                "result_files": workflow._profile_result_files(output, "pilot"),
                "slurm_job_id": "123",
            }
            marker["profile_verification_sha256"] = workflow.object_sha256(marker)
            workflow.write_json_once(workflow.profile_marker_path(output), marker)
            with self.assertRaises(FileNotFoundError):
                workflow.require_profile_ledger_authenticated(
                    self.spec, "pilot", cluster_root=root
                )

    def test_retry_quarantine_is_recoverable_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad" / "result.json"
            source.parent.mkdir()
            source.write_text("invalid\n", encoding="utf-8")
            destination = root / "quarantine" / "result.json"
            self.assertTrue(
                supplementary._quarantine_move(
                    self.spec, source, destination, cluster_root=root
                )
            )
            self.assertFalse(source.exists())
            self.assertEqual(destination.read_text(encoding="utf-8"), "invalid\n")
            source.parent.mkdir(exist_ok=True)
            source.write_text("second\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                supplementary._quarantine_move(
                    self.spec, source, destination, cluster_root=root
                )

    def test_artifact_inventory_rejects_symlinked_roots_and_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            unit = root / "unit"
            unit.mkdir(parents=True)
            (unit / "result.json").write_text("{}\n", encoding="utf-8")
            alias = root / "alias"
            outside = Path(directory) / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            try:
                os.symlink(unit, alias)
            except (OSError, NotImplementedError) as error:  # pragma: no cover
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "symlink"):
                supplementary._inventory(root, alias)
            alias.unlink()
            os.symlink(outside, unit / "escaped.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                supplementary._inventory(root, unit)

    def test_immutable_json_and_ledger_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text('{"ok": true}\n', encoding="utf-8")
            alias = root / "alias.json"
            try:
                os.symlink(target, alias)
            except (OSError, NotImplementedError) as error:  # pragma: no cover
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "symlink"):
                workflow.write_json_once(alias, {"ok": True})

            ledger = root / "ledger.jsonl"
            workflow.append_ledger(ledger, {"event": "one"})
            ledger_alias = root / "ledger-alias.jsonl"
            os.symlink(ledger, ledger_alias)
            with self.assertRaisesRegex(ValueError, "symlink"):
                workflow.verify_ledger(ledger_alias)
            with self.assertRaisesRegex(ValueError, "symlink"):
                workflow.append_ledger(ledger_alias, {"event": "two"})

    def test_retry_state_is_rechecked_at_array_execution(self) -> None:
        retry = {"indices": [2, 5], "selected_state_sha256": ["a" * 64, "b" * 64]}
        supplementary._require_selected_retry_state(retry, 5, "b" * 64)
        with self.assertRaisesRegex(ValueError, "stale"):
            supplementary._require_selected_retry_state(retry, 5, "c" * 64)

    def test_main_retry_map_rejects_changed_selected_byte_state(self) -> None:
        matrix = {
            "matrix_sha256": "d" * 64,
            "cells": [self._cell("world_model", "1")],
        }
        stage_map = workflow.build_stage_map(matrix, "pilot", "world_model")
        payload = {
            "schema_version": workflow.RETRY_MAP_SCHEMA,
            "status": "strongly_audited_missing_or_invalid_only",
            "profile": "pilot",
            "stage": "world_model",
            "matrix_sha256": matrix["matrix_sha256"],
            "stage_map_sha256": stage_map["map_sha256"],
            "audit_id": "101",
            "indices": [0],
            "cell_ids": [stage_map["entries"][0]["cell_id"]],
            "selected_state_sha256": ["a" * 64],
            "actions": ["quarantine_invalid"],
            "slurm_job_id": "101",
        }
        payload["retry_map_sha256"] = workflow.object_sha256(payload)
        with self.assertRaisesRegex(ValueError, "stale"):
            workflow.validate_retry_map(
                payload,
                profile="pilot",
                stage="world_model",
                matrix=matrix,
                stage_map=stage_map,
                current_states=["b" * 64],
                current_actions=["quarantine_invalid"],
            )

    def test_main_retry_submission_exports_digest_and_rechecks_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            spec = copy.deepcopy(self.spec)
            spec.update(
                workspace_root=str(root),
                source_root=str(root / "source"),
                environment_root=str(root / "venv"),
                state_root=str(root / "cluster_state"),
                results_root=str(root / "results"),
            )
            (root / "source").mkdir(parents=True)
            output = workflow.profile_output(spec, "pilot")
            output.mkdir(parents=True)
            cell = self._cell("world_model", "a" * 24)
            matrix = {"matrix_sha256": "d" * 64, "cells": [cell]}
            stage_map = workflow.build_stage_map(matrix, "pilot", "world_model")
            workflow.write_json_once(
                workflow.map_path(output, "world_model"), stage_map
            )
            retry_path = workflow.retry_map_path(output, "world_model", "101")
            retry = {
                "schema_version": workflow.RETRY_MAP_SCHEMA,
                "status": "strongly_audited_missing_or_invalid_only",
                "profile": "pilot",
                "stage": "world_model",
                "matrix_sha256": matrix["matrix_sha256"],
                "stage_map_sha256": stage_map["map_sha256"],
                "audit_id": "101",
                "indices": [0],
                "cell_ids": [cell["cell_id"]],
                "selected_state_sha256": [
                    workflow.main_cell_state_sha256(spec, output, cell)
                ],
                "actions": ["run_missing"],
                "slurm_job_id": "101",
            }
            retry["retry_map_sha256"] = workflow.object_sha256(retry)
            workflow.write_json_once(retry_path, retry)
            workflow.append_ledger(
                workflow.ledger_path(spec),
                {
                    "event": "retry_map_audited",
                    "profile": "pilot",
                    "stage": "world_model",
                    "retry_map": retry_path.relative_to(output).as_posix(),
                    "retry_map_sha256": retry["retry_map_sha256"],
                    "retry_cell_count": 1,
                    "slurm_job_id": "101",
                },
            )
            with mock.patch.object(
                workflow, "load_validated_main_run", return_value=({}, {}, matrix)
            ), mock.patch.object(workflow, "require_previous_stage_verified"):
                plan = submit.submit_stage(
                    spec,
                    "pilot",
                    "world_model",
                    execute=False,
                    retry_map_path=retry_path,
                )
            export = next(
                token for token in plan["array_command"] if token.startswith("--export=")
            )
            self.assertIn(
                f"NEURIPS_RETRY_MAP_SHA256={retry['retry_map_sha256']}", export
            )
            worker_environment = {
                "SLURM_ARRAY_JOB_ID": "202",
                "SLURM_ARRAY_TASK_ID": "0",
                "NEURIPS_RETRY_MAP_SHA256": retry["retry_map_sha256"],
            }
            with mock.patch.object(
                workflow, "load_validated_main_run", return_value=({}, {}, matrix)
            ), mock.patch.object(
                workflow, "require_previous_stage_verified"
            ), mock.patch.object(
                workflow.subprocess, "run"
            ), mock.patch.dict(
                os.environ, worker_environment, clear=False
            ):
                completed = workflow.run_array_cell(
                    "pilot",
                    "world_model",
                    0,
                    spec=spec,
                    checkout=root / "source",
                    retry_map_path=retry_path,
                )
            self.assertEqual(completed, cell["cell_id"])
            worker_events = workflow.verify_ledger(workflow.ledger_path(spec))
            bound = [
                row
                for row in worker_events
                if row.get("event") in {"cell_started", "cell_completed"}
            ]
            self.assertEqual(len(bound), 2)
            self.assertTrue(
                all(
                    row["retry_map_sha256"] == retry["retry_map_sha256"]
                    for row in bound
                )
            )
            bad_environment = dict(worker_environment)
            bad_environment["NEURIPS_RETRY_MAP_SHA256"] = "0" * 64
            with mock.patch.object(
                workflow, "load_validated_main_run", return_value=({}, {}, matrix)
            ), mock.patch.object(
                workflow, "require_previous_stage_verified"
            ), mock.patch.dict(
                os.environ, bad_environment, clear=False
            ):
                with self.assertRaisesRegex(ValueError, "Slurm export"):
                    workflow.run_array_cell(
                        "pilot",
                        "world_model",
                        0,
                        spec=spec,
                        checkout=root / "source",
                        retry_map_path=retry_path,
                    )

            artifact = workflow.main_cell_artifact_path(spec, output, cell)
            artifact.mkdir(parents=True)
            (artifact / "partial.bin").write_bytes(b"changed")
            with mock.patch.object(
                workflow, "load_validated_main_run", return_value=({}, {}, matrix)
            ), mock.patch.object(workflow, "require_previous_stage_verified"):
                with self.assertRaisesRegex(ValueError, "stale"):
                    submit.submit_stage(
                        spec,
                        "pilot",
                        "world_model",
                        execute=False,
                        retry_map_path=retry_path,
                    )

    def test_array_submission_rejects_duplicate_initial_inflight_and_retry_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retry_path = root / "retry.json"
            retry = {
                "cell_ids": ["world_model-" + "a" * 24],
                "retry_map_sha256": "b" * 64,
            }
            workflow.write_json_once(retry_path, retry)

            initial_ledger = root / "initial.jsonl"
            workflow.append_ledger(
                initial_ledger,
                {
                    "event": "job_submitted",
                    "kind": "stage_array",
                    "profile": "pilot",
                    "stage": "world_model",
                    "command": ["sbatch"],
                },
            )
            with mock.patch.object(workflow, "ledger_path", return_value=initial_ledger):
                with self.assertRaisesRegex(RuntimeError, "initial array"):
                    submit._reject_duplicate_supplementary_array(
                        self.spec,
                        kind="stage_array",
                        profile="pilot",
                        stage="world_model",
                        retry_map_path=None,
                    )

            active_ledger = root / "active.jsonl"
            workflow.append_ledger(
                active_ledger,
                {
                    "event": "cell_started",
                    "profile": "pilot",
                    "stage": "world_model",
                    "cell_id": retry["cell_ids"][0],
                },
            )
            with mock.patch.object(workflow, "ledger_path", return_value=active_ledger):
                with self.assertRaisesRegex(RuntimeError, "in flight"):
                    submit._reject_duplicate_supplementary_array(
                        self.spec,
                        kind="stage_array",
                        profile="pilot",
                        stage="world_model",
                        retry_map_path=retry_path,
                    )

            map_ledger = root / "map.jsonl"
            workflow.append_ledger(
                map_ledger,
                {
                    "event": "job_submitted",
                    "kind": "stage_array",
                    "profile": "pilot",
                    "stage": "world_model",
                    "command": [f"NEURIPS_RETRY_MAP={retry_path.resolve()}"],
                },
            )
            with mock.patch.object(workflow, "ledger_path", return_value=map_ledger):
                with self.assertRaisesRegex(RuntimeError, "already submitted"):
                    submit._reject_duplicate_supplementary_array(
                        self.spec,
                        kind="stage_array",
                        profile="pilot",
                        stage="world_model",
                        retry_map_path=retry_path,
                    )

    def test_lstat_snapshot_and_quarantine_preserve_symlink_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            outside = Path(directory) / "outside.txt"
            outside.write_text("untouched\n", encoding="utf-8")
            artifact = root / "results" / "bad"
            artifact.parent.mkdir()
            try:
                os.symlink(outside, artifact)
            except (OSError, NotImplementedError) as error:  # pragma: no cover
                self.skipTest(f"symlinks unavailable: {error}")
            snapshot = workflow.tree_state(
                self.spec, artifact, cluster_root=root
            )
            self.assertEqual(snapshot["entries"][0]["type"], "symlink")
            destination = root / "cluster_state" / "quarantine" / "bad"
            workflow.quarantine_cluster_node(
                self.spec, artifact, destination, cluster_root=root
            )
            self.assertFalse(artifact.exists())
            self.assertFalse(artifact.is_symlink())
            self.assertTrue(destination.is_symlink())
            self.assertEqual(outside.read_text(encoding="utf-8"), "untouched\n")

            internal = root / "results" / "internal"
            internal.mkdir()
            os.symlink(outside, internal / "escaped")
            state = workflow.tree_state(self.spec, internal, cluster_root=root)
            self.assertIn("symlink", {row["type"] for row in state["entries"]})
            internal_destination = root / "cluster_state" / "quarantine" / "internal"
            workflow.quarantine_cluster_node(
                self.spec, internal, internal_destination, cluster_root=root
            )
            self.assertTrue((internal_destination / "escaped").is_symlink())

    def test_registered_workspace_ancestor_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                os.symlink(outside, root / "results")
            except (OSError, NotImplementedError) as error:  # pragma: no cover
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "symlink"):
                supplementary.pixel_output(self.spec, cluster_root=root)
            with self.assertRaisesRegex(ValueError, "escapes"):
                workflow.require_safe_cluster_path(
                    self.spec,
                    root / "results" / ".." / ".." / "outside",
                    cluster_root=root,
                )

    def test_controls_runtime_receipt_rejects_cpu_and_wrong_gpu(self) -> None:
        versions = {
            name: self.spec["runtime"][name]
            for name in ("python", "jax", "jaxlib", "numpy", "dm-control", "mujoco")
        }
        runtime = {
            **versions,
            "backend": "gpu",
            "visible_device_count": 1,
            "device_platforms": ["gpu"],
            "device_kinds": ["NVIDIA L40S"],
            "cuda_visible_devices": "0",
            "environment": self.spec["runtime"]["environment"],
        }
        preflight = {
            "versions": versions,
            "environment": self.spec["runtime"]["environment"],
            "jax_runtime": {
                "backend": "gpu",
                "visible_device_count": 1,
                "device_platforms": ["gpu"],
                "device_kinds": ["NVIDIA L40S"],
            },
            "slurm": {"cuda_visible_devices": "0"},
        }
        supplementary._validate_controls_worker_runtime(
            runtime, spec=self.spec, preflight=preflight
        )
        cpu = copy.deepcopy(runtime)
        cpu.update(
            backend="cpu", device_platforms=["cpu"], device_kinds=["Apple CPU"]
        )
        with self.assertRaisesRegex(ValueError, "single L40S"):
            supplementary._validate_controls_worker_runtime(
                cpu, spec=self.spec, preflight=preflight
            )
        wrong_gpu = copy.deepcopy(runtime)
        wrong_gpu["device_kinds"] = ["NVIDIA A100"]
        wrong_preflight = copy.deepcopy(preflight)
        wrong_preflight["jax_runtime"]["device_kinds"] = ["NVIDIA A100"]
        with self.assertRaisesRegex(ValueError, "single L40S"):
            supplementary._validate_controls_worker_runtime(
                wrong_gpu, spec=self.spec, preflight=wrong_preflight
            )

    def test_post_pilot_feasibility_uses_frozen_scale_and_blocks_short_lifetime(self) -> None:
        workspace = {"remaining_seconds_at_observation": 10_000_000_000}
        quota = {"remaining_quota_kib": 10_000_000_000, "remaining_inodes": 10_000_000}
        record = feasibility.assess_post_pilot_feasibility(
            self.spec,
            measured_gpu_seconds=1000.0,
            measured_artifact_bytes=1_000_000,
            measured_artifact_files=1000,
            measured_result_updates=4_320_000,
            workspace_record=workspace,
            quota_record=quota,
        )
        self.assertEqual(
            record["frozen_campaign"]["total_execution_units"], 6992
        )
        self.assertEqual(
            record["frozen_campaign"]["total_update_equivalents"], 396_540_000
        )
        short = {
            "remaining_seconds_at_observation": 3600
            * self.spec["workspace_allocator"]["minimum_remaining_hours"]
            + 1
        }
        with self.assertRaisesRegex(RuntimeError, "extension"):
            feasibility.assess_post_pilot_feasibility(
                self.spec,
                measured_gpu_seconds=1000.0,
                measured_artifact_bytes=1_000_000,
                measured_artifact_files=1000,
                measured_result_updates=4_320_000,
                workspace_record=short,
                quota_record=quota,
            )

    def test_controls_gpu_receipt_requires_exact_ledger_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {
                "cell_id": "world-" + "a" * 24,
                "identity_sha256": "b" * 64,
            }
            freeze = {
                "map_sha256": {"world": "c" * 64},
                "run_contract_sha256": "d" * 64,
            }
            versions = {
                name: self.spec["runtime"][name]
                for name in ("python", "jax", "jaxlib", "numpy", "dm-control", "mujoco")
            }
            worker_runtime = {
                **versions,
                "backend": "gpu",
                "visible_device_count": 1,
                "device_platforms": ["gpu"],
                "device_kinds": ["NVIDIA L40S"],
                "cuda_visible_devices": "0",
                "environment": self.spec["runtime"]["environment"],
            }
            preflight = {
                "preflight_sha256": "e" * 64,
                "versions": versions,
                "environment": self.spec["runtime"]["environment"],
                "jax_runtime": {
                    "backend": "gpu",
                    "visible_device_count": 1,
                    "device_platforms": ["gpu"],
                    "device_kinds": ["NVIDIA L40S"],
                },
                "slurm": {"cuda_visible_devices": "0"},
            }
            marker = {
                "schema_version": supplementary.CONTROLS_UNIT_SCHEMA,
                "status": "canonical_stage_validator_passed_on_gpu",
                "profile": "development",
                "stage": "world",
                "cell_id": entry["cell_id"],
                "identity_sha256": entry["identity_sha256"],
                "map_sha256": freeze["map_sha256"]["world"],
                "run_contract_sha256": freeze["run_contract_sha256"],
                "preflight_sha256": preflight["preflight_sha256"],
                "worker_runtime": worker_runtime,
                "worker_runtime_sha256": workflow.object_sha256(worker_runtime),
                "files": [{"path": "x", "bytes": 1, "sha256": "f" * 64}],
                "slurm_array_job_id": "123",
                "slurm_task_id": "0",
            }
            marker["unit_verification_sha256"] = workflow.object_sha256(marker)
            marker_path = supplementary.controls_unit_marker_path(
                self.spec,
                "development",
                "world",
                entry["cell_id"],
                cluster_root=root,
            )
            workflow.write_json_once(marker_path, marker)
            ledger_path = workflow.ledger_path(self.spec, cluster_root=root)
            workflow.append_ledger(
                ledger_path,
                {
                    "event": "controls_unit_gpu_verified",
                    "profile": "development",
                    "stage": "world",
                    "cell_id": entry["cell_id"],
                    "unit_verification_sha256": marker["unit_verification_sha256"],
                    "slurm_array_job_id": "123",
                    "slurm_task_id": "0",
                },
            )
            rows = workflow.verify_ledger(ledger_path)
            verified = supplementary.require_controls_unit_verified(
                self.spec,
                "development",
                "world",
                entry,
                checkout=WORKSPACE,
                cluster_root=root,
                authenticate_artifact=False,
                _freeze=freeze,
                _preflight=preflight,
                _ledger_rows=rows,
            )
            self.assertEqual(
                verified["unit_verification_sha256"],
                marker["unit_verification_sha256"],
            )
            with self.assertRaisesRegex(ValueError, "ledger"):
                supplementary.require_controls_unit_verified(
                    self.spec,
                    "development",
                    "world",
                    entry,
                    checkout=WORKSPACE,
                    cluster_root=root,
                    authenticate_artifact=False,
                    _freeze=freeze,
                    _preflight=preflight,
                    _ledger_rows=[],
                )

    def test_pixel_quick_missing_scan_authenticates_receipt(self) -> None:
        entry = {"array_index": 0, "cell_id": "pixel-" + "a" * 24}
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "receipt.json"
            marker.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                supplementary,
                "_read_regular_json",
                return_value={"count": 1, "entries": [entry]},
            ), mock.patch.object(
                supplementary, "pixel_unit_marker_path", return_value=marker
            ), mock.patch.object(
                supplementary, "require_pixel_unit_verified", return_value={}
            ) as verifier:
                missing = supplementary.incomplete_pixel_indices(
                    self.spec,
                    checkout=WORKSPACE,
                    cluster_root=Path(directory),
                    strong_validation=False,
                )
            self.assertEqual(missing, [])
            self.assertFalse(verifier.call_args.kwargs["authenticate_artifact"])

    def test_pixel_verifier_uses_cluster_authenticated_aggregate_marker(self) -> None:
        marker = {
            "expected_artifacts": 30,
            "tracks": ["equal_compiler_flops", "equal_updates"],
            "aggregate_sha256": "a" * 64,
            "profile_verification_sha256": "b" * 64,
        }
        with mock.patch.object(
            supplementary, "require_pixel_profile_verified", return_value=marker
        ) as verifier:
            result = verify_cluster_evidence.verify_pixel(
                self.spec, cluster_root=Path("/private/tmp/nonexistent-cluster")
            )
        verifier.assert_called_once()
        self.assertEqual(result["artifacts"], 30)
        self.assertEqual(result["aggregate_sha256"], "a" * 64)

    def test_valid_completed_pixel_profile_branch_has_no_unbound_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate_path = root / "aggregate.json"
            aggregate_path.write_text("{}\n", encoding="utf-8")
            marker_path = root / "profile.json"
            entry = {"cell_id": "pixel-" + "a" * 24}
            freeze = {
                "source_commit": "b" * 40,
                "run_contract_sha256": "c" * 64,
                "cluster_freeze_sha256": "d" * 64,
            }
            stage_map = {"map_sha256": "e" * 64, "entries": [entry]}
            receipt = {"unit_verification_sha256": "f" * 64}
            aggregate = {
                "requested_tracks": list(self.spec["supplementary"]["pixel"]["tracks"]),
                "expected_input_count": 30,
                "tracks": {
                    track: {} for track in self.spec["supplementary"]["pixel"]["tracks"]
                },
                "analysis_sha256": "1" * 64,
            }
            pilot = {"profile_verification_sha256": "2" * 64}
            marker = {
                "schema_version": supplementary.PIXEL_PROFILE_SCHEMA,
                "status": "verified_aggregated_complete_on_gpu",
                "source_commit": freeze["source_commit"],
                "run_contract_sha256": freeze["run_contract_sha256"],
                "cluster_freeze_sha256": freeze["cluster_freeze_sha256"],
                "map_sha256": stage_map["map_sha256"],
                "expected_artifacts": 30,
                "unit_verifications": [
                    {
                        "cell_id": entry["cell_id"],
                        "unit_verification_sha256": receipt["unit_verification_sha256"],
                    }
                ],
                "aggregate_sha256": workflow.file_sha256(aggregate_path),
                "aggregate_analysis_sha256": aggregate["analysis_sha256"],
                "tracks": sorted(self.spec["supplementary"]["pixel"]["tracks"]),
                "matched_pilot_profile_verification_sha256": pilot[
                    "profile_verification_sha256"
                ],
                "slurm_job_id": "123",
            }
            marker["profile_verification_sha256"] = workflow.object_sha256(marker)
            workflow.write_json_once(marker_path, marker)
            ledger = [
                {
                    "event": "pixel_profile_verified",
                    "profile_verification_sha256": marker[
                        "profile_verification_sha256"
                    ],
                    "map_sha256": marker["map_sha256"],
                    "aggregate_sha256": marker["aggregate_sha256"],
                    "slurm_job_id": "123",
                }
            ]
            fake_pixel = mock.Mock()
            fake_pixel.verify_aggregate.return_value = aggregate
            with mock.patch.object(
                supplementary, "pixel_profile_marker_path", return_value=marker_path
            ), mock.patch.object(
                supplementary, "require_pixel_freeze", return_value=freeze
            ), mock.patch.object(
                supplementary, "_read_regular_json", return_value=stage_map
            ), mock.patch.object(
                supplementary, "require_pixel_unit_verified", return_value=receipt
            ), mock.patch.object(
                supplementary, "_pixel_artifact_root", return_value=root / "artifact"
            ), mock.patch.object(
                supplementary, "pixel_aggregate_path", return_value=aggregate_path
            ), mock.patch.object(
                supplementary, "_pixel_module", return_value=fake_pixel
            ), mock.patch.object(
                supplementary, "require_authenticated_main_profile", return_value=pilot
            ), mock.patch.object(
                workflow, "verify_ledger", return_value=ledger
            ):
                verified = supplementary.require_pixel_profile_verified(
                    self.spec, checkout=WORKSPACE, cluster_root=root
                )
            self.assertEqual(
                verified["profile_verification_sha256"],
                marker["profile_verification_sha256"],
            )
            fake_pixel.verify_aggregate.assert_called_once()

    def test_z_cluster_success_token(self) -> None:
        print("NEURIPS_CLUSTER_WORKFLOW_TESTS_OK")

    def test_ledger_is_hash_chained_and_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            workflow.append_ledger(path, {"event": "first", "job_id": "1"})
            workflow.append_ledger(path, {"event": "second", "job_id": "2"})
            rows = workflow.verify_ledger(path)
            self.assertEqual([row["sequence"] for row in rows], [0, 1])
            rows[0]["job_id"] = "forged"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                workflow.verify_ledger(path)

    def test_profile_marker_binds_matrix_results_slurm_id_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "profile"
            output.mkdir()
            matrix = {
                "profile": "confirmatory",
                "matrix_sha256": "e" * 64,
                "cells": [{"cell_id": "one"}],
            }
            workflow.write_json_once(output / "matrix.json", matrix)
            (output / "analysis.json").write_text("analysis\n", encoding="utf-8")
            (output / "REPORT.md").write_text("report\n", encoding="utf-8")
            marker = {
                "schema_version": workflow.PROFILE_MARKER_SCHEMA,
                "status": "verified_complete",
                "profile": "confirmatory",
                "matrix_sha256": matrix["matrix_sha256"],
                "expected_cell_count": 1,
                "result_files": workflow._profile_result_files(
                    output, "confirmatory"
                ),
                "slurm_job_id": "12345",
            }
            marker["profile_verification_sha256"] = workflow.object_sha256(marker)
            workflow.write_json_once(workflow.profile_marker_path(output), marker)
            self.assertEqual(
                workflow.require_profile_verified(output, "confirmatory"), marker
            )

            forged = copy.deepcopy(marker)
            forged["result_files"]["analysis.json"] = "0" * 64
            forged["profile_verification_sha256"] = workflow.object_sha256(
                {
                    key: value
                    for key, value in forged.items()
                    if key != "profile_verification_sha256"
                }
            )
            workflow.profile_marker_path(output).write_text(
                json.dumps(forged), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "marker is invalid"):
                workflow.require_profile_verified(output, "confirmatory")

            external = Path(directory) / "external.json"
            external.write_text(json.dumps(marker), encoding="utf-8")
            workflow.profile_marker_path(output).unlink()
            try:
                os.symlink(external, workflow.profile_marker_path(output))
            except (OSError, NotImplementedError) as error:  # pragma: no cover
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                workflow.require_profile_verified(output, "confirmatory")


if __name__ == "__main__":
    unittest.main()
