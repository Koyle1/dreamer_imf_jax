from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import actor_gap_roadmap_study as study
from dreamer_imf_compare.artifacts import write_json_atomic


class ActorGapRoadmapStudyTests(unittest.TestCase):
    def test_diagnostic_strict_replay_discards_first_full_execution(self) -> None:
        first = ({"execution": 1}, {"trace": np.asarray([1])})
        second = ({"execution": 2}, {"trace": np.asarray([2])})
        with mock.patch.object(
            study, "_compute_diagnostic_cell", side_effect=[first, second]
        ) as compute:
            retained = study._compute_diagnostic_cell_after_discarded_warmup(
                Path("/unused"), {}, {}
            )
        self.assertEqual(compute.call_count, 2)
        self.assertEqual(retained[0], second[0])
        np.testing.assert_array_equal(retained[1]["trace"], second[1]["trace"])

    def test_calibration_strict_replay_discards_first_full_derivation(self) -> None:
        first = ([{"execution": 1}], {"trace": np.asarray([1])})
        second = ([{"execution": 2}], {"trace": np.asarray([2])})
        with mock.patch.object(
            study, "_derive_training_only_calibration", side_effect=[first, second]
        ) as derive:
            retained = study._derive_training_only_calibration_after_discarded_warmup(
                {}
            )
        self.assertEqual(derive.call_count, 2)
        self.assertEqual(retained[0], second[0])
        np.testing.assert_array_equal(retained[1]["trace"], second[1]["trace"])

    def test_exclusive_json_publication_is_atomic_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "evidence.json"
            with mock.patch.object(study.os, "link", side_effect=OSError("crash")):
                with self.assertRaisesRegex(OSError, "crash"):
                    study._write_json_exclusive(destination, {"complete": True})
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.iterdir()), [])

            study._write_json_exclusive(destination, {"version": 1})
            original = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                study._write_json_exclusive(destination, {"version": 2})
            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [destination])

    def test_multifile_bundle_is_visible_only_after_atomic_directory_publish(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "cell"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with study._staged_artifact_directory(final) as staging:
                    study._write_json_exclusive(staging / "result.json", {"ok": True})
                    raise RuntimeError("interrupted")
            self.assertFalse(final.exists())
            self.assertEqual(len(list(root.glob(".cell.partial-*"))), 1)

            with study._staged_artifact_directory(final) as staging:
                study._write_json_exclusive(staging / "result.json", {"ok": True})
                study._write_npz_exclusive(
                    staging / "trace.npz", {"x": np.asarray([1])}
                )
            self.assertTrue((final / "result.json").is_file())
            self.assertTrue((final / "trace.npz").is_file())

    def test_non_strict_cell_verification_cannot_publish_a_marker(self) -> None:
        for verifier in (
            study.verify_diagnostic_cell,
            study.verify_model_cell,
            study.verify_evaluation_cell,
        ):
            with self.subTest(verifier=verifier.__name__):
                with self.assertRaisesRegex(ValueError, "requires strict replay"):
                    verifier(
                        "/path/that/must/not/be/read",
                        0,
                        strict_replay=False,
                        submission_authorization={},
                    )

    def test_endpoint_warmup_exercises_the_complete_executed_action_path(self) -> None:
        source = inspect.getsource(study._run_endpoint_action_sequence_arm)
        warm = source.index("warm_decision = _select_endpoint_executed_action")
        seal = source.index("warmed = True")
        live = source.index(") = _select_endpoint_executed_action", seal)
        self.assertLess(warm, seal)
        self.assertGreater(live, seal)

    def test_action_sequence_ensemble_uses_common_posterior_random_numbers(
        self,
    ) -> None:
        source = inspect.getsource(study._run_action_sequence_arm)
        self.assertIn("actor-gap-action-sequence-posterior-crn", source)
        self.assertIn("posterior_keys = [common_posterior_key] * len(worlds)", source)
        self.assertNotIn('"actor-gap-action-sequence-posterior",', source)

    def test_live_controller_latency_includes_each_episode_first_step(self) -> None:
        for controller in (
            study._run_flowmpc_arm,
            study._run_action_sequence_arm,
            study._run_endpoint_action_sequence_arm,
        ):
            source = inspect.getsource(controller)
            self.assertNotIn("if step > 0", source)
            self.assertIn("timed_steps += 1", source)

    def test_every_live_controller_records_bounded_imagined_coverage(self) -> None:
        for controller in (
            study._run_flowmpc_arm,
            study._run_action_sequence_arm,
            study._run_endpoint_action_sequence_arm,
        ):
            source = inspect.getsource(controller)
            self.assertIn("IMAGINED_COVERAGE_STEP_INTERVAL", source)
            self.assertIn("IMAGINED_COVERAGE_PARTICLES", source)
            self.assertIn('("proposal",', source)
            self.assertIn('("best",', source)
            self.assertIn('("reference",', source)
            self.assertIn('trace[f"imagined_{name}"]', source)

    def test_controller_summary_retains_safeguard_telemetry(self) -> None:
        source = inspect.getsource(study._controller_summary)
        for field in (
            "mean_trust_backtracks",
            "frozen_reference_fallback_fraction",
            "mean_heldout_acceptance_improvement",
            "heldout_acceptance_fraction",
            "executed_behavior_violation_fraction",
        ):
            self.assertIn(f'"{field}"', source)

    def test_recursive_and_flowmpc_imagined_points_have_planner_query_shapes(
        self,
    ) -> None:
        import jax
        import jax.numpy as jnp
        from imf_dreamer_jax import (
            DreamerConfig,
            ReBRACConfig,
            initial_state,
            init_rebrac_state,
        )
        from imf_dreamer_jax.imf import init_imf
        from imf_dreamer_jax.world_model import init_world_model
        from dreamer_imf_compare.actor_gap_model_training import endpoint_condition_dim

        config = DreamerConfig(
            observation_shape=(3,),
            action_dim=2,
            deterministic_dim=4,
            stochastic_dim=2,
            embedding_dim=4,
            hidden_dim=8,
            prior="imf",
            burn_in=1,
            imf_trajectory_enabled=True,
            imf_trajectory_clean_probability=1.0,
            imf_trajectory_corrupted_probability=0.0,
            imf_trajectory_suffix_probability=0.0,
        )
        world = init_world_model(config, jax.random.PRNGKey(1))
        belief = initial_state(config, 1)
        current = jnp.zeros((1, 3), dtype=jnp.float32)
        sequences = jnp.zeros((3, 5, 2), dtype=jnp.float32)
        noises = jnp.zeros((4, 5, 2), dtype=jnp.float32)
        observations, actions = study._imagined_action_sequence_points(
            world, config, belief, current, sequences, noises
        )
        self.assertEqual(observations.shape, (3, 4, 5, 3))
        self.assertEqual(actions.shape, (3, 4, 5, 2))

        rebrac = init_rebrac_state(
            jax.random.PRNGKey(2),
            ReBRACConfig(state_dim=3, action_dim=2, hidden_dim=8),
        )
        observations, actions = study._flowmpc_imagined_actor_points(
            rebrac.actor, world, config, belief, current, noises
        )
        self.assertEqual(observations.shape, (4, 5, 3))
        self.assertEqual(actions.shape, (4, 5, 2))

        endpoint_checkpoint = {
            "params": init_imf(
                jax.random.PRNGKey(3),
                sample_dim=3,
                condition_dim=endpoint_condition_dim(config, 5),
                hidden_dim=8,
                depth=1,
            ),
            "maximum_chunk_horizon": 5,
            "observation_mean": jnp.zeros((3,), dtype=jnp.float32),
            "observation_std": jnp.ones((3,), dtype=jnp.float32),
        }
        endpoint_noises = jnp.zeros((4, 5, 3), dtype=jnp.float32)
        for direct_any_step in (False, True):
            observations, actions = study._endpoint_imagined_action_sequence_points(
                endpoint_checkpoint,
                config,
                current,
                sequences,
                endpoint_noises,
                direct_any_step=direct_any_step,
            )
            self.assertEqual(observations.shape, (3, 4, 5, 3))
            self.assertEqual(actions.shape, (3, 4, 5, 2))

    def test_exact_counterfactual_includes_causal_horizon5_planner_objective(
        self,
    ) -> None:
        source = inspect.getsource(study._exact_counterfactual_result)
        self.assertIn("horizon5_planner_objective", source)
        self.assertIn("simulator_planner_objective", source)
        self.assertIn("_recursive_action_sequence_objective", source)
        self.assertIn("imagined_proposal_observations", source)
        self.assertIn("imagined_best_observations", source)
        self.assertIn("imagined_reference_observations", source)

    def test_final_report_recovery_reauthenticates_external_dependency(self) -> None:
        source = inspect.getsource(study.finalize)
        recovery = source.index('if not (root / "verified/final.json").is_file():')
        authenticate = source.index("authenticate_dependency", recovery)
        rebuild = source.index("_build_final_report", recovery)
        self.assertLess(authenticate, rebuild)

    def _complete_rows(self) -> dict[str, list[dict[str, float | int | str]]]:
        rows: dict[str, list[dict[str, float | int | str]]] = {}
        for arm_index, arm in enumerate(study.ALL_ARMS):
            rows[arm] = [
                {
                    "arm": arm,
                    "world_model_seed": world_seed,
                    "actor_seed": actor_seed,
                    "mean_return": float(
                        100 * world_index + 10 * actor_index + arm_index
                    ),
                }
                for world_index, world_seed in enumerate(study.WORLD_SEEDS)
                for actor_index, actor_seed in enumerate(study.ACTOR_SEEDS)
            ]
        return rows

    def test_dependency_free_self_test(self) -> None:
        study.self_test()

    def test_hierarchical_units_average_nested_actors_before_worlds(self) -> None:
        rows = self._complete_rows()
        units = study._world_seed_arm_units(rows)

        first_arm = study.ALL_ARMS[0]
        self.assertEqual(
            [row["world_model_seed"] for row in units[first_arm]],
            list(study.WORLD_SEEDS),
        )
        self.assertEqual(
            [row["nested_actor_seeds"] for row in units[first_arm]],
            [list(study.ACTOR_SEEDS)] * len(study.WORLD_SEEDS),
        )
        np.testing.assert_allclose(
            [row["world_seed_mean_return"] for row in units[first_arm]],
            [5.0, 105.0, 205.0],
        )
        summary = study._arm_summary(first_arm, units[first_arm])
        self.assertEqual(summary["raw_return_iqm"], 105.0)
        self.assertEqual(summary["normalized_return_iqm"], 0.105)

    def test_hierarchical_units_reject_missing_or_duplicate_nested_actor(self) -> None:
        rows = self._complete_rows()
        arm = study.ALL_ARMS[0]
        rows[arm] = rows[arm][1:]
        with self.assertRaisesRegex(ValueError, "exactly two nested actor seeds"):
            study._world_seed_arm_units(rows)

        rows = self._complete_rows()
        rows[arm].append(dict(rows[arm][0]))
        with self.assertRaisesRegex(ValueError, "exactly two nested actor seeds"):
            study._world_seed_arm_units(rows)

    def test_contrast_is_paired_at_actor_and_world_seed_levels(self) -> None:
        rows = self._complete_rows()
        baseline = study.ALL_ARMS[0]
        candidate = study.ALL_ARMS[1]
        units = study._world_seed_arm_units(rows)
        contrast = study._contrast_summary(
            "synthetic", candidate, baseline, rows, units
        )

        self.assertEqual(len(contrast["paired_actor_cells"]), 6)
        self.assertEqual(len(contrast["paired_world_seed_units"]), 3)
        self.assertEqual(contrast["raw_return_delta_iqm"], 1.0)
        self.assertEqual(contrast["favorable_world_seed_fraction"], 1.0)
        self.assertEqual(contrast["favorable_nested_actor_cell_fraction"], 1.0)

        unpaired = self._complete_rows()
        unpaired[candidate] = unpaired[candidate][:-1]
        unpaired_units = {
            **units,
            candidate: units[candidate],
        }
        with self.assertRaisesRegex(ValueError, "not paired"):
            study._contrast_summary(
                "bad", candidate, baseline, unpaired, unpaired_units
            )

    def test_stage_body_binds_every_cell_marker_and_upstream_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cells = []
            validated = {}
            for index in range(2):
                marker_path = root / f"verified/cell-{index}.json"
                write_json_atomic(marker_path, {"immutable": index})
                cell = {
                    "index": index,
                    "cell_id": f"cell-{index}",
                    "marker_path": str(marker_path.relative_to(root)),
                }
                cells.append(cell)
                validated[index] = {
                    "marker_sha256": f"marker-{index}",
                    "result_file_sha256": f"result-{index}",
                }
            manifest = {
                "source_commit": "c" * 40,
                "manifest_sha256": "m" * 64,
                "diagnostic_cells": cells,
            }

            with mock.patch.object(
                study,
                "_validate_cell_marker",
                side_effect=lambda _root, _manifest, cell, _stage: validated[
                    int(cell["index"])
                ],
            ), mock.patch.object(
                study,
                "_validate_calibration_marker",
                return_value={"marker_sha256": "calibration-marker"},
            ):
                body = study._stage_marker_body(root, manifest, "diagnostic")

        self.assertEqual(body["completed_cells"], 2)
        self.assertEqual(body["expected_cells"], 2)
        self.assertIs(body["strict_replay_for_every_cell"], True)
        self.assertEqual(
            [row["cell_id"] for row in body["cells"]],
            ["cell-0", "cell-1"],
        )
        self.assertEqual(body["calibration_marker_sha256"], "calibration-marker")
        self.assertTrue(all(row["marker_file_sha256"] for row in body["cells"]))

    def test_stage_validation_rejects_a_modified_stage_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker_path = root / study._STAGE_PATHS["diagnostic"]
            write_json_atomic(marker_path, {"status": "tampered"})
            body = {
                "schema_version": study.STAGE_MARKER_SCHEMA,
                "status": "verified",
                "stage": "diagnostic",
            }
            execution = ({"job_id": "1"}, {"kind": "test"}, {"gpu": True})
            with mock.patch.object(
                study, "_stage_marker_body", return_value=body
            ), mock.patch.object(
                study,
                "_validate_retained_stage_verifier_execution_binding",
                return_value=execution,
            ):
                with self.assertRaisesRegex(ValueError, "stage marker is invalid"):
                    study._validate_stage_marker(root, {}, "diagnostic")

    def test_endpoint_filter_controls_the_executed_action(self) -> None:
        candidate = np.asarray([0.8, -0.7], dtype=np.float32)
        reference = np.asarray([0.1, -0.2], dtype=np.float32)
        action, accepted, distance, executed_safe = (
            study._apply_endpoint_behavior_filter(
                candidate, reference, np.asarray(0.4, np.float32), 0.2
            )
        )
        np.testing.assert_array_equal(np.asarray(action), reference)
        self.assertFalse(bool(np.asarray(accepted)))
        self.assertEqual(float(np.asarray(distance)), 0.0)
        self.assertTrue(bool(np.asarray(executed_safe)))

        close_candidate = np.asarray([0.15, -0.15], dtype=np.float32)
        action, accepted, _, executed_safe = study._apply_endpoint_behavior_filter(
            close_candidate, reference, np.asarray(0.1, np.float32), 0.2
        )
        np.testing.assert_array_equal(np.asarray(action), close_candidate)
        self.assertTrue(bool(np.asarray(accepted)))
        self.assertTrue(bool(np.asarray(executed_safe)))

    def test_submission_map_is_immutable_index_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            cell = {
                "index": 0,
                "cell_id": "diagnostic-211",
                "result_path": "diagnostics/diagnostic-211/result.json",
                "trace_path": "diagnostics/diagnostic-211/traces.npz",
                "marker_path": "verified/diagnostic-211.json",
            }
            manifest = {
                "source_commit": "c" * 40,
                "manifest_sha256": "m" * 64,
                "diagnostic_cells": [cell],
            }
            write_json_atomic(root / "manifest.json", manifest)
            upstream = {
                "stage": "calibration",
                "marker_sha256": "u" * 64,
            }
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_submission_upstream_marker", return_value=upstream
            ):
                submission = study.write_submission_map(
                    root, "diagnostic", [{"index": 0, "mode": "new"}]
                )
                map_path = root / submission["map_path"]
                validated = study.validate_submission_map(
                    root,
                    map_path,
                    "diagnostic",
                    0,
                    expected_file_sha256=study.benchmark.file_sha256(map_path),
                )
                authorization = {
                    "map_path": submission["map_path"],
                    "map_file_sha256": study.benchmark.file_sha256(map_path),
                    "map_sha256": submission["map_sha256"],
                    "mode": "new",
                }
                self.assertEqual(
                    study._validate_retained_submission_authorization(
                        root, manifest, cell, "diagnostic", authorization
                    ),
                    authorization,
                )
                result_path = root / cell["result_path"]
                trace_path = root / cell["trace_path"]
                write_json_atomic(result_path, {"complete": True})
                study.benchmark._write_npz_atomic(
                    trace_path, {"value": np.asarray([1], dtype=np.int32)}
                )
                verified = study.validate_submission_map(
                    root,
                    map_path,
                    "diagnostic",
                    0,
                    expected_file_sha256=study.benchmark.file_sha256(map_path),
                    action="verify",
                )
                self.assertEqual(verified, submission)
                with self.assertRaisesRegex(ValueError, "already has retained"):
                    study.validate_submission_map(
                        root,
                        map_path,
                        "diagnostic",
                        0,
                        expected_file_sha256=study.benchmark.file_sha256(map_path),
                        action="create",
                    )
            self.assertEqual(validated, submission)
            self.assertEqual(len(submission["entries"]), 1)
            entry = submission["entries"][0]
            self.assertEqual(entry["index"], 0)
            self.assertEqual(entry["mode"], "new")
            self.assertEqual(entry["retained_data_files"], [])
            self.assertEqual(
                entry["artifact_state_sha256"],
                study._submission_entry_digest(entry),
            )

            tampered = dict(submission)
            tampered_entry = {
                "index": 1,
                "mode": "new",
                "retained_data_files": [],
            }
            tampered_entry["artifact_state_sha256"] = study._submission_entry_digest(
                tampered_entry
            )
            tampered["entries"] = [tampered_entry]
            write_json_atomic(map_path, tampered)
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_submission_upstream_marker", return_value=upstream
            ):
                with self.assertRaisesRegex(ValueError, "authorization differs"):
                    study.validate_submission_map(
                        root,
                        map_path,
                        "diagnostic",
                        0,
                        expected_file_sha256=study.benchmark.file_sha256(map_path),
                    )

    def test_array_receipt_binds_live_job_and_exact_map_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            cell = {
                "index": 0,
                "cell_id": "diagnostic-211",
                "result_path": "diagnostics/diagnostic-211/result.json",
                "trace_path": "diagnostics/diagnostic-211/traces.npz",
                "marker_path": "verified/diagnostic-211.json",
            }
            manifest = {
                "source_commit": "c" * 40,
                "manifest_sha256": "m" * 64,
                "diagnostic_cells": [cell],
            }
            write_json_atomic(root / "manifest.json", manifest)
            upstream = {"stage": "calibration", "marker_sha256": "u" * 64}
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_submission_upstream_marker", return_value=upstream
            ):
                submission = study.write_submission_map(
                    root, "diagnostic", [{"index": 0, "mode": "new"}]
                )
                map_path = root / submission["map_path"]
                authorization = {
                    "map_path": submission["map_path"],
                    "map_file_sha256": study.benchmark.file_sha256(map_path),
                    "map_sha256": submission["map_sha256"],
                    "mode": "new",
                }
                receipt = {
                    "schema_version": study.SUBMISSION_RECORD_SCHEMA,
                    "status": "submitted",
                    "label": "diagnostic",
                    "source_commit": manifest["source_commit"],
                    "output_root": str(root),
                    "intent_path": submission["map_path"],
                    "intent_file_sha256": authorization["map_file_sha256"],
                    "jobs": {"array": "123", "verifier": "124"},
                    "dependency_policies": {
                        "verifier": "afterok_array_kill_on_invalid_dependency"
                    },
                    "failure": None,
                    "launch_policy": "held_until_receipt_persisted_before_release",
                }
                receipt["receipt_sha256"] = study.benchmark.object_sha256(receipt)
                receipt_path = study._submission_receipt_path(root, authorization)
                study._write_json_exclusive(receipt_path, receipt)
                scheduler = {
                    "job_id": "125",
                    "array_job_id": "123",
                    "array_task_id": 0,
                    "node_name": "gpu01",
                    "boot_id": "boot-a",
                    "process_id": 10,
                    "process_start_ticks": 20,
                }
                with mock.patch.object(study, "_validate_slurm_allocation_certificate"):
                    binding = study._validate_array_submission_receipt(
                        root,
                        manifest,
                        cell,
                        "diagnostic",
                        authorization,
                        scheduler,
                    )
                self.assertEqual(binding["array_job_id"], "123")
                self.assertEqual(binding["verifier_job_id"], "124")
                with mock.patch.object(
                    study, "_validate_slurm_allocation_certificate"
                ), self.assertRaisesRegex(ValueError, "scheduler binding"):
                    study._validate_array_submission_receipt(
                        root,
                        manifest,
                        cell,
                        "diagnostic",
                        authorization,
                        {**scheduler, "array_job_id": "999"},
                    )

    def test_live_slurm_certificate_binds_array_parent_task_node_and_gpu(self) -> None:
        node = os.uname().nodename
        raw = (
            "JobId=125 ArrayJobId=123 ArrayTaskId=0 JobState=RUNNING "
            f"BatchHost={node} Account=dep_inin_dat Partition=gpu-l40s "
            "AllocTRES=cpu=8,mem=64G,node=1,gres/gpu=1,gres/gpu:l40s=1 "
            "TresPerNode=gres/gpu:1"
        )
        environment = {
            "CUDA_VISIBLE_DEVICES": "0",
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            study.subprocess, "check_output", return_value=raw
        ):
            certificate = study._slurm_allocation_certificate(
                "125", array_job_id="123", array_task_id=0
            )
        validated = study._validate_slurm_allocation_certificate(
            certificate,
            job_id="125",
            node_name=node,
            array_job_id="123",
            array_task_id=0,
        )
        self.assertEqual(validated, certificate)
        self.assertEqual(certificate["raw_record"], raw)
        self.assertEqual(certificate["canonical_record"]["ArrayTaskId"], "0")
        self.assertEqual(certificate["slurm_job_gpus"], "")
        self.assertEqual(certificate["slurm_step_gpus"], "")
        self.assertEqual(
            certificate["canonical_record_sha256"],
            study.benchmark.object_sha256(certificate["canonical_record"]),
        )
        with mock.patch.dict(
            os.environ,
            {
                **environment,
                "SLURM_JOB_GPUS": "3",
                "SLURM_STEP_GPUS": "3",
            },
            clear=True,
        ), mock.patch.object(study.subprocess, "check_output", return_value=raw):
            hinted = study._slurm_allocation_certificate(
                "125", array_job_id="123", array_task_id=0
            )
        self.assertEqual(hinted["slurm_job_gpus"], "3")
        self.assertEqual(hinted["slurm_step_gpus"], "3")
        tampered = {**certificate, "array_task_id": 1}
        tampered["certificate_sha256"] = study._unsigned_digest(
            tampered, "certificate_sha256"
        )
        with self.assertRaisesRegex(ValueError, "allocation certificate"):
            study._validate_slurm_allocation_certificate(
                tampered,
                job_id="125",
                node_name=node,
                array_job_id="123",
                array_task_id=0,
            )
        malformed_allocations = (
            raw.replace("gres/gpu=1,", "gres/gpu=2,"),
            raw.replace("gres/gpu=1,", ""),
            raw.replace("gres/gpu:l40s=1", "gres/gpu:a100=1"),
            raw.replace("TresPerNode=gres/gpu:1", "TresPerNode=gres/gpu:2"),
            raw.replace("cpu=8", "cpu=7"),
            raw.replace("mem=64G", "mem=32G"),
            raw.replace("node=1", "node=2"),
        )
        for malformed in malformed_allocations:
            with self.subTest(malformed=malformed), mock.patch.dict(
                os.environ, environment, clear=True
            ), mock.patch.object(
                study.subprocess, "check_output", return_value=malformed
            ), self.assertRaisesRegex(
                ValueError, "one-GPU"
            ):
                study._slurm_allocation_certificate(
                    "125", array_job_id="123", array_task_id=0
                )
        for variable in (
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "CUDA_VISIBLE_DEVICES",
        ):
            with self.subTest(variable=variable), mock.patch.dict(
                os.environ, {**environment, variable: "0,1"}, clear=True
            ), mock.patch.object(
                study.subprocess, "check_output", return_value=raw
            ), self.assertRaisesRegex(
                ValueError, "one-GPU"
            ):
                study._slurm_allocation_certificate(
                    "125", array_job_id="123", array_task_id=0
                )
        tampered_raw = {
            **certificate,
            "raw_record": raw.replace("Account=dep_inin_dat", "Account=other"),
        }
        tampered_raw["raw_record_sha256"] = study.benchmark.object_sha256(
            tampered_raw["raw_record"]
        )
        tampered_raw["certificate_sha256"] = study._unsigned_digest(
            tampered_raw, "certificate_sha256"
        )
        with self.assertRaisesRegex(ValueError, "allocation certificate"):
            study._validate_slurm_allocation_certificate(
                tampered_raw,
                job_id="125",
                node_name=node,
                array_job_id="123",
                array_task_id=0,
            )
        multi_gpu_raw = raw.replace(
            "gres/gpu=1,gres/gpu:l40s=1",
            "gres/gpu=2,gres/gpu:l40s=1",
        ).replace("TresPerNode=gres/gpu:1", "TresPerNode=gres/gpu:2")
        forged_multi_gpu = {
            **certificate,
            "raw_record": multi_gpu_raw,
            "alloc_tres": certificate["alloc_tres"].replace(
                "gres/gpu=1,gres/gpu:l40s=1",
                "gres/gpu=2,gres/gpu:l40s=1",
            ),
            "tres_per_node": "gres/gpu:2",
            "canonical_record": study._parse_scontrol_record(multi_gpu_raw),
        }
        forged_multi_gpu["raw_record_sha256"] = study.benchmark.object_sha256(
            forged_multi_gpu["raw_record"]
        )
        forged_multi_gpu["canonical_record_sha256"] = study.benchmark.object_sha256(
            forged_multi_gpu["canonical_record"]
        )
        forged_multi_gpu["certificate_sha256"] = study._unsigned_digest(
            forged_multi_gpu, "certificate_sha256"
        )
        with self.assertRaisesRegex(ValueError, "allocation certificate"):
            study._validate_slurm_allocation_certificate(
                forged_multi_gpu,
                job_id="125",
                node_name=node,
                array_job_id="123",
                array_task_id=0,
            )

    def test_gpu_and_fresh_process_certificates_are_fail_closed(self) -> None:
        runtime = {
            "backend": "gpu",
            "device_platforms": ["gpu"],
            "device_kinds": ["NVIDIA L40S"],
            "visible_device_count": 1,
            "cuda_visible_devices": "0",
            "jax_enable_x64": False,
            "jax_enable_compilation_cache": "true",
            "jax_compilation_cache_dir": "/work2/cache",
            "jax_persistent_cache_min_compile_time_secs": "0",
            "jax_persistent_cache_min_entry_size_bytes": "-1",
            "jax_raise_persistent_cache_errors": "true",
        }
        self.assertEqual(study._require_single_gpu_runtime(runtime), runtime)
        with self.assertRaisesRegex(ValueError, "one-GPU/compiler"):
            study._require_single_gpu_runtime({**runtime, "backend": "cpu"})
        with self.assertRaisesRegex(ValueError, "one-GPU/compiler"):
            study._require_single_gpu_runtime(
                {
                    **runtime,
                    "device_platforms": ["gpu", "gpu"],
                    "device_kinds": ["NVIDIA L40S", "NVIDIA L40S"],
                    "visible_device_count": 2,
                    "cuda_visible_devices": "0,1",
                }
            )
        with self.assertRaisesRegex(ValueError, "one-GPU/compiler"):
            study._require_single_gpu_runtime({**runtime, "cuda_visible_devices": None})
        creation = {
            "node_name": "gpu01",
            "boot_id": "boot-a",
            "process_id": 10,
            "process_start_ticks": 20,
        }
        study._require_distinct_replay_processes(
            creation, {**creation, "process_id": 11}
        )
        with self.assertRaisesRegex(ValueError, "fresh Python process"):
            study._require_distinct_replay_processes(creation, creation)

    def test_runtime_homogeneity_excludes_scheduler_gpu_ordinal(self) -> None:
        runtime = {
            "python": "3.12.0",
            "python_executable": "/opt/venv/bin/python",
            "python_executable_sha256": "a" * 64,
            "platform_system": "Linux",
            "platform_release": "6.8.0",
            "platform_machine": "x86_64",
            "numpy_version": "2.1.0",
            "jax_version": "0.4.38",
            "jaxlib_version": "0.4.38",
            "dm_control_version": "1.0.31",
            "mujoco_version": "3.3.5",
            "environment_package_count": 123,
            "environment_packages_sha256": "b" * 64,
            "backend": "gpu",
            "device_platforms": ["gpu"],
            "device_kinds": ["NVIDIA L40S"],
            "devices": ["cuda:0"],
            "visible_device_count": 1,
            "xla_platform_version": "CUDA 12",
            "jax_enable_x64": False,
            "cuda_visible_devices": "0",
            "jax_enable_compilation_cache": "true",
            "jax_compilation_cache_dir": "/work2/cache",
            "jax_persistent_cache_min_compile_time_secs": "0",
            "jax_persistent_cache_min_entry_size_bytes": "-1",
            "jax_persistent_cache_enable_xla_caches": "all",
            "jax_raise_persistent_cache_errors": "true",
        }
        identity = study.benchmark.runtime_homogeneity_identity(runtime)
        moved = study.benchmark.runtime_homogeneity_identity(
            {**runtime, "cuda_visible_devices": "3", "devices": ["cuda:3"]}
        )
        self.assertEqual(identity, moved)
        changed_kind = study.benchmark.runtime_homogeneity_identity(
            {**runtime, "device_kinds": ["NVIDIA A100"]}
        )
        self.assertNotEqual(identity, changed_kind)

    def test_preflight_marker_requires_all_executed_gate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "source_commit": "c" * 40,
                "manifest_sha256": "m" * 64,
                "dependency_root": "/immutable/dependency",
                "diagnostic_cells": [{}, {}, {}],
                "model_cells": [{}] * 21,
                "evaluation_cells": [{}] * 90,
                "preflight": {
                    "marker_path": "verified/preflight.json",
                    "gate_paths": {
                        gate: f"verified/preflight-{gate}.json"
                        for gate in study.PREFLIGHT_GATES
                    },
                },
            }
            write_json_atomic(root / "manifest.json", manifest)
            authenticated = {
                "source_commit": "d" * 40,
                "manifest_file_sha256": "a" * 64,
                "report_file_sha256": "b" * 64,
            }
            scheduler = {
                "job_id": "123",
                "node_name": "gpu01",
                "boot_id": "boot-a",
                "process_id": 10,
                "process_start_ticks": 20,
            }
            receipt = {"job_id": "123"}
            runtime = {"runtime": "same"}
            execution = (scheduler, receipt, runtime)
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_live_single_execution_binding", return_value=execution
            ):
                for gate in study.PREFLIGHT_GATES:
                    study.write_preflight_gate(root, gate, {"passed": True})
            with mock.patch.object(
                study, "authenticate_dependency", return_value=authenticated
            ), mock.patch.object(
                study,
                "_validate_retained_single_execution_binding",
                return_value=execution,
            ), mock.patch.object(
                study.benchmark,
                "runtime_homogeneity_identity",
                return_value={"runtime": "same"},
            ):
                body = study._preflight_marker_body(root, manifest)
            self.assertEqual(set(body["gate_evidence"]), set(study.PREFLIGHT_GATES))
            self.assertTrue(
                all(body[f"{gate}_verified"] for gate in study.PREFLIGHT_GATES)
            )
            missing = root / manifest["preflight"]["gate_paths"]["single_gpu"]
            missing.unlink()
            with mock.patch.object(
                study, "authenticate_dependency", return_value=authenticated
            ), mock.patch.object(
                study,
                "_validate_retained_single_execution_binding",
                return_value=execution,
            ), mock.patch.object(
                study.benchmark,
                "runtime_homogeneity_identity",
                return_value={"runtime": "same"},
            ):
                with self.assertRaises(FileNotFoundError):
                    study._preflight_marker_body(root, manifest)


if __name__ == "__main__":
    unittest.main()
