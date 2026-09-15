from __future__ import annotations

import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_comparison.cluster.actor_gap_roadmap import (
    submit_actor_gap_roadmap as roadmap_submitter,
)
from dreamer_imf_compare import actor_gap_roadmap_study as study
from dreamer_imf_compare.artifacts import write_json_atomic


class ActorGapRoadmapStudyTests(unittest.TestCase):
    def test_live_a0_context_rejects_one_ulp_action_drift(self) -> None:
        action = np.asarray([0.5, -0.25], dtype=np.float32)
        trace = {"actions": action[None, None]}
        context = {
            "step": 0,
            "world_model_seed": 211,
            "actor_seed": 311,
            "evaluation_seed": 401,
            "host_action": action.copy(),
        }
        selected = study._validated_a0_noise_contexts(
            trace,
            [context],
            [0],
            world_seed=211,
            actor_seed=311,
            evaluation_seed=401,
        )
        self.assertIs(selected[0][1], context)

        changed = dict(context)
        changed_action = action.copy()
        changed_action[0] = np.nextafter(changed_action[0], np.float32(np.inf))
        changed["host_action"] = changed_action
        with self.assertRaisesRegex(ValueError, r"step 0: max_abs="):
            study._validated_a0_noise_contexts(
                trace,
                [changed],
                [0],
                world_seed=211,
                actor_seed=311,
                evaluation_seed=401,
            )

    def test_live_a0_context_rejects_missing_duplicate_or_wrong_identity(self) -> None:
        trace = {"actions": np.zeros((1, 2, 2), dtype=np.float32)}
        context = {
            "step": 0,
            "world_model_seed": 211,
            "actor_seed": 311,
            "evaluation_seed": 401,
            "host_action": np.zeros((2,), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "duplicates step 0"):
            study._validated_a0_noise_contexts(
                trace,
                [context, dict(context)],
                [0],
                world_seed=211,
                actor_seed=311,
                evaluation_seed=401,
            )
        with self.assertRaisesRegex(ValueError, "missing step 1"):
            study._validated_a0_noise_contexts(
                trace,
                [context],
                [1],
                world_seed=211,
                actor_seed=311,
                evaluation_seed=401,
            )
        with self.assertRaisesRegex(ValueError, "identity differs at step 0"):
            study._validated_a0_noise_contexts(
                trace,
                [context],
                [0],
                world_seed=223,
                actor_seed=311,
                evaluation_seed=401,
            )

    def test_diagnostics_consume_ephemeral_live_a0_contexts(self) -> None:
        for diagnostic in (
            study._one_step_prediction_rows,
            study._independent_noise_result,
            study._exact_counterfactual_result,
        ):
            source = inspect.getsource(diagnostic)
            self.assertNotIn("observe_step(", source)
        independent = inspect.getsource(study._independent_noise_result)
        self.assertNotIn("jit_flowmpc_adapt_actor", independent)
        self.assertIn("captured_during_exact_live_a0_rollout", independent)

        compute = inspect.getsource(study._compute_diagnostic_cell)
        capture = compute.index("diagnostic_capture=a0_capture")
        bridge = compute.index("_assert_trace_subset_close", capture)
        diagnostics = compute.index("_independent_noise_result", bridge)
        self.assertLess(capture, bridge)
        self.assertLess(bridge, diagnostics)

    def test_diagnostic_primer_and_each_reader_execute_one_full_cell(self) -> None:
        a0_writer = inspect.getsource(study.prime_diagnostic_a0_cache)
        primer = inspect.getsource(study.prime_diagnostic_cell)
        creation = inspect.getsource(study.run_diagnostic_cell)
        verification = inspect.getsource(study.verify_diagnostic_cell)
        self.assertEqual(a0_writer.count("_prime_diagnostic_a0_only("), 1)
        self.assertNotIn("_compute_diagnostic_cell(", a0_writer)
        self.assertLess(
            primer.index("_current_diagnostic_a0_primer_reference"),
            primer.index("_compute_diagnostic_cell"),
        )
        self.assertNotIn("_prime_diagnostic_a0_only", primer)
        self.assertEqual(creation.count("_compute_diagnostic_cell("), 1)
        self.assertEqual(verification.count("_compute_diagnostic_cell("), 1)
        self.assertNotIn("_compute_diagnostic_cell_after_discarded_warmup", creation)
        self.assertNotIn(
            "_compute_diagnostic_cell_after_discarded_warmup", verification
        )

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

    def test_evaluation_primer_is_discard_only_for_new_and_retry_states(self) -> None:
        commit = "c" * 40
        cell = {
            "index": 4,
            "cell_id": "evaluation-O1_recursive_action_sequence_gradient-211-311",
            "arm": "O1_recursive_action_sequence_gradient",
            "result_path": "evaluation/cell/result.json",
            "trace_path": "evaluation/cell/traces.npz",
            "creation_cache_seal_path": "verified/cache-seals/cell-creation.json",
            "verification_cache_seal_path": (
                "verified/cache-seals/cell-verification.json"
            ),
            "marker_path": "verified/cell.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "evaluation_cells": [cell],
        }
        scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 4,
        }
        computed = (
            {"finite": 1.0},
            {"actions": np.asarray([[0.25]], dtype=np.float32)},
            {"timed_steps": 1.0},
        )
        for mode in ("new", "verification_only"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_json_atomic(root / "manifest.json", manifest)
                if mode == "verification_only":
                    result_path = root / cell["result_path"]
                    trace_path = root / cell["trace_path"]
                    result_path.parent.mkdir(parents=True)
                    result_path.write_bytes(b"immutable-result")
                    trace_path.write_bytes(b"immutable-trace")
                    creation_seal_path = root / cell["creation_cache_seal_path"]
                    creation_seal_path.parent.mkdir(parents=True)
                    creation_seal_path.write_bytes(b"immutable-creation-seal")
                cache = (
                    root
                    / "jax-compilation-cache"
                    / "actor-gap-roadmap"
                    / commit
                    / "job-12345-task-4"
                )
                cache.mkdir(parents=True)
                (cache / "compiled-entry").write_bytes(b"fake compiled executable")
                execution = (
                    scheduler,
                    {"receipt": "read-only"},
                    {"jax_compilation_cache_dir": str(cache)},
                )
                with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                    study,
                    "_validate_retained_submission_authorization",
                    return_value={"mode": mode},
                ), mock.patch.object(
                    study, "_validate_stage_marker"
                ), mock.patch.object(
                    study, "_live_array_execution_binding", return_value=execution
                ), mock.patch.object(
                    study, "_compute_evaluation_cell", return_value=computed
                ), mock.patch.dict(
                    os.environ,
                    {"JAX_COMPILATION_CACHE_DIR": str(cache)},
                    clear=False,
                ):
                    result = study.prime_evaluation_cell(
                        root,
                        4,
                        submission_authorization={"authorized": True},
                    )
                self.assertEqual(result["status"], "discarded_without_cell_publication")
                self.assertTrue(result["publication_artifacts_unchanged"])
                self.assertEqual(
                    result["publication_snapshot_before"],
                    result["publication_snapshot_after"],
                )
                self.assertFalse((root / cell["marker_path"]).exists())
                if mode == "new":
                    self.assertFalse((root / cell["result_path"]).exists())
                    self.assertFalse((root / cell["trace_path"]).exists())
                    self.assertFalse((root / cell["creation_cache_seal_path"]).exists())
                else:
                    self.assertEqual(
                        (root / cell["result_path"]).read_bytes(), b"immutable-result"
                    )
                    self.assertEqual(
                        (root / cell["trace_path"]).read_bytes(), b"immutable-trace"
                    )
                    self.assertEqual(
                        (root / cell["creation_cache_seal_path"]).read_bytes(),
                        b"immutable-creation-seal",
                    )
                self.assertFalse((root / cell["verification_cache_seal_path"]).exists())

    def test_evaluation_primer_rejects_attempted_publication(self) -> None:
        commit = "c" * 40
        cell = {
            "index": 4,
            "cell_id": "evaluation-O1_recursive_action_sequence_gradient-211-311",
            "arm": "O1_recursive_action_sequence_gradient",
            "result_path": "evaluation/cell/result.json",
            "trace_path": "evaluation/cell/traces.npz",
            "creation_cache_seal_path": "verified/cache-seals/cell-creation.json",
            "verification_cache_seal_path": (
                "verified/cache-seals/cell-verification.json"
            ),
            "marker_path": "verified/cell.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "evaluation_cells": [cell],
        }
        scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "manifest.json", manifest)
            cache = (
                root
                / "jax-compilation-cache"
                / "actor-gap-roadmap"
                / commit
                / "job-12345-task-4"
            )
            cache.mkdir(parents=True)
            (cache / "compiled-entry").write_bytes(b"fake compiled executable")
            execution = (
                scheduler,
                {"receipt": "read-only"},
                {"jax_compilation_cache_dir": str(cache)},
            )
            result_path = root / cell["result_path"]

            def publish_during_compute(*_args: object) -> tuple[dict, dict, dict]:
                result_path.parent.mkdir(parents=True)
                result_path.write_bytes(b"primer-must-not-publish-this")
                return (
                    {"finite": 1.0},
                    {"actions": np.asarray([[0.25]], dtype=np.float32)},
                    {"timed_steps": 1.0},
                )

            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value={"mode": "new"},
            ), mock.patch.object(study, "_validate_stage_marker"), mock.patch.object(
                study, "_live_array_execution_binding", return_value=execution
            ), mock.patch.object(
                study, "_compute_evaluation_cell", side_effect=publish_during_compute
            ), mock.patch.dict(
                os.environ,
                {"JAX_COMPILATION_CACHE_DIR": str(cache)},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    ValueError, "primer changed publication artifacts"
                ):
                    study.prime_evaluation_cell(
                        root,
                        4,
                        submission_authorization={"authorized": True},
                    )
            self.assertEqual(result_path.read_bytes(), b"primer-must-not-publish-this")
            self.assertFalse((root / cell["trace_path"]).exists())
            self.assertFalse((root / cell["marker_path"]).exists())

    def test_diagnostic_a0_writer_has_its_own_discard_only_receipt(self) -> None:
        commit = "c" * 40
        cell = {
            "index": 1,
            "cell_id": "diagnostic-223",
            "world_model_seed": 223,
            "result_path": "diagnostics/diagnostic-223/result.json",
            "trace_path": "diagnostics/diagnostic-223/traces.npz",
            "creation_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-creation.json"
            ),
            "verification_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-verification.json"
            ),
            "marker_path": "verified/diagnostic-223.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "diagnostic_cells": [cell],
        }
        scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 1,
        }
        prepass = {
            "actor_seeds": list(study.ACTOR_SEEDS),
            "trace_sha256": ["a" * 64, "b" * 64],
            "wall_seconds": 1.0,
            "dependency_equality_gate_applied": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "manifest.json", manifest)
            cache = (
                root
                / "jax-compilation-cache"
                / "actor-gap-roadmap"
                / commit
                / "job-12345-task-1"
            )
            cache.mkdir(parents=True)
            (cache / "compiled-entry").write_bytes(b"compiled")
            execution = (
                scheduler,
                {"receipt": "writer"},
                {"jax_compilation_cache_dir": str(cache)},
            )
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value={"mode": "new"},
            ), mock.patch.object(
                study,
                "_validate_calibration_marker",
                return_value={"training_only": True},
            ), mock.patch.object(
                study, "_live_array_execution_binding", return_value=execution
            ), mock.patch.object(
                study, "_prime_diagnostic_a0_only", return_value=prepass
            ), mock.patch.dict(
                os.environ,
                {"JAX_COMPILATION_CACHE_DIR": str(cache)},
                clear=False,
            ):
                receipt = study.prime_diagnostic_a0_cache(
                    root,
                    1,
                    submission_authorization={"authorized": True},
                )
            self.assertEqual(
                receipt["schema_version"],
                study.DIAGNOSTIC_A0_PRIMER_RECEIPT_SCHEMA,
            )
            self.assertEqual(receipt["actor_seeds"], list(study.ACTOR_SEEDS))
            self.assertEqual(receipt["trace_sha256"], ["a" * 64, "b" * 64])
            self.assertFalse(receipt["dependency_equality_gate_applied"])
            self.assertTrue(receipt["publication_artifacts_unchanged"])
            self.assertTrue((root / receipt["receipt_path"]).is_file())
            for key in (
                "result_path",
                "trace_path",
                "creation_cache_seal_path",
                "verification_cache_seal_path",
                "marker_path",
            ):
                self.assertFalse((root / cell[key]).exists())

    def test_diagnostic_full_primer_requires_external_a0_and_is_discard_only(
        self,
    ) -> None:
        commit = "c" * 40
        cell = {
            "index": 1,
            "cell_id": "diagnostic-223",
            "world_model_seed": 223,
            "result_path": "diagnostics/diagnostic-223/result.json",
            "trace_path": "diagnostics/diagnostic-223/traces.npz",
            "creation_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-creation.json"
            ),
            "verification_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-verification.json"
            ),
            "marker_path": "verified/diagnostic-223.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "diagnostic_cells": [cell],
        }
        scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 1,
        }
        computed = (
            {"finite": 1.0},
            {"actions": np.asarray([[0.25]], dtype=np.float32)},
        )
        events: list[str] = []

        def full_diagnostic(*_args: object) -> tuple[dict, dict]:
            events.append("full")
            return computed

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "manifest.json", manifest)
            cache = (
                root
                / "jax-compilation-cache"
                / "actor-gap-roadmap"
                / commit
                / "job-12345-task-1"
            )
            cache.mkdir(parents=True)
            (cache / "compiled-entry").write_bytes(b"compiled")
            execution = (
                scheduler,
                {"receipt": "primer"},
                {"jax_compilation_cache_dir": str(cache)},
            )
            a0_path = root / "runtime/diagnostic-a0-primers/job-12345-task-1.json"
            write_json_atomic(
                a0_path,
                {
                    "actor_seeds": list(study.ACTOR_SEEDS),
                    "trace_sha256": ["a" * 64, "b" * 64],
                    "dependency_equality_gate_applied": False,
                },
            )
            a0_reference = {
                "path": str(a0_path.relative_to(root)),
                "file_sha256": study.benchmark.file_sha256(a0_path),
                "receipt_sha256": "r" * 64,
                "discarded_wall_seconds": 1.0,
                "cache_tree_sha256_after_a0_prepass": "c" * 64,
            }
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value={"mode": "new"},
            ), mock.patch.object(
                study,
                "_validate_calibration_marker",
                return_value={"training_only": True},
            ), mock.patch.object(
                study, "_live_array_execution_binding", return_value=execution
            ), mock.patch.object(
                study,
                "_current_diagnostic_a0_primer_reference",
                return_value=a0_reference,
            ), mock.patch.object(
                study, "_compute_diagnostic_cell", side_effect=full_diagnostic
            ), mock.patch.dict(
                os.environ,
                {"JAX_COMPILATION_CACHE_DIR": str(cache)},
                clear=False,
            ):
                receipt = study.prime_diagnostic_cell(
                    root,
                    1,
                    submission_authorization={"authorized": True},
                )
            self.assertEqual(events, ["full"])
            self.assertEqual(
                receipt["a0_only_prepass_actor_seeds"], list(study.ACTOR_SEEDS)
            )
            self.assertEqual(
                receipt["a0_only_prepass_trace_sha256"], ["a" * 64, "b" * 64]
            )
            self.assertTrue(receipt["a0_only_prepass_before_full_diagnostic"])
            self.assertFalse(
                receipt["a0_only_prepass_dependency_equality_gate_applied"]
            )
            self.assertTrue(receipt["a0_only_prepass_separate_process"])
            self.assertEqual(receipt["a0_only_prepass_receipt"], a0_reference)
            self.assertTrue(receipt["publication_artifacts_unchanged"])
            self.assertEqual(
                receipt["receipt_sha256"],
                study._unsigned_digest(receipt, "receipt_sha256"),
            )
            for key in (
                "result_path",
                "trace_path",
                "creation_cache_seal_path",
                "verification_cache_seal_path",
                "marker_path",
            ):
                self.assertFalse((root / cell[key]).exists())

    def test_cache_mutation_fails_before_evaluation_bundle_publication(self) -> None:
        commit = "c" * 40
        cell = {
            "index": 4,
            "cell_id": "evaluation-O1_recursive_action_sequence_gradient-211-311",
            "arm": "O1_recursive_action_sequence_gradient",
            "result_path": "evaluation/cell/result.json",
            "trace_path": "evaluation/cell/traces.npz",
            "creation_cache_seal_path": "verified/cache-seals/cell-creation.json",
            "verification_cache_seal_path": (
                "verified/cache-seals/cell-verification.json"
            ),
            "marker_path": "verified/cell.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "evaluation_cells": [cell],
        }
        scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "verified").mkdir()
            (root / "verified/stage-models.json").write_bytes(b"authenticated")
            cache = (
                root
                / "jax-compilation-cache"
                / "actor-gap-roadmap"
                / commit
                / "job-12345-task-4"
            )
            cache.mkdir(parents=True)
            runtime = {"jax_compilation_cache_dir": str(cache)}
            execution = (scheduler, {"receipt": "creation"}, runtime)
            computed = (
                {"finite": 1.0},
                {"actions": np.asarray([[0.25]], dtype=np.float32)},
                {"timed_steps": 1.0},
            )
            with mock.patch.object(
                study, "write_manifest", return_value=manifest
            ), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value={"mode": "new"},
            ), mock.patch.object(
                study, "_live_array_execution_binding", return_value=execution
            ), mock.patch.object(
                study,
                "_evaluation_cache_reader_certificate",
                return_value={"cache": "reader"},
            ), mock.patch.object(
                study,
                "_current_evaluation_primer_receipt_reference",
                return_value={"receipt_sha256": "p" * 64},
            ), mock.patch.object(
                study, "_validate_stage_marker"
            ), mock.patch.object(
                study, "_compute_evaluation_cell", return_value=computed
            ) as compute, mock.patch.object(
                study,
                "_assert_evaluation_cache_sealed",
                side_effect=ValueError("evaluation cache changed after primer"),
            ) as seal:
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    study.run_evaluation_cell(
                        "/unused-dependency",
                        root,
                        4,
                        submission_authorization={"authorized": True},
                    )
            compute.assert_called_once()
            seal.assert_called_once()
            self.assertFalse((root / cell["result_path"]).exists())
            self.assertFalse((root / cell["trace_path"]).exists())
            self.assertFalse((root / cell["creation_cache_seal_path"]).exists())
            self.assertFalse((root / cell["verification_cache_seal_path"]).exists())
            self.assertFalse((root / cell["marker_path"]).exists())

    def test_cache_mutation_fails_before_evaluation_marker_publication(self) -> None:
        commit = "c" * 40
        cell = {
            "index": 4,
            "cell_id": "evaluation-F0_uniform_prior_unconstrained-211-311",
            "arm": "F0_uniform_prior_unconstrained",
            "world_model_seed": 211,
            "actor_seed": 311,
            "evaluation_seeds": [401],
            "result_path": "evaluation/cell/result.json",
            "trace_path": "evaluation/cell/traces.npz",
            "creation_cache_seal_path": "verified/cache-seals/cell-creation.json",
            "verification_cache_seal_path": (
                "verified/cache-seals/cell-verification.json"
            ),
            "marker_path": "verified/cell.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "evaluation_cells": [cell],
        }
        creation_scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 4,
        }
        verification_scheduler = {
            "job_id": "99999",
            "array_job_id": "99000",
            "array_task_id": 4,
        }
        creation_authorization = {"mode": "new"}
        verification_authorization = {"mode": "verification_only"}
        creation_receipt = {"receipt": "creation"}
        creation_runtime = {"jax_compilation_cache_dir": "/cache/job-12345-task-4"}
        verification_runtime = {"jax_compilation_cache_dir": "/cache/job-99999-task-4"}
        creation_certificate = {"cache": "creation-reader"}
        verification_certificate = {"cache": "verification-reader"}
        creation_primer_receipt = {"receipt_sha256": "c" * 64}
        verification_primer_receipt = {"receipt_sha256": "v" * 64}
        replay_core = {
            "schema_version": study.EVALUATION_SCHEMA,
            "status": "complete",
            "source_commit": commit,
            "manifest_sha256": manifest["manifest_sha256"],
            "cell_id": cell["cell_id"],
            "cell_index": 4,
            "task": study.TASK,
            "arm": cell["arm"],
            "controller_family": "uniform_prior",
            "world_model_seed": 211,
            "actor_seed": 311,
            "evaluation_seeds": [401],
            "episode_returns": [0.0],
            "mean_return": 0.0,
            "normalized_mean_return": 0.0,
            "action_saturation_fraction": 0.0,
            "mean_objective_improvement": 0.0,
            "nonnegative_objective_improvement_fraction": 1.0,
            "mean_gradient_norm": 0.0,
            "mean_parameter_delta": 0.0,
            "mean_anchor_action_drift": 0.0,
            "mean_current_action_drift": 0.0,
            "reference_budget_violation_fraction": 0.0,
            "mean_trust_backtracks": 0.0,
            "frozen_reference_fallback_fraction": 0.0,
            "mean_heldout_acceptance_improvement": 0.0,
            "heldout_acceptance_fraction": 1.0,
            "acceptance_fraction": 1.0,
            "behavior_filter_fallback_fraction": 0.0,
            "executed_behavior_violation_fraction": 0.0,
            "mean_behavior_distance": 0.0,
            "objective_evaluations_per_step": [1],
            "coverage": {"fraction": 1.0},
            "imagined_coverage": {"fraction": 1.0},
            "imagined_coverage_protocol": "test-only",
            "a0_dependency_replay_verified": False,
            "source_reward_checkpoint_sha256": "r" * 64,
            "source_rebrac_checkpoint_sha256": "b" * 64,
            "model_checkpoint_sha256": "w" * 64,
            "calibration_arrays_file_sha256": "a" * 64,
            "dependency_trace_file_sha256": "t" * 64,
            "discarded_pure_compile_warmup": True,
        }
        core = {
            **replay_core,
            "creation_cache_reader_certificate": creation_certificate,
            "creation_primer_receipt": creation_primer_receipt,
            "creation_submission_authorization": creation_authorization,
            "creation_scheduler_provenance": creation_scheduler,
            "creation_submission_receipt": creation_receipt,
        }
        trace = {"actions": np.asarray([[[0.25, -0.25]]], dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "manifest.json", manifest)
            trace_path = root / cell["trace_path"]
            trace_path.parent.mkdir(parents=True)
            np.savez(trace_path, **trace)
            result = {
                **core,
                "core_sha256": study.benchmark.object_sha256(core),
                "trace_file_sha256": study.benchmark.file_sha256(trace_path),
                "trace_sha256": study.benchmark.array_sha256(trace),
                "timing": {"timed_steps": 1.0},
                "wall_seconds": 1.0,
                "runtime": creation_runtime,
            }
            write_json_atomic(root / cell["result_path"], result)

            verification_execution = (
                verification_scheduler,
                {"receipt": "verification"},
                verification_runtime,
            )
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value=verification_authorization,
            ), mock.patch.object(
                study,
                "_live_array_execution_binding",
                return_value=verification_execution,
            ), mock.patch.object(
                study,
                "_evaluation_cache_reader_certificate",
                return_value=verification_certificate,
            ), mock.patch.object(
                study,
                "_current_evaluation_primer_receipt_reference",
                return_value=verification_primer_receipt,
            ), mock.patch.object(
                study,
                "_validate_retained_creation_binding",
                return_value=(
                    creation_authorization,
                    creation_scheduler,
                    creation_receipt,
                    creation_runtime,
                ),
            ), mock.patch.object(
                study, "_validate_evaluation_cache_reader_certificate"
            ), mock.patch.object(
                study, "_validate_evaluation_primer_receipt_reference"
            ), mock.patch.object(
                study,
                "_validate_evaluation_cache_seal",
                return_value={"seal_sha256": "s" * 64},
            ), mock.patch.object(
                study, "_require_distinct_replay_processes"
            ), mock.patch.object(
                study,
                "_compute_evaluation_cell",
                return_value=(replay_core, trace, {"timed_steps": 1.0}),
            ) as compute, mock.patch.object(
                study,
                "_assert_evaluation_cache_sealed",
                side_effect=ValueError("evaluation cache changed after primer"),
            ) as seal, mock.patch.object(
                study, "_write_verified_marker"
            ) as publish_marker:
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    study.verify_evaluation_cell(
                        root,
                        4,
                        strict_replay=True,
                        submission_authorization={"authorized": True},
                    )
            compute.assert_called_once()
            seal.assert_called_once()
            publish_marker.assert_not_called()
            self.assertFalse((root / cell["marker_path"]).exists())

    def test_cache_mutation_fails_before_diagnostic_bundle_publication(self) -> None:
        commit = "c" * 40
        cell = {
            "index": 1,
            "cell_id": "diagnostic-223",
            "world_model_seed": 223,
            "result_path": "diagnostics/diagnostic-223/result.json",
            "trace_path": "diagnostics/diagnostic-223/traces.npz",
            "creation_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-creation.json"
            ),
            "verification_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-verification.json"
            ),
            "marker_path": "verified/diagnostic-223.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "diagnostic_cells": [cell],
        }
        scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 1,
        }
        computed = (
            {
                "schema_version": study.DIAGNOSTIC_SCHEMA,
                "status": "complete",
                "finite": 1.0,
            },
            {"actions": np.asarray([[0.25]], dtype=np.float32)},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution = (
                scheduler,
                {"receipt": "creation"},
                {"jax_compilation_cache_dir": "/cache/job-12345-task-1"},
            )
            with mock.patch.object(
                study, "write_manifest", return_value=manifest
            ), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value={"mode": "new"},
            ), mock.patch.object(
                study, "_live_array_execution_binding", return_value=execution
            ), mock.patch.object(
                study,
                "_evaluation_cache_reader_certificate",
                return_value={"cache": "reader"},
            ), mock.patch.object(
                study,
                "_current_evaluation_primer_receipt_reference",
                return_value={"receipt_sha256": "p" * 64},
            ), mock.patch.object(
                study,
                "_validate_calibration_marker",
                return_value={"training_only": True},
            ), mock.patch.object(
                study, "_compute_diagnostic_cell", return_value=computed
            ) as compute, mock.patch.object(
                study,
                "_assert_evaluation_cache_sealed",
                side_effect=ValueError("diagnostic cache changed after primer"),
            ) as seal:
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    study.run_diagnostic_cell(
                        "/unused-dependency",
                        root,
                        1,
                        submission_authorization={"authorized": True},
                    )
            compute.assert_called_once()
            seal.assert_called_once()
            for key in (
                "result_path",
                "trace_path",
                "creation_cache_seal_path",
                "verification_cache_seal_path",
                "marker_path",
            ):
                self.assertFalse((root / cell[key]).exists())

    def test_existing_evaluation_marker_is_invalid_without_verification_seal(
        self,
    ) -> None:
        cell = {
            "index": 0,
            "cell_id": "evaluation-O1-211-311",
            "result_path": "evaluation/cell/result.json",
            "trace_path": "evaluation/cell/traces.npz",
            "creation_cache_seal_path": "verified/cache-seals/cell-creation.json",
            "verification_cache_seal_path": (
                "verified/cache-seals/cell-verification.json"
            ),
            "marker_path": "verified/cell.json",
        }
        manifest = {
            "source_commit": "c" * 40,
            "manifest_sha256": "m" * 64,
            "evaluation_cells": [cell],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {
                "runtime": {},
                "creation_submission_authorization": {},
                "creation_scheduler_provenance": {},
                "creation_cache_reader_certificate": {},
                "creation_primer_receipt": {},
            }
            write_json_atomic(root / cell["result_path"], result)
            write_json_atomic(root / cell["creation_cache_seal_path"], {"sealed": True})
            marker = {
                "schema_version": study.MARKER_SCHEMA,
                "status": "verified",
                "stage": "evaluation",
                "source_commit": manifest["source_commit"],
                "manifest_sha256": manifest["manifest_sha256"],
                "cell_id": cell["cell_id"],
                "cell_index": 0,
                "result_file_sha256": study.benchmark.file_sha256(
                    root / cell["result_path"]
                ),
                "strict_policy_model_environment_replay": True,
                "strict_replay_wall_seconds": 1.0,
                "verification_submission_authorization": {},
                "verification_scheduler_provenance": {},
                "verification_cache_reader_certificate": {},
                "verification_primer_receipt": {},
                "verification_runtime": {},
                "creation_cache_seal_file_sha256": study.benchmark.file_sha256(
                    root / cell["creation_cache_seal_path"]
                ),
                "creation_cache_seal_sha256": "s" * 64,
            }
            marker["marker_sha256"] = study.benchmark.object_sha256(marker)
            write_json_atomic(root / cell["marker_path"], marker)
            original_seal_validator = study._validate_evaluation_cache_seal
            phases: list[str] = []

            def validate_seal(
                seal_root: Path,
                seal_manifest: dict,
                seal_cell: dict,
                phase: str,
                *,
                stage: str = "evaluation",
            ) -> dict:
                phases.append(phase)
                if phase == "creation":
                    return {"seal_sha256": "s" * 64}
                return original_seal_validator(
                    seal_root, seal_manifest, seal_cell, phase, stage=stage
                )

            scheduler = {
                "job_id": "123",
                "array_job_id": "120",
                "array_task_id": 0,
            }
            with mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                side_effect=[{"mode": "new"}, {"mode": "verification_only"}],
            ), mock.patch.object(
                study, "_validate_scheduler_record", return_value=scheduler
            ), mock.patch.object(
                study, "_validate_array_submission_receipt", return_value={}
            ), mock.patch.object(
                study, "_require_single_gpu_runtime", return_value={}
            ), mock.patch.object(
                study, "_require_preflight_runtime_match"
            ), mock.patch.object(
                study, "_validate_evaluation_cache_reader_certificate"
            ), mock.patch.object(
                study, "_validate_evaluation_primer_receipt_reference"
            ), mock.patch.object(
                study, "_validate_evaluation_cache_seal", side_effect=validate_seal
            ):
                with self.assertRaises(FileNotFoundError):
                    study._validate_cell_marker(root, manifest, cell, "evaluation")
            self.assertEqual(phases, ["creation", "verification"])
            self.assertFalse((root / cell["verification_cache_seal_path"]).exists())

    def test_cache_mutation_fails_before_diagnostic_marker_publication(self) -> None:
        commit = "c" * 40
        cell = {
            "index": 1,
            "cell_id": "diagnostic-223",
            "world_model_seed": 223,
            "result_path": "diagnostics/diagnostic-223/result.json",
            "trace_path": "diagnostics/diagnostic-223/traces.npz",
            "creation_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-creation.json"
            ),
            "verification_cache_seal_path": (
                "verified/cache-seals/diagnostic-223-verification.json"
            ),
            "marker_path": "verified/diagnostic-223.json",
        }
        manifest = {
            "source_commit": commit,
            "manifest_sha256": "m" * 64,
            "diagnostic_cells": [cell],
        }
        creation_scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 1,
        }
        verification_scheduler = {
            "job_id": "99999",
            "array_job_id": "99000",
            "array_task_id": 1,
        }
        creation_authorization = {"mode": "new"}
        verification_authorization = {"mode": "verification_only"}
        creation_runtime = {"jax_compilation_cache_dir": "/cache/job-12345-task-1"}
        verification_runtime = {"jax_compilation_cache_dir": "/cache/job-99999-task-1"}
        base_core = {
            "schema_version": study.DIAGNOSTIC_SCHEMA,
            "status": "complete",
            "source_commit": commit,
            "manifest_sha256": manifest["manifest_sha256"],
            "cell_id": cell["cell_id"],
            "cell_index": 1,
            "task": study.TASK,
            "world_model_seed": 223,
            "actor_rows": [],
            "diagnostic_only_no_selection_or_mutation": True,
        }
        creation_certificate = {"cache": "creation"}
        creation_primer = {"receipt_sha256": "c" * 64}
        core = {
            **base_core,
            "separate_discarded_full_diagnostic_primer": True,
            "diagnostic_primer_a0_actor_seeds": list(study.ACTOR_SEEDS),
            "creation_cache_reader_certificate": creation_certificate,
            "creation_primer_receipt": creation_primer,
            "creation_submission_authorization": creation_authorization,
            "creation_scheduler_provenance": creation_scheduler,
            "creation_submission_receipt": {"receipt": "creation"},
        }
        trace = {"actions": np.asarray([[[0.25, -0.25]]], dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json_atomic(root / "manifest.json", manifest)
            trace_path = root / cell["trace_path"]
            trace_path.parent.mkdir(parents=True)
            np.savez(trace_path, **trace)
            result = {
                **core,
                "core_sha256": study.benchmark.object_sha256(core),
                "trace_file_sha256": study.benchmark.file_sha256(trace_path),
                "trace_sha256": study.benchmark.array_sha256(trace),
                "wall_seconds": 1.0,
                "runtime": creation_runtime,
            }
            write_json_atomic(root / cell["result_path"], result)
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value=verification_authorization,
            ), mock.patch.object(
                study,
                "_live_array_execution_binding",
                return_value=(
                    verification_scheduler,
                    {"receipt": "verification"},
                    verification_runtime,
                ),
            ), mock.patch.object(
                study,
                "_evaluation_cache_reader_certificate",
                return_value={"cache": "verification"},
            ), mock.patch.object(
                study,
                "_current_evaluation_primer_receipt_reference",
                return_value={"receipt_sha256": "v" * 64},
            ), mock.patch.object(
                study,
                "_validate_retained_creation_binding",
                return_value=(
                    creation_authorization,
                    creation_scheduler,
                    {"receipt": "creation"},
                    creation_runtime,
                ),
            ), mock.patch.object(
                study, "_validate_evaluation_cache_reader_certificate"
            ), mock.patch.object(
                study, "_validate_evaluation_primer_receipt_reference"
            ), mock.patch.object(
                study,
                "_validate_evaluation_cache_seal",
                return_value={"seal_sha256": "s" * 64},
            ), mock.patch.object(
                study, "_require_distinct_replay_processes"
            ), mock.patch.object(
                study, "_compute_diagnostic_cell", return_value=(base_core, trace)
            ) as compute, mock.patch.object(
                study,
                "_assert_evaluation_cache_sealed",
                side_effect=ValueError("diagnostic cache changed after primer"),
            ) as cache_check, mock.patch.object(
                study, "_write_verified_marker"
            ) as publish_marker:
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    study.verify_diagnostic_cell(
                        root,
                        1,
                        submission_authorization={"authorized": True},
                    )
            compute.assert_called_once()
            cache_check.assert_called_once()
            publish_marker.assert_not_called()
            self.assertFalse((root / cell["marker_path"]).exists())

    def test_nonfinite_retained_and_replayed_traces_cannot_publish_marker(
        self,
    ) -> None:
        cell = {
            "index": 0,
            "cell_id": "evaluation-F0-211-311",
            "arm": "F0_uniform_prior_unconstrained",
            "world_model_seed": 211,
            "actor_seed": 311,
            "evaluation_seeds": [401],
            "result_path": "evaluation/cell/result.json",
            "trace_path": "evaluation/cell/traces.npz",
            "creation_cache_seal_path": "verified/cache-seals/cell-creation.json",
            "verification_cache_seal_path": (
                "verified/cache-seals/cell-verification.json"
            ),
            "marker_path": "verified/cell.json",
        }
        manifest = {
            "source_commit": "c" * 40,
            "manifest_sha256": "m" * 64,
            "evaluation_cells": [cell],
        }
        creation_authorization = {"mode": "new"}
        verification_authorization = {"mode": "verification_only"}
        creation_scheduler = {"job_id": "1", "array_job_id": "1", "array_task_id": 0}
        verification_scheduler = {
            "job_id": "2",
            "array_job_id": "2",
            "array_task_id": 0,
        }
        creation_receipt = {"receipt": "creation"}
        creation_runtime = {"jax_compilation_cache_dir": "/cache/job-1-task-0"}
        verification_runtime = {"jax_compilation_cache_dir": "/cache/job-2-task-0"}
        creation_certificate = {"cache": "creation"}
        creation_primer = {"receipt_sha256": "c" * 64}
        verification_primer = {"receipt_sha256": "v" * 64}
        replay_core = {
            "schema_version": study.EVALUATION_SCHEMA,
            "status": "complete",
            "source_commit": manifest["source_commit"],
            "manifest_sha256": manifest["manifest_sha256"],
            "cell_id": cell["cell_id"],
            "cell_index": 0,
            "task": study.TASK,
            "arm": cell["arm"],
            "controller_family": "uniform_prior",
            "world_model_seed": 211,
            "actor_seed": 311,
            "evaluation_seeds": [401],
            "episode_returns": [0.0],
            "mean_return": 0.0,
            "normalized_mean_return": 0.0,
            "action_saturation_fraction": 0.0,
            "mean_objective_improvement": 0.0,
            "nonnegative_objective_improvement_fraction": 1.0,
            "mean_gradient_norm": 0.0,
            "mean_parameter_delta": 0.0,
            "mean_anchor_action_drift": 0.0,
            "mean_current_action_drift": 0.0,
            "reference_budget_violation_fraction": 0.0,
            "mean_trust_backtracks": 0.0,
            "frozen_reference_fallback_fraction": 0.0,
            "mean_heldout_acceptance_improvement": 0.0,
            "heldout_acceptance_fraction": 1.0,
            "acceptance_fraction": 1.0,
            "behavior_filter_fallback_fraction": 0.0,
            "executed_behavior_violation_fraction": 0.0,
            "mean_behavior_distance": 0.0,
            "objective_evaluations_per_step": [1],
            "coverage": {"fraction": 1.0},
            "imagined_coverage": {"fraction": 1.0},
            "imagined_coverage_protocol": "test-only",
            "a0_dependency_replay_verified": False,
            "source_reward_checkpoint_sha256": "r" * 64,
            "source_rebrac_checkpoint_sha256": "b" * 64,
            "model_checkpoint_sha256": "w" * 64,
            "calibration_arrays_file_sha256": "a" * 64,
            "dependency_trace_file_sha256": "t" * 64,
            "discarded_pure_compile_warmup": True,
        }

        def write_retained(root: Path, retained_trace: dict) -> None:
            trace_path = root / cell["trace_path"]
            trace_path.parent.mkdir(parents=True)
            np.savez(trace_path, **retained_trace)
            core = {
                **replay_core,
                "creation_cache_reader_certificate": creation_certificate,
                "creation_primer_receipt": creation_primer,
                "creation_submission_authorization": creation_authorization,
                "creation_scheduler_provenance": creation_scheduler,
                "creation_submission_receipt": creation_receipt,
            }
            result = {
                **core,
                "core_sha256": study.benchmark.object_sha256(core),
                "trace_file_sha256": study.benchmark.file_sha256(trace_path),
                "trace_sha256": study.benchmark.array_sha256(retained_trace),
                "timing": {"timed_steps": 1.0},
                "wall_seconds": 1.0,
                "runtime": creation_runtime,
            }
            write_json_atomic(root / cell["result_path"], result)

        finite_trace = {"actions": np.asarray([[[0.25, -0.25]]], dtype=np.float32)}
        nonfinite_trace = {"actions": np.asarray([[[np.nan, -0.25]]], dtype=np.float32)}
        for label, retained_trace, replay_trace, error, message in (
            (
                "retained",
                nonfinite_trace,
                finite_trace,
                ValueError,
                "trace payload differs",
            ),
            (
                "replay",
                finite_trace,
                nonfinite_trace,
                FloatingPointError,
                "strict replay output is not finite",
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_json_atomic(root / "manifest.json", manifest)
                write_retained(root, retained_trace)
                verification_execution = (
                    verification_scheduler,
                    {"receipt": "verification"},
                    verification_runtime,
                )
                with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                    study,
                    "_validate_retained_submission_authorization",
                    return_value=verification_authorization,
                ), mock.patch.object(
                    study,
                    "_live_array_execution_binding",
                    return_value=verification_execution,
                ), mock.patch.object(
                    study,
                    "_evaluation_cache_reader_certificate",
                    return_value={"cache": "verification"},
                ), mock.patch.object(
                    study,
                    "_current_evaluation_primer_receipt_reference",
                    return_value=verification_primer,
                ), mock.patch.object(
                    study,
                    "_validate_retained_creation_binding",
                    return_value=(
                        creation_authorization,
                        creation_scheduler,
                        creation_receipt,
                        creation_runtime,
                    ),
                ), mock.patch.object(
                    study, "_validate_evaluation_cache_reader_certificate"
                ), mock.patch.object(
                    study, "_validate_evaluation_primer_receipt_reference"
                ), mock.patch.object(
                    study, "_require_distinct_replay_processes"
                ), mock.patch.object(
                    study,
                    "_validate_evaluation_cache_seal",
                    return_value={"seal_sha256": "s" * 64},
                ) as validate_seal, mock.patch.object(
                    study,
                    "_compute_evaluation_cell",
                    return_value=(replay_core, replay_trace, {"timed_steps": 1.0}),
                ) as compute, mock.patch.object(
                    study, "_assert_evaluation_cache_sealed"
                ) as assert_sealed, mock.patch.object(
                    study, "_write_verified_marker"
                ) as publish_marker:
                    with self.assertRaisesRegex(error, message):
                        study.verify_evaluation_cell(
                            root,
                            0,
                            strict_replay=True,
                            submission_authorization={"authorized": True},
                        )
                if label == "retained":
                    validate_seal.assert_not_called()
                    compute.assert_not_called()
                else:
                    validate_seal.assert_called_once()
                    compute.assert_called_once()
                assert_sealed.assert_not_called()
                publish_marker.assert_not_called()
                self.assertFalse((root / cell["marker_path"]).exists())

    def test_same_array_task_ignores_only_process_and_allocation_snapshots(
        self,
    ) -> None:
        primer = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 4,
            "node_name": "firehorse01",
            "boot_id": "boot-a",
            "process_id": 101,
            "process_start_ticks": 1001,
            "allocation_certificate": {"snapshot": "primer"},
        }
        reader = {
            **primer,
            "process_id": 202,
            "process_start_ticks": 2002,
            "allocation_certificate": {"snapshot": "reader"},
        }
        study._require_same_array_task(primer, reader)

        changed_values = {
            "job_id": "12346",
            "array_job_id": "12001",
            "array_task_id": 5,
            "node_name": "firehorse02",
            "boot_id": "boot-b",
        }
        for field, value in changed_values.items():
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "not one array task"
            ):
                study._require_same_array_task(primer, {**reader, field: value})

    def test_replay_difference_reports_one_ulp_without_relaxing_equality(self) -> None:
        retained = np.asarray([[np.float32(0.2283480018377304)]])
        replay = np.nextafter(retained, np.float32(np.inf))
        self.assertIsNone(
            study._first_trace_replay_difference(
                {"residual_norm": retained}, {"residual_norm": retained.copy()}
            )
        )
        difference = study._first_trace_replay_difference(
            {"residual_norm": retained}, {"residual_norm": replay}
        )
        self.assertEqual(difference["array"], "residual_norm")
        self.assertEqual(difference["index"], [0, 0])
        expected_ulp = float(replay[0, 0]) - float(retained[0, 0])
        self.assertEqual(difference["absolute_difference"], expected_ulp)
        self.assertEqual(difference["maximum_absolute_difference"], expected_ulp)
        self.assertNotEqual(
            difference["retained_element_hex"], difference["replay_element_hex"]
        )
        semantic = study._first_core_replay_difference(
            {"metric": [1.0, 2.0]},
            {"metric": [1.0, np.nextafter(2.0, np.inf)]},
            ("metric",),
        )
        self.assertEqual(semantic["path"], "metric[1]")
        self.assertGreater(semantic["absolute_difference"], 0.0)

    def test_replay_difference_reports_priority_missing_dtype_and_shape(self) -> None:
        retained = {
            "residual_norm": np.asarray([1.0, 2.0], dtype=np.float32),
            "actions": np.asarray([0.0], dtype=np.float32),
        }
        replay = {
            "residual_norm": np.asarray([1.0, 3.0], dtype=np.float32),
            "actions": np.asarray([1.0], dtype=np.float32),
        }
        priority = study._first_trace_replay_difference(retained, replay)
        self.assertEqual(priority["array"], "residual_norm")
        self.assertEqual(priority["index"], [1])

        missing = study._first_trace_replay_difference(
            {"actions": np.asarray([0.0], dtype=np.float32)}, {}
        )
        self.assertEqual(missing["kind"], "missing_array")
        self.assertEqual(missing["array"], "actions")

        dtype = study._first_trace_replay_difference(
            {"actions": np.asarray([0.0], dtype=np.float32)},
            {"actions": np.asarray([0.0], dtype=np.float64)},
        )
        self.assertEqual(dtype["kind"], "descriptor")
        self.assertEqual(dtype["retained_dtype"], "<f4")
        self.assertEqual(dtype["replay_dtype"], "<f8")

        shape = study._first_trace_replay_difference(
            {"actions": np.asarray([0.0], dtype=np.float32)},
            {"actions": np.asarray([[0.0]], dtype=np.float32)},
        )
        self.assertEqual(shape["kind"], "descriptor")
        self.assertEqual(shape["retained_shape"], [1])
        self.assertEqual(shape["replay_shape"], [1, 1])

        core_equal = {"metric": [1.0, {"nested": True}]}
        self.assertIsNone(
            study._first_core_replay_difference(
                core_equal, {"metric": [1.0, {"nested": True}]}, ("metric",)
            )
        )
        core_missing = study._first_core_replay_difference(
            {"metric": 1.0}, {}, ("metric",)
        )
        self.assertEqual(core_missing["kind"], "missing_key")
        self.assertEqual(core_missing["path"], "metric")
        core_extra = study._first_core_replay_difference(
            {"metric": 1.0}, {"metric": 1.0, "unexpected": 2.0}, ("metric",)
        )
        self.assertEqual(core_extra["kind"], "missing_key")
        self.assertEqual(core_extra["path"], "unexpected")
        self.assertFalse(core_extra["retained_present"])
        self.assertTrue(core_extra["replay_present"])

    def test_retained_cache_reader_certificate_uses_retained_runtime_path(self) -> None:
        commit = "c" * 40
        manifest = {"source_commit": commit, "manifest_sha256": "m" * 64}
        cell = {"cell_id": "cell-4", "index": 4}
        scheduler = {
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 4,
        }
        cache = f"/work2/cache/actor-gap-roadmap/{commit}/job-12345-task-4"
        certificate = {
            "schema_version": study.EVALUATION_CACHE_READER_SCHEMA,
            "status": "primed_cache_reader",
            "source_commit": commit,
            "manifest_sha256": manifest["manifest_sha256"],
            "cell_id": cell["cell_id"],
            "cell_index": 4,
            "job_id": "12345",
            "array_job_id": "12000",
            "array_task_id": 4,
            "cache_directory": cache,
            "cache_tree_sha256_after_discarded_primer": "a" * 64,
            "separate_discard_only_primer_process": True,
        }
        runtime = {"jax_compilation_cache_dir": cache}
        with mock.patch.dict(
            os.environ,
            {
                "JAX_COMPILATION_CACHE_DIR": (
                    f"/work2/cache/actor-gap-roadmap/{commit}/job-99999-task-4"
                )
            },
            clear=False,
        ):
            self.assertEqual(
                study._validate_evaluation_cache_reader_certificate(
                    certificate, manifest, cell, scheduler, runtime
                ),
                certificate,
            )

            retry_scheduler = {
                "job_id": "99999",
                "array_job_id": "99000",
                "array_task_id": 4,
            }
            retry_runtime = {
                "jax_compilation_cache_dir": (
                    f"/work2/cache/actor-gap-roadmap/{commit}/job-99999-task-4"
                )
            }
            with self.assertRaisesRegex(ValueError, "certificate is invalid"):
                study._validate_evaluation_cache_reader_certificate(
                    certificate,
                    manifest,
                    cell,
                    retry_scheduler,
                    retry_runtime,
                )

            retry_certificate = {
                **certificate,
                "job_id": "99999",
                "array_job_id": "99000",
                "cache_directory": retry_runtime["jax_compilation_cache_dir"],
                "cache_tree_sha256_after_discarded_primer": "b" * 64,
            }
            self.assertEqual(
                study._validate_evaluation_cache_reader_certificate(
                    retry_certificate,
                    manifest,
                    cell,
                    retry_scheduler,
                    retry_runtime,
                ),
                retry_certificate,
            )
            for field, invalid in (
                ("cell_index", 5),
                ("array_task_id", 5),
                ("source_commit", "d" * 40),
                ("cache_tree_sha256_after_discarded_primer", "not-a-digest"),
                ("separate_discard_only_primer_process", False),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "certificate is invalid"
                ):
                    study._validate_evaluation_cache_reader_certificate(
                        {**retry_certificate, field: invalid},
                        manifest,
                        cell,
                        retry_scheduler,
                        retry_runtime,
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
                creation_seal_path = (
                    root / f"verified/cache-seals/cell-{index}-creation.json"
                )
                verification_seal_path = (
                    root / f"verified/cache-seals/cell-{index}-verification.json"
                )
                write_json_atomic(
                    creation_seal_path, {"seal_sha256": f"creation-{index}"}
                )
                write_json_atomic(
                    verification_seal_path,
                    {"seal_sha256": f"verification-{index}"},
                )
                cell = {
                    "index": index,
                    "cell_id": f"cell-{index}",
                    "creation_cache_seal_path": str(
                        creation_seal_path.relative_to(root)
                    ),
                    "verification_cache_seal_path": str(
                        verification_seal_path.relative_to(root)
                    ),
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
                "creation_cache_seal_path": (
                    "verified/cache-seals/diagnostic-211-creation.json"
                ),
                "verification_cache_seal_path": (
                    "verified/cache-seals/diagnostic-211-verification.json"
                ),
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
                write_json_atomic(
                    root / cell["creation_cache_seal_path"], {"sealed": True}
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

    def test_missing_creation_seal_blocks_retry_classification_and_authorization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            cell = {
                "index": 0,
                "cell_id": "evaluation-O1-211-311",
                "result_path": "evaluation/cell/result.json",
                "trace_path": "evaluation/cell/traces.npz",
                "creation_cache_seal_path": ("verified/cache-seals/cell-creation.json"),
                "verification_cache_seal_path": (
                    "verified/cache-seals/cell-verification.json"
                ),
                "marker_path": "verified/cell.json",
            }
            manifest = {
                "source_commit": "c" * 40,
                "manifest_sha256": "m" * 64,
                "evaluation_cells": [cell],
            }
            write_json_atomic(root / "manifest.json", manifest)
            write_json_atomic(root / cell["result_path"], {"complete": True})
            study.benchmark._write_npz_atomic(
                root / cell["trace_path"],
                {"value": np.asarray([1], dtype=np.int32)},
            )

            with self.assertRaisesRegex(ValueError, "partial immutable artifacts"):
                roadmap_submitter._classify_cells(root, manifest, "evaluation")

            upstream = {"stage": "model", "marker_sha256": "u" * 64}
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_submission_upstream_marker", return_value=upstream
            ):
                with self.assertRaisesRegex(
                    ValueError, "verification-only submission artifacts differ"
                ):
                    study.write_submission_map(
                        root,
                        "evaluation",
                        [{"index": 0, "mode": "verification_only"}],
                    )
            self.assertFalse((root / cell["creation_cache_seal_path"]).exists())

    def test_new_and_retry_cache_seals_bind_attempts_and_artifact_hashes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            cell = {
                "index": 0,
                "cell_id": "evaluation-O1-211-311",
                "result_path": "evaluation/cell/result.json",
                "trace_path": "evaluation/cell/traces.npz",
                "creation_cache_seal_path": ("verified/cache-seals/cell-creation.json"),
                "verification_cache_seal_path": (
                    "verified/cache-seals/cell-verification.json"
                ),
                "marker_path": "verified/cell.json",
            }
            manifest = {
                "source_commit": "c" * 40,
                "manifest_sha256": "m" * 64,
                "evaluation_cells": [cell],
            }
            write_json_atomic(root / "manifest.json", manifest)
            upstream = {"stage": "model", "marker_sha256": "u" * 64}
            reader_scheduler = {
                "job_id": "12345",
                "array_job_id": "12000",
                "array_task_id": 0,
                "node_name": "gpu01",
                "boot_id": "boot-a",
                "process_id": 101,
                "process_start_ticks": 1001,
            }
            sealer_scheduler = {
                **reader_scheduler,
                "process_id": 102,
                "process_start_ticks": 1002,
            }
            runtime = {"jax_compilation_cache_dir": "/immutable/job-cache"}
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_submission_upstream_marker", return_value=upstream
            ):
                new_map = study.write_submission_map(
                    root, "evaluation", [{"index": 0, "mode": "new"}]
                )
            self.assertEqual(new_map["attempt"], 1)
            new_map_path = root / new_map["map_path"]
            new_authorization = {
                "map_path": new_map["map_path"],
                "map_file_sha256": study.benchmark.file_sha256(new_map_path),
                "map_sha256": new_map["map_sha256"],
                "mode": "new",
            }

            write_json_atomic(root / cell["result_path"], {"complete": True})
            study.benchmark._write_npz_atomic(
                root / cell["trace_path"],
                {"value": np.asarray([1], dtype=np.int32)},
            )
            with mock.patch.object(
                study,
                "_job_scoped_evaluation_cache",
                return_value=runtime["jax_compilation_cache_dir"],
            ), mock.patch.object(
                study, "_assert_evaluation_cache_sealed", return_value="f" * 64
            ):
                creation_seal = study._evaluation_cache_seal_body(
                    root,
                    manifest,
                    cell,
                    "creation",
                    reader_authorization=new_authorization,
                    reader_scheduler=reader_scheduler,
                    reader_receipt={"receipt": "creation-reader"},
                    reader_runtime=runtime,
                    sealer_authorization=new_authorization,
                    sealer_scheduler=sealer_scheduler,
                    sealer_receipt={"receipt": "creation-sealer"},
                    sealer_runtime=runtime,
                )
            write_json_atomic(root / cell["creation_cache_seal_path"], creation_seal)

            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_submission_upstream_marker", return_value=upstream
            ):
                retry_map = study.write_submission_map(
                    root,
                    "evaluation",
                    [{"index": 0, "mode": "verification_only"}],
                )
                retry_map_path = root / retry_map["map_path"]
                study.validate_submission_map(
                    root,
                    retry_map_path,
                    "evaluation",
                    0,
                    expected_file_sha256=study.benchmark.file_sha256(retry_map_path),
                )
            self.assertEqual(retry_map["attempt"], 2)
            retained = retry_map["entries"][0]["retained_data_files"]
            self.assertEqual(
                [record["path"] for record in retained],
                [
                    cell["result_path"],
                    cell["trace_path"],
                    cell["creation_cache_seal_path"],
                ],
            )
            for record in retained:
                self.assertEqual(
                    record["file_sha256"],
                    study.benchmark.file_sha256(root / record["path"]),
                )

            original_creation_seal = (
                root / cell["creation_cache_seal_path"]
            ).read_bytes()
            write_json_atomic(
                root / cell["creation_cache_seal_path"], {"mutated": True}
            )
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study, "_submission_upstream_marker", return_value=upstream
            ):
                with self.assertRaisesRegex(ValueError, "artifact digests changed"):
                    study.validate_submission_map(
                        root,
                        retry_map_path,
                        "evaluation",
                        0,
                        expected_file_sha256=study.benchmark.file_sha256(
                            retry_map_path
                        ),
                    )
            (root / cell["creation_cache_seal_path"]).write_bytes(
                original_creation_seal
            )

            retry_authorization = {
                "map_path": retry_map["map_path"],
                "map_file_sha256": study.benchmark.file_sha256(retry_map_path),
                "map_sha256": retry_map["map_sha256"],
                "mode": "verification_only",
            }
            write_json_atomic(root / cell["marker_path"], {"verified": True})
            with mock.patch.object(
                study,
                "_job_scoped_evaluation_cache",
                return_value=runtime["jax_compilation_cache_dir"],
            ), mock.patch.object(
                study, "_assert_evaluation_cache_sealed", return_value="f" * 64
            ):
                verification_seal = study._evaluation_cache_seal_body(
                    root,
                    manifest,
                    cell,
                    "verification",
                    reader_authorization=retry_authorization,
                    reader_scheduler=reader_scheduler,
                    reader_receipt={"receipt": "verification-reader"},
                    reader_runtime=runtime,
                    sealer_authorization=retry_authorization,
                    sealer_scheduler=sealer_scheduler,
                    sealer_receipt={"receipt": "verification-sealer"},
                    sealer_runtime=runtime,
                )
            self.assertEqual(
                creation_seal["reader_submission_authorization"]["map_path"],
                "submissions/evaluation-attempt-001.json",
            )
            self.assertEqual(
                verification_seal["reader_submission_authorization"]["map_path"],
                "submissions/evaluation-attempt-002.json",
            )
            self.assertEqual(
                [row["path"] for row in creation_seal["artifact_bindings"]],
                [cell["result_path"], cell["trace_path"]],
            )
            self.assertEqual(
                [row["path"] for row in verification_seal["artifact_bindings"]],
                [cell["result_path"], cell["trace_path"], cell["marker_path"]],
            )
            for body in (creation_seal, verification_seal):
                for binding in body["artifact_bindings"]:
                    self.assertEqual(
                        binding["file_sha256"],
                        study.benchmark.file_sha256(root / binding["path"]),
                    )
                self.assertEqual(
                    body["seal_sha256"],
                    study._unsigned_digest(body, "seal_sha256"),
                )

    def test_cache_sealer_publication_transitions_from_new_to_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = {
                "index": 0,
                "cell_id": "evaluation-O1-211-311",
                "result_path": "evaluation/cell/result.json",
                "trace_path": "evaluation/cell/traces.npz",
                "creation_cache_seal_path": ("verified/cache-seals/cell-creation.json"),
                "verification_cache_seal_path": (
                    "verified/cache-seals/cell-verification.json"
                ),
                "marker_path": "verified/cell.json",
            }
            manifest = {
                "source_commit": "c" * 40,
                "manifest_sha256": "m" * 64,
                "evaluation_cells": [cell],
            }
            write_json_atomic(root / "manifest.json", manifest)
            write_json_atomic(
                root / cell["result_path"],
                {
                    "creation_cache_reader_certificate": {"cache": "creation"},
                    "creation_primer_receipt": {"receipt": "creation"},
                },
            )
            study.benchmark._write_npz_atomic(
                root / cell["trace_path"],
                {"value": np.asarray([1], dtype=np.int32)},
            )
            runtime = {"jax_compilation_cache_dir": "/immutable/job-cache"}
            reader_scheduler = {
                "job_id": "123",
                "array_job_id": "120",
                "array_task_id": 0,
                "node_name": "gpu01",
                "boot_id": "boot-a",
                "process_id": 100,
                "process_start_ticks": 1000,
            }
            sealer_scheduler = {
                **reader_scheduler,
                "process_id": 101,
                "process_start_ticks": 1001,
            }
            new_authorization = {
                "map_path": "submissions/evaluation-attempt-001.json",
                "map_file_sha256": "1" * 64,
                "map_sha256": "2" * 64,
                "mode": "new",
            }
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value=new_authorization,
            ), mock.patch.object(
                study,
                "_live_array_execution_binding",
                return_value=(sealer_scheduler, {"receipt": "sealer"}, runtime),
            ), mock.patch.object(
                study,
                "_validate_retained_creation_binding",
                return_value=(
                    new_authorization,
                    reader_scheduler,
                    {"receipt": "reader"},
                    runtime,
                ),
            ), mock.patch.object(
                study, "_validate_evaluation_cache_reader_certificate"
            ), mock.patch.object(
                study, "_validate_evaluation_primer_receipt_reference"
            ), mock.patch.object(
                study,
                "_job_scoped_evaluation_cache",
                return_value=runtime["jax_compilation_cache_dir"],
            ), mock.patch.object(
                study, "_assert_evaluation_cache_sealed", return_value="f" * 64
            ):
                creation = study.seal_evaluation_cache_reader(
                    root,
                    0,
                    phase="creation",
                    submission_authorization=new_authorization,
                )
                with self.assertRaisesRegex(ValueError, "already exists"):
                    study.seal_evaluation_cache_reader(
                        root,
                        0,
                        phase="creation",
                        submission_authorization=new_authorization,
                    )
            self.assertEqual(creation["phase"], "creation")
            self.assertEqual(creation["reader_submission_authorization"]["mode"], "new")
            self.assertTrue((root / cell["creation_cache_seal_path"]).is_file())

            retry_authorization = {
                "map_path": "submissions/evaluation-attempt-002.json",
                "map_file_sha256": "3" * 64,
                "map_sha256": "4" * 64,
                "mode": "verification_only",
            }
            marker = {
                "schema_version": study.MARKER_SCHEMA,
                "status": "verified",
                "stage": "evaluation",
                "source_commit": manifest["source_commit"],
                "manifest_sha256": manifest["manifest_sha256"],
                "cell_id": cell["cell_id"],
                "cell_index": 0,
                "result_file_sha256": study.benchmark.file_sha256(
                    root / cell["result_path"]
                ),
                "trace_file_sha256": study.benchmark.file_sha256(
                    root / cell["trace_path"]
                ),
                "verification_submission_authorization": retry_authorization,
                "verification_scheduler_provenance": reader_scheduler,
                "verification_cache_reader_certificate": {"cache": "verification"},
                "verification_primer_receipt": {"receipt": "verification"},
                "verification_runtime": runtime,
            }
            marker["marker_sha256"] = study.benchmark.object_sha256(marker)
            write_json_atomic(root / cell["marker_path"], marker)
            with mock.patch.object(study, "validate_manifest"), mock.patch.object(
                study,
                "_validate_retained_submission_authorization",
                return_value=retry_authorization,
            ), mock.patch.object(
                study,
                "_live_array_execution_binding",
                return_value=(sealer_scheduler, {"receipt": "sealer"}, runtime),
            ), mock.patch.object(
                study,
                "_validate_evaluation_cache_seal",
                return_value=creation,
            ), mock.patch.object(
                study, "_validate_scheduler_record", return_value=reader_scheduler
            ), mock.patch.object(
                study,
                "_validate_array_submission_receipt",
                return_value={"receipt": "reader"},
            ), mock.patch.object(
                study, "_require_single_gpu_runtime", return_value=runtime
            ), mock.patch.object(
                study, "_require_preflight_runtime_match"
            ), mock.patch.object(
                study, "_validate_evaluation_cache_reader_certificate"
            ), mock.patch.object(
                study, "_validate_evaluation_primer_receipt_reference"
            ), mock.patch.object(
                study,
                "_job_scoped_evaluation_cache",
                return_value=runtime["jax_compilation_cache_dir"],
            ), mock.patch.object(
                study, "_assert_evaluation_cache_sealed", return_value="f" * 64
            ):
                verification = study.seal_evaluation_cache_reader(
                    root,
                    0,
                    phase="verification",
                    submission_authorization=retry_authorization,
                )
                with self.assertRaisesRegex(ValueError, "already exists"):
                    study.seal_evaluation_cache_reader(
                        root,
                        0,
                        phase="verification",
                        submission_authorization=retry_authorization,
                    )
            self.assertEqual(verification["phase"], "verification")
            self.assertEqual(
                verification["reader_submission_authorization"]["mode"],
                "verification_only",
            )
            self.assertTrue((root / cell["verification_cache_seal_path"]).is_file())
            self.assertEqual(
                [row["path"] for row in verification["artifact_bindings"]],
                [cell["result_path"], cell["trace_path"], cell["marker_path"]],
            )

    def test_array_receipt_binds_live_job_and_exact_map_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            cell = {
                "index": 0,
                "cell_id": "diagnostic-211",
                "result_path": "diagnostics/diagnostic-211/result.json",
                "trace_path": "diagnostics/diagnostic-211/traces.npz",
                "creation_cache_seal_path": (
                    "verified/cache-seals/diagnostic-211-creation.json"
                ),
                "verification_cache_seal_path": (
                    "verified/cache-seals/diagnostic-211-verification.json"
                ),
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
        node = "firehorse01.sc.intern.uni-leipzig.de"
        batch_host = "firehorse01"
        raw = (
            "JobId=125 ArrayJobId=123 ArrayTaskId=0 JobState=RUNNING "
            f"BatchHost={batch_host} Account=dep_inin_dat Partition=gpu-l40s "
            "AllocTRES=cpu=8,mem=64G,node=1,gres/gpu=1,gres/gpu:l40s=1 "
            "TresPerNode=gres/gpu:1"
        )
        environment = {
            "CUDA_VISIBLE_DEVICES": "0",
        }
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            study.os, "uname", return_value=mock.Mock(nodename=node)
        ), mock.patch.object(study.subprocess, "check_output", return_value=raw):
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
        ), mock.patch.object(
            study.os, "uname", return_value=mock.Mock(nodename=node)
        ), mock.patch.object(
            study.subprocess, "check_output", return_value=raw
        ):
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
                study.os, "uname", return_value=mock.Mock(nodename=node)
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
                study.os, "uname", return_value=mock.Mock(nodename=node)
            ), mock.patch.object(
                study.subprocess, "check_output", return_value=raw
            ), self.assertRaisesRegex(
                ValueError, "one-GPU"
            ):
                study._slurm_allocation_certificate(
                    "125", array_job_id="123", array_task_id=0
                )

        self.assertTrue(study._same_slurm_node(batch_host, node))
        self.assertTrue(study._same_slurm_node(node, batch_host))
        self.assertTrue(study._same_slurm_node("FIREHORSE01", node + "."))
        for other in (
            "",
            "firehorse010.sc.intern.uni-leipzig.de",
            "firehorse01.evil.example",
            "firehorse01..sc.intern.uni-leipzig.de",
            "-firehorse01.sc.intern.uni-leipzig.de",
            "firehorse01/sc.intern.uni-leipzig.de",
        ):
            with self.subTest(other=other):
                self.assertFalse(study._same_slurm_node(batch_host, other))
        with self.assertRaisesRegex(ValueError, "one-GPU"):
            with mock.patch.dict(
                os.environ, environment, clear=True
            ), mock.patch.object(
                study.os,
                "uname",
                return_value=mock.Mock(nodename="firehorse02.sc.intern.uni-leipzig.de"),
            ), mock.patch.object(
                study.subprocess, "check_output", return_value=raw
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
        wrong_host_raw = raw.replace("BatchHost=firehorse01", "BatchHost=firehorse02")
        forged_wrong_host = {
            **certificate,
            "raw_record": wrong_host_raw,
            "batch_host": "firehorse02",
            "canonical_record": study._parse_scontrol_record(wrong_host_raw),
        }
        forged_wrong_host["raw_record_sha256"] = study.benchmark.object_sha256(
            forged_wrong_host["raw_record"]
        )
        forged_wrong_host["canonical_record_sha256"] = study.benchmark.object_sha256(
            forged_wrong_host["canonical_record"]
        )
        forged_wrong_host["certificate_sha256"] = study._unsigned_digest(
            forged_wrong_host, "certificate_sha256"
        )
        with self.assertRaisesRegex(ValueError, "allocation certificate"):
            study._validate_slurm_allocation_certificate(
                forged_wrong_host,
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
        commit = "c" * 40
        shared = {
            **runtime,
            "jax_compilation_cache_dir": (
                f"/work2/cache/actor-gap-roadmap/{commit}/shared"
            ),
        }
        job_scoped = {
            **runtime,
            "jax_compilation_cache_dir": (
                f"/work2/cache/actor-gap-roadmap/{commit}/job-12345-task-4"
            ),
        }
        with mock.patch.object(
            study, "_validate_preflight_gate", return_value={"runtime": shared}
        ):
            study._require_preflight_runtime_match(
                Path("/unused"), {}, job_scoped, stage="evaluation"
            )
            study._require_preflight_runtime_match(
                Path("/unused"), {}, shared, stage="model"
            )
            with self.assertRaisesRegex(ValueError, "job-cache scoped"):
                study._require_preflight_runtime_match(
                    Path("/unused"),
                    {},
                    {**job_scoped, "jax_compilation_cache_dir": "/work2/other-cache"},
                    stage="evaluation",
                )
            with self.assertRaisesRegex(ValueError, "shared-cache scoped"):
                study._require_preflight_runtime_match(
                    Path("/unused"), {}, job_scoped, stage="model"
                )

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
