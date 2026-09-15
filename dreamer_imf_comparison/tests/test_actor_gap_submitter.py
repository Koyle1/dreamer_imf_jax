from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SUBMITTER_PATH = (
    SOURCE_ROOT
    / "dreamer_imf_comparison/cluster/actor_gap_roadmap/submit_actor_gap_roadmap.py"
)
SPEC = importlib.util.spec_from_file_location(
    "actor_gap_submitter_tested", SUBMITTER_PATH
)
assert SPEC is not None and SPEC.loader is not None
submitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submitter)


class ActorGapSubmitterTests(unittest.TestCase):
    @staticmethod
    def _terminal_observation(job_id: str, state: str = "COMPLETED") -> dict:
        exit_code = "0:0" if state == "COMPLETED" else "1:0"
        raw = f"{job_id}|{job_id}|{state}|{exit_code}\n"
        summary = "COMPLETED" if state == "COMPLETED" else "FAILED"
        return {
            "job_id": job_id,
            "state": summary,
            "records": [
                {
                    "job_id_raw": job_id,
                    "job_id": job_id,
                    "state": state,
                    "exit_code": exit_code,
                }
            ],
            "raw_output": raw,
            "raw_output_sha256": submitter.study.benchmark.object_sha256(raw),
        }

    @staticmethod
    def _single_marker_fixture(root: Path) -> tuple[SimpleNamespace, Path, dict]:
        source_commit = "c" * 40
        manifest = {"source_commit": source_commit, "manifest_sha256": "m" * 64}
        submitter.study._write_json_exclusive(root / "manifest.json", manifest)
        arguments = SimpleNamespace(
            output_root=root,
            source_commit=source_commit,
            sbatch=Path("/tmp/study.sbatch"),
            source_root=Path("/tmp/source"),
            dependency_root=Path("/tmp/dependency"),
            model_updates=1,
            evaluation_episodes=1,
        )
        intent = submitter._write_intent(
            root, "preflight", arguments, {"slurm_stage": "preflight"}
        )
        receipt_path = submitter._write_receipt(
            intent, "preflight", arguments, {"job": "123"}
        )
        receipt = submitter.study.read_json(receipt_path)
        marker_path = root / "verified/preflight.json"
        marker = {
            "source_commit": source_commit,
            "marker_sha256": "a" * 64,
            "scheduler_provenance": {"job_id": "123"},
            "submission_receipt": {
                "receipt_path": str(receipt_path.relative_to(root)),
                "receipt_file_sha256": submitter.study.benchmark.file_sha256(
                    receipt_path
                ),
                "receipt_sha256": receipt["receipt_sha256"],
            },
        }
        submitter.study._write_json_exclusive(marker_path, marker)
        return arguments, marker_path, marker

    def test_sacct_array_rows_are_aggregated_fail_closed(self) -> None:
        # JobIDRaw is a distinct numeric allocation id on the target cluster;
        # JobID is the field that retains the parent_index array identity.
        completed = "123|123_0|COMPLETED|0:0\n" "124|123_1|COMPLETED|0:0\n"
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=["", completed],
        ) as command:
            self.assertEqual(submitter._slurm_state("123"), "COMPLETED")
        self.assertIn("--array", command.call_args_list[1].args[0])
        self.assertIn(
            "--format=JobIDRaw,JobID,State,ExitCode",
            command.call_args_list[1].args[0],
        )

        failed = "123|123_0|COMPLETED|0:0\n124|123_1|FAILED|1:0\n"
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=["", failed],
        ):
            self.assertEqual(submitter._slurm_state("123"), "FAILED")

        active = "123|123_0|COMPLETED|0:0\n124|123_1|RUNNING|0:0\n"
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=["", active],
        ):
            self.assertEqual(submitter._slurm_state("123"), "RUNNING")

    def test_mixed_squeue_array_states_are_active(self) -> None:
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            return_value="PENDING\nRUNNING\n",
        ):
            self.assertEqual(submitter._slurm_state("123"), "RUNNING")

    def test_absent_squeue_job_falls_back_to_sacct_but_other_errors_propagate(
        self,
    ) -> None:
        absent = subprocess.CalledProcessError(
            1,
            ["squeue"],
            stderr="slurm_load_jobs error: Invalid job id specified",
        )
        completed = "123|123|COMPLETED|0:0\n"
        with mock.patch.object(
            submitter.subprocess, "check_output", side_effect=[absent, completed]
        ) as command:
            self.assertEqual(submitter._slurm_state("123"), "COMPLETED")
        self.assertEqual(len(command.call_args_list), 2)

        unavailable = subprocess.CalledProcessError(
            1, ["squeue"], stderr="Unable to contact slurm controller"
        )
        with mock.patch.object(
            submitter.subprocess, "check_output", side_effect=unavailable
        ), self.assertRaises(subprocess.CalledProcessError):
            submitter._slurm_state("123")

    def test_federated_sbatch_identifier_is_rejected_without_truncation(self) -> None:
        self.assertIsNone(submitter._parse_sbatch_job_id("123;remote-cluster"))
        with mock.patch.object(
            submitter.subprocess, "check_output", return_value="123;remote-cluster\n"
        ), self.assertRaisesRegex(RuntimeError, "invalid job id"):
            submitter._sbatch(Path("/tmp/study.sbatch"), {"AGR_STAGE": "preflight"})

    def test_afterok_verifier_kills_invalid_dependency(self) -> None:
        with mock.patch.object(
            submitter.subprocess, "check_output", return_value="12345\n"
        ) as command:
            job = submitter._sbatch(
                Path("/tmp/study.sbatch"),
                {"AGR_STAGE": "verify-models"},
                dependency_job="12344",
            )
        self.assertEqual(job, "12345")
        invoked = command.call_args.args[0]
        self.assertIn("--dependency=afterok:12344", invoked)
        self.assertIn("--kill-on-invalid-dep=yes", invoked)

    def test_sbatch_has_verifier_only_marker_completion_recovery_stages(self) -> None:
        sbatch = (
            SOURCE_ROOT
            / "dreamer_imf_comparison/cluster/actor_gap_roadmap/actor_gap_roadmap.sbatch"
        ).read_text(encoding="utf-8")
        for stage in ("verify-preflight", "verify-calibration", "verify-final"):
            with self.subTest(stage=stage):
                self.assertIn(f"  {stage})", sbatch)

    def test_nonzero_sbatch_reply_is_immutably_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                output_root=root,
                source_commit="c" * 40,
                sbatch=Path("/tmp/study.sbatch"),
            )
            intent = submitter._write_intent(
                root, "preflight", arguments, {"slurm_stage": "preflight"}
            )
            failure = subprocess.CalledProcessError(1, ["sbatch"])
            with mock.patch.object(submitter, "_sbatch", side_effect=failure):
                with self.assertRaises(subprocess.CalledProcessError):
                    submitter._submit_single_job(
                        intent,
                        "preflight",
                        arguments,
                        {"AGR_STAGE": "preflight"},
                    )
            receipt = submitter.study.read_json(submitter._receipt_path(intent))
            self.assertEqual(receipt["status"], "ambiguous_submission")
            self.assertEqual(receipt["jobs"], {})
            with self.assertRaisesRegex(RuntimeError, "manual resolution"):
                submitter._assert_no_unresolved_submission(root, "preflight")

    def test_preexec_submission_failure_is_immutably_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                output_root=root,
                source_commit="c" * 40,
                sbatch=Path("/tmp/study.sbatch"),
            )
            intent = submitter._write_intent(
                root, "preflight", arguments, {"slurm_stage": "preflight"}
            )
            with mock.patch.object(
                submitter, "_sbatch", side_effect=FileNotFoundError("sbatch")
            ):
                with self.assertRaises(FileNotFoundError):
                    submitter._submit_single_job(
                        intent,
                        "preflight",
                        arguments,
                        {"AGR_STAGE": "preflight"},
                    )
            receipt = submitter.study.read_json(submitter._receipt_path(intent))
            self.assertEqual(receipt["status"], "submission_failed")
            submitter._assert_no_unresolved_submission(root, "preflight")

    def test_malformed_success_output_is_immutably_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                output_root=root,
                source_commit="c" * 40,
                sbatch=Path("/tmp/study.sbatch"),
            )
            intent = submitter._write_intent(
                root, "preflight", arguments, {"slurm_stage": "preflight"}
            )
            with mock.patch.object(
                submitter, "_sbatch", side_effect=RuntimeError("bad job id")
            ):
                with self.assertRaises(RuntimeError):
                    submitter._submit_single_job(
                        intent,
                        "preflight",
                        arguments,
                        {"AGR_STAGE": "preflight"},
                    )
            receipt = submitter.study.read_json(submitter._receipt_path(intent))
            self.assertEqual(receipt["status"], "ambiguous_submission")
            with self.assertRaisesRegex(RuntimeError, "manual resolution"):
                submitter._assert_no_unresolved_submission(root, "preflight")

    def test_successful_single_job_is_held_until_receipt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(
                output_root=root,
                source_commit="c" * 40,
                sbatch=Path("/tmp/study.sbatch"),
            )
            intent = submitter._write_intent(
                root, "preflight", arguments, {"slurm_stage": "preflight"}
            )
            with mock.patch.object(
                submitter, "_sbatch", return_value="123"
            ) as submit, mock.patch.object(
                submitter.subprocess, "check_call"
            ) as release, mock.patch.object(
                submitter,
                "_release_observation",
                side_effect=[
                    {
                        "classification": "user_held",
                        "rows": [{"state": "PENDING", "reason": "JobHeldUser"}],
                        "terminal": None,
                    },
                    {
                        "classification": "active",
                        "rows": [{"state": "PENDING", "reason": "Resources"}],
                        "terminal": None,
                    },
                ],
            ):
                job, receipt_path = submitter._submit_single_job(
                    intent,
                    "preflight",
                    arguments,
                    {"AGR_STAGE": "preflight"},
                )
            self.assertEqual(job, "123")
            self.assertTrue(receipt_path.is_file())
            self.assertTrue(submitter._release_path(receipt_path).is_file())
            self.assertIs(submit.call_args.kwargs["hold"], True)
            release.assert_called_once_with(["scontrol", "release", "123"])

    def test_partial_array_is_cancelled_once_then_idempotently_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(output_root=root, source_commit="c" * 40)
            submitter.study._write_json_exclusive(
                root / "manifest.json",
                {"source_commit": "c" * 40, "manifest_sha256": "m" * 64},
            )
            map_path = root / "submissions/model-attempt-001.json"
            body = {
                "schema_version": submitter.study.SUBMISSION_MAP_SCHEMA,
                "status": "frozen_before_submission",
                "stage": "model",
                "source_commit": "c" * 40,
                "map_sha256": "",
            }
            body["map_sha256"] = submitter._record_digest(body, "map_sha256")
            submitter.study._write_json_exclusive(map_path, body)
            submitter._write_receipt(
                map_path,
                "model",
                arguments,
                {"array": "123"},
                status="partial_submission",
                failure={"boundary": "verifier_sbatch"},
            )
            active = {
                "job_id": "123",
                "state": "RUNNING",
                "queued_output": "PENDING",
            }
            terminal = self._terminal_observation("123", "CANCELLED")
            with mock.patch.object(
                submitter, "_validate_retained_submission_map"
            ), mock.patch.object(
                submitter, "_slurm_observation", side_effect=[active, terminal]
            ), mock.patch.object(
                submitter.subprocess, "check_call"
            ) as cancel:
                submitter._assert_no_unresolved_submission(root, "model")
            cancel.assert_called_once_with(["scancel", "123"])
            receipt_path = root / "submissions/model-attempt-001.receipt.json"
            self.assertTrue(submitter._cancellation_path(receipt_path).is_file())

            # The canonical outcome is sufficient for a second recovery pass;
            # no scheduler call or duplicate cancellation is made.
            with mock.patch.object(
                submitter, "_validate_retained_submission_map"
            ), mock.patch.object(
                submitter, "_slurm_observation"
            ) as observation, mock.patch.object(
                submitter.subprocess, "check_call"
            ) as second_cancel:
                submitter._assert_no_unresolved_submission(root, "model")
            observation.assert_not_called()
            second_cancel.assert_not_called()

    def test_retained_submission_map_rehashes_verification_only_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_commit = "c" * 40
            manifest = {
                "source_commit": source_commit,
                "manifest_sha256": "m" * 64,
                "diagnostic_cells": [
                    {
                        "index": 0,
                        "result_path": "diagnostics/result.json",
                        "trace_path": "diagnostics/trace.npz",
                        "creation_cache_seal_path": (
                            "verified/cache-seals/diagnostic-creation.json"
                        ),
                        "verification_cache_seal_path": (
                            "verified/cache-seals/diagnostic-verification.json"
                        ),
                        "marker_path": "verified/diagnostic.json",
                    }
                ],
            }
            submitter.study._write_json_exclusive(root / "manifest.json", manifest)
            (root / "diagnostics").mkdir()
            (root / "diagnostics/result.json").write_text("result", encoding="utf-8")
            (root / "diagnostics/trace.npz").write_text("trace", encoding="utf-8")
            creation_seal = root / "verified/cache-seals/diagnostic-creation.json"
            creation_seal.parent.mkdir(parents=True)
            creation_seal.write_text("seal", encoding="utf-8")
            cell = manifest["diagnostic_cells"][0]
            entry = {
                "index": 0,
                "mode": "verification_only",
                "retained_data_files": submitter.study._retained_data_file_records(
                    root, cell, "diagnostic"
                ),
            }
            entry["artifact_state_sha256"] = submitter.study._submission_entry_digest(
                entry
            )
            map_path = root / "submissions/diagnostic-attempt-001.json"
            body = {
                "schema_version": submitter.study.SUBMISSION_MAP_SCHEMA,
                "status": "frozen_before_submission",
                "stage": "diagnostic",
                "attempt": 1,
                "source_commit": source_commit,
                "manifest_sha256": manifest["manifest_sha256"],
                "upstream_stage": "calibration",
                "upstream_marker_sha256": "u" * 64,
                "entries": [entry],
                "map_path": str(map_path.relative_to(root)),
            }
            body["map_sha256"] = submitter._record_digest(body, "map_sha256")
            submitter.study._write_json_exclusive(map_path, body)
            arguments = SimpleNamespace(output_root=root, source_commit=source_commit)
            submitter._write_receipt(
                map_path,
                "diagnostic",
                arguments,
                {},
                status="submission_failed",
                failure={"boundary": "array_sbatch"},
            )
            upstream = {"stage": "calibration", "marker_sha256": "u" * 64}
            with mock.patch.object(
                submitter.study, "_submission_upstream_marker", return_value=upstream
            ):
                submitter._assert_no_unresolved_submission(root, "diagnostic")
                rebound = {**body, "source_commit": "d" * 40}
                rebound["map_sha256"] = submitter._record_digest(rebound, "map_sha256")
                with self.assertRaisesRegex(ValueError, "submission map is invalid"):
                    submitter._validate_retained_submission_map(
                        root, "diagnostic", map_path, rebound, manifest
                    )
                (root / "diagnostics/trace.npz").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "digests changed"):
                    submitter._assert_no_unresolved_submission(root, "diagnostic")

    def test_ambiguous_verifier_submission_cancels_known_array_but_stays_manual(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = SimpleNamespace(output_root=root, source_commit="c" * 40)
            submitter.study._write_json_exclusive(
                root / "manifest.json",
                {"source_commit": "c" * 40, "manifest_sha256": "m" * 64},
            )
            map_path = root / "submissions/model-attempt-001.json"
            body = {
                "schema_version": submitter.study.SUBMISSION_MAP_SCHEMA,
                "status": "frozen_before_submission",
                "stage": "model",
                "source_commit": "c" * 40,
                "map_sha256": "",
            }
            body["map_sha256"] = submitter._record_digest(body, "map_sha256")
            submitter.study._write_json_exclusive(map_path, body)
            submitter._write_receipt(
                map_path,
                "model",
                arguments,
                {"array": "123"},
                status="ambiguous_submission",
                failure={"boundary": "verifier_sbatch"},
            )
            active = {
                "job_id": "123",
                "state": "RUNNING",
                "queued_output": "PENDING",
            }
            terminal = self._terminal_observation("123", "CANCELLED")
            with mock.patch.object(
                submitter, "_validate_retained_submission_map"
            ), mock.patch.object(
                submitter, "_slurm_observation", side_effect=[active, terminal]
            ), mock.patch.object(
                submitter.subprocess, "check_call"
            ) as cancel:
                with self.assertRaisesRegex(RuntimeError, "manual resolution"):
                    submitter._assert_no_unresolved_submission(root, "model")
            cancel.assert_called_once_with(["scancel", "123"])

            with mock.patch.object(
                submitter, "_validate_retained_submission_map"
            ), mock.patch.object(submitter.subprocess, "check_call") as second_cancel:
                with self.assertRaisesRegex(RuntimeError, "manual resolution"):
                    submitter._assert_no_unresolved_submission(root, "model")
            second_cancel.assert_not_called()

    def test_mixed_array_with_any_user_held_row_is_not_reconciled_active(self) -> None:
        output = "RUNNING|None\nPENDING|JobHeldUser\n"
        with mock.patch.object(
            submitter.subprocess, "check_output", return_value=output
        ):
            observed = submitter._release_observation("123")
        self.assertEqual(observed["classification"], "user_held")

    def test_lost_release_reply_reconciles_only_after_no_rows_are_held(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments = SimpleNamespace(output_root=root, source_commit="c" * 40)
            intent = submitter._write_intent(
                root, "preflight", arguments, {"slurm_stage": "preflight"}
            )
            receipt_path = submitter._write_receipt(
                intent, "preflight", arguments, {"job": "123"}
            )
            held = {
                "classification": "user_held",
                "rows": [{"state": "PENDING", "reason": "JobHeldUser"}],
                "terminal": None,
            }
            active = {
                "classification": "active",
                "rows": [{"state": "RUNNING", "reason": "None"}],
                "terminal": None,
            }
            error = subprocess.CalledProcessError(1, ["scontrol", "release", "123"])
            with mock.patch.object(
                submitter, "_release_observation", side_effect=[held, active]
            ), mock.patch.object(submitter.subprocess, "check_call", side_effect=error):
                outcome = submitter._release_or_reconcile(receipt_path, "123")
            self.assertEqual(outcome["status"], "reconciled_active")
            self.assertEqual(outcome["after"]["classification"], "active")
            attempts = list(root.glob("submissions/*.release-attempt-*.json"))
            self.assertEqual(len(attempts), 1)

    def test_stage_lock_serializes_check_through_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with submitter._stage_submission_lock(root, "model"):
                with self.assertRaisesRegex(RuntimeError, "owns the model stage lock"):
                    with submitter._stage_submission_lock(root, "model"):
                        self.fail("second submitter entered the protected section")

    def test_marker_completion_attestation_retains_replayable_sacct_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, marker_path, marker = self._single_marker_fixture(root)
            accounting = self._terminal_observation("123")
            with mock.patch.object(
                submitter, "_slurm_observation", return_value=accounting
            ):
                self.assertTrue(
                    submitter._try_attest_marker_completion(
                        root, "preflight", marker_path, marker
                    )
                )
            attestations = list(
                (root / "verified/scheduler-completions").glob("*.json")
            )
            self.assertEqual(len(attestations), 1)
            retained = submitter.study.read_json(attestations[0])
            self.assertEqual(retained["accounting_observation"], accounting)

            # Once persisted, scheduler-accounting retention is no longer a
            # hidden dependency of the evidence chain.
            with mock.patch.object(
                submitter,
                "_slurm_observation",
                side_effect=AssertionError("scheduler should not be queried"),
            ):
                self.assertTrue(
                    submitter._try_attest_marker_completion(
                        root, "preflight", marker_path, marker
                    )
                )

            tampered = dict(retained)
            tampered["accounting_observation"] = {
                **accounting,
                "raw_output": "123|123|FAILED|1:0\n",
            }
            tampered["attestation_sha256"] = submitter._record_digest(
                tampered, "attestation_sha256"
            )
            attestations[0].write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "accounting"):
                submitter._try_attest_marker_completion(
                    root, "preflight", marker_path, marker
                )

    def test_completion_recovery_is_bound_to_exact_immutable_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments, marker_path, marker = self._single_marker_fixture(root)
            fake_receipt = root / "submissions/future.receipt.json"
            with mock.patch.object(
                submitter,
                "_submit_single_job",
                return_value=("124", fake_receipt),
            ):
                payload = submitter._submit_marker_completion_recovery(
                    arguments, "preflight", marker_path, marker
                )
            self.assertEqual(payload["recovery_job"], "124")
            intents = list(
                (root / "submissions").glob("preflight-verifier-*.intent.json")
            )
            self.assertEqual(len(intents), 1)
            intent = submitter.study.read_json(intents[0])
            self.assertEqual(
                intent["payload"],
                {
                    "slurm_stage": "verify-preflight",
                    "reason": "immutable_marker_scheduler_completion_recovery",
                    "mode": "verification_only",
                    "marker_path": str(marker_path.relative_to(root)),
                    "marker_file_sha256": submitter.study.benchmark.file_sha256(
                        marker_path
                    ),
                    "marker_sha256": marker["marker_sha256"],
                },
            )

            recovery_receipt = submitter._write_receipt(
                intents[0],
                "preflight-verifier",
                arguments,
                {"job": "124"},
            )
            self.assertTrue(recovery_receipt.is_file())
            with mock.patch.object(
                submitter,
                "_slurm_observation",
                side_effect=[
                    self._terminal_observation("123", "FAILED"),
                    self._terminal_observation("124"),
                ],
            ):
                self.assertTrue(
                    submitter._try_attest_marker_completion(
                        root, "preflight", marker_path, marker
                    )
                )

    def test_unrelated_completed_recovery_receipt_cannot_attest_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments, marker_path, marker = self._single_marker_fixture(root)
            wrong_intent = submitter._write_intent(
                root,
                "preflight-verifier",
                arguments,
                {
                    "slurm_stage": "verify-preflight",
                    "reason": "immutable_marker_scheduler_completion_recovery",
                    "mode": "verification_only",
                    "marker_path": str(marker_path.relative_to(root)),
                    "marker_file_sha256": submitter.study.benchmark.file_sha256(
                        marker_path
                    ),
                    "marker_sha256": "b" * 64,
                },
            )
            submitter._write_receipt(
                wrong_intent,
                "preflight-verifier",
                arguments,
                {"job": "124"},
            )
            with mock.patch.object(
                submitter,
                "_slurm_observation",
                return_value=self._terminal_observation("123", "FAILED"),
            ) as observation:
                self.assertFalse(
                    submitter._try_attest_marker_completion(
                        root, "preflight", marker_path, marker
                    )
                )
            observation.assert_called_once_with("123")

    def test_preflight_dispatch_recovers_an_existing_unattested_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments, marker_path, marker = self._single_marker_fixture(root)
            arguments.stage = "preflight"
            expected = {"status": "marker_completion_recovery_submitted"}
            with mock.patch.object(
                submitter.study, "validate_preflight_marker", return_value=marker
            ), mock.patch.object(
                submitter, "_try_attest_marker_completion", return_value=False
            ), mock.patch.object(
                submitter,
                "_submit_marker_completion_recovery",
                return_value=expected,
            ) as recovery, redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(submitter._dispatch(arguments), 0)
            recovery.assert_called_once_with(
                arguments, "preflight", marker_path, marker
            )


if __name__ == "__main__":
    unittest.main()
