"""No-Slurm causal submission tests with real immutable marker/intent files.

Only subprocess.check_output and source/dependency authentication are mocked.
Stage marker hashes, receipt binding, exclusive writes and completion records
use the actual implementation against a fresh temporary study for every test.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from dreamer_imf_compare import parallel_study as study
from dreamer_imf_compare.parallel_collection import BudgetLedger, _write_json

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
STAGES = ("preflight", "collect", "fit", "evaluate", "verify")


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submitter = load_script("_parallel_submit_test_subject", "submit_parallel_study.py")
with mock.patch.dict(sys.modules, {"submit_parallel_study": submitter}):
    advancer = load_script(
        "_parallel_advance_test_subject", "advance_parallel_study.py"
    )


def accounting(job="101", *, state="COMPLETED", code="0:0", array=False):
    jobs = [f"{job}_{seed}" for seed in range(3)] if array else [str(job)]
    return "".join(f"{identity}|{state}|{code}|\n" for identity in jobs)


class SubmissionFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="parallel-submission-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "study with spaces"
        self.root.mkdir()
        for folder in ("markers", "submissions", "logs"):
            (self.root / folder).mkdir()
        self.budget = Path(temporary.name) / "native-budget.jsonl"
        with BudgetLedger(self.budget):
            pass
        self.manifest = {"source_commit": "f" * 40, "budget_path": str(self.budget)}
        self.write("manifest.json", self.manifest)
        self.write(
            "report.json",
            {"promotion": {"promote": False}, "diagnostic": "negative control"},
        )
        for module in (submitter, advancer):
            patcher = mock.patch.object(
                module, "authenticate", return_value=(self.manifest, {})
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=AssertionError("unexpected scheduler call"),
        )
        self.scheduler = patcher.start()
        self.addCleanup(patcher.stop)
        self.stdout = io.StringIO()
        self.output_context = contextlib.redirect_stdout(self.stdout)
        self.output_context.__enter__()
        self.addCleanup(self.output_context.__exit__, None, None, None)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, value)
        return path

    def read(self, relative):
        return json.loads((self.root / relative).read_text())

    def exists(self, relative):
        return (self.root / relative).exists()

    def prerequisite(self, stage, job="101"):
        """Install actual authenticated marker(s) bound to a stage receipt."""
        self.write(f"submissions/{stage}.json", {"stage": stage, "job": str(job)})
        names = [f"fit-{seed}" for seed in range(3)] if stage == "fit" else [stage]
        for index, name in enumerate(names):
            directory = self.root / f"evidence-{name}"
            directory.mkdir()
            self.write(f"evidence-{name}/result.json", {"stage": name, "passed": True})
            environment = {
                "SLURM_JOB_ID": f"{job}_{index}" if stage == "fit" else str(job),
                "SLURM_ARRAY_JOB_ID": str(job) if stage == "fit" else "",
                "SLURM_ARRAY_TASK_ID": str(index) if stage == "fit" else "",
            }
            with mock.patch.dict(os.environ, environment):
                study.marker(
                    self.root,
                    name,
                    directory,
                    additional_files=(
                        (self.root / "report.json",) if stage == "verify" else ()
                    ),
                )

    def earlier_completions(self, *, omit=None):
        for index, stage in enumerate(STAGES[:-1], start=101):
            if stage == omit:
                continue
            self.prerequisite(stage, str(index))
            rows = [
                row.split("|")[:3]
                for row in accounting(str(index), array=stage == "fit").splitlines()
            ]
            self.write(
                f"submissions/{stage}-completed.json",
                {"stage": stage, "job": str(index), "accounting": rows},
            )

    def final_scheduler(self, command, **kwargs):
        self.assertEqual(command[0], "sacct")
        job = command[command.index("-j") + 1]
        rows = accounting(job, array=job == "103")
        if "Start,End" in command[-1]:
            rows = "".join(
                line
                + "2026-10-05T00:00:00|2026-10-05T00:00:07|7|gres/gpu=1|gpu-node|\n"
                for line in rows.splitlines()
            )
        return rows

    def commands(self):
        return [call.args[0] for call in self.scheduler.call_args_list]


class TerminalAccountingTests(SubmissionFixture):
    def test_positive_single_terminal_zero_exit(self):
        self.scheduler.side_effect = [accounting()]
        self.assertEqual(submitter.terminal("101", 1), [["101", "COMPLETED", "0:0"]])
        self.assertEqual(
            self.commands()[0],
            [
                "sacct",
                "-X",
                "-j",
                "101",
                "--noheader",
                "--parsable2",
                "--format=JobID,State,ExitCode",
            ],
        )

    def test_positive_fit_array_requires_three_completed_tasks(self):
        self.scheduler.side_effect = [accounting(array=True)]
        rows = submitter.terminal("101", 3)
        self.assertEqual([row[0] for row in rows], ["101_0", "101_1", "101_2"])

    def test_wrong_array_count_or_empty_accounting_fails_closed(self):
        for output in (
            "",
            accounting(),
            accounting(array=True).splitlines()[0] + "\n",
            accounting(array=True) + "101_3|COMPLETED|0:0|\n",
        ):
            with self.subTest(output=output):
                self.scheduler.side_effect = [output]
                with self.assertRaises(RuntimeError):
                    submitter.terminal("101", 3)

    def test_pending_running_and_failed_states_never_retry(self):
        for state in (
            "PENDING",
            "RUNNING",
            "COMPLETING",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
        ):
            with self.subTest(state=state):
                self.scheduler.reset_mock()
                self.scheduler.side_effect = [accounting(state=state)]
                with self.assertRaises(RuntimeError):
                    submitter.terminal("101", 1)
                self.scheduler.assert_called_once()

    def test_completed_nonzero_exit_or_signal_is_not_success(self):
        for code in ("1:0", "0:9", "1:9", "0", ""):
            with self.subTest(code=code):
                self.scheduler.side_effect = [accounting(code=code)]
                with self.assertRaises(RuntimeError):
                    submitter.terminal("101", 1)

    def test_one_bad_array_member_blocks_whole_stage(self):
        self.scheduler.side_effect = [
            accounting(array=True).replace("101_2|COMPLETED|0:0", "101_2|COMPLETED|2:0")
        ]
        with self.assertRaises(RuntimeError):
            submitter.terminal("101", 3)

    def test_duplicate_unrelated_or_wrong_index_array_rows_fail(self):
        valid = accounting(array=True)
        for output in (
            accounting() * 3,
            valid.replace("101_2", "101_1"),
            valid.replace("101_2", "101_3"),
            valid.replace("101", "999"),
        ):
            with self.subTest(output=output):
                self.scheduler.side_effect = [output]
                with self.assertRaises(RuntimeError):
                    submitter.terminal("101", 3)

    def test_unrelated_single_job_cannot_authenticate_predecessor(self):
        self.scheduler.side_effect = [accounting("999")]
        with self.assertRaises(RuntimeError):
            submitter.terminal("101", 1)

    def test_invalid_requested_identity_or_count_never_queries_scheduler(self):
        for job, count in (
            ("101;other", 1),
            ("", 1),
            ("-1", 1),
            ("101", 0),
            ("101", 2),
            ("101", 4),
        ):
            with self.subTest(job=job, count=count), self.assertRaises(ValueError):
                submitter.terminal(job, count)
        self.scheduler.assert_not_called()

    def test_malformed_accounting_row_cannot_pass_as_completion(self):
        self.scheduler.side_effect = ["101|COMPLETED|\n"]
        with self.assertRaises((RuntimeError, ValueError)):
            submitter.terminal("101", 1)


class ScienceSubmissionTests(SubmissionFixture):
    def test_preflight_positive_intent_precedes_submission_and_receipt(self):
        def scheduler(command, **kwargs):
            self.assertEqual(command[0], "sbatch")
            self.assertTrue(self.exists("submissions/preflight-intent.json"))
            self.assertFalse(self.exists("submissions/preflight.json"))
            return "201;cluster\n"

        self.scheduler.side_effect = scheduler
        self.assertEqual(submitter.submit(self.root, "preflight"), "201")
        intent = self.read("submissions/preflight-intent.json")
        self.assertEqual(intent["source_commit"], self.manifest["source_commit"])
        self.assertIsNone(intent["predecessor"])
        receipt = self.read("submissions/preflight.json")
        command = receipt["command"]
        self.assertEqual(receipt["job"], "201")
        self.assertIn("--gres=gpu:1", command)
        self.assertIn("--partition=gpu-a30", command)
        self.assertFalse(any(arg.startswith("--dependency=") for arg in command))
        self.assertFalse(any(arg.startswith("--array=") for arg in command))
        self.assertEqual(command[-2:], [str(self.root), "preflight"])

    def test_fit_positive_authenticates_collect_and_submits_exact_three_seed_array(
        self,
    ):
        self.prerequisite("collect")
        self.scheduler.side_effect = [accounting(), "202\n"]
        submitter.submit(self.root, "fit")
        self.assertEqual(
            [command[0] for command in self.commands()], ["sacct", "sbatch"]
        )
        self.assertIn("--array=0-2%3", self.commands()[1])
        self.assertEqual(
            self.read("submissions/fit-intent.json")["accounting"],
            [["101", "COMPLETED", "0:0"]],
        )

    def test_evaluate_authenticates_all_three_fit_markers_after_array_success(self):
        self.prerequisite("fit")
        self.scheduler.side_effect = [accounting(array=True), "203\n"]
        with mock.patch.object(submitter, "require", wraps=study.require) as require:
            submitter.submit(self.root, "evaluate")
        self.assertEqual(
            require.call_args_list,
            [mock.call(self.root, f"fit-{seed}") for seed in range(3)],
        )
        self.assertEqual(
            len(self.read("submissions/evaluate-intent.json")["accounting"]), 3
        )

    def test_failed_array_member_prevents_marker_and_next_submission(self):
        self.prerequisite("fit")
        self.scheduler.side_effect = [
            accounting(array=True).replace("101_1|COMPLETED", "101_1|FAILED")
        ]
        with mock.patch.object(submitter, "require", wraps=study.require) as require:
            with self.assertRaises(RuntimeError):
                submitter.submit(self.root, "evaluate")
        require.assert_not_called()
        self.assertEqual([command[0] for command in self.commands()], ["sacct"])
        self.assertFalse(self.exists("submissions/evaluate-intent.json"))

    def test_changed_marker_artifact_blocks_science_even_after_scheduler_success(self):
        self.prerequisite("collect")
        (self.root / "evidence-collect/result.json").write_text('{"passed":false}')
        self.scheduler.side_effect = [accounting()]
        with self.assertRaisesRegex(ValueError, "artifact differs"):
            submitter.submit(self.root, "fit")
        self.assertEqual([command[0] for command in self.commands()], ["sacct"])
        self.assertFalse(self.exists("submissions/fit-intent.json"))

    def test_missing_fit_marker_blocks_evaluate(self):
        self.prerequisite("fit")
        (self.root / "markers/fit-2.json").unlink()
        self.scheduler.side_effect = [accounting(array=True)]
        with self.assertRaises(FileNotFoundError):
            submitter.submit(self.root, "evaluate")
        self.assertFalse(self.exists("submissions/evaluate-intent.json"))
        self.assertEqual(len(self.commands()), 1)

    def test_source_authentication_failure_precedes_all_scheduler_calls(self):
        with mock.patch.object(
            submitter, "authenticate", side_effect=ValueError("source differs")
        ):
            with self.assertRaisesRegex(ValueError, "source differs"):
                submitter.submit(self.root, "preflight")
        self.scheduler.assert_not_called()
        self.assertFalse(self.exists("submissions/preflight-intent.json"))

    def test_existing_intent_refuses_duplicate_without_scheduler_lookup(self):
        self.write("submissions/preflight-intent.json", {"stage": "preflight"})
        with self.assertRaises(FileExistsError):
            submitter.submit(self.root, "preflight")
        self.scheduler.assert_not_called()

    def test_existing_receipt_refuses_duplicate_even_without_intent(self):
        self.write("submissions/preflight.json", {"stage": "preflight", "job": "101"})
        with self.assertRaises(FileExistsError):
            submitter.submit(self.root, "preflight")
        self.scheduler.assert_not_called()

    def test_exclusive_intent_write_resolves_check_then_write_race(self):
        original = submitter._write_json

        def contested_write(path, value):
            original(path, {"owner": "concurrent submitter"})
            original(path, value)

        with mock.patch.object(submitter, "_write_json", side_effect=contested_write):
            with self.assertRaises(FileExistsError):
                submitter.submit(self.root, "preflight")
        self.scheduler.assert_not_called()
        self.assertEqual(
            self.read("submissions/preflight-intent.json"),
            {"owner": "concurrent submitter"},
        )

    def test_unknown_submission_response_preserves_intent_and_blocks_retry(self):
        self.scheduler.side_effect = ["Submitted batch job 201\n"]
        with self.assertRaisesRegex(RuntimeError, "unrecognized submission response"):
            submitter.submit(self.root, "preflight", continue_chain=True)
        self.assertTrue(self.exists("submissions/preflight-intent.json"))
        self.assertFalse(self.exists("submissions/preflight.json"))
        self.assertFalse(self.exists("submissions/handoff-preflight-intent.json"))
        with self.assertRaises(FileExistsError):
            submitter.submit(self.root, "preflight")
        self.scheduler.assert_called_once()

    def test_receipt_write_is_exclusive_and_collision_does_not_resubmit(self):
        original = submitter._write_json

        def contested_receipt(path, value):
            if path.name == "preflight.json":
                original(path, {"stage": "preflight", "job": "999"})
            original(path, value)

        self.scheduler.side_effect = ["201\n"]
        with mock.patch.object(submitter, "_write_json", side_effect=contested_receipt):
            with self.assertRaises(FileExistsError):
                submitter.submit(self.root, "preflight")
        self.assertTrue(self.exists("submissions/preflight-intent.json"))
        self.assertEqual(self.read("submissions/preflight.json")["job"], "999")
        with self.assertRaises(FileExistsError):
            submitter.submit(self.root, "preflight")
        self.scheduler.assert_called_once()

    def test_scheduler_exception_preserves_intent_without_automatic_retry(self):
        self.scheduler.side_effect = subprocess.CalledProcessError(
            1, ["sbatch"], output="uncertain failure"
        )
        with self.assertRaises(subprocess.CalledProcessError):
            submitter.submit(self.root, "preflight")
        self.assertTrue(self.exists("submissions/preflight-intent.json"))
        self.assertFalse(self.exists("submissions/preflight.json"))
        with self.assertRaises(FileExistsError):
            submitter.submit(self.root, "preflight")
        self.scheduler.assert_called_once()


class HandoffTests(SubmissionFixture):
    def test_continue_chain_schedules_only_cpu_afterany_gate_not_next_science(self):
        self.scheduler.side_effect = ["201\n", "301\n"]
        submitter.submit(self.root, "preflight", continue_chain=True)
        science, gate = self.commands()
        self.assertEqual(science[-1], "preflight")
        self.assertEqual(gate[-1], "preflight")
        self.assertIn("--partition=cpu-zen3", gate)
        self.assertIn("--dependency=afterany:201", gate)
        self.assertFalse(any(arg.startswith("--gres=") for arg in gate))
        self.assertTrue(gate[-4].endswith("parallel_handoff.sh"))
        self.assertFalse(self.exists("submissions/collect-intent.json"))
        self.assertEqual(
            self.read("submissions/handoff-preflight.json")["predecessor_job"], "201"
        )

    def test_unknown_gate_response_keeps_both_science_receipt_and_gate_intent(self):
        self.scheduler.side_effect = ["201\n", "unrecognized response"]
        with self.assertRaisesRegex(RuntimeError, "unrecognized handoff response"):
            submitter.submit(self.root, "preflight", continue_chain=True)
        self.assertTrue(self.exists("submissions/preflight.json"))
        self.assertTrue(self.exists("submissions/handoff-preflight-intent.json"))
        self.assertFalse(self.exists("submissions/handoff-preflight.json"))
        with self.assertRaises(FileExistsError):
            submitter.handoff(self.root, "preflight", "201")
        self.assertEqual(self.scheduler.call_count, 2)

    def test_existing_handoff_receipt_refuses_duplicate(self):
        self.write(
            "submissions/handoff-collect.json", {"job": "301", "stage": "collect"}
        )
        with self.assertRaises(FileExistsError):
            submitter.handoff(self.root, "collect", "201")
        self.scheduler.assert_not_called()


class AdvancementTests(SubmissionFixture):
    def test_positive_gate_authenticates_then_advances_once_and_registers_next_gate(
        self,
    ):
        self.prerequisite("preflight")
        self.scheduler.side_effect = [accounting(), accounting(), "202\n", "302\n"]
        advancer.advance(self.root, "preflight")
        self.assertEqual(
            [command[0] for command in self.commands()],
            ["sacct", "sacct", "sbatch", "sbatch"],
        )
        self.assertTrue(self.exists("submissions/preflight-completed.json"))
        self.assertEqual(self.read("submissions/collect.json")["job"], "202")
        self.assertEqual(self.read("submissions/handoff-collect.json")["job"], "302")
        self.assertFalse(self.exists("submissions/fit-intent.json"))

    def test_pending_predecessor_never_advances_or_retries(self):
        self.prerequisite("collect")
        self.scheduler.side_effect = [accounting(state="PENDING")]
        with mock.patch.object(advancer, "submit") as submit:
            with self.assertRaises(RuntimeError):
                advancer.advance(self.root, "collect")
        submit.assert_not_called()
        self.scheduler.assert_called_once()
        self.assertFalse(self.exists("submissions/collect-completed.json"))

    def test_bad_marker_never_records_completion_or_advances(self):
        self.prerequisite("preflight")
        (self.root / "evidence-preflight/result.json").write_text('{"passed":false}')
        self.scheduler.side_effect = [accounting()]
        with mock.patch.object(advancer, "submit") as submit:
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                advancer.advance(self.root, "preflight")
        submit.assert_not_called()
        self.assertFalse(self.exists("submissions/preflight-completed.json"))
        self.assertFalse(self.exists("submissions/collect-intent.json"))

    def test_fit_advance_requires_exactly_three_successful_tasks_and_all_markers(self):
        self.prerequisite("fit")
        self.scheduler.side_effect = [accounting(array=True)]
        with mock.patch.object(advancer, "submit") as submit, mock.patch.object(
            advancer, "require", wraps=study.require
        ) as require:
            advancer.advance(self.root, "fit")
        submit.assert_called_once_with(self.root, "evaluate", continue_chain=True)
        self.assertEqual(
            require.call_args_list,
            [mock.call(self.root, f"fit-{seed}") for seed in range(3)],
        )
        self.assertEqual(
            len(self.read("submissions/fit-completed.json")["accounting"]), 3
        )

    def test_existing_completion_record_refuses_repeated_advancement(self):
        self.prerequisite("collect")
        self.write(
            "submissions/collect-completed.json", {"stage": "collect", "job": "101"}
        )
        self.scheduler.side_effect = [accounting()]
        with mock.patch.object(advancer, "submit") as submit:
            with self.assertRaises(FileExistsError):
                advancer.advance(self.root, "collect")
        submit.assert_not_called()

    def test_final_completion_binds_all_stage_records_and_report_hash_not_scientific_success(
        self,
    ):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        self.scheduler.side_effect = self.final_scheduler
        with mock.patch.object(advancer, "submit") as submit:
            advancer.advance(self.root, "verify")
        submit.assert_not_called()
        completion = self.read("completion.json")
        self.assertTrue(completion["completed"])
        self.assertTrue(completion["scientific_success_not_implied"])
        self.assertEqual(completion["source_commit"], self.manifest["source_commit"])
        self.assertEqual(
            completion["manifest_sha256"], study.sha(self.root / "manifest.json")
        )
        self.assertEqual(
            completion["report_sha256"], study.sha(self.root / "report.json")
        )
        self.assertEqual(set(completion["stages"]), set(STAGES))
        self.assertTrue(
            all(
                completion["stages"][stage]
                == self.read(f"submissions/{stage}-completed.json")
                for stage in STAGES
            )
        )
        self.assertEqual(set(completion["slurm_allocation_evidence"]), set(STAGES))
        self.assertTrue(
            all(
                "gres/gpu=1" in value
                for value in completion["slurm_allocation_evidence"].values()
            )
        )
        commands = self.commands()
        self.assertEqual(len(commands), 11)  # verify + five terminal/resource pairs.
        self.assertEqual(sum("Start,End" in command[-1] for command in commands), 5)
        self.assertEqual(
            [
                command[command.index("-j") + 1]
                for command in commands
                if "Start,End" not in command[-1]
            ],
            ["105", "101", "102", "103", "104", "105"],
        )
        self.assertIn(
            "PARALLEL_TRAJECTORY_SCHEDULER_COMPLETION_VERIFIED", self.stdout.getvalue()
        )

    def test_missing_earlier_stage_record_prevents_final_completion(self):
        self.prerequisite("verify", "105")
        self.earlier_completions(omit="collect")
        self.scheduler.side_effect = [accounting("105")]
        with mock.patch.object(advancer, "submit") as submit:
            with self.assertRaises(FileNotFoundError):
                advancer.advance(self.root, "verify")
        submit.assert_not_called()
        self.assertFalse(self.exists("completion.json"))

    def test_failed_final_job_does_not_complete_even_with_all_previous_records(self):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        self.scheduler.side_effect = [accounting("105", code="1:0")]
        with self.assertRaises(RuntimeError):
            advancer.advance(self.root, "verify")
        self.assertFalse(self.exists("completion.json"))
        self.assertFalse(self.exists("submissions/verify-completed.json"))

    def test_final_completion_rejects_wrong_stage_in_earlier_attestation(self):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        path = self.root / "submissions/preflight-completed.json"
        value = json.loads(path.read_text())
        value["stage"] = "collect"
        path.write_text(json.dumps(value))
        self.scheduler.side_effect = self.final_scheduler
        with self.assertRaisesRegex(ValueError, "attestation differs"):
            advancer.advance(self.root, "verify")
        self.assertFalse(self.exists("completion.json"))

    def test_final_completion_rejects_wrong_job_in_earlier_attestation(self):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        path = self.root / "submissions/collect-completed.json"
        value = json.loads(path.read_text())
        value["job"] = "999"
        path.write_text(json.dumps(value))
        self.scheduler.side_effect = self.final_scheduler
        with self.assertRaisesRegex(ValueError, "attestation differs"):
            advancer.advance(self.root, "verify")
        self.assertFalse(self.exists("completion.json"))

    def test_final_completion_rejects_bad_saved_fit_exit_even_if_current_sacct_is_good(
        self,
    ):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        path = self.root / "submissions/fit-completed.json"
        value = json.loads(path.read_text())
        value["accounting"][2][2] = "2:0"
        path.write_text(json.dumps(value))
        self.scheduler.side_effect = self.final_scheduler
        with self.assertRaisesRegex(ValueError, "attestation differs"):
            advancer.advance(self.root, "verify")
        self.assertFalse(self.exists("completion.json"))

    def test_final_completion_rechecks_earlier_terminal_state_instead_of_trusting_saved_success(
        self,
    ):
        self.prerequisite("verify", "105")
        self.earlier_completions()

        def scheduler(command, **kwargs):
            job = command[command.index("-j") + 1]
            return (
                accounting("103", array=True, code="1:0")
                if job == "103"
                else self.final_scheduler(command, **kwargs)
            )

        self.scheduler.side_effect = scheduler
        with self.assertRaises(RuntimeError):
            advancer.advance(self.root, "verify")
        self.assertFalse(self.exists("completion.json"))

    def test_final_completion_rejects_a_report_changed_after_verify_marker(self):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        (self.root / "report.json").write_text('{"promotion":{"promote":true}}')
        self.scheduler.side_effect = [accounting("105")]
        with self.assertRaisesRegex(ValueError, "artifact differs"):
            advancer.advance(self.root, "verify")
        self.assertFalse(self.exists("completion.json"))

    def test_final_completion_reauthenticates_earlier_collection_artifacts(self):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        (self.root / "evidence-collect/result.json").write_text('{"passed":false}')
        self.scheduler.side_effect = self.final_scheduler
        with self.assertRaisesRegex(ValueError, "artifact differs"):
            advancer.advance(self.root, "verify")
        self.assertFalse(self.exists("completion.json"))

    def test_final_completion_reauthenticates_each_fit_seed_marker(self):
        self.prerequisite("verify", "105")
        self.earlier_completions()
        (self.root / "evidence-fit-2/result.json").write_text('{"passed":false}')
        self.scheduler.side_effect = self.final_scheduler
        with mock.patch.object(advancer, "require", wraps=study.require) as require:
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                advancer.advance(self.root, "verify")
        checked = [call.args[1] for call in require.call_args_list]
        self.assertTrue({"fit-0", "fit-1", "fit-2"}.issubset(checked))
        self.assertFalse(self.exists("completion.json"))


if __name__ == "__main__":
    program = unittest.main(exit=False)
    if program.result.wasSuccessful():
        print("PARALLEL_SUBMISSION_VERIFIED")
    sys.exit(0 if program.result.wasSuccessful() else 1)
