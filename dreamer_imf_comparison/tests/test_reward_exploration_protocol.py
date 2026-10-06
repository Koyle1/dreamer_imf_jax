from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import reward_exploration_protocol as p

SUBMITTER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/submit_reward_exploration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tested_reward_exploration_submitter", SUBMITTER_PATH
)
submitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submitter)


def submission_fixture(root, manifest, stage, job="123"):
    p.write_json_exclusive(
        root / "submissions" / f"{stage}-intent.json",
        dict(stage=stage, manifest_sha256=manifest["manifest_sha256"], command=[]),
    )
    p.write_json_exclusive(
        root / "submissions" / f"{stage}.json",
        dict(
            stage=stage,
            status="submitted",
            job=job,
            manifest_sha256=manifest["manifest_sha256"],
            command=[],
            intent_sha256=p.sha(root / "submissions" / f"{stage}-intent.json"),
        ),
    )


class ProtocolTests(unittest.TestCase):
    def test_exact_frozen_registration(self):
        protocol = p.load_protocol()
        self.assertEqual(protocol["parent"]["checkpoint_sha256"], p.PARENT_SHA256)
        self.assertEqual(protocol["dataset_sha256"], p.DATASET_SHA256)
        self.assertEqual(
            protocol["planned_native_steps_per_cell"],
            protocol["collection_native_steps"]
            + len(protocol["evaluation_milestones"])
            * protocol["evaluation_episodes"]
            * protocol["episode_native_steps"],
        )
        self.assertEqual(
            protocol["native_step_ceiling_total"],
            len(p.ARMS) * len(p.SEEDS) * protocol["native_step_ceiling_per_cell"],
        )

    def test_every_top_level_mutation_fails_closed(self):
        protocol = p.load_protocol()
        for key in protocol:
            with self.subTest(key=key):
                changed = copy.deepcopy(protocol)
                changed[key] = None
                with self.assertRaises(ValueError):
                    p.validate_protocol(changed)
        changed = dict(protocol, surprise="ignored settings are forbidden")
        with self.assertRaises(ValueError):
            p.validate_protocol(changed)

    def test_nested_mutations_and_bool_numeric_aliases_rejected(self):
        protocol = p.load_protocol()
        for key in ("parent", "matched_fit", "ensemble_fit", "arms"):
            changed = copy.deepcopy(protocol)
            changed[key][next(iter(changed[key]))] = False
            with self.assertRaises(ValueError):
                p.validate_protocol(changed)
        changed = copy.deepcopy(protocol)
        changed["exploration_blocks_per_cycle"] = True
        with self.assertRaises(ValueError):
            p.validate_protocol(changed)

    def test_cell_grid_is_exact_and_unique(self):
        cells = [p.cell_for_index(index) for index in range(12)]
        self.assertEqual(
            [(cell["arm"], cell["seed"]) for cell in cells],
            [(arm, seed) for arm in "ABCD" for seed in (701, 702, 703)],
        )
        self.assertEqual(len({cell["cell_id"] for cell in cells}), 12)
        for invalid in (-1, 12, 1.0, True, "1"):
            with self.assertRaises(ValueError):
                p.cell_for_index(invalid)

    def test_paired_schedule_is_exact_twenty_percent_in_blocks(self):
        # 80000 / 2 / 16 = 2500 decisions per environment, exactly 25 cycles.
        for seed in p.SEEDS:
            c = [p.collection_policy("C", seed, index) for index in range(2500)]
            d = [p.collection_policy("D", seed, index) for index in range(2500)]
            self.assertEqual(c.count("random"), 500)
            self.assertEqual(d.count("explore"), 500)
            self.assertEqual(
                [value != "task" for value in c], [value != "task" for value in d]
            )
            for offset in range(0, 2500, 20):
                self.assertEqual(len(set(c[offset : offset + 20])), 1)
            for arm in "AB":
                self.assertTrue(
                    all(
                        p.collection_policy(arm, seed, index) == "task"
                        for index in range(2500)
                    )
                )

    def test_fresh_evaluation_seeds_paired_and_disjoint(self):
        sets = [
            set(p.eval_seeds(seed, milestone))
            for seed in p.SEEDS
            for milestone in (40000, 80000)
        ]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        self.assertEqual(len(sets) * 5, 30)
        for seed, milestone in ((431, 40000), (701, 40001), (True, 40000)):
            with self.assertRaises(ValueError):
                p.eval_seeds(seed, milestone)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_atomic_exclusive_publication_never_overwrites(self):
        target = self.root / "record.json"
        p.write_json_exclusive(target, {"value": 1})
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            p.write_json_exclusive(target, {"value": 2})
        self.assertEqual(before, target.read_bytes())
        self.assertEqual(list(self.root.iterdir()), [target])

    def test_finite_json_and_numpy_evidence(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                p.write_json_exclusive(self.root / "bad.json", {"nested": [value]})
            np.savez(self.root / "bad.npz", array=[value])
            with self.assertRaises(ValueError):
                p.artifact(self.root / "bad.npz")
        np.savez(self.root / "good.npz", array=[1.0, 2.0])
        p.verify_artifact(p.artifact(self.root / "good.npz"))

    def test_duplicate_keys_and_json_nan_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
            path = self.root / "invalid.json"
            path.write_text(raw)
            with self.assertRaises(ValueError):
                p.read(path)

    def test_hash_tampering_and_symlinks_rejected(self):
        path = self.root / "file.bin"
        path.write_bytes(b"valid")
        reference = p.artifact(path)
        path.write_bytes(b"other")
        with self.assertRaises(ValueError):
            p.verify_artifact(reference)
        link = self.root / "link.bin"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            p.artifact(link)

    def _marker_fixture(self):
        manifest = dict(manifest_sha256="m" * 64, source_commit="c" * 40)
        p.write_json_exclusive(self.root / "manifest.json", manifest)
        path = self.root / "match/report.json"
        p.write_json_exclusive(path, {"score": 1.0})
        submission_fixture(self.root, manifest, "match")
        return manifest, path

    def test_marker_checks_manifest_artifact_and_stage(self):
        manifest, path = self._marker_fixture()
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ):
            p.write_marker(self.root, "match", [path])
            self.assertEqual(p.require_marker(self.root, "match")["stage"], "match")
            path.write_text('{"score":2.0}')
            with self.assertRaises(ValueError):
                p.require_marker(self.root, "match")

    def test_missing_empty_or_escaping_marker_rejected(self):
        manifest, path = self._marker_fixture()
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ):
            with self.assertRaises(FileNotFoundError):
                p.require_marker(self.root, "preflight")
            with self.assertRaises(ValueError):
                p.write_marker(self.root, "match", [])
            with self.assertRaises(ValueError):
                p.write_marker(self.root, "unknown", [path])
            with self.assertRaises(ValueError):
                p._inside(self.root, self.root / "../outside")

    def test_marker_is_write_once(self):
        manifest, path = self._marker_fixture()
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ):
            p.write_marker(self.root, "match", [path])
            with self.assertRaises(FileExistsError):
                p.write_marker(self.root, "match", [path])

    def test_marker_requires_matching_submission_receipt_and_runtime(self):
        manifest, path = self._marker_fixture()
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ):
            with mock.patch.dict(
                os.environ, {"SLURM_JOB_ID": "999"}
            ), self.assertRaises(ValueError):
                p.write_marker(self.root, "match", [path])
            with self.assertRaises(FileNotFoundError):
                p.write_marker(self.root, "preflight", [path])
            p.write_marker(self.root, "match", [path])
            receipt = self.root / "submissions/match.json"
            value = p.read(receipt)
            value["job"] = "999"
            receipt.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                p.require_marker(self.root, "match")


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        p._initialize_budget(self.root, 0)

    def test_failed_call_charge_survives_restart(self):
        with p.NativeStepBudget(self.root, 0) as ledger:
            ledger.reserve(32, "collect", {"before_environment_call": True})
            ledger.record_reset(16)
            self.assertEqual(ledger.total, 32)
        with p.NativeStepBudget(self.root, 0) as restarted:
            self.assertEqual(restarted.total, 32)
            self.assertEqual(restarted.snapshot()["reset_count"], 16)
            restarted.reserve(2, "evaluate")
            self.assertEqual(restarted.total, 34)

    def test_budget_cannot_overrun_or_refund(self):
        with p.NativeStepBudget(self.root, 0) as ledger:
            ledger.reserve(90000)
            ledger.reserve(10000, "evaluate")
            before = ledger.snapshot()
            for invalid in (-2, 0, 1, True, 2.0, 2):
                with self.assertRaises(ValueError):
                    ledger.reserve(invalid)
                self.assertEqual(ledger.snapshot(), before)
            with self.assertRaises(ValueError):
                ledger.reserve(2, "refund")

    def test_two_budget_handles_serialize_shared_native_ceiling(self):
        with p.NativeStepBudget(self.root, 0) as first, p.NativeStepBudget(
            self.root, 0
        ) as second:
            first.reserve(60000)
            second.reserve(40000)
            self.assertEqual(first.total, 100000)
            with self.assertRaises(ValueError):
                first.reserve(2)
            self.assertEqual(second.total, 100000)

    def test_missing_empty_truncated_or_wrong_cell_cannot_reset(self):
        with self.assertRaises(ValueError):
            p.NativeStepBudget(self.root, 1)
        path = self.root / "native-budget.jsonl"
        path.write_bytes(b"")
        with self.assertRaises(ValueError):
            p.NativeStepBudget(self.root, 0)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            p.NativeStepBudget(self.root, 0)
        with self.assertRaises(FileExistsError):
            p._initialize_budget(self.root, 0)

    def test_broken_hash_and_partial_tail_fail_closed(self):
        with p.NativeStepBudget(self.root, 0) as ledger:
            ledger.reserve(32)
        path = self.root / "native-budget.jsonl"
        saved = path.read_bytes()
        path.write_bytes(saved[:-1])
        with self.assertRaises(ValueError):
            p.NativeStepBudget(self.root, 0)
        path.write_bytes(saved.replace(b'"native_steps":32', b'"native_steps":30'))
        with self.assertRaises(ValueError):
            p.NativeStepBudget(self.root, 0)

    def test_live_truncation_detected(self):
        with p.NativeStepBudget(self.root, 0) as ledger:
            ledger.reserve(20)
            path = self.root / "native-budget.jsonl"
            path.write_bytes(path.read_bytes().splitlines(keepends=True)[0])
            with self.assertRaises(ValueError):
                ledger.reserve(2)

    def test_nonfinite_detail_never_charged(self):
        with p.NativeStepBudget(self.root, 0) as ledger:
            before = ledger.snapshot()
            with self.assertRaises(ValueError):
                ledger.reserve(2, detail={"nan": float("nan")})
            self.assertEqual(ledger.snapshot(), before)


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def _git_source(self):
        source = self.root / "source"
        source.mkdir()
        (source / "module.py").write_text("VALUE = 1\n")
        (source / ".gitignore").write_text("ignored/\n")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "add", "module.py", ".gitignore"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Protocol Test",
                "-c",
                "user.email=protocol-test@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        return source

    def test_exact_source_detects_changes_and_ignored_executable_injection(self):
        source = self._git_source()
        identity = p.source_identity(source)
        self.assertEqual(len(identity["commit"]), 40)
        self.assertIn("module.py", identity["files"])
        (source / "module.py").write_text("VALUE = 2\n")
        with self.assertRaises(ValueError):
            p.source_identity(source)
        (source / "module.py").write_text("VALUE = 1\n")
        (source / "ignored").mkdir()
        (source / "ignored/injected.py").write_text("VALUE = 999\n")
        with self.assertRaises(ValueError):
            p.source_identity(source)

    def test_distribution_bytes_not_only_versions_are_hashed(self):
        library = self.root / "library.py"
        executable = self.root / "python"
        executable.write_bytes(b"python-runtime")
        library.write_text("VALUE = 1\n")
        distribution = SimpleNamespace(
            version="1.0",
            files=[Path("library.py")],
            read_text=lambda name: "Name: example\nVersion: 1.0\n",
            locate_file=lambda item: self.root / item,
        )
        with mock.patch.object(
            p.importlib.metadata, "distribution", return_value=distribution
        ), mock.patch.object(p.sys, "executable", str(executable)):
            before = p.runtime_identity()
            library.write_text("VALUE = 2\n")
            after = p.runtime_identity()
        self.assertEqual(
            before["packages"]["numpy"]["metadata_sha256"],
            after["packages"]["numpy"]["metadata_sha256"],
        )
        self.assertNotEqual(
            before["packages"]["numpy"]["files_sha256"],
            after["packages"]["numpy"]["files_sha256"],
        )

    def test_wrong_parent_or_common_dataset_sha_is_rejected(self):
        parent = self.root / "parent"
        parent.mkdir()
        (parent / "checkpoint_200000.pkl").write_bytes(b"not the registered checkpoint")
        dataset = self.root / "dataset.npz"
        dataset.write_bytes(b"not the registered dataset")
        with self.assertRaises(ValueError):
            p._parent_identity(parent, dataset, p.load_protocol())

    def test_authentication_rejects_tampered_manifest_before_external_reads(self):
        p.write_json_exclusive(
            self.root / "manifest.json",
            {"manifest_sha256": "0" * 64, "source_commit": "c" * 40},
        )
        with mock.patch.object(p, "source_identity") as source:
            with self.assertRaises(ValueError):
                p.authenticate(self.root)
            source.assert_not_called()


class SubmitterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.manifest = dict(
            source="/exact/source",
            source_commit="c" * 40,
            manifest_sha256="m" * 64,
            inputs=dict(
                parent_cell="/parent/cell", dataset=dict(path="/data/common.npz")
            ),
        )
        p.write_json_exclusive(self.root / "manifest.json", self.manifest)
        for index in range(12):
            directory = p.cell_directory(self.root, index)
            directory.mkdir(parents=True)
            p._initialize_budget(directory, index)
        self.auth = mock.patch.object(
            p, "authenticate", return_value=(self.manifest, p.load_protocol())
        )
        self.auth.start()
        self.addCleanup(self.auth.stop)

    def _submit_match(self):
        with mock.patch.object(
            submitter.subprocess, "check_output", return_value="123\n"
        ):
            return submitter.submit(self.root, "match")

    def test_duplicate_submit_never_invokes_scheduler(self):
        self._submit_match()
        with mock.patch.object(submitter.subprocess, "check_output") as scheduler:
            with self.assertRaises(FileExistsError):
                submitter.submit(self.root, "match")
            scheduler.assert_not_called()

    def test_ambiguous_attempt_is_immutable_and_cannot_retry(self):
        for response in ("abc", "123;remote", "123\n456"):
            with self.subTest(response=response):
                root = self.root / response.replace("\n", "-").replace(";", "-")
                root.mkdir()
                p.write_json_exclusive(root / "manifest.json", self.manifest)
                with mock.patch.object(
                    submitter.subprocess, "check_output", return_value=response
                ), self.assertRaises(RuntimeError):
                    submitter.submit(root, "match")
                receipt = p.read(root / "submissions/match.json")
                self.assertEqual(receipt["status"], "ambiguous_submission")
                with self.assertRaises(FileExistsError):
                    submitter.submit(root, "match")

    def test_scheduler_error_retains_intent_and_failure_receipt(self):
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(1, ["sbatch"]),
        ), self.assertRaises(subprocess.CalledProcessError):
            submitter.submit(self.root, "match")
        self.assertTrue((self.root / "submissions/match-intent.json").is_file())
        self.assertEqual(
            p.read(self.root / "submissions/match.json")["status"],
            "ambiguous_submission",
        )

    def test_pending_or_running_predecessor_blocks_progress(self):
        self._submit_match()
        for state in ("PENDING", "RUNNING"):
            with mock.patch.object(
                submitter.subprocess, "check_output", return_value=f"123|{state}\n"
            ), self.assertRaises(RuntimeError):
                submitter.submit(self.root, "preflight")
        self.assertFalse((self.root / "submissions/preflight-intent.json").exists())

    def test_complete_accounting_without_marker_blocks_progress(self):
        self._submit_match()
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=["", "123|COMPLETED|0:0\n"],
        ), self.assertRaises(FileNotFoundError):
            submitter.submit(self.root, "preflight")
        self.assertFalse((self.root / "submissions/preflight-intent.json").exists())

    def test_valid_prior_evidence_allows_exact_next_stage(self):
        self._submit_match()
        path = self.root / "match/report.json"
        p.write_json_exclusive(path, {"verified": True})
        p.write_marker(self.root, "match", [path])
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=["", "123|COMPLETED|0:0\n", "124\n"],
        ) as scheduler:
            self.assertEqual(submitter.submit(self.root, "preflight"), "124")
            command = scheduler.call_args_list[-1].args[0]
            self.assertIn("--time=02:00:00", command)
            self.assertIn("--partition=gpu-a30", command)

    def test_tampered_marker_never_submits(self):
        self._submit_match()
        path = self.root / "match/report.json"
        p.write_json_exclusive(path, {"verified": True})
        p.write_marker(self.root, "match", [path])
        path.write_text('{"verified":false}')
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=["", "123|COMPLETED|0:0\n"],
        ) as scheduler, self.assertRaises(ValueError):
            submitter.submit(self.root, "preflight")
        self.assertEqual(scheduler.call_count, 2)

    def test_failed_cell_cannot_be_resubmitted_or_erase_budget(self):
        with p.NativeStepBudget(p.cell_directory(self.root, 0), 0) as budget:
            budget.reserve(20)
        with mock.patch.object(
            submitter.subprocess, "check_output"
        ) as scheduler, self.assertRaises(ValueError):
            submitter.submit(self.root, "cells")
        scheduler.assert_not_called()
        with p.NativeStepBudget(p.cell_directory(self.root, 0), 0) as budget:
            self.assertEqual(budget.total, 20)

    def test_array_accounting_requires_all_twelve_successful_rows(self):
        complete = "".join(f"123_{index}|COMPLETED|0:0\n" for index in range(12))
        with mock.patch.object(
            submitter.subprocess, "check_output", side_effect=["", complete]
        ):
            self.assertEqual(submitter.terminal("123", 12)["expected_rows"], 12)
        for wrong in (
            complete.replace("123_11|COMPLETED|0:0\n", ""),
            complete.replace("123_0|COMPLETED|0:0", "123_0|FAILED|1:0"),
            complete + "123|COMPLETED|0:0\n",
            "",
        ):
            with mock.patch.object(
                submitter.subprocess, "check_output", side_effect=["", wrong]
            ), self.assertRaises(RuntimeError):
                submitter.terminal("123", 12)

    def test_purged_squeue_job_falls_back_but_controller_failure_blocks(self):
        absent = subprocess.CalledProcessError(
            1, ["squeue"], stderr="slurm_load_jobs error: Invalid job id specified"
        )
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=[absent, "123|COMPLETED|0:0\n"],
        ):
            self.assertEqual(submitter.terminal("123")["job"], "123")
        down = subprocess.CalledProcessError(
            1, ["squeue"], stderr="Unable to contact slurm controller"
        )
        with mock.patch.object(
            submitter.subprocess, "check_output", side_effect=down
        ), self.assertRaises(subprocess.CalledProcessError):
            submitter.terminal("123")

    def test_cell_array_command_and_handoff_are_bounded(self):
        p.write_json_exclusive(
            self.root / "submissions/preflight-intent.json",
            {
                "stage": "preflight",
                "manifest_sha256": self.manifest["manifest_sha256"],
                "command": [],
            },
        )
        p.write_json_exclusive(
            self.root / "submissions/preflight.json",
            {
                "stage": "preflight",
                "status": "submitted",
                "job": "122",
                "manifest_sha256": self.manifest["manifest_sha256"],
                "command": [],
                "intent_sha256": p.sha(self.root / "submissions/preflight-intent.json"),
            },
        )
        path = self.root / "preflight/report.json"
        p.write_json_exclusive(path, {"verified": True})
        p.write_marker(self.root, "preflight", [path])
        with mock.patch.object(
            submitter.subprocess,
            "check_output",
            side_effect=["", "122|COMPLETED|0:0\n", "123\n", "124\n"],
        ) as scheduler:
            submitter.submit(self.root, "cells", continue_chain=True)
            commands = [call.args[0] for call in scheduler.call_args_list]
            self.assertIn("--array=0-11%4", commands[2])
            self.assertIn("--time=08:00:00", commands[2])
            self.assertIn("--dependency=afterany:123", commands[3])
            self.assertIn("--partition=cpu-zen3", commands[3])

    def test_launcher_separate_verifier_and_fresh_scoped_cache(self):
        launcher = SUBMITTER_PATH.with_name("reward_exploration_job.sh")
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text()
        self.assertIn('"verify-$EXPLORE_STAGE"', text)
        self.assertIn("mktemp -d", text)
        self.assertIn("SLURM_ARRAY_TASK_ID", text)
        self.assertIn("JAX_COMPILATION_CACHE_DIR", text)
        self.assertIn("Python/3.12.3-GCCcore-13.3.0", text)
        self.assertIn("venv-dreamer-ablation-jax0433/bin/python", text)


if __name__ == "__main__":
    unittest.main()
