"""Offline protocol, corruption, scheduler and budget acceptance checks."""

import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import posterior_reward_protocol as p
from dreamer_imf_compare import reward_exploration_protocol as legacy

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/submit_posterior_reward.py"
SPEC = importlib.util.spec_from_file_location("posterior_reward_submitter_test", SCRIPT)
submitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submitter)


def receipt(root, manifest, stage, job="123"):
    intent = root / "submissions" / f"{stage}-intent.json"
    p.write_json_exclusive(
        intent,
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
            intent_sha256=p.sha(intent),
        ),
    )


class ProtocolTests(unittest.TestCase):
    def test_budget_configuration_is_arithmetic_not_a_claim(self):
        q = p.load_protocol()
        self.assertEqual(q["parent"]["checkpoint_sha256"], p.PARENT_SHA256)
        self.assertEqual(q["dataset_sha256"], p.DATASET_SHA256)
        planned = (
            q["collection_native_steps"]
            + len(q["evaluation_milestones"])
            * q["evaluation_episodes"]
            * q["episode_native_steps"]
        )
        self.assertEqual(planned, q["planned_native_steps_per_cell"])
        self.assertEqual(planned * p.CELL_COUNT, 990000)
        self.assertEqual(p.CELL_CEILING * p.CELL_COUNT, q["native_step_ceiling_total"])
        self.assertEqual(q["evaluation_milestones"], [0, 40000, 80000])
        self.assertEqual(q["primary_comparison"], "C-B")

    def test_every_protocol_mutation_rejected(self):
        q = p.load_protocol()
        for key in q:
            with self.subTest(key=key):
                changed = copy.deepcopy(q)
                changed[key] = None
                with self.assertRaises(ValueError):
                    p.validate_protocol(changed)
        changed = copy.deepcopy(q)
        changed["arms"]["C"]["distillation_lambda"] = 0.0
        with self.assertRaises(ValueError):
            p.validate_protocol(changed)
        changed = copy.deepcopy(q)
        changed["distillation_temperature"] = True
        with self.assertRaises(ValueError):
            p.validate_protocol(changed)

    def test_grid_seed_pairing_and_domain_separation(self):
        grid = [p.cell_for_index(i) for i in range(9)]
        self.assertEqual(
            [(v["arm"], v["seed"], v["readout_seed"]) for v in grid],
            [(a, s, s - 901) for a in "ABC" for s in (901, 902, 903)],
        )
        collect = [p.collection_seed(s, w) for s in p.SEEDS for w in range(16)]
        evaluate = [
            v for s in p.SEEDS for m in (0, 40000, 80000) for v in p.eval_seeds(s, m)
        ]
        self.assertEqual(len(set(collect)), 48)
        self.assertEqual(len(set(evaluate)), 90)
        self.assertFalse(set(collect) & set(evaluate))
        for invalid in (-1, 9, True, 1.0, "1"):
            with self.assertRaises(ValueError):
                p.cell_for_index(invalid)
        for args in ((900, 0), (901, -1), (901, True), (901, 16)):
            with self.assertRaises(ValueError):
                p.collection_seed(*args)
        with self.assertRaises(ValueError):
            p.eval_seeds(901, True)

    def test_legacy_namespace_and_protocol_unchanged(self):
        self.assertEqual(legacy.CELL_COUNT if hasattr(legacy, "CELL_COUNT") else 12, 12)
        self.assertEqual(legacy.SEEDS, (701, 702, 703))
        self.assertEqual(legacy.ARMS, ("A", "B", "C", "D"))
        self.assertEqual(legacy.load_protocol()["native_step_ceiling_per_cell"], 100000)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)

    def manifest(self):
        manifest = dict(
            manifest_sha256="a" * 64,
            source_commit="b" * 40,
            source="/source",
            upstream={"path": "/upstream"},
        )
        p.write_json_exclusive(self.root / "manifest.json", manifest)
        return manifest

    def test_exclusive_and_finite_evidence(self):
        p.write_json_exclusive(self.root / "item.json", {"value": 1})
        with self.assertRaises(FileExistsError):
            p.write_json_exclusive(self.root / "item.json", {"value": 2})
        self.assertEqual(p.read(self.root / "item.json"), {"value": 1})
        with self.assertRaises(ValueError):
            p.write_json_exclusive(self.root / "nan.json", {"value": float("nan")})
        np.savez(self.root / "bad.npz", x=[np.inf])
        with self.assertRaises(ValueError):
            p.artifact(self.root / "bad.npz")

    def test_marker_replay_accepts_different_verifier_job_not_modified_evidence(self):
        manifest = self.manifest()
        receipt(self.root, manifest, "preflight")
        evidence = self.root / "preflight" / "result.json"
        p.write_json_exclusive(evidence, {"finite": True})
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ):
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
                p.write_marker(self.root, "preflight", [evidence])
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "456"}):
                p.require_marker(self.root, "preflight")
                with self.assertRaises(ValueError):
                    p.write_marker(self.root, "preflight", [evidence])
            with mock.patch.dict(os.environ, {}, clear=True):
                p.require_marker(self.root, "preflight")
                with self.assertRaises(ValueError):
                    p.write_marker(self.root, "preflight", [evidence])
            evidence.write_text('{"finite":false}')
            with self.assertRaises(ValueError):
                p.require_marker(self.root, "preflight")

    def test_array_marker_unpadded_index_and_runtime_matching(self):
        manifest = self.manifest()
        receipt(self.root, manifest, "cells")
        directory = p.cell_directory(self.root, 0)
        p._initialize_budget(directory, 0)
        evidence = directory / "result.json"
        p.write_json_exclusive(evidence, {"ok": True})
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ):
            with mock.patch.dict(
                os.environ, {"SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0"}
            ):
                p.write_marker(self.root, "cell-0", [evidence])
            p.require_marker(self.root, "cell-0")
            with self.assertRaises(ValueError):
                p.require_marker(self.root, "cell-00")
            with p.NativeStepBudget(directory, 0) as budget:
                budget.reserve(1)
            with self.assertRaises(ValueError):
                p.require_marker(self.root, "cell-0")

    def test_marker_rejects_no_evidence_and_escape(self):
        manifest = self.manifest()
        receipt(self.root, manifest, "preflight")
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ):
            with self.assertRaises(ValueError):
                p.write_marker(self.root, "preflight", [])
            with self.assertRaises(ValueError):
                p.write_marker(self.root, "preflight", [self.root / "../escaped"])

    def test_budget_charge_before_call_survives_restart_and_ceiling(self):
        p._initialize_budget(self.root, 0)
        with p.NativeStepBudget(self.root, 0) as budget:
            budget.reserve(1, "collect", "before first native simulator call")
            budget.record_reset(16)
        with p.NativeStepBudget(self.root, 0) as budget:
            self.assertEqual(budget.total, 1)
            self.assertEqual(budget.snapshot()["reset_count"], 16)
            budget.reserve(119999, "evaluate")
            with self.assertRaises(ValueError):
                budget.reserve(1)
            self.assertEqual(budget.total, 120000)
        with self.assertRaises(FileExistsError):
            p._initialize_budget(self.root, 0)

    def test_budget_tampering_partial_tail_and_replacement_fail(self):
        for corruption in (b"", b'{"partial":', b"{}\n"):
            with self.subTest(
                corruption=corruption
            ), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                p._initialize_budget(root, 0)
                path = root / "native-budget.jsonl"
                with p.NativeStepBudget(root, 0) as budget:
                    budget.reserve(2)
                    if corruption:
                        with path.open("ab") as stream:
                            stream.write(corruption)
                    else:
                        path.write_bytes(b"")
                    with self.assertRaises((ValueError, KeyError)):
                        budget.snapshot()
        p._initialize_budget(self.root, 0)
        with self.assertRaises(ValueError):
            p.NativeStepBudget(self.root, 1)

    def test_invalid_budget_requests_never_charge(self):
        p._initialize_budget(self.root, 0)
        with p.NativeStepBudget(self.root, 0) as budget:
            for value in (-1, 0, 1.0, True, "2"):
                with self.assertRaises(ValueError):
                    budget.reserve(value)
            with self.assertRaises(ValueError):
                budget.reserve(2, "retry")
            self.assertEqual(budget.total, 0)


class SchedulerTests(FileTests):
    def test_bundle_requires_exact_commit_and_immutable_bytes(self):
        bundle = self.root / "deployment.bundle"
        p.write_bytes_exclusive(bundle, b"git bundle fixture")
        commit = "a" * 40
        with mock.patch.object(
            p.subprocess, "check_output", side_effect=[b"ok", f"{commit} HEAD\n"]
        ):
            identity = p.bundle_identity(bundle, commit)
            self.assertEqual(identity["source_commit"], commit)
            self.assertEqual(identity["sha256"], p.sha(bundle))
        with mock.patch.object(
            p.subprocess, "check_output", side_effect=[b"ok", "b" * 40 + " HEAD\n"]
        ):
            with self.assertRaises(ValueError):
                p.bundle_identity(bundle, commit)
        with self.assertRaises(ValueError):
            p.bundle_identity(None, commit)

    def test_accounting_requires_no_live_work_all_nine_and_zero_exit(self):
        rows = "\n".join(
            f"123_{i}|COMPLETED|0:0|10|2026-10-14T00:00:00|2026-10-14T00:00:10|cpu=8,gres/gpu=1"
            for i in range(9)
        )
        with mock.patch.object(
            submitter.subprocess, "check_output", side_effect=["", rows]
        ):
            self.assertEqual(submitter.terminal("123", 9)["expected_rows"], 9)
        for active, output in (
            ("123_1|RUNNING", rows),
            ("", rows.replace("0:0", "1:0", 1)),
            ("", rows.replace("COMPLETED", "FAILED", 1)),
            ("", rows.splitlines()[0]),
        ):
            with mock.patch.object(
                submitter.subprocess, "check_output", side_effect=[active, output]
            ):
                with self.assertRaises(RuntimeError):
                    submitter.terminal("123", 9)

    def test_ambiguous_submission_preserves_intent_and_prevents_retry(self):
        self.manifest()
        with mock.patch.object(
            submitter.subprocess, "check_output", return_value="123;othercluster"
        ):
            with self.assertRaises(RuntimeError):
                submitter._launch(self.root, "preflight", ["sbatch"])
        self.assertEqual(
            p.read(self.root / "submissions/preflight.json")["status"],
            "ambiguous_submission",
        )
        with mock.patch.object(submitter.subprocess, "check_output") as call:
            with self.assertRaises(FileExistsError):
                submitter._launch(self.root, "preflight", ["sbatch"])
            call.assert_not_called()

    def test_pending_running_or_failed_receipt_never_resubmitted(self):
        manifest = self.manifest()
        receipt(self.root, manifest, "preflight")
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ), mock.patch.object(submitter, "_launch") as launch:
            with self.assertRaises(FileExistsError):
                submitter.submit(self.root, "preflight", True)
            launch.assert_not_called()

    def test_cells_require_predecessor_attestation_and_three_concurrency(self):
        manifest = self.manifest()
        for i in range(9):
            p._initialize_budget(p.cell_directory(self.root, i), i)
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ), mock.patch.object(
            submitter, "attest", return_value={"verified": True}
        ) as attest, mock.patch.object(
            submitter, "_launch", return_value="124"
        ) as launch, mock.patch.object(
            submitter, "handoff"
        ) as handoff:
            self.assertEqual(submitter.submit(self.root, "cells", True), "124")
            attest.assert_called_once_with(self.root, "preflight")
            command = launch.call_args.args[2]
            self.assertIn("--array=0-8%3", command)
            self.assertIn("--partition=gpu-a30", command)
            self.assertIn("--time=08:00:00", command)
            self.assertEqual(launch.call_args.args[3], {"verified": True})
            handoff.assert_called_once_with(self.root, "cells", "124")

    def test_no_cells_submission_if_any_failed_call_was_charged(self):
        manifest = self.manifest()
        for i in range(9):
            p._initialize_budget(p.cell_directory(self.root, i), i)
        with p.NativeStepBudget(p.cell_directory(self.root, 5), 5) as budget:
            budget.reserve(1)
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ), mock.patch.object(submitter, "_launch") as launch:
            with self.assertRaises(ValueError):
                submitter.submit(self.root, "cells")
            launch.assert_not_called()

    def test_finalization_has_cpu_afterok_handoff_and_terminal_attestation(self):
        manifest = self.manifest()
        receipt(self.root, manifest, "finalize")
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ), mock.patch.object(submitter, "_launch", return_value="124") as launch:
            submitter.handoff(self.root, "finalize", "123")
            command = launch.call_args.args[2]
            self.assertIn("--dependency=afterok:123", command)
            self.assertIn("--partition=cpu-zen3", command)
            self.assertIn("advance-finalize", command)
        with mock.patch.object(submitter, "attest") as attest, mock.patch.object(
            submitter, "submit"
        ) as launch:
            submitter.advance(self.root, "finalize")
            attest.assert_called_once_with(self.root, "finalize")
            launch.assert_not_called()

    def test_attestation_checks_markers_and_is_write_once(self):
        manifest = self.manifest()
        receipt(self.root, manifest, "finalize")
        with mock.patch.object(
            p, "authenticate", return_value=(manifest, p.load_protocol())
        ), mock.patch.object(
            submitter, "terminal", return_value={"complete": True}
        ), mock.patch.object(
            p, "require_marker", return_value={"marker_sha256": "d" * 64}
        ):
            first = submitter.attest(self.root, "finalize")
            self.assertEqual(first, submitter.attest(self.root, "finalize"))
            with mock.patch.object(
                submitter, "terminal", return_value={"complete": False}
            ):
                with self.assertRaises(ValueError):
                    submitter.attest(self.root, "finalize")

    def test_shell_syntax_and_separate_verifier_process(self):
        script = SCRIPT.with_name("posterior_reward_job.sh")
        subprocess.run(["bash", "-n", str(script)], check=True)
        source = script.read_text()
        self.assertIn('"verify-$POSTERIOR_STAGE"', source)
        self.assertIn('"$POSTERIOR_STAGE" "${POSTERIOR_ARGS[@]}"', source)
        self.assertIn("JAX_PLATFORMS=cpu", source)
        self.assertNotIn("reward_exploration_runner", source)


class RmtTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.protocol = p.load_protocol()
        selections = {}
        for group in ("hz", "hmean"):
            choices = {}
            for seed in range(3):
                candidate = dict(
                    update=10, validation={"nll": 0.1}, train={"nll": 0.05}
                )
                choices[str(seed)] = candidate
                directory = self.root / "heads" / f"reward-{group}-a0.01-s{seed}"
                p.write_json_exclusive(
                    directory / "selection.json",
                    dict(
                        seed=seed,
                        alpha=0.01,
                        width=128,
                        depth=2,
                        classes=3,
                        selected=candidate,
                        candidates=[candidate],
                    ),
                )
                np.savez(
                    directory / "normalization.npz", mean=np.zeros(4), std=np.ones(4)
                )
                p.write_bytes_exclusive(
                    directory / "checkpoint-00010.pkl",
                    b"authenticated checkpoint fixture",
                )
            selections[group] = dict(alpha=0.01, selections=choices)
        p.write_json_exclusive(self.root / "selections.json", selections)
        p.write_json_exclusive(self.root / "report.json", {"selections": selections})
        p.write_json_exclusive(
            self.root / "manifest.json",
            dict(source_commit=self.protocol["rmt_source_commit"], preflight=False),
        )
        inventory = {
            str(path.relative_to(self.root)): p.sha(path)
            for path in self.root.rglob("*")
            if path.is_file()
        }
        p.write_json_exclusive(
            self.root / "evaluated.json", dict(stage="evaluated", artifacts=inventory)
        )
        self.protocol["rmt_report_sha256"] = p.sha(self.root / "report.json")
        self.protocol["rmt_evaluated_sha256"] = p.sha(self.root / "evaluated.json")
        p.write_json_exclusive(
            self.root / "verified.json",
            dict(
                exact_replay=True,
                report_sha256=self.protocol["rmt_report_sha256"],
                evaluated_sha256=self.protocol["rmt_evaluated_sha256"],
                source_commit=self.protocol["rmt_source_commit"],
                frozen_digest=self.protocol["rmt_frozen_digest"],
                slurm_job="historical-job",
            ),
        )

    def test_authenticates_selected_weights_and_norms_for_all_seeds(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "new-job"}):
            binding = p._rmt_identity(self.root, self.protocol)
        self.assertEqual(set(binding["selected_heads"]), {"hz", "hmean"})
        for group in ("hz", "hmean"):
            self.assertEqual(set(binding["selected_heads"][group]), {"0", "1", "2"})
            self.assertEqual(binding["selected_heads"][group]["0"]["update"], 10)
            self.assertEqual(len(binding["selected_heads"][group]["0"]["files"]), 3)

    def test_changed_head_rejected_even_with_resealed_inventory(self):
        path = self.root / "heads/reward-hz-a0.01-s0/checkpoint-00010.pkl"
        path.write_bytes(b"changed checkpoint")
        with self.assertRaises(ValueError):
            p._rmt_identity(self.root, self.protocol)
        seal = p.read(self.root / "evaluated.json")
        seal["artifacts"][str(path.relative_to(self.root))] = p.sha(path)
        (self.root / "evaluated.json").write_bytes(p.canonical(seal) + b"\n")
        marker = p.read(self.root / "verified.json")
        marker["evaluated_sha256"] = p.sha(self.root / "evaluated.json")
        (self.root / "verified.json").write_bytes(p.canonical(marker) + b"\n")
        with self.assertRaises(ValueError):
            p._rmt_identity(self.root, self.protocol)

    def test_wrong_frozen_parent_or_unverified_rmt_rejected(self):
        marker = p.read(self.root / "verified.json")
        for field, value in (
            ("exact_replay", False),
            ("frozen_digest", "0" * 64),
            ("source_commit", "0" * 40),
        ):
            corrupted = dict(marker, **{field: value})
            (self.root / "verified.json").write_bytes(p.canonical(corrupted) + b"\n")
            with self.assertRaises(ValueError):
                p._rmt_identity(self.root, self.protocol)
        (self.root / "verified.json").write_bytes(p.canonical(marker) + b"\n")
        p._rmt_identity(self.root, self.protocol)


if __name__ == "__main__":
    unittest.main()
