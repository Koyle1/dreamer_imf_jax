"""CPU-only adversarial authentication and independent-replay controls."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import parallel_study as study
from dreamer_imf_compare import parallel_runner as runner
from dreamer_imf_compare.parallel_collection import (
    BudgetLedger,
    _history_arrays,
    _write_json,
    _write_npz,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "study"
        self.root.mkdir()
        (self.root / "markers").mkdir()
        (self.root / "submissions").mkdir()
        self.budget = self.base / "budget.jsonl"
        with BudgetLedger(self.budget):
            pass
        write_json(self.root / "manifest.json", {"budget_path": str(self.budget)})

    def make_marker(self, stage="collect", payload=None, extras=()):
        directory = self.root / stage
        directory.mkdir()
        _write_json(directory / "artifact.json", {"result": True})
        study.marker(self.root, stage, directory, payload, additional_files=extras)
        return directory

    def test_marker_hashes_inventory_and_extra_report(self):
        report = self.root / "report.json"
        _write_json(report, {"done": True})
        directory = self.make_marker("verify", extras=(report,))
        verified = study.require(self.root, "verify")
        self.assertIn("report.json", verified["files"])
        report.write_text('{"done":false}')
        with self.assertRaisesRegex(ValueError, "artifact differs"):
            study.require(self.root, "verify")
        # Restore original bytes, then prove new unregistered files also fail.
        report.write_text('{"done":true}\n')
        (directory / "unexpected.txt").write_text("not registered")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            study.require(self.root, "verify")

    def test_stage_budget_prefix_allows_append_but_rejects_reset(self):
        with BudgetLedger(self.budget) as ledger:
            reservation = ledger.reserve(2, "failed preflight", category="preflight")
            ledger.finish(reservation, 1, outcome="failed")
        self.make_marker()
        bound = self.budget.read_bytes()
        with BudgetLedger(self.budget) as ledger:
            ledger.reserve(2, "next attempt", category="preflight")
        study.require(self.root, "collect")
        self.budget.write_bytes(bound.splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "missing or reset"):
            study.require(self.root, "collect")

    def test_dataset_payload_must_name_a_hashed_artifact(self):
        outsider = self.base / "outside.npz"
        _write_npz(outsider, {"data": np.zeros(1)})
        self.make_marker(payload={"dataset": str(outsider)})
        with self.assertRaisesRegex(ValueError, "escapes"):
            study.dataset(self.root)

    def test_slurm_receipt_is_bound_to_stage_marker(self):
        with mock.patch.dict("os.environ", {"SLURM_JOB_ID": "123"}):
            self.make_marker()
        write_json(
            self.root / "submissions/collect.json", {"stage": "collect", "job": "999"}
        )
        with self.assertRaisesRegex(ValueError, "another Slurm job"):
            study.require(self.root, "collect")

    def test_profile_authenticates_prediction_and_fit_before_setup(self):
        with mock.patch.object(
            study, "require", side_effect=[{}, ValueError("changed fit")]
        ) as require, mock.patch.object(study, "setup") as setup:
            with self.assertRaisesRegex(ValueError, "changed fit"):
                study.profile_stage(self.root, "head", 1, 2)
        self.assertEqual(
            require.call_args_list,
            [mock.call(self.root, "predictions"), mock.call(self.root, "fit-1")],
        )
        setup.assert_not_called()

    def test_verify_authenticates_every_fit_before_gpu_setup(self):
        calls = []

        def require(root, stage):
            calls.append(stage)
            if stage == "fit-2":
                raise ValueError("changed fit two")
            return {}

        with mock.patch.object(
            study, "require", side_effect=require
        ), mock.patch.object(study, "setup") as setup:
            with self.assertRaisesRegex(ValueError, "changed fit two"):
                study.verify_stage(self.root)
        self.assertTrue(
            {"fit-0", "fit-1", "fit-2", "evaluate", "profile", "predictions"}.issubset(
                calls
            )
        )
        setup.assert_not_called()

    def test_profile_children_use_importable_module_when_parent_is_main(self):
        teacher = mock.Mock(before="unchanged")
        teacher.frozen_digest.return_value = "unchanged"
        with mock.patch.object(study, "__name__", "__main__"), mock.patch.object(
            study, "require", return_value={}
        ), mock.patch.object(
            study,
            "setup",
            return_value=(teacher, {"cells": {"categorical": "unused"}}, {}),
        ), mock.patch.object(
            study, "dataset", return_value=({}, self.base)
        ), mock.patch.object(
            study, "marker"
        ), mock.patch.object(
            runner, "normalized_data", return_value={}
        ), mock.patch.object(
            runner, "evaluate_baseline"
        ), mock.patch.object(
            runner, "evaluate_head"
        ), mock.patch(
            "dreamer_imf_compare.parallel_frozen.FrozenModel", return_value=teacher
        ), mock.patch.object(
            study.subprocess, "run"
        ) as launch:
            study.evaluate_stage(self.root)
        self.assertEqual(launch.call_count, 12)
        self.assertTrue(
            all(
                call.args[0][3] == "dreamer_imf_compare.parallel_study"
                for call in launch.call_args_list
            )
        )


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "current-source"
        self.protocol_path = self.source / "protocol.json"
        self.protocol = study.read(study.PROTOCOL)
        write_json(self.protocol_path, self.protocol)
        self.commits = {str(self.source): "current-commit"}
        self.cells = {}
        for arm in ("imf", "categorical"):
            stem = "imf-conditional-study" if arm == "imf" else "dreamer-ablation-study"
            parent = self.base / stem / self.protocol[f"{arm}_commit"]
            cell = parent / "cells" / arm / "seed_431"
            cell.mkdir(parents=True)
            self.cells[arm] = cell
            source = self.base / f"{arm}-source"
            self.commits[str(source)] = self.protocol[f"{arm}_commit"]
            write_json(
                parent / "manifest.json",
                {
                    "source_commit": self.protocol[f"{arm}_commit"],
                    "source_root": str(source),
                },
            )
            (cell / "checkpoint_200000.pkl").write_bytes(
                b"immutable checkpoint fixture"
            )
            (cell / "config.yaml").write_text("fixture: true\n")
            evaluation = {
                "native_steps": 200000,
                "returns": [1.0],
                "learner_updates": 12,
            }
            write_json(cell / "evaluation_200000.json", evaluation)
            complete = dict(
                completed=True,
                arm=arm,
                seed=431,
                upstream_commit=self.protocol["upstream_commit"],
                checkpoints=[
                    dict(
                        native_steps=200000,
                        path="checkpoint_200000.pkl",
                        sha256=study.sha(cell / "checkpoint_200000.pkl"),
                    )
                ],
                evaluations=[evaluation],
            )
            write_json(cell / "complete.json", complete)
        self.upstream = self.base / "dreamerv3-upstream-e3f0224"
        self.commits[str(self.upstream)] = self.protocol["upstream_commit"]
        self.budget, self.root = (
            self.base / "shared-budget.jsonl",
            self.base / "new-study",
        )
        self.patches = [
            mock.patch.object(study, "BASE", self.base),
            mock.patch.object(study, "SOURCE", self.source),
            mock.patch.object(study, "PROTOCOL", self.protocol_path),
            mock.patch.object(
                study, "clean_commit", side_effect=lambda p: self.commits[str(p)]
            ),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_bundle_binds_exact_main_and_rejects_changed_bytes(self):
        bundle = self.base / "source.bundle"
        bundle.write_bytes(b"bundle fixture")
        with mock.patch.object(
            study.subprocess,
            "check_output",
            return_value="current-commit refs/heads/main\n",
        ):
            study.register(self.root, self.budget, bundle)
        manifest, _ = study.authenticate(self.root)
        self.assertEqual(manifest["deployment_bundle"]["sha256"], study.sha(bundle))
        bundle.write_bytes(b"altered bundle")
        with self.assertRaisesRegex(ValueError, "bundle changed"):
            study.authenticate(self.root)

    def test_bundle_other_branch_or_commit_rejected_before_registration(self):
        bundle = self.base / "wrong.bundle"
        bundle.write_bytes(b"bundle fixture")
        for heads in (
            "wrong-commit refs/heads/main\n",
            "current-commit refs/heads/other\n",
        ):
            with mock.patch.object(
                study.subprocess, "check_output", return_value=heads
            ):
                with self.assertRaisesRegex(ValueError, "bundle does not bind"):
                    study.register(self.root, self.budget, bundle)
        self.assertFalse(self.root.exists())

    def test_registration_retains_failed_and_pending_budget_charges(self):
        with BudgetLedger(self.budget) as ledger:
            ticket = ledger.reserve(2, "failed attempt", category="preflight")
            ledger.finish(ticket, 1, outcome="failed")
            ledger.reserve(2, "interrupted attempt", category="preflight")
        original = self.budget.read_bytes()
        study.register(self.root, self.budget)
        self.assertEqual(original, self.budget.read_bytes())
        manifest, _ = study.authenticate(self.root)
        self.assertEqual(manifest["budget_at_registration"]["charged"], 4)
        self.assertEqual(manifest["budget_at_registration"]["actual"], 1)
        self.assertEqual(len(manifest["budget_at_registration"]["pending"]), 1)

    def test_authentication_rechecks_upstream_and_parent_checkout(self):
        study.register(self.root, self.budget)
        study.authenticate(self.root)
        self.commits[str(self.upstream)] = "changed"
        with self.assertRaisesRegex(ValueError, "upstream checkout changed"):
            study.authenticate(self.root)
        self.commits[str(self.upstream)] = self.protocol["upstream_commit"]
        self.commits[str(self.base / "imf-source")] = "changed"
        with self.assertRaisesRegex(ValueError, "dependency source checkout changed"):
            study.authenticate(self.root)

    def test_incomplete_progress_is_not_mistaken_for_checkpoint_provenance(self):
        (self.cells["imf"] / "complete.json").unlink()
        write_json(
            self.cells["imf"] / "progress.json",
            {"native_steps": 200000, "evaluations": []},
        )
        with self.assertRaises(FileNotFoundError):
            study.register(self.root, self.budget)
        self.assertFalse(self.root.exists())

    def test_parent_seed_and_exact_200k_evaluation_are_authenticated(self):
        path = self.cells["imf"] / "complete.json"
        value = study.read(path)
        value["seed"] = 432
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "completion identity"):
            study.register(self.root, self.budget)
        value["seed"] = 431
        write_json(path, value)
        write_json(
            self.cells["imf"] / "evaluation_200000.json", {"native_steps": 300000}
        )
        with self.assertRaisesRegex(ValueError, "evaluation differs"):
            study.register(self.root, self.budget)


class FakeTeacher:
    feature_dim = 2
    obs_keys = ("q",)

    def initial(self):
        return np.zeros(2, np.float32)

    def observe(self, carry, obs, action, seed):
        feature = (0.25 * carry + obs["q"] + action[0] + seed * 0.0001).astype(
            np.float32
        )
        return feature.copy(), feature.copy()


class RawReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.location = Path(self.temporary.name)
        self.teacher = FakeTeacher()
        raw = [
            dict(q=np.asarray([0, 0], np.float32), reward=np.float32(0)),
            dict(q=np.asarray([1, 2], np.float32), reward=np.float32(0.1)),
        ]
        base_actions = np.asarray([[0.1]], np.float32)
        carry, initial = self.teacher.observe(
            self.teacher.initial(), raw[0], np.zeros(1), 101
        )
        carry, anchor = self.teacher.observe(carry, raw[1], base_actions[0], 102)
        base = _history_arrays(
            raw,
            [x["q"] for x in raw],
            base_actions,
            [0.1],
            [initial, anchor],
            [101, 102],
            [False],
            np.zeros(1),
        )
        _write_npz(self.location / "episode-000.npz", base)
        plans, observations, rewards, targets = [], [], [], []
        for plan in range(4):
            actions = np.asarray([[0.2 + plan * 0.1], [0.4]], np.float32)
            future_raw = [
                raw[1],
                dict(q=np.asarray([plan + 2, 4], np.float32), reward=np.float32(0.2)),
                dict(q=np.asarray([plan + 3, 5], np.float32), reward=np.float32(0.3)),
            ]
            branch_features, state = [anchor], carry.copy()
            for j in range(2):
                state, feature = self.teacher.observe(
                    state, future_raw[j + 1], actions[j], 103 + j
                )
                branch_features.append(feature)
            history = _history_arrays(
                future_raw,
                [x["q"] for x in future_raw],
                actions,
                [0.2, 0.3],
                branch_features,
                [102, 103, 104],
                [False, False],
                base_actions[0],
            )
            _write_npz(self.location / f"branch-000-001-{plan}.npz", history)
            if plan == 0:
                _write_npz(self.location / "branch-000-001-0-duplicate.npz", history)
            plans.append(actions)
            observations.append([x["q"] for x in future_raw[1:]])
            rewards.append([0.2, 0.3])
            targets.append(branch_features[1:])
        self.data = dict(
            episode=np.zeros(4, np.int32),
            anchor=np.ones(4, np.int32),
            plan=np.arange(4),
            mode=np.zeros(4),
            split=np.zeros(4),
            start=np.repeat(anchor[None], 4, axis=0),
            targets=np.asarray(targets),
            actions=np.asarray(plans),
            observations=np.asarray(observations),
            initial_observation=np.repeat(raw[1]["q"][None], 4, axis=0),
            rewards=np.asarray(rewards),
        )

    def test_raw_reconstruction_checks_seed_action_and_anchor_indexing(self):
        starts, targets = study.reconstruct_beliefs(
            self.teacher, self.location, self.data, [3, 1, 0, 2]
        )
        np.testing.assert_array_equal(starts, self.data["start"][[3, 1, 0, 2]])
        np.testing.assert_array_equal(targets, self.data["targets"][[3, 1, 0, 2]])
        corrupted = dict(self.data, targets=np.full_like(self.data["targets"], 900))
        _, independent = study.reconstruct_beliefs(
            self.teacher, self.location, corrupted, np.arange(4)
        )
        self.assertFalse(np.array_equal(independent, corrupted["targets"]))
        self.data["actions"][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "branch actions"):
            study.reconstruct_beliefs(self.teacher, self.location, self.data, [0])

    def test_collection_grid_duplicate_raw_and_budget_counts_are_rechecked(self):
        protocol = dict(
            episodes=1,
            anchors=[1],
            horizon=2,
            action_repeat=2,
            duplicate_branches=1,
            episode_split=[1, 0, 0],
        )
        _write_npz(
            self.location / "duplicates.npz", {k: v[:1] for k, v in self.data.items()}
        )
        _write_json(
            self.location / "manifest.json",
            dict(
                actual_steps=24,
                charged_steps=24,
                episode_counts=[1, 0, 0],
                duplicate_rows=1,
            ),
        )
        with mock.patch(
            "dreamer_imf_compare.parallel_collection.episode_design",
            return_value=[dict(episode=0, split=0, mode=0)],
        ), mock.patch(
            "dreamer_imf_compare.parallel_collection.duplicate_design",
            return_value=[(0, 1, 0)],
        ), mock.patch(
            "dreamer_imf_compare.parallel_collection.DECISIONS", 2
        ):
            study.verify_collection_artifacts(self.location, self.data, protocol)
            self.data["split"][0] = 2
            with self.assertRaisesRegex(ValueError, "split or collection mode"):
                study.verify_collection_artifacts(self.location, self.data, protocol)


class ProfileValidationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = dict(
            head_seeds=[0, 1, 2], flow_steps=[1, 2, 4], particles=32, horizon=15
        )
        cells = [("imf", 0, 1), ("imf", 0, 4), ("categorical", 0, 1)] + [
            ("head", s, n) for s in range(3) for n in (1, 2, 4)
        ]
        timing = dict(samples=[0.01] * 20, median_seconds=0.01, p95_seconds=0.01)
        self.profiles = {
            f"{m}-{s}-{n}": dict(
                mode=m,
                seed=s,
                flow_steps=n,
                particles=32,
                horizon=15,
                device="NVIDIA A30",
                platform_version="fixture",
                device_identity=dict(
                    node="n1", slurm_job="123", visible_devices="0", ids=[0]
                ),
                precision=dict(student_flow="float32", upstream_actual="bfloat16"),
                batches={
                    b: dict(latent=copy.deepcopy(timing), decoded=copy.deepcopy(timing))
                    for b in ("batch1", "batch64")
                },
            )
            for m, s, n in cells
        }
        self.marker = dict(slurm_job="123")

    def test_complete_matching_physical_gpu_profile_grid(self):
        study.validate_profile_grid(self.profiles, self.marker, self.protocol)
        self.profiles["head-1-2"]["device_identity"]["visible_devices"] = "1"
        with self.assertRaisesRegex(ValueError, "same registered physical"):
            study.validate_profile_grid(self.profiles, self.marker, self.protocol)

    def test_wrong_job_missing_variant_and_timing_summary_fail(self):
        with self.assertRaisesRegex(ValueError, "allocation identity"):
            study.validate_profile_grid(
                self.profiles, {"slurm_job": "999"}, self.protocol
            )
        missing = dict(self.profiles)
        del missing["head-2-4"]
        with self.assertRaisesRegex(ValueError, "profiling grid"):
            study.validate_profile_grid(missing, self.marker, self.protocol)
        self.profiles["head-2-1"]["batches"]["batch1"]["decoded"][
            "median_seconds"
        ] = 0.001
        with self.assertRaisesRegex(ValueError, "summary differs"):
            study.validate_profile_grid(self.profiles, self.marker, self.protocol)


class PreflightReplayTests(unittest.TestCase):
    """A replay proof is written only after every retained input agrees."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "preflight"
        self.location = self.directory / "collection" / "attempt-fixture"
        self.location.mkdir(parents=True)
        self.dataset_path = self.location / "dataset.npz"
        self.proof_path = self.directory / "replay.json"
        self.data = dict(
            start=np.arange(24, dtype=np.float32).reshape(8, 3) / 10,
            targets=np.arange(360, dtype=np.float32).reshape(8, 15, 3) / 100,
            actions=np.zeros((8, 15, 2), np.float32),
        )
        _write_npz(self.dataset_path, self.data)
        self.observations = (
            np.arange(8 * 32 * 15 * 2, dtype=np.float32).reshape(8, 32, 15, 2) / 1000
        )
        self.rewards = np.full((8, 32, 15), 0.25, np.float32)
        self.packet = dict(
            indices=np.arange(8),
            observations=self.observations.copy(),
            rewards=self.rewards.copy(),
        )
        _write_npz(self.directory / "predictions.npz", self.packet)
        self.result = dict(
            dataset=str(self.dataset_path), parent_pid=-1, frozen_digest="unchanged"
        )
        _write_json(self.directory / "result.json", self.result)
        self.teacher = mock.Mock(before="unchanged")
        self.teacher.frozen_digest.return_value = "unchanged"
        self.predictor = object()
        patches = [
            mock.patch.object(study, "setup", return_value=(self.teacher, {}, {})),
            mock.patch.object(
                study, "clean_commit", return_value="fixture-source-commit"
            ),
            mock.patch.object(runner, "load_predictor", return_value=self.predictor),
            mock.patch.object(
                runner, "predictions", return_value=(self.observations, self.rewards)
            ),
            mock.patch(
                "dreamer_imf_compare.parallel_collection.reconstruct_features",
                return_value=(self.data["start"][0], self.data["targets"][0]),
            ),
            mock.patch(
                "dreamer_imf_compare.parallel_frozen.ReacherAdapter",
                side_effect=AssertionError(
                    "replay must not instantiate an environment"
                ),
            ),
            mock.patch.object(
                study, "collect", side_effect=AssertionError("replay must not collect")
            ),
        ]
        self.mocks = []
        for patch in patches:
            self.mocks.append(patch.start())
            self.addCleanup(patch.stop)

    def test_observation_or_reward_mismatch_cannot_write_replay_proof(self):
        for name, message in (
            ("observations", "preflight observations"),
            ("rewards", "preflight rewards"),
        ):
            with self.subTest(name=name):
                packet = {key: value.copy() for key, value in self.packet.items()}
                packet[name].flat[0] += 1
                np.savez(self.directory / "predictions.npz", **packet)
                with self.assertRaisesRegex(ValueError, message):
                    study.preflight_replay(self.root)
                self.assertFalse(self.proof_path.exists())
        self.mocks[4].assert_not_called()

    def test_start_or_posterior_mismatch_cannot_write_replay_proof(self):
        for name, message in (
            ("start", "preflight belief"),
            ("targets", "preflight posterior"),
        ):
            with self.subTest(name=name):
                data = {key: value.copy() for key, value in self.data.items()}
                data[name].flat[0] += 1
                np.savez(self.dataset_path, **data)
                with self.assertRaisesRegex(ValueError, message):
                    study.preflight_replay(self.root)
                self.assertFalse(self.proof_path.exists())

    def test_matching_replay_writes_exact_source_and_artifact_proof_without_environment(
        self,
    ):
        study.preflight_replay(self.root)
        proof = study.read(self.proof_path)
        self.assertTrue(proof["exact"])
        self.assertTrue(proof["independent_process"])
        self.assertEqual(proof["pid"], os.getpid())
        self.assertEqual(proof["source_commit"], "fixture-source-commit")
        self.assertEqual(
            proof["predictions_sha256"], study.sha(self.directory / "predictions.npz")
        )
        self.assertEqual(proof["dataset_sha256"], study.sha(self.dataset_path))
        self.mocks[2].assert_called_once_with(self.directory / "fit", self.teacher)
        self.assertEqual(self.mocks[3].call_args.args[3], 821)
        np.testing.assert_array_equal(self.mocks[3].call_args.args[2], np.arange(8))
        self.mocks[4].assert_called_once_with(
            self.teacher,
            self.location / "episode-000.npz",
            anchor=100,
            branch_path=self.location / "branch-000-100-0.npz",
        )
        self.mocks[5].assert_not_called()
        self.mocks[6].assert_not_called()

    def test_same_process_or_incomplete_replay_indices_fail_without_proof(self):
        write_json(
            self.directory / "result.json", dict(self.result, parent_pid=os.getpid())
        )
        with self.assertRaisesRegex(ValueError, "distinct process"):
            study.preflight_replay(self.root)
        self.assertFalse(self.proof_path.exists())
        self.mocks[2].assert_not_called()
        write_json(self.directory / "result.json", self.result)
        np.savez(
            self.directory / "predictions.npz",
            **dict(self.packet, indices=np.arange(7)),
        )
        with self.assertRaisesRegex(ValueError, "complete preflight replay indices"):
            study.preflight_replay(self.root)
        self.assertFalse(self.proof_path.exists())
        self.mocks[2].assert_not_called()


class FullVerificationControlTests(unittest.TestCase):
    """Exercise final verification orchestration with deterministic toy heads."""

    def setUp(self):
        AuthenticationTests.setUp(self)
        self.protocol = study.read(study.PROTOCOL)
        self.protocol.update(updates=1, validation_period=1)
        self.data = dict(
            split=np.asarray([0, 1, 2]),
            episode=np.arange(3),
            anchor=np.full(3, 100),
            plan=np.zeros(3),
            start=np.zeros((3, 2), np.float32),
            targets=np.zeros((3, 15, 2), np.float32),
            actions=np.zeros((3, 15, 1), np.float32),
            observations=np.zeros((3, 15, 2), np.float32),
            rewards=np.zeros((3, 15), np.float64),
            initial_observation=np.zeros((3, 2), np.float32),
        )
        self.norm = runner.normalized_data(self.data)
        indices = np.asarray([2])
        observations, rewards = np.zeros((1, 32, 15, 2), np.float32), np.zeros(
            (1, 32, 15), np.float32
        )
        self.packet = dict(indices=indices, observations=observations, rewards=rewards)
        for seed in range(3):
            train = self.root / "fit" / str(seed)
            train.mkdir(parents=True)
            _write_npz(train / "normalization.npz", self.norm)
            with (train / "checkpoint_00001.pkl").open("wb") as stream:
                pickle.dump({"fixture": np.zeros(1, np.float32)}, stream)
            _write_npz(
                train / "validation_00001.npz",
                dict(self.packet, indices=np.asarray([1])),
            )
            row = dict(
                update=1,
                score=0.0,
                observation_mse=0.0,
                cumulative_reward_mse=0.0,
                checkpoint="checkpoint_00001.pkl",
            )
            write_json(
                train / "selection.json",
                dict(
                    seed=seed,
                    preflight=False,
                    updates=1,
                    selected="checkpoint_00001.pkl",
                    score=0.0,
                    validations=[row],
                    frozen_digest="unchanged",
                ),
            )
            directory = self.root / "evaluation" / f"head-{seed}"
            directory.mkdir(parents=True)
            for nfe in (1, 2, 4):
                _write_npz(directory / f"nfe{nfe}.npz", self.packet)
            _write_npz(directory / "composed.npz", self.packet)
            write_json(
                directory / "report.json",
                dict(
                    seed=seed,
                    variants={str(n): {"verified": True} for n in (1, 2, 4)},
                    distribution_checks=[{"compared": True}, {"compared": True}],
                ),
            )
        for arm in ("imf", "categorical"):
            directory = self.root / "evaluation" / arm
            directory.mkdir()
            _write_npz(
                directory / "beliefs.npz",
                dict(
                    indices=indices,
                    start=self.data["start"][indices],
                    targets=self.data["targets"][indices],
                ),
            )
            nfes = (1, 4) if arm == "imf" else (1,)
            for nfe in nfes:
                _write_npz(directory / f"nfe{nfe}.npz", self.packet)
            _write_npz(
                directory / "posterior_heads.npz",
                dict(
                    indices=indices,
                    observations=observations[:, :1],
                    rewards=rewards[:, :1],
                ),
            )
            report = {str(n): {"verified": True} for n in nfes}
            report["posterior_heads"] = {"verified": True}
            if arm == "imf":
                report["persistence_zero"] = {"verified": True}
            write_json(directory / "report.json", report)
        profiles = ProfileValidationTests()
        profiles.setUp()
        for name, value in profiles.profiles.items():
            write_json(self.root / "profiles" / f"{name}.json", value)
        teacher = mock.Mock(before="unchanged", feature_dim=2)
        teacher.frozen_digest.return_value = "unchanged"
        teacher.rollout.side_effect = lambda starts, acts, *args: np.zeros(
            (len(starts), 32, 15, 2), np.float32
        )
        teacher.decode.side_effect = lambda features: (
            np.zeros((*features.shape[:-1], 2), np.float32),
            np.zeros(features.shape[:-1], np.float32),
        )
        manifest = dict(
            budget_path=str(self.budget),
            budget_at_registration=study._budget_prefix(self.budget),
            cells={"categorical": "unused"},
        )
        self.patches = [
            mock.patch.object(study, "require", return_value={"slurm_job": "123"}),
            mock.patch.object(
                study, "setup", return_value=(teacher, manifest, self.protocol)
            ),
            mock.patch.object(study, "dataset", return_value=(self.data, self.base)),
            mock.patch.object(study, "verify_collection_artifacts"),
            mock.patch.object(
                study,
                "reconstruct_beliefs",
                side_effect=lambda model, loc, data, idx: (
                    data["start"][idx],
                    data["targets"][idx],
                ),
            ),
            mock.patch.object(
                runner, "predictions", return_value=(observations, rewards)
            ),
            mock.patch.object(runner, "summary", return_value={"verified": True}),
            mock.patch.object(
                runner, "prefix_rows", return_value=(np.asarray([0]), np.asarray([0]))
            ),
            mock.patch(
                "dreamer_imf_compare.parallel_metrics.compare_distributions",
                return_value={"compared": True},
            ),
            mock.patch(
                "dreamer_imf_compare.parallel_metrics.promotion",
                return_value={"promote": False},
            ),
            mock.patch(
                "dreamer_imf_compare.parallel_frozen.FrozenModel", return_value=teacher
            ),
            mock.patch.object(study, "marker"),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_final_report_contains_all_nfes_and_full_replay_attestation(self):
        study.verify_stage(self.root)
        report = study.read(self.root / "report.json")
        self.assertEqual(set(report["head_variants"]), {"0", "1", "2"})
        self.assertTrue(
            all(set(v) == {"1", "2", "4"} for v in report["head_variants"].values())
        )
        self.assertEqual(report["independent_replay"]["imf_posterior_rows"], 3)
        self.assertTrue(report["independent_replay"]["posterior_heads_replayed"])
        self.assertEqual(report["budget"]["charged_upper_bound"], 0)

    def test_corrupt_posterior_predictions_fail_independent_decoder_replay(self):
        path = self.root / "evaluation/imf/posterior_heads.npz"
        packet = runner.load_npz(path)
        packet["observations"] += 1
        np.savez(path, **packet)
        with self.assertRaisesRegex(ValueError, "posterior decoder observations"):
            study.verify_stage(self.root)
        self.assertFalse((self.root / "report.json").exists())

    def test_validation_cannot_use_a_training_subset(self):
        path = self.root / "fit/0/validation_00001.npz"
        packet = runner.load_npz(path)
        packet["indices"] = np.asarray([0])
        np.savez(path, **packet)
        with self.assertRaisesRegex(ValueError, "complete validation indices"):
            study.verify_stage(self.root)


if __name__ == "__main__":
    unittest.main()
