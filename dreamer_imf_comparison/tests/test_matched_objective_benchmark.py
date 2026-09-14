from __future__ import annotations

from dataclasses import fields
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare.artifacts import read_json
import dreamer_imf_compare.matched_objective_benchmark as benchmark
from dreamer_imf_compare.matched_objective_protocol import ARM_ORDER


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


class MatchedObjectiveBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(PROJECT / "matched_objective_protocol.json")
        cls.source = benchmark.build_source_manifest(WORKSPACE)

    def test_compute_plan_rederivation_crosses_a_process_boundary(self) -> None:
        compute_cell = {
            "stage": "compute_plan",
            "cell_id": "compute_plan-" + "a" * 24,
            "task": "dmc_reacher_easy",
            "profile": "smoke",
            "source_sha256": "b" * 64,
        }
        dataset_cell = {
            "stage": "dataset",
            "cell_id": "dataset-" + "c" * 24,
            "task": compute_cell["task"],
        }
        matrix = {
            "matrix_sha256": "d" * 64,
            "cells": [dataset_cell, compute_cell],
        }
        plan = {
            "arms": {
                arm: {"compiler_cost_analysis": {}} for arm in benchmark.ARM_ORDER
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            exemplar = benchmark._cell_result_path(output, dataset_cell)
            exemplar.parent.mkdir(parents=True)
            benchmark.write_json_atomic(
                exemplar,
                {"observation_shape": [6], "action_dim": 2},
            )
            with mock.patch.object(benchmark, "_cell_identity"), mock.patch.object(
                benchmark, "_compute_candidates", return_value={}
            ), mock.patch.object(
                benchmark,
                "build_compute_plan_in_fresh_process",
                return_value=(plan, {}),
            ) as independent_build, mock.patch.object(
                benchmark, "validate_compute_plan_cell"
            ) as local_validation, mock.patch.object(
                benchmark, "_validate_compute_files"
            ), mock.patch.object(
                benchmark, "validate_compute_plan_cell_in_fresh_process"
            ) as independent_validation:
                result = benchmark.run_compute_plan_cell(
                    compute_cell,
                    self.protocol,
                    matrix,
                    output,
                )
        self.assertEqual(result["cell_id"], compute_cell["cell_id"])
        self.assertFalse(
            local_validation.call_args.kwargs["rederive_compiler_evidence"]
        )
        independent_build.assert_called_once()
        independent_validation.assert_called_once_with(
            plan,
            self.protocol,
            matrix,
            compute_cell,
            output,
        )
        print("COMPUTE_BUILD_AND_REDERIVATION_BOUNDARIES_VERIFIED")

    def test_exact_smoke_and_pilot_hpo_matrices(self) -> None:
        smoke = benchmark.build_matrix(
            self.protocol, "smoke", source_manifest=self.source
        )
        benchmark.validate_matrix(smoke, self.protocol, self.source)
        self.assertEqual(len(smoke["cells"]), 22)
        self.assertEqual(
            benchmark.expected_matrix_counts(self.protocol, "smoke"),
            {
                "dataset": 1,
                "compute_plan": 1,
                "world_model": 4,
                "rollout": 12,
                "actor": 4,
            },
        )

    def test_git_identity_captures_staged_patch_and_rejectable_untracked_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "benchmark@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Benchmark Test"],
                check=True,
            )
            tracked = root / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "initial"],
                check=True,
            )
            tracked.write_text("VALUE = 2\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            (root / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
            identity = benchmark._git_identity(
                root, ["tracked.py", "untracked.py"]
            )
            self.assertEqual(identity["commit_status"], "complete")
            self.assertEqual(identity["dirty_patch_status"], "complete")
            self.assertGreater(identity["dirty_patch_bytes"], 0)
            self.assertEqual(identity["untracked_status"], "complete")
            self.assertEqual(identity["untracked_files"], ["untracked.py"])
        pilot = benchmark.build_pilot_hpo_matrix(
            self.protocol, source_manifest=self.source
        )
        benchmark.validate_pilot_hpo_matrix(pilot, self.protocol, self.source)
        self.assertEqual(len(pilot["cells"]), 606)
        self.assertEqual(
            benchmark.expected_hpo_matrix_counts(self.protocol),
            {
                "dataset": 6,
                "compute_plan": 24,
                "world_model": 144,
                "rollout": 144,
                "actor": 288,
            },
        )

    def test_git_identity_timeout_is_bounded_and_pilot_freeze_fails_closed(self) -> None:
        timeout = subprocess.TimeoutExpired(cmd=["git"], timeout=5)
        with mock.patch.object(benchmark.subprocess, "run", side_effect=timeout) as run:
            identity = benchmark._git_identity(Path("/tmp/fake-workspace"), ["source.py"])
        self.assertEqual(identity["commit_status"], "timeout_5_seconds")
        self.assertEqual(identity["dirty_patch_status"], "skipped_without_commit")
        self.assertEqual(identity["untracked_status"], "skipped_without_commit")
        self.assertEqual(identity["command_timeout_seconds"], 5)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

        timed_out_source = copy.deepcopy(self.source)
        timed_out_source["git"] = copy.deepcopy(timed_out_source["git"])
        timed_out_source["git"].update(
            {
                "commit_status": "timeout_5_seconds",
                "dirty_patch_status": "skipped_without_commit",
                "untracked_status": "skipped_without_commit",
            }
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            benchmark, "build_source_manifest", return_value=timed_out_source
        ):
            with self.assertRaisesRegex(
                RuntimeError, "clean committed relevant source tree"
            ):
                benchmark.freeze_pilot_hpo_run(
                    self.protocol,
                    Path(directory) / "pilot",
                    workspace=Path(directory),
                )

    def test_source_manifest_covers_certification_code(self) -> None:
        source_paths = set(benchmark._source_files(WORKSPACE))
        required = {
            "dreamer_imf_comparison/scripts/verify_neurips_core.py",
            "dreamer_imf_comparison/scripts/verify_smoke_idempotence.py",
            "dreamer_imf_comparison/tests/test_matched_objective_benchmark.py",
            "dreamer_imf_comparison/tests/test_rollout_replay_validation.py",
            "dreamer_imf_comparison/cluster/neurips/cluster_spec.json",
            "dreamer_imf_comparison/cluster/neurips/supplementary_plan.json",
            "dreamer_imf_comparison/requirements-neurips-cuda12-lock.txt",
            "dreamer_imf_comparison/tests/__init__.py",
            "dreamer_imf_comparison/NEURIPS_READINESS_REPORT.md",
        }
        self.assertTrue(required.issubset(source_paths), required - source_paths)
        for relative in (
            "dreamer_imf_comparison/dreamer_imf_compare/matched_objective_benchmark.py",
            "dreamer_imf_comparison/dreamer_imf_compare/matched_objective_diagnostics.py",
            "dreamer_imf_comparison/dreamer_imf_compare/neurips_controls.py",
        ):
            self.assertNotIn(".local_study", (WORKSPACE / relative).read_text())

    def test_source_manifest_rejects_changed_cluster_execution_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            cluster = (
                workspace / "dreamer_imf_comparison" / "cluster" / "neurips"
            )
            cluster.mkdir(parents=True)
            specification = cluster / "cluster_spec.json"
            specification.write_text('{"array_concurrency": 4}\n', encoding="utf-8")
            relative = "dreamer_imf_comparison/cluster/neurips/cluster_spec.json"
            with mock.patch.object(benchmark, "_REQUIRED_SOURCE_PATHS", (relative,)):
                source = benchmark.build_source_manifest(workspace)
            paths = {entry["path"] for entry in source["files"]}
            self.assertEqual(
                paths,
                {"dreamer_imf_comparison/cluster/neurips/cluster_spec.json"},
            )
            specification.write_text('{"array_concurrency": 8}\n', encoding="utf-8")
            with mock.patch.object(benchmark, "_REQUIRED_SOURCE_PATHS", (relative,)):
                with self.assertRaisesRegex(
                    ValueError, "does not match the current benchmark source tree"
                ):
                    benchmark.validate_source_manifest(source, workspace)

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            cluster = workspace / "dreamer_imf_comparison" / "cluster" / "neurips"
            cluster.mkdir(parents=True)
            outside = workspace / "outside.json"
            outside.write_text('{"array_concurrency": 4}\n', encoding="utf-8")
            specification = cluster / "cluster_spec.json"
            try:
                os.symlink(outside, specification)
            except (OSError, NotImplementedError) as error:  # pragma: no cover
                self.skipTest(f"symlinks unavailable: {error}")
            relative = "dreamer_imf_comparison/cluster/neurips/cluster_spec.json"
            with mock.patch.object(benchmark, "_REQUIRED_SOURCE_PATHS", (relative,)):
                with self.assertRaisesRegex(ValueError, "contains a symlink"):
                    benchmark.build_source_manifest(workspace)

    def test_vectorized_fold_in_schedule_matches_scalar_jax_derivation(self) -> None:
        import jax

        base = benchmark.derive_jax_key("objective-key-test", 17)
        actual = benchmark._folded_key_rows(base, 257)
        expected = np.stack(
            [
                np.asarray(
                    jax.random.key_data(jax.random.fold_in(base, update)),
                    dtype=np.uint32,
                )
                for update in range(257)
            ]
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(benchmark._folded_key_rows(base, 0).shape, (0, 2))

    def test_strong_dataset_validation_rejects_tiny_relabelled_replay(self) -> None:
        matrix = benchmark.build_matrix(
            self.protocol, "smoke", source_manifest=self.source
        )
        cell = next(row for row in matrix["cells"] if row["stage"] == "dataset")
        episodes, transitions = 2, 3
        arrays = {
            "observations": np.zeros((episodes, transitions, 1), np.float32),
            "actions": np.zeros((episodes, transitions, 1), np.float32),
            "rewards": np.zeros((episodes, transitions), np.float32),
            "continuations": np.ones((episodes, transitions), np.float32),
            "is_first": np.asarray(
                [[True, False, False], [True, False, False]], dtype=np.bool_
            ),
            "episode_ids": np.arange(episodes, dtype=np.int32),
            "train_episode_ids": np.asarray([0], dtype=np.int32),
            "test_episode_ids": np.asarray([1], dtype=np.int32),
        }
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "dataset.npz"
            np.savez_compressed(data_path, **arrays)
            result = {
                **benchmark._result_identity(cell, benchmark.DATASET_SCHEMA),
                "matrix_sha256": matrix["matrix_sha256"],
                "resolved_task": benchmark.resolved_task(cell["task"]),
                "collection_policy": self.protocol["data"]["collection_policy"],
                "native_action_steps": episodes * (transitions - 1),
                "episodes": episodes,
                "steps_per_episode": transitions - 1,
                "smoke_truncated_episodes": True,
                "observation_shape": [1],
                "action_dim": 1,
                "train_episode_ids": [0],
                "test_episode_ids": [1],
                "dataset_sha256": benchmark.array_sha256(arrays),
                "dataset_file_sha256": benchmark.file_sha256(data_path),
                "wall_seconds": 0.1,
            }
            with self.assertRaisesRegex(ValueError, "frozen profile"):
                benchmark.validate_dataset_result(
                    result,
                    cell,
                    data_path,
                    protocol=self.protocol,
                    matrix=matrix,
                )

    def test_actor_environment_replay_rejects_forged_rewards_after_rehash(self) -> None:
        matrix = benchmark.build_matrix(
            self.protocol, "smoke", source_manifest=self.source
        )
        cell = next(row for row in matrix["cells"] if row["stage"] == "actor")
        evaluation_seed = benchmark.derive_seed(
            "actor-evaluation",
            cell["task"],
            cell["world_model_seed"],
            cell["actor_seed"],
            0,
        )

        class FakeDMCAdapter:
            action_dim = 1

            def __init__(self, task: str, *, seed: int, action_repeat: int) -> None:
                self.step_index = 0
                self.assertions = (task, seed, action_repeat)

            def reset(self) -> np.ndarray:
                self.step_index = 0
                return np.zeros((1,), dtype=np.float32)

            def step(self, action: np.ndarray):
                self.step_index += 1
                return type(
                    "Transition",
                    (),
                    {
                        "observation": np.asarray([self.step_index], dtype=np.float32),
                        "reward": float(action[0]),
                        "continuation": 0.0 if self.step_index == 2 else 1.0,
                        "is_last": self.step_index == 2,
                    },
                )()

            def close(self) -> None:
                return None

        honest_traces = {
            "actions": np.asarray([[[0.25], [0.5]]], dtype=np.float32),
            "rewards": np.asarray([[0.25, 0.5]], dtype=np.float64),
            "continuations": np.asarray([[1.0, 0.0]], dtype=np.float64),
            "is_last": np.asarray([[False, True]], dtype=np.bool_),
            "lengths": np.asarray([2], dtype=np.int32),
            "evaluation_seeds": np.asarray([evaluation_seed], dtype=np.uint32),
        }
        policy_actions = [0.25, 0.5]
        policy_step = {"value": 0}

        def fake_initial_state(config, batch_size):
            del config, batch_size
            policy_step["value"] = 0
            return None

        def fake_jit_act(params, observation, previous_action, belief, key, config, deterministic):
            del params, observation, previous_action, belief, key, config, deterministic
            value = policy_actions[policy_step["value"]]
            policy_step["value"] += 1
            return np.asarray([[value]], dtype=np.float32), None

        state = SimpleNamespace(params=None)
        config = SimpleNamespace(action_dim=1)
        with (
            mock.patch.object(benchmark, "DMCAdapter", FakeDMCAdapter),
            mock.patch("imf_dreamer_jax.initial_state", fake_initial_state),
            mock.patch("imf_dreamer_jax.jit_act", fake_jit_act),
        ):
            benchmark._validate_actor_environment_replay(
                honest_traces,
                np.asarray([0.75], dtype=np.float64),
                state=state,
                config=config,
                task=cell["task"],
                world_model_seed=int(cell["world_model_seed"]),
                actor_seed=int(cell["actor_seed"]),
                maximum_steps=int(self.protocol["data"]["native_episode_limit"]),
                action_repeat=int(self.protocol["data"]["action_repeat"]),
            )
            # The genuine rewards would be [0.25, 0.5]. This forged pair has
            # the same total, so internal return recomputation alone cannot catch it.
            forged = {
                **honest_traces,
                "rewards": np.asarray([[1.25, -0.5]], dtype=np.float64),
            }
            with self.assertRaisesRegex(ValueError, "replayed DMC"):
                benchmark._validate_actor_environment_replay(
                    forged,
                    np.asarray([0.75], dtype=np.float64),
                    state=state,
                    config=config,
                    task=cell["task"],
                    world_model_seed=int(cell["world_model_seed"]),
                    actor_seed=int(cell["actor_seed"]),
                    maximum_steps=int(self.protocol["data"]["native_episode_limit"]),
                    action_repeat=int(self.protocol["data"]["action_repeat"]),
                )

    def test_source_and_candidate_change_cell_identity(self) -> None:
        original = benchmark.build_matrix(
            self.protocol, "smoke", source_manifest=self.source
        )
        changed_source = copy.deepcopy(self.source)
        changed_source["files"][0]["sha256"] = "0" * 64
        changed_source["source_sha256"] = benchmark.object_sha256(
            {key: value for key, value in changed_source.items() if key != "source_sha256"}
        )
        changed = benchmark.build_matrix(
            self.protocol, "smoke", source_manifest=changed_source
        )
        self.assertNotEqual(original["matrix_sha256"], changed["matrix_sha256"])
        pilot = benchmark.build_pilot_hpo_matrix(
            self.protocol, source_manifest=self.source
        )
        shortcut_worlds = [
            cell
            for cell in pilot["cells"]
            if cell["stage"] == "world_model"
            and cell["arm"] == "shortcut_forcing"
            and cell["task"] == "dmc_reacher_easy"
            and cell["world_model_seed"] == 211
        ]
        self.assertEqual(len(shortcut_worlds), 12)
        self.assertEqual(len({cell["cell_id"] for cell in shortcut_worlds}), 12)
        self.assertEqual(len({cell["candidate_id"] for cell in shortcut_worlds}), 12)

    def _synthetic_trial_manifest(self) -> dict:
        units = [
            (task, int(seed))
            for task in self.protocol["profiles"]["pilot"]["tasks"]
            for seed in self.protocol["profiles"]["pilot"]["world_model_seeds"]
        ]
        trials = []
        for arm in ARM_ORDER:
            for index, candidate in enumerate(
                benchmark.pilot_candidates(self.protocol, arm)
            ):
                rows = []
                for unit_index, (task, seed) in enumerate(units):
                    rows.append(
                        {
                            "task": task,
                            "world_model_seed": seed,
                            "rollout_auc": float(index + 0.001 * unit_index),
                            "nested_actor_return": float(-index + 0.001 * unit_index),
                            "full_forward_backward_train_flops_per_update": float(
                                1000 + index
                            ),
                            "artifact_sha256s": [
                                hashlib.sha256(
                                    f"{arm}-{index}-{unit_index}-{artifact}".encode()
                                ).hexdigest()
                                for artifact in range(4)
                            ],
                        }
                    )
                trials.append(
                    {
                        "arm": arm,
                        "candidate_id": candidate["candidate_id"],
                        "overrides": candidate["overrides"],
                        "units": rows,
                    }
                )
        manifest = {
            "schema_version": "matched-objective-hpo-trials-v1",
            "status": "complete",
            "protocol_sha256": benchmark.protocol_digest(self.protocol),
            "source_sha256": self.source["source_sha256"],
            "selection_profile": "pilot",
            "selection_budget_track": "equal_updates",
            "confirmatory_outcomes_accessed": False,
            "trials": trials,
        }
        manifest["trial_manifest_sha256"] = benchmark.object_sha256(manifest)
        return manifest

    def test_hpo_selection_requires_every_trial_and_unit(self) -> None:
        manifest = self._synthetic_trial_manifest()
        benchmark.validate_hpo_trial_manifest(
            manifest, self.protocol, source_sha256=self.source["source_sha256"]
        )
        selection = benchmark.select_hpo_candidates(
            manifest, self.protocol, source_sha256=self.source["source_sha256"]
        )
        benchmark.validate_hpo_selection_manifest(
            selection, self.protocol, source_sha256=self.source["source_sha256"]
        )
        for arm in ARM_ORDER:
            self.assertEqual(
                selection["selected"][arm]["candidate_id"],
                benchmark.pilot_candidates(self.protocol, arm)[0]["candidate_id"],
            )
        tampered = copy.deepcopy(manifest)
        tampered["trials"][0]["units"].pop()
        tampered["trial_manifest_sha256"] = benchmark.object_sha256(
            {
                key: value
                for key, value in tampered.items()
                if key != "trial_manifest_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "cover every frozen pilot"):
            benchmark.validate_hpo_trial_manifest(
                tampered,
                self.protocol,
                source_sha256=self.source["source_sha256"],
            )

    def test_every_candidate_resolves_every_config_field(self) -> None:
        from imf_dreamer_jax import DreamerConfig

        expected = {field.name for field in fields(DreamerConfig)}
        for arm in ARM_ORDER:
            for candidate in benchmark.pilot_candidates(self.protocol, arm):
                config = benchmark.make_config(
                    self.protocol,
                    "pilot",
                    arm,
                    (5,),
                    2,
                    candidate=candidate,
                )
                self.assertEqual(set(config.__dataclass_fields__), expected)
                self.assertEqual(config.prior, "shortcut" if arm == "shortcut_forcing" else "imf")

    def test_rollout_metric_matches_hand_computed_auc(self) -> None:
        horizons = self.protocol["evaluation"]["rollout_horizons"]
        maximum = max(horizons)
        samples = np.zeros((2, 1, maximum, 1), np.float32)
        for step in range(maximum):
            samples[:, :, step, :] = np.sqrt(step + 1)
        windows = {
            "target_observations": np.zeros((1, maximum, 1), np.float32),
            "target_rewards": np.zeros((1, maximum), np.float32),
            "target_continuations": np.ones((1, maximum), np.float32),
            "training_observation_std": np.ones((1,), np.float32),
        }
        result = benchmark.normalized_rollout_statistics(
            samples,
            np.zeros((2, 1, maximum), np.float32),
            windows,
            horizons,
            np.ones((2, 1, maximum), np.float32),
        )
        self.assertAlmostEqual(
            result["normalized_free_running_rollout_error_auc"], 15.5, places=5
        )
        self.assertEqual(
            result["per_horizon"]["30"]["continuation_predictive_mean_brier"],
            0.0,
        )

    def test_claim_boundary_and_confirmatory_bootstrap_count(self) -> None:
        smoke, _ = benchmark.analyze_units(
            benchmark.synthetic_units(self.protocol, "smoke", effect=0.1),
            self.protocol,
            "smoke",
            resamples=10,
        )
        self.assertFalse(smoke["superiority"]["passed"])
        self.assertFalse(
            smoke["practical_significance"][
                "practically_meaningful_superiority_claim_allowed"
            ]
        )
        self.assertEqual(
            smoke["practical_significance"]["evidence_class"],
            "engineering_smoke",
        )
        with self.assertRaisesRegex(ValueError, "exactly the registered bootstrap count"):
            benchmark.analyze_units(
                benchmark.synthetic_units(
                    self.protocol, "confirmatory", effect=0.1
                ),
                self.protocol,
                "confirmatory",
                resamples=10,
            )

    def _practical_metrics(
        self,
        *,
        rollout_by_track: dict[str, tuple[float, float]] | None = None,
        actor_by_track: dict[str, float] | None = None,
    ) -> dict:
        rollout_by_track = rollout_by_track or {
            track: (0.05, 0.5) for track in benchmark.BUDGET_TRACK_ORDER
        }
        actor_by_track = actor_by_track or {
            track: 0.05 for track in benchmark.BUDGET_TRACK_ORDER
        }
        return {
            track: {
                benchmark.PRIMARY_METRICS[0]: {
                    "contrast": rollout_by_track[track][0],
                    "arm_iqm": {
                        "shortcut_forcing": rollout_by_track[track][1],
                        "trajectory_imf": (
                            rollout_by_track[track][1]
                            - rollout_by_track[track][0]
                        ),
                    },
                },
                benchmark.PRIMARY_METRICS[1]: {
                    "contrast": actor_by_track[track],
                    "arm_iqm": {
                        "shortcut_forcing": 0.4,
                        "trajectory_imf": 0.4 + actor_by_track[track],
                    },
                },
            }
            for track in benchmark.BUDGET_TRACK_ORDER
        }

    def test_practical_significance_requires_both_thresholds_on_both_tracks(self) -> None:
        exact_boundary = benchmark._evaluate_practical_significance(
            self._practical_metrics(),
            self.protocol,
            evidence_class="confirmatory",
            statistical_superiority_passed=True,
        )
        self.assertTrue(exact_boundary["all_thresholds_met"])
        self.assertTrue(
            exact_boundary["practically_meaningful_superiority_claim_allowed"]
        )

        for failed_track in benchmark.BUDGET_TRACK_ORDER:
            with self.subTest(case="rollout_absolute_only", track=failed_track):
                rollout = {
                    track: ((0.05, 1.0) if track == failed_track else (0.05, 0.5))
                    for track in benchmark.BUDGET_TRACK_ORDER
                }
                result = benchmark._evaluate_practical_significance(
                    self._practical_metrics(rollout_by_track=rollout),
                    self.protocol,
                    evidence_class="confirmatory",
                    statistical_superiority_passed=True,
                )
                self.assertFalse(result["tracks"][failed_track][benchmark.PRIMARY_METRICS[0]]["passed"])
                self.assertFalse(result["all_thresholds_met"])
                self.assertFalse(result["practically_meaningful_superiority_claim_allowed"])

            with self.subTest(case="rollout_relative_only", track=failed_track):
                rollout = {
                    track: ((0.04, 0.2) if track == failed_track else (0.05, 0.5))
                    for track in benchmark.BUDGET_TRACK_ORDER
                }
                result = benchmark._evaluate_practical_significance(
                    self._practical_metrics(rollout_by_track=rollout),
                    self.protocol,
                    evidence_class="confirmatory",
                    statistical_superiority_passed=True,
                )
                self.assertFalse(result["tracks"][failed_track][benchmark.PRIMARY_METRICS[0]]["passed"])
                self.assertFalse(result["all_thresholds_met"])

            with self.subTest(case="actor", track=failed_track):
                actor = {
                    track: (0.049 if track == failed_track else 0.05)
                    for track in benchmark.BUDGET_TRACK_ORDER
                }
                result = benchmark._evaluate_practical_significance(
                    self._practical_metrics(actor_by_track=actor),
                    self.protocol,
                    evidence_class="confirmatory",
                    statistical_superiority_passed=True,
                )
                self.assertFalse(result["tracks"][failed_track][benchmark.PRIMARY_METRICS[1]]["passed"])
                self.assertFalse(result["all_thresholds_met"])

        statistically_failed = benchmark._evaluate_practical_significance(
            self._practical_metrics(),
            self.protocol,
            evidence_class="confirmatory",
            statistical_superiority_passed=False,
        )
        self.assertTrue(statistically_failed["all_thresholds_met"])
        self.assertFalse(
            statistically_failed["practically_meaningful_superiority_claim_allowed"]
        )

    def test_rehashed_practical_decision_tamper_is_rejected(self) -> None:
        matrix = benchmark.build_matrix(
            self.protocol, "smoke", source_manifest=self.source
        )
        analysis, _ = benchmark.analyze_units(
            benchmark.synthetic_units(self.protocol, "smoke", effect=0.1),
            self.protocol,
            "smoke",
            resamples=10,
        )
        analysis.update(
            {
                "source_sha256": matrix["source_sha256"],
                "matrix_sha256": matrix["matrix_sha256"],
                "selection_sha256": matrix["selection_sha256"],
            }
        )
        analysis["analysis_sha256"] = benchmark._analysis_digest(analysis)
        benchmark.validate_analysis(analysis, matrix, self.protocol)

        tampered = copy.deepcopy(analysis)
        tampered["practical_significance"]["tracks"]["equal_updates"][
            benchmark.PRIMARY_METRICS[0]
        ]["passed"] = not tampered["practical_significance"]["tracks"][
            "equal_updates"
        ][benchmark.PRIMARY_METRICS[0]]["passed"]
        tampered["analysis_sha256"] = benchmark._analysis_digest(tampered)
        with self.assertRaisesRegex(
            ValueError, "practical-significance decision is not reproducible"
        ):
            benchmark.validate_analysis(tampered, matrix, self.protocol)

    def test_verified_summary_cannot_hide_a_practical_failure(self) -> None:
        matrix = {
            "profile": "confirmatory",
            "cells": [{"cell_id": "one"}],
            "protocol_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "matrix_sha256": "c" * 64,
        }
        practical = {
            "all_thresholds_met": False,
            "practically_meaningful_superiority_claim_allowed": False,
        }
        analysis = {
            "analysis_sha256": "d" * 64,
            "superiority": {"passed": True},
            "practical_significance": practical,
        }
        summary = benchmark._verified_run_summary(matrix, analysis)
        self.assertTrue(summary["superiority"]["passed"])
        self.assertEqual(summary["practical_significance"], practical)
        self.assertFalse(
            summary["practically_meaningful_superiority_claim_allowed"]
        )

    def test_equal_flop_nearest_integer_and_tolerance(self) -> None:
        target, allocation = benchmark.allocate_equal_flops(
            {"shortcut_forcing": 10.0, "trajectory_imf": 4.0},
            reference_arm="shortcut_forcing",
            reference_updates=2,
            tolerance=0.01,
        )
        self.assertEqual(target, 20.0)
        self.assertEqual(allocation["trajectory_imf"]["updates"], 5)
        with self.assertRaisesRegex(ValueError, "above tolerance"):
            benchmark.allocate_equal_flops(
                {"shortcut_forcing": 10.0, "trajectory_imf": 7.0},
                reference_arm="shortcut_forcing",
                reference_updates=1,
                tolerance=0.01,
            )

    def test_jax_update_keys_are_unique_at_confirmatory_length(self) -> None:
        import jax
        import jax.numpy as jnp

        base = benchmark.derive_jax_key("unit-test", "world-objective")
        keys = jax.vmap(
            lambda update: jax.random.key_data(jax.random.fold_in(base, update))
        )(jnp.arange(100_000, dtype=jnp.uint32))
        host = np.asarray(keys, dtype=np.uint32)
        self.assertEqual(np.unique(host, axis=0).shape[0], 100_000)

    def test_objective_schedules_share_canonical_uniforms(self) -> None:
        import jax.numpy as jnp
        from imf_dreamer_jax import (
            sample_shortcut_schedule,
            sample_trajectory_schedule,
        )

        uniforms = jnp.asarray(
            np.random.default_rng(4).uniform(size=(2, 5, 6)), dtype=jnp.float32
        )
        shortcut = sample_shortcut_schedule(
            benchmark.derive_jax_key("unused-shortcut-key"),
            2,
            5,
            k_max=8,
            canonical_uniforms=uniforms[..., :2],
        )
        trajectory = sample_trajectory_schedule(
            benchmark.derive_jax_key("unused-trajectory-key"),
            2,
            5,
            canonical_uniforms=uniforms,
        )
        self.assertTrue(np.isfinite(np.asarray(shortcut.tau)).all())
        self.assertTrue(np.isfinite(np.asarray(trajectory.t)).all())
        repeated = sample_trajectory_schedule(
            benchmark.derive_jax_key("different-unused-key"),
            2,
            5,
            canonical_uniforms=uniforms,
        )
        self.assertTrue(np.array_equal(np.asarray(trajectory.r), np.asarray(repeated.r)))
        self.assertTrue(np.array_equal(np.asarray(trajectory.t), np.asarray(repeated.t)))

    def test_matched_arms_share_exact_recurrence_initialization(self) -> None:
        import jax
        from imf_dreamer_jax import create_agent, world_model_parameter_counts

        key = benchmark.derive_jax_key("shared-recurrence-test")
        states = []
        for arm in ARM_ORDER:
            config = benchmark.make_config(self.protocol, "smoke", arm, (5,), 2)
            states.append(create_agent(config, key))
        left = states[0].params.world_model["recurrence"]
        right = states[1].params.world_model["recurrence"]
        left_leaves, left_tree = jax.tree_util.tree_flatten(left)
        right_leaves, right_tree = jax.tree_util.tree_flatten(right)
        self.assertEqual(left_tree, right_tree)
        for left_value, right_value in zip(left_leaves, right_leaves, strict=True):
            self.assertTrue(np.array_equal(np.asarray(left_value), np.asarray(right_value)))
        shortcut_config = benchmark.make_config(
            self.protocol, "smoke", "shortcut_forcing", (5,), 2
        )
        trajectory_config = benchmark.make_config(
            self.protocol, "smoke", "trajectory_imf", (5,), 2
        )
        shortcut_counts = world_model_parameter_counts(
            states[0].params.world_model, shortcut_config
        )
        trajectory_counts = world_model_parameter_counts(
            states[1].params.world_model, trajectory_config
        )
        self.assertEqual(shortcut_counts.total - shortcut_counts.active, 0)
        self.assertEqual(
            trajectory_counts.total - trajectory_counts.active,
            3 * trajectory_config.deterministic_dim,
        )

    def test_runtime_fingerprint_binds_full_software_environment(self) -> None:
        runtime = benchmark.runtime_fingerprint()
        identity = benchmark.runtime_homogeneity_identity(runtime)
        required = {
            "python_executable",
            "python_executable_sha256",
            "platform_system",
            "platform_release",
            "platform_machine",
            "numpy_version",
            "jax_version",
            "jaxlib_version",
            "dm_control_version",
            "mujoco_version",
            "environment_package_count",
            "environment_packages_sha256",
        }
        self.assertTrue(required.issubset(identity))
        self.assertRegex(identity["python_executable_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(identity["environment_packages_sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(identity["environment_package_count"], 0)

        extended = required - {"jax_version", "jaxlib_version"}
        legacy = {key: value for key, value in runtime.items() if key not in extended}
        legacy_identity = benchmark.runtime_homogeneity_identity(legacy)
        self.assertFalse(extended & set(legacy_identity))
        partial = dict(legacy)
        partial["numpy_version"] = runtime["numpy_version"]
        with self.assertRaisesRegex(ValueError, "extended runtime"):
            benchmark.runtime_homogeneity_identity(partial)


if __name__ == "__main__":
    unittest.main()
