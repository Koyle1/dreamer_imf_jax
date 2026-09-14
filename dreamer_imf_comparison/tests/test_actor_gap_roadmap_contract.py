from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import actor_gap_roadmap_study as study
from dreamer_imf_compare.artifacts import write_json_atomic


class ActorGapRoadmapContractTests(unittest.TestCase):
    def test_cell_creation_and_strict_replay_are_separate_process_commands(
        self,
    ) -> None:
        source_root = Path(study.__file__).resolve().parents[2]
        runner = (
            source_root
            / "dreamer_imf_comparison/scripts/run_actor_gap_roadmap_study.py"
        ).read_text(encoding="utf-8")
        sbatch = (
            source_root
            / "dreamer_imf_comparison/cluster/actor_gap_roadmap/actor_gap_roadmap.sbatch"
        ).read_text(encoding="utf-8")
        for stage in ("diagnostic", "model", "evaluation"):
            self.assertIn(f'"verify-{stage}-cell"', runner)
            self.assertIn(f'"${{RUNNER}}" verify-{stage}-cell', sbatch)
            creation = sbatch.index(f'"${{RUNNER}}" {stage}-cell')
            verification = sbatch.index(f'"${{RUNNER}}" verify-{stage}-cell')
            self.assertLess(creation, verification)

    def test_source_manifest_closes_over_flowmpc_dependency_module(self) -> None:
        source_root = Path(study.__file__).resolve().parents[2]
        files = study.benchmark._source_files(source_root)
        self.assertIn(
            "dreamer_imf_comparison/dreamer_imf_compare/flowmpc_actor_study.py",
            files,
        )

    def _dependency_manifest(self) -> dict:
        evaluation_cells = []
        next_seed = 10_000
        for world_seed in study.WORLD_SEEDS:
            for actor_seed in study.ACTOR_SEEDS:
                evaluation_cells.append(
                    {
                        "cell_id": f"old-evaluation-{world_seed}-{actor_seed}",
                        "world_model_seed": world_seed,
                        "actor_seed": actor_seed,
                        "evaluation_seeds": list(range(next_seed, next_seed + 5)),
                    }
                )
                next_seed += 5
        return {
            "evaluation_cells": evaluation_cells,
            "inference_tuning": {"environment_seed": 999},
        }

    def test_matrix_is_the_exact_declared_cartesian_product(self) -> None:
        dependency = self._dependency_manifest()
        matrix = study.build_study_matrix(dependency, evaluation_episodes=2)

        self.assertEqual(len(matrix["diagnostic_cells"]), 3)
        self.assertEqual(len(matrix["model_cells"]), 21)
        self.assertEqual(
            len(matrix["evaluation_cells"]),
            len(study.FRESH_ARMS) * len(study.WORLD_SEEDS) * len(study.ACTOR_SEEDS),
        )
        self.assertEqual(
            {
                (row["family"], row["world_model_seed"], row.get("actor_seed"))
                for row in matrix["model_cells"]
            },
            {
                (
                    family,
                    world_seed,
                    (
                        actor_seed
                        if family in study.ACTOR_CONDITIONED_MODEL_FAMILIES
                        else None
                    ),
                )
                for family in study.MODEL_FAMILIES
                for world_seed in study.WORLD_SEEDS
                for actor_seed in (
                    study.ACTOR_SEEDS
                    if family in study.ACTOR_CONDITIONED_MODEL_FAMILIES
                    else (None,)
                )
            },
        )
        self.assertEqual(
            {
                (row["arm"], row["world_model_seed"], row["actor_seed"])
                for row in matrix["evaluation_cells"]
            },
            {
                (arm, world_seed, actor_seed)
                for arm in study.FRESH_ARMS
                for world_seed in study.WORLD_SEEDS
                for actor_seed in study.ACTOR_SEEDS
            },
        )
        for stage in ("diagnostic", "model", "evaluation"):
            rows = matrix[f"{stage}_cells"]
            self.assertEqual([row["index"] for row in rows], list(range(len(rows))))
            self.assertEqual(len({row["cell_id"] for row in rows}), len(rows))
            self.assertEqual(len({row["result_path"] for row in rows}), len(rows))
            self.assertEqual(len({row["marker_path"] for row in rows}), len(rows))

    def test_evaluation_inherits_only_the_declared_dependency_seed_prefix(self) -> None:
        dependency = self._dependency_manifest()
        before = copy.deepcopy(dependency)
        matrix = study.build_study_matrix(dependency, evaluation_episodes=2)
        source = {
            (row["world_model_seed"], row["actor_seed"]): row
            for row in dependency["evaluation_cells"]
        }
        for row in matrix["evaluation_cells"]:
            old = source[(row["world_model_seed"], row["actor_seed"])]
            self.assertEqual(row["evaluation_seeds"], old["evaluation_seeds"][:2])
            self.assertEqual(row["dependency_cell_id"], old["cell_id"])
        self.assertEqual(dependency, before)

    def test_matrix_rejects_an_episode_request_beyond_frozen_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            study.build_study_matrix(
                self._dependency_manifest(),
                evaluation_episodes=study.dependency.EVALUATION_EPISODES + 1,
            )

    def test_manifest_declares_every_arm_and_only_valid_factorial_contrasts(
        self,
    ) -> None:
        dependency = self._dependency_manifest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependency["reward_root"] = str(root / "reward")
            dependency_root = root / "dependency"
            dependency_root.mkdir()
            reward_root = root / "reward"
            reward_root.mkdir()
            source_root = root / "sources"
            source_root.mkdir()
            reward_sources = []
            for world_seed in study.WORLD_SEEDS:
                row = {"world_model_seed": world_seed}
                for path_key, digest_key in (
                    ("reward_result", "reward_result_sha256"),
                    ("reward_checkpoint", "reward_checkpoint_sha256"),
                    ("reward_marker", "reward_marker_sha256"),
                    ("dataset", "dataset_file_sha256"),
                    ("world_model_checkpoint", "world_model_checkpoint_sha256"),
                ):
                    source_path = source_root / f"{world_seed}-{path_key}.json"
                    write_json_atomic(
                        source_path, {"world_model_seed": world_seed, "kind": path_key}
                    )
                    row[path_key] = str(source_path.resolve(strict=True))
                    row[digest_key] = study.benchmark.file_sha256(source_path)
                reward_sources.append(row)
            dependency["reward_sources"] = reward_sources
            write_json_atomic(dependency_root / "manifest.json", {"stub": 1})
            write_json_atomic(dependency_root / "report.json", {"stub": 2})
            authenticated = {
                "root": str(dependency_root.resolve(strict=True)),
                "source_commit": study.DEPENDENCY_SOURCE_COMMIT,
                "manifest": dependency,
                "report": {},
                "manifest_file_sha256": study.benchmark.file_sha256(
                    dependency_root / "manifest.json"
                ),
                "report_file_sha256": study.benchmark.file_sha256(
                    dependency_root / "report.json"
                ),
                "reward_root": str(reward_root.resolve(strict=True)),
            }
            source_manifest = {
                "git": {
                    "commit_status": "complete",
                    "dirty_patch_status": "complete",
                    "untracked_status": "complete",
                    "dirty_patch_bytes": 0,
                    "untracked_files": [],
                    "commit": "c" * 40,
                }
            }
            with mock.patch.object(
                study, "authenticate_dependency", return_value=authenticated
            ), mock.patch.object(
                study.benchmark, "build_source_manifest", return_value=source_manifest
            ), mock.patch.object(
                study.benchmark, "validate_source_manifest"
            ), mock.patch.object(
                study, "_dependency_manifest", return_value=dependency
            ), mock.patch.object(
                study,
                "DEPENDENCY_MANIFEST_FILE_SHA256",
                authenticated["manifest_file_sha256"],
            ), mock.patch.object(
                study,
                "DEPENDENCY_REPORT_FILE_SHA256",
                authenticated["report_file_sha256"],
            ), mock.patch.object(
                study, "_git_commit", return_value="c" * 40
            ):
                manifest = study.build_manifest(
                    dependency_root, model_updates=7, evaluation_episodes=2
                )

        self.assertEqual(tuple(manifest["all_arms"]), study.ALL_ARMS)
        self.assertEqual(len(set(manifest["all_arms"])), len(study.ALL_ARMS))
        self.assertEqual(manifest["model_training"]["updates"], 7)
        for candidate, baseline in manifest["factorial_contrasts"].values():
            self.assertIn(candidate, study.ALL_ARMS)
            self.assertIn(baseline, study.ALL_ARMS)
            self.assertNotEqual(candidate, baseline)

    def test_training_transition_alignment_is_reset_safe(self) -> None:
        arrays = {
            "train_episode_ids": np.asarray([0], dtype=np.int32),
            "observations": np.asarray(
                [[[0.0], [1.0], [2.0], [3.0], [4.0]]], dtype=np.float32
            ),
            "actions": np.asarray(
                [[[0.0], [10.0], [20.0], [30.0], [40.0]]], dtype=np.float32
            ),
            "rewards": np.asarray(
                [[0.0, 100.0, 200.0, 300.0, 400.0]], dtype=np.float32
            ),
            "continuations": np.asarray([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        }
        states, actions, rewards, next_states, continuations = (
            study._training_transitions(arrays)
        )
        np.testing.assert_array_equal(states[:, 0], [0.0, 1.0])
        np.testing.assert_array_equal(actions[:, 0], [10.0, 20.0])
        np.testing.assert_array_equal(rewards, [100.0, 200.0])
        np.testing.assert_array_equal(next_states[:, 0], [1.0, 2.0])
        np.testing.assert_array_equal(continuations, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
