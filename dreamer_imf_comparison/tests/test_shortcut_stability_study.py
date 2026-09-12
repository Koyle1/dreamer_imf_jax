from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np

from imf_dreamer_jax import DreamerConfig

from dreamer_imf_compare.artifacts import write_json_atomic
from dreamer_imf_compare.matched_objective_benchmark import object_sha256
from dreamer_imf_compare.shortcut_stability_study import (
    RESULT_SCHEMA,
    aggregate,
    prepare_manifest,
    run_entry,
    validate_manifest,
    validate_result,
)


class ShortcutStabilityStudyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archived = self.root / "archived"
        self.output = self.root / "diagnostic"
        self.archived.mkdir()
        config = DreamerConfig(
            observation_shape=(3,),
            action_dim=2,
            deterministic_dim=8,
            stochastic_dim=4,
            embedding_dim=8,
            hidden_dim=16,
            prior="shortcut",
            shortcut_training_k_max=4,
        )
        cells = []
        for task in ("dmc_reacher_easy", "dmc_pendulum_swingup"):
            for seed in (211, 223, 227):
                dataset_id = f"dataset-{task}-{seed}"
                dataset = {
                    "stage": "dataset",
                    "cell_id": dataset_id,
                    "task": task,
                    "world_model_seed": seed,
                }
                cells.append(dataset)
                dataset_directory = self.archived / "dataset" / dataset_id
                dataset_directory.mkdir(parents=True)
                np.savez_compressed(
                    dataset_directory / "dataset.npz",
                    observations=np.zeros((2, 32, 3), np.float32),
                    actions=np.zeros((2, 32, 2), np.float32),
                    rewards=np.zeros((2, 32), np.float32),
                    continuations=np.ones((2, 32), np.float32),
                    is_first=np.zeros((2, 32), np.bool_),
                    train_episode_ids=np.asarray([0], np.int32),
                    test_episode_ids=np.asarray([1], np.int32),
                )
                write_json_atomic(dataset_directory / "result.json", {"status": "complete"})
                world_id = f"world-{task}-{seed}"
                world = {
                    "stage": "world_model",
                    "cell_id": world_id,
                    "task": task,
                    "world_model_seed": seed,
                    "arm": "shortcut_forcing",
                    "budget_track": "equal_updates",
                    "dependencies": [dataset_id],
                }
                cells.append(world)
                world_directory = self.archived / "world_model" / world_id
                world_directory.mkdir(parents=True)
                write_json_atomic(
                    world_directory / "result.json",
                    {
                        "status": "complete",
                        "runtime_config": asdict(config),
                        "final_metrics": {"prior": float(2_000 + seed)},
                    },
                )
        write_json_atomic(self.archived / "matrix.json", {"cells": cells})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_freezes_six_by_eight_paired_cells(self) -> None:
        manifest = prepare_manifest(self.archived, self.output, updates=100)
        validate_manifest(manifest)
        self.assertEqual(len(manifest["entries"]), 48)
        self.assertEqual(len({row["entry_id"] for row in manifest["entries"]}), 48)
        self.assertTrue(all(row["base"]["archived_final_prior_loss"] > 1_000 for row in manifest["entries"]))
        self.assertIsNone(manifest["objective"]["sampling_clip"])

    def test_aggregate_requires_all_repaired_cells_to_pass(self) -> None:
        manifest = prepare_manifest(self.archived, self.output, updates=100)
        for entry in manifest["entries"]:
            result = {
                "schema_version": RESULT_SCHEMA,
                "status": "complete",
                "failure_reason": None,
                "manifest_sha256": manifest["manifest_sha256"],
                "source_commit": manifest["source_commit"],
                "index": entry["index"],
                "entry_id": entry["entry_id"],
                "task": entry["base"]["task"],
                "world_model_seed": entry["base"]["world_model_seed"],
                "variant": entry["variant"],
                "runtime_config": entry["base"]["archived_runtime_config"],
                "archived_world_cell_id": entry["base"]["archived_world_cell_id"],
                "archived_final_prior_loss": entry["base"]["archived_final_prior_loss"],
                "updates_requested": 100,
                "updates_completed": 100,
                "checkpoint_metrics": [{"update": 100, "metrics": {"prior": 1.0}}],
                "final_metrics": {"prior": 1.0},
                "prior_parameter_health": {"finite": True, "max_abs": 1.0, "sha256": "test"},
                "prior_spectral_norms": [{"path": "prior.weight", "shape": [1, 1], "spectral_norm": 1.0}],
                "max_prior_spectral_norm": 1.0,
                "rollout_health": {"horizon": 15, "finite": True, "max_abs": 1.0, "feature_max_abs": 1.0},
                "sampling_clip": None,
                "wall_seconds": 1.0,
            }
            result["result_sha256"] = object_sha256(result)
            validate_result(result, entry, manifest)
            directory = self.output / "cells" / entry["entry_id"]
            directory.mkdir(parents=True)
            write_json_atomic(directory / "result.json", result)
        summary = aggregate(self.output)
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["production_variant_passed"], 6)
        self.assertEqual(summary["cells"], 48)

    def test_one_update_cell_exercises_unclipped_training_and_rollout(self) -> None:
        manifest = prepare_manifest(self.archived, self.output, updates=1)
        result = run_entry(self.output, 7)
        self.assertEqual(result["entry_id"], manifest["entries"][7]["entry_id"])
        self.assertEqual(result["updates_completed"], 1)
        self.assertIsNone(result["sampling_clip"])
        self.assertTrue(result["prior_parameter_health"]["finite"])
        self.assertTrue(result["rollout_health"]["finite"])


if __name__ == "__main__":
    unittest.main()
