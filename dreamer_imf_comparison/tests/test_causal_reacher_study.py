from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dreamer_imf_compare import causal_reacher_study as study
from dreamer_imf_compare import matched_objective_benchmark as benchmark


class CausalReacherManifestTests(unittest.TestCase):
    def make_pilot(self, root: Path) -> None:
        cells = []
        for seed in study.WORLD_MODEL_SEEDS:
            dataset_id = f"dataset-{seed}"
            dataset_directory = root / "dataset" / dataset_id
            dataset_directory.mkdir(parents=True)
            (dataset_directory / "dataset.npz").write_bytes(f"dataset-{seed}".encode())
            (dataset_directory / "result.json").write_text(
                json.dumps(
                    {
                        "dataset_file_sha256": benchmark.file_sha256(
                            dataset_directory / "dataset.npz"
                        ),
                        "dataset_sha256": f"payload-{seed}",
                    }
                )
            )
            cells.append(
                {
                    "stage": "dataset",
                    "cell_id": dataset_id,
                    "task": study.TASK,
                    "world_model_seed": seed,
                }
            )
            for stage in ("world_model", "rollout", "actor"):
                cell_id = f"shortcut-{stage}-{seed}"
                cell = {
                    "stage": stage,
                    "cell_id": cell_id,
                    "task": study.TASK,
                    "world_model_seed": seed,
                    "arm": "shortcut_forcing",
                    "candidate_id": "shortcut-selected",
                    "budget_track": "equal_updates",
                }
                if stage == "rollout":
                    cell["nfe"] = 4
                if stage == "actor":
                    cell["actor_seed"] = study.ACTOR_SEED
                cells.append(cell)
                result_directory = root / stage / cell_id
                result_directory.mkdir(parents=True)
                (result_directory / "result.json").write_text("{}\n")
        (root / "frozen_protocol.json").write_text(
            json.dumps(
                {
                    "profiles": {
                        "pilot": {
                            "budgets": {
                                "equal_updates": {
                                    "world_model_updates": 10_000,
                                    "actor_updates": 10_000,
                                }
                            }
                        }
                    },
                    "evaluation": {"primary_nfe": {"shortcut_forcing": 4}},
                }
            )
        )
        (root / "matrix.json").write_text(
            json.dumps(
                {
                    "source_sha256": "old-source",
                    "protocol_sha256": "old-protocol",
                    "cells": cells,
                }
            )
        )
        (root / "hpo_selection.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "selection_sha256": "selection",
                    "selected": {
                        "shortcut_forcing": {
                            "candidate_id": "shortcut-selected",
                            "overrides": {
                                "training_k_max": 8,
                                "objective_loss_scale": 0.25,
                            },
                        },
                        "trajectory_imf": {
                            "candidate_id": "imf-selected",
                            "overrides": {
                                "imf_boundary_fraction": 0.25,
                                "objective_loss_scale": 0.25,
                            },
                        },
                    },
                }
            )
        )

    def test_manifest_schedules_only_imf_and_keeps_shortcut_reference_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pilot(root)
            with mock.patch.object(study, "_git_commit", return_value="a" * 40):
                manifest = study.build_manifest(root)
            self.assertEqual(len(manifest["scheduled_cells"]), 2)
            self.assertTrue(
                all(row["arm"] == "trajectory_imf" for row in manifest["scheduled_cells"])
            )
            self.assertTrue(
                all(
                    row["mode"] == "reference_only_never_scheduled"
                    for row in manifest["shortcut_baseline"]
                )
            )
            self.assertFalse(manifest["interpretation"]["claim_eligible"])
            self.assertTrue(
                manifest["interpretation"][
                    "additional_counterfactual_labels_for_new_imf_only"
                ]
            )


if __name__ == "__main__":
    unittest.main()
