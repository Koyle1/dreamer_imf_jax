from __future__ import annotations

import copy
import unittest

from dreamer_imf_compare import correct_actor_training as study


class CorrectActorTrainingTests(unittest.TestCase):
    def _stage_marker(self) -> dict:
        marker = {
            "schema_version": study.CLUSTER_STAGE_SCHEMA,
            "status": "verified_complete",
            "profile": "pilot",
            "stage": "dataset",
            "matrix_sha256": "a" * 64,
            "map_sha256": "b" * 64,
            "verified_cell_count": 1,
            "result_files": [
                {"cell_id": "dataset-test", "path": "dataset/result.json", "sha256": "c" * 64}
            ],
            "slurm_job_id": "123",
        }
        marker["stage_verification_sha256"] = study.benchmark.object_sha256(marker)
        return marker

    def test_authenticated_source_stage_marker(self) -> None:
        study.validate_cluster_stage_verification(
            self._stage_marker(),
            stage="dataset",
            expected_cells=1,
            matrix_sha256="a" * 64,
        )

    def test_forged_source_stage_marker_is_rejected(self) -> None:
        for key, value in (
            ("status", "complete"),
            ("verified_cell_count", 2),
            ("matrix_sha256", "d" * 64),
        ):
            marker = copy.deepcopy(self._stage_marker())
            marker[key] = value
            with self.assertRaises(ValueError):
                study.validate_cluster_stage_verification(
                    marker,
                    stage="dataset",
                    expected_cells=1,
                    matrix_sha256="a" * 64,
                )

    def test_training_constants_are_frozen(self) -> None:
        self.assertEqual(study.EXPECTED_CELLS, 288)
        self.assertEqual(study.PREPARATION_UPDATES, 500)
        self.assertEqual(study.ACTOR_UPDATES, 10_000)
        self.assertEqual(study.BEHAVIOR_KL_SCALE, 0.0)
        self.assertEqual(study.RETURN_SCALE_EMA_DECAY, 0.99)

    def test_fail_closed_self_test(self) -> None:
        study.self_test()


if __name__ == "__main__":
    unittest.main()
