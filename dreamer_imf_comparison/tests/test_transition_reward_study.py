from __future__ import annotations

import unittest

import numpy as np

from dreamer_imf_compare import transition_reward_study as study


class TransitionRewardStudyTests(unittest.TestCase):
    def test_frozen_scope_and_counts(self) -> None:
        self.assertEqual(study.TASK, "dmc_reacher_easy")
        self.assertEqual(study.EXPECTED_REWARD_CELLS, 3)
        self.assertEqual(study.EXPECTED_ACTOR_CELLS, 6)
        self.assertEqual(study.REWARD_UPDATES, 10_000)
        self.assertEqual(study.HELDOUT_BATCHES, 32)

    def test_paired_aggregation(self) -> None:
        result = study.aggregate_synthetic_units(
            [
                {
                    "state_action_reward": 0.5,
                    "original_trajectory_imf": 0.25,
                    "shortcut_forcing": 0.4,
                },
                {
                    "state_action_reward": 0.7,
                    "original_trajectory_imf": 0.4,
                    "shortcut_forcing": 0.6,
                },
            ]
        )
        self.assertAlmostEqual(result["state_action_reward"], 0.6)
        self.assertAlmostEqual(result["original_trajectory_imf"], 0.325)
        self.assertAlmostEqual(result["shortcut_forcing"], 0.5)

    def test_fail_closed_self_test(self) -> None:
        study.self_test()

    def test_normalization_uses_training_episodes_only(self) -> None:
        arrays = {
            "train_episode_ids": np.asarray([0], dtype=np.int32),
            "observations": np.asarray(
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]],
                ],
                dtype=np.float32,
            ),
            "continuations": np.ones((2, 3), dtype=np.float32),
        }
        mean, std, count = study._training_observation_statistics(arrays)
        np.testing.assert_array_equal(mean, np.asarray([2.0, 3.0], np.float32))
        np.testing.assert_array_equal(std, np.asarray([1.0, 1.0], np.float32))
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
