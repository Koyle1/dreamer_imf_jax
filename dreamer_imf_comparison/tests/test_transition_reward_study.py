from __future__ import annotations

import unittest

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
                    "direct_transition_reward": 0.5,
                    "original_trajectory_imf": 0.25,
                    "shortcut_forcing": 0.4,
                },
                {
                    "direct_transition_reward": 0.7,
                    "original_trajectory_imf": 0.4,
                    "shortcut_forcing": 0.6,
                },
            ]
        )
        self.assertAlmostEqual(result["direct_transition_reward"], 0.6)
        self.assertAlmostEqual(result["original_trajectory_imf"], 0.325)
        self.assertAlmostEqual(result["shortcut_forcing"], 0.5)

    def test_fail_closed_self_test(self) -> None:
        study.self_test()


if __name__ == "__main__":
    unittest.main()
