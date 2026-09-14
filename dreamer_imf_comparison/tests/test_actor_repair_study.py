from __future__ import annotations

import unittest

from dreamer_imf_compare import actor_repair_study as study


class ActorRepairStudyTests(unittest.TestCase):
    def test_matrix_contains_every_preregistered_cell(self) -> None:
        cells = study.build_cell_matrix()
        study.validate_cell_matrix(cells)
        self.assertEqual(len(cells), 59)
        self.assertEqual(sum(cell["stage"] == "random_policy" for cell in cells), 1)
        self.assertEqual(sum(cell["stage"] == "gradient_oracle" for cell in cells), 2)
        self.assertEqual(sum(cell["stage"] == "safe_mpo_bandit" for cell in cells), 2)

    def test_primary_beta_is_not_selected_from_results(self) -> None:
        self.assertEqual(study.PRIMARY_BETA, 0.1)
        self.assertEqual(set(study.BETAS), {0.0, 0.1, 0.3})
        for stage in study.PURE_STAGES + study.WORLD_STAGES:
            observed = {
                cell["behavior_kl_scale"]
                for cell in study.build_cell_matrix()
                if cell["stage"] == stage
            }
            self.assertEqual(observed, set(study.BETAS))

    def test_world_arms_differ_only_by_reward_source(self) -> None:
        self.assertIs(
            study._world_signal("analytic_reward_learned_dynamics"),
            study.previous.analytic_reacher_reward,
        )
        self.assertIsNone(study._world_signal("learned_reward_learned_dynamics"))

    def test_evaluation_scenarios_are_shared_and_deterministic(self) -> None:
        first = study.shared_evaluation_seeds(25)
        second = study.shared_evaluation_seeds(25)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))


if __name__ == "__main__":
    unittest.main()
