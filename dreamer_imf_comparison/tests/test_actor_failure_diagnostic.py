"""Dependency-light contracts for the decisive actor-failure diagnostic."""

from __future__ import annotations

import unittest

from dreamer_imf_compare import actor_failure_diagnostic as study


class ActorFailureDiagnosticTest(unittest.TestCase):
    def test_matrix_is_complete_and_exact(self) -> None:
        cells = study.build_cell_matrix()
        study.validate_cell_matrix(cells)
        self.assertEqual(len(cells), 49)
        self.assertEqual(sum(cell["stage"] == "actor" for cell in cells), 48)

    def test_evaluation_scenarios_do_not_depend_on_arm(self) -> None:
        seeds = study.shared_evaluation_seeds(50)
        self.assertEqual(seeds, study.shared_evaluation_seeds(50))
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_each_arm_has_independent_actor_repeats(self) -> None:
        cells = [cell for cell in study.build_cell_matrix() if cell["stage"] == "actor"]
        for arm in study.ARMS:
            selected = [cell for cell in cells if cell["arm"] == arm]
            self.assertEqual(len(selected), 8)
            self.assertEqual(
                {cell["world_model_seed"] for cell in selected},
                set(study.WORLD_MODEL_SEEDS),
            )
            self.assertEqual(
                {cell["actor_seed"] for cell in selected}, set(study.ACTOR_SEEDS)
            )

    def test_signal_ladder_is_exact(self) -> None:
        self.assertEqual(study.signal_functions("base_mtp_pmpo"), (None, None))
        reward, continuation = study.signal_functions("analytic_reward_pmpo")
        self.assertIs(reward, study.analytic_reacher_reward)
        self.assertIsNone(continuation)
        reward, continuation = study.signal_functions(
            "analytic_reward_unit_continuation_pmpo"
        )
        self.assertIs(reward, study.analytic_reacher_reward)
        self.assertIs(continuation, study.unit_continuation)


if __name__ == "__main__":
    unittest.main()
