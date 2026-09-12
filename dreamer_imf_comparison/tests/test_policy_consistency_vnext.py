from __future__ import annotations

import unittest

import numpy as np

from dreamer_imf_compare.shared_probe_bank import (
    advantage_training_arrays,
    build_probe_plan,
    shared_probe_manifest,
    validate_shared_probe_bank,
)
from dreamer_imf_compare.advantage_repair_study import materialize_advantage_batch


def arrays() -> dict[str, np.ndarray]:
    episodes, steps, action_dim = 4, 24, 2
    return {
        "observations": np.zeros((episodes, steps, 3), np.float32),
        "actions": np.linspace(
            -0.8, 0.8, episodes * steps * action_dim, dtype=np.float32
        ).reshape(episodes, steps, action_dim),
        "rewards": np.zeros((episodes, steps), np.float32),
        "continuations": np.ones((episodes, steps), np.float32),
        "is_first": np.zeros((episodes, steps), np.bool_),
        "episode_ids": np.arange(episodes, dtype=np.int32),
        "train_episode_ids": np.asarray([0, 1], np.int32),
        "test_episode_ids": np.asarray([2, 3], np.int32),
    }


class SharedProbeBankTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("SHARED_PROBE_BANK_VERIFIED")

    def complete_bank(self) -> dict[str, np.ndarray]:
        plan = build_probe_plan(
            arrays(),
            task="dmc_reacher_easy",
            world_model_seed=211,
            probes=4,
            horizons=(1, 3, 5),
            action_delta=0.5,
            intervention_steps=3,
            stochastic_dim=3,
            draws=2,
        )
        probes = len(plan["episode_ids"])
        candidates = plan["action_sequences"].shape[1]
        horizon_count = len(plan["horizons"])
        return {
            **plan,
            "simulator_returns": np.zeros(
                (probes, candidates, horizon_count), np.float32
            ),
            "horizon_mask": np.ones((probes, horizon_count), np.float32),
            "maximum_replay_error": np.asarray([0.0], np.float64),
            "candidate_pool_size": np.asarray([4], np.int32),
            "selection_horizon": np.asarray([5], np.int32),
            "selection_return_ranges": np.ones((probes,), np.float32),
            "action_delta": np.asarray([0.5], np.float32),
            "intervention_steps": np.asarray([3], np.int32),
        }

    def test_plan_is_deterministic_and_policy_independent(self) -> None:
        first = self.complete_bank()
        second = self.complete_bank()
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        replay = first["action_sequences"][:, 0]
        lower_first_dimension = first["action_sequences"][:, 1]
        self.assertTrue(
            np.any(np.abs(lower_first_dimension[:, :3, 0] - replay[:, :3, 0]) > 0.0)
        )
        np.testing.assert_array_equal(lower_first_dimension[:, 3:], replay[:, 3:])
        manifest = shared_probe_manifest(
            first,
            task="dmc_reacher_easy",
            world_model_seed=211,
            dataset_sha256="dataset",
            action_delta=0.5,
        )
        self.assertTrue(manifest["policy_independent"])
        self.assertIn("same_standard_normal", manifest["noise_rule"])
        self.assertEqual(manifest["candidate_count"], 5)
        self.assertEqual(manifest["intervention_steps"], 3)
        self.assertEqual(manifest["informative_states_at_selection_horizon"], 4)

    def test_one_digest_binds_every_arm_and_mutation_changes_it(self) -> None:
        bank = self.complete_bank()
        validate_shared_probe_bank(bank)
        manifest = shared_probe_manifest(
            bank,
            task="dmc_reacher_easy",
            world_model_seed=211,
            dataset_sha256="dataset",
            action_delta=0.5,
        )
        arm_bindings = {
            arm: manifest["probe_bank_sha256"]
            for arm in ("shortcut_forcing", "trajectory_imf", "causal_trajectory_imf")
        }
        self.assertEqual(len(set(arm_bindings.values())), 1)
        mutated = {name: value.copy() for name, value in bank.items()}
        mutated["action_sequences"][0, 0, 0, 0] += 0.01
        changed = shared_probe_manifest(
            mutated,
            task="dmc_reacher_easy",
            world_model_seed=211,
            dataset_sha256="dataset",
            action_delta=0.5,
        )
        self.assertNotEqual(
            manifest["probe_bank_sha256"], changed["probe_bank_sha256"]
        )
        training = advantage_training_arrays(
            bank, episodes=4, steps=24, horizons=(1, 3, 5)
        )
        self.assertEqual(
            training["advantage_action_sequences"].shape, (4, 24, 5, 5, 2)
        )
        self.assertEqual(int(np.sum(training["advantage_mask"])), 12)

    def test_train_probe_materializes_one_labeled_decision_per_row(self) -> None:
        bank = self.complete_bank()
        materialized = materialize_advantage_batch(
            arrays(), bank, np.asarray([0, 1], np.int32), burn_in=1
        )
        self.assertEqual(materialized["observations"].shape, (2, 2, 3))
        self.assertEqual(
            materialized["advantage_action_sequences"].shape, (2, 2, 5, 5, 2)
        )
        np.testing.assert_array_equal(
            np.asarray(materialized["advantage_mask"][:, 0]), 0.0
        )
        self.assertEqual(
            int(np.sum(np.asarray(materialized["advantage_mask"][:, 1]))), 6
        )


if __name__ == "__main__":
    unittest.main()
