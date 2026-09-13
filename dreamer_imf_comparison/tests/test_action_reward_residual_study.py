from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from dreamer_imf_compare import action_reward_residual_study as study
from dreamer_imf_compare import shared_probe_bank
from imf_dreamer_jax import (
    ActionRewardResidualConfig,
    DreamerConfig,
    RSSMState,
    attach_action_reward_residual,
    create_agent,
    init_running_rms,
    jit_train_action_reward_residual,
)
from imf_dreamer_jax.nn import tree_global_norm


_VERIFIER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_action_reward_residual_study.py"
)
_VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_action_reward_residual_study", _VERIFIER_PATH
)
assert _VERIFIER_SPEC is not None and _VERIFIER_SPEC.loader is not None
_VERIFIER = importlib.util.module_from_spec(_VERIFIER_SPEC)
_VERIFIER_SPEC.loader.exec_module(_VERIFIER)


def config() -> DreamerConfig:
    return DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=5,
        stochastic_dim=3,
        embedding_dim=4,
        hidden_dim=12,
        prior="imf",
        imf_trajectory_enabled=True,
        imf_sampling_steps=1,
        reward_loss="symexp_twohot",
        reward_prediction_horizon=8,
        reward_bins=51,
        reward_output_init_scale=1.0,
    )


def tree_delta(left: object, right: object) -> float:
    return float(
        tree_global_norm(jax.tree_util.tree_map(lambda x, y: y - x, left, right))
    )


def decision_batch(cfg: DreamerConfig) -> dict[str, jax.Array]:
    batch, steps, candidates, horizon = 2, 4, 3, 15
    return {
        "observations": jax.random.normal(
            jax.random.key(1), (batch, steps, *cfg.observation_shape)
        ),
        "actions": jax.random.uniform(
            jax.random.key(2), (batch, steps, cfg.action_dim), minval=-1.0, maxval=1.0
        ),
        "is_first": jnp.zeros((batch, steps), dtype=jnp.bool_).at[:, 0].set(True),
        "advantage_action_sequences": jax.random.uniform(
            jax.random.key(3),
            (batch, steps, candidates, horizon, cfg.action_dim),
            minval=-1.0,
            maxval=1.0,
        ),
        "advantage_target_returns": jax.random.normal(
            jax.random.key(4),
            (batch, steps, candidates, len(study.DENSE_TRAINING_HORIZONS)),
        ),
        "advantage_mask": jnp.ones(
            (batch, steps, len(study.DENSE_TRAINING_HORIZONS))
        ),
    }


class ActionRewardResidualStudyTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ACTION_REWARD_RESIDUAL_STUDY_VERIFIED")

    def test_frozen_design_has_only_one_trainable_subtree_and_two_seeds(self) -> None:
        self.assertEqual(
            study.SCHEMA, "trajectory-imf-action-reward-residual-study-v3"
        )
        self.assertEqual(
            study.SPARSE_SCHEMA, "trajectory-imf-action-reward-residual-study-v2"
        )
        self.assertEqual(
            study.CENTERED_SCHEMA, "trajectory-imf-action-reward-residual-study-v1"
        )
        self.assertEqual(study.WORLD_MODEL_SEEDS, (211, 223))
        self.assertEqual(study.HORIZONS, (5, 15))
        objective = ActionRewardResidualConfig(
            horizons=study.DENSE_TRAINING_HORIZONS
        )
        self.assertEqual(objective.horizons, tuple(range(1, 16)))
        self.assertEqual(objective.huber_delta, 1.0)
        self.assertFalse(hasattr(objective, "output_l2_scale"))

    def test_objective_round_trips_through_json_sequence_types(self) -> None:
        objective = ActionRewardResidualConfig(horizons=list(range(1, 16)))
        self.assertEqual(objective.horizons, tuple(range(1, 16)))

    def test_preflight_verifier_accepts_recorded_gpu_runtime_schema(self) -> None:
        manifest = {"source_commit": "abc", "manifest_sha256": "def"}
        preflight = {
            "status": "complete",
            "source_commit": "abc",
            "manifest_sha256": "def",
            "full_library_suite_passed": True,
            "full_comparison_suite_passed": True,
            "runtime": {"backend": "gpu"},
        }
        self.assertTrue(_VERIFIER._preflight_evidence_complete(preflight, manifest))
        preflight["runtime"]["backend"] = "cpu"
        self.assertFalse(_VERIFIER._preflight_evidence_complete(preflight, manifest))

    def test_shared_probe_evaluator_uses_transition_residual(self) -> None:
        cfg = config()
        source = create_agent(cfg, jax.random.key(20))
        attached = attach_action_reward_residual(source, cfg, jax.random.key(21))
        residual = attached.params.world_model["reward_action_residual"]
        shifted_residual = {
            "layers": (
                residual["layers"][0],
                {
                    **residual["layers"][1],
                    "bias": jnp.ones_like(residual["layers"][1]["bias"]),
                },
            )
        }
        shifted_world = {
            **attached.params.world_model,
            "reward_action_residual": shifted_residual,
        }
        initial = RSSMState(
            jnp.zeros((1, cfg.deterministic_dim)),
            jnp.zeros((1, cfg.stochastic_dim)),
        )
        actions = jnp.zeros((1, 2, 3, cfg.action_dim))
        noise = jnp.zeros((1, 1, 3, cfg.stochastic_dim))
        baseline = shared_probe_bank._model_probe_returns(
            source.params.world_model, initial, actions, noise, cfg, (1, 3)
        )
        zero_residual = shared_probe_bank._model_probe_returns(
            attached.params.world_model, initial, actions, noise, cfg, (1, 3)
        )
        shifted = shared_probe_bank._model_probe_returns(
            shifted_world, initial, actions, noise, cfg, (1, 3)
        )
        np.testing.assert_array_equal(np.asarray(zero_residual), np.asarray(baseline))
        self.assertGreater(float(jnp.max(jnp.abs(shifted - baseline))), 0.0)

    def test_jitted_study_update_preserves_base_reward_and_dynamics(self) -> None:
        cfg = config()
        source = create_agent(cfg, jax.random.key(10))
        state = attach_action_reward_residual(source, cfg, jax.random.key(11))
        before = state
        updated, details, rms = jit_train_action_reward_residual(
            state,
            decision_batch(cfg),
            jax.random.key(12),
            cfg,
            ActionRewardResidualConfig(horizons=study.DENSE_TRAINING_HORIZONS),
            init_running_rms(),
        )
        self.assertTrue(np.isfinite(float(details.total)))
        self.assertGreater(float(rms.count), 0.0)
        self.assertGreater(
            tree_delta(
                before.params.world_model["reward_action_residual"],
                updated.params.world_model["reward_action_residual"],
            ),
            0.0,
        )
        for name in source.params.world_model:
            self.assertEqual(
                tree_delta(source.params.world_model[name], updated.params.world_model[name]),
                0.0,
            )
            self.assertEqual(
                tree_delta(
                    source.model_optimizer.first_moment[name],
                    updated.model_optimizer.first_moment[name],
                ),
                0.0,
            )
            self.assertEqual(
                tree_delta(
                    source.model_optimizer.second_moment[name],
                    updated.model_optimizer.second_moment[name],
                ),
                0.0,
            )

    def test_dense_probe_contract_preserves_design_and_legacy_labels(self) -> None:
        horizons = np.asarray((1, 3, 5), np.int32)
        dense_horizons = np.arange(1, 16, dtype=np.int32)
        dense_returns = np.arange(45, dtype=np.float32).reshape((1, 3, 15))

        def bank(selected: np.ndarray, returns: np.ndarray) -> dict[str, np.ndarray]:
            steps = int(selected[-1])
            return {
                "episode_ids": np.asarray([0], np.int32),
                "anchors": np.asarray([2], np.int32),
                "horizons": selected,
                "action_sequences": np.zeros((1, 3, steps, 2), np.float32),
                "model_noise": np.zeros((2, 1, steps, 3), np.float32),
                "simulator_returns": returns,
                "horizon_mask": np.ones((1, len(selected)), np.float32),
                "maximum_replay_error": np.asarray([0.0], np.float64),
                "candidate_pool_size": np.asarray([8], np.int32),
                "selection_horizon": np.asarray([5], np.int32),
                "selection_return_ranges": np.asarray([30.0], np.float32),
                "action_delta": np.asarray([0.5], np.float32),
                "intervention_steps": np.asarray([5], np.int32),
            }

        legacy_indices = [0, 2, 4]
        source = bank(horizons, dense_returns[:, :, legacy_indices])
        dense = bank(dense_horizons, dense_returns.copy())
        self.assertEqual(study.dense_probe_legacy_max_error(source, dense), 0.0)
        dense["simulator_returns"] = dense["simulator_returns"].copy()
        dense["simulator_returns"][0, 0, 4] += 0.25
        self.assertAlmostEqual(
            study.dense_probe_legacy_max_error(source, dense), 0.25
        )
        dense["action_sequences"] = dense["action_sequences"].copy()
        dense["action_sequences"][0, 0, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "fixed design field"):
            study.dense_probe_legacy_max_error(source, dense)

    def test_all_prefix_return_operator_is_full_rank(self) -> None:
        discount = 0.99
        operator = np.tril(
            np.power(discount, np.arange(15, dtype=np.float64))[None, :]
            * np.ones((15, 1), np.float64)
        )
        self.assertEqual(np.linalg.matrix_rank(operator), 15)

    def test_dense_replay_extends_only_the_common_behavior_suffix(self) -> None:
        time = 20
        actions = np.arange(time, dtype=np.float32).reshape((1, time, 1)) / 20.0
        arrays = {
            "observations": np.arange(time, dtype=np.float32).reshape((1, time, 1)),
            "actions": actions,
            "rewards": (actions[..., 0] + 1.0).astype(np.float32),
            "continuations": np.ones((1, time), np.float32),
        }
        source_actions = np.repeat(actions[:, 2:7, :][:, None], 3, axis=1)
        source_actions[0, 1, :, 0] -= 0.25
        source_actions[0, 2, :, 0] += 0.25

        class FakeAdapter:
            def __init__(self, *_args, **_kwargs) -> None:
                self.step_index = 0

            def reset(self) -> np.ndarray:
                self.step_index = 0
                return np.asarray([0.0], np.float32)

            def snapshot(self) -> int:
                return self.step_index

            def restore(self, snapshot: int) -> None:
                self.step_index = snapshot

            def step(self, action: np.ndarray) -> SimpleNamespace:
                self.step_index += 1
                return SimpleNamespace(
                    observation=np.asarray([self.step_index], np.float32),
                    reward=float(action[0] + 1.0),
                    continuation=1.0,
                    is_last=False,
                )

            def close(self) -> None:
                pass

        source = {
            "episode_ids": np.asarray([0], np.int32),
            "anchors": np.asarray([2], np.int32),
            "horizons": np.asarray([1, 3, 5], np.int32),
            "action_sequences": source_actions,
            "model_noise": np.zeros((2, 1, 5, 3), np.float32),
            "simulator_returns": np.zeros((1, 3, 3), np.float32),
            "horizon_mask": np.ones((1, 3), np.float32),
            "maximum_replay_error": np.asarray([0.0], np.float64),
            "candidate_pool_size": np.asarray([8], np.int32),
            "selection_horizon": np.asarray([5], np.int32),
            "selection_return_ranges": np.asarray([1.0], np.float32),
            "action_delta": np.asarray([0.5], np.float32),
            "intervention_steps": np.asarray([5], np.int32),
        }
        discount = 0.99
        legacy_indices = (0, 2, 4)
        for candidate in range(3):
            cumulative = 0.0
            for offset in range(5):
                cumulative += discount**offset * (
                    float(source_actions[0, candidate, offset, 0]) + 1.0
                )
                if offset in legacy_indices:
                    source["simulator_returns"][
                        0, candidate, legacy_indices.index(offset)
                    ] = cumulative
        with mock.patch.object(shared_probe_bank, "DMCAdapter", FakeAdapter):
            dense = shared_probe_bank.relabel_shared_probe_bank_horizons(
                arrays,
                source,
                task="fake_task",
                world_model_seed=211,
                action_repeat=1,
                horizons=tuple(range(1, 16)),
                discount=discount,
            )
        np.testing.assert_array_equal(
            dense["action_sequences"][:, :, :5], source_actions
        )
        expected_tail = np.repeat(actions[:, None, 7:17], 3, axis=1)
        np.testing.assert_array_equal(
            dense["action_sequences"][:, :, 5:], expected_tail
        )
        self.assertLessEqual(study.dense_probe_legacy_max_error(source, dense), 1e-6)


if __name__ == "__main__":
    unittest.main()
