from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dreamer_imf_compare import action_reward_residual_study as study
from imf_dreamer_jax import (
    ActionRewardResidualConfig,
    DreamerConfig,
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
    batch, steps, candidates, horizon = 2, 4, 3, 5
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
            jax.random.key(4), (batch, steps, candidates, 3)
        ),
        "advantage_mask": jnp.ones((batch, steps, 3)),
    }


class ActionRewardResidualStudyTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ACTION_REWARD_RESIDUAL_STUDY_VERIFIED")

    def test_frozen_design_has_only_one_trainable_subtree_and_two_seeds(self) -> None:
        self.assertEqual(study.WORLD_MODEL_SEEDS, (211, 223))
        self.assertEqual(study.HORIZONS, (5, 15))
        objective = ActionRewardResidualConfig()
        aligned = objective.advantage_objective()
        self.assertEqual(aligned.advantage_magnitude_scale, 1.0)
        self.assertEqual(aligned.advantage_ranking_scale, 0.0)
        self.assertEqual(aligned.advantage_flat_scale, 0.0)

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
            ActionRewardResidualConfig(),
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


if __name__ == "__main__":
    unittest.main()
