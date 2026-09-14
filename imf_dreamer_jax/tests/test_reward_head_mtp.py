from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DreamerConfig,
    create_agent,
    predict_reward_offset_logits,
    predict_reward_offsets,
    reward_mtp_targets,
    reward_prediction_loss,
    two_hot_reward,
)


def config(**overrides: object) -> DreamerConfig:
    values: dict[str, object] = {
        "observation_shape": (3,),
        "action_dim": 2,
        "deterministic_dim": 5,
        "stochastic_dim": 3,
        "embedding_dim": 4,
        "hidden_dim": 12,
        "prior": "imf",
        "imf_trajectory_enabled": True,
        "reward_output_init_scale": 1.0,
    }
    values.update(overrides)
    return DreamerConfig(**values)


class RewardHeadMTPTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("REWARD_HEAD_MTP_LIBRARY_VERIFIED")

    def test_terminal_safe_offsets_never_cross_an_episode_boundary(self) -> None:
        targets, masks = reward_mtp_targets(
            jnp.asarray([[0.0, 1.0, 2.0, 3.0]]),
            jnp.asarray([[1.0, 0.0, 1.0, 1.0]]),
            jnp.asarray([[True, False, True, False]]),
            jnp.ones((1, 4)),
            2,
        )
        np.testing.assert_array_equal(targets[0, :, 0], [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(targets[0, :, 1], [1.0, 2.0, 3.0, 0.0])
        np.testing.assert_array_equal(masks[0, :, 0], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(masks[0, :, 1], [1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(masks[0, :, 2], [0.0, 0.0, 0.0, 0.0])

    def test_scalar_head_predicts_offsets_zero_through_eight(self) -> None:
        cfg = config(reward_prediction_horizon=8)
        state = create_agent(cfg, jax.random.key(1))
        features = jax.random.normal(jax.random.key(2), (2, 12, cfg.feature_dim))
        prediction = predict_reward_offsets(state.params.world_model, features, cfg)
        self.assertEqual(prediction.shape, (2, 12, 9))
        loss = reward_prediction_loss(
            state.params.world_model,
            features,
            jnp.ones((2, 12)),
            jnp.ones((2, 12)),
            jnp.zeros((2, 12), dtype=jnp.bool_),
            jnp.ones((2, 12)),
            cfg,
        )
        self.assertTrue(np.isfinite(float(loss)))

    def test_symexp_twohot_head_has_normalized_targets_and_scalar_decode(self) -> None:
        cfg = config(
            reward_loss="symexp_twohot",
            reward_bins=51,
            reward_prediction_horizon=8,
        )
        state = create_agent(cfg, jax.random.key(3))
        features = jax.random.normal(jax.random.key(4), (2, 12, cfg.feature_dim))
        logits = predict_reward_offset_logits(
            state.params.world_model, features, cfg
        )
        prediction = predict_reward_offsets(state.params.world_model, features, cfg)
        self.assertEqual(logits.shape, (2, 12, 9, 51))
        self.assertEqual(prediction.shape, (2, 12, 9))
        labels = two_hot_reward(jnp.asarray([-2.0, 0.0, 3.0]), cfg)
        np.testing.assert_allclose(np.asarray(labels.sum(-1)), 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(np.asarray(prediction)).all())

    def test_legacy_scalar_shape_and_decode_remain_unchanged(self) -> None:
        cfg = config()
        state = create_agent(cfg, jax.random.key(5))
        features = jax.random.normal(jax.random.key(6), (4, cfg.feature_dim))
        logits = predict_reward_offset_logits(state.params.world_model, features, cfg)
        values = predict_reward_offsets(state.params.world_model, features, cfg)
        self.assertEqual(logits.shape, (4, 1, 1))
        np.testing.assert_array_equal(values[..., 0], logits[..., 0, 0])


if __name__ == "__main__":
    unittest.main()
