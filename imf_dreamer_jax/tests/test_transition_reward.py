from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DreamerConfig,
    TransitionRewardConfig,
    attach_transition_reward_head,
    create_agent,
    init_transition_reward_state,
    jit_train_transition_reward_step,
    predict_reward,
    predict_transition_reward,
    transition_reward_loss,
)


def config(**overrides: object) -> DreamerConfig:
    values: dict[str, object] = {
        "observation_shape": (3,),
        "action_dim": 2,
        "deterministic_dim": 5,
        "stochastic_dim": 3,
        "embedding_dim": 4,
        "hidden_dim": 16,
        "prior": "imf",
        "imf_trajectory_enabled": True,
        "reward_loss": "mse",
        "reward_prediction_horizon": 0,
    }
    values.update(overrides)
    return DreamerConfig(**values)


def assert_tree_equal(test: unittest.TestCase, left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    test.assertEqual(left_tree, right_tree)
    test.assertEqual(len(left_leaves), len(right_leaves))
    for left_value, right_value in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_value, right_value)


def batch(cfg: DreamerConfig) -> dict[str, jax.Array]:
    actions = jax.random.uniform(
        jax.random.key(3), (4, 7, cfg.action_dim), minval=-1.0, maxval=1.0
    )
    rewards = 0.8 * actions[..., 0] - 0.35 * actions[..., 1] + 0.2
    return {
        "observations": jax.random.normal(
            jax.random.key(4), (4, 7, *cfg.observation_shape)
        ),
        "actions": actions,
        "rewards": rewards,
        "is_first": jnp.zeros((4, 7), dtype=jnp.bool_).at[:, 0].set(True),
        "loss_mask": jnp.ones((4, 7)),
    }


class TransitionRewardTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("IMF_TRANSITION_REWARD_VERIFIED")

    def test_legacy_transition_reward_is_bitwise_unchanged(self) -> None:
        cfg = config(reward_output_init_scale=1.0)
        source = create_agent(cfg, jax.random.key(1))
        previous = jax.random.normal(jax.random.key(5), (6, cfg.feature_dim))
        following = jax.random.normal(jax.random.key(6), (6, cfg.feature_dim))
        actions = jax.random.normal(jax.random.key(7), (6, cfg.action_dim))
        expected = predict_reward(source.params.world_model, following, cfg)
        actual = predict_transition_reward(
            source.params.world_model, previous, actions, following, cfg
        )
        np.testing.assert_array_equal(actual, expected)

    def test_direct_head_is_a_replacement_and_preserves_actor_gradients(self) -> None:
        cfg = config()
        source = create_agent(cfg, jax.random.key(10))
        reward_state = init_transition_reward_state(cfg, jax.random.key(11))
        # Make every layer active so the derivative test does not rely on
        # random cancellation or on training convergence.
        layers = []
        for layer in reward_state.params["layers"]:
            layers.append(
                {
                    "weight": jnp.full_like(layer["weight"], 0.05),
                    "bias": jnp.full_like(layer["bias"], 0.01),
                }
            )
        reward_state = reward_state._replace(params={"layers": tuple(layers)})
        attached = attach_transition_reward_head(source, reward_state)
        previous = jnp.ones((2, cfg.feature_dim))
        following = 1.1 * previous
        actions = jnp.ones((2, cfg.action_dim))

        prediction = predict_transition_reward(
            attached.params.world_model, previous, actions, following, cfg
        )
        altered_reward = jax.tree_util.tree_map(
            lambda value: value + 100.0,
            attached.params.world_model["reward"],
        )
        altered = {
            **attached.params.world_model,
            "reward": altered_reward,
        }
        np.testing.assert_array_equal(
            prediction,
            predict_transition_reward(altered, previous, actions, following, cfg),
        )
        action_gradient = jax.grad(
            lambda value: jnp.sum(
                predict_transition_reward(
                    attached.params.world_model, previous, value, following, cfg
                )
            )
        )(actions)
        self.assertGreater(float(jnp.linalg.norm(action_gradient)), 0.0)

    def test_isolated_mse_training_reduces_loss_and_freezes_source(self) -> None:
        cfg = config()
        source = create_agent(cfg, jax.random.key(20))
        frozen = source.params.world_model
        reward_state = init_transition_reward_state(cfg, jax.random.key(21))
        data = batch(cfg)
        objective = TransitionRewardConfig(learning_rate=1e-2, grad_clip=1.0)
        loss_key = jax.random.key(22)
        initial = transition_reward_loss(
            reward_state.params, frozen, data, loss_key, cfg
        ).loss
        original_source = jax.tree_util.tree_map(lambda value: value.copy(), frozen)
        for _ in range(200):
            reward_state, metrics = jit_train_transition_reward_step(
                reward_state, frozen, data, loss_key, cfg, objective
            )
        final = transition_reward_loss(
            reward_state.params, frozen, data, loss_key, cfg
        ).loss
        self.assertTrue(np.isfinite(float(metrics.grad_norm)))
        self.assertLess(float(final), 0.05 * float(initial))
        assert_tree_equal(self, frozen, original_source)

        attached = attach_transition_reward_head(source, reward_state)
        for name, subtree in source.params.world_model.items():
            assert_tree_equal(self, subtree, attached.params.world_model[name])
        self.assertEqual(
            set(attached.params.world_model) - set(source.params.world_model),
            {"reward_transition"},
        )

    def test_rejects_non_imf_and_non_scalar_objectives(self) -> None:
        with self.assertRaisesRegex(ValueError, "trajectory iMF"):
            init_transition_reward_state(
                config(prior="gaussian", imf_trajectory_enabled=False),
                jax.random.key(30),
            )
        with self.assertRaisesRegex(ValueError, "scalar one-step MSE"):
            init_transition_reward_state(
                config(reward_prediction_horizon=1), jax.random.key(31)
            )


if __name__ == "__main__":
    unittest.main()
