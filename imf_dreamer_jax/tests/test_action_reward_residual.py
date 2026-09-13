from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    ActionRewardResidualConfig,
    DreamerConfig,
    attach_action_reward_residual,
    create_agent,
    imagine,
    init_running_rms,
    predict_action_reward_residual,
    predict_reward,
    predict_transition_reward,
    train_action_reward_residual,
)
from imf_dreamer_jax.nn import tree_global_norm
from imf_dreamer_jax.types import RSSMState


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
        "imf_sampling_steps": 1,
        "imagination_horizon": 2,
        "reward_output_init_scale": 1.0,
        "model_learning_rate": 1e-2,
    }
    values.update(overrides)
    return DreamerConfig(**values)


def tree_delta(left: object, right: object) -> float:
    return float(
        tree_global_norm(jax.tree_util.tree_map(lambda x, y: y - x, left, right))
    )


def decision_batch(cfg: DreamerConfig) -> dict[str, jax.Array]:
    batch, steps, candidates, horizon = 2, 3, 2, 5
    candidate_actions = jnp.zeros(
        (batch, steps, candidates, horizon, cfg.action_dim)
    )
    candidate_actions = candidate_actions.at[:, :, 0, :, 0].set(-0.75)
    candidate_actions = candidate_actions.at[:, :, 1, :, 0].set(0.75)
    targets = jnp.zeros((batch, steps, candidates, 3))
    targets = targets.at[:, :, 0, :].set(-1.0)
    targets = targets.at[:, :, 1, :].set(1.0)
    return {
        "observations": jax.random.normal(
            jax.random.key(10), (batch, steps, *cfg.observation_shape)
        ),
        "actions": jnp.zeros((batch, steps, cfg.action_dim)),
        "is_first": jnp.zeros((batch, steps), dtype=jnp.bool_).at[:, 0].set(True),
        "advantage_action_sequences": candidate_actions,
        "advantage_target_returns": targets,
        "advantage_mask": jnp.ones((batch, steps, 3)),
    }


class ActionRewardResidualTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ACTION_REWARD_RESIDUAL_LIBRARY_VERIFIED")

    def test_zero_initialized_residual_is_an_exact_identity(self) -> None:
        cfg = config()
        source = create_agent(cfg, jax.random.key(1))
        attached = attach_action_reward_residual(source, cfg, jax.random.key(2))
        previous = jax.random.normal(jax.random.key(3), (7, cfg.feature_dim))
        following = jax.random.normal(jax.random.key(4), (7, cfg.feature_dim))
        actions = jax.random.normal(jax.random.key(5), (7, cfg.action_dim))
        expected = predict_reward(source.params.world_model, following, cfg)
        actual = predict_transition_reward(
            attached.params.world_model, previous, actions, following, cfg
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            predict_action_reward_residual(
                attached.params.world_model, previous, actions, following, cfg
            ),
            jnp.zeros((7,)),
        )
        self.assertEqual(
            set(attached.model_optimizer.first_moment)
            - set(source.model_optimizer.first_moment),
            {"reward_action_residual"},
        )

    def test_residual_inputs_are_gradient_isolated(self) -> None:
        cfg = config()
        state = attach_action_reward_residual(
            create_agent(cfg, jax.random.key(20)), cfg, jax.random.key(21)
        )
        residual = state.params.world_model["reward_action_residual"]
        layers = list(residual["layers"])
        output = dict(layers[-1])
        output["weight"] = jnp.ones_like(output["weight"])
        layers[-1] = output
        model = {
            **state.params.world_model,
            "reward_action_residual": {"layers": tuple(layers)},
        }
        previous = jnp.ones((1, cfg.feature_dim))
        action = jnp.ones((1, cfg.action_dim))
        following = 2.0 * jnp.ones((1, cfg.feature_dim))

        def objective(p, a, n):
            return jnp.sum(predict_action_reward_residual(model, p, a, n, cfg))

        gradients = jax.grad(objective, argnums=(0, 1, 2))(
            previous, action, following
        )
        self.assertEqual(float(tree_global_norm(gradients)), 0.0)

    def test_training_changes_only_the_residual_and_its_moments(self) -> None:
        cfg = config()
        source = create_agent(cfg, jax.random.key(30))
        state = attach_action_reward_residual(source, cfg, jax.random.key(31))
        initial = state
        rms = init_running_rms()
        data = decision_batch(cfg)
        objective = ActionRewardResidualConfig(output_l2_scale=1e-4)
        details = None
        for update in range(3):
            state, details, rms = train_action_reward_residual(
                state,
                data,
                jax.random.fold_in(jax.random.key(32), update),
                cfg,
                objective,
                rms,
            )
        self.assertIsNotNone(details)
        self.assertTrue(np.isfinite(float(details.total)))
        self.assertGreater(
            tree_delta(
                initial.params.world_model["reward_action_residual"],
                state.params.world_model["reward_action_residual"],
            ),
            0.0,
        )
        for name in source.params.world_model:
            self.assertEqual(
                tree_delta(source.params.world_model[name], state.params.world_model[name]),
                0.0,
            )
            self.assertEqual(
                tree_delta(
                    source.model_optimizer.first_moment[name],
                    state.model_optimizer.first_moment[name],
                ),
                0.0,
            )
            self.assertEqual(
                tree_delta(
                    source.model_optimizer.second_moment[name],
                    state.model_optimizer.second_moment[name],
                ),
                0.0,
            )

    def test_imagination_uses_the_residual_without_mutating_dynamics(self) -> None:
        cfg = config()
        source = create_agent(cfg, jax.random.key(40))
        state = attach_action_reward_residual(source, cfg, jax.random.key(41))
        residual = state.params.world_model["reward_action_residual"]
        layers = list(residual["layers"])
        output = dict(layers[-1])
        output["bias"] = jnp.ones_like(output["bias"])
        layers[-1] = output
        shifted_world = {
            **state.params.world_model,
            "reward_action_residual": {"layers": tuple(layers)},
        }
        start = RSSMState(
            jnp.zeros((2, cfg.deterministic_dim)),
            jnp.zeros((2, cfg.stochastic_dim)),
        )
        base = imagine(source.params, start, jax.random.key(42), cfg)
        shifted = imagine(
            source.params._replace(world_model=shifted_world),
            start,
            jax.random.key(42),
            cfg,
        )
        np.testing.assert_allclose(shifted.rewards, base.rewards + 1.0, atol=1e-6)
        np.testing.assert_array_equal(shifted.features, base.features)


if __name__ == "__main__":
    unittest.main()
