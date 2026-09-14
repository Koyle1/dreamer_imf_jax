from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DreamerConfig,
    FlowMPCConfig,
    ReBRACConfig,
    ReBRACDataset,
    RSSMState,
    flowmpc_adapt_actor,
    flowmpc_objective,
    init_rebrac_state,
    init_transition_reward_head,
    init_world_model,
    rebrac_actor,
    rebrac_actor_loss,
    rebrac_critic_loss,
    rebrac_critics,
    rebrac_update,
    train_rebrac_chunk,
)


def dreamer_config() -> DreamerConfig:
    return DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=5,
        stochastic_dim=3,
        embedding_dim=4,
        hidden_dim=16,
        prior="imf",
        imf_trajectory_enabled=True,
        reward_loss="mse",
        reward_prediction_horizon=0,
    )


def rebrac_config() -> ReBRACConfig:
    return ReBRACConfig(state_dim=3, action_dim=2)


def rebrac_batch(cfg: ReBRACConfig) -> ReBRACDataset:
    states = jax.random.normal(jax.random.key(1), (cfg.batch_size, cfg.state_dim))
    actions = jnp.tanh(states[:, : cfg.action_dim])
    next_states = states + 0.05
    next_actions = jnp.tanh(next_states[:, : cfg.action_dim])
    rewards = 0.4 * states[:, 0] - 0.2 * actions[:, 1]
    dones = jnp.zeros((cfg.batch_size,), dtype=jnp.float32)
    return ReBRACDataset(states, actions, rewards, next_states, next_actions, dones)


def assert_tree_equal(test: unittest.TestCase, left: object, right: object) -> None:
    left_values, left_structure = jax.tree_util.tree_flatten(left)
    right_values, right_structure = jax.tree_util.tree_flatten(right)
    test.assertEqual(left_structure, right_structure)
    for left_value, right_value in zip(left_values, right_values, strict=True):
        np.testing.assert_array_equal(left_value, right_value)


def tree_distance(left: object, right: object) -> float:
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: x - y, left, right)
    )
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves)))


class ReBRACTests(unittest.TestCase):
    def test_reference_architecture_and_action_range(self) -> None:
        cfg = rebrac_config()
        state = init_rebrac_state(jax.random.key(2), cfg)
        self.assertEqual(len(state.actor["hidden"]), 3)
        self.assertEqual(len(state.critics["members"]), 2)
        self.assertNotIn("norm_scale", state.actor["output"])
        for layer in state.actor["hidden"]:
            self.assertNotIn("norm_scale", layer)
            self.assertNotIn("norm_bias", layer)
        for member in state.critics["members"]:
            for layer in member["hidden"]:
                self.assertIn("norm_scale", layer)
                self.assertIn("norm_bias", layer)
        actions = rebrac_actor(state.actor, jnp.ones((7, cfg.state_dim)))
        self.assertEqual(actions.shape, (7, cfg.action_dim))
        self.assertTrue(bool(jnp.all(jnp.abs(actions) <= 1.0)))
        q_values = rebrac_critics(
            state.critics, jnp.ones((7, cfg.state_dim)), actions
        )
        self.assertEqual(q_values.shape, (2, 7))

    def test_losses_are_the_released_rebrac_equations(self) -> None:
        cfg = rebrac_config()
        state = init_rebrac_state(jax.random.key(3), cfg)
        batch = rebrac_batch(cfg)
        actor_loss, (mean_q, mean_bc, q_scale) = rebrac_actor_loss(
            state.actor, state.critics, batch, cfg
        )
        actions = rebrac_actor(state.actor, batch.states)
        q = jnp.min(rebrac_critics(state.critics, batch.states, actions), axis=0)
        bc = jnp.sum(jnp.square(actions - batch.actions), axis=-1)
        expected_scale = 1.0 / jnp.mean(jnp.abs(q))
        np.testing.assert_allclose(q_scale, expected_scale, rtol=1e-6)
        np.testing.assert_allclose(mean_q, jnp.mean(q), rtol=1e-6)
        np.testing.assert_allclose(mean_bc, jnp.mean(bc), rtol=1e-6)
        np.testing.assert_allclose(
            actor_loss,
            jnp.mean(cfg.actor_bc_coefficient * bc - expected_scale * q),
            rtol=1e-6,
        )

        noise = jax.random.normal(jax.random.key(4), batch.actions.shape)
        critic_loss, (_, target_q, mean_critic_bc) = rebrac_critic_loss(
            state.critics,
            state.target_critics,
            state.target_actor,
            batch,
            noise,
            cfg,
        )
        next_action = rebrac_actor(state.target_actor, batch.next_states)
        next_action = jnp.clip(
            next_action
            + jnp.clip(
                noise * cfg.policy_noise, -cfg.noise_clip, cfg.noise_clip
            ),
            -1.0,
            1.0,
        )
        critic_bc = jnp.sum(jnp.square(next_action - batch.next_actions), axis=-1)
        next_q = jnp.min(
            rebrac_critics(state.target_critics, batch.next_states, next_action),
            axis=0,
        )
        expected_target = batch.rewards + cfg.discount * (
            next_q - cfg.critic_bc_coefficient * critic_bc
        )
        prediction = rebrac_critics(state.critics, batch.states, batch.actions)
        expected_loss = jnp.sum(
            jnp.mean(jnp.square(prediction - expected_target[None]), axis=1)
        )
        np.testing.assert_allclose(target_q, jnp.mean(expected_target), rtol=1e-6)
        np.testing.assert_allclose(mean_critic_bc, jnp.mean(critic_bc), rtol=1e-6)
        np.testing.assert_allclose(critic_loss, expected_loss, rtol=1e-6)

    def test_delayed_update_and_target_timing_match_reference(self) -> None:
        cfg = rebrac_config()
        initial = init_rebrac_state(jax.random.key(5), cfg)
        batch = rebrac_batch(cfg)
        first, first_metrics = rebrac_update(
            initial, batch, jax.random.key(6), cfg
        )
        self.assertEqual(float(first_metrics.actor_updated), 1.0)
        self.assertGreater(tree_distance(first.actor, initial.actor), 0.0)
        self.assertGreater(tree_distance(first.critics, initial.critics), 0.0)
        # Released functional code Polyak-updates the actor target from the
        # pre-update actor, so it remains bitwise equal on the first step.
        assert_tree_equal(self, first.target_actor, initial.target_actor)
        self.assertGreater(
            tree_distance(first.target_critics, initial.target_critics), 0.0
        )

        second, second_metrics = rebrac_update(
            first, batch, jax.random.key(7), cfg
        )
        self.assertEqual(float(second_metrics.actor_updated), 0.0)
        assert_tree_equal(self, second.actor, first.actor)
        assert_tree_equal(self, second.target_actor, first.target_actor)
        assert_tree_equal(self, second.target_critics, first.target_critics)

    def test_training_chunk_is_exactly_resumable(self) -> None:
        cfg = rebrac_config()
        initial = init_rebrac_state(jax.random.key(8), cfg)
        batch = rebrac_batch(cfg)
        full, _ = train_rebrac_chunk(
            initial, batch, jax.random.key(9), updates=2, config=cfg
        )
        split, _ = train_rebrac_chunk(
            initial, batch, jax.random.key(9), updates=1, config=cfg
        )
        split, _ = train_rebrac_chunk(
            split, batch, jax.random.key(9), updates=1, config=cfg
        )
        assert_tree_equal(self, full, split)


class FlowMPCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dreamer = dreamer_config()
        self.rebrac = rebrac_config()
        world = init_world_model(self.dreamer, jax.random.key(10))
        reward = init_transition_reward_head(
            self.dreamer, jax.random.key(11), hidden_dim=16
        )
        self.world = {**world, "reward_transition": reward}
        self.policy = init_rebrac_state(jax.random.key(12), self.rebrac)
        self.initial = RSSMState(
            jnp.zeros((1, self.dreamer.deterministic_dim)),
            jnp.zeros((1, self.dreamer.stochastic_dim)),
        )
        self.observation = jnp.zeros((1, *self.dreamer.observation_shape))
        self.controller = FlowMPCConfig(
            horizon=2, particles=8, inner_steps=1, step_size=1e-7
        )
        self.noises = jax.random.normal(
            jax.random.key(13),
            (
                self.controller.particles,
                self.controller.horizon,
                self.dreamer.stochastic_dim,
            ),
        )

    def test_objective_is_discounted_stage_reward_plus_terminal_q(self) -> None:
        details = flowmpc_objective(
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial,
            self.observation,
            self.noises,
            self.dreamer,
            self.rebrac,
            self.controller,
        )
        discounts = self.controller.discount ** jnp.arange(
            self.controller.horizon
        )
        expected_stage = jnp.mean(jnp.sum(details.rewards * discounts[None], axis=1))
        expected_terminal = self.controller.discount**self.controller.horizon * jnp.mean(
            details.terminal_q
        )
        np.testing.assert_allclose(details.stage_return, expected_stage, rtol=1e-6)
        np.testing.assert_allclose(
            details.terminal_value, expected_terminal, rtol=1e-6
        )
        np.testing.assert_allclose(
            details.objective, expected_stage + expected_terminal, rtol=1e-6
        )

    def test_fixed_noise_pathwise_update_changes_only_actor(self) -> None:
        frozen_world = jax.tree_util.tree_map(lambda value: value.copy(), self.world)
        frozen_critics = jax.tree_util.tree_map(
            lambda value: value.copy(), self.policy.critics
        )
        first = flowmpc_adapt_actor(
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial,
            self.observation,
            self.noises,
            self.dreamer,
            self.rebrac,
            self.controller,
        )
        second = flowmpc_adapt_actor(
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial,
            self.observation,
            self.noises,
            self.dreamer,
            self.rebrac,
            self.controller,
        )
        self.assertGreater(float(first.gradient_norm), 0.0)
        self.assertGreater(float(first.parameter_delta), 0.0)
        self.assertGreaterEqual(
            float(first.objective_after), float(first.objective_before) - 1e-7
        )
        assert_tree_equal(self, first, second)
        assert_tree_equal(self, self.world, frozen_world)
        assert_tree_equal(self, self.policy.critics, frozen_critics)

    def test_terminal_critic_contributes_to_policy_gradient(self) -> None:
        full = jax.grad(
            lambda actor: flowmpc_objective(
                actor,
                self.policy.critics,
                self.world,
                self.initial,
                self.observation,
                self.noises,
                self.dreamer,
                self.rebrac,
                self.controller,
            ).objective
        )(self.policy.actor)
        zero_critics = jax.tree_util.tree_map(
            jnp.zeros_like, self.policy.critics
        )
        without_terminal = jax.grad(
            lambda actor: flowmpc_objective(
                actor,
                zero_critics,
                self.world,
                self.initial,
                self.observation,
                self.noises,
                self.dreamer,
                self.rebrac,
                self.controller,
            ).objective
        )(self.policy.actor)
        self.assertGreater(tree_distance(full, without_terminal), 0.0)

    def test_zero_horizon_matches_papers_terminal_q_ablation(self) -> None:
        controller = FlowMPCConfig(
            horizon=0, particles=8, inner_steps=1, step_size=1e-7
        )
        details = flowmpc_objective(
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial,
            self.observation,
            jnp.empty((8, 0, self.dreamer.stochastic_dim)),
            self.dreamer,
            self.rebrac,
            controller,
        )
        self.assertEqual(details.rewards.shape, (8, 0))
        np.testing.assert_allclose(details.stage_return, 0.0, atol=0.0)
        np.testing.assert_allclose(
            details.objective, details.terminal_value, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
