from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.agent import (
    actor_distribution,
    behavior_cloning_loss,
    create_agent,
    critic,
    critic_logits,
    diagonal_normal_kl,
    diverse_imagination_starts,
    snapshot_behavior_prior,
    symexp,
    symlog,
    train_actor_critic,
    train_behavior_cloning,
    two_hot_symlog,
)
from imf_dreamer_jax.config import DreamerConfig
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
        "prior": "gaussian",
        "imagination_horizon": 4,
        "reward_output_init_scale": 1.0,
    }
    values.update(overrides)
    return DreamerConfig(**values)


def tree_delta(left: object, right: object) -> float:
    return float(
        tree_global_norm(
            jax.tree_util.tree_map(lambda x, y: y - x, left, right)
        )
    )


class Dreamer4ActorAblationTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("DREAMER4_ACTOR_ABLATION_TESTS_OK")

    def test_symlog_round_trip_and_two_hot_interpolation(self) -> None:
        cfg = config(critic_bins=51)
        values = jnp.asarray([-100.0, -1.25, 0.0, 2.5, 100.0])
        np.testing.assert_allclose(symexp(symlog(values)), values, rtol=2e-6)
        encoded = two_hot_symlog(values, cfg)
        np.testing.assert_allclose(jnp.sum(encoded, axis=-1), 1.0, atol=1e-6)
        self.assertTrue(bool(jnp.all(encoded >= 0.0)))
        self.assertTrue(bool(jnp.all(jnp.sum(encoded > 0.0, axis=-1) <= 2)))

    def test_distributional_critic_decodes_symmetric_zero_initialization(self) -> None:
        cfg = config(critic_bins=51, critic_output_init_scale=0.0)
        state = create_agent(cfg, jax.random.key(1))
        features = jax.random.normal(jax.random.key(2), (7, cfg.feature_dim))
        logits = critic_logits(state.params.critic, features, cfg)
        np.testing.assert_array_equal(logits, np.zeros((7, 51), dtype=np.float32))
        np.testing.assert_allclose(
            critic(state.params.critic, features, cfg), 0.0, atol=1e-5
        )

    def test_diverse_starts_choose_exactly_one_reproducible_state_per_sequence(self) -> None:
        deterministic = jnp.arange(4 * 6 * 2, dtype=jnp.float32).reshape(4, 6, 2)
        stochastic = -jnp.arange(4 * 6, dtype=jnp.float32).reshape(4, 6, 1)
        states = RSSMState(deterministic, stochastic)
        first = diverse_imagination_starts(states, 2, jax.random.key(3))
        second = diverse_imagination_starts(states, 2, jax.random.key(3))
        self.assertEqual(first.deterministic.shape, (4, 2))
        self.assertEqual(first.stochastic.shape, (4, 1))
        np.testing.assert_array_equal(first.deterministic, second.deterministic)
        np.testing.assert_array_equal(first.stochastic, second.stochastic)
        for row in range(4):
            candidates = np.asarray(deterministic[row, 2:])
            self.assertTrue(
                any(np.array_equal(first.deterministic[row], item) for item in candidates)
            )

    def test_behavior_cloning_reduces_replay_action_nll(self) -> None:
        cfg = config(behavior_cloning_learning_rate=3e-3)
        state = create_agent(cfg, jax.random.key(4))
        features = jax.random.normal(jax.random.key(5), (64, cfg.feature_dim))
        actions = jnp.full((64, cfg.action_dim), 0.55)
        before = float(behavior_cloning_loss(state.params.actor, features, actions, cfg))
        for _ in range(25):
            state, _ = train_behavior_cloning(state, features, actions, cfg)
        after = float(behavior_cloning_loss(state.params.actor, features, actions, cfg))
        self.assertLess(after, before - 0.05)
        self.assertEqual(int(state.model_optimizer.step), 0)
        self.assertEqual(int(state.critic_optimizer.step), 0)

    def test_behavior_prior_snapshot_is_immutable_across_pmpo_update(self) -> None:
        cfg = config(
            actor_gradient="pmpo",
            behavior_kl_scale=0.2,
            critic_bins=51,
            actor_entropy_scale=0.0,
        )
        state = create_agent(cfg, jax.random.key(6))
        prior = snapshot_behavior_prior(state.params.actor)
        prior_copy = snapshot_behavior_prior(prior)
        world_before = state.params.world_model
        model_step_before = int(state.model_optimizer.step)
        start = RSSMState(
            jax.random.normal(jax.random.key(7), (16, cfg.deterministic_dim)),
            jax.random.normal(jax.random.key(8), (16, cfg.stochastic_dim)),
        )
        updated, metrics = train_actor_critic(
            state,
            start,
            jax.random.key(9),
            cfg,
            behavior_prior=prior,
        )
        self.assertTrue(np.isfinite(np.asarray(tuple(metrics))).all())
        self.assertGreater(tree_delta(state.params.actor, updated.params.actor), 0.0)
        self.assertEqual(tree_delta(world_before, updated.params.world_model), 0.0)
        self.assertEqual(int(updated.model_optimizer.step), model_step_before)
        self.assertEqual(tree_delta(prior_copy, prior), 0.0)

    def test_pmpo_kl_requires_explicit_prior(self) -> None:
        cfg = config(actor_gradient="pmpo", behavior_kl_scale=0.1)
        state = create_agent(cfg, jax.random.key(10))
        start = RSSMState(
            jnp.zeros((4, cfg.deterministic_dim)),
            jnp.zeros((4, cfg.stochastic_dim)),
        )
        with self.assertRaisesRegex(ValueError, "behavior_prior is required"):
            train_actor_critic(state, start, jax.random.key(11), cfg)

    def test_normal_kl_is_zero_only_for_matching_policy(self) -> None:
        cfg = config()
        state = create_agent(cfg, jax.random.key(12))
        features = jax.random.normal(jax.random.key(13), (5, cfg.feature_dim))
        distribution = actor_distribution(state.params.actor, features, cfg)
        np.testing.assert_allclose(
            diagonal_normal_kl(distribution, distribution), 0.0, atol=1e-6
        )
        shifted = distribution._replace(mean=distribution.mean + 0.5)
        self.assertTrue(bool(jnp.all(diagonal_normal_kl(shifted, distribution) > 0.0)))

    def test_invalid_ablation_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "critic_bins"):
            config(critic_bins=10)
        with self.assertRaisesRegex(ValueError, "imagination_start_mode"):
            config(imagination_start_mode="invalid")
        with self.assertRaisesRegex(ValueError, "actor_gradient"):
            config(actor_gradient="invalid")
        valid = replace(config(), imagination_start_mode="one_per_sequence")
        self.assertEqual(valid.imagination_start_mode, "one_per_sequence")


if __name__ == "__main__":
    unittest.main()
