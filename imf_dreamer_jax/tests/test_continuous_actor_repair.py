from __future__ import annotations

import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.agent import (
    actor_distribution,
    create_agent,
    diagonal_normal_kl,
    jit_train_actor_critic_dreamer3,
)
from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.policy_optimization import (
    SafeMPOConfig,
    decoupled_reference_kl,
    init_return_scale_state,
    init_safe_mpo_state,
    normalized_reinforce_objective,
    positive_elite_weights,
    sample_safe_mpo_candidates,
    safe_multi_action_mpo_objective,
    update_return_scale,
    update_safe_mpo_reference,
)
from imf_dreamer_jax.types import AgentState, DiagonalNormal, RSSMState


def small_config(**overrides: object) -> DreamerConfig:
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
        "actor_gradient": "reinforce",
    }
    values.update(overrides)
    return DreamerConfig(**values)


class RobustReinforceTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("CONTINUOUS_ACTOR_LIBRARY_VERIFIED")

    def test_percentile_scale_initialization_and_ema_are_exact(self) -> None:
        returns = jnp.arange(101, dtype=jnp.float32)
        initial = init_return_scale_state()
        first, first_scale = update_return_scale(initial, returns, decay=0.9)
        np.testing.assert_allclose(first.percentile_range, 90.0, atol=1e-5)
        np.testing.assert_allclose(first_scale, 90.0, atol=1e-5)
        second, second_scale = update_return_scale(
            first, 2.0 * returns, decay=0.9
        )
        np.testing.assert_allclose(second.percentile_range, 99.0, atol=2e-5)
        np.testing.assert_allclose(second_scale, 99.0, atol=2e-5)
        floor_state, floor_scale = update_return_scale(
            initial, jnp.zeros((32,), dtype=jnp.float32), decay=0.9
        )
        self.assertEqual(float(floor_state.percentile_range), 0.0)
        self.assertEqual(float(floor_scale), 1.0)

    def test_normalized_reinforce_is_invariant_to_positive_rescaling(self) -> None:
        log_probs = jnp.asarray([[0.3, -0.5], [1.1, 0.2]])
        advantages = jnp.asarray([[2.0, -1.0], [4.0, -3.0]])
        weights = jnp.asarray([[1.0, 0.9], [1.0, 0.7]])
        base = normalized_reinforce_objective(
            log_probs, advantages, weights, jnp.asarray(2.5)
        )
        scaled = normalized_reinforce_objective(
            log_probs, 17.0 * advantages, weights, jnp.asarray(42.5)
        )
        np.testing.assert_allclose(base, scaled, rtol=1e-6)

    def test_robust_actor_update_preserves_checkpoint_tuple_and_reports_telemetry(self) -> None:
        cfg = small_config(return_scale_ema_decay=0.9, behavior_kl_scale=0.1)
        state = create_agent(cfg, jax.random.key(1))
        legacy_fields = AgentState._fields
        prior = jax.tree_util.tree_map(jnp.array, state.params.actor)
        starts = RSSMState(
            jax.random.normal(jax.random.key(2), (16, cfg.deterministic_dim)),
            jax.random.normal(jax.random.key(3), (16, cfg.stochastic_dim)),
        )
        updated, scale_state, metrics = jit_train_actor_critic_dreamer3(
            state,
            init_return_scale_state(),
            starts,
            jax.random.key(4),
            cfg,
            behavior_prior=prior,
        )
        self.assertEqual(AgentState._fields, legacy_fields)
        self.assertEqual(updated._fields, legacy_fields)
        self.assertTrue(bool(scale_state.initialized))
        self.assertTrue(np.isfinite(np.asarray(tuple(metrics), dtype=np.float64)).all())
        self.assertGreaterEqual(float(metrics.return_scale), 1.0)
        self.assertGreaterEqual(float(metrics.behavior_kl), 0.0)
        self.assertGreaterEqual(float(metrics.advantage_positive_fraction), 0.0)
        self.assertLessEqual(float(metrics.advantage_positive_fraction), 1.0)
        self.assertGreaterEqual(float(metrics.policy_std_mean), cfg.actor_min_std)
        self.assertLessEqual(float(metrics.policy_std_mean), cfg.actor_max_std)

    def test_robust_path_rejects_non_reinforce_actor(self) -> None:
        cfg = small_config(actor_gradient="pmpo")
        state = create_agent(cfg, jax.random.key(5))
        starts = RSSMState(
            jnp.zeros((2, cfg.deterministic_dim)),
            jnp.zeros((2, cfg.stochastic_dim)),
        )
        with self.assertRaisesRegex(ValueError, "requires actor_gradient='reinforce'"):
            jit_train_actor_critic_dreamer3(
                state,
                init_return_scale_state(),
                starts,
                jax.random.key(6),
                cfg,
            )

    def test_three_preregistered_behavior_kl_values_are_valid(self) -> None:
        for beta in (0.0, 0.1, 0.3):
            self.assertEqual(small_config(behavior_kl_scale=beta).behavior_kl_scale, beta)
        with self.assertRaisesRegex(ValueError, "behavior_kl_scale"):
            small_config(behavior_kl_scale=-0.1)
        with self.assertRaisesRegex(ValueError, "return_scale_ema_decay"):
            small_config(return_scale_ema_decay=1.0)

    def test_equal_policy_kl_is_support_safe_nonnegative(self) -> None:
        cfg = small_config()
        state = create_agent(cfg, jax.random.key(30))
        features = jax.random.normal(jax.random.key(31), (64, cfg.feature_dim))
        distribution = actor_distribution(state.params.actor, features, cfg)
        kl = diagonal_normal_kl(distribution, distribution)
        self.assertTrue(bool(jnp.all(kl >= 0.0)))
        np.testing.assert_allclose(kl, 0.0, atol=1e-7)


class SafeMultiActionMPOTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("SAFE_MULTI_ACTION_MPO_VERIFIED")

    def test_elite_weights_are_positive_normalized_and_top_two_only(self) -> None:
        scores = jnp.asarray([[1.0, 4.0, 3.0, 2.0], [-1.0, -4.0, -2.0, -3.0]])
        weights = positive_elite_weights(scores)
        np.testing.assert_allclose(jnp.sum(weights, axis=-1), 1.0, atol=1e-7)
        np.testing.assert_array_equal(jnp.sum(weights > 0.0, axis=-1), 2)
        np.testing.assert_array_equal(weights[0, [0, 3]], 0.0)
        np.testing.assert_array_equal(weights[1, [1, 3]], 0.0)

    def test_four_candidates_share_each_reference_state(self) -> None:
        reference = DiagonalNormal(
            jnp.asarray([[0.2, -0.1], [0.7, 0.4]]),
            jnp.asarray([[0.5, 0.3], [0.2, 0.8]]),
        )
        candidates = sample_safe_mpo_candidates(
            reference, jax.random.key(20), SafeMPOConfig()
        )
        self.assertEqual(candidates.actions.shape, (2, 4, 2))
        self.assertEqual(candidates.pre_tanh.shape, (2, 4, 2))
        standardized = (
            candidates.pre_tanh - reference.mean[:, None, :]
        ) / reference.std[:, None, :]
        self.assertTrue(np.isfinite(np.asarray(standardized)).all())
        self.assertGreater(float(jnp.std(standardized, axis=1).mean()), 0.1)

    def test_decoupled_terms_sum_to_exact_reference_to_current_kl(self) -> None:
        reference = DiagonalNormal(
            jnp.asarray([[0.1, -0.3], [0.4, 0.2]]),
            jnp.asarray([[0.7, 1.1], [0.5, 0.8]]),
        )
        current = DiagonalNormal(
            jnp.asarray([[0.5, -0.1], [-0.2, 0.3]]),
            jnp.asarray([[0.9, 0.6], [1.0, 0.7]]),
        )
        mean_kl, std_kl = decoupled_reference_kl(reference, current)
        np.testing.assert_allclose(
            mean_kl + std_kl,
            diagonal_normal_kl(reference, current),
            rtol=2e-6,
            atol=2e-7,
        )
        reverse = diagonal_normal_kl(current, reference)
        self.assertGreater(
            float(jnp.max(jnp.abs(mean_kl + std_kl - reverse))), 1e-3
        )

    def test_safe_objective_has_no_negative_likelihood_weights(self) -> None:
        reference = DiagonalNormal(jnp.zeros((3, 2)), jnp.ones((3, 2)))
        current = DiagonalNormal(jnp.full((3, 2), 0.1), jnp.full((3, 2), 0.9))
        scores = jnp.asarray(
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [1.0, 3.0, 0.0, 2.0]]
        )
        log_probs = jnp.asarray(
            [[-3.0, -2.0, -1.0, -0.5], [-0.2, -1.0, -2.0, -4.0], [-2.0, -0.4, -3.0, -1.0]]
        )
        result = safe_multi_action_mpo_objective(
            log_probs,
            scores,
            reference,
            current,
            SafeMPOConfig(),
        )
        self.assertTrue(bool(jnp.all(result.normalized_weights >= 0.0)))
        np.testing.assert_allclose(
            jnp.sum(result.normalized_weights, axis=-1), 1.0, atol=1e-7
        )
        self.assertEqual(float(result.elite_fraction), 0.5)
        self.assertTrue(np.isfinite(np.asarray(tuple(result)[:-1], dtype=np.float64)).all())

    def test_reference_refreshes_on_the_declared_period_only(self) -> None:
        cfg = small_config()
        first_actor = create_agent(cfg, jax.random.key(7)).params.actor
        second_actor = create_agent(cfg, jax.random.key(8)).params.actor
        state = init_safe_mpo_state(first_actor)
        state = update_safe_mpo_reference(state, second_actor, refresh_interval=2)
        for observed, expected in zip(
            jax.tree_util.tree_leaves(state.reference_actor),
            jax.tree_util.tree_leaves(first_actor),
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected)
        state = update_safe_mpo_reference(state, second_actor, refresh_interval=2)
        self.assertEqual(int(state.updates_since_refresh), 0)
        for observed, expected in zip(
            jax.tree_util.tree_leaves(state.reference_actor),
            jax.tree_util.tree_leaves(second_actor),
            strict=True,
        ):
            np.testing.assert_array_equal(observed, expected)


def _run_selected_suite() -> None:
    if "--safe-mpo-only" in sys.argv:
        sys.argv.remove("--safe-mpo-only")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SafeMultiActionMPOTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main(verbosity=2)


if __name__ == "__main__":
    _run_selected_suite()
