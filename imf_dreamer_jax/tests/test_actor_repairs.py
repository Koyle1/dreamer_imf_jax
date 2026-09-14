from __future__ import annotations

from dataclasses import replace
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.agent import (
    actor_distribution,
    create_agent,
    imagine,
    init_actor,
    init_critic,
    jit_train_actor_critic,
    loss_bearing_imagination_starts,
    sample_actor,
    squashed_normal_entropy_sample,
    tanh_normal_log_prob,
    train_actor_critic,
)
from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.nn import mlp, tree_global_norm
from imf_dreamer_jax.types import AgentParams, RSSMState


def small_config(**overrides: object) -> DreamerConfig:
    values: dict[str, object] = {
        "observation_shape": (3,),
        "action_dim": 2,
        "deterministic_dim": 5,
        "stochastic_dim": 3,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "prior": "gaussian",
        "imagination_horizon": 4,
        "actor_init_scale": 0.01,
        "actor_mean_bound": 5.0,
        "actor_min_std": 0.1,
        "actor_max_std": 1.0,
        "reward_output_init_scale": 1.0,
    }
    values.update(overrides)
    return DreamerConfig(**values)


def actor_with_output_bias(params: object, bias: jax.Array) -> object:
    layers = list(params["layers"])
    layers[-1] = {**layers[-1], "bias": bias}
    return {**params, "layers": tuple(layers)}


def zero_tree(tree: object) -> object:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def scale_reward_output(model: object, factor: float) -> object:
    reward = model["reward"]
    layers = list(reward["layers"])
    layers[-1] = {
        "weight": factor * layers[-1]["weight"],
        "bias": factor * layers[-1]["bias"],
    }
    return {
        **model,
        "reward": {**reward, "layers": tuple(layers)},
    }


class ActorDistributionTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ACTOR_DISTRIBUTION_REPAIRS_OK")

    def test_actor_output_gain_is_exactly_small(self) -> None:
        key = jax.random.key(1)
        small = init_actor(small_config(actor_init_scale=0.01), key)
        reference = init_actor(small_config(actor_init_scale=1.0), key)
        np.testing.assert_allclose(
            small["layers"][-1]["weight"],
            0.01 * reference["layers"][-1]["weight"],
            rtol=2e-6,
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            small["layers"][-1]["bias"],
            np.zeros((4,), dtype=np.float32),
        )

    def test_distribution_parameters_are_bounded(self) -> None:
        cfg = small_config(actor_mean_bound=3.0, actor_min_std=0.2, actor_max_std=0.7)
        params = init_actor(cfg, jax.random.key(2))
        params = actor_with_output_bias(
            params, jnp.asarray([100.0, -100.0, -100.0, 100.0])
        )
        distribution = actor_distribution(
            params, jnp.zeros((3, cfg.feature_dim)), cfg
        )
        self.assertTrue(bool(jnp.all(jnp.abs(distribution.mean) <= 3.0)))
        self.assertTrue(bool(jnp.all(jnp.abs(distribution.mean) > 2.99)))
        self.assertTrue(bool(jnp.all(distribution.std >= 0.2)))
        self.assertTrue(bool(jnp.all(distribution.std <= 0.7)))

    def test_change_of_variables_matches_direct_reference(self) -> None:
        mean = jnp.asarray([[0.2, -0.4]], dtype=jnp.float32)
        std = jnp.asarray([[0.7, 1.2]], dtype=jnp.float32)
        pre_tanh = jnp.asarray([[0.3, -1.1]], dtype=jnp.float32)
        observed = np.asarray(tanh_normal_log_prob(mean, std, pre_tanh))
        mean_np, std_np, z_np = map(np.asarray, (mean, std, pre_tanh))
        normal = -0.5 * ((z_np - mean_np) / std_np) ** 2
        normal -= np.log(std_np) + 0.5 * math.log(2.0 * math.pi)
        reference = np.sum(normal - np.log(1.0 - np.tanh(z_np) ** 2), axis=-1)
        np.testing.assert_allclose(observed, reference, rtol=2e-6, atol=2e-6)
        extreme = tanh_normal_log_prob(
            jnp.zeros((1, 2)), jnp.ones((1, 2)), jnp.asarray([[30.0, -30.0]])
        )
        self.assertTrue(np.isfinite(np.asarray(extreme)).all())

    def test_squashed_entropy_exposes_saturation_that_fake_entropy_hides(self) -> None:
        mean = jnp.asarray([[8.0]])
        std = jnp.ones_like(mean)
        actual = float(squashed_normal_entropy_sample(mean, std, mean)[0])
        fake_normal_entropy = 0.5 + 0.5 * math.log(2.0 * math.pi)
        self.assertLess(actual, fake_normal_entropy - 10.0)
        entropy_gradient = jax.grad(
            lambda location: squashed_normal_entropy_sample(
                location[None, None], jnp.ones((1, 1)), location[None, None]
            )[0]
        )(jnp.asarray(5.0))
        self.assertLess(float(entropy_gradient), -1.9)

    def test_actor_sample_reports_its_exact_density(self) -> None:
        cfg = small_config()
        params = init_actor(cfg, jax.random.key(3))
        features = jnp.ones((7, cfg.feature_dim))
        sample = sample_actor(params, features, jax.random.key(4), cfg)
        expected = tanh_normal_log_prob(
            sample.distribution.mean, sample.distribution.std, sample.pre_tanh
        )
        np.testing.assert_allclose(sample.log_prob, expected, rtol=1e-6)
        np.testing.assert_allclose(sample.entropy, -expected, rtol=1e-6)
        self.assertTrue(bool(jnp.all(jnp.abs(sample.action) <= 1.0)))


class ActorCriticStabilityTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ACTOR_CRITIC_STABILITY_OK")

    def test_default_stops_model_action_gradient_but_dynamics_option_does_not(self) -> None:
        reinforce_cfg = small_config(actor_gradient="reinforce")
        dynamics_cfg = replace(reinforce_cfg, actor_gradient="dynamics")
        state = create_agent(reinforce_cfg, jax.random.key(10))
        start = RSSMState(
            jnp.zeros((4, reinforce_cfg.deterministic_dim)),
            jnp.zeros((4, reinforce_cfg.stochastic_dim)),
        )

        def imagined_reward(actor_params: object, cfg: DreamerConfig) -> jax.Array:
            params = AgentParams(
                state.params.world_model, actor_params, state.params.critic
            )
            return jnp.sum(imagine(params, start, jax.random.key(11), cfg).rewards)

        reinforce_grad = jax.grad(imagined_reward)(state.params.actor, reinforce_cfg)
        dynamics_grad = jax.grad(imagined_reward)(state.params.actor, dynamics_cfg)
        self.assertEqual(float(tree_global_norm(reinforce_grad)), 0.0)
        self.assertGreater(float(tree_global_norm(dynamics_grad)), 1e-7)

    def test_critic_output_is_exactly_zero_when_requested(self) -> None:
        cfg = small_config(critic_output_init_scale=0.0)
        params = init_critic(cfg, jax.random.key(101))
        values = mlp(params, jax.random.normal(jax.random.key(102), (9, cfg.feature_dim)))
        np.testing.assert_array_equal(values, np.zeros((9, 1), dtype=np.float32))

    def test_every_post_burn_in_state_becomes_an_imagination_start(self) -> None:
        deterministic = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3)
        stochastic = jnp.arange(2 * 5 * 2, dtype=jnp.float32).reshape(2, 5, 2)
        starts = loss_bearing_imagination_starts(
            RSSMState(deterministic, stochastic), burn_in=2
        )
        self.assertEqual(starts.deterministic.shape, (6, 3))
        self.assertEqual(starts.stochastic.shape, (6, 2))
        np.testing.assert_array_equal(starts.deterministic, deterministic[:, 2:].reshape(6, 3))
        with self.assertRaisesRegex(ValueError, "leave at least one"):
            loss_bearing_imagination_starts(
                RSSMState(deterministic, stochastic), burn_in=5
            )

    def test_actor_and_critic_updates_use_linear_learning_rate_warmup(self) -> None:
        full_cfg = small_config(actor_critic_warmup_steps=0)
        warm_cfg = replace(full_cfg, actor_critic_warmup_steps=100)
        state = create_agent(full_cfg, jax.random.key(103))
        start = RSSMState(
            jax.random.normal(jax.random.key(104), (8, full_cfg.deterministic_dim)),
            jax.random.normal(jax.random.key(105), (8, full_cfg.stochastic_dim)),
        )
        full, _ = train_actor_critic(state, start, jax.random.key(106), full_cfg)
        warm, _ = train_actor_critic(state, start, jax.random.key(106), warm_cfg)

        def delta_norm(before: object, after: object) -> float:
            delta = jax.tree_util.tree_map(lambda x, y: y - x, before, after)
            return float(tree_global_norm(delta))

        for before, full_after, warm_after in (
            (state.params.actor, full.params.actor, warm.params.actor),
            (state.params.critic, full.params.critic, warm.params.critic),
        ):
            full_delta = delta_norm(before, full_after)
            warm_delta = delta_norm(before, warm_after)
            self.assertGreater(full_delta, 0.0)
            np.testing.assert_allclose(warm_delta, full_delta / 100.0, rtol=2e-3)

    def test_return_normalization_makes_positive_reward_rescaling_invariant(self) -> None:
        cfg = small_config(
            actor_entropy_scale=0.0,
            actor_action_l2_scale=0.0,
            return_normalization_epsilon=1e-12,
        )
        state = create_agent(cfg, jax.random.key(12))
        critic_params = zero_tree(state.params.critic)
        common_params = state.params._replace(critic=critic_params)
        base = state._replace(params=common_params, slow_critic=critic_params)
        base_model = scale_reward_output(common_params.world_model, 100.0)
        base = base._replace(
            params=common_params._replace(world_model=base_model)
        )
        scaled_model = scale_reward_output(common_params.world_model, 1700.0)
        scaled = base._replace(
            params=common_params._replace(world_model=scaled_model)
        )
        base_updated, base_metrics = train_actor_critic(
            base, RSSMState(
                jnp.zeros((6, cfg.deterministic_dim)),
                jnp.zeros((6, cfg.stochastic_dim)),
            ), jax.random.key(13), cfg
        )
        scaled_updated, scaled_metrics = train_actor_critic(
            scaled, RSSMState(
                jnp.zeros((6, cfg.deterministic_dim)),
                jnp.zeros((6, cfg.stochastic_dim)),
            ), jax.random.key(13), cfg
        )
        for base_leaf, scaled_leaf in zip(
            jax.tree_util.tree_leaves(base_updated.params.actor),
            jax.tree_util.tree_leaves(scaled_updated.params.actor),
            strict=True,
        ):
            np.testing.assert_allclose(base_leaf, scaled_leaf, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(
            scaled_metrics.return_scale,
            17.0 * base_metrics.return_scale,
            rtol=3e-5,
        )
        self.assertGreater(float(base_metrics.return_scale), 1.0)
        self.assertGreater(float(base_metrics.actor_grad_norm), 1e-6)

    def test_action_regularization_has_an_independent_actor_gradient(self) -> None:
        cfg = small_config(actor_entropy_scale=0.0, actor_action_l2_scale=0.0)
        state = create_agent(cfg, jax.random.key(14))
        critic_params = zero_tree(state.params.critic)
        model = scale_reward_output(state.params.world_model, 0.0)
        params = state.params._replace(world_model=model, critic=critic_params)
        state = state._replace(params=params, slow_critic=critic_params)
        start = RSSMState(
            jnp.zeros((5, cfg.deterministic_dim)),
            jnp.zeros((5, cfg.stochastic_dim)),
        )
        _, no_regularization = train_actor_critic(
            state, start, jax.random.key(15), cfg
        )
        _, with_regularization = train_actor_critic(
            state,
            start,
            jax.random.key(15),
            replace(cfg, actor_action_l2_scale=1.0),
        )
        self.assertLess(float(no_regularization.actor_grad_norm), 1e-9)
        self.assertEqual(float(no_regularization.return_scale), 1.0)
        self.assertGreater(float(with_regularization.actor_grad_norm), 1e-5)

    def test_slow_critic_is_an_exact_ema_and_legacy_none_is_supported(self) -> None:
        cfg = small_config(slow_critic_fraction=0.25)
        state = create_agent(cfg, jax.random.key(16))
        start = RSSMState(
            jnp.zeros((5, cfg.deterministic_dim)),
            jnp.zeros((5, cfg.stochastic_dim)),
        )
        updated, metrics = train_actor_critic(
            state, start, jax.random.key(17), cfg
        )
        for old, online, target in zip(
            jax.tree_util.tree_leaves(state.slow_critic),
            jax.tree_util.tree_leaves(updated.params.critic),
            jax.tree_util.tree_leaves(updated.slow_critic),
            strict=True,
        ):
            np.testing.assert_allclose(
                target, 0.75 * old + 0.25 * online, rtol=2e-6, atol=2e-7
            )
        measured_delta = tree_global_norm(
            jax.tree_util.tree_map(
                lambda online, target: online - target,
                updated.params.critic,
                updated.slow_critic,
            )
        )
        np.testing.assert_allclose(metrics.slow_critic_delta, measured_delta, rtol=1e-6)
        legacy, legacy_metrics = train_actor_critic(
            state._replace(slow_critic=None), start, jax.random.key(18), cfg
        )
        self.assertIsNotNone(legacy.slow_critic)
        self.assertTrue(np.isfinite(float(legacy_metrics.slow_critic_delta)))


class ActorMetricTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ACTOR_METRICS_OK")

    def test_jitted_update_exposes_finite_scalar_diagnostics(self) -> None:
        cfg = small_config()
        state = create_agent(cfg, jax.random.key(20))
        start = RSSMState(
            jnp.zeros((8, cfg.deterministic_dim)),
            jnp.zeros((8, cfg.stochastic_dim)),
        )
        updated, metrics = jit_train_actor_critic(
            state, start, jax.random.key(21), cfg
        )
        expected_names = (
            "actor_loss",
            "critic_loss",
            "mean_imagined_return",
            "actor_grad_norm",
            "critic_grad_norm",
            "action_saturation",
            "mean_pre_tanh_abs",
            "squashed_entropy",
            "return_scale",
            "slow_critic_delta",
            "deterministic_action_mean",
            "pre_tanh_mean",
            "policy_std_mean",
            "behavior_kl",
            "advantage_mean",
            "advantage_positive_fraction",
        )
        self.assertEqual(metrics._fields, expected_names)
        values = np.asarray([float(value) for value in metrics])
        self.assertTrue(np.isfinite(values).all())
        self.assertGreaterEqual(float(metrics.actor_grad_norm), 0.0)
        self.assertGreater(float(metrics.critic_grad_norm), 0.0)
        self.assertGreaterEqual(float(metrics.action_saturation), 0.0)
        self.assertLessEqual(float(metrics.action_saturation), 1.0)
        self.assertLessEqual(float(metrics.mean_pre_tanh_abs), cfg.actor_mean_bound)
        self.assertGreaterEqual(float(metrics.return_scale), 1.0)
        self.assertGreaterEqual(float(metrics.slow_critic_delta), 0.0)
        self.assertGreaterEqual(float(metrics.policy_std_mean), cfg.actor_min_std)
        self.assertLessEqual(float(metrics.policy_std_mean), cfg.actor_max_std)
        self.assertGreaterEqual(float(metrics.behavior_kl), 0.0)
        self.assertGreaterEqual(float(metrics.advantage_positive_fraction), 0.0)
        self.assertLessEqual(float(metrics.advantage_positive_fraction), 1.0)
        self.assertEqual(int(updated.actor_optimizer.step), 1)
        self.assertEqual(int(updated.critic_optimizer.step), 1)

    def test_saturation_and_squashed_entropy_metrics_detect_boundary_policy(self) -> None:
        cfg = small_config(
            actor_mean_bound=5.0,
            actor_min_std=0.1,
            actor_max_std=0.2,
        )
        state = create_agent(cfg, jax.random.key(22))
        extreme_bias = jnp.asarray([8.0, 8.0, -8.0, -8.0])
        extreme_actor = actor_with_output_bias(state.params.actor, extreme_bias)
        state = state._replace(params=state.params._replace(actor=extreme_actor))
        start = RSSMState(
            jnp.zeros((8, cfg.deterministic_dim)),
            jnp.zeros((8, cfg.stochastic_dim)),
        )
        _, metrics = train_actor_critic(state, start, jax.random.key(23), cfg)
        self.assertGreater(float(metrics.action_saturation), 0.99)
        self.assertGreater(float(metrics.mean_pre_tanh_abs), 4.5)
        self.assertLess(float(metrics.squashed_entropy), -5.0)
        self.assertGreater(float(metrics.actor_grad_norm), 1e-5)


if __name__ == "__main__":
    unittest.main()
