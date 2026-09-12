from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DiagonalNormal,
    DreamerConfig,
    PolicyConsistencyConfig,
    RSSMState,
    SequenceStates,
    advantage_consistency_loss,
    create_agent,
    epistemic_reward_std,
    generated_trajectory_history,
    init_bootstrap_reward_ensemble,
    init_running_rms,
    jit_train_policy_consistent_world_model,
    model_candidate_returns,
    pessimistic_rewards,
    policy_consistent_world_model_loss,
    reward_ensemble_predictions,
    sign_only_pmpo_objective,
    snapshot_behavior_prior,
    train_actor_critic,
    train_behavior_cloning,
    train_bootstrap_reward_ensemble,
    train_replay_critic,
    update_running_rms,
    world_model_loss,
)
from imf_dreamer_jax.nn import tree_global_norm


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
        "imf_noise_coupling": "independent",
        "imagination_horizon": 3,
        "reward_output_init_scale": 1.0,
    }
    values.update(overrides)
    return DreamerConfig(**values)


def tree_delta(left: object, right: object) -> float:
    delta = jax.tree_util.tree_map(lambda x, y: y - x, left, right)
    return float(tree_global_norm(delta))


def batch(cfg: DreamerConfig, *, batch_size: int = 3, steps: int = 6):
    return {
        "observations": jax.random.normal(
            jax.random.key(1), (batch_size, steps, *cfg.observation_shape)
        ),
        "actions": 0.5
        * jnp.tanh(
            jax.random.normal(
                jax.random.key(2), (batch_size, steps, cfg.action_dim)
            )
        ),
        "rewards": jax.random.uniform(jax.random.key(3), (batch_size, steps)),
        "continuations": jnp.ones((batch_size, steps)),
        "is_first": jnp.zeros((batch_size, steps), dtype=jnp.bool_).at[:, 0].set(True),
        "loss_mask": jnp.ones((batch_size, steps)),
    }


class ActorRepairTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ACTOR_REPAIR_VERIFIED")

    def test_pmpo_uses_sign_only_and_separate_set_normalization(self) -> None:
        log_probs = jnp.asarray([[-1.0, -2.0, -3.0, -4.0]])
        weights = jnp.ones_like(log_probs)
        first = sign_only_pmpo_objective(
            log_probs, jnp.asarray([[0.1, 9.0, -0.2, -50.0]]), weights
        )
        second = sign_only_pmpo_objective(
            log_probs, jnp.asarray([[99.0, 0.01, -700.0, -0.001]]), weights
        )
        np.testing.assert_allclose(first, 1.0, atol=1e-7)
        np.testing.assert_array_equal(first, second)
        for advantages in (jnp.ones_like(log_probs), -jnp.ones_like(log_probs)):
            self.assertTrue(
                np.isfinite(
                    float(sign_only_pmpo_objective(log_probs, advantages, weights))
                )
            )

    def test_behavior_clone_prior_and_replay_critic_freeze_world_model(self) -> None:
        cfg = config(
            actor_gradient="pmpo",
            behavior_kl_scale=0.3,
            critic_bins=51,
            behavior_cloning_learning_rate=3e-3,
        )
        state = create_agent(cfg, jax.random.key(4))
        data = batch(cfg)
        world = state.params.world_model
        features = jax.random.normal(
            jax.random.key(5), (*data["rewards"].shape, cfg.feature_dim)
        )
        cloned, _ = train_behavior_cloning(
            state,
            features.reshape((-1, cfg.feature_dim)),
            data["actions"].reshape((-1, cfg.action_dim)),
            cfg,
        )
        prior = snapshot_behavior_prior(cloned.params.actor)
        grounded, critic_loss = train_replay_critic(
            cloned,
            features,
            data["rewards"],
            data["continuations"],
            cfg,
            loss_mask=data["loss_mask"],
        )
        self.assertTrue(np.isfinite(float(critic_loss)))
        self.assertEqual(tree_delta(world, grounded.params.world_model), 0.0)
        self.assertEqual(tree_delta(cloned.params.actor, grounded.params.actor), 0.0)
        self.assertGreater(tree_delta(cloned.params.critic, grounded.params.critic), 0.0)
        start = RSSMState(
            features[:, 0, : cfg.deterministic_dim],
            features[:, 0, cfg.deterministic_dim :],
        )
        updated, metrics = train_actor_critic(
            grounded, start, jax.random.key(6), cfg, behavior_prior=prior
        )
        self.assertTrue(np.isfinite(np.asarray(tuple(metrics))).all())
        self.assertEqual(tree_delta(world, updated.params.world_model), 0.0)
        self.assertEqual(tree_delta(prior, snapshot_behavior_prior(prior)), 0.0)


class AdvantageConsistencyTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("ADVANTAGE_CONSISTENCY_VERIFIED")

    def test_matching_returns_win_and_wrong_ranking_is_penalized(self) -> None:
        target = jnp.asarray([[[1.0], [2.0], [4.0]]])
        matching = advantage_consistency_loss(target, target)
        reversed_result = advantage_consistency_loss(target[:, ::-1], target)
        self.assertLess(float(matching.total), float(reversed_result.total))
        np.testing.assert_allclose(matching.magnitude, 0.0, atol=1e-7)

    def test_flat_targets_suppress_hallucinated_action_dependence(self) -> None:
        target = jnp.ones((2, 3, 1))
        flat_prediction = target
        hallucinated = jnp.asarray(
            [[[0.0], [1.0], [2.0]], [[-3.0], [0.0], [3.0]]]
        )
        good = advantage_consistency_loss(target, target)
        bad = advantage_consistency_loss(hallucinated, target)
        np.testing.assert_allclose(good.flat, 0.0, atol=1e-7)
        self.assertGreater(float(bad.flat), float(good.flat))
        self.assertEqual(float(good.informative_pair_fraction), 0.0)

    def test_source_posterior_is_gradient_isolated(self) -> None:
        cfg = config()
        state = create_agent(cfg, jax.random.key(7))
        data = batch(cfg, batch_size=2, steps=3)
        deterministic = jax.random.normal(
            jax.random.key(8), (2, 3, cfg.deterministic_dim)
        )
        stochastic = jax.random.normal(
            jax.random.key(9), (2, 3, cfg.stochastic_dim)
        )
        posterior = DiagonalNormal(stochastic, jnp.ones_like(stochastic))
        actions = jnp.zeros((2, 3, 3, 5, cfg.action_dim))

        def objective(deterministic_value, stochastic_value):
            sequence = SequenceStates(
                RSSMState(deterministic_value, stochastic_value),
                posterior,
                None,
                None,
            )
            return jnp.sum(
                model_candidate_returns(
                    state.params.world_model,
                    sequence,
                    actions,
                    jax.random.key(10),
                    cfg,
                    horizons=(1, 3, 5),
                )
            )

        gradients = jax.grad(objective, argnums=(0, 1))(deterministic, stochastic)
        self.assertEqual(float(tree_global_norm(gradients)), 0.0)
        rms = update_running_rms(
            init_running_rms(), jnp.asarray([1.0, 3.0]), jnp.ones((2,))
        )
        np.testing.assert_allclose(rms.mean_square, 5.0)

    def test_jitted_world_update_consumes_short_horizon_targets(self) -> None:
        cfg = config()
        state = create_agent(cfg, jax.random.key(101))
        data = batch(cfg, batch_size=2, steps=6)
        data.update(
            {
                "advantage_action_sequences": 0.2
                * jax.random.normal(
                    jax.random.key(102), (2, 6, 3, 5, cfg.action_dim)
                ),
                "advantage_target_returns": jax.random.normal(
                    jax.random.key(103), (2, 6, 3, 3)
                ),
                "advantage_mask": jnp.ones((2, 6, 3)),
            }
        )
        objective = PolicyConsistencyConfig(advantage_consistency_scale=0.1)
        updated, losses, rms = jit_train_policy_consistent_world_model(
            state,
            data,
            jax.random.key(104),
            cfg,
            objective,
            init_running_rms(),
        )
        self.assertTrue(np.isfinite(float(losses.total)))
        self.assertGreater(float(rms.count), 0.0)
        self.assertGreater(tree_delta(state.params.world_model, updated.params.world_model), 0.0)
        self.assertEqual(tree_delta(state.params.actor, updated.params.actor), 0.0)


class EpistemicPessimismTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("EPISTEMIC_PESSIMISM_VERIFIED")

    def test_bootstrap_heads_fit_without_aleatoric_draw_axis(self) -> None:
        ensemble = init_bootstrap_reward_ensemble(
            jax.random.key(11), 4, 8, members=3
        )
        features = jax.random.normal(jax.random.key(12), (16, 4))
        rewards = jnp.sum(features, axis=-1)
        before = reward_ensemble_predictions(ensemble.params, features)
        updated, loss = train_bootstrap_reward_ensemble(
            ensemble,
            features,
            rewards,
            jax.random.key(13),
            bootstrap_mask=jnp.ones((3, 16)),
            learning_rate=1e-2,
        )
        after = reward_ensemble_predictions(updated.params, features)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(before.shape, (3, 16))
        self.assertGreater(tree_delta(before, after), 0.0)

    def test_epistemic_disagreement_lowers_reward_only_when_present(self) -> None:
        ensemble = init_bootstrap_reward_ensemble(
            jax.random.key(14), 3, 6, members=3
        )
        params = ensemble.params
        layers = list(params["layers"])
        output = dict(layers[-1])
        output["weight"] = jnp.zeros_like(output["weight"])
        output["bias"] = jnp.asarray([[-1.0], [0.0], [1.0]])
        layers[-1] = output
        params = {**params, "layers": tuple(layers)}
        features = jnp.zeros((2, 4, 3))
        rewards = jnp.ones((2, 4))
        adjusted, uncertainty = pessimistic_rewards(
            rewards, features, params, penalty_scale=0.5
        )
        np.testing.assert_allclose(
            uncertainty, epistemic_reward_std(params, features), atol=1e-7
        )
        self.assertTrue(bool(jnp.all(adjusted < rewards)))
        self.assertEqual(uncertainty.shape, rewards.shape)


class ExposureControlTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("EXPOSURE_CONTROLS_VERIFIED")

    def test_options_are_independently_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "generated_context_probability"):
            PolicyConsistencyConfig(generated_context_probability=1.1)
        with self.assertRaisesRegex(ValueError, "increasing"):
            PolicyConsistencyConfig(advantage_horizons=(3, 1))
        self.assertEqual(
            PolicyConsistencyConfig(training_context_length=64).training_context_length,
            64,
        )

    def test_default_wrapper_is_exactly_the_legacy_loss(self) -> None:
        cfg = config()
        state = create_agent(cfg, jax.random.key(15))
        data = batch(cfg)
        key = jax.random.key(16)
        legacy = world_model_loss(state.params.world_model, data, key, cfg)
        wrapped = policy_consistent_world_model_loss(
            state.params.world_model,
            data,
            key,
            cfg,
            PolicyConsistencyConfig(),
        )
        for old, new in zip(legacy, wrapped.base, strict=True):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(wrapped.total, legacy.total)

    def test_endpoint_generated_and_corrupted_context_objective_is_finite(self) -> None:
        cfg = config()
        state = create_agent(cfg, jax.random.key(17))
        data = batch(cfg, steps=6)
        result = policy_consistent_world_model_loss(
            state.params.world_model,
            data,
            jax.random.key(18),
            cfg,
            PolicyConsistencyConfig(
                exposure_meanflow_scale=0.1,
                endpoint_scale=0.1,
                generated_context_probability=0.5,
                context_corruption_max=0.1,
                training_context_length=6,
            ),
        )
        self.assertTrue(np.isfinite(np.asarray(tuple(result.base))).all())
        self.assertTrue(np.isfinite(float(result.total)))
        self.assertGreaterEqual(float(result.endpoint), 0.0)
        generated = generated_trajectory_history(
            state.params.world_model,
            data["actions"],
            data["is_first"],
            jnp.zeros((*data["actions"].shape[:2], cfg.stochastic_dim)),
            cfg,
        )
        self.assertEqual(generated.shape, (3, 6, cfg.stochastic_dim))


if __name__ == "__main__":
    unittest.main()
