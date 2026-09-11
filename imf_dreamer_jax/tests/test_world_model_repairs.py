from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.replay import SequenceReplayBuffer
from imf_dreamer_jax.types import DiagonalNormal, RSSMState, SequenceStates
from imf_dreamer_jax.world_model import (
    _distance_consistency_loss,
    init_world_model,
    observe_sequence,
    predict_reward,
    predict_reward_logits,
    world_model_loss,
)


def repaired_config(prior: str, *, burn_in: int = 0) -> DreamerConfig:
    return DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=4,
        stochastic_dim=2,
        embedding_dim=4,
        hidden_dim=6,
        prior=prior,
        burn_in=burn_in,
        overshooting_horizon=15,
        overshooting_distances=(5, 15),
        overshooting_scale=0.2,
        imf_boundary_fraction=0.5,
        imf_time_mean=-0.4,
        imf_time_std=1.0,
        imf_adaptive_power=1.0,
        imf_adaptive_epsilon=0.01,
        imf_velocity_scale=1.0,
        imagination_horizon=3,
    )


def make_batch(config: DreamerConfig, *, batch: int = 2, time: int = 20):
    is_first = jnp.zeros((batch, time), dtype=jnp.bool_).at[:, 0].set(True)
    loss_mask = jnp.ones((batch, time), dtype=jnp.float32)
    loss_mask = loss_mask.at[:, : config.burn_in].set(0.0)
    return {
        "observations": jax.random.normal(
            jax.random.key(200), (batch, time, *config.observation_shape)
        ),
        "actions": jnp.tanh(
            jax.random.normal(
                jax.random.key(201), (batch, time, config.action_dim)
            )
        ),
        "rewards": jax.random.normal(jax.random.key(202), (batch, time)),
        "continuations": jnp.ones((batch, time), dtype=jnp.float32),
        "is_first": is_first,
        "loss_mask": loss_mask,
    }


class RecurrentContextTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("RECURRENT_CONTEXT_REPAIRS_OK")

    def test_replay_materializes_burn_in_flags_and_episode_safe_rows(self) -> None:
        replay = SequenceReplayBuffer(20, (1,), 1, seed=3)
        for episode_offset in (0.0, 100.0):
            for step in range(5):
                replay.append(
                    [episode_offset + step],
                    [step / 10.0],
                    episode_offset + step,
                    float(step < 4),
                    is_first=step == 0,
                )
        sampled = replay.sample(batch_size=32, sequence_length=3, burn_in=2)
        self.assertEqual(sampled["observations"].shape, (32, 5, 1))
        self.assertEqual(sampled["is_first"].dtype, jnp.bool_)
        np.testing.assert_array_equal(
            sampled["loss_mask"],
            np.tile(np.asarray([0, 0, 1, 1, 1], np.float32), (32, 1)),
        )
        self.assertTrue(bool(jnp.all(sampled["is_first"][:, 0])))
        self.assertFalse(bool(jnp.any(sampled["is_first"][:, 1:])))
        np.testing.assert_array_equal(
            np.diff(np.asarray(sampled["rewards"]), axis=1),
            np.ones((32, 4), dtype=np.float32),
        )

    def test_true_boundary_erases_prefix_state_and_boundary_action(self) -> None:
        config = repaired_config("gaussian")
        params = init_world_model(config, jax.random.key(203))
        first_observations = jax.random.normal(jax.random.key(204), (1, 5, 3))
        second_observations = first_observations.at[:, :2].add(1000.0)
        first_actions = jax.random.normal(jax.random.key(205), (1, 5, 2))
        second_actions = first_actions.at[:, :2].add(1000.0).at[:, 2].set(-999.0)
        is_first = jnp.asarray([[False, False, True, False, False]])

        first = observe_sequence(
            params,
            first_observations,
            first_actions,
            jax.random.key(206),
            config,
            is_first=is_first,
        )
        second = observe_sequence(
            params,
            second_observations,
            second_actions,
            jax.random.key(206),
            config,
            is_first=is_first,
        )
        np.testing.assert_allclose(
            first.states.deterministic[:, 2:], second.states.deterministic[:, 2:], atol=0.0
        )
        np.testing.assert_allclose(
            first.states.stochastic[:, 2:], second.states.stochastic[:, 2:], atol=0.0
        )

        # Positive-control mutation: without the boundary flag, the modified
        # prefix remains visible in the supposedly new episode.
        unreset = observe_sequence(
            params,
            second_observations,
            second_actions,
            jax.random.key(206),
            config,
        )
        self.assertGreater(
            float(
                jnp.max(
                    jnp.abs(first.states.deterministic[:, 2:] - unreset.states.deterministic[:, 2:])
                )
            ),
            1e-5,
        )

    def test_burn_in_targets_are_masked_but_context_remains_differentiable(self) -> None:
        config = repaired_config("gaussian")
        params = init_world_model(config, jax.random.key(207))
        batch = make_batch(config, batch=1, time=4)
        batch["loss_mask"] = jnp.asarray([[0.0, 0.0, 1.0, 1.0]])

        def objective(rewards):
            return world_model_loss(
                params,
                {**batch, "rewards": rewards},
                jax.random.key(208),
                config,
            ).total

        gradient = jax.grad(objective)(batch["rewards"])
        np.testing.assert_array_equal(gradient[:, :2], jnp.zeros((1, 2)))
        self.assertGreater(float(jnp.linalg.norm(gradient[:, 2:])), 0.0)

    def test_bounded_reward_head_starts_small_and_never_leaves_support(self) -> None:
        config = repaired_config("gaussian")
        config = DreamerConfig(
            **{
                **config.__dict__,
                "reward_min": 0.0,
                "reward_max": 1.0,
                "reward_initial_value": 0.01,
                "reward_output_init_scale": 0.0,
                "reward_loss": "binary_cross_entropy",
            }
        )
        params = init_world_model(config, jax.random.key(209))
        features = jax.random.normal(jax.random.key(210), (32, config.feature_dim))
        initial = predict_reward(params, features, config)
        np.testing.assert_allclose(initial, 0.01, rtol=2e-6, atol=2e-7)

        reward_layers = list(params["reward"]["layers"])
        for bias in (-100.0, 100.0):
            reward_layers[-1] = {
                **reward_layers[-1],
                "bias": jnp.asarray([bias], dtype=jnp.float32),
            }
            changed = {**params, "reward": {"layers": tuple(reward_layers)}}
            prediction = predict_reward(changed, features, config)
            self.assertTrue(bool(jnp.all(prediction >= 0.0)))
            self.assertTrue(bool(jnp.all(prediction <= 1.0)))
            self.assertTrue(np.isfinite(np.asarray(prediction)).all())
        logits = predict_reward_logits(params, features)
        self.assertTrue(np.isfinite(np.asarray(logits)).all())

    def test_fractional_binary_cross_entropy_reward_loss_is_finite(self) -> None:
        base = repaired_config("gaussian")
        config = DreamerConfig(
            **{
                **base.__dict__,
                "reward_min": 0.0,
                "reward_max": 1.0,
                "reward_initial_value": 0.01,
                "reward_output_init_scale": 0.0,
                "reward_loss": "binary_cross_entropy",
            }
        )
        batch = make_batch(config, batch=2, time=20)
        batch["rewards"] = jax.random.uniform(jax.random.key(211), (2, 20))
        params = init_world_model(config, jax.random.key(212))
        loss = world_model_loss(params, batch, jax.random.key(213), config).reward
        self.assertTrue(np.isfinite(float(loss)))
        self.assertGreater(float(loss), 0.0)

    def test_reward_configuration_rejects_partial_or_incompatible_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "both be set"):
            DreamerConfig(reward_min=0.0)
        with self.assertRaisesRegex(ValueError, "requires finite reward bounds"):
            DreamerConfig(reward_loss="binary_cross_entropy")
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            DreamerConfig(reward_min=0.0, reward_max=1.0, reward_initial_value=0.0)


class MultistepObjectiveTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("MULTISTEP_OBJECTIVE_REPAIRS_OK")

    def test_both_priors_have_finite_nonzero_exact_5_and_15_losses(self) -> None:
        for index, prior in enumerate(("gaussian", "imf")):
            with self.subTest(prior=prior):
                config = repaired_config(prior, burn_in=2)
                params = init_world_model(config, jax.random.key(210 + index))
                losses = world_model_loss(
                    params,
                    make_batch(config),
                    jax.random.key(212 + index),
                    config,
                )
                values = np.asarray(
                    [
                        losses.overshooting,
                        losses.overshooting_distance_5,
                        losses.overshooting_distance_15,
                    ]
                )
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue((values > 0.0).all())
                np.testing.assert_allclose(values[0], np.mean(values[1:]), rtol=1e-6)
                if prior == "imf":
                    self.assertGreater(float(losses.imf_loss_u), 0.0)
                    self.assertGreater(float(losses.imf_loss_v), 0.0)

    def test_multistep_posterior_targets_are_stopped_for_both_priors(self) -> None:
        for index, prior in enumerate(("gaussian", "imf")):
            with self.subTest(prior=prior):
                config = repaired_config(prior)
                params = init_world_model(config, jax.random.key(220 + index))
                batch = make_batch(config, batch=1, time=8)
                base = observe_sequence(
                    params,
                    batch["observations"],
                    batch["actions"],
                    jax.random.key(222 + index),
                    config,
                    is_first=batch["is_first"],
                )
                # Fix rollout start states independently of the variable target
                # posterior so this derivative isolates target leakage.
                fixed_states = RSSMState(
                    jax.lax.stop_gradient(base.states.deterministic),
                    jax.lax.stop_gradient(base.states.stochastic),
                )
                fixed_std = jax.lax.stop_gradient(base.posterior.std)

                def objective(target_mean):
                    sequence = SequenceStates(
                        fixed_states,
                        DiagonalNormal(target_mean, fixed_std),
                        base.prior_mean,
                        base.prior_std,
                    )
                    return _distance_consistency_loss(
                        params,
                        sequence,
                        batch["actions"],
                        batch["is_first"],
                        batch["loss_mask"],
                        jax.random.key(224 + index),
                        config,
                        5,
                    )

                gradient = jax.grad(objective)(base.posterior.mean)
                np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))

    def test_one_step_prior_target_is_stopped_and_prior_itself_trains(self) -> None:
        for index, prior in enumerate(("gaussian", "imf")):
            with self.subTest(prior=prior):
                config = repaired_config(prior)
                params = init_world_model(config, jax.random.key(230 + index))
                batch = make_batch(config, batch=2, time=1)

                def posterior_objective(posterior_params):
                    changed = dict(params)
                    changed["posterior"] = posterior_params
                    return world_model_loss(
                        changed, batch, jax.random.key(232 + index), config
                    ).prior

                posterior_gradient = jax.grad(posterior_objective)(params["posterior"])
                posterior_norm = sum(
                    float(jnp.sum(jnp.abs(leaf)))
                    for leaf in jax.tree_util.tree_leaves(posterior_gradient)
                )
                self.assertEqual(posterior_norm, 0.0)

                def prior_objective(prior_params):
                    changed = dict(params)
                    changed["prior"] = prior_params
                    return world_model_loss(
                        changed, batch, jax.random.key(232 + index), config
                    ).prior

                prior_gradient = jax.grad(prior_objective)(params["prior"])
                prior_norm = sum(
                    float(jnp.sum(jnp.abs(leaf)))
                    for leaf in jax.tree_util.tree_leaves(prior_gradient)
                )
                self.assertGreater(prior_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
