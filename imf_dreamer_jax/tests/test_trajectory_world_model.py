from __future__ import annotations

from dataclasses import replace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import imf_dreamer_jax.world_model as world_model_module
from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.world_model import (
    init_world_model,
    observe_sequence,
    transition_deterministic,
    trajectory_deterministic_conditions,
    world_model_loss,
)
from imf_dreamer_jax.types import RSSMState


def trajectory_config(**overrides: object) -> DreamerConfig:
    config = DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=5,
        stochastic_dim=3,
        embedding_dim=4,
        hidden_dim=7,
        prior="imf",
        imf_trajectory_enabled=True,
        imf_boundary_velocity_supervision=True,
        overshooting_horizon=1,
        overshooting_distances=(1,),
        imagination_horizon=3,
    )
    return replace(config, **overrides)


def make_batch(config: DreamerConfig, *, batch: int = 2, time: int = 6):
    return {
        "observations": jax.random.normal(
            jax.random.key(901), (batch, time, *config.observation_shape)
        ),
        "actions": jnp.tanh(
            jax.random.normal(
                jax.random.key(902), (batch, time, config.action_dim)
            )
        ),
        "rewards": jax.random.normal(jax.random.key(903), (batch, time)),
        "continuations": jnp.ones((batch, time), dtype=jnp.float32),
        "is_first": jnp.asarray(
            [[True, False, False, False, False, False]] * batch,
            dtype=jnp.bool_,
        ),
        "loss_mask": jnp.ones((batch, time), dtype=jnp.float32),
    }


class TrajectoryConfigurationTests(unittest.TestCase):
    def test_trajectory_mode_is_opt_in_and_method_name_is_distinct(self) -> None:
        legacy = DreamerConfig(
            observation_shape=(3,),
            action_dim=2,
            deterministic_dim=5,
            stochastic_dim=3,
            embedding_dim=4,
            hidden_dim=7,
            prior="imf",
        )
        trajectory = replace(legacy, imf_trajectory_enabled=True)
        self.assertFalse(legacy.imf_trajectory_enabled)
        self.assertEqual(legacy.method_name, "imf_rssm_one_step")
        self.assertEqual(trajectory.method_name, "trajectory_imf_rssm")

        legacy_params = init_world_model(legacy, jax.random.key(904))
        trajectory_params = init_world_model(trajectory, jax.random.key(904))
        self.assertEqual(
            legacy_params["recurrence"]["input"]["weight"].shape[0],
            legacy.stochastic_dim + legacy.action_dim,
        )
        self.assertEqual(
            trajectory_params["recurrence"]["input"]["weight"].shape[0],
            trajectory.stochastic_dim + trajectory.action_dim + 2,
        )

    def test_trajectory_configuration_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            DreamerConfig(imf_trajectory_enabled=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "requires prior='imf'"):
            DreamerConfig(prior="gaussian", imf_trajectory_enabled=True)
        with self.assertRaisesRegex(ValueError, "requires independent base noise"):
            DreamerConfig(
                prior="imf",
                imf_trajectory_enabled=True,
                imf_noise_coupling="posterior",
            )
        # The pre-existing one-token objective still supports this coupling.
        legacy_coupled = DreamerConfig(
            prior="imf",
            imf_trajectory_enabled=False,
            imf_noise_coupling="posterior",
        )
        self.assertEqual(legacy_coupled.imf_noise_coupling, "posterior")
        for name in (
            "imf_trajectory_clean_probability",
            "imf_trajectory_corrupted_probability",
            "imf_trajectory_suffix_probability",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    DreamerConfig(**{name: -0.1})
        with self.assertRaisesRegex(ValueError, "positive mass"):
            DreamerConfig(
                imf_trajectory_clean_probability=0.0,
                imf_trajectory_corrupted_probability=0.0,
                imf_trajectory_suffix_probability=0.0,
            )
        for value in (-0.01, 1.01):
            with self.subTest(history_noise_max=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                    DreamerConfig(imf_trajectory_history_noise_max=value)


class TrajectoryConditionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = trajectory_config()
        self.params = init_world_model(self.config, jax.random.key(910))
        self.batch = make_batch(self.config, batch=1)

    def conditions(
        self,
        history: jax.Array,
        times: jax.Array,
        *,
        actions: jax.Array | None = None,
        is_first: jax.Array | None = None,
    ) -> jax.Array:
        return trajectory_deterministic_conditions(
            self.params,
            history,
            times,
            self.batch["actions"] if actions is None else actions,
            self.batch["is_first"] if is_first is None else is_first,
            self.config,
        )

    def test_clean_history_conditions_equal_observed_deterministic_sequence(self) -> None:
        sequence = observe_sequence(
            self.params,
            self.batch["observations"],
            self.batch["actions"],
            jax.random.key(911),
            self.config,
            is_first=self.batch["is_first"],
        )
        clean = self.conditions(
            sequence.states.stochastic,
            jnp.zeros((1, 6, 1), dtype=jnp.float32),
        )
        np.testing.assert_allclose(
            clean, sequence.states.deterministic, rtol=1e-6, atol=1e-6
        )

        state = RSSMState(
            sequence.states.deterministic[:, 2],
            sequence.states.stochastic[:, 2],
        )
        implicit_clean = transition_deterministic(
            self.params, state, self.batch["actions"][:, 3], self.config
        )
        explicit_clean = transition_deterministic(
            self.params,
            state,
            self.batch["actions"][:, 3],
            self.config,
            stochastic_time=jnp.zeros((1, 1), dtype=jnp.float32),
        )
        np.testing.assert_array_equal(implicit_clean, explicit_clean)

    def test_conditions_are_strictly_causal_and_noise_times_are_not_dropped(self) -> None:
        history = jax.random.normal(jax.random.key(912), (1, 6, 3))
        zero_times = jnp.zeros((1, 6, 1), dtype=jnp.float32)
        baseline = self.conditions(history, zero_times)

        changed_history = history.at[:, 3].add(25.0)
        history_result = self.conditions(changed_history, zero_times)
        np.testing.assert_array_equal(history_result[:, :4], baseline[:, :4])
        self.assertGreater(
            float(jnp.max(jnp.abs(history_result[:, 4:] - baseline[:, 4:]))),
            1e-6,
        )

        changed_times = zero_times.at[:, 2, 0].set(0.9)
        time_result = self.conditions(history, changed_times)
        np.testing.assert_array_equal(time_result[:, :3], baseline[:, :3])
        self.assertGreater(
            float(jnp.max(jnp.abs(time_result[:, 3:] - baseline[:, 3:]))),
            1e-7,
        )

    def test_episode_reset_erases_history_time_and_boundary_action(self) -> None:
        history = jax.random.normal(jax.random.key(913), (1, 6, 3))
        times = jax.random.uniform(jax.random.key(914), (1, 6, 1))
        actions = self.batch["actions"]
        is_first = jnp.asarray([[True, False, False, True, False, False]])
        baseline = self.conditions(
            history, times, actions=actions, is_first=is_first
        )

        changed_history = history.at[:, :3].add(1000.0)
        changed_times = times.at[:, :3].set(1.0)
        changed_actions = actions.at[:, :3].add(1000.0).at[:, 3].set(-999.0)
        reset_result = self.conditions(
            changed_history,
            changed_times,
            actions=changed_actions,
            is_first=is_first,
        )
        np.testing.assert_allclose(
            reset_result[:, 3:], baseline[:, 3:], rtol=0.0, atol=0.0
        )


class TrajectoryObjectiveTests(unittest.TestCase):
    def test_query_and_history_corruption_share_each_tokens_noise_path(self) -> None:
        config = trajectory_config(
            imf_trajectory_clean_probability=0.0,
            imf_trajectory_corrupted_probability=1.0,
            imf_trajectory_suffix_probability=0.0,
            # This switch belongs to legacy scalar iMF.  Trajectory iMF must
            # still supervise the boundary head used by its JVP tangent.
            imf_boundary_velocity_supervision=False,
        )
        params = init_world_model(config, jax.random.key(915))
        seen: dict[str, object] = {}
        real_corrupt = world_model_module.corrupt_trajectory
        real_loss = world_model_module.trajectory_imf_loss

        def record_corrupt(targets, noise, times):
            seen["history_noise"] = noise
            return real_corrupt(targets, noise, times)

        def record_loss(*args, **kwargs):
            seen["query_noise"] = kwargs["noise"]
            seen["boundary_velocity_supervision"] = kwargs[
                "boundary_velocity_supervision"
            ]
            return real_loss(*args, **kwargs)

        with (
            mock.patch.object(
                world_model_module, "corrupt_trajectory", side_effect=record_corrupt
            ),
            mock.patch.object(
                world_model_module, "trajectory_imf_loss", side_effect=record_loss
            ),
        ):
            world_model_module.world_model_loss(
                params, make_batch(config), jax.random.key(916), config
            )
        np.testing.assert_array_equal(seen["history_noise"], seen["query_noise"])
        self.assertIs(seen["boundary_velocity_supervision"], True)

    def test_jitted_loss_and_gradients_are_finite_for_all_schedule_regimes(self) -> None:
        regimes = {
            "clean": (1.0, 0.0, 0.0),
            "corrupted": (0.0, 1.0, 0.0),
            "suffix": (0.0, 0.0, 1.0),
        }
        for index, (name, probabilities) in enumerate(regimes.items()):
            with self.subTest(regime=name):
                config = trajectory_config(
                    imf_trajectory_clean_probability=probabilities[0],
                    imf_trajectory_corrupted_probability=probabilities[1],
                    imf_trajectory_suffix_probability=probabilities[2],
                )
                params = init_world_model(config, jax.random.key(920 + index))
                batch = make_batch(config)

                def objective(model_params):
                    return world_model_loss(
                        model_params,
                        batch,
                        jax.random.key(930 + index),
                        config,
                    ).total

                value, gradients = jax.jit(jax.value_and_grad(objective))(params)
                self.assertTrue(np.isfinite(float(value)))
                gradient_leaves = jax.tree_util.tree_leaves(gradients)
                self.assertTrue(
                    all(np.isfinite(np.asarray(leaf)).all() for leaf in gradient_leaves)
                )
                prior_norm = sum(
                    float(jnp.sum(jnp.abs(leaf)))
                    for leaf in jax.tree_util.tree_leaves(gradients["prior"])
                )
                self.assertGreater(prior_norm, 0.0)

    def test_shortcut_endpoint_and_overshooting_are_absent_not_merely_downweighted(self) -> None:
        config = trajectory_config(
            imf_endpoint_scale=91.0,
            imf_shortcut_scale=73.0,
            overshooting_horizon=5,
            overshooting_distances=(2, 3, 5),
            overshooting_scale=57.0,
        )
        params = init_world_model(config, jax.random.key(940))
        losses = world_model_loss(
            params, make_batch(config), jax.random.key(941), config
        )
        for value in (
            losses.imf_endpoint,
            losses.imf_shortcut,
            losses.overshooting,
            losses.overshooting_distance_5,
            losses.overshooting_distance_15,
        ):
            self.assertEqual(float(value), 0.0)
        expected_total = (
            config.reconstruction_scale * losses.reconstruction
            + config.reward_scale * losses.reward
            + config.continuation_scale * losses.continuation
            + config.prior_scale * losses.prior
            + config.representation_scale * losses.representation
        )
        np.testing.assert_allclose(losses.total, expected_total, rtol=1e-6, atol=1e-6)
        self.assertGreater(float(losses.imf_loss_u), 0.0)
        self.assertGreater(float(losses.imf_loss_v), 0.0)


if __name__ == "__main__":
    unittest.main()
