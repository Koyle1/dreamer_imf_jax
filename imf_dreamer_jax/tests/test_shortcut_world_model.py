from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imf_dreamer_jax.world_model as world_model_module
from imf_dreamer_jax.agent import (
    create_agent,
    imagine,
    jit_train_actor_critic,
    train_world_model,
)
from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.shortcut import shortcut_forcing_loss
from imf_dreamer_jax.world_model import (
    init_world_model,
    initial_state,
    observe_sequence,
    prior_sampling_nfe,
    sample_prior,
    sample_prior_with_nfe,
    shortcut_deterministic_conditions,
    transition_deterministic,
    world_model_loss,
    world_model_parameter_counts,
)


def shortcut_config(**overrides: object) -> DreamerConfig:
    config = DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=5,
        stochastic_dim=3,
        embedding_dim=4,
        hidden_dim=7,
        prior="shortcut",
        shortcut_training_k_max=4,
        shortcut_sampling_steps=4,
        overshooting_horizon=1,
        overshooting_distances=(1,),
        imagination_horizon=3,
    )
    return replace(config, **overrides)


def make_batch(config: DreamerConfig, *, batch: int = 2, time: int = 6):
    first = jnp.zeros((batch, time), dtype=jnp.bool_)
    first = first.at[:, 0].set(True)
    if time >= 5:
        first = first.at[:, 3].set(True)
    return {
        "observations": jax.random.normal(
            jax.random.key(1201), (batch, time, *config.observation_shape)
        ),
        "actions": jnp.tanh(
            jax.random.normal(
                jax.random.key(1202), (batch, time, config.action_dim)
            )
        ),
        "rewards": jax.random.normal(jax.random.key(1203), (batch, time)),
        "continuations": jnp.ones((batch, time), dtype=jnp.float32),
        "is_first": first,
        "loss_mask": jnp.ones((batch, time), dtype=jnp.float32),
    }


class ShortcutConfigurationTests(unittest.TestCase):
    def test_mode_is_opt_in_and_legacy_recurrences_are_unchanged(self) -> None:
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
        shortcut = shortcut_config()
        self.assertEqual(legacy.method_name, "imf_rssm_one_step")
        self.assertEqual(trajectory.method_name, "trajectory_imf_rssm")
        self.assertEqual(shortcut.method_name, "shortcut_forcing_rssm")
        self.assertIsNone(legacy.shortcut_training_k_max)

        legacy_params = init_world_model(legacy, jax.random.key(1204))
        trajectory_params = init_world_model(trajectory, jax.random.key(1204))
        shortcut_params = init_world_model(shortcut, jax.random.key(1204))
        self.assertEqual(
            legacy_params["recurrence"]["input"]["weight"].shape[0],
            legacy.stochastic_dim + legacy.action_dim,
        )
        self.assertEqual(
            trajectory_params["recurrence"]["input"]["weight"].shape[0],
            trajectory.stochastic_dim + trajectory.action_dim + 2,
        )
        self.assertEqual(
            shortcut_params["recurrence"]["input"]["weight"].shape[0],
            shortcut.stochastic_dim + shortcut.action_dim + 2,
        )
        trajectory_leaves, trajectory_tree = jax.tree_util.tree_flatten(
            trajectory_params["recurrence"]
        )
        shortcut_leaves, shortcut_tree = jax.tree_util.tree_flatten(
            shortcut_params["recurrence"]
        )
        self.assertEqual(trajectory_tree, shortcut_tree)
        for trajectory_value, shortcut_value in zip(
            trajectory_leaves, shortcut_leaves, strict=True
        ):
            np.testing.assert_array_equal(trajectory_value, shortcut_value)

    def test_training_k_max_is_mandatory_and_schedule_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit shortcut_training_k_max"):
            DreamerConfig(prior="shortcut")
        with self.assertRaisesRegex(ValueError, "power of two"):
            DreamerConfig(prior="shortcut", shortcut_training_k_max=3)
        with self.assertRaisesRegex(ValueError, "power of two"):
            DreamerConfig(
                prior="shortcut",
                shortcut_training_k_max=4,
                shortcut_sampling_steps=3,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            DreamerConfig(
                prior="shortcut",
                shortcut_training_k_max=4,
                shortcut_sampling_steps=8,
            )
        with self.assertRaisesRegex(ValueError, "only when prior='shortcut'"):
            DreamerConfig(prior="imf", shortcut_training_k_max=4)
        with self.assertRaisesRegex(ValueError, "shortcut_sampling_clip"):
            shortcut_config(shortcut_sampling_clip=0.0)
        with self.assertRaisesRegex(ValueError, "shortcut_sampling_clip"):
            shortcut_config(shortcut_sampling_clip=float("inf"))
        with self.assertRaisesRegex(ValueError, "shortcut_intermediate_clip"):
            shortcut_config(shortcut_intermediate_clip=0.0)
        with self.assertRaisesRegex(ValueError, "shortcut_bootstrap_ema_decay"):
            shortcut_config(shortcut_bootstrap_ema_decay=1.0)

    def test_signal_time_and_step_size_must_be_supplied_as_a_pair(self) -> None:
        config = shortcut_config()
        params = init_world_model(config, jax.random.key(1205))
        state = initial_state(config, 2)
        actions = jnp.zeros((2, config.action_dim), dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            transition_deterministic(
                params,
                state,
                actions,
                config,
                stochastic_time=jnp.ones((2, 1)),
            )


class ShortcutContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = shortcut_config()
        self.params = init_world_model(self.config, jax.random.key(1210))
        self.batch = make_batch(self.config, batch=1)

    def conditions(self, z, tau, step, *, actions=None, is_first=None):
        return shortcut_deterministic_conditions(
            self.params,
            z,
            tau,
            step,
            self.batch["actions"] if actions is None else actions,
            self.batch["is_first"] if is_first is None else is_first,
            self.config,
        )

    def test_clean_sequence_matches_observation_recurrence(self) -> None:
        sequence = observe_sequence(
            self.params,
            self.batch["observations"],
            self.batch["actions"],
            jax.random.key(1211),
            self.config,
            is_first=self.batch["is_first"],
        )
        clean_conditions = self.conditions(
            sequence.states.stochastic,
            jnp.ones((1, 6, 1), dtype=jnp.float32),
            jnp.zeros((1, 6, 1), dtype=jnp.float32),
        )
        np.testing.assert_allclose(
            clean_conditions, sequence.states.deterministic, rtol=0.0, atol=0.0
        )

    def test_latent_time_and_step_are_strictly_causal_and_retained(self) -> None:
        z = jax.random.normal(jax.random.key(1212), (1, 6, 3))
        tau = jax.random.uniform(jax.random.key(1213), (1, 6, 1))
        step = 0.05 + 0.2 * jax.random.uniform(jax.random.key(1214), (1, 6, 1))
        no_resets = jnp.asarray([[True, False, False, False, False, False]])
        baseline = self.conditions(z, tau, step, is_first=no_resets)

        changed_z = self.conditions(
            z.at[:, 2].add(50.0), tau, step, is_first=no_resets
        )
        changed_tau = self.conditions(
            z, tau.at[:, 2, 0].set(0.99), step, is_first=no_resets
        )
        changed_step = self.conditions(
            z, tau, step.at[:, 2, 0].set(0.49), is_first=no_resets
        )
        for changed in (changed_z, changed_tau, changed_step):
            np.testing.assert_array_equal(changed[:, :3], baseline[:, :3])
            self.assertGreater(
                float(jnp.max(jnp.abs(changed[:, 3:] - baseline[:, 3:]))),
                1e-7,
            )

    def test_episode_reset_erases_prefix_coordinates_and_boundary_action(self) -> None:
        z = jax.random.normal(jax.random.key(1215), (1, 6, 3))
        tau = jax.random.uniform(jax.random.key(1216), (1, 6, 1))
        step = 0.1 + jax.random.uniform(jax.random.key(1217), (1, 6, 1)) / 4
        actions = self.batch["actions"]
        is_first = jnp.asarray([[True, False, False, True, False, False]])
        baseline = self.conditions(
            z, tau, step, actions=actions, is_first=is_first
        )

        changed = self.conditions(
            z.at[:, :3].add(1000.0),
            tau.at[:, :3].set(0.0),
            step.at[:, :3].set(1.0),
            actions=actions.at[:, :3].add(1000.0).at[:, 3].set(-999.0),
            is_first=is_first,
        )
        np.testing.assert_allclose(changed[:, 3:], baseline[:, 3:], rtol=0.0, atol=0.0)

    def test_equation_seven_rebuilds_context_for_z_prime_and_midpoint(self) -> None:
        config = replace(self.config, shortcut_support_safe_bootstrap=False)
        real_conditions = world_model_module.shortcut_deterministic_conditions
        calls: list[tuple[jax.Array, jax.Array, jax.Array]] = []

        def record(params, z, tau, step, actions, is_first, config):
            calls.append((z, tau, step))
            return real_conditions(params, z, tau, step, actions, is_first, config)

        with mock.patch.object(
            world_model_module,
            "shortcut_deterministic_conditions",
            side_effect=record,
        ):
            world_model_loss(
                self.params, self.batch, jax.random.key(1218), config
            )

        self.assertEqual(len(calls), 3)
        primal_z, primal_tau, primal_step = calls[0]
        first_z, first_tau, half_step = calls[1]
        second_z, midpoint_tau, second_step = calls[2]
        np.testing.assert_array_equal(first_z, primal_z)
        np.testing.assert_array_equal(first_tau, primal_tau)
        np.testing.assert_allclose(half_step, primal_step / 2.0, atol=0.0)
        np.testing.assert_allclose(midpoint_tau, primal_tau + half_step, atol=0.0)
        np.testing.assert_array_equal(second_step, half_step)
        self.assertGreater(float(jnp.max(jnp.abs(second_z - primal_z))), 1e-6)

        rebuilt = real_conditions(
            self.params,
            second_z,
            midpoint_tau,
            second_step,
            self.batch["actions"],
            self.batch["is_first"],
            config,
        )
        stale = real_conditions(
            self.params,
            primal_z,
            primal_tau,
            second_step,
            self.batch["actions"],
            self.batch["is_first"],
            config,
        )
        self.assertGreater(float(jnp.max(jnp.abs(rebuilt - stale))), 1e-6)


class ShortcutStabilityRepairTests(unittest.TestCase):
    def test_ema_teacher_updates_after_student(self) -> None:
        config = shortcut_config(shortcut_bootstrap_ema_decay=0.9)
        state = create_agent(config, jax.random.key(1240))
        self.assertIsNotNone(state.world_model_teacher)
        old_teacher = state.world_model_teacher
        updated, _ = train_world_model(
            state, make_batch(config), jax.random.key(1241), config
        )
        expected = jax.tree_util.tree_map(
            lambda old, new: 0.9 * old + 0.1 * new,
            old_teacher,
            updated.params.world_model,
        )
        for actual, wanted in zip(
            jax.tree_util.tree_leaves(updated.world_model_teacher),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_allclose(actual, wanted, rtol=2e-6, atol=2e-6)
        self.assertGreater(
            sum(
                float(jnp.sum(jnp.abs(new - old)))
                for new, old in zip(
                    jax.tree_util.tree_leaves(updated.params.world_model),
                    jax.tree_util.tree_leaves(old_teacher),
                    strict=True,
                )
            ),
            0.0,
        )

    def test_bootstrap_intermediate_is_bounded(self) -> None:
        targets = jnp.zeros((1, 3, 2), dtype=jnp.float32)
        noise = jnp.ones_like(targets)

        def explosive_teacher(z, tau, step):
            del tau, step
            return z + 1e6

        details = shortcut_forcing_loss(
            lambda z, tau, step: z,
            targets,
            k_max=4,
            teacher_predict_clean=explosive_teacher,
            intermediate_clip=4.0,
            noise=noise,
            tau=jnp.zeros((1, 3, 1), dtype=jnp.float32),
            step_size=jnp.ones((1, 3, 1), dtype=jnp.float32),
            return_details=True,
        )
        self.assertLessEqual(float(jnp.max(jnp.abs(details.intermediate))), 4.0)

    def test_stable_x_space_bootstrap_matches_velocity_form(self) -> None:
        targets = jax.random.normal(jax.random.key(1242), (2, 4, 3))
        noise = jax.random.normal(jax.random.key(1243), targets.shape)
        tau = jnp.full((2, 4, 1), 0.5, dtype=jnp.float32)
        step = jnp.full((2, 4, 1), 0.5, dtype=jnp.float32)

        def predictor(z, signal_time, step_size):
            return 0.7 * z + 0.2 * signal_time - 0.3 * step_size

        details = shortcut_forcing_loss(
            predictor,
            targets,
            k_max=4,
            support_safe_bootstrap=False,
            noise=noise,
            tau=tau,
            step_size=step,
            reduction="none",
            return_details=True,
        )
        velocity_form = jnp.square(1.0 - tau[..., 0]) * jnp.sum(
            jnp.square(
                details.prediction_velocity - details.bootstrap_target_velocity
            ),
            axis=-1,
        )
        np.testing.assert_allclose(
            details.bootstrap_loss, velocity_form, rtol=2e-6, atol=2e-6
        )

    def test_finest_tokens_never_request_below_minimum_step(self) -> None:
        requested_steps: list[np.ndarray] = []

        def teacher(z, tau, step):
            del tau
            requested_steps.append(np.asarray(step))
            return z

        details = shortcut_forcing_loss(
            lambda z, tau, step: z,
            jnp.zeros((1, 4, 2), dtype=jnp.float32),
            k_max=4,
            teacher_predict_clean=teacher,
            noise=jnp.ones((1, 4, 2), dtype=jnp.float32),
            tau=jnp.asarray([[[0.0], [0.5], [0.0], [0.75]]]),
            step_size=jnp.asarray([[[1.0], [0.5], [0.25], [0.25]]]),
            return_details=True,
        )
        self.assertEqual(len(requested_steps), 2)
        for step in requested_steps:
            self.assertGreaterEqual(float(step.min()), 0.25)
        finest = np.asarray(details.is_finest[..., 0])
        np.testing.assert_array_equal(
            np.asarray(details.intermediate)[finest],
            np.asarray(details.corrupted)[finest],
        )


class ShortcutObjectiveAndSamplingTests(unittest.TestCase):
    def test_sampling_clip_prevents_recursive_latent_explosion(self) -> None:
        noise = jnp.zeros((2, 1, 3), dtype=jnp.float32)

        def explosive_predict_clean(state, tau, step_size):
            del tau, step_size
            return jnp.full_like(state, 1e20)

        unbounded = world_model_module.sample_shortcut_steps(
            explosive_predict_clean, noise, steps=4
        )
        bounded = world_model_module.sample_shortcut_steps(
            explosive_predict_clean,
            noise,
            steps=4,
            clean_prediction_clip=10.0,
        )
        self.assertGreater(float(jnp.max(jnp.abs(unbounded))), 1e19)
        np.testing.assert_array_equal(bounded, jnp.full_like(bounded, 10.0))
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            world_model_module.sample_shortcut_steps(
                explosive_predict_clean,
                noise,
                steps=4,
                clean_prediction_clip=float("inf"),
            )

    def test_clipped_shortcut_imagination_keeps_actor_update_finite(self) -> None:
        config = shortcut_config(
            shortcut_sampling_clip=10.0,
            imagination_horizon=15,
            reward_output_init_scale=1.0,
        )
        state = create_agent(config, jax.random.key(1230))
        prior = state.params.world_model["prior"]
        layers = list(prior["layers"])
        layers[-1] = {
            "weight": jnp.zeros_like(layers[-1]["weight"]),
            "bias": jnp.full_like(layers[-1]["bias"], 1e20),
        }
        model = {
            **state.params.world_model,
            "prior": {**prior, "layers": tuple(layers)},
        }
        state = state._replace(
            params=state.params._replace(world_model=model)
        )
        start = initial_state(config, 8)
        imagined = imagine(
            state.params, start, jax.random.key(1231), config
        )
        self.assertTrue(
            all(
                np.isfinite(np.asarray(getattr(imagined, field))).all()
                for field in imagined._fields
            )
        )
        self.assertLessEqual(
            float(jnp.max(jnp.abs(imagined.features))),
            config.shortcut_sampling_clip,
        )
        updated, metrics = jit_train_actor_critic(
            state, start, jax.random.key(1232), config
        )
        self.assertTrue(np.isfinite(np.asarray(metrics)).all())
        for tree in (
            updated.params.actor,
            updated.params.critic,
            updated.actor_optimizer,
            updated.critic_optimizer,
        ):
            self.assertTrue(
                all(
                    np.isfinite(np.asarray(leaf)).all()
                    for leaf in jax.tree_util.tree_leaves(tree)
                )
            )

    def test_jitted_objective_and_gradients_are_finite_and_prior_is_exact_branch(self) -> None:
        config = shortcut_config()
        params = init_world_model(config, jax.random.key(1220))
        batch = make_batch(config)

        def objective(model_params):
            details = world_model_loss(
                model_params, batch, jax.random.key(1221), config
            )
            return details.total, details

        (value, details), gradients = jax.jit(
            jax.value_and_grad(objective, has_aux=True)
        )(params)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(
            all(
                np.isfinite(np.asarray(leaf)).all()
                for leaf in jax.tree_util.tree_leaves(gradients)
            )
        )
        prior_norm = sum(
            float(jnp.sum(jnp.abs(leaf)))
            for leaf in jax.tree_util.tree_leaves(gradients["prior"])
        )
        recurrence_norm = sum(
            float(jnp.sum(jnp.abs(leaf)))
            for leaf in jax.tree_util.tree_leaves(gradients["recurrence"])
        )
        self.assertGreater(prior_norm, 0.0)
        self.assertGreater(recurrence_norm, 0.0)
        np.testing.assert_allclose(details.imf_shortcut, details.prior, rtol=0.0, atol=0.0)
        for absent in (
            details.imf_loss_u,
            details.imf_loss_v,
            details.imf_endpoint,
            details.overshooting,
            details.overshooting_distance_5,
            details.overshooting_distance_15,
        ):
            self.assertEqual(float(absent), 0.0)

    def test_shortcut_target_is_stopped_but_prior_parameters_train(self) -> None:
        config = shortcut_config()
        params = init_world_model(config, jax.random.key(1222))
        batch = make_batch(config, batch=1)

        def posterior_objective(posterior_params):
            changed = dict(params)
            changed["posterior"] = posterior_params
            return world_model_loss(
                changed, batch, jax.random.key(1223), config
            ).prior

        posterior_gradient = jax.grad(posterior_objective)(params["posterior"])
        posterior_norm = sum(
            float(jnp.sum(jnp.abs(leaf)))
            for leaf in jax.tree_util.tree_leaves(posterior_gradient)
        )
        self.assertEqual(posterior_norm, 0.0)

    def test_sampling_uses_configured_k_steps_and_reports_exact_nfe(self) -> None:
        config = shortcut_config(shortcut_sampling_steps=4)
        params = init_world_model(config, jax.random.key(1224))
        condition = jax.random.normal(
            jax.random.key(1225), (2, config.deterministic_dim)
        )
        noise = jax.random.normal(
            jax.random.key(1226), (2, config.stochastic_dim)
        )
        real_sampler = world_model_module.sample_shortcut_steps
        with mock.patch.object(
            world_model_module, "sample_shortcut_steps", wraps=real_sampler
        ) as wrapped:
            result = sample_prior_with_nfe(
                params, condition, None, config, noise=noise
            )
        self.assertEqual(wrapped.call_count, 1)
        self.assertEqual(wrapped.call_args.kwargs["steps"], 4)
        self.assertEqual(
            wrapped.call_args.kwargs["clean_prediction_clip"],
            config.shortcut_sampling_clip,
        )
        self.assertEqual(int(result.nfe), 4)
        self.assertEqual(prior_sampling_nfe(config), 4)
        self.assertEqual(prior_sampling_nfe(config, transitions=3), 12)
        self.assertEqual(result.stochastic.shape, noise.shape)

        one_step, _ = sample_prior(
            params,
            condition,
            None,
            replace(config, shortcut_sampling_steps=1),
            noise=noise,
        )
        four_step, _ = sample_prior(
            params, condition, None, config, noise=noise
        )
        self.assertGreater(float(jnp.max(jnp.abs(one_step - four_step))), 1e-6)

    def test_imagination_routes_through_configured_shortcut_sampler(self) -> None:
        config = shortcut_config(shortcut_sampling_steps=4, imagination_horizon=3)
        agent_state = create_agent(config, jax.random.key(1227))
        start = initial_state(config, 2)
        real_sampler = world_model_module.sample_shortcut_steps
        with mock.patch.object(
            world_model_module, "sample_shortcut_steps", wraps=real_sampler
        ) as wrapped:
            imagined = imagine(
                agent_state.params,
                start,
                jax.random.key(1228),
                config,
                horizon=3,
            )
        self.assertEqual(wrapped.call_count, 1)
        self.assertEqual(wrapped.call_args.kwargs["steps"], 4)
        self.assertEqual(imagined.actions.shape, (2, 3, config.action_dim))
        self.assertEqual(prior_sampling_nfe(config, transitions=3), 12)

    def test_parameter_count_report_is_complete_and_mode_sensitive(self) -> None:
        shortcut = shortcut_config()
        imf = replace(
            shortcut,
            prior="imf",
            shortcut_training_k_max=None,
            imf_trajectory_enabled=True,
        )
        shortcut_counts = world_model_parameter_counts(
            init_world_model(shortcut, jax.random.key(1229)), shortcut
        )
        imf_counts = world_model_parameter_counts(
            init_world_model(imf, jax.random.key(1229)), imf
        )
        self.assertEqual(shortcut_counts.active, shortcut_counts.total)
        self.assertEqual(shortcut_counts.prior_active, shortcut_counts.prior_total)
        self.assertGreater(shortcut_counts.total, 0)
        self.assertNotEqual(shortcut_counts.total, imf_counts.total)


if __name__ == "__main__":
    unittest.main()
