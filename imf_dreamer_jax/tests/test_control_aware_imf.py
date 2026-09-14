from __future__ import annotations

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.control_aware_imf import (
    APPROXIMATE_DETERMINISTIC_DENSITY_LABEL,
    CVAML_COMPATIBLE_SQUARED_LABEL,
    DIRECT_ACTION_CHUNK_ENDPOINT_LABEL,
    EMPIRICAL_BEHAVIOR_REJECTION_LABEL,
    EXPLICIT_STOCHASTIC_DENSITY_LABEL,
    ActionChunkEndpointConfig,
    PlannerValueEquivalenceConfig,
    PolicyDensityTiltConfig,
    action_chunk_endpoint_imf_loss,
    approximate_fixed_variance_gaussian_log_probability,
    build_action_chunk_endpoint_batch,
    cvaml_compatible_bellman_residual_loss,
    diagonal_gaussian_log_probability,
    jit_action_chunk_endpoint_imf_loss,
    jit_build_action_chunk_endpoint_batch,
    jit_cvaml_compatible_bellman_residual_loss,
    jit_planner_matched_bellman_residual_loss,
    jit_policy_density_tilt_weights,
    planner_bellman_backup,
    planner_matched_bellman_residual_loss,
    policy_density_tilt_weights,
)
from imf_dreamer_jax.imf import (
    improved_meanflow_loss,
    init_imf,
    sample_imf_one_step,
)


class PolicyDensityTiltTests(unittest.TestCase):
    def test_explicit_diagonal_gaussian_log_probability(self) -> None:
        actions = jnp.asarray([[0.0, 2.0], [1.0, -1.0]], dtype=jnp.float32)
        mean = jnp.asarray([[0.0, 0.0], [0.5, -0.5]], dtype=jnp.float32)
        log_std = jnp.log(jnp.asarray([1.0, 2.0], dtype=jnp.float32))
        actual = diagonal_gaussian_log_probability(actions, mean, log_std)
        std = np.asarray([1.0, 2.0])
        expected = -0.5 * np.sum(
            np.square((np.asarray(actions) - np.asarray(mean)) / std)
            + 2.0 * np.log(std)
            + math.log(2.0 * math.pi),
            axis=-1,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_deterministic_gaussian_wrapper_is_numerically_explicitly_labeled(
        self,
    ) -> None:
        actions = jnp.asarray([[0.0, 0.5], [0.2, -0.1]], dtype=jnp.float32)
        deterministic = jnp.asarray([[0.1, 0.4], [0.0, 0.0]], dtype=jnp.float32)
        actual = approximate_fixed_variance_gaussian_log_probability(
            actions, deterministic, fixed_std=0.25
        )
        expected = diagonal_gaussian_log_probability(
            actions, deterministic, math.log(0.25)
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertIn("approximate", APPROXIMATE_DETERMINISTIC_DENSITY_LABEL)
        self.assertIn("explicit", EXPLICIT_STOCHASTIC_DENSITY_LABEL)

    def test_policy_tilt_matches_unclipped_itpo_formula(self) -> None:
        probabilities = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
        weights = policy_density_tilt_weights(
            jnp.log(probabilities),
            PolicyDensityTiltConfig(eta=1.0, maximum_weight=10.0),
        )
        expected = probabilities / jnp.mean(probabilities)
        np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(jnp.mean(weights), 1.0, rtol=0.0, atol=2e-6)

    def test_eta_zero_reduces_exactly_to_uniform_masked_weighting(self) -> None:
        details = policy_density_tilt_weights(
            jnp.asarray([jnp.nan, -3.0, -jnp.inf, 5.0]),
            PolicyDensityTiltConfig(eta=0.0),
            mask=jnp.asarray([1, 0, 1, 1]),
            return_details=True,
        )
        np.testing.assert_array_equal(
            details.weights, jnp.asarray([1.0, 0.0, 1.0, 1.0])
        )
        self.assertEqual(float(details.valid_count), 3.0)
        self.assertEqual(float(details.invalid_log_probability_count), 2.0)

    def test_extreme_scores_are_finite_bounded_and_mean_one(self) -> None:
        config = PolicyDensityTiltConfig(
            eta=3.0,
            log_weight_clip=12.0,
            minimum_weight=0.2,
            maximum_weight=1.5,
        )
        details = policy_density_tilt_weights(
            jnp.asarray([-1.0e30, -100.0, 0.0, 100.0, 1.0e30]),
            config,
            return_details=True,
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(details.weights))))
        self.assertGreaterEqual(float(jnp.min(details.weights)), 0.2 - 2e-6)
        self.assertLessEqual(float(jnp.max(details.weights)), 1.5 + 2e-6)
        np.testing.assert_allclose(jnp.mean(details.weights), 1.0, atol=3e-6)
        self.assertGreater(float(details.clipped_fraction), 0.0)
        self.assertGreaterEqual(float(details.effective_sample_size), 1.0)

    def test_nonfinite_positive_eta_entries_receive_zero_weight(self) -> None:
        details = policy_density_tilt_weights(
            jnp.asarray([0.0, jnp.nan, -jnp.inf]),
            PolicyDensityTiltConfig(eta=1.0),
            return_details=True,
        )
        np.testing.assert_array_equal(details.weights, jnp.asarray([1.0, 0.0, 0.0]))
        self.assertEqual(float(details.invalid_log_probability_count), 2.0)

    def test_jitted_tilt_matches_eager_path(self) -> None:
        log_probabilities = jnp.asarray([-3.0, -2.0, -1.0, 0.0])
        config = PolicyDensityTiltConfig(eta=0.5, maximum_weight=2.0)
        eager = policy_density_tilt_weights(log_probabilities, config)
        compiled = jit_policy_density_tilt_weights(log_probabilities, config)
        np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)

    def test_density_tilt_negative_validation(self) -> None:
        invalid_configs = (
            {"eta": -0.1},
            {"log_weight_clip": 0.0},
            {"minimum_weight": -0.1},
            {"minimum_weight": 1.1},
            {"maximum_weight": 0.9},
            {"eta": float("nan")},
        )
        for arguments in invalid_configs:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                PolicyDensityTiltConfig(**arguments)
        with self.assertRaises(ValueError):
            policy_density_tilt_weights(jnp.asarray(1.0))
        with self.assertRaises(ValueError):
            policy_density_tilt_weights(jnp.ones((2,)), mask=jnp.ones((2, 1)))
        with self.assertRaises(ValueError):
            diagonal_gaussian_log_probability(jnp.ones((2, 2)), jnp.ones((2, 1)), 0.0)
        with self.assertRaises(ValueError):
            approximate_fixed_variance_gaussian_log_probability(
                jnp.ones((2, 2)), jnp.ones((2, 2)), fixed_std=0.0
            )
        with self.assertRaises(ValueError):
            approximate_fixed_variance_gaussian_log_probability(
                jnp.ones((2, 2)), jnp.ones((2, 1)), fixed_std=0.5
            )


class PlannerMatchedBellmanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PlannerValueEquivalenceConfig(
            discount=0.5, normalization_epsilon=1e-6
        )
        self.model_reward = jnp.asarray([1.0, 2.0])
        self.model_continuation = jnp.asarray([0.5, 0.0])
        self.model_value = jnp.asarray([4.0, 9.0])
        self.real_reward = jnp.asarray([0.0, 1.0])
        self.real_continuation = jnp.asarray([1.0, 1.0])
        self.real_value = jnp.asarray([2.0, 0.0])

    def test_planner_backup_includes_continuation(self) -> None:
        backup = planner_bellman_backup(
            self.model_reward,
            self.model_continuation,
            self.model_value,
            discount=self.config.discount,
        )
        np.testing.assert_allclose(backup, jnp.asarray([2.0, 2.0]))

    def test_deterministic_loss_is_exact_normalized_squared_residual(self) -> None:
        details = planner_matched_bellman_residual_loss(
            self.model_reward,
            self.model_continuation,
            self.model_value,
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
            normalization_scale=2.0,
            return_details=True,
        )
        np.testing.assert_allclose(details.model_backup, jnp.asarray([2.0, 2.0]))
        np.testing.assert_allclose(details.real_target, jnp.asarray([1.0, 1.0]))
        np.testing.assert_allclose(details.normalized_residual, 0.5)
        np.testing.assert_allclose(details.per_example_loss, 0.25)
        np.testing.assert_allclose(details.loss, 0.25)
        np.testing.assert_allclose(details.variance_correction, 0.0)

    def test_masked_mean_uses_only_planner_selected_examples(self) -> None:
        loss = planner_matched_bellman_residual_loss(
            self.model_reward,
            self.model_continuation,
            self.model_value,
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
            mask=jnp.asarray([1.0, 0.0]),
        )
        self.assertEqual(float(loss), 1.0)
        none = planner_matched_bellman_residual_loss(
            self.model_reward,
            self.model_continuation,
            self.model_value,
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
            mask=jnp.asarray([1.0, 0.0]),
            reduction="none",
        )
        np.testing.assert_array_equal(none, jnp.asarray([1.0, 0.0]))

        # Weighted means are invariant to a common positive rescaling, even
        # when the total weight is below one.
        fractional = planner_matched_bellman_residual_loss(
            self.model_reward,
            self.model_continuation,
            self.model_value,
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
            mask=jnp.asarray([0.1, 0.0]),
        )
        self.assertEqual(float(fractional), 1.0)

    def test_real_bellman_target_is_gradient_stopped(self) -> None:
        def objective(real_reward, real_continuation, real_value):
            return planner_matched_bellman_residual_loss(
                self.model_reward,
                self.model_continuation,
                self.model_value,
                real_reward,
                real_continuation,
                real_value,
                self.config,
            )

        gradients = jax.grad(objective, argnums=(0, 1, 2))(
            self.real_reward, self.real_continuation, self.real_value
        )
        for gradient in gradients:
            np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))

    def test_cvaml_variance_subtraction_is_explicit_and_unclipped(self) -> None:
        zero_samples = jnp.zeros((2, 1), dtype=jnp.float32)
        details = cvaml_compatible_bellman_residual_loss(
            jnp.asarray([[1.0], [3.0]]),
            zero_samples,
            zero_samples,
            jnp.asarray([2.0]),
            jnp.asarray([0.0]),
            jnp.asarray([0.0]),
            PlannerValueEquivalenceConfig(discount=1.0),
            normalization_scale=2.0,
            return_details=True,
        )
        # Normalized samples are [0.5, 1.5], so the unbiased sample variance is
        # 0.5 and its K=2 sample-mean correction is 0.25.
        np.testing.assert_allclose(details.uncorrected_loss, 0.0, atol=1e-7)
        np.testing.assert_allclose(details.variance_correction, 0.25, atol=1e-7)
        np.testing.assert_allclose(details.loss, -0.25, atol=1e-7)
        self.assertIn("squared", CVAML_COMPATIBLE_SQUARED_LABEL)

    def test_cvaml_uses_unbiased_variance_of_sample_mean_correction(self) -> None:
        samples = jnp.asarray([[0.0, 3.0], [2.0, 1.0], [4.0, 5.0]])
        zeros = jnp.zeros_like(samples)
        targets = jnp.asarray([1.5, 2.0])
        details = cvaml_compatible_bellman_residual_loss(
            samples,
            zeros,
            zeros,
            targets,
            jnp.zeros_like(targets),
            jnp.zeros_like(targets),
            PlannerValueEquivalenceConfig(discount=1.0),
            reduction="none",
            return_details=True,
        )
        expected = jnp.square(jnp.mean(samples, axis=0) - targets) - (
            jnp.var(samples, axis=0, ddof=1) / samples.shape[0]
        )
        np.testing.assert_allclose(details.per_example_loss, expected, atol=1e-7)

    def test_identical_model_samples_reduce_to_deterministic_squared_loss(self) -> None:
        deterministic = planner_matched_bellman_residual_loss(
            self.model_reward,
            self.model_continuation,
            self.model_value,
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
            normalization_scale=2.0,
        )
        repeat = lambda value: jnp.repeat(value[None], 3, axis=0)
        stochastic = cvaml_compatible_bellman_residual_loss(
            repeat(self.model_reward),
            repeat(self.model_continuation),
            repeat(self.model_value),
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
            normalization_scale=2.0,
        )
        np.testing.assert_allclose(stochastic, deterministic, rtol=0.0, atol=1e-7)

    def test_cvaml_supports_nonleading_sample_axis(self) -> None:
        rewards = jnp.asarray([[1.0, 3.0], [2.0, 4.0]])
        zeros = jnp.zeros_like(rewards)
        details = cvaml_compatible_bellman_residual_loss(
            rewards,
            zeros,
            zeros,
            jnp.asarray([2.0, 3.0]),
            jnp.zeros((2,)),
            jnp.zeros((2,)),
            PlannerValueEquivalenceConfig(discount=1.0),
            sample_axis=1,
            return_details=True,
        )
        np.testing.assert_allclose(details.model_backup, jnp.asarray([2.0, 3.0]))
        np.testing.assert_allclose(details.loss, -1.0, atol=1e-7)

    def test_cvaml_real_target_is_gradient_stopped(self) -> None:
        samples = jnp.stack((self.model_reward, self.model_reward + 1.0))
        continuations = jnp.stack((self.model_continuation,) * 2)
        values = jnp.stack((self.model_value,) * 2)

        def objective(real_reward):
            return cvaml_compatible_bellman_residual_loss(
                samples,
                continuations,
                values,
                real_reward,
                self.real_continuation,
                self.real_value,
                self.config,
            )

        np.testing.assert_array_equal(
            jax.grad(objective)(self.real_reward), jnp.zeros_like(self.real_reward)
        )

    def test_jitted_bellman_paths_match_eager(self) -> None:
        arguments = (
            self.model_reward,
            self.model_continuation,
            self.model_value,
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
        )
        eager = planner_matched_bellman_residual_loss(*arguments)
        compiled = jit_planner_matched_bellman_residual_loss(*arguments)
        np.testing.assert_allclose(compiled, eager)

        repeat = lambda value: jnp.repeat(value[None], 2, axis=0)
        stochastic_arguments = (
            repeat(self.model_reward),
            repeat(self.model_continuation),
            repeat(self.model_value),
            self.real_reward,
            self.real_continuation,
            self.real_value,
            self.config,
        )
        eager_stochastic = cvaml_compatible_bellman_residual_loss(*stochastic_arguments)
        compiled_stochastic = jit_cvaml_compatible_bellman_residual_loss(
            *stochastic_arguments
        )
        np.testing.assert_allclose(compiled_stochastic, eager_stochastic)

    def test_bellman_negative_validation(self) -> None:
        for arguments in (
            {"discount": -0.1},
            {"discount": 1.1},
            {"normalization_epsilon": 0.0},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                PlannerValueEquivalenceConfig(**arguments)
        with self.assertRaises(ValueError):
            planner_bellman_backup(
                jnp.ones((2,)), jnp.ones((3,)), jnp.ones((2,)), discount=0.9
            )
        with self.assertRaises(ValueError):
            planner_matched_bellman_residual_loss(
                self.model_reward,
                self.model_continuation,
                self.model_value,
                self.real_reward,
                self.real_continuation,
                self.real_value,
                self.config,
                normalization_scale=-1.0,
            )
        with self.assertRaises(ValueError):
            planner_matched_bellman_residual_loss(
                self.model_reward,
                self.model_continuation,
                self.model_value,
                self.real_reward,
                self.real_continuation,
                self.real_value,
                self.config,
                reduction="median",  # type: ignore[arg-type]
            )
        singleton = self.model_reward[None]
        with self.assertRaisesRegex(ValueError, "at least two"):
            cvaml_compatible_bellman_residual_loss(
                singleton,
                self.model_continuation[None],
                self.model_value[None],
                self.real_reward,
                self.real_continuation,
                self.real_value,
                self.config,
            )
        with self.assertRaisesRegex(ValueError, "out of range"):
            cvaml_compatible_bellman_residual_loss(
                jnp.ones((2, 2)),
                jnp.ones((2, 2)),
                jnp.ones((2, 2)),
                jnp.ones((2,)),
                jnp.ones((2,)),
                jnp.ones((2,)),
                self.config,
                sample_axis=2,
            )
        with self.assertRaises(ValueError):
            cvaml_compatible_bellman_residual_loss(
                jnp.ones((2, 3)),
                jnp.ones((2, 3)),
                jnp.ones((2, 3)),
                jnp.ones((2,)),
                jnp.ones((2,)),
                jnp.ones((2,)),
                self.config,
            )


class ActionChunkEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = jnp.asarray(
            [[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]]]
        )
        self.actions = jnp.asarray([[[0.1], [0.2], [0.3]]])

    def test_one_step_batch_reduces_to_ordinary_transition_pairs(self) -> None:
        batch = build_action_chunk_endpoint_batch(
            self.states, self.actions, ActionChunkEndpointConfig(horizons=(1,))
        )
        expected_conditions = jnp.concatenate(
            (self.states[:, :-1], self.actions), axis=-1
        ).reshape((3, 3))
        expected_targets = self.states[:, 1:].reshape((3, 2))
        np.testing.assert_array_equal(batch.conditions, expected_conditions)
        np.testing.assert_array_equal(batch.targets, expected_targets)
        np.testing.assert_array_equal(batch.start_index, jnp.arange(3))
        np.testing.assert_array_equal(batch.horizon, jnp.ones((3,), dtype=jnp.int32))
        np.testing.assert_array_equal(batch.action_mask, jnp.ones((3, 1), dtype=bool))
        self.assertTrue(bool(jnp.all(batch.rejection.accepted)))

    def test_any_step_batch_contains_all_requested_direct_endpoints(self) -> None:
        batch = build_action_chunk_endpoint_batch(
            self.states,
            self.actions,
            ActionChunkEndpointConfig(horizons=(1, 2, 3)),
        )
        np.testing.assert_array_equal(batch.horizon, jnp.asarray([1, 1, 1, 2, 2, 3]))
        np.testing.assert_array_equal(
            batch.start_index, jnp.asarray([0, 1, 2, 0, 1, 0])
        )
        np.testing.assert_array_equal(
            batch.targets[:, 0], jnp.asarray([1.0, 2.0, 3.0, 2.0, 3.0, 3.0])
        )
        np.testing.assert_allclose(
            batch.action_chunks[3, :, 0], jnp.asarray([0.1, 0.2, 0.0])
        )
        np.testing.assert_array_equal(
            batch.action_mask[3], jnp.asarray([True, True, False])
        )
        # state(2) + padded actions(3) + action mask(3)
        self.assertEqual(batch.conditions.shape, (6, 8))
        self.assertIn("any_step", DIRECT_ACTION_CHUNK_ENDPOINT_LABEL)

    def test_behavior_rejection_metadata_is_length_aware_and_explicit(self) -> None:
        config = ActionChunkEndpointConfig(
            horizons=(1, 2),
            minimum_mean_behavior_log_density=-1.5,
            maximum_behavior_distance=0.5,
            behavior_density_is_approximate=True,
        )
        batch = build_action_chunk_endpoint_batch(
            self.states,
            self.actions,
            config,
            valid_transitions=jnp.asarray([[1, 0, 1]]),
            behavior_log_densities=jnp.asarray([[-1.0, -2.0, -1.0]]),
            behavior_distances=jnp.asarray([[0.1, 0.2, 0.8]]),
        )
        metadata = batch.rejection
        # Horizon-one rows precede horizon-two rows.
        np.testing.assert_allclose(metadata.mean_log_density[:3], [-1.0, -2.0, -1.0])
        np.testing.assert_allclose(metadata.joint_log_density[3:], [-3.0, -3.0])
        np.testing.assert_allclose(metadata.mean_log_density[3:], [-1.5, -1.5])
        np.testing.assert_allclose(metadata.maximum_distance[3:], [0.2, 0.8])
        np.testing.assert_array_equal(
            metadata.rejected_by_density[:3], [False, True, False]
        )
        np.testing.assert_array_equal(
            metadata.rejected_by_distance[:3], [False, False, True]
        )
        np.testing.assert_array_equal(
            metadata.rejected_by_invalid_transition[:3], [False, True, False]
        )
        self.assertTrue(bool(jnp.all(metadata.density_is_approximate)))
        self.assertIn("not_a_support_set", EMPIRICAL_BEHAVIOR_REJECTION_LABEL)

    def test_invalid_behavior_metadata_rejects_without_a_configured_threshold(
        self,
    ) -> None:
        batch = build_action_chunk_endpoint_batch(
            self.states,
            self.actions,
            ActionChunkEndpointConfig(horizons=(1, 2)),
            behavior_log_densities=jnp.asarray([[-1.0, jnp.nan, -1.0]]),
            behavior_distances=jnp.asarray([[0.1, -0.2, 0.3]]),
        )
        metadata = batch.rejection
        self.assertTrue(bool(metadata.rejected_by_density[1]))
        self.assertTrue(bool(metadata.rejected_by_distance[1]))
        # The length-two chunk from start zero contains both invalid entries.
        self.assertTrue(bool(metadata.rejected_by_density[3]))
        self.assertTrue(bool(metadata.rejected_by_distance[3]))

    def test_custom_start_conditions_replace_state_prefix_only(self) -> None:
        start_conditions = jnp.asarray([[[5.0], [6.0], [7.0]]])
        batch = build_action_chunk_endpoint_batch(
            self.states,
            self.actions,
            ActionChunkEndpointConfig(horizons=(1,)),
            start_conditions=start_conditions,
        )
        np.testing.assert_array_equal(batch.start_conditions, start_conditions[0])
        np.testing.assert_array_equal(
            batch.conditions,
            jnp.concatenate((start_conditions, self.actions), axis=-1)[0],
        )
        np.testing.assert_array_equal(batch.targets, self.states[0, 1:])

    def test_endpoint_loss_reduces_to_existing_ordinary_imf_endpoint_atom(self) -> None:
        batch = build_action_chunk_endpoint_batch(
            self.states, self.actions, ActionChunkEndpointConfig(horizons=(1,))
        )
        params = init_imf(
            jax.random.key(10), sample_dim=2, condition_dim=3, hidden_dim=7, depth=2
        )
        noise = jax.random.normal(jax.random.key(11), batch.targets.shape)
        actual = action_chunk_endpoint_imf_loss(
            params,
            batch,
            jax.random.key(12),
            noise=noise,
            return_details=True,
        )
        ordinary = improved_meanflow_loss(
            params,
            batch.targets,
            batch.conditions,
            jax.random.key(12),
            noise=noise,
            r=0.0,
            t=1.0,
            meanflow_scale=0.0,
            velocity_scale=0.0,
            endpoint_scale=1.0,
            weights=batch.rejection.accepted.astype(jnp.float32),
            boundary_velocity_supervision=True,
            return_details=True,
        )
        direct_prediction = sample_imf_one_step(params, batch.conditions, noise=noise)
        np.testing.assert_allclose(
            actual.prediction, direct_prediction, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(actual.loss, ordinary.loss, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            actual.per_example_loss,
            ordinary.raw_loss_endpoint,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_endpoint_target_is_stopped_and_tilt_weights_are_respected(self) -> None:
        batch = build_action_chunk_endpoint_batch(
            self.states, self.actions, ActionChunkEndpointConfig(horizons=(1,))
        )
        params = init_imf(
            jax.random.key(20), sample_dim=2, condition_dim=3, hidden_dim=6, depth=2
        )
        noise = jax.random.normal(jax.random.key(21), batch.targets.shape)
        tilt = jnp.asarray([0.5, 1.0, 1.5])
        details = action_chunk_endpoint_imf_loss(
            params,
            batch,
            jax.random.key(22),
            noise=noise,
            tilt_weights=tilt,
            return_details=True,
        )
        expected = jnp.sum(details.per_example_loss * tilt) / jnp.sum(tilt)
        np.testing.assert_allclose(details.loss, expected, rtol=1e-6, atol=1e-6)

        def objective(targets):
            changed = batch._replace(targets=targets)
            return action_chunk_endpoint_imf_loss(
                params, changed, jax.random.key(22), noise=noise
            )

        np.testing.assert_array_equal(
            jax.grad(objective)(batch.targets), jnp.zeros_like(batch.targets)
        )

    def test_all_rejected_endpoint_mean_is_finite_zero(self) -> None:
        batch = build_action_chunk_endpoint_batch(
            self.states,
            self.actions,
            ActionChunkEndpointConfig(
                horizons=(1,), minimum_mean_behavior_log_density=0.0
            ),
            behavior_log_densities=jnp.full((1, 3), -10.0),
        )
        params = init_imf(
            jax.random.key(30), sample_dim=2, condition_dim=3, hidden_dim=5, depth=2
        )
        loss = action_chunk_endpoint_imf_loss(
            params, batch, jax.random.key(31), noise=jnp.ones_like(batch.targets)
        )
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertEqual(float(loss), 0.0)

    def test_jitted_batch_and_endpoint_loss(self) -> None:
        config = ActionChunkEndpointConfig(horizons=(1, 2))
        eager_batch = build_action_chunk_endpoint_batch(
            self.states, self.actions, config
        )
        compiled_batch = jit_build_action_chunk_endpoint_batch(
            self.states, self.actions, config
        )
        np.testing.assert_array_equal(compiled_batch.targets, eager_batch.targets)
        np.testing.assert_array_equal(compiled_batch.conditions, eager_batch.conditions)
        params = init_imf(
            jax.random.key(40),
            sample_dim=2,
            condition_dim=eager_batch.conditions.shape[-1],
            hidden_dim=6,
            depth=2,
        )
        noise = jax.random.normal(jax.random.key(41), eager_batch.targets.shape)
        eager = action_chunk_endpoint_imf_loss(
            params, eager_batch, jax.random.key(42), noise=noise
        )
        compiled = jit_action_chunk_endpoint_imf_loss(
            params, compiled_batch, jax.random.key(42), noise=noise
        )
        np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)

    def test_action_chunk_negative_validation(self) -> None:
        invalid_configs = (
            {"horizons": ()},
            {"horizons": 1},
            {"horizons": (2, 1)},
            {"horizons": (1, 1)},
            {"horizons": (0,)},
            {"minimum_mean_behavior_log_density": float("nan")},
            {"maximum_behavior_distance": -1.0},
        )
        for arguments in invalid_configs:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ActionChunkEndpointConfig(**arguments)
        with self.assertRaises(ValueError):
            build_action_chunk_endpoint_batch(
                self.states[:, :-1], self.actions, ActionChunkEndpointConfig()
            )
        with self.assertRaises(ValueError):
            build_action_chunk_endpoint_batch(
                self.states,
                self.actions,
                ActionChunkEndpointConfig(horizons=(4,)),
            )
        with self.assertRaisesRegex(ValueError, "required"):
            build_action_chunk_endpoint_batch(
                self.states,
                self.actions,
                ActionChunkEndpointConfig(minimum_mean_behavior_log_density=-3.0),
            )
        with self.assertRaisesRegex(ValueError, "required"):
            build_action_chunk_endpoint_batch(
                self.states,
                self.actions,
                ActionChunkEndpointConfig(maximum_behavior_distance=1.0),
            )
        with self.assertRaises(ValueError):
            build_action_chunk_endpoint_batch(
                self.states,
                self.actions,
                ActionChunkEndpointConfig(),
                behavior_log_densities=jnp.ones((1, 2)),
            )
        with self.assertRaises(ValueError):
            build_action_chunk_endpoint_batch(
                self.states,
                self.actions,
                ActionChunkEndpointConfig(),
                start_conditions=jnp.ones((1, 2, 1)),
            )


def tearDownModule() -> None:
    print("CONTROL_AWARE_IMF_UNIT_TESTS_OK")


if __name__ == "__main__":
    unittest.main()
