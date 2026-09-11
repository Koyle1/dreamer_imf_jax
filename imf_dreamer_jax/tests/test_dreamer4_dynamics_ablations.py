from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.imf import (
    improved_meanflow_loss,
    init_imf,
    sample_imf_one_step,
    sample_imf_steps,
    transport_imf_interval,
)
from imf_dreamer_jax.nn import tree_global_norm
from imf_dreamer_jax.world_model import init_world_model, sample_prior


class Dreamer4DynamicsAblationTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("DREAMER4_DYNAMICS_ABLATION_TESTS_OK")

    def setUp(self) -> None:
        self.params = init_imf(jax.random.key(1), 3, 4, 12, depth=2)
        self.condition = jax.random.normal(jax.random.key(2), (8, 4))
        self.target = jax.random.normal(jax.random.key(3), (8, 3))
        self.noise = jax.random.normal(jax.random.key(4), (8, 3))

    def test_one_requested_step_is_exactly_the_existing_sampler(self) -> None:
        expected = sample_imf_one_step(
            self.params, self.condition, noise=self.noise
        )
        actual = sample_imf_steps(
            self.params, self.condition, noise=self.noise, steps=1
        )
        np.testing.assert_array_equal(actual, expected)

    def test_requested_sampling_steps_change_random_model_trajectory(self) -> None:
        one = sample_imf_steps(
            self.params, self.condition, noise=self.noise, steps=1
        )
        four = sample_imf_steps(
            self.params, self.condition, noise=self.noise, steps=4
        )
        self.assertGreater(float(jnp.max(jnp.abs(one - four))), 1e-6)
        self.assertTrue(np.isfinite(np.asarray(four)).all())
        with self.assertRaisesRegex(ValueError, "positive integer"):
            sample_imf_steps(self.params, self.condition, noise=self.noise, steps=0)

    def test_interval_transport_rejects_reversed_scalar_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "r <= t"):
            transport_imf_interval(
                self.params, self.noise, self.condition, 0.8, 0.2
            )

    def test_signal_ramp_uses_clean_signal_not_noise_fraction(self) -> None:
        r = jnp.asarray([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]])
        t = jnp.asarray([[0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]])
        details = improved_meanflow_loss(
            self.params,
            self.target,
            self.condition,
            jax.random.key(5),
            noise=self.noise,
            r=r,
            t=t,
            signal_weight_floor=0.1,
            signal_weight_scale=0.9,
            return_details=True,
        )
        np.testing.assert_allclose(
            details.signal_weights, 0.1 + 0.9 * (1.0 - np.asarray(t[:, 0]))
        )
        expected = np.average(
            np.asarray(details.per_sample_loss),
            weights=np.asarray(details.signal_weights),
        )
        np.testing.assert_allclose(details.loss, expected, rtol=2e-6)

    def test_endpoint_can_be_the_primary_regression_objective(self) -> None:
        details = improved_meanflow_loss(
            self.params,
            self.target,
            self.condition,
            jax.random.key(6),
            noise=self.noise,
            meanflow_scale=0.0,
            velocity_scale=0.0,
            endpoint_scale=1.0,
            return_details=True,
        )
        np.testing.assert_allclose(
            details.loss, jnp.mean(details.raw_loss_endpoint), rtol=2e-6
        )
        self.assertGreater(float(details.loss), 0.0)

    def test_shortcut_bootstrap_is_zero_for_zero_field_and_trains_random_field(self) -> None:
        zero_params = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        zero = improved_meanflow_loss(
            zero_params,
            self.target,
            self.condition,
            jax.random.key(7),
            noise=self.noise,
            r=0.0,
            t=1.0,
            meanflow_scale=0.0,
            velocity_scale=0.0,
            shortcut_scale=1.0,
            return_details=True,
        )
        np.testing.assert_array_equal(zero.raw_loss_shortcut, np.zeros((8,)))

        def objective(parameters: object) -> jax.Array:
            return improved_meanflow_loss(
                parameters,
                self.target,
                self.condition,
                jax.random.key(8),
                noise=self.noise,
                r=0.0,
                t=1.0,
                meanflow_scale=0.0,
                velocity_scale=0.0,
                shortcut_scale=1.0,
            )

        loss, gradient = jax.value_and_grad(objective)(self.params)
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(tree_global_norm(gradient)), 1e-8)

    def test_world_model_sampling_steps_are_wired_to_prior(self) -> None:
        base = DreamerConfig(
            observation_shape=(3,),
            action_dim=2,
            deterministic_dim=4,
            stochastic_dim=3,
            embedding_dim=4,
            hidden_dim=12,
            prior="imf",
        )
        model = init_world_model(base, jax.random.key(9))
        one, _ = sample_prior(
            model,
            self.condition,
            None,
            base,
            noise=self.noise,
        )
        four, _ = sample_prior(
            model,
            self.condition,
            None,
            DreamerConfig(**{**base.__dict__, "imf_sampling_steps": 4}),
            noise=self.noise,
        )
        self.assertGreater(float(jnp.max(jnp.abs(one - four))), 1e-6)

    def test_invalid_weighting_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "signal weighting"):
            DreamerConfig(
                imf_signal_weight_floor=0.0,
                imf_signal_weight_scale=0.0,
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            DreamerConfig(imf_shortcut_scale=-1.0)


if __name__ == "__main__":
    unittest.main()

