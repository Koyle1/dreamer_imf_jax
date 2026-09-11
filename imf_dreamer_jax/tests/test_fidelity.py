from __future__ import annotations

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DreamerConfig,
    imf_field,
    improved_meanflow_loss,
    init_imf,
    sample_imf_one_step,
)
from imf_dreamer_jax.fidelity import (
    posterior_sequence_noise,
    scale_condition_gradient,
    transport_base_noise,
)
from imf_dreamer_jax.world_model import (
    _distance_consistency_loss,
    init_world_model,
    observe_sequence,
    world_model_loss,
)
import imf_dreamer_jax.world_model as world_model_module


def tiny_config(**overrides: object) -> DreamerConfig:
    values: dict[str, object] = {
        "observation_shape": (3,),
        "action_dim": 2,
        "deterministic_dim": 4,
        "stochastic_dim": 2,
        "embedding_dim": 4,
        "hidden_dim": 6,
        "prior": "imf",
        "overshooting_distances": (1,),
        "imagination_horizon": 3,
    }
    values.update(overrides)
    return DreamerConfig(**values)


def tiny_batch(config: DreamerConfig, *, batch: int = 2, time: int = 4):
    return {
        "observations": jax.random.normal(
            jax.random.key(700), (batch, time, *config.observation_shape)
        ),
        "actions": jnp.tanh(
            jax.random.normal(
                jax.random.key(701), (batch, time, config.action_dim)
            )
        ),
        "rewards": jax.random.normal(jax.random.key(702), (batch, time)),
        "continuations": jnp.ones((batch, time), dtype=jnp.float32),
    }


class FidelityConfigurationTests(unittest.TestCase):
    def test_legacy_defaults_and_repaired_values_are_explicit(self) -> None:
        legacy = DreamerConfig()
        self.assertEqual(legacy.imf_noise_coupling, "independent")
        self.assertEqual(legacy.imf_endpoint_scale, 0.0)
        self.assertEqual(legacy.imf_condition_gradient_scale, 1.0)
        self.assertFalse(legacy.imf_boundary_velocity_supervision)

        repaired = tiny_config(
            imf_noise_coupling="posterior",
            imf_endpoint_scale=0.75,
            imf_condition_gradient_scale=1.0,
            imf_boundary_velocity_supervision=True,
        )
        self.assertEqual(repaired.imf_noise_coupling, "posterior")
        self.assertEqual(repaired.imf_endpoint_scale, 0.75)
        self.assertTrue(repaired.imf_boundary_velocity_supervision)

    def test_invalid_fidelity_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "imf_noise_coupling"):
            tiny_config(imf_noise_coupling="shared")
        with self.assertRaisesRegex(ValueError, "imf_endpoint_scale"):
            tiny_config(imf_endpoint_scale=-0.1)
        with self.assertRaisesRegex(ValueError, "condition_gradient_scale"):
            tiny_config(imf_condition_gradient_scale=1.1)
        with self.assertRaisesRegex(ValueError, "boolean"):
            tiny_config(imf_boundary_velocity_supervision=1)


class FidelityIMFObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = init_imf(
            jax.random.key(710), sample_dim=3, condition_dim=4, hidden_dim=9, depth=2
        )
        self.target = jax.random.normal(jax.random.key(711), (6, 3))
        self.condition = jax.random.normal(jax.random.key(712), (6, 4))
        self.noise = jax.random.normal(jax.random.key(713), (6, 3))
        self.r = jnp.full((6, 1), 0.2)
        self.t = jnp.full((6, 1), 0.7)

    def _details(self, **kwargs: object):
        return improved_meanflow_loss(
            self.params,
            self.target,
            self.condition,
            jax.random.key(714),
            noise=self.noise,
            r=self.r,
            t=self.t,
            return_details=True,
            **kwargs,
        )

    def test_legacy_loss_defaults_are_numerically_unchanged(self) -> None:
        implicit = self._details()
        explicit = self._details(
            endpoint_scale=0.0,
            condition_gradient_scale=1.0,
            boundary_velocity_supervision=False,
        )
        for left, right in zip(implicit, explicit, strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            implicit.raw_loss_endpoint, jnp.zeros((self.target.shape[0],))
        )

    def test_boundary_velocity_supervision_uses_the_jvp_boundary_head(self) -> None:
        details = self._details(boundary_velocity_supervision=True)
        target_velocity = self.noise - self.target
        expected = jnp.mean(
            jnp.square(details.marginal_velocity - target_velocity), axis=-1
        )
        legacy_expected = jnp.mean(
            jnp.square(details.velocity_prediction - target_velocity), axis=-1
        )
        np.testing.assert_allclose(details.raw_loss_v, expected, rtol=1e-6, atol=1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(expected - legacy_expected))), 1e-5)

    def test_endpoint_is_direct_one_step_mse_with_nonzero_quality_gradient(self) -> None:
        scale = 0.8
        details = self._details(endpoint_scale=scale)
        zeros = jnp.zeros((self.target.shape[0], 1), dtype=self.target.dtype)
        ones = jnp.ones_like(zeros)
        expected_sample = self.noise - imf_field(
            self.params, self.noise, self.condition, zeros, ones
        )
        expected_mse = jnp.mean(jnp.square(expected_sample - self.target), axis=-1)
        np.testing.assert_allclose(details.endpoint_prediction, expected_sample, rtol=1e-6)
        np.testing.assert_allclose(details.raw_loss_endpoint, expected_mse, rtol=1e-6)

        without_endpoint = self._details(endpoint_scale=0.0)
        np.testing.assert_allclose(
            details.per_sample_loss - without_endpoint.per_sample_loss,
            scale * expected_mse,
            rtol=2e-6,
            atol=2e-6,
        )

        def endpoint_quality(params):
            result = improved_meanflow_loss(
                params,
                self.target,
                self.condition,
                jax.random.key(714),
                noise=self.noise,
                r=self.r,
                t=self.t,
                endpoint_scale=1.0,
                return_details=True,
            )
            return jnp.mean(result.raw_loss_endpoint)

        before, gradient = jax.value_and_grad(endpoint_quality)(self.params)
        gradient_norm = sum(
            float(jnp.sum(jnp.square(leaf)))
            for leaf in jax.tree_util.tree_leaves(gradient)
        )
        self.assertGreater(gradient_norm, 0.0)
        updated = jax.tree_util.tree_map(
            lambda parameter, grad: parameter - 1e-3 * grad,
            self.params,
            gradient,
        )
        after = endpoint_quality(updated)
        self.assertLess(float(after), float(before))

    def test_condition_gradient_scaling_preserves_value_and_scales_gradient(self) -> None:
        def objective(condition, scale):
            return improved_meanflow_loss(
                self.params,
                self.target,
                condition,
                jax.random.key(715),
                noise=self.noise,
                r=self.r,
                t=self.t,
                endpoint_scale=0.6,
                boundary_velocity_supervision=True,
                condition_gradient_scale=scale,
            )

        value_zero = objective(self.condition, 0.0)
        value_full = objective(self.condition, 1.0)
        np.testing.assert_allclose(value_zero, value_full, rtol=0.0, atol=1e-6)
        gradient_zero = jax.grad(lambda value: objective(value, 0.0))(self.condition)
        gradient_quarter = jax.grad(lambda value: objective(value, 0.25))(self.condition)
        gradient_full = jax.grad(lambda value: objective(value, 1.0))(self.condition)
        np.testing.assert_array_equal(gradient_zero, jnp.zeros_like(gradient_zero))
        np.testing.assert_allclose(
            gradient_quarter, 0.25 * gradient_full, rtol=2e-5, atol=2e-6
        )

    def test_repaired_mode_keeps_one_nfe_sampler(self) -> None:
        sample = sample_imf_one_step(self.params, self.condition, noise=self.noise)
        self.assertEqual(sample.shape, self.noise.shape)
        jaxpr = jax.make_jaxpr(
            lambda condition, noise: sample_imf_one_step(
                self.params, condition, noise=noise
            )
        )(self.condition, self.noise)
        dot_count = sum(
            equation.primitive.name == "dot_general" for equation in jaxpr.jaxpr.eqns
        )
        self.assertEqual(dot_count, 3, "the depth-two sampler must use one MLP evaluation")


class PosteriorNoiseCouplingTests(unittest.TestCase):
    def test_noise_helpers_reproduce_observation_reparameterization(self) -> None:
        config = tiny_config()
        params = init_world_model(config, jax.random.key(720))
        batch = tiny_batch(config)
        key = jax.random.key(721)
        sequence = observe_sequence(
            params, batch["observations"], batch["actions"], key, config
        )
        epsilon = posterior_sequence_noise(
            key,
            batch["actions"].shape[0],
            batch["actions"].shape[1],
            config.stochastic_dim,
            dtype=sequence.states.stochastic.dtype,
        )
        np.testing.assert_allclose(
            sequence.states.stochastic,
            sequence.posterior.mean + sequence.posterior.std * epsilon,
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertIsNone(transport_base_noise(epsilon, "independent"))
        np.testing.assert_array_equal(
            transport_base_noise(epsilon, "posterior"), epsilon
        )
        scaled = scale_condition_gradient(sequence.states.deterministic, 0.0)
        np.testing.assert_array_equal(scaled, sequence.states.deterministic)

    def test_main_prior_receives_exact_posterior_epsilon(self) -> None:
        config = tiny_config(
            imf_noise_coupling="posterior",
            imf_endpoint_scale=0.5,
            imf_boundary_velocity_supervision=True,
        )
        params = init_world_model(config, jax.random.key(730))
        batch = tiny_batch(config)
        key = jax.random.key(731)
        sequence_key = jax.random.split(key, 3)[0]
        expected = posterior_sequence_noise(
            sequence_key,
            batch["actions"].shape[0],
            batch["actions"].shape[1],
            config.stochastic_dim,
        ).reshape((-1, config.stochastic_dim))
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        original = world_model_module.improved_meanflow_loss

        def capture(*args: object, **kwargs: object):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        with mock.patch.object(world_model_module, "improved_meanflow_loss", side_effect=capture):
            losses = world_model_loss(params, batch, key, config)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][1]["noise"], expected)
        self.assertGreater(float(losses.imf_endpoint), 0.0)

    def test_overshooting_target_and_transport_share_posterior_epsilon(self) -> None:
        config = tiny_config(
            imf_noise_coupling="posterior",
            imf_endpoint_scale=0.5,
            imf_boundary_velocity_supervision=True,
        )
        params = init_world_model(config, jax.random.key(740))
        batch = tiny_batch(config, batch=1, time=4)
        sequence = observe_sequence(
            params,
            batch["observations"],
            batch["actions"],
            jax.random.key(741),
            config,
        )
        captured: list[tuple[tuple[object, ...], dict[str, object]]] = []
        original = world_model_module.improved_meanflow_loss

        def capture(*args: object, **kwargs: object):
            captured.append((args, kwargs))
            return original(*args, **kwargs)

        mask = jnp.ones((1, 4), dtype=jnp.float32)
        with mock.patch.object(world_model_module, "improved_meanflow_loss", side_effect=capture):
            loss = _distance_consistency_loss(
                params,
                sequence,
                batch["actions"],
                jnp.zeros((1, 4), dtype=jnp.bool_),
                mask,
                jax.random.key(742),
                config,
                2,
            )
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(len(captured), 1)
        args, kwargs = captured[0]
        target_sample = args[1]
        epsilon = kwargs["noise"]
        target_mean = sequence.posterior.mean[:, 2:].reshape((-1, config.stochastic_dim))
        target_std = sequence.posterior.std[:, 2:].reshape((-1, config.stochastic_dim))
        np.testing.assert_allclose(
            target_sample, target_mean + target_std * epsilon, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
