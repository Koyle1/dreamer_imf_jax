from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.imf import (
    imf_field,
    imf_outputs,
    imf_velocity,
    improved_meanflow_loss,
    init_imf,
    sample_imf_one_step,
    sample_time_pairs,
)


class OfficialIMFCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = init_imf(
            jax.random.key(100), sample_dim=3, condition_dim=4, hidden_dim=11, depth=2
        )
        self.target = jax.random.normal(jax.random.key(101), (8, 3))
        self.condition = jax.random.normal(jax.random.key(102), (8, 4))
        self.noise = jax.random.normal(jax.random.key(103), (8, 3))
        self.r = jnp.linspace(0.05, 0.35, 8)[:, None]
        self.t = jnp.linspace(0.55, 0.95, 8)[:, None]

    def test_shared_network_has_auxiliary_velocity_and_one_nfe_sampler(self) -> None:
        self.assertEqual(self.params["layers"][-1]["bias"].shape, (6,))
        u, v = imf_outputs(
            self.params, self.noise, self.condition, self.r, self.t
        )
        self.assertEqual(u.shape, self.noise.shape)
        self.assertEqual(v.shape, self.noise.shape)
        self.assertGreater(float(jnp.max(jnp.abs(u - v))), 1e-5)
        np.testing.assert_allclose(
            imf_field(self.params, self.noise, self.condition, self.r, self.t), u
        )
        np.testing.assert_allclose(
            imf_velocity(self.params, self.noise, self.condition, self.r, self.t), v
        )

        sample = sample_imf_one_step(
            self.params, self.condition, noise=self.noise
        )
        zeros = jnp.zeros_like(self.r)
        ones = jnp.ones_like(self.r)
        expected_u, _ = imf_outputs(
            self.params, self.noise, self.condition, zeros, ones
        )
        np.testing.assert_allclose(sample, self.noise - expected_u, rtol=1e-6, atol=1e-6)

        jaxpr = jax.make_jaxpr(
            lambda condition, noise: sample_imf_one_step(
                self.params, condition, noise=noise
            )
        )(self.condition, self.noise)
        dot_count = sum(
            equation.primitive.name == "dot_general" for equation in jaxpr.jaxpr.eqns
        )
        self.assertEqual(dot_count, 3, "the depth-two shared MLP must run exactly once")

    def test_predicted_velocity_is_the_jvp_state_tangent(self) -> None:
        details = improved_meanflow_loss(
            self.params,
            self.target,
            self.condition,
            jax.random.key(104),
            noise=self.noise,
            r=self.r,
            t=self.t,
            return_details=True,
        )
        z = (1.0 - self.t) * self.target + self.t * self.noise
        predicted_v = imf_velocity(
            self.params, z, self.condition, self.t, self.t
        )

        def average_field(z_value, r_value, t_value):
            return imf_field(
                self.params, z_value, self.condition, r_value, t_value
            )

        u, correct_jvp = jax.jvp(
            average_field,
            (z, self.r, self.t),
            (predicted_v, jnp.zeros_like(self.r), jnp.ones_like(self.t)),
        )
        correct = u + (self.t - self.r) * jax.lax.stop_gradient(correct_jvp)
        np.testing.assert_allclose(details.marginal_velocity, predicted_v, rtol=1e-6)
        np.testing.assert_allclose(details.jvp, correct_jvp, rtol=1e-6)
        np.testing.assert_allclose(details.prediction, correct, rtol=1e-6)

        # Positive-control mutation: the old implementation incorrectly used
        # the average-velocity head itself as the state tangent.
        wrong_velocity = imf_field(
            self.params, z, self.condition, self.t, self.t
        )
        _, wrong_jvp = jax.jvp(
            average_field,
            (z, self.r, self.t),
            (wrong_velocity, jnp.zeros_like(self.r), jnp.ones_like(self.t)),
        )
        mutated = u + (self.t - self.r) * jax.lax.stop_gradient(wrong_jvp)
        self.assertGreater(float(jnp.max(jnp.abs(correct - mutated))), 1e-4)

    def test_dual_raw_losses_and_official_adaptive_weighting(self) -> None:
        power = 1.0
        epsilon = 0.01
        velocity_scale = 0.7
        details = improved_meanflow_loss(
            self.params,
            self.target,
            self.condition,
            jax.random.key(105),
            noise=self.noise,
            r=self.r,
            t=self.t,
            adaptive_power=power,
            adaptive_epsilon=epsilon,
            velocity_scale=velocity_scale,
            return_details=True,
        )
        self.assertEqual(details.raw_loss_u.shape, (8,))
        self.assertEqual(details.raw_loss_v.shape, (8,))
        self.assertTrue(np.isfinite(np.asarray(details.loss)))
        self.assertGreater(float(jnp.mean(details.raw_loss_u)), 0.0)
        self.assertGreater(float(jnp.mean(details.raw_loss_v)), 0.0)

        summed_u = details.raw_loss_u * self.target.shape[-1]
        summed_v = details.raw_loss_v * self.target.shape[-1]
        expected_weight_u = jnp.power(summed_u + epsilon, power)
        expected_weight_v = jnp.power(summed_v + epsilon, power)
        expected_per_sample = (
            summed_u / expected_weight_u
            + velocity_scale * summed_v / expected_weight_v
        )
        np.testing.assert_allclose(details.adaptive_weight_u, expected_weight_u, rtol=1e-6)
        np.testing.assert_allclose(details.adaptive_weight_v, expected_weight_v, rtol=1e-6)
        np.testing.assert_allclose(details.per_sample_loss, expected_per_sample, rtol=1e-6)
        np.testing.assert_allclose(details.loss, jnp.mean(expected_per_sample), rtol=1e-6)

        def objective(params):
            return improved_meanflow_loss(
                params,
                self.target,
                self.condition,
                jax.random.key(105),
                noise=self.noise,
                r=self.r,
                t=self.t,
                adaptive_power=power,
                adaptive_epsilon=epsilon,
                velocity_scale=velocity_scale,
            )

        gradients = jax.grad(objective)(self.params)
        output_gradient = gradients["layers"][-1]["weight"]
        self.assertGreater(float(jnp.linalg.norm(output_gradient[:, :3])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(output_gradient[:, 3:])), 0.0)

        def without_auxiliary_loss(params):
            return improved_meanflow_loss(
                params,
                self.target,
                self.condition,
                jax.random.key(105),
                noise=self.noise,
                r=self.r,
                t=self.t,
                adaptive_power=power,
                adaptive_epsilon=epsilon,
                velocity_scale=0.0,
            )

        without_auxiliary_gradient = jax.grad(without_auxiliary_loss)(self.params)
        auxiliary_gradient_delta = jnp.linalg.norm(
            output_gradient[:, 3:]
            - without_auxiliary_gradient["layers"][-1]["weight"][:, 3:]
        )
        self.assertGreater(float(auxiliary_gradient_delta), 1e-5)

        def manual_correct(params):
            result = improved_meanflow_loss(
                params,
                self.target,
                self.condition,
                jax.random.key(105),
                noise=self.noise,
                r=self.r,
                t=self.t,
                adaptive_power=power,
                adaptive_epsilon=epsilon,
                velocity_scale=velocity_scale,
                return_details=True,
            )
            u_sum = result.raw_loss_u * self.target.shape[-1]
            v_sum = result.raw_loss_v * self.target.shape[-1]
            return jnp.mean(
                u_sum / jax.lax.stop_gradient((u_sum + epsilon) ** power)
                + velocity_scale
                * v_sum
                / jax.lax.stop_gradient((v_sum + epsilon) ** power)
            )

        def unstopped_mutation(params):
            result = improved_meanflow_loss(
                params,
                self.target,
                self.condition,
                jax.random.key(105),
                noise=self.noise,
                r=self.r,
                t=self.t,
                adaptive_power=power,
                adaptive_epsilon=epsilon,
                velocity_scale=velocity_scale,
                return_details=True,
            )
            u_sum = result.raw_loss_u * self.target.shape[-1]
            v_sum = result.raw_loss_v * self.target.shape[-1]
            return jnp.mean(
                u_sum / (u_sum + epsilon) ** power
                + velocity_scale * v_sum / (v_sum + epsilon) ** power
            )

        correct_gradient = jax.grad(manual_correct)(self.params)
        mutant_gradient = jax.grad(unstopped_mutation)(self.params)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(gradients),
            jax.tree_util.tree_leaves(correct_gradient),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        gradient_delta = sum(
            float(jnp.sum(jnp.abs(left - right)))
            for left, right in zip(
                jax.tree_util.tree_leaves(gradients),
                jax.tree_util.tree_leaves(mutant_gradient),
                strict=True,
            )
        )
        self.assertGreater(gradient_delta, 1e-3)

    def test_logit_normal_ordering_and_exact_half_boundary_mass(self) -> None:
        key = jax.random.key(106)
        r, t = sample_time_pairs(
            key,
            100,
            boundary_fraction=0.5,
            time_mean=-0.4,
            time_std=1.0,
        )
        raw = jax.nn.sigmoid(
            jax.random.normal(key, (100, 2), dtype=jnp.float32) - 0.4
        )
        expected_t = jnp.max(raw, axis=-1, keepdims=True)
        expected_r = jnp.min(raw, axis=-1, keepdims=True)
        expected_r = expected_r.at[:50].set(expected_t[:50])
        np.testing.assert_array_equal(t, expected_t)
        np.testing.assert_array_equal(r, expected_r)
        self.assertEqual(int(jnp.sum(r == t)), 50)
        self.assertTrue(bool(jnp.all((0.0 < r) & (r <= t) & (t < 1.0))))


def tearDownModule() -> None:
    print("OFFICIAL_IMF_CORE_REPAIRS_OK")


if __name__ == "__main__":
    unittest.main()
