from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    imf_field,
    imf_velocity,
    improved_meanflow_loss,
    init_imf,
    jit_improved_meanflow_loss,
    sample_imf_one_step,
    sample_time_pairs,
)


def tearDownModule() -> None:
    print("IMF_TESTS_OK")


class ImprovedMeanFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = jax.random.key(11)
        self.params = init_imf(self.key, sample_dim=3, condition_dim=4, hidden_dim=8, depth=2)
        self.target = jax.random.normal(jax.random.key(12), (5, 3))
        self.condition = jax.random.normal(jax.random.key(13), (5, 4))
        self.noise = jax.random.normal(jax.random.key(14), (5, 3))
        self.r = jnp.full((5, 1), 0.2)
        self.t = jnp.full((5, 1), 0.7)

    def test_one_step_sample_uses_one_network_evaluation(self) -> None:
        sample = sample_imf_one_step(self.params, self.condition, noise=self.noise)
        expected = self.noise - imf_field(
            self.params, self.noise, self.condition, jnp.zeros((5, 1)), jnp.ones((5, 1))
        )
        np.testing.assert_allclose(sample, expected, rtol=1e-6, atol=1e-6)
        jaxpr = jax.make_jaxpr(
            lambda condition, noise: sample_imf_one_step(
                self.params, condition, noise=noise
            )
        )(self.condition, self.noise)
        dot_count = sum(equation.primitive.name == "dot_general" for equation in jaxpr.jaxpr.eqns)
        self.assertEqual(dot_count, 3, "depth-two MLP must be evaluated exactly once")

    def test_compound_jvp_matches_direct_construction(self) -> None:
        details = improved_meanflow_loss(
            self.params,
            self.target,
            self.condition,
            self.key,
            noise=self.noise,
            r=self.r,
            t=self.t,
            return_details=True,
        )
        interpolated = (1.0 - self.t) * self.target + self.t * self.noise
        marginal = imf_velocity(
            self.params, interpolated, self.condition, self.t, self.t
        )

        def field(z, r, t):
            return imf_field(self.params, z, self.condition, r, t)

        value, tangent = jax.jvp(
            field,
            (interpolated, self.r, self.t),
            (marginal, jnp.zeros_like(self.r), jnp.ones_like(self.t)),
        )
        expected = value + (self.t - self.r) * jax.lax.stop_gradient(tangent)
        np.testing.assert_allclose(details.prediction, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(details.jvp, tangent, rtol=1e-6, atol=1e-6)

    def test_jitted_loss_has_finite_nonzero_parameter_gradients(self) -> None:
        def objective(params):
            return jit_improved_meanflow_loss(
                params,
                self.target,
                self.condition,
                self.key,
                noise=self.noise,
                r=self.r,
                t=self.t,
            )

        loss, gradients = jax.jit(jax.value_and_grad(objective))(self.params)
        self.assertTrue(np.isfinite(np.asarray(loss)).all())
        leaves = jax.tree_util.tree_leaves(gradients)
        self.assertTrue(all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves))
        self.assertGreater(sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves), 0.0)

    def test_time_pairs_obey_order_and_boundary_mass(self) -> None:
        r, t = sample_time_pairs(jax.random.key(3), 128, boundary_fraction=0.25)
        self.assertTrue(bool(jnp.all((0 <= r) & (r <= t) & (t <= 1))))
        boundary_r, boundary_t = sample_time_pairs(
            jax.random.key(4), 16, boundary_fraction=1.0
        )
        np.testing.assert_array_equal(boundary_r, boundary_t)


if __name__ == "__main__":
    unittest.main()
