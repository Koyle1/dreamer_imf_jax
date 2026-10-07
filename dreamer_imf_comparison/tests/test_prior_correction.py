import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax

from imf_dreamer_jax import prior_correction as pc
from imf_dreamer_jax.imf import improved_meanflow_loss


class PriorCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(17)
        self.p = pc.init(self.key, 2, 1, 16, 1)
        self.h = jnp.ones((8, 1))
        self.mean = jnp.ones((8, 2)) * 2
        self.std = jnp.ones((8, 2)) * 0.4
        self.noise = jax.random.normal(self.key, (8, 2))
        self.target = self.mean + 1

    def test_identity_and_explicit_bypass(self):
        base = pc.base_sample(self.mean, self.std, self.noise)
        for steps in (1, 2, 4):
            np.testing.assert_array_equal(
                pc.sample(self.p, self.h, self.mean, self.std, self.noise, steps=steps),
                base,
            )
        np.testing.assert_array_equal(
            pc.sample(None, self.h, self.mean, self.std, self.noise, bypass=True), base
        )

    def test_nonstandard_source_matches_reference_objective(self):
        # Nontrivial parameters exercise the detached JVP, not just zero heads.
        from imf_dreamer_jax.imf import init_imf

        p = init_imf(self.key, 2, 1, 16, 1)
        kw = dict(noise=self.noise, r=0.2, t=0.8, adaptive_power=0.0)
        actual = pc.loss(p, self.target, self.h, self.mean, self.std, self.key, **kw)
        expected = improved_meanflow_loss(
            p,
            self.target,
            self.h,
            jax.random.split(self.key)[1],
            noise=self.mean + self.std * self.noise,
            r=0.2,
            t=0.8,
            adaptive_power=0.0,
            boundary_velocity_supervision=True,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_all_parent_inputs_detached_and_output_head_learns(self):
        def f(p, x, h, m, s):
            return pc.loss(p, x, h, m, s, self.key)

        grads = jax.grad(f, argnums=(0, 1, 2, 3, 4))(
            self.p, self.target, self.h, self.mean, self.std
        )
        self.assertGreater(float(optax.global_norm(grads[0])), 0)
        for g in grads[1:]:
            np.testing.assert_array_equal(g, 0)
        for g in jax.tree.leaves(grads):
            self.assertTrue(np.isfinite(g).all())

    def test_analytic_translation_direction_and_boundary(self):
        # q is a translated p under an analytical deterministic coupling.
        # Correct u=v=base-target=-delta at every interval.
        delta = jnp.array([1.0, -2.0])
        p = dict(
            layers=tuple(self.p["layers"][:-1])
            + (
                dict(
                    weight=self.p["layers"][-1]["weight"],
                    bias=jnp.concatenate((-delta, -delta)),
                ),
            )
        )
        source = self.mean + self.std * self.noise
        for r, t in ((0.2, 0.8), (0.5, 0.5), (0.0, 1.0)):
            value = pc.loss(
                p,
                source + delta,
                self.h,
                self.mean,
                self.std,
                self.key,
                noise=self.noise,
                r=r,
                t=t,
            )
            self.assertLess(float(value), 1e-8)
        np.testing.assert_allclose(
            pc.sample(p, self.h, self.mean, self.std, self.noise),
            source + delta,
            atol=1e-6,
        )

    def test_conditional_synthetic_learning_independent_source_target(self):
        p = pc.init(self.key, 1, 1, 32, 2)
        optimizer = optax.adam(3e-3)
        state = optimizer.init(p)

        @jax.jit
        def update(p, state, key):
            a, b, c = jax.random.split(key, 3)
            h = jax.random.uniform(a, (128, 1), minval=-1, maxval=1)
            target = 1.5 * h + 0.2 * jax.random.normal(b, h.shape)
            value, grad = jax.value_and_grad(pc.loss)(
                p, target, h, 0.5 * h, jnp.full_like(h, 0.2), c
            )
            updates, state = optimizer.update(grad, state, p)
            return optax.apply_updates(p, updates), state, value

        for i in range(400):
            p, state, value = update(p, state, jax.random.fold_in(self.key, i))
        h = jnp.repeat(jnp.array([[-1.0], [1.0]]), 1024, axis=0)
        noise = jax.random.normal(jax.random.PRNGKey(999), h.shape)
        base = 0.5 * h + 0.2 * noise
        prediction = pc.sample(p, h, 0.5 * h, jnp.full_like(h, 0.2), noise)
        self.assertTrue(np.isfinite(value))
        self.assertLess(
            float(jnp.mean((prediction - 1.5 * h) ** 2)),
            0.5 * float(jnp.mean((base - 1.5 * h) ** 2)),
        )
        self.assertGreater(
            float(prediction[1024:].mean() - prediction[:1024].mean()), 2.0
        )

    def test_invalid_shape_and_penalty_rejected(self):
        with self.assertRaises(ValueError):
            pc.base_sample(self.mean, self.std[:1], self.noise)
        with self.assertRaises(ValueError):
            pc.loss(
                self.p,
                self.target,
                self.h,
                self.mean,
                self.std,
                self.key,
                endpoint_scale=1.0,
            )


if __name__ == "__main__":
    unittest.main()
