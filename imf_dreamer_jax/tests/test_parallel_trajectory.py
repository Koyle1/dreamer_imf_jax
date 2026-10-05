"""Numerical, structural and actual-learning checks for the joint field."""

from __future__ import annotations

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from imf_dreamer_jax import parallel_trajectory as trajectory


class ParallelTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = trajectory.init(
            jax.random.key(0), 3, 2, horizon=3, width=12, heads=3, layers=1, ff_width=24
        )
        cls.targets = jax.random.normal(jax.random.key(1), (2, 3, 3))
        cls.start = jax.random.normal(jax.random.key(2), (2, 3))
        cls.actions = jax.random.normal(jax.random.key(3), (2, 3, 2))
        cls.noise = jax.random.normal(jax.random.key(4), (2, 3, 3))
        cls.r = jnp.asarray([0.12, 0.23])
        cls.t = jnp.asarray([0.62, 0.87])

    def details(self, **kwargs):
        return trajectory.loss(
            self.params,
            self.targets,
            self.start,
            self.actions,
            jax.random.key(5),
            noise=self.noise,
            r=self.r,
            t=self.t,
            return_details=True,
            **kwargs,
        )

    def test_default_architecture_and_complete_joint_feature_shape(self):
        params = trajectory.init(jax.random.key(6), 2048 + 512, 2)
        self.assertEqual(params["position"].shape, (15, 128))
        self.assertEqual(len(params["blocks"]), 4)
        for block in params["blocks"]:
            self.assertEqual(block["qkv"]["weight"].shape, (128, 3, 4, 32))
            self.assertEqual(block["ff_in"]["weight"].shape, (128, 512))
            self.assertEqual(block["ff_out"]["weight"].shape, (512, 128))
        self.assertEqual(params["u"]["weight"].shape, (128, 2 * 2560))
        self.assertEqual(params["v"]["weight"].shape, (128, 2 * 2560))
        self.assertTrue(
            all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(params))
        )
        actual = jax.jit(trajectory.outputs)(
            params,
            jnp.ones((1, 15, 2560)),
            jnp.zeros((1, 2560)),
            jnp.zeros((1, 15, 2)),
            0.0,
            1.0,
        )
        for head in actual:
            self.assertEqual(head.shape, (1, 15, 2560))
            self.assertTrue(np.isfinite(np.asarray(head)).all())
        value, gradients = jax.jit(jax.value_and_grad(trajectory.loss))(
            params,
            jax.random.normal(jax.random.key(7), (1, 15, 2560)),
            jnp.zeros((1, 2560)),
            jnp.ones((1, 15, 2)),
            jax.random.key(8),
        )
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(
            all(
                np.isfinite(np.asarray(x)).all()
                for x in jax.tree_util.tree_leaves(gradients)
            )
        )
        for head in ("u", "v"):
            # Both affine factors and shifts must receive real gradients at
            # the full 2048+512 dimensional diagnostic configuration.
            alpha_gradient, beta_gradient = jnp.split(
                gradients[head]["weight"], 2, axis=-1
            )
            self.assertGreater(float(jnp.linalg.norm(alpha_gradient)), 1e-6)
            self.assertGreater(float(jnp.linalg.norm(beta_gradient)), 1e-6)

    def test_affine_readout_can_change_noise_outside_additive_head_subspace(self):
        features, width = 12, 4
        params = trajectory.init(
            jax.random.key(10),
            features,
            1,
            horizon=2,
            width=width,
            heads=1,
            layers=1,
            ff_width=8,
        )
        beta_weights = params["u"]["weight"][:, features:]
        _, _, basis = jnp.linalg.svd(beta_weights, full_matrices=False)
        direction = jax.random.normal(jax.random.key(11), (features,))
        direction = direction - (direction @ basis.T) @ basis
        direction = direction / jnp.linalg.norm(direction)
        np.testing.assert_allclose(beta_weights @ direction, 0.0, atol=1e-7)
        start, actions = jnp.zeros((1, features)), jnp.zeros((1, 2, 1))
        noise = jnp.broadcast_to(direction, (1, 2, features))
        generated = trajectory.sample(params, start, actions, noise)
        # A plain additive head cannot change this projection at all; the
        # diagonal path removes essentially all of it at initialization.
        self.assertGreater(
            float(jnp.min(jnp.abs((generated - noise) @ direction))), 0.8
        )
        self.assertLess(float(jnp.max(jnp.abs(generated @ direction))), 0.2)

        # Per-coordinate diagonal scaling can express arbitrary diagonal
        # Gaussian covariance, including rank strictly greater than width.
        params = jax.tree_util.tree_map(jnp.zeros_like, params)
        std = jnp.linspace(0.25, 1.5, features)
        mu = jnp.linspace(-0.5, 0.5, features)
        params["u"]["bias"] = jnp.concatenate((-std, -mu))
        noise = jax.random.normal(jax.random.key(12), (4, 2, features))
        generated = trajectory.sample(
            params, jnp.zeros((4, features)), jnp.zeros((4, 2, 1)), noise
        )
        np.testing.assert_allclose(generated, noise * std + mu, atol=3e-7)

    def test_shapes_time_forms_and_rejections(self):
        expected = trajectory.outputs(
            self.params, self.noise, self.start, self.actions, 0.2, 0.8
        )
        for r, t in (
            (jnp.full((2,), 0.2), jnp.full((2,), 0.8)),
            (jnp.full((2, 1), 0.2), jnp.full((2, 1), 0.8)),
            (jnp.full((2, 1, 1), 0.2), jnp.full((2, 1, 1), 0.8)),
        ):
            actual = trajectory.outputs(
                self.params, self.noise, self.start, self.actions, r, t
            )
            for first, second in zip(expected, actual):
                np.testing.assert_allclose(first, second, atol=1e-7)
        with self.assertRaises(ValueError):
            trajectory.outputs(
                self.params, self.noise, self.start, self.actions, jnp.ones((2, 3)), 1.0
            )
        with self.assertRaises(ValueError):
            trajectory.outputs(
                self.params, self.noise[:, :2], self.start, self.actions, 0.0, 1.0
            )
        with self.assertRaises(ValueError):
            trajectory.outputs(
                self.params, self.noise, self.start, self.actions[:, :2], 0.0, 1.0
            )
        with self.assertRaises(ValueError):
            trajectory.loss(
                self.params,
                self.targets,
                self.start,
                self.actions,
                jax.random.key(0),
                r=0.0,
            )
        with self.assertRaises(ValueError):
            trajectory.loss(
                self.params,
                self.targets,
                self.start,
                self.actions,
                jax.random.key(0),
                noise=self.noise[:, :2],
            )
        for steps in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                trajectory.sample(
                    self.params, self.start, self.actions, self.noise, steps
                )
        for kwargs in ({"width": 11}, {"heads": 0}, {"layers": True}, {"ff_width": -2}):
            with self.assertRaises(ValueError):
                trajectory.init(jax.random.key(0), 3, 2, **kwargs)

    def test_complete_tensor_jvp_has_cross_token_terms(self):
        details = self.details()
        boundary_v = trajectory.outputs(
            self.params,
            details.interpolated,
            self.start,
            self.actions,
            details.t,
            details.t,
        )[1]

        def field(x, r, t):
            return trajectory.outputs(self.params, x, self.start, self.actions, r, t)[0]

        value, tangent = jax.jvp(
            field,
            (details.interpolated, details.r, details.t),
            (boundary_v, jnp.zeros_like(details.r), jnp.ones_like(details.t)),
        )
        np.testing.assert_allclose(details.field, value, atol=1e-6)
        np.testing.assert_allclose(details.jvp, tangent, atol=1e-6)
        np.testing.assert_allclose(
            details.prediction, value + (details.t - details.r) * tangent, atol=1e-6
        )
        epsilon = 1e-3
        finite_difference = (
            field(
                details.interpolated + epsilon * boundary_v,
                details.r,
                details.t + epsilon,
            )
            - field(
                details.interpolated - epsilon * boundary_v,
                details.r,
                details.t - epsilon,
            )
        ) / (2 * epsilon)
        np.testing.assert_allclose(tangent, finite_difference, rtol=3e-3, atol=5e-5)
        # Positive-control mutation: only perturb a token's own state while
        # keeping the other physical-time tokens fixed.  That is NOT this JVP.
        tokenwise = []
        for index in range(3):
            local_v = jnp.zeros_like(boundary_v).at[:, index].set(boundary_v[:, index])
            _, local_jvp = jax.jvp(
                field,
                (details.interpolated, details.r, details.t),
                (local_v, jnp.zeros_like(details.r), jnp.ones_like(details.t)),
            )
            tokenwise.append(local_jvp[:, index])
        tokenwise = jnp.stack(tokenwise, axis=1)
        self.assertGreater(float(jnp.max(jnp.abs(tangent - tokenwise))), 1e-3)
        # The supplied plan is a real conditioning input, including across tokens.
        changed_actions = self.actions.at[:, -1].add(1.0)
        changed = trajectory.outputs(
            self.params, self.noise, self.start, changed_actions, 0.0, 1.0
        )[0]
        original = trajectory.outputs(
            self.params, self.noise, self.start, self.actions, 0.0, 1.0
        )[0]
        self.assertGreater(
            float(jnp.max(jnp.abs(changed[:, 0] - original[:, 0]))), 1e-4
        )

    def test_boundary_auxiliary_supervision_and_joint_adaptive_reduction(self):
        details = self.details()
        interval_v = trajectory.outputs(
            self.params,
            details.interpolated,
            self.start,
            self.actions,
            details.r,
            details.t,
        )[1]
        square_u = (details.prediction - (self.noise - self.targets)) ** 2
        square_v = (details.marginal_velocity - (self.noise - self.targets)) ** 2
        sums_u, sums_v = (jnp.sum(x, axis=(1, 2)) for x in (square_u, square_v))
        np.testing.assert_allclose(details.adaptive_weight_u, sums_u + 0.01, atol=1e-6)
        np.testing.assert_allclose(details.adaptive_weight_v, sums_v + 0.01, atol=1e-6)
        np.testing.assert_allclose(
            details.raw_loss_v, jnp.mean(square_v, axis=(1, 2)), atol=1e-6
        )
        np.testing.assert_allclose(
            details.per_sample_loss, sums_u / (sums_u + 0.01) + sums_v / (sums_v + 0.01)
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(interval_v - details.marginal_velocity))), 1e-4
        )
        boundary = trajectory.loss(
            self.params,
            self.targets,
            self.start,
            self.actions,
            jax.random.key(5),
            noise=self.noise,
            r=self.t,
            t=self.t,
            return_details=True,
        )
        np.testing.assert_allclose(boundary.prediction, boundary.field, atol=1e-7)
        automatic = trajectory.loss(
            self.params,
            self.targets,
            self.start,
            self.actions,
            jax.random.key(8),
            return_details=True,
        )
        np.testing.assert_allclose(automatic.r[0], automatic.t[0])
        self.assertTrue(bool(jnp.all(automatic.r <= automatic.t)))

    def test_stopped_derivative_matches_fixed_surrogate_and_finite_difference(self):
        base = self.details()

        def objective(params):
            return trajectory.loss(
                params,
                self.targets,
                self.start,
                self.actions,
                jax.random.key(5),
                noise=self.noise,
                r=self.r,
                t=self.t,
            )

        def fixed_surrogate(params):
            u = trajectory.outputs(
                params, base.interpolated, self.start, self.actions, base.r, base.t
            )[0]
            v = trajectory.outputs(
                params, base.interpolated, self.start, self.actions, base.t, base.t
            )[1]
            corrected = u + (base.t - base.r) * base.jvp
            return jnp.mean(
                jnp.sum((corrected - base.regression_target) ** 2, axis=(1, 2))
                / base.adaptive_weight_u
                + jnp.sum((v - base.regression_target) ** 2, axis=(1, 2))
                / base.adaptive_weight_v
            )

        gradients = jax.grad(objective)(self.params)
        actual, unravel = ravel_pytree(gradients)
        expected, _ = ravel_pytree(jax.grad(fixed_surrogate)(self.params))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
        self.assertTrue(np.isfinite(np.asarray(actual)).all())
        self.assertGreater(float(jnp.linalg.norm(actual)), 1e-3)
        for subtree in (
            gradients["u"],
            gradients["v"],
            gradients["blocks"],
            gradients["action"],
        ):
            flattened, _ = ravel_pytree(subtree)
            self.assertGreater(float(jnp.linalg.norm(flattened)), 1e-5)
        flat_params, unravel = ravel_pytree(self.params)
        direction = actual / jnp.linalg.norm(actual)
        epsilon = 1e-3
        finite_difference = (
            fixed_surrogate(unravel(flat_params + epsilon * direction))
            - fixed_surrogate(unravel(flat_params - epsilon * direction))
        ) / (2 * epsilon)
        np.testing.assert_allclose(
            finite_difference, jnp.vdot(actual, direction), rtol=2e-3, atol=2e-4
        )
        # A finite difference of the varying-stop objective is a negative
        # control, not a valid AD oracle. Adaptive power one is near constant.
        varying_stops_fd = (
            objective(unravel(flat_params + epsilon * direction))
            - objective(unravel(flat_params - epsilon * direction))
        ) / (2 * epsilon)
        self.assertGreater(abs(float(finite_difference - varying_stops_fd)), 1e-2)
        no_auxiliary = jax.grad(
            lambda p: trajectory.loss(
                p,
                self.targets,
                self.start,
                self.actions,
                jax.random.key(5),
                noise=self.noise,
                r=self.r,
                t=self.t,
                velocity_scale=0.0,
            )
        )(self.params)
        # The v head enters the main objective only through the stopped JVP.
        for leaf in jax.tree_util.tree_leaves(no_auxiliary["v"]):
            np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

    def test_sampling_is_flow_time_refinement_not_physical_time_recursion(self):
        original = trajectory.outputs
        for steps in (1, 2, 4):
            with mock.patch.object(trajectory, "outputs", wraps=original) as called:
                actual = trajectory.sample(
                    self.params, self.start, self.actions, self.noise, steps
                )
            self.assertEqual(called.call_count, steps)
            for index, call in enumerate(called.call_args_list):
                self.assertEqual(call.args[1].shape, self.noise.shape)
                self.assertIs(call.args[2], self.start)
                self.assertIs(call.args[3], self.actions)
                self.assertAlmostEqual(call.args[4], 1.0 - (index + 1) / steps)
                self.assertAlmostEqual(call.args[5], 1.0 - index / steps)
            manual = self.noise
            for index in range(steps):
                r, t = 1.0 - (index + 1) / steps, 1.0 - index / steps
                manual = (
                    manual
                    - (t - r)
                    * original(self.params, manual, self.start, self.actions, r, t)[0]
                )
            np.testing.assert_allclose(actual, manual, atol=1e-6)
            compiled = jax.jit(lambda p, s, a, n: trajectory.sample(p, s, a, n, steps))(
                self.params, self.start, self.actions, self.noise
            )
            np.testing.assert_allclose(compiled, manual, rtol=2e-5, atol=1e-6)

    def test_analytical_correlated_gaussian_transport_and_jvp_identity(self):
        params = trajectory.init(
            jax.random.key(21), 2, 1, horizon=3, width=8, heads=2, layers=1, ff_width=16
        )
        matrix = jnp.asarray([[1.2, 0.4, 0.0], [0.4, 0.9, 0.3], [0.0, 0.3, 1.5]])
        covariance = jnp.kron(matrix, jnp.asarray([[1.0, 0.2], [0.2, 0.7]]))
        eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)

        def mean(start, actions):
            return start[:, None] + jnp.concatenate((actions, -0.5 * actions), axis=-1)

        def oracle(params, x, start, actions, r, t):
            del params
            r, t = trajectory._time(r, x).reshape(-1, 1), trajectory._time(
                t, x
            ).reshape(-1, 1)
            mu = mean(start, actions).reshape(x.shape[0], -1)
            flat = x.reshape(x.shape[0], -1)
            centered_eigen = (flat - (1.0 - t) * mu) @ eigenvectors
            variance_t = (1.0 - t) ** 2 * eigenvalues + t**2
            variance_r = (1.0 - r) ** 2 * eigenvalues + r**2
            at_r = (1.0 - r) * mu + (
                centered_eigen * jnp.sqrt(variance_r / variance_t)
            ) @ eigenvectors.T
            velocity = (
                -mu
                + (centered_eigen * (t - (1.0 - t) * eigenvalues) / variance_t)
                @ eigenvectors.T
            )
            delta = t - r
            average = (flat - at_r) / jnp.where(delta == 0, 1.0, delta)
            average = jnp.where(delta == 0, velocity, average)
            return average.reshape(x.shape), velocity.reshape(x.shape)

        batch = 4096
        start = jnp.broadcast_to(jnp.asarray([[0.4, -0.2]]), (batch, 2))
        actions = jnp.broadcast_to(jnp.asarray([[[0.3], [-0.2], [0.7]]]), (batch, 3, 1))
        noise = jax.random.normal(jax.random.key(22), (batch, 3, 2))
        expected = (
            mean(start, actions).reshape(batch, -1)
            + ((noise.reshape(batch, -1) @ eigenvectors) * jnp.sqrt(eigenvalues))
            @ eigenvectors.T
        )
        with mock.patch.object(trajectory, "outputs", oracle):
            for steps in (1, 2, 4):
                sample = trajectory.sample(
                    params, start, actions, noise, steps
                ).reshape(batch, -1)
                np.testing.assert_allclose(sample, expected, rtol=2e-5, atol=3e-6)
            details = trajectory.loss(
                params,
                expected[:4].reshape(4, 3, 2),
                start[:4],
                actions[:4],
                jax.random.key(23),
                noise=noise[:4],
                r=0.17,
                t=0.73,
                return_details=True,
            )
        np.testing.assert_allclose(
            details.prediction, details.marginal_velocity, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            jnp.mean(sample, axis=0), mean(start, actions)[0].reshape(-1), atol=0.065
        )
        np.testing.assert_allclose(
            np.cov(np.asarray(sample), rowvar=False), covariance, rtol=0.1, atol=0.06
        )


class ControlledLearningSmokeTests(unittest.TestCase):
    def test_gaussian_variance_learns_beyond_backbone_width(self):
        import optax

        features, width, horizon = 12, 4, 2
        target_variance = 0.25
        params = trajectory.init(
            jax.random.key(40),
            features,
            1,
            horizon=horizon,
            width=width,
            heads=1,
            layers=1,
            ff_width=16,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(3e-3, weight_decay=1e-4)
        )
        state = optimizer.init(params)
        starts, actions = jnp.zeros((64, features)), jnp.zeros((64, horizon, 1))

        @jax.jit
        def update(params, state, key):
            target_key, loss_key = jax.random.split(key)
            targets = 0.5 * jax.random.normal(target_key, (64, horizon, features))
            value, gradient = jax.value_and_grad(trajectory.loss)(
                params, targets, starts, actions, loss_key
            )
            updates, state = optimizer.update(gradient, state, params)
            return optax.apply_updates(params, updates), state, value

        test_noise = jax.random.normal(jax.random.key(41), (4096, horizon, features))

        @jax.jit
        def samples(params):
            return trajectory.sample(
                params,
                jnp.zeros((4096, features)),
                jnp.zeros((4096, horizon, 1)),
                test_noise,
            ).reshape(4096, horizon * features)

        before = np.cov(np.asarray(samples(params)), rowvar=False)
        before_error = float(np.mean(np.abs(np.diag(before) - target_variance)))
        for index in range(6000):
            params, state, value = update(
                params, state, jax.random.fold_in(jax.random.key(42), index)
            )
        generated = np.asarray(samples(params))
        covariance = np.cov(generated, rowvar=False)
        after_error = float(np.mean(np.abs(np.diag(covariance) - target_variance)))
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
        print(
            f"BEYOND_WIDTH_VARIANCE features={features} width={width} before_error={before_error:.6f} after_error={after_error:.6f} minimum_eigenvalue={minimum_eigenvalue:.6f}",
            flush=True,
        )
        self.assertTrue(np.isfinite(float(value)))
        self.assertLess(after_error, 0.35 * before_error)
        self.assertLess(after_error, 0.08)
        self.assertGreater(minimum_eigenvalue, 0.08)
        self.assertLess(float(np.max(np.abs(np.mean(generated, axis=0)))), 0.10)
        off_diagonal = covariance - np.diag(np.diag(covariance))
        self.assertLess(float(np.max(np.abs(off_diagonal))), 0.08)

    def test_action_conditioned_small_system_actually_learns(self):
        import optax

        def system(starts, actions):
            carry = starts
            future = []
            for index in range(actions.shape[1]):
                action = actions[:, index, 0]
                carry = jnp.stack(
                    (
                        0.7 * carry[:, 0] + 0.65 * action,
                        0.4 * carry[:, 1] - 0.35 * carry[:, 0] + 0.45 * action,
                    ),
                    axis=-1,
                )
                future.append(carry)
            return jnp.stack(future, axis=1)

        train_start = jax.random.uniform(
            jax.random.key(31), (256, 2), minval=-1.0, maxval=1.0
        )
        train_actions = jax.random.uniform(
            jax.random.key(32), (256, 3, 1), minval=-1.0, maxval=1.0
        )
        train_targets = system(train_start, train_actions)
        test_start = jax.random.uniform(
            jax.random.key(33), (32, 2), minval=-1.0, maxval=1.0
        )
        test_actions = jax.random.uniform(
            jax.random.key(34), (32, 3, 1), minval=-1.0, maxval=1.0
        )
        truth = system(test_start, test_actions)
        noise = jax.random.normal(jax.random.key(35), (8, 32, 3, 2))
        params = trajectory.init(
            jax.random.key(36),
            2,
            1,
            horizon=3,
            width=24,
            heads=2,
            layers=1,
            ff_width=48,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(3e-3, weight_decay=1e-4)
        )
        state = optimizer.init(params)

        @jax.jit
        def update(params, state, key):
            index_key, loss_key = jax.random.split(key)
            index = jax.random.randint(index_key, (32,), 0, train_start.shape[0])
            value, gradient = jax.value_and_grad(trajectory.loss)(
                params,
                train_targets[index],
                train_start[index],
                train_actions[index],
                loss_key,
            )
            updates, state = optimizer.update(gradient, state, params)
            return optax.apply_updates(params, updates), state, value

        @jax.jit
        def prediction(params, actions):
            return jnp.mean(
                jax.vmap(lambda n: trajectory.sample(params, test_start, actions, n))(
                    noise
                ),
                axis=0,
            )

        before = float(jnp.mean((prediction(params, test_actions) - truth) ** 2))
        for index in range(650):
            params, state, value = update(
                params, state, jax.random.fold_in(jax.random.key(37), index)
            )
        after_prediction = prediction(params, test_actions)
        after = float(jnp.mean((after_prediction - truth) ** 2))
        wrong_actions = float(
            jnp.mean((prediction(params, test_actions[::-1]) - truth) ** 2)
        )
        persistence = float(jnp.mean((test_start[:, None] - truth) ** 2))
        print(
            f"CONTROLLED_LEARNING before={before:.6f} after={after:.6f} shuffled_actions={wrong_actions:.6f} persistence={persistence:.6f}",
            flush=True,
        )
        self.assertTrue(np.isfinite(float(value)))
        self.assertLess(after, 0.30 * before)
        self.assertLess(after, 0.40 * persistence)
        self.assertGreater(wrong_actions, 2.0 * after)


if __name__ == "__main__":
    unittest.main()
