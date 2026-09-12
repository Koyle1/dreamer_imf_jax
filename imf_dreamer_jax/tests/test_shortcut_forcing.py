from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Keep this focused test runnable against an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imf_dreamer_jax.shortcut import (
    sample_shortcut_schedule,
    sample_shortcut_steps,
    shortcut_forcing_loss,
    shortcut_schedule_is_valid,
)


def _affine_predict_jax(z: jax.Array, tau: jax.Array, step: jax.Array) -> jax.Array:
    feature_bias = jnp.arange(z.shape[-1], dtype=z.dtype) * 0.17
    return 0.6 * z + 0.2 * tau - 0.3 * step + feature_bias


def _affine_predict_numpy(
    z: np.ndarray, tau: np.ndarray, step: np.ndarray
) -> np.ndarray:
    feature_bias = np.arange(z.shape[-1], dtype=z.dtype) * 0.17
    return 0.6 * z + 0.2 * tau - 0.3 * step + feature_bias


def _numpy_equation_seven(
    targets: np.ndarray,
    noise: np.ndarray,
    tau: np.ndarray,
    step: np.ndarray,
    *,
    k_max: int,
) -> dict[str, np.ndarray]:
    """Independent NumPy transcription of Dreamer 4 Equation (7)."""

    corrupted = (1.0 - tau) * noise + tau * targets
    prediction = _affine_predict_numpy(corrupted, tau, step)
    half = step / 2.0
    first_prediction = _affine_predict_numpy(corrupted, tau, half)
    first_velocity = (first_prediction - corrupted) / (1.0 - tau)
    intermediate = corrupted + first_velocity * half
    midpoint = tau + half
    second_prediction = _affine_predict_numpy(intermediate, midpoint, half)
    second_velocity = (second_prediction - intermediate) / (1.0 - midpoint)
    target_velocity = (first_velocity + second_velocity) / 2.0
    prediction_velocity = (prediction - corrupted) / (1.0 - tau)
    flow = np.sum(np.square(prediction - targets), axis=-1)
    bootstrap = np.square(1.0 - tau[..., 0]) * np.sum(
        np.square(prediction_velocity - target_velocity), axis=-1
    )
    branch = np.where(step[..., 0] == 1.0 / k_max, flow, bootstrap)
    ramp = 0.9 * tau[..., 0] + 0.1
    return {
        "corrupted": corrupted,
        "prediction": prediction,
        "first_velocity": first_velocity,
        "intermediate": intermediate,
        "second_velocity": second_velocity,
        "target_velocity": target_velocity,
        "flow": flow,
        "bootstrap": bootstrap,
        "branch": branch,
        "per_token": ramp * branch,
    }


class ShortcutScheduleTests(unittest.TestCase):
    def test_k_max_is_required_and_must_be_power_of_two(self) -> None:
        for function in (sample_shortcut_schedule, shortcut_forcing_loss):
            parameter = inspect.signature(function).parameters["k_max"]
            self.assertIs(parameter.default, inspect.Parameter.empty)
        for invalid in (True, 0, -2, 3, 12):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sample_shortcut_schedule(jax.random.key(0), 2, 3, k_max=invalid)

    def test_tokenwise_schedule_obeys_grid_and_endpoints_under_jit(self) -> None:
        sampler = jax.jit(
            lambda key: sample_shortcut_schedule(key, 64, 32, k_max=8)
        )
        schedule = sampler(jax.random.key(1))
        self.assertTrue(bool(shortcut_schedule_is_valid(schedule, k_max=8)))
        counts = np.asarray(schedule.step_count)
        self.assertTrue(set(np.unique(counts)).issubset({1, 2, 4, 8}))
        self.assertGreaterEqual(len(np.unique(counts)), 3)
        np.testing.assert_allclose(
            np.asarray(schedule.tau * schedule.step_count),
            np.rint(np.asarray(schedule.tau * schedule.step_count)),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertTrue(bool(jnp.all(schedule.tau + schedule.step_size <= 1.0)))
        self.assertTrue(bool(jnp.all(schedule.is_finest == (counts == 8))))

        corrupted = schedule._replace(tau=schedule.tau.at[0, 0, 0].set(0.3))
        self.assertFalse(bool(shortcut_schedule_is_valid(corrupted, k_max=8)))

    def test_single_level_schedule_has_only_noise_endpoint(self) -> None:
        schedule = sample_shortcut_schedule(jax.random.key(2), 3, 5, k_max=1)
        np.testing.assert_array_equal(schedule.step_count, 1)
        np.testing.assert_array_equal(schedule.step_size, 1.0)
        np.testing.assert_array_equal(schedule.tau, 0.0)
        np.testing.assert_array_equal(schedule.is_finest, True)


class ShortcutEquationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = jnp.asarray(
            [
                [[0.2, -0.4], [1.0, 0.3], [-0.1, 0.8], [0.7, -0.2]],
                [[-0.5, 0.1], [0.4, -0.9], [0.6, 0.2], [-0.3, 0.5]],
            ],
            dtype=jnp.float32,
        )
        self.noise = jnp.asarray(
            [
                [[0.9, 0.5], [-0.6, 0.2], [0.3, -0.8], [0.1, 0.4]],
                [[0.2, -0.7], [0.8, 0.6], [-0.4, 0.9], [0.5, -0.1]],
            ],
            dtype=jnp.float32,
        )
        self.tau = jnp.asarray(
            [[0.50, 0.50, 0.00, 0.75], [0.25, 0.00, 0.50, 0.00]],
            dtype=jnp.float32,
        )[..., None]
        self.step = jnp.asarray(
            [[0.25, 0.50, 1.00, 0.25], [0.25, 1.00, 0.50, 0.50]],
            dtype=jnp.float32,
        )[..., None]

    def test_both_branches_match_independent_numpy_reference(self) -> None:
        details = shortcut_forcing_loss(
            _affine_predict_jax,
            self.targets,
            k_max=4,
            support_safe_bootstrap=False,
            noise=self.noise,
            tau=self.tau,
            step_size=self.step,
            reduction="none",
            return_details=True,
        )
        expected = _numpy_equation_seven(
            np.asarray(self.targets),
            np.asarray(self.noise),
            np.asarray(self.tau),
            np.asarray(self.step),
            k_max=4,
        )
        for actual_name, expected_name in (
            ("corrupted", "corrupted"),
            ("prediction", "prediction"),
            ("first_velocity", "first_velocity"),
            ("intermediate", "intermediate"),
            ("second_velocity", "second_velocity"),
            ("bootstrap_target_velocity", "target_velocity"),
            ("flow_loss", "flow"),
            ("bootstrap_loss", "bootstrap"),
            ("branch_loss", "branch"),
            ("loss", "per_token"),
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(details, actual_name)),
                expected[expected_name],
                rtol=2e-6,
                atol=2e-6,
            )
        self.assertTrue(bool(jnp.any(details.is_finest)))
        self.assertTrue(bool(jnp.any(~details.is_finest)))

    def test_mask_weights_and_ramp_have_literal_weighted_mean_semantics(self) -> None:
        mask = jnp.asarray([[1, 1, 0, 0], [1, 0, 1, 1]], dtype=jnp.float32)
        weights = jnp.asarray([[1, 2, 7, 7], [3, 7, 4, 5]], dtype=jnp.float32)

        def objective(targets: jax.Array) -> jax.Array:
            return shortcut_forcing_loss(
                _affine_predict_jax,
                targets,
                k_max=4,
                noise=self.noise,
                tau=self.tau,
                step_size=self.step,
                token_mask=mask,
                weights=weights,
            )

        loss, gradient = jax.jit(jax.value_and_grad(objective))(self.targets)
        raw = shortcut_forcing_loss(
            _affine_predict_jax,
            self.targets,
            k_max=4,
            noise=self.noise,
            tau=self.tau,
            step_size=self.step,
            token_mask=mask,
            weights=weights,
            reduction="none",
            return_details=True,
        )
        expected = jnp.sum(raw.per_token_loss * mask * weights) / jnp.sum(mask * weights)
        np.testing.assert_allclose(loss, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(np.asarray(raw.loss)[np.asarray(mask) == 0], 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        self.assertGreater(float(jnp.sum(jnp.abs(gradient))), 0.0)

        no_ramp = jnp.sum(raw.branch_loss * mask * weights) / jnp.sum(mask * weights)
        self.assertGreater(float(jnp.abs(loss - no_ramp)), 1e-4)

    def test_jitted_loss_can_sample_noise_and_tokenwise_schedule(self) -> None:
        compiled = jax.jit(
            lambda key: shortcut_forcing_loss(
                _affine_predict_jax, self.targets, key, k_max=8
            )
        )
        first = compiled(jax.random.key(40))
        second = compiled(jax.random.key(41))
        self.assertTrue(bool(jnp.isfinite(first)))
        self.assertTrue(bool(jnp.isfinite(second)))
        self.assertNotEqual(float(first), float(second))

    def test_empty_mask_returns_finite_zero(self) -> None:
        result = shortcut_forcing_loss(
            _affine_predict_jax,
            self.targets,
            k_max=4,
            noise=self.noise,
            tau=self.tau,
            step_size=self.step,
            token_mask=jnp.zeros(self.targets.shape[:2]),
        )
        self.assertTrue(bool(jnp.isfinite(result)))
        self.assertEqual(float(result), 0.0)

    def test_bootstrap_teacher_is_stop_gradient(self) -> None:
        targets = jnp.asarray([[[0.4], [-0.2]]], dtype=jnp.float32)
        noise = jnp.asarray([[[1.1], [0.3]]], dtype=jnp.float32)
        tau = jnp.zeros((1, 2, 1), dtype=jnp.float32)
        step = jnp.ones((1, 2, 1), dtype=jnp.float32)

        def predictor(parameter: jax.Array, z: jax.Array, t: jax.Array, d: jax.Array):
            return parameter * (0.7 * z + 0.2 * t + 0.3 * d + 0.5)

        def implementation(parameter: jax.Array) -> jax.Array:
            return shortcut_forcing_loss(
                lambda z, t, d: predictor(parameter, z, t, d),
                targets,
                k_max=2,
                support_safe_bootstrap=False,
                noise=noise,
                tau=tau,
                step_size=step,
            )

        def reference(parameter: jax.Array, *, stop_teacher: bool) -> jax.Array:
            corrupted = noise
            half = step / 2.0
            first = (predictor(parameter, corrupted, tau, half) - corrupted) / (
                1.0 - tau
            )
            intermediate = corrupted + half * first
            midpoint = tau + half
            second = (
                predictor(parameter, intermediate, midpoint, half) - intermediate
            ) / (1.0 - midpoint)
            teacher = (first + second) / 2.0
            if stop_teacher:
                teacher = jax.lax.stop_gradient(teacher)
            predicted = (predictor(parameter, corrupted, tau, step) - corrupted) / (
                1.0 - tau
            )
            return 0.1 * jnp.mean(jnp.square(predicted - teacher))

        parameter = jnp.asarray(0.8)
        actual = jax.jit(jax.grad(implementation))(parameter)
        stopped = jax.grad(lambda p: reference(p, stop_teacher=True))(parameter)
        unstopped = jax.grad(lambda p: reference(p, stop_teacher=False))(parameter)
        np.testing.assert_allclose(actual, stopped, rtol=2e-6, atol=2e-6)
        self.assertGreater(float(jnp.abs(actual - unstopped)), 1e-3)

    def test_second_half_step_rebuilds_updated_causal_sequence(self) -> None:
        targets = self.targets[:1]
        noise = self.noise[:1]
        tau = jnp.zeros((1, 4, 1), dtype=jnp.float32)
        step = jnp.ones((1, 4, 1), dtype=jnp.float32)

        def previous(value: jax.Array) -> jax.Array:
            return jnp.concatenate((jnp.zeros_like(value[:, :1]), value[:, :-1]), axis=1)

        def causal_predict(z: jax.Array, t: jax.Array, d: jax.Array) -> jax.Array:
            # Rebuilding these prefix features on every call is the contract.
            return 0.4 * z + 0.7 * previous(z) + 0.2 * t + 0.5 * previous(t) + 0.1 * d

        details = shortcut_forcing_loss(
            causal_predict,
            targets,
            k_max=2,
            support_safe_bootstrap=False,
            noise=noise,
            tau=tau,
            step_size=step,
            return_details=True,
        )
        expected_second = causal_predict(
            details.intermediate, tau + step / 2.0, step / 2.0
        )
        np.testing.assert_allclose(details.second_prediction, expected_second, atol=1e-6)

        stale_second = (
            0.4 * details.intermediate
            + 0.7 * previous(details.corrupted)
            + 0.2 * (tau + step / 2.0)
            + 0.5 * previous(tau)
            + 0.1 * (step / 2.0)
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(details.second_prediction - stale_second))), 1e-3
        )


class ShortcutSamplerTests(unittest.TestCase):
    def test_jitted_sampler_matches_manual_x_prediction_updates(self) -> None:
        noise = jnp.asarray(
            [[[0.2, -0.3], [0.7, 0.1]], [[-0.4, 0.6], [0.5, -0.2]]],
            dtype=jnp.float32,
        )

        def predictor(z: jax.Array, tau: jax.Array, step: jax.Array) -> jax.Array:
            return 0.25 * z + 0.4 * tau + 0.2 * step + 0.1

        for steps in (1, 2, 4):
            compiled = jax.jit(
                lambda value, count=steps: sample_shortcut_steps(
                    predictor, value, steps=count
                )
            )
            actual = compiled(noise)
            expected = np.asarray(noise)
            d = 1.0 / steps
            for index in range(steps):
                tau = index * d
                clean = 0.25 * expected + 0.4 * tau + 0.2 * d + 0.1
                expected = expected + d * (clean - expected) / (1.0 - tau)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

            wrong_without_x_to_v_conversion = np.asarray(noise).copy()
            for index in range(steps):
                tau = index * d
                clean = (
                    0.25 * wrong_without_x_to_v_conversion
                    + 0.4 * tau
                    + 0.2 * d
                    + 0.1
                )
                wrong_without_x_to_v_conversion += d * (
                    clean - wrong_without_x_to_v_conversion
                )
            if steps > 1:
                self.assertGreater(
                    float(np.max(np.abs(expected - wrong_without_x_to_v_conversion))),
                    1e-3,
                )

    def test_constant_velocity_path_reaches_analytic_endpoint(self) -> None:
        noise = jax.random.normal(jax.random.key(9), (2, 3, 2))
        velocity = jnp.asarray([[[0.3, -0.7]]], dtype=noise.dtype)

        def predictor(z: jax.Array, tau: jax.Array, step: jax.Array) -> jax.Array:
            del step
            return z + (1.0 - tau) * velocity

        expected = noise + velocity
        for steps in (1, 2, 4):
            np.testing.assert_allclose(
                sample_shortcut_steps(predictor, noise, steps=steps),
                expected,
                rtol=2e-6,
                atol=2e-6,
            )

    def test_sampler_rejects_bad_step_count_and_output_shape(self) -> None:
        noise = jnp.zeros((1, 2, 1), dtype=jnp.float32)
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                sample_shortcut_steps(lambda z, t, d: z, noise, steps=invalid)
        with self.assertRaises(ValueError):
            sample_shortcut_steps(lambda z, t, d: z[..., 0], noise, steps=1)


if __name__ == "__main__":
    unittest.main()
