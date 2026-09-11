from __future__ import annotations

from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Keep this focused test file runnable against an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imf_dreamer_jax.imf import improved_meanflow_loss, init_imf
from imf_dreamer_jax.trajectory import (
    causal_corrupted_history_contexts,
    corrupt_trajectory,
    jit_trajectory_imf_loss,
    naive_joint_context_jvp,
    sample_trajectory_schedule,
    sample_trajectory_time_pairs,
    trajectory_imf_loss,
    trajectory_partial_jvp,
)


def _linear_cross_term_params() -> dict[str, object]:
    """u=z+2*previous_z+5*previous_t+3*t and v=1."""

    weight = jnp.zeros((6, 2), dtype=jnp.float32)
    weight = weight.at[0, 0].set(1.0)  # current z
    weight = weight.at[1, 0].set(2.0)  # previous z in causal context
    weight = weight.at[2, 0].set(5.0)  # previous time in causal context
    weight = weight.at[5, 0].set(3.0)  # current t
    bias = jnp.asarray((0.0, 1.0), dtype=jnp.float32)
    return {"layers": ({"weight": weight, "bias": bias},)}


class TrajectoryIMFCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = 3
        self.steps = 4
        self.features = 2
        self.condition_features = 3
        self.params = init_imf(
            jax.random.key(1),
            sample_dim=self.features,
            condition_dim=self.condition_features,
            hidden_dim=9,
            depth=2,
        )
        self.targets = jax.random.normal(
            jax.random.key(2), (self.batch, self.steps, self.features)
        )
        self.conditions = jax.random.normal(
            jax.random.key(3),
            (self.batch, self.steps, self.condition_features),
        )
        self.noise = jax.random.normal(jax.random.key(4), self.targets.shape)
        self.r = jnp.linspace(0.05, 0.35, self.batch * self.steps).reshape(
            self.batch, self.steps, 1
        )
        self.t = jnp.linspace(0.55, 0.95, self.batch * self.steps).reshape(
            self.batch, self.steps, 1
        )

    def test_matches_flattened_scalar_conditional_imf(self) -> None:
        trajectory = trajectory_imf_loss(
            self.params,
            self.targets,
            self.conditions,
            jax.random.key(5),
            noise=self.noise,
            r=self.r,
            t=self.t,
            return_details=True,
        )
        flattened = improved_meanflow_loss(
            self.params,
            self.targets.reshape((-1, self.features)),
            self.conditions.reshape((-1, self.condition_features)),
            jax.random.key(5),
            noise=self.noise.reshape((-1, self.features)),
            r=self.r.reshape((-1, 1)),
            t=self.t.reshape((-1, 1)),
            boundary_velocity_supervision=True,
            return_details=True,
        )
        np.testing.assert_allclose(trajectory.loss, flattened.loss, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(
            trajectory.prediction.reshape((-1, self.features)),
            flattened.prediction,
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            trajectory.jvp.reshape((-1, self.features)),
            flattened.jvp,
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            trajectory.supervised_velocity,
            trajectory.marginal_velocity,
            rtol=0.0,
            atol=0.0,
        )

    def test_single_token_reduces_to_ordinary_imf(self) -> None:
        target = self.targets[:1, :1]
        condition = self.conditions[:1, :1]
        noise = self.noise[:1, :1]
        r = self.r[:1, :1]
        t = self.t[:1, :1]
        actual = trajectory_imf_loss(
            self.params,
            target,
            condition,
            jax.random.key(6),
            noise=noise,
            r=r,
            t=t,
        )
        expected = improved_meanflow_loss(
            self.params,
            target[:, 0],
            condition[:, 0],
            jax.random.key(6),
            noise=noise[:, 0],
            r=r[:, 0],
            t=t[:, 0],
            boundary_velocity_supervision=True,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_mask_and_weights_match_flattened_imf(self) -> None:
        mask = jnp.asarray(
            [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=jnp.float32
        )
        weights = jnp.linspace(0.5, 1.5, self.batch * self.steps).reshape(
            self.batch, self.steps
        )
        actual = trajectory_imf_loss(
            self.params,
            self.targets,
            self.conditions,
            jax.random.key(7),
            noise=self.noise,
            r=self.r,
            t=self.t,
            token_mask=mask,
            weights=weights,
            signal_weight_floor=0.7,
            signal_weight_scale=0.4,
            return_details=True,
        )
        expected = improved_meanflow_loss(
            self.params,
            self.targets.reshape((-1, self.features)),
            self.conditions.reshape((-1, self.condition_features)),
            jax.random.key(7),
            noise=self.noise.reshape((-1, self.features)),
            r=self.r.reshape((-1, 1)),
            t=self.t.reshape((-1, 1)),
            weights=(mask * weights).reshape((-1,)),
            signal_weight_floor=0.7,
            signal_weight_scale=0.4,
            boundary_velocity_supervision=True,
        )
        np.testing.assert_allclose(actual.loss, expected, rtol=2e-6, atol=2e-6)
        none = trajectory_imf_loss(
            self.params,
            self.targets,
            self.conditions,
            jax.random.key(7),
            noise=self.noise,
            r=self.r,
            t=self.t,
            token_mask=mask,
            reduction="none",
        )
        self.assertEqual(none.shape, mask.shape)
        np.testing.assert_array_equal(np.asarray(none)[np.asarray(mask) == 0], 0.0)

    def test_public_jitted_loss_accepts_explicit_token_arrays(self) -> None:
        loss = jit_trajectory_imf_loss(
            self.params,
            self.targets,
            self.conditions,
            jax.random.key(8),
            noise=self.noise,
            r=self.r,
            t=self.t,
            token_mask=jnp.ones((self.batch, self.steps)),
        )
        self.assertTrue(bool(jnp.isfinite(loss)))

    def test_public_jitted_loss_can_sample_noise_and_token_times(self) -> None:
        first = jit_trajectory_imf_loss(
            self.params,
            self.targets,
            self.conditions,
            jax.random.key(80),
        )
        second = jit_trajectory_imf_loss(
            self.params,
            self.targets,
            self.conditions,
            jax.random.key(81),
        )
        self.assertTrue(bool(jnp.isfinite(first)))
        self.assertTrue(bool(jnp.isfinite(second)))
        self.assertNotEqual(float(first), float(second))

    def test_empty_token_mask_has_finite_zero_mean_loss(self) -> None:
        loss = trajectory_imf_loss(
            self.params,
            self.targets,
            self.conditions,
            jax.random.key(82),
            noise=self.noise,
            r=self.r,
            t=self.t,
            token_mask=jnp.zeros((self.batch, self.steps)),
        )
        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertEqual(float(loss), 0.0)


class TrajectoryContextTests(unittest.TestCase):
    def test_history_context_is_strictly_causal_and_mask_aware(self) -> None:
        history = jnp.asarray([[[1.0], [20.0], [3.0], [4.0]]])
        times = jnp.asarray([[[0.1], [0.2], [0.3], [0.4]]])
        mask = jnp.asarray([[1.0, 0.0, 1.0, 1.0]])
        base = jnp.asarray([[[10.0], [11.0], [12.0], [13.0]]])
        context = causal_corrupted_history_contexts(
            history, times, base_conditions=base, token_mask=mask
        )
        expected = jnp.asarray(
            [
                [
                    [10.0, 0.0, 0.0, 0.0],
                    [11.0, 1.0, 0.1, 1.0],
                    [12.0, 1.0, 0.1, 1.0],
                    [13.0, 3.0, 0.3, 1.0],
                ]
            ]
        )
        np.testing.assert_allclose(context, expected, rtol=0.0, atol=1e-7)

        changed_future = history.at[:, 3].set(999.0)
        changed_context = causal_corrupted_history_contexts(
            changed_future, times, base_conditions=base, token_mask=mask
        )
        np.testing.assert_array_equal(changed_context[:, :4], context[:, :4])

    def test_corruption_helper_uses_per_token_times(self) -> None:
        target = jnp.asarray([[[1.0], [2.0], [3.0]]])
        noise = jnp.asarray([[[5.0], [6.0], [7.0]]])
        times = jnp.asarray([[0.0, 0.5, 1.0]])
        np.testing.assert_allclose(
            corrupt_trajectory(target, noise, times),
            jnp.asarray([[[1.0], [4.0], [7.0]]]),
        )

    def test_explicit_initial_history_is_marked_valid(self) -> None:
        history = jnp.asarray([[[1.0], [2.0]]])
        contexts = causal_corrupted_history_contexts(
            history,
            0.0,
            initial_history=jnp.asarray([[9.0]]),
        )
        np.testing.assert_array_equal(contexts[:, 0], jnp.asarray([[9.0, 0.0, 1.0]]))

    def test_schedules_cover_all_three_context_regimes(self) -> None:
        clean = sample_trajectory_schedule(
            jax.random.key(10), 4, 6, mode="clean_context"
        )
        corrupt = sample_trajectory_schedule(
            jax.random.key(11), 4, 6, mode="corrupted_context"
        )
        suffix = sample_trajectory_schedule(
            jax.random.key(12), 4, 6, mode="future_suffix"
        )
        self.assertTrue(bool(jnp.all(clean.history_t == 0.0)))
        self.assertTrue(bool(jnp.all(clean.loss_mask == 1.0)))
        self.assertTrue(bool(jnp.any(corrupt.history_t > 0.0)))
        self.assertTrue(bool(jnp.all(corrupt.loss_mask == 1.0)))
        for row, start in enumerate(np.asarray(suffix.suffix_start)):
            self.assertTrue(bool(jnp.all(suffix.history_t[row, :start] == 0.0)))
            self.assertTrue(bool(jnp.all(suffix.loss_mask[row, :start] == 0.0)))
            self.assertTrue(bool(jnp.all(suffix.loss_mask[row, start:] == 1.0)))
        for schedule in (clean, corrupt, suffix):
            self.assertEqual(schedule.r.shape, (4, 6, 1))
            self.assertTrue(bool(jnp.all(schedule.r <= schedule.t)))

        mixed = sample_trajectory_schedule(jax.random.key(13), 64, 6, mode="mixed")
        self.assertTrue(set(np.asarray(mixed.pattern).tolist()).issubset({0, 1, 2}))
        self.assertGreaterEqual(len(set(np.asarray(mixed.pattern).tolist())), 2)

    def test_time_pair_helper_is_independent_and_ordered(self) -> None:
        r, t = sample_trajectory_time_pairs(
            jax.random.key(14), 32, 7, boundary_fraction=0.25
        )
        self.assertEqual(r.shape, (32, 7, 1))
        self.assertTrue(bool(jnp.all((0.0 < r) & (r <= t) & (t < 1.0))))
        self.assertGreater(float(jnp.std(t[..., 0], axis=1).mean()), 0.01)

    def test_suffix_schedule_starts_on_a_valid_token_under_padding(self) -> None:
        mask = jnp.asarray(
            [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
            dtype=jnp.float32,
        )
        schedule = sample_trajectory_schedule(
            jax.random.key(15),
            3,
            5,
            mode="future_suffix",
            token_mask=mask,
        )
        for row in (0, 1):
            start = int(schedule.suffix_start[row])
            self.assertEqual(float(mask[row, start]), 1.0)
            self.assertGreater(float(jnp.sum(schedule.loss_mask[row])), 0.0)
        self.assertEqual(float(jnp.sum(schedule.loss_mask[2])), 0.0)


class TrajectoryJVPTests(unittest.TestCase):
    def test_partial_jvp_holds_context_fixed_and_joint_control_does_not(self) -> None:
        params = _linear_cross_term_params()
        z = jnp.asarray([[[1.0], [2.0], [3.0]]], dtype=jnp.float32)
        r = jnp.zeros((1, 3, 1), dtype=jnp.float32)
        t = jnp.asarray([[[0.2], [0.4], [0.6]]], dtype=jnp.float32)
        context = causal_corrupted_history_contexts(z, t)
        correct = trajectory_partial_jvp(params, z, context, r, t)
        naive = naive_joint_context_jvp(params, z, None, r, t)

        np.testing.assert_allclose(correct.field, naive.field, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(correct.marginal_velocity, 1.0, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(correct.jvp, 4.0, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            naive.jvp,
            jnp.asarray([[[4.0], [11.0], [11.0]]]),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertGreater(float(jnp.max(jnp.abs(naive.jvp - correct.jvp))), 6.0)

    def test_vectorized_partial_jvp_matches_tokenwise_manual_jvp(self) -> None:
        batch, steps, features, condition_features = 2, 3, 2, 4
        params = init_imf(
            jax.random.key(20), features, condition_features, hidden_dim=7, depth=2
        )
        z = jax.random.normal(jax.random.key(21), (batch, steps, features))
        condition = jax.random.normal(
            jax.random.key(22), (batch, steps, condition_features)
        )
        r = jnp.full((batch, steps, 1), 0.2)
        t = jnp.full((batch, steps, 1), 0.7)
        actual = trajectory_partial_jvp(params, z, condition, r, t)

        manual = []
        for batch_index in range(batch):
            row = []
            for step_index in range(steps):
                z_token = z[batch_index : batch_index + 1, step_index]
                c_token = condition[batch_index : batch_index + 1, step_index]
                r_token = r[batch_index : batch_index + 1, step_index]
                t_token = t[batch_index : batch_index + 1, step_index]
                token_result = trajectory_partial_jvp(
                    params,
                    z_token[:, None],
                    c_token[:, None],
                    r_token[:, None],
                    t_token[:, None],
                )
                row.append(token_result.jvp[0, 0])
            manual.append(jnp.stack(row))
        np.testing.assert_allclose(actual.jvp, jnp.stack(manual), rtol=2e-6, atol=2e-6)


class TrajectoryJITTests(unittest.TestCase):
    def test_loss_and_parameter_gradients_are_finite_for_every_pattern(self) -> None:
        batch, steps, features, base_features = 2, 4, 2, 2
        targets = jax.random.normal(jax.random.key(30), (batch, steps, features))
        token_noise = jax.random.normal(jax.random.key(31), targets.shape)
        base = jax.random.normal(jax.random.key(33), (batch, steps, base_features))
        condition_features = base_features + features + 2
        params = init_imf(
            jax.random.key(34), features, condition_features, hidden_dim=8, depth=2
        )

        def objective(params_value, schedule):
            corrupted_history = corrupt_trajectory(
                targets, token_noise, schedule.history_t
            )
            conditions = causal_corrupted_history_contexts(
                corrupted_history, schedule.history_t, base_conditions=base
            )
            return trajectory_imf_loss(
                params_value,
                targets,
                conditions,
                jax.random.key(35),
                noise=token_noise,
                r=schedule.r,
                t=schedule.t,
                token_mask=schedule.loss_mask,
            )

        compiled = jax.jit(jax.value_and_grad(objective))
        for index, mode in enumerate(
            ("clean_context", "corrupted_context", "future_suffix")
        ):
            schedule = sample_trajectory_schedule(
                jax.random.key(40 + index), batch, steps, mode=mode
            )
            loss, gradients = compiled(params, schedule)
            self.assertTrue(np.isfinite(np.asarray(loss)).all(), mode)
            leaves = jax.tree_util.tree_leaves(gradients)
            self.assertTrue(
                all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves), mode
            )
            self.assertGreater(
                sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves), 0.0, mode
            )


def tearDownModule() -> None:
    print("TRAJECTORY_IMF_UNIT_TESTS_OK")


if __name__ == "__main__":
    unittest.main()
