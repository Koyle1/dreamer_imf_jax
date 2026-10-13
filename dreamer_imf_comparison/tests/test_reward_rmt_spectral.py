import copy
import json
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dreamer_imf_compare import reward_rmt_spectral as spectral
from dreamer_imf_compare.reward_readout_study import mlp_logits


def tiny_params():
    return (
        {
            "w": jnp.array([[0.4, 0.1, -0.2], [0.3, -0.5, 0.7]]),
            "b": jnp.array([0.1, 0.2, 0.3]),
        },
        {
            "w": jnp.array([[0.2, -0.4, 0.3], [0.6, 0.2, 0.5], [-0.3, 0.7, -0.4]]),
            "b": jnp.array([0.1, -0.1, 0.2]),
        },
    )


def flat_gradient(tree):
    return np.concatenate(
        [np.asarray(layer[k]).reshape(-1) for layer in tree for k in ("w", "b")]
    )


class SpectralTests(unittest.TestCase):
    def test_exact_spectrum_and_reward_signal_in_low_and_high_variance(self):
        rng = np.random.default_rng(31)
        q, _ = np.linalg.qr(np.column_stack([np.ones(80), rng.normal(size=(80, 3))]))
        x = q[:, 1:] * np.array([10, 1, 0.1]) * np.sqrt(80)
        ep, mode = np.repeat(np.arange(8), 10), np.repeat([0, 1], 40)
        for component in (0, 2):
            report, arrays = spectral.analyze_features(
                x, x[:, component], ep, mode, seed=5, resamples=4, permutations=8
            )
            np.testing.assert_allclose(
                arrays["eigenvalues"], [100, 1, 0.01], atol=1e-12
            )
            self.assertAlmostEqual(report["linear_reward_r2"], 1.0)
            expected = np.zeros(3)
            expected[component] = 1
            np.testing.assert_allclose(
                arrays["component_linear_reward_r2"], expected, atol=1e-12
            )
            self.assertLess(report["spectrum"]["entropy_effective_rank"], 1.1)
            self.assertIn(2, arrays["stability_component_indices"])

    def test_cross_episode_alignment_matches_dense_reference(self):
        rng = np.random.default_rng(18)
        x = rng.normal(size=(24, 5))
        y = rng.integers(0, 3, 24)
        ep = np.repeat(np.arange(4), 6)
        r, _ = spectral.analyze_features(
            x, y, ep, np.zeros(24), seed=1, resamples=0, permutations=0
        )
        a, b = x - x.mean(0), y - y.mean()
        gram, target = a @ a.T, np.outer(b, b)
        outside = ep[:, None] != ep[None]
        expected = (gram[outside] @ target[outside]) / (
            np.linalg.norm(gram[outside]) * np.linalg.norm(target[outside])
        )
        self.assertAlmostEqual(r["cross_episode_alignment"], expected, places=12)

    def test_episode_permutation_preserves_correlations_modes_and_null_control(self):
        rng = np.random.default_rng(27)
        ep = np.repeat(np.arange(20), 8)
        mode = np.repeat(np.arange(2), 80)
        episode_label = np.tile(np.arange(10) % 3, 2)
        y = np.repeat(episode_label, 8).astype(float)
        x = np.column_stack([y, rng.normal(size=len(y)) * 0.01])
        before = copy.deepcopy((x, y, ep, mode))
        r, a = spectral.analyze_features(
            x, y, ep, mode, seed=91, resamples=12, permutations=100
        )
        self.assertLess(r["episode_block_permutation_p"], 0.03)
        self.assertGreater(np.median(a["bootstrap_reward_moment_cosine"]), 0.99)
        for mapping in a["permutation_episode_indices"]:
            np.testing.assert_array_equal(
                mode.reshape(20, 8)[:, 0], mode.reshape(20, 8)[mapping, 0]
            )
            np.testing.assert_array_equal(np.sort(mapping[:10]), np.arange(10))
            perm_y = y.reshape(20, 8)[mapping]
            self.assertTrue(np.all(np.diff(perm_y, axis=1) == 0))
        # A deliberately block-shuffled label relation must fail the positive
        # signal check, while individual transition shuffling is never used.
        shuffled = np.repeat(episode_label[rng.permutation(20)], 8)
        null, _ = spectral.analyze_features(
            x, shuffled, ep, mode, seed=91, resamples=0, permutations=100
        )
        self.assertGreater(null["episode_block_permutation_p"], 0.05)
        for actual, original in zip((x, y, ep, mode), before):
            np.testing.assert_array_equal(actual, original)
        rerun, arrays = spectral.analyze_features(
            x, y, ep, mode, seed=91, resamples=12, permutations=100
        )
        self.assertEqual(r, rerun)
        for key in a:
            np.testing.assert_array_equal(a[key], arrays[key])

    def test_degenerate_and_invalid_feature_inputs(self):
        r, a = spectral.analyze_features(
            np.ones((4, 3)),
            np.ones(4),
            np.arange(4),
            np.zeros(4),
            seed=0,
            resamples=2,
            permutations=2,
        )
        self.assertTrue(r["spectrum"]["degenerate"])
        self.assertIsNone(r["alignment"])
        self.assertIsNone(r["episode_block_permutation_p"])

        self.assertEqual(a["bootstrap_projected_variance"].shape, (2, 0))
        json.dumps(r, allow_nan=False)
        for value in a.values():
            self.assertTrue(np.isfinite(value).all())
        with self.assertRaisesRegex(ValueError, "mode"):
            spectral.analyze_features(
                np.ones((4, 3)),
                np.ones(4),
                np.zeros(4),
                np.arange(4),
                seed=0,
                resamples=2,
                permutations=2,
            )
        with self.assertRaises(ValueError):
            spectral.analyze_features(
                np.full((4, 3), np.nan),
                np.ones(4),
                np.arange(4),
                np.zeros(4),
                seed=0,
                resamples=2,
                permutations=2,
            )
        r, a = spectral.analyze_features(
            np.arange(8).reshape(4, 2),
            np.arange(4),
            np.zeros(4),
            np.zeros(4),
            seed=0,
            resamples=0,
            permutations=4,
        )
        self.assertIsNone(r["cross_episode_alignment"])
        self.assertIsNone(r["episode_block_permutation_p"])

    def test_episode_memorization_and_mode_confound_do_not_pass_permutation_control(
        self,
    ):
        ep = np.repeat(np.arange(12), 8)
        mode = np.repeat([0, 1], 48)
        y = np.repeat(np.arange(12) % 3, 8)
        memorizer = np.eye(12)[ep]
        r, _ = spectral.analyze_features(
            memorizer, y, ep, mode, seed=71, resamples=2, permutations=31
        )
        self.assertAlmostEqual(r["linear_reward_r2"], 1.0)
        self.assertEqual(r["episode_block_permutation_p"], 1.0)
        r, _ = spectral.analyze_features(
            mode[:, None], mode, ep, mode, seed=71, resamples=2, permutations=31
        )
        self.assertAlmostEqual(r["alignment"], 1.0)
        self.assertEqual(r["episode_block_permutation_p"], 1.0)

    def test_traced_kernel_equals_all_logit_jacobians(self):
        params = tiny_params()
        x = np.array([[0.4, 0.8], [-0.7, 0.5], [0.1, -0.8], [0.0, 0.0]], np.float32)
        jacobians = []
        for row in x:
            jacobians.append(
                np.stack(
                    [
                        flat_gradient(
                            jax.grad(lambda p: mlp_logits(p, jnp.asarray(row))[c])(
                                params
                            )
                        )
                        for c in range(3)
                    ]
                )
            )
        dense = np.einsum(
            "ncp,mcp->nm",
            np.asarray(jacobians, np.float64),
            np.asarray(jacobians, np.float64),
        )
        report, arrays = spectral.kernel_diagnostics(
            params, x, np.arange(4) % 3, np.array([0, 0, 1, 1])
        )
        np.testing.assert_allclose(arrays["kernel"], dense, rtol=2e-7, atol=1e-8)
        self.assertGreaterEqual(np.linalg.eigvalsh(arrays["kernel"]).min(), -1e-10)
        self.assertEqual(report["classes"], 3)
        with self.assertRaisesRegex(ValueError, "256"):
            spectral.kernel_diagnostics(
                params, np.ones((257, 2)), np.zeros(257), np.arange(257)
            )

    def test_gradpca_matches_dense_centered_class_mean_reference(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(21, 2)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 7 + [2] * 4)
        params = tiny_params()
        gradients = np.stack(
            [
                flat_gradient(jax.grad(lambda p: jnp.max(mlp_logits(p, row)))(params))
                for row in x
            ]
        ).astype(np.float64)
        means = np.stack([gradients[y == k].mean(0) for k in range(3)])
        mean = gradients.mean(0)
        _, s, vh = np.linalg.svd(means - mean, full_matrices=False)
        rank = np.sum(s > s[0] * 1e-5)
        state = spectral.fit_gradpca(params, x, y)
        self.assertEqual(int(state["numerical_rank"]), rank)
        self.assertLessEqual(rank, 2)
        np.testing.assert_allclose(state["mean"], mean, rtol=2e-7, atol=2e-8)
        np.testing.assert_allclose(
            state["basis"] @ state["basis"].T, np.eye(rank), atol=1e-10
        )
        expected = np.linalg.norm(
            (gradients - mean) @ vh[:rank].T, axis=1
        ) / np.linalg.norm(gradients - mean, axis=1)
        np.testing.assert_allclose(
            spectral.score_gradpca(params, x, state), expected, rtol=1e-5, atol=1e-6
        )

    def test_gradpca_ties_and_rank_zero_are_explicit_and_finite(self):
        params = ({"w": jnp.zeros((2, 3)), "b": jnp.zeros(3)},)
        x = np.ones((6, 2), np.float32)
        state = spectral.fit_gradpca(params, x, np.arange(6) % 3)
        self.assertTrue(state["degenerate"])
        self.assertEqual(int(state["numerical_rank"]), 0)
        expected = flat_gradient(
            jax.grad(lambda p: jnp.max(mlp_logits(p, jnp.ones(2))))(params)
        )
        np.testing.assert_allclose(state["mean"], expected)
        np.testing.assert_array_equal(
            spectral.score_gradpca(params, x, state), np.zeros(6)
        )
        one = spectral.fit_gradpca(tiny_params(), x, np.zeros(6))
        self.assertEqual(int(one["rank_cap"]), 0)
        with self.assertRaisesRegex(ValueError, "classes"):
            spectral.fit_gradpca(params, x, np.full(6, 3))

    def test_nearest_distance_blocked_matches_dense_and_is_train_only(self):
        rng = np.random.default_rng(81)
        train, query = rng.normal(size=(520, 7)), rng.normal(size=(68, 7))
        expected = np.sqrt(
            np.mean((query[:, None] - train[None]) ** 2, axis=-1).min(-1)
        )
        np.testing.assert_allclose(
            spectral.nearest_distance(train, query), expected, atol=1e-14
        )
        np.testing.assert_allclose(
            spectral.nearest_distance(train, train[:10]), 0, atol=2e-8
        )
        with self.assertRaises(ValueError):
            spectral.nearest_distance(train, np.zeros((2, 8)))


if __name__ == "__main__":
    unittest.main()
