"""Exact fixtures and negative controls for the frozen reward probe contract."""

import copy
import json
import sys
import unittest

import numpy as np

from dreamer_imf_compare.reward_probe_metrics import (
    HORIZONS,
    decomposition,
    fit_affine,
    fit_ridge,
    predict_affine,
    predict_ridge,
    summarize,
)


def fixture():
    episode = np.repeat(np.arange(6), 4)
    plan = np.tile(np.arange(4), 6)
    truth = np.broadcast_to((plan % 3)[:, None], (24, 15)).astype(float).copy()
    return dict(
        pred=truth.copy(),
        truth=truth,
        episode=episode,
        mode=np.where(episode % 2, "random", "policy"),
        plan=plan,
        return_scale=5.0,
        bootstrap_reps=200,
    )


class RidgeTests(unittest.TestCase):
    def test_linear_toy_recoverability_and_shuffled_label_negative_control(self):
        rng = np.random.default_rng(102)
        x = rng.uniform(-1, 1, (240, 4))
        y = 1 + x @ np.array([0.2, -0.3, 0.15, 0.1])
        fit = fit_ridge(x[:160], y[:160], 0.001)
        predicted = predict_ridge(fit, x[160:])
        self.assertLess(np.mean((predicted - y[160:]) ** 2), 1e-10)
        wrong = fit_ridge(x[:160], rng.permutation(y[:160]), 0.001)
        self.assertGreater(
            np.mean((predict_ridge(wrong, x[160:]) - y[160:]) ** 2), 0.02
        )
        self.assertEqual(fit["solver"], "primal")
        json.dumps(fit, allow_nan=False)

    def test_sum_loss_penalty_matches_independent_augmented_least_squares(self):
        rng = np.random.default_rng(14)
        for n, d in ((30, 4), (4, 30)):
            with self.subTest(n=n, d=d):
                x, y = rng.normal(size=(n, d)), rng.normal(size=n)
                alpha = 0.1
                fit = fit_ridge(x, y, alpha)
                standardized = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 0.01)
                design = np.concatenate((standardized, np.sqrt(alpha) * np.eye(d)))
                reference = np.linalg.lstsq(
                    design, np.r_[y - y.mean(), np.zeros(d)], rcond=None
                )[0]
                np.testing.assert_allclose(
                    fit["coefficient"], reference, rtol=1e-10, atol=1e-10
                )
                self.assertEqual(fit["solver"], "primal" if d <= n else "dual")
                duplicate = fit_ridge(
                    np.repeat(x, 2, axis=0), np.repeat(y, 2), 2 * alpha
                )
                np.testing.assert_allclose(
                    duplicate["coefficient"], fit["coefficient"], atol=1e-10
                )

    def test_train_only_coordinate_scales_and_holdout_leakage_refutation(self):
        x = np.array([[0, 5, 0], [1, 5, 0.001], [2, 5, 0.002], [3, 5, 0.003]])
        y = 0.5 + 0.2 * x[:, 0]
        fit = fit_ridge(x, y, 0.01)
        np.testing.assert_allclose(fit["mean"], x.mean(axis=0))
        np.testing.assert_allclose(fit["std"], [x[:, 0].std(), 0.01, 0.01])
        shifted = x + [20, 0, 0]
        before = copy.deepcopy(fit)
        output = predict_ridge(fit, shifted)
        self.assertEqual(fit, before)
        np.testing.assert_allclose(output, 2)
        # Refits on shifted test rows would instead remove the shift entirely.
        leaked = copy.deepcopy(fit)
        leaked["mean"] = shifted.mean(axis=0).tolist()
        self.assertGreater(np.max(abs(predict_ridge(leaked, shifted) - output)), 0.8)
        # Merely seeing extra test rows cannot affect an existing fit/prediction.
        mixed = np.concatenate((shifted, shifted * 100))
        np.testing.assert_allclose(predict_ridge(fit, mixed)[:4], output)

    def test_batched_temporal_inputs_match_flat_fit_and_preserve_shapes(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(8, 15, 6))
        y = 0.7 + 0.1 * x[..., 0]
        batched = fit_ridge(x, y, 0.1)
        flat = fit_ridge(x.reshape(-1, 6), y.reshape(-1), 0.1)
        self.assertEqual(batched, flat)
        self.assertEqual(predict_ridge(batched, x).shape, (8, 15))
        self.assertEqual(batched["n_train_examples"], 120)

    def test_collinear_constant_and_single_example_features_are_finite(self):
        x = np.tile(np.arange(10.0)[:, None], (1, 12))
        x[:, 0] = 1
        for alpha in (0.001, 0.1, 10, 1000):
            fitted = fit_ridge(x, np.linspace(0, 2, 10), alpha)
            self.assertTrue(np.isfinite(predict_ridge(fitted, x)).all())
        constant = fit_ridge(np.ones((10, 5)), np.arange(10) / 5, 0.001)
        np.testing.assert_allclose(predict_ridge(constant, np.ones((2, 5))), 0.9)
        fitted = fit_ridge(np.ones((1, 5)), np.array([1.4]), 0.001)
        np.testing.assert_allclose(predict_ridge(fitted, np.zeros((3, 5))), 1.4)

    def test_full_study_dimensions_stable_with_collinear_and_constant_inputs(self):
        rng = np.random.default_rng(916)
        x = rng.uniform(-1, 1, (9600, 2560))
        x[:, -1] = 5
        x[:, -2] = 100 * x[:, 0]
        y = 1 + 0.2 * x[:, 0] - 0.1 * x[:, 1]
        fitted = fit_ridge(x, y, 0.001)
        error = np.mean((predict_ridge(fitted, x[:128]) - y[:128]) ** 2)
        self.assertLess(error, 1e-12)
        self.assertEqual(fitted["solver"], "primal")
        self.assertEqual(fitted["n_train_examples"], 9600)
        self.assertEqual(len(fitted["coefficient"]), 2560)
        self.assertEqual(fitted["std"][-1], 0.01)

    def test_clipping_and_no_input_mutation(self):
        x = np.array([[-1.0], [0], [1]])
        y = np.array([-0.5, 1, 2.5])
        before_x, before_y = x.copy(), y.copy()
        fitted = fit_ridge(x, y, 0.001)
        prediction = predict_ridge(fitted, x * 100)
        np.testing.assert_allclose(prediction, [0, 1, 2])
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(y, before_y)

    def test_bad_shapes_nonfinite_parameters_and_penalties_fail_closed(self):
        x, y = np.ones((5, 3)), np.ones(5)
        for value in (0, -1, np.inf, np.nan, True, [1], "bad", 1j):
            with self.subTest(alpha=value), self.assertRaises(ValueError):
                fit_ridge(x, y, value)
        for bad_x, bad_y in (
            (x[:, 0], y),
            (x, y[:3]),
            (x, y[:, None]),
            (x[:0], y[:0]),
            (x * np.nan, y),
            (x + 1j, y),
            (x, y * np.inf),
            (np.full_like(x, 1e308), y),
        ):
            with self.subTest(shape=np.shape(bad_x)), self.assertRaises(ValueError):
                fit_ridge(bad_x, bad_y, 0.1)
        fit = fit_ridge(x, y, 0.1)
        for key, value in (
            ("mean", [0]),
            ("std", [0, 1, 1]),
            ("coefficient", [np.nan] * 3),
            ("intercept", [0]),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                predict_ridge({**fit, key: value}, x)
        for bad in ({}, None):
            with self.assertRaises(ValueError):
                predict_ridge(bad, x)


class CalibrationTests(unittest.TestCase):
    def test_exact_global_calibration_and_clipping(self):
        pred = np.arange(30.0).reshape(2, 15) / 30
        truth = 0.2 + 1.5 * pred
        fitted = fit_affine(pred, truth)
        self.assertAlmostEqual(fitted["slope"], 1.5)
        self.assertAlmostEqual(fitted["intercept"], 0.2)
        np.testing.assert_allclose(predict_affine(fitted, pred), truth)
        np.testing.assert_allclose(
            predict_affine(fitted, np.array([-100, 100])), [0, 2]
        )
        json.dumps(fitted, allow_nan=False)

    def test_constant_predictor_uses_train_target_mean(self):
        fitted = fit_affine(np.ones((3, 15)), np.tile([0, 1, 2], (3, 5)))
        self.assertEqual(fitted["slope"], 0)
        self.assertEqual(fitted["intercept"], 1)
        self.assertTrue(fitted["constant_prediction"])
        np.testing.assert_allclose(predict_affine(fitted, np.arange(20.0)), 1)

    def test_tiny_nonconstant_predictions_not_misclassified_as_constant(self):
        pred = np.array([0.0, 1e-200, 2e-200])
        fitted = fit_affine(pred, np.array([0.0, 1.0, 2.0]))
        self.assertFalse(fitted["constant_prediction"])
        np.testing.assert_allclose(predict_affine(fitted, pred), [0, 1, 2], atol=1e-14)

    def test_calibration_no_test_refit_and_invalid_input_checks(self):
        fitted = fit_affine(np.arange(4.0), 0.5 * np.arange(4.0))
        before = copy.deepcopy(fitted)
        np.testing.assert_allclose(
            predict_affine(fitted, np.array([1.0, 3.0, 100.0])), [0.5, 1.5, 2]
        )
        self.assertEqual(fitted, before)
        for left, right in (
            (1, 1),
            ([], []),
            ([0, 1], [0]),
            ([np.nan], [1]),
            ([1j], [1]),
        ):
            with self.assertRaises(ValueError):
                fit_affine(left, right)
        for bad in (
            {},
            {"slope": np.inf, "intercept": 1},
            {"slope": 1, "intercept": [0]},
        ):
            with self.assertRaises(ValueError):
                predict_affine(bad, [1])


class SummaryTests(unittest.TestCase):
    def test_exact_horizons_bias_and_squared_shared_normalization(self):
        data = fixture()
        data["pred"] += 0.5
        result = summarize(**data)
        self.assertEqual(tuple(result["horizons"]), tuple(map(str, HORIZONS)))
        for h in HORIZONS:
            item = result["horizons"][str(h)]
            self.assertAlmostEqual(item["reward_mse"]["mean"], 0.01)
            self.assertAlmostEqual(item["reward_bias"]["mean"], 0.1)
            self.assertAlmostEqual(item["cumulative_reward_mse"]["mean"], (h / 10) ** 2)
            self.assertAlmostEqual(item["cumulative_reward_bias"]["mean"], h / 10)
        self.assertEqual(set(result["by_mode"]), {"policy", "random"})
        self.assertEqual(set(result["by_plan"]), {"0", "1", "2", "3"})
        json.dumps(result, allow_nan=False)

    def test_endpoint_not_prefix_average_and_class_metrics_use_prefix(self):
        data = fixture()
        data["pred"][:, 0] += 1
        result = summarize(**data)
        self.assertAlmostEqual(result["horizons"]["3"]["reward_mse"]["mean"], 0)
        self.assertAlmostEqual(
            result["horizons"]["3"]["cumulative_reward_mse"]["mean"], 0.04
        )
        self.assertAlmostEqual(
            result["horizons"]["3"]["reward_classes"]["1"]["reward_mse"]["mean"],
            0.04 / 3,
        )

    def test_bootstrap_uses_episodes_not_rows_and_exact_resampling_reference(self):
        data = fixture()
        index = np.r_[np.zeros(20, dtype=int), 4, 8]
        for key in ("pred", "truth", "episode", "mode", "plan"):
            data[key] = data[key][index]
        data["pred"] += data["episode"][:, None]
        data["return_scale"] = 1
        result = summarize(**data)["horizons"]["1"]["reward_mse"]
        self.assertAlmostEqual(result["mean"], (0 + 1 + 4) / 3)
        self.assertNotAlmostEqual(result["mean"], 5 / 22)
        draws = np.random.default_rng(0).integers(0, 3, size=(200, 3))
        expected = np.quantile(
            np.array([0.0, 1.0, 4.0])[draws].mean(axis=1), (0.025, 0.975)
        )
        np.testing.assert_allclose(result["ci95"], expected)
        self.assertEqual(result["n_episodes"], 3)
        self.assertEqual(result["n_units"], 22)

    def test_duplicate_branches_do_not_tighten_ci(self):
        data = fixture()
        data["pred"] += 0.1 * data["episode"][:, None]
        first = summarize(**data)
        for key in ("pred", "truth", "episode", "mode", "plan"):
            data[key] = np.repeat(data[key], 5, axis=0)
        second = summarize(**data)
        for h in map(str, HORIZONS):
            for metric in ("reward_mse", "cumulative_reward_mse"):
                self.assertAlmostEqual(
                    first["horizons"][h][metric]["mean"],
                    second["horizons"][h][metric]["mean"],
                )
                np.testing.assert_allclose(
                    first["horizons"][h][metric]["ci95"],
                    second["horizons"][h][metric]["ci95"],
                    atol=1e-15,
                )

    def test_paired_delta_is_candidate_minus_baseline_and_not_unpaired_ci(self):
        data = fixture()
        data["return_scale"] = 1
        offset = data["episode"][:, None] + 1
        data["baseline"] = data["truth"] + offset
        data["pred"] = data["truth"] + np.sqrt(offset**2 + 0.5)
        report = summarize(**data)
        delta = report["horizons"]["1"]["delta_vs_baseline"]["reward_mse"]
        self.assertAlmostEqual(delta["mean"], 0.5)
        np.testing.assert_allclose(delta["ci95"], [0.5, 0.5], atol=1e-14)
        # Each separate error interval is broad; paired differences are constant.
        error = report["horizons"]["1"]["reward_mse"]
        self.assertGreater(error["ci95"][1] - error["ci95"][0], 10)
        data["pred"] = data["truth"].copy()
        improved = summarize(**data)["horizons"]["15"]["delta_vs_baseline"][
            "cumulative_reward_mse"
        ]
        self.assertLess(improved["ci95"][1], 0)
        data["pred"] = data["baseline"].copy()
        identical = summarize(**data)["horizons"]["15"]["delta_vs_baseline"][
            "cumulative_reward_mse"
        ]
        self.assertEqual(identical["mean"], 0)
        self.assertEqual(identical["ci95"], [0, 0])

    def test_zero_rewards_missing_classes_and_missing_plans_are_explicit(self):
        data = fixture()
        data["truth"][:] = 0
        data["pred"][:] = 0
        data["plan"][:] = 0
        report = summarize(**data)
        self.assertEqual(report["coverage"]["status"], "inconclusive")
        self.assertEqual(report["coverage"]["episodes_with_positive_reward"], 0)
        for cls in ("1", "2"):
            item = report["horizons"]["15"]["reward_classes"][cls]
            self.assertEqual(item["n_steps"], 0)
            self.assertIsNone(item["reward_mse"]["mean"])
            self.assertEqual(item["reward_mse"]["ci95"], [None, None])
            self.assertEqual(item["reward_mse"]["status"], "unavailable")
        for plan in ("1", "2", "3"):
            self.assertEqual(report["by_plan"][plan]["n_rows"], 0)
            self.assertIsNone(
                report["by_plan"][plan]["horizons"]["1"]["reward_mse"]["mean"]
            )
        json.dumps(report, allow_nan=False)

    def test_class_conditioning_counts_and_episode_weighting(self):
        data = fixture()
        data["truth"][:] = 0
        data["truth"][0, :1] = 1
        data["truth"][4, :15] = 1
        data["pred"] = data["truth"].copy()
        data["pred"][0, :1] += 1
        data["return_scale"] = 1
        item = summarize(**data)["horizons"]["15"]["reward_classes"]["1"]
        self.assertEqual(item["n_steps"], 16)
        self.assertEqual(item["n_rows"], 2)
        self.assertEqual(item["n_episodes"], 2)
        self.assertEqual(item["reward_mse"]["mean"], 0.5)
        self.assertEqual(item["prediction_mean_raw"]["mean"], 1.5)

    def test_single_episode_has_point_estimate_but_no_spurious_interval(self):
        data = fixture()
        data["episode"][:] = 0
        stat = summarize(**data)["horizons"]["1"]["reward_mse"]
        self.assertEqual(stat["mean"], 0)
        self.assertEqual(stat["ci95"], [None, None])
        self.assertEqual(stat["status"], "inconclusive")

    def test_mode_and_plan_strata_expose_bad_subgroup(self):
        data = fixture()
        data["pred"][data["plan"] == 3] += 1
        report = summarize(**data)
        self.assertEqual(
            report["by_plan"]["0"]["horizons"]["15"]["cumulative_reward_mse"]["mean"], 0
        )
        self.assertAlmostEqual(
            report["by_plan"]["3"]["horizons"]["15"]["cumulative_reward_mse"]["mean"], 9
        )
        self.assertGreater(
            report["by_mode"]["random"]["horizons"]["15"]["cumulative_reward_mse"][
                "mean"
            ],
            0,
        )

    def test_nonfinite_and_shape_errors_fail_without_silent_filtering(self):
        data = fixture()
        cases = {
            "pred": [
                data["pred"][:, :10],
                data["pred"][:, None],
                data["pred"] + 1j,
                np.full((24, 15), np.inf),
                np.full((24, 15), 1e308),
            ],
            "truth": [data["truth"] + 0.1, data["truth"][:-1]],
            "episode": [data["episode"].astype(float), np.full(24, -1)],
            "mode": [np.zeros(23), [None] * 24],
            "plan": [np.full(24, 4)],
            "return_scale": [0, -1, np.nan, [1], True],
            "bootstrap_reps": [0, 1, 1.5, True],
            "baseline": [np.ones((1, 15)), np.full((24, 15), np.nan)],
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    summarize(**{**data, key: value})

    def test_input_immutable_and_large_episode_ids_not_collapsed(self):
        data = fixture()
        data["episode"] += 2**54
        snapshot = copy.deepcopy(data)
        first, second = summarize(**data), summarize(**data)
        self.assertEqual(first, second)
        self.assertEqual(first["n_episodes"], 6)
        for key in ("pred", "truth", "episode", "mode", "plan"):
            np.testing.assert_array_equal(data[key], snapshot[key])


class DecompositionTests(unittest.TestCase):
    def test_random_identity_matches_independent_direct_sums(self):
        rng = np.random.default_rng(213)
        truth = rng.integers(0, 3, size=(20, 15)).astype(float)
        pred, post = rng.normal(size=truth.shape), rng.normal(size=truth.shape)
        result = decomposition(pred, post, truth, 3)
        for h in HORIZONS:
            base = (post[:, :h].sum(axis=1) - truth[:, :h].sum(axis=1)) / 3
            extra = (pred[:, :h].sum(axis=1) - post[:, :h].sum(axis=1)) / 3
            total = (pred[:, :h].sum(axis=1) - truth[:, :h].sum(axis=1)) / 3
            terms = result["horizons"][str(h)]
            for name, expected in (
                ("base", np.mean(base**2)),
                ("extra", np.mean(extra**2)),
                ("cross", 2 * np.mean(base * extra)),
                ("total", np.mean(total**2)),
            ):
                self.assertAlmostEqual(terms[name], expected, places=11)
            self.assertLess(abs(terms["identity_residual"]), 1e-11)
        json.dumps(result, allow_nan=False)

    def test_negative_cross_cancellation_refutes_naive_error_fractions(self):
        truth = np.zeros((2, 15))
        report = decomposition(truth, np.ones_like(truth), truth, 1)
        item = report["horizons"]["15"]
        self.assertEqual(
            item,
            {
                "base": 225,
                "extra": 225,
                "cross": -450,
                "total": 0,
                "identity_residual": 0,
            },
        )
        self.assertGreater(item["base"] + item["extra"], item["total"])
        self.assertNotIn("fraction", item)

    def test_positive_cross_and_zero_extra_controls(self):
        truth = np.zeros((3, 15))
        post = np.ones_like(truth)
        self.assertEqual(
            decomposition(2 * post, post, truth, 1)["horizons"]["1"]["cross"], 2
        )
        item = decomposition(post, post, truth, 1)["horizons"]["15"]
        self.assertEqual(item["extra"], 0)
        self.assertEqual(item["cross"], 0)
        self.assertEqual(item["base"], item["total"])

    def test_invalid_shapes_nonfinite_and_scales_rejected(self):
        a = np.zeros((2, 15))
        for pred, post, truth, scale in (
            (a, a[:1], a, 1),
            (a, a + 1j, a, 1),
            (a, a * np.nan, a, 1),
            (a, a, a, 0),
            (a[:, :10], a[:, :10], a[:, :10], 1),
            (np.full_like(a, 1e308), a, a, 1),
        ):
            with self.assertRaises(ValueError):
                decomposition(pred, post, truth, scale)


if __name__ == "__main__":
    program = unittest.main(exit=False)
    if program.result.wasSuccessful():
        print("REWARD_PROBE_METRICS_VERIFIED")
    sys.exit(0 if program.result.wasSuccessful() else 1)
