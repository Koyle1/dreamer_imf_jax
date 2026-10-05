"""Independent analytic/loop-reference controls; NumPy only, no frozen model."""

import copy
import json
import sys
import unittest

import numpy as np

from dreamer_imf_compare.parallel_metrics import (
    HORIZONS,
    compare_distributions,
    fit_scales,
    promotion,
    split_counts,
    summarize,
)


def fixture(episodes=6, particles=8):
    ep = np.repeat(np.arange(episodes), 4)
    plan = np.tile(np.arange(4), episodes)
    rewards = np.repeat((plan + 1)[:, None], 15, axis=1).astype(float)
    obs = np.repeat(np.stack([plan, plan + 2], -1)[:, None], 15, axis=1).astype(float)
    return {
        "pred_obs": np.repeat(obs[:, None], particles, axis=1),
        "pred_rewards": np.repeat(rewards[:, None], particles, axis=1),
        "truth_obs": obs,
        "truth_rewards": rewards,
        "episode": ep,
        "anchor": np.zeros(len(ep), dtype=int),
        "plan": plan,
        "obs_scale": np.ones(2),
        "return_scale": 1.0,
        "bootstrap_reps": 200,
    }


def summary(**changes):
    data = fixture()
    data.update(changes)
    return summarize(**data)


def comparison(data=None, *, prefix=None, obs_b=None, rewards_b=None, **options):
    data = fixture() if data is None else data
    return compare_distributions(
        data["pred_obs"],
        data["pred_rewards"],
        data["pred_obs"] if obs_b is None else obs_b,
        data["pred_rewards"] if rewards_b is None else rewards_b,
        data["episode"],
        data["obs_scale"],
        data["return_scale"],
        prefix_horizon=prefix,
        bootstrap_reps=200,
        **options,
    )


class ScaleTests(unittest.TestCase):
    def test_exact_train_only_population_std_and_full_return_rms(self):
        obs = np.array([[[0, 3], [2, 3]], [[4, 3], [6, 3]]], float)
        rewards = np.array([[1, 2], [0, 4]], float)
        scale, ret = fit_scales(obs, rewards, split=np.zeros(2, int))
        np.testing.assert_allclose(scale, [np.sqrt(5), 0.01])
        self.assertAlmostEqual(ret, np.sqrt((9 + 16) / 2))
        # Held-out extremes do not contribute after explicit train slicing.
        all_obs = np.concatenate((obs, obs[:1] + 1000))
        all_rewards = np.concatenate((rewards, rewards[:1] + 1000))
        labels = np.array([0, 0, 2])
        sliced = fit_scales(all_obs[labels == 0], all_rewards[labels == 0])
        np.testing.assert_array_equal(sliced[0], scale)
        self.assertEqual(sliced[1], ret)
        with self.assertRaises(ValueError):
            fit_scales(all_obs, all_rewards, split=labels)

    def test_scale_floor_and_bad_inputs(self):
        scale, ret = fit_scales(np.zeros((2, 3, 1)), np.zeros((2, 3)))
        np.testing.assert_array_equal(scale, [0.01])
        self.assertEqual(ret, 0.01)
        for obs, rewards in (
            (np.empty((0, 3, 1)), np.empty((0, 3))),
            (np.ones((2, 3, 1)), np.ones((2, 4))),
            (np.full((2, 3, 1), np.nan), np.zeros((2, 3))),
            (np.zeros((2, 3, 1)), np.full((2, 3), np.inf)),
            (np.zeros((2, 3, 1)), np.full((2, 3), 1e308)),
        ):
            with self.subTest(shape=obs.shape), self.assertRaises(ValueError):
                fit_scales(obs, rewards)

    def test_exact_split_counts_and_zero_positive_breakdown(self):
        ep = np.repeat(np.arange(64), 16)
        split = np.repeat(
            np.r_[np.zeros(40), np.ones(12), np.full(12, 2)].astype(int), 16
        )
        rewards = np.zeros((1024, 15))
        rewards[split == 2, 0] = 1
        result = split_counts(ep, split, rewards)
        self.assertEqual(
            [result[key]["rows"] for key in ("train", "validation", "test")],
            [640, 192, 192],
        )
        self.assertEqual(
            [result[key]["episodes"] for key in ("train", "validation", "test")],
            [40, 12, 12],
        )
        self.assertEqual(result["train"]["zero_return_rows"], 640)
        self.assertEqual(result["test"]["positive_return_episodes"], 12)
        split[0] = 1
        with self.assertRaisesRegex(ValueError, "crosses"):
            split_counts(ep, split, rewards)

    def test_absent_split_and_negative_reward_counts(self):
        result = split_counts(
            np.array([0, 1]), np.array([2, 2]), np.array([[1, -1], [-1, -1]])
        )
        self.assertEqual(result["train"]["rows"], 0)
        self.assertEqual(result["test"]["zero_return_rows"], 1)
        self.assertEqual(result["test"]["negative_return_rows"], 1)
        self.assertEqual(result["test"]["episodes_with_positive_reward"], 1)
        for episode, split in (([], []), ([0.0], [0]), ([0], [3]), ([0], [0, 1])):
            with self.assertRaises(ValueError):
                split_counts(episode, split)


class SummaryTests(unittest.TestCase):
    def test_perfect_prediction_all_required_horizons(self):
        result = summary()
        self.assertEqual(list(result["horizons"]), [str(h) for h in HORIZONS])
        self.assertTrue(result["coverage"]["adequate"])
        self.assertFalse(result["failures"]["material_action_failure"])
        for metrics in result["horizons"].values():
            for name in (
                "observation_mse",
                "reward_mse",
                "cumulative_reward_mse",
                "observation_crps",
                "reward_crps",
                "cumulative_reward_crps",
            ):
                self.assertEqual(metrics[name]["mean"], 0.0)
            self.assertEqual(metrics["observation_interval_coverage"]["mean"], 1.0)
            self.assertEqual(
                metrics["actions"]["ranking_pairwise_accuracy"]["mean"], 1.0
            )
            self.assertEqual(metrics["actions"]["complete_anchors"], 6)
        json.dumps(result, allow_nan=False)

    def test_normalized_endpoint_and_cumulative_errors_exact(self):
        data = fixture()
        data["pred_obs"] += 2
        data["pred_rewards"] -= 1
        data["obs_scale"] *= 2
        data["return_scale"] = 2
        result = summarize(**data)
        for h in HORIZONS:
            metrics = result["horizons"][str(h)]
            self.assertEqual(metrics["observation_mse"]["mean"], 1)
            self.assertEqual(metrics["reward_mse"]["mean"], 0.25)
            self.assertEqual(metrics["cumulative_reward_mse"]["mean"], h * h / 4)
            self.assertEqual(metrics["cumulative_reward_crps"]["mean"], h / 2)

    def test_empirical_crps_independent_double_loop_not_particle_mse(self):
        data = fixture(particles=3)
        offsets = np.array([-2, 0, 5], float)
        data["pred_obs"] += offsets[None, :, None, None]
        data["pred_rewards"] += offsets[None, :, None]
        result = summarize(**data)["horizons"]["1"]
        empirical = (
            sum(abs(x) for x in offsets) / 3
            - sum(abs(x - y) for x in offsets for y in offsets) / 18
        )
        self.assertAlmostEqual(result["observation_crps"]["mean"], empirical)
        self.assertAlmostEqual(result["reward_crps"]["mean"], empirical)
        self.assertAlmostEqual(result["observation_mse"]["mean"], 1.0)
        self.assertNotEqual(result["observation_mse"]["mean"], np.mean(offsets**2))

    def test_empirical_interval_coverage_is_not_assumed_nominal(self):
        data = fixture(particles=2)
        data["pred_obs"][:, 0] -= 1
        data["pred_obs"][:, 1] += 1
        data["pred_rewards"] += 2
        h1 = summarize(**data)["horizons"]["1"]
        self.assertEqual(h1["observation_interval_coverage"]["mean"], 1)
        self.assertEqual(h1["reward_interval_coverage"]["mean"], 0)

    def test_crps_translation_stability(self):
        data = fixture(particles=3)
        data["pred_obs"] += np.array([-2, 0, 5])[None, :, None, None]
        first = summarize(**data)["horizons"]["1"]["observation_crps"]["mean"]
        data["pred_obs"] += 1e12
        data["truth_obs"] += 1e12
        second = summarize(**data)["horizons"]["1"]["observation_crps"]["mean"]
        self.assertEqual(first, second)

    def test_endpoint_not_trajectory_average(self):
        data = fixture()
        data["pred_obs"][:, :, 0] += 10
        result = summarize(**data)
        self.assertEqual(result["horizons"]["1"]["observation_mse"]["mean"], 100)
        self.assertEqual(result["horizons"]["3"]["observation_mse"]["mean"], 0)

    def test_episode_estimand_and_bootstrap_not_rows_or_particles(self):
        data = fixture(episodes=2)
        # Unequal row counts: duplicate the zero-error first episode only.
        data["pred_obs"][data["episode"] == 1] += 2
        index = np.r_[np.arange(4), np.arange(8)]
        for key in (
            "pred_obs",
            "pred_rewards",
            "truth_obs",
            "truth_rewards",
            "episode",
            "anchor",
            "plan",
        ):
            data[key] = data[key][index]
        score = summarize(**data)["horizons"]["1"]["observation_mse"]
        self.assertEqual(score["mean"], 2)
        self.assertEqual(score["ci95"], [0, 4])
        self.assertEqual(score["n_episodes"], 2)
        self.assertEqual(score["n_units"], 12)

    def test_duplicate_replays_do_not_shrink_episode_confidence_interval(self):
        data = fixture()
        data["pred_obs"] += data["episode"][:, None, None, None]
        first = summarize(**data)
        for key in (
            "pred_obs",
            "pred_rewards",
            "truth_obs",
            "truth_rewards",
            "episode",
            "anchor",
            "plan",
        ):
            data[key] = np.repeat(data[key], 3, axis=0)
        second = summarize(**data)
        a, b = first["horizons"]["15"], second["horizons"]["15"]
        self.assertEqual(a["observation_mse"]["ci95"], b["observation_mse"]["ci95"])
        self.assertEqual(a["observation_mse"]["mean"], b["observation_mse"]["mean"])
        self.assertEqual(a["actions"], b["actions"])

    def test_sparse_and_absent_positive_rewards_are_inconclusive(self):
        for positive_episodes in (0, 4, 5):
            data = fixture()
            mask = data["episode"] >= positive_episodes
            data["truth_rewards"][mask] = 0
            data["pred_rewards"][mask] = 0
            result = summarize(**data)
            self.assertEqual(
                result["coverage"]["episodes_with_positive_reward"], positive_episodes
            )
            self.assertEqual(result["coverage"]["adequate"], positive_episodes >= 5)
            positive = result["horizons"]["1"]["positive_return"]["reward_mse"]
            self.assertEqual(positive["n_episodes"], positive_episodes)
            if positive_episodes == 0:
                self.assertIsNone(positive["mean"])
                self.assertEqual(positive["status"], "unavailable")

    def test_missing_plan_is_inconclusive_even_with_positive_rewards(self):
        data = fixture()
        index = data["plan"] != 3
        for key in (
            "pred_obs",
            "pred_rewards",
            "truth_obs",
            "truth_rewards",
            "episode",
            "anchor",
            "plan",
        ):
            data[key] = data[key][index]
        result = summarize(**data)
        self.assertFalse(result["coverage"]["adequate"])
        self.assertFalse(result["coverage"]["action_coverage_adequate"])

    def test_single_episode_no_spurious_ci(self):
        result = summarize(**fixture(episodes=1))
        stat = result["horizons"]["1"]["observation_mse"]
        self.assertEqual(stat["status"], "inconclusive")
        self.assertEqual(stat["ci95"], [None, None])
        self.assertFalse(result["coverage"]["adequate"])

    def test_reversed_action_effects_are_material_failure(self):
        data = fixture()
        data["pred_rewards"] = 5 - data["pred_rewards"]
        result = summarize(**data)
        self.assertTrue(result["failures"]["material_action_failure"])
        actions = result["horizons"]["1"]["actions"]
        self.assertEqual(actions["ranking_pairwise_accuracy"]["mean"], 0)
        effect = actions["paired_effects"]["0_to_1"]
        self.assertEqual(effect["predicted_return_effect"]["mean"], -1)
        self.assertEqual(effect["true_return_effect"]["mean"], 1)
        self.assertEqual(effect["return_effect_mse"]["mean"], 4)
        self.assertTrue(effect["material_failure"])

    def test_observation_action_failure_and_small_tolerated_effect(self):
        for size, material in ((1.0, True), (0.01, False)):
            data = fixture()
            data["pred_obs"] += size * data["plan"][:, None, None, None]
            result = summarize(**data)
            self.assertEqual(result["failures"]["material_action_failure"], material)

    def test_action_predicted_ties_half_credit_true_ties_excluded(self):
        data = fixture()
        data["pred_rewards"][:] = 0
        ranking = summarize(**data)["horizons"]["1"]["actions"][
            "ranking_pairwise_accuracy"
        ]
        self.assertEqual(ranking["mean"], 0.5)
        data["truth_rewards"][:] = 1
        ranking = summarize(**data)["horizons"]["1"]["actions"][
            "ranking_pairwise_accuracy"
        ]
        self.assertIsNone(ranking["mean"])

    def test_deterministic_single_particle_baseline_supported(self):
        result = summarize(**fixture(particles=1))
        self.assertEqual(result["horizons"]["1"]["reward_crps"]["mean"], 0)

    def test_validation_rejects_wrong_shapes_labels_scales_and_nonfinite(self):
        data = fixture()
        cases = {
            "pred_obs": [
                data["pred_obs"][:, :, :10],
                data["pred_obs"][:, 0],
                np.full_like(data["pred_obs"], np.nan),
                data["pred_obs"].astype(complex) + 1j,
            ],
            "pred_rewards": [
                data["pred_rewards"][:, :, :10],
                np.full_like(data["pred_rewards"], np.inf),
            ],
            "truth_obs": [data["truth_obs"][1:]],
            "truth_rewards": [data["truth_rewards"][:, 0]],
            "episode": [
                data["episode"].astype(float),
                np.full(len(data["episode"]), -1),
            ],
            "anchor": [data["anchor"][:-1]],
            "plan": [np.full(len(data["plan"]), 4)],
            "obs_scale": [[1], [1, 0], [np.nan, 1]],
            "return_scale": [0, -1, np.inf, [1], "invalid"],
            "bootstrap_reps": [0, 1, 1.2, True],
            "random_seed": [-1, 0.5, True],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    summarize(**{**data, name: value})

    def test_large_episode_labels_keep_fingerprint_identity(self):
        data = fixture()
        data["episode"] += 2**54
        first = summarize(**data)
        data["episode"] += 1
        second = summarize(**data)
        self.assertNotEqual(first["data_fingerprint"], second["data_fingerprint"])

    def test_action_effect_noise_correction_independent_reference(self):
        data = fixture(particles=2)
        data["pred_rewards"][:, 0] -= 1
        data["pred_rewards"][:, 1] += 1
        effect = summarize(**data)["horizons"]["1"]["actions"]["paired_effects"][
            "0_to_1"
        ]
        # Each particle mean has sample variance 2 / S2 = 1; two independent
        # means therefore contribute MC variance 2, even when realized means agree.
        self.assertEqual(effect["return_effect_mse"]["mean"], 0)
        self.assertEqual(effect["return_effect_debiased_mse"]["mean"], -2)
        self.assertFalse(effect["material_failure"])


class DistributionTests(unittest.TestCase):
    def test_identical_and_particle_permuted_distributions_no_failure(self):
        data = fixture()
        rng = np.random.default_rng(123)
        data["pred_obs"] += rng.normal(size=data["pred_obs"].shape)
        data["pred_rewards"] += rng.normal(size=data["pred_rewards"].shape)
        first = comparison(data)
        permuted = comparison(
            data,
            obs_b=data["pred_obs"][:, ::-1],
            rewards_b=data["pred_rewards"][:, ::-1],
        )
        self.assertFalse(first["material_failure"])
        self.assertFalse(permuted["material_failure"])
        for h in HORIZONS:
            for name in ("observation_energy", "cumulative_reward_energy"):
                self.assertAlmostEqual(
                    first["horizons"][str(h)][name]["mean"],
                    permuted["horizons"][str(h)][name]["mean"],
                )

    def test_exact_unbiased_energy_independent_loop_unequal_particle_counts(self):
        data = fixture(particles=3)
        left = np.array([-2, 0, 3.0])
        right = np.array([-1, 1, 2, 4.0])
        data["pred_obs"][:] = left[None, :, None, None]
        obs_b = np.broadcast_to(right[None, :, None, None], (24, 4, 15, 2)).copy()
        rewards_b = np.repeat(data["pred_rewards"][:, :1], 4, axis=1)
        result = comparison(data, obs_b=obs_b, rewards_b=rewards_b)
        cross = sum(abs(x - y) for x in left for y in right) / (len(left) * len(right))
        within_left = (
            sum(abs(left[i] - left[j]) for i in range(3) for j in range(3) if i != j)
            / 6
        )
        within_right = (
            sum(abs(right[i] - right[j]) for i in range(4) for j in range(4) if i != j)
            / 12
        )
        self.assertAlmostEqual(
            result["horizons"]["1"]["observation_energy"]["mean"],
            2 * cross - within_left - within_right,
        )
        # Reversing sets must not change the discrepancy, including unequal S.
        reverse = compare_distributions(
            obs_b,
            rewards_b,
            data["pred_obs"],
            data["pred_rewards"],
            data["episode"],
            data["obs_scale"],
            data["return_scale"],
            bootstrap_reps=200,
        )
        self.assertEqual(result["horizons"], reverse["horizons"])

    def test_deterministic_shift_material_and_zero_control_not(self):
        data = fixture()
        self.assertFalse(comparison(data)["material_failure"])
        result = comparison(data, obs_b=data["pred_obs"] + 1)
        self.assertTrue(result["material_failure"])
        self.assertEqual(result["horizons"]["1"]["observation_energy"]["mean"], 2)
        self.assertTrue(
            comparison(data, rewards_b=data["pred_rewards"] + 1)["material_failure"]
        )

    def test_variance_change_detected_even_when_means_match(self):
        data = fixture(particles=8)
        alternate = data["pred_obs"].copy()
        alternate[:, :4] -= 2
        alternate[:, 4:] += 2
        result = comparison(data, obs_b=alternate)
        self.assertTrue(result["material_failure"])

    def test_independent_mc_draws_do_not_require_pathwise_equality(self):
        data = fixture(episodes=12, particles=32)
        rng = np.random.default_rng(402)
        second_obs = data["pred_obs"] + rng.normal(0, 0.3, data["pred_obs"].shape)
        second_rewards = data["pred_rewards"] + rng.normal(
            0, 0.03, data["pred_rewards"].shape
        )
        data["pred_obs"] += rng.normal(0, 0.3, data["pred_obs"].shape)
        data["pred_rewards"] += rng.normal(0, 0.03, data["pred_rewards"].shape)
        self.assertGreater(np.mean((data["pred_obs"] - second_obs) ** 2), 0.1)
        result = comparison(data, obs_b=second_obs, rewards_b=second_rewards)
        self.assertFalse(result["material_failure"])

    def test_prefix_ignores_changed_suffix_but_checks_every_prefix_decision(self):
        data = fixture()
        alternate = data["pred_obs"].copy()
        alternate[:, :, 5:] += 20
        result = comparison(data, prefix=5, obs_b=alternate)
        self.assertEqual(result["comparison"], "prefix")
        self.assertEqual(list(result["horizons"]), ["1", "2", "3", "4", "5"])
        self.assertFalse(result["material_failure"])
        alternate[:, :, 1] += 1  # h2 is deliberately not a primary scoring horizon.
        self.assertTrue(comparison(data, prefix=5, obs_b=alternate)["material_failure"])

    def test_four_episodes_cannot_conclude_distribution_check_adequate(self):
        result = comparison(fixture(episodes=4))
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["material_failure"])

    def test_comparison_invalid_inputs_fail(self):
        data = fixture()
        for prefix in (0, 16, 0.5, True):
            with self.assertRaises(ValueError):
                comparison(data, prefix=prefix)
        for tolerance in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                comparison(data, material_tolerance=tolerance)
        for obs in (
            data["pred_obs"][:, :1],
            data["pred_obs"][:-1],
            np.full_like(data["pred_obs"], np.inf),
        ):
            with self.assertRaises(ValueError):
                comparison(data, obs_b=obs)


class PromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = fixture()
        data[
            "pred_obs"
        ] += 1  # Common offset preserves action effects, positive denominator.
        data["pred_rewards"] += 0.1
        cls.report = summarize(**data)
        cls.checks = [comparison(data), comparison(data, prefix=5)]

    def inputs(self):
        return {
            "seed_reports": {s: copy.deepcopy(self.report) for s in range(3)},
            "baseline": copy.deepcopy(self.report),
            "student_latencies": {s: {"batch1": 1.0, "batch64": 2.0} for s in range(3)},
            "baseline_latency": {"batch1": 2.0, "batch64": 4.0},
            "distribution_checks": {s: copy.deepcopy(self.checks) for s in range(3)},
        }

    def test_boundary_eligible_all_three_seeds(self):
        data = self.inputs()
        for report in data["seed_reports"].values():
            for metrics in report["horizons"].values():
                metrics["observation_mse"]["mean"] *= 1.1
                metrics["cumulative_reward_mse"]["mean"] *= 1.1
        result = promotion(**data)
        self.assertTrue(result["promote"], result["reasons"])
        self.assertEqual(result["seeds"]["2"]["speedups"], {"batch1": 2, "batch64": 2})
        json.dumps(result, allow_nan=False)

    def test_one_bad_seed_cannot_be_hidden_by_mean(self):
        data = self.inputs()
        data["seed_reports"][0]["horizons"]["15"]["observation_mse"]["mean"] = 0
        data["seed_reports"][1]["horizons"]["15"]["observation_mse"]["mean"] = 0
        data["seed_reports"][2]["horizons"]["15"]["observation_mse"]["mean"] *= 1.1001
        self.assertFalse(promotion(**data)["promote"])

    def test_each_primary_metric_horizon_and_speed_required(self):
        for h in HORIZONS:
            for metric in ("observation_mse", "cumulative_reward_mse"):
                data = self.inputs()
                data["seed_reports"][1]["horizons"][str(h)][metric]["mean"] *= 1.2
                self.assertFalse(promotion(**data)["promote"])
        for batch in ("batch1", "batch64"):
            data = self.inputs()
            data["student_latencies"][2][batch] *= 1.001
            self.assertFalse(promotion(**data)["promote"])

    def test_zero_baseline_no_epsilon_false_favorable(self):
        data = self.inputs()
        baseline = summary()
        data["baseline"] = baseline
        data["seed_reports"] = {s: copy.deepcopy(baseline) for s in range(3)}
        self.assertTrue(promotion(**data)["promote"])
        data["seed_reports"][1]["horizons"]["1"]["observation_mse"]["mean"] = 1e-30
        result = promotion(**data)
        self.assertFalse(result["promote"])
        self.assertIsNone(result["seeds"]["1"]["error_ratios"]["1"]["observation_mse"])

    def test_sparse_coverage_action_failure_and_fingerprint_blocks(self):
        for field, value in (
            ("coverage", {"adequate": False}),
            ("failures", {"material_action_failure": True}),
            ("data_fingerprint", "different"),
        ):
            data = self.inputs()
            data["seed_reports"][0][field] = value
            self.assertFalse(promotion(**data)["promote"])
        data = self.inputs()
        data["baseline"]["coverage"]["adequate"] = False
        self.assertFalse(promotion(**data)["promote"])

    def test_missing_or_failed_distribution_checks_blocks(self):
        for checks in (
            [],
            self.checks[:1],
            [self.checks[0], self.checks[0]],
            [self.checks[0], {**self.checks[1], "material_failure": True}],
            [self.checks[0], {**self.checks[1], "status": "inconclusive"}],
        ):
            data = self.inputs()
            data["distribution_checks"][2] = checks
            self.assertFalse(promotion(**data)["promote"])

    def test_nonfinite_negative_missing_errors_latencies_and_seeds_blocks(self):
        for value in (np.nan, np.inf, -1, None, True):
            data = self.inputs()
            data["seed_reports"][0]["horizons"]["1"]["observation_mse"]["mean"] = value
            result = promotion(**data)
            self.assertFalse(result["promote"])
            json.dumps(result, allow_nan=False)
        for value in (0, -1, np.nan, None, True):
            data = self.inputs()
            data["student_latencies"][1]["batch64"] = value
            self.assertFalse(promotion(**data)["promote"])
        for key in ("seed_reports", "student_latencies", "distribution_checks"):
            data = self.inputs()
            del data[key][2]
            self.assertFalse(promotion(**data)["promote"])
        data = self.inputs()
        data["seed_reports"]["0"] = data["seed_reports"][0]
        self.assertFalse(promotion(**data)["promote"])

    def test_missing_fields_fail_closed(self):
        data = self.inputs()
        data["seed_reports"][1] = {}
        self.assertFalse(promotion(**data)["promote"])
        data["baseline"] = {}
        self.assertFalse(promotion(**data)["promote"])

    def test_malformed_nested_reports_fail_closed_without_favorable_partial(self):
        for name in ("coverage", "failures", "horizons"):
            data = self.inputs()
            data["seed_reports"][1][name] = None
            self.assertFalse(promotion(**data)["promote"])
        data = self.inputs()
        data["baseline"] = None
        self.assertFalse(promotion(**data)["promote"])

    def test_stale_adequate_flags_do_not_hide_sparse_or_missing_evidence(self):
        data = self.inputs()
        data["seed_reports"][0]["coverage"]["episodes_with_positive_reward"] = 4
        result = promotion(**data)
        self.assertFalse(result["promote"])
        self.assertEqual(result["status"], "inconclusive")
        for field, value in (
            ("n_episodes", 4),
            ("horizons", {}),
            ("material_tolerance", 100),
            ("prefix_horizon", 0),
        ):
            data = self.inputs()
            data["distribution_checks"][0][1][field] = value
            self.assertFalse(promotion(**data)["promote"])
        data = self.inputs()
        data["seed_reports"][1]["horizons"]["3"]["actions"]["material_failure"] = True
        self.assertFalse(promotion(**data)["promote"])


if __name__ == "__main__":
    program = unittest.main(exit=False)
    if program.result.wasSuccessful():
        print("PARALLEL_METRICS_VERIFIED")
    sys.exit(0 if program.result.wasSuccessful() else 1)
