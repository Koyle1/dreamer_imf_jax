import unittest
import numpy as np
from dreamer_imf_compare.reward_diagnosis_metrics import (
    boundary_bins,
    nll,
    calibration,
    summary,
    action_metrics,
)
from dreamer_imf_compare.reward_diagnosis_data import (
    check_branch,
    validate_endpoint,
    group_features,
)


class DiagnosisTests(unittest.TestCase):
    def test_boundary_inclusivity_and_ar2_ambiguity(self):
        np.testing.assert_array_equal(
            boundary_bins(np.array([0.79, 0.8, 1.0, 1.01, 1.2, 1.21]), 1),
            [0, 1, 1, 2, 2, 3],
        )
        validate_endpoint(np.array([0.5, 2.0]), np.array([1, 1]), 1.0)
        for distance, label in ((0.5, 0), (2.0, 2)):
            with self.assertRaises(ValueError):
                validate_endpoint(np.array([distance]), np.array([label]), 1.0)
        with self.assertRaises(ValueError):
            boundary_bins(np.array([-1.0]), 1.0)

    def test_categorical_scores_and_calibration(self):
        y = np.tile(np.arange(3), (6, 5))
        p = np.eye(3)[y]
        np.testing.assert_array_equal(nll(p, y), 0)
        r = summary(
            y, y, np.arange(6), probabilities=p, boundary=np.zeros_like(y), baseline=y
        )
        self.assertEqual(r["overall"]["mse"]["mean"], 0)
        self.assertEqual(r["overall"]["brier"]["mean"], 0)
        self.assertEqual(r["false_positive_rate"]["mean"], 0)
        self.assertEqual(r["endpoint_distance_strata"]["1"]["count"], 0)
        self.assertIsNone(r["endpoint_distance_strata"]["1"]["mse"]["mean"])
        self.assertEqual(calibration(p, y)["positive"][9]["frequency"], 1)
        self.assertFalse(r["material_exploratory_improvement"])
        uniform = np.ones_like(p) / 3
        np.testing.assert_allclose(nll(uniform, y), np.log(3))
        bad = p.copy()
        bad[0, 0] = [0.2, 0.2, 0.2]
        with self.assertRaises(ValueError):
            nll(bad, y)

    def test_episode_clustering_not_particle_or_row_replication(self):
        y = np.zeros((2, 15))
        pred = np.stack([np.ones(15), np.ones(15) * 2])
        a = summary(pred, y, np.array([0, 1]))
        b = summary(
            np.repeat(pred, [10, 1], axis=0),
            np.zeros((11, 15)),
            np.array([0] * 10 + [1]),
        )
        self.assertEqual(a["overall"]["mse"]["mean"], 2.5)
        self.assertEqual(a["overall"]["mse"]["mean"], b["overall"]["mse"]["mean"])
        self.assertEqual(a["overall"]["mse"]["ci95"], b["overall"]["mse"]["ci95"])
        self.assertEqual(a["reward_conclusions"], "inconclusive")

    def test_paired_improvement_flag_and_negative_control(self):
        y = np.ones((6, 15))
        baseline = np.zeros_like(y)
        good = summary(y, y, np.arange(6), baseline=baseline)
        bad = summary(np.ones_like(y) * 3, y, np.arange(6), baseline=baseline)
        self.assertTrue(good["material_exploratory_improvement"])
        self.assertFalse(bad["material_exploratory_improvement"])

    def test_ranking_and_effects(self):
        plans = np.tile(np.arange(4), 6)
        ep = np.repeat(np.arange(6), 4)
        truth = np.repeat((plans % 3)[:, None], 15, axis=1)
        r = action_metrics(truth, truth, ep, np.ones(24, int) * 100, plans)
        self.assertEqual(r["ranking_accuracy"]["mean"], 1)
        self.assertTrue(
            all(v["mse"]["mean"] == 0 for v in r["paired_effects"].values())
        )
        with self.assertRaises(ValueError):
            action_metrics(
                truth[:-1], truth[:-1], ep[:-1], np.ones(23, int) * 100, plans[:-1]
            )

    def test_branch_alignment_and_reset_negative_controls(self):
        raw = [
            dict(
                position=np.array([i, 0]),
                to_target=np.array([0.0, 1.0]),
                velocity=np.zeros(2),
                reward=np.asarray(i % 3),
                is_first=False,
                is_last=False,
            )
            for i in range(16)
        ]
        obs = np.array(
            [np.r_[o["position"], o["to_target"], o["velocity"]] for o in raw],
            np.float32,
        )
        rewards = np.array([o["reward"] for o in raw[1:]])
        features = np.ones((16, 4), np.float32)
        branch = dict(actions=np.ones((15, 2)), rewards=rewards, features=features)
        data = dict(
            observations=obs[None, 1:],
            actions=branch["actions"][None],
            rewards=rewards[None],
            start=features[None, 0],
            targets=features[None, 1:],
        )
        previous, distance = check_branch(data, 0, branch, raw)
        np.testing.assert_array_equal(previous, obs[:-1])
        np.testing.assert_array_equal(distance, 1)
        raw[1]["is_first"] = True
        with self.assertRaises(ValueError):
            check_branch(data, 0, branch, raw)
        raw[1]["is_first"] = False
        data["rewards"] = np.roll(rewards, 1)[None]
        with self.assertRaises(ValueError):
            check_branch(data, 0, branch, raw)

    def test_input_groups_do_not_include_rewards(self):
        f = np.arange(8).reshape(2, 4)
        o = np.ones((2, 6))
        a = np.zeros((2, 2))
        np.testing.assert_array_equal(group_features("h", f, deter=2), f[:, :2])
        np.testing.assert_array_equal(group_features("z", f, deter=2), f[:, 2:])
        x = group_features("history_action", f, o, o * 2, a, deter=2)
        self.assertEqual(x.shape, (2, 14))
        np.testing.assert_array_equal(x[:, :6], o)


if __name__ == "__main__":
    unittest.main()
