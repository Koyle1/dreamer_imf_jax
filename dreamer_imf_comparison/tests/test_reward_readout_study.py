"""Offline study orchestration and selection contracts, no simulator."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import jax
import numpy as np

from dreamer_imf_compare import reward_readout_study as study


def fixture():
    episode = np.repeat(np.arange(64), 16)
    return dict(
        episode=episode,
        split=np.repeat(np.repeat(np.arange(3), [40, 12, 12]), 16),
        mode=episode % 2,
        anchor=np.tile(np.repeat([100, 200, 300, 400], 4), 64),
        plan=np.tile(np.arange(4), 256),
        rewards=np.ones((1024, 15)),
        observations=np.zeros((1024, 15, 6)),
        actions=np.zeros((1024, 15, 2)),
        start=np.broadcast_to(np.zeros(1), (1024, 2560)),
        targets=np.broadcast_to(np.zeros(1), (1024, 15, 2560)),
    )


class Contracts(unittest.TestCase):
    def test_exact_episode_grid_and_balanced_split(self):
        study.validate_data(fixture())

    def test_split_leakage_fails(self):
        data = fixture()
        data["split"][0] = 1
        with self.assertRaisesRegex(ValueError, "split leakage"):
            study.validate_data(data)

    def test_wrong_reward_semantics_and_nonfinite_fail(self):
        for value in (0.5, -1, 3, np.nan):
            data = fixture()
            data["rewards"][0, 0] = value
            with self.assertRaises(ValueError):
                study.validate_data(data)

    def test_duplicate_branch_fails(self):
        data = fixture()
        data["plan"][0] = 1
        with self.assertRaisesRegex(ValueError, "branch grid"):
            study.validate_data(data)

    def test_validation_score_is_sum_reward_not_mean_step_error(self):
        pred = np.zeros((2, 15))
        truth = np.ones((2, 15))
        self.assertEqual(study.validation_score(pred, truth, 3), 25)
        pred[:, 0] = 15
        self.assertEqual(study.validation_score(pred, truth, 3), 0)

    def test_selection_ties_choose_earlier_and_reject_nan(self):
        self.assertEqual(
            study.first_minimum([dict(score=2), dict(score=1), dict(score=1)]), 1
        )
        for records in ([], [dict(score=np.nan)], [dict(score=np.inf)]):
            with self.assertRaises(ValueError):
                study.first_minimum(records)

    def test_validation_shape_and_finiteness(self):
        for pred, scale in (
            (np.zeros((2, 3)), 1),
            (np.full((2, 15), np.nan), 1),
            (np.zeros((2, 15)), 0),
        ):
            with self.assertRaises(ValueError):
                study.validation_score(pred, np.zeros((2, 15)), scale)

    def test_group_order_matches_frozen_feature_contract(self):
        features = np.arange(2560)[None, None]
        data = dict(targets=features, observations=np.zeros((1, 1, 6)))
        np.testing.assert_array_equal(
            study.group_features(data, "h"), features[..., :2048]
        )
        np.testing.assert_array_equal(
            study.group_features(data, "z"), features[..., 2048:]
        )
        np.testing.assert_array_equal(study.group_features(data, "hz"), features)
        with self.assertRaises(ValueError):
            study.group_features(data, "other")

    def test_exact_replay_rejects_small_difference(self):
        study.equal(np.ones(3), np.ones(3), "same")
        with self.assertRaisesRegex(ValueError, "replay differs"):
            study.equal(np.ones(3), np.ones(3) + 1e-12, "perturbed")

    def test_protocol_is_bounded_and_offline(self):
        p = study.read(study.PROTOCOL)
        self.assertEqual(p["additional_simulator_steps"], 0)
        self.assertEqual(p["mlp_seeds"], [0, 1, 2])
        self.assertEqual(p["mlp_updates"], 5000)
        self.assertEqual(p["ridge_alpha"], [0.001, 0.1, 10, 1000])
        self.assertEqual(p["posterior_draws"], 8)
        source = Path(study.__file__).read_text()
        self.assertNotIn("ReacherAdapter", source)
        self.assertNotIn("dm_control", source)
        self.assertNotIn(".policy(", source)


class ReadoutTraining(unittest.TestCase):
    def test_probability_mean_not_mode_and_finite_gradients(self):
        params = study.init_mlp(jax.random.PRNGKey(4), 6)
        params = jax.tree.map(lambda x: x * 0, params)
        pred = study.mlp_prediction(params, np.zeros((2, 15, 6), np.float32))
        np.testing.assert_allclose(pred, 1.0)
        gradient = jax.grad(lambda p: study.mlp_prediction(p, np.ones((3, 6))).sum())(
            params
        )
        self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(gradient)))

    def test_train_only_normalization_validation_only_selection_and_learning(self):
        rng = np.random.default_rng(421)
        x = rng.normal(size=(32, 15, 6)).astype(np.float32)
        y = (x[..., 0] > 0).astype(np.int32) * 2
        vx = rng.normal(size=(8, 15, 6)).astype(np.float32)
        vy = (vx[..., 0] > 0).astype(np.int32) * 2
        # An unused extreme test tensor must not affect train normalization.
        test = np.full((4, 15, 6), 1000, np.float32)
        original = x.copy()
        protocol = copy.deepcopy(study.read(study.PROTOCOL))
        protocol.update(mlp_updates=200, mlp_validation_period=50, mlp_batch=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mlp_0"
            record = study.train_mlp(path, x, y, vx, vy, 15, 0, protocol)
            norm = study.load_npz(path / "normalization.npz")
            np.testing.assert_allclose(
                norm["mean"], x.mean((0, 1), dtype=np.float64), atol=1e-8
            )
            self.assertTrue(np.all(norm["mean"] < test.mean()))
            records = json.loads((path / "selection.json").read_text())
            self.assertEqual(len(records["validations"]), 4)
            chosen = records["validations"][study.first_minimum(records["validations"])]
            self.assertEqual(record["checkpoint"], chosen["checkpoint"])
            reader = study.Readout(Path(tmp), record)
            pred = reader(vx)
            self.assertLess(np.mean((pred - vy) ** 2), 0.3)
            self.assertTrue(np.isfinite(reader(test)).all())
            for values in records["training"]:
                self.assertTrue(np.isfinite(values["gradient_norm"]))
        np.testing.assert_array_equal(x, original)


if __name__ == "__main__":
    unittest.main()
