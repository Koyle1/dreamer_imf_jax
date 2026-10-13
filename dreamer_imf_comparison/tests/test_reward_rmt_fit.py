import copy
import json
from pathlib import Path
import tempfile
import unittest

import jax
import numpy as np

from dreamer_imf_compare import reward_capacity_fit as old
from dreamer_imf_compare import reward_rmt_fit as fit
from dreamer_imf_compare.parallel_runner import load_npz
from dreamer_imf_compare.prior_correction_study import load
from dreamer_imf_compare.reward_readout_study import read

PROTOCOL = dict(updates=2, learning_rate=3e-4, batch_size=8, minibatch_seed=6211)


def fixture():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=32)
    vx = rng.normal(size=(13, 4)).astype(np.float32) + 4
    vy = rng.integers(0, 3, size=13)
    return x, y, vx, vy


class RMTFitTests(unittest.TestCase):
    def test_early_candidate_inventory_and_input_validation(self):
        expected = [0] + list(range(10, 251, 10)) + list(range(500, 5001, 250))
        self.assertEqual(fit.candidate_steps(), expected)
        self.assertEqual(fit.candidate_steps(2), [0, 2])
        self.assertEqual(fit.candidate_steps(11), [0, 10, 11])
        self.assertEqual(fit.candidate_steps(0), [0])
        for bad in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                fit.candidate_steps(bad)
        x, y, vx, vy = fixture()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                fit.fit_cell(Path(td) / "bad", x, y + 10, vx, vy, 0, 0, PROTOCOL)
            self.assertFalse((Path(td) / "bad").exists())
            with self.assertRaises(ValueError):
                fit.fit_cell(Path(td) / "bad", x, y, vx[:, :-1], vy, 0, 0, PROTOCOL)
            with self.assertRaises(ValueError):
                fit.fit_cell(Path(td) / "bad", x, y, vx, vy, -1, 0, PROTOCOL)

    def test_exact_zero_regularization_matches_current_fitter(self):
        x, y, vx, vy = fixture()
        original = copy.deepcopy((x, y, vx, vy))
        with tempfile.TemporaryDirectory() as td:
            fresh, previous = Path(td) / "new", Path(td) / "old"
            result = fit.fit_cell(fresh, x, y, vx, vy, 0, 0, PROTOCOL)
            old.fit(previous, x, y, vx, vy, 128, 0, dict(PROTOCOL, validation_period=1))
            for a, b in zip(
                jax.tree.leaves(load(fresh / "checkpoint-00002.pkl")),
                jax.tree.leaves(load(previous / "checkpoint-00002.pkl")),
            ):
                np.testing.assert_array_equal(a, b)
            verified = fit.verify_cell(fresh, x, y, vx, vy, 0, 0, PROTOCOL)
            self.assertEqual(verified, result)
            self.assertEqual(result["candidates"][0]["update"], 0)
            for candidate in result["candidates"]:
                self.assertEqual(
                    fit.scores(
                        load_npz(fresh / f"candidate-{candidate['update']:05d}.npz")[
                            "validation"
                        ],
                        vy,
                    ),
                    candidate["validation"],
                )
            with self.assertRaises(FileExistsError):
                fit.fit_cell(fresh, x, y, vx, vy, 0, 0, PROTOCOL)
            for actual, expected in zip((x, y, vx, vy), original):
                np.testing.assert_array_equal(actual, expected)

    def test_weight_penalty_has_no_bias_gradient(self):
        params = fit.init(jax.random.PRNGKey(0), 4)
        alpha = 1e-4
        grad = jax.grad(lambda p: fit.weight_penalty(p, alpha))(params)
        expected = (
            alpha / 2 * sum(float(np.sum(np.asarray(p["w"]) ** 2)) for p in params)
        )
        self.assertAlmostEqual(
            float(fit.weight_penalty(params, alpha)), expected, places=7
        )
        for p, g in zip(params, grad):
            np.testing.assert_allclose(g["w"], alpha * p["w"], rtol=1e-7)
            np.testing.assert_array_equal(g["b"], np.zeros_like(g["b"]))

    def test_binary_endpoint_fit_and_frozen_inputs(self):
        x, y, vx, vy = fixture()
        y, vy = y > 0, vy > 0
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "binary"
            result = fit.fit_cell(root, x, y, vx, vy, 1, 1e-4, PROTOCOL, classes=2)
            fit.verify_cell(root, x, y, vx, vy, 1, 1e-4, PROTOCOL, classes=2)
            probs = load_npz(root / "candidate-00002.npz")["validation"]
            self.assertEqual(probs.shape, (len(vy), 2))
            self.assertEqual(result["classes"], 2)
            self.assertGreater(result["gradients"][0]["weight_penalty"], 0)
            with self.assertRaises(ValueError):
                fit.verify_cell(root, x, y, vx, vy, 1, 0, PROTOCOL, classes=2)

    def test_train_only_normalization_wrong_selection_and_prediction_tamper(self):
        x, y, vx, vy = fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cell"
            fit.fit_cell(root, x, y, vx, vy, 0, 0, PROTOCOL)
            norm = load_npz(root / "normalization.npz")
            np.testing.assert_array_equal(norm["mean"], old.normalize(x)["mean"])
            self.assertFalse(
                np.array_equal(
                    norm["mean"], old.normalize(np.concatenate([x, vx]))["mean"]
                )
            )
            with self.assertRaisesRegex(ValueError, "binding"):
                fit.verify_cell(root, x, y, vx * 2, vy, 0, 0, PROTOCOL)
            path = root / "selection.json"
            record = read(path)
            bad = copy.deepcopy(record)
            bad["selected"]["update"] = 999
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "selection"):
                fit.verify_cell(root, x, y, vx, vy, 0, 0, PROTOCOL)
            path.write_text(json.dumps(record))
            candidate = root / "candidate-00002.npz"
            raw = load_npz(candidate)
            raw["validation"] = raw["validation"][:, ::-1]
            np.savez_compressed(candidate, **raw)
            with self.assertRaisesRegex(ValueError, "exact replay"):
                fit.verify_cell(root, x, y, vx, vy, 0, 0, PROTOCOL)

    def test_family_selection_is_mean_of_seedwise_validation_minima(self):
        def record(alpha, seed, first, last):
            candidates = [
                dict(update=10, validation=dict(nll=first)),
                dict(update=20, validation=dict(nll=last)),
            ]
            return dict(
                alpha=alpha,
                seed=seed,
                candidates=candidates,
                selected=fit.select(candidates),
                test_mse=999 if alpha else 0,
            )

        families = {
            "0": [record(0, 0, 1.0, 0.9), record(0, 1, 0.2, 1.5)],
            "0.0001": [record(1e-4, 0, 0.5, 0.6), record(1e-4, 1, 0.5, 0.5)],
        }
        selected = fit.select_family(families)
        self.assertEqual(selected["alpha"], 1e-4)
        self.assertEqual(selected["mean_validation_nll"], 0.5)
        self.assertEqual(selected["selections"]["1"]["update"], 10)
        ties = {"0": [record(0, 0, 0.5, 0.5)], "0.01": [record(0.01, 0, 0.5, 0.5)]}
        self.assertEqual(fit.select_family(ties)["alpha"], 0)
        with self.assertRaisesRegex(ValueError, "unmatched seeds"):
            fit.select_family(dict(families, **{"1": [record(1, 8, 0.1, 0.1)]}))
        with self.assertRaises(ValueError):
            fit.select_family({"0": [record(0, 0, float("nan"), 0.5)]})

    def test_ridge_matches_analytical_normal_equations_and_unpenalized_intercept(self):
        x, y, vx, vy = fixture()
        y, vy = y + 10.0, vy + 10.0
        protocol = dict(ridge_ranks=[], ridge_alphas=[0.1])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ridge"
            record = fit.fit_ridge(root, x, y, vx, vy, protocol)
            fit.verify_ridge(root, x, y, vx, vy, protocol)
            xd = x.astype(float)
            std = np.maximum(xd.std(0), 0.01)
            z = (xd - xd.mean(0)) / std
            coefficient = np.linalg.solve(
                z.T @ z / len(z) + 0.1 * np.eye(4), z.T @ (y - y.mean()) / len(y)
            )
            predicted = (vx - xd.mean(0)) / std @ coefficient + y.mean()
            np.testing.assert_allclose(
                fit.predict_ridge(root, vx), predicted, rtol=1e-13, atol=1e-13
            )
            params = load_npz(root / record["selected"]["checkpoint"])
            self.assertEqual(float(params["intercept"]), y.mean())
            saved = load_npz(root / "ridge-predictions-000.npz")
            np.testing.assert_array_equal(
                saved["validation_clipped"], np.clip(saved["validation"], 0, 2)
            )
            self.assertNotIn("nll", record["selected"]["validation"])
            with self.assertRaises(FileExistsError):
                fit.fit_ridge(root, x, y, vx, vy, protocol)

    def test_ridge_all_candidates_replay_train_only_and_fail_closed_tamper(self):
        x, y, vx, vy = fixture()
        protocol = dict(ridge_ranks=[1, 2], ridge_alphas=[1e-6, 1, 100])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "ridge"
            record = fit.fit_ridge(root, x, y, vx, vy, protocol)
            self.assertEqual(len(record["candidates"]), 9)
            fit.verify_ridge(root, x, y, vx, vy, protocol)
            transform = load_npz(root / "ridge-transform.npz")
            np.testing.assert_array_equal(transform["mean"], x.astype(float).mean(0))
            with self.assertRaisesRegex(ValueError, "binding"):
                fit.verify_ridge(root, x, y, vx, vy[::-1], protocol)
            path = root / "ridge-selection.json"
            bad = copy.deepcopy(record)
            bad["selected"] = dict(bad["selected"], rank=999)
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "selection"):
                fit.verify_ridge(root, x, y, vx, vy, protocol)
            path.write_text(json.dumps(record))
            params = load_npz(root / "ridge-000.npz")
            params["coefficient"][0] += 1
            np.savez_compressed(root / "ridge-000.npz", **params)
            with self.assertRaisesRegex(ValueError, "coefficients"):
                fit.verify_ridge(root, x, y, vx, vy, protocol)

    def test_low_variance_reward_signal_must_not_be_discarded(self):
        # Both coordinates have similar marginal scale, but PCA's small
        # eigenvalue direction carries all of the second target's reward signal.
        u = np.tile([-1, -1, 1, 1], 20).astype(float)
        v = np.tile([-1, 1, -1, 1], 20).astype(float)
        x = np.stack([u + 0.03 * v, u - 0.03 * v], -1).astype(np.float32)
        protocol = dict(ridge_ranks=[1], ridge_alphas=[1e-6, 1e-2])
        with tempfile.TemporaryDirectory() as td:
            high = fit.fit_ridge(Path(td) / "high", x, u + 1, x, u + 1, protocol)
            low = fit.fit_ridge(Path(td) / "low", x, v + 1, x, v + 1, protocol)
            self.assertLess(high["selected"]["validation"]["mse"], 1e-5)
            self.assertEqual(low["selected"]["rank"], 2)
            self.assertLess(low["selected"]["validation"]["mse"], 1e-5)
            low_rank = [r for r in low["candidates"] if r["rank"] == 1]
            self.assertTrue(all(r["validation"]["mse"] > 0.99 for r in low_rank))
            fit.verify_ridge(Path(td) / "low", x, v + 1, x, v + 1, protocol)

    def test_ridge_selection_uses_raw_validation_error_and_fixed_ties(self):
        rows = [
            dict(rank=1, alpha=1, validation=dict(mse=0.4, clipped_mse=0.3)),
            dict(rank=2, alpha=1e-6, validation=dict(mse=0.5, clipped_mse=0.1)),
        ]
        self.assertEqual(fit._select_ridge(rows)["rank"], 1)
        ties = [
            dict(rank=2, alpha=1, validation=dict(mse=0.4)),
            dict(rank=1, alpha=1e-6, validation=dict(mse=0.4)),
            dict(rank=1, alpha=100, validation=dict(mse=0.4)),
        ]
        self.assertEqual(fit._select_ridge(ties)["alpha"], 100)

    def test_constant_features_and_unexpected_artifacts_fail_closed(self):
        x = np.ones((8, 3), np.float32)
        y = np.arange(8) % 3
        protocol = dict(ridge_ranks=[1], ridge_alphas=[1e-6, 100])
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "constant"
            record = fit.fit_ridge(root, x, y, x, y, protocol)
            self.assertEqual(record["selected"]["rank"], 1)
            self.assertEqual(record["selected"]["alpha"], 100)
            np.testing.assert_array_equal(
                fit.predict_ridge(root, x), np.full(8, y.mean())
            )
            fit.verify_ridge(root, x, y, x, y, protocol)
            (root / "unexpected.npz").write_bytes(b"not a retained candidate")
            with self.assertRaisesRegex(ValueError, "artifact inventory"):
                fit.verify_ridge(root, x, y, x, y, protocol)


if __name__ == "__main__":
    unittest.main()
