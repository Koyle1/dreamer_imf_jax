import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dreamer_imf_compare import reward_rmt_study as study
from dreamer_imf_compare.parallel_collection import _write_json, _write_npz
from dreamer_imf_compare.parallel_runner import load_npz
from dreamer_imf_compare.reward_diagnosis_data import group_features, validate_endpoint
from dreamer_imf_compare.reward_readout_study import read


class RMTStudyTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(64)
        n = 24
        data = dict(
            targets=rng.normal(size=(n, 15, 12)).astype(np.float32),
            observations=rng.normal(size=(n, 15, 6)).astype(np.float32),
            actions=rng.uniform(-1, 1, (n, 15, 2)).astype(np.float32),
            rewards=rng.integers(0, 3, (n, 15)).astype(np.float32),
            episode=np.repeat(np.arange(6), 4),
            split=np.repeat([0, 1, 2], 8),
            mode=np.repeat([0, 1, 0, 1, 0, 1], 4),
            plan=np.tile(np.arange(4), 6),
            anchor=np.full(n, 100),
        )
        inputs = dict(
            previous=np.zeros_like(data["observations"]),
            distance=np.where(data["rewards"] == 2, 0.5, 1.5),
        )
        stats = dict(
            h=data["targets"][..., :8],
            z=data["targets"][..., 8:],
            mean=data["targets"][..., 8:] * 0.8,
            std=np.ones_like(data["targets"][..., 8:]),
        )
        return data, inputs, stats

    def test_authentication_rejects_wrong_dependency_and_changed_source(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_json(root / "verified.json", dict(exact_replay=False))
            _write_json(root / "manifest.json", dict(source_commit="not-the-source"))
            with self.assertRaisesRegex(ValueError, "unauthenticated"):
                study.authenticate(root)
            with patch.object(study, "commit", return_value="different-source"):
                with self.assertRaisesRegex(ValueError, "source or fixed protocol"):
                    study.setup(root)

    def test_invalid_normalization_and_posterior_identity_fail_before_fitting(self):
        data, inputs, stats = self.fixture()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_json(root / "manifest.json", dict(preflight=False))
            altered = dict(stats, h=stats["h"] + 1)
            _write_npz(root / "posterior.npz", dict(indices=np.arange(24), **altered))
            study.fresh_seal(root, "prepared")

            def features(d, inp, group):
                return group_features(
                    group,
                    d["targets"],
                    d["observations"],
                    inp["previous"],
                    d["actions"],
                    deter=8,
                )

            with patch.object(study.capacity, "features", features):
                with self.assertRaisesRegex(ValueError, "retained h"):
                    study.feature_context(root, (data, inputs, {}, {}))

    def test_split_isolation_balancing_and_reward_timing(self):
        data, inputs, _ = self.fixture()
        study.validate_splits(data)
        broken = copy.deepcopy(data)
        broken["split"][0] = 1
        with self.assertRaisesRegex(ValueError, "leaked"):
            study.validate_splits(broken)
        eps = np.repeat(np.arange(40), 240)
        rows = study.balanced_subset(eps, 256, 71)
        counts = np.bincount(eps[rows])
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(len(np.unique(rows)), 256)
        validate_endpoint(inputs["distance"], data["rewards"], 1.0)
        # Reward 1 may have an inside or outside endpoint, but is never recast
        # as either missing native reward observation.
        inputs["distance"][data["rewards"] == 1] = 0.5
        validate_endpoint(inputs["distance"], data["rewards"], 1.0)
        self.assertFalse(np.array_equal(data["rewards"], inputs["distance"] <= 1.0))

    def test_paired_promotion_and_no_ensemble_shortcut(self):
        p = dict(study.PROTOCOL, bootstrap_reps=20)
        y = np.ones((6, 15))
        poor = np.broadcast_to([0.5, 0.4, 0.1], (6, 15, 3))
        good = np.broadcast_to([0.1, 0.8, 0.1], (6, 15, 3))
        base, candidate = [study.error_arrays(v, y, True) for v in (poor, good)]
        result = study.promotion(candidate, base, np.arange(6), 6, p)
        self.assertTrue(result["exploratory_screen_passed"])
        self.assertFalse(
            study.promotion(candidate, base, np.arange(6), 4, p)[
                "exploratory_screen_passed"
            ]
        )
        a = np.broadcast_to([1.0, 0.0, 0.0], (6, 15, 3))
        b = np.broadcast_to([0.0, 0.0, 1.0], (6, 15, 3))
        self.assertEqual(
            (
                study.error_arrays(a, y, True)["mse"]
                + study.error_arrays(b, y, True)["mse"]
            ).mean()
            / 2,
            1,
        )
        self.assertEqual(study.error_arrays((a + b) / 2, y, True)["mse"].mean(), 0)
        self.assertNotIn("nll", study.error_arrays(y, y, False))

    def test_full_tiny_pipeline_independent_replay_and_immutable_inputs(self):
        data, inputs, stats = self.fixture()
        before = copy.deepcopy((data, inputs))
        p = dict(
            study.protocol(True),
            updates=1,
            batch_size=16,
            alphas=[0.0, 0.01],
            kernel_steps=[0, 1],
            kernel_examples=12,
            bootstrap_reps=8,
            ridge_ranks=[2],
            ridge_alphas=[0.001, 0.1],
        )
        context = (
            data,
            inputs,
            dict(radius=1.0),
            dict(path="fixture", diagnosis=dict(frozen_digest="frozen")),
        )
        teacher = Mock()
        teacher.frozen_digest.return_value = "frozen"

        def features(data, inputs, group):
            return group_features(
                group,
                data["targets"],
                data["observations"],
                inputs["previous"],
                data["actions"],
                deter=8,
            )

        with tempfile.TemporaryDirectory() as td, patch.object(
            study.capacity, "features", features
        ):
            out = Path(td)
            _write_json(out / "manifest.json", dict(protocol=p, preflight=True))
            _write_npz(out / "posterior.npz", dict(indices=np.arange(24), **stats))
            study.fresh_seal(out, "prepared")
            chosen = study.fit_models(out, context)
            self.assertEqual(
                set(chosen),
                set(study.INPUTS)
                | {"endpoint-" + g for g in study.ENDPOINT_INPUTS}
                | {"ridge-" + g for g in study.GROUPS},
            )
            study.evaluate(out, context)
            report = read(out / "report.json")
            self.assertEqual(report["coverage"]["train"]["episodes"], 2)
            self.assertEqual(report["coverage"]["test"]["transitions"], 120)
            self.assertEqual(report["protocol"]["additional_simulator_steps"], 0)
            raw = load_npz(out / "predictions.npz")
            np.testing.assert_array_equal(raw["indices__test"], np.arange(16, 24))
            self.assertEqual(
                report["selections"]["endpoint-h"]["alpha"],
                report["selections"]["h"]["alpha"],
            )
            self.assertIn("numerical_rank", report["diagnostics"]["gradpca"]["h"])
            ridge_pair = report["comparisons"]["ridge-h__vs_dense_early"]
            self.assertIn("cumulative_mse", ridge_pair["paired_deltas"])
            self.assertNotIn("nll", ridge_pair["paired_deltas"])
            self.assertIn("no NLL", ridge_pair["probabilistic_screen"])
            with patch.object(
                study, "frozen_teacher", return_value=(teacher, Path(td))
            ), patch.object(
                study.posterior, "extract_statistics", return_value=stats
            ), patch.object(
                study, "authenticate", return_value=context
            ), patch.object(
                study, "commit", return_value="fixture"
            ):
                study.verify(out, context)
            self.assertTrue(read(out / "verified.json")["exact_replay"])
            study.exact_inventory(out, "evaluated", extras=("verified.json",))
            _write_json(out / "unregistered.json", {})
            with self.assertRaisesRegex(ValueError, "unregistered"):
                study.exact_inventory(out, "evaluated", extras=("verified.json",))
            with self.assertRaises(FileExistsError):
                study.fresh_seal(out, "prepared")
        for original, current in zip(before, (data, inputs)):
            for key in original:
                np.testing.assert_array_equal(original[key], current[key])

    def test_exact_prediction_inventory_and_tamper_rejection(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            study.equal_arrays(dict(a=np.zeros(2)), {}, "test")
        with self.assertRaises(ValueError):
            study.equal_arrays(dict(a=np.zeros(2)), dict(a=np.ones(2)), "test")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            _write_json(out / "a.json", dict(value=1))
            study.fresh_seal(out, "prepared")
            # Simulate external corruption, not a production overwrite path.
            (out / "a.json").write_text("{}")
            with self.assertRaises(ValueError):
                study.exact_inventory(out, "prepared")

    def test_submitter_exclusive_and_ambiguity_fail_closed(self):
        spec = importlib.util.spec_from_file_location(
            "submit_rmt", Path(__file__).parents[1] / "scripts/submit_reward_rmt.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "source.bundle"
            bundle.write_bytes(b"fixture")
            commit = "a" * 40
            sha = hashlib.sha256(b"fixture").hexdigest()
            output, dep = root / "output", root / "dep"
            dep.mkdir()
            (dep / "verified.json").write_text("{}")
            with patch.object(module, "validate_dependency"), patch.object(
                module.subprocess,
                "check_output",
                side_effect=[commit, "", commit + " refs/heads/main\n"],
            ), patch.object(
                module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "unclear", ""),
            ):
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    module.submit(root / "source", output, dep, commit, bundle, sha)
            self.assertFalse(
                module.read(output / "submission-uncertain.json")[
                    "automatic_retry_allowed"
                ]
            )
            self.assertTrue((output / "intent.json").exists())
            with patch.object(module, "validate_dependency"), patch.object(
                module.subprocess,
                "check_output",
                side_effect=[commit, "", commit + " HEAD\n"],
            ), patch.object(module.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    module.submit(root / "source", output, dep, commit, bundle, sha)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
