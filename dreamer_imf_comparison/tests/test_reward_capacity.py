import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
from dreamer_imf_compare import reward_capacity as study
from dreamer_imf_compare import reward_capacity_fit as fitting
from dreamer_imf_compare.reward_diagnosis_fit import fit_head
from dreamer_imf_compare.reward_readout_study import init_mlp, read
from dreamer_imf_compare.prior_correction_study import load
from dreamer_imf_compare.parallel_collection import _write_json
from dreamer_imf_compare.reward_diagnosis import check_seal


class CapacityTests(unittest.TestCase):
    def test_architecture_and_exact_128_reference_initializer(self):
        for seed in (0, 1, 2):
            a = fitting.init(jax.random.PRNGKey(seed), 6, 128)
            b = init_mlp(jax.random.PRNGKey(seed), 6)
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                np.testing.assert_array_equal(x, y)
        counts = []
        for width in study.PROTOCOL["widths"]:
            p = fitting.init(jax.random.PRNGKey(0), 6, width)
            counts.append(sum(x.size for x in jax.tree.leaves(p)))
        self.assertEqual(counts, sorted(set(counts)))
        self.assertEqual(len(list(study.cells(study.PROTOCOL))), 45)

    def test_matched_128_updates_and_independent_candidate_replay(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(4, 15, 6)).astype(np.float32)
        y = rng.integers(0, 3, (4, 15)).astype(np.float32)
        vx = x[:2] * 2 + 10
        vy = y[:2]
        p = study.protocol(True)
        with tempfile.TemporaryDirectory() as td:
            new, old = Path(td) / "new", Path(td) / "old"
            fitting.fit(new, x, y, vx, vy, 128, 0, p)
            fit_head(old, x, y, vx, vy, 0, p)
            for step in (1, 2):
                for a, b in zip(
                    jax.tree.leaves(load(new / f"checkpoint-{step:05d}.pkl")),
                    jax.tree.leaves(load(old / f"checkpoint-{step:05d}.pkl")),
                ):
                    np.testing.assert_array_equal(a, b)
            fitting.verify_fit(new, x, y, vx, vy, 128, 0, p)
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(
                    fitting.normalize(x)["mean"],
                    fitting.normalize(np.concatenate([x, vx]))["mean"],
                )
            path = new / "selection.json"
            record = read(path)
            record["selected"] = dict(record["selected"], update=999)
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "selection"):
                fitting.verify_fit(new, x, y, vx, vy, 128, 0, p)

    def test_ties_and_invalid_candidates(self):
        rows = [
            dict(update=500, validation=dict(nll=1.0)),
            dict(update=250, validation=dict(nll=1.0)),
        ]
        self.assertEqual(fitting.select(rows)["update"], 250)
        with self.assertRaises(ValueError):
            fitting.select([])
        with self.assertRaises(ValueError):
            fitting.select([dict(update=1, validation=dict(nll=float("nan")))])

    def test_tiny_full_pipeline_split_replay_and_input_immutability(self):
        rng = np.random.default_rng(8)
        data = dict(
            targets=rng.normal(size=(12, 15, 2560)).astype(np.float32),
            observations=rng.normal(size=(12, 15, 6)).astype(np.float32),
            actions=rng.normal(size=(12, 15, 2)).astype(np.float32),
            rewards=rng.integers(0, 3, (12, 15)).astype(np.float32),
            episode=np.repeat([1, 2, 3], 4),
            split=np.repeat([0, 1, 2], 4),
        )
        inputs = dict(
            previous=np.zeros_like(data["observations"]), distance=np.ones((12, 15))
        )
        before = copy.deepcopy(data)
        p = dict(
            study.protocol(True),
            widths=[64, 128, 512],
            updates=1,
            validation_period=1,
            bootstrap_reps=8,
        )
        context = (data, inputs, dict(radius=1.0), {})
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            _write_json(out / "manifest.json", dict(protocol=p, preflight=True))
            study.fit(out, context)
            raw, records = study.generate(out, context, verify_candidates=True)
            other, other_records = study.generate(out, context)
            for key in raw:
                np.testing.assert_array_equal(raw[key], other[key])
            self.assertEqual(records, other_records)
            np.testing.assert_array_equal(raw["indices__test"], np.arange(8, 12))
            r = study.report(raw, records, data, inputs, 1.0, p)
            self.assertEqual(r["additional_simulator_steps"], 0)
            self.assertEqual(len(records), 15)
            self.assertEqual(
                r["width_vs_128"]["hz-w128__selected__test"]["paired_delta"]["mse"][
                    "mean"
                ],
                0,
            )
            for k in data:
                np.testing.assert_array_equal(data[k], before[k])
            path = out / "heads/h-w64-s0/selection.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaises(ValueError):
                check_seal(out, "fitted")

    def test_seed_averaging_is_errors_not_prediction_ensemble(self):
        y = np.ones((2, 15))
        p0 = np.broadcast_to([1.0, 0.0, 0.0], (2, 15, 3))
        p1 = np.broadcast_to([0.0, 0.0, 1.0], (2, 15, 3))
        mean_error = (
            study.error_arrays(p0, y)["mse"] + study.error_arrays(p1, y)["mse"]
        ) / 2
        self.assertEqual(mean_error.mean(), 1.0)
        self.assertEqual(study.error_arrays((p0 + p1) / 2, y)["mse"].mean(), 0.0)

    def test_report_pairing_gap_and_improvement_flag_analytically(self):
        p = dict(study.PROTOCOL, groups=["observation", "h"], bootstrap_reps=30)
        data = dict(rewards=np.ones((18, 15)), episode=np.arange(18))
        inputs = dict(distance=np.ones((18, 15)))
        raw = {
            "indices__" + split: np.arange(6 * i, 6 * i + 6)
            for i, split in enumerate(("train", "validation", "test"))
        }
        for name, group, width, seed in study.cells(p):
            label = (
                1 if group == "observation" or width == 512 else (2 if seed == 1 else 0)
            )
            for checkpoint in ("selected", "final"):
                for split in ("train", "validation", "test"):
                    raw[f"{name}__{checkpoint}__{split}"] = np.broadcast_to(
                        np.eye(3)[label], (6, 15, 3)
                    )
        r = study.report(raw, {}, data, inputs, 1.0, p)
        self.assertEqual(
            r["seed_average_errors"]["h-w128__selected__test"]["mse"]["mean"], 1.0
        )
        contrast = r["width_vs_128"]["h-w512__selected__test"]
        self.assertEqual(contrast["paired_delta"]["mse"]["ci95"], [-1.0, -1.0])
        self.assertTrue(contrast["material_test_improvement"])
        self.assertFalse(
            r["width_vs_128"]["h-w64__selected__test"]["material_test_improvement"]
        )
        self.assertEqual(
            r["latent_observation_gaps"]["h-w512__selected__test"]["change_vs_128_gap"][
                "mse"
            ]["mean"],
            -1.0,
        )

    def test_exclusive_submission_and_wrong_binding_fail_closed(self):
        spec = importlib.util.spec_from_file_location(
            "submit_capacity",
            Path(__file__).parents[1] / "scripts/submit_reward_capacity.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dependency = root / "dep"
            dependency.mkdir()
            (dependency / "verified.json").write_text(
                json.dumps(dict(exact_replay=True))
            )
            bundle = root / "bundle"
            bundle.write_bytes(b"bundle")
            args = (
                root,
                root / "out",
                dependency,
                "abc",
                bundle,
                hashlib.sha256(b"bundle").hexdigest(),
            )
            with patch.object(
                module.subprocess, "check_output", side_effect=["abc", ""]
            ), patch.object(
                module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "123\n", ""),
            ):
                module.submit(*args)
            with patch.object(
                module.subprocess, "check_output", side_effect=["abc", ""]
            ), patch.object(module.subprocess, "run") as call:
                with self.assertRaises(FileExistsError):
                    module.submit(*args)
                call.assert_not_called()
            with patch.object(
                module.subprocess, "check_output", return_value="wrong"
            ), patch.object(module.subprocess, "run") as call:
                with self.assertRaises(ValueError):
                    module.submit(*args)
                call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
