import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import jax
import numpy as np
from test_prior_correction_runner import fixture, ToyAdapter
from imf_dreamer_jax import prior_correction as field
from dreamer_imf_compare.prior_correction import normalization
from dreamer_imf_compare.prior_correction_study import dump
from dreamer_imf_compare.parallel_collection import _write_json, _write_npz
from dreamer_imf_compare.reward_diagnosis import (
    fit,
    check_fit,
    generate,
    report,
    check_seal,
    config,
)
from dreamer_imf_compare.reward_diagnosis_data import GROUPS, posterior_draws


class RunnerTests(unittest.TestCase):
    def test_submitter_rejects_duplicate_and_ambiguous_retry(self):
        script = Path(__file__).parents[1] / "scripts/submit_reward_diagnosis.py"
        spec = importlib.util.spec_from_file_location("diagnosis_submit", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent, correction = root / "parent", root / "correction"
            parent.mkdir()
            correction.mkdir()
            (parent / "report.json").write_text("{}")
            (correction / "verified.json").write_text("{}")
            bundle = root / "source.bundle"
            bundle.write_bytes(b"source")
            args = (
                root,
                root / "output",
                parent,
                correction,
                "abc",
                bundle,
                hashlib.sha256(b"source").hexdigest(),
            )
            with patch.object(
                module.subprocess, "check_output", side_effect=["abc", ""]
            ), patch.object(
                module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "12345\n", ""),
            ) as scheduler:
                module.submit(*args)
                self.assertEqual(scheduler.call_count, 1)
            with patch.object(
                module.subprocess, "check_output", side_effect=["abc", ""]
            ), patch.object(module.subprocess, "run") as scheduler:
                with self.assertRaises(FileExistsError):
                    module.submit(*args)
                scheduler.assert_not_called()
            ambiguous = (root, root / "ambiguous", *args[2:])
            with patch.object(
                module.subprocess, "check_output", side_effect=["abc", ""]
            ), patch.object(
                module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 0, "unrecognized response", ""
                ),
            ):
                with self.assertRaises(RuntimeError):
                    module.submit(*ambiguous)
            evidence = json.loads(
                (root / "ambiguous/submission-uncertain.json").read_text()
            )
            self.assertFalse(evidence["automatic_retry_allowed"])

    def test_matched_fit_transfer_replay_and_tamper_rejection(self):
        data = fixture()
        data["mode"] = np.zeros(len(data["start"]), int)
        adapter = ToyAdapter()
        teacher = adapter.teacher
        teacher.frozen_digest = lambda: "frozen"
        before = copy.deepcopy(data)
        previous = np.zeros_like(data["observations"])
        distance = np.ones_like(data["rewards"])
        context = (teacher, data, previous, distance, Path("/unused"))
        protocol = config(True)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            correction = out / "correction"
            correction.mkdir()
            dump(
                correction / "checkpoint.pkl",
                field.init(jax.random.PRNGKey(1), 2, 2, 8, 1),
            )
            _write_npz(correction / "normalization.npz", normalization(data, 2))
            _write_json(
                out / "manifest.json",
                dict(
                    protocol=protocol,
                    preflight=True,
                    frozen_digest="frozen",
                    correction=dict(
                        path=str(correction), selected=dict(checkpoint="checkpoint.pkl")
                    ),
                ),
            )
            fit(out, context)
            check_fit(out, context)
            for g in GROUPS:
                self.assertTrue((out / "heads" / f"{g}-0" / "selection.json").is_file())

            def draws(*args):
                ix = args[3]
                yield 0, ix, data["targets"][ix]

            with patch(
                "dreamer_imf_compare.reward_diagnosis.FrozenPrior", return_value=adapter
            ), patch("dreamer_imf_compare.reward_diagnosis.posterior_draws", draws):
                raw, dig = generate(out, context)
                other, dig2 = generate(out, context)
            self.assertEqual(dig, dig2)
            for k, v in raw.items():
                np.testing.assert_array_equal(v, other[k])
            result = report(raw, data, distance, 1.0, 20)
            self.assertEqual(set(result["seed_average_errors"]), set(GROUPS))
            self.assertIn("corrected__hz-0", result["metrics"])
            self.assertIn("draw__hz-0", result["metrics"])
            self.assertNotIn("corrected__observation-0", result["metrics"])
            for k in data:
                np.testing.assert_array_equal(data[k], before[k])
            # Altering a candidate must be rejected before loading its checkpoint.
            target = out / "heads/hz-0/selection.json"
            original = target.read_bytes()
            target.write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                check_seal(out, "fitted")

    def test_refiltering_reproducible_shared_anchors_and_no_simulator(self):
        data = dict(
            episode=np.array([52] * 4), anchor=np.array([2] * 4), plan=np.arange(4)
        )
        raw = [dict(v=i) for i in range(16)]
        history = dict(
            previous_action=np.zeros(2),
            actions=np.ones((16, 2)),
            observe_seeds=np.arange(16),
        )
        teacher = type("T", (), {})()
        teacher.initial = lambda: 0
        teacher.observe = lambda carry, obs, action, seed: (
            carry + 1,
            np.array([carry, obs["v"], seed % 100, action[0]], np.float32),
        )
        with patch(
            "dreamer_imf_compare.reward_diagnosis_data.load_raw_history",
            return_value=(history, raw),
        ):
            a = list(posterior_draws(teacher, data, Path("/unused"), np.arange(4), 2))
            b = list(posterior_draws(teacher, data, Path("/unused"), np.arange(4), 2))
        self.assertEqual(len(a), 2)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left[2], right[2])
            for j in range(1, 4):
                np.testing.assert_array_equal(left[2][0], left[2][j])


if __name__ == "__main__":
    unittest.main()
