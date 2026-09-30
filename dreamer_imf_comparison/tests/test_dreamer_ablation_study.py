import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from dreamer_imf_compare import dreamer_ablation_study as s
from dreamer_imf_compare.dreamer_ablation_runner import _settings


class StudyTests(unittest.TestCase):
    def test_exact_paired_cells(self):
        p = s.read(s.PROTOCOL)
        self.assertEqual(len(s.cells(p, "preflight")), 3)
        rows = s.cells(p, "training")
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            {(r["arm"], r["seed"]) for r in rows},
            {(a, w) for a in p["arms"] for w in p["seeds"]},
        )
        self.assertEqual(p["preflight"]["train_ratio"], p["train_ratio"])

    def test_exclusive_and_finite(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "evidence.json"
            s.publish(path, {"finite": 1})
            with self.assertRaises(FileExistsError):
                s.publish(path, {})
            self.assertEqual(s.read(path), {"finite": 1})
        for x in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                s.finite({"metrics": [x]})

    def test_cell_hash_budget_and_marker(self):
        p = s.read(s.PROTOCOL)
        c = s.cells(p, "preflight")[0]
        with tempfile.TemporaryDirectory() as root:
            d = s.cell_dir(root, "preflight", c)
            d.mkdir(parents=True)
            (Path(root) / "verified").mkdir()
            (d / "checkpoint.pkl").write_bytes(b"test-checkpoint")
            r = dict(
                arm=c["arm"],
                seed=c["seed"],
                preflight=True,
                completed=True,
                from_scratch=True,
                finite_parameters=True,
                upstream_commit=p["upstream_commit"],
                native_steps=4096,
                agent_transitions=2048,
                replay_rows=2080,
                reset_rows=32,
                learner_updates=2,
                evaluations=[
                    dict(
                        native_steps=4096,
                        returns=[1.0],
                        episode_seeds=[123],
                        episode_native_steps=[1000],
                        mean_return=1.0,
                    )
                ],
                checkpoints=[
                    dict(path="checkpoint.pkl", sha256=s.filehash(d / "checkpoint.pkl"))
                ],
            )
            r["settings"] = _settings(p, True)
            s.publish(d / "complete.json", r)
            with mock.patch.object(s, "manifest", return_value=({}, p)):
                s.verify_cell(root, "preflight", c)
                s.verify_cell(root, "preflight", c)
                broken = copy.deepcopy(r)
                broken["native_steps"] += 2
                (d / "complete.json").write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    s.verify_cell(root, "preflight", c)
                (d / "complete.json").write_text(json.dumps(r))
                (d / "checkpoint.pkl").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    s.verify_cell(root, "preflight", c)

    def test_submit_never_duplicates_uncertain_attempt(self):
        p = s.read(s.PROTOCOL)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "submissions").mkdir()
            s.publish(Path(root) / "submissions/preflight-intent.json", {})
            with mock.patch.object(s, "manifest", return_value=({}, p)):
                with self.assertRaises(FileExistsError):
                    s.submit(root, "preflight", "gpu-l40s")


if __name__ == "__main__":
    unittest.main()
