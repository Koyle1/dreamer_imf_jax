import copy
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare.joint_diagnostics import JointSchedule, assess, retained_views
from dreamer_imf_compare.staged_runner import Schedule
from dreamer_imf_compare import joint_study as study


class JointTests(unittest.TestCase):
    def test_only_freeze_mask_changes(self):
        old, new = Schedule(), JointSchedule()
        metrics = dict(
            calibration=dict(
                shared_core_model_grad_norm=2.0, shared_core_dynamics_grad_norm=0.5
            ),
            quality_gate=dict(normalized_latent_mse_5=0.1, normalized_reward_mse_5=0.0),
        )
        for native in range(0, 500001, 1000):
            if native % 50000 == 0:
                old.observe(native, metrics)
                new.observe(native, metrics)
            self.assertEqual(
                new.controls(native), dict(old.controls(native), transition_only=False)
            )
        self.assertTrue(old.controls(490000)["transition_only"])
        self.assertFalse(new.controls(490000)["transition_only"])

    def batch(self):
        return dict(
            reward=np.zeros((16, 64), np.float32),
            action=np.arange(16 * 64 * 2).reshape(16, 64, 2),
            is_first=np.zeros((16, 64), bool),
        )

    def test_no_synthetic_positive_coverage(self):
        batch = self.batch()
        views = retained_views(batch)
        self.assertEqual(set(views), {"ordinary_prefix"})
        self.assertEqual(views["ordinary_prefix"]["reward"].shape, (2, 16))
        batch["reward"][12, 50] = 1
        views = retained_views(batch)
        selected = views["positive_selected"]
        self.assertEqual(np.count_nonzero(selected["reward"]), 1)
        self.assertEqual(selected["reward"].shape, (1, 16))
        np.testing.assert_array_equal(selected["action"], batch["action"][12:13, 45:61])
        selected["reward"][:] = 0
        self.assertEqual(batch["reward"].sum(), 1)

    def rows(self):
        row = dict(
            valid_paths=20,
            positive_paths=10,
            zero_paths=10,
            observation_mse=0.1,
            persistence_mse=1.0,
            reward_mse=0.1,
            zero_reward_mse=0.5,
            positive_reward_mse=0.1,
            positive_zero_reward_mse=1.0,
        )
        return dict(model=row, shuffled_actions=dict(row, observation_mse=1.0))

    def test_negative_controls_and_positive_control(self):
        rows = self.rows()
        self.assertEqual(assess(rows)["status"], "beats_controls_on_this_batch")
        self.assertFalse(assess(rows)["useful_control_certified"])
        self.assertFalse(assess(rows)["influences_training"])
        # Unconditional/action-insensitive predictor cannot beat shuffled control.
        rows["shuffled_actions"] = copy.deepcopy(rows["model"])
        self.assertEqual(assess(rows)["status"], "fails_negative_controls")
        rows = self.rows()
        rows["model"]["reward_mse"] = rows["model"]["zero_reward_mse"]
        self.assertEqual(assess(rows)["status"], "fails_negative_controls")
        rows = self.rows()
        rows["model"]["positive_paths"] = 0
        self.assertEqual(assess(rows)["status"], "insufficient_reward_coverage")
        rows["model"]["reward_mse"] = float("nan")
        with self.assertRaises(ValueError):
            assess(rows)

    def test_protocol_is_one_seed_imf_and_unchanged_budget(self):
        base = Path(__file__).parents[1]
        p = json.loads((base / "joint_protocol.json").read_text())
        old = json.loads((base / "staged_protocol.json").read_text())
        self.assertEqual(p["arms"], ["imf"])
        self.assertEqual(p["seeds"], [431])
        for key in (
            "native_steps",
            "action_repeat",
            "eval_at_native_steps",
            "eval_episodes",
            "envs",
            "batch_size",
            "batch_length",
            "train_ratio",
            "model_size",
            "preflight",
        ):
            self.assertEqual(p[key], old[key])

    def test_extra_verification_precedes_marker(self):
        with mock.patch.object(
            study.evidence, "read", return_value={}
        ), mock.patch.object(
            study, "verify_extra", side_effect=ValueError("bad extra evidence")
        ), mock.patch.object(
            study.staged, "_original_verify"
        ) as original:
            with self.assertRaisesRegex(ValueError, "bad extra"):
                study.verify_cell(
                    Path(tempfile.gettempdir()),
                    "training",
                    dict(index=0, arm="imf", seed=431),
                )
            original.assert_not_called()

    def test_other_cells_rejected_before_access(self):
        with mock.patch.object(study.evidence, "read") as read:
            for arm, seed, index in (
                ("gaussian", 431, 0),
                ("imf", 433, 0),
                ("imf", 431, 1),
            ):
                with self.assertRaises(ValueError):
                    study.verify_cell(
                        "unused", "training", dict(index=index, arm=arm, seed=seed)
                    )
            read.assert_not_called()

    def fixture(self):
        directory = Path(tempfile.mkdtemp(prefix="imf-joint-evidence-"))
        np.savez(directory / "recent.npz", reward=np.zeros((16, 65), np.float32))
        np.savez(
            directory / "diagnostic_batch.npz", reward=np.zeros((2, 16), np.float32)
        )
        clocks = {
            f"opt/state/{g}/3/count": 8
            for g in ("representation", "transition", "actor")
        }
        payload = dict(
            learner_unchanged=True,
            batch_path="recent.npz",
            learner_updates=8,
            optimizer_clocks=clocks,
            views={
                "ordinary_prefix": dict(
                    rows=self.rows(), assessment=assess(self.rows())
                )
            },
        )
        study.evidence.publish(directory / "coverage.json", payload)
        study.evidence.publish(
            directory / "diagnostic.json", dict(learner_unchanged=True)
        )
        with (directory / "checkpoint.pkl").open("wb") as file:
            pickle.dump(dict(params={k: np.array(v) for k, v in clocks.items()}), file)
        record = lambda path: dict(
            path=path, sha256=study.evidence.filehash(directory / path)
        )
        result = dict(
            native_steps=4096,
            learner_updates=8,
            checkpoints=[record("checkpoint.pkl")],
            staged=dict(
                protocol="imf-joint-one-seed-v1",
                native=4096,
                updates=8,
                controls=[
                    dict(
                        values=dict(
                            transition_only=False, actor_enabled=False, imag_horizon=5
                        )
                    ),
                    dict(
                        values=dict(
                            transition_only=False, actor_enabled=True, imag_horizon=15
                        )
                    ),
                ],
                freeze_checks=[dict(passed=True)],
                gate_interpretation="legacy scheduling heuristic; not a control certificate",
                batch_sha256=study.evidence.filehash(
                    directory / "diagnostic_batch.npz"
                ),
                diagnostics=[record("diagnostic.json")],
                coverage_diagnostics=[record("coverage.json")],
                retained_batches=[
                    dict(record("recent.npz"), positive_rewards=0, total_rewards=1040)
                ],
            ),
        )
        return directory, result, payload

    def test_extra_verifier_passes_positive_fixture_and_rejects_tamper(self):
        directory, result, _ = self.fixture()
        study.verify_extra(directory, result, True)
        (directory / "recent.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "digest"):
            study.verify_extra(directory, result, True)

    def test_extra_verifier_rejects_frozen_clock_even_with_rebound_digest(self):
        directory, result, payload = self.fixture()
        payload["optimizer_clocks"]["opt/state/representation/3/count"] = 7
        (directory / "coverage.json").write_text(json.dumps(payload))
        result["staged"]["coverage_diagnostics"][0]["sha256"] = study.evidence.filehash(
            directory / "coverage.json"
        )
        with self.assertRaisesRegex(ValueError, "clocks"):
            study.verify_extra(directory, result, True)

    def test_extra_verifier_rejects_false_control_assessment(self):
        directory, result, payload = self.fixture()
        payload["views"]["ordinary_prefix"]["assessment"][
            "useful_control_certified"
        ] = True
        (directory / "coverage.json").write_text(json.dumps(payload))
        result["staged"]["coverage_diagnostics"][0]["sha256"] = study.evidence.filehash(
            directory / "coverage.json"
        )
        with self.assertRaisesRegex(ValueError, "assessment"):
            study.verify_extra(directory, result, True)


if __name__ == "__main__":
    unittest.main()
