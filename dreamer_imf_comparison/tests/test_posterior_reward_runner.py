"""Independent online accounting tests; no simulator or cluster dependency."""

from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import posterior_reward_runner as runner
from dreamer_imf_compare import posterior_reward_verify as verify
from dreamer_imf_compare.parallel_collection import _write_npz
from dreamer_imf_compare.posterior_reward_protocol import write_json_exclusive


def episode():
    value = dict(
        reward=np.r_[0.0, np.tile([0.0, 1.0, 2.0, 0.0, 0.0], 100)].astype(np.float32),
        is_first=np.arange(501) == 0,
        is_last=np.arange(501) == 500,
        action=np.zeros((501, 2), np.float32),
        previous_action=np.zeros((501, 2), np.float32),
    )
    for name in ("original", "student", "teacher", "control"):
        value["log/" + name + "_reward"] = value["reward"].copy()
    return value


def collection():
    rows = []
    for worker in range(16):
        for ep in range(5):
            row = episode()
            row.update(
                worker=np.full(501, worker, np.int32),
                native_delta=np.r_[0, np.full(500, 2)].astype(np.int32),
                episode_id=np.full(501, worker * 100000 + ep, np.int32),
                collection_mode=np.zeros(501, np.int32),
                proposed_action=row["action"].copy(),
            )
            rows.append(row)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


class OnlineAccounting(unittest.TestCase):
    def test_both_optimizer_clocks_are_explicit_and_integral(self):
        value = dict(
            params={
                "opt/step/value": np.array(48985),
                "reward_aux_opt/step/value": np.array(0),
            }
        )
        self.assertEqual(
            verify.optimizer_clocks(value), dict(original=48985, auxiliary=0)
        )
        for invalid in (-1, 0.5, np.nan, [0]):
            value["params"]["reward_aux_opt/step/value"] = np.asarray(invalid)
            with self.assertRaises(ValueError):
                verify.optimizer_clocks(value)

    def test_compile_monitor_records_real_jax_events_separately(self):
        import jax

        runner.start_compilation_monitor()
        before = (
            runner.compilation_evidence()["events"]
            .get("/jax/core/compile/test", {})
            .get("seconds", 0)
        )
        jax.monitoring.record_event_duration_secs("/jax/core/compile/test", 0.125)
        after = runner.compilation_evidence()["events"]["/jax/core/compile/test"][
            "seconds"
        ]
        self.assertEqual(after - before, 0.125)

    def test_native_return_and_per_readout_errors(self):
        data = episode()
        result = verify.verify_episode(data)
        self.assertEqual(result["return"], 300.0)
        self.assertEqual(result["native_steps"], 1000)
        self.assertEqual(result["positive_decisions"], 200)
        for value in result["reward_errors"].values():
            self.assertEqual(value["mse"], 0.0)
        data["log/teacher_reward"][:] = 0
        result = verify.verify_episode(data)
        self.assertEqual(result["reward_errors"]["student"]["mse"], 0.0)
        self.assertGreater(result["reward_errors"]["teacher"]["mse"], 0.0)

    def test_evaluation_timing_and_invalid_labels_fail_closed(self):
        for key, index, value in (
            ("reward", 0, 2),
            ("reward", 1, 3),
            ("is_first", 1, True),
            ("is_last", 450, True),
            ("previous_action", 50, 1),
            ("action", 100, 1.1),
        ):
            with self.subTest(key=key):
                data = episode()
                data[key][index] = value
                with self.assertRaises(ValueError):
                    verify.verify_episode(data)

    def test_zero_reward_coverage_is_not_success(self):
        data = episode()
        data["reward"][:] = 0
        result = verify.verify_episode(data)
        self.assertEqual(result["positive_decisions"], 0)
        self.assertIsNone(result["reward_errors"]["student"]["positive"]["mse"])

    def test_complete_collection_reconstructs_decisions_resets_and_native_cost(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            (directory / "transitions").mkdir()
            _write_npz(directory / "transitions/chunk-00000.npz", collection())
            result, paths = verify.verify_collection(directory)
            self.assertEqual(
                result,
                dict(
                    native_steps=80000,
                    decisions=40000,
                    rows=40080,
                    reset_rows=80,
                    positive_decisions=16000,
                ),
            )
            self.assertEqual(len(paths), 1)

    def test_collection_corruptions_do_not_pass_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            (directory / "transitions").mkdir()
            (directory / "transitions/chunk-00000.npz").touch()
            for field, index, value in (
                ("native_delta", 0, 2),
                ("previous_action", 10, 1),
                ("episode_id", 100, 11),
                ("collection_mode", 1, 1),
                ("worker", 0, 16),
                ("action", 500, 1),
                ("proposed_action", 40, 0.5),
                ("reward", 0, 1),
                ("is_last", 499, True),
            ):
                with self.subTest(field=field):
                    data = collection()
                    data[field][index] = value
                    with mock.patch.object(verify, "load_arrays", return_value=data):
                        with self.assertRaises(ValueError):
                            verify.verify_collection(directory)

    def test_evidence_is_exclusive_and_replay_exact(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pkl"
            runner.save_pickle(path, {"x": np.array([1.0])})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.save_pickle(path, {"x": np.array([2.0])})
            self.assertEqual(original, path.read_bytes())
            path = Path(td) / "probe.npz"
            arrays = {"x": np.array([1.0], np.float32)}
            _write_npz(path, arrays)
            runner.compare_arrays(path, arrays)
            with self.assertRaisesRegex(ValueError, "replay"):
                runner.compare_arrays(
                    path, {"x": np.nextafter(arrays["x"], np.float32(2.0))}
                )
            with self.assertRaisesRegex(ValueError, "fields"):
                runner.compare_arrays(path, {})
            with self.assertRaisesRegex(ValueError, "replay"):
                runner.compare_arrays(path, {"x": arrays["x"].astype(np.float64)})

    def test_retained_episode_pool_keeps_real_environment_identity(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            for ep in range(10):
                _write_npz(directory / f"eval-40000-{ep}.npz", episode())
            write_json_exclusive(
                directory / "evaluation-40000.json",
                dict(episodes=[dict(seed=10 + ep) for ep in range(10)]),
            )
            result = runner.load_evaluations(directory, 40000)
            self.assertEqual(result["reward"].shape, (10, 501))
            np.testing.assert_array_equal(result["episode_seed"], np.arange(10, 20))

    def test_preflight_fixture_is_synthetic_and_nonmutating(self):
        from dreamer_imf_compare.posterior_reward_probe import validate_histories

        batch = {k: np.repeat(v[None, :65], 16, 0) for k, v in episode().items()}
        for key in ("position", "to_target", "velocity"):
            batch[key] = np.zeros((16, 65, 2), np.float32)
        batch["is_terminal"] = np.zeros((16, 65), bool)
        # Parent minibatches may contain proposed Gaussian actions, whereas the
        # prediction interface requires bounded simulator-executed actions.
        batch["action"][:] = np.linspace(-2.5, 2.5, 65)[None, :, None]
        before = {k: v.copy() for k, v in batch.items()}
        result = runner._probe_fixture(batch)
        proxy = types.SimpleNamespace(
            obs_space={
                k: types.SimpleNamespace(shape=batch[k].shape[2:])
                for k in (
                    "position",
                    "to_target",
                    "velocity",
                    "reward",
                    "is_first",
                    "is_last",
                    "is_terminal",
                )
            }
        )
        validate_histories(proxy, result)
        np.testing.assert_array_equal(
            result["action"],
            np.clip(np.tile(before["action"][:10], (1, 8, 1))[:, :501], -1, 1),
        )
        self.assertEqual(result["reward"].shape, (10, 501))
        self.assertEqual(result["is_first"].sum(), 10)
        self.assertEqual(result["is_last"].sum(), 10)
        self.assertFalse(result["is_terminal"].any())
        for k in before:
            np.testing.assert_array_equal(batch[k], before[k])
        np.testing.assert_array_equal(
            result["previous_action"][:, 1:], result["action"][:, :-1]
        )
        # Only the synthetic fixture is converted. Real retained histories must
        # still fail closed for even a single out-of-bounds action.
        result["action"][0, 10, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "simulator bounds"):
            validate_histories(proxy, result)
        for invalid in (np.nan, np.inf, -np.inf):
            batch["action"][0, 0, 0] = invalid
            with self.assertRaisesRegex(ValueError, "nonfinite.*fixture.*action"):
                runner._probe_fixture(batch)

    def test_checkpoint_import_preserves_every_existing_optimizer_value(self):
        import sys

        with tempfile.TemporaryDirectory() as td:
            parent = dict(
                params={"dyn/w": np.ones(2), "opt/m": np.ones(2) * 4},
                counters=dict(updates=10, actions=20),
            )
            runner.save_pickle(Path(td) / "checkpoint_200000.pkl", parent)
            initial = dict(
                params={
                    **{k: np.zeros_like(v) for k, v in parent["params"].items()},
                    "reward_student/w": np.zeros(2),
                    "reward_aux_opt/m": np.zeros(2),
                },
                counters={},
            )
            agent = mock.Mock()
            state = [initial]
            agent.save.side_effect = lambda: state[0]
            agent.load.side_effect = lambda value: state.__setitem__(0, value)
            fake = types.SimpleNamespace(
                import_rmt_heads=lambda *a: {"reward_student/w": np.array([2.0, 3.0])}
            )
            with mock.patch.dict(
                sys.modules, {"dreamer_imf_compare.posterior_reward_agent": fake}
            ):
                result = runner.load_continuation(agent, td, td, 901)
            self.assertEqual(result["readout_seed"], 0)
            self.assertEqual(state[0]["counters"], parent["counters"])
            for k, v in parent["params"].items():
                np.testing.assert_array_equal(state[0]["params"][k], v)
            np.testing.assert_array_equal(state[0]["params"]["reward_aux_opt/m"], 0)

    def test_scheduler_allocation_summary_requires_all_nine_successes(self):
        def accounting(args, **kwargs):
            job = args[args.index("-j") + 1]
            ids = [f"{job}_{i}" for i in range(9)] if job == "102" else [job]
            return "\n".join(f"{i}|COMPLETED|0:0|12|gres/gpu=1|" for i in ids)

        with mock.patch.object(
            verify,
            "read",
            side_effect=lambda p: {
                "job": {"preflight": "101", "cells": "102", "probes": "103"}[p.stem]
            },
        ):
            with mock.patch.object(
                verify.subprocess, "check_output", side_effect=accounting
            ):
                result = verify.allocation_runtime("/unused")
                self.assertEqual(len(result["records"]), 11)
                self.assertEqual(result["allocated_wall_seconds"], 132)
            with mock.patch.object(
                verify.subprocess, "check_output", return_value="101|PENDING|0:0|0||"
            ):
                with self.assertRaises(ValueError):
                    verify.allocation_runtime("/unused")


if __name__ == "__main__":
    unittest.main()
