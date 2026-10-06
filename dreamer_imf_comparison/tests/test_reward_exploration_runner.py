"""Independent accounting, action semantics, and no-interaction probe tests."""

from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import reward_exploration_runner as runner
from dreamer_imf_compare import reward_exploration_verify as verify
from dreamer_imf_compare import reward_exploration_protocol as protocol


def episode():
    data = dict(
        reward=np.r_[0.0, np.tile([0.0, 1.0, 2.0, 0.0, 0.0], 100)].astype(np.float32),
        is_first=np.arange(501) == 0,
        is_last=np.arange(501) == 500,
        action=np.zeros((501, 2), np.float32),
        previous_action=np.zeros((501, 2), np.float32),
    )
    for key in ("log/control_reward", "log/original_reward"):
        data[key] = data["reward"].copy()
    for key in (
        "log/disagreement",
        "log/action_entropy",
        "log/feature_rms",
        "log/feature_zscore_rms",
    ):
        data[key] = np.ones(501, np.float32)
    return data


class RunnerTests(unittest.TestCase):
    def test_preflight_source_survives_prefetch_without_mutating_input(self):
        source_batch = {"reward": np.ones((16, 65), np.float32)}
        source = runner.preflight_batches(source_batch)
        # Upstream prefetch can ask for additional batches beyond the two that
        # the learner consumes. Exhaustion in its worker terminates the process.
        batches = [next(source) for _ in range(100)]
        self.assertEqual(len({id(batch) for batch in batches}), 100)
        batches[0]["seed"] = 7
        self.assertNotIn("seed", source_batch)
        self.assertNotIn("seed", batches[1])
        self.assertIs(batches[-1]["reward"], source_batch["reward"])

    def test_reward_and_action_timing_positive_control(self):
        data = episode()
        result = verify.verify_episode(data)
        self.assertEqual(result["return"], 300)
        self.assertEqual(result["control"]["mse"], 0)
        self.assertEqual(result["control"]["positive"]["count"], 200)
        data["previous_action"][5] = 1
        with self.assertRaisesRegex(ValueError, "indexing"):
            verify.verify_episode(data)

    def test_missing_reset_end_and_bad_reward_rejected(self):
        for key, index, value in (
            ("is_first", 10, True),
            ("is_last", 10, True),
            ("reward", 0, 1),
            ("reward", 2, 3),
        ):
            data = episode()
            data[key][index] = value
            with self.assertRaises(ValueError):
                verify.verify_episode(data)

    def test_zero_reward_is_not_reported_as_positive_success(self):
        data = episode()
        data["reward"][:] = 0
        result = verify.verify_episode(data)
        self.assertEqual(result["control"]["positive_count"], 0)
        self.assertIsNone(result["control"]["positive"]["mse"])
        self.assertGreater(result["control"]["mse"], 0)

    def test_budget_reservation_precedes_call_and_failure_never_refunds(self):
        obj = object.__new__(runner.BudgetedReacher)
        obj.native_steps = 0
        obj.kind = "collect"
        events = []
        obj.budget = types.SimpleNamespace(
            reserve=lambda n, k: events.append(("charge", n, k)),
            record_reset=lambda: events.append(("reset",)),
        )

        def step(self, action):
            events.append(("step",))
            raise RuntimeError("simulated failure")

        with mock.patch.object(runner.base.ProprioReacher, "step", step):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                obj.step(dict(reset=False, action=np.zeros(2)))
        self.assertEqual(events, [("charge", 2, "collect"), ("step",)])

    def test_reset_calls_cost_zero_native_steps(self):
        obj = object.__new__(runner.BudgetedReacher)
        obj.native_steps = 0
        obj.kind = "evaluate"
        budget = mock.Mock()
        obj.budget = budget
        with mock.patch.object(
            runner.base.ProprioReacher, "step", return_value={"is_first": True}
        ):
            self.assertTrue(obj.step(dict(reset=True))["is_first"])
        budget.record_reset.assert_called_once()
        budget.reserve.assert_not_called()

    def test_native_repeat_mismatch_fails_closed(self):
        obj = object.__new__(runner.BudgetedReacher)
        obj.native_steps = 0
        obj.kind = "collect"
        obj.budget = mock.Mock()
        with mock.patch.object(runner.base.ProprioReacher, "step", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "unexpected native"):
                obj.step(dict(reset=False))
        obj.budget.reserve.assert_called_once_with(2, "collect")

    def test_atomic_pickle_no_overwrite_and_chunks(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "checkpoint.pkl"
            runner.save_pickle(path, {"a": np.ones(2)})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.save_pickle(path, {"a": np.zeros(2)})
            self.assertEqual(path.read_bytes(), original)
            writer = runner.RawRows(Path(td) / "rows")
            row = {"reward": np.array(1.0)}
            writer.add(row)
            row["reward"][...] = 2
            writer.flush()
            with np.load(writer.paths[0]) as f:
                self.assertEqual(float(f["reward"][0]), 1.0)

    def test_actual_action_is_clipped_and_recurrent(self):
        import jax

        carry = (
            {},
            {"deter": [jax.device_put(np.ones(2)), jax.device_put(np.ones(2))]},
            {},
            {
                "action": [
                    jax.device_put(np.zeros(2, np.float32)),
                    jax.device_put(np.zeros(2, np.float32)),
                ]
            },
        )
        changed, action = runner.actual_actions(carry, [[3, -2], [0.3, 0.2]])
        np.testing.assert_array_equal(
            action, [[1, -1], [np.float32(0.3), np.float32(0.2)]]
        )
        np.testing.assert_array_equal(jax.device_get(changed[-1]["action"]), action)
        np.testing.assert_array_equal(jax.device_get(carry[-1]["action"]), 0)

    def test_full_paired_schedule_fraction(self):
        for seed in protocol.SEEDS:
            C = [protocol.collection_policy("C", seed, i) for i in range(2500)]
            D = [protocol.collection_policy("D", seed, i) for i in range(2500)]
            self.assertEqual(C.count("random"), 500)
            self.assertEqual(D.count("explore"), 500)
            self.assertEqual([v == "random" for v in C], [v == "explore" for v in D])

    def test_probe_metric_units_and_negative_coverage(self):
        data = dict(
            truth_reward=np.zeros((20, 15)),
            truth_observation=np.zeros((20, 15, 6)),
            episode=np.repeat(np.arange(5), 4),
            anchor=np.tile([100, 200, 300, 400], 5),
            posterior_control=np.ones((20, 15)),
            posterior_original=np.ones((20, 15)),
            predicted_control=np.ones((20, 8, 15)),
            predicted_original=np.ones((20, 8, 15)),
            predicted_observation=np.ones((20, 8, 15, 6)),
            ensemble_means=np.zeros((5, 20, 15, 6)),
            observation_mean=np.zeros(6),
            observation_std=np.ones(6),
        )
        report = verify.probe_metrics(data, 2)
        self.assertEqual(
            report["horizons"]["15"]["predicted_control"]["cumulative_mse_raw"], 225
        )
        self.assertEqual(
            report["horizons"]["15"]["predicted_control"]["cumulative_mse_normalized"],
            56.25,
        )
        self.assertIn("inconclusive", report["reward_sensitive_conclusion"])
        self.assertIsNone(report["ensemble"]["error_disagreement_correlation"])

    def test_preflight_retained_batch_does_not_claim_historical_carry(self):
        import elements

        obs, act = runner.spaces(elements)
        spaces = {
            **obs,
            **act,
            "episode_id": elements.Space(np.int32),
            "stepid": elements.Space(np.uint8, (20,)),
            "consec": elements.Space(np.int32),
            "dyn/deter": elements.Space(np.float32, (8,)),
        }
        agent = types.SimpleNamespace(
            spaces=spaces, config=types.SimpleNamespace(replay_context=1)
        )
        with tempfile.TemporaryDirectory() as td:
            raw = {
                k: np.zeros((16, 65, *s.shape), s.dtype)
                for k, s in {**obs, **act}.items()
            }
            np.savez(Path(td) / "replay_sample_200000_48985.npz", **raw)
            batch, _ = runner.retained_batch(td, agent)
            np.testing.assert_array_equal(batch["dyn/deter"], 0)
            self.assertEqual(batch["episode_id"].shape, (16, 65))

    def test_full_probe_shape_compilation_and_exact_replay_without_simulator(self):
        import elements
        import jax.numpy as jnp
        import ninjax as nj
        from test_reward_exploration_agent import fixture
        from dreamer_imf_compare.reward_exploration_agent import install
        from dreamer_imf_compare.reward_exploration_probe import retained_probe
        from dreamer_imf_compare.conditional_schedule import CONTROLS

        cfg, _, act, data = fixture()
        obs, _ = runner.spaces(elements)
        vector = data.pop("vector")
        for i, key in enumerate(runner.OBS_KEYS):
            data[key] = vector[:, :, i * 2 : (i + 1) * 2]
        cls = install("D")
        model = object.__new__(cls)
        model.__init__(obs, act, cfg)
        state = nj.init(model.train)({}, model.init_train(2), data, seed=0)
        for key, value in CONTROLS.items():
            state[f"staged_{key}/value"] = jnp.asarray(value)
        rng = np.random.default_rng(871)
        raw = {
            k: rng.normal(size=(5, 501, 2)).astype(np.float32) for k in runner.OBS_KEYS
        }
        raw.update(
            reward=rng.integers(0, 3, (5, 501)).astype(np.float32),
            is_first=np.broadcast_to(np.arange(501) == 0, (5, 501)),
            is_last=np.broadcast_to(np.arange(501) == 500, (5, 501)),
            is_terminal=np.zeros((5, 501), bool),
            action=rng.uniform(-1, 1, (5, 501, 2)).astype(np.float32),
        )
        raw["previous_action"] = np.concatenate(
            [np.zeros((5, 1, 2), np.float32), raw["action"][:, :-1]], 1
        )
        proxy = types.SimpleNamespace(model=model, params=state, obs_space=obs)
        with tempfile.TemporaryDirectory() as td:
            path = retained_probe(proxy, td, 40000, 701, data_override=raw)
            retained_probe(proxy, td, 40000, 701, verify=True, data_override=raw)
            result = verify.probe_metrics(verify.load_arrays(path), 2)
            self.assertEqual(result["particles"], 8)
            self.assertEqual(set(result["horizons"]), {"1", "3", "5", "10", "15"})


if __name__ == "__main__":
    unittest.main()
