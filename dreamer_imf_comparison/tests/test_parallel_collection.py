"""Deterministic injected simulator tests; no dm_control or model training."""

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from dreamer_imf_compare import parallel_collection as pc


class FakeEnvironment:
    action_low = np.asarray([-1.0, -0.5], np.float32)
    action_high = np.asarray([1.0, 0.5], np.float32)
    action_repeat = 2

    def __init__(self, seed):
        self.seed = seed
        self.native_steps = 0
        self.closed = False
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2, np.float64)
        self.clock = 0

    def observation(self):
        # Deliberately different insertion order from model.obs_keys, extra
        # integer data, and float64 precision that must survive raw histories.
        return dict(
            aux=np.asarray(self.clock, np.float64),
            position=self.position.copy(),
            status=np.asarray(self.clock, np.int32),
        )

    def reset(self):
        self.position = self.rng.normal(size=2) * 0.01
        self.clock = 0
        return self.observation()

    def step(self, action):
        reward = 0.0
        for _ in range(2):
            self.native_steps += 1
            self.clock += 1
            self.position += (
                np.asarray(action, np.float64) * 0.01 + self.rng.normal(size=2) * 0.0001
            )
            reward += float(self.position[0] - self.position[1] + 0.123456789012345)
        return SimpleNamespace(
            observation=self.observation(),
            reward=reward,
            is_last=self.clock == 1000,
            native_steps=2,
        )

    def snapshot(self):
        return (
            self.position.copy(),
            self.clock,
            copy.deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, snapshot):
        self.position, self.clock, state = copy.deepcopy(snapshot)
        self.rng.bit_generator.state = state

    def close(self):
        self.closed = True


class FakeModel:
    obs_keys = ("position", "aux")

    def initial(self):
        return dict(history=0.0, count=0)

    def observe(self, carry, obs, previous_action, seed):
        # Mutate carry in place: collector must deeply isolate branch carries.
        carry["history"] += (
            float(np.asarray(previous_action).sum()) + (seed % 31) * 0.0001
        )
        carry["count"] += 1
        feature = np.asarray(
            [*obs["position"], obs["aux"], carry["history"], carry["count"]], np.float32
        )
        return carry, feature

    def action(self, feature, seed):
        return np.asarray(
            [np.tanh(feature[0]) * 0.4, np.sin(seed % 31) * 0.2], np.float32
        )

    def frozen_digest(self):
        return "fake-frozen-parameters-v1"


class BudgetTests(unittest.TestCase):
    def test_simultaneous_reservations_cannot_race_past_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            with pc.BudgetLedger(Path(directory) / "budget", limit=20) as ledger:

                def attempt(index):
                    try:
                        return ledger.reserve(2, f"thread {index}")
                    except pc.BudgetExceeded:
                        return None

                with ThreadPoolExecutor(max_workers=8) as pool:
                    accepted = [
                        item
                        for item in pool.map(attempt, range(40))
                        if item is not None
                    ]
                self.assertEqual(len(accepted), 10)
                self.assertEqual(len(set(accepted)), 10)
                self.assertEqual(ledger.charged, 20)

    def test_reservations_persist_without_refunds_and_cannot_exceed_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            with pc.BudgetLedger(path, limit=10) as ledger:
                first = ledger.reserve(2, "failed step")
                ledger.finish(first, 1, outcome="failed")
                pending = ledger.reserve(2, "interrupted step")
                self.assertEqual((ledger.charged, ledger.actual), (4, 1))
                old = path.read_bytes()
                with self.assertRaises(BlockingIOError):
                    pc.BudgetLedger(path, limit=10)
            with pc.BudgetLedger(path, limit=10) as ledger:
                self.assertEqual(ledger.pending, {pending: 2})
                self.assertEqual((ledger.charged, ledger.actual), (4, 1))
                ledger.reserve(6, "remaining attempt")
                self.assertEqual(ledger.remaining, 0)
                with self.assertRaises(pc.BudgetExceeded):
                    ledger.reserve(1, "must not execute")
            self.assertTrue(path.read_bytes().startswith(old))

    def test_preflight_cap_and_limit_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            with pc.BudgetLedger(Path(directory) / "budget") as ledger:
                ledger.reserve(4800, "all preflight capacity", category="preflight")
                with self.assertRaises(pc.BudgetExceeded):
                    ledger.reserve(1, "extra preflight", category="preflight")
                ledger.reserve(95200, "main capacity")
                with self.assertRaises(pc.BudgetExceeded):
                    ledger.reserve(1, "extra")
            with self.assertRaises(ValueError):
                pc.BudgetLedger(Path(directory) / "bad", limit=100001)
            with self.assertRaises(ValueError):
                pc.BudgetLedger(Path(directory) / "bool", limit=True)

    def test_corrupt_partial_or_different_limit_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget"
            with pc.BudgetLedger(path, limit=20) as ledger:
                ledger.reserve(2, "positive control")
            with self.assertRaises(pc.CollectionIntegrityError):
                pc.BudgetLedger(path, limit=21)
            with path.open("ab") as stream:
                stream.write(b'{"partial":')
            old = path.read_bytes()
            with self.assertRaises(pc.CollectionIntegrityError):
                pc.BudgetLedger(path, limit=20)
            self.assertEqual(old, path.read_bytes())

    def test_finish_cannot_release_budget_or_claim_unreserved_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            with pc.BudgetLedger(Path(directory) / "budget") as ledger:
                item = ledger.reserve(2, "attempt")
                for actual in (-1, 3, True):
                    with self.assertRaises((ValueError, pc.CollectionIntegrityError)):
                        ledger.finish(item, actual)
                ledger.finish(item, 0, outcome="failed")
                self.assertEqual(ledger.charged, 2)
                with self.assertRaises(pc.CollectionIntegrityError):
                    ledger.finish(item, 0)


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.environments = []

        def factory(seed):
            env = FakeEnvironment(seed)
            cls.environments.append(env)
            return env

        with pc.BudgetLedger(cls.root / "budget.jsonl") as ledger:
            cls.result = pc.collect(
                cls.root / "artifacts", FakeModel(), factory, budget=ledger
            )
            cls.charged, cls.actual = ledger.charged, ledger.actual
        with np.load(cls.result.dataset, allow_pickle=False) as archive:
            cls.arrays = {name: archive[name] for name in archive.files}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_independently_measured_counts_and_budget(self):
        self.assertEqual(len(self.environments), 64)
        self.assertTrue(all(env.closed for env in self.environments))
        self.assertEqual(sum(env.native_steps for env in self.environments), 95200)
        self.assertEqual((self.charged, self.actual), (95200, 95200))
        with np.load(
            self.result.attempt / "duplicates.npz", allow_pickle=False
        ) as duplicate:
            self.assertEqual(duplicate["episode"].shape, (16,))
        main_steps = sum(
            2 * len(np.load(path)["actions"])
            for path in self.result.attempt.glob("episode-*.npz")
        )
        branch_steps = sum(
            2 * len(np.load(path)["actions"])
            for path in self.result.attempt.glob("branch-*.npz")
            if "duplicate" not in path.name
        )
        self.assertEqual(main_steps + branch_steps, 94720)

    def test_dataset_shapes_types_and_finite_values(self):
        expected = dict(
            episode=(1024,),
            split=(1024,),
            mode=(1024,),
            anchor=(1024,),
            plan=(1024,),
            start=(1024, 5),
            actions=(1024, 15, 2),
            targets=(1024, 15, 5),
            observations=(1024, 15, 3),
            rewards=(1024, 15),
            initial_observation=(1024, 3),
        )
        self.assertEqual(set(self.arrays), set(expected))
        for name, shape in expected.items():
            self.assertEqual(self.arrays[name].shape, shape)
            self.assertFalse(self.arrays[name].dtype.hasobject)
            self.assertTrue(np.isfinite(self.arrays[name]).all())
        self.assertEqual(self.arrays["targets"].dtype, np.float32)
        self.assertEqual(self.arrays["rewards"].dtype, np.float64)

    def test_episode_level_split_modes_and_anchor_cartesian_product(self):
        data = self.arrays
        for split, episodes in enumerate((40, 12, 12)):
            selected = data["split"] == split
            self.assertEqual(len(np.unique(data["episode"][selected])), episodes)
            for mode in (0, 1):
                self.assertEqual(
                    len(np.unique(data["episode"][selected & (data["mode"] == mode)])),
                    episodes // 2,
                )
        for episode in range(64):
            mask = data["episode"] == episode
            self.assertEqual(len(np.unique(data["split"][mask])), 1)
            self.assertEqual(
                set(zip(data["anchor"][mask], data["plan"][mask])),
                {(a, p) for a in (100, 200, 300, 400) for p in range(4)},
            )

    def test_exogenous_plans_prefixes_and_held_actions(self):
        data = self.arrays
        for episode in range(64):
            with np.load(
                self.result.attempt / f"episode-{episode:03d}.npz", allow_pickle=False
            ) as base:
                for anchor in (100, 200, 300, 400):
                    selected = (data["episode"] == episode) & (data["anchor"] == anchor)
                    plans = data["actions"][selected]
                    np.testing.assert_array_equal(plans[0], 0)
                    held = FakeModel().action(
                        base["features"][anchor], pc._seed(31, episode, anchor)
                    )
                    np.testing.assert_array_equal(
                        plans[1], np.broadcast_to(held, (15, 2))
                    )
                    np.testing.assert_array_equal(plans[2, :5], plans[3, :5])
                    self.assertFalse(np.array_equal(plans[2, 5:], plans[3, 5:]))
                    np.testing.assert_array_equal(
                        data["targets"][selected][2, :5],
                        data["targets"][selected][3, :5],
                    )
                    np.testing.assert_array_equal(
                        data["observations"][selected][2, :5],
                        data["observations"][selected][3, :5],
                    )

    def test_mode_one_is_independent_twenty_percent_replacement_not_uniform_policy(
        self,
    ):
        replacements, decisions = 0, 0
        for episode in range(64):
            with np.load(
                self.result.attempt / f"episode-{episode:03d}.npz", allow_pickle=False
            ) as base:
                flags = base["uniform_replaced"]
                for decision, seed in enumerate(base["replacement_seeds"]):
                    rng = np.random.default_rng(int(seed))
                    expected = episode % 2 == 1 and rng.random() < 0.2
                    self.assertEqual(bool(flags[decision]), expected)
                    if expected:
                        action = rng.uniform(
                            FakeEnvironment.action_low, FakeEnvironment.action_high
                        ).astype(np.float32)
                        np.testing.assert_array_equal(base["actions"][decision], action)
                np.testing.assert_array_equal(
                    base["actions"][~flags], base["policy_actions"][~flags]
                )
                if episode % 2:
                    replacements += int(flags.sum())
                    decisions += len(flags)
                else:
                    self.assertFalse(flags.any())
        self.assertGreater(replacements / decisions, 0.18)
        self.assertLess(replacements / decisions, 0.22)

    def test_reward_and_posterior_alignment_raw_precision_and_reconstruction(self):
        data = self.arrays
        for episode, anchor, plan in ((0, 100, 0), (5, 200, 0), (63, 400, 3)):
            index = np.flatnonzero(
                (data["episode"] == episode)
                & (data["anchor"] == anchor)
                & (data["plan"] == plan)
            )[0]
            base_path = self.result.attempt / f"episode-{episode:03d}.npz"
            branch_path = (
                self.result.attempt / f"branch-{episode:03d}-{anchor:03d}-{plan}.npz"
            )
            base, raw = pc.load_raw_history(base_path)
            branch, braw = pc.load_raw_history(branch_path)
            self.assertEqual(raw[0]["position"].dtype, np.float64)
            self.assertEqual(raw[0]["status"].dtype, np.int32)
            self.assertEqual(len(raw), 501)
            self.assertEqual(len(braw), 16)
            np.testing.assert_array_equal(
                data["start"][index], base["features"][anchor]
            )
            np.testing.assert_array_equal(
                data["initial_observation"][index], base["observations"][anchor]
            )
            np.testing.assert_array_equal(
                data["targets"][index], branch["features"][1:]
            )
            np.testing.assert_array_equal(data["rewards"][index], branch["rewards"])
            np.testing.assert_array_equal(
                branch["observations"][:, 2], 2 * np.arange(anchor, anchor + 16)
            )
            start, future = pc.reconstruct_features(
                FakeModel(), base_path, anchor=anchor, branch_path=branch_path
            )
            np.testing.assert_array_equal(start, data["start"][index])
            np.testing.assert_array_equal(future, data["targets"][index])

    def test_branch_restore_leaves_base_trajectory_unchanged(self):
        for episode in (0, 1, 63):
            base, raw = pc.load_raw_history(
                self.result.attempt / f"episode-{episode:03d}.npz"
            )
            env = FakeEnvironment(pc.BASE_SEED + episode)
            obs = env.reset()
            np.testing.assert_array_equal(obs["position"], raw[0]["position"])
            for t, action in enumerate(base["actions"]):
                step = env.step(action)
                np.testing.assert_array_equal(
                    step.observation["position"], raw[t + 1]["position"]
                )
                self.assertEqual(step.reward, base["rewards"][t])

    def test_manifest_hashes_and_duplicate_replay_evidence(self):
        import hashlib

        manifest = json.loads(self.result.manifest.read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["episode_counts"], [40, 12, 12])
        self.assertEqual(len(manifest["duplicate_checks"]), 16)
        self.assertTrue(all(check["exact"] for check in manifest["duplicate_checks"]))
        self.assertEqual(manifest["snapshot_count"], 256)
        self.assertEqual(
            set(manifest["snapshot_artifacts"]),
            {path.name for path in self.result.attempt.glob("anchor-*.pkl")},
        )
        self.assertEqual(len(manifest["snapshot_artifacts"]), 256)
        self.assertTrue(
            set(manifest["snapshot_artifacts"]).issubset(manifest["sha256"])
        )
        for name, expected in manifest["sha256"].items():
            self.assertEqual(
                hashlib.sha256((self.result.attempt / name).read_bytes()).hexdigest(),
                expected,
            )

    def test_all_anchor_forensics_preserve_state_belief_rewards_and_plans(self):
        data = self.arrays
        paths = sorted(self.result.attempt.glob("anchor-*.pkl"))
        self.assertEqual(len(paths), 256)
        # These pickle files were written by this test process, not untrusted
        # inputs. No portable restore is attempted; adapter checks stay intact.
        for episode in range(64):
            base, raw = pc.load_raw_history(
                self.result.attempt / f"episode-{episode:03d}.npz"
            )
            for anchor in pc.ANCHORS:
                with (
                    self.result.attempt / f"anchor-{episode:03d}-{anchor:03d}.pkl"
                ).open("rb") as stream:
                    forensic = pickle.load(stream)
                self.assertEqual(forensic["kind"], "process-local-forensic-anchor")
                self.assertFalse(forensic["portable_restore"])
                self.assertTrue(forensic["trusted_pickle_only"])
                self.assertEqual(
                    (forensic["episode"], forensic["anchor"]), (episode, anchor)
                )
                self.assertEqual(forensic["episode_seed"], pc.BASE_SEED + episode)
                self.assertEqual(forensic["model_digest"], FakeModel().frozen_digest())
                self.assertEqual(forensic["belief"]["count"], anchor + 1)
                self.assertEqual(
                    np.float32(forensic["belief"]["history"]),
                    base["features"][anchor, 3],
                )
                np.testing.assert_array_equal(
                    forensic["simulator_snapshot"][0], raw[anchor]["position"]
                )
                self.assertEqual(forensic["simulator_snapshot"][1], 2 * anchor)
                self.assertGreaterEqual(forensic["native_steps_at_capture"], 2 * anchor)
                np.testing.assert_array_equal(
                    forensic["start"], base["features"][anchor]
                )
                np.testing.assert_array_equal(
                    forensic["observation"], base["observations"][anchor]
                )
                np.testing.assert_array_equal(
                    forensic["previous_action"], base["actions"][anchor - 1]
                )
                self.assertEqual(forensic["reward"], base["rewards"][anchor - 1])
                self.assertEqual(
                    forensic["observe_seed"], int(base["observe_seeds"][anchor])
                )
                for key, value in raw[anchor].items():
                    np.testing.assert_array_equal(
                        forensic["raw_observation"][key], value
                    )
                selected = (data["episode"] == episode) & (data["anchor"] == anchor)
                np.testing.assert_array_equal(
                    forensic["action_plans"], data["actions"][selected]
                )
                np.testing.assert_array_equal(
                    forensic["branch_observe_seeds"],
                    [pc._seed(40, episode, anchor, t) for t in range(15)],
                )
        # Existing independently measured counter assertion also proves that
        # saving every snapshot did not add simulator transitions.
        self.assertEqual(sum(env.native_steps for env in self.environments), 95200)


class FailureTests(unittest.TestCase):
    def test_preflight_failed_attempt_budget_restart_and_no_overwrite(self):
        class Broken(FakeEnvironment):
            def step(self, action):
                self.native_steps += 1
                raise RuntimeError("injected native failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with pc.BudgetLedger(root / "budget") as ledger:
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    pc.collect(
                        root / "out", FakeModel(), Broken, budget=ledger, preflight=True
                    )
                self.assertEqual((ledger.charged, ledger.actual), (2, 1))
            failed = next((root / "out").iterdir())
            evidence = {p.name: p.read_bytes() for p in failed.iterdir()}
            with pc.BudgetLedger(root / "budget") as ledger:
                result = pc.collect(
                    root / "out",
                    FakeModel(),
                    FakeEnvironment,
                    budget=ledger,
                    preflight=True,
                )
                self.assertEqual(
                    (result.charged_steps, result.actual_steps), (1510, 1510)
                )
                self.assertEqual((ledger.charged, ledger.actual), (1512, 1511))
                self.assertEqual(ledger.preflight_charged, 1512)
            manifest = json.loads(result.manifest.read_text())
            self.assertEqual(manifest["snapshot_count"], 4)
            self.assertEqual(len(list(result.attempt.glob("anchor-*.pkl"))), 4)
            self.assertTrue(
                set(manifest["snapshot_artifacts"]).issubset(manifest["sha256"])
            )
            self.assertNotEqual(result.attempt, failed)
            self.assertEqual(
                evidence, {p.name: p.read_bytes() for p in failed.iterdir()}
            )
            self.assertFalse((failed / "manifest.json").exists())

    def test_budget_refusal_happens_before_environment_instantiation(self):
        with tempfile.TemporaryDirectory() as directory:
            with pc.BudgetLedger(Path(directory) / "budget", limit=100) as ledger:
                factory = mock.Mock()
                with self.assertRaises(pc.BudgetExceeded):
                    pc.collect(
                        Path(directory) / "out", FakeModel(), factory, budget=ledger
                    )
                factory.assert_not_called()
                self.assertEqual(ledger.charged, 0)

    def test_reservation_is_durable_before_any_native_step(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget"

            class CheckBeforeStep(FakeEnvironment):
                def step(self, action):
                    last = json.loads(path.read_text().splitlines()[-1])
                    if last["kind"] != "reserve" or last["count"] != 2:
                        raise AssertionError(
                            "native step had no prior durable reservation"
                        )
                    return super().step(action)

            with pc.BudgetLedger(path) as ledger:
                pc.collect(
                    Path(directory) / "out",
                    FakeModel(),
                    CheckBeforeStep,
                    budget=ledger,
                    preflight=True,
                )

    def test_restore_counter_rewind_and_wrong_physics_are_rejected(self):
        class Rewinds(FakeEnvironment):
            def snapshot(self):
                return super().snapshot(), self.native_steps

            def restore(self, snapshot):
                state, self.native_steps = snapshot
                super().restore(state)

        class BadPhysics(FakeEnvironment):
            def restore(self, snapshot):
                super().restore(snapshot)
                self.position[0] += self.native_steps * 1e-5

        class RawOnlyMismatch(FakeEnvironment):
            def observation(self):
                obs = super().observation()
                obs["status"] = np.asarray(self.native_steps * 1e-100, np.float64)
                return obs

        for factory in (Rewinds, BadPhysics, RawOnlyMismatch):
            with self.subTest(
                factory=factory.__name__
            ), tempfile.TemporaryDirectory() as directory:
                with pc.BudgetLedger(Path(directory) / "budget") as ledger:
                    with self.assertRaises(pc.CollectionIntegrityError):
                        pc.collect(
                            Path(directory) / "out",
                            FakeModel(),
                            factory,
                            budget=ledger,
                            preflight=True,
                        )

    def test_input_mutation_cannot_modify_raw_history_or_future_action_plans(self):
        class MutatesInputs(FakeModel):
            action_calls = 0

            def observe(self, carry, obs, previous_action, seed):
                result = super().observe(carry, obs, previous_action, seed)
                obs["position"][:] = 123456
                previous_action[:] = 999
                return result

            def action(self, feature, seed):
                self.action_calls += 1
                return super().action(feature, seed)

        with tempfile.TemporaryDirectory() as directory:
            model = MutatesInputs()
            with pc.BudgetLedger(Path(directory) / "budget") as ledger:
                result = pc.collect(
                    Path(directory) / "out",
                    model,
                    FakeEnvironment,
                    budget=ledger,
                    preflight=True,
                )
            self.assertEqual(
                model.action_calls, 504
            )  # 500 base decisions + 4 anchor-held draws.
            base, raw = pc.load_raw_history(result.attempt / "episode-000.npz")
            self.assertLess(np.max(np.abs(base["actions"])), 1)
            self.assertLess(np.max(np.abs(raw[-1]["position"])), 10)
            branch, _ = pc.load_raw_history(result.attempt / "branch-000-100-0.npz")
            np.testing.assert_array_equal(branch["actions"], 0)

    def test_reservation_write_failure_prevents_native_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnvironment(pc.BASE_SEED)
            with pc.BudgetLedger(Path(directory) / "budget") as ledger:
                with mock.patch.object(
                    ledger, "_append", side_effect=OSError("injected fsync error")
                ):
                    with self.assertRaisesRegex(OSError, "injected fsync"):
                        pc.collect(
                            Path(directory) / "out",
                            FakeModel(),
                            lambda seed: env,
                            budget=ledger,
                            preflight=True,
                        )
                self.assertEqual(ledger.charged, 0)
            self.assertEqual(env.native_steps, 0)
            self.assertTrue(env.closed)

    def test_atomic_exclusive_artifacts_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.npz"
            pc._write_npz(path, {"x": np.asarray([1, 2])})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                pc._write_npz(path, {"x": np.asarray([3, 4])})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            for value in (np.asarray([np.nan]), np.asarray([{}], object)):
                with self.assertRaises(pc.CollectionIntegrityError):
                    pc._write_npz(Path(directory) / "bad.npz", {"bad": value})

    def test_invalid_action_and_early_terminal_stop_without_retry(self):
        class BadAction(FakeModel):
            def action(self, feature, seed):
                return np.asarray([2, 0], np.float32)

        class EarlyTerminal(FakeEnvironment):
            def step(self, action):
                step = super().step(action)
                step.is_last = True
                return step

        for model, factory, charged in (
            (BadAction(), FakeEnvironment, 0),
            (FakeModel(), EarlyTerminal, 2),
        ):
            with self.subTest(
                charged=charged
            ), tempfile.TemporaryDirectory() as directory:
                with pc.BudgetLedger(Path(directory) / "budget") as ledger:
                    with self.assertRaises(pc.CollectionIntegrityError):
                        pc.collect(
                            Path(directory) / "out",
                            model,
                            factory,
                            budget=ledger,
                            preflight=True,
                        )
                    self.assertEqual(ledger.charged, charged)

    def test_teacher_parameter_change_invalidates_dataset(self):
        class Mutates(FakeModel):
            calls = 0

            def frozen_digest(self):
                self.calls += 1
                return str(self.calls)

        with tempfile.TemporaryDirectory() as directory:
            with pc.BudgetLedger(Path(directory) / "budget") as ledger:
                with self.assertRaisesRegex(
                    pc.CollectionIntegrityError, "parameters changed"
                ):
                    pc.collect(
                        Path(directory) / "out",
                        Mutates(),
                        FakeEnvironment,
                        budget=ledger,
                        preflight=True,
                    )
            attempt = next((Path(directory) / "out").iterdir())
            self.assertFalse((attempt / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
