"""Root adapter/runner integration with fake physics and a real tiny JAX fit.

No checkpoint, dm_control installation, GPU, production directory, or real
native-step budget is touched. Test artifacts live in unique temporary roots.
"""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from dreamer_imf_compare.parallel_frozen import (
    FrozenModel,
    ReacherAdapter,
    parameter_digest,
)
from dreamer_imf_compare import parallel_runner as runner


class FrozenDigestTests(unittest.TestCase):
    def make_model(self):
        model = FrozenModel.__new__(FrozenModel)
        model.host_params = {
            "dyn/kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
            "pol/bias": np.asarray([0.1, 0.2], np.float32),
            "opt/state": np.asarray([97], np.int32),
        }
        model.params = {
            key: jax.device_put(value.copy())
            for key, value in model.host_params.items()
            if not key.startswith("opt/")
        }
        model.live_keys = frozenset(model.params)
        return model

    def test_digest_reads_live_arrays_instead_of_preserved_host_copy(self):
        model = self.make_model()
        before = model.frozen_digest()
        self.assertEqual(before, parameter_digest(model.host_params))
        host_before = {key: value.copy() for key, value in model.host_params.items()}
        model.params["dyn/kernel"] = model.params["dyn/kernel"].at[0, 0].set(42)
        self.assertNotEqual(model.frozen_digest(), before)
        for key, value in host_before.items():
            np.testing.assert_array_equal(model.host_params[key], value)

    def test_live_key_addition_and_removal_fail_closed(self):
        for change in ("add", "remove"):
            with self.subTest(change=change):
                model = self.make_model()
                if change == "add":
                    model.params["unexpected/key"] = jnp.zeros(1)
                else:
                    model.params.pop("pol/bias")
                with self.assertRaisesRegex(ValueError, "keys changed"):
                    model.frozen_digest()


class _TimeStep:
    def __init__(self, observation, reward=None, first=False, last=False, discount=1.0):
        self.observation = observation
        self.reward = reward
        self.discount = discount
        self._first, self._last = first, last

    def first(self):
        return self._first

    def last(self):
        return self._last


class _Physics:
    def __init__(self):
        self.state = np.zeros(2)
        self.data = SimpleNamespace(time=0.0)
        for name, shape in (
            ("qacc_warmstart", (2,)),
            ("ctrl", (2,)),
            ("qfrc_applied", (2,)),
            ("xfrc_applied", (2, 6)),
            ("mocap_pos", (0, 3)),
            ("mocap_quat", (0, 4)),
            ("userdata", (3,)),
        ):
            setattr(self.data, name, np.zeros(shape))

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state[:] = state

    def reset_context(self):
        return nullcontext()


class _DMEnvironment:
    def __init__(self, seed=7):
        self.physics = _Physics()
        self.task = SimpleNamespace(random=np.random.RandomState(seed))
        self._step_count = 0
        self._reset_next_step = False
        self.fail_before = None
        self.fail_after = None
        self.calls = 0
        self.closed = False

    def action_spec(self):
        return SimpleNamespace(shape=(2,), minimum=-1.0, maximum=1.0)

    def observation(self):
        return dict(
            position=self.physics.state.copy(),
            to_target=1 - self.physics.state,
            velocity=np.full(2, self.physics.data.time),
        )

    def reset(self):
        self._step_count = 0
        self._reset_next_step = False
        self.physics.state[:] = self.task.random.normal(size=2)
        self.physics.data.time = 0.0
        return _TimeStep(self.observation(), first=True)

    def step(self, action):
        self.calls += 1
        if self.calls == self.fail_before:
            raise RuntimeError("before physics")
        self.physics.state += (
            np.asarray(action) + self.task.random.normal(size=2) * 0.01
        )
        self.physics.data.time += 0.02
        self._step_count += 1
        for name in (
            "qacc_warmstart",
            "ctrl",
            "qfrc_applied",
            "xfrc_applied",
            "userdata",
        ):
            getattr(self.physics.data, name)[:] += 0.125
        if self.calls == self.fail_after:
            raise RuntimeError("after physics")
        return _TimeStep(self.observation(), reward=float(self.physics.state.sum()))

    def close(self):
        self.closed = True


class ReacherAdapterTests(unittest.TestCase):
    def adapter(self, dm=None):
        dm = _DMEnvironment() if dm is None else dm
        load = mock.Mock(return_value=dm)
        module = SimpleNamespace(suite=SimpleNamespace(load=load))
        with mock.patch.dict("sys.modules", {"dm_control": module}):
            adapter = ReacherAdapter(431)
        load.assert_called_once_with(
            "reacher", "hard", task_kwargs={"random": 431, "time_limit": 20.0}
        )
        return adapter

    def test_ar2_reward_summing_and_observation_flags(self):
        dm = _DMEnvironment()
        adapter = self.adapter(dm)
        reset = adapter.reset()
        self.assertTrue(reset["is_first"])
        self.assertFalse(reset["is_terminal"])
        np.testing.assert_array_equal(adapter.action_low, [-1, -1])
        reference = copy.deepcopy(dm)
        action = np.asarray([0.3, -0.2], np.float32)
        expected = reference.step(action).reward + reference.step(action).reward
        result = adapter.step(action)
        self.assertEqual(result.reward, expected)
        self.assertEqual(
            float(result.observation["reward"]), float(np.float32(expected))
        )
        self.assertEqual((result.native_steps, adapter.native_steps), (2, 2))
        self.assertFalse(result.observation["is_first"])
        adapter.close()
        self.assertTrue(dm.closed)

    def test_throw_after_physics_counts_started_interval_including_second_repeat(self):
        for before, after, expected in ((1, None, 0), (None, 1, 1), (None, 2, 2)):
            with self.subTest(before=before, after=after):
                dm = _DMEnvironment()
                dm.fail_before, dm.fail_after = before, after
                adapter = self.adapter(dm)
                adapter.reset()
                with self.assertRaisesRegex(RuntimeError, "physics"):
                    adapter.step(np.zeros(2, np.float32))
                self.assertEqual(adapter.native_steps, expected)
                self.assertEqual(dm._step_count, expected)
                self.assertAlmostEqual(dm.physics.data.time, expected * 0.02)

    def test_after_physics_failure_is_charged_by_collection_ledger(self):
        from dreamer_imf_compare import parallel_collection as collection

        class Policy:
            obs_keys = FrozenModel.obs_keys

            def initial(self):
                return 0

            def observe(self, carry, obs, previous_action, seed):
                return carry + 1, np.concatenate([obs[key] for key in self.obs_keys])

            def action(self, feature, seed):
                return np.zeros(2, np.float32)

            def frozen_digest(self):
                return "immutable-fake-policy"

        dm = _DMEnvironment()
        dm.fail_after = 1
        adapter = self.adapter(dm)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with collection.BudgetLedger(root / "budget.jsonl") as ledger:
                with self.assertRaisesRegex(RuntimeError, "after physics"):
                    collection.collect(
                        root / "artifacts",
                        Policy(),
                        lambda seed: adapter,
                        budget=ledger,
                        preflight=True,
                    )
                self.assertEqual((ledger.charged, ledger.actual), (2, 1))
                self.assertFalse(ledger.pending)
            failure = json.loads(
                next((root / "artifacts").glob("*/failure.json")).read_text()
            )
            self.assertEqual(
                (failure["charged_steps"], failure["actual_steps"]), (2, 1)
            )
        self.assertEqual(adapter.native_steps, 1)
        self.assertTrue(dm.closed)

    def test_snapshot_replays_physics_wrapper_rng_and_keeps_lifetime_counter(self):
        adapter = self.adapter()
        adapter.reset()
        snapshot = adapter.snapshot()
        action = np.asarray([0.125, -0.0625], np.float32)
        first = adapter.step(action)
        counter = adapter.native_steps
        adapter.dm._reset_next_step = True
        adapter.restore(snapshot)
        self.assertEqual(adapter.native_steps, counter)
        self.assertEqual(adapter.dm._step_count, snapshot["step_count"])
        self.assertEqual(adapter.dm._reset_next_step, snapshot["reset_next_step"])
        self.assertEqual(adapter.dm.physics.data.time, snapshot["time"])
        np.testing.assert_array_equal(adapter.dm.physics.get_state(), snapshot["state"])
        for name, value in snapshot["extras"].items():
            np.testing.assert_array_equal(getattr(adapter.dm.physics.data, name), value)
        second = adapter.step(action)
        self.assertEqual(first.reward, second.reward)
        for name, value in first.observation.items():
            np.testing.assert_array_equal(value, second.observation[name])
        self.assertEqual(adapter.native_steps, 4)

    def test_cross_instance_and_cross_reset_generation_snapshots_are_rejected(self):
        adapter = self.adapter()
        adapter.reset()
        snapshot = adapter.snapshot()
        other = self.adapter()
        other.reset()
        with self.assertRaisesRegex(
            ValueError, "another simulator or reset generation"
        ):
            other.restore(snapshot)
        adapter.reset()
        current = adapter.dm.physics.get_state()
        with self.assertRaisesRegex(ValueError, "reset generation"):
            adapter.restore(snapshot)
        np.testing.assert_array_equal(adapter.dm.physics.get_state(), current)
        self.assertEqual(adapter.native_steps, 0)


class _TinyTeacher:
    feature_dim = 4
    action_dim = 2

    def __init__(self):
        self.params = {"decoder/scale": np.asarray([1.0], np.float32)}

    def frozen_digest(self):
        return parameter_digest(self.params)

    def decode(self, features):
        features = jnp.asarray(features)
        return (
            features[..., :2] * float(self.params["decoder/scale"][0]),
            features[..., 2] * 0.1,
        )


def _tiny_data():
    rng = np.random.default_rng(704)
    starts = rng.normal(size=(8, 4)).astype(np.float32)
    actions = rng.uniform(-1, 1, (8, 15, 2)).astype(np.float32)
    targets = np.repeat(starts[:, None], 15, axis=1)
    targets[..., :2] += actions.cumsum(axis=1) * 0.05
    targets[..., 2] += np.arange(1, 16, dtype=np.float32)[None] * 0.02
    starts[:, 3] = 2.0
    targets[..., 3] = 2.0
    return dict(
        start=starts,
        actions=actions,
        targets=targets,
        observations=targets[..., :2].copy(),
        rewards=targets[..., 2] * 0.1,
        split=np.asarray([0, 0, 0, 0, 1, 1, 2, 2], np.int8),
        episode=np.arange(8, dtype=np.int32),
        anchor=np.full(8, 100, np.int32),
        plan=np.tile(np.arange(4, dtype=np.int8), 2),
        initial_observation=starts[:, :2].copy(),
    )


def _tiny_protocol():
    return dict(
        field=dict(horizon=15, width=8, heads=2, layers=1, ff_width=16),
        velocity_readout="affine",
        gradient_clip=1.0,
        learning_rate=3e-4,
        weight_decay=1e-4,
        updates=4,
        validation_period=2,
        batch_size=2,
    )


class RunnerNormalizationTests(unittest.TestCase):
    def test_all_normalization_statistics_exclude_validation_and_test(self):
        data = _tiny_data()
        expected = runner.normalized_data(data)
        poisoned = {name: value.copy() for name, value in data.items()}
        for name in ("start", "targets", "observations", "rewards"):
            poisoned[name][poisoned["split"] != 0] = np.nan
        actual = runner.normalized_data(poisoned)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        train = data["split"] == 0
        all_latents = np.concatenate(
            (data["start"][train, None], data["targets"][train]), axis=1
        )
        np.testing.assert_allclose(
            expected["mean"], all_latents.mean((0, 1)), atol=1e-6
        )
        np.testing.assert_allclose(
            expected["std"], np.maximum(all_latents.std((0, 1)), 0.01), atol=1e-6
        )
        self.assertEqual(float(expected["std"][-1]), float(np.float32(0.01)))
        np.testing.assert_allclose(
            expected["obs_scale"],
            np.maximum(
                data["observations"][train].astype(np.float64).std((0, 1)), 0.01
            ),
        )
        self.assertAlmostEqual(
            float(expected["return_scale"]),
            float(
                np.sqrt(np.mean(data["rewards"][train].astype(np.float64).sum(1) ** 2))
            ),
        )

    def test_missing_training_partition_fails(self):
        data = _tiny_data()
        data["split"][:] = 2
        with self.assertRaisesRegex(ValueError, "no training"):
            runner.normalized_data(data)


class TinyFitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name) / "fit"
        cls.data, cls.teacher, cls.protocol = (
            _tiny_data(),
            _TinyTeacher(),
            _tiny_protocol(),
        )
        cls.before = cls.teacher.frozen_digest()
        cls.validation_calls = []
        actual_predictions = runner.predictions

        def recording_predictions(predictor, data, indices, seed, **kwargs):
            cls.validation_calls.append(np.asarray(indices).copy())
            return actual_predictions(predictor, data, indices, seed, **kwargs)

        with mock.patch.object(
            runner, "predictions", side_effect=recording_predictions
        ):
            cls.result = runner.fit(
                cls.directory, cls.data, cls.teacher, cls.protocol, 0
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_four_real_updates_use_only_validation_for_selection_and_keep_teacher_frozen(
        self,
    ):
        self.assertEqual(self.result["updates"], 4)
        self.assertEqual(self.teacher.frozen_digest(), self.before)
        self.assertEqual(len(self.validation_calls), 2)
        for indices in self.validation_calls:
            np.testing.assert_array_equal(indices, [4, 5])
            self.assertTrue(np.all(self.data["split"][indices] == 1))
        records = self.result["validations"]
        self.assertEqual([record["update"] for record in records], [2, 4])
        self.assertEqual(
            self.result["selected"],
            min(records, key=lambda record: record["score"])["checkpoint"],
        )
        logs = [
            json.loads(line)
            for line in (self.directory / "training.jsonl").read_text().splitlines()
        ]
        self.assertEqual([entry["update"] for entry in logs], [1, 4])
        self.assertTrue(all(entry["update_norm"] > 0 for entry in logs))
        for entry in logs:
            self.assertTrue(all(np.isfinite(value) for value in entry.values()))
        norm = runner.load_npz(self.directory / "normalization.npz")
        for name, value in runner.normalized_data(self.data).items():
            np.testing.assert_array_equal(norm[name], value)

    def test_saved_validation_predictions_independently_reproduce_selection_score(self):
        norm = runner.load_npz(self.directory / "normalization.npz")
        for record in self.result["validations"]:
            saved = runner.load_npz(
                self.directory / f"validation_{record['update']:05d}.npz"
            )
            indices = saved["indices"]
            obs_error = np.mean(
                (
                    (saved["observations"].mean(1) - self.data["observations"][indices])
                    / norm["obs_scale"]
                )
                ** 2
            )
            return_error = np.mean(
                (
                    (
                        saved["rewards"].mean(1).sum(1)
                        - self.data["rewards"][indices].sum(1)
                    )
                    / norm["return_scale"]
                )
                ** 2
            )
            self.assertAlmostEqual(
                record["score"], float(obs_error + return_error), places=6
            )

    def test_selected_head_predicts_direct_and_composed_particles_without_teacher_changes(
        self,
    ):
        predictor = runner.load_predictor(self.directory, self.teacher)
        starts, actions = self.data["start"][:2], self.data["actions"][:2]
        noise = jax.random.normal(jax.random.PRNGKey(122), (2, 15, 4))
        for steps in (1, 2, 4):
            features = predictor.predict_trajectory(
                starts, actions, noise, flow_steps=steps
            )
            self.assertEqual(features.shape, (2, 15, 4))
            self.assertTrue(np.isfinite(np.asarray(features)).all())
        key = jax.random.PRNGKey(123)
        direct = predictor.particles(starts, actions, key, count=3)
        composed = predictor.particles(starts, actions, key, count=3, composed=True)
        self.assertEqual(direct.shape, (2, 3, 15, 4))
        self.assertEqual(composed.shape, direct.shape)
        np.testing.assert_array_equal(
            np.asarray(direct)[:, :, :5], np.asarray(composed)[:, :, :5]
        )
        obs, rewards = runner.predictions(
            predictor, self.data, np.asarray([6, 7]), 114, count=3
        )
        self.assertEqual(obs.shape, (2, 3, 15, 2))
        self.assertEqual(rewards.shape, (2, 3, 15))
        self.assertTrue(np.isfinite(obs).all() and np.isfinite(rewards).all())
        self.assertEqual(self.teacher.frozen_digest(), self.before)

    def test_loading_selected_head_with_mutated_teacher_is_rejected(self):
        changed = _TinyTeacher()
        changed.params["decoder/scale"][0] = 1.5
        with self.assertRaisesRegex(ValueError, "different frozen teacher"):
            runner.load_predictor(self.directory, changed)

    def test_fitted_parameters_differ_from_initial_parameters(self):
        from imf_dreamer_jax import parallel_trajectory as field

        initial = field.init(jax.random.PRNGKey(0), 4, 2, **self.protocol["field"])
        fitted = runner.load_predictor(self.directory, self.teacher).params
        self.assertEqual(jax.tree.structure(initial), jax.tree.structure(fitted))
        self.assertTrue(
            any(
                not np.array_equal(np.asarray(left), np.asarray(right))
                for left, right in zip(
                    jax.tree.leaves(initial), jax.tree.leaves(fitted)
                )
            )
        )


class FitNegativeControls(unittest.TestCase):
    def test_equal_validation_scores_select_first_checkpoint_not_last(self):
        data, teacher, protocol = _tiny_data(), _TinyTeacher(), _tiny_protocol()

        def exact_predictions(predictor, data, indices, seed, **kwargs):
            self.assertTrue(np.all(data["split"][indices] == 1))
            return (
                data["observations"][indices, None].copy(),
                data["rewards"][indices, None].copy(),
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                runner, "predictions", side_effect=exact_predictions
            ):
                result = runner.fit(
                    Path(directory) / "tie-fit", data, teacher, protocol, 2
                )
            self.assertEqual(result["selected"], "checkpoint_00002.pkl")
            self.assertEqual(
                [entry["score"] for entry in result["validations"]], [0.0, 0.0]
            )

    def test_teacher_mutation_during_fit_prevents_selection_artifact(self):
        data, teacher, protocol = _tiny_data(), _TinyTeacher(), _tiny_protocol()
        protocol["updates"] = 2

        def mutating_predictions(predictor, data, indices, seed, **kwargs):
            teacher.params["decoder/scale"][0] = 9.0
            return (
                data["observations"][indices, None].copy(),
                data["rewards"][indices, None].copy(),
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "mutated-fit"
            with mock.patch.object(
                runner, "predictions", side_effect=mutating_predictions
            ):
                with self.assertRaisesRegex(RuntimeError, "teacher mutated"):
                    runner.fit(destination, data, teacher, protocol, 3)
            self.assertFalse((destination / "selection.json").exists())


if __name__ == "__main__":
    unittest.main()
