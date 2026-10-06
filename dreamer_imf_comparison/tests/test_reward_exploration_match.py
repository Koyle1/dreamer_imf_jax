"""Offline matched fitting contracts; tiny real Ninjax fits, no simulator."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamer_imf_compare import reward_exploration_match as match
from dreamer_imf_compare.reward_exploration_agent import (
    build_control_head,
    build_ensemble,
    episode_bootstrap,
    predicted_mean_disagreement,
)


def config():
    return elements.Config(
        dict(
            rewhead=dict(
                layers=1,
                units=16,
                act="silu",
                norm="rms",
                output="symexp_twohot",
                outscale=0.0,
                winit="trunc_normal_in",
                bins=255,
            )
        )
    )


def fixture():
    episode = np.repeat(np.arange(64), 16)
    return dict(
        episode=episode,
        split=np.repeat(np.repeat(np.arange(3), [40, 12, 12]), 16),
        mode=episode % 2,
        anchor=np.tile(np.repeat([100, 200, 300, 400], 4), 64),
        plan=np.tile(np.arange(4), 256),
        rewards=np.ones((1024, 15), np.float32),
        observations=np.zeros((1024, 15, 6), np.float32),
        actions=np.zeros((1024, 15, 2), np.float32),
        start=np.broadcast_to(np.zeros(1, np.float32), (1024, 2560)),
        targets=np.broadcast_to(np.zeros(1, np.float32), (1024, 15, 2560)),
    )


class SplitSelectionContracts(unittest.TestCase):
    def test_production_update_budget_and_seed_count_are_frozen(self):
        protocol = dict(
            matched_fit=match.MATCH_DEFAULTS, ensemble_fit=match.ENSEMBLE_DEFAULTS
        )
        fitted, ensemble = match.protocol_settings(protocol)
        self.assertEqual(fitted["seeds"], [0, 1, 2])
        self.assertEqual(
            (fitted["updates"], fitted["batch_size"], fitted["validation_period"]),
            (5000, 256, 250),
        )
        self.assertEqual(fitted["learning_rate"], 3e-4)
        for name in ("matched_fit", "ensemble_fit"):
            wrong = copy.deepcopy(protocol)
            wrong[name]["updates"] -= 1
            with self.assertRaisesRegex(ValueError, "frozen budgets"):
                match.protocol_settings(wrong)

    def test_exact_episode_split_and_negative_leakage_control(self):
        data = fixture()
        train, val, test = match.split_indices(data)
        self.assertEqual([len(x) for x in (train, val, test)], [640, 192, 192])
        self.assertEqual(
            [len(np.unique(data["episode"][x])) for x in (train, val, test)],
            [40, 12, 12],
        )
        data["split"][0] = 1
        with self.assertRaisesRegex(ValueError, "split leakage"):
            match.split_indices(data)

    def test_train_normalizers_ignore_heldout_values_and_reject_heldout_rows(self):
        data = fixture()
        train, val, test = match.split_indices(data)
        baseline = match.train_normalization(data, train)
        data["targets"] = np.broadcast_to(
            (data["split"] * 10000)[:, None, None], (1024, 15, 2560)
        )
        data["observations"][val] = 50000
        data["observations"][test] = -90000
        changed = match.train_normalization(data, train)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], changed[key])
        self.assertEqual(changed["mean"].shape, (2560,))
        self.assertEqual(changed["observation_mean"].shape, (6,))
        np.testing.assert_array_equal(changed["std"], np.full(2560, 0.01, np.float32))
        with self.assertRaisesRegex(ValueError, "training rows"):
            match.train_normalization(data, np.concatenate([train, val[:1]]))

    def test_same_indices_for_all_six_candidates_no_initialization_seed_dependency(
        self,
    ):
        indices = match.minibatch_indices(9600, match.MATCH_DEFAULTS)
        self.assertEqual(indices.shape, (5000, 256))
        self.assertTrue(indices.min() >= 0 and indices.max() < 9600)
        for _kind in match.KINDS:
            for _seed in match.MATCH_DEFAULTS["seeds"]:
                np.testing.assert_array_equal(
                    indices, match.minibatch_indices(9600, match.MATCH_DEFAULTS)
                )
        changed = indices.copy()
        changed[0, 0] += 1
        self.assertNotEqual(match.array_digest(indices), match.array_digest(changed))

    def test_validation_only_selection_ties_keep_earliest_seed(self):
        rows = [
            dict(kind=k, seed=s, score=1 if s else 2, test_score=-1000 * (s == 2))
            for k in match.KINDS
            for s in (0, 1, 2)
        ]
        chosen = match.select_primary(rows)
        self.assertEqual([chosen[k]["seed"] for k in match.KINDS], [1, 1])
        for record in rows:
            record["test_score"] = float("nan")
        self.assertEqual(chosen, match.select_primary(rows))
        rows[0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            match.select_primary(rows)

    def test_transition_alignment_uses_start_then_previous_posterior(self):
        data = dict(
            start=np.array([[90.0, 91.0]]),
            targets=np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
            actions=np.array([[[0.1], [0.2], [0.3]]]),
            observations=np.array([[[10.0], [20.0], [30.0]]]),
        )
        features, actions, observations = match.transition_arrays(data, np.array([0]))
        np.testing.assert_array_equal(features, [[[90, 91], [1, 2], [3, 4]]])
        np.testing.assert_array_equal(actions, data["actions"])
        np.testing.assert_array_equal(observations, data["observations"])

    def test_bootstrap_membership_is_episode_level_and_shared_with_online(self):
        episodes = np.repeat(np.arange(40), 16)
        unique, masks, rows = match.episode_bootstrap(episodes, 5, 9191)
        np.testing.assert_array_equal(masks, episode_bootstrap(unique, 9191))
        np.testing.assert_array_equal(rows, episode_bootstrap(episodes, 9191))
        self.assertTrue(masks.any(1).all())
        self.assertGreater(len({tuple(row) for row in masks}), 1)
        with self.assertRaisesRegex(ValueError, "five"):
            match.episode_bootstrap(episodes, 4, 9191)

    def test_identical_means_zero_disagreement_no_favorable_correlation_requirement(
        self,
    ):
        predictions = np.broadcast_to(np.float32(0.01), (5, 3, 15, 6))
        np.testing.assert_array_equal(
            match.disagreement(predictions), np.zeros((3, 15))
        )
        np.testing.assert_array_equal(
            match.disagreement(predictions), predicted_mean_disagreement(predictions)
        )
        result = match.ensemble_diagnostics(
            predictions, np.zeros((3, 15, 6)), {"shuffled": predictions}
        )
        self.assertIsNone(result["actual"]["pearson"])
        self.assertIsNone(result["actual"]["spearman"])
        self.assertEqual(result["actual"]["transitions"], 45)
        # Negative association is retained, never rejected or tuned away.
        result = match._association(np.arange(5), -np.arange(5))
        self.assertAlmostEqual(result["pearson"], -1)
        self.assertAlmostEqual(result["spearman"], -1)

    def test_no_parent_model_instantiation_or_simulator_in_match_source(self):
        source = Path(match.__file__).read_text()
        for forbidden in (
            "FrozenModel(",
            "ReacherAdapter(",
            "dm_control",
            ".policy(",
            ".train(",
        ):
            self.assertNotIn(forbidden, source)

    def test_raw_report_reconstructs_all_seed_metrics_and_positive_coverage(self):
        data = fixture()
        rows = np.flatnonzero(data["split"] == 2)
        predictions = {
            "existing": np.zeros((192, 15)),
            "categorical": np.ones((192, 15)),
        }
        predictions.update(
            {
                f"{kind}_{seed}": predictions[kind]
                for kind in match.KINDS
                for seed in range(3)
            }
        )
        report = match._reward_metrics(data, rows, predictions, {}, 15)
        self.assertEqual(len(report["candidate_test"]), 6)
        self.assertEqual(report["candidate_test"]["existing_0"]["raw_step_mse"], 1)
        self.assertEqual(
            report["candidate_test"]["existing_0"]["normalized_cumulative_15_mse"], 1
        )
        self.assertEqual(report["candidate_test"]["categorical_2"]["raw_step_mse"], 0)
        self.assertEqual(
            report["test"]["existing"]["coverage"]["episodes_with_positive_reward"], 12
        )
        self.assertEqual(report["additional_simulator_steps"], 0)

    def test_export_loader_binds_hashes_and_refuses_parent_namespace_or_corruption(
        self,
    ):
        for failure in (None, "corruption", "namespace"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                (output / "ensemble").mkdir()
                control = dict(
                    kind="categorical", params={"control_rew/weight": np.ones(2)}
                )
                if failure == "namespace":
                    control["params"]["rew/weight"] = np.zeros(2)
                ensemble = dict(
                    kind="ensemble",
                    params={"explore_ensemble/weight": np.zeros(2)},
                    bootstrap_seed=9191,
                )
                match._dump(output / "categorical_control.pkl", control)
                match._dump(output / "ensemble/ensemble.pkl", ensemble)
                exports = dict(
                    control={
                        "categorical": dict(
                            path="categorical_control.pkl",
                            sha256=match.sha(output / "categorical_control.pkl"),
                        )
                    },
                    ensemble=dict(
                        path="ensemble/ensemble.pkl",
                        sha256=match.sha(output / "ensemble/ensemble.pkl"),
                    ),
                )
                match._write_json(output / "exports.json", exports)
                match._write_json(
                    output / "manifest.json",
                    dict(
                        protocol=dict(
                            matched_fit=match.MATCH_DEFAULTS,
                            ensemble_fit=match.ENSEMBLE_DEFAULTS,
                        )
                    ),
                )
                match._write_json(
                    output / "match_completed.json",
                    dict(
                        stage="match",
                        additional_simulator_steps=0,
                        artifacts={
                            str(p.relative_to(output)): match.sha(p)
                            for p in output.rglob("*")
                            if p.is_file()
                        },
                    ),
                )
                if failure == "corruption":
                    # Direct fixture mutation models post-completion disk corruption.
                    with (output / "categorical_control.pkl").open("ab") as stream:
                        stream.write(b"invalid")
                if failure:
                    with self.assertRaises(ValueError):
                        match.load_export(output, "categorical")
                else:
                    loaded = match.load_export(output, "categorical")
                    self.assertEqual(
                        set(loaded),
                        {
                            "control_rew/weight",
                            "explore_ensemble/weight",
                            "explore_bootstrap_seed/value",
                        },
                    )
                    self.assertEqual(int(loaded["explore_bootstrap_seed/value"]), 9191)


class SharedHeadTraining(unittest.TestCase):
    def setUp(self):
        self.config = config()
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(24, 15, 6)).astype(np.float32)
        self.y = (self.x[..., 0] > 0).astype(np.float32) * 2
        self.vx = rng.normal(size=(4, 15, 6)).astype(np.float32)
        self.vy = (self.vx[..., 0] > 0).astype(np.float32) * 2
        self.normalization = dict(mean=self.x.mean((0, 1)), std=self.x.std((0, 1)))

    def test_both_heads_independent_initialization_and_finite_gradients(self):
        for kind in match.KINDS:
            first = match.initialize_control(
                kind, self.config, 6, 0, self.normalization
            )
            second = match.initialize_control(
                kind, self.config, 6, 1, self.normalization
            )
            self.assertTrue(any(not np.array_equal(first[k], second[k]) for k in first))
            _, loss = match._head_functions(kind, self.config)
            grad = jax.grad(lambda p: loss(p, self.x[:2], self.y[:2], seed=0)[1])(first)
            self.assertTrue(all(np.isfinite(v).all() for v in grad.values()))
            self.assertTrue(any(np.any(v != 0) for v in grad.values()))
            self.assertTrue(all(k.startswith("control_rew/") for k in first))
            if kind == "categorical":
                for key in match.CONTROL_FIXED:
                    np.testing.assert_array_equal(grad[key], np.zeros_like(grad[key]))

    def test_existing_preprocessing_loss_and_expected_raw_decoder_are_upstream_exact(
        self,
    ):
        import embodied.jax
        from embodied.jax import nets

        head = embodied.jax.MLPHead(
            elements.Space(np.float32, ()), **self.config.rewhead, name="control_rew"
        )
        fn = nj.pure(
            lambda x, y: (head(nets.cast(x), 2).pred(), head(nets.cast(x), 2).loss(y))
        )
        params = match.initialize_control(
            "existing", self.config, 6, 23, self.normalization
        )
        expected = jax.jit(lambda p, x, y: fn(p, x, y, seed=0)[1])(
            params, self.x, self.y
        )
        actual = match.predictor("existing", self.config)(params, self.x)
        np.testing.assert_array_equal(expected[0], actual)
        _, loss = match._head_functions("existing", self.config)
        expected_loss = jax.jit(lambda p, x, y: fn(p, x, y, seed=0)[1][1].mean())(
            params, self.x, self.y
        )
        actual_loss = jax.jit(lambda p, x, y: loss(p, x, y, seed=0)[1])(
            params, self.x, self.y
        )
        self.assertEqual(float(expected_loss), float(actual_loss))

    def test_both_shared_modules_train_export_exactly_replay_and_remain_immutable(self):
        settings = dict(
            match.MATCH_DEFAULTS, updates=6, batch_size=32, validation_period=2
        )
        indices = match.minibatch_indices(self.y.size, settings)
        before = self.x.copy()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            records = []
            for kind in match.KINDS:
                record = match.fit_candidate(
                    output / f"{kind}_0",
                    kind,
                    0,
                    self.config,
                    self.x,
                    self.y,
                    self.vx,
                    self.vy,
                    self.normalization,
                    indices,
                    15,
                    settings,
                )
                records.append(record)
                saved = json.loads(
                    (output / record["directory"] / "selection.json").read_text()
                )
                self.assertEqual(saved["updates_completed"], 6)
                self.assertEqual([x["update"] for x in saved["validations"]], [2, 4, 6])
                self.assertEqual(saved["minibatch_sha256"], match.array_digest(indices))
                self.assertEqual(
                    record["update"],
                    saved["validations"][match.first_minimum(saved["validations"])][
                        "update"
                    ],
                )
                self.assertTrue(
                    all(np.isfinite(x["gradient_norm"]) for x in saved["training"])
                )
                export = match.export_control(
                    output, record, self.config, self.normalization, self.vx
                )
                self.assertEqual(export["sha256"], match.sha(output / export["path"]))
                payload = match._load(output / export["path"])
                online = build_control_head(kind, self.config)
                from embodied.jax import nets

                online_predict = nj.pure(lambda x: online(nets.cast(x), 2).pred())
                pred = jax.jit(lambda p, x: online_predict(p, x, seed=0)[1])(
                    payload["params"], self.vx
                )
                retained = match.load_npz(
                    output
                    / record["directory"]
                    / f"validation_{record['update']:05d}.npz"
                )
                np.testing.assert_array_equal(pred, retained["prediction"])
                if kind == "categorical":
                    for key, name in zip(match.CONTROL_FIXED, ("mean", "std")):
                        np.testing.assert_array_equal(
                            payload["params"][key], self.normalization[name]
                        )
                with self.assertRaises(FileExistsError):
                    match.export_control(
                        output, record, self.config, self.normalization, self.vx
                    )
            self.assertEqual(match.select_primary(records)["existing"]["seed"], 0)
        np.testing.assert_array_equal(self.x, before)

    def test_categorical_prediction_uses_probability_mean_not_argmax(self):
        params = match.initialize_control(
            "categorical", self.config, 6, 0, self.normalization
        )
        params = {
            k: (v if k in match.CONTROL_FIXED else jnp.zeros_like(v))
            for k, v in params.items()
        }
        pred = match.predictor("categorical", self.config)(params, self.x)
        np.testing.assert_array_equal(pred, np.ones(self.y.shape))

    def test_common_ensemble_train_only_fit_fixed_stats_and_shared_export(self):
        rng = np.random.default_rng(59)
        x = rng.normal(size=(40, 3, 6)).astype(np.float32)
        data = dict(
            start=x[:, 0],
            targets=x,
            actions=rng.uniform(-1, 1, (40, 3, 2)).astype(np.float32),
            observations=(x * 2 + 1),
            rewards=np.ones((40, 3)),
            split=np.zeros(40, np.int32),
            episode=np.arange(40),
        )
        rows = np.arange(40)
        normalization = match.train_normalization(data, rows)
        settings = dict(match.ENSEMBLE_DEFAULTS, updates=6, batch_size=16)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "ensemble"
            payload = match.fit_ensemble(directory, data, rows, normalization, settings)
            self.assertEqual(payload["bootstrap_seed"], 9191)
            self.assertEqual(payload["bonus_clip"], 10)
            self.assertGreaterEqual(payload["scale"], float(np.float32(1e-4)))
            replay = match._load(directory / "ensemble.pkl")
            features, actions, observations = match.transition_arrays(data, rows)
            pred = np.asarray(
                match.ensemble_predictor()(replay["params"], features, actions)
            )
            retained = match.load_npz(directory / "train_calibration.npz")
            np.testing.assert_array_equal(pred, retained["prediction"])
            desired = np.float32(
                max(float(np.quantile(match.disagreement(pred), 0.95)), 1e-4)
            )
            self.assertEqual(payload["scale"], float(desired))
            original = match._load(directory / "initial.pkl")
            for key in match.ENSEMBLE_FIXED[:-1]:
                np.testing.assert_array_equal(original[key], replay["params"][key])
            self.assertTrue(
                any(
                    not np.array_equal(v, replay["params"][k])
                    for k, v in original.items()
                    if k not in match.ENSEMBLE_FIXED
                )
            )
            ensemble = build_ensemble()
            online = nj.pure(lambda f, a: ensemble(f, a))
            online_fn = jax.jit(lambda p, f, a: online(p, f, a, seed=0)[1])
            np.testing.assert_array_equal(
                online_fn(replay["params"], features, actions), pred
            )
            data["split"][0] = 2
            with self.assertRaisesRegex(ValueError, "training rows"):
                match.fit_ensemble(
                    Path(tmp) / "invalid", data, rows, normalization, settings
                )


if __name__ == "__main__":
    unittest.main()
