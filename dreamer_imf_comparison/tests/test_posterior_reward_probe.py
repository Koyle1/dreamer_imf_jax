"""Exact no-physics inference, analytical scores, and false-favorable controls."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamer_imf_compare import posterior_reward_metrics as metrics
from dreamer_imf_compare import posterior_reward_probe as probe


def histories():
    position = np.arange(10 * 501 * 2, dtype=np.float32).reshape(10, 501, 2) / 10000
    action = np.tile(
        np.linspace(-0.5, 0.5, 501, dtype=np.float32)[None, :, None], (10, 1, 2)
    )
    first, last = np.zeros((10, 501), bool), np.zeros((10, 501), bool)
    first[:, 0], last[:, -1] = True, True
    return dict(
        position=position,
        to_target=position + 0.1,
        velocity=position - 0.2,
        reward=np.tile(np.arange(501, dtype=np.float32) % 3, (10, 1)),
        is_first=first,
        is_last=last,
        is_terminal=np.zeros_like(first),
        action=action,
        previous_action=np.concatenate(
            [np.zeros((10, 1, 2), np.float32), action[:, :-1]], 1
        ),
        episode_seed=np.arange(100, 110, dtype=np.uint32),
    )


class Categorical:
    def __init__(self, features):
        value = features.sum(-1).astype(jnp.float32)
        self.logits = jnp.stack([value * 0, value * 0.1, -value * 0.1], -1)

    def probabilities(self):
        return jax.nn.softmax(self.logits, -1)

    def pred(self):
        return self.probabilities() @ jnp.arange(3, dtype=jnp.float32)


class Native:
    def __init__(self, features):
        self.value = features.sum(-1).astype(jnp.float32)

    def pred(self):
        return self.value

    def loss(self, truth):
        return (self.value - truth) ** 2 + 7


def tiny_proxy(*, missing_stats=False, bad_std=False):
    calls = dict(observe=0, imagine=0, teacher=[], student=[])

    class Encoder:
        def initial(self, batch):
            return jnp.zeros((batch, 2))

        def __call__(self, carry, obs, reset, training):
            assert not training
            return carry, {}, obs["position"]

    class Dynamics:
        def initial(self, batch):
            return dict(deter=jnp.zeros((batch, 2)), stoch=jnp.zeros((batch, 1, 2)))

        def observe(self, carry, tokens, actions, reset, training):
            calls["observe"] += 1
            assert not training
            mean = tokens[..., None, :] * 2
            std = jnp.ones_like(mean) * (0 if bad_std else 0.3)
            z = mean + std * jax.random.normal(nj.seed(), mean.shape)
            features = dict(deter=actions["action"] + tokens, stoch=z)
            if not missing_stats:
                features.update(mean=mean, std=std)
            return carry, {}, features

        def imagine(self, starts, actions, horizon, training):
            calls["imagine"] += 1
            assert not training and horizon == 15 and set(starts) == {"deter", "stoch"}
            # A shape-preserving action-conditioned scan with independent noise.
            h = starts["deter"][:, None, :] + jnp.cumsum(actions["action"], axis=1)
            noise = jax.random.normal(nj.seed(), (len(h), horizon, 1, 2))
            z = starts["stoch"][:, None] + noise * 0.1
            return starts, dict(deter=h, stoch=z), actions

    class Decoder:
        def initial(self, batch):
            return jnp.zeros((batch, 2))

        def __call__(self, carry, feat, reset, training):
            assert not training
            return (
                carry,
                {},
                {
                    key: SimpleNamespace(pred=lambda: feat["deter"])
                    for key in probe.OBS_KEYS
                },
            )

    def head(name, features, bdims):
        assert bdims == 2
        calls[name].append(features.shape)
        return Categorical(features)

    model = SimpleNamespace(
        enc=Encoder(),
        dyn=Dynamics(),
        dec=Decoder(),
        feat2tensor=lambda f: jnp.concatenate(
            [f["deter"], f["stoch"].reshape(*f["deter"].shape[:2], -1)], -1
        ),
        rew=lambda f, bdims: Native(f),
        reward_student=lambda f, bdims: head("student", f, bdims),
        reward_teacher=lambda f, bdims: head("teacher", f, bdims),
    )
    spaces = {
        key: SimpleNamespace(shape=value.shape[2:])
        for key, value in histories().items()
        if key not in ("action", "previous_action", "episode_seed")
    }
    return SimpleNamespace(
        model=model,
        params={"untouched": jnp.array([3.25])},
        obs_space=spaces,
        calls=calls,
    )


def analytical_arrays(positive=10):
    truth = np.ones((40, 15), np.float64)
    truth[4 * positive :] = 0
    probability = np.eye(3)[truth.astype(int)]
    arrays = dict(
        truth_reward=truth,
        truth_observation=np.zeros((40, 15, 6)),
        action=np.zeros((40, 15, 2)),
        episode=np.repeat(np.arange(10), 4),
        episode_seed=np.repeat(np.arange(100, 110, dtype=np.uint32), 4),
        anchor=np.tile(probe.ANCHORS, 10),
        predicted_observation=np.ones((40, 8, 15, 6)),
    )
    for name in metrics.HEADS:
        imagined = name.startswith("predicted_")
        values = np.repeat(truth[:, None], 8, 1) if imagined else truth.copy()
        arrays[name] = values
        if name.endswith("original"):
            arrays[name + "_native_loss"] = np.full_like(values, 7.0)
        else:
            arrays[name + "_probabilities"] = (
                np.repeat(probability[:, None], 8, 1)
                if imagined
                else probability.copy()
            )
    return arrays


def study_fixture(positive=10):
    base = metrics.summarize_arrays(analytical_arrays(positive))
    # Keep only evidence used by the reporter to make 81-record controls cheap.
    base = {
        key: base[key]
        for key in ("rows", "particles", "episodes", "truth_digest", "by_episode")
    }
    for value in base["by_episode"].values():
        value["pooled"] = {
            "predicted_student": {
                k: 1.0 for k in ("reward_mse", "nll", "cumulative_reward_mse")
            },
            "predicted_original": {"reward_mse": 2.0, "cumulative_reward_mse": 4.0},
        }
        value["horizons"] = {
            str(h): copy.deepcopy(value["pooled"]) for h in metrics.HORIZONS
        }
    cells, records = [], []
    for arm in metrics.ARMS:
        for seed in metrics.SEEDS:
            delta = dict(A=0, B=10, C=70)[arm]
            cells.append(
                dict(
                    arm=arm,
                    seed=seed,
                    evaluations={
                        str(m): dict(
                            episodes=[
                                dict(seed=e, return_=100 + delta)
                                for e in range(100, 110)
                            ]
                        )
                        for m in metrics.MILESTONES
                    },
                )
            )
            for milestone in metrics.MILESTONES:
                for source in metrics.ARMS:
                    summary = copy.deepcopy(base)
                    if arm == "C":
                        for group in summary["by_episode"].values():
                            group["pooled"]["predicted_student"]["reward_mse"] = 0.8
                    records.append(
                        dict(
                            arm=arm,
                            seed=seed,
                            milestone=milestone,
                            source_arm=source,
                            summary=summary,
                        )
                    )
    return cells, records


def set_metric(records, arm, metric, value, *, seed=None, scope="pooled"):
    for record in records:
        if (
            record["arm"] != arm
            or record["milestone"] != 80000
            or (seed is not None and record["seed"] != seed)
        ):
            continue
        for group in record["summary"]["by_episode"].values():
            if scope == "pooled":
                group[scope]["predicted_student"][metric] = value
            else:
                for horizon in group["horizons"].values():
                    horizon["predicted_student"][metric] = value


class PureProbeTests(unittest.TestCase):
    def test_same_pass_teacher_actual_mean_action_alignment_and_no_mutation(self):
        proxy, data = tiny_proxy(), histories()
        before = copy.deepcopy(data)
        params = jax.tree.map(np.asarray, proxy.params)
        arrays = probe.predict_arrays(proxy, data, 19)
        self.assertEqual(proxy.calls["observe"], 1)
        self.assertEqual(proxy.calls["imagine"], 1)
        self.assertEqual(proxy.calls["teacher"], [(40, 15, 4)])
        self.assertEqual(proxy.calls["student"], [(40, 15, 4), (320, 15, 4)])
        self.assertFalse(any("predicted_teacher" in k for k in arrays))
        np.testing.assert_array_equal(
            arrays["posterior_mean"], probe._windows(data["position"], 1) * 2
        )
        np.testing.assert_array_equal(
            arrays["action"], probe._windows(data["action"], 0)
        )
        np.testing.assert_array_equal(
            arrays["truth_reward"], probe._windows(data["reward"], 1)
        )
        expected_h = np.stack(
            [
                data["position"][:, t] + data["previous_action"][:, t]
                for t in probe.ANCHORS
            ],
            1,
        ).reshape(40, 2)
        expected_h += arrays["action"][:, 0]
        np.testing.assert_allclose(
            arrays["predicted_observation"][:, 0, 0, :2], expected_h, atol=1e-7
        )
        real_h = probe._windows(data["position"] + data["previous_action"], 1)
        expected = Categorical(
            jnp.concatenate(
                [jnp.asarray(real_h), jnp.asarray(arrays["posterior_mean"])], -1
            )
        ).pred()
        np.testing.assert_allclose(arrays["posterior_teacher"], expected, atol=1e-7)
        self.assertFalse(
            np.array_equal(arrays["posterior_student"], arrays["posterior_teacher"])
        )
        for name in data:
            np.testing.assert_array_equal(data[name], before[name])
        np.testing.assert_array_equal(proxy.params["untouched"], params["untouched"])
        self.assertEqual(metrics.summarize_arrays(arrays)["episodes"], 10)

    def test_fixed_seed_exact_arrays_and_seed_changes_only_stochastic_predictions(self):
        proxy, data = tiny_proxy(), histories()
        first, second = (probe.predict_arrays(proxy, data, 19) for _ in range(2))
        self.assertEqual(set(first), set(second))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        other = probe.predict_arrays(proxy, data, 20)
        self.assertFalse(
            np.array_equal(first["predicted_student"], other["predicted_student"])
        )
        for name in (
            "truth_reward",
            "action",
            "posterior_mean",
            "posterior_teacher",
            "history_digest",
        ):
            np.testing.assert_array_equal(first[name], other[name])

    def test_immutable_create_independent_verify_and_tamper_controls(self):
        proxy, data = tiny_proxy(), histories()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probe.npz"
            self.assertEqual(probe.create_probe(path, proxy, data, 19), path)
            self.assertEqual(probe.verify_probe(path, proxy, data, 19), path)
            with self.assertRaises(FileExistsError):
                probe.create_probe(path, proxy, data, 19)
            with self.assertRaisesRegex(ValueError, "replay differs"):
                probe.verify_probe(path, proxy, data, 20)
            with np.load(path) as retained:
                modified = {k: retained[k] for k in retained.files}
            modified["truth_reward"] = modified["truth_reward"].astype(np.float64)
            bad = Path(temp) / "wrong-dtype.npz"
            np.savez(bad, **modified)
            with self.assertRaisesRegex(ValueError, "truth_reward"):
                probe.verify_probe(bad, proxy, data, 19)
            changed_history = copy.deepcopy(data)
            changed_history["position"][0, 50, 0] += 1
            with self.assertRaisesRegex(ValueError, "history_digest"):
                probe.verify_probe(path, proxy, changed_history, 19)
            modified.pop("anchor")
            missing = Path(temp) / "missing-field.npz"
            np.savez(missing, **modified)
            with self.assertRaisesRegex(ValueError, "fields differ"):
                probe.verify_probe(missing, proxy, data, 19)

    def test_history_invalid_inputs_fail_before_inference(self):
        controls = [
            lambda d: d.pop("episode_seed"),
            lambda d: d["is_first"].__setitem__((0, 20), True),
            lambda d: d["is_last"].__setitem__((0, -1), False),
            lambda d: d["is_terminal"].__setitem__((0, 2), True),
            lambda d: d["reward"].__setitem__((0, 2), 0.5),
            lambda d: d["reward"].__setitem__((0, 0), 1),
            lambda d: d["previous_action"].__setitem__((0, 1, 0), 0.9),
            lambda d: d["position"].__setitem__((0, 1, 0), np.nan),
            lambda d: d["action"].__setitem__((0, 1, 0), 2),
            lambda d: d["episode_seed"].__setitem__(0, 101),
            lambda d: d.__setitem__("position", d["position"][:, :-1]),
            lambda d: d.__setitem__("is_first", d["is_first"].astype(int)),
        ]
        for mutate in controls:
            data, proxy = histories(), tiny_proxy()
            mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                probe.predict_arrays(proxy, data, 19)
            self.assertEqual(proxy.calls["observe"], 0)
        for seed in (True, -1, 2**32, 1.5):
            with self.assertRaises(ValueError):
                probe.predict_arrays(tiny_proxy(), histories(), seed)

    def test_no_second_pass_fallback_or_invalid_standard_deviation(self):
        for kwargs, message in (
            (dict(missing_stats=True), "same-pass"),
            (dict(bad_std=True), "nonpositive"),
        ):
            proxy = tiny_proxy(**kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, message
            ):
                probe.predict_arrays(proxy, histories(), 19)
            self.assertEqual(proxy.calls["observe"], 1)

    def test_immutable_prediction_rejects_parameter_writes_and_creation(self):
        for key, error, message in (
            ("untouched", ValueError, "modified checkpoint state"),
            ("new_parameter", RuntimeError, "create=True"),
        ):
            proxy = tiny_proxy()

            def writes(features, bdims):
                nj.context()[key] = jnp.ones(1)
                return Native(features)

            proxy.model.rew = writes
            with self.subTest(key=key), self.assertRaisesRegex(error, message):
                probe.predict_arrays(proxy, histories(), 19)
            np.testing.assert_array_equal(proxy.params["untouched"], [3.25])
            self.assertEqual(set(proxy.params), {"untouched"})

    def test_bfloat16_outputs_have_portable_float32_evidence(self):
        proxy = tiny_proxy()

        class BfloatNative(Native):
            def pred(self):
                return self.value.astype(jnp.bfloat16)

        proxy.model.rew = lambda features, bdims: BfloatNative(features)
        arrays = probe.predict_arrays(proxy, histories(), 19)
        self.assertEqual(arrays["predicted_original"].dtype, np.dtype("float32"))
        self.assertEqual(metrics.summarize_arrays(arrays)["episodes"], 10)


class AnalyticalMetricTests(unittest.TestCase):
    def test_exact_scores_horizons_calibration_native_objective_and_coverage(self):
        arrays = analytical_arrays()
        arrays["predicted_original"] += np.arange(1, 16)[None, None, :]
        result = metrics.summarize_arrays(arrays)
        self.assertEqual(result["pooled"]["observation_mse"], 1)
        for horizon in metrics.HORIZONS:
            view = result["horizons"][str(horizon)]
            self.assertEqual(view["predicted_original"]["reward_mse"], horizon**2)
            self.assertEqual(
                view["predicted_original"]["cumulative_reward_mse"],
                (horizon * (horizon + 1) / 2) ** 2,
            )
            self.assertEqual(view["predicted_student"]["nll"], 0)
            self.assertEqual(view["predicted_student"]["brier"], 0)
            self.assertEqual(view["predicted_student"]["interval90_coverage"], 1)
        for name in ("posterior_original", "predicted_original"):
            self.assertEqual(result["pooled"][name]["native_objective"], 7)
            self.assertNotIn("nll", result["pooled"][name])
            self.assertNotIn("brier", result["pooled"][name])
        self.assertEqual(
            result["calibration"]["predicted_student"]["top_label"]["ece"], 0
        )
        self.assertEqual(result["coverage"]["class_counts"], [0, 600, 0])
        self.assertEqual(result["coverage"]["class_episode_counts"], [0, 10, 0])
        self.assertIsNone(result["pooled"]["predicted_student"]["zero_reward_mse"])
        json.dumps(result, allow_nan=False)

    def test_predictive_mixture_before_nll_and_cumulative_not_particle_errors(self):
        arrays = analytical_arrays()
        probability = arrays["predicted_student_probabilities"]
        probability[:, :4] = [0.1, 0.9, 0]
        probability[:, 4:] = [0.9, 0.1, 0]
        arrays["predicted_student"] = probability @ np.arange(3)
        result = metrics.summarize_arrays(arrays)["pooled"]["predicted_student"]
        self.assertAlmostEqual(result["nll"], -np.log(0.5))
        self.assertNotAlmostEqual(result["nll"], (-np.log(0.9) - np.log(0.1)) / 2)
        self.assertAlmostEqual(result["reward_mse"], 0.25)
        self.assertAlmostEqual(result["brier"], 0.5)
        self.assertAlmostEqual(result["cumulative_reward_mse"], 7.5**2)

    def test_certain_wrong_predictions_report_zero_probability_and_calibration(self):
        arrays = analytical_arrays()
        arrays["posterior_student_probabilities"][...] = [1, 0, 0]
        arrays["posterior_student"][...] = 0
        result = metrics.summarize_arrays(arrays)
        score = result["pooled"]["posterior_student"]
        self.assertAlmostEqual(score["nll"], -np.log(metrics.NLL_FLOOR))
        self.assertEqual(score["zero_true_class_probability"], 1)
        self.assertEqual(score["interval90_coverage"], 0)
        self.assertEqual(score["brier"], 2)
        self.assertEqual(
            result["calibration"]["posterior_student"]["top_label"]["ece"], 1
        )

    def test_sparse_positive_and_all_zero_are_inconclusive(self):
        for number in (0, 4, 5):
            result = metrics.summarize_arrays(analytical_arrays(number))
            self.assertEqual(result["coverage"]["positive_episodes"], number)
            self.assertEqual(
                result["coverage"]["status"],
                "adequate" if number >= 5 else "inconclusive",
            )
            self.assertEqual(
                result["coverage"]["class_counts"], [(10 - number) * 60, number * 60, 0]
            )

    def test_wrong_shapes_nonfinite_inconsistent_probabilities_and_fiction_fail(self):
        controls = [
            lambda a: a.pop("truth_reward"),
            lambda a: a.__setitem__("predicted_teacher", a["predicted_student"]),
            lambda a: a.__setitem__("truth_reward", a["truth_reward"][:39]),
            lambda a: a.__setitem__("predicted_student", a["predicted_student"][:, :7]),
            lambda a: a["truth_reward"].__setitem__((0, 0), np.inf),
            lambda a: a["truth_reward"].__setitem__((0, 0), 0.5),
            lambda a: a["posterior_student"].__setitem__((0, 0), 0.5),
            lambda a: a["predicted_student_probabilities"].__setitem__(
                (0, 0, 0, 0), 0.1
            ),
            lambda a: a["predicted_student_probabilities"].__setitem__(
                (0, 0, 0, 0), -1
            ),
            lambda a: a["anchor"].__setitem__(0, 99),
            lambda a: a["episode_seed"].__setitem__(0, 101),
        ]
        for mutate in controls:
            arrays = analytical_arrays()
            mutate(arrays)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                metrics.summarize_arrays(arrays)

    def test_metrics_import_does_not_load_jax_or_agent(self):
        code = "import sys; from dreamer_imf_compare import posterior_reward_metrics; assert 'jax' not in sys.modules; assert 'dreamerv3' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", code], check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class PairedStudyTests(unittest.TestCase):
    def test_deployed_comparisons_use_original_A_without_categorical_fiction(self):
        cells, records = study_fixture()
        report = metrics.paired_report(cells, records)
        ca = report["contrasts"]["C-A"]["80000"]
        self.assertIn("shadow student", ca["prediction_family"])
        self.assertAlmostEqual(ca["pooled"]["reward_mse"]["mean"], -0.2)
        deployed = ca["deployed_control"]
        self.assertEqual(
            deployed["heads"], {"C": "predicted_student", "A": "predicted_original"}
        )
        self.assertAlmostEqual(deployed["pooled"]["reward_mse"]["mean"], -1.2)
        self.assertAlmostEqual(deployed["pooled"]["cumulative_reward_mse"]["mean"], -3)
        self.assertNotIn("nll", deployed["pooled"])
        self.assertIsNone(deployed["categorical_nll"])

    def test_learning_curve_auc_checkpoint_mean_and_nested_bootstrap(self):
        cells, records = study_fixture()
        for cell in cells:
            if cell["arm"] == "C":
                for milestone, value in (("0", 120), ("40000", 150), ("80000", 210)):
                    for episode in cell["evaluations"][milestone]["episodes"]:
                        episode["return_"] = value
        curves = metrics.paired_report(cells, records)["learning_curves"]
        self.assertAlmostEqual(curves["arms"]["C"]["checkpoint_mean"]["mean"], 160)
        self.assertAlmostEqual(curves["arms"]["C"]["normalized_auc"]["mean"], 157.5)
        self.assertAlmostEqual(
            curves["arms"]["C"]["auc_return_native_steps"]["mean"], 157.5 * 80000
        )
        self.assertAlmostEqual(
            curves["contrasts"]["C-B"]["normalized_auc"]["mean"], 47.5
        )
        groups = np.arange(90).reshape(3, 3, 10)
        result = metrics._learning_curve(groups)
        rng = np.random.default_rng(912731)
        selected = rng.integers(0, 3, (2000, 3))
        environments = rng.integers(0, 10, (2000, 3, 3, 10))
        reference = groups[
            np.arange(3)[None, :, None, None], selected[:, None, :, None], environments
        ].mean((2, 3))
        reference = reference @ np.array([0.25, 0.5, 0.25])
        np.testing.assert_array_equal(
            result["normalized_auc"]["ci95"], np.quantile(reference, [0.025, 0.975])
        )
        self.assertEqual(result["normalized_auc"]["independent_seeds"], 3)

    def test_complete_balanced_study_passes_and_reports_all_contrasts(self):
        cells, records = study_fixture()
        report = metrics.paired_report(cells, records)
        self.assertEqual(report["decision"]["status"], "screen_passed")
        self.assertEqual(report["inference"]["independent_continuation_seeds"], 3)
        self.assertEqual(report["inference"]["source_policy_strata"], 3)
        self.assertEqual(set(report["contrasts"]), {"B-A", "C-B", "C-A"})
        for contrast in report["contrasts"].values():
            self.assertEqual(set(contrast), {"0", "40000", "80000"})
            for result in contrast.values():
                self.assertEqual(set(result["horizons"]), {"1", "3", "5", "10", "15"})
                self.assertEqual(result["returns"]["bootstrap_reps"], 2000)
        self.assertEqual(report["decision"]["returns"]["mean"], 60)
        self.assertEqual(report["decision"]["returns"]["ci95"], [60, 60])
        self.assertIn("weak", report["decision"]["returns"]["interpretation"])

    def test_screen_uses_pooled15_and_seed_mean_not_all_horizons_or_seeds(self):
        cells, records = study_fixture()
        set_metric(records, "C", "reward_mse", 1.2, seed=901)
        set_metric(records, "C", "reward_mse", 0.7, seed=902)
        set_metric(records, "C", "reward_mse", 0.7, seed=903)
        set_metric(records, "C", "nll", 100, scope="horizons")
        report = metrics.paired_report(cells, records)
        self.assertTrue(report["decision"]["passed"])
        self.assertAlmostEqual(
            report["decision"]["predicted_student"]["reward_mse"]["C"], 2.6 / 3
        )

    def test_metric_safety_and_benefit_boundaries(self):
        for metric, limit in (
            ("reward_mse", 0.9),
            ("nll", 1.1),
            ("cumulative_reward_mse", 1.1),
        ):
            for value, expected in (
                (limit - 1e-8, True),
                (limit, True),
                (limit + 1e-8, False),
            ):
                cells, records = study_fixture()
                set_metric(records, "C", metric, value)
                with self.subTest(metric=metric, value=value):
                    self.assertEqual(
                        metrics.paired_report(cells, records)["decision"]["passed"],
                        expected,
                    )
        cells, records = study_fixture()
        set_metric(records, "B", "reward_mse", 0)
        set_metric(records, "C", "reward_mse", 0)
        self.assertFalse(metrics.paired_report(cells, records)["decision"]["passed"])

    def test_return_all_seed_positive_and_mean_50_not_just_pooled_improvement(self):
        for deltas, expected in (
            ((-1, 100, 100), False),
            ((49, 49, 49), False),
            ((1, 74, 75), True),
        ):
            cells, records = study_fixture()
            for cell in cells:
                if cell["arm"] == "C":
                    for episode in cell["evaluations"]["80000"]["episodes"]:
                        episode["return_"] = 110 + deltas[cell["seed"] - 901]
            with self.subTest(deltas=deltas):
                self.assertEqual(
                    metrics.paired_report(cells, records)["decision"]["passed"],
                    expected,
                )

    def test_source_policy_duplicates_never_inflate_positive_environment_count(self):
        cells, records = study_fixture(4)
        decision = metrics.paired_report(cells, records)["decision"]
        self.assertEqual(decision["status"], "inconclusive")
        self.assertTrue(
            all(
                group["positive_environment_groups"] == 4
                for group in decision["coverage"].values()
            )
        )
        # A fifth environment positive in just one source block is a fifth group.
        for record in records:
            if record["source_arm"] == "C":
                group = record["summary"]["by_episode"]["104"]
                group["positive_count"], group["class_counts"] = 1, [59, 1, 0]
        self.assertEqual(
            metrics.paired_report(cells, records)["decision"]["status"], "screen_passed"
        )

    def test_hierarchical_bootstrap_exact_reference_and_not_pseudoreplicates(self):
        groups = np.repeat(np.array([[1], [2], [3]]), 10, axis=1)
        result = metrics._hierarchical(groups)
        rng = np.random.default_rng(912731)
        top = rng.integers(0, 3, (2000, 3))
        nested = rng.integers(0, 10, (2000, 3, 10))
        reference = groups[top[..., None], nested].mean((1, 2))
        np.testing.assert_array_equal(
            result["ci95"], np.quantile(reference, [0.025, 0.975])
        )
        self.assertEqual(result["ci95"], [1, 3])
        self.assertEqual(result["independent_seeds"], 3)
        self.assertEqual(result["environment_groups_per_seed"], 10)

    def test_incomplete_duplicate_unpaired_nonfinite_or_false_coverage_rejected(self):
        controls = [
            lambda c, r: c.pop(),
            lambda c, r: c.append(copy.deepcopy(c[0])),
            lambda c, r: r.pop(),
            lambda c, r: r.append(copy.deepcopy(r[0])),
            lambda c, r: c[0]["evaluations"].pop("0"),
            lambda c, r: c[0]["evaluations"]["0"]["episodes"].pop(),
            lambda c, r: c[0]["evaluations"]["0"]["episodes"][0].__setitem__(
                "seed", 200
            ),
            lambda c, r: c[0]["evaluations"]["0"]["episodes"][0].__setitem__(
                "return_", np.nan
            ),
            lambda c, r: r[0]["summary"].__setitem__("truth_digest", "f" * 64),
            lambda c, r: r[0]["summary"]["by_episode"]["100"].__setitem__(
                "positive_count", 61
            ),
            lambda c, r: r[0]["summary"]["by_episode"]["100"]["pooled"][
                "predicted_student"
            ].__setitem__("nll", -1),
            lambda c, r: r[0]["summary"]["by_episode"]["100"]["pooled"][
                "predicted_student"
            ].__setitem__("reward_mse", np.inf),
        ]
        for mutate in controls:
            cells, records = study_fixture()
            mutate(cells, records)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                metrics.paired_report(cells, records)


if __name__ == "__main__":
    unittest.main()
