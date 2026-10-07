import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import prior_correction as field
from dreamer_imf_compare.prior_correction import normalization, rollout, validate_data
from dreamer_imf_compare.prior_correction_study import (
    conditional_samples,
    fit,
    load_npz,
    predictions,
    PROTOCOL,
    read,
    reports,
    score,
)


def fixture():
    n = 12
    rng = np.random.default_rng(2)
    data = dict(
        start=rng.normal(size=(n, 4)).astype("float32"),
        targets=rng.normal(size=(n, 15, 4)).astype("float32"),
        actions=rng.uniform(-1, 1, (n, 15, 2)).astype("float32"),
        observations=rng.normal(size=(n, 15, 6)).astype("float32"),
        rewards=rng.integers(0, 3, (n, 15)).astype("float32"),
        episode=np.repeat(np.arange(3), 4),
        split=np.repeat(np.arange(3), 4),
        anchor=np.ones(n, int) * 100,
        plan=np.tile(np.arange(4), 3),
    )
    for e in range(3):
        rows = slice(e * 4, e * 4 + 4)
        data["start"][rows] = data["start"][e * 4]
    return data


class ToyAdapter:
    def __init__(self):
        self.checks = 0
        self.teacher = SimpleNamespace(
            deter=2,
            feature_dim=4,
            decode=lambda f: (jnp.concatenate((f, f[..., :2]), -1), f[..., 0]),
            parameter_count=0,
        )
        self.teacher.rollout = lambda s, a, seed, particles, flow_steps: rollout(
            self,
            None,
            None,
            jnp.asarray(s),
            jnp.asarray(a),
            jax.random.PRNGKey(seed),
            particles=particles,
            bypass=True,
        )

    def core(self, f, a):
        return 0.8 * f[..., :2] + 0.1 * f[..., 2:] + a

    def prior(self, h):
        return 0.5 * h, jnp.ones_like(h) * 0.2

    def assert_frozen(self):
        self.checks += 1


class RunnerTests(unittest.TestCase):
    def test_split_isolation_and_negative_controls(self):
        data = fixture()
        validate_data(data, 4)
        norm = normalization(data, 2)
        data["targets"][data["split"] != 0] += 10000
        for k, v in normalization(data, 2).items():
            np.testing.assert_array_equal(norm[k], v)
        for mutate in (
            lambda d: d["split"].__setitem__(0, 1),
            lambda d: d["targets"].__setitem__((0, 0, 0), np.nan),
            lambda d: d["actions"].__setitem__((0, 0, 0), 2),
            lambda d: d["plan"].__setitem__(0, 1),
        ):
            damaged = fixture()
            mutate(damaged)
            with self.assertRaises(ValueError):
                validate_data(damaged, 4)

    def test_identity_rollout_causality_and_real_target_not_input(self):
        data, adapter = fixture(), ToyAdapter()
        norm = normalization(data, 2)
        params = field.init(jax.random.PRNGKey(0), 2, 2, 8, 1)
        starts = jnp.asarray(data["start"][:1])
        actions = jnp.asarray(data["actions"][:1])
        key = jax.random.PRNGKey(12)
        base = rollout(adapter, params, norm, starts, actions, key, bypass=True)
        corr = rollout(adapter, params, norm, starts, actions, key)
        np.testing.assert_array_equal(base, corr)
        changed = actions.at[:, 5:].set(0)
        future = rollout(adapter, params, norm, starts, changed, key)
        np.testing.assert_array_equal(corr[:, :, :5], future[:, :, :5])
        self.assertGreater(
            float(jnp.max(jnp.abs(corr[:, :, 6:] - future[:, :, 6:]))), 0.01
        )

    def test_tiny_fit_metrics_and_prediction_replay(self):
        adapter, data = ToyAdapter(), fixture()
        config = dict(
            PROTOCOL,
            updates=2,
            validation_period=1,
            batch_size=8,
            hidden_dim=8,
            depth=1,
            validation_particles=2,
        )
        retained = copy.deepcopy(data)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            params, norm = fit(output, adapter, data, config)
            selection = read(output / "selection.json")
            self.assertEqual([r["update"] for r in selection["candidates"]], [0, 1, 2])
            self.assertEqual(
                selection["selected"],
                min(selection["candidates"], key=lambda r: (r["score"], r["update"])),
            )
            test = np.flatnonzero(data["split"] == 2)
            report, raw = reports(adapter, params, norm, data, test, output)
            self.assertEqual(
                set(report),
                {
                    "base",
                    "corrected",
                    "corrected_four",
                    "original_imf_four",
                    "real_posterior",
                },
            )
            again = predictions(adapter, params, norm, data, test)
            for k in again:
                np.testing.assert_array_equal(again[k], raw["corrected"][k])
                np.testing.assert_array_equal(
                    again[k], load_npz(output / "test-corrected.npz")[k]
                )
            self.assertGreater(adapter.checks, 0)
        for k in data:
            np.testing.assert_array_equal(data[k], retained[k])

    def test_selection_score_uses_final_observables_and_fails_bad_predictions(self):
        data = fixture()
        ix = np.arange(4)
        pred = dict(
            observation=data["observations"][ix, None], reward=data["rewards"][ix, None]
        )
        self.assertEqual(score(pred, data, ix, np.ones(6), 1.0), 0)
        pred["reward"] = pred["reward"] + 1
        self.assertGreater(score(pred, data, ix, np.ones(6), 1.0), 200)

    def test_energy_scores_measure_transformed_not_base_distribution(self):
        data, adapter = fixture(), ToyAdapter()
        norm = normalization(data, 2)
        p = field.init(jax.random.PRNGKey(1), 2, 2, 8, 1)
        ix = np.arange(8, 12)
        initial, raw = conditional_samples(adapter, p, norm, data, ix)
        self.assertEqual(initial["base"], initial["corrected"])
        np.testing.assert_array_equal(raw["base"], raw["corrected"])
        layers = list(p["layers"])
        layers[-1] = dict(layers[-1], bias=jnp.ones(4) * 20)
        changed, _ = conditional_samples(
            adapter, dict(layers=tuple(layers)), norm, data, ix
        )
        self.assertGreater(
            changed["corrected"]["energy_score"], changed["base"]["energy_score"] + 5
        )


if __name__ == "__main__":
    unittest.main()
