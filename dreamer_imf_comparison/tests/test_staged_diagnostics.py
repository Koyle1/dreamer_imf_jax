"""Read-only diagnostics against actual tiny upstream encoders/RSSMs/heads."""

import json
from pathlib import Path
import unittest

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml
import dreamerv3.agent as upstream
from dreamer_imf_compare import dreamer_ablation_dynamics as dynamics
from dreamer_imf_compare.staged_diagnostics import (
    evaluate_batch,
    _distribution,
    _evaluate_arrays,
)


class DiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nn.COMPUTE_DTYPE = jnp.float32
        cfg = ruamel.yaml.YAML(typ="safe").load(
            (Path(upstream.__file__).parent / "configs.yaml").read_text()
        )["defaults"]["agent"]
        cfg["dyn"]["rssm"].update(
            deter=8, hidden=8, stoch=2, classes=3, blocks=2, imglayers=1, dynlayers=1
        )
        for name in ("enc", "dec"):
            cfg[name]["simple"].update(units=8, layers=1)
        for name in ("rewhead", "conhead", "policy", "value"):
            cfg[name].update(units=8, layers=1)
        cfg["rewhead"]["outscale"] = 1.0
        obs = {
            "vector": elements.Space(np.float32, (3,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "is_last": elements.Space(bool),
        }
        act = {"action": elements.Space(np.float32, (2,))}
        cls.batch = dict(
            vector=np.arange(21, dtype=np.float32).reshape(1, 7, 3) / 21,
            reward=np.ones((1, 7), np.float32),
            is_first=np.zeros((1, 7), bool),
            is_terminal=np.zeros((1, 7), bool),
            is_last=np.zeros((1, 7), bool),
            action=np.zeros((1, 7, 2), np.float32),
        )
        cls.models = []
        original = upstream.rssm.RSSM
        try:
            for dyn in (dynamics.GaussianRSSM, dynamics.IMFRSSM):
                upstream.rssm.RSSM = dyn
                model = object.__new__(upstream.Agent)
                upstream.Agent.__init__(model, obs, act, elements.Config(cfg))

                def initialize():
                    batch = jax.tree.map(jnp.asarray, cls.batch)
                    _, _, tok = model.enc(
                        model.enc.initial(1), batch, batch["is_first"], False
                    )
                    _, _, _, feat, _ = model.dyn.loss(
                        model.dyn.initial(1),
                        tok,
                        {"action": batch["action"]},
                        batch["is_first"],
                        False,
                    )
                    model.dec(model.dec.initial(1), feat, batch["is_first"], False)
                    model.rew(model.feat2tensor(feat), 2)
                    return 0

                params, _ = nj.pure(initialize)({}, seed=7, create=True)
                cls.models.append((model, params))
        finally:
            upstream.rssm.RSSM = original

    def test_distribution_not_arbitrary_pairing(self):
        x = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3)
        self.assertEqual(float(_distribution(x, x[::-1])["sliced_wasserstein2"]), 0)
        self.assertGreater(float(_distribution(x, x + 10)["mean_mse"]), 90)

    def test_actual_models_finite_repeatable_and_immutable(self):
        for model, params in self.models:
            before = {k: np.asarray(v).copy() for k, v in params.items()}
            out, ref = evaluate_batch(model, params, self.batch, 9)
            again, _ = evaluate_batch(model, params, self.batch, 9, ref)
            json.dumps(out, allow_nan=False)
            self.assertIsNotNone(out["quality_gate"])
            self.assertGreater(out["calibration"]["shared_core_dynamics_grad_norm"], 0)
            self.assertTrue(all(v == 0 for v in again["drift"].values()))
            self.assertEqual(out["rollout"], again["rollout"])
            for k in params:
                np.testing.assert_array_equal(before[k], params[k])
            if isinstance(model.dyn, dynamics.IMFRSSM):
                self.assertEqual(len(out["raw_velocity_time_bins"]), 4)
            else:
                self.assertEqual(out["one_vs_four_sample_mse_common_noise"], 0)

    def test_altered_posterior_has_drift_and_changed_prior_is_detected(self):
        model, params = self.models[1]
        out, ref = evaluate_batch(model, params, self.batch, 13)
        altered = dict(params)
        keys = [k for k in altered if "obsnormal" in k and k.endswith("bias")]
        self.assertEqual(len(keys), 1)
        altered[keys[0]] = altered[keys[0]] + 1
        changed, _ = evaluate_batch(model, altered, self.batch, 13, ref)
        self.assertGreater(changed["drift"]["mean_mse"], 0)
        bad = dict(params)
        last = sorted(
            k for k in bad if k.startswith("dyn/imf") and k.endswith("_bias")
        )[-1]
        bad[last] = bad[last] + 10
        changed, _ = evaluate_batch(model, bad, self.batch, 13)
        self.assertGreater(
            changed["distribution"]["1"]["mean_mse"],
            out["distribution"]["1"]["mean_mse"],
        )

    def test_mismatched_reference_and_all_reset_fail_closed(self):
        model, params = self.models[0]
        _, ref = evaluate_batch(model, params, self.batch, 3)
        with self.assertRaisesRegex(ValueError, "same fixed batch"):
            evaluate_batch(model, params, self.batch, 4, ref)
        changed = dict(self.batch, reward=self.batch["reward"] + 1)
        with self.assertRaisesRegex(ValueError, "same fixed batch"):
            evaluate_batch(model, params, changed, 3, ref)
        reset = dict(self.batch, is_first=np.ones((1, 7), bool))
        with self.assertRaisesRegex(ValueError, "No uninterrupted"):
            evaluate_batch(model, params, reset, 3)

    def test_bfloat16_and_explicit_transfer_guard(self):
        nn.COMPUTE_DTYPE = jnp.bfloat16
        _evaluate_arrays.clear_cache()
        try:
            for model, params in self.models:
                with jax.transfer_guard("disallow"):
                    metrics, _ = evaluate_batch(model, params, self.batch, 9)
                self.assertIsNotNone(metrics["quality_gate"])
        finally:
            nn.COMPUTE_DTYPE = jnp.float32
            _evaluate_arrays.clear_cache()

    def test_nonfinite_parameters_fail_closed(self):
        model, params = self.models[1]
        bad = dict(params)
        bad["dyn/imf0_bias"] = bad["dyn/imf0_bias"] * np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            evaluate_batch(model, bad, self.batch, 9)

    def test_reward_gradient_routing_and_reset_mask(self):
        model, params = self.models[0]
        original = model.config
        model.config = model.config.update(reward_grad=False)
        _evaluate_arrays.clear_cache()
        try:
            metrics, _ = evaluate_batch(model, params, self.batch, 9)
            self.assertEqual(
                metrics["gradient_norms"]["reward_scaled"]["shared_core_l2"], 0
            )
            self.assertAlmostEqual(
                metrics["calibration"]["shared_core_model_grad_norm"],
                metrics["gradient_norms"]["observation_scaled"]["shared_core_l2"],
                places=5,
            )
            reset = dict(self.batch, is_first=self.batch["is_first"].copy())
            reset["is_first"][:, 3] = True
            metrics, _ = evaluate_batch(model, params, reset, 9)
            self.assertIsNone(metrics["quality_gate"])
            self.assertEqual(metrics["rollout"]["1"]["5"]["valid_paths"], 0)
            self.assertGreater(metrics["rollout"]["1"]["1"]["valid_paths"], 0)
        finally:
            model.config = original
            _evaluate_arrays.clear_cache()


if __name__ == "__main__":
    unittest.main()
