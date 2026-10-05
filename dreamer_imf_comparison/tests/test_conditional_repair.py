import json
from pathlib import Path
import unittest
from unittest import mock

import elements
import embodied.jax.nets as nn
import dreamerv3
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
from ruamel.yaml import YAML

from dreamer_imf_compare.conditional_dynamics import (
    balanced_alignment,
    conditional_kl,
    install,
)
from dreamer_imf_compare.conditional_schedule import (
    ConditionalSchedule,
    CONTROLS,
    retained_views,
)
from dreamer_imf_compare.joint_diagnostics import _observable_arrays


class ConditionalRepairTests(unittest.TestCase):
    def test_analytic_conditional_kl_not_fixed_normal(self):
        mean = jnp.array([[[3.0, -4.0]]])
        std = jnp.array([[[0.4, 0.8]]])
        np.testing.assert_allclose(conditional_kl(mean, std, mean, std), 0, atol=1e-7)
        self.assertGreater(
            float(
                conditional_kl(mean, std, jnp.zeros_like(mean), jnp.ones_like(std))[0]
            ),
            10,
        )
        np.testing.assert_allclose(
            conditional_kl(mean, std, mean + 1, std),
            (1 / 0.4**2 + 1 / 0.8**2) / 2,
            rtol=1e-6,
        )

    def test_balanced_gradient_routing_and_free_nats(self):
        args = (
            jnp.ones((1, 2, 3)) * 2,
            jnp.ones((1, 2, 3)) * 0.4,
            jnp.zeros((1, 2, 3)),
            jnp.ones((1, 2, 3)) * 0.8,
        )
        dyn = jax.grad(
            lambda *a: balanced_alignment(*a)[0].sum(), argnums=(0, 1, 2, 3)
        )(*args)
        rep = jax.grad(
            lambda *a: balanced_alignment(*a)[1].sum(), argnums=(0, 1, 2, 3)
        )(*args)
        for g in dyn[:2] + rep[2:]:
            np.testing.assert_array_equal(g, 0)
        for g in dyn[2:] + rep[:2]:
            self.assertGreater(float(jnp.linalg.norm(g)), 0)
            self.assertTrue(np.isfinite(g).all())
        q, s = args[:2]
        lo = jax.grad(lambda q: balanced_alignment(q, s, q * 0 + 2, s)[1].sum())(q)
        np.testing.assert_array_equal(lo, 0)

    def test_fixed_controls_ignore_reward_gate_and_evaluation_outcomes(self):
        schedule = ConditionalSchedule()
        for native in (0, 2048, 50000, 100000, 200000, 500000):
            schedule.observe(native, {"arbitrary": float("nan")})
            self.assertEqual(schedule.controls(native), CONTROLS)

    def test_positive_selection_retains_history_and_eligible_endpoints(self):
        batch = dict(reward=np.zeros((16, 65)), is_first=np.zeros((16, 65), bool))
        batch["is_first"][:, 0] = True
        batch["reward"][3, 2] = 2  # ineligible; cannot certify a five-step endpoint
        self.assertNotIn("positive_selected", retained_views(batch))
        batch["reward"][4, 11] = 2
        value = retained_views(batch)["positive_selected"]
        self.assertEqual(value["reward"].shape, (1, 65))
        self.assertEqual(value["reward"][0, 11], 2)
        batch["is_first"][4, 10] = True
        self.assertNotIn("positive_selected", retained_views(batch))

    def test_protocol_scope_and_budget(self):
        root = Path(__file__).parents[1]
        old = json.loads((root / "joint_protocol.json").read_text())
        new = json.loads((root / "conditional_protocol.json").read_text())
        for k in (
            "arms",
            "seeds",
            "native_steps",
            "action_repeat",
            "envs",
            "train_ratio",
            "eval_at_native_steps",
            "eval_episodes",
            "batch_size",
            "batch_length",
            "preflight",
        ):
            self.assertEqual(new[k], old[k])

    def test_strict_verification_rejects_wrong_cells_before_loading(self):
        from dreamer_imf_compare import conditional_study as study

        with mock.patch.object(study.evidence, "read") as read:
            for cell in (
                dict(index=1, arm="imf", seed=431),
                dict(index=0, arm="gaussian", seed=431),
            ):
                with self.assertRaises(ValueError):
                    study.verify_cell("unused", "training", cell)
            read.assert_not_called()

    def test_extra_validation_runs_before_marker_publication(self):
        from dreamer_imf_compare import conditional_study as study

        with mock.patch.object(
            study.evidence, "read", return_value={}
        ), mock.patch.object(
            study.joint, "verify_extra", side_effect=ValueError("invalid extra")
        ), mock.patch.object(
            study.joint.staged, "_original_verify"
        ) as marker:
            with self.assertRaisesRegex(ValueError, "invalid extra"):
                study.verify_cell(
                    "unused", "training", dict(index=0, arm="imf", seed=431)
                )
            marker.assert_not_called()

    def test_complete_training_sampler_and_critic_reward_gradient_boundaries(self):
        nn.COMPUTE_DTYPE = jnp.float32
        cfg = YAML(typ="safe").load(
            (Path(dreamerv3.__file__).parent / "configs.yaml").read_text()
        )["defaults"]["agent"]
        cfg = elements.Config(cfg).update(
            {
                "dyn.rssm.deter": 8,
                "dyn.rssm.hidden": 8,
                "dyn.rssm.stoch": 2,
                "dyn.rssm.classes": 3,
                "dyn.rssm.blocks": 2,
                "enc.simple.units": 8,
                "dec.simple.units": 8,
                "enc.simple.layers": 1,
                "dec.simple.layers": 1,
                "rewhead.units": 8,
                "conhead.units": 8,
                "policy.units": 8,
                "policy.layers": 1,
                "value.units": 8,
                "value.layers": 1,
                "opt.warmup": 0,
                "repval_grad": False,
                "reward_grad": True,
            }
        )
        cfg = elements.Config(**cfg, replay_context=0)
        obs = {
            "vector": elements.Space(np.float32, (3,)),
            "reward": elements.Space(np.float32),
            **{k: elements.Space(bool) for k in ("is_first", "is_last", "is_terminal")},
        }
        act = {"action": elements.Space(np.float32, (2,), -1, 1)}
        cls = install("imf")
        model = object.__new__(cls)
        model.__init__(obs, act, cfg)
        data = {
            k: jnp.zeros((1, 8, *s.shape), s.dtype) for k, s in dict(obs, **act).items()
        }
        data["vector"] = jnp.arange(24, dtype=jnp.float32).reshape(1, 8, 3) / 24
        data["reward"] = jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0]])
        data["is_first"] = data["is_first"].at[:, 0].set(True)
        data["stepid"] = jnp.zeros((1, 8, 20), jnp.uint8)
        initial = model.init_train(1)
        state = nj.init(model.train)({}, initial, data, seed=0)
        for k, v in CONTROLS.items():
            state[f"staged_{k}/value"] = jnp.asarray(v)
        before = {k: np.array(v) for k, v in state.items()}
        train = jax.jit(nj.pure(model.train))
        carry = initial
        for seed in range(3):
            state, (carry, _, _) = train(state, carry, data, seed=seed + 1)
        for key, value in state.items():
            self.assertTrue(np.isfinite(value).all(), key)
        for group in ("representation", "transition", "actor"):
            self.assertEqual(int(state[f"opt/state/{group}/3/count"]), 3)
        for prefix in ("enc/", "dyn/priornormal/", "dyn/imf", "rew/", "pol/", "val/"):
            self.assertTrue(
                any(
                    not np.array_equal(before[k], state[k])
                    for k in before
                    if k.startswith(prefix)
                ),
                prefix,
            )
        prev = {"action": jnp.zeros_like(data["action"])}
        observations = {k: data[k] for k in obs}

        def losses():
            return model.loss(
                (model.enc.initial(1), model.dyn.initial(1), model.dec.initial(1)),
                observations,
                prev,
                True,
            )[1][2]["losses"]

        selected = {
            k: v
            for k, v in state.items()
            if k.startswith(("enc/", "dyn/", "rew/", "val/"))
        }

        def objective(p, name):
            return nj.pure(losses)(dict(state, **p), seed=51)[1][name].mean()

        gradients = {
            name: jax.jit(jax.grad(lambda p: objective(p, name)))(selected)
            for name in ("repval", "rew", "aux_dyn")
        }
        for key, g in gradients["repval"].items():
            if key.startswith(("enc/", "dyn/", "rew/")):
                np.testing.assert_array_equal(g, 0, err_msg=key)
        norm = lambda name, prefix: sum(
            float(jnp.sum(g * g))
            for k, g in gradients[name].items()
            if k.startswith(prefix)
        )
        self.assertGreater(norm("repval", "val/"), 0)
        self.assertGreater(norm("rew", "enc/"), 0)
        self.assertGreater(norm("rew", "dyn/obs"), 0)
        # The auxiliary distribution must never replace the implicit sampler.
        with mock.patch.object(
            type(model.dyn),
            "_prior_normal",
            side_effect=AssertionError("auxiliary sampler used"),
        ):
            generated = nj.pure(model.dyn._sample_prior)(
                state, jnp.ones((2, 8)), seed=123
            )[1]
        self.assertEqual(generated.shape, (2, 2, 3))
        # Both diagnostic sampling modes execute; four-step must not silently mean one.
        values = [{k: v for k, v in data.items() if k in set(obs) | set(act)}][0]
        one = _observable_arrays(model, state, values, 17, 1)
        four = _observable_arrays(model, state, values, 17, 4)
        self.assertEqual(int(four["model"]["positive_paths"]), 2)
        self.assertFalse(
            np.isclose(
                float(one["model"]["observation_mse"]),
                float(four["model"]["observation_mse"]),
                rtol=1e-7,
                atol=1e-9,
            )
        )


if __name__ == "__main__":
    unittest.main()
