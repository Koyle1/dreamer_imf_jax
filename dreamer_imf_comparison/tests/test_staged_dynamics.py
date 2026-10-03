"""Behavioral checks for exact frozen groups and trace-safe phase changes."""

import unittest
from unittest import mock
from pathlib import Path

import elements
import embodied.jax.nets as nn
import dreamerv3
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
import ruamel.yaml

from dreamer_imf_compare.staged_dynamics import (
    grouped_optimizer,
    parameter_group,
    validate_controls,
    install,
)
from imf_dreamer_jax.imf import init_imf, sample_imf_steps, sample_time_pairs
from dreamerv3 import agent as upstream


class StagedDynamicsTests(unittest.TestCase):
    def test_complete_upstream_train_phase_switches(self):
        nn.COMPUTE_DTYPE = jnp.float32
        cfg = ruamel.yaml.YAML(typ="safe").load(
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
                "opt.wd": 0.01,
            }
        )
        cfg = elements.Config(**cfg, replay_context=0)
        obs = {
            "vector": elements.Space(np.float32, (3,)),
            "reward": elements.Space(np.float32),
            **{k: elements.Space(bool) for k in ("is_first", "is_last", "is_terminal")},
        }
        acts = {"action": elements.Space(np.float32, (2,), -1, 1)}
        for arm in ("gaussian", "imf"):
            cls = install(arm)
            model = object.__new__(cls)
            model.__init__(obs, acts, cfg)
            data = {
                k: jnp.zeros((1, 3, *s.shape), s.dtype)
                for k, s in dict(obs, **acts).items()
            }
            data["stepid"] = jnp.zeros((1, 3, 20), jnp.uint8)
            data["vector"] = jnp.ones((1, 3, 3))
            initial = model.init_train(1)
            state = nj.init(model.train)({}, initial, data, seed=0)
            train = jax.jit(nj.pure(model.train))
            old = {k: np.array(v) for k, v in state.items()}
            state, (carry, _, _) = train(state, initial, data, seed=1)
            for key, value in state.items():
                self.assertTrue(np.isfinite(value).all(), key)
            for key in old:
                if key.startswith(("pol/", "val/", "slowval/", "retnorm/")):
                    np.testing.assert_array_equal(old[key], state[key], err_msg=key)
            state = dict(state)
            state["staged_actor_enabled/value"] = jnp.array(True)
            state["staged_transition_only/value"] = jnp.array(True)
            state["staged_imag_horizon/value"] = jnp.array(15)
            state["staged_sample_steps/value"] = jnp.array(1)
            old = {k: np.array(v) for k, v in state.items()}
            state, _ = train(state, carry, data, seed=2)
            for key, value in state.items():
                self.assertTrue(np.isfinite(value).all(), key)
            for key in old:
                if (
                    key.startswith(("enc/", "dyn/", "dec/", "rew/", "con/"))
                    and parameter_group(key) == "representation"
                ):
                    np.testing.assert_array_equal(old[key], state[key], err_msg=key)
            self.assertTrue(
                any(
                    not np.array_equal(old[k], state[k])
                    for k in old
                    if k.startswith("pol/")
                )
            )

    def test_zero_variance_logging_metric_cannot_poison_training(self):
        original = upstream.imag_loss

        def with_zero_variance_metric(*args, **kwargs):
            losses, outs, metrics = original(*args, **kwargs)
            reward = args[1]
            # Exactly zero variance on every device, with a genuine dependency
            # on reward-head parameters. Its derivative is undefined at zero;
            # logging this finite value must never enter the loss backward pass.
            metrics["adv_std"] = jnp.stack([reward, reward], -1).std(-1).mean()
            return losses, outs, metrics

        with mock.patch.object(upstream, "imag_loss", with_zero_variance_metric):
            self.test_complete_upstream_train_phase_switches()

    def test_controls_fail_closed(self):
        for update in (
            {"imag_horizon": 6},
            {"sample_steps": 2},
            {"max_gap": 0},
            {"dyn_scale": float("nan")},
            {"actor_enabled": 1},
            {"extra": True},
        ):
            with self.assertRaises(ValueError):
                validate_controls(update)

    def test_groups(self):
        expected = {
            "dyn/imf0_weight": "transition",
            "dyn/prior0/kernel": "transition",
            "dyn/obsnormal/kernel": "representation",
            "dyn/gru/kernel": "representation",
            "enc/layer/kernel": "representation",
            "pol/layer/kernel": "actor",
        }
        for key, group in expected.items():
            self.assertEqual(parameter_group(key), group)

    def test_momentum_weight_decay_and_clocks_freeze_exactly(self):
        params = {
            "enc/w": jnp.array([1.0, 2.0]),
            "dyn/imf0_weight": jnp.array([2.0, 3.0]),
            "pol/w": jnp.array([3.0, 4.0]),
        }

        # Actual Ninjax state, persistent across JIT calls. A later phase must
        # change masks without retracing or stale Python schedule constants.
        class Learner(nj.Module):
            def __call__(self, actor, transition_only):
                parameter_tree = self.sub("params", nj.Tree, lambda: params)
                p = parameter_tree.read()
                enabled = lambda g: (
                    actor
                    if g == "actor"
                    else (
                        ~transition_only if g == "representation" else jnp.array(True)
                    )
                )
                tx = grouped_optimizer(optax.adamw(0.01, weight_decay=0.3), enabled)
                state_tree = self.sub("state", nj.Tree, tx.init, p)
                state = state_tree.read()
                updates, newstate = tx.update(jax.tree.map(jnp.ones_like, p), state, p)
                newparams = optax.apply_updates(p, updates)
                parameter_tree.write(newparams)
                state_tree.write(newstate)
                return newparams, newstate

        learner = Learner(name="test")
        fn = jax.jit(nj.pure(learner))
        state = nj.init(learner)({}, jnp.array(True), jnp.array(False), seed=0)
        state, (p1, s1) = fn(state, jnp.array(True), jnp.array(False), seed=0)
        state, (p2, s2) = fn(state, jnp.array(False), jnp.array(True), seed=1)
        for group, key in (("actor", "pol/w"), ("representation", "enc/w")):
            np.testing.assert_array_equal(p1[key], p2[key])
            for a, b in zip(jax.tree.leaves(s1[group]), jax.tree.leaves(s2[group])):
                np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(p1["dyn/imf0_weight"], p2["dyn/imf0_weight"]))
        state, (p3, _) = fn(state, jnp.array(True), jnp.array(False), seed=2)
        self.assertFalse(np.array_equal(p2["pol/w"], p3["pol/w"]))

    def test_gap_curriculum_preserves_boundary(self):
        r, t = sample_time_pairs(jax.random.PRNGKey(1), 100)
        clipped = t - jnp.minimum(t - r, 0.1)
        self.assertTrue(np.all(np.asarray(t - clipped) <= 0.100001))
        np.testing.assert_array_equal(clipped[:50], t[:50])
        np.testing.assert_allclose(t - jnp.minimum(t - r, 1.0), r, atol=1e-7)

    def test_one_and_four_step_sampling_finite(self):
        params = init_imf(jax.random.PRNGKey(0), 4, 3, hidden_dim=8, depth=1)
        condition = jnp.ones((2, 3))
        for steps in (1, 4):
            out = sample_imf_steps(
                params, condition, jax.random.PRNGKey(1), steps=steps
            )
            self.assertEqual(out.shape, (2, 4))
            self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
