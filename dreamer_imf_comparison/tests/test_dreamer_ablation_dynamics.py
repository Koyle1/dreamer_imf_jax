"""Tests against the pinned upstream RSSM and actual Ninjax transformations."""

import unittest

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
from dreamerv3 import rssm
from dreamer_imf_compare import dreamer_ablation_dynamics as adapter


class DynamicsTest(unittest.TestCase):
    def setUp(self):
        nn.COMPUTE_DTYPE = jnp.float32
        self.tokens = jnp.ones((2, 3, 5))
        self.acts = {"action": jnp.ones((2, 3, 2))}
        self.reset = jnp.array([[True, False, False], [False, True, False]])

    def model(self, cls):
        return cls(
            {"action": elements.Space(np.float32, (2,))},
            name="dyn",
            deter=8,
            hidden=8,
            stoch=2,
            classes=3,
            blocks=2,
            imglayers=1,
            obslayers=1,
            dynlayers=1,
        )

    def initialize(self, model):
        fun = nj.pure(
            lambda: model.loss(
                model.initial(2), self.tokens, self.acts, self.reset, True
            )
        )
        params, result = fun({}, seed=7, create=True)
        return fun, params, result

    def finite(self, tree):
        for x in jax.tree.leaves(tree):
            self.assertTrue(np.isfinite(x).all())

    def test_install_baseline_and_invalid(self):
        original = rssm.RSSM
        try:
            rssm.RSSM = adapter._UPSTREAM_RSSM
            adapter.install("categorical")
            self.assertIs(rssm.RSSM, adapter._UPSTREAM_RSSM)
            with self.assertRaises(ValueError):
                adapter.install("bad")
            adapter.install("gaussian")
            self.assertIs(rssm.RSSM, adapter.GaussianRSSM)
            with self.assertRaises(RuntimeError):
                adapter.install("categorical")
            adapter.install("imf")
            self.assertIs(rssm.RSSM, adapter.IMFRSSM)
        finally:
            rssm.RSSM = original

    def test_shapes_jitted_loss_gradients_and_imagination(self):
        for cls in (adapter.GaussianRSSM, adapter.IMFRSSM):
            with self.subTest(arm=cls.__name__):
                model = self.model(cls)
                fun, params, result = self.initialize(model)
                carry, entries, losses, feat, metrics = result
                self.assertEqual(entries["stoch"].shape, (2, 3, 2, 3))
                self.assertEqual(carry["stoch"].shape, (2, 2, 3))
                self.assertEqual(model.entry_space["stoch"].shape, (2, 3))
                self.assertEqual(losses["dyn"].shape, (2, 3))
                self.assertEqual(losses["rep"].shape, (2, 3))

                def objective(p):
                    _, out = fun(p, seed=11)
                    return sum(v.mean() for v in out[2].values())

                value, grad = jax.jit(jax.value_and_grad(objective))(params)
                self.finite((value, grad))
                self.assertGreater(sum(float(jnp.sum(x * x)) for x in grad.values()), 0)
                prior_grad = [v for k, v in grad.items() if "prior" in k or "imf" in k]
                self.assertGreater(sum(float(jnp.sum(x * x)) for x in prior_grad), 0)
                imagination = nj.pure(lambda c: model.imagine(c, self.acts, 3, True))
                _, imagined = jax.jit(lambda p, c: imagination(p, c, seed=19))(
                    params, carry
                )
                self.assertEqual(imagined[1]["stoch"].shape, (2, 3, 2, 3))
                self.finite(imagined)
                policy = lambda c: {"action": jnp.tanh(c["deter"][..., :2])}
                rollout = nj.pure(lambda c: model.imagine(c, policy, 3, True))
                _, out = rollout(params, carry, seed=21)
                self.finite(out)
                self.assertEqual(out[2]["action"].shape, (2, 3, 2))

    def test_shared_initialization_representation_and_reset(self):
        records = []
        for cls in (adapter.GaussianRSSM, adapter.IMFRSSM):
            model = self.model(cls)
            _, params, result = self.initialize(model)
            records.append((params, result))
            observe = nj.pure(
                lambda c: model.observe(
                    c,
                    self.tokens[:, 0],
                    {"action": self.acts["action"][:, 0]},
                    jnp.ones(2, bool),
                    False,
                    single=True,
                )
            )
            zero = model.initial(2)
            dirty = jax.tree.map(lambda x: x + 91, zero)
            _, a = observe(params, zero, seed=23)
            _, b = observe(params, dirty, seed=23)
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                np.testing.assert_array_equal(x, y)
        p, q = records[0][0], records[1][0]
        common = set(p) & set(q)
        self.assertTrue(any("obsnormal" in k for k in common))
        self.assertTrue(any("dyngru" in k for k in common))
        for key in common:
            np.testing.assert_array_equal(p[key], q[key], err_msg=key)
        for name in ("stoch", "deter"):
            np.testing.assert_array_equal(
                records[0][1][3][name], records[1][1][3][name]
            )
        np.testing.assert_array_equal(records[0][1][2]["rep"], records[1][1][2]["rep"])

    def test_detached_dynamics_targets_and_condition_gradients(self):
        for cls in (adapter.GaussianRSSM, adapter.IMFRSSM):
            model = self.model(cls)
            _, params, _ = self.initialize(model)
            fun = nj.pure(model._dynamics_loss)
            condition = jnp.ones((2, 3, 8))
            target = jnp.ones((2, 3, 2, 3))
            objective = lambda c, t: fun(params, c, t, seed=5)[1].mean()
            gc, gt = jax.jit(jax.grad(objective, (0, 1)))(condition, target)
            self.finite((gc, gt))
            np.testing.assert_array_equal(gt, jnp.zeros_like(gt))
            self.assertGreater(float(jnp.linalg.norm(gc)), 0)

    def test_normal_kl_reference(self):
        zero = jnp.zeros((2, 2, 3))
        np.testing.assert_allclose(adapter.normal_kl(zero, jnp.ones_like(zero)), 0)
        np.testing.assert_allclose(
            adapter.normal_kl(jnp.ones_like(zero), jnp.ones_like(zero)), 3
        )

    def test_upstream_bfloat16_compute(self):
        nn.COMPUTE_DTYPE = jnp.bfloat16
        try:
            for cls in (adapter.GaussianRSSM, adapter.IMFRSSM):
                model = self.model(cls)
                fun, params, result = self.initialize(model)
                self.assertEqual(result[0]["stoch"].dtype, jnp.bfloat16)
                objective = lambda p: sum(
                    x.mean() for x in fun(p, seed=2)[1][2].values()
                )
                self.finite(jax.jit(jax.value_and_grad(objective))(params))
                rollout = nj.pure(lambda c: model.imagine(c, self.acts, 3, True))
                _, imagined = rollout(params, result[0], seed=4)
                self.finite(imagined)
        finally:
            nn.COMPUTE_DTYPE = jnp.float32


if __name__ == "__main__":
    unittest.main()
