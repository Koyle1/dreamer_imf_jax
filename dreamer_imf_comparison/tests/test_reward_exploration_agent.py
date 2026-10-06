"""No-simulator structural, gradient, indexing, and compiled update checks."""

from pathlib import Path
import unittest
from unittest import mock

import dreamerv3
from dreamerv3 import agent as upstream
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
from ruamel.yaml import YAML

from dreamer_imf_compare.conditional_dynamics import ConditionalAgent
from dreamer_imf_compare.conditional_schedule import CONTROLS
from dreamer_imf_compare.reward_exploration_agent import (
    BONUS_CAP,
    ENSEMBLE_SIZE,
    RewardCategorical,
    action_tensor,
    build_control_head,
    build_ensemble,
    episode_bootstrap,
    install,
    predicted_mean_disagreement,
    transition_mask,
)


def fixture():
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
            "rewhead.layers": 1,
            "conhead.units": 8,
            "conhead.layers": 1,
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
        "vector": elements.Space(np.float32, (6,)),
        "reward": elements.Space(np.float32),
        **{k: elements.Space(bool) for k in ("is_first", "is_last", "is_terminal")},
    }
    act = {"action": elements.Space(np.float32, (2,), -1, 1)}
    data = {
        k: jnp.zeros((2, 4, *s.shape), s.dtype) for k, s in dict(obs, **act).items()
    }
    data["vector"] = jnp.arange(48, dtype=jnp.float32).reshape(2, 4, 6) / 48
    data["action"] = jnp.arange(16, dtype=jnp.float32).reshape(2, 4, 2) / 16
    data["reward"] = jnp.array([[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0]])
    data["is_first"] = data["is_first"].at[:, 0].set(True)
    data["episode_id"] = jnp.array([[11, 11, 11, 11], [17, 17, 17, 17]], jnp.int32)
    data["stepid"] = jnp.zeros((2, 4, 20), jnp.uint8)
    return cfg, obs, act, data


class RewardExplorationUnitTests(unittest.TestCase):
    def test_bootstrap_episode_stability_independence_and_negative_control(self):
        ids = jnp.repeat(jnp.arange(400, dtype=jnp.int32)[:, None], 3, 1)
        masks = np.asarray(episode_bootstrap(ids, 7))
        self.assertEqual(masks.shape, (ENSEMBLE_SIZE, 400, 3))
        np.testing.assert_array_equal(masks[:, :, 0], masks[:, :, 2])
        np.testing.assert_array_equal(episode_bootstrap(ids[::-1], 7), masks[:, ::-1])
        self.assertTrue(
            np.all((masks.mean((1, 2)) > 0.35) & (masks.mean((1, 2)) < 0.65))
        )
        self.assertFalse(np.array_equal(masks[0], masks[1]))
        self.assertFalse(np.array_equal(masks, episode_bootstrap(ids, 8)))

    def test_reset_terminal_and_action_alignment(self):
        first = jnp.array([[True, False, True, False, False]])
        last = jnp.array([[False, True, False, True, False]])
        ids = jnp.array([[4, 4, 5, 5, 6]], jnp.int32)
        np.testing.assert_array_equal(
            transition_mask(first, last, ids), [[True, False, True, False]]
        )
        actions = {"z": jnp.array([[[4.0, 5.0], [6.0, 7.0]]]), "a": jnp.array([[0, 2]])}
        spaces = {
            "z": elements.Space(np.float32, (2,)),
            "a": elements.Space(np.int32, (), 0, 3),
        }
        np.testing.assert_array_equal(
            action_tensor(actions, spaces), [[[1, 0, 0, 4, 5], [0, 0, 1, 6, 7]]]
        )

    def test_predicted_mean_disagreement_not_state_sample_variance(self):
        same = jnp.broadcast_to(jnp.array([[[3.0, -7.0], [100.0, -100.0]]]), (5, 2, 2))
        np.testing.assert_array_equal(predicted_mean_disagreement(same), 0)
        changed = same.at[0].add(5)
        np.testing.assert_allclose(predicted_mean_disagreement(changed), [4, 4])
        self.assertGreater(float(jnp.var(same)), 100)
        ensemble = build_ensemble(observation_dim=2)
        x, a = jnp.ones((3, 4)), jnp.zeros((3, 2))
        state = nj.init(ensemble.bonus)({}, x, a, seed=9)
        state["explore_ensemble/bonus_scale"] = jnp.array(1e-12)
        bonus = nj.pure(ensemble.bonus)(state, x, a, seed=2)[1]
        np.testing.assert_allclose(bonus, BONUS_CAP)

    def test_categorical_expectation_loss_and_unsupported_label(self):
        output = RewardCategorical(jnp.log(jnp.array([[0.2, 0.3, 0.5]])))
        np.testing.assert_allclose(output.pred(), [1.3], rtol=1e-6)
        np.testing.assert_allclose(
            output.loss(jnp.array([2.0])), [-np.log(0.5)], rtol=1e-6
        )
        self.assertTrue(np.isnan(output.loss(jnp.array([3.0]))).all())

    def test_offline_head_namespace_original_architecture_and_fixed_norms(self):
        cfg, _, _, _ = fixture()
        x = jnp.arange(28, dtype=jnp.float32).reshape(2, 14) / 28
        old = embodied.jax.MLPHead(
            elements.Space(np.float32), **cfg.rewhead, name="rew"
        )
        new = build_control_head("existing", cfg)
        original = nj.init(lambda: old(x, 1).pred())({}, seed=3)
        duplicate = {
            k.replace("rew/", "control_rew/", 1): v for k, v in original.items()
        }
        np.testing.assert_array_equal(
            nj.pure(lambda: old(x, 1).pred())(original, seed=0)[1],
            nj.pure(lambda: new(x, 1).pred())(duplicate, seed=0)[1],
        )
        categorical = build_control_head("categorical", cfg)
        fn = lambda x: categorical(x, 1).loss(jnp.array([0.0, 2.0])).mean()
        state = nj.init(fn)({}, x, seed=2)
        self.assertEqual(state["control_rew/hidden0/kernel"].shape, (14, 128))
        self.assertEqual(state["control_rew/hidden1/kernel"].shape, (128, 128))
        self.assertEqual(state["control_rew/logits/kernel"].shape, (128, 3))
        gradient = jax.grad(lambda x: nj.pure(fn)(state, x, seed=1)[1])(x)
        np.testing.assert_array_equal(gradient, 0)
        gradients = jax.grad(lambda p: nj.pure(fn)(p, x, seed=1)[1])(state)
        np.testing.assert_array_equal(gradients["control_rew/input_mean"], 0)
        np.testing.assert_array_equal(gradients["control_rew/input_std"], 0)
        self.assertGreater(
            float(jnp.linalg.norm(gradients["control_rew/logits/kernel"])), 0
        )

    def test_factory_all_arms_share_exploration_implementation(self):
        self.assertEqual(install("A").control_kind, "existing")
        for arm in ("B", "C", "D"):
            cls = install(arm)
            self.assertEqual(cls.control_kind, "categorical")
            self.assertTrue(issubclass(cls, ConditionalAgent))
            self.assertIs(cls.train, install("A").train)
        with self.assertRaises(ValueError):
            install("random")


class RewardExplorationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.obs, cls.act, cls.data = fixture()
        impl = install("D")
        cls.model = object.__new__(impl)
        cls.model.__init__(cls.obs, cls.act, cls.cfg)
        cls.initial = cls.model.init_train(2)
        cls.state = nj.init(cls.model.train)({}, cls.initial, cls.data, seed=0)
        for key, value in CONTROLS.items():
            cls.state[f"staged_{key}/value"] = jnp.asarray(value)

    def test_full_compiled_updates_finite_all_optimizers_and_frozen_normalizers(self):
        model, data, state = self.model, self.data, dict(self.state)
        before = {k: np.array(v) for k, v in state.items()}
        train = jax.jit(nj.pure(model.train))
        carry = self.initial
        for seed in (31, 32):
            state, (carry, _, metrics) = train(state, carry, data, seed=seed)
        for key, value in state.items():
            self.assertTrue(np.isfinite(value).all(), key)
        for key, value in metrics.items():
            self.assertTrue(np.isfinite(value).all(), key)
        for prefix in ("opt", "control_opt", "ensemble_opt", "explore_opt"):
            self.assertEqual(int(state[f"{prefix}/step/value"]), 2, prefix)
        for prefix in (
            "enc/",
            "dyn/imf",
            "rew/",
            "pol/",
            "val/",
            "control_rew/",
            "explore_ensemble/member",
            "explore_pol/",
            "explore_val/",
        ):
            self.assertTrue(
                any(
                    not np.array_equal(before[k], state[k])
                    for k in before
                    if k.startswith(prefix)
                ),
                prefix,
            )
        for prefix in ("control_rew/", "explore_ensemble/"):
            for suffix in (
                "input_mean",
                "input_std",
                "target_mean",
                "target_std",
                "bonus_scale",
            ):
                key = prefix + suffix
                if key in state:
                    np.testing.assert_array_equal(state[key], before[key])
        self.assertEqual(float(metrics["ensemble/valid"]), 6)
        np.testing.assert_array_equal(carry[-1]["action"], data["action"][:, -1])
        self.assertNotIn(model.control_rew, model.modules)
        self.assertNotIn(model.ensemble, model.modules)
        self.assertNotIn(model.explore_pol, model.modules)
        self.assertFalse(
            any(
                k.startswith("opt/")
                and any(s in k for s in ("control_rew", "explore_", "ensemble"))
                for k in state
            )
        )
        self.assertEqual(model.ext_space["episode_id"].dtype, np.dtype(np.int32))

    def test_gradient_routing_preserves_original_reward_and_blocks_auxiliary_representation(
        self,
    ):
        model, state = self.model, self.state
        # Parent reward/value heads have zero output initialization: one real
        # update is required before their representation gradients can be nonzero.
        state, _ = jax.jit(nj.pure(model.train))(
            state, self.initial, self.data, seed=42
        )
        observations = {k: self.data[k] for k in self.obs}
        prev = {"action": jnp.zeros_like(self.data["action"])}
        carry = self.initial[:3]

        def objective(which):
            _, (_, entries, outs, _) = model.loss(
                carry, observations, prev, training=False
            )
            features = model.feat2tensor(outs["repfeat"])
            if which == "original":
                return outs["losses"]["rew"].mean()
            if which == "control":
                return model._control_loss(features, observations["reward"])[0]
            if which == "ensemble":
                return model._ensemble_loss(
                    features[:, :-1],
                    self.data["action"][:, :-1],
                    observations["vector"][:, 1:],
                    jnp.ones((2, 3), bool),
                    self.data["episode_id"][:, :-1],
                )[0]
            starts = model.dyn.starts(entries[1], carry[1], 4)
            first = jax.tree.map(
                lambda x: x.reshape((8, 1, *x.shape[2:])), outs["repfeat"]
            )
            return model._explore_loss(starts, first, False)[0]

        selected = {
            k: v
            for k, v in state.items()
            if k.startswith(
                (
                    "enc/",
                    "dyn/",
                    "rew/",
                    "control_rew/",
                    "explore_ensemble/",
                    "explore_pol/",
                    "explore_val/",
                )
            )
        }
        norms = {}
        for which in ("original", "control", "ensemble", "explore"):
            fn = nj.pure(lambda: objective(which))
            gradients = jax.jit(jax.grad(lambda p: fn(dict(state, **p), seed=12)[1]))(
                selected
            )
            norms[which] = {
                prefix: sum(
                    float(jnp.sum(g * g))
                    for k, g in gradients.items()
                    if k.startswith(prefix)
                )
                for prefix in (
                    "enc/",
                    "dyn/",
                    "rew/",
                    "control_rew/",
                    "explore_ensemble/",
                    "explore_pol/",
                    "explore_val/",
                )
            }
            self.assertTrue(
                all(np.isfinite(g).all() for g in gradients.values()), which
            )
        self.assertGreater(norms["original"]["enc/"], 0)
        self.assertGreater(norms["original"]["dyn/"], 0)
        self.assertGreater(norms["original"]["rew/"], 0)
        for which in ("control", "ensemble", "explore"):
            for prefix in ("enc/", "dyn/", "rew/"):
                self.assertEqual(norms[which][prefix], 0, (which, prefix))
        self.assertGreater(norms["control"]["control_rew/"], 0)
        self.assertGreater(norms["ensemble"]["explore_ensemble/"], 0)
        self.assertEqual(norms["explore"]["explore_ensemble/"], 0)
        self.assertGreater(norms["explore"]["explore_pol/"], 0)
        self.assertGreater(norms["explore"]["explore_val/"], 0)

    def test_collection_policy_selected_action_is_carry_and_eval_uses_task(self):
        model = self.model
        obs = {k: self.data[k][:, 0] for k in self.obs}
        outputs = {}
        for mode in ("train", "eval", "explore"):
            _, (carry, act, out) = nj.pure(model.policy)(
                self.state, self.initial, obs, mode, seed=1
            )
            np.testing.assert_array_equal(carry[-1]["action"], act["action"])
            self.assertEqual(out["log/control_reward"].shape, (2,))
            self.assertTrue(np.isfinite(out["log/disagreement"]).all())
            outputs[mode] = act["action"]
        np.testing.assert_array_equal(outputs["train"], outputs["eval"])
        self.assertFalse(np.array_equal(outputs["train"], outputs["explore"]))

    def test_existing_arm_compiles_and_parent_checkpoint_names_are_unchanged(self):
        parent = object.__new__(ConditionalAgent)
        parent.__init__(self.obs, self.act, self.cfg)
        parent_state = nj.init(parent.train)({}, self.initial, self.data, seed=0)
        for key, value in parent_state.items():
            self.assertIn(key, self.state)
            self.assertEqual(value.shape, self.state[key].shape, key)
            self.assertEqual(value.dtype, self.state[key].dtype, key)
        impl = install("A")
        model = object.__new__(impl)
        model.__init__(self.obs, self.act, self.cfg)
        state = nj.init(model.train)({}, self.initial, self.data, seed=0)
        for key, value in CONTROLS.items():
            state[f"staged_{key}/value"] = jnp.asarray(value)
        # A parent restore is a plain exact-name overwrite; no remapping or
        # initialization of the old optimizer state is necessary.
        state.update(parent_state)
        for key, value in CONTROLS.items():
            state[f"staged_{key}/value"] = jnp.asarray(value)
        result, (_, _, metrics) = jax.jit(nj.pure(model.train))(
            state, self.initial, self.data, seed=10
        )
        self.assertTrue(all(np.isfinite(v).all() for v in result.values()))
        for prefix in ("opt", "control_opt", "ensemble_opt", "explore_opt"):
            self.assertEqual(int(metrics[f"{prefix}/updates"]), 1)
        prefixes = (
            "explore_pol/",
            "explore_val/",
            "explore_slowval/",
            "explore_ensemble/",
            "explore_opt/",
            "ensemble_opt/",
        )
        self.assertEqual(
            {k: v.shape for k, v in self.state.items() if k.startswith(prefixes)},
            {k: v.shape for k, v in state.items() if k.startswith(prefixes)},
        )

    def test_imagined_bonus_is_successor_reward_not_predecessor_reward(self):
        model = self.model
        observations = {k: self.data[k] for k in self.obs}
        prev = {"action": jnp.zeros_like(self.data["action"])}

        def features():
            return model.loss(self.initial[:3], observations, prev, False)[1]

        _, (carry, entries, outs, _) = nj.pure(features)(self.state, seed=1)
        starts = model.dyn.starts(entries[1], carry[1], 4)
        first = jax.tree.map(lambda x: x.reshape((8, 1, *x.shape[2:])), outs["repfeat"])
        bonuses = jnp.broadcast_to(jnp.arange(1, 16, dtype=jnp.float32), (8, 15))
        original = upstream.imag_loss
        captured = []

        def spy(actions, reward, *args, **kwargs):
            captured.append(reward)
            return original(actions, reward, *args, **kwargs)

        with mock.patch.object(
            type(model.ensemble), "bonus", return_value=bonuses
        ), mock.patch.object(upstream, "imag_loss", side_effect=spy):
            nj.pure(model._explore_loss)(self.state, starts, first, False, seed=2)
        self.assertTrue(captured)
        np.testing.assert_array_equal(captured[-1][:, 0], 0)
        np.testing.assert_array_equal(captured[-1][:, 1:], bonuses)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("REWARD_EXPLORATION_AGENT_VERIFIED")
