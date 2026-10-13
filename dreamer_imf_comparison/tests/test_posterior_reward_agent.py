"""No-simulator online-adapter gates, including adverse teacher controls."""

import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

if importlib.util.find_spec("dreamerv3") is None:
    upstream_path = (
        Path(__file__).resolve().parents[3]
        / "dreamer_imf_comparison/external/dreamerv3"
    )
    if (upstream_path / "dreamerv3").is_dir():
        sys.path.insert(0, str(upstream_path))

import dreamerv3
import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from ruamel.yaml import YAML

from dreamer_imf_compare.conditional_dynamics import ConditionalAgent
from dreamer_imf_compare.conditional_schedule import CONTROLS
from dreamer_imf_compare.posterior_reward_agent import (
    ALPHA,
    Float32RewardHead,
    LAYERS,
    RewardCategorical,
    distillation_targets,
    import_rmt_heads,
    install,
    posterior_mean_features,
    readout_data_loss,
    reward_mask,
    weight_penalty,
)
from dreamer_imf_compare.reward_rmt_fit import init as rmt_init
from dreamer_imf_compare.reward_readout_study import mlp_logits as reference_logits


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
    data["reward"] = jnp.array([[0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 2.0, 1.0]])
    data["is_first"] = data["is_first"].at[:, 0].set(True)
    data["is_terminal"] = data["is_terminal"].at[:, -1].set(True)
    data["stepid"] = jnp.zeros((2, 4, 20), jnp.uint8)
    return cfg, obs, act, data


def model_for(arm, cfg, obs, act):
    cls = install(arm)
    model = object.__new__(cls)
    model.__init__(obs, act, cfg)
    return model


def exact_tree(test, a, b):
    test.assertEqual(jax.tree.structure(a), jax.tree.structure(b))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


class PosteriorRewardUnitTests(unittest.TestCase):
    def test_float32_head_exact_reference_and_frozen_normalization(self):
        head = Float32RewardHead(name="reward_student")
        x = jnp.arange(28, dtype=jnp.float32).reshape(2, 14) / 31
        state = nj.init(lambda x: head(x, 1).logits)({}, x, seed=0)
        params = rmt_init(jax.random.PRNGKey(13), 14)
        state["reward_student/input_mean"] = jnp.linspace(-1, 1, 14)
        state["reward_student/input_std"] = jnp.linspace(0.01, 2, 14)
        for name, layer in zip(LAYERS, params):
            state[f"reward_student/{name}_kernel"] = layer["w"]
            state[f"reward_student/{name}_bias"] = layer["b"]
        old = nn.COMPUTE_DTYPE
        try:
            nn.COMPUTE_DTYPE = jnp.bfloat16
            actual = nj.pure(lambda x: head(x, 1).logits)(state, x, seed=2)[1]
        finally:
            nn.COMPUTE_DTYPE = old
        expected = reference_logits(
            params,
            (x - state["reward_student/input_mean"])
            / state["reward_student/input_std"],
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, jnp.float32)
        fn = nj.pure(lambda x: head(x, 1).loss(jnp.array([1.0, 2.0])).mean())
        np.testing.assert_array_equal(jax.grad(lambda x: fn(state, x, seed=0)[1])(x), 0)
        grad = jax.grad(lambda p: fn(p, x, seed=0)[1])(state)
        np.testing.assert_array_equal(grad["reward_student/input_mean"], 0)
        np.testing.assert_array_equal(grad["reward_student/input_std"], 0)
        self.assertGreater(
            float(jnp.linalg.norm(grad["reward_student/logits_bias"])), 0
        )

    def test_penalty_bias_exclusion_and_gradient(self):
        params = rmt_init(jax.random.PRNGKey(1), 4)
        grad = jax.grad(weight_penalty)(params)
        for p, g in zip(params, grad):
            np.testing.assert_array_equal(g["b"], 0)
            np.testing.assert_allclose(g["w"], ALPHA * p["w"], atol=0, rtol=0)
        changed = tuple(dict(p, b=p["b"] + 1000) for p in params)
        self.assertEqual(float(weight_penalty(params)), float(weight_penalty(changed)))

    def test_incoming_reward_mask_retains_terminal_and_rejects_unsupported(self):
        mask = reward_mask(jnp.array([[True, False, False, True, False]]))
        np.testing.assert_array_equal(mask, [[False, True, True, False, True]])
        logits = jnp.zeros((1, 5, 3))
        params = ({"w": jnp.zeros((1, 3)), "b": jnp.zeros(3)},)
        # Invalid reset rewards are masked; valid terminal reward 2 is retained.
        reward = jnp.array([[99.0, 1.0, 2.0, 99.0, 0.0]])
        loss = readout_data_loss(params, jnp.zeros((1, 5, 1)), reward, mask)
        np.testing.assert_allclose(loss, np.log(3), rtol=1e-6)
        self.assertTrue(np.isnan(RewardCategorical(logits).loss(reward)[0, 0]))
        self.assertTrue(
            np.isnan(
                readout_data_loss(
                    params, jnp.zeros((1, 5, 1)), reward.at[0, 2].set(3), mask
                )
            )
        )

    def test_zero_lambda_is_hard_label_and_teacher_targets_stopped(self):
        params = rmt_init(jax.random.PRNGKey(2), 4)
        x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
        reward, valid = jnp.array([0.0, 1.0, 2.0, 0.0]), jnp.ones(4, bool)
        teacher = jax.nn.softmax(jnp.arange(12, dtype=jnp.float32).reshape(4, 3))
        hard = readout_data_loss(params, x, reward, valid)
        soft = readout_data_loss(
            params, x, reward, valid, distillation_targets(reward, teacher, 0.0)
        )
        np.testing.assert_allclose(hard, soft, atol=1e-6, rtol=1e-6)
        derivative = jax.grad(
            lambda t: readout_data_loss(
                params, x, reward, valid, distillation_targets(reward, t, 0.5)
            )
        )(teacher)
        np.testing.assert_array_equal(derivative, 0)
        with self.assertRaises(ValueError):
            distillation_targets(reward, teacher, 1.0)

    def test_synthetic_informative_uninformative_and_unavailable_information(self):
        # The exact categorical optimum is the conditional mean target. An
        # informative privileged teacher helps under corrupted hard labels;
        # a wrong teacher can hurt even when hard labels are already correct.
        true = jnp.array([[0.9, 0.1, 0.0], [0.0, 0.1, 0.9]])
        corrupted = jnp.array([[0.6, 0.1, 0.3], [0.3, 0.1, 0.6]])
        mixed = 0.5 * (true + corrupted)
        expected_ce = lambda p: -jnp.sum(
            true * jnp.log(jnp.maximum(p, 1e-9)), -1
        ).mean()
        self.assertLess(float(expected_ce(mixed)), float(expected_ce(corrupted)))
        self.assertGreater(
            float(expected_ce(0.5 * (true + 1 / 3))), float(expected_ce(true))
        )
        # Identical student inputs cannot recover a reward revealed only to T.
        reward = jnp.array([0.0, 2.0])
        teacher = jax.nn.one_hot(reward.astype(int), 3)
        targets = distillation_targets(reward, teacher, 0.5)
        shared_optimum = targets.mean(0)
        np.testing.assert_array_equal(shared_optimum, [0.5, 0.0, 0.5])
        self.assertEqual(
            float(jnp.mean((shared_optimum @ jnp.arange(3) - reward) ** 2)), 1.0
        )

    def test_posterior_mean_layout_preserves_precision(self):
        h = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        mu = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3) / 7
        result = posterior_mean_features({"deter": h, "stoch": mu * 0}, {"mean": mu}, 1)
        np.testing.assert_array_equal(result, np.concatenate([h, mu.reshape(2, 6)], -1))

    def test_import_selected_rmt_heads_and_fail_closed(self):
        heads = [
            Float32RewardHead(name="reward_teacher"),
            Float32RewardHead(name="reward_student"),
        ]
        x = jnp.ones((2, 14))
        state = nj.init(lambda: tuple(h(x, 1).pred() for h in heads))({}, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write = lambda p, v: p.write_text(json.dumps(v))
            chosen = {}
            parameters = rmt_init(jax.random.PRNGKey(99), 14)
            for group in ("hmean", "hz"):
                record = {"update": 10, "validation": {"nll": 0.4}}
                chosen[group] = {"alpha": 0.01, "selections": {"1": record}}
                directory = root / "heads" / f"reward-{group}-a0.01-s1"
                directory.mkdir(parents=True)
                write(
                    directory / "selection.json",
                    dict(
                        seed=1,
                        alpha=0.01,
                        width=128,
                        depth=2,
                        classes=3,
                        selected=record,
                    ),
                )
                (directory / "checkpoint-00010.pkl").write_bytes(
                    pickle.dumps(parameters)
                )
                np.savez(
                    directory / "normalization.npz", mean=np.zeros(14), std=np.ones(14)
                )
            write(root / "selections.json", chosen)
            write(root / "report.json", {})
            write(
                root / "evaluated.json",
                dict(
                    stage="evaluated",
                    artifacts={
                        str(p.relative_to(root)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in root.rglob("*")
                        if p.is_file()
                    },
                ),
            )
            write(
                root / "verified.json",
                dict(
                    exact_replay=True,
                    **{
                        f"{k}_sha256": hashlib.sha256(
                            (root / f"{k}.json").read_bytes()
                        ).hexdigest()
                        for k in ("report", "evaluated")
                    },
                ),
            )
            exported = import_rmt_heads(root, 1, state)
            self.assertEqual(set(exported), set(state))
            for head in heads:
                actual = nj.pure(lambda: head(x, 1).logits)(exported, seed=0)[1]
                np.testing.assert_array_equal(actual, reference_logits(parameters, x))
            with self.assertRaises(ValueError):
                import_rmt_heads(root, 3, state)
            write(root / "report.json", {"tampered": True})
            with self.assertRaisesRegex(ValueError, "binding"):
                import_rmt_heads(root, 1, state)


class PosteriorRewardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.obs, cls.act, cls.data = fixture()
        cls.model = model_for("C", cls.cfg, cls.obs, cls.act)
        cls.initial = cls.model.init_train(2)
        cls.state = nj.init(cls.model.train)({}, cls.initial, cls.data, seed=0)
        for key, value in CONTROLS.items():
            cls.state[f"staged_{key}/value"] = jnp.asarray(value)

    def test_all_arms_have_same_auxiliary_and_parent_namespaces(self):
        for arm in ("A", "B", "C"):
            model = model_for(arm, self.cfg, self.obs, self.act)
            self.assertEqual(model.arm, arm)
            self.assertNotIn(model.reward_teacher, model.modules)
            self.assertNotIn(model.reward_student, model.modules)
        self.assertFalse(any(k.startswith("explore") for k in self.state))
        self.assertFalse(
            any(k.startswith("opt/") and "reward_teacher" in k for k in self.state)
        )
        with self.assertRaises(ValueError):
            install("D")

    def test_arm_a_parent_parameter_and_optimizer_update_exact(self):
        model = model_for("A", self.cfg, self.obs, self.act)
        parent = object.__new__(ConditionalAgent)
        parent.__init__(self.obs, self.act, self.cfg)
        actual, _ = nj.pure(model.train)(self.state, self.initial, self.data, seed=17)
        expected, _ = nj.pure(parent.train)(
            self.state, self.initial, self.data, seed=17
        )
        for key in self.state:
            if not key.startswith(
                ("reward_teacher/", "reward_student/", "reward_aux_opt/")
            ):
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_same_pass_statistics_no_added_observe_or_rng_and_parent_loss_equal(self):
        observations = {k: self.data[k] for k in self.obs}
        previous = {"action": jnp.zeros_like(self.data["action"])}
        model = model_for("A", self.cfg, self.obs, self.act)
        parent = object.__new__(ConditionalAgent)
        parent.__init__(self.obs, self.act, self.cfg)
        calls = []
        original = model.dyn.observe

        def observe(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        with mock.patch.object(model.dyn, "observe", side_effect=observe):
            _, actual = nj.pure(model.loss)(
                self.state,
                self.initial[:3],
                observations,
                previous,
                training=False,
                seed=13,
            )
        self.assertEqual(len(calls), 1)
        _, expected = nj.pure(parent.loss)(
            self.state,
            self.initial[:3],
            observations,
            previous,
            training=False,
            seed=13,
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        exact_tree(self, actual[1][0], expected[1][0])
        exact_tree(self, actual[1][2]["repfeat"], expected[1][2]["repfeat"])
        self.assertEqual(set(actual[1][2]["repfeat"]), {"deter", "stoch"})
        self.assertEqual(set(actual[1][2]["posterior"]), {"mean", "std"})
        self.assertEqual(set(actual[1][0][1]), {"deter", "stoch"})

    def test_compiled_updates_finite_once_parent_and_auxiliaries(self):
        state = dict(self.state)
        train = jax.jit(nj.pure(self.model.train))
        carry = self.initial
        for seed in (51, 52):
            state, (carry, _, metrics) = train(state, carry, self.data, seed=seed)
        for key, value in state.items():
            self.assertTrue(np.isfinite(value).all(), key)
        for key, value in metrics.items():
            self.assertTrue(np.isfinite(value).all(), key)
        self.assertEqual(int(state["reward_aux_opt/step/value"]), 2)
        self.assertEqual(int(state["opt/step/value"]), 2)
        for prefix in (
            "enc/",
            "dyn/imf",
            "rew/",
            "pol/",
            "val/",
            "reward_teacher/",
            "reward_student/",
        ):
            self.assertTrue(
                any(
                    not np.array_equal(state[k], self.state[k])
                    for k in self.state
                    if k.startswith(prefix)
                ),
                prefix,
            )
        for prefix in ("reward_teacher", "reward_student"):
            for suffix in ("input_mean", "input_std"):
                key = f"{prefix}/{suffix}"
                np.testing.assert_array_equal(state[key], self.state[key])
            self.assertEqual(int(metrics[f"{prefix}/valid_count"]), 6)
            self.assertGreater(float(metrics[f"{prefix}/update_parameter_ratio"]), 0)
        np.testing.assert_array_equal(carry[-1]["action"], self.data["action"][:, -1])

    def test_auxiliary_gradients_detached_original_reward_gradients_alive(self):
        state, _ = jax.jit(nj.pure(self.model.train))(
            self.state, self.initial, self.data, seed=41
        )
        observations = {k: self.data[k] for k in self.obs}
        previous = {"action": jnp.zeros_like(self.data["action"])}

        def objective(aux):
            _, (_, _, outputs, _) = self.model.loss(
                self.initial[:3], observations, previous, training=False
            )
            if not aux:
                return outputs["losses"]["rew"].mean()
            student = self.model.feat2tensor(outputs["repfeat"])
            teacher = posterior_mean_features(
                outputs["repfeat"], outputs["posterior"], 2
            )
            reward = observations["reward"]
            targets = distillation_targets(
                reward, self.model.reward_teacher(teacher, 2).probabilities(), 0.5
            )
            return self.model._auxiliary_loss(
                student, teacher, reward, reward_mask(observations["is_first"]), targets
            )[0]

        for aux in (False, True):
            parameters = {
                k: v
                for k, v in state.items()
                if k.startswith(("enc/", "dyn/", "reward_teacher/", "reward_student/"))
            }
            grad = jax.grad(
                lambda p: nj.pure(objective)(dict(state, **p), aux, seed=7)[1]
            )(parameters)
            norm = float(
                optax.global_norm(
                    {k: v for k, v in grad.items() if k.startswith(("enc/", "dyn/"))}
                )
            )
            if aux:
                self.assertEqual(norm, 0)
            else:
                self.assertGreater(norm, 0)

    def test_pre_update_teacher_targets_and_single_aux_optimizer_call(self):
        captured = {}
        original = self.model._auxiliary_loss

        def capture(student, teacher, reward, valid, targets):
            expected = distillation_targets(
                reward, self.model.reward_teacher(teacher, 2).probabilities(), 0.5
            )
            captured["difference"] = jnp.max(jnp.abs(expected - targets))
            return original(student, teacher, reward, valid, targets)

        with mock.patch.object(self.model, "_auxiliary_loss", side_effect=capture):
            state, _ = nj.pure(self.model.train)(
                self.state, self.initial, self.data, seed=15
            )
        self.assertEqual(float(captured["difference"]), 0)
        self.assertEqual(int(state["reward_aux_opt/step/value"]), 1)

    def test_imagination_student_only_and_policy_same_actions_as_parent(self):
        features = jnp.ones((2, 3, 14))
        for arm in ("A", "B", "C"):
            model = model_for(arm, self.cfg, self.obs, self.act)
            with mock.patch.object(
                model.reward_teacher,
                "__call__",
                side_effect=AssertionError("teacher leaked to imagination"),
            ):
                result = nj.pure(model.imagined_reward)(
                    self.state, features, 2, seed=3
                )[1]
            expected = nj.pure(
                lambda: (model.rew if arm == "A" else model.reward_student)(
                    features, 2
                ).pred()
            )(self.state, seed=3)[1]
            np.testing.assert_array_equal(result, expected)
        obs = {k: self.data[k][:, 1] for k in self.obs}
        parent = object.__new__(ConditionalAgent)
        parent.__init__(self.obs, self.act, self.cfg)
        for mode in ("train", "eval"):
            _, actual = nj.pure(self.model.policy)(
                self.state, self.initial, obs, mode=mode, seed=29
            )
            _, expected = nj.pure(parent.policy)(
                self.state, self.initial, obs, mode=mode, seed=29
            )
            exact_tree(self, actual[:2], expected[:2])
            self.assertEqual(set(actual[0][1]), {"deter", "stoch"})
            for key in ("student_probabilities", "teacher_probabilities"):
                np.testing.assert_allclose(
                    actual[2]["log/" + key].sum(-1), 1, atol=1e-6
                )
            self.assertNotIn("log/original_categorical_nll", actual[2])


if __name__ == "__main__":
    unittest.main()
