"""Retained-history fixtures and real JAX/Ninjax inference, without physics."""

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamer_imf_compare import reward_rmt_posterior as posterior
from dreamer_imf_compare.parallel_collection import _history_arrays
from dreamer_imf_compare.parallel_frozen import FrozenModel, parameter_digest
from dreamer_imf_compare.reward_diagnosis_data import flatten


class FakeTeacher:
    arm, deter, feature_dim, action_dim = "imf", 2, 4, 2

    def __init__(self):
        self.weights = np.zeros(2, np.float32)
        self.before = self.frozen_digest()
        self.calls = []

    def frozen_digest(self):
        return hashlib.sha256(self.weights.tobytes()).hexdigest()

    def initial(self):
        return dict(h=np.zeros(2, np.float32), z=np.zeros(2, np.float32))

    def observe(self, carry, obs, previous_action, seed):
        self.calls.append(
            (int(seed), np.asarray(previous_action).copy(), bool(obs["is_first"]))
        )
        if bool(obs["is_first"]):
            carry["h"] *= 0
            carry["z"] *= 0
        carry["h"] = carry["h"] + carry["z"] + previous_action
        carry["z"] = np.asarray(obs["position"], np.float32) + np.float32(seed % 7)
        feature = np.r_[carry["h"], carry["z"]]
        # The wrapper must protect evidence even from in-place host adapters.
        obs["position"][...] = -999
        previous_action[...] = -999
        return carry, feature


def fake_stats(teacher, carry, obs, action, seed):
    mean = np.asarray(obs["position"], np.float32).copy()
    state, feature = teacher.observe(carry, obs, action, seed)
    return state, dict(
        h=feature[:2], mean=mean, std=np.ones(2, np.float32), z=feature[2:]
    )


def observation(e, t, reset=False):
    return dict(
        position=np.array([e + t, t], np.float32),
        to_target=np.array([0.2, 0.3], np.float32),
        velocity=np.zeros(2, np.float32),
        reward=np.asarray(t % 3, np.float32),
        is_first=np.asarray(reset),
        is_last=np.asarray(False),
        is_terminal=np.asarray(False),
    )


def make_history(teacher, raw, actions, seeds, previous, start=None, feature=None):
    if start is None:
        carry, feature = teacher.observe(
            teacher.initial(), copy.deepcopy(raw[0]), previous.copy(), seeds[0]
        )
    else:
        carry = copy.deepcopy(start)
    features = [feature.copy()]
    carries = [copy.deepcopy(carry)]
    for t, action in enumerate(actions):
        carry, feature = teacher.observe(
            carry, copy.deepcopy(raw[t + 1]), action.copy(), seeds[t + 1]
        )
        features.append(feature)
        carries.append(copy.deepcopy(carry))
    history = _history_arrays(
        raw,
        np.stack([flatten(x) for x in raw]),
        actions,
        [x["reward"] for x in raw[1:]],
        features,
        seeds,
        np.zeros(len(actions), bool),
        previous,
    )
    return history, carries


def fixture(root):
    teacher, rows = FakeTeacher(), []
    for episode in range(3):
        raw = [observation(episode, t, t == 0) for t in range(5)]
        actions = np.tile([0.1, -0.2], (4, 1)).astype(np.float32)
        seeds = np.arange(episode * 100, episode * 100 + 5, dtype=np.uint32)
        history, carries = make_history(
            teacher, raw, actions, seeds, np.zeros(2, np.float32)
        )
        np.savez(root / f"episode-{episode:03d}.npz", **history)
        for anchor in (2, 4):
            for plan in (0, 1):
                braw = [copy.deepcopy(raw[anchor])] + [
                    observation(episode, anchor + step + plan) for step in range(1, 16)
                ]
                ba = np.tile([0.3 * plan, -0.1], (15, 1)).astype(np.float32)
                bs = np.r_[
                    seeds[anchor],
                    np.arange(
                        1000 + 100 * episode + 20 * anchor,
                        1015 + 100 * episode + 20 * anchor,
                    ),
                ].astype(np.uint32)
                branch, _ = make_history(
                    teacher,
                    braw,
                    ba,
                    bs,
                    actions[anchor - 1],
                    carries[anchor],
                    history["features"][anchor],
                )
                np.savez(
                    root / f"branch-{episode:03d}-{anchor:03d}-{plan}.npz", **branch
                )
                rows.append(
                    dict(
                        episode=episode,
                        anchor=anchor,
                        plan=plan,
                        split=episode,
                        start=branch["features"][0],
                        targets=branch["features"][1:],
                        observations=branch["observations"][1:],
                        actions=ba,
                        rewards=branch["rewards"],
                    )
                )
    teacher.calls.clear()
    return teacher, {k: np.asarray([r[k] for r in rows]) for k in rows[0]}


def tiny_jax_teacher():
    """Use the real FrozenModel.observe with a tiny seed-consuming Ninjax body."""
    teacher = object.__new__(FrozenModel)
    teacher.arm, teacher.deter, teacher.stoch, teacher.classes = "imf", 2, 1, 2
    teacher.feature_dim, teacher.action_dim = 4, 2
    teacher.host_params = {"constant": np.ones(2, np.float32)}
    teacher.params = jax.tree.map(jnp.asarray, teacher.host_params)
    teacher.live_keys = frozenset(teacher.params)
    teacher.before = parameter_digest(teacher.host_params)
    teacher.obs_space = {
        "position": SimpleNamespace(shape=(2,), dtype=np.float32),
        "is_first": SimpleNamespace(shape=(), dtype=bool),
    }

    class Encoder:
        def initial(self, batch):
            return jnp.zeros((batch, 2))

        def __call__(self, carry, obs, reset, training, single):
            # An encoder RNG draw must not shift the posterior's RNG draw.
            noise = jax.random.normal(nj.seed(), obs["position"].shape)
            token = obs["position"] + noise
            return jnp.where(reset[:, None], 0, carry) + 1, {}, token

    class Dynamics:
        def initial(self, batch):
            return dict(
                deter=jnp.zeros((batch, 2), jnp.bfloat16),
                stoch=jnp.zeros((batch, 1, 2), jnp.bfloat16),
            )

        def observe(self, carry, tokens, action, reset, training, single):
            h = jnp.where(reset[:, None], 0, carry["deter"] + carry["stoch"][:, 0])
            h = h + action["action"]
            mean = tokens[:, None, :]
            std = jnp.ones_like(mean)
            z = (mean + jax.random.normal(nj.seed(), mean.shape)).astype(jnp.bfloat16)
            state = dict(deter=h.astype(jnp.bfloat16), stoch=z)
            return state, {}, dict(**state, mean=mean, std=std)

    model = teacher.model = SimpleNamespace(
        enc=Encoder(),
        dyn=Dynamics(),
        feat2tensor=lambda f: jnp.concatenate(
            [f["deter"], f["stoch"].reshape(1, -1)], -1
        ),
    )

    def body(carry, obs, previous):
        enc, dyn = carry
        enc, _, tokens = model.enc(enc, obs, obs["is_first"], False, single=True)
        dyn, _, feat = model.dyn.observe(
            dyn, tokens, {"action": previous}, obs["is_first"], False, single=True
        )
        return (enc, dyn), model.feat2tensor(feat).astype(jnp.float32)

    teacher._observe = jax.jit(
        lambda p, c, o, a, key: nj.pure(body)(p, c, o, a, seed=key)[1]
    )
    return teacher


class PosteriorTests(unittest.TestCase):
    def test_observe_isolates_inputs_and_keeps_actual_draw(self):
        teacher, obs = FakeTeacher(), observation(0, 0, True)
        carry, action = teacher.initial(), np.ones(2, np.float32)
        saved = copy.deepcopy((carry, obs, action))
        with patch.object(posterior, "_statistics_call", side_effect=fake_stats):
            state, stats = posterior.observe_statistics(teacher, carry, obs, action, 13)
        for actual, expected in zip(
            jax.tree.leaves((carry, obs, action)), jax.tree.leaves(saved)
        ):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(state["z"], stats["z"])
        self.assertFalse(np.array_equal(stats["z"], stats["mean"]))
        self.assertEqual([c[0] for c in teacher.calls], [13, 13])
        for c in teacher.calls:
            np.testing.assert_array_equal(c[1], action)
        self.assertEqual(teacher.frozen_digest(), teacher.before)

    def test_real_jax_ninjax_body_seed_reset_and_frozen_model_interface(self):
        teacher = tiny_jax_teacher()
        carry = teacher.initial()
        for t, seed in enumerate((5, 29, 31, 32)):
            obs = dict(
                position=np.array([t, -t], np.float32), is_first=np.asarray(t in (0, 3))
            )
            action = np.array([1, 0], np.float32)
            original, feature = teacher.observe(carry, obs, action, seed)
            result, stats = posterior.observe_statistics(
                teacher, carry, obs, action, seed
            )
            posterior._tree_exact(original, result, "test")
            np.testing.assert_array_equal(np.r_[stats["h"], stats["z"]], feature)
            self.assertFalse(np.array_equal(stats["mean"], stats["z"]))
            self.assertEqual(stats["mean"].shape, (2,))
            carry = result
        self.assertEqual(teacher.frozen_digest(), teacher.before)
        self.assertEqual(posterior._COMPILED[teacher]._cache_size(), 1)

    def test_extract_all_splits_exact_features_and_arbitrary_index_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            teacher, data = fixture(root)
            before = copy.deepcopy(data)
            hashes = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch.object(posterior, "_statistics_call", side_effect=fake_stats):
                stats = posterior.extract_statistics(teacher, data, root)
                indices = np.array([11, 0, 6, 3])
                subset = posterior.extract_statistics(teacher, data, root, indices)
            self.assertEqual(stats["h"].shape, (12, 15, 2))
            np.testing.assert_array_equal(
                np.concatenate([stats["h"], stats["z"]], -1), data["targets"]
            )
            for name in stats:
                np.testing.assert_array_equal(subset[name], stats[name][indices])
            for name in data:
                np.testing.assert_array_equal(data[name], before[name])
            self.assertEqual(hashes, {p.name: p.read_bytes() for p in root.iterdir()})
            self.assertEqual(set(data["split"]), {0, 1, 2})

    def test_subset_reconstructs_whole_prefix_with_original_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            teacher, data = fixture(Path(td))
            with patch.object(posterior, "_statistics_call", side_effect=fake_stats):
                posterior.extract_statistics(teacher, data, td, np.array([2]))
            seeds = [v[0] for v in teacher.calls]
            # Seed routing duplicates calls for exact statistics/canonical parity,
            # and filtering reaches anchor four without resetting at the anchor.
            self.assertEqual(seeds[:10], np.repeat(np.arange(5), 2).tolist())
            self.assertEqual(sum(x[2] for x in teacher.calls), 2)
            self.assertEqual(len(seeds), 2 * (5 + 15))

    def test_fail_closed_on_statistics_ulp_carry_or_nonfinite(self):
        for defect in ("ulp", "carry", "nonfinite", "std"):
            teacher = FakeTeacher()

            def bad(*args):
                carry, stats = fake_stats(*args)
                if defect == "ulp":
                    stats["z"][0] = np.nextafter(stats["z"][0], np.float32(np.inf))
                elif defect == "carry":
                    carry["h"][0] += 1
                elif defect == "nonfinite":
                    stats["mean"][0] = np.nan
                else:
                    stats["std"][0] = 0
                return carry, stats

            with patch.object(posterior, "_statistics_call", side_effect=bad):
                with self.assertRaises(ValueError):
                    posterior.observe_statistics(
                        teacher,
                        teacher.initial(),
                        observation(0, 1),
                        np.zeros(2, np.float32),
                        13,
                    )

    def test_hash_guards_reject_preexisting_and_during_call_mutations(self):
        teacher = FakeTeacher()
        teacher.weights[0] += 1
        with self.assertRaisesRegex(ValueError, "before extraction"):
            posterior.observe_statistics(
                teacher, teacher.initial(), observation(0, 1), np.zeros(2), 1
            )
        teacher = FakeTeacher()

        def mutate(*args):
            result = fake_stats(*args)
            teacher.weights[0] += 1
            return result

        with patch.object(posterior, "_statistics_call", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during extraction"):
                posterior.observe_statistics(
                    teacher, teacher.initial(), observation(0, 1), np.zeros(2), 1
                )

    def test_retained_seed_reset_previous_action_and_index_negative_controls(self):
        for defect in ("seed", "reset", "previous", "anchor_seed", "feature"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                teacher, data = fixture(root)
                path = root / (
                    "episode-000.npz" if defect == "reset" else "branch-000-002-0.npz"
                )
                with np.load(path) as archive:
                    values = {k: archive[k].copy() for k in archive.files}
                if defect == "seed":
                    values["observe_seeds"][1] += 1
                elif defect == "anchor_seed":
                    values["observe_seeds"][0] += 1
                elif defect == "previous":
                    values["previous_action"][0] += 1
                elif defect == "feature":
                    values["features"][1, 0] += 1
                else:
                    index = values["raw_keys"].tolist().index("is_first")
                    values[f"raw_{index:03d}"][1] = True
                np.savez(path, **values)
                with patch.object(
                    posterior, "_statistics_call", side_effect=fake_stats
                ):
                    with self.assertRaises(ValueError):
                        posterior.extract_statistics(teacher, data, root, np.array([0]))
        with tempfile.TemporaryDirectory() as td:
            teacher, data = fixture(Path(td))
            for indices in ([], [0, 0], [-1], [12], [0.5], [[0]], [True]):
                with self.assertRaisesRegex(ValueError, "indices"):
                    posterior.extract_statistics(teacher, data, td, indices)
        with self.assertRaisesRegex(ValueError, "uint32"):
            posterior.observe_statistics(FakeTeacher(), {}, {}, np.zeros(2), -1)


if __name__ == "__main__":
    unittest.main()
