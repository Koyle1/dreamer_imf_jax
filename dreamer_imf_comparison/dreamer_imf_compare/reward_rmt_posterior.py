"""Read-only posterior statistics on the original retained inference path.

No environment, optimizer, policy, or production inference method is modified.
Statistics use a separate pure observation execution with the *same* seed. Its
carry and sampled features must exactly match canonical ``teacher.observe``;
the canonical sampled carry is always the one continued through the history.
Extra JIT outputs can change compiler scheduling. Any resulting mismatch is an
integrity failure, never a reason to relax equality or silently resample.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
import weakref

import numpy as np

from .parallel_collection import load_raw_history
from .reward_diagnosis_data import check_branch, flatten

_COMPILED = weakref.WeakKeyDictionary()


def _exact(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError("exact posterior replay differs: " + label)


def _tree_exact(actual, expected, label):
    import jax

    if jax.tree.structure(actual) != jax.tree.structure(expected):
        raise ValueError("posterior carry structure differs: " + label)
    for i, (a, b) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected))):
        _exact(a, b, f"{label} leaf {i}")


def _integer(value, label):
    value = np.asarray(value)
    if value.shape or value.dtype.kind not in "iu" or not 0 <= int(value) < 2**32:
        raise ValueError("invalid nonnegative uint32 integer: " + label)
    return int(value)


@contextmanager
def _frozen(teacher):
    before = teacher.frozen_digest()
    if before != teacher.before:
        raise ValueError("frozen posterior parameters differ before extraction")
    try:
        yield
    finally:
        if teacher.frozen_digest() != before:
            raise ValueError("frozen posterior parameters changed during extraction")


def _statistics_call(teacher, carry, observation, previous_action, seed):
    """Execute unchanged observation module calls; only expose their outputs."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj

    if teacher not in _COMPILED:
        model = teacher.model
        deter = teacher.deter

        def body(carry, obs, previous):
            enc, dyn = carry
            reset = obs["is_first"]
            enc, _, tokens = model.enc(enc, obs, reset, False, single=True)
            dyn, _, feat = model.dyn.observe(
                dyn, tokens, {"action": previous}, reset, False, single=True
            )
            # Preserve FrozenModel's exact feature conversion and partition,
            # including its deployed mixed-precision casts. Extra mean/std
            # outputs can still change lowering, hence the caller's strict gate.
            feature = model.feat2tensor(feat).astype(jnp.float32)
            # Do not recompute _normal, draw another sample, or change dyn carry.
            stats = {
                name: feat[key].astype(jnp.float32).reshape((1, -1))
                for name, key in (
                    ("mean", "mean"),
                    ("std", "std"),
                )
            }
            stats.update(h=feature[:, :deter], z=feature[:, deter:])
            return (enc, dyn), stats

        pure = nj.pure(body)
        _COMPILED[teacher] = jax.jit(
            lambda p, c, o, a, key: pure(
                p, c, o, a, seed=key, create=False, modify=False
            )[1]
        )
    obs = {
        k: np.asarray(observation[k], dtype=s.dtype).reshape((1, *s.shape))
        for k, s in teacher.obs_space.items()
    }
    result, stats = _COMPILED[teacher](
        teacher.params,
        jax.tree.map(jax.device_put, carry),
        jax.tree.map(jax.device_put, obs),
        jax.device_put(np.asarray(previous_action, np.float32)[None]),
        jax.random.PRNGKey(seed),
    )
    return jax.tree.map(np.asarray, jax.device_get(result)), {
        k: np.asarray(jax.device_get(v))[0] for k, v in stats.items()
    }


def _observe(teacher, carry, obs, previous_action, seed):
    if teacher.arm != "imf":
        raise ValueError("posterior statistics require the continuous iMF teacher")
    seed = _integer(seed, "observation seed")
    action = np.asarray(previous_action)
    if action.shape != (teacher.action_dim,) or not np.isfinite(action).all():
        raise ValueError("invalid previous action")
    # Both calls receive isolated copies, including for adapters that use in-place
    # host normalization. Neither can mutate the caller's retained evidence.
    result, feature = teacher.observe(
        copy.deepcopy(carry), copy.deepcopy(obs), action.copy(), seed
    )
    stat_carry, stats = _statistics_call(
        teacher, copy.deepcopy(carry), copy.deepcopy(obs), action.copy(), seed
    )
    _tree_exact(stat_carry, result, "statistics versus canonical carry")
    if set(stats) != {"h", "mean", "std", "z"}:
        raise ValueError("invalid posterior statistics fields")
    widths = dict(h=teacher.deter, mean=teacher.feature_dim - teacher.deter)
    for name, value in stats.items():
        value = np.asarray(value)
        width = widths["h" if name == "h" else "mean"]
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError("invalid posterior statistic: " + name)
        stats[name] = value.copy()
    if np.any(stats["std"] <= 0):
        raise ValueError("posterior standard deviation must be positive")
    _exact(np.r_[stats["h"], stats["z"]], feature, "statistics versus canonical h/z")
    return result, stats


def observe_statistics(teacher, carry, obs, previous_action, seed):
    """Return canonical carry and flattened h/mean/std/z without resampling.

    A standalone call checks frozen parameter hashes before and after inference.
    ``extract_statistics`` amortizes this expensive full-parameter hashing over
    extraction, while checking exact carry/features at every observation.
    """
    with _frozen(teacher):
        return _observe(teacher, carry, obs, previous_action, seed)


def _validate_history(history, raw, stop, teacher):
    if stop >= len(raw) or len(raw) < 2:
        raise ValueError("anchor outside retained episode")
    if np.asarray(history["features"]).shape != (len(raw), teacher.feature_dim):
        raise ValueError("invalid retained episode feature dimensions")
    if len(history["observe_seeds"]) != len(raw):
        raise ValueError("invalid episode seed count")
    if np.asarray(history["actions"]).shape != (len(raw) - 1, teacher.action_dim):
        raise ValueError("invalid episode action count")
    if len(history["rewards"]) != len(raw) - 1:
        raise ValueError("invalid episode reward count")
    for t, obs in enumerate(raw[: stop + 1]):
        if bool(obs["is_first"]) != (t == 0):
            raise ValueError("retained episode reset mask differs")
        if bool(obs["is_last"]) or bool(obs.get("is_terminal", False)):
            raise ValueError("terminal before retained anchor")
        _exact(flatten(obs), history["observations"][t], "episode observation index")
        if t:
            _exact(obs["reward"], history["rewards"][t - 1], "episode reward index")


def extract_statistics(teacher, data, location, indices=None):
    """Replay selected retained branches; return N-by-H-by-feature arrays.

    ``indices=None`` extracts every split. Otherwise output rows follow the exact
    order of a nonempty unique integer index vector (not episode iteration order).
    The full original prefix is reconstructed even when only one row is selected.
    Original sampling seeds and sampled carries are used, including at branching.
    """
    target = np.asarray(data["targets"])
    if (
        target.ndim != 3
        or target.shape[1:] != (15, teacher.feature_dim)
        or not np.isfinite(target).all()
    ):
        raise ValueError("invalid retained posterior target shape or finiteness")
    n, horizon = target.shape[:2]
    indices = np.arange(n) if indices is None else np.asarray(indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not len(indices)
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= n)
    ):
        raise ValueError("indices must be nonempty unique in-range integers")
    indices = indices.astype(np.int64)
    for key in ("episode", "anchor", "plan"):
        values = np.asarray(data[key])
        if values.shape != (n,) or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError("invalid branch metadata: " + key)
    if np.any(data["anchor"][indices] == 0):
        raise ValueError("retained branches must start after the initial reset")
    widths = dict(h=teacher.deter, mean=teacher.feature_dim - teacher.deter)
    outputs = {
        name: np.empty(
            (len(indices), horizon, widths["h" if name == "h" else "mean"]),
            np.float32,
        )
        for name in ("h", "mean", "std", "z")
    }
    destinations = {int(row): dest for dest, row in enumerate(indices)}
    location = Path(location)
    with _frozen(teacher):
        for episode in np.unique(data["episode"][indices]):
            rows = indices[data["episode"][indices] == episode]
            history, raw = load_raw_history(location / f"episode-{episode:03d}.npz")
            stop = int(np.max(data["anchor"][rows]))
            _validate_history(history, raw, stop, teacher)
            carry = teacher.initial()
            for t in range(stop + 1):
                action = history["actions"][t - 1] if t else history["previous_action"]
                carry, stats = _observe(
                    teacher, carry, raw[t], action, history["observe_seeds"][t]
                )
                feature = np.r_[stats["h"], stats["z"]]
                _exact(feature, history["features"][t], f"episode {episode} step {t}")
                for i in rows[data["anchor"][rows] == t]:
                    plan = data["plan"][i]
                    branch, branch_raw = load_raw_history(
                        location / f"branch-{episode:03d}-{t:03d}-{plan}.npz"
                    )
                    check_branch(data, i, branch, branch_raw)
                    if set(branch_raw[0]) != set(raw[t]):
                        raise ValueError("branch anchor observation schema differs")
                    for name in raw[t]:
                        _exact(
                            branch_raw[0][name], raw[t][name], "branch anchor " + name
                        )
                    _exact(branch["previous_action"], action, "branch previous action")
                    if len(branch["observe_seeds"]) != horizon + 1:
                        raise ValueError("invalid branch seed count")
                    _exact(
                        branch["observe_seeds"][0],
                        history["observe_seeds"][t],
                        "branch anchor seed",
                    )
                    _exact(feature, data["start"][i], "dataset start belief")
                    state = copy.deepcopy(carry)
                    for step in range(horizon):
                        obs = branch_raw[step + 1]
                        if bool(obs.get("is_terminal", False)):
                            raise ValueError("terminal inside retained branch")
                        state, future = _observe(
                            teacher,
                            state,
                            obs,
                            branch["actions"][step],
                            branch["observe_seeds"][step + 1],
                        )
                        _exact(
                            np.r_[future["h"], future["z"]],
                            target[i, step],
                            f"retained branch {i} future {step}",
                        )
                        for name in outputs:
                            outputs[name][destinations[int(i)], step] = future[name]
    return outputs
