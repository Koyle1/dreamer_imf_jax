"""Pure retained-history posterior/reward probes; no simulator or learner calls.

Each input contains ten complete 501-row evaluation histories. Incoming actions
are retained explicitly; anchor t predicts rewards/observations t+1..t+15 using
outgoing actions t..t+14. One actual posterior pass exposes h,z,mean,std. The
teacher is evaluated only on that real-observation posterior, never imagined
states. Its mean is not recomputed and no extra posterior sample is drawn.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

EPISODES, ROWS, PARTICLES, HORIZON = 10, 501, 8, 15
ANCHORS = (100, 200, 300, 400)
OBS_KEYS = ("position", "to_target", "velocity")


def _finite(value, name, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("nonfinite or nonnumeric retained field: " + name)
    if shape is not None and array.shape != shape:
        raise ValueError("retained field shape differs: " + name)
    return array


def validate_histories(proxy, data):
    """Validate complete episodes and exact incoming/outgoing action alignment."""
    if not {*OBS_KEYS, "reward", "is_first", "is_last", "is_terminal"}.issubset(
        proxy.obs_space
    ):
        raise ValueError("observation space lacks required retained AR2 fields")
    needed = set(proxy.obs_space) | {"action", "previous_action", "episode_seed"}
    if not needed.issubset(data):
        raise ValueError(
            "missing retained history fields: " + str(sorted(needed - set(data)))
        )
    selected = {}
    for key, space in proxy.obs_space.items():
        selected[key] = _finite(data[key], key, (EPISODES, ROWS, *space.shape)).copy()
    for key in OBS_KEYS:
        _finite(data[key], key, (EPISODES, ROWS, 2))
    for key in ("action", "previous_action"):
        selected[key] = _finite(data[key], key, (EPISODES, ROWS, 2)).copy()
        if np.any(np.abs(selected[key]) > 1):
            raise ValueError("retained action exceeds simulator bounds")
    ids = _finite(data["episode_seed"], "episode_seed", (EPISODES,))
    if (
        ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != EPISODES
        or np.any(ids < 0)
        or np.any(ids >= 2**32)
    ):
        raise ValueError("episode_seed must contain ten distinct uint32 identities")
    selected["episode_seed"] = ids.astype(np.uint32)
    first, last, terminal = (
        selected[name] for name in ("is_first", "is_last", "is_terminal")
    )
    if any(value.dtype.kind != "b" for value in (first, last, terminal)):
        raise ValueError("episode boundary masks must be boolean")
    if (
        not first[:, 0].all()
        or first[:, 1:].any()
        or not last[:, -1].all()
        or last[:, :-1].any()
        or terminal[:, :-1].any()
    ):
        raise ValueError("retained evaluation episodes are incomplete or cross resets")
    reward = selected["reward"]
    if not np.isin(reward, (0, 1, 2)).all() or (reward[:, 0] != 0).any():
        raise ValueError("reward must be AR2 labels 0/1/2 with zero reset reward")
    if np.any(selected["previous_action"][:, 0] != 0) or not np.array_equal(
        selected["previous_action"][:, 1:], selected["action"][:, :-1]
    ):
        raise ValueError("incoming action is not the preceding executed action")
    return selected


def _digest(data):
    value = hashlib.sha256()
    for name, raw in sorted(data.items()):
        array = np.ascontiguousarray(raw)
        value.update(name.encode())
        value.update(str((array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return np.frombuffer(value.digest(), dtype=np.uint8).copy()


def _windows(value, offset):
    return np.stack(
        [value[:, t + offset : t + offset + HORIZON] for t in ANCHORS], 1
    ).reshape(EPISODES * len(ANCHORS), HORIZON, *value.shape[2:])


def predict_arrays(proxy, data, seed):
    """Predict from immutable histories, returning finite NumPy arrays only.

    ``proxy`` exposes model, params, obs_space. ``data`` has every observation
    array [10,501,...], action/previous_action [10,501,2], episode_seed [10].
    Extra policy-log fields are intentionally unused. N=40 episode-major anchor
    rows, S=8, H=15. Posterior heads have [N,H]; imagined heads [N,S,H]. Original
    head losses are retained under *_native_loss, never invented three-class
    probabilities. All auxiliary probabilities have a final class axis of 3.
    """
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) < 2**32
    ):
        raise ValueError("probe seed must be a uint32 integer")
    retained = validate_histories(proxy, data)
    truth_reward = _windows(retained["reward"], 1)
    truth_observation = _windows(np.concatenate([retained[k] for k in OBS_KEYS], -1), 1)
    action = _windows(retained["action"], 0)
    import jax
    import jax.numpy as jnp
    import ninjax as nj

    model = proxy.model
    n = EPISODES * len(ANCHORS)

    def body(observations, previous, planned, truth):
        reset = observations["is_first"]
        _, _, tokens = model.enc(
            model.enc.initial(EPISODES), observations, reset, False
        )
        _, _, posterior = model.dyn.observe(
            model.dyn.initial(EPISODES), tokens, {"action": previous}, reset, False
        )
        if not {"deter", "stoch", "mean", "std"}.issubset(posterior):
            raise ValueError("same-pass posterior mean/std are absent")

        def windows(value):
            return jnp.stack(
                [value[:, t + 1 : t + 1 + HORIZON] for t in ANCHORS], 1
            ).reshape((n, HORIZON, *value.shape[2:]))

        features = {name: windows(posterior[name]) for name in ("deter", "stoch")}
        means, stds = (
            windows(posterior[name]).astype(jnp.float32).reshape(n, HORIZON, -1)
            for name in ("mean", "std")
        )
        student_input = model.feat2tensor(features)
        teacher_input = jnp.concatenate(
            [features["deter"].astype(jnp.float32), means], -1
        )
        original = model.rew(student_input, 2)
        student = model.reward_student(student_input, 2)
        teacher = model.reward_teacher(teacher_input, 2)
        starts = {
            name: posterior[name][:, jnp.asarray(ANCHORS)].reshape(
                (n, *posterior[name].shape[2:])
            )
            for name in ("deter", "stoch")
        }
        starts = jax.tree.map(
            lambda value: jnp.repeat(value, PARTICLES, axis=0), starts
        )
        _, imagined, _ = model.dyn.imagine(
            starts, {"action": jnp.repeat(planned, PARTICLES, axis=0)}, HORIZON, False
        )
        imagined_input = model.feat2tensor(imagined)
        predicted_original = model.rew(imagined_input, 2)
        predicted_student = model.reward_student(imagined_input, 2)
        _, _, decoded = model.dec(
            model.dec.initial(n * PARTICLES),
            imagined,
            jnp.zeros((n * PARTICLES, HORIZON), bool),
            False,
        )
        observations = jnp.concatenate([decoded[key].pred() for key in OBS_KEYS], -1)
        return dict(
            posterior_original=original.pred(),
            posterior_student=student.pred(),
            posterior_teacher=teacher.pred(),
            posterior_original_native_loss=original.loss(truth),
            posterior_student_probabilities=student.probabilities(),
            posterior_teacher_probabilities=teacher.probabilities(),
            posterior_mean=means,
            posterior_std=stds,
            predicted_original=predicted_original.pred().reshape(n, PARTICLES, HORIZON),
            predicted_student=predicted_student.pred().reshape(n, PARTICLES, HORIZON),
            predicted_original_native_loss=predicted_original.loss(
                jnp.repeat(truth, PARTICLES, axis=0)
            ).reshape(n, PARTICLES, HORIZON),
            predicted_student_probabilities=predicted_student.probabilities().reshape(
                n, PARTICLES, HORIZON, 3
            ),
            predicted_observation=observations.reshape(n, PARTICLES, HORIZON, 6),
        )

    pure = nj.pure(body)

    @jax.jit
    def predict(params, obs, previous, plans, truth):
        # Ninjax scan discovers accessed parameters in its prepass. Disabling
        # both create and modify skips that discovery and passes an empty state
        # into the recurrent body. Permit local context bookkeeping, but reject
        # any actual parameter/state mutation before returning probe evidence.
        after, values = pure(
            params,
            obs,
            previous,
            plans,
            truth,
            seed=int(seed),
            create=False,
            modify=True,
        )
        if set(after) != set(params):
            raise ValueError("prediction probe changed checkpoint state keys")
        if any(
            after[key].shape != params[key].shape
            or after[key].dtype != params[key].dtype
            for key in params
        ):
            raise ValueError("prediction probe modified checkpoint state shape/dtype")
        unchanged = jnp.stack(
            [jnp.array_equal(after[key], params[key]) for key in params]
        ).all()
        return values, unchanged

    values, unchanged = predict(
        proxy.params,
        jax.device_put({k: retained[k] for k in proxy.obs_space}),
        jax.device_put(retained["previous_action"]),
        jax.device_put(action),
        jax.device_put(truth_reward),
    )
    if not bool(jax.device_get(unchanged)):
        raise ValueError("prediction probe modified checkpoint state")
    # Materialize portable float32 evidence even if an upstream output uses the
    # backbone's bfloat16 storage dtype (NumPy reports that as a void dtype).
    arrays = {
        name: np.asarray(jax.device_get(value), dtype=np.float32)
        for name, value in values.items()
    }
    shapes = dict(
        posterior_original=(n, HORIZON),
        posterior_student=(n, HORIZON),
        posterior_teacher=(n, HORIZON),
        posterior_original_native_loss=(n, HORIZON),
        posterior_student_probabilities=(n, HORIZON, 3),
        posterior_teacher_probabilities=(n, HORIZON, 3),
        predicted_original=(n, PARTICLES, HORIZON),
        predicted_student=(n, PARTICLES, HORIZON),
        predicted_original_native_loss=(n, PARTICLES, HORIZON),
        predicted_student_probabilities=(n, PARTICLES, HORIZON, 3),
        predicted_observation=(n, PARTICLES, HORIZON, 6),
    )
    for name, value in arrays.items():
        _finite(value, name, shapes.get(name))
    if np.any(arrays["posterior_std"] <= 0):
        raise ValueError("nonpositive same-pass posterior standard deviation")
    arrays.update(
        truth_reward=truth_reward,
        truth_observation=truth_observation,
        action=action,
        episode=np.repeat(np.arange(EPISODES, dtype=np.int32), len(ANCHORS)),
        episode_seed=np.repeat(retained["episode_seed"], len(ANCHORS)),
        anchor=np.tile(np.asarray(ANCHORS, np.int32), EPISODES),
        probe_seed=np.asarray(seed, np.uint32),
        history_digest=_digest(retained),
    )
    return arrays


def create_probe(path, proxy, data, seed):
    """Create a new immutable NPZ; refuse existing evidence instead of replacing."""
    from .parallel_collection import _write_npz

    arrays = predict_arrays(proxy, data, seed)
    path = Path(path)
    _write_npz(path, arrays)
    return path


def verify_probe(path, proxy, data, seed):
    """Independently rerun fixed inputs and demand exact fields/dtypes/array values."""
    arrays = predict_arrays(proxy, data, seed)
    with np.load(path, allow_pickle=False) as retained:
        if set(retained.files) != set(arrays):
            raise ValueError("probe replay fields differ")
        for name, value in arrays.items():
            if (
                retained[name].dtype != value.dtype
                or retained[name].shape != value.shape
                or not np.array_equal(retained[name], value)
            ):
                raise ValueError("exact probe replay differs: " + name)
    return Path(path)
