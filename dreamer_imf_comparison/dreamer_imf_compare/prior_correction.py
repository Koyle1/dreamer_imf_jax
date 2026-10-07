"""Isolated inference adapter; never patches RSSM or calls agent.train/policy."""

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import prior_correction as field


def validate_data(data, feature_dim):
    required = (
        "start",
        "targets",
        "actions",
        "observations",
        "rewards",
        "episode",
        "anchor",
        "plan",
        "split",
    )
    if any(k not in data for k in required):
        raise ValueError("missing retained data")
    n = len(data["start"])
    if data["targets"].shape != (n, 15, feature_dim):
        raise ValueError("expected 15 real posterior target features")
    shapes = dict(
        start=(n, feature_dim),
        actions=(n, 15, 2),
        observations=(n, 15, 6),
        rewards=(n, 15),
        episode=(n,),
        anchor=(n,),
        plan=(n,),
        split=(n,),
    )
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f"invalid {name}")
    if not np.isfinite(data["targets"]).all():
        raise ValueError("nonfinite targets")
    if set(data["split"].tolist()) != {0, 1, 2}:
        raise ValueError("train/validation/test required")
    for episode in np.unique(data["episode"]):
        if len(np.unique(data["split"][data["episode"] == episode])) != 1:
            raise ValueError("episode split leakage")
    if np.any(np.abs(data["actions"]) > 1):
        raise ValueError("unbounded action")
    keys = list(zip(data["episode"], data["anchor"], data["plan"]))
    if len(set(keys)) != n:
        raise ValueError("duplicate branches")
    for episode, anchor in set(zip(data["episode"], data["anchor"])):
        rows = (data["episode"] == episode) & (data["anchor"] == anchor)
        if set(data["plan"][rows].tolist()) != {0, 1, 2, 3}:
            raise ValueError("all four matched action plans required")


def normalization(data, deter):
    values = data["targets"][data["split"] == 0]
    mean = values.mean((0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(values.std((0, 1), dtype=np.float64), 0.01).astype(np.float32)
    return dict(
        hmean=mean[:deter], hstd=std[:deter], zmean=mean[deter:], zstd=std[deter:]
    )


class FrozenPrior:
    def __init__(self, teacher):
        import ninjax as nj
        import embodied.jax.nets as nn

        if teacher.arm != "imf":
            raise ValueError("continuous repaired iMF parent required")
        self.teacher = teacher
        self.before = teacher.frozen_digest()

        def prior(h):
            mean, std = teacher.model.dyn._prior_normal(nn.cast(h))
            return mean.reshape(h.shape[0], -1), std.reshape(h.shape[0], -1)

        def core(feature, action):
            f = teacher.unpack(feature)
            return teacher.model.dyn._core(f["deter"], f["stoch"], nn.cast(action))

        self.prior = jax.jit(lambda h: nj.pure(prior)(teacher.params, h, seed=0)[1])
        self.core = jax.jit(lambda f, a: nj.pure(core)(teacher.params, f, a, seed=0)[1])

    def assert_frozen(self):
        if self.teacher.frozen_digest() != self.before:
            raise ValueError("frozen parent mutated")


def corrected(params, norm, h, mean, std, noise, *, bypass=False, steps=1):
    # Bypass in physical coordinates avoids unnecessary roundtrip rounding.
    if bypass:
        return field.base_sample(mean, std, noise)
    m, s = (mean - norm["zmean"]) / norm["zstd"], std / norm["zstd"]
    source = field.base_sample(m, s, noise)
    z = field.sample(
        params, (h - norm["hmean"]) / norm["hstd"], m, s, noise, steps=steps
    )
    # Add only the learned displacement to the physical base: exact identity
    # even when standardization/de-standardization would introduce rounding.
    return field.base_sample(mean, std, noise) + (z - source) * norm["zstd"]


def rollout(
    adapter, params, norm, starts, actions, key, *, particles=32, bypass=False, steps=1
):
    """Only initial real belief and open-loop actions condition future predictions."""
    teacher = adapter.teacher
    b, horizon = actions.shape[:2]
    feature = jnp.repeat(starts, particles, axis=0)
    actions = jnp.repeat(actions, particles, axis=0).swapaxes(0, 1)
    keys = jax.random.split(key, horizon)

    def step(f, pair):
        action, k = pair
        h = adapter.core(f, action).astype(jnp.float32)
        mean, std = adapter.prior(h)
        noise = jax.random.normal(k, mean.shape)
        z = corrected(params, norm, h, mean, std, noise, bypass=bypass, steps=steps)
        # Match frozen parent's deployed feature precision for both paths.
        z = z.astype(jnp.bfloat16).astype(jnp.float32)
        f = jnp.concatenate((h, z), -1)
        return f, f

    _, values = jax.lax.scan(step, feature, (actions, keys))
    return values.swapaxes(0, 1).reshape(b, particles, horizon, teacher.feature_dim)
