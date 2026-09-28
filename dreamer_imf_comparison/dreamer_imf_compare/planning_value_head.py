"""Separate, CPU-only finite-return regression; never a replacement ReBRAC critic.

Targets must be recorded continuations of the *frozen* policy from each supplied
state. Replay rewards following other actions are not policy-value labels. This
module cannot establish that provenance: the study's authenticated data builder
must do so. No validation or test inputs enter fitting or its normalizers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HeadSettings:
    hidden_width: int = 128
    updates: int = 2000
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    maximum_steps: int = 1000
    normalization_floor: float = 1e-6


def _settings(settings: HeadSettings) -> None:
    if not isinstance(settings, HeadSettings):
        raise ValueError("settings must be HeadSettings")
    for name in ("hidden_width", "updates", "batch_size", "maximum_steps"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if settings.maximum_steps != 1000:
        raise ValueError("the time input contract is exactly 1000 native steps")
    for name in ("learning_rate", "normalization_floor"):
        value = getattr(settings, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    _gamma(settings.gamma)


def _gamma(gamma: float) -> None:
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite in [0, 1)")


def _real(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite real numbers")
    with np.errstate(over="ignore"):
        result = result.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be representable as finite float64")
    return result


def _remaining(value: Any) -> np.ndarray:
    result = _real(value, "remaining_steps")
    if (
        np.any(result != np.floor(result))
        or np.any(result < 0)
        or np.any(result > 1000)
    ):
        raise ValueError("remaining_steps must be integers in [0, 1000]")
    return result.astype(np.int32)


def finite_return_cap(remaining_steps: Any, *, gamma: float = 0.99) -> np.ndarray:
    """Maximum finite return for rewards in [0, 1], without terminal bootstrap."""
    _gamma(gamma)
    remaining = _remaining(remaining_steps)
    if gamma == 0:
        return (remaining > 0).astype(np.float64)
    cap = -np.expm1(remaining.astype(np.float64) * np.log(gamma)) / (1 - gamma)
    return np.where(remaining == 0, 0.0, cap)


def mc_suffix_returns(
    rewards: Any, continuations: Any, *, gamma: float = 0.99
) -> np.ndarray:
    """Return T+1 suffix values; reward[t] follows action[t] from state[t].

    The recorded episode boundary has value exactly zero, even if its native
    timeout continuation is one. A true terminal continuation cuts the suffix.
    No model, critic, or unrecorded continuation is bootstrapped.
    """
    _gamma(gamma)
    rewards = _real(rewards, "rewards")
    continuations = _real(continuations, "continuations")
    if rewards.ndim != 1 or rewards.shape != continuations.shape:
        raise ValueError("rewards and continuations must be matching vectors")
    if len(rewards) > 1000:
        raise ValueError("recorded episode exceeds 1000 native steps")
    if np.any((rewards < 0) | (rewards > 1)):
        raise ValueError("rewards must lie in [0, 1]")
    if np.any((continuations < 0) | (continuations > 1)):
        raise ValueError("continuations must lie in [0, 1]")
    values = np.zeros(len(rewards) + 1, np.float64)
    for index in range(len(rewards) - 1, -1, -1):
        values[index] = (
            rewards[index] + gamma * continuations[index] * values[index + 1]
        )
    return values


def _inputs(observations: Any, remaining_steps: Any) -> tuple[np.ndarray, np.ndarray]:
    observations = _real(observations, "observations")
    remaining = _remaining(remaining_steps)
    if (
        observations.ndim != 2
        or observations.shape[1] == 0
        or remaining.shape != (len(observations),)
    ):
        raise ValueError("expected observations [N,D] and remaining_steps [N]")
    with np.errstate(over="ignore"):
        observations = observations.astype(np.float32)
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be representable as float32")
    return observations, remaining


def _digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
    header = json.dumps([array.dtype.str, array.shape], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\n" + array.tobytes()).hexdigest()


def _cpu():
    # Import lazily and use a scoped default device; do not change process-wide
    # JAX configuration or steal any world-model GPU allocation.
    import jax

    devices = jax.devices("cpu")
    if not devices or devices[0].platform != "cpu":
        raise RuntimeError("planning-value regression requires a CPU device")
    return devices[0]


def _forward(params, inputs):
    import jax.numpy as jnp

    hidden = inputs
    for layer in params[:-1]:
        hidden = jnp.maximum(hidden @ layer["weight"] + layer["bias"], 0)
    return (hidden @ params[-1]["weight"] + params[-1]["bias"])[..., 0]


def fit_head(
    train_observations: Any,
    train_remaining_steps: Any,
    train_returns: Any,
    *,
    seed: int,
    settings: HeadSettings = HeadSettings(),
    sample_weights: Any = None,
) -> dict[str, Any]:
    """Fit one fixed head, uniformly by default or by fixed row probabilities.

    Optional weights define both the training-only normalization measure and
    sampling probabilities. They are not importance weights derived from a test
    split. Sampling is with replacement for exactly ``settings.updates`` updates.
    The returned dictionary is JSON serializable and contains no live JAX arrays.
    """
    _settings(settings)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    observations, remaining = _inputs(train_observations, train_remaining_steps)
    targets = _real(train_returns, "train_returns")
    count = len(observations)
    if count == 0 or targets.shape != (count,):
        raise ValueError("training requires nonempty compatible vectors")
    cap = finite_return_cap(remaining, gamma=settings.gamma)
    # Analytic geometric sums and the reverse finite-MC recurrence can differ
    # by a few float64 ULPs. This is only arithmetic roundoff, not target slack.
    roundoff = 16 * np.finfo(np.float64).eps * np.maximum(1.0, cap)
    if (
        np.any(targets < 0)
        or np.any(targets > cap + roundoff)
        or np.any(targets[remaining == 0] != 0)
    ):
        raise ValueError("train_returns violate the finite [0, 1]-reward cap")
    weights = (
        np.ones(count, np.float64)
        if sample_weights is None
        else _real(sample_weights, "sample_weights")
    )
    if weights.shape != (count,) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("sample_weights must be nonnegative with positive mass")
    # Scaling first avoids overflow for valid but large unnormalized weights.
    weights = weights / np.max(weights)
    weights = weights / np.sum(weights)
    obs64 = observations.astype(np.float64)
    obs_mean = np.sum(weights[:, None] * obs64, axis=0)
    obs_std = np.maximum(
        np.sqrt(np.sum(weights[:, None] * (obs64 - obs_mean) ** 2, axis=0)),
        settings.normalization_floor,
    )
    target_mean = float(np.sum(weights * targets))
    target_std = float(
        max(
            np.sqrt(np.sum(weights * (targets - target_mean) ** 2)),
            settings.normalization_floor,
        )
    )
    obs_mean, obs_std = obs_mean.astype(np.float32), obs_std.astype(np.float32)
    target_mean, target_std = np.float32(target_mean), np.float32(target_std)
    if not all(
        np.all(np.isfinite(x)) for x in (obs_mean, obs_std, target_mean, target_std)
    ):
        raise ValueError("training normalizers must be representable as float32")
    features = np.concatenate(
        (
            (observations.astype(np.float64) - obs_mean) / obs_std,
            remaining[:, None] / 1000.0,
        ),
        axis=1,
    ).astype(np.float32)
    normalized_targets = ((targets - target_mean) / target_std).astype(np.float32)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(normalized_targets)):
        raise ValueError("normalization generated nonfinite values")

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax.optim import adam_update, init_adam

    device = _cpu()
    with jax.default_device(device):
        inputs, labels, probabilities = (
            jax.device_put(x, device)
            for x in (features, normalized_targets, weights.astype(np.float32))
        )
        key = jax.device_put(jax.random.key(seed), device)
        sizes = (features.shape[1], settings.hidden_width, settings.hidden_width, 1)
        keys = jax.random.split(key, 4)
        params = tuple(
            {
                "weight": jax.random.normal(
                    keys[index], (left, right), dtype=jnp.float32
                )
                * np.float32(np.sqrt(2 / left)),
                "bias": jnp.zeros(right, jnp.float32),
            }
            for index, (left, right) in enumerate(
                zip(sizes[:-1], sizes[1:], strict=True)
            )
        )
        optimizer = init_adam(params)

        def loss(parameters, x, y):
            return jnp.mean(jnp.square(_forward(parameters, x) - y))

        def step(carry, unused):
            parameters, state, random_key = carry
            random_key, draw_key = jax.random.split(random_key)
            indices = jax.random.choice(
                draw_key, count, (settings.batch_size,), replace=True, p=probabilities
            )
            gradients = jax.grad(loss)(parameters, inputs[indices], labels[indices])
            parameters, state = adam_update(
                parameters, gradients, state, learning_rate=settings.learning_rate
            )
            return (parameters, state, random_key), None

        initial = _forward(params, inputs)
        initial_mse = float(jnp.sum(probabilities * jnp.square(initial - labels)))
        (params, optimizer, _), _ = jax.jit(
            lambda p, state, rng: jax.lax.scan(
                step, (p, state, rng), None, length=settings.updates
            )
        )(params, optimizer, keys[-1])
        final = _forward(params, inputs)
        final_mse = float(jnp.sum(probabilities * jnp.square(final - labels)))
        for leaf in jax.tree_util.tree_leaves(params):
            if any(item.platform != "cpu" for item in leaf.devices()):
                raise RuntimeError("planning-value fit unexpectedly left CPU")
        parameters = [
            {
                name: np.asarray(value, np.float32).tolist()
                for name, value in layer.items()
            }
            for layer in params
        ]
        if int(optimizer.step) != settings.updates:
            raise RuntimeError("optimizer update count differs")
    if not np.isfinite(initial_mse) or not np.isfinite(final_mse):
        raise ValueError("training produced nonfinite loss")
    head = dict(
        schema="finite_frozen_policy_value_head_v1",
        settings=asdict(settings),
        normalizers=dict(
            observation_mean=obs_mean.tolist(),
            observation_std=obs_std.tolist(),
            return_mean=float(target_mean),
            return_std=float(target_std),
        ),
        layers=parameters,
        training=dict(
            seed=seed,
            updates=settings.updates,
            examples=count,
            platform="cpu",
            sampling="fixed_row_probabilities_with_replacement",
            initial_normalized_mse=initial_mse,
            final_normalized_mse=final_mse,
            observation_sha256=_digest(observations),
            remaining_sha256=_digest(remaining),
            return_sha256=_digest(targets),
            normalized_weight_sha256=_digest(weights),
            sampling_probability_sha256=_digest(weights.astype(np.float32)),
        ),
    )
    head_bytes(head)  # Reject every nonfinite serialized parameter, not just loss.
    return head


def predict_head(
    head: dict[str, Any], observations: Any, remaining_steps: Any
) -> np.ndarray:
    """Bound finite return at inference; a zero-step query returns exactly zero."""
    if head.get("schema") != "finite_frozen_policy_value_head_v1":
        raise ValueError("unsupported planning-value head schema")
    settings = HeadSettings(**head["settings"])
    _settings(settings)
    observations, remaining = _inputs(observations, remaining_steps)
    stats = head["normalizers"]
    mean = _real(stats["observation_mean"], "observation_mean")
    std = _real(stats["observation_std"], "observation_std")
    target_mean = _real(stats["return_mean"], "return_mean")
    target_std = _real(stats["return_std"], "return_std")
    if (
        mean.shape != (observations.shape[1],)
        or std.shape != mean.shape
        or np.any(std <= 0)
        or target_mean.shape
        or target_std.shape
        or target_std <= 0
    ):
        raise ValueError("incompatible or invalid head normalizers")
    features = np.concatenate(
        ((observations.astype(np.float64) - mean) / std, remaining[:, None] / 1000.0),
        axis=1,
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("normalized query is not finite float32")
    sizes = (observations.shape[1] + 1, settings.hidden_width, settings.hidden_width, 1)
    if len(head["layers"]) != 3:
        raise ValueError("head must have exactly two hidden layers")
    parameters = []
    for layer, left, right in zip(head["layers"], sizes[:-1], sizes[1:], strict=True):
        weight, bias = _real(layer["weight"], "weight"), _real(layer["bias"], "bias")
        if weight.shape != (left, right) or bias.shape != (right,):
            raise ValueError("head layer shape differs from settings")
        parameters.append(
            dict(weight=weight.astype(np.float32), bias=bias.astype(np.float32))
        )
    import jax

    device = _cpu()
    with jax.default_device(device):
        parameters = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, device), parameters
        )
        raw = np.asarray(
            _forward(parameters, jax.device_put(features, device)), np.float64
        )
    raw = raw * target_std + target_mean
    if not np.all(np.isfinite(raw)):
        raise ValueError("head prediction is nonfinite")
    cap = finite_return_cap(remaining, gamma=settings.gamma)
    # Round a float32 cap downward when necessary, so output never exceeds the
    # mathematical float64 cap merely because the final cast rounds upward.
    cap32 = cap.astype(np.float32)
    cap32 = np.where(
        cap32.astype(np.float64) > cap, np.nextafter(cap32, np.float32(-np.inf)), cap32
    )
    bounded = np.clip(raw, 0, cap32).astype(np.float32)
    return np.where(remaining == 0, np.float32(0), bounded)


def head_bytes(head: dict[str, Any]) -> bytes:
    """Deterministic, non-pickle serialization suitable for SHA-256 bindings."""
    return json.dumps(
        head, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
