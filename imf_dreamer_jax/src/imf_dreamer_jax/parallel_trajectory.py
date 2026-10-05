"""Diagnostic-only joint-trajectory Improved MeanFlow field.

Each token is a *complete* future feature (both deterministic h and stochastic
z), not a physical-time transition.  Attention is bidirectional over the known
action plan.  Noise time r/t is shared by the trajectory, and sampling takes
flow-time steps over the entire tensor without an autoregressive state scan.

The default model is a width-128, four-head, four-block transformer with
512-wide feed-forward layers and no dropout.  All parameter leaves are arrays,
so ordinary JAX transformations and Optax optimizers work without special
static arguments or a dependency on the production Dreamer implementation.

Both velocity readouts are elementwise affine: (1+alpha)*x+beta, where the
transformer predicts alpha and beta for every feature coordinate.  This full-
rank diagonal path is essential when feature_dim exceeds backbone width: a
purely additive width-128 projection could never remove noise orthogonal to
its fixed output subspace.  There is no division by noise time.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .imf import sample_time_pairs
from .nn import Params, init_linear, linear


class LossDetails(NamedTuple):
    """Diagnostics; raw losses average over all horizon-by-feature entries."""

    loss: Array
    per_sample_loss: Array
    field: Array
    prediction: Array
    regression_target: Array
    jvp: Array
    marginal_velocity: Array
    interpolated: Array
    noise: Array
    r: Array
    t: Array
    raw_loss_u: Array
    raw_loss_v: Array
    adaptive_weight_u: Array
    adaptive_weight_v: Array


def _positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def init(
    key: Array,
    feature_dim: int,
    action_dim: int,
    horizon: int = 15,
    width: int = 128,
    heads: int = 4,
    layers: int = 4,
    ff_width: int = 512,
) -> Params:
    """Initialize trainable parameters; reduced sizes are useful for CPU tests."""

    for name, value in (
        ("feature_dim", feature_dim),
        ("action_dim", action_dim),
        ("horizon", horizon),
        ("width", width),
        ("heads", heads),
        ("layers", layers),
        ("ff_width", ff_width),
    ):
        _positive_integer(name, value)
    if width % heads:
        raise ValueError("width must be divisible by heads")
    keys = iter(jax.random.split(key, 8 + 4 * layers))

    def dense(input_dim, output_dim, gain=1.0):
        return init_linear(next(keys), input_dim, output_dim, gain=gain)

    def norm():
        return {
            "scale": jnp.ones((width,), dtype=jnp.float32),
            "bias": jnp.zeros((width,), dtype=jnp.float32),
        }

    params = {
        "x": dense(feature_dim, width),
        "start": dense(feature_dim, width),
        "action": dense(action_dim, width),
        "time_in": dense(3, width),
        "time_out": dense(width, width),
        "position": 0.02
        * jax.random.normal(next(keys), (horizon, width), dtype=jnp.float32),
    }
    blocks = []
    for _ in range(layers):
        # Shape records the head count without integer/static parameter leaves.
        qkv = dense(width, 3 * width)
        qkv = {
            "weight": qkv["weight"].reshape(width, 3, heads, width // heads),
            "bias": qkv["bias"].reshape(3, heads, width // heads),
        }
        blocks.append(
            {
                "attention_norm": norm(),
                "qkv": qkv,
                "attention_out": dense(width, width, gain=1.0 / math.sqrt(layers)),
                "ff_norm": norm(),
                "ff_in": dense(width, ff_width, gain=math.sqrt(2.0)),
                "ff_out": dense(ff_width, width, gain=1.0 / math.sqrt(layers)),
            }
        )
    params.update(
        {
            "blocks": tuple(blocks),
            "final_norm": norm(),
            "u": dense(width, 2 * feature_dim, gain=0.05),
            "v": dense(width, 2 * feature_dim, gain=0.05),
        }
    )
    return params


def _norm(params: Params, x: Array) -> Array:
    center = x - jnp.mean(x, axis=-1, keepdims=True)
    inverse_std = jax.lax.rsqrt(
        jnp.mean(jnp.square(center), axis=-1, keepdims=True) + 1e-5
    )
    return center * inverse_std * params["scale"] + params["bias"]


def _time(value: Array | float, reference: Array) -> Array:
    value = jnp.asarray(value, dtype=reference.dtype)
    batch = reference.shape[0]
    if value.ndim == 0:
        return jnp.broadcast_to(value, (batch, 1, 1))
    if value.shape in ((batch,), (batch, 1), (batch, 1, 1)):
        return value.reshape(batch, 1, 1)
    raise ValueError("time must be scalar, [batch], [batch, 1], or [batch, 1, 1]")


def _check_shapes(params: Params, x: Array, start: Array, actions: Array) -> None:
    horizon, _ = params["position"].shape
    features = params["x"]["weight"].shape[0]
    action_dim = params["action"]["weight"].shape[0]
    if x.ndim != 3 or x.shape[1:] != (horizon, features) or x.shape[0] < 1:
        raise ValueError(
            f"x must have shape [batch, {horizon}, {features}] with nonempty batch"
        )
    if start.shape != (x.shape[0], features):
        raise ValueError("start must have shape [batch, feature_dim]")
    if actions.shape != (x.shape[0], horizon, action_dim):
        raise ValueError("actions must have shape [batch, horizon, action_dim]")


def outputs(
    params: Params,
    x: Array,
    start: Array,
    actions: Array,
    r: Array | float,
    t: Array | float,
) -> tuple[Array, Array]:
    """Return joint average u and auxiliary instantaneous v, both [B,H,D].

    The initial state and the complete *known* action plan are conditioning
    inputs.  This model intentionally imposes no causal attention mask; its
    learned action-prefix consistency must be measured, not assumed.
    """

    x, start, actions = (jnp.asarray(v, dtype=jnp.float32) for v in (x, start, actions))
    _check_shapes(params, x, start, actions)
    r, t = _time(r, x), _time(t, x)
    time_embedding = linear(
        params["time_out"],
        jax.nn.silu(linear(params["time_in"], jnp.concatenate((r, t, t - r), axis=-1))),
    )
    hidden = (
        linear(params["x"], x)
        + linear(params["start"], start)[:, None, :]
        + linear(params["action"], actions)
        + params["position"][None, :, :]
        + time_embedding
    )
    for block in params["blocks"]:
        normalized = _norm(block["attention_norm"], hidden)
        qkv = jnp.einsum("bhi,iqkd->bhqkd", normalized, block["qkv"]["weight"])
        qkv = qkv + block["qkv"]["bias"]
        query, key, value = (qkv[:, :, index] for index in range(3))
        logits = jnp.einsum("bhkd,bjkd->bkhj", query, key) / math.sqrt(query.shape[-1])
        attention = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("bkhj,bjkd->bhkd", attention, value).reshape(hidden.shape)
        hidden = hidden + linear(block["attention_out"], attended)
        hidden = hidden + linear(
            block["ff_out"],
            jax.nn.gelu(linear(block["ff_in"], _norm(block["ff_norm"], hidden))),
        )
    hidden = _norm(params["final_norm"], hidden)

    def affine_velocity(head):
        alpha, beta = jnp.split(linear(head, hidden), 2, axis=-1)
        return (1.0 + alpha) * x + beta

    return affine_velocity(params["u"]), affine_velocity(params["v"])


def loss(
    params: Params,
    targets: Array,
    start: Array,
    actions: Array,
    key: Array,
    *,
    return_details: bool = False,
    noise: Array | None = None,
    r: Array | float | None = None,
    t: Array | float | None = None,
    adaptive_power: float = 1.0,
    adaptive_epsilon: float = 0.01,
    boundary_fraction: float = 0.5,
    meanflow_scale: float = 1.0,
    velocity_scale: float = 1.0,
) -> Array | LossDetails:
    """Whole-trajectory iMF with the auxiliary head supervised at (t,t).

    x_t=(1-t)*targets+t*noise, v=v(x_t,start,actions,t,t),
    D_t u = JVP[u; (v,0,1)], prediction=u+(t-r)*stop(D_t u).
    Both prediction and v regress to stop(noise-targets).  Adaptive denominators
    sum squared errors over *all* H*D coordinates and are also stopped.  The
    default logit-normal times and boundary mass match the existing iMF core.

    An ordinary finite difference of this function is NOT its AD derivative:
    derivative tests must freeze the correction and weights at the base point.
    """

    targets, start, actions = (
        jnp.asarray(v, dtype=jnp.float32) for v in (targets, start, actions)
    )
    _check_shapes(params, targets, start, actions)
    if (r is None) != (t is None):
        raise ValueError("r and t must both be supplied or both omitted")
    for name, value in (
        ("adaptive_power", adaptive_power),
        ("meanflow_scale", meanflow_scale),
        ("velocity_scale", velocity_scale),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(adaptive_epsilon) or adaptive_epsilon <= 0:
        raise ValueError("adaptive_epsilon must be finite and positive")
    if not 0 <= boundary_fraction <= 1:
        raise ValueError("boundary_fraction must lie in [0, 1]")
    noise_key, time_key = jax.random.split(key)
    if noise is None:
        noise = jax.random.normal(noise_key, targets.shape, dtype=targets.dtype)
    else:
        noise = jnp.asarray(noise, dtype=targets.dtype)
    if noise.shape != targets.shape:
        raise ValueError("noise must have the target shape")
    if r is None:
        r, t = sample_time_pairs(
            time_key, targets.shape[0], boundary_fraction=boundary_fraction
        )
    r, t = _time(r, targets), _time(t, targets)
    interpolated = (1.0 - t) * targets + t * noise
    marginal_velocity = outputs(params, interpolated, start, actions, t, t)[1]

    def field(x_value, r_value, t_value):
        return outputs(params, x_value, start, actions, r_value, t_value)[0]

    value, tangent = jax.jvp(
        field,
        (interpolated, r, t),
        (marginal_velocity, jnp.zeros_like(r), jnp.ones_like(t)),
    )
    prediction = value + (t - r) * jax.lax.stop_gradient(tangent)
    regression_target = jax.lax.stop_gradient(noise - targets)
    square_u = jnp.square(prediction - regression_target)
    square_v = jnp.square(marginal_velocity - regression_target)
    summed_u, summed_v = (jnp.sum(v, axis=(1, 2)) for v in (square_u, square_v))
    weight_u = jnp.power(summed_u + adaptive_epsilon, adaptive_power)
    weight_v = jnp.power(summed_v + adaptive_epsilon, adaptive_power)
    per_sample = meanflow_scale * summed_u / jax.lax.stop_gradient(
        weight_u
    ) + velocity_scale * summed_v / jax.lax.stop_gradient(weight_v)
    result = jnp.mean(per_sample)
    if not return_details:
        return result
    return LossDetails(
        result,
        per_sample,
        value,
        prediction,
        regression_target,
        tangent,
        marginal_velocity,
        interpolated,
        noise,
        r,
        t,
        jnp.mean(square_u, axis=(1, 2)),
        jnp.mean(square_v, axis=(1, 2)),
        weight_u,
        weight_v,
    )


def sample(
    params: Params,
    start: Array,
    actions: Array,
    noise: Array,
    flow_steps: int = 1,
) -> Array:
    """Generate all H future joint features in flow_steps whole-field calls.

    flow_steps is a static positive Python integer under jax.jit.  Every call
    uses all H tokens; increasing it refines *noise time*, not physical time.
    At one step this is exactly noise-u(noise,start,actions,0,1).
    """

    _positive_integer("flow_steps", flow_steps)
    state = jnp.asarray(noise, dtype=jnp.float32)
    _check_shapes(params, state, start, actions)
    for index in range(flow_steps):
        t, r = 1.0 - index / flow_steps, 1.0 - (index + 1) / flow_steps
        state = state - (t - r) * outputs(params, state, start, actions, r, t)[0]
    return state
