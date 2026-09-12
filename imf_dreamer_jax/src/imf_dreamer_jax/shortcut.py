"""Dreamer 4 x-prediction shortcut forcing for causal sequences.

This module implements Equation (7) of *Training Agents Inside of Scalable
World Models* as a model-agnostic JAX primitive.  Signal time follows the
paper's convention: ``tau=0`` is pure noise and ``tau=1`` is clean data.

The paper does not disclose the training value of ``K_max``.  It is therefore
a required argument wherever a training schedule or loss is sampled; this
library deliberately has no hidden default for it.  The predictor passed to
the loss has signature ``predict_clean(z_tau, tau, step_size)`` and must return
an array with the same ``[batch, time, features]`` shape as ``z_tau``.  Each
call receives the *entire* updated sequence.  A causal predictor must rebuild
its history features from those arguments on every call; caching a context
from the original corrupted sequence would not implement Equation (7).
Actions and other immutable causal inputs can be closed over by the callable.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


PredictClean = Callable[[Array, Array, Array], Array]
Reduction = Literal["none", "mean", "sum"]


class ShortcutSchedule(NamedTuple):
    """Independent discrete shortcut coordinates for every sequence token.

    ``step_count`` is sampled uniformly from ``{1, 2, ..., k_max}`` restricted
    to powers of two, ``step_size = 1 / step_count``, and
    ``tau = grid_index / step_count`` for a conditionally uniform
    ``grid_index`` in ``{0, ..., step_count - 1}``.
    """

    tau: Array
    step_size: Array
    step_count: Array
    is_finest: Array


class ShortcutForcingLossDetails(NamedTuple):
    """Observable components of the x-prediction shortcut-forcing loss."""

    loss: Array
    branch_loss: Array
    per_token_loss: Array
    weighted_per_token_loss: Array
    prediction: Array
    prediction_velocity: Array
    bootstrap_target_velocity: Array
    bootstrap_target_clean: Array
    first_prediction: Array
    first_velocity: Array
    intermediate: Array
    second_prediction: Array
    second_velocity: Array
    flow_loss: Array
    bootstrap_loss: Array
    corrupted: Array
    noise: Array
    tau: Array
    step_size: Array
    is_finest: Array
    ramp_weights: Array
    token_mask: Array
    token_weights: Array


def _validate_sequence(name: str, value: Array) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, time, features]")
    if any(dimension <= 0 for dimension in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError(f"{name} must have a floating dtype")


def _validate_k_max(k_max: int) -> int:
    if not isinstance(k_max, int) or isinstance(k_max, bool) or k_max <= 0:
        raise ValueError("k_max must be a positive integer")
    if k_max & (k_max - 1):
        raise ValueError("k_max must be a power of two")
    return k_max


def _token_column(value: Array | float, reference: Array, name: str) -> Array:
    """Broadcast a scalar or per-token value to ``[batch, time, 1]``."""

    result = jnp.asarray(value, dtype=reference.dtype)
    batch, time = reference.shape[:2]
    if result.ndim == 0:
        return jnp.broadcast_to(result, (batch, time, 1))
    if result.shape == (batch, time):
        return result[..., None]
    if result.shape == (batch, time, 1):
        return result
    if result.shape == (batch, 1):
        return jnp.broadcast_to(result[:, None, :], (batch, time, 1))
    if result.shape == (batch, 1, 1):
        return jnp.broadcast_to(result, (batch, time, 1))
    raise ValueError(
        f"{name} must be scalar, [batch, time], [batch, time, 1], "
        f"[batch, 1], or [batch, 1, 1], got {result.shape}"
    )


def _token_values(
    value: Array | float | None,
    reference: Array,
    name: str,
    *,
    default: float,
) -> Array:
    if value is None:
        return jnp.full(reference.shape[:2], default, dtype=reference.dtype)
    result = _token_column(value, reference, name)
    return result[..., 0]


def _predict_checked(
    predict_clean: PredictClean,
    state: Array,
    tau: Array,
    step_size: Array,
) -> Array:
    prediction = predict_clean(state, tau, step_size)
    if prediction.shape != state.shape:
        raise ValueError(
            "predict_clean must return the same [batch, time, features] shape "
            "as its state input"
        )
    return prediction


def sample_shortcut_schedule(
    key: Array,
    batch_size: int,
    sequence_length: int,
    *,
    k_max: int,
    dtype: jnp.dtype = jnp.float32,
    canonical_uniforms: Array | None = None,
) -> ShortcutSchedule:
    """Sample the published power-of-two schedule independently per token.

    ``k_max`` is mandatory because its Dreamer 4 training value is not
    disclosed.  Conditional on a sampled step count ``K``, signal time is
    uniform on ``{0, 1/K, ..., (K-1)/K}``, so ``tau + step_size <= 1``.
    """

    k_max = _validate_k_max(k_max)
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError("dtype must be floating")

    maximum_exponent = int(math.log2(k_max))
    if canonical_uniforms is None:
        exponent_key, grid_key = jax.random.split(key)
        exponent = jax.random.randint(
            exponent_key,
            (batch_size, sequence_length, 1),
            minval=0,
            maxval=maximum_exponent + 1,
            dtype=jnp.int32,
        )
    else:
        uniforms = jnp.asarray(canonical_uniforms, dtype=dtype)
        if uniforms.shape != (batch_size, sequence_length, 2):
            raise ValueError(
                "canonical_uniforms must have shape [batch, time, 2]"
            )
        exponent = jnp.floor(
            uniforms[..., :1] * float(maximum_exponent + 1)
        ).astype(jnp.int32)
        exponent = jnp.clip(exponent, 0, maximum_exponent)
    step_count = jnp.left_shift(jnp.ones_like(exponent), exponent)

    if canonical_uniforms is None:
        # K divides K_max for every supported K.  Sampling on the largest grid
        # and reducing modulo K is exactly uniform on the selected K-grid.
        largest_grid_index = jax.random.randint(
            grid_key,
            (batch_size, sequence_length, 1),
            minval=0,
            maxval=k_max,
            dtype=jnp.int32,
        )
        grid_index = jnp.mod(largest_grid_index, step_count)
    else:
        grid_index = jnp.floor(
            uniforms[..., 1:2] * step_count.astype(dtype)
        ).astype(jnp.int32)
        grid_index = jnp.minimum(grid_index, step_count - 1)
    step_size = jnp.reciprocal(step_count.astype(dtype))
    tau = grid_index.astype(dtype) * step_size
    return ShortcutSchedule(
        tau=tau,
        step_size=step_size,
        step_count=step_count,
        is_finest=step_count == k_max,
    )


def shortcut_schedule_is_valid(schedule: ShortcutSchedule, *, k_max: int) -> Array:
    """Return a JAX scalar checking all discrete-grid schedule invariants."""

    k_max = _validate_k_max(k_max)
    shape = schedule.tau.shape
    if (
        len(shape) != 3
        or shape[-1] != 1
        or schedule.step_size.shape != shape
        or schedule.step_count.shape != shape
        or schedule.is_finest.shape != shape
    ):
        raise ValueError("all schedule fields must share shape [batch, time, 1]")
    if not jnp.issubdtype(schedule.tau.dtype, jnp.floating):
        raise ValueError("schedule tau must have a floating dtype")

    count = schedule.step_count
    positive = count > 0
    power_of_two = jnp.bitwise_and(count, count - 1) == 0
    supported = positive & power_of_two & (count <= k_max)
    expected_step = jnp.reciprocal(count.astype(schedule.tau.dtype))
    tolerance = 8.0 * jnp.finfo(schedule.tau.dtype).eps
    grid_coordinate = schedule.tau * count.astype(schedule.tau.dtype)
    on_grid = jnp.abs(grid_coordinate - jnp.rint(grid_coordinate)) <= tolerance
    finite = jnp.isfinite(schedule.tau) & jnp.isfinite(schedule.step_size)
    endpoint = (schedule.tau >= 0.0) & (
        schedule.tau + schedule.step_size <= 1.0 + tolerance
    )
    consistent = jnp.abs(schedule.step_size - expected_step) <= tolerance
    finest = schedule.is_finest == (count == k_max)
    return jnp.all(supported & on_grid & finite & endpoint & consistent & finest)


def shortcut_forcing_loss(
    predict_clean: PredictClean,
    targets: Array,
    key: Array | None = None,
    *,
    k_max: int,
    teacher_predict_clean: PredictClean | None = None,
    intermediate_clip: float | None = None,
    support_safe_bootstrap: bool = True,
    noise: Array | None = None,
    tau: Array | float | None = None,
    step_size: Array | float | None = None,
    token_mask: Array | float | None = None,
    weights: Array | float | None = None,
    reduction: Reduction = "mean",
    return_details: bool = False,
) -> Array | ShortcutForcingLossDetails:
    """Compute the tokenwise x-prediction shortcut-forcing objective.

    For ``z_tau = (1-tau) z_0 + tau z_1``, the finest schedule level uses
    ``||f(z_tau,tau,d)-z_1||^2``.  Coarser levels distill two half steps in
    velocity space and multiply that regression by ``(1-tau)^2`` to return it
    to x-space, exactly as in Dreamer 4 Equation (7).  The two-half-step target
    is stopped in its entirety.  Finally, every token is multiplied by the
    published ramp ``0.9*tau + 0.1``.

    ``tau`` and ``step_size`` must either both be supplied or both omitted.  If
    omitted, they are sampled from :func:`sample_shortcut_schedule`.  ``key``
    is required whenever either the schedule or Gaussian noise is sampled.
    ``teacher_predict_clean`` defaults to the online predictor for compatibility;
    training loops should normally supply an exponential-moving-average teacher.
    ``intermediate_clip`` bounds only the stopped bootstrap trajectory, never the
    clean-data target.  With ``support_safe_bootstrap=True``, finest-level tokens
    remain unchanged during teacher composition so no call receives a step below
    ``1/k_max`` and those unused predictions cannot corrupt later causal context.

    In particular, the second teacher evaluation receives the complete
    ``intermediate`` sequence and midpoint times, allowing a block-causal model
    to rebuild every token's context from the updated prefix.  The callable
    must accept the half-step coordinates it is given, including
    ``1 / (2*k_max)`` at finest-level token positions in a mixed sequence.
    """

    k_max = _validate_k_max(k_max)
    _validate_sequence("targets", targets)
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(support_safe_bootstrap, bool):
        raise ValueError("support_safe_bootstrap must be boolean")
    if intermediate_clip is not None and (
        not math.isfinite(intermediate_clip) or intermediate_clip <= 0.0
    ):
        raise ValueError("intermediate_clip must be finite and positive")
    if (tau is None) != (step_size is None):
        raise ValueError("tau and step_size must both be supplied or both omitted")
    if key is None and (noise is None or tau is None):
        raise ValueError("key is required when noise or schedule is omitted")

    if key is None:
        noise_key = schedule_key = None
    else:
        noise_key, schedule_key = jax.random.split(key)

    if noise is None:
        noise = jax.random.normal(noise_key, targets.shape, dtype=targets.dtype)
    else:
        noise = jnp.asarray(noise, dtype=targets.dtype)
        if noise.shape != targets.shape:
            raise ValueError("noise must have the same shape as targets")

    if tau is None:
        schedule = sample_shortcut_schedule(
            schedule_key,
            targets.shape[0],
            targets.shape[1],
            k_max=k_max,
            dtype=targets.dtype,
        )
        tau_column = schedule.tau
        step_column = schedule.step_size
        is_finest = schedule.is_finest
    else:
        tau_column = _token_column(tau, targets, "tau")
        step_column = _token_column(step_size, targets, "step_size")
        finest_step = jnp.asarray(1.0 / k_max, dtype=targets.dtype)
        is_finest = step_column == finest_step

    corrupted = (1.0 - tau_column) * noise + tau_column * targets
    prediction = _predict_checked(
        predict_clean, corrupted, tau_column, step_column
    )

    teacher = predict_clean if teacher_predict_clean is None else teacher_predict_clean
    half_step = step_column / 2.0
    if support_safe_bootstrap:
        active_bootstrap = ~is_finest
        finest_step = jnp.asarray(1.0 / k_max, dtype=targets.dtype)
        teacher_step = jnp.where(active_bootstrap, half_step, finest_step)
    else:
        active_bootstrap = jnp.ones_like(is_finest, dtype=jnp.bool_)
        teacher_step = half_step
    first_prediction = _predict_checked(
        teacher, corrupted, tau_column, teacher_step
    )
    first_velocity = (first_prediction - corrupted) / (1.0 - tau_column)
    proposed_intermediate = corrupted + first_velocity * half_step
    if intermediate_clip is not None:
        proposed_intermediate = jnp.clip(
            proposed_intermediate, -intermediate_clip, intermediate_clip
        )
    intermediate = jnp.where(
        active_bootstrap, proposed_intermediate, corrupted
    )
    midpoint_tau = jnp.where(
        active_bootstrap, tau_column + half_step, tau_column
    )
    second_prediction = _predict_checked(
        teacher, intermediate, midpoint_tau, teacher_step
    )
    second_velocity = (second_prediction - intermediate) / (1.0 - midpoint_tau)
    bootstrap_target_velocity = jax.lax.stop_gradient(
        (first_velocity + second_velocity) / 2.0
    )
    prediction_velocity = (prediction - corrupted) / (1.0 - tau_column)
    bootstrap_target_clean = jax.lax.stop_gradient(
        corrupted + (1.0 - tau_column) * bootstrap_target_velocity
    )

    flow_loss = jnp.sum(jnp.square(prediction - targets), axis=-1)
    # This is algebraically identical to the paper's scaled velocity-space
    # expression but avoids dividing the trainable student output by 1-tau.
    bootstrap_loss = jnp.sum(
        jnp.square(prediction - bootstrap_target_clean), axis=-1
    )
    branch_loss = jnp.where(is_finest[..., 0], flow_loss, bootstrap_loss)
    ramp_weights = 0.9 * tau_column[..., 0] + 0.1
    per_token_loss = ramp_weights * branch_loss

    mask = _token_values(token_mask, targets, "token_mask", default=1.0)
    token_weights = _token_values(weights, targets, "weights", default=1.0)
    aggregation_weights = mask * token_weights
    weighted_per_token_loss = aggregation_weights * per_token_loss
    if reduction == "none":
        loss = weighted_per_token_loss
    elif reduction == "sum":
        loss = jnp.sum(weighted_per_token_loss)
    else:
        denominator = jnp.maximum(
            jnp.sum(aggregation_weights), jnp.finfo(targets.dtype).tiny
        )
        loss = jnp.sum(weighted_per_token_loss) / denominator

    if not return_details:
        return loss
    return ShortcutForcingLossDetails(
        loss=loss,
        branch_loss=branch_loss,
        per_token_loss=per_token_loss,
        weighted_per_token_loss=weighted_per_token_loss,
        prediction=prediction,
        prediction_velocity=prediction_velocity,
        bootstrap_target_velocity=bootstrap_target_velocity,
        bootstrap_target_clean=bootstrap_target_clean,
        first_prediction=first_prediction,
        first_velocity=first_velocity,
        intermediate=intermediate,
        second_prediction=second_prediction,
        second_velocity=second_velocity,
        flow_loss=flow_loss,
        bootstrap_loss=bootstrap_loss,
        corrupted=corrupted,
        noise=noise,
        tau=tau_column,
        step_size=step_column,
        is_finest=is_finest,
        ramp_weights=ramp_weights,
        token_mask=mask,
        token_weights=token_weights,
    )


def sample_shortcut_steps(
    predict_clean: PredictClean,
    noise: Array,
    *,
    steps: int,
    clean_prediction_clip: float | None = None,
) -> Array:
    """Generate a sequence using exactly ``steps`` x-prediction evaluations.

    Starting from standard-normal ``noise`` at ``tau=0``, each evaluation is
    converted to a velocity and advanced by ``d=1/steps``.  The caller samples
    the noise explicitly, which keeps this primitive deterministic and easy to
    compose with larger JAX PRNG pipelines.  ``clean_prediction_clip`` is an
    optional sampling-only safety envelope for recursively generated latent
    tokens.  It leaves the shortcut-forcing training objective unchanged.
    """

    _validate_sequence("noise", noise)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if clean_prediction_clip is not None and (
        not math.isfinite(clean_prediction_clip) or clean_prediction_clip <= 0.0
    ):
        raise ValueError("clean_prediction_clip must be finite and positive")
    step_scalar = jnp.asarray(1.0 / steps, dtype=noise.dtype)

    def sampling_step(index: int, state: Array) -> Array:
        tau_scalar = index.astype(noise.dtype) * step_scalar
        tau_column = jnp.full(
            (*noise.shape[:2], 1), tau_scalar, dtype=noise.dtype
        )
        step_column = jnp.full_like(tau_column, step_scalar)
        clean = _predict_checked(predict_clean, state, tau_column, step_column)
        if clean_prediction_clip is not None:
            clean = jnp.clip(
                clean, -clean_prediction_clip, clean_prediction_clip
            )
        velocity = (clean - state) / (1.0 - tau_column)
        return state + step_column * velocity

    return jax.lax.fori_loop(0, steps, sampling_step, noise)


__all__ = [
    "PredictClean",
    "ShortcutForcingLossDetails",
    "ShortcutSchedule",
    "sample_shortcut_schedule",
    "sample_shortcut_steps",
    "shortcut_forcing_loss",
    "shortcut_schedule_is_valid",
]
