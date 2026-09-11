#!/usr/bin/env python3
"""Independent acceptance checks for x-prediction shortcut forcing."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from imf_dreamer_jax.shortcut import (
    sample_shortcut_schedule,
    sample_shortcut_steps,
    shortcut_forcing_loss,
    shortcut_schedule_is_valid,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def predict_jax(z: jax.Array, tau: jax.Array, step: jax.Array) -> jax.Array:
    feature = 0.13 * jnp.arange(z.shape[-1], dtype=z.dtype)
    previous = jnp.concatenate((jnp.zeros_like(z[:, :1]), z[:, :-1]), axis=1)
    previous_tau = jnp.concatenate(
        (jnp.zeros_like(tau[:, :1]), tau[:, :-1]), axis=1
    )
    return (
        0.55 * z
        + 0.21 * previous
        + 0.17 * tau
        + 0.09 * previous_tau
        - 0.2 * step
        + feature
    )


def predict_numpy(z: np.ndarray, tau: np.ndarray, step: np.ndarray) -> np.ndarray:
    feature = 0.13 * np.arange(z.shape[-1], dtype=z.dtype)
    previous = np.concatenate((np.zeros_like(z[:, :1]), z[:, :-1]), axis=1)
    previous_tau = np.concatenate(
        (np.zeros_like(tau[:, :1]), tau[:, :-1]), axis=1
    )
    return (
        0.55 * z
        + 0.21 * previous
        + 0.17 * tau
        + 0.09 * previous_tau
        - 0.2 * step
        + feature
    )


def numpy_equation_seven(
    targets: np.ndarray,
    noise: np.ndarray,
    tau: np.ndarray,
    step: np.ndarray,
    k_max: int,
) -> dict[str, np.ndarray]:
    corrupted = (1.0 - tau) * noise + tau * targets
    prediction = predict_numpy(corrupted, tau, step)
    half = step / 2.0
    first_prediction = predict_numpy(corrupted, tau, half)
    first_velocity = (first_prediction - corrupted) / (1.0 - tau)
    intermediate = corrupted + first_velocity * half
    midpoint = tau + half
    second_prediction = predict_numpy(intermediate, midpoint, half)
    second_velocity = (second_prediction - intermediate) / (1.0 - midpoint)
    teacher = (first_velocity + second_velocity) / 2.0
    predicted_velocity = (prediction - corrupted) / (1.0 - tau)
    flow = np.sum(np.square(prediction - targets), axis=-1)
    bootstrap = np.square(1.0 - tau[..., 0]) * np.sum(
        np.square(predicted_velocity - teacher), axis=-1
    )
    branch = np.where(step[..., 0] == 1.0 / k_max, flow, bootstrap)
    ramp = 0.9 * tau[..., 0] + 0.1
    return {
        "corrupted": corrupted,
        "prediction": prediction,
        "first_velocity": first_velocity,
        "intermediate": intermediate,
        "second_prediction": second_prediction,
        "second_velocity": second_velocity,
        "teacher": teacher,
        "flow": flow,
        "bootstrap": bootstrap,
        "branch": branch,
        "weighted": ramp * branch,
    }


def fixtures() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    targets = jnp.asarray(
        [[[0.2, -0.4], [1.0, 0.3], [-0.1, 0.8], [0.7, -0.2]]],
        dtype=jnp.float32,
    )
    noise = jnp.asarray(
        [[[0.9, 0.5], [-0.6, 0.2], [0.3, -0.8], [0.1, 0.4]]],
        dtype=jnp.float32,
    )
    tau = jnp.asarray([[0.50, 0.50, 0.00, 0.75]], dtype=jnp.float32)[..., None]
    step = jnp.asarray([[0.25, 0.50, 1.00, 0.25]], dtype=jnp.float32)[..., None]
    return targets, noise, tau, step


def verify_schedule() -> None:
    signature = inspect.signature(sample_shortcut_schedule)
    require(
        signature.parameters["k_max"].default is inspect.Parameter.empty,
        "undisclosed k_max acquired a silent default",
    )
    schedule = jax.jit(
        lambda key: sample_shortcut_schedule(key, 128, 32, k_max=16)
    )(jax.random.key(701))
    require(bool(shortcut_schedule_is_valid(schedule, k_max=16)), "valid schedule rejected")
    counts = np.asarray(schedule.step_count)
    require(set(np.unique(counts)) == {1, 2, 4, 8, 16}, "power-of-two levels missing")
    require(
        bool(jnp.all(schedule.tau + schedule.step_size <= 1.0)),
        "schedule crosses the data endpoint",
    )
    require(
        np.array_equal(np.asarray(schedule.is_finest), counts == 16),
        "finest branch marker is inconsistent",
    )

    off_grid = schedule._replace(tau=schedule.tau.at[0, 0, 0].set(0.3))
    require(
        not bool(shortcut_schedule_is_valid(off_grid, k_max=16)),
        "off-grid positive control was accepted",
    )
    bad_count = schedule._replace(
        step_count=schedule.step_count.at[0, 0, 0].set(3)
    )
    require(
        not bool(shortcut_schedule_is_valid(bad_count, k_max=16)),
        "non-power-of-two positive control was accepted",
    )
    try:
        sample_shortcut_schedule(jax.random.key(0), 1, 1, k_max=3)
    except ValueError:
        pass
    else:
        raise RuntimeError("non-power-of-two k_max was accepted")
    print("SHORTCUT_SCHEDULE_VERIFIED")


def verify_equation() -> None:
    targets, noise, tau, step = fixtures()
    details = shortcut_forcing_loss(
        predict_jax,
        targets,
        k_max=4,
        noise=noise,
        tau=tau,
        step_size=step,
        reduction="none",
        return_details=True,
    )
    reference = numpy_equation_seven(
        np.asarray(targets), np.asarray(noise), np.asarray(tau), np.asarray(step), 4
    )
    comparisons = (
        (details.corrupted, reference["corrupted"]),
        (details.prediction, reference["prediction"]),
        (details.first_velocity, reference["first_velocity"]),
        (details.intermediate, reference["intermediate"]),
        (details.second_prediction, reference["second_prediction"]),
        (details.second_velocity, reference["second_velocity"]),
        (details.bootstrap_target_velocity, reference["teacher"]),
        (details.flow_loss, reference["flow"]),
        (details.bootstrap_loss, reference["bootstrap"]),
        (details.branch_loss, reference["branch"]),
        (details.loss, reference["weighted"]),
    )
    for actual, expected in comparisons:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    require(bool(jnp.any(details.is_finest)), "finest branch was not exercised")
    require(bool(jnp.any(~details.is_finest)), "bootstrap branch was not exercised")

    # The second call must rebuild causal prefix features from z' and midpoint
    # times.  This stale-context construction is a deliberate negative oracle.
    old_z = np.asarray(details.corrupted)
    old_tau = np.asarray(tau)
    new_z = np.asarray(details.intermediate)
    midpoint = old_tau + np.asarray(step) / 2.0
    old_previous = np.concatenate((np.zeros_like(old_z[:, :1]), old_z[:, :-1]), axis=1)
    old_previous_tau = np.concatenate(
        (np.zeros_like(old_tau[:, :1]), old_tau[:, :-1]), axis=1
    )
    feature = 0.13 * np.arange(new_z.shape[-1], dtype=new_z.dtype)
    stale_second = (
        0.55 * new_z
        + 0.21 * old_previous
        + 0.17 * midpoint
        + 0.09 * old_previous_tau
        - 0.2 * np.asarray(step) / 2.0
        + feature
    )
    require(
        float(np.max(np.abs(reference["second_prediction"] - stale_second))) > 1e-3,
        "stale-context negative control did not differ",
    )
    no_x_scaling = np.sum(
        np.square(
            np.asarray(details.prediction_velocity)
            - np.asarray(details.bootstrap_target_velocity)
        ),
        axis=-1,
    )
    bootstrap_tokens = ~np.asarray(details.is_finest[..., 0])
    require(
        float(
            np.max(
                np.abs(
                    np.asarray(details.bootstrap_loss)[bootstrap_tokens]
                    - no_x_scaling[bootstrap_tokens]
                )
            )
        )
        > 1e-3,
        "missing x-space scaling negative control did not differ",
    )
    print("SHORTCUT_EQUATION_VERIFIED")


def verify_controls() -> None:
    targets, noise, tau, step = fixtures()
    mask = jnp.asarray([[1.0, 0.0, 1.0, 1.0]], dtype=jnp.float32)
    weights = jnp.asarray([[1.0, 9.0, 2.0, 3.0]], dtype=jnp.float32)

    def objective(scale: jax.Array):
        def predictor(z: jax.Array, t: jax.Array, d: jax.Array) -> jax.Array:
            return scale * predict_jax(z, t, d)

        return shortcut_forcing_loss(
            predictor,
            targets,
            k_max=4,
            noise=noise,
            tau=tau,
            step_size=step,
            token_mask=mask,
            weights=weights,
            reduction="none",
            return_details=True,
        )

    details, derivative = jax.jit(jax.jvp, static_argnums=(0,))(
        objective, (jnp.asarray(0.8),), (jnp.asarray(1.0),)
    )
    require(bool(jnp.all(jnp.isfinite(derivative.loss))), "jitted derivative is nonfinite")
    require(
        bool(jnp.all(details.loss[mask == 0.0] == 0.0)),
        "masked token contributes loss",
    )
    require(
        bool(jnp.all(derivative.loss[mask == 0.0] == 0.0)),
        "masked token contributes a direct loss derivative",
    )
    expected_ramp = 0.9 * tau[..., 0] + 0.1
    np.testing.assert_allclose(details.ramp_weights, expected_ramp, atol=1e-7)
    require(
        float(jnp.max(jnp.abs(details.per_token_loss - details.branch_loss))) > 1e-4,
        "ramp negative control did not differ",
    )

    # Isolate stop-gradient with a scalar causal predictor and compare against
    # both the correct and deliberately unstopped reference derivatives.
    bootstrap_tau = jnp.zeros((1, 2, 1), dtype=jnp.float32)
    bootstrap_step = jnp.ones((1, 2, 1), dtype=jnp.float32)
    small_targets = targets[:, :2, :1]
    small_noise = noise[:, :2, :1]

    def parameterized(parameter, z, t, d):
        return parameter * (0.7 * z + 0.2 * t + 0.3 * d + 0.5)

    def actual_loss(parameter):
        return shortcut_forcing_loss(
            lambda z, t, d: parameterized(parameter, z, t, d),
            small_targets,
            k_max=2,
            noise=small_noise,
            tau=bootstrap_tau,
            step_size=bootstrap_step,
        )

    def reference_loss(parameter, stop_teacher):
        z = small_noise
        half = bootstrap_step / 2.0
        b1 = (parameterized(parameter, z, bootstrap_tau, half) - z) / (
            1.0 - bootstrap_tau
        )
        z_prime = z + half * b1
        midpoint = bootstrap_tau + half
        b2 = (parameterized(parameter, z_prime, midpoint, half) - z_prime) / (
            1.0 - midpoint
        )
        teacher = (b1 + b2) / 2.0
        if stop_teacher:
            teacher = jax.lax.stop_gradient(teacher)
        predicted = (
            parameterized(parameter, z, bootstrap_tau, bootstrap_step) - z
        ) / (1.0 - bootstrap_tau)
        return 0.1 * jnp.mean(jnp.square(predicted - teacher))

    point = jnp.asarray(0.8)
    actual_gradient = jax.jit(jax.grad(actual_loss))(point)
    stopped_gradient = jax.grad(lambda p: reference_loss(p, True))(point)
    unstopped_gradient = jax.grad(lambda p: reference_loss(p, False))(point)
    np.testing.assert_allclose(actual_gradient, stopped_gradient, rtol=2e-6, atol=2e-6)
    require(
        float(jnp.abs(actual_gradient - unstopped_gradient)) > 1e-3,
        "unstopped-teacher negative control did not differ",
    )

    sampler_noise = jnp.asarray([[[0.2], [-0.4], [0.7]]], dtype=jnp.float32)

    def sampler_predict(z, t, d):
        return 0.3 * z + 0.25 * t + 0.15 * d + 0.2

    for count in (1, 2, 4):
        actual = jax.jit(
            lambda value, steps=count: sample_shortcut_steps(
                sampler_predict, value, steps=steps
            )
        )(sampler_noise)
        expected = np.asarray(sampler_noise)
        d = 1.0 / count
        for index in range(count):
            signal = index * d
            clean = 0.3 * expected + 0.25 * signal + 0.15 * d + 0.2
            expected = expected + d * (clean - expected) / (1.0 - signal)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        if count > 1:
            wrong = np.asarray(sampler_noise)
            for index in range(count):
                signal = index * d
                clean = 0.3 * wrong + 0.25 * signal + 0.15 * d + 0.2
                wrong = wrong + d * (clean - wrong)
            require(
                float(np.max(np.abs(expected - wrong))) > 1e-3,
                f"{count}-step no-x-to-v negative control did not differ",
            )
    print("SHORTCUT_CONTROLS_VERIFIED")


def verify_tests() -> None:
    test_file = PROJECT_ROOT / "tests" / "test_shortcut_forcing.py"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    require(completed.returncode == 0, "shortcut-forcing pytest suite failed")
    print("SHORTCUT_TESTS_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--schedule", action="store_true")
    group.add_argument("--equation", action="store_true")
    group.add_argument("--controls", action="store_true")
    group.add_argument("--tests", action="store_true")
    arguments = parser.parse_args()
    if arguments.schedule:
        verify_schedule()
    elif arguments.equation:
        verify_equation()
    elif arguments.controls:
        verify_controls()
    else:
        verify_tests()


if __name__ == "__main__":
    main()
