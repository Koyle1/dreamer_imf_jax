#!/usr/bin/env python3
"""Independent acceptance checks for the trajectory iMF primitive."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

# The verifier is intentionally runnable from a source checkout without an
# editable install; this also makes the gate independent of packaging state.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from imf_dreamer_jax.imf import improved_meanflow_loss, init_imf
from imf_dreamer_jax.trajectory import (
    causal_corrupted_history_contexts,
    corrupt_trajectory,
    naive_joint_context_jvp,
    sample_trajectory_schedule,
    trajectory_imf_loss,
    trajectory_partial_jvp,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_core() -> None:
    batch, steps, features, condition_features = 3, 4, 2, 3
    params = init_imf(
        jax.random.key(101), features, condition_features, hidden_dim=9, depth=2
    )
    targets = jax.random.normal(jax.random.key(102), (batch, steps, features))
    conditions = jax.random.normal(
        jax.random.key(103), (batch, steps, condition_features)
    )
    noise = jax.random.normal(jax.random.key(104), targets.shape)
    r = jnp.linspace(0.05, 0.35, batch * steps).reshape(batch, steps, 1)
    t = jnp.linspace(0.55, 0.95, batch * steps).reshape(batch, steps, 1)
    mask = jnp.asarray(
        [[1, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=jnp.float32
    )
    weights = jnp.linspace(0.5, 1.5, batch * steps).reshape(batch, steps)

    details = trajectory_imf_loss(
        params,
        targets,
        conditions,
        jax.random.key(105),
        noise=noise,
        r=r,
        t=t,
        token_mask=mask,
        weights=weights,
        signal_weight_floor=0.7,
        signal_weight_scale=0.4,
        return_details=True,
    )
    scalar = improved_meanflow_loss(
        params,
        targets.reshape((-1, features)),
        conditions.reshape((-1, condition_features)),
        jax.random.key(105),
        noise=noise.reshape((-1, features)),
        r=r.reshape((-1, 1)),
        t=t.reshape((-1, 1)),
        weights=(mask * weights).reshape((-1,)),
        signal_weight_floor=0.7,
        signal_weight_scale=0.4,
        boundary_velocity_supervision=True,
        return_details=True,
    )
    np.testing.assert_allclose(details.loss, scalar.loss, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        details.prediction.reshape((-1, features)),
        scalar.prediction,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        details.supervised_velocity,
        details.marginal_velocity,
        rtol=0.0,
        atol=0.0,
    )
    unaggregated = trajectory_imf_loss(
        params,
        targets,
        conditions,
        jax.random.key(105),
        noise=noise,
        r=r,
        t=t,
        token_mask=mask,
        reduction="none",
    )
    require(unaggregated.shape == (batch, steps), "reduction='none' lost token axes")
    require(
        bool(jnp.all(unaggregated[mask == 0] == 0)),
        "masked tokens contribute nonzero loss",
    )

    one_token = trajectory_imf_loss(
        params,
        targets[:1, :1],
        conditions[:1, :1],
        jax.random.key(106),
        noise=noise[:1, :1],
        r=r[:1, :1],
        t=t[:1, :1],
    )
    ordinary = improved_meanflow_loss(
        params,
        targets[:1, 0],
        conditions[:1, 0],
        jax.random.key(106),
        noise=noise[:1, 0],
        r=r[:1, 0],
        t=t[:1, 0],
        boundary_velocity_supervision=True,
    )
    np.testing.assert_allclose(one_token, ordinary, rtol=2e-6, atol=2e-6)
    print("TRAJECTORY_IMF_CORE_VERIFIED")


def cross_term_params() -> dict[str, object]:
    # Input order is current z, previous z/time/valid, r, t.
    # u = z + 2 previous_z + 5 previous_t + 3 t; v = 1.
    weight = jnp.zeros((6, 2), dtype=jnp.float32)
    weight = weight.at[0, 0].set(1.0)
    weight = weight.at[1, 0].set(2.0)
    weight = weight.at[2, 0].set(5.0)
    weight = weight.at[5, 0].set(3.0)
    return {
        "layers": (
            {
                "weight": weight,
                "bias": jnp.asarray((0.0, 1.0), dtype=jnp.float32),
            },
        )
    }


def verify_jvp() -> None:
    params = cross_term_params()
    z = jnp.asarray([[[1.0], [2.0], [3.0]]], dtype=jnp.float32)
    r = jnp.zeros((1, 3, 1), dtype=jnp.float32)
    t = jnp.asarray([[[0.2], [0.4], [0.6]]], dtype=jnp.float32)
    fixed_context = causal_corrupted_history_contexts(z, t)
    correct = trajectory_partial_jvp(params, z, fixed_context, r, t)
    diagonal = naive_joint_context_jvp(params, z, None, r, t)

    np.testing.assert_allclose(correct.field, diagonal.field, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(correct.marginal_velocity, 1.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(correct.jvp, 4.0, rtol=0.0, atol=1e-6)
    expected_joint = jnp.asarray([[[4.0], [11.0], [11.0]]])
    np.testing.assert_allclose(diagonal.jvp, expected_joint, rtol=0.0, atol=1e-6)
    cross_term = diagonal.jvp - correct.jvp
    np.testing.assert_allclose(
        cross_term, jnp.asarray([[[0.0], [7.0], [7.0]]]), rtol=0.0, atol=1e-6
    )
    require(
        float(jnp.max(jnp.abs(cross_term))) > 6.0,
        "positive control failed to expose a causal-context cross term",
    )
    print("TRAJECTORY_IMF_JVP_VERIFIED")


def verify_jit() -> None:
    batch, steps, features, base_features = 2, 4, 2, 2
    targets = jax.random.normal(jax.random.key(201), (batch, steps, features))
    token_noise = jax.random.normal(jax.random.key(202), targets.shape)
    base = jax.random.normal(jax.random.key(204), (batch, steps, base_features))
    params = init_imf(
        jax.random.key(205),
        features,
        base_features + features + 2,
        hidden_dim=8,
        depth=2,
    )

    def objective(params_value, schedule):
        corrupted = corrupt_trajectory(targets, token_noise, schedule.history_t)
        conditions = causal_corrupted_history_contexts(
            corrupted, schedule.history_t, base_conditions=base
        )
        return trajectory_imf_loss(
            params_value,
            targets,
            conditions,
            jax.random.key(206),
            noise=token_noise,
            r=schedule.r,
            t=schedule.t,
            token_mask=schedule.loss_mask,
        )

    compiled = jax.jit(jax.value_and_grad(objective))
    observed = []
    for index, mode in enumerate(
        ("clean_context", "corrupted_context", "future_suffix")
    ):
        schedule = sample_trajectory_schedule(
            jax.random.key(210 + index), batch, steps, mode=mode
        )
        loss, gradient = compiled(params, schedule)
        leaves = jax.tree_util.tree_leaves(gradient)
        require(bool(jnp.isfinite(loss)), f"{mode} produced nonfinite loss")
        require(
            all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves),
            f"{mode} produced nonfinite gradients",
        )
        gradient_norm = sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves)
        require(gradient_norm > 0.0, f"{mode} produced zero parameter gradient")
        observed.append(float(loss))
    require(len(observed) == 3, "not every corruption regime was evaluated")
    print("TRAJECTORY_IMF_JIT_VERIFIED")


def verify_tests() -> None:
    test_file = PROJECT_ROOT / "tests" / "test_trajectory_imf.py"
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
    require(completed.returncode == 0, "trajectory iMF pytest suite failed")
    print("TRAJECTORY_IMF_TESTS_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--core", action="store_true")
    group.add_argument("--jvp", action="store_true")
    group.add_argument("--jit", action="store_true")
    group.add_argument("--tests", action="store_true")
    arguments = parser.parse_args()
    if arguments.core:
        verify_core()
    elif arguments.jvp:
        verify_jvp()
    elif arguments.jit:
        verify_jit()
    else:
        verify_tests()


if __name__ == "__main__":
    main()
