"""Frozen conditional-source iMF. No density or final-distribution KL is claimed.

For fixed conditioning h, x_t=(1-t)*posterior+t*base and v=base-posterior.
The existing MeanFlow identity is unchanged; only its source distribution differs.
Coordinates are standardized with train-only constants by the caller.
"""

import jax
import jax.numpy as jnp

from .imf import init_imf, improved_meanflow_loss, sample_imf_steps


def init(key, sample_dim, condition_dim, hidden_dim=256, depth=2):
    params = init_imf(key, sample_dim, condition_dim, hidden_dim, depth)
    layers = list(params["layers"])
    layers[-1] = jax.tree.map(jnp.zeros_like, layers[-1])
    return dict(layers=tuple(layers))


def base_sample(mean, std, noise):
    if mean.ndim != 2 or mean.shape != std.shape or mean.shape != noise.shape:
        raise ValueError("mean, std and noise must be matching matrices")
    return mean + std * noise


def sample(params, condition, mean, std, noise, *, steps=1, bypass=False):
    """Identity at initialization; bypass never evaluates the flow network."""
    base = base_sample(mean, std, noise)
    if bypass:
        return base
    return sample_imf_steps(params, condition, noise=base, steps=steps)


def loss(params, target, condition, mean, std, key, *, noise=None, **kwargs):
    """Only correction parameters receive gradients, including through the source.

    Target and source samples are independently coupled conditional on h.
    Endpoint regression is intentionally disabled: a paired endpoint MSE would
    favor the posterior mean rather than its predictive distribution.
    """
    target, condition, mean, std = jax.tree.map(
        jax.lax.stop_gradient, (target, condition, mean, std)
    )
    source_key, objective_key = jax.random.split(key)
    if noise is None:
        noise = jax.random.normal(source_key, target.shape, dtype=target.dtype)
    source = jax.lax.stop_gradient(base_sample(mean, std, noise))
    if "endpoint_scale" in kwargs or "shortcut_scale" in kwargs:
        raise ValueError("endpoint/shortcut penalties are not part of this diagnostic")
    return improved_meanflow_loss(
        params,
        target,
        condition,
        objective_key,
        noise=source,
        boundary_velocity_supervision=True,
        **kwargs
    )
