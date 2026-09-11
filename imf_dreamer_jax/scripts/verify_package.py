#!/usr/bin/env python3
"""Verify packaging, dependency isolation, public exports, and quickstart."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import tomllib

import jax
import jax.numpy as jnp

import imf_dreamer_jax as library


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = " ".join(metadata["project"]["dependencies"]).lower()
    assert "torch" not in dependencies
    assert "flax" not in dependencies
    assert "optax" not in dependencies
    source = "\n".join(
        path.read_text() for path in (ROOT / "src" / "imf_dreamer_jax").rglob("*.py")
    )
    assert "import torch" not in source and "from torch" not in source
    assert importlib.metadata.version("imf-dreamer-jax") == library.__version__
    for public_name in (
        "DreamerConfig",
        "create_agent",
        "jit_train_world_model",
        "jit_train_actor_critic",
        "jit_cem_plan",
        "save_checkpoint",
    ):
        assert public_name in library.__all__ and hasattr(library, public_name)

    config = library.DreamerConfig(
        observation_shape=(3,),
        action_dim=1,
        deterministic_dim=4,
        stochastic_dim=2,
        embedding_dim=4,
        hidden_dim=6,
        prior="imf",
        overshooting_horizon=2,
        imagination_horizon=2,
    )
    init_key, train_key, act_key = jax.random.split(jax.random.key(101), 3)
    agent = library.create_agent(config, init_key)
    batch = {
        "observations": jnp.zeros((2, 4, 3), dtype=jnp.float32),
        "actions": jnp.zeros((2, 4, 1), dtype=jnp.float32),
        "rewards": jnp.zeros((2, 4), dtype=jnp.float32),
        "continuations": jnp.ones((2, 4), dtype=jnp.float32),
    }
    agent, losses = library.jit_train_world_model(agent, batch, train_key, config)
    assert bool(jnp.isfinite(losses.total))
    belief = library.initial_state(config, 1)
    action, _ = library.jit_act(
        agent.params,
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        belief,
        act_key,
        config,
        deterministic=True,
    )
    assert action.shape == (1, 1)
    readme = (ROOT / "README.md").read_text()
    assert "## Quickstart" in readme and "compact" in readme
    print("PACKAGE_VERIFY_OK")


if __name__ == "__main__":
    main()
