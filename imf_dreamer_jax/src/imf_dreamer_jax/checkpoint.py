"""Atomic checkpoint round-trips for trusted local files."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from .config import DreamerConfig
from .types import AgentState


CHECKPOINT_VERSION = 2


def save_checkpoint(
    path: str | os.PathLike[str],
    state: AgentState,
    config: DreamerConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save host arrays. Only load checkpoints from trusted sources."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "config": asdict(config),
        "state": jax.device_get(state),
        "metadata": dict(metadata or {}),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=destination.name + ".", delete=False
        ) as handle:
            temporary_name = handle.name
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_checkpoint(
    path: str | os.PathLike[str],
) -> tuple[AgentState, DreamerConfig, dict[str, Any]]:
    """Load a checkpoint created by :func:`save_checkpoint`.

    Pickle is intentionally used to retain arbitrary JAX PyTree structure;
    therefore this function must only be called on trusted local artifacts.
    """

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    config = DreamerConfig(**payload["config"])
    state = jax.tree_util.tree_map(jnp.asarray, payload["state"])
    return state, config, dict(payload.get("metadata", {}))
