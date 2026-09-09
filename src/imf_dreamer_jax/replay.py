"""Host-side episode-safe sequence replay feeding device JAX arrays."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class _ReplayItem:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    continuation: float
    is_first: bool


class SequenceReplayBuffer:
    """Bounded replay buffer that never samples across episode boundaries."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_dim: int,
        *,
        seed: int = 0,
    ) -> None:
        if capacity <= 0 or action_dim <= 0:
            raise ValueError("capacity and action_dim must be positive")
        if not observation_shape or any(value <= 0 for value in observation_shape):
            raise ValueError("observation_shape must be nonempty and positive")
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)
        self.action_dim = int(action_dim)
        self._items: deque[_ReplayItem] = deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(
        self,
        observation: object,
        action: object,
        reward: float,
        continuation: float,
        *,
        is_first: bool = False,
    ) -> None:
        observation_value = np.asarray(observation)
        action_value = np.asarray(action, dtype=np.float32)
        if observation_value.shape != self.observation_shape:
            raise ValueError("observation has the wrong shape")
        if action_value.shape != (self.action_dim,):
            raise ValueError("action has the wrong shape")
        if not np.isfinite(observation_value).all() or not np.isfinite(action_value).all():
            raise ValueError("replay values must be finite")
        if not np.isfinite(reward) or not np.isfinite(continuation):
            raise ValueError("reward and continuation must be finite")
        if not 0 <= continuation <= 1:
            raise ValueError("continuation must lie in [0, 1]")
        self._items.append(
            _ReplayItem(
                observation_value.copy(),
                action_value.copy(),
                float(reward),
                float(continuation),
                bool(is_first),
            )
        )

    def _valid_starts(self, sequence_length: int) -> list[int]:
        items = list(self._items)
        return [
            start
            for start in range(len(items) - sequence_length + 1)
            if not any(
                items[index].is_first or items[index - 1].continuation == 0.0
                for index in range(start + 1, start + sequence_length)
            )
        ]

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        device: jax.Device | None = None,
    ) -> Mapping[str, jax.Array]:
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        items = list(self._items)
        valid = self._valid_starts(sequence_length)
        if not valid:
            raise ValueError("replay contains no episode-safe sequence of this length")
        starts = self._rng.choice(valid, size=batch_size, replace=True)
        rows = [[items[int(start) + offset] for offset in range(sequence_length)] for start in starts]
        arrays = {
            "observations": jnp.asarray(
                np.stack([[item.observation for item in row] for row in rows])
            ),
            "actions": jnp.asarray(
                np.stack([[item.action for item in row] for row in rows]), dtype=jnp.float32
            ),
            "rewards": jnp.asarray(
                [[item.reward for item in row] for row in rows], dtype=jnp.float32
            ),
            "continuations": jnp.asarray(
                [[item.continuation for item in row] for row in rows], dtype=jnp.float32
            ),
        }
        if device is not None:
            arrays = jax.tree_util.tree_map(lambda value: jax.device_put(value, device), arrays)
        return arrays

