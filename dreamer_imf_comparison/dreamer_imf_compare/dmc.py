"""Minimal deterministic DeepMind Control adapter shared by compact arms."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np


def split_task(task: str) -> tuple[str, str]:
    if not task.startswith("dmc_"):
        raise ValueError("task must begin with dmc_")
    body = task[4:]
    if body.startswith("ball_in_cup_"):
        return "ball_in_cup", body[len("ball_in_cup_") :]
    domain, separator, task_name = body.partition("_")
    if not separator or not domain or not task_name:
        raise ValueError(f"malformed DMC task {task!r}")
    return domain, task_name


def flatten_observation(observation: Mapping[str, Any]) -> np.ndarray:
    if not observation:
        raise ValueError("observation mapping cannot be empty")
    pieces = []
    for name in sorted(observation):
        value = np.asarray(observation[name], dtype=np.float32).reshape(-1)
        if not np.isfinite(value).all():
            raise ValueError(f"observation {name!r} is non-finite")
        pieces.append(value)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def environment_seed(train_seed: int, index: int = 0, *, offset: int = 0) -> int:
    """Match the pinned DreamerV3 environment seed formula exactly."""

    if train_seed < 0 or index < 0 or offset < 0:
        raise ValueError("seed components must be nonnegative")
    return hash((int(train_seed) + int(offset), int(index))) % (2**32 - 1)


def _load_standard_domain(
    domain: str, task_name: str, random_state: np.random.RandomState
) -> Any:
    """Load one suite domain without importing every optional DMC domain."""

    import dm_control

    suite_name = "dm_control.suite"
    suite_package = sys.modules.get(suite_name)
    if suite_package is None:
        dm_spec = importlib.util.find_spec("dm_control")
        if dm_spec is None or not dm_spec.submodule_search_locations:
            raise ImportError("cannot locate dm_control package")
        suite_root = Path(next(iter(dm_spec.submodule_search_locations))) / "suite"
        suite_package = types.ModuleType(suite_name)
        suite_package.__path__ = [str(suite_root)]  # type: ignore[attr-defined]
        suite_package.__package__ = suite_name
        suite_spec = importlib.machinery.ModuleSpec(
            suite_name, loader=None, is_package=True
        )
        suite_spec.submodule_search_locations = [str(suite_root)]
        suite_package.__spec__ = suite_spec
        sys.modules[suite_name] = suite_package
        setattr(dm_control, "suite", suite_package)
    domain_module = importlib.import_module(f"{suite_name}.{domain}")
    if task_name not in domain_module.SUITE:
        raise ValueError(f"task {task_name!r} does not exist in domain {domain!r}")
    return domain_module.SUITE[task_name](random=random_state)


@dataclass(frozen=True)
class Step:
    observation: np.ndarray
    reward: float
    continuation: float
    is_last: bool


@dataclass(frozen=True)
class DMCSnapshot:
    """State required to replay a transition without advancing an environment.

    Snapshots are process-local and adapter-specific.  They are intentionally
    not serialized: their sole purpose is paired finite differences with an
    exact restore before every perturbation.
    """

    adapter_identity: int
    physics_state: np.ndarray
    step_count: int
    reset_next_step: bool
    task_random_state: object | None


class DMCAdapter:
    """Native-termination proprioceptive DMC with normalized actions."""

    def __init__(self, task: str, *, seed: int, action_repeat: int = 1) -> None:
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        # This comparison is proprioceptive. Disabling OpenGL prevents GLFW
        # from requiring the macOS main thread and removes rendering overhead.
        os.environ.setdefault("MUJOCO_GL", "disable")
        domain, task_name = split_task(task)
        self.task = task
        self.seed = int(seed)
        self.action_repeat = int(action_repeat)
        self._environment = _load_standard_domain(
            domain, task_name, np.random.RandomState(self.seed)
        )
        self._action_spec = self._environment.action_spec()
        self.action_shape = tuple(self._action_spec.shape)
        if len(self.action_shape) != 1:
            raise ValueError("comparison requires a flat continuous action space")
        observation_spec = self._environment.observation_spec()
        self.observation_shape = (
            sum(int(np.prod(observation_spec[name].shape)) for name in sorted(observation_spec)),
        )

    @property
    def action_dim(self) -> int:
        return self.action_shape[0]

    def reset(self) -> np.ndarray:
        return flatten_observation(self._environment.reset().observation)

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != self.action_shape:
            raise ValueError(f"action must have shape {self.action_shape}")
        normalized = np.clip(normalized, -1.0, 1.0)
        minimum = np.asarray(self._action_spec.minimum, dtype=np.float32)
        maximum = np.asarray(self._action_spec.maximum, dtype=np.float32)
        return minimum + (normalized + 1.0) * 0.5 * (maximum - minimum)

    def step(self, action: np.ndarray) -> Step:
        native_action = self._denormalize_action(action)
        reward = 0.0
        continuation = 1.0
        time_step = None
        for _ in range(self.action_repeat):
            time_step = self._environment.step(native_action)
            reward += float(time_step.reward or 0.0)
            continuation *= float(1.0 if time_step.discount is None else time_step.discount)
            if time_step.last():
                break
        assert time_step is not None
        return Step(
            flatten_observation(time_step.observation),
            reward,
            continuation,
            bool(time_step.last()),
        )

    def snapshot(self) -> DMCSnapshot:
        """Capture physics, time-limit, reset, and task-RNG state."""

        task_random = getattr(self._environment.task, "random", None)
        random_state = None
        if task_random is not None and hasattr(task_random, "get_state"):
            random_state = copy.deepcopy(task_random.get_state())
        return DMCSnapshot(
            adapter_identity=id(self),
            physics_state=np.asarray(self._environment.physics.get_state()).copy(),
            step_count=int(getattr(self._environment, "_step_count", 0)),
            reset_next_step=bool(getattr(self._environment, "_reset_next_step", False)),
            task_random_state=random_state,
        )

    def restore(self, snapshot: DMCSnapshot) -> None:
        """Restore an exact snapshot captured from this adapter."""

        if not isinstance(snapshot, DMCSnapshot) or snapshot.adapter_identity != id(self):
            raise ValueError("DMC snapshot belongs to a different adapter")
        physics = self._environment.physics
        with physics.reset_context():
            physics.set_state(np.asarray(snapshot.physics_state).copy())
        if hasattr(self._environment, "_step_count"):
            self._environment._step_count = snapshot.step_count
        if hasattr(self._environment, "_reset_next_step"):
            self._environment._reset_next_step = snapshot.reset_next_step
        task_random = getattr(self._environment.task, "random", None)
        if snapshot.task_random_state is not None:
            if task_random is None or not hasattr(task_random, "set_state"):
                raise RuntimeError("DMC task RNG cannot be restored safely")
            task_random.set_state(copy.deepcopy(snapshot.task_random_state))

    def close(self) -> None:
        self._environment.close()


__all__ = [
    "DMCAdapter",
    "DMCSnapshot",
    "Step",
    "environment_seed",
    "flatten_observation",
    "split_task",
]
