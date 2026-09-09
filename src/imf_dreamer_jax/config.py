"""Validated static configurations for the JAX implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


PriorKind = Literal["gaussian", "imf"]


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class DreamerConfig:
    """Dimensions, objectives, and optimizer settings.

    The object is frozen and hashable so it can be passed as a static argument
    to :func:`jax.jit`. Set ``prior='imf'`` and ``overshooting_horizon=1`` for
    one-step iMF, or raise the overshooting horizon for the multistep variant.
    """

    observation_shape: tuple[int, ...] = (8,)
    action_dim: int = 2
    deterministic_dim: int = 64
    stochastic_dim: int = 16
    embedding_dim: int = 64
    hidden_dim: int = 64
    prior: PriorKind = "imf"
    min_std: float = 0.1
    max_std: float = 2.0
    kl_free_nats: float = 0.0
    prior_scale: float = 1.0
    representation_scale: float = 0.05
    reconstruction_scale: float = 1.0
    reward_scale: float = 1.0
    continuation_scale: float = 1.0
    overshooting_horizon: int = 1
    overshooting_scale: float = 1.0
    imf_boundary_fraction: float = 0.1
    imagination_horizon: int = 5
    discount: float = 0.99
    lambda_: float = 0.95
    actor_entropy_scale: float = 1e-3
    model_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    grad_clip: float = 100.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.observation_shape, tuple) or not self.observation_shape:
            raise ValueError("observation_shape must be a nonempty tuple")
        for entry in self.observation_shape:
            _positive_int("observation_shape entry", entry)
        for name in (
            "action_dim",
            "deterministic_dim",
            "stochastic_dim",
            "embedding_dim",
            "hidden_dim",
            "overshooting_horizon",
            "imagination_horizon",
        ):
            _positive_int(name, getattr(self, name))
        if self.prior not in ("gaussian", "imf"):
            raise ValueError("prior must be 'gaussian' or 'imf'")
        if not 0.0 < self.min_std <= self.max_std:
            raise ValueError("standard deviations must satisfy 0 < min_std <= max_std")
        if self.kl_free_nats < 0:
            raise ValueError("kl_free_nats must be nonnegative")
        for name in (
            "prior_scale",
            "representation_scale",
            "reconstruction_scale",
            "reward_scale",
            "continuation_scale",
            "overshooting_scale",
            "actor_entropy_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.imf_boundary_fraction <= 1:
            raise ValueError("imf_boundary_fraction must be in [0, 1]")
        if not 0 <= self.discount <= 1 or not 0 <= self.lambda_ <= 1:
            raise ValueError("discount and lambda_ must be in [0, 1]")
        for name in (
            "model_learning_rate",
            "actor_learning_rate",
            "critic_learning_rate",
            "grad_clip",
            "adam_epsilon",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("adam_beta1", "adam_beta2"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")

    @property
    def observation_dim(self) -> int:
        return math.prod(self.observation_shape)

    @property
    def feature_dim(self) -> int:
        return self.deterministic_dim + self.stochastic_dim

    @property
    def method_name(self) -> str:
        if self.prior == "gaussian":
            return "gaussian_rssm"
        return "imf_rssm_multistep" if self.overshooting_horizon > 1 else "imf_rssm_one_step"


@dataclass(frozen=True)
class PlannerConfig:
    """Static settings for common-random-number CEM planning."""

    horizon: int = 5
    population: int = 256
    elite_count: int = 32
    iterations: int = 4
    noise_samples: int = 8
    discount: float = 0.99
    risk_beta: float = 1.0
    behavior_weight: float = 0.02
    epistemic_threshold: float = math.inf
    terminal_value: bool = True
    min_std: float = 0.05
    max_std: float = 1.0

    def __post_init__(self) -> None:
        for name in ("horizon", "population", "elite_count", "iterations", "noise_samples"):
            _positive_int(name, getattr(self, name))
        if self.elite_count > self.population:
            raise ValueError("elite_count cannot exceed population")
        if not 0 <= self.discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if self.risk_beta < 0 or self.behavior_weight < 0:
            raise ValueError("planner penalties must be nonnegative")
        if self.epistemic_threshold < 0:
            raise ValueError("epistemic_threshold must be nonnegative")
        if not 0 < self.min_std <= self.max_std:
            raise ValueError("planner std bounds are invalid")

