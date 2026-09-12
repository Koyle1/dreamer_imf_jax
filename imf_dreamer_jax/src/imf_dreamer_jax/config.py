"""Validated static configurations for the JAX implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .fidelity import IMFNoiseCoupling


PriorKind = Literal["gaussian", "imf", "shortcut"]
ActorGradient = Literal["reinforce", "dynamics", "pmpo"]
ImaginationStartMode = Literal["all", "one_per_sequence"]
RewardLoss = Literal["mse", "binary_cross_entropy"]


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class DreamerConfig:
    """Dimensions, objectives, and optimizer settings.

    The object is frozen and hashable so it can be passed as a static argument
    to :func:`jax.jit`. ``overshooting_distances`` declares the explicit
    multistep consistency distances; ``overshooting_horizon`` is retained for
    checkpoint compatibility.
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
    reward_min: float | None = None
    reward_max: float | None = None
    reward_initial_value: float = 0.0
    reward_output_init_scale: float = 0.0
    reward_loss: RewardLoss = "mse"
    continuation_scale: float = 1.0
    overshooting_horizon: int = 1
    burn_in: int = 0
    overshooting_distances: tuple[int, ...] = (1,)
    overshooting_scale: float = 1.0
    imf_boundary_fraction: float = 0.5
    imf_time_mean: float = -0.4
    imf_time_std: float = 1.0
    imf_adaptive_power: float = 1.0
    imf_adaptive_epsilon: float = 0.01
    imf_meanflow_scale: float = 1.0
    imf_velocity_scale: float = 1.0
    imf_noise_coupling: IMFNoiseCoupling = "independent"
    imf_endpoint_scale: float = 0.0
    imf_shortcut_scale: float = 0.0
    imf_signal_weight_floor: float = 1.0
    imf_signal_weight_scale: float = 0.0
    imf_sampling_steps: int = 1
    imf_condition_gradient_scale: float = 1.0
    imf_boundary_velocity_supervision: bool = False
    imf_trajectory_enabled: bool = False
    imf_trajectory_clean_probability: float = 1.0 / 3.0
    imf_trajectory_corrupted_probability: float = 1.0 / 3.0
    imf_trajectory_suffix_probability: float = 1.0 / 3.0
    imf_trajectory_history_noise_max: float = 1.0
    imf_causal_consistency_scale: float = 0.0
    imf_causal_reward_scale: float = 1.0
    imf_causal_huber_delta: float = 1.0
    imf_causal_normalization_epsilon: float = 1e-3
    shortcut_training_k_max: int | None = None
    shortcut_sampling_steps: int = 4
    shortcut_bootstrap_ema_decay: float | None = 0.999
    shortcut_intermediate_clip: float | None = 4.0
    shortcut_support_safe_bootstrap: bool = True
    shortcut_sampling_clip: float | None = None
    imagination_horizon: int = 5
    discount: float = 0.99
    lambda_: float = 0.95
    actor_entropy_scale: float = 1e-3
    actor_init_scale: float = 0.01
    actor_mean_bound: float = 5.0
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0
    actor_gradient: ActorGradient = "reinforce"
    actor_action_l2_scale: float = 1e-3
    behavior_cloning_learning_rate: float = 3e-4
    behavior_kl_scale: float = 0.0
    pmpo_positive_temperature: float = 1.0
    pmpo_negative_temperature: float = 1.0
    pmpo_positive_weight: float = 1.0
    pmpo_negative_weight: float = 1.0
    imagination_start_mode: ImaginationStartMode = "all"
    critic_output_init_scale: float = 0.0
    critic_bins: int = 1
    critic_symlog_min: float = -20.0
    critic_symlog_max: float = 20.0
    actor_critic_warmup_steps: int = 0
    slow_critic_fraction: float = 0.02
    return_normalization_epsilon: float = 1e-6
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
            "critic_bins",
            "imf_sampling_steps",
            "shortcut_sampling_steps",
        ):
            _positive_int(name, getattr(self, name))
        if (
            not isinstance(self.burn_in, int)
            or isinstance(self.burn_in, bool)
            or self.burn_in < 0
        ):
            raise ValueError(f"burn_in must be a nonnegative integer, got {self.burn_in!r}")
        if (
            not isinstance(self.overshooting_distances, tuple)
            or not self.overshooting_distances
        ):
            raise ValueError("overshooting_distances must be a nonempty tuple")
        for distance in self.overshooting_distances:
            _positive_int("overshooting_distances entry", distance)
        if len(set(self.overshooting_distances)) != len(self.overshooting_distances):
            raise ValueError("overshooting_distances must not contain duplicates")
        if self.prior not in ("gaussian", "imf", "shortcut"):
            raise ValueError("prior must be 'gaussian', 'imf', or 'shortcut'")
        if self.prior == "shortcut":
            if self.shortcut_training_k_max is None:
                raise ValueError(
                    "shortcut prior requires an explicit shortcut_training_k_max"
                )
            _positive_int("shortcut_training_k_max", self.shortcut_training_k_max)
            if self.shortcut_training_k_max & (self.shortcut_training_k_max - 1):
                raise ValueError("shortcut_training_k_max must be a power of two")
            if self.shortcut_sampling_steps & (self.shortcut_sampling_steps - 1):
                raise ValueError("shortcut_sampling_steps must be a power of two")
            if self.shortcut_sampling_steps > self.shortcut_training_k_max:
                raise ValueError(
                    "shortcut_sampling_steps cannot exceed shortcut_training_k_max"
                )
        elif self.shortcut_training_k_max is not None:
            raise ValueError(
                "shortcut_training_k_max is valid only when prior='shortcut'"
            )
        if not isinstance(self.shortcut_support_safe_bootstrap, bool):
            raise ValueError("shortcut_support_safe_bootstrap must be boolean")
        if self.shortcut_bootstrap_ema_decay is not None and (
            not math.isfinite(self.shortcut_bootstrap_ema_decay)
            or not 0.0 <= self.shortcut_bootstrap_ema_decay < 1.0
        ):
            raise ValueError(
                "shortcut_bootstrap_ema_decay must be None or finite in [0, 1)"
            )
        for name in ("shortcut_intermediate_clip", "shortcut_sampling_clip"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be None or finite and positive")
        if self.reward_loss not in ("mse", "binary_cross_entropy"):
            raise ValueError("reward_loss must be 'mse' or 'binary_cross_entropy'")
        if (self.reward_min is None) != (self.reward_max is None):
            raise ValueError("reward_min and reward_max must either both be set or both be None")
        if self.reward_min is not None:
            if (
                not math.isfinite(self.reward_min)
                or not math.isfinite(self.reward_max)
                or self.reward_min >= self.reward_max
            ):
                raise ValueError("reward bounds must be finite and satisfy reward_min < reward_max")
            if not self.reward_min < self.reward_initial_value < self.reward_max:
                raise ValueError("bounded reward_initial_value must lie strictly inside its support")
        elif self.reward_loss == "binary_cross_entropy":
            raise ValueError("binary_cross_entropy reward loss requires finite reward bounds")
        if not math.isfinite(self.reward_initial_value):
            raise ValueError("reward_initial_value must be finite")
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
            "imf_adaptive_power",
            "imf_meanflow_scale",
            "imf_velocity_scale",
            "imf_endpoint_scale",
            "imf_shortcut_scale",
            "imf_signal_weight_floor",
            "imf_signal_weight_scale",
            "imf_causal_consistency_scale",
            "imf_causal_reward_scale",
            "actor_action_l2_scale",
            "reward_output_init_scale",
            "critic_output_init_scale",
            "behavior_kl_scale",
            "pmpo_positive_weight",
            "pmpo_negative_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.imf_boundary_fraction <= 1:
            raise ValueError("imf_boundary_fraction must be in [0, 1]")
        if self.imf_noise_coupling not in ("independent", "posterior"):
            raise ValueError("imf_noise_coupling must be 'independent' or 'posterior'")
        if (
            not math.isfinite(self.imf_condition_gradient_scale)
            or not 0 <= self.imf_condition_gradient_scale <= 1
        ):
            raise ValueError("imf_condition_gradient_scale must be finite and in [0, 1]")
        if not isinstance(self.imf_boundary_velocity_supervision, bool):
            raise ValueError("imf_boundary_velocity_supervision must be boolean")
        if not isinstance(self.imf_trajectory_enabled, bool):
            raise ValueError("imf_trajectory_enabled must be boolean")
        if self.imf_trajectory_enabled and self.prior != "imf":
            raise ValueError("trajectory iMF requires prior='imf'")
        if self.imf_trajectory_enabled and self.imf_noise_coupling != "independent":
            raise ValueError("trajectory iMF requires independent base noise")
        if self.imf_causal_consistency_scale > 0.0 and not self.imf_trajectory_enabled:
            raise ValueError("causal consistency requires trajectory iMF mode")
        trajectory_probabilities = (
            self.imf_trajectory_clean_probability,
            self.imf_trajectory_corrupted_probability,
            self.imf_trajectory_suffix_probability,
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in trajectory_probabilities
        ):
            raise ValueError(
                "trajectory pattern probabilities must be finite and nonnegative"
            )
        if sum(trajectory_probabilities) <= 0.0:
            raise ValueError("trajectory pattern probabilities must have positive mass")
        if (
            not math.isfinite(self.imf_trajectory_history_noise_max)
            or not 0.0 <= self.imf_trajectory_history_noise_max <= 1.0
        ):
            raise ValueError("imf_trajectory_history_noise_max must be in [0, 1]")
        if not math.isfinite(self.imf_time_mean):
            raise ValueError("imf_time_mean must be finite")
        if self.imf_signal_weight_floor + self.imf_signal_weight_scale <= 0:
            raise ValueError("iMF signal weighting must be positive somewhere")
        for name in (
            "imf_time_std",
            "imf_adaptive_epsilon",
            "imf_causal_huber_delta",
            "imf_causal_normalization_epsilon",
            "actor_init_scale",
            "actor_mean_bound",
            "actor_min_std",
            "actor_max_std",
            "return_normalization_epsilon",
            "behavior_cloning_learning_rate",
            "pmpo_positive_temperature",
            "pmpo_negative_temperature",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.actor_min_std > self.actor_max_std:
            raise ValueError("actor std bounds must satisfy actor_min_std <= actor_max_std")
        if self.actor_gradient not in ("reinforce", "dynamics", "pmpo"):
            raise ValueError("actor_gradient must be 'reinforce', 'dynamics', or 'pmpo'")
        if self.imagination_start_mode not in ("all", "one_per_sequence"):
            raise ValueError(
                "imagination_start_mode must be 'all' or 'one_per_sequence'"
            )
        if self.critic_bins != 1 and self.critic_bins < 3:
            raise ValueError("critic_bins must be 1 or at least 3")
        if self.critic_bins > 1 and self.critic_bins % 2 == 0:
            raise ValueError("distributional critic_bins must be odd")
        if (
            not math.isfinite(self.critic_symlog_min)
            or not math.isfinite(self.critic_symlog_max)
            or self.critic_symlog_min >= 0.0
            or self.critic_symlog_max <= 0.0
            or not math.isclose(
                -self.critic_symlog_min,
                self.critic_symlog_max,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "critic symlog support must be finite and symmetric around zero"
            )
        if (
            not isinstance(self.actor_critic_warmup_steps, int)
            or isinstance(self.actor_critic_warmup_steps, bool)
            or self.actor_critic_warmup_steps < 0
        ):
            raise ValueError("actor_critic_warmup_steps must be a nonnegative integer")
        if (
            not math.isfinite(self.slow_critic_fraction)
            or not 0 < self.slow_critic_fraction <= 1
        ):
            raise ValueError("slow_critic_fraction must be finite and in (0, 1]")
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
        if self.prior == "shortcut":
            return "shortcut_forcing_rssm"
        if self.imf_trajectory_enabled:
            return "trajectory_imf_rssm"
        multistep = self.overshooting_horizon > 1 or any(
            distance > 1 for distance in self.overshooting_distances
        )
        return "imf_rssm_multistep" if multistep else "imf_rssm_one_step"


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
