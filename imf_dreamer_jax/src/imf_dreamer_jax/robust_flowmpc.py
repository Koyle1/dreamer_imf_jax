"""Causally separable robustness primitives for FlowMPC experiments.

The functions in this module deliberately expose the stages of a controller
update instead of silently combining them.  A caller can independently choose
whether actor parameters persist, enforce deterministic reference-policy action
drift budgets, evaluate a proposal on an independently supplied noise bank,
or compare gradient and CEM search over the same bounded action residuals.

Reference-action proximity is only a behavioral drift measurement on the fixed
states supplied by the caller.  Ensemble mean-minus-standard-deviation values
are reported as heuristic risk scores, not confidence bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from numbers import Real
from typing import Callable, Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .flowmpc import (
    FlowMPCConfig,
    FlowMPCUpdate,
    ReBRACConfig,
    flowmpc_adapt_actor,
    flowmpc_objective,
    rebrac_actor,
)
from .nn import Params
from .types import RSSMState

PersistenceMode: TypeAlias = Literal["persistent", "reset"]
HeldOutAcceptanceMode: TypeAlias = Literal["disabled", "held_out_noise"]
ActorChoice: TypeAlias = Literal["adaptation_start", "frozen_reference"]
ActionSequenceObjective: TypeAlias = Callable[[Array], Array]


def _finite_float(name: str, value: float, *, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_eager_array_condition(condition: Array, message: str) -> None:
    """Raise for a false concrete condition and defer checks while tracing."""

    try:
        valid = bool(condition)
    except jax.errors.TracerBoolConversionError:
        return
    if not valid:
        raise ValueError(message)


@dataclass(frozen=True)
class ActionTrustRegionConfig:
    """Budgets for deterministic action drift from a frozen reference actor."""

    anchor_mean_squared_budget: float = 1e-2
    current_max_absolute_budget: float = 1e-1
    backtrack_ratio: float = 0.5
    max_backtracks: int = 8
    feasibility_tolerance: float = 1e-7
    mode: Literal["reference_action"] = "reference_action"

    def __post_init__(self) -> None:
        if self.mode != "reference_action":
            raise ValueError("mode must be 'reference_action'")
        for name in (
            "anchor_mean_squared_budget",
            "current_max_absolute_budget",
            "feasibility_tolerance",
        ):
            _finite_float(name, getattr(self, name), nonnegative=True)
        _finite_float("backtrack_ratio", self.backtrack_ratio)
        if not 0.0 < self.backtrack_ratio < 1.0:
            raise ValueError("backtrack_ratio must be in (0, 1)")
        _nonnegative_int("max_backtracks", self.max_backtracks)


@dataclass(frozen=True)
class PersistenceConfig:
    """Select cumulative actor adaptation or per-environment-step reset."""

    mode: PersistenceMode = "persistent"

    def __post_init__(self) -> None:
        if self.mode not in ("persistent", "reset"):
            raise ValueError("mode must be 'persistent' or 'reset'")


@dataclass(frozen=True)
class HeldOutAcceptanceConfig:
    """Independent-noise acceptance and fallback choices.

    ``baseline`` controls which actor the candidate must improve upon, while
    ``fallback`` controls which actor is executed after rejection.  Neither
    choice changes persistence or reference-action constraints.
    """

    mode: HeldOutAcceptanceMode = "disabled"
    minimum_improvement: float = 0.0
    baseline: ActorChoice = "adaptation_start"
    fallback: ActorChoice = "adaptation_start"

    def __post_init__(self) -> None:
        if self.mode not in ("disabled", "held_out_noise"):
            raise ValueError("mode must be 'disabled' or 'held_out_noise'")
        _finite_float("minimum_improvement", self.minimum_improvement)
        for name in ("baseline", "fallback"):
            if getattr(self, name) not in (
                "adaptation_start",
                "frozen_reference",
            ):
                raise ValueError(
                    f"{name} must be 'adaptation_start' or 'frozen_reference'"
                )


@dataclass(frozen=True)
class ActionSequenceDomainConfig:
    """Static bounds for residual action-sequence decision variables."""

    horizon: int
    action_dim: int
    residual_limit: float
    action_minimum: float = -1.0
    action_maximum: float = 1.0
    mode: Literal["reference_residual_action_sequence"] = (
        "reference_residual_action_sequence"
    )

    def __post_init__(self) -> None:
        if self.mode != "reference_residual_action_sequence":
            raise ValueError("mode must be 'reference_residual_action_sequence'")
        _positive_int("horizon", self.horizon)
        _positive_int("action_dim", self.action_dim)
        _finite_float("residual_limit", self.residual_limit)
        if self.residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        _finite_float("action_minimum", self.action_minimum)
        _finite_float("action_maximum", self.action_maximum)
        if self.action_minimum >= self.action_maximum:
            raise ValueError("action_minimum must be smaller than action_maximum")


@dataclass(frozen=True)
class GradientActionSequenceConfig:
    """Projected gradient-ascent settings for residual action sequences."""

    iterations: int = 8
    step_size: float = 5e-2
    gradient_clip_norm: float | None = None
    mode: Literal["gradient_action_sequence"] = "gradient_action_sequence"

    def __post_init__(self) -> None:
        if self.mode != "gradient_action_sequence":
            raise ValueError("mode must be 'gradient_action_sequence'")
        _positive_int("iterations", self.iterations)
        _finite_float("step_size", self.step_size)
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if self.gradient_clip_norm is not None:
            _finite_float("gradient_clip_norm", self.gradient_clip_norm)
            if self.gradient_clip_norm <= 0.0:
                raise ValueError("gradient_clip_norm must be positive when set")


@dataclass(frozen=True)
class CEMActionSequenceConfig:
    """Deterministic CEM settings for the shared action-sequence domain."""

    iterations: int = 4
    population: int = 64
    elite_count: int = 8
    initial_std: float = 0.5
    minimum_std: float = 0.02
    maximum_std: float = 1.0
    mode: Literal["cem_action_sequence"] = "cem_action_sequence"

    def __post_init__(self) -> None:
        if self.mode != "cem_action_sequence":
            raise ValueError("mode must be 'cem_action_sequence'")
        for name in ("iterations", "population", "elite_count"):
            _positive_int(name, getattr(self, name))
        if self.elite_count > self.population:
            raise ValueError("elite_count cannot exceed population")
        for name in ("initial_std", "minimum_std", "maximum_std"):
            _finite_float(name, getattr(self, name))
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.minimum_std <= self.initial_std <= self.maximum_std:
            raise ValueError("initial_std must lie between minimum_std and maximum_std")


@dataclass(frozen=True)
class RelativePessimismConfig:
    """Heuristic ensemble-relative mean-minus-SD risk-score settings."""

    risk_coefficient: float = 1.0
    minimum_members: int = 2
    mode: Literal["mean_sd_risk"] = "mean_sd_risk"

    def __post_init__(self) -> None:
        if self.mode != "mean_sd_risk":
            raise ValueError("mode must be 'mean_sd_risk'")
        _finite_float("risk_coefficient", self.risk_coefficient, nonnegative=True)
        _positive_int("minimum_members", self.minimum_members)
        if self.minimum_members < 2:
            raise ValueError("minimum_members must be at least two")


class ActionTrustRegionMetrics(NamedTuple):
    """Measured deterministic action drift and budget feasibility."""

    anchor_mean_squared_action_drift: Array
    current_max_absolute_action_drift: Array
    feasible: Array


class ProjectedReferenceActions(NamedTuple):
    """Action proposal scaled exactly along its reference-action segment."""

    anchor_actions: Array
    current_action: Array
    scale: Array
    metrics: ActionTrustRegionMetrics


class ActorBacktrackResult(NamedTuple):
    """Largest checked feasible actor proposal along a parameter step."""

    actor: Params
    metrics: ActionTrustRegionMetrics
    step_scale: Array
    backtracks: Array
    start_feasible: Array
    used_reference_fallback: Array


class ReferenceActorState(NamedTuple):
    """Frozen zero-step actor plus the explicitly carried actor state."""

    frozen_reference_actor: Params
    carried_actor: Params
    environment_step: Array


class HeldOutAcceptanceMetrics(NamedTuple):
    """Acceptance result from objectives on an independently supplied bank."""

    candidate_objective: Array
    baseline_objective: Array
    improvement: Array
    accepted: Array
    used_fallback: Array


class ActionSequenceDomain(NamedTuple):
    """Dynamic residual bounds centered on a frozen reference sequence."""

    reference_actions: Array
    residual_minimum: Array
    residual_maximum: Array


class ActionSequenceProposal(NamedTuple):
    """Shared decision-variable schema used by gradient ascent and CEM."""

    residuals: Array


class ActionSequenceSearchResult(NamedTuple):
    """Matched result schema for both action-sequence optimizers."""

    proposal: ActionSequenceProposal
    action_sequence: Array
    initial_objective: Array
    final_objective: Array
    objective_improvement: Array
    objective_evaluations: Array


class RelativePessimismMetrics(NamedTuple):
    """Paired ensemble improvements and their heuristic risk score."""

    risk_score: Array
    mean_improvement: Array
    improvement_standard_deviation: Array
    worst_improvement: Array
    candidate_mean: Array
    frozen_reference_mean: Array
    member_improvements: Array


def _coerce_reference_action_pairs(
    reference_anchor_actions: Array,
    candidate_anchor_actions: Array,
    reference_current_action: Array,
    candidate_current_action: Array,
) -> tuple[Array, Array, Array, Array]:
    dtype = jnp.result_type(
        reference_anchor_actions,
        candidate_anchor_actions,
        reference_current_action,
        candidate_current_action,
        jnp.float32,
    )
    reference_anchor_actions = jnp.asarray(reference_anchor_actions, dtype=dtype)
    candidate_anchor_actions = jnp.asarray(candidate_anchor_actions, dtype=dtype)
    reference_current_action = jnp.asarray(reference_current_action, dtype=dtype)
    candidate_current_action = jnp.asarray(candidate_current_action, dtype=dtype)
    if reference_anchor_actions.ndim != 2 or reference_anchor_actions.shape[0] == 0:
        raise ValueError("reference anchor actions must be a nonempty matrix")
    if candidate_anchor_actions.shape != reference_anchor_actions.shape:
        raise ValueError("candidate anchor actions must match reference anchors")
    action_dim = reference_anchor_actions.shape[1]
    if action_dim == 0:
        raise ValueError("anchor actions must have a nonempty action dimension")
    if reference_current_action.shape != (action_dim,):
        raise ValueError("reference current action must have shape [action_dim]")
    if candidate_current_action.shape != (action_dim,):
        raise ValueError("candidate current action must match the reference action")
    _require_eager_array_condition(
        jnp.all(jnp.isfinite(reference_anchor_actions))
        & jnp.all(jnp.isfinite(candidate_anchor_actions))
        & jnp.all(jnp.isfinite(reference_current_action))
        & jnp.all(jnp.isfinite(candidate_current_action)),
        "reference and candidate actions must be finite",
    )
    return (
        jax.lax.stop_gradient(reference_anchor_actions),
        candidate_anchor_actions,
        jax.lax.stop_gradient(reference_current_action),
        candidate_current_action,
    )


def _action_trust_region_metrics(
    reference_anchor_actions: Array,
    candidate_anchor_actions: Array,
    reference_current_action: Array,
    candidate_current_action: Array,
    config: ActionTrustRegionConfig,
) -> ActionTrustRegionMetrics:
    anchor_delta = candidate_anchor_actions - reference_anchor_actions
    current_delta = candidate_current_action - reference_current_action
    anchor_mean_squared = jnp.mean(jnp.sum(jnp.square(anchor_delta), axis=-1))
    current_max_absolute = jnp.max(jnp.abs(current_delta))
    finite = jnp.logical_and(
        jnp.isfinite(anchor_mean_squared), jnp.isfinite(current_max_absolute)
    )
    feasible = jnp.logical_and(
        finite,
        jnp.logical_and(
            anchor_mean_squared
            <= config.anchor_mean_squared_budget + config.feasibility_tolerance,
            current_max_absolute
            <= config.current_max_absolute_budget + config.feasibility_tolerance,
        ),
    )
    return ActionTrustRegionMetrics(
        anchor_mean_squared,
        current_max_absolute,
        feasible,
    )


def measure_reference_action_drift(
    reference_anchor_actions: Array,
    candidate_anchor_actions: Array,
    reference_current_action: Array,
    candidate_current_action: Array,
    config: ActionTrustRegionConfig,
) -> ActionTrustRegionMetrics:
    """Measure deterministic reference-action drift on fixed supplied states."""

    values = _coerce_reference_action_pairs(
        reference_anchor_actions,
        candidate_anchor_actions,
        reference_current_action,
        candidate_current_action,
    )
    return _action_trust_region_metrics(*values, config)


def project_reference_actions(
    reference_anchor_actions: Array,
    proposed_anchor_actions: Array,
    reference_current_action: Array,
    proposed_current_action: Array,
    config: ActionTrustRegionConfig,
) -> ProjectedReferenceActions:
    """Project actions along the proposal segment to satisfy both budgets."""

    (
        reference_anchor_actions,
        proposed_anchor_actions,
        reference_current_action,
        proposed_current_action,
    ) = _coerce_reference_action_pairs(
        reference_anchor_actions,
        proposed_anchor_actions,
        reference_current_action,
        proposed_current_action,
    )
    unscaled = _action_trust_region_metrics(
        reference_anchor_actions,
        proposed_anchor_actions,
        reference_current_action,
        proposed_current_action,
        config,
    )
    dtype = reference_anchor_actions.dtype
    tiny = jnp.finfo(dtype).tiny
    anchor_scale = jnp.where(
        unscaled.anchor_mean_squared_action_drift > 0.0,
        jnp.sqrt(
            jnp.asarray(config.anchor_mean_squared_budget, dtype=dtype)
            / jnp.maximum(unscaled.anchor_mean_squared_action_drift, tiny)
        ),
        jnp.asarray(1.0, dtype=dtype),
    )
    current_scale = jnp.where(
        unscaled.current_max_absolute_action_drift > 0.0,
        jnp.asarray(config.current_max_absolute_budget, dtype=dtype)
        / jnp.maximum(unscaled.current_max_absolute_action_drift, tiny),
        jnp.asarray(1.0, dtype=dtype),
    )
    scale = jnp.minimum(1.0, jnp.minimum(anchor_scale, current_scale))
    projected_anchors = reference_anchor_actions + scale * (
        proposed_anchor_actions - reference_anchor_actions
    )
    projected_current = reference_current_action + scale * (
        proposed_current_action - reference_current_action
    )
    metrics = _action_trust_region_metrics(
        reference_anchor_actions,
        projected_anchors,
        reference_current_action,
        projected_current,
        config,
    )
    return ProjectedReferenceActions(
        projected_anchors,
        projected_current,
        scale,
        metrics,
    )


def _validate_actor_state_inputs(
    fixed_anchor_states: Array, current_state: Array
) -> tuple[Array, Array]:
    fixed_anchor_states = jnp.asarray(fixed_anchor_states)
    current_state = jnp.asarray(current_state)
    if fixed_anchor_states.ndim != 2 or fixed_anchor_states.shape[0] == 0:
        raise ValueError("fixed_anchor_states must be a nonempty matrix")
    state_dim = fixed_anchor_states.shape[1]
    if state_dim == 0:
        raise ValueError("fixed_anchor_states must have a nonempty state dimension")
    if current_state.shape != (1, state_dim):
        raise ValueError("current_state must have shape [1, state_dim]")
    _require_eager_array_condition(
        jnp.all(jnp.isfinite(fixed_anchor_states))
        & jnp.all(jnp.isfinite(current_state)),
        "fixed anchor states and current_state must be finite",
    )
    return (
        jax.lax.stop_gradient(fixed_anchor_states),
        jax.lax.stop_gradient(current_state),
    )


def measure_actor_reference_action_drift(
    frozen_reference_actor: Params,
    candidate_actor: Params,
    fixed_anchor_states: Array,
    current_state: Array,
    config: ActionTrustRegionConfig,
) -> ActionTrustRegionMetrics:
    """Evaluate actor action drift from a frozen policy on fixed anchor states."""

    fixed_anchor_states, current_state = _validate_actor_state_inputs(
        fixed_anchor_states, current_state
    )
    reference_anchor_actions = jax.lax.stop_gradient(
        rebrac_actor(frozen_reference_actor, fixed_anchor_states)
    )
    reference_current_action = jax.lax.stop_gradient(
        rebrac_actor(frozen_reference_actor, current_state)[0]
    )
    candidate_anchor_actions = rebrac_actor(candidate_actor, fixed_anchor_states)
    candidate_current_action = rebrac_actor(candidate_actor, current_state)[0]
    return measure_reference_action_drift(
        reference_anchor_actions,
        candidate_anchor_actions,
        reference_current_action,
        candidate_current_action,
        config,
    )


def _interpolate_actor(
    start_actor: Params, proposed_actor: Params, scale: Array
) -> Params:
    return jax.tree_util.tree_map(
        lambda start, proposed: start + scale.astype(start.dtype) * (proposed - start),
        start_actor,
        proposed_actor,
    )


def backtrack_actor_proposal(
    frozen_reference_actor: Params,
    adaptation_start_actor: Params,
    proposed_actor: Params,
    fixed_anchor_states: Array,
    current_state: Array,
    config: ActionTrustRegionConfig,
) -> ActorBacktrackResult:
    """Return the first feasible actor on a deterministic backtracking path.

    Proposal scales are checked in descending order ``1, ratio, ...``.  The
    unchanged adaptation-start actor is checked next.  A final frozen-reference
    candidate makes failure explicit and guarantees a feasible last resort even
    when an externally supplied persistent start was already infeasible.
    """

    fixed_anchor_states, current_state = _validate_actor_state_inputs(
        fixed_anchor_states, current_state
    )
    proposal_scales = tuple(
        config.backtrack_ratio**index for index in range(config.max_backtracks + 1)
    )
    scaled_candidates = tuple(
        _interpolate_actor(
            adaptation_start_actor,
            proposed_actor,
            jnp.asarray(scale, dtype=jnp.float32),
        )
        for scale in proposal_scales
    )
    candidates = scaled_candidates + (
        adaptation_start_actor,
        frozen_reference_actor,
    )
    stacked_candidates = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values, axis=0), *candidates
    )
    candidate_metrics = jax.vmap(
        lambda actor: measure_actor_reference_action_drift(
            frozen_reference_actor,
            actor,
            fixed_anchor_states,
            current_state,
            config,
        )
    )(stacked_candidates)
    selected_index = jnp.argmax(candidate_metrics.feasible.astype(jnp.int32))
    selected_actor = jax.tree_util.tree_map(
        lambda values: values[selected_index], stacked_candidates
    )
    selected_metrics = jax.tree_util.tree_map(
        lambda values: values[selected_index], candidate_metrics
    )
    start_index = len(proposal_scales)
    reference_index = start_index + 1
    scale_values = jnp.asarray((*proposal_scales, 0.0, 0.0), dtype=jnp.float32)
    return ActorBacktrackResult(
        selected_actor,
        selected_metrics,
        scale_values[selected_index],
        selected_index.astype(jnp.int32),
        candidate_metrics.feasible[start_index],
        selected_index == reference_index,
    )


def init_reference_actor_state(reference_actor: Params) -> ReferenceActorState:
    """Create immutable-looking frozen and carried copies of an actor PyTree."""

    frozen = jax.tree_util.tree_map(
        lambda value: jax.lax.stop_gradient(jnp.array(value)), reference_actor
    )
    carried = jax.tree_util.tree_map(jnp.array, reference_actor)
    return ReferenceActorState(
        frozen,
        carried,
        jnp.asarray(0, dtype=jnp.int32),
    )


def begin_actor_adaptation_step(
    state: ReferenceActorState, config: PersistenceConfig
) -> Params:
    """Select the carried actor or frozen reference as this step's start."""

    if config.mode == "persistent":
        return state.carried_actor
    return state.frozen_reference_actor


def finish_actor_adaptation_step(
    state: ReferenceActorState,
    selected_actor: Params,
    config: PersistenceConfig,
) -> ReferenceActorState:
    """Carry a selected actor in persistent mode or discard it in reset mode."""

    if config.mode == "persistent":
        carried = selected_actor
    else:
        carried = state.frozen_reference_actor
    return ReferenceActorState(
        state.frozen_reference_actor,
        carried,
        state.environment_step + 1,
    )


def propose_flowmpc_actor(
    state: ReferenceActorState,
    critic_params: Params,
    world_model_params: Params,
    initial_state: RSSMState,
    current_observation: Array,
    proposal_noises: Array,
    dreamer_config: DreamerConfig,
    rebrac_config: ReBRACConfig,
    flowmpc_config: FlowMPCConfig,
    persistence_config: PersistenceConfig,
) -> FlowMPCUpdate:
    """Form an existing FlowMPC proposal from the explicit persistence start."""

    adaptation_start_actor = begin_actor_adaptation_step(state, persistence_config)
    return flowmpc_adapt_actor(
        adaptation_start_actor,
        critic_params,
        world_model_params,
        initial_state,
        current_observation,
        proposal_noises,
        dreamer_config,
        rebrac_config,
        flowmpc_config,
    )


def held_out_noise_acceptance(
    candidate_objective: Array,
    baseline_objective: Array,
    config: HeldOutAcceptanceConfig,
) -> HeldOutAcceptanceMetrics:
    """Decide acceptance from scalar objectives on a held-out noise bank."""

    candidate_objective = jnp.asarray(candidate_objective)
    baseline_objective = jnp.asarray(baseline_objective)
    if candidate_objective.shape != () or baseline_objective.shape != ():
        raise ValueError("held-out candidate and baseline objectives must be scalars")
    improvement = candidate_objective - baseline_objective
    finite = jnp.logical_and(
        jnp.isfinite(candidate_objective), jnp.isfinite(baseline_objective)
    )
    if config.mode == "held_out_noise":
        accepted = jnp.logical_and(finite, improvement >= config.minimum_improvement)
    else:
        accepted = jnp.asarray(True)
    used_fallback = jnp.logical_and(
        config.mode == "held_out_noise", jnp.logical_not(accepted)
    )
    return HeldOutAcceptanceMetrics(
        candidate_objective,
        baseline_objective,
        improvement,
        accepted,
        used_fallback,
    )


def _actor_choice(
    adaptation_start_actor: Params,
    frozen_reference_actor: Params,
    choice: ActorChoice,
) -> Params:
    if choice == "adaptation_start":
        return adaptation_start_actor
    return frozen_reference_actor


def evaluate_flowmpc_held_out_acceptance(
    candidate_actor: Params,
    adaptation_start_actor: Params,
    frozen_reference_actor: Params,
    critic_params: Params,
    world_model_params: Params,
    initial_state: RSSMState,
    current_observation: Array,
    held_out_noises: Array,
    dreamer_config: DreamerConfig,
    rebrac_config: ReBRACConfig,
    flowmpc_config: FlowMPCConfig,
    acceptance_config: HeldOutAcceptanceConfig,
) -> HeldOutAcceptanceMetrics:
    """Evaluate candidate and configured baseline on the same held-out noises."""

    baseline_actor = _actor_choice(
        adaptation_start_actor,
        frozen_reference_actor,
        acceptance_config.baseline,
    )
    candidate = flowmpc_objective(
        candidate_actor,
        critic_params,
        world_model_params,
        initial_state,
        current_observation,
        held_out_noises,
        dreamer_config,
        rebrac_config,
        flowmpc_config,
    ).objective
    baseline = flowmpc_objective(
        jax.lax.stop_gradient(baseline_actor),
        critic_params,
        world_model_params,
        initial_state,
        current_observation,
        held_out_noises,
        dreamer_config,
        rebrac_config,
        flowmpc_config,
    ).objective
    return held_out_noise_acceptance(candidate, baseline, acceptance_config)


def apply_held_out_noise_fallback(
    candidate_actor: Params,
    adaptation_start_actor: Params,
    frozen_reference_actor: Params,
    metrics: HeldOutAcceptanceMetrics,
    config: HeldOutAcceptanceConfig,
) -> Params:
    """Choose the candidate or the independently configured fallback actor."""

    if metrics.accepted.shape != ():
        raise ValueError("acceptance flag must be scalar")
    fallback_actor = _actor_choice(
        adaptation_start_actor,
        frozen_reference_actor,
        config.fallback,
    )
    return jax.tree_util.tree_map(
        lambda candidate, fallback: jnp.where(metrics.accepted, candidate, fallback),
        candidate_actor,
        fallback_actor,
    )


def make_action_sequence_domain(
    reference_actions: Array, config: ActionSequenceDomainConfig
) -> ActionSequenceDomain:
    """Construct exact residual bounds around a frozen reference sequence."""

    reference_actions = jnp.asarray(
        reference_actions,
        dtype=jnp.result_type(reference_actions, jnp.float32),
    )
    expected_shape = (config.horizon, config.action_dim)
    if reference_actions.shape != expected_shape:
        raise ValueError(
            f"reference_actions must have shape {expected_shape}, "
            f"got {reference_actions.shape}"
        )
    _require_eager_array_condition(
        jnp.all(jnp.isfinite(reference_actions)),
        "reference_actions must be finite",
    )
    _require_eager_array_condition(
        jnp.all(reference_actions >= config.action_minimum)
        & jnp.all(reference_actions <= config.action_maximum),
        "reference_actions must lie inside the action bounds",
    )
    reference_actions = jax.lax.stop_gradient(reference_actions)
    residual_minimum = jnp.maximum(
        -config.residual_limit,
        config.action_minimum - reference_actions,
    )
    residual_maximum = jnp.minimum(
        config.residual_limit,
        config.action_maximum - reference_actions,
    )
    return ActionSequenceDomain(
        reference_actions,
        residual_minimum,
        residual_maximum,
    )


def _validate_action_sequence_domain(domain: ActionSequenceDomain) -> None:
    if domain.reference_actions.ndim != 2 or domain.reference_actions.size == 0:
        raise ValueError("action-sequence domain must be a nonempty matrix")
    if domain.residual_minimum.shape != domain.reference_actions.shape:
        raise ValueError("domain residual_minimum has the wrong shape")
    if domain.residual_maximum.shape != domain.reference_actions.shape:
        raise ValueError("domain residual_maximum has the wrong shape")
    _require_eager_array_condition(
        jnp.all(jnp.isfinite(domain.reference_actions))
        & jnp.all(jnp.isfinite(domain.residual_minimum))
        & jnp.all(jnp.isfinite(domain.residual_maximum)),
        "action-sequence domain values must be finite",
    )
    _require_eager_array_condition(
        jnp.all(domain.residual_minimum <= domain.residual_maximum),
        "action-sequence residual bounds are inconsistent",
    )


def make_action_sequence_proposal(
    residuals: Array, domain: ActionSequenceDomain
) -> ActionSequenceProposal:
    """Project shared residual decision variables into their declared domain."""

    _validate_action_sequence_domain(domain)
    residuals = jnp.asarray(residuals, dtype=domain.reference_actions.dtype)
    if residuals.shape != domain.reference_actions.shape:
        raise ValueError("proposal residuals must match the action-sequence domain")
    _require_eager_array_condition(
        jnp.all(jnp.isfinite(residuals)),
        "proposal residuals must be finite",
    )
    projected = jnp.clip(residuals, domain.residual_minimum, domain.residual_maximum)
    return ActionSequenceProposal(projected)


def action_sequence_from_proposal(
    proposal: ActionSequenceProposal, domain: ActionSequenceDomain
) -> Array:
    """Map a residual proposal to its bounded executable action sequence."""

    proposal = make_action_sequence_proposal(proposal.residuals, domain)
    return domain.reference_actions + proposal.residuals


def action_sequence_objective_adapter(
    proposal: ActionSequenceProposal,
    domain: ActionSequenceDomain,
    objective_fn: ActionSequenceObjective,
) -> Array:
    """Evaluate a scalar objective through the shared residual parameterization."""

    action_sequence = action_sequence_from_proposal(proposal, domain)
    objective = jnp.asarray(objective_fn(action_sequence))
    if objective.shape != ():
        raise ValueError("action-sequence objective must return a scalar")
    return objective


def batched_action_sequence_objective_adapter(
    proposals: ActionSequenceProposal,
    domain: ActionSequenceDomain,
    objective_fn: ActionSequenceObjective,
) -> Array:
    """Vectorize the same scalar objective adapter over residual proposals."""

    residuals = jnp.asarray(proposals.residuals)
    if residuals.ndim != 3 or residuals.shape[1:] != domain.reference_actions.shape:
        raise ValueError(
            "batched proposal residuals must have shape "
            "[candidate, horizon, action_dim]"
        )
    if residuals.shape[0] == 0:
        raise ValueError("batched proposals must contain at least one candidate")
    return jax.vmap(
        lambda candidate: action_sequence_objective_adapter(
            ActionSequenceProposal(candidate), domain, objective_fn
        )
    )(residuals)


def _clip_action_sequence_gradient(
    gradient: Array, maximum_norm: float | None
) -> Array:
    if maximum_norm is None:
        return gradient
    norm = jnp.sqrt(jnp.sum(jnp.square(gradient)))
    tiny = jnp.finfo(gradient.dtype).tiny
    scale = jnp.minimum(1.0, maximum_norm / jnp.maximum(norm, tiny))
    return gradient * scale


def gradient_action_sequence_search(
    objective_fn: ActionSequenceObjective,
    domain: ActionSequenceDomain,
    initial_proposal: ActionSequenceProposal,
    config: GradientActionSequenceConfig,
) -> ActionSequenceSearchResult:
    """Run projected gradient ascent over the shared residual variables."""

    current = make_action_sequence_proposal(initial_proposal.residuals, domain)

    def variable_objective(residuals: Array) -> Array:
        return action_sequence_objective_adapter(
            ActionSequenceProposal(residuals), domain, objective_fn
        )

    initial_objective: Array | None = None
    best_objective: Array | None = None
    best_residuals = current.residuals
    residuals = current.residuals
    for index in range(config.iterations):
        objective, gradient = jax.value_and_grad(variable_objective)(residuals)
        if index == 0:
            initial_objective = objective
            best_objective = objective
        else:
            assert best_objective is not None
            better = objective > best_objective
            best_residuals = jnp.where(better, residuals, best_residuals)
            best_objective = jnp.where(better, objective, best_objective)
        gradient = _clip_action_sequence_gradient(gradient, config.gradient_clip_norm)
        residuals = make_action_sequence_proposal(
            residuals + config.step_size * gradient, domain
        ).residuals

    final_candidate_objective = variable_objective(residuals)
    assert initial_objective is not None and best_objective is not None
    better = final_candidate_objective > best_objective
    best_residuals = jnp.where(better, residuals, best_residuals)
    best_objective = jnp.where(better, final_candidate_objective, best_objective)
    proposal = ActionSequenceProposal(best_residuals)
    actions = action_sequence_from_proposal(proposal, domain)
    return ActionSequenceSearchResult(
        proposal,
        actions,
        initial_objective,
        best_objective,
        best_objective - initial_objective,
        jnp.asarray(config.iterations + 1, dtype=jnp.int32),
    )


def cem_action_sequence_search(
    objective_fn: ActionSequenceObjective,
    domain: ActionSequenceDomain,
    initial_proposal: ActionSequenceProposal,
    standard_normal_proposals: Array,
    config: CEMActionSequenceConfig,
) -> ActionSequenceSearchResult:
    """Run deterministic CEM using a caller-supplied standard-normal bank."""

    initial = make_action_sequence_proposal(initial_proposal.residuals, domain)
    standard_normal_proposals = jnp.asarray(
        standard_normal_proposals, dtype=domain.reference_actions.dtype
    )
    expected_shape = (
        config.iterations,
        config.population,
        *domain.reference_actions.shape,
    )
    if standard_normal_proposals.shape != expected_shape:
        raise ValueError(
            f"standard_normal_proposals must have shape {expected_shape}, "
            f"got {standard_normal_proposals.shape}"
        )
    _require_eager_array_condition(
        jnp.all(jnp.isfinite(standard_normal_proposals)),
        "standard_normal_proposals must be finite",
    )
    mean = initial.residuals
    std = jnp.full_like(mean, config.initial_std)
    initial_objective = action_sequence_objective_adapter(initial, domain, objective_fn)
    best_residuals = mean
    best_objective = initial_objective

    for index in range(config.iterations):
        candidates = mean[None] + std[None] * standard_normal_proposals[index]
        candidates = jnp.clip(
            candidates,
            domain.residual_minimum[None],
            domain.residual_maximum[None],
        )
        # Retaining the current mean as a candidate gives deterministic elitism.
        candidates = candidates.at[0].set(mean)
        scores = batched_action_sequence_objective_adapter(
            ActionSequenceProposal(candidates), domain, objective_fn
        )
        _, elite_indices = jax.lax.top_k(scores, config.elite_count)
        elites = candidates[elite_indices]
        mean = make_action_sequence_proposal(jnp.mean(elites, axis=0), domain).residuals
        std = jnp.clip(
            jnp.std(elites, axis=0),
            config.minimum_std,
            config.maximum_std,
        )
        iteration_index = jnp.argmax(scores)
        iteration_objective = scores[iteration_index]
        better = iteration_objective > best_objective
        best_residuals = jnp.where(better, candidates[iteration_index], best_residuals)
        best_objective = jnp.where(better, iteration_objective, best_objective)

    mean_proposal = ActionSequenceProposal(mean)
    mean_objective = action_sequence_objective_adapter(
        mean_proposal, domain, objective_fn
    )
    better = mean_objective > best_objective
    best_residuals = jnp.where(better, mean, best_residuals)
    best_objective = jnp.where(better, mean_objective, best_objective)
    proposal = ActionSequenceProposal(best_residuals)
    actions = action_sequence_from_proposal(proposal, domain)
    evaluations = 2 + config.iterations * config.population
    return ActionSequenceSearchResult(
        proposal,
        actions,
        initial_objective,
        best_objective,
        best_objective - initial_objective,
        jnp.asarray(evaluations, dtype=jnp.int32),
    )


def ensemble_relative_pessimism(
    candidate_member_objectives: Array,
    frozen_reference_member_objectives: Array,
    config: RelativePessimismConfig,
) -> RelativePessimismMetrics:
    """Compute paired ensemble improvement and a mean-minus-SD risk score."""

    candidate = jnp.asarray(
        candidate_member_objectives,
        dtype=jnp.result_type(candidate_member_objectives, jnp.float32),
    )
    reference = jnp.asarray(frozen_reference_member_objectives, dtype=candidate.dtype)
    if candidate.ndim != 1:
        raise ValueError("candidate_member_objectives must be a vector")
    if reference.shape != candidate.shape:
        raise ValueError("candidate and frozen-reference vectors must match")
    if candidate.shape[0] < config.minimum_members:
        raise ValueError("ensemble objective vectors have fewer than minimum_members")
    _require_eager_array_condition(
        jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(reference)),
        "ensemble objective vectors must be finite",
    )
    frozen_reference = jax.lax.stop_gradient(reference)
    improvements = candidate - frozen_reference
    mean_improvement = jnp.mean(improvements)
    improvement_sd = jnp.std(improvements)
    risk_score = mean_improvement - config.risk_coefficient * improvement_sd
    return RelativePessimismMetrics(
        risk_score,
        mean_improvement,
        improvement_sd,
        jnp.min(improvements),
        jnp.mean(candidate),
        jnp.mean(frozen_reference),
        improvements,
    )


def ensemble_relative_risk_score(
    candidate_member_objectives: Array,
    frozen_reference_member_objectives: Array,
    config: RelativePessimismConfig,
) -> Array:
    """Return only the differentiable ensemble-relative heuristic risk score."""

    return ensemble_relative_pessimism(
        candidate_member_objectives,
        frozen_reference_member_objectives,
        config,
    ).risk_score


jit_measure_reference_action_drift = partial(jax.jit, static_argnames=("config",))(
    measure_reference_action_drift
)
jit_project_reference_actions = partial(jax.jit, static_argnames=("config",))(
    project_reference_actions
)
jit_measure_actor_reference_action_drift = partial(
    jax.jit, static_argnames=("config",)
)(measure_actor_reference_action_drift)
jit_backtrack_actor_proposal = partial(jax.jit, static_argnames=("config",))(
    backtrack_actor_proposal
)
jit_begin_actor_adaptation_step = partial(jax.jit, static_argnames=("config",))(
    begin_actor_adaptation_step
)
jit_finish_actor_adaptation_step = partial(jax.jit, static_argnames=("config",))(
    finish_actor_adaptation_step
)
jit_propose_flowmpc_actor = partial(
    jax.jit,
    static_argnames=(
        "dreamer_config",
        "rebrac_config",
        "flowmpc_config",
        "persistence_config",
    ),
)(propose_flowmpc_actor)
jit_held_out_noise_acceptance = partial(jax.jit, static_argnames=("config",))(
    held_out_noise_acceptance
)
jit_evaluate_flowmpc_held_out_acceptance = partial(
    jax.jit,
    static_argnames=(
        "dreamer_config",
        "rebrac_config",
        "flowmpc_config",
        "acceptance_config",
    ),
)(evaluate_flowmpc_held_out_acceptance)
jit_apply_held_out_noise_fallback = partial(jax.jit, static_argnames=("config",))(
    apply_held_out_noise_fallback
)
jit_make_action_sequence_domain = partial(jax.jit, static_argnames=("config",))(
    make_action_sequence_domain
)
jit_gradient_action_sequence_search = partial(
    jax.jit, static_argnames=("objective_fn", "config")
)(gradient_action_sequence_search)
jit_cem_action_sequence_search = partial(
    jax.jit, static_argnames=("objective_fn", "config")
)(cem_action_sequence_search)
jit_ensemble_relative_pessimism = partial(jax.jit, static_argnames=("config",))(
    ensemble_relative_pessimism
)
jit_ensemble_relative_risk_score = partial(jax.jit, static_argnames=("config",))(
    ensemble_relative_risk_score
)


__all__ = [
    "ActionSequenceDomain",
    "ActionSequenceDomainConfig",
    "ActionSequenceObjective",
    "ActionSequenceProposal",
    "ActionSequenceSearchResult",
    "ActionTrustRegionConfig",
    "ActionTrustRegionMetrics",
    "ActorBacktrackResult",
    "ActorChoice",
    "CEMActionSequenceConfig",
    "GradientActionSequenceConfig",
    "HeldOutAcceptanceConfig",
    "HeldOutAcceptanceMetrics",
    "HeldOutAcceptanceMode",
    "PersistenceConfig",
    "PersistenceMode",
    "ProjectedReferenceActions",
    "ReferenceActorState",
    "RelativePessimismConfig",
    "RelativePessimismMetrics",
    "action_sequence_from_proposal",
    "action_sequence_objective_adapter",
    "apply_held_out_noise_fallback",
    "backtrack_actor_proposal",
    "batched_action_sequence_objective_adapter",
    "begin_actor_adaptation_step",
    "cem_action_sequence_search",
    "ensemble_relative_pessimism",
    "ensemble_relative_risk_score",
    "evaluate_flowmpc_held_out_acceptance",
    "finish_actor_adaptation_step",
    "gradient_action_sequence_search",
    "held_out_noise_acceptance",
    "init_reference_actor_state",
    "jit_apply_held_out_noise_fallback",
    "jit_backtrack_actor_proposal",
    "jit_begin_actor_adaptation_step",
    "jit_cem_action_sequence_search",
    "jit_ensemble_relative_pessimism",
    "jit_ensemble_relative_risk_score",
    "jit_evaluate_flowmpc_held_out_acceptance",
    "jit_finish_actor_adaptation_step",
    "jit_gradient_action_sequence_search",
    "jit_held_out_noise_acceptance",
    "jit_make_action_sequence_domain",
    "jit_measure_actor_reference_action_drift",
    "jit_measure_reference_action_drift",
    "jit_project_reference_actions",
    "jit_propose_flowmpc_actor",
    "make_action_sequence_domain",
    "make_action_sequence_proposal",
    "measure_actor_reference_action_drift",
    "measure_reference_action_drift",
    "project_reference_actions",
    "propose_flowmpc_actor",
]
