"""Deterministic model-side training for the actor-gap roadmap study.

This module owns the five *model* interventions in the roadmap and deliberately
does not perform artifact I/O or dependency authentication.  The caller must
pass already-authenticated, immutable trajectory-iMF world-model parameters,
replay arrays, and (for actor-specific cells) a frozen ReBRAC state.

The registered matrix contains 21 cells:

* three uniform prior-only continuations;
* six actor-specific approximate policy-density tilted prior continuations;
* six actor-specific proof-consistent CVAML prior continuations;
* three direct horizon-one endpoint iMF models; and
* three direct any-step endpoint iMF models.

All prior families start from the exact same source prior and create a fresh
zero-moment Adam state.  No other source subtree is returned or updated.  The
two endpoint families use the same initialization, minibatch/start uniforms,
network shape, maximum action-chunk layout, optimizer, and compound iMF loss;
only their registered horizon schedule differs.  In particular, the endpoint
loss is not endpoint regression alone: it retains the average-velocity and
instantaneous-velocity terms and uses a strictly positive endpoint scale.

The functions are deterministic given their explicit inputs and schedules.
``replay_model_cell_payload`` reruns a cell and verifies all deterministic
digests, allowing a study-level verifier to bind its own files and markers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any, Literal, Mapping, NamedTuple, Sequence

import numpy as np

ModelFamily = Literal[
    "uniform_prior",
    "approximate_policy_tilt_prior",
    "proof_consistent_cvaml_value_prior",
    "endpoint_h1_imf",
    "endpoint_anystep_imf",
]

SCHEMA = "trajectory-imf-actor-gap-model-training-v1"
REPLAY_SCHEMA = "trajectory-imf-actor-gap-model-replay-v1"
APPROXIMATE_POLICY_DENSITY_LABEL = (
    "approximate_fixed_variance_gaussian_around_deterministic_action"
)
PROOF_CONSISTENT_CVAML_LABEL = (
    "sample_mean_squared_residual_minus_ddof1_sample_variance_over_K"
)
ENDPOINT_OBJECTIVE_LABEL = (
    "full_improved_meanflow_compound_loss_with_positive_endpoint_scale"
)

PRIOR_FAMILIES: tuple[ModelFamily, ...] = (
    "uniform_prior",
    "approximate_policy_tilt_prior",
    "proof_consistent_cvaml_value_prior",
)
ACTOR_SPECIFIC_PRIOR_FAMILIES: tuple[ModelFamily, ...] = (
    "approximate_policy_tilt_prior",
    "proof_consistent_cvaml_value_prior",
)
ENDPOINT_FAMILIES: tuple[ModelFamily, ...] = (
    "endpoint_h1_imf",
    "endpoint_anystep_imf",
)
MODEL_FAMILY_ORDER: tuple[ModelFamily, ...] = PRIOR_FAMILIES + ENDPOINT_FAMILIES


@dataclass(frozen=True)
class ModelCellSpec:
    """One separately attributable registered model-training cell."""

    index: int
    cell_id: str
    family: ModelFamily
    world_model_seed: int
    actor_seed: int | None


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Frozen optimizer, schedule, and objective settings for one study."""

    updates: int = 5_000
    batch_size: int = 16
    sequence_length: int = 32
    burn_in: int = 8
    learning_rate: float = 3e-4
    grad_clip: float = 10.0
    approximate_policy_std: float = 0.20
    tilt_eta: float = 0.0
    cvaml_samples: int = 4
    cvaml_scale: float = 1.0
    bellman_target_scale: float = 1.0
    planner_discount: float = 0.99
    maximum_chunk_horizon: int = 5
    endpoint_scale: float = 1.0
    endpoint_hidden_dim: int | None = None
    endpoint_depth: int = 2

    def __post_init__(self) -> None:
        for name in (
            "updates",
            "batch_size",
            "sequence_length",
            "maximum_chunk_horizon",
            "endpoint_depth",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.burn_in, int)
            or isinstance(self.burn_in, bool)
            or self.burn_in < 0
            or self.burn_in >= self.sequence_length
        ):
            raise ValueError("burn_in must be a nonnegative sequence prefix")
        if self.maximum_chunk_horizon >= self.sequence_length:
            raise ValueError("maximum_chunk_horizon must be below sequence_length")
        if (
            not isinstance(self.cvaml_samples, int)
            or isinstance(self.cvaml_samples, bool)
            or self.cvaml_samples < 2
        ):
            raise ValueError("cvaml_samples must be an integer of at least two")
        if self.endpoint_hidden_dim is not None and (
            not isinstance(self.endpoint_hidden_dim, int)
            or isinstance(self.endpoint_hidden_dim, bool)
            or self.endpoint_hidden_dim <= 0
        ):
            raise ValueError("endpoint_hidden_dim must be positive when supplied")
        for name in (
            "learning_rate",
            "grad_clip",
            "approximate_policy_std",
            "bellman_target_scale",
            "endpoint_scale",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("tilt_eta", "cvaml_scale"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not isinstance(self.planner_discount, (int, float))
            or isinstance(self.planner_discount, bool)
            or not math.isfinite(float(self.planner_discount))
            or not 0.0 <= self.planner_discount <= 1.0
        ):
            raise ValueError("planner_discount must be finite and lie in [0, 1]")


class ModelTrainState(NamedTuple):
    """Only the cell's permitted trainable subtree and its fresh Adam state."""

    params: Any
    optimizer: Any


class PriorLossMetrics(NamedTuple):
    """Scalar decomposition of one prior-only objective evaluation."""

    total: Any
    trajectory: Any
    cvaml_corrected: Any
    cvaml_uncorrected: Any
    cvaml_variance_correction: Any
    tilt_effective_sample_size_fraction: Any
    tilt_clipped_fraction: Any
    valid_token_fraction: Any


class EndpointLossMetrics(NamedTuple):
    """Scalar decomposition of one direct endpoint-iMF evaluation."""

    total: Any
    meanflow_raw_mse: Any
    velocity_raw_mse: Any
    endpoint_raw_mse: Any
    accepted_fraction: Any
    horizon: Any


class TrainStepMetrics(NamedTuple):
    """One optimizer step plus the pre-clipping gradient norm."""

    total: Any
    primary: Any
    auxiliary: Any
    uncorrected_auxiliary: Any
    correction: Any
    accepted_fraction: Any
    tilt_effective_sample_size_fraction: Any
    tilt_clipped_fraction: Any
    horizon: Any
    grad_norm: Any


def build_model_cell_specs(
    world_model_seeds: Sequence[int], actor_seeds: Sequence[int]
) -> tuple[ModelCellSpec, ...]:
    """Return the canonical 21-cell layout for three worlds and two actors.

    The function accepts explicit seeds so tests and future protocols can use
    the same construction.  The roadmap contract itself requires exactly
    three unique world seeds and two unique actor seeds.
    """

    worlds = tuple(int(value) for value in world_model_seeds)
    actors = tuple(int(value) for value in actor_seeds)
    if len(worlds) != 3 or len(set(worlds)) != 3:
        raise ValueError("the roadmap requires three unique world-model seeds")
    if len(actors) != 2 or len(set(actors)) != 2:
        raise ValueError("the roadmap requires two unique actor seeds")
    cells: list[ModelCellSpec] = []
    for family in MODEL_FAMILY_ORDER:
        nested_actors: tuple[int | None, ...] = (
            actors if family in ACTOR_SPECIFIC_PRIOR_FAMILIES else (None,)
        )
        for world_seed in worlds:
            for actor_seed in nested_actors:
                suffix = "" if actor_seed is None else f"-{actor_seed}"
                cell_id = f"model-{family}-{world_seed}{suffix}"
                cells.append(
                    ModelCellSpec(
                        index=len(cells),
                        cell_id=cell_id,
                        family=family,
                        world_model_seed=world_seed,
                        actor_seed=actor_seed,
                    )
                )
    if len(cells) != 21:
        raise AssertionError("canonical model matrix must contain 21 cells")
    return tuple(cells)


def validate_model_cell_spec(spec: ModelCellSpec) -> None:
    if spec.family not in MODEL_FAMILY_ORDER:
        raise ValueError(f"unknown model family: {spec.family!r}")
    if (
        not isinstance(spec.index, int)
        or isinstance(spec.index, bool)
        or spec.index < 0
    ):
        raise ValueError("cell index must be a nonnegative integer")
    if not spec.cell_id:
        raise ValueError("cell_id must be nonempty")
    actor_specific = spec.family in ACTOR_SPECIFIC_PRIOR_FAMILIES
    if actor_specific != (spec.actor_seed is not None):
        raise ValueError("actor-specific family/actor-seed identity mismatch")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _derive_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(_canonical_json(parts)).digest()[:4], "big")


def _tree_sha256(tree: Any) -> str:
    """Hash a PyTree without serializing object identities or device metadata."""

    import jax

    digest = hashlib.sha256()
    leaves, structure = jax.tree_util.tree_flatten(tree)
    descriptor = str(structure).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    for index, leaf in enumerate(leaves):
        value = np.ascontiguousarray(np.asarray(leaf))
        header = _canonical_json([index, value.dtype.str, list(value.shape)])
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def schedule_sha256(schedule: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(schedule):
        value = np.ascontiguousarray(np.asarray(schedule[name]))
        header = _canonical_json([name, value.dtype.str, list(value.shape)])
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_training_schedule(
    replay_arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    world_model_seed: int,
    objective: ModelTrainingConfig,
) -> dict[str, np.ndarray]:
    """Build one world-seed schedule shared by every model family.

    Crucially, family and actor seed are absent from the derivation.  Thus the
    three prior objectives see identical minibatches and random streams, and
    the h1/any-step pair shares minibatches, endpoint start uniforms, and the
    *stored* pair of horizon schedules.  A cell selects one horizon column but
    never generates a family-specific schedule.
    """

    required = ("observations", "actions", "train_episode_ids")
    if any(name not in replay_arrays for name in required):
        raise KeyError(f"replay arrays must contain {required}")
    observations = np.asarray(replay_arrays["observations"])
    actions = np.asarray(replay_arrays["actions"])
    train_ids = np.asarray(replay_arrays["train_episode_ids"], dtype=np.int32)
    if observations.ndim < 3 or actions.ndim != 3:
        raise ValueError("replay observations/actions must be episodic sequences")
    if observations.shape[:2] != actions.shape[:2]:
        raise ValueError("replay observations and actions must align")
    if train_ids.ndim != 1 or train_ids.size == 0:
        raise ValueError("train_episode_ids must be a nonempty vector")
    if np.any(train_ids < 0) or np.any(train_ids >= observations.shape[0]):
        raise ValueError("train_episode_ids contains an out-of-range episode")
    transitions = observations.shape[1]
    if transitions < objective.sequence_length:
        raise ValueError("replay episodes are shorter than sequence_length")

    random = np.random.default_rng(
        _derive_seed("actor-gap-model-schedule", task, int(world_model_seed))
    )
    episode_ids = random.choice(
        train_ids,
        size=(objective.updates, objective.batch_size),
        replace=True,
    ).astype(np.int32)
    starts = random.integers(
        0,
        transitions - objective.sequence_length + 1,
        size=(objective.updates, objective.batch_size),
        dtype=np.int32,
    )
    endpoint_start_uniforms = random.random(
        (objective.updates, objective.batch_size), dtype=np.float32
    )
    # A shuffled complete block prevents an incidental monotone curriculum,
    # while retaining exactly balanced horizons whenever updates is divisible
    # by maximum_chunk_horizon.
    horizon_values: list[int] = []
    while len(horizon_values) < objective.updates:
        block = np.arange(1, objective.maximum_chunk_horizon + 1, dtype=np.int32)
        random.shuffle(block)
        horizon_values.extend(int(value) for value in block)
    endpoint_horizons_any_step = np.asarray(
        horizon_values[: objective.updates], dtype=np.int32
    )
    endpoint_horizons_h1 = np.ones((objective.updates,), dtype=np.int32)

    import jax

    base_key = jax.random.PRNGKey(
        _derive_seed("actor-gap-model-objective", task, int(world_model_seed))
    )
    objective_keys = np.asarray(
        jax.device_get(
            jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
                np.arange(objective.updates, dtype=np.uint32)
            )
        ),
        dtype=np.uint32,
    )
    schedule = {
        "episode_ids": episode_ids,
        "starts": starts,
        "objective_keys": objective_keys,
        "endpoint_start_uniforms": endpoint_start_uniforms,
        "endpoint_horizons_h1": endpoint_horizons_h1,
        "endpoint_horizons_any_step": endpoint_horizons_any_step,
    }
    validate_training_schedule(schedule, replay_arrays, objective)
    return schedule


def validate_training_schedule(
    schedule: Mapping[str, np.ndarray],
    replay_arrays: Mapping[str, np.ndarray],
    objective: ModelTrainingConfig,
) -> None:
    expected = {
        "episode_ids": (objective.updates, objective.batch_size),
        "starts": (objective.updates, objective.batch_size),
        "objective_keys": (objective.updates, 2),
        "endpoint_start_uniforms": (objective.updates, objective.batch_size),
        "endpoint_horizons_h1": (objective.updates,),
        "endpoint_horizons_any_step": (objective.updates,),
    }
    if set(schedule) != set(expected):
        raise ValueError("training schedule fields are incomplete")
    for name, shape in expected.items():
        if np.asarray(schedule[name]).shape != shape:
            raise ValueError(f"training schedule {name} has the wrong shape")
    train_ids = set(
        int(value) for value in np.asarray(replay_arrays["train_episode_ids"]).tolist()
    )
    if not set(np.asarray(schedule["episode_ids"]).reshape(-1).tolist()) <= train_ids:
        raise ValueError("training schedule uses a non-training episode")
    maximum_start = (
        np.asarray(replay_arrays["observations"]).shape[1] - objective.sequence_length
    )
    starts = np.asarray(schedule["starts"])
    if np.any(starts < 0) or np.any(starts > maximum_start):
        raise ValueError("training schedule contains an invalid sequence start")
    uniforms = np.asarray(schedule["endpoint_start_uniforms"])
    if not np.all(np.isfinite(uniforms) & (uniforms >= 0.0) & (uniforms < 1.0)):
        raise ValueError("endpoint start uniforms must lie in [0, 1)")
    if not np.all(np.asarray(schedule["endpoint_horizons_h1"]) == 1):
        raise ValueError("horizon-one schedule must contain only one")
    any_horizons = np.asarray(schedule["endpoint_horizons_any_step"])
    if np.any(any_horizons < 1) or np.any(
        any_horizons > objective.maximum_chunk_horizon
    ):
        raise ValueError("any-step schedule contains an out-of-range horizon")
    keys = np.asarray(schedule["objective_keys"])
    if keys.dtype != np.uint32 or np.unique(keys, axis=0).shape[0] != objective.updates:
        raise ValueError("objective RNG keys must be unique uint32 pairs")


def require_canonical_training_schedule(
    retained_schedule: Mapping[str, np.ndarray],
    replay_arrays: Mapping[str, np.ndarray],
    *,
    task: str,
    world_model_seed: int,
    objective: ModelTrainingConfig,
) -> dict[str, np.ndarray]:
    """Rederive and exactly authenticate the family-independent schedule."""

    retained = {name: np.asarray(value) for name, value in retained_schedule.items()}
    validate_training_schedule(retained, replay_arrays, objective)
    canonical = build_training_schedule(
        replay_arrays,
        task=task,
        world_model_seed=world_model_seed,
        objective=objective,
    )
    if set(retained) != set(canonical):
        raise ValueError("retained training schedule fields are not canonical")
    differing = [
        name
        for name in sorted(canonical)
        if retained[name].dtype != canonical[name].dtype
        or retained[name].shape != canonical[name].shape
        or not np.array_equal(retained[name], canonical[name])
    ]
    if differing:
        raise ValueError(
            "retained training schedule differs from canonical derivation: "
            + ", ".join(differing)
        )
    return canonical


def materialize_sequence_batch(
    replay_arrays: Mapping[str, np.ndarray],
    schedule: Mapping[str, np.ndarray],
    update: int,
    objective: ModelTrainingConfig,
) -> dict[str, Any]:
    """Materialize one reset-aware batch with an explicit burn-in mask."""

    import jax.numpy as jnp

    if not 0 <= int(update) < objective.updates:
        raise ValueError("update is outside the frozen schedule")
    episodes = np.asarray(schedule["episode_ids"])[update]
    starts = np.asarray(schedule["starts"])[update]
    required = ("observations", "actions", "rewards", "continuations")
    if any(name not in replay_arrays for name in required):
        raise KeyError(f"replay arrays must contain {required}")
    batch: dict[str, np.ndarray] = {}
    names = required + (("is_first",) if "is_first" in replay_arrays else ())
    for name in names:
        values = np.asarray(replay_arrays[name])
        batch[name] = np.stack(
            [
                values[
                    int(episode), int(start) : int(start) + objective.sequence_length
                ]
                for episode, start in zip(episodes, starts, strict=True)
            ]
        )
    if "is_first" not in batch:
        batch["is_first"] = np.zeros(
            (objective.batch_size, objective.sequence_length), dtype=np.bool_
        )
    loss_mask = np.ones(
        (objective.batch_size, objective.sequence_length), dtype=np.float32
    )
    loss_mask[:, : objective.burn_in] = 0.0
    batch["loss_mask"] = loss_mask
    return {name: jnp.asarray(value) for name, value in batch.items()}


def _validate_trajectory_source(
    frozen_world_model: Mapping[str, Any], config: Any
) -> None:
    if config.prior != "imf" or not config.imf_trajectory_enabled:
        raise ValueError("model continuation requires a trajectory-iMF source")
    required = {
        "encoder",
        "decoder",
        "recurrence",
        "posterior",
        "prior",
        "reward",
        "continuation",
    }
    if not required <= set(frozen_world_model):
        raise ValueError("authenticated source world model is incomplete")


def _control_subtrees(control_state: Any) -> tuple[Any, Any]:
    if hasattr(control_state, "actor") and hasattr(control_state, "critics"):
        return control_state.actor, control_state.critics
    if isinstance(control_state, Mapping) and {
        "actor",
        "critics",
    } <= set(control_state):
        return control_state["actor"], control_state["critics"]
    raise ValueError("actor-specific training requires frozen actor and critics")


def _replace_prior(
    frozen_world_model: Mapping[str, Any], prior_params: Any
) -> dict[str, Any]:
    result = dict(frozen_world_model)
    result["prior"] = prior_params
    return result


def _valid_token_mask(batch: Mapping[str, Any]) -> Any:
    import jax.numpy as jnp

    return jnp.asarray(batch["loss_mask"]) * (
        ~jnp.asarray(batch["is_first"], dtype=jnp.bool_)
    ).astype(batch["loss_mask"].dtype)


def _trajectory_tilt_weights(
    batch: Mapping[str, Any],
    actor_params: Any,
    objective: ModelTrainingConfig,
) -> tuple[Any, Any, Any]:
    import jax.numpy as jnp
    from imf_dreamer_jax import rebrac_actor
    from imf_dreamer_jax.control_aware_imf import (
        PolicyDensityTiltConfig,
        approximate_fixed_variance_gaussian_log_probability,
        policy_density_tilt_weights,
    )

    observations = jnp.asarray(batch["observations"])
    flat_observations = observations.reshape((*observations.shape[:2], -1))
    previous = jnp.concatenate(
        (jnp.zeros_like(flat_observations[:, :1]), flat_observations[:, :-1]),
        axis=1,
    )
    policy_actions = rebrac_actor(actor_params, previous)
    actions = jnp.asarray(batch["actions"], dtype=policy_actions.dtype)
    log_probability = approximate_fixed_variance_gaussian_log_probability(
        actions,
        policy_actions,
        fixed_std=objective.approximate_policy_std,
    )
    mask = _valid_token_mask(batch) > 0.0
    details = policy_density_tilt_weights(
        log_probability,
        PolicyDensityTiltConfig(eta=objective.tilt_eta),
        mask=mask,
        return_details=True,
    )
    fraction = details.effective_sample_size / jnp.maximum(details.valid_count, 1.0)
    return details.weights, fraction, details.clipped_fraction


def _cvaml_prior_loss(
    prior_params: Any,
    frozen_world_model: Mapping[str, Any],
    sequence: Any,
    batch: Mapping[str, Any],
    key: Any,
    dreamer_config: Any,
    objective: ModelTrainingConfig,
    control_state: Any,
) -> Any:
    """Proof-consistent one-step auxiliary for the deployed FlowMPC functional.

    It uses the frozen published ``r(o_t, a_t)`` head, a fixed planner discount
    (equivalently continuation one), and the frozen ReBRAC terminal minimum-Q.
    Neither the legacy latent reward decoder nor the learned continuation head
    participates in this objective.
    """

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import rebrac_actor, rebrac_critics
    from imf_dreamer_jax.control_aware_imf import (
        PlannerValueEquivalenceConfig,
        cvaml_compatible_bellman_residual_loss,
    )
    from imf_dreamer_jax.types import RSSMState
    from imf_dreamer_jax.world_model import (
        decode,
        predict_state_action_reward_from_observation,
        sample_prior_with_nfe,
        transition_deterministic,
    )

    actor_params, critic_params = _control_subtrees(control_state)
    full_model = _replace_prior(frozen_world_model, prior_params)
    batch_size, steps = batch["actions"].shape[:2]
    pair_steps = steps - 1

    def flatten(value: Any) -> Any:
        return value.reshape((batch_size * pair_steps, value.shape[-1]))

    current = RSSMState(
        flatten(jax.lax.stop_gradient(sequence.states.deterministic[:, :-1])),
        flatten(jax.lax.stop_gradient(sequence.states.stochastic[:, :-1])),
    )
    action = flatten(jnp.asarray(batch["actions"][:, 1:]))
    current_observation = jnp.asarray(batch["observations"][:, :-1]).reshape(
        (batch_size * pair_steps, *dreamer_config.observation_shape)
    )
    if "reward_transition" not in full_model:
        raise ValueError(
            "planner-matched CVAML requires the published state-action reward head"
        )
    # Match FlowMPC exactly: r(o_t, a_t) is evaluated before the stochastic
    # transition, and the planner uses a fixed discount rather than a learned
    # continuation probability.  The logged action is the decision variable
    # for this replay-supervised Bellman comparison.
    current_reward = predict_state_action_reward_from_observation(
        full_model["reward_transition"],
        current_observation,
        action,
        dreamer_config,
    )
    deterministic = transition_deterministic(
        full_model, current, action, dreamer_config
    )
    noise = jax.random.normal(
        key,
        (
            objective.cvaml_samples,
            batch_size * pair_steps,
            dreamer_config.stochastic_dim,
        ),
        dtype=deterministic.dtype,
    )

    def one_sample(sample_noise: Any) -> Any:
        stochastic = sample_prior_with_nfe(
            full_model,
            deterministic,
            None,
            dreamer_config,
            noise=sample_noise,
        ).stochastic
        next_feature = jnp.concatenate((deterministic, stochastic), axis=-1)
        next_observation = decode(full_model, next_feature, dreamer_config).reshape(
            (batch_size * pair_steps, -1)
        )
        next_action = rebrac_actor(actor_params, next_observation)
        next_value = jnp.min(
            rebrac_critics(critic_params, next_observation, next_action),
            axis=0,
        )
        return next_value

    model_next_value = jax.vmap(one_sample)(noise)
    model_reward = jnp.broadcast_to(current_reward[None], model_next_value.shape)
    model_continuation = jnp.ones_like(model_next_value)
    next_observation_real = jnp.asarray(batch["observations"][:, 1:]).reshape(
        (batch_size * pair_steps, -1)
    )
    real_next_action = rebrac_actor(actor_params, next_observation_real)
    real_next_value = jnp.min(
        rebrac_critics(critic_params, next_observation_real, real_next_action),
        axis=0,
    )
    real_reward = jnp.asarray(batch["rewards"][:, 1:]).reshape((-1,))
    real_continuation = jnp.ones_like(real_reward)
    valid = (
        jnp.asarray(batch["loss_mask"][:, 1:])
        * (~jnp.asarray(batch["is_first"][:, 1:], dtype=jnp.bool_)).astype(
            real_reward.dtype
        )
        * (jnp.asarray(batch["continuations"][:, :-1]) > 0.0).astype(real_reward.dtype)
    ).reshape((-1,))
    return cvaml_compatible_bellman_residual_loss(
        model_reward,
        model_continuation,
        model_next_value,
        real_reward,
        real_continuation,
        real_next_value,
        PlannerValueEquivalenceConfig(discount=objective.planner_discount),
        sample_axis=0,
        normalization_scale=objective.bellman_target_scale,
        mask=valid,
        return_details=True,
    )


def trajectory_prior_compound_loss(
    prior_params: Any,
    frozen_world_model: Mapping[str, Any],
    batch: Mapping[str, Any],
    key: Any,
    dreamer_config: Any,
    objective: ModelTrainingConfig,
    *,
    family: ModelFamily,
    control_state: Any | None = None,
) -> PriorLossMetrics:
    """Evaluate one trajectory-iMF prior-only continuation objective."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax.trajectory import (
        corrupt_trajectory,
        sample_trajectory_schedule,
        trajectory_imf_loss,
    )
    from imf_dreamer_jax.world_model import (
        observe_sequence,
        trajectory_deterministic_conditions,
    )

    if family not in PRIOR_FAMILIES:
        raise ValueError("trajectory_prior_compound_loss requires a prior family")
    _validate_trajectory_source(frozen_world_model, dreamer_config)
    if (family in ACTOR_SPECIFIC_PRIOR_FAMILIES) != (control_state is not None):
        raise ValueError("control state presence differs from family identity")
    full_model = _replace_prior(frozen_world_model, prior_params)
    sequence_key, noise_key, uniform_key, schedule_key, loss_key, cvaml_key = (
        jax.random.split(key, 6)
    )
    sequence = observe_sequence(
        full_model,
        batch["observations"],
        batch["actions"],
        sequence_key,
        dreamer_config,
        is_first=batch["is_first"],
    )
    targets = jax.lax.stop_gradient(sequence.states.stochastic)
    noise = jax.random.normal(noise_key, targets.shape, dtype=targets.dtype)
    canonical_uniforms = jax.random.uniform(
        uniform_key, (*targets.shape[:2], 6), dtype=targets.dtype
    )
    schedule = sample_trajectory_schedule(
        schedule_key,
        targets.shape[0],
        targets.shape[1],
        token_mask=batch["loss_mask"],
        dtype=targets.dtype,
        boundary_fraction=dreamer_config.imf_boundary_fraction,
        time_mean=dreamer_config.imf_time_mean,
        time_std=dreamer_config.imf_time_std,
        history_noise_max=dreamer_config.imf_trajectory_history_noise_max,
        pattern_probabilities=(
            dreamer_config.imf_trajectory_clean_probability,
            dreamer_config.imf_trajectory_corrupted_probability,
            dreamer_config.imf_trajectory_suffix_probability,
        ),
        canonical_uniforms=canonical_uniforms,
    )
    corrupted_history = corrupt_trajectory(targets, noise, schedule.history_t)
    conditions = trajectory_deterministic_conditions(
        full_model,
        corrupted_history,
        schedule.history_t,
        batch["actions"],
        batch["is_first"],
        dreamer_config,
    )
    if family == "approximate_policy_tilt_prior":
        actor_params, _ = _control_subtrees(control_state)
        weights, ess_fraction, clipped_fraction = _trajectory_tilt_weights(
            batch, actor_params, objective
        )
    else:
        weights = None
        ess_fraction = jnp.asarray(1.0, dtype=targets.dtype)
        clipped_fraction = jnp.asarray(0.0, dtype=targets.dtype)
    trajectory = trajectory_imf_loss(
        prior_params,
        targets,
        conditions,
        loss_key,
        noise=noise,
        r=schedule.r,
        t=schedule.t,
        token_mask=schedule.loss_mask,
        weights=weights,
        adaptive_power=dreamer_config.imf_adaptive_power,
        adaptive_epsilon=dreamer_config.imf_adaptive_epsilon,
        meanflow_scale=dreamer_config.imf_meanflow_scale,
        velocity_scale=dreamer_config.imf_velocity_scale,
        signal_weight_floor=dreamer_config.imf_signal_weight_floor,
        signal_weight_scale=dreamer_config.imf_signal_weight_scale,
        condition_gradient_scale=dreamer_config.imf_condition_gradient_scale,
        boundary_velocity_supervision=True,
    )
    zero = jnp.asarray(0.0, dtype=targets.dtype)
    corrected = zero
    uncorrected = zero
    correction = zero
    if family == "proof_consistent_cvaml_value_prior":
        details = _cvaml_prior_loss(
            prior_params,
            frozen_world_model,
            sequence,
            batch,
            cvaml_key,
            dreamer_config,
            objective,
            control_state,
        )
        corrected = details.loss
        uncorrected = details.uncorrected_loss
        correction = details.variance_correction
    total = trajectory + objective.cvaml_scale * corrected
    return PriorLossMetrics(
        total=total,
        trajectory=trajectory,
        cvaml_corrected=corrected,
        cvaml_uncorrected=uncorrected,
        cvaml_variance_correction=correction,
        tilt_effective_sample_size_fraction=ess_fraction,
        tilt_clipped_fraction=clipped_fraction,
        valid_token_fraction=jnp.mean(_valid_token_mask(batch) > 0.0),
    )


def _endpoint_statistics(
    frozen_world_model: Mapping[str, Any], dreamer_config: Any
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    if "reward_transition" in frozen_world_model:
        head = frozen_world_model["reward_transition"]
        if isinstance(head, Mapping) and {
            "observation_mean",
            "observation_std",
        } <= set(head):
            mean = jnp.asarray(head["observation_mean"], dtype=jnp.float32)
            std = jnp.asarray(head["observation_std"], dtype=jnp.float32)
            if mean.shape == (dreamer_config.observation_dim,) and std.shape == (
                dreamer_config.observation_dim,
            ):
                return mean, jnp.maximum(std, 1e-6)
    return (
        jnp.zeros((dreamer_config.observation_dim,), dtype=jnp.float32),
        jnp.ones((dreamer_config.observation_dim,), dtype=jnp.float32),
    )


def endpoint_condition_dim(dreamer_config: Any, maximum_chunk_horizon: int) -> int:
    if maximum_chunk_horizon <= 0:
        raise ValueError("maximum_chunk_horizon must be positive")
    return (
        dreamer_config.observation_dim
        + maximum_chunk_horizon * dreamer_config.action_dim
        + maximum_chunk_horizon
    )


def build_endpoint_condition(
    start_observation: Any,
    action_chunks: Any,
    horizon: Any,
    dreamer_config: Any,
    *,
    observation_mean: Any,
    observation_std: Any,
) -> Any:
    """Build the exact fixed-width endpoint condition used at train and test.

    ``action_chunks`` must already have ``maximum_chunk_horizon`` entries.
    Entries beyond ``horizon`` are zeroed internally, and the same binary mask
    used during training is appended.  ``horizon`` may be scalar or have the
    leading shape of ``start_observation``.
    """

    import jax.numpy as jnp

    observation = jnp.asarray(start_observation)
    actions = jnp.asarray(action_chunks, dtype=observation.dtype)
    if observation.ndim < 2:
        raise ValueError("start_observation must include batch and feature axes")
    if observation.shape[-len(dreamer_config.observation_shape) :] != (
        dreamer_config.observation_shape
    ):
        raise ValueError("start_observation has the wrong trailing shape")
    leading = observation.shape[: -len(dreamer_config.observation_shape)]
    if actions.ndim != len(leading) + 2 or actions.shape[:-2] != leading:
        raise ValueError("action_chunks leading dimensions differ from observation")
    if actions.shape[-1] != dreamer_config.action_dim:
        raise ValueError("action_chunks has the wrong action dimension")
    maximum_horizon = actions.shape[-2]
    if maximum_horizon <= 0:
        raise ValueError("action_chunks must contain at least one action")
    if (
        isinstance(horizon, (int, np.integer))
        and not 1 <= int(horizon) <= maximum_horizon
    ):
        raise ValueError("horizon lies outside the padded action chunk")
    horizon_array = jnp.asarray(horizon, dtype=jnp.int32)
    try:
        horizon_array = jnp.broadcast_to(horizon_array, leading)
    except ValueError as error:
        raise ValueError(
            "horizon must broadcast to the observation leading shape"
        ) from error
    horizon_array = jnp.clip(horizon_array, 1, maximum_horizon)
    offsets = jnp.arange(maximum_horizon, dtype=jnp.int32)
    mask = offsets < horizon_array[..., None]
    masked_actions = jnp.where(mask[..., None], actions, 0.0)
    flat_observation = observation.reshape((*leading, dreamer_config.observation_dim))
    mean = jnp.asarray(observation_mean, dtype=flat_observation.dtype)
    std = jnp.asarray(observation_std, dtype=flat_observation.dtype)
    if mean.shape != (dreamer_config.observation_dim,) or std.shape != (
        dreamer_config.observation_dim,
    ):
        raise ValueError("endpoint observation statistics have the wrong shape")
    standardized = (flat_observation - mean) / jnp.maximum(std, 1e-6)
    condition = jnp.concatenate(
        (
            standardized,
            masked_actions.reshape((*leading, -1)),
            mask.astype(standardized.dtype),
        ),
        axis=-1,
    )
    expected = endpoint_condition_dim(dreamer_config, maximum_horizon)
    if condition.shape != (*leading, expected):
        raise AssertionError("endpoint inference condition dimension differs")
    return condition


def sample_endpoint_observation(
    endpoint_params: Any,
    start_observation: Any,
    action_chunks: Any,
    horizon: Any,
    dreamer_config: Any,
    *,
    observation_mean: Any,
    observation_std: Any,
    key: Any | None = None,
    noise: Any | None = None,
) -> Any:
    """Sample a direct endpoint with the exact standardized training layout."""

    import jax.numpy as jnp
    from imf_dreamer_jax.imf import sample_imf_one_step

    condition = build_endpoint_condition(
        start_observation,
        action_chunks,
        horizon,
        dreamer_config,
        observation_mean=observation_mean,
        observation_std=observation_std,
    )
    leading = condition.shape[:-1]
    flat_condition = condition.reshape((-1, condition.shape[-1]))
    flat_noise = None
    if noise is not None:
        noise_array = jnp.asarray(noise, dtype=flat_condition.dtype)
        expected = (*leading, dreamer_config.observation_dim)
        if noise_array.shape != expected:
            raise ValueError("endpoint noise has the wrong shape")
        flat_noise = noise_array.reshape((-1, dreamer_config.observation_dim))
    standardized = sample_imf_one_step(
        endpoint_params, flat_condition, key, noise=flat_noise
    ).reshape((*leading, dreamer_config.observation_dim))
    mean = jnp.asarray(observation_mean, dtype=standardized.dtype)
    std = jnp.asarray(observation_std, dtype=standardized.dtype)
    observation = standardized * std + mean
    return observation.reshape((*leading, *dreamer_config.observation_shape))


def build_matched_endpoint_examples(
    batch: Mapping[str, Any],
    start_uniforms: Any,
    horizon: Any,
    dreamer_config: Any,
    objective: ModelTrainingConfig,
    *,
    observation_mean: Any,
    observation_std: Any,
) -> tuple[Any, Any, Any]:
    """Build a fixed-size, fixed-condition-layout direct endpoint batch.

    Each sequence contributes exactly one example.  ``horizon`` may be a JAX
    scalar, so one compiled function serves all any-step horizons.  Actions
    are aligned as ``actions[t+1]`` for ``observation[t] -> observation[t+1]``.
    """

    import jax.numpy as jnp

    observations = jnp.asarray(batch["observations"])
    actions = jnp.asarray(batch["actions"])
    if observations.shape[:2] != actions.shape[:2]:
        raise ValueError("endpoint observations/actions must align")
    batch_size, state_count = observations.shape[:2]
    if state_count != objective.sequence_length:
        raise ValueError("endpoint batch differs from registered sequence length")
    uniforms = jnp.asarray(start_uniforms, dtype=jnp.float32)
    if uniforms.shape != (batch_size,):
        raise ValueError("endpoint start uniforms need one value per sequence")
    horizon = jnp.asarray(horizon, dtype=jnp.int32)
    horizon = jnp.clip(horizon, 1, objective.maximum_chunk_horizon)
    possible_starts = state_count - horizon
    start = jnp.minimum(
        jnp.floor(uniforms * possible_starts).astype(jnp.int32),
        possible_starts - 1,
    )
    rows = jnp.arange(batch_size, dtype=jnp.int32)
    flat_observations = observations.reshape((batch_size, state_count, -1))
    mean = jnp.asarray(observation_mean, dtype=flat_observations.dtype)
    std = jnp.asarray(observation_std, dtype=flat_observations.dtype)
    if mean.shape != (dreamer_config.observation_dim,) or std.shape != (
        dreamer_config.observation_dim,
    ):
        raise ValueError("endpoint observation statistics have the wrong shape")
    standardized = (flat_observations - mean) / jnp.maximum(std, 1e-6)
    start_condition = standardized[rows, start]
    target = standardized[rows, start + horizon]

    offsets = jnp.arange(objective.maximum_chunk_horizon, dtype=jnp.int32)
    action_mask = offsets < horizon
    # Stored action at k+1 generated the transition from state k to k+1.
    action_indices = jnp.minimum(start[:, None] + offsets[None, :] + 1, state_count - 1)
    selected_actions = actions[rows[:, None], action_indices]
    selected_actions = jnp.where(action_mask[None, :, None], selected_actions, 0.0)
    condition = jnp.concatenate(
        (
            start_condition,
            selected_actions.reshape((batch_size, -1)),
            jnp.broadcast_to(action_mask[None], (batch_size, action_mask.size)).astype(
                start_condition.dtype
            ),
        ),
        axis=-1,
    )
    expected_dim = endpoint_condition_dim(
        dreamer_config, objective.maximum_chunk_horizon
    )
    if condition.shape != (batch_size, expected_dim):
        raise AssertionError("endpoint condition dimension is not matched")

    continuations = jnp.asarray(batch["continuations"])
    is_first = jnp.asarray(batch["is_first"], dtype=jnp.bool_)
    loss_mask = jnp.asarray(batch["loss_mask"])
    current_indices = jnp.minimum(start[:, None] + offsets[None, :], state_count - 1)
    next_indices = jnp.minimum(current_indices + 1, state_count - 1)
    per_step_valid = (
        (continuations[rows[:, None], current_indices] > 0.0)
        & (~is_first[rows[:, None], next_indices])
        & (loss_mask[rows[:, None], next_indices] > 0.0)
    )
    valid = jnp.all(jnp.where(action_mask[None], per_step_valid, True), axis=-1)
    return target, condition, valid.astype(target.dtype)


def endpoint_compound_loss(
    endpoint_params: Any,
    frozen_world_model: Mapping[str, Any],
    batch: Mapping[str, Any],
    start_uniforms: Any,
    horizon: Any,
    key: Any,
    dreamer_config: Any,
    objective: ModelTrainingConfig,
) -> EndpointLossMetrics:
    """Evaluate the matched full compound iMF endpoint objective."""

    import jax.numpy as jnp
    from imf_dreamer_jax.imf import improved_meanflow_loss

    if objective.endpoint_scale <= 0.0:
        raise ValueError("endpoint_scale must stay positive for endpoint models")
    mean, std = _endpoint_statistics(frozen_world_model, dreamer_config)
    target, condition, weights = build_matched_endpoint_examples(
        batch,
        start_uniforms,
        horizon,
        dreamer_config,
        objective,
        observation_mean=mean,
        observation_std=std,
    )
    details = improved_meanflow_loss(
        endpoint_params,
        target,
        condition,
        key,
        boundary_fraction=dreamer_config.imf_boundary_fraction,
        time_mean=dreamer_config.imf_time_mean,
        time_std=dreamer_config.imf_time_std,
        adaptive_power=dreamer_config.imf_adaptive_power,
        adaptive_epsilon=dreamer_config.imf_adaptive_epsilon,
        meanflow_scale=dreamer_config.imf_meanflow_scale,
        velocity_scale=dreamer_config.imf_velocity_scale,
        endpoint_scale=objective.endpoint_scale,
        shortcut_scale=0.0,
        signal_weight_floor=dreamer_config.imf_signal_weight_floor,
        signal_weight_scale=dreamer_config.imf_signal_weight_scale,
        condition_gradient_scale=dreamer_config.imf_condition_gradient_scale,
        boundary_velocity_supervision=True,
        weights=weights,
        return_details=True,
    )

    def weighted_mean(value: Any) -> Any:
        return jnp.sum(value * weights) / jnp.maximum(
            jnp.sum(weights), jnp.finfo(value.dtype).tiny
        )

    return EndpointLossMetrics(
        total=details.loss,
        meanflow_raw_mse=weighted_mean(details.raw_loss_u),
        velocity_raw_mse=weighted_mean(details.raw_loss_v),
        endpoint_raw_mse=weighted_mean(details.raw_loss_endpoint),
        accepted_fraction=jnp.mean(weights > 0.0),
        horizon=jnp.asarray(horizon, dtype=jnp.float32),
    )


def init_model_train_state(
    spec: ModelCellSpec,
    frozen_world_model: Mapping[str, Any],
    dreamer_config: Any,
    objective: ModelTrainingConfig,
    *,
    task: str,
) -> ModelTrainState:
    """Initialize only the permitted subtree with a fresh Adam time base."""

    import jax
    from imf_dreamer_jax.imf import init_imf
    from imf_dreamer_jax.optim import init_adam

    validate_model_cell_spec(spec)
    _validate_trajectory_source(frozen_world_model, dreamer_config)
    if spec.family in PRIOR_FAMILIES:
        # JAX arrays are immutable; making new leaves makes the isolation
        # explicit and prevents a caller's container identity from leaking.
        params = jax.tree_util.tree_map(
            lambda value: jax.numpy.array(value), frozen_world_model["prior"]
        )
    else:
        hidden_dim = (
            dreamer_config.hidden_dim
            if objective.endpoint_hidden_dim is None
            else objective.endpoint_hidden_dim
        )
        # Family is intentionally absent: h1 and any-step are exactly matched.
        key = jax.random.PRNGKey(
            _derive_seed("actor-gap-endpoint-init", task, int(spec.world_model_seed))
        )
        params = init_imf(
            key,
            sample_dim=dreamer_config.observation_dim,
            condition_dim=endpoint_condition_dim(
                dreamer_config, objective.maximum_chunk_horizon
            ),
            hidden_dim=hidden_dim,
            depth=objective.endpoint_depth,
        )
    optimizer = init_adam(params)
    if int(np.asarray(optimizer.step)) != 0:
        raise AssertionError("fresh model optimizer did not start at step zero")
    return ModelTrainState(params=params, optimizer=optimizer)


def train_prior_step(
    state: ModelTrainState,
    frozen_world_model: Mapping[str, Any],
    batch: Mapping[str, Any],
    key: Any,
    dreamer_config: Any,
    objective: ModelTrainingConfig,
    *,
    family: ModelFamily,
    control_state: Any | None = None,
) -> tuple[ModelTrainState, TrainStepMetrics]:
    """Apply one clipped Adam update to the prior and nothing else."""

    import jax
    from imf_dreamer_jax.nn import clip_by_global_norm
    from imf_dreamer_jax.optim import adam_update

    def loss_fn(params: Any) -> tuple[Any, PriorLossMetrics]:
        details = trajectory_prior_compound_loss(
            params,
            frozen_world_model,
            batch,
            key,
            dreamer_config,
            objective,
            family=family,
            control_state=control_state,
        )
        return details.total, details

    (_, details), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    gradients, grad_norm = clip_by_global_norm(gradients, objective.grad_clip)
    params, optimizer = adam_update(
        state.params,
        gradients,
        state.optimizer,
        learning_rate=objective.learning_rate,
        beta1=dreamer_config.adam_beta1,
        beta2=dreamer_config.adam_beta2,
        epsilon=dreamer_config.adam_epsilon,
    )
    return ModelTrainState(params, optimizer), TrainStepMetrics(
        total=details.total,
        primary=details.trajectory,
        auxiliary=details.cvaml_corrected,
        uncorrected_auxiliary=details.cvaml_uncorrected,
        correction=details.cvaml_variance_correction,
        accepted_fraction=details.valid_token_fraction,
        tilt_effective_sample_size_fraction=(
            details.tilt_effective_sample_size_fraction
        ),
        tilt_clipped_fraction=details.tilt_clipped_fraction,
        horizon=jax.numpy.asarray(0.0, dtype=details.total.dtype),
        grad_norm=grad_norm,
    )


def train_endpoint_step(
    state: ModelTrainState,
    frozen_world_model: Mapping[str, Any],
    batch: Mapping[str, Any],
    start_uniforms: Any,
    horizon: Any,
    key: Any,
    dreamer_config: Any,
    objective: ModelTrainingConfig,
) -> tuple[ModelTrainState, TrainStepMetrics]:
    """Apply one clipped Adam update to a fresh direct endpoint model."""

    import jax
    from imf_dreamer_jax.nn import clip_by_global_norm
    from imf_dreamer_jax.optim import adam_update

    def loss_fn(params: Any) -> tuple[Any, EndpointLossMetrics]:
        details = endpoint_compound_loss(
            params,
            frozen_world_model,
            batch,
            start_uniforms,
            horizon,
            key,
            dreamer_config,
            objective,
        )
        return details.total, details

    (_, details), gradients = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    gradients, grad_norm = clip_by_global_norm(gradients, objective.grad_clip)
    params, optimizer = adam_update(
        state.params,
        gradients,
        state.optimizer,
        learning_rate=objective.learning_rate,
        beta1=dreamer_config.adam_beta1,
        beta2=dreamer_config.adam_beta2,
        epsilon=dreamer_config.adam_epsilon,
    )
    return ModelTrainState(params, optimizer), TrainStepMetrics(
        total=details.total,
        primary=details.meanflow_raw_mse + details.velocity_raw_mse,
        auxiliary=details.endpoint_raw_mse,
        uncorrected_auxiliary=details.endpoint_raw_mse,
        correction=jax.numpy.asarray(0.0, dtype=details.total.dtype),
        accepted_fraction=details.accepted_fraction,
        tilt_effective_sample_size_fraction=jax.numpy.asarray(
            1.0, dtype=details.total.dtype
        ),
        tilt_clipped_fraction=jax.numpy.asarray(0.0, dtype=details.total.dtype),
        horizon=details.horizon,
        grad_norm=grad_norm,
    )


def _finite_tree(tree: Any) -> bool:
    import jax

    return all(
        bool(np.all(np.isfinite(np.asarray(value))))
        for value in jax.tree_util.tree_leaves(tree)
    )


def _scalar_metrics(metrics: TrainStepMetrics) -> dict[str, float]:
    import jax

    return {
        name: float(np.asarray(jax.device_get(value)))
        for name, value in metrics._asdict().items()
    }


def train_model_cell_payload(
    spec: ModelCellSpec,
    frozen_world_model: Mapping[str, Any],
    dreamer_config: Any,
    replay_arrays: Mapping[str, np.ndarray],
    objective: ModelTrainingConfig,
    *,
    task: str,
    control_state: Any | None = None,
    schedule: Mapping[str, np.ndarray] | None = None,
    jit: bool = True,
) -> dict[str, Any]:
    """Train one deterministic cell and return an I/O-agnostic payload.

    The returned checkpoint contains only the permitted trainable subtree and
    fresh optimizer.  Study code is expected to serialize it atomically and
    bind it to its own manifest, source commit, and authenticated dependencies.
    """

    import jax
    import jax.numpy as jnp

    validate_model_cell_spec(spec)
    _validate_trajectory_source(frozen_world_model, dreamer_config)
    actor_specific = spec.family in ACTOR_SPECIFIC_PRIOR_FAMILIES
    if actor_specific != (control_state is not None):
        raise ValueError("actor-specific cell/control-state mismatch")
    if schedule is None:
        schedule = build_training_schedule(
            replay_arrays,
            task=task,
            world_model_seed=spec.world_model_seed,
            objective=objective,
        )
    schedule = {name: np.asarray(value) for name, value in schedule.items()}
    validate_training_schedule(schedule, replay_arrays, objective)
    source_before = _tree_sha256(frozen_world_model)
    state = init_model_train_state(
        spec, frozen_world_model, dreamer_config, objective, task=task
    )
    initial_params_sha256 = _tree_sha256(state.params)
    if int(np.asarray(state.optimizer.step)) != 0:
        raise AssertionError("model continuation inherited optimizer moments")

    if spec.family in PRIOR_FAMILIES:

        def step_function(
            current: ModelTrainState,
            batch: Mapping[str, Any],
            key: Any,
            _: Any,
            __: Any,
        ) -> tuple[ModelTrainState, TrainStepMetrics]:
            return train_prior_step(
                current,
                frozen_world_model,
                batch,
                key,
                dreamer_config,
                objective,
                family=spec.family,
                control_state=control_state,
            )

    else:

        def step_function(
            current: ModelTrainState,
            batch: Mapping[str, Any],
            key: Any,
            start_uniforms: Any,
            horizon: Any,
        ) -> tuple[ModelTrainState, TrainStepMetrics]:
            return train_endpoint_step(
                current,
                frozen_world_model,
                batch,
                start_uniforms,
                horizon,
                key,
                dreamer_config,
                objective,
            )

    compiled_step = jax.jit(step_function) if jit else step_function
    # Compile/warm without mutating the live state.  This mirrors the study's
    # deterministic first-execution repair and keeps compilation out of timing.
    warm_batch = materialize_sequence_batch(replay_arrays, schedule, 0, objective)
    warm_horizon = (
        schedule["endpoint_horizons_h1"][0]
        if spec.family == "endpoint_h1_imf"
        else schedule["endpoint_horizons_any_step"][0]
    )
    warm_output = compiled_step(
        state,
        warm_batch,
        jnp.asarray(schedule["objective_keys"][0], dtype=jnp.uint32),
        jnp.asarray(schedule["endpoint_start_uniforms"][0]),
        jnp.asarray(warm_horizon, dtype=jnp.int32),
    )
    jax.block_until_ready(warm_output[1].total)

    metric_rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for update in range(objective.updates):
        batch = materialize_sequence_batch(replay_arrays, schedule, update, objective)
        if spec.family == "endpoint_h1_imf":
            horizon = schedule["endpoint_horizons_h1"][update]
        else:
            horizon = schedule["endpoint_horizons_any_step"][update]
        state, metrics = compiled_step(
            state,
            batch,
            jnp.asarray(schedule["objective_keys"][update], dtype=jnp.uint32),
            jnp.asarray(schedule["endpoint_start_uniforms"][update]),
            jnp.asarray(horizon, dtype=jnp.int32),
        )
        jax.block_until_ready(metrics.total)
        row = _scalar_metrics(metrics)
        if not np.all(np.isfinite(np.asarray(list(row.values()), dtype=np.float64))):
            raise FloatingPointError("model training produced non-finite metrics")
        metric_rows.append(row)
    wall_seconds = time.perf_counter() - started

    source_after = _tree_sha256(frozen_world_model)
    if source_after != source_before:
        raise RuntimeError("frozen source world model changed during training")
    if int(np.asarray(state.optimizer.step)) != objective.updates:
        raise RuntimeError("fresh optimizer step count differs from update budget")
    if not _finite_tree(state.params) or not _finite_tree(state.optimizer):
        raise FloatingPointError("model checkpoint contains non-finite arrays")
    final_params_sha256 = _tree_sha256(state.params)
    final_optimizer_sha256 = _tree_sha256(state.optimizer)
    immutable_subtrees = sorted(
        name
        for name in frozen_world_model
        if spec.family in ENDPOINT_FAMILIES or name != "prior"
    )
    final_metrics = metric_rows[-1]
    metric_summary = {
        "final": final_metrics,
        "mean": {
            name: float(np.mean([row[name] for row in metric_rows]))
            for name in final_metrics
        },
        "maximum_grad_norm": float(max(row["grad_norm"] for row in metric_rows)),
    }
    mean, std = _endpoint_statistics(frozen_world_model, dreamer_config)
    checkpoint = {
        "trainable_subtree": (
            "prior" if spec.family in PRIOR_FAMILIES else "endpoint_imf"
        ),
        "params": jax.device_get(state.params),
        "optimizer": jax.device_get(state.optimizer),
        "observation_mean": np.asarray(jax.device_get(mean)),
        "observation_std": np.asarray(jax.device_get(std)),
        "maximum_chunk_horizon": objective.maximum_chunk_horizon,
        "condition_dim": (
            dreamer_config.deterministic_dim
            if spec.family in PRIOR_FAMILIES
            else endpoint_condition_dim(dreamer_config, objective.maximum_chunk_horizon)
        ),
    }
    result = {
        "schema_version": SCHEMA,
        "cell": asdict(spec),
        "task": task,
        "objective": asdict(objective),
        "schedule_sha256": schedule_sha256(schedule),
        "source_world_model_sha256_before": source_before,
        "source_world_model_sha256_after": source_after,
        "immutable_source_subtrees": immutable_subtrees,
        "trainable_subtree": checkpoint["trainable_subtree"],
        "initial_trainable_sha256": initial_params_sha256,
        "final_trainable_sha256": final_params_sha256,
        "final_optimizer_sha256": final_optimizer_sha256,
        "optimizer_initial_step": 0,
        "optimizer_final_step": int(np.asarray(state.optimizer.step)),
        "fresh_zero_moment_adam": True,
        "policy_density_label": (
            APPROXIMATE_POLICY_DENSITY_LABEL
            if spec.family == "approximate_policy_tilt_prior"
            else None
        ),
        "cvaml_estimator_label": (
            PROOF_CONSISTENT_CVAML_LABEL
            if spec.family == "proof_consistent_cvaml_value_prior"
            else None
        ),
        "endpoint_objective_label": (
            ENDPOINT_OBJECTIVE_LABEL if spec.family in ENDPOINT_FAMILIES else None
        ),
        "matched_endpoint_initialization_namespace": (
            f"actor-gap-endpoint-init:{task}:{spec.world_model_seed}"
            if spec.family in ENDPOINT_FAMILIES
            else None
        ),
        "metrics": metric_summary,
        "wall_seconds_excluding_discarded_compile_warmup": float(wall_seconds),
        "checkpoint": checkpoint,
        "schedule": schedule,
    }
    return result


def deterministic_payload_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return only replay-stable fields; runtime is intentionally excluded."""

    checkpoint = payload["checkpoint"]
    return {
        "schema_version": payload.get("schema_version"),
        "cell": payload.get("cell"),
        "task": payload.get("task"),
        "objective": payload.get("objective"),
        "schedule_sha256": payload.get("schedule_sha256"),
        "source_world_model_sha256_before": payload.get(
            "source_world_model_sha256_before"
        ),
        "source_world_model_sha256_after": payload.get(
            "source_world_model_sha256_after"
        ),
        "immutable_source_subtrees": payload.get("immutable_source_subtrees"),
        "trainable_subtree": payload.get("trainable_subtree"),
        "initial_trainable_sha256": payload.get("initial_trainable_sha256"),
        "final_trainable_sha256": payload.get("final_trainable_sha256"),
        "final_optimizer_sha256": payload.get("final_optimizer_sha256"),
        "optimizer_initial_step": payload.get("optimizer_initial_step"),
        "optimizer_final_step": payload.get("optimizer_final_step"),
        "fresh_zero_moment_adam": payload.get("fresh_zero_moment_adam"),
        "policy_density_label": payload.get("policy_density_label"),
        "cvaml_estimator_label": payload.get("cvaml_estimator_label"),
        "endpoint_objective_label": payload.get("endpoint_objective_label"),
        "matched_endpoint_initialization_namespace": payload.get(
            "matched_endpoint_initialization_namespace"
        ),
        "metrics": payload.get("metrics"),
        "checkpoint_trainable_subtree": checkpoint.get("trainable_subtree"),
        "checkpoint_params_sha256": _tree_sha256(checkpoint.get("params")),
        "checkpoint_optimizer_sha256": _tree_sha256(checkpoint.get("optimizer")),
        "checkpoint_observation_mean_sha256": hashlib.sha256(
            np.ascontiguousarray(checkpoint.get("observation_mean")).tobytes()
        ).hexdigest(),
        "checkpoint_observation_std_sha256": hashlib.sha256(
            np.ascontiguousarray(checkpoint.get("observation_std")).tobytes()
        ).hexdigest(),
        "checkpoint_maximum_chunk_horizon": checkpoint.get("maximum_chunk_horizon"),
        "checkpoint_condition_dim": checkpoint.get("condition_dim"),
    }


def replay_model_cell_payload(
    retained_payload: Mapping[str, Any],
    spec: ModelCellSpec,
    frozen_world_model: Mapping[str, Any],
    dreamer_config: Any,
    replay_arrays: Mapping[str, np.ndarray],
    objective: ModelTrainingConfig,
    *,
    task: str,
    control_state: Any | None = None,
    jit: bool = True,
) -> dict[str, Any]:
    """Rerun one cell and require exact deterministic payload identity."""

    schedule = retained_payload.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("retained payload is missing its immutable schedule")
    replayed = train_model_cell_payload(
        spec,
        frozen_world_model,
        dreamer_config,
        replay_arrays,
        objective,
        task=task,
        control_state=control_state,
        schedule=schedule,
        jit=jit,
    )
    retained_identity = deterministic_payload_identity(retained_payload)
    replayed_identity = deterministic_payload_identity(replayed)
    if retained_identity != replayed_identity:
        differing = sorted(
            name
            for name in set(retained_identity) | set(replayed_identity)
            if retained_identity.get(name) != replayed_identity.get(name)
        )
        raise ValueError(
            "model cell strict replay identity differs: " + ", ".join(differing)
        )
    return {
        "schema_version": REPLAY_SCHEMA,
        "status": "verified",
        "cell_id": spec.cell_id,
        "schedule_sha256": retained_payload["schedule_sha256"],
        "final_trainable_sha256": retained_payload["final_trainable_sha256"],
        "final_optimizer_sha256": retained_payload["final_optimizer_sha256"],
        "source_world_model_sha256": retained_payload[
            "source_world_model_sha256_before"
        ],
        "strict_deterministic_replay": True,
    }


__all__ = [
    "ACTOR_SPECIFIC_PRIOR_FAMILIES",
    "APPROXIMATE_POLICY_DENSITY_LABEL",
    "ENDPOINT_FAMILIES",
    "ENDPOINT_OBJECTIVE_LABEL",
    "MODEL_FAMILY_ORDER",
    "ModelCellSpec",
    "ModelTrainState",
    "ModelTrainingConfig",
    "PRIOR_FAMILIES",
    "PROOF_CONSISTENT_CVAML_LABEL",
    "SCHEMA",
    "build_matched_endpoint_examples",
    "build_endpoint_condition",
    "build_model_cell_specs",
    "build_training_schedule",
    "deterministic_payload_identity",
    "endpoint_compound_loss",
    "endpoint_condition_dim",
    "init_model_train_state",
    "materialize_sequence_batch",
    "replay_model_cell_payload",
    "require_canonical_training_schedule",
    "sample_endpoint_observation",
    "schedule_sha256",
    "train_endpoint_step",
    "train_model_cell_payload",
    "train_prior_step",
    "trajectory_prior_compound_loss",
    "validate_model_cell_spec",
    "validate_training_schedule",
]
