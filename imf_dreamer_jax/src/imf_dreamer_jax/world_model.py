"""Functional recurrent world model with Gaussian or conditional iMF prior."""

from __future__ import annotations

from functools import partial
import math
from typing import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .fidelity import posterior_sequence_noise, transport_base_noise
from .imf import improved_meanflow_loss, init_imf, sample_imf_steps
from .nn import Params, gru, init_gru, init_mlp, mlp
from .shortcut import (
    sample_shortcut_schedule,
    sample_shortcut_steps,
    shortcut_forcing_loss,
)
from .trajectory import (
    corrupt_trajectory,
    sample_trajectory_schedule,
    trajectory_imf_loss,
)
from .types import (
    DiagonalNormal,
    ParameterCounts,
    PriorSample,
    RSSMState,
    SequenceStates,
    WorldModelLoss,
)


Batch = Mapping[str, Array]


def init_reward_head(config: DreamerConfig, key: Array) -> Params:
    """Initialize the flattened offset-by-support reward prediction head."""

    reward = init_mlp(
        key,
        config.feature_dim,
        config.hidden_dim,
        config.reward_head_output_dim,
        output_gain=config.reward_output_init_scale,
    )
    reward_layers = list(reward["layers"])
    reward_output = dict(reward_layers[-1])
    if config.reward_loss == "symexp_twohot":
        # Uniform logits are the neutral categorical initialization used by
        # distributional Dreamer heads.
        reward_bias = 0.0
    elif config.reward_min is None:
        reward_bias = config.reward_initial_value
    else:
        fraction = (config.reward_initial_value - config.reward_min) / (
            config.reward_max - config.reward_min
        )
        reward_bias = math.log(fraction) - math.log1p(-fraction)
    reward_output["bias"] = jnp.full_like(reward_output["bias"], reward_bias)
    reward_layers[-1] = reward_output
    return {"layers": tuple(reward_layers)}


def init_world_model(config: DreamerConfig, key: Array) -> Params:
    """Initialize all recurrent world-model parameters."""

    (
        encoder_key,
        decoder_key,
        recurrence_key,
        posterior_key,
        reward_key,
        continuation_key,
        prior_key,
        _,
    ) = jax.random.split(key, 8)
    reward = init_reward_head(config, reward_key)

    params: Params = {
        "encoder": init_mlp(
            encoder_key, config.observation_dim, config.hidden_dim, config.embedding_dim
        ),
        "decoder": init_mlp(
            decoder_key, config.feature_dim, config.hidden_dim, config.observation_dim
        ),
        # The two matched one-step objectives share an exactly shaped and
        # exactly initialized recurrence.  Both reserve two signal-coordinate
        # slots; trajectory iMF canonically pads its unused step-size slot with
        # zero.  Gaussian mode retains the coordinate-free recurrence.
        "recurrence": init_gru(
            recurrence_key,
            config.stochastic_dim
            + config.action_dim
            + (2 if config.prior == "shortcut" or config.imf_trajectory_enabled else 0),
            config.deterministic_dim,
        ),
        "posterior": init_mlp(
            posterior_key,
            config.deterministic_dim + config.embedding_dim,
            config.hidden_dim,
            2 * config.stochastic_dim,
        ),
        "reward": reward,
        "continuation": init_mlp(
            continuation_key, config.feature_dim, config.hidden_dim, 1
        ),
    }
    if config.prior == "gaussian":
        params["prior"] = init_mlp(
            prior_key,
            config.deterministic_dim,
            config.hidden_dim,
            2 * config.stochastic_dim,
        )
    elif config.prior == "shortcut":
        params["prior"] = init_mlp(
            prior_key,
            config.stochastic_dim + config.deterministic_dim + 2,
            config.hidden_dim,
            config.stochastic_dim,
            depth=2,
        )
    else:
        params["prior"] = init_imf(
            prior_key,
            config.stochastic_dim,
            config.deterministic_dim,
            config.hidden_dim,
            depth=2,
        )
    return params


def initial_state(config: DreamerConfig, batch_size: int, *, dtype=jnp.float32) -> RSSMState:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return RSSMState(
        jnp.zeros((batch_size, config.deterministic_dim), dtype=dtype),
        jnp.zeros((batch_size, config.stochastic_dim), dtype=dtype),
    )


def preprocess_observation(observation: Array) -> Array:
    value = jnp.asarray(observation)
    if value.dtype == jnp.uint8:
        value = value.astype(jnp.float32) / 255.0
    elif not jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.float32)
    return value


def encode(params: Params, observation: Array, config: DreamerConfig) -> Array:
    observation = preprocess_observation(observation)
    if observation.shape[-len(config.observation_shape) :] != config.observation_shape:
        raise ValueError("observation has the wrong trailing shape")
    flat = observation.reshape((*observation.shape[: -len(config.observation_shape)], -1))
    return mlp(params["encoder"], flat)


def decode(params: Params, feature: Array, config: DreamerConfig) -> Array:
    flat = mlp(params["decoder"], feature)
    return flat.reshape((*feature.shape[:-1], *config.observation_shape))


def _normal_stats(raw: Array, config: DreamerConfig) -> DiagonalNormal:
    mean, raw_std = jnp.split(raw, 2, axis=-1)
    std = config.min_std + (config.max_std - config.min_std) * jax.nn.sigmoid(raw_std)
    return DiagonalNormal(mean, std)


def posterior(
    params: Params,
    deterministic: Array,
    embedding: Array,
    config: DreamerConfig,
) -> DiagonalNormal:
    raw = mlp(params["posterior"], jnp.concatenate((deterministic, embedding), axis=-1))
    return _normal_stats(raw, config)


def prior_distribution(
    params: Params,
    deterministic: Array,
    config: DreamerConfig,
) -> DiagonalNormal:
    if config.prior != "gaussian":
        raise ValueError("prior_distribution is defined only for the Gaussian prior")
    return _normal_stats(mlp(params["prior"], deterministic), config)


def transition_deterministic(
    params: Params,
    state: RSSMState,
    action: Array,
    config: DreamerConfig,
    *,
    stochastic_time: Array | None = None,
    stochastic_step_size: Array | None = None,
) -> Array:
    if action.shape != (*state.stochastic.shape[:-1], config.action_dim):
        raise ValueError("action has the wrong shape")
    inputs = (state.stochastic, action)
    if config.imf_trajectory_enabled:
        if stochastic_step_size is not None:
            raise ValueError(
                "stochastic_step_size is available only in shortcut mode"
            )
        if stochastic_time is None:
            stochastic_time = jnp.zeros(
                (*state.stochastic.shape[:-1], 1), dtype=state.stochastic.dtype
            )
        else:
            stochastic_time = jnp.asarray(stochastic_time, dtype=state.stochastic.dtype)
            if stochastic_time.shape == state.stochastic.shape[:-1]:
                stochastic_time = stochastic_time[..., None]
            if stochastic_time.shape != (*state.stochastic.shape[:-1], 1):
                raise ValueError("stochastic_time must have one scalar per state")
        inputs = (*inputs, stochastic_time, jnp.zeros_like(stochastic_time))
    elif config.prior == "shortcut":
        if (stochastic_time is None) != (stochastic_step_size is None):
            raise ValueError(
                "shortcut stochastic_time and stochastic_step_size must be supplied together"
            )
        if stochastic_time is None:
            # States installed by posterior observation or completed sampling
            # are clean endpoint tokens, with no remaining integration step.
            stochastic_time = jnp.ones(
                (*state.stochastic.shape[:-1], 1), dtype=state.stochastic.dtype
            )
            stochastic_step_size = jnp.zeros_like(stochastic_time)
        else:
            stochastic_time = jnp.asarray(stochastic_time, dtype=state.stochastic.dtype)
            stochastic_step_size = jnp.asarray(
                stochastic_step_size, dtype=state.stochastic.dtype
            )
            if stochastic_time.shape == state.stochastic.shape[:-1]:
                stochastic_time = stochastic_time[..., None]
            if stochastic_step_size.shape == state.stochastic.shape[:-1]:
                stochastic_step_size = stochastic_step_size[..., None]
            expected_shape = (*state.stochastic.shape[:-1], 1)
            if stochastic_time.shape != expected_shape:
                raise ValueError("stochastic_time must have one scalar per state")
            if stochastic_step_size.shape != expected_shape:
                raise ValueError(
                    "stochastic_step_size must have one scalar per state"
                )
        inputs = (*inputs, stochastic_time, stochastic_step_size)
    elif stochastic_time is not None or stochastic_step_size is not None:
        raise ValueError(
            "stochastic signal coordinates require trajectory iMF or shortcut mode"
        )
    inputs = jnp.concatenate(inputs, axis=-1)
    return gru(params["recurrence"], inputs, state.deterministic)


def trajectory_deterministic_conditions(
    params: Params,
    corrupted_history: Array,
    history_times: Array,
    actions: Array,
    is_first: Array,
    config: DreamerConfig,
) -> Array:
    """Build reset-safe causal prior conditions from corrupted posterior history.

    The condition emitted for token ``k`` is computed from only the previous
    recurrent state, the action aligned with token ``k``, and the corrupted
    latent/time pair at ``k-1``.  The current corrupted latent is installed in
    the carry only *after* emitting that condition.  At an episode boundary,
    both the recurrent carry and boundary action are erased before use.

    This helper is pure and scan-based so it can be used inside a jitted loss.
    It is intentionally specific to trajectory mode: the extra scalar time is
    part of the recurrent sufficient state and cannot be silently discarded.
    """

    if not config.imf_trajectory_enabled or config.prior != "imf":
        raise ValueError("trajectory conditions require trajectory iMF mode")
    if corrupted_history.ndim != 3:
        raise ValueError(
            "corrupted_history must have shape [batch, time, stochastic_dim]"
        )
    batch, time, stochastic_dim = corrupted_history.shape
    if stochastic_dim != config.stochastic_dim:
        raise ValueError("corrupted_history has the wrong stochastic dimension")
    if actions.shape != (batch, time, config.action_dim):
        raise ValueError("actions must have shape [batch, time, action_dim]")
    history_times = jnp.asarray(history_times, dtype=corrupted_history.dtype)
    if history_times.shape == (batch, time):
        history_times = history_times[..., None]
    if history_times.shape != (batch, time, 1):
        raise ValueError("history_times must have one scalar per trajectory token")
    if is_first.shape != (batch, time):
        raise ValueError("is_first must have shape [batch, time]")
    is_first = jnp.asarray(is_first, dtype=jnp.bool_)

    state = initial_state(config, batch, dtype=corrupted_history.dtype)
    initial_time = jnp.zeros((batch, 1), dtype=corrupted_history.dtype)

    def scan_step(
        carry: tuple[RSSMState, Array],
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[RSSMState, Array], Array]:
        previous_state, previous_time = carry
        current_history, current_time, action, first = inputs
        reset = first[:, None]
        previous_state = RSSMState(
            jnp.where(
                reset,
                jnp.zeros_like(previous_state.deterministic),
                previous_state.deterministic,
            ),
            jnp.where(
                reset,
                jnp.zeros_like(previous_state.stochastic),
                previous_state.stochastic,
            ),
        )
        previous_time = jnp.where(reset, jnp.zeros_like(previous_time), previous_time)
        action = jnp.where(reset, jnp.zeros_like(action), action)
        deterministic = transition_deterministic(
            params,
            previous_state,
            action,
            config,
            stochastic_time=previous_time,
        )
        next_state = RSSMState(deterministic, current_history)
        return (next_state, current_time), deterministic

    _, deterministic = jax.lax.scan(
        scan_step,
        (state, initial_time),
        (
            jnp.swapaxes(corrupted_history, 0, 1),
            jnp.swapaxes(history_times, 0, 1),
            jnp.swapaxes(actions, 0, 1),
            jnp.swapaxes(is_first, 0, 1),
        ),
    )
    return jnp.swapaxes(deterministic, 0, 1)


def shortcut_deterministic_conditions(
    params: Params,
    latent_sequence: Array,
    signal_times: Array,
    step_sizes: Array,
    actions: Array,
    is_first: Array,
    config: DreamerConfig,
) -> Array:
    """Rebuild reset-safe causal conditions for one shortcut predictor call.

    The condition for token ``k`` sees only the previous token's latent,
    signal time, and step size.  This function intentionally accepts the full
    mutable ``(z, tau, d)`` sequence: Equation 7 invokes it independently for
    the primal prediction, first half step, and the ``z_prime``/midpoint second
    half step.  Reusing a condition computed from an earlier call would be a
    stale-context implementation of the objective.
    """

    if config.prior != "shortcut":
        raise ValueError("shortcut conditions require prior='shortcut'")
    if latent_sequence.ndim != 3:
        raise ValueError("latent_sequence must have shape [batch, time, stochastic_dim]")
    batch, time, stochastic_dim = latent_sequence.shape
    if stochastic_dim != config.stochastic_dim:
        raise ValueError("latent_sequence has the wrong stochastic dimension")
    if actions.shape != (batch, time, config.action_dim):
        raise ValueError("actions must have shape [batch, time, action_dim]")

    def coordinates(value: Array, name: str) -> Array:
        value = jnp.asarray(value, dtype=latent_sequence.dtype)
        if value.shape == (batch, time):
            value = value[..., None]
        if value.shape != (batch, time, 1):
            raise ValueError(f"{name} must have one scalar per sequence token")
        return value

    signal_times = coordinates(signal_times, "signal_times")
    step_sizes = coordinates(step_sizes, "step_sizes")
    if is_first.shape != (batch, time):
        raise ValueError("is_first must have shape [batch, time]")
    is_first = jnp.asarray(is_first, dtype=jnp.bool_)

    state = initial_state(config, batch, dtype=latent_sequence.dtype)
    # A clean zero latent is the canonical empty-prefix state in shortcut
    # mode.  The step coordinate is zero because no denoising step exists.
    initial_time = jnp.ones((batch, 1), dtype=latent_sequence.dtype)
    initial_step = jnp.zeros((batch, 1), dtype=latent_sequence.dtype)

    def scan_step(
        carry: tuple[RSSMState, Array, Array],
        inputs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[tuple[RSSMState, Array, Array], Array]:
        previous_state, previous_time, previous_step = carry
        current_latent, current_time, current_step, action, first = inputs
        reset = first[:, None]
        previous_state = RSSMState(
            jnp.where(
                reset,
                jnp.zeros_like(previous_state.deterministic),
                previous_state.deterministic,
            ),
            jnp.where(
                reset,
                jnp.zeros_like(previous_state.stochastic),
                previous_state.stochastic,
            ),
        )
        previous_time = jnp.where(
            reset, jnp.ones_like(previous_time), previous_time
        )
        previous_step = jnp.where(
            reset, jnp.zeros_like(previous_step), previous_step
        )
        action = jnp.where(reset, jnp.zeros_like(action), action)
        deterministic = transition_deterministic(
            params,
            previous_state,
            action,
            config,
            stochastic_time=previous_time,
            stochastic_step_size=previous_step,
        )
        next_state = RSSMState(deterministic, current_latent)
        return (next_state, current_time, current_step), deterministic

    _, deterministic = jax.lax.scan(
        scan_step,
        (state, initial_time, initial_step),
        (
            jnp.swapaxes(latent_sequence, 0, 1),
            jnp.swapaxes(signal_times, 0, 1),
            jnp.swapaxes(step_sizes, 0, 1),
            jnp.swapaxes(actions, 0, 1),
            jnp.swapaxes(is_first, 0, 1),
        ),
    )
    return jnp.swapaxes(deterministic, 0, 1)


def shortcut_predict_clean_sequence(
    params: Params,
    latent_sequence: Array,
    signal_times: Array,
    step_sizes: Array,
    actions: Array,
    is_first: Array,
    config: DreamerConfig,
) -> Array:
    """Predict clean tokens after freshly rebuilding their causal contexts."""

    signal_times = jnp.asarray(signal_times, dtype=latent_sequence.dtype)
    step_sizes = jnp.asarray(step_sizes, dtype=latent_sequence.dtype)
    if signal_times.shape == latent_sequence.shape[:2]:
        signal_times = signal_times[..., None]
    if step_sizes.shape == latent_sequence.shape[:2]:
        step_sizes = step_sizes[..., None]
    coordinate_shape = (*latent_sequence.shape[:2], 1)
    if signal_times.shape != coordinate_shape or step_sizes.shape != coordinate_shape:
        raise ValueError("signal_times and step_sizes must have one scalar per token")
    conditions = shortcut_deterministic_conditions(
        params,
        latent_sequence,
        signal_times,
        step_sizes,
        actions,
        is_first,
        config,
    )
    batch, time, stochastic_dim = latent_sequence.shape
    inputs = jnp.concatenate(
        (latent_sequence, conditions, signal_times, step_sizes), axis=-1
    ).reshape((batch * time, -1))
    prediction = mlp(params["prior"], inputs, activation=jax.nn.silu)
    return prediction.reshape((batch, time, stochastic_dim))


def prior_sampling_nfe(config: DreamerConfig, *, transitions: int = 1) -> int:
    """Return the exact number of learned prior evaluations for generation."""

    if not isinstance(transitions, int) or isinstance(transitions, bool) or transitions <= 0:
        raise ValueError("transitions must be a positive integer")
    if config.prior == "gaussian":
        per_transition = 1
    elif config.prior == "shortcut":
        per_transition = config.shortcut_sampling_steps
    else:
        per_transition = config.imf_sampling_steps
    return transitions * per_transition


def world_model_parameter_counts(
    params: Params, config: DreamerConfig
) -> ParameterCounts:
    """Report structurally active and total parameters for a configured arm.

    The matched trajectory and shortcut arms share a two-coordinate recurrent
    interface.  Trajectory iMF canonically pads the second coordinate with
    zero, so the corresponding GRU input-weight row is structurally inactive
    and is excluded from the active count.  Other counts are structural array
    sizes; they do not infer temporary zero gradients from a minibatch.
    """

    def count(tree: Params) -> int:
        return sum(int(value.size) for value in jax.tree_util.tree_leaves(tree))

    total = count(params)
    prior_total = count(params["prior"])
    inactive = 3 * config.deterministic_dim if config.imf_trajectory_enabled else 0
    return ParameterCounts(
        active=total - inactive,
        total=total,
        prior_active=prior_total,
        prior_total=prior_total,
    )


def sample_prior_with_nfe(
    params: Params,
    deterministic: Array,
    key: Array | None,
    config: DreamerConfig,
    *,
    noise: Array | None = None,
) -> PriorSample:
    if noise is None and key is None:
        raise ValueError("key is required when noise is omitted")
    if config.prior == "gaussian":
        distribution = prior_distribution(params, deterministic, config)
        if noise is None:
            noise = jax.random.normal(key, distribution.mean.shape, dtype=distribution.mean.dtype)
        return PriorSample(
            distribution.mean + distribution.std * noise,
            distribution,
            jnp.asarray(1, dtype=jnp.int32),
        )
    if config.prior == "shortcut":
        if noise is None:
            noise = jax.random.normal(
                key,
                (deterministic.shape[0], config.stochastic_dim),
                dtype=deterministic.dtype,
            )
        elif noise.shape != (deterministic.shape[0], config.stochastic_dim):
            raise ValueError("noise has the wrong shape")

        sequence_noise = noise[:, None, :]
        sequence_condition = deterministic[:, None, :]

        def predict_clean(state: Array, tau: Array, step_size: Array) -> Array:
            inputs = jnp.concatenate(
                (state, sequence_condition, tau, step_size), axis=-1
            )
            batch, time = state.shape[:2]
            output = mlp(
                params["prior"],
                inputs.reshape((batch * time, -1)),
                activation=jax.nn.silu,
            )
            return output.reshape(state.shape)

        stochastic = sample_shortcut_steps(
            predict_clean,
            sequence_noise,
            steps=config.shortcut_sampling_steps,
            clean_prediction_clip=config.shortcut_sampling_clip,
        )[:, 0]
        return PriorSample(
            stochastic,
            None,
            jnp.asarray(config.shortcut_sampling_steps, dtype=jnp.int32),
        )
    stochastic = sample_imf_steps(
        params["prior"],
        deterministic,
        key,
        noise=noise,
        steps=config.imf_sampling_steps,
    )
    return PriorSample(
        stochastic,
        None,
        jnp.asarray(config.imf_sampling_steps, dtype=jnp.int32),
    )


def sample_prior(
    params: Params,
    deterministic: Array,
    key: Array | None,
    config: DreamerConfig,
    *,
    noise: Array | None = None,
) -> tuple[Array, DiagonalNormal | None]:
    """Compatibility wrapper returning the sample and optional distribution."""

    result = sample_prior_with_nfe(
        params, deterministic, key, config, noise=noise
    )
    return result.stochastic, result.distribution


def observe_step(
    params: Params,
    observation: Array,
    previous_action: Array,
    state: RSSMState,
    key: Array,
    config: DreamerConfig,
) -> tuple[RSSMState, DiagonalNormal, DiagonalNormal | None]:
    deterministic = transition_deterministic(params, state, previous_action, config)
    distribution = posterior(params, deterministic, encode(params, observation, config), config)
    stochastic = distribution.mean + distribution.std * jax.random.normal(
        key, distribution.mean.shape, dtype=distribution.mean.dtype
    )
    prior = (
        prior_distribution(params, deterministic, config)
        if config.prior == "gaussian"
        else None
    )
    return RSSMState(deterministic, stochastic), distribution, prior


def observe_sequence(
    params: Params,
    observations: Array,
    actions: Array,
    key: Array,
    config: DreamerConfig,
    *,
    start: RSSMState | None = None,
    is_first: Array | None = None,
) -> SequenceStates:
    expected_ndim = 2 + len(config.observation_shape)
    if observations.ndim != expected_ndim or observations.shape[2:] != config.observation_shape:
        raise ValueError("observations must have shape [batch, time, *observation_shape]")
    if actions.shape != (*observations.shape[:2], config.action_dim):
        raise ValueError("actions must have shape [batch, time, action_dim]")
    batch, time = actions.shape[:2]
    if is_first is None:
        is_first = jnp.zeros((batch, time), dtype=jnp.bool_)
    elif is_first.shape != (batch, time):
        raise ValueError("is_first must have shape [batch, time]")
    else:
        is_first = jnp.asarray(is_first, dtype=jnp.bool_)
    state = initial_state(config, batch) if start is None else start
    keys = jax.random.split(key, time)

    def scan_step(carry: RSSMState, inputs: tuple[Array, Array, Array, Array]):
        observation, action, first, step_key = inputs
        reset = first[:, None]
        carry = RSSMState(
            jnp.where(reset, jnp.zeros_like(carry.deterministic), carry.deterministic),
            jnp.where(reset, jnp.zeros_like(carry.stochastic), carry.stochastic),
        )
        # An action stored next to an episode's reset observation cannot have
        # arisen from the new state and must not leak the previous episode.
        action = jnp.where(reset, jnp.zeros_like(action), action)
        next_state, post, prior = observe_step(
            params, observation, action, carry, step_key, config
        )
        if prior is None:
            prior_mean = jnp.zeros_like(post.mean)
            prior_std = jnp.zeros_like(post.std)
        else:
            prior_mean, prior_std = prior
        outputs = (
            next_state.deterministic,
            next_state.stochastic,
            post.mean,
            post.std,
            prior_mean,
            prior_std,
        )
        return next_state, outputs

    _, outputs = jax.lax.scan(
        scan_step,
        state,
        (
            jnp.swapaxes(observations, 0, 1),
            jnp.swapaxes(actions, 0, 1),
            jnp.swapaxes(is_first, 0, 1),
            keys,
        ),
    )
    deterministic, stochastic, post_mean, post_std, prior_mean, prior_std = (
        jnp.swapaxes(value, 0, 1) for value in outputs
    )
    return SequenceStates(
        RSSMState(deterministic, stochastic),
        DiagonalNormal(post_mean, post_std),
        prior_mean if config.prior == "gaussian" else None,
        prior_std if config.prior == "gaussian" else None,
    )


def predict_reward_logits(params: Params, feature: Array) -> Array:
    """Return the raw scalar reward-head output before support transforms."""

    return mlp(params["reward"], feature)[..., 0]


def reward_support(config: DreamerConfig, *, dtype: jnp.dtype = jnp.float32) -> Array:
    """Return the symlog support for a categorical reward head."""

    if config.reward_loss != "symexp_twohot" or config.reward_bins <= 1:
        raise ValueError("reward support requires a symexp_twohot reward head")
    return jnp.linspace(
        config.reward_symlog_min,
        config.reward_symlog_max,
        config.reward_bins,
        dtype=dtype,
    )


def two_hot_reward(target: Array, config: DreamerConfig) -> Array:
    """Encode rewards by linear interpolation on the configured symlog support."""

    support = reward_support(config, dtype=target.dtype)
    transformed = jnp.clip(
        jnp.sign(target) * jnp.log1p(jnp.abs(target)),
        config.reward_symlog_min,
        config.reward_symlog_max,
    )
    width = support[1] - support[0]
    position = (transformed - support[0]) / width
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, config.reward_bins - 1)
    upper_weight = position - lower.astype(position.dtype)
    lower_weight = 1.0 - upper_weight
    encoded = (
        jax.nn.one_hot(lower, config.reward_bins, dtype=target.dtype)
        * lower_weight[..., None]
    )
    encoded += (
        jax.nn.one_hot(upper, config.reward_bins, dtype=target.dtype)
        * upper_weight[..., None]
    )
    return encoded


def predict_reward_offset_logits(
    params: Params, feature: Array, config: DreamerConfig
) -> Array:
    """Return logits as ``[..., offset, support]`` for offsets zero through L."""

    raw = mlp(params["reward"], feature)
    expected = config.reward_head_output_dim
    if raw.shape[-1] != expected:
        raise ValueError(
            f"reward output dimension {raw.shape[-1]} does not match configured {expected}"
        )
    return raw.reshape(
        (*raw.shape[:-1], config.reward_prediction_horizon + 1, config.reward_bins)
    )


def predict_reward_offsets(
    params: Params, feature: Array, config: DreamerConfig
) -> Array:
    """Decode all reward offsets to scalar expectations."""

    logits = predict_reward_offset_logits(params, feature, config)
    if config.reward_loss == "symexp_twohot":
        mean_symlog = jnp.sum(
            jax.nn.softmax(logits, axis=-1)
            * reward_support(config, dtype=logits.dtype),
            axis=-1,
        )
        return jnp.sign(mean_symlog) * jnp.expm1(jnp.abs(mean_symlog))
    raw = logits[..., 0]
    if config.reward_min is None:
        return raw
    return config.reward_min + (
        config.reward_max - config.reward_min
    ) * jax.nn.sigmoid(raw)


def predict_reward(
    params: Params, feature: Array, config: DreamerConfig | None = None
) -> Array:
    """Predict reward, optionally constrained to the configured finite support."""

    if config is None:
        return predict_reward_logits(params, feature)
    return predict_reward_offsets(params, feature, config)[..., 0]


def predict_continuation_logits(params: Params, feature: Array) -> Array:
    return mlp(params["continuation"], feature)[..., 0]


def gaussian_kl(q: DiagonalNormal, p: DiagonalNormal, *, stop_q: bool = False) -> Array:
    q_mean = jax.lax.stop_gradient(q.mean) if stop_q else q.mean
    q_std = jax.lax.stop_gradient(q.std) if stop_q else q.std
    return jnp.sum(
        jnp.log(p.std / q_std)
        + (jnp.square(q_std) + jnp.square(q_mean - p.mean)) / (2.0 * jnp.square(p.std))
        - 0.5,
        axis=-1,
    )


def representation_kl(distribution: DiagonalNormal) -> Array:
    variance = jnp.square(distribution.std)
    return 0.5 * jnp.sum(
        jnp.square(distribution.mean) + variance - jnp.log(variance) - 1.0,
        axis=-1,
    )


def _binary_cross_entropy_with_logits(logits: Array, labels: Array) -> Array:
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _masked_mean(value: Array, mask: Array) -> Array:
    """Average per-step values over an explicit ``[batch, time]`` mask."""

    if value.shape[:2] != mask.shape:
        raise ValueError("masked values must begin with the mask shape")
    if value.ndim > 2:
        value = jnp.mean(value, axis=tuple(range(2, value.ndim)))
    mask = jnp.asarray(mask, dtype=value.dtype)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, value.dtype))


def reward_mtp_targets(
    rewards: Array,
    continuations: Array,
    is_first: Array,
    loss_mask: Array,
    horizon: int,
) -> tuple[Array, Array]:
    """Build future reward targets without crossing padding or episode boundaries.

    Offset zero is the reward aligned with the current latent.  Offset ``n``
    is valid only when the target is in bounds, every intervening transition
    continues, and no intervening token starts a new episode.
    """

    rewards = jnp.asarray(rewards)
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, time]")
    if continuations.shape != rewards.shape or is_first.shape != rewards.shape:
        raise ValueError("continuations and is_first must match rewards")
    if loss_mask.shape != rewards.shape:
        raise ValueError("loss_mask must match rewards")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    batch, time = rewards.shape
    targets: list[Array] = []
    masks: list[Array] = []
    anchor_mask = jnp.asarray(loss_mask, dtype=rewards.dtype)
    continuation = jnp.asarray(continuations, dtype=rewards.dtype)
    first = jnp.asarray(is_first, dtype=jnp.bool_)
    for offset in range(horizon + 1):
        if offset >= time:
            target = jnp.zeros_like(rewards)
            valid = jnp.zeros_like(rewards)
        else:
            target = jnp.pad(
                rewards[:, offset:], ((0, 0), (0, offset))
            )
            valid_prefix = jnp.ones((batch, time - offset), dtype=rewards.dtype)
            for step in range(offset):
                valid_prefix = valid_prefix * continuation[
                    :, step : step + time - offset
                ]
                valid_prefix = valid_prefix * (~first[
                    :, step + 1 : step + 1 + time - offset
                ]).astype(rewards.dtype)
            valid = jnp.pad(valid_prefix, ((0, 0), (0, offset)))
        targets.append(target)
        masks.append(anchor_mask * valid)
    return (
        jax.lax.stop_gradient(jnp.stack(targets, axis=-1)),
        jax.lax.stop_gradient(jnp.stack(masks, axis=-1)),
    )


def reward_prediction_loss(
    params: Params,
    feature: Array,
    rewards: Array,
    continuations: Array,
    is_first: Array,
    loss_mask: Array,
    config: DreamerConfig,
) -> Array:
    """Terminal-safe scalar or categorical multi-token reward loss."""

    if feature.shape[:2] != rewards.shape:
        raise ValueError("reward features must begin with [batch, time]")
    targets, masks = reward_mtp_targets(
        rewards,
        continuations,
        is_first,
        loss_mask,
        config.reward_prediction_horizon,
    )
    logits = predict_reward_offset_logits(params, feature, config)
    if config.reward_loss == "symexp_twohot":
        labels = two_hot_reward(targets, config)
        per_target = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    elif config.reward_loss == "binary_cross_entropy":
        normalized = (targets - config.reward_min) / (config.reward_max - config.reward_min)
        per_target = _binary_cross_entropy_with_logits(logits[..., 0], normalized)
    else:
        prediction = predict_reward_offsets(params, feature, config)
        per_target = jnp.square(prediction - targets)
    denominator = jnp.maximum(jnp.sum(masks), jnp.asarray(1.0, masks.dtype))
    return jnp.sum(per_target * masks) / denominator


def _pseudo_huber(value: Array, delta: float) -> Array:
    """Quadratic near zero and linear in the tails without a branch."""

    delta_value = jnp.asarray(delta, dtype=value.dtype)
    scaled = value / delta_value
    return jnp.square(delta_value) * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)


def trajectory_causal_consistency_loss(
    params: Params,
    sequence: SequenceStates,
    batch: Batch,
    loss_mask: Array,
    key: Array,
    config: DreamerConfig,
) -> tuple[Array, Array, Array]:
    """Match paired simulator action effects with coupled iMF samples.

    Both branches start from the same posterior state and deliberately reuse
    the same immutable JAX key. Stochastic sampling therefore cancels instead
    of masquerading as an action effect. Stopped per-feature RMS scales make
    the robust loss sensitive to effect direction without discarding magnitude.
    """

    required = (
        "causal_actions_lower",
        "causal_actions_upper",
        "causal_observations_lower",
        "causal_observations_upper",
        "causal_rewards_lower",
        "causal_rewards_upper",
        "causal_mask",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"causal consistency batch is missing {missing}")
    batch_size, steps = loss_mask.shape
    action_shape = (batch_size, steps, config.action_dim)
    observation_shape = (batch_size, steps, *config.observation_shape)
    for name in ("causal_actions_lower", "causal_actions_upper"):
        if batch[name].shape != action_shape:
            raise ValueError(f"{name} must have shape {action_shape}")
    for name in ("causal_observations_lower", "causal_observations_upper"):
        if batch[name].shape != observation_shape:
            raise ValueError(f"{name} must have shape {observation_shape}")
    for name in ("causal_rewards_lower", "causal_rewards_upper", "causal_mask"):
        if batch[name].shape != (batch_size, steps):
            raise ValueError(f"{name} must have shape {(batch_size, steps)}")

    previous = RSSMState(
        jnp.concatenate(
            (
                jnp.zeros_like(sequence.states.deterministic[:, :1]),
                sequence.states.deterministic[:, :-1],
            ),
            axis=1,
        ),
        jnp.concatenate(
            (
                jnp.zeros_like(sequence.states.stochastic[:, :1]),
                sequence.states.stochastic[:, :-1],
            ),
            axis=1,
        ),
    )

    def flatten(value: Array) -> Array:
        return value.reshape((batch_size * steps, value.shape[-1]))

    previous_flat = RSSMState(
        flatten(previous.deterministic), flatten(previous.stochastic)
    )

    def paired_prediction(action: Array) -> tuple[Array, Array]:
        deterministic = transition_deterministic(
            params, previous_flat, flatten(action), config
        )
        # Calling the pure sampler with the same key is exact common randomness.
        stochastic, _ = sample_prior(params, deterministic, key, config)
        feature = jnp.concatenate((deterministic, stochastic), axis=-1)
        observation = decode(params, feature, config).reshape(observation_shape)
        reward = predict_reward(params, feature, config).reshape((batch_size, steps))
        return observation, reward

    lower_observation, lower_reward = paired_prediction(batch["causal_actions_lower"])
    upper_observation, upper_reward = paired_prediction(batch["causal_actions_upper"])
    predicted_observation_delta = upper_observation - lower_observation
    predicted_reward_delta = upper_reward - lower_reward
    target_observation_delta = jax.lax.stop_gradient(
        preprocess_observation(batch["causal_observations_upper"])
        - preprocess_observation(batch["causal_observations_lower"])
    )
    target_reward_delta = jax.lax.stop_gradient(
        batch["causal_rewards_upper"] - batch["causal_rewards_lower"]
    )

    causal_mask = loss_mask * jnp.asarray(batch["causal_mask"], dtype=loss_mask.dtype)
    masked_count = jnp.maximum(
        jnp.sum(causal_mask), jnp.asarray(1.0, dtype=loss_mask.dtype)
    )
    observation_mask = causal_mask.reshape(
        (*causal_mask.shape,) + (1,) * len(config.observation_shape)
    )
    observation_scale = jnp.sqrt(
        jnp.sum(jnp.square(target_observation_delta) * observation_mask, axis=(0, 1))
        / masked_count
        + config.imf_causal_normalization_epsilon**2
    )
    reward_scale = jnp.sqrt(
        jnp.sum(jnp.square(target_reward_delta) * causal_mask) / masked_count
        + config.imf_causal_normalization_epsilon**2
    )
    observation_residual = (
        predicted_observation_delta - target_observation_delta
    ) / jax.lax.stop_gradient(observation_scale)
    reward_residual = (
        predicted_reward_delta - target_reward_delta
    ) / jax.lax.stop_gradient(reward_scale)
    observation_per_transition = jnp.mean(
        _pseudo_huber(
            observation_residual, config.imf_causal_huber_delta
        ).reshape((batch_size, steps, -1)),
        axis=-1,
    )
    observation_loss = _masked_mean(observation_per_transition, causal_mask)
    reward_loss = _masked_mean(
        _pseudo_huber(reward_residual, config.imf_causal_huber_delta),
        causal_mask,
    )
    total = observation_loss + config.imf_causal_reward_scale * reward_loss
    return total, observation_loss, reward_loss


def _distance_consistency_loss(
    params: Params,
    sequence: SequenceStates,
    actions: Array,
    is_first: Array,
    loss_mask: Array,
    key: Array,
    config: DreamerConfig,
    distance: int,
) -> Array:
    """Vectorized posterior consistency at one exact temporal distance."""

    batch, time = actions.shape[:2]
    starts = time - distance
    if distance <= 0 or starts <= 0:
        return jnp.asarray(0.0, dtype=actions.dtype)

    def flatten(value: Array) -> Array:
        return value.reshape((batch * starts, value.shape[-1]))

    imagined = RSSMState(
        flatten(sequence.states.deterministic[:, :starts]),
        flatten(sequence.states.stochastic[:, :starts]),
    )
    final_prior: DiagonalNormal | None = None
    rollout_keys = jax.random.split(jax.random.fold_in(key, distance), distance + 2)
    for offset in range(1, distance + 1):
        action = flatten(actions[:, offset : offset + starts])
        deterministic = transition_deterministic(params, imagined, action, config)
        stochastic, final_prior = sample_prior(
            params, deterministic, rollout_keys[offset - 1], config
        )
        imagined = RSSMState(deterministic, stochastic)

    # A prediction must never jump across an episode reset. Burn-in start states
    # are useful context, but only loss-bearing target steps contribute.
    valid = jnp.asarray(loss_mask[:, distance:], dtype=actions.dtype)
    for offset in range(1, distance + 1):
        valid = valid * (1.0 - is_first[:, offset : offset + starts].astype(actions.dtype))
    flat_valid = valid.reshape((-1,))
    target = DiagonalNormal(
        flatten(sequence.posterior.mean[:, distance:]),
        flatten(sequence.posterior.std[:, distance:]),
    )
    if config.prior == "gaussian":
        if final_prior is None:
            raise AssertionError("Gaussian rollout did not produce a prior distribution")
        per_pair = jnp.maximum(
            gaussian_kl(target, final_prior, stop_q=True), config.kl_free_nats
        )
        return jnp.sum(per_pair * flat_valid) / jnp.maximum(
            jnp.sum(flat_valid), jnp.asarray(1.0, actions.dtype)
        )
    if config.prior == "shortcut":
        # Shortcut forcing is trained as a joint causal sequence objective in
        # world_model_loss.  Adding this legacy pairwise overshooting loss
        # would violate the matched single-objective arm.
        return jnp.asarray(0.0, dtype=actions.dtype)

    target_noise_key, loss_key = rollout_keys[-2:]
    posterior_epsilon = jax.random.normal(
        target_noise_key, target.mean.shape, dtype=target.mean.dtype
    )
    target_sample = target.mean + target.std * posterior_epsilon
    return improved_meanflow_loss(
        params["prior"],
        jax.lax.stop_gradient(target_sample),
        imagined.deterministic,
        loss_key,
        noise=transport_base_noise(
            posterior_epsilon, config.imf_noise_coupling
        ),
        boundary_fraction=config.imf_boundary_fraction,
        time_mean=config.imf_time_mean,
        time_std=config.imf_time_std,
        adaptive_power=config.imf_adaptive_power,
        adaptive_epsilon=config.imf_adaptive_epsilon,
        meanflow_scale=config.imf_meanflow_scale,
        velocity_scale=config.imf_velocity_scale,
        endpoint_scale=config.imf_endpoint_scale,
        shortcut_scale=config.imf_shortcut_scale,
        signal_weight_floor=config.imf_signal_weight_floor,
        signal_weight_scale=config.imf_signal_weight_scale,
        condition_gradient_scale=config.imf_condition_gradient_scale,
        boundary_velocity_supervision=config.imf_boundary_velocity_supervision,
        weights=flat_valid,
    )


def multistep_consistency_losses(
    params: Params,
    sequence: SequenceStates,
    actions: Array,
    is_first: Array,
    loss_mask: Array,
    key: Array,
    config: DreamerConfig,
) -> tuple[Array, Array, Array]:
    """Return aggregate and exact distance-5/distance-15 objectives.

    All temporal starts for a distance are flattened into one batch, so the
    cost is linear in each requested rollout distance rather than nesting a
    Python loop over every start and every intermediate distance.
    """

    distances = tuple(config.overshooting_distances)
    if (
        not any(distance > 1 for distance in distances)
        and config.overshooting_horizon > 1
    ):
        distances = tuple(range(2, config.overshooting_horizon + 1))
    terms: list[Array] = []
    distance_5 = jnp.asarray(0.0, dtype=actions.dtype)
    distance_15 = jnp.asarray(0.0, dtype=actions.dtype)
    for distance in distances:
        # Distance one is already the primary prior objective. Historically an
        # overshooting horizon of one disabled this auxiliary loss, so retain
        # that behavior while repaired protocols request exact 5/15 distances.
        if distance <= 1 or distance >= actions.shape[1]:
            continue
        term = _distance_consistency_loss(
            params,
            sequence,
            actions,
            is_first,
            loss_mask,
            key,
            config,
            distance,
        )
        terms.append(term)
        if distance == 5:
            distance_5 = term
        elif distance == 15:
            distance_15 = term
    aggregate = (
        jnp.mean(jnp.stack(terms))
        if terms
        else jnp.asarray(0.0, dtype=actions.dtype)
    )
    return aggregate, distance_5, distance_15


def overshooting_loss(
    params: Params,
    sequence: SequenceStates,
    actions: Array,
    key: Array,
    config: DreamerConfig,
    *,
    is_first: Array | None = None,
    loss_mask: Array | None = None,
) -> Array:
    shape = actions.shape[:2]
    if is_first is None:
        is_first = jnp.zeros(shape, dtype=jnp.bool_)
    if loss_mask is None:
        loss_mask = jnp.ones(shape, dtype=actions.dtype)
    return multistep_consistency_losses(
        params, sequence, actions, is_first, loss_mask, key, config
    )[0]


def world_model_loss(
    params: Params,
    batch: Batch,
    key: Array,
    config: DreamerConfig,
    *,
    shortcut_teacher_params: Params | None = None,
) -> WorldModelLoss:
    required = ("observations", "actions", "rewards", "continuations")
    if any(name not in batch for name in required):
        raise KeyError(f"batch must contain {required}")
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    continuations = batch["continuations"]
    if rewards.shape != actions.shape[:2] or continuations.shape != actions.shape[:2]:
        raise ValueError("rewards and continuations must have shape [batch, time]")
    is_first = batch.get("is_first", jnp.zeros(actions.shape[:2], dtype=jnp.bool_))
    if is_first.shape != actions.shape[:2]:
        raise ValueError("is_first must have shape [batch, time]")
    is_first = jnp.asarray(is_first, dtype=jnp.bool_)
    if "loss_mask" in batch:
        loss_mask = jnp.asarray(batch["loss_mask"], dtype=actions.dtype)
        if loss_mask.shape != actions.shape[:2]:
            raise ValueError("loss_mask must have shape [batch, time]")
    else:
        if config.burn_in >= actions.shape[1]:
            raise ValueError("burn_in must leave at least one loss-bearing step")
        loss_mask = jnp.ones(actions.shape[:2], dtype=actions.dtype)
        if config.burn_in:
            loss_mask = loss_mask.at[:, : config.burn_in].set(0.0)
    # Preserve the exact legacy RNG streams when the optional causal term is
    # disabled; deriving a fourth key via a larger split would perturb all
    # three existing streams even at a zero loss scale.
    sequence_key, prior_key, overshooting_key = jax.random.split(key, 3)
    causal_key = jax.random.fold_in(key, 0xCA05A1)
    sequence = observe_sequence(
        params,
        observations,
        actions,
        sequence_key,
        config,
        is_first=is_first,
    )
    feature = sequence.states.feature
    targets = preprocess_observation(observations)
    reconstruction = _masked_mean(
        jnp.square(decode(params, feature, config) - targets), loss_mask
    )
    reward = reward_prediction_loss(
        params,
        feature,
        rewards,
        continuations,
        is_first,
        loss_mask,
        config,
    )
    continuation = _masked_mean(
        _binary_cross_entropy_with_logits(
            predict_continuation_logits(params, feature), continuations
        ),
        loss_mask,
    )
    representation = _masked_mean(representation_kl(sequence.posterior), loss_mask)
    zero = jnp.asarray(0.0, dtype=actions.dtype)
    imf_loss_u = zero
    imf_loss_v = zero
    imf_endpoint = zero
    imf_shortcut = zero
    causal_consistency = zero
    causal_observation = zero
    causal_reward = zero
    if config.prior == "gaussian":
        prior_distribution_value = DiagonalNormal(sequence.prior_mean, sequence.prior_std)
        prior = _masked_mean(
            jnp.maximum(
                gaussian_kl(
                    sequence.posterior, prior_distribution_value, stop_q=True
                ),
                config.kl_free_nats,
            ),
            loss_mask,
        )
    elif config.prior == "shortcut":
        if config.shortcut_training_k_max is None:
            raise AssertionError("validated shortcut config is missing training K_max")
        target_stochastic = jax.lax.stop_gradient(sequence.states.stochastic)

        def predict_clean(
            latent_sequence: Array, signal_times: Array, step_sizes: Array
        ) -> Array:
            # Do not hoist this recurrent reconstruction out of the callback.
            # shortcut_forcing_loss invokes it with the primal sequence, the
            # first half-step coordinates, and z_prime/midpoint coordinates.
            return shortcut_predict_clean_sequence(
                params,
                latent_sequence,
                signal_times,
                step_sizes,
                actions,
                is_first,
                config,
            )

        teacher_params = (
            params if shortcut_teacher_params is None else shortcut_teacher_params
        )

        def teacher_predict_clean(
            latent_sequence: Array, signal_times: Array, step_sizes: Array
        ) -> Array:
            return shortcut_predict_clean_sequence(
                teacher_params,
                latent_sequence,
                signal_times,
                step_sizes,
                actions,
                is_first,
                config,
            )

        objective_noise = jax.random.normal(
            jax.random.fold_in(prior_key, 0),
            target_stochastic.shape,
            dtype=target_stochastic.dtype,
        )
        schedule_uniforms = jax.random.uniform(
            jax.random.fold_in(prior_key, 1),
            (*target_stochastic.shape[:2], 6),
            dtype=target_stochastic.dtype,
        )
        shortcut_schedule = sample_shortcut_schedule(
            jax.random.fold_in(prior_key, 2),
            target_stochastic.shape[0],
            target_stochastic.shape[1],
            k_max=config.shortcut_training_k_max,
            dtype=target_stochastic.dtype,
            canonical_uniforms=schedule_uniforms[..., :2],
        )
        prior_details = shortcut_forcing_loss(
            predict_clean,
            target_stochastic,
            None,
            k_max=config.shortcut_training_k_max,
            teacher_predict_clean=teacher_predict_clean,
            intermediate_clip=config.shortcut_intermediate_clip,
            support_safe_bootstrap=config.shortcut_support_safe_bootstrap,
            noise=objective_noise,
            tau=shortcut_schedule.tau,
            step_size=shortcut_schedule.step_size,
            token_mask=loss_mask,
            return_details=True,
        )
        prior = prior_details.loss
        # Reuse the existing public diagnostic field for compatibility.  In
        # shortcut prior mode it denotes the complete published Eq. 7 loss;
        # legacy iMF modes retain their original self-consistency meaning.
        imf_shortcut = prior_details.loss
    elif config.imf_trajectory_enabled:
        # One schedule-mixture expectation replaces separate context,
        # shortcut, and overshooting penalties.  The posterior target and the
        # history exposed as context are stopped data for this prior objective;
        # gradients still train the recurrent condition encoder itself.
        schedule_key = jax.random.fold_in(prior_key, 2)
        trajectory_noise_key = jax.random.fold_in(prior_key, 0)
        loss_key = jax.random.fold_in(prior_key, 3)
        target_stochastic = jax.lax.stop_gradient(sequence.states.stochastic)
        # Independent E_k are required by the conditional-flow derivation.
        # The same E_k defines the token's query path and any later context
        # view of that token; only the time coordinate changes.
        trajectory_noise = jax.random.normal(
            trajectory_noise_key,
            target_stochastic.shape,
            dtype=target_stochastic.dtype,
        )
        schedule_uniforms = jax.random.uniform(
            jax.random.fold_in(prior_key, 1),
            (*target_stochastic.shape[:2], 6),
            dtype=target_stochastic.dtype,
        )
        schedule = sample_trajectory_schedule(
            schedule_key,
            actions.shape[0],
            actions.shape[1],
            token_mask=loss_mask,
            dtype=target_stochastic.dtype,
            boundary_fraction=config.imf_boundary_fraction,
            time_mean=config.imf_time_mean,
            time_std=config.imf_time_std,
            history_noise_max=config.imf_trajectory_history_noise_max,
            pattern_probabilities=(
                config.imf_trajectory_clean_probability,
                config.imf_trajectory_corrupted_probability,
                config.imf_trajectory_suffix_probability,
            ),
            canonical_uniforms=schedule_uniforms,
        )
        corrupted_history = corrupt_trajectory(
            target_stochastic, trajectory_noise, schedule.history_t
        )
        conditions = trajectory_deterministic_conditions(
            params,
            corrupted_history,
            schedule.history_t,
            actions,
            is_first,
            config,
        )
        prior_details = trajectory_imf_loss(
            params["prior"],
            target_stochastic,
            conditions,
            loss_key,
            noise=trajectory_noise,
            r=schedule.r,
            t=schedule.t,
            token_mask=schedule.loss_mask,
            adaptive_power=config.imf_adaptive_power,
            adaptive_epsilon=config.imf_adaptive_epsilon,
            meanflow_scale=config.imf_meanflow_scale,
            velocity_scale=config.imf_velocity_scale,
            signal_weight_floor=config.imf_signal_weight_floor,
            signal_weight_scale=config.imf_signal_weight_scale,
            condition_gradient_scale=config.imf_condition_gradient_scale,
            # Equation (10) uses v(z, c, Q, Q) as both the JVP tangent and
            # auxiliary target.  The legacy scalar-mode switch is therefore
            # intentionally not consulted by trajectory mode.
            boundary_velocity_supervision=True,
            return_details=True,
        )
        prior = prior_details.loss
        imf_loss_u = _masked_mean(prior_details.raw_loss_u, schedule.loss_mask)
        imf_loss_v = _masked_mean(prior_details.raw_loss_v, schedule.loss_mask)
        # These remain exact zeros even if legacy scales are nonzero: they are
        # intentionally absent from the trajectory objective, not hidden in it.
        imf_endpoint = zero
        imf_shortcut = zero
        if config.imf_causal_consistency_scale > 0.0:
            (
                causal_consistency,
                causal_observation,
                causal_reward,
            ) = trajectory_causal_consistency_loss(
                params,
                sequence,
                batch,
                loss_mask,
                causal_key,
                config,
            )
    else:
        transport_noise = None
        if config.imf_noise_coupling == "posterior":
            transport_noise = posterior_sequence_noise(
                sequence_key,
                actions.shape[0],
                actions.shape[1],
                config.stochastic_dim,
                dtype=sequence.states.stochastic.dtype,
            ).reshape((-1, config.stochastic_dim))
        prior_details = improved_meanflow_loss(
            params["prior"],
            jax.lax.stop_gradient(
                sequence.states.stochastic.reshape((-1, config.stochastic_dim))
            ),
            sequence.states.deterministic.reshape((-1, config.deterministic_dim)),
            prior_key,
            noise=transport_noise,
            boundary_fraction=config.imf_boundary_fraction,
            time_mean=config.imf_time_mean,
            time_std=config.imf_time_std,
            adaptive_power=config.imf_adaptive_power,
            adaptive_epsilon=config.imf_adaptive_epsilon,
            meanflow_scale=config.imf_meanflow_scale,
            velocity_scale=config.imf_velocity_scale,
            endpoint_scale=config.imf_endpoint_scale,
            shortcut_scale=config.imf_shortcut_scale,
            signal_weight_floor=config.imf_signal_weight_floor,
            signal_weight_scale=config.imf_signal_weight_scale,
            condition_gradient_scale=config.imf_condition_gradient_scale,
            boundary_velocity_supervision=config.imf_boundary_velocity_supervision,
            weights=loss_mask.reshape((-1,)),
            return_details=True,
        )
        prior = prior_details.loss
        imf_loss_u = _masked_mean(
            prior_details.raw_loss_u.reshape(actions.shape[:2]), loss_mask
        )
        imf_loss_v = _masked_mean(
            prior_details.raw_loss_v.reshape(actions.shape[:2]), loss_mask
        )
        imf_endpoint = _masked_mean(
            prior_details.raw_loss_endpoint.reshape(actions.shape[:2]), loss_mask
        )
        imf_shortcut = _masked_mean(
            prior_details.raw_loss_shortcut.reshape(actions.shape[:2]), loss_mask
        )
    if config.imf_trajectory_enabled or config.prior == "shortcut":
        overshooting = zero
        distance_5 = zero
        distance_15 = zero
    else:
        overshooting, distance_5, distance_15 = multistep_consistency_losses(
            params,
            sequence,
            actions,
            is_first,
            loss_mask,
            overshooting_key,
            config,
        )
    total = (
        config.reconstruction_scale * reconstruction
        + config.reward_scale * reward
        + config.continuation_scale * continuation
        + config.prior_scale * prior
        + config.representation_scale * representation
        + config.overshooting_scale * overshooting
        + config.imf_causal_consistency_scale * causal_consistency
    )
    return WorldModelLoss(
        total=total,
        reconstruction=reconstruction,
        reward=reward,
        continuation=continuation,
        prior=prior,
        representation=representation,
        overshooting=overshooting,
        imf_loss_u=imf_loss_u,
        imf_loss_v=imf_loss_v,
        imf_endpoint=imf_endpoint,
        imf_shortcut=imf_shortcut,
        overshooting_distance_5=distance_5,
        overshooting_distance_15=distance_15,
        causal_consistency=causal_consistency,
        causal_observation=causal_observation,
        causal_reward=causal_reward,
    )


jit_observe_sequence = partial(jax.jit, static_argnames=("config",))(observe_sequence)
jit_world_model_loss = partial(jax.jit, static_argnames=("config",))(world_model_loss)
