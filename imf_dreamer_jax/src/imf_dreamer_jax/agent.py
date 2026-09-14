"""Actor, critic, acting, and functional training state."""

from __future__ import annotations

from functools import partial
import math
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .nn import Params, clip_by_global_norm, init_mlp, mlp, tree_global_norm
from .optim import adam_update, init_adam
from .policy_optimization import (
    ReturnScaleState,
    init_return_scale_state,
    normalized_reinforce_objective,
    update_return_scale,
)
from .types import (
    ActorCriticMetrics,
    ActorSample,
    AgentParams,
    AgentState,
    DiagonalNormal,
    Imagination,
    RSSMState,
    WorldModelLoss,
)
from .uncertainty import pessimistic_rewards
from .world_model import (
    initial_state,
    init_world_model,
    observe_step,
    predict_continuation_logits,
    predict_transition_reward,
    sample_prior,
    transition_deterministic,
    world_model_loss,
)


ImaginationSignalFn = Callable[
    [Params, Array, Array, Array, DreamerConfig], Array
]


def init_actor(config: DreamerConfig, key: Array) -> Params:
    """Initialize the policy with a deliberately small output projection."""

    return init_mlp(
        key,
        config.feature_dim,
        config.hidden_dim,
        2 * config.action_dim,
        output_gain=config.actor_init_scale,
    )


def init_critic(config: DreamerConfig, key: Array) -> Params:
    return init_mlp(
        key,
        config.feature_dim,
        config.hidden_dim,
        config.critic_bins,
        output_gain=config.critic_output_init_scale,
    )


def actor_distribution(
    params: Params,
    feature: Array,
    config: DreamerConfig,
) -> DiagonalNormal:
    """Return a bounded pre-tanh mean and bounded diagonal deviation."""

    raw = mlp(params, feature)
    mean, raw_std = jnp.split(raw, 2, axis=-1)
    mean = config.actor_mean_bound * jnp.tanh(mean / config.actor_mean_bound)
    std = config.actor_min_std + (
        config.actor_max_std - config.actor_min_std
    ) * jax.nn.sigmoid(raw_std)
    return DiagonalNormal(mean, std)


def tanh_normal_log_prob(mean: Array, std: Array, pre_tanh: Array) -> Array:
    """Log density of ``tanh(pre_tanh)`` under a squashed diagonal Normal.

    Taking the pre-tanh sample directly avoids the unstable inverse-tanh that
    would otherwise be required when actions have rounded to exactly +/-1.
    The Jacobian expression is algebraically exact and remains finite for very
    large-magnitude samples.
    """

    if mean.shape != std.shape or mean.shape != pre_tanh.shape:
        raise ValueError("mean, std, and pre_tanh must have identical shapes")
    normal_log_prob = -0.5 * jnp.square((pre_tanh - mean) / std)
    normal_log_prob -= jnp.log(std) + 0.5 * math.log(2.0 * math.pi)
    log_abs_tanh_jacobian = 2.0 * (
        math.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)
    )
    return jnp.sum(normal_log_prob - log_abs_tanh_jacobian, axis=-1)


def squashed_normal_entropy_sample(
    mean: Array, std: Array, pre_tanh: Array
) -> Array:
    """One reparameterized Monte-Carlo sample of squashed-policy entropy."""

    return -tanh_normal_log_prob(mean, std, pre_tanh)


def sample_actor(
    params: Params,
    feature: Array,
    key: Array | None,
    config: DreamerConfig,
    *,
    deterministic: bool = False,
) -> ActorSample:
    """Sample the bounded squashed policy and report its exact density."""

    distribution = actor_distribution(params, feature, config)
    if deterministic:
        pre_tanh = distribution.mean
    else:
        if key is None:
            raise ValueError("key is required for stochastic actions")
        noise = jax.random.normal(
            key, distribution.mean.shape, dtype=distribution.mean.dtype
        )
        pre_tanh = distribution.mean + distribution.std * noise
    action = jnp.tanh(pre_tanh)
    log_prob = tanh_normal_log_prob(
        distribution.mean, distribution.std, pre_tanh
    )
    entropy = squashed_normal_entropy_sample(
        distribution.mean, distribution.std, pre_tanh
    )
    return ActorSample(action, pre_tanh, log_prob, entropy, distribution)


def actor(
    params: Params,
    feature: Array,
    key: Array | None,
    config: DreamerConfig,
    *,
    deterministic: bool = False,
) -> tuple[Array, Array]:
    """Compatibility wrapper returning ``(action, squashed_entropy_sample)``."""

    sample = sample_actor(
        params, feature, key, config, deterministic=deterministic
    )
    return sample.action, sample.entropy


def diagonal_normal_kl(
    left: DiagonalNormal,
    right: DiagonalNormal,
) -> Array:
    """KL(left || right), summed over the event dimension."""

    if left.mean.shape != left.std.shape or right.mean.shape != right.std.shape:
        raise ValueError("normal means and standard deviations must match")
    if left.mean.shape != right.mean.shape:
        raise ValueError("normal distributions must have identical shapes")
    variance_ratio = jnp.square(left.std / right.std)
    location = jnp.square((left.mean - right.mean) / right.std)
    value = jnp.sum(
        jnp.log(right.std / left.std)
        + 0.5 * (variance_ratio + location - 1.0),
        axis=-1,
    )
    # Each analytic KL is nonnegative. Roundoff near equality can otherwise
    # produce tiny negative penalties and misleading telemetry.
    return jnp.maximum(value, 0.0)


def snapshot_behavior_prior(actor_params: Params) -> Params:
    """Create a gradient-isolated functional snapshot of a behavior policy."""

    return jax.tree_util.tree_map(
        lambda value: jax.lax.stop_gradient(jnp.array(value, copy=True)),
        actor_params,
    )


def behavior_cloning_loss(
    actor_params: Params,
    features: Array,
    actions: Array,
    config: DreamerConfig,
) -> Array:
    """Negative log likelihood of replay actions under the squashed actor."""

    if features.shape[:-1] != actions.shape[:-1]:
        raise ValueError("features and actions must have matching leading shapes")
    if features.shape[-1] != config.feature_dim:
        raise ValueError("features have the wrong final dimension")
    if actions.shape[-1] != config.action_dim:
        raise ValueError("actions have the wrong final dimension")
    clipped = jnp.clip(actions, -1.0 + 1e-6, 1.0 - 1e-6)
    pre_tanh = jnp.arctanh(clipped)
    distribution = actor_distribution(actor_params, features, config)
    return -jnp.mean(
        tanh_normal_log_prob(distribution.mean, distribution.std, pre_tanh)
    )


def symlog(value: Array) -> Array:
    """Signed logarithm used by Dreamer-style distributional value heads."""

    return jnp.sign(value) * jnp.log1p(jnp.abs(value))


def symexp(value: Array) -> Array:
    """Inverse of :func:`symlog`."""

    return jnp.sign(value) * jnp.expm1(jnp.abs(value))


def critic_support(config: DreamerConfig, *, dtype: jnp.dtype = jnp.float32) -> Array:
    if config.critic_bins <= 1:
        raise ValueError("critic support is defined only for a distributional critic")
    return jnp.linspace(
        config.critic_symlog_min,
        config.critic_symlog_max,
        config.critic_bins,
        dtype=dtype,
    )


def critic_logits(params: Params, feature: Array, config: DreamerConfig) -> Array:
    logits = mlp(params, feature)
    if logits.shape[-1] != config.critic_bins:
        raise ValueError("critic output dimension does not match critic_bins")
    return logits


def two_hot_symlog(target: Array, config: DreamerConfig) -> Array:
    """Encode scalar returns by linear interpolation on a symlog support."""

    support = critic_support(config, dtype=target.dtype)
    transformed = jnp.clip(
        symlog(target), config.critic_symlog_min, config.critic_symlog_max
    )
    width = support[1] - support[0]
    position = (transformed - support[0]) / width
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, config.critic_bins - 1)
    upper_weight = position - lower.astype(position.dtype)
    lower_weight = 1.0 - upper_weight
    encoded = jax.nn.one_hot(lower, config.critic_bins, dtype=target.dtype) * lower_weight[..., None]
    encoded += jax.nn.one_hot(upper, config.critic_bins, dtype=target.dtype) * upper_weight[..., None]
    return encoded


def critic(params: Params, feature: Array, config: DreamerConfig | None = None) -> Array:
    """Return scalar values from either scalar or symlog/two-hot logits."""

    output = mlp(params, feature)
    if config is None or config.critic_bins == 1:
        if output.shape[-1] != 1:
            raise ValueError("distributional critic evaluation requires config")
        return output[..., 0]
    if output.shape[-1] != config.critic_bins:
        raise ValueError("critic output dimension does not match critic_bins")
    mean_symlog = jnp.sum(
        jax.nn.softmax(output, axis=-1)
        * critic_support(config, dtype=output.dtype),
        axis=-1,
    )
    return symexp(mean_symlog)


def loss_bearing_imagination_starts(
    states: RSSMState, burn_in: int
) -> RSSMState:
    """Flatten every post-burn-in posterior state into an imagination start."""

    if not isinstance(burn_in, int) or isinstance(burn_in, bool) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    if states.deterministic.ndim != 3 or states.stochastic.ndim != 3:
        raise ValueError("sequence states must have [batch, time, feature] shape")
    if states.deterministic.shape[:2] != states.stochastic.shape[:2]:
        raise ValueError("deterministic and stochastic sequence prefixes must match")
    batch, time = states.deterministic.shape[:2]
    if burn_in >= time:
        raise ValueError("burn_in must leave at least one imagination start")
    return RSSMState(
        states.deterministic[:, burn_in:].reshape(
            (batch * (time - burn_in), states.deterministic.shape[-1])
        ),
        states.stochastic[:, burn_in:].reshape(
            (batch * (time - burn_in), states.stochastic.shape[-1])
        ),
    )


def diverse_imagination_starts(
    states: RSSMState,
    burn_in: int,
    key: Array,
) -> RSSMState:
    """Choose one uniformly random post-burn-in state from each sequence.

    This follows Dreamer 4's preference for context diversity over launching
    many highly correlated rollouts from every time step of a replay chunk.
    """

    if not isinstance(burn_in, int) or isinstance(burn_in, bool) or burn_in < 0:
        raise ValueError("burn_in must be a nonnegative integer")
    if states.deterministic.ndim != 3 or states.stochastic.ndim != 3:
        raise ValueError("sequence states must have [batch, time, feature] shape")
    if states.deterministic.shape[:2] != states.stochastic.shape[:2]:
        raise ValueError("deterministic and stochastic sequence prefixes must match")
    batch, time = states.deterministic.shape[:2]
    if burn_in >= time:
        raise ValueError("burn_in must leave at least one imagination start")
    offsets = jax.random.randint(key, (batch,), burn_in, time)
    rows = jnp.arange(batch)
    return RSSMState(
        states.deterministic[rows, offsets],
        states.stochastic[rows, offsets],
    )


def create_agent(config: DreamerConfig, key: Array) -> AgentState:
    """Create parameters and independent Adam states from one PRNG key."""

    model_key, actor_key, critic_key = jax.random.split(key, 3)
    params = AgentParams(
        init_world_model(config, model_key),
        init_actor(config, actor_key),
        init_critic(config, critic_key),
    )
    return AgentState(
        params,
        init_adam(params.world_model),
        init_adam(params.actor),
        init_adam(params.critic),
        params.critic,
        (
            params.world_model
            if config.prior == "shortcut"
            and config.shortcut_bootstrap_ema_decay is not None
            else None
        ),
    )


def act(
    params: AgentParams,
    observation: Array,
    previous_action: Array,
    state: RSSMState,
    key: Array,
    config: DreamerConfig,
    *,
    deterministic: bool = False,
) -> tuple[Array, RSSMState]:
    posterior_key, actor_key = jax.random.split(key)
    next_state, _, _ = observe_step(
        params.world_model,
        observation,
        previous_action,
        state,
        posterior_key,
        config,
    )
    action, _ = actor(
        params.actor, next_state.feature, actor_key, config, deterministic=deterministic
    )
    return action, next_state


def lambda_returns(
    rewards: Array,
    values: Array,
    continuations: Array,
    *,
    discount: float,
    lambda_: float,
) -> Array:
    if rewards.shape != continuations.shape:
        raise ValueError("rewards and continuations must have identical shape")
    if values.shape != (*rewards.shape[:-1], rewards.shape[-1] + 1):
        raise ValueError("values must have one additional horizon element")
    accumulator = values[..., -1]
    reversed_returns: list[Array] = []
    for index in range(rewards.shape[-1] - 1, -1, -1):
        bootstrap = (1.0 - lambda_) * values[..., index + 1] + lambda_ * accumulator
        accumulator = rewards[..., index] + discount * continuations[..., index] * bootstrap
        reversed_returns.append(accumulator)
    return jnp.stack(tuple(reversed(reversed_returns)), axis=-1)


def imagine(
    params: AgentParams,
    start: RSSMState,
    key: Array,
    config: DreamerConfig,
    *,
    horizon: int | None = None,
    reward_fn: ImaginationSignalFn | None = None,
    continuation_fn: ImaginationSignalFn | None = None,
) -> Imagination:
    horizon = config.imagination_horizon if horizon is None else horizon
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    step_keys = jax.random.split(key, horizon)

    def step(state: RSSMState, step_key: Array):
        previous_feature = state.feature
        actor_key, prior_key = jax.random.split(step_key)
        actor_sample = sample_actor(
            params.actor, state.feature, actor_key, config
        )
        if config.actor_gradient in ("reinforce", "pmpo"):
            model_action = jax.lax.stop_gradient(actor_sample.action)
        else:
            model_action = actor_sample.action
        deterministic = transition_deterministic(
            params.world_model, state, model_action, config
        )
        stochastic, _ = sample_prior(
            params.world_model, deterministic, prior_key, config
        )
        next_state = RSSMState(deterministic, stochastic)
        reward = (
            predict_transition_reward(
                params.world_model,
                previous_feature,
                actor_sample.action,
                next_state.feature,
                config,
            )
            if reward_fn is None
            else reward_fn(
                params.world_model,
                previous_feature,
                actor_sample.action,
                next_state.feature,
                config,
            )
        )
        continuation = (
            jax.nn.sigmoid(
                predict_continuation_logits(params.world_model, next_state.feature)
            )
            if continuation_fn is None
            else continuation_fn(
                params.world_model,
                previous_feature,
                actor_sample.action,
                next_state.feature,
                config,
            )
        )
        if reward.shape != actor_sample.action.shape[:-1]:
            raise ValueError("imagination reward_fn returned the wrong shape")
        if continuation.shape != reward.shape:
            raise ValueError("imagination continuation_fn returned the wrong shape")
        output = (
            next_state.feature,
            actor_sample.action,
            reward,
            continuation,
            actor_sample.entropy,
            tanh_normal_log_prob(
                actor_sample.distribution.mean,
                actor_sample.distribution.std,
                jax.lax.stop_gradient(actor_sample.pre_tanh),
            ),
            actor_sample.distribution.mean,
            actor_sample.pre_tanh,
        )
        return next_state, output

    _, outputs = jax.lax.scan(step, start, step_keys)
    (
        features,
        actions,
        rewards,
        continuations,
        entropies,
        log_probs,
        pre_tanh_means,
        pre_tanh_values,
    ) = (
        jnp.swapaxes(value, 0, 1) for value in outputs
    )
    features = jnp.concatenate((start.feature[:, None], features), axis=1)
    return Imagination(
        features,
        actions,
        rewards,
        continuations,
        entropies,
        log_probs,
        pre_tanh_means,
        pre_tanh_values,
    )


def train_world_model(
    state: AgentState,
    batch: Mapping[str, Array],
    key: Array,
    config: DreamerConfig,
) -> tuple[AgentState, WorldModelLoss]:
    """Run one clipped Adam update of the recurrent world model."""

    def objective(model_params: Params):
        losses = world_model_loss(
            model_params,
            batch,
            key,
            config,
            shortcut_teacher_params=state.world_model_teacher,
        )
        return losses.total, losses

    (_, losses), gradients = jax.value_and_grad(objective, has_aux=True)(
        state.params.world_model
    )
    gradients, _ = clip_by_global_norm(gradients, config.grad_clip)
    model_params, optimizer = adam_update(
        state.params.world_model,
        gradients,
        state.model_optimizer,
        learning_rate=config.model_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    world_model_teacher = state.world_model_teacher
    if config.prior == "shortcut" and config.shortcut_bootstrap_ema_decay is not None:
        if world_model_teacher is None:
            world_model_teacher = state.params.world_model
        world_model_teacher = _ema(
            world_model_teacher,
            model_params,
            1.0 - config.shortcut_bootstrap_ema_decay,
        )
    elif config.prior != "shortcut":
        world_model_teacher = None
    params = AgentParams(model_params, state.params.actor, state.params.critic)
    return AgentState(
        params,
        optimizer,
        state.actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        world_model_teacher,
    ), losses


def train_behavior_cloning(
    state: AgentState,
    features: Array,
    actions: Array,
    config: DreamerConfig,
) -> tuple[AgentState, Array]:
    """Update only the actor on replay actions; the world model stays frozen."""

    objective = lambda actor_params: behavior_cloning_loss(
        actor_params, features, actions, config
    )
    loss, gradients = jax.value_and_grad(objective)(state.params.actor)
    gradients, _ = clip_by_global_norm(gradients, config.grad_clip)
    actor_params, actor_optimizer = adam_update(
        state.params.actor,
        gradients,
        state.actor_optimizer,
        learning_rate=config.behavior_cloning_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    params = AgentParams(state.params.world_model, actor_params, state.params.critic)
    return AgentState(
        params,
        state.model_optimizer,
        actor_optimizer,
        state.critic_optimizer,
        state.slow_critic,
        state.world_model_teacher,
    ), loss


def _discount_weights(
    continuations: Array, *, discount: float
) -> Array:
    """Dreamer-style cumulative survival weights for imagined transitions."""

    first = jnp.ones_like(continuations[:, :1])
    remaining = jnp.cumprod(
        discount * continuations[:, :-1], axis=1
    )
    return jnp.concatenate((first, remaining), axis=1)


def _weighted_mean(values: Array, weights: Array) -> Array:
    if values.shape != weights.shape:
        raise ValueError("weighted values and weights must have identical shapes")
    return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1e-8)


def _fixed_denominator_weighted_mean(values: Array, weights: Array) -> Array:
    """Weight survival without an action-dependent self-normalizing denominator."""

    if values.shape != weights.shape:
        raise ValueError("weighted values and weights must have identical shapes")
    return jnp.mean(values * weights)


def sign_only_pmpo_objective(
    log_probs: Array,
    advantages: Array,
    weights: Array,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> Array:
    """Dreamer 4 PMPO objective using only advantage signs.

    Positive and negative sets are normalized separately, so rescaling an
    advantage without crossing zero cannot change the objective. Empty sets
    contribute zero and never create a division by zero.
    """

    if log_probs.shape != advantages.shape or log_probs.shape != weights.shape:
        raise ValueError("PMPO log_probs, advantages, and weights must match")
    if positive_weight < 0.0 or negative_weight < 0.0:
        raise ValueError("PMPO set weights must be nonnegative")
    total_set_weight = positive_weight + negative_weight
    if total_set_weight <= 0.0:
        raise ValueError("at least one PMPO set weight must be positive")
    positive = (advantages >= 0.0).astype(log_probs.dtype)
    negative = 1.0 - positive

    def set_mean(mask: Array) -> Array:
        mass = jnp.sum(weights * mask)
        mean = jnp.sum(log_probs * weights * mask) / jnp.maximum(mass, 1e-8)
        return jnp.where(mass > 0.0, mean, 0.0)

    return (
        (positive_weight / total_set_weight) * set_mean(positive)
        - (negative_weight / total_set_weight) * set_mean(negative)
    )


def _ema(target: Params, source: Params, fraction: float) -> Params:
    return jax.tree_util.tree_map(
        lambda old, new: (1.0 - fraction) * old + fraction * new,
        target,
        source,
    )


def _warmup_learning_rate(base: float, optimizer_step: Array, warmup_steps: int) -> Array:
    """Linear warmup evaluated for the update that is about to be applied."""

    if warmup_steps == 0:
        return jnp.asarray(base, dtype=jnp.float32)
    fraction = jnp.minimum(
        1.0,
        (optimizer_step.astype(jnp.float32) + 1.0) / float(warmup_steps),
    )
    return jnp.asarray(base, dtype=jnp.float32) * fraction


def train_replay_critic(
    state: AgentState,
    features: Array,
    rewards: Array,
    continuations: Array,
    config: DreamerConfig,
    *,
    loss_mask: Array | None = None,
) -> tuple[AgentState, Array]:
    """Ground the critic on real replay returns, freezing actor and model."""

    if features.ndim != 3 or features.shape[-1] != config.feature_dim:
        raise ValueError("features must have shape [batch, time, feature_dim]")
    if rewards.shape != features.shape[:2] or continuations.shape != rewards.shape:
        raise ValueError("rewards and continuations must match feature leading axes")
    if loss_mask is None:
        loss_mask = jnp.ones_like(rewards)
    elif loss_mask.shape != rewards.shape:
        raise ValueError("loss_mask must match rewards")
    features = jax.lax.stop_gradient(features)
    rewards = jax.lax.stop_gradient(rewards)
    continuations = jax.lax.stop_gradient(continuations)
    loss_mask = jax.lax.stop_gradient(jnp.asarray(loss_mask, dtype=rewards.dtype))
    slow_critic = state.params.critic if state.slow_critic is None else state.slow_critic
    stopped_values = jax.lax.stop_gradient(critic(slow_critic, features, config))
    value_path = jnp.concatenate((stopped_values, stopped_values[:, -1:]), axis=1)
    targets = jax.lax.stop_gradient(
        lambda_returns(
            rewards,
            value_path,
            continuations,
            discount=config.discount,
            lambda_=config.lambda_,
        )
    )

    def objective(critic_params: Params) -> Array:
        if config.critic_bins == 1:
            per_step = jnp.square(critic(critic_params, features, config) - targets)
        else:
            logits = critic_logits(critic_params, features, config)
            labels = two_hot_symlog(targets, config)
            per_step = -jnp.sum(
                labels * jax.nn.log_softmax(logits, axis=-1), axis=-1
            )
        return jnp.sum(per_step * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)

    loss, gradients = jax.value_and_grad(objective)(state.params.critic)
    gradients, _ = clip_by_global_norm(gradients, config.grad_clip)
    critic_params, critic_optimizer = adam_update(
        state.params.critic,
        gradients,
        state.critic_optimizer,
        learning_rate=_warmup_learning_rate(
            config.critic_learning_rate,
            state.critic_optimizer.step,
            config.actor_critic_warmup_steps,
        ),
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    updated_slow_critic = _ema(
        slow_critic, critic_params, config.slow_critic_fraction
    )
    params = AgentParams(state.params.world_model, state.params.actor, critic_params)
    return AgentState(
        params,
        state.model_optimizer,
        state.actor_optimizer,
        critic_optimizer,
        updated_slow_critic,
        state.world_model_teacher,
    ), loss


def _train_actor_critic(
    state: AgentState,
    start: RSSMState,
    key: Array,
    config: DreamerConfig,
    return_scale_state: ReturnScaleState,
    *,
    use_percentile_ema_scale: bool,
    behavior_prior: Params | None = None,
    reward_ensemble_params: Params | None = None,
    epistemic_penalty_scale: float = 0.0,
    reward_fn: ImaginationSignalFn | None = None,
    continuation_fn: ImaginationSignalFn | None = None,
) -> tuple[AgentState, ReturnScaleState, ActorCriticMetrics]:
    """Update a stable actor and critic from imagined trajectories.

    The default estimator is a stopped-action likelihood-ratio objective. This
    prevents the policy from following arbitrary derivatives of the learned
    transition/reward model while retaining an opt-in pathwise ``dynamics``
    estimator. Lambda-return targets bootstrap from an EMA critic, and the
    advantage is normalized by the current imagined-return scale.
    """

    if epistemic_penalty_scale < 0.0:
        raise ValueError("epistemic_penalty_scale must be nonnegative")
    if epistemic_penalty_scale > 0.0 and reward_ensemble_params is None:
        raise ValueError(
            "reward_ensemble_params is required for epistemic pessimism"
        )
    start = jax.tree_util.tree_map(jax.lax.stop_gradient, start)
    actor_key, critic_rollout_key = jax.random.split(key)
    slow_critic = (
        state.params.critic
        if state.slow_critic is None
        else state.slow_critic
    )

    def actor_objective(actor_params: Params):
        params = AgentParams(state.params.world_model, actor_params, state.params.critic)
        imagined = imagine(
            params,
            start,
            actor_key,
            config,
            reward_fn=reward_fn,
            continuation_fn=continuation_fn,
        )
        target_values = critic(slow_critic, imagined.features, config)
        imagined_rewards = imagined.rewards
        if epistemic_penalty_scale > 0.0:
            imagined_rewards, _ = pessimistic_rewards(
                imagined_rewards,
                jax.lax.stop_gradient(imagined.features[:, 1:]),
                reward_ensemble_params,
                penalty_scale=epistemic_penalty_scale,
            )
        returns = lambda_returns(
            imagined_rewards,
            target_values,
            imagined.continuations,
            discount=config.discount,
            lambda_=config.lambda_,
        )
        if use_percentile_ema_scale:
            updated_return_scale_state, return_scale = update_return_scale(
                return_scale_state,
                returns,
                decay=config.return_scale_ema_decay,
            )
        else:
            return_mean = jax.lax.stop_gradient(jnp.mean(returns))
            return_scale = jax.lax.stop_gradient(
                jnp.maximum(
                    1.0,
                    jnp.sqrt(
                        jnp.mean(jnp.square(returns - return_mean))
                        + config.return_normalization_epsilon**2
                    ),
                )
            )
            updated_return_scale_state = return_scale_state
        weights = jax.lax.stop_gradient(
            _discount_weights(imagined.continuations, discount=config.discount)
        )
        baseline = critic(
            state.params.critic, imagined.features[:, :-1], config
        )
        raw_advantage = jax.lax.stop_gradient(returns - baseline)
        if config.actor_gradient in ("reinforce", "pmpo"):
            if config.actor_gradient == "reinforce":
                reward_objective = normalized_reinforce_objective(
                    imagined.log_probs,
                    raw_advantage,
                    weights,
                    return_scale,
                )
            else:
                reward_objective = sign_only_pmpo_objective(
                    imagined.log_probs,
                    raw_advantage / return_scale,
                    weights,
                    positive_weight=config.pmpo_positive_weight,
                    negative_weight=config.pmpo_negative_weight,
                )
        else:
            reward_objective = _fixed_denominator_weighted_mean(
                returns / return_scale, weights
            )
        behavior_kl = jnp.asarray(0.0, dtype=returns.dtype)
        if config.behavior_kl_scale > 0.0:
            if behavior_prior is None:
                raise ValueError(
                    "behavior_prior is required when behavior_kl_scale is positive"
                )
            policy_distribution = actor_distribution(
                actor_params,
                jax.lax.stop_gradient(imagined.features[:, :-1]),
                config,
            )
            prior_distribution = actor_distribution(
                behavior_prior,
                jax.lax.stop_gradient(imagined.features[:, :-1]),
                config,
            )
            behavior_kl = _fixed_denominator_weighted_mean(
                diagonal_normal_kl(policy_distribution, prior_distribution),
                weights,
            )
        entropy = _fixed_denominator_weighted_mean(imagined.entropies, weights)
        action_l2 = _fixed_denominator_weighted_mean(
            jnp.mean(jnp.square(imagined.pre_tanh_values), axis=-1), weights
        )
        loss = (
            -reward_objective
            - config.actor_entropy_scale * entropy
            + config.actor_action_l2_scale * action_l2
            + config.behavior_kl_scale * behavior_kl
        )
        auxiliary = (
            imagined,
            returns,
            return_scale,
            updated_return_scale_state,
            raw_advantage,
            behavior_kl,
        )
        return loss, auxiliary

    (actor_loss, actor_auxiliary), actor_gradients = jax.value_and_grad(
        actor_objective, has_aux=True
    )(state.params.actor)
    actor_gradients, actor_grad_norm = clip_by_global_norm(
        actor_gradients, config.grad_clip
    )
    actor_params, actor_optimizer = adam_update(
        state.params.actor,
        actor_gradients,
        state.actor_optimizer,
        learning_rate=_warmup_learning_rate(
            config.actor_learning_rate,
            state.actor_optimizer.step,
            config.actor_critic_warmup_steps,
        ),
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    rollout_params = AgentParams(state.params.world_model, actor_params, state.params.critic)
    imagined = imagine(
        rollout_params,
        start,
        critic_rollout_key,
        config,
        reward_fn=reward_fn,
        continuation_fn=continuation_fn,
    )
    target_values = critic(slow_critic, imagined.features, config)
    imagined_rewards = imagined.rewards
    if epistemic_penalty_scale > 0.0:
        imagined_rewards, _ = pessimistic_rewards(
            imagined_rewards,
            jax.lax.stop_gradient(imagined.features[:, 1:]),
            reward_ensemble_params,
            penalty_scale=epistemic_penalty_scale,
        )
    targets = jax.lax.stop_gradient(
        lambda_returns(
            imagined_rewards,
            target_values,
            imagined.continuations,
            discount=config.discount,
            lambda_=config.lambda_,
        )
    )
    critic_features = jax.lax.stop_gradient(imagined.features[:, :-1])
    critic_weights = jax.lax.stop_gradient(
        _discount_weights(imagined.continuations, discount=config.discount)
    )

    def critic_objective(critic_params: Params) -> Array:
        if config.critic_bins == 1:
            predictions = critic(critic_params, critic_features, config)
            per_step = jnp.square(predictions - targets)
        else:
            logits = critic_logits(critic_params, critic_features, config)
            labels = two_hot_symlog(targets, config)
            per_step = -jnp.sum(
                labels * jax.nn.log_softmax(logits, axis=-1), axis=-1
            )
        return _fixed_denominator_weighted_mean(per_step, critic_weights)

    critic_loss, critic_gradients = jax.value_and_grad(critic_objective)(
        state.params.critic
    )
    critic_gradients, critic_grad_norm = clip_by_global_norm(
        critic_gradients, config.grad_clip
    )
    critic_params, critic_optimizer = adam_update(
        state.params.critic,
        critic_gradients,
        state.critic_optimizer,
        learning_rate=_warmup_learning_rate(
            config.critic_learning_rate,
            state.critic_optimizer.step,
            config.actor_critic_warmup_steps,
        ),
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    updated_slow_critic = _ema(
        slow_critic, critic_params, config.slow_critic_fraction
    )
    slow_critic_delta = tree_global_norm(
        jax.tree_util.tree_map(
            lambda online, target: online - target,
            critic_params,
            updated_slow_critic,
        )
    )
    params = AgentParams(state.params.world_model, actor_params, critic_params)
    new_state = AgentState(
        params,
        state.model_optimizer,
        actor_optimizer,
        critic_optimizer,
        updated_slow_critic,
        state.world_model_teacher,
    )
    (
        actor_imagined,
        actor_returns,
        return_scale,
        updated_return_scale_state,
        raw_advantage,
        behavior_kl,
    ) = actor_auxiliary
    actor_weights = _discount_weights(
        actor_imagined.continuations, discount=config.discount
    )
    policy_distribution = actor_distribution(
        state.params.actor,
        jax.lax.stop_gradient(actor_imagined.features[:, :-1]),
        config,
    )
    metrics = ActorCriticMetrics(
        actor_loss,
        critic_loss,
        jnp.mean(actor_returns[:, 0]),
        actor_grad_norm,
        critic_grad_norm,
        _weighted_mean(
            jnp.mean(jnp.abs(actor_imagined.actions) >= 0.95, axis=-1),
            actor_weights,
        ),
        _weighted_mean(
            jnp.mean(jnp.abs(actor_imagined.pre_tanh_means), axis=-1),
            actor_weights,
        ),
        _weighted_mean(actor_imagined.entropies, actor_weights),
        return_scale,
        slow_critic_delta,
        _weighted_mean(
            jnp.mean(jnp.tanh(actor_imagined.pre_tanh_means), axis=-1),
            actor_weights,
        ),
        _weighted_mean(
            jnp.mean(actor_imagined.pre_tanh_means, axis=-1),
            actor_weights,
        ),
        _weighted_mean(
            jnp.mean(policy_distribution.std, axis=-1),
            actor_weights,
        ),
        behavior_kl,
        _weighted_mean(raw_advantage, actor_weights),
        _weighted_mean(
            (raw_advantage > 0.0).astype(raw_advantage.dtype),
            actor_weights,
        ),
    )
    return new_state, updated_return_scale_state, metrics


def train_actor_critic(
    state: AgentState,
    start: RSSMState,
    key: Array,
    config: DreamerConfig,
    *,
    behavior_prior: Params | None = None,
    reward_ensemble_params: Params | None = None,
    epistemic_penalty_scale: float = 0.0,
    reward_fn: ImaginationSignalFn | None = None,
    continuation_fn: ImaginationSignalFn | None = None,
) -> tuple[AgentState, ActorCriticMetrics]:
    """Compatibility update using the historical per-batch RMS scale.

    New continuous-control experiments should call
    :func:`train_actor_critic_dreamer3`, which carries the robust percentile
    EMA explicitly without changing the checkpointed ``AgentState`` tuple.
    """

    updated, _, metrics = _train_actor_critic(
        state,
        start,
        key,
        config,
        init_return_scale_state(dtype=start.deterministic.dtype),
        use_percentile_ema_scale=False,
        behavior_prior=behavior_prior,
        reward_ensemble_params=reward_ensemble_params,
        epistemic_penalty_scale=epistemic_penalty_scale,
        reward_fn=reward_fn,
        continuation_fn=continuation_fn,
    )
    return updated, metrics


def train_actor_critic_dreamer3(
    state: AgentState,
    return_scale_state: ReturnScaleState,
    start: RSSMState,
    key: Array,
    config: DreamerConfig,
    *,
    behavior_prior: Params | None = None,
    reward_ensemble_params: Params | None = None,
    epistemic_penalty_scale: float = 0.0,
    reward_fn: ImaginationSignalFn | None = None,
    continuation_fn: ImaginationSignalFn | None = None,
) -> tuple[AgentState, ReturnScaleState, ActorCriticMetrics]:
    """Dreamer-style REINFORCE update with a 5th--95th percentile EMA scale."""

    if config.actor_gradient != "reinforce":
        raise ValueError(
            "train_actor_critic_dreamer3 requires actor_gradient='reinforce'"
        )
    return _train_actor_critic(
        state,
        start,
        key,
        config,
        return_scale_state,
        use_percentile_ema_scale=True,
        behavior_prior=behavior_prior,
        reward_ensemble_params=reward_ensemble_params,
        epistemic_penalty_scale=epistemic_penalty_scale,
        reward_fn=reward_fn,
        continuation_fn=continuation_fn,
    )


jit_act = partial(jax.jit, static_argnames=("config", "deterministic"))(act)
jit_imagine = partial(
    jax.jit,
    static_argnames=("config", "horizon", "reward_fn", "continuation_fn"),
)(imagine)
jit_train_behavior_cloning = partial(jax.jit, static_argnames=("config",))(
    train_behavior_cloning
)
jit_train_replay_critic = partial(jax.jit, static_argnames=("config",))(
    train_replay_critic
)
jit_train_world_model = partial(jax.jit, static_argnames=("config",))(train_world_model)
jit_train_actor_critic = partial(
    jax.jit,
    static_argnames=(
        "config",
        "epistemic_penalty_scale",
        "reward_fn",
        "continuation_fn",
    ),
)(train_actor_critic)
jit_train_actor_critic_dreamer3 = partial(
    jax.jit,
    static_argnames=(
        "config",
        "epistemic_penalty_scale",
        "reward_fn",
        "continuation_fn",
    ),
)(train_actor_critic_dreamer3)


__all__ = [
    "act",
    "actor",
    "actor_distribution",
    "behavior_cloning_loss",
    "create_agent",
    "critic",
    "critic_logits",
    "critic_support",
    "diagonal_normal_kl",
    "diverse_imagination_starts",
    "imagine",
    "ImaginationSignalFn",
    "initial_state",
    "jit_act",
    "jit_imagine",
    "jit_train_behavior_cloning",
    "jit_train_replay_critic",
    "jit_train_actor_critic",
    "jit_train_actor_critic_dreamer3",
    "jit_train_world_model",
    "lambda_returns",
    "loss_bearing_imagination_starts",
    "sample_actor",
    "snapshot_behavior_prior",
    "sign_only_pmpo_objective",
    "squashed_normal_entropy_sample",
    "symexp",
    "symlog",
    "tanh_normal_log_prob",
    "train_actor_critic",
    "train_actor_critic_dreamer3",
    "train_behavior_cloning",
    "train_replay_critic",
    "train_world_model",
    "two_hot_symlog",
]
