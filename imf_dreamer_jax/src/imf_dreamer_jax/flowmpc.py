"""Paper-faithful ReBRAC initialization and FlowMPC policy adaptation.

The controller follows Algorithm 1 of ITPO/FlowMPC.  A deterministic policy
is first trained offline with the released ReBRAC update.  At deployment, only
that policy is changed: a fixed transition/reward model and fixed terminal
critic define a Monte-Carlo finite-horizon objective, and the policy receives
one or more direct gradient-ascent steps before its current action is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .config import DreamerConfig
from .nn import Params, tree_global_norm
from .optim import adam_update, init_adam
from .types import AdamState, RSSMState
from .world_model import (
    decode,
    predict_state_action_reward_from_observation,
    preprocess_observation,
    sample_prior,
    transition_deterministic,
)


@dataclass(frozen=True)
class ReBRACConfig:
    """Reference ReBRAC defaults used by the FlowMPC paper."""

    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 3
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    discount: float = 0.99
    target_update_rate: float = 5e-3
    actor_bc_coefficient: float = 1.0
    critic_bc_coefficient: float = 1.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    normalize_q: bool = True
    batch_size: int = 1024

    def __post_init__(self) -> None:
        for name in ("state_dim", "action_dim", "hidden_dim", "hidden_layers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_layers != 3:
            raise ValueError("the paper-faithful ReBRAC network has three hidden layers")
        if self.batch_size != 1024:
            raise ValueError("the paper-faithful ReBRAC batch size is 1024")
        if self.policy_frequency != 2:
            raise ValueError("the released ReBRAC actor updates every two critic steps")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "actor_bc_coefficient",
            "critic_bc_coefficient",
            "policy_noise",
            "noise_clip",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not 0.0 < self.target_update_rate <= 1.0:
            raise ValueError("target_update_rate must be in (0, 1]")


@dataclass(frozen=True)
class FlowMPCConfig:
    """Inference-time parameters from the paper's MPC objective."""

    horizon: int = 5
    particles: int = 4096
    inner_steps: int = 1
    step_size: float = 5e-5
    discount: float = 0.99

    def __post_init__(self) -> None:
        if (
            not isinstance(self.horizon, int)
            or isinstance(self.horizon, bool)
            or self.horizon < 0
        ):
            raise ValueError("horizon must be a nonnegative integer")
        for name in ("particles", "inner_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if not math.isfinite(self.discount) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")


class ReBRACDataset(NamedTuple):
    states: Array
    actions: Array
    rewards: Array
    next_states: Array
    next_actions: Array
    dones: Array


class ReBRACState(NamedTuple):
    step: Array
    actor: Params
    target_actor: Params
    critics: Params
    target_critics: Params
    actor_optimizer: AdamState
    critic_optimizer: AdamState


class ReBRACMetrics(NamedTuple):
    critic_loss: Array
    actor_loss: Array
    q_min: Array
    target_q: Array
    actor_bc_penalty: Array
    critic_bc_penalty: Array
    q_scale: Array
    actor_updated: Array


class FlowMPCObjective(NamedTuple):
    objective: Array
    stage_return: Array
    terminal_value: Array
    rewards: Array
    terminal_q: Array


class FlowMPCUpdate(NamedTuple):
    actor: Params
    objective_before: Array
    objective_after: Array
    stage_return: Array
    terminal_value: Array
    gradient_norm: Array
    parameter_delta: Array


def _uniform(key: Array, shape: tuple[int, ...], bound: float) -> Array:
    return jax.random.uniform(
        key, shape, minval=-bound, maxval=bound, dtype=jnp.float32
    )


def _init_hidden_layer(
    key: Array, input_dim: int, output_dim: int, *, layer_norm: bool
) -> Params:
    layer = {
        "weight": _uniform(key, (input_dim, output_dim), math.sqrt(1.0 / input_dim)),
        "bias": jnp.full((output_dim,), 0.1, dtype=jnp.float32),
    }
    if layer_norm:
        layer["norm_scale"] = jnp.ones((output_dim,), dtype=jnp.float32)
        layer["norm_bias"] = jnp.zeros((output_dim,), dtype=jnp.float32)
    return layer


def _init_output_layer(
    key: Array, input_dim: int, output_dim: int, *, bound: float
) -> Params:
    weight_key, bias_key = jax.random.split(key)
    return {
        "weight": _uniform(weight_key, (input_dim, output_dim), bound),
        "bias": _uniform(bias_key, (output_dim,), bound),
    }


def _init_rebrac_network(
    key: Array,
    input_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    output_dim: int,
    *,
    output_bound: float,
    layer_norm: bool,
) -> Params:
    keys = jax.random.split(key, hidden_layers + 1)
    layers: list[Params] = []
    width = input_dim
    for index in range(hidden_layers):
        layers.append(
            _init_hidden_layer(
                keys[index], width, hidden_dim, layer_norm=layer_norm
            )
        )
        width = hidden_dim
    output = _init_output_layer(
        keys[-1], width, output_dim, bound=output_bound
    )
    return {"hidden": tuple(layers), "output": output}


def init_rebrac_actor(key: Array, config: ReBRACConfig) -> Params:
    """Initialize the deterministic actor as in the released ReBRAC code."""

    return _init_rebrac_network(
        key,
        config.state_dim,
        config.hidden_dim,
        config.hidden_layers,
        config.action_dim,
        output_bound=1e-3,
        layer_norm=False,
    )


def init_rebrac_critics(key: Array, config: ReBRACConfig) -> Params:
    """Initialize the two LayerNorm critics used by ReBRAC."""

    keys = jax.random.split(key, 2)
    return {
        "members": tuple(
            _init_rebrac_network(
                member_key,
                config.state_dim + config.action_dim,
                config.hidden_dim,
                config.hidden_layers,
                1,
                output_bound=3e-3,
                layer_norm=True,
            )
            for member_key in keys
        )
    }


def _layer_norm(layer: Params, value: Array) -> Array:
    mean = jnp.mean(value, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
    normalized = (value - mean) * jax.lax.rsqrt(variance + 1e-6)
    return normalized * layer["norm_scale"] + layer["norm_bias"]


def _network(params: Params, value: Array, *, layer_norm: bool) -> Array:
    for layer in params["hidden"]:
        value = jax.nn.relu(value @ layer["weight"] + layer["bias"])
        if layer_norm:
            value = _layer_norm(layer, value)
    output = params["output"]
    return value @ output["weight"] + output["bias"]


def rebrac_actor(params: Params, states: Array) -> Array:
    """Evaluate the deterministic tanh policy."""

    return jnp.tanh(_network(params, states, layer_norm=False))


def rebrac_critics(params: Params, states: Array, actions: Array) -> Array:
    """Evaluate the two scalar LayerNorm critics as ``[critic, ...]``."""

    if states.shape[:-1] != actions.shape[:-1]:
        raise ValueError("critic states and actions must share leading dimensions")
    inputs = jnp.concatenate((states, actions), axis=-1)
    return jnp.stack(
        tuple(
            _network(member, inputs, layer_norm=True)[..., 0]
            for member in params["members"]
        ),
        axis=0,
    )


def init_rebrac_state(key: Array, config: ReBRACConfig) -> ReBRACState:
    actor_key, critic_key = jax.random.split(key)
    actor = init_rebrac_actor(actor_key, config)
    critics = init_rebrac_critics(critic_key, config)
    target_actor = jax.tree_util.tree_map(jnp.array, actor)
    target_critics = jax.tree_util.tree_map(jnp.array, critics)
    return ReBRACState(
        jnp.asarray(0, dtype=jnp.int32),
        actor,
        target_actor,
        critics,
        target_critics,
        init_adam(actor),
        init_adam(critics),
    )


def _validate_dataset(dataset: ReBRACDataset, config: ReBRACConfig) -> None:
    size = dataset.states.shape[0]
    if size <= 0 or dataset.states.shape != (size, config.state_dim):
        raise ValueError("states have the wrong shape")
    if dataset.next_states.shape != dataset.states.shape:
        raise ValueError("next_states must match states")
    if dataset.actions.shape != (size, config.action_dim):
        raise ValueError("actions have the wrong shape")
    if dataset.next_actions.shape != dataset.actions.shape:
        raise ValueError("next_actions must match actions")
    if dataset.rewards.shape != (size,) or dataset.dones.shape != (size,):
        raise ValueError("rewards and dones must be vectors")


def _polyak(target: Params, source: Params, rate: float) -> Params:
    return jax.tree_util.tree_map(
        lambda old, new: (1.0 - rate) * old + rate * new, target, source
    )


def rebrac_actor_loss(
    actor_params: Params,
    critic_params: Params,
    batch: ReBRACDataset,
    config: ReBRACConfig,
) -> tuple[Array, tuple[Array, Array, Array]]:
    actions = rebrac_actor(actor_params, batch.states)
    bc_penalty = jnp.sum(jnp.square(actions - batch.actions), axis=-1)
    q_values = jnp.min(rebrac_critics(critic_params, batch.states, actions), axis=0)
    q_scale = jnp.asarray(1.0, dtype=q_values.dtype)
    if config.normalize_q:
        q_scale = jax.lax.stop_gradient(
            1.0 / jnp.mean(jnp.abs(q_values))
        )
    loss = jnp.mean(config.actor_bc_coefficient * bc_penalty - q_scale * q_values)
    return loss, (jnp.mean(q_values), jnp.mean(bc_penalty), q_scale)


def rebrac_critic_loss(
    critic_params: Params,
    target_critic_params: Params,
    target_actor_params: Params,
    batch: ReBRACDataset,
    noise: Array,
    config: ReBRACConfig,
) -> tuple[Array, tuple[Array, Array, Array]]:
    if noise.shape != batch.actions.shape:
        raise ValueError("target-policy noise must match actions")
    next_actions = rebrac_actor(target_actor_params, batch.next_states)
    smoothing = jnp.clip(
        noise * config.policy_noise, -config.noise_clip, config.noise_clip
    )
    next_actions = jnp.clip(next_actions + smoothing, -1.0, 1.0)
    critic_bc = jnp.sum(jnp.square(next_actions - batch.next_actions), axis=-1)
    next_q = jnp.min(
        rebrac_critics(target_critic_params, batch.next_states, next_actions), axis=0
    )
    next_q = next_q - config.critic_bc_coefficient * critic_bc
    target_q = jax.lax.stop_gradient(
        batch.rewards + (1.0 - batch.dones) * config.discount * next_q
    )
    predictions = rebrac_critics(critic_params, batch.states, batch.actions)
    loss = jnp.sum(jnp.mean(jnp.square(predictions - target_q[None]), axis=1))
    return loss, (
        jnp.mean(jnp.min(predictions, axis=0)),
        jnp.mean(target_q),
        jnp.mean(critic_bc),
    )


def rebrac_update(
    state: ReBRACState,
    batch: ReBRACDataset,
    key: Array,
    config: ReBRACConfig,
) -> tuple[ReBRACState, ReBRACMetrics]:
    """Apply one released ReBRAC critic step and its delayed actor step."""

    _validate_dataset(batch, config)
    noise = jax.random.normal(key, batch.actions.shape, dtype=batch.actions.dtype)

    def critic_objective(params: Params):
        return rebrac_critic_loss(
            params,
            state.target_critics,
            state.target_actor,
            batch,
            noise,
            config,
        )

    (critic_loss, critic_aux), critic_gradients = jax.value_and_grad(
        critic_objective, has_aux=True
    )(state.critics)
    critics, critic_optimizer = adam_update(
        state.critics,
        critic_gradients,
        state.critic_optimizer,
        learning_rate=config.critic_learning_rate,
    )
    should_update = state.step % config.policy_frequency == 0

    def update_actor(_: None):
        def actor_objective(params: Params):
            return rebrac_actor_loss(params, critics, batch, config)

        (actor_loss, actor_aux), actor_gradients = jax.value_and_grad(
            actor_objective, has_aux=True
        )(state.actor)
        actor, actor_optimizer = adam_update(
            state.actor,
            actor_gradients,
            state.actor_optimizer,
            learning_rate=config.actor_learning_rate,
        )
        # Match the released functional update: the actor target receives the
        # pre-update actor, while the critic target receives the new critics.
        target_actor = _polyak(
            state.target_actor, state.actor, config.target_update_rate
        )
        target_critics = _polyak(
            state.target_critics, critics, config.target_update_rate
        )
        return (
            actor,
            actor_optimizer,
            target_actor,
            target_critics,
            actor_loss,
            actor_aux[1],
            actor_aux[2],
        )

    def skip_actor(_: None):
        zero = jnp.asarray(0.0, dtype=critic_loss.dtype)
        return (
            state.actor,
            state.actor_optimizer,
            state.target_actor,
            state.target_critics,
            zero,
            zero,
            jnp.asarray(1.0, dtype=critic_loss.dtype),
        )

    (
        actor,
        actor_optimizer,
        target_actor,
        target_critics,
        actor_loss,
        actor_bc,
        q_scale,
    ) = jax.lax.cond(should_update, update_actor, skip_actor, operand=None)
    updated = ReBRACState(
        state.step + 1,
        actor,
        target_actor,
        critics,
        target_critics,
        actor_optimizer,
        critic_optimizer,
    )
    return updated, ReBRACMetrics(
        critic_loss,
        actor_loss,
        critic_aux[0],
        critic_aux[1],
        actor_bc,
        critic_aux[2],
        q_scale,
        should_update.astype(jnp.float32),
    )


def train_rebrac_chunk(
    state: ReBRACState,
    dataset: ReBRACDataset,
    key: Array,
    *,
    updates: int,
    config: ReBRACConfig,
) -> tuple[ReBRACState, ReBRACMetrics]:
    """Run a resumable deterministic block of replay updates on device."""

    _validate_dataset(dataset, config)
    if not isinstance(updates, int) or isinstance(updates, bool) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    initial_metrics = ReBRACMetrics(*(jnp.asarray(0.0) for _ in range(8)))

    def body(_: int, carry: tuple[ReBRACState, ReBRACMetrics]):
        current, _ = carry
        update_key = jax.random.fold_in(key, current.step)
        batch_key, noise_key = jax.random.split(update_key)
        indices = jax.random.randint(
            batch_key,
            (config.batch_size,),
            minval=0,
            maxval=dataset.states.shape[0],
        )
        batch = jax.tree_util.tree_map(lambda value: value[indices], dataset)
        return rebrac_update(current, batch, noise_key, config)

    return jax.lax.fori_loop(0, updates, body, (state, initial_metrics))


def _flatten_observation(observation: Array, config: DreamerConfig) -> Array:
    value = preprocess_observation(observation)
    if value.shape[-len(config.observation_shape) :] != config.observation_shape:
        raise ValueError("FlowMPC observation has the wrong shape")
    return value.reshape((*value.shape[: -len(config.observation_shape)], -1))


def flowmpc_objective(
    actor_params: Params,
    critic_params: Params,
    world_model_params: Params,
    initial_state: RSSMState,
    current_observation: Array,
    noises: Array,
    dreamer_config: DreamerConfig,
    rebrac_config: ReBRACConfig,
    flowmpc_config: FlowMPCConfig,
) -> FlowMPCObjective:
    """Evaluate the paper's finite-horizon reward-plus-terminal-Q objective."""

    if dreamer_config.prior != "imf" or not dreamer_config.imf_trajectory_enabled:
        raise ValueError("FlowMPC adaptation requires trajectory iMF")
    if "reward_transition" not in world_model_params:
        raise ValueError("FlowMPC requires the published state-action reward head")
    if initial_state.deterministic.shape != (1, dreamer_config.deterministic_dim):
        raise ValueError("FlowMPC initial deterministic state must have batch one")
    if initial_state.stochastic.shape != (1, dreamer_config.stochastic_dim):
        raise ValueError("FlowMPC initial stochastic state must have batch one")
    expected_noise = (
        flowmpc_config.particles,
        flowmpc_config.horizon,
        dreamer_config.stochastic_dim,
    )
    if noises.shape != expected_noise:
        raise ValueError("FlowMPC noise bank has the wrong shape")
    flat_current = _flatten_observation(current_observation, dreamer_config)
    if flat_current.shape != (1, rebrac_config.state_dim):
        raise ValueError("FlowMPC current observation must have batch one")
    if rebrac_config.state_dim != dreamer_config.observation_dim:
        raise ValueError("ReBRAC state dimension must equal decoded observation size")
    if rebrac_config.action_dim != dreamer_config.action_dim:
        raise ValueError("ReBRAC and world-model action dimensions differ")
    particles = flowmpc_config.particles
    state = RSSMState(
        jnp.broadcast_to(initial_state.deterministic, (particles, dreamer_config.deterministic_dim)),
        jnp.broadcast_to(initial_state.stochastic, (particles, dreamer_config.stochastic_dim)),
    )
    observation = jnp.broadcast_to(
        current_observation, (particles, *dreamer_config.observation_shape)
    )

    def step(
        carry: tuple[RSSMState, Array], noise: Array
    ) -> tuple[tuple[RSSMState, Array], Array]:
        latent, observed = carry
        flat = _flatten_observation(observed, dreamer_config)
        action = rebrac_actor(actor_params, flat)
        reward = predict_state_action_reward_from_observation(
            world_model_params["reward_transition"],
            observed,
            action,
            dreamer_config,
        )
        deterministic = transition_deterministic(
            world_model_params, latent, action, dreamer_config
        )
        stochastic, _ = sample_prior(
            world_model_params,
            deterministic,
            None,
            dreamer_config,
            noise=noise,
        )
        next_latent = RSSMState(deterministic, stochastic)
        next_observation = decode(
            world_model_params, next_latent.feature, dreamer_config
        )
        return (next_latent, next_observation), reward

    (terminal_state, terminal_observation), rewards = jax.lax.scan(
        step, (state, observation), jnp.swapaxes(noises, 0, 1)
    )
    del terminal_state
    rewards = jnp.swapaxes(rewards, 0, 1)
    terminal_flat = _flatten_observation(terminal_observation, dreamer_config)
    terminal_action = rebrac_actor(actor_params, terminal_flat)
    terminal_q = jnp.min(
        rebrac_critics(critic_params, terminal_flat, terminal_action), axis=0
    )
    discounts = flowmpc_config.discount ** jnp.arange(
        flowmpc_config.horizon, dtype=rewards.dtype
    )
    per_particle_stage = jnp.sum(rewards * discounts[None], axis=1)
    stage_return = jnp.mean(per_particle_stage)
    terminal_value = (flowmpc_config.discount ** flowmpc_config.horizon) * jnp.mean(
        terminal_q
    )
    return FlowMPCObjective(
        stage_return + terminal_value,
        stage_return,
        terminal_value,
        rewards,
        terminal_q,
    )


def flowmpc_adapt_actor(
    actor_params: Params,
    critic_params: Params,
    world_model_params: Params,
    initial_state: RSSMState,
    current_observation: Array,
    noises: Array,
    dreamer_config: DreamerConfig,
    rebrac_config: ReBRACConfig,
    flowmpc_config: FlowMPCConfig,
) -> FlowMPCUpdate:
    """Take the paper's persistent direct gradient-ascent policy step(s)."""

    initial_actor = actor_params
    before = flowmpc_objective(
        actor_params,
        critic_params,
        world_model_params,
        initial_state,
        current_observation,
        noises,
        dreamer_config,
        rebrac_config,
        flowmpc_config,
    )
    gradient_norm = jnp.asarray(0.0, dtype=before.objective.dtype)
    for _ in range(flowmpc_config.inner_steps):
        def objective(params: Params) -> Array:
            return flowmpc_objective(
                params,
                critic_params,
                world_model_params,
                initial_state,
                current_observation,
                noises,
                dreamer_config,
                rebrac_config,
                flowmpc_config,
            ).objective

        gradients = jax.grad(objective)(actor_params)
        gradient_norm = tree_global_norm(gradients)
        actor_params = jax.tree_util.tree_map(
            lambda parameter, gradient: parameter + flowmpc_config.step_size * gradient,
            actor_params,
            gradients,
        )
    after = flowmpc_objective(
        actor_params,
        critic_params,
        world_model_params,
        initial_state,
        current_observation,
        noises,
        dreamer_config,
        rebrac_config,
        flowmpc_config,
    )
    parameter_delta = tree_global_norm(
        jax.tree_util.tree_map(
            lambda updated, original: updated - original,
            actor_params,
            initial_actor,
        )
    )
    return FlowMPCUpdate(
        actor_params,
        before.objective,
        after.objective,
        after.stage_return,
        after.terminal_value,
        gradient_norm,
        parameter_delta,
    )


jit_rebrac_update = partial(jax.jit, static_argnames=("config",))(rebrac_update)
jit_train_rebrac_chunk = partial(
    jax.jit, static_argnames=("updates", "config")
)(train_rebrac_chunk)
jit_flowmpc_objective = partial(
    jax.jit,
    static_argnames=("dreamer_config", "rebrac_config", "flowmpc_config"),
)(flowmpc_objective)
jit_flowmpc_adapt_actor = partial(
    jax.jit,
    static_argnames=("dreamer_config", "rebrac_config", "flowmpc_config"),
)(flowmpc_adapt_actor)


__all__ = [
    "FlowMPCConfig",
    "FlowMPCObjective",
    "FlowMPCUpdate",
    "ReBRACConfig",
    "ReBRACDataset",
    "ReBRACMetrics",
    "ReBRACState",
    "flowmpc_adapt_actor",
    "flowmpc_objective",
    "init_rebrac_actor",
    "init_rebrac_critics",
    "init_rebrac_state",
    "jit_flowmpc_adapt_actor",
    "jit_flowmpc_objective",
    "jit_rebrac_update",
    "jit_train_rebrac_chunk",
    "rebrac_actor",
    "rebrac_actor_loss",
    "rebrac_critic_loss",
    "rebrac_critics",
    "rebrac_update",
    "train_rebrac_chunk",
]
