from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DreamerConfig,
    create_agent,
    imagine,
    initial_state,
    jit_imagine,
    train_actor_critic,
)
from imf_dreamer_jax.world_model import (
    predict_continuation_logits,
    predict_transition_reward,
)


def _model_reward(params, previous_feature, action, next_feature, config):
    return predict_transition_reward(
        params, previous_feature, action, next_feature, config
    )


def _model_continuation(params, previous_feature, action, next_feature, config):
    del previous_feature, action, config
    return jax.nn.sigmoid(predict_continuation_logits(params, next_feature))


def _constant_reward(params, previous_feature, action, next_feature, config):
    del params, previous_feature, next_feature, config
    return jnp.full(action.shape[:-1], 7.0, dtype=action.dtype)


def _constant_continuation(params, previous_feature, action, next_feature, config):
    del params, previous_feature, next_feature, config
    return jnp.full(action.shape[:-1], 0.25, dtype=action.dtype)


def _positive_action_reward(params, previous_feature, action, next_feature, config):
    del params, previous_feature, next_feature, config
    return action[..., 0]


def _negative_action_reward(params, previous_feature, action, next_feature, config):
    del params, previous_feature, next_feature, config
    return -action[..., 0]


def _unit_continuation(params, previous_feature, action, next_feature, config):
    del params, previous_feature, next_feature, config
    return jnp.ones(action.shape[:-1], dtype=action.dtype)


def _zero_continuation(params, previous_feature, action, next_feature, config):
    del params, previous_feature, next_feature, config
    return jnp.zeros(action.shape[:-1], dtype=action.dtype)


class ActorFailureDiagnosticOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DreamerConfig(
            observation_shape=(3,),
            action_dim=1,
            deterministic_dim=5,
            stochastic_dim=3,
            embedding_dim=4,
            hidden_dim=8,
            prior="gaussian",
            reward_min=0.0,
            reward_max=1.0,
            reward_initial_value=0.5,
            imagination_horizon=4,
            actor_gradient="pmpo",
            critic_bins=7,
            critic_symlog_min=-5.0,
            critic_symlog_max=5.0,
        )
        self.state = create_agent(self.config, jax.random.PRNGKey(7))
        start = initial_state(self.config, 6)
        self.start = type(start)(
            start.deterministic + 0.1,
            start.stochastic - 0.2,
        )

    def test_explicit_model_functions_are_exactly_default(self) -> None:
        key = jax.random.PRNGKey(11)
        default = imagine(self.state.params, self.start, key, self.config)
        explicit = imagine(
            self.state.params,
            self.start,
            key,
            self.config,
            reward_fn=_model_reward,
            continuation_fn=_model_continuation,
        )
        for left, right in zip(default, explicit, strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_jitted_overrides_replace_both_signals(self) -> None:
        imagined = jit_imagine(
            self.state.params,
            self.start,
            jax.random.PRNGKey(13),
            self.config,
            reward_fn=_constant_reward,
            continuation_fn=_constant_continuation,
        )
        np.testing.assert_array_equal(
            np.asarray(imagined.rewards), np.full((6, 4), 7.0, np.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(imagined.continuations), np.full((6, 4), 0.25, np.float32)
        )

    def test_reward_override_changes_actor_and_critic_updates(self) -> None:
        key = jax.random.PRNGKey(17)
        positive, _ = train_actor_critic(
            self.state,
            self.start,
            key,
            self.config,
            reward_fn=_positive_action_reward,
            continuation_fn=_unit_continuation,
        )
        negative, _ = train_actor_critic(
            self.state,
            self.start,
            key,
            self.config,
            reward_fn=_negative_action_reward,
            continuation_fn=_unit_continuation,
        )
        positive_actor = jax.tree_util.tree_leaves(positive.params.actor)
        negative_actor = jax.tree_util.tree_leaves(negative.params.actor)
        positive_critic = jax.tree_util.tree_leaves(positive.params.critic)
        negative_critic = jax.tree_util.tree_leaves(negative.params.critic)
        self.assertTrue(any(
            not np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(positive_actor, negative_actor, strict=True)
        ))
        self.assertTrue(any(
            not np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(positive_critic, negative_critic, strict=True)
        ))

    def test_continuation_override_changes_actor_and_critic_updates(self) -> None:
        key = jax.random.PRNGKey(19)
        continuing, _ = train_actor_critic(
            self.state,
            self.start,
            key,
            self.config,
            reward_fn=_positive_action_reward,
            continuation_fn=_unit_continuation,
        )
        terminal, _ = train_actor_critic(
            self.state,
            self.start,
            key,
            self.config,
            reward_fn=_positive_action_reward,
            continuation_fn=_zero_continuation,
        )
        for continuing_tree, terminal_tree in (
            (continuing.params.actor, terminal.params.actor),
            (continuing.params.critic, terminal.params.critic),
        ):
            self.assertTrue(any(
                not np.array_equal(np.asarray(left), np.asarray(right))
                for left, right in zip(
                    jax.tree_util.tree_leaves(continuing_tree),
                    jax.tree_util.tree_leaves(terminal_tree),
                    strict=True,
                )
            ))


if __name__ == "__main__":
    unittest.main()
