from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dreamer_imf_compare import reward_head_mtp_study as study
from imf_dreamer_jax import (
    DreamerConfig,
    create_agent,
    init_running_rms,
    jit_train_reward_head,
    reconfigure_reward_head,
    reward_context_predictions,
)
from imf_dreamer_jax.nn import tree_global_norm


def config() -> DreamerConfig:
    return DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=5,
        stochastic_dim=3,
        embedding_dim=4,
        hidden_dim=12,
        prior="imf",
        imf_trajectory_enabled=True,
        reward_output_init_scale=1.0,
    )


def tree_delta(left: object, right: object) -> float:
    return float(tree_global_norm(jax.tree_util.tree_map(lambda x, y: y - x, left, right)))


def batch(cfg: DreamerConfig) -> dict[str, jax.Array]:
    batch_size, steps, candidates, rollout_steps = 2, 10, 3, 5
    result = {
        "observations": jax.random.normal(
            jax.random.key(1), (batch_size, steps, *cfg.observation_shape)
        ),
        "actions": 0.2 * jax.random.normal(
            jax.random.key(2), (batch_size, steps, cfg.action_dim)
        ),
        "rewards": jax.random.uniform(jax.random.key(3), (batch_size, steps)),
        "continuations": jnp.ones((batch_size, steps)),
        "is_first": jnp.zeros((batch_size, steps), dtype=jnp.bool_).at[:, 0].set(True),
        "loss_mask": jnp.ones((batch_size, steps)),
        "advantage_action_sequences": 0.2 * jax.random.normal(
            jax.random.key(4),
            (batch_size, steps, candidates, rollout_steps, cfg.action_dim),
        ),
        "advantage_target_returns": jax.random.normal(
            jax.random.key(5), (batch_size, steps, candidates, 3)
        ),
        "advantage_mask": jnp.ones((batch_size, steps, 3)),
    }
    return result


class RewardHeadMTPStudyTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        print("REWARD_HEAD_MTP_STUDY_VERIFIED")

    def test_four_arms_form_the_registered_two_by_two_factorial(self) -> None:
        self.assertEqual(len(study.ARMS), 4)
        factors = {
            arm: (study.arm_uses_mtp(arm), study.arm_uses_advantage(arm))
            for arm in study.ARMS
        }
        self.assertEqual(set(factors.values()), {(False, False), (True, False), (False, True), (True, True)})
        for arm in study.ARMS:
            objective = study.objective_for_arm(arm)
            self.assertEqual(objective.advantage_scale > 0.0, study.arm_uses_advantage(arm))
            self.assertGreater(objective.corrupted_scale, 0.0)
            self.assertGreater(objective.generated_scale, 0.0)

    def test_reward_only_update_preserves_every_nonreward_parameter_and_moment(self) -> None:
        old_config = config()
        state = create_agent(old_config, jax.random.key(10))
        new_config = study.reward_config_for_arm(
            old_config, "twohot_mtp_advantage", reward_bins=51
        )
        state = reconfigure_reward_head(
            state, old_config, new_config, jax.random.key(11)
        )
        before = state
        data = batch(new_config)
        updated, losses, rms = jit_train_reward_head(
            state,
            data,
            jax.random.key(12),
            new_config,
            study.objective_for_arm("twohot_mtp_advantage"),
            init_running_rms(),
            advantage_batch=data,
        )
        self.assertTrue(np.isfinite(float(losses.total)))
        self.assertGreater(float(rms.count), 0.0)
        self.assertGreater(
            tree_delta(before.params.world_model["reward"], updated.params.world_model["reward"]),
            0.0,
        )
        for name in before.params.world_model:
            if name == "reward":
                continue
            self.assertEqual(
                tree_delta(before.params.world_model[name], updated.params.world_model[name]), 0.0
            )
            self.assertEqual(
                tree_delta(
                    before.model_optimizer.first_moment[name],
                    updated.model_optimizer.first_moment[name],
                ),
                0.0,
            )
            self.assertEqual(
                tree_delta(
                    before.model_optimizer.second_moment[name],
                    updated.model_optimizer.second_moment[name],
                ),
                0.0,
            )
        predictions = reward_context_predictions(
            updated.params.world_model,
            data,
            jax.random.key(13),
            new_config,
            study.objective_for_arm("twohot_mtp_advantage"),
        )
        for values in (predictions.posterior, predictions.corrupted, predictions.generated):
            self.assertEqual(values.shape, (2, 10, 9))
            self.assertTrue(np.isfinite(np.asarray(values)).all())


if __name__ == "__main__":
    unittest.main()
