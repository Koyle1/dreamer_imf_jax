from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DreamerConfig,
    PlannerConfig,
    RSSMState,
    SequenceReplayBuffer,
    create_agent,
    energy_score,
    evaluate_action_sequences,
    initial_state,
    jit_act,
    jit_cem_plan,
    jit_imagine,
    jit_sample_prior_ensemble,
    jit_train_actor_critic,
    lambda_returns,
    load_checkpoint,
    predictive_moments,
    save_checkpoint,
    stack_ensemble,
)


def tearDownModule() -> None:
    print("AGENT_TESTS_OK")


def config() -> DreamerConfig:
    return DreamerConfig(
        observation_shape=(4,),
        action_dim=2,
        deterministic_dim=6,
        stochastic_dim=3,
        embedding_dim=5,
        hidden_dim=8,
        prior="imf",
        imagination_horizon=3,
    )


class AgentTests(unittest.TestCase):
    def test_act_imagination_and_actor_critic_update(self) -> None:
        cfg = config()
        state = create_agent(cfg, jax.random.key(70))
        belief = initial_state(cfg, 2)
        action, belief = jit_act(
            state.params,
            jnp.zeros((2, 4)),
            jnp.zeros((2, 2)),
            belief,
            jax.random.key(71),
            cfg,
            deterministic=True,
        )
        self.assertEqual(action.shape, (2, 2))
        self.assertTrue(bool(jnp.all(jnp.abs(action) <= 1)))
        imagined = jit_imagine(state.params, belief, jax.random.key(72), cfg)
        self.assertEqual(imagined.features.shape, (2, 4, cfg.feature_dim))
        new_state, metrics = jit_train_actor_critic(
            state, belief, jax.random.key(73), cfg
        )
        self.assertTrue(np.isfinite(np.asarray(jnp.stack(tuple(metrics)))).all())
        self.assertEqual(int(new_state.actor_optimizer.step), 1)
        self.assertEqual(int(new_state.critic_optimizer.step), 1)

    def test_lambda_returns_respect_terminal_continuations(self) -> None:
        rewards = jnp.asarray([[1.0, 2.0]])
        values = jnp.asarray([[0.0, 10.0, 20.0]])
        continuations = jnp.asarray([[1.0, 0.0]])
        result = lambda_returns(rewards, values, continuations, discount=1.0, lambda_=1.0)
        np.testing.assert_allclose(result, [[3.0, 2.0]], rtol=1e-6)

    def test_episode_safe_replay_is_reproducible(self) -> None:
        first = SequenceReplayBuffer(20, (2,), 1, seed=5)
        second = SequenceReplayBuffer(20, (2,), 1, seed=5)
        for replay in (first, second):
            replay.append([0, 0], [0], 0, 1, is_first=True)
            replay.append([1, 0], [0], 1, 0)
            replay.append([10, 0], [0], 10, 1, is_first=True)
            replay.append([11, 0], [0], 11, 1)
            replay.append([12, 0], [0], 12, 0)
        sample_a = first.sample(8, 2)
        sample_b = second.sample(8, 2)
        np.testing.assert_array_equal(sample_a["rewards"], sample_b["rewards"])
        differences = np.asarray(sample_a["rewards"][:, 1] - sample_a["rewards"][:, 0])
        np.testing.assert_array_equal(differences, np.ones_like(differences))

    def test_uncertainty_decomposition_and_energy_score(self) -> None:
        cfg = config()
        agents = [create_agent(cfg, jax.random.key(seed)) for seed in (80, 81)]
        ensemble = stack_ensemble([agent.params.world_model for agent in agents])
        samples = jit_sample_prior_ensemble(
            ensemble,
            jnp.zeros((4, cfg.deterministic_dim)),
            jax.random.key(82),
            cfg,
            noise_samples=4,
        )
        moments = predictive_moments(samples)
        np.testing.assert_allclose(
            moments.predictive_variance,
            moments.aleatoric_variance + moments.epistemic_variance,
            rtol=2e-5,
            atol=2e-5,
        )
        score = energy_score(samples, jnp.zeros((4, cfg.stochastic_dim)))
        self.assertEqual(score.shape, (4,))
        self.assertTrue(np.isfinite(np.asarray(score)).all())

    def test_common_noise_planner_and_checkpoint_round_trip(self) -> None:
        cfg = config()
        agents = [create_agent(cfg, jax.random.key(seed)) for seed in (90, 91)]
        ensemble = stack_ensemble([agent.params.world_model for agent in agents])
        beliefs = RSSMState(
            jnp.zeros((2, cfg.deterministic_dim)),
            jnp.zeros((2, cfg.stochastic_dim)),
        )
        planner = PlannerConfig(
            horizon=2,
            population=8,
            elite_count=2,
            iterations=1,
            noise_samples=2,
            terminal_value=True,
        )
        duplicate_actions = jnp.zeros((3, planner.horizon, cfg.action_dim))
        scores, _, _, _ = evaluate_action_sequences(
            ensemble,
            agents[0].params.critic,
            beliefs,
            duplicate_actions,
            jax.random.key(92),
            cfg,
            planner,
        )
        np.testing.assert_array_equal(scores, jnp.repeat(scores[:1], 3))
        plan = jit_cem_plan(
            ensemble,
            agents[0].params.critic,
            beliefs,
            jax.random.key(93),
            cfg,
            planner,
        )
        self.assertEqual(plan.action.shape, (cfg.action_dim,))
        self.assertTrue(bool(jnp.all(jnp.abs(plan.action) <= 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.ckpt"
            save_checkpoint(path, agents[0], cfg, metadata={"step": 7})
            restored, restored_config, metadata = load_checkpoint(path)
            self.assertEqual(restored_config, cfg)
            self.assertEqual(metadata, {"step": 7})
            for left, right in zip(
                jax.tree_util.tree_leaves(agents[0]),
                jax.tree_util.tree_leaves(restored),
                strict=True,
            ):
                np.testing.assert_array_equal(left, right)

    def test_source_tree_has_no_torch_runtime_dependency(self) -> None:
        source = Path(__file__).parents[1] / "src" / "imf_dreamer_jax"
        combined = "\n".join(path.read_text() for path in source.rglob("*.py"))
        self.assertNotIn("import torch", combined)
        self.assertNotIn("from torch", combined)


if __name__ == "__main__":
    unittest.main()

