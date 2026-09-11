from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax import (
    DreamerConfig,
    create_agent,
    init_world_model,
    jit_observe_sequence,
    jit_train_world_model,
    jit_world_model_loss,
)


def tearDownModule() -> None:
    print("WORLD_MODEL_TESTS_OK")


def small_config(prior: str, overshooting_horizon: int = 1) -> DreamerConfig:
    return DreamerConfig(
        observation_shape=(5,),
        action_dim=2,
        deterministic_dim=7,
        stochastic_dim=3,
        embedding_dim=6,
        hidden_dim=8,
        prior=prior,
        overshooting_horizon=overshooting_horizon,
        imagination_horizon=3,
    )


def make_batch(config: DreamerConfig) -> dict[str, jax.Array]:
    return {
        "observations": jax.random.normal(jax.random.key(20), (2, 5, *config.observation_shape)),
        "actions": jnp.tanh(jax.random.normal(jax.random.key(21), (2, 5, config.action_dim))),
        "rewards": jax.random.normal(jax.random.key(22), (2, 5)),
        "continuations": jnp.asarray([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=jnp.float32),
    }


class WorldModelTests(unittest.TestCase):
    def test_all_three_ablation_configs_compile_and_update(self) -> None:
        configurations = (
            small_config("gaussian"),
            small_config("imf"),
            small_config("imf", overshooting_horizon=3),
        )
        self.assertEqual(
            [config.method_name for config in configurations],
            ["gaussian_rssm", "imf_rssm_one_step", "imf_rssm_multistep"],
        )
        for index, config in enumerate(configurations):
            with self.subTest(method=config.method_name):
                batch = make_batch(config)
                state = create_agent(config, jax.random.key(30 + index))
                before = state.params.world_model["reward"]["layers"][-1]["weight"]
                new_state, losses = jit_train_world_model(
                    state, batch, jax.random.key(40 + index), config
                )
                values = jnp.stack(tuple(losses))
                self.assertTrue(np.isfinite(np.asarray(values)).all())
                after = new_state.params.world_model["reward"]["layers"][-1]["weight"]
                self.assertGreater(float(jnp.max(jnp.abs(after - before))), 0.0)
                self.assertEqual(int(new_state.model_optimizer.step), 1)
                if config.method_name == "imf_rssm_multistep":
                    self.assertGreaterEqual(float(losses.overshooting), 0.0)

    def test_observe_sequence_shapes_and_key_determinism(self) -> None:
        config = small_config("imf")
        params = init_world_model(config, jax.random.key(50))
        batch = make_batch(config)
        first = jit_observe_sequence(
            params, batch["observations"], batch["actions"], jax.random.key(51), config
        )
        second = jit_observe_sequence(
            params, batch["observations"], batch["actions"], jax.random.key(51), config
        )
        self.assertEqual(first.states.deterministic.shape, (2, 5, 7))
        self.assertEqual(first.states.stochastic.shape, (2, 5, 3))
        self.assertIsNone(first.prior_mean)
        np.testing.assert_array_equal(first.states.stochastic, second.states.stochastic)

    def test_pixel_shaped_uint8_observations_are_supported(self) -> None:
        config = DreamerConfig(
            observation_shape=(3, 4, 4),
            action_dim=2,
            deterministic_dim=6,
            stochastic_dim=3,
            embedding_dim=5,
            hidden_dim=8,
            prior="gaussian",
        )
        params = init_world_model(config, jax.random.key(60))
        batch = {
            "observations": jax.random.randint(
                jax.random.key(61), (2, 3, 3, 4, 4), 0, 256, dtype=jnp.uint8
            ),
            "actions": jnp.zeros((2, 3, 2), dtype=jnp.float32),
            "rewards": jnp.zeros((2, 3), dtype=jnp.float32),
            "continuations": jnp.ones((2, 3), dtype=jnp.float32),
        }
        losses = jit_world_model_loss(params, batch, jax.random.key(62), config)
        self.assertTrue(np.isfinite(np.asarray(jnp.stack(tuple(losses)))).all())


if __name__ == "__main__":
    unittest.main()

