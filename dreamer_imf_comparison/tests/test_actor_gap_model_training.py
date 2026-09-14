from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from dreamer_imf_compare.actor_gap_model_training import (
    APPROXIMATE_POLICY_DENSITY_LABEL,
    ENDPOINT_OBJECTIVE_LABEL,
    PROOF_CONSISTENT_CVAML_LABEL,
    ModelCellSpec,
    ModelTrainingConfig,
    _cvaml_prior_loss,
    build_endpoint_condition,
    build_matched_endpoint_examples,
    build_model_cell_specs,
    build_training_schedule,
    endpoint_condition_dim,
    init_model_train_state,
    materialize_sequence_batch,
    replay_model_cell_payload,
    require_canonical_training_schedule,
    schedule_sha256,
    train_endpoint_step,
    train_model_cell_payload,
    train_prior_step,
    trajectory_prior_compound_loss,
    validate_model_cell_spec,
)
from imf_dreamer_jax import (
    DreamerConfig,
    ReBRACConfig,
    init_rebrac_state,
)
from imf_dreamer_jax.world_model import init_world_model
from imf_dreamer_jax.world_model import init_transition_reward_head


def _config() -> DreamerConfig:
    return DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=4,
        stochastic_dim=2,
        embedding_dim=4,
        hidden_dim=8,
        prior="imf",
        burn_in=1,
        imf_trajectory_enabled=True,
        imf_trajectory_clean_probability=1.0,
        imf_trajectory_corrupted_probability=0.0,
        imf_trajectory_suffix_probability=0.0,
    )


def _objective(*, updates: int = 1) -> ModelTrainingConfig:
    return ModelTrainingConfig(
        updates=updates,
        batch_size=2,
        sequence_length=6,
        burn_in=1,
        maximum_chunk_horizon=2,
        endpoint_hidden_dim=8,
        endpoint_depth=1,
        tilt_eta=0.1,
        cvaml_samples=2,
        cvaml_scale=0.1,
    )


def _replay() -> dict[str, np.ndarray]:
    random = np.random.default_rng(17)
    episodes, steps = 3, 8
    observations = random.normal(size=(episodes, steps, 3)).astype(np.float32)
    actions = np.tanh(random.normal(size=(episodes, steps, 2))).astype(np.float32)
    rewards = random.normal(size=(episodes, steps)).astype(np.float32)
    continuations = np.ones((episodes, steps), dtype=np.float32)
    is_first = np.zeros((episodes, steps), dtype=np.bool_)
    is_first[:, 0] = True
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "continuations": continuations,
        "is_first": is_first,
        "train_episode_ids": np.asarray([0, 1], dtype=np.int32),
        "test_episode_ids": np.asarray([2], dtype=np.int32),
    }


def _world() -> dict:
    config = _config()
    world = init_world_model(config, jax.random.PRNGKey(3))
    world["reward_transition"] = init_transition_reward_head(
        config,
        jax.random.PRNGKey(4),
        observation_mean=jnp.zeros((3,)),
        observation_std=jnp.ones((3,)),
        hidden_dim=8,
    )
    return world


def _control():
    return init_rebrac_state(
        jax.random.PRNGKey(9),
        ReBRACConfig(state_dim=3, action_dim=2, hidden_dim=8),
    )


class ActorGapModelTrainingTest(unittest.TestCase):
    def test_canonical_matrix_has_exact_actor_specific_21_cells(self) -> None:
        cells = build_model_cell_specs((211, 223, 227), (311, 313))
        self.assertEqual(len(cells), 21)
        self.assertEqual([cell.index for cell in cells], list(range(21)))
        counts = {
            family: sum(cell.family == family for cell in cells)
            for family in {cell.family for cell in cells}
        }
        self.assertEqual(
            counts,
            {
                "uniform_prior": 3,
                "approximate_policy_tilt_prior": 6,
                "proof_consistent_cvaml_value_prior": 6,
                "endpoint_h1_imf": 3,
                "endpoint_anystep_imf": 3,
            },
        )
        for cell in cells:
            self.assertEqual(
                cell.actor_seed is not None,
                cell.family
                in {
                    "approximate_policy_tilt_prior",
                    "proof_consistent_cvaml_value_prior",
                },
            )
            validate_model_cell_spec(cell)

    def test_schedule_is_family_independent_and_horizons_balanced(self) -> None:
        objective = ModelTrainingConfig(
            updates=10,
            batch_size=2,
            sequence_length=6,
            burn_in=1,
            maximum_chunk_horizon=5,
        )
        first = build_training_schedule(
            _replay(),
            task="dmc_reacher_easy",
            world_model_seed=211,
            objective=objective,
        )
        second = build_training_schedule(
            _replay(),
            task="dmc_reacher_easy",
            world_model_seed=211,
            objective=objective,
        )
        other_world = build_training_schedule(
            _replay(),
            task="dmc_reacher_easy",
            world_model_seed=223,
            objective=objective,
        )
        self.assertEqual(schedule_sha256(first), schedule_sha256(second))
        self.assertNotEqual(schedule_sha256(first), schedule_sha256(other_world))
        np.testing.assert_array_equal(first["endpoint_horizons_h1"], 1)
        unique, counts = np.unique(
            first["endpoint_horizons_any_step"], return_counts=True
        )
        np.testing.assert_array_equal(unique, np.arange(1, 6))
        np.testing.assert_array_equal(counts, 2)

    def test_canonical_schedule_rejects_valid_but_rekeyed_rng_entry(self) -> None:
        objective = _objective(updates=2)
        replay = _replay()
        schedule = build_training_schedule(
            replay,
            task="dmc_reacher_easy",
            world_model_seed=211,
            objective=objective,
        )
        changed = {name: value.copy() for name, value in schedule.items()}
        changed["objective_keys"][0, 0] ^= np.uint32(1)
        with self.assertRaisesRegex(ValueError, "canonical derivation"):
            require_canonical_training_schedule(
                changed,
                replay,
                task="dmc_reacher_easy",
                world_model_seed=211,
                objective=objective,
            )
        canonical = require_canonical_training_schedule(
            schedule,
            replay,
            task="dmc_reacher_easy",
            world_model_seed=211,
            objective=objective,
        )
        self.assertEqual(schedule_sha256(canonical), schedule_sha256(schedule))

    def test_materialization_uses_only_frozen_training_episodes(self) -> None:
        objective = _objective()
        schedule = build_training_schedule(
            _replay(), task="task", world_model_seed=211, objective=objective
        )
        batch = materialize_sequence_batch(_replay(), schedule, 0, objective)
        self.assertEqual(batch["observations"].shape, (2, 6, 3))
        np.testing.assert_array_equal(np.asarray(batch["loss_mask"][:, 0]), 0.0)
        np.testing.assert_array_equal(np.asarray(batch["loss_mask"][:, 1:]), 1.0)

    def test_endpoint_alignment_and_fixed_condition_dimension(self) -> None:
        config = _config()
        objective = ModelTrainingConfig(
            updates=1,
            batch_size=1,
            sequence_length=6,
            burn_in=0,
            maximum_chunk_horizon=3,
            endpoint_hidden_dim=8,
            endpoint_depth=1,
        )
        observations = jnp.arange(18, dtype=jnp.float32).reshape((1, 6, 3))
        actions = jnp.arange(12, dtype=jnp.float32).reshape((1, 6, 2))
        batch = {
            "observations": observations,
            "actions": actions,
            "rewards": jnp.zeros((1, 6)),
            "continuations": jnp.ones((1, 6)),
            "is_first": jnp.zeros((1, 6), dtype=jnp.bool_),
            "loss_mask": jnp.ones((1, 6)),
        }
        target, condition, valid = build_matched_endpoint_examples(
            batch,
            jnp.asarray([0.0]),
            jnp.asarray(2),
            config,
            objective,
            observation_mean=jnp.zeros((3,)),
            observation_std=jnp.ones((3,)),
        )
        np.testing.assert_array_equal(
            np.asarray(target[0]), np.asarray(observations[0, 2])
        )
        self.assertEqual(condition.shape[-1], endpoint_condition_dim(config, 3))
        np.testing.assert_array_equal(
            np.asarray(condition[0, 3:7]), np.asarray(actions[0, 1:3]).reshape(-1)
        )
        np.testing.assert_array_equal(np.asarray(condition[0, -3:]), [1.0, 1.0, 0.0])
        inference_condition = build_endpoint_condition(
            observations[:, 0],
            jnp.concatenate((actions[:, 1:3], jnp.zeros((1, 1, 2))), axis=1),
            2,
            config,
            observation_mean=jnp.zeros((3,)),
            observation_std=jnp.ones((3,)),
        )
        np.testing.assert_array_equal(
            np.asarray(condition), np.asarray(inference_condition)
        )
        self.assertEqual(float(valid[0]), 1.0)
        reset_batch = dict(batch)
        reset_batch["is_first"] = batch["is_first"].at[0, 2].set(True)
        _, _, reset_valid = build_matched_endpoint_examples(
            reset_batch,
            jnp.asarray([0.0]),
            jnp.asarray(2),
            config,
            objective,
            observation_mean=jnp.zeros((3,)),
            observation_std=jnp.ones((3,)),
        )
        self.assertEqual(float(reset_valid[0]), 0.0)

    def test_endpoint_pair_has_identical_initialization_and_full_loss(self) -> None:
        objective = _objective()
        h1 = ModelCellSpec(0, "h1", "endpoint_h1_imf", 211, None)
        any_step = ModelCellSpec(1, "any", "endpoint_anystep_imf", 211, None)
        h1_state = init_model_train_state(
            h1, _world(), _config(), objective, task="task"
        )
        any_state = init_model_train_state(
            any_step, _world(), _config(), objective, task="task"
        )
        for left, right in zip(
            jax.tree_util.tree_leaves(h1_state.params),
            jax.tree_util.tree_leaves(any_state.params),
            strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        schedule = build_training_schedule(
            _replay(), task="task", world_model_seed=211, objective=objective
        )
        batch = materialize_sequence_batch(_replay(), schedule, 0, objective)
        updated, metrics = train_endpoint_step(
            h1_state,
            _world(),
            batch,
            jnp.asarray(schedule["endpoint_start_uniforms"][0]),
            jnp.asarray(1),
            jnp.asarray(schedule["objective_keys"][0]),
            _config(),
            objective,
        )
        self.assertEqual(int(updated.optimizer.step), 1)
        self.assertTrue(np.isfinite(float(metrics.total)))
        self.assertGreaterEqual(float(metrics.auxiliary), 0.0)
        self.assertGreater(float(metrics.accepted_fraction), 0.0)

    def test_prior_step_updates_only_returned_prior_from_fresh_adam(self) -> None:
        world = _world()
        before = [
            np.asarray(value).copy() for value in jax.tree_util.tree_leaves(world)
        ]
        objective = _objective()
        spec = ModelCellSpec(0, "uniform", "uniform_prior", 211, None)
        state = init_model_train_state(spec, world, _config(), objective, task="task")
        self.assertEqual(int(state.optimizer.step), 0)
        schedule = build_training_schedule(
            _replay(), task="task", world_model_seed=211, objective=objective
        )
        batch = materialize_sequence_batch(_replay(), schedule, 0, objective)
        updated, metrics = train_prior_step(
            state,
            world,
            batch,
            jnp.asarray(schedule["objective_keys"][0]),
            _config(),
            objective,
            family="uniform_prior",
        )
        self.assertEqual(int(updated.optimizer.step), 1)
        self.assertTrue(np.isfinite(float(metrics.total)))
        after = [np.asarray(value) for value in jax.tree_util.tree_leaves(world)]
        for left, right in zip(before, after, strict=True):
            np.testing.assert_array_equal(left, right)
        differences = [
            np.max(np.abs(np.asarray(left) - np.asarray(right)))
            for left, right in zip(
                jax.tree_util.tree_leaves(state.params),
                jax.tree_util.tree_leaves(updated.params),
                strict=True,
            )
        ]
        self.assertGreater(max(differences), 0.0)

    def test_actor_specific_tilt_and_cvaml_objectives_are_finite(self) -> None:
        objective = _objective()
        schedule = build_training_schedule(
            _replay(), task="task", world_model_seed=211, objective=objective
        )
        batch = materialize_sequence_batch(_replay(), schedule, 0, objective)
        world = _world()
        key = jnp.asarray(schedule["objective_keys"][0])
        tilt = trajectory_prior_compound_loss(
            world["prior"],
            world,
            batch,
            key,
            _config(),
            objective,
            family="approximate_policy_tilt_prior",
            control_state=_control(),
        )
        cvaml = trajectory_prior_compound_loss(
            world["prior"],
            world,
            batch,
            key,
            _config(),
            objective,
            family="proof_consistent_cvaml_value_prior",
            control_state=_control(),
        )
        self.assertTrue(np.isfinite(float(tilt.total)))
        self.assertGreater(float(tilt.tilt_effective_sample_size_fraction), 0.0)
        self.assertTrue(np.isfinite(float(cvaml.total)))
        self.assertAlmostEqual(
            float(cvaml.cvaml_corrected),
            float(cvaml.cvaml_uncorrected - cvaml.cvaml_variance_correction),
            places=5,
        )

    def test_cvaml_exactly_matches_deployed_planner_and_ignores_legacy_heads(
        self,
    ) -> None:
        from imf_dreamer_jax import rebrac_actor, rebrac_critics
        from imf_dreamer_jax.types import RSSMState
        from imf_dreamer_jax.world_model import (
            decode,
            observe_sequence,
            predict_state_action_reward_from_observation,
            sample_prior_with_nfe,
            transition_deterministic,
        )

        # The deployed controller discount is a fixed objective parameter; it
        # must not silently inherit a potentially different world-model value.
        config = replace(_config(), discount=0.25)
        objective = _objective()
        world = _world()
        control = _control()
        schedule = build_training_schedule(
            _replay(), task="task", world_model_seed=211, objective=objective
        )
        batch = materialize_sequence_batch(_replay(), schedule, 0, objective)
        sequence = observe_sequence(
            world,
            batch["observations"],
            batch["actions"],
            jax.random.PRNGKey(31),
            config,
            is_first=batch["is_first"],
        )
        cvaml_key = jax.random.PRNGKey(37)
        details = _cvaml_prior_loss(
            world["prior"],
            world,
            sequence,
            batch,
            cvaml_key,
            config,
            objective,
            control,
        )

        batch_size, steps = batch["actions"].shape[:2]
        pair_steps = steps - 1

        def flatten(value):
            return value.reshape((batch_size * pair_steps, value.shape[-1]))

        current = RSSMState(
            flatten(sequence.states.deterministic[:, :-1]),
            flatten(sequence.states.stochastic[:, :-1]),
        )
        action = flatten(batch["actions"][:, 1:])
        current_observation = batch["observations"][:, :-1].reshape(
            (batch_size * pair_steps, *config.observation_shape)
        )
        direct_reward = predict_state_action_reward_from_observation(
            world["reward_transition"], current_observation, action, config
        )
        deterministic = transition_deterministic(world, current, action, config)
        noise = jax.random.normal(
            cvaml_key,
            (objective.cvaml_samples, batch_size * pair_steps, config.stochastic_dim),
            dtype=deterministic.dtype,
        )

        def sampled_q(sample_noise):
            stochastic = sample_prior_with_nfe(
                world, deterministic, None, config, noise=sample_noise
            ).stochastic
            feature = jnp.concatenate((deterministic, stochastic), axis=-1)
            observation = decode(world, feature, config).reshape(
                (batch_size * pair_steps, -1)
            )
            next_action = rebrac_actor(control.actor, observation)
            return jnp.min(
                rebrac_critics(control.critics, observation, next_action), axis=0
            )

        expected_model_backup = direct_reward + objective.planner_discount * jnp.mean(
            jax.vmap(sampled_q)(noise), axis=0
        )
        real_observation = batch["observations"][:, 1:].reshape(
            (batch_size * pair_steps, -1)
        )
        real_action = rebrac_actor(control.actor, real_observation)
        real_q = jnp.min(
            rebrac_critics(control.critics, real_observation, real_action), axis=0
        )
        expected_real_target = batch["rewards"][:, 1:].reshape(-1) + (
            objective.planner_discount * real_q
        )
        np.testing.assert_allclose(
            np.asarray(details.model_backup),
            np.asarray(expected_model_backup),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(details.real_target),
            np.asarray(expected_real_target),
            rtol=1e-6,
            atol=1e-6,
        )

        altered = dict(world)
        altered["reward"] = jax.tree_util.tree_map(
            lambda value: value + 123.0, world["reward"]
        )
        altered["continuation"] = jax.tree_util.tree_map(
            lambda value: value - 321.0, world["continuation"]
        )
        altered_details = _cvaml_prior_loss(
            altered["prior"],
            altered,
            sequence,
            batch,
            cvaml_key,
            config,
            objective,
            control,
        )
        for name in ("loss", "model_backup", "real_target", "variance_correction"):
            np.testing.assert_array_equal(
                np.asarray(getattr(details, name)),
                np.asarray(getattr(altered_details, name)),
            )

    def test_payload_labels_isolation_and_strict_replay(self) -> None:
        objective = _objective()
        spec = ModelCellSpec(0, "uniform", "uniform_prior", 211, None)
        world = _world()
        payload = train_model_cell_payload(
            spec,
            world,
            _config(),
            _replay(),
            objective,
            task="task",
            jit=True,
        )
        self.assertEqual(payload["trainable_subtree"], "prior")
        self.assertEqual(payload["optimizer_initial_step"], 0)
        self.assertEqual(payload["optimizer_final_step"], 1)
        self.assertEqual(
            payload["source_world_model_sha256_before"],
            payload["source_world_model_sha256_after"],
        )
        replay = replay_model_cell_payload(
            payload,
            spec,
            world,
            _config(),
            _replay(),
            objective,
            task="task",
            jit=True,
        )
        self.assertTrue(replay["strict_deterministic_replay"])

        tilt_spec = ModelCellSpec(1, "tilt", "approximate_policy_tilt_prior", 211, 311)
        cvaml_spec = ModelCellSpec(
            2, "cvaml", "proof_consistent_cvaml_value_prior", 211, 311
        )
        endpoint_spec = ModelCellSpec(3, "endpoint", "endpoint_h1_imf", 211, None)
        self.assertEqual(
            train_model_cell_payload(
                tilt_spec,
                world,
                _config(),
                _replay(),
                objective,
                task="task",
                control_state=_control(),
                jit=False,
            )["policy_density_label"],
            APPROXIMATE_POLICY_DENSITY_LABEL,
        )
        self.assertEqual(
            train_model_cell_payload(
                cvaml_spec,
                world,
                _config(),
                _replay(),
                objective,
                task="task",
                control_state=_control(),
                jit=False,
            )["cvaml_estimator_label"],
            PROOF_CONSISTENT_CVAML_LABEL,
        )
        self.assertEqual(
            train_model_cell_payload(
                endpoint_spec,
                world,
                _config(),
                _replay(),
                objective,
                task="task",
                jit=False,
            )["endpoint_objective_label"],
            ENDPOINT_OBJECTIVE_LABEL,
        )

    def test_negative_contracts_reject_conflation(self) -> None:
        with self.assertRaisesRegex(ValueError, "actor-specific"):
            validate_model_cell_spec(ModelCellSpec(0, "bad", "uniform_prior", 211, 311))
        with self.assertRaisesRegex(ValueError, "control-state"):
            train_model_cell_payload(
                ModelCellSpec(0, "tilt", "approximate_policy_tilt_prior", 211, 311),
                _world(),
                _config(),
                _replay(),
                _objective(),
                task="task",
                control_state=None,
                jit=False,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            ModelTrainingConfig(endpoint_scale=0.0)


if __name__ == "__main__":
    unittest.main()
