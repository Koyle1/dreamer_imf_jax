from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.flowmpc import (
    FlowMPCConfig,
    ReBRACConfig,
    flowmpc_adapt_actor,
    flowmpc_objective,
    init_rebrac_state,
)
from imf_dreamer_jax.robust_flowmpc import (
    ActionSequenceDomain,
    ActionSequenceDomainConfig,
    ActionSequenceProposal,
    ActionSequenceSearchResult,
    ActionTrustRegionConfig,
    CEMActionSequenceConfig,
    GradientActionSequenceConfig,
    HeldOutAcceptanceConfig,
    PersistenceConfig,
    RelativePessimismConfig,
    action_sequence_from_proposal,
    action_sequence_objective_adapter,
    apply_held_out_noise_fallback,
    backtrack_actor_proposal,
    batched_action_sequence_objective_adapter,
    begin_actor_adaptation_step,
    cem_action_sequence_search,
    ensemble_relative_pessimism,
    ensemble_relative_risk_score,
    evaluate_flowmpc_held_out_acceptance,
    finish_actor_adaptation_step,
    gradient_action_sequence_search,
    held_out_noise_acceptance,
    init_reference_actor_state,
    jit_apply_held_out_noise_fallback,
    jit_backtrack_actor_proposal,
    jit_begin_actor_adaptation_step,
    jit_cem_action_sequence_search,
    jit_ensemble_relative_pessimism,
    jit_ensemble_relative_risk_score,
    jit_evaluate_flowmpc_held_out_acceptance,
    jit_finish_actor_adaptation_step,
    jit_gradient_action_sequence_search,
    jit_held_out_noise_acceptance,
    jit_make_action_sequence_domain,
    jit_measure_reference_action_drift,
    jit_project_reference_actions,
    jit_propose_flowmpc_actor,
    make_action_sequence_domain,
    make_action_sequence_proposal,
    measure_actor_reference_action_drift,
    measure_reference_action_drift,
    project_reference_actions,
    propose_flowmpc_actor,
)
from imf_dreamer_jax.types import RSSMState
from imf_dreamer_jax.world_model import (
    init_transition_reward_head,
    init_world_model,
)


def assert_tree_equal(test: unittest.TestCase, left: object, right: object) -> None:
    left_values, left_structure = jax.tree_util.tree_flatten(left)
    right_values, right_structure = jax.tree_util.tree_flatten(right)
    test.assertEqual(left_structure, right_structure)
    for left_value, right_value in zip(left_values, right_values, strict=True):
        np.testing.assert_array_equal(left_value, right_value)


def assert_tree_close(
    test: unittest.TestCase,
    left: object,
    right: object,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> None:
    left_values, left_structure = jax.tree_util.tree_flatten(left)
    right_values, right_structure = jax.tree_util.tree_flatten(right)
    test.assertEqual(left_structure, right_structure)
    for left_value, right_value in zip(left_values, right_values, strict=True):
        np.testing.assert_allclose(left_value, right_value, rtol=rtol, atol=atol)


def constant_actor(logits: jax.Array, *, state_dim: int = 3) -> dict[str, object]:
    """Build a small actor compatible with ``flowmpc.rebrac_actor``."""

    width = 2
    hidden: list[dict[str, jax.Array]] = []
    input_width = state_dim
    for _ in range(3):
        hidden.append(
            {
                "weight": jnp.zeros((input_width, width), dtype=jnp.float32),
                "bias": jnp.zeros((width,), dtype=jnp.float32),
            }
        )
        input_width = width
    return {
        "hidden": tuple(hidden),
        "output": {
            "weight": jnp.zeros((width, logits.shape[0]), dtype=jnp.float32),
            "bias": jnp.asarray(logits, dtype=jnp.float32),
        },
    }


class ConfigValidationTests(unittest.TestCase):
    def test_mode_labels_are_explicit_and_separate(self) -> None:
        self.assertEqual(ActionTrustRegionConfig().mode, "reference_action")
        self.assertEqual(PersistenceConfig("persistent").mode, "persistent")
        self.assertEqual(PersistenceConfig("reset").mode, "reset")
        self.assertEqual(HeldOutAcceptanceConfig().mode, "disabled")
        self.assertEqual(
            HeldOutAcceptanceConfig(mode="held_out_noise").mode,
            "held_out_noise",
        )
        self.assertEqual(
            ActionSequenceDomainConfig(2, 3, 0.5).mode,
            "reference_residual_action_sequence",
        )
        self.assertEqual(
            GradientActionSequenceConfig().mode, "gradient_action_sequence"
        )
        self.assertEqual(CEMActionSequenceConfig().mode, "cem_action_sequence")
        self.assertEqual(RelativePessimismConfig().mode, "mean_sd_risk")

    def test_invalid_trust_and_persistence_configs_fail_closed(self) -> None:
        invalid_trust = (
            {"anchor_mean_squared_budget": -1.0},
            {"current_max_absolute_budget": float("nan")},
            {"backtrack_ratio": 0.0},
            {"backtrack_ratio": 1.0},
            {"max_backtracks": -1},
            {"max_backtracks": True},
            {"feasibility_tolerance": -1.0},
            {"anchor_mean_squared_budget": True},
            {"anchor_mean_squared_budget": "small"},
            {"mode": "parameter_norm"},
        )
        for kwargs in invalid_trust:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActionTrustRegionConfig(**kwargs)  # type: ignore[arg-type]
        for mode in ("carry", "local"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                PersistenceConfig(mode=mode)  # type: ignore[arg-type]

    def test_invalid_acceptance_config_rejects_ambiguous_choices(self) -> None:
        invalid = (
            {"mode": "validation"},
            {"minimum_improvement": float("inf")},
            {"minimum_improvement": jnp.asarray(0.0)},
            {"baseline": "candidate"},
            {"fallback": "candidate"},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HeldOutAcceptanceConfig(**kwargs)  # type: ignore[arg-type]

    def test_invalid_search_configs_are_rejected(self) -> None:
        invalid_domains = (
            {"horizon": 0, "action_dim": 2, "residual_limit": 0.2},
            {"horizon": 2, "action_dim": 0, "residual_limit": 0.2},
            {"horizon": 2, "action_dim": 2, "residual_limit": 0.0},
            {"horizon": 2, "action_dim": 2, "residual_limit": None},
            {
                "horizon": 2,
                "action_dim": 2,
                "residual_limit": 0.2,
                "action_minimum": 1.0,
                "action_maximum": 1.0,
            },
        )
        for kwargs in invalid_domains:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActionSequenceDomainConfig(**kwargs)
        with self.assertRaises(ValueError):
            GradientActionSequenceConfig(iterations=0)
        with self.assertRaises(ValueError):
            GradientActionSequenceConfig(step_size=0.0)
        with self.assertRaises(ValueError):
            GradientActionSequenceConfig(step_size=True)
        with self.assertRaises(ValueError):
            GradientActionSequenceConfig(gradient_clip_norm=0.0)
        with self.assertRaises(ValueError):
            CEMActionSequenceConfig(population=3, elite_count=4)
        with self.assertRaises(ValueError):
            CEMActionSequenceConfig(initial_std=0.01, minimum_std=0.02)
        with self.assertRaises(ValueError):
            RelativePessimismConfig(risk_coefficient=-0.1)
        with self.assertRaises(ValueError):
            RelativePessimismConfig(minimum_members=1)


class ActionTrustRegionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_anchors = jnp.zeros((3, 2), dtype=jnp.float32)
        self.reference_current = jnp.zeros((2,), dtype=jnp.float32)

    def test_measurement_uses_declared_mean_square_and_current_max(self) -> None:
        candidate_anchors = jnp.asarray(
            [[0.2, 0.0], [0.0, -0.4], [0.1, 0.2]], dtype=jnp.float32
        )
        candidate_current = jnp.asarray([0.3, -0.1], dtype=jnp.float32)
        config = ActionTrustRegionConfig(
            anchor_mean_squared_budget=0.1,
            current_max_absolute_budget=0.25,
        )
        metrics = measure_reference_action_drift(
            self.reference_anchors,
            candidate_anchors,
            self.reference_current,
            candidate_current,
            config,
        )
        expected_msd = np.mean([0.2**2, 0.4**2, 0.1**2 + 0.2**2])
        np.testing.assert_allclose(
            metrics.anchor_mean_squared_action_drift, expected_msd, rtol=1e-6
        )
        np.testing.assert_allclose(
            metrics.current_max_absolute_action_drift, 0.3, rtol=1e-6
        )
        self.assertFalse(bool(metrics.feasible))
        jitted = jit_measure_reference_action_drift(
            self.reference_anchors,
            candidate_anchors,
            self.reference_current,
            candidate_current,
            config,
        )
        assert_tree_close(self, metrics, jitted)

    def test_exact_action_segment_projection_satisfies_both_budgets(self) -> None:
        proposed_anchors = jnp.tile(jnp.asarray([[0.4, 0.0]]), (3, 1))
        proposed_current = jnp.asarray([0.4, 0.0])
        config = ActionTrustRegionConfig(
            anchor_mean_squared_budget=0.04,
            current_max_absolute_budget=0.1,
            feasibility_tolerance=1e-6,
        )
        result = project_reference_actions(
            self.reference_anchors,
            proposed_anchors,
            self.reference_current,
            proposed_current,
            config,
        )
        # Anchor scale would be 0.5; the stricter current-action scale is 0.25.
        np.testing.assert_allclose(result.scale, 0.25, rtol=1e-6)
        np.testing.assert_allclose(result.current_action, [0.1, 0.0], rtol=1e-6)
        self.assertTrue(bool(result.metrics.feasible))
        jitted = jit_project_reference_actions(
            self.reference_anchors,
            proposed_anchors,
            self.reference_current,
            proposed_current,
            config,
        )
        assert_tree_close(self, result, jitted)

    def test_projection_handles_zero_drift_and_zero_budgets(self) -> None:
        config = ActionTrustRegionConfig(
            anchor_mean_squared_budget=0.0,
            current_max_absolute_budget=0.0,
            feasibility_tolerance=0.0,
        )
        unchanged = project_reference_actions(
            self.reference_anchors,
            self.reference_anchors,
            self.reference_current,
            self.reference_current,
            config,
        )
        np.testing.assert_allclose(unchanged.scale, 1.0, atol=0.0)
        self.assertTrue(bool(unchanged.metrics.feasible))
        moved = project_reference_actions(
            self.reference_anchors,
            jnp.ones_like(self.reference_anchors),
            self.reference_current,
            jnp.ones_like(self.reference_current),
            config,
        )
        np.testing.assert_allclose(moved.scale, 0.0, atol=0.0)
        np.testing.assert_array_equal(moved.anchor_actions, self.reference_anchors)
        np.testing.assert_array_equal(moved.current_action, self.reference_current)

    def test_actor_backtracking_selects_largest_checked_feasible_scale(self) -> None:
        reference = constant_actor(jnp.asarray([0.0, 0.0]))
        proposed = constant_actor(jnp.asarray([1.0, 0.0]))
        states = jnp.zeros((4, 3))
        current = jnp.zeros((1, 3))
        config = ActionTrustRegionConfig(
            anchor_mean_squared_budget=0.07,
            current_max_absolute_budget=0.25,
            backtrack_ratio=0.5,
            max_backtracks=3,
        )
        result = backtrack_actor_proposal(
            reference, reference, proposed, states, current, config
        )
        np.testing.assert_allclose(result.step_scale, 0.25, rtol=1e-6)
        self.assertEqual(int(result.backtracks), 2)
        self.assertTrue(bool(result.start_feasible))
        self.assertFalse(bool(result.used_reference_fallback))
        self.assertTrue(bool(result.metrics.feasible))
        jitted = jit_backtrack_actor_proposal(
            reference, reference, proposed, states, current, config
        )
        assert_tree_close(self, result, jitted)

    def test_backtracking_exposes_infeasible_start_and_reference_fallback(self) -> None:
        reference = constant_actor(jnp.asarray([0.0, 0.0]))
        infeasible = constant_actor(jnp.asarray([1.0, 0.0]))
        config = ActionTrustRegionConfig(
            anchor_mean_squared_budget=0.0,
            current_max_absolute_budget=0.0,
            max_backtracks=0,
            feasibility_tolerance=0.0,
        )
        result = backtrack_actor_proposal(
            reference,
            infeasible,
            infeasible,
            jnp.zeros((2, 3)),
            jnp.zeros((1, 3)),
            config,
        )
        self.assertFalse(bool(result.start_feasible))
        self.assertTrue(bool(result.used_reference_fallback))
        self.assertTrue(bool(result.metrics.feasible))
        assert_tree_equal(self, result.actor, reference)

    def test_actor_measurement_is_on_fixed_states_and_reference_is_frozen(self) -> None:
        reference = constant_actor(jnp.asarray([0.0, 0.0]))
        candidate = constant_actor(jnp.asarray([0.2, -0.1]))
        config = ActionTrustRegionConfig(
            anchor_mean_squared_budget=1.0,
            current_max_absolute_budget=1.0,
        )
        metrics = measure_actor_reference_action_drift(
            reference,
            candidate,
            jnp.zeros((3, 3)),
            jnp.ones((1, 3)),
            config,
        )
        expected = np.tanh(0.2) ** 2 + np.tanh(-0.1) ** 2
        np.testing.assert_allclose(
            metrics.anchor_mean_squared_action_drift, expected, rtol=1e-6
        )
        self.assertTrue(bool(metrics.feasible))

        reference_gradient = jax.grad(
            lambda reference_actions: measure_reference_action_drift(
                reference_actions,
                jnp.full((3, 2), 0.1),
                jnp.zeros((2,)),
                jnp.full((2,), 0.1),
                config,
            ).anchor_mean_squared_action_drift
        )(jnp.zeros((3, 2)))
        np.testing.assert_array_equal(
            reference_gradient, jnp.zeros_like(reference_gradient)
        )

    def test_action_and_state_shape_errors_are_not_silently_broadcast(self) -> None:
        config = ActionTrustRegionConfig()
        with self.assertRaises(ValueError):
            measure_reference_action_drift(
                jnp.zeros((0, 2)),
                jnp.zeros((0, 2)),
                jnp.zeros((2,)),
                jnp.zeros((2,)),
                config,
            )
        with self.assertRaises(ValueError):
            measure_reference_action_drift(
                jnp.zeros((2, 2)),
                jnp.full((2, 2), jnp.nan),
                jnp.zeros((2,)),
                jnp.zeros((2,)),
                config,
            )
        with self.assertRaises(ValueError):
            measure_reference_action_drift(
                jnp.zeros((2, 2)),
                jnp.zeros((2, 3)),
                jnp.zeros((2,)),
                jnp.zeros((2,)),
                config,
            )
        with self.assertRaises(ValueError):
            measure_actor_reference_action_drift(
                constant_actor(jnp.zeros((2,))),
                constant_actor(jnp.zeros((2,))),
                jnp.zeros((2, 3)),
                jnp.zeros((3,)),
                config,
            )
        with self.assertRaises(ValueError):
            measure_actor_reference_action_drift(
                constant_actor(jnp.zeros((2,))),
                constant_actor(jnp.zeros((2,))),
                jnp.full((2, 3), jnp.nan),
                jnp.zeros((1, 3)),
                config,
            )


class PersistenceAndAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = {"value": jnp.asarray([0.0, 1.0])}
        self.selected = {"value": jnp.asarray([2.0, 3.0])}
        self.state = init_reference_actor_state(self.reference)

    def test_persistent_mode_carries_selected_actor_without_mutating_reference(
        self,
    ) -> None:
        config = PersistenceConfig("persistent")
        assert_tree_equal(
            self,
            begin_actor_adaptation_step(self.state, config),
            self.reference,
        )
        next_state = finish_actor_adaptation_step(self.state, self.selected, config)
        assert_tree_equal(self, next_state.frozen_reference_actor, self.reference)
        assert_tree_equal(
            self, begin_actor_adaptation_step(next_state, config), self.selected
        )
        assert_tree_equal(
            self,
            jit_begin_actor_adaptation_step(next_state, config),
            self.selected,
        )
        assert_tree_equal(
            self,
            jit_finish_actor_adaptation_step(self.state, self.selected, config),
            next_state,
        )
        self.assertEqual(int(next_state.environment_step), 1)

    def test_reset_mode_starts_from_and_restores_frozen_reference(self) -> None:
        persistent_state = finish_actor_adaptation_step(
            self.state, self.selected, PersistenceConfig("persistent")
        )
        reset = PersistenceConfig("reset")
        assert_tree_equal(
            self,
            begin_actor_adaptation_step(persistent_state, reset),
            self.reference,
        )
        reset_state = finish_actor_adaptation_step(
            persistent_state, self.selected, reset
        )
        assert_tree_equal(self, reset_state.carried_actor, self.reference)
        assert_tree_equal(self, reset_state.frozen_reference_actor, self.reference)
        self.assertEqual(int(reset_state.environment_step), 2)

    def test_disabled_acceptance_does_not_sneak_in_a_gain_gate(self) -> None:
        config = HeldOutAcceptanceConfig(mode="disabled", minimum_improvement=100.0)
        metrics = held_out_noise_acceptance(-5.0, 10.0, config)
        self.assertTrue(bool(metrics.accepted))
        self.assertFalse(bool(metrics.used_fallback))
        chosen = apply_held_out_noise_fallback(
            self.selected,
            self.reference,
            {"value": jnp.asarray([-2.0, -2.0])},
            metrics,
            config,
        )
        assert_tree_equal(self, chosen, self.selected)

    def test_held_out_rejection_uses_independently_selected_fallback(self) -> None:
        frozen = {"value": jnp.asarray([-2.0, -2.0])}
        config = HeldOutAcceptanceConfig(
            mode="held_out_noise",
            minimum_improvement=0.1,
            baseline="adaptation_start",
            fallback="frozen_reference",
        )
        metrics = held_out_noise_acceptance(1.0, 0.95, config)
        self.assertFalse(bool(metrics.accepted))
        self.assertTrue(bool(metrics.used_fallback))
        np.testing.assert_allclose(metrics.improvement, 0.05, rtol=1e-6)
        chosen = apply_held_out_noise_fallback(
            self.selected, self.reference, frozen, metrics, config
        )
        assert_tree_equal(self, chosen, frozen)
        assert_tree_equal(
            self,
            jit_apply_held_out_noise_fallback(
                self.selected,
                self.reference,
                frozen,
                metrics,
                config,
            ),
            frozen,
        )
        assert_tree_close(
            self,
            metrics,
            jit_held_out_noise_acceptance(1.0, 0.95, config),
        )

        start_fallback = HeldOutAcceptanceConfig(
            mode="held_out_noise",
            minimum_improvement=0.1,
            fallback="adaptation_start",
        )
        chosen = apply_held_out_noise_fallback(
            self.selected,
            self.reference,
            frozen,
            held_out_noise_acceptance(1.0, 0.95, start_fallback),
            start_fallback,
        )
        assert_tree_equal(self, chosen, self.reference)

    def test_held_out_acceptance_rejects_nonfinite_and_nonscalar_scores(self) -> None:
        config = HeldOutAcceptanceConfig(mode="held_out_noise")
        self.assertFalse(bool(held_out_noise_acceptance(jnp.nan, 0.0, config).accepted))
        self.assertFalse(bool(held_out_noise_acceptance(1.0, jnp.inf, config).accepted))
        with self.assertRaises(ValueError):
            held_out_noise_acceptance(jnp.ones((2,)), 0.0, config)


class ActionSequenceSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.domain_config = ActionSequenceDomainConfig(
            horizon=2,
            action_dim=2,
            residual_limit=0.8,
        )
        self.domain = make_action_sequence_domain(
            jnp.zeros((2, 2), dtype=jnp.float32), self.domain_config
        )
        self.initial = make_action_sequence_proposal(
            jnp.zeros((2, 2), dtype=jnp.float32), self.domain
        )
        self.target = jnp.asarray([[0.35, -0.25], [0.2, 0.4]])

    def objective(self, actions: jax.Array) -> jax.Array:
        return -jnp.sum(jnp.square(actions - self.target))

    def test_domain_intersects_residual_and_action_bounds_exactly(self) -> None:
        config = ActionSequenceDomainConfig(2, 2, 0.5)
        reference = jnp.asarray([[0.8, -0.8], [0.0, 0.0]])
        domain = make_action_sequence_domain(reference, config)
        jitted_domain = jit_make_action_sequence_domain(reference, config)
        assert_tree_equal(self, domain, jitted_domain)
        np.testing.assert_allclose(domain.residual_maximum[0], [0.2, 0.5], rtol=1e-6)
        np.testing.assert_allclose(domain.residual_minimum[0], [-0.5, -0.2], rtol=1e-6)
        proposal = make_action_sequence_proposal(
            jnp.asarray([[2.0, -2.0], [-2.0, 2.0]]), domain
        )
        actions = action_sequence_from_proposal(proposal, domain)
        np.testing.assert_allclose(actions, [[1.0, -1.0], [-0.5, 0.5]])
        self.assertTrue(bool(jnp.all(actions >= -1.0)))
        self.assertTrue(bool(jnp.all(actions <= 1.0)))

        integer_domain = make_action_sequence_domain(
            jnp.zeros((2, 2), dtype=jnp.int32), config
        )
        self.assertTrue(
            jnp.issubdtype(integer_domain.reference_actions.dtype, jnp.floating)
        )

    def test_gradient_and_cem_share_proposal_domain_and_result_schemas(self) -> None:
        gradient_config = GradientActionSequenceConfig(
            iterations=12, step_size=0.15, gradient_clip_norm=2.0
        )
        cem_config = CEMActionSequenceConfig(
            iterations=5,
            population=96,
            elite_count=12,
            initial_std=0.5,
            minimum_std=0.01,
            maximum_std=0.8,
        )
        bank = jax.random.normal(
            jax.random.key(41),
            (
                cem_config.iterations,
                cem_config.population,
                *self.domain.reference_actions.shape,
            ),
        )
        gradient = gradient_action_sequence_search(
            self.objective, self.domain, self.initial, gradient_config
        )
        cem = cem_action_sequence_search(
            self.objective, self.domain, self.initial, bank, cem_config
        )
        self.assertIsInstance(gradient, ActionSequenceSearchResult)
        self.assertIsInstance(cem, ActionSequenceSearchResult)
        self.assertIsInstance(gradient.proposal, ActionSequenceProposal)
        self.assertIsInstance(cem.proposal, ActionSequenceProposal)
        self.assertEqual(gradient.action_sequence.shape, (2, 2))
        self.assertEqual(cem.action_sequence.shape, (2, 2))
        self.assertGreater(float(gradient.objective_improvement), 0.38)
        self.assertGreater(float(cem.objective_improvement), 0.35)
        self.assertEqual(
            int(gradient.objective_evaluations), gradient_config.iterations + 1
        )
        self.assertEqual(
            int(cem.objective_evaluations),
            2 + cem_config.iterations * cem_config.population,
        )
        for result in (gradient, cem):
            self.assertTrue(
                bool(jnp.all(result.proposal.residuals >= self.domain.residual_minimum))
            )
            self.assertTrue(
                bool(jnp.all(result.proposal.residuals <= self.domain.residual_maximum))
            )

        gradient_jit = jit_gradient_action_sequence_search(
            self.objective, self.domain, self.initial, gradient_config
        )
        cem_jit = jit_cem_action_sequence_search(
            self.objective, self.domain, self.initial, bank, cem_config
        )
        assert_tree_close(self, gradient, gradient_jit)
        assert_tree_close(self, cem, cem_jit)

    def test_cem_is_deterministic_and_does_not_use_objective_gradients(self) -> None:
        target = self.target

        def stopped_objective(actions: jax.Array) -> jax.Array:
            actions = jax.lax.stop_gradient(actions)
            return -jnp.sum(jnp.square(actions - target))

        gradient_config = GradientActionSequenceConfig(iterations=5, step_size=0.2)
        cem_config = CEMActionSequenceConfig(
            iterations=5,
            population=128,
            elite_count=16,
            initial_std=0.5,
            minimum_std=0.01,
            maximum_std=0.8,
        )
        bank = jax.random.normal(
            jax.random.key(42),
            (
                cem_config.iterations,
                cem_config.population,
                *self.domain.reference_actions.shape,
            ),
        )
        gradient = gradient_action_sequence_search(
            stopped_objective, self.domain, self.initial, gradient_config
        )
        first = cem_action_sequence_search(
            stopped_objective, self.domain, self.initial, bank, cem_config
        )
        second = cem_action_sequence_search(
            stopped_objective, self.domain, self.initial, bank, cem_config
        )
        np.testing.assert_allclose(gradient.objective_improvement, 0.0, atol=0.0)
        self.assertGreater(float(first.objective_improvement), 0.35)
        assert_tree_equal(self, first, second)

    def test_objective_adapters_use_one_scalar_contract(self) -> None:
        score = action_sequence_objective_adapter(
            self.initial, self.domain, self.objective
        )
        batch = ActionSequenceProposal(
            jnp.stack((self.initial.residuals, self.initial.residuals + 0.1))
        )
        scores = batched_action_sequence_objective_adapter(
            batch, self.domain, self.objective
        )
        np.testing.assert_allclose(scores[0], score)
        with self.assertRaises(ValueError):
            action_sequence_objective_adapter(
                self.initial,
                self.domain,
                lambda actions: actions[:, 0],
            )
        with self.assertRaises(ValueError):
            batched_action_sequence_objective_adapter(
                ActionSequenceProposal(jnp.zeros((2, 2))),
                self.domain,
                self.objective,
            )

    def test_malformed_domains_proposals_and_cem_banks_fail(self) -> None:
        with self.assertRaises(ValueError):
            make_action_sequence_domain(jnp.zeros((2, 3)), self.domain_config)
        with self.assertRaises(ValueError):
            make_action_sequence_domain(jnp.full((2, 2), 1.1), self.domain_config)
        with self.assertRaises(ValueError):
            make_action_sequence_domain(jnp.full((2, 2), jnp.nan), self.domain_config)
        with self.assertRaises(ValueError):
            make_action_sequence_proposal(jnp.zeros((2, 3)), self.domain)
        with self.assertRaises(ValueError):
            make_action_sequence_proposal(jnp.full((2, 2), jnp.nan), self.domain)
        malformed = ActionSequenceDomain(
            self.domain.reference_actions,
            jnp.zeros((1, 2)),
            self.domain.residual_maximum,
        )
        with self.assertRaises(ValueError):
            make_action_sequence_proposal(jnp.zeros((2, 2)), malformed)
        inconsistent = ActionSequenceDomain(
            self.domain.reference_actions,
            jnp.ones((2, 2)),
            jnp.zeros((2, 2)),
        )
        with self.assertRaises(ValueError):
            make_action_sequence_proposal(jnp.zeros((2, 2)), inconsistent)
        with self.assertRaises(ValueError):
            batched_action_sequence_objective_adapter(
                ActionSequenceProposal(jnp.zeros((0, 2, 2))),
                self.domain,
                self.objective,
            )
        config = CEMActionSequenceConfig(iterations=2, population=8, elite_count=2)
        with self.assertRaises(ValueError):
            cem_action_sequence_search(
                self.objective,
                self.domain,
                self.initial,
                jnp.zeros((2, 7, 2, 2)),
                config,
            )
        with self.assertRaises(ValueError):
            cem_action_sequence_search(
                self.objective,
                self.domain,
                self.initial,
                jnp.full((2, 8, 2, 2), jnp.nan),
                config,
            )


class RelativePessimismTests(unittest.TestCase):
    def test_mean_sd_is_named_risk_score_and_uses_paired_improvements(self) -> None:
        candidate = jnp.asarray([2.0, 4.0, 3.0])
        reference = jnp.asarray([1.0, 1.0, 1.0])
        config = RelativePessimismConfig(risk_coefficient=1.0)
        metrics = ensemble_relative_pessimism(candidate, reference, config)
        improvements = np.asarray([1.0, 3.0, 2.0])
        np.testing.assert_allclose(metrics.member_improvements, improvements)
        np.testing.assert_allclose(metrics.mean_improvement, improvements.mean())
        np.testing.assert_allclose(
            metrics.improvement_standard_deviation, improvements.std()
        )
        np.testing.assert_allclose(
            metrics.risk_score,
            improvements.mean() - improvements.std(),
            rtol=1e-6,
        )
        self.assertFalse(hasattr(metrics, "lower_confidence_bound"))
        assert_tree_close(
            self,
            metrics,
            jit_ensemble_relative_pessimism(candidate, reference, config),
        )

    def test_pairing_cancels_shared_member_bias_and_freezes_reference_gradient(
        self,
    ) -> None:
        candidate = jnp.asarray([2.0, 4.0, 3.0])
        reference = jnp.asarray([1.0, 1.0, 1.0])
        shared_bias = jnp.asarray([100.0, -50.0, 7.0])
        config = RelativePessimismConfig(risk_coefficient=0.7)
        original = ensemble_relative_risk_score(candidate, reference, config)
        shifted = ensemble_relative_risk_score(
            candidate + shared_bias, reference + shared_bias, config
        )
        np.testing.assert_allclose(original, shifted, rtol=1e-6, atol=1e-6)
        reference_gradient = jax.grad(
            lambda frozen: ensemble_relative_risk_score(candidate, frozen, config)
        )(reference)
        np.testing.assert_array_equal(reference_gradient, jnp.zeros_like(reference))
        candidate_gradient = jax.grad(
            lambda proposed: ensemble_relative_risk_score(proposed, reference, config)
        )(candidate)
        self.assertTrue(bool(jnp.all(jnp.isfinite(candidate_gradient))))
        self.assertGreater(float(jnp.linalg.norm(candidate_gradient)), 0.0)

    def test_relative_pessimism_rejects_unpaired_or_tiny_ensembles(self) -> None:
        config = RelativePessimismConfig()
        with self.assertRaises(ValueError):
            ensemble_relative_pessimism(jnp.zeros((2, 2)), jnp.zeros((2, 2)), config)
        with self.assertRaises(ValueError):
            ensemble_relative_pessimism(jnp.zeros((3,)), jnp.zeros((2,)), config)
        with self.assertRaises(ValueError):
            ensemble_relative_pessimism(jnp.zeros((1,)), jnp.zeros((1,)), config)
        with self.assertRaises(ValueError):
            ensemble_relative_pessimism(
                jnp.asarray([0.0, jnp.nan]), jnp.zeros((2,)), config
            )

    def test_integer_inputs_are_promoted_and_scalar_risk_jit_matches(self) -> None:
        config = RelativePessimismConfig(risk_coefficient=0.5)
        candidate = jnp.asarray([2, 4, 3], dtype=jnp.int32)
        reference = jnp.asarray([1, 1, 1], dtype=jnp.int32)
        score = ensemble_relative_risk_score(candidate, reference, config)
        self.assertTrue(jnp.issubdtype(score.dtype, jnp.floating))
        np.testing.assert_allclose(
            score,
            jit_ensemble_relative_risk_score(candidate, reference, config),
            rtol=1e-6,
        )


class ExistingFlowMPCAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dreamer = DreamerConfig(
            observation_shape=(3,),
            action_dim=2,
            deterministic_dim=5,
            stochastic_dim=3,
            embedding_dim=4,
            hidden_dim=16,
            prior="imf",
            imf_trajectory_enabled=True,
            reward_loss="mse",
            reward_prediction_horizon=0,
        )
        self.rebrac = ReBRACConfig(state_dim=3, action_dim=2)
        world = init_world_model(self.dreamer, jax.random.key(50))
        reward = init_transition_reward_head(
            self.dreamer, jax.random.key(51), hidden_dim=16
        )
        self.world = {**world, "reward_transition": reward}
        self.policy = init_rebrac_state(jax.random.key(52), self.rebrac)
        self.state = init_reference_actor_state(self.policy.actor)
        self.initial_latent = RSSMState(
            jnp.zeros((1, self.dreamer.deterministic_dim)),
            jnp.zeros((1, self.dreamer.stochastic_dim)),
        )
        self.observation = jnp.zeros((1, *self.dreamer.observation_shape))
        self.flow = FlowMPCConfig(horizon=1, particles=4, inner_steps=1, step_size=1e-7)
        self.proposal_noises = jax.random.normal(
            jax.random.key(53),
            (self.flow.particles, self.flow.horizon, self.dreamer.stochastic_dim),
        )
        self.held_out_noises = jax.random.normal(
            jax.random.key(54),
            (self.flow.particles, self.flow.horizon, self.dreamer.stochastic_dim),
        )

    def test_proposal_adapter_reuses_existing_flowmpc_update_exactly(self) -> None:
        persistence = PersistenceConfig("persistent")
        adapted = propose_flowmpc_actor(
            self.state,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.proposal_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
            persistence,
        )
        direct = flowmpc_adapt_actor(
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.proposal_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
        )
        assert_tree_equal(self, adapted, direct)
        jitted = jit_propose_flowmpc_actor(
            self.state,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.proposal_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
            persistence,
        )
        assert_tree_close(self, adapted, jitted)

    def test_held_out_evaluator_uses_configured_baseline_on_supplied_bank(self) -> None:
        candidate = flowmpc_adapt_actor(
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.proposal_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
        ).actor
        altered_start = jax.tree_util.tree_map(
            lambda value: value + 1e-5, self.policy.actor
        )
        # Infinite thresholds are intentionally invalid configuration.
        with self.assertRaises(ValueError):
            HeldOutAcceptanceConfig(mode="held_out_noise", minimum_improvement=-jnp.inf)
        config = HeldOutAcceptanceConfig(
            mode="held_out_noise",
            minimum_improvement=-1e6,
            baseline="frozen_reference",
            fallback="adaptation_start",
        )
        metrics = evaluate_flowmpc_held_out_acceptance(
            candidate,
            altered_start,
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.held_out_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
            config,
        )
        expected_candidate = flowmpc_objective(
            candidate,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.held_out_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
        ).objective
        expected_reference = flowmpc_objective(
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.held_out_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
        ).objective
        np.testing.assert_allclose(metrics.candidate_objective, expected_candidate)
        np.testing.assert_allclose(metrics.baseline_objective, expected_reference)
        self.assertTrue(bool(metrics.accepted))
        jitted = jit_evaluate_flowmpc_held_out_acceptance(
            candidate,
            altered_start,
            self.policy.actor,
            self.policy.critics,
            self.world,
            self.initial_latent,
            self.observation,
            self.held_out_noises,
            self.dreamer,
            self.rebrac,
            self.flow,
            config,
        )
        # XLA may fuse the long stochastic rollout differently from eager JAX;
        # this tolerance is far below the controller's decision margin.
        self.assertEqual(bool(metrics.accepted), bool(jitted.accepted))
        self.assertEqual(bool(metrics.used_fallback), bool(jitted.used_fallback))
        assert_tree_close(self, metrics, jitted, rtol=1e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
