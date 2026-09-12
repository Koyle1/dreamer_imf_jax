"""Adversarial tests for the frozen matched-objective study contract."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from dreamer_imf_compare.matched_objective_protocol import (
    ACTOR_SEEDS,
    ARM_ORDER,
    BUDGET_TRACK_ORDER,
    EXPLORATORY_ABLATIONS,
    NFE_FRONTIER,
    PILOT_ACTOR_SEEDS,
    PILOT_TASKS,
    PILOT_WORLD_MODEL_SEEDS,
    PRIMARY_METRICS,
    REQUIRED_ARTIFACTS,
    TASKS,
    WORLD_MODEL_SEEDS,
    evaluate_superiority,
    expected_cells,
    protocol_digest,
    read_matched_objective_protocol,
    validate_matched_objective_protocol,
)


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT / "matched_objective_protocol.json"


def favorable_intervals() -> dict[str, dict[str, dict[str, float]]]:
    return {
        track: {
            metric: {"lower": 0.01, "upper": 0.20}
            for metric in PRIMARY_METRICS
        }
        for track in BUDGET_TRACK_ORDER
    }


class MatchedObjectiveProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_matched_objective_protocol(PROTOCOL_PATH)

    def assert_invalid(self, mutation) -> None:
        candidate = deepcopy(self.protocol)
        mutation(candidate)
        with self.assertRaises(ValueError):
            validate_matched_objective_protocol(candidate)

    def test_protocol_loads_and_digest_is_canonical(self) -> None:
        validate_matched_objective_protocol(self.protocol)
        digest = protocol_digest(self.protocol)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        reordered = dict(reversed(list(self.protocol.items())))
        self.assertEqual(protocol_digest(reordered), digest)
        changed = deepcopy(self.protocol)
        changed["study_name"] += " (registered copy)"
        self.assertNotEqual(protocol_digest(changed), digest)

    def test_exactly_two_confirmatory_arms_and_two_exploratory_controls(self) -> None:
        self.assertEqual(tuple(self.protocol["arm_order"]), ARM_ORDER)
        self.assertEqual(tuple(self.protocol["arms"]), ARM_ORDER)
        ablations = self.protocol["exploratory_ablations"]
        self.assertFalse(ablations["may_support_primary_superiority_claim"])
        self.assertEqual(tuple(ablations["ablations"]), EXPLORATORY_ABLATIONS)
        self.assert_invalid(lambda p: p["arms"].update({"third_confirmatory_arm": {}}))
        self.assert_invalid(
            lambda p: p["exploratory_ablations"].__setitem__(
                "may_support_primary_superiority_claim", True
            )
        )

    def test_shortcut_equation_7_fidelity_mutations_are_rejected(self) -> None:
        mutations = {
            "not_x_prediction": lambda p: p["arms"]["shortcut_forcing"].__setitem__(
                "parameterization", "velocity"
            ),
            "shared_token_time": lambda p: p["arms"]["shortcut_forcing"].__setitem__(
                "independent_per_token_signal_levels", False
            ),
            "non_power_of_two_kmax": lambda p: p["arms"]["shortcut_forcing"]["schedule"].__setitem__(
                "training_k_max", 6
            ),
            "pretend_kmax_disclosed": lambda p: p["arms"]["shortcut_forcing"]["schedule"].__setitem__(
                "training_k_max_paper_disclosed", True
            ),
            "wrong_grid": lambda p: p["arms"]["shortcut_forcing"]["schedule"].__setitem__(
                "signal_level", "tau~Uniform(0,1)"
            ),
            "one_half_step": lambda p: p["arms"]["shortcut_forcing"]["bootstrap"].__setitem__(
                "two_half_steps", False
            ),
            "no_stop_gradient": lambda p: p["arms"]["shortcut_forcing"]["bootstrap"].__setitem__(
                "target_stop_gradient", False
            ),
            "wrong_space": lambda p: p["arms"]["shortcut_forcing"]["loss"].__setitem__(
                "coordinate_space", "velocity_space"
            ),
            "wrong_scaling": lambda p: p["arms"]["shortcut_forcing"]["loss"].__setitem__(
                "bootstrap_scale", "1"
            ),
            "wrong_ramp": lambda p: p["arms"]["shortcut_forcing"]["loss"]["ramp"].__setitem__(
                "slope", 1.0
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.assert_invalid(mutation)

    def test_trajectory_objective_and_negative_control_mutations_are_rejected(self) -> None:
        mutations = {
            "shared_time": lambda p: p["arms"]["trajectory_imf"].__setitem__(
                "independent_per_token_signal_levels", False
            ),
            "full_jvp": lambda p: p["arms"]["trajectory_imf"].__setitem__(
                "fixed_context_partial_jvp", False
            ),
            "different_noise_path": lambda p: p["arms"]["trajectory_imf"].__setitem__(
                "same_noise_path_for_query_history_pair", False
            ),
            "unequal_context_mixture": lambda p: p["arms"]["trajectory_imf"]["context_patterns"].update(
                {"clean_context": 0.5, "corrupted_context": 0.25, "future_suffix": 0.25}
            ),
            "legacy_overshooting": lambda p: p["arms"]["trajectory_imf"]["separate_legacy_losses"].__setitem__(
                "overshooting_loss", True
            ),
            "wrong_diagonal_control": lambda p: p["exploratory_ablations"]["ablations"][
                "joint_diagonal_context_jvp_negative_control"
            ].__setitem__("change_only", "delete_cross_token_outputs"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.assert_invalid(mutation)

    def test_randomness_is_paired_without_falsely_matching_schedule_distributions(self) -> None:
        randomness = self.protocol["shared_world_model"]["objective_randomness"]
        self.assertTrue(randomness["same_base_gaussian_samples_within_paired_minibatch"])
        self.assertTrue(randomness["paired_canonical_uniforms_by_token"])
        self.assertFalse(randomness["identical_schedule_distribution_required"])
        self.assertNotEqual(randomness["shortcut_schedule"], randomness["trajectory_schedule"])
        self.assert_invalid(
            lambda p: p["shared_world_model"]["objective_randomness"].__setitem__(
                "identical_schedule_distribution_required", True
            )
        )

    def test_shared_controls_report_active_compute_and_optimizer_treatment(self) -> None:
        matching = self.protocol["parameter_matching"]
        self.assertTrue(matching["report_total_parameters"])
        self.assertTrue(matching["report_active_parameters"])
        self.assertTrue(matching["inactive_objective_heads_excluded_from_active_count"])
        self.assertTrue(matching["unused_arm_specific_heads_not_instantiated"])
        self.assertTrue(matching["report_compiled_forward_flops"])
        self.assertTrue(matching["report_compiled_forward_and_backward_train_flops"])
        optimization = self.protocol["shared_world_model"]["optimization"]
        self.assertEqual(optimization["optimizer"], "library_custom_global_norm_clipped_Adam")
        self.assertEqual(optimization["weight_decay"], 0.0)
        self.assertTrue(optimization["world_model_ema_enabled"])
        self.assertEqual(
            optimization["shortcut_bootstrap_teacher"],
            "stop_gradient_of_ema_parameters_decay_0.999",
        )
        self.assert_invalid(lambda p: p["parameter_matching"].__setitem__("report_active_parameters", False))
        self.assert_invalid(
            lambda p: p["parameter_matching"].__setitem__(
                "inactive_objective_heads_excluded_from_active_count", False
            )
        )
        self.assert_invalid(
            lambda p: p["shared_world_model"]["optimization"].__setitem__(
                "world_model_ema_enabled", False
            )
        )

    def test_hpo_is_equal_budget_pilot_only_and_confirmatory_sealed(self) -> None:
        policy = self.protocol["hyperparameter_policy"]
        self.assertEqual(tuple(policy["selection_tasks"]), PILOT_TASKS)
        self.assertEqual(tuple(policy["selection_world_model_seeds"]), PILOT_WORLD_MODEL_SEEDS)
        self.assertEqual(
            tuple(policy["selection_actor_seeds_nested_within_world_model_seed"]),
            PILOT_ACTOR_SEEDS,
        )
        self.assertTrue(set(policy["selection_tasks"]).isdisjoint(TASKS))
        self.assertTrue(set(policy["selection_world_model_seeds"]).isdisjoint(WORLD_MODEL_SEEDS))
        self.assertTrue(set(policy["selection_actor_seeds_nested_within_world_model_seed"]).isdisjoint(ACTOR_SEEDS))
        self.assertEqual(policy["candidate_evaluations_per_arm"], 12)
        self.assertEqual(
            len(policy["search"]["shortcut_forcing"]["training_k_max"])
            * len(policy["search"]["shortcut_forcing"]["objective_loss_scale"]),
            12,
        )
        self.assertEqual(
            len(policy["search"]["trajectory_imf"]["imf_boundary_fraction"])
            * len(policy["search"]["trajectory_imf"]["objective_loss_scale"]),
            12,
        )
        self.assertEqual(
            policy["selection_aggregation"]["actor_return"],
            "arithmetic_mean_over_evaluation_episodes_within_actor_seed_then_arithmetic_mean_over_2_nested_actor_seeds_within_task_x_world_model_seed_then_interquartile_mean_over_6_task_x_world_model_seed_units",
        )
        self.assert_invalid(
            lambda p: p["hyperparameter_policy"]["selection_aggregation"].__setitem__(
                "rollout_auc", "arithmetic_mean"
            )
        )
        self.assertTrue(policy["confirmatory_data_checkpoints_and_outcomes_sealed_until_selection_frozen"])
        self.assert_invalid(
            lambda p: p["hyperparameter_policy"].__setitem__("candidate_evaluations_per_arm", 3)
        )
        self.assert_invalid(
            lambda p: p["hyperparameter_policy"]["search"]["shortcut_forcing"].__setitem__(
                "objective_loss_scale", [0.5, 1.0]
            )
        )
        self.assert_invalid(
            lambda p: p["hyperparameter_policy"].__setitem__(
                "confirmatory_data_checkpoints_and_outcomes_sealed_until_selection_frozen", False
            )
        )

    def test_both_compute_tracks_are_mandatory_and_have_distinct_roles(self) -> None:
        self.assertEqual(tuple(self.protocol["budget_track_order"]), BUDGET_TRACK_ORDER)
        tracks = self.protocol["budget_tracks"]
        self.assertEqual(tracks["equal_updates"]["claim_role"], "primary_objective_isolation_track")
        self.assertEqual(tracks["equal_compiler_flops"]["claim_role"], "required_compute_robustness_track")
        self.assertIn("forward_and_backward", tracks["equal_compiler_flops"]["cost_source"])
        self.assert_invalid(lambda p: p["budget_tracks"].pop("equal_compiler_flops"))
        self.assert_invalid(
            lambda p: p["budget_tracks"]["equal_compiler_flops"].__setitem__(
                "allocation_frozen_before_outcome_evaluation", False
            )
        )
        self.assert_invalid(
            lambda p: p["budget_tracks"]["equal_compiler_flops"].__setitem__(
                "relative_tolerance", 0.1
            )
        )

    def test_six_task_confirmatory_suite_and_nested_seed_cell_counts(self) -> None:
        confirmatory = self.protocol["profiles"]["confirmatory"]
        self.assertEqual(tuple(confirmatory["tasks"]), TASKS)
        self.assertEqual(tuple(confirmatory["world_model_seeds"]), WORLD_MODEL_SEEDS)
        self.assertEqual(tuple(confirmatory["actor_seeds_nested_within_world_model_seed"]), ACTOR_SEEDS)
        cells = expected_cells(self.protocol, "confirmatory")
        self.assertEqual(len(cells["datasets"]), 6 * 10)
        self.assertEqual(len(cells["world_models"]), 2 * 6 * 10 * 2)
        self.assertEqual(len(cells["rollout_evaluations"]), 2 * 6 * 10 * 2 * 3)
        self.assertEqual(len(cells["actors"]), 2 * 6 * 10 * 2 * 3)
        keys = {
            (cell["budget_track"], cell["task"], cell["world_model_seed"], cell["arm"], cell["actor_seed"])
            for cell in cells["actors"]
        }
        self.assertEqual(len(keys), len(cells["actors"]))
        self.assert_invalid(lambda p: p["profiles"]["confirmatory"]["tasks"].pop())
        self.assert_invalid(
            lambda p: p["statistics"].__setitem__(
                "actor_seeds_are_nested_not_independent_replicates", False
            )
        )

    def test_nfe_rollout_calibration_and_return_normalization_are_frozen(self) -> None:
        evaluation = self.protocol["evaluation"]
        self.assertEqual(tuple(evaluation["nfe_frontier"]), NFE_FRONTIER)
        self.assertEqual(evaluation["primary_nfe"], {"shortcut_forcing": 4, "trajectory_imf": 1})
        self.assertEqual(
            evaluation["actor_metric"]["normalization"],
            "dmc_episodic_return_divided_by_1000",
        )
        self.assertIn("training_split_standard_deviation", evaluation["rollout_metric"]["normalizer"])
        secondary = set(evaluation["secondary_metrics"])
        self.assertTrue(
            {
                "energy_score",
                "central_90_percent_interval_coverage",
                "central_90_percent_interval_calibration_absolute_error_at_each_horizon",
            }
            <= secondary
        )
        self.assertTrue(
            {
                "compiler_reported_forward_flops",
                "compiler_reported_forward_and_backward_train_flops",
                "compiler_reported_inference_flops_per_transition",
            }
            <= secondary
        )
        self.assert_invalid(lambda p: p["evaluation"].__setitem__("nfe_frontier", [1, 4]))
        self.assert_invalid(
            lambda p: p["evaluation"]["actor_metric"].__setitem__("normalization", "raw_return")
        )

    def test_both_diagnostics_are_required_and_stochastic_repetition_is_real(self) -> None:
        diagnostics = self.protocol["diagnostics"]
        self.assertEqual(len(diagnostics["required_before_confirmatory_interpretation"]), 2)
        stochastic = diagnostics["genuinely_stochastic_transition_diagnostic"]
        self.assertGreaterEqual(stochastic["repeated_futures_per_identical_condition"], 64)
        self.assertIn("exogenous_Bernoulli_branch", stochastic["mechanisms"])
        self.assert_invalid(
            lambda p: p["diagnostics"]["genuinely_stochastic_transition_diagnostic"].__setitem__(
                "repeated_futures_per_identical_condition", 1
            )
        )
        self.assert_invalid(
            lambda p: p["diagnostics"].__setitem__(
                "diagnostic_scores_may_substitute_for_primary_outcomes", True
            )
        )

    def test_conjunctive_superiority_passes_only_four_favorable_confirmatory_intervals(self) -> None:
        decision = evaluate_superiority(
            self.protocol,
            favorable_intervals(),
            evidence_class="confirmatory",
        )
        self.assertTrue(decision.passed)
        self.assertEqual(decision.required_interval_count, 4)
        self.assertEqual(decision.satisfied_interval_count, 4)
        self.assertEqual(decision.failures, ())

    def test_practical_effect_thresholds_are_frozen_before_outcomes(self) -> None:
        practical = self.protocol["statistics"]["practical_significance"]
        self.assertEqual(
            practical["status"], "frozen_before_pilot_or_confirmatory_outcomes"
        )
        self.assertEqual(
            practical["thresholds"]["normalized_free_running_rollout_error_auc"],
            {
                "minimum_absolute_iqm_contrast": 0.05,
                "minimum_relative_iqm_reduction": 0.10,
                "relative_denominator": (
                    "shortcut_forcing_rollout_auc_clipped_below_1e-6"
                ),
            },
        )
        self.assertEqual(
            practical["thresholds"]["real_environment_normalized_return_iqm"]
            ["minimum_absolute_iqm_contrast"],
            0.05,
        )
        changed = deepcopy(self.protocol)
        changed["statistics"]["practical_significance"]["thresholds"][
            "real_environment_normalized_return_iqm"
        ]["minimum_absolute_iqm_contrast"] = 0.0
        with self.assertRaisesRegex(ValueError, "actor minimum absolute contrast"):
            validate_matched_objective_protocol(changed)

    def test_zero_adverse_missing_and_nonconfirmatory_intervals_fail(self) -> None:
        cases = {}
        zero = favorable_intervals()
        zero["equal_updates"][PRIMARY_METRICS[0]]["lower"] = 0.0
        cases["zero_boundary"] = (zero, "confirmatory")
        adverse = favorable_intervals()
        adverse["equal_compiler_flops"][PRIMARY_METRICS[1]] = {"lower": -0.2, "upper": -0.1}
        cases["adverse"] = (adverse, "confirmatory")
        missing = favorable_intervals()
        del missing["equal_updates"][PRIMARY_METRICS[1]]
        cases["missing"] = (missing, "confirmatory")
        cases["smoke"] = (favorable_intervals(), "engineering_smoke")
        cases["pilot"] = (favorable_intervals(), "development_pilot")
        for name, (intervals, evidence_class) in cases.items():
            with self.subTest(name=name):
                decision = evaluate_superiority(
                    self.protocol,
                    intervals,
                    evidence_class=evidence_class,
                )
                self.assertFalse(decision.passed)
                self.assertTrue(decision.failures)

    def test_malformed_and_nonfinite_intervals_fail_closed(self) -> None:
        malformed_values = [
            {"lower": 0.1},
            {"lower": 0.2, "upper": 0.1},
            {"lower": float("nan"), "upper": 0.2},
            {"lower": True, "upper": 0.2},
        ]
        for value in malformed_values:
            intervals = favorable_intervals()
            intervals["equal_updates"][PRIMARY_METRICS[0]] = value
            with self.subTest(value=value):
                decision = evaluate_superiority(
                    self.protocol,
                    intervals,
                    evidence_class="confirmatory",
                )
                self.assertFalse(decision.passed)
                self.assertTrue(any(":invalid:" in failure for failure in decision.failures))

    def test_provenance_and_video_claim_boundary_cannot_be_relaxed(self) -> None:
        self.assertEqual(tuple(self.protocol["provenance"]["required_artifacts"]), REQUIRED_ARTIFACTS)
        self.assertIn("world_model_and_actor_checkpoints", REQUIRED_ARTIFACTS)
        self.assertIn("raw_predictive_draws", REQUIRED_ARTIFACTS)
        self.assertIn("sampled_objective_noise_times_and_step_sizes", REQUIRED_ARTIFACTS)
        self.assertIn("hpo_trial_metrics_and_selection_manifest", REQUIRED_ARTIFACTS)
        self.assertTrue(self.protocol["video_track"]["required_before_broad_video_world_model_claims"])
        self.assert_invalid(lambda p: p["provenance"]["required_artifacts"].remove("raw_predictive_draws"))
        self.assert_invalid(
            lambda p: p["video_track"].__setitem__(
                "required_before_broad_video_world_model_claims", False
            )
        )

    def test_unknown_keys_and_nonfinite_json_are_rejected(self) -> None:
        self.assert_invalid(lambda p: p.__setitem__("post_hoc_rule", True))
        raw = PROTOCOL_PATH.read_text(encoding="utf-8")
        value = json.loads(raw)
        value["parameter_matching"]["relative_active_parameter_gap_trigger"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.json"
            path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_matched_objective_protocol(path)
        duplicate = raw.replace(
            '"status": "frozen_before_outcomes"',
            '"status": "tampered",\n  "status": "frozen_before_outcomes"',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_matched_objective_protocol(path)


if __name__ == "__main__":
    unittest.main()
