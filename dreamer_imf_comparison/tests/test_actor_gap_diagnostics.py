from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

import numpy as np

from dreamer_imf_compare.actor_gap_diagnostics import (
    COUNTERFACTUAL_SCHEMA,
    COVERAGE_SCHEMA,
    HORIZON_SCHEMA,
    NOISE_SCHEMA,
    RESIDUAL_SCHEMA,
    CounterfactualProvenance,
    UnidentifiableCounterfactualError,
    component_residual_decomposition,
    counterfactual_candidate_diagnostics,
    fit_coverage_calibration,
    horizon_stratified_aggregate,
    independent_noise_diagnostics,
    score_observation_action_coverage,
    self_test,
)


class CoverageDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_observations = np.asarray(
            [
                [-2.0, 7.0],
                [-1.0, 7.0],
                [0.0, 7.0],
                [1.0, 7.0],
                [2.0, 7.0],
            ]
        )
        self.train_actions = np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]])

    def test_train_only_calibration_scores_near_above_far_control(self):
        calibration = fit_coverage_calibration(
            self.train_observations, self.train_actions
        )
        near = score_observation_action_coverage(calibration, [[0.1, 7.0]], [[0.1]])
        far = score_observation_action_coverage(calibration, [[100.0, 7.0]], [[100.0]])
        self.assertEqual(calibration.schema_version, COVERAGE_SCHEMA)
        self.assertEqual(calibration.calibration_partition, "train_only")
        self.assertEqual(calibration.constant_observation_dimensions, (1,))
        self.assertGreater(
            far.points[0].nearest_neighbor_distance,
            near.points[0].nearest_neighbor_distance,
        )
        self.assertGreater(near.points[0].kernel_density, far.points[0].kernel_density)
        self.assertGreater(
            near.points[0].conservative_coverage_score,
            far.points[0].conservative_coverage_score,
        )
        self.assertFalse(far.points[0].within_train_distance_q95)

    def test_calibration_metadata_is_immutable_and_json_friendly(self):
        observations = self.train_observations.copy()
        actions = self.train_actions.copy()
        calibration = fit_coverage_calibration(observations, actions, k=2)
        original_digest = calibration.training_data_sha256
        observations[:] = 999.0
        actions[:] = -999.0
        self.assertEqual(calibration.training_data_sha256, original_digest)
        self.assertIsInstance(calibration.standardized_train_points, tuple)
        self.assertIsInstance(calibration.standardized_train_points[0], tuple)
        with self.assertRaises(FrozenInstanceError):
            calibration.k = 1  # type: ignore[misc]
        serialized = json.loads(json.dumps(calibration.to_dict()))
        metadata = calibration.metadata_dict()
        self.assertEqual(serialized["calibration_partition"], "train_only")
        self.assertIn("standardized_train_points", serialized)
        self.assertNotIn("standardized_train_points", metadata)
        self.assertEqual(len(original_digest), 64)

    def test_calibration_is_not_affected_by_evaluation_rows(self):
        first = fit_coverage_calibration(self.train_observations, self.train_actions)
        score_observation_action_coverage(first, [[1e6, 7.0]], [[-1e6]])
        second = fit_coverage_calibration(self.train_observations, self.train_actions)
        self.assertEqual(first, second)
        self.assertEqual(first.training_data_sha256, second.training_data_sha256)

    def test_leave_one_out_calibration_and_k_are_explicit(self):
        calibration = fit_coverage_calibration(
            self.train_observations, self.train_actions, k=2, bandwidth=0.5
        )
        self.assertEqual(calibration.k, 2)
        self.assertEqual(calibration.bandwidth, 0.5)
        self.assertTrue(
            all(value > 0.0 for value in calibration.train_leave_one_out_knn_distances)
        )
        self.assertTrue(
            all(
                0.0 <= value <= 1.0
                for value in calibration.train_leave_one_out_kernel_densities
            )
        )

    def test_blockwise_coverage_is_independent_of_chunk_size(self):
        random = np.random.default_rng(12)
        observations = random.normal(size=(31, 4))
        actions = random.normal(size=(31, 2))
        one_row_blocks = fit_coverage_calibration(
            observations, actions, k=3, distance_chunk_size=1
        )
        one_block = fit_coverage_calibration(
            observations, actions, k=3, distance_chunk_size=100
        )
        np.testing.assert_allclose(
            one_row_blocks.train_leave_one_out_knn_distances,
            one_block.train_leave_one_out_knn_distances,
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            one_row_blocks.train_leave_one_out_kernel_densities,
            one_block.train_leave_one_out_kernel_densities,
            atol=1e-12,
            rtol=0.0,
        )
        query_observations = random.normal(size=(7, 4))
        query_actions = random.normal(size=(7, 2))
        first = score_observation_action_coverage(
            one_row_blocks,
            query_observations,
            query_actions,
            distance_chunk_size=1,
        )
        second = score_observation_action_coverage(
            one_row_blocks,
            query_observations,
            query_actions,
            distance_chunk_size=50,
        )
        np.testing.assert_allclose(
            [point.nearest_neighbor_distance for point in first.points],
            [point.nearest_neighbor_distance for point in second.points],
            atol=1e-12,
            rtol=0.0,
        )

    def test_coverage_shape_and_numerical_negative_controls(self):
        with self.assertRaisesRegex(ValueError, "equal row counts"):
            fit_coverage_calibration(self.train_observations, [[0.0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_coverage_calibration([[0.0], [np.nan]], [[0.0], [1.0]])
        with self.assertRaisesRegex(ValueError, "between 1"):
            fit_coverage_calibration(self.train_observations, self.train_actions, k=5)
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_coverage_calibration(
                self.train_observations, self.train_actions, bandwidth=0.0
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            fit_coverage_calibration(
                self.train_observations,
                self.train_actions,
                distance_chunk_size=0,
            )
        calibration = fit_coverage_calibration(
            self.train_observations, self.train_actions
        )
        with self.assertRaisesRegex(ValueError, "feature count"):
            score_observation_action_coverage(calibration, [[0.0, 1.0, 2.0]], [[0.0]])


class ResidualAndAggregationTests(unittest.TestCase):
    def test_component_residuals_close_exactly_to_bellman_error(self):
        result = component_residual_decomposition(
            predicted_rewards=[2.0, -1.0],
            target_rewards=[1.0, -0.5],
            predicted_continuations=[0.8, 0.25],
            target_continuations=[0.5, 1.0],
            predicted_next_values=[4.0, 2.0],
            target_next_values=[3.0, -1.0],
            discount=np.asarray([0.9, 0.5]),
        )
        self.assertEqual(result.schema_version, RESIDUAL_SCHEMA)
        first = result.rows[0]
        self.assertAlmostEqual(first.reward_residual, 1.0)
        self.assertAlmostEqual(first.continuation_residual, 0.81)
        self.assertAlmostEqual(first.value_projected_transition_residual, 0.72)
        self.assertAlmostEqual(first.bellman_residual, 2.53)
        for row in result.rows:
            self.assertAlmostEqual(
                row.bellman_residual,
                row.reward_residual
                + row.continuation_residual
                + row.value_projected_transition_residual,
            )
            self.assertAlmostEqual(row.decomposition_closure_error, 0.0)
        self.assertEqual(
            set(result.metric_values()),
            {
                "reward_residual",
                "continuation_residual",
                "value_projected_transition_residual",
                "bellman_residual",
            },
        )
        json.dumps(result.to_dict())

    def test_component_residual_validation_rejects_bad_probabilities_and_shapes(self):
        valid = dict(
            predicted_rewards=[0.0],
            target_rewards=[0.0],
            predicted_continuations=[0.5],
            target_continuations=[0.5],
            predicted_next_values=[0.0],
            target_next_values=[0.0],
            discount=0.99,
        )
        bad_probability = dict(valid)
        bad_probability["predicted_continuations"] = [1.1]
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            component_residual_decomposition(**bad_probability)
        bad_shape = dict(valid)
        bad_shape["target_rewards"] = [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "share"):
            component_residual_decomposition(**bad_shape)
        bad_discount = dict(valid)
        bad_discount["discount"] = np.nan
        with self.assertRaisesRegex(ValueError, "discount"):
            component_residual_decomposition(**bad_discount)

    def test_horizon_aggregate_equal_weights_world_models_not_nested_rows(self):
        result = horizon_stratified_aggregate(
            world_model_ids=["wm-a", "wm-a", "wm-a", "wm-b", "wm-a", "wm-b"],
            horizons=[1, 1, 1, 1, 3, 3],
            metrics={
                "bellman_residual": [0.0, 0.0, 0.0, 10.0, 2.0, 4.0],
                "reward_residual": [1.0, 1.0, 1.0, 5.0, 3.0, 7.0],
            },
        )
        self.assertEqual(result.schema_version, HORIZON_SCHEMA)
        self.assertEqual(result.top_level_unit, "world_model")
        horizon_one = next(
            row
            for row in result.strata
            if row.horizon == 1 and row.metric == "bellman_residual"
        )
        self.assertEqual(horizon_one.nested_sample_count, 4)
        self.assertEqual(horizon_one.world_model_count, 2)
        self.assertEqual(horizon_one.mean_of_world_model_means, 5.0)
        self.assertNotEqual(horizon_one.mean_of_world_model_means, 2.5)
        self.assertEqual(
            [row.nested_sample_count for row in horizon_one.per_world_model],
            [3, 1],
        )
        self.assertIsInstance(result.strata, tuple)
        self.assertIsInstance(horizon_one.per_world_model, tuple)
        with self.assertRaises(FrozenInstanceError):
            horizon_one.world_model_count = 99  # type: ignore[misc]
        json.dumps(result.to_dict())

    def test_horizon_aggregate_validates_integer_horizon_and_metric_lengths(self):
        with self.assertRaisesRegex(ValueError, "integers"):
            horizon_stratified_aggregate(["wm"], [1.5], {"x": [1.0]})
        with self.assertRaisesRegex(ValueError, "length"):
            horizon_stratified_aggregate(["wm-a", "wm-b"], [1, 1], {"x": [1.0]})
        with self.assertRaisesRegex(ValueError, "finite"):
            horizon_stratified_aggregate(["wm"], [1], {"x": [np.inf]})


class IndependentNoiseTests(unittest.TestCase):
    def test_gain_and_gradient_cosine_positive_and_opposed_controls(self):
        aligned = independent_noise_diagnostics(
            proposal_objective_improvements=[0.0, 1.0],
            heldout_objective_improvements=[1.0, 3.0],
            proposal_gradients=[[1.0, 0.0], [0.0, 2.0]],
            heldout_gradients=[[2.0, 0.0], [0.0, 1.0]],
            proposal_noise_ids=["fit-1", "fit-2"],
            heldout_noise_ids=["heldout-1", "heldout-2"],
        )
        opposed = independent_noise_diagnostics(
            proposal_objective_improvements=[0.0, 1.0],
            heldout_objective_improvements=[-1.0, 0.0],
            proposal_gradients=[[1.0, 0.0], [0.0, 2.0]],
            heldout_gradients=[[-2.0, 0.0], [0.0, -1.0]],
            proposal_noise_ids=["fit"],
            heldout_noise_ids=["heldout"],
        )
        self.assertEqual(aligned.schema_version, NOISE_SCHEMA)
        self.assertEqual(aligned.proposal_mean_improvement, 0.5)
        self.assertEqual(aligned.heldout_mean_improvement, 2.0)
        self.assertEqual(aligned.mean_generalization_gap, 1.5)
        self.assertEqual(aligned.heldout_positive_improvement_fraction, 1.0)
        self.assertAlmostEqual(aligned.mean_gradient_cosine, 1.0)
        self.assertEqual(opposed.proposal_mean_improvement, 0.5)
        self.assertEqual(opposed.heldout_mean_improvement, -0.5)
        self.assertEqual(opposed.mean_generalization_gap, -1.0)
        self.assertEqual(opposed.heldout_positive_improvement_fraction, 0.0)
        self.assertAlmostEqual(opposed.mean_gradient_cosine, -1.0)
        payload = aligned.to_dict()
        self.assertEqual(
            payload["generalization_gap_convention"],
            "heldout_improvement_minus_proposal_improvement",
        )
        self.assertNotIn("mean_gain", payload)
        self.assertNotIn("positive_gain_fraction", payload)
        self.assertNotIn("reference_noise_ids", payload)
        self.assertNotIn("independent_noise_ids", payload)
        json.dumps(payload)

    def test_zero_gradients_are_counted_and_not_fabricated_as_cosines(self):
        result = independent_noise_diagnostics(
            [0.0, 0.0],
            [0.0, 0.0],
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            proposal_noise_ids=["fit"],
            heldout_noise_ids=["heldout"],
        )
        self.assertEqual(result.cosine_pairs, 0)
        self.assertIsNone(result.mean_gradient_cosine)
        self.assertEqual(result.zero_proposal_gradient_count, 1)
        self.assertEqual(result.zero_heldout_gradient_count, 1)

    def test_independence_and_shape_negative_controls(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            independent_noise_diagnostics(
                [0.0],
                [1.0],
                [[1.0]],
                [[1.0]],
                proposal_noise_ids=["same"],
                heldout_noise_ids=["same"],
            )
        with self.assertRaisesRegex(ValueError, "share shape"):
            independent_noise_diagnostics(
                [0.0],
                [1.0, 2.0],
                [[1.0]],
                [[1.0]],
                proposal_noise_ids=["fit"],
                heldout_noise_ids=["heldout"],
            )


class CounterfactualDiagnosticsTests(unittest.TestCase):
    def test_offline_input_is_marked_unidentifiable_without_reading_arrays(self):
        result = counterfactual_candidate_diagnostics(
            predicted_returns=None,
            simulator_returns=None,
            provenance=CounterfactualProvenance.offline_only(),
        )
        self.assertEqual(result.schema_version, COUNTERFACTUAL_SCHEMA)
        self.assertFalse(result.identifiable)
        self.assertIsNone(result.metrics)
        self.assertIn("offline tuples", result.unidentifiable_reason)
        payload = json.loads(json.dumps(result.to_dict()))
        self.assertFalse(payload["identifiable"])
        self.assertIsNone(payload["metrics"])
        self.assertFalse(payload["provenance"]["exact_simulator_branch"])

    def test_missing_provenance_is_marked_and_strict_offline_mode_rejects(self):
        missing = counterfactual_candidate_diagnostics(
            [[0.0, 1.0]], [[0.0, 1.0]], provenance=None
        )
        self.assertFalse(missing.identifiable)
        self.assertIn("explicit", missing.unidentifiable_reason)
        with self.assertRaisesRegex(
            UnidentifiableCounterfactualError, "offline tuples"
        ):
            counterfactual_candidate_diagnostics(
                [[0.0, 1.0]],
                [[0.0, 1.0]],
                provenance=CounterfactualProvenance.offline_only(),
                require_identifiable=True,
            )

    def test_exact_provenance_requires_all_flags_and_common_reset_ids(self):
        with self.assertRaisesRegex(ValueError, "snapshot/restore"):
            CounterfactualProvenance(
                source_kind="exact_simulator_branch",
                exact_simulator_branch=True,
                snapshot_restore_verified=False,
                common_reset_ids=("reset",),
            )
        with self.assertRaisesRegex(ValueError, "common_reset_ids"):
            CounterfactualProvenance(
                source_kind="exact_simulator_branch",
                exact_simulator_branch=True,
                snapshot_restore_verified=True,
            )
        with self.assertRaisesRegex(ValueError, "cannot assert"):
            CounterfactualProvenance(
                source_kind="offline_only",
                exact_simulator_branch=True,
                snapshot_restore_verified=True,
                common_reset_ids=("reset",),
            )
        with self.assertRaisesRegex(ValueError, "simulator_id"):
            CounterfactualProvenance(
                source_kind="exact_simulator_branch",
                exact_simulator_branch=True,
                snapshot_restore_verified=True,
                common_reset_ids=("reset",),
            )

    def test_exact_branch_perfect_candidate_ranking_regret_and_calibration(self):
        truth = np.asarray([[0.0, 1.0, 3.0], [4.0, 2.0, -1.0]])
        provenance = CounterfactualProvenance.verified_exact_branch(
            ["reset-a", "reset-b"], simulator_id="dmc-reacher"
        )
        result = counterfactual_candidate_diagnostics(
            truth, truth, provenance=provenance
        )
        self.assertTrue(result.identifiable)
        self.assertIsNotNone(result.metrics)
        metrics = result.metrics
        assert metrics is not None
        self.assertEqual(metrics.pairwise_ranking_accuracy, 1.0)
        self.assertAlmostEqual(metrics.mean_within_reset_spearman, 1.0)
        self.assertEqual(metrics.top1_accuracy, 1.0)
        self.assertEqual(metrics.mean_simulator_regret, 0.0)
        self.assertEqual(metrics.return_rmse, 0.0)
        self.assertEqual(metrics.pairwise_gain_rmse, 0.0)
        self.assertAlmostEqual(metrics.calibration_slope_simulator_on_predicted, 1.0)
        self.assertAlmostEqual(
            metrics.calibration_intercept_simulator_on_predicted, 0.0
        )
        json.dumps(result.to_dict())

    def test_reversed_candidate_control_detects_ranking_and_regret_failure(self):
        truth = np.asarray([[0.0, 1.0, 3.0], [4.0, 2.0, -1.0]])
        result = counterfactual_candidate_diagnostics(
            -truth,
            truth,
            provenance=CounterfactualProvenance.verified_exact_branch(
                ["reset-a", "reset-b"], simulator_id="sim"
            ),
        )
        metrics = result.metrics
        assert metrics is not None
        self.assertEqual(metrics.pairwise_ranking_accuracy, 0.0)
        self.assertAlmostEqual(metrics.mean_within_reset_spearman, -1.0)
        self.assertEqual(metrics.top1_accuracy, 0.0)
        self.assertGreater(metrics.mean_simulator_regret, 0.0)
        self.assertLess(metrics.calibration_slope_simulator_on_predicted, 0.0)

    def test_tied_truth_and_constant_prediction_are_reported_not_overclaimed(self):
        result = counterfactual_candidate_diagnostics(
            np.zeros((2, 3)),
            np.ones((2, 3)),
            provenance=CounterfactualProvenance.verified_exact_branch(
                ["reset-a", "reset-b"], simulator_id="sim"
            ),
        )
        metrics = result.metrics
        assert metrics is not None
        self.assertEqual(metrics.informative_reset_count, 0)
        self.assertEqual(metrics.comparable_pair_count, 0)
        self.assertIsNone(metrics.pairwise_ranking_accuracy)
        self.assertIsNone(metrics.top1_accuracy)
        self.assertIsNone(metrics.mean_simulator_regret)
        self.assertIsNone(metrics.calibration_slope_simulator_on_predicted)

    def test_exact_branch_validates_reset_alignment_and_return_shapes(self):
        provenance = CounterfactualProvenance.verified_exact_branch(
            ["only-one"], simulator_id="sim"
        )
        with self.assertRaisesRegex(ValueError, "every candidate row"):
            counterfactual_candidate_diagnostics(
                np.zeros((2, 2)), np.zeros((2, 2)), provenance=provenance
            )
        with self.assertRaisesRegex(ValueError, "share shape"):
            counterfactual_candidate_diagnostics(
                np.zeros((1, 2)), np.zeros((1, 3)), provenance=provenance
            )
        duplicate = CounterfactualProvenance.verified_exact_branch(
            ["same", "same"], simulator_id="sim"
        )
        with self.assertRaisesRegex(ValueError, "uniquely"):
            counterfactual_candidate_diagnostics(
                np.zeros((2, 2)), np.zeros((2, 2)), provenance=duplicate
            )

    def test_self_test_exercises_positive_and_fail_closed_controls(self):
        self_test()


if __name__ == "__main__":
    unittest.main()
