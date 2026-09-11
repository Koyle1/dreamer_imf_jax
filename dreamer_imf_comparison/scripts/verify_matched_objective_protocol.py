#!/usr/bin/env python3
"""Verify the frozen trajectory-iMF/shortcut-forcing study protocol."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from dreamer_imf_compare.matched_objective_protocol import (  # noqa: E402
    ACTOR_SEEDS,
    ARM_ORDER,
    BUDGET_TRACK_ORDER,
    EXPLORATORY_ABLATIONS,
    NFE_FRONTIER,
    PRIMARY_METRICS,
    REQUIRED_ARTIFACTS,
    TASKS,
    WORLD_MODEL_SEEDS,
    evaluate_superiority,
    expected_cells,
    protocol_digest,
    read_matched_objective_protocol,
)


PROTOCOL_PATH = PROJECT / "matched_objective_protocol.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _positive_intervals() -> dict[str, dict[str, dict[str, float]]]:
    return {
        track: {
            metric: {"lower": 0.01, "upper": 0.1}
            for metric in PRIMARY_METRICS
        }
        for track in BUDGET_TRACK_ORDER
    }


def verify_contract() -> None:
    protocol = read_matched_objective_protocol(PROTOCOL_PATH)
    _require(len(protocol_digest(protocol)) == 64, "protocol digest is not SHA-256 shaped")
    _require(tuple(protocol["arm_order"]) == ARM_ORDER, "confirmatory arm order drifted")
    _require(tuple(protocol["arms"]) == ARM_ORDER, "protocol no longer has exactly two confirmatory arms")
    _require(
        tuple(protocol["exploratory_ablations"]["ablations"]) == EXPLORATORY_ABLATIONS,
        "exploratory controls drifted",
    )
    _require(
        not protocol["exploratory_ablations"]["may_support_primary_superiority_claim"],
        "an exploratory ablation was allowed into the primary claim",
    )

    shortcut = protocol["arms"]["shortcut_forcing"]
    _require(shortcut["parameterization"] == "x_prediction", "shortcut baseline is not Eq. 7 x-prediction")
    _require(shortcut["independent_per_token_signal_levels"], "shortcut token times are not independent")
    _require(shortcut["independent_per_token_step_sizes"], "shortcut token step sizes are not independent")
    _require(shortcut["bootstrap"]["two_half_steps"], "shortcut bootstrap is not two half steps")
    _require(shortcut["bootstrap"]["target_stop_gradient"], "shortcut bootstrap is not stopped")
    _require(shortcut["loss"]["coordinate_space"] == "x_space", "shortcut loss left x-space")
    _require(shortcut["loss"]["bootstrap_scale"] == "(1-tau)^2", "shortcut scaling drifted")
    _require(shortcut["loss"]["ramp"] == {"expression": "0.9*tau+0.1", "slope": 0.9, "intercept": 0.1}, "shortcut ramp drifted")
    schedule = shortcut["schedule"]
    _require(schedule["training_k_max"] == "pilot_selection_manifest", "baseline K_max is not pilot-selected")
    _require(schedule["training_k_max_candidates"] == [4, 8, 16], "baseline K_max grid drifted")
    _require(not schedule["training_k_max_paper_disclosed"], "protocol falsely claims paper disclosure of K_max")

    proposed = protocol["arms"]["trajectory_imf"]
    _require(proposed["fixed_context_partial_jvp"], "trajectory objective lost its partial JVP")
    _require(proposed["same_noise_path_for_query_history_pair"], "trajectory query/history noise paths differ")
    _require(all(not value for value in proposed["separate_legacy_losses"].values()), "a legacy auxiliary loss is enabled")
    _require(not proposed["teacher_corrupted_suffix_is_self_generated"], "teacher-corrupted suffix was mislabeled as self-generated")

    objective_randomness = protocol["shared_world_model"]["objective_randomness"]
    _require(objective_randomness["same_base_gaussian_samples_within_paired_minibatch"], "base Gaussian pairing missing")
    _require(objective_randomness["paired_canonical_uniforms_by_token"], "canonical uniform pairing missing")
    _require(not objective_randomness["identical_schedule_distribution_required"], "different objectives were falsely forced to share a schedule distribution")

    matching = protocol["parameter_matching"]
    for key in (
        "report_total_parameters",
        "report_active_parameters",
        "inactive_objective_heads_excluded_from_active_count",
        "unused_arm_specific_heads_not_instantiated",
        "report_compiled_forward_flops",
        "report_compiled_forward_and_backward_train_flops",
        "report_compiled_inference_flops_per_transition",
    ):
        _require(matching[key], f"required parameter/compute report disabled: {key}")
    optimization = protocol["shared_world_model"]["optimization"]
    _require(optimization["optimizer"] == "library_custom_global_norm_clipped_Adam", "optimizer contract drifted")
    _require(optimization["weight_decay"] == 0.0, "unimplemented weight decay entered the protocol")
    _require(not optimization["world_model_ema_enabled"], "unimplemented world-model EMA entered the protocol")

    executable = protocol["canonical_executable_config"]
    _require(executable["training_batch"] == {
        "batch_size": 32,
        "sequence_length": 32,
        "burn_in": 8,
        "loss_bearing_steps_per_sequence": 24,
        "sampling": "uniform_contiguous_train_episode_chunks_with_reset_mask_and_no_train_test_crossing",
        "short_episode_handling": "right_pad_and_zero_loss_mask",
    }, "canonical batch contract drifted")
    _require(executable["dreamer_config_non_task_shape_common"]["actor_gradient"] == "reinforce", "actor algorithm drifted")
    _require(executable["dreamer_config_non_task_shape_common"]["imagination_horizon"] == 15, "actor horizon drifted")
    _require(protocol["compute_accounting"]["world_model_update"]["trajectory_imf_include_primal_and_tangent_work_of_JVP"], "iMF JVP work is absent from compute accounting")
    _require(protocol["compute_accounting"]["world_model_update"]["shortcut_include_clean_target_and_both_sequential_half_step_bootstrap_calls"], "shortcut bootstrap calls are absent from compute accounting")

    pairing = protocol["data"]["pairing"]
    _require(all(value is True for value in pairing.values()), "data/split/minibatch/evaluation pairing drifted")
    _require(
        protocol["evaluation"]["paired_evaluation_starts_and_environment_noise"],
        "evaluation streams are not paired",
    )

    _require(tuple(protocol["budget_track_order"]) == BUDGET_TRACK_ORDER, "compute tracks drifted")
    tracks = protocol["budget_tracks"]
    _require(tracks["equal_updates"]["claim_role"] == "primary_objective_isolation_track", "equal updates is not the objective-isolation track")
    _require(tracks["equal_compiler_flops"]["claim_role"] == "required_compute_robustness_track", "equal FLOPs is not required compute robustness")
    _require("forward_and_backward" in tracks["equal_compiler_flops"]["cost_source"], "equal-FLOP accounting omits backward compute")

    confirmatory = protocol["profiles"]["confirmatory"]
    _require(tuple(confirmatory["tasks"]) == TASKS, "confirmatory suite is not the frozen six tasks")
    _require(tuple(confirmatory["world_model_seeds"]) == WORLD_MODEL_SEEDS, "world-model seeds drifted")
    _require(tuple(confirmatory["actor_seeds_nested_within_world_model_seed"]) == ACTOR_SEEDS, "nested actor seeds drifted")
    hpo = protocol["hyperparameter_policy"]
    _require(set(hpo["selection_tasks"]).isdisjoint(TASKS), "HPO tasks leak into the confirmatory suite")
    _require(set(hpo["selection_world_model_seeds"]).isdisjoint(WORLD_MODEL_SEEDS), "HPO world seeds leak")
    _require(set(hpo["selection_actor_seeds_nested_within_world_model_seed"]).isdisjoint(ACTOR_SEEDS), "HPO actor seeds leak")
    _require(hpo["candidate_evaluations_per_arm"] == 12, "HPO budget drifted")
    _require(hpo["equal_candidate_evaluations_per_arm"], "HPO candidate budgets are unequal")
    _require(
        len(hpo["search"]["shortcut_forcing"]["training_k_max"])
        * len(hpo["search"]["shortcut_forcing"]["objective_loss_scale"])
        == hpo["candidate_evaluations_per_arm"]
        and len(hpo["search"]["trajectory_imf"]["imf_boundary_fraction"])
        * len(hpo["search"]["trajectory_imf"]["objective_loss_scale"])
        == hpo["candidate_evaluations_per_arm"],
        "an arm does not consume the same frozen HPO candidate budget",
    )
    _require(hpo["confirmatory_data_checkpoints_and_outcomes_sealed_until_selection_frozen"], "confirmatory outcomes are not sealed during HPO")

    cells = expected_cells(protocol, "confirmatory")
    expected_world_models = len(BUDGET_TRACK_ORDER) * len(TASKS) * len(WORLD_MODEL_SEEDS) * len(ARM_ORDER)
    _require(len(cells["datasets"]) == len(TASKS) * len(WORLD_MODEL_SEEDS), "dataset-cell count drifted")
    _require(len(cells["world_models"]) == expected_world_models, "world-model-cell count drifted")
    _require(len(cells["rollout_evaluations"]) == expected_world_models * len(NFE_FRONTIER), "NFE-cell count drifted")
    _require(len(cells["actors"]) == expected_world_models * len(ACTOR_SEEDS), "nested actor-cell count drifted")

    artifacts = tuple(protocol["provenance"]["required_artifacts"])
    _require(artifacts == REQUIRED_ARTIFACTS, "artifact retention contract drifted")
    for required in (
        "world_model_and_actor_checkpoints",
        "raw_predictive_draws",
        "sampled_objective_noise_times_and_step_sizes",
        "minibatch_indices",
        "raw_action_traces",
        "hpo_trial_metrics_and_selection_manifest",
    ):
        _require(required in artifacts, f"required provenance artifact absent: {required}")
    print("MATCHED_OBJECTIVE_PROTOCOL_VERIFIED")


def verify_statistics() -> None:
    protocol = read_matched_objective_protocol(PROTOCOL_PATH)
    statistics = protocol["statistics"]
    rule = statistics["superiority_rule"]
    _require(rule["required_budget_tracks"] == list(BUDGET_TRACK_ORDER), "superiority omits a compute track")
    _require(rule["required_primary_metrics"] == list(PRIMARY_METRICS), "superiority omits a primary metric")
    _require(rule["conjunctive"], "superiority rule is not conjunctive")
    _require(not rule["compensating_wins_allowed"], "superiority permits compensating wins")
    _require(not rule["secondary_metrics_may_substitute"], "secondary endpoints may substitute")
    _require(rule["primary_objective_isolation_track"] == "equal_updates", "objective-isolation track drifted")
    _require(rule["required_compute_robustness_track"] == "equal_compiler_flops", "compute-robustness track drifted")
    _require(statistics["actor_seeds_are_nested_not_independent_replicates"], "actor seeds became independent replicates")
    _require(statistics["bootstrap"]["resample_world_model_seeds_within_task"], "bootstrap lost task stratification")
    _require(statistics["bootstrap"]["same_resampled_world_model_indices_applied_jointly_to_both_arms"], "bootstrap lost arm pairing")
    _require(not statistics["bootstrap"]["actor_seeds_resampled"], "nested actor seeds were resampled")
    _require(not statistics["bootstrap"]["tasks_resampled"], "fixed tasks were resampled")
    _require(statistics["bootstrap"]["recompute_each_arm_6x10_matrix_IQM_then_the_paired_arm_difference_each_replicate"], "bootstrap does not recompute the registered estimand")
    _require(statistics["bootstrap"]["resamples"] >= 10000, "bootstrap is undersized")
    multiplicity = statistics["multiplicity"]
    expected_interval = 1.0 - (1.0 - multiplicity["familywise_confidence"]) / multiplicity["number_of_confirmatory_intervals"]
    _require(math.isclose(multiplicity["interval_confidence"], expected_interval, abs_tol=1e-12), "Bonferroni confidence drifted")
    _require(math.isclose(statistics["bootstrap"]["two_sided_percentile_lower_quantile"], 0.00625, abs_tol=1e-12), "bootstrap lower quantile drifted")
    _require(math.isclose(statistics["bootstrap"]["two_sided_percentile_upper_quantile"], 0.99375, abs_tol=1e-12), "bootstrap upper quantile drifted")

    evaluation = protocol["evaluation"]
    _require(evaluation["nfe_frontier"] == list(NFE_FRONTIER), "NFE frontier drifted")
    _require(evaluation["primary_nfe"] == {"shortcut_forcing": 4, "trajectory_imf": 1}, "primary NFE comparison drifted")
    _require(evaluation["rollout_metric"]["direction"] == "lower_is_better", "rollout direction drifted")
    _require(evaluation["actor_metric"]["direction"] == "higher_is_better", "return direction drifted")
    _require(evaluation["actor_metric"]["normalization"] == "dmc_episodic_return_divided_by_1000", "return normalization drifted")
    _require("training_split_standard_deviation" in evaluation["rollout_metric"]["normalizer"], "rollout normalization is not train-only scale normalization")
    _require(protocol["diagnostics"]["genuinely_stochastic_transition_diagnostic"]["repeated_futures_per_identical_condition"] >= 64, "stochastic diagnostic is not distributional")

    positive = _positive_intervals()
    decision = evaluate_superiority(protocol, positive, evidence_class="confirmatory")
    _require(decision.passed and decision.satisfied_interval_count == 4, "four favorable intervals did not pass")
    for track in BUDGET_TRACK_ORDER:
        for metric in PRIMARY_METRICS:
            boundary = _positive_intervals()
            boundary[track][metric]["lower"] = 0.0
            _require(
                not evaluate_superiority(protocol, boundary, evidence_class="confirmatory").passed,
                f"zero lower bound passed at {track}.{metric}",
            )
    missing = _positive_intervals()
    del missing["equal_compiler_flops"][PRIMARY_METRICS[1]]
    _require(not evaluate_superiority(protocol, missing, evidence_class="confirmatory").passed, "missing interval passed")
    nonfinite = _positive_intervals()
    nonfinite["equal_updates"][PRIMARY_METRICS[0]]["lower"] = float("nan")
    _require(not evaluate_superiority(protocol, nonfinite, evidence_class="confirmatory").passed, "NaN interval passed")
    _require(not evaluate_superiority(protocol, positive, evidence_class="engineering_smoke").passed, "smoke evidence passed")
    _require(not evaluate_superiority(protocol, positive, evidence_class="development_pilot").passed, "pilot evidence passed")
    print("MATCHED_OBJECTIVE_STATISTICS_VERIFIED")


def verify_tests() -> None:
    environment = dict(os.environ)
    paths = [str(PROJECT)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(PROJECT / "tests"),
            "-p",
            "test_matched_objective_protocol.py",
            "-v",
        ],
        cwd=WORKSPACE,
        env=environment,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print("MATCHED_OBJECTIVE_PROTOCOL_TESTS_VERIFIED")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract", action="store_true", help="verify objective, fairness, cells, and artifacts")
    mode.add_argument("--statistics", action="store_true", help="verify endpoints and fail-closed superiority logic")
    mode.add_argument("--tests", action="store_true", help="run adversarial unit tests")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.contract:
        verify_contract()
    elif args.statistics:
        verify_statistics()
    else:
        verify_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
