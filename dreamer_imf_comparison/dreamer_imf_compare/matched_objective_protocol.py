"""Frozen matched-objective protocol for trajectory iMF versus shortcut forcing.

The module is intentionally dependency-free.  It validates the preregistered
study contract before any runner consumes it and implements the sole
confirmatory superiority decision.  Exploratory ablations never enter that
decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trajectory-imf-shortcut-objective-v2"
ARM_ORDER = ("shortcut_forcing", "trajectory_imf")
EXPLORATORY_ABLATIONS = (
    "shared_time_only",
    "clean_context_only",
    "joint_diagonal_context_jvp_negative_control",
)
BUDGET_TRACK_ORDER = ("equal_updates", "equal_compiler_flops")
PROFILE_ORDER = ("smoke", "pilot", "confirmatory")
TASKS = (
    "dmc_cartpole_swingup",
    "dmc_ball_in_cup_catch",
    "dmc_cheetah_run",
    "dmc_walker_walk",
    "dmc_finger_spin",
    "dmc_hopper_hop",
)
WORLD_MODEL_SEEDS = (17, 27, 37, 47, 57, 67, 77, 87, 97, 107)
ACTOR_SEEDS = (101, 202, 303)
PILOT_TASKS = ("dmc_reacher_easy", "dmc_pendulum_swingup")
PILOT_WORLD_MODEL_SEEDS = (211, 223, 227)
PILOT_ACTOR_SEEDS = (311, 313)
NFE_FRONTIER = (1, 2, 4)
ROLLOUT_HORIZONS = (1, 2, 4, 8, 15, 30)
PILOT_CANDIDATES_PER_ARM = 12
DMC_TASK_REGISTRY = {
    "dmc_cartpole_swingup": ("cartpole", "swingup"),
    "dmc_ball_in_cup_catch": ("ball_in_cup", "catch"),
    "dmc_cheetah_run": ("cheetah", "run"),
    "dmc_walker_walk": ("walker", "walk"),
    "dmc_finger_spin": ("finger", "spin"),
    "dmc_hopper_hop": ("hopper", "hop"),
    "dmc_reacher_easy": ("reacher", "easy"),
    "dmc_pendulum_swingup": ("pendulum", "swingup"),
}
DREAMER_CONFIG_TASK_SHAPE_FIELDS = ("observation_shape", "action_dim")
DREAMER_CONFIG_NON_TASK_SHAPE_FIELDS = (
    "deterministic_dim",
    "stochastic_dim",
    "embedding_dim",
    "hidden_dim",
    "prior",
    "min_std",
    "max_std",
    "kl_free_nats",
    "prior_scale",
    "representation_scale",
    "reconstruction_scale",
    "reward_scale",
    "reward_min",
    "reward_max",
    "reward_initial_value",
    "reward_output_init_scale",
    "reward_loss",
    "continuation_scale",
    "overshooting_horizon",
    "burn_in",
    "overshooting_distances",
    "overshooting_scale",
    "imf_boundary_fraction",
    "imf_time_mean",
    "imf_time_std",
    "imf_adaptive_power",
    "imf_adaptive_epsilon",
    "imf_meanflow_scale",
    "imf_velocity_scale",
    "imf_noise_coupling",
    "imf_endpoint_scale",
    "imf_shortcut_scale",
    "imf_signal_weight_floor",
    "imf_signal_weight_scale",
    "imf_sampling_steps",
    "imf_condition_gradient_scale",
    "imf_boundary_velocity_supervision",
    "imf_trajectory_enabled",
    "imf_trajectory_clean_probability",
    "imf_trajectory_corrupted_probability",
    "imf_trajectory_suffix_probability",
    "imf_trajectory_history_noise_max",
    "imf_causal_consistency_scale",
    "imf_causal_reward_scale",
    "imf_causal_huber_delta",
    "imf_causal_normalization_epsilon",
    "shortcut_training_k_max",
    "shortcut_sampling_steps",
    "shortcut_bootstrap_ema_decay",
    "shortcut_intermediate_clip",
    "shortcut_support_safe_bootstrap",
    "shortcut_sampling_clip",
    "imagination_horizon",
    "discount",
    "lambda_",
    "actor_entropy_scale",
    "actor_init_scale",
    "actor_mean_bound",
    "actor_min_std",
    "actor_max_std",
    "actor_gradient",
    "actor_action_l2_scale",
    "behavior_cloning_learning_rate",
    "behavior_kl_scale",
    "pmpo_positive_temperature",
    "pmpo_negative_temperature",
    "pmpo_positive_weight",
    "pmpo_negative_weight",
    "imagination_start_mode",
    "critic_output_init_scale",
    "critic_bins",
    "critic_symlog_min",
    "critic_symlog_max",
    "actor_critic_warmup_steps",
    "slow_critic_fraction",
    "return_normalization_epsilon",
    "model_learning_rate",
    "actor_learning_rate",
    "critic_learning_rate",
    "grad_clip",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
)
FROZEN_DREAMER_CONFIG_COMMON = {
    "deterministic_dim": 128,
    "stochastic_dim": 32,
    "embedding_dim": 128,
    "hidden_dim": 256,
    "min_std": 0.1,
    "max_std": 2.0,
    "kl_free_nats": 0.0,
    "prior_scale": 1.0,
    "representation_scale": 0.05,
    "reconstruction_scale": 1.0,
    "reward_scale": 1.0,
    "reward_min": None,
    "reward_max": None,
    "reward_initial_value": 0.0,
    "reward_output_init_scale": 0.0,
    "reward_loss": "mse",
    "continuation_scale": 1.0,
    "overshooting_horizon": 1,
    "burn_in": 8,
    "overshooting_distances": [1],
    "overshooting_scale": 0.0,
    "imf_boundary_fraction": 0.5,
    "imf_time_mean": -0.4,
    "imf_time_std": 1.0,
    "imf_adaptive_power": 1.0,
    "imf_adaptive_epsilon": 0.01,
    "imf_meanflow_scale": 1.0,
    "imf_velocity_scale": 1.0,
    "imf_noise_coupling": "independent",
    "imf_endpoint_scale": 0.0,
    "imf_shortcut_scale": 0.0,
    "imf_signal_weight_floor": 1.0,
    "imf_signal_weight_scale": 0.0,
    "imf_sampling_steps": 1,
    "imf_condition_gradient_scale": 1.0,
    "imf_boundary_velocity_supervision": True,
    "imf_trajectory_clean_probability": 1.0 / 3.0,
    "imf_trajectory_corrupted_probability": 1.0 / 3.0,
    "imf_trajectory_suffix_probability": 1.0 / 3.0,
    "imf_trajectory_history_noise_max": 1.0,
    "imf_causal_consistency_scale": 0.0,
    "imf_causal_reward_scale": 1.0,
    "imf_causal_huber_delta": 1.0,
    "imf_causal_normalization_epsilon": 0.001,
    "shortcut_sampling_steps": 4,
    "shortcut_bootstrap_ema_decay": 0.999,
    "shortcut_intermediate_clip": 4.0,
    "shortcut_support_safe_bootstrap": True,
    "shortcut_sampling_clip": None,
    "imagination_horizon": 15,
    "discount": 0.99,
    "lambda_": 0.95,
    "actor_entropy_scale": 0.001,
    "actor_init_scale": 0.01,
    "actor_mean_bound": 5.0,
    "actor_min_std": 0.1,
    "actor_max_std": 1.0,
    "actor_gradient": "reinforce",
    "actor_action_l2_scale": 0.001,
    "behavior_cloning_learning_rate": 0.0003,
    "behavior_kl_scale": 0.0,
    "pmpo_positive_temperature": 1.0,
    "pmpo_negative_temperature": 1.0,
    "pmpo_positive_weight": 1.0,
    "pmpo_negative_weight": 1.0,
    "imagination_start_mode": "one_per_sequence",
    "critic_output_init_scale": 0.0,
    "critic_bins": 1,
    "critic_symlog_min": -20.0,
    "critic_symlog_max": 20.0,
    "actor_critic_warmup_steps": 0,
    "slow_critic_fraction": 0.02,
    "return_normalization_epsilon": 0.000001,
    "model_learning_rate": 0.0003,
    "actor_learning_rate": 0.0003,
    "critic_learning_rate": 0.0003,
    "grad_clip": 100.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
}
PRIMARY_METRICS = (
    "normalized_free_running_rollout_error_auc",
    "real_environment_normalized_return_iqm",
)
SECONDARY_METRICS = (
    "rollout_error_at_each_horizon",
    "reward_prediction_error",
    "continuation_prediction_error",
    "energy_score",
    "central_90_percent_interval_coverage",
    "central_90_percent_interval_width",
    "central_90_percent_interval_calibration_absolute_error_at_each_horizon",
    "continuation_predictive_mean_brier_at_each_horizon",
    "inference_field_evaluations_per_transition",
    "total_parameters",
    "active_parameters",
    "compiler_reported_forward_flops",
    "compiler_reported_forward_and_backward_train_flops",
    "compiler_reported_inference_flops_per_transition",
    "actor_imagined_transitions_per_update",
    "actor_prior_field_evaluations_per_update",
    "compiler_reported_actor_critic_forward_and_backward_train_flops",
    "synchronized_update_wall_clock_seconds",
    "wall_clock_seconds",
)
REQUIRED_ARTIFACTS = (
    "frozen_protocol",
    "source_commit_and_dirty_patch",
    "dependency_lock",
    "hardware_and_software_environment",
    "dataset_payload_and_digest",
    "episode_split_indices",
    "minibatch_indices",
    "sampled_objective_noise_times_and_step_sizes",
    "evaluation_start_indices",
    "evaluation_episode_seeds",
    "compiler_ir_and_cost_analysis",
    "world_model_and_actor_checkpoints",
    "parameter_tree_hashes_and_counts",
    "hpo_trial_metrics_and_selection_manifest",
    "resolved_per_arm_DreamerConfig_and_sha256",
    "unoptimized_and_optimized_HLO_with_sha256",
    "full_update_compute_and_synchronized_walltime_records",
    "diagnostic_raw_samples_estimands_and_threshold_interpretations",
    "raw_predictive_draws",
    "raw_action_traces",
    "raw_per_seed_primary_and_secondary_metrics",
    "bootstrap_draw_indices_and_intervals",
    "machine_readable_summary_and_human_report",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "study_name",
    "claim",
    "claim_boundary",
    "source",
    "arm_order",
    "arms",
    "exploratory_ablations",
    "shared_world_model",
    "canonical_executable_config",
    "parameter_matching",
    "compute_accounting",
    "data",
    "randomness",
    "hyperparameter_policy",
    "budget_track_order",
    "budget_tracks",
    "profiles",
    "diagnostics",
    "evaluation",
    "statistics",
    "provenance",
    "video_track",
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{path} keys differ; missing={missing}, extra={extra}")


def _equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise ValueError(f"{path} must equal {expected!r}, got {actual!r}")


def _bool(value: Any, expected: bool, path: str) -> None:
    if value is not expected:
        raise ValueError(f"{path} must be {expected}")


def _positive_int(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _unit_interval(value: Any, path: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not math.isfinite(result) or not valid:
        brackets = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{path} must lie in {brackets}")
    return result


def _finite(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _unique_nonnegative_ints(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a nonempty list")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{path}[{index}] must be a nonnegative integer")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{path} must contain unique seeds")
    return tuple(result)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def read_matched_objective_protocol(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate a matched-objective protocol JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    if not isinstance(value, dict):
        raise ValueError("matched-objective protocol must be a JSON object")
    validate_matched_objective_protocol(value)
    return value


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of canonical validated protocol JSON."""

    validate_matched_objective_protocol(protocol)
    return hashlib.sha256(_canonical_json(protocol)).hexdigest()


def _validate_claim_and_source(protocol: Mapping[str, Any]) -> None:
    _equal(protocol["status"], "frozen_before_outcomes", "status")
    if not isinstance(protocol["study_name"], str) or not protocol["study_name"].strip():
        raise ValueError("study_name must be a nonempty string")
    if not isinstance(protocol["claim"], str) or "outperforms" not in protocol["claim"]:
        raise ValueError("claim must state the prespecified superiority claim")
    if "faithful-to-published-Dreamer-4-Equation-7 reimplementation with declared choices" not in protocol["claim"]:
        raise ValueError("claim must use the bounded published-equation baseline wording")

    boundary = _mapping(protocol["claim_boundary"], "claim_boundary")
    _expect_keys(
        boundary,
        {"baseline_identity", "not_claimed", "smoke_or_pilot_supports_superiority"},
        "claim_boundary",
    )
    _equal(
        boundary["baseline_identity"],
        "faithful-to-published-Dreamer-4-Equation-7 reimplementation with declared choices",
        "claim_boundary.baseline_identity",
    )
    _equal(
        boundary["not_claimed"],
        [
            "official_unreleased_Dreamer4_code",
            "the_2B_parameter_Minecraft_system",
            "broad_video_world_model_superiority_without_the_video_track",
            "a_full_Dreamer4_agent_or_training_system_reproduction",
        ],
        "claim_boundary.not_claimed",
    )
    _bool(
        boundary["smoke_or_pilot_supports_superiority"],
        False,
        "claim_boundary.smoke_or_pilot_supports_superiority",
    )

    source = _mapping(protocol["source"], "source")
    _expect_keys(
        source,
        {
            "title",
            "identifier",
            "uri",
            "equations",
            "training_k_max_disclosed",
            "declared_reimplementation_choices",
        },
        "source",
    )
    _equal(source["title"], "Training Agents Inside of Scalable World Models", "source.title")
    _equal(source["identifier"], "arXiv:2509.24527v1", "source.identifier")
    _equal(source["uri"], "https://arxiv.org/abs/2509.24527", "source.uri")
    _equal(source["equations"], [4, 7], "source.equations")
    _bool(source["training_k_max_disclosed"], False, "source.training_k_max_disclosed")
    _equal(
        source["declared_reimplementation_choices"],
        [
            "pilot_selected_training_k_max_from_{4,8,16}",
            "primary_clean_observed_prefix_then_unmodified_generated_history",
            "state_observations_and_current_library_architecture",
            "custom_clipped_Adam_without_weight_decay_with_shortcut_bootstrap_EMA_teacher",
            "bounded_shortcut_bootstrap_intermediate_at_absolute_value_4",
            "support_safe_finest_tokens_without_below_training_support_teacher_queries",
            "unclipped_shortcut_generation_with_fail_closed_divergence_gate",
        ],
        "source.declared_reimplementation_choices",
    )


def _validate_shortcut_arm(arm: Mapping[str, Any]) -> None:
    _expect_keys(
        arm,
        {
            "role",
            "objective",
            "parameterization",
            "independent_per_token_signal_levels",
            "independent_per_token_step_sizes",
            "schedule",
            "bootstrap",
            "loss",
            "generation",
        },
        "arms.shortcut_forcing",
    )
    _equal(arm["role"], "baseline", "arms.shortcut_forcing.role")
    _equal(
        arm["objective"],
        "dreamer4_equation_7_x_prediction_shortcut_forcing",
        "arms.shortcut_forcing.objective",
    )
    _equal(arm["parameterization"], "x_prediction", "arms.shortcut_forcing.parameterization")
    _bool(
        arm["independent_per_token_signal_levels"],
        True,
        "arms.shortcut_forcing.independent_per_token_signal_levels",
    )
    _bool(
        arm["independent_per_token_step_sizes"],
        True,
        "arms.shortcut_forcing.independent_per_token_step_sizes",
    )

    schedule = _mapping(arm["schedule"], "arms.shortcut_forcing.schedule")
    _expect_keys(
        schedule,
        {
            "step_count_values",
            "training_k_max",
            "training_k_max_candidates",
            "training_k_max_selection_profile",
            "training_k_max_paper_disclosed",
            "training_k_max_choice",
            "step_size",
            "signal_level",
            "signal_index_support",
            "sampling",
        },
        "arms.shortcut_forcing.schedule",
    )
    _equal(
        schedule["step_count_values"],
        "powers_of_two_through_k_max_inclusive",
        "arms.shortcut_forcing.schedule.step_count_values",
    )
    _equal(
        schedule["training_k_max"],
        "pilot_selection_manifest",
        "arms.shortcut_forcing.schedule.training_k_max",
    )
    candidates = schedule["training_k_max_candidates"]
    _equal(candidates, [4, 8, 16], "arms.shortcut_forcing.schedule.training_k_max_candidates")
    if any(not _is_power_of_two(value) or value < max(NFE_FRONTIER) for value in candidates):
        raise ValueError("every shortcut K_max candidate must be a power of two covering the NFE frontier")
    _equal(
        schedule["training_k_max_selection_profile"],
        "pilot",
        "arms.shortcut_forcing.schedule.training_k_max_selection_profile",
    )
    _bool(
        schedule["training_k_max_paper_disclosed"],
        False,
        "arms.shortcut_forcing.schedule.training_k_max_paper_disclosed",
    )
    _equal(
        schedule["training_k_max_choice"],
        "selected_only_on_pilot_equal_updates_by_the_frozen_joint_selection_rule",
        "arms.shortcut_forcing.schedule.training_k_max_choice",
    )
    _equal(schedule["step_size"], "1/K", "arms.shortcut_forcing.schedule.step_size")
    _equal(schedule["signal_level"], "tau=j/K", "arms.shortcut_forcing.schedule.signal_level")
    _equal(
        schedule["signal_index_support"],
        "j_in_{0,...,K-1}",
        "arms.shortcut_forcing.schedule.signal_index_support",
    )
    _equal(
        schedule["sampling"],
        "independent_uniform_over_step_counts_then_grid_points_per_token",
        "arms.shortcut_forcing.schedule.sampling",
    )

    bootstrap = _mapping(arm["bootstrap"], "arms.shortcut_forcing.bootstrap")
    _expect_keys(
        bootstrap,
        {
            "finest_step_target",
            "nonfinest_target",
            "two_half_steps",
            "target_stop_gradient",
            "target_expression",
            "teacher",
            "intermediate_clip",
            "support_safe_composition",
        },
        "arms.shortcut_forcing.bootstrap",
    )
    _equal(bootstrap["finest_step_target"], "clean_data", "shortcut bootstrap finest target")
    _equal(
        bootstrap["nonfinest_target"],
        "two_sequential_half_steps",
        "shortcut bootstrap nonfinest target",
    )
    _bool(bootstrap["two_half_steps"], True, "shortcut bootstrap two_half_steps")
    _bool(bootstrap["target_stop_gradient"], True, "shortcut bootstrap target_stop_gradient")
    _equal(
        bootstrap["target_expression"],
        "x_target=z_tau+(1-tau)*stop_gradient((b1+b2)/2)",
        "shortcut bootstrap target_expression",
    )
    _equal(
        bootstrap["teacher"],
        "ema_parameters_decay_0.999_updated_after_student",
        "shortcut bootstrap teacher",
    )
    _equal(
        _finite(bootstrap["intermediate_clip"], "shortcut intermediate clip"),
        4.0,
        "shortcut intermediate clip",
    )
    _equal(
        bootstrap["support_safe_composition"],
        "finest_tokens_keep_z_tau_and_teacher_step_never_below_1_over_k_max",
        "shortcut support-safe composition",
    )

    loss = _mapping(arm["loss"], "arms.shortcut_forcing.loss")
    _expect_keys(loss, {"coordinate_space", "bootstrap_scale", "ramp"}, "shortcut loss")
    _equal(loss["coordinate_space"], "x_space", "shortcut loss coordinate_space")
    _equal(loss["bootstrap_scale"], "(1-tau)^2", "shortcut loss bootstrap_scale")
    ramp = _mapping(loss["ramp"], "arms.shortcut_forcing.loss.ramp")
    _expect_keys(ramp, {"expression", "slope", "intercept"}, "shortcut loss ramp")
    _equal(ramp["expression"], "0.9*tau+0.1", "shortcut loss ramp expression")
    _equal(_finite(ramp["slope"], "shortcut ramp slope"), 0.9, "shortcut ramp slope")
    _equal(_finite(ramp["intercept"], "shortcut ramp intercept"), 0.1, "shortcut ramp intercept")
    _validate_generation(arm["generation"], primary_nfe=4, path="arms.shortcut_forcing.generation")


def _validate_generation(value: Any, *, primary_nfe: int, path: str) -> None:
    generation = _mapping(value, path)
    _expect_keys(
        generation,
        {"supported_field_evaluations", "primary_field_evaluations", "nfe_equals_steps"},
        path,
    )
    _equal(generation["supported_field_evaluations"], list(NFE_FRONTIER), f"{path}.supported")
    _equal(generation["primary_field_evaluations"], primary_nfe, f"{path}.primary")
    _bool(generation["nfe_equals_steps"], True, f"{path}.nfe_equals_steps")


def _validate_trajectory_arm(arm: Mapping[str, Any]) -> None:
    _expect_keys(
        arm,
        {
            "role",
            "objective",
            "parameterization",
            "independent_per_token_signal_levels",
            "fixed_context_partial_jvp",
            "same_noise_path_for_query_history_pair",
            "context_patterns",
            "context_pattern_sampling_unit",
            "teacher_corrupted_suffix_is_self_generated",
            "separate_legacy_losses",
            "generation",
        },
        "arms.trajectory_imf",
    )
    _equal(arm["role"], "proposed", "arms.trajectory_imf.role")
    _equal(
        arm["objective"],
        "fixed_context_per_token_time_trajectory_imf",
        "arms.trajectory_imf.objective",
    )
    _equal(
        arm["parameterization"],
        "velocity_with_partial_time_jvp",
        "arms.trajectory_imf.parameterization",
    )
    for key in (
        "independent_per_token_signal_levels",
        "fixed_context_partial_jvp",
        "same_noise_path_for_query_history_pair",
    ):
        _bool(arm[key], True, f"arms.trajectory_imf.{key}")
    patterns = _mapping(arm["context_patterns"], "arms.trajectory_imf.context_patterns")
    _expect_keys(
        patterns,
        {"clean_context", "corrupted_context", "teacher_corrupted_suffix_supervision"},
        "context patterns",
    )
    probabilities = [_finite(patterns[key], f"context_patterns.{key}") for key in patterns]
    if any(value <= 0.0 for value in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("trajectory context-pattern probabilities must be positive and sum to one")
    if len({round(value, 15) for value in probabilities}) != 1:
        raise ValueError("trajectory context patterns must use the frozen equal mixture")
    _equal(
        arm["context_pattern_sampling_unit"],
        "one_pattern_per_training_sequence",
        "trajectory context-pattern sampling unit",
    )
    _bool(
        arm["teacher_corrupted_suffix_is_self_generated"],
        False,
        "trajectory teacher-corrupted suffix label",
    )
    legacy = _mapping(arm["separate_legacy_losses"], "arms.trajectory_imf.separate_legacy_losses")
    _expect_keys(legacy, {"shortcut_loss", "context_loss", "overshooting_loss"}, "legacy losses")
    for key in legacy:
        _bool(legacy[key], False, f"arms.trajectory_imf.separate_legacy_losses.{key}")
    _validate_generation(arm["generation"], primary_nfe=1, path="arms.trajectory_imf.generation")


def _validate_arms_and_ablations(protocol: Mapping[str, Any]) -> None:
    _equal(protocol["arm_order"], list(ARM_ORDER), "arm_order")
    arms = _mapping(protocol["arms"], "arms")
    if set(arms) != set(ARM_ORDER):
        raise ValueError("arms must contain exactly the two confirmatory arms in canonical order")
    _validate_shortcut_arm(_mapping(arms["shortcut_forcing"], "arms.shortcut_forcing"))
    _validate_trajectory_arm(_mapping(arms["trajectory_imf"], "arms.trajectory_imf"))

    section = _mapping(protocol["exploratory_ablations"], "exploratory_ablations")
    _expect_keys(
        section,
        {"confirmatory_arm_count_remains_two", "may_support_primary_superiority_claim", "ablations"},
        "exploratory_ablations",
    )
    _bool(section["confirmatory_arm_count_remains_two"], True, "exploratory arm count boundary")
    _bool(section["may_support_primary_superiority_claim"], False, "exploratory claim boundary")
    ablations = _mapping(section["ablations"], "exploratory_ablations.ablations")
    if set(ablations) != set(EXPLORATORY_ABLATIONS):
        raise ValueError("exploratory ablations must be the three prespecified one-factor controls")
    expected = {
        "shared_time_only": {
            "base_arm": "trajectory_imf",
            "change_only": "replace_independent_per_target_query_intervals_with_one_shared_query_interval_per_sequence_while_preserving_the_three_way_context_mixture_and_fixed_context_JVP",
            "purpose": "isolate_per_target_time_heterogeneity_as_one_factor",
        },
        "clean_context_only": {
            "base_arm": "trajectory_imf",
            "change_only": "set_all_teacher_history_exposure_times_to_zero_while_preserving_independent_per_target_query_intervals_and_the_fixed_context_JVP",
            "purpose": "isolate_teacher_history_corruption_supervision_as_one_factor",
        },
        "joint_diagonal_context_jvp_negative_control": {
            "base_arm": "trajectory_imf",
            "change_only": "rebuild_causal_prefix_context_inside_a_whole_sequence_diagonal_JVP_while_preserving_all_other_settings",
            "exposed_unwanted_terms": "for_target_k_sum_over_j_less_than_k_of_partial_u_over_partial_c_k_times_partial_c_k_over_partial_(z_j,t_j)",
            "purpose": "expose_unwanted_prefix_cross_terms_that_are_absent_from_the_fixed_context_partial_JVP",
        },
    }
    _equal(dict(ablations), expected, "exploratory_ablations.ablations")


def _validate_shared_controls(protocol: Mapping[str, Any]) -> None:
    shared = _mapping(protocol["shared_world_model"], "shared_world_model")
    _expect_keys(
        shared,
        {
            "architecture",
            "observation_modality",
            "trunk",
            "causal_action_alignment",
            "same_causal_target",
            "same_action_inputs",
            "objective_randomness",
            "same_model_trunk",
            "same_latent_representation",
            "same_decoder_reward_continuation_heads",
            "same_representation_objectives",
            "same_optimizer_and_hyperparameters",
            "same_data_and_minibatch_order_when_budget_allows",
            "objective_required_head_exception_only",
            "world_model_frozen_during_primary_actor_training",
            "context_protocol",
            "optimization",
            "end_to_end_sensitivity",
            "actor_critic",
        },
        "shared_world_model",
    )
    _equal(shared["architecture"], "causal_recurrent_state_space_model", "shared architecture")
    _equal(shared["observation_modality"], "state", "shared observation modality")
    trunk = _mapping(shared["trunk"], "shared_world_model.trunk")
    _expect_keys(
        trunk,
        {
            "encoder_width",
            "deterministic_state_size",
            "stochastic_state_size",
            "transition_width",
            "recurrent_signal_coordinate_slots",
        },
        "shared_world_model.trunk",
    )
    for key, value in trunk.items():
        _positive_int(value, f"shared_world_model.trunk.{key}")
    _equal(
        trunk["recurrent_signal_coordinate_slots"],
        2,
        "shared recurrent signal-coordinate slots",
    )
    _equal(
        shared["causal_action_alignment"],
        "action_t_conditions_prediction_of_observation_t_plus_1",
        "shared causal action alignment",
    )
    for key in (
        "same_causal_target",
        "same_action_inputs",
        "same_model_trunk",
        "same_latent_representation",
        "same_decoder_reward_continuation_heads",
        "same_representation_objectives",
        "same_optimizer_and_hyperparameters",
        "same_data_and_minibatch_order_when_budget_allows",
        "objective_required_head_exception_only",
        "world_model_frozen_during_primary_actor_training",
    ):
        _bool(shared[key], True, f"shared_world_model.{key}")
    objective_randomness = _mapping(
        shared["objective_randomness"], "shared_world_model.objective_randomness"
    )
    _expect_keys(
        objective_randomness,
        {
            "same_base_gaussian_samples_within_paired_minibatch",
            "paired_canonical_uniforms_by_token",
            "objective_specific_deterministic_time_transform",
            "identical_schedule_distribution_required",
            "shortcut_schedule",
            "trajectory_schedule",
        },
        "shared_world_model.objective_randomness",
    )
    for key in (
        "same_base_gaussian_samples_within_paired_minibatch",
        "paired_canonical_uniforms_by_token",
        "objective_specific_deterministic_time_transform",
    ):
        _bool(objective_randomness[key], True, f"objective_randomness.{key}")
    _bool(
        objective_randomness["identical_schedule_distribution_required"],
        False,
        "objective_randomness.identical_schedule_distribution_required",
    )
    _equal(
        objective_randomness["shortcut_schedule"],
        "discrete_power_of_two_grid",
        "objective_randomness.shortcut_schedule",
    )
    _equal(
        objective_randomness["trajectory_schedule"],
        "continuous_R_Q_tau_and_context_pattern",
        "objective_randomness.trajectory_schedule",
    )
    context = _mapping(shared["context_protocol"], "shared_world_model.context_protocol")
    _expect_keys(
        context,
        {
            "primary_training_start_context",
            "primary_rollout_context",
            "same_primary_context_for_both_arms",
            "extra_context_noise_during_primary_evaluation",
            "dreamer4_context_corruption_status",
            "dreamer4_context_corruption_track",
            "dreamer4_context_corruption_may_support_primary_claim",
        },
        "shared_world_model.context_protocol",
    )
    _equal(
        context["primary_training_start_context"],
        "clean_posterior_context_from_replay_after_burn_in",
        "primary training start context",
    )
    _equal(
        context["primary_rollout_context"],
        "clean_observed_prefix_then_unmodified_generated_history",
        "primary rollout context",
    )
    _bool(context["same_primary_context_for_both_arms"], True, "shared primary context")
    _bool(context["extra_context_noise_during_primary_evaluation"], False, "primary context noise")
    _equal(context["dreamer4_context_corruption_status"], "paper_implementation_ambiguity", "D4 context ambiguity")
    _equal(context["dreamer4_context_corruption_track"], "optional_nonconfirmatory_sensitivity", "D4 context sensitivity")
    _bool(context["dreamer4_context_corruption_may_support_primary_claim"], False, "D4 context claim role")
    optimization = _mapping(shared["optimization"], "shared_world_model.optimization")
    _expect_keys(
        optimization,
        {
            "optimizer",
            "gradient_clipping",
            "weight_decay",
            "world_model_ema_enabled",
            "world_model_ema_use",
            "slow_critic_ema_is_not_world_model_ema",
            "shortcut_bootstrap_teacher",
        },
        "shared_world_model.optimization",
    )
    _equal(optimization["optimizer"], "library_custom_global_norm_clipped_Adam", "shared optimizer")
    _equal(optimization["gradient_clipping"], "global_norm_before_Adam_moment_updates", "shared gradient clipping")
    _equal(_finite(optimization["weight_decay"], "shared weight decay"), 0.0, "shared weight decay")
    _bool(optimization["world_model_ema_enabled"], True, "world-model EMA")
    _equal(
        optimization["world_model_ema_use"],
        "shortcut_bootstrap_teacher_only_not_evaluation_or_actor_rollout",
        "world-model EMA use",
    )
    _bool(optimization["slow_critic_ema_is_not_world_model_ema"], True, "slow critic distinction")
    _equal(
        optimization["shortcut_bootstrap_teacher"],
        "stop_gradient_of_ema_parameters_decay_0.999",
        "shortcut bootstrap teacher",
    )
    sensitivity = _mapping(shared["end_to_end_sensitivity"], "end_to_end_sensitivity")
    _expect_keys(sensitivity, {"enabled", "label", "may_support_primary_superiority_claim"}, "end_to_end")
    _bool(sensitivity["enabled"], False, "end_to_end_sensitivity.enabled")
    _equal(sensitivity["label"], "optional_nonconfirmatory_sensitivity", "end_to_end label")
    _bool(sensitivity["may_support_primary_superiority_claim"], False, "end_to_end claim boundary")
    actor = _mapping(shared["actor_critic"], "shared_world_model.actor_critic")
    _expect_keys(
        actor,
        {
            "algorithm",
            "same_initialization_within_paired_cell",
            "same_optimizer_and_hyperparameters",
            "same_imagination_horizon",
            "same_value_targets",
            "same_return_normalization",
        },
        "shared_world_model.actor_critic",
    )
    _equal(actor["algorithm"], "shared_imagination_actor_critic", "actor algorithm")
    for key in set(actor) - {"algorithm"}:
        _bool(actor[key], True, f"shared_world_model.actor_critic.{key}")

    matching = _mapping(protocol["parameter_matching"], "parameter_matching")
    _expect_keys(
        matching,
        {
            "primary_control",
            "report_total_parameters",
            "report_active_parameters",
            "active_parameter_definition",
            "inactive_objective_heads_excluded_from_active_count",
            "unused_arm_specific_heads_not_instantiated",
            "report_compiled_forward_flops",
            "report_compiled_forward_and_backward_train_flops",
            "report_compiled_inference_flops_per_transition",
            "objective_required_heads_may_differ",
            "relative_active_parameter_gap_trigger",
            "triggered_sensitivity",
            "triggered_sensitivity_hidden_width_candidates",
            "triggered_sensitivity_selection_rule",
            "triggered_sensitivity_claim_role",
        },
        "parameter_matching",
    )
    _equal(matching["primary_control"], "identical_shared_trunk_dimensions", "parameter primary control")
    for key in (
        "report_total_parameters",
        "report_active_parameters",
        "inactive_objective_heads_excluded_from_active_count",
        "unused_arm_specific_heads_not_instantiated",
        "report_compiled_forward_flops",
        "report_compiled_forward_and_backward_train_flops",
        "report_compiled_inference_flops_per_transition",
        "objective_required_heads_may_differ",
    ):
        _bool(matching[key], True, f"parameter_matching.{key}")
    _equal(
        matching["active_parameter_definition"],
        "instantiated_parameters_reachable_by_a_structural_gradient_path_from_the_compiled_arm_loss",
        "active parameter definition",
    )
    trigger = _unit_interval(
        matching["relative_active_parameter_gap_trigger"],
        "parameter_matching.relative_active_parameter_gap_trigger",
        open_interval=True,
    )
    _equal(trigger, 0.02, "parameter gap trigger")
    _equal(
        matching["triggered_sensitivity"],
        "prespecified_joint_hidden_width_parameter_match_without_outcome_access",
        "parameter sensitivity",
    )
    _equal(
        matching["triggered_sensitivity_hidden_width_candidates"],
        [240, 248, 256, 264, 272],
        "parameter sensitivity width candidates",
    )
    _equal(
        matching["triggered_sensitivity_selection_rule"],
        "enumerate_all_25_shortcut_x_trajectory_width_pairs_then_minimize_relative_active_world_model_parameter_gap_then_sum_absolute_width_deviation_from_256_then_sum_compiler_train_flops_then_lexicographic_shortcut_width_trajectory_width_without_outcome_access",
        "parameter sensitivity selection rule",
    )
    _equal(matching["triggered_sensitivity_claim_role"], "supporting_not_primary", "parameter role")


def _validate_canonical_executable_config(protocol: Mapping[str, Any]) -> None:
    executable = _mapping(protocol["canonical_executable_config"], "canonical_executable_config")
    _expect_keys(
        executable,
        {
            "contract",
            "task_shape_fields",
            "dreamer_config_non_task_shape_common",
            "dreamer_config_arm_values",
            "candidate_override_mapping",
            "resolution_contract",
            "training_batch",
            "precision",
            "optimizer_semantics",
            "loss_semantics",
            "primary_actor_execution",
        },
        "canonical_executable_config",
    )
    _equal(
        executable["contract"],
        "every_DreamerConfig_field_except_task_resolved_observation_shape_and_action_dim_is_explicit_and_no_library_default_may_be_inferred",
        "executable config contract",
    )
    _equal(
        executable["task_shape_fields"],
        list(DREAMER_CONFIG_TASK_SHAPE_FIELDS),
        "DreamerConfig task-shape fields",
    )
    common = _mapping(
        executable["dreamer_config_non_task_shape_common"],
        "canonical_executable_config.dreamer_config_non_task_shape_common",
    )
    arm_specific = {"prior", "imf_trajectory_enabled", "shortcut_training_k_max"}
    expected_common = set(DREAMER_CONFIG_NON_TASK_SHAPE_FIELDS) - arm_specific
    _expect_keys(common, expected_common, "DreamerConfig common fields")
    _equal(common, FROZEN_DREAMER_CONFIG_COMMON, "frozen DreamerConfig common values")

    arm_values = _mapping(executable["dreamer_config_arm_values"], "DreamerConfig arm values")
    if set(arm_values) != set(ARM_ORDER):
        raise ValueError("DreamerConfig arm values must follow canonical arm order")
    shortcut = _mapping(arm_values["shortcut_forcing"], "DreamerConfig shortcut arm values")
    trajectory = _mapping(arm_values["trajectory_imf"], "DreamerConfig trajectory arm values")
    _expect_keys(shortcut, arm_specific, "DreamerConfig shortcut arm values")
    _expect_keys(trajectory, arm_specific, "DreamerConfig trajectory arm values")
    _equal(shortcut["prior"], "shortcut", "shortcut DreamerConfig prior")
    _bool(shortcut["imf_trajectory_enabled"], False, "shortcut trajectory mode")
    selection = _mapping(shortcut["shortcut_training_k_max"], "shortcut K_max resolution")
    _expect_keys(selection, {"value_source", "allowed_values"}, "shortcut K_max resolution")
    _equal(selection["value_source"], "immutable_pilot_selection_manifest", "shortcut K_max source")
    _equal(selection["allowed_values"], [4, 8, 16], "shortcut K_max allowed values")
    _equal(trajectory, {"prior": "imf", "imf_trajectory_enabled": True, "shortcut_training_k_max": None}, "trajectory DreamerConfig arm values")

    _equal(
        executable["candidate_override_mapping"],
        {
            "objective_loss_scale": "prior_scale",
            "shortcut_forcing.training_k_max": "shortcut_training_k_max",
            "trajectory_imf.imf_boundary_fraction": "imf_boundary_fraction",
        },
        "candidate override mapping",
    )
    resolution = _mapping(executable["resolution_contract"], "config resolution contract")
    _expect_keys(
        resolution,
        {
            "pilot_candidate_configs_are_complete_and_executable",
            "confirmatory_execution_requires_immutable_selection_manifest",
            "resolved_config_must_contain_exactly_all_83_non_task_shape_DreamerConfig_fields",
            "selected_values_are_substituted_before_DreamerConfig_construction",
            "unresolved_placeholders_or_unknown_keys_are_fatal",
            "resolved_config_and_sha256_are_retained_per_run",
        },
        "config resolution contract",
    )
    for key, value in resolution.items():
        _bool(value, True, f"config resolution contract.{key}")

    batch = _mapping(executable["training_batch"], "canonical training batch")
    _expect_keys(
        batch,
        {"batch_size", "sequence_length", "burn_in", "loss_bearing_steps_per_sequence", "sampling", "short_episode_handling"},
        "canonical training batch",
    )
    _equal(batch["batch_size"], 32, "training batch size")
    _equal(batch["sequence_length"], 32, "training sequence length")
    _equal(batch["burn_in"], 8, "training burn in")
    _equal(batch["loss_bearing_steps_per_sequence"], 24, "loss-bearing sequence steps")
    _equal(common["burn_in"], batch["burn_in"], "DreamerConfig/training batch burn in")
    _equal(batch["sampling"], "uniform_contiguous_train_episode_chunks_with_reset_mask_and_no_train_test_crossing", "batch sampling")
    _equal(batch["short_episode_handling"], "right_pad_and_zero_loss_mask", "short-episode handling")

    precision = _mapping(executable["precision"], "canonical precision")
    _equal(
        precision,
        {
            "parameter_dtype": "float32",
            "activation_dtype": "float32",
            "optimizer_state_dtype": "float32",
            "loss_accumulation_dtype": "float32",
            "mixed_precision": False,
            "jax_enable_x64": False,
        },
        "canonical precision",
    )
    optimizer = _mapping(executable["optimizer_semantics"], "optimizer semantics")
    _expect_keys(
        optimizer,
        {"implementation", "gradient_transform", "update", "weight_decay", "world_model_ema", "slow_critic_update"},
        "optimizer semantics",
    )
    _equal(optimizer["implementation"], "imf_dreamer_jax.optim.adam_update", "optimizer implementation")
    _equal(optimizer["gradient_transform"], "global_norm_clip_at_DreamerConfig.grad_clip_before_first_and_second_moment_updates", "optimizer clipping")
    _equal(optimizer["update"], "bias_corrected_Adam_parameter_minus_lr_mhat_over_(sqrt(vhat)+epsilon)", "Adam update")
    _equal(optimizer["weight_decay"], 0.0, "optimizer weight decay")
    _equal(
        optimizer["world_model_ema"],
        "shortcut_bootstrap_teacher_only_decay_0.999",
        "optimizer world-model EMA",
    )
    _equal(optimizer["slow_critic_update"], "slow=(1-slow_critic_fraction)*slow+slow_critic_fraction*online_after_each_critic_update", "slow critic update")

    losses = _mapping(executable["loss_semantics"], "loss semantics")
    _expect_keys(losses, {"world_model_total", "trajectory_prior", "trajectory_boundary_velocity_supervision", "shortcut_prior", "behavior_prior", "critic", "actor"}, "loss semantics")
    _equal(losses["behavior_prior"], "disabled_behavior_kl_scale_zero_no_behavior_snapshot_passed", "behavior prior")
    _equal(common["behavior_kl_scale"], 0.0, "disabled behavior-prior scale")
    _equal(common["overshooting_scale"], 0.0, "disabled legacy overshooting scale")
    _equal(common["imf_endpoint_scale"], 0.0, "disabled endpoint loss")
    _equal(common["imf_shortcut_scale"], 0.0, "disabled legacy iMF shortcut loss")
    _bool(common["imf_boundary_velocity_supervision"], True, "trajectory boundary supervision")
    _equal(common["actor_gradient"], "reinforce", "actor gradient")
    _equal(common["critic_bins"], 1, "critic bins")

    actor = _mapping(executable["primary_actor_execution"], "primary actor execution")
    _expect_keys(
        actor,
        {"world_model_parameters", "start_context", "start_count_per_actor_update", "imagination_horizon", "sampler_by_arm", "behavior_prior", "real_environment_evaluation_action", "extra_generated_history_corruption"},
        "primary actor execution",
    )
    _equal(actor["world_model_parameters"], "frozen_final_prespecified_checkpoint", "actor frozen world model")
    _equal(actor["start_context"], "one_uniform_post_burn_in_posterior_state_per_replay_sequence", "actor start context")
    _equal(actor["start_count_per_actor_update"], batch["batch_size"], "actor start count")
    _equal(actor["imagination_horizon"], common["imagination_horizon"], "actor horizon")
    _equal(actor["sampler_by_arm"], {"shortcut_forcing": "stochastic_4_step_shortcut_sampler_without_output_clipping", "trajectory_imf": "stochastic_1_step_iMF_sampler"}, "actor samplers")
    _equal(actor["behavior_prior"], "disabled", "actor behavior prior")
    _equal(actor["real_environment_evaluation_action"], "deterministic_tanh_of_actor_mean", "actor evaluation action")
    _bool(actor["extra_generated_history_corruption"], False, "actor generated-history corruption")


def _validate_compute_accounting(protocol: Mapping[str, Any]) -> None:
    accounting = _mapping(protocol["compute_accounting"], "compute_accounting")
    _expect_keys(
        accounting,
        {"compiler_api", "counting_unit", "world_model_update", "actor_critic_update", "static_signature_record", "hlo_retention", "walltime_protocol", "equal_flop_allocation_costs"},
        "compute_accounting",
    )
    _equal(accounting["compiler_api"], "jax_jit_lower_compile_cost_analysis", "compute compiler API")
    _equal(accounting["counting_unit"], "compiler_reported_flops_for_the_complete_compiled_update", "compute counting unit")
    world = _mapping(accounting["world_model_update"], "world-model compute")
    _expect_keys(world, {"compiled_entrypoint", "include_forward_and_backward", "include_gradient_global_norm_clip_and_Adam_update", "trajectory_imf_include_primal_and_tangent_work_of_JVP", "shortcut_include_clean_target_and_both_sequential_half_step_bootstrap_calls", "exclude_host_input_pipeline"}, "world-model compute")
    _equal(world["compiled_entrypoint"], "jit_train_world_model", "world-model compute entrypoint")
    for key in set(world) - {"compiled_entrypoint"}:
        _bool(world[key], True, f"world-model compute.{key}")
    actor = _mapping(accounting["actor_critic_update"], "actor-critic compute")
    _expect_keys(actor, {"compiled_entrypoint", "include_forward_and_backward", "include_actor_and_critic_losses_global_norm_clips_Adam_updates_and_slow_critic_update", "imagined_rollouts_per_update", "start_states_per_rollout", "horizon", "imagined_transitions_per_update_formula", "imagined_transitions_per_update", "prior_field_evaluations_per_transition", "prior_field_evaluations_per_update", "report_imagined_transitions", "report_prior_NFE", "report_full_compiled_forward_backward_FLOPs", "report_synchronized_walltime"}, "actor-critic compute")
    _equal(actor["compiled_entrypoint"], "jit_train_actor_critic", "actor compute entrypoint")
    _equal(actor["imagined_rollouts_per_update"], 2, "actor imagined rollouts")
    _equal(actor["start_states_per_rollout"], 32, "actor start states")
    _equal(actor["horizon"], 15, "actor compute horizon")
    _equal(actor["imagined_transitions_per_update_formula"], "2*batch_size*imagination_horizon", "actor transition formula")
    _equal(actor["imagined_transitions_per_update"], 960, "actor transitions per update")
    _equal(actor["prior_field_evaluations_per_transition"], {"shortcut_forcing": 4, "trajectory_imf": 1}, "actor NFE per transition")
    _equal(actor["prior_field_evaluations_per_update"], {"shortcut_forcing": 3840, "trajectory_imf": 960}, "actor NFE per update")
    for key in ("include_forward_and_backward", "include_actor_and_critic_losses_global_norm_clips_Adam_updates_and_slow_critic_update", "report_imagined_transitions", "report_prior_NFE", "report_full_compiled_forward_backward_FLOPs", "report_synchronized_walltime"):
        _bool(actor[key], True, f"actor-critic compute.{key}")
    signature = _mapping(accounting["static_signature_record"], "compute static signature")
    _expect_keys(signature, {"batch_size", "sequence_length", "burn_in", "imagination_start_shape", "observation_shape", "action_dim", "parameter_activation_optimizer_and_loss_dtype", "device_platform_model_and_count_recorded", "jax_and_xla_versions_recorded", "arm_task_budget_track_and_NFE_are_static_signature_keys"}, "compute static signature")
    _equal(signature["batch_size"], 32, "compute batch size")
    _equal(signature["sequence_length"], 32, "compute sequence length")
    _equal(signature["burn_in"], 8, "compute burn in")
    _equal(signature["imagination_start_shape"], "[32,160]", "compute imagination shape")
    _equal(signature["parameter_activation_optimizer_and_loss_dtype"], "float32", "compute precision")
    for key in ("device_platform_model_and_count_recorded", "jax_and_xla_versions_recorded", "arm_task_budget_track_and_NFE_are_static_signature_keys"):
        _bool(signature[key], True, f"compute signature.{key}")
    hlo = _mapping(accounting["hlo_retention"], "HLO retention")
    for key in ("retain_unoptimized_and_optimized_HLO_text", "retain_compiler_cost_analysis_JSON", "retain_SHA256_for_each_HLO_and_cost_record"):
        _bool(hlo.get(key), True, f"HLO retention.{key}")
    _expect_keys(hlo, {"retain_unoptimized_and_optimized_HLO_text", "retain_compiler_cost_analysis_JSON", "retain_SHA256_for_each_HLO_and_cost_record"}, "HLO retention")
    wall = _mapping(accounting["walltime_protocol"], "walltime protocol")
    _expect_keys(wall, {"block_until_ready_before_and_after_timed_region", "compile_time_excluded", "warmup_updates", "timed_updates", "summary", "same_exclusive_device_allocation"}, "walltime protocol")
    _bool(wall["block_until_ready_before_and_after_timed_region"], True, "walltime synchronization")
    _bool(wall["compile_time_excluded"], True, "walltime compile exclusion")
    _equal(wall["warmup_updates"], 10, "walltime warmups")
    _equal(wall["timed_updates"], 100, "walltime updates")
    _equal(wall["summary"], "median_seconds_per_update_and_interquartile_range", "walltime summary")
    _bool(wall["same_exclusive_device_allocation"], True, "walltime device allocation")
    allocation = _mapping(accounting["equal_flop_allocation_costs"], "equal-FLOP allocation costs")
    _expect_keys(allocation, {"world_model", "actor_critic", "matched_separately", "no_forward_only_or_NFE_proxy_allowed"}, "equal-FLOP allocation costs")
    _equal(allocation["world_model"], "full_compiled_world_model_update_forward_backward_FLOPs", "world-model allocation cost")
    _equal(allocation["actor_critic"], "full_compiled_actor_critic_update_forward_backward_FLOPs_including_all_imagined_transition_NFE", "actor allocation cost")
    _bool(allocation["matched_separately"], True, "separate compute matching")
    _bool(allocation["no_forward_only_or_NFE_proxy_allowed"], True, "compute proxy prohibition")


def _validate_data_and_randomness(protocol: Mapping[str, Any]) -> None:
    data = _mapping(protocol["data"], "data")
    _expect_keys(
        data,
        {
            "suite",
            "tasks",
            "task_registry",
            "task_shape_resolution",
            "action_repeat",
            "native_episode_limit",
            "confirmatory_replay_steps_per_task_world_seed",
            "collection_policy",
            "episode_split",
            "pairing",
        },
        "data",
    )
    _equal(data["suite"], "DeepMind_Control_Suite_state_observations", "data.suite")
    _equal(data["tasks"], list(TASKS), "data.tasks")
    registry = _mapping(data["task_registry"], "data.task_registry")
    expected_registry = {
        task: {"suite": "dm_control", "domain_name": domain, "task_name": name}
        for task, (domain, name) in DMC_TASK_REGISTRY.items()
    }
    _equal(registry, expected_registry, "data.task_registry")
    shape_resolution = _mapping(data["task_shape_resolution"], "data.task_shape_resolution")
    _equal(
        shape_resolution,
        {
            "observation_shape": "flattened_float32_observation_spec_in_sorted_mapping_key_order",
            "action_dim": "flattened_float32_action_spec",
            "record_resolved_shapes_in_run_manifest": True,
            "fail_if_environment_spec_differs_within_task": True,
        },
        "data.task_shape_resolution",
    )
    _equal(_positive_int(data["action_repeat"], "data.action_repeat"), 1, "data.action_repeat")
    _equal(_positive_int(data["native_episode_limit"], "data.native_episode_limit"), 1000, "episode limit")
    _equal(
        _positive_int(data["confirmatory_replay_steps_per_task_world_seed"], "data.confirmatory replay"),
        500000,
        "data.confirmatory replay",
    )
    policy = _mapping(data["collection_policy"], "data.collection_policy")
    _expect_keys(policy, {"kind", "uniform_probability", "smooth_probability"}, "collection policy")
    _equal(policy["kind"], "seeded_uniform_smooth_mixture", "collection policy kind")
    uniform = _unit_interval(policy["uniform_probability"], "collection uniform")
    smooth = _unit_interval(policy["smooth_probability"], "collection smooth")
    if not math.isclose(uniform + smooth, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("collection-policy probabilities must sum to one")
    _equal(uniform, 0.7, "collection uniform")
    split = _mapping(data["episode_split"], "data.episode_split")
    _expect_keys(
        split,
        {"unit", "train_fraction", "test_fraction", "train_test_disjoint", "normalization_statistics_from_train_only"},
        "episode split",
    )
    _equal(split["unit"], "whole_episode", "episode split unit")
    train = _unit_interval(split["train_fraction"], "train fraction", open_interval=True)
    test = _unit_interval(split["test_fraction"], "test fraction", open_interval=True)
    if not math.isclose(train + test, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("train and test fractions must sum to one")
    _equal(train, 0.8, "train fraction")
    _bool(split["train_test_disjoint"], True, "train_test_disjoint")
    _bool(split["normalization_statistics_from_train_only"], True, "train-only normalization")
    pairing = _mapping(data["pairing"], "data.pairing")
    _expect_keys(
        pairing,
        {
            "dataset_content_addressed",
            "identical_dataset_digest_across_arms_and_budget_tracks",
            "identical_episode_split_across_arms_and_budget_tracks",
            "identical_evaluation_starts_across_arms_and_nfe_points",
            "equal_update_track_identical_minibatch_indices",
            "equal_flop_track_common_minibatch_prefix",
        },
        "data.pairing",
    )
    for key, value in pairing.items():
        _bool(value, True, f"data.pairing.{key}")

    randomness = _mapping(protocol["randomness"], "randomness")
    _expect_keys(
        randomness,
        {
            "world_model_seeds",
            "actor_seeds_nested_within_world_model_seed",
            "pilot_world_model_seeds",
            "pilot_actor_seeds_nested_within_world_model_seed",
            "seed_derivation",
            "independent_unit",
            "paired_common_random_numbers",
        },
        "randomness",
    )
    _equal(_unique_nonnegative_ints(randomness["world_model_seeds"], "world seeds"), WORLD_MODEL_SEEDS, "world seeds")
    _equal(
        _unique_nonnegative_ints(randomness["actor_seeds_nested_within_world_model_seed"], "actor seeds"),
        ACTOR_SEEDS,
        "actor seeds",
    )
    _equal(
        _unique_nonnegative_ints(randomness["pilot_world_model_seeds"], "pilot world seeds"),
        PILOT_WORLD_MODEL_SEEDS,
        "pilot world seeds",
    )
    _equal(
        _unique_nonnegative_ints(
            randomness["pilot_actor_seeds_nested_within_world_model_seed"], "pilot actor seeds"
        ),
        PILOT_ACTOR_SEEDS,
        "pilot actor seeds",
    )
    if set(WORLD_MODEL_SEEDS) & set(PILOT_WORLD_MODEL_SEEDS):
        raise ValueError("pilot and confirmatory world-model seeds must be disjoint")
    if set(ACTOR_SEEDS) & set(PILOT_ACTOR_SEEDS):
        raise ValueError("pilot and confirmatory actor seeds must be disjoint")
    _equal(
        randomness["seed_derivation"],
        "sha256_namespace_then_32_bit_numpy_or_environment_seed_and_64_bit_JAX_key_with_update_fold_in",
        "seed derivation",
    )
    _equal(randomness["independent_unit"], "task_x_world_model_seed", "independent unit")
    expected_common = [
        "dataset_collection",
        "episode_split",
        "minibatch_indices",
        "model_initialization_for_shared_parameters",
        "actor_initialization",
        "base_gaussian_noise",
        "canonical_schedule_uniforms",
        "evaluation_starts",
        "environment_evaluation_noise",
    ]
    _equal(randomness["paired_common_random_numbers"], expected_common, "paired common random numbers")


def _validate_hyperparameter_policy(protocol: Mapping[str, Any]) -> None:
    policy = _mapping(protocol["hyperparameter_policy"], "hyperparameter_policy")
    _expect_keys(
        policy,
        {
            "primary_non_objective_hyperparameters_shared_across_arms",
            "objective_specific_hyperparameters_may_differ",
            "selection_profile",
            "selection_tasks",
            "selection_world_model_seeds",
            "selection_actor_seeds_nested_within_world_model_seed",
            "candidate_evaluations_per_arm",
            "equal_candidate_evaluations_per_arm",
            "selection_budget_track",
            "selection_track_frozen_before_pilot_outcomes",
            "search",
            "selection_rule",
            "ranking_semantics",
            "selection_aggregation",
            "selected_candidate_materialization",
            "pilot_cells_disjoint_from_confirmatory_cells",
            "confirmatory_data_checkpoints_and_outcomes_sealed_until_selection_frozen",
            "no_task_specific_confirmatory_tuning",
            "all_trials_and_selection_manifest_retained",
        },
        "hyperparameter_policy",
    )
    for key in (
        "primary_non_objective_hyperparameters_shared_across_arms",
        "objective_specific_hyperparameters_may_differ",
        "equal_candidate_evaluations_per_arm",
        "pilot_cells_disjoint_from_confirmatory_cells",
        "confirmatory_data_checkpoints_and_outcomes_sealed_until_selection_frozen",
        "no_task_specific_confirmatory_tuning",
        "all_trials_and_selection_manifest_retained",
        "selection_track_frozen_before_pilot_outcomes",
    ):
        _bool(policy[key], True, f"hyperparameter_policy.{key}")
    _equal(policy["selection_profile"], "pilot", "HPO selection profile")
    _equal(policy["selection_tasks"], list(PILOT_TASKS), "HPO selection tasks")
    _equal(
        _unique_nonnegative_ints(policy["selection_world_model_seeds"], "HPO world seeds"),
        PILOT_WORLD_MODEL_SEEDS,
        "HPO world seeds",
    )
    _equal(
        _unique_nonnegative_ints(
            policy["selection_actor_seeds_nested_within_world_model_seed"], "HPO actor seeds"
        ),
        PILOT_ACTOR_SEEDS,
        "HPO actor seeds",
    )
    profile = protocol["profiles"]["pilot"]
    _equal(policy["selection_tasks"], profile["tasks"], "HPO/profile pilot tasks")
    _equal(policy["selection_world_model_seeds"], profile["world_model_seeds"], "HPO/profile world seeds")
    _equal(
        policy["selection_actor_seeds_nested_within_world_model_seed"],
        profile["actor_seeds_nested_within_world_model_seed"],
        "HPO/profile actor seeds",
    )
    if not set(policy["selection_tasks"]).isdisjoint(TASKS):
        raise ValueError("HPO task names must be disjoint from the six confirmatory task names")
    candidate_count = _positive_int(
        policy["candidate_evaluations_per_arm"], "HPO candidate evaluations per arm"
    )
    _equal(policy["selection_budget_track"], "equal_updates", "HPO selection budget track")
    _equal(candidate_count, PILOT_CANDIDATES_PER_ARM, "HPO candidate evaluations per arm")
    search = _mapping(policy["search"], "hyperparameter_policy.search")
    _expect_keys(search, {"method", *ARM_ORDER}, "hyperparameter_policy.search")
    _equal(search["method"], "exhaustive_frozen_cartesian_grid", "HPO search method")
    expected_scale_grid = [0.25, 0.5, 1.0, 2.0]
    shortcut = _mapping(search["shortcut_forcing"], "shortcut HPO search space")
    _expect_keys(
        shortcut,
        {"training_k_max", "objective_loss_scale", "cartesian_product", "candidate_count"},
        "shortcut HPO search space",
    )
    _equal(shortcut["training_k_max"], [4, 8, 16], "shortcut HPO K_max grid")
    _equal(shortcut["objective_loss_scale"], expected_scale_grid, "shortcut HPO scale grid")
    _bool(shortcut["cartesian_product"], True, "shortcut HPO Cartesian product")
    _equal(shortcut["candidate_count"], candidate_count, "shortcut HPO candidate count")
    trajectory = _mapping(search["trajectory_imf"], "trajectory HPO search space")
    _expect_keys(
        trajectory,
        {"imf_boundary_fraction", "objective_loss_scale", "cartesian_product", "candidate_count"},
        "trajectory HPO search space",
    )
    _equal(trajectory["imf_boundary_fraction"], [0.25, 0.5, 0.75], "trajectory boundary grid")
    _equal(trajectory["objective_loss_scale"], expected_scale_grid, "trajectory HPO scale grid")
    _bool(trajectory["cartesian_product"], True, "trajectory HPO Cartesian product")
    _equal(trajectory["candidate_count"], candidate_count, "trajectory HPO candidate count")
    if len(shortcut["training_k_max"]) * len(shortcut["objective_loss_scale"]) != candidate_count:
        raise ValueError("shortcut HPO Cartesian grid does not consume the frozen budget")
    if len(trajectory["imf_boundary_fraction"]) * len(trajectory["objective_loss_scale"]) != candidate_count:
        raise ValueError("trajectory HPO Cartesian grid does not consume the frozen budget")
    _equal(
        policy["selection_rule"],
        "within_equal_updates_track_minimize_sum_of_pilot_ranks_for_rollout_auc_and_negative_nested_actor_return_then_lower_full_forward_backward_train_flops_then_lexicographic_config_hash",
        "HPO selection rule",
    )
    _equal(
        policy["ranking_semantics"],
        "ordinal_ranks_1_to_12_per_metric_with_ties_broken_by_lexicographic_config_hash_before_rank_sum",
        "HPO ranking semantics",
    )
    aggregation = _mapping(policy["selection_aggregation"], "HPO selection aggregation")
    expected_aggregation = {
        "unit_order": "protocol_selection_task_order_then_protocol_selection_world_model_seed_order",
        "iqm_definition": "integral_of_empirical_quantile_function_over_0.25_to_0.75_divided_by_0.5_with_fractional_endpoint_weights",
        "rollout_auc": "interquartile_mean_over_6_task_x_world_model_seed_units",
        "actor_return": "arithmetic_mean_over_evaluation_episodes_within_actor_seed_then_arithmetic_mean_over_2_nested_actor_seeds_within_task_x_world_model_seed_then_interquartile_mean_over_6_task_x_world_model_seed_units",
        "compiler_flops": "arithmetic_mean_over_6_task_x_world_model_seed_units",
    }
    _expect_keys(aggregation, set(expected_aggregation), "HPO selection aggregation")
    for key, value in expected_aggregation.items():
        _equal(aggregation[key], value, f"HPO selection aggregation {key}")
    _equal(
        policy["selected_candidate_materialization"],
        "write_immutable_pilot_selection_manifest_then_resolve_complete_per_arm_DreamerConfig_before_unsealing_confirmatory_data",
        "HPO selection materialization",
    )


def _validate_budget_tracks(protocol: Mapping[str, Any]) -> None:
    _equal(protocol["budget_track_order"], list(BUDGET_TRACK_ORDER), "budget_track_order")
    tracks = _mapping(protocol["budget_tracks"], "budget_tracks")
    if set(tracks) != set(BUDGET_TRACK_ORDER):
        raise ValueError("budget tracks must contain both frozen tracks in canonical order")
    updates = _mapping(tracks["equal_updates"], "budget_tracks.equal_updates")
    _expect_keys(
        updates,
        {
            "claim_role",
            "matching_unit",
            "world_model_updates_equal",
            "actor_updates_equal",
            "examples_per_update_equal",
            "exact_minibatch_indices_equal",
            "report_realized_compiler_flops",
            "report_wall_clock",
            "report_parameter_counts",
        },
        "equal_updates",
    )
    _equal(updates["claim_role"], "primary_objective_isolation_track", "equal_updates claim role")
    _equal(updates["matching_unit"], "optimizer_updates_and_data_exposure", "equal_updates matching unit")
    for key in set(updates) - {"claim_role", "matching_unit"}:
        _bool(updates[key], True, f"budget_tracks.equal_updates.{key}")

    flops = _mapping(tracks["equal_compiler_flops"], "budget_tracks.equal_compiler_flops")
    _expect_keys(
        flops,
        {
            "claim_role",
            "matching_unit",
            "cost_source",
            "reference_arm",
            "allocation_rule",
            "relative_tolerance",
            "world_model_and_actor_budgets_matched_separately",
            "allocation_frozen_before_outcome_evaluation",
            "common_minibatch_prefix",
            "report_realized_updates",
            "report_realized_compiler_flops",
            "report_wall_clock",
            "report_parameter_counts",
        },
        "equal_compiler_flops",
    )
    _equal(flops["claim_role"], "required_compute_robustness_track", "equal FLOP claim role")
    _equal(flops["matching_unit"], "compiler_reported_full_forward_backward_update_flops", "equal FLOP unit")
    _equal(
        flops["cost_source"],
        "jax_jit_lower_compile_cost_analysis_of_complete_forward_and_backward_world_model_and_actor_critic_updates_including_JVP_bootstrap_optimizer_and_imagined_transition_NFE",
        "equal FLOP source",
    )
    _equal(flops["reference_arm"], "shortcut_forcing", "equal FLOP reference arm")
    _equal(
        flops["allocation_rule"],
        "nearest_nonnegative_integer_updates_not_exceeding_reference_flops_when_tied",
        "equal FLOP allocation rule",
    )
    tolerance = _unit_interval(flops["relative_tolerance"], "equal FLOP tolerance", open_interval=True)
    _equal(tolerance, 0.002, "equal FLOP tolerance")
    for key in (
        "world_model_and_actor_budgets_matched_separately",
        "allocation_frozen_before_outcome_evaluation",
        "common_minibatch_prefix",
        "report_realized_updates",
        "report_realized_compiler_flops",
        "report_wall_clock",
        "report_parameter_counts",
    ):
        _bool(flops[key], True, f"budget_tracks.equal_compiler_flops.{key}")


def _validate_profiles(protocol: Mapping[str, Any]) -> None:
    profiles = _mapping(protocol["profiles"], "profiles")
    if set(profiles) != set(PROFILE_ORDER):
        raise ValueError("profiles must be smoke, pilot, confirmatory in canonical order")
    expected = {
        "smoke": {
            "evidence_class": "engineering_smoke",
            "claim_eligible": False,
            "tasks": TASKS[:1],
            "world_model_seeds": WORLD_MODEL_SEEDS[:1],
            "actor_seeds": ACTOR_SEEDS[:1],
            "replay": 200,
            "windows": 2,
            "draws": 8,
            "episodes": 1,
            "updates": 2,
        },
        "pilot": {
            "evidence_class": "development_pilot",
            "claim_eligible": False,
            "tasks": PILOT_TASKS,
            "world_model_seeds": PILOT_WORLD_MODEL_SEEDS,
            "actor_seeds": PILOT_ACTOR_SEEDS,
            "replay": 50000,
            "windows": 64,
            "draws": 32,
            "episodes": 5,
            "updates": 10000,
        },
        "confirmatory": {
            "evidence_class": "confirmatory",
            "claim_eligible": True,
            "tasks": TASKS,
            "world_model_seeds": WORLD_MODEL_SEEDS,
            "actor_seeds": ACTOR_SEEDS,
            "replay": 500000,
            "windows": 256,
            "draws": 64,
            "episodes": 10,
            "updates": 100000,
        },
    }
    profile_keys = {
        "evidence_class",
        "claim_eligible",
        "tasks",
        "world_model_seeds",
        "actor_seeds_nested_within_world_model_seed",
        "replay_steps_per_task_world_seed",
        "rollout_windows_per_task_world_seed",
        "predictive_draws_per_window",
        "real_environment_evaluation_episodes",
        "budgets",
    }
    previous = {"replay": 0, "windows": 0, "draws": 0, "episodes": 0, "updates": 0}
    for name in PROFILE_ORDER:
        profile = _mapping(profiles[name], f"profiles.{name}")
        _expect_keys(profile, profile_keys, f"profiles.{name}")
        frozen = expected[name]
        _equal(profile["evidence_class"], frozen["evidence_class"], f"profiles.{name}.evidence_class")
        _bool(profile["claim_eligible"], frozen["claim_eligible"], f"profiles.{name}.claim_eligible")
        _equal(profile["tasks"], list(frozen["tasks"]), f"profiles.{name}.tasks")
        _equal(
            _unique_nonnegative_ints(profile["world_model_seeds"], f"profiles.{name}.world seeds"),
            frozen["world_model_seeds"],
            f"profiles.{name}.world seeds",
        )
        _equal(
            _unique_nonnegative_ints(
                profile["actor_seeds_nested_within_world_model_seed"], f"profiles.{name}.actor seeds"
            ),
            frozen["actor_seeds"],
            f"profiles.{name}.actor seeds",
        )
        numeric_fields = {
            "replay": "replay_steps_per_task_world_seed",
            "windows": "rollout_windows_per_task_world_seed",
            "draws": "predictive_draws_per_window",
            "episodes": "real_environment_evaluation_episodes",
        }
        for short, field in numeric_fields.items():
            value = _positive_int(profile[field], f"profiles.{name}.{field}")
            _equal(value, frozen[short], f"profiles.{name}.{field}")
            if value <= previous[short]:
                raise ValueError(f"profile {field} must increase strictly from smoke to confirmatory")
            previous[short] = value
        budgets = _mapping(profile["budgets"], f"profiles.{name}.budgets")
        if set(budgets) != set(BUDGET_TRACK_ORDER):
            raise ValueError(f"profiles.{name}.budgets must define both tracks")
        equal_updates = _mapping(budgets["equal_updates"], f"profiles.{name}.budgets.equal_updates")
        _expect_keys(equal_updates, {"world_model_updates", "actor_updates"}, "equal-update profile budget")
        equal_flops = _mapping(budgets["equal_compiler_flops"], f"profiles.{name}.budgets.equal_compiler_flops")
        _expect_keys(
            equal_flops,
            {"reference_world_model_updates", "reference_actor_updates"},
            "equal-FLOP profile budget",
        )
        for field in ("world_model_updates", "actor_updates"):
            value = _positive_int(equal_updates[field], f"profiles.{name}.budgets.equal_updates.{field}")
            _equal(value, frozen["updates"], f"profiles.{name}.budgets.equal_updates.{field}")
        for field in ("reference_world_model_updates", "reference_actor_updates"):
            value = _positive_int(equal_flops[field], f"profiles.{name}.budgets.equal_compiler_flops.{field}")
            _equal(value, frozen["updates"], f"profiles.{name}.budgets.equal_compiler_flops.{field}")
        if frozen["updates"] <= previous["updates"]:
            raise ValueError("profile update budgets must increase strictly")
        previous["updates"] = frozen["updates"]


def _validate_diagnostics_and_evaluation(protocol: Mapping[str, Any]) -> None:
    diagnostics = _mapping(protocol["diagnostics"], "diagnostics")
    _expect_keys(
        diagnostics,
        {
            "required_before_confirmatory_interpretation",
            "deterministic_transition_diagnostic",
            "genuinely_stochastic_transition_diagnostic",
            "diagnostic_budget_track",
            "diagnostic_scores_may_substitute_for_primary_outcomes",
            "diagnostic_thresholds_may_pass_or_fail_superiority_claim",
        },
        "diagnostics",
    )
    _equal(
        diagnostics["required_before_confirmatory_interpretation"],
        ["deterministic_transition_diagnostic", "genuinely_stochastic_transition_diagnostic"],
        "required diagnostics",
    )
    deterministic = _mapping(diagnostics["deterministic_transition_diagnostic"], "deterministic diagnostic")
    _expect_keys(
        deterministic,
        {
            "required",
            "purpose",
            "equation",
            "initial_state_distribution",
            "action_process",
            "episode_length",
            "training_episodes",
            "test_initial_conditions",
            "repeated_futures_per_identical_condition",
            "world_model_updates",
            "actor_updates",
            "estimands",
            "interpretation_thresholds",
            "threshold_role",
        },
        "deterministic diagnostic",
    )
    _bool(deterministic["required"], True, "deterministic diagnostic required")
    _equal(deterministic["purpose"], "detect_bias_and_compounding_error_without_aleatoric_ambiguity", "deterministic purpose")
    _equal(deterministic["equation"], "x_(t+1)=0.85*x_t+0.10*tanh(2*x_t)+0.20*a_t", "deterministic equation")
    _equal(deterministic["initial_state_distribution"], "x_0~Uniform(-1,1)", "deterministic initial state")
    _equal(deterministic["action_process"], "a_t=clip(0.6*sin(0.17*t+phi)+0.2*u_t,-1,1),phi~Uniform(0,2*pi),u_t~Uniform(-1,1)", "deterministic action process")
    for key, expected in {
        "episode_length": 64,
        "training_episodes": 4096,
        "test_initial_conditions": 512,
        "repeated_futures_per_identical_condition": 1,
        "world_model_updates": 20000,
        "actor_updates": 0,
    }.items():
        _equal(_positive_int(deterministic[key], f"deterministic {key}", minimum=0), expected, f"deterministic {key}")
    _equal(
        deterministic["estimands"],
        [
            "one_step_normalized_RMSE",
            "horizon_30_normalized_RMSE",
            "normalized_rollout_error_AUC",
            "horizon_30_to_horizon_1_error_ratio",
        ],
        "deterministic estimands",
    )
    _equal(
        deterministic["interpretation_thresholds"],
        {
            "one_step_normalized_RMSE_max": 0.05,
            "horizon_30_normalized_RMSE_max": 0.25,
            "horizon_30_to_horizon_1_error_ratio_max": 5.0,
        },
        "deterministic thresholds",
    )
    _equal(deterministic["threshold_role"], "engineering_interpretation_only_non_primary", "deterministic threshold role")
    stochastic = _mapping(diagnostics["genuinely_stochastic_transition_diagnostic"], "stochastic diagnostic")
    _expect_keys(
        stochastic,
        {
            "required",
            "purpose",
            "mechanisms",
            "equation",
            "initial_state_distribution",
            "action_process",
            "episode_length",
            "training_episodes",
            "test_conditions",
            "repeated_futures_per_identical_condition",
            "predictive_draws_per_condition",
            "oracle_reference_futures_per_condition",
            "evaluation_futures_per_condition",
            "rollout_horizons",
            "branch_probability_calibration_bins",
            "identical_condition_and_action_prefix",
            "world_model_updates",
            "actor_updates",
            "estimands",
            "interpretation_thresholds",
            "interpretation_pass_rule",
            "threshold_role",
        },
        "stochastic diagnostic",
    )
    _bool(stochastic["required"], True, "stochastic diagnostic required")
    _equal(stochastic["purpose"], "test_distributional_fidelity_and_calibration_under_aleatoric_branching", "stochastic purpose")
    _equal(stochastic["mechanisms"], ["exogenous_Bernoulli_branch", "exogenous_Gaussian_innovation"], "stochastic mechanisms")
    _equal(stochastic["equation"], "b_t~Bernoulli(sigmoid(2*x_t+0.5*a_t));eta_t~Normal(0,0.15^2);x_(t+1)=0.70*x_t+0.20*a_t+0.60*(2*b_t-1)+eta_t", "stochastic equation")
    _equal(stochastic["initial_state_distribution"], "x_0~Uniform(-1,1)", "stochastic initial state")
    _equal(stochastic["action_process"], deterministic["action_process"], "stochastic action process")
    for key, expected in {
        "episode_length": 64,
        "training_episodes": 8192,
        "test_conditions": 512,
        "repeated_futures_per_identical_condition": 64,
        "predictive_draws_per_condition": 64,
        "oracle_reference_futures_per_condition": 64,
        "evaluation_futures_per_condition": 64,
        "branch_probability_calibration_bins": 10,
        "world_model_updates": 40000,
        "actor_updates": 0,
    }.items():
        _equal(_positive_int(stochastic[key], f"stochastic {key}", minimum=0), expected, f"stochastic {key}")
    _bool(stochastic["identical_condition_and_action_prefix"], True, "stochastic identical prefix")
    _equal(stochastic["rollout_horizons"], [1, 2, 4, 8, 15, 30], "stochastic rollout horizons")
    _equal(
        stochastic["estimands"],
        [
            "energy_score_over_oracle_ratio_horizon_1",
            "energy_score_over_oracle_ratio_horizon_8",
            "energy_score_over_oracle_ratio_horizon_30",
            "central_90_percent_interval_coverage_horizon_1",
            "central_90_percent_interval_coverage_horizon_8",
            "central_90_percent_interval_coverage_horizon_30",
            "branch_probability_expected_calibration_error",
            "normalized_rollout_error_AUC",
        ],
        "stochastic estimands",
    )
    _equal(
        stochastic["interpretation_thresholds"],
        {
            "energy_score_over_oracle_ratio_horizon_1_max": 1.25,
            "energy_score_over_oracle_ratio_horizon_8_max": 1.25,
            "energy_score_over_oracle_ratio_horizon_30_max": 1.25,
            "central_90_percent_interval_coverage_horizon_1_min": 0.85,
            "central_90_percent_interval_coverage_horizon_1_max": 0.95,
            "central_90_percent_interval_coverage_horizon_8_min": 0.85,
            "central_90_percent_interval_coverage_horizon_8_max": 0.95,
            "central_90_percent_interval_coverage_horizon_30_min": 0.85,
            "central_90_percent_interval_coverage_horizon_30_max": 0.95,
            "branch_probability_expected_calibration_error_max": 0.05,
        },
        "stochastic thresholds",
    )
    _equal(
        stochastic["interpretation_pass_rule"],
        "all_registered_energy_score_ratios_at_horizons_1_8_30_are_at_most_their_maxima_and_all_registered_coverages_are_within_their_inclusive_minima_and_maxima_and_branch_probability_expected_calibration_error_is_at_most_its_maximum",
        "stochastic threshold pass rule",
    )
    _equal(stochastic["threshold_role"], "engineering_interpretation_only_non_primary", "stochastic threshold role")
    _equal(diagnostics["diagnostic_budget_track"], "equal_updates", "diagnostic budget track")
    _bool(diagnostics["diagnostic_scores_may_substitute_for_primary_outcomes"], False, "diagnostic substitution")
    _bool(diagnostics["diagnostic_thresholds_may_pass_or_fail_superiority_claim"], False, "diagnostic threshold claim role")

    evaluation = _mapping(protocol["evaluation"], "evaluation")
    _expect_keys(
        evaluation,
        {
            "world_model_checkpoint",
            "actor_checkpoint",
            "rollout_context_length",
            "rollout_horizons",
            "rollout_metric",
            "actor_metric",
            "nfe_frontier",
            "primary_nfe",
            "secondary_metrics",
            "paired_evaluation_starts_and_environment_noise",
            "raw_predictive_draws_retained",
            "raw_action_traces_retained",
        },
        "evaluation",
    )
    _equal(evaluation["world_model_checkpoint"], "final_prespecified_training_update", "world checkpoint")
    _equal(evaluation["actor_checkpoint"], "final_prespecified_training_update", "actor checkpoint")
    _positive_int(evaluation["rollout_context_length"], "rollout context length")
    _equal(evaluation["rollout_horizons"], list(ROLLOUT_HORIZONS), "rollout horizons")
    _equal(evaluation["nfe_frontier"], list(NFE_FRONTIER), "NFE frontier")
    _equal(evaluation["primary_nfe"], {"shortcut_forcing": 4, "trajectory_imf": 1}, "primary NFE")
    rollout = _mapping(evaluation["rollout_metric"], "evaluation.rollout_metric")
    _expect_keys(
        rollout,
        {
            "name",
            "direction",
            "free_running_after_context",
            "normalizer",
            "per_horizon_formula",
            "auc_formula",
            "within_task_world_seed_estimand",
            "cross_task_seed_aggregation",
            "aggregation",
        },
        "rollout metric",
    )
    _equal(rollout["name"], PRIMARY_METRICS[0], "rollout metric name")
    _equal(rollout["direction"], "lower_is_better", "rollout metric direction")
    _bool(rollout["free_running_after_context"], True, "rollout metric free running")
    _equal(
        rollout["normalizer"],
        "per_dimension_training_split_standard_deviation_clipped_below_1e-6",
        "rollout normalizer",
    )
    _equal(
        rollout["per_horizon_formula"],
        "e(h)=(1/(W*M*D))*sum_w,sum_m,sum_d[((xhat_(w,m,h,d)-x_(w,h,d))/max(training_split_std_d,1e-6))^2]",
        "rollout per-horizon formula",
    )
    _equal(
        rollout["auc_formula"],
        "AUC=(1/(h_last-h_first))*sum_i[0.5*(e(h_i)+e(h_(i+1)))*(h_(i+1)-h_i)]_for_h=[1,2,4,8,15,30]",
        "rollout AUC formula",
    )
    _equal(rollout["within_task_world_seed_estimand"], "one_AUC_after_averaging_all_prespecified_windows_draws_and_dimensions", "rollout unit estimand")
    _equal(rollout["cross_task_seed_aggregation"], "IQM_of_the_6_tasks_by_10_world_model_seeds_matrix", "rollout cross-unit aggregation")
    _equal(rollout["aggregation"], "normalized_trapezoidal_AUC_over_prespecified_native_step_horizons", "rollout aggregation")
    actor = _mapping(evaluation["actor_metric"], "evaluation.actor_metric")
    _expect_keys(
        actor,
        {
            "name",
            "direction",
            "world_model_frozen",
            "normalization",
            "episode_aggregation_within_actor_seed",
            "nested_actor_seed_aggregation",
            "matrix_construction",
            "task_seed_aggregation",
            "hierarchy",
        },
        "actor metric",
    )
    _equal(actor["name"], PRIMARY_METRICS[1], "actor metric name")
    _equal(actor["direction"], "higher_is_better", "actor metric direction")
    _bool(actor["world_model_frozen"], True, "actor metric frozen world model")
    _equal(actor["normalization"], "dmc_episodic_return_divided_by_1000", "actor normalization")
    _equal(actor["episode_aggregation_within_actor_seed"], "arithmetic_mean_of_10_normalized_episode_returns", "actor episode aggregation")
    _equal(actor["nested_actor_seed_aggregation"], "arithmetic_mean_of_the_3_actor_seed_episode_means_within_each_task_world_model_seed_arm_cell", "actor nested aggregation")
    _equal(actor["matrix_construction"], "assemble_the_nested_actor_means_as_a_6_task_by_10_world_model_seed_matrix_per_arm_and_budget_track", "actor matrix construction")
    _equal(actor["task_seed_aggregation"], "interquartile_mean_of_the_60_matrix_entries_with_symmetric_25_percent_trimming", "actor task-seed aggregation")
    _equal(actor["hierarchy"], "episode_to_actor_seed_mean_to_nested_actor_mean_to_6x10_matrix_to_IQM", "actor hierarchy")
    _equal(evaluation["secondary_metrics"], list(SECONDARY_METRICS), "evaluation.secondary_metrics")
    for key in (
        "paired_evaluation_starts_and_environment_noise",
        "raw_predictive_draws_retained",
        "raw_action_traces_retained",
    ):
        _bool(evaluation[key], True, f"evaluation.{key}")


def _validate_statistics(protocol: Mapping[str, Any]) -> None:
    statistics = _mapping(protocol["statistics"], "statistics")
    _expect_keys(
        statistics,
        {
            "independent_experimental_unit",
            "actor_seeds_are_nested_not_independent_replicates",
            "paired_contrasts_within_task_world_seed",
            "bootstrap",
            "multiplicity",
            "primary_contrasts",
            "practical_significance",
            "superiority_rule",
            "report_raw_seed_values",
            "report_final_checkpoint_only_for_primary_inference",
        },
        "statistics",
    )
    _equal(statistics["independent_experimental_unit"], "task_x_world_model_seed", "statistical unit")
    for key in (
        "actor_seeds_are_nested_not_independent_replicates",
        "paired_contrasts_within_task_world_seed",
        "report_raw_seed_values",
        "report_final_checkpoint_only_for_primary_inference",
    ):
        _bool(statistics[key], True, f"statistics.{key}")
    bootstrap = _mapping(statistics["bootstrap"], "statistics.bootstrap")
    _expect_keys(
        bootstrap,
        {
            "method",
            "resamples",
            "seed",
            "resample_world_model_seeds_within_task",
            "world_model_indices_drawn_per_task_per_replicate",
            "same_resampled_world_model_indices_applied_jointly_to_both_arms",
            "actor_seeds_resampled",
            "tasks_resampled",
            "retain_task_weights",
            "recompute_each_arm_6x10_matrix_IQM_then_the_paired_arm_difference_each_replicate",
            "two_sided_percentile_lower_quantile",
            "two_sided_percentile_upper_quantile",
        },
        "bootstrap",
    )
    _equal(bootstrap["method"], "task_stratified_paired_percentile_bootstrap", "bootstrap method")
    if _positive_int(bootstrap["resamples"], "bootstrap.resamples") < 10000:
        raise ValueError("confirmatory bootstrap requires at least 10000 resamples")
    _positive_int(bootstrap["seed"], "bootstrap.seed", minimum=0)
    _bool(bootstrap["resample_world_model_seeds_within_task"], True, "bootstrap seed resampling")
    _equal(bootstrap["world_model_indices_drawn_per_task_per_replicate"], len(WORLD_MODEL_SEEDS), "bootstrap draws per task")
    _bool(bootstrap["same_resampled_world_model_indices_applied_jointly_to_both_arms"], True, "bootstrap arm pairing")
    _bool(bootstrap["actor_seeds_resampled"], False, "bootstrap nested actors")
    _bool(bootstrap["tasks_resampled"], False, "bootstrap task resampling")
    _bool(bootstrap["retain_task_weights"], True, "bootstrap task weights")
    _bool(bootstrap["recompute_each_arm_6x10_matrix_IQM_then_the_paired_arm_difference_each_replicate"], True, "bootstrap IQM recomputation")
    multiplicity = _mapping(statistics["multiplicity"], "statistics.multiplicity")
    _expect_keys(
        multiplicity,
        {
            "family",
            "number_of_confirmatory_intervals",
            "familywise_confidence",
            "interval_confidence",
            "two_sided_tail_probability_each",
            "reported_percentiles",
            "correction",
        },
        "multiplicity",
    )
    _equal(multiplicity["family"], "two_primary_metrics_by_two_budget_tracks", "multiplicity family")
    _equal(multiplicity["number_of_confirmatory_intervals"], 4, "multiplicity interval count")
    family = _unit_interval(multiplicity["familywise_confidence"], "familywise confidence", open_interval=True)
    interval = _unit_interval(multiplicity["interval_confidence"], "interval confidence", open_interval=True)
    _equal(family, 0.95, "familywise confidence")
    expected_interval = 1.0 - (1.0 - family) / 4.0
    if not math.isclose(interval, expected_interval, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("interval confidence must be the prespecified Bonferroni value")
    tail = (1.0 - interval) / 2.0
    if not math.isclose(_unit_interval(multiplicity["two_sided_tail_probability_each"], "multiplicity tail"), tail, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("multiplicity tail probability disagrees with interval confidence")
    percentiles = multiplicity["reported_percentiles"]
    if (
        not isinstance(percentiles, list)
        or len(percentiles) != 2
        or not math.isclose(_finite(percentiles[0], "lower reported percentile"), 100.0 * tail, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(_finite(percentiles[1], "upper reported percentile"), 100.0 * (1.0 - tail), rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("reported percentiles disagree with interval confidence")
    if not math.isclose(_unit_interval(bootstrap["two_sided_percentile_lower_quantile"], "bootstrap lower quantile"), tail, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("bootstrap lower quantile disagrees with multiplicity correction")
    if not math.isclose(_unit_interval(bootstrap["two_sided_percentile_upper_quantile"], "bootstrap upper quantile"), 1.0 - tail, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("bootstrap upper quantile disagrees with multiplicity correction")
    _equal(multiplicity["correction"], "bonferroni", "multiplicity correction")
    contrasts = _mapping(statistics["primary_contrasts"], "statistics.primary_contrasts")
    if set(contrasts) != set(PRIMARY_METRICS):
        raise ValueError("primary contrasts must contain exactly both primary metrics")
    _equal(
        contrasts[PRIMARY_METRICS[0]],
        {"formula": "IQM_6x10(shortcut_forcing_AUC)-IQM_6x10(trajectory_imf_AUC)", "favorable_direction": "positive"},
        "rollout primary contrast",
    )
    _equal(
        contrasts[PRIMARY_METRICS[1]],
        {"formula": "IQM_6x10(trajectory_imf_nested_actor_returns)-IQM_6x10(shortcut_forcing_nested_actor_returns)", "favorable_direction": "positive"},
        "actor primary contrast",
    )
    practical = _mapping(
        statistics["practical_significance"], "statistics.practical_significance"
    )
    _expect_keys(
        practical,
        {
            "status",
            "claim_role",
            "required_budget_tracks",
            "thresholds",
            "criterion",
            "threshold_failure_blocks_practically_meaningful_wording",
        },
        "practical significance",
    )
    _equal(
        practical["status"],
        "frozen_before_pilot_or_confirmatory_outcomes",
        "practical-significance freeze status",
    )
    _equal(
        practical["claim_role"],
        "descriptive_interpretation_gate_not_a_substitute_for_adjusted_confidence_intervals",
        "practical-significance claim role",
    )
    _equal(
        practical["required_budget_tracks"],
        list(BUDGET_TRACK_ORDER),
        "practical-significance budget tracks",
    )
    practical_thresholds = _mapping(
        practical["thresholds"], "practical-significance thresholds"
    )
    if set(practical_thresholds) != set(PRIMARY_METRICS):
        raise ValueError("practical thresholds must contain exactly both primary metrics")
    rollout_threshold = _mapping(
        practical_thresholds[PRIMARY_METRICS[0]], "rollout practical threshold"
    )
    _expect_keys(
        rollout_threshold,
        {
            "minimum_absolute_iqm_contrast",
            "minimum_relative_iqm_reduction",
            "relative_denominator",
        },
        "rollout practical threshold",
    )
    _equal(
        _finite(
            rollout_threshold["minimum_absolute_iqm_contrast"],
            "rollout minimum absolute contrast",
        ),
        0.05,
        "rollout minimum absolute contrast",
    )
    _equal(
        _finite(
            rollout_threshold["minimum_relative_iqm_reduction"],
            "rollout minimum relative reduction",
        ),
        0.10,
        "rollout minimum relative reduction",
    )
    _equal(
        rollout_threshold["relative_denominator"],
        "shortcut_forcing_rollout_auc_clipped_below_1e-6",
        "rollout relative denominator",
    )
    actor_threshold = _mapping(
        practical_thresholds[PRIMARY_METRICS[1]], "actor practical threshold"
    )
    _expect_keys(
        actor_threshold,
        {"minimum_absolute_iqm_contrast", "raw_dmc_return_equivalent"},
        "actor practical threshold",
    )
    _equal(
        _finite(
            actor_threshold["minimum_absolute_iqm_contrast"],
            "actor minimum absolute contrast",
        ),
        0.05,
        "actor minimum absolute contrast",
    )
    _equal(
        _finite(
            actor_threshold["raw_dmc_return_equivalent"],
            "actor raw return equivalent",
        ),
        50.0,
        "actor raw return equivalent",
    )
    _equal(
        practical["criterion"],
        "all_required_point_estimate_thresholds_met_on_both_budget_tracks",
        "practical-significance criterion",
    )
    _bool(
        practical["threshold_failure_blocks_practically_meaningful_wording"],
        True,
        "practical-significance wording boundary",
    )
    rule = _mapping(statistics["superiority_rule"], "statistics.superiority_rule")
    _expect_keys(
        rule,
        {
            "eligible_evidence_class",
            "primary_objective_isolation_track",
            "required_compute_robustness_track",
            "required_budget_tracks",
            "required_primary_metrics",
            "criterion",
            "conjunctive",
            "missing_or_nonfinite_is_failure",
            "secondary_metrics_may_substitute",
            "compensating_wins_allowed",
            "smoke_or_pilot_may_support_claim",
        },
        "superiority rule",
    )
    _equal(rule["eligible_evidence_class"], "confirmatory", "eligible evidence class")
    _equal(rule["primary_objective_isolation_track"], "equal_updates", "objective-isolation track")
    _equal(rule["required_compute_robustness_track"], "equal_compiler_flops", "compute robustness track")
    _equal(rule["required_budget_tracks"], list(BUDGET_TRACK_ORDER), "superiority budget tracks")
    _equal(rule["required_primary_metrics"], list(PRIMARY_METRICS), "superiority primary metrics")
    _equal(
        rule["criterion"],
        "every_required_finite_interval_has_lower_bound_strictly_greater_than_zero",
        "superiority criterion",
    )
    for key in ("conjunctive", "missing_or_nonfinite_is_failure"):
        _bool(rule[key], True, f"superiority_rule.{key}")
    for key in ("secondary_metrics_may_substitute", "compensating_wins_allowed", "smoke_or_pilot_may_support_claim"):
        _bool(rule[key], False, f"superiority_rule.{key}")


def _validate_provenance_and_video(protocol: Mapping[str, Any]) -> None:
    provenance = _mapping(protocol["provenance"], "provenance")
    _expect_keys(
        provenance,
        {
            "hash_algorithm",
            "artifacts_immutable_after_registration",
            "atomic_manifest_write",
            "required_artifacts",
            "manifest_records_relative_path_size_and_sha256",
            "content_addressed_dataset_and_checkpoints",
        },
        "provenance",
    )
    _equal(provenance["hash_algorithm"], "sha256", "provenance hash")
    for key in (
        "artifacts_immutable_after_registration",
        "atomic_manifest_write",
        "manifest_records_relative_path_size_and_sha256",
        "content_addressed_dataset_and_checkpoints",
    ):
        _bool(provenance[key], True, f"provenance.{key}")
    _equal(provenance["required_artifacts"], list(REQUIRED_ARTIFACTS), "required provenance artifacts")

    video = _mapping(protocol["video_track"], "video_track")
    _expect_keys(
        video,
        {"execution", "required_before_broad_video_world_model_claims", "minimum_requirements", "may_support_state_control_superiority_without_execution"},
        "video_track",
    )
    _equal(video["execution"], "optional_separately_labeled_extension", "video execution")
    _bool(video["required_before_broad_video_world_model_claims"], True, "video claim requirement")
    _equal(
        video["minimum_requirements"],
        [
            "pixel_observations",
            "temporally_coherent_free_running_video_rollouts",
            "perceptual_and_task_relevant_fidelity_metrics",
            "the_same_nfe_frontier_and_compute_tracks",
        ],
        "video requirements",
    )
    _bool(video["may_support_state_control_superiority_without_execution"], False, "video state claim boundary")


def validate_matched_objective_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject any drift from the frozen, publication-suitable study contract."""

    protocol = _mapping(protocol, "protocol")
    _expect_keys(protocol, _TOP_LEVEL_KEYS, "protocol")
    _equal(protocol["schema_version"], SCHEMA_VERSION, "schema_version")
    _validate_claim_and_source(protocol)
    _validate_arms_and_ablations(protocol)
    _validate_shared_controls(protocol)
    _validate_canonical_executable_config(protocol)
    _validate_compute_accounting(protocol)
    _validate_data_and_randomness(protocol)
    _validate_budget_tracks(protocol)
    _validate_profiles(protocol)
    _validate_hyperparameter_policy(protocol)
    _validate_diagnostics_and_evaluation(protocol)
    _validate_statistics(protocol)
    _validate_provenance_and_video(protocol)


def expected_cells(protocol: Mapping[str, Any], profile: str) -> dict[str, list[dict[str, Any]]]:
    """Enumerate all paired confirmatory-arm cells for a protocol profile.

    Datasets are generated once per task/world-model seed.  Actor seeds are
    nested under each trained world model, never promoted to independent
    world-model replicates.
    """

    validate_matched_objective_protocol(protocol)
    if profile not in PROFILE_ORDER:
        raise ValueError(f"unknown profile {profile!r}")
    selected = protocol["profiles"][profile]
    datasets: list[dict[str, Any]] = []
    world_models: list[dict[str, Any]] = []
    rollout_evaluations: list[dict[str, Any]] = []
    actors: list[dict[str, Any]] = []
    actor_evaluations: list[dict[str, Any]] = []
    for task in selected["tasks"]:
        for world_seed in selected["world_model_seeds"]:
            datasets.append({"task": task, "world_model_seed": world_seed})
            for track in BUDGET_TRACK_ORDER:
                for arm in ARM_ORDER:
                    base = {
                        "budget_track": track,
                        "task": task,
                        "world_model_seed": world_seed,
                        "arm": arm,
                    }
                    world_models.append(dict(base))
                    for nfe in NFE_FRONTIER:
                        rollout_evaluations.append({**base, "nfe": nfe})
                    for actor_seed in selected["actor_seeds_nested_within_world_model_seed"]:
                        actor = {**base, "actor_seed": actor_seed}
                        actors.append(actor)
                        actor_evaluations.append(dict(actor))
    return {
        "datasets": datasets,
        "world_models": world_models,
        "rollout_evaluations": rollout_evaluations,
        "actors": actors,
        "actor_evaluations": actor_evaluations,
    }


@dataclass(frozen=True)
class SuperiorityDecision:
    """Auditable result of the frozen four-interval conjunction."""

    passed: bool
    evidence_class: str
    required_interval_count: int
    satisfied_interval_count: int
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interval_bounds(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, Mapping) or set(value) != {"lower", "upper"}:
        raise ValueError(f"{path} must be an object containing exactly lower and upper")
    lower = _finite(value["lower"], f"{path}.lower")
    upper = _finite(value["upper"], f"{path}.upper")
    if lower > upper:
        raise ValueError(f"{path}.lower cannot exceed upper")
    return lower, upper


def evaluate_superiority(
    protocol: Mapping[str, Any],
    intervals: Mapping[str, Any],
    *,
    evidence_class: str,
) -> SuperiorityDecision:
    """Apply the preregistered conjunctive rule to adjusted confidence intervals.

    A pass requires all four arm contrasts (two metrics by two compute tracks)
    to have a finite lower bound strictly above zero.  Missing or malformed
    intervals are recorded as failures, as is any non-confirmatory evidence
    class.  Extra exploratory results are ignored and cannot substitute.
    """

    validate_matched_objective_protocol(protocol)
    if not isinstance(intervals, Mapping):
        raise ValueError("intervals must be an object keyed by budget track")
    failures: list[str] = []
    satisfied = 0
    required = len(BUDGET_TRACK_ORDER) * len(PRIMARY_METRICS)
    eligible = protocol["statistics"]["superiority_rule"]["eligible_evidence_class"]
    if evidence_class != eligible:
        failures.append(f"evidence_class:{evidence_class!r}_is_not_{eligible!r}")
    for track in BUDGET_TRACK_ORDER:
        values = intervals.get(track)
        if not isinstance(values, Mapping):
            failures.append(f"{track}:missing")
            continue
        for metric in PRIMARY_METRICS:
            path = f"{track}.{metric}"
            if metric not in values:
                failures.append(f"{path}:missing")
                continue
            try:
                lower, _ = _interval_bounds(values[metric], path)
            except ValueError as error:
                failures.append(f"{path}:invalid:{error}")
                continue
            if lower <= 0.0:
                failures.append(f"{path}:lower_bound_not_strictly_positive")
            else:
                satisfied += 1
    return SuperiorityDecision(
        passed=not failures and satisfied == required,
        evidence_class=evidence_class,
        required_interval_count=required,
        satisfied_interval_count=satisfied,
        failures=tuple(failures),
    )


__all__ = [
    "ACTOR_SEEDS",
    "ARM_ORDER",
    "BUDGET_TRACK_ORDER",
    "EXPLORATORY_ABLATIONS",
    "NFE_FRONTIER",
    "PILOT_ACTOR_SEEDS",
    "PILOT_TASKS",
    "PILOT_WORLD_MODEL_SEEDS",
    "PRIMARY_METRICS",
    "PROFILE_ORDER",
    "REQUIRED_ARTIFACTS",
    "ROLLOUT_HORIZONS",
    "SCHEMA_VERSION",
    "SECONDARY_METRICS",
    "SuperiorityDecision",
    "TASKS",
    "WORLD_MODEL_SEEDS",
    "evaluate_superiority",
    "expected_cells",
    "protocol_digest",
    "read_matched_objective_protocol",
    "validate_matched_objective_protocol",
]
