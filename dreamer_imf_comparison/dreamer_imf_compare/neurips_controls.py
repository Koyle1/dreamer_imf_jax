"""Immutable matched strong-control and trajectory-iMF factorial benchmark.

This module intentionally imports the parent matched-objective harness for data,
seed, schedule, rollout, actor, compiler, and provenance primitives.  It does
not alter the two-arm confirmatory analysis or expose a superiority decision.
The controls are a separate, descriptive family.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from .artifacts import read_json, write_json_atomic
from . import matched_objective_benchmark as parent
from .matched_objective_protocol import (
    BUDGET_TRACK_ORDER,
    protocol_digest as parent_protocol_digest,
    validate_matched_objective_protocol,
)


CONTROL_SCHEMA = "trajectory-imf-neurips-controls-v1"
MATRIX_SCHEMA = "trajectory-imf-neurips-controls-matrix-v1"
COMPUTE_SCHEMA = "trajectory-imf-neurips-controls-compute-v1"
WORLD_SCHEMA = "trajectory-imf-neurips-controls-world-v1"
ROLLOUT_SCHEMA = "trajectory-imf-neurips-controls-rollout-v1"
ACTOR_SCHEMA = "trajectory-imf-neurips-controls-actor-v1"
ANALYSIS_SCHEMA = "trajectory-imf-neurips-controls-analysis-v1"
MANIFEST_SCHEMA = "trajectory-imf-neurips-controls-artifacts-v1"
SELECTION_SCHEMA = "trajectory-imf-neurips-controls-selection-v1"

PROFILE_ORDER = ("smoke", "development", "confirmatory")
STAGE_ORDER = ("compute", "world", "rollout", "actor")
CONTEXT_NAMES = ("clean", "corrupted", "suffix")
CONTROL_RELATIVE_PATHS = (
    "dreamer_imf_comparison/neurips_controls_protocol.json",
    "dreamer_imf_comparison/dreamer_imf_compare/neurips_controls.py",
    "dreamer_imf_comparison/scripts/run_neurips_controls.py",
    "dreamer_imf_comparison/scripts/verify_neurips_controls.py",
    "dreamer_imf_comparison/tests/test_neurips_controls.py",
    "dreamer_imf_comparison/NEURIPS_CONTROLS.md",
)
_VALIDATED_DATASETS: set[tuple[str, str, str, str]] = set()
_VALIDATED_COMPUTE_PLANS: set[tuple[str, str, str]] = set()
_VALIDATED_ROLLOUT_REPLAYS: set[tuple[str, str, str, str]] = set()
_VALIDATED_ACTOR_REPLAYS: set[tuple[str, str, str, str]] = set()


class ControlLoss(NamedTuple):
    total: Any
    reconstruction: Any
    reward: Any
    continuation: Any
    prior: Any
    representation: Any
    temporal_increment: Any
    endpoint_certificate: Any
    imf_loss_u: Any
    imf_loss_v: Any


class ControlSchedule(NamedTuple):
    r: Any
    t: Any
    history_t: Any
    loss_mask: Any
    pattern: Any
    suffix_start: Any


def canonical_bytes(value: Any) -> bytes:
    return parent.canonical_bytes(value)


def object_sha256(value: Any) -> str:
    return parent.object_sha256(value)


def file_sha256(path: str | Path) -> str:
    return parent.file_sha256(path)


def _without_digest(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result.pop(key, None)
    return result


def read_controls_protocol(path: str | Path) -> dict[str, Any]:
    protocol = read_json(path)
    validate_controls_protocol(protocol)
    return protocol


def controls_protocol_digest(protocol: Mapping[str, Any]) -> str:
    validate_controls_protocol(protocol)
    return object_sha256(protocol)


def _context_probabilities(mask: int) -> tuple[float, float, float]:
    if not isinstance(mask, int) or isinstance(mask, bool) or not 1 <= mask <= 7:
        raise ValueError("context mask must be an integer in [1, 7]")
    enabled = tuple(bool(mask & (1 << index)) for index in range(3))
    count = sum(enabled)
    return tuple((1.0 / count if value else 0.0) for value in enabled)


def expected_arm_specs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen 2x2x(nonempty 2^3) design deterministically."""

    reference = []
    for raw in protocol["reference_arms"]:
        row = {**dict(raw), "endpoint_certificate_enabled": False, "endpoint_certificate_mass": 0.0}
        if row["arm_id"] == "trajectory_endpoint_certificate":
            row.update(
                query_history_noise_coupling="shared",
                query_history_time_relation="separate",
                clean_context_enabled=True,
                corrupted_context_enabled=True,
                future_suffix_enabled=True,
                context_mask=7,
                context_probabilities=[1.0 / 3.0] * 3,
                endpoint_certificate_enabled=True,
                endpoint_certificate_mass=float(
                    protocol["endpoint_certificate_control"]["mixture_mass"]
                ),
            )
        reference.append(row)
    factorial = protocol["factorial"]
    levels = factorial["levels"]
    proposed = factorial["proposed_cell"]
    rows: list[dict[str, Any]] = []
    for noise in levels["query_history_noise_coupling"]:
        for relation in levels["query_history_time_relation"]:
            for mask in range(1, 8):
                enabled = tuple(bool(mask & (1 << index)) for index in range(3))
                candidate = {
                    "query_history_noise_coupling": noise,
                    "query_history_time_relation": relation,
                    "clean_context_enabled": enabled[0],
                    "corrupted_context_enabled": enabled[1],
                    "future_suffix_enabled": enabled[2],
                }
                is_proposed = all(candidate[key] == proposed[key] for key in candidate)
                arm_id = (
                    proposed["arm_id"]
                    if is_proposed
                    else (
                        "trajectory_f__noise-"
                        + noise
                        + "__time-"
                        + relation
                        + "__ctx-"
                        + format(mask, "03b")
                    )
                )
                rows.append(
                    {
                        "arm_id": arm_id,
                        "family": "trajectory_imf",
                        "role": "proposed" if is_proposed else "factorial_ablation",
                        "primary_nfe": 1,
                        "training_objective": "fixed_context_per_token_time_trajectory_imf",
                        **candidate,
                        "context_mask": mask,
                        "context_probabilities": list(_context_probabilities(mask)),
                        "endpoint_certificate_enabled": False,
                        "endpoint_certificate_mass": 0.0,
                    }
                )
    result = reference + rows
    if len({row["arm_id"] for row in result}) != len(result):
        raise AssertionError("control arm expansion produced duplicate ids")
    return result


def _profile_arm_specs(
    protocol: Mapping[str, Any], profile: str
) -> list[dict[str, Any]]:
    arms = expected_arm_specs(protocol)
    policy = protocol["profile_arm_policy"]
    if profile in ("smoke", "development"):
        return arms
    if profile == "confirmatory":
        included = list(policy["confirmatory_arm_ids"])
        by_id = {arm["arm_id"]: arm for arm in arms}
        if len(included) != len(set(included)) or set(included) - set(by_id):
            raise ValueError("confirmatory arm policy contains invalid ids")
        return [by_id[arm_id] for arm_id in included]
    raise ValueError(f"unknown controls profile {profile!r}")


def _assert_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} keys are incomplete or contain extras")


def validate_controls_protocol(
    protocol: Mapping[str, Any], parent_protocol: Mapping[str, Any] | None = None
) -> None:
    """Fail closed on every claim-relevant controls declaration."""

    top = {
        "schema_version",
        "status",
        "study_name",
        "parent_protocol",
        "claim_boundary",
        "reference_arms",
        "temporal_increment_control",
        "endpoint_certificate_control",
        "factorial",
        "expected_total_arms",
        "profile_arm_policy",
        "profiles",
        "development_selection",
        "matched_controls",
        "evaluation",
        "statistics",
        "provenance",
    }
    _assert_exact_keys(protocol, top, "controls protocol")
    if (
        protocol["schema_version"] != CONTROL_SCHEMA
        or protocol["status"] != "frozen_before_control_outcomes"
        or protocol["study_name"]
        != "Matched strong controls and factorial trajectory-iMF ablations"
    ):
        raise ValueError("controls protocol identity/status mismatch")
    boundary = protocol["claim_boundary"]
    if boundary != {
        "role": "secondary_descriptive_controls",
        "may_modify_or_substitute_for_primary_gate": False,
        "may_support_primary_superiority_claim": False,
        "primary_gate_accessed_by_runner": False,
        "smoke_or_development_is_claim_evidence": False,
        "dreamer4_label_allowed": False,
        "shortcut_label": (
            "paper-derived Dreamer-4-Equation-7-style shortcut-forcing "
            "reimplementation"
        ),
    }:
        raise ValueError("controls claim boundary is not isolated from the primary gate")
    expected_references = [
        {
            "arm_id": "shortcut_forcing",
            "family": "shortcut_forcing",
            "role": "paper_derived_reference",
            "primary_nfe": 4,
            "training_objective": "parent_protocol_shortcut_forcing",
        },
        {
            "arm_id": "ordinary_imf",
            "family": "ordinary_imf",
            "role": "strong_baseline",
            "primary_nfe": 1,
            "training_objective": (
                "ordinary_conditional_iMF_on_one_step_posterior_transitions_without_"
                "trajectory_schedule_or_corrupted_history"
            ),
        },
        {
            "arm_id": "gaussian_rssm",
            "family": "gaussian_rssm",
            "role": "strong_baseline",
            "primary_nfe": 1,
            "training_objective": "diagonal_Gaussian_RSSM_prior_with_stop_posterior_KL",
        },
        {
            "arm_id": "temporal_increment_imf",
            "family": "temporal_increment_imf",
            "role": "strong_baseline",
            "primary_nfe": 1,
            "training_objective": (
                "ordinary_iMF_plus_stopped_one_step_vs_two_half_step_endpoint_consistency"
            ),
        },
        {
            "arm_id": "trajectory_endpoint_certificate",
            "family": "trajectory_endpoint_certificate",
            "role": "exploratory_theorem_aligned_ablation",
            "primary_nfe": 1,
            "training_objective": (
                "proposed_trajectory_iMF_plus_positive_mass_raw_r_zero_endpoint_certificate"
            ),
        },
    ]
    if protocol["reference_arms"] != expected_references:
        raise ValueError("reference arms drifted from the frozen definitions")
    factorial = protocol["factorial"]
    expected_factor_keys = {
        "base_family",
        "full_factorial",
        "factors_in_id_order",
        "levels",
        "invalid_combinations",
        "context_probabilities",
        "expected_trajectory_arms",
        "proposed_cell",
        "nonproposed_arm_id_format",
    }
    _assert_exact_keys(factorial, expected_factor_keys, "factorial")
    if (
        factorial["base_family"] != "trajectory_imf"
        or factorial["full_factorial"] is not True
        or factorial["factors_in_id_order"]
        != [
            "query_history_noise_coupling",
            "query_history_time_relation",
            "clean_context_enabled",
            "corrupted_context_enabled",
            "future_suffix_enabled",
        ]
        or set(factorial["levels"])
        != {
            "query_history_noise_coupling",
            "query_history_time_relation",
            "clean_context_enabled",
            "corrupted_context_enabled",
            "future_suffix_enabled",
        }
        or factorial["expected_trajectory_arms"] != 28
        or factorial["levels"]["query_history_noise_coupling"]
        != ["shared", "independent"]
        or factorial["levels"]["query_history_time_relation"]
        != ["separate", "tied_on_corrupted_positions"]
        or factorial["levels"]["clean_context_enabled"] != [False, True]
        or factorial["levels"]["corrupted_context_enabled"] != [False, True]
        or factorial["levels"]["future_suffix_enabled"] != [False, True]
        or factorial["invalid_combinations"]
        != "exclude_only_the_four_combinations_with_all_three_context_components_disabled"
        or factorial["context_probabilities"]
        != "uniform_after_renormalizing_over_enabled_components"
        or factorial["nonproposed_arm_id_format"]
        != "trajectory_f__noise-{noise}__time-{time}__ctx-{three_bit_CRS_mask}"
    ):
        raise ValueError("factorial levels or exclusions drifted")
    # Avoid calling the expander here because callers use this validator before
    # expansion; reproduce only the closed-form count assertion.
    count = len(protocol["reference_arms"]) + 2 * 2 * 7
    if protocol["expected_total_arms"] != count or count != 33:
        raise ValueError("expected control arm count drifted")
    arm_policy = protocol["profile_arm_policy"]
    if arm_policy != {
        "frozen_before_any_control_outcome": True,
        "smoke": "all_registered_33",
        "development": "all_registered_33",
        "confirmatory": "prespecified_six_references_and_proposed",
        "confirmatory_arm_ids": [
            "shortcut_forcing",
            "ordinary_imf",
            "gaussian_rssm",
            "temporal_increment_imf",
            "trajectory_endpoint_certificate",
            "trajectory_imf",
        ],
        "nonproposed_factorial_cells_confirmatory_excluded": True,
        "rationale": (
            "The complete 28-cell factorial is a development-only mechanism study; "
            "repeating it over both confirmatory tracks would create 19806 cells and "
            "invite post-hoc ablation inference. Confirmatory controls are limited "
            "prospectively to six named reference/proposed arms."
        ),
    }:
        raise ValueError("profile arm policy drifted after its pre-outcome freeze")
    proposed = factorial["proposed_cell"]
    if proposed != {
        "arm_id": "trajectory_imf",
        "query_history_noise_coupling": "shared",
        "query_history_time_relation": "separate",
        "clean_context_enabled": True,
        "corrupted_context_enabled": True,
        "future_suffix_enabled": True,
    }:
        raise ValueError("proposed factorial cell drifted")
    if set(protocol["profiles"]) != set(PROFILE_ORDER):
        raise ValueError("controls profiles are incomplete")
    expected_profile_metadata = {
        "smoke": ("smoke", "engineering_smoke", ["equal_updates"], 64),
        "development": ("pilot", "development_only", ["equal_updates"], 2000),
        "confirmatory": (
            "confirmatory",
            "confirmatory_secondary_descriptive",
            list(BUDGET_TRACK_ORDER),
            20000,
        ),
    }
    for profile in PROFILE_ORDER:
        row = protocol["profiles"][profile]
        _assert_exact_keys(
            row,
            {
                "parent_profile",
                "evidence_class",
                "claim_eligible",
                "tasks",
                "world_model_seeds",
                "actor_seeds_nested_within_world_model_seed",
                "run_budget_tracks",
                "analysis_bootstrap_resamples",
            },
            f"controls {profile} profile",
        )
        expected_parent, expected_evidence, expected_tracks, expected_resamples = (
            expected_profile_metadata[profile]
        )
        if (
            row.get("claim_eligible") is not False
            or row.get("parent_profile") != expected_parent
            or row.get("evidence_class") != expected_evidence
            or row.get("run_budget_tracks") != expected_tracks
            or row.get("analysis_bootstrap_resamples") != expected_resamples
        ):
            raise ValueError("controls profile metadata drifted")
        if not row.get("tasks") or not row.get("world_model_seeds") or not row.get(
            "actor_seeds_nested_within_world_model_seed"
        ):
            raise ValueError("controls profile units are empty")
    selection = protocol["development_selection"]
    expected_selection = {
        "selection_tasks": ["dmc_reacher_easy", "dmc_pendulum_swingup"],
        "confirmatory_tasks_must_be_unread": True,
        "selected_families": [
            "ordinary_imf",
            "gaussian_rssm",
            "temporal_increment_imf",
        ],
        "candidate_grid": {
            "ordinary_imf": [
                {"candidate_id": "ordinary_scale_0p3", "prior_scale": 0.3},
                {"candidate_id": "ordinary_scale_1p0", "prior_scale": 1.0},
                {"candidate_id": "ordinary_scale_3p0", "prior_scale": 3.0},
            ],
            "gaussian_rssm": [
                {"candidate_id": "gaussian_scale_0p3", "prior_scale": 0.3},
                {"candidate_id": "gaussian_scale_1p0", "prior_scale": 1.0},
                {"candidate_id": "gaussian_scale_3p0", "prior_scale": 3.0},
            ],
            "temporal_increment_imf": [
                {
                    "candidate_id": "tic_scale_0p1",
                    "prior_scale": 1.0,
                    "temporal_increment_scale": 0.1,
                },
                {
                    "candidate_id": "tic_scale_0p3",
                    "prior_scale": 1.0,
                    "temporal_increment_scale": 0.3,
                },
                {
                    "candidate_id": "tic_scale_1p0",
                    "prior_scale": 1.0,
                    "temporal_increment_scale": 1.0,
                },
            ],
        },
        "smoke_candidates": {
            "ordinary_imf": "ordinary_scale_1p0",
            "gaussian_rssm": "gaussian_scale_1p0",
            "temporal_increment_imf": "tic_scale_0p3",
        },
        "selection_estimand": (
            "mean_over_tasks_of_within_task_mean_world_seed_rank_of_rollout_AUC_plus_"
            "rank_of_negative_nested_actor_return_divided_by_two"
        ),
        "selection_budget_track": "equal_updates",
        "within_metric_rank_tie_break": "lexicographically_smallest_candidate_id",
        "tie_break": "lexicographically_smallest_candidate_id",
        "factorial_settings_tuned": False,
        "primary_or_confirmatory_outcomes_may_affect_selection": False,
    }
    if selection != expected_selection:
        raise ValueError("development-only selection boundary drifted")
    for family in selection["selected_families"]:
        candidates = selection["candidate_grid"].get(family)
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise ValueError("each selected control family needs a real candidate grid")
        ids = [candidate.get("candidate_id") for candidate in candidates]
        if len(set(ids)) != len(ids) or any(not value for value in ids):
            raise ValueError("control candidate ids are invalid")
    statistics = protocol["statistics"]
    if statistics != {
        "independent_unit": "task_x_world_model_seed",
        "actor_seeds_are_nested": True,
        "inference_role": "descriptive_only",
        "interval_method": "task_stratified_paired_percentile_bootstrap",
        "interval_percentiles": [2.5, 97.5],
        "bootstrap_seed": 314159,
        "multiplicity_adjusted_decision": False,
        "superiority_decision_field_permitted": False,
        "primary_gate_fields_permitted": False,
    }:
        raise ValueError("control inference could contaminate the primary decision")
    certificate = protocol["endpoint_certificate_control"]
    if certificate != {
        "claim_role": "exploratory_only",
        "may_modify_or_substitute_for_primary_gate": False,
        "base_arm": "trajectory_imf",
        "mixture_mass": 0.25,
        "r": 0.0,
        "s_sampling": "independent_uniform_[0,1]_per_token_from_objective_fold_in_5",
        "terms": "raw_unadapted_sum_of_squared_F_regression_and_boundary_v_regression",
        "implementation": (
            "trajectory_imf_loss_with_adaptive_power_zero_on_the_r_zero_slice"
        ),
        "context": "same_realized_causal_context_as_the_proposed_trajectory_draw",
        "inference_change": False,
    }:
        raise ValueError("endpoint-certificate control drifted or entered the primary gate")
    if protocol["temporal_increment_control"] != {
        "description": (
            "A deliberately simple MeLISA-inspired temporal-increment consistency "
            "control; it is not a MeLISA reproduction."
        ),
        "direct_endpoint": "one full iMF transport step from the retained base noise",
        "teacher_endpoint": (
            "two half iMF transport steps from the same retained base noise and fixed "
            "condition"
        ),
        "loss": "masked_mean_square(direct_endpoint-stop_gradient(teacher_endpoint))",
        "default_scale": 0.3,
        "self_distillation_teacher": "live_parameters_with_stop_gradient",
        "extra_context_corruption": False,
    }:
        raise ValueError("temporal-increment control drifted from its frozen definition")
    if protocol["matched_controls"] != {
        "canonical_data": "import_parent_dataset_cell_and_validator",
        "canonical_minibatch_schedule": "import_parent_batch_schedule_and_materializer",
        "canonical_objective_key": (
            "sha256_namespace_world-objective_task_world-seed_folded_by_update"
        ),
        "canonical_rollout_windows": "import_parent_rollout_window_derivation",
        "canonical_actor_initialization_and_budget": (
            "import_parent_actor_seed_schedule_and_Dreamer_actor_critic"
        ),
        "canonical_actor_evaluation": (
            "import_parent_DMC_policy_evaluation_seeds_and_episodes"
        ),
        "same_shared_dimensions_optimizer_precision_and_common_losses": True,
        "world_model_frozen_during_actor_learning": True,
        "parameter_and_compiler_flops_reported_per_arm": True,
        "equal_flop_reference_arm": "shortcut_forcing",
        "nearest_integer_allocation_tie_break": "lower_update_count",
        "equal_flop_relative_tolerance": 0.002,
        "smoke_runs_equal_updates_only_but_compiles_both_allocations": True,
    }:
        raise ValueError("matched-controls declaration drifted")
    if protocol["evaluation"] != {
        "rollout_horizons": [1, 2, 4, 8, 15, 30],
        "rollout_context_length": 8,
        "rollout_metric": "normalized_free_running_rollout_error_auc",
        "actor_metric": "real_environment_normalized_return",
        "factorial_and_controls_nfe": 1,
        "shortcut_nfe": 4,
        "paired_rollout_starts_noise_and_environment_seeds": True,
        "retain_raw_predictive_draws_actions_rewards_continuations_and_terminals": True,
    }:
        raise ValueError("controls evaluation contract drifted")
    if protocol["provenance"] != {
        "freeze_before_execution": True,
        "source_file_hashes": True,
        "runtime_fingerprint": True,
        "resolved_config_and_digest_per_arm": True,
        "compiler_IR_and_cost_analysis": True,
        "world_and_actor_checkpoints": True,
        "checkpoint_recomputed_rollouts": True,
        "checkpoint_replayed_actor_environment_trajectories": True,
        "exact_artifact_manifest": True,
        "finalized_roots_are_immutable_to_runner": True,
        "development_selection_requires_clean_committed_source_and_single_visible_device": True,
        "confirmatory_requires_clean_committed_source_and_single_visible_device": True,
    }:
        raise ValueError("controls provenance contract drifted")
    if parent_protocol is not None:
        validate_matched_objective_protocol(parent_protocol)
        identity = protocol["parent_protocol"]
        if (
            identity.get("schema_version") != parent_protocol.get("schema_version")
            or identity.get("sha256") != parent_protocol_digest(parent_protocol)
        ):
            raise ValueError("pinned parent protocol identity mismatch")
        for profile, row in protocol["profiles"].items():
            source = parent_protocol["profiles"][row["parent_profile"]]
            for key in (
                "tasks",
                "world_model_seeds",
                "actor_seeds_nested_within_world_model_seed",
            ):
                if row[key] != source[key]:
                    raise ValueError(f"controls {profile} {key} differs from parent")
        evaluation = protocol["evaluation"]
        if (
            evaluation["rollout_horizons"] != parent_protocol["evaluation"]["rollout_horizons"]
            or evaluation["rollout_context_length"]
            != parent_protocol["evaluation"]["rollout_context_length"]
            or evaluation["shortcut_nfe"]
            != parent_protocol["evaluation"]["primary_nfe"]["shortcut_forcing"]
            or evaluation["factorial_and_controls_nfe"]
            != parent_protocol["evaluation"]["primary_nfe"]["trajectory_imf"]
        ):
            raise ValueError("controls evaluation differs from the parent benchmark")
        tolerance = parent_protocol["budget_tracks"]["equal_compiler_flops"][
            "relative_tolerance"
        ]
        if protocol["matched_controls"]["equal_flop_relative_tolerance"] != tolerance:
            raise ValueError("controls equal-FLOP tolerance differs from parent")


def _workspace_root(path: str | Path | None = None) -> Path:
    return Path(path).resolve() if path is not None else Path(__file__).resolve().parents[2]


def build_controls_source_manifest(workspace: str | Path | None = None) -> dict[str, Any]:
    root = _workspace_root(workspace)
    parent_manifest = parent.build_source_manifest(root)
    records: list[dict[str, Any]] = []
    for relative in CONTROL_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"controls source dependency is absent or symlinked: {relative}")
        records.append(
            {"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)}
        )
    all_source_paths = sorted(
        {entry["path"] for entry in parent_manifest["files"]} | set(CONTROL_RELATIVE_PATHS)
    )
    payload = {
        "schema_version": "trajectory-imf-neurips-controls-source-v1",
        "parent_source_manifest": parent_manifest,
        "control_files": records,
        "git": parent._git_identity(root, all_source_paths),
    }
    payload["source_sha256"] = object_sha256(payload)
    return payload


def validate_controls_source_manifest(
    manifest: Mapping[str, Any], workspace: str | Path | None = None
) -> None:
    root = _workspace_root(workspace)
    if set(manifest) != {
        "schema_version",
        "parent_source_manifest",
        "control_files",
        "git",
        "source_sha256",
    } or manifest.get("schema_version") != "trajectory-imf-neurips-controls-source-v1":
        raise ValueError("controls source manifest schema is invalid")
    if manifest.get("source_sha256") != object_sha256(
        _without_digest(manifest, "source_sha256")
    ):
        raise ValueError("controls source manifest digest mismatch")
    parent.validate_source_manifest(manifest["parent_source_manifest"], root)
    expected: list[dict[str, Any]] = []
    for relative in CONTROL_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"controls source dependency is absent or symlinked: {relative}")
        expected.append(
            {"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)}
        )
    if manifest["control_files"] != expected:
        raise ValueError("controls source files differ from the frozen manifest")
    git = manifest["git"]
    required_git = {
        "commit",
        "dirty_patch_sha256",
        "dirty_patch_bytes",
        "commit_status",
        "commit_stderr",
        "dirty_patch_status",
        "dirty_patch_stderr",
        "untracked_files",
        "untracked_status",
        "untracked_stderr",
        "command_timeout_seconds",
        "fallback",
    }
    if not isinstance(git, Mapping) or set(git) != required_git:
        raise ValueError("controls source Git identity is incomplete")
    digest = git["dirty_patch_sha256"]
    all_source_paths = {
        entry["path"] for entry in manifest["parent_source_manifest"]["files"]
    } | set(CONTROL_RELATIVE_PATHS)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(git["dirty_patch_bytes"], int)
        or git["dirty_patch_bytes"] < 0
        or not all(
            isinstance(git[field], str)
            for field in (
                "commit",
                "commit_status",
                "commit_stderr",
                "dirty_patch_status",
                "dirty_patch_stderr",
                "untracked_status",
                "untracked_stderr",
                "fallback",
            )
        )
        or git["command_timeout_seconds"] != parent.GIT_COMMAND_TIMEOUT_SECONDS
        or git["fallback"]
        != "complete_relevant_source_file_size_and_sha256_manifest"
        or not isinstance(git["untracked_files"], list)
        or git["untracked_files"] != sorted(set(git["untracked_files"]))
        or any(path not in all_source_paths for path in git["untracked_files"])
    ):
        raise ValueError("controls source Git identity is invalid")
    if git["dirty_patch_status"] == "complete":
        if git["dirty_patch_bytes"] == 0 and digest != hashlib.sha256(b"").hexdigest():
            raise ValueError("empty controls source patch has the wrong digest")
    if git["commit_status"] == "complete" and (
        len(git["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in git["commit"])
    ):
        raise ValueError("controls source commit hash is invalid")


def _candidate_grid(protocol: Mapping[str, Any], family: str) -> list[dict[str, Any]]:
    return [dict(row) for row in protocol["development_selection"]["candidate_grid"][family]]


def _candidate_by_id(protocol: Mapping[str, Any], family: str, candidate_id: str) -> dict[str, Any]:
    matches = [
        row for row in _candidate_grid(protocol, family) if row["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate candidate {candidate_id!r} for {family}")
    return matches[0]


def _neutral_parent_candidates(parent_protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shortcut_forcing": next(
            candidate
            for candidate in parent.pilot_candidates(parent_protocol, "shortcut_forcing")
            if candidate["overrides"] == {"training_k_max": 4, "objective_loss_scale": 1.0}
        ),
        "trajectory_imf": next(
            candidate
            for candidate in parent.pilot_candidates(parent_protocol, "trajectory_imf")
            if candidate["overrides"]
            == {"imf_boundary_fraction": 0.5, "objective_loss_scale": 1.0}
        ),
    }


def _validate_parent_selection(
    selection: Mapping[str, Any], parent_protocol: Mapping[str, Any], source_sha256: str
) -> dict[str, Any]:
    # Use the parent's exact validator.  A controls confirmatory run is frozen
    # from the same source tree and therefore cannot silently consume an old
    # pilot selected under another implementation.
    return parent.validate_hpo_selection_manifest(
        selection, parent_protocol, source_sha256=source_sha256
    )


def _validate_controls_selection(
    selection: Mapping[str, Any], protocol: Mapping[str, Any], source_sha256: str
) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "controls_protocol_sha256",
        "development_analysis_sha256",
        "development_matrix_sha256",
        "development_source_sha256",
        "selection_profile",
        "selection_budget_track",
        "confirmatory_outcomes_accessed",
        "selected",
        "candidate_rank_evidence",
        "selection_sha256",
    }
    if set(selection) != required or selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("controls selection schema is invalid")
    if selection.get("selection_sha256") != object_sha256(
        _without_digest(selection, "selection_sha256")
    ):
        raise ValueError("controls selection digest mismatch")
    if (
        selection.get("status") != "complete"
        or selection.get("controls_protocol_sha256") != controls_protocol_digest(protocol)
        or selection.get("development_source_sha256") != source_sha256
        or selection.get("selection_profile") != "development"
        or selection.get("selection_budget_track") != "equal_updates"
        or selection.get("confirmatory_outcomes_accessed") is not False
        or set(selection.get("selected", {}))
        != set(protocol["development_selection"]["selected_families"])
        or set(selection.get("candidate_rank_evidence", {}))
        != set(protocol["development_selection"]["selected_families"])
    ):
        raise ValueError("controls selection violates the development-only boundary")
    for field in ("development_analysis_sha256", "development_matrix_sha256"):
        digest = selection.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"controls selection {field} is not a SHA-256 digest")
    for family, candidate in selection["selected"].items():
        expected = _candidate_by_id(protocol, family, candidate.get("candidate_id"))
        if candidate != expected:
            raise ValueError("selected controls candidate differs from frozen grid")
        evidence = selection["candidate_rank_evidence"][family]
        expected_ids = {
            value["candidate_id"] for value in _candidate_grid(protocol, family)
        }
        if (
            not isinstance(evidence, list)
            or {row.get("candidate_id") for row in evidence} != expected_ids
        ):
            raise ValueError("controls selection rank evidence is incomplete")
        for row in evidence:
            if set(row) != {"candidate_id", "selection_score", "task_scores"}:
                raise ValueError("controls selection rank row schema is invalid")
            scores = np.asarray(row["task_scores"], dtype=np.float64)
            if (
                scores.shape
                != (len(protocol["development_selection"]["selection_tasks"]),)
                or not np.isfinite(scores).all()
                or not math.isfinite(float(row["selection_score"]))
                or float(row["selection_score"]) != float(np.mean(scores))
            ):
                raise ValueError("controls selection rank evidence is invalid")
        winner = min(
            evidence,
            key=lambda row: (float(row["selection_score"]), row["candidate_id"]),
        )
        if winner["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("controls selected candidate differs from rank evidence")
    return dict(selection)


def _arm_runs(
    protocol: Mapping[str, Any],
    profile: str,
    controls_selection: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    arms = _profile_arm_specs(protocol, profile)
    selected_families = set(protocol["development_selection"]["selected_families"])
    runs: list[dict[str, Any]] = []
    for arm in arms:
        family = arm["family"]
        if profile == "development" and family in selected_families:
            candidates = _candidate_grid(protocol, family)
        elif profile == "confirmatory" and family in selected_families:
            if controls_selection is None:
                raise ValueError("confirmatory controls require a development selection")
            candidates = [dict(controls_selection["selected"][family])]
        elif profile == "smoke" and family in selected_families:
            candidate_id = protocol["development_selection"]["smoke_candidates"][family]
            candidates = [_candidate_by_id(protocol, family, candidate_id)]
        else:
            candidates = [{"candidate_id": "fixed", "prior_scale": 1.0}]
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            run_id = arm["arm_id"] if candidate_id == "fixed" else f"{arm['arm_id']}@@{candidate_id}"
            runs.append(
                {
                    "run_id": run_id,
                    "arm_id": arm["arm_id"],
                    "family": family,
                    "role": arm["role"],
                    "primary_nfe": arm["primary_nfe"],
                    "arm_spec": arm,
                    "candidate": candidate,
                }
            )
    if len({run["run_id"] for run in runs}) != len(runs):
        raise AssertionError("control run expansion produced duplicate ids")
    return runs


def _cell_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    digest = object_sha256(value)
    return {**value, "identity_sha256": digest, "cell_id": f"{value['stage']}-{digest[:24]}"}


def _matrix_digest(matrix: Mapping[str, Any]) -> str:
    return object_sha256(_without_digest(matrix, "matrix_sha256"))


def build_controls_matrix(
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    profile: str,
    *,
    source_manifest: Mapping[str, Any],
    parent_selection: Mapping[str, Any] | None = None,
    controls_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_controls_protocol(protocol, parent_protocol)
    validate_controls_source_manifest(source_manifest)
    if profile not in PROFILE_ORDER:
        raise ValueError(f"unknown controls profile {profile!r}")
    selected = protocol["profiles"][profile]
    source_sha = str(source_manifest["source_sha256"])
    parent_source_sha = source_manifest["parent_source_manifest"]["source_sha256"]
    if profile == "confirmatory":
        if parent_selection is None or controls_selection is None:
            raise ValueError("confirmatory controls require both frozen selections")
        parent_selection = _validate_parent_selection(
            parent_selection, parent_protocol, parent_source_sha
        )
        controls_selection = _validate_controls_selection(
            controls_selection, protocol, source_sha
        )
    elif parent_selection is not None or controls_selection is not None:
        raise ValueError("smoke/development controls cannot consume confirmatory selections")
    runs = _arm_runs(protocol, profile, controls_selection)
    tasks = [parent.canonical_task_id(task) for task in selected["tasks"]]
    cells: list[dict[str, Any]] = []
    for task in tasks:
        cells.append(
            _cell_identity(
                {
                    "stage": "compute",
                    "profile": profile,
                    "task": task,
                    "world_model_seed": None,
                    "budget_track": None,
                    "run_id": None,
                    "actor_seed": None,
                    "controls_protocol_sha256": controls_protocol_digest(protocol),
                    "source_sha256": source_sha,
                }
            )
        )
        compute_id = cells[-1]["cell_id"]
        for world_seed in selected["world_model_seeds"]:
            for track in selected["run_budget_tracks"]:
                for run in runs:
                    base = {
                        "profile": profile,
                        "task": task,
                        "world_model_seed": int(world_seed),
                        "budget_track": track,
                        "run_id": run["run_id"],
                        "arm_id": run["arm_id"],
                        "actor_seed": None,
                        "controls_protocol_sha256": controls_protocol_digest(protocol),
                        "source_sha256": source_sha,
                        "compute_cell_id": compute_id,
                    }
                    world = _cell_identity({"stage": "world", **base})
                    cells.append(world)
                    cells.append(
                        _cell_identity(
                            {
                                "stage": "rollout",
                                **base,
                                "world_cell_id": world["cell_id"],
                            }
                        )
                    )
                    for actor_seed in selected["actor_seeds_nested_within_world_model_seed"]:
                        cells.append(
                            _cell_identity(
                                {
                                    "stage": "actor",
                                    **base,
                                    "actor_seed": int(actor_seed),
                                    "world_cell_id": world["cell_id"],
                                }
                            )
                        )
    rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    cells.sort(key=lambda cell: (rank[cell["stage"]], cell["cell_id"]))
    parent_candidates = (
        {
            arm: {
                "candidate_id": parent_selection["selected"][arm]["candidate_id"],
                "overrides": parent_selection["selected"][arm]["overrides"],
            }
            for arm in ("shortcut_forcing", "trajectory_imf")
        }
        if profile == "confirmatory"
        else _neutral_parent_candidates(parent_protocol)
    )
    matrix = {
        "schema_version": MATRIX_SCHEMA,
        "status": "frozen_before_control_outcomes",
        "profile": profile,
        "evidence_class": selected["evidence_class"],
        "claim_eligible": False,
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "parent_protocol_sha256": parent_protocol_digest(parent_protocol),
        "source_sha256": source_sha,
        "parent_source_sha256": parent_source_sha,
        "run_budget_tracks": list(selected["run_budget_tracks"]),
        "arm_specs": _profile_arm_specs(protocol, profile),
        "arm_runs": runs,
        "parent_candidates": parent_candidates,
        "parent_selection": parent_selection,
        "parent_selection_sha256": (
            parent_selection.get("selection_sha256") if parent_selection is not None else None
        ),
        "controls_selection": controls_selection,
        "controls_selection_sha256": (
            controls_selection.get("selection_sha256")
            if controls_selection is not None
            else None
        ),
        "cells": cells,
    }
    matrix["matrix_sha256"] = _matrix_digest(matrix)
    return matrix


def validate_controls_matrix(
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "status",
        "profile",
        "evidence_class",
        "claim_eligible",
        "controls_protocol_sha256",
        "parent_protocol_sha256",
        "source_sha256",
        "parent_source_sha256",
        "run_budget_tracks",
        "arm_specs",
        "arm_runs",
        "parent_candidates",
        "parent_selection",
        "parent_selection_sha256",
        "controls_selection",
        "controls_selection_sha256",
        "cells",
        "matrix_sha256",
    }
    _assert_exact_keys(matrix, required, "controls matrix")
    if (
        matrix["schema_version"] != MATRIX_SCHEMA
        or matrix["status"] != "frozen_before_control_outcomes"
        or matrix["claim_eligible"] is not False
        or matrix["matrix_sha256"] != _matrix_digest(matrix)
    ):
        raise ValueError("controls matrix identity/digest mismatch")
    expected = build_controls_matrix(
        protocol,
        parent_protocol,
        matrix["profile"],
        source_manifest=source_manifest,
        parent_selection=matrix.get("parent_selection"),
        controls_selection=matrix.get("controls_selection"),
    )
    if matrix != expected:
        raise ValueError("controls matrix differs from its frozen derivation")
    ids = [cell["cell_id"] for cell in matrix["cells"]]
    if len(set(ids)) != len(ids):
        raise ValueError("controls matrix contains duplicate cells")
    for cell in matrix["cells"]:
        payload = {
            key: value
            for key, value in cell.items()
            if key not in ("cell_id", "identity_sha256")
        }
        digest = object_sha256(payload)
        if cell["identity_sha256"] != digest or cell["cell_id"] != f"{cell['stage']}-{digest[:24]}":
            raise ValueError("controls cell identity mismatch")


def _run_by_id(matrix: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    matches = [row for row in matrix["arm_runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate control run {run_id!r}")
    return dict(matches[0])


def make_control_config(
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    run: Mapping[str, Any],
    observation_shape: Sequence[int],
    action_dim: int,
) -> Any:
    """Resolve one complete config from the parent's canonical template."""

    profile = matrix["profile"]
    parent_profile = {
        "smoke": "smoke",
        "development": "pilot",
        "confirmatory": "confirmatory",
    }[profile]
    family = run["family"]
    parent_arm = "shortcut_forcing" if family == "shortcut_forcing" else "trajectory_imf"
    base = parent.make_config(
        parent_protocol,
        parent_profile,
        parent_arm,
        observation_shape,
        action_dim,
        nfe=int(run["primary_nfe"]),
        candidate=matrix["parent_candidates"][parent_arm],
    )
    candidate = run["candidate"]
    prior_scale = float(candidate.get("prior_scale", base.prior_scale))
    if family == "shortcut_forcing":
        return replace(base, prior_scale=prior_scale)
    if family in ("trajectory_imf", "trajectory_endpoint_certificate"):
        # Factor settings enter the dynamic objective vector.  Keeping this
        # config identical across all 28 cells makes initialization, recurrent
        # structure, actor sampling, and inference exactly shared.
        return replace(base, prior_scale=prior_scale)
    if family in ("ordinary_imf", "temporal_increment_imf"):
        return replace(
            base,
            prior="imf",
            imf_trajectory_enabled=False,
            shortcut_training_k_max=None,
            imf_sampling_steps=1,
            prior_scale=prior_scale,
            overshooting_horizon=1,
            overshooting_distances=(1,),
            overshooting_scale=0.0,
        )
    if family == "gaussian_rssm":
        return replace(
            base,
            prior="gaussian",
            imf_trajectory_enabled=False,
            shortcut_training_k_max=None,
            prior_scale=prior_scale,
            overshooting_horizon=1,
            overshooting_distances=(1,),
            overshooting_scale=0.0,
        )
    raise ValueError(f"unknown controls family {family!r}")


def _masked_mean(value: Any, mask: Any) -> Any:
    import jax.numpy as jnp

    if value.ndim > 2:
        value = jnp.mean(value, axis=tuple(range(2, value.ndim)))
    mask = jnp.asarray(mask, dtype=value.dtype)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, value.dtype))


def trajectory_control_vector(run: Mapping[str, Any]) -> np.ndarray:
    spec = run["arm_spec"]
    if run["family"] not in ("trajectory_imf", "trajectory_endpoint_certificate"):
        raise ValueError("trajectory control vector requested for another family")
    return np.asarray(
        [
            1.0 if spec["query_history_noise_coupling"] == "shared" else 0.0,
            1.0 if spec["query_history_time_relation"] == "tied_on_corrupted_positions" else 0.0,
            *spec["context_probabilities"],
            1.0 if spec.get("endpoint_certificate_enabled") else 0.0,
            float(spec.get("endpoint_certificate_mass", 0.0)),
        ],
        dtype=np.float32,
    )


def sample_control_schedule(
    canonical_uniforms: Any,
    token_mask: Any,
    probabilities: Any,
    *,
    boundary_fraction: float,
    time_mean: float,
    time_std: float,
    history_noise_max: float,
) -> ControlSchedule:
    """Dynamic-probability form of the library's canonical mixed schedule.

    With probabilities ``(1/3,1/3,1/3)`` this is algebraically and
    numerically identical to ``sample_trajectory_schedule(...,
    canonical_uniforms=...)``.  Making only the probabilities dynamic lets all
    context subsets share one compiled factorial kernel.
    """

    import jax
    import jax.numpy as jnp

    uniforms = jnp.asarray(canonical_uniforms)
    batch_size, sequence_length, columns = uniforms.shape
    if columns != 6:
        raise ValueError("canonical controls uniforms need six columns")
    probabilities = jnp.asarray(probabilities, dtype=uniforms.dtype)
    epsilon = jnp.finfo(uniforms.dtype).eps
    query_uniforms = jnp.clip(uniforms[..., :2], epsilon, 1.0 - epsilon)
    logits = jax.scipy.special.ndtri(query_uniforms) * time_std + time_mean
    samples = jax.nn.sigmoid(logits)
    lower = jnp.min(samples, axis=-1, keepdims=True)
    t = jnp.max(samples, axis=-1, keepdims=True)
    r = jnp.where(uniforms[..., 2:3] < boundary_fraction, t, lower)
    history_t_random = history_noise_max * uniforms[..., 3:4]
    normalized = probabilities / jnp.sum(probabilities)
    cumulative = jnp.cumsum(normalized)
    pattern = jnp.sum(
        uniforms[:, 0, 4][:, None] >= cumulative[None, :], axis=-1
    ).astype(jnp.int32)
    base_mask = jnp.asarray(token_mask, dtype=uniforms.dtype)
    valid = base_mask > 0.0
    valid_count = jnp.sum(valid, axis=1, dtype=jnp.int32)
    suffix_rank = jnp.floor(
        uniforms[:, 0, 5] * jnp.maximum(valid_count, 1).astype(uniforms.dtype)
    ).astype(jnp.int32)
    cumulative_valid = jnp.cumsum(valid.astype(jnp.int32), axis=1)
    candidate = valid & (cumulative_valid > suffix_rank[:, None])
    positions = jnp.broadcast_to(
        jnp.arange(sequence_length, dtype=jnp.int32)[None, :], candidate.shape
    )
    suffix_start = jnp.min(
        jnp.where(candidate, positions, sequence_length), axis=1
    )
    suffix_start = jnp.where(valid_count > 0, suffix_start, 0)
    in_suffix = jnp.arange(sequence_length)[None, :] >= suffix_start[:, None]
    clean = jnp.zeros_like(history_t_random)
    suffix_history = jnp.where(in_suffix[..., None], history_t_random, clean)
    history_t = jnp.where(
        (pattern == 0)[:, None, None],
        clean,
        jnp.where((pattern == 1)[:, None, None], history_t_random, suffix_history),
    )
    suffix_loss_mask = base_mask * in_suffix.astype(uniforms.dtype)
    loss_mask = jnp.where((pattern == 2)[:, None], suffix_loss_mask, base_mask)
    return ControlSchedule(r, t, history_t, loss_mask, pattern, suffix_start)


def trajectory_control_loss(
    params: Any,
    batch: Mapping[str, Any],
    key: Any,
    config: Any,
    control: Any,
    *,
    endpoint_certificate_enabled: bool = False,
) -> ControlLoss:
    """The canonical trajectory objective with five dynamic factorial knobs."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        corrupt_trajectory,
        decode,
        observe_sequence,
        predict_continuation_logits,
        predict_reward,
        preprocess_observation,
        representation_kl,
        trajectory_deterministic_conditions,
        trajectory_imf_loss,
    )
    from imf_dreamer_jax.world_model import _binary_cross_entropy_with_logits

    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    continuations = batch["continuations"]
    is_first = jnp.asarray(batch["is_first"], dtype=jnp.bool_)
    loss_mask = jnp.asarray(batch["loss_mask"], dtype=actions.dtype)
    sequence_key, prior_key, _ = jax.random.split(key, 3)
    sequence = observe_sequence(
        params, observations, actions, sequence_key, config, is_first=is_first
    )
    feature = sequence.states.feature
    targets = preprocess_observation(observations)
    reconstruction = _masked_mean(jnp.square(decode(params, feature, config) - targets), loss_mask)
    reward = _masked_mean(
        jnp.square(predict_reward(params, feature, config) - rewards), loss_mask
    )
    continuation = _masked_mean(
        _binary_cross_entropy_with_logits(
            predict_continuation_logits(params, feature), continuations
        ),
        loss_mask,
    )
    representation = _masked_mean(
        representation_kl(sequence.posterior), loss_mask
    )
    target_stochastic = jax.lax.stop_gradient(sequence.states.stochastic)
    query_noise = jax.random.normal(
        jax.random.fold_in(prior_key, 0),
        target_stochastic.shape,
        dtype=target_stochastic.dtype,
    )
    schedule_uniforms = jax.random.uniform(
        jax.random.fold_in(prior_key, 1),
        (*target_stochastic.shape[:2], 6),
        dtype=target_stochastic.dtype,
    )
    probabilities = control[2:5]
    schedule = sample_control_schedule(
        schedule_uniforms,
        loss_mask,
        probabilities,
        boundary_fraction=config.imf_boundary_fraction,
        time_mean=config.imf_time_mean,
        time_std=config.imf_time_std,
        history_noise_max=config.imf_trajectory_history_noise_max,
    )
    independent_history_noise = jax.random.normal(
        jax.random.fold_in(prior_key, 4),
        target_stochastic.shape,
        dtype=target_stochastic.dtype,
    )
    shared = control[0]
    history_noise = shared * query_noise + (1.0 - shared) * independent_history_noise
    # Tying applies only where the chosen context component corrupts a token;
    # exact clean prefixes remain clean, so the three component factors retain
    # their intended meaning in the full factorial.
    tied_history_t = jnp.where(schedule.history_t > 0.0, schedule.t, 0.0)
    tied = control[1]
    history_t = tied * tied_history_t + (1.0 - tied) * schedule.history_t
    corrupted_history = corrupt_trajectory(target_stochastic, history_noise, history_t)
    conditions = trajectory_deterministic_conditions(
        params, corrupted_history, history_t, actions, is_first, config
    )
    details = trajectory_imf_loss(
        params["prior"],
        target_stochastic,
        conditions,
        jax.random.fold_in(prior_key, 3),
        noise=query_noise,
        r=schedule.r,
        t=schedule.t,
        token_mask=schedule.loss_mask,
        adaptive_power=config.imf_adaptive_power,
        adaptive_epsilon=config.imf_adaptive_epsilon,
        meanflow_scale=config.imf_meanflow_scale,
        velocity_scale=config.imf_velocity_scale,
        signal_weight_floor=config.imf_signal_weight_floor,
        signal_weight_scale=config.imf_signal_weight_scale,
        condition_gradient_scale=config.imf_condition_gradient_scale,
        boundary_velocity_supervision=True,
        return_details=True,
    )
    zero = jnp.asarray(0.0, dtype=details.loss.dtype)
    if endpoint_certificate_enabled:
        certificate_s = jax.random.uniform(
            jax.random.fold_in(prior_key, 5),
            (*target_stochastic.shape[:2], 1),
            dtype=target_stochastic.dtype,
        )
        certificate_details = trajectory_imf_loss(
            params["prior"],
            target_stochastic,
            conditions,
            jax.random.fold_in(prior_key, 6),
            noise=query_noise,
            r=jnp.zeros_like(certificate_s),
            t=certificate_s,
            token_mask=schedule.loss_mask,
            adaptive_power=0.0,
            adaptive_epsilon=config.imf_adaptive_epsilon,
            meanflow_scale=1.0,
            velocity_scale=1.0,
            signal_weight_floor=1.0,
            signal_weight_scale=0.0,
            condition_gradient_scale=config.imf_condition_gradient_scale,
            boundary_velocity_supervision=True,
            return_details=True,
        )
        certificate = certificate_details.loss
        prior_loss = details.loss + control[6] * certificate
        endpoint_metric = certificate
    else:
        prior_loss = details.loss
        endpoint_metric = zero
    total = (
        config.reconstruction_scale * reconstruction
        + config.reward_scale * reward
        + config.continuation_scale * continuation
        + config.prior_scale * prior_loss
        + config.representation_scale * representation
    )
    return ControlLoss(
        total,
        reconstruction,
        reward,
        continuation,
        prior_loss,
        representation,
        zero,
        endpoint_metric,
        _masked_mean(details.raw_loss_u, schedule.loss_mask),
        _masked_mean(details.raw_loss_v, schedule.loss_mask),
    )


def temporal_increment_loss(
    params: Any,
    batch: Mapping[str, Any],
    key: Any,
    config: Any,
    scale: Any,
) -> ControlLoss:
    """Ordinary iMF plus a transparent one-vs-two transport consistency term."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import observe_sequence, sample_imf_steps, world_model_loss

    base_key, sequence_key, noise_key = jax.random.split(key, 3)
    base = world_model_loss(params, batch, base_key, config)
    sequence = observe_sequence(
        params,
        batch["observations"],
        batch["actions"],
        sequence_key,
        config,
        is_first=batch["is_first"],
    )
    targets = jax.lax.stop_gradient(sequence.states.stochastic)
    conditions = sequence.states.deterministic
    flat_target = targets.reshape((-1, config.stochastic_dim))
    flat_condition = conditions.reshape((-1, config.deterministic_dim))
    noise = jax.random.normal(noise_key, flat_target.shape, dtype=flat_target.dtype)
    direct = sample_imf_steps(
        params["prior"], flat_condition, None, noise=noise, steps=1
    )
    teacher = sample_imf_steps(
        params["prior"], flat_condition, None, noise=noise, steps=2
    )
    per_token = jnp.mean(jnp.square(direct - jax.lax.stop_gradient(teacher)), axis=-1)
    consistency = _masked_mean(per_token.reshape(batch["loss_mask"].shape), batch["loss_mask"])
    total = base.total + scale * consistency
    return ControlLoss(
        total,
        base.reconstruction,
        base.reward,
        base.continuation,
        base.prior,
        base.representation,
        consistency,
        jnp.asarray(0.0, dtype=total.dtype),
        base.imf_loss_u,
        base.imf_loss_v,
    )


def train_trajectory_control(
    state: Any,
    batch: Mapping[str, Any],
    key: Any,
    config: Any,
    control: Any,
    *,
    endpoint_certificate_enabled: bool = False,
) -> tuple[Any, ControlLoss]:
    import jax
    from imf_dreamer_jax.optim import adam_update
    from imf_dreamer_jax.nn import clip_by_global_norm
    from imf_dreamer_jax.types import AgentParams, AgentState

    def objective(model_params: Any) -> tuple[Any, ControlLoss]:
        losses = trajectory_control_loss(
            model_params,
            batch,
            key,
            config,
            control,
            endpoint_certificate_enabled=endpoint_certificate_enabled,
        )
        return losses.total, losses

    (_, losses), gradients = jax.value_and_grad(objective, has_aux=True)(
        state.params.world_model
    )
    gradients, _ = clip_by_global_norm(gradients, config.grad_clip)
    model_params, optimizer = adam_update(
        state.params.world_model,
        gradients,
        state.model_optimizer,
        learning_rate=config.model_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    return (
        AgentState(
            AgentParams(model_params, state.params.actor, state.params.critic),
            optimizer,
            state.actor_optimizer,
            state.critic_optimizer,
            state.slow_critic,
        ),
        losses,
    )


def train_trajectory_endpoint_control(
    state: Any, batch: Mapping[str, Any], key: Any, config: Any, control: Any
) -> tuple[Any, ControlLoss]:
    """Statically include the exploratory endpoint certificate only in its arm."""

    return train_trajectory_control(
        state,
        batch,
        key,
        config,
        control,
        endpoint_certificate_enabled=True,
    )


def train_temporal_increment(
    state: Any, batch: Mapping[str, Any], key: Any, config: Any, scale: Any
) -> tuple[Any, ControlLoss]:
    import jax
    from imf_dreamer_jax.optim import adam_update
    from imf_dreamer_jax.nn import clip_by_global_norm
    from imf_dreamer_jax.types import AgentParams, AgentState

    def objective(model_params: Any) -> tuple[Any, ControlLoss]:
        losses = temporal_increment_loss(model_params, batch, key, config, scale)
        return losses.total, losses

    (_, losses), gradients = jax.value_and_grad(objective, has_aux=True)(
        state.params.world_model
    )
    gradients, _ = clip_by_global_norm(gradients, config.grad_clip)
    model_params, optimizer = adam_update(
        state.params.world_model,
        gradients,
        state.model_optimizer,
        learning_rate=config.model_learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
    )
    return (
        AgentState(
            AgentParams(model_params, state.params.actor, state.params.critic),
            optimizer,
            state.actor_optimizer,
            state.critic_optimizer,
            state.slow_critic,
        ),
        losses,
    )


def world_train_executable(run: Mapping[str, Any], config: Any) -> tuple[Any, Any]:
    """Return a jitted update and its dynamic objective argument."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import jit_train_world_model

    family = run["family"]
    if family == "trajectory_imf":
        executable = jax.jit(train_trajectory_control, static_argnames=("config",))
        argument = jnp.asarray(trajectory_control_vector(run))
    elif family == "trajectory_endpoint_certificate":
        executable = jax.jit(
            train_trajectory_endpoint_control, static_argnames=("config",)
        )
        argument = jnp.asarray(trajectory_control_vector(run))
    elif family == "temporal_increment_imf":
        executable = jax.jit(train_temporal_increment, static_argnames=("config",))
        argument = jnp.asarray(
            float(run["candidate"].get("temporal_increment_scale", 0.3)),
            dtype=jnp.float32,
        )
    else:
        executable = jit_train_world_model
        argument = None
    return executable, argument


def execute_world_update(
    executable: Any,
    argument: Any,
    state: Any,
    batch: Mapping[str, Any],
    key: Any,
    config: Any,
) -> tuple[Any, Any]:
    return (
        executable(state, batch, key, config, argument)
        if argument is not None
        else executable(state, batch, key, config)
    )


def lower_world_update(
    executable: Any,
    argument: Any,
    state: Any,
    batch: Mapping[str, Any],
    key: Any,
    config: Any,
) -> Any:
    return (
        executable.lower(state, batch, key, config, argument)
        if argument is not None
        else executable.lower(state, batch, key, config)
    )


def _parent_profile(protocol: Mapping[str, Any], profile: str) -> str:
    return str(protocol["profiles"][profile]["parent_profile"])


def _parent_matrix(
    parent_protocol: Mapping[str, Any],
    controls_matrix: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    profile = controls_matrix["profile"]
    parent_source = source_manifest["parent_source_manifest"]
    if profile == "smoke":
        return parent.build_matrix(parent_protocol, "smoke", source_manifest=parent_source)
    if profile == "development":
        return parent.build_pilot_hpo_matrix(parent_protocol, source_manifest=parent_source)
    return parent.build_matrix(
        parent_protocol,
        "confirmatory",
        source_manifest=parent_source,
        selection_manifest=controls_matrix["parent_selection"],
    )


def _parent_dataset_cell(
    parent_matrix: Mapping[str, Any], task: str, world_model_seed: int
) -> dict[str, Any]:
    matches = [
        cell
        for cell in parent_matrix["cells"]
        if cell["stage"] == "dataset"
        and cell["task"] == task
        and int(cell["world_model_seed"]) == int(world_model_seed)
    ]
    if len(matches) != 1:
        raise ValueError("controls unit lacks one canonical parent dataset cell")
    return dict(matches[0])


def _cell_directory(root: str | Path, cell: Mapping[str, Any]) -> Path:
    cell_id = cell.get("cell_id")
    if not isinstance(cell_id, str) or re.fullmatch(
        r"(?:compute|world|rollout|actor)-[0-9a-f]{24}", cell_id
    ) is None:
        raise ValueError("controls cell id is not a canonical path-safe identity")
    return Path(root) / "cells" / cell_id


def _cell_result_path(root: str | Path, cell: Mapping[str, Any]) -> Path:
    return _cell_directory(root, cell) / "result.json"


def _matrix_cell(matrix: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [cell for cell in matrix["cells"] if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate controls cell {cell_id!r}")
    return dict(matches[0])


def _compute_cell(matrix: Mapping[str, Any], task: str) -> dict[str, Any]:
    matches = [
        cell
        for cell in matrix["cells"]
        if cell["stage"] == "compute" and cell["task"] == task
    ]
    if len(matches) != 1:
        raise ValueError("controls task lacks exactly one compute cell")
    return dict(matches[0])


def _write_npz(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    return parent._write_npz_atomic(path, arrays)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    return parent.load_npz(path)


def _metrics(value: Any) -> dict[str, float]:
    mapping = value if isinstance(value, Mapping) else value._asdict()
    result = {str(key): float(np.asarray(item)) for key, item in mapping.items()}
    if not np.isfinite(np.asarray(list(result.values()), dtype=np.float64)).all():
        raise FloatingPointError("controls training produced a non-finite metric")
    return result


def _count_parameters(tree: Any) -> int:
    return parent._count_parameters(tree)


def _tree_digest(tree: Any) -> str:
    return parent._tree_digest(tree)


def _strict_source_ready(source: Mapping[str, Any]) -> None:
    for label, git in (
        ("parent", source["parent_source_manifest"]["git"]),
        ("controls", source["git"]),
    ):
        if (
            git.get("commit_status") != "complete"
            or git.get("dirty_patch_status") != "complete"
            or git.get("untracked_status") != "complete"
            or git.get("dirty_patch_bytes") != 0
            or git.get("untracked_files") != []
        ):
            raise ValueError(
                f"confirmatory controls require clean committed {label} source"
            )


def _read_selection_input(path: str | Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if source.is_dir():
        source = source / ("hpo_selection.json" if label == "parent" else "controls_selection.json")
    if not source.is_file():
        raise FileNotFoundError(f"{label} selection artifact is absent: {source}")
    return read_json(source)


def freeze_controls_run(
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    profile: str,
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
    parent_selection: Mapping[str, Any] | None = None,
    controls_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze an immutable controls matrix before any control outcome exists."""

    root = Path(output_root).resolve()
    workspace_root = _workspace_root(workspace)
    if (root / "FINALIZED.json").exists():
        raise ValueError("a finalized controls root is immutable")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("controls freeze requires a new empty output root")
    validate_controls_protocol(protocol, parent_protocol)
    source = build_controls_source_manifest(workspace_root)
    if profile in ("development", "confirmatory"):
        _strict_source_ready(source)
        if parent.runtime_fingerprint()["visible_device_count"] != 1:
            raise ValueError(
                "development/confirmatory controls require one visible device"
            )
    matrix = build_controls_matrix(
        protocol,
        parent_protocol,
        profile,
        source_manifest=source,
        parent_selection=parent_selection,
        controls_selection=controls_selection,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "frozen_controls_protocol.json", dict(protocol))
    write_json_atomic(root / "frozen_parent_protocol.json", dict(parent_protocol))
    write_json_atomic(root / "source_manifest.json", source)
    write_json_atomic(root / "matrix.json", matrix)
    parent_matrix = _parent_matrix(parent_protocol, matrix, source)
    write_json_atomic(root / "parent_dataset_matrix.json", parent_matrix)
    if parent_selection is not None:
        write_json_atomic(root / "parent_selection.json", dict(parent_selection))
    if controls_selection is not None:
        write_json_atomic(root / "controls_selection.json", dict(controls_selection))
    parent._write_environment_and_dependencies(
        root, strict=(profile == "confirmatory")
    )
    git = source["git"]
    (root / "source_identity.txt").write_text(
        f"commit={git['commit']}\n"
        f"dirty_patch_sha256={git['dirty_patch_sha256']}\n"
        f"source_sha256={source['source_sha256']}\n",
        encoding="utf-8",
    )
    freeze = {
        "schema_version": "trajectory-imf-neurips-controls-freeze-v1",
        "status": "frozen_before_control_outcomes",
        "profile": profile,
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "parent_protocol_sha256": parent_protocol_digest(parent_protocol),
        "source_sha256": source["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "registered_arm_count": len(expected_arm_specs(protocol)),
        "arm_count": len(matrix["arm_specs"]),
        "run_count": len(matrix["arm_runs"]),
        "cell_count": len(matrix["cells"]),
        "frozen_unix_time": time.time(),
    }
    freeze["freeze_sha256"] = object_sha256(freeze)
    write_json_atomic(root / "freeze.json", freeze)
    return freeze


def load_frozen_controls(root: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = Path(root)
    return (
        read_json(base / "frozen_controls_protocol.json"),
        read_json(base / "frozen_parent_protocol.json"),
        read_json(base / "source_manifest.json"),
        read_json(base / "matrix.json"),
    )


def run_canonical_datasets(
    output_root: str | Path,
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> int:
    root = Path(output_root)
    if (root / "FINALIZED.json").exists():
        raise ValueError("a finalized controls root is immutable")
    parent_matrix = read_json(root / "parent_dataset_matrix.json")
    parent_source = source["parent_source_manifest"]
    parent_profile = _parent_profile(protocol, matrix["profile"])
    if parent_profile == "pilot":
        parent.validate_pilot_hpo_matrix(parent_matrix, parent_protocol, parent_source)
    else:
        parent.validate_matrix(parent_matrix, parent_protocol, parent_source)
    count = 0
    selected = protocol["profiles"][matrix["profile"]]
    for task in selected["tasks"]:
        task = parent.canonical_task_id(task)
        for world_seed in selected["world_model_seeds"]:
            cell = _parent_dataset_cell(parent_matrix, task, int(world_seed))
            parent.run_dataset_cell(
                cell,
                parent_protocol,
                parent_matrix,
                root / "parent_dataset",
            )
            count += 1
    return count


def run_canonical_dataset_unit(
    output_root: str | Path,
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    dataset_cell_id: str | None = None,
    task: str | None = None,
    world_model_seed: int | None = None,
) -> dict[str, Any]:
    """Execute exactly one frozen parent dataset cell for array-safe dispatch."""

    if (Path(output_root) / "FINALIZED.json").exists():
        raise ValueError("a finalized controls root is immutable")
    parent_matrix = read_json(Path(output_root) / "parent_dataset_matrix.json")
    parent_source = source["parent_source_manifest"]
    parent_profile = _parent_profile(protocol, matrix["profile"])
    if parent_profile == "pilot":
        parent.validate_pilot_hpo_matrix(parent_matrix, parent_protocol, parent_source)
    else:
        parent.validate_matrix(parent_matrix, parent_protocol, parent_source)
    dataset_cells = [
        cell for cell in parent_matrix["cells"] if cell["stage"] == "dataset"
    ]
    if dataset_cell_id is not None:
        if task is not None or world_model_seed is not None:
            raise ValueError("select a dataset by cell id or task/seed, not both")
        matches = [cell for cell in dataset_cells if cell["cell_id"] == dataset_cell_id]
    else:
        if task is None or world_model_seed is None:
            raise ValueError("one dataset task and world-model seed are required")
        canonical_task = parent.canonical_task_id(task)
        matches = [
            cell
            for cell in dataset_cells
            if cell["task"] == canonical_task
            and int(cell["world_model_seed"]) == int(world_model_seed)
        ]
    if len(matches) != 1:
        raise ValueError("dataset selector does not resolve exactly one frozen cell")
    return parent.run_dataset_cell(
        matches[0], parent_protocol, parent_matrix, Path(output_root) / "parent_dataset"
    )


def _dataset_artifacts(
    root: str | Path,
    matrix: Mapping[str, Any],
    task: str,
    world_model_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    parent_matrix = read_json(Path(root) / "parent_dataset_matrix.json")
    cell = _parent_dataset_cell(parent_matrix, task, world_model_seed)
    directory = parent.stage_directory(Path(root) / "parent_dataset", cell)
    result = read_json(directory / "result.json")
    arrays = _load_npz(directory / "dataset.npz")
    cache_key = (
        str(directory.resolve()),
        file_sha256(directory / "result.json"),
        file_sha256(directory / "dataset.npz"),
        str(parent_matrix["matrix_sha256"]),
    )
    if cache_key not in _VALIDATED_DATASETS:
        parent_protocol = read_json(Path(root) / "frozen_parent_protocol.json")
        parent.validate_dataset_result(
            result,
            cell,
            directory / "dataset.npz",
            protocol=parent_protocol,
            matrix=parent_matrix,
        )
        _VALIDATED_DATASETS.add(cache_key)
    return result, arrays, directory


def _general_equal_flop_allocation(
    costs: Mapping[str, float],
    *,
    reference_run: str,
    reference_updates: int,
    tolerance: float,
    enforce: bool,
) -> tuple[float, dict[str, dict[str, float | int | bool]]]:
    if reference_run not in costs or reference_updates <= 0:
        raise ValueError("equal-FLOP reference is invalid")
    values = {key: float(value) for key, value in costs.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("compiler FLOP costs must be finite and positive")
    target = values[reference_run] * reference_updates
    allocation: dict[str, dict[str, float | int | bool]] = {}
    for run_id, cost in values.items():
        ratio = target / cost
        candidates = (max(1, math.floor(ratio)), max(1, math.ceil(ratio)))
        updates = min(candidates, key=lambda value: (abs(value * cost - target), value))
        cumulative = cost * updates
        relative = abs(cumulative - target) / target
        if enforce and relative > tolerance:
            raise ValueError(
                f"{run_id} nearest-integer allocation misses target by {relative:.6g}"
            )
        allocation[run_id] = {
            "flops_per_update": cost,
            "updates": int(updates),
            "cumulative_flops": cumulative,
            "relative_target_error": relative,
            "within_registered_tolerance": relative <= tolerance,
        }
    return target, allocation


def _write_content_addressed_text(root: Path, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = root / "compiler_ir" / f"{digest}.txt"
    if path.exists():
        if file_sha256(path) != digest:
            raise ValueError("content-addressed compiler IR was altered")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        # Writing source/result payloads is centralized here; atomic rename
        # prevents a killed compile from leaving a valid-looking IR file.
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    return digest


def _parameter_counts(state: Any, config: Any) -> dict[str, int]:
    from imf_dreamer_jax import world_model_parameter_counts

    world = world_model_parameter_counts(state.params.world_model, config)
    actor = _count_parameters(state.params.actor)
    critic = _count_parameters(state.params.critic)
    return {
        "world_model_total": int(world.total),
        "world_model_active": int(world.active),
        "prior_total": int(world.prior_total),
        "prior_active": int(world.prior_active),
        "actor": actor,
        "critic": critic,
        "agent_total": int(world.total) + actor + critic,
        "agent_active": int(world.active) + actor + critic,
    }


def build_controls_compute_plan(
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    task: str,
    observation_shape: Sequence[int],
    action_dim: int,
    output_root: str | Path,
) -> dict[str, Any]:
    """Compile exact update/inference graphs and allocate both matched tracks."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import create_agent, initial_state, jit_train_actor_critic

    task = parent.canonical_task_id(task)
    compute_cell = _compute_cell(matrix, task)
    directory = _cell_directory(output_root, compute_cell)
    result_path = directory / "result.json"
    if result_path.is_file():
        result = read_json(result_path)
        validate_controls_compute_plan(
            result, protocol, parent_protocol, matrix, compute_cell, output_root
        )
        return result
    directory.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {}
    world_costs: dict[str, float] = {}
    actor_costs: dict[str, float] = {}
    for run_index, run in enumerate(matrix["arm_runs"]):
        config = make_control_config(
            parent_protocol, matrix, run, observation_shape, action_dim
        )
        state = create_agent(
            config, parent.derive_jax_key("compile", matrix["profile"], task, run_index)
        )
        dummy = parent._dummy_batch(config, _parent_profile(protocol, matrix["profile"]))
        executable, argument = world_train_executable(run, config)
        world_lowered = lower_world_update(
            executable,
            argument,
            state,
            dummy,
            parent.derive_jax_key("compile-world", matrix["profile"], task, run_index),
            config,
        )
        world_unoptimized = world_lowered.as_text()
        world_compiled = world_lowered.compile()
        actor_lowered = jit_train_actor_critic.lower(
            state,
            initial_state(config, 32),
            parent.derive_jax_key("compile-actor", matrix["profile"], task, run_index),
            config,
        )
        actor_unoptimized = actor_lowered.as_text()
        actor_compiled = actor_lowered.compile()
        transition = jax.jit(
            lambda parameters, latent, action, noise: parent.inference_transition(
                parameters, latent, action, noise, config
            )
        )
        inference_lowered = transition.lower(
            state.params.world_model,
            initial_state(config, 1),
            jnp.zeros((1, action_dim), jnp.float32),
            jnp.zeros((1, config.stochastic_dim), jnp.float32),
        )
        inference_unoptimized = inference_lowered.as_text()
        inference_compiled = inference_lowered.compile()
        world_text = world_compiled.as_text()
        actor_text = actor_compiled.as_text()
        inference_text = inference_compiled.as_text()
        world_cost = parent._compiled_flops(world_compiled, f"{run['run_id']} world")
        actor_cost = parent._compiled_flops(actor_compiled, f"{run['run_id']} actor")
        inference_cost = parent._compiled_flops(
            inference_compiled, f"{run['run_id']} inference"
        )
        world_costs[run["run_id"]] = world_cost
        actor_costs[run["run_id"]] = actor_cost
        rows[run["run_id"]] = {
            "arm_id": run["arm_id"],
            "family": run["family"],
            "candidate": run["candidate"],
            "runtime_config": asdict(config),
            "runtime_config_sha256": object_sha256(asdict(config)),
            "objective_argument": (
                None if argument is None else np.asarray(argument).tolist()
            ),
            "objective_argument_sha256": object_sha256(
                None if argument is None else np.asarray(argument).tolist()
            ),
            "parameters": _parameter_counts(state, config),
            "compiler_flops": {
                "world_forward_and_backward_train": world_cost,
                "actor_forward_and_backward_train": actor_cost,
                "inference_per_transition": inference_cost,
            },
            "compiler_ir_sha256": {
                "world_train": _write_content_addressed_text(Path(output_root), world_text),
                "world_train_unoptimized": _write_content_addressed_text(
                    Path(output_root), world_unoptimized
                ),
                "actor_train": _write_content_addressed_text(Path(output_root), actor_text),
                "actor_train_unoptimized": _write_content_addressed_text(
                    Path(output_root), actor_unoptimized
                ),
                "inference": _write_content_addressed_text(Path(output_root), inference_text),
                "inference_unoptimized": _write_content_addressed_text(
                    Path(output_root), inference_unoptimized
                ),
            },
            "compiler_cost_analysis": {
                "world_train": parent._compiler_cost_analysis(
                    world_compiled, f"{run['run_id']} world"
                ),
                "actor_train": parent._compiler_cost_analysis(
                    actor_compiled, f"{run['run_id']} actor"
                ),
                "inference": parent._compiler_cost_analysis(
                    inference_compiled, f"{run['run_id']} inference"
                ),
            },
            "structural_nfe_per_transition": int(run["primary_nfe"]),
        }
    parent_profile = _parent_profile(protocol, matrix["profile"])
    budgets = parent_protocol["profiles"][parent_profile]["budgets"]
    equal_updates_world = int(budgets["equal_updates"]["world_model_updates"])
    equal_updates_actor = int(budgets["equal_updates"]["actor_updates"])
    reference = next(
        run["run_id"] for run in matrix["arm_runs"] if run["arm_id"] == "shortcut_forcing"
    )
    tolerance = float(protocol["matched_controls"]["equal_flop_relative_tolerance"])
    world_target, world_equal_flops = _general_equal_flop_allocation(
        world_costs,
        reference_run=reference,
        reference_updates=int(
            budgets["equal_compiler_flops"]["reference_world_model_updates"]
        ),
        tolerance=tolerance,
        enforce=matrix["profile"] == "confirmatory",
    )
    actor_target, actor_equal_flops = _general_equal_flop_allocation(
        actor_costs,
        reference_run=reference,
        reference_updates=int(
            budgets["equal_compiler_flops"]["reference_actor_updates"]
        ),
        tolerance=tolerance,
        enforce=matrix["profile"] == "confirmatory",
    )
    tracks = {
        "equal_updates": {
            "world_model_target_updates": equal_updates_world,
            "actor_target_updates": equal_updates_actor,
            "allocations": {
                run_id: {
                    "world_model": {
                        "flops_per_update": world_costs[run_id],
                        "updates": equal_updates_world,
                        "cumulative_flops": world_costs[run_id] * equal_updates_world,
                        "relative_target_error": None,
                        "within_registered_tolerance": None,
                    },
                    "actor": {
                        "flops_per_update": actor_costs[run_id],
                        "updates": equal_updates_actor,
                        "cumulative_flops": actor_costs[run_id] * equal_updates_actor,
                        "relative_target_error": None,
                        "within_registered_tolerance": None,
                    },
                }
                for run_id in world_costs
            },
        },
        "equal_compiler_flops": {
            "world_model_target_flops": world_target,
            "actor_target_flops": actor_target,
            "allocations": {
                run_id: {
                    "world_model": world_equal_flops[run_id],
                    "actor": actor_equal_flops[run_id],
                }
                for run_id in world_costs
            },
        },
    }
    result = {
        "schema_version": COMPUTE_SCHEMA,
        "status": "complete",
        "cell_id": compute_cell["cell_id"],
        "matrix_sha256": matrix["matrix_sha256"],
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "source_sha256": matrix["source_sha256"],
        "profile": matrix["profile"],
        "task": task,
        "runtime": parent.runtime_fingerprint(),
        "relative_flop_tolerance": tolerance,
        "arms": rows,
        "tracks": tracks,
    }
    result["plan_sha256"] = object_sha256(result)
    validate_controls_compute_plan(
        result, protocol, parent_protocol, matrix, compute_cell, output_root
    )
    write_json_atomic(result_path, result)
    return result


def validate_controls_compute_plan(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    cell: Mapping[str, Any],
    output_root: str | Path,
) -> None:
    required = {
        "schema_version",
        "status",
        "cell_id",
        "matrix_sha256",
        "controls_protocol_sha256",
        "source_sha256",
        "profile",
        "task",
        "runtime",
        "relative_flop_tolerance",
        "arms",
        "tracks",
        "plan_sha256",
    }
    _assert_exact_keys(result, required, "controls compute plan")
    if (
        result["schema_version"] != COMPUTE_SCHEMA
        or result["status"] != "complete"
        or result["cell_id"] != cell["cell_id"]
        or result["matrix_sha256"] != matrix["matrix_sha256"]
        or result["controls_protocol_sha256"] != controls_protocol_digest(protocol)
        or result["source_sha256"] != matrix["source_sha256"]
        or result["plan_sha256"] != object_sha256(
            _without_digest(result, "plan_sha256")
        )
    ):
        raise ValueError("controls compute identity/digest mismatch")
    if (
        result["profile"] != matrix["profile"]
        or result["task"] != parent.canonical_task_id(cell["task"])
        or result["relative_flop_tolerance"]
        != float(protocol["matched_controls"]["equal_flop_relative_tolerance"])
    ):
        raise ValueError("controls compute profile/task/tolerance mismatch")
    runtime = result["runtime"]
    runtime_keys = {
        "python",
        "jax_version",
        "jaxlib_version",
        "backend",
        "device_platforms",
        "device_kinds",
        "devices",
        "visible_device_count",
        "xla_platform_version",
        "jax_enable_x64",
        "cuda_visible_devices",
    }
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != runtime_keys
        or runtime.get("jax_enable_x64") is not False
        or not isinstance(runtime.get("visible_device_count"), int)
        or runtime["visible_device_count"] <= 0
        or runtime["visible_device_count"] != len(runtime.get("devices", []))
        or runtime["visible_device_count"] != len(runtime.get("device_platforms", []))
        or runtime["visible_device_count"] != len(runtime.get("device_kinds", []))
    ):
        raise ValueError("controls compute runtime fingerprint is incomplete")
    parent.runtime_homogeneity_identity(runtime)
    if set(result["arms"]) != {run["run_id"] for run in matrix["arm_runs"]}:
        raise ValueError("controls compute arm set is incomplete")
    if set(result["tracks"]) != set(BUDGET_TRACK_ORDER):
        raise ValueError("controls compute tracks are incomplete")
    cache_key = (
        str(Path(output_root).resolve()),
        str(result["plan_sha256"]),
        str(matrix["matrix_sha256"]),
    )
    if cache_key in _VALIDATED_COMPUTE_PLANS:
        for row in result["arms"].values():
            for digest in row.get("compiler_ir_sha256", {}).values():
                path = Path(output_root) / "compiler_ir" / f"{digest}.txt"
                if not path.is_file() or path.is_symlink() or file_sha256(path) != digest:
                    raise ValueError("controls compiler IR artifact digest mismatch")
        return
    world_costs: dict[str, float] = {}
    actor_costs: dict[str, float] = {}
    from imf_dreamer_jax import create_agent

    row_keys = {
        "arm_id",
        "family",
        "candidate",
        "runtime_config",
        "runtime_config_sha256",
        "objective_argument",
        "objective_argument_sha256",
        "parameters",
        "compiler_flops",
        "compiler_ir_sha256",
        "compiler_cost_analysis",
        "structural_nfe_per_transition",
    }
    parameter_keys = {
        "world_model_total",
        "world_model_active",
        "prior_total",
        "prior_active",
        "actor",
        "critic",
        "agent_total",
        "agent_active",
    }
    flop_keys = {
        "world_forward_and_backward_train",
        "actor_forward_and_backward_train",
        "inference_per_transition",
    }
    ir_keys = {
        "world_train",
        "world_train_unoptimized",
        "actor_train",
        "actor_train_unoptimized",
        "inference",
        "inference_unoptimized",
    }
    cost_keys = {"world_train", "actor_train", "inference"}
    for run_index, run in enumerate(matrix["arm_runs"]):
        row = result["arms"][run["run_id"]]
        _assert_exact_keys(row, row_keys, "controls compute arm")
        if (
            row.get("arm_id") != run["arm_id"]
            or row.get("family") != run["family"]
            or row.get("candidate") != run["candidate"]
            or row.get("runtime_config_sha256")
            != object_sha256(row.get("runtime_config"))
            or row.get("objective_argument_sha256")
            != object_sha256(row.get("objective_argument"))
            or row.get("structural_nfe_per_transition") != run["primary_nfe"]
        ):
            raise ValueError("controls compute arm identity/config mismatch")
        config = make_control_config(
            parent_protocol,
            matrix,
            run,
            row["runtime_config"]["observation_shape"],
            int(row["runtime_config"]["action_dim"]),
        )
        if canonical_bytes(row["runtime_config"]) != canonical_bytes(asdict(config)):
            raise ValueError("controls compute runtime config does not rederive")
        _, argument = world_train_executable(run, config)
        expected_argument = None if argument is None else np.asarray(argument).tolist()
        if canonical_bytes(row["objective_argument"]) != canonical_bytes(expected_argument):
            raise ValueError("controls compute objective argument does not rederive")
        parameters = row["parameters"]
        if set(parameters) != parameter_keys or any(
            not isinstance(parameters[key], int) or parameters[key] <= 0
            for key in parameter_keys
        ):
            raise ValueError("controls compute parameter counts are invalid")
        if (
            parameters["world_model_active"] > parameters["world_model_total"]
            or parameters["prior_active"] > parameters["prior_total"]
            or parameters["agent_active"] > parameters["agent_total"]
            or parameters["agent_total"]
            != parameters["world_model_total"] + parameters["actor"] + parameters["critic"]
            or parameters["agent_active"]
            != parameters["world_model_active"] + parameters["actor"] + parameters["critic"]
        ):
            raise ValueError("controls active/total parameter accounting is inconsistent")
        expected_state = create_agent(
            config,
            parent.derive_jax_key("compile", matrix["profile"], cell["task"], run_index),
        )
        if parameters != _parameter_counts(expected_state, config):
            raise ValueError("controls parameter counts do not rederive")
        if set(row["compiler_ir_sha256"]) != ir_keys:
            raise ValueError("controls compiler IR digest set is incomplete")
        for digest in row["compiler_ir_sha256"].values():
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("controls compiler IR digest is invalid")
            path = Path(output_root) / "compiler_ir" / f"{digest}.txt"
            if not path.is_file() or path.is_symlink() or file_sha256(path) != digest:
                raise ValueError("controls compiler IR artifact digest mismatch")
        if set(row["compiler_flops"]) != flop_keys or set(row["compiler_cost_analysis"]) != cost_keys:
            raise ValueError("controls compiler FLOP/cost-analysis fields are incomplete")
        for value in row["compiler_flops"].values():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("controls compiler FLOP evidence is invalid")
        expected_from_analysis: dict[str, float] = {}
        for name, rows in row["compiler_cost_analysis"].items():
            if not isinstance(rows, list) or not rows:
                raise ValueError("controls raw compiler cost analysis is empty")
            total = 0.0
            for cost_row in rows:
                if not isinstance(cost_row, Mapping) or not cost_row:
                    raise ValueError("controls raw compiler cost row is invalid")
                values = [float(value) for value in cost_row.values()]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("controls raw compiler cost row is non-finite")
                total += float(cost_row.get("flops", 0.0))
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("controls raw compiler FLOP total is invalid")
            expected_from_analysis[name] = total
        if row["compiler_flops"] != {
            "world_forward_and_backward_train": expected_from_analysis["world_train"],
            "actor_forward_and_backward_train": expected_from_analysis["actor_train"],
            "inference_per_transition": expected_from_analysis["inference"],
        }:
            raise ValueError("controls compiler FLOPs do not rederive from raw analysis")
        world_costs[run["run_id"]] = float(
            row["compiler_flops"]["world_forward_and_backward_train"]
        )
        actor_costs[run["run_id"]] = float(
            row["compiler_flops"]["actor_forward_and_backward_train"]
        )
    parent_profile = _parent_profile(protocol, matrix["profile"])
    budgets = parent_protocol["profiles"][parent_profile]["budgets"]
    equal_world = int(budgets["equal_updates"]["world_model_updates"])
    equal_actor = int(budgets["equal_updates"]["actor_updates"])
    expected_equal_updates = {
        "world_model_target_updates": equal_world,
        "actor_target_updates": equal_actor,
        "allocations": {
            run_id: {
                "world_model": {
                    "flops_per_update": world_costs[run_id],
                    "updates": equal_world,
                    "cumulative_flops": world_costs[run_id] * equal_world,
                    "relative_target_error": None,
                    "within_registered_tolerance": None,
                },
                "actor": {
                    "flops_per_update": actor_costs[run_id],
                    "updates": equal_actor,
                    "cumulative_flops": actor_costs[run_id] * equal_actor,
                    "relative_target_error": None,
                    "within_registered_tolerance": None,
                },
            }
            for run_id in world_costs
        },
    }
    if result["tracks"]["equal_updates"] != expected_equal_updates:
        raise ValueError("controls equal-update allocation does not rederive")
    reference = next(
        run["run_id"] for run in matrix["arm_runs"] if run["arm_id"] == "shortcut_forcing"
    )
    tolerance = float(protocol["matched_controls"]["equal_flop_relative_tolerance"])
    world_target, world_allocations = _general_equal_flop_allocation(
        world_costs,
        reference_run=reference,
        reference_updates=int(
            budgets["equal_compiler_flops"]["reference_world_model_updates"]
        ),
        tolerance=tolerance,
        enforce=matrix["profile"] == "confirmatory",
    )
    actor_target, actor_allocations = _general_equal_flop_allocation(
        actor_costs,
        reference_run=reference,
        reference_updates=int(
            budgets["equal_compiler_flops"]["reference_actor_updates"]
        ),
        tolerance=tolerance,
        enforce=matrix["profile"] == "confirmatory",
    )
    expected_equal_flops = {
        "world_model_target_flops": world_target,
        "actor_target_flops": actor_target,
        "allocations": {
            run_id: {
                "world_model": world_allocations[run_id],
                "actor": actor_allocations[run_id],
            }
            for run_id in world_costs
        },
    }
    if result["tracks"]["equal_compiler_flops"] != expected_equal_flops:
        raise ValueError("controls equal-compiler-FLOP allocation does not rederive")
    _VALIDATED_COMPUTE_PLANS.add(cache_key)


def run_all_compute_cells(
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> int:
    selected = protocol["profiles"][matrix["profile"]]
    count = 0
    for task in selected["tasks"]:
        task = parent.canonical_task_id(task)
        exemplar_seed = int(selected["world_model_seeds"][0])
        dataset, _, _ = _dataset_artifacts(output_root, matrix, task, exemplar_seed)
        build_controls_compute_plan(
            protocol,
            parent_protocol,
            matrix,
            task,
            dataset["observation_shape"],
            int(dataset["action_dim"]),
            output_root,
        )
        count += 1
    return count


def _execution_row(protocol: Mapping[str, Any], matrix: Mapping[str, Any]) -> Mapping[str, Any]:
    return parent.EXECUTION_SPEC["profiles"][_parent_profile(protocol, matrix["profile"])]


def _allocation(
    compute: Mapping[str, Any], cell: Mapping[str, Any], stage: str
) -> Mapping[str, Any]:
    return compute["tracks"][cell["budget_track"]]["allocations"][cell["run_id"]][stage]


def run_world_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    if cell["stage"] != "world":
        raise ValueError("run_world_cell requires a world cell")
    result_path = _cell_result_path(output_root, cell)
    if result_path.is_file():
        result = read_json(result_path)
        validate_world_cell(result, cell, protocol, parent_protocol, matrix, output_root)
        return result
    import jax
    from imf_dreamer_jax import create_agent, load_checkpoint, save_checkpoint

    run = _run_by_id(matrix, cell["run_id"])
    dataset_result, arrays, _ = _dataset_artifacts(
        output_root, matrix, cell["task"], int(cell["world_model_seed"])
    )
    compute_cell = _matrix_cell(matrix, cell["compute_cell_id"])
    compute = read_json(_cell_result_path(output_root, compute_cell))
    validate_controls_compute_plan(
        compute, protocol, parent_protocol, matrix, compute_cell, output_root
    )
    config = make_control_config(
        parent_protocol,
        matrix,
        run,
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
    )
    allocation = _allocation(compute, cell, "world_model")
    updates = int(allocation["updates"])
    execution = _execution_row(protocol, matrix)
    batch_size = int(execution["batch_size"])
    sequence_length = int(execution["sequence_length"])
    schedule = parent._batch_schedule(
        arrays,
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        updates=updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    directory = _cell_directory(output_root, cell)
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = _write_npz(directory / "batch_schedule.npz", schedule)
    objective_base_key = parent.derive_jax_key(
        "world-objective", cell["task"], cell["world_model_seed"]
    )
    objective_rows = parent._folded_key_rows(objective_base_key, updates)
    key_path = _write_npz(
        directory / "objective_rng_keys.npz",
        {
            "world_model_loss_keys": objective_rows,
            "base_noise_fold_in": np.asarray([0], np.int32),
            "canonical_uniform_fold_in": np.asarray([1], np.int32),
            "schedule_fold_in": np.asarray([2], np.int32),
            "loss_fold_in": np.asarray([3], np.int32),
            "independent_history_noise_fold_in": np.asarray([4], np.int32),
            "endpoint_certificate_time_fold_in": np.asarray([5], np.int32),
            "endpoint_certificate_loss_fold_in": np.asarray([6], np.int32),
        },
    )
    initialization_key = parent.derive_jax_key(
        "world-init", cell["task"], cell["world_model_seed"]
    )
    initial = create_agent(config, initialization_key)
    checkpoint_path = directory / "checkpoint.pkl"
    state = initial
    start_update = 0
    accumulated_wall = 0.0
    latest: dict[str, float] | None = None
    if checkpoint_path.is_file():
        state, stored_config, metadata = load_checkpoint(checkpoint_path)
        if stored_config != config or metadata.get("cell_id") != cell["cell_id"]:
            raise ValueError("partial controls world checkpoint identity mismatch")
        start_update = int(metadata.get("completed_updates", -1))
        accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        latest = metadata.get("last_metrics")
    if not 0 <= start_update <= updates:
        raise ValueError("partial controls world update count is invalid")
    executable, argument = world_train_executable(run, config)
    started = time.perf_counter()
    checkpoint_every = int(execution["checkpoint_every_updates"])
    for update in range(start_update, updates):
        batch = parent._materialize_batch(
            arrays,
            schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        state, losses = execute_world_update(
            executable,
            argument,
            state,
            batch,
            jax.random.fold_in(objective_base_key, update),
            config,
        )
        latest = _metrics(losses)
        completed = update + 1
        if completed % checkpoint_every == 0 or completed == updates:
            save_checkpoint(
                checkpoint_path,
                state,
                config,
                metadata={
                    "stage": "controls_world",
                    "cell_id": cell["cell_id"],
                    "run_id": cell["run_id"],
                    "dataset_sha256": dataset_result["dataset_sha256"],
                    "completed_updates": completed,
                    "last_metrics": latest,
                    "wall_seconds": accumulated_wall + time.perf_counter() - started,
                },
            )
    if latest is None:
        raise RuntimeError("controls world cell completed no update")
    total_wall = accumulated_wall + time.perf_counter() - started
    config_payload = asdict(config)
    result = {
        "schema_version": WORLD_SCHEMA,
        "status": "complete",
        "cell_id": cell["cell_id"],
        "matrix_sha256": matrix["matrix_sha256"],
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "source_sha256": matrix["source_sha256"],
        "profile": matrix["profile"],
        "task": cell["task"],
        "world_model_seed": int(cell["world_model_seed"]),
        "budget_track": cell["budget_track"],
        "run_id": cell["run_id"],
        "arm_id": cell["arm_id"],
        "dataset_sha256": dataset_result["dataset_sha256"],
        "dataset_result_sha256": file_sha256(
            _dataset_artifacts(
                output_root, matrix, cell["task"], int(cell["world_model_seed"])
            )[2]
            / "result.json"
        ),
        "compute_plan_sha256": compute["plan_sha256"],
        "updates": updates,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "runtime_config": config_payload,
        "runtime_config_sha256": object_sha256(config_payload),
        "objective_argument": (
            None if argument is None else np.asarray(argument).tolist()
        ),
        "objective_argument_sha256": object_sha256(
            None if argument is None else np.asarray(argument).tolist()
        ),
        "initial_world_parameter_sha256": _tree_digest(initial.params.world_model),
        "final_world_parameter_sha256": _tree_digest(state.params.world_model),
        "final_metrics": latest,
        "compiler_flops_per_update": float(allocation["flops_per_update"]),
        "realized_compiler_flops": float(allocation["flops_per_update"]) * updates,
        "batch_schedule_sha256": parent.array_sha256(schedule),
        "batch_schedule_file_sha256": file_sha256(schedule_path),
        "objective_rng_keys_file_sha256": file_sha256(key_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "runtime": parent.runtime_fingerprint(),
        "wall_seconds": total_wall,
    }
    validate_world_cell(result, cell, protocol, parent_protocol, matrix, output_root)
    write_json_atomic(result_path, result)
    return result


def validate_world_cell(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> None:
    required_result_keys = {
        "schema_version",
        "status",
        "cell_id",
        "matrix_sha256",
        "controls_protocol_sha256",
        "source_sha256",
        "profile",
        "task",
        "world_model_seed",
        "budget_track",
        "run_id",
        "arm_id",
        "dataset_sha256",
        "dataset_result_sha256",
        "compute_plan_sha256",
        "updates",
        "batch_size",
        "sequence_length",
        "runtime_config",
        "runtime_config_sha256",
        "objective_argument",
        "objective_argument_sha256",
        "initial_world_parameter_sha256",
        "final_world_parameter_sha256",
        "final_metrics",
        "compiler_flops_per_update",
        "realized_compiler_flops",
        "batch_schedule_sha256",
        "batch_schedule_file_sha256",
        "objective_rng_keys_file_sha256",
        "checkpoint_sha256",
        "runtime",
        "wall_seconds",
    }
    _assert_exact_keys(result, required_result_keys, "controls world result")
    if result.get("schema_version") != WORLD_SCHEMA or result.get("status") != "complete":
        raise ValueError("controls world result schema/status mismatch")
    identity = {
        "cell_id": cell["cell_id"],
        "matrix_sha256": matrix["matrix_sha256"],
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "source_sha256": matrix["source_sha256"],
        "profile": matrix["profile"],
        "task": cell["task"],
        "world_model_seed": int(cell["world_model_seed"]),
        "budget_track": cell["budget_track"],
        "run_id": cell["run_id"],
        "arm_id": cell["arm_id"],
    }
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("controls world result identity mismatch")
    run = _run_by_id(matrix, cell["run_id"])
    dataset_result, arrays, dataset_directory = _dataset_artifacts(
        output_root, matrix, cell["task"], int(cell["world_model_seed"])
    )
    compute_cell = _matrix_cell(matrix, cell["compute_cell_id"])
    compute = read_json(_cell_result_path(output_root, compute_cell))
    validate_controls_compute_plan(
        compute, protocol, parent_protocol, matrix, compute_cell, output_root
    )
    config = make_control_config(
        parent_protocol,
        matrix,
        run,
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
    )
    allocation = _allocation(compute, cell, "world_model")
    execution = _execution_row(protocol, matrix)
    updates = int(allocation["updates"])
    schedule = _load_npz(_cell_directory(output_root, cell) / "batch_schedule.npz")
    expected_schedule = parent._batch_schedule(
        arrays,
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        updates=updates,
        batch_size=int(execution["batch_size"]),
        sequence_length=int(execution["sequence_length"]),
    )
    if set(schedule) != set(expected_schedule) or any(
        not np.array_equal(schedule[key], expected_schedule[key]) for key in schedule
    ):
        raise ValueError("controls world minibatch schedule is not canonical")
    keys = _load_npz(_cell_directory(output_root, cell) / "objective_rng_keys.npz")
    expected_key_rows = parent._folded_key_rows(
        parent.derive_jax_key("world-objective", cell["task"], cell["world_model_seed"]),
        updates,
    )
    expected_key_artifact = {
        "world_model_loss_keys": expected_key_rows,
        "base_noise_fold_in": np.asarray([0], np.int32),
        "canonical_uniform_fold_in": np.asarray([1], np.int32),
        "schedule_fold_in": np.asarray([2], np.int32),
        "loss_fold_in": np.asarray([3], np.int32),
        "independent_history_noise_fold_in": np.asarray([4], np.int32),
        "endpoint_certificate_time_fold_in": np.asarray([5], np.int32),
        "endpoint_certificate_loss_fold_in": np.asarray([6], np.int32),
    }
    if set(keys) != set(expected_key_artifact) or any(
        not np.array_equal(keys[name], expected)
        for name, expected in expected_key_artifact.items()
    ):
        raise ValueError("controls world objective keys are not canonical")
    from imf_dreamer_jax import create_agent, load_checkpoint

    initial = create_agent(
        config,
        parent.derive_jax_key("world-init", cell["task"], cell["world_model_seed"]),
    )
    checkpoint = _cell_directory(output_root, cell) / "checkpoint.pkl"
    state, stored_config, metadata = load_checkpoint(checkpoint)
    _, argument = world_train_executable(run, config)
    expected_argument = None if argument is None else np.asarray(argument).tolist()
    final_metrics = result.get("final_metrics")
    if (
        not isinstance(final_metrics, Mapping)
        or not final_metrics
        or not all(math.isfinite(float(value)) for value in final_metrics.values())
        or canonical_bytes(metadata.get("last_metrics")) != canonical_bytes(final_metrics)
    ):
        raise ValueError("controls world final metrics are invalid or differ from checkpoint")
    if (
        stored_config != config
        or metadata.get("stage") != "controls_world"
        or metadata.get("cell_id") != cell["cell_id"]
        or metadata.get("run_id") != cell["run_id"]
        or metadata.get("dataset_sha256") != dataset_result["dataset_sha256"]
        or metadata.get("completed_updates") != updates
        or int(np.asarray(state.model_optimizer.step)) != updates
        or result.get("dataset_sha256") != dataset_result["dataset_sha256"]
        or result.get("dataset_result_sha256") != file_sha256(dataset_directory / "result.json")
        or result.get("runtime_config_sha256") != object_sha256(asdict(config))
        or canonical_bytes(result.get("runtime_config")) != canonical_bytes(asdict(config))
        or result.get("objective_argument_sha256")
        != object_sha256(result.get("objective_argument"))
        or canonical_bytes(result.get("objective_argument"))
        != canonical_bytes(expected_argument)
        or result.get("initial_world_parameter_sha256") != _tree_digest(initial.params.world_model)
        or result.get("final_world_parameter_sha256") != _tree_digest(state.params.world_model)
        or result.get("checkpoint_sha256") != file_sha256(checkpoint)
        or result.get("updates") != updates
        or result.get("batch_size") != int(execution["batch_size"])
        or result.get("sequence_length") != int(execution["sequence_length"])
        or result.get("compute_plan_sha256") != compute["plan_sha256"]
        or result.get("compiler_flops_per_update") != float(allocation["flops_per_update"])
        or result.get("realized_compiler_flops")
        != float(allocation["flops_per_update"]) * updates
        or result.get("batch_schedule_sha256") != parent.array_sha256(schedule)
        or result.get("batch_schedule_file_sha256")
        != file_sha256(_cell_directory(output_root, cell) / "batch_schedule.npz")
        or result.get("objective_rng_keys_file_sha256")
        != file_sha256(_cell_directory(output_root, cell) / "objective_rng_keys.npz")
    ):
        raise ValueError("controls world checkpoint/config/accounting mismatch")
    if (
        not math.isfinite(float(result["wall_seconds"]))
        or float(result["wall_seconds"]) < 0.0
        or not math.isfinite(float(metadata.get("wall_seconds", math.nan)))
    ):
        raise ValueError("controls world wall-clock evidence is invalid")
    if parent.runtime_homogeneity_identity(result["runtime"]) != parent.runtime_homogeneity_identity(
        compute["runtime"]
    ):
        raise ValueError("controls world runtime differs from compiler runtime")


def run_rollout_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    if cell["stage"] != "rollout":
        raise ValueError("run_rollout_cell requires a rollout cell")
    result_path = _cell_result_path(output_root, cell)
    if result_path.is_file():
        result = read_json(result_path)
        validate_rollout_cell(result, cell, protocol, parent_protocol, matrix, output_root)
        return result
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import load_checkpoint
    world_cell = _matrix_cell(matrix, cell["world_cell_id"])
    world_result = read_json(_cell_result_path(output_root, world_cell))
    validate_world_cell(
        world_result, world_cell, protocol, parent_protocol, matrix, output_root
    )
    run = _run_by_id(matrix, cell["run_id"])
    dataset_result, dataset, dataset_directory = _dataset_artifacts(
        output_root, matrix, cell["task"], int(cell["world_model_seed"])
    )
    config = make_control_config(
        parent_protocol,
        matrix,
        run,
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
    )
    state, stored_config, metadata = load_checkpoint(
        _cell_directory(output_root, world_cell) / "checkpoint.pkl"
    )
    if stored_config != config or metadata.get("cell_id") != world_cell["cell_id"]:
        raise ValueError("controls rollout world checkpoint identity mismatch")
    windows = parent._rollout_windows(
        dataset,
        parent_protocol,
        _parent_profile(protocol, matrix["profile"]),
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        stochastic_dim=config.stochastic_dim,
    )
    started = time.perf_counter()
    start = parent.posterior_mean_filter(
        state.params.world_model,
        jnp.asarray(windows["context_observations"]),
        jnp.asarray(windows["context_actions"]),
        config,
    )
    sampler = jax.jit(parent.open_loop_samples_with_continuation, static_argnames=("config",))
    observations, rewards, continuations = sampler(
        state.params.world_model,
        start,
        jnp.asarray(windows["future_actions"]),
        jnp.asarray(windows["noise"]),
        config,
    )
    raw = {
        "observation_samples": np.asarray(observations),
        "reward_samples": np.asarray(rewards),
        "continuation_samples": np.asarray(continuations),
        "target_observations": windows["target_observations"],
        "target_rewards": windows["target_rewards"],
        "target_continuations": windows["target_continuations"],
        "context_observations": windows["context_observations"],
        "context_actions": windows["context_actions"],
        "future_actions": windows["future_actions"],
        "noise": windows["noise"],
        "episode_ids": windows["episode_ids"],
        "anchors": windows["anchors"],
        "training_observation_std": windows["training_observation_std"],
    }
    metrics = parent.normalized_rollout_statistics(
        raw["observation_samples"],
        raw["reward_samples"],
        windows,
        protocol["evaluation"]["rollout_horizons"],
        raw["continuation_samples"],
    )
    directory = _cell_directory(output_root, cell)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = _write_npz(directory / "predictive_draws.npz", raw)
    result = {
        "schema_version": ROLLOUT_SCHEMA,
        "status": "complete",
        "cell_id": cell["cell_id"],
        "matrix_sha256": matrix["matrix_sha256"],
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "source_sha256": matrix["source_sha256"],
        "profile": matrix["profile"],
        "task": cell["task"],
        "world_model_seed": int(cell["world_model_seed"]),
        "budget_track": cell["budget_track"],
        "run_id": cell["run_id"],
        "arm_id": cell["arm_id"],
        "world_cell_id": world_cell["cell_id"],
        "world_checkpoint_sha256": world_result["checkpoint_sha256"],
        "dataset_sha256": dataset_result["dataset_sha256"],
        "dataset_result_sha256": file_sha256(dataset_directory / "result.json"),
        "runtime_config": asdict(config),
        "runtime_config_sha256": object_sha256(asdict(config)),
        "structural_nfe_per_transition": int(run["primary_nfe"]),
        "windows": int(len(windows["episode_ids"])),
        "predictive_draws_per_window": int(windows["noise"].shape[0]),
        **metrics,
        "raw_predictive_draws_sha256": file_sha256(raw_path),
        "runtime": parent.runtime_fingerprint(),
        "wall_seconds": time.perf_counter() - started,
    }
    validate_rollout_cell(result, cell, protocol, parent_protocol, matrix, output_root)
    write_json_atomic(result_path, result)
    return result


def _replay_control_rollout(
    raw: Mapping[str, np.ndarray],
    world_cell: Mapping[str, Any],
    config: Any,
    output_root: str | Path,
) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import load_checkpoint
    state, _, _ = load_checkpoint(_cell_directory(output_root, world_cell) / "checkpoint.pkl")
    start = parent.posterior_mean_filter(
        state.params.world_model,
        jnp.asarray(raw["context_observations"]),
        jnp.asarray(raw["context_actions"]),
        config,
    )
    sampler = jax.jit(parent.open_loop_samples_with_continuation, static_argnames=("config",))
    observations, rewards, continuations = sampler(
        state.params.world_model,
        start,
        jnp.asarray(raw["future_actions"]),
        jnp.asarray(raw["noise"]),
        config,
    )
    return {
        "observation_samples": np.asarray(observations),
        "reward_samples": np.asarray(rewards),
        "continuation_samples": np.asarray(continuations),
    }


def validate_rollout_cell(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> None:
    required_result_keys = {
        "schema_version",
        "status",
        "cell_id",
        "matrix_sha256",
        "controls_protocol_sha256",
        "source_sha256",
        "profile",
        "task",
        "world_model_seed",
        "budget_track",
        "run_id",
        "arm_id",
        "world_cell_id",
        "world_checkpoint_sha256",
        "dataset_sha256",
        "dataset_result_sha256",
        "runtime_config",
        "runtime_config_sha256",
        "structural_nfe_per_transition",
        "windows",
        "predictive_draws_per_window",
        "per_horizon",
        "normalized_free_running_rollout_error_auc",
        "auc_coordinates",
        "auc_rule",
        "raw_predictive_draws_sha256",
        "runtime",
        "wall_seconds",
    }
    _assert_exact_keys(result, required_result_keys, "controls rollout result")
    if result.get("schema_version") != ROLLOUT_SCHEMA or result.get("status") != "complete":
        raise ValueError("controls rollout schema/status mismatch")
    if any(
        result.get(key) != value
        for key, value in {
            "cell_id": cell["cell_id"],
            "matrix_sha256": matrix["matrix_sha256"],
            "controls_protocol_sha256": controls_protocol_digest(protocol),
            "source_sha256": matrix["source_sha256"],
            "profile": matrix["profile"],
            "task": cell["task"],
            "world_model_seed": int(cell["world_model_seed"]),
            "budget_track": cell["budget_track"],
            "run_id": cell["run_id"],
            "arm_id": cell["arm_id"],
        }.items()
    ):
        raise ValueError("controls rollout identity mismatch")
    world_cell = _matrix_cell(matrix, cell["world_cell_id"])
    world_result = read_json(_cell_result_path(output_root, world_cell))
    validate_world_cell(
        world_result, world_cell, protocol, parent_protocol, matrix, output_root
    )
    run = _run_by_id(matrix, cell["run_id"])
    dataset_result, dataset, dataset_directory = _dataset_artifacts(
        output_root, matrix, cell["task"], int(cell["world_model_seed"])
    )
    config = make_control_config(
        parent_protocol,
        matrix,
        run,
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
    )
    raw_path = _cell_directory(output_root, cell) / "predictive_draws.npz"
    if not raw_path.is_file() or result.get("raw_predictive_draws_sha256") != file_sha256(raw_path):
        raise ValueError("controls rollout raw artifact digest mismatch")
    raw = _load_npz(raw_path)
    required = {
        "observation_samples",
        "reward_samples",
        "continuation_samples",
        "target_observations",
        "target_rewards",
        "target_continuations",
        "context_observations",
        "context_actions",
        "future_actions",
        "noise",
        "episode_ids",
        "anchors",
        "training_observation_std",
    }
    if set(raw) != required or not all(np.isfinite(raw[name]).all() for name in required):
        raise ValueError("controls rollout raw artifact is incomplete/non-finite")
    expected_windows = parent._rollout_windows(
        dataset,
        parent_protocol,
        _parent_profile(protocol, matrix["profile"]),
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        stochastic_dim=config.stochastic_dim,
    )
    for name, expected in expected_windows.items():
        if not np.array_equal(raw[name], expected):
            raise ValueError(f"controls rollout field {name} is not canonical")
    recomputed = parent.normalized_rollout_statistics(
        raw["observation_samples"],
        raw["reward_samples"],
        expected_windows,
        protocol["evaluation"]["rollout_horizons"],
        raw["continuation_samples"],
    )
    for key, value in recomputed.items():
        if canonical_bytes(result.get(key)) != canonical_bytes(value):
            raise ValueError("controls rollout saved metrics do not recompute")
    if (
        result.get("world_cell_id") != world_cell["cell_id"]
        or result.get("world_checkpoint_sha256") != world_result["checkpoint_sha256"]
        or result.get("dataset_sha256") != dataset_result["dataset_sha256"]
        or result.get("dataset_result_sha256") != file_sha256(dataset_directory / "result.json")
        or result.get("runtime_config_sha256") != object_sha256(asdict(config))
        or canonical_bytes(result.get("runtime_config")) != canonical_bytes(asdict(config))
        or result.get("structural_nfe_per_transition") != run["primary_nfe"]
        or result.get("windows") != len(expected_windows["episode_ids"])
        or result.get("predictive_draws_per_window") != expected_windows["noise"].shape[0]
    ):
        raise ValueError("controls rollout dependency/config/count mismatch")
    if (
        parent.runtime_homogeneity_identity(result["runtime"])
        != parent.runtime_homogeneity_identity(world_result["runtime"])
        or not math.isfinite(float(result["wall_seconds"]))
        or float(result["wall_seconds"]) < 0.0
    ):
        raise ValueError("controls rollout runtime/wall-clock evidence is invalid")
    replay_key = (
        str(Path(output_root).resolve()),
        str(result["raw_predictive_draws_sha256"]),
        str(result["world_checkpoint_sha256"]),
        str(result["runtime_config_sha256"]),
    )
    if replay_key not in _VALIDATED_ROLLOUT_REPLAYS:
        replayed = _replay_control_rollout(raw, world_cell, config, output_root)
        parent._validate_rollout_replay_samples(raw, replayed)
        _VALIDATED_ROLLOUT_REPLAYS.add(replay_key)


def run_actor_cell(
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    if cell["stage"] != "actor":
        raise ValueError("run_actor_cell requires an actor cell")
    result_path = _cell_result_path(output_root, cell)
    if result_path.is_file():
        result = read_json(result_path)
        validate_actor_cell(result, cell, protocol, parent_protocol, matrix, output_root)
        return result
    import jax
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        create_agent,
        diverse_imagination_starts,
        jit_observe_sequence,
        jit_train_actor_critic,
        load_checkpoint,
        save_checkpoint,
    )

    world_cell = _matrix_cell(matrix, cell["world_cell_id"])
    world_result = read_json(_cell_result_path(output_root, world_cell))
    validate_world_cell(
        world_result, world_cell, protocol, parent_protocol, matrix, output_root
    )
    run = _run_by_id(matrix, cell["run_id"])
    dataset_result, arrays, _ = _dataset_artifacts(
        output_root, matrix, cell["task"], int(cell["world_model_seed"])
    )
    compute_cell = _matrix_cell(matrix, cell["compute_cell_id"])
    compute = read_json(_cell_result_path(output_root, compute_cell))
    config = make_control_config(
        parent_protocol,
        matrix,
        run,
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
    )
    world_state, stored_config, world_metadata = load_checkpoint(
        _cell_directory(output_root, world_cell) / "checkpoint.pkl"
    )
    if stored_config != config or world_metadata.get("cell_id") != world_cell["cell_id"]:
        raise ValueError("controls actor world checkpoint identity mismatch")
    fresh = create_agent(
        config,
        parent.derive_jax_key(
            "actor-init", cell["task"], cell["world_model_seed"], cell["actor_seed"]
        ),
    )
    source_world = world_state.params.world_model
    state = AgentState(
        AgentParams(source_world, fresh.params.actor, fresh.params.critic),
        world_state.model_optimizer,
        fresh.actor_optimizer,
        fresh.critic_optimizer,
        fresh.slow_critic,
    )
    allocation = _allocation(compute, cell, "actor")
    updates = int(allocation["updates"])
    execution = _execution_row(protocol, matrix)
    batch_size = int(execution["batch_size"])
    sequence_length = int(execution["sequence_length"])
    actor_schedule_seed = parent.derive_seed(
        "actor-batches", cell["world_model_seed"], cell["actor_seed"]
    )
    schedule = parent._batch_schedule(
        arrays,
        task=cell["task"],
        world_model_seed=actor_schedule_seed,
        updates=updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    directory = _cell_directory(output_root, cell)
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = _write_npz(directory / "batch_schedule.npz", schedule)
    checkpoint_path = directory / "checkpoint.pkl"
    start_update = 0
    accumulated_wall = 0.0
    latest: dict[str, float] | None = None
    if checkpoint_path.is_file():
        state, resume_config, metadata = load_checkpoint(checkpoint_path)
        if resume_config != config or metadata.get("cell_id") != cell["cell_id"]:
            raise ValueError("partial controls actor checkpoint identity mismatch")
        start_update = int(metadata.get("completed_updates", -1))
        accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        latest = metadata.get("last_metrics")
    if not 0 <= start_update <= updates:
        raise ValueError("partial controls actor update count is invalid")
    posterior_key = parent.derive_jax_key(
        "actor-posterior", cell["task"], cell["world_model_seed"], cell["actor_seed"]
    )
    start_key = parent.derive_jax_key(
        "actor-start", cell["task"], cell["world_model_seed"], cell["actor_seed"]
    )
    objective_key = parent.derive_jax_key(
        "actor-objective", cell["task"], cell["world_model_seed"], cell["actor_seed"]
    )
    started = time.perf_counter()
    checkpoint_every = int(execution["checkpoint_every_updates"])
    for update in range(start_update, updates):
        batch = parent._materialize_batch(
            arrays,
            schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        sequence = jit_observe_sequence(
            state.params.world_model,
            batch["observations"],
            batch["actions"],
            jax.random.fold_in(posterior_key, update),
            config,
            is_first=batch["is_first"],
        )
        starts = diverse_imagination_starts(
            sequence.states,
            config.burn_in,
            jax.random.fold_in(start_key, update),
        )
        state, actor_metrics = jit_train_actor_critic(
            state,
            starts,
            jax.random.fold_in(objective_key, update),
            config,
        )
        latest = _metrics(actor_metrics)
        completed = update + 1
        if completed % checkpoint_every == 0 or completed == updates:
            save_checkpoint(
                checkpoint_path,
                state,
                config,
                metadata={
                    "stage": "controls_actor",
                    "cell_id": cell["cell_id"],
                    "run_id": cell["run_id"],
                    "world_checkpoint_sha256": world_result["checkpoint_sha256"],
                    "completed_updates": completed,
                    "last_metrics": latest,
                    "wall_seconds": accumulated_wall + time.perf_counter() - started,
                },
            )
    if latest is None:
        raise RuntimeError("controls actor cell completed no update")
    world_delta = parent._tree_max_abs_difference(state.params.world_model, source_world)
    if world_delta != 0.0:
        raise RuntimeError("controls actor training changed the frozen world model")
    parent_profile = _parent_profile(protocol, matrix["profile"])
    episodes = int(
        parent_protocol["profiles"][parent_profile]["real_environment_evaluation_episodes"]
    )
    maximum_steps = int(parent_protocol["data"]["native_episode_limit"])
    episode_returns, traces = parent._evaluate_actor_policy(
        state,
        config,
        task=cell["task"],
        world_model_seed=int(cell["world_model_seed"]),
        actor_seed=int(cell["actor_seed"]),
        episodes=episodes,
        maximum_steps=maximum_steps,
    )
    trace_path = _write_npz(directory / "action_traces.npz", traces)
    save_checkpoint(
        checkpoint_path,
        state,
        config,
        metadata={
            "stage": "controls_actor",
            "cell_id": cell["cell_id"],
            "run_id": cell["run_id"],
            "world_checkpoint_sha256": world_result["checkpoint_sha256"],
            "completed_updates": updates,
            "last_metrics": latest,
            "wall_seconds": accumulated_wall + time.perf_counter() - started,
        },
    )
    normalized = [float(value) / 1000.0 for value in episode_returns]
    result = {
        "schema_version": ACTOR_SCHEMA,
        "status": "complete",
        "cell_id": cell["cell_id"],
        "matrix_sha256": matrix["matrix_sha256"],
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "source_sha256": matrix["source_sha256"],
        "profile": matrix["profile"],
        "task": cell["task"],
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
        "budget_track": cell["budget_track"],
        "run_id": cell["run_id"],
        "arm_id": cell["arm_id"],
        "world_cell_id": world_cell["cell_id"],
        "world_checkpoint_sha256": world_result["checkpoint_sha256"],
        "updates": updates,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "structural_nfe_per_imagined_transition": int(run["primary_nfe"]),
        "imagined_transitions_per_update": 2 * batch_size * config.imagination_horizon,
        "prior_field_evaluations_per_update": (
            2 * batch_size * config.imagination_horizon * int(run["primary_nfe"])
        ),
        "compiler_flops_per_update": float(allocation["flops_per_update"]),
        "realized_compiler_flops": float(allocation["flops_per_update"]) * updates,
        "world_model_frozen": True,
        "world_model_parameter_delta": world_delta,
        "initial_actor_parameter_sha256": _tree_digest(fresh.params.actor),
        "initial_critic_parameter_sha256": _tree_digest(fresh.params.critic),
        "final_metrics": latest,
        "episode_returns": [float(value) for value in episode_returns],
        "normalized_episode_returns": normalized,
        "normalized_episode_return_mean": float(np.mean(normalized)),
        "evaluation_seed_derivation": "sha256(actor-evaluation,task,world_model_seed,actor_seed,episode)",
        "batch_schedule_sha256": parent.array_sha256(schedule),
        "batch_schedule_file_sha256": file_sha256(schedule_path),
        "raw_action_traces_sha256": file_sha256(trace_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "runtime": parent.runtime_fingerprint(),
        "wall_seconds": accumulated_wall + time.perf_counter() - started,
    }
    validate_actor_cell(result, cell, protocol, parent_protocol, matrix, output_root)
    write_json_atomic(result_path, result)
    return result


def validate_actor_cell(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> None:
    required_result_keys = {
        "schema_version",
        "status",
        "cell_id",
        "matrix_sha256",
        "controls_protocol_sha256",
        "source_sha256",
        "profile",
        "task",
        "world_model_seed",
        "actor_seed",
        "budget_track",
        "run_id",
        "arm_id",
        "world_cell_id",
        "world_checkpoint_sha256",
        "updates",
        "batch_size",
        "sequence_length",
        "structural_nfe_per_imagined_transition",
        "imagined_transitions_per_update",
        "prior_field_evaluations_per_update",
        "compiler_flops_per_update",
        "realized_compiler_flops",
        "world_model_frozen",
        "world_model_parameter_delta",
        "initial_actor_parameter_sha256",
        "initial_critic_parameter_sha256",
        "final_metrics",
        "episode_returns",
        "normalized_episode_returns",
        "normalized_episode_return_mean",
        "evaluation_seed_derivation",
        "batch_schedule_sha256",
        "batch_schedule_file_sha256",
        "raw_action_traces_sha256",
        "checkpoint_sha256",
        "runtime",
        "wall_seconds",
    }
    _assert_exact_keys(result, required_result_keys, "controls actor result")
    if result.get("schema_version") != ACTOR_SCHEMA or result.get("status") != "complete":
        raise ValueError("controls actor schema/status mismatch")
    identity = {
        "cell_id": cell["cell_id"],
        "matrix_sha256": matrix["matrix_sha256"],
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "source_sha256": matrix["source_sha256"],
        "profile": matrix["profile"],
        "task": cell["task"],
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
        "budget_track": cell["budget_track"],
        "run_id": cell["run_id"],
        "arm_id": cell["arm_id"],
    }
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("controls actor identity mismatch")
    world_cell = _matrix_cell(matrix, cell["world_cell_id"])
    world_result = read_json(_cell_result_path(output_root, world_cell))
    validate_world_cell(
        world_result, world_cell, protocol, parent_protocol, matrix, output_root
    )
    run = _run_by_id(matrix, cell["run_id"])
    dataset_result, arrays, _ = _dataset_artifacts(
        output_root, matrix, cell["task"], int(cell["world_model_seed"])
    )
    compute_cell = _matrix_cell(matrix, cell["compute_cell_id"])
    compute = read_json(_cell_result_path(output_root, compute_cell))
    validate_controls_compute_plan(
        compute, protocol, parent_protocol, matrix, compute_cell, output_root
    )
    config = make_control_config(
        parent_protocol,
        matrix,
        run,
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
    )
    allocation = _allocation(compute, cell, "actor")
    updates = int(allocation["updates"])
    execution = _execution_row(protocol, matrix)
    actor_schedule_seed = parent.derive_seed(
        "actor-batches", cell["world_model_seed"], cell["actor_seed"]
    )
    expected_schedule = parent._batch_schedule(
        arrays,
        task=cell["task"],
        world_model_seed=actor_schedule_seed,
        updates=updates,
        batch_size=int(execution["batch_size"]),
        sequence_length=int(execution["sequence_length"]),
    )
    directory = _cell_directory(output_root, cell)
    schedule = _load_npz(directory / "batch_schedule.npz")
    if set(schedule) != set(expected_schedule) or any(
        not np.array_equal(schedule[key], expected_schedule[key]) for key in schedule
    ):
        raise ValueError("controls actor minibatch schedule is not canonical")
    traces = _load_npz(directory / "action_traces.npz")
    required_traces = {
        "actions",
        "rewards",
        "continuations",
        "is_last",
        "lengths",
        "evaluation_seeds",
    }
    if set(traces) != required_traces:
        raise ValueError("controls actor traces are incomplete")
    from imf_dreamer_jax import create_agent, load_checkpoint

    state, stored_config, metadata = load_checkpoint(directory / "checkpoint.pkl")
    world_state, world_config, world_metadata = load_checkpoint(
        _cell_directory(output_root, world_cell) / "checkpoint.pkl"
    )
    fresh = create_agent(
        config,
        parent.derive_jax_key(
            "actor-init", cell["task"], cell["world_model_seed"], cell["actor_seed"]
        ),
    )
    if (
        stored_config != config
        or world_config != config
        or world_metadata.get("cell_id") != world_cell["cell_id"]
        or metadata.get("stage") != "controls_actor"
        or metadata.get("cell_id") != cell["cell_id"]
        or metadata.get("run_id") != cell["run_id"]
        or metadata.get("completed_updates") != updates
        or metadata.get("world_checkpoint_sha256") != world_result["checkpoint_sha256"]
        or int(np.asarray(state.actor_optimizer.step)) != updates
        or int(np.asarray(state.critic_optimizer.step)) != updates
        or parent._tree_max_abs_difference(
            state.params.world_model, world_state.params.world_model
        )
        != 0.0
        or result.get("world_cell_id") != world_cell["cell_id"]
        or result.get("world_checkpoint_sha256") != world_result["checkpoint_sha256"]
        or result.get("updates") != updates
        or result.get("batch_size") != int(execution["batch_size"])
        or result.get("sequence_length") != int(execution["sequence_length"])
        or result.get("structural_nfe_per_imagined_transition")
        != int(run["primary_nfe"])
        or result.get("imagined_transitions_per_update")
        != 2 * int(execution["batch_size"]) * int(config.imagination_horizon)
        or result.get("prior_field_evaluations_per_update")
        != 2
        * int(execution["batch_size"])
        * int(config.imagination_horizon)
        * int(run["primary_nfe"])
        or result.get("compiler_flops_per_update") != float(allocation["flops_per_update"])
        or result.get("realized_compiler_flops")
        != float(allocation["flops_per_update"]) * updates
        or result.get("world_model_frozen") is not True
        or result.get("world_model_parameter_delta") != 0.0
        or result.get("initial_actor_parameter_sha256") != _tree_digest(fresh.params.actor)
        or result.get("initial_critic_parameter_sha256") != _tree_digest(fresh.params.critic)
        or result.get("batch_schedule_sha256") != parent.array_sha256(schedule)
        or result.get("batch_schedule_file_sha256") != file_sha256(directory / "batch_schedule.npz")
        or result.get("raw_action_traces_sha256") != file_sha256(directory / "action_traces.npz")
        or result.get("checkpoint_sha256") != file_sha256(directory / "checkpoint.pkl")
    ):
        raise ValueError("controls actor checkpoint/accounting mismatch")
    episodes = int(
        parent_protocol["profiles"][_parent_profile(protocol, matrix["profile"])][
            "real_environment_evaluation_episodes"
        ]
    )
    returns = np.asarray(result.get("episode_returns"), dtype=np.float64)
    if returns.shape != (episodes,) or not np.isfinite(returns).all():
        raise ValueError("controls actor return count/value is invalid")
    rewards = np.asarray(traces["rewards"], dtype=np.float64)
    lengths = np.asarray(traces["lengths"], dtype=np.int64)
    actions = np.asarray(traces["actions"])
    continuations = np.asarray(traces["continuations"])
    terminals = np.asarray(traces["is_last"])
    steps = np.arange(actions.shape[1])[None, :]
    valid = steps < lengths[:, None]
    if (
        actions.shape[0] != episodes
        or rewards.shape != actions.shape[:2]
        or continuations.shape != actions.shape[:2]
        or terminals.shape != actions.shape[:2]
        or traces["evaluation_seeds"].shape != (episodes,)
        or terminals.dtype != np.bool_
        or lengths.shape != (episodes,)
        or np.any(lengths <= 0)
        or np.any(lengths > actions.shape[1])
        or not np.isfinite(actions).all()
        or not np.isfinite(rewards).all()
        or not np.isfinite(continuations).all()
        or np.any(actions[valid] < -1.0)
        or np.any(actions[valid] > 1.0)
        or np.any(continuations[valid] < 0.0)
        or np.any(continuations[valid] > 1.0)
        or np.any(rewards[~valid] != 0.0)
        or np.any(continuations[~valid] != 0.0)
        or np.any(terminals[~valid])
    ):
        raise ValueError("controls actor trace shapes/values are invalid")
    for episode, length in enumerate(lengths):
        if np.any(terminals[episode, :length][:-1]):
            raise ValueError("controls actor trace continued after termination")
    recomputed = np.asarray(
        [float(np.sum(rewards[index, :length])) for index, length in enumerate(lengths)],
        dtype=np.float64,
    )
    if not np.array_equal(recomputed, returns):
        raise ValueError("controls actor returns do not recompute from retained rewards")
    normalized = (returns / 1000.0).tolist()
    if (
        result.get("normalized_episode_returns") != normalized
        or result.get("normalized_episode_return_mean") != float(np.mean(normalized))
    ):
        raise ValueError("controls normalized actor returns do not recompute")
    expected_seeds = np.asarray(
        [
            parent.derive_seed(
                "actor-evaluation",
                cell["task"],
                cell["world_model_seed"],
                cell["actor_seed"],
                episode,
            )
            for episode in range(episodes)
        ],
        dtype=np.uint32,
    )
    if not np.array_equal(traces["evaluation_seeds"], expected_seeds):
        raise ValueError("controls actor evaluation seeds are not canonical")
    final_metrics = result.get("final_metrics")
    if (
        not isinstance(final_metrics, Mapping)
        or not final_metrics
        or not all(math.isfinite(float(value)) for value in final_metrics.values())
        or canonical_bytes(metadata.get("last_metrics")) != canonical_bytes(final_metrics)
        or result.get("evaluation_seed_derivation")
        != "sha256(actor-evaluation,task,world_model_seed,actor_seed,episode)"
        or parent.runtime_homogeneity_identity(result["runtime"])
        != parent.runtime_homogeneity_identity(compute["runtime"])
        or not math.isfinite(float(result["wall_seconds"]))
        or float(result["wall_seconds"]) < 0.0
    ):
        raise ValueError("controls actor metrics/runtime provenance is invalid")
    replay_key = (
        str(Path(output_root).resolve()),
        str(result["raw_action_traces_sha256"]),
        str(result["checkpoint_sha256"]),
        str(cell["cell_id"]),
    )
    if replay_key not in _VALIDATED_ACTOR_REPLAYS:
        parent._validate_actor_environment_replay(
            traces,
            returns,
            state=state,
            config=config,
            task=cell["task"],
            world_model_seed=int(cell["world_model_seed"]),
            actor_seed=int(cell["actor_seed"]),
            maximum_steps=int(parent_protocol["data"]["native_episode_limit"]),
            action_repeat=1,
        )
        _VALIDATED_ACTOR_REPLAYS.add(replay_key)


def run_stage_cells(
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
    stages: Sequence[str] = STAGE_ORDER,
) -> dict[str, int]:
    root = Path(output_root)
    if (root / "FINALIZED.json").exists():
        raise ValueError("a finalized controls root is immutable")
    requested = tuple(stages)
    if any(stage not in STAGE_ORDER for stage in requested):
        raise ValueError("unknown controls stage")
    counts = {stage: 0 for stage in requested}
    if "compute" in requested:
        counts["compute"] = run_all_compute_cells(
            protocol, parent_protocol, matrix, output_root
        )
    for stage in ("world", "rollout", "actor"):
        if stage not in requested:
            continue
        for cell in (candidate for candidate in matrix["cells"] if candidate["stage"] == stage):
            if stage == "world":
                run_world_cell(cell, protocol, parent_protocol, matrix, output_root)
            elif stage == "rollout":
                run_rollout_cell(cell, protocol, parent_protocol, matrix, output_root)
            else:
                run_actor_cell(cell, protocol, parent_protocol, matrix, output_root)
            counts[stage] += 1
    return counts


def run_control_cell(
    cell_id: str,
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Execute exactly one frozen controls cell, preserving normal validation/resume."""

    if (Path(output_root) / "FINALIZED.json").exists():
        raise ValueError("a finalized controls root is immutable")
    cell = _matrix_cell(matrix, cell_id)
    if cell["stage"] == "compute":
        selected = protocol["profiles"][matrix["profile"]]
        exemplar_seed = int(selected["world_model_seeds"][0])
        dataset, _, _ = _dataset_artifacts(
            output_root, matrix, cell["task"], exemplar_seed
        )
        return build_controls_compute_plan(
            protocol,
            parent_protocol,
            matrix,
            cell["task"],
            dataset["observation_shape"],
            int(dataset["action_dim"]),
            output_root,
        )
    if cell["stage"] == "world":
        return run_world_cell(
            cell, protocol, parent_protocol, matrix, output_root
        )
    if cell["stage"] == "rollout":
        return run_rollout_cell(
            cell, protocol, parent_protocol, matrix, output_root
        )
    if cell["stage"] == "actor":
        return run_actor_cell(
            cell, protocol, parent_protocol, matrix, output_root
        )
    raise ValueError("controls matrix contains an unknown stage")


def _actor_nested_mean(
    matrix: Mapping[str, Any], output_root: str | Path, run_id: str, track: str, task: str, seed: int
) -> float:
    cells = [
        cell
        for cell in matrix["cells"]
        if cell["stage"] == "actor"
        and cell["run_id"] == run_id
        and cell["budget_track"] == track
        and cell["task"] == task
        and int(cell["world_model_seed"]) == int(seed)
    ]
    expected = len(
        {
            int(cell["actor_seed"])
            for cell in matrix["cells"]
            if cell["stage"] == "actor"
            and cell["task"] == task
            and int(cell["world_model_seed"]) == int(seed)
        }
    )
    if len(cells) != expected or expected <= 0:
        raise ValueError("controls nested actor cells are incomplete")
    values = [
        float(read_json(_cell_result_path(output_root, cell))["normalized_episode_return_mean"])
        for cell in cells
    ]
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def collect_control_units(
    matrix: Mapping[str, Any], output_root: str | Path
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for track in matrix["run_budget_tracks"]:
        pairs = sorted(
            {
                (cell["task"], int(cell["world_model_seed"]))
                for cell in matrix["cells"]
                if cell["stage"] == "world" and cell["budget_track"] == track
            }
        )
        for task, seed in pairs:
            values: dict[str, Any] = {}
            for run in matrix["arm_runs"]:
                rollout_cells = [
                    cell
                    for cell in matrix["cells"]
                    if cell["stage"] == "rollout"
                    and cell["run_id"] == run["run_id"]
                    and cell["budget_track"] == track
                    and cell["task"] == task
                    and int(cell["world_model_seed"]) == seed
                ]
                if len(rollout_cells) != 1:
                    raise ValueError("controls rollout unit is incomplete")
                rollout = read_json(_cell_result_path(output_root, rollout_cells[0]))
                values[run["run_id"]] = {
                    "rollout_auc": float(
                        rollout["normalized_free_running_rollout_error_auc"]
                    ),
                    "actor_return": _actor_nested_mean(
                        matrix, output_root, run["run_id"], track, task, seed
                    ),
                }
            units.append(
                {
                    "task": task,
                    "world_model_seed": seed,
                    "budget_track": track,
                    "runs": values,
                }
            )
    return units


def _descriptive_bootstrap(
    units: Sequence[Mapping[str, Any]],
    run_id: str,
    comparator: str,
    metric: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    tasks = sorted({unit["task"] for unit in units})
    grouped = {task: [unit for unit in units if unit["task"] == task] for task in tasks}
    random = np.random.default_rng(seed)

    def contrast(rows: Sequence[Mapping[str, Any]]) -> float:
        if metric == "rollout_auc":
            values = [
                float(row["runs"][comparator][metric])
                - float(row["runs"][run_id][metric])
                for row in rows
            ]
        else:
            values = [
                float(row["runs"][run_id][metric])
                - float(row["runs"][comparator][metric])
                for row in rows
            ]
        return parent.interquartile_mean(values)

    estimate = contrast(units)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled: list[Mapping[str, Any]] = []
        for task in tasks:
            rows = grouped[task]
            indices = random.integers(0, len(rows), size=len(rows))
            sampled.extend(rows[int(value)] for value in indices)
        draws[index] = contrast(sampled)
    return {
        "estimand": (
            "comparator_minus_run_rollout_auc"
            if metric == "rollout_auc"
            else "run_minus_comparator_actor_return"
        ),
        "favorable_to_run_direction": "positive",
        "estimate": estimate,
        "interval_lower": float(np.quantile(draws, 0.025)),
        "interval_upper": float(np.quantile(draws, 0.975)),
        "interval_percentiles": [2.5, 97.5],
        "resamples": resamples,
        "seed": seed,
        "decision_role": "descriptive_only_no_superiority_decision",
    }


def _rank_candidates(
    units: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], family: str
) -> list[dict[str, Any]]:
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    run_ids = [f"{family}@@{candidate_id}" for candidate_id in candidate_ids]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for unit in units:
        grouped.setdefault(str(unit["task"]), []).append(unit)
    scores = {candidate_id: [] for candidate_id in candidate_ids}
    for rows in grouped.values():
        per_task = {candidate_id: [] for candidate_id in candidate_ids}
        for row in rows:
            rollout_order = sorted(
                range(len(run_ids)),
                key=lambda index: (
                    float(row["runs"][run_ids[index]]["rollout_auc"]),
                    candidate_ids[index],
                ),
            )
            actor_order = sorted(
                range(len(run_ids)),
                key=lambda index: (
                    -float(row["runs"][run_ids[index]]["actor_return"]),
                    candidate_ids[index],
                ),
            )
            rollout_rank = {index: rank + 1 for rank, index in enumerate(rollout_order)}
            actor_rank = {index: rank + 1 for rank, index in enumerate(actor_order)}
            for index, candidate_id in enumerate(candidate_ids):
                per_task[candidate_id].append(
                    0.5 * (rollout_rank[index] + actor_rank[index])
                )
        for candidate_id in candidate_ids:
            scores[candidate_id].append(float(np.mean(per_task[candidate_id])))
    return [
        {
            "candidate_id": candidate_id,
            "selection_score": float(np.mean(scores[candidate_id])),
            "task_scores": scores[candidate_id],
        }
        for candidate_id in candidate_ids
    ]


def build_controls_selection(
    analysis: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if analysis.get("profile") != "development":
        raise ValueError("controls selection can be built only from development analysis")
    units = [
        unit for unit in analysis["units"] if unit["budget_track"] == "equal_updates"
    ]
    selected: dict[str, Any] = {}
    ranks: dict[str, Any] = {}
    for family in protocol["development_selection"]["selected_families"]:
        candidates = _candidate_grid(protocol, family)
        rows = _rank_candidates(units, candidates, family)
        winner = min(
            rows, key=lambda row: (float(row["selection_score"]), row["candidate_id"])
        )
        selected[family] = _candidate_by_id(protocol, family, winner["candidate_id"])
        ranks[family] = rows
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "status": "complete",
        "controls_protocol_sha256": controls_protocol_digest(protocol),
        "development_analysis_sha256": analysis["analysis_sha256"],
        "development_matrix_sha256": analysis["matrix_sha256"],
        "development_source_sha256": analysis["source_sha256"],
        "selection_profile": "development",
        "selection_budget_track": "equal_updates",
        "confirmatory_outcomes_accessed": False,
        "selected": selected,
        "candidate_rank_evidence": ranks,
    }
    selection["selection_sha256"] = object_sha256(selection)
    return selection


def build_controls_analysis(
    protocol: Mapping[str, Any], matrix: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    units = collect_control_units(matrix, output_root)
    resamples = int(protocol["profiles"][matrix["profile"]]["analysis_bootstrap_resamples"])
    summaries: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for track in matrix["run_budget_tracks"]:
        track_units = [unit for unit in units if unit["budget_track"] == track]
        summaries[track] = {}
        contrasts[track] = {}
        for run in matrix["arm_runs"]:
            run_id = run["run_id"]
            rollout_values = [unit["runs"][run_id]["rollout_auc"] for unit in track_units]
            actor_values = [unit["runs"][run_id]["actor_return"] for unit in track_units]
            summaries[track][run_id] = {
                "arm_id": run["arm_id"],
                "family": run["family"],
                "candidate_id": run["candidate"]["candidate_id"],
                "unit_count": len(track_units),
                "rollout_auc_iqm": parent.interquartile_mean(rollout_values),
                "actor_return_iqm": parent.interquartile_mean(actor_values),
            }
            contrasts[track][run_id] = {}
            for comparator_index, comparator in enumerate(
                ("shortcut_forcing", "trajectory_imf")
            ):
                if comparator not in track_units[0]["runs"]:
                    raise ValueError("controls analysis comparator is absent")
                contrasts[track][run_id][comparator] = {
                    metric: _descriptive_bootstrap(
                        track_units,
                        run_id,
                        comparator,
                        metric,
                        resamples=resamples,
                        seed=int(protocol["statistics"]["bootstrap_seed"])
                        + comparator_index * 1000003
                        + int(object_sha256([track, run_id, metric])[:8], 16),
                    )
                    for metric in ("rollout_auc", "actor_return")
                }
    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "profile": matrix["profile"],
        "evidence_class": matrix["evidence_class"],
        "claim_eligible": False,
        "controls_protocol_sha256": matrix["controls_protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "inference_role": "descriptive_only",
        "primary_gate_accessed": False,
        "superiority_decision": None,
        "multiplicity_adjusted_decision": None,
        "independent_unit": "task_x_world_model_seed",
        "actor_seed_treatment": "nested_mean_within_task_x_world_model_seed",
        "units": units,
        "summaries": summaries,
        "descriptive_contrasts": contrasts,
    }
    analysis["analysis_sha256"] = object_sha256(analysis)
    return analysis


def validate_controls_analysis(
    analysis: Mapping[str, Any],
    protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> None:
    _assert_exact_keys(
        analysis,
        {
            "schema_version",
            "status",
            "profile",
            "evidence_class",
            "claim_eligible",
            "controls_protocol_sha256",
            "source_sha256",
            "matrix_sha256",
            "inference_role",
            "primary_gate_accessed",
            "superiority_decision",
            "multiplicity_adjusted_decision",
            "independent_unit",
            "actor_seed_treatment",
            "units",
            "summaries",
            "descriptive_contrasts",
            "analysis_sha256",
        },
        "controls analysis",
    )
    if (
        analysis.get("schema_version") != ANALYSIS_SCHEMA
        or analysis.get("status") != "complete"
        or analysis.get("profile") != matrix["profile"]
        or analysis.get("claim_eligible") is not False
        or analysis.get("inference_role") != "descriptive_only"
        or analysis.get("primary_gate_accessed") is not False
        or analysis.get("superiority_decision") is not None
        or analysis.get("multiplicity_adjusted_decision") is not None
        or analysis.get("analysis_sha256")
        != object_sha256(_without_digest(analysis, "analysis_sha256"))
    ):
        raise ValueError("controls analysis identity or claim isolation mismatch")
    expected = build_controls_analysis(protocol, matrix, output_root)
    if canonical_bytes(analysis) != canonical_bytes(expected):
        raise ValueError("controls analysis does not rederive from authenticated raw cells")


def render_controls_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# NeurIPS controls report",
        "",
        f"Profile: `{analysis['profile']}` (`{analysis['evidence_class']}`).",
        "",
        "These results are secondary and descriptive. They do not access, alter, or substitute for the frozen four-interval primary superiority gate.",
        "",
    ]
    summaries = analysis["summaries"]
    if not summaries or set(summaries) - set(BUDGET_TRACK_ORDER):
        raise ValueError("controls report summaries contain no tracks or an unknown budget track")
    for track in (value for value in BUDGET_TRACK_ORDER if value in summaries):
        rows = summaries[track]
        lines.extend(
            [
                f"## {track}",
                "",
                "| Run | Family | Rollout AUC IQM (lower) | Actor return IQM (higher) |",
                "|---|---|---:|---:|",
            ]
        )
        # analysis.json is written canonically with sorted object keys.  Render in
        # an explicit order so the in-memory report and its JSON-rederived report
        # are byte-identical regardless of mapping insertion order.
        for run_id in sorted(rows):
            row = rows[run_id]
            lines.append(
                f"| `{run_id}` | `{row['family']}` | {row['rollout_auc_iqm']:.8g} | {row['actor_return_iqm']:.8g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The 95% intervals retained in `analysis.json` are unadjusted descriptive intervals across registered task-by-world-seed units. No interval in this controls family produces a superiority pass/fail decision.",
            "",
            "`temporal_increment_imf` is a simple one-step-versus-two-half-step consistency control inspired by temporal-increment ideas; it is not a reproduction of MeLISA.",
            "",
        ]
    )
    return "\n".join(lines)


def _safe_relative_path(root: Path, path: Path) -> str:
    if path.is_symlink():
        raise ValueError("controls artifacts may not be symlinks")
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
        raise ValueError("controls artifact path is not canonical and relative")
    return relative


def _retained_artifact_paths(root: Path) -> list[Path]:
    excluded = {"artifact_manifest.json", "FINALIZED.json"}
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("controls output contains a symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in excluded:
            if path.name.endswith(".tmp"):
                raise ValueError("controls output contains an unfinished temporary artifact")
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def build_controls_artifact_manifest(
    root: str | Path, matrix: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    base = Path(root).resolve()
    files = [
        {
            "path": _safe_relative_path(base, path),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in _retained_artifact_paths(base)
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "profile": matrix["profile"],
        "claim_eligible": False,
        "controls_protocol_sha256": matrix["controls_protocol_sha256"],
        "source_sha256": matrix["source_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "file_count": len(files),
        "files": files,
    }
    manifest["artifact_manifest_sha256"] = object_sha256(manifest)
    return manifest


def validate_controls_artifact_manifest(
    manifest: Mapping[str, Any],
    root: str | Path,
    matrix: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    base = Path(root).resolve()
    _assert_exact_keys(
        manifest,
        {
            "schema_version",
            "status",
            "profile",
            "claim_eligible",
            "controls_protocol_sha256",
            "source_sha256",
            "matrix_sha256",
            "analysis_sha256",
            "file_count",
            "files",
            "artifact_manifest_sha256",
        },
        "controls artifact manifest",
    )
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("profile") != matrix["profile"]
        or manifest.get("claim_eligible") is not False
        or manifest.get("controls_protocol_sha256")
        != matrix["controls_protocol_sha256"]
        or manifest.get("source_sha256") != matrix["source_sha256"]
        or manifest.get("matrix_sha256") != matrix["matrix_sha256"]
        or manifest.get("analysis_sha256") != analysis["analysis_sha256"]
        or manifest.get("artifact_manifest_sha256")
        != object_sha256(_without_digest(manifest, "artifact_manifest_sha256"))
    ):
        raise ValueError("controls artifact manifest identity/digest mismatch")
    expected_paths = {
        _safe_relative_path(base, path): path for path in _retained_artifact_paths(base)
    }
    entries = manifest.get("files")
    if not isinstance(entries, list) or manifest.get("file_count") != len(entries):
        raise ValueError("controls artifact manifest file count is invalid")
    listed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("controls artifact manifest entry schema is invalid")
        pure = PurePosixPath(str(entry["path"]))
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != entry["path"]:
            raise ValueError("controls artifact path is not canonical and relative")
        if entry["path"] in listed:
            raise ValueError("controls artifact manifest contains duplicate paths")
        listed[entry["path"]] = entry
    if set(listed) != set(expected_paths):
        raise ValueError("controls artifact manifest does not match the exact retained file set")
    for relative, path in expected_paths.items():
        entry = listed[relative]
        if entry["size"] != path.stat().st_size or entry["sha256"] != file_sha256(path):
            raise ValueError("controls artifact file digest/size mismatch")


def _validate_environment(root: Path, matrix: Mapping[str, Any]) -> None:
    dependency_path = root / "dependency_lock.txt"
    environment_path = root / "environment.json"
    identity_path = root / "source_identity.txt"
    if not dependency_path.is_file() or not dependency_path.read_text(encoding="utf-8").strip():
        raise ValueError("controls dependency lock is absent or empty")
    if not environment_path.is_file() or not identity_path.is_file():
        raise ValueError("controls frozen runtime provenance is incomplete")
    environment = read_json(environment_path)
    if "jax" not in environment or environment["jax"].get("jax_enable_x64") is not False:
        raise ValueError("controls frozen JAX runtime is incomplete or x64-enabled")
    runtimes = [
        read_json(_cell_result_path(root, cell))["runtime"]
        for cell in matrix["cells"]
        if cell["stage"] == "compute"
    ]
    if not runtimes:
        raise ValueError("controls runtime provenance has no compute cell")
    identity = parent.runtime_homogeneity_identity(runtimes[0])
    if any(parent.runtime_homogeneity_identity(value) != identity for value in runtimes[1:]):
        raise ValueError("controls compute workers are heterogeneous")
    frozen_identity = {
        "python": environment["python"],
        "jax_version": environment["jax"]["version"],
        "jaxlib_version": environment["jax"]["jaxlib_version"],
        "backend": environment["jax"]["backend"],
        "device_platforms": environment["jax"]["device_platforms"],
        "device_kinds": environment["jax"]["device_kinds"],
        "visible_device_count": environment["jax"]["visible_device_count"],
        "xla_platform_version": environment["jax"]["xla_platform_version"],
        "jax_enable_x64": environment["jax"]["jax_enable_x64"],
    }
    if frozen_identity != identity:
        raise ValueError("controls freeze runtime differs from compute runtime")
    source = read_json(root / "source_manifest.json")
    git = source["git"]
    expected_identity_text = (
        f"commit={git['commit']}\n"
        f"dirty_patch_sha256={git['dirty_patch_sha256']}\n"
        f"source_sha256={source['source_sha256']}\n"
    )
    if identity_path.read_text(encoding="utf-8") != expected_identity_text:
        raise ValueError("controls plain-text source identity differs from manifest")
    if matrix["profile"] in ("development", "confirmatory"):
        if identity["visible_device_count"] != 1:
            raise ValueError("development/confirmatory controls did not use one visible device")
    if matrix["profile"] == "confirmatory":
        if environment.get("dependency_capture_status") != "complete":
            raise ValueError("confirmatory controls dependency capture is incomplete")


def _validate_frozen_selection_files(root: Path, matrix: Mapping[str, Any]) -> None:
    """Bind retained selection copies to the selections embedded in the matrix."""

    parent_path = root / "parent_selection.json"
    controls_path = root / "controls_selection.json"
    profile = matrix["profile"]
    if profile == "confirmatory":
        for path, key in (
            (parent_path, "parent_selection"),
            (controls_path, "controls_selection"),
        ):
            if not path.is_file() or canonical_bytes(read_json(path)) != canonical_bytes(
                matrix[key]
            ):
                raise ValueError(
                    f"retained {path.name} differs from the selection embedded in the matrix"
                )
    else:
        if parent_path.exists():
            raise ValueError("nonconfirmatory controls retain an unexpected parent selection")
        if profile == "smoke" and controls_path.exists():
            raise ValueError("smoke controls retain an unexpected controls selection")


def validate_all_control_cells(
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, int]:
    counts = {stage: 0 for stage in STAGE_ORDER}
    for cell in matrix["cells"]:
        path = _cell_result_path(output_root, cell)
        if not path.is_file():
            raise FileNotFoundError(f"controls cell is incomplete: {cell['cell_id']}")
        result = read_json(path)
        if cell["stage"] == "compute":
            validate_controls_compute_plan(
                result, protocol, parent_protocol, matrix, cell, output_root
            )
        elif cell["stage"] == "world":
            validate_world_cell(
                result, cell, protocol, parent_protocol, matrix, output_root
            )
        elif cell["stage"] == "rollout":
            validate_rollout_cell(
                result, cell, protocol, parent_protocol, matrix, output_root
            )
        elif cell["stage"] == "actor":
            validate_actor_cell(
                result, cell, protocol, parent_protocol, matrix, output_root
            )
        else:
            raise ValueError("controls matrix contains an unknown stage")
        counts[cell["stage"]] += 1
    return counts


def finalize_controls_run(
    output_root: str | Path, *, workspace: str | Path | None = None
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if (root / "FINALIZED.json").exists():
        return verify_controls_output(root, workspace=workspace)
    protocol, parent_protocol, source, matrix = load_frozen_controls(root)
    validate_controls_protocol(protocol, parent_protocol)
    validate_controls_source_manifest(source, workspace)
    validate_controls_matrix(matrix, protocol, parent_protocol, source)
    cell_counts = validate_all_control_cells(
        protocol, parent_protocol, matrix, root
    )
    analysis = build_controls_analysis(protocol, matrix, root)
    write_json_atomic(root / "analysis.json", analysis)
    if matrix["profile"] == "development":
        selection = build_controls_selection(analysis, protocol)
        write_json_atomic(root / "controls_selection.json", selection)
    report = render_controls_report(analysis)
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = build_controls_artifact_manifest(root, matrix, analysis)
    write_json_atomic(root / "artifact_manifest.json", manifest)
    # Authenticate every pre-final artifact before publishing the immutable
    # marker.  A failed validation must leave the root resumable rather than
    # strand it behind a marker that the runner will never overwrite.
    summary = verify_controls_output(
        root, workspace=workspace, _require_finalized_marker=False
    )
    marker = {
        "schema_version": "trajectory-imf-neurips-controls-final-v1",
        "status": "finalized_immutable_to_runner",
        "profile": matrix["profile"],
        "claim_eligible": False,
        "matrix_sha256": matrix["matrix_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "analysis_file_sha256": file_sha256(root / "analysis.json"),
        "report_file_sha256": file_sha256(root / "REPORT.md"),
        "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
        "artifact_manifest_file_sha256": file_sha256(root / "artifact_manifest.json"),
        "cell_counts": cell_counts,
    }
    marker["finalized_sha256"] = object_sha256(marker)
    write_json_atomic(root / "FINALIZED.json", marker)
    # FINALIZED.json is the last write.  Independent callers verify the marker
    # through the public default path; no fallible post-marker action can leave
    # an internally rejected root falsely marked final.
    return summary


def verify_controls_output(
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
    _require_finalized_marker: bool = True,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    protocol, parent_protocol, source, matrix = load_frozen_controls(root)
    validate_controls_protocol(protocol, parent_protocol)
    validate_controls_source_manifest(source, workspace)
    validate_controls_matrix(matrix, protocol, parent_protocol, source)
    freeze = read_json(root / "freeze.json")
    _assert_exact_keys(
        freeze,
        {
            "schema_version",
            "status",
            "profile",
            "controls_protocol_sha256",
            "parent_protocol_sha256",
            "source_sha256",
            "matrix_sha256",
            "registered_arm_count",
            "arm_count",
            "run_count",
            "cell_count",
            "frozen_unix_time",
            "freeze_sha256",
        },
        "controls freeze",
    )
    if (
        freeze.get("schema_version") != "trajectory-imf-neurips-controls-freeze-v1"
        or freeze.get("status") != "frozen_before_control_outcomes"
        or freeze.get("profile") != matrix["profile"]
        or freeze.get("controls_protocol_sha256") != matrix["controls_protocol_sha256"]
        or freeze.get("parent_protocol_sha256") != matrix["parent_protocol_sha256"]
        or freeze.get("freeze_sha256")
        != object_sha256(_without_digest(freeze, "freeze_sha256"))
        or freeze.get("matrix_sha256") != matrix["matrix_sha256"]
        or freeze.get("source_sha256") != source["source_sha256"]
        or freeze.get("registered_arm_count") != 33
        or freeze.get("arm_count") != len(_profile_arm_specs(protocol, matrix["profile"]))
        or freeze.get("run_count") != len(matrix["arm_runs"])
        or freeze.get("cell_count") != len(matrix["cells"])
        or not math.isfinite(float(freeze.get("frozen_unix_time", math.nan)))
    ):
        raise ValueError("controls freeze identity/digest mismatch")
    if matrix["profile"] in ("development", "confirmatory"):
        _strict_source_ready(source)
    counts = validate_all_control_cells(protocol, parent_protocol, matrix, root)
    _validate_environment(root, matrix)
    _validate_frozen_selection_files(root, matrix)
    analysis = read_json(root / "analysis.json")
    validate_controls_analysis(analysis, protocol, matrix, root)
    if (root / "REPORT.md").read_text(encoding="utf-8") != render_controls_report(analysis):
        raise ValueError("controls human report does not rederive from analysis")
    if matrix["profile"] == "development":
        selection = read_json(root / "controls_selection.json")
        expected_selection = build_controls_selection(analysis, protocol)
        if canonical_bytes(selection) != canonical_bytes(expected_selection):
            raise ValueError("controls development selection does not rederive")
    manifest = read_json(root / "artifact_manifest.json")
    validate_controls_artifact_manifest(manifest, root, matrix, analysis)
    summary = {
        "status": "verified",
        "profile": matrix["profile"],
        "claim_eligible": False,
        "arm_count": len(matrix["arm_specs"]),
        "run_count": len(matrix["arm_runs"]),
        "cell_counts": counts,
        "matrix_sha256": matrix["matrix_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
        "primary_gate_accessed": False,
        "superiority_decision": None,
    }
    if not _require_finalized_marker:
        return summary
    marker = read_json(root / "FINALIZED.json")
    _assert_exact_keys(
        marker,
        {
            "schema_version",
            "status",
            "profile",
            "claim_eligible",
            "matrix_sha256",
            "analysis_sha256",
            "analysis_file_sha256",
            "report_file_sha256",
            "artifact_manifest_sha256",
            "artifact_manifest_file_sha256",
            "cell_counts",
            "finalized_sha256",
        },
        "controls finalization marker",
    )
    if (
        marker.get("schema_version") != "trajectory-imf-neurips-controls-final-v1"
        or marker.get("status") != "finalized_immutable_to_runner"
        or marker.get("profile") != matrix["profile"]
        or marker.get("claim_eligible") is not False
        or marker.get("matrix_sha256") != matrix["matrix_sha256"]
        or marker.get("analysis_sha256") != analysis["analysis_sha256"]
        or marker.get("analysis_file_sha256") != file_sha256(root / "analysis.json")
        or marker.get("report_file_sha256") != file_sha256(root / "REPORT.md")
        or marker.get("artifact_manifest_sha256")
        != manifest["artifact_manifest_sha256"]
        or marker.get("artifact_manifest_file_sha256")
        != file_sha256(root / "artifact_manifest.json")
        or marker.get("cell_counts") != counts
        or marker.get("finalized_sha256")
        != object_sha256(_without_digest(marker, "finalized_sha256"))
    ):
        raise ValueError("controls finalization marker is invalid")
    return summary


def run_controls_all(
    protocol: Mapping[str, Any],
    parent_protocol: Mapping[str, Any],
    profile: str,
    output_root: str | Path,
    *,
    workspace: str | Path | None = None,
    parent_selection: Mapping[str, Any] | None = None,
    controls_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if not (root / "matrix.json").is_file():
        freeze_controls_run(
            protocol,
            parent_protocol,
            profile,
            root,
            workspace=workspace,
            parent_selection=parent_selection,
            controls_selection=controls_selection,
        )
    frozen_protocol, frozen_parent, source, matrix = load_frozen_controls(root)
    if canonical_bytes(frozen_protocol) != canonical_bytes(protocol):
        raise ValueError("requested controls protocol differs from frozen root")
    if canonical_bytes(frozen_parent) != canonical_bytes(parent_protocol):
        raise ValueError("requested parent protocol differs from frozen root")
    if matrix["profile"] != profile:
        raise ValueError("requested controls profile differs from frozen root")
    if (root / "FINALIZED.json").exists():
        return verify_controls_output(root, workspace=workspace)
    validate_controls_source_manifest(source, workspace)
    validate_controls_matrix(
        matrix, frozen_protocol, frozen_parent, source
    )
    run_canonical_datasets(
        root, frozen_protocol, frozen_parent, source, matrix
    )
    run_stage_cells(frozen_protocol, frozen_parent, matrix, root)
    return finalize_controls_run(root, workspace=workspace)


__all__ = [
    "ACTOR_SCHEMA",
    "ANALYSIS_SCHEMA",
    "COMPUTE_SCHEMA",
    "CONTROL_SCHEMA",
    "MATRIX_SCHEMA",
    "ROLLOUT_SCHEMA",
    "SELECTION_SCHEMA",
    "WORLD_SCHEMA",
    "build_controls_analysis",
    "build_controls_artifact_manifest",
    "build_controls_matrix",
    "build_controls_selection",
    "build_controls_source_manifest",
    "controls_protocol_digest",
    "expected_arm_specs",
    "finalize_controls_run",
    "freeze_controls_run",
    "make_control_config",
    "read_controls_protocol",
    "run_controls_all",
    "run_control_cell",
    "run_canonical_dataset_unit",
    "run_stage_cells",
    "trajectory_control_loss",
    "trajectory_control_vector",
    "validate_controls_analysis",
    "validate_controls_artifact_manifest",
    "validate_controls_matrix",
    "validate_controls_protocol",
    "validate_controls_source_manifest",
    "verify_controls_output",
]
