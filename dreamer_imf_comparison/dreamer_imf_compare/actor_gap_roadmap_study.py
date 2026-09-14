"""Frozen exploratory study for the seven trajectory-iMF actor-gap remedies.

The study deliberately separates controller-side safeguards from model-side
training changes.  It authenticates (but never mutates) the completed ReBRAC +
ITPO dependency, derives every calibration quantity from training replay, uses
the dependency's held-out environment seeds, and aggregates actor seeds inside
each world-model seed before treating the three world seeds as top-level units.

This is an exploratory Reacher study, not a full FlowMPC, DreamerV4, or
multi-task reproduction.  In particular, the density of deterministic ReBRAC
is represented only by an explicitly labelled fixed-variance Gaussian kernel.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import read_json
from . import matched_objective_benchmark as benchmark
from . import flowmpc_actor_study as dependency

SCHEMA = "trajectory-imf-actor-gap-roadmap-study-v1"
CALIBRATION_SCHEMA = "trajectory-imf-actor-gap-calibration-v1"
DIAGNOSTIC_SCHEMA = "trajectory-imf-actor-gap-diagnostic-cell-v1"
MODEL_SCHEMA = "trajectory-imf-actor-gap-model-cell-v1"
EVALUATION_SCHEMA = "trajectory-imf-actor-gap-evaluation-cell-v1"
MARKER_SCHEMA = "trajectory-imf-actor-gap-marker-v1"
PREFLIGHT_MARKER_SCHEMA = "trajectory-imf-actor-gap-preflight-marker-v1"
PREFLIGHT_GATE_SCHEMA = "trajectory-imf-actor-gap-preflight-gate-v1"
SUBMISSION_MAP_SCHEMA = "trajectory-imf-actor-gap-submission-map-v1"
SUBMISSION_RECORD_SCHEMA = "trajectory-imf-actor-gap-slurm-submission-record-v1"
SLURM_ALLOCATION_SCHEMA = "trajectory-imf-actor-gap-slurm-allocation-v3"
SLURM_NODE_FQDN_SUFFIX = ".sc.intern.uni-leipzig.de"
STAGE_MARKER_SCHEMA = "trajectory-imf-actor-gap-stage-marker-v1"
REPORT_SCHEMA = "trajectory-imf-actor-gap-report-v1"
CHECKPOINT_VERSION = 1
TASK = "dmc_reacher_easy"
DEPENDENCY_SOURCE_COMMIT = "d872e6eccc4bcd9557297c8b38a904d9067179c5"
DEPENDENCY_MANIFEST_FILE_SHA256 = (
    "24942a4cba6debfc841b8a01571b54d2517c81637f3cb9ec1fef3a61146ea92a"
)
DEPENDENCY_REPORT_FILE_SHA256 = (
    "da29451575ed7d69e1f42b097dbe253986c8e833eddd6e5000e06a0853fd341e"
)
DEPENDENCY_MANIFEST_SHA256 = (
    "8cd712d2ce6497987e9ac5721ec52f90f6421920d695409418dc7afa04b39a9e"
)

WORLD_SEEDS = (211, 223, 227)
ACTOR_SEEDS = (311, 313)
MODEL_FAMILIES = (
    "uniform_prior",
    "approximate_policy_tilt_prior",
    "proof_consistent_cvaml_value_prior",
    "endpoint_h1_imf",
    "endpoint_anystep_imf",
)
REFERENCE_ARMS = ("R0_zero_shot_dependency", "R1_unconstrained_dependency")
FRESH_ARMS = (
    "A0_persistent_unconstrained_replay",
    "A1_persistent_trust",
    "A2_reset_trust",
    "A3_persistent_trust_heldout",
    "O1_recursive_action_sequence_gradient",
    "O2_recursive_action_sequence_cem",
    "P0_recursive_cem_ensemble_relative_mean",
    "P1_recursive_cem_relative_pessimism",
    "F0_uniform_prior_unconstrained",
    "F1_uniform_prior_trust",
    "F2_policy_tilt_prior_unconstrained",
    "F3_policy_tilt_prior_trust",
    "V1_cvaml_value_prior_unconstrained",
    "K0_endpoint_h1_recursive_cem_filtered",
    "K1_endpoint_anystep_direct_cem_filtered",
)
ALL_ARMS = REFERENCE_ARMS + FRESH_ARMS
FACTORIAL_CONTRASTS = {
    "trust": ["A1_persistent_trust", "A0_persistent_unconstrained_replay"],
    "persistence": ["A2_reset_trust", "A1_persistent_trust"],
    "heldout_acceptance": ["A3_persistent_trust_heldout", "A1_persistent_trust"],
    "optimizer": [
        "O2_recursive_action_sequence_cem",
        "O1_recursive_action_sequence_gradient",
    ],
    "ensemble_averaging": [
        "P0_recursive_cem_ensemble_relative_mean",
        "O2_recursive_action_sequence_cem",
    ],
    "relative_pessimism": [
        "P1_recursive_cem_relative_pessimism",
        "P0_recursive_cem_ensemble_relative_mean",
    ],
    "policy_tilt_unconstrained": [
        "F2_policy_tilt_prior_unconstrained",
        "F0_uniform_prior_unconstrained",
    ],
    "policy_tilt_trust": [
        "F3_policy_tilt_prior_trust",
        "F1_uniform_prior_trust",
    ],
    "value_equivalence": [
        "V1_cvaml_value_prior_unconstrained",
        "F0_uniform_prior_unconstrained",
    ],
    "direct_chunks": [
        "K1_endpoint_anystep_direct_cem_filtered",
        "K0_endpoint_h1_recursive_cem_filtered",
    ],
}

ACTOR_CONDITIONED_MODEL_FAMILIES = (
    "approximate_policy_tilt_prior",
    "proof_consistent_cvaml_value_prior",
)
SHARED_MODEL_FAMILIES = tuple(
    family
    for family in MODEL_FAMILIES
    if family not in ACTOR_CONDITIONED_MODEL_FAMILIES
)

MODEL_UPDATES = 5_000
MODEL_BATCH_SIZE = 16
MODEL_SEQUENCE_LENGTH = 32
MODEL_LEARNING_RATE = 3e-4
MODEL_GRAD_CLIP = 10.0
CVAML_SAMPLES = 4
CVAML_SCALE = 1.0
PLANNER_DISCOUNT = 0.99
CHUNK_HORIZONS = (1, 2, 3, 4, 5)
CHUNK_ENDPOINT_SCALE = 1.0
APPROXIMATE_POLICY_STD = 0.20
TILT_ETA_GRID = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
MINIMUM_TILT_ESS_FRACTION = 0.50
MAXIMUM_TILT_CLIPPED_FRACTION = 0.05
ANCHOR_COUNT = 256
TRUST_ANCHOR_MSE_BUDGET = 0.01
TRUST_CURRENT_LINF_BUDGET = 0.10
HELDOUT_MINIMUM_IMPROVEMENT = 0.0
PESSIMISM_COEFFICIENT = 1.0
CONTROLLER_HORIZON = 5
FLOWMPC_PARTICLES = 4_096
ACTION_SEQUENCE_PARTICLES = 64
CONTROLLER_STEP_SIZE = 5e-6
ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS = 10
ACTION_SEQUENCE_RESIDUAL_LIMIT = 0.10
IMAGINED_COVERAGE_STEP_INTERVAL = 50
IMAGINED_COVERAGE_PARTICLES = 8
EVALUATION_EPISODES = 2
MAXIMUM_ENVIRONMENT_STEPS = 1_000
PREFLIGHT_GATES = (
    "self_test",
    "implementation",
    "single_gpu",
    "library_suite",
    "comparison_suite",
)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _unsigned_digest(value: Mapping[str, Any], field: str) -> str:
    payload = json.loads(json.dumps(value))
    payload.pop(field, None)
    return benchmark.object_sha256(payload)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype.kind not in "fci" or bool(np.all(np.isfinite(value)))
    if isinstance(value, np.generic):
        return value.dtype.kind not in "fci" or bool(np.isfinite(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _write_pickle_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=destination.name + ".",
            delete=False,
        ) as handle:
            temporary = handle.name
            pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _write_npz_exclusive(path: str | Path, payload: Mapping[str, np.ndarray]) -> Path:
    """Publish a complete NPZ with an atomic no-overwrite hard-link seal."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=destination.name + ".", suffix=".npz"
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        benchmark._write_npz_atomic(temporary, payload)
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish complete immutable JSON without overwriting evidence."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=destination.name + ".",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is both atomic and no-overwrite.  A crash while
        # writing can therefore leave only an ignorable temporary, never a
        # truncated scientific artifact at the canonical destination.
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _fsync_directory(directory: str | Path) -> None:
    """Persist a newly linked directory entry before reporting success."""

    descriptor = os.open(Path(directory), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _staged_artifact_directory(final_directory: str | Path):
    """Publish a multi-file scientific cell with one atomic directory rename.

    Interrupted staging directories are deliberately retained under hidden,
    unique names as crash evidence.  Because no canonical path becomes visible
    before the rename, the canonical submitter can safely retry that one cell.
    """

    destination = Path(final_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"artifact bundle already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.partial-",
        )
    )
    try:
        yield staging
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        # Never delete crash evidence.  The hidden unique staging directory is
        # outside the canonical artifact paths and cannot authorize a result.
        raise


def _load_pickle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError("roadmap checkpoint must contain a dictionary")
    return value


def _reject_partial_artifacts(result_path: Path, *artifact_paths: Path) -> None:
    """Fail closed instead of overwriting evidence from an interrupted cell."""

    if result_path.exists():
        return
    retained = [str(path) for path in artifact_paths if path.exists()]
    if retained:
        raise ValueError(
            "partial immutable cell artifacts require a fresh audited output root: "
            + ", ".join(retained)
        )


def _validate_dependency_marker(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    marker_path = root / str(cell["marker_path"])
    result_path = root / str(cell["result_path"])
    marker = read_json(marker_path)
    if (
        marker.get("schema_version") != dependency.MARKER_SCHEMA
        or marker.get("status") != "verified"
        or marker.get("stage") != stage
        or marker.get("source_commit") != manifest.get("source_commit")
        or marker.get("manifest_sha256") != manifest.get("manifest_sha256")
        or marker.get("cell_id") != cell.get("cell_id")
        or marker.get("cell_index") != cell.get("index")
        or marker.get("result_file_sha256") != benchmark.file_sha256(result_path)
        or marker.get("marker_sha256") != _unsigned_digest(marker, "marker_sha256")
    ):
        raise ValueError(f"dependency {stage} marker is invalid")
    return marker


def authenticate_dependency(dependency_root: str | Path) -> dict[str, Any]:
    """Authenticate the old immutable study without executing its old source."""

    root = Path(dependency_root).resolve(strict=True)
    manifest_path = root / "manifest.json"
    report_path = root / "report.json"
    manifest_file_sha256 = benchmark.file_sha256(manifest_path)
    report_file_sha256 = benchmark.file_sha256(report_path)
    if (
        manifest_file_sha256 != DEPENDENCY_MANIFEST_FILE_SHA256
        or report_file_sha256 != DEPENDENCY_REPORT_FILE_SHA256
    ):
        raise ValueError("FlowMPC dependency files changed before deserialization")
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    if (
        manifest.get("schema_version") != dependency.SCHEMA
        or manifest.get("status") != "frozen_before_execution"
        or manifest.get("task") != TASK
        or tuple(manifest.get("world_model_seeds", ())) != WORLD_SEEDS
        or tuple(manifest.get("actor_seeds", ())) != ACTOR_SEEDS
        or manifest.get("manifest_sha256")
        != _unsigned_digest(manifest, "manifest_sha256")
        or report.get("schema_version") != dependency.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("source_commit") != manifest.get("source_commit")
        or report.get("manifest_sha256") != manifest.get("manifest_sha256")
        or report.get("completed_rebrac_cells") != dependency.EXPECTED_REBRAC_CELLS
        or report.get("completed_evaluation_cells")
        != dependency.EXPECTED_EVALUATION_CELLS
        or report.get("report_sha256") != _unsigned_digest(report, "report_sha256")
        or not _finite_tree(report)
        or manifest.get("source_commit") != DEPENDENCY_SOURCE_COMMIT
        or manifest.get("manifest_sha256") != DEPENDENCY_MANIFEST_SHA256
    ):
        raise ValueError("FlowMPC dependency identity is invalid")
    for cell in manifest["rebrac_cells"]:
        marker = _validate_dependency_marker(root, manifest, cell, "rebrac")
        checkpoint = root / str(cell["checkpoint_path"])
        if marker.get("checkpoint_sha256") != benchmark.file_sha256(checkpoint):
            raise ValueError("dependency ReBRAC checkpoint digest differs")
    for cell in manifest["evaluation_cells"]:
        marker = _validate_dependency_marker(root, manifest, cell, "evaluation")
        result = read_json(root / str(cell["result_path"]))
        trace = root / str(cell["trace_path"])
        if (
            marker.get("strict_policy_and_environment_replay") is not True
            or marker.get("trace_file_sha256") != benchmark.file_sha256(trace)
            or result.get("trace_file_sha256") != benchmark.file_sha256(trace)
            or result.get("trace_sha256")
            != benchmark.array_sha256(benchmark.load_npz(trace))
            or result.get("evaluation_seeds") != cell.get("evaluation_seeds")
        ):
            raise ValueError("dependency evaluation evidence is invalid")
    reward_root = Path(str(manifest["reward_root"])).resolve(strict=True)
    if benchmark.file_sha256(reward_root / "manifest.json") != manifest.get(
        "reward_manifest_file_sha256"
    ) or benchmark.file_sha256(reward_root / "report.json") != manifest.get(
        "reward_report_file_sha256"
    ):
        raise ValueError("dependency reward root changed")
    reward_sources = manifest.get("reward_sources", ())
    if (
        len(reward_sources) != len(WORLD_SEEDS)
        or tuple(int(row["world_model_seed"]) for row in reward_sources) != WORLD_SEEDS
    ):
        raise ValueError("dependency reward-source closure differs")
    for source in reward_sources:
        for path_key, digest_key in (
            ("reward_result", "reward_result_sha256"),
            ("reward_checkpoint", "reward_checkpoint_sha256"),
            ("reward_marker", "reward_marker_sha256"),
            ("dataset", "dataset_file_sha256"),
            ("world_model_checkpoint", "world_model_checkpoint_sha256"),
        ):
            source_path = Path(str(source[path_key])).resolve(strict=True)
            if benchmark.file_sha256(source_path) != source[digest_key]:
                raise ValueError(f"dependency reward source changed: {path_key}")
    return {
        "root": str(root),
        "source_commit": manifest["source_commit"],
        "manifest": manifest,
        "report": report,
        "manifest_file_sha256": manifest_file_sha256,
        "report_file_sha256": report_file_sha256,
        "reward_root": str(reward_root),
    }


def build_study_matrix(
    dependency_manifest: Mapping[str, Any],
    *,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, list[dict[str, Any]]]:
    """Build the deterministic logical cells without reading result values."""

    if not 1 <= evaluation_episodes <= dependency.EVALUATION_EPISODES:
        raise ValueError("evaluation_episodes exceeds the frozen dependency seeds")
    dependency_evaluations = {
        (int(row["world_model_seed"]), int(row["actor_seed"])): row
        for row in dependency_manifest["evaluation_cells"]
    }
    diagnostic_cells = [
        {
            "index": index,
            "cell_id": f"diagnostic-{world_seed}",
            "world_model_seed": world_seed,
            "actor_seeds": list(ACTOR_SEEDS),
            "result_path": f"diagnostics/diagnostic-{world_seed}/result.json",
            "trace_path": f"diagnostics/diagnostic-{world_seed}/traces.npz",
            "marker_path": f"verified/diagnostic-{world_seed}.json",
        }
        for index, world_seed in enumerate(WORLD_SEEDS)
    ]
    model_cells: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        actor_seeds: tuple[int | None, ...] = (
            tuple(ACTOR_SEEDS)
            if family in ACTOR_CONDITIONED_MODEL_FAMILIES
            else (None,)
        )
        for world_seed in WORLD_SEEDS:
            for actor_seed in actor_seeds:
                actor_suffix = "" if actor_seed is None else f"-{actor_seed}"
                cell_id = f"model-{family}-{world_seed}{actor_suffix}"
                model_cells.append(
                    {
                        "index": len(model_cells),
                        "cell_id": cell_id,
                        "family": family,
                        "world_model_seed": world_seed,
                        "actor_seed": actor_seed,
                        "result_path": f"models/{cell_id}/result.json",
                        "checkpoint_path": f"models/{cell_id}/checkpoint.pkl",
                        "schedule_path": f"models/{cell_id}/schedule.npz",
                        "marker_path": f"verified/{cell_id}.json",
                    }
                )
    evaluation_cells: list[dict[str, Any]] = []
    for world_seed in WORLD_SEEDS:
        for actor_seed in ACTOR_SEEDS:
            dependency_cell = dependency_evaluations[(world_seed, actor_seed)]
            seeds = [int(value) for value in dependency_cell["evaluation_seeds"]][
                :evaluation_episodes
            ]
            for arm in FRESH_ARMS:
                cell_id = f"evaluation-{arm}-{world_seed}-{actor_seed}"
                evaluation_cells.append(
                    {
                        "index": len(evaluation_cells),
                        "cell_id": cell_id,
                        "arm": arm,
                        "world_model_seed": world_seed,
                        "actor_seed": actor_seed,
                        "evaluation_seeds": seeds,
                        "dependency_cell_id": dependency_cell["cell_id"],
                        "result_path": f"evaluation/{cell_id}/result.json",
                        "trace_path": f"evaluation/{cell_id}/traces.npz",
                        "marker_path": f"verified/{cell_id}.json",
                    }
                )
    return {
        "diagnostic_cells": diagnostic_cells,
        "model_cells": model_cells,
        "evaluation_cells": evaluation_cells,
    }


def build_manifest(
    dependency_root: str | Path,
    *,
    model_updates: int = MODEL_UPDATES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    if model_updates <= 0:
        raise ValueError("model_updates must be positive")
    source_manifest = benchmark.build_source_manifest()
    source_git = source_manifest["git"]
    if (
        source_git.get("commit_status") != "complete"
        or source_git.get("dirty_patch_status") != "complete"
        or source_git.get("untracked_status") != "complete"
        or source_git.get("dirty_patch_bytes") != 0
        or source_git.get("untracked_files")
        or source_git.get("commit") != _git_commit()
    ):
        raise ValueError("roadmap execution requires a clean committed source tree")
    authenticated = authenticate_dependency(dependency_root)
    matrix = build_study_matrix(
        authenticated["manifest"], evaluation_episodes=evaluation_episodes
    )
    dependency_tuning_seed = int(
        authenticated["manifest"]["inference_tuning"]["environment_seed"]
    )
    evaluation_seeds = {
        seed for cell in matrix["evaluation_cells"] for seed in cell["evaluation_seeds"]
    }
    if dependency_tuning_seed in evaluation_seeds:
        raise ValueError("dependency tuning seed leaked into roadmap evaluation")
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "source_manifest": source_manifest,
        "task": TASK,
        "claim_eligible": False,
        "evidence_class": "exploratory_single_task_actor_gap_factorial",
        "dependency_root": authenticated["root"],
        "dependency_source_commit": authenticated["source_commit"],
        "dependency_manifest_file_sha256": authenticated["manifest_file_sha256"],
        "dependency_report_file_sha256": authenticated["report_file_sha256"],
        "reward_root": authenticated["reward_root"],
        "world_model_seeds": list(WORLD_SEEDS),
        "actor_seeds_nested_within_world_model_seed": list(ACTOR_SEEDS),
        "reference_arms": list(REFERENCE_ARMS),
        "fresh_arms": list(FRESH_ARMS),
        "all_arms": list(ALL_ARMS),
        "model_families": list(MODEL_FAMILIES),
        "preflight": {
            "marker_path": "verified/preflight.json",
            "gate_paths": {
                gate: f"verified/preflight-{gate.replace('_', '-')}.json"
                for gate in PREFLIGHT_GATES
            },
        },
        "calibration": {
            "partition": "training_replay_only",
            "result_path": "calibration/result.json",
            "arrays_path": "calibration/arrays.npz",
            "marker_path": "verified/calibration.json",
            "anchor_count": ANCHOR_COUNT,
            "approximate_policy_std": APPROXIMATE_POLICY_STD,
            "tilt_eta_grid": list(TILT_ETA_GRID),
            "minimum_ess_fraction": MINIMUM_TILT_ESS_FRACTION,
            "maximum_clipped_fraction": MAXIMUM_TILT_CLIPPED_FRACTION,
        },
        "model_training": {
            "updates": int(model_updates),
            "batch_size": MODEL_BATCH_SIZE,
            "sequence_length": MODEL_SEQUENCE_LENGTH,
            "learning_rate": MODEL_LEARNING_RATE,
            "grad_clip": MODEL_GRAD_CLIP,
            "trainable_subtree": "prior_only_or_fresh_endpoint_imf_only",
            "new_optimizer": "zero_moment_adam",
            "cvaml_samples": CVAML_SAMPLES,
            "cvaml_scale": CVAML_SCALE,
            "cvaml_estimator": "proof_consistent_sample_var_ddof1_divided_by_K",
            "planner_discount": PLANNER_DISCOUNT,
            "planner_reward_interface": "frozen_state_action_reward_r_observation_action",
            "planner_continuation": "fixed_one_matching_flowmpc",
            "chunk_horizons": list(CHUNK_HORIZONS),
            "chunk_endpoint_scale": CHUNK_ENDPOINT_SCALE,
        },
        "controller": {
            "horizon": CONTROLLER_HORIZON,
            "flowmpc_particles": FLOWMPC_PARTICLES,
            "action_sequence_particles": ACTION_SEQUENCE_PARTICLES,
            "flowmpc_step_size": CONTROLLER_STEP_SIZE,
            "trust_anchor_mse_sum_budget": TRUST_ANCHOR_MSE_BUDGET,
            "trust_current_linf_budget": TRUST_CURRENT_LINF_BUDGET,
            "heldout_minimum_improvement": HELDOUT_MINIMUM_IMPROVEMENT,
            "pessimism_coefficient": PESSIMISM_COEFFICIENT,
            "action_sequence_objective_evaluations": ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS,
            "action_sequence_residual_limit": ACTION_SEQUENCE_RESIDUAL_LIMIT,
            "gradient_and_cem_same_decision_variables": True,
            "ensemble_posterior_common_random_numbers": True,
            "equal_objective_calls_not_claimed_equal_flops": True,
            "discarded_pure_compile_warmup": True,
        },
        "diagnostics": {
            "proposal_and_heldout_objectives": "post_update_minus_pre_update",
            "independent_noise_primary_quantity": (
                "heldout_improvement_minus_proposal_improvement_generalization_gap"
            ),
            "exact_counterfactual_horizon": CONTROLLER_HORIZON,
            "exact_counterfactual_maximum_reset_rows": 6,
            "exact_counterfactual_candidate_sequences": (
                "a0_reference_trace_and_full_horizon_plus_minus_0.1_action_dim_0"
            ),
            "exact_counterfactual_planner_particles": ACTION_SEQUENCE_PARTICLES,
            "exact_counterfactual_terminal_value": (
                "same_frozen_rebrac_min_q_on_predicted_or_real_terminal_observation"
            ),
            "imagined_coverage_scopes": ["proposal", "best", "reference"],
            "live_planner_imagined_coverage_step_interval": (
                IMAGINED_COVERAGE_STEP_INTERVAL
            ),
            "live_planner_imagined_coverage_particles": (IMAGINED_COVERAGE_PARTICLES),
        },
        "evaluation_episodes": int(evaluation_episodes),
        "maximum_environment_steps": MAXIMUM_ENVIRONMENT_STEPS,
        "dependency_tuning_seed": dependency_tuning_seed,
        **matrix,
        "factorial_contrasts": FACTORIAL_CONTRASTS,
        "declared_limitations": [
            "single exploratory DMC Reacher task with three top-level world-model seeds",
            "two nested actor seeds and two held-out episodes per actor are not independent top-level units",
            "deterministic ReBRAC has no Lebesgue action density; policy tilt uses a labelled fixed-variance Gaussian approximation",
            "relative mean-minus-SD pessimism is a heuristic risk score, not a confidence bound",
            "the three pessimism ensemble members are the same top-level world seeds and are statistically coupled",
            "equal action-sequence objective calls do not imply equal FLOPs because gradient search has backward passes",
            "four-candidate two-iteration CEM is an objective-call diagnostic, not a tuned derivative-free controller",
            "the exact horizon-5 counterfactual uses a fixed three-sequence local diagnostic proposal set, not the complete live CEM population",
            "endpoint models operate in decoded observation space and are not a full FlowMPC policy-tilt reproduction",
            "proof-consistent CVAML differs at finite K from the paper main-text shorthand and released implementation",
            "no evaluation return is used for calibration, selection, or mutation of this frozen matrix",
        ],
    }
    body["manifest_sha256"] = benchmark.object_sha256(body)
    validate_manifest(body)
    return body


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "frozen_before_execution"
        or manifest.get("source_commit") != _git_commit()
        or manifest.get("task") != TASK
        or manifest.get("claim_eligible") is not False
        or manifest.get("evidence_class")
        != "exploratory_single_task_actor_gap_factorial"
        or manifest.get("dependency_manifest_file_sha256")
        != DEPENDENCY_MANIFEST_FILE_SHA256
        or manifest.get("dependency_report_file_sha256")
        != DEPENDENCY_REPORT_FILE_SHA256
        or tuple(manifest.get("world_model_seeds", ())) != WORLD_SEEDS
        or tuple(manifest.get("actor_seeds_nested_within_world_model_seed", ()))
        != ACTOR_SEEDS
        or tuple(manifest.get("reference_arms", ())) != REFERENCE_ARMS
        or tuple(manifest.get("fresh_arms", ())) != FRESH_ARMS
        or tuple(manifest.get("all_arms", ())) != ALL_ARMS
        or tuple(manifest.get("model_families", ())) != MODEL_FAMILIES
        or len(manifest.get("diagnostic_cells", ())) != len(WORLD_SEEDS)
        or len(manifest.get("model_cells", ())) != 21
        or len(manifest.get("evaluation_cells", ()))
        != len(WORLD_SEEDS) * len(ACTOR_SEEDS) * len(FRESH_ARMS)
        or manifest.get("manifest_sha256")
        != _unsigned_digest(manifest, "manifest_sha256")
    ):
        raise ValueError("actor-gap roadmap manifest identity is invalid")
    benchmark.validate_source_manifest(manifest["source_manifest"])
    source_git = manifest["source_manifest"]["git"]
    if (
        source_git.get("commit") != manifest["source_commit"]
        or source_git.get("dirty_patch_bytes") != 0
        or source_git.get("untracked_files")
    ):
        raise ValueError("roadmap source closure is not a clean exact commit")
    dependency_root = Path(str(manifest["dependency_root"]))
    if str(dependency_root.resolve(strict=True)) != str(manifest["dependency_root"]):
        raise ValueError("roadmap dependency root is not canonical and absolute")
    if (
        benchmark.file_sha256(dependency_root / "manifest.json")
        != manifest["dependency_manifest_file_sha256"]
        or benchmark.file_sha256(dependency_root / "report.json")
        != manifest["dependency_report_file_sha256"]
    ):
        raise ValueError("actor-gap roadmap dependency changed")
    dependency_manifest = _dependency_manifest(manifest)
    reward_sources = dependency_manifest.get("reward_sources", ())
    if (
        len(reward_sources) != len(WORLD_SEEDS)
        or tuple(int(row["world_model_seed"]) for row in reward_sources) != WORLD_SEEDS
    ):
        raise ValueError("roadmap dependency reward-source closure differs")
    for source in reward_sources:
        for path_key, digest_key in (
            ("reward_result", "reward_result_sha256"),
            ("reward_checkpoint", "reward_checkpoint_sha256"),
            ("reward_marker", "reward_marker_sha256"),
            ("dataset", "dataset_file_sha256"),
            ("world_model_checkpoint", "world_model_checkpoint_sha256"),
        ):
            source_path = Path(str(source[path_key])).resolve(strict=True)
            if benchmark.file_sha256(source_path) != source[digest_key]:
                raise ValueError(f"roadmap dependency source changed: {path_key}")
    if manifest.get("dependency_source_commit") != DEPENDENCY_SOURCE_COMMIT:
        raise ValueError("roadmap dependency commit is not the frozen d872 study")
    if manifest.get("preflight") != {
        "marker_path": "verified/preflight.json",
        "gate_paths": {
            gate: f"verified/preflight-{gate.replace('_', '-')}.json"
            for gate in PREFLIGHT_GATES
        },
    }:
        raise ValueError("roadmap preflight contract differs")
    calibration_expected = {
        "partition": "training_replay_only",
        "result_path": "calibration/result.json",
        "arrays_path": "calibration/arrays.npz",
        "marker_path": "verified/calibration.json",
        "anchor_count": ANCHOR_COUNT,
        "approximate_policy_std": APPROXIMATE_POLICY_STD,
        "tilt_eta_grid": list(TILT_ETA_GRID),
        "minimum_ess_fraction": MINIMUM_TILT_ESS_FRACTION,
        "maximum_clipped_fraction": MAXIMUM_TILT_CLIPPED_FRACTION,
    }
    if manifest.get("calibration") != calibration_expected:
        raise ValueError("roadmap calibration contract differs")
    model_settings = manifest.get("model_training", {})
    if (
        not isinstance(model_settings.get("updates"), int)
        or isinstance(model_settings.get("updates"), bool)
        or int(model_settings["updates"]) <= 0
        or {key: value for key, value in model_settings.items() if key != "updates"}
        != {
            "batch_size": MODEL_BATCH_SIZE,
            "sequence_length": MODEL_SEQUENCE_LENGTH,
            "learning_rate": MODEL_LEARNING_RATE,
            "grad_clip": MODEL_GRAD_CLIP,
            "trainable_subtree": "prior_only_or_fresh_endpoint_imf_only",
            "new_optimizer": "zero_moment_adam",
            "cvaml_samples": CVAML_SAMPLES,
            "cvaml_scale": CVAML_SCALE,
            "cvaml_estimator": ("proof_consistent_sample_var_ddof1_divided_by_K"),
            "planner_discount": PLANNER_DISCOUNT,
            "planner_reward_interface": (
                "frozen_state_action_reward_r_observation_action"
            ),
            "planner_continuation": "fixed_one_matching_flowmpc",
            "chunk_horizons": list(CHUNK_HORIZONS),
            "chunk_endpoint_scale": CHUNK_ENDPOINT_SCALE,
        }
    ):
        raise ValueError("roadmap model-training contract differs")
    if manifest.get("controller") != {
        "horizon": CONTROLLER_HORIZON,
        "flowmpc_particles": FLOWMPC_PARTICLES,
        "action_sequence_particles": ACTION_SEQUENCE_PARTICLES,
        "flowmpc_step_size": CONTROLLER_STEP_SIZE,
        "trust_anchor_mse_sum_budget": TRUST_ANCHOR_MSE_BUDGET,
        "trust_current_linf_budget": TRUST_CURRENT_LINF_BUDGET,
        "heldout_minimum_improvement": HELDOUT_MINIMUM_IMPROVEMENT,
        "pessimism_coefficient": PESSIMISM_COEFFICIENT,
        "action_sequence_objective_evaluations": (
            ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS
        ),
        "action_sequence_residual_limit": ACTION_SEQUENCE_RESIDUAL_LIMIT,
        "gradient_and_cem_same_decision_variables": True,
        "ensemble_posterior_common_random_numbers": True,
        "equal_objective_calls_not_claimed_equal_flops": True,
        "discarded_pure_compile_warmup": True,
    }:
        raise ValueError("roadmap controller contract differs")
    if manifest.get("diagnostics") != {
        "proposal_and_heldout_objectives": "post_update_minus_pre_update",
        "independent_noise_primary_quantity": (
            "heldout_improvement_minus_proposal_improvement_generalization_gap"
        ),
        "exact_counterfactual_horizon": CONTROLLER_HORIZON,
        "exact_counterfactual_maximum_reset_rows": 6,
        "exact_counterfactual_candidate_sequences": (
            "a0_reference_trace_and_full_horizon_plus_minus_0.1_action_dim_0"
        ),
        "exact_counterfactual_planner_particles": ACTION_SEQUENCE_PARTICLES,
        "exact_counterfactual_terminal_value": (
            "same_frozen_rebrac_min_q_on_predicted_or_real_terminal_observation"
        ),
        "imagined_coverage_scopes": ["proposal", "best", "reference"],
        "live_planner_imagined_coverage_step_interval": (
            IMAGINED_COVERAGE_STEP_INTERVAL
        ),
        "live_planner_imagined_coverage_particles": IMAGINED_COVERAGE_PARTICLES,
    }:
        raise ValueError("roadmap diagnostic contract differs")
    if (
        not isinstance(manifest.get("evaluation_episodes"), int)
        or isinstance(manifest.get("evaluation_episodes"), bool)
        or not 1
        <= int(manifest["evaluation_episodes"])
        <= dependency.EVALUATION_EPISODES
        or manifest.get("maximum_environment_steps") != MAXIMUM_ENVIRONMENT_STEPS
        or manifest.get("factorial_contrasts") != FACTORIAL_CONTRASTS
    ):
        raise ValueError("roadmap evaluation or contrast contract differs")
    indices = {
        key: [int(row["index"]) for row in manifest[key]]
        for key in ("diagnostic_cells", "model_cells", "evaluation_cells")
    }
    if any(values != list(range(len(values))) for values in indices.values()):
        raise ValueError("roadmap cell indices are not contiguous")
    expected_models = {
        (
            family,
            world_seed,
            actor_seed if family in ACTOR_CONDITIONED_MODEL_FAMILIES else None,
        )
        for family in MODEL_FAMILIES
        for world_seed in WORLD_SEEDS
        for actor_seed in (
            ACTOR_SEEDS if family in ACTOR_CONDITIONED_MODEL_FAMILIES else (None,)
        )
    }
    observed_models = {
        (row["family"], int(row["world_model_seed"]), row.get("actor_seed"))
        for row in manifest["model_cells"]
    }
    if observed_models != expected_models:
        raise ValueError("roadmap model factorial differs")
    expected_evaluations = {
        (arm, world_seed, actor_seed)
        for arm in FRESH_ARMS
        for world_seed in WORLD_SEEDS
        for actor_seed in ACTOR_SEEDS
    }
    observed_evaluations = {
        (row["arm"], int(row["world_model_seed"]), int(row["actor_seed"]))
        for row in manifest["evaluation_cells"]
    }
    if observed_evaluations != expected_evaluations:
        raise ValueError("roadmap evaluation factorial differs")
    artifact_paths: list[str] = []
    for collection in ("diagnostic_cells", "model_cells", "evaluation_cells"):
        for row in manifest[collection]:
            for key in (
                "result_path",
                "checkpoint_path",
                "schedule_path",
                "trace_path",
                "marker_path",
            ):
                if key not in row:
                    continue
                value = str(row[key])
                pure = PurePosixPath(value)
                if (
                    not value
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or pure.as_posix() != value
                ):
                    raise ValueError("roadmap artifact path is not canonical")
                artifact_paths.append(value)
    calibration_paths = [
        str(manifest["calibration"][key])
        for key in ("result_path", "arrays_path", "marker_path")
    ]
    artifact_paths.extend(calibration_paths)
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("roadmap artifact paths are not unique")
    old = _dependency_manifest(manifest)
    if str(Path(str(manifest["reward_root"])).resolve()) != str(
        Path(str(old["reward_root"])).resolve()
    ):
        raise ValueError("roadmap reward dependency root differs")
    expected_matrix = build_study_matrix(
        old, evaluation_episodes=int(manifest["evaluation_episodes"])
    )
    if any(
        manifest[key] != expected_matrix[key]
        for key in ("diagnostic_cells", "model_cells", "evaluation_cells")
    ):
        raise ValueError("roadmap cell matrix differs from the frozen dependency")
    dependency_evaluations = {
        (int(row["world_model_seed"]), int(row["actor_seed"])): row
        for row in old["evaluation_cells"]
    }
    for cell in manifest["evaluation_cells"]:
        inherited = dependency_evaluations[
            (int(cell["world_model_seed"]), int(cell["actor_seed"]))
        ]
        if (
            len(cell["evaluation_seeds"]) != int(manifest["evaluation_episodes"])
            or int(manifest["dependency_tuning_seed"]) in cell["evaluation_seeds"]
            or cell["evaluation_seeds"]
            != inherited["evaluation_seeds"][: int(manifest["evaluation_episodes"])]
            or cell["dependency_cell_id"] != inherited["cell_id"]
        ):
            raise ValueError("roadmap evaluation seed contract differs")


def write_manifest(
    dependency_root: str | Path,
    output_root: str | Path,
    *,
    model_updates: int = MODEL_UPDATES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    path = Path(output_root) / "manifest.json"
    if path.is_file():
        manifest = read_json(path)
        validate_manifest(manifest)
        if (
            str(Path(dependency_root).resolve(strict=True))
            != str(Path(str(manifest["dependency_root"])).resolve(strict=True))
            or int(manifest["model_training"]["updates"]) != int(model_updates)
            or int(manifest["evaluation_episodes"]) != int(evaluation_episodes)
        ):
            raise ValueError("existing roadmap manifest arguments differ")
        return manifest
    manifest = build_manifest(
        dependency_root,
        model_updates=model_updates,
        evaluation_episodes=evaluation_episodes,
    )
    _write_json_exclusive(path, manifest)
    return manifest


def write_preflight_gate(
    output_root: str | Path,
    gate: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Record one gate only after its verifier has returned successfully."""

    if gate not in PREFLIGHT_GATES:
        raise ValueError(f"unknown preflight gate: {gate}")
    if not evidence or not _finite_tree(evidence):
        raise ValueError("preflight gate evidence must be nonempty and finite")
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    path = root / str(manifest["preflight"]["gate_paths"][gate])
    if path.is_file():
        return _validate_preflight_gate(root, manifest, gate)
    scheduler, receipt, runtime = _live_single_execution_binding(
        root, manifest, "preflight"
    )
    body = {
        "schema_version": PREFLIGHT_GATE_SCHEMA,
        "status": "verified",
        "gate": gate,
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "evidence": dict(evidence),
        "scheduler_provenance": scheduler,
        "submission_receipt": receipt,
        "runtime": runtime,
    }
    body["gate_sha256"] = benchmark.object_sha256(body)
    _write_json_exclusive(path, body)
    return body


def _validate_preflight_gate(
    root: Path, manifest: Mapping[str, Any], gate: str
) -> dict[str, Any]:
    if gate not in PREFLIGHT_GATES:
        raise ValueError(f"unknown preflight gate: {gate}")
    path = root / str(manifest["preflight"]["gate_paths"][gate])
    body = read_json(path)
    scheduler, receipt, runtime = _validate_retained_single_execution_binding(
        root,
        manifest,
        "preflight",
        body.get("scheduler_provenance", {}),
        body.get("submission_receipt", {}),
        body.get("runtime", {}),
    )
    if (
        body.get("schema_version") != PREFLIGHT_GATE_SCHEMA
        or body.get("status") != "verified"
        or body.get("gate") != gate
        or body.get("source_commit") != manifest["source_commit"]
        or body.get("manifest_sha256") != manifest["manifest_sha256"]
        or not isinstance(body.get("evidence"), Mapping)
        or not body["evidence"]
        or not _finite_tree(body["evidence"])
        or body.get("scheduler_provenance") != scheduler
        or body.get("submission_receipt") != receipt
        or body.get("runtime") != runtime
        or body.get("gate_sha256") != _unsigned_digest(body, "gate_sha256")
    ):
        raise ValueError(f"preflight {gate} gate is invalid")
    return body


def _require_preflight_runtime_match(
    root: Path, manifest: Mapping[str, Any], runtime: Mapping[str, Any]
) -> None:
    gate = _validate_preflight_gate(root, manifest, "single_gpu")
    if benchmark.runtime_homogeneity_identity(
        runtime
    ) != benchmark.runtime_homogeneity_identity(gate["runtime"]):
        raise ValueError("worker runtime differs from the frozen preflight runtime")


def _preflight_marker_body(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    authenticated = authenticate_dependency(manifest["dependency_root"])
    gates = {
        gate: _validate_preflight_gate(root, manifest, gate) for gate in PREFLIGHT_GATES
    }
    runtime_identities = [
        benchmark.runtime_homogeneity_identity(gates[gate]["runtime"])
        for gate in PREFLIGHT_GATES
    ]
    if any(identity != runtime_identities[0] for identity in runtime_identities[1:]):
        raise ValueError("preflight gate runtime identities differ")
    return {
        "schema_version": PREFLIGHT_MARKER_SCHEMA,
        "status": "verified",
        "stage": "preflight",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": benchmark.file_sha256(root / "manifest.json"),
        "dependency_source_commit": authenticated["source_commit"],
        "dependency_manifest_file_sha256": authenticated["manifest_file_sha256"],
        "dependency_report_file_sha256": authenticated["report_file_sha256"],
        "diagnostic_cells": len(manifest["diagnostic_cells"]),
        "model_cells": len(manifest["model_cells"]),
        "evaluation_cells": len(manifest["evaluation_cells"]),
        "claim_eligible": False,
        "gate_evidence": {
            gate: {
                "path": manifest["preflight"]["gate_paths"][gate],
                "file_sha256": benchmark.file_sha256(
                    root / str(manifest["preflight"]["gate_paths"][gate])
                ),
                "gate_sha256": gates[gate]["gate_sha256"],
            }
            for gate in PREFLIGHT_GATES
        },
        **{f"{gate}_verified": True for gate in PREFLIGHT_GATES},
    }


def write_preflight_marker(output_root: str | Path) -> dict[str, Any]:
    """Persist the complete preflight gate as immutable authenticated evidence."""

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    marker = _preflight_marker_body(root, manifest)
    scheduler, receipt, runtime = _live_single_execution_binding(
        root, manifest, "preflight"
    )
    _require_preflight_runtime_match(root, manifest, runtime)
    marker.update(
        {
            "scheduler_provenance": scheduler,
            "submission_receipt": receipt,
            "runtime": runtime,
        }
    )
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / str(manifest["preflight"]["marker_path"])
    if marker_path.is_file():
        if read_json(marker_path) != marker:
            raise ValueError("existing preflight marker differs")
        return marker
    _write_json_exclusive(marker_path, marker)
    return marker


def validate_preflight_marker(output_root: str | Path) -> dict[str, Any]:
    """Independently reproduce and validate the durable preflight marker."""

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    marker_path = root / str(manifest["preflight"]["marker_path"])
    marker = read_json(marker_path)
    scheduler, receipt, runtime = _validate_retained_single_execution_binding(
        root,
        manifest,
        "preflight",
        marker.get("scheduler_provenance", {}),
        marker.get("submission_receipt", {}),
        marker.get("runtime", {}),
    )
    _require_preflight_runtime_match(root, manifest, runtime)
    expected = _preflight_marker_body(root, manifest)
    expected.update(
        {
            "scheduler_provenance": scheduler,
            "submission_receipt": receipt,
            "runtime": runtime,
        }
    )
    expected["marker_sha256"] = benchmark.object_sha256(expected)
    if marker != expected:
        raise ValueError("preflight marker is invalid")
    return marker


def _cell(manifest: Mapping[str, Any], stage: str, index: int) -> Mapping[str, Any]:
    rows = [
        row for row in manifest[f"{stage}_cells"] if int(row["index"]) == int(index)
    ]
    if len(rows) != 1:
        raise ValueError(f"{stage} cell is absent or duplicated")
    return rows[0]


def _submission_upstream_marker(
    root: Path, manifest: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    if stage == "diagnostic":
        path = root / str(manifest["calibration"]["marker_path"])
        expected_stage = "calibration"
    elif stage == "model":
        path = root / _STAGE_PATHS["diagnostic"]
        expected_stage = "diagnostic"
    elif stage == "evaluation":
        path = root / _STAGE_PATHS["model"]
        expected_stage = "model"
    else:
        raise ValueError(f"unknown submission stage: {stage}")
    marker = read_json(path)
    if (
        marker.get("status") != "verified"
        or marker.get("stage") != expected_stage
        or marker.get("source_commit") != manifest["source_commit"]
        or marker.get("manifest_sha256") != manifest["manifest_sha256"]
        or marker.get("marker_sha256") != _unsigned_digest(marker, "marker_sha256")
    ):
        raise ValueError(f"{stage} submission upstream marker is invalid")
    return marker


def _cell_artifact_paths(
    root: Path, cell: Mapping[str, Any], stage: str
) -> tuple[Path, ...]:
    keys = {
        "diagnostic": ("result_path", "trace_path", "marker_path"),
        "model": ("result_path", "schedule_path", "checkpoint_path", "marker_path"),
        "evaluation": ("result_path", "trace_path", "marker_path"),
    }[stage]
    return tuple(root / str(cell[key]) for key in keys)


def _retained_data_file_records(
    root: Path, cell: Mapping[str, Any], stage: str
) -> list[dict[str, str]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "file_sha256": benchmark.file_sha256(path),
        }
        for path in _cell_artifact_paths(root, cell, stage)[:-1]
    ]


def _submission_entry_digest(entry: Mapping[str, Any]) -> str:
    payload = dict(entry)
    payload.pop("artifact_state_sha256", None)
    return benchmark.object_sha256(payload)


def write_submission_map(
    output_root: str | Path,
    stage: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze one exact new/verification-only array subset before submission."""

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    upstream = _submission_upstream_marker(root, manifest, stage)
    normalized = [
        {"index": int(entry["index"]), "mode": str(entry["mode"])} for entry in entries
    ]
    if (
        not normalized
        or len({entry["index"] for entry in normalized}) != len(normalized)
        or any(
            entry["mode"] not in {"new", "verification_only"} for entry in normalized
        )
    ):
        raise ValueError("submission map entries are empty, duplicated, or invalid")
    for entry in normalized:
        cell = _cell(manifest, stage, entry["index"])
        paths = _cell_artifact_paths(root, cell, stage)
        data_paths, marker_path = paths[:-1], paths[-1]
        if entry["mode"] == "new" and any(path.exists() for path in paths):
            raise ValueError("new submission index already has retained artifacts")
        if entry["mode"] == "verification_only" and (
            not all(path.is_file() for path in data_paths) or marker_path.exists()
        ):
            raise ValueError("verification-only submission artifacts differ")
        entry["retained_data_files"] = (
            []
            if entry["mode"] == "new"
            else _retained_data_file_records(root, cell, stage)
        )
        entry["artifact_state_sha256"] = _submission_entry_digest(entry)
    submission_dir = root / "submissions"
    attempt = 1
    while (submission_dir / f"{stage}-attempt-{attempt:03d}.json").exists():
        attempt += 1
    relative_path = f"submissions/{stage}-attempt-{attempt:03d}.json"
    body = {
        "schema_version": SUBMISSION_MAP_SCHEMA,
        "status": "frozen_before_submission",
        "stage": stage,
        "attempt": attempt,
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "upstream_stage": upstream["stage"],
        "upstream_marker_sha256": upstream["marker_sha256"],
        "entries": normalized,
        "map_path": relative_path,
    }
    body["map_sha256"] = benchmark.object_sha256(body)
    path = root / relative_path
    if path.exists():
        raise ValueError("submission map path already exists")
    _write_json_exclusive(path, body)
    return body


def validate_submission_map(
    output_root: str | Path,
    map_path: str | Path,
    stage: str,
    index: int,
    *,
    expected_file_sha256: str,
    action: str = "create",
) -> dict[str, Any]:
    """Authorize exactly one array index from an immutable submission map."""

    if action not in {"create", "verify"}:
        raise ValueError("submission-map action must be create or verify")

    root = Path(output_root).resolve(strict=True)
    path = Path(map_path).resolve(strict=True)
    submission_root = (root / "submissions").resolve(strict=True)
    if not path.is_relative_to(submission_root):
        raise ValueError("submission map is outside the output root")
    if benchmark.file_sha256(path) != expected_file_sha256:
        raise ValueError("submission map file digest differs")
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    body = read_json(path)
    upstream = _submission_upstream_marker(root, manifest, stage)
    matching = [
        entry for entry in body.get("entries", ()) if int(entry["index"]) == int(index)
    ]
    if (
        body.get("schema_version") != SUBMISSION_MAP_SCHEMA
        or body.get("status") != "frozen_before_submission"
        or body.get("stage") != stage
        or body.get("source_commit") != manifest["source_commit"]
        or body.get("manifest_sha256") != manifest["manifest_sha256"]
        or body.get("upstream_stage") != upstream["stage"]
        or body.get("upstream_marker_sha256") != upstream["marker_sha256"]
        or body.get("map_path") != str(path.relative_to(root))
        or body.get("map_sha256") != _unsigned_digest(body, "map_sha256")
        or len(matching) != 1
    ):
        raise ValueError("submission map authorization differs")
    normalized = list(body.get("entries", ()))
    if len({int(entry["index"]) for entry in normalized}) != len(normalized) or any(
        set(entry) != {"index", "mode", "retained_data_files", "artifact_state_sha256"}
        or not isinstance(entry["index"], int)
        or isinstance(entry["index"], bool)
        or entry["mode"] not in {"new", "verification_only"}
        or not isinstance(entry["retained_data_files"], list)
        or entry["artifact_state_sha256"] != _submission_entry_digest(entry)
        for entry in normalized
    ):
        raise ValueError("submission map entries differ")
    authorized = matching[0]
    cell = _cell(manifest, stage, int(index))
    paths = _cell_artifact_paths(root, cell, stage)
    data_paths, marker_path = paths[:-1], paths[-1]
    if action == "create":
        if authorized["mode"] == "new" and any(path.exists() for path in paths):
            raise ValueError("new submission index already has retained artifacts")
        if authorized["mode"] == "verification_only" and (
            not all(path.is_file() for path in data_paths) or marker_path.exists()
        ):
            raise ValueError("verification-only submission index changed state")
    elif not all(path.is_file() for path in data_paths) or marker_path.exists():
        raise ValueError(
            "verification action requires complete unmarked data artifacts"
        )
    if authorized["mode"] == "verification_only" and authorized[
        "retained_data_files"
    ] != _retained_data_file_records(root, cell, stage):
        raise ValueError("verification-only retained artifact digests changed")
    return body


def _validate_retained_submission_authorization(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the immutable map that authorized a cell action."""

    relative_path = str(authorization.get("map_path", ""))
    map_path = (root / relative_path).resolve(strict=True)
    submission_root = (root / "submissions").resolve(strict=True)
    if not map_path.is_relative_to(submission_root):
        raise ValueError("retained submission map is outside the output root")
    map_file_sha256 = benchmark.file_sha256(map_path)
    if map_file_sha256 != authorization.get("map_file_sha256"):
        raise ValueError("retained submission map file digest differs")
    body = read_json(map_path)
    upstream = _submission_upstream_marker(root, manifest, stage)
    matching = [
        entry
        for entry in body.get("entries", ())
        if int(entry["index"]) == int(cell["index"])
    ]
    canonical = {
        "map_path": relative_path,
        "map_file_sha256": map_file_sha256,
        "map_sha256": body.get("map_sha256"),
        "mode": str(matching[0]["mode"]) if len(matching) == 1 else None,
    }
    if (
        body.get("schema_version") != SUBMISSION_MAP_SCHEMA
        or body.get("status") != "frozen_before_submission"
        or body.get("stage") != stage
        or body.get("source_commit") != manifest["source_commit"]
        or body.get("manifest_sha256") != manifest["manifest_sha256"]
        or body.get("upstream_stage") != upstream["stage"]
        or body.get("upstream_marker_sha256") != upstream["marker_sha256"]
        or body.get("map_path") != relative_path
        or body.get("map_sha256") != _unsigned_digest(body, "map_sha256")
        or len(matching) != 1
        or matching[0].get("mode") not in {"new", "verification_only"}
        or set(matching[0])
        != {"index", "mode", "retained_data_files", "artifact_state_sha256"}
        or matching[0].get("artifact_state_sha256")
        != _submission_entry_digest(matching[0])
        or (
            matching[0].get("mode") == "verification_only"
            and matching[0].get("retained_data_files")
            != _retained_data_file_records(root, cell, stage)
        )
        or dict(authorization) != canonical
    ):
        raise ValueError("retained submission authorization differs")
    return canonical


def _require_single_gpu_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a worker exposes exactly one JAX GPU."""

    canonical = dict(runtime)
    selector = str(canonical.get("cuda_visible_devices") or "")
    if (
        canonical.get("backend") != "gpu"
        or canonical.get("device_platforms") != ["gpu"]
        or len(canonical.get("device_kinds", ())) != 1
        or not str(canonical["device_kinds"][0])
        or canonical.get("visible_device_count") != 1
        or not selector
        or "," in selector
        or canonical.get("jax_enable_x64") is not False
        or str(canonical.get("jax_enable_compilation_cache", "")).lower() != "true"
        or not str(canonical.get("jax_compilation_cache_dir") or "").startswith("/")
        or canonical.get("jax_persistent_cache_min_compile_time_secs") != "0"
        or canonical.get("jax_persistent_cache_min_entry_size_bytes") != "-1"
        or str(canonical.get("jax_raise_persistent_cache_errors", "")).lower() != "true"
    ):
        raise ValueError(
            "roadmap worker did not satisfy the frozen one-GPU/compiler runtime"
        )
    return canonical


def _linux_process_identity() -> dict[str, Any]:
    process_stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    if ")" not in process_stat:
        raise ValueError("Linux process identity is unavailable")
    # The parenthesized process name may contain spaces, so fields following it
    # must be parsed separately. Linux stat field 22 is tail index 19.
    process_tail = process_stat.rsplit(")", 1)[1].split()
    if len(process_tail) <= 19:
        raise ValueError("Linux process start identity is unavailable")
    boot_id = (
        Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    )
    if not boot_id:
        raise ValueError("Linux boot identity is unavailable")
    return {
        "node_name": os.uname().nodename,
        "boot_id": boot_id,
        "process_id": os.getpid(),
        "process_start_ticks": int(process_tail[19]),
    }


def _parse_scontrol_record(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise ValueError("Slurm allocation record contains duplicate fields")
        fields[key] = value
    return fields


def _is_single_l40s_allocation(alloc_tres: str, tres_per_node: str) -> bool:
    """Validate the GPU count from Slurm's allocation, not optional env hints."""

    entries: dict[str, str] = {}
    for token in alloc_tres.split(","):
        if "=" not in token:
            return False
        key, value = token.split("=", 1)
        if not key or key in entries:
            return False
        entries[key] = value
    gpu_entries = {
        key: value
        for key, value in entries.items()
        if key == "gres/gpu" or key.startswith("gres/gpu:")
    }
    return (
        entries.get("cpu") == "8"
        and entries.get("mem") == "64G"
        and entries.get("node") == "1"
        and entries.get("gres/gpu") == "1"
        and entries.get("gres/gpu:l40s") == "1"
        and all(value == "1" for value in gpu_entries.values())
        and tres_per_node == "gres/gpu:1"
    )


def _same_slurm_node(batch_host: str, runtime_node: str) -> bool:
    """Match Slurm's short host label to the kernel's FQDN without prefix guessing."""

    names = []
    for value in (batch_host, runtime_node):
        normalized = value.lower()
        if normalized.endswith("."):
            normalized = normalized[:-1]
        if (
            not normalized
            or not normalized.isascii()
            or any(
                not label
                or label.startswith("-")
                or label.endswith("-")
                or any(
                    not (character.isalnum() or character == "-") for character in label
                )
                for label in normalized.split(".")
            )
        ):
            return False
        names.append(normalized)
    slurm_name, kernel_name = names
    if slurm_name == kernel_name:
        return True
    if "." not in slurm_name:
        return kernel_name == slurm_name + SLURM_NODE_FQDN_SUFFIX
    if "." not in kernel_name:
        return slurm_name == kernel_name + SLURM_NODE_FQDN_SUFFIX
    return False


def _slurm_allocation_certificate(
    job_id: str,
    *,
    array_job_id: str | None = None,
    array_task_id: int | None = None,
) -> dict[str, Any]:
    """Query Slurm itself and bind this process to one allocated GPU job."""

    raw = subprocess.check_output(
        ["scontrol", "show", "job", "-o", job_id], text=True
    ).strip()
    records = [
        _parse_scontrol_record(line) for line in raw.splitlines() if line.strip()
    ]
    if array_job_id is None:
        candidates = [record for record in records if record.get("JobId") == job_id]
    else:
        candidates = [
            record
            for record in records
            if record.get("ArrayJobId") == array_job_id
            and record.get("ArrayTaskId") == str(array_task_id)
            and record.get("JobId") in {job_id, f"{array_job_id}_{array_task_id}"}
        ]
    if len(candidates) != 1:
        raise ValueError("Slurm allocation record does not identify this exact task")
    record = candidates[0]
    job_gpus = str(os.environ.get("SLURM_JOB_GPUS") or "")
    step_gpus = str(os.environ.get("SLURM_STEP_GPUS") or "")
    cuda_selector = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "")
    alloc_tres = str(record.get("AllocTRES") or "")
    tres_per_node = str(record.get("TresPerNode") or "")
    if (
        record.get("JobState") != "RUNNING"
        or not _same_slurm_node(str(record.get("BatchHost") or ""), os.uname().nodename)
        or record.get("Account") != "dep_inin_dat"
        or record.get("Partition") != "gpu-l40s"
        or not _is_single_l40s_allocation(alloc_tres, tres_per_node)
        or (job_gpus and "," in job_gpus)
        or (step_gpus and "," in step_gpus)
        or not cuda_selector
        or "," in cuda_selector
    ):
        raise ValueError("Slurm allocation is not the frozen one-GPU contract")
    body = {
        "schema_version": SLURM_ALLOCATION_SCHEMA,
        "queried_job_id": job_id,
        "record_job_id": record["JobId"],
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
        "job_state": record["JobState"],
        "batch_host": record["BatchHost"],
        "account": record["Account"],
        "partition": record["Partition"],
        "alloc_tres": alloc_tres,
        "tres_per_node": tres_per_node,
        "gres": str(record.get("Gres") or ""),
        "slurm_job_gpus": job_gpus,
        "slurm_step_gpus": step_gpus,
        "cuda_visible_devices": cuda_selector,
        "raw_record": raw,
        "canonical_record": record,
        "raw_record_sha256": benchmark.object_sha256(raw),
        "canonical_record_sha256": benchmark.object_sha256(record),
    }
    body["certificate_sha256"] = benchmark.object_sha256(body)
    return body


def _validate_slurm_allocation_certificate(
    certificate: Mapping[str, Any],
    *,
    job_id: str,
    node_name: str,
    array_job_id: str | None = None,
    array_task_id: int | None = None,
) -> dict[str, Any]:
    canonical = dict(certificate)
    expected_keys = {
        "schema_version",
        "queried_job_id",
        "record_job_id",
        "array_job_id",
        "array_task_id",
        "job_state",
        "batch_host",
        "account",
        "partition",
        "alloc_tres",
        "tres_per_node",
        "gres",
        "slurm_job_gpus",
        "slurm_step_gpus",
        "cuda_visible_devices",
        "raw_record",
        "canonical_record",
        "raw_record_sha256",
        "canonical_record_sha256",
        "certificate_sha256",
    }
    raw = canonical.get("raw_record")
    retained_record = canonical.get("canonical_record")
    if (
        not isinstance(raw, str)
        or not raw
        or raw.strip() != raw
        or not isinstance(retained_record, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in retained_record.items()
        )
        or canonical.get("raw_record_sha256") != benchmark.object_sha256(raw)
        or canonical.get("canonical_record_sha256")
        != benchmark.object_sha256(retained_record)
    ):
        raise ValueError("retained Slurm allocation certificate is invalid")
    records = [
        _parse_scontrol_record(line) for line in raw.splitlines() if line.strip()
    ]
    if array_job_id is None:
        candidates = [record for record in records if record.get("JobId") == job_id]
    else:
        candidates = [
            record
            for record in records
            if record.get("ArrayJobId") == array_job_id
            and record.get("ArrayTaskId") == str(array_task_id)
            and record.get("JobId") in {job_id, f"{array_job_id}_{array_task_id}"}
        ]
    if len(candidates) != 1 or dict(retained_record) != candidates[0]:
        raise ValueError("retained Slurm allocation certificate is invalid")
    record = candidates[0]
    if (
        set(canonical) != expected_keys
        or canonical.get("schema_version") != SLURM_ALLOCATION_SCHEMA
        or canonical.get("queried_job_id") != job_id
        or canonical.get("record_job_id")
        not in (
            {job_id}
            if array_job_id is None
            else {job_id, f"{array_job_id}_{array_task_id}"}
        )
        or canonical.get("array_job_id") != array_job_id
        or canonical.get("array_task_id") != array_task_id
        or canonical.get("record_job_id") != record.get("JobId")
        or canonical.get("job_state") != record.get("JobState")
        or canonical.get("job_state") != "RUNNING"
        or canonical.get("batch_host") != record.get("BatchHost")
        or not _same_slurm_node(str(canonical.get("batch_host") or ""), node_name)
        or canonical.get("account") != record.get("Account")
        or canonical.get("account") != "dep_inin_dat"
        or canonical.get("partition") != record.get("Partition")
        or canonical.get("partition") != "gpu-l40s"
        or canonical.get("alloc_tres") != str(record.get("AllocTRES") or "")
        or canonical.get("tres_per_node") != str(record.get("TresPerNode") or "")
        or canonical.get("gres") != str(record.get("Gres") or "")
        or not _is_single_l40s_allocation(
            str(canonical.get("alloc_tres") or ""),
            str(canonical.get("tres_per_node") or ""),
        )
        or (
            bool(canonical.get("slurm_job_gpus"))
            and "," in str(canonical.get("slurm_job_gpus"))
        )
        or (
            bool(canonical.get("slurm_step_gpus"))
            and "," in str(canonical.get("slurm_step_gpus"))
        )
        or not str(canonical.get("cuda_visible_devices") or "")
        or "," in str(canonical.get("cuda_visible_devices"))
        or canonical.get("certificate_sha256")
        != _unsigned_digest(canonical, "certificate_sha256")
    ):
        raise ValueError("retained Slurm allocation certificate is invalid")
    return canonical


def _array_scheduler_provenance(index: int) -> dict[str, Any]:
    """Capture the exact Slurm array task executing this process."""

    required = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if any(value is None or not str(value).isdigit() for value in required.values()):
        raise ValueError("roadmap cell requires numeric Slurm array provenance")
    if int(str(required["array_task_id"])) != int(index):
        raise ValueError("Slurm array task id differs from the roadmap cell index")
    process = _linux_process_identity()
    job_id = str(required["job_id"])
    array_job_id = str(required["array_job_id"])
    return {
        "job_id": job_id,
        "array_job_id": array_job_id,
        "array_task_id": int(str(required["array_task_id"])),
        **process,
        "allocation_certificate": _slurm_allocation_certificate(
            job_id,
            array_job_id=array_job_id,
            array_task_id=int(str(required["array_task_id"])),
        ),
    }


def _single_scheduler_provenance() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None or not job_id.isdigit():
        raise ValueError("roadmap stage requires numeric Slurm job provenance")
    if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        raise ValueError("roadmap single-job stage cannot execute as an array task")
    process = _linux_process_identity()
    return {
        "job_id": job_id,
        **process,
        "allocation_certificate": _slurm_allocation_certificate(job_id),
    }


def _validate_scheduler_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    canonical = {
        "job_id": str(record.get("job_id", "")),
        "array_job_id": str(record.get("array_job_id", "")),
        "array_task_id": int(record.get("array_task_id", -1)),
        "node_name": str(record.get("node_name", "")),
        "boot_id": str(record.get("boot_id", "")),
        "process_id": int(record.get("process_id", -1)),
        "process_start_ticks": int(record.get("process_start_ticks", -1)),
        "allocation_certificate": dict(record.get("allocation_certificate", {})),
    }
    if (
        dict(record) != canonical
        or not canonical["job_id"].isdigit()
        or not canonical["array_job_id"].isdigit()
        or canonical["array_task_id"] != int(index)
        or not canonical["node_name"]
        or not canonical["boot_id"]
        or canonical["process_id"] <= 0
        or canonical["process_start_ticks"] < 0
    ):
        raise ValueError("retained Slurm process provenance is invalid")
    _validate_slurm_allocation_certificate(
        canonical["allocation_certificate"],
        job_id=canonical["job_id"],
        node_name=canonical["node_name"],
        array_job_id=canonical["array_job_id"],
        array_task_id=canonical["array_task_id"],
    )
    return canonical


def _submission_receipt_path(root: Path, authorization: Mapping[str, Any]) -> Path:
    map_path = root / str(authorization["map_path"])
    return map_path.with_name(map_path.stem + ".receipt.json")


def _validate_single_submission_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    label: str,
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    """Find and authenticate the one single-job receipt for this live job."""

    job_id = str(scheduler.get("job_id", ""))
    candidates: list[tuple[Path, dict[str, Any], str]] = []
    for receipt_path in sorted(
        (root / "submissions").glob(f"{label}-attempt-*.receipt.json")
    ):
        receipt_file_sha256 = benchmark.file_sha256(receipt_path)
        receipt = read_json(receipt_path)
        if receipt.get("jobs") == {"job": job_id}:
            candidates.append((receipt_path, receipt, receipt_file_sha256))
    if len(candidates) != 1:
        raise ValueError("live single-job submission receipt is absent or duplicated")
    receipt_path, receipt, receipt_file_sha256 = candidates[0]
    relative_intent = str(receipt.get("intent_path", ""))
    intent_path = (root / relative_intent).resolve(strict=True)
    submission_root = (root / "submissions").resolve(strict=True)
    if not intent_path.is_relative_to(submission_root):
        raise ValueError("single-job submission intent is outside the output root")
    intent_file_sha256 = benchmark.file_sha256(intent_path)
    intent = read_json(intent_path)
    expected_receipt_path = intent_path.with_name(
        intent_path.name[: -len(".intent.json")] + ".receipt.json"
    )
    payload = intent.get("payload", {})
    expected_slurm_stage = {
        "preflight": "preflight",
        "calibration": "calibration",
        "finalize": "finalize",
        "diagnostic-verifier": "verify-diagnostics",
        "model-verifier": "verify-models",
        "evaluation-verifier": "verify-evaluations",
    }.get(label)
    manifest_binding_valid = (
        intent.get("manifest_sha256") in {None, manifest["manifest_sha256"]}
        if label == "preflight"
        else intent.get("manifest_sha256") == manifest["manifest_sha256"]
    )
    mode_valid = (
        "mode" not in payload
        if label == "preflight" or label.endswith("-verifier")
        else (
            payload.get("mode") in {"new", "verification_only"}
            if label == "calibration"
            else payload.get("mode") in {"new", "report_recovery"}
        )
    )
    if (
        receipt_path != expected_receipt_path
        or receipt.get("schema_version") != SUBMISSION_RECORD_SCHEMA
        or receipt.get("status") != "submitted"
        or receipt.get("label") != label
        or receipt.get("source_commit") != manifest["source_commit"]
        or receipt.get("output_root") != str(root.resolve(strict=True))
        or receipt.get("intent_file_sha256") != intent_file_sha256
        or receipt.get("dependency_policies") != {}
        or receipt.get("failure") is not None
        or receipt.get("launch_policy") != "held_until_receipt_persisted_before_release"
        or receipt.get("receipt_sha256") != _unsigned_digest(receipt, "receipt_sha256")
        or intent.get("schema_version") != SUBMISSION_RECORD_SCHEMA
        or intent.get("status") != "frozen_before_submission"
        or intent.get("label") != label
        or intent.get("source_commit") != manifest["source_commit"]
        or intent.get("output_root") != str(root.resolve(strict=True))
        or not manifest_binding_valid
        or not isinstance(payload, Mapping)
        or payload.get("slurm_stage") != expected_slurm_stage
        or not mode_valid
        or intent.get("intent_sha256") != _unsigned_digest(intent, "intent_sha256")
    ):
        raise ValueError("single-job submission receipt or intent differs")
    return {
        "intent_path": str(intent_path.relative_to(root)),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "intent_payload": dict(payload),
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "job_id": job_id,
    }


def _live_single_execution_binding(
    root: Path, manifest: Mapping[str, Any], label: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scheduler = _single_scheduler_provenance()
    receipt = _validate_single_submission_receipt(root, manifest, label, scheduler)
    runtime = _require_single_gpu_runtime(benchmark.runtime_fingerprint())
    return scheduler, receipt, runtime


def _validate_single_scheduler_record(record: Mapping[str, Any]) -> dict[str, Any]:
    canonical = {
        "job_id": str(record.get("job_id", "")),
        "node_name": str(record.get("node_name", "")),
        "boot_id": str(record.get("boot_id", "")),
        "process_id": int(record.get("process_id", -1)),
        "process_start_ticks": int(record.get("process_start_ticks", -1)),
        "allocation_certificate": dict(record.get("allocation_certificate", {})),
    }
    if (
        dict(record) != canonical
        or not canonical["job_id"].isdigit()
        or not canonical["node_name"]
        or not canonical["boot_id"]
        or canonical["process_id"] <= 0
        or canonical["process_start_ticks"] < 0
    ):
        raise ValueError("retained single-job process provenance is invalid")
    _validate_slurm_allocation_certificate(
        canonical["allocation_certificate"],
        job_id=canonical["job_id"],
        node_name=canonical["node_name"],
    )
    return canonical


def _validate_retained_single_execution_binding(
    root: Path,
    manifest: Mapping[str, Any],
    label: str,
    scheduler_record: Mapping[str, Any],
    receipt_record: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scheduler = _validate_single_scheduler_record(scheduler_record)
    receipt = _validate_single_submission_receipt(root, manifest, label, scheduler)
    runtime = _require_single_gpu_runtime(runtime_record)
    if dict(receipt_record) != receipt or dict(runtime_record) != runtime:
        raise ValueError("retained single-job execution binding differs")
    return scheduler, receipt, runtime


def _validate_array_stage_verifier_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    stage: str,
    scheduler: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Authenticate an array receipt whose afterok job is this stage verifier."""

    job_id = str(scheduler.get("job_id", ""))
    candidates: list[tuple[Path, dict[str, Any], str]] = []
    for receipt_path in sorted(
        (root / "submissions").glob(f"{stage}-attempt-*.receipt.json")
    ):
        receipt_file_sha256 = benchmark.file_sha256(receipt_path)
        receipt = read_json(receipt_path)
        if str(receipt.get("jobs", {}).get("verifier", "")) == job_id:
            candidates.append((receipt_path, receipt, receipt_file_sha256))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("stage-verifier array receipt is duplicated")
    receipt_path, receipt, receipt_file_sha256 = candidates[0]
    relative_map = str(receipt.get("intent_path", ""))
    map_path = (root / relative_map).resolve(strict=True)
    submission_root = (root / "submissions").resolve(strict=True)
    expected_receipt_path = map_path.with_name(map_path.stem + ".receipt.json")
    if not map_path.is_relative_to(submission_root):
        raise ValueError("stage-verifier map is outside the output root")
    map_file_sha256 = benchmark.file_sha256(map_path)
    submission_map = read_json(map_path)
    if (
        receipt_path != expected_receipt_path
        or receipt.get("schema_version") != SUBMISSION_RECORD_SCHEMA
        or receipt.get("status") != "submitted"
        or receipt.get("label") != stage
        or receipt.get("source_commit") != manifest["source_commit"]
        or receipt.get("output_root") != str(root.resolve(strict=True))
        or receipt.get("intent_file_sha256") != map_file_sha256
        or set(receipt.get("jobs", {})) != {"array", "verifier"}
        or receipt.get("dependency_policies")
        != {"verifier": "afterok_array_kill_on_invalid_dependency"}
        or receipt.get("failure") is not None
        or receipt.get("launch_policy") != "held_until_receipt_persisted_before_release"
        or receipt.get("receipt_sha256") != _unsigned_digest(receipt, "receipt_sha256")
        or submission_map.get("schema_version") != SUBMISSION_MAP_SCHEMA
        or submission_map.get("status") != "frozen_before_submission"
        or submission_map.get("stage") != stage
        or submission_map.get("source_commit") != manifest["source_commit"]
        or submission_map.get("manifest_sha256") != manifest["manifest_sha256"]
        or submission_map.get("map_path") != relative_map
        or submission_map.get("map_sha256")
        != _unsigned_digest(submission_map, "map_sha256")
    ):
        raise ValueError("stage-verifier array receipt or map differs")
    return {
        "kind": "array_afterok_verifier",
        "map_path": relative_map,
        "map_file_sha256": map_file_sha256,
        "map_sha256": submission_map["map_sha256"],
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "array_job_id": str(receipt["jobs"]["array"]),
        "job_id": job_id,
    }


def _live_stage_verifier_execution_binding(
    root: Path, manifest: Mapping[str, Any], stage: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scheduler = _single_scheduler_provenance()
    receipt = _validate_array_stage_verifier_receipt(root, manifest, stage, scheduler)
    if receipt is None:
        receipt = {
            "kind": "standalone_verifier",
            **_validate_single_submission_receipt(
                root, manifest, f"{stage}-verifier", scheduler
            ),
        }
    runtime = _require_single_gpu_runtime(benchmark.runtime_fingerprint())
    _require_preflight_runtime_match(root, manifest, runtime)
    return scheduler, receipt, runtime


def _validate_retained_stage_verifier_execution_binding(
    root: Path,
    manifest: Mapping[str, Any],
    stage: str,
    scheduler_record: Mapping[str, Any],
    receipt_record: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scheduler = _validate_single_scheduler_record(scheduler_record)
    kind = receipt_record.get("kind")
    if kind == "array_afterok_verifier":
        receipt = _validate_array_stage_verifier_receipt(
            root, manifest, stage, scheduler
        )
        if receipt is None:
            raise ValueError("retained array stage-verifier receipt is absent")
    elif kind == "standalone_verifier":
        receipt = {
            "kind": "standalone_verifier",
            **_validate_single_submission_receipt(
                root, manifest, f"{stage}-verifier", scheduler
            ),
        }
    else:
        raise ValueError("retained stage-verifier receipt kind is invalid")
    runtime = _require_single_gpu_runtime(runtime_record)
    _require_preflight_runtime_match(root, manifest, runtime)
    if dict(receipt_record) != receipt or dict(runtime_record) != runtime:
        raise ValueError("retained stage-verifier execution binding differs")
    return scheduler, receipt, runtime


def _validate_array_submission_receipt(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
    authorization: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind retained evidence to the immutable receipt for its live array job."""

    canonical_authorization = _validate_retained_submission_authorization(
        root, manifest, cell, stage, authorization
    )
    receipt_path = _submission_receipt_path(root, canonical_authorization)
    receipt_file_sha256 = benchmark.file_sha256(receipt_path)
    receipt = read_json(receipt_path)
    jobs = receipt.get("jobs", {})
    expected_scheduler = {
        "job_id": str(scheduler.get("job_id", "")),
        "array_job_id": str(scheduler.get("array_job_id", "")),
        "array_task_id": int(scheduler.get("array_task_id", -1)),
    }
    _validate_slurm_allocation_certificate(
        scheduler.get("allocation_certificate", {}),
        job_id=expected_scheduler["job_id"],
        node_name=str(scheduler.get("node_name", "")),
        array_job_id=expected_scheduler["array_job_id"],
        array_task_id=expected_scheduler["array_task_id"],
    )
    if (
        receipt.get("schema_version") != SUBMISSION_RECORD_SCHEMA
        or receipt.get("status") != "submitted"
        or receipt.get("label") != stage
        or receipt.get("source_commit") != manifest["source_commit"]
        or receipt.get("output_root") != str(root.resolve(strict=True))
        or receipt.get("intent_path") != canonical_authorization["map_path"]
        or receipt.get("intent_file_sha256")
        != canonical_authorization["map_file_sha256"]
        or receipt.get("dependency_policies")
        != {"verifier": "afterok_array_kill_on_invalid_dependency"}
        or receipt.get("failure") is not None
        or receipt.get("launch_policy") != "held_until_receipt_persisted_before_release"
        or set(jobs) != {"array", "verifier"}
        or any(not str(value).isdigit() for value in jobs.values())
        or str(jobs.get("array")) != expected_scheduler["array_job_id"]
        or expected_scheduler["array_task_id"] != int(cell["index"])
        or not expected_scheduler["job_id"].isdigit()
        or receipt.get("receipt_sha256") != _unsigned_digest(receipt, "receipt_sha256")
    ):
        raise ValueError("array submission receipt or live scheduler binding differs")
    return {
        "receipt_path": str(receipt_path.relative_to(root)),
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "array_job_id": str(jobs["array"]),
        "verifier_job_id": str(jobs["verifier"]),
    }


def _live_array_execution_binding(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
    authorization: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scheduler = _array_scheduler_provenance(int(cell["index"]))
    receipt = _validate_array_submission_receipt(
        root, manifest, cell, stage, authorization, scheduler
    )
    runtime = _require_single_gpu_runtime(benchmark.runtime_fingerprint())
    _require_preflight_runtime_match(root, manifest, runtime)
    return scheduler, receipt, runtime


def _validate_retained_creation_binding(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    authorization = _validate_retained_submission_authorization(
        root,
        manifest,
        cell,
        stage,
        result.get("creation_submission_authorization", {}),
    )
    scheduler = _validate_scheduler_record(
        result.get("creation_scheduler_provenance", {}), int(cell["index"])
    )
    receipt = _validate_array_submission_receipt(
        root, manifest, cell, stage, authorization, scheduler
    )
    runtime = _require_single_gpu_runtime(result.get("runtime", {}))
    _require_preflight_runtime_match(root, manifest, runtime)
    if (
        authorization["mode"] != "new"
        or result.get("creation_submission_receipt") != receipt
        or dict(result.get("runtime", {})) != runtime
    ):
        raise ValueError("retained creation execution binding differs")
    return authorization, scheduler, receipt, runtime


def _require_distinct_replay_processes(
    creation: Mapping[str, Any], verification: Mapping[str, Any]
) -> None:
    fields = ("node_name", "boot_id", "process_id", "process_start_ticks")
    if tuple(creation[field] for field in fields) == tuple(
        verification[field] for field in fields
    ):
        raise ValueError("strict replay must execute in a fresh Python process")


def _dependency_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(manifest["dependency_root"])) / "manifest.json"
    if benchmark.file_sha256(path) != manifest["dependency_manifest_file_sha256"]:
        raise ValueError("dependency manifest changed")
    return read_json(path)


def _dependency_source(
    dependency_manifest: Mapping[str, Any], world_seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in dependency_manifest["reward_sources"]
        if int(row["world_model_seed"]) == int(world_seed)
    ]
    if len(rows) != 1:
        raise ValueError("dependency reward source is absent or duplicated")
    return rows[0]


def _dependency_rebrac_cell(
    dependency_manifest: Mapping[str, Any], world_seed: int, actor_seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in dependency_manifest["rebrac_cells"]
        if int(row["world_model_seed"]) == int(world_seed)
        and int(row["actor_seed"]) == int(actor_seed)
    ]
    if len(rows) != 1:
        raise ValueError("dependency ReBRAC cell is absent or duplicated")
    return rows[0]


def _load_dependency_evaluation_trace(
    manifest: Mapping[str, Any],
    dependency_manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], Path]:
    """Authenticate dependency marker/result/bytes before loading its trace."""

    root = Path(str(manifest["dependency_root"]))
    marker = _validate_dependency_marker(root, dependency_manifest, cell, "evaluation")
    if marker.get("strict_policy_and_environment_replay") is not True:
        raise ValueError("dependency evaluation was not strict-replay authenticated")
    result_path = root / str(cell["result_path"])
    result = read_json(result_path)
    trace_path = (root / str(cell["trace_path"])).resolve(strict=True)
    trace_file_sha256 = benchmark.file_sha256(trace_path)
    if (
        marker.get("trace_file_sha256") != trace_file_sha256
        or result.get("trace_file_sha256") != trace_file_sha256
        or result.get("evaluation_seeds") != cell.get("evaluation_seeds")
    ):
        raise ValueError("dependency evaluation trace identity differs")
    trace = benchmark.load_npz(trace_path)
    if result.get("trace_sha256") != benchmark.array_sha256(trace):
        raise ValueError("dependency evaluation trace payload differs")
    return trace, trace_path


def _load_control_inputs(
    manifest: Mapping[str, Any], world_seed: int, actor_seed: int
) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    """Load digest-bound world/reward/ReBRAC inputs without old-source checks."""

    import jax
    from imf_dreamer_jax import load_checkpoint

    old = _dependency_manifest(manifest)
    root = Path(str(manifest["dependency_root"]))
    source = _dependency_source(old, world_seed)
    reward_checkpoint = Path(str(source["reward_checkpoint"])).resolve(strict=True)
    if benchmark.file_sha256(reward_checkpoint) != source["reward_checkpoint_sha256"]:
        raise ValueError("dependency reward checkpoint digest differs before load")
    reward_state, dreamer_config, metadata = load_checkpoint(
        source["reward_checkpoint"]
    )
    if (
        metadata.get("stage") != "itpo_state_action_reward"
        or "reward_transition" not in reward_state.params.world_model
        or benchmark.file_sha256(source["reward_checkpoint"])
        != source["reward_checkpoint_sha256"]
    ):
        raise ValueError("dependency reward checkpoint differs")
    rebrac_cell = _dependency_rebrac_cell(old, world_seed, actor_seed)
    rebrac_path = (root / str(rebrac_cell["checkpoint_path"])).resolve(strict=True)
    rebrac_marker = _validate_dependency_marker(root, old, rebrac_cell, "rebrac")
    if rebrac_marker.get("checkpoint_sha256") != benchmark.file_sha256(rebrac_path):
        raise ValueError("dependency ReBRAC checkpoint digest differs before load")
    rebrac_state, rebrac_config, rebrac_metadata = dependency._load_rebrac_checkpoint(
        rebrac_path
    )
    if rebrac_metadata.get("completed_updates") != old["rebrac_updates"] or not all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree_util.tree_leaves(rebrac_state)
    ):
        raise ValueError("dependency ReBRAC checkpoint differs")
    return (
        reward_state.params.world_model,
        dreamer_config,
        rebrac_state,
        rebrac_config,
        source,
    )


def _training_transitions(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the repository's reset-safe transition alignment as NumPy rows."""

    train_ids = np.asarray(arrays["train_episode_ids"], dtype=np.int64)
    observations = np.asarray(arrays["observations"][train_ids], dtype=np.float32)
    actions = np.asarray(arrays["actions"][train_ids], dtype=np.float32)
    rewards = np.asarray(arrays["rewards"][train_ids], dtype=np.float32)
    continuations = np.asarray(arrays["continuations"][train_ids], dtype=np.float32)
    valid = continuations[:, :-2] > 0.0
    states = observations[:, :-2][valid].reshape((-1, observations.shape[-1]))
    next_states = observations[:, 1:-1][valid].reshape((-1, observations.shape[-1]))
    selected_actions = actions[:, 1:-1][valid].reshape((-1, actions.shape[-1]))
    selected_rewards = rewards[:, 1:-1][valid].reshape((-1,))
    selected_continuations = continuations[:, 1:-1][valid].reshape((-1,))
    if states.shape[0] == 0:
        raise ValueError("training replay contains no valid transitions")
    return (
        states,
        selected_actions,
        selected_rewards,
        next_states,
        selected_continuations,
    )


def _actor_actions(
    actor: Any, observations: np.ndarray, *, chunk: int = 4096
) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import rebrac_actor

    function = jax.jit(rebrac_actor)
    rows = []
    for start in range(0, observations.shape[0], chunk):
        value = function(actor, jnp.asarray(observations[start : start + chunk]))
        rows.append(np.asarray(jax.device_get(value), dtype=np.float32))
    return np.concatenate(rows, axis=0)


def _actor_min_q_values(
    actor: Any, critics: Any, observations: np.ndarray, *, chunk: int = 4096
) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import rebrac_actor, rebrac_critics

    @jax.jit
    def function(states: Any) -> Any:
        actions = rebrac_actor(actor, states)
        return jnp.min(rebrac_critics(critics, states, actions), axis=0)

    rows = []
    for start in range(0, observations.shape[0], chunk):
        value = function(jnp.asarray(observations[start : start + chunk]))
        rows.append(np.asarray(jax.device_get(value), dtype=np.float32))
    return np.concatenate(rows, axis=0)


def _mixture_log_density(
    actions: np.ndarray,
    actor_actions: Sequence[np.ndarray],
    *,
    fixed_std: float,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax.control_aware_imf import (
        approximate_fixed_variance_gaussian_log_probability,
    )

    values = [
        approximate_fixed_variance_gaussian_log_probability(
            jnp.asarray(actions), jnp.asarray(mean), fixed_std=fixed_std
        )
        for mean in actor_actions
    ]
    stacked = jnp.stack(values, axis=0)
    mixture = jax.scipy.special.logsumexp(stacked, axis=0) - math.log(len(values))
    return np.asarray(jax.device_get(mixture), dtype=np.float32)


def _select_tilt_eta(log_density: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax.control_aware_imf import (
        PolicyDensityTiltConfig,
        policy_density_tilt_weights,
    )

    candidates: list[dict[str, float]] = []
    for eta in TILT_ETA_GRID:
        details = policy_density_tilt_weights(
            jnp.asarray(log_density),
            PolicyDensityTiltConfig(eta=eta),
            return_details=True,
        )
        row = {
            "eta": float(eta),
            "effective_sample_size_fraction": float(
                np.asarray(jax.device_get(details.effective_sample_size))
                / log_density.size
            ),
            "clipped_fraction": float(
                np.asarray(jax.device_get(details.clipped_fraction))
            ),
        }
        candidates.append(row)
    eligible = [
        row
        for row in candidates
        if row["effective_sample_size_fraction"] >= MINIMUM_TILT_ESS_FRACTION
        and row["clipped_fraction"] <= MAXIMUM_TILT_CLIPPED_FRACTION
    ]
    if not eligible:
        raise ValueError("no training-only policy-tilt eta passes the frozen rule")
    return max(eligible, key=lambda row: row["eta"])["eta"], candidates


def _coverage_arrays(prefix: str, calibration: Any) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_observation_mean": np.asarray(calibration.observation_mean),
        f"{prefix}_observation_scale": np.asarray(calibration.observation_scale),
        f"{prefix}_action_mean": np.asarray(calibration.action_mean),
        f"{prefix}_action_scale": np.asarray(calibration.action_scale),
        f"{prefix}_train_points": np.asarray(calibration.standardized_train_points),
        f"{prefix}_train_knn": np.asarray(
            calibration.train_leave_one_out_knn_distances
        ),
        f"{prefix}_train_density": np.asarray(
            calibration.train_leave_one_out_kernel_densities
        ),
        f"{prefix}_quantile_levels": np.asarray(calibration.quantile_levels),
        f"{prefix}_distance_quantiles": np.asarray(calibration.distance_quantiles),
        f"{prefix}_density_quantiles": np.asarray(calibration.density_quantiles),
    }


def _coverage_from_arrays(
    prefix: str, row: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> Any:
    from .actor_gap_diagnostics import CoverageCalibration

    return CoverageCalibration(
        train_sample_count=int(row["train_sample_count"]),
        observation_dim=int(row["observation_dim"]),
        action_dim=int(row["action_dim"]),
        k=int(row["k"]),
        bandwidth=float(row["bandwidth"]),
        observation_mean=tuple(arrays[f"{prefix}_observation_mean"].tolist()),
        observation_scale=tuple(arrays[f"{prefix}_observation_scale"].tolist()),
        action_mean=tuple(arrays[f"{prefix}_action_mean"].tolist()),
        action_scale=tuple(arrays[f"{prefix}_action_scale"].tolist()),
        constant_observation_dimensions=tuple(
            int(value) for value in row["constant_observation_dimensions"]
        ),
        constant_action_dimensions=tuple(
            int(value) for value in row["constant_action_dimensions"]
        ),
        standardized_train_points=tuple(
            tuple(float(value) for value in item)
            for item in arrays[f"{prefix}_train_points"]
        ),
        train_leave_one_out_knn_distances=tuple(
            float(value) for value in arrays[f"{prefix}_train_knn"]
        ),
        train_leave_one_out_kernel_densities=tuple(
            float(value) for value in arrays[f"{prefix}_train_density"]
        ),
        quantile_levels=tuple(
            float(value) for value in arrays[f"{prefix}_quantile_levels"]
        ),
        distance_quantiles=tuple(
            float(value) for value in arrays[f"{prefix}_distance_quantiles"]
        ),
        density_quantiles=tuple(
            float(value) for value in arrays[f"{prefix}_density_quantiles"]
        ),
        training_data_sha256=str(row["training_data_sha256"]),
    )


def _derive_training_only_calibration(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Re-derive calibration from digest-bound training replay and actors."""

    from .actor_gap_diagnostics import fit_coverage_calibration

    old = _dependency_manifest(manifest)
    output_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for world_seed in WORLD_SEEDS:
        source = _dependency_source(old, world_seed)
        dataset_path = Path(str(source["dataset"])).resolve(strict=True)
        if benchmark.file_sha256(dataset_path) != source["dataset_file_sha256"]:
            raise ValueError("calibration dataset bytes differ before load")
        replay = benchmark.load_npz(dataset_path)
        if benchmark.array_sha256(replay) != source["dataset_sha256"]:
            raise ValueError("calibration dataset payload differs")
        states, actions, rewards, next_states, _continuations = _training_transitions(
            replay
        )
        # Exact coverage is quadratic in rows. A deterministic 2048-row train
        # subsample is frozen here and remains large relative to observation dim.
        calibration_count = min(2048, states.shape[0])
        calibration_indices = np.linspace(
            0, states.shape[0] - 1, calibration_count, dtype=np.int64
        )
        coverage = fit_coverage_calibration(
            states[calibration_indices],
            actions[calibration_indices],
            k=min(5, calibration_count - 1),
            distance_chunk_size=256,
        )
        prefix = f"world_{world_seed}"
        output_arrays.update(_coverage_arrays(prefix, coverage))
        anchor_indices = np.linspace(
            0, states.shape[0] - 1, min(ANCHOR_COUNT, states.shape[0]), dtype=np.int64
        )
        output_arrays[f"{prefix}_anchor_observations"] = states[anchor_indices]
        for actor_seed in ACTOR_SEEDS:
            _, _, rebrac_state, _, _ = _load_control_inputs(
                manifest, world_seed, actor_seed
            )
            actor_mean = _actor_actions(rebrac_state.actor, states)
            next_value = _actor_min_q_values(
                rebrac_state.actor, rebrac_state.critics, next_states
            )
            log_density = _mixture_log_density(
                actions, [actor_mean], fixed_std=APPROXIMATE_POLICY_STD
            )
            local_eta, eta_candidates = _select_tilt_eta(log_density)
            behavior_distance = np.sqrt(
                np.mean(np.square(actions - actor_mean), axis=-1)
            )
            # This is the exact one-step target used by the deployed FlowMPC
            # interface and CVAML: state-action reward, fixed continuation one.
            real_backup = rewards + PLANNER_DISCOUNT * next_value
            actor_prefix = f"{prefix}_actor_{actor_seed}"
            output_arrays[f"{actor_prefix}_behavior_log_density"] = log_density
            output_arrays[f"{actor_prefix}_behavior_distance"] = behavior_distance
            rows.append(
                {
                    "world_model_seed": world_seed,
                    "actor_seed": actor_seed,
                    "partition": "training_replay_only",
                    "transition_count": int(states.shape[0]),
                    "local_maximum_eligible_tilt_eta": float(local_eta),
                    "tilt_eta_candidates": eta_candidates,
                    "approximate_policy_std": APPROXIMATE_POLICY_STD,
                    "density_label": "approximate_fixed_variance_gaussian_around_deterministic_action",
                    "behavior_distance_q95": float(
                        np.quantile(behavior_distance, 0.95)
                    ),
                    "bellman_target_definition": (
                        "reward_state_action_plus_planner_discount_times_next_min_q"
                    ),
                    "bellman_target_scale": float(
                        max(
                            np.subtract(*np.quantile(real_backup, [0.75, 0.25])),
                            1e-3,
                        )
                    ),
                    **coverage.metadata_dict(),
                }
            )
    global_tilt_eta = min(float(row["local_maximum_eligible_tilt_eta"]) for row in rows)
    for row in rows:
        row["selected_tilt_eta"] = global_tilt_eta
    return rows, output_arrays


def _derive_training_only_calibration_after_discarded_warmup(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Stabilize first-execution compilation before retaining calibration.

    The complete first derivation is pure: it does not touch the environment,
    output directory, or persistent model state.  Discarding it symmetrically
    in the creator and independent verifier prevents first-execution compiler
    paths from being mistaken for immutable scientific evidence.
    """

    _derive_training_only_calibration(manifest)
    return _derive_training_only_calibration(manifest)


def run_calibration(
    dependency_root: str | Path,
    output_root: str | Path,
    *,
    model_updates: int = MODEL_UPDATES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    """Fit every controller/model calibration using training replay only."""

    manifest = write_manifest(
        dependency_root,
        output_root,
        model_updates=model_updates,
        evaluation_episodes=evaluation_episodes,
    )
    root = Path(output_root)
    validate_preflight_marker(root)
    result_path = root / str(manifest["calibration"]["result_path"])
    if result_path.is_file():
        return read_json(result_path)
    _reject_partial_artifacts(
        result_path, root / str(manifest["calibration"]["arrays_path"])
    )
    creation_scheduler, creation_receipt, creation_runtime = (
        _live_single_execution_binding(root, manifest, "calibration")
    )
    _require_preflight_runtime_match(root, manifest, creation_runtime)
    if creation_receipt.get("intent_payload", {}).get("mode") != "new":
        raise ValueError("calibration result lacks a new-mode creation receipt")
    started = time.perf_counter()
    rows, output_arrays = _derive_training_only_calibration_after_discarded_warmup(
        manifest
    )
    global_tilt_eta = min(float(row["selected_tilt_eta"]) for row in rows)
    arrays_path = root / str(manifest["calibration"]["arrays_path"])
    if arrays_path.parent != result_path.parent:
        raise ValueError("calibration artifacts do not share one bundle directory")
    with _staged_artifact_directory(result_path.parent) as staging:
        staged_arrays_path = staging / arrays_path.name
        staged_result_path = staging / result_path.name
        _write_npz_exclusive(staged_arrays_path, output_arrays)
        result = {
            "schema_version": CALIBRATION_SCHEMA,
            "status": "complete",
            "source_commit": manifest["source_commit"],
            "manifest_sha256": manifest["manifest_sha256"],
            "task": TASK,
            "evaluation_seeds_accessed": False,
            "discarded_full_calibration_warmup": True,
            "selected_global_tilt_eta": global_tilt_eta,
            "rows": rows,
            "arrays_file_sha256": benchmark.file_sha256(staged_arrays_path),
            "arrays_sha256": benchmark.array_sha256(output_arrays),
            "creation_scheduler_provenance": creation_scheduler,
            "creation_submission_receipt": creation_receipt,
            "wall_seconds": time.perf_counter() - started,
            "runtime": creation_runtime,
        }
        if not _finite_tree(result):
            raise FloatingPointError("calibration result is not finite")
        _write_json_exclusive(staged_result_path, result)
    return result


def verify_calibration(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    validate_preflight_marker(root)
    result_path = root / str(manifest["calibration"]["result_path"])
    arrays_path = root / str(manifest["calibration"]["arrays_path"])
    result = read_json(result_path)
    creation_scheduler, creation_receipt, creation_runtime = (
        _validate_retained_single_execution_binding(
            root,
            manifest,
            "calibration",
            result.get("creation_scheduler_provenance", {}),
            result.get("creation_submission_receipt", {}),
            result.get("runtime", {}),
        )
    )
    _require_preflight_runtime_match(root, manifest, creation_runtime)
    if creation_receipt.get("intent_payload", {}).get("mode") != "new":
        raise ValueError("calibration result lacks a new-mode creation receipt")
    arrays_file_sha256 = benchmark.file_sha256(arrays_path)
    if (
        result.get("schema_version") != CALIBRATION_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("evaluation_seeds_accessed") is not False
        or len(result.get("rows", ())) != len(WORLD_SEEDS) * len(ACTOR_SEEDS)
        or result.get("arrays_file_sha256") != arrays_file_sha256
        or not _finite_tree(result)
    ):
        raise ValueError("calibration result is invalid")
    # Authenticate the immutable bytes before deserializing the NumPy archive.
    arrays = benchmark.load_npz(arrays_path)
    if result.get("arrays_sha256") != benchmark.array_sha256(arrays):
        raise ValueError("calibration array payload is invalid")
    expected_identities = [
        (world_seed, actor_seed)
        for world_seed in WORLD_SEEDS
        for actor_seed in ACTOR_SEEDS
    ]
    observed_identities = [
        (int(row["world_model_seed"]), int(row["actor_seed"])) for row in result["rows"]
    ]
    if observed_identities != expected_identities:
        raise ValueError("calibration rows are missing, duplicated, or reordered")
    expected_array_names: set[str] = set()
    for world_seed in WORLD_SEEDS:
        prefix = f"world_{world_seed}"
        expected_array_names.update(
            {
                f"{prefix}_{suffix}"
                for suffix in (
                    "observation_mean",
                    "observation_scale",
                    "action_mean",
                    "action_scale",
                    "train_points",
                    "train_knn",
                    "train_density",
                    "quantile_levels",
                    "distance_quantiles",
                    "density_quantiles",
                    "anchor_observations",
                )
            }
        )
        for actor_seed in ACTOR_SEEDS:
            actor_prefix = f"{prefix}_actor_{actor_seed}"
            expected_array_names.update(
                {
                    f"{actor_prefix}_behavior_log_density",
                    f"{actor_prefix}_behavior_distance",
                }
            )
    if set(arrays) != expected_array_names or not _finite_tree(arrays):
        raise ValueError("calibration arrays are incomplete or non-finite")
    if result.get("discarded_full_calibration_warmup") is not True:
        raise ValueError("calibration compiler warmup certificate is absent")
    expected_rows, expected_arrays = (
        _derive_training_only_calibration_after_discarded_warmup(manifest)
    )
    if (
        benchmark.object_sha256(expected_rows)
        != benchmark.object_sha256(result["rows"])
        or benchmark.array_sha256(expected_arrays) != result["arrays_sha256"]
    ):
        raise ValueError(
            "calibration does not rederive from authenticated training data"
        )
    for row in result["rows"]:
        prefix = f"world_{int(row['world_model_seed'])}"
        actor_prefix = f"{prefix}_actor_{int(row['actor_seed'])}"
        calibration = _coverage_from_arrays(prefix, row, arrays)
        if (
            calibration.metadata_dict()["training_data_sha256"]
            != row["training_data_sha256"]
        ):
            raise ValueError("coverage calibration round trip differs")
        selected, candidates = _select_tilt_eta(
            np.asarray(arrays[f"{actor_prefix}_behavior_log_density"])
        )
        candidate_values = np.asarray(
            [
                (
                    item["eta"],
                    item["effective_sample_size_fraction"],
                    item["clipped_fraction"],
                )
                for item in candidates
            ],
            dtype=np.float64,
        )
        retained_values = np.asarray(
            [
                (
                    item["eta"],
                    item["effective_sample_size_fraction"],
                    item["clipped_fraction"],
                )
                for item in row["tilt_eta_candidates"]
            ],
            dtype=np.float64,
        )
        distances = np.asarray(arrays[f"{actor_prefix}_behavior_distance"])
        if (
            row.get("partition") != "training_replay_only"
            or row.get("density_label")
            != "approximate_fixed_variance_gaussian_around_deterministic_action"
            or row.get("bellman_target_definition")
            != "reward_state_action_plus_planner_discount_times_next_min_q"
            or row.get("approximate_policy_std") != APPROXIMATE_POLICY_STD
            or not math.isclose(
                float(row["local_maximum_eligible_tilt_eta"]),
                float(selected),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or candidate_values.shape != retained_values.shape
            or not np.allclose(candidate_values, retained_values, rtol=1e-6, atol=1e-7)
            or not math.isclose(
                float(row["behavior_distance_q95"]),
                float(np.quantile(distances, 0.95)),
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            or float(row["bellman_target_scale"]) <= 0.0
            or row.get("selected_tilt_eta") != result.get("selected_global_tilt_eta")
        ):
            raise ValueError("policy-tilt eta is not globally frozen")
    if result.get("selected_global_tilt_eta") != min(
        float(row["local_maximum_eligible_tilt_eta"]) for row in result["rows"]
    ):
        raise ValueError("global policy-tilt eta is not conservative")
    marker_path = root / str(manifest["calibration"]["marker_path"])
    if marker_path.is_file():
        return _validate_calibration_marker(root, manifest)
    verification_scheduler, verification_receipt, verification_runtime = (
        _live_single_execution_binding(root, manifest, "calibration")
    )
    _require_preflight_runtime_match(root, manifest, verification_runtime)
    _require_distinct_replay_processes(creation_scheduler, verification_scheduler)
    if benchmark.runtime_homogeneity_identity(
        creation_runtime
    ) != benchmark.runtime_homogeneity_identity(verification_runtime):
        raise ValueError("calibration creation and replay runtimes differ")
    verification_mode = verification_receipt.get("intent_payload", {}).get("mode")
    if verification_mode not in {"new", "verification_only"} or (
        verification_mode == "new" and verification_receipt != creation_receipt
    ):
        raise ValueError("calibration verification receipt mode differs")
    marker = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": "calibration",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "result_file_sha256": benchmark.file_sha256(result_path),
        "arrays_file_sha256": benchmark.file_sha256(arrays_path),
        "training_only": True,
        "selected_global_tilt_eta": result["selected_global_tilt_eta"],
        "calibration_rows": len(result["rows"]),
        "verification_scheduler_provenance": verification_scheduler,
        "verification_submission_receipt": verification_receipt,
        "verification_runtime": verification_runtime,
    }
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    _write_json_exclusive(marker_path, marker)
    return marker


def _validate_calibration_marker(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the retained rederivation certificate without repeating it."""

    marker_path = root / str(manifest["calibration"]["marker_path"])
    result_path = root / str(manifest["calibration"]["result_path"])
    arrays_path = root / str(manifest["calibration"]["arrays_path"])
    marker = read_json(marker_path)
    if (
        marker.get("schema_version") != MARKER_SCHEMA
        or marker.get("status") != "verified"
        or marker.get("stage") != "calibration"
        or marker.get("source_commit") != manifest["source_commit"]
        or marker.get("manifest_sha256") != manifest["manifest_sha256"]
        or marker.get("result_file_sha256") != benchmark.file_sha256(result_path)
        or marker.get("arrays_file_sha256") != benchmark.file_sha256(arrays_path)
        or marker.get("training_only") is not True
        or marker.get("calibration_rows") != len(WORLD_SEEDS) * len(ACTOR_SEEDS)
        or marker.get("marker_sha256") != _unsigned_digest(marker, "marker_sha256")
    ):
        raise ValueError("calibration marker is invalid")
    result = read_json(result_path)
    creation_scheduler, creation_receipt, creation_runtime = (
        _validate_retained_single_execution_binding(
            root,
            manifest,
            "calibration",
            result.get("creation_scheduler_provenance", {}),
            result.get("creation_submission_receipt", {}),
            result.get("runtime", {}),
        )
    )
    verification_scheduler, verification_receipt, verification_runtime = (
        _validate_retained_single_execution_binding(
            root,
            manifest,
            "calibration",
            marker.get("verification_scheduler_provenance", {}),
            marker.get("verification_submission_receipt", {}),
            marker.get("verification_runtime", {}),
        )
    )
    _require_distinct_replay_processes(creation_scheduler, verification_scheduler)
    _require_preflight_runtime_match(root, manifest, creation_runtime)
    _require_preflight_runtime_match(root, manifest, verification_runtime)
    verification_mode = verification_receipt.get("intent_payload", {}).get("mode")
    if (
        result.get("schema_version") != CALIBRATION_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("discarded_full_calibration_warmup") is not True
        or result.get("arrays_file_sha256") != marker["arrays_file_sha256"]
        or result.get("selected_global_tilt_eta") != marker["selected_global_tilt_eta"]
        or creation_receipt.get("intent_payload", {}).get("mode") != "new"
        or verification_mode not in {"new", "verification_only"}
        or (verification_mode == "new" and verification_receipt != creation_receipt)
        or marker.get("verification_submission_receipt") != verification_receipt
        or benchmark.runtime_homogeneity_identity(creation_runtime)
        != benchmark.runtime_homogeneity_identity(verification_runtime)
    ):
        raise ValueError("calibration result certificate is invalid")
    return marker


def _calibration_row(
    output_root: str | Path,
    manifest: Mapping[str, Any],
    world_seed: int,
    actor_seed: int,
) -> Mapping[str, Any]:
    result = read_json(Path(output_root) / str(manifest["calibration"]["result_path"]))
    rows = [
        row
        for row in result["rows"]
        if int(row["world_model_seed"]) == int(world_seed)
        and int(row["actor_seed"]) == int(actor_seed)
    ]
    if len(rows) != 1:
        raise ValueError("calibration world row is absent or duplicated")
    return rows[0]


def _write_verified_marker(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
    *,
    result_path: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": stage,
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(result_path),
    }
    if extra:
        marker.update(dict(extra))
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / str(cell["marker_path"])
    if marker_path.is_file():
        if read_json(marker_path) != marker:
            raise ValueError(f"existing {stage} marker differs")
        return marker
    _write_json_exclusive(marker_path, marker)
    return marker


def _assert_trace_subset_close(
    retained: Mapping[str, np.ndarray],
    regenerated: Mapping[str, np.ndarray],
    *,
    prefix: str,
    episodes: int,
) -> None:
    for name in ("actions", "rewards", "continuations", "is_last", "lengths"):
        left = np.asarray(retained[f"{prefix}_{name}"])[:episodes]
        right = np.asarray(regenerated[name])[:episodes]
        if name != "lengths":
            maximum_length = int(np.max(right.shape[1:2], initial=0))
            left = left[:, :maximum_length]
        equal = left.dtype == right.dtype and np.array_equal(left, right)
        if not equal:
            maximum = (
                float(
                    np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
                )
                if left.shape == right.shape and left.size
                else math.inf
            )
            raise ValueError(f"A0 dependency replay differs for {name}: {maximum}")


def _replay_observations(
    evaluation_seeds: Sequence[int], trace: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Recreate current/next observations while proving environment replay."""

    from .dmc import DMCAdapter

    actions = np.asarray(trace["actions"], dtype=np.float32)
    lengths = np.asarray(trace["lengths"], dtype=np.int64)
    observations = np.zeros(
        (len(evaluation_seeds), actions.shape[1], 6), dtype=np.float32
    )
    next_observations = np.zeros_like(observations)
    for episode, environment_seed in enumerate(evaluation_seeds):
        environment = DMCAdapter(TASK, seed=int(environment_seed), action_repeat=1)
        try:
            observation = environment.reset()
            for step in range(int(lengths[episode])):
                observations[episode, step] = observation
                transition = environment.step(actions[episode, step])
                if not (
                    transition.reward == trace["rewards"][episode, step]
                    and transition.continuation == trace["continuations"][episode, step]
                    and bool(transition.is_last)
                    == bool(trace["is_last"][episode, step])
                ):
                    raise ValueError(
                        "environment replay differs from retained A0 trace"
                    )
                next_observations[episode, step] = transition.observation
                observation = transition.observation
        finally:
            environment.close()
    return observations, next_observations


def _flowmpc_config(rebrac_config: Any) -> Any:
    from imf_dreamer_jax import FlowMPCConfig

    return FlowMPCConfig(
        horizon=CONTROLLER_HORIZON,
        particles=FLOWMPC_PARTICLES,
        inner_steps=1,
        step_size=CONTROLLER_STEP_SIZE,
        discount=float(rebrac_config.discount),
    )


def _one_step_prediction_rows(
    world_model: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    observations: np.ndarray,
    next_observations: np.ndarray,
    trace: Mapping[str, np.ndarray],
    a0_beliefs: Sequence[Sequence[Any]],
    *,
    world_seed: int,
    actor_seed: int,
    maximum_rows: int = 64,
) -> dict[str, np.ndarray]:
    """Generate component and horizon inputs from frozen on-policy prefixes."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        RSSMState,
        decode,
        rebrac_actor,
        rebrac_critics,
        sample_prior,
    )
    from imf_dreamer_jax.world_model import (
        predict_continuation_logits,
        predict_state_action_reward_from_observation,
        transition_deterministic,
    )

    actions = np.asarray(trace["actions"], dtype=np.float32)
    lengths = np.asarray(trace["lengths"], dtype=np.int64)
    target_rewards = np.asarray(trace["rewards"], dtype=np.float64)
    target_continuations = np.asarray(trace["continuations"], dtype=np.float64)
    selected: list[tuple[int, int]] = []
    for episode, length in enumerate(lengths):
        selected.extend((episode, step) for step in range(int(length)))
    if len(selected) > maximum_rows:
        indices = np.linspace(0, len(selected) - 1, maximum_rows, dtype=np.int64)
        selected = [selected[int(index)] for index in indices]
    selected_set = set(selected)

    predicted_rewards: list[float] = []
    predicted_continuations: list[float] = []
    predicted_values: list[float] = []
    target_values: list[float] = []
    output_target_rewards: list[float] = []
    output_target_continuations: list[float] = []
    horizon_rows: list[tuple[int, float, float, float]] = []

    if len(a0_beliefs) != len(lengths):
        raise ValueError("captured A0 belief episodes do not match the trace")
    for episode in range(len(lengths)):
        beliefs = list(a0_beliefs[episode])
        if len(beliefs) != int(lengths[episode]):
            raise ValueError("captured A0 beliefs do not match the episode length")

        for step, belief in enumerate(beliefs):
            if (episode, step) not in selected_set:
                continue
            particle_count = FLOWMPC_PARTICLES
            action = jnp.broadcast_to(
                jnp.asarray(actions[episode, step]),
                (particle_count, dreamer_config.action_dim),
            )
            current_observation = jnp.asarray(observations[episode, step][None])
            reward = predict_state_action_reward_from_observation(
                world_model["reward_transition"],
                current_observation,
                action[:1],
                dreamer_config,
            )[0]
            particle_belief = RSSMState(
                jnp.broadcast_to(
                    belief.deterministic,
                    (particle_count, dreamer_config.deterministic_dim),
                ),
                jnp.broadcast_to(
                    belief.stochastic,
                    (particle_count, dreamer_config.stochastic_dim),
                ),
            )
            deterministic = transition_deterministic(
                world_model, particle_belief, action, dreamer_config
            )
            planner_noise_key = benchmark.derive_jax_key(
                "flowmpc-noise",
                TASK,
                world_seed,
                actor_seed,
                int(np.asarray(trace["evaluation_seeds"])[episode]),
            )
            noise = jax.random.normal(
                jax.random.fold_in(planner_noise_key, step),
                (
                    particle_count,
                    CONTROLLER_HORIZON,
                    dreamer_config.stochastic_dim,
                ),
            )[:, 0]
            stochastic, _ = sample_prior(
                world_model, deterministic, None, dreamer_config, noise=noise
            )
            predicted_state = RSSMState(deterministic, stochastic)
            feature = predicted_state.feature
            predicted_observation = decode(world_model, feature, dreamer_config)
            predicted_action = rebrac_actor(rebrac_state.actor, predicted_observation)
            predicted_value = jnp.min(
                rebrac_critics(
                    rebrac_state.critics, predicted_observation, predicted_action
                ),
                axis=0,
            )
            actual_next = jnp.asarray(next_observations[episode, step][None])
            target_action = rebrac_actor(rebrac_state.actor, actual_next)
            target_value = jnp.min(
                rebrac_critics(rebrac_state.critics, actual_next, target_action),
                axis=0,
            )[0]
            predicted_rewards.append(float(reward))
            predicted_continuations.append(
                float(
                    jnp.mean(
                        jax.nn.sigmoid(
                            predict_continuation_logits(world_model, feature)
                        )
                    )
                )
            )
            predicted_values.append(float(jnp.mean(predicted_value)))
            target_values.append(float(target_value))
            output_target_rewards.append(float(target_rewards[episode, step]))
            output_target_continuations.append(
                float(target_continuations[episode, step])
            )

        for start in range(int(lengths[episode])):
            for horizon in (1, 3, 5):
                if start + horizon > int(lengths[episode]):
                    continue
                particle_count = 16
                state = RSSMState(
                    jnp.broadcast_to(
                        beliefs[start].deterministic,
                        (particle_count, dreamer_config.deterministic_dim),
                    ),
                    jnp.broadcast_to(
                        beliefs[start].stochastic,
                        (particle_count, dreamer_config.stochastic_dim),
                    ),
                )
                observed = jnp.broadcast_to(
                    jnp.asarray(observations[episode, start]),
                    (particle_count, *dreamer_config.observation_shape),
                )
                predicted_return = jnp.zeros((particle_count,), jnp.float32)
                noise_key = benchmark.derive_jax_key(
                    "actor-gap-horizon-noise",
                    world_seed,
                    actor_seed,
                    episode,
                    start,
                    horizon,
                )
                noises = jax.random.normal(
                    noise_key,
                    (particle_count, horizon, dreamer_config.stochastic_dim),
                )
                for offset in range(horizon):
                    action = jnp.broadcast_to(
                        jnp.asarray(actions[episode, start + offset]),
                        (particle_count, dreamer_config.action_dim),
                    )
                    reward = predict_state_action_reward_from_observation(
                        world_model["reward_transition"],
                        observed,
                        action,
                        dreamer_config,
                    )
                    predicted_return += (
                        float(rebrac_config.discount) ** offset
                    ) * reward
                    deterministic = transition_deterministic(
                        world_model, state, action, dreamer_config
                    )
                    stochastic, _ = sample_prior(
                        world_model,
                        deterministic,
                        None,
                        dreamer_config,
                        noise=noises[:, offset],
                    )
                    state = RSSMState(deterministic, stochastic)
                    observed = decode(world_model, state.feature, dreamer_config)
                target_observation = next_observations[episode, start + horizon - 1]
                observation_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.asarray(jax.device_get(jnp.mean(observed, axis=0)))
                                - target_observation
                            )
                        )
                    )
                )
                real_return = float(
                    sum(
                        (float(rebrac_config.discount) ** offset)
                        * target_rewards[episode, start + offset]
                        for offset in range(horizon)
                    )
                )
                horizon_rows.append(
                    (
                        horizon,
                        observation_rmse,
                        abs(float(jnp.mean(predicted_return)) - real_return),
                        float(jnp.std(predicted_return)),
                    )
                )
    horizon_array = np.asarray(horizon_rows, dtype=np.float64)
    return {
        "predicted_rewards": np.asarray(predicted_rewards, dtype=np.float64),
        "target_rewards": np.asarray(output_target_rewards, dtype=np.float64),
        "predicted_continuations": np.asarray(
            predicted_continuations, dtype=np.float64
        ),
        "target_continuations": np.asarray(
            output_target_continuations, dtype=np.float64
        ),
        "predicted_next_values": np.asarray(predicted_values, dtype=np.float64),
        "target_next_values": np.asarray(target_values, dtype=np.float64),
        "horizons": horizon_array[:, 0].astype(np.int32),
        "horizon_observation_rmse": horizon_array[:, 1],
        "horizon_reward_absolute_error": horizon_array[:, 2],
        "horizon_predictive_reward_sd": horizon_array[:, 3],
    }


def _flatten_jax_tree(tree: Any) -> np.ndarray:
    import jax

    leaves = [
        np.asarray(jax.device_get(value), dtype=np.float64).reshape(-1)
        for value in jax.tree_util.tree_leaves(tree)
    ]
    return np.concatenate(leaves, axis=0)


def _imagined_action_sequence_points(
    world_model: Any,
    dreamer_config: Any,
    belief: Any,
    current_observation: Any,
    action_sequences: Any,
    noises: Any,
) -> tuple[Any, Any]:
    """Return every stage-reward query pair for fixed action sequences.

    The output axes are ``[candidate, particle, horizon, ...]``.  Using the
    same supplied particle bank as the planner makes proposal, selected-best,
    and frozen-reference coverage directly comparable without adding another
    source of stochastic variation.
    """

    import jax.numpy as jnp
    from imf_dreamer_jax import RSSMState, decode
    from imf_dreamer_jax.world_model import sample_prior, transition_deterministic

    sequences = jnp.asarray(action_sequences)
    particle_noises = jnp.asarray(noises)
    if sequences.ndim != 3:
        raise ValueError("action_sequences must have [candidate, horizon, action]")
    if particle_noises.ndim != 3:
        raise ValueError("noises must have [particle, horizon, stochastic]")
    if sequences.shape[1] != particle_noises.shape[1]:
        raise ValueError("action sequences and noises must share horizon")
    candidates, horizon, _ = sequences.shape
    particles = particle_noises.shape[0]
    batch = candidates * particles
    state = RSSMState(
        jnp.broadcast_to(
            belief.deterministic,
            (candidates, particles, dreamer_config.deterministic_dim),
        ).reshape(batch, dreamer_config.deterministic_dim),
        jnp.broadcast_to(
            belief.stochastic,
            (candidates, particles, dreamer_config.stochastic_dim),
        ).reshape(batch, dreamer_config.stochastic_dim),
    )
    observation = jnp.broadcast_to(
        current_observation,
        (candidates, particles, *dreamer_config.observation_shape),
    ).reshape(batch, *dreamer_config.observation_shape)
    observation_rows = []
    action_rows = []
    for step in range(horizon):
        actions = jnp.broadcast_to(
            sequences[:, None, step],
            (candidates, particles, dreamer_config.action_dim),
        ).reshape(batch, dreamer_config.action_dim)
        observation_rows.append(
            observation.reshape(
                candidates, particles, *dreamer_config.observation_shape
            )
        )
        action_rows.append(actions.reshape(candidates, particles, -1))
        deterministic = transition_deterministic(
            world_model, state, actions, dreamer_config
        )
        step_noises = jnp.broadcast_to(
            particle_noises[None, :, step],
            (candidates, particles, dreamer_config.stochastic_dim),
        ).reshape(batch, dreamer_config.stochastic_dim)
        stochastic, _ = sample_prior(
            world_model,
            deterministic,
            None,
            dreamer_config,
            noise=step_noises,
        )
        state = RSSMState(deterministic, stochastic)
        observation = decode(world_model, state.feature, dreamer_config)
    return jnp.stack(observation_rows, axis=2), jnp.stack(action_rows, axis=2)


def _validated_a0_noise_contexts(
    trace: Mapping[str, np.ndarray],
    contexts: Sequence[Mapping[str, Any]],
    chosen_steps: Sequence[int],
    *,
    world_seed: int,
    actor_seed: int,
    evaluation_seed: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Bind ephemeral live controller contexts to exact retained A0 actions."""

    contexts_by_step: dict[int, Mapping[str, Any]] = {}
    for context in contexts:
        step = int(context.get("step", -1))
        if step in contexts_by_step:
            raise ValueError(f"captured A0 context duplicates step {step}")
        contexts_by_step[step] = context
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for step_value in chosen_steps:
        step = int(step_value)
        if step not in contexts_by_step:
            raise ValueError(f"captured A0 context is missing step {step}")
        context = contexts_by_step[step]
        if (
            int(context.get("world_model_seed", -1)) != world_seed
            or int(context.get("actor_seed", -1)) != actor_seed
            or int(context.get("evaluation_seed", -1)) != evaluation_seed
        ):
            raise ValueError(f"captured A0 context identity differs at step {step}")
        retained_action = np.asarray(trace["actions"][0, step], dtype=np.float32)
        live_action = np.asarray(context["host_action"], dtype=np.float32)
        if not np.array_equal(live_action, retained_action):
            maximum = (
                float(
                    np.max(
                        np.abs(
                            live_action.astype(np.float64)
                            - retained_action.astype(np.float64)
                        )
                    )
                )
                if live_action.shape == retained_action.shape and live_action.size
                else math.inf
            )
            raise ValueError(
                "captured persistent FlowMPC action differs from A0 trace at "
                f"step {step}: max_abs={maximum}"
            )
        selected.append((step, context))
    return selected


def _independent_noise_result(
    world_model: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    trace: Mapping[str, np.ndarray],
    a0_contexts: Sequence[Mapping[str, Any]],
    *,
    world_seed: int,
    actor_seed: int,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import jit_flowmpc_objective
    from .actor_gap_diagnostics import independent_noise_diagnostics

    config = _flowmpc_config(rebrac_config)
    length = int(np.asarray(trace["lengths"])[0])
    if length <= 0:
        raise ValueError("independent-noise diagnostic requires a nonempty episode")
    # These checkpoints cover the exact short-horizon regime in which the h15
    # ordering drift was observed, while retaining the actual persistent A0
    # actor rather than restarting from the offline policy at each row.
    chosen = np.asarray(
        sorted({min(step, length - 1) for step in (0, 1, 5, 15)}),
        dtype=np.int64,
    )
    evaluation_seed = int(np.asarray(trace["evaluation_seeds"])[0])
    proposal_key = benchmark.derive_jax_key(
        "flowmpc-noise", TASK, world_seed, actor_seed, evaluation_seed
    )
    heldout_key = benchmark.derive_jax_key(
        "actor-gap-independent-noise",
        TASK,
        world_seed,
        actor_seed,
        evaluation_seed,
    )
    proposal_objectives: list[float] = []
    heldout_objectives: list[float] = []
    proposal_gradients: list[np.ndarray] = []
    heldout_gradients: list[np.ndarray] = []
    proposal_ids: list[str] = []
    heldout_ids: list[str] = []
    context_rows: list[dict[str, Any]] = []
    selected_contexts = _validated_a0_noise_contexts(
        trace,
        a0_contexts,
        chosen,
        world_seed=world_seed,
        actor_seed=actor_seed,
        evaluation_seed=evaluation_seed,
    )
    for step, context in selected_contexts:
        belief = context["belief"]
        current = context["current_observation"]
        adaptation_start = context["adaptation_start_actor"]
        updated_actor = context["updated_actor"]
        proposal_noise = context["proposal_noise"]
        shape = (
            config.particles,
            config.horizon,
            dreamer_config.stochastic_dim,
        )
        regenerated_proposal_noise = jax.random.normal(
            jax.random.fold_in(proposal_key, step), shape
        )
        if not np.array_equal(
            np.asarray(jax.device_get(proposal_noise), dtype=np.float32),
            np.asarray(jax.device_get(regenerated_proposal_noise), dtype=np.float32),
        ):
            raise ValueError(f"captured FlowMPC proposal noise differs at step {step}")
        heldout_noise = jax.random.normal(jax.random.fold_in(heldout_key, step), shape)

        def objective(actor: Any, noises: Any) -> Any:
            return jit_flowmpc_objective(
                actor,
                rebrac_state.critics,
                world_model,
                belief,
                current,
                noises,
                dreamer_config,
                rebrac_config,
                config,
            ).objective

        proposal_before, proposal_gradient = jax.value_and_grad(objective)(
            adaptation_start, proposal_noise
        )
        proposal_after = objective(updated_actor, proposal_noise)
        heldout_before, heldout_gradient = jax.value_and_grad(objective)(
            adaptation_start, heldout_noise
        )
        heldout_after = objective(updated_actor, heldout_noise)
        proposal_objectives.append(float(proposal_after - proposal_before))
        heldout_objectives.append(float(heldout_after - heldout_before))
        proposal_gradients.append(_flatten_jax_tree(proposal_gradient))
        heldout_gradients.append(_flatten_jax_tree(heldout_gradient))
        proposal_ids.append(
            f"flowmpc-proposal-{world_seed}-{actor_seed}-{evaluation_seed}-{step}"
        )
        heldout_ids.append(
            f"heldout-{world_seed}-{actor_seed}-{evaluation_seed}-{step}"
        )
        context_rows.append(
            {
                "step": step,
                "belief_sha256": benchmark._tree_digest(belief),
                "adaptation_start_actor_sha256": benchmark._tree_digest(
                    adaptation_start
                ),
                "updated_actor_sha256": benchmark._tree_digest(updated_actor),
                "proposal_noise_sha256": benchmark.array_sha256(
                    {
                        "proposal_noise": np.asarray(
                            jax.device_get(proposal_noise), dtype=np.float32
                        )
                    }
                ),
            }
        )
    result = independent_noise_diagnostics(
        proposal_objectives,
        heldout_objectives,
        np.stack(proposal_gradients),
        np.stack(heldout_gradients),
        proposal_noise_ids=proposal_ids,
        heldout_noise_ids=heldout_ids,
    ).to_dict()
    result.update(
        {
            "controller_state": "exact_persistent_a0_prefix",
            "belief_key_namespace": "flowmpc-posterior",
            "proposal_noise_key_namespace": "flowmpc-noise",
            "heldout_noise_key_namespace": "actor-gap-independent-noise",
            "evaluated_steps": chosen.tolist(),
            "persistent_controller_action_replay_verified": True,
            "controller_context_source": "captured_during_exact_live_a0_rollout",
            "controller_contexts": context_rows,
        }
    )
    return result


def _exact_counterfactual_result(
    world_model: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    observations: np.ndarray,
    trace: Mapping[str, np.ndarray],
    a0_beliefs: Sequence[Any],
    evaluation_seed: int,
    *,
    world_seed: int,
    actor_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        rebrac_actor,
        rebrac_critics,
    )
    from imf_dreamer_jax.world_model import predict_state_action_reward_from_observation
    from .actor_gap_diagnostics import (
        CounterfactualProvenance,
        counterfactual_candidate_diagnostics,
    )
    from .dmc import DMCAdapter

    length = int(np.asarray(trace["lengths"])[0])
    selected_steps = np.linspace(0, max(length - 1, 0), min(8, length), dtype=np.int64)
    selected = set(int(value) for value in selected_steps)
    predicted_rows: list[list[float]] = []
    simulator_rows: list[list[float]] = []
    reset_ids: list[str] = []
    horizon = CONTROLLER_HORIZON
    eligible_horizon_starts = max(0, length - horizon + 1)
    horizon_steps = (
        np.linspace(
            0,
            eligible_horizon_starts - 1,
            min(6, eligible_horizon_starts),
            dtype=np.int64,
        )
        if eligible_horizon_starts
        else np.asarray([], dtype=np.int64)
    )
    horizon_selected = set(int(value) for value in horizon_steps)
    predicted_horizon_rows: list[list[float]] = []
    simulator_horizon_rows: list[list[float]] = []
    horizon_reset_ids: list[str] = []
    imagined_proposal_observations: list[np.ndarray] = []
    imagined_proposal_actions: list[np.ndarray] = []
    imagined_best_observations: list[np.ndarray] = []
    imagined_best_actions: list[np.ndarray] = []
    imagined_reference_observations: list[np.ndarray] = []
    imagined_reference_actions: list[np.ndarray] = []
    if len(a0_beliefs) != length:
        raise ValueError("captured A0 beliefs do not match counterfactual episode")
    planner_noise_key = benchmark.derive_jax_key(
        "actor-gap-exact-h5-planner-noise",
        TASK,
        world_seed,
        actor_seed,
        evaluation_seed,
    )
    planner_objective = jax.jit(
        lambda local_belief, local_observation, action_sequence, local_noises: (
            _recursive_action_sequence_objective(
                world_model,
                dreamer_config,
                rebrac_state,
                rebrac_config,
                local_belief,
                local_observation,
                action_sequence,
                local_noises,
            )
        )
    )
    imagine_points = jax.jit(
        lambda local_belief, local_observation, action_sequences, local_noises: (
            _imagined_action_sequence_points(
                world_model,
                dreamer_config,
                local_belief,
                local_observation,
                action_sequences,
                local_noises,
            )
        )
    )

    def simulator_planner_objective(
        environment: Any, action_sequence: np.ndarray
    ) -> tuple[float, tuple[tuple[float, float, bool, bytes], ...]]:
        discounted_reward = 0.0
        transitions: list[tuple[float, float, bool, bytes]] = []
        final_observation: np.ndarray | None = None
        bootstrap = True
        executed_steps = 0
        for offset, candidate_action in enumerate(action_sequence):
            transition = environment.step(candidate_action)
            final_observation = np.asarray(transition.observation, dtype=np.float32)
            discounted_reward += (float(rebrac_config.discount) ** offset) * float(
                transition.reward
            )
            transitions.append(
                (
                    float(transition.reward),
                    float(transition.continuation),
                    bool(transition.is_last),
                    final_observation.tobytes(order="C"),
                )
            )
            executed_steps += 1
            if transition.is_last:
                bootstrap = False
                break
        terminal_value = 0.0
        if bootstrap:
            assert final_observation is not None
            terminal_observation = jnp.asarray(final_observation[None])
            terminal_action = rebrac_actor(rebrac_state.actor, terminal_observation)
            terminal_value = float(
                jax.device_get(
                    jnp.min(
                        rebrac_critics(
                            rebrac_state.critics,
                            terminal_observation,
                            terminal_action,
                        ),
                        axis=0,
                    )[0]
                )
            )
        objective = (
            discounted_reward
            + (float(rebrac_config.discount) ** executed_steps) * terminal_value
        )
        return objective, tuple(transitions)

    environment = DMCAdapter(TASK, seed=int(evaluation_seed), action_repeat=1)
    try:
        observation = environment.reset()
        for step in range(length):
            reference = np.asarray(trace["actions"][0, step], dtype=np.float32)
            current = jnp.asarray(observation[None], dtype=jnp.float32)
            belief = a0_beliefs[step]
            if step in selected:
                candidates = np.stack(
                    (
                        reference,
                        np.clip(reference + np.asarray([0.1, 0.0]), -1.0, 1.0),
                        np.clip(reference - np.asarray([0.1, 0.0]), -1.0, 1.0),
                    )
                ).astype(np.float32)
                predicted = predict_state_action_reward_from_observation(
                    world_model["reward_transition"],
                    jnp.broadcast_to(
                        jnp.asarray(observation),
                        (candidates.shape[0], *dreamer_config.observation_shape),
                    ),
                    jnp.asarray(candidates),
                    dreamer_config,
                )
                snapshot = environment.snapshot()
                rewards: list[float] = []
                for candidate in candidates:
                    environment.restore(snapshot)
                    first = environment.step(candidate)
                    environment.restore(snapshot)
                    replay = environment.step(candidate)
                    if not (
                        first.reward == replay.reward
                        and first.continuation == replay.continuation
                        and first.is_last == replay.is_last
                        and np.array_equal(first.observation, replay.observation)
                    ):
                        raise ValueError("snapshot-restored branch is not exact")
                    rewards.append(float(first.reward))
                environment.restore(snapshot)
                predicted_rows.append(
                    np.asarray(jax.device_get(predicted), dtype=np.float64).tolist()
                )
                simulator_rows.append(rewards)
                reset_ids.append(
                    f"dmc-reacher-{evaluation_seed}-{world_seed}-{actor_seed}-{step}"
                )
            if step in horizon_selected:
                reference_sequence = np.asarray(
                    trace["actions"][0, step : step + horizon], dtype=np.float32
                )
                positive_sequence = reference_sequence.copy()
                positive_sequence[:, 0] = np.clip(
                    positive_sequence[:, 0] + ACTION_SEQUENCE_RESIDUAL_LIMIT,
                    -1.0,
                    1.0,
                )
                negative_sequence = reference_sequence.copy()
                negative_sequence[:, 0] = np.clip(
                    negative_sequence[:, 0] - ACTION_SEQUENCE_RESIDUAL_LIMIT,
                    -1.0,
                    1.0,
                )
                candidate_sequences = np.stack(
                    (reference_sequence, positive_sequence, negative_sequence)
                )
                planner_noises = jax.random.normal(
                    jax.random.fold_in(planner_noise_key, step),
                    (
                        ACTION_SEQUENCE_PARTICLES,
                        horizon,
                        dreamer_config.stochastic_dim,
                    ),
                )
                predicted_objectives = [
                    float(
                        jax.device_get(
                            planner_objective(
                                belief,
                                current,
                                jnp.asarray(candidate_sequence),
                                planner_noises,
                            )
                        )
                    )
                    for candidate_sequence in candidate_sequences
                ]
                imagined_observations, imagined_actions = imagine_points(
                    belief,
                    current,
                    jnp.asarray(candidate_sequences),
                    planner_noises,
                )
                imagined_observations = np.asarray(
                    jax.device_get(imagined_observations), dtype=np.float32
                )
                imagined_actions = np.asarray(
                    jax.device_get(imagined_actions), dtype=np.float32
                )
                best_index = int(np.argmax(np.asarray(predicted_objectives)))
                imagined_proposal_observations.append(
                    imagined_observations.reshape(-1, *dreamer_config.observation_shape)
                )
                imagined_proposal_actions.append(
                    imagined_actions.reshape(-1, dreamer_config.action_dim)
                )
                imagined_best_observations.append(
                    imagined_observations[best_index].reshape(
                        -1, *dreamer_config.observation_shape
                    )
                )
                imagined_best_actions.append(
                    imagined_actions[best_index].reshape(-1, dreamer_config.action_dim)
                )
                imagined_reference_observations.append(
                    imagined_observations[0].reshape(
                        -1, *dreamer_config.observation_shape
                    )
                )
                imagined_reference_actions.append(
                    imagined_actions[0].reshape(-1, dreamer_config.action_dim)
                )
                snapshot = environment.snapshot()
                simulator_objectives: list[float] = []
                for candidate_sequence in candidate_sequences:
                    environment.restore(snapshot)
                    first_objective, first_certificate = simulator_planner_objective(
                        environment, candidate_sequence
                    )
                    environment.restore(snapshot)
                    replay_objective, replay_certificate = simulator_planner_objective(
                        environment, candidate_sequence
                    )
                    if (
                        first_objective != replay_objective
                        or first_certificate != replay_certificate
                    ):
                        raise ValueError(
                            "snapshot-restored H5 planner branch is not exact"
                        )
                    simulator_objectives.append(first_objective)
                environment.restore(snapshot)
                predicted_horizon_rows.append(predicted_objectives)
                simulator_horizon_rows.append(simulator_objectives)
                horizon_reset_ids.append(
                    "dmc-reacher-h5-planner-"
                    f"{evaluation_seed}-{world_seed}-{actor_seed}-{step}"
                )
            transition = environment.step(reference)
            observation = transition.observation
            if transition.is_last:
                break
    finally:
        environment.close()
    provenance = CounterfactualProvenance.verified_exact_branch(
        reset_ids,
        simulator_id="dm_control.reacher.easy/action_repeat=1",
        branch_protocol="process_local_physics_time_limit_task_rng_snapshot_restore_twice",
    )
    result = counterfactual_candidate_diagnostics(
        predicted_rows,
        simulator_rows,
        provenance=provenance,
        require_identifiable=True,
    ).to_dict()
    if not horizon_reset_ids:
        raise ValueError("exact H5 counterfactual diagnostic has no eligible branch")
    horizon_provenance = CounterfactualProvenance.verified_exact_branch(
        horizon_reset_ids,
        simulator_id="dm_control.reacher.easy/action_repeat=1",
        branch_protocol=(
            "h5_action_sequence_process_local_snapshot_restore_twice_with_"
            "discounted_rewards_and_frozen_rebrac_terminal_value"
        ),
    )
    result["diagnostic_scope"] = "one_step_immediate_reward"
    result["horizon5_planner_objective"] = counterfactual_candidate_diagnostics(
        predicted_horizon_rows,
        simulator_horizon_rows,
        provenance=horizon_provenance,
        require_identifiable=True,
    ).to_dict()
    result["horizon5_protocol"] = {
        "horizon": horizon,
        "particles": ACTION_SEQUENCE_PARTICLES,
        "candidate_sequences": (
            "a0_reference_trace_and_full_horizon_plus_minus_0.1_action_dim_0"
        ),
        "predicted_objective": (
            "world_model_expected_discounted_state_action_reward_plus_frozen_"
            "rebrac_terminal_min_q"
        ),
        "simulator_objective": (
            "exact_discounted_simulator_reward_plus_same_frozen_rebrac_"
            "terminal_min_q_on_real_terminal_observation"
        ),
        "common_world_model_noise_across_candidates": True,
    }
    coverage_arrays = {
        "proposal_observations": np.concatenate(imagined_proposal_observations, axis=0),
        "proposal_actions": np.concatenate(imagined_proposal_actions, axis=0),
        "best_observations": np.concatenate(imagined_best_observations, axis=0),
        "best_actions": np.concatenate(imagined_best_actions, axis=0),
        "reference_observations": np.concatenate(
            imagined_reference_observations, axis=0
        ),
        "reference_actions": np.concatenate(imagined_reference_actions, axis=0),
    }
    return result, coverage_arrays


def _dependency_evaluation_cell(
    old: Mapping[str, Any], world_seed: int, actor_seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in old["evaluation_cells"]
        if int(row["world_model_seed"]) == int(world_seed)
        and int(row["actor_seed"]) == int(actor_seed)
    ]
    if len(rows) != 1:
        raise ValueError("dependency evaluation cell is absent or duplicated")
    return rows[0]


def _compute_diagnostic_cell(
    output_root: str | Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from .actor_gap_diagnostics import (
        component_residual_decomposition,
        score_observation_action_coverage,
    )

    root = Path(output_root)
    old = _dependency_manifest(manifest)
    calibration_arrays = benchmark.load_npz(
        root / str(manifest["calibration"]["arrays_path"])
    )
    world_seed = int(cell["world_model_seed"])
    output_traces: dict[str, np.ndarray] = {}
    actor_rows: list[dict[str, Any]] = []
    for actor_seed in ACTOR_SEEDS:
        world, dreamer, rebrac_state, rebrac, source = _load_control_inputs(
            manifest, world_seed, actor_seed
        )
        dependency_cell = _dependency_evaluation_cell(old, world_seed, actor_seed)
        evaluation_seeds = [
            int(value)
            for value in dependency_cell["evaluation_seeds"][
                : int(manifest["evaluation_episodes"])
            ]
        ]
        row = _calibration_row(root, manifest, world_seed, actor_seed)
        anchors = calibration_arrays[f"world_{world_seed}_anchor_observations"]
        a0_capture: dict[str, Any] = {}
        _, trace, _ = _run_flowmpc_arm(
            world,
            dreamer,
            rebrac_state,
            rebrac,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=evaluation_seeds,
            maximum_steps=int(manifest["maximum_environment_steps"]),
            trust=False,
            persistence="persistent",
            heldout_acceptance_enabled=False,
            anchor_observations=anchors,
            diagnostic_capture=a0_capture,
        )
        retained, retained_path = _load_dependency_evaluation_trace(
            manifest,
            old,
            dependency_cell,
        )
        _assert_trace_subset_close(
            retained,
            trace,
            prefix="flowmpc",
            episodes=len(evaluation_seeds),
        )
        if a0_capture.get("schema_version") != "trajectory-imf-live-a0-context-v1":
            raise ValueError("live A0 diagnostic context was not captured")
        captured_beliefs = a0_capture.get("beliefs_by_episode")
        captured_noise_contexts = a0_capture.get("noise_contexts")
        if not isinstance(captured_beliefs, list) or not isinstance(
            captured_noise_contexts, list
        ):
            raise ValueError("live A0 diagnostic context is incomplete")
        observations, next_observations = _replay_observations(evaluation_seeds, trace)
        lengths = np.asarray(trace["lengths"], dtype=np.int64)
        mask = np.arange(trace["actions"].shape[1])[None] < lengths[:, None]
        coverage = _coverage_from_arrays(f"world_{world_seed}", row, calibration_arrays)
        coverage_result = score_observation_action_coverage(
            coverage,
            observations[mask],
            np.asarray(trace["actions"])[mask],
            distance_chunk_size=256,
        ).to_dict()
        predictions = _one_step_prediction_rows(
            world,
            dreamer,
            rebrac_state,
            rebrac,
            observations,
            next_observations,
            trace,
            captured_beliefs,
            world_seed=world_seed,
            actor_seed=actor_seed,
        )
        learned_continuation_residuals = component_residual_decomposition(
            predictions["predicted_rewards"],
            predictions["target_rewards"],
            predictions["predicted_continuations"],
            predictions["target_continuations"],
            predictions["predicted_next_values"],
            predictions["target_next_values"],
            discount=float(rebrac.discount),
        ).to_dict()
        planner_residuals = component_residual_decomposition(
            predictions["predicted_rewards"],
            predictions["target_rewards"],
            np.ones_like(predictions["predicted_continuations"]),
            np.ones_like(predictions["target_continuations"]),
            predictions["predicted_next_values"],
            predictions["target_next_values"],
            discount=float(rebrac.discount),
        ).to_dict()
        noise = _independent_noise_result(
            world,
            dreamer,
            rebrac_state,
            rebrac,
            trace,
            captured_noise_contexts,
            world_seed=world_seed,
            actor_seed=actor_seed,
        )
        counterfactual, imagined_coverage_arrays = _exact_counterfactual_result(
            world,
            dreamer,
            rebrac_state,
            rebrac,
            observations,
            trace,
            captured_beliefs[0],
            evaluation_seeds[0],
            world_seed=world_seed,
            actor_seed=actor_seed,
        )
        imagined_coverage = {}
        for scope in ("proposal", "best", "reference"):
            scored = score_observation_action_coverage(
                coverage,
                imagined_coverage_arrays[f"{scope}_observations"],
                imagined_coverage_arrays[f"{scope}_actions"],
                distance_chunk_size=256,
            ).to_dict()
            imagined_coverage[scope] = {
                key: value for key, value in scored.items() if key != "points"
            }
        horizon_summaries = []
        for horizon in (1, 3, 5):
            horizon_mask = predictions["horizons"] == horizon
            horizon_summaries.append(
                {
                    "horizon": horizon,
                    "nested_count": int(np.sum(horizon_mask)),
                    "mean_observation_rmse": float(
                        np.mean(predictions["horizon_observation_rmse"][horizon_mask])
                    ),
                    "mean_reward_absolute_error": float(
                        np.mean(
                            predictions["horizon_reward_absolute_error"][horizon_mask]
                        )
                    ),
                    "mean_predictive_reward_sd": float(
                        np.mean(
                            predictions["horizon_predictive_reward_sd"][horizon_mask]
                        )
                    ),
                }
            )
        actor_rows.append(
            {
                "actor_seed": actor_seed,
                "evaluation_seeds": evaluation_seeds,
                "a0_dependency_replay_verified": True,
                "coverage": {
                    key: value
                    for key, value in coverage_result.items()
                    if key != "points"
                },
                "imagined_horizon5_coverage": imagined_coverage,
                "component_residuals": planner_residuals,
                "component_residual_interface": (
                    "flowmpc_state_action_reward_fixed_discount_no_continuation"
                ),
                "component_residual_belief_path": (
                    "exact_a0_flowmpc_posterior_keyed_on_policy_prefix"
                ),
                "component_residual_terminal_policy": "frozen_rebrac_actor",
                "component_residual_next_value_estimator": (
                    f"mean_over_{FLOWMPC_PARTICLES}_actual_flowmpc_step0_noise_particles"
                ),
                "learned_continuation_component_residuals": (
                    learned_continuation_residuals
                ),
                "horizon_predictive_particles": 16,
                "horizon_summaries": horizon_summaries,
                "independent_noise": noise,
                "exact_counterfactual": counterfactual,
                "reward_checkpoint_sha256": source["reward_checkpoint_sha256"],
                "rebrac_checkpoint_sha256": benchmark.file_sha256(
                    Path(str(manifest["dependency_root"]))
                    / str(
                        _dependency_rebrac_cell(old, world_seed, actor_seed)[
                            "checkpoint_path"
                        ]
                    )
                ),
                "dependency_trace_file_sha256": benchmark.file_sha256(retained_path),
            }
        )
        prefix = f"actor_{actor_seed}"
        for name, value in trace.items():
            output_traces[f"{prefix}_a0_{name}"] = np.asarray(value)
        output_traces[f"{prefix}_observations"] = observations
        output_traces[f"{prefix}_next_observations"] = next_observations
        for name, value in predictions.items():
            output_traces[f"{prefix}_{name}"] = np.asarray(value)
        for name, value in imagined_coverage_arrays.items():
            output_traces[f"{prefix}_imagined_h5_{name}"] = np.asarray(value)
    core = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "world_model_seed": world_seed,
        "actor_rows": actor_rows,
        "diagnostic_only_no_selection_or_mutation": True,
    }
    return core, output_traces


def _compute_diagnostic_cell_after_discarded_warmup(
    root: Path, manifest: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    """Mirror the discarded first full execution in creation and replay."""

    _compute_diagnostic_cell(root, manifest, cell)
    retained_started = time.perf_counter()
    core, traces = _compute_diagnostic_cell(root, manifest, cell)
    return core, traces, time.perf_counter() - retained_started


def run_diagnostic_cell(
    dependency_root: str | Path,
    output_root: str | Path,
    index: int,
    *,
    submission_authorization: Mapping[str, Any],
    model_updates: int = MODEL_UPDATES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    manifest = write_manifest(
        dependency_root,
        output_root,
        model_updates=model_updates,
        evaluation_episodes=evaluation_episodes,
    )
    root = Path(output_root)
    cell = _cell(manifest, "diagnostic", index)
    current_authorization = _validate_retained_submission_authorization(
        root, manifest, cell, "diagnostic", submission_authorization
    )
    result_path = root / str(cell["result_path"])
    if result_path.is_file():
        if current_authorization["mode"] != "verification_only":
            raise ValueError(
                "retained diagnostic evidence requires verification-only submission"
            )
        return read_json(result_path)
    if current_authorization["mode"] != "new":
        raise ValueError("new diagnostic evidence requires a new-mode submission")
    creation_scheduler, creation_receipt, creation_runtime = (
        _live_array_execution_binding(
            root, manifest, cell, "diagnostic", current_authorization
        )
    )
    _reject_partial_artifacts(result_path, root / str(cell["trace_path"]))
    calibration_marker = _validate_calibration_marker(root, manifest)
    if calibration_marker.get("training_only") is not True:
        raise ValueError("diagnostics require authenticated train-only calibration")
    # Exercise the complete deterministic diagnostic once before retaining
    # evidence.  The discarded pass does replay exact simulator branches in an
    # isolated environment instance, but publishes no artifact and mutates no
    # checkpoint.  This mirrors the verifier and absorbs first-execution
    # compiler/autotuning effects before the retained pass.
    total_started = time.perf_counter()
    core, traces, retained_compute_wall_seconds = (
        _compute_diagnostic_cell_after_discarded_warmup(root, manifest, cell)
    )
    core["discarded_full_diagnostic_warmup"] = True
    core["creation_submission_authorization"] = current_authorization
    core["creation_scheduler_provenance"] = creation_scheduler
    core["creation_submission_receipt"] = creation_receipt
    trace_path = root / str(cell["trace_path"])
    if trace_path.parent != result_path.parent:
        raise ValueError("diagnostic artifacts do not share one bundle directory")
    with _staged_artifact_directory(result_path.parent) as staging:
        staged_trace_path = staging / trace_path.name
        staged_result_path = staging / result_path.name
        _write_npz_exclusive(staged_trace_path, traces)
        result = {
            **core,
            "trace_file_sha256": benchmark.file_sha256(staged_trace_path),
            "trace_sha256": benchmark.array_sha256(traces),
            "core_sha256": benchmark.object_sha256(core),
            "wall_seconds": time.perf_counter() - total_started,
            "wall_seconds_scope": "result_creation_including_discarded_full_warmup",
            "retained_compute_wall_seconds": retained_compute_wall_seconds,
            "runtime": creation_runtime,
        }
        if not _finite_tree(result):
            raise FloatingPointError("diagnostic result is not finite")
        _write_json_exclusive(staged_result_path, result)
    return result


def verify_diagnostic_cell(
    output_root: str | Path,
    index: int,
    *,
    strict_replay: bool = True,
    submission_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if strict_replay is not True:
        raise ValueError("diagnostic certification requires strict replay")
    verification_started = time.perf_counter()
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell(manifest, "diagnostic", index)
    verification_authorization = _validate_retained_submission_authorization(
        root, manifest, cell, "diagnostic", submission_authorization
    )
    verification_scheduler, verification_receipt, verification_runtime = (
        _live_array_execution_binding(
            root, manifest, cell, "diagnostic", verification_authorization
        )
    )
    result_path = root / str(cell["result_path"])
    trace_path = root / str(cell["trace_path"])
    result = read_json(result_path)
    creation_authorization, creation_scheduler, _, _ = (
        _validate_retained_creation_binding(root, manifest, cell, "diagnostic", result)
    )
    _require_distinct_replay_processes(creation_scheduler, verification_scheduler)
    trace_file_sha256 = benchmark.file_sha256(trace_path)
    core_keys = (
        "schema_version",
        "status",
        "source_commit",
        "manifest_sha256",
        "cell_id",
        "cell_index",
        "task",
        "world_model_seed",
        "actor_rows",
        "diagnostic_only_no_selection_or_mutation",
        "discarded_full_diagnostic_warmup",
        "creation_submission_authorization",
        "creation_scheduler_provenance",
        "creation_submission_receipt",
    )
    core = {key: result[key] for key in core_keys}
    if (
        result.get("schema_version") != DIAGNOSTIC_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("cell_id") != cell["cell_id"]
        or result.get("cell_index") != int(cell["index"])
        or result.get("trace_file_sha256") != trace_file_sha256
        or result.get("core_sha256") != benchmark.object_sha256(core)
        or not _finite_tree(result)
    ):
        raise ValueError("diagnostic cell contract differs")
    traces = benchmark.load_npz(trace_path)
    if result.get("trace_sha256") != benchmark.array_sha256(traces):
        raise ValueError("diagnostic trace payload differs")
    replay_core, replay_traces, _ = _compute_diagnostic_cell_after_discarded_warmup(
        root, manifest, cell
    )
    replay_core["discarded_full_diagnostic_warmup"] = True
    replay_core["creation_submission_authorization"] = result[
        "creation_submission_authorization"
    ]
    replay_core["creation_scheduler_provenance"] = result[
        "creation_scheduler_provenance"
    ]
    replay_core["creation_submission_receipt"] = result["creation_submission_receipt"]
    if replay_core != core:
        raise ValueError("diagnostic semantic replay differs")
    if benchmark.array_sha256(replay_traces) != result["trace_sha256"]:
        raise ValueError("diagnostic strict bitwise trace replay differs")
    return _write_verified_marker(
        root,
        manifest,
        cell,
        "diagnostic",
        result_path=result_path,
        extra={
            "trace_file_sha256": result["trace_file_sha256"],
            "strict_policy_model_environment_replay": True,
            "verification_submission_authorization": verification_authorization,
            "verification_scheduler_provenance": verification_scheduler,
            "verification_submission_receipt": verification_receipt,
            "verification_runtime": verification_runtime,
            "strict_replay_wall_seconds": time.perf_counter() - verification_started,
        },
    )


def _flowmpc_imagined_actor_points(
    actor: Any,
    world_model: Any,
    dreamer_config: Any,
    belief: Any,
    current_observation: Any,
    noises: Any,
) -> tuple[Any, Any]:
    """Return the stage-reward observation-action queries of one objective."""

    import jax.numpy as jnp
    from imf_dreamer_jax import RSSMState, decode, rebrac_actor
    from imf_dreamer_jax.world_model import sample_prior, transition_deterministic

    particle_noises = jnp.asarray(noises)
    if particle_noises.ndim != 3:
        raise ValueError(
            "FlowMPC imagined noises must have [particle, horizon, latent]"
        )
    particles = particle_noises.shape[0]
    state = RSSMState(
        jnp.broadcast_to(
            belief.deterministic,
            (particles, dreamer_config.deterministic_dim),
        ),
        jnp.broadcast_to(
            belief.stochastic,
            (particles, dreamer_config.stochastic_dim),
        ),
    )
    observation = jnp.broadcast_to(
        current_observation,
        (particles, *dreamer_config.observation_shape),
    )
    observation_rows = []
    action_rows = []
    for step in range(particle_noises.shape[1]):
        action = rebrac_actor(actor, observation.reshape(particles, -1))
        observation_rows.append(observation)
        action_rows.append(action)
        deterministic = transition_deterministic(
            world_model, state, action, dreamer_config
        )
        stochastic, _ = sample_prior(
            world_model,
            deterministic,
            None,
            dreamer_config,
            noise=particle_noises[:, step],
        )
        state = RSSMState(deterministic, stochastic)
        observation = decode(world_model, state.feature, dreamer_config)
    return jnp.stack(observation_rows, axis=1), jnp.stack(action_rows, axis=1)


def _run_flowmpc_arm(
    world_model: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    *,
    world_seed: int,
    actor_seed: int,
    evaluation_seeds: list[int],
    maximum_steps: int,
    trust: bool,
    persistence: str,
    heldout_acceptance_enabled: bool,
    anchor_observations: np.ndarray,
    diagnostic_capture: dict[str, Any] | None = None,
) -> tuple[list[float], dict[str, np.ndarray], dict[str, float]]:
    """Run one FlowMPC arm with isolated trust/persistence/acceptance factors."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        initial_state,
        jit_flowmpc_adapt_actor,
        jit_flowmpc_objective,
        observe_step,
        rebrac_actor,
    )
    from imf_dreamer_jax.nn import tree_global_norm
    from imf_dreamer_jax.robust_flowmpc import (
        ActionTrustRegionConfig,
        HeldOutAcceptanceConfig,
        PersistenceConfig,
        ReferenceActorState,
        apply_held_out_noise_fallback,
        evaluate_flowmpc_held_out_acceptance,
        init_reference_actor_state,
        jit_backtrack_actor_proposal,
        jit_finish_actor_adaptation_step,
        measure_actor_reference_action_drift,
    )
    from .dmc import DMCAdapter

    if persistence not in ("persistent", "reset"):
        raise ValueError("unknown persistence mode")
    if diagnostic_capture is not None:
        if trust or persistence != "persistent" or heldout_acceptance_enabled:
            raise ValueError("diagnostic capture is restricted to the exact A0 arm")
        if diagnostic_capture:
            raise ValueError("diagnostic capture must be empty before the A0 rollout")
        diagnostic_capture.update(
            {
                "schema_version": "trajectory-imf-live-a0-context-v1",
                "beliefs_by_episode": [],
                "noise_contexts": [],
            }
        )
    flow_config = _flowmpc_config(rebrac_config)
    persistence_config = PersistenceConfig(mode=persistence)
    trust_config = ActionTrustRegionConfig(
        anchor_mean_squared_budget=TRUST_ANCHOR_MSE_BUDGET,
        current_max_absolute_budget=TRUST_CURRENT_LINF_BUDGET,
    )
    acceptance_config = HeldOutAcceptanceConfig(
        mode="held_out_noise" if heldout_acceptance_enabled else "disabled",
        minimum_improvement=HELDOUT_MINIMUM_IMPROVEMENT,
        baseline="adaptation_start",
        fallback="adaptation_start",
    )
    anchors = jnp.asarray(anchor_observations, dtype=jnp.float32)
    actor_function = jax.jit(rebrac_actor)

    @jax.jit
    def observe_function(observation: Any, previous_action: Any, belief: Any, key: Any):
        return observe_step(
            world_model,
            observation,
            previous_action,
            belief,
            key,
            dreamer_config,
        )[0]

    def select_actor(
        state: ReferenceActorState,
        belief: Any,
        current: Any,
        proposal_noises: Any,
        heldout_noises: Any,
    ) -> tuple[ReferenceActorState, Any, dict[str, Any]]:
        adaptation_start = (
            state.carried_actor
            if persistence_config.mode == "persistent"
            else state.frozen_reference_actor
        )
        update = jit_flowmpc_adapt_actor(
            adaptation_start,
            rebrac_state.critics,
            world_model,
            belief,
            current,
            proposal_noises,
            dreamer_config,
            rebrac_config,
            flow_config,
        )
        selected = update.actor
        backtracks = jnp.asarray(0, dtype=jnp.int32)
        used_reference_fallback = jnp.asarray(False)
        if trust:
            constrained = jit_backtrack_actor_proposal(
                state.frozen_reference_actor,
                adaptation_start,
                selected,
                anchors,
                current,
                trust_config,
            )
            selected = constrained.actor
            backtracks = constrained.backtracks
            used_reference_fallback = constrained.used_reference_fallback
        acceptance_improvement = jnp.asarray(0.0, dtype=jnp.float32)
        accepted = jnp.asarray(True)
        if heldout_acceptance_enabled:
            acceptance = evaluate_flowmpc_held_out_acceptance(
                selected,
                adaptation_start,
                state.frozen_reference_actor,
                rebrac_state.critics,
                world_model,
                belief,
                current,
                heldout_noises,
                dreamer_config,
                rebrac_config,
                flow_config,
                acceptance_config,
            )
            selected = apply_held_out_noise_fallback(
                selected,
                adaptation_start,
                state.frozen_reference_actor,
                acceptance,
                acceptance_config,
            )
            acceptance_improvement = acceptance.improvement
            accepted = acceptance.accepted
        # Acceptance can fall back to an already-carried persistent actor that
        # is infeasible at this new state. Re-project the actually selected
        # actor so the A3 trust intervention remains true after rejection too.
        if trust:
            executed_constraint = jit_backtrack_actor_proposal(
                state.frozen_reference_actor,
                adaptation_start,
                selected,
                anchors,
                current,
                trust_config,
            )
            selected = executed_constraint.actor
            backtracks = backtracks + executed_constraint.backtracks
            used_reference_fallback = jnp.logical_or(
                used_reference_fallback,
                executed_constraint.used_reference_fallback,
            )
        executed_drift = measure_actor_reference_action_drift(
            state.frozen_reference_actor,
            selected,
            anchors,
            current,
            trust_config,
        )
        selected_objective = jit_flowmpc_objective(
            selected,
            rebrac_state.critics,
            world_model,
            belief,
            current,
            proposal_noises,
            dreamer_config,
            rebrac_config,
            flow_config,
        ).objective
        parameter_delta = tree_global_norm(
            jax.tree_util.tree_map(
                lambda candidate, start: candidate - start,
                selected,
                adaptation_start,
            )
        )
        next_state = jit_finish_actor_adaptation_step(
            state, selected, persistence_config
        )
        telemetry = {
            "objective_before": update.objective_before,
            "objective_after": selected_objective,
            "gradient_norm": update.gradient_norm,
            "parameter_delta": parameter_delta,
            "anchor_drift": executed_drift.anchor_mean_squared_action_drift,
            "current_drift": executed_drift.current_max_absolute_action_drift,
            "within_reference_budgets": executed_drift.feasible,
            "backtracks": backtracks,
            "used_reference_fallback": used_reference_fallback,
            "heldout_improvement": acceptance_improvement,
            "accepted": accepted,
        }
        return next_state, selected, update.actor, telemetry

    world_digest = benchmark._tree_digest(world_model)
    rebrac_digest = benchmark._tree_digest(rebrac_state)
    sequence_names = (
        "actions",
        "observations",
        "rewards",
        "continuations",
        "is_last",
        "objective_before",
        "objective_after",
        "gradient_norm",
        "parameter_delta",
        "anchor_drift",
        "current_drift",
        "backtracks",
        "used_reference_fallback",
        "heldout_improvement",
        "accepted",
        "within_reference_budgets",
    )
    sequences: dict[str, list[np.ndarray]] = {name: [] for name in sequence_names}
    imagined_points: dict[str, list[np.ndarray]] = {
        f"{scope}_{kind}": []
        for scope in ("proposal", "best", "reference")
        for kind in ("observations", "actions")
    }
    imagine_function = jax.jit(
        lambda actor, belief, current, noises: _flowmpc_imagined_actor_points(
            actor,
            world_model,
            dreamer_config,
            belief,
            current,
            noises,
        )
    )
    returns: list[float] = []
    total_timed_seconds = 0.0
    timed_steps = 0
    warmed = False
    for episode_index, evaluation_seed in enumerate(evaluation_seeds):
        environment = DMCAdapter(TASK, seed=int(evaluation_seed), action_repeat=1)
        episode: dict[str, list[Any]] = {name: [] for name in sequence_names}
        episode_beliefs: list[Any] = []
        try:
            observation = environment.reset()
            belief = initial_state(dreamer_config, 1)
            previous_action = jnp.zeros((1, dreamer_config.action_dim), jnp.float32)
            reference_state = init_reference_actor_state(rebrac_state.actor)
            posterior_key = benchmark.derive_jax_key(
                "flowmpc-posterior", TASK, world_seed, actor_seed, evaluation_seed
            )
            noise_key = benchmark.derive_jax_key(
                "flowmpc-noise", TASK, world_seed, actor_seed, evaluation_seed
            )
            heldout_key = benchmark.derive_jax_key(
                "actor-gap-heldout-acceptance",
                TASK,
                world_seed,
                actor_seed,
                evaluation_seed,
            )
            for step in range(maximum_steps):
                current = jnp.asarray(observation[None], dtype=jnp.float32)
                shape = (
                    flow_config.particles,
                    flow_config.horizon,
                    dreamer_config.stochastic_dim,
                )
                noises = jax.random.normal(jax.random.fold_in(noise_key, step), shape)
                heldout_noises = jax.random.normal(
                    jax.random.fold_in(heldout_key, step), shape
                )
                if not warmed:
                    warm_belief = observe_function(
                        current,
                        previous_action,
                        belief,
                        jax.random.fold_in(posterior_key, step),
                    )
                    warm_state, warm_actor, warm_proposed_actor, _ = select_actor(
                        reference_state,
                        warm_belief,
                        current,
                        noises,
                        heldout_noises,
                    )
                    del warm_state
                    jax.block_until_ready(actor_function(warm_actor, current))
                    for warm_coverage_actor in (
                        warm_proposed_actor,
                        warm_actor,
                        reference_state.frozen_reference_actor,
                    ):
                        warm_observations, warm_actions = imagine_function(
                            warm_coverage_actor,
                            warm_belief,
                            current,
                            noises[:IMAGINED_COVERAGE_PARTICLES],
                        )
                        jax.block_until_ready(warm_observations)
                        jax.block_until_ready(warm_actions)
                    warmed = True
                started = time.perf_counter()
                candidate_belief = observe_function(
                    current,
                    previous_action,
                    belief,
                    jax.random.fold_in(posterior_key, step),
                )
                adaptation_start_actor = (
                    reference_state.carried_actor
                    if persistence_config.mode == "persistent"
                    else reference_state.frozen_reference_actor
                )
                (
                    reference_state,
                    selected_actor,
                    proposed_actor,
                    telemetry,
                ) = select_actor(
                    reference_state,
                    candidate_belief,
                    current,
                    noises,
                    heldout_noises,
                )
                action = actor_function(selected_actor, current)
                host_action = np.asarray(jax.device_get(action[0]), dtype=np.float32)
                jax.block_until_ready(action)
                if diagnostic_capture is not None:
                    episode_beliefs.append(candidate_belief)
                    if episode_index == 0 and step <= 15:
                        diagnostic_capture["noise_contexts"].append(
                            {
                                "world_model_seed": int(world_seed),
                                "actor_seed": int(actor_seed),
                                "evaluation_seed": int(evaluation_seed),
                                "step": int(step),
                                "belief": candidate_belief,
                                "current_observation": current,
                                "adaptation_start_actor": adaptation_start_actor,
                                "updated_actor": proposed_actor,
                                "proposal_noise": noises,
                                "host_action": host_action.copy(),
                            }
                        )
                elapsed = time.perf_counter() - started
                total_timed_seconds += elapsed
                timed_steps += 1
                transition = environment.step(host_action)
                episode["actions"].append(host_action)
                episode["observations"].append(np.asarray(observation, np.float32))
                episode["rewards"].append(float(transition.reward))
                episode["continuations"].append(float(transition.continuation))
                episode["is_last"].append(bool(transition.is_last))
                for name, value in telemetry.items():
                    episode[name].append(np.asarray(jax.device_get(value)))
                if step % IMAGINED_COVERAGE_STEP_INTERVAL == 0:
                    coverage_noises = noises[:IMAGINED_COVERAGE_PARTICLES]
                    for scope, actor in (
                        ("proposal", proposed_actor),
                        ("best", selected_actor),
                        ("reference", reference_state.frozen_reference_actor),
                    ):
                        imagined_observations, imagined_actions = imagine_function(
                            actor, candidate_belief, current, coverage_noises
                        )
                        imagined_points[f"{scope}_observations"].append(
                            np.asarray(
                                jax.device_get(imagined_observations), dtype=np.float32
                            ).reshape(-1, *dreamer_config.observation_shape)
                        )
                        imagined_points[f"{scope}_actions"].append(
                            np.asarray(
                                jax.device_get(imagined_actions), dtype=np.float32
                            ).reshape(-1, dreamer_config.action_dim)
                        )
                observation = transition.observation
                belief = candidate_belief
                previous_action = action
                if transition.is_last:
                    break
        finally:
            environment.close()
        if diagnostic_capture is not None:
            diagnostic_capture["beliefs_by_episode"].append(tuple(episode_beliefs))
        returns.append(float(np.sum(np.asarray(episode["rewards"], np.float64))))
        for name in sequence_names:
            if name in ("actions",):
                dtype = np.float32
            elif name in ("observations",):
                dtype = np.float32
            elif name in (
                "is_last",
                "used_reference_fallback",
                "accepted",
                "within_reference_budgets",
            ):
                dtype = np.bool_
            elif name == "backtracks":
                dtype = np.int32
            elif name in ("rewards", "continuations"):
                dtype = np.float64
            else:
                dtype = np.float32
            sequences[name].append(np.asarray(episode[name], dtype=dtype))
    trace: dict[str, np.ndarray] = {}
    lengths: np.ndarray | None = None
    for name in sequence_names:
        trailing = {
            "actions": (dreamer_config.action_dim,),
            "observations": dreamer_config.observation_shape,
        }.get(name, ())
        dtype = sequences[name][0].dtype
        values, local_lengths = dependency._pad_sequences(
            sequences[name], trailing_shape=trailing, dtype=dtype
        )
        if lengths is None:
            lengths = local_lengths
        elif not np.array_equal(lengths, local_lengths):
            raise RuntimeError("FlowMPC arm telemetry lengths differ")
        trace[name] = values
    assert lengths is not None
    trace["lengths"] = lengths
    trace["evaluation_seeds"] = np.asarray(evaluation_seeds, dtype=np.uint32)
    for name, values in imagined_points.items():
        if not values:
            raise RuntimeError(
                "FlowMPC controller produced no imagined coverage points"
            )
        trace[f"imagined_{name}"] = np.concatenate(values, axis=0)
    if (
        benchmark._tree_digest(world_model) != world_digest
        or benchmark._tree_digest(rebrac_state) != rebrac_digest
    ):
        raise RuntimeError("FlowMPC arm changed a frozen model or ReBRAC state")
    timing = {
        "timed_steps": float(timed_steps),
        "total_timed_seconds": total_timed_seconds,
        "discarded_compile_warmup_steps": 1.0,
        "mean_milliseconds_per_step": (
            1000.0 * total_timed_seconds / timed_steps if timed_steps else 0.0
        ),
    }
    return returns, trace, timing


def _recursive_action_sequence_objective(
    world_model: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    belief: Any,
    current_observation: Any,
    action_sequence: Any,
    noises: Any,
) -> Any:
    """Evaluate exactly the same residual action sequence for gradient and CEM."""

    import jax.numpy as jnp
    from imf_dreamer_jax import RSSMState, decode, rebrac_actor, rebrac_critics
    from imf_dreamer_jax.world_model import (
        predict_state_action_reward_from_observation,
        sample_prior,
        transition_deterministic,
    )

    particles = noises.shape[0]
    state = RSSMState(
        jnp.broadcast_to(
            belief.deterministic,
            (particles, dreamer_config.deterministic_dim),
        ),
        jnp.broadcast_to(
            belief.stochastic,
            (particles, dreamer_config.stochastic_dim),
        ),
    )
    observation = jnp.broadcast_to(
        current_observation,
        (particles, *dreamer_config.observation_shape),
    )
    stage = jnp.zeros((particles,), dtype=observation.dtype)
    for step in range(action_sequence.shape[0]):
        action = jnp.broadcast_to(
            action_sequence[step], (particles, dreamer_config.action_dim)
        )
        reward = predict_state_action_reward_from_observation(
            world_model["reward_transition"], observation, action, dreamer_config
        )
        stage = stage + (float(rebrac_config.discount) ** step) * reward
        deterministic = transition_deterministic(
            world_model, state, action, dreamer_config
        )
        stochastic, _ = sample_prior(
            world_model,
            deterministic,
            None,
            dreamer_config,
            noise=noises[:, step],
        )
        state = RSSMState(deterministic, stochastic)
        observation = decode(world_model, state.feature, dreamer_config)
    terminal_action = rebrac_actor(rebrac_state.actor, observation)
    terminal_q = jnp.min(
        rebrac_critics(rebrac_state.critics, observation, terminal_action), axis=0
    )
    return jnp.mean(stage) + (
        float(rebrac_config.discount) ** action_sequence.shape[0]
    ) * jnp.mean(terminal_q)


def _reference_action_sequence(
    world_model: Any,
    dreamer_config: Any,
    actor: Any,
    belief: Any,
    current_observation: Any,
) -> Any:
    import jax.numpy as jnp
    from imf_dreamer_jax import RSSMState, decode, rebrac_actor
    from imf_dreamer_jax.world_model import sample_prior, transition_deterministic

    state = belief
    observation = current_observation
    actions = []
    zero_noise = jnp.zeros((1, dreamer_config.stochastic_dim), jnp.float32)
    for _ in range(CONTROLLER_HORIZON):
        action = rebrac_actor(actor, observation)[0]
        actions.append(action)
        batched_action = action[None]
        deterministic = transition_deterministic(
            world_model, state, batched_action, dreamer_config
        )
        stochastic, _ = sample_prior(
            world_model,
            deterministic,
            None,
            dreamer_config,
            noise=zero_noise,
        )
        state = RSSMState(deterministic, stochastic)
        observation = decode(world_model, state.feature, dreamer_config)
    return jnp.stack(actions, axis=0)


def _run_action_sequence_arm(
    worlds: Sequence[Any],
    dreamer_configs: Sequence[Any],
    rebrac_state: Any,
    rebrac_config: Any,
    *,
    world_seed: int,
    actor_seed: int,
    evaluation_seeds: list[int],
    maximum_steps: int,
    optimizer: str,
    relative_risk_coefficient: float | None,
) -> tuple[list[float], dict[str, np.ndarray], dict[str, float]]:
    """Run equal-variable action-sequence gradient/CEM and ensemble controls."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state, observe_step
    from imf_dreamer_jax.robust_flowmpc import (
        ActionSequenceDomainConfig,
        ActionSequenceProposal,
        CEMActionSequenceConfig,
        GradientActionSequenceConfig,
        RelativePessimismConfig,
        cem_action_sequence_search,
        ensemble_relative_risk_score,
        gradient_action_sequence_search,
        make_action_sequence_domain,
    )
    from .dmc import DMCAdapter

    if optimizer not in ("gradient", "cem"):
        raise ValueError("optimizer must be gradient or cem")
    if relative_risk_coefficient is None and len(worlds) != 1:
        raise ValueError("absolute action-sequence search uses one world")
    if relative_risk_coefficient is not None and len(worlds) < 2:
        raise ValueError("relative pessimism requires an ensemble")
    digests = tuple(benchmark._tree_digest(world) for world in worlds)
    rebrac_digest = benchmark._tree_digest(rebrac_state)
    domain_config = ActionSequenceDomainConfig(
        horizon=CONTROLLER_HORIZON,
        action_dim=int(rebrac_config.action_dim),
        residual_limit=ACTION_SEQUENCE_RESIDUAL_LIMIT,
    )
    gradient_config = GradientActionSequenceConfig(
        iterations=ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS - 1,
        step_size=0.05,
        gradient_clip_norm=10.0,
    )
    cem_config = CEMActionSequenceConfig(
        iterations=2,
        population=4,
        elite_count=2,
        initial_std=0.10,
        minimum_std=0.01,
        maximum_std=0.10,
    )
    if 2 + cem_config.iterations * cem_config.population != (
        ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS
    ):
        raise AssertionError("CEM objective-call budget differs")

    def search(
        beliefs: Sequence[Any],
        current: Any,
        noises: Any,
        proposal_bank: Any,
    ) -> Any:
        reference = _reference_action_sequence(
            worlds[0], dreamer_configs[0], rebrac_state.actor, beliefs[0], current
        )
        domain = make_action_sequence_domain(reference, domain_config)
        initial = ActionSequenceProposal(jnp.zeros_like(reference))
        if relative_risk_coefficient is None:
            objective = lambda actions: _recursive_action_sequence_objective(
                worlds[0],
                dreamer_configs[0],
                rebrac_state,
                rebrac_config,
                beliefs[0],
                current,
                actions,
                noises,
            )
        else:
            reference_members = jnp.stack(
                [
                    _recursive_action_sequence_objective(
                        model,
                        config,
                        rebrac_state,
                        rebrac_config,
                        belief,
                        current,
                        reference,
                        noises,
                    )
                    for model, config, belief in zip(
                        worlds, dreamer_configs, beliefs, strict=True
                    )
                ]
            )

            def objective(actions: Any) -> Any:
                candidate_members = jnp.stack(
                    [
                        _recursive_action_sequence_objective(
                            model,
                            config,
                            rebrac_state,
                            rebrac_config,
                            belief,
                            current,
                            actions,
                            noises,
                        )
                        for model, config, belief in zip(
                            worlds, dreamer_configs, beliefs, strict=True
                        )
                    ]
                )
                return ensemble_relative_risk_score(
                    candidate_members,
                    reference_members,
                    RelativePessimismConfig(
                        risk_coefficient=float(relative_risk_coefficient)
                    ),
                )

        if optimizer == "gradient":
            result = gradient_action_sequence_search(
                objective, domain, initial, gradient_config
            )
            actual_proposal_sequences = result.action_sequence[None]
        else:
            result = cem_action_sequence_search(
                objective, domain, initial, proposal_bank, cem_config
            )
            # The first CEM population is an actual evaluated proposal set and
            # is available without changing the frozen search implementation.
            first_population = (
                jnp.clip(
                    cem_config.initial_std * proposal_bank[0],
                    domain.residual_minimum[None],
                    domain.residual_maximum[None],
                )
                .at[0]
                .set(jnp.zeros_like(reference))
            )
            actual_proposal_sequences = reference[None] + first_population
        return result, reference, actual_proposal_sequences

    # Jitting one complete search prevents Python dispatch from dominating the
    # equal-objective-call comparison. Model/config objects are closed constants.
    search_function = jax.jit(search)
    imagine_function = jax.jit(
        lambda belief, current, action_sequences, noises: (
            _imagined_action_sequence_points(
                worlds[0],
                dreamer_configs[0],
                belief,
                current,
                action_sequences,
                noises,
            )
        )
    )
    observe_functions = [
        jax.jit(
            lambda observation, previous_action, belief, key, model=model, config=config: observe_step(
                model, observation, previous_action, belief, key, config
            )[
                0
            ]
        )
        for model, config in zip(worlds, dreamer_configs, strict=True)
    ]
    returns: list[float] = []
    sequence_names = (
        "actions",
        "observations",
        "rewards",
        "continuations",
        "is_last",
        "objective_before",
        "objective_after",
        "objective_improvement",
        "residual_norm",
        "objective_evaluations",
    )
    sequences: dict[str, list[np.ndarray]] = {name: [] for name in sequence_names}
    imagined_points: dict[str, list[np.ndarray]] = {
        f"{scope}_{kind}": []
        for scope in ("proposal", "best", "reference")
        for kind in ("observations", "actions")
    }
    warmed = False
    total_timed_seconds = 0.0
    timed_steps = 0
    for evaluation_seed in evaluation_seeds:
        environment = DMCAdapter(TASK, seed=int(evaluation_seed), action_repeat=1)
        episode: dict[str, list[Any]] = {name: [] for name in sequence_names}
        try:
            observation = environment.reset()
            beliefs = [initial_state(config, 1) for config in dreamer_configs]
            previous_action = jnp.zeros((1, rebrac_config.action_dim), jnp.float32)
            # Common random numbers isolate between-model disagreement from
            # posterior sampling noise.  In particular, reordering the focal
            # world in an ensemble must not silently assign it a new draw.
            common_posterior_key = benchmark.derive_jax_key(
                "actor-gap-action-sequence-posterior-crn",
                TASK,
                actor_seed,
                evaluation_seed,
            )
            posterior_keys = [common_posterior_key] * len(worlds)
            noise_key = benchmark.derive_jax_key(
                "actor-gap-action-sequence-noise",
                TASK,
                world_seed,
                actor_seed,
                evaluation_seed,
            )
            proposal_key = benchmark.derive_jax_key(
                "actor-gap-cem-proposals",
                TASK,
                world_seed,
                actor_seed,
                evaluation_seed,
            )
            for step in range(maximum_steps):
                current = jnp.asarray(observation[None], dtype=jnp.float32)
                noises = jax.random.normal(
                    jax.random.fold_in(noise_key, step),
                    (
                        ACTION_SEQUENCE_PARTICLES,
                        CONTROLLER_HORIZON,
                        dreamer_configs[0].stochastic_dim,
                    ),
                )
                proposal_bank = jax.random.normal(
                    jax.random.fold_in(proposal_key, step),
                    (
                        cem_config.iterations,
                        cem_config.population,
                        CONTROLLER_HORIZON,
                        rebrac_config.action_dim,
                    ),
                )
                if not warmed:
                    warm_beliefs = [
                        function(
                            current,
                            previous_action,
                            belief,
                            jax.random.fold_in(key, step),
                        )
                        for function, belief, key in zip(
                            observe_functions, beliefs, posterior_keys, strict=True
                        )
                    ]
                    warm_result, warm_reference, warm_proposals = search_function(
                        tuple(warm_beliefs), current, noises, proposal_bank
                    )
                    jax.block_until_ready(warm_result.action_sequence)
                    jax.block_until_ready(warm_proposals)
                    for warm_sequences in (
                        warm_proposals,
                        warm_result.action_sequence[None],
                        warm_reference[None],
                    ):
                        warm_observations, warm_actions = imagine_function(
                            warm_beliefs[0],
                            current,
                            warm_sequences,
                            noises[:IMAGINED_COVERAGE_PARTICLES],
                        )
                        jax.block_until_ready(warm_observations)
                        jax.block_until_ready(warm_actions)
                    warmed = True
                started = time.perf_counter()
                beliefs = [
                    function(
                        current,
                        previous_action,
                        belief,
                        jax.random.fold_in(key, step),
                    )
                    for function, belief, key in zip(
                        observe_functions, beliefs, posterior_keys, strict=True
                    )
                ]
                result, reference_sequence, proposal_sequences = search_function(
                    tuple(beliefs), current, noises, proposal_bank
                )
                action = result.action_sequence[0]
                jax.block_until_ready(action)
                elapsed = time.perf_counter() - started
                total_timed_seconds += elapsed
                timed_steps += 1
                host_action = np.asarray(jax.device_get(action), dtype=np.float32)
                transition = environment.step(host_action)
                episode["actions"].append(host_action)
                episode["observations"].append(np.asarray(observation, np.float32))
                episode["rewards"].append(float(transition.reward))
                episode["continuations"].append(float(transition.continuation))
                episode["is_last"].append(bool(transition.is_last))
                episode["objective_before"].append(float(result.initial_objective))
                episode["objective_after"].append(float(result.final_objective))
                episode["objective_improvement"].append(
                    float(result.objective_improvement)
                )
                episode["residual_norm"].append(
                    float(jnp.linalg.norm(result.proposal.residuals))
                )
                episode["objective_evaluations"].append(
                    int(result.objective_evaluations)
                )
                if step % IMAGINED_COVERAGE_STEP_INTERVAL == 0:
                    coverage_noises = noises[:IMAGINED_COVERAGE_PARTICLES]
                    for scope, action_sequences in (
                        ("proposal", proposal_sequences),
                        ("best", result.action_sequence[None]),
                        ("reference", reference_sequence[None]),
                    ):
                        imagined_observations, imagined_actions = imagine_function(
                            beliefs[0], current, action_sequences, coverage_noises
                        )
                        imagined_points[f"{scope}_observations"].append(
                            np.asarray(
                                jax.device_get(imagined_observations), dtype=np.float32
                            ).reshape(-1, *dreamer_configs[0].observation_shape)
                        )
                        imagined_points[f"{scope}_actions"].append(
                            np.asarray(
                                jax.device_get(imagined_actions), dtype=np.float32
                            ).reshape(-1, rebrac_config.action_dim)
                        )
                observation = transition.observation
                previous_action = action[None]
                if transition.is_last:
                    break
        finally:
            environment.close()
        returns.append(float(np.sum(np.asarray(episode["rewards"], np.float64))))
        for name in sequence_names:
            dtype = (
                np.bool_
                if name == "is_last"
                else np.int32 if name == "objective_evaluations" else np.float32
            )
            sequences[name].append(np.asarray(episode[name], dtype=dtype))
    trace: dict[str, np.ndarray] = {}
    lengths: np.ndarray | None = None
    for name in sequence_names:
        trailing = {
            "actions": (rebrac_config.action_dim,),
            "observations": dreamer_configs[0].observation_shape,
        }.get(name, ())
        values, local_lengths = dependency._pad_sequences(
            sequences[name], trailing_shape=trailing, dtype=sequences[name][0].dtype
        )
        if lengths is None:
            lengths = local_lengths
        elif not np.array_equal(lengths, local_lengths):
            raise RuntimeError("action-sequence telemetry lengths differ")
        trace[name] = values
    assert lengths is not None
    trace["lengths"] = lengths
    trace["evaluation_seeds"] = np.asarray(evaluation_seeds, dtype=np.uint32)
    for name, values in imagined_points.items():
        if not values:
            raise RuntimeError("live planner produced no imagined coverage points")
        trace[f"imagined_{name}"] = np.concatenate(values, axis=0)
    if tuple(benchmark._tree_digest(world) for world in worlds) != digests:
        raise RuntimeError("action-sequence search changed a frozen world model")
    if benchmark._tree_digest(rebrac_state) != rebrac_digest:
        raise RuntimeError("action-sequence search changed frozen ReBRAC")
    timing = {
        "timed_steps": float(timed_steps),
        "total_timed_seconds": total_timed_seconds,
        "discarded_compile_warmup_steps": 1.0,
        "mean_milliseconds_per_step": (
            1000.0 * total_timed_seconds / timed_steps if timed_steps else 0.0
        ),
    }
    return returns, trace, timing


def _model_cell_for(
    manifest: Mapping[str, Any],
    family: str,
    world_seed: int,
    actor_seed: int | None,
) -> Mapping[str, Any]:
    expected_actor = actor_seed if family in ACTOR_CONDITIONED_MODEL_FAMILIES else None
    rows = [
        row
        for row in manifest["model_cells"]
        if row["family"] == family
        and int(row["world_model_seed"]) == int(world_seed)
        and row.get("actor_seed") == expected_actor
    ]
    if len(rows) != 1:
        raise ValueError("trained model cell is absent or duplicated")
    return rows[0]


def _model_training_inputs(
    output_root: str | Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, Mapping[str, np.ndarray], Any, Mapping[str, Any]]:
    from .actor_gap_model_training import ModelCellSpec, ModelTrainingConfig

    world_seed = int(cell["world_model_seed"])
    actor_seed = cell.get("actor_seed")
    load_actor_seed = ACTOR_SEEDS[0] if actor_seed is None else int(actor_seed)
    world, dreamer, rebrac_state, rebrac_config, source = _load_control_inputs(
        manifest, world_seed, load_actor_seed
    )
    if not math.isclose(
        float(rebrac_config.discount), PLANNER_DISCOUNT, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("dependency planner discount differs from the frozen study")
    dataset_path = Path(str(source["dataset"])).resolve(strict=True)
    if benchmark.file_sha256(dataset_path) != source["dataset_file_sha256"]:
        raise ValueError("model-training dataset digest differs before load")
    replay = benchmark.load_npz(dataset_path)
    if benchmark.array_sha256(replay) != source["dataset_sha256"]:
        raise ValueError("model-training dataset array payload differs")
    calibration = _calibration_row(output_root, manifest, world_seed, load_actor_seed)
    settings = manifest["model_training"]
    objective = ModelTrainingConfig(
        updates=int(settings["updates"]),
        batch_size=int(settings["batch_size"]),
        sequence_length=int(settings["sequence_length"]),
        burn_in=int(dreamer.burn_in),
        learning_rate=float(settings["learning_rate"]),
        grad_clip=float(settings["grad_clip"]),
        approximate_policy_std=APPROXIMATE_POLICY_STD,
        tilt_eta=float(calibration["selected_tilt_eta"]),
        cvaml_samples=int(settings["cvaml_samples"]),
        cvaml_scale=float(settings["cvaml_scale"]),
        planner_discount=float(settings["planner_discount"]),
        bellman_target_scale=float(calibration["bellman_target_scale"]),
        maximum_chunk_horizon=max(CHUNK_HORIZONS),
        endpoint_scale=float(settings["chunk_endpoint_scale"]),
    )
    spec = ModelCellSpec(
        index=int(cell["index"]),
        cell_id=str(cell["cell_id"]),
        family=str(cell["family"]),
        world_model_seed=world_seed,
        actor_seed=None if actor_seed is None else int(actor_seed),
    )
    control_state = (
        rebrac_state if cell["family"] in ACTOR_CONDITIONED_MODEL_FAMILIES else None
    )
    return spec, objective, world, dreamer, replay, control_state, source


def train_model_cell(
    dependency_root: str | Path,
    output_root: str | Path,
    index: int,
    *,
    submission_authorization: Mapping[str, Any],
    model_updates: int = MODEL_UPDATES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    from .actor_gap_model_training import (
        deterministic_payload_identity,
        train_model_cell_payload,
    )

    manifest = write_manifest(
        dependency_root,
        output_root,
        model_updates=model_updates,
        evaluation_episodes=evaluation_episodes,
    )
    root = Path(output_root)
    cell = _cell(manifest, "model", index)
    current_authorization = _validate_retained_submission_authorization(
        root, manifest, cell, "model", submission_authorization
    )
    result_path = root / str(cell["result_path"])
    if result_path.is_file():
        if current_authorization["mode"] != "verification_only":
            raise ValueError(
                "retained model evidence requires verification-only submission"
            )
        return read_json(result_path)
    if current_authorization["mode"] != "new":
        raise ValueError("new model evidence requires a new-mode submission")
    creation_scheduler, creation_receipt, creation_runtime = (
        _live_array_execution_binding(
            root, manifest, cell, "model", current_authorization
        )
    )
    _reject_partial_artifacts(
        result_path,
        root / str(cell["schedule_path"]),
        root / str(cell["checkpoint_path"]),
    )
    diagnostic_stage = root / "verified/stage-diagnostics.json"
    if not diagnostic_stage.is_file():
        raise ValueError("model training requires the authenticated diagnostic stage")
    _validate_stage_marker(root, manifest, "diagnostic")
    spec, objective, world, dreamer, replay, control_state, source = (
        _model_training_inputs(root, manifest, cell)
    )
    total_started = time.perf_counter()
    payload = train_model_cell_payload(
        spec,
        world,
        dreamer,
        replay,
        objective,
        task=TASK,
        control_state=control_state,
        jit=True,
    )
    schedule = payload.pop("schedule")
    checkpoint = payload.pop("checkpoint")
    schedule_path = root / str(cell["schedule_path"])
    checkpoint_payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "schema_version": MODEL_SCHEMA,
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "creation_submission_authorization": current_authorization,
        "creation_scheduler_provenance": creation_scheduler,
        "creation_submission_receipt": creation_receipt,
        "engine_checkpoint": checkpoint,
    }
    checkpoint_path = root / str(cell["checkpoint_path"])
    identity_payload = {**payload, "schedule": schedule, "checkpoint": checkpoint}
    if len({result_path.parent, schedule_path.parent, checkpoint_path.parent}) != 1:
        raise ValueError("model artifacts do not share one bundle directory")
    with _staged_artifact_directory(result_path.parent) as staging:
        staged_schedule_path = staging / schedule_path.name
        staged_checkpoint_path = staging / checkpoint_path.name
        staged_result_path = staging / result_path.name
        _write_npz_exclusive(staged_schedule_path, schedule)
        _write_pickle_atomic(staged_checkpoint_path, checkpoint_payload)
        result = {
            "schema_version": MODEL_SCHEMA,
            "status": "complete",
            "source_commit": manifest["source_commit"],
            "manifest_sha256": manifest["manifest_sha256"],
            "cell_id": cell["cell_id"],
            "cell_index": int(cell["index"]),
            "creation_submission_authorization": current_authorization,
            "creation_scheduler_provenance": creation_scheduler,
            "creation_submission_receipt": creation_receipt,
            "task": TASK,
            "family": cell["family"],
            "world_model_seed": int(cell["world_model_seed"]),
            "actor_seed": cell.get("actor_seed"),
            "engine_result": payload,
            "deterministic_payload_identity": deterministic_payload_identity(
                identity_payload
            ),
            "schedule_file_sha256": benchmark.file_sha256(staged_schedule_path),
            "schedule_sha256": benchmark.array_sha256(schedule),
            "checkpoint_sha256": benchmark.file_sha256(staged_checkpoint_path),
            "dataset_file_sha256": source["dataset_file_sha256"],
            "dataset_sha256": source["dataset_sha256"],
            "reward_checkpoint_sha256": source["reward_checkpoint_sha256"],
            "control_checkpoint_sha256": (
                None
                if cell.get("actor_seed") is None
                else benchmark.file_sha256(
                    Path(str(manifest["dependency_root"]))
                    / str(
                        _dependency_rebrac_cell(
                            _dependency_manifest(manifest),
                            int(cell["world_model_seed"]),
                            int(cell["actor_seed"]),
                        )["checkpoint_path"]
                    )
                )
            ),
            "wall_seconds": time.perf_counter() - total_started,
            "wall_seconds_scope": (
                "result_creation_including_discarded_compile_warmup_and_artifact_writes"
            ),
            "retained_training_wall_seconds": float(
                payload["wall_seconds_excluding_discarded_compile_warmup"]
            ),
            "runtime": creation_runtime,
        }
        if not _finite_tree(result):
            raise FloatingPointError("model result is not finite")
        _write_json_exclusive(staged_result_path, result)
    return result


def verify_model_cell(
    output_root: str | Path,
    index: int,
    *,
    strict_replay: bool = True,
    submission_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if strict_replay is not True:
        raise ValueError("model certification requires strict replay")
    from .actor_gap_model_training import (
        deterministic_payload_identity,
        replay_model_cell_payload,
        require_canonical_training_schedule,
        schedule_sha256,
    )

    verification_started = time.perf_counter()
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell(manifest, "model", index)
    verification_authorization = _validate_retained_submission_authorization(
        root, manifest, cell, "model", submission_authorization
    )
    verification_scheduler, verification_receipt, verification_runtime = (
        _live_array_execution_binding(
            root, manifest, cell, "model", verification_authorization
        )
    )
    result_path = root / str(cell["result_path"])
    schedule_path = root / str(cell["schedule_path"])
    checkpoint_path = root / str(cell["checkpoint_path"])
    result = read_json(result_path)
    creation_authorization, creation_scheduler, _, _ = (
        _validate_retained_creation_binding(root, manifest, cell, "model", result)
    )
    _require_distinct_replay_processes(creation_scheduler, verification_scheduler)
    if (
        result.get("schema_version") != MODEL_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("cell_id") != cell["cell_id"]
        or result.get("cell_index") != int(cell["index"])
        or result.get("family") != cell["family"]
        or result.get("world_model_seed") != int(cell["world_model_seed"])
        or result.get("actor_seed") != cell.get("actor_seed")
        or result.get("schedule_file_sha256") != benchmark.file_sha256(schedule_path)
        or result.get("checkpoint_sha256") != benchmark.file_sha256(checkpoint_path)
        or not _finite_tree(result)
    ):
        raise ValueError("model cell result contract differs")
    # Hash before deserializing.
    checkpoint_payload = _load_pickle(checkpoint_path)
    if (
        checkpoint_payload.get("schema_version") != MODEL_SCHEMA
        or checkpoint_payload.get("source_commit") != manifest["source_commit"]
        or checkpoint_payload.get("manifest_sha256") != manifest["manifest_sha256"]
        or checkpoint_payload.get("cell_id") != cell["cell_id"]
        or checkpoint_payload.get("creation_submission_authorization")
        != creation_authorization
        or checkpoint_payload.get("creation_scheduler_provenance")
        != result.get("creation_scheduler_provenance")
        or checkpoint_payload.get("creation_submission_receipt")
        != result.get("creation_submission_receipt")
    ):
        raise ValueError("model checkpoint identity differs")
    retained_schedule = benchmark.load_npz(schedule_path)
    if (
        benchmark.array_sha256(retained_schedule) != result["schedule_sha256"]
        or result["engine_result"].get("schedule_sha256") != result["schedule_sha256"]
    ):
        raise ValueError("model schedule array payload differs")
    spec, objective, world, dreamer, replay, control_state, source = (
        _model_training_inputs(root, manifest, cell)
    )
    schedule = require_canonical_training_schedule(
        retained_schedule,
        replay,
        task=TASK,
        world_model_seed=spec.world_model_seed,
        objective=objective,
    )
    if (
        schedule_sha256(schedule) != result["schedule_sha256"]
        or benchmark.array_sha256(schedule) != result["schedule_sha256"]
    ):
        raise ValueError("canonical model schedule digest differs")
    expected_control_sha256 = (
        None
        if cell.get("actor_seed") is None
        else benchmark.file_sha256(
            Path(str(manifest["dependency_root"]))
            / str(
                _dependency_rebrac_cell(
                    _dependency_manifest(manifest),
                    int(cell["world_model_seed"]),
                    int(cell["actor_seed"]),
                )["checkpoint_path"]
            )
        )
    )
    if (
        result.get("dataset_file_sha256") != source["dataset_file_sha256"]
        or result.get("dataset_sha256") != source["dataset_sha256"]
        or result.get("reward_checkpoint_sha256") != source["reward_checkpoint_sha256"]
        or result.get("control_checkpoint_sha256") != expected_control_sha256
    ):
        raise ValueError("model source artifact binding differs")
    retained = {
        **result["engine_result"],
        "schedule": schedule,
        "checkpoint": checkpoint_payload["engine_checkpoint"],
    }
    if result.get("deterministic_payload_identity") != (
        deterministic_payload_identity(retained)
    ):
        raise ValueError("model deterministic identity differs")
    replay_record = replay_model_cell_payload(
        retained,
        spec,
        world,
        dreamer,
        replay,
        objective,
        task=TASK,
        control_state=control_state,
        jit=True,
    )
    if replay_record.get("strict_deterministic_replay") is not True:
        raise ValueError("model strict deterministic replay was not certified")
    return _write_verified_marker(
        root,
        manifest,
        cell,
        "model",
        result_path=result_path,
        extra={
            "checkpoint_sha256": result["checkpoint_sha256"],
            "schedule_file_sha256": result["schedule_file_sha256"],
            "strict_deterministic_replay": True,
            "trainable_subtree": result["engine_result"]["trainable_subtree"],
            "verification_submission_authorization": verification_authorization,
            "verification_scheduler_provenance": verification_scheduler,
            "verification_submission_receipt": verification_receipt,
            "verification_runtime": verification_runtime,
            "strict_replay_wall_seconds": time.perf_counter() - verification_started,
        },
    )


def _load_trained_model(
    output_root: str | Path,
    manifest: Mapping[str, Any],
    family: str,
    world_seed: int,
    actor_seed: int,
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
    root = Path(output_root)
    cell = _model_cell_for(manifest, family, world_seed, actor_seed)
    marker = _validate_cell_marker(root, manifest, cell, "model")
    result = read_json(root / str(cell["result_path"]))
    checkpoint_path = root / str(cell["checkpoint_path"])
    if (
        marker.get("status") != "verified"
        or marker.get("stage") != "model"
        or marker.get("strict_deterministic_replay") is not True
        or marker.get("checkpoint_sha256") != benchmark.file_sha256(checkpoint_path)
        or result.get("checkpoint_sha256") != marker.get("checkpoint_sha256")
    ):
        raise ValueError("trained model lacks a strict authenticated marker")
    checkpoint_payload = _load_pickle(checkpoint_path)
    if (
        checkpoint_payload.get("checkpoint_version") != CHECKPOINT_VERSION
        or checkpoint_payload.get("schema_version") != MODEL_SCHEMA
        or checkpoint_payload.get("source_commit") != manifest["source_commit"]
        or checkpoint_payload.get("manifest_sha256") != manifest["manifest_sha256"]
        or checkpoint_payload.get("cell_id") != cell["cell_id"]
    ):
        raise ValueError("trained model checkpoint identity differs")
    engine_checkpoint = checkpoint_payload["engine_checkpoint"]
    source_actor = ACTOR_SEEDS[0] if cell.get("actor_seed") is None else actor_seed
    world, dreamer, _, _, _ = _load_control_inputs(manifest, world_seed, source_actor)
    if engine_checkpoint["trainable_subtree"] == "prior":
        model = dict(world)
        model["prior"] = engine_checkpoint["params"]
        return model, dreamer, engine_checkpoint, result
    if engine_checkpoint["trainable_subtree"] != "endpoint_imf":
        raise ValueError("unknown model checkpoint trainable subtree")
    return world, dreamer, engine_checkpoint, result


def _endpoint_action_sequence_objective(
    endpoint_checkpoint: Mapping[str, Any],
    source_world: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    current_observation: Any,
    action_sequence: Any,
    noises: Any,
    *,
    direct_any_step: bool,
    behavior_distance_threshold: float,
) -> tuple[Any, Any]:
    import jax.numpy as jnp
    from imf_dreamer_jax import rebrac_actor, rebrac_critics
    from .actor_gap_model_training import sample_endpoint_observation
    from imf_dreamer_jax.world_model import (
        predict_state_action_reward_from_observation,
    )

    particles = noises.shape[0]
    maximum_horizon = int(endpoint_checkpoint["maximum_chunk_horizon"])
    mean = jnp.asarray(endpoint_checkpoint["observation_mean"], jnp.float32)
    std = jnp.asarray(endpoint_checkpoint["observation_std"], jnp.float32)
    start = jnp.broadcast_to(
        current_observation,
        (particles, *dreamer_config.observation_shape),
    )

    def predict_endpoint(observation: Any, prefix_length: int, noise: Any) -> Any:
        padded = jnp.zeros(
            (maximum_horizon, dreamer_config.action_dim), dtype=action_sequence.dtype
        )
        if direct_any_step:
            padded = padded.at[:prefix_length].set(action_sequence[:prefix_length])
            endpoint_start = start
            horizon = prefix_length
        else:
            padded = padded.at[0].set(action_sequence[prefix_length - 1])
            endpoint_start = observation
            horizon = 1
        padded_batch = jnp.broadcast_to(
            padded, (particles, maximum_horizon, dreamer_config.action_dim)
        )
        return sample_endpoint_observation(
            endpoint_checkpoint["params"],
            endpoint_start,
            padded_batch,
            horizon,
            dreamer_config,
            observation_mean=mean,
            observation_std=std,
            noise=noise,
        )

    observation = start
    stage = jnp.zeros((particles,), jnp.float32)
    maximum_behavior_distance = jnp.asarray(0.0, jnp.float32)
    for step in range(action_sequence.shape[0]):
        action = jnp.broadcast_to(
            action_sequence[step], (particles, dreamer_config.action_dim)
        )
        policy_action = rebrac_actor(rebrac_state.actor, observation)
        step_distance = jnp.mean(
            jnp.sqrt(jnp.mean(jnp.square(action - policy_action), axis=-1))
        )
        maximum_behavior_distance = jnp.maximum(
            maximum_behavior_distance, step_distance
        )
        reward = predict_state_action_reward_from_observation(
            source_world["reward_transition"], observation, action, dreamer_config
        )
        stage = stage + (float(rebrac_config.discount) ** step) * reward
        observation = predict_endpoint(observation, step + 1, noises[:, step])
    terminal_action = rebrac_actor(rebrac_state.actor, observation)
    terminal_q = jnp.min(
        rebrac_critics(rebrac_state.critics, observation, terminal_action), axis=0
    )
    objective = jnp.mean(stage) + (
        float(rebrac_config.discount) ** action_sequence.shape[0]
    ) * jnp.mean(terminal_q)
    accepted = maximum_behavior_distance <= behavior_distance_threshold
    filtered = jnp.where(
        accepted,
        objective,
        jnp.asarray(-1e6, objective.dtype) - maximum_behavior_distance,
    )
    return filtered, maximum_behavior_distance


def _endpoint_imagined_action_sequence_points(
    endpoint_checkpoint: Mapping[str, Any],
    dreamer_config: Any,
    current_observation: Any,
    action_sequences: Any,
    noises: Any,
    *,
    direct_any_step: bool,
) -> tuple[Any, Any]:
    """Return the exact observation-action queries made by an endpoint planner."""

    import jax
    import jax.numpy as jnp
    from .actor_gap_model_training import sample_endpoint_observation

    sequences = jnp.asarray(action_sequences)
    particle_noises = jnp.asarray(noises)
    if sequences.ndim != 3 or particle_noises.ndim != 3:
        raise ValueError("endpoint imagined coverage inputs have invalid ranks")
    if sequences.shape[1] != particle_noises.shape[1]:
        raise ValueError("endpoint action sequences and noises must share horizon")
    particles = particle_noises.shape[0]
    maximum_horizon = int(endpoint_checkpoint["maximum_chunk_horizon"])
    mean = jnp.asarray(endpoint_checkpoint["observation_mean"], jnp.float32)
    std = jnp.asarray(endpoint_checkpoint["observation_std"], jnp.float32)
    start = jnp.broadcast_to(
        current_observation,
        (particles, *dreamer_config.observation_shape),
    )

    def one_sequence(action_sequence: Any) -> tuple[Any, Any]:
        observation = start
        observation_rows = []
        action_rows = []
        for step in range(action_sequence.shape[0]):
            action = jnp.broadcast_to(
                action_sequence[step], (particles, dreamer_config.action_dim)
            )
            observation_rows.append(observation)
            action_rows.append(action)
            padded = jnp.zeros(
                (maximum_horizon, dreamer_config.action_dim),
                dtype=action_sequence.dtype,
            )
            if direct_any_step:
                padded = padded.at[: step + 1].set(action_sequence[: step + 1])
                endpoint_start = start
                endpoint_horizon = step + 1
            else:
                padded = padded.at[0].set(action_sequence[step])
                endpoint_start = observation
                endpoint_horizon = 1
            observation = sample_endpoint_observation(
                endpoint_checkpoint["params"],
                endpoint_start,
                jnp.broadcast_to(
                    padded,
                    (particles, maximum_horizon, dreamer_config.action_dim),
                ),
                endpoint_horizon,
                dreamer_config,
                observation_mean=mean,
                observation_std=std,
                noise=particle_noises[:, step],
            )
        return jnp.stack(observation_rows, axis=1), jnp.stack(action_rows, axis=1)

    return jax.vmap(one_sequence)(sequences)


def _apply_endpoint_behavior_filter(
    candidate_action: Any,
    reference_action: Any,
    behavior_distance: Any,
    behavior_distance_threshold: float,
) -> tuple[Any, Any, Any, Any]:
    """Make the endpoint sequence filter govern the action actually executed."""

    import jax.numpy as jnp

    accepted = behavior_distance <= behavior_distance_threshold
    action = jnp.where(accepted, candidate_action, reference_action)
    executed_distance = jnp.sqrt(jnp.mean(jnp.square(action - reference_action)))
    executed_accepted = executed_distance <= behavior_distance_threshold
    return action, accepted, executed_distance, executed_accepted


def _select_endpoint_executed_action(
    endpoint_checkpoint: Mapping[str, Any],
    source_world: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    current: Any,
    search_result: Any,
    reference: Any,
    noises: Any,
    *,
    direct_any_step: bool,
    behavior_distance_threshold: float,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Apply the exact post-search decision path used by the environment."""

    candidate_action = search_result.action_sequence[0]
    _, behavior_distance = _endpoint_action_sequence_objective(
        endpoint_checkpoint,
        source_world,
        dreamer_config,
        rebrac_state,
        rebrac_config,
        current,
        search_result.action_sequence,
        noises,
        direct_any_step=direct_any_step,
        behavior_distance_threshold=behavior_distance_threshold,
    )
    reference_action = reference[0]
    (
        action,
        behavior_accepted,
        executed_behavior_distance,
        executed_behavior_accepted,
    ) = _apply_endpoint_behavior_filter(
        candidate_action,
        reference_action,
        behavior_distance,
        behavior_distance_threshold,
    )
    return (
        action,
        candidate_action,
        reference_action,
        behavior_distance,
        behavior_accepted,
        executed_behavior_distance,
        executed_behavior_accepted,
    )


def _run_endpoint_action_sequence_arm(
    source_world: Any,
    dreamer_config: Any,
    endpoint_checkpoint: Mapping[str, Any],
    rebrac_state: Any,
    rebrac_config: Any,
    *,
    world_seed: int,
    actor_seed: int,
    evaluation_seeds: list[int],
    maximum_steps: int,
    direct_any_step: bool,
    behavior_distance_threshold: float,
) -> tuple[list[float], dict[str, np.ndarray], dict[str, float]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state, observe_step
    from imf_dreamer_jax.robust_flowmpc import (
        ActionSequenceDomainConfig,
        ActionSequenceProposal,
        CEMActionSequenceConfig,
        cem_action_sequence_search,
        make_action_sequence_domain,
    )
    from .dmc import DMCAdapter

    source_digest = benchmark._tree_digest(source_world)
    endpoint_digest = benchmark._tree_digest(endpoint_checkpoint["params"])
    rebrac_digest = benchmark._tree_digest(rebrac_state)
    domain_config = ActionSequenceDomainConfig(
        horizon=CONTROLLER_HORIZON,
        action_dim=rebrac_config.action_dim,
        residual_limit=ACTION_SEQUENCE_RESIDUAL_LIMIT,
    )
    cem_config = CEMActionSequenceConfig(
        iterations=2,
        population=4,
        elite_count=2,
        initial_std=0.10,
        minimum_std=0.01,
        maximum_std=0.10,
    )

    def search(belief: Any, current: Any, noises: Any, proposals: Any) -> Any:
        reference = _reference_action_sequence(
            source_world,
            dreamer_config,
            rebrac_state.actor,
            belief,
            current,
        )
        domain = make_action_sequence_domain(reference, domain_config)
        initial = ActionSequenceProposal(jnp.zeros_like(reference))

        def objective(actions: Any) -> Any:
            return _endpoint_action_sequence_objective(
                endpoint_checkpoint,
                source_world,
                dreamer_config,
                rebrac_state,
                rebrac_config,
                current,
                actions,
                noises,
                direct_any_step=direct_any_step,
                behavior_distance_threshold=behavior_distance_threshold,
            )[0]

        result = cem_action_sequence_search(
            objective, domain, initial, proposals, cem_config
        )
        first_population = (
            jnp.clip(
                cem_config.initial_std * proposals[0],
                domain.residual_minimum[None],
                domain.residual_maximum[None],
            )
            .at[0]
            .set(jnp.zeros_like(reference))
        )
        return result, reference, reference[None] + first_population

    search_function = jax.jit(search)
    imagine_function = jax.jit(
        lambda current, action_sequences, noises: (
            _endpoint_imagined_action_sequence_points(
                endpoint_checkpoint,
                dreamer_config,
                current,
                action_sequences,
                noises,
                direct_any_step=direct_any_step,
            )
        )
    )

    @jax.jit
    def observe_function(observation: Any, previous_action: Any, belief: Any, key: Any):
        return observe_step(
            source_world,
            observation,
            previous_action,
            belief,
            key,
            dreamer_config,
        )[0]

    returns: list[float] = []
    names = (
        "actions",
        "observations",
        "rewards",
        "continuations",
        "is_last",
        "objective_before",
        "objective_after",
        "objective_improvement",
        "residual_norm",
        "behavior_distance",
        "behavior_accepted",
        "reference_actions",
        "used_behavior_fallback",
        "executed_behavior_distance",
        "executed_behavior_accepted",
        "objective_evaluations",
    )
    sequences: dict[str, list[np.ndarray]] = {name: [] for name in names}
    imagined_points: dict[str, list[np.ndarray]] = {
        f"{scope}_{kind}": []
        for scope in ("proposal", "best", "reference")
        for kind in ("observations", "actions")
    }
    warmed = False
    timed_seconds = 0.0
    timed_steps = 0
    for evaluation_seed in evaluation_seeds:
        environment = DMCAdapter(TASK, seed=int(evaluation_seed), action_repeat=1)
        episode: dict[str, list[Any]] = {name: [] for name in names}
        try:
            observation = environment.reset()
            belief = initial_state(dreamer_config, 1)
            previous_action = jnp.zeros((1, dreamer_config.action_dim), jnp.float32)
            posterior_key = benchmark.derive_jax_key(
                "actor-gap-endpoint-posterior",
                TASK,
                world_seed,
                actor_seed,
                evaluation_seed,
            )
            noise_key = benchmark.derive_jax_key(
                "actor-gap-endpoint-noise",
                TASK,
                world_seed,
                actor_seed,
                evaluation_seed,
            )
            proposal_key = benchmark.derive_jax_key(
                "actor-gap-endpoint-cem",
                TASK,
                world_seed,
                actor_seed,
                evaluation_seed,
            )
            for step in range(maximum_steps):
                current = jnp.asarray(observation[None], dtype=jnp.float32)
                noises = jax.random.normal(
                    jax.random.fold_in(noise_key, step),
                    (
                        ACTION_SEQUENCE_PARTICLES,
                        CONTROLLER_HORIZON,
                        dreamer_config.observation_dim,
                    ),
                )
                proposals = jax.random.normal(
                    jax.random.fold_in(proposal_key, step),
                    (
                        cem_config.iterations,
                        cem_config.population,
                        CONTROLLER_HORIZON,
                        dreamer_config.action_dim,
                    ),
                )
                if not warmed:
                    warm_belief = observe_function(
                        current,
                        previous_action,
                        belief,
                        jax.random.fold_in(posterior_key, step),
                    )
                    warm, warm_reference, warm_proposals = search_function(
                        warm_belief, current, noises, proposals
                    )
                    warm_decision = _select_endpoint_executed_action(
                        endpoint_checkpoint,
                        source_world,
                        dreamer_config,
                        rebrac_state,
                        rebrac_config,
                        current,
                        warm,
                        warm_reference,
                        noises,
                        direct_any_step=direct_any_step,
                        behavior_distance_threshold=behavior_distance_threshold,
                    )
                    jax.block_until_ready(warm_decision[0])
                    jax.block_until_ready(warm_proposals)
                    for warm_sequences in (
                        warm_proposals,
                        warm.action_sequence[None],
                        warm_reference[None],
                    ):
                        warm_observations, warm_actions = imagine_function(
                            current,
                            warm_sequences,
                            noises[:IMAGINED_COVERAGE_PARTICLES],
                        )
                        jax.block_until_ready(warm_observations)
                        jax.block_until_ready(warm_actions)
                    warmed = True
                started = time.perf_counter()
                belief = observe_function(
                    current,
                    previous_action,
                    belief,
                    jax.random.fold_in(posterior_key, step),
                )
                result, reference, proposal_sequences = search_function(
                    belief, current, noises, proposals
                )
                (
                    action,
                    candidate_action,
                    reference_action,
                    behavior_distance,
                    behavior_accepted,
                    executed_behavior_distance,
                    executed_behavior_accepted,
                ) = _select_endpoint_executed_action(
                    endpoint_checkpoint,
                    source_world,
                    dreamer_config,
                    rebrac_state,
                    rebrac_config,
                    current,
                    result,
                    reference,
                    noises,
                    direct_any_step=direct_any_step,
                    behavior_distance_threshold=behavior_distance_threshold,
                )
                jax.block_until_ready(action)
                elapsed = time.perf_counter() - started
                timed_seconds += elapsed
                timed_steps += 1
                host_action = np.asarray(jax.device_get(action), dtype=np.float32)
                transition = environment.step(host_action)
                episode["actions"].append(host_action)
                episode["observations"].append(np.asarray(observation, np.float32))
                episode["rewards"].append(float(transition.reward))
                episode["continuations"].append(float(transition.continuation))
                episode["is_last"].append(bool(transition.is_last))
                episode["objective_before"].append(float(result.initial_objective))
                episode["objective_after"].append(float(result.final_objective))
                episode["objective_improvement"].append(
                    float(result.objective_improvement)
                )
                episode["residual_norm"].append(
                    float(jnp.linalg.norm(result.proposal.residuals))
                )
                distance = float(behavior_distance)
                episode["behavior_distance"].append(distance)
                accepted = bool(behavior_accepted)
                episode["behavior_accepted"].append(accepted)
                episode["reference_actions"].append(
                    np.asarray(jax.device_get(reference_action), dtype=np.float32)
                )
                episode["used_behavior_fallback"].append(not accepted)
                episode["executed_behavior_distance"].append(
                    float(executed_behavior_distance)
                )
                episode["executed_behavior_accepted"].append(
                    bool(executed_behavior_accepted)
                )
                episode["objective_evaluations"].append(
                    int(result.objective_evaluations)
                )
                if step % IMAGINED_COVERAGE_STEP_INTERVAL == 0:
                    coverage_noises = noises[:IMAGINED_COVERAGE_PARTICLES]
                    for scope, action_sequences in (
                        ("proposal", proposal_sequences),
                        ("best", result.action_sequence[None]),
                        ("reference", reference[None]),
                    ):
                        imagined_observations, imagined_actions = imagine_function(
                            current, action_sequences, coverage_noises
                        )
                        imagined_points[f"{scope}_observations"].append(
                            np.asarray(
                                jax.device_get(imagined_observations), dtype=np.float32
                            ).reshape(-1, *dreamer_config.observation_shape)
                        )
                        imagined_points[f"{scope}_actions"].append(
                            np.asarray(
                                jax.device_get(imagined_actions), dtype=np.float32
                            ).reshape(-1, dreamer_config.action_dim)
                        )
                observation = transition.observation
                previous_action = action[None]
                if transition.is_last:
                    break
        finally:
            environment.close()
        returns.append(float(np.sum(np.asarray(episode["rewards"], np.float64))))
        for name in names:
            dtype = (
                np.bool_
                if name
                in (
                    "is_last",
                    "behavior_accepted",
                    "used_behavior_fallback",
                    "executed_behavior_accepted",
                )
                else np.int32 if name == "objective_evaluations" else np.float32
            )
            sequences[name].append(np.asarray(episode[name], dtype=dtype))
    trace: dict[str, np.ndarray] = {}
    lengths: np.ndarray | None = None
    for name in names:
        trailing = {
            "actions": (dreamer_config.action_dim,),
            "observations": dreamer_config.observation_shape,
            "reference_actions": (dreamer_config.action_dim,),
        }.get(name, ())
        values, local_lengths = dependency._pad_sequences(
            sequences[name], trailing_shape=trailing, dtype=sequences[name][0].dtype
        )
        if lengths is None:
            lengths = local_lengths
        elif not np.array_equal(lengths, local_lengths):
            raise RuntimeError("endpoint-controller telemetry lengths differ")
        trace[name] = values
    assert lengths is not None
    trace["lengths"] = lengths
    trace["evaluation_seeds"] = np.asarray(evaluation_seeds, dtype=np.uint32)
    for name, values in imagined_points.items():
        if not values:
            raise RuntimeError("endpoint planner produced no imagined coverage points")
        trace[f"imagined_{name}"] = np.concatenate(values, axis=0)
    if (
        benchmark._tree_digest(source_world) != source_digest
        or benchmark._tree_digest(endpoint_checkpoint["params"]) != endpoint_digest
        or benchmark._tree_digest(rebrac_state) != rebrac_digest
    ):
        raise RuntimeError("endpoint controller changed a frozen input")
    timing = {
        "timed_steps": float(timed_steps),
        "total_timed_seconds": timed_seconds,
        "discarded_compile_warmup_steps": 1.0,
        "mean_milliseconds_per_step": (
            1000.0 * timed_seconds / timed_steps if timed_steps else 0.0
        ),
    }
    return returns, trace, timing


def _evaluation_model_family(arm: str) -> str | None:
    return {
        "F0_uniform_prior_unconstrained": "uniform_prior",
        "F1_uniform_prior_trust": "uniform_prior",
        "F2_policy_tilt_prior_unconstrained": "approximate_policy_tilt_prior",
        "F3_policy_tilt_prior_trust": "approximate_policy_tilt_prior",
        "V1_cvaml_value_prior_unconstrained": "proof_consistent_cvaml_value_prior",
        "K0_endpoint_h1_recursive_cem_filtered": "endpoint_h1_imf",
        "K1_endpoint_anystep_direct_cem_filtered": "endpoint_anystep_imf",
    }.get(arm)


def _masked_trace_values(trace: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in trace:
        return np.asarray([], dtype=np.float64)
    lengths = np.asarray(trace["lengths"], dtype=np.int64)
    values = np.asarray(trace[name])
    mask = np.arange(values.shape[1])[None] < lengths[:, None]
    if values.ndim > 2:
        mask = np.broadcast_to(mask[..., None], values.shape)
    return np.asarray(values[mask], dtype=np.float64)


def _compute_evaluation_cell(
    output_root: str | Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float]]:
    from .actor_gap_diagnostics import score_observation_action_coverage

    root = Path(output_root)
    world_seed = int(cell["world_model_seed"])
    actor_seed = int(cell["actor_seed"])
    arm = str(cell["arm"])
    evaluation_seeds = [int(value) for value in cell["evaluation_seeds"]]
    source_world, dreamer, rebrac_state, rebrac, source = _load_control_inputs(
        manifest, world_seed, actor_seed
    )
    calibration_arrays_path = root / str(manifest["calibration"]["arrays_path"])
    calibration_arrays = benchmark.load_npz(calibration_arrays_path)
    calibration = _calibration_row(root, manifest, world_seed, actor_seed)
    anchors = calibration_arrays[f"world_{world_seed}_anchor_observations"]
    family = _evaluation_model_family(arm)
    model_checkpoint_sha256: str | None = None
    model_world = source_world
    endpoint_checkpoint: Mapping[str, Any] | None = None
    if family is not None:
        model_world, dreamer, trained_checkpoint, model_result = _load_trained_model(
            root, manifest, family, world_seed, actor_seed
        )
        model_checkpoint_sha256 = str(model_result["checkpoint_sha256"])
        if trained_checkpoint["trainable_subtree"] == "endpoint_imf":
            endpoint_checkpoint = trained_checkpoint

    if arm.startswith(("A0_", "A1_", "A2_", "A3_", "F0_", "F1_", "F2_", "F3_", "V1_")):
        trust = arm in {
            "A1_persistent_trust",
            "A2_reset_trust",
            "A3_persistent_trust_heldout",
            "F1_uniform_prior_trust",
            "F3_policy_tilt_prior_trust",
        }
        persistence = "reset" if arm == "A2_reset_trust" else "persistent"
        heldout = arm == "A3_persistent_trust_heldout"
        returns, trace, timing = _run_flowmpc_arm(
            model_world,
            dreamer,
            rebrac_state,
            rebrac,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=evaluation_seeds,
            maximum_steps=int(manifest["maximum_environment_steps"]),
            trust=trust,
            persistence=persistence,
            heldout_acceptance_enabled=heldout,
            anchor_observations=anchors,
        )
        controller_family = "persistent_actor_gradient"
    elif arm in {
        "O1_recursive_action_sequence_gradient",
        "O2_recursive_action_sequence_cem",
    }:
        returns, trace, timing = _run_action_sequence_arm(
            [source_world],
            [dreamer],
            rebrac_state,
            rebrac,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=evaluation_seeds,
            maximum_steps=int(manifest["maximum_environment_steps"]),
            optimizer="gradient" if arm.startswith("O1_") else "cem",
            relative_risk_coefficient=None,
        )
        controller_family = "residual_action_sequence"
    elif arm in {
        "P0_recursive_cem_ensemble_relative_mean",
        "P1_recursive_cem_relative_pessimism",
    }:
        ordered_worlds = (world_seed,) + tuple(
            seed for seed in WORLD_SEEDS if seed != world_seed
        )
        ensemble_worlds = []
        ensemble_configs = []
        for member_seed in ordered_worlds:
            member_world, member_config, _, _, _ = _load_control_inputs(
                manifest, member_seed, actor_seed
            )
            ensemble_worlds.append(member_world)
            ensemble_configs.append(member_config)
        returns, trace, timing = _run_action_sequence_arm(
            ensemble_worlds,
            ensemble_configs,
            rebrac_state,
            rebrac,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=evaluation_seeds,
            maximum_steps=int(manifest["maximum_environment_steps"]),
            optimizer="cem",
            relative_risk_coefficient=(0.0 if arm.startswith("P0_") else 1.0),
        )
        controller_family = "ensemble_relative_residual_action_sequence"
    elif arm in {
        "K0_endpoint_h1_recursive_cem_filtered",
        "K1_endpoint_anystep_direct_cem_filtered",
    }:
        if endpoint_checkpoint is None:
            raise ValueError("endpoint arm lacks its authenticated endpoint model")
        returns, trace, timing = _run_endpoint_action_sequence_arm(
            source_world,
            dreamer,
            endpoint_checkpoint,
            rebrac_state,
            rebrac,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=evaluation_seeds,
            maximum_steps=int(manifest["maximum_environment_steps"]),
            direct_any_step=arm.startswith("K1_"),
            behavior_distance_threshold=float(calibration["behavior_distance_q95"]),
        )
        controller_family = "direct_endpoint_residual_action_sequence"
    else:
        raise ValueError(f"unknown roadmap evaluation arm: {arm}")

    old = _dependency_manifest(manifest)
    old_cell = _dependency_evaluation_cell(old, world_seed, actor_seed)
    old_trace, old_trace_path = _load_dependency_evaluation_trace(
        manifest,
        old,
        old_cell,
    )
    a0_replay_verified: bool | None = None
    if arm == "A0_persistent_unconstrained_replay":
        _assert_trace_subset_close(
            old_trace,
            trace,
            prefix="flowmpc",
            episodes=len(evaluation_seeds),
        )
        a0_replay_verified = True
    lengths = np.asarray(trace["lengths"], dtype=np.int64)
    mask = np.arange(trace["actions"].shape[1])[None] < lengths[:, None]
    coverage_object = _coverage_from_arrays(
        f"world_{world_seed}", calibration, calibration_arrays
    )
    coverage = score_observation_action_coverage(
        coverage_object,
        np.asarray(trace["observations"])[mask],
        np.asarray(trace["actions"])[mask],
        distance_chunk_size=256,
    ).to_dict()
    imagined_coverage = {}
    for scope in ("proposal", "best", "reference"):
        observation_key = f"imagined_{scope}_observations"
        action_key = f"imagined_{scope}_actions"
        if observation_key not in trace or action_key not in trace:
            raise ValueError(
                f"live controller lacks {scope} imagined coverage evidence"
            )
        scored = score_observation_action_coverage(
            coverage_object,
            np.asarray(trace[observation_key]),
            np.asarray(trace[action_key]),
            distance_chunk_size=256,
        ).to_dict()
        imagined_coverage[scope] = {
            key: value for key, value in scored.items() if key != "points"
        }

    def mean_or_zero(name: str) -> float:
        values = _masked_trace_values(trace, name)
        return float(np.mean(values)) if values.size else 0.0

    action_values = np.asarray(trace["actions"])[mask]
    core = {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "arm": arm,
        "controller_family": controller_family,
        "world_model_seed": world_seed,
        "actor_seed": actor_seed,
        "evaluation_seeds": evaluation_seeds,
        "episode_returns": [float(value) for value in returns],
        "mean_return": float(np.mean(returns)),
        "normalized_mean_return": float(np.mean(returns) / 1000.0),
        "action_saturation_fraction": float(np.mean(np.abs(action_values) >= 0.95)),
        "mean_objective_improvement": (
            mean_or_zero("objective_improvement")
            if "objective_improvement" in trace
            else mean_or_zero("objective_after") - mean_or_zero("objective_before")
        ),
        "nonnegative_objective_improvement_fraction": (
            float(
                np.mean(
                    _masked_trace_values(trace, "objective_after")
                    >= _masked_trace_values(trace, "objective_before")
                )
            )
            if "objective_before" in trace
            else 0.0
        ),
        "mean_gradient_norm": (
            mean_or_zero("gradient_norm") if "gradient_norm" in trace else None
        ),
        "mean_parameter_delta": (
            mean_or_zero("parameter_delta") if "parameter_delta" in trace else None
        ),
        "mean_anchor_action_drift": (
            mean_or_zero("anchor_drift") if "anchor_drift" in trace else None
        ),
        "mean_current_action_drift": (
            mean_or_zero("current_drift") if "current_drift" in trace else None
        ),
        "reference_budget_violation_fraction": (
            1.0 - mean_or_zero("within_reference_budgets")
            if "within_reference_budgets" in trace
            else None
        ),
        "mean_trust_backtracks": (
            mean_or_zero("backtracks") if "backtracks" in trace else None
        ),
        "frozen_reference_fallback_fraction": (
            mean_or_zero("used_reference_fallback")
            if "used_reference_fallback" in trace
            else None
        ),
        "mean_heldout_acceptance_improvement": (
            mean_or_zero("heldout_improvement")
            if arm == "A3_persistent_trust_heldout"
            else None
        ),
        "heldout_acceptance_fraction": (
            mean_or_zero("accepted") if arm == "A3_persistent_trust_heldout" else None
        ),
        "acceptance_fraction": (
            mean_or_zero("accepted")
            if "accepted" in trace
            else (
                mean_or_zero("behavior_accepted")
                if "behavior_accepted" in trace
                else None
            )
        ),
        "behavior_filter_fallback_fraction": (
            mean_or_zero("used_behavior_fallback")
            if "used_behavior_fallback" in trace
            else None
        ),
        "executed_behavior_violation_fraction": (
            1.0 - mean_or_zero("executed_behavior_accepted")
            if "executed_behavior_accepted" in trace
            else None
        ),
        "mean_behavior_distance": (
            mean_or_zero("behavior_distance") if "behavior_distance" in trace else None
        ),
        "objective_evaluations_per_step": (
            sorted(set(_masked_trace_values(trace, "objective_evaluations").tolist()))
            if "objective_evaluations" in trace
            else []
        ),
        "coverage": {key: value for key, value in coverage.items() if key != "points"},
        "imagined_coverage": imagined_coverage,
        "imagined_coverage_protocol": {
            "step_interval": IMAGINED_COVERAGE_STEP_INTERVAL,
            "shared_actual_planner_noise_particles": IMAGINED_COVERAGE_PARTICLES,
            "scopes": ["proposal", "best", "reference"],
            "proposal_semantics": (
                "pre_safeguard_actor_for_policy_adaptation; first_actual_cem_"
                "population_for_cem; retained_best_for_gradient"
            ),
            "best_semantics": (
                "post_safeguard_selected_actor_for_policy_adaptation; retained_"
                "search_best_for_action_sequence_controllers"
            ),
            "reference_semantics": "frozen_rebrac_policy_nominal_horizon5_plan",
            "ensemble_world_semantics": (
                "focal_world_model_only_for_ensemble_action_sequence_arms"
            ),
            "occupancy_scope": (
                "stage_reward_queries_only_terminal_critic_query_excluded"
            ),
        },
        "a0_dependency_replay_verified": a0_replay_verified,
        "source_reward_checkpoint_sha256": source["reward_checkpoint_sha256"],
        "source_rebrac_checkpoint_sha256": benchmark.file_sha256(
            Path(str(manifest["dependency_root"]))
            / str(
                _dependency_rebrac_cell(old, world_seed, actor_seed)["checkpoint_path"]
            )
        ),
        "model_checkpoint_sha256": model_checkpoint_sha256,
        "calibration_arrays_file_sha256": benchmark.file_sha256(
            calibration_arrays_path
        ),
        "dependency_trace_file_sha256": benchmark.file_sha256(old_trace_path),
        "discarded_pure_compile_warmup": True,
    }
    return core, trace, timing


def run_evaluation_cell(
    dependency_root: str | Path,
    output_root: str | Path,
    index: int,
    *,
    submission_authorization: Mapping[str, Any],
    model_updates: int = MODEL_UPDATES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    manifest = write_manifest(
        dependency_root,
        output_root,
        model_updates=model_updates,
        evaluation_episodes=evaluation_episodes,
    )
    root = Path(output_root)
    cell = _cell(manifest, "evaluation", index)
    current_authorization = _validate_retained_submission_authorization(
        root, manifest, cell, "evaluation", submission_authorization
    )
    result_path = root / str(cell["result_path"])
    if result_path.is_file():
        if current_authorization["mode"] != "verification_only":
            raise ValueError(
                "retained evaluation evidence requires verification-only submission"
            )
        return read_json(result_path)
    if current_authorization["mode"] != "new":
        raise ValueError("new evaluation evidence requires a new-mode submission")
    creation_scheduler, creation_receipt, creation_runtime = (
        _live_array_execution_binding(
            root, manifest, cell, "evaluation", current_authorization
        )
    )
    _reject_partial_artifacts(result_path, root / str(cell["trace_path"]))
    if not (root / "verified/stage-models.json").is_file():
        raise ValueError("evaluation requires the complete authenticated model stage")
    _validate_stage_marker(root, manifest, "model")
    started = time.perf_counter()
    core, trace, timing = _compute_evaluation_cell(root, manifest, cell)
    core["creation_submission_authorization"] = current_authorization
    core["creation_scheduler_provenance"] = creation_scheduler
    core["creation_submission_receipt"] = creation_receipt
    trace_path = root / str(cell["trace_path"])
    if trace_path.parent != result_path.parent:
        raise ValueError("evaluation artifacts do not share one bundle directory")
    with _staged_artifact_directory(result_path.parent) as staging:
        staged_trace_path = staging / trace_path.name
        staged_result_path = staging / result_path.name
        _write_npz_exclusive(staged_trace_path, trace)
        result = {
            **core,
            "core_sha256": benchmark.object_sha256(core),
            "trace_file_sha256": benchmark.file_sha256(staged_trace_path),
            "trace_sha256": benchmark.array_sha256(trace),
            "timing": timing,
            "wall_seconds": time.perf_counter() - started,
            "wall_seconds_scope": (
                "result_creation_including_discarded_compile_warmup_and_trace_write"
            ),
            "runtime": creation_runtime,
        }
        if not _finite_tree(result):
            raise FloatingPointError("evaluation result is not finite")
        _write_json_exclusive(staged_result_path, result)
    return result


def verify_evaluation_cell(
    output_root: str | Path,
    index: int,
    *,
    strict_replay: bool = True,
    submission_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if strict_replay is not True:
        raise ValueError("evaluation certification requires strict replay")
    verification_started = time.perf_counter()
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell(manifest, "evaluation", index)
    verification_authorization = _validate_retained_submission_authorization(
        root, manifest, cell, "evaluation", submission_authorization
    )
    verification_scheduler, verification_receipt, verification_runtime = (
        _live_array_execution_binding(
            root, manifest, cell, "evaluation", verification_authorization
        )
    )
    result_path = root / str(cell["result_path"])
    trace_path = root / str(cell["trace_path"])
    result = read_json(result_path)
    creation_authorization, creation_scheduler, _, _ = (
        _validate_retained_creation_binding(root, manifest, cell, "evaluation", result)
    )
    _require_distinct_replay_processes(creation_scheduler, verification_scheduler)
    trace_file_sha256 = benchmark.file_sha256(trace_path)
    core_keys = (
        "schema_version",
        "status",
        "source_commit",
        "manifest_sha256",
        "cell_id",
        "cell_index",
        "task",
        "arm",
        "controller_family",
        "world_model_seed",
        "actor_seed",
        "evaluation_seeds",
        "episode_returns",
        "mean_return",
        "normalized_mean_return",
        "action_saturation_fraction",
        "mean_objective_improvement",
        "nonnegative_objective_improvement_fraction",
        "mean_gradient_norm",
        "mean_parameter_delta",
        "mean_anchor_action_drift",
        "mean_current_action_drift",
        "reference_budget_violation_fraction",
        "mean_trust_backtracks",
        "frozen_reference_fallback_fraction",
        "mean_heldout_acceptance_improvement",
        "heldout_acceptance_fraction",
        "acceptance_fraction",
        "behavior_filter_fallback_fraction",
        "executed_behavior_violation_fraction",
        "mean_behavior_distance",
        "objective_evaluations_per_step",
        "coverage",
        "imagined_coverage",
        "imagined_coverage_protocol",
        "a0_dependency_replay_verified",
        "source_reward_checkpoint_sha256",
        "source_rebrac_checkpoint_sha256",
        "model_checkpoint_sha256",
        "calibration_arrays_file_sha256",
        "dependency_trace_file_sha256",
        "discarded_pure_compile_warmup",
        "creation_submission_authorization",
        "creation_scheduler_provenance",
        "creation_submission_receipt",
    )
    core = {key: result[key] for key in core_keys}
    if (
        result.get("schema_version") != EVALUATION_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("cell_id") != cell["cell_id"]
        or result.get("cell_index") != int(cell["index"])
        or result.get("arm") != cell["arm"]
        or result.get("evaluation_seeds") != cell["evaluation_seeds"]
        or result.get("core_sha256") != benchmark.object_sha256(core)
        or result.get("trace_file_sha256") != trace_file_sha256
        or not _finite_tree(result)
    ):
        raise ValueError("evaluation cell contract differs")
    trace = benchmark.load_npz(trace_path)
    if result.get("trace_sha256") != benchmark.array_sha256(trace):
        raise ValueError("evaluation trace payload differs")
    trust_arms = {
        "A1_persistent_trust",
        "A2_reset_trust",
        "A3_persistent_trust_heldout",
        "F1_uniform_prior_trust",
        "F3_policy_tilt_prior_trust",
    }
    arm = str(result["arm"])
    sequence_search = arm.startswith(("O1_", "O2_", "P0_", "P1_", "K0_", "K1_"))
    endpoint_arm = arm.startswith(("K0_", "K1_"))
    if (
        (
            arm == "A0_persistent_unconstrained_replay"
            and result.get("a0_dependency_replay_verified") is not True
        )
        or (
            arm in trust_arms
            and not math.isclose(
                float(result["reference_budget_violation_fraction"]),
                0.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        or (
            sequence_search
            and result.get("objective_evaluations_per_step")
            != [ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS]
        )
    ):
        raise ValueError("evaluation intervention certificate differs")
    if endpoint_arm:
        lengths = np.asarray(trace["lengths"], dtype=np.int64)
        mask = np.arange(trace["actions"].shape[1])[None] < lengths[:, None]
        accepted = np.asarray(trace["behavior_accepted"], dtype=np.bool_)[mask]
        fallback = np.asarray(trace["used_behavior_fallback"], dtype=np.bool_)[mask]
        executed_safe = np.asarray(trace["executed_behavior_accepted"], dtype=np.bool_)[
            mask
        ]
        actions = np.asarray(trace["actions"])[mask]
        references = np.asarray(trace["reference_actions"])[mask]
        if (
            not np.array_equal(fallback, np.logical_not(accepted))
            or not bool(np.all(executed_safe))
            or not np.array_equal(actions[fallback], references[fallback])
            or not math.isclose(
                float(result["executed_behavior_violation_fraction"]),
                0.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("endpoint behavior filter did not govern execution")
    replay_core, replay_trace, _ = _compute_evaluation_cell(root, manifest, cell)
    replay_core["creation_submission_authorization"] = result[
        "creation_submission_authorization"
    ]
    replay_core["creation_scheduler_provenance"] = result[
        "creation_scheduler_provenance"
    ]
    replay_core["creation_submission_receipt"] = result["creation_submission_receipt"]
    if replay_core != core:
        raise ValueError("evaluation semantic replay differs")
    if benchmark.array_sha256(replay_trace) != result["trace_sha256"]:
        raise ValueError("evaluation strict bitwise trace replay differs")
    return _write_verified_marker(
        root,
        manifest,
        cell,
        "evaluation",
        result_path=result_path,
        extra={
            "trace_file_sha256": result["trace_file_sha256"],
            "strict_policy_model_environment_replay": True,
            "arm": cell["arm"],
            "verification_submission_authorization": verification_authorization,
            "verification_scheduler_provenance": verification_scheduler,
            "verification_submission_receipt": verification_receipt,
            "verification_runtime": verification_runtime,
            "strict_replay_wall_seconds": time.perf_counter() - verification_started,
        },
    )


_STAGE_PATHS = {
    "diagnostic": "verified/stage-diagnostics.json",
    "model": "verified/stage-models.json",
    "evaluation": "verified/stage-evaluations.json",
}

_STRICT_MARKER_FIELDS = {
    "diagnostic": "strict_policy_model_environment_replay",
    "model": "strict_deterministic_replay",
    "evaluation": "strict_policy_model_environment_replay",
}


def _validate_cell_marker(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    """Validate one retained marker and every artifact digest it advertises."""

    marker_path = root / str(cell["marker_path"])
    result_path = root / str(cell["result_path"])
    marker = read_json(marker_path)
    if (
        marker.get("schema_version") != MARKER_SCHEMA
        or marker.get("status") != "verified"
        or marker.get("stage") != stage
        or marker.get("source_commit") != manifest["source_commit"]
        or marker.get("manifest_sha256") != manifest["manifest_sha256"]
        or marker.get("cell_id") != cell["cell_id"]
        or marker.get("cell_index") != int(cell["index"])
        or marker.get("result_file_sha256") != benchmark.file_sha256(result_path)
        or marker.get(_STRICT_MARKER_FIELDS[stage]) is not True
        or not isinstance(marker.get("strict_replay_wall_seconds"), (int, float))
        or not math.isfinite(float(marker["strict_replay_wall_seconds"]))
        or float(marker["strict_replay_wall_seconds"]) < 0.0
        or marker.get("marker_sha256") != _unsigned_digest(marker, "marker_sha256")
    ):
        raise ValueError(f"{stage} cell marker is invalid")
    result = read_json(result_path)
    creation_authorization = _validate_retained_submission_authorization(
        root,
        manifest,
        cell,
        stage,
        result.get("creation_submission_authorization", {}),
    )
    verification_authorization = _validate_retained_submission_authorization(
        root,
        manifest,
        cell,
        stage,
        marker.get("verification_submission_authorization", {}),
    )
    creation_scheduler = _validate_scheduler_record(
        result.get("creation_scheduler_provenance", {}), int(cell["index"])
    )
    verification_scheduler = _validate_scheduler_record(
        marker.get("verification_scheduler_provenance", {}), int(cell["index"])
    )
    creation_receipt = _validate_array_submission_receipt(
        root,
        manifest,
        cell,
        stage,
        creation_authorization,
        creation_scheduler,
    )
    verification_receipt = _validate_array_submission_receipt(
        root,
        manifest,
        cell,
        stage,
        verification_authorization,
        verification_scheduler,
    )
    creation_runtime = _require_single_gpu_runtime(result.get("runtime", {}))
    verification_runtime = _require_single_gpu_runtime(
        marker.get("verification_runtime", {})
    )
    _require_preflight_runtime_match(root, manifest, creation_runtime)
    _require_preflight_runtime_match(root, manifest, verification_runtime)
    if benchmark.runtime_homogeneity_identity(
        creation_runtime
    ) != benchmark.runtime_homogeneity_identity(verification_runtime):
        raise ValueError("creation and strict-replay runtime identities differ")
    _require_distinct_replay_processes(creation_scheduler, verification_scheduler)
    expected_schema = {
        "diagnostic": DIAGNOSTIC_SCHEMA,
        "model": MODEL_SCHEMA,
        "evaluation": EVALUATION_SCHEMA,
    }[stage]
    if (
        result.get("schema_version") != expected_schema
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("cell_id") != cell["cell_id"]
        or result.get("cell_index") != int(cell["index"])
        or result.get("task") != TASK
        or not _finite_tree(result)
        or result.get("creation_submission_receipt") != creation_receipt
        or marker.get("verification_submission_receipt") != verification_receipt
        or dict(result.get("runtime", {})) != creation_runtime
        or dict(marker.get("verification_runtime", {})) != verification_runtime
        or creation_authorization["mode"] != "new"
        or verification_authorization["mode"] not in {"new", "verification_only"}
        or (
            verification_authorization["mode"] == "new"
            and verification_authorization != creation_authorization
        )
    ):
        raise ValueError(f"{stage} cell result identity is invalid")
    if stage == "diagnostic" and (
        result.get("world_model_seed") != int(cell["world_model_seed"])
        or result.get("diagnostic_only_no_selection_or_mutation") is not True
        or result.get("discarded_full_diagnostic_warmup") is not True
        or result.get("wall_seconds_scope")
        != "result_creation_including_discarded_full_warmup"
        or float(result.get("retained_compute_wall_seconds", -1.0)) < 0.0
    ):
        raise ValueError("diagnostic cell semantics are invalid")
    if stage == "model" and (
        result.get("family") != cell["family"]
        or result.get("world_model_seed") != int(cell["world_model_seed"])
        or result.get("actor_seed") != cell.get("actor_seed")
        or result.get("wall_seconds_scope")
        != "result_creation_including_discarded_compile_warmup_and_artifact_writes"
        or float(result.get("retained_training_wall_seconds", -1.0)) < 0.0
    ):
        raise ValueError("model cell semantics are invalid")
    if stage == "evaluation" and (
        result.get("arm") != cell["arm"]
        or result.get("world_model_seed") != int(cell["world_model_seed"])
        or result.get("actor_seed") != int(cell["actor_seed"])
        or result.get("evaluation_seeds") != cell["evaluation_seeds"]
        or result.get("wall_seconds_scope")
        != "result_creation_including_discarded_compile_warmup_and_trace_write"
    ):
        raise ValueError("evaluation cell semantics are invalid")
    if stage in {"diagnostic", "evaluation"}:
        trace_path = root / str(cell["trace_path"])
        if (
            marker.get("trace_file_sha256") != benchmark.file_sha256(trace_path)
            or result.get("trace_file_sha256") != benchmark.file_sha256(trace_path)
            or result.get("trace_sha256")
            != benchmark.array_sha256(benchmark.load_npz(trace_path))
        ):
            raise ValueError(f"{stage} trace digest is invalid")
    elif stage == "model":
        checkpoint_path = root / str(cell["checkpoint_path"])
        schedule_path = root / str(cell["schedule_path"])
        if (
            marker.get("checkpoint_sha256") != benchmark.file_sha256(checkpoint_path)
            or result.get("checkpoint_sha256") != benchmark.file_sha256(checkpoint_path)
            or marker.get("schedule_file_sha256")
            != benchmark.file_sha256(schedule_path)
            or result.get("schedule_file_sha256")
            != benchmark.file_sha256(schedule_path)
            or result.get("schedule_sha256")
            != benchmark.array_sha256(benchmark.load_npz(schedule_path))
        ):
            raise ValueError("model checkpoint or schedule digest is invalid")
    return marker


def _stage_marker_body(
    root: Path, manifest: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    if stage not in _STAGE_PATHS:
        raise ValueError(f"unknown roadmap stage: {stage}")
    cells = manifest[f"{stage}_cells"]
    entries: list[dict[str, Any]] = []
    for cell in cells:
        marker_path = root / str(cell["marker_path"])
        marker = _validate_cell_marker(root, manifest, cell, stage)
        entries.append(
            {
                "cell_id": cell["cell_id"],
                "cell_index": int(cell["index"]),
                "marker_file_sha256": benchmark.file_sha256(marker_path),
                "marker_sha256": marker["marker_sha256"],
                "result_file_sha256": marker["result_file_sha256"],
            }
        )
    upstream: dict[str, str] = {}
    if stage == "diagnostic":
        calibration = _validate_calibration_marker(root, manifest)
        upstream["calibration_marker_sha256"] = calibration["marker_sha256"]
    elif stage == "model":
        upstream_marker = _validate_stage_marker(root, manifest, "diagnostic")
        upstream["diagnostic_stage_marker_sha256"] = upstream_marker["marker_sha256"]
    else:
        upstream_marker = _validate_stage_marker(root, manifest, "model")
        upstream["model_stage_marker_sha256"] = upstream_marker["marker_sha256"]
    return {
        "schema_version": STAGE_MARKER_SCHEMA,
        "status": "verified",
        "stage": stage,
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_cells": len(entries),
        "expected_cells": len(cells),
        "strict_replay_for_every_cell": True,
        "cells": entries,
        **upstream,
    }


def _validate_stage_marker(
    root: Path, manifest: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    marker_path = root / _STAGE_PATHS[stage]
    marker = read_json(marker_path)
    scheduler, receipt, runtime = _validate_retained_stage_verifier_execution_binding(
        root,
        manifest,
        stage,
        marker.get("scheduler_provenance", {}),
        marker.get("submission_receipt", {}),
        marker.get("runtime", {}),
    )
    expected = _stage_marker_body(root, manifest, stage)
    expected.update(
        {
            "scheduler_provenance": scheduler,
            "submission_receipt": receipt,
            "runtime": runtime,
        }
    )
    expected["marker_sha256"] = benchmark.object_sha256(expected)
    if marker != expected:
        raise ValueError(f"{stage} stage marker is invalid")
    return marker


def _verify_stage(output_root: str | Path, stage: str) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    marker_path = root / _STAGE_PATHS[stage]
    if marker_path.is_file():
        return _validate_stage_marker(root, manifest, stage)
    body = _stage_marker_body(root, manifest, stage)
    scheduler, receipt, runtime = _live_stage_verifier_execution_binding(
        root, manifest, stage
    )
    body.update(
        {
            "scheduler_provenance": scheduler,
            "submission_receipt": receipt,
            "runtime": runtime,
        }
    )
    body["marker_sha256"] = benchmark.object_sha256(body)
    _write_json_exclusive(marker_path, body)
    return body


def verify_diagnostic_stage(output_root: str | Path) -> dict[str, Any]:
    """Authenticate all three diagnostic cells as one immutable stage."""

    return _verify_stage(output_root, "diagnostic")


def verify_model_stage(output_root: str | Path) -> dict[str, Any]:
    """Authenticate all 21 model cells and the diagnostic dependency."""

    return _verify_stage(output_root, "model")


def verify_evaluation_stage(output_root: str | Path) -> dict[str, Any]:
    """Authenticate all 90 evaluation cells and the model dependency."""

    return _verify_stage(output_root, "evaluation")


def _world_seed_arm_units(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Nest actor seeds inside world seeds before any across-world summary."""

    result: dict[str, list[dict[str, Any]]] = {}
    for arm in ALL_ARMS:
        rows = rows_by_arm[arm]
        units: list[dict[str, Any]] = []
        for world_seed in WORLD_SEEDS:
            nested = [
                row for row in rows if int(row["world_model_seed"]) == int(world_seed)
            ]
            if {int(row["actor_seed"]) for row in nested} != set(ACTOR_SEEDS) or len(
                nested
            ) != len(ACTOR_SEEDS):
                raise ValueError(f"arm {arm} lacks exactly two nested actor seeds")
            nested = sorted(nested, key=lambda row: int(row["actor_seed"]))
            actor_means = [float(row["mean_return"]) for row in nested]
            units.append(
                {
                    "world_model_seed": int(world_seed),
                    "nested_actor_seeds": [int(row["actor_seed"]) for row in nested],
                    "nested_actor_mean_returns": actor_means,
                    "world_seed_mean_return": float(np.mean(actor_means)),
                }
            )
        result[arm] = units
    return result


def _arm_summary(arm: str, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [row["world_seed_mean_return"] for row in units], dtype=np.float64
    )
    return {
        "arm": arm,
        "world_seed_units": list(units),
        "raw_return_iqm": benchmark.interquartile_mean(values),
        "normalized_return_iqm": benchmark.interquartile_mean(values) / 1000.0,
        "world_seed_mean": float(np.mean(values)),
        "world_seed_standard_deviation": float(np.std(values)),
    }


def _contrast_summary(
    name: str,
    candidate: str,
    baseline: str,
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    units_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    baseline_cells = {
        (int(row["world_model_seed"]), int(row["actor_seed"])): row
        for row in rows_by_arm[baseline]
    }
    candidate_cells = {
        (int(row["world_model_seed"]), int(row["actor_seed"])): row
        for row in rows_by_arm[candidate]
    }
    if set(candidate_cells) != set(baseline_cells):
        raise ValueError(f"contrast {name} is not paired")
    cell_deltas = [
        {
            "world_model_seed": world_seed,
            "actor_seed": actor_seed,
            "candidate_mean_return": float(candidate_cells[key]["mean_return"]),
            "baseline_mean_return": float(baseline_cells[key]["mean_return"]),
            "candidate_minus_baseline": float(
                candidate_cells[key]["mean_return"] - baseline_cells[key]["mean_return"]
            ),
        }
        for key in sorted(candidate_cells)
        for world_seed, actor_seed in (key,)
    ]
    candidate_world = {
        int(row["world_model_seed"]): float(row["world_seed_mean_return"])
        for row in units_by_arm[candidate]
    }
    baseline_world = {
        int(row["world_model_seed"]): float(row["world_seed_mean_return"])
        for row in units_by_arm[baseline]
    }
    world_deltas = [
        {
            "world_model_seed": seed,
            "candidate_minus_baseline": candidate_world[seed] - baseline_world[seed],
        }
        for seed in WORLD_SEEDS
    ]
    delta_values = np.asarray(
        [row["candidate_minus_baseline"] for row in world_deltas], np.float64
    )
    return {
        "name": name,
        "candidate": candidate,
        "baseline": baseline,
        "paired_actor_cells": cell_deltas,
        "paired_world_seed_units": world_deltas,
        "raw_return_delta_iqm": benchmark.interquartile_mean(delta_values),
        "normalized_return_delta_iqm": benchmark.interquartile_mean(delta_values)
        / 1000.0,
        "favorable_world_seed_fraction": float(np.mean(delta_values > 0.0)),
        "favorable_nested_actor_cell_fraction": float(
            np.mean([row["candidate_minus_baseline"] > 0.0 for row in cell_deltas])
        ),
    }


def _dependency_reference_rows(
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    old = _dependency_manifest(manifest)
    old_root = Path(str(manifest["dependency_root"]))
    count = int(manifest["evaluation_episodes"])
    rows = {arm: [] for arm in REFERENCE_ARMS}
    for world_seed in WORLD_SEEDS:
        for actor_seed in ACTOR_SEEDS:
            cell = _dependency_evaluation_cell(old, world_seed, actor_seed)
            result = read_json(old_root / str(cell["result_path"]))
            for arm, key in (
                ("R0_zero_shot_dependency", "zero_shot_episode_returns"),
                ("R1_unconstrained_dependency", "flowmpc_episode_returns"),
            ):
                returns = np.asarray(result[key], dtype=np.float64)[:count]
                if returns.size != count or not np.isfinite(returns).all():
                    raise ValueError("dependency reference returns are incomplete")
                rows[arm].append(
                    {
                        "arm": arm,
                        "world_model_seed": int(world_seed),
                        "actor_seed": int(actor_seed),
                        "evaluation_seeds": list(cell["evaluation_seeds"][:count]),
                        "episode_returns": returns.tolist(),
                        "mean_return": float(np.mean(returns)),
                        "dependency_result_file_sha256": benchmark.file_sha256(
                            old_root / str(cell["result_path"])
                        ),
                    }
                )
    return rows


def _mean_numeric_fields(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[str, float | None]:
    return {
        field: (
            None
            if not [row[field] for row in rows if row[field] is not None]
            else float(
                np.mean([float(row[field]) for row in rows if row[field] is not None])
            )
        )
        for field in fields
    }


def _diagnostic_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actor_rows = [
        {**row, "world_model_seed": int(result["world_model_seed"])}
        for result in results
        for row in result["actor_rows"]
    ]
    if len(actor_rows) != len(WORLD_SEEDS) * len(ACTOR_SEEDS):
        raise ValueError("diagnostic report lacks the six nested actor rows")
    horizon_summary: list[dict[str, Any]] = []
    for horizon in (1, 3, 5):
        world_units = []
        for world_seed in WORLD_SEEDS:
            values = [
                item
                for row in actor_rows
                if int(row["world_model_seed"]) == world_seed
                for item in row["horizon_summaries"]
                if int(item["horizon"]) == horizon
            ]
            if len(values) != len(ACTOR_SEEDS):
                raise ValueError("diagnostic horizon is incomplete")
            world_units.append(
                {
                    "world_model_seed": world_seed,
                    "mean_observation_rmse": float(
                        np.mean([row["mean_observation_rmse"] for row in values])
                    ),
                    "mean_reward_absolute_error": float(
                        np.mean([row["mean_reward_absolute_error"] for row in values])
                    ),
                    "mean_predictive_reward_sd": float(
                        np.mean([row["mean_predictive_reward_sd"] for row in values])
                    ),
                }
            )
        horizon_summary.append(
            {
                "horizon": horizon,
                "world_seed_units": world_units,
                "top_level_world_seed_mean_observation_rmse": float(
                    np.mean([row["mean_observation_rmse"] for row in world_units])
                ),
                "top_level_world_seed_mean_reward_absolute_error": float(
                    np.mean([row["mean_reward_absolute_error"] for row in world_units])
                ),
                "top_level_world_seed_mean_predictive_reward_sd": float(
                    np.mean([row["mean_predictive_reward_sd"] for row in world_units])
                ),
            }
        )
    residual_names = (
        "reward_residual",
        "continuation_residual",
        "value_projected_transition_residual",
        "bellman_residual",
    )
    residual_rmse = {}
    for name in residual_names:
        actor_values = []
        for row in actor_rows:
            values = np.asarray(
                [item[name] for item in row["component_residuals"]["rows"]],
                np.float64,
            )
            actor_values.append(float(np.sqrt(np.mean(np.square(values)))))
        residual_rmse[name] = float(np.mean(actor_values))
    counterfactual_rows = [row["exact_counterfactual"] for row in actor_rows]
    if not all(row.get("identifiable") is True for row in counterfactual_rows):
        raise ValueError("exact counterfactual diagnostic lost identifiability")
    horizon5_counterfactual_rows = [
        row["horizon5_planner_objective"] for row in counterfactual_rows
    ]
    if not all(row.get("identifiable") is True for row in horizon5_counterfactual_rows):
        raise ValueError("exact H5 planner counterfactual lost identifiability")
    counterfactual_fields = (
        "return_rmse",
        "pairwise_gain_rmse",
        "pairwise_ranking_accuracy",
        "top1_accuracy",
        "mean_simulator_regret",
    )
    coverage_fields = (
        "mean_conservative_coverage_score",
        "minimum_conservative_coverage_score",
        "mean_nearest_neighbor_distance",
        "within_train_distance_q95_fraction",
    )
    return {
        "nested_actor_rows": len(actor_rows),
        "all_a0_dependency_replays_verified": all(
            row.get("a0_dependency_replay_verified") is True for row in actor_rows
        ),
        "mean_coverage": _mean_numeric_fields(
            [row["coverage"] for row in actor_rows],
            coverage_fields,
        ),
        "mean_imagined_horizon5_coverage": {
            scope: _mean_numeric_fields(
                [row["imagined_horizon5_coverage"][scope] for row in actor_rows],
                coverage_fields,
            )
            for scope in ("proposal", "best", "reference")
        },
        "mean_component_residual_rmse": residual_rmse,
        "horizon_summary": horizon_summary,
        "independent_noise": _mean_numeric_fields(
            [row["independent_noise"] for row in actor_rows],
            (
                "proposal_mean_improvement",
                "proposal_positive_improvement_fraction",
                "heldout_mean_improvement",
                "heldout_positive_improvement_fraction",
                "mean_generalization_gap",
                "positive_generalization_gap_fraction",
                "mean_gradient_cosine",
            ),
        ),
        "exact_counterfactual": _mean_numeric_fields(
            [row["metrics"] for row in counterfactual_rows],
            counterfactual_fields,
        ),
        "exact_horizon5_planner_counterfactual": _mean_numeric_fields(
            [row["metrics"] for row in horizon5_counterfactual_rows],
            counterfactual_fields,
        ),
    }


def _model_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in MODEL_FAMILIES:
        rows = [row for row in results if row["family"] == family]
        expected = (
            len(WORLD_SEEDS) * len(ACTOR_SEEDS)
            if family in ACTOR_CONDITIONED_MODEL_FAMILIES
            else len(WORLD_SEEDS)
        )
        if len(rows) != expected:
            raise ValueError(f"model family {family} is incomplete")
        finals = [row["engine_result"]["metrics"]["final"] for row in rows]
        summary[family] = {
            "cells": len(rows),
            "mean_final_total_loss": float(np.mean([row["total"] for row in finals])),
            "mean_final_primary_loss": float(
                np.mean([row["primary"] for row in finals])
            ),
            "mean_final_auxiliary_loss": float(
                np.mean([row["auxiliary"] for row in finals])
            ),
            "maximum_gradient_norm": float(
                max(
                    row["engine_result"]["metrics"]["maximum_grad_norm"] for row in rows
                )
            ),
            "total_wall_seconds": float(sum(row["wall_seconds"] for row in rows)),
        }
    return summary


def _controller_summary(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "action_saturation_fraction",
        "mean_objective_improvement",
        "nonnegative_objective_improvement_fraction",
        "mean_gradient_norm",
        "mean_parameter_delta",
        "mean_anchor_action_drift",
        "mean_current_action_drift",
        "reference_budget_violation_fraction",
        "mean_trust_backtracks",
        "frozen_reference_fallback_fraction",
        "mean_heldout_acceptance_improvement",
        "heldout_acceptance_fraction",
        "acceptance_fraction",
        "behavior_filter_fallback_fraction",
        "executed_behavior_violation_fraction",
        "mean_behavior_distance",
    )
    coverage_fields = (
        "mean_conservative_coverage_score",
        "minimum_conservative_coverage_score",
        "mean_nearest_neighbor_distance",
        "within_train_distance_q95_fraction",
    )
    summary = {}
    for arm in FRESH_ARMS:
        rows = [row for row in evaluations if row["arm"] == arm]
        if len(rows) != len(WORLD_SEEDS) * len(ACTOR_SEEDS):
            raise ValueError(f"controller telemetry for {arm} is incomplete")
        summary[arm] = {
            **_mean_numeric_fields(rows, fields),
            "mean_milliseconds_per_step": float(
                np.mean([row["timing"]["mean_milliseconds_per_step"] for row in rows])
            ),
            "total_timed_controller_seconds": float(
                sum(row["timing"]["total_timed_seconds"] for row in rows)
            ),
            "mean_coverage": _mean_numeric_fields(
                [row["coverage"] for row in rows],
                coverage_fields,
            ),
            "mean_imagined_coverage": {
                scope: _mean_numeric_fields(
                    [row["imagined_coverage"][scope] for row in rows],
                    coverage_fields,
                )
                for scope in ("proposal", "best", "reference")
            },
            "objective_evaluations_per_step": sorted(
                {
                    int(value)
                    for row in rows
                    for value in row["objective_evaluations_per_step"]
                }
            ),
        }
    return summary


def _calibration_summary(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    result_path = root / str(manifest["calibration"]["result_path"])
    arrays_path = root / str(manifest["calibration"]["arrays_path"])
    result = read_json(result_path)
    marker = verify_calibration(root)
    rows = sorted(
        result["rows"],
        key=lambda row: (int(row["world_model_seed"]), int(row["actor_seed"])),
    )
    return {
        "partition": "training_replay_only",
        "evaluation_seeds_accessed": False,
        "selected_global_tilt_eta": float(result["selected_global_tilt_eta"]),
        "rows": [
            {
                "world_model_seed": int(row["world_model_seed"]),
                "actor_seed": int(row["actor_seed"]),
                "transition_count": int(row["transition_count"]),
                "local_maximum_eligible_tilt_eta": float(
                    row["local_maximum_eligible_tilt_eta"]
                ),
                "behavior_distance_q95": float(row["behavior_distance_q95"]),
                "bellman_target_scale": float(row["bellman_target_scale"]),
                "training_data_sha256": row["training_data_sha256"],
            }
            for row in rows
        ],
        "result_file_sha256": benchmark.file_sha256(result_path),
        "arrays_file_sha256": benchmark.file_sha256(arrays_path),
        "marker_sha256": marker["marker_sha256"],
        "wall_seconds": float(result["wall_seconds"]),
    }


def _build_final_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    reference_rows = _dependency_reference_rows(manifest)
    evaluations = [
        read_json(root / str(cell["result_path"]))
        for cell in manifest["evaluation_cells"]
    ]
    rows_by_arm: dict[str, list[Mapping[str, Any]]] = {
        **reference_rows,
        **{
            arm: [row for row in evaluations if row["arm"] == arm] for arm in FRESH_ARMS
        },
    }
    units_by_arm = _world_seed_arm_units(rows_by_arm)
    arm_summaries = {arm: _arm_summary(arm, units_by_arm[arm]) for arm in ALL_ARMS}
    contrasts = {
        name: _contrast_summary(name, pair[0], pair[1], rows_by_arm, units_by_arm)
        for name, pair in manifest["factorial_contrasts"].items()
    }
    comparisons = {
        f"{arm}_minus_{reference}": _contrast_summary(
            f"{arm}_minus_{reference}",
            arm,
            reference,
            rows_by_arm,
            units_by_arm,
        )
        for arm in FRESH_ARMS
        for reference in REFERENCE_ARMS
    }
    diagnostics = [
        read_json(root / str(cell["result_path"]))
        for cell in manifest["diagnostic_cells"]
    ]
    models = [
        read_json(root / str(cell["result_path"])) for cell in manifest["model_cells"]
    ]
    diagnostic_markers = [
        read_json(root / str(cell["marker_path"]))
        for cell in manifest["diagnostic_cells"]
    ]
    model_markers = [
        read_json(root / str(cell["marker_path"])) for cell in manifest["model_cells"]
    ]
    evaluation_markers = [
        read_json(root / str(cell["marker_path"]))
        for cell in manifest["evaluation_cells"]
    ]
    result_creation_seconds = float(
        sum(row["wall_seconds"] for row in diagnostics + models + evaluations)
    )
    strict_verification_seconds = float(
        sum(
            marker["strict_replay_wall_seconds"]
            for marker in diagnostic_markers + model_markers + evaluation_markers
        )
    )
    observed_best = max(ALL_ARMS, key=lambda arm: arm_summaries[arm]["raw_return_iqm"])
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "task": TASK,
        "evidence_class": manifest["evidence_class"],
        "claim_eligible": False,
        "completed_diagnostic_cells": len(diagnostics),
        "completed_model_cells": len(models),
        "completed_evaluation_cells": len(evaluations),
        "strict_diagnostic_markers": len(manifest["diagnostic_cells"]),
        "strict_model_markers": len(manifest["model_cells"]),
        "strict_evaluation_markers": len(manifest["evaluation_cells"]),
        "aggregation_protocol": (
            "episode_mean_then_equal_nested_actor_mean_then_three_top_level_"
            "world_model_seed_IQM"
        ),
        "arm_summaries": arm_summaries,
        "factorial_contrasts": contrasts,
        "comparisons_to_dependency_references": comparisons,
        "calibration_summary": _calibration_summary(root, manifest),
        "diagnostic_summary": _diagnostic_summary(diagnostics),
        "model_training_summary": _model_summary(models),
        "controller_telemetry": _controller_summary(evaluations),
        "runtime_evidence": {
            "scope": (
                "per_cell_result_creation_plus_per_cell_strict_replay_verification;_"
                "excludes_preflight_calibration_stage_finalization_slurm_queue_and_"
                "scheduler_overhead"
            ),
            "total_diagnostic_wall_seconds": float(
                sum(row["wall_seconds"] for row in diagnostics)
            ),
            "total_model_wall_seconds": float(
                sum(row["wall_seconds"] for row in models)
            ),
            "total_evaluation_wall_seconds": float(
                sum(row["wall_seconds"] for row in evaluations)
            ),
            "mean_evaluation_cell_wall_seconds": float(
                np.mean([row["wall_seconds"] for row in evaluations])
            ),
            "total_result_creation_wall_seconds": result_creation_seconds,
            "total_strict_replay_verification_wall_seconds": (
                strict_verification_seconds
            ),
            "total_accounted_cell_wall_seconds": (
                result_creation_seconds + strict_verification_seconds
            ),
            "runtime_fingerprints": sorted(
                {
                    benchmark.object_sha256(row["runtime"])
                    for row in diagnostics + models + evaluations
                }
            ),
            "per_diagnostic_cell": [
                {
                    "cell_id": row["cell_id"],
                    "wall_seconds": float(row["wall_seconds"]),
                    "strict_replay_verification_wall_seconds": float(
                        diagnostic_markers[index]["strict_replay_wall_seconds"]
                    ),
                }
                for index, row in enumerate(diagnostics)
            ],
            "per_model_cell": [
                {
                    "cell_id": row["cell_id"],
                    "family": row["family"],
                    "wall_seconds": float(row["wall_seconds"]),
                    "retained_training_wall_seconds": float(
                        row["retained_training_wall_seconds"]
                    ),
                    "strict_replay_verification_wall_seconds": float(
                        model_markers[index]["strict_replay_wall_seconds"]
                    ),
                }
                for index, row in enumerate(models)
            ],
            "per_evaluation_cell": [
                {
                    "cell_id": row["cell_id"],
                    "arm": row["arm"],
                    "wall_seconds": float(row["wall_seconds"]),
                    "mean_milliseconds_per_step": float(
                        row["timing"]["mean_milliseconds_per_step"]
                    ),
                    "strict_replay_verification_wall_seconds": float(
                        evaluation_markers[index]["strict_replay_wall_seconds"]
                    ),
                }
                for index, row in enumerate(evaluations)
            ],
        },
        "descriptive_observed_best_arm_no_selection_claim": observed_best,
        "descriptive_observed_best_raw_return_iqm": arm_summaries[observed_best][
            "raw_return_iqm"
        ],
        "limitations": manifest["declared_limitations"],
        "claim_status": "exploratory_single_task_diagnostic_only",
        "interpretation": (
            "The frozen factorial diagnoses mechanisms and generates hypotheses; "
            "it does not establish a confirmatory or NeurIPS-level performance claim."
        ),
    }
    report["report_sha256"] = benchmark.object_sha256(report)
    return report


def _write_final_marker(
    root: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    expected_mode: str,
) -> dict[str, Any]:
    if expected_mode not in {"new", "report_recovery"}:
        raise ValueError("finalization mode is invalid")
    evaluation_stage = _validate_stage_marker(root, manifest, "evaluation")
    scheduler, receipt, runtime = _live_single_execution_binding(
        root, manifest, "finalize"
    )
    _require_preflight_runtime_match(root, manifest, runtime)
    if receipt.get("intent_payload", {}).get("mode") != expected_mode:
        raise ValueError("finalization receipt mode differs from artifact state")
    marker: dict[str, Any] = {
        "schema_version": STAGE_MARKER_SCHEMA,
        "status": "verified",
        "stage": "final",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_stage_marker_sha256": evaluation_stage["marker_sha256"],
        "report_file_sha256": benchmark.file_sha256(root / "report.json"),
        "report_sha256": report["report_sha256"],
        "strict_diagnostic_cells": len(manifest["diagnostic_cells"]),
        "strict_model_cells": len(manifest["model_cells"]),
        "strict_evaluation_cells": len(manifest["evaluation_cells"]),
        "scheduler_provenance": scheduler,
        "submission_receipt": receipt,
        "runtime": runtime,
        "finalization_mode": expected_mode,
    }
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / "verified/final.json"
    if marker_path.is_file():
        if read_json(marker_path) != marker:
            raise ValueError("existing final marker differs")
        return marker
    _write_json_exclusive(marker_path, marker)
    return marker


def finalize(output_root: str | Path) -> dict[str, Any]:
    """Aggregate the fully authenticated frozen matrix without selecting on it."""

    root = Path(output_root)
    if (root / "report.json").is_file():
        # A scheduler interruption can land after the atomic report rename but
        # before the final marker rename.  Recreate only that missing marker,
        # and only after independently proving that the retained report is the
        # unique report implied by the authenticated inputs.
        if not (root / "verified/final.json").is_file():
            manifest = read_json(root / "manifest.json")
            validate_manifest(manifest)
            authenticate_dependency(manifest["dependency_root"])
            _validate_stage_marker(root, manifest, "evaluation")
            report = read_json(root / "report.json")
            if report != _build_final_report(root, manifest):
                raise ValueError("retained unmarked final report differs")
            _write_final_marker(root, manifest, report, expected_mode="report_recovery")
        return validate_final(root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    authenticate_dependency(manifest["dependency_root"])
    _validate_stage_marker(root, manifest, "evaluation")
    report = _build_final_report(root, manifest)
    if not _finite_tree(report):
        raise FloatingPointError("actor-gap roadmap report is not finite")
    _write_json_exclusive(root / "report.json", report)
    _write_final_marker(root, manifest, report, expected_mode="new")
    return report


def validate_final(output_root: str | Path) -> dict[str, Any]:
    """Independently rederive every final statistic and artifact binding."""

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    authenticate_dependency(manifest["dependency_root"])
    _validate_stage_marker(root, manifest, "diagnostic")
    _validate_stage_marker(root, manifest, "model")
    _validate_stage_marker(root, manifest, "evaluation")
    report = read_json(root / "report.json")
    expected = _build_final_report(root, manifest)
    if (
        report != expected
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("claim_eligible") is not False
        or report.get("completed_diagnostic_cells") != len(WORLD_SEEDS)
        or report.get("completed_model_cells") != 21
        or report.get("completed_evaluation_cells")
        != len(WORLD_SEEDS) * len(ACTOR_SEEDS) * len(FRESH_ARMS)
        or report.get("report_sha256") != _unsigned_digest(report, "report_sha256")
        or not _finite_tree(report)
    ):
        raise ValueError("final actor-gap roadmap report is invalid")
    marker = read_json(root / "verified/final.json")
    scheduler, receipt, runtime = _validate_retained_single_execution_binding(
        root,
        manifest,
        "finalize",
        marker.get("scheduler_provenance", {}),
        marker.get("submission_receipt", {}),
        marker.get("runtime", {}),
    )
    _require_preflight_runtime_match(root, manifest, runtime)
    finalization_mode = receipt.get("intent_payload", {}).get("mode")
    expected_marker = {
        "schema_version": STAGE_MARKER_SCHEMA,
        "status": "verified",
        "stage": "final",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluation_stage_marker_sha256": read_json(root / _STAGE_PATHS["evaluation"])[
            "marker_sha256"
        ],
        "report_file_sha256": benchmark.file_sha256(root / "report.json"),
        "report_sha256": report["report_sha256"],
        "strict_diagnostic_cells": len(manifest["diagnostic_cells"]),
        "strict_model_cells": len(manifest["model_cells"]),
        "strict_evaluation_cells": len(manifest["evaluation_cells"]),
        "scheduler_provenance": scheduler,
        "submission_receipt": receipt,
        "runtime": runtime,
        "finalization_mode": finalization_mode,
    }
    expected_marker["marker_sha256"] = benchmark.object_sha256(expected_marker)
    if finalization_mode not in {"new", "report_recovery"} or marker != expected_marker:
        raise ValueError("final actor-gap roadmap marker is invalid")
    return report


def self_test() -> None:
    """Exercise the frozen matrix and the required hierarchical contrasts."""

    old_cells = []
    seed = 10_000
    for world_seed in WORLD_SEEDS:
        for actor_seed in ACTOR_SEEDS:
            old_cells.append(
                {
                    "cell_id": f"old-{world_seed}-{actor_seed}",
                    "world_model_seed": world_seed,
                    "actor_seed": actor_seed,
                    "evaluation_seeds": list(range(seed, seed + 5)),
                }
            )
            seed += 5
    matrix = build_study_matrix(
        {
            "evaluation_cells": old_cells,
            "inference_tuning": {"environment_seed": 999},
        },
        evaluation_episodes=2,
    )
    if (
        len(matrix["diagnostic_cells"]) != 3
        or len(matrix["model_cells"]) != 21
        or len(matrix["evaluation_cells"]) != 90
    ):
        raise AssertionError("roadmap matrix cardinality is wrong")
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm_index, arm in enumerate(ALL_ARMS):
        rows_by_arm[arm] = [
            {
                "arm": arm,
                "world_model_seed": world_seed,
                "actor_seed": actor_seed,
                "mean_return": float(10 * world_index + actor_index + arm_index),
            }
            for world_index, world_seed in enumerate(WORLD_SEEDS)
            for actor_index, actor_seed in enumerate(ACTOR_SEEDS)
        ]
    units = _world_seed_arm_units(rows_by_arm)
    contrast = _contrast_summary(
        "synthetic",
        ALL_ARMS[1],
        ALL_ARMS[0],
        rows_by_arm,
        units,
    )
    if (
        len(contrast["paired_actor_cells"]) != 6
        or len(contrast["paired_world_seed_units"]) != 3
        or not math.isclose(contrast["raw_return_delta_iqm"], 1.0)
        or contrast["favorable_world_seed_fraction"] != 1.0
    ):
        raise AssertionError("hierarchical paired aggregation is wrong")
    unsigned = {"status": "verified", "value": 1}
    signed = {**unsigned, "marker_sha256": benchmark.object_sha256(unsigned)}
    if signed["marker_sha256"] != _unsigned_digest(signed, "marker_sha256"):
        raise AssertionError("marker digest round trip is wrong")
    if _finite_tree({"bad": np.asarray([0.0, np.inf])}):
        raise AssertionError("non-finite artifact escaped validation")
