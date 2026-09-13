"""Corrected pilot actor training over the complete frozen HPO grid.

The original matched-objective pilot used the historical continuous PMPO
actor.  Because that actor failed an exact control problem, its outcomes may
not be reused for HPO.  This module leaves the authenticated datasets, compute
plans, world models, and rollout results untouched, but retrains every nested
actor seed from a fresh initialization with percentile-EMA normalized
REINFORCE and no fixed behavior-policy KL penalty.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import read_json, write_json_atomic
from . import matched_objective_benchmark as benchmark
from .matched_objective_protocol import (
    ARM_ORDER,
    validate_matched_objective_protocol,
)

canonical_task_id = benchmark.canonical_task_id


SCHEMA = "trajectory-imf-correct-actor-training-v1"
RESULT_SCHEMA = "trajectory-imf-correct-actor-cell-v1"
MARKER_SCHEMA = "trajectory-imf-correct-actor-marker-v1"
REPORT_SCHEMA = "trajectory-imf-correct-actor-report-v1"
PREPARATION_UPDATES = 500
ACTOR_UPDATES = 10_000
BEHAVIOR_KL_SCALE = 0.0
RETURN_SCALE_EMA_DECAY = 0.99
EXPECTED_CELLS = 288
REQUIRED_INPUT_STAGES = ("dataset", "compute_plan", "world_model", "rollout")
CLUSTER_STAGE_SCHEMA = "trajectory-imf-cluster-stage-verification-v1"


def validate_cluster_stage_verification(
    verification: Mapping[str, Any],
    *,
    stage: str,
    expected_cells: int,
    matrix_sha256: str,
) -> None:
    """Authenticate a completed source-pilot stage marker."""

    required = {
        "schema_version",
        "status",
        "profile",
        "stage",
        "matrix_sha256",
        "map_sha256",
        "verified_cell_count",
        "result_files",
        "slurm_job_id",
        "stage_verification_sha256",
    }
    if set(verification) != required:
        raise ValueError(f"pilot {stage} verification schema is invalid")
    result_files = verification.get("result_files")
    if (
        verification.get("schema_version") != CLUSTER_STAGE_SCHEMA
        or verification.get("status") != "verified_complete"
        or verification.get("profile") != "pilot"
        or verification.get("stage") != stage
        or verification.get("matrix_sha256") != matrix_sha256
        or int(verification.get("verified_cell_count", -1)) != int(expected_cells)
        or not isinstance(result_files, list)
        or len(result_files) != int(expected_cells)
        or verification.get("stage_verification_sha256")
        != _unsigned_digest(verification, "stage_verification_sha256")
    ):
        raise ValueError(f"pilot {stage} verification is incomplete or mismatched")
    expected_result_keys = {"cell_id", "path", "sha256"}
    if any(
        not isinstance(row, Mapping)
        or set(row) != expected_result_keys
        or not isinstance(row["cell_id"], str)
        or not isinstance(row["path"], str)
        or not isinstance(row["sha256"], str)
        or len(row["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in row["sha256"])
        for row in result_files
    ):
        raise ValueError(f"pilot {stage} verification result-file rows are invalid")
    ids = [str(row["cell_id"]) for row in result_files]
    if len(ids) != len(set(ids)):
        raise ValueError(f"pilot {stage} verification duplicates a result cell")


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
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _input_files(root: Path) -> dict[str, Path]:
    return {
        "source_manifest": root / "source_manifest.json",
        "protocol": root / "frozen_protocol.json",
        "matrix": root / "matrix.json",
        **{
            f"{stage}_verification": root / "cluster_stage_verification" / f"{stage}.json"
            for stage in REQUIRED_INPUT_STAGES
        },
    }


def load_and_validate_input(
    pilot_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the canonical pilot grid and its authenticated input stages."""

    root = Path(pilot_root).resolve(strict=True)
    files = _input_files(root)
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise ValueError(f"pilot dependency files are missing: {missing}")
    source = read_json(files["source_manifest"])
    protocol = read_json(files["protocol"])
    matrix = read_json(files["matrix"])
    validate_matched_objective_protocol(protocol)
    benchmark.validate_pilot_hpo_matrix(matrix, protocol, source)
    expected_counts = benchmark.expected_hpo_matrix_counts(protocol)
    if expected_counts.get("actor") != EXPECTED_CELLS:
        raise ValueError("frozen pilot no longer contains exactly 288 actor cells")
    for stage in REQUIRED_INPUT_STAGES:
        verification = read_json(files[f"{stage}_verification"])
        validate_cluster_stage_verification(
            verification,
            stage=stage,
            expected_cells=int(expected_counts[stage]),
            matrix_sha256=str(matrix["matrix_sha256"]),
        )
    return root, source, protocol, matrix


def _corrected_cell(
    original: Mapping[str, Any], index: int, source_commit: str
) -> dict[str, Any]:
    identity = {
        "index": int(index),
        "stage": "correct_actor",
        "source_commit": source_commit,
        "original_actor_cell_id": str(original["cell_id"]),
        "task": canonical_task_id(str(original["task"])),
        "arm": str(original["arm"]),
        "candidate_id": str(original["candidate_id"]),
        "world_model_seed": int(original["world_model_seed"]),
        "actor_seed": int(original["actor_seed"]),
        "budget_track": str(original["budget_track"]),
        "dependencies": list(original["dependencies"]),
    }
    digest = benchmark.object_sha256(identity)
    return {
        **identity,
        "cell_id": f"correct-actor-{digest[:24]}",
        "result_path": f"cells/correct-actor-{digest[:24]}/result.json",
        "checkpoint_path": f"cells/correct-actor-{digest[:24]}/checkpoint.pkl",
        "schedule_path": f"cells/correct-actor-{digest[:24]}/batch_schedule.npz",
        "trace_path": f"cells/correct-actor-{digest[:24]}/action_traces.npz",
        "marker_path": f"verified/correct-actor-{digest[:24]}.json",
    }


def build_manifest(
    pilot_root: str | Path,
    *,
    preparation_updates: int = PREPARATION_UPDATES,
    actor_updates: int = ACTOR_UPDATES,
) -> dict[str, Any]:
    if preparation_updates <= 0 or actor_updates <= 0:
        raise ValueError("actor preparation and training updates must be positive")
    root, source, protocol, matrix = load_and_validate_input(pilot_root)
    source_commit = _git_commit()
    original_actor_cells = benchmark.matrix_cells(matrix, "actor")
    if len(original_actor_cells) != EXPECTED_CELLS:
        raise ValueError("pilot actor grid does not contain exactly 288 cells")
    cells = [
        _corrected_cell(cell, index, source_commit)
        for index, cell in enumerate(original_actor_cells)
    ]
    profile = protocol["profiles"]["pilot"]
    files = _input_files(root)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": source_commit,
        "pilot_root": str(root),
        "pilot_source_sha256": source["source_sha256"],
        "pilot_protocol_sha256": matrix["protocol_sha256"],
        "pilot_matrix_sha256": matrix["matrix_sha256"],
        "input_file_sha256s": {
            name: benchmark.file_sha256(path) for name, path in files.items()
        },
        "old_actor_results_used": False,
        "selection_rule": protocol["hyperparameter_policy"]["selection_rule"],
        "selection_profile": "pilot",
        "claim_eligible": False,
        "independent_unit": "task_x_world_model_seed",
        "actor_seed_role": "nested_conditional_optimization_variance",
        "tasks": [canonical_task_id(task) for task in profile["tasks"]],
        "world_model_seeds": [int(seed) for seed in profile["world_model_seeds"]],
        "actor_seeds": [
            int(seed) for seed in profile["actor_seeds_nested_within_world_model_seed"]
        ],
        "preparation_updates": int(preparation_updates),
        "actor_updates": int(actor_updates),
        "actor_gradient": "reinforce",
        "return_scale": "ema_p95_minus_p5_clipped_below_one",
        "return_scale_ema_decay": RETURN_SCALE_EMA_DECAY,
        "behavior_kl_scale": BEHAVIOR_KL_SCALE,
        "preparation": "behavior_cloning_plus_replay_critic",
        "real_environment_evaluation_episodes": int(
            profile["real_environment_evaluation_episodes"]
        ),
        "maximum_environment_steps": int(protocol["data"]["native_episode_limit"]),
        "cells": cells,
    }
    body["manifest_sha256"] = benchmark.object_sha256(body)
    validate_manifest(body, pilot_root=root)
    return body


def validate_manifest(
    manifest: Mapping[str, Any], *, pilot_root: str | Path | None = None
) -> None:
    required = {
        "schema_version",
        "status",
        "source_commit",
        "pilot_root",
        "pilot_source_sha256",
        "pilot_protocol_sha256",
        "pilot_matrix_sha256",
        "input_file_sha256s",
        "old_actor_results_used",
        "selection_rule",
        "selection_profile",
        "claim_eligible",
        "independent_unit",
        "actor_seed_role",
        "tasks",
        "world_model_seeds",
        "actor_seeds",
        "preparation_updates",
        "actor_updates",
        "actor_gradient",
        "return_scale",
        "return_scale_ema_decay",
        "behavior_kl_scale",
        "preparation",
        "real_environment_evaluation_episodes",
        "maximum_environment_steps",
        "cells",
        "manifest_sha256",
    }
    if set(manifest) != required:
        raise ValueError("corrected actor manifest keys are incomplete or contain extras")
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "frozen_before_execution"
        or manifest.get("old_actor_results_used") is not False
        or manifest.get("claim_eligible") is not False
        or manifest.get("selection_profile") != "pilot"
        or manifest.get("actor_gradient") != "reinforce"
        or float(manifest.get("behavior_kl_scale", math.nan)) != 0.0
        or float(manifest.get("return_scale_ema_decay", math.nan)) != 0.99
        or manifest.get("preparation") != "behavior_cloning_plus_replay_critic"
        or int(manifest.get("preparation_updates", -1)) <= 0
        or int(manifest.get("actor_updates", -1)) <= 0
        or manifest.get("manifest_sha256")
        != _unsigned_digest(manifest, "manifest_sha256")
    ):
        raise ValueError("corrected actor manifest identity or training contract is invalid")
    if manifest["source_commit"] != _git_commit():
        raise ValueError("corrected actor manifest source differs from the executing checkout")
    root, source, protocol, matrix = load_and_validate_input(
        pilot_root if pilot_root is not None else str(manifest["pilot_root"])
    )
    if str(root) != str(Path(str(manifest["pilot_root"])).resolve()):
        raise ValueError("corrected actor pilot root differs")
    files = _input_files(root)
    if manifest["input_file_sha256s"] != {
        name: benchmark.file_sha256(path) for name, path in files.items()
    }:
        raise ValueError("corrected actor input file digests changed")
    if (
        manifest["pilot_source_sha256"] != source["source_sha256"]
        or manifest["pilot_protocol_sha256"] != matrix["protocol_sha256"]
        or manifest["pilot_matrix_sha256"] != matrix["matrix_sha256"]
        or manifest["selection_rule"]
        != protocol["hyperparameter_policy"]["selection_rule"]
    ):
        raise ValueError("corrected actor source, protocol, or matrix binding differs")
    originals = benchmark.matrix_cells(matrix, "actor")
    expected = [
        _corrected_cell(original, index, str(manifest["source_commit"]))
        for index, original in enumerate(originals)
    ]
    if manifest["cells"] != expected or len(expected) != EXPECTED_CELLS:
        raise ValueError("corrected actor cell grid differs from all 288 pilot actors")
    identifiers = [cell["cell_id"] for cell in manifest["cells"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("corrected actor cell ids are duplicated")


def write_manifest(
    pilot_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = build_manifest(pilot_root, **settings)
    path = Path(output_root) / "manifest.json"
    if path.is_file():
        stored = read_json(path)
        if stored != manifest:
            raise ValueError("existing corrected actor manifest differs")
    else:
        write_json_atomic(path, manifest)
    return manifest


def _cell_by_index(manifest: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    rows = [cell for cell in manifest["cells"] if int(cell["index"]) == int(index)]
    if len(rows) != 1:
        raise ValueError("corrected actor cell index is absent or duplicated")
    return rows[0]


def _original_actor_cell(
    matrix: Mapping[str, Any], cell: Mapping[str, Any]
) -> Mapping[str, Any]:
    rows = [
        row
        for row in benchmark.matrix_cells(matrix, "actor")
        if row["cell_id"] == cell["original_actor_cell_id"]
    ]
    if len(rows) != 1:
        raise ValueError("original actor cell is absent or duplicated")
    original = rows[0]
    for key in (
        "task",
        "arm",
        "candidate_id",
        "world_model_seed",
        "actor_seed",
        "budget_track",
        "dependencies",
    ):
        if original[key] != cell[key]:
            raise ValueError(f"corrected actor {key} differs from original grid")
    return original


def _authenticated_source_result(
    root: Path,
    matrix: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
) -> tuple[dict[str, Any], Path]:
    """Load a source artifact authenticated by its original stage verifier.

    Reused artifacts were verified under the exact 456d-era source. Replaying
    architecture-sensitive initialization checks under later source would be
    neither equivalent nor desirable. The signed stage marker instead binds
    the exact result bytes, after which stage-specific supplementary hashes
    are checked below.
    """

    if cell.get("stage") != stage:
        raise ValueError(f"source dependency is not a {stage} cell")
    marker = read_json(root / "cluster_stage_verification" / f"{stage}.json")
    expected_cells = benchmark.expected_hpo_matrix_counts(
        read_json(root / "frozen_protocol.json")
    )[stage]
    validate_cluster_stage_verification(
        marker,
        stage=stage,
        expected_cells=int(expected_cells),
        matrix_sha256=str(matrix["matrix_sha256"]),
    )
    path = benchmark._cell_result_path(root, cell)
    relative = str(path.relative_to(root))
    rows = [
        row
        for row in marker["result_files"]
        if row["cell_id"] == cell["cell_id"]
    ]
    if (
        len(rows) != 1
        or rows[0]["path"] != relative
        or not path.is_file()
        or rows[0]["sha256"] != benchmark.file_sha256(path)
    ):
        raise ValueError(f"authenticated {stage} result binding differs")
    result = read_json(path)
    for key in (
        "cell_id",
        "stage",
        "task",
        "arm",
        "candidate_id",
        "world_model_seed",
        "actor_seed",
        "budget_track",
        "profile",
        "evidence_class",
        "source_sha256",
        "protocol_sha256",
        "config_template_sha256",
        "selection_sha256",
    ):
        if result.get(key) != cell.get(key):
            raise ValueError(f"authenticated {stage} result {key} differs")
    if result.get("matrix_sha256") != matrix["matrix_sha256"]:
        raise ValueError(f"authenticated {stage} result matrix differs")
    return result, path


def _load_dependencies(
    manifest: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    root, _, protocol, matrix = load_and_validate_input(manifest["pilot_root"])
    original = _original_actor_cell(matrix, cell)
    world_cell = benchmark._dependency_cell(matrix, original, "world_model")
    dataset_cell = benchmark._dependency_cell(matrix, original, "dataset")
    compute_cell = benchmark._dependency_cell(matrix, original, "compute_plan")
    world_result, _ = _authenticated_source_result(
        root, matrix, world_cell, "world_model"
    )
    dataset_result, _ = _authenticated_source_result(
        root, matrix, dataset_cell, "dataset"
    )
    compute_result, _ = _authenticated_source_result(
        root, matrix, compute_cell, "compute_plan"
    )
    world_directory = benchmark.stage_directory(root, world_cell)
    dataset_directory = benchmark.stage_directory(root, dataset_cell)
    benchmark.validate_dataset_result(
        dataset_result,
        dataset_cell,
        dataset_directory / "dataset.npz",
        protocol=protocol,
        matrix=matrix,
    )
    benchmark.validate_compute_plan_cell(
        compute_result, protocol, matrix, compute_cell, root
    )
    benchmark._validate_compute_files(
        compute_result, benchmark.stage_directory(root, compute_cell)
    )
    benchmark.validate_world_result(
        world_result,
        world_cell,
        world_directory,
    )
    return (
        protocol,
        matrix,
        original,
        world_directory,
        dataset_directory,
        world_result,
        dataset_result,
    )


def corrected_config(
    manifest: Mapping[str, Any], cell: Mapping[str, Any]
) -> tuple[Any, Mapping[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    (
        protocol,
        matrix,
        original,
        world_directory,
        dataset_directory,
        world_result,
        dataset_result,
    ) = _load_dependencies(manifest, cell)
    config = benchmark.make_config(
        protocol,
        original["profile"],
        original["arm"],
        dataset_result["observation_shape"],
        int(dataset_result["action_dim"]),
        output_root=manifest["pilot_root"],
        candidate=benchmark._candidate_for_cell(matrix, original),
    )
    config = replace(
        config,
        actor_gradient="reinforce",
        behavior_kl_scale=0.0,
        critic_bins=51,
        critic_output_init_scale=0.0,
        return_scale_ema_decay=float(manifest["return_scale_ema_decay"]),
    )
    return (
        config,
        original,
        world_result,
        dataset_result,
        world_directory,
        dataset_directory,
    )


def _schedule(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    config: Any,
) -> dict[str, np.ndarray]:
    execution = benchmark._profile_execution("pilot")
    return benchmark._batch_schedule(
        arrays,
        task=str(cell["task"]),
        world_model_seed=benchmark.derive_seed(
            "correct-actor-shared-batches",
            cell["task"],
            cell["world_model_seed"],
            cell["actor_seed"],
        ),
        updates=int(manifest["preparation_updates"]) + int(manifest["actor_updates"]),
        batch_size=int(execution["batch_size"]),
        sequence_length=int(execution["sequence_length"]),
    )


def _checkpoint_metadata(
    *,
    cell: Mapping[str, Any],
    world_result: Mapping[str, Any],
    preparation_updates: int,
    actor_updates: int,
    return_scale_state: Any,
    latest: Mapping[str, Any] | None,
    latest_bc: float,
    latest_replay_critic: float,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "stage": "correct_actor",
        "cell_id": cell["cell_id"],
        "original_actor_cell_id": cell["original_actor_cell_id"],
        "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
        "completed_preparation_updates": int(preparation_updates),
        "completed_actor_updates": int(actor_updates),
        "return_scale_percentile_range": float(return_scale_state.percentile_range),
        "return_scale_initialized": bool(return_scale_state.initialized),
        "last_metrics": None if latest is None else dict(latest),
        "last_behavior_cloning_loss": float(latest_bc),
        "last_replay_critic_loss": float(latest_replay_critic),
        "wall_seconds": float(wall_seconds),
    }


def run_cell(
    pilot_root: str | Path,
    output_root: str | Path,
    index: int,
    **settings: Any,
) -> dict[str, Any]:
    """Train one fresh corrected actor without mutating the source world model."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        ReturnScaleState,
        create_agent,
        diverse_imagination_starts,
        init_return_scale_state,
        jit_observe_sequence,
        jit_train_actor_critic_dreamer3,
        jit_train_behavior_cloning,
        jit_train_replay_critic,
        load_checkpoint,
        save_checkpoint,
    )

    manifest = write_manifest(pilot_root, output_root, **settings)
    cell = _cell_by_index(manifest, index)
    directory = Path(output_root) / Path(str(cell["result_path"])).parent
    result_path = Path(output_root) / str(cell["result_path"])
    if result_path.is_file():
        result = read_json(result_path)
        validate_cell_result(result, cell, manifest, output_root, strict_replay=False)
        return result
    (
        config,
        original,
        world_result,
        dataset_result,
        world_directory,
        dataset_directory,
    ) = corrected_config(manifest, cell)
    world_state, _, world_metadata = load_checkpoint(world_directory / "checkpoint.pkl")
    if world_metadata.get("cell_id") != benchmark._dependency_cell(
        load_and_validate_input(manifest["pilot_root"])[3], original, "world_model"
    )["cell_id"]:
        raise ValueError("corrected actor world-model checkpoint identity mismatch")
    fresh = create_agent(
        config,
        benchmark.derive_jax_key(
            "actor-init", cell["task"], cell["world_model_seed"], cell["actor_seed"]
        ),
    )
    initial_actor_digest = benchmark._tree_digest(fresh.params.actor)
    initial_critic_digest = benchmark._tree_digest(fresh.params.critic)
    frozen_world = world_state.params.world_model
    frozen_world_digest = benchmark._tree_digest(frozen_world)
    state = AgentState(
        AgentParams(frozen_world, fresh.params.actor, fresh.params.critic),
        world_state.model_optimizer,
        fresh.actor_optimizer,
        fresh.critic_optimizer,
        fresh.slow_critic,
        world_state.world_model_teacher,
    )
    arrays = benchmark.load_npz(dataset_directory / "dataset.npz")
    schedule = _schedule(arrays, manifest, cell, config)
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = Path(output_root) / str(cell["schedule_path"])
    if schedule_path.is_file():
        if benchmark.array_sha256(benchmark.load_npz(schedule_path)) != benchmark.array_sha256(schedule):
            raise ValueError("existing corrected actor schedule differs")
    else:
        benchmark._write_npz_atomic(schedule_path, schedule)
    checkpoint_path = Path(output_root) / str(cell["checkpoint_path"])
    prep_done = 0
    actor_done = 0
    accumulated_wall = 0.0
    latest: dict[str, Any] | None = None
    latest_bc = math.nan
    latest_replay_critic = math.nan
    return_scale_state = init_return_scale_state()
    if checkpoint_path.is_file():
        state, stored_config, metadata = load_checkpoint(checkpoint_path)
        if stored_config != config or metadata.get("cell_id") != cell["cell_id"]:
            raise ValueError("partial corrected actor checkpoint identity mismatch")
        if metadata.get("world_model_checkpoint_sha256") != world_result["checkpoint_sha256"]:
            raise ValueError("partial corrected actor world-model dependency mismatch")
        prep_done = int(metadata.get("completed_preparation_updates", -1))
        actor_done = int(metadata.get("completed_actor_updates", -1))
        accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        latest = metadata.get("last_metrics")
        latest_bc = float(metadata.get("last_behavior_cloning_loss", math.nan))
        latest_replay_critic = float(metadata.get("last_replay_critic_loss", math.nan))
        return_scale_state = ReturnScaleState(
            jnp.asarray(metadata.get("return_scale_percentile_range", 0.0), jnp.float32),
            jnp.asarray(metadata.get("return_scale_initialized", False)),
        )
    total_prep = int(manifest["preparation_updates"])
    total_actor = int(manifest["actor_updates"])
    if not (0 <= prep_done <= total_prep and 0 <= actor_done <= total_actor):
        raise ValueError("partial corrected actor update counters are invalid")
    if prep_done != total_prep and actor_done != 0:
        raise ValueError("actor optimization began before preparation completed")
    execution = benchmark._profile_execution("pilot")
    batch_size = int(execution["batch_size"])
    sequence_length = int(execution["sequence_length"])
    checkpoint_every = int(execution["checkpoint_every_updates"])
    posterior_key = benchmark.derive_jax_key(
        "correct-actor-posterior",
        cell["task"],
        cell["world_model_seed"],
        cell["actor_seed"],
    )
    start_key = benchmark.derive_jax_key(
        "correct-actor-start",
        cell["task"],
        cell["world_model_seed"],
        cell["actor_seed"],
    )
    objective_key = benchmark.derive_jax_key(
        "correct-actor-objective",
        cell["task"],
        cell["world_model_seed"],
        cell["actor_seed"],
    )
    started = time.perf_counter()

    def save_progress() -> None:
        save_checkpoint(
            checkpoint_path,
            state,
            config,
            metadata=_checkpoint_metadata(
                cell=cell,
                world_result=world_result,
                preparation_updates=prep_done,
                actor_updates=actor_done,
                return_scale_state=return_scale_state,
                latest=latest,
                latest_bc=latest_bc,
                latest_replay_critic=latest_replay_critic,
                wall_seconds=accumulated_wall + time.perf_counter() - started,
            ),
        )

    for update in range(prep_done, total_prep):
        replay = benchmark._materialize_batch(
            arrays,
            schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        sequence = jit_observe_sequence(
            state.params.world_model,
            replay["observations"],
            replay["actions"],
            jax.random.fold_in(posterior_key, update),
            config,
            is_first=replay["is_first"],
        )
        features = sequence.states.feature[:, config.burn_in :]
        actions = replay["actions"][:, config.burn_in :]
        state, bc_loss = jit_train_behavior_cloning(
            state,
            features.reshape((-1, config.feature_dim)),
            actions.reshape((-1, config.action_dim)),
            config,
        )
        state, replay_critic_loss = jit_train_replay_critic(
            state,
            sequence.states.feature,
            replay["rewards"],
            replay["continuations"],
            config,
            loss_mask=replay["loss_mask"],
        )
        latest_bc = float(bc_loss)
        latest_replay_critic = float(replay_critic_loss)
        prep_done = update + 1
        if prep_done % checkpoint_every == 0 or prep_done == total_prep:
            save_progress()

    for update in range(actor_done, total_actor):
        schedule_index = total_prep + update
        replay = benchmark._materialize_batch(
            arrays,
            schedule,
            schedule_index,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        sequence = jit_observe_sequence(
            state.params.world_model,
            replay["observations"],
            replay["actions"],
            jax.random.fold_in(posterior_key, schedule_index),
            config,
            is_first=replay["is_first"],
        )
        starts = diverse_imagination_starts(
            sequence.states,
            config.burn_in,
            jax.random.fold_in(start_key, update),
        )
        state, return_scale_state, metrics = jit_train_actor_critic_dreamer3(
            state,
            return_scale_state,
            starts,
            jax.random.fold_in(objective_key, update),
            config,
            behavior_prior=None,
        )
        latest = benchmark._metrics_dict(metrics)
        actor_done = update + 1
        if actor_done % checkpoint_every == 0 or actor_done == total_actor:
            save_progress()
    if latest is None or not _finite_tree(latest):
        raise FloatingPointError("corrected actor completed without finite telemetry")
    if benchmark._tree_digest(state.params.world_model) != frozen_world_digest:
        raise RuntimeError("corrected actor training changed the frozen world model")
    episodes = int(manifest["real_environment_evaluation_episodes"])
    maximum_steps = int(manifest["maximum_environment_steps"])
    returns, traces = benchmark._evaluate_actor_policy(
        state,
        config,
        task=str(cell["task"]),
        world_model_seed=int(cell["world_model_seed"]),
        actor_seed=int(cell["actor_seed"]),
        episodes=episodes,
        maximum_steps=maximum_steps,
    )
    trace_path = Path(output_root) / str(cell["trace_path"])
    benchmark._write_npz_atomic(trace_path, traces)
    total_wall = accumulated_wall + time.perf_counter() - started
    save_progress()
    normalized = np.asarray(returns, dtype=np.float64) / 1000.0
    lengths = np.asarray(traces["lengths"], dtype=np.int64)
    actions = np.asarray(traces["actions"], dtype=np.float32)
    mask = np.arange(actions.shape[1])[None, :] < lengths[:, None]
    action_saturation = float(
        np.mean(np.abs(actions)[mask] >= 0.95) if np.any(mask) else 0.0
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "original_actor_cell_id": cell["original_actor_cell_id"],
        "task": cell["task"],
        "arm": cell["arm"],
        "candidate_id": cell["candidate_id"],
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
        "budget_track": cell["budget_track"],
        "preparation_updates": total_prep,
        "actor_updates": total_actor,
        "behavior_kl_scale": 0.0,
        "actor_gradient": "reinforce",
        "return_scale_ema_decay": float(manifest["return_scale_ema_decay"]),
        "world_model_cell_id": benchmark._dependency_cell(
            load_and_validate_input(manifest["pilot_root"])[3], original, "world_model"
        )["cell_id"],
        "dataset_cell_id": benchmark._dependency_cell(
            load_and_validate_input(manifest["pilot_root"])[3], original, "dataset"
        )["cell_id"],
        "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
        "dataset_sha256": dataset_result["dataset_sha256"],
        "world_model_frozen": True,
        "world_model_parameter_delta": 0.0,
        "behavior_prior_used": False,
        "initial_actor_parameter_sha256": initial_actor_digest,
        "initial_critic_parameter_sha256": initial_critic_digest,
        "final_metrics": latest,
        "final_behavior_cloning_loss": latest_bc,
        "final_replay_critic_loss": latest_replay_critic,
        "final_return_scale_percentile_range": float(
            return_scale_state.percentile_range
        ),
        "episode_returns": [float(value) for value in returns],
        "normalized_episode_returns": [float(value) for value in normalized],
        "normalized_episode_return_mean": float(np.mean(normalized)),
        "normalized_episode_return_median": float(np.median(normalized)),
        "action_saturation_fraction": action_saturation,
        "imagined_real_per_step_return_gap": float(
            latest["mean_imagined_return"] / config.imagination_horizon
            - np.mean(normalized)
        ),
        "checkpoint_sha256": benchmark.file_sha256(checkpoint_path),
        "batch_schedule_file_sha256": benchmark.file_sha256(schedule_path),
        "batch_schedule_sha256": benchmark.array_sha256(schedule),
        "raw_action_traces_sha256": benchmark.file_sha256(trace_path),
        "final_actor_parameter_sha256": benchmark._tree_digest(state.params.actor),
        "final_critic_parameter_sha256": benchmark._tree_digest(state.params.critic),
        "runtime_config": asdict(config),
        "runtime_config_sha256": benchmark.object_sha256(asdict(config)),
        "runtime": benchmark.runtime_fingerprint(),
        "wall_seconds": total_wall,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if not _finite_tree(result):
        raise FloatingPointError("corrected actor result contains non-finite values")
    validate_cell_result(result, cell, manifest, output_root, strict_replay=False)
    write_json_atomic(result_path, result)
    return result


def _expected_evaluation_seeds(cell: Mapping[str, Any], episodes: int) -> np.ndarray:
    return np.asarray(
        [
            benchmark.derive_seed(
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


def validate_cell_result(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_root: str | Path,
    *,
    strict_replay: bool,
) -> None:
    from imf_dreamer_jax import load_checkpoint

    required = {
        "schema_version",
        "status",
        "source_commit",
        "manifest_sha256",
        "cell_id",
        "cell_index",
        "original_actor_cell_id",
        "task",
        "arm",
        "candidate_id",
        "world_model_seed",
        "actor_seed",
        "budget_track",
        "preparation_updates",
        "actor_updates",
        "behavior_kl_scale",
        "actor_gradient",
        "return_scale_ema_decay",
        "world_model_cell_id",
        "dataset_cell_id",
        "world_model_checkpoint_sha256",
        "dataset_sha256",
        "world_model_frozen",
        "world_model_parameter_delta",
        "behavior_prior_used",
        "initial_actor_parameter_sha256",
        "initial_critic_parameter_sha256",
        "final_metrics",
        "final_behavior_cloning_loss",
        "final_replay_critic_loss",
        "final_return_scale_percentile_range",
        "episode_returns",
        "normalized_episode_returns",
        "normalized_episode_return_mean",
        "normalized_episode_return_median",
        "action_saturation_fraction",
        "imagined_real_per_step_return_gap",
        "checkpoint_sha256",
        "batch_schedule_file_sha256",
        "batch_schedule_sha256",
        "raw_action_traces_sha256",
        "final_actor_parameter_sha256",
        "final_critic_parameter_sha256",
        "runtime_config",
        "runtime_config_sha256",
        "runtime",
        "wall_seconds",
        "slurm_job_id",
    }
    if set(result) != required:
        raise ValueError("corrected actor result keys are incomplete or contain extras")
    identity = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "original_actor_cell_id": cell["original_actor_cell_id"],
        "task": cell["task"],
        "arm": cell["arm"],
        "candidate_id": cell["candidate_id"],
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
        "budget_track": cell["budget_track"],
    }
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("corrected actor result identity mismatch")
    if (
        int(result.get("preparation_updates", -1))
        != int(manifest["preparation_updates"])
        or int(result.get("actor_updates", -1)) != int(manifest["actor_updates"])
        or result.get("actor_gradient") != "reinforce"
        or float(result.get("behavior_kl_scale", math.nan)) != 0.0
        or float(result.get("return_scale_ema_decay", math.nan))
        != float(manifest["return_scale_ema_decay"])
        or result.get("world_model_frozen") is not True
        or float(result.get("world_model_parameter_delta", math.nan)) != 0.0
        or result.get("behavior_prior_used") is not False
        or not _finite_tree(result)
    ):
        raise ValueError("corrected actor training contract or metrics are invalid")
    config, original, world_result, dataset_result, world_directory, dataset_directory = (
        corrected_config(manifest, cell)
    )
    _, _, _, matrix = load_and_validate_input(manifest["pilot_root"])
    expected_world_cell = benchmark._dependency_cell(matrix, original, "world_model")
    expected_dataset_cell = benchmark._dependency_cell(matrix, original, "dataset")
    if (
        result.get("world_model_cell_id") != expected_world_cell["cell_id"]
        or result.get("dataset_cell_id") != expected_dataset_cell["cell_id"]
    ):
        raise ValueError("corrected actor dependency cell identity mismatch")
    root = Path(output_root)
    checkpoint_path = root / str(cell["checkpoint_path"])
    schedule_path = root / str(cell["schedule_path"])
    trace_path = root / str(cell["trace_path"])
    for field, path in (
        ("checkpoint_sha256", checkpoint_path),
        ("batch_schedule_file_sha256", schedule_path),
        ("raw_action_traces_sha256", trace_path),
    ):
        if not path.is_file() or result.get(field) != benchmark.file_sha256(path):
            raise ValueError(f"corrected actor {path.name} digest mismatch")
    state, stored_config, metadata = load_checkpoint(checkpoint_path)
    if (
        stored_config != config
        or benchmark.object_sha256(result.get("runtime_config"))
        != benchmark.object_sha256(asdict(config))
    ):
        raise ValueError("corrected actor runtime config mismatch")
    if result.get("runtime_config_sha256") != benchmark.object_sha256(asdict(config)):
        raise ValueError("corrected actor runtime config digest mismatch")
    if (
        metadata.get("stage") != "correct_actor"
        or metadata.get("cell_id") != cell["cell_id"]
        or metadata.get("world_model_checkpoint_sha256")
        != world_result["checkpoint_sha256"]
        or int(metadata.get("completed_preparation_updates", -1))
        != int(manifest["preparation_updates"])
        or int(metadata.get("completed_actor_updates", -1))
        != int(manifest["actor_updates"])
        or metadata.get("return_scale_initialized") is not True
    ):
        raise ValueError("corrected actor checkpoint metadata mismatch")
    source_state, _, _ = load_checkpoint(world_directory / "checkpoint.pkl")
    if (
        benchmark._tree_digest(state.params.world_model)
        != benchmark._tree_digest(source_state.params.world_model)
        or result.get("world_model_checkpoint_sha256")
        != world_result["checkpoint_sha256"]
        or result.get("dataset_sha256") != dataset_result["dataset_sha256"]
        or result.get("final_actor_parameter_sha256")
        != benchmark._tree_digest(state.params.actor)
        or result.get("final_critic_parameter_sha256")
        != benchmark._tree_digest(state.params.critic)
    ):
        raise ValueError("corrected actor checkpoint or dependency changed")
    from imf_dreamer_jax import create_agent

    fresh = create_agent(
        config,
        benchmark.derive_jax_key(
            "actor-init", cell["task"], cell["world_model_seed"], cell["actor_seed"]
        ),
    )
    if (
        result.get("initial_actor_parameter_sha256")
        != benchmark._tree_digest(fresh.params.actor)
        or result.get("initial_critic_parameter_sha256")
        != benchmark._tree_digest(fresh.params.critic)
        or int(np.asarray(fresh.actor_optimizer.step)) != 0
        or int(np.asarray(fresh.critic_optimizer.step)) != 0
    ):
        raise ValueError("corrected actor did not bind a fresh initialization")
    expected_steps = int(manifest["preparation_updates"]) + int(
        manifest["actor_updates"]
    )
    if (
        int(np.asarray(state.actor_optimizer.step)) != expected_steps
        or int(np.asarray(state.critic_optimizer.step)) != expected_steps
    ):
        raise ValueError("corrected actor optimizer steps differ from the contract")
    arrays = benchmark.load_npz(dataset_directory / "dataset.npz")
    expected_schedule = _schedule(arrays, manifest, cell, config)
    observed_schedule = benchmark.load_npz(schedule_path)
    if (
        benchmark.array_sha256(observed_schedule)
        != benchmark.array_sha256(expected_schedule)
        or result.get("batch_schedule_sha256")
        != benchmark.array_sha256(expected_schedule)
    ):
        raise ValueError("corrected actor did not use the deterministic shared schedule")
    traces = benchmark.load_npz(trace_path)
    required_traces = {
        "actions",
        "rewards",
        "continuations",
        "is_last",
        "lengths",
        "evaluation_seeds",
    }
    if set(traces) != required_traces:
        raise ValueError("corrected actor raw trace schema is incomplete")
    episodes = int(manifest["real_environment_evaluation_episodes"])
    if not np.array_equal(
        traces["evaluation_seeds"], _expected_evaluation_seeds(cell, episodes)
    ):
        raise ValueError("corrected actor evaluation scenarios are not paired")
    lengths = np.asarray(traces["lengths"], dtype=np.int64)
    rewards = np.asarray(traces["rewards"], dtype=np.float64)
    actions = np.asarray(traces["actions"])
    continuations = np.asarray(traces["continuations"], dtype=np.float64)
    terminals = np.asarray(traces["is_last"])
    valid_steps = np.arange(actions.shape[1])[None, :] < lengths[:, None]
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
        or np.any(continuations[valid_steps] < 0.0)
        or np.any(continuations[valid_steps] > 1.0)
    ):
        raise ValueError("corrected actor raw trace arrays are invalid")
    recomputed = np.asarray(
        [float(np.sum(rewards[i, : int(length)])) for i, length in enumerate(lengths)],
        dtype=np.float64,
    )
    recorded = np.asarray(result.get("episode_returns"), dtype=np.float64)
    if (
        recorded.shape != (episodes,)
        or not np.array_equal(recomputed, recorded)
        or not np.array_equal(
            recorded / 1000.0,
            np.asarray(result.get("normalized_episode_returns"), dtype=np.float64),
        )
        or float(np.mean(recorded / 1000.0))
        != float(result.get("normalized_episode_return_mean"))
    ):
        raise ValueError("corrected actor returns do not recompute from raw traces")
    if strict_replay:
        benchmark._validate_actor_environment_replay(
            traces,
            recorded,
            state=state,
            config=config,
            task=str(cell["task"]),
            world_model_seed=int(cell["world_model_seed"]),
            actor_seed=int(cell["actor_seed"]),
            maximum_steps=int(manifest["maximum_environment_steps"]),
            action_repeat=1,
        )


def verify_cell(
    output_root: str | Path, index: int, *, strict_replay: bool = True
) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell_by_index(manifest, index)
    result_path = root / str(cell["result_path"])
    result = read_json(result_path)
    validate_cell_result(
        result, cell, manifest, root, strict_replay=strict_replay
    )
    marker: dict[str, Any] = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(result_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "batch_schedule_file_sha256": result["batch_schedule_file_sha256"],
        "raw_action_traces_sha256": result["raw_action_traces_sha256"],
        "strict_policy_and_environment_replay": bool(strict_replay),
    }
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / str(cell["marker_path"])
    if marker_path.is_file():
        if read_json(marker_path) != marker:
            raise ValueError("existing corrected actor verification marker differs")
    else:
        write_json_atomic(marker_path, marker)
    return marker


def validate_marker(
    marker: Mapping[str, Any],
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_root: str | Path,
) -> None:
    root = Path(output_root)
    result = read_json(root / str(cell["result_path"]))
    expected = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(root / str(cell["result_path"])),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "batch_schedule_file_sha256": result["batch_schedule_file_sha256"],
        "raw_action_traces_sha256": result["raw_action_traces_sha256"],
        "strict_policy_and_environment_replay": True,
    }
    if marker.get("marker_sha256") != _unsigned_digest(marker, "marker_sha256"):
        raise ValueError("corrected actor marker digest mismatch")
    observed = dict(marker)
    observed.pop("marker_sha256", None)
    if observed != expected:
        raise ValueError("corrected actor verification marker mismatch")


def validate_all_cells(output_root: str | Path) -> list[dict[str, Any]]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    results: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        result = read_json(root / str(cell["result_path"]))
        validate_cell_result(result, cell, manifest, root, strict_replay=False)
        marker = read_json(root / str(cell["marker_path"]))
        validate_marker(marker, cell, manifest, root)
        results.append(result)
    if len(results) != EXPECTED_CELLS:
        raise ValueError("corrected actor result count differs from 288")
    return results


def _corrected_result_for(
    results: Sequence[Mapping[str, Any]],
    *,
    original_actor_cell_id: str,
) -> Mapping[str, Any]:
    rows = [
        row
        for row in results
        if row["original_actor_cell_id"] == original_actor_cell_id
    ]
    if len(rows) != 1:
        raise ValueError("corrected actor result is absent or duplicated")
    return rows[0]


def build_corrected_trials(output_root: str | Path) -> dict[str, Any]:
    """Build the original trial schema without reading old actor results."""

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    results = validate_all_cells(root)
    pilot_root, source, protocol, matrix = load_and_validate_input(
        manifest["pilot_root"]
    )
    profile = protocol["profiles"]["pilot"]
    trials: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        for candidate in benchmark.pilot_candidates(protocol, arm):
            units: list[dict[str, Any]] = []
            for raw_task in profile["tasks"]:
                task = canonical_task_id(raw_task)
                for world_seed in profile["world_model_seeds"]:
                    worlds = [
                        cell
                        for cell in benchmark.matrix_cells(matrix, "world_model")
                        if cell["task"] == task
                        and cell["world_model_seed"] == int(world_seed)
                        and cell["arm"] == arm
                        and cell["candidate_id"] == candidate["candidate_id"]
                    ]
                    rollouts = [
                        cell
                        for cell in benchmark.matrix_cells(matrix, "rollout")
                        if cell["task"] == task
                        and cell["world_model_seed"] == int(world_seed)
                        and cell["arm"] == arm
                        and cell["candidate_id"] == candidate["candidate_id"]
                    ]
                    if len(worlds) != 1 or len(rollouts) != 1:
                        raise ValueError("corrected selection dependency is absent or duplicated")
                    world = worlds[0]
                    rollout = rollouts[0]
                    world_result, world_path = _authenticated_source_result(
                        pilot_root, matrix, world, "world_model"
                    )
                    benchmark.validate_world_result(
                        world_result,
                        world,
                        benchmark.stage_directory(pilot_root, world),
                    )
                    rollout_result, rollout_path = _authenticated_source_result(
                        pilot_root, matrix, rollout, "rollout"
                    )
                    rollout_metric = float(
                        rollout_result.get(
                            "normalized_free_running_rollout_error_auc", math.nan
                        )
                    )
                    raw_rollout = (
                        benchmark.stage_directory(pilot_root, rollout)
                        / "predictive_draws.npz"
                    )
                    if (
                        not math.isfinite(rollout_metric)
                        or rollout_metric < 0.0
                        or not raw_rollout.is_file()
                        or rollout_result.get("raw_predictive_draws_sha256")
                        != benchmark.file_sha256(raw_rollout)
                    ):
                        raise ValueError("authenticated rollout metric or draws differ")
                    actor_returns: list[float] = []
                    actor_hashes: list[str] = []
                    for actor_seed in profile[
                        "actor_seeds_nested_within_world_model_seed"
                    ]:
                        originals = [
                            cell
                            for cell in benchmark.matrix_cells(matrix, "actor")
                            if cell["task"] == task
                            and cell["world_model_seed"] == int(world_seed)
                            and cell["arm"] == arm
                            and cell["candidate_id"] == candidate["candidate_id"]
                            and cell["actor_seed"] == int(actor_seed)
                        ]
                        if len(originals) != 1:
                            raise ValueError("corrected nested actor is absent or duplicated")
                        corrected = _corrected_result_for(
                            results, original_actor_cell_id=originals[0]["cell_id"]
                        )
                        actor_returns.append(
                            float(corrected["normalized_episode_return_mean"])
                        )
                        corrected_cell = next(
                            cell
                            for cell in manifest["cells"]
                            if cell["cell_id"] == corrected["cell_id"]
                        )
                        actor_hashes.append(
                            benchmark.file_sha256(
                                root / str(corrected_cell["result_path"])
                            )
                        )
                    compute_cell = benchmark._dependency_cell(
                        matrix, world, "compute_plan"
                    )
                    compute_result, _ = _authenticated_source_result(
                        pilot_root, matrix, compute_cell, "compute_plan"
                    )
                    benchmark.validate_compute_plan_cell(
                        compute_result, protocol, matrix, compute_cell, pilot_root
                    )
                    benchmark._validate_compute_files(
                        compute_result,
                        benchmark.stage_directory(pilot_root, compute_cell),
                    )
                    units.append(
                        {
                            "task": task,
                            "world_model_seed": int(world_seed),
                            "rollout_auc": float(
                                rollout_metric
                            ),
                            "nested_actor_return": float(np.mean(actor_returns)),
                            "full_forward_backward_train_flops_per_update": float(
                                compute_result["tracks"]["equal_updates"]["allocations"][arm][
                                    "world_model"
                                ]["flops_per_update"]
                            ),
                            "artifact_sha256s": [
                                benchmark.file_sha256(world_path),
                                benchmark.file_sha256(rollout_path),
                                *actor_hashes,
                            ],
                        }
                    )
            trials.append(
                {
                    "arm": arm,
                    "candidate_id": candidate["candidate_id"],
                    "overrides": candidate["overrides"],
                    "units": units,
                }
            )
    trial_manifest: dict[str, Any] = {
        "schema_version": "matched-objective-hpo-trials-v1",
        "status": "complete",
        "protocol_sha256": matrix["protocol_sha256"],
        "source_sha256": source["source_sha256"],
        "selection_profile": "pilot",
        "selection_budget_track": "equal_updates",
        "confirmatory_outcomes_accessed": False,
        "trials": trials,
    }
    trial_manifest["trial_manifest_sha256"] = benchmark.object_sha256(
        trial_manifest
    )
    benchmark.validate_hpo_trial_manifest(
        trial_manifest, protocol, source_sha256=source["source_sha256"]
    )
    return trial_manifest


def select_candidates(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    _, source, protocol, _ = load_and_validate_input(manifest["pilot_root"])
    trials = build_corrected_trials(root)
    selection = benchmark.select_hpo_candidates(
        trials, protocol, source_sha256=source["source_sha256"]
    )
    for filename, payload in (
        ("corrected_hpo_trials.json", trials),
        ("corrected_hpo_selection.json", selection),
    ):
        path = root / filename
        if path.is_file() and read_json(path) != payload:
            raise ValueError(f"existing corrected selection artifact differs: {filename}")
        write_json_atomic(path, payload)
    return selection


def _selected_units(
    trials: Mapping[str, Any], selection: Mapping[str, Any], arm: str
) -> Sequence[Mapping[str, Any]]:
    selected_id = selection["selected"][arm]["candidate_id"]
    rows = [
        trial
        for trial in trials["trials"]
        if trial["arm"] == arm and trial["candidate_id"] == selected_id
    ]
    if len(rows) != 1:
        raise ValueError("corrected selected trial is absent or duplicated")
    return rows[0]["units"]


def finalize(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    existing_report = root / "report.json"
    if existing_report.is_file():
        return validate_final(root)
    manifest = read_json(root / "manifest.json")
    results = validate_all_cells(root)
    selection = select_candidates(root)
    trials = read_json(root / "corrected_hpo_trials.json")
    units_by_arm = {
        arm: list(_selected_units(trials, selection, arm)) for arm in ARM_ORDER
    }
    selected_return_iqm = {
        arm: benchmark.interquartile_mean(
            [float(unit["nested_actor_return"]) for unit in units_by_arm[arm]]
        )
        for arm in ARM_ORDER
    }
    pairs: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        for seed in manifest["world_model_seeds"]:
            values = {}
            for arm in ARM_ORDER:
                rows = [
                    unit
                    for unit in units_by_arm[arm]
                    if unit["task"] == task
                    and int(unit["world_model_seed"]) == int(seed)
                ]
                if len(rows) != 1:
                    raise ValueError("corrected selected task/seed unit is absent")
                values[arm] = float(rows[0]["nested_actor_return"])
            pairs.append(
                {
                    "task": task,
                    "world_model_seed": int(seed),
                    "shortcut_forcing": values["shortcut_forcing"],
                    "trajectory_imf": values["trajectory_imf"],
                    "trajectory_imf_minus_shortcut": values["trajectory_imf"]
                    - values["shortcut_forcing"],
                }
            )
    selected_ids = {
        arm: selection["selected"][arm]["candidate_id"] for arm in ARM_ORDER
    }
    selected_results = [
        result
        for result in results
        if result["candidate_id"] == selected_ids[result["arm"]]
    ]
    stability = {
        arm: {
            "cells": len([row for row in selected_results if row["arm"] == arm]),
            "mean_action_saturation_fraction": float(
                np.mean(
                    [
                        row["action_saturation_fraction"]
                        for row in selected_results
                        if row["arm"] == arm
                    ]
                )
            ),
            "mean_actor_grad_norm": float(
                np.mean(
                    [
                        row["final_metrics"]["actor_grad_norm"]
                        for row in selected_results
                        if row["arm"] == arm
                    ]
                )
            ),
            "mean_imagined_real_per_step_return_gap": float(
                np.mean(
                    [
                        row["imagined_real_per_step_return_gap"]
                        for row in selected_results
                        if row["arm"] == arm
                    ]
                )
            ),
        }
        for arm in ARM_ORDER
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_cells": len(results),
        "registered_cells": len(manifest["cells"]),
        "strict_replay_markers": len(manifest["cells"]),
        "corrected_hpo_trial_manifest_sha256": trials["trial_manifest_sha256"],
        "corrected_hpo_selection_sha256": selection["selection_sha256"],
        "selected": selection["selected"],
        "selected_nested_actor_return_iqm": selected_return_iqm,
        "trajectory_imf_minus_shortcut_iqm": selected_return_iqm["trajectory_imf"]
        - selected_return_iqm["shortcut_forcing"],
        "paired_selected_units": pairs,
        "selected_stability": stability,
        "total_actor_cell_wall_seconds": float(
            sum(float(result["wall_seconds"]) for result in results)
        ),
        "runtime": benchmark.runtime_fingerprint(),
        "claim_status": "diagnostic_only_pilot_hpo_data_reused_for_selection",
        "limitations": [
            "pilot tasks and seeds are HPO selection data, not held-out confirmatory data",
            "world models and rollout artifacts are reused unchanged from the authenticated pilot",
            "actor compiler FLOPs were not recompiled for a confirmatory matched-compute claim",
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["report_sha256"] = benchmark.object_sha256(report)
    path = root / "report.json"
    if path.is_file() and read_json(path) != report:
        raise ValueError("existing corrected actor report differs")
    write_json_atomic(path, report)
    return report


def validate_selection_result(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    _, source, protocol, _ = load_and_validate_input(manifest["pilot_root"])
    trials = read_json(root / "corrected_hpo_trials.json")
    expected_trials = build_corrected_trials(root)
    if trials != expected_trials:
        raise ValueError("stored corrected HPO trials differ from actor evidence")
    selection = read_json(root / "corrected_hpo_selection.json")
    expected_selection = benchmark.select_hpo_candidates(
        trials, protocol, source_sha256=source["source_sha256"]
    )
    if selection != expected_selection:
        raise ValueError("stored corrected HPO selection differs from frozen rank rule")
    return selection


def validate_final(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    report = read_json(root / "report.json")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("completed_cells") != EXPECTED_CELLS
        or report.get("registered_cells") != EXPECTED_CELLS
        or report.get("strict_replay_markers") != EXPECTED_CELLS
        or report.get("claim_status")
        != "diagnostic_only_pilot_hpo_data_reused_for_selection"
        or report.get("report_sha256") != _unsigned_digest(report, "report_sha256")
        or not _finite_tree(report)
    ):
        raise ValueError("corrected actor final report identity or metrics are invalid")
    selection = validate_selection_result(root)
    if report.get("corrected_hpo_selection_sha256") != selection["selection_sha256"]:
        raise ValueError("corrected actor report selection binding differs")
    if set(report.get("selected_nested_actor_return_iqm", {})) != set(ARM_ORDER):
        raise ValueError("corrected actor report is missing an arm")
    expected_delta = (
        float(report["selected_nested_actor_return_iqm"]["trajectory_imf"])
        - float(report["selected_nested_actor_return_iqm"]["shortcut_forcing"])
    )
    if float(report.get("trajectory_imf_minus_shortcut_iqm", math.nan)) != expected_delta:
        raise ValueError("corrected actor IQM contrast does not recompute")
    return report


def self_test() -> None:
    """Exercise fail-closed record checks without requiring benchmark artifacts."""

    valid = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "source_commit": "a" * 40,
        "manifest_sha256": "b" * 64,
        "cell_id": "correct-actor-test",
        "cell_index": 0,
        "original_actor_cell_id": "actor-test",
        "task": "dmc_reacher_easy",
        "arm": "trajectory_imf",
        "candidate_id": "trajectory_imf-test",
        "world_model_seed": 211,
        "actor_seed": 311,
        "budget_track": "equal_updates",
        "preparation_updates": 500,
        "actor_updates": 10_000,
        "behavior_kl_scale": 0.0,
        "actor_gradient": "reinforce",
        "return_scale_ema_decay": 0.99,
        "world_model_frozen": True,
        "world_model_parameter_delta": 0.0,
        "behavior_prior_used": False,
        "final_metrics": {"actor_grad_norm": 1.0},
    }
    if not _finite_tree(valid):
        raise AssertionError("positive finite control failed")
    for mutation in (
        {"world_model_parameter_delta": 1e-6},
        {"world_model_frozen": False},
        {"behavior_kl_scale": 0.1},
        {"behavior_prior_used": True},
        {"actor_gradient": "pmpo"},
        {"final_metrics": {"actor_grad_norm": math.nan}},
    ):
        forged = {**valid, **mutation}
        accepted = (
            forged["world_model_parameter_delta"] == 0.0
            and forged["world_model_frozen"] is True
            and forged["behavior_kl_scale"] == 0.0
            and forged["behavior_prior_used"] is False
            and forged["actor_gradient"] == "reinforce"
            and _finite_tree(forged)
        )
        if accepted:
            raise AssertionError("forged corrected actor control was accepted")


__all__ = [
    "ACTOR_UPDATES",
    "BEHAVIOR_KL_SCALE",
    "EXPECTED_CELLS",
    "PREPARATION_UPDATES",
    "REPORT_SCHEMA",
    "RESULT_SCHEMA",
    "SCHEMA",
    "build_corrected_trials",
    "build_manifest",
    "corrected_config",
    "finalize",
    "load_and_validate_input",
    "run_cell",
    "select_candidates",
    "self_test",
    "validate_all_cells",
    "validate_cell_result",
    "validate_cluster_stage_verification",
    "validate_final",
    "validate_manifest",
    "validate_marker",
    "validate_selection_result",
    "verify_cell",
    "write_manifest",
]
