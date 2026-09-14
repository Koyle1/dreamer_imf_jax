"""iMF-only Reacher study for a direct transition-conditioned reward head."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from .artifacts import read_json, write_json_atomic
from . import correct_actor_training as corrected
from . import matched_objective_benchmark as benchmark


SCHEMA = "trajectory-imf-direct-transition-reward-study-v1"
REWARD_RESULT_SCHEMA = "trajectory-imf-direct-transition-reward-cell-v1"
ACTOR_RESULT_SCHEMA = "trajectory-imf-direct-transition-reward-actor-v1"
MARKER_SCHEMA = "trajectory-imf-direct-transition-reward-marker-v1"
REPORT_SCHEMA = "trajectory-imf-direct-transition-reward-report-v1"
TASK = "dmc_reacher_easy"
REWARD_UPDATES = 10_000
HELDOUT_BATCHES = 32
EXPECTED_REWARD_CELLS = 3
EXPECTED_ACTOR_CELLS = 6


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


def _matching_source_cell(
    manifest: Mapping[str, Any],
    *,
    arm: str,
    candidate_id: str,
    world_seed: int,
    actor_seed: int,
) -> Mapping[str, Any]:
    rows = [
        cell
        for cell in manifest["cells"]
        if cell["task"] == TASK
        and cell["arm"] == arm
        and cell["candidate_id"] == candidate_id
        and int(cell["world_model_seed"]) == int(world_seed)
        and int(cell["actor_seed"]) == int(actor_seed)
    ]
    if len(rows) != 1:
        raise ValueError("selected corrected actor cell is absent or duplicated")
    return rows[0]


def _reference_row(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = root / str(cell["result_path"])
    marker_path = root / str(cell["marker_path"])
    result = read_json(result_path)
    marker = read_json(marker_path)
    expected_marker = {
        "schema_version": corrected.MARKER_SCHEMA,
        "status": "verified",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(result_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "batch_schedule_file_sha256": result["batch_schedule_file_sha256"],
        "raw_action_traces_sha256": result["raw_action_traces_sha256"],
        "strict_policy_and_environment_replay": True,
    }
    unsigned_marker = dict(marker)
    marker_digest = unsigned_marker.pop("marker_sha256", None)
    if (
        unsigned_marker != expected_marker
        or marker_digest != benchmark.object_sha256(unsigned_marker)
        or result.get("schema_version") != corrected.RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("cell_id") != cell["cell_id"]
        or result.get("task") != cell["task"]
        or result.get("arm") != cell["arm"]
        or result.get("candidate_id") != cell["candidate_id"]
        or int(result.get("world_model_seed", -1))
        != int(cell["world_model_seed"])
        or int(result.get("actor_seed", -1)) != int(cell["actor_seed"])
        or result.get("world_model_frozen") is not True
        or result.get("world_model_parameter_delta") != 0.0
        or not _finite_tree(result)
    ):
        raise ValueError("corrected actor reference is invalid")
    checkpoint_path = root / str(cell["checkpoint_path"])
    schedule_path = root / str(cell["schedule_path"])
    trace_path = root / str(cell["trace_path"])
    if (
        benchmark.file_sha256(checkpoint_path) != result["checkpoint_sha256"]
        or benchmark.file_sha256(schedule_path)
        != result["batch_schedule_file_sha256"]
        or benchmark.file_sha256(trace_path) != result["raw_action_traces_sha256"]
    ):
        raise ValueError("corrected actor reference artifact digest differs")
    return {
        "cell_id": cell["cell_id"],
        "result": str(result_path.resolve()),
        "result_sha256": benchmark.file_sha256(result_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "schedule": str(schedule_path.resolve()),
        "schedule_file_sha256": result["batch_schedule_file_sha256"],
        "schedule_sha256": result["batch_schedule_sha256"],
        "normalized_episode_return_mean": result[
            "normalized_episode_return_mean"
        ],
        "initial_actor_parameter_sha256": result[
            "initial_actor_parameter_sha256"
        ],
        "initial_critic_parameter_sha256": result[
            "initial_critic_parameter_sha256"
        ],
        "world_model_cell_id": result["world_model_cell_id"],
        "dataset_cell_id": result["dataset_cell_id"],
        "world_model_checkpoint_sha256": result[
            "world_model_checkpoint_sha256"
        ],
        "dataset_sha256": result["dataset_sha256"],
        "runtime_config": result["runtime_config"],
        "runtime_config_sha256": result["runtime_config_sha256"],
        "evaluation_seeds": [
            int(value)
            for value in benchmark.load_npz(trace_path)["evaluation_seeds"]
        ],
    }


def _validate_corrected_root(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate a completed older-source study without replaying its code."""

    manifest = read_json(root / "manifest.json")
    report = read_json(root / "report.json")
    selection = read_json(root / "corrected_hpo_selection.json")
    if (
        manifest.get("schema_version") != corrected.SCHEMA
        or manifest.get("status") != "frozen_before_execution"
        or manifest.get("manifest_sha256")
        != _unsigned_digest(manifest, "manifest_sha256")
        or report.get("schema_version") != corrected.REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("completed_cells") != corrected.EXPECTED_CELLS
        or report.get("strict_replay_markers") != corrected.EXPECTED_CELLS
        or report.get("report_sha256") != _unsigned_digest(report, "report_sha256")
        or selection.get("selection_sha256")
        != _unsigned_digest(selection, "selection_sha256")
        or report.get("corrected_hpo_selection_sha256")
        != selection["selection_sha256"]
    ):
        raise ValueError("corrected actor root is incomplete or unauthenticated")
    return manifest, report, selection


def _authenticated_pilot_result(
    pilot_root: Path,
    matrix: Mapping[str, Any],
    protocol: Mapping[str, Any],
    cell_id: str,
    stage: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    cells = [
        cell
        for cell in benchmark.matrix_cells(matrix, stage)
        if cell["cell_id"] == cell_id
    ]
    if len(cells) != 1:
        raise ValueError(f"selected {stage} dependency is absent or duplicated")
    cell = cells[0]
    verification = read_json(
        pilot_root / "cluster_stage_verification" / f"{stage}.json"
    )
    corrected.validate_cluster_stage_verification(
        verification,
        stage=stage,
        expected_cells=benchmark.expected_hpo_matrix_counts(protocol)[stage],
        matrix_sha256=matrix["matrix_sha256"],
    )
    rows = [
        row
        for row in verification["result_files"]
        if row["cell_id"] == cell_id
    ]
    if len(rows) != 1:
        raise ValueError(f"selected {stage} result is not authenticated")
    result_path = pilot_root / rows[0]["path"]
    if benchmark.file_sha256(result_path) != rows[0]["sha256"]:
        raise ValueError(f"selected {stage} result bytes changed")
    result = read_json(result_path)
    if result.get("cell_id") != cell_id or result.get("stage") != stage:
        raise ValueError(f"selected {stage} result identity differs")
    return cell, result


def build_manifest(
    corrected_root: str | Path,
    *,
    reward_updates: int = REWARD_UPDATES,
    heldout_batches: int = HELDOUT_BATCHES,
) -> dict[str, Any]:
    """Freeze selected Reacher iMF sources and paired actor references."""

    if reward_updates <= 0 or heldout_batches <= 0:
        raise ValueError("reward updates and held-out batches must be positive")
    root = Path(corrected_root).resolve(strict=True)
    source_manifest, report, selection = _validate_corrected_root(root)
    if report["corrected_hpo_selection_sha256"] != selection["selection_sha256"]:
        raise ValueError("corrected actor report/selection binding differs")
    trajectory_id = selection["selected"]["trajectory_imf"]["candidate_id"]
    shortcut_id = selection["selected"]["shortcut_forcing"]["candidate_id"]
    world_seeds = [int(value) for value in source_manifest["world_model_seeds"]]
    actor_seeds = [int(value) for value in source_manifest["actor_seeds"]]
    if len(world_seeds) != EXPECTED_REWARD_CELLS or len(actor_seeds) != 2:
        raise ValueError("corrected pilot seed grid differs from the frozen design")
    sources: list[dict[str, Any]] = []
    actor_references: list[dict[str, Any]] = []
    for world_seed in world_seeds:
        trajectory_cells = {
            actor_seed: _matching_source_cell(
                source_manifest,
                arm="trajectory_imf",
                candidate_id=trajectory_id,
                world_seed=world_seed,
                actor_seed=actor_seed,
            )
            for actor_seed in actor_seeds
        }
        shortcut_cells = {
            actor_seed: _matching_source_cell(
                source_manifest,
                arm="shortcut_forcing",
                candidate_id=shortcut_id,
                world_seed=world_seed,
                actor_seed=actor_seed,
            )
            for actor_seed in actor_seeds
        }
        trajectory_references = {
            actor_seed: _reference_row(
                root, source_manifest, trajectory_cells[actor_seed]
            )
            for actor_seed in actor_seeds
        }
        shortcut_references = {
            actor_seed: _reference_row(
                root, source_manifest, shortcut_cells[actor_seed]
            )
            for actor_seed in actor_seeds
        }
        first_reference = trajectory_references[actor_seeds[0]]
        config_values = dict(first_reference["runtime_config"])
        config_values["observation_shape"] = tuple(config_values["observation_shape"])
        config_values["overshooting_distances"] = tuple(
            config_values["overshooting_distances"]
        )
        from imf_dreamer_jax import DreamerConfig

        config = DreamerConfig(**config_values)
        if (
            config.prior != "imf"
            or not config.imf_trajectory_enabled
            or config.reward_loss != "mse"
            or config.reward_prediction_horizon != 0
            or config.actor_gradient != "reinforce"
            or config.behavior_kl_scale != 0.0
        ):
            raise ValueError("selected source is not the required scalar-reward iMF actor")
        pilot_root = Path(source_manifest["pilot_root"])
        matrix = read_json(pilot_root / "matrix.json")
        protocol = read_json(pilot_root / "frozen_protocol.json")
        world_cell, world_result = _authenticated_pilot_result(
            pilot_root,
            matrix,
            protocol,
            first_reference["world_model_cell_id"],
            "world_model",
        )
        dataset_cell, dataset_result = _authenticated_pilot_result(
            pilot_root,
            matrix,
            protocol,
            first_reference["dataset_cell_id"],
            "dataset",
        )
        world_directory = benchmark.stage_directory(pilot_root, world_cell)
        dataset_directory = benchmark.stage_directory(pilot_root, dataset_cell)
        source_checkpoint = world_directory / "checkpoint.pkl"
        dataset = dataset_directory / "dataset.npz"
        if (
            benchmark.file_sha256(source_checkpoint)
            != world_result["checkpoint_sha256"]
            or benchmark.file_sha256(dataset) != dataset_result["dataset_file_sha256"]
            or first_reference["world_model_checkpoint_sha256"]
            != world_result["checkpoint_sha256"]
            or first_reference["dataset_sha256"] != dataset_result["dataset_sha256"]
        ):
            raise ValueError("selected world-model or dataset bytes changed")
        sources.append(
            {
                "world_model_seed": world_seed,
                "world_model_cell_id": world_result["cell_id"],
                "world_model_checkpoint": str(source_checkpoint.resolve()),
                "world_model_checkpoint_sha256": world_result[
                    "checkpoint_sha256"
                ],
                "world_model_parameter_sha256": world_result[
                    "world_model_parameter_sha256"
                ],
                "dataset_cell_id": dataset_result["cell_id"],
                "dataset": str(dataset.resolve()),
                "dataset_file_sha256": dataset_result["dataset_file_sha256"],
                "dataset_sha256": dataset_result["dataset_sha256"],
                "runtime_config": asdict(config),
                "runtime_config_sha256": benchmark.object_sha256(asdict(config)),
            }
        )
        for actor_seed in actor_seeds:
            trajectory = trajectory_references[actor_seed]
            shortcut = shortcut_references[actor_seed]
            if (
                trajectory["schedule_sha256"] != shortcut["schedule_sha256"]
                or trajectory["initial_actor_parameter_sha256"]
                != shortcut["initial_actor_parameter_sha256"]
                or trajectory["initial_critic_parameter_sha256"]
                != shortcut["initial_critic_parameter_sha256"]
                or trajectory["evaluation_seeds"] != shortcut["evaluation_seeds"]
            ):
                raise ValueError("selected shortcut and iMF references are not paired")
            actor_references.append(
                {
                    "world_model_seed": world_seed,
                    "actor_seed": actor_seed,
                    "trajectory_imf": trajectory,
                    "shortcut_forcing": shortcut,
                }
            )
    source_commit = _git_commit()
    reward_cells = [
        {
            "index": index,
            "world_model_seed": seed,
            "cell_id": f"transition-reward-{seed}",
            "result_path": f"reward/seed-{seed}/result.json",
            "checkpoint_path": f"reward/seed-{seed}/checkpoint.pkl",
            "schedule_path": f"reward/seed-{seed}/schedule.npz",
            "heldout_schedule_path": f"reward/seed-{seed}/heldout_schedule.npz",
            "marker_path": f"verified/reward-seed-{seed}.json",
        }
        for index, seed in enumerate(world_seeds)
    ]
    actor_cells = [
        {
            "index": index,
            "world_model_seed": reference["world_model_seed"],
            "actor_seed": reference["actor_seed"],
            "cell_id": (
                f"transition-reward-actor-{reference['world_model_seed']}"
                f"-{reference['actor_seed']}"
            ),
            "result_path": (
                f"actor/seed-{reference['world_model_seed']}"
                f"-actor-{reference['actor_seed']}/result.json"
            ),
            "checkpoint_path": (
                f"actor/seed-{reference['world_model_seed']}"
                f"-actor-{reference['actor_seed']}/checkpoint.pkl"
            ),
            "schedule_path": (
                f"actor/seed-{reference['world_model_seed']}"
                f"-actor-{reference['actor_seed']}/schedule.npz"
            ),
            "trace_path": (
                f"actor/seed-{reference['world_model_seed']}"
                f"-actor-{reference['actor_seed']}/action_traces.npz"
            ),
            "marker_path": (
                f"verified/actor-seed-{reference['world_model_seed']}"
                f"-actor-{reference['actor_seed']}.json"
            ),
        }
        for index, reference in enumerate(actor_references)
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": source_commit,
        "corrected_root": str(root),
        "corrected_manifest_file_sha256": benchmark.file_sha256(
            root / "manifest.json"
        ),
        "corrected_report_file_sha256": benchmark.file_sha256(
            root / "report.json"
        ),
        "corrected_selection_file_sha256": benchmark.file_sha256(
            root / "corrected_hpo_selection.json"
        ),
        "task": TASK,
        "arm": "trajectory_imf",
        "selected_trajectory_candidate_id": trajectory_id,
        "selected_shortcut_candidate_id": shortcut_id,
        "world_model_seeds": world_seeds,
        "actor_seeds": actor_seeds,
        "reward_updates": int(reward_updates),
        "heldout_batches": int(heldout_batches),
        "reward_objective": "masked_one_step_scalar_mse",
        "reward_inputs": ["previous_feature", "action", "feature_delta"],
        "reward_hidden_layers": 3,
        "reward_learning_rate": 3e-4,
        "reward_grad_clip": 1.0,
        "source_world_model_frozen": True,
        "actor_training": {
            "preparation_updates": corrected.PREPARATION_UPDATES,
            "actor_updates": corrected.ACTOR_UPDATES,
            "actor_gradient": "reinforce",
            "return_scale_ema_decay": corrected.RETURN_SCALE_EMA_DECAY,
            "behavior_kl_scale": corrected.BEHAVIOR_KL_SCALE,
        },
        "source_artifacts": sources,
        "actor_references": actor_references,
        "reward_cells": reward_cells,
        "actor_cells": actor_cells,
        "claim_eligible": False,
        "evidence_class": "exploratory_selected_reacher_imf_reward_only",
    }
    body["manifest_sha256"] = benchmark.object_sha256(body)
    validate_manifest(body)
    return body


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on the frozen intervention and immutable references."""

    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "frozen_before_execution"
        or manifest.get("source_commit") != _git_commit()
        or manifest.get("task") != TASK
        or manifest.get("arm") != "trajectory_imf"
        or manifest.get("reward_objective") != "masked_one_step_scalar_mse"
        or manifest.get("reward_inputs")
        != ["previous_feature", "action", "feature_delta"]
        or int(manifest.get("reward_hidden_layers", -1)) != 3
        or manifest.get("source_world_model_frozen") is not True
        or manifest.get("claim_eligible") is not False
        or len(manifest.get("reward_cells", [])) != EXPECTED_REWARD_CELLS
        or len(manifest.get("actor_cells", [])) != EXPECTED_ACTOR_CELLS
        or manifest.get("manifest_sha256")
        != _unsigned_digest(manifest, "manifest_sha256")
    ):
        raise ValueError("transition-reward manifest contract is invalid")
    if any(
        reference["trajectory_imf"]["schedule_sha256"]
        != reference["shortcut_forcing"]["schedule_sha256"]
        or reference["trajectory_imf"]["evaluation_seeds"]
        != reference["shortcut_forcing"]["evaluation_seeds"]
        for reference in manifest["actor_references"]
    ):
        raise ValueError("actor references are not paired")
    root = Path(str(manifest["corrected_root"]))
    for filename, field in (
        ("manifest.json", "corrected_manifest_file_sha256"),
        ("report.json", "corrected_report_file_sha256"),
        ("corrected_hpo_selection.json", "corrected_selection_file_sha256"),
    ):
        if benchmark.file_sha256(root / filename) != manifest[field]:
            raise ValueError(f"immutable corrected reference changed: {filename}")
    for source in manifest["source_artifacts"]:
        if (
            benchmark.file_sha256(source["world_model_checkpoint"])
            != source["world_model_checkpoint_sha256"]
            or benchmark.file_sha256(source["dataset"])
            != source["dataset_file_sha256"]
        ):
            raise ValueError("source world-model or dataset artifact changed")
    for reference in manifest["actor_references"]:
        for arm in ("trajectory_imf", "shortcut_forcing"):
            row = reference[arm]
            if (
                benchmark.file_sha256(row["result"]) != row["result_sha256"]
                or benchmark.file_sha256(row["checkpoint"])
                != row["checkpoint_sha256"]
                or benchmark.file_sha256(row["schedule"])
                != row["schedule_file_sha256"]
            ):
                raise ValueError("immutable actor reference changed")


def write_manifest(
    corrected_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    path = Path(output_root) / "manifest.json"
    if path.is_file():
        manifest = read_json(path)
        validate_manifest(manifest)
        return manifest
    manifest = build_manifest(corrected_root, **settings)
    write_json_atomic(path, manifest)
    return manifest


def _source(manifest: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest["source_artifacts"]
        if int(row["world_model_seed"]) == int(seed)
    ]
    if len(rows) != 1:
        raise ValueError("world-model source is absent or duplicated")
    return rows[0]


def _source_config(source: Mapping[str, Any]) -> Any:
    """Recreate the corrected actor config after a JSON round trip."""

    from imf_dreamer_jax import DreamerConfig

    values = dict(source["runtime_config"])
    values["observation_shape"] = tuple(values["observation_shape"])
    values["overshooting_distances"] = tuple(values["overshooting_distances"])
    config = DreamerConfig(**values)
    if benchmark.object_sha256(asdict(config)) != source["runtime_config_sha256"]:
        raise ValueError("stored corrected runtime config digest differs")
    return config


def _actor_reference(
    manifest: Mapping[str, Any], seed: int, actor_seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest["actor_references"]
        if int(row["world_model_seed"]) == int(seed)
        and int(row["actor_seed"]) == int(actor_seed)
    ]
    if len(rows) != 1:
        raise ValueError("actor reference is absent or duplicated")
    return rows[0]


def _cell(manifest: Mapping[str, Any], stage: str, index: int) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest[f"{stage}_cells"]
        if int(row["index"]) == int(index)
    ]
    if len(rows) != 1:
        raise ValueError(f"{stage} cell index is absent or duplicated")
    return rows[0]


def _reward_schedules(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    execution = benchmark._profile_execution("pilot")
    settings = {
        "batch_size": int(execution["batch_size"]),
        "sequence_length": int(execution["sequence_length"]),
    }
    train = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=benchmark.derive_seed(
            "direct-transition-reward-train", seed
        ),
        updates=int(manifest["reward_updates"]),
        **settings,
    )
    heldout = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=benchmark.derive_seed(
            "direct-transition-reward-heldout", seed
        ),
        updates=int(manifest["heldout_batches"]),
        **settings,
    )
    if benchmark.array_sha256(train) == benchmark.array_sha256(heldout):
        raise ValueError("training and held-out reward schedules unexpectedly match")
    return train, heldout


def _evaluate_reward_mse(
    reward_params: Any,
    frozen_world: Any,
    arrays: Mapping[str, np.ndarray],
    schedule: Mapping[str, np.ndarray],
    config: Any,
    *,
    seed: int,
) -> dict[str, float]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        jit_observe_sequence,
        predict_reward,
        transition_reward_loss,
    )

    execution = benchmark._profile_execution("pilot")
    sequence_length = int(execution["sequence_length"])
    direct_squared = 0.0
    source_squared = 0.0
    count = 0.0
    for index in range(int(schedule["episode_ids"].shape[0])):
        batch = benchmark._materialize_batch(
            arrays,
            schedule,
            index,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        key = benchmark.derive_jax_key(
            "direct-transition-reward-evaluation", seed, index
        )
        direct = transition_reward_loss(
            reward_params, frozen_world, batch, key, config
        )
        sequence = jit_observe_sequence(
            frozen_world,
            batch["observations"],
            batch["actions"],
            key,
            config,
            is_first=batch["is_first"],
        )
        source_prediction = predict_reward(
            frozen_world, sequence.states.feature, config
        )
        mask = np.asarray(batch["loss_mask"], dtype=np.float64)
        targets = np.asarray(batch["rewards"], dtype=np.float64)
        direct_prediction = np.asarray(jax.device_get(direct.predictions), dtype=np.float64)
        source_prediction = np.asarray(jax.device_get(source_prediction), dtype=np.float64)
        direct_squared += float(np.sum(np.square(direct_prediction - targets) * mask))
        source_squared += float(np.sum(np.square(source_prediction - targets) * mask))
        count += float(np.sum(mask))
    if count <= 0.0:
        raise ValueError("held-out reward schedule has no valid targets")
    metrics = {
        "direct_transition_mse": direct_squared / count,
        "source_state_only_mse": source_squared / count,
        "valid_targets": count,
    }
    if not _finite_tree(metrics):
        raise FloatingPointError("held-out reward metrics are not finite")
    return metrics


def train_reward_cell(
    corrected_root: str | Path,
    output_root: str | Path,
    index: int,
    **settings: Any,
) -> dict[str, Any]:
    """Fit one direct scalar reward head while preserving the source model."""

    import jax
    import numpy as onp
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        TransitionRewardConfig,
        attach_transition_reward_head,
        create_agent,
        init_transition_reward_state,
        jit_train_transition_reward_step,
        load_checkpoint,
        save_checkpoint,
    )
    from imf_dreamer_jax.nn import tree_global_norm

    manifest = write_manifest(corrected_root, output_root, **settings)
    cell = _cell(manifest, "reward", index)
    result_path = Path(output_root) / str(cell["result_path"])
    if result_path.is_file():
        return read_json(result_path)
    seed = int(cell["world_model_seed"])
    source = _source(manifest, seed)
    source_path = Path(str(source["world_model_checkpoint"]))
    source_state, _, _ = load_checkpoint(source_path)
    config = _source_config(source)
    frozen_world = source_state.params.world_model
    source_digest = benchmark._tree_digest(frozen_world)
    if source_digest != source["world_model_parameter_sha256"]:
        raise ValueError("loaded source world-model parameter digest differs")
    arrays = benchmark.load_npz(source["dataset"])
    if benchmark.array_sha256(arrays) != source["dataset_sha256"]:
        raise ValueError("loaded reward dataset payload differs")
    train_schedule, heldout_schedule = _reward_schedules(
        arrays, manifest, seed
    )
    directory = result_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = Path(output_root) / str(cell["schedule_path"])
    heldout_path = Path(output_root) / str(cell["heldout_schedule_path"])
    benchmark._write_npz_atomic(schedule_path, train_schedule)
    benchmark._write_npz_atomic(heldout_path, heldout_schedule)
    reward_state = init_transition_reward_state(
        config,
        benchmark.derive_jax_key("direct-transition-reward-init", seed),
    )
    initial_reward = jax.tree_util.tree_map(
        lambda value: value.copy(), reward_state.params
    )
    initial_metrics = _evaluate_reward_mse(
        reward_state.params,
        frozen_world,
        arrays,
        heldout_schedule,
        config,
        seed=seed,
    )
    objective = TransitionRewardConfig(
        learning_rate=float(manifest["reward_learning_rate"]),
        grad_clip=float(manifest["reward_grad_clip"]),
    )
    execution = benchmark._profile_execution("pilot")
    sequence_length = int(execution["sequence_length"])
    train_key = benchmark.derive_jax_key("direct-transition-reward-loss", seed)
    latest = None
    started = time.perf_counter()
    for update in range(int(manifest["reward_updates"])):
        batch = benchmark._materialize_batch(
            arrays,
            train_schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        reward_state, metrics = jit_train_transition_reward_step(
            reward_state,
            frozen_world,
            batch,
            jax.random.fold_in(train_key, update),
            config,
            objective,
        )
        latest = {
            "loss": float(onp.asarray(metrics.loss)),
            "grad_norm": float(onp.asarray(metrics.grad_norm)),
        }
    if latest is None or not _finite_tree(latest):
        raise FloatingPointError("transition reward training produced no finite update")
    final_metrics = _evaluate_reward_mse(
        reward_state.params,
        frozen_world,
        arrays,
        heldout_schedule,
        config,
        seed=seed,
    )
    head_delta = float(
        tree_global_norm(
            jax.tree_util.tree_map(
                lambda before, after: after - before,
                initial_reward,
                reward_state.params,
            )
        )
    )
    if head_delta <= 0.0:
        raise RuntimeError("direct transition reward parameters did not update")
    fresh = create_agent(
        config, benchmark.derive_jax_key("direct-transition-reward-shell", seed)
    )
    shell = AgentState(
        AgentParams(frozen_world, fresh.params.actor, fresh.params.critic),
        source_state.model_optimizer,
        fresh.actor_optimizer,
        fresh.critic_optimizer,
        fresh.slow_critic,
        source_state.world_model_teacher,
    )
    attached = attach_transition_reward_head(shell, reward_state)
    if benchmark._tree_digest(frozen_world) != source_digest:
        raise RuntimeError("reward training changed the source world model")
    for name, subtree in frozen_world.items():
        if benchmark._tree_digest(subtree) != benchmark._tree_digest(
            attached.params.world_model[name]
        ):
            raise RuntimeError(f"reward attachment changed source subtree {name}")
    checkpoint_path = Path(output_root) / str(cell["checkpoint_path"])
    save_checkpoint(
        checkpoint_path,
        attached,
        config,
        metadata={
            "stage": "direct_transition_reward",
            "cell_id": cell["cell_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "source_world_model_checkpoint_sha256": source[
                "world_model_checkpoint_sha256"
            ],
            "completed_reward_updates": int(manifest["reward_updates"]),
        },
    )
    result = {
        "schema_version": REWARD_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "arm": "trajectory_imf",
        "world_model_seed": seed,
        "reward_updates": int(manifest["reward_updates"]),
        "source_world_model_checkpoint_sha256": source[
            "world_model_checkpoint_sha256"
        ],
        "source_world_model_parameter_sha256": source_digest,
        "source_world_model_parameter_delta": 0.0,
        "trainable_subtrees": ["reward_transition"],
        "reward_parameter_delta": head_delta,
        "reward_parameter_count": int(
            sum(value.size for value in jax.tree_util.tree_leaves(reward_state.params))
        ),
        "final_training_metrics": latest,
        "initial_heldout_metrics": initial_metrics,
        "final_heldout_metrics": final_metrics,
        "checkpoint_sha256": benchmark.file_sha256(checkpoint_path),
        "training_schedule_file_sha256": benchmark.file_sha256(schedule_path),
        "training_schedule_sha256": benchmark.array_sha256(train_schedule),
        "heldout_schedule_file_sha256": benchmark.file_sha256(heldout_path),
        "heldout_schedule_sha256": benchmark.array_sha256(heldout_schedule),
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if not _finite_tree(result):
        raise FloatingPointError("transition reward result contains non-finite values")
    write_json_atomic(result_path, result)
    return result


def verify_reward_cell(output_root: str | Path, index: int) -> dict[str, Any]:
    """Authenticate one fitted reward head and its frozen source subtrees."""

    from imf_dreamer_jax import load_checkpoint

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell(manifest, "reward", index)
    result_path = root / str(cell["result_path"])
    result = read_json(result_path)
    source = _source(manifest, int(cell["world_model_seed"]))
    required_identity = {
        "schema_version": REWARD_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "arm": "trajectory_imf",
        "world_model_seed": int(cell["world_model_seed"]),
    }
    if any(result.get(key) != value for key, value in required_identity.items()):
        raise ValueError("transition reward result identity differs")
    checkpoint = root / str(cell["checkpoint_path"])
    train_schedule = root / str(cell["schedule_path"])
    heldout_schedule = root / str(cell["heldout_schedule_path"])
    if (
        result.get("source_world_model_parameter_delta") != 0.0
        or result.get("trainable_subtrees") != ["reward_transition"]
        or float(result.get("reward_parameter_delta", 0.0)) <= 0.0
        or int(result.get("reward_updates", -1)) != int(manifest["reward_updates"])
        or result.get("checkpoint_sha256") != benchmark.file_sha256(checkpoint)
        or result.get("training_schedule_file_sha256")
        != benchmark.file_sha256(train_schedule)
        or result.get("heldout_schedule_file_sha256")
        != benchmark.file_sha256(heldout_schedule)
        or result.get("training_schedule_sha256")
        != benchmark.array_sha256(benchmark.load_npz(train_schedule))
        or result.get("heldout_schedule_sha256")
        != benchmark.array_sha256(benchmark.load_npz(heldout_schedule))
        or not _finite_tree(result)
    ):
        raise ValueError("transition reward result contract or files differ")
    state, config, metadata = load_checkpoint(checkpoint)
    source_state, _, _ = load_checkpoint(
        source["world_model_checkpoint"]
    )
    if (
        config != _source_config(source)
        or metadata.get("stage") != "direct_transition_reward"
        or metadata.get("cell_id") != cell["cell_id"]
        or metadata.get("manifest_sha256") != manifest["manifest_sha256"]
        or int(metadata.get("completed_reward_updates", -1))
        != int(manifest["reward_updates"])
        or set(state.params.world_model) - set(source_state.params.world_model)
        != {"reward_transition"}
    ):
        raise ValueError("transition reward checkpoint structure differs")
    for name, subtree in source_state.params.world_model.items():
        if benchmark._tree_digest(subtree) != benchmark._tree_digest(
            state.params.world_model[name]
        ):
            raise ValueError(f"transition reward changed source subtree {name}")
    marker = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": "reward",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(result_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
    }
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / str(cell["marker_path"])
    if marker_path.is_file() and read_json(marker_path) != marker:
        raise ValueError("existing transition reward marker differs")
    write_json_atomic(marker_path, marker)
    return marker


def _actor_checkpoint_metadata(
    *,
    cell: Mapping[str, Any],
    reward_result: Mapping[str, Any],
    preparation_updates: int,
    actor_updates: int,
    return_scale_state: Any,
    latest: Mapping[str, Any] | None,
    latest_bc: float,
    latest_replay_critic: float,
    wall_seconds: float,
) -> dict[str, Any]:
    return {
        "stage": "direct_transition_reward_actor",
        "cell_id": cell["cell_id"],
        "reward_checkpoint_sha256": reward_result["checkpoint_sha256"],
        "completed_preparation_updates": int(preparation_updates),
        "completed_actor_updates": int(actor_updates),
        "return_scale_percentile_range": float(return_scale_state.percentile_range),
        "return_scale_initialized": bool(return_scale_state.initialized),
        "last_metrics": None if latest is None else dict(latest),
        "last_behavior_cloning_loss": float(latest_bc),
        "last_replay_critic_loss": float(latest_replay_critic),
        "wall_seconds": float(wall_seconds),
    }


def train_actor_cell(
    corrected_root: str | Path,
    output_root: str | Path,
    index: int,
    **settings: Any,
) -> dict[str, Any]:
    """Run the corrected actor loop with only the iMF reward head changed."""

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

    manifest = write_manifest(corrected_root, output_root, **settings)
    cell = _cell(manifest, "actor", index)
    result_path = Path(output_root) / str(cell["result_path"])
    if result_path.is_file():
        return read_json(result_path)
    seed = int(cell["world_model_seed"])
    actor_seed = int(cell["actor_seed"])
    reference = _actor_reference(manifest, seed, actor_seed)
    reward_cell = next(
        row
        for row in manifest["reward_cells"]
        if int(row["world_model_seed"]) == seed
    )
    reward_result = read_json(Path(output_root) / str(reward_cell["result_path"]))
    reward_checkpoint = Path(output_root) / str(reward_cell["checkpoint_path"])
    if benchmark.file_sha256(reward_checkpoint) != reward_result["checkpoint_sha256"]:
        raise ValueError("actor reward checkpoint digest differs")
    world_state, config, _ = load_checkpoint(reward_checkpoint)
    fresh = create_agent(
        config,
        benchmark.derive_jax_key("actor-init", TASK, seed, actor_seed),
    )
    initial_actor_digest = benchmark._tree_digest(fresh.params.actor)
    initial_critic_digest = benchmark._tree_digest(fresh.params.critic)
    for arm in ("trajectory_imf", "shortcut_forcing"):
        if (
            reference[arm]["initial_actor_parameter_sha256"]
            != initial_actor_digest
            or reference[arm]["initial_critic_parameter_sha256"]
            != initial_critic_digest
        ):
            raise ValueError("reward-only actor initialization is not paired")
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
    source = _source(manifest, seed)
    arrays = benchmark.load_npz(source["dataset"])
    schedule = benchmark.load_npz(reference["trajectory_imf"]["schedule"])
    if (
        benchmark.array_sha256(schedule)
        != reference["trajectory_imf"]["schedule_sha256"]
    ):
        raise ValueError("paired actor schedule payload differs")
    directory = result_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = Path(output_root) / str(cell["schedule_path"])
    benchmark._write_npz_atomic(schedule_path, schedule)
    checkpoint_path = Path(output_root) / str(cell["checkpoint_path"])
    prep_done = 0
    actor_done = 0
    accumulated_wall = 0.0
    latest = None
    latest_bc = math.nan
    latest_replay_critic = math.nan
    return_scale_state = init_return_scale_state()
    if checkpoint_path.is_file():
        state, stored_config, metadata = load_checkpoint(checkpoint_path)
        if stored_config != config or metadata.get("cell_id") != cell["cell_id"]:
            raise ValueError("partial transition-reward actor checkpoint differs")
        prep_done = int(metadata.get("completed_preparation_updates", -1))
        actor_done = int(metadata.get("completed_actor_updates", -1))
        accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        latest = metadata.get("last_metrics")
        latest_bc = float(metadata.get("last_behavior_cloning_loss", math.nan))
        latest_replay_critic = float(
            metadata.get("last_replay_critic_loss", math.nan)
        )
        return_scale_state = ReturnScaleState(
            jnp.asarray(metadata.get("return_scale_percentile_range", 0.0), jnp.float32),
            jnp.asarray(metadata.get("return_scale_initialized", False)),
        )
    training = manifest["actor_training"]
    total_prep = int(training["preparation_updates"])
    total_actor = int(training["actor_updates"])
    if not (0 <= prep_done <= total_prep and 0 <= actor_done <= total_actor):
        raise ValueError("partial actor counters are invalid")
    if prep_done != total_prep and actor_done != 0:
        raise ValueError("actor updates began before preparation completed")
    execution = benchmark._profile_execution("pilot")
    sequence_length = int(execution["sequence_length"])
    checkpoint_every = int(execution["checkpoint_every_updates"])
    posterior_key = benchmark.derive_jax_key(
        "correct-actor-posterior", TASK, seed, actor_seed
    )
    start_key = benchmark.derive_jax_key(
        "correct-actor-start", TASK, seed, actor_seed
    )
    objective_key = benchmark.derive_jax_key(
        "correct-actor-objective", TASK, seed, actor_seed
    )
    started = time.perf_counter()

    def save_progress() -> None:
        save_checkpoint(
            checkpoint_path,
            state,
            config,
            metadata=_actor_checkpoint_metadata(
                cell=cell,
                reward_result=reward_result,
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
        raise FloatingPointError("reward-only actor produced no finite telemetry")
    if benchmark._tree_digest(state.params.world_model) != frozen_world_digest:
        raise RuntimeError("actor training changed the reward-adapted world model")
    episodes = int(
        read_json(Path(manifest["corrected_root"]) / "manifest.json")[
            "real_environment_evaluation_episodes"
        ]
    )
    maximum_steps = int(
        read_json(Path(manifest["corrected_root"]) / "manifest.json")[
            "maximum_environment_steps"
        ]
    )
    returns, traces = benchmark._evaluate_actor_policy(
        state,
        config,
        task=TASK,
        world_model_seed=seed,
        actor_seed=actor_seed,
        episodes=episodes,
        maximum_steps=maximum_steps,
    )
    if [int(value) for value in traces["evaluation_seeds"]] != reference[
        "trajectory_imf"
    ]["evaluation_seeds"]:
        raise RuntimeError("reward-only actor evaluation seeds are not paired")
    trace_path = Path(output_root) / str(cell["trace_path"])
    benchmark._write_npz_atomic(trace_path, traces)
    total_wall = accumulated_wall + time.perf_counter() - started
    save_progress()
    normalized = np.asarray(returns, dtype=np.float64) / 1000.0
    lengths = np.asarray(traces["lengths"], dtype=np.int64)
    actions = np.asarray(traces["actions"], dtype=np.float32)
    mask = np.arange(actions.shape[1])[None, :] < lengths[:, None]
    result = {
        "schema_version": ACTOR_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "arm": "trajectory_imf_direct_transition_reward",
        "world_model_seed": seed,
        "actor_seed": actor_seed,
        "preparation_updates": total_prep,
        "actor_updates": total_actor,
        "reward_checkpoint_sha256": reward_result["checkpoint_sha256"],
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
        "action_saturation_fraction": float(
            np.mean(np.abs(actions)[mask] >= 0.95) if np.any(mask) else 0.0
        ),
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
        "reference_trajectory_imf_result_sha256": reference["trajectory_imf"][
            "result_sha256"
        ],
        "reference_shortcut_result_sha256": reference["shortcut_forcing"][
            "result_sha256"
        ],
        "wall_seconds": total_wall,
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if not _finite_tree(result):
        raise FloatingPointError("reward-only actor result contains non-finite values")
    write_json_atomic(result_path, result)
    return result


def verify_actor_cell(
    output_root: str | Path,
    index: int,
    *,
    strict_replay: bool = True,
) -> dict[str, Any]:
    """Authenticate an actor cell and optionally replay its real evaluation."""

    from imf_dreamer_jax import load_checkpoint

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell(manifest, "actor", index)
    result_path = root / str(cell["result_path"])
    result = read_json(result_path)
    identity = {
        "schema_version": ACTOR_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "arm": "trajectory_imf_direct_transition_reward",
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
    }
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError("transition-reward actor result identity differs")
    checkpoint = root / str(cell["checkpoint_path"])
    schedule_path = root / str(cell["schedule_path"])
    trace_path = root / str(cell["trace_path"])
    reference = _actor_reference(
        manifest, int(cell["world_model_seed"]), int(cell["actor_seed"])
    )
    if (
        result.get("world_model_frozen") is not True
        or result.get("world_model_parameter_delta") != 0.0
        or result.get("behavior_prior_used") is not False
        or int(result.get("preparation_updates", -1))
        != int(manifest["actor_training"]["preparation_updates"])
        or int(result.get("actor_updates", -1))
        != int(manifest["actor_training"]["actor_updates"])
        or result.get("checkpoint_sha256") != benchmark.file_sha256(checkpoint)
        or result.get("batch_schedule_file_sha256")
        != benchmark.file_sha256(schedule_path)
        or result.get("batch_schedule_sha256")
        != reference["trajectory_imf"]["schedule_sha256"]
        or result.get("raw_action_traces_sha256")
        != benchmark.file_sha256(trace_path)
        or result.get("initial_actor_parameter_sha256")
        != reference["trajectory_imf"]["initial_actor_parameter_sha256"]
        or result.get("initial_critic_parameter_sha256")
        != reference["trajectory_imf"]["initial_critic_parameter_sha256"]
        or not _finite_tree(result)
    ):
        raise ValueError("transition-reward actor contract or artifact differs")
    state, config, metadata = load_checkpoint(checkpoint)
    reward_cell = next(
        row
        for row in manifest["reward_cells"]
        if int(row["world_model_seed"]) == int(cell["world_model_seed"])
    )
    reward_checkpoint = root / str(reward_cell["checkpoint_path"])
    reward_state, reward_config, _ = load_checkpoint(reward_checkpoint)
    if (
        config != reward_config
        or metadata.get("stage") != "direct_transition_reward_actor"
        or metadata.get("cell_id") != cell["cell_id"]
        or benchmark._tree_digest(state.params.world_model)
        != benchmark._tree_digest(reward_state.params.world_model)
        or result.get("final_actor_parameter_sha256")
        != benchmark._tree_digest(state.params.actor)
        or result.get("final_critic_parameter_sha256")
        != benchmark._tree_digest(state.params.critic)
    ):
        raise ValueError("transition-reward actor checkpoint differs")
    traces = benchmark.load_npz(trace_path)
    required = {
        "actions",
        "rewards",
        "continuations",
        "is_last",
        "lengths",
        "evaluation_seeds",
    }
    if set(traces) != required or [
        int(value) for value in traces["evaluation_seeds"]
    ] != reference["trajectory_imf"]["evaluation_seeds"]:
        raise ValueError("transition-reward actor traces are unpaired")
    lengths = np.asarray(traces["lengths"], dtype=np.int64)
    rewards = np.asarray(traces["rewards"], dtype=np.float64)
    recomputed = np.asarray(
        [float(np.sum(rewards[row, :length])) for row, length in enumerate(lengths)]
    )
    if not np.array_equal(
        recomputed, np.asarray(result["episode_returns"], dtype=np.float64)
    ):
        raise ValueError("transition-reward actor returns do not match traces")
    if strict_replay:
        corrected_manifest = read_json(
            Path(manifest["corrected_root"]) / "manifest.json"
        )
        benchmark._validate_actor_environment_replay(
            traces,
            recomputed,
            state=state,
            config=config,
            task=TASK,
            world_model_seed=int(cell["world_model_seed"]),
            actor_seed=int(cell["actor_seed"]),
            maximum_steps=int(corrected_manifest["maximum_environment_steps"]),
            action_repeat=1,
        )
    marker = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": "actor",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(result_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "strict_policy_and_environment_replay": bool(strict_replay),
    }
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / str(cell["marker_path"])
    if marker_path.is_file() and read_json(marker_path) != marker:
        raise ValueError("existing transition-reward actor marker differs")
    write_json_atomic(marker_path, marker)
    return marker


def _validate_marker(
    root: Path,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    stage: str,
) -> Mapping[str, Any]:
    marker = read_json(root / str(cell["marker_path"]))
    result_path = root / str(cell["result_path"])
    expected = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": stage,
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(result_path),
        "checkpoint_sha256": read_json(result_path)["checkpoint_sha256"],
    }
    if stage == "actor":
        expected["strict_policy_and_environment_replay"] = True
    observed = dict(marker)
    digest = observed.pop("marker_sha256", None)
    if observed != expected or digest != benchmark.object_sha256(observed):
        raise ValueError(f"{stage} marker differs")
    return marker


def finalize(output_root: str | Path) -> dict[str, Any]:
    """Aggregate paired Reacher returns without promoting exploratory evidence."""

    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    reward_results = []
    for cell in manifest["reward_cells"]:
        _validate_marker(root, manifest, cell, "reward")
        reward_results.append(read_json(root / str(cell["result_path"])))
    actor_results = []
    for cell in manifest["actor_cells"]:
        _validate_marker(root, manifest, cell, "actor")
        actor_results.append(read_json(root / str(cell["result_path"])))
    paired_units = []
    for world_seed in manifest["world_model_seeds"]:
        direct_rows = [
            row
            for row in actor_results
            if int(row["world_model_seed"]) == int(world_seed)
        ]
        references = [
            row
            for row in manifest["actor_references"]
            if int(row["world_model_seed"]) == int(world_seed)
        ]
        if len(direct_rows) != 2 or len(references) != 2:
            raise ValueError("paired world-seed actor unit is incomplete")
        direct = float(np.mean([row["normalized_episode_return_mean"] for row in direct_rows]))
        trajectory = float(
            np.mean(
                [
                    row["trajectory_imf"]["normalized_episode_return_mean"]
                    for row in references
                ]
            )
        )
        shortcut = float(
            np.mean(
                [
                    row["shortcut_forcing"]["normalized_episode_return_mean"]
                    for row in references
                ]
            )
        )
        paired_units.append(
            {
                "task": TASK,
                "world_model_seed": int(world_seed),
                "direct_transition_reward": direct,
                "original_trajectory_imf": trajectory,
                "shortcut_forcing": shortcut,
                "direct_minus_original_imf": direct - trajectory,
                "direct_minus_shortcut": direct - shortcut,
            }
        )
    iqm = {
        name: benchmark.interquartile_mean([float(row[name]) for row in paired_units])
        for name in (
            "direct_transition_reward",
            "original_trajectory_imf",
            "shortcut_forcing",
        )
    }
    direct_minus_original = iqm["direct_transition_reward"] - iqm[
        "original_trajectory_imf"
    ]
    direct_minus_shortcut = iqm["direct_transition_reward"] - iqm[
        "shortcut_forcing"
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "task": TASK,
        "completed_reward_cells": len(reward_results),
        "completed_actor_cells": len(actor_results),
        "strict_actor_replay_markers": len(actor_results),
        "paired_units": paired_units,
        "normalized_return_iqm": iqm,
        "direct_minus_original_imf_iqm": direct_minus_original,
        "direct_minus_shortcut_iqm": direct_minus_shortcut,
        "favorable_world_seed_fraction_vs_original_imf": float(
            np.mean([row["direct_minus_original_imf"] > 0.0 for row in paired_units])
        ),
        "favorable_world_seed_fraction_vs_shortcut": float(
            np.mean([row["direct_minus_shortcut"] > 0.0 for row in paired_units])
        ),
        "reward_heldout_mse": {
            "source_state_only_mean": float(
                np.mean(
                    [
                        row["final_heldout_metrics"]["source_state_only_mse"]
                        for row in reward_results
                    ]
                )
            ),
            "direct_transition_mean": float(
                np.mean(
                    [
                        row["final_heldout_metrics"]["direct_transition_mse"]
                        for row in reward_results
                    ]
                )
            ),
        },
        "actor_stability": {
            "mean_action_saturation_fraction": float(
                np.mean([row["action_saturation_fraction"] for row in actor_results])
            ),
            "mean_actor_grad_norm": float(
                np.mean([row["final_metrics"]["actor_grad_norm"] for row in actor_results])
            ),
            "mean_imagined_real_per_step_return_gap": float(
                np.mean(
                    [row["imagined_real_per_step_return_gap"] for row in actor_results]
                )
            ),
        },
        "total_reward_wall_seconds": float(
            sum(row["wall_seconds"] for row in reward_results)
        ),
        "total_actor_wall_seconds": float(
            sum(row["wall_seconds"] for row in actor_results)
        ),
        "claim_status": "exploratory_selected_reacher_intervention_only",
        "supports_reward_head_hypothesis": bool(
            direct_minus_original > 0.0 and direct_minus_shortcut > 0.0
        ),
        "limitations": [
            "Reacher only; no task-level replication",
            "selected pilot candidates and pilot seeds are reused, so this is diagnostic evidence",
            "shortcut is an immutable paired reference and was not retrained",
            "the intervention adds a small direct reward MLP while freezing all source dynamics",
        ],
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["report_sha256"] = benchmark.object_sha256(report)
    if not _finite_tree(report):
        raise FloatingPointError("transition reward report contains non-finite values")
    path = root / "report.json"
    if path.is_file() and read_json(path) != report:
        raise ValueError("existing transition reward report differs")
    write_json_atomic(path, report)
    return report


def validate_final(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    report = read_json(root / "report.json")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("completed_reward_cells") != EXPECTED_REWARD_CELLS
        or report.get("completed_actor_cells") != EXPECTED_ACTOR_CELLS
        or report.get("strict_actor_replay_markers") != EXPECTED_ACTOR_CELLS
        or report.get("claim_status")
        != "exploratory_selected_reacher_intervention_only"
        or report.get("report_sha256")
        != _unsigned_digest(report, "report_sha256")
        or not _finite_tree(report)
    ):
        raise ValueError("transition reward final report is invalid")
    return report


def aggregate_synthetic_units(units: list[Mapping[str, float]]) -> dict[str, float]:
    """Small pure helper used to verify paired aggregation logic."""

    required = {
        "direct_transition_reward",
        "original_trajectory_imf",
        "shortcut_forcing",
    }
    if not units or any(not required.issubset(row) for row in units):
        raise ValueError("synthetic paired units are incomplete")
    return {
        name: benchmark.interquartile_mean([float(row[name]) for row in units])
        for name in sorted(required)
    }


def self_test() -> None:
    units = [
        {
            "direct_transition_reward": 0.4,
            "original_trajectory_imf": 0.2,
            "shortcut_forcing": 0.3,
        },
        {
            "direct_transition_reward": 0.6,
            "original_trajectory_imf": 0.5,
            "shortcut_forcing": 0.45,
        },
        {
            "direct_transition_reward": 0.5,
            "original_trajectory_imf": 0.3,
            "shortcut_forcing": 0.4,
        },
    ]
    summary = aggregate_synthetic_units(units)
    expected = {
        "direct_transition_reward": 0.5,
        "original_trajectory_imf": 0.31666666666666665,
        "shortcut_forcing": 0.39166666666666666,
    }
    if any(not math.isclose(summary[name], value) for name, value in expected.items()):
        raise AssertionError("paired aggregation positive control failed")
    for invalid in ([], [{"direct_transition_reward": 1.0}]):
        try:
            aggregate_synthetic_units(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("paired aggregation accepted incomplete evidence")


__all__ = [
    "ACTOR_RESULT_SCHEMA",
    "EXPECTED_ACTOR_CELLS",
    "EXPECTED_REWARD_CELLS",
    "HELDOUT_BATCHES",
    "MARKER_SCHEMA",
    "REPORT_SCHEMA",
    "REWARD_RESULT_SCHEMA",
    "REWARD_UPDATES",
    "SCHEMA",
    "TASK",
    "aggregate_synthetic_units",
    "build_manifest",
    "finalize",
    "self_test",
    "train_actor_cell",
    "train_reward_cell",
    "validate_final",
    "validate_manifest",
    "verify_actor_cell",
    "verify_reward_cell",
    "write_manifest",
]
