"""Small intervention study for action-consistent trajectory iMF on Reacher.

This module deliberately does not reopen or mutate the frozen 606-cell pilot.
It reuses its authenticated Reacher datasets and shortcut results, augments the
training data with paired simulator probes from exactly restored states, and
trains only the selected trajectory-iMF candidate.  The resulting comparison
is exploratory because the new arm receives additional counterfactual labels.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import read_json, write_json_atomic
from .dmc import DMCAdapter
from . import matched_objective_benchmark as benchmark
from .policy_alignment_diagnostics import aggregate_metric_rows


SCHEMA = "causal-reacher-small-v1"
TASK = "dmc_reacher_easy"
WORLD_MODEL_SEEDS = (211, 223)
ACTOR_SEED = 311
ACTION_DELTA = 0.1
CAUSAL_SCALE = 0.1
CAUSAL_REWARD_SCALE = 1.0
CAUSAL_HUBER_DELTA = 1.0
CAUSAL_NORMALIZATION_EPSILON = 1e-3


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def _stage_cells(matrix: Mapping[str, Any], stage: str) -> list[Mapping[str, Any]]:
    return [cell for cell in matrix["cells"] if cell["stage"] == stage]


def _one_cell(
    matrix: Mapping[str, Any],
    stage: str,
    **fields: Any,
) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in _stage_cells(matrix, stage)
        if all(cell.get(name) == value for name, value in fields.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {stage} cell for {fields}, found {len(matches)}")
    return matches[0]


def _pilot_artifacts(pilot_root: str | Path) -> tuple[dict[str, Any], ...]:
    root = Path(pilot_root)
    protocol = read_json(root / "frozen_protocol.json")
    matrix = read_json(root / "matrix.json")
    selection = read_json(root / "hpo_selection.json")
    if selection.get("status") != "complete":
        raise ValueError("the immutable pilot selection is incomplete")
    if set(selection.get("selected", {})) != {"shortcut_forcing", "trajectory_imf"}:
        raise ValueError("the immutable pilot selection has unexpected arms")
    return protocol, matrix, selection


def build_manifest(pilot_root: str | Path) -> dict[str, Any]:
    """Bind the exact small run and every read-only shortcut comparator."""

    root = Path(pilot_root).resolve()
    protocol, matrix, selection = _pilot_artifacts(root)
    trajectory = selection["selected"]["trajectory_imf"]
    shortcut = selection["selected"]["shortcut_forcing"]
    scheduled = []
    comparator = []
    for seed in WORLD_MODEL_SEEDS:
        dataset = _one_cell(
            matrix, "dataset", task=TASK, world_model_seed=seed
        )
        dataset_directory = root / "dataset" / dataset["cell_id"]
        dataset_result = read_json(dataset_directory / "result.json")
        if dataset_result["dataset_file_sha256"] != benchmark.file_sha256(
            dataset_directory / "dataset.npz"
        ):
            raise ValueError("pilot dataset digest mismatch")
        scheduled.append(
            {
                "arm": "trajectory_imf",
                "task": TASK,
                "world_model_seed": seed,
                "actor_seed": ACTOR_SEED,
                "dataset_cell_id": dataset["cell_id"],
                "dataset_sha256": dataset_result["dataset_sha256"],
                "candidate_id": trajectory["candidate_id"],
                "candidate_overrides": trajectory["overrides"],
                "world_model_updates": int(
                    protocol["profiles"]["pilot"]["budgets"]["equal_updates"][
                        "world_model_updates"
                    ]
                ),
                "actor_updates": int(
                    protocol["profiles"]["pilot"]["budgets"]["equal_updates"][
                        "actor_updates"
                    ]
                ),
            }
        )
        shortcut_world = _one_cell(
            matrix,
            "world_model",
            task=TASK,
            world_model_seed=seed,
            arm="shortcut_forcing",
            candidate_id=shortcut["candidate_id"],
            budget_track="equal_updates",
        )
        shortcut_rollout = _one_cell(
            matrix,
            "rollout",
            task=TASK,
            world_model_seed=seed,
            arm="shortcut_forcing",
            candidate_id=shortcut["candidate_id"],
            budget_track="equal_updates",
            nfe=int(protocol["evaluation"]["primary_nfe"]["shortcut_forcing"]),
        )
        shortcut_actor = _one_cell(
            matrix,
            "actor",
            task=TASK,
            world_model_seed=seed,
            actor_seed=ACTOR_SEED,
            arm="shortcut_forcing",
            candidate_id=shortcut["candidate_id"],
            budget_track="equal_updates",
        )
        rows = []
        for cell in (shortcut_world, shortcut_rollout, shortcut_actor):
            result_path = root / cell["stage"] / cell["cell_id"] / "result.json"
            rows.append(
                {
                    "stage": cell["stage"],
                    "cell_id": cell["cell_id"],
                    "result_path": str(result_path),
                    "result_sha256": benchmark.file_sha256(result_path),
                }
            )
        comparator.append(
            {
                "arm": "shortcut_forcing",
                "mode": "reference_only_never_scheduled",
                "task": TASK,
                "world_model_seed": seed,
                "actor_seed": ACTOR_SEED,
                "candidate_id": shortcut["candidate_id"],
                "artifacts": rows,
            }
        )
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "pilot_root": str(root),
        "pilot_source_sha256": matrix["source_sha256"],
        "pilot_protocol_sha256": matrix["protocol_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "task": TASK,
        "world_model_seeds": list(WORLD_MODEL_SEEDS),
        "actor_seeds": [ACTOR_SEED],
        "scheduled_cells": scheduled,
        "shortcut_baseline": comparator,
        "causal_objective": {
            "finite_difference_action_delta": ACTION_DELTA,
            "scale": CAUSAL_SCALE,
            "reward_scale": CAUSAL_REWARD_SCALE,
            "pseudo_huber_delta": CAUSAL_HUBER_DELTA,
            "normalization_epsilon": CAUSAL_NORMALIZATION_EPSILON,
            "paired_common_random_numbers": True,
            "paired_simulator_state_restore": True,
            "target": "one_step_observation_and_reward_action_effect",
        },
        "interpretation": {
            "evidence_class": "exploratory_intervention_diagnostic",
            "claim_eligible": False,
            "additional_counterfactual_labels_for_new_imf_only": True,
            "fair_objective_only_superiority_comparison": False,
        },
    }
    manifest["manifest_sha256"] = benchmark.object_sha256(manifest)
    return manifest


def write_manifest(pilot_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    path = root / "manifest.json"
    manifest = build_manifest(pilot_root)
    if path.is_file():
        existing = read_json(path)
        if existing != manifest:
            raise ValueError("existing small-study manifest differs from the frozen plan")
        return existing
    write_json_atomic(path, manifest)
    return manifest


def _scheduled_row(manifest: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    matches = [
        row
        for row in manifest["scheduled_cells"]
        if int(row["world_model_seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise ValueError("seed is outside the frozen small-study manifest")
    return matches[0]


def prepare_seed(
    pilot_root: str | Path,
    output_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Replay the pilot dataset and collect paired local interventions."""

    output = Path(output_root)
    manifest = write_manifest(pilot_root, output)
    row = _scheduled_row(manifest, seed)
    directory = output / "dataset" / f"seed-{seed}"
    result_path = directory / "result.json"
    probe_path = directory / "causal_probes.npz"
    copied_dataset_path = directory / "dataset.npz"
    if result_path.is_file():
        result = read_json(result_path)
        if (
            result.get("status") == "complete"
            and result.get("probe_file_sha256") == benchmark.file_sha256(probe_path)
            and result.get("dataset_file_sha256")
            == benchmark.file_sha256(copied_dataset_path)
        ):
            return result
        raise ValueError("existing causal-probe result is incomplete or invalid")

    pilot_dataset_directory = (
        Path(pilot_root) / "dataset" / row["dataset_cell_id"]
    )
    arrays = benchmark.load_npz(pilot_dataset_directory / "dataset.npz")
    if benchmark.array_sha256(arrays) != row["dataset_sha256"]:
        raise ValueError("pilot dataset payload differs from the manifest")
    episodes, transitions = arrays["actions"].shape[:2]
    action_dim = arrays["actions"].shape[-1]
    lower_actions = np.asarray(arrays["actions"], dtype=np.float32).copy()
    upper_actions = lower_actions.copy()
    lower_observations = np.asarray(arrays["observations"], dtype=np.float32).copy()
    upper_observations = lower_observations.copy()
    lower_rewards = np.asarray(arrays["rewards"], dtype=np.float32).copy()
    upper_rewards = lower_rewards.copy()
    causal_mask = np.zeros((episodes, transitions), dtype=np.float32)
    train_ids = set(np.asarray(arrays["train_episode_ids"], dtype=np.int64).tolist())
    random = np.random.default_rng(
        benchmark.derive_seed("causal-probe-directions", TASK, seed)
    )
    environment = DMCAdapter(
        TASK,
        seed=benchmark.derive_seed("dataset-environment", TASK, seed),
        action_repeat=1,
    )
    if environment.action_dim != action_dim:
        raise ValueError("pilot dataset and Reacher action dimensions differ")
    maximum_replay_error = 0.0
    started = time.perf_counter()
    try:
        for episode in range(episodes):
            initial = environment.reset()
            maximum_replay_error = max(
                maximum_replay_error,
                float(np.max(np.abs(initial - arrays["observations"][episode, 0]))),
            )
            for index in range(1, transitions):
                action = np.asarray(arrays["actions"][episode, index], np.float32)
                snapshot = environment.snapshot()
                if episode in train_ids and arrays["continuations"][episode, index - 1] > 0.0:
                    direction = random.choice((-1.0, 1.0), size=action_dim).astype(np.float32)
                    direction /= np.sqrt(float(action_dim))
                    lower = np.clip(action - ACTION_DELTA * direction, -1.0, 1.0)
                    upper = np.clip(action + ACTION_DELTA * direction, -1.0, 1.0)
                    environment.restore(snapshot)
                    lower_step = environment.step(lower)
                    environment.restore(snapshot)
                    upper_step = environment.step(upper)
                    lower_actions[episode, index] = lower
                    upper_actions[episode, index] = upper
                    lower_observations[episode, index] = lower_step.observation
                    upper_observations[episode, index] = upper_step.observation
                    lower_rewards[episode, index] = lower_step.reward
                    upper_rewards[episode, index] = upper_step.reward
                    causal_mask[episode, index] = 1.0
                environment.restore(snapshot)
                actual = environment.step(action)
                maximum_replay_error = max(
                    maximum_replay_error,
                    float(
                        np.max(
                            np.abs(
                                actual.observation
                                - arrays["observations"][episode, index]
                            )
                        )
                    ),
                    abs(float(actual.reward) - float(arrays["rewards"][episode, index])),
                    abs(
                        float(actual.continuation)
                        - float(arrays["continuations"][episode, index])
                    ),
                )
                if actual.is_last:
                    break
    finally:
        environment.close()
    if maximum_replay_error > 1e-6:
        raise ValueError(
            f"causal probe replay differs from pilot data by {maximum_replay_error}"
        )
    probes = {
        "causal_actions_lower": lower_actions,
        "causal_actions_upper": upper_actions,
        "causal_observations_lower": lower_observations,
        "causal_observations_upper": upper_observations,
        "causal_rewards_lower": lower_rewards,
        "causal_rewards_upper": upper_rewards,
        "causal_mask": causal_mask,
    }
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pilot_dataset_directory / "dataset.npz", copied_dataset_path)
    benchmark._write_npz_atomic(probe_path, probes)
    result = {
        "schema_version": SCHEMA,
        "stage": "causal_probe_dataset",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "dataset_sha256": row["dataset_sha256"],
        "dataset_file_sha256": benchmark.file_sha256(copied_dataset_path),
        "probe_sha256": benchmark.array_sha256(probes),
        "probe_file_sha256": benchmark.file_sha256(probe_path),
        "paired_transitions": int(np.sum(causal_mask)),
        "extra_simulator_steps": int(2 * np.sum(causal_mask)),
        "maximum_base_replay_error": maximum_replay_error,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json_atomic(result_path, result)
    return result


def _trajectory_config(
    protocol: Mapping[str, Any],
    selection: Mapping[str, Any],
    observation_shape: Sequence[int],
    action_dim: int,
) -> Any:
    from imf_dreamer_jax import DreamerConfig

    common = dict(
        protocol["canonical_executable_config"]["dreamer_config_non_task_shape_common"]
    )
    common["overshooting_distances"] = tuple(common["overshooting_distances"])
    common.update(
        observation_shape=tuple(int(value) for value in observation_shape),
        action_dim=int(action_dim),
        prior="imf",
        shortcut_training_k_max=None,
        shortcut_sampling_steps=4,
        imf_sampling_steps=1,
        imf_trajectory_enabled=True,
        imf_boundary_fraction=float(
            selection["selected"]["trajectory_imf"]["overrides"][
                "imf_boundary_fraction"
            ]
        ),
        prior_scale=float(
            selection["selected"]["trajectory_imf"]["overrides"][
                "objective_loss_scale"
            ]
        ),
        imf_causal_consistency_scale=CAUSAL_SCALE,
        imf_causal_reward_scale=CAUSAL_REWARD_SCALE,
        imf_causal_huber_delta=CAUSAL_HUBER_DELTA,
        imf_causal_normalization_epsilon=CAUSAL_NORMALIZATION_EPSILON,
    )
    return DreamerConfig(**common)


def _load_augmented_dataset(output_root: Path, seed: int) -> dict[str, np.ndarray]:
    directory = output_root / "dataset" / f"seed-{seed}"
    result = read_json(directory / "result.json")
    arrays = benchmark.load_npz(directory / "dataset.npz")
    probes = benchmark.load_npz(directory / "causal_probes.npz")
    if result["probe_sha256"] != benchmark.array_sha256(probes):
        raise ValueError("causal probe payload digest mismatch")
    overlap = set(arrays) & set(probes)
    if overlap:
        raise ValueError(f"causal probe fields collide with pilot dataset: {overlap}")
    return {**arrays, **probes}


def train_seed(
    pilot_root: str | Path,
    output_root: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Train one causal-iMF world model and its single nested actor seed."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        create_agent,
        diverse_imagination_starts,
        jit_observe_sequence,
        jit_train_actor_critic,
        jit_train_world_model,
        load_checkpoint,
        save_checkpoint,
    )

    pilot = Path(pilot_root)
    output = Path(output_root)
    manifest = write_manifest(pilot, output)
    row = _scheduled_row(manifest, seed)
    protocol, _, selection = _pilot_artifacts(pilot)
    arrays = _load_augmented_dataset(output, seed)
    config = _trajectory_config(
        protocol,
        selection,
        arrays["observations"].shape[2:],
        int(arrays["actions"].shape[-1]),
    )
    world_updates = int(row["world_model_updates"])
    actor_updates = int(row["actor_updates"])
    batch_size = 32
    sequence_length = 32
    world_directory = output / "world_model" / f"seed-{seed}"
    world_directory.mkdir(parents=True, exist_ok=True)
    world_checkpoint = world_directory / "checkpoint.pkl"
    world_result_path = world_directory / "result.json"
    schedule = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=seed,
        updates=world_updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    benchmark._write_npz_atomic(world_directory / "batch_schedule.npz", schedule)
    objective_key = benchmark.derive_jax_key("world-objective", TASK, seed)
    if world_result_path.is_file():
        world_result = read_json(world_result_path)
        world_state, stored_config, metadata = load_checkpoint(world_checkpoint)
        if stored_config != config or metadata.get("completed_updates") != world_updates:
            raise ValueError("existing causal world-model checkpoint is incompatible")
    else:
        if world_checkpoint.is_file():
            world_state, stored_config, metadata = load_checkpoint(world_checkpoint)
            if stored_config != config:
                raise ValueError("partial causal world-model config mismatch")
            start_update = int(metadata.get("completed_updates", -1))
            accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        else:
            world_state = create_agent(
                config, benchmark.derive_jax_key("world-init", TASK, seed)
            )
            start_update = 0
            accumulated_wall = 0.0
        latest = None
        started = time.perf_counter()
        for update in range(start_update, world_updates):
            batch = benchmark._materialize_batch(
                arrays,
                schedule,
                update,
                sequence_length=sequence_length,
                burn_in=config.burn_in,
            )
            world_state, metrics = jit_train_world_model(
                world_state, batch, jax.random.fold_in(objective_key, update), config
            )
            latest = benchmark._metrics_dict(metrics)
            completed = update + 1
            if completed % 1000 == 0 or completed == world_updates:
                save_checkpoint(
                    world_checkpoint,
                    world_state,
                    config,
                    metadata={
                        "stage": "causal_world_model",
                        "cell_id": f"causal-world-{seed}",
                        "completed_updates": completed,
                        "last_metrics": latest,
                        "wall_seconds": accumulated_wall
                        + time.perf_counter()
                        - started,
                    },
                )
        if latest is None:
            raise RuntimeError("causal world-model cell performed no updates")
        world_result = {
            "schema_version": SCHEMA,
            "stage": "world_model",
            "status": "complete",
            "source_commit": manifest["source_commit"],
            "manifest_sha256": manifest["manifest_sha256"],
            "world_model_seed": seed,
            "updates": world_updates,
            "runtime_config": asdict(config),
            "runtime_config_sha256": benchmark.object_sha256(asdict(config)),
            "final_metrics": latest,
            "checkpoint_sha256": benchmark.file_sha256(world_checkpoint),
            "batch_schedule_sha256": benchmark.array_sha256(schedule),
            "wall_seconds": accumulated_wall + time.perf_counter() - started,
            "runtime": benchmark.runtime_fingerprint(),
        }
        write_json_atomic(world_result_path, world_result)

    rollout_directory = output / "rollout" / f"seed-{seed}"
    rollout_directory.mkdir(parents=True, exist_ok=True)
    rollout_result_path = rollout_directory / "result.json"
    if rollout_result_path.is_file():
        rollout_result = read_json(rollout_result_path)
    else:
        windows = benchmark._rollout_windows(
            arrays,
            protocol,
            "pilot",
            task=TASK,
            world_model_seed=seed,
            stochastic_dim=config.stochastic_dim,
        )
        start = benchmark.posterior_mean_filter(
            world_state.params.world_model,
            jnp.asarray(windows["context_observations"]),
            jnp.asarray(windows["context_actions"]),
            config,
        )
        sampler = jax.jit(
            benchmark.open_loop_samples_with_continuation,
            static_argnames=("config",),
        )
        observation_samples, reward_samples, continuation_samples = sampler(
            world_state.params.world_model,
            start,
            jnp.asarray(windows["future_actions"]),
            jnp.asarray(windows["noise"]),
            config,
        )
        metrics = benchmark.normalized_rollout_statistics(
            np.asarray(observation_samples),
            np.asarray(reward_samples),
            windows,
            protocol["evaluation"]["rollout_horizons"],
            np.asarray(continuation_samples),
        )
        raw_path = benchmark._write_npz_atomic(
            rollout_directory / "predictive_draws.npz",
            {
                "observation_samples": np.asarray(observation_samples),
                "reward_samples": np.asarray(reward_samples),
                "continuation_samples": np.asarray(continuation_samples),
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
            },
        )
        rollout_result = {
            "schema_version": SCHEMA,
            "stage": "rollout",
            "status": "complete",
            "source_commit": manifest["source_commit"],
            "manifest_sha256": manifest["manifest_sha256"],
            "world_model_seed": seed,
            "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
            "nfe": 1,
            **metrics,
            "raw_sha256": benchmark.file_sha256(raw_path),
        }
        write_json_atomic(rollout_result_path, rollout_result)

    actor_directory = output / "actor" / f"seed-{seed}-actor-{ACTOR_SEED}"
    actor_directory.mkdir(parents=True, exist_ok=True)
    actor_checkpoint = actor_directory / "checkpoint.pkl"
    actor_result_path = actor_directory / "result.json"
    if actor_result_path.is_file():
        actor_result = read_json(actor_result_path)
    else:
        fresh = create_agent(
            config,
            benchmark.derive_jax_key("actor-init", TASK, seed, ACTOR_SEED),
        )
        actor_state = AgentState(
            AgentParams(
                world_state.params.world_model,
                fresh.params.actor,
                fresh.params.critic,
            ),
            world_state.model_optimizer,
            fresh.actor_optimizer,
            fresh.critic_optimizer,
            fresh.slow_critic,
            None,
        )
        actor_schedule_seed = benchmark.derive_seed(
            "actor-batches", seed, ACTOR_SEED
        )
        actor_schedule = benchmark._batch_schedule(
            arrays,
            task=TASK,
            world_model_seed=actor_schedule_seed,
            updates=actor_updates,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        benchmark._write_npz_atomic(
            actor_directory / "batch_schedule.npz", actor_schedule
        )
        posterior_key = benchmark.derive_jax_key(
            "actor-posterior", TASK, seed, ACTOR_SEED
        )
        start_key = benchmark.derive_jax_key(
            "actor-start", TASK, seed, ACTOR_SEED
        )
        actor_objective_key = benchmark.derive_jax_key(
            "actor-objective", TASK, seed, ACTOR_SEED
        )
        latest = None
        started = time.perf_counter()
        for update in range(actor_updates):
            batch = benchmark._materialize_batch(
                arrays,
                actor_schedule,
                update,
                sequence_length=sequence_length,
                burn_in=config.burn_in,
            )
            sequence = jit_observe_sequence(
                actor_state.params.world_model,
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
            actor_state, metrics = jit_train_actor_critic(
                actor_state,
                starts,
                jax.random.fold_in(actor_objective_key, update),
                config,
            )
            latest = benchmark._metrics_dict(metrics)
        if latest is None:
            raise RuntimeError("causal actor cell performed no updates")
        returns, traces = benchmark._evaluate_actor_policy(
            actor_state,
            config,
            task=TASK,
            world_model_seed=seed,
            actor_seed=ACTOR_SEED,
            episodes=int(protocol["profiles"]["pilot"]["real_environment_evaluation_episodes"]),
            maximum_steps=int(protocol["data"]["native_episode_limit"]),
        )
        action_path = benchmark._write_npz_atomic(
            actor_directory / "action_traces.npz", traces
        )
        save_checkpoint(
            actor_checkpoint,
            actor_state,
            config,
            metadata={
                "stage": "causal_actor",
                "cell_id": f"seed-{seed}-actor-{ACTOR_SEED}",
                "completed_updates": actor_updates,
                "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
                "last_metrics": latest,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        actor_result = {
            "schema_version": SCHEMA,
            "stage": "actor",
            "status": "complete",
            "source_commit": manifest["source_commit"],
            "manifest_sha256": manifest["manifest_sha256"],
            "world_model_seed": seed,
            "actor_seed": ACTOR_SEED,
            "updates": actor_updates,
            "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
            "world_model_frozen": True,
            "final_metrics": latest,
            "episode_returns": [float(value) for value in returns],
            "normalized_episode_returns": [float(value) / 1000.0 for value in returns],
            "normalized_episode_return_mean": float(np.mean(returns) / 1000.0),
            "checkpoint_sha256": benchmark.file_sha256(actor_checkpoint),
            "raw_action_traces_sha256": benchmark.file_sha256(action_path),
            "wall_seconds": time.perf_counter() - started,
            "runtime": benchmark.runtime_fingerprint(),
        }
        write_json_atomic(actor_result_path, actor_result)
    result = {
        "schema_version": SCHEMA,
        "stage": "seed_complete",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "world_model_seed": seed,
        "world_model_result_sha256": benchmark.file_sha256(world_result_path),
        "rollout_result_sha256": benchmark.file_sha256(rollout_result_path),
        "actor_result_sha256": benchmark.file_sha256(actor_result_path),
        "rollout_auc": rollout_result["normalized_free_running_rollout_error_auc"],
        "actor_normalized_return_mean": actor_result["normalized_episode_return_mean"],
    }
    write_json_atomic(output / "seeds" / f"seed-{seed}.json", result)
    return result


def finalize(
    pilot_root: str | Path,
    output_root: str | Path,
    baseline_diagnostics: str | Path,
) -> dict[str, Any]:
    """Compare new results with immutable shortcut artifacts and write a report."""

    pilot = Path(pilot_root)
    output = Path(output_root)
    manifest = write_manifest(pilot, output)
    protocol, matrix, selection = _pilot_artifacts(pilot)
    shortcut_id = selection["selected"]["shortcut_forcing"]["candidate_id"]
    rows = []
    diagnostic_rows = []
    for seed in WORLD_MODEL_SEEDS:
        new_rollout = read_json(output / "rollout" / f"seed-{seed}" / "result.json")
        new_actor = read_json(
            output / "actor" / f"seed-{seed}-actor-{ACTOR_SEED}" / "result.json"
        )
        shortcut_rollout_cell = _one_cell(
            matrix,
            "rollout",
            task=TASK,
            world_model_seed=seed,
            arm="shortcut_forcing",
            candidate_id=shortcut_id,
            budget_track="equal_updates",
            nfe=int(protocol["evaluation"]["primary_nfe"]["shortcut_forcing"]),
        )
        shortcut_actor_cell = _one_cell(
            matrix,
            "actor",
            task=TASK,
            world_model_seed=seed,
            actor_seed=ACTOR_SEED,
            arm="shortcut_forcing",
            candidate_id=shortcut_id,
            budget_track="equal_updates",
        )
        old_rollout = read_json(
            pilot / "rollout" / shortcut_rollout_cell["cell_id"] / "result.json"
        )
        old_actor = read_json(
            pilot / "actor" / shortcut_actor_cell["cell_id"] / "result.json"
        )
        rows.append(
            {
                "world_model_seed": seed,
                "causal_imf_rollout_auc": float(
                    new_rollout["normalized_free_running_rollout_error_auc"]
                ),
                "shortcut_rollout_auc": float(
                    old_rollout["normalized_free_running_rollout_error_auc"]
                ),
                "favorable_rollout_difference": float(
                    old_rollout["normalized_free_running_rollout_error_auc"]
                    - new_rollout["normalized_free_running_rollout_error_auc"]
                ),
                "causal_imf_actor_return": float(
                    new_actor["normalized_episode_return_mean"]
                ),
                "shortcut_actor_return": float(
                    old_actor["normalized_episode_return_mean"]
                ),
                "favorable_actor_difference": float(
                    new_actor["normalized_episode_return_mean"]
                    - old_actor["normalized_episode_return_mean"]
                ),
            }
        )
        diagnostic_rows.append(
            read_json(output / "diagnostics" / f"seed-{seed}.json")["row"]
        )
    baseline_report = read_json(baseline_diagnostics)
    baseline_metrics = {}
    for metric in ("actor_visited_model_error", "action_ranking", "gradient_fidelity"):
        matches = [
            value
            for value in baseline_report["metrics"][metric]
            if value["task"] == TASK and value["arm"] == "shortcut_forcing"
        ]
        if len(matches) != 1:
            raise ValueError(f"baseline diagnostic report lacks one Reacher {metric} row")
        baseline_metrics[metric] = matches[0]
    new_metrics = {
        metric: aggregate_metric_rows(diagnostic_rows, metric)[0]
        for metric in ("actor_visited_model_error", "action_ranking", "gradient_fidelity")
    }
    actor_differences = np.asarray(
        [row["favorable_actor_difference"] for row in rows], dtype=np.float64
    )
    rollout_differences = np.asarray(
        [row["favorable_rollout_difference"] for row in rows], dtype=np.float64
    )
    comparison = {
        "schema_version": SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "evidence_class": "exploratory_intervention_diagnostic",
        "claim_eligible": False,
        "rows": rows,
        "summary": {
            "causal_imf_actor_return_mean": float(
                np.mean([row["causal_imf_actor_return"] for row in rows])
            ),
            "shortcut_actor_return_mean": float(
                np.mean([row["shortcut_actor_return"] for row in rows])
            ),
            "mean_favorable_actor_difference": float(np.mean(actor_differences)),
            "actor_favorable_seeds": int(np.sum(actor_differences > 0.0)),
            "causal_imf_rollout_auc_mean": float(
                np.mean([row["causal_imf_rollout_auc"] for row in rows])
            ),
            "shortcut_rollout_auc_mean": float(
                np.mean([row["shortcut_rollout_auc"] for row in rows])
            ),
            "mean_favorable_rollout_difference": float(np.mean(rollout_differences)),
            "rollout_favorable_seeds": int(np.sum(rollout_differences > 0.0)),
        },
        "causal_imf_policy_diagnostics": new_metrics,
        "existing_shortcut_policy_diagnostics": baseline_metrics,
        "limitations": [
            "two Reacher world-model seeds and one nested actor seed only",
            "new iMF receives paired simulator counterfactual labels unavailable to the immutable shortcut baseline",
            "policy diagnostics are descriptive and sampled on each actor's own visited states",
            "no pendulum world model is retrained in this Reacher-first study",
        ],
    }
    comparison["comparison_sha256"] = benchmark.object_sha256(comparison)
    write_json_atomic(output / "comparison.json", comparison)
    summary = comparison["summary"]
    report = "\n".join(
        [
            "# Small Reacher causal trajectory-iMF study",
            "",
            "This is an exploratory intervention diagnostic, not claim-bearing evidence.",
            "The immutable shortcut baseline is read only and is never retrained.",
            "",
            "| Metric | Causal trajectory-iMF | Existing shortcut | Favorable difference |",
            "|---|---:|---:|---:|",
            (
                "| Actor normalized return | "
                f"{summary['causal_imf_actor_return_mean']:.6g} | "
                f"{summary['shortcut_actor_return_mean']:.6g} | "
                f"{summary['mean_favorable_actor_difference']:.6g} |"
            ),
            (
                "| Rollout error AUC | "
                f"{summary['causal_imf_rollout_auc_mean']:.6g} | "
                f"{summary['shortcut_rollout_auc_mean']:.6g} | "
                f"{summary['mean_favorable_rollout_difference']:.6g} |"
            ),
            "",
            (
                f"Actor favored causal iMF on {summary['actor_favorable_seeds']}/"
                f"{len(WORLD_MODEL_SEEDS)} seeds; rollout AUC favored it on "
                f"{summary['rollout_favorable_seeds']}/{len(WORLD_MODEL_SEEDS)} seeds."
            ),
            "",
            "The new arm uses additional paired counterfactual simulator labels, so this result",
            "tests whether causal supervision repairs policy consistency; it cannot establish a",
            "matched-data objective-superiority claim against shortcut forcing.",
            "",
        ]
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return comparison


__all__ = [
    "ACTION_DELTA",
    "ACTOR_SEED",
    "CAUSAL_SCALE",
    "SCHEMA",
    "TASK",
    "WORLD_MODEL_SEEDS",
    "build_manifest",
    "finalize",
    "prepare_seed",
    "train_seed",
    "write_manifest",
]
