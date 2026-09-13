"""Matched Reacher study for a transition-conditioned reward residual."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from .artifacts import read_json, write_json_atomic
from . import matched_objective_benchmark as benchmark
from . import reward_head_mtp_study as mtp
from .advantage_repair_study import materialize_advantage_batch
from .policy_consistency_vnext import ACTOR_SEED, HORIZONS, TASK, _tree_delta
from .shared_probe_bank import evaluate_shared_probe_bank


SCHEMA = "trajectory-imf-action-reward-residual-study-v1"
WORLD_MODEL_SEEDS = mtp.WORLD_MODEL_SEEDS


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _source_row(manifest: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    rows = [
        row for row in manifest["source_artifacts"]
        if int(row["world_model_seed"]) == int(seed)
    ]
    if len(rows) != 1:
        raise ValueError("source seed is absent or duplicated")
    return rows[0]


def build_manifest(
    mtp_root: str | Path,
    output_root: str | Path | None = None,
    *,
    residual_updates: int = 1000,
    actor_updates: int = 3000,
    preparation_updates: int = 500,
    evaluation_episodes: int = 5,
) -> dict[str, Any]:
    del output_root
    if min(residual_updates, actor_updates, preparation_updates, evaluation_episodes) <= 0:
        raise ValueError("update and evaluation counts must be positive")
    base = Path(mtp_root).resolve(strict=True)
    base_manifest = read_json(base / "manifest.json")
    base_report = read_json(base / "report.json")
    if base_report.get("status") != "complete":
        raise ValueError("source reward-head MTP study is incomplete")
    if base_manifest.get("manifest_sha256") != base_report.get("manifest_sha256"):
        raise ValueError("source reward-head MTP manifest/report mismatch")
    if residual_updates > int(base_manifest["reward_updates"]):
        raise ValueError("residual updates exceed the authenticated source schedule")
    sources = []
    for seed in WORLD_MODEL_SEEDS:
        source = mtp._source_row(base_manifest, seed)
        tag = f"seed-{seed}-dreamer4_twohot_mtp"
        reward_directory = base / "reward" / tag
        reward_result = read_json(reward_directory / "result.json")
        checkpoint = reward_directory / "checkpoint.pkl"
        schedule = reward_directory / "schedules.npz"
        checks = (
            (checkpoint, reward_result["checkpoint_sha256"], "MTP checkpoint"),
            (schedule, reward_result["schedule_file_sha256"], "MTP schedule"),
            (Path(source["test_probe_bank"]), source["test_probe_bank_file_sha256"], "test probe"),
            (Path(source["train_probe_bank"]), source["train_probe_bank_file_sha256"], "train probe"),
        )
        for path, digest, label in checks:
            if benchmark.file_sha256(path) != digest:
                raise ValueError(f"{label} digest mismatch for seed {seed}")
        if benchmark.array_sha256(benchmark.load_npz(source["dataset"])) != source["dataset_sha256"]:
            raise ValueError("dataset payload digest mismatch")
        sources.append(
            {
                "world_model_seed": seed,
                "dataset": source["dataset"],
                "dataset_sha256": source["dataset_sha256"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": reward_result["checkpoint_sha256"],
                "source_schedule": str(schedule.resolve()),
                "source_schedule_sha256": reward_result["schedule_file_sha256"],
                "test_probe_bank": source["test_probe_bank"],
                "test_probe_bank_sha256": source["test_probe_bank_sha256"],
                "test_probe_bank_file_sha256": source["test_probe_bank_file_sha256"],
                "train_probe_bank": source["train_probe_bank"],
                "train_probe_bank_sha256": source["train_probe_bank_sha256"],
                "train_probe_bank_file_sha256": source["train_probe_bank_file_sha256"],
            }
        )
    from imf_dreamer_jax import ActionRewardResidualConfig

    objective = ActionRewardResidualConfig()
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "task": TASK,
        "world_model_seeds": list(WORLD_MODEL_SEEDS),
        "imagination_horizons": list(HORIZONS),
        "residual_updates": int(residual_updates),
        "actor_updates": int(actor_updates),
        "preparation_updates": int(preparation_updates),
        "evaluation_episodes": int(evaluation_episodes),
        "trainable_world_subtrees": ["reward_action_residual"],
        "objective": asdict(objective),
        "mtp_root": str(base),
        "mtp_manifest_sha256": base_manifest["manifest_sha256"],
        "mtp_report_file_sha256": benchmark.file_sha256(base / "report.json"),
        "source_artifacts": sources,
        "base_mtp_reference": {
            "reward_groups": base_report["reward_groups"]["dreamer4_twohot_mtp"],
            "probe_groups": base_report["probe_groups"]["dreamer4_twohot_mtp"],
            "actor_groups": [
                row for row in base_report["actor_groups"]
                if row["arm"] == "dreamer4_twohot_mtp"
            ],
        },
        "shortcut_reference": base_report["frozen_shortcut_reference"],
        "interpretation": {
            "claim_eligible": False,
            "evidence_class": "exploratory_two_seed_reacher_action_residual",
            "trajectory_model_frozen": True,
            "base_reward_head_frozen": True,
            "matched_actor_protocol": True,
        },
    }
    manifest = _json_native(manifest)
    return {**manifest, "manifest_sha256": benchmark.object_sha256(manifest)}


def write_manifest(
    mtp_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    expected = _json_native(build_manifest(mtp_root, output_root, **settings))
    output = Path(output_root)
    path = output / "manifest.json"
    if path.is_file():
        existing = read_json(path)
        if existing != expected:
            raise ValueError("existing action-residual manifest differs from frozen design")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, expected)
    return expected


def record_preflight(
    mtp_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = write_manifest(mtp_root, output_root, **settings)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("preflight must run inside Slurm")
    result = {
        "schema_version": SCHEMA,
        "stage": "preflight",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "slurm_job_id": job_id,
        "full_library_suite_passed": True,
        "full_comparison_suite_passed": True,
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(Path(output_root) / "preflight.json", result)
    return result


def _residual_metrics(details: Any, rms: Any) -> dict[str, float]:
    metrics = {
        "total": float(np.asarray(details.total)),
        "centered_return": float(np.asarray(details.centered_return)),
        "output_l2": float(np.asarray(details.output_l2)),
        "advantage_running_rms": float(
            np.sqrt(float(np.asarray(rms.mean_square)) + 1e-6)
        ),
        "informative_pair_fraction": float(
            np.asarray(details.advantage.informative_pair_fraction)
        ),
    }
    if not np.isfinite(np.asarray(list(metrics.values()))).all():
        raise FloatingPointError("residual training produced non-finite metrics")
    return metrics


def train_residual_cell(
    mtp_root: str | Path,
    output_root: str | Path,
    seed: int,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import (
        ActionRewardResidualConfig,
        attach_action_reward_residual,
        init_running_rms,
        jit_train_action_reward_residual,
        load_checkpoint,
        save_checkpoint,
    )

    manifest = write_manifest(mtp_root, output_root, **settings)
    if seed not in WORLD_MODEL_SEEDS:
        raise ValueError("residual cell is outside the frozen design")
    source = _source_row(manifest, seed)
    directory = Path(output_root) / "residual" / f"seed-{seed}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    source_checkpoint = Path(source["checkpoint"])
    state, config, _ = load_checkpoint(source_checkpoint)
    if (
        config.prior != "imf"
        or not config.imf_trajectory_enabled
        or config.reward_loss != "symexp_twohot"
        or config.reward_prediction_horizon != 8
    ):
        raise ValueError("source is not the required trajectory-iMF MTP checkpoint")
    state = attach_action_reward_residual(
        state,
        config,
        benchmark.derive_jax_key("action-reward-residual-init", TASK, seed),
    )
    initial = state
    arrays = benchmark.load_npz(source["dataset"])
    train_bank = benchmark.load_npz(source["train_probe_bank"])
    if benchmark.array_sha256(train_bank) != source["train_probe_bank_sha256"]:
        raise ValueError("training probe bank payload digest mismatch")
    source_schedule = benchmark.load_npz(source["source_schedule"])
    updates = int(manifest["residual_updates"])
    probe_indices = np.asarray(source_schedule["probe_indices"][:updates], np.int32)
    if probe_indices.shape != (updates, 32):
        raise ValueError("source counterfactual schedule has the wrong shape")
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = benchmark._write_npz_atomic(
        directory / "schedule.npz", {"probe_indices": probe_indices}
    )
    objective = ActionRewardResidualConfig(**manifest["objective"])
    rms = init_running_rms()
    key = benchmark.derive_jax_key("action-reward-residual-objective", TASK, seed)
    started = time.perf_counter()
    latest = None
    for update in range(updates):
        decision_batch = materialize_advantage_batch(
            arrays,
            train_bank,
            probe_indices[update],
            burn_in=config.burn_in,
        )
        state, details, rms = jit_train_action_reward_residual(
            state,
            decision_batch,
            jax.random.fold_in(key, update),
            config,
            objective,
            rms,
        )
        latest = _residual_metrics(details, rms)
    if latest is None:
        raise RuntimeError("residual cell performed no updates")
    source_names = tuple(
        name for name in state.params.world_model if name != "reward_action_residual"
    )
    parameter_deltas = {
        name: _tree_delta(initial.params.world_model[name], state.params.world_model[name])
        for name in source_names
    }
    first_deltas = {
        name: _tree_delta(
            initial.model_optimizer.first_moment[name],
            state.model_optimizer.first_moment[name],
        )
        for name in source_names
    }
    second_deltas = {
        name: _tree_delta(
            initial.model_optimizer.second_moment[name],
            state.model_optimizer.second_moment[name],
        )
        for name in source_names
    }
    residual_delta = _tree_delta(
        initial.params.world_model["reward_action_residual"],
        state.params.world_model["reward_action_residual"],
    )
    if any(parameter_deltas.values()) or any(first_deltas.values()) or any(second_deltas.values()):
        raise RuntimeError("action-residual training mutated a frozen source subtree")
    if residual_delta <= 0.0:
        raise RuntimeError("action-residual parameters did not update")
    checkpoint = directory / "checkpoint.pkl"
    save_checkpoint(
        checkpoint,
        state,
        config,
        metadata={
            "stage": "action_reward_residual",
            "completed_updates": updates,
            "source_checkpoint_sha256": source["checkpoint_sha256"],
        },
    )
    if benchmark.file_sha256(source_checkpoint) != source["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint changed during residual adaptation")
    residual_parameters = int(sum(
        value.size
        for value in jax.tree_util.tree_leaves(
            state.params.world_model["reward_action_residual"]
        )
    ))
    result = {
        "schema_version": SCHEMA,
        "stage": "residual",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "checkpoint_sha256": benchmark.file_sha256(checkpoint),
        "source_parameter_deltas": parameter_deltas,
        "source_first_moment_deltas": first_deltas,
        "source_second_moment_deltas": second_deltas,
        "residual_parameter_delta": residual_delta,
        "residual_parameters": residual_parameters,
        "residual_updates": updates,
        "objective": asdict(objective),
        "schedule_file_sha256": benchmark.file_sha256(schedule_path),
        "final_metrics": latest,
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def evaluate_residual_cell(
    mtp_root: str | Path,
    output_root: str | Path,
    seed: int,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import load_checkpoint, reward_context_predictions

    manifest = write_manifest(mtp_root, output_root, **settings)
    source = _source_row(manifest, seed)
    residual_directory = Path(output_root) / "residual" / f"seed-{seed}"
    residual_result = read_json(residual_directory / "result.json")
    checkpoint = residual_directory / "checkpoint.pkl"
    if benchmark.file_sha256(checkpoint) != residual_result["checkpoint_sha256"]:
        raise ValueError("residual checkpoint digest mismatch")
    directory = Path(output_root) / "evaluation" / f"seed-{seed}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    state, config, _ = load_checkpoint(checkpoint)
    bank = benchmark.load_npz(source["test_probe_bank"])
    if benchmark.array_sha256(bank) != source["test_probe_bank_sha256"]:
        raise ValueError("held-out probe payload digest mismatch")
    arrays = benchmark.load_npz(source["dataset"])
    probe = evaluate_shared_probe_bank(
        state.params.world_model, config, arrays, bank
    )
    schedule = mtp._heldout_schedule(
        arrays, seed, batch_size=64, sequence_length=32
    )
    batch = benchmark._materialize_batch(
        arrays, schedule, 0, sequence_length=32, burn_in=config.burn_in
    )
    predictions = reward_context_predictions(
        state.params.world_model,
        batch,
        benchmark.derive_jax_key("reward-head-mtp-heldout-context", seed),
        config,
        mtp.objective_for_arm("dreamer4_twohot_mtp"),
    )
    host = jax.device_get(predictions)
    context_metrics = {}
    for name in ("posterior", "corrupted", "generated"):
        values = np.asarray(getattr(host, name))
        overall, by_offset = mtp._masked_mse_by_offset(
            values, host.targets, host.mask
        )
        context_metrics[name] = {
            "mse": overall,
            "mse_by_offset": by_offset,
            "offset_zero_mean_calibration_error": mtp._mean_calibration_error(
                values[..., 0],
                np.asarray(host.targets)[..., 0],
                np.asarray(host.mask)[..., 0],
            ),
        }
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = benchmark._write_npz_atomic(
        directory / "evaluation_arrays.npz",
        {
            "probe_predicted_returns": probe.pop("predicted_returns"),
            "probe_predicted_return_draws": probe.pop("predicted_return_draws"),
            "simulator_returns": bank["simulator_returns"],
            "horizon_mask": bank["horizon_mask"],
            "posterior_rewards": np.asarray(host.posterior),
            "corrupted_rewards": np.asarray(host.corrupted),
            "generated_rewards": np.asarray(host.generated),
            "reward_targets": np.asarray(host.targets),
            "reward_mask": np.asarray(host.mask),
        },
    )
    result = {
        "schema_version": SCHEMA,
        "stage": "evaluation",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "world_model_checkpoint_sha256": residual_result["checkpoint_sha256"],
        "metrics_by_horizon": probe["metrics_by_horizon"],
        "reward_context_metrics": context_metrics,
        "raw_sha256": benchmark.file_sha256(raw_path),
    }
    write_json_atomic(result_path, result)
    return result


def train_actor_cell(
    mtp_root: str | Path,
    output_root: str | Path,
    seed: int,
    horizon: int,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        create_agent,
        diverse_imagination_starts,
        jit_observe_sequence,
        jit_train_actor_critic,
        jit_train_behavior_cloning,
        jit_train_replay_critic,
        load_checkpoint,
        save_checkpoint,
        snapshot_behavior_prior,
    )

    manifest = write_manifest(mtp_root, output_root, **settings)
    if seed not in WORLD_MODEL_SEEDS or horizon not in HORIZONS:
        raise ValueError("actor cell is outside the frozen design")
    source = _source_row(manifest, seed)
    residual_directory = Path(output_root) / "residual" / f"seed-{seed}"
    residual_result = read_json(residual_directory / "result.json")
    world_checkpoint = residual_directory / "checkpoint.pkl"
    if benchmark.file_sha256(world_checkpoint) != residual_result["checkpoint_sha256"]:
        raise ValueError("actor source residual checkpoint digest mismatch")
    directory = Path(output_root) / "actor" / f"seed-{seed}-h{horizon}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    world_state, stored_config, _ = load_checkpoint(world_checkpoint)
    config = replace(
        stored_config,
        actor_gradient="pmpo",
        behavior_kl_scale=0.3,
        critic_bins=51,
        critic_output_init_scale=0.0,
        imagination_horizon=horizon,
    )
    fresh = create_agent(
        config,
        benchmark.derive_jax_key(
            "reward-head-mtp-actor-init", TASK, seed, ACTOR_SEED, horizon
        ),
    )
    frozen_world = world_state.params.world_model
    state = AgentState(
        AgentParams(frozen_world, fresh.params.actor, fresh.params.critic),
        world_state.model_optimizer,
        fresh.actor_optimizer,
        fresh.critic_optimizer,
        fresh.slow_critic,
        world_state.world_model_teacher,
    )
    arrays = benchmark.load_npz(source["dataset"])
    batch_size, sequence_length = 32, 32
    actor_updates = int(manifest["actor_updates"])
    preparation_updates = int(manifest["preparation_updates"])
    schedule = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=benchmark.derive_seed(
            "reward-head-mtp-actor-batches", seed, ACTOR_SEED, horizon
        ),
        updates=preparation_updates + actor_updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    posterior_key = benchmark.derive_jax_key(
        "reward-head-mtp-actor-posterior", seed, ACTOR_SEED, horizon
    )
    started = time.perf_counter()
    latest_bc = math.nan
    latest_replay_critic = math.nan
    for update in range(preparation_updates):
        replay = benchmark._materialize_batch(
            arrays, schedule, update, sequence_length=sequence_length, burn_in=config.burn_in
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
        state, critic_loss = jit_train_replay_critic(
            state,
            sequence.states.feature,
            replay["rewards"],
            replay["continuations"],
            config,
            loss_mask=replay["loss_mask"],
        )
        latest_bc = float(bc_loss)
        latest_replay_critic = float(critic_loss)
    behavior_prior = snapshot_behavior_prior(state.params.actor)
    start_key = benchmark.derive_jax_key(
        "reward-head-mtp-actor-start", seed, ACTOR_SEED, horizon
    )
    objective_key = benchmark.derive_jax_key(
        "reward-head-mtp-actor-objective", seed, ACTOR_SEED, horizon
    )
    latest = None
    for update in range(actor_updates):
        schedule_index = preparation_updates + update
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
            sequence.states, config.burn_in, jax.random.fold_in(start_key, update)
        )
        state, metrics = jit_train_actor_critic(
            state,
            starts,
            jax.random.fold_in(objective_key, update),
            config,
            behavior_prior=behavior_prior,
            reward_ensemble_params=None,
            epistemic_penalty_scale=0.0,
        )
        latest = benchmark._metrics_dict(metrics)
    if latest is None:
        raise RuntimeError("actor cell performed no updates")
    world_delta = _tree_delta(frozen_world, state.params.world_model)
    if world_delta != 0.0:
        raise RuntimeError("actor mutated the frozen residual world model")
    returns, traces = benchmark._evaluate_actor_policy(
        state,
        config,
        task=TASK,
        world_model_seed=seed,
        actor_seed=ACTOR_SEED,
        episodes=int(manifest["evaluation_episodes"]),
        maximum_steps=1000,
    )
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = benchmark._write_npz_atomic(directory / "action_traces.npz", traces)
    checkpoint_path = directory / "checkpoint.pkl"
    save_checkpoint(
        checkpoint_path,
        state,
        config,
        metadata={
            "stage": "action_reward_residual_actor",
            "source_world_model_checkpoint_sha256": residual_result["checkpoint_sha256"],
            "completed_updates": actor_updates,
            "preparation_updates": preparation_updates,
            "behavior_prior_frozen": True,
        },
    )
    result = {
        "schema_version": SCHEMA,
        "stage": "actor",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "actor_seed": ACTOR_SEED,
        "imagination_horizon": horizon,
        "world_model_checkpoint_sha256": residual_result["checkpoint_sha256"],
        "world_model_frozen": True,
        "world_model_parameter_delta": world_delta,
        "behavior_prior_frozen": True,
        "preparation_updates": preparation_updates,
        "actor_updates": actor_updates,
        "final_behavior_cloning_loss": latest_bc,
        "final_replay_critic_loss": latest_replay_critic,
        "final_metrics": latest,
        "episode_returns": [float(value) for value in returns],
        "normalized_episode_returns": [float(value) / 1000.0 for value in returns],
        "normalized_episode_return_mean": float(np.mean(returns) / 1000.0),
        "checkpoint_sha256": benchmark.file_sha256(checkpoint_path),
        "raw_action_traces_sha256": benchmark.file_sha256(trace_path),
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def _group_reward(rows: list[Mapping[str, Any]], context: str) -> dict[str, Any]:
    return {
        "mean_mse": float(np.mean([
            row["reward_context_metrics"][context]["mse"] for row in rows
        ])),
        "mean_offset_zero_calibration_error": float(np.mean([
            row["reward_context_metrics"][context]["offset_zero_mean_calibration_error"]
            for row in rows
        ])),
        "per_seed_mse": {
            str(row["world_model_seed"]): row["reward_context_metrics"][context]["mse"]
            for row in rows
        },
    }


def finalize(
    mtp_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = write_manifest(mtp_root, output_root, **settings)
    output = Path(output_root)
    residual_rows = [read_json(path) for path in sorted(output.glob("residual/*/result.json"))]
    evaluation_rows = [read_json(path) for path in sorted(output.glob("evaluation/*/result.json"))]
    actor_rows = [read_json(path) for path in sorted(output.glob("actor/*/result.json"))]
    complete = (
        len(residual_rows) == len(WORLD_MODEL_SEEDS)
        and len(evaluation_rows) == len(WORLD_MODEL_SEEDS)
        and len(actor_rows) == len(WORLD_MODEL_SEEDS) * len(HORIZONS)
    )
    reward_groups = {
        context: _group_reward(evaluation_rows, context)
        for context in ("posterior", "corrupted", "generated")
    } if evaluation_rows else {}
    probe_groups = {
        horizon: {
            "pairwise_accuracy_mean": float(np.mean([
                row["metrics_by_horizon"][horizon]["pairwise_accuracy"]
                for row in evaluation_rows
            ])),
            "mean_state_spearman": float(np.mean([
                row["metrics_by_horizon"][horizon]["mean_state_spearman"]
                for row in evaluation_rows
            ])),
            "mean_simulator_regret": float(np.mean([
                row["metrics_by_horizon"][horizon]["mean_simulator_regret"]
                for row in evaluation_rows
            ])),
        }
        for horizon in ("1", "3", "5", "15")
    } if evaluation_rows else {}
    actor_groups = []
    for horizon in HORIZONS:
        rows = [row for row in actor_rows if row["imagination_horizon"] == horizon]
        if rows:
            actor_groups.append(
                {
                    "imagination_horizon": horizon,
                    "cells": len(rows),
                    "mean_normalized_return": float(np.mean([
                        row["normalized_episode_return_mean"] for row in rows
                    ])),
                    "per_seed": {
                        str(row["world_model_seed"]): row["normalized_episode_return_mean"]
                        for row in rows
                    },
                }
            )
    base = manifest["base_mtp_reference"]
    actor_deltas = {}
    for row in actor_groups:
        horizon = row["imagination_horizon"]
        base_actor = next(
            item for item in base["actor_groups"]
            if item["imagination_horizon"] == horizon
        )
        shortcut_actor = next(
            item for item in manifest["shortcut_reference"]["actor_groups"]
            if item["imagination_horizon"] == horizon
        )
        actor_deltas[str(horizon)] = {
            "versus_base_mtp": row["mean_normalized_return"] - base_actor["mean_normalized_return"],
            "versus_shortcut": row["mean_normalized_return"] - shortcut_actor["mean_normalized_return"],
        }
    probe_deltas = {
        horizon: {
            "pairwise_accuracy_versus_base_mtp": (
                probe_groups[horizon]["pairwise_accuracy_mean"]
                - base["probe_groups"][horizon]["pairwise_accuracy_mean"]
            ),
            "regret_versus_base_mtp": (
                probe_groups[horizon]["mean_simulator_regret"]
                - base["probe_groups"][horizon]["mean_simulator_regret"]
            ),
        }
        for horizon in probe_groups
    }
    reward_deltas = {
        context: {
            "mse_versus_base_mtp": (
                reward_groups[context]["mean_mse"]
                - base["reward_groups"][context]["mean_mse"]
            ),
            "calibration_versus_base_mtp": (
                reward_groups[context]["mean_offset_zero_calibration_error"]
                - base["reward_groups"][context]["mean_offset_zero_calibration_error"]
            ),
        }
        for context in reward_groups
    }
    report = {
        "schema_version": SCHEMA,
        "stage": "final",
        "status": "complete" if complete else "partial",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_residual_cells": len(residual_rows),
        "expected_residual_cells": len(WORLD_MODEL_SEEDS),
        "completed_evaluation_cells": len(evaluation_rows),
        "expected_evaluation_cells": len(WORLD_MODEL_SEEDS),
        "completed_actor_cells": len(actor_rows),
        "expected_actor_cells": len(WORLD_MODEL_SEEDS) * len(HORIZONS),
        "reward_groups": reward_groups,
        "probe_groups": probe_groups,
        "actor_groups": actor_groups,
        "matched_deltas": {
            "reward": reward_deltas,
            "probe": probe_deltas,
            "actor": actor_deltas,
        },
        "base_mtp_reference": base,
        "shortcut_reference": manifest["shortcut_reference"],
        "aggregate_cell_wall_seconds": float(sum(
            float(row.get("wall_seconds", 0.0))
            for row in residual_rows + actor_rows
        )),
        "interpretation": manifest["interpretation"],
    }
    write_json_atomic(output / "report.json", report)
    return report


__all__ = [
    "SCHEMA",
    "WORLD_MODEL_SEEDS",
    "build_manifest",
    "evaluate_residual_cell",
    "finalize",
    "record_preflight",
    "train_actor_cell",
    "train_residual_cell",
    "write_manifest",
]
