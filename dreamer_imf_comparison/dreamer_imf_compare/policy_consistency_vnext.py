"""Gated Reacher study for policy-consistency interventions.

The study reads frozen pilot and causal-iMF checkpoints, trains new actors only,
and evaluates all arms on one immutable probe bank.  It never writes into an
input result tree.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from .artifacts import read_json, write_json_atomic
from . import matched_objective_benchmark as benchmark
from .shared_probe_bank import (
    evaluate_shared_probe_bank,
    generate_shared_probe_bank,
    shared_probe_manifest,
    validate_shared_probe_bank,
)


SCHEMA = "trajectory-imf-policy-consistency-vnext-v1"
TASK = "dmc_reacher_easy"
WORLD_MODEL_SEEDS = (211, 223)
ACTOR_SEED = 311
ARMS = ("shortcut_forcing", "trajectory_imf", "causal_trajectory_imf")
HORIZONS = (5, 15)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _one_cell(matrix: Mapping[str, Any], stage: str, **fields: Any) -> Mapping[str, Any]:
    matches = [
        cell
        for cell in matrix["cells"]
        if cell["stage"] == stage
        and all(cell.get(name) == value for name, value in fields.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {stage} cell for {fields}, found {len(matches)}")
    return matches[0]


def _pilot(pilot_root: str | Path):
    root = Path(pilot_root)
    protocol = read_json(root / "frozen_protocol.json")
    matrix = read_json(root / "matrix.json")
    selection = read_json(root / "hpo_selection.json")
    if selection.get("status") != "complete":
        raise ValueError("pilot selection is incomplete")
    return root, protocol, matrix, selection


def _dataset_paths(pilot_root: str | Path, seed: int) -> tuple[Path, Mapping[str, Any]]:
    root, _, matrix, _ = _pilot(pilot_root)
    cell = _one_cell(matrix, "dataset", task=TASK, world_model_seed=seed)
    directory = root / "dataset" / cell["cell_id"]
    result = read_json(directory / "result.json")
    path = directory / "dataset.npz"
    if result["dataset_file_sha256"] != benchmark.file_sha256(path):
        raise ValueError("frozen pilot dataset digest mismatch")
    return path, result


def _world_checkpoint(
    pilot_root: str | Path,
    causal_root: str | Path,
    seed: int,
    arm: str,
) -> tuple[Path, Mapping[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "causal_trajectory_imf":
        directory = Path(causal_root) / "world_model" / f"seed-{seed}"
        result = read_json(directory / "result.json")
    else:
        root, _, matrix, selection = _pilot(pilot_root)
        candidate = selection["selected"][arm]["candidate_id"]
        cell = _one_cell(
            matrix,
            "world_model",
            task=TASK,
            world_model_seed=seed,
            arm=arm,
            candidate_id=candidate,
            budget_track="equal_updates",
        )
        directory = root / "world_model" / cell["cell_id"]
        result = read_json(directory / "result.json")
    checkpoint = directory / "checkpoint.pkl"
    if result["checkpoint_sha256"] != benchmark.file_sha256(checkpoint):
        raise ValueError("frozen world-model checkpoint digest mismatch")
    return checkpoint, result


def build_manifest(
    pilot_root: str | Path,
    causal_root: str | Path,
    *,
    actor_updates: int,
    preparation_updates: int,
    epistemic_scales: tuple[float, ...] = (0.0,),
) -> dict[str, Any]:
    if actor_updates <= 0 or preparation_updates <= 0:
        raise ValueError("study update counts must be positive")
    if not epistemic_scales or any(value < 0.0 for value in epistemic_scales):
        raise ValueError("epistemic scales must be nonempty and nonnegative")
    sources = []
    for seed in WORLD_MODEL_SEEDS:
        dataset_path, dataset_result = _dataset_paths(pilot_root, seed)
        worlds = {}
        for arm in ARMS:
            checkpoint, result = _world_checkpoint(pilot_root, causal_root, seed, arm)
            worlds[arm] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        sources.append(
            {
                "world_model_seed": seed,
                "dataset": str(dataset_path.resolve()),
                "dataset_sha256": dataset_result["dataset_sha256"],
                "world_models": worlds,
            }
        )
    manifest = {
        "schema_version": SCHEMA,
        "source_commit": _git_commit(),
        "status": "frozen_before_execution",
        "task": TASK,
        "world_model_seeds": list(WORLD_MODEL_SEEDS),
        "actor_seed": ACTOR_SEED,
        "arms": list(ARMS),
        "imagination_horizons": list(HORIZONS),
        "actor_updates": int(actor_updates),
        "preparation_updates": int(preparation_updates),
        "actor_repair": {
            "pmpo": "sign_only_separately_normalized_positive_and_negative_sets",
            "behavior_prior": "replay_behavior_clone_frozen_before_rl",
            "reverse_kl_scale": 0.3,
            "critic": "51_bin_symlog_two_hot_with_slow_ema",
            "real_replay_value_grounding": True,
        },
        "epistemic_penalty_scales": [float(value) for value in epistemic_scales],
        "shared_probe": {
            "states": 16,
            "candidates": "replay_plus_coordinatewise_minus_plus_0.1",
            "horizons": [1, 3, 5, 15],
            "draws": 8,
            "policy_independent": True,
        },
        "source_artifacts": sources,
        "interpretation": {
            "claim_eligible": False,
            "evidence_class": "exploratory_paired_reacher_intervention",
            "frozen_world_models": True,
            "paired_actor_repairs_applied_to_every_arm": True,
        },
    }
    digest = benchmark.object_sha256(manifest)
    return {**manifest, "manifest_sha256": digest}


def write_manifest(
    pilot_root: str | Path,
    causal_root: str | Path,
    output_root: str | Path,
    *,
    actor_updates: int,
    preparation_updates: int,
    epistemic_scales: tuple[float, ...] = (0.0,),
) -> dict[str, Any]:
    output = Path(output_root)
    path = output / "manifest.json"
    expected = build_manifest(
        pilot_root,
        causal_root,
        actor_updates=actor_updates,
        preparation_updates=preparation_updates,
        epistemic_scales=epistemic_scales,
    )
    if path.is_file():
        existing = read_json(path)
        if existing != expected:
            raise ValueError("existing vNext manifest differs from requested study")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, expected)
    return expected


def _source_row(manifest: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    matches = [
        row for row in manifest["source_artifacts"] if row["world_model_seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError("manifest seed source is missing or duplicated")
    return matches[0]


def prepare_probe_bank(
    pilot_root: str | Path,
    causal_root: str | Path,
    output_root: str | Path,
    seed: int,
    *,
    actor_updates: int,
    preparation_updates: int,
    epistemic_scales: tuple[float, ...] = (0.0,),
    action_repeat: int = 1,
) -> dict[str, Any]:
    from imf_dreamer_jax import load_checkpoint

    manifest = write_manifest(
        pilot_root,
        causal_root,
        output_root,
        actor_updates=actor_updates,
        preparation_updates=preparation_updates,
        epistemic_scales=epistemic_scales,
    )
    source = _source_row(manifest, seed)
    directory = Path(output_root) / "probe_bank" / f"seed-{seed}"
    result_path = directory / "result.json"
    bank_path = directory / "probe_bank.npz"
    if result_path.is_file():
        result = read_json(result_path)
        if result["probe_bank_file_sha256"] != benchmark.file_sha256(bank_path):
            raise ValueError("existing shared probe bank digest mismatch")
        return result
    arrays = benchmark.load_npz(source["dataset"])
    stochastic_dims = {
        int(load_checkpoint(Path(source["world_models"][arm]["checkpoint"]))[1].stochastic_dim)
        for arm in ARMS
    }
    if len(stochastic_dims) != 1:
        raise ValueError("shared probe models have different stochastic dimensions")
    bank = generate_shared_probe_bank(
        arrays,
        task=TASK,
        world_model_seed=seed,
        action_repeat=action_repeat,
        probes=manifest["shared_probe"]["states"],
        horizons=manifest["shared_probe"]["horizons"],
        stochastic_dim=stochastic_dims.pop(),
        draws=manifest["shared_probe"]["draws"],
    )
    contract = shared_probe_manifest(
        bank,
        task=TASK,
        world_model_seed=seed,
        dataset_sha256=source["dataset_sha256"],
        action_delta=0.1,
    )
    directory.mkdir(parents=True, exist_ok=True)
    benchmark._write_npz_atomic(bank_path, bank)
    result = {
        "schema_version": SCHEMA,
        "stage": "probe_bank",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        **contract,
        "probe_bank_file_sha256": benchmark.file_sha256(bank_path),
        "maximum_replay_error": float(bank["maximum_replay_error"][0]),
    }
    write_json_atomic(result_path, result)
    return result


def evaluate_probe_arm(
    pilot_root: str | Path,
    causal_root: str | Path,
    output_root: str | Path,
    seed: int,
    arm: str,
    *,
    actor_updates: int,
    preparation_updates: int,
    epistemic_scales: tuple[float, ...] = (0.0,),
) -> dict[str, Any]:
    from imf_dreamer_jax import load_checkpoint

    manifest = write_manifest(
        pilot_root,
        causal_root,
        output_root,
        actor_updates=actor_updates,
        preparation_updates=preparation_updates,
        epistemic_scales=epistemic_scales,
    )
    source = _source_row(manifest, seed)
    probe_result = read_json(
        Path(output_root) / "probe_bank" / f"seed-{seed}" / "result.json"
    )
    bank = benchmark.load_npz(
        Path(output_root) / "probe_bank" / f"seed-{seed}" / "probe_bank.npz"
    )
    validate_shared_probe_bank(bank)
    if probe_result["probe_bank_sha256"] != benchmark.array_sha256(bank):
        raise ValueError("probe result is not bound to retained bank")
    directory = Path(output_root) / "probe_evaluation" / f"seed-{seed}-{arm}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    checkpoint = Path(source["world_models"][arm]["checkpoint"])
    state, config, _ = load_checkpoint(checkpoint)
    arrays = benchmark.load_npz(source["dataset"])
    evaluation = evaluate_shared_probe_bank(
        state.params.world_model, config, arrays, bank
    )
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = benchmark._write_npz_atomic(
        directory / "predicted_returns.npz",
        {
            "predicted_returns": evaluation.pop("predicted_returns"),
            "predicted_return_draws": evaluation.pop("predicted_return_draws"),
            "simulator_returns": bank["simulator_returns"],
            "horizon_mask": bank["horizon_mask"],
        },
    )
    result = {
        "schema_version": SCHEMA,
        "stage": "probe_evaluation",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "arm": arm,
        "world_model_checkpoint_sha256": source["world_models"][arm][
            "checkpoint_sha256"
        ],
        **evaluation,
        "raw_sha256": benchmark.file_sha256(raw_path),
    }
    write_json_atomic(result_path, result)
    return result


def _tree_delta(left: Any, right: Any) -> float:
    import jax

    left_values, left_tree = jax.tree_util.tree_flatten(left)
    right_values, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree:
        raise ValueError("parameter tree structures differ")
    return max(
        (
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(left_values, right_values, strict=True)
        ),
        default=0.0,
    )


def train_repaired_actor(
    pilot_root: str | Path,
    causal_root: str | Path,
    output_root: str | Path,
    seed: int,
    arm: str,
    horizon: int,
    *,
    actor_updates: int,
    preparation_updates: int,
    epistemic_scale: float = 0.0,
    epistemic_scales: tuple[float, ...] = (0.0,),
    evaluation_episodes: int = 5,
) -> dict[str, Any]:
    """Apply the same frozen-world actor repair to one arm and horizon."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        AgentParams,
        AgentState,
        create_agent,
        diverse_imagination_starts,
        init_bootstrap_reward_ensemble,
        jit_observe_sequence,
        jit_train_actor_critic,
        jit_train_behavior_cloning,
        jit_train_bootstrap_reward_ensemble,
        jit_train_replay_critic,
        load_checkpoint,
        save_checkpoint,
        snapshot_behavior_prior,
    )

    if horizon not in HORIZONS or arm not in ARMS:
        raise ValueError("actor cell is outside the frozen vNext design")
    if epistemic_scale not in epistemic_scales:
        raise ValueError("epistemic scale is absent from the manifest")
    manifest = write_manifest(
        pilot_root,
        causal_root,
        output_root,
        actor_updates=actor_updates,
        preparation_updates=preparation_updates,
        epistemic_scales=epistemic_scales,
    )
    source = _source_row(manifest, seed)
    tag = f"seed-{seed}-{arm}-h{horizon}-u{epistemic_scale:g}"
    directory = Path(output_root) / "actor" / tag
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    world_state, stored_config, _ = load_checkpoint(
        Path(source["world_models"][arm]["checkpoint"])
    )
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
            "vnext-actor-init", TASK, seed, ACTOR_SEED, horizon, epistemic_scale
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
    batch_size = 32
    sequence_length = 32
    total_schedule = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=benchmark.derive_seed(
            "vnext-actor-batches", seed, ACTOR_SEED, horizon, epistemic_scale
        ),
        updates=preparation_updates + actor_updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    ensemble = (
        init_bootstrap_reward_ensemble(
            benchmark.derive_jax_key(
                "vnext-reward-ensemble", seed, ACTOR_SEED, horizon
            ),
            config.feature_dim,
            config.hidden_dim,
            members=5,
        )
        if epistemic_scale > 0.0
        else None
    )
    posterior_key = benchmark.derive_jax_key(
        "vnext-actor-posterior", seed, ACTOR_SEED, horizon, epistemic_scale
    )
    started = time.perf_counter()
    latest_bc = math.nan
    latest_replay_critic = math.nan
    for update in range(preparation_updates):
        replay = benchmark._materialize_batch(
            arrays,
            total_schedule,
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
        state, critic_loss = jit_train_replay_critic(
            state,
            sequence.states.feature,
            replay["rewards"],
            replay["continuations"],
            config,
            loss_mask=replay["loss_mask"],
        )
        if ensemble is not None:
            ensemble, _ = jit_train_bootstrap_reward_ensemble(
                ensemble,
                features,
                replay["rewards"][:, config.burn_in :],
                jax.random.fold_in(posterior_key, 1_000_000 + update),
                learning_rate=config.model_learning_rate,
                grad_clip=config.grad_clip,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                epsilon=config.adam_epsilon,
            )
        latest_bc = float(bc_loss)
        latest_replay_critic = float(critic_loss)
    behavior_prior = snapshot_behavior_prior(state.params.actor)
    latest = None
    start_key = benchmark.derive_jax_key(
        "vnext-actor-start", seed, ACTOR_SEED, horizon, epistemic_scale
    )
    objective_key = benchmark.derive_jax_key(
        "vnext-actor-objective", seed, ACTOR_SEED, horizon, epistemic_scale
    )
    for update in range(actor_updates):
        schedule_index = preparation_updates + update
        replay = benchmark._materialize_batch(
            arrays,
            total_schedule,
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
        state, metrics = jit_train_actor_critic(
            state,
            starts,
            jax.random.fold_in(objective_key, update),
            config,
            behavior_prior=behavior_prior,
            reward_ensemble_params=None if ensemble is None else ensemble.params,
            epistemic_penalty_scale=epistemic_scale,
        )
        latest = benchmark._metrics_dict(metrics)
    if latest is None:
        raise RuntimeError("actor repair performed no actor update")
    world_delta = _tree_delta(frozen_world, state.params.world_model)
    prior_delta = _tree_delta(behavior_prior, snapshot_behavior_prior(behavior_prior))
    if world_delta != 0.0 or prior_delta != 0.0:
        raise RuntimeError("actor-only repair mutated a frozen parameter tree")
    returns, traces = benchmark._evaluate_actor_policy(
        state,
        config,
        task=TASK,
        world_model_seed=seed,
        actor_seed=ACTOR_SEED,
        episodes=evaluation_episodes,
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
            "stage": "policy_consistency_vnext_actor",
            "source_world_model_checkpoint_sha256": source["world_models"][arm][
                "checkpoint_sha256"
            ],
            "completed_updates": actor_updates,
            "preparation_updates": preparation_updates,
            "behavior_prior_frozen": True,
            "epistemic_penalty_scale": epistemic_scale,
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
        "arm": arm,
        "imagination_horizon": horizon,
        "epistemic_penalty_scale": epistemic_scale,
        "world_model_checkpoint_sha256": source["world_models"][arm][
            "checkpoint_sha256"
        ],
        "world_model_frozen": True,
        "world_model_parameter_delta": world_delta,
        "behavior_prior_frozen": True,
        "behavior_prior_parameter_delta": prior_delta,
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
        "runtime_config": asdict(config),
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def finalize(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    manifest = read_json(output / "manifest.json")
    actor_rows = [read_json(path) for path in sorted(output.glob("actor/*/result.json"))]
    probe_rows = [
        read_json(path) for path in sorted(output.glob("probe_evaluation/*/result.json"))
    ]
    actor_groups = []
    for scale in manifest["epistemic_penalty_scales"]:
        for horizon in HORIZONS:
            for arm in ARMS:
                rows = [
                    row
                    for row in actor_rows
                    if row["arm"] == arm
                    and row["imagination_horizon"] == horizon
                    and row["epistemic_penalty_scale"] == scale
                ]
                if rows:
                    values = np.asarray(
                        [row["normalized_episode_return_mean"] for row in rows],
                        dtype=np.float64,
                    )
                    actor_groups.append(
                        {
                            "arm": arm,
                            "imagination_horizon": horizon,
                            "epistemic_penalty_scale": scale,
                            "cells": len(rows),
                            "mean_normalized_return": float(np.mean(values)),
                            "per_seed": {
                                str(row["world_model_seed"]): row[
                                    "normalized_episode_return_mean"
                                ]
                                for row in rows
                            },
                            "mean_imagined_return": float(
                                np.mean(
                                    [
                                        row["final_metrics"]["mean_imagined_return"]
                                        for row in rows
                                    ]
                                )
                            ),
                            "mean_action_saturation": float(
                                np.mean(
                                    [
                                        row["final_metrics"]["action_saturation"]
                                        for row in rows
                                    ]
                                )
                            ),
                        }
                    )
    probe_groups = {
        arm: {
            horizon: {
                "pairwise_accuracy_mean": float(
                    np.mean(
                        [
                            row["metrics_by_horizon"][horizon]["pairwise_accuracy"]
                            for row in probe_rows
                            if row["arm"] == arm
                            and row["metrics_by_horizon"][horizon]["pairwise_accuracy"]
                            is not None
                        ]
                    )
                ),
                "mean_simulator_regret": float(
                    np.mean(
                        [
                            row["metrics_by_horizon"][horizon][
                                "mean_simulator_regret"
                            ]
                            for row in probe_rows
                            if row["arm"] == arm
                        ]
                    )
                ),
            }
            for horizon in ("1", "3", "5", "15")
        }
        for arm in ARMS
        if any(row["arm"] == arm for row in probe_rows)
    }
    expected_probe_cells = len(WORLD_MODEL_SEEDS) * len(ARMS)
    expected_actor_cells = (
        len(WORLD_MODEL_SEEDS)
        * len(ARMS)
        * len(HORIZONS)
        * len(manifest["epistemic_penalty_scales"])
    )
    report = {
        "schema_version": SCHEMA,
        "stage": "final",
        "status": "complete" if len(actor_rows) == expected_actor_cells and len(probe_rows) == expected_probe_cells else "partial",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_actor_cells": len(actor_rows),
        "expected_actor_cells": expected_actor_cells,
        "completed_probe_cells": len(probe_rows),
        "expected_probe_cells": expected_probe_cells,
        "actor_groups": actor_groups,
        "shared_probe_groups": probe_groups,
        "interpretation": manifest["interpretation"],
    }
    write_json_atomic(output / "report.json", report)
    return report


__all__ = [
    "ACTOR_SEED",
    "ARMS",
    "HORIZONS",
    "SCHEMA",
    "TASK",
    "WORLD_MODEL_SEEDS",
    "build_manifest",
    "evaluate_probe_arm",
    "finalize",
    "prepare_probe_bank",
    "train_repaired_actor",
    "write_manifest",
]
