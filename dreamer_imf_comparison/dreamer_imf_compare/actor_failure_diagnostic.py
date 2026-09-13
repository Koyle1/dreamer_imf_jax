"""Shared-scenario causal ladder for diagnosing Reacher actor collapse."""

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
from . import reward_head_mtp_study as mtp
from .dmc import DMCAdapter
from .policy_consistency_vnext import _tree_delta


SCHEMA = "trajectory-imf-actor-failure-diagnostic-v1"
RESULT_SCHEMA = "trajectory-imf-actor-failure-cell-v1"
REPORT_SCHEMA = "trajectory-imf-actor-failure-report-v1"
TASK = "dmc_reacher_easy"
WORLD_MODEL_SEEDS = (211, 223)
ACTOR_SEEDS = (311, 313)
HORIZONS = (5, 15)
ARMS = (
    "bc_no_pmpo",
    "base_mtp_pmpo",
    "residual_pmpo",
    "analytic_reward_pmpo",
    "analytic_reward_unit_continuation_pmpo",
    "synthetic_action_reward_pmpo",
)
PMPO_ARMS = ARMS[1:]
REACHER_TO_TARGET = slice(2, 4)
REACHER_SUCCESS_RADIUS = 0.06
SYNTHETIC_ACTION_TARGET = 0.5


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def analytic_reacher_reward_from_observation(observation: Any) -> Any:
    """Exact DMC Reacher Easy reward for one-step flattened observations.

    The adapter sorts ``position``, ``to_target``, and ``velocity``. Reacher's
    target and finger radii are 0.05 and 0.01, and the task uses zero-margin
    tolerance, so reward is the indicator ``||to_target|| <= 0.06``.
    """

    import jax.numpy as jnp

    value = jnp.asarray(observation)
    if value.shape[-1] != 6:
        raise ValueError("Reacher oracle requires flattened observation dimension 6")
    distance = jnp.linalg.norm(value[..., REACHER_TO_TARGET], axis=-1)
    return (distance <= REACHER_SUCCESS_RADIUS).astype(value.dtype)


def analytic_reacher_reward(
    params: Any,
    previous_feature: Any,
    action: Any,
    next_feature: Any,
    config: Any,
) -> Any:
    """Recompute sparse task reward from the model's decoded next observation."""

    del previous_feature, action
    from imf_dreamer_jax import decode

    return analytic_reacher_reward_from_observation(
        decode(params, next_feature, config)
    )


def unit_continuation(
    params: Any,
    previous_feature: Any,
    action: Any,
    next_feature: Any,
    config: Any,
) -> Any:
    """Remove learned continuation for the nonterminating part of Reacher."""

    del params, previous_feature, next_feature, config
    import jax.numpy as jnp

    return jnp.ones(action.shape[:-1], dtype=action.dtype)


def synthetic_action_reward(
    params: Any,
    previous_feature: Any,
    action: Any,
    next_feature: Any,
    config: Any,
) -> Any:
    """Known concave action reward used only to test PMPO competence."""

    del params, previous_feature, next_feature, config
    import jax.numpy as jnp

    return 1.0 - jnp.mean(
        jnp.square(action - SYNTHETIC_ACTION_TARGET), axis=-1
    )


def signal_functions(arm: str) -> tuple[Any | None, Any | None]:
    if arm in ("bc_no_pmpo", "base_mtp_pmpo", "residual_pmpo"):
        return None, None
    if arm == "analytic_reward_pmpo":
        return analytic_reacher_reward, None
    if arm == "analytic_reward_unit_continuation_pmpo":
        return analytic_reacher_reward, unit_continuation
    if arm == "synthetic_action_reward_pmpo":
        return synthetic_action_reward, unit_continuation
    raise ValueError(f"unknown diagnostic arm {arm!r}")


def shared_evaluation_seeds(episodes: int) -> list[int]:
    if episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    return [
        benchmark.derive_seed("actor-failure-shared-evaluation", TASK, episode)
        for episode in range(episodes)
    ]


def build_cell_matrix() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = [
        {
            "index": 0,
            "stage": "random",
            "arm": "random_policy",
            "world_model_seed": None,
            "actor_seed": None,
            "horizon": None,
            "result_path": "random/result.json",
            "raw_path": "random/action_traces.npz",
        }
    ]
    for arm in ARMS:
        for world_model_seed in WORLD_MODEL_SEEDS:
            for actor_seed in ACTOR_SEEDS:
                for horizon in HORIZONS:
                    cells.append(
                        {
                            "index": len(cells),
                            "stage": "actor",
                            "arm": arm,
                            "world_model_seed": world_model_seed,
                            "actor_seed": actor_seed,
                            "horizon": horizon,
                            "result_path": (
                                f"actor/{arm}/seed-{world_model_seed}-"
                                f"actor-{actor_seed}-h{horizon}/result.json"
                            ),
                            "raw_path": (
                                f"actor/{arm}/seed-{world_model_seed}-"
                                f"actor-{actor_seed}-h{horizon}/action_traces.npz"
                            ),
                        }
                    )
    for cell in cells:
        identity = {key: cell[key] for key in (
            "index", "stage", "arm", "world_model_seed", "actor_seed", "horizon"
        )}
        cell["cell_id"] = benchmark.object_sha256(identity)[:24]
    return cells


def validate_cell_matrix(cells: Sequence[Mapping[str, Any]]) -> None:
    expected = build_cell_matrix()
    if _json_native(list(cells)) != expected:
        raise ValueError("diagnostic cell matrix differs from the frozen design")
    ids = [cell["cell_id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("diagnostic cell ids are not unique")


def _source_row(manifest: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    rows = [
        row for row in manifest["source_artifacts"]
        if int(row["world_model_seed"]) == int(seed)
    ]
    if len(rows) != 1:
        raise ValueError("source world-model seed is absent or duplicated")
    return rows[0]


def build_manifest(
    mtp_root: str | Path,
    residual_root: str | Path,
    *,
    actor_updates: int = 3000,
    preparation_updates: int = 500,
    evaluation_episodes: int = 50,
) -> dict[str, Any]:
    if min(actor_updates, preparation_updates, evaluation_episodes) <= 0:
        raise ValueError("update and evaluation counts must be positive")
    base = Path(mtp_root).resolve(strict=True)
    residual = Path(residual_root).resolve(strict=True)
    base_manifest = read_json(base / "manifest.json")
    base_report = read_json(base / "report.json")
    residual_manifest = read_json(residual / "manifest.json")
    residual_report = read_json(residual / "report.json")
    if (
        base_report.get("status") != "complete"
        or base_report.get("manifest_sha256") != base_manifest.get("manifest_sha256")
    ):
        raise ValueError("base MTP dependency is incomplete")
    if (
        residual_report.get("status") != "complete"
        or residual_report.get("manifest_sha256")
        != residual_manifest.get("manifest_sha256")
        or residual_manifest.get("objective_identity")
        != "exact_dense_quadratic_plus_relative_error_control_h1_to_h15"
    ):
        raise ValueError("exact residual dependency is incomplete or incompatible")
    cells = build_cell_matrix()
    sources = []
    for seed in WORLD_MODEL_SEEDS:
        base_source = mtp._source_row(base_manifest, seed)
        base_directory = base / "reward" / f"seed-{seed}-dreamer4_twohot_mtp"
        base_result = read_json(base_directory / "result.json")
        base_checkpoint = base_directory / "checkpoint.pkl"
        residual_directory = residual / "residual" / f"seed-{seed}"
        residual_result = read_json(residual_directory / "result.json")
        residual_checkpoint = residual_directory / "checkpoint.pkl"
        checks = (
            (base_checkpoint, base_result["checkpoint_sha256"], "base checkpoint"),
            (
                residual_checkpoint,
                residual_result["checkpoint_sha256"],
                "residual checkpoint",
            ),
            (
                Path(base_source["dataset"]),
                None,
                "source dataset",
            ),
        )
        for path, digest, label in checks:
            if not path.is_file():
                raise ValueError(f"{label} is missing for seed {seed}")
            if digest is not None and benchmark.file_sha256(path) != digest:
                raise ValueError(f"{label} digest mismatch for seed {seed}")
        arrays = benchmark.load_npz(base_source["dataset"])
        if benchmark.array_sha256(arrays) != base_source["dataset_sha256"]:
            raise ValueError("source dataset payload digest mismatch")
        if residual_result.get("source_checkpoint_sha256") != base_result["checkpoint_sha256"]:
            raise ValueError("residual is not derived from the selected MTP checkpoint")
        sources.append(
            {
                "world_model_seed": seed,
                "dataset": str(Path(base_source["dataset"]).resolve()),
                "dataset_sha256": base_source["dataset_sha256"],
                "base_checkpoint": str(base_checkpoint.resolve()),
                "base_checkpoint_sha256": base_result["checkpoint_sha256"],
                "residual_checkpoint": str(residual_checkpoint.resolve()),
                "residual_checkpoint_sha256": residual_result["checkpoint_sha256"],
            }
        )
    body = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "task": TASK,
        "world_model_seeds": list(WORLD_MODEL_SEEDS),
        "actor_seeds": list(ACTOR_SEEDS),
        "horizons": list(HORIZONS),
        "arms": list(ARMS),
        "actor_updates": actor_updates,
        "preparation_updates": preparation_updates,
        "evaluation_episodes": evaluation_episodes,
        "maximum_environment_steps": 1000,
        "shared_evaluation_seeds": shared_evaluation_seeds(evaluation_episodes),
        "source_artifacts": sources,
        "dependencies": {
            "mtp_root": str(base),
            "mtp_manifest_sha256": base_manifest["manifest_sha256"],
            "mtp_report_file_sha256": benchmark.file_sha256(base / "report.json"),
            "residual_root": str(residual),
            "residual_manifest_sha256": residual_manifest["manifest_sha256"],
            "residual_report_file_sha256": benchmark.file_sha256(
                residual / "report.json"
            ),
        },
        "analytic_reward": {
            "flattened_observation_order": ["position", "to_target", "velocity"],
            "to_target_indices": [2, 4],
            "target_radius": 0.05,
            "finger_radius": 0.01,
            "success_radius": REACHER_SUCCESS_RADIUS,
            "semantics": "indicator_l2_to_target_at_most_sum_of_geom_radii",
        },
        "synthetic_competence": {
            "action_target": SYNTHETIC_ACTION_TARGET,
            "required_mean_absolute_error": 0.25,
            "required_fraction_improved_over_bc": 0.75,
            "required_mean_improvement_over_bc": 0.05,
        },
        "effect_thresholds": {
            "normalized_return_mean": 0.005,
            "world_model_seed_fraction": 1.0,
        },
        "independent_unit": "world_model_seed",
        "actor_seed_role": "conditional_optimization_variance_only",
        "cells": cells,
    }
    body["manifest_sha256"] = benchmark.object_sha256(body)
    return body


def write_manifest(
    mtp_root: str | Path,
    residual_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = build_manifest(mtp_root, residual_root, **settings)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.is_file():
        if read_json(path) != manifest:
            raise ValueError("existing diagnostic manifest differs from frozen design")
    else:
        write_json_atomic(path, manifest)
    return manifest


def _numpy_analytic_reward(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations)
    if values.shape[-1] != 6:
        raise ValueError("Reacher observations must have dimension 6")
    return (
        np.linalg.norm(values[..., REACHER_TO_TARGET], axis=-1)
        <= REACHER_SUCCESS_RADIUS
    ).astype(np.float32)


def record_preflight(
    mtp_root: str | Path,
    residual_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import decode, jit_observe_sequence, load_checkpoint

    manifest = write_manifest(mtp_root, residual_root, output_root, **settings)
    output = Path(output_root)
    path = output / "preflight.json"
    if path.is_file():
        return read_json(path)
    validation = {}
    for seed in WORLD_MODEL_SEEDS:
        source = _source_row(manifest, seed)
        arrays = benchmark.load_npz(source["dataset"])
        analytic = _numpy_analytic_reward(arrays["observations"])
        mask = ~np.asarray(arrays["is_first"], dtype=np.bool_)
        maximum_error = float(np.max(np.abs(
            analytic[mask] - np.asarray(arrays["rewards"])[mask]
        )))
        if maximum_error > 1e-6:
            raise ValueError("analytic Reacher reward does not reproduce raw replay")
        state, config, _ = load_checkpoint(source["base_checkpoint"])
        if config.observation_shape != (6,) or config.action_dim != 2:
            raise ValueError("source checkpoint is not the expected Reacher model")
        schedule = mtp._heldout_schedule(
            arrays, seed, batch_size=64, sequence_length=32
        )
        batch = benchmark._materialize_batch(
            arrays, schedule, 0, sequence_length=32, burn_in=config.burn_in
        )
        sequence = jit_observe_sequence(
            state.params.world_model,
            batch["observations"],
            batch["actions"],
            benchmark.derive_jax_key("actor-failure-preflight-posterior", seed),
            config,
            is_first=batch["is_first"],
        )
        decoded = np.asarray(jax.device_get(
            decode(state.params.world_model, sequence.states.feature, config)
        ))
        decoded_reward = _numpy_analytic_reward(decoded)
        targets = np.asarray(batch["rewards"])
        heldout_mask = np.asarray(batch["loss_mask"], dtype=np.bool_)
        validation[str(seed)] = {
            "raw_reward_max_abs_error": maximum_error,
            "decoded_reward_accuracy": float(np.mean(
                decoded_reward[heldout_mask] == targets[heldout_mask]
            )),
            "decoded_reward_mse": float(np.mean(np.square(
                decoded_reward[heldout_mask] - targets[heldout_mask]
            ))),
            "decoded_positive_rate": float(np.mean(decoded_reward[heldout_mask])),
            "target_positive_rate": float(np.mean(targets[heldout_mask])),
        }
    result = {
        "schema_version": SCHEMA,
        "stage": "preflight",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "analytic_reward_validation": validation,
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json_atomic(path, result)
    return result


def _pack_traces(
    actions: Sequence[np.ndarray],
    rewards: Sequence[np.ndarray],
    continuations: Sequence[np.ndarray],
    terminals: Sequence[np.ndarray],
    evaluation_seeds: Sequence[int],
    action_dim: int,
) -> dict[str, np.ndarray]:
    lengths = np.asarray([len(trace) for trace in actions], dtype=np.int32)
    maximum = int(lengths.max(initial=0))
    packed_actions = np.zeros((len(actions), maximum, action_dim), np.float32)
    packed_rewards = np.zeros((len(actions), maximum), np.float64)
    packed_continuations = np.zeros((len(actions), maximum), np.float64)
    packed_terminals = np.zeros((len(actions), maximum), np.bool_)
    for index, trace in enumerate(actions):
        length = len(trace)
        packed_actions[index, :length] = trace
        packed_rewards[index, :length] = rewards[index]
        packed_continuations[index, :length] = continuations[index]
        packed_terminals[index, :length] = terminals[index]
    return {
        "actions": packed_actions,
        "rewards": packed_rewards,
        "continuations": packed_continuations,
        "is_last": packed_terminals,
        "lengths": lengths,
        "evaluation_seeds": np.asarray(evaluation_seeds, dtype=np.uint32),
    }


def _evaluate_policy_shared(
    state: Any,
    config: Any,
    *,
    evaluation_seeds: Sequence[int],
    world_model_seed: int,
    actor_seed: int,
    horizon: int,
    maximum_steps: int,
) -> tuple[list[float], dict[str, np.ndarray]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state, jit_act

    returns: list[float] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    continuations: list[np.ndarray] = []
    terminals: list[np.ndarray] = []
    for episode, evaluation_seed in enumerate(evaluation_seeds):
        environment = DMCAdapter(TASK, seed=int(evaluation_seed), action_repeat=1)
        episode_actions: list[np.ndarray] = []
        episode_rewards: list[float] = []
        episode_continuations: list[float] = []
        episode_terminals: list[bool] = []
        try:
            observation = environment.reset()
            belief = initial_state(config, 1)
            previous_action = jnp.zeros((1, config.action_dim), jnp.float32)
            key = benchmark.derive_jax_key(
                "actor-failure-shared-policy-evaluation",
                world_model_seed,
                actor_seed,
                horizon,
                episode,
            )
            for step in range(maximum_steps):
                action, belief = jit_act(
                    state.params,
                    jnp.asarray(observation[None]),
                    previous_action,
                    belief,
                    jax.random.fold_in(key, step),
                    config,
                    deterministic=True,
                )
                host_action = np.asarray(action[0], dtype=np.float32)
                transition = environment.step(host_action)
                episode_actions.append(host_action)
                episode_rewards.append(float(transition.reward))
                episode_continuations.append(float(transition.continuation))
                episode_terminals.append(bool(transition.is_last))
                observation = transition.observation
                previous_action = action
                if transition.is_last:
                    break
        finally:
            environment.close()
        action_array = np.asarray(episode_actions, dtype=np.float32)
        reward_array = np.asarray(episode_rewards, dtype=np.float64)
        actions.append(action_array)
        rewards.append(reward_array)
        continuations.append(np.asarray(episode_continuations, dtype=np.float64))
        terminals.append(np.asarray(episode_terminals, dtype=np.bool_))
        returns.append(float(np.sum(reward_array)))
    return returns, _pack_traces(
        actions,
        rewards,
        continuations,
        terminals,
        evaluation_seeds,
        config.action_dim,
    )


def _evaluate_random_shared(
    *, evaluation_seeds: Sequence[int], maximum_steps: int
) -> tuple[list[float], dict[str, np.ndarray]]:
    returns: list[float] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    continuations: list[np.ndarray] = []
    terminals: list[np.ndarray] = []
    action_dim = 2
    for evaluation_seed in evaluation_seeds:
        environment = DMCAdapter(TASK, seed=int(evaluation_seed), action_repeat=1)
        random = np.random.default_rng(
            benchmark.derive_seed("actor-failure-random-actions", int(evaluation_seed))
        )
        episode_actions: list[np.ndarray] = []
        episode_rewards: list[float] = []
        episode_continuations: list[float] = []
        episode_terminals: list[bool] = []
        try:
            environment.reset()
            for _ in range(maximum_steps):
                action = random.uniform(-1.0, 1.0, action_dim).astype(np.float32)
                transition = environment.step(action)
                episode_actions.append(action)
                episode_rewards.append(float(transition.reward))
                episode_continuations.append(float(transition.continuation))
                episode_terminals.append(bool(transition.is_last))
                if transition.is_last:
                    break
        finally:
            environment.close()
        action_array = np.asarray(episode_actions, dtype=np.float32)
        reward_array = np.asarray(episode_rewards, dtype=np.float64)
        actions.append(action_array)
        rewards.append(reward_array)
        continuations.append(np.asarray(episode_continuations, dtype=np.float64))
        terminals.append(np.asarray(episode_terminals, dtype=np.bool_))
        returns.append(float(np.sum(reward_array)))
    return returns, _pack_traces(
        actions,
        rewards,
        continuations,
        terminals,
        evaluation_seeds,
        action_dim,
    )


def _trace_metrics(returns: Sequence[float], traces: Mapping[str, np.ndarray]) -> dict[str, Any]:
    lengths = np.asarray(traces["lengths"], dtype=np.int32)
    mask = np.arange(traces["actions"].shape[1])[None, :] < lengths[:, None]
    actions = np.asarray(traces["actions"])
    target_error = np.abs(actions - SYNTHETIC_ACTION_TARGET)
    valid_target_error = target_error[mask[..., None].repeat(actions.shape[-1], axis=-1)]
    normalized = np.asarray(returns, dtype=np.float64) / 1000.0
    return {
        "episode_returns": [float(value) for value in returns],
        "normalized_episode_returns": normalized.tolist(),
        "normalized_return_mean": float(np.mean(normalized)),
        "normalized_return_median": float(np.median(normalized)),
        "success_fraction": float(np.mean(normalized > 0.0)),
        "action_target_mean_absolute_error": float(np.mean(valid_target_error)),
        "mean_action": np.mean(actions[mask], axis=0).tolist(),
    }


def run_random_cell(
    mtp_root: str | Path,
    residual_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = write_manifest(mtp_root, residual_root, output_root, **settings)
    cell = manifest["cells"][0]
    directory = Path(output_root) / "random"
    result_path = directory / "result.json"
    raw_path = directory / "action_traces.npz"
    if result_path.is_file():
        return read_json(result_path)
    started = time.perf_counter()
    returns, traces = _evaluate_random_shared(
        evaluation_seeds=manifest["shared_evaluation_seeds"],
        maximum_steps=int(manifest["maximum_environment_steps"]),
    )
    directory.mkdir(parents=True, exist_ok=True)
    benchmark._write_npz_atomic(raw_path, traces)
    result = {
        "schema_version": RESULT_SCHEMA,
        "stage": "random",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": cell["index"],
        "arm": cell["arm"],
        "world_model_seed": None,
        "actor_seed": None,
        "horizon": None,
        "actor_updates": 0,
        "preparation_updates": 0,
        "metrics": _trace_metrics(returns, traces),
        "raw_action_traces_sha256": benchmark.file_sha256(raw_path),
        "wall_seconds": time.perf_counter() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def _cell_by_index(manifest: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    rows = [cell for cell in manifest["cells"] if int(cell["index"]) == int(index)]
    if len(rows) != 1:
        raise ValueError("diagnostic cell index is absent or duplicated")
    return rows[0]


def run_actor_cell(
    mtp_root: str | Path,
    residual_root: str | Path,
    output_root: str | Path,
    index: int,
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

    manifest = write_manifest(mtp_root, residual_root, output_root, **settings)
    cell = _cell_by_index(manifest, index)
    if cell["stage"] != "actor":
        raise ValueError("selected index is not an actor cell")
    directory = Path(output_root) / Path(cell["result_path"]).parent
    result_path = Path(output_root) / cell["result_path"]
    raw_path = Path(output_root) / cell["raw_path"]
    if result_path.is_file():
        return read_json(result_path)
    seed = int(cell["world_model_seed"])
    actor_seed = int(cell["actor_seed"])
    horizon = int(cell["horizon"])
    arm = str(cell["arm"])
    source = _source_row(manifest, seed)
    checkpoint_key = "residual_checkpoint" if arm == "residual_pmpo" else "base_checkpoint"
    checkpoint_digest_key = checkpoint_key + "_sha256"
    checkpoint = Path(source[checkpoint_key])
    if benchmark.file_sha256(checkpoint) != source[checkpoint_digest_key]:
        raise ValueError("actor source checkpoint digest mismatch")
    world_state, stored_config, _ = load_checkpoint(checkpoint)
    if (
        stored_config.prior != "imf"
        or not stored_config.imf_trajectory_enabled
        or stored_config.reward_loss != "symexp_twohot"
        or stored_config.reward_prediction_horizon != 8
    ):
        raise ValueError("actor source is not the required trajectory-iMF MTP model")
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
            "actor-failure-actor-init", TASK, seed, actor_seed, horizon
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
            "actor-failure-shared-batches", seed, actor_seed, horizon
        ),
        updates=preparation_updates + actor_updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    posterior_key = benchmark.derive_jax_key(
        "actor-failure-shared-posterior", seed, actor_seed, horizon
    )
    started = time.perf_counter()
    latest_bc = math.nan
    latest_replay_critic = math.nan
    for update in range(preparation_updates):
        replay = benchmark._materialize_batch(
            arrays, schedule, update, sequence_length=sequence_length,
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
        latest_bc = float(bc_loss)
        latest_replay_critic = float(critic_loss)
    behavior_prior = snapshot_behavior_prior(state.params.actor)
    reward_fn, continuation_fn = signal_functions(arm)
    latest = None
    if arm != "bc_no_pmpo":
        start_key = benchmark.derive_jax_key(
            "actor-failure-shared-start", seed, actor_seed, horizon
        )
        objective_key = benchmark.derive_jax_key(
            "actor-failure-shared-objective", seed, actor_seed, horizon
        )
        for update in range(actor_updates):
            schedule_index = preparation_updates + update
            replay = benchmark._materialize_batch(
                arrays, schedule, schedule_index, sequence_length=sequence_length,
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
                reward_ensemble_params=None,
                epistemic_penalty_scale=0.0,
                reward_fn=reward_fn,
                continuation_fn=continuation_fn,
            )
            latest = benchmark._metrics_dict(metrics)
    world_delta = _tree_delta(frozen_world, state.params.world_model)
    if world_delta != 0.0:
        raise RuntimeError("actor training mutated the frozen world model")
    returns, traces = _evaluate_policy_shared(
        state,
        config,
        evaluation_seeds=manifest["shared_evaluation_seeds"],
        world_model_seed=seed,
        actor_seed=actor_seed,
        horizon=horizon,
        maximum_steps=int(manifest["maximum_environment_steps"]),
    )
    directory.mkdir(parents=True, exist_ok=True)
    benchmark._write_npz_atomic(raw_path, traces)
    checkpoint_path = directory / "checkpoint.pkl"
    save_checkpoint(
        checkpoint_path,
        state,
        config,
        metadata={
            "stage": "actor_failure_diagnostic",
            "arm": arm,
            "cell_id": cell["cell_id"],
            "source_world_model_checkpoint_sha256": source[checkpoint_digest_key],
            "completed_updates": 0 if arm == "bc_no_pmpo" else actor_updates,
            "preparation_updates": preparation_updates,
            "shared_evaluation_seeds": True,
        },
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "stage": "actor",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": cell["index"],
        "arm": arm,
        "world_model_seed": seed,
        "actor_seed": actor_seed,
        "horizon": horizon,
        "source_checkpoint_sha256": source[checkpoint_digest_key],
        "world_model_parameter_delta": world_delta,
        "behavior_prior_frozen": True,
        "preparation_updates": preparation_updates,
        "actor_updates": 0 if arm == "bc_no_pmpo" else actor_updates,
        "final_behavior_cloning_loss": latest_bc,
        "final_replay_critic_loss": latest_replay_critic,
        "final_metrics": latest,
        "metrics": _trace_metrics(returns, traces),
        "checkpoint_sha256": benchmark.file_sha256(checkpoint_path),
        "raw_action_traces_sha256": benchmark.file_sha256(raw_path),
        "wall_seconds": time.perf_counter() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def validate_cell_result(
    result: Mapping[str, Any],
    cell: Mapping[str, Any],
    manifest: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
) -> None:
    required = {
        "schema_version", "stage", "status", "source_commit", "manifest_sha256",
        "cell_id", "cell_index", "arm", "world_model_seed", "actor_seed", "horizon",
        "actor_updates", "preparation_updates", "metrics", "raw_action_traces_sha256",
        "wall_seconds", "slurm_job_id", "runtime",
    }
    if result.get("stage") == "actor":
        required |= {
            "source_checkpoint_sha256", "world_model_parameter_delta",
            "behavior_prior_frozen", "final_behavior_cloning_loss",
            "final_replay_critic_loss", "final_metrics", "checkpoint_sha256",
        }
    if set(result) != required:
        raise ValueError("diagnostic cell result schema is not exact")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("diagnostic cell identity is invalid")
    for key, result_key in (
        ("cell_id", "cell_id"), ("index", "cell_index"), ("arm", "arm"),
        ("world_model_seed", "world_model_seed"), ("actor_seed", "actor_seed"),
        ("horizon", "horizon"), ("stage", "stage"),
    ):
        if result.get(result_key) != cell.get(key):
            raise ValueError(f"diagnostic result differs from cell field {key}")
    expected_seeds = np.asarray(manifest["shared_evaluation_seeds"], dtype=np.uint32)
    if not np.array_equal(np.asarray(raw["evaluation_seeds"]), expected_seeds):
        raise ValueError("diagnostic result did not use the shared evaluation scenarios")
    lengths = np.asarray(raw["lengths"], dtype=np.int32)
    rewards = np.asarray(raw["rewards"], dtype=np.float64)
    if len(lengths) != int(manifest["evaluation_episodes"]):
        raise ValueError("diagnostic trace has the wrong episode count")
    recomputed = np.asarray([
        float(np.sum(rewards[index, :length]))
        for index, length in enumerate(lengths)
    ])
    recorded = np.asarray(result["metrics"]["episode_returns"], dtype=np.float64)
    if not np.array_equal(recomputed, recorded):
        raise ValueError("diagnostic episode returns do not recompute from raw traces")
    if not math.isclose(
        float(np.mean(recomputed) / 1000.0),
        float(result["metrics"]["normalized_return_mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("diagnostic normalized mean does not recompute")
    if result["stage"] == "actor":
        expected_updates = 0 if result["arm"] == "bc_no_pmpo" else manifest["actor_updates"]
        if (
            result["actor_updates"] != expected_updates
            or result["preparation_updates"] != manifest["preparation_updates"]
            or result["world_model_parameter_delta"] != 0.0
            or result["behavior_prior_frozen"] is not True
        ):
            raise ValueError("diagnostic actor training contract is invalid")


def _paired_deltas(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str
) -> list[float]:
    keyed = {
        (row["arm"], row["world_model_seed"], row["actor_seed"], row["horizon"]): row
        for row in rows if row["stage"] == "actor"
    }
    values = []
    for world_model_seed in WORLD_MODEL_SEEDS:
        for actor_seed in ACTOR_SEEDS:
            for horizon in HORIZONS:
                left_row = keyed[(left, world_model_seed, actor_seed, horizon)]
                right_row = keyed[(right, world_model_seed, actor_seed, horizon)]
                values.append(float(left_row["metrics"][metric]) - float(right_row["metrics"][metric]))
    return values


def _seed_level_deltas(values: Sequence[float]) -> list[float]:
    """Collapse actor-seed/horizon repeats before interpreting an effect."""

    values_per_world_model = len(ACTOR_SEEDS) * len(HORIZONS)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(WORLD_MODEL_SEEDS) * values_per_world_model,):
        raise ValueError("paired diagnostic vector has the wrong shape")
    return [
        float(np.mean(array[index:index + values_per_world_model]))
        for index in range(0, len(array), values_per_world_model)
    ]


def diagnostic_decision(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    competence_errors = [
        -value for value in _paired_deltas(
            rows,
            "synthetic_action_reward_pmpo",
            "bc_no_pmpo",
            "action_target_mean_absolute_error",
        )
    ]
    synthetic_rows = [row for row in rows if row["arm"] == "synthetic_action_reward_pmpo"]
    synthetic_error = float(np.mean([
        row["metrics"]["action_target_mean_absolute_error"] for row in synthetic_rows
    ]))
    competence = manifest["synthetic_competence"]
    improved_fraction = float(np.mean(np.asarray(competence_errors) > 0.0))
    mean_improvement = float(np.mean(competence_errors))
    competence_passed = bool(
        synthetic_error <= competence["required_mean_absolute_error"]
        and improved_fraction >= competence["required_fraction_improved_over_bc"]
        and mean_improvement >= competence["required_mean_improvement_over_bc"]
    )
    return_metric = "normalized_return_mean"
    residual_rollback = _paired_deltas(rows, "base_mtp_pmpo", "residual_pmpo", return_metric)
    analytic_reward = _paired_deltas(rows, "analytic_reward_pmpo", "base_mtp_pmpo", return_metric)
    unit_continuation_delta = _paired_deltas(
        rows,
        "analytic_reward_unit_continuation_pmpo",
        "analytic_reward_pmpo",
        return_metric,
    )
    threshold = float(manifest["effect_thresholds"]["normalized_return_mean"])
    fraction_threshold = float(
        manifest["effect_thresholds"]["world_model_seed_fraction"]
    )

    def supported(values: Sequence[float]) -> bool:
        array = np.asarray(_seed_level_deltas(values), dtype=np.float64)
        return bool(
            float(np.mean(array)) >= threshold
            and float(np.mean(array > 0.0)) >= fraction_threshold
        )

    residual_harm = supported(residual_rollback)
    reward_binding = supported(analytic_reward)
    continuation_binding = supported(unit_continuation_delta)
    if not competence_passed:
        classification = "actor_harness_competence_not_established"
        next_action = "stop_model_changes_and_repair_actor_critic_harness"
    elif residual_harm:
        classification = "residual_induced_regression_supported"
        next_action = "remove_residual_and_retain_base_mtp"
    elif reward_binding:
        classification = "reward_head_bottleneck_supported"
        next_action = "retain_analytic_control_and_test_deployable_reward_head"
    elif continuation_binding:
        classification = "continuation_bottleneck_supported"
        next_action = "calibrate_or_refit_continuation_head"
    else:
        classification = "reward_and_continuation_not_sufficient_to_explain_failure"
        next_action = "audit_transition_decoder_critic_and_actor_occupancy"
    return {
        "classification": classification,
        "next_action": next_action,
        "actor_harness_competence_passed": competence_passed,
        "synthetic_action_target_mean_absolute_error": synthetic_error,
        "synthetic_improvement_over_bc_mean": mean_improvement,
        "synthetic_improvement_over_bc_fraction": improved_fraction,
        "residual_harm_supported": residual_harm,
        "reward_head_binding_supported": reward_binding,
        "continuation_binding_supported": continuation_binding,
        "paired_base_minus_residual": residual_rollback,
        "seed_level_base_minus_residual": _seed_level_deltas(residual_rollback),
        "paired_analytic_minus_base": analytic_reward,
        "seed_level_analytic_minus_base": _seed_level_deltas(analytic_reward),
        "paired_analytic_unit_continuation_minus_analytic": unit_continuation_delta,
        "seed_level_analytic_unit_continuation_minus_analytic": (
            _seed_level_deltas(unit_continuation_delta)
        ),
        "independent_unit": manifest["independent_unit"],
        "actor_seed_role": manifest["actor_seed_role"],
    }


def _arm_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for arm in ("random_policy",) + ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        horizons = (None,) if arm == "random_policy" else HORIZONS
        for horizon in horizons:
            selected = [row for row in arm_rows if row["horizon"] == horizon]
            if not selected:
                continue
            values = [row["metrics"]["normalized_return_mean"] for row in selected]
            groups.append({
                "arm": arm,
                "horizon": horizon,
                "cells": len(selected),
                "mean_normalized_return": float(np.mean(values)),
                "median_cell_normalized_return": float(np.median(values)),
                "mean_episode_median": float(np.mean([
                    row["metrics"]["normalized_return_median"] for row in selected
                ])),
            })
    return groups


def finalize(
    mtp_root: str | Path,
    residual_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = write_manifest(mtp_root, residual_root, output_root, **settings)
    output = Path(output_root)
    rows = []
    for cell in manifest["cells"]:
        result_path = output / cell["result_path"]
        raw_path = output / cell["raw_path"]
        if not result_path.is_file() or not raw_path.is_file():
            raise ValueError(f"diagnostic cell {cell['cell_id']} is incomplete")
        result = read_json(result_path)
        if benchmark.file_sha256(raw_path) != result["raw_action_traces_sha256"]:
            raise ValueError("diagnostic raw trace digest mismatch")
        raw = benchmark.load_npz(raw_path)
        validate_cell_result(result, cell, manifest, raw)
        rows.append(result)
    preflight = read_json(output / "preflight.json")
    if (
        preflight.get("status") != "complete"
        or preflight.get("manifest_sha256") != manifest["manifest_sha256"]
        or any(
            row["raw_reward_max_abs_error"] > 1e-6
            for row in preflight["analytic_reward_validation"].values()
        )
    ):
        raise ValueError("diagnostic preflight is incomplete")
    decision = diagnostic_decision(rows, manifest)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_cells": len(rows),
        "arm_groups": _arm_groups(rows),
        "decision": decision,
        "analytic_reward_validation": preflight["analytic_reward_validation"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
    }
    report["report_sha256"] = benchmark.object_sha256(report)
    write_json_atomic(output / "report.json", report)
    lines = [
        "# Actor-failure decisive diagnostic",
        "",
        f"Source commit: `{manifest['source_commit']}`",
        "",
        "| Arm | Horizon | Cells | Mean normalized return | Mean episode median |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["arm_groups"]:
        horizon = "—" if row["horizon"] is None else str(row["horizon"])
        lines.append(
            f"| {row['arm']} | {horizon} | {row['cells']} | "
            f"{row['mean_normalized_return']:.6f} | {row['mean_episode_median']:.6f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- Classification: `{decision['classification']}`",
        f"- Next action: `{decision['next_action']}`",
        f"- Actor-harness competence: `{decision['actor_harness_competence_passed']}`",
        f"- Residual harm supported: `{decision['residual_harm_supported']}`",
        f"- Reward-head bottleneck supported: `{decision['reward_head_binding_supported']}`",
        f"- Continuation bottleneck supported: `{decision['continuation_binding_supported']}`",
        "",
        "World-model seeds are the independent units. Actor seeds and episodes are conditional diagnostics only.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def validate_complete_evidence(evidence_root: str | Path) -> dict[str, Any]:
    root = Path(evidence_root).resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    expected_digest = manifest.pop("manifest_sha256", None)
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("status") != "frozen_before_execution"
        or benchmark.object_sha256(manifest) != expected_digest
    ):
        raise ValueError("diagnostic manifest is invalid")
    manifest["manifest_sha256"] = expected_digest
    validate_cell_matrix(manifest["cells"])
    if (
        manifest["arms"] != list(ARMS)
        or manifest["world_model_seeds"] != list(WORLD_MODEL_SEEDS)
        or manifest["actor_seeds"] != list(ACTOR_SEEDS)
        or manifest["horizons"] != list(HORIZONS)
        or manifest["shared_evaluation_seeds"]
        != shared_evaluation_seeds(manifest["evaluation_episodes"])
    ):
        raise ValueError("diagnostic manifest protocol is not frozen")
    preflight = read_json(root / "preflight.json")
    if (
        preflight.get("schema_version") != SCHEMA
        or preflight.get("stage") != "preflight"
        or preflight.get("status") != "complete"
        or preflight.get("source_commit") != manifest["source_commit"]
        or preflight.get("manifest_sha256") != manifest["manifest_sha256"]
        or set(preflight.get("analytic_reward_validation", {}))
        != {str(seed) for seed in WORLD_MODEL_SEEDS}
        or any(
            metrics.get("raw_reward_max_abs_error") != 0.0
            for metrics in preflight["analytic_reward_validation"].values()
        )
    ):
        raise ValueError("diagnostic preflight evidence is invalid")
    rows = []
    for cell in manifest["cells"]:
        result_path = root / cell["result_path"]
        raw_path = root / cell["raw_path"]
        result = read_json(result_path)
        if benchmark.file_sha256(raw_path) != result["raw_action_traces_sha256"]:
            raise ValueError("diagnostic retained trace digest mismatch")
        validate_cell_result(result, cell, manifest, benchmark.load_npz(raw_path))
        rows.append(result)
    report = read_json(root / "report.json")
    digest = report.pop("report_sha256", None)
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("completed_cells") != len(manifest["cells"])
        or benchmark.object_sha256(report) != digest
    ):
        raise ValueError("diagnostic report is invalid")
    report["report_sha256"] = digest
    recomputed = diagnostic_decision(rows, manifest)
    if (
        report["decision"] != recomputed
        or report["arm_groups"] != _arm_groups(rows)
        or report.get("analytic_reward_validation")
        != preflight["analytic_reward_validation"]
    ):
        raise ValueError("diagnostic report does not recompute from cell evidence")
    return report


__all__ = [
    "ACTOR_SEEDS",
    "ARMS",
    "HORIZONS",
    "RESULT_SCHEMA",
    "SCHEMA",
    "TASK",
    "WORLD_MODEL_SEEDS",
    "analytic_reacher_reward",
    "analytic_reacher_reward_from_observation",
    "build_cell_matrix",
    "build_manifest",
    "diagnostic_decision",
    "finalize",
    "record_preflight",
    "run_actor_cell",
    "run_random_cell",
    "shared_evaluation_seeds",
    "signal_functions",
    "synthetic_action_reward",
    "unit_continuation",
    "validate_cell_matrix",
    "validate_cell_result",
    "validate_complete_evidence",
    "write_manifest",
]
