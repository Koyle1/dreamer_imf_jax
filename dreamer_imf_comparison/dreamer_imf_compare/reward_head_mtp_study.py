"""Frozen four-arm Reacher study for the isolated reward-model bottleneck."""

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
from .advantage_repair_study import materialize_advantage_batch
from .policy_consistency_vnext import ACTOR_SEED, HORIZONS, TASK, _tree_delta
from .shared_probe_bank import evaluate_shared_probe_bank


SCHEMA = "trajectory-imf-reward-head-mtp-study-v1"
WORLD_MODEL_SEEDS = (211, 223)
ARMS = (
    "one_step_mse",
    "dreamer4_twohot_mtp",
    "counterfactual_advantage",
    "twohot_mtp_advantage",
)


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


def arm_uses_mtp(arm: str) -> bool:
    if arm not in ARMS:
        raise ValueError(f"unknown reward arm {arm!r}")
    return arm in ("dreamer4_twohot_mtp", "twohot_mtp_advantage")


def arm_uses_advantage(arm: str) -> bool:
    if arm not in ARMS:
        raise ValueError(f"unknown reward arm {arm!r}")
    return arm in ("counterfactual_advantage", "twohot_mtp_advantage")


def reward_config_for_arm(config: Any, arm: str, *, reward_bins: int) -> Any:
    """Return the only model-configuration difference permitted by an arm."""

    if arm_uses_mtp(arm):
        return replace(
            config,
            reward_loss="symexp_twohot",
            reward_prediction_horizon=8,
            reward_bins=reward_bins,
            reward_min=None,
            reward_max=None,
        )
    return replace(
        config,
        reward_loss="mse",
        reward_prediction_horizon=0,
        reward_bins=1,
    )


def objective_for_arm(arm: str):
    from imf_dreamer_jax import RewardHeadObjectiveConfig

    return RewardHeadObjectiveConfig(
        posterior_scale=1.0,
        corrupted_scale=0.5,
        generated_scale=0.5,
        context_corruption_max=0.1,
        generated_context_mix=0.25,
        advantage_scale=0.1 if arm_uses_advantage(arm) else 0.0,
        advantage_horizons=(1, 3, 5),
        advantage_magnitude_scale=0.25,
        advantage_ranking_scale=1.0,
        advantage_flat_scale=0.1,
    )


def build_manifest(
    baseline_root: str | Path,
    train_probe_root: str | Path,
    *,
    reward_updates: int = 1000,
    actor_updates: int = 3000,
    preparation_updates: int = 500,
    reward_bins: int = 255,
    evaluation_episodes: int = 5,
) -> dict[str, Any]:
    if min(reward_updates, actor_updates, preparation_updates, evaluation_episodes) <= 0:
        raise ValueError("update and evaluation counts must be positive")
    if reward_bins <= 1:
        raise ValueError("reward_bins must exceed one")
    baseline = Path(baseline_root).resolve(strict=True)
    train_root = Path(train_probe_root).resolve(strict=True)
    baseline_manifest = read_json(baseline / "manifest.json")
    baseline_report = read_json(baseline / "report.json")
    train_report = read_json(train_root / "report.json")
    if baseline_report.get("status") != "complete" or train_report.get("status") != "complete":
        raise ValueError("source studies must be complete")
    sources = []
    for seed in WORLD_MODEL_SEEDS:
        source = _source_row(baseline_manifest, seed)
        checkpoint = Path(source["world_models"]["trajectory_imf"]["checkpoint"])
        checkpoint_digest = source["world_models"]["trajectory_imf"]["checkpoint_sha256"]
        dataset = Path(source["dataset"])
        test_bank = baseline / "probe_bank" / f"seed-{seed}" / "probe_bank.npz"
        test_result = read_json(test_bank.with_name("result.json"))
        train_bank = train_root / "train_probe_bank" / f"seed-{seed}" / "probe_bank.npz"
        train_result = read_json(train_bank.with_name("result.json"))
        checks = (
            (checkpoint, checkpoint_digest, "world checkpoint"),
            (test_bank, test_result["probe_bank_file_sha256"], "held-out probe bank"),
            (train_bank, train_result["probe_bank_file_sha256"], "train probe bank"),
        )
        for path, digest, label in checks:
            if benchmark.file_sha256(path) != digest:
                raise ValueError(f"{label} digest mismatch")
        if benchmark.array_sha256(benchmark.load_npz(dataset)) != source["dataset_sha256"]:
            raise ValueError("dataset payload digest mismatch")
        sources.append(
            {
                "world_model_seed": seed,
                "dataset": str(dataset.resolve()),
                "dataset_sha256": source["dataset_sha256"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_digest,
                "test_probe_bank": str(test_bank.resolve()),
                "test_probe_bank_sha256": test_result["probe_bank_sha256"],
                "test_probe_bank_file_sha256": test_result["probe_bank_file_sha256"],
                "train_probe_bank": str(train_bank.resolve()),
                "train_probe_bank_sha256": train_result["probe_bank_sha256"],
                "train_probe_bank_file_sha256": train_result["probe_bank_file_sha256"],
            }
        )
    objectives = {arm: asdict(objective_for_arm(arm)) for arm in ARMS}
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "task": TASK,
        "world_model_seeds": list(WORLD_MODEL_SEEDS),
        "arms": list(ARMS),
        "factorial_design": {
            "reward_representation": ["scalar_mse", "symexp_twohot_mtp_0_to_8"],
            "counterfactual_advantage": [False, True],
        },
        "reward_updates": int(reward_updates),
        "actor_updates": int(actor_updates),
        "preparation_updates": int(preparation_updates),
        "evaluation_episodes": int(evaluation_episodes),
        "imagination_horizons": list(HORIZONS),
        "reward_bins": int(reward_bins),
        "trainable_world_subtrees": ["reward"],
        "objectives": objectives,
        "baseline_root": str(baseline),
        "baseline_manifest_sha256": baseline_manifest["manifest_sha256"],
        "baseline_report_file_sha256": benchmark.file_sha256(baseline / "report.json"),
        "train_probe_root": str(train_root),
        "train_probe_report_file_sha256": benchmark.file_sha256(train_root / "report.json"),
        "source_artifacts": sources,
        "frozen_shortcut_reference": {
            "actor_groups": [
                row for row in baseline_report["actor_groups"]
                if row["arm"] == "shortcut_forcing"
                and float(row["epistemic_penalty_scale"]) == 0.0
            ],
            "probe_groups": baseline_report["shared_probe_groups"]["shortcut_forcing"],
        },
        "interpretation": {
            "claim_eligible": False,
            "evidence_class": "exploratory_two_seed_reacher_reward_ablation",
            "state_model_frozen": True,
            "shortcut_is_frozen_external_reference": True,
            "fine_tuning_not_end_to_end_matched_compute": True,
        },
    }
    manifest = _json_native(manifest)
    return {**manifest, "manifest_sha256": benchmark.object_sha256(manifest)}


def write_manifest(
    baseline_root: str | Path,
    train_probe_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    expected = _json_native(build_manifest(baseline_root, train_probe_root, **settings))
    output = Path(output_root)
    path = output / "manifest.json"
    if path.is_file():
        existing = read_json(path)
        if existing != expected:
            raise ValueError("existing reward-head MTP manifest differs from frozen design")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, expected)
    return expected


def _advantage_schedule(seed: int, updates: int, batch_size: int, probes: int) -> np.ndarray:
    random = np.random.default_rng(
        benchmark.derive_seed("reward-head-mtp-probes", TASK, seed)
    )
    return random.integers(0, probes, size=(updates, batch_size), dtype=np.int32)


def _reward_metrics(details: Any, rms: Any) -> dict[str, float]:
    result = {
        "total": float(np.asarray(details.total)),
        "posterior": float(np.asarray(details.posterior)),
        "corrupted": float(np.asarray(details.corrupted)),
        "generated": float(np.asarray(details.generated)),
        "advantage_total": float(np.asarray(details.advantage.total)),
        "advantage_magnitude": float(np.asarray(details.advantage.magnitude)),
        "advantage_ranking": float(np.asarray(details.advantage.ranking)),
        "advantage_flat": float(np.asarray(details.advantage.flat)),
        "advantage_informative_pair_fraction": float(
            np.asarray(details.advantage.informative_pair_fraction)
        ),
        "advantage_running_rms": float(
            np.sqrt(float(np.asarray(rms.mean_square)) + 1e-6)
        ),
    }
    if not np.isfinite(np.asarray(list(result.values()))).all():
        raise FloatingPointError("reward training produced non-finite metrics")
    return result


def train_reward_cell(
    baseline_root: str | Path,
    train_probe_root: str | Path,
    output_root: str | Path,
    seed: int,
    arm: str,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import (
        init_running_rms,
        jit_train_reward_head,
        load_checkpoint,
        reconfigure_reward_head,
        save_checkpoint,
    )

    manifest = write_manifest(baseline_root, train_probe_root, output_root, **settings)
    if seed not in WORLD_MODEL_SEEDS or arm not in ARMS:
        raise ValueError("reward cell is outside the frozen design")
    source = _source_row(manifest, seed)
    directory = Path(output_root) / "reward" / f"seed-{seed}-{arm}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    source_checkpoint = Path(source["checkpoint"])
    state, source_config, _ = load_checkpoint(source_checkpoint)
    if source_config.prior != "imf" or not source_config.imf_trajectory_enabled:
        raise ValueError("source checkpoint is not trajectory iMF")
    config = reward_config_for_arm(
        source_config, arm, reward_bins=int(manifest["reward_bins"])
    )
    state = reconfigure_reward_head(
        state,
        source_config,
        config,
        benchmark.derive_jax_key("reward-head-mtp-init", seed),
        reset_output=True,
    )
    initial_state = state
    arrays = benchmark.load_npz(source["dataset"])
    train_bank = benchmark.load_npz(source["train_probe_bank"])
    if benchmark.array_sha256(train_bank) != source["train_probe_bank_sha256"]:
        raise ValueError("training probe bank payload digest mismatch")
    updates = int(manifest["reward_updates"])
    batch_size = 32
    sequence_length = 32
    base_schedule = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=benchmark.derive_seed("reward-head-mtp-base", seed),
        updates=updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    advantage_schedule = _advantage_schedule(
        seed, updates, batch_size, len(train_bank["anchors"])
    )
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = benchmark._write_npz_atomic(
        directory / "schedules.npz",
        {
            **{f"base_{name}": value for name, value in base_schedule.items()},
            "probe_indices": advantage_schedule,
        },
    )
    objective = objective_for_arm(arm)
    rms = init_running_rms()
    key = benchmark.derive_jax_key("reward-head-mtp-objective", seed)
    started = time.perf_counter()
    latest = None
    for update in range(updates):
        batch = benchmark._materialize_batch(
            arrays,
            base_schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        advantage_batch = materialize_advantage_batch(
            arrays,
            train_bank,
            advantage_schedule[update],
            burn_in=config.burn_in,
        )
        state, details, rms = jit_train_reward_head(
            state,
            batch,
            jax.random.fold_in(key, update),
            config,
            objective,
            rms,
            advantage_batch=advantage_batch,
        )
        latest = _reward_metrics(details, rms)
    if latest is None:
        raise RuntimeError("reward cell performed no updates")
    parameter_deltas = {
        name: _tree_delta(initial_state.params.world_model[name], state.params.world_model[name])
        for name in initial_state.params.world_model
    }
    first_moment_deltas = {
        name: _tree_delta(
            initial_state.model_optimizer.first_moment[name],
            state.model_optimizer.first_moment[name],
        )
        for name in initial_state.params.world_model
    }
    second_moment_deltas = {
        name: _tree_delta(
            initial_state.model_optimizer.second_moment[name],
            state.model_optimizer.second_moment[name],
        )
        for name in initial_state.params.world_model
    }
    leaked = {
        name: (parameter_deltas[name], first_moment_deltas[name], second_moment_deltas[name])
        for name in parameter_deltas
        if name != "reward"
        and (parameter_deltas[name] != 0.0 or first_moment_deltas[name] != 0.0 or second_moment_deltas[name] != 0.0)
    }
    if leaked or parameter_deltas["reward"] <= 0.0:
        raise RuntimeError(f"reward-only freeze contract failed: {leaked}")
    checkpoint = directory / "checkpoint.pkl"
    save_checkpoint(
        checkpoint,
        state,
        config,
        metadata={
            "stage": "reward_head_mtp",
            "arm": arm,
            "completed_updates": updates,
            "source_checkpoint_sha256": source["checkpoint_sha256"],
        },
    )
    if benchmark.file_sha256(source_checkpoint) != source["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint changed during reward adaptation")
    reward_parameters = int(sum(value.size for value in jax.tree_util.tree_leaves(state.params.world_model["reward"])))
    result = {
        "schema_version": SCHEMA,
        "stage": "reward",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "arm": arm,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "checkpoint_sha256": benchmark.file_sha256(checkpoint),
        "reward_parameters": reward_parameters,
        "reward_updates": updates,
        "objective": asdict(objective),
        "runtime_config": asdict(config),
        "world_model_subtree_parameter_deltas": parameter_deltas,
        "world_model_subtree_first_moment_deltas": first_moment_deltas,
        "world_model_subtree_second_moment_deltas": second_moment_deltas,
        "nonreward_frozen": True,
        "final_metrics": latest,
        "schedule_file_sha256": benchmark.file_sha256(schedule_path),
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def _heldout_schedule(
    arrays: Mapping[str, np.ndarray], seed: int, *, batch_size: int, sequence_length: int
) -> dict[str, np.ndarray]:
    ids = np.asarray(arrays["test_episode_ids"], dtype=np.int32)
    transitions = int(arrays["observations"].shape[1])
    random = np.random.default_rng(
        benchmark.derive_seed("reward-head-mtp-heldout", TASK, seed)
    )
    return {
        "episode_ids": random.choice(ids, size=(1, batch_size), replace=True).astype(np.int32),
        "starts": random.integers(
            0,
            transitions - sequence_length + 1,
            size=(1, batch_size),
            dtype=np.int32,
        ),
    }


def _masked_mse_by_offset(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> tuple[float, list[float]]:
    square = np.square(np.asarray(prediction, np.float64) - np.asarray(target, np.float64))
    weights = np.asarray(mask, np.float64)
    overall = float(np.sum(square * weights) / max(float(np.sum(weights)), 1.0))
    by_offset = [
        float(np.sum(square[..., index] * weights[..., index]) / max(float(np.sum(weights[..., index])), 1.0))
        for index in range(square.shape[-1])
    ]
    return overall, by_offset


def _mean_calibration_error(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, *, bins: int = 10
) -> float:
    selected = np.asarray(mask, dtype=bool)
    predicted = np.asarray(prediction, np.float64)[selected]
    observed = np.asarray(target, np.float64)[selected]
    if predicted.size == 0:
        raise ValueError("calibration mask is empty")
    lower = float(np.min(observed))
    upper = float(np.max(observed))
    if not upper > lower:
        return float(np.mean(np.abs(predicted - observed)))
    edges = np.linspace(lower, upper, bins + 1)
    assignments = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, bins - 1)
    error = 0.0
    for index in range(bins):
        active = assignments == index
        if np.any(active):
            error += float(np.mean(active)) * abs(
                float(np.mean(predicted[active])) - float(np.mean(observed[active]))
            )
    return error


def evaluate_reward_cell(
    baseline_root: str | Path,
    train_probe_root: str | Path,
    output_root: str | Path,
    seed: int,
    arm: str,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import load_checkpoint, reward_context_predictions

    manifest = write_manifest(baseline_root, train_probe_root, output_root, **settings)
    source = _source_row(manifest, seed)
    reward_directory = Path(output_root) / "reward" / f"seed-{seed}-{arm}"
    reward_result = read_json(reward_directory / "result.json")
    checkpoint = reward_directory / "checkpoint.pkl"
    if reward_result["checkpoint_sha256"] != benchmark.file_sha256(checkpoint):
        raise ValueError("reward checkpoint digest mismatch")
    directory = Path(output_root) / "evaluation" / f"seed-{seed}-{arm}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    state, config, _ = load_checkpoint(checkpoint)
    bank = benchmark.load_npz(source["test_probe_bank"])
    if benchmark.array_sha256(bank) != source["test_probe_bank_sha256"]:
        raise ValueError("held-out probe payload digest mismatch")
    probe = evaluate_shared_probe_bank(state.params.world_model, config, benchmark.load_npz(source["dataset"]), bank)
    arrays = benchmark.load_npz(source["dataset"])
    schedule = _heldout_schedule(arrays, seed, batch_size=64, sequence_length=32)
    batch = benchmark._materialize_batch(
        arrays,
        schedule,
        0,
        sequence_length=32,
        burn_in=config.burn_in,
    )
    predictions = reward_context_predictions(
        state.params.world_model,
        batch,
        benchmark.derive_jax_key("reward-head-mtp-heldout-context", seed),
        config,
        objective_for_arm(arm),
    )
    host = jax.device_get(predictions)
    context_metrics = {}
    for name in ("posterior", "corrupted", "generated"):
        values = np.asarray(getattr(host, name))
        overall, by_offset = _masked_mse_by_offset(values, host.targets, host.mask)
        context_metrics[name] = {
            "mse": overall,
            "mse_by_offset": by_offset,
            "offset_zero_mean_calibration_error": _mean_calibration_error(
                values[..., 0], np.asarray(host.targets)[..., 0], np.asarray(host.mask)[..., 0]
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
        "arm": arm,
        "world_model_checkpoint_sha256": reward_result["checkpoint_sha256"],
        "metrics_by_horizon": probe["metrics_by_horizon"],
        "reward_context_metrics": context_metrics,
        "raw_sha256": benchmark.file_sha256(raw_path),
    }
    write_json_atomic(result_path, result)
    return result


def train_actor_cell(
    baseline_root: str | Path,
    train_probe_root: str | Path,
    output_root: str | Path,
    seed: int,
    arm: str,
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

    manifest = write_manifest(baseline_root, train_probe_root, output_root, **settings)
    if seed not in WORLD_MODEL_SEEDS or arm not in ARMS or horizon not in HORIZONS:
        raise ValueError("actor cell is outside the frozen design")
    source = _source_row(manifest, seed)
    reward_directory = Path(output_root) / "reward" / f"seed-{seed}-{arm}"
    reward_result = read_json(reward_directory / "result.json")
    world_checkpoint = reward_directory / "checkpoint.pkl"
    if reward_result["checkpoint_sha256"] != benchmark.file_sha256(world_checkpoint):
        raise ValueError("actor source reward checkpoint digest mismatch")
    directory = Path(output_root) / "actor" / f"seed-{seed}-{arm}-h{horizon}"
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
        benchmark.derive_jax_key("reward-head-mtp-actor-init", TASK, seed, ACTOR_SEED, horizon),
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
        raise RuntimeError("actor cell performed no actor update")
    world_delta = _tree_delta(frozen_world, state.params.world_model)
    if world_delta != 0.0:
        raise RuntimeError("actor mutated the frozen world model")
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
            "stage": "reward_head_mtp_actor",
            "source_world_model_checkpoint_sha256": reward_result["checkpoint_sha256"],
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
        "arm": arm,
        "imagination_horizon": horizon,
        "world_model_checkpoint_sha256": reward_result["checkpoint_sha256"],
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


def finalize(
    baseline_root: str | Path,
    train_probe_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    manifest = write_manifest(baseline_root, train_probe_root, output_root, **settings)
    output = Path(output_root)
    reward_rows = [read_json(path) for path in sorted(output.glob("reward/*/result.json"))]
    evaluation_rows = [
        read_json(path) for path in sorted(output.glob("evaluation/*/result.json"))
    ]
    actor_rows = [read_json(path) for path in sorted(output.glob("actor/*/result.json"))]
    expected_reward = len(WORLD_MODEL_SEEDS) * len(ARMS)
    expected_actor = expected_reward * len(HORIZONS)
    probe_groups: dict[str, Any] = {}
    reward_groups: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in evaluation_rows if row["arm"] == arm]
        probe_groups[arm] = {
            horizon: {
                "pairwise_accuracy_mean": float(np.mean([
                    row["metrics_by_horizon"][horizon]["pairwise_accuracy"] for row in rows
                ])),
                "mean_state_spearman": float(np.mean([
                    row["metrics_by_horizon"][horizon]["mean_state_spearman"] for row in rows
                ])),
                "mean_simulator_regret": float(np.mean([
                    row["metrics_by_horizon"][horizon]["mean_simulator_regret"] for row in rows
                ])),
            }
            for horizon in ("1", "3", "5", "15")
        } if rows else {}
        reward_groups[arm] = {
            context: {
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
            for context in ("posterior", "corrupted", "generated")
        } if rows else {}
    actor_groups = []
    for horizon in HORIZONS:
        for arm in ARMS:
            rows = [
                row for row in actor_rows
                if row["arm"] == arm and row["imagination_horizon"] == horizon
            ]
            if rows:
                actor_groups.append(
                    {
                        "arm": arm,
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
    complete = (
        len(reward_rows) == expected_reward
        and len(evaluation_rows) == expected_reward
        and len(actor_rows) == expected_actor
    )
    report = {
        "schema_version": SCHEMA,
        "stage": "final",
        "status": "complete" if complete else "partial",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_reward_cells": len(reward_rows),
        "expected_reward_cells": expected_reward,
        "completed_evaluation_cells": len(evaluation_rows),
        "expected_evaluation_cells": expected_reward,
        "completed_actor_cells": len(actor_rows),
        "expected_actor_cells": expected_actor,
        "reward_groups": reward_groups,
        "probe_groups": probe_groups,
        "actor_groups": actor_groups,
        "frozen_shortcut_reference": manifest["frozen_shortcut_reference"],
        "aggregate_cell_wall_seconds": float(sum(
            float(row.get("wall_seconds", 0.0)) for row in reward_rows + actor_rows
        )),
        "interpretation": manifest["interpretation"],
    }
    write_json_atomic(output / "report.json", report)
    return report


__all__ = [
    "ARMS",
    "SCHEMA",
    "WORLD_MODEL_SEEDS",
    "arm_uses_advantage",
    "arm_uses_mtp",
    "build_manifest",
    "evaluate_reward_cell",
    "finalize",
    "objective_for_arm",
    "reward_config_for_arm",
    "train_actor_cell",
    "train_reward_cell",
    "write_manifest",
]
