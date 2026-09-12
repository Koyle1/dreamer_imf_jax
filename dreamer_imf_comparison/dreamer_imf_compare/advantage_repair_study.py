"""Small gated Reacher study for decision-aligned iMF fine-tuning.

The study is append-only and reads the completed frozen-world actor/probe study.
It fine-tunes only iMF arms on train-split simulator advantages, reevaluates the
same held-out probe bank, and applies the same repaired actor loop.  Endpoint,
exposure, and epistemic controls remain explicit variants rather than an
undocumented combined recipe.
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
from .policy_consistency_vnext import ACTOR_SEED, HORIZONS, TASK, _tree_delta
from .shared_probe_bank import (
    evaluate_shared_probe_bank,
    generate_shared_probe_bank,
    shared_probe_manifest,
    validate_shared_probe_bank,
)


SCHEMA = "trajectory-imf-advantage-repair-study-v1"
REPAIR_ARMS = ("trajectory_imf", "causal_trajectory_imf")
WORLD_MODEL_SEEDS = (211, 223)
VARIANTS = (
    "advantage_reward",
    "advantage",
    "advantage_endpoint",
    "advantage_exposure",
)


def _json_native(value: Any) -> Any:
    """Canonicalize tuples and scalar containers before persistence/comparison."""

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


def _objective(variant: str):
    from imf_dreamer_jax import PolicyConsistencyConfig

    if variant not in VARIANTS:
        raise ValueError(f"unknown repair variant {variant!r}")
    common = dict(
        advantage_consistency_scale=0.1,
        advantage_horizons=(1, 3, 5),
        advantage_magnitude_scale=0.25,
        advantage_ranking_scale=1.0,
        advantage_flat_scale=0.1,
        training_context_length=32,
    )
    if variant == "advantage_endpoint":
        common["endpoint_scale"] = 0.1
    elif variant == "advantage_exposure":
        common.update(
            exposure_meanflow_scale=0.1,
            endpoint_scale=0.1,
            generated_context_probability=0.5,
            context_corruption_max=0.1,
        )
    return PolicyConsistencyConfig(**common)


def build_manifest(
    baseline_root: str | Path,
    *,
    variant: str,
    world_updates: int = 1000,
    actor_updates: int = 3000,
    preparation_updates: int = 500,
    epistemic_scales: tuple[float, ...] = (0.0, 1.0),
) -> dict[str, Any]:
    if world_updates <= 0 or actor_updates <= 0 or preparation_updates <= 0:
        raise ValueError("update counts must be positive")
    if not epistemic_scales or any(value < 0.0 for value in epistemic_scales):
        raise ValueError("epistemic scales must be nonempty and nonnegative")
    objective = _objective(variant)
    baseline = Path(baseline_root).resolve(strict=True)
    baseline_manifest = read_json(baseline / "manifest.json")
    baseline_report = read_json(baseline / "report.json")
    if baseline_report.get("status") != "complete":
        raise ValueError("baseline policy-consistency study is incomplete")
    if (
        baseline_report.get("completed_actor_cells")
        != baseline_report.get("expected_actor_cells")
        or baseline_report.get("completed_probe_cells")
        != baseline_report.get("expected_probe_cells")
    ):
        raise ValueError("baseline result counts are incomplete")
    sources = []
    for seed in WORLD_MODEL_SEEDS:
        source = _source_row(baseline_manifest, seed)
        test_bank = baseline / "probe_bank" / f"seed-{seed}" / "probe_bank.npz"
        test_result = read_json(test_bank.with_name("result.json"))
        if test_result["probe_bank_file_sha256"] != benchmark.file_sha256(test_bank):
            raise ValueError("baseline test probe digest mismatch")
        worlds = {}
        for arm in REPAIR_ARMS:
            checkpoint = Path(source["world_models"][arm]["checkpoint"])
            digest = source["world_models"][arm]["checkpoint_sha256"]
            if benchmark.file_sha256(checkpoint) != digest:
                raise ValueError("source iMF checkpoint digest mismatch")
            worlds[arm] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": digest,
            }
        causal_checkpoint = Path(worlds["causal_trajectory_imf"]["checkpoint"])
        causal_dataset_directory = (
            causal_checkpoint.parents[2] / "dataset" / f"seed-{seed}"
        )
        causal_probe = causal_dataset_directory / "causal_probes.npz"
        causal_result_path = causal_dataset_directory / "result.json"
        causal_result = read_json(causal_result_path)
        if (
            causal_result.get("status") != "complete"
            or int(causal_result.get("world_model_seed", -1)) != seed
            or causal_result.get("dataset_sha256") != source["dataset_sha256"]
            or causal_result.get("probe_file_sha256")
            != benchmark.file_sha256(causal_probe)
        ):
            raise ValueError("causal iMF source probe evidence is invalid")
        sources.append(
            {
                "world_model_seed": seed,
                "dataset": source["dataset"],
                "dataset_sha256": source["dataset_sha256"],
                "test_probe_bank": str(test_bank),
                "test_probe_bank_sha256": test_result["probe_bank_sha256"],
                "causal_probe": str(causal_probe.resolve()),
                "causal_probe_sha256": causal_result["probe_sha256"],
                "causal_probe_file_sha256": causal_result["probe_file_sha256"],
                "causal_probe_result_file_sha256": benchmark.file_sha256(
                    causal_result_path
                ),
                "world_models": worlds,
            }
        )
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "baseline_root": str(baseline),
        "baseline_manifest_sha256": baseline_manifest["manifest_sha256"],
        "baseline_report_file_sha256": benchmark.file_sha256(baseline / "report.json"),
        "task": TASK,
        "world_model_seeds": list(WORLD_MODEL_SEEDS),
        "repair_arms": list(REPAIR_ARMS),
        "variant": variant,
        "trainable_world_subtrees": (
            ["reward"] if variant == "advantage_reward" else "all"
        ),
        "objective": asdict(objective),
        "world_updates": int(world_updates),
        "actor_updates": int(actor_updates),
        "preparation_updates": int(preparation_updates),
        "imagination_horizons": list(HORIZONS),
        "epistemic_scales": [float(value) for value in epistemic_scales],
        "train_probe": {
            "split": "train",
            "states": 32,
            "candidate_pool_size": 1024,
            "horizons": [1, 3, 5],
            "action_delta": 0.5,
            "intervention_steps": 5,
            "minimum_informative_fraction": 0.5,
            "selection": "simulator_return_range_at_horizon_5_before_model_evaluation",
        },
        "source_artifacts": sources,
        "interpretation": {
            "claim_eligible": False,
            "evidence_class": "exploratory_reacher_mechanism_intervention",
            "shortcut_is_frozen_external_reference": True,
            "fine_tuning_not_matched_compute_superiority_evidence": True,
        },
    }
    manifest = _json_native(manifest)
    return {**manifest, "manifest_sha256": benchmark.object_sha256(manifest)}


def write_manifest(
    baseline_root: str | Path,
    output_root: str | Path,
    **settings: Any,
) -> dict[str, Any]:
    expected = _json_native(build_manifest(baseline_root, **settings))
    output = Path(output_root)
    path = output / "manifest.json"
    if path.is_file():
        existing = read_json(path)
        if existing != expected:
            differing = sorted(
                key
                for key in set(existing) | set(expected)
                if existing.get(key) != expected.get(key)
            )
            details = {
                key: {"existing": existing.get(key), "expected": expected.get(key)}
                for key in differing
            }
            raise ValueError(
                "existing advantage-repair manifest differs: "
                + json.dumps(details, sort_keys=True)
            )
        return existing
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, expected)
    return expected


def prepare_train_probe_bank(
    baseline_root: str | Path,
    output_root: str | Path,
    seed: int,
    **settings: Any,
) -> dict[str, Any]:
    from imf_dreamer_jax import load_checkpoint

    manifest = write_manifest(baseline_root, output_root, **settings)
    source = _source_row(manifest, seed)
    directory = Path(output_root) / "train_probe_bank" / f"seed-{seed}"
    result_path = directory / "result.json"
    bank_path = directory / "probe_bank.npz"
    if result_path.is_file():
        result = read_json(result_path)
        if result["probe_bank_file_sha256"] != benchmark.file_sha256(bank_path):
            raise ValueError("retained train probe digest mismatch")
        return result
    dimensions = set()
    burn_ins = set()
    for arm in REPAIR_ARMS:
        _, config, _ = load_checkpoint(source["world_models"][arm]["checkpoint"])
        dimensions.add(config.stochastic_dim)
        burn_ins.add(config.burn_in)
    if len(dimensions) != 1 or len(burn_ins) != 1:
        raise ValueError("repair arms disagree on latent dimension or burn-in")
    arrays = benchmark.load_npz(source["dataset"])
    specification = manifest["train_probe"]
    bank = generate_shared_probe_bank(
        arrays,
        task=TASK,
        world_model_seed=seed,
        action_repeat=1,
        probes=specification["states"],
        horizons=specification["horizons"],
        action_delta=specification["action_delta"],
        intervention_steps=specification["intervention_steps"],
        selection_pool_multiplier=(
            specification["candidate_pool_size"] // specification["states"]
        ),
        minimum_informative_fraction=specification["minimum_informative_fraction"],
        stochastic_dim=dimensions.pop(),
        draws=8,
        episode_ids_key="train_episode_ids",
        minimum_anchor=burn_ins.pop(),
    )
    contract = shared_probe_manifest(
        bank,
        task=TASK,
        world_model_seed=seed,
        dataset_sha256=source["dataset_sha256"],
        action_delta=specification["action_delta"],
    )
    directory.mkdir(parents=True, exist_ok=True)
    benchmark._write_npz_atomic(bank_path, bank)
    result = {
        "schema_version": SCHEMA,
        "stage": "train_probe_bank",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        **contract,
        "dataset_split": "train",
        "probe_bank_file_sha256": benchmark.file_sha256(bank_path),
        "maximum_replay_error": float(bank["maximum_replay_error"][0]),
    }
    write_json_atomic(result_path, result)
    return result


def _advantage_schedule(seed: int, updates: int, batch_size: int, probes: int) -> np.ndarray:
    random = np.random.default_rng(
        benchmark.derive_seed("advantage-repair-probes", TASK, seed)
    )
    return random.integers(0, probes, size=(updates, batch_size), dtype=np.int32)


def materialize_advantage_batch(
    arrays: Mapping[str, np.ndarray],
    bank: Mapping[str, np.ndarray],
    probe_indices: np.ndarray,
    *,
    burn_in: int,
) -> dict[str, Any]:
    """Create one replay-grounded batch with one labeled decision per row."""

    import jax.numpy as jnp

    validate_shared_probe_bank(bank)
    indices = np.asarray(probe_indices, dtype=np.int32)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(bank["anchors"])):
        raise ValueError("probe_indices are invalid")
    sequence_length = burn_in + 1
    episodes = np.asarray(bank["episode_ids"])[indices]
    anchors = np.asarray(bank["anchors"])[indices]
    starts = anchors - burn_in
    if np.any(starts < 0):
        raise ValueError("train probes do not contain the requested burn-in")
    result: dict[str, np.ndarray] = {}
    for name in ("observations", "actions", "rewards", "continuations", "is_first"):
        result[name] = np.stack(
            [
                arrays[name][int(episode), int(start) : int(start) + sequence_length]
                for episode, start in zip(episodes, starts, strict=True)
            ]
        )
    batch_size = len(indices)
    candidates = bank["action_sequences"].shape[1]
    rollout_steps = bank["action_sequences"].shape[2]
    action_dim = bank["action_sequences"].shape[3]
    horizon_count = len(bank["horizons"])
    candidate_actions = np.zeros(
        (batch_size, sequence_length, candidates, rollout_steps, action_dim),
        np.float32,
    )
    target_returns = np.zeros(
        (batch_size, sequence_length, candidates, horizon_count), np.float32
    )
    advantage_mask = np.zeros(
        (batch_size, sequence_length, horizon_count), np.float32
    )
    candidate_actions[:, -1] = np.asarray(bank["action_sequences"])[indices]
    target_returns[:, -1] = np.asarray(bank["simulator_returns"])[indices]
    advantage_mask[:, -1] = np.asarray(bank["horizon_mask"])[indices]
    loss_mask = np.zeros((batch_size, sequence_length), np.float32)
    loss_mask[:, -1] = 1.0
    result.update(
        advantage_action_sequences=candidate_actions,
        advantage_target_returns=target_returns,
        advantage_mask=advantage_mask,
        loss_mask=loss_mask,
    )
    return {name: jnp.asarray(value) for name, value in result.items()}


def _repair_metrics(details: Any, rms: Any) -> dict[str, float]:
    result = {
        f"base_{name}": value
        for name, value in benchmark._metrics_dict(details.base).items()
    }
    for name in (
        "total",
        "magnitude",
        "ranking",
        "flat",
        "informative_pair_fraction",
    ):
        result[f"advantage_{name}"] = float(np.asarray(getattr(details.advantage, name)))
    result["exposure_meanflow"] = float(np.asarray(details.exposure_meanflow))
    result["endpoint"] = float(np.asarray(details.endpoint))
    result["total"] = float(np.asarray(details.total))
    result["advantage_running_rms"] = float(
        np.sqrt(float(np.asarray(rms.mean_square)) + 1e-6)
    )
    if not np.isfinite(np.asarray(list(result.values()))).all():
        raise FloatingPointError("repair training produced non-finite metrics")
    return result


def train_world_repair(
    baseline_root: str | Path,
    output_root: str | Path,
    seed: int,
    arm: str,
    **settings: Any,
) -> dict[str, Any]:
    import jax
    from imf_dreamer_jax import (
        init_running_rms,
        jit_train_policy_consistent_world_model,
        load_checkpoint,
        save_checkpoint,
    )

    if arm not in REPAIR_ARMS:
        raise ValueError("world repair is restricted to iMF arms")
    manifest = write_manifest(baseline_root, output_root, **settings)
    source = _source_row(manifest, seed)
    train_directory = Path(output_root) / "train_probe_bank" / f"seed-{seed}"
    train_result = read_json(train_directory / "result.json")
    bank = benchmark.load_npz(train_directory / "probe_bank.npz")
    if train_result["probe_bank_sha256"] != benchmark.array_sha256(bank):
        raise ValueError("world repair train bank digest mismatch")
    directory = Path(output_root) / "world_repair" / f"seed-{seed}-{arm}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    checkpoint_source = Path(source["world_models"][arm]["checkpoint"])
    source_digest = source["world_models"][arm]["checkpoint_sha256"]
    state, config, _ = load_checkpoint(checkpoint_source)
    if config.prior != "imf" or not config.imf_trajectory_enabled:
        raise ValueError("world repair source is not trajectory iMF")
    arrays = benchmark.load_npz(source["dataset"])
    if arm == "causal_trajectory_imf":
        causal_probe_path = Path(source["causal_probe"])
        if (
            benchmark.file_sha256(causal_probe_path)
            != source["causal_probe_file_sha256"]
        ):
            raise ValueError("causal probe file digest mismatch")
        causal_arrays = benchmark.load_npz(causal_probe_path)
        if benchmark.array_sha256(causal_arrays) != source["causal_probe_sha256"]:
            raise ValueError("causal probe payload digest mismatch")
        overlap = set(arrays) & set(causal_arrays)
        if overlap:
            raise ValueError(f"causal probe fields overlap base dataset: {sorted(overlap)}")
        arrays = {**arrays, **causal_arrays}
    updates = int(manifest["world_updates"])
    batch_size = 32
    sequence_length = 32
    base_schedule = benchmark._batch_schedule(
        arrays,
        task=TASK,
        world_model_seed=benchmark.derive_seed("advantage-repair-base", seed),
        updates=updates,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    probe_schedule = _advantage_schedule(seed, updates, batch_size, len(bank["anchors"]))
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = benchmark._write_npz_atomic(
        directory / "schedules.npz",
        {**{f"base_{name}": value for name, value in base_schedule.items()}, "probe_indices": probe_schedule},
    )
    objective = _objective(manifest["variant"])
    rms = init_running_rms()
    source_world = state.params.world_model
    trainable_subtrees = (
        ("reward",) if manifest["trainable_world_subtrees"] == ["reward"] else None
    )
    key = benchmark.derive_jax_key("advantage-repair-objective", seed)
    started = time.perf_counter()
    latest = None
    for update in range(updates):
        base_batch = benchmark._materialize_batch(
            arrays,
            base_schedule,
            update,
            sequence_length=sequence_length,
            burn_in=config.burn_in,
        )
        advantage_batch = materialize_advantage_batch(
            arrays, bank, probe_schedule[update], burn_in=config.burn_in
        )
        state, details, rms = jit_train_policy_consistent_world_model(
            state,
            base_batch,
            jax.random.fold_in(key, update),
            config,
            objective,
            rms,
            advantage_batch=advantage_batch,
            trainable_world_subtrees=trainable_subtrees,
        )
        latest = _repair_metrics(details, rms)
    if latest is None:
        raise RuntimeError("world repair performed no updates")
    subtree_deltas = {
        name: _tree_delta(source_world[name], state.params.world_model[name])
        for name in source_world
    }
    delta = max(subtree_deltas.values(), default=0.0)
    if not math.isfinite(delta) or delta <= 0.0:
        raise RuntimeError("world repair did not change finite world parameters")
    if trainable_subtrees == ("reward",):
        leaked = {
            name: value for name, value in subtree_deltas.items()
            if name != "reward" and value != 0.0
        }
        if leaked or subtree_deltas["reward"] <= 0.0:
            raise RuntimeError(f"reward-only repair violated its freeze contract: {leaked}")
    checkpoint = directory / "checkpoint.pkl"
    save_checkpoint(
        checkpoint,
        state,
        config,
        metadata={
            "stage": "advantage_world_repair",
            "source_checkpoint_sha256": source_digest,
            "completed_updates": updates,
            "variant": manifest["variant"],
            "objective": manifest["objective"],
        },
    )
    if benchmark.file_sha256(checkpoint_source) != source_digest:
        raise RuntimeError("source checkpoint changed during world repair")
    result = {
        "schema_version": SCHEMA,
        "stage": "world_repair",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "arm": arm,
        "variant": manifest["variant"],
        "source_checkpoint_sha256": source_digest,
        "checkpoint_sha256": benchmark.file_sha256(checkpoint),
        "train_probe_bank_sha256": train_result["probe_bank_sha256"],
        "updates": updates,
        "world_model_parameter_delta": delta,
        "world_model_subtree_parameter_deltas": subtree_deltas,
        "trainable_world_subtrees": manifest["trainable_world_subtrees"],
        "final_metrics": latest,
        "schedule_file_sha256": benchmark.file_sha256(schedule_path),
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


def evaluate_world_repair(
    baseline_root: str | Path,
    output_root: str | Path,
    seed: int,
    arm: str,
    **settings: Any,
) -> dict[str, Any]:
    from imf_dreamer_jax import load_checkpoint

    manifest = write_manifest(baseline_root, output_root, **settings)
    source = _source_row(manifest, seed)
    world_directory = Path(output_root) / "world_repair" / f"seed-{seed}-{arm}"
    world_result = read_json(world_directory / "result.json")
    checkpoint = world_directory / "checkpoint.pkl"
    if world_result["checkpoint_sha256"] != benchmark.file_sha256(checkpoint):
        raise ValueError("repaired checkpoint digest mismatch")
    bank = benchmark.load_npz(source["test_probe_bank"])
    if source["test_probe_bank_sha256"] != benchmark.array_sha256(bank):
        raise ValueError("held-out test bank digest mismatch")
    directory = Path(output_root) / "probe_evaluation" / f"seed-{seed}-{arm}"
    result_path = directory / "result.json"
    if result_path.is_file():
        return read_json(result_path)
    state, config, _ = load_checkpoint(checkpoint)
    arrays = benchmark.load_npz(source["dataset"])
    evaluation = evaluate_shared_probe_bank(state.params.world_model, config, arrays, bank)
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
        "variant": manifest["variant"],
        "world_model_checkpoint_sha256": world_result["checkpoint_sha256"],
        **evaluation,
        "raw_sha256": benchmark.file_sha256(raw_path),
    }
    write_json_atomic(result_path, result)
    return result


def finalize(baseline_root: str | Path, output_root: str | Path, **settings: Any) -> dict[str, Any]:
    manifest = write_manifest(baseline_root, output_root, **settings)
    output = Path(output_root)
    worlds = [read_json(path) for path in sorted(output.glob("world_repair/*/result.json"))]
    probes = [read_json(path) for path in sorted(output.glob("probe_evaluation/*/result.json"))]
    expected = len(WORLD_MODEL_SEEDS) * len(REPAIR_ARMS)
    groups = {}
    for arm in REPAIR_ARMS:
        rows = [row for row in probes if row["arm"] == arm]
        groups[arm] = {
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
    report = {
        "schema_version": SCHEMA,
        "stage": "final",
        "status": "complete" if len(worlds) == expected and len(probes) == expected else "partial",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "completed_world_cells": len(worlds),
        "expected_world_cells": expected,
        "completed_probe_cells": len(probes),
        "expected_probe_cells": expected,
        "probe_groups": groups,
        "interpretation": manifest["interpretation"],
    }
    write_json_atomic(output / "report.json", report)
    return report


__all__ = [
    "REPAIR_ARMS",
    "SCHEMA",
    "VARIANTS",
    "WORLD_MODEL_SEEDS",
    "build_manifest",
    "evaluate_world_repair",
    "finalize",
    "materialize_advantage_batch",
    "prepare_train_probe_bank",
    "train_world_repair",
    "write_manifest",
]
