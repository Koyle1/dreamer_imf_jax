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
from .shared_probe_bank import (
    evaluate_shared_probe_bank,
    relabel_shared_probe_bank_horizons,
    validate_shared_probe_bank,
)


SCHEMA = "trajectory-imf-action-reward-residual-study-v3"
SPARSE_SCHEMA = "trajectory-imf-action-reward-residual-study-v2"
CENTERED_SCHEMA = "trajectory-imf-action-reward-residual-study-v1"
WORLD_MODEL_SEEDS = mtp.WORLD_MODEL_SEEDS
DENSE_TRAINING_HORIZONS = tuple(range(1, 16))


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
    centered_root: str | Path,
    sparse_root: str | Path,
    residual_updates: int = 1000,
    actor_updates: int = 3000,
    preparation_updates: int = 500,
    evaluation_episodes: int = 5,
) -> dict[str, Any]:
    del output_root
    if min(residual_updates, actor_updates, preparation_updates, evaluation_episodes) <= 0:
        raise ValueError("update and evaluation counts must be positive")
    base = Path(mtp_root).resolve(strict=True)
    centered = Path(centered_root).resolve(strict=True)
    sparse = Path(sparse_root).resolve(strict=True)
    base_manifest = read_json(base / "manifest.json")
    base_report = read_json(base / "report.json")
    centered_manifest = read_json(centered / "manifest.json")
    centered_report = read_json(centered / "report.json")
    sparse_manifest = read_json(sparse / "manifest.json")
    sparse_report = read_json(sparse / "report.json")
    if base_report.get("status") != "complete":
        raise ValueError("source reward-head MTP study is incomplete")
    if base_manifest.get("manifest_sha256") != base_report.get("manifest_sha256"):
        raise ValueError("source reward-head MTP manifest/report mismatch")
    if (
        centered_manifest.get("schema_version") != CENTERED_SCHEMA
        or centered_report.get("schema_version") != CENTERED_SCHEMA
        or centered_report.get("status") != "complete"
        or centered_manifest.get("manifest_sha256")
        != centered_report.get("manifest_sha256")
        or centered_report.get("completed_residual_cells") != len(WORLD_MODEL_SEEDS)
        or centered_report.get("completed_evaluation_cells") != len(WORLD_MODEL_SEEDS)
        or centered_report.get("completed_actor_cells")
        != len(WORLD_MODEL_SEEDS) * len(HORIZONS)
    ):
        raise ValueError("centered residual reference is incomplete or incompatible")
    if (
        sparse_manifest.get("schema_version") != SPARSE_SCHEMA
        or sparse_report.get("schema_version") != SPARSE_SCHEMA
        or sparse_report.get("status") != "complete"
        or sparse_manifest.get("manifest_sha256")
        != sparse_report.get("manifest_sha256")
        or sparse_manifest.get("objective", {}).get("horizons") != [1, 3, 5]
        or sparse_report.get("completed_residual_cells") != len(WORLD_MODEL_SEEDS)
        or sparse_report.get("completed_evaluation_cells") != len(WORLD_MODEL_SEEDS)
        or sparse_report.get("completed_actor_cells")
        != len(WORLD_MODEL_SEEDS) * len(HORIZONS)
    ):
        raise ValueError("sparse residual reference is incomplete or incompatible")
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
        arrays = benchmark.load_npz(source["dataset"])
        if benchmark.array_sha256(arrays) != source["dataset_sha256"]:
            raise ValueError("dataset payload digest mismatch")
        train_bank = benchmark.load_npz(source["train_probe_bank"])
        validate_shared_probe_bank(train_bank)
        if (
            [int(value) for value in train_bank["horizons"]] != [1, 3, 5]
            or train_bank["action_sequences"].shape[2] != 5
            or any(
                int(anchor) + DENSE_TRAINING_HORIZONS[-1]
                > arrays["actions"].shape[1]
                or not np.all(
                    arrays["continuations"][
                        int(episode),
                        int(anchor) : int(anchor) + DENSE_TRAINING_HORIZONS[-1],
                    ]
                    > 0.0
                )
                for episode, anchor in zip(
                    train_bank["episode_ids"], train_bank["anchors"], strict=True
                )
            )
        ):
            raise ValueError("source train probe cannot be extended to horizon 15")
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

    objective = ActionRewardResidualConfig(horizons=DENSE_TRAINING_HORIZONS)
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
        "objective_identity": "uncentered_pseudo_huber_all_prefix_horizons_1_to_15",
        "normalization_basis": "centered_target_return_running_rms",
        "dense_probe_design": {
            "source_design_reused": True,
            "source_intervention_horizon": 5,
            "training_horizons": list(DENSE_TRAINING_HORIZONS),
            "simulator_replay_required": True,
            "legacy_horizon_agreement": [1, 3, 5],
            "tail_action_rule": "recorded_behavior_suffix_common_across_candidates",
            "noise_tail_rule": "deterministic_scoped_extension_after_exact_source_prefix",
            "action_repeat": 1,
            "discount": 0.99,
        },
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
        "centered_residual_root": str(centered),
        "centered_residual_manifest_sha256": centered_manifest["manifest_sha256"],
        "centered_residual_report_file_sha256": benchmark.file_sha256(
            centered / "report.json"
        ),
        "centered_residual_reference": {
            "reward_groups": centered_report["reward_groups"],
            "probe_groups": centered_report["probe_groups"],
            "actor_groups": centered_report["actor_groups"],
        },
        "sparse_residual_root": str(sparse),
        "sparse_residual_manifest_sha256": sparse_manifest["manifest_sha256"],
        "sparse_residual_report_file_sha256": benchmark.file_sha256(
            sparse / "report.json"
        ),
        "sparse_residual_reference": {
            "reward_groups": sparse_report["reward_groups"],
            "probe_groups": sparse_report["probe_groups"],
            "actor_groups": sparse_report["actor_groups"],
        },
        "interpretation": {
            "claim_eligible": False,
            "evidence_class": "exploratory_two_seed_reacher_dense_residual_return",
            "trajectory_model_frozen": True,
            "base_reward_head_frozen": True,
            "matched_actor_protocol": True,
            "single_loss_term": True,
            "dense_horizon_identification": True,
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
        "uncentered_return": float(np.asarray(details.uncentered_return)),
        "advantage_running_rms": float(
            np.sqrt(float(np.asarray(rms.mean_square)) + 1e-6)
        ),
    }
    if not np.isfinite(np.asarray(list(metrics.values()))).all():
        raise FloatingPointError("residual training produced non-finite metrics")
    return metrics


def dense_probe_legacy_max_error(
    source_bank: Mapping[str, np.ndarray],
    dense_bank: Mapping[str, np.ndarray],
) -> float:
    """Return the largest relabeling error at legacy supervised horizons."""

    validate_shared_probe_bank(source_bank)
    validate_shared_probe_bank(dense_bank)
    for name in ("episode_ids", "anchors"):
        if not np.array_equal(np.asarray(source_bank[name]), np.asarray(dense_bank[name])):
            raise ValueError(f"dense probe changed fixed design field {name!r}")
    source_horizon = int(np.asarray(source_bank["action_sequences"]).shape[2])
    if not np.array_equal(
        np.asarray(source_bank["action_sequences"]),
        np.asarray(dense_bank["action_sequences"])[:, :, :source_horizon],
    ):
        raise ValueError("dense probe changed fixed design field 'action_sequences'")
    if not np.array_equal(
        np.asarray(source_bank["model_noise"]),
        np.asarray(dense_bank["model_noise"])[:, :, :source_horizon],
    ):
        raise ValueError("dense probe changed fixed design field 'model_noise'")
    for name in (
        "candidate_pool_size",
        "selection_horizon",
        "selection_return_ranges",
        "action_delta",
        "intervention_steps",
    ):
        if not np.array_equal(np.asarray(source_bank[name]), np.asarray(dense_bank[name])):
            raise ValueError(f"dense probe changed fixed design field {name!r}")
    dense_actions = np.asarray(dense_bank["action_sequences"])
    if dense_actions.shape[2] > source_horizon and not np.all(
        dense_actions[:, 1:, source_horizon:]
        == dense_actions[:, :1, source_horizon:]
    ):
        raise ValueError("dense probe tail must be common across candidates")
    legacy_horizons = [int(value) for value in source_bank["horizons"]]
    dense_horizons = [int(value) for value in dense_bank["horizons"]]
    if tuple(dense_horizons) != DENSE_TRAINING_HORIZONS:
        raise ValueError("dense probe does not contain every horizon 1 through 15")
    if any(horizon not in dense_horizons for horizon in legacy_horizons):
        raise ValueError("dense probe omitted a legacy horizon")
    maximum_error = 0.0
    for source_index, horizon in enumerate(legacy_horizons):
        dense_index = dense_horizons.index(horizon)
        if not np.array_equal(
            np.asarray(source_bank["horizon_mask"])[:, source_index],
            np.asarray(dense_bank["horizon_mask"])[:, dense_index],
        ):
            raise ValueError(f"dense probe validity differs at horizon {horizon}")
        maximum_error = max(
            maximum_error,
            float(
                np.max(
                    np.abs(
                        np.asarray(source_bank["simulator_returns"])[
                            :, :, source_index
                        ]
                        - np.asarray(dense_bank["simulator_returns"])[
                            :, :, dense_index
                        ]
                    )
                )
            ),
        )
    return maximum_error


def prepare_dense_probe_cell(
    mtp_root: str | Path,
    output_root: str | Path,
    seed: int,
    **settings: Any,
) -> dict[str, Any]:
    """Relabel the frozen train probe design at every horizon 1 through 15."""

    manifest = write_manifest(mtp_root, output_root, **settings)
    if seed not in WORLD_MODEL_SEEDS:
        raise ValueError("dense probe cell is outside the frozen design")
    source = _source_row(manifest, seed)
    directory = Path(output_root) / "dense_probe" / f"seed-{seed}"
    result_path = directory / "result.json"
    bank_path = directory / "probe_bank.npz"
    if result_path.is_file():
        result = read_json(result_path)
        if benchmark.file_sha256(bank_path) != result["probe_bank_file_sha256"]:
            raise ValueError("existing dense probe artifact digest mismatch")
        return result
    arrays = benchmark.load_npz(source["dataset"])
    if benchmark.array_sha256(arrays) != source["dataset_sha256"]:
        raise ValueError("dense probe source dataset payload digest mismatch")
    source_bank = benchmark.load_npz(source["train_probe_bank"])
    if benchmark.array_sha256(source_bank) != source["train_probe_bank_sha256"]:
        raise ValueError("dense probe source design payload digest mismatch")
    started = time.perf_counter()
    dense_bank = relabel_shared_probe_bank_horizons(
        arrays,
        source_bank,
        task=TASK,
        world_model_seed=seed,
        action_repeat=int(manifest["dense_probe_design"]["action_repeat"]),
        horizons=DENSE_TRAINING_HORIZONS,
        discount=float(manifest["dense_probe_design"]["discount"]),
    )
    legacy_max_error = dense_probe_legacy_max_error(source_bank, dense_bank)
    if legacy_max_error > 1e-6:
        raise ValueError(
            "dense simulator labels disagree with the frozen legacy labels: "
            f"{legacy_max_error}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    benchmark._write_npz_atomic(bank_path, dense_bank)
    result = {
        "schema_version": SCHEMA,
        "stage": "dense_probe",
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "world_model_seed": seed,
        "source_probe_bank_sha256": source["train_probe_bank_sha256"],
        "source_probe_bank_file_sha256": source["train_probe_bank_file_sha256"],
        "probe_bank_sha256": benchmark.array_sha256(dense_bank),
        "probe_bank_file_sha256": benchmark.file_sha256(bank_path),
        "horizons": list(DENSE_TRAINING_HORIZONS),
        "legacy_horizon_max_abs_error": legacy_max_error,
        "maximum_replay_error": float(
            np.asarray(dense_bank["maximum_replay_error"])[0]
        ),
        "wall_seconds": time.perf_counter() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
    }
    write_json_atomic(result_path, result)
    return result


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
    dense_probe_directory = Path(output_root) / "dense_probe" / f"seed-{seed}"
    dense_probe_result = read_json(dense_probe_directory / "result.json")
    dense_probe_path = dense_probe_directory / "probe_bank.npz"
    if (
        dense_probe_result.get("status") != "complete"
        or dense_probe_result.get("manifest_sha256") != manifest["manifest_sha256"]
        or dense_probe_result.get("horizons") != list(DENSE_TRAINING_HORIZONS)
        or benchmark.file_sha256(dense_probe_path)
        != dense_probe_result.get("probe_bank_file_sha256")
    ):
        raise ValueError("dense training probe evidence is invalid")
    train_bank = benchmark.load_npz(dense_probe_path)
    if benchmark.array_sha256(train_bank) != dense_probe_result["probe_bank_sha256"]:
        raise ValueError("dense training probe payload digest mismatch")
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
        "dense_probe_bank_sha256": dense_probe_result["probe_bank_sha256"],
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
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
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
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": benchmark.runtime_fingerprint(),
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
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
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
    dense_probe_rows = [
        read_json(path) for path in sorted(output.glob("dense_probe/*/result.json"))
    ]
    residual_rows = [read_json(path) for path in sorted(output.glob("residual/*/result.json"))]
    evaluation_rows = [read_json(path) for path in sorted(output.glob("evaluation/*/result.json"))]
    actor_rows = [read_json(path) for path in sorted(output.glob("actor/*/result.json"))]
    complete = (
        len(dense_probe_rows) == len(WORLD_MODEL_SEEDS)
        and len(residual_rows) == len(WORLD_MODEL_SEEDS)
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
    centered = manifest["centered_residual_reference"]
    sparse = manifest["sparse_residual_reference"]
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
        centered_actor = next(
            item for item in centered["actor_groups"]
            if item["imagination_horizon"] == horizon
        )
        sparse_actor = next(
            item for item in sparse["actor_groups"]
            if item["imagination_horizon"] == horizon
        )
        actor_deltas[str(horizon)] = {
            "versus_base_mtp": row["mean_normalized_return"] - base_actor["mean_normalized_return"],
            "versus_shortcut": row["mean_normalized_return"] - shortcut_actor["mean_normalized_return"],
            "versus_centered_residual": (
                row["mean_normalized_return"]
                - centered_actor["mean_normalized_return"]
            ),
            "versus_sparse_residual": (
                row["mean_normalized_return"]
                - sparse_actor["mean_normalized_return"]
            ),
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
            "pairwise_accuracy_versus_centered_residual": (
                probe_groups[horizon]["pairwise_accuracy_mean"]
                - centered["probe_groups"][horizon]["pairwise_accuracy_mean"]
            ),
            "regret_versus_centered_residual": (
                probe_groups[horizon]["mean_simulator_regret"]
                - centered["probe_groups"][horizon]["mean_simulator_regret"]
            ),
            "pairwise_accuracy_versus_sparse_residual": (
                probe_groups[horizon]["pairwise_accuracy_mean"]
                - sparse["probe_groups"][horizon]["pairwise_accuracy_mean"]
            ),
            "regret_versus_sparse_residual": (
                probe_groups[horizon]["mean_simulator_regret"]
                - sparse["probe_groups"][horizon]["mean_simulator_regret"]
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
            "mse_versus_centered_residual": (
                reward_groups[context]["mean_mse"]
                - centered["reward_groups"][context]["mean_mse"]
            ),
            "calibration_versus_centered_residual": (
                reward_groups[context]["mean_offset_zero_calibration_error"]
                - centered["reward_groups"][context][
                    "mean_offset_zero_calibration_error"
                ]
            ),
            "mse_versus_sparse_residual": (
                reward_groups[context]["mean_mse"]
                - sparse["reward_groups"][context]["mean_mse"]
            ),
            "calibration_versus_sparse_residual": (
                reward_groups[context]["mean_offset_zero_calibration_error"]
                - sparse["reward_groups"][context][
                    "mean_offset_zero_calibration_error"
                ]
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
        "completed_dense_probe_cells": len(dense_probe_rows),
        "expected_dense_probe_cells": len(WORLD_MODEL_SEEDS),
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
        "centered_residual_reference": centered,
        "sparse_residual_reference": sparse,
        "aggregate_cell_wall_seconds": float(sum(
            float(row.get("wall_seconds", 0.0))
            for row in dense_probe_rows + residual_rows + actor_rows
        )),
        "interpretation": manifest["interpretation"],
    }
    write_json_atomic(output / "report.json", report)
    return report


__all__ = [
    "SCHEMA",
    "DENSE_TRAINING_HORIZONS",
    "SPARSE_SCHEMA",
    "WORLD_MODEL_SEEDS",
    "build_manifest",
    "dense_probe_legacy_max_error",
    "evaluate_residual_cell",
    "finalize",
    "prepare_dense_probe_cell",
    "record_preflight",
    "train_actor_cell",
    "train_residual_cell",
    "write_manifest",
]
