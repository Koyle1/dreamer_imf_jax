"""Paper-faithful ReBRAC plus ITPO controller study on trajectory iMF Reacher.

The controller follows Algorithm 1 of the ITPO/FlowMPC paper.  The unavoidable
domain adaptation is explicit: the published method uses fully observed D4RL
states and a policy-tilted state-space MeanFlow model, whereas this diagnostic
uses decoded states from the already-trained trajectory-iMF RSSM.  No learned
actor from an earlier study is reused.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from .artifacts import read_json, write_json_atomic
from .dmc import DMCAdapter
from . import matched_objective_benchmark as benchmark


SCHEMA = "trajectory-imf-flowmpc-actor-study-v1"
REBRAC_RESULT_SCHEMA = "trajectory-imf-flowmpc-rebrac-cell-v1"
TUNING_RESULT_SCHEMA = "trajectory-imf-flowmpc-tuning-v1"
EVALUATION_RESULT_SCHEMA = "trajectory-imf-flowmpc-evaluation-cell-v1"
MARKER_SCHEMA = "trajectory-imf-flowmpc-marker-v1"
REPORT_SCHEMA = "trajectory-imf-flowmpc-report-v1"
CHECKPOINT_VERSION = 1
TASK = "dmc_reacher_easy"
EXPECTED_REBRAC_CELLS = 6
EXPECTED_EVALUATION_CELLS = 6
PAPER_URL = "https://arxiv.org/abs/2603.22430"
REBRAC_URL = (
    "https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/rebrac.py"
)

REBRAC_UPDATES = 1_000_000
REBRAC_CHUNK_UPDATES = 10_000
REBRAC_CHECKPOINT_EVERY = 100_000
TUNING_HORIZON = 5
TUNING_INNER_STEPS = 1
TUNING_PARTICLES = 4096
TUNING_STEP_SIZES = (5e-7, 5e-6, 5e-5, 5e-4)
TUNING_EPISODES = 1
EVALUATION_EPISODES = 5
MAXIMUM_ENVIRONMENT_STEPS = 1000


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
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _load_pickle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ReBRAC checkpoint must contain a dictionary")
    return payload


def _reward_source_rows(reward_root: Path) -> list[dict[str, Any]]:
    """Authenticate the completed reward study without executing old source."""

    manifest_path = reward_root / "manifest.json"
    report_path = reward_root / "report.json"
    manifest = read_json(manifest_path)
    report = read_json(report_path)
    if (
        manifest.get("schema_version")
        != "trajectory-imf-itpo-state-action-reward-study-v2"
        or manifest.get("status") != "frozen_before_execution"
        or manifest.get("manifest_sha256")
        != _unsigned_digest(manifest, "manifest_sha256")
        or report.get("schema_version")
        != "trajectory-imf-itpo-state-action-reward-report-v2"
        or report.get("status") != "complete"
        or report.get("source_commit") != manifest.get("source_commit")
        or report.get("manifest_sha256") != manifest.get("manifest_sha256")
        or report.get("completed_reward_cells") != 3
        or report.get("strict_actor_replay_markers") != 6
        or report.get("report_sha256") != _unsigned_digest(report, "report_sha256")
    ):
        raise ValueError("completed reward-head study is not authenticated")
    sources = {
        int(row["world_model_seed"]): row for row in manifest["source_artifacts"]
    }
    rows: list[dict[str, Any]] = []
    for cell in manifest["reward_cells"]:
        seed = int(cell["world_model_seed"])
        source = sources.get(seed)
        if source is None:
            raise ValueError("reward cell has no source artifact")
        result_path = reward_root / str(cell["result_path"])
        checkpoint_path = reward_root / str(cell["checkpoint_path"])
        marker_path = reward_root / str(cell["marker_path"])
        result = read_json(result_path)
        marker = read_json(marker_path)
        if (
            result.get("schema_version")
            != "trajectory-imf-itpo-state-action-reward-cell-v2"
            or result.get("status") != "complete"
            or result.get("manifest_sha256") != manifest["manifest_sha256"]
            or result.get("cell_id") != cell["cell_id"]
            or result.get("source_world_model_parameter_delta") != 0.0
            or result.get("trainable_subtrees") != ["reward_transition"]
            or result.get("reward_head_family") != "itpo_state_action_mlp"
            or result.get("checkpoint_sha256")
            != benchmark.file_sha256(checkpoint_path)
            or marker.get("schema_version")
            != "trajectory-imf-itpo-state-action-reward-marker-v2"
            or marker.get("status") != "verified"
            or marker.get("stage") != "reward"
            or marker.get("cell_id") != cell["cell_id"]
            or marker.get("result_file_sha256")
            != benchmark.file_sha256(result_path)
            or marker.get("checkpoint_sha256") != result["checkpoint_sha256"]
            or marker.get("marker_sha256")
            != _unsigned_digest(marker, "marker_sha256")
        ):
            raise ValueError("reward checkpoint or marker is invalid")
        for path_key, digest_key in (
            ("world_model_checkpoint", "world_model_checkpoint_sha256"),
            ("dataset", "dataset_file_sha256"),
        ):
            if benchmark.file_sha256(source[path_key]) != source[digest_key]:
                raise ValueError(f"reward source {path_key} changed")
        rows.append(
            {
                "world_model_seed": seed,
                "reward_result": str(result_path.resolve()),
                "reward_result_sha256": benchmark.file_sha256(result_path),
                "reward_checkpoint": str(checkpoint_path.resolve()),
                "reward_checkpoint_sha256": result["checkpoint_sha256"],
                "reward_marker": str(marker_path.resolve()),
                "reward_marker_sha256": benchmark.file_sha256(marker_path),
                "dataset": str(Path(source["dataset"]).resolve()),
                "dataset_file_sha256": source["dataset_file_sha256"],
                "dataset_sha256": source["dataset_sha256"],
                "world_model_checkpoint": str(
                    Path(source["world_model_checkpoint"]).resolve()
                ),
                "world_model_checkpoint_sha256": source[
                    "world_model_checkpoint_sha256"
                ],
                "world_model_parameter_sha256": source[
                    "world_model_parameter_sha256"
                ],
                "runtime_config": source["runtime_config"],
                "runtime_config_sha256": source["runtime_config_sha256"],
            }
        )
    if len(rows) != 3 or len({row["world_model_seed"] for row in rows}) != 3:
        raise ValueError("reward study must provide exactly three world seeds")
    return rows


def _paper_rebrac_config(state_dim: int, action_dim: int) -> dict[str, Any]:
    from imf_dreamer_jax import ReBRACConfig

    return asdict(ReBRACConfig(state_dim=state_dim, action_dim=action_dim))


def _evaluation_seeds(world_seed: int, actor_seed: int) -> list[int]:
    return [
        benchmark.derive_seed(
            "flowmpc-evaluation", TASK, world_seed, actor_seed, episode
        )
        for episode in range(EVALUATION_EPISODES)
    ]


def build_manifest(
    reward_root: str | Path,
    *,
    rebrac_updates: int = REBRAC_UPDATES,
) -> dict[str, Any]:
    """Freeze the new actor study and all immutable reward dependencies."""

    if rebrac_updates <= 0:
        raise ValueError("ReBRAC updates must be positive")
    root = Path(reward_root).resolve(strict=True)
    reward_sources = _reward_source_rows(root)
    actor_seeds = [311, 313]
    world_seeds = [int(row["world_model_seed"]) for row in reward_sources]
    state_dim = int(np.prod(reward_sources[0]["runtime_config"]["observation_shape"]))
    action_dim = int(reward_sources[0]["runtime_config"]["action_dim"])
    rebrac_config = _paper_rebrac_config(state_dim, action_dim)
    rebrac_cells: list[dict[str, Any]] = []
    evaluation_cells: list[dict[str, Any]] = []
    for world_seed in world_seeds:
        for actor_seed in actor_seeds:
            cell_id = f"rebrac-{world_seed}-{actor_seed}"
            rebrac_cells.append(
                {
                    "index": len(rebrac_cells),
                    "cell_id": cell_id,
                    "world_model_seed": world_seed,
                    "actor_seed": actor_seed,
                    "result_path": f"rebrac/{cell_id}/result.json",
                    "checkpoint_path": f"rebrac/{cell_id}/checkpoint.pkl",
                    "marker_path": f"verified/{cell_id}.json",
                }
            )
            evaluation_id = f"flowmpc-eval-{world_seed}-{actor_seed}"
            evaluation_cells.append(
                {
                    "index": len(evaluation_cells),
                    "cell_id": evaluation_id,
                    "world_model_seed": world_seed,
                    "actor_seed": actor_seed,
                    "rebrac_cell_id": cell_id,
                    "evaluation_seeds": _evaluation_seeds(world_seed, actor_seed),
                    "result_path": f"evaluation/{evaluation_id}/result.json",
                    "trace_path": f"evaluation/{evaluation_id}/traces.npz",
                    "marker_path": f"verified/{evaluation_id}.json",
                }
            )
    tuning_world_seed = world_seeds[0]
    tuning_actor_seed = actor_seeds[0]
    tuning_seed = benchmark.derive_seed("flowmpc-tuning", TASK)
    if any(
        tuning_seed in cell["evaluation_seeds"] for cell in evaluation_cells
    ):
        raise RuntimeError("tuning and evaluation environment seeds overlap")
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "frozen_before_execution",
        "source_commit": _git_commit(),
        "task": TASK,
        "claim_eligible": False,
        "evidence_class": "exploratory_reacher_controller_diagnostic",
        "old_learned_actors_reused": False,
        "reward_root": str(root),
        "reward_manifest_file_sha256": benchmark.file_sha256(root / "manifest.json"),
        "reward_report_file_sha256": benchmark.file_sha256(root / "report.json"),
        "reward_sources": reward_sources,
        "world_model_seeds": world_seeds,
        "actor_seeds": actor_seeds,
        "rebrac_reference": REBRAC_URL,
        "flowmpc_reference": PAPER_URL,
        "rebrac_config": rebrac_config,
        "rebrac_updates": int(rebrac_updates),
        "rebrac_chunk_updates": REBRAC_CHUNK_UPDATES,
        "rebrac_checkpoint_every": REBRAC_CHECKPOINT_EVERY,
        "rebrac_state_normalization": False,
        "rebrac_reward_normalization": False,
        "rebrac_timeout_transitions_skipped": True,
        "rebrac_cells": rebrac_cells,
        "inference_tuning": {
            "world_model_seed": tuning_world_seed,
            "actor_seed": tuning_actor_seed,
            "environment_seed": tuning_seed,
            "episodes_per_candidate": TUNING_EPISODES,
            "horizon": TUNING_HORIZON,
            "inner_steps": TUNING_INNER_STEPS,
            "particles": TUNING_PARTICLES,
            "step_sizes": list(TUNING_STEP_SIZES),
            "selection_metric": "raw_episode_return",
            "tie_break": "smallest_step_size",
            "result_path": "tuning/result.json",
            "trace_path": "tuning/traces.npz",
            "marker_path": "verified/tuning.json",
        },
        "evaluation_episodes": EVALUATION_EPISODES,
        "maximum_environment_steps": MAXIMUM_ENVIRONMENT_STEPS,
        "evaluation_cells": evaluation_cells,
        "primary_comparison": "flowmpc_minus_same_rebrac_zero_shot",
        "paper_matches": {
            "offline_policy": "released ReBRAC update",
            "controller_objective": "discounted_stage_reward_plus_terminal_min_twin_q",
            "policy_update": "persistent_plain_gradient_ascent_at_every_real_state",
            "noise": "resampled_per_real_state_and_fixed_across_inner_steps",
            "tuning_seed": "one_environment_seed_disjoint_from_evaluation",
        },
        "declared_deviations": [
            "DMC Reacher rather than D4RL Gym-MuJoCo",
            "50k random/smooth offline transitions rather than a D4RL dataset",
            "latent recurrent trajectory-iMF dynamics decoded to observation space rather than the paper's fully observed state-space MeanFlow model",
            "existing trajectory-iMF dynamics were not trained with FlowMPC policy-tilted weighting",
            "the paper has no Reacher regularization coefficients; released ReBRAC defaults beta_actor=beta_critic=1 are used",
            "H=5, E=1, and M=4096 are fixed to common paper settings while only the paper's four non-Hopper step sizes are tuned",
            "five rather than ten evaluation episodes are used to match the existing Reacher pilot",
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
        or manifest.get("old_learned_actors_reused") is not False
        or manifest.get("primary_comparison")
        != "flowmpc_minus_same_rebrac_zero_shot"
        or len(manifest.get("rebrac_cells", ())) != EXPECTED_REBRAC_CELLS
        or len(manifest.get("evaluation_cells", ()))
        != EXPECTED_EVALUATION_CELLS
        or manifest.get("manifest_sha256")
        != _unsigned_digest(manifest, "manifest_sha256")
    ):
        raise ValueError("FlowMPC actor manifest identity is invalid")
    config = manifest.get("rebrac_config", {})
    expected_config = _paper_rebrac_config(
        int(config.get("state_dim", -1)), int(config.get("action_dim", -1))
    )
    if config != expected_config:
        raise ValueError("ReBRAC config differs from the frozen paper defaults")
    tuning = manifest.get("inference_tuning", {})
    if (
        int(tuning.get("horizon", -1)) != TUNING_HORIZON
        or int(tuning.get("inner_steps", -1)) != TUNING_INNER_STEPS
        or int(tuning.get("particles", -1)) != TUNING_PARTICLES
        or tuple(tuning.get("step_sizes", ())) != TUNING_STEP_SIZES
        or int(tuning.get("episodes_per_candidate", -1)) != TUNING_EPISODES
    ):
        raise ValueError("FlowMPC inference tuning grid differs")
    evaluation_seeds = {
        int(seed)
        for cell in manifest["evaluation_cells"]
        for seed in cell["evaluation_seeds"]
    }
    if int(tuning["environment_seed"]) in evaluation_seeds:
        raise ValueError("FlowMPC tuning seed leaked into evaluation")
    reward_root = Path(str(manifest["reward_root"]))
    if (
        benchmark.file_sha256(reward_root / "manifest.json")
        != manifest["reward_manifest_file_sha256"]
        or benchmark.file_sha256(reward_root / "report.json")
        != manifest["reward_report_file_sha256"]
    ):
        raise ValueError("reward study root changed")
    for source in manifest["reward_sources"]:
        for path_key, digest_key in (
            ("reward_result", "reward_result_sha256"),
            ("reward_checkpoint", "reward_checkpoint_sha256"),
            ("reward_marker", "reward_marker_sha256"),
            ("dataset", "dataset_file_sha256"),
            ("world_model_checkpoint", "world_model_checkpoint_sha256"),
        ):
            if benchmark.file_sha256(source[path_key]) != source[digest_key]:
                raise ValueError(f"immutable source changed: {path_key}")


def write_manifest(
    reward_root: str | Path,
    output_root: str | Path,
    *,
    rebrac_updates: int = REBRAC_UPDATES,
) -> dict[str, Any]:
    path = Path(output_root) / "manifest.json"
    if path.is_file():
        manifest = read_json(path)
        validate_manifest(manifest)
        return manifest
    manifest = build_manifest(reward_root, rebrac_updates=rebrac_updates)
    write_json_atomic(path, manifest)
    return manifest


def _cell(manifest: Mapping[str, Any], stage: str, index: int) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest[f"{stage}_cells"]
        if int(row["index"]) == int(index)
    ]
    if len(rows) != 1:
        raise ValueError(f"{stage} cell is absent or duplicated")
    return rows[0]


def _source(manifest: Mapping[str, Any], world_seed: int) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest["reward_sources"]
        if int(row["world_model_seed"]) == int(world_seed)
    ]
    if len(rows) != 1:
        raise ValueError("reward source is absent or duplicated")
    return rows[0]


def _dreamer_config(source: Mapping[str, Any]) -> Any:
    from imf_dreamer_jax import DreamerConfig

    values = dict(source["runtime_config"])
    values["observation_shape"] = tuple(values["observation_shape"])
    values["overshooting_distances"] = tuple(values["overshooting_distances"])
    config = DreamerConfig(**values)
    if benchmark.object_sha256(asdict(config)) != source["runtime_config_sha256"]:
        raise ValueError("reward source runtime config digest differs")
    return config


def _build_rebrac_dataset(arrays: Mapping[str, np.ndarray]) -> Any:
    """Convert shifted episodic replay to the released ReBRAC tuple layout."""

    import jax.numpy as jnp
    from imf_dreamer_jax import ReBRACDataset

    train_ids = np.asarray(arrays["train_episode_ids"], dtype=np.int64)
    observations = np.asarray(arrays["observations"][train_ids], dtype=np.float32)
    actions = np.asarray(arrays["actions"][train_ids], dtype=np.float32)
    rewards = np.asarray(arrays["rewards"][train_ids], dtype=np.float32)
    continuations = np.asarray(arrays["continuations"][train_ids], dtype=np.float32)
    # Entry i (i>=1) stores action/reward for obs[i-1] -> obs[i].  Exclude
    # the native final timeout just as qlearning_dataset(..., terminate_on_end=False)
    # and retain genuine terminal transitions while removing padded successors.
    valid = continuations[:, :-2] > 0.0
    states = observations[:, :-2][valid].reshape((-1, observations.shape[-1]))
    next_states = observations[:, 1:-1][valid].reshape((-1, observations.shape[-1]))
    selected_actions = actions[:, 1:-1][valid]
    next_actions = actions[:, 2:][valid]
    selected_rewards = rewards[:, 1:-1][valid]
    dones = 1.0 - continuations[:, 1:-1][valid]
    dataset = ReBRACDataset(
        jnp.asarray(states),
        jnp.asarray(selected_actions),
        jnp.asarray(selected_rewards),
        jnp.asarray(next_states),
        jnp.asarray(next_actions),
        jnp.asarray(dones),
    )
    if dataset.states.shape[0] <= 0:
        raise ValueError("ReBRAC replay has no valid transitions")
    return dataset


def _rebrac_config(manifest: Mapping[str, Any]) -> Any:
    from imf_dreamer_jax import ReBRACConfig

    return ReBRACConfig(**manifest["rebrac_config"])


def _save_rebrac_checkpoint(
    path: str | Path,
    state: Any,
    config: Any,
    metadata: Mapping[str, Any],
) -> Path:
    import jax

    return _write_pickle_atomic(
        path,
        {
            "version": CHECKPOINT_VERSION,
            "config": asdict(config),
            "state": jax.device_get(state),
            "metadata": dict(metadata),
        },
    )


def _load_rebrac_checkpoint(path: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import ReBRACConfig

    payload = _load_pickle(path)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported ReBRAC checkpoint version")
    config = ReBRACConfig(**payload["config"])
    state = jax.tree_util.tree_map(jnp.asarray, payload["state"])
    return state, config, dict(payload.get("metadata", {}))


def train_rebrac_cell(
    reward_root: str | Path,
    output_root: str | Path,
    index: int,
    *,
    rebrac_updates: int = REBRAC_UPDATES,
) -> dict[str, Any]:
    """Train one fresh released-ReBRAC actor and terminal twin critic."""

    import jax
    from imf_dreamer_jax import init_rebrac_state, jit_train_rebrac_chunk

    manifest = write_manifest(
        reward_root, output_root, rebrac_updates=rebrac_updates
    )
    cell = _cell(manifest, "rebrac", index)
    result_path = Path(output_root) / str(cell["result_path"])
    if result_path.is_file():
        return read_json(result_path)
    source = _source(manifest, int(cell["world_model_seed"]))
    arrays = benchmark.load_npz(source["dataset"])
    if benchmark.array_sha256(arrays) != source["dataset_sha256"]:
        raise ValueError("loaded ReBRAC dataset payload differs")
    dataset = _build_rebrac_dataset(arrays)
    config = _rebrac_config(manifest)
    if (
        dataset.states.shape[-1] != config.state_dim
        or dataset.actions.shape[-1] != config.action_dim
    ):
        raise ValueError("ReBRAC replay dimensions differ from config")
    checkpoint_path = Path(output_root) / str(cell["checkpoint_path"])
    training_key = benchmark.derive_jax_key(
        "flowmpc-rebrac-training",
        TASK,
        int(cell["world_model_seed"]),
        int(cell["actor_seed"]),
    )
    state = init_rebrac_state(
        benchmark.derive_jax_key(
            "flowmpc-rebrac-init",
            TASK,
            int(cell["world_model_seed"]),
            int(cell["actor_seed"]),
        ),
        config,
    )
    initial_actor_digest = benchmark._tree_digest(state.actor)
    initial_critic_digest = benchmark._tree_digest(state.critics)
    completed = 0
    accumulated_wall = 0.0
    latest: dict[str, float] | None = None
    if checkpoint_path.is_file():
        state, stored_config, metadata = _load_rebrac_checkpoint(checkpoint_path)
        if stored_config != config or metadata.get("cell_id") != cell["cell_id"]:
            raise ValueError("partial ReBRAC checkpoint identity differs")
        completed = int(metadata.get("completed_updates", -1))
        accumulated_wall = float(metadata.get("wall_seconds", 0.0))
        latest = metadata.get("latest_metrics")
        if (
            metadata.get("initial_actor_parameter_sha256")
            != initial_actor_digest
            or metadata.get("initial_critic_parameter_sha256")
            != initial_critic_digest
        ):
            raise ValueError("partial ReBRAC initialization differs")
    total = int(manifest["rebrac_updates"])
    chunk = int(manifest["rebrac_chunk_updates"])
    checkpoint_every = int(manifest["rebrac_checkpoint_every"])
    if not 0 <= completed <= total:
        raise ValueError("partial ReBRAC update counter is invalid")
    started = time.perf_counter()

    def save_progress() -> None:
        _save_rebrac_checkpoint(
            checkpoint_path,
            state,
            config,
            {
                "stage": "flowmpc_rebrac",
                "cell_id": cell["cell_id"],
                "manifest_sha256": manifest["manifest_sha256"],
                "completed_updates": completed,
                "training_key": np.asarray(jax.device_get(training_key)).tolist(),
                "initial_actor_parameter_sha256": initial_actor_digest,
                "initial_critic_parameter_sha256": initial_critic_digest,
                "latest_metrics": latest,
                "wall_seconds": accumulated_wall + time.perf_counter() - started,
            },
        )

    while completed < total:
        updates = min(chunk, total - completed)
        state, metrics = jit_train_rebrac_chunk(
            state, dataset, training_key, updates=updates, config=config
        )
        # Synchronize before telemetry and checkpoint timing.
        latest = {
            name: float(np.asarray(jax.device_get(value)))
            for name, value in metrics._asdict().items()
        }
        completed += updates
        if (
            completed % checkpoint_every == 0
            or completed == total
            or updates != chunk
        ):
            save_progress()
    if latest is None or not _finite_tree(latest):
        raise FloatingPointError("ReBRAC training produced no finite telemetry")
    final_actor_digest = benchmark._tree_digest(state.actor)
    final_critic_digest = benchmark._tree_digest(state.critics)
    if (
        final_actor_digest == initial_actor_digest
        or final_critic_digest == initial_critic_digest
    ):
        raise RuntimeError("ReBRAC trainable parameters did not change")
    total_wall = accumulated_wall + time.perf_counter() - started
    save_progress()
    result = {
        "schema_version": REBRAC_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
        "updates": total,
        "dataset_transitions": int(dataset.states.shape[0]),
        "dataset_file_sha256": source["dataset_file_sha256"],
        "dataset_sha256": source["dataset_sha256"],
        "rebrac_config": asdict(config),
        "initial_actor_parameter_sha256": initial_actor_digest,
        "initial_critic_parameter_sha256": initial_critic_digest,
        "final_actor_parameter_sha256": final_actor_digest,
        "final_critic_parameter_sha256": final_critic_digest,
        "latest_metrics": latest,
        "checkpoint_sha256": benchmark.file_sha256(checkpoint_path),
        "training_rng": "fold_in(base_key, optimizer_step)",
        "state_normalized": False,
        "reward_normalized": False,
        "timeout_transitions_skipped": True,
        "wall_seconds": total_wall,
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if not _finite_tree(result):
        raise FloatingPointError("ReBRAC result contains non-finite values")
    write_json_atomic(result_path, result)
    return result


def verify_rebrac_cell(output_root: str | Path, index: int) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell(manifest, "rebrac", index)
    result_path = root / str(cell["result_path"])
    checkpoint_path = root / str(cell["checkpoint_path"])
    result = read_json(result_path)
    identity = {
        "schema_version": REBRAC_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
    }
    if any(result.get(name) != value for name, value in identity.items()):
        raise ValueError("ReBRAC result identity differs")
    source = _source(manifest, int(cell["world_model_seed"]))
    state, config, metadata = _load_rebrac_checkpoint(checkpoint_path)
    if (
        result.get("updates") != manifest["rebrac_updates"]
        or result.get("dataset_file_sha256") != source["dataset_file_sha256"]
        or result.get("checkpoint_sha256")
        != benchmark.file_sha256(checkpoint_path)
        or result.get("rebrac_config") != manifest["rebrac_config"]
        or result.get("state_normalized") is not False
        or result.get("reward_normalized") is not False
        or result.get("timeout_transitions_skipped") is not True
        or config != _rebrac_config(manifest)
        or metadata.get("stage") != "flowmpc_rebrac"
        or metadata.get("cell_id") != cell["cell_id"]
        or metadata.get("manifest_sha256") != manifest["manifest_sha256"]
        or metadata.get("completed_updates") != manifest["rebrac_updates"]
        or benchmark._tree_digest(state.actor)
        != result.get("final_actor_parameter_sha256")
        or benchmark._tree_digest(state.critics)
        != result.get("final_critic_parameter_sha256")
        or result.get("initial_actor_parameter_sha256")
        != metadata.get("initial_actor_parameter_sha256")
        or result.get("initial_critic_parameter_sha256")
        != metadata.get("initial_critic_parameter_sha256")
        or not _finite_tree(result)
    ):
        raise ValueError("ReBRAC result or checkpoint contract differs")
    marker = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": "rebrac",
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
        raise ValueError("existing ReBRAC marker differs")
    write_json_atomic(marker_path, marker)
    return marker


def _rebrac_cell_for(
    manifest: Mapping[str, Any], world_seed: int, actor_seed: int
) -> Mapping[str, Any]:
    rows = [
        row
        for row in manifest["rebrac_cells"]
        if int(row["world_model_seed"]) == int(world_seed)
        and int(row["actor_seed"]) == int(actor_seed)
    ]
    if len(rows) != 1:
        raise ValueError("paired ReBRAC cell is absent or duplicated")
    return rows[0]


def _load_control_inputs(
    root: Path, manifest: Mapping[str, Any], world_seed: int, actor_seed: int
) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    import jax
    from imf_dreamer_jax import load_checkpoint

    source = _source(manifest, world_seed)
    reward_state, dreamer_config, metadata = load_checkpoint(
        source["reward_checkpoint"]
    )
    if (
        metadata.get("stage") != "itpo_state_action_reward"
        or "reward_transition" not in reward_state.params.world_model
        or benchmark._tree_digest(reward_state.params.world_model)
        == source["world_model_parameter_sha256"]
    ):
        raise ValueError("attached reward checkpoint structure differs")
    expected_config = _dreamer_config(source)
    if dreamer_config != expected_config:
        raise ValueError("attached reward checkpoint config differs")
    rebrac_cell = _rebrac_cell_for(manifest, world_seed, actor_seed)
    marker_path = root / str(rebrac_cell["marker_path"])
    marker = read_json(marker_path)
    if (
        marker.get("status") != "verified"
        or marker.get("stage") != "rebrac"
        or marker.get("cell_id") != rebrac_cell["cell_id"]
        or marker.get("marker_sha256")
        != _unsigned_digest(marker, "marker_sha256")
    ):
        raise ValueError("ReBRAC input lacks an authenticated marker")
    rebrac_path = root / str(rebrac_cell["checkpoint_path"])
    rebrac_state, rebrac_config, rebrac_metadata = _load_rebrac_checkpoint(
        rebrac_path
    )
    if (
        rebrac_config != _rebrac_config(manifest)
        or rebrac_metadata.get("completed_updates") != manifest["rebrac_updates"]
        or marker.get("checkpoint_sha256") != benchmark.file_sha256(rebrac_path)
        or not all(
            np.all(np.isfinite(np.asarray(value)))
            for value in jax.tree_util.tree_leaves(rebrac_state)
        )
    ):
        raise ValueError("ReBRAC control input is invalid")
    return (
        reward_state.params.world_model,
        dreamer_config,
        rebrac_state,
        rebrac_config,
        source,
    )


def _pad_sequences(
    sequences: list[np.ndarray], *, trailing_shape: tuple[int, ...], dtype: Any
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(value) for value in sequences], dtype=np.int32)
    maximum = int(lengths.max(initial=0))
    output = np.zeros((len(sequences), maximum, *trailing_shape), dtype=dtype)
    for index, value in enumerate(sequences):
        output[index, : len(value)] = np.asarray(value, dtype=dtype)
    return output, lengths


def _run_controller(
    world_model: Any,
    dreamer_config: Any,
    rebrac_state: Any,
    rebrac_config: Any,
    flowmpc_config: Any,
    *,
    world_seed: int,
    actor_seed: int,
    evaluation_seeds: list[int],
    adapted: bool,
    maximum_steps: int,
) -> tuple[list[float], dict[str, np.ndarray], dict[str, float]]:
    """Evaluate zero-shot or Algorithm-1 adaptation with episode resets."""

    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import (
        initial_state,
        jit_flowmpc_adapt_actor,
        observe_step,
        rebrac_actor,
    )

    world_digest = benchmark._tree_digest(world_model)
    critic_digest = benchmark._tree_digest(rebrac_state.critics)
    base_actor_digest = benchmark._tree_digest(rebrac_state.actor)
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

    returns: list[float] = []
    action_sequences: list[np.ndarray] = []
    reward_sequences: list[np.ndarray] = []
    continuation_sequences: list[np.ndarray] = []
    terminal_sequences: list[np.ndarray] = []
    objective_before_sequences: list[np.ndarray] = []
    objective_after_sequences: list[np.ndarray] = []
    gradient_sequences: list[np.ndarray] = []
    parameter_delta_sequences: list[np.ndarray] = []
    timed_seconds = 0.0
    timed_steps = 0
    for episode, evaluation_seed in enumerate(evaluation_seeds):
        environment = DMCAdapter(TASK, seed=int(evaluation_seed), action_repeat=1)
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        continuations: list[float] = []
        terminals: list[bool] = []
        objective_before: list[float] = []
        objective_after: list[float] = []
        gradients: list[float] = []
        parameter_deltas: list[float] = []
        try:
            observation = environment.reset()
            belief = initial_state(dreamer_config, 1)
            previous_action = jnp.zeros((1, dreamer_config.action_dim), jnp.float32)
            actor = rebrac_state.actor
            posterior_key = benchmark.derive_jax_key(
                "flowmpc-posterior", TASK, world_seed, actor_seed, evaluation_seed
            )
            noise_key = benchmark.derive_jax_key(
                "flowmpc-noise", TASK, world_seed, actor_seed, evaluation_seed
            )
            for step in range(maximum_steps):
                started = time.perf_counter()
                if adapted:
                    belief = observe_function(
                        jnp.asarray(observation[None]),
                        previous_action,
                        belief,
                        jax.random.fold_in(posterior_key, step),
                    )
                    noises = jax.random.normal(
                        jax.random.fold_in(noise_key, step),
                        (
                            flowmpc_config.particles,
                            flowmpc_config.horizon,
                            dreamer_config.stochastic_dim,
                        ),
                        dtype=jnp.float32,
                    )
                    update = jit_flowmpc_adapt_actor(
                        actor,
                        rebrac_state.critics,
                        world_model,
                        belief,
                        jnp.asarray(observation[None]),
                        noises,
                        dreamer_config,
                        rebrac_config,
                        flowmpc_config,
                    )
                    actor = update.actor
                    objective_before.append(float(update.objective_before))
                    objective_after.append(float(update.objective_after))
                    gradients.append(float(update.gradient_norm))
                    parameter_deltas.append(float(update.parameter_delta))
                action = actor_function(
                    actor, jnp.asarray(observation[None], dtype=jnp.float32)
                )
                host_action = np.asarray(jax.device_get(action[0]), dtype=np.float32)
                elapsed = time.perf_counter() - started
                # Exclude the first step of each episode from latency so XLA
                # compilation does not contaminate the paper-style step timing.
                if step > 0:
                    timed_seconds += elapsed
                    timed_steps += 1
                transition = environment.step(host_action)
                actions.append(host_action)
                rewards.append(float(transition.reward))
                continuations.append(float(transition.continuation))
                terminals.append(bool(transition.is_last))
                observation = transition.observation
                previous_action = action
                if transition.is_last:
                    break
        finally:
            environment.close()
        returns.append(float(np.sum(np.asarray(rewards, dtype=np.float64))))
        action_sequences.append(np.asarray(actions, dtype=np.float32))
        reward_sequences.append(np.asarray(rewards, dtype=np.float64))
        continuation_sequences.append(np.asarray(continuations, dtype=np.float64))
        terminal_sequences.append(np.asarray(terminals, dtype=np.bool_))
        objective_before_sequences.append(np.asarray(objective_before, dtype=np.float32))
        objective_after_sequences.append(np.asarray(objective_after, dtype=np.float32))
        gradient_sequences.append(np.asarray(gradients, dtype=np.float32))
        parameter_delta_sequences.append(
            np.asarray(parameter_deltas, dtype=np.float32)
        )
    action_values, lengths = _pad_sequences(
        action_sequences,
        trailing_shape=(dreamer_config.action_dim,),
        dtype=np.float32,
    )
    reward_values, reward_lengths = _pad_sequences(
        reward_sequences, trailing_shape=(), dtype=np.float64
    )
    continuation_values, continuation_lengths = _pad_sequences(
        continuation_sequences, trailing_shape=(), dtype=np.float64
    )
    terminal_values, terminal_lengths = _pad_sequences(
        terminal_sequences, trailing_shape=(), dtype=np.bool_
    )
    for observed in (reward_lengths, continuation_lengths, terminal_lengths):
        if not np.array_equal(observed, lengths):
            raise RuntimeError("controller trace lengths differ")
    trace: dict[str, np.ndarray] = {
        "actions": action_values,
        "rewards": reward_values,
        "continuations": continuation_values,
        "is_last": terminal_values,
        "lengths": lengths,
        "evaluation_seeds": np.asarray(evaluation_seeds, dtype=np.uint32),
    }
    if adapted:
        for name, sequences in (
            ("objective_before", objective_before_sequences),
            ("objective_after", objective_after_sequences),
            ("gradient_norm", gradient_sequences),
            ("parameter_delta", parameter_delta_sequences),
        ):
            values, metric_lengths = _pad_sequences(
                sequences, trailing_shape=(), dtype=np.float32
            )
            if not np.array_equal(metric_lengths, lengths):
                raise RuntimeError("FlowMPC telemetry lengths differ")
            trace[name] = values
    if (
        benchmark._tree_digest(world_model) != world_digest
        or benchmark._tree_digest(rebrac_state.critics) != critic_digest
        or benchmark._tree_digest(rebrac_state.actor) != base_actor_digest
    ):
        raise RuntimeError("controller evaluation changed a frozen source")
    timing = {
        "timed_steps": float(timed_steps),
        "total_timed_seconds": timed_seconds,
        "mean_milliseconds_per_step": (
            1000.0 * timed_seconds / timed_steps if timed_steps else 0.0
        ),
    }
    return returns, trace, timing


def _controller_config(
    manifest: Mapping[str, Any], *, step_size: float
) -> Any:
    from imf_dreamer_jax import FlowMPCConfig

    tuning = manifest["inference_tuning"]
    return FlowMPCConfig(
        horizon=int(tuning["horizon"]),
        particles=int(tuning["particles"]),
        inner_steps=int(tuning["inner_steps"]),
        step_size=float(step_size),
        discount=float(manifest["rebrac_config"]["discount"]),
    )


def _trace_with_prefix(
    output: dict[str, np.ndarray], prefix: str, trace: Mapping[str, np.ndarray]
) -> None:
    for name, value in trace.items():
        output[f"{prefix}_{name}"] = np.asarray(value)


def _assert_trace_close(
    retained: Mapping[str, np.ndarray], regenerated: Mapping[str, np.ndarray]
) -> None:
    if set(retained) != set(regenerated):
        raise ValueError("controller replay trace keys differ")
    for name in retained:
        left = np.asarray(retained[name])
        right = np.asarray(regenerated[name])
        if left.dtype.kind in "f":
            equal = np.allclose(left, right, rtol=1e-5, atol=1e-5, equal_nan=False)
        else:
            equal = np.array_equal(left, right)
        if not equal:
            maximum = (
                float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                if left.shape == right.shape and left.size
                else math.inf
            )
            raise ValueError(
                f"controller replay differs for {name}: max_abs={maximum}"
            )


def run_tuning(
    reward_root: str | Path,
    output_root: str | Path,
    *,
    rebrac_updates: int = REBRAC_UPDATES,
) -> dict[str, Any]:
    """Select the paper step size on one environment seed only."""

    manifest = write_manifest(
        reward_root, output_root, rebrac_updates=rebrac_updates
    )
    root = Path(output_root)
    tuning = manifest["inference_tuning"]
    result_path = root / str(tuning["result_path"])
    if result_path.is_file():
        return read_json(result_path)
    world_seed = int(tuning["world_model_seed"])
    actor_seed = int(tuning["actor_seed"])
    world, dreamer, rebrac_state, rebrac, source = _load_control_inputs(
        root, manifest, world_seed, actor_seed
    )
    environment_seeds = [int(tuning["environment_seed"])] * int(
        tuning["episodes_per_candidate"]
    )
    candidates: list[dict[str, Any]] = []
    traces: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for index, step_size in enumerate(tuning["step_sizes"]):
        controller = _controller_config(manifest, step_size=float(step_size))
        returns, trace, timing = _run_controller(
            world,
            dreamer,
            rebrac_state,
            rebrac,
            controller,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=environment_seeds,
            adapted=True,
            maximum_steps=int(manifest["maximum_environment_steps"]),
        )
        _trace_with_prefix(traces, f"candidate_{index}", trace)
        candidates.append(
            {
                "index": index,
                "step_size": float(step_size),
                "episode_returns": returns,
                "mean_return": float(np.mean(returns)),
                "mean_milliseconds_per_step": timing[
                    "mean_milliseconds_per_step"
                ],
            }
        )
    selected = sorted(
        candidates, key=lambda row: (-float(row["mean_return"]), row["step_size"])
    )[0]
    trace_path = root / str(tuning["trace_path"])
    benchmark._write_npz_atomic(trace_path, traces)
    rebrac_cell = _rebrac_cell_for(manifest, world_seed, actor_seed)
    result = {
        "schema_version": TUNING_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "task": TASK,
        "world_model_seed": world_seed,
        "actor_seed": actor_seed,
        "environment_seed": int(tuning["environment_seed"]),
        "evaluation_seeds_accessed": False,
        "candidates": candidates,
        "selected_index": int(selected["index"]),
        "selected_step_size": float(selected["step_size"]),
        "reward_checkpoint_sha256": source["reward_checkpoint_sha256"],
        "rebrac_checkpoint_sha256": benchmark.file_sha256(
            root / str(rebrac_cell["checkpoint_path"])
        ),
        "trace_file_sha256": benchmark.file_sha256(trace_path),
        "trace_sha256": benchmark.array_sha256(traces),
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if not _finite_tree(result):
        raise FloatingPointError("FlowMPC tuning result is not finite")
    write_json_atomic(result_path, result)
    return result


def verify_tuning(
    output_root: str | Path, *, strict_replay: bool = True
) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    tuning = manifest["inference_tuning"]
    result_path = root / str(tuning["result_path"])
    trace_path = root / str(tuning["trace_path"])
    result = read_json(result_path)
    traces = benchmark.load_npz(trace_path)
    if (
        result.get("schema_version") != TUNING_RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("source_commit") != manifest["source_commit"]
        or result.get("manifest_sha256") != manifest["manifest_sha256"]
        or result.get("task") != TASK
        or result.get("evaluation_seeds_accessed") is not False
        or result.get("environment_seed") != tuning["environment_seed"]
        or len(result.get("candidates", ())) != len(tuning["step_sizes"])
        or result.get("trace_file_sha256") != benchmark.file_sha256(trace_path)
        or result.get("trace_sha256") != benchmark.array_sha256(traces)
        or not _finite_tree(result)
    ):
        raise ValueError("FlowMPC tuning result contract differs")
    expected = sorted(
        result["candidates"],
        key=lambda row: (-float(row["mean_return"]), row["step_size"]),
    )[0]
    if (
        result.get("selected_index") != expected["index"]
        or result.get("selected_step_size") != expected["step_size"]
        or [row["step_size"] for row in result["candidates"]]
        != tuning["step_sizes"]
    ):
        raise ValueError("FlowMPC tuning selection differs")
    if strict_replay:
        world_seed = int(tuning["world_model_seed"])
        actor_seed = int(tuning["actor_seed"])
        world, dreamer, rebrac_state, rebrac, _ = _load_control_inputs(
            root, manifest, world_seed, actor_seed
        )
        regenerated: dict[str, np.ndarray] = {}
        for index, step_size in enumerate(tuning["step_sizes"]):
            _, trace, _ = _run_controller(
                world,
                dreamer,
                rebrac_state,
                rebrac,
                _controller_config(manifest, step_size=float(step_size)),
                world_seed=world_seed,
                actor_seed=actor_seed,
                evaluation_seeds=[int(tuning["environment_seed"])],
                adapted=True,
                maximum_steps=int(manifest["maximum_environment_steps"]),
            )
            _trace_with_prefix(regenerated, f"candidate_{index}", trace)
        _assert_trace_close(traces, regenerated)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": "tuning",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "result_file_sha256": benchmark.file_sha256(result_path),
        "trace_file_sha256": result["trace_file_sha256"],
        "strict_policy_and_environment_replay": bool(strict_replay),
        "evaluation_seeds_accessed": False,
    }
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / str(tuning["marker_path"])
    if marker_path.is_file() and read_json(marker_path) != marker:
        raise ValueError("existing tuning marker differs")
    write_json_atomic(marker_path, marker)
    return marker


def _prefix_trace(trace: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}_{name}": np.asarray(value) for name, value in trace.items()}


def _trace_returns(trace: Mapping[str, np.ndarray]) -> np.ndarray:
    rewards = np.asarray(trace["rewards"], dtype=np.float64)
    lengths = np.asarray(trace["lengths"], dtype=np.int64)
    return np.asarray(
        [np.sum(rewards[index, :length]) for index, length in enumerate(lengths)],
        dtype=np.float64,
    )


def run_evaluation_cell(
    reward_root: str | Path,
    output_root: str | Path,
    index: int,
    *,
    rebrac_updates: int = REBRAC_UPDATES,
) -> dict[str, Any]:
    """Evaluate one paired zero-shot ReBRAC and FlowMPC unit."""

    manifest = write_manifest(
        reward_root, output_root, rebrac_updates=rebrac_updates
    )
    root = Path(output_root)
    cell = _cell(manifest, "evaluation", index)
    result_path = root / str(cell["result_path"])
    if result_path.is_file():
        return read_json(result_path)
    tuning_marker = read_json(root / str(manifest["inference_tuning"]["marker_path"]))
    if (
        tuning_marker.get("status") != "verified"
        or tuning_marker.get("strict_policy_and_environment_replay") is not True
    ):
        raise ValueError("evaluation requires strictly authenticated tuning")
    tuning_result = read_json(root / str(manifest["inference_tuning"]["result_path"]))
    world_seed = int(cell["world_model_seed"])
    actor_seed = int(cell["actor_seed"])
    world, dreamer, rebrac_state, rebrac, source = _load_control_inputs(
        root, manifest, world_seed, actor_seed
    )
    controller = _controller_config(
        manifest, step_size=float(tuning_result["selected_step_size"])
    )
    evaluation_seeds = [int(value) for value in cell["evaluation_seeds"]]
    started = time.perf_counter()
    zero_returns, zero_trace, zero_timing = _run_controller(
        world,
        dreamer,
        rebrac_state,
        rebrac,
        controller,
        world_seed=world_seed,
        actor_seed=actor_seed,
        evaluation_seeds=evaluation_seeds,
        adapted=False,
        maximum_steps=int(manifest["maximum_environment_steps"]),
    )
    adapted_returns, adapted_trace, adapted_timing = _run_controller(
        world,
        dreamer,
        rebrac_state,
        rebrac,
        controller,
        world_seed=world_seed,
        actor_seed=actor_seed,
        evaluation_seeds=evaluation_seeds,
        adapted=True,
        maximum_steps=int(manifest["maximum_environment_steps"]),
    )
    traces = {
        **_prefix_trace(zero_trace, "zero_shot"),
        **_prefix_trace(adapted_trace, "flowmpc"),
    }
    trace_path = root / str(cell["trace_path"])
    benchmark._write_npz_atomic(trace_path, traces)
    zero = np.asarray(zero_returns, dtype=np.float64)
    adapted = np.asarray(adapted_returns, dtype=np.float64)
    adapted_lengths = np.asarray(adapted_trace["lengths"], dtype=np.int64)
    adapted_mask = (
        np.arange(adapted_trace["actions"].shape[1])[None]
        < adapted_lengths[:, None]
    )
    zero_lengths = np.asarray(zero_trace["lengths"], dtype=np.int64)
    zero_mask = (
        np.arange(zero_trace["actions"].shape[1])[None] < zero_lengths[:, None]
    )
    gradients = np.asarray(adapted_trace["gradient_norm"], dtype=np.float64)[
        adapted_mask
    ]
    deltas = np.asarray(adapted_trace["parameter_delta"], dtype=np.float64)[
        adapted_mask
    ]
    objective_gain = (
        np.asarray(adapted_trace["objective_after"], dtype=np.float64)[adapted_mask]
        - np.asarray(adapted_trace["objective_before"], dtype=np.float64)[adapted_mask]
    )
    rebrac_cell = _rebrac_cell_for(manifest, world_seed, actor_seed)
    result = {
        "schema_version": EVALUATION_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "world_model_seed": world_seed,
        "actor_seed": actor_seed,
        "evaluation_seeds": evaluation_seeds,
        "evaluation_seeds_disjoint_from_tuning": int(
            manifest["inference_tuning"]["environment_seed"]
        )
        not in evaluation_seeds,
        "controller_config": asdict(controller),
        "selected_tuning_result_sha256": benchmark.file_sha256(
            root / str(manifest["inference_tuning"]["result_path"])
        ),
        "reward_checkpoint_sha256": source["reward_checkpoint_sha256"],
        "rebrac_checkpoint_sha256": benchmark.file_sha256(
            root / str(rebrac_cell["checkpoint_path"])
        ),
        "zero_shot_episode_returns": zero.tolist(),
        "flowmpc_episode_returns": adapted.tolist(),
        "paired_episode_deltas": (adapted - zero).tolist(),
        "zero_shot_mean_return": float(np.mean(zero)),
        "flowmpc_mean_return": float(np.mean(adapted)),
        "flowmpc_minus_zero_shot_mean": float(np.mean(adapted - zero)),
        "zero_shot_normalized_mean_return": float(np.mean(zero) / 1000.0),
        "flowmpc_normalized_mean_return": float(np.mean(adapted) / 1000.0),
        "action_saturation_fraction": {
            "zero_shot": float(
                np.mean(np.abs(zero_trace["actions"])[zero_mask] >= 0.95)
            ),
            "flowmpc": float(
                np.mean(np.abs(adapted_trace["actions"])[adapted_mask] >= 0.95)
            ),
        },
        "mean_pathwise_gradient_norm": float(np.mean(gradients)),
        "mean_per_step_actor_parameter_delta": float(np.mean(deltas)),
        "mean_predicted_objective_gain": float(np.mean(objective_gain)),
        "nonnegative_predicted_objective_gain_fraction": float(
            np.mean(objective_gain >= -1e-6)
        ),
        "mean_milliseconds_per_step": {
            "zero_shot": zero_timing["mean_milliseconds_per_step"],
            "flowmpc": adapted_timing["mean_milliseconds_per_step"],
        },
        "world_model_frozen": True,
        "terminal_critic_frozen": True,
        "base_actor_reset_each_episode": True,
        "adapted_actor_persistent_within_episode": True,
        "trace_file_sha256": benchmark.file_sha256(trace_path),
        "trace_sha256": benchmark.array_sha256(traces),
        "wall_seconds": time.perf_counter() - started,
        "runtime": benchmark.runtime_fingerprint(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if not _finite_tree(result):
        raise FloatingPointError("FlowMPC evaluation result is not finite")
    write_json_atomic(result_path, result)
    return result


def _unprefix_trace(
    arrays: Mapping[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    token = prefix + "_"
    result = {
        name[len(token) :]: np.asarray(value)
        for name, value in arrays.items()
        if name.startswith(token)
    }
    if not result:
        raise ValueError(f"trace prefix {prefix!r} is absent")
    return result


def verify_evaluation_cell(
    output_root: str | Path,
    index: int,
    *,
    strict_replay: bool = True,
) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    cell = _cell(manifest, "evaluation", index)
    result_path = root / str(cell["result_path"])
    trace_path = root / str(cell["trace_path"])
    result = read_json(result_path)
    traces = benchmark.load_npz(trace_path)
    identity = {
        "schema_version": EVALUATION_RESULT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "task": TASK,
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
    }
    if any(result.get(name) != value for name, value in identity.items()):
        raise ValueError("FlowMPC evaluation identity differs")
    zero_trace = _unprefix_trace(traces, "zero_shot")
    adapted_trace = _unprefix_trace(traces, "flowmpc")
    zero_returns = _trace_returns(zero_trace)
    adapted_returns = _trace_returns(adapted_trace)
    tuning_result_path = root / str(manifest["inference_tuning"]["result_path"])
    tuning_result = read_json(tuning_result_path)
    controller = _controller_config(
        manifest, step_size=float(tuning_result["selected_step_size"])
    )
    source = _source(manifest, int(cell["world_model_seed"]))
    rebrac_cell = _rebrac_cell_for(
        manifest, int(cell["world_model_seed"]), int(cell["actor_seed"])
    )
    required_trace_keys = {
        "actions",
        "rewards",
        "continuations",
        "is_last",
        "lengths",
        "evaluation_seeds",
    }
    if (
        set(zero_trace) != required_trace_keys
        or set(adapted_trace)
        != required_trace_keys
        | {
            "objective_before",
            "objective_after",
            "gradient_norm",
            "parameter_delta",
        }
        or result.get("evaluation_seeds") != cell["evaluation_seeds"]
        or result.get("evaluation_seeds_disjoint_from_tuning") is not True
        or result.get("controller_config") != asdict(controller)
        or result.get("reward_checkpoint_sha256")
        != source["reward_checkpoint_sha256"]
        or result.get("rebrac_checkpoint_sha256")
        != benchmark.file_sha256(root / str(rebrac_cell["checkpoint_path"]))
        or result.get("selected_tuning_result_sha256")
        != benchmark.file_sha256(tuning_result_path)
        or result.get("trace_file_sha256") != benchmark.file_sha256(trace_path)
        or result.get("trace_sha256") != benchmark.array_sha256(traces)
        or not np.allclose(
            zero_returns,
            np.asarray(result.get("zero_shot_episode_returns")),
            rtol=0.0,
            atol=1e-10,
        )
        or not np.allclose(
            adapted_returns,
            np.asarray(result.get("flowmpc_episode_returns")),
            rtol=0.0,
            atol=1e-10,
        )
        or not np.allclose(
            adapted_returns - zero_returns,
            np.asarray(result.get("paired_episode_deltas")),
            rtol=0.0,
            atol=1e-10,
        )
        or result.get("world_model_frozen") is not True
        or result.get("terminal_critic_frozen") is not True
        or result.get("base_actor_reset_each_episode") is not True
        or result.get("adapted_actor_persistent_within_episode") is not True
        or not _finite_tree(result)
    ):
        raise ValueError("FlowMPC evaluation result contract differs")
    if not np.array_equal(
        zero_trace["evaluation_seeds"], adapted_trace["evaluation_seeds"]
    ) or [int(value) for value in zero_trace["evaluation_seeds"]] != cell[
        "evaluation_seeds"
    ]:
        raise ValueError("zero-shot and FlowMPC evaluation seeds are unpaired")
    if strict_replay:
        world_seed = int(cell["world_model_seed"])
        actor_seed = int(cell["actor_seed"])
        world, dreamer, rebrac_state, rebrac, _ = _load_control_inputs(
            root, manifest, world_seed, actor_seed
        )
        _, replay_zero, _ = _run_controller(
            world,
            dreamer,
            rebrac_state,
            rebrac,
            controller,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=cell["evaluation_seeds"],
            adapted=False,
            maximum_steps=int(manifest["maximum_environment_steps"]),
        )
        _, replay_adapted, _ = _run_controller(
            world,
            dreamer,
            rebrac_state,
            rebrac,
            controller,
            world_seed=world_seed,
            actor_seed=actor_seed,
            evaluation_seeds=cell["evaluation_seeds"],
            adapted=True,
            maximum_steps=int(manifest["maximum_environment_steps"]),
        )
        _assert_trace_close(zero_trace, replay_zero)
        _assert_trace_close(adapted_trace, replay_adapted)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "status": "verified",
        "stage": "evaluation",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "cell_id": cell["cell_id"],
        "cell_index": int(cell["index"]),
        "result_file_sha256": benchmark.file_sha256(result_path),
        "trace_file_sha256": result["trace_file_sha256"],
        "strict_policy_and_environment_replay": bool(strict_replay),
    }
    marker["marker_sha256"] = benchmark.object_sha256(marker)
    marker_path = root / str(cell["marker_path"])
    if marker_path.is_file() and read_json(marker_path) != marker:
        raise ValueError("existing FlowMPC evaluation marker differs")
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
    if (
        marker.get("schema_version") != MARKER_SCHEMA
        or marker.get("status") != "verified"
        or marker.get("stage") != stage
        or marker.get("source_commit") != manifest["source_commit"]
        or marker.get("manifest_sha256") != manifest["manifest_sha256"]
        or marker.get("cell_id") != cell["cell_id"]
        or marker.get("cell_index") != cell["index"]
        or marker.get("result_file_sha256") != benchmark.file_sha256(result_path)
        or marker.get("marker_sha256")
        != _unsigned_digest(marker, "marker_sha256")
    ):
        raise ValueError(f"{stage} marker is invalid")
    return marker


def finalize(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)
    for cell in manifest["rebrac_cells"]:
        _validate_marker(root, manifest, cell, "rebrac")
    tuning = manifest["inference_tuning"]
    tuning_marker = read_json(root / str(tuning["marker_path"]))
    tuning_result_path = root / str(tuning["result_path"])
    if (
        tuning_marker.get("schema_version") != MARKER_SCHEMA
        or tuning_marker.get("status") != "verified"
        or tuning_marker.get("stage") != "tuning"
        or tuning_marker.get("manifest_sha256") != manifest["manifest_sha256"]
        or tuning_marker.get("result_file_sha256")
        != benchmark.file_sha256(tuning_result_path)
        or tuning_marker.get("strict_policy_and_environment_replay") is not True
        or tuning_marker.get("marker_sha256")
        != _unsigned_digest(tuning_marker, "marker_sha256")
    ):
        raise ValueError("tuning marker is invalid")
    evaluations: list[dict[str, Any]] = []
    for cell in manifest["evaluation_cells"]:
        marker = _validate_marker(root, manifest, cell, "evaluation")
        if marker.get("strict_policy_and_environment_replay") is not True:
            raise ValueError("evaluation marker lacks strict replay")
        evaluations.append(read_json(root / str(cell["result_path"])))
    units: list[dict[str, Any]] = []
    for world_seed in manifest["world_model_seeds"]:
        rows = [
            row
            for row in evaluations
            if int(row["world_model_seed"]) == int(world_seed)
        ]
        if len(rows) != len(manifest["actor_seeds"]):
            raise ValueError("world-seed unit lacks nested actor cells")
        zero = np.concatenate(
            [np.asarray(row["zero_shot_episode_returns"], np.float64) for row in rows]
        )
        adapted = np.concatenate(
            [np.asarray(row["flowmpc_episode_returns"], np.float64) for row in rows]
        )
        units.append(
            {
                "task": TASK,
                "world_model_seed": int(world_seed),
                "zero_shot_mean_return": float(np.mean(zero)),
                "flowmpc_mean_return": float(np.mean(adapted)),
                "flowmpc_minus_zero_shot": float(np.mean(adapted - zero)),
            }
        )
    zero_units = np.asarray(
        [row["zero_shot_mean_return"] for row in units], dtype=np.float64
    )
    adapted_units = np.asarray(
        [row["flowmpc_mean_return"] for row in units], dtype=np.float64
    )
    deltas = adapted_units - zero_units
    reward_report = read_json(Path(manifest["reward_root"]) / "report.json")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": manifest["source_commit"],
        "manifest_sha256": manifest["manifest_sha256"],
        "task": TASK,
        "completed_rebrac_cells": len(manifest["rebrac_cells"]),
        "completed_evaluation_cells": len(evaluations),
        "strict_rebrac_markers": len(manifest["rebrac_cells"]),
        "strict_evaluation_markers": len(evaluations),
        "selected_step_size": read_json(tuning_result_path)["selected_step_size"],
        "primary_raw_return_iqm": {
            "rebrac_zero_shot": benchmark.interquartile_mean(zero_units),
            "flowmpc": benchmark.interquartile_mean(adapted_units),
            "flowmpc_minus_zero_shot": benchmark.interquartile_mean(deltas),
        },
        "primary_normalized_return_iqm": {
            "rebrac_zero_shot": benchmark.interquartile_mean(zero_units) / 1000.0,
            "flowmpc": benchmark.interquartile_mean(adapted_units) / 1000.0,
            "flowmpc_minus_zero_shot": benchmark.interquartile_mean(deltas)
            / 1000.0,
        },
        "favorable_world_seed_fraction": float(np.mean(deltas > 0.0)),
        "paired_world_seed_units": units,
        "controller_telemetry": {
            "mean_pathwise_gradient_norm": float(
                np.mean([row["mean_pathwise_gradient_norm"] for row in evaluations])
            ),
            "mean_per_step_actor_parameter_delta": float(
                np.mean(
                    [
                        row["mean_per_step_actor_parameter_delta"]
                        for row in evaluations
                    ]
                )
            ),
            "mean_predicted_objective_gain": float(
                np.mean([row["mean_predicted_objective_gain"] for row in evaluations])
            ),
            "mean_nonnegative_predicted_objective_gain_fraction": float(
                np.mean(
                    [
                        row["nonnegative_predicted_objective_gain_fraction"]
                        for row in evaluations
                    ]
                )
            ),
            "mean_action_saturation_fraction": {
                mode: float(
                    np.mean(
                        [row["action_saturation_fraction"][mode] for row in evaluations]
                    )
                )
                for mode in ("zero_shot", "flowmpc")
            },
            "mean_milliseconds_per_step": {
                mode: float(
                    np.mean(
                        [row["mean_milliseconds_per_step"][mode] for row in evaluations]
                    )
                )
                for mode in ("zero_shot", "flowmpc")
            },
        },
        "secondary_nonmatched_previous_actor_iqm": reward_report[
            "normalized_return_iqm"
        ],
        "runtime_evidence": {
            "total_rebrac_wall_seconds": float(
                sum(
                    read_json(root / str(cell["result_path"]))["wall_seconds"]
                    for cell in manifest["rebrac_cells"]
                )
            ),
            "tuning_wall_seconds": read_json(tuning_result_path)["wall_seconds"],
            "total_evaluation_wall_seconds": float(
                sum(row["wall_seconds"] for row in evaluations)
            ),
            "runtime": evaluations[0]["runtime"],
        },
        "limitations": manifest["declared_deviations"],
        "claim_status": "exploratory_controller_diagnostic_only",
        "supports_itpo_controller_on_reacher": bool(
            benchmark.interquartile_mean(deltas) > 0.0
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    report["report_sha256"] = benchmark.object_sha256(report)
    write_json_atomic(root / "report.json", report)
    return report


def validate_final(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    report = read_json(root / "report.json")
    manifest = read_json(root / "manifest.json")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("source_commit") != manifest.get("source_commit")
        or report.get("manifest_sha256") != manifest.get("manifest_sha256")
        or report.get("completed_rebrac_cells") != EXPECTED_REBRAC_CELLS
        or report.get("completed_evaluation_cells")
        != EXPECTED_EVALUATION_CELLS
        or report.get("strict_rebrac_markers") != EXPECTED_REBRAC_CELLS
        or report.get("strict_evaluation_markers")
        != EXPECTED_EVALUATION_CELLS
        or report.get("claim_status") != "exploratory_controller_diagnostic_only"
        or report.get("report_sha256") != _unsigned_digest(report, "report_sha256")
        or not _finite_tree(report)
    ):
        raise ValueError("final FlowMPC actor report is invalid")
    return report


def aggregate_synthetic_units(units: list[Mapping[str, float]]) -> dict[str, float]:
    zero = np.asarray([row["zero"] for row in units], dtype=np.float64)
    adapted = np.asarray([row["adapted"] for row in units], dtype=np.float64)
    return {
        "zero_iqm": benchmark.interquartile_mean(zero),
        "adapted_iqm": benchmark.interquartile_mean(adapted),
        "delta_iqm": benchmark.interquartile_mean(adapted - zero),
    }


def self_test() -> None:
    """Exercise shifted replay alignment, seed separation, and aggregation."""

    arrays = {
        "observations": np.asarray(
            [[[0.0], [1.0], [2.0], [3.0], [4.0]]], dtype=np.float32
        ),
        "actions": np.asarray(
            [[[0.0], [10.0], [20.0], [30.0], [40.0]]], dtype=np.float32
        ),
        "rewards": np.asarray([[0.0, 100.0, 200.0, 300.0, 400.0]], np.float32),
        "continuations": np.asarray([[1.0, 1.0, 0.0, 0.0, 0.0]], np.float32),
        "train_episode_ids": np.asarray([0], np.int32),
    }
    dataset = _build_rebrac_dataset(arrays)
    # Includes real transitions 0->1 and 1->2, including the terminal at 2;
    # removes padded successor and the native final-timeout slot.
    if (
        not np.array_equal(np.asarray(dataset.states[:, 0]), [0.0, 1.0])
        or not np.array_equal(np.asarray(dataset.actions[:, 0]), [10.0, 20.0])
        or not np.array_equal(np.asarray(dataset.rewards), [100.0, 200.0])
        or not np.array_equal(np.asarray(dataset.next_actions[:, 0]), [20.0, 30.0])
        or not np.array_equal(np.asarray(dataset.dones), [0.0, 1.0])
    ):
        raise AssertionError("shifted ReBRAC replay alignment is wrong")
    tuning_seed = benchmark.derive_seed("flowmpc-tuning", TASK)
    evaluation = {
        seed
        for world_seed in (211, 223, 227)
        for actor_seed in (311, 313)
        for seed in _evaluation_seeds(world_seed, actor_seed)
    }
    if tuning_seed in evaluation:
        raise AssertionError("tuning seed leaked into evaluation")
    summary = aggregate_synthetic_units(
        [
            {"zero": 1.0, "adapted": 1.5},
            {"zero": 2.0, "adapted": 2.5},
            {"zero": 3.0, "adapted": 3.5},
        ]
    )
    if not math.isclose(summary["delta_iqm"], 0.5):
        raise AssertionError("paired world-seed aggregation is wrong")
