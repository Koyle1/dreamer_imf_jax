#!/usr/bin/env python3
"""Run and verify policy-alignment diagnostics against a frozen pilot.

The ``run`` command reads authenticated checkpoints and simulator seeds but
writes only to a separate diagnostic directory.  It never mutates the pilot
root.  Verification subcommands intentionally avoid JAX and dm_control.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

# Verification subcommands remain dependency-light and do not import the full
# comparison package. Execution imports JAX and dm_control lazily in ``run``.
DIAGNOSTIC_MODULE = REPOSITORY / "dreamer_imf_compare" / "policy_alignment_diagnostics.py"
DIAGNOSTIC_SPEC = importlib.util.spec_from_file_location(
    "policy_alignment_diagnostics", DIAGNOSTIC_MODULE
)
if DIAGNOSTIC_SPEC is None or DIAGNOSTIC_SPEC.loader is None:
    raise ImportError(f"cannot load diagnostic module {DIAGNOSTIC_MODULE}")
diagnostics = importlib.util.module_from_spec(DIAGNOSTIC_SPEC)
DIAGNOSTIC_SPEC.loader.exec_module(diagnostics)

ARM_NAMES = diagnostics.ARM_NAMES
PRESERVATION_SCHEMA = diagnostics.PRESERVATION_SCHEMA
REPORT_SCHEMA = diagnostics.REPORT_SCHEMA
actor_number_summary = diagnostics.actor_number_summary
aggregate_metric_rows = diagnostics.aggregate_metric_rows
imagined_real_alignment = diagnostics.imagined_real_alignment
validate_report = diagnostics.validate_report
verify_metric_coverage = diagnostics.verify_metric_coverage


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_cells(matrix: Mapping[str, Any], stage: str) -> list[Mapping[str, Any]]:
    return [cell for cell in matrix["cells"] if cell["stage"] == stage]


def _dependency(
    matrix: Mapping[str, Any], cell: Mapping[str, Any], stage: str
) -> Mapping[str, Any]:
    by_id = {entry["cell_id"]: entry for entry in matrix["cells"]}
    matches = [by_id[value] for value in cell["dependencies"] if by_id[value]["stage"] == stage]
    if len(matches) != 1:
        raise ValueError(f"cell {cell['cell_id']} has {len(matches)} {stage} dependencies")
    return matches[0]


def _candidate_cells(
    matrix: Mapping[str, Any], selection: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    selected = {arm: selection["selected"][arm]["candidate_id"] for arm in ARM_NAMES}
    cells = [
        cell
        for cell in _stage_cells(matrix, "actor")
        if cell["candidate_id"] == selected[cell["arm"]]
    ]
    expected = {
        (task, arm)
        for task in ("dmc_pendulum_swingup", "dmc_reacher_easy")
        for arm in ARM_NAMES
    }
    observed = {(cell["task"], cell["arm"]) for cell in cells}
    if observed != expected:
        raise ValueError("selected candidate actors do not cover the pilot task-arm grid")
    return cells


def _sample_cells(
    selected: Sequence[Mapping[str, Any]], cells_per_group: int
) -> list[Mapping[str, Any]]:
    if cells_per_group <= 0:
        raise ValueError("cells per group must be positive")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in selected:
        groups[(cell["task"], cell["arm"])].append(cell)
    chosen: list[Mapping[str, Any]] = []
    for key, cells in sorted(groups.items()):
        ordered = sorted(cells, key=lambda row: (row["world_model_seed"], row["actor_seed"], row["cell_id"]))
        group: list[Mapping[str, Any]] = []
        used_world_seeds: set[int] = set()
        for cell in ordered:
            seed = int(cell["world_model_seed"])
            if seed in used_world_seeds:
                continue
            group.append(cell)
            used_world_seeds.add(seed)
            if len(group) == cells_per_group:
                break
        if len(group) < cells_per_group:
            for cell in ordered:
                if cell not in group:
                    group.append(cell)
                    if len(group) == cells_per_group:
                        break
        if len(group) != cells_per_group:
            raise ValueError(f"insufficient selected actors for group {key}")
        chosen.extend(group)
    return chosen


def _repeat_state(state: Any, count: int) -> Any:
    import jax.numpy as jnp
    from imf_dreamer_jax import RSSMState

    return RSSMState(
        jnp.repeat(state.deterministic, count, axis=0),
        jnp.repeat(state.stochastic, count, axis=0),
    )


def _discounted_model_returns(
    reward_samples: Any, continuation_samples: Any, discount: float
) -> Any:
    import jax.numpy as jnp

    rewards = jnp.asarray(reward_samples)
    continuations = jnp.asarray(continuation_samples)
    prefix = jnp.ones((*continuations.shape[:-1], 1), dtype=continuations.dtype)
    factors = discount * continuations[..., :-1]
    weights = jnp.concatenate([prefix, jnp.cumprod(factors, axis=-1)], axis=-1)
    return jnp.sum(weights * rewards, axis=-1)


def _model_action_gradient_function():
    import jax
    import jax.numpy as jnp
    from dreamer_imf_compare import matched_objective_benchmark as benchmark

    def objective(action, params, start, suffix, noise, discount, config):
        actions = jnp.concatenate([action[None, None], suffix[None]], axis=1)
        _, rewards, continuations = benchmark.open_loop_samples_with_continuation(
            params, start, actions, noise, config
        )
        return jnp.mean(_discounted_model_returns(rewards, continuations, discount))

    return jax.jit(jax.grad(objective), static_argnames=("config",))


def _simulate_open_loop(
    environment: Any,
    snapshot: Any,
    actions: np.ndarray,
    *,
    discount: float,
) -> tuple[float, Any]:
    environment.restore(snapshot)
    total = 0.0
    weight = 1.0
    first = None
    for action in np.asarray(actions, dtype=np.float32):
        transition = environment.step(action)
        if first is None:
            first = transition
        total += weight * float(transition.reward)
        weight *= float(discount) * float(transition.continuation)
        if transition.is_last:
            break
    if first is None:
        raise ValueError("open-loop simulator branch cannot be empty")
    return float(total), first


def _candidate_actions(action: np.ndarray, delta: float) -> np.ndarray:
    values = [np.asarray(action, dtype=np.float32).copy()]
    for dimension in range(action.size):
        for direction in (-1.0, 1.0):
            candidate = np.asarray(action, dtype=np.float32).copy()
            candidate[dimension] = np.clip(
                candidate[dimension] + direction * delta, -1.0, 1.0
            )
            values.append(candidate)
    return np.stack(values)


def _finite_difference_gradient(
    environment: Any,
    snapshot: Any,
    action: np.ndarray,
    suffix: np.ndarray,
    *,
    epsilon: float,
    discount: float,
) -> np.ndarray:
    gradient = np.zeros_like(action, dtype=np.float64)
    for dimension in range(action.size):
        lower = np.asarray(action, dtype=np.float32).copy()
        upper = np.asarray(action, dtype=np.float32).copy()
        lower[dimension] = np.clip(lower[dimension] - epsilon, -1.0, 1.0)
        upper[dimension] = np.clip(upper[dimension] + epsilon, -1.0, 1.0)
        denominator = float(upper[dimension] - lower[dimension])
        if denominator <= 0.0:
            raise ValueError("finite-difference action interval collapsed")
        lower_return, _ = _simulate_open_loop(
            environment,
            snapshot,
            np.concatenate([lower[None], suffix], axis=0),
            discount=discount,
        )
        upper_return, _ = _simulate_open_loop(
            environment,
            snapshot,
            np.concatenate([upper[None], suffix], axis=0),
            discount=discount,
        )
        gradient[dimension] = (upper_return - lower_return) / denominator
    environment.restore(snapshot)
    return gradient


def _diagnose_actor_cell(
    pilot_root: Path,
    matrix: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    output_root: Path,
    episode: int,
    states_per_cell: int,
    horizon: int,
    draws: int,
    action_delta: float,
    finite_difference_epsilon: float,
    discount: float,
    sampler: Any,
    gradient_function: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state, jit_act, load_checkpoint
    from dreamer_imf_compare.dmc import DMCAdapter
    from dreamer_imf_compare import matched_objective_benchmark as benchmark

    actor_directory = pilot_root / "actor" / cell["cell_id"]
    result = read_json(actor_directory / "result.json")
    state, config, metadata = load_checkpoint(actor_directory / "checkpoint.pkl")
    if metadata.get("cell_id") != cell["cell_id"]:
        raise ValueError("actor checkpoint identity differs from the matrix")
    if result.get("checkpoint_sha256") != file_sha256(actor_directory / "checkpoint.pkl"):
        raise ValueError("actor checkpoint digest differs from the authenticated result")
    traces = np.load(actor_directory / "action_traces.npz")
    actions = np.asarray(traces["actions"][episode], dtype=np.float32)
    length = int(np.asarray(traces["lengths"])[episode])
    evaluation_seed = int(np.asarray(traces["evaluation_seeds"])[episode])
    if length <= horizon + 8:
        raise ValueError("actor trace is too short for policy-alignment diagnostics")
    sample_steps = np.linspace(
        max(8, int(config.burn_in)), length - horizon - 1, states_per_cell, dtype=np.int32
    )
    sample_set = set(int(value) for value in sample_steps)

    dataset_cell = _dependency(matrix, cell, "dataset")
    dataset = np.load(pilot_root / "dataset" / dataset_cell["cell_id"] / "dataset.npz")
    observations = np.asarray(dataset["observations"], dtype=np.float32)
    train_ids = np.asarray(dataset["train_episode_ids"], dtype=np.int64)
    observation_std = np.maximum(
        np.std(observations[train_ids].reshape((-1, *observations.shape[2:])), axis=0),
        1e-6,
    ).astype(np.float32)

    model_observations: list[np.ndarray] = []
    model_rewards: list[np.ndarray] = []
    model_continuations: list[np.ndarray] = []
    target_observations: list[np.ndarray] = []
    target_rewards: list[float] = []
    target_continuations: list[float] = []
    predicted_rank_returns: list[np.ndarray] = []
    simulator_rank_returns: list[np.ndarray] = []
    model_gradients: list[np.ndarray] = []
    simulator_gradients: list[np.ndarray] = []
    action_replay_max_error = 0.0

    environment = DMCAdapter(cell["task"], seed=evaluation_seed, action_repeat=1)
    try:
        observation = environment.reset()
        belief = initial_state(config, 1)
        previous_action = jnp.zeros((1, config.action_dim), jnp.float32)
        key = benchmark.derive_jax_key(
            "actor-evaluation-action",
            cell["task"],
            cell["world_model_seed"],
            cell["actor_seed"],
            episode,
        )
        for step in range(int(sample_steps[-1]) + 1):
            policy_action, belief = jit_act(
                state.params,
                jnp.asarray(observation[None]),
                previous_action,
                belief,
                jax.random.fold_in(key, step),
                config,
                deterministic=True,
            )
            host_action = np.asarray(policy_action[0], dtype=np.float32)
            action_replay_max_error = max(
                action_replay_max_error,
                float(np.max(np.abs(host_action - actions[step]))),
            )
            if step in sample_set:
                snapshot = environment.snapshot()
                suffix = actions[step + 1 : step + horizon]
                candidates = _candidate_actions(host_action, action_delta)
                branch_actions = np.repeat(
                    np.concatenate([host_action[None], suffix], axis=0)[None],
                    candidates.shape[0],
                    axis=0,
                )
                branch_actions[:, 0] = candidates
                random = np.random.default_rng(
                    benchmark.derive_seed("policy-alignment", cell["cell_id"], episode, step)
                )
                shared_noise = random.normal(
                    size=(draws, horizon, 1, config.stochastic_dim)
                ).astype(np.float32)
                noise = np.broadcast_to(
                    shared_noise,
                    (draws, horizon, candidates.shape[0], config.stochastic_dim),
                ).copy()
                start = _repeat_state(belief, candidates.shape[0])
                obs_samples, reward_samples, continuation_samples = sampler(
                    state.params.world_model,
                    start,
                    jnp.asarray(branch_actions),
                    jnp.asarray(noise),
                    config,
                )
                obs_samples = np.asarray(obs_samples)
                reward_samples = np.asarray(reward_samples)
                continuation_samples = np.asarray(continuation_samples)
                predicted = np.asarray(
                    _discounted_model_returns(
                        reward_samples, continuation_samples, discount
                    ).mean(axis=0)
                )
                true_returns = []
                baseline_transition = None
                for index, sequence in enumerate(branch_actions):
                    value, first = _simulate_open_loop(
                        environment, snapshot, sequence, discount=discount
                    )
                    true_returns.append(value)
                    if index == 0:
                        baseline_transition = first
                assert baseline_transition is not None
                environment.restore(snapshot)
                single_noise = jnp.asarray(shared_noise)
                model_gradient = gradient_function(
                    jnp.asarray(host_action),
                    state.params.world_model,
                    belief,
                    jnp.asarray(suffix),
                    single_noise,
                    float(discount),
                    config,
                )
                simulator_gradient = _finite_difference_gradient(
                    environment,
                    snapshot,
                    host_action,
                    suffix,
                    epsilon=finite_difference_epsilon,
                    discount=discount,
                )
                model_observations.append(obs_samples[:, 0, 0])
                model_rewards.append(reward_samples[:, 0, 0])
                model_continuations.append(continuation_samples[:, 0, 0])
                target_observations.append(np.asarray(baseline_transition.observation))
                target_rewards.append(float(baseline_transition.reward))
                target_continuations.append(float(baseline_transition.continuation))
                predicted_rank_returns.append(predicted)
                simulator_rank_returns.append(np.asarray(true_returns))
                model_gradients.append(np.asarray(model_gradient))
                simulator_gradients.append(simulator_gradient)
                environment.restore(snapshot)
            transition = environment.step(host_action)
            observation = transition.observation
            previous_action = policy_action
            if transition.is_last:
                break
    finally:
        environment.close()
    if action_replay_max_error > 1e-5:
        raise ValueError(
            f"actor policy replay differs from retained actions by {action_replay_max_error}"
        )
    raw = {
        "sample_steps": sample_steps,
        "observation_samples": np.stack(model_observations, axis=1),
        "reward_samples": np.stack(model_rewards, axis=1),
        "continuation_samples": np.stack(model_continuations, axis=1),
        "target_observations": np.stack(target_observations),
        "target_rewards": np.asarray(target_rewards),
        "target_continuations": np.asarray(target_continuations),
        "observation_std": observation_std,
        "predicted_returns": np.stack(predicted_rank_returns),
        "simulator_returns": np.stack(simulator_rank_returns),
        "model_gradients": np.stack(model_gradients),
        "simulator_gradients": np.stack(simulator_gradients),
        "action_replay_max_abs_error": np.asarray(action_replay_max_error),
    }
    raw_directory = output_root / "cells" / cell["cell_id"]
    raw_directory.mkdir(parents=True, exist_ok=True)
    raw_path = raw_directory / "diagnostics.npz"
    np.savez_compressed(raw_path, **raw)
    row = {
        "task": cell["task"],
        "arm": cell["arm"],
        "cell_id": cell["cell_id"],
        "raw": {name: np.asarray(value).tolist() for name, value in raw.items()},
    }
    manifest = {
        "cell_id": cell["cell_id"],
        "task": cell["task"],
        "arm": cell["arm"],
        "world_model_seed": int(cell["world_model_seed"]),
        "actor_seed": int(cell["actor_seed"]),
        "candidate_id": cell["candidate_id"],
        "raw_path": str(raw_path.relative_to(output_root)),
        "raw_sha256": file_sha256(raw_path),
        "actor_checkpoint_sha256": result["checkpoint_sha256"],
    }
    return row, manifest


def run_diagnostics(arguments: argparse.Namespace) -> None:
    import jax
    from dreamer_imf_compare import matched_objective_benchmark as benchmark

    pilot_root = Path(arguments.pilot_root).resolve()
    output_path = Path(arguments.output).resolve()
    output_root = output_path.parent
    if output_root == pilot_root or pilot_root in output_root.parents:
        raise ValueError("diagnostic output must be outside the frozen pilot root")
    if not (pilot_root / "hpo_selection.json").is_file():
        raise ValueError("pilot finalization and HPO selection must complete before diagnostics")
    protocol = read_json(REPOSITORY / "matched_objective_protocol.json")
    matrix = read_json(pilot_root / "matrix.json")
    selection = read_json(pilot_root / "hpo_selection.json")
    selected = _candidate_cells(matrix, selection)
    sampled = _sample_cells(selected, arguments.cells_per_group)
    all_actor_records = [
        read_json(pilot_root / "actor" / cell["cell_id"] / "result.json")
        for cell in _stage_cells(matrix, "actor")
    ]
    actor_numbers = actor_number_summary(
        all_actor_records, expected_cells=len(_stage_cells(matrix, "actor"))
    )
    alignment_rows = []
    grouped_selected: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cell in selected:
        grouped_selected[(cell["task"], cell["arm"])].append(
            read_json(pilot_root / "actor" / cell["cell_id"] / "result.json")
        )
    for (task, arm), records in sorted(grouped_selected.items()):
        alignment_rows.append({"task": task, "arm": arm, **imagined_real_alignment(records)})

    sampler = jax.jit(
        benchmark.open_loop_samples_with_continuation, static_argnames=("config",)
    )
    gradient_function = _model_action_gradient_function()
    raw_rows = []
    manifest = []
    for index, cell in enumerate(sampled, start=1):
        print(
            f"diagnostic cell {index}/{len(sampled)} {cell['task']} "
            f"{cell['arm']} {cell['cell_id']}",
            flush=True,
        )
        row, manifest_row = _diagnose_actor_cell(
            pilot_root,
            matrix,
            cell,
            output_root=output_root,
            episode=arguments.episode,
            states_per_cell=arguments.states_per_cell,
            horizon=arguments.horizon,
            draws=arguments.draws,
            action_delta=arguments.action_delta,
            finite_difference_epsilon=arguments.finite_difference_epsilon,
            discount=arguments.discount,
            sampler=sampler,
            gradient_function=gradient_function,
        )
        raw_rows.append(row)
        manifest.append(manifest_row)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": arguments.pilot_source_commit,
        "pilot_root": str(pilot_root),
        "diagnostic_output_root": str(output_root),
        "config": {
            "episode": arguments.episode,
            "cells_per_task_arm": arguments.cells_per_group,
            "states_per_cell": arguments.states_per_cell,
            "counterfactual_horizon": arguments.horizon,
            "predictive_draws": arguments.draws,
            "action_delta": arguments.action_delta,
            "finite_difference_epsilon": arguments.finite_difference_epsilon,
            "discount": arguments.discount,
            "selection_sha256": selection["selection_sha256"],
            "diagnostic_source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
            ).strip(),
        },
        "sample_manifest": manifest,
        "actor_numbers": actor_numbers,
        "metrics": {
            "actor_visited_model_error": aggregate_metric_rows(
                raw_rows, "actor_visited_model_error"
            ),
            "imagined_real_alignment": alignment_rows,
            "action_ranking": aggregate_metric_rows(raw_rows, "action_ranking"),
            "gradient_fidelity": aggregate_metric_rows(raw_rows, "gradient_fidelity"),
        },
    }
    validate_report(report, forbidden_root=str(pilot_root))
    write_json_atomic(output_path, report)
    print(f"POLICY_ALIGNMENT_DIAGNOSTICS_COMPLETE {output_path}")


def snapshot_preservation(arguments: argparse.Namespace) -> None:
    pilot_root = Path(arguments.pilot_root).resolve()
    source_root = Path(arguments.source_root).resolve()
    matrix = read_json(pilot_root / "matrix.json")
    actor_cells = _stage_cells(matrix, "actor")
    prefix = int(arguments.prefix_count)
    if prefix < 0 or prefix > len(actor_cells):
        raise ValueError("preservation prefix is outside the actor matrix")
    manifest = []
    for index, cell in enumerate(actor_cells[:prefix]):
        result_path = pilot_root / "actor" / cell["cell_id"] / "result.json"
        if not result_path.is_file():
            raise ValueError(f"preservation prefix actor {index} is missing")
        manifest.append(f"{index}\0{cell['cell_id']}\0{file_sha256(result_path)}\n")
    marker_root = pilot_root / "cluster_stage_verification"
    output = {
        "schema_version": PRESERVATION_SCHEMA,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root, text=True
        ).splitlines(),
        "protocol_sha256": file_sha256(source_root / "dreamer_imf_comparison" / "matched_objective_protocol.json"),
        "matrix_sha256": file_sha256(pilot_root / "matrix.json"),
        "stage_markers": {
            path.name: file_sha256(path) for path in sorted(marker_root.glob("*.json"))
        },
        "actor_completed_prefix_count": prefix,
        "actor_completed_prefix_manifest_sha256": hashlib.sha256(
            "".join(manifest).encode()
        ).hexdigest(),
    }
    write_json_atomic(arguments.output, output)
    print("FROZEN_PILOT_SNAPSHOT_COMPLETE")


def verify_preservation(arguments: argparse.Namespace) -> None:
    before = read_json(arguments.before)
    after = read_json(arguments.after)
    if before.get("schema_version") != PRESERVATION_SCHEMA or after.get("schema_version") != PRESERVATION_SCHEMA:
        raise ValueError("preservation schema mismatch")
    for key in (
        "source_commit",
        "source_status",
        "protocol_sha256",
        "matrix_sha256",
        "stage_markers",
        "actor_completed_prefix_count",
        "actor_completed_prefix_manifest_sha256",
    ):
        if before.get(key) != after.get(key):
            raise ValueError(f"frozen pilot preservation mismatch in {key}")
    if before["source_status"]:
        raise ValueError("frozen source was already dirty before diagnostics")
    print("FROZEN_PILOT_PRESERVATION_VERIFIED")


def verify_report_command(arguments: argparse.Namespace) -> None:
    report = read_json(arguments.report)
    validate_report(report, forbidden_root=arguments.forbid_root)
    for row in report["sample_manifest"]:
        for key in ("raw_path", "raw_sha256", "actor_checkpoint_sha256"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("sample manifest does not authenticate raw evidence")
    print("POLICY_ALIGNMENT_REPORT_VERIFIED")


def verify_coverage_command(arguments: argparse.Namespace) -> None:
    report = read_json(arguments.report)
    verify_metric_coverage(
        report, metrics=arguments.metrics, tasks=arguments.tasks, arms=arguments.arms
    )
    print("POLICY_ALIGNMENT_COVERAGE_VERIFIED")


def verify_numbers(arguments: argparse.Namespace) -> None:
    report = read_json(arguments.report)
    validate_report(report)
    summary = report["actor_numbers"]
    if summary["completed_cells"] != summary["expected_cells"]:
        raise ValueError("actor numbers were captured before all pilot cells completed")
    expected_groups = {
        (task, arm)
        for task in ("dmc_pendulum_swingup", "dmc_reacher_easy")
        for arm in ARM_NAMES
    }
    rows = summary.get("groups", [])
    if {(row.get("task"), row.get("arm")) for row in rows} != expected_groups:
        raise ValueError("actor number summary does not cover both pilot tasks and arms")
    if sum(int(row["cells"]) for row in rows) != summary["completed_cells"]:
        raise ValueError("actor grouped cell counts do not reconcile")
    print("ACTOR_NUMBERS_VERIFIED")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--pilot-root", required=True)
    run.add_argument("--pilot-source-commit", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--cells-per-group", type=int, default=2)
    run.add_argument("--states-per-cell", type=int, default=4)
    run.add_argument("--episode", type=int, default=0)
    run.add_argument("--horizon", type=int, default=5)
    run.add_argument("--draws", type=int, default=8)
    run.add_argument("--action-delta", type=float, default=0.25)
    run.add_argument("--finite-difference-epsilon", type=float, default=0.05)
    run.add_argument("--discount", type=float, default=0.99)
    run.set_defaults(function=run_diagnostics)

    snapshot = commands.add_parser("snapshot-preservation")
    snapshot.add_argument("--pilot-root", required=True)
    snapshot.add_argument("--source-root", required=True)
    snapshot.add_argument("--prefix-count", required=True, type=int)
    snapshot.add_argument("--output", required=True)
    snapshot.set_defaults(function=snapshot_preservation)

    verify = commands.add_parser("verify-report")
    verify.add_argument("report")
    verify.add_argument("--forbid-root", required=True)
    verify.set_defaults(function=verify_report_command)

    coverage = commands.add_parser("verify-coverage")
    coverage.add_argument("report")
    coverage.add_argument("--metrics", nargs="+", required=True)
    coverage.add_argument("--tasks", nargs="+", required=True)
    coverage.add_argument("--arms", nargs="+", required=True)
    coverage.set_defaults(function=verify_coverage_command)

    preserve = commands.add_parser("verify-preservation")
    preserve.add_argument("before")
    preserve.add_argument("after")
    preserve.set_defaults(function=verify_preservation)

    numbers = commands.add_parser("verify-numbers")
    numbers.add_argument("report")
    numbers.set_defaults(function=verify_numbers)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
