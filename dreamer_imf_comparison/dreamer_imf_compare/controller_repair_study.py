"""Immutable, staged frozen-checkpoint controller repair and decision audit.

No dependency artifact is written. Final environments cannot run until the
diagnostic stage authenticates and its predeclared reference-mode rule is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import numpy as np

from . import mechanism_replication as m
from . import matched_objective_benchmark as b

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "dreamer_imf_comparison/controller_repair_protocol.json"
MODULE = "dreamer_imf_compare.controller_repair_study"
STAGES = ("preflight", "diagnostics", "evaluation")
ROLES = ("primer", "create", "replay")
FILES = {"result.json", "trace.npz"} | {
    f"{role}-{kind}.json" for role in ROLES for kind in ("receipt", "seal")
}
TRAIN_FILES = {"result.json", "checkpoint.pkl", "replay.npz", "schedules.pkl"}


def validate_protocol(p):
    if p != m.read(PROTOCOL):
        raise ValueError("protocol is not exact committed protocol")
    m.validate_controller_protocol(p["controller"])
    groups = [
        p["evaluation_environment_seeds"],
        p["preflight"]["evaluation_seeds"],
        p["diagnostics"]["calibration_environment_seeds"],
        p["diagnostics"]["validation_environment_seeds"],
    ]
    if any(len(set(g)) != len(g) for g in groups) or any(
        set(a) & set(c) for a, c in itertools.combinations(groups, 2)
    ):
        raise ValueError("environment seed groups overlap or repeat")
    if len(cells(p, "evaluation")) != p["expected_evaluation_cells"]:
        raise ValueError("evaluation count differs")


def cells(p, stage):
    pairs = [
        dict(training_index=i, task=t, world_model_seed=w)
        for i, (t, w) in enumerate(
            itertools.product(p["tasks"], p["world_model_seeds"])
        )
    ]
    if stage == "preflight":
        return [
            dict(index=i, **pairs[i * len(p["world_model_seeds"])])
            for i in range(len(p["tasks"]))
        ]
    if stage == "diagnostics":
        return [
            dict(index=i, actor_seed=p["diagnostics"]["actor_seed"], **pair)
            for i, pair in enumerate(pairs)
        ]
    if stage == "evaluation":
        return [
            dict(index=i, **pair, actor_seed=a, arm=arm)
            for i, (pair, a, arm) in enumerate(
                itertools.product(pairs, p["nested_actor_seeds"], p["arms"])
            )
        ]
    raise ValueError("unknown stage")


def dependency_index(dependency, p, *, verify_artifacts=True):
    """Authenticate report -> stage markers -> exact training files before pickle."""
    dependency = Path(dependency).resolve()
    dm = m.read(dependency / "manifest.json")
    report = m.read(dependency / "report.json")
    final = m.read(dependency / "verified.json")
    if (
        dm["source_commit"] != p["dependency_commit"]
        or m.digest(dm) != p["dependency_manifest_sha256"]
        or b.file_sha256(dependency / "report.json")
        != p["dependency_report_file_sha256"]
        or final
        != dict(
            cell={"stage": "final"},
            files={"report.json": p["dependency_report_file_sha256"]},
            manifest_sha256=m.digest(dm),
            source_commit=p["dependency_commit"],
        )
    ):
        raise ValueError("immutable dependency identity differs")
    expected = m.cells(dm["protocol"], "training")
    stage = report["stages"]["training"]
    if stage["manifest_sha256"] != m.digest(dm) or set(stage["cell_markers"]) != {
        str(c["index"]) for c in expected
    }:
        raise ValueError("dependency training stage incomplete")
    result = {}
    for c in expected:
        directory = m.cell_dir(dependency, "training", c["index"])
        mark = m.read(directory / "verified.json")
        if (
            b.file_sha256(directory / "verified.json")
            != stage["cell_markers"][str(c["index"])]
            or mark["cell"] != c
            or set(mark["files"]) != TRAIN_FILES
            or mark["manifest_sha256"] != m.digest(dm)
            or mark["source_commit"] != p["dependency_commit"]
        ):
            raise ValueError("dependency training marker differs")
        for name, sha in mark["files"].items():
            m.require_sha256(sha, "dependency artifact")
            if verify_artifacts and b.file_sha256(directory / name) != sha:
                raise ValueError("dependency training artifact differs")
        info = m.read(directory / "result.json")
        if info["cell"] != c or not m.finite(info):
            raise ValueError("dependency cell or finiteness differs")
        result[str(c["index"])] = dict(
            cell=c,
            files=mark["files"],
            marker_sha256=b.file_sha256(directory / "verified.json"),
        )
    if len(result) != len(p["tasks"]) * len(p["world_model_seeds"]):
        raise ValueError("dependency checkpoint matrix differs")
    return result


def manifest(root):
    value = m.read(Path(root) / "manifest.json")
    m.exact_fields(
        value,
        {
            "schema",
            "source_commit",
            "source_root",
            "protocol",
            "protocol_sha256",
            "dependency_root",
            "dependency_index",
            "created_unix",
        },
        "repair manifest",
    )
    validate_protocol(value["protocol"])
    if value["source_commit"] != m.clean_commit() or value[
        "protocol_sha256"
    ] != m.digest(value["protocol"]):
        raise ValueError("source/protocol manifest identity differs")
    if value["schema"] != "imf-controller-repair-v1" or value["source_root"] != str(
        SOURCE
    ):
        raise ValueError("manifest source path/schema differs")
    if value["dependency_index"] != dependency_index(
        value["dependency_root"], value["protocol"], verify_artifacts=False
    ):
        raise ValueError("dependency index no longer binds authenticated old report")
    return value


def register(root, dependency):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("fresh output required")
    p = m.read(PROTOCOL)
    validate_protocol(p)
    index = dependency_index(dependency, p)
    value = dict(
        schema="imf-controller-repair-v1",
        source_commit=m.clean_commit(),
        source_root=str(SOURCE),
        protocol=p,
        protocol_sha256=m.digest(p),
        dependency_root=str(Path(dependency).resolve()),
        dependency_index=index,
        created_unix=time.time(),
    )
    m.publish(root / "manifest.json", value)
    print("CONTROLLER_REPAIR_REGISTERED", m.digest(value), flush=True)


def authenticate_checkpoint(value, cell):
    entry = value["dependency_index"][str(cell["training_index"])]
    if any(cell[k] != entry["cell"][k] for k in ("task", "world_model_seed")):
        raise ValueError("checkpoint pairing differs")
    directory = m.cell_dir(value["dependency_root"], "training", cell["training_index"])
    if b.file_sha256(directory / "verified.json") != entry["marker_sha256"]:
        raise ValueError("dependency marker changed")
    for name, sha in entry["files"].items():
        if b.file_sha256(directory / name) != sha:
            raise ValueError("dependency artifact changed")
    return directory


def evaluate_simple(p, cell, training_dir, reference_mode, *, preflight=False):
    """Frozen actor or repaired endpoint controller; no training/update path."""
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import initial_state, observe_step, rebrac_actor
    from .dmc import DMCAdapter
    from .repaired_controllers import make_endpoint_planner

    model = m.load_checkpoint(training_dir)
    cfg = model["config"]
    frozen = b._tree_digest(
        {k: model[k] for k in ("world", "reward_world", "policies", "endpoints")}
    )
    actor = model["policies"][cell["actor_seed"]].actor
    actor_fn = jax.jit(rebrac_actor)
    is_baseline = cell["arm"] == "B0_frozen_rebrac"
    planner = (
        None
        if is_baseline
        else make_endpoint_planner(
            model,
            cell["actor_seed"],
            direct_any_step=cell["arm"].startswith("K1"),
            controller=p["controller"],
            reference_mode=reference_mode,
        )
    )

    @jax.jit
    def observe(obs, previous, belief, key):
        return observe_step(model["reward_world"], obs, previous, belief, key, cfg)[0]

    seeds = (
        p["preflight"]["evaluation_seeds"]
        if preflight
        else p["evaluation_environment_seeds"]
    )
    steps = (
        p["preflight"]["evaluation_steps"]
        if preflight
        else p["maximum_environment_steps"]
    )
    names = ["actions", "observations", "rewards", "continuations", "is_last"]
    component_names = [
        f"{which}_{field}"
        for which in ("proposal", "reference", "executed")
        for field in (
            "stage_reward",
            "terminal_value",
            "objective",
            "behavior_distance",
            "feasible",
            "violation",
        )
    ]
    names += component_names + [
        "used_fallback",
        "residual_norm",
        "executed_action_change",
        "objective_evaluations",
    ]
    sequences = {name: [] for name in names}
    lengths, returns = [], []
    seconds, timed_steps, warmed = 0.0, 0, False
    for seed in seeds:
        env = DMCAdapter(cell["task"], seed=seed, action_repeat=p["action_repeat"])
        episode = {name: [] for name in names}
        try:
            observation = env.reset()
            belief = initial_state(cfg, 1)
            previous = jnp.zeros((1, cfg.action_dim), jnp.float32)
            key = b.derive_jax_key(
                "controller-repair-paired",
                cell["task"],
                cell["world_model_seed"],
                cell["actor_seed"],
                seed,
            )
            for step in range(steps):
                current = jnp.asarray(observation[None], jnp.float32)
                noise = jax.random.normal(
                    jax.random.fold_in(key, 3 * step + 1),
                    (
                        p["controller"]["action_sequence_particles"],
                        p["controller"]["horizon"],
                        cfg.observation_dim,
                    ),
                )
                proposals = jax.random.normal(
                    jax.random.fold_in(key, 3 * step + 2),
                    (2, 4, p["controller"]["horizon"], cfg.action_dim),
                )
                posterior = jax.random.fold_in(key, 3 * step)
                if not warmed:
                    wb = (
                        belief
                        if is_baseline
                        else observe(current, previous, belief, posterior)
                    )
                    warm = (
                        actor_fn(actor, current)
                        if is_baseline
                        else planner(wb, current, noise, proposals)
                    )
                    jax.block_until_ready(warm)
                    warmed = True
                started = time.perf_counter()
                # Baseline uses observations only; belief is not part of its action.
                if not is_baseline:
                    belief = observe(current, previous, belief, posterior)
                if is_baseline:
                    action = actor_fn(actor, current)[0]
                    telemetry = {name: 0.0 for name in names[5:]}
                else:
                    planned = planner(belief, current, noise, proposals)
                    action = planned["executed_sequence"][0]
                    telemetry = {
                        f"{which}_{field}": planned[which][field]
                        for which in ("proposal", "reference", "executed")
                        for field in (
                            "stage_reward",
                            "terminal_value",
                            "objective",
                            "behavior_distance",
                            "feasible",
                            "violation",
                        )
                    }
                    telemetry.update(
                        used_fallback=planned["used_fallback"],
                        residual_norm=planned["residual_norm"],
                        executed_action_change=jnp.max(
                            jnp.abs(action - planned["reference_sequence"][0])
                        ),
                        objective_evaluations=planned["evaluations"],
                    )
                jax.block_until_ready(action)
                host = np.asarray(action, np.float32)
                seconds += time.perf_counter() - started
                timed_steps += 1
                transition = env.step(host)
                for name, item in dict(
                    actions=host,
                    observations=observation.copy(),
                    rewards=transition.reward,
                    continuations=transition.continuation,
                    is_last=transition.is_last,
                    **telemetry,
                ).items():
                    episode[name].append(np.asarray(item))
                observation = transition.observation
                previous = jnp.asarray(host[None])
                if transition.is_last:
                    break
        finally:
            env.close()
        lengths.append(len(episode["rewards"]))
        returns.append(float(np.sum(episode["rewards"])))
        for name in names:
            arr = np.asarray(episode[name])
            padded = np.zeros((steps, *arr.shape[1:]), dtype=arr.dtype)
            padded[: len(arr)] = arr
            sequences[name].append(padded)
    trace = {name: np.stack(arr) for name, arr in sequences.items()}
    trace["lengths"] = np.asarray(lengths, np.int32)
    if frozen != b._tree_digest(
        {k: model[k] for k in ("world", "reward_world", "policies", "endpoints")}
    ):
        raise ValueError("evaluation modified frozen checkpoint")
    return (
        returns,
        trace,
        dict(
            total_timed_seconds=seconds,
            timed_steps=timed_steps,
            mean_milliseconds_per_step=1000 * seconds / timed_steps,
            discarded_compile_warmup_steps=1,
        ),
    )


def evaluate(p, cell, training_dir, modes, *, preflight=False):
    if cell["arm"].startswith("A"):
        from . import actor_gap_roadmap_study as r

        model = m.load_checkpoint(training_dir)
        returns, trace, timing = r._run_flowmpc_arm(
            model["reward_world"],
            model["config"],
            model["policies"][cell["actor_seed"]],
            model["rebrac_config"],
            world_seed=cell["world_model_seed"],
            actor_seed=cell["actor_seed"],
            task=cell["task"],
            evaluation_seeds=(
                p["preflight"]["evaluation_seeds"]
                if preflight
                else p["evaluation_environment_seeds"]
            ),
            maximum_steps=(
                p["preflight"]["evaluation_steps"]
                if preflight
                else p["maximum_environment_steps"]
            ),
            trust=True,
            persistence="persistent",
            heldout_acceptance_enabled=cell["arm"].startswith("A3"),
            anchor_observations=model["anchors"],
        )
    else:
        mode = modes["direct" if cell["arm"].startswith("K1") else "recursive"]
        returns, trace, timing = evaluate_simple(
            p, cell, training_dir, mode, preflight=preflight
        )
    if not m.finite(trace) or not m.finite(returns):
        raise ValueError("nonfinite evaluation")
    mask = np.arange(trace["actions"].shape[1])[None] < trace["lengths"][:, None]
    telemetry = {
        name: float(np.mean(a[mask]))
        for name, a in trace.items()
        if a.shape == mask.shape and name not in ("rewards", "continuations", "is_last")
    }
    core = dict(
        **cell,
        episode_returns=returns,
        evaluation_environment_seeds=(
            p["preflight"]["evaluation_seeds"]
            if preflight
            else p["evaluation_environment_seeds"]
        ),
        action_saturation_fraction=float(
            np.mean(np.abs(trace["actions"][mask]) >= 0.95)
        ),
        telemetry=telemetry,
        realized_controller_config=p["controller"],
        reference_modes=modes,
        endpoint_objective_metrics_applicable=cell["arm"].startswith("K"),
        trace_sha256=b.array_sha256(trace),
        checkpoint_sha256=b.file_sha256(Path(training_dir) / "checkpoint.pkl"),
    )
    return core, trace, timing


def cell_output(root, stage, index, variant=None):
    path = m.cell_dir(root, stage, index)
    return path / variant if variant else path


def compute(root, stage, index, variant):
    value = manifest(root)
    p = value["protocol"]
    cell = cells(p, stage)[index]
    training = authenticate_checkpoint(value, cell)
    if stage == "diagnostics" or variant == "diagnostics":
        from .repair_diagnostics import evaluate_diagnostics

        diagnostic_cell = dict(cell, actor_seed=p["diagnostics"]["actor_seed"])
        core, trace, timing = evaluate_diagnostics(
            p, diagnostic_cell, training, preflight=stage == "preflight"
        )
    else:
        modes = {"recursive": "latent", "direct": "latent"}
        if stage == "evaluation":
            selected = verify_selection(root)
            modes = selected["reference_modes"]
        elif variant.endswith("endpoint"):
            modes = {"recursive": "endpoint", "direct": "endpoint"}
        if stage == "preflight":
            arm = variant.removesuffix("-endpoint")
            cell = dict(cell, actor_seed=p["nested_actor_seeds"][0], arm=arm)
        core, trace, timing = evaluate(
            p, cell, training, modes, preflight=stage == "preflight"
        )
    core.update(
        checkpoint_sha256=b.file_sha256(training / "checkpoint.pkl"),
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        trace_sha256=b.array_sha256(trace),
    )
    if not m.finite(core) or not m.finite(trace):
        raise ValueError("nonfinite compute output")
    return core, trace, timing


def role(root, stage, index, variant, mode, cache):
    from .cache_fingerprint import cache_tree_sha256

    output = cell_output(root, stage, index, variant)
    output.mkdir(parents=True, exist_ok=True)
    if (
        mode != "primer"
        and cache_tree_sha256(cache)
        != m.read(output / "primer-seal.json")["cache_sha256"]
    ):
        raise ValueError("cache changed before reader")
    started = time.perf_counter()
    core, trace, timing = compute(root, stage, index, variant)
    runtime = m.runtime()
    if mode == "create":
        m.publish(output / "result.json", core)
        with (output / "trace.npz").open("xb") as handle:
            np.savez_compressed(handle, **trace)
    elif mode == "replay":
        if m.digest(core) != m.digest(m.read(output / "result.json")):
            raise ValueError("repair semantic replay differs")
        m.exact_trace(trace, b.load_npz(output / "trace.npz"))
    m.publish(
        output / f"{mode}-receipt.json",
        dict(
            mode=mode,
            pid=os.getpid(),
            runtime=runtime,
            source_commit=core["source_commit"],
            manifest_sha256=core["manifest_sha256"],
            core_sha256=m.digest(core),
            trace_sha256=b.array_sha256(trace),
            checkpoint_sha256=core["checkpoint_sha256"],
            cache_sha256=cache_tree_sha256(cache),
            timing=timing,
            wall_seconds=time.perf_counter() - started,
        ),
    )


def sealed_compute(root, stage, index, variant=None):
    from .cache_fingerprint import cache_tree_sha256

    value = manifest(root)
    output = cell_output(root, stage, index, variant)
    cache = (
        Path(root)
        / "caches"
        / f"job-{os.environ['SLURM_JOB_ID']}-index-{index}-{variant or stage}"
    )
    cache.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        JAX_COMPILATION_CACHE_DIR=str(cache),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="-1",
        JAX_RAISE_PERSISTENT_CACHE_ERRORS="true",
    )
    fingerprint = None
    receipts = []
    for mode in ROLES:
        cmd = [
            sys.executable,
            "-m",
            MODULE,
            "role",
            "--root",
            str(root),
            "--stage",
            stage,
            "--index",
            str(index),
            "--mode",
            mode,
            "--cache",
            str(cache),
        ]
        if variant:
            cmd += ["--variant", variant]
        subprocess.run(cmd, check=True, env=env)
        receipt = m.read(output / f"{mode}-receipt.json")
        post = cache_tree_sha256(cache)
        if receipt["cache_sha256"] != post or (
            fingerprint is not None and fingerprint != post
        ):
            raise ValueError("post-exit cache differs")
        fingerprint = post
        receipts.append(receipt)
        m.publish(
            output / f"{mode}-seal.json",
            dict(
                cache_sha256=post,
                receipt_sha256=b.file_sha256(output / f"{mode}-receipt.json"),
            ),
        )
    if (
        len({r["pid"] for r in receipts}) != 3
        or len({m.digest(r["runtime"]) for r in receipts}) != 1
    ):
        raise ValueError("independent process/runtime identity differs")
    mark = dict(
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        cell=cells(value["protocol"], stage)[index],
        stage=stage,
        variant=variant,
        strict_bitwise_replay=True,
        cache_sha256=fingerprint,
        files={name: b.file_sha256(output / name) for name in sorted(FILES)},
    )
    m.publish(output / "verified.json", mark)
    verify_cell(root, stage, index, variant)
    print("CONTROLLER_REPAIR_CELL_VERIFIED", stage, index, variant, flush=True)


def verify_cell(root, stage, index, variant=None):
    value = manifest(root)
    cell = cells(value["protocol"], stage)[index]
    output = cell_output(root, stage, index, variant)
    mark = m.read(output / "verified.json")
    expected = dict(
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        cell=cell,
        stage=stage,
        variant=variant,
        strict_bitwise_replay=True,
    )
    if (
        m.digest({k: mark.get(k) for k in expected}) != m.digest(expected)
        or mark.get("strict_bitwise_replay") is not True
    ):
        raise ValueError("strict cell marker identity differs")
    if set(mark.get("files", {})) != FILES:
        raise ValueError("exact strict file set required")
    m.exact_fields(mark, set(expected) | {"files", "cache_sha256"}, "repair marker")
    if {path.name for path in output.iterdir()} != FILES | {"verified.json"}:
        raise ValueError("unregistered files in cell output")
    for name, sha in mark["files"].items():
        m.require_sha256(sha, "repair file")
        if b.file_sha256(output / name) != sha:
            raise ValueError("bound repair artifact changed")
    result, trace = m.read(output / "result.json"), b.load_npz(output / "trace.npz")
    m.require_sha256(mark["cache_sha256"], "repair cache")
    for key in ("trace_sha256", "checkpoint_sha256", "manifest_sha256"):
        m.require_sha256(result.get(key), key)
    checkpoint_sha = value["dependency_index"][str(cell["training_index"])]["files"][
        "checkpoint.pkl"
    ]
    if (
        not m.finite(result)
        or not m.finite(trace)
        or result["trace_sha256"] != b.array_sha256(trace)
        or result["checkpoint_sha256"] != checkpoint_sha
        or result["manifest_sha256"] != m.digest(value)
        or result["source_commit"] != value["source_commit"]
    ):
        raise ValueError("result/trace/dependency binding differs")
    if m.digest({k: result.get(k) for k in cell}) != m.digest(cell):
        raise ValueError("result cell differs")
    if stage == "evaluation" or (stage == "preflight" and variant != "diagnostics"):
        m.exact_fields(
            result,
            set(cell)
            | {
                "actor_seed",
                "arm",
                "episode_returns",
                "evaluation_environment_seeds",
                "action_saturation_fraction",
                "telemetry",
                "realized_controller_config",
                "reference_modes",
                "endpoint_objective_metrics_applicable",
                "trace_sha256",
                "checkpoint_sha256",
                "source_commit",
                "manifest_sha256",
            },
            "repair evaluation result",
        )
        for returned in result["episode_returns"]:
            m.require_number(returned, "episode return")
        expected_arm = (
            cell["arm"] if stage == "evaluation" else variant.removesuffix("-endpoint")
        )
        expected_seeds = (
            value["protocol"]["evaluation_environment_seeds"]
            if stage == "evaluation"
            else value["protocol"]["preflight"]["evaluation_seeds"]
        )
        if (
            result.get("arm") != expected_arm
            or result.get("evaluation_environment_seeds") != expected_seeds
            or len(result.get("episode_returns", [])) != len(expected_seeds)
            or result.get("realized_controller_config")
            != value["protocol"]["controller"]
        ):
            raise ValueError("realized evaluation arm/seeds/config differs")
        modes = (
            m.read(Path(root) / "selection.json")["reference_modes"]
            if stage == "evaluation"
            else dict(
                recursive="endpoint" if variant.endswith("-endpoint") else "latent",
                direct="endpoint" if variant.endswith("-endpoint") else "latent",
            )
        )
        expected_actor = (
            cell["actor_seed"]
            if stage == "evaluation"
            else value["protocol"]["nested_actor_seeds"][0]
        )
        if set(modes) != {"recursive", "direct"} or any(
            mode not in value["protocol"]["diagnostics"]["reference_modes"]
            for mode in modes.values()
        ):
            raise ValueError("unregistered reference mode")
        if (
            result.get("reference_modes") != modes
            or result.get("actor_seed") != expected_actor
        ):
            raise ValueError("reference mode or actor identity differs")
        needed = {
            "actions",
            "observations",
            "rewards",
            "continuations",
            "is_last",
            "lengths",
        }
        if not needed <= set(trace) or trace["lengths"].shape != (len(expected_seeds),):
            raise ValueError("evaluation trace schema differs")
        maxsteps = (
            value["protocol"]["maximum_environment_steps"]
            if stage == "evaluation"
            else value["protocol"]["preflight"]["evaluation_steps"]
        )
        if (
            trace["rewards"].shape != (len(expected_seeds), maxsteps)
            or trace["lengths"].dtype.kind not in "iu"
            or trace["actions"].shape[:2] != trace["rewards"].shape
            or np.any(trace["lengths"] < 1)
            or np.any(trace["lengths"] > maxsteps)
        ):
            raise ValueError("evaluation trace dimensions/length differ")
        mask = np.arange(maxsteps)[None] < trace["lengths"][:, None]
        recomputed = [
            float(np.sum(row[: int(length)]))
            for row, length in zip(trace["rewards"], trace["lengths"])
        ]
        if not np.allclose(recomputed, result["episode_returns"], rtol=0, atol=1e-8):
            raise ValueError("reported returns differ from trace rewards")
        if np.any(np.abs(trace["actions"][mask]) > 1):
            raise ValueError("executed action out of bounds")
    else:
        from .repair_diagnostics import (
            resolve_settings,
            summarize_trace,
            validate_diagnostic_trace,
        )

        settings = resolve_settings(value["protocol"], preflight=stage == "preflight")
        m.exact_fields(
            result,
            set(cell)
            | {
                "actor_seed",
                "schema",
                "preflight",
                "realized_settings",
                "realized_controller",
                "discount",
                "checkpoint_sha256",
                "replay_file_sha256",
                "frozen_model_sha256",
                "trace_sha256",
                "continuation_semantics",
                "model_terminal_semantics",
                "diagnostic_scope",
                "trace_codes",
                "training_support",
                "baseline",
                "summary",
                "source_commit",
                "manifest_sha256",
            },
            "diagnostic result",
        )
        if (
            result.get("preflight") is not (stage == "preflight")
            or result.get("actor_seed") != settings["actor_seed"]
            or result.get("discount") != 0.99
            or result.get("realized_controller") != value["protocol"]["controller"]
        ):
            raise ValueError("realized diagnostic preflight/actor/controller differs")
        validate_diagnostic_trace(trace, settings)
        if result["realized_settings"] != settings or result[
            "summary"
        ] != summarize_trace(trace):
            raise ValueError("diagnostic trace/summary/config differs")
    receipts = [m.read(output / f"{mode}-receipt.json") for mode in ROLES]
    if (
        len({r["pid"] for r in receipts}) != 3
        or len({m.digest(r["runtime"]) for r in receipts}) != 1
    ):
        raise ValueError("process/runtime identity differs")
    for mode, receipt in zip(ROLES, receipts):
        m.exact_fields(
            receipt,
            {
                "mode",
                "pid",
                "runtime",
                "source_commit",
                "manifest_sha256",
                "core_sha256",
                "trace_sha256",
                "checkpoint_sha256",
                "cache_sha256",
                "timing",
                "wall_seconds",
            },
            "repair receipt",
        )
        m.validate_runtime(receipt["runtime"])
        m.require_number(receipt["wall_seconds"], "role wall seconds")
        for key in (
            "manifest_sha256",
            "core_sha256",
            "trace_sha256",
            "checkpoint_sha256",
            "cache_sha256",
        ):
            m.require_sha256(receipt[key], key)
        if type(receipt["pid"]) is not int or receipt["pid"] <= 0:
            raise ValueError("invalid role process identity")
        seal = m.read(output / f"{mode}-seal.json")
        if (
            receipt["mode"] != mode
            or receipt["manifest_sha256"] != m.digest(value)
            or receipt["source_commit"] != value["source_commit"]
            or receipt["checkpoint_sha256"] != checkpoint_sha
            or not m.finite(receipt)
            or receipt["wall_seconds"] <= 0
            or receipt["cache_sha256"] != mark["cache_sha256"]
            or seal
            != dict(
                cache_sha256=mark["cache_sha256"],
                receipt_sha256=b.file_sha256(output / f"{mode}-receipt.json"),
            )
        ):
            raise ValueError("role receipt/seal differs")
        # The discarded compiler writer is not a scientific equality oracle.
        if mode != "primer" and (
            receipt["core_sha256"] != m.digest(result)
            or receipt["trace_sha256"] != result["trace_sha256"]
        ):
            raise ValueError("strict reader result binding differs")
    return mark


def variants(p):
    return (
        ["diagnostics"]
        + p["arms"]
        + ["K0_repaired_recursive-endpoint", "K1_repaired_direct-endpoint"]
    )


def verify_stage(root, stage):
    value = manifest(root)
    p = value["protocol"]
    expected = cells(p, stage)
    submission = m.read(Path(root) / "submissions" / f"{stage}.json")
    if (
        submission["manifest_sha256"] != m.digest(value)
        or submission["stage"] != stage
        or submission["count"] != len(expected)
    ):
        raise ValueError("submission binding differs")
    released = m.read(Path(root) / "submissions" / f"{stage}.released.json")
    intent = m.read(Path(root) / "submissions" / f"{stage}.intent.json")
    if (
        intent["manifest_sha256"] != m.digest(value)
        or intent["stage"] != stage
        or intent["count"] != len(expected)
        or intent["script_sha256"]
        != hashlib.sha256(intent["script"].encode()).hexdigest()
        or submission["script_sha256"] != intent["script_sha256"]
    ):
        raise ValueError("immutable submission intent differs")
    if released["job_id"] != submission["job_id"]:
        raise ValueError("release identity differs")
    acct = m.accounting(submission["job_id"], len(expected))
    found = {x.name for x in (Path(root) / stage).iterdir() if x.is_dir()}
    if found != {f"{c['index']:03d}" for c in expected}:
        raise ValueError("stage directories differ")
    marks = {}
    runtimes = set()
    for cell in expected:
        for variant in variants(p) if stage == "preflight" else [None]:
            verify_cell(root, stage, cell["index"], variant)
            directory = cell_output(root, stage, cell["index"], variant)
            marks[f"{cell['index']}:{variant or stage}"] = b.file_sha256(
                directory / "verified.json"
            )
            runtimes.add(m.digest(m.read(directory / "create-receipt.json")["runtime"]))
    if len(runtimes) != 1:
        raise ValueError("stage device/runtime differs")
    result = dict(
        stage=stage,
        manifest_sha256=m.digest(value),
        cell_markers=marks,
        accounting=acct,
        runtime_sha256=next(iter(runtimes)),
    )
    if stage != "preflight":
        prior = m.read(Path(root) / "verified/preflight.json")
        if prior["runtime_sha256"] != result["runtime_sha256"]:
            raise ValueError("runtime differs from preflight")
    path = Path(root) / "verified" / f"{stage}.json"
    if path.exists():
        retained = m.read(path)
        m.exact_fields(retained, result, "repair stage certificate")
        m.validate_accounting(
            retained["accounting"], submission["job_id"], len(expected)
        )
        if m.digest(retained) != m.digest(result):
            raise ValueError("stage certificate changed")
    else:
        m.publish(path, result)
    print("CONTROLLER_REPAIR_STAGE_VERIFIED", stage, flush=True)
    return result


def choose_reference_modes(records, p):
    if {(r["task"], r["world_model_seed"]) for r in records} != set(
        itertools.product(p["tasks"], p["world_model_seeds"])
    ) or len(records) != len(p["tasks"]) * len(p["world_model_seeds"]):
        raise ValueError("incomplete diagnostic matrix")
    selection = p["selection"]
    modes, evidence = {}, {}
    for family in p["diagnostics"]["families"]:

        def average(split, mode, field, task=None):
            rows = [r for r in records if task is None or r["task"] == task]
            vals = [r["summary"][split][family][mode][field] for r in rows]
            if not m.finite(vals):
                raise ValueError("nonfinite mechanism selection metric")
            return float(np.mean(vals))

        cal_latent = average("calibration", "latent", "reference_feasible_fraction")
        cal_improvement = (
            average("calibration", "endpoint", "reference_feasible_fraction")
            - cal_latent
        )
        validation_feasibility = average(
            "validation", "endpoint", "reference_feasible_fraction"
        ) - average("validation", "latent", "reference_feasible_fraction")
        # Compare executed levels at matched starts, not gains relative to two
        # different reference plans, which would confound the intervention.
        task_gains = {
            t: average("validation", "endpoint", "real_executed_objective_mean", t)
            - average("validation", "latent", "real_executed_objective_mean", t)
            for t in p["tasks"]
        }
        gain = float(np.mean(list(task_gains.values())))
        eligible = (
            1 - cal_latent
            >= selection["calibration_latent_infeasible_fraction_at_least"]
            and cal_improvement
            >= selection["calibration_feasibility_improvement_at_least"]
            and validation_feasibility
            >= selection["validation_feasibility_improvement_at_least"]
            and gain
            > selection["validation_real_executed_gain_improvement_strictly_above"]
            and sum(x > 0 for x in task_gains.values())
            >= selection["validation_tasks_with_positive_gain_improvement_at_least"]
        )
        modes[family] = "endpoint" if eligible else "latent"
        evidence[family] = dict(
            calibration_latent_feasibility=cal_latent,
            calibration_feasibility_improvement=cal_improvement,
            validation_feasibility_improvement=validation_feasibility,
            validation_real_gain_improvement=gain,
            validation_task_gain_improvements=task_gains,
            endpoint_reference_eligible=eligible,
        )
    return dict(
        reference_modes=modes,
        evidence=evidence,
        interpretation="Bounded diagnostic-selected interface repair, not confirmation; no unsupported reward/critic/world retraining.",
    )


def selection_payload(root):
    value = manifest(root)
    p = value["protocol"]
    stage = verify_stage(root, "diagnostics")
    records = [
        m.read(m.cell_dir(root, "diagnostics", c["index"]) / "result.json")
        for c in cells(p, "diagnostics")
    ]
    return dict(
        **choose_reference_modes(records, p),
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        diagnostic_markers=stage["cell_markers"],
    )


def select(root):
    payload = selection_payload(root)
    path = Path(root) / "selection.json"
    if path.exists():
        if m.digest(m.read(path)) != m.digest(payload):
            raise ValueError("selection differs")
    else:
        m.publish(path, payload)
    print(
        "CONTROLLER_REPAIR_SELECTION_VERIFIED", payload["reference_modes"], flush=True
    )
    return payload


def verify_selection(root):
    payload = selection_payload(root)
    if m.digest(m.read(Path(root) / "selection.json")) != m.digest(payload):
        raise ValueError("unsealed or changed selection")
    return payload


def worker(root, stage, index):
    value = manifest(root)
    submission = m.read(Path(root) / "submissions" / f"{stage}.json")
    if (
        submission["job_id"] != os.environ.get("SLURM_ARRAY_JOB_ID")
        or submission["manifest_sha256"] != m.digest(value)
        or str(index) != os.environ.get("SLURM_ARRAY_TASK_ID")
    ):
        raise ValueError("worker scheduler/manifest identity differs")
    if stage != "preflight":
        verify_stage(root, STAGES[STAGES.index(stage) - 1])
    if stage == "evaluation":
        verify_selection(root)
    directory = m.cell_dir(root, stage, index)
    directory.mkdir(parents=True, exist_ok=False)
    for variant in variants(value["protocol"]) if stage == "preflight" else [None]:
        sealed_compute(root, stage, index, variant)
    print("CONTROLLER_REPAIR_WORKER_COMPLETE", stage, index, flush=True)


def launch(root, stage):
    root = Path(root).resolve()
    value = manifest(root)
    p = value["protocol"]
    if stage != "preflight":
        verify_stage(root, STAGES[STAGES.index(stage) - 1])
    if stage == "evaluation":
        verify_selection(root)
    receipt = root / "submissions" / f"{stage}.json"
    intent = root / "submissions" / f"{stage}.intent.json"
    if receipt.exists() or intent.exists():
        raise FileExistsError(
            "already submitted or uncertain; inspect, never duplicate"
        )
    count = len(cells(p, stage))
    settings = p["execution"]
    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "module purge",
            "module load Python/3.12.3-GCCcore-13.3.0",
            "source /work2/ci72buri-dreamer_imf_neurips/venv-cuda12/bin/activate",
            "export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORM_NAME=gpu JAX_ENABLE_X64=0 MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false",
            f"export PYTHONPATH={shlex.quote(str(SOURCE / 'imf_dreamer_jax/src'))}:{shlex.quote(str(SOURCE / 'dreamer_imf_comparison'))}",
            f'python -m {MODULE} worker --root {shlex.quote(str(root))} --stage {stage} --index "$SLURM_ARRAY_TASK_ID"',
        ]
    )
    m.publish(
        intent,
        dict(
            stage=stage,
            count=count,
            manifest_sha256=m.digest(value),
            script=script,
            script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        ),
    )
    (root / "logs").mkdir(exist_ok=True)
    cmd = [
        "sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        "--account=dep_inin_dat",
        f"--partition={settings['gpu_partition']}",
        f"--gres=gpu:{settings['gpu_type']}:1",
        f"--cpus-per-task={settings['cpus']}",
        f"--mem={settings['memory_gb']}G",
        f"--time={settings['time_limit']}",
        f"--array=0-{count-1}%{settings['array_concurrency']}",
        f"--job-name=imf-repair-{stage}",
        f"--output={root}/logs/{stage}-%A_%a.out",
        f"--error={root}/logs/{stage}-%A_%a.err",
    ]
    response = subprocess.check_output(cmd, input=script, text=True).strip()
    job = response.split(";")[0]
    if not re.fullmatch(r"[0-9]+", job):
        raise RuntimeError("uncertain scheduler response; preserve intent")
    m.publish(
        receipt,
        dict(
            stage=stage,
            count=count,
            job_id=job,
            manifest_sha256=m.digest(value),
            script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        ),
    )
    subprocess.run(["scontrol", "release", job], check=True)
    m.publish(
        root / "submissions" / f"{stage}.released.json",
        dict(job_id=job, released_unix=time.time()),
    )
    print("CONTROLLER_REPAIR_SUBMITTED", stage, job, flush=True)


def finalize(root):
    from .mechanism_replication_analysis import analyze

    root = Path(root)
    value = manifest(root)
    p = value["protocol"]
    stages = {stage: verify_stage(root, stage) for stage in STAGES}
    selection = verify_selection(root)
    records = [
        m.read(m.cell_dir(root, "evaluation", c["index"]) / "result.json")
        for c in cells(p, "evaluation")
    ]
    report = analyze(records, p)
    report.update(
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        dependency_commit=p["dependency_commit"],
        selection=selection,
        stages=stages,
        records=records,
        diagnostics=[
            m.read(m.cell_dir(root, "diagnostics", c["index"]) / "result.json")
            for c in cells(p, "diagnostics")
        ],
        limitations=p["limitations"],
        runtime=dict(
            allocation_gpu_seconds_by_stage={
                stage: sum(int(r[3]) for r in cert["accounting"]["records"])
                for stage, cert in stages.items()
            },
            evaluations={
                str(c["index"]): {
                    mode: m.read(
                        m.cell_dir(root, "evaluation", c["index"])
                        / f"{mode}-receipt.json"
                    )["wall_seconds"]
                    for mode in ROLES
                }
                for c in cells(p, "evaluation")
            },
            controller_timing={
                str(c["index"]): m.read(
                    m.cell_dir(root, "evaluation", c["index"]) / "create-receipt.json"
                )["timing"]
                for c in cells(p, "evaluation")
            },
        ),
    )
    if not m.finite(report):
        raise ValueError("nonfinite final report")
    if (root / "report.json").exists():
        if m.digest(m.read(root / "report.json")) != m.digest(report):
            raise ValueError("final report differs")
    else:
        m.publish(root / "report.json", report)
    marker = dict(
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        report_sha256=b.file_sha256(root / "report.json"),
        selection_sha256=b.file_sha256(root / "selection.json"),
    )
    if (root / "verified.json").exists():
        if m.digest(m.read(root / "verified.json")) != m.digest(marker):
            raise ValueError("final marker differs")
    else:
        m.publish(root / "verified.json", marker)
    print("CONTROLLER_REPAIR_FINAL_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "register",
            "launch",
            "worker",
            "role",
            "verify",
            "select",
            "finalize",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dependency", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--variant")
    parser.add_argument("--mode", choices=ROLES)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        register(args.root, args.dependency)
    elif args.command == "launch":
        launch(args.root, args.stage)
    elif args.command == "worker":
        worker(args.root, args.stage, args.index)
    elif args.command == "role":
        role(args.root, args.stage, args.index, args.variant, args.mode, args.cache)
    elif args.command == "verify":
        verify_stage(args.root, args.stage)
    elif args.command == "select":
        select(args.root)
    else:
        finalize(args.root)


if __name__ == "__main__":
    main()
