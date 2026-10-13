"""Matched posterior-reward online continuation with bounded real interaction.

Only run_cell/evaluate step environments. Preflight and probes use retained data.
All experiment artifacts are exclusive-create; historical dependencies are read-only.
"""

from __future__ import annotations

import argparse
import json
import pickle
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

from . import dreamer_ablation_runner as base
from .parallel_collection import _write_npz
from .reward_exploration_runner import (
    BudgetedReacher,
    RawRows,
    actual_actions,
    current_policy,
    config_for,
    group_deltas,
    spaces,
    retained_batch,
    preflight_batches,
)
from .posterior_reward_protocol import (
    read,
    sha,
    write_json_exclusive as write,
    write_bytes_exclusive,
    cell_for_index,
    cell_directory,
    eval_seeds,
)

MILESTONES = (0, 40000, 80000)
_COMPILATION = {}


def start_compilation_monitor():
    """Record JAX's measured compiler stages, not first-update wall time."""
    import jax

    def listener(event, seconds, **metadata):
        if event.startswith("/jax/core/compile/"):
            record = _COMPILATION.setdefault(event, dict(seconds=0.0, events=0))
            record["seconds"] += float(seconds)
            record["events"] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)


def compilation_evidence():
    return dict(
        events={k: dict(v) for k, v in _COMPILATION.items()},
        note="JAX 0.4.33 compiler stage durations; backend event includes cache retrieval if used; do not sum overlapping stages",
    )


def save_pickle(path, value):
    write_bytes_exclusive(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def load_continuation(agent, parent_cell, rmt_full, seed):
    from .parallel_frozen import parameter_digest
    from .posterior_reward_agent import import_rmt_heads

    with (Path(parent_cell) / "checkpoint_200000.pkl").open("rb") as stream:
        parent = pickle.load(stream)
    initialized = agent.save()
    params = dict(initialized["params"])
    for key, value in parent["params"].items():
        if key not in params or params[key].shape != value.shape:
            raise ValueError("parent namespace/shape incompatible: " + key)
        params[key] = value.copy()
    exported = import_rmt_heads(rmt_full, seed - 901, params)
    if any(k in parent["params"] or k not in params for k in exported):
        raise ValueError("readout import touches original or unknown parameters")
    for key, value in exported.items():
        if params[key].shape != np.asarray(value).shape:
            raise ValueError("readout import shape mismatch")
        params[key] = np.asarray(value, params[key].dtype)
    agent.load(dict(params=params, counters=parent["counters"]))
    loaded = agent.save()
    original = {k: loaded["params"][k] for k in parent["params"]}
    if parameter_digest(original) != parameter_digest(parent["params"]):
        raise ValueError("original parameters/optimizer changed during import")
    return dict(
        parent_parameter_digest=parameter_digest(original),
        counters=parent["counters"],
        imported_auxiliary_keys=sorted(exported),
        original_keys=sorted(original),
        auxiliary_parameter_digest=parameter_digest(
            {k: loaded["params"][k] for k in exported}
        ),
        readout_seed=seed - 901,
    )


def make_agent(output, parent_cell, directory, arm, seed, protocol):
    config, elements, main = config_for(
        parent_cell, directory, seed, protocol["upstream_path"]
    )
    from .posterior_reward_agent import install
    from .staged_dynamics import set_controls
    from .conditional_schedule import CONTROLS
    from .posterior_reward_protocol import read

    obs, act = spaces(elements)
    agent = install(arm)(obs, act, base._agent_config(config, elements))
    manifest = read(Path(output) / "manifest.json")
    binding = load_continuation(agent, parent_cell, manifest["rmt"]["path"], seed)
    set_controls(agent, CONTROLS)
    config.save(str(directory / "config.yaml"))
    base._finite(agent.params, "loaded continuation")
    return agent, config, main, binding


def runtime_evidence(agent):
    result = base._runtime_telemetry(agent, "cuda")
    fixed = ("input_mean", "input_std")
    auxiliary = {
        prefix: sum(
            int(x.size)
            for k, x in agent.params.items()
            if k.startswith(prefix + "/") and k.rsplit("/", 1)[-1] not in fixed
        )
        for prefix in ("reward_teacher", "reward_student")
    }
    result.update(
        auxiliary_trainable_parameters_by_module=auxiliary,
        total_trainable_parameters=result["model_parameter_count"]
        + sum(auxiliary.values()),
        resident_parameter_and_optimizer_elements=sum(
            int(x.size) for x in agent.params.values()
        ),
        note="teacher uses real posterior information only, never imagined control; policy diagnostics evaluate all three readouts",
    )
    return result


def evaluate(agent, main, config, seed, milestone, directory, budget):
    begun, episodes, latencies = time.monotonic(), [], []
    with current_policy(agent):
        for ep, envseed in enumerate(eval_seeds(seed, milestone)):
            base_env = BudgetedReacher(envseed, budget, "evaluate")
            env = main.wrap_env(base_env, config)
            agent.n_actions.value = int(
                np.random.SeedSequence(
                    [20261014, seed, 821, milestone, ep]
                ).generate_state(1)[0]
            )
            rows = []
            try:
                obs = env.step(dict(reset=True, action=np.zeros(2, np.float32)))
                carry = agent.init_policy(1)
                previous = np.zeros(2, np.float32)
                while True:
                    tick = time.perf_counter()
                    carry, action, outs = agent.policy(
                        carry,
                        {k: np.asarray(v)[None] for k, v in obs.items()},
                        mode="eval",
                    )
                    carry, clipped = actual_actions(carry, action["action"])
                    elapsed = time.perf_counter() - tick
                    if len(rows) >= 2:
                        latencies.append(elapsed)
                    rows.append(
                        {
                            **obs,
                            "action": clipped[0],
                            "previous_action": previous,
                            **{
                                k: np.asarray(v)[0]
                                for k, v in outs.items()
                                if k.startswith("log/")
                            },
                        }
                    )
                    if obs["is_last"]:
                        break
                    previous = clipped[0]
                    obs = env.step(dict(reset=False, action=previous))
                path = directory / f"eval-{milestone}-{ep}.npz"
                _write_npz(path, {k: np.stack([r[k] for r in rows]) for k in rows[0]})
                episodes.append(
                    dict(
                        seed=envseed,
                        path=path.name,
                        sha256=sha(path),
                        return_=float(sum(float(r["reward"]) for r in rows)),
                        native_steps=base_env.native_steps,
                    )
                )
            finally:
                env.close()
    return dict(
        milestone=milestone,
        episodes=episodes,
        mean_return=float(np.mean([r["return_"] for r in episodes])),
        policy_latency_seconds=dict(
            count=len(latencies),
            median=float(np.median(latencies)),
            p95=float(np.percentile(latencies, 95)),
            includes_host_transfer=True,
            includes_all_readout_diagnostics=True,
            warmup_calls_excluded_per_episode=2,
        ),
        wall_seconds=time.monotonic() - begun,
        mode="task-policy; real environment rewards only; never enters replay",
    )


def _setup(protocol):
    base._imports(protocol)
    from embodied.jax import internal

    internal.setup(
        platform="cuda",
        compute_dtype="bfloat16",
        transfer_guard=False,
        prealloc=False,
        compilation_cache=False,
    )


def _proxy(directory, arm, seed, protocol, checkpoint):
    import jax
    from .posterior_reward_agent import install

    config, elements, _ = config_for(
        directory, directory, seed, protocol["upstream_path"]
    )
    obs, act = spaces(elements)
    cls = install(arm)
    model = object.__new__(cls)
    model.__init__(obs, act, base._agent_config(config, elements))
    with Path(checkpoint).open("rb") as stream:
        saved = pickle.load(stream)
    # Prediction never needs optimizer accumulators. Keep them in the immutable
    # checkpoint, not resident on the GPU alongside each prediction batch.
    params = jax.device_put(
        {
            k: v
            for k, v in saved["params"].items()
            if not k.startswith(("opt/", "reward_aux_opt/"))
        }
    )
    return types.SimpleNamespace(model=model, params=params, obs_space=obs), saved


def _probe_fixture(batch):
    """Synthetic bounded history tiled from a minibatch; not scientific evidence."""
    result = {}
    keys = (
        *("position", "to_target", "velocity"),
        "reward",
        "is_first",
        "is_last",
        "is_terminal",
        "action",
    )
    for key in keys:
        value = batch[key][:10]
        reps = (1, 8) + (1,) * (value.ndim - 2)
        result[key] = np.tile(value, reps)[:, :501].copy()
    # Legacy minibatches can retain unbounded Gaussian policy proposals. This
    # discarded fixture exercises the executed-action history interface used by
    # evaluate(), so mirror its clipping before reconstructing incoming actions.
    # Never rewrite the source batch or relax validation of scientific histories.
    if not np.isfinite(result["action"]).all():
        raise ValueError("nonfinite preflight fixture action")
    result["action"] = np.clip(result["action"], -1, 1)
    result["is_first"][:] = False
    result["is_first"][:, 0] = True
    result["is_last"][:] = False
    result["is_last"][:, -1] = True
    result["is_terminal"][:] = False
    result["reward"][:, 0] = 0
    result["previous_action"] = np.concatenate(
        [np.zeros((10, 1, 2), np.float32), result["action"][:, :-1]], 1
    )
    result["episode_seed"] = np.arange(10, dtype=np.uint32)
    return result


def compare_arrays(path, arrays):
    from .parallel_runner import load_npz

    saved = load_npz(path)
    if set(saved) != set(arrays):
        raise ValueError("probe fields differ")
    for key, value in arrays.items():
        if (
            saved[key].dtype != value.dtype
            or saved[key].shape != value.shape
            or not np.array_equal(saved[key], value)
        ):
            raise ValueError("exact prediction replay differs: " + key)


def preflight_arm(output, parent_cell, arm, protocol, verify=False):
    import jax
    from .parallel_frozen import parameter_digest
    from .parallel_runner import load_npz
    from .posterior_reward_probe import predict_arrays

    directory = Path(output) / "preflight" / arm
    if verify:
        result = read(directory / "result.json")
        proxy, saved = _proxy(
            directory, arm, 901, protocol, directory / "checkpoint.pkl"
        )
        if result["checkpoint_sha256"] != sha(directory / "checkpoint.pkl"):
            raise ValueError("preflight checkpoint hash changed")
        data = load_npz(directory / "probe-inputs.npz")
        arrays = predict_arrays(proxy, data, 291901)
        compare_arrays(directory / "probe.npz", arrays)
        write(
            directory / "replayed.json",
            dict(
                exact=True,
                checkpoint_sha256=result["checkpoint_sha256"],
                process_id=os.getpid(),
                native_steps=0,
                probe_sha256=sha(directory / "probe.npz"),
                parameter_digest=parameter_digest(saved["params"]),
            ),
        )
        return
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    agent, config, _, binding = make_agent(
        output, parent_cell, directory, arm, 901, protocol
    )
    before = agent.save()
    batch, batch_path = retained_batch(parent_cell, agent)
    stream = iter(agent.stream(preflight_batches(batch)))
    carry = agent.init_train(16)
    for _ in range(2):
        carry, _, metrics = agent.train(carry, next(stream))
        jax.block_until_ready(agent.params)
        base._finite(metrics, "preflight")
    if agent.pending_mets is not None:
        metrics = agent._take_outs(agent.pending_mets)
    base._finite(agent.params, "preflight state")
    after = agent.save()
    from .posterior_reward_verify import optimizer_clocks

    before_clocks, after_clocks = optimizer_clocks(before), optimizer_clocks(after)
    if before_clocks["auxiliary"] != 0 or any(
        after_clocks[k] - before_clocks[k] != 2 for k in before_clocks
    ):
        raise ValueError("preflight optimizer update clocks")
    if after["counters"]["updates"] - before["counters"]["updates"] != 2:
        raise ValueError("preflight learner clock")
    deltas = group_deltas(before["params"], after["params"])
    for group in (
        "enc",
        "dyn",
        "rew",
        "pol",
        "val",
        "reward_teacher",
        "reward_student",
    ):
        if not deltas.get(group, {}).get("changed"):
            raise ValueError("required group inactive: " + group)
    agent.load(after)
    if parameter_digest(agent.save()["params"]) != parameter_digest(after["params"]):
        raise ValueError("checkpoint roundtrip mismatch")
    obs = {k: batch[k][:, 0] for k in agent.obs_space}
    with current_policy(agent):
        for mode in ("train", "eval"):
            _, actions, logs = agent.policy(agent.init_policy(16), obs, mode=mode)
            base._finite((actions, logs), "preflight policy")
    save_pickle(directory / "checkpoint.pkl", after)
    data = _probe_fixture(batch)
    _write_npz(directory / "probe-inputs.npz", data)
    proxy, _ = _proxy(directory, arm, 901, protocol, directory / "checkpoint.pkl")
    arrays = predict_arrays(proxy, data, 291901)
    _write_npz(directory / "probe.npz", arrays)
    write(
        directory / "result.json",
        dict(
            arm=arm,
            native_steps=0,
            updates=2,
            full_batch_shape=list(batch["is_first"].shape),
            parent_binding=binding,
            batch_sha256=sha(batch_path),
            checkpoint_sha256=sha(directory / "checkpoint.pkl"),
            parameter_digest=parameter_digest(after["params"]),
            parameter_deltas=deltas,
            optimizer_clocks_before=before_clocks,
            optimizer_clocks_after=after_clocks,
            runtime=runtime_evidence(agent),
            compilation=compilation_evidence(),
            metrics={k: float(v) for k, v in metrics.items() if np.ndim(v) == 0},
            process_id=os.getpid(),
            wall_seconds=time.monotonic() - started,
            synthetic_fixture="tiled retained batch with bounded synthetic actions; not scientific predictions",
        ),
    )
    print("POSTERIOR_REWARD_PREFLIGHT_ARM_CREATED", arm, flush=True)


def run_cell(output, parent_cell, index, protocol):
    import jax
    from .posterior_reward_protocol import (
        cell_for_index,
        collection_seed,
        NativeStepBudget,
    )

    output = Path(output)
    cell = cell_for_index(index)
    directory = output / "cells" / cell["cell_id"]
    # Genesis budget exists from registration, but started evidence is exclusive.
    directory.mkdir(parents=True, exist_ok=True)
    write(
        directory / "started.json",
        {**cell, "job_id": os.environ.get("SLURM_JOB_ID"), "time": time.time()},
    )
    budget = NativeStepBudget(directory, index)
    if budget.total:
        raise ValueError("this cell already spent native budget; no automatic restart")
    start = time.monotonic()
    agent, config, main, binding = make_agent(
        output, parent_cell, directory, cell["arm"], cell["seed"], protocol
    )
    runtime = runtime_evidence(agent)
    write(
        directory / "initialization.json",
        dict(binding=binding, runtime=runtime, seconds=time.monotonic() - start),
    )
    replay = main.make_replay(config, "replay")
    raw = RawRows(directory / "transitions")
    envs, bases = [], []
    evaluations, checkpoints, update_timings = [], [], []
    updates = decisions = native = rows = resets = 0
    first_update_at = None
    next_progress = 1000
    metrics = {}
    stream = None
    episodes = np.full(16, -1, np.int32)
    worker_steps = np.zeros(16, np.int32)
    modes = {"task": 0, "random": 0, "explore": 0}
    try:
        for worker in range(16):
            envseed = collection_seed(cell["seed"], worker)
            env = BudgetedReacher(envseed, budget, "collect")
            bases.append(env)
            envs.append(main.wrap_env(env, config))
        carry = agent.init_policy(16)
        carry_train = agent.init_train(16)
        actions = [dict(reset=True, action=np.zeros(2, np.float32)) for _ in envs]
        path = directory / "checkpoint-0.pkl"
        save_pickle(path, agent.save())
        checkpoints.append(dict(path=path.name, sha256=sha(path), native_steps=0))
        evaluation = evaluate(agent, main, config, cell["seed"], 0, directory, budget)
        evaluations.append(evaluation)
        write(directory / "evaluation-0.json", evaluation)
        for milestone in (40000, 80000):
            while native < milestone:
                # Reacher has fixed-length episodes. All sixteen workers remain
                # in lockstep; every milestone is divisible by16*AR2.
                previous = [x["action"].copy() for x in actions]
                was_reset = [bool(x["reset"]) for x in actions]
                observations = [e.step(a) for e, a in zip(envs, actions)]
                deltas = [0 if x else 2 for x in was_reset]
                if len(set(deltas)) != 1:
                    raise ValueError("unexpected asynchronous reset boundary")
                native += sum(deltas)
                obs = {
                    k: np.stack([r[k] for r in observations]) for k in observations[0]
                }
                for i, reset in enumerate(was_reset):
                    if reset:
                        episodes[i] += 1
                # No exploration intervention: all arms use their task policy.
                mode = "task"
                carry, proposed, outs = agent.policy(carry, obs, mode="train")
                proposed_action = np.asarray(proposed["action"], np.float32)
                executed = proposed_action.copy()
                executed = np.clip(executed, -1, 1)
                executed[obs["is_last"]] = 0
                carry, executed = actual_actions(carry, executed)
                for worker, observation in enumerate(observations):
                    eid = int(worker * 100000 + episodes[worker])
                    extra = {
                        k: v[worker]
                        for k, v in outs.items()
                        if not k.startswith("log/")
                    }
                    transition = {
                        **observation,
                        "action": executed[worker],
                        **extra,
                        "episode_id": np.int32(eid),
                    }
                    replay.add(transition, worker)
                    raw.add(
                        {
                            **transition,
                            "worker": np.int32(worker),
                            "native_delta": np.int32(deltas[worker]),
                            "previous_action": previous[worker],
                            "proposed_action": proposed_action[worker],
                            "collection_mode": np.int32(
                                {"task": 0, "random": 1, "explore": 2}[mode]
                            ),
                            **{
                                k: v[worker]
                                for k, v in outs.items()
                                if k.startswith("log/")
                            },
                        }
                    )
                    rows += 1
                    resets += int(was_reset[worker])
                    actions[worker] = dict(
                        action=executed[worker], reset=bool(observation["is_last"])
                    )
                    if not observation["is_last"]:
                        worker_steps[worker] += 1
                        modes[mode] += 1
                decisions += sum(d > 0 for d in deltas)
                if stream is None and len(replay) >= 16 * 64:
                    stream = iter(
                        agent.stream(main.make_stream(config, replay, "train"))
                    )
                    first_update_at = decisions
                # Deterministic new-decision credit. No reset rows and no offline
                # readout updates buy additional world-model updates.
                target_updates = (
                    0 if first_update_at is None else (decisions - first_update_at) // 2
                )
                while updates < target_updates:
                    timed = updates < 3 or (updates + 1) % 1000 == 0
                    if timed:
                        jax.block_until_ready(agent.params)
                    tick = time.monotonic()
                    batch = next(stream)
                    carry_train, out, metrics = agent.train(carry_train, batch)
                    updates += 1
                    if timed:
                        jax.block_until_ready(agent.params)
                        update_timings.append(
                            dict(
                                update=updates,
                                seconds=time.monotonic() - tick,
                                synchronized_before_and_after=True,
                                includes_replay_fetch=True,
                            )
                        )
                        _write_npz(
                            directory / f"batch-{updates:05d}.npz",
                            {
                                k: np.asarray(jax.device_get(v))
                                for k, v in batch.items()
                                if k != "seed"
                            },
                        )
                    base._finite(metrics, "training metrics")
                    if "replay" in out:
                        replay.update(out["replay"])
                if native >= next_progress:
                    raw.flush()
                    # The wrapper normally returns one-update-delayed metrics.
                    # Drain the completed update before labelling its evidence.
                    if agent.pending_mets is not None:
                        metrics = agent._take_outs(agent.pending_mets)
                    base._finite(metrics, "progress metrics")
                    progress = dict(
                        **cell,
                        native_steps=native,
                        learner_updates=updates,
                        metrics_learner_update=updates,
                        budget=budget.snapshot(),
                        collection_modes=modes,
                        seconds=time.monotonic() - start,
                        metrics={
                            k: float(v) for k, v in metrics.items() if np.ndim(v) == 0
                        },
                    )
                    write(directory / f"progress-{native:06d}.json", progress)
                    print(
                        json.dumps(
                            dict(
                                event="progress",
                                **cell,
                                native_steps=native,
                                learner_updates=updates,
                            )
                        ),
                        flush=True,
                    )
                    next_progress = (native // 1000 + 1) * 1000
            if native != milestone:
                raise RuntimeError("collection milestone overshot")
            base._finite(agent.params, "checkpoint")
            if agent.pending_mets is not None:
                metrics = agent._take_outs(agent.pending_mets)
                base._finite(metrics, "final milestone metrics")
            path = directory / f"checkpoint-{milestone}.pkl"
            save_pickle(path, agent.save())
            checkpoints.append(
                dict(path=path.name, sha256=sha(path), native_steps=milestone)
            )
            evaluation = evaluate(
                agent, main, config, cell["seed"], milestone, directory, budget
            )
            evaluations.append(evaluation)
            write(directory / f"evaluation-{milestone}.json", evaluation)
        raw.flush()
        if (
            native != sum(x.native_steps for x in bases)
            or native != 80000
            or budget.total != 110000
        ):
            raise ValueError("native budget reconciliation failed")
        if updates != int(agent.n_updates) - int(binding["counters"]["updates"]):
            raise ValueError("learner clock reconciliation failed")
        result = dict(
            **cell,
            native_collection_steps=native,
            budget=budget.snapshot(),
            learner_updates=updates,
            first_update_at=first_update_at,
            training_decisions=decisions,
            replay_rows=rows,
            reset_rows=resets,
            collection_modes=modes,
            evaluations=evaluations,
            checkpoints=checkpoints,
            runtime=runtime,
            update_timings=update_timings,
            compilation=compilation_evidence(),
            wall_seconds=time.monotonic() - start,
            replay_restart=True,
            parent_world_seed=431,
            parent_native_steps=200000,
        )
        write(directory / "result.json", result)
        print("POSTERIOR_REWARD_CELL_CREATED", index, flush=True)
    except BaseException as exc:
        raw.flush()
        write(
            directory / "failure.json",
            dict(
                error=repr(exc),
                budget=budget.snapshot(),
                native_collection_steps=native,
                learner_updates=updates,
                wall_seconds=time.monotonic() - start,
            ),
        )
        raise
    finally:
        for env in envs:
            env.close()
        budget.close()


def load_evaluations(directory, milestone):
    from .parallel_runner import load_npz

    rows = [
        load_npz(Path(directory) / f"eval-{milestone}-{ep}.npz") for ep in range(10)
    ]
    keys = rows[0].keys()
    if any(r.keys() != keys for r in rows):
        raise ValueError("evaluation schemas differ")
    result = {k: np.stack([r[k] for r in rows]) for k in keys}
    record = read(Path(directory) / f"evaluation-{milestone}.json")
    result["episode_seed"] = np.asarray(
        [r["seed"] for r in record["episodes"]], np.uint32
    )
    return result


def probes(output, protocol, index, verify=False):
    """Every target checkpoint sees each source policy's same real histories."""
    import jax
    from .posterior_reward_probe import predict_arrays
    from .posterior_reward_protocol import require_marker

    for i in range(9):
        require_marker(output, f"cell-{i}", authenticate_source=False)
    target = cell_for_index(index)
    directory = cell_directory(output, index)
    dest = Path(output) / "probes" / target["cell_id"]
    if not verify:
        dest.mkdir(parents=True, exist_ok=False)
    records = []
    started = time.monotonic()
    for milestone in MILESTONES:
        proxy, _ = _proxy(
            directory,
            target["arm"],
            target["seed"],
            protocol,
            directory / f"checkpoint-{milestone}.pkl",
        )
        for source_index in (index % 3, index % 3 + 3, index % 3 + 6):
            source = cell_for_index(source_index)
            source_dir = cell_directory(output, source_index)
            data = load_evaluations(source_dir, milestone)
            # Common draws across target models, unique source/seed/milestone.
            key = int(
                np.random.SeedSequence(
                    [20261014, target["seed"], milestone, source_index // 3, 332]
                ).generate_state(1)[0]
            )
            arrays = predict_arrays(proxy, data, key)
            path = dest / f"m{milestone}-source{source['arm']}.npz"
            if verify:
                compare_arrays(path, arrays)
            else:
                _write_npz(path, arrays)
            records.append(
                dict(
                    path=path.name,
                    sha256=sha(path),
                    milestone=milestone,
                    target_arm=target["arm"],
                    source_arm=source["arm"],
                    seed=target["seed"],
                    prediction_seed=key,
                    checkpoint_sha256=sha(directory / f"checkpoint-{milestone}.pkl"),
                    inputs=[
                        dict(
                            path=str(source_dir / f"eval-{milestone}-{e}.npz"),
                            sha256=sha(source_dir / f"eval-{milestone}-{e}.npz"),
                        )
                        for e in range(10)
                    ],
                )
            )
        jax.tree.map(lambda x: x.delete(), proxy.params)
        jax.clear_caches()
    if verify:
        original = read(dest / "created.json")
        if original["records"] != records or original["process_id"] == os.getpid():
            raise ValueError("independent probe identity mismatch")
        write(
            dest / "replayed.json",
            dict(
                exact=True,
                records=records,
                process_id=os.getpid(),
                seconds=time.monotonic() - started,
                compilation=compilation_evidence(),
                native_steps=0,
            ),
        )
    else:
        write(
            dest / "created.json",
            dict(
                records=records,
                process_id=os.getpid(),
                seconds=time.monotonic() - started,
                compilation=compilation_evidence(),
                native_steps=0,
            ),
        )
    print(
        (
            "POSTERIOR_REWARD_PROBE_CELL_VERIFIED"
            if verify
            else "POSTERIOR_REWARD_PROBE_CELL_CREATED"
        ),
        index,
        flush=True,
    )


def child(stage, output, **kwargs):
    command = [
        sys.executable,
        "-u",
        "-m",
        __package__ + ".posterior_reward_runner",
        stage,
        "--output",
        str(output),
    ]
    for name, value in kwargs.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "preflight-arm",
            "verify-preflight-arm",
            "verify-preflight",
            "cell",
            "verify-cell",
            "probes",
            "probe-cell",
            "verify-probe-cell",
            "verify-probes",
            "finalize",
            "verify-finalize",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--arm", choices=("A", "B", "C"))
    args = parser.parse_args()
    from .posterior_reward_protocol import authenticate, require_marker

    manifest, protocol = authenticate(args.output)
    protocol = dict(protocol, upstream_path=manifest["upstream"]["path"])
    parent = Path(manifest["inputs"]["parent_cell"])
    if args.stage in (
        "preflight-arm",
        "verify-preflight-arm",
        "cell",
        "probe-cell",
        "verify-probe-cell",
    ):
        start_compilation_monitor()
    if args.stage == "preflight":
        for arm in ("A", "B", "C"):
            child("preflight-arm", args.output, arm=arm)
            child("verify-preflight-arm", args.output, arm=arm)
    elif args.stage in ("preflight-arm", "verify-preflight-arm"):
        _setup(protocol)
        preflight_arm(
            args.output,
            parent,
            args.arm,
            protocol,
            verify=args.stage.startswith("verify"),
        )
    elif args.stage == "cell":
        require_marker(args.output, "preflight")
        if args.index is None:
            raise ValueError("cell index required")
        base._imports(protocol)
        run_cell(args.output, parent, args.index, protocol)
    elif args.stage == "probes":
        for index in range(9):
            child("probe-cell", args.output, index=index)
            child("verify-probe-cell", args.output, index=index)
    elif args.stage in ("probe-cell", "verify-probe-cell"):
        _setup(protocol)
        probes(
            args.output, protocol, args.index, verify=args.stage.startswith("verify")
        )
    else:
        from . import posterior_reward_verify as verify

        {
            "verify-preflight": verify.verify_preflight,
            "verify-cell": lambda o: verify.verify_cell(o, args.index),
            "verify-probes": verify.verify_probes,
            "finalize": verify.finalize,
            "verify-finalize": verify.verify_final,
        }[args.stage](args.output)


if __name__ == "__main__":
    main()
