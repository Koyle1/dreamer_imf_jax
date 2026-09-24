"""Frozen three-task replication: fresh training, four arms, two paired contrasts.

No historical trajectories or outcomes are used as controls. Publication is
exclusive, evaluation is independently replayed, and stage advancement is
permitted only after successful Slurm completion and artifact verification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import subprocess
import sys
import time

import numpy as np

from . import matched_objective_benchmark as b

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "dreamer_imf_comparison/mechanism_replication_protocol.json"
STAGES = ("preflight", "training", "evaluation")
MODULE = "dreamer_imf_compare.mechanism_replication"


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def publish(path, value):
    """Exclusive creation: even an identical existing artifact is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def pickle_new(path, value):
    import jax

    with Path(path).open("xb") as handle:
        pickle.dump(jax.device_get(value), handle, protocol=5)


def finite(tree):
    """Check arrays/scalars without converting configuration strings to floats."""
    if isinstance(tree, dict):
        return all(finite(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return all(finite(v) for v in tree)
    if tree is None or isinstance(tree, (str, bool)):
        return True
    a = np.asarray(tree)
    return a.dtype.kind in "biufc" and bool(np.isfinite(a).all())


def exact_trace(left, right):
    if set(left) != set(right):
        raise ValueError("trace field set differs")
    for name in sorted(left):
        a, c = np.asarray(left[name]), np.asarray(right[name])
        if a.dtype != c.dtype or a.shape != c.shape or a.tobytes() != c.tobytes():
            raise ValueError(f"strict bitwise trace replay differs: {name}")


def cells(protocol, stage):
    if stage == "preflight":
        return [
            dict(index=i, task=t, world_model_seed=431)
            for i, t in enumerate(protocol["tasks"])
        ]
    pairs = [
        dict(task=t, world_model_seed=w)
        for t, w in itertools.product(protocol["tasks"], protocol["world_model_seeds"])
    ]
    if stage == "training":
        return [dict(index=i, **p) for i, p in enumerate(pairs)]
    if stage == "evaluation":
        return [
            dict(index=i, training_index=j, **p, actor_seed=a, arm=arm)
            for i, (j, p, a, arm) in enumerate(
                (j, p, a, arm)
                for j, p in enumerate(pairs)
                for a in protocol["nested_actor_seeds"]
                for arm in protocol["arms"]
            )
        ]
    raise ValueError("unknown stage")


def clean_commit():
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    # Only tracked changes affect deployed code. Untracked user notes are not imported.
    if subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"]
    ):
        raise ValueError("source has tracked modifications")
    return commit


def manifest(root):
    m = read(Path(root) / "manifest.json")
    if m["source_commit"] != clean_commit() or m["protocol"] != read(PROTOCOL):
        raise ValueError("source/protocol differs from frozen manifest")
    if m["protocol_sha256"] != digest(m["protocol"]):
        raise ValueError("protocol digest differs")
    return m


def register(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError("registration requires a fresh output root")
    protocol = read(PROTOCOL)
    m = dict(
        schema="imf-mechanism-replication-v1",
        source_commit=clean_commit(),
        protocol=protocol,
        protocol_sha256=digest(protocol),
        source_root=str(SOURCE),
        created_unix=time.time(),
    )
    publish(root / "manifest.json", m)
    print("MECHANISM_REPLICATION_REGISTERED", digest(m), flush=True)


def runtime():
    import jax
    import jaxlib
    import importlib.metadata

    devices = jax.devices()
    if any(d.platform != "gpu" for d in devices):
        raise RuntimeError("GPU execution required")
    return dict(
        python=sys.version.split()[0],
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        numpy=np.__version__,
        mujoco=importlib.metadata.version("mujoco"),
        dm_control=importlib.metadata.version("dm-control"),
        devices=[d.device_kind for d in devices],
        x64=bool(jax.config.jax_enable_x64),
    )


def collect(task, seed, settings):
    from .dmc import DMCAdapter

    env = DMCAdapter(task, seed=b.derive_seed("imf-replication-data-env", task, seed))
    n, steps = settings["episodes"], settings["steps_per_episode"]
    arrays = dict(
        observations=np.empty((n, steps + 1, *env.observation_shape), np.float32),
        actions=np.zeros((n, steps + 1, env.action_dim), np.float32),
        rewards=np.zeros((n, steps + 1), np.float32),
        continuations=np.ones((n, steps + 1), np.float32),
        is_first=np.zeros((n, steps + 1), bool),
    )
    rng = np.random.default_rng(
        b.derive_seed("imf-replication-data-actions", task, seed)
    )
    try:
        for episode in range(n):
            arrays["observations"][episode, 0] = env.reset()
            arrays["is_first"][episode, 0] = True
            uniform = rng.random() < settings["uniform_episode_probability"]
            smooth = np.zeros(env.action_dim, np.float32)
            for step in range(1, steps + 1):
                smooth = np.clip(
                    0.92 * smooth + rng.normal(0, 0.35, env.action_dim), -1, 1
                ).astype(np.float32)
                action = (
                    rng.uniform(-1, 1, env.action_dim).astype(np.float32)
                    if uniform
                    else smooth
                )
                tr = env.step(action)
                for name, value in (
                    ("observations", tr.observation),
                    ("actions", action),
                    ("rewards", tr.reward),
                    ("continuations", tr.continuation),
                ):
                    arrays[name][episode, step] = value
                if tr.is_last and step < steps:
                    arrays["observations"][episode, step + 1 :] = tr.observation
                    arrays["continuations"][episode, step + 1 :] = 0
                    break
    finally:
        env.close()
    arrays.update(
        episode_ids=np.arange(n, dtype=np.int32),
        train_episode_ids=np.arange(settings["train_episodes"], dtype=np.int32),
        test_episode_ids=np.arange(settings["train_episodes"], n, dtype=np.int32),
    )
    assert not set(arrays["train_episode_ids"]) & set(arrays["test_episode_ids"])
    assert finite(arrays)
    return arrays


def config_for(protocol, arrays):
    from imf_dreamer_jax import DreamerConfig

    cfg = dict(protocol["world_model_template"])
    cfg["observation_shape"] = tuple(arrays["observations"].shape[2:])
    cfg["action_dim"] = arrays["actions"].shape[-1]
    cfg["overshooting_distances"] = tuple(cfg["overshooting_distances"])
    return DreamerConfig(**cfg)


def train(protocol, cell, directory, *, preflight=False):
    import jax
    from imf_dreamer_jax import (
        create_agent,
        jit_train_world_model,
        TransitionRewardConfig,
        init_transition_reward_state,
        jit_train_transition_reward_step,
        attach_transition_reward_head,
        ReBRACConfig,
        init_rebrac_state,
        jit_train_rebrac_chunk,
    )
    from . import transition_reward_study as reward
    from . import flowmpc_actor_study as flow
    from . import actor_gap_model_training as endpoint
    from . import actor_gap_roadmap_study as roadmap

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    task, seed = cell["task"], cell["world_model_seed"]
    settings = dict(protocol["data"])
    budgets = dict(protocol["checkpoint_plan"])
    actors = protocol["nested_actor_seeds"]
    if preflight:
        pf = protocol["preflight"]
        settings.update(
            {k: pf[k] for k in ("episodes", "steps_per_episode", "train_episodes")}
        )
        for name, key in (
            ("world_model_updates", "world_updates"),
            ("reward_head_updates", "reward_updates"),
            ("rebrac_updates", "rebrac_updates"),
            ("endpoint_model_updates", "endpoint_updates"),
        ):
            budgets[name] = pf[key]
        actors = pf["actor_seeds"]
    started = time.perf_counter()
    arrays = collect(task, seed, settings)
    cfg = config_for(protocol, arrays)
    key = lambda name: b.derive_jax_key("imf-replication-training-v1", name, task, seed)
    schedules = {}

    def schedule(name, count, heldout=False):
        source = dict(arrays)
        if heldout:
            source["train_episode_ids"] = arrays["test_episode_ids"]
        value = b._batch_schedule(
            source,
            task=task,
            world_model_seed=b.derive_seed(name, seed),
            updates=count,
            batch_size=settings["batch_size"],
            sequence_length=settings["sequence_length"],
        )
        schedules[name] = value
        return value

    def batch(schedule_, index):
        return b._materialize_batch(
            arrays,
            schedule_,
            index,
            sequence_length=settings["sequence_length"],
            burn_in=cfg.burn_in,
        )

    state = create_agent(cfg, key("world-init"))
    init_world = b._tree_digest(state.params.world_model)
    world_schedule = schedule("world", budgets["world_model_updates"])
    for i in range(budgets["world_model_updates"]):
        state, metrics = jit_train_world_model(
            state, batch(world_schedule, i), jax.random.fold_in(key("world"), i), cfg
        )
        if i % 1000 == 0:
            print("world", task, seed, i, float(metrics.total), flush=True)
    world = state.params.world_model
    frozen_digest = b._tree_digest(world)
    if (
        frozen_digest == init_world
        or int(state.model_optimizer.step) != budgets["world_model_updates"]
    ):
        raise ValueError("world model failed to train expected updates")
    mean, std, normalization_count = reward._training_observation_statistics(arrays)
    head = init_transition_reward_state(
        cfg, key("reward-init"), observation_mean=mean, observation_std=std
    )
    reward_schedule = schedule("reward", budgets["reward_head_updates"])
    heldout = schedule(
        "reward-test", 2 if preflight else settings["reward_test_batches"], heldout=True
    )
    for i in range(budgets["reward_head_updates"]):
        head, metrics = jit_train_transition_reward_step(
            head,
            world,
            batch(reward_schedule, i),
            jax.random.fold_in(key("reward"), i),
            cfg,
            TransitionRewardConfig(),
        )
    reward_metrics = reward._evaluate_reward_mse(
        head.params,
        world,
        arrays,
        heldout,
        cfg,
        seed=b.derive_seed(task, seed, "heldout"),
    )
    attached = attach_transition_reward_head(state, head).params.world_model
    if b._tree_digest(world) != frozen_digest or any(
        b._tree_digest(world[k]) != b._tree_digest(attached[k]) for k in world
    ):
        raise ValueError("reward fitting modified frozen world")
    dataset = flow._build_rebrac_dataset(arrays)
    rebrac_cfg = ReBRACConfig(state_dim=cfg.observation_dim, action_dim=cfg.action_dim)
    policies, policy_info = {}, {}
    states, actions, _, _, _ = roadmap._training_transitions(arrays)
    anchors = states[
        np.linspace(0, len(states) - 1, min(256, len(states)), dtype=np.int64)
    ]
    thresholds = {}
    for actor in actors:
        r = init_rebrac_state(
            b.derive_jax_key("imf-replication-rebrac-init", task, seed, actor),
            rebrac_cfg,
        )
        initial = b._tree_digest(r.actor)
        training_key = b.derive_jax_key(
            "imf-replication-rebrac-training", task, seed, actor
        )
        count = budgets["rebrac_updates"]
        for i in range(0, count, 10000):
            r, rm = jit_train_rebrac_chunk(
                r,
                dataset,
                training_key,
                updates=min(10000, count - i),
                config=rebrac_cfg,
            )
        if initial == b._tree_digest(r.actor) or int(r.step) != count:
            raise ValueError("ReBRAC actor did not update")
        policies[actor] = r
        policy_info[actor] = dict(
            updates=count,
            initial_actor_sha256=initial,
            state_updates=int(r.step),
            final_actor_sha256=b._tree_digest(r.actor),
            final_state_sha256=b._tree_digest(r),
        )
        distance = np.sqrt(
            np.mean((actions - roadmap._actor_actions(r.actor, states)) ** 2, axis=-1)
        )
        thresholds[actor] = float(np.quantile(distance, 0.95))
    endpoints, endpoint_info = {}, {}
    for i, family in enumerate(endpoint.ENDPOINT_FAMILIES):
        spec = endpoint.ModelCellSpec(i, f"{task}-{seed}-{family}", family, seed, None)
        payload = endpoint.train_model_cell_payload(
            spec,
            attached,
            cfg,
            arrays,
            endpoint.ModelTrainingConfig(updates=budgets["endpoint_model_updates"]),
            task=task,
        )
        endpoints[family] = payload["checkpoint"]
        endpoint_info[family] = endpoint.deterministic_payload_identity(payload)
    if b._tree_digest(world) != frozen_digest:
        raise ValueError("endpoint/policy training modified source")
    checkpoint = dict(
        world=world,
        reward_world=attached,
        config=asdict(cfg),
        rebrac_config=asdict(rebrac_cfg),
        policies=policies,
        endpoints=endpoints,
        anchors=anchors,
        thresholds=thresholds,
    )
    if not finite(checkpoint):
        raise FloatingPointError("nonfinite trained checkpoint")
    pickle_new(directory / "checkpoint.pkl", checkpoint)
    with (directory / "replay.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    pickle_new(directory / "schedules.pkl", schedules)
    result = dict(
        cell=cell,
        preflight=preflight,
        runtime=runtime(),
        source_world_sha256=frozen_digest,
        config=asdict(cfg),
        budgets={
            k: budgets[k]
            for k in (
                "world_model_updates",
                "reward_head_updates",
                "rebrac_updates",
                "endpoint_model_updates",
            )
        },
        reward_optimizer_updates=int(head.optimizer.step),
        normalization_train_targets=normalization_count,
        reward_test=reward_metrics,
        policies=policy_info,
        endpoints=endpoint_info,
        replay_sha256=b.array_sha256(arrays),
        schedule_sha256={k: b.array_sha256(v) for k, v in schedules.items()},
        wall_seconds=time.perf_counter() - started,
    )
    publish(directory / "result.json", result)
    return result


def load_checkpoint(directory):
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import DreamerConfig, ReBRACConfig

    with (Path(directory) / "checkpoint.pkl").open("rb") as handle:
        value = pickle.load(handle)
    if not finite(value):
        raise ValueError("nonfinite loaded checkpoint")
    cfg = dict(value["config"])
    cfg["observation_shape"] = tuple(cfg["observation_shape"])
    cfg["overshooting_distances"] = tuple(cfg["overshooting_distances"])
    value["config"] = DreamerConfig(**cfg)
    value["rebrac_config"] = ReBRACConfig(**value["rebrac_config"])
    for name in ("world", "reward_world", "policies", "endpoints"):
        # Endpoint dictionaries include static metadata: convert only array leaves.
        value[name] = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, value[name]
        )
    return value


def verify_training(protocol, directory, *, preflight=False):
    """Reload published bytes; check update clocks, split, config and frozen roots."""
    from .actor_gap_model_training import _tree_sha256

    directory = Path(directory)
    result = read(directory / "result.json")
    with (directory / "checkpoint.pkl").open("rb") as handle:
        model = pickle.load(handle)
    arrays = b.load_npz(directory / "replay.npz")
    with (directory / "schedules.pkl").open("rb") as handle:
        schedules = pickle.load(handle)
    if not finite(model) or not finite(arrays) or not finite(result):
        raise ValueError("nonfinite training artifact")
    cfg = config_for(protocol, arrays)
    if digest(model["config"]) != digest(asdict(cfg)):
        raise ValueError("checkpoint config differs")
    p = protocol["preflight"] if preflight else protocol["checkpoint_plan"]
    expected = dict(
        world_model_updates=(
            p["world_updates"] if preflight else p["world_model_updates"]
        ),
        reward_head_updates=(
            p["reward_updates"] if preflight else p["reward_head_updates"]
        ),
        rebrac_updates=p["rebrac_updates"],
        endpoint_model_updates=(
            p["endpoint_updates"] if preflight else p["endpoint_model_updates"]
        ),
    )
    if (
        result["budgets"] != expected
        or result["reward_optimizer_updates"] != expected["reward_head_updates"]
    ):
        raise ValueError("training update counts differ")
    if b.array_sha256(arrays) != result["replay_sha256"]:
        raise ValueError("replay identity differs")
    train_ids, test_ids = set(arrays["train_episode_ids"]), set(
        arrays["test_episode_ids"]
    )
    if not train_ids or not test_ids or train_ids & test_ids:
        raise ValueError("invalid episode split")
    for name, schedule_ in schedules.items():
        ids = test_ids if name == "reward-test" else train_ids
        if not set(schedule_["episode_ids"].ravel()) <= ids:
            raise ValueError("training/held-out episode leakage")
        if b.array_sha256(schedule_) != result["schedule_sha256"][name]:
            raise ValueError("schedule identity differs")
    if b._tree_digest(model["world"]) != result["source_world_sha256"]:
        raise ValueError("source world differs")
    for key, value in model["world"].items():
        if b._tree_digest(value) != b._tree_digest(model["reward_world"][key]):
            raise ValueError("reward attachment changed frozen source")
    actors = (
        protocol["preflight"]["actor_seeds"]
        if preflight
        else protocol["nested_actor_seeds"]
    )
    if set(model["policies"]) != set(actors):
        raise ValueError("policy seed set differs")
    for actor in actors:
        info, policy = result["policies"][str(actor)], model["policies"][actor]
        if (
            int(policy.step) != expected["rebrac_updates"]
            or b._tree_digest(policy) != info["final_state_sha256"]
        ):
            raise ValueError("policy checkpoint update/digest differs")
    for family, checkpoint in model["endpoints"].items():
        info = result["endpoints"][family]
        if int(checkpoint["optimizer"].step) != expected["endpoint_model_updates"]:
            raise ValueError("endpoint update count differs")
        if _tree_sha256(checkpoint["params"]) != info["checkpoint_params_sha256"]:
            raise ValueError("endpoint parameters differ")
    return result


def evaluate(protocol, cell, training_dir, *, preflight=False):
    from . import actor_gap_roadmap_study as roadmap
    from .actor_gap_diagnostics import (
        fit_coverage_calibration,
        score_observation_action_coverage,
    )

    model = load_checkpoint(training_dir)
    cfg, rc = model["config"], model["rebrac_config"]
    actor, arm = cell["actor_seed"], cell["arm"]
    seeds = (
        protocol["preflight"]["evaluation_seeds"]
        if preflight
        else protocol["evaluation_environment_seeds"]
    )
    steps = (
        protocol["preflight"]["evaluation_steps"]
        if preflight
        else protocol["maximum_environment_steps"]
    )
    common = dict(
        world_seed=cell["world_model_seed"],
        actor_seed=actor,
        evaluation_seeds=seeds,
        maximum_steps=steps,
        task=cell["task"],
    )
    if arm.startswith("A"):
        returns, trace, timing = roadmap._run_flowmpc_arm(
            model["reward_world"],
            cfg,
            model["policies"][actor],
            rc,
            **common,
            trust=True,
            persistence="persistent",
            heldout_acceptance_enabled=arm.startswith("A3"),
            anchor_observations=model["anchors"],
        )
    else:
        family = "endpoint_h1_imf" if arm.startswith("K0") else "endpoint_anystep_imf"
        returns, trace, timing = roadmap._run_endpoint_action_sequence_arm(
            model["reward_world"],
            cfg,
            model["endpoints"][family],
            model["policies"][actor],
            rc,
            **common,
            direct_any_step=arm.startswith("K1"),
            behavior_distance_threshold=model["thresholds"][actor],
        )
    if not finite(trace) or not finite(returns):
        raise ValueError("nonfinite evaluation")
    mask = np.arange(trace["actions"].shape[1])[None] < trace["lengths"][:, None]
    telemetry = {
        k: float(np.mean(v[mask]))
        for k, v in trace.items()
        if v.shape == mask.shape and k not in ("rewards", "is_last", "continuations")
    }
    replay = b.load_npz(Path(training_dir) / "replay.npz")
    states, actions, _, _, _ = roadmap._training_transitions(replay)
    ids = np.linspace(0, len(states) - 1, min(2048, len(states)), dtype=np.int64)
    coverage = fit_coverage_calibration(
        states[ids], actions[ids], k=5, distance_chunk_size=256
    )
    real_coverage = score_observation_action_coverage(
        coverage,
        trace["observations"][mask],
        trace["actions"][mask],
        distance_chunk_size=256,
    ).to_dict()
    real_coverage.pop("points", None)
    core = dict(
        **cell,
        evaluation_environment_seeds=seeds,
        episode_returns=returns,
        action_saturation_fraction=float(
            np.mean(np.abs(trace["actions"][mask]) >= 0.95)
        ),
        telemetry=telemetry,
        coverage=real_coverage,
        trace_sha256=b.array_sha256(trace),
        checkpoint_sha256=b.file_sha256(Path(training_dir) / "checkpoint.pkl"),
    )
    return core, trace, timing


def cell_dir(root, stage, index):
    return Path(root) / stage / f"{index:03d}"


def marker(root, directory, cell, files, **extra):
    m = manifest(root)
    result = dict(
        source_commit=m["source_commit"],
        manifest_sha256=digest(m),
        cell=cell,
        files={name: b.file_sha256(Path(directory) / name) for name in files},
        **extra,
    )
    publish(Path(directory) / "verified.json", result)
    return result


def verify_files(root, directory):
    m = manifest(root)
    directory = Path(directory)
    mark = read(directory / "verified.json")
    if (
        mark["manifest_sha256"] != digest(m)
        or mark["source_commit"] != m["source_commit"]
    ):
        raise ValueError("marker identity mismatch")
    for name, sha in mark["files"].items():
        if "/" in name or name in (".", "..") or b.file_sha256(directory / name) != sha:
            raise ValueError("bound artifact differs")
    result = read(directory / "result.json")
    if not finite(result):
        raise ValueError("nonfinite result")
    if mark.get("strict_bitwise_replay"):
        create, replay = read(directory / "create-receipt.json"), read(
            directory / "replay-receipt.json"
        )
        if create["core_sha256"] != digest(result) or replay["core_sha256"] != digest(
            result
        ):
            raise ValueError("reader result binding differs")
        if (
            create["trace_sha256"] != replay["trace_sha256"]
            or result["trace_sha256"] != create["trace_sha256"]
        ):
            raise ValueError("reader trace binding differs")
        receipts = [
            read(directory / f"{mode}-receipt.json")
            for mode in ("primer", "create", "replay")
        ]
        if (
            len({r["pid"] for r in receipts}) != 3
            or len({digest(r["runtime"]) for r in receipts}) != 1
        ):
            raise ValueError("reader process/runtime identity differs")
        for mode in ("primer", "create", "replay"):
            seal = read(directory / f"{mode}-seal.json")
            if seal["cache_sha256"] != mark["cache_sha256"] or seal[
                "receipt_sha256"
            ] != b.file_sha256(directory / f"{mode}-receipt.json"):
                raise ValueError("reader seal differs")
    return mark


def role(root, stage, index, arm_index, mode, cache):
    """Child process: never compare a compiler-writer trajectory to readers."""
    from .cache_fingerprint import cache_tree_sha256

    m = manifest(root)
    p = m["protocol"]
    cell = cells(p, stage)[index]
    pf = stage == "preflight"
    if pf:
        cell = dict(
            cell, actor_seed=p["preflight"]["actor_seeds"][0], arm=p["arms"][arm_index]
        )
    directory = cell_dir(root, stage, index)
    train_dir = (
        directory / "training"
        if pf
        else cell_dir(root, "training", cell["training_index"])
    )
    if not pf:
        verify_files(root, train_dir)
    output = directory / f"arm-{arm_index}" if pf else directory
    output.mkdir(parents=True, exist_ok=True)
    if mode != "primer":
        expected = read(output / "primer-seal.json")["cache_sha256"]
        if cache_tree_sha256(Path(cache)) != expected:
            raise ValueError("cache changed before reader")
    started = time.perf_counter()
    core, trace, timing = evaluate(p, cell, train_dir, preflight=pf)
    current_runtime = runtime()
    if (
        not pf
        and current_runtime
        != read(cell_dir(root, "preflight", 0) / "result.json")["runtime"]
    ):
        raise ValueError("evaluation runtime differs from authenticated preflight")
    receipt = dict(
        mode=mode,
        pid=os.getpid(),
        runtime=current_runtime,
        core_sha256=digest(core),
        trace_sha256=b.array_sha256(trace),
        cache_sha256=cache_tree_sha256(Path(cache)),
        wall_seconds=time.perf_counter() - started,
        timing=timing,
    )
    if mode == "create":
        publish(output / "result.json", core)
        with (output / "trace.npz").open("xb") as handle:
            np.savez_compressed(handle, **trace)
    elif mode == "replay":
        if core != read(output / "result.json"):
            raise ValueError("evaluation semantic replay differs")
        exact_trace(trace, b.load_npz(output / "trace.npz"))
    publish(output / f"{mode}-receipt.json", receipt)


def sealed_evaluation(root, stage, index, arm_index=0):
    from .cache_fingerprint import cache_tree_sha256

    directory = cell_dir(root, stage, index)
    output = directory / f"arm-{arm_index}" if stage == "preflight" else directory
    job = os.environ["SLURM_JOB_ID"]
    cache = Path(root) / "caches" / f"job-{job}-task-{index}-arm-{arm_index}"
    cache.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        JAX_COMPILATION_CACHE_DIR=str(cache),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="-1",
        JAX_RAISE_PERSISTENT_CACHE_ERRORS="true",
    )
    pids = []
    reference_cache = None
    for mode in ("primer", "create", "replay"):
        subprocess.run(
            [
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
                "--arm-index",
                str(arm_index),
                "--mode",
                mode,
                "--cache",
                str(cache),
            ],
            env=env,
            check=True,
        )
        receipt = read(output / f"{mode}-receipt.json")
        post = cache_tree_sha256(cache)
        if receipt["cache_sha256"] != post or (
            reference_cache is not None and reference_cache != post
        ):
            raise ValueError("post-exit cache fingerprint differs")
        if (
            pids
            and receipt["runtime"] != read(output / "primer-receipt.json")["runtime"]
        ):
            raise ValueError("reader runtime differs")
        pids.append(receipt["pid"])
        reference_cache = post
        publish(
            output / f"{mode}-seal.json",
            dict(
                cache_sha256=post,
                receipt_sha256=b.file_sha256(output / f"{mode}-receipt.json"),
            ),
        )
    if len(set(pids)) != 3:
        raise ValueError("primer/create/replay process identities are not distinct")
    marker(
        root,
        output,
        read(output / "result.json"),
        ["result.json", "trace.npz"]
        + [
            f"{mode}-{kind}.json"
            for mode in ("primer", "create", "replay")
            for kind in ("receipt", "seal")
        ],
        strict_bitwise_replay=True,
        cache_sha256=reference_cache,
    )
    print(
        "MECHANISM_REPLICATION_EVALUATION_VERIFIED", stage, index, arm_index, flush=True
    )


def accounting(job, count=None):
    out = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            str(job),
            "--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES",
        ],
        text=True,
    )
    rows = [line.split("|") for line in out.splitlines() if line.strip()]
    expected = {f"{job}_{i}" for i in range(count)} if count else {str(job)}
    found = {r[0]: r for r in rows if r[0] in expected}
    if set(found) != expected or any(
        r[1:3] != ["COMPLETED", "0:0"] for r in found.values()
    ):
        raise RuntimeError(f"scheduler not fully successful: {out}")
    return dict(raw=out, records=list(found.values()))


def stage_verify(root, stage):
    m = manifest(root)
    receipt = read(Path(root) / "submissions" / f"{stage}.json")
    expected = cells(m["protocol"], stage)
    acct = accounting(receipt["job_id"], len(expected))
    marks = {}
    for cell in expected:
        directory = cell_dir(root, stage, cell["index"])
        mark = verify_files(root, directory)
        if any(mark["cell"].get(k) != v for k, v in cell.items()):
            raise ValueError("cell marker mismatch")
        if stage == "preflight":
            verify_training(m["protocol"], directory / "training", preflight=True)
            for i in range(4):
                verify_files(root, directory / f"arm-{i}")
                arm_result = read(directory / f"arm-{i}" / "result.json")
                if arm_result["checkpoint_sha256"] != b.file_sha256(
                    directory / "training/checkpoint.pkl"
                ):
                    raise ValueError("preflight checkpoint binding differs")
        elif stage == "training":
            training_result = verify_training(m["protocol"], directory)
            preflight_result = read(cell_dir(root, "preflight", 0) / "result.json")
            if training_result["runtime"] != preflight_result["runtime"]:
                raise ValueError("training runtime differs from preflight")
        marks[str(cell["index"])] = b.file_sha256(directory / "verified.json")
    result = dict(
        stage=stage, manifest_sha256=digest(m), accounting=acct, cell_markers=marks
    )
    target = Path(root) / "verified" / f"{stage}.json"
    if target.exists():
        if read(target)["cell_markers"] != marks:
            raise ValueError("stage marker changed")
    else:
        publish(target, result)
    print("MECHANISM_REPLICATION_STAGE_VERIFIED", stage, flush=True)
    return result


def worker(root, stage, index):
    m = manifest(root)
    p = m["protocol"]
    receipt = read(Path(root) / "submissions" / f"{stage}.json")
    if str(receipt["job_id"]) != os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise ValueError("worker scheduler identity differs")
    if digest(m) != receipt["manifest_sha256"]:
        raise ValueError("worker submission manifest differs")
    cell = cells(p, stage)[index]
    directory = cell_dir(root, stage, index)
    if directory.exists():
        raise FileExistsError("cell exists; preserve evidence; no automatic retries")
    if stage != "preflight":
        stage_verify(root, STAGES[STAGES.index(stage) - 1])
    if stage in ("preflight", "training"):
        training_cache = (
            Path(root)
            / "caches"
            / f"training-job-{os.environ['SLURM_JOB_ID']}-task-{index}"
        )
        training_cache.mkdir(parents=True, exist_ok=False)
        subprocess.run(
            [
                sys.executable,
                "-m",
                MODULE,
                "train",
                "--root",
                str(root),
                "--stage",
                stage,
                "--index",
                str(index),
            ],
            check=True,
            env=dict(
                os.environ,
                JAX_COMPILATION_CACHE_DIR=str(training_cache),
                JAX_ENABLE_COMPILATION_CACHE="true",
            ),
        )
        verify_training(
            p,
            directory / "training" if stage == "preflight" else directory,
            preflight=stage == "preflight",
        )
        if stage == "preflight":
            for arm_index in range(4):
                sealed_evaluation(root, stage, index, arm_index)
            publish(
                directory / "result.json",
                dict(cell=cell, runtime=runtime(), status="passed"),
            )
            marker(root, directory, cell, ["result.json"])
        else:
            marker(
                root,
                directory,
                cell,
                ["result.json", "checkpoint.pkl", "replay.npz", "schedules.pkl"],
            )
    else:
        directory.mkdir(parents=True, exist_ok=False)
        sealed_evaluation(root, stage, index)
    print("MECHANISM_REPLICATION_CELL_VERIFIED", stage, index, flush=True)


def launch(root, stage):
    root = Path(root).resolve()
    m = manifest(root)
    if stage not in STAGES:
        raise ValueError("unknown launch stage")
    if stage != "preflight":
        stage_verify(root, STAGES[STAGES.index(stage) - 1])
    receipt_path = root / "submissions" / f"{stage}.json"
    intent_path = root / "submissions" / f"{stage}.intent.json"
    if receipt_path.exists() or intent_path.exists():
        raise FileExistsError(
            "stage already submitted or submission uncertain; inspect, do not duplicate"
        )
    count = len(cells(m["protocol"], stage))
    publish(intent_path, dict(manifest_sha256=digest(m), stage=stage, count=count))
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    cmd = f'python -m dreamer_imf_compare.mechanism_replication worker --root {shlex.quote(str(root))} --stage {stage} --index "$SLURM_ARRAY_TASK_ID"'
    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "module purge",
            "module load Python/3.12.3-GCCcore-13.3.0",
            "source /work2/ci72buri-dreamer_imf_neurips/venv-cuda12/bin/activate",
            "export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORM_NAME=gpu JAX_ENABLE_X64=0 MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false",
            f"export PYTHONPATH={shlex.quote(str(SOURCE / 'imf_dreamer_jax/src'))}:{shlex.quote(str(SOURCE / 'dreamer_imf_comparison'))}",
            cmd,
        ]
    )
    output = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--no-requeue",
            "--account=dep_inin_dat",
            "--partition=gpu-l40s",
            "--gres=gpu:1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=12:00:00",
            f"--array=0-{count-1}%4",
            f"--job-name=imf-replication-{stage}",
            f"--output={logs}/{stage}-%A_%a.out",
            f"--error={logs}/{stage}-%A_%a.err",
        ],
        input=script,
        text=True,
    ).strip()
    job = output.split(";")[0]
    if not re.fullmatch(r"[0-9]+", job):
        raise RuntimeError("unrecognized scheduler response; preserve intent")
    publish(
        receipt_path,
        dict(
            job_id=job,
            manifest_sha256=digest(m),
            stage=stage,
            count=count,
            script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        ),
    )
    subprocess.run(["scontrol", "release", job], check=True)
    publish(
        root / "submissions" / f"{stage}.released.json",
        dict(job_id=job, released_unix=time.time()),
    )
    print("MECHANISM_REPLICATION_SUBMITTED", stage, job, flush=True)


def finalize(root):
    from .mechanism_replication_analysis import analyze

    root = Path(root)
    m = manifest(root)
    stages = {s: stage_verify(root, s) for s in STAGES}
    records = [
        read(cell_dir(root, "evaluation", c["index"]) / "result.json")
        for c in cells(m["protocol"], "evaluation")
    ]
    report = analyze(records, m["protocol"])
    evaluation_runtime = {}
    for cell in cells(m["protocol"], "evaluation"):
        directory = cell_dir(root, "evaluation", cell["index"])
        roles = {
            mode: read(directory / f"{mode}-receipt.json")
            for mode in ("primer", "create", "replay")
        }
        evaluation_runtime[str(cell["index"])] = {
            "role_wall_seconds": {mode: r["wall_seconds"] for mode, r in roles.items()},
            "controller_latency": roles["create"]["timing"],
        }
    training_results = [
        read(cell_dir(root, "training", c["index"]) / "result.json")
        for c in cells(m["protocol"], "training")
    ]
    runtime_summary = {
        "allocation_gpu_seconds_by_stage": {
            s: sum(int(r[3]) for r in value["accounting"]["records"])
            for s, value in stages.items()
        },
        "training_wall_seconds_by_cell": {
            str(r["cell"]["index"]): r["wall_seconds"] for r in training_results
        },
        "evaluation_by_cell": evaluation_runtime,
        "scope": "one_GPU_per_allocation; successful stage allocations; queue time excluded; process timings exclude shell seals",
    }
    report.update(
        source_commit=m["source_commit"],
        manifest_sha256=digest(m),
        artifact_verification="all stage/cell hashes and bitwise evaluation replay verified",
        scope="fixed_three_task_two_contrast_replication",
        limitations=m["protocol"]["limitations"],
        stages=stages,
        records=records,
        training_results=training_results,
        runtime=runtime_summary,
    )
    if (root / "report.json").exists():
        if read(root / "report.json") != report:
            raise ValueError("existing final report differs")
    else:
        publish(root / "report.json", report)
        marker(root, root, {"stage": "final"}, ["report.json"])
    final_marker = read(root / "verified.json")
    if final_marker["manifest_sha256"] != digest(m) or final_marker["files"] != {
        "report.json": b.file_sha256(root / "report.json")
    }:
        raise ValueError("final report binding differs")
    print("MECHANISM_REPLICATION_FINAL_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("register", "launch", "worker", "train", "role", "verify", "finalize"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--arm-index", type=int, default=0)
    parser.add_argument("--mode", choices=("primer", "create", "replay"))
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        register(args.root)
    elif args.command == "launch":
        launch(args.root, args.stage)
    elif args.command == "worker":
        worker(args.root, args.stage, args.index)
    elif args.command == "train":
        p = manifest(args.root)["protocol"]
        directory = cell_dir(args.root, args.stage, args.index)
        train(
            p,
            cells(p, args.stage)[args.index],
            directory / "training" if args.stage == "preflight" else directory,
            preflight=args.stage == "preflight",
        )
    elif args.command == "role":
        role(args.root, args.stage, args.index, args.arm_index, args.mode, args.cache)
    elif args.command == "verify":
        stage_verify(args.root, args.stage)
    else:
        finalize(args.root)


if __name__ == "__main__":
    main()
