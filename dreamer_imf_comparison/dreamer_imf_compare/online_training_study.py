"""Bounded episode-block online fine-tuning and paired A1 evaluation.

Original evidence is read-only. New artifacts are exclusive publications. Failed
or uncertain submissions never trigger an automatic retry. Training archives the
full learner state and replay episodes; evaluation is independently replayed in
separate cache-reader processes before its marker is accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import os
from pathlib import Path
import pickle
import re
import shlex
import subprocess
import sys
import time

import numpy as np

from . import mechanism_replication as m
from . import matched_objective_benchmark as b
from . import controller_repair_study as repair

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "dreamer_imf_comparison/online_training_protocol.json"
MODULE = "dreamer_imf_compare.online_training_study"
STAGES = ("preflight", "training", "evaluation")


def cells(p, stage):
    if stage == "preflight":
        return [
            dict(index=0, world_model_seed=p["world_model_seeds"][0], training_index=0)
        ]
    if stage == "training":
        return [
            dict(index=i * 2 + j, training_index=i, world_model_seed=w, arm=a)
            for i, w in enumerate(p["world_model_seeds"])
            for j, a in enumerate(("P", "M"))
        ]
    if stage == "evaluation":
        rows = []
        for i, w in enumerate(p["world_model_seeds"]):
            rows.append(dict(training_index=i, world_model_seed=w, arm="F", steps=0))
            for a in ("P", "M"):
                for steps in p["evaluation_steps"][1:]:
                    rows.append(
                        dict(training_index=i, world_model_seed=w, arm=a, steps=steps)
                    )
        return [dict(index=i, **r) for i, r in enumerate(rows)]
    raise ValueError("unknown stage")


def file_digest(path):
    return b.file_sha256(path)


def write_npz(path, arrays):
    if not m.finite(arrays):
        raise ValueError("nonfinite arrays")
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def write_pickle(path, value):
    import jax

    with Path(path).open("xb") as f:
        pickle.dump(jax.device_get(value), f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    # Only called after exact artifact hash authentication, on our own checkpoints.
    with Path(path).open("rb") as f:
        return pickle.load(f)


def manifest(root):
    value = m.read(Path(root) / "manifest.json")
    if (
        value["source_commit"] != m.clean_commit()
        or value["source_root"] != str(SOURCE)
        or value["protocol"] != m.read(PROTOCOL)
        or value["protocol_sha256"] != m.digest(value["protocol"])
    ):
        raise ValueError("online source/protocol identity differs")
    if value["dependency_index"] != repair.dependency_index(
        value["dependency_root"], m.read(repair.PROTOCOL), verify_artifacts=False
    ):
        raise ValueError("online dependency binding differs")
    return value


def register(root, dependency):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("fresh output root required")
    p = m.read(PROTOCOL)
    index = repair.dependency_index(dependency, m.read(repair.PROTOCOL))
    if [index[str(i)]["cell"]["world_model_seed"] for i in range(3)] != p[
        "world_model_seeds"
    ]:
        raise ValueError("dependency pairing differs")
    value = dict(
        schema=p["schema"],
        source_commit=m.clean_commit(),
        source_root=str(SOURCE),
        protocol=p,
        protocol_sha256=m.digest(p),
        dependency_root=str(Path(dependency).resolve()),
        dependency_index=index,
        created_unix=time.time(),
    )
    m.publish(root / "manifest.json", value)
    print("ONLINE_STUDY_REGISTERED", m.digest(value), flush=True)


def dependency(value, index):
    entry = value["dependency_index"][str(index)]
    cell = dict(training_index=index, **entry["cell"])
    directory = repair.authenticate_checkpoint(value, cell)
    return m.load_checkpoint(directory), b.load_npz(directory / "replay.npz")


def seal(directory, value, context):
    directory = Path(directory)
    files = {p.name: file_digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
    if "verified.json" in files:
        raise FileExistsError("already sealed")
    mark = dict(
        context=context,
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        files=files,
    )
    m.publish(directory / "verified.json", mark)
    return mark


def verify_directory(directory, value, context):
    directory = Path(directory)
    mark = m.read(directory / "verified.json")
    if (
        mark["context"] != context
        or mark["source_commit"] != value["source_commit"]
        or mark["manifest_sha256"] != m.digest(value)
        or {p.name for p in directory.iterdir() if p.is_file()}
        != set(mark["files"]) | {"verified.json"}
    ):
        raise ValueError("artifact inventory or binding differs")
    for name, sha in mark["files"].items():
        if Path(name).name != name or file_digest(directory / name) != sha:
            raise ValueError("artifact digest differs")
    return mark


def training_index(cell):
    return cell["training_index"] * 2 + (cell["arm"] == "M")


def epoch_dir(root, cell, episode):
    return (
        Path(root) / "training" / f"{training_index(cell):03d}" / f"epoch-{episode:03d}"
    )


def epoch_context(cell, episode):
    return dict(stage="training_epoch", cell=cell, episode=episode)


def training_seed(p, world, episode):
    seed = b.derive_seed(p["training_seed_namespace"], world, episode)
    if seed in p["evaluation_seeds"] or seed == p["preflight_seed"]:
        raise ValueError("training/evaluation seed collision")
    return seed


def train(root, index):
    from .online_learning import initialize, update, export_model
    from .online_collector import make_collector

    root = Path(root)
    value = manifest(root)
    p = value["protocol"]
    cell = cells(p, "training")[index]
    model, offline = dependency(value, cell["training_index"])
    state = initialize(model, p["actor_seed"])
    collector = make_collector(model["config"], model["rebrac_config"], p["controller"])
    online = []
    parent = file_digest(root / "manifest.json")
    for episode in range(1, p["training_steps"] // p["episode_steps"] + 1):
        started = time.perf_counter()
        directory = epoch_dir(root, cell, episode)
        context = epoch_context(cell, episode)
        # An existing unsealed partial epoch is intentionally not overwritten.
        if directory.exists():
            verify_epoch(root, value, cell, episode, parent, base_model=model)
            state = read_pickle(directory / "learner.pkl")
            online.append(b.load_npz(directory / "episode.npz"))
            parent = file_digest(directory / "verified.json")
            continue
        directory.mkdir(parents=True, exist_ok=False)
        current = export_model(state, model, p["actor_seed"])
        before = parameter_digests(current, p["actor_seed"])
        arrays, trace, timing = collector.rollout(
            current,
            p["actor_seed"],
            cell["world_model_seed"],
            training_seed(p, cell["world_model_seed"], episode),
            maximum_steps=p["episode_steps"],
            exploration_std=p["exploration_std"],
            training=True,
        )
        validate_episode(arrays, p["episode_steps"], require_native_end=True)
        online.append(arrays)
        learning_started = time.perf_counter()
        state, metrics = update(
            state,
            offline,
            online,
            full_model=cell["arm"] == "M",
            seed=b.derive_seed("online-update", cell["world_model_seed"], episode),
            world_updates=p["world_updates_per_episode"] if cell["arm"] == "M" else 0,
            policy_updates=p["policy_updates_per_episode"],
            batch_size=p["batch_size"],
            sequence_length=p["sequence_length"],
        )
        learning_seconds = time.perf_counter() - learning_started
        current = export_model(state, model, p["actor_seed"])
        after = parameter_digests(current, p["actor_seed"])
        validate_changes(before, after, cell["arm"])
        if not finite_learner(state) or not m.finite(metrics):
            raise ValueError("nonfinite learner")
        write_npz(directory / "episode.npz", arrays)
        write_npz(directory / "trace.npz", trace)
        write_pickle(directory / "learner.pkl", state)
        if episode * p["episode_steps"] in p["evaluation_steps"]:
            write_pickle(directory / "model.pkl", current)
        result = dict(
            context=context,
            parent_sha256=parent,
            steps=episode * p["episode_steps"],
            environment_seed=training_seed(p, cell["world_model_seed"], episode),
            before=before,
            after=after,
            update_metrics=metrics,
            collection=timing,
            return_=float(np.sum(arrays["rewards"], dtype=np.float64)),
            positive_reward_steps=int(np.sum(arrays["rewards"] > 0)),
            learning_seconds=learning_seconds,
            runtime=m.runtime(),
            wall_seconds=time.perf_counter() - started,
        )
        m.publish(directory / "result.json", result)
        seal(directory, value, context)
        verify_epoch(root, value, cell, episode, parent, base_model=model)
        parent = file_digest(directory / "verified.json")
        print(
            "ONLINE_EPOCH_VERIFIED",
            index,
            episode,
            result["steps"],
            result["return_"],
            flush=True,
        )
    m.publish(
        root / "training" / f"{index:03d}" / "complete.json",
        dict(
            cell=cell,
            manifest_sha256=m.digest(value),
            final_epoch_sha256=parent,
            steps=p["training_steps"],
        ),
    )


def parameter_digests(model, actor):
    return dict(
        world=b._tree_digest(model["world"]),
        reward=b._tree_digest(model["reward_world"]["reward_transition"]),
        actor=b._tree_digest(model["policies"][actor].actor),
        critic=b._tree_digest(model["policies"][actor].critics),
    )


def finite_learner(state):
    return m.finite(
        {k: v for k, v in state.items() if k not in ("config", "rebrac_config")}
    )


def snapshot_digest(value):
    """Stable across serialization; never hash Python object pointer bytes."""
    if is_dataclass(value):
        return snapshot_digest(asdict(value))
    if isinstance(value, dict):
        return m.digest(
            [
                [repr(k), snapshot_digest(v)]
                for k, v in sorted(value.items(), key=lambda x: repr(x[0]))
            ]
        )
    if isinstance(value, (tuple, list)):
        return m.digest([snapshot_digest(v) for v in value])
    if value is None or isinstance(value, (str, bool, int, float)):
        return m.digest(value)
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise ValueError("object array in checkpoint")
    return b.array_sha256({"value": a})


def validate_alignment(ep, trace, seed):
    for name in ("actions", "rewards", "continuations", "is_last"):
        if not np.array_equal(ep[name][:, 1:], trace[name]):
            raise ValueError("episode/trace transition alignment differs")
    if (
        not np.array_equal(ep["observations"][:, :-1], trace["observations"])
        or trace["lengths"].tolist() != [ep["rewards"].shape[1] - 1]
        or trace["evaluation_seeds"].tolist() != [seed]
        or not m.finite(trace)
    ):
        raise ValueError("episode/trace observation or seed differs")


def validate_changes(before, after, arm):
    if arm == "F":
        if before != after:
            raise ValueError("frozen arm changed")
        return
    for key in ("actor", "critic"):
        if before[key] == after[key]:
            raise ValueError("policy or critic did not update")
    for key in ("world", "reward"):
        if (before[key] != after[key]) != (arm == "M"):
            raise ValueError("world/reward update scope differs")


def validate_episode(ep, steps, *, require_native_end):
    if not m.finite(ep):
        raise ValueError("nonfinite episode")
    if ep["observations"].shape[:2] != (1, steps + 1):
        raise ValueError("episode length differs")
    for k in ("actions", "rewards", "continuations", "is_first", "is_last"):
        if ep[k].shape[:2] != (1, steps + 1):
            raise ValueError("shifted array length differs")
    if not ep["is_first"][0, 0] or np.any(ep["is_first"][0, 1:]):
        raise ValueError("reset alignment differs")
    if np.any(ep["actions"][0, 0] != 0) or ep["rewards"][0, 0] != 0:
        raise ValueError("initial transition must be empty")
    if np.any(np.abs(ep["actions"]) > 1) or np.any(
        (ep["continuations"] < 0) | (ep["continuations"] > 1)
    ):
        raise ValueError("invalid action/continuation support")
    if np.any(ep["is_last"][0, :-1]) or (
        require_native_end and not ep["is_last"][0, -1]
    ):
        raise ValueError("early or missing native boundary")


def verify_epoch(root, value, cell, episode, parent, *, base_model=None):
    from .online_learning import export_model, initialize

    directory = epoch_dir(root, cell, episode)
    mark = verify_directory(directory, value, epoch_context(cell, episode))
    p = value["protocol"]
    expected = {"learner.pkl", "episode.npz", "trace.npz", "result.json"}
    if episode * p["episode_steps"] in p["evaluation_steps"]:
        expected.add("model.pkl")
    if set(mark["files"]) != expected:
        raise ValueError("epoch artifact set differs")
    r = m.read(directory / "result.json")
    if (
        r["context"] != epoch_context(cell, episode)
        or r["parent_sha256"] != parent
        or r["steps"] != episode * p["episode_steps"]
        or r["environment_seed"] != training_seed(p, cell["world_model_seed"], episode)
        or not m.finite(r)
    ):
        raise ValueError("epoch chain differs")
    ep = b.load_npz(directory / "episode.npz")
    validate_episode(ep, p["episode_steps"], require_native_end=True)
    if r["return_"] != float(np.sum(ep["rewards"], dtype=np.float64)):
        raise ValueError("return differs")
    validate_changes(r["before"], r["after"], cell["arm"])
    validate_alignment(ep, b.load_npz(directory / "trace.npz"), r["environment_seed"])
    if base_model is None:
        base_model = dependency(value, cell["training_index"])[0]
    state = read_pickle(directory / "learner.pkl")
    initial = initialize(base_model, p["actor_seed"])
    if (
        not finite_learner(state)
        or state["config"] != initial["config"]
        or state["rebrac_config"] != initial["rebrac_config"]
    ):
        raise ValueError("learner configuration/finiteness differs")
    current = export_model(state, base_model, p["actor_seed"])
    if parameter_digests(current, p["actor_seed"]) != r["after"]:
        raise ValueError("learner parameters differ")
    expected_before = (
        parameter_digests(base_model, p["actor_seed"])
        if episode == 1
        else m.read(epoch_dir(root, cell, episode - 1) / "result.json")["after"]
    )
    if r["before"] != expected_before:
        raise ValueError("parameter ancestry differs")
    updates = p["world_updates_per_episode"] if cell["arm"] == "M" else 0
    for key, count in (
        ("world_updates", updates),
        ("reward_updates", updates),
        ("policy_updates", p["policy_updates_per_episode"]),
    ):
        if r["update_metrics"][key] != count:
            raise ValueError("update budget differs")
    policy_steps = episode * p["policy_updates_per_episode"]
    policy = state["policy"]
    start = initial["policy"]
    delay = state["rebrac_config"].policy_frequency
    actor_updates = (int(start.step) + policy_steps + delay - 1) // delay - (
        int(start.step) + delay - 1
    ) // delay
    if (
        int(state["world_optimizer"].step) != episode * updates
        or int(state["reward_optimizer"].step) != episode * updates
        or int(policy.step) != int(start.step) + policy_steps
        or int(policy.critic_optimizer.step)
        != int(start.critic_optimizer.step) + policy_steps
        or int(policy.actor_optimizer.step)
        != int(start.actor_optimizer.step) + actor_updates
    ):
        raise ValueError("optimizer update clock differs")
    if "model.pkl" in expected:
        saved = read_pickle(directory / "model.pkl")
        if snapshot_digest(saved) != snapshot_digest(current):
            raise ValueError("evaluation snapshot differs from learner")
    return mark


def evaluation_model(root, value, cell):
    p = value["protocol"]
    if cell["arm"] == "F":
        return dependency(value, cell["training_index"])[0]
    tc = cells(p, "training")[training_index(cell)]
    ep = cell["steps"] // p["episode_steps"]
    directory = epoch_dir(root, tc, ep)
    verify_directory(directory, value, epoch_context(tc, ep))
    return read_pickle(directory / "model.pkl")


def compute_evaluation(root, value, cell):
    from .online_collector import make_collector

    p = value["protocol"]
    model = evaluation_model(root, value, cell)
    collector = make_collector(model["config"], model["rebrac_config"], p["controller"])
    before = parameter_digests(model, p["actor_seed"])
    episodes, traces, metrics = [], [], []
    for seed in p["evaluation_seeds"]:
        ep, trace, timing = collector.rollout(
            model,
            p["actor_seed"],
            cell["world_model_seed"],
            seed,
            maximum_steps=p["episode_steps"],
            training=False,
        )
        validate_episode(ep, p["episode_steps"], require_native_end=True)
        episodes.append(ep)
        traces.append(trace)
        metrics.append(timing)
    if before != parameter_digests(model, p["actor_seed"]):
        raise ValueError("evaluation updated base learner")
    arrays = {
        f"episode_{i}_{k}": v for i, ep in enumerate(episodes) for k, v in ep.items()
    }
    arrays.update(
        {f"trace_{i}_{k}": v for i, tr in enumerate(traces) for k, v in tr.items()}
    )
    core = dict(
        cell=cell,
        parameters=before,
        episode_returns=[
            float(np.sum(ep["rewards"], dtype=np.float64)) for ep in episodes
        ],
        evaluation_seeds=p["evaluation_seeds"],
        trace_sha256=b.array_sha256(arrays),
    )
    return core, arrays, metrics


def role(root, index, mode):
    value = manifest(root)
    cell = cells(value["protocol"], "evaluation")[index]
    out = Path(root) / "evaluation" / f"{index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    core, arrays, metrics = compute_evaluation(root, value, cell)
    if mode == "create":
        m.publish(out / "result.json", core)
        write_npz(out / "trace.npz", arrays)
    elif mode == "replay":
        if core != m.read(out / "result.json"):
            raise ValueError("online evaluation semantic replay differs")
        m.exact_trace(b.load_npz(out / "trace.npz"), arrays)
    from .cache_fingerprint import cache_tree_sha256

    cache = cache_tree_sha256(Path(os.environ["JAX_COMPILATION_CACHE_DIR"]))
    m.publish(
        out / f"{mode}.json",
        dict(
            pid=os.getpid(),
            core_sha256=m.digest(core),
            cache=cache,
            runtime=m.runtime(),
            metrics=metrics,
            wall_seconds=time.perf_counter() - started,
        ),
    )


def cache_role(root, index, mode):
    subprocess.run(
        [
            sys.executable,
            "-m",
            MODULE,
            "role",
            "--root",
            str(root),
            "--index",
            str(index),
            "--mode",
            mode,
        ],
        check=True,
    )
    from .cache_fingerprint import cache_tree_sha256

    out = Path(root) / "evaluation" / f"{index:03d}"
    receipt = m.read(out / f"{mode}.json")
    cache = cache_tree_sha256(Path(os.environ["JAX_COMPILATION_CACHE_DIR"]))
    if receipt["cache"] != cache:
        raise ValueError("in-process/post-exit cache fingerprint differs")
    m.publish(
        out / f"{mode}-seal.json",
        dict(cache=cache, receipt_sha256=file_digest(out / f"{mode}.json")),
    )
    return receipt


def evaluate(root, index):
    value = manifest(root)
    out = Path(root) / "evaluation" / f"{index:03d}"
    if out.exists():
        raise FileExistsError("evaluation evidence already exists; no overwrite")
    receipts = [
        cache_role(root, index, mode) for mode in ("primer", "create", "replay")
    ]
    if len({r["pid"] for r in receipts}) != 3 or any(
        r["cache"] != receipts[0]["cache"] for r in receipts
    ):
        raise ValueError("reader execution/cache identities differ")
    seal(
        out,
        value,
        dict(stage="evaluation", cell=cells(value["protocol"], "evaluation")[index]),
    )
    print("ONLINE_EVALUATION_VERIFIED", index, flush=True)


def preflight(root):
    import jax
    from .online_learning import initialize, update, export_model
    from .online_collector import make_collector

    value = manifest(root)
    p = value["protocol"]
    if jax.default_backend() != "gpu":
        raise ValueError("GPU preflight required")
    out = Path(root) / "preflight" / "000"
    out.mkdir(parents=True, exist_ok=False)
    model, offline = dependency(value, 0)
    collector = make_collector(model["config"], model["rebrac_config"], p["controller"])
    reports = {}
    for arm in ("P", "M"):
        state = initialize(model, p["actor_seed"])
        before = parameter_digests(model, p["actor_seed"])
        ep, trace, timing = collector.rollout(
            model,
            p["actor_seed"],
            431,
            p["preflight_seed"],
            maximum_steps=p["preflight_steps"],
            exploration_std=p["exploration_std"],
            training=True,
        )
        validate_episode(ep, p["preflight_steps"], require_native_end=False)
        # A short preflight truncation is a segment boundary, not an absorbing terminal.
        ep["is_last"][0, -1] = True
        learning_started = time.perf_counter()
        state, losses = update(
            state,
            offline,
            [ep],
            full_model=arm == "M",
            seed=98107,
            world_updates=p["preflight_updates"] if arm == "M" else 0,
            policy_updates=p["preflight_updates"],
            batch_size=p["batch_size"],
            sequence_length=p["sequence_length"],
        )
        learning_seconds = time.perf_counter() - learning_started
        updated = export_model(state, model, p["actor_seed"])
        warm_started = time.perf_counter()
        discarded_state, discarded_metrics = update(
            state,
            offline,
            [ep],
            full_model=arm == "M",
            seed=98113,
            world_updates=p["preflight_updates"] if arm == "M" else 0,
            policy_updates=p["preflight_updates"],
            batch_size=p["batch_size"],
            sequence_length=p["sequence_length"],
        )
        warm_learning_seconds = time.perf_counter() - warm_started
        if not finite_learner(discarded_state) or not m.finite(discarded_metrics):
            raise ValueError("nonfinite discarded timing probe")
        after = parameter_digests(updated, p["actor_seed"])
        validate_changes(before, after, arm)
        if not finite_learner(state) or not m.finite(losses):
            raise ValueError("nonfinite preflight training")
        # Two fresh resets using one frozen model must exactly agree, including dynamics.
        a, ta, _ = collector.rollout(
            updated, p["actor_seed"], 431, 98109, maximum_steps=12
        )
        c, tc, _ = collector.rollout(
            updated, p["actor_seed"], 431, 98109, maximum_steps=12
        )
        m.exact_trace(a, c)
        m.exact_trace(ta, tc)
        write_pickle(out / f"{arm}-learner.pkl", state)
        write_pickle(out / f"{arm}-model.pkl", updated)
        write_npz(out / f"{arm}-episode.npz", ep)
        reports[arm] = dict(
            before=before,
            after=after,
            losses=losses,
            collection=timing,
            learning_seconds=learning_seconds,
            discarded_warm_learning_seconds=warm_learning_seconds,
            rough_training_seconds=p["training_steps"]
            * timing["mean_milliseconds_per_step"]
            / 1000
            + (p["training_steps"] / p["episode_steps"])
            * p["policy_updates_per_episode"]
            * warm_learning_seconds
            / p["preflight_updates"],
        )
    if parameter_digests(model, p["actor_seed"]) != before:
        raise ValueError("preflight mutated source model")
    m.publish(
        out / "reader-inputs.json",
        dict(
            manifest_sha256=m.digest(value),
            models={a: file_digest(out / f"{a}-model.pkl") for a in ("P", "M")},
        ),
    )
    # Production-style independent cache readers, using both actually updated models.
    from .cache_fingerprint import cache_tree_sha256

    for index, arm in enumerate(("P", "M")):
        cache = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).parent / (
            Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).name + f"-{arm}-replay"
        )
        cache.mkdir(exist_ok=False)
        env = dict(os.environ, JAX_COMPILATION_CACHE_DIR=str(cache))
        receipts = []
        for mode in ("primer", "create", "replay"):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    MODULE,
                    "preflight-role",
                    "--root",
                    str(root),
                    "--index",
                    str(index),
                    "--mode",
                    mode,
                ],
                env=env,
                check=True,
            )
            receipt = m.read(out / f"{arm}-{mode}.json")
            if receipt["cache"] != cache_tree_sha256(cache):
                raise ValueError("preflight reader post-exit cache differs")
            m.publish(
                out / f"{arm}-{mode}-seal.json",
                dict(
                    cache=receipt["cache"],
                    receipt_sha256=file_digest(out / f"{arm}-{mode}.json"),
                ),
            )
            receipts.append(receipt)
        if (
            len({r["pid"] for r in receipts}) != 3
            or len({r["cache"] for r in receipts}) != 1
            or receipts[1]["core_sha256"] != receipts[2]["core_sha256"]
        ):
            raise ValueError("preflight independent reader replay differs")
    m.publish(out / "result.json", dict(reports=reports, runtime=m.runtime()))
    seal(out, value, dict(stage="preflight"))
    print("ONLINE_PREFLIGHT_VERIFIED", flush=True)


def preflight_role(root, index, mode):
    from .online_collector import make_collector
    from .cache_fingerprint import cache_tree_sha256

    value = manifest(root)
    p = value["protocol"]
    out = Path(root) / "preflight" / "000"
    arm = ("P", "M")[index]
    inputs = m.read(out / "reader-inputs.json")
    if inputs["manifest_sha256"] != m.digest(value) or inputs["models"][
        arm
    ] != file_digest(out / f"{arm}-model.pkl"):
        raise ValueError("preflight reader input differs")
    model = read_pickle(out / f"{arm}-model.pkl")
    collector = make_collector(model["config"], model["rebrac_config"], p["controller"])
    ep, trace, metrics = collector.rollout(
        model, p["actor_seed"], p["world_model_seeds"][0], 98109, maximum_steps=12
    )
    validate_episode(ep, 12, require_native_end=False)
    validate_alignment(ep, trace, 98109)
    arrays = {
        **{f"ep_{k}": v for k, v in ep.items()},
        **{f"trace_{k}": v for k, v in trace.items()},
    }
    core = dict(
        arrays_sha256=b.array_sha256(arrays), model_sha256=inputs["models"][arm]
    )
    if mode == "create":
        write_npz(out / f"{arm}-reader.npz", arrays)
        m.publish(out / f"{arm}-reader.json", core)
    elif mode == "replay":
        m.exact_trace(b.load_npz(out / f"{arm}-reader.npz"), arrays)
        if core != m.read(out / f"{arm}-reader.json"):
            raise ValueError("preflight semantic replay differs")
    m.publish(
        out / f"{arm}-{mode}.json",
        dict(
            pid=os.getpid(),
            core_sha256=m.digest(core),
            cache=cache_tree_sha256(Path(os.environ["JAX_COMPILATION_CACHE_DIR"])),
            metrics=metrics,
        ),
    )


def verify_preflight(root, value):
    from .online_learning import initialize, export_model

    p = value["protocol"]
    out = Path(root) / "preflight" / "000"
    base = dependency(value, 0)[0]
    initial = initialize(base, p["actor_seed"])
    reports = m.read(out / "result.json")["reports"]
    inputs = m.read(out / "reader-inputs.json")
    if inputs["manifest_sha256"] != m.digest(value) or not m.finite(reports):
        raise ValueError("invalid preflight evidence")
    for arm in ("P", "M"):
        state = read_pickle(out / f"{arm}-learner.pkl")
        model = read_pickle(out / f"{arm}-model.pkl")
        r = reports[arm]
        if (
            not finite_learner(state)
            or state["config"] != initial["config"]
            or state["rebrac_config"] != initial["rebrac_config"]
            or snapshot_digest(model)
            != snapshot_digest(export_model(state, base, p["actor_seed"]))
            or r["before"] != parameter_digests(base, p["actor_seed"])
            or r["after"] != parameter_digests(model, p["actor_seed"])
            or inputs["models"][arm] != file_digest(out / f"{arm}-model.pkl")
        ):
            raise ValueError("preflight checkpoint differs")
        validate_changes(r["before"], r["after"], arm)
        count = p["preflight_updates"] if arm == "M" else 0
        if (
            int(state["world_optimizer"].step) != count
            or int(state["reward_optimizer"].step) != count
            or int(state["policy"].step)
            != int(initial["policy"].step) + p["preflight_updates"]
            or r["losses"]["world_updates"] != count
            or r["losses"]["reward_updates"] != count
            or r["losses"]["policy_updates"] != p["preflight_updates"]
        ):
            raise ValueError("preflight update clock differs")
        ep = b.load_npz(out / f"{arm}-episode.npz")
        validate_episode(ep, p["preflight_steps"], require_native_end=True)
        core = m.read(out / f"{arm}-reader.json")
        if (
            core["arrays_sha256"]
            != b.array_sha256(b.load_npz(out / f"{arm}-reader.npz"))
            or core["model_sha256"] != inputs["models"][arm]
        ):
            raise ValueError("preflight replay artifact differs")
        receipts = []
        for mode in ("primer", "create", "replay"):
            receipt = m.read(out / f"{arm}-{mode}.json")
            if m.read(out / f"{arm}-{mode}-seal.json") != dict(
                cache=receipt["cache"],
                receipt_sha256=file_digest(out / f"{arm}-{mode}.json"),
            ):
                raise ValueError("preflight cache seal differs")
            if mode != "primer" and receipt["core_sha256"] != m.digest(core):
                raise ValueError("preflight reader differs")
            receipts.append(receipt)
        if (
            len({r["pid"] for r in receipts}) != 3
            or len({r["cache"] for r in receipts}) != 1
        ):
            raise ValueError("preflight process/cache identities differ")


def controller_diagnostics(trace):
    gain = trace["objective_after"] - trace["objective_before"]
    return dict(
        mean_predicted_objective_gain=float(np.mean(gain)),
        nonnegative_gain_fraction=float(np.mean(gain >= 0)),
        executed_action_saturation=float(np.mean(np.abs(trace["actions"]) >= 0.99)),
        clean_action_saturation=float(np.mean(np.abs(trace["clean_actions"]) >= 0.99)),
        mean_gradient_norm=float(np.mean(trace["gradient_norm"])),
        mean_parameter_delta=float(np.mean(trace["parameter_delta"])),
        mean_anchor_drift=float(np.mean(trace["anchor_drift"])),
        max_current_drift=float(np.max(trace["current_drift"])),
        trust_feasible_fraction=float(np.mean(trace["within_reference_budgets"])),
        reference_fallback_fraction=float(np.mean(trace["used_reference_fallback"])),
    )


def measured_evidence(root, p):
    training, evaluation = [], []
    for cell in cells(p, "training"):
        rows = [
            m.read(epoch_dir(root, cell, ep) / "result.json")
            for ep in range(1, p["training_steps"] // p["episode_steps"] + 1)
        ]
        training.append(
            dict(
                cell=cell,
                collection_returns=[r["return_"] for r in rows],
                positive_reward_steps=sum(r["positive_reward_steps"] for r in rows),
                collection_timed_seconds=sum(
                    r["collection"]["total_timed_seconds"] for r in rows
                ),
                learning_seconds=sum(r["learning_seconds"] for r in rows),
                epoch_wall_seconds=sum(r["wall_seconds"] for r in rows),
                updates={
                    k: sum(r["update_metrics"][k] for r in rows)
                    for k in ("world_updates", "reward_updates", "policy_updates")
                },
                final_update_metrics=rows[-1]["update_metrics"],
            )
        )
    for cell in cells(p, "evaluation"):
        out = Path(root) / "evaluation" / f"{cell['index']:03d}"
        arrays = b.load_npz(out / "trace.npz")
        diagnostics = []
        for i in range(len(p["evaluation_seeds"])):
            prefix = f"trace_{i}_"
            diagnostics.append(
                controller_diagnostics(
                    {
                        k[len(prefix) :]: v
                        for k, v in arrays.items()
                        if k.startswith(prefix)
                    }
                )
            )
        receipts = {
            mode: m.read(out / f"{mode}.json")
            for mode in ("primer", "create", "replay")
        }
        evaluation.append(
            dict(
                cell=cell,
                episode_diagnostics=diagnostics,
                collection_metrics=receipts["create"]["metrics"],
                process_wall_seconds={
                    mode: r["wall_seconds"] for mode, r in receipts.items()
                },
            )
        )
    return dict(
        training=training,
        evaluation=evaluation,
        timing_note="epoch wall includes collection compilation and serialization before verification; allocation accounting separately includes verification and startup; discarded primer/replay reported separately",
    )


def verify_stage(root, stage):
    root = Path(root)
    value = manifest(root)
    p = value["protocol"]
    submission = m.read(root / "submissions" / f"{stage}.json")
    intent = m.read(root / "submissions" / f"{stage}.intent.json")
    if submission["manifest_sha256"] != m.digest(value) or intent[
        "manifest_sha256"
    ] != m.digest(value):
        raise ValueError("submission manifest differs")
    if (
        intent["script_sha256"] != hashlib.sha256(intent["script"].encode()).hexdigest()
        or submission["script_sha256"] != intent["script_sha256"]
    ):
        raise ValueError("submission script differs")
    if (
        m.read(root / "submissions" / f"{stage}.released.json")["job_id"]
        != submission["job_id"]
    ):
        raise ValueError("submission release differs")
    expected = cells(p, stage)
    if any(
        x["stage"] != stage or x["count"] != len(expected) for x in (intent, submission)
    ):
        raise ValueError("submission matrix differs")
    acct = m.accounting(submission["job_id"], len(expected))
    markers = {}
    for cell in expected:
        directory = root / stage / f"{cell['index']:03d}"
        if stage == "training":
            base_model = dependency(value, cell["training_index"])[0]
            parent = file_digest(root / "manifest.json")
            for ep in range(1, p["training_steps"] // p["episode_steps"] + 1):
                verify_epoch(root, value, cell, ep, parent, base_model=base_model)
                parent = file_digest(epoch_dir(root, cell, ep) / "verified.json")
            complete = m.read(directory / "complete.json")
            if complete != dict(
                cell=cell,
                manifest_sha256=m.digest(value),
                final_epoch_sha256=parent,
                steps=p["training_steps"],
            ):
                raise ValueError("training completion differs")
            markers[str(cell["index"])] = file_digest(directory / "complete.json")
        else:
            context = (
                dict(stage=stage)
                if stage == "preflight"
                else dict(stage=stage, cell=cell)
            )
            verify_directory(directory, value, context)
            if stage == "preflight":
                verify_preflight(root, value)
            if stage == "evaluation":
                result = m.read(directory / "result.json")
                if result["cell"] != cell or not m.finite(result):
                    raise ValueError("invalid evaluation result")
                if (
                    result["evaluation_seeds"] != p["evaluation_seeds"]
                    or len(result["episode_returns"]) != len(p["evaluation_seeds"])
                    or result["parameters"]
                    != parameter_digests(
                        evaluation_model(root, value, cell), p["actor_seed"]
                    )
                ):
                    raise ValueError("evaluation protocol or parameters differ")
                arr = b.load_npz(directory / "trace.npz")
                if b.array_sha256(arr) != result["trace_sha256"]:
                    raise ValueError("trace differs")
                for i, ret in enumerate(result["episode_returns"]):
                    prefix = f"episode_{i}_"
                    episode = {
                        k[len(prefix) :]: v
                        for k, v in arr.items()
                        if k.startswith(prefix)
                    }
                    prefix = f"trace_{i}_"
                    trace = {
                        k[len(prefix) :]: v
                        for k, v in arr.items()
                        if k.startswith(prefix)
                    }
                    validate_episode(
                        episode, p["episode_steps"], require_native_end=True
                    )
                    validate_alignment(episode, trace, p["evaluation_seeds"][i])
                    if not np.array_equal(trace["actions"], trace["clean_actions"]):
                        raise ValueError("evaluation exploration differs")
                    if (
                        float(np.sum(arr[f"episode_{i}_rewards"], dtype=np.float64))
                        != ret
                    ):
                        raise ValueError("evaluation return differs")
                receipts = [
                    m.read(directory / f"{mode}.json")
                    for mode in ("primer", "create", "replay")
                ]
                for mode, receipt in zip(("primer", "create", "replay"), receipts):
                    if m.read(directory / f"{mode}-seal.json") != dict(
                        cache=receipt["cache"],
                        receipt_sha256=file_digest(directory / f"{mode}.json"),
                    ):
                        raise ValueError("evaluation cache seal differs")
                    if mode != "primer" and receipt["core_sha256"] != m.digest(result):
                        raise ValueError("evaluation reader binding differs")
                if len({r["pid"] for r in receipts}) != 3 or any(
                    r["cache"] != receipts[0]["cache"] for r in receipts
                ):
                    raise ValueError("evaluation process/cache differs")
            markers[str(cell["index"])] = file_digest(directory / "verified.json")
    payload = dict(
        stage=stage, manifest_sha256=m.digest(value), markers=markers, accounting=acct
    )
    path = root / "verified" / f"{stage}.json"
    if path.exists():
        if m.read(path) != payload:
            raise ValueError("stage attestation differs")
    else:
        m.publish(path, payload)
    print("ONLINE_STAGE_VERIFIED", stage, flush=True)
    return payload


def launch(root, stage):
    root = Path(root).resolve()
    value = manifest(root)
    p = value["protocol"]
    if stage != "preflight":
        verify_stage(root, STAGES[STAGES.index(stage) - 1])
    out = root / "submissions"
    intent = out / f"{stage}.intent.json"
    if intent.exists() or (out / f"{stage}.json").exists():
        raise FileExistsError("submitted or uncertain; never duplicate")
    count = len(cells(p, stage))
    e = p["execution"]
    script = "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "module purge",
            "module load Python/3.12.3-GCCcore-13.3.0",
            "source /work2/ci72buri-dreamer_imf_neurips/venv-cuda12/bin/activate",
            "export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORM_NAME=gpu JAX_ENABLE_X64=0 MUJOCO_GL=disable XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1",
            f"export PYTHONPATH={shlex.quote(str(SOURCE / 'dreamer_imf_comparison'))}:{shlex.quote(str(SOURCE / 'imf_dreamer_jax/src'))}",
            f'export JAX_COMPILATION_CACHE_DIR={shlex.quote(str(root / "caches"))}/{stage}-${{SLURM_ARRAY_JOB_ID}}-${{SLURM_ARRAY_TASK_ID}}',
            'mkdir "$JAX_COMPILATION_CACHE_DIR"',
            f'python -m {MODULE} worker --root {shlex.quote(str(root))} --stage {stage} --index "$SLURM_ARRAY_TASK_ID"',
        ]
    )
    sha = hashlib.sha256(script.encode()).hexdigest()
    m.publish(
        intent,
        dict(
            stage=stage,
            count=count,
            manifest_sha256=m.digest(value),
            script=script,
            script_sha256=sha,
        ),
    )
    (root / "logs").mkdir(exist_ok=True)
    (root / "caches").mkdir(exist_ok=True)
    cmd = [
        "sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        f"--account={e['account']}",
        f"--partition={e['partition']}",
        f"--gres=gpu:{e['gpu_type']}:1",
        f"--cpus-per-task={e['cpus']}",
        f"--mem={e['memory_gb']}G",
        f"--time={e['time_limit']}",
        f"--array=0-{count-1}%{e['concurrency']}",
        f"--job-name=imf-online-{stage}",
        f"--output={root}/logs/{stage}-%A_%a.out",
        f"--error={root}/logs/{stage}-%A_%a.err",
    ]
    response = subprocess.check_output(cmd, input=script, text=True).strip()
    job = response.split(";")[0]
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("uncertain scheduler response; preserve intent")
    m.publish(
        out / f"{stage}.json",
        dict(
            stage=stage,
            count=count,
            manifest_sha256=m.digest(value),
            job_id=job,
            script_sha256=sha,
        ),
    )
    subprocess.run(["scontrol", "release", job], check=True)
    m.publish(out / f"{stage}.released.json", dict(job_id=job))
    print("ONLINE_SUBMITTED", stage, job, flush=True)


def summarize(records, p):
    expected = cells(p, "evaluation")
    if len(records) != len(expected) or {m.digest(r["cell"]) for r in records} != {
        m.digest(c) for c in expected
    }:
        raise ValueError("incomplete/duplicate evaluation matrix")
    for r in records:
        if (
            len(r["episode_returns"]) != len(p["evaluation_seeds"])
            or not m.finite(r["episode_returns"])
            or r.get("evaluation_seeds", p["evaluation_seeds"]) != p["evaluation_seeds"]
        ):
            raise ValueError("evaluation sample set differs")
    values = {
        (r["cell"]["world_model_seed"], r["cell"]["arm"], r["cell"]["steps"]): float(
            np.mean(r["episode_returns"])
        )
        for r in records
    }
    curves = {}
    for arm in p["arms"]:
        curves[arm] = {
            str(step): [
                values[w, "F", 0] if arm == "F" or step == 0 else values[w, arm, step]
                for w in p["world_model_seeds"]
            ]
            for step in p["evaluation_steps"]
        }
    contrasts = {}
    for a, c in (("P", "F"), ("M", "F"), ("M", "P")):
        delta = (
            np.asarray(curves[a][str(p["training_steps"])])
            - curves[c][str(p["training_steps"])]
        )
        contrasts[f"{a}_minus_{c}"] = dict(
            world_seed_deltas=delta.tolist(),
            mean=float(delta.mean()),
            favorable_fraction=float(np.mean(delta > 0)),
        )
    return dict(
        curves=curves,
        contrasts=contrasts,
        normalized_learning_curve_area={
            arm: [
                float(
                    np.trapezoid(
                        [curves[arm][str(s)][i] for s in p["evaluation_steps"]],
                        p["evaluation_steps"],
                    )
                    / p["training_steps"]
                )
                for i in range(3)
            ]
            for arm in p["arms"]
        },
        statistical_unit="three paired pretrained world-model seeds; actor and five evaluation episodes nested",
        limitations=[
            "exploratory one-task offline-to-online fine-tuning, not independent confirmation or from-scratch benchmark",
            "P/M have equal additional real steps and policy updates, unequal total compute and evolving occupancies",
            "F parameters frozen, but A1 inference adaptation still executes; original training compute is additional",
            "fixed old normalization and anchors, retained ReBRAC BC penalties, fresh world/reward Adam states",
            "no best-checkpoint selection; sparse rewards can leave learning unsuccessful",
        ],
    )


def finalize(root):
    root = Path(root)
    value = manifest(root)
    stages = {s: verify_stage(root, s) for s in STAGES}
    records = [
        m.read(root / "evaluation" / f"{c['index']:03d}" / "result.json")
        for c in cells(value["protocol"], "evaluation")
    ]
    result = dict(
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        analysis=summarize(records, value["protocol"]),
        stages=stages,
        measured_evidence=measured_evidence(root, value["protocol"]),
    )
    path = root / "report.json"
    if path.exists():
        if m.read(path) != result:
            raise ValueError("final report differs")
    else:
        m.publish(path, result)
    marker = dict(report_sha256=file_digest(path), manifest_sha256=m.digest(value))
    if (root / "verified.json").exists():
        if m.read(root / "verified.json") != marker:
            raise ValueError("final marker differs")
    else:
        m.publish(root / "verified.json", marker)
    print("ONLINE_BENCHMARK_FINAL_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "register",
            "launch",
            "worker",
            "role",
            "preflight-role",
            "verify",
            "finalize",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dependency", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=("primer", "create", "replay"))
    args = parser.parse_args()
    if args.command == "register":
        register(args.root, args.dependency)
    elif args.command == "launch":
        launch(args.root, args.stage)
    elif args.command == "verify":
        verify_stage(args.root, args.stage)
    elif args.command == "role":
        role(args.root, args.index, args.mode)
    elif args.command == "preflight-role":
        preflight_role(args.root, args.index, args.mode)
    elif args.command == "finalize":
        finalize(args.root)
    elif args.stage == "preflight":
        preflight(args.root)
    elif args.stage == "training":
        train(args.root, args.index)
    elif args.stage == "evaluation":
        evaluate(args.root, args.index)
    else:
        parser.error("worker requires stage")


if __name__ == "__main__":
    main()
