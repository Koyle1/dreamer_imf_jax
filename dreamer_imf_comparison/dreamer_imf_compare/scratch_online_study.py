"""Exclusive, dependency-free online trajectory-iMF benchmark.

Only newly collected, budgeted training episodes enter initialization statistics
or replay. Evaluation runs in independent compiler-cache reader processes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import numpy as np

from . import online_training_study as u
from . import mechanism_replication as m
from . import matched_objective_benchmark as b
from .cache_fingerprint import cache_tree_sha256

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "dreamer_imf_comparison/scratch_online_protocol.json"
MODULE = "dreamer_imf_compare.scratch_online_study"
STAGES = ("preflight", "training", "evaluation")
MODES = ("primer", "create", "replay")


def validate_protocol(p):
    if (
        p["schema"] != "trajectory-imf-scratch-online-v1"
        or p["task"] != "dmc_reacher_hard"
        or p["action_repeat"] != 2
        or p["training_native_steps"] != 500000
        or p["episode_native_steps"] != 1000
        or p["prefill_native_steps"] != 5000
        or p["evaluation_native_steps"] != list(range(100000, 500001, 100000))
        or len(set(p["world_model_seeds"])) != 3
        or len(set(p["evaluation_seeds"])) != 5
        or p["updates_per_episode"] != 250
    ):
        raise ValueError("scratch budget/protocol differs")
    training = [
        training_seed(p, w, i) for w in p["world_model_seeds"] for i in range(1, 501)
    ]
    if len(set(training)) != 1500:
        raise ValueError("training seed collision")


def training_seed(p, seed, episode):
    value = b.derive_seed(p["training_seed_namespace"], seed, episode)
    if value in p["evaluation_seeds"] + [p["preflight_seed"]]:
        raise ValueError("training/evaluation seed collision")
    return value


def cells(p, stage):
    if stage == "preflight":
        return [dict(index=0, world_model_seed=p["world_model_seeds"][0])]
    if stage == "training":
        return [
            dict(index=i, world_model_seed=w)
            for i, w in enumerate(p["world_model_seeds"])
        ]
    if stage == "evaluation":
        rows = [
            dict(training_index=i, world_model_seed=w, native_steps=s)
            for i, w in enumerate(p["world_model_seeds"])
            for s in p["evaluation_native_steps"]
        ]
        return [dict(index=i, **c) for i, c in enumerate(rows)]
    raise ValueError("unknown stage")


def manifest(root):
    v = m.read(Path(root) / "manifest.json")
    if (
        v["source_commit"] != m.clean_commit()
        or v["source_root"] != str(SOURCE)
        or v["protocol"] != m.read(PROTOCOL)
        or v["protocol_sha256"] != m.digest(v["protocol"])
        or v["dependencies"] != []
    ):
        raise ValueError("scratch source/protocol/dependency identity differs")
    validate_protocol(v["protocol"])
    return v


def register(root):
    root = Path(root).resolve()
    p = m.read(PROTOCOL)
    validate_protocol(p)
    v = dict(
        source_commit=m.clean_commit(),
        source_root=str(SOURCE),
        protocol=p,
        protocol_sha256=m.digest(p),
        dependencies=[],
    )
    root.mkdir(parents=True, exist_ok=False)
    m.publish(root / "manifest.json", v)
    (root / "submissions").mkdir()
    (root / "verified").mkdir()
    print("SCRATCH_ONLINE_REGISTERED", m.digest(v), flush=True)


def context(stage, cell, episode=None):
    return dict(stage=stage, cell=cell, episode=episode)


def epoch_dir(root, cell, episode):
    return Path(root) / "training" / f"{cell['index']:03d}" / f"epoch-{episode:03d}"


def stack_episodes(episodes):
    return {k: np.concatenate([e[k] for e in episodes], axis=0) for k in episodes[0]}


def split_episodes(arrays):
    return [
        {k: v[i : i + 1] for k, v in arrays.items()}
        for i in range(arrays["rewards"].shape[0])
    ]


def validate_collection(ep, tr, seed, decisions, native_steps, *, complete):
    u.validate_episode(ep, decisions, require_native_end=complete)
    u.validate_alignment(ep, tr, seed)
    counts = np.asarray(tr["native_steps"])
    if counts.shape != (1, decisions) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("native count shape/type differs")
    if np.any(counts < 1) or np.any(counts > 2) or int(counts.sum()) != native_steps:
        raise ValueError("native budget differs")


def fresh_model(p, seed, episodes=None):
    from .scratch_initialization import initialize_model

    # Only caller-validated fresh collection reaches this internal constructor.
    tagged = None if episodes is None else [{**ep, "training": True} for ep in episodes]
    return initialize_model(p, seed, tagged)


def clocks(state, count):
    if not u.finite_learner(state):
        raise ValueError("nonfinite learner")
    if any(
        int(state[k].step) != count for k in ("world_optimizer", "reward_optimizer")
    ):
        raise ValueError("world/reward clock differs")
    pol = state["policy"]
    if (
        int(pol.step) != count
        or int(pol.critic_optimizer.step) != count
        or int(pol.actor_optimizer.step) != (count + 1) // 2
    ):
        raise ValueError("policy clock differs")


def bootstrap(root, value, cell):
    from .online_collector import make_collector
    from .online_learning import initialize

    p = value["protocol"]
    out = Path(root) / "training" / f"{cell['index']:03d}" / "bootstrap"
    if out.exists():
        return verify_bootstrap(root, value, cell)
    out.mkdir(parents=True, exist_ok=False)
    model = fresh_model(p, cell["world_model_seed"])
    collector = make_collector(model["config"], model["rebrac_config"], p["controller"])
    episodes, traces, metrics = [], [], []
    for i in range(1, 6):
        seed = training_seed(p, cell["world_model_seed"], i)
        ep, tr, timing = collector.rollout(
            model,
            p["actor_seed"],
            cell["world_model_seed"],
            seed,
            maximum_steps=500,
            action_repeat=2,
            training=True,
            random_policy=True,
        )
        validate_collection(ep, tr, seed, 500, 1000, complete=True)
        episodes.append(ep)
        traces.append(tr)
        metrics.append(timing)
    base = fresh_model(p, cell["world_model_seed"], episodes)
    clocks(initialize(base, p["actor_seed"]), 0)
    u.write_pickle(out / "initial.pkl", base)
    u.write_npz(out / "episodes.npz", stack_episodes(episodes))
    u.write_npz(
        out / "traces.npz",
        {f"{i}_{k}": v for i, tr in enumerate(traces) for k, v in tr.items()},
    )
    m.publish(
        out / "result.json",
        dict(
            native_steps=5000,
            decision_steps=2500,
            initialization="random_no_dependencies",
            metrics=metrics,
            initial_digest=u.snapshot_digest(base),
        ),
    )
    u.seal(out, value, context("bootstrap", cell))
    return verify_bootstrap(root, value, cell)


def verify_bootstrap(root, value, cell):
    from .online_learning import initialize

    out = Path(root) / "training" / f"{cell['index']:03d}" / "bootstrap"
    mark = u.verify_directory(out, value, context("bootstrap", cell))
    if set(mark["files"]) != {
        "initial.pkl",
        "episodes.npz",
        "traces.npz",
        "result.json",
    }:
        raise ValueError("bootstrap inventory differs")
    ep = split_episodes(b.load_npz(out / "episodes.npz"))
    tr = b.load_npz(out / "traces.npz")
    if len(ep) != 5:
        raise ValueError("prefill count differs")
    p = value["protocol"]
    for i, e in enumerate(ep):
        trace = {k[len(f"{i}_") :]: v for k, v in tr.items() if k.startswith(f"{i}_")}
        validate_collection(
            e,
            trace,
            training_seed(p, cell["world_model_seed"], i + 1),
            500,
            1000,
            complete=True,
        )
    base = u.read_pickle(out / "initial.pkl")
    regenerated = fresh_model(p, cell["world_model_seed"], ep)
    r = m.read(out / "result.json")
    if (
        u.snapshot_digest(base) != u.snapshot_digest(regenerated)
        or r["initial_digest"] != u.snapshot_digest(base)
        or r["native_steps"] != 5000
        or r["decision_steps"] != 2500
        or not m.finite(r)
    ):
        raise ValueError("fresh initialization differs")
    clocks(initialize(base, p["actor_seed"]), 0)
    return base, ep, u.file_digest(out / "verified.json")


def train(root, index):
    from .online_collector import make_collector
    from .online_learning import initialize, update, export_model

    value = manifest(root)
    p = value["protocol"]
    cell = cells(p, "training")[index]
    base, episodes, parent = bootstrap(root, value, cell)
    state = initialize(base, p["actor_seed"])
    collector = make_collector(base["config"], base["rebrac_config"], p["controller"])
    for number in range(5, 501):
        out = epoch_dir(root, cell, number)
        if out.exists():
            verify_epoch(root, value, cell, number, parent, base)
            state = u.read_pickle(out / "learner.pkl")
            if number > 5:
                episodes.append(b.load_npz(out / "episode.npz"))
            parent = u.file_digest(out / "verified.json")
            continue
        out.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        current = export_model(state, base, p["actor_seed"])
        before = u.parameter_digests(current, p["actor_seed"])
        timing = None
        if number > 5:
            seed = training_seed(p, cell["world_model_seed"], number)
            ep, tr, timing = collector.rollout(
                current,
                p["actor_seed"],
                cell["world_model_seed"],
                seed,
                maximum_steps=500,
                action_repeat=2,
                training=True,
                exploration_std=p["exploration_std"],
            )
            validate_collection(ep, tr, seed, 500, 1000, complete=True)
            episodes.append(ep)
            u.write_npz(out / "episode.npz", ep)
            u.write_npz(out / "trace.npz", tr)
        count = p["updates_per_episode"] * (5 if number == 5 else 1)
        learning_started = time.perf_counter()
        state, metrics = update(
            state,
            None,
            episodes,
            full_model=True,
            seed=b.derive_seed(
                "scratch-online-update", cell["world_model_seed"], number
            ),
            world_updates=count,
            policy_updates=count,
            batch_size=p["batch_size"],
            sequence_length=p["sequence_length"],
        )
        learning_seconds = time.perf_counter() - learning_started
        clocks(state, number * p["updates_per_episode"])
        model = export_model(state, base, p["actor_seed"])
        after = u.parameter_digests(model, p["actor_seed"])
        u.validate_changes(before, after, "M")
        u.write_pickle(out / "learner.pkl", state)
        if number * 1000 in p["evaluation_native_steps"]:
            u.write_pickle(out / "model.pkl", model)
        m.publish(
            out / "result.json",
            dict(
                cell=cell,
                episode=number,
                native_steps=number * 1000,
                decision_steps=number * 500,
                parent_sha256=parent,
                before=before,
                after=after,
                update_metrics=metrics,
                collection=timing,
                learning_seconds=learning_seconds,
                return_=(
                    None if number == 5 else float(ep["rewards"].sum(dtype=np.float64))
                ),
                positive_reward_decisions=(
                    None if number == 5 else int(np.sum(ep["rewards"] > 0))
                ),
                wall_seconds=time.perf_counter() - started,
            ),
        )
        u.seal(out, value, context("epoch", cell, number))
        verify_epoch(root, value, cell, number, parent, base)
        parent = u.file_digest(out / "verified.json")
        print("SCRATCH_EPOCH_VERIFIED", index, number, number * 1000, flush=True)
    m.publish(
        Path(root) / "training" / f"{index:03d}" / "complete.json",
        dict(
            cell=cell,
            manifest_sha256=m.digest(value),
            final_sha256=parent,
            native_steps=500000,
            decision_steps=250000,
        ),
    )


def verify_epoch(root, value, cell, number, parent, base):
    from .online_learning import export_model

    p = value["protocol"]
    out = epoch_dir(root, cell, number)
    mark = u.verify_directory(out, value, context("epoch", cell, number))
    expected = {"learner.pkl", "result.json"}
    if number > 5:
        expected |= {"episode.npz", "trace.npz"}
    if number * 1000 in p["evaluation_native_steps"]:
        expected.add("model.pkl")
    if set(mark["files"]) != expected:
        raise ValueError("epoch inventory differs")
    r = m.read(out / "result.json")
    if (
        r["cell"] != cell
        or r["episode"] != number
        or r["native_steps"] != number * 1000
        or r["decision_steps"] != number * 500
        or r["parent_sha256"] != parent
        or not m.finite(r)
    ):
        raise ValueError("epoch identity/budget differs")
    state = u.read_pickle(out / "learner.pkl")
    clocks(state, number * p["updates_per_episode"])
    if (
        state["config"] != base["config"]
        or state["rebrac_config"] != base["rebrac_config"]
    ):
        raise ValueError("learner config differs")
    model = export_model(state, base, p["actor_seed"])
    if u.parameter_digests(model, p["actor_seed"]) != r["after"]:
        raise ValueError("learner digest differs")
    before = (
        u.parameter_digests(base, p["actor_seed"])
        if number == 5
        else m.read(epoch_dir(root, cell, number - 1) / "result.json")["after"]
    )
    if before != r["before"]:
        raise ValueError("parameter ancestry differs")
    u.validate_changes(r["before"], r["after"], "M")
    count = p["updates_per_episode"] * (5 if number == 5 else 1)
    if r["update_metrics"]["offline_fraction"] != 0 or any(
        r["update_metrics"][k] != count
        for k in ("world_updates", "reward_updates", "policy_updates")
    ):
        raise ValueError("online-only update budget differs")
    if "model.pkl" in expected and u.snapshot_digest(
        u.read_pickle(out / "model.pkl")
    ) != u.snapshot_digest(model):
        raise ValueError("snapshot differs")
    if number > 5:
        ep = b.load_npz(out / "episode.npz")
        tr = b.load_npz(out / "trace.npz")
        validate_collection(
            ep,
            tr,
            training_seed(p, cell["world_model_seed"], number),
            500,
            1000,
            complete=True,
        )
        if r["collection"]["native_steps"] != 1000 or r["return_"] != float(
            ep["rewards"].sum(dtype=np.float64)
        ):
            raise ValueError("collection evidence differs")


def reader_model(root, value, stage, index):
    if stage == "preflight":
        out = Path(root) / "preflight" / "000"
        inputs = m.read(out / "inputs.json")
        if inputs["model_sha256"] != u.file_digest(out / "model.pkl") or inputs[
            "manifest_sha256"
        ] != m.digest(value):
            raise ValueError("preflight input binding differs")
        return u.read_pickle(out / "model.pkl")
    cell = cells(value["protocol"], "evaluation")[index]
    tc = cells(value["protocol"], "training")[cell["training_index"]]
    out = epoch_dir(root, tc, cell["native_steps"] // 1000)
    u.verify_directory(out, value, context("epoch", tc, cell["native_steps"] // 1000))
    return u.read_pickle(out / "model.pkl")


def compute_evaluation(root, value, stage, index):
    from .online_collector import make_collector

    p = value["protocol"]
    cell = cells(p, stage)[index]
    model = reader_model(root, value, stage, index)
    collector = make_collector(model["config"], model["rebrac_config"], p["controller"])
    seeds = [p["preflight_seed"]] if stage == "preflight" else p["evaluation_seeds"]
    decisions = p["preflight_decisions"] if stage == "preflight" else 500
    arrays, metrics, returns = {}, [], []
    for i, seed in enumerate(seeds):
        ep, tr, timing = collector.rollout(
            model,
            p["actor_seed"],
            cell["world_model_seed"],
            seed,
            maximum_steps=decisions,
            action_repeat=2,
        )
        validate_collection(
            ep, tr, seed, decisions, 2 * decisions, complete=stage == "evaluation"
        )
        arrays.update({f"episode_{i}_{k}": v for k, v in ep.items()})
        arrays.update({f"trace_{i}_{k}": v for k, v in tr.items()})
        metrics.append(timing)
        returns.append(float(ep["rewards"].sum(dtype=np.float64)))
    core = dict(
        cell=cell,
        evaluation_seeds=seeds,
        episode_returns=returns,
        native_steps_per_episode=2 * decisions,
        decisions_per_episode=decisions,
        parameters=u.parameter_digests(model, p["actor_seed"]),
        trace_sha256=b.array_sha256(arrays),
    )
    return core, arrays, metrics


def role(root, stage, index, mode):
    value = manifest(root)
    out = Path(root) / stage / f"{index:03d}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    core, arrays, metrics = compute_evaluation(root, value, stage, index)
    if mode == "create":
        m.publish(out / "result.json", core)
        u.write_npz(out / "trace.npz", arrays)
    if mode == "replay":
        if core != m.read(out / "result.json"):
            raise ValueError("scratch semantic replay differs")
        m.exact_trace(b.load_npz(out / "trace.npz"), arrays)
    m.publish(
        out / f"{mode}.json",
        dict(
            pid=os.getpid(),
            core_sha256=m.digest(core),
            cache=cache_tree_sha256(Path(os.environ["JAX_COMPILATION_CACHE_DIR"])),
            metrics=metrics,
            wall_seconds=time.perf_counter() - started,
            runtime=m.runtime(),
        ),
    )


def readers(root, value, stage, index):
    out = Path(root) / stage / f"{index:03d}"
    for mode in MODES:
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
                "--mode",
                mode,
            ],
            check=True,
        )
        receipt = m.read(out / f"{mode}.json")
        cache = cache_tree_sha256(Path(os.environ["JAX_COMPILATION_CACHE_DIR"]))
        if cache != receipt["cache"]:
            raise ValueError("post-exit cache differs")
        m.publish(
            out / f"{mode}-seal.json",
            dict(cache=cache, receipt_sha256=u.file_digest(out / f"{mode}.json")),
        )
    verify_readers(root, value, stage, index)
    u.seal(out, value, context(stage, cells(value["protocol"], stage)[index]))
    print("SCRATCH_ONLINE_CELL_VERIFIED", stage, index, flush=True)


def verify_readers(root, value, stage, index):
    p = value["protocol"]
    out = Path(root) / stage / f"{index:03d}"
    core = m.read(out / "result.json")
    arrays = b.load_npz(out / "trace.npz")
    cell = cells(p, stage)[index]
    seeds = [p["preflight_seed"]] if stage == "preflight" else p["evaluation_seeds"]
    decisions = p["preflight_decisions"] if stage == "preflight" else 500
    if (
        core["cell"] != cell
        or core["evaluation_seeds"] != seeds
        or not m.finite(core)
        or core["trace_sha256"] != b.array_sha256(arrays)
        or len(core["episode_returns"]) != len(seeds)
        or core["native_steps_per_episode"] != 2 * decisions
        or core["decisions_per_episode"] != decisions
        or core["parameters"]
        != u.parameter_digests(reader_model(root, value, stage, index), p["actor_seed"])
    ):
        raise ValueError("evaluation input/protocol differs")
    for i, seed in enumerate(seeds):
        ep = {
            k[len(f"episode_{i}_") :]: v
            for k, v in arrays.items()
            if k.startswith(f"episode_{i}_")
        }
        tr = {
            k[len(f"trace_{i}_") :]: v
            for k, v in arrays.items()
            if k.startswith(f"trace_{i}_")
        }
        validate_collection(
            ep, tr, seed, decisions, decisions * 2, complete=stage == "evaluation"
        )
        if float(ep["rewards"].sum(dtype=np.float64)) != core["episode_returns"][
            i
        ] or not np.array_equal(tr["actions"], tr["clean_actions"]):
            raise ValueError("evaluation return/action differs")
    receipts = []
    for mode in MODES:
        r = m.read(out / f"{mode}.json")
        receipts.append(r)
        if m.read(out / f"{mode}-seal.json") != dict(
            cache=r["cache"], receipt_sha256=u.file_digest(out / f"{mode}.json")
        ):
            raise ValueError("reader seal differs")
        if mode != "primer" and r["core_sha256"] != m.digest(core):
            raise ValueError("reader core differs")
    if (
        len({r["pid"] for r in receipts}) != 3
        or len({r["cache"] for r in receipts}) != 1
    ):
        raise ValueError("reader process/cache differs")


def preflight(root):
    from .online_collector import make_collector
    from .online_learning import initialize, update, export_model

    value = manifest(root)
    p = value["protocol"]
    seed = p["world_model_seeds"][0]
    out = Path(root) / "preflight" / "000"
    out.mkdir(parents=True, exist_ok=False)
    base = fresh_model(p, seed)
    collector = make_collector(base["config"], base["rebrac_config"], p["controller"])
    ep, tr, timing = collector.rollout(
        base,
        p["actor_seed"],
        seed,
        p["preflight_seed"],
        maximum_steps=p["preflight_decisions"],
        action_repeat=2,
        training=True,
        random_policy=True,
    )
    validate_collection(
        ep,
        tr,
        p["preflight_seed"],
        p["preflight_decisions"],
        p["preflight_decisions"] * 2,
        complete=False,
    )
    base = fresh_model(p, seed, [ep])
    state = initialize(base, p["actor_seed"])
    clocks(state, 0)
    count = p["preflight_updates"]
    state, metrics = update(
        state,
        None,
        [ep],
        full_model=True,
        seed=seed,
        world_updates=count,
        policy_updates=count,
        batch_size=p["batch_size"],
        sequence_length=p["sequence_length"],
    )
    clocks(state, count)
    model = export_model(state, base, p["actor_seed"])
    u.validate_changes(
        u.parameter_digests(base, p["actor_seed"]),
        u.parameter_digests(model, p["actor_seed"]),
        "M",
    )
    u.write_pickle(out / "model.pkl", model)
    u.write_pickle(out / "learner.pkl", state)
    u.write_npz(out / "prefill.npz", ep)
    u.write_npz(out / "prefill-trace.npz", tr)
    m.publish(
        out / "inputs.json",
        dict(
            manifest_sha256=m.digest(value),
            model_sha256=u.file_digest(out / "model.pkl"),
            metrics=metrics,
            collection=timing,
        ),
    )
    # World training and random collection may have populated this cache. Give
    # the discarded controller primer its own exact-shape fresh cache.
    old = Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
    new = old.with_name(old.name + "-readers")
    new.mkdir(exist_ok=False)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(new)
    readers(root, value, "preflight", 0)
    print("SCRATCH_ONLINE_PREFLIGHT_VERIFIED", flush=True)


def verify_preflight(root, value):
    from .online_learning import initialize, export_model

    p = value["protocol"]
    out = Path(root) / "preflight" / "000"
    ep = b.load_npz(out / "prefill.npz")
    tr = b.load_npz(out / "prefill-trace.npz")
    validate_collection(
        ep,
        tr,
        p["preflight_seed"],
        p["preflight_decisions"],
        2 * p["preflight_decisions"],
        complete=False,
    )
    base = fresh_model(p, p["world_model_seeds"][0], [ep])
    clocks(initialize(base, p["actor_seed"]), 0)
    state = u.read_pickle(out / "learner.pkl")
    clocks(state, p["preflight_updates"])
    model = reader_model(root, value, "preflight", 0)
    if u.snapshot_digest(model) != u.snapshot_digest(
        export_model(state, base, p["actor_seed"])
    ):
        raise ValueError("preflight learner differs")
    metrics = m.read(out / "inputs.json")["metrics"]
    if metrics["offline_fraction"] != 0 or any(
        metrics[k] != p["preflight_updates"]
        for k in ("world_updates", "reward_updates", "policy_updates")
    ):
        raise ValueError("preflight update/data scope differs")
    u.validate_changes(
        u.parameter_digests(base, p["actor_seed"]),
        u.parameter_digests(model, p["actor_seed"]),
        "M",
    )
    verify_readers(root, value, "preflight", 0)


def verify_stage(root, stage):
    root = Path(root)
    value = manifest(root)
    p = value["protocol"]
    sub = m.read(root / "submissions" / f"{stage}.json")
    intent = m.read(root / "submissions" / f"{stage}.intent.json")
    if (
        sub["manifest_sha256"] != m.digest(value)
        or intent["manifest_sha256"] != m.digest(value)
        or sub["script_sha256"] != hashlib.sha256(intent["script"].encode()).hexdigest()
        or sub["script_sha256"] != intent["script_sha256"]
        or m.read(root / "submissions" / f"{stage}.released.json")["job_id"]
        != sub["job_id"]
        or any(
            x["count"] != len(cells(p, stage)) or x["stage"] != stage
            for x in (sub, intent)
        )
    ):
        raise ValueError("submission binding differs")
    acct = m.accounting(sub["job_id"], len(cells(p, stage)))
    markers = {}
    for cell in cells(p, stage):
        out = root / stage / f"{cell['index']:03d}"
        if stage == "training":
            base, _, parent = verify_bootstrap(root, value, cell)
            for number in range(5, 501):
                verify_epoch(root, value, cell, number, parent, base)
                parent = u.file_digest(epoch_dir(root, cell, number) / "verified.json")
            expected = dict(
                cell=cell,
                manifest_sha256=m.digest(value),
                final_sha256=parent,
                native_steps=500000,
                decision_steps=250000,
            )
            if m.read(out / "complete.json") != expected:
                raise ValueError("training completion differs")
            markers[str(cell["index"])] = u.file_digest(out / "complete.json")
        else:
            u.verify_directory(out, value, context(stage, cell))
            if stage == "preflight":
                verify_preflight(root, value)
            else:
                verify_readers(root, value, stage, cell["index"])
            markers[str(cell["index"])] = u.file_digest(out / "verified.json")
    payload = dict(
        stage=stage, manifest_sha256=m.digest(value), markers=markers, accounting=acct
    )
    target = root / "verified" / f"{stage}.json"
    if target.exists():
        if m.read(target) != payload:
            raise ValueError("stage attestation differs")
    else:
        m.publish(target, payload)
    print("SCRATCH_ONLINE_STAGE_VERIFIED", stage, flush=True)
    return payload


def launch(root, stage):
    root = Path(root).resolve()
    value = manifest(root)
    p = value["protocol"]
    if stage != "preflight":
        verify_stage(root, STAGES[STAGES.index(stage) - 1])
    out = root / "submissions"
    if (out / f"{stage}.intent.json").exists() or (out / f"{stage}.json").exists():
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
        out / f"{stage}.intent.json",
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
    response = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--hold",
            "--no-requeue",
            f"--account={e['account']}",
            f"--partition={e['partition']}",
            f"--gres=gpu:{e['gpu_type']}:1",
            f"--cpus-per-task={e['cpus']}",
            f"--mem={e['memory_gb']}G",
            f"--time={e['time_limits'][stage]}",
            f"--array=0-{count-1}%{e['concurrency']}",
            f"--job-name=imf-scratch-{stage}",
            f"--output={root}/logs/{stage}-%A_%a.out",
            f"--error={root}/logs/{stage}-%A_%a.err",
        ],
        input=script,
        text=True,
    ).strip()
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
    print("SCRATCH_ONLINE_SUBMITTED", stage, job, flush=True)


def summarize(records, p):
    expected = cells(p, "evaluation")
    if len(records) != len(expected) or {m.digest(r["cell"]) for r in records} != {
        m.digest(c) for c in expected
    }:
        raise ValueError("incomplete/duplicate matrix")
    for r in records:
        if (
            r["evaluation_seeds"] != p["evaluation_seeds"]
            or len(r["episode_returns"]) != 5
            or not m.finite(r)
        ):
            raise ValueError("evaluation sample set differs")
    vals = {
        (r["cell"]["world_model_seed"], r["cell"]["native_steps"]): float(
            np.mean(r["episode_returns"])
        )
        for r in records
    }
    curves = {
        str(s): [vals[w, s] for w in p["world_model_seeds"]]
        for s in p["evaluation_native_steps"]
    }
    final = np.asarray(curves["500000"])
    reference = p["published_reference"]["reported_return"]
    return dict(
        curves=curves,
        final_seed_returns=final.tolist(),
        final_mean=float(final.mean()),
        final_median=float(np.median(final)),
        published_reference=p["published_reference"],
        final_minus_published_by_seed=(final - reference).tolist(),
        final_mean_minus_published=float(final.mean() - reference),
        fraction_seeds_above_published=float(np.mean(final > reference)),
        change_100k_to_500k=(final - np.asarray(curves["100000"])).tolist(),
        statistical_unit="three independently initialized training seeds; five episodes nested per checkpoint",
        limitations=[
            "one task and three seeds, exploratory evidence not significance or general superiority",
            "published reference, not an independently reproduced DreamerV3 baseline; architecture, learner, replay ratio, parallel environments and compute differ",
            "ReBRAC behavioral regularization and inference-time A1 are retained; no guarantee online learning succeeds",
            "random prefill included in 500k native budget; evaluation and separate preflight interactions excluded and disclosed",
            "fixed checkpoints, no best-checkpoint selection; no pretrained weights, offline replay or inherited statistics",
        ],
    )


def finalize(root):
    root = Path(root)
    value = manifest(root)
    p = value["protocol"]
    stages = {s: verify_stage(root, s) for s in STAGES}
    records = [
        m.read(root / "evaluation" / f"{c['index']:03d}" / "result.json")
        for c in cells(p, "evaluation")
    ]
    training, evaluations = [], []
    for cell in cells(p, "training"):
        out = root / "training" / f"{cell['index']:03d}"
        rows = [m.read(epoch_dir(root, cell, n) / "result.json") for n in range(5, 501)]
        boot = m.read(out / "bootstrap" / "result.json")
        training.append(
            dict(
                cell=cell,
                bootstrap=boot,
                native_steps=500000,
                decision_steps=250000,
                final_metrics=rows[-1]["update_metrics"],
                updates={
                    k: sum(r["update_metrics"][k] for r in rows)
                    for k in ("world_updates", "reward_updates", "policy_updates")
                },
                learning_seconds=sum(r["learning_seconds"] for r in rows),
                epoch_wall_seconds=sum(r["wall_seconds"] for r in rows),
                collection_returns=[
                    r["return_"] for r in rows if r["return_"] is not None
                ],
                positive_reward_decisions=sum(
                    r["positive_reward_decisions"] or 0 for r in rows
                ),
            )
        )
    for cell in cells(p, "evaluation"):
        out = root / "evaluation" / f"{cell['index']:03d}"
        arr = b.load_npz(out / "trace.npz")
        diagnostics = [
            u.controller_diagnostics(
                {
                    k[len(f"trace_{i}_") :]: v
                    for k, v in arr.items()
                    if k.startswith(f"trace_{i}_")
                }
            )
            for i in range(5)
        ]
        evaluations.append(
            dict(
                cell=cell,
                diagnostics=diagnostics,
                receipts={mode: m.read(out / f"{mode}.json") for mode in MODES},
            )
        )
    report = dict(
        source_commit=value["source_commit"],
        manifest_sha256=m.digest(value),
        analysis=summarize(records, p),
        stages=stages,
        training=training,
        evaluation=evaluations,
    )
    if not m.finite(report):
        raise ValueError("nonfinite report")
    path = root / "report.json"
    if path.exists():
        if m.read(path) != report:
            raise ValueError("report differs")
    else:
        m.publish(path, report)
    mark = dict(report_sha256=u.file_digest(path), manifest_sha256=m.digest(value))
    if (root / "verified.json").exists():
        if m.read(root / "verified.json") != mark:
            raise ValueError("final marker differs")
    else:
        m.publish(root / "verified.json", mark)
    print("SCRATCH_ONLINE_FINAL_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("register", "launch", "worker", "role", "verify", "finalize"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    if args.command == "register":
        register(args.root)
    elif args.command == "launch":
        launch(args.root, args.stage)
    elif args.command == "verify":
        verify_stage(args.root, args.stage)
    elif args.command == "finalize":
        finalize(args.root)
    elif args.command == "role":
        role(args.root, args.stage, args.index, args.mode)
    elif args.stage == "preflight":
        preflight(args.root)
    elif args.stage == "training":
        train(args.root, args.index)
    elif args.stage == "evaluation":
        if (args.root / "evaluation" / f"{args.index:03d}").exists():
            raise FileExistsError("existing evaluation; never overwrite")
        readers(args.root, manifest(args.root), "evaluation", args.index)
    else:
        parser.error("worker requires stage")


if __name__ == "__main__":
    main()
