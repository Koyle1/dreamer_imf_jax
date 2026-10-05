"""Immutable registration and gated execution for the isolated trajectory study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from .parallel_collection import _write_json, _write_npz, BudgetLedger, collect

SOURCE = Path(__file__).resolve().parents[2]
BASE = Path("/work2/ci72buri-dreamer_imf_neurips")
PROTOCOL = SOURCE / "dreamer_imf_comparison/parallel_protocol.json"
STAGES = ("preflight", "collect", "fit", "evaluate", "verify")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def clean_commit(path):
    if subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError(f"dirty source {path}")
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def register(root, budget_path, bundle_path=None):
    p = read(PROTOCOL)
    root = Path(root).resolve()
    sources = dict(
        imf=BASE / "imf-conditional-source-89dc0d7",
        categorical=BASE / "dreamer-ablation-source-6b3d394",
    )
    cells = dict(
        imf=BASE / "imf-conditional-study" / p["imf_commit"] / "cells/imf/seed_431",
        categorical=BASE
        / "dreamer-ablation-study"
        / p["categorical_commit"]
        / "cells/categorical/seed_431",
    )
    bindings = {}
    for arm, cell in cells.items():
        study = cell.parents[2]
        original = read(study / "manifest.json")
        if original["source_commit"] != p[f"{arm}_commit"]:
            raise ValueError("dependency source mismatch")
        sources[arm] = Path(original["source_root"])
        if clean_commit(sources[arm]) != p[f"{arm}_commit"]:
            raise ValueError("dependency checkout mismatch")
        # progress.json does not contain checkpoint hashes. Both preregistered
        # parent jobs are complete; authenticate the predetermined 200k file
        # against their immutable completion records, never a later checkpoint.
        complete = read(cell / "complete.json")
        if (
            complete.get("completed") is not True
            or complete.get("arm") != arm
            or complete.get("seed") != p["parent_seed"]
            or complete.get("upstream_commit") != p["upstream_commit"]
        ):
            raise ValueError("parent completion identity mismatch")
        step = p["checkpoint_native_steps"]
        candidates = [v for v in complete["checkpoints"] if v["native_steps"] == step]
        if (
            len(candidates) != 1
            or candidates[0]["path"] != f"checkpoint_{step}.pkl"
            or sha(cell / candidates[0]["path"]) != candidates[0]["sha256"]
        ):
            raise ValueError("checkpoint not authenticated by retained completion")
        evaluations = [v for v in complete["evaluations"] if v["native_steps"] == step]
        if len(evaluations) != 1 or evaluations[0] != read(
            cell / f"evaluation_{step}.json"
        ):
            raise ValueError("checkpoint evaluation differs from parent completion")
        binding_paths = [
            cell / name
            for name in (
                f"checkpoint_{step}.pkl",
                "config.yaml",
                f"evaluation_{step}.json",
                "complete.json",
            )
        ]
        binding_paths.append(study / "manifest.json")
        bindings[arm] = {str(path): sha(path) for path in binding_paths}
    upstream = BASE / "dreamerv3-upstream-e3f0224"
    if clean_commit(upstream) != p["upstream_commit"]:
        raise ValueError("upstream identity mismatch")
    source_commit = clean_commit(SOURCE)
    bundle = None
    if bundle_path is not None:
        bundle_path = Path(bundle_path).resolve()
        heads = subprocess.check_output(
            ["git", "bundle", "list-heads", str(bundle_path)], text=True
        )
        if f"{source_commit} refs/heads/main" not in heads.splitlines():
            raise ValueError("deployment bundle does not bind main source commit")
        bundle = dict(path=str(bundle_path), sha256=sha(bundle_path), heads=heads)
    # A failed preflight consumes budget permanently. A new immutable study
    # registration may share that same ledger; bind its prefix, never reset it.
    with BudgetLedger(budget_path) as ledger:
        budget_snapshot = dict(
            charged=ledger.charged,
            actual=ledger.actual,
            preflight_charged=ledger.preflight_charged,
            pending=ledger.pending,
            records=ledger._records,
            chain_head=ledger._digest,
            limit=ledger.limit,
        )
    root.mkdir(parents=True, exist_ok=False)
    for name in ("markers", "submissions", "logs", "profiles"):
        (root / name).mkdir()
    _write_json(root / "protocol.json", p)
    _write_json(
        root / "manifest.json",
        dict(
            schema=p["schema"],
            source=str(SOURCE),
            source_commit=source_commit,
            protocol_sha256=sha(PROTOCOL),
            cells={k: str(v) for k, v in cells.items()},
            dependencies=bindings,
            source_dependencies={
                arm: dict(path=str(path), commit=p[f"{arm}_commit"])
                for arm, path in sources.items()
            },
            upstream=str(upstream),
            upstream_commit=p["upstream_commit"],
            budget_path=str(Path(budget_path).resolve()),
            budget_at_registration=budget_snapshot,
            deployment_bundle=bundle,
        ),
    )
    print("PARALLEL_TRAJECTORY_REGISTERED", sha(root / "manifest.json"), flush=True)


def authenticate(root):
    root = Path(root)
    m, p = read(root / "manifest.json"), read(root / "protocol.json")
    if (
        m["source_commit"] != clean_commit(SOURCE)
        or m["source"] != str(SOURCE)
        or m["protocol_sha256"] != sha(PROTOCOL)
        or p != read(PROTOCOL)
    ):
        raise ValueError("source or protocol identity differs")
    for files in m["dependencies"].values():
        for path, digest in files.items():
            if sha(path) != digest:
                raise ValueError(f"frozen dependency changed {path}")
    if (
        m["upstream_commit"] != p["upstream_commit"]
        or clean_commit(m["upstream"]) != p["upstream_commit"]
    ):
        raise ValueError("upstream checkout changed")
    if set(m["source_dependencies"]) != {"imf", "categorical"}:
        raise ValueError("missing dependency source identity")
    for arm, dependency in m["source_dependencies"].items():
        if (
            dependency["commit"] != p[f"{arm}_commit"]
            or clean_commit(dependency["path"]) != p[f"{arm}_commit"]
        ):
            raise ValueError("dependency source checkout changed")
    _authenticate_budget_prefix(m)
    if (
        m.get("deployment_bundle")
        and sha(m["deployment_bundle"]["path"]) != m["deployment_bundle"]["sha256"]
    ):
        raise ValueError("deployment bundle changed")
    return m, p


def _authenticate_budget_prefix(manifest):
    """Read-only check safe alongside concurrent fit workers and later appends."""
    snapshot = manifest["budget_at_registration"]
    count, previous = 0, "0" * 64
    with Path(manifest["budget_path"]).open() as stream:
        for line in stream:
            if count == snapshot["records"]:
                break
            if not line.endswith("\n"):
                raise ValueError("partial registered budget prefix")
            event = json.loads(line)
            digest = event.pop("hash")
            serialized = json.dumps(
                event, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            if (
                event["seq"] != count
                or event["previous"] != previous
                or hashlib.sha256(serialized.encode()).hexdigest() != digest
            ):
                raise ValueError("registered budget prefix changed")
            previous, count = digest, count + 1
    if count != snapshot["records"] or previous != snapshot["chain_head"]:
        raise ValueError("registered budget prefix missing or reset")


def _budget_prefix(path):
    count, last = 0, None
    with Path(path).open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("partial budget record at stage completion")
            last, count = json.loads(line), count + 1
    if not count:
        raise ValueError("empty shared budget ledger")
    return dict(records=count, chain_head=last["hash"])


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if root not in path.parents:
        raise ValueError("artifact escapes study root")
    return str(path.relative_to(root))


def marker(root, stage, directory, payload=None, *, additional_files=()):
    root, directory = Path(root), Path(directory)
    artifact_root = _inside(root, directory)
    paths = [f for f in sorted(directory.rglob("*")) if f.is_file()]
    paths.extend(map(Path, additional_files))
    files = {_inside(root, f): sha(f) for f in paths}
    if not files:
        raise ValueError("empty stage evidence")
    budget_path = read(root / "manifest.json")["budget_path"]
    value = dict(
        stage=stage,
        manifest_sha256=sha(root / "manifest.json"),
        files=files,
        payload=payload or {},
        artifact_root=artifact_root,
        additional_files=[_inside(root, f) for f in additional_files],
        budget_prefix=_budget_prefix(budget_path),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        array_job=os.environ.get("SLURM_ARRAY_JOB_ID"),
        array_index=os.environ.get("SLURM_ARRAY_TASK_ID"),
    )
    _write_json(root / "markers" / f"{stage}.json", value)
    print("PARALLEL_TRAJECTORY_STAGE_VERIFIED", stage, flush=True)


def require(root, stage):
    root = Path(root)
    value = read(root / "markers" / f"{stage}.json")
    if value["stage"] != stage or value["manifest_sha256"] != sha(
        root / "manifest.json"
    ):
        raise ValueError("stage marker identity mismatch")
    _authenticate_budget_prefix(
        dict(
            read(root / "manifest.json"), budget_at_registration=value["budget_prefix"]
        )
    )
    artifact_root = root / value["artifact_root"]
    _inside(root, artifact_root)
    observed = {_inside(root, f) for f in artifact_root.rglob("*") if f.is_file()}
    observed.update(value["additional_files"])
    if not value["files"] or observed != set(value["files"]):
        raise ValueError("stage file inventory differs")
    for name, digest in value["files"].items():
        path = root / name
        if root.resolve() not in path.resolve().parents or sha(path) != digest:
            raise ValueError("stage artifact differs")
    submission_stage = (
        "fit"
        if stage.startswith("fit-")
        else "evaluate" if stage in ("predictions", "profile") else stage
    )
    receipt = root / "submissions" / f"{submission_stage}.json"
    if receipt.exists():
        submitted = read(receipt)
        if submitted["stage"] != submission_stage:
            raise ValueError("stage receipt identity differs")
        if stage.startswith("fit-"):
            if (
                str(value["array_job"]) != str(submitted["job"])
                or str(value["array_index"]) != stage.split("-")[1]
            ):
                raise ValueError("fit marker belongs to another Slurm task")
        elif str(value["slurm_job"]) != str(submitted["job"]):
            raise ValueError("stage marker belongs to another Slurm job")
    return value


def setup(root, arm="imf"):
    m, p = authenticate(root)
    if str(m["upstream"]) not in sys.path:
        sys.path.insert(0, str(m["upstream"]))
    import jax
    from embodied.jax import internal

    internal.setup(
        platform="cuda",
        compute_dtype="bfloat16",
        transfer_guard=False,
        compilation_cache=False,
    )
    if not all(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required")
    from .parallel_frozen import FrozenModel

    return FrozenModel(m["cells"][arm], arm), m, p


def dataset(root):
    from .parallel_runner import load_npz

    done = require(root, "collect")
    path = Path(done["payload"]["dataset"])
    if _inside(root, path) not in done["files"]:
        raise ValueError("dataset not authenticated by collection marker")
    return load_npz(path), path.parent


def preflight(root):
    import jax
    from .parallel_frozen import ReacherAdapter, FrozenModel
    from .parallel_runner import load_npz, fit, load_predictor, predictions, finite

    teacher, m, p = setup(root)
    directory = Path(root) / "preflight"
    directory.mkdir(exist_ok=False)
    begun = time.monotonic()
    with BudgetLedger(m["budget_path"]) as budget:
        collection = collect(
            directory / "collection",
            teacher,
            ReacherAdapter,
            budget=budget,
            preflight=True,
        )
        budget_snapshot = dict(charged=budget.charged, actual=budget.actual)
    data = load_npz(collection.dataset)
    selection = fit(directory / "fit", data, teacher, p, 0, preflight=True)
    predictor = load_predictor(directory / "fit", teacher)
    indices = np.arange(8)
    o, r = predictions(predictor, data, indices, 821)
    repeated = predictions(predictor, data, indices, 821)
    if not np.array_equal(o, repeated[0]) or not np.array_equal(r, repeated[1]):
        raise ValueError("independent forward replay differs within preflight")
    # Exercise the largest registered inference shape before spending the main
    # simulator budget. These are predictions only, not extra environment steps.
    rows = np.arange(64) % len(data["start"])
    for nfe in (1, 2, 4):
        features = predictor.particles(
            data["start"][rows], data["actions"][rows], jax.random.PRNGKey(822), 32, nfe
        )
        finite(teacher.decode(features))
    from .parallel_collection import reconstruct_features

    for arm in ("imf", "categorical"):
        model = teacher if arm == "imf" else FrozenModel(m["cells"][arm], arm)
        start, target = reconstruct_features(
            model,
            collection.attempt / "episode-000.npz",
            anchor=100,
            branch_path=collection.attempt / "branch-000-100-0.npz",
        )
        if arm == "imf" and (
            not np.array_equal(start, data["start"][0])
            or not np.array_equal(target, data["targets"][0])
        ):
            raise ValueError("posterior target replay mismatch")
        for nfe in ((1, 4) if arm == "imf" else (1,)):
            features = model.rollout(
                np.repeat(start[None], 64, axis=0), data["actions"][rows], 123, 32, nfe
            )
            finite(model.decode(features))
        if model.frozen_digest() != model.before:
            raise ValueError("frozen teacher changed")
    _write_npz(
        directory / "predictions.npz", dict(observations=o, rewards=r, indices=indices)
    )
    _write_json(
        directory / "result.json",
        dict(
            seconds=time.monotonic() - begun,
            budget=budget_snapshot,
            selection=selection,
            frozen_digest=teacher.before,
            gpu=str(jax.devices()[0].device_kind),
            full_size=True,
            parent_pid=os.getpid(),
            dataset=str(collection.dataset),
            runtime=dict(
                python=sys.version,
                executable=sys.executable,
                packages={
                    name: importlib.metadata.version(name)
                    for name in (
                        "jax",
                        "jaxlib",
                        "numpy",
                        "optax",
                        "ninjax",
                        "dm-control",
                        "mujoco",
                    )
                },
            ),
        ),
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "dreamer_imf_compare.parallel_study",
            "preflight-replay",
            "--root",
            str(root),
        ],
        check=True,
    )
    marker(root, "preflight", directory)
    print("PARALLEL_TRAJECTORY_PREFLIGHT_VERIFIED", flush=True)


def preflight_replay(root):
    """Independent process, retained inputs, and no simulator interactions."""
    from .parallel_runner import load_npz, load_predictor, predictions
    from .parallel_collection import reconstruct_features

    teacher, _, _ = setup(root)
    directory = Path(root) / "preflight"
    result = read(directory / "result.json")
    if result["parent_pid"] == os.getpid():
        raise ValueError("preflight replay must run in a distinct process")
    data = load_npz(result["dataset"])
    retained = load_npz(directory / "predictions.npz")
    _same_array(retained["indices"], np.arange(8), "complete preflight replay indices")
    predictor = load_predictor(directory / "fit", teacher)
    obs, reward = predictions(predictor, data, retained["indices"], 821)
    _same_array(obs, retained["observations"], "independent preflight observations")
    _same_array(reward, retained["rewards"], "independent preflight rewards")
    location = Path(result["dataset"]).parent
    start, target = reconstruct_features(
        teacher,
        location / "episode-000.npz",
        anchor=100,
        branch_path=location / "branch-000-100-0.npz",
    )
    _same_array(start, data["start"][0], "independent preflight belief")
    _same_array(target, data["targets"][0], "independent preflight posterior")
    _write_json(
        directory / "replay.json",
        dict(
            independent_process=True,
            pid=os.getpid(),
            source_commit=clean_commit(SOURCE),
            predictions_sha256=sha(directory / "predictions.npz"),
            dataset_sha256=sha(result["dataset"]),
            exact=True,
        ),
    )
    print("PARALLEL_PREFLIGHT_INDEPENDENT_REPLAY_VERIFIED", flush=True)


def collection_stage(root):
    require(root, "preflight")
    from .parallel_frozen import ReacherAdapter

    teacher, m, _ = setup(root)
    directory = Path(root) / "collection"
    directory.mkdir(exist_ok=False)
    with BudgetLedger(m["budget_path"]) as budget:
        value = collect(directory, teacher, ReacherAdapter, budget=budget)
    marker(root, "collect", directory, dict(dataset=str(value.dataset)))


def fit_stage(root, seed):
    if seed not in (0, 1, 2):
        raise ValueError("head seed outside registered cells")
    from .parallel_runner import fit

    teacher, _, p = setup(root)
    data, _ = dataset(root)
    directory = Path(root) / "fit" / str(seed)
    result = fit(directory, data, teacher, p, seed)
    marker(root, f"fit-{seed}", directory, result)


def evaluate_stage(root):
    from .parallel_frozen import FrozenModel
    from .parallel_runner import evaluate_head, evaluate_baseline, normalized_data

    for seed in range(3):
        require(root, f"fit-{seed}")
    teacher, m, _ = setup(root)
    data, location = dataset(root)
    directory = Path(root) / "evaluation"
    directory.mkdir(exist_ok=False)
    norm = normalized_data(data)
    evaluate_baseline(directory / "imf", data, teacher, norm)
    cat = FrozenModel(m["cells"]["categorical"], "categorical")
    evaluate_baseline(
        directory / "categorical",
        data,
        cat,
        norm,
        categorical=True,
        dataset_dir=location,
    )
    for seed in range(3):
        evaluate_head(
            directory / f"head-{seed}",
            data,
            teacher,
            Path(root) / "fit" / str(seed),
            seed,
        )
    marker(root, "predictions", directory)
    # Dedicated processes on this same allocation/device isolate allocator peaks.
    for mode, seed, nfe in [("imf", 0, 1), ("imf", 0, 4), ("categorical", 0, 1)] + [
        ("head", s, n) for s in range(3) for n in (1, 2, 4)
    ]:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "dreamer_imf_compare.parallel_study",
                "profile",
                "--root",
                str(root),
                "--mode",
                mode,
                "--seed",
                str(seed),
                "--nfe",
                str(nfe),
            ],
            check=True,
        )
    for model in (teacher, cat):
        if model.frozen_digest() != model.before:
            raise ValueError("teacher changed during evaluation")
    marker(root, "evaluate", directory)
    marker(root, "profile", Path(root) / "profiles")


def profile_stage(root, mode, seed, nfe):
    from .parallel_runner import profile, load_npz

    if mode not in ("head", "imf", "categorical") or seed not in (0, 1, 2):
        raise ValueError("unregistered profiling mode or seed")
    if (
        mode == "categorical"
        and (seed != 0 or nfe != 1)
        or mode == "imf"
        and (seed != 0 or nfe not in (1, 4))
        or mode == "head"
        and nfe not in (1, 2, 4)
    ):
        raise ValueError("unregistered profiling cell")
    require(root, "predictions")
    if mode == "head":
        require(root, f"fit-{seed}")
    teacher, _, _ = setup(root, "categorical" if mode == "categorical" else "imf")
    data, _ = dataset(root)
    if mode == "categorical":
        beliefs = load_npz(Path(root) / "evaluation/categorical/beliefs.npz")
        if not np.array_equal(beliefs["indices"], np.flatnonzero(data["split"] == 2)):
            raise ValueError("categorical profile belief indices differ")
        data["start"][beliefs["indices"]] = beliefs["start"]
    result = profile(teacher, data, mode, seed, nfe, Path(root) / "fit" / str(seed))
    _write_json(Path(root) / "profiles" / f"{mode}-{seed}-{nfe}.json", result)


def _same_array(actual, expected, label):
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"independent array replay differs: {label}")


def reconstruct_beliefs(model, location, data, indices):
    """Independently filter retained raw histories; never trust stored beliefs.

    Each episode prefix is filtered once. Branch carries are copied at their
    anchor and action/observation/reward alignment is checked before inference.
    This consumes no environment interactions or simulation budget.
    """
    from .parallel_collection import load_raw_history

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("belief replay requires unique nonempty row indices")
    horizon = data["actions"].shape[1]
    starts = np.empty((len(indices), model.feature_dim), np.float32)
    targets = np.empty((len(indices), horizon, model.feature_dim), np.float32)
    positions = {int(row): i for i, row in enumerate(indices)}
    for eid in np.unique(data["episode"][indices]):
        rows = indices[data["episode"][indices] == eid]
        base, raw = load_raw_history(Path(location) / f"episode-{eid:03d}.npz")
        carry, feature = model.observe(
            model.initial(),
            raw[0],
            base["previous_action"],
            int(base["observe_seeds"][0]),
        )
        last_anchor = int(data["anchor"][rows].max())
        if last_anchor >= len(raw):
            raise ValueError("raw episode does not reach requested anchor")
        for t in range(1, last_anchor + 1):
            carry, feature = model.observe(
                carry, raw[t], base["actions"][t - 1], int(base["observe_seeds"][t])
            )
            for row in rows[data["anchor"][rows] == t]:
                plan, at = int(data["plan"][row]), positions[int(row)]
                branch, observations = load_raw_history(
                    Path(location) / f"branch-{eid:03d}-{t:03d}-{plan}.npz"
                )
                if len(observations) != horizon + 1 or set(raw[t]) != set(
                    observations[0]
                ):
                    raise ValueError("raw branch schema or horizon differs")
                for name in raw[t]:
                    _same_array(
                        observations[0][name], raw[t][name], "branch anchor " + name
                    )
                _same_array(
                    branch["previous_action"],
                    base["actions"][t - 1],
                    "branch previous action",
                )
                _same_array(branch["actions"], data["actions"][row], "branch actions")
                _same_array(branch["rewards"], data["rewards"][row], "branch rewards")
                flat = np.stack(
                    [
                        np.concatenate(
                            [np.asarray(o[k]).reshape(-1) for k in model.obs_keys]
                        )
                        for o in observations
                    ]
                ).astype(np.float32)
                _same_array(
                    flat[0], data["initial_observation"][row], "initial observation"
                )
                _same_array(flat[1:], data["observations"][row], "future observations")
                starts[at], state = feature, copy.deepcopy(carry)
                for j in range(horizon):
                    state, future = model.observe(
                        state,
                        observations[j + 1],
                        branch["actions"][j],
                        int(branch["observe_seeds"][j + 1]),
                    )
                    targets[at, j] = future
    return starts, targets


def verify_collection_artifacts(location, data, protocol):
    from .parallel_collection import duplicate_design, episode_design, DECISIONS
    from .parallel_runner import load_npz

    expected = {
        (entry["episode"], anchor, plan): entry
        for entry in episode_design()
        for anchor in protocol["anchors"]
        for plan in range(4)
    }
    actual = list(
        zip(map(int, data["episode"]), map(int, data["anchor"]), map(int, data["plan"]))
    )
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("incomplete or duplicate episode/anchor/plan grid")
    for row, key in enumerate(actual):
        if (
            data["split"][row] != expected[key]["split"]
            or data["mode"][row] != expected[key]["mode"]
        ):
            raise ValueError("episode split or collection mode differs")
    manifest = read(Path(location) / "manifest.json")
    count = (
        protocol["episodes"] * DECISIONS * protocol["action_repeat"]
        + len(expected) * protocol["horizon"] * protocol["action_repeat"]
        + protocol["duplicate_branches"]
        * protocol["horizon"]
        * protocol["action_repeat"]
    )
    if (
        manifest["actual_steps"] != count
        or manifest["charged_steps"] != count
        or manifest["episode_counts"] != protocol["episode_split"]
        or manifest["duplicate_rows"] != protocol["duplicate_branches"]
    ):
        raise ValueError("collection counts differ from registered design")
    duplicates = load_npz(Path(location) / "duplicates.npz")
    duplicate_keys = list(
        zip(
            map(int, duplicates["episode"]),
            map(int, duplicates["anchor"]),
            map(int, duplicates["plan"]),
        )
    )
    if len(duplicate_keys) != protocol["duplicate_branches"] or set(
        duplicate_keys
    ) != set(duplicate_design()):
        raise ValueError("duplicate restoration design differs")
    lookup = {key: row for row, key in enumerate(actual)}
    for row, key in enumerate(duplicate_keys):
        for name in data:
            _same_array(
                duplicates[name][row], data[name][lookup[key]], "duplicate " + name
            )
        episode, anchor, plan = key
        stem = f"branch-{episode:03d}-{anchor:03d}-{plan}"
        original = load_npz(Path(location) / f"{stem}.npz")
        replay = load_npz(Path(location) / f"{stem}-duplicate.npz")
        if set(original) != set(replay):
            raise ValueError("duplicate raw history schema differs")
        for name in original:
            _same_array(replay[name], original[name], "duplicate raw " + name)


def validate_profile_grid(profiles, evaluation_marker, protocol):
    """Authenticate cell identities, same physical allocation, and raw timings."""
    cells = [("imf", 0, 1), ("imf", 0, 4), ("categorical", 0, 1)]
    cells += [
        ("head", s, n) for s in protocol["head_seeds"] for n in protocol["flow_steps"]
    ]
    if set(profiles) != {f"{m}-{s}-{n}" for m, s, n in cells}:
        raise ValueError("incomplete or unexpected profiling grid")
    common = None
    for mode, seed, nfe in cells:
        profile = profiles[f"{mode}-{seed}-{nfe}"]
        if (
            profile["mode"],
            profile["seed"],
            profile["flow_steps"],
            profile["particles"],
            profile["horizon"],
        ) != (mode, seed, nfe, protocol["particles"], protocol["horizon"]):
            raise ValueError("profile metadata differs from filename or protocol")
        identity = profile["device_identity"]
        if (
            not identity.get("node")
            or not identity.get("visible_devices")
            or not identity.get("slurm_job")
            or not identity.get("ids")
            or identity["slurm_job"] != evaluation_marker["slurm_job"]
        ):
            raise ValueError("profile lacks physical GPU/allocation identity")
        signature = (profile["device"], profile["platform_version"], identity)
        if common is None:
            common = signature
        if signature != common or "A30" not in profile["device"]:
            raise ValueError(
                "profiles did not use the same registered physical A30 GPU"
            )
        if profile["precision"] != {
            "student_flow": "float32",
            "upstream_actual": "bfloat16",
        }:
            raise ValueError("profile precision differs from deployed protocol")
        if set(profile["batches"]) != {"batch1", "batch64"}:
            raise ValueError("profile batch grid differs")
        for batch in profile["batches"].values():
            for kind in ("latent", "decoded"):
                value = batch[kind]
                samples = np.asarray(value["samples"], np.float64)
                if (
                    samples.shape != (20,)
                    or not np.isfinite(samples).all()
                    or np.any(samples <= 0)
                ):
                    raise ValueError("invalid synchronized timing samples")
                if value["median_seconds"] != float(np.median(samples)) or value[
                    "p95_seconds"
                ] != float(np.percentile(samples, 95)):
                    raise ValueError("profile timing summary differs from samples")


def verify_stage(root):
    """Separate process: regenerate predictions and recompute all reported scores."""
    import jax
    from .parallel_runner import (
        load_npz,
        load_predictor,
        predictions,
        normalized_data,
        summary,
        prefix_rows,
        Predictor,
    )
    from .parallel_metrics import compare_distributions, promotion, split_counts
    from .parallel_frozen import FrozenModel

    evaluation_marker = require(root, "evaluate")
    require(root, "profile")
    require(root, "predictions")
    for seed in range(3):
        require(root, f"fit-{seed}")
    teacher, m, p = setup(root)
    data, location = dataset(root)
    verify_collection_artifacts(location, data, p)
    # Refilter *all* iMF training/validation/test targets from retained raw
    # observations, not merely a saved test-belief cache.
    reconstructed_start, reconstructed_targets = reconstruct_beliefs(
        teacher, location, data, np.arange(len(data["split"]))
    )
    _same_array(reconstructed_start, data["start"], "all iMF start beliefs")
    _same_array(reconstructed_targets, data["targets"], "all iMF posterior targets")
    indices = np.flatnonzero(data["split"] == 2)
    norm = normalized_data(data)

    def equal(a, b, label):
        if a != b:
            raise ValueError(f"independent metric recomputation differs: {label}")

    heads, all_head_variants, checks, latencies = {}, {}, {}, {}
    for seed in range(3):
        train = Path(root) / "fit" / str(seed)
        select = read(train / "selection.json")
        stored_norm = load_npz(train / "normalization.npz")
        if set(stored_norm) != set(norm):
            raise ValueError("normalization schema differs")
        for name in norm:
            _same_array(
                stored_norm[name], norm[name], "train-only normalization " + name
            )
        expected_updates = list(
            range(p["validation_period"], p["updates"] + 1, p["validation_period"])
        )
        if (
            select["seed"] != seed
            or select.get("preflight") is not False
            or [row["update"] for row in select["validations"]] != expected_updates
        ):
            raise ValueError("validation schedule, head seed, or run kind differs")
        independent = []
        for row in select["validations"]:
            val = load_npz(train / f"validation_{row['update']:05d}.npz")
            vi = val["indices"]
            _same_array(
                vi, np.flatnonzero(data["split"] == 1), "complete validation indices"
            )
            if row["checkpoint"] != f"checkpoint_{row['update']:05d}.pkl":
                raise ValueError("validation checkpoint identity differs")
            with (train / row["checkpoint"]).open("rb") as stream:
                checkpoint_params = jax.tree.map(jax.device_put, pickle.load(stream))
            candidate = Predictor(teacher, checkpoint_params, stored_norm)
            candidate_replay = predictions(
                candidate, data, vi, 7139, count=p["particles"]
            )
            _same_array(
                candidate_replay[0],
                val["observations"],
                "checkpoint validation observations",
            )
            _same_array(
                candidate_replay[1], val["rewards"], "checkpoint validation rewards"
            )
            om = float(
                np.mean(
                    (
                        (val["observations"].mean(1) - data["observations"][vi])
                        / norm["obs_scale"]
                    )
                    ** 2
                )
            )
            rm = float(
                np.mean(
                    (
                        (val["rewards"].mean(1).sum(1) - data["rewards"][vi].sum(1))
                        / norm["return_scale"]
                    )
                    ** 2
                )
            )
            if (
                row["score"] != om + rm
                or row["observation_mse"] != om
                or row["cumulative_reward_mse"] != rm
            ):
                raise ValueError("selection score mismatch")
            independent.append((om + rm, row["update"], row["checkpoint"]))
        if (
            select["selected"] != min(independent)[2]
            or select["score"] != min(independent)[0]
            or select["updates"] != p["updates"]
        ):
            raise ValueError("checkpoint selection/count mismatch")
        directory = Path(root) / "evaluation" / f"head-{seed}"
        report = read(directory / "report.json")
        predictor = load_predictor(train, teacher)
        if report["seed"] != seed or set(report["variants"]) != {"1", "2", "4"}:
            raise ValueError("head report cells differ")
        for nfe in (1, 2, 4):
            raw = load_npz(directory / f"nfe{nfe}.npz")
            _same_array(raw["indices"], indices, "test prediction indices")
            replay = predictions(
                predictor, data, indices, 19900 + seed * 100, steps=nfe
            )
            if not np.array_equal(raw["observations"], replay[0]) or not np.array_equal(
                raw["rewards"], replay[1]
            ):
                raise ValueError(
                    f"strict head prediction replay differs seed{seed} NFE{nfe}"
                )
            equal(
                summary(data, indices, *replay, norm),
                report["variants"][str(nfe)],
                "head",
            )
        direct = load_npz(directory / "nfe1.npz")
        composed = load_npz(directory / "composed.npz")
        _same_array(composed["indices"], indices, "composed prediction indices")
        replay = predictions(
            predictor, data, indices, 81900 + seed * 100, composed=True
        )
        if not np.array_equal(
            composed["observations"], replay[0]
        ) or not np.array_equal(composed["rewards"], replay[1]):
            raise ValueError("composed replay differs")
        l, r = prefix_rows(data, indices)
        independent_checks = [
            compare_distributions(
                direct["observations"],
                direct["rewards"],
                *replay,
                data["episode"][indices],
                norm["obs_scale"],
                float(norm["return_scale"]),
            ),
            compare_distributions(
                direct["observations"][l],
                direct["rewards"][l],
                direct["observations"][r],
                direct["rewards"][r],
                data["episode"][indices[l]],
                norm["obs_scale"],
                float(norm["return_scale"]),
                prefix_horizon=5,
            ),
        ]
        equal(independent_checks, report["distribution_checks"], "distribution")
        heads[str(seed)] = report["variants"]["1"]
        all_head_variants[str(seed)] = report["variants"]
        checks[str(seed)] = independent_checks
        latency = read(Path(root) / "profiles" / f"head-{seed}-1.json")
        latencies[str(seed)] = {
            key: v["decoded"]["median_seconds"] for key, v in latency["batches"].items()
        }
    baselines = {}
    for arm in ("imf", "categorical"):
        model = teacher if arm == "imf" else FrozenModel(m["cells"][arm], arm)
        directory = Path(root) / "evaluation" / arm
        old = read(directory / "report.json")
        beliefs = load_npz(directory / "beliefs.npz")
        _same_array(beliefs["indices"], indices, "baseline belief indices")
        if arm == "imf":
            starts, targets = (
                reconstructed_start[indices],
                reconstructed_targets[indices],
            )
        else:
            starts, targets = reconstruct_beliefs(model, location, data, indices)
        _same_array(beliefs["start"], starts, "independent " + arm + " start beliefs")
        _same_array(
            beliefs["targets"], targets, "independent " + arm + " posterior targets"
        )
        for nfe in ((1, 4) if arm == "imf" else (1,)):
            raw = load_npz(directory / f"nfe{nfe}.npz")
            _same_array(raw["indices"], indices, "baseline prediction indices")
            obs, reward = [], []
            for i in range(0, len(indices), 8):
                f = model.rollout(
                    starts[i : i + 8],
                    data["actions"][indices[i : i + 8]],
                    67100 + i,
                    32,
                    nfe,
                )
                o, r = model.decode(f)
                obs.append(np.asarray(jax.device_get(o)))
                reward.append(np.asarray(jax.device_get(r)))
            o, r = np.concatenate(obs), np.concatenate(reward)
            if not np.array_equal(o, raw["observations"]) or not np.array_equal(
                r, raw["rewards"]
            ):
                raise ValueError("baseline strict replay differs")
            equal(summary(data, indices, o, r, norm), old[str(nfe)], "baseline")
        posterior = load_npz(directory / "posterior_heads.npz")
        _same_array(posterior["indices"], indices, "posterior head indices")
        po, pr = model.decode(targets)
        po, pr = (
            np.asarray(jax.device_get(po))[:, None],
            np.asarray(jax.device_get(pr))[:, None],
        )
        _same_array(po, posterior["observations"], "posterior decoder observations")
        _same_array(pr, posterior["rewards"], "posterior reward head")
        equal(
            summary(data, indices, po, pr, norm),
            old["posterior_heads"],
            "posterior heads",
        )
        if arm == "imf":
            persistence = np.repeat(
                data["initial_observation"][indices, None, None], p["horizon"], axis=2
            )
            zero_reward = np.zeros((len(indices), 1, p["horizon"]), np.float32)
            equal(
                summary(data, indices, persistence, zero_reward, norm),
                old["persistence_zero"],
                "persistence/zero-reward controls",
            )
        if model.frozen_digest() != model.before:
            raise ValueError("teacher changed during independent verification")
        baselines[arm] = old
    profiles = {
        f.stem: read(f) for f in sorted((Path(root) / "profiles").glob("*.json"))
    }
    validate_profile_grid(profiles, evaluation_marker, p)
    bprofile = profiles["imf-0-4"]
    bl = {k: v["decoded"]["median_seconds"] for k, v in bprofile["batches"].items()}
    conclusion = promotion(heads, baselines["imf"]["4"], latencies, bl, checks)
    with BudgetLedger(m["budget_path"]) as ledger:
        if not 0 <= ledger.actual <= ledger.charged <= p["native_budget"]:
            raise ValueError("over-limit or inconsistent simulation budget")
        budget = dict(
            actual_known=ledger.actual,
            charged_upper_bound=ledger.charged,
            actual=ledger.actual,
            charged=ledger.charged,
            limit=ledger.limit,
            pending_reservations=ledger.pending,
            preflight_charged=ledger.preflight_charged,
            actual_is_lower_bound=bool(ledger.pending),
            accounting="conservative_unresolved" if ledger.pending else "settled",
            registration_snapshot=m["budget_at_registration"],
        )
    report = dict(
        schema=p["schema"],
        manifest_sha256=sha(Path(root) / "manifest.json"),
        head_results=heads,
        baselines=baselines,
        head_variants=all_head_variants,
        distribution_checks=checks,
        profiles=profiles,
        promotion=conclusion,
        budget=budget,
        independent_replay=dict(
            imf_posterior_rows=len(data["split"]),
            categorical_posterior_rows=len(indices),
            validation_checkpoints_per_seed=len(expected_updates),
            validation_predictions_replayed=True,
            posterior_heads_replayed=True,
            persistence_zero_controls_recomputed=True,
        ),
        splits=split_counts(data["episode"], data["split"], data["rewards"]),
        statistical_unit="12 held-out environment episodes nested under one frozen world-model seed; three head initialization seeds",
        limitations=p["limitations"],
    )
    _write_json(Path(root) / "report.json", report)
    marker(
        root,
        "verify",
        Path(root) / "evaluation",
        dict(report_sha256=sha(Path(root) / "report.json")),
        additional_files=(Path(root) / "report.json",),
    )
    print(
        "PARALLEL_TRAJECTORY_FINAL_VERIFIED",
        sha(Path(root) / "report.json"),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=(*STAGES, "register", "profile", "preflight-replay")
    )
    p.add_argument("--root", required=True)
    p.add_argument("--budget-path")
    p.add_argument("--bundle-path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("head", "imf", "categorical"))
    p.add_argument("--nfe", type=int, choices=(1, 2, 4), default=1)
    a = p.parse_args()
    if a.stage == "register":
        if not a.budget_path or not a.bundle_path:
            p.error("--budget-path and --bundle-path required")
        register(a.root, a.budget_path, a.bundle_path)
    elif a.stage == "preflight-replay":
        preflight_replay(a.root)
    elif a.stage == "profile":
        profile_stage(a.root, a.mode, a.seed, a.nfe)
    else:
        {
            "preflight": preflight,
            "collect": collection_stage,
            "fit": lambda r: fit_stage(r, a.seed),
            "evaluate": evaluate_stage,
            "verify": verify_stage,
        }[a.stage](Path(a.root))


if __name__ == "__main__":
    main()
