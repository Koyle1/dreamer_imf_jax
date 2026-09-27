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
LEGACY_RESULT_COMMITS = frozenset({"b7e22a5ae4e7d8473624908283f683e0b9b1280a"})
REPLAY_MODES = ("primer", "create", "replay")
TRAINING_FILES = frozenset(
    {"result.json", "checkpoint.pkl", "replay.npz", "schedules.pkl"}
)
EVALUATION_FILES = frozenset(
    {"result.json", "trace.npz"}
    | {f"{mode}-{kind}.json" for mode in REPLAY_MODES for kind in ("receipt", "seal")}
)


def exact_fields(value, names, label):
    """A marker cannot choose a weaker schema by deleting or adding fields."""
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{label} field set differs")


def require_sha256(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"invalid {label} digest")


def require_number(value, label, *, minimum=None):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(f"invalid {label}")
    if minimum is not None and value < minimum:
        raise ValueError(f"invalid {label}")


def validate_runtime(value):
    exact_fields(
        value,
        ("python", "jax", "jaxlib", "numpy", "mujoco", "dm_control", "devices", "x64"),
        "runtime",
    )
    if any(
        not isinstance(value[k], str) or not value[k]
        for k in value
        if k not in ("devices", "x64")
    ):
        raise ValueError("invalid runtime versions")
    if (
        not isinstance(value["devices"], list)
        or not value["devices"]
        or any(not isinstance(d, str) or not d for d in value["devices"])
        or value["x64"] is not False
    ):
        raise ValueError("invalid device/precision runtime")


def validate_controller_protocol(controller):
    """Bind the declared knobs to the constants used by the legacy runners.

    These runners do not take arbitrary controller settings. Unsupported changes
    must fail before loading a checkpoint or stepping an environment.
    """
    from . import actor_gap_roadmap_study as roadmap

    realized = dict(
        horizon=roadmap.CONTROLLER_HORIZON,
        flowmpc_step_size=roadmap.CONTROLLER_STEP_SIZE,
        flowmpc_particles=roadmap.FLOWMPC_PARTICLES,
        action_sequence_particles=roadmap.ACTION_SEQUENCE_PARTICLES,
        action_sequence_objective_evaluations=roadmap.ACTION_SEQUENCE_OBJECTIVE_EVALUATIONS,
        action_sequence_residual_limit=roadmap.ACTION_SEQUENCE_RESIDUAL_LIMIT,
        heldout_minimum_improvement=roadmap.HELDOUT_MINIMUM_IMPROVEMENT,
        trust_anchor_mse_sum_budget=roadmap.TRUST_ANCHOR_MSE_BUDGET,
        trust_current_linf_budget=roadmap.TRUST_CURRENT_LINF_BUDGET,
    )
    exact_fields(controller, realized, "controller protocol")
    for name, expected in realized.items():
        require_number(controller[name], f"controller {name}")
        if controller[name] != expected or (
            type(expected) is int and type(controller[name]) is not int
        ):
            raise ValueError(f"unsupported controller protocol value: {name}")
    # The endpoint runner constructs this fixed two-by-four CEM budget.
    if realized["action_sequence_objective_evaluations"] != 2 + 2 * 4:
        raise ValueError("controller objective budget differs from realized CEM")
    return realized


def read(path):
    def pairs(items):
        value = {}
        for name, item in items:
            if name in value:
                raise ValueError(f"duplicate JSON field: {name}")
            value[name] = item
        return value

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid_constant
    )


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
    validate_training_result(result, preflight=preflight)
    with (directory / "checkpoint.pkl").open("rb") as handle:
        model = pickle.load(handle)
    arrays = b.load_npz(directory / "replay.npz")
    with (directory / "schedules.pkl").open("rb") as handle:
        schedules = pickle.load(handle)
    if (
        not finite(model)
        or not finite(arrays)
        or not finite(result)
        or not finite(schedules)
    ):
        raise ValueError("nonfinite training artifact")
    cfg = config_for(protocol, arrays)
    if digest(model["config"]) != digest(asdict(cfg)) or digest(
        result["config"]
    ) != digest(asdict(cfg)):
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
    schedule_names = {"world", "reward", "reward-test"}
    exact_fields(schedules, schedule_names, "training schedules")
    exact_fields(result["schedule_sha256"], schedule_names, "training schedule digests")
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
    if set(result["policies"]) != set(map(str, actors)):
        raise ValueError("policy result seed set differs")
    for actor in actors:
        info, policy = result["policies"][str(actor)], model["policies"][actor]
        if (
            int(policy.step) != expected["rebrac_updates"]
            or b._tree_digest(policy) != info["final_state_sha256"]
        ):
            raise ValueError("policy checkpoint update/digest differs")
    from .actor_gap_model_training import ENDPOINT_FAMILIES

    exact_fields(model["endpoints"], ENDPOINT_FAMILIES, "endpoint checkpoints")
    exact_fields(result["endpoints"], ENDPOINT_FAMILIES, "endpoint results")
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

    realized_controller = validate_controller_protocol(protocol["controller"])
    if cell["arm"] not in read(PROTOCOL)["arms"]:
        raise ValueError("unsupported controller arm")
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
        realized_controller_config=realized_controller,
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


def verify_bound_files(m, directory, expected_files, *, replay=False):
    """Authenticate an exact artifact set; reusable by separately frozen studies.

    The caller supplies its already authenticated manifest and stage schema.
    No source identity override or legacy-manifest fallback occurs here.
    """
    directory = Path(directory)
    if directory.is_symlink() or (directory / "verified.json").is_symlink():
        raise ValueError("symlinked artifact directory/marker")
    mark = read(directory / "verified.json")
    fields = {"source_commit", "manifest_sha256", "cell", "files"}
    if replay:
        fields |= {"strict_bitwise_replay", "cache_sha256"}
    exact_fields(mark, fields, "cell marker")
    exact_fields(mark["files"], expected_files, "bound artifact")
    if (
        mark["manifest_sha256"] != digest(m)
        or mark["source_commit"] != m["source_commit"]
    ):
        raise ValueError("marker identity mismatch")
    if replay:
        if mark["strict_bitwise_replay"] is not True:
            raise ValueError("strict bitwise replay is mandatory")
        require_sha256(mark["cache_sha256"], "cache")
    for name, sha in mark["files"].items():
        require_sha256(sha, "artifact")
        path = directory / name
        if (
            Path(name).name != name
            or name in (".", "..")
            or path.is_symlink()
            or not path.is_file()
            or b.file_sha256(path) != sha
        ):
            raise ValueError("bound artifact differs")
    return mark


def verify_replay_receipts(directory, mark, result, *, expected_runtime=None):
    """Bind retained trace, reader cores, distinct processes and all cache seals."""
    directory = Path(directory)
    if mark.get("strict_bitwise_replay") is not True:
        raise ValueError("strict bitwise replay is mandatory")
    require_sha256(mark["cache_sha256"], "cache")
    trace = b.load_npz(directory / "trace.npz")
    if (
        not trace
        or not finite(trace)
        or b.array_sha256(trace) != result["trace_sha256"]
    ):
        raise ValueError("retained trace binding differs or is nonfinite")
    receipts = []
    for mode in REPLAY_MODES:
        receipt = read(directory / f"{mode}-receipt.json")
        exact_fields(
            receipt,
            (
                "mode",
                "pid",
                "runtime",
                "core_sha256",
                "trace_sha256",
                "cache_sha256",
                "wall_seconds",
                "timing",
            ),
            "replay receipt",
        )
        if (
            receipt["mode"] != mode
            or type(receipt["pid"]) is not int
            or receipt["pid"] <= 0
        ):
            raise ValueError("reader mode/process identity differs")
        validate_runtime(receipt["runtime"])
        if expected_runtime is not None and receipt["runtime"] != expected_runtime:
            raise ValueError("reader runtime differs from authenticated preflight")
        require_number(receipt["wall_seconds"], "reader wall time", minimum=0)
        validate_timing(receipt["timing"])
        for key in ("core_sha256", "trace_sha256", "cache_sha256"):
            require_sha256(receipt[key], key)
        # Primer executes in the compiler-writer process, so its trajectory is
        # deliberately not compared to the two independent cache readers.
        if mode != "primer" and (
            receipt["core_sha256"] != digest(result)
            or receipt["trace_sha256"] != result["trace_sha256"]
        ):
            raise ValueError("reader result/trace binding differs")
        seal = read(directory / f"{mode}-seal.json")
        exact_fields(seal, ("cache_sha256", "receipt_sha256"), "cache seal")
        if (
            receipt["cache_sha256"] != mark["cache_sha256"]
            or seal["cache_sha256"] != mark["cache_sha256"]
            or seal["receipt_sha256"]
            != b.file_sha256(directory / f"{mode}-receipt.json")
        ):
            raise ValueError("reader receipt/cache seal differs")
        receipts.append(receipt)
    if (
        len({r["pid"] for r in receipts}) != 3
        or len({digest(r["runtime"]) for r in receipts}) != 1
    ):
        raise ValueError("reader process/runtime identity differs")
    return trace


def validate_timing(timing):
    exact_fields(
        timing,
        (
            "discarded_compile_warmup_steps",
            "mean_milliseconds_per_step",
            "timed_steps",
            "total_timed_seconds",
        ),
        "reader timing",
    )
    for key, value in timing.items():
        require_number(value, f"reader {key}", minimum=0)
    if timing["timed_steps"] <= 0:
        raise ValueError("reader contains no timed steps")


def validate_evaluation_result(
    result, protocol, cell, *, preflight=False, legacy=False
):
    fields = set(cell) | {
        "evaluation_environment_seeds",
        "episode_returns",
        "action_saturation_fraction",
        "telemetry",
        "coverage",
        "trace_sha256",
        "checkpoint_sha256",
    }
    if not legacy or "realized_controller_config" in result:
        fields.add("realized_controller_config")
    exact_fields(result, fields, "evaluation result")
    if not finite(result) or digest({k: result[k] for k in cell}) != digest(cell):
        raise ValueError("evaluation cell identity differs or is nonfinite")
    if "realized_controller_config" in result:
        realized = validate_controller_protocol(protocol["controller"])
        if (
            validate_controller_protocol(result["realized_controller_config"])
            != realized
        ):
            raise ValueError("realized controller configuration differs")
    seeds = (
        protocol["preflight"]["evaluation_seeds"]
        if preflight
        else protocol["evaluation_environment_seeds"]
    )
    if digest(result["evaluation_environment_seeds"]) != digest(seeds):
        raise ValueError("evaluation environment seeds differ")
    if not isinstance(result["episode_returns"], list) or len(
        result["episode_returns"]
    ) != len(seeds):
        raise ValueError("evaluation episode set differs")
    for value in result["episode_returns"]:
        require_number(value, "episode return")
    require_number(
        result["action_saturation_fraction"], "saturation fraction", minimum=0
    )
    if result["action_saturation_fraction"] > 1:
        raise ValueError("invalid saturation fraction")
    if not isinstance(result["telemetry"], dict) or not isinstance(
        result["coverage"], dict
    ):
        raise ValueError("invalid evaluation telemetry/coverage")
    for key in ("trace_sha256", "checkpoint_sha256"):
        require_sha256(result[key], key)


def _artifact_context(root, directory, protocol):
    """Infer the schema from its canonical location, never from marker flags."""
    root, directory = Path(root).absolute(), Path(directory).absolute()
    try:
        parts = directory.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("artifact outside study") from exc
    if directory.resolve() != root.resolve().joinpath(*parts):
        raise ValueError("artifact location traverses a symlink")
    if not parts:
        return "final", {"stage": "final"}, None
    if (
        len(parts) not in (2, 3)
        or parts[0] not in STAGES
        or not re.fullmatch(r"[0-9]{3}", parts[1])
    ):
        raise ValueError("noncanonical artifact location")
    stage, index = parts[0], int(parts[1])
    expected = cells(protocol, stage)
    if index >= len(expected):
        raise ValueError("unexpected cell index")
    cell = expected[index]
    if len(parts) == 3:
        if stage != "preflight" or not re.fullmatch(r"arm-[0-9]+", parts[2]):
            raise ValueError("noncanonical preflight artifact")
        arm_index = int(parts[2][4:])
        if arm_index >= len(protocol["arms"]) or parts[2] != f"arm-{arm_index}":
            raise ValueError("unexpected preflight arm")
        cell = dict(
            cell,
            actor_seed=protocol["preflight"]["actor_seeds"][0],
            arm=protocol["arms"][arm_index],
        )
        return "preflight_arm", cell, directory.parent / "training"
    dependency = (
        cell_dir(root, "training", cell["training_index"])
        if stage == "evaluation"
        else None
    )
    return stage, cell, dependency


def verify_files(root, directory):
    m = manifest(root)
    protocol, directory = m["protocol"], Path(directory)
    stage, cell, training_dir = _artifact_context(root, directory, protocol)
    replay = stage in ("evaluation", "preflight_arm")
    expected_files = (
        EVALUATION_FILES
        if replay
        else (
            TRAINING_FILES
            if stage == "training"
            else {"report.json"} if stage == "final" else {"result.json"}
        )
    )
    if replay or stage == "training":
        if {path.name for path in directory.iterdir()} != set(expected_files) | {
            "verified.json"
        }:
            raise ValueError("stage artifact directory set differs")
    mark = verify_bound_files(m, directory, expected_files, replay=replay)
    result = read(directory / ("report.json" if stage == "final" else "result.json"))
    if not finite(result):
        raise ValueError("nonfinite result")
    if replay:
        validate_evaluation_result(
            result,
            protocol,
            cell,
            preflight=stage == "preflight_arm",
            legacy=m["source_commit"] in LEGACY_RESULT_COMMITS,
        )
        if digest(mark["cell"]) != digest(result):
            raise ValueError("evaluation marker/result differs")
        if stage == "evaluation":
            dependency = verify_files(root, training_dir)
            expected_checkpoint = dependency["files"]["checkpoint.pkl"]
            runtime_dir = cell_dir(root, "preflight", 0)
            pf = verify_bound_files(m, runtime_dir, {"result.json"})
            pf_result = read(runtime_dir / "result.json")
            _validate_preflight_result(pf_result, cells(protocol, "preflight")[0])
            if digest(pf["cell"]) != digest(pf_result["cell"]):
                raise ValueError("preflight marker/result differs")
            expected_runtime = pf_result["runtime"]
        else:
            training = read(training_dir / "result.json")
            validate_training_result(
                training,
                cell={k: cell[k] for k in ("index", "task", "world_model_seed")},
                preflight=True,
            )
            expected_runtime = training["runtime"]
            expected_checkpoint = b.file_sha256(training_dir / "checkpoint.pkl")
        if result["checkpoint_sha256"] != expected_checkpoint:
            raise ValueError("evaluation checkpoint binding differs")
        trace = verify_replay_receipts(
            directory, mark, result, expected_runtime=expected_runtime
        )
        required = {
            "actions",
            "observations",
            "rewards",
            "continuations",
            "is_last",
            "lengths",
            "evaluation_seeds",
        }
        if not required <= set(trace):
            raise ValueError("incomplete retained trace")
        lengths = np.asarray(trace["lengths"])
        steps = (
            protocol["preflight"]["evaluation_steps"]
            if stage == "preflight_arm"
            else protocol["maximum_environment_steps"]
        )
        if (
            lengths.shape != (len(result["episode_returns"]),)
            or lengths.dtype.kind not in "iu"
            or np.any(lengths <= 0)
            or np.any(lengths > steps)
        ):
            raise ValueError("invalid trace episode lengths")
        if any(
            np.asarray(trace[k]).shape[:2] != (len(lengths), int(lengths.max()))
            for k in required - {"lengths", "evaluation_seeds"}
        ):
            raise ValueError("trace episode/step shape differs")
        if (
            np.asarray(trace["evaluation_seeds"]).dtype.kind not in "iu"
            or trace["evaluation_seeds"].tolist()
            != result["evaluation_environment_seeds"]
        ):
            raise ValueError("retained trace evaluation seeds differ")
        rewards = np.asarray(trace["rewards"])
        if rewards.ndim != 2 or rewards.dtype not in (
            np.dtype("float32"),
            np.dtype("float64"),
        ):
            raise ValueError("invalid retained reward dtype/shape")
        mask = np.arange(rewards.shape[1])[None] < lengths[:, None]
        reward_sums = np.sum(np.where(mask, rewards, 0), axis=1, dtype=np.float64)
        # Episode returns use the original float64 environment rewards, while
        # historical traces retain float32. Bound only that storage rounding;
        # trace replay and the two reader result digests remain bitwise exact.
        rounding = (
            np.sum(
                np.where(mask, np.abs(np.spacing(rewards)), 0), axis=1, dtype=np.float64
            )
            / 2
        )
        rounding += (
            8
            * np.finfo(np.float64).eps
            * np.maximum(1, np.sum(np.abs(rewards), axis=1, dtype=np.float64))
        )
        if np.any(
            np.abs(np.asarray(result["episode_returns"]) - reward_sums) > rounding
        ):
            raise ValueError("episode returns differ from retained rewards")
        if (
            np.asarray(trace["actions"]).ndim != 3
            or np.asarray(trace["actions"]).shape[-1] < 1
        ):
            raise ValueError("invalid retained action shape")
        saturation = float(np.mean(np.abs(trace["actions"][mask]) >= 0.95))
        if result["action_saturation_fraction"] != saturation:
            raise ValueError("saturation statistic differs from retained actions")
    else:
        if digest(mark["cell"]) != digest(cell):
            raise ValueError("cell marker identity differs")
        if stage == "preflight":
            _validate_preflight_result(result, cell)
        elif stage == "training":
            validate_training_result(result, cell=cell, preflight=False)
            validate_runtime(result["runtime"])
        else:
            validate_final_report(result, m)
            for stage_name in STAGES:
                if not (Path(root) / "verified" / f"{stage_name}.json").is_file():
                    raise ValueError("final stage aggregate is missing")
                checked = stage_verify(root, stage_name)
                if digest(checked) != digest(result["stages"][stage_name]):
                    raise ValueError("final stage artifact binding differs")
            for key, stage_name in (
                ("records", "evaluation"),
                ("training_results", "training"),
            ):
                bound = [
                    read(cell_dir(root, stage_name, c["index"]) / "result.json")
                    for c in cells(protocol, stage_name)
                ]
                if digest(result[key]) != digest(bound):
                    raise ValueError("final result artifact binding differs")
            for c in cells(protocol, "evaluation"):
                output = cell_dir(root, "evaluation", c["index"])
                roles = {
                    mode: read(output / f"{mode}-receipt.json") for mode in REPLAY_MODES
                }
                bound_timing = dict(
                    role_wall_seconds={
                        mode: r["wall_seconds"] for mode, r in roles.items()
                    },
                    controller_latency=roles["create"]["timing"],
                )
                if digest(
                    result["runtime"]["evaluation_by_cell"][str(c["index"])]
                ) != digest(bound_timing):
                    raise ValueError("final reader timing artifact binding differs")
    return mark


def _validate_preflight_result(result, cell):
    exact_fields(result, ("cell", "runtime", "status"), "preflight result")
    if digest(result["cell"]) != digest(cell) or result["status"] != "passed":
        raise ValueError("preflight result identity/status differs")
    validate_runtime(result["runtime"])


def validate_training_result(result, *, cell=None, preflight=False):
    exact_fields(
        result,
        (
            "cell",
            "preflight",
            "runtime",
            "source_world_sha256",
            "config",
            "budgets",
            "reward_optimizer_updates",
            "normalization_train_targets",
            "reward_test",
            "policies",
            "endpoints",
            "replay_sha256",
            "schedule_sha256",
            "wall_seconds",
        ),
        "training result",
    )
    if not finite(result) or result["preflight"] is not preflight:
        raise ValueError("invalid training result")
    if cell is not None and digest(result["cell"]) != digest(cell):
        raise ValueError("training result cell differs")
    require_number(result["wall_seconds"], "training wall time", minimum=0)
    for name in ("source_world_sha256", "replay_sha256"):
        require_sha256(result[name], name)


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
    if (
        set(found) != expected
        or len([r for r in rows if r[0] in expected]) != len(expected)
        or any(r[1:3] != ["COMPLETED", "0:0"] for r in found.values())
    ):
        raise RuntimeError(f"scheduler not fully successful: {out}")
    result = dict(raw=out, records=list(found.values()))
    validate_accounting(result, str(job), count)
    return result


def validate_accounting(value, job, count):
    exact_fields(value, ("raw", "records"), "accounting")
    if not isinstance(value["raw"], str) or not isinstance(value["records"], list):
        raise ValueError("invalid accounting")
    if not isinstance(job, str) or not re.fullmatch(r"[0-9]+", job):
        raise ValueError("invalid accounting job identity")
    expected = {f"{job}_{i}" for i in range(count)} if count else {str(job)}
    rows = value["records"]
    if (
        len(rows) != len(expected)
        or any(not isinstance(r, list) or len(r) != 5 for r in rows)
        or {r[0] for r in rows} != expected
    ):
        raise ValueError("accounting cell set differs")
    for row in rows:
        if (
            any(not isinstance(x, str) for x in row)
            or row[1:3] != ["COMPLETED", "0:0"]
            or not re.fullmatch(r"[0-9]+", row[3])
            or "gres/gpu=1" not in row[4].split(",")
        ):
            raise ValueError("accounting status/allocation differs")
    raw_rows = [line.split("|") for line in value["raw"].splitlines() if line.strip()]
    bound = [r for r in raw_rows if r[0] in expected]
    if sorted(bound) != sorted(rows):
        raise ValueError("accounting raw/record binding differs")


def validate_submission(receipt, m, stage):
    exact_fields(
        receipt,
        ("job_id", "manifest_sha256", "stage", "count", "script_sha256"),
        "submission",
    )
    if (
        receipt["manifest_sha256"] != digest(m)
        or receipt["stage"] != stage
        or type(receipt["count"]) is not int
        or receipt["count"] != len(cells(m["protocol"], stage))
        or not isinstance(receipt["job_id"], str)
        or not re.fullmatch(r"[0-9]+", receipt["job_id"])
    ):
        raise ValueError("submission identity/count differs")
    require_sha256(receipt["script_sha256"], "submission script")


def validate_stage_record(value, m, stage, job):
    exact_fields(
        value,
        ("stage", "manifest_sha256", "accounting", "cell_markers"),
        "stage aggregate",
    )
    expected = cells(m["protocol"], stage)
    if value["stage"] != stage or value["manifest_sha256"] != digest(m):
        raise ValueError("stage aggregate identity differs")
    exact_fields(
        value["cell_markers"], {str(c["index"]) for c in expected}, "stage cell markers"
    )
    for sha in value["cell_markers"].values():
        require_sha256(sha, "stage cell marker")
    validate_accounting(value["accounting"], job, len(expected))


def validate_final_report(report, m):
    """Strict final schema; finalize additionally reconstructs it from artifacts."""
    exact_fields(
        report,
        (
            "scope",
            "tasks",
            "cells",
            "episodes",
            "arm_task_iqm",
            "contrasts",
            "source_commit",
            "manifest_sha256",
            "artifact_verification",
            "limitations",
            "stages",
            "records",
            "training_results",
            "runtime",
        ),
        "final report",
    )
    p = m["protocol"]
    if (
        not finite(report)
        or report["source_commit"] != m["source_commit"]
        or report["manifest_sha256"] != digest(m)
        or report["scope"] != "fixed_three_task_two_contrast_replication"
        or report["artifact_verification"]
        != "all stage/cell hashes and bitwise evaluation replay verified"
        or report["limitations"] != p["limitations"]
        or report["tasks"] != p["tasks"]
    ):
        raise ValueError("final report identity differs")
    expected = cells(p, "evaluation")
    if (
        type(report["cells"]) is not int
        or report["cells"] != len(expected)
        or type(report["episodes"]) is not int
        or report["episodes"] != len(expected) * len(p["evaluation_environment_seeds"])
        or not isinstance(report["records"], list)
        or len(report["records"]) != len(expected)
    ):
        raise ValueError("final evaluation cell set differs")
    for result, cell in zip(report["records"], expected):
        validate_evaluation_result(
            result, p, cell, legacy=m["source_commit"] in LEGACY_RESULT_COMMITS
        )
    expected_training = cells(p, "training")
    if not isinstance(report["training_results"], list) or len(
        report["training_results"]
    ) != len(expected_training):
        raise ValueError("final training cell set differs")
    for result, cell in zip(report["training_results"], expected_training):
        validate_training_result(result, cell=cell)
    exact_fields(report["stages"], STAGES, "final stages")
    exact_fields(
        report["runtime"],
        (
            "allocation_gpu_seconds_by_stage",
            "training_wall_seconds_by_cell",
            "evaluation_by_cell",
            "scope",
        ),
        "final runtime",
    )
    exact_fields(
        report["runtime"]["allocation_gpu_seconds_by_stage"],
        STAGES,
        "final stage runtime",
    )
    exact_fields(
        report["runtime"]["training_wall_seconds_by_cell"],
        {str(c["index"]) for c in expected_training},
        "final training runtime",
    )
    exact_fields(
        report["runtime"]["evaluation_by_cell"],
        {str(c["index"]) for c in expected},
        "final evaluation runtime",
    )
    for stage in STAGES:
        # No inferred job id is trusted independently: the raw/records bindings
        # and stage_verify's authenticated submission are checked separately.
        records = report["stages"][stage].get("accounting", {}).get("records", [])
        if not records or not records[0] or not isinstance(records[0][0], str):
            raise ValueError("missing final accounting")
        validate_stage_record(
            report["stages"][stage], m, stage, records[0][0].split("_")[0]
        )
        value = report["runtime"]["allocation_gpu_seconds_by_stage"][stage]
        if type(value) is not int or value != sum(int(r[3]) for r in records):
            raise ValueError("final allocation accounting differs")
    from .mechanism_replication_analysis import analyze

    analysis = analyze(report["records"], p)
    for key in ("tasks", "cells", "episodes", "arm_task_iqm", "contrasts"):
        if digest(report[key]) != digest(analysis[key]):
            raise ValueError("final reconstructed analysis differs")
    expected_training_times = {
        str(r["cell"]["index"]): r["wall_seconds"] for r in report["training_results"]
    }
    if digest(report["runtime"]["training_wall_seconds_by_cell"]) != digest(
        expected_training_times
    ):
        raise ValueError("final training timing binding differs")
    for value in report["runtime"]["evaluation_by_cell"].values():
        exact_fields(
            value, ("role_wall_seconds", "controller_latency"), "final cell runtime"
        )
        exact_fields(
            value["role_wall_seconds"], REPLAY_MODES, "final reader wall times"
        )
        for seconds in value["role_wall_seconds"].values():
            require_number(seconds, "final reader wall time", minimum=0)
        validate_timing(value["controller_latency"])
    if (
        report["runtime"]["scope"]
        != "one_GPU_per_allocation; successful stage allocations; queue time excluded; process timings exclude shell seals"
    ):
        raise ValueError("final runtime scope differs")


def stage_verify(root, stage):
    m = manifest(root)
    receipt = read(Path(root) / "submissions" / f"{stage}.json")
    validate_submission(receipt, m, stage)
    expected = cells(m["protocol"], stage)
    acct = accounting(receipt["job_id"], len(expected))
    marks = {}
    for cell in expected:
        directory = cell_dir(root, stage, cell["index"])
        verify_files(root, directory)
        if stage == "preflight":
            training_result = verify_training(
                m["protocol"], directory / "training", preflight=True
            )
            validate_training_result(training_result, cell=cell, preflight=True)
            if training_result["runtime"] != read(directory / "result.json")["runtime"]:
                raise ValueError("preflight training/runtime differs")
            for i in range(len(m["protocol"]["arms"])):
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
    validate_stage_record(result, m, stage, receipt["job_id"])
    target = Path(root) / "verified" / f"{stage}.json"
    if target.exists():
        saved = read(target)
        validate_stage_record(saved, m, stage, receipt["job_id"])
        if digest(saved) != digest(result):
            raise ValueError("stage aggregate changed")
    else:
        publish(target, result)
    print("MECHANISM_REPLICATION_STAGE_VERIFIED", stage, flush=True)
    return result


def worker(root, stage, index):
    m = manifest(root)
    p = m["protocol"]
    receipt = read(Path(root) / "submissions" / f"{stage}.json")
    validate_submission(receipt, m, stage)
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
        if digest(read(root / "report.json")) != digest(report):
            raise ValueError("existing final report differs")
    else:
        publish(root / "report.json", report)
        marker(root, root, {"stage": "final"}, ["report.json"])
    verify_files(root, root)
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
