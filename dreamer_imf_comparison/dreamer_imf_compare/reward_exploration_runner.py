"""Budgeted reward-readout/exploration continuation, isolated from old studies.

Only this runner steps environments. Offline fitting, preflight, and verification
do not. Original checkpoints and retained data are read-only dependencies.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from . import dreamer_ablation_runner as base
from .parallel_collection import _write_npz
from .reward_readout_study import sha

SOURCE = Path(__file__).resolve().parents[2]
OBS_KEYS = ("position", "to_target", "velocity")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from .reward_exploration_protocol import write_json_exclusive

    write_json_exclusive(Path(path), value)


def save_pickle(path, value):
    from .reward_exploration_protocol import write_bytes_exclusive

    write_bytes_exclusive(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def spaces(elements):
    obs = {k: elements.Space(np.float32, (2,)) for k in OBS_KEYS}
    obs["reward"] = elements.Space(np.float32, ())
    obs.update(
        {k: elements.Space(bool) for k in ("is_first", "is_last", "is_terminal")}
    )
    return obs, {"action": elements.Space(np.float32, (2,), -1, 1)}


def config_for(parent_cell, directory, seed, upstream_path):
    """Keep parent's model/learning configuration; change only runtime location."""
    from ruamel.yaml import YAML

    _, elements, _, main = base._imports({"upstream_path": upstream_path})
    values = YAML(typ="safe").load((Path(parent_cell) / "config.yaml").read_text())
    # Parent YAML predates wrapper-only performance flags. Config.update cannot
    # introduce new keys, so insert explicitly before creating immutable Config.
    values["jax"].update(precompile=False, profiler=False)
    values.update(logdir=str(directory), seed=int(seed))
    values["batch_size"], values["batch_length"] = 16, 64
    values["run"]["envs"] = 16
    values["run"]["train_ratio"] = 512.0
    config = elements.Config(values)
    if not config.agent.reward_grad or config.agent.repval_grad:
        raise ValueError("parent gradient protocol changed")
    return config, elements, main


def group_deltas(before, after):
    groups = {}
    for key, old in before.items():
        value = np.asarray(after[key], np.float64)
        old = np.asarray(old, np.float64)
        if value.shape != old.shape or not np.isfinite(value).all():
            raise ValueError("invalid updated state " + key)
        group = key.split("/")[0]
        record = groups.setdefault(
            group, dict(squared_delta=0.0, elements=0, changed=0)
        )
        record["squared_delta"] += float(np.sum((value - old) ** 2))
        record["elements"] += int(old.size)
        record["changed"] += int(np.count_nonzero(value != old))
    for record in groups.values():
        record["delta_norm"] = float(np.sqrt(record.pop("squared_delta")))
    return groups


def runtime_evidence(agent):
    result = base._runtime_telemetry(agent, "cuda")
    fixed = {"input_mean", "input_std", "target_mean", "target_std", "bonus_scale"}
    prefixes = ("control_rew", "explore_ensemble", "explore_pol", "explore_val")
    counts = {
        prefix: sum(
            int(value.size)
            for key, value in agent.params.items()
            if key.startswith(prefix + "/") and key.split("/")[-1] not in fixed
        )
        for prefix in prefixes
    }
    result.update(
        auxiliary_trainable_parameters_by_module=counts,
        total_trainable_parameters=result["model_parameter_count"]
        + sum(counts.values()),
        resident_parameter_and_optimizer_elements=sum(
            int(x.size) for x in agent.params.values()
        ),
        auxiliary_target_parameters=sum(
            int(x.size)
            for k, x in agent.params.items()
            if k.startswith("explore_slowval/")
        ),
    )
    return result


def load_continuation(agent, parent_cell, export):
    """Merge fresh auxiliary state with exact parent parameters AND optimizer."""
    from .parallel_frozen import parameter_digest

    with (Path(parent_cell) / "checkpoint_200000.pkl").open("rb") as stream:
        parent = pickle.load(stream)
    initialized = agent.save()
    for key, value in parent["params"].items():
        if (
            key not in initialized["params"]
            or initialized["params"][key].shape != value.shape
        ):
            raise ValueError("incompatible parent state " + key)
    original_keys = set(parent["params"])
    parameters = dict(initialized["params"])
    parameters.update(parent["params"])
    # Common exploration initialization must not depend on how many random keys
    # the two reward architectures consumed at construction. Start from cloned
    # parent actor/value/target weights in all arms, with fresh auxiliary Adam.
    for key, value in parent["params"].items():
        prefix, _, suffix = key.partition("/")
        if prefix in ("pol", "val", "slowval"):
            destination = "explore_" + prefix + "/" + suffix
            if (
                destination not in parameters
                or parameters[destination].shape != value.shape
            ):
                raise ValueError("exploration clone mismatch " + destination)
            parameters[destination] = value.copy()
    for key, value in export.items():
        if key in original_keys or key not in parameters:
            raise ValueError("auxiliary import touches parent or unknown state " + key)
        if parameters[key].shape != np.asarray(value).shape:
            raise ValueError("auxiliary import shape " + key)
        parameters[key] = np.asarray(value, dtype=parameters[key].dtype)
    agent.load(dict(params=parameters, counters=parent["counters"]))
    loaded = agent.save()
    if parameter_digest(
        {k: loaded["params"][k] for k in original_keys}
    ) != parameter_digest(parent["params"]):
        raise ValueError("parent state changed during continuation initialization")
    return dict(
        parent_parameter_digest=parameter_digest(parent["params"]),
        counters=parent["counters"],
        imported_auxiliary_keys=sorted(export),
        original_keys=sorted(original_keys),
    )


def matched_export(output, arm):
    from .reward_exploration_match import load_export

    return load_export(
        Path(output) / "match", "existing" if arm == "A" else "categorical"
    )


def make_agent(output, parent_cell, directory, arm, seed, protocol):
    config, elements, main = config_for(
        parent_cell, directory, seed, protocol["upstream_path"]
    )
    from .reward_exploration_agent import install
    from .staged_dynamics import set_controls
    from .conditional_schedule import CONTROLS

    obs_space, act_space = spaces(elements)
    agent = install(arm)(obs_space, act_space, base._agent_config(config, elements))
    binding = load_continuation(agent, parent_cell, matched_export(output, arm))
    set_controls(agent, CONTROLS)
    config.save(str(directory / "config.yaml"))
    base._finite(agent.params, "loaded continuation state")
    return agent, config, main, binding


def retained_batch(parent_cell, agent):
    path = Path(parent_cell) / "replay_sample_200000_48985.npz"
    with np.load(path, allow_pickle=False) as f:
        data = {k: np.array(f[k]) for k in f.files}
    data.pop("seed", None)
    B, T = data["is_first"].shape
    # The retained batch intentionally contains only real observations/actions,
    # not historical replay carry entries. This is an explicitly recomputed
    # zero-carry GPU integration probe, not a historical-gradient reconstruction.
    for key, space in agent.spaces.items():
        if key not in data:
            data[key] = np.zeros((B, T, *space.shape), space.dtype)
    data["episode_id"] = np.arange(B, dtype=np.int32)[:, None] * 1000 + np.cumsum(
        data["is_first"], axis=1, dtype=np.int32
    )
    if set(data) != set(agent.spaces):
        raise ValueError(f"retained batch keys differ: {set(data) ^ set(agent.spaces)}")
    if (B, T) != (16, 64 + agent.config.replay_context):
        raise ValueError("preflight is not the full training shape")
    return data, path


def preflight_arm(output, parent_cell, arm, protocol):
    import jax

    directory = Path(output) / "preflight" / arm
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    agent, config, _, binding = make_agent(
        output, parent_cell, directory, arm, 701, protocol
    )
    before = agent.save()
    batch, batch_path = retained_batch(parent_cell, agent)
    stream = agent.stream(iter([dict(batch), dict(batch)]))
    carry = agent.init_train(config.batch_size)
    stream = iter(stream)
    for _ in range(2):
        data = next(stream)
        carry, _, metrics = agent.train(carry, data)
        jax.block_until_ready(agent.params)
        base._finite(metrics, "preflight metrics")
    if agent.pending_mets is not None:
        metrics = agent._take_outs(agent.pending_mets)
    base._finite(metrics, "preflight final metrics")
    base._finite(agent.params, "preflight final parameters")
    after = agent.save()
    if after["counters"]["updates"] - before["counters"]["updates"] != 2:
        raise ValueError("preflight did not execute two full learner updates")
    delta = group_deltas(before["params"], after["params"])
    for group in ("enc", "dyn", "rew", "control_rew", "explore_pol", "explore_val"):
        if group not in delta or not delta[group]["changed"]:
            raise ValueError("preflight inactive required group " + group)
    # Both policy traces must run using the exact production observation shape;
    # no environment is constructed, and no rollout is interpreted as return.
    obs = {k: batch[k][:, 0] for k in agent.obs_space}
    for mode in ("train", "explore", "eval"):
        _, action, outs = agent.policy(agent.init_policy(16), obs, mode=mode)
        base._finite((action, outs), "preflight " + mode)
    # Exercise the exact long prediction-probe shapes without a simulator. The
    # repeated retained observations are only a compilation/integration input.
    from .reward_exploration_probe import retained_probe

    probe_data = {
        k: np.tile(batch[k][:5], (1, 8, *([1] * (batch[k].ndim - 2))))[:, :501]
        for k in agent.obs_space
    }
    probe_data["action"] = np.tile(batch["action"][:5], (1, 8, 1))[:, :501]
    probe_data["previous_action"] = np.concatenate(
        [np.zeros((5, 1, 2), np.float32), probe_data["action"][:, :-1]], 1
    )
    retained_probe(agent, directory, 40000, 701, data_override=probe_data)
    retained_probe(agent, directory, 40000, 701, verify=True, data_override=probe_data)
    save_pickle(directory / "checkpoint.pkl", after)
    result = dict(
        arm=arm,
        native_steps=0,
        updates=2,
        full_batch_shape=list(batch["is_first"].shape),
        parent_binding=binding,
        retained_batch_sha256=sha(batch_path),
        parameter_deltas=delta,
        metrics={k: float(v) for k, v in metrics.items() if np.ndim(v) == 0},
        runtime=runtime_evidence(agent),
        wall_seconds=time.monotonic() - started,
    )
    write(directory / "result.json", result)
    print("REWARD_EXPLORATION_PREFLIGHT_ARM_CREATED", arm, flush=True)


class BudgetedReacher(base.ProprioReacher):
    """Charge before native execution; a failed call never refunds its charge."""

    def __init__(self, seed, budget, kind):
        super().__init__(seed, repeat=2, episode_native_steps=1000)
        self.budget, self.kind = budget, kind

    def step(self, action):
        before = self.native_steps
        reset = bool(action["reset"])
        if reset:
            self.budget.record_reset()
        else:
            self.budget.reserve(2, self.kind)
        obs = super().step(action)
        if self.native_steps - before != (0 if reset else 2):
            raise RuntimeError(
                "unexpected native/reset boundary; charged budget retained"
            )
        return obs


def actual_actions(carry, action):
    """Keep recurrent previous-action input equal to clipped/executed control."""
    import jax

    clipped = np.clip(np.asarray(action, np.float32), -1, 1)
    previous = dict(carry[-1])
    previous["action"] = [
        jax.device_put(row, old.sharding)
        for row, old in zip(clipped, previous["action"])
    ]
    return (*carry[:-1], previous), clipped


class RawRows:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows, self.paths, self.count = [], [], 0

    def add(self, row):
        self.rows.append({k: np.asarray(v).copy() for k, v in row.items()})
        self.count += 1
        if len(self.rows) >= 1024:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        keys = set(self.rows[0])
        if any(set(x) != keys for x in self.rows):
            raise ValueError("raw transition schema changed")
        path = self.directory / f"chunk-{len(self.paths):04d}.npz"
        _write_npz(path, {k: np.stack([x[k] for x in self.rows]) for k in keys})
        self.paths.append(path)
        self.rows.clear()


@contextlib.contextmanager
def current_policy(agent):
    import jax
    from embodied.jax import internal

    old_policy, old_pending = agent.policy_params, agent.pending_sync
    old_counter = int(agent.n_actions)
    agent.policy_params = internal.move(
        {k: agent.params[k].copy() for k in agent.policy_keys},
        agent.policy_params_sharding,
    )
    agent.pending_sync = None
    try:
        yield
    finally:
        jax.tree.map(lambda x: x.delete(), agent.policy_params)
        agent.policy_params, agent.pending_sync = old_policy, old_pending
        agent.n_actions.value = old_counter


def evaluate(agent, main, config, seed, milestone, directory, budget):
    from .reward_exploration_protocol import eval_seeds

    begun = time.monotonic()
    episodes = []
    with current_policy(agent):
        for ep, envseed in enumerate(eval_seeds(seed, milestone)):
            base_env = BudgetedReacher(envseed, budget, "evaluate")
            env = main.wrap_env(base_env, config)
            rows = []
            agent.n_actions.value = int(
                np.random.SeedSequence([seed, 821, milestone, ep]).generate_state(1)[0]
            )
            try:
                obs = env.step(dict(reset=True, action=np.zeros(2, np.float32)))
                carry = agent.init_policy(1)
                previous = np.zeros(2, np.float32)
                while True:
                    carry, action, outs = agent.policy(
                        carry,
                        {k: np.asarray(v)[None] for k, v in obs.items()},
                        mode="eval",
                    )
                    carry, clipped = actual_actions(carry, action["action"])
                    row = {
                        **obs,
                        "action": clipped[0],
                        "previous_action": previous,
                        **{k: v[0] for k, v in outs.items() if k.startswith("log/")},
                    }
                    rows.append(row)
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
        mean_return=float(np.mean([x["return_"] for x in episodes])),
        wall_seconds=time.monotonic() - begun,
        mode="task-policy only; no intrinsic evaluation reward",
    )


def run_cell(output, parent_cell, index, protocol):
    import jax
    from .reward_exploration_protocol import (
        cell_for_index,
        collection_policy,
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
            envseed = int(
                np.random.SeedSequence([cell["seed"], 741, worker]).generate_state(1)[0]
            )
            env = BudgetedReacher(envseed, budget, "collect")
            bases.append(env)
            envs.append(main.wrap_env(env, config))
        carry = agent.init_policy(16)
        carry_train = agent.init_train(16)
        actions = [dict(reset=True, action=np.zeros(2, np.float32)) for _ in envs]
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
                # Mode is associated with action at this observation, not the
                # just-completed incoming transition. C/D schedules are identical.
                decision_index = int(worker_steps[0])
                mode = collection_policy(cell["arm"], cell["seed"], decision_index)
                carry, proposed, outs = agent.policy(
                    carry, obs, mode="explore" if mode == "explore" else "train"
                )
                proposed_action = np.asarray(proposed["action"], np.float32)
                executed = proposed_action.copy()
                if mode == "random":
                    for worker in range(16):
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [cell["seed"], 759, worker, decision_index]
                            )
                        )
                        executed[worker] = rng.uniform(-1, 1, 2)
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
                    tick = time.monotonic()
                    batch = next(stream)
                    carry_train, out, metrics = agent.train(carry_train, batch)
                    updates += 1
                    if updates <= 3 or updates % 1000 == 0:
                        jax.block_until_ready(agent.params)
                        update_timings.append(
                            dict(update=updates, seconds=time.monotonic() - tick)
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
                    progress = dict(
                        **cell,
                        native_steps=native,
                        learner_updates=updates,
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
            path = directory / f"checkpoint-{milestone}.pkl"
            save_pickle(path, agent.save())
            checkpoints.append(
                dict(path=path.name, sha256=sha(path), native_steps=milestone)
            )
            evaluation = evaluate(
                agent, main, config, cell["seed"], milestone, directory, budget
            )
            from .reward_exploration_probe import retained_probe

            probe_path = retained_probe(agent, directory, milestone, cell["seed"])
            evaluation["recomputed_probe"] = dict(
                path=probe_path.name, sha256=sha(probe_path), native_steps=0
            )
            evaluations.append(evaluation)
            write(directory / f"evaluation-{milestone}.json", evaluation)
        raw.flush()
        if (
            native != sum(x.native_steps for x in bases)
            or native != 80000
            or budget.total != 90000
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
            wall_seconds=time.monotonic() - start,
            replay_restart=True,
            parent_world_seed=431,
            parent_native_steps=200000,
        )
        write(directory / "result.json", result)
        print("REWARD_EXPLORATION_CELL_CREATED", index, flush=True)
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


def replay_probes(output, index, protocol):
    """Separate-process replay against saved checkpoints, with no agent.train."""
    import types
    import jax
    from embodied.jax import internal
    from .reward_exploration_protocol import cell_for_index, cell_directory
    from .reward_exploration_agent import install
    from .reward_exploration_probe import retained_probe

    base._imports(protocol)
    internal.setup(
        platform="cuda",
        compute_dtype="bfloat16",
        transfer_guard=False,
        prealloc=False,
        compilation_cache=False,
    )
    cell = cell_for_index(index)
    directory = cell_directory(output, index)
    config, elements, _ = config_for(
        directory, directory, cell["seed"], protocol["upstream_path"]
    )
    obs, act = spaces(elements)
    cls = install(cell["arm"])
    model = object.__new__(cls)
    model.__init__(obs, act, base._agent_config(config, elements))
    for milestone in (40000, 80000):
        with (directory / f"checkpoint-{milestone}.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        params = jax.device_put(saved["params"])
        proxy = types.SimpleNamespace(model=model, params=params, obs_space=obs)
        retained_probe(proxy, directory, milestone, cell["seed"], verify=True)
        jax.tree.map(lambda x: x.delete(), params)
    write(
        directory / "prediction_replay.json",
        dict(exact=True, milestones=[40000, 80000], native_steps=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "match",
            "verify-match",
            "preflight",
            "preflight-arm",
            "verify-preflight",
            "cell",
            "verify-cell",
            "finalize",
            "verify-final",
            "verify-finalize",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-cell", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--arm", choices=("A", "D"))
    args = parser.parse_args()
    from .reward_exploration_protocol import authenticate, require_marker

    manifest, protocol = authenticate(args.output)
    parent = args.parent_cell or Path(manifest["inputs"]["parent_cell"])
    dataset = args.dataset or Path(manifest["inputs"]["dataset"]["path"])
    if (
        parent.resolve() != Path(manifest["inputs"]["parent_cell"]).resolve()
        or dataset.resolve() != Path(manifest["inputs"]["dataset"]["path"]).resolve()
    ):
        raise ValueError("CLI dependencies differ from the immutable manifest")
    protocol = dict(protocol, upstream_path=manifest["upstream"]["path"])
    if args.stage == "match":
        base._imports(protocol)
        from embodied.jax import internal

        internal.setup(
            platform="cuda",
            compute_dtype="bfloat16",
            transfer_guard=False,
            prealloc=False,
            compilation_cache=False,
        )
        from .reward_exploration_match import run_match

        run_match(args.output / "match", dataset, parent, protocol)
    elif args.stage == "verify-match":
        base._imports(protocol)
        from embodied.jax import internal

        internal.setup(
            platform="cuda",
            compute_dtype="bfloat16",
            transfer_guard=False,
            prealloc=False,
            compilation_cache=False,
        )
        from .reward_exploration_match import verify_match
        from .reward_exploration_protocol import write_marker

        verify_match(args.output / "match", dataset, parent, protocol)
        write_marker(
            args.output,
            "match",
            sorted(p for p in (args.output / "match").rglob("*") if p.is_file()),
            details={"native_steps": 0},
        )
    elif args.stage in ("preflight", "preflight-arm"):
        require_marker(args.output, "match")
        if args.stage == "preflight-arm":
            preflight_arm(args.output, parent, args.arm, protocol)
        else:
            for arm in ("A", "D"):
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        __package__ + ".reward_exploration_runner",
                        "preflight-arm",
                        "--output",
                        str(args.output),
                        "--parent-cell",
                        str(parent),
                        "--dataset",
                        str(dataset),
                        "--arm",
                        arm,
                    ],
                    check=True,
                )
    elif args.stage == "cell":
        require_marker(args.output, "preflight")
        run_cell(args.output, parent, args.index, protocol)
    else:
        from .reward_exploration_verify import (
            verify_cell,
            verify_preflight,
            finalize,
            verify_final,
        )

        if args.stage == "verify-cell":
            replay_probes(args.output, args.index, protocol)
            verify_cell(args.output, args.index)
        elif args.stage == "verify-preflight":
            verify_preflight(args.output)
        elif args.stage == "finalize":
            finalize(args.output)
        else:
            verify_final(args.output)


if __name__ == "__main__":
    main()
