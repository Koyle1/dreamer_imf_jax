"""Exact-budget online experiment using the pinned, unmodified Dreamer learner.

Only collection/accounting replaces embodied.run.train: reset observations remain
in replay but do not consume interaction budget or train-ratio credit. Evaluation
uses current parameters and independent policy RNG, preserving training state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

UPSTREAM_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
ARMS = ("categorical", "gaussian", "imf")


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _settings(protocol, preflight):
    result = dict(
        native_steps=500000,
        action_repeat=2,
        eval_at_native_steps=[100000, 200000, 300000, 400000, 500000],
        eval_episodes=5,
        envs=16,
        train_ratio=512,
        model_size="size12m",
        batch_size=16,
        batch_length=64,
        jax_platform="cuda",
        episode_native_steps=1000,
    )
    result.update({k: protocol[k] for k in result if k in protocol})
    if preflight:
        result.update(
            native_steps=4096,
            eval_at_native_steps=[4096],
            eval_episodes=1,
            max_updates=2,
        )
        result.update(protocol.get("preflight", {}))
    for name in (
        "native_steps",
        "action_repeat",
        "envs",
        "batch_size",
        "batch_length",
        "eval_episodes",
    ):
        if int(result[name]) != result[name] or result[name] <= 0:
            raise ValueError(f"Invalid {name}: {result[name]}")
    marks = result["eval_at_native_steps"]
    if not marks or marks != sorted(set(marks)) or marks[-1] != result["native_steps"]:
        raise ValueError("Evaluation checkpoints must increase and end at native_steps")
    if any(x <= 0 or x % result["action_repeat"] for x in marks):
        raise ValueError(
            "Checkpoint budgets must be positive multiples of action_repeat"
        )
    if result["episode_native_steps"] % result["action_repeat"]:
        raise ValueError("Episode length must be divisible by action_repeat")
    if not preflight and (
        result["native_steps"] != 500000
        or result["action_repeat"] != 2
        or marks != [100000, 200000, 300000, 400000, 500000]
        or result["eval_episodes"] != 5
        or result["model_size"] != "size12m"
        or result["train_ratio"] != 512
    ):
        raise ValueError("Production protocol differs from the approved experiment")
    return result


def _imports(protocol):
    upstream = Path(protocol["upstream_path"]).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"Wrong upstream commit: {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty.strip():
        raise ValueError("Upstream checkout contains modified tracked files")
    sys.path.insert(0, str(upstream))
    import dreamerv3

    if Path(dreamerv3.__file__).resolve().parent != upstream / "dreamerv3":
        raise RuntimeError("A different Dreamer is already imported")
    import elements
    import embodied
    from dreamerv3 import main

    return upstream, elements, embodied, main


class ProprioReacher:
    """No renderer is created; count simulator transitions at the native boundary."""

    def __init__(self, seed, repeat=2, episode_native_steps=1000):
        from dm_control import suite
        from embodied.envs.from_dm import FromDM
        import embodied

        self.native_steps = 0
        self.resets = 0
        # Reacher's native control timestep is 0.02 seconds (1000 default steps).
        self.dm = suite.load(
            "reacher",
            "hard",
            task_kwargs={
                "random": int(seed),
                "time_limit": episode_native_steps * 0.02,
            },
        )
        parent = self

        class Counted:
            def observation_spec(self):
                return parent.dm.observation_spec()

            def action_spec(self):
                return parent.dm.action_spec()

            def reset(self):
                parent.resets += 1
                return parent.dm.reset()

            def step(self, action):
                result = parent.dm.step(action)
                parent.native_steps += 1
                return result

        self.env = embodied.wrappers.ActionRepeat(FromDM(Counted()), repeat)

    @property
    def obs_space(self):
        return self.env.obs_space

    @property
    def act_space(self):
        return self.env.act_space

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.dm.close()


def _config(upstream, elements, settings, seed, directory):
    import ruamel.yaml

    configs = ruamel.yaml.YAML(typ="safe").load(
        (upstream / "dreamerv3/configs.yaml").read_text()
    )
    cfg = elements.Config(configs["defaults"]).update(configs[settings["model_size"]])
    return cfg.update(
        {
            "logdir": str(directory),
            "seed": seed,
            "task": "dmc_reacher_hard",
            "batch_size": settings["batch_size"],
            "batch_length": settings["batch_length"],
            "jax.platform": settings["jax_platform"],
            "jax.prealloc": False,
            "jax.compute_dtype": (
                "float32" if settings["jax_platform"] == "cpu" else "bfloat16"
            ),
            "run.envs": settings["envs"],
            "run.train_ratio": float(settings["train_ratio"]),
            "env.dmc.repeat": settings["action_repeat"],
            "env.dmc.image": False,
        }
    )


def _agent_config(config, elements):
    return elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )


def _finite(tree, label):
    import jax

    for value in jax.tree.leaves(tree):
        # Upstream enables transfer_guard=disallow on GPU. Validation is an
        # intentional host read, not an accidental implicit transfer.
        array = np.asarray(jax.device_get(value))
        if array.dtype.kind not in ("O", "U", "S") and not np.isfinite(array).all():
            raise FloatingPointError(f"Nonfinite {label}")


def _runtime_telemetry(agent, requested_platform):
    """Count precisely the modules passed to the unmodified optimizer."""
    devices = [
        dict(
            id=int(d.id),
            platform=d.platform,
            device_kind=d.device_kind,
            process_index=int(d.process_index),
            platform_version=str(d.client.platform_version),
        )
        for d in agent.train_devices
    ]
    if requested_platform == "cuda" and not all(
        d["platform"] == "gpu" and "cuda" in d["platform_version"].lower()
        for d in devices
    ):
        raise RuntimeError(f"CUDA requested but actual learner devices are {devices}")
    counts = {
        module.path: sum(
            int(x.size)
            for key, x in agent.params.items()
            if key.startswith(module.path + "/")
        )
        for module in agent.model.modules
    }
    if not counts or not all(counts.values()):
        raise RuntimeError("Cannot identify optimizer model parameter modules")
    return dict(
        devices=devices,
        model_parameter_count=sum(counts.values()),
        model_parameters_by_module=counts,
        model_parameter_definition="Exact upstream optimizer modules; excludes optimizer state, slow value target and normalizers",
    )


def _evaluate(agent, main, config, settings, seed, checkpoint):
    import jax
    from embodied.jax import internal

    # Evaluation gets the *current* checkpoint parameters, not upstream's delayed
    # actor copy. Restore both actor copies and RNG counter after evaluation.
    old_policy, old_pending = agent.policy_params, agent.pending_sync
    old_counter = int(agent.n_actions)
    agent.policy_params = internal.move(
        {k: agent.params[k].copy() for k in agent.policy_keys},
        agent.policy_params_sharding,
    )
    agent.pending_sync = None
    returns, lengths, episode_seeds = [], [], []
    try:
        for episode in range(settings["eval_episodes"]):
            episode_seed = int(
                np.random.SeedSequence([seed, 977, episode]).generate_state(1)[0]
            )
            episode_seeds.append(episode_seed)
            agent.n_actions.value = int(
                np.random.SeedSequence([seed, 991, checkpoint, episode]).generate_state(
                    1
                )[0]
            )
            base = ProprioReacher(
                episode_seed,
                settings["action_repeat"],
                settings["episode_native_steps"],
            )
            env = main.wrap_env(base, config)
            try:
                obs = env.step(
                    {
                        "reset": True,
                        "action": np.zeros(env.act_space["action"].shape, np.float32),
                    }
                )
                carry = agent.init_policy(1)
                total = 0.0
                while not obs["is_last"]:
                    carry, act, _ = agent.policy(
                        carry,
                        {k: np.asarray(v)[None] for k, v in obs.items()},
                        mode="eval",
                    )
                    obs = env.step(
                        {"reset": False, **{k: v[0] for k, v in act.items()}}
                    )
                    total += float(obs["reward"])
                returns.append(total)
                lengths.append(base.native_steps)
            finally:
                env.close()
    finally:
        jax.tree.map(lambda x: x.delete(), agent.policy_params)
        agent.policy_params, agent.pending_sync = old_policy, old_pending
        agent.n_actions.value = old_counter
    _finite(returns, "evaluation returns")
    return dict(
        native_steps=checkpoint,
        returns=returns,
        episode_native_steps=lengths,
        episode_seeds=episode_seeds,
        mean_return=float(np.mean(returns)),
        policy_mode="upstream stochastic eval",
        learner_updates=int(agent.n_updates),
    )


def run_cell(root, arm, seed, preflight=False):
    """Run one fresh cell; an existing (even incomplete) cell is never overwritten."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    root = Path(root).resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    settings = _settings(protocol, preflight)
    directory = root / ("preflight" if preflight else "cells") / arm / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _json(
        directory / "started.json",
        dict(
            arm=arm,
            seed=seed,
            preflight=preflight,
            settings=settings,
            from_scratch=True,
        ),
    )
    upstream, elements, embodied, main = _imports(protocol)
    from .dreamer_ablation_dynamics import install

    install(arm)
    from dreamerv3.agent import Agent
    import jax

    config = _config(upstream, elements, settings, seed, directory)
    config.save(str(directory / "config.yaml"))
    bases, envs = [], []
    try:
        for index in range(settings["envs"]):
            envseed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
            base = ProprioReacher(
                envseed, settings["action_repeat"], settings["episode_native_steps"]
            )
            bases.append(base)
            envs.append(main.wrap_env(base, config))
        agent = Agent(
            envs[0].obs_space,
            {k: v for k, v in envs[0].act_space.items() if k != "reset"},
            _agent_config(config, elements),
        )
        _finite(agent.params, "initial parameters")
        runtime = _runtime_telemetry(agent, settings["jax_platform"])
        runtime["initialization_seconds"] = time.monotonic() - started
        _json(directory / "runtime.json", runtime)
        replay = main.make_replay(config, "replay")
        stream = None
        carry_train = agent.init_train(config.batch_size)
        carry = agent.init_policy(len(envs))
        acts = [
            {
                "reset": True,
                "action": np.zeros(env.act_space["action"].shape, np.float32),
            }
            for env in envs
        ]
        should_train = elements.when.Ratio(
            settings["train_ratio"] / (config.batch_size * config.batch_length)
        )
        native_steps = transitions = replay_rows = updates = 0
        evaluations = []
        metrics_last = {}
        checkpoint_records = []
        update_timings = []
        next_progress = 1000

        def progress():
            elapsed = time.monotonic() - started
            payload = dict(
                native_steps=native_steps,
                learner_updates=updates,
                wall_seconds=elapsed,
                evaluations=evaluations,
                metrics=metrics_last,
                update_timing_samples=update_timings,
                **runtime,
            )
            _json(directory / "progress.json", payload)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "arm": arm,
                        "seed": seed,
                        "native_steps": native_steps,
                        "learner_updates": updates,
                        "wall_seconds": elapsed,
                        "last_update_seconds": (
                            update_timings[-1]["seconds"] if update_timings else None
                        ),
                    }
                ),
                flush=True,
            )

        progress()
        for mark in settings["eval_at_native_steps"]:
            while native_steps < mark:
                count = min(
                    len(envs), (mark - native_steps) // settings["action_repeat"]
                )
                if count == 0:
                    raise RuntimeError("Nondivisible remaining native budget")
                selected = list(range(count))
                before = [bases[i].native_steps for i in selected]
                observations = [envs[i].step(acts[i]) for i in selected]
                deltas = [bases[i].native_steps - n for i, n in zip(selected, before)]
                obs = {
                    k: np.stack([x[k] for x in observations]) for k in observations[0]
                }
                subset = jax.tree.map(
                    lambda x: [x[i] for i in selected],
                    carry,
                    is_leaf=lambda x: isinstance(x, list),
                )
                subset, action, outs = agent.policy(subset, obs, mode="train")
                carry = jax.tree.map(
                    lambda old, new: new + old[count:],
                    carry,
                    subset,
                    is_leaf=lambda x: isinstance(x, list),
                )
                for local, worker in enumerate(selected):
                    nextact = {
                        k: v[local] * (not observations[local]["is_last"])
                        for k, v in action.items()
                    }
                    acts[worker] = {
                        **nextact,
                        "reset": bool(observations[local]["is_last"]),
                    }
                    tran = {
                        **observations[local],
                        **nextact,
                        **{k: v[local] for k, v in outs.items()},
                    }
                    replay.add(tran, worker)
                    replay_rows += 1
                    native_steps += deltas[local]
                    transitions += int(deltas[local] > 0)
                    if (
                        not deltas[local]
                        or len(replay) < config.batch_size * config.batch_length
                    ):
                        continue
                    if stream is None:
                        stream = iter(
                            agent.stream(main.make_stream(config, replay, "train"))
                        )
                    for _ in range(should_train(transitions)):
                        if updates >= settings.get("max_updates", float("inf")):
                            break
                        timed = updates < 8 or (updates + 1) % 1000 == 0
                        update_start = time.monotonic()
                        carry_train, outputs, metrics = agent.train(
                            carry_train, next(stream)
                        )
                        updates += 1
                        if timed:
                            # Explicit synchronization makes GPU timings execution
                            # times instead of asynchronous dispatch-only timings.
                            jax.block_until_ready(agent.params)
                            update_timings.append(
                                dict(
                                    update=updates,
                                    seconds=time.monotonic() - update_start,
                                    includes_replay_fetch=True,
                                    synchronized=True,
                                )
                            )
                        _finite(metrics, "training metrics")
                        if "replay" in outputs:
                            replay.update(outputs["replay"])
                        metrics_last = {
                            k: float(np.asarray(v))
                            for k, v in metrics.items()
                            if np.asarray(v).ndim == 0
                        }
                if native_steps > mark:
                    raise RuntimeError("Native interaction budget overshot")
                if native_steps >= next_progress:
                    progress()
                    next_progress = (native_steps // 1000 + 1) * 1000
            _finite(agent.params, "checkpoint parameters")
            # Drain delayed metrics to validate the final update, without training.
            if agent.pending_mets is not None:
                metrics = agent._take_outs(agent.pending_mets)
                _finite(metrics, "final training metrics")
                metrics_last = {
                    k: float(np.asarray(v))
                    for k, v in metrics.items()
                    if np.asarray(v).ndim == 0
                }
            evaluation = _evaluate(agent, main, config, settings, seed, mark)
            evaluations.append(evaluation)
            checkpoint = directory / f"checkpoint_{mark}.pkl"
            with checkpoint.open("xb") as file:
                pickle.dump(agent.save(), file, protocol=pickle.HIGHEST_PROTOCOL)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            checkpoint_records.append(
                dict(native_steps=mark, path=checkpoint.name, sha256=digest)
            )
            _json(directory / f"evaluation_{mark}.json", evaluation)
            progress()
        if not updates or updates != int(agent.n_updates):
            raise RuntimeError("No learner updates or inconsistent update count")
        if preflight and updates < 2:
            raise RuntimeError(
                "Preflight requires at least two genuine learner updates"
            )
        if native_steps != sum(x.native_steps for x in bases):
            raise RuntimeError("Simulator and training native-step counters disagree")
        if replay_rows != transitions + sum(x.resets for x in bases):
            raise RuntimeError("Reset/transition/replay row accounting disagrees")
        result = dict(
            arm=arm,
            seed=seed,
            preflight=preflight,
            completed=True,
            upstream_commit=UPSTREAM_COMMIT,
            native_steps=native_steps,
            agent_transitions=transitions,
            replay_rows=replay_rows,
            reset_rows=sum(x.resets for x in bases),
            learner_updates=updates,
            replay_sampled_steps=updates * config.batch_size * config.batch_length,
            heldout_native_steps=sum(
                sum(x["episode_native_steps"]) for x in evaluations
            ),
            evaluations=evaluations,
            checkpoints=checkpoint_records,
            metrics=metrics_last,
            finite_parameters=True,
            from_scratch=True,
            settings=settings,
            runtime=runtime,
            update_timing_samples=update_timings,
            parameter_elements_including_optimizer=sum(
                int(x.size) for x in jax.tree.leaves(agent.params)
            ),
            wall_seconds=time.monotonic() - started,
        )
        _json(directory / "complete.json", result)
        return result
    except BaseException as exc:
        _json(
            directory / "failed.json",
            dict(error=repr(exc), wall_seconds=time.monotonic() - started),
        )
        raise
    finally:
        for env in envs:
            env.close()
