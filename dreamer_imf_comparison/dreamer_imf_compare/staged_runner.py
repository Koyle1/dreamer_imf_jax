"""Staged scratch learning around the unchanged native-step collection harness."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np

from . import dreamer_ablation_runner as base


class Schedule:
    """Preregistered gates consume model diagnostics, never evaluation returns."""

    def __init__(self):
        self.scale = 1.0
        self.passes = 0
        self.reliable = False
        self.last_observed = -1

    def observe(self, native, metrics):
        if native <= self.last_observed:
            return
        self.last_observed = native
        calibration = metrics["calibration"]
        numerator = calibration["shared_core_model_grad_norm"]
        denominator = calibration["shared_core_dynamics_grad_norm"]
        if not np.isfinite([numerator, denominator]).all():
            raise ValueError("Nonfinite gradient calibration")
        if denominator > 1e-12 and numerator > 1e-12:
            target = float(np.clip(numerator / denominator, 0.1, 10.0))
            self.scale = 0.5 * self.scale + 0.5 * target
        gate = metrics["quality_gate"]
        passed = (
            native >= 150000
            and gate is not None
            and (
                gate["normalized_latent_mse_5"] <= 0.5
                and gate["normalized_reward_mse_5"] <= 1.0
            )
        )
        self.passes = self.passes + 1 if passed else 0
        # Two consecutive passes to advance; immediately retreat after failure.
        self.reliable = self.passes >= 2

    def controls(self, native):
        return dict(
            actor_enabled=native >= 50000,
            transition_only=native >= 50000 and native % 50000 >= 10000,
            max_gap=(
                0.1
                if native < 50000
                else 0.25 if native < 100000 else 0.5 if native < 150000 else 1.0
            ),
            sample_steps=1 if self.reliable else 4,
            imag_horizon=15 if self.reliable else 5,
            dyn_scale=self.scale,
        )


def tree_digest(params, prefixes=None):
    import jax

    result = hashlib.sha256()
    for key, value in sorted(params.items()):
        if prefixes is not None and not key.startswith(prefixes):
            continue
        value = np.asarray(jax.device_get(value))
        result.update(key.encode())
        result.update(str((value.shape, value.dtype)).encode())
        result.update(value.tobytes())
    return result.hexdigest()


@contextmanager
def installed_runner(root, arm, seed, preflight=False):
    """Hooks are scoped to this cell's isolated process, not source modifications."""
    protocol = json.loads((Path(root) / "protocol.json").read_text())
    base._imports(protocol)
    from dreamerv3 import agent as upstream
    from . import dreamer_ablation_dynamics as original
    from . import staged_dynamics as dynamics
    from .staged_diagnostics import evaluate_batch
    import jax

    old_install, old_agent, old_json, old_evaluate, old_config = (
        original.install,
        upstream.Agent,
        base._json,
        base._evaluate,
        base._config,
    )
    directory = (
        Path(root) / ("preflight" if preflight else "cells") / arm / f"seed_{seed}"
    )
    conditional = protocol.get("schema") == "imf-conditional-one-seed-v1"
    joint = protocol.get("schema") == "imf-joint-one-seed-v1" or conditional
    if joint:
        from .joint_diagnostics import JointSchedule, retained_views, evaluate_controls
    schedule = JointSchedule() if joint else Schedule()
    if conditional:
        from . import conditional_dynamics
        from . import conditional_schedule as repaired_schedule

        schedule = repaired_schedule.ConditionalSchedule()

        def repaired_config(*args, **kwargs):
            return old_config(*args, **kwargs).update(
                {"agent.repval_grad": False, "agent.reward_grad": True}
            )

        base._config = repaired_config
    state = dict(
        native=0,
        updates=0,
        controls=[],
        diagnostics=[],
        freeze_checks=[],
        protocol=protocol["schema"],
    )
    if joint:
        state["retained_batches"] = []
        state["coverage_diagnostics"] = []
        state["gate_interpretation"] = (
            "diagnostics only; fixed controls never consume gate outcomes"
            if conditional
            else "legacy scheduling heuristic; not a control certificate"
        )
    if conditional:
        state["metrics_snapshots"] = []
    reference = batch = None
    recent = None
    next_diagnostic = 50000

    def diagnostic(agent, force=False):
        nonlocal reference, next_diagnostic
        if batch is None or (not force and state["native"] < next_diagnostic):
            return
        name = f"diagnostic_{state['native']}_{state['updates']}.json"
        if any(r["path"] == name for r in state["diagnostics"]):
            return
        before = tree_digest(agent.params)
        # Explicit read-only diagnostic transfers; upstream disallows accidental ones.
        with jax.transfer_guard("allow"):
            metrics, reference = evaluate_batch(
                agent.model, agent.params, batch, seed + 900000, reference
            )
            if joint:
                if recent is None:
                    raise ValueError("Missing contemporaneous replay sample")
                batch_name = f"replay_sample_{state['native']}_{state['updates']}.npz"
                with (directory / batch_name).open("xb") as out:
                    np.savez_compressed(out, **recent)
                state["retained_batches"].append(
                    dict(
                        path=batch_name,
                        sha256=hashlib.sha256(
                            (directory / batch_name).read_bytes()
                        ).hexdigest(),
                        native_steps=state["native"],
                        learner_updates=state["updates"],
                        positive_rewards=int(np.count_nonzero(recent["reward"] > 0)),
                        total_rewards=int(recent["reward"].size),
                        sampling="current learner minibatch, before this update; not held-out",
                    )
                )
                views = (
                    repaired_schedule.retained_views if conditional else retained_views
                )(recent)
                evidence = {
                    key: evaluate_controls(
                        agent.model,
                        agent.params,
                        value,
                        seed + 910000,
                        steps=(
                            int(
                                np.asarray(
                                    jax.device_get(
                                        agent.params["staged_sample_steps/value"]
                                    )
                                )
                            )
                            if conditional
                            else 1
                        ),
                    )
                    for key, value in views.items()
                }
                clocks = {
                    key: int(np.asarray(jax.device_get(value)))
                    for key, value in agent.params.items()
                    if key.startswith("opt/state/") and key.endswith("/3/count")
                }
                coverage_name = f"coverage_{state['native']}_{state['updates']}.json"
                old_json(
                    directory / coverage_name,
                    dict(
                        native_steps=state["native"],
                        learner_updates=state["updates"],
                        batch_path=batch_name,
                        views=evidence,
                        optimizer_clocks=clocks,
                        learner_unchanged=tree_digest(agent.params) == before,
                    ),
                )
                state["coverage_diagnostics"].append(
                    dict(
                        path=coverage_name,
                        sha256=hashlib.sha256(
                            (directory / coverage_name).read_bytes()
                        ).hexdigest(),
                    )
                )
        if tree_digest(agent.params) != before:
            raise AssertionError("Diagnostics modified learner state")
        schedule.observe(state["native"], metrics)
        payload = dict(
            native_steps=state["native"],
            learner_updates=state["updates"],
            metrics=metrics,
            next_controls=schedule.controls(state["native"]),
            learner_unchanged=True,
        )
        old_json(directory / name, payload)
        state["diagnostics"].append(
            dict(
                path=name,
                sha256=hashlib.sha256((directory / name).read_bytes()).hexdigest(),
            )
        )
        next_diagnostic = (state["native"] // 50000 + 1) * 50000

    def factory(*args, **kwargs):
        agent = (
            conditional_dynamics.ConditionalAgent
            if conditional
            else dynamics.StagedAgent
        )(*args, **kwargs)
        train, policy = agent.train, agent.policy

        def tracked_policy(carry, obs, mode="train"):
            if mode == "train":
                state["native"] += int(np.count_nonzero(~obs["is_first"])) * 2
            return policy(carry, obs, mode=mode)

        def tracked_train(carry, data):
            nonlocal batch, recent
            if joint and (
                recent is None or state["native"] >= next_diagnostic or preflight
            ):
                fields = set(agent.model.obs_space) | set(agent.model.act_space)
                recent = {
                    k: np.asarray(jax.device_get(v)).copy()
                    for k, v in data.items()
                    if k in fields
                }
            if batch is None:
                fields = set(agent.model.obs_space) | set(agent.model.act_space)
                batch = {
                    k: np.asarray(jax.device_get(v))[:2, :16].copy()
                    for k, v in data.items()
                    if k in fields
                }
                np.savez(directory / "diagnostic_batch.npz", **batch)
                state["batch_sha256"] = hashlib.sha256(
                    (directory / "diagnostic_batch.npz").read_bytes()
                ).hexdigest()
            controls = schedule.controls(state["native"])
            if preflight and not conditional:
                # Exercise all phases on full geometry, without spending 150k steps.
                controls = schedule.controls(
                    [0, 60000, 100000, 160000][state["updates"] % 4]
                )
                if state["updates"] >= 4:
                    controls.update(sample_steps=1, imag_horizon=15)
            changed = (
                not state["controls"] or controls != state["controls"][-1]["values"]
            )
            audit = changed or preflight
            frozen_prefixes = ()
            if not controls["actor_enabled"]:
                frozen_prefixes += (
                    "pol/",
                    "val/",
                    "slowval/",
                    "retnorm/",
                    "valnorm/",
                    "advnorm/",
                )
            if controls["transition_only"]:
                frozen_prefixes += ("enc/", "dec/", "rew/", "con/")
                frozen_prefixes += tuple(
                    k
                    for k in agent.params
                    if k.startswith("dyn/")
                    and dynamics.parameter_group(k) == "representation"
                )
            frozen = tree_digest(agent.params, frozen_prefixes) if audit else None
            if changed:
                dynamics.set_controls(agent, controls)
            if changed:
                state["controls"].append(
                    dict(
                        native_steps=state["native"],
                        update=state["updates"],
                        values=controls,
                    )
                )
            result = train(carry, data)
            state["updates"] += 1
            if audit:
                if tree_digest(agent.params, frozen_prefixes) != frozen:
                    raise AssertionError("Inactive model/actor state changed")
                state["freeze_checks"].append(
                    dict(
                        update=state["updates"],
                        prefixes=list(frozen_prefixes),
                        passed=True,
                    )
                )
            if not preflight:
                diagnostic(agent)
            return result

        agent.train, agent.policy = tracked_train, tracked_policy
        return agent

    def install(which):
        if which != arm:
            raise ValueError("Arm mismatch")
        (conditional_dynamics.install if conditional else dynamics.install)(which)
        upstream.Agent = factory

    def evaluate(agent, *args, **kwargs):
        # Final check measures the final parameters, even if collection just ended.
        diagnostic(agent, force=True)
        return old_evaluate(agent, *args, **kwargs)

    def write(path, value):
        if Path(path).name in ("progress.json", "complete.json"):
            value = dict(value, staged=state.copy())
        if conditional and Path(path).name == "progress.json":
            # Preserve snapshots instead of leaving only the final overwritten metrics.
            snapshot = directory / f"metrics_{state['native']}_{state['updates']}.json"
            if not snapshot.exists():
                old_json(snapshot, value)
            evaluations = value.get("evaluations", [])
            if evaluations and evaluations[-1]["native_steps"] == state["native"]:
                name = f"metrics_evaluation_{state['native']}.json"
                if not (directory / name).exists():
                    old_json(directory / name, value["metrics"])
                    state["metrics_snapshots"].append(
                        dict(
                            path=name,
                            sha256=hashlib.sha256(
                                (directory / name).read_bytes()
                            ).hexdigest(),
                            native_steps=state["native"],
                        )
                    )
        if conditional and Path(path).name == "complete.json":
            value["staged"]["config_sha256"] = hashlib.sha256(
                (directory / "config.yaml").read_bytes()
            ).hexdigest()
        old_json(path, value)

    original.install, base._json, base._evaluate = install, write, evaluate
    try:
        yield
    finally:
        original.install, upstream.Agent, base._json, base._evaluate, base._config = (
            old_install,
            old_agent,
            old_json,
            old_evaluate,
            old_config,
        )


def run_cell(root, arm, seed, preflight=False):
    if arm not in ("gaussian", "imf"):
        raise ValueError("Only matched continuous arms are registered")
    with installed_runner(root, arm, seed, preflight):
        base.run_cell(root, arm, seed, preflight)
    directory = (
        Path(root) / ("preflight" if preflight else "cells") / arm / f"seed_{seed}"
    )
    return json.loads((directory / "complete.json").read_text())
