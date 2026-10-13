"""Frozen-feature capacity sweep. No agent, policy or simulator is constructed."""

import argparse
from pathlib import Path
import os
import time
import jax
import numpy as np
from .parallel_collection import _write_json, _write_npz
from .parallel_runner import load_npz
from .prior_correction_study import load
from .reward_readout_study import (
    read,
    sha,
    commit,
    SOURCE,
    validate_data,
    equal,
    authenticate_parent,
)
from .reward_diagnosis import seal, check_seal, indices
from .reward_diagnosis_data import GROUPS, group_features
from .reward_diagnosis_metrics import summary, boundary_bins, nll
from .reward_probe_metrics import _EpisodeBootstrap
from . import reward_capacity_fit as fitting

DEPENDENCY_COMMIT = "e1fc21daf19c989456335316a1dce7de8bc60499"
DEPENDENCY_REPORT = "f1b7e9af472856c069a8e4ea7d2c29b2d6204fdf487d5c2933dd32a09bc4b689"
PROTOCOL = dict(
    schema="frozen-reward-capacity-v1",
    widths=[64, 128, 512],
    depth=2,
    groups=list(GROUPS),
    seeds=[0, 1, 2],
    updates=5000,
    batch_size=256,
    learning_rate=0.0003,
    minibatch_seed=6211,
    validation_period=250,
    bootstrap_reps=2000,
    selection="validation NLL; earliest tie",
    additional_simulator_steps=0,
    statistical_unit="episode; average errors over readout seeds, not predictions",
    limitations=[
        "Reused test episodes: exploratory, not confirmatory",
        "One frozen world-model seed; readout seeds are not world-model replications",
        "Fixed updates, not equal FLOPs or guaranteed equal convergence",
        "Pointwise bootstrap intervals, not multiple-comparison adjusted",
        "Persistent readout error does not prove missing information",
        "No online-return or backbone-capacity claim",
    ],
)


def protocol(preflight):
    return (
        dict(PROTOCOL, updates=2, validation_period=1, seeds=[0])
        if preflight
        else dict(PROTOCOL)
    )


def authenticate(path):
    path = Path(path)
    v = read(path / "verified.json")
    if (
        v.get("exact_replay") is not True
        or v["report_sha256"] != DEPENDENCY_REPORT
        or sha(path / "report.json") != DEPENDENCY_REPORT
        or v["evaluated_sha256"] != sha(path / "evaluated.json")
    ):
        raise ValueError("unverified or wrong diagnosis dependency")
    check_seal(path, "evaluated")
    m = read(path / "manifest.json")
    if m["source_commit"] != DEPENDENCY_COMMIT or m["preflight"]:
        raise ValueError("wrong diagnosis source")
    _, dataset = authenticate_parent(Path(m["parent"]))
    if str(dataset) != m["dataset"] or sha(dataset) != m["dataset_sha256"]:
        raise ValueError("retained dataset differs")
    for file, digest in m["raw_bindings"].items():
        if sha(file) != digest:
            raise ValueError("retained raw trajectory changed")
    data = load_npz(dataset)
    validate_data(data)
    inputs = load_npz(path / "inputs.npz")
    if (
        inputs["previous"].shape != data["observations"].shape
        or inputs["distance"].shape != data["rewards"].shape
    ):
        raise ValueError("invalid retained input shapes")
    binding = dict(
        path=str(path.resolve()),
        marker_sha256=sha(path / "verified.json"),
        report_sha256=sha(path / "report.json"),
        manifest_sha256=sha(path / "manifest.json"),
        dataset_sha256=m["dataset_sha256"],
        inputs_sha256=sha(path / "inputs.npz"),
        frozen_digest=m["frozen_digest"],
    )
    return data, inputs, m["geometry"], binding


def initialize(output, dependency, preflight):
    source = commit(SOURCE)
    context = authenticate(dependency)
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise ValueError("GPU execution required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "manifest.json",
        dict(
            source_commit=source,
            protocol=protocol(preflight),
            preflight=preflight,
            dependency=context[3],
            geometry=context[2],
            devices=[str(d) for d in jax.devices()],
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            initialized_unix=time.time(),
        ),
    )
    return context


def setup(output):
    m = read(Path(output) / "manifest.json")
    if m["source_commit"] != commit(SOURCE) or m["protocol"] != protocol(
        m["preflight"]
    ):
        raise ValueError("source or protocol changed")
    context = authenticate(m["dependency"]["path"])
    if context[3] != m["dependency"] or context[2] != m["geometry"]:
        raise ValueError("dependency changed")
    return context


def features(data, inputs, group):
    return group_features(
        group,
        data["targets"],
        data["observations"],
        inputs["previous"],
        data["actions"],
        data["targets"].shape[-1] - 512,
    )


def cells(p):
    for group in p["groups"]:
        for width in p["widths"]:
            for seed in p["seeds"]:
                yield f"{group}-w{width}-s{seed}", group, width, seed


def fit(output, context):
    output = Path(output)
    data, inputs, _, _ = context
    m = read(output / "manifest.json")
    train, val = [indices(data, s, m["preflight"]) for s in (0, 1)]
    for name, group, width, seed in cells(m["protocol"]):
        x = features(data, inputs, group)
        fitting.fit(
            output / "heads" / name,
            x[train],
            data["rewards"][train],
            x[val],
            data["rewards"][val],
            width,
            seed,
            m["protocol"],
        )
    seal(output, "fitted")
    print("REWARD_CAPACITY_FITTED", flush=True)


def generate(output, context, verify_candidates=False):
    output = Path(output)
    check_seal(output, "fitted")
    data, inputs, _, _ = context
    m = read(output / "manifest.json")
    ix = {
        label: indices(data, s, m["preflight"])
        for s, label in enumerate(("train", "validation", "test"))
    }
    raw = {"indices__" + label: rows for label, rows in ix.items()}
    selections = {}
    for name, group, width, seed in cells(m["protocol"]):
        x = features(data, inputs, group)
        directory = output / "heads" / name
        record = read(directory / "selection.json")
        if verify_candidates:
            fitting.verify_fit(
                directory,
                x[ix["train"]],
                data["rewards"][ix["train"]],
                x[ix["validation"]],
                data["rewards"][ix["validation"]],
                width,
                seed,
                m["protocol"],
            )
        norm = load_npz(directory / "normalization.npz")
        selections[name] = record
        for checkpoint, step in (
            ("selected", record["selected"]["update"]),
            ("final", m["protocol"]["updates"]),
        ):
            params = load(directory / f"checkpoint-{step:05d}.pkl")
            for label, rows in ix.items():
                raw[f"{name}__{checkpoint}__{label}"] = fitting.predict(
                    params, norm, x[rows]
                )
    return raw, selections


def error_arrays(probs, y):
    pred = probs @ np.arange(3)
    return dict(
        mse=(pred - y) ** 2,
        nll=nll(probs, y),
        brier=((probs - np.eye(3)[y.astype(int)]) ** 2).sum(-1),
        cumulative_reward_mse=(pred.sum(-1) - y.sum(-1)) ** 2,
    )


def report(raw, records, data, inputs, radius, p):
    boot = _EpisodeBootstrap(p["bootstrap_reps"])
    individual, averages, contrasts, gaps = {}, {}, {}, {}
    bins = boundary_bins(inputs["distance"], radius)
    for name, group, width, seed in cells(p):
        for checkpoint in ("selected", "final"):
            for split in ("train", "validation", "test"):
                key = f"{name}__{checkpoint}__{split}"
                rows = raw["indices__" + split]
                probs = raw[key]
                individual[key] = summary(
                    probs @ np.arange(3),
                    data["rewards"][rows],
                    data["episode"][rows],
                    probabilities=probs,
                    boundary=bins[rows],
                    reps=p["bootstrap_reps"],
                )

    def aggregate(group, width, checkpoint, split):
        rows = raw["indices__" + split]
        byseed = [
            error_arrays(
                raw[f"{group}-w{width}-s{s}__{checkpoint}__{split}"],
                data["rewards"][rows],
            )
            for s in p["seeds"]
        ]
        return {
            metric: np.stack([e[metric] for e in byseed]).mean(0)
            for metric in byseed[0]
        }

    def stats(errors, rows):
        ep = data["episode"][rows]
        return {
            key: boot.stat(
                value.reshape(-1),
                (
                    ep
                    if value.ndim == 1
                    else np.broadcast_to(ep[:, None], value.shape).reshape(-1)
                ),
            )
            for key, value in errors.items()
        }

    for group in p["groups"]:
        for width in p["widths"]:
            for checkpoint in ("selected", "final"):
                for split in ("train", "validation", "test"):
                    rows = raw["indices__" + split]
                    errors = aggregate(group, width, checkpoint, split)
                    base = aggregate(group, 128, checkpoint, split)
                    key = f"{group}-w{width}__{checkpoint}__{split}"
                    averages[key] = stats(errors, rows)
                    delta = stats({k: errors[k] - base[k] for k in errors}, rows)
                    baseline = stats(base, rows)["mse"]["mean"]
                    reduction = (
                        None
                        if baseline == 0
                        else 1 - averages[key]["mse"]["mean"] / baseline
                    )
                    positive = len(
                        np.unique(
                            data["episode"][rows][(data["rewards"][rows] > 0).any(1)]
                        )
                    )
                    contrasts[key] = dict(
                        paired_delta=delta,
                        relative_mse_reduction=reduction,
                        positive_episodes=positive,
                        material_test_improvement=bool(
                            split == "test"
                            and reduction is not None
                            and reduction >= 0.1
                            and delta["mse"]["ci95"][1] is not None
                            and delta["mse"]["ci95"][1] < 0
                            and positive >= 5
                        ),
                    )
                    if group in ("h", "z", "hz"):
                        obs = aggregate("observation", width, checkpoint, split)
                        obsbase = aggregate("observation", 128, checkpoint, split)
                        gaps[key] = dict(
                            vs_same_width_observation=stats(
                                {k: errors[k] - obs[k] for k in errors}, rows
                            ),
                            change_vs_128_gap=stats(
                                {
                                    k: (errors[k] - obs[k]) - (base[k] - obsbase[k])
                                    for k in errors
                                },
                                rows,
                            ),
                        )
    return dict(
        protocol=p,
        cells=records,
        individual=individual,
        seed_average_errors=averages,
        width_vs_128=contrasts,
        latent_observation_gaps=gaps,
        additional_simulator_steps=0,
        test_episodes=np.unique(data["episode"][raw["indices__test"]]).tolist(),
    )


def evaluate(output, context):
    output = Path(output)
    raw, records = generate(output, context)
    m = read(output / "manifest.json")
    _write_npz(output / "predictions.npz", raw)
    _write_json(
        output / "report.json",
        report(
            raw, records, context[0], context[1], context[2]["radius"], m["protocol"]
        ),
    )
    seal(output, "evaluated")
    print("REWARD_CAPACITY_EVALUATED_REPLAY_REQUIRED", flush=True)


def verify(output, context):
    output = Path(output)
    check_seal(output, "evaluated")
    raw, records = generate(output, context, verify_candidates=True)
    saved = load_npz(output / "predictions.npz")
    if set(raw) != set(saved):
        raise ValueError("prediction inventory differs")
    for key in raw:
        equal(raw[key], saved[key], "capacity prediction replay " + key)
    m = read(output / "manifest.json")
    result = report(
        raw, records, context[0], context[1], context[2]["radius"], m["protocol"]
    )
    if result != read(output / "report.json"):
        raise ValueError("independent capacity metrics differ")
    # Re-authenticate input evidence after all computation, not just before it.
    if authenticate(m["dependency"]["path"])[3] != m["dependency"]:
        raise ValueError("input dependency changed during computation")
    _write_json(
        output / "verified.json",
        dict(
            exact_replay=True,
            report_sha256=sha(output / "report.json"),
            evaluated_sha256=sha(output / "evaluated.json"),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            finished_unix=time.time(),
        ),
    )
    print("REWARD_CAPACITY_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "fit", "evaluate", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency", type=Path)
    a = parser.parse_args()
    if a.stage in ("preflight", "fit"):
        if a.dependency is None:
            parser.error("dependency required")
        context = initialize(a.output, a.dependency, a.stage == "preflight")
        fit(a.output, context)
        if a.stage == "preflight":
            evaluate(a.output, context)
    else:
        context = setup(a.output)
        (evaluate if a.stage == "evaluate" else verify)(a.output, context)


if __name__ == "__main__":
    main()
