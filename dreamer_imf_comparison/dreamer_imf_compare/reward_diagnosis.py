"""Matched frozen-feature reward diagnosis; independent replay, zero simulator steps."""

import argparse
import os
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .parallel_collection import _write_json, _write_npz
from .parallel_runner import load_npz, finite
from .reward_readout_study import (
    authenticate_parent,
    setup_teacher,
    sha,
    read,
    commit,
    SOURCE,
    equal,
    validate_data,
)
from .prior_correction import FrozenPrior, rollout
from .prior_correction_study import load
from .reward_diagnosis_data import (
    GROUPS,
    group_features,
    retained_inputs,
    geometry,
    validate_endpoint,
    posterior_draws,
)
from .reward_diagnosis_fit import fit_head, Head
from .reward_diagnosis_metrics import boundary_bins, summary, action_metrics, nll
from .reward_probe_metrics import _EpisodeBootstrap

PROTOCOL = dict(
    schema="matched-reward-diagnosis-v1",
    groups=list(GROUPS),
    seeds=[0, 1, 2],
    updates=5000,
    batch_size=256,
    learning_rate=0.0003,
    hidden=[128, 128],
    validation_period=250,
    selection="validation categorical NLL; earliest ties",
    minibatch_seed=6211,
    particles=32,
    posterior_draws=8,
    bootstrap_reps=2000,
    additional_simulator_steps=0,
    correction_commit="d26e93c4b83ecaa28abe8518d46ca6139d8c6773",
    correction_update=500,
    endpoint_bins=[0.8, 1.0, 1.2],
    limitations=[
        "Previously inspected test episodes: exploratory, not confirmation",
        "One frozen world-model seed, not three independent world models",
        "Native intermediate observation unavailable; endpoint distance is not a summed reward label",
        "Readout failure cannot establish information absence",
        "Probability metrics for the original scalar reward expectation are unavailable",
    ],
)


def config(preflight):
    return (
        dict(PROTOCOL, updates=2, validation_period=1, seeds=[0], posterior_draws=1)
        if preflight
        else dict(PROTOCOL)
    )


def correction_binding(path):
    path = Path(path)
    marker = read(path / "verified.json")
    created = read(path / "created.json")
    if (
        marker.get("exact_replay") is not True
        or marker["created_sha256"] != sha(path / "created.json")
        or marker["report_sha256"] != sha(path / "report.json")
    ):
        raise ValueError("correction not verified")
    for name, digest in created["artifacts"].items():
        if Path(name).name != name or sha(path / name) != digest:
            raise ValueError("correction artifact changed")
    m = read(path / "manifest.json")
    s = read(path / "selection.json")["selected"]
    if (
        m["source_commit"] != PROTOCOL["correction_commit"]
        or m["preflight"]
        or s["update"] != PROTOCOL["correction_update"]
    ):
        raise ValueError("wrong frozen correction")
    return dict(
        path=str(path.resolve()),
        marker_sha256=sha(path / "verified.json"),
        created_sha256=sha(path / "created.json"),
        dataset_sha256=m["dataset_sha256"],
        frozen_digest=m["frozen_digest"],
        selected=s,
    )


def artifacts(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def seal(root, stage):
    _write_json(
        Path(root) / f"{stage}.json", dict(stage=stage, artifacts=artifacts(root))
    )


def check_seal(root, stage):
    root = Path(root)
    value = read(root / f"{stage}.json")
    if value.get("stage") != stage or not value.get("artifacts"):
        raise ValueError("invalid stage marker")
    for name, digest in value["artifacts"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError("changed or invalid artifact: " + name)
    return value


def initialize(parent, correction, output, preflight):
    source = commit(SOURCE)
    parent_manifest, dataset = authenticate_parent(Path(parent))
    binding = correction_binding(correction)
    if sha(dataset) != binding["dataset_sha256"]:
        raise ValueError("correction and diagnosis datasets differ")
    teacher = setup_teacher(parent_manifest)
    if teacher.frozen_digest() != binding["frozen_digest"]:
        raise ValueError("correction parent differs")
    data = load_npz(dataset)
    validate_data(data)
    previous, distance, raw_bindings = retained_inputs(data, dataset.parent)
    geom = geometry()
    validate_endpoint(distance, data["rewards"], geom["radius"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_npz(output / "inputs.npz", dict(previous=previous, distance=distance))
    _write_json(
        output / "manifest.json",
        dict(
            source_commit=source,
            parent=str(Path(parent).resolve()),
            dataset=str(dataset),
            dataset_sha256=sha(dataset),
            correction=binding,
            protocol=config(preflight),
            preflight=preflight,
            geometry=geom,
            raw_bindings=raw_bindings,
            frozen_digest=teacher.frozen_digest(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            devices=[str(d) for d in jax.devices()],
            initialized_unix=time.time(),
        ),
    )
    return teacher, data, previous, distance, dataset.parent


def setup(output):
    m = read(Path(output) / "manifest.json")
    if m["source_commit"] != commit(SOURCE) or m["protocol"] != config(m["preflight"]):
        raise ValueError("source/protocol mismatch")
    pm, dataset = authenticate_parent(Path(m["parent"]))
    if str(dataset) != m["dataset"] or sha(dataset) != m["dataset_sha256"]:
        raise ValueError("dataset changed")
    if correction_binding(m["correction"]["path"]) != m["correction"]:
        raise ValueError("correction dependency changed")
    for path, digest in m["raw_bindings"].items():
        if sha(path) != digest:
            raise ValueError("raw input changed")
    teacher = setup_teacher(pm)
    if teacher.frozen_digest() != m["frozen_digest"] or geometry() != m["geometry"]:
        raise ValueError("frozen environment/model changed")
    data = load_npz(dataset)
    validate_data(data)
    previous, distance, bindings = retained_inputs(data, dataset.parent)
    retained = load_npz(Path(output) / "inputs.npz")
    equal(previous, retained["previous"], "history inputs")
    equal(distance, retained["distance"], "distance")
    if bindings != m["raw_bindings"]:
        raise ValueError("raw binding mismatch")
    validate_endpoint(distance, data["rewards"], m["geometry"]["radius"])
    return teacher, data, previous, distance, dataset.parent


def indices(data, split, preflight):
    ix = np.flatnonzero(data["split"] == split)
    if preflight:
        ix = ix[data["episode"][ix] == data["episode"][ix].min()]
    return ix


def fit(output, context):
    output = Path(output)
    teacher, data, previous, _, _ = context
    m = read(output / "manifest.json")
    p = m["protocol"]
    train, val = [indices(data, s, m["preflight"]) for s in (0, 1)]
    for group in GROUPS:
        x = group_features(
            group,
            data["targets"],
            data["observations"],
            previous,
            data["actions"],
            teacher.deter,
        )
        for seed in p["seeds"]:
            fit_head(
                output / "heads" / f"{group}-{seed}",
                x[train],
                data["rewards"][train],
                x[val],
                data["rewards"][val],
                seed,
                p,
            )
    if teacher.frozen_digest() != m["frozen_digest"]:
        raise ValueError("parent changed during fit")
    seal(output, "fitted")
    print("REWARD_DIAGNOSIS_FIT_VERIFIED", flush=True)


def check_fit(output, context):
    """Recompute normalization and validation NLL for EVERY saved candidate."""
    output = Path(output)
    check_seal(output, "fitted")
    teacher, data, previous, _, _ = context
    m = read(output / "manifest.json")
    p = m["protocol"]
    tr, va = [indices(data, s, m["preflight"]) for s in (0, 1)]
    from .reward_readout_study import mlp_logits

    for group in GROUPS:
        x = group_features(
            group,
            data["targets"],
            data["observations"],
            previous,
            data["actions"],
            teacher.deter,
        )
        flat = x[tr].reshape(-1, x.shape[-1])
        mean = flat.mean(0, dtype=np.float64).astype(np.float32)
        std = np.maximum(flat.std(0, dtype=np.float64), 0.01).astype(np.float32)
        for seed in p["seeds"]:
            directory = output / "heads" / f"{group}-{seed}"
            norm = load_npz(directory / "normalization.npz")
            sel = read(directory / "selection.json")
            equal(mean, norm["mean"], "train-only mean")
            equal(std, norm["std"], "train-only std")
            expected = list(
                range(p["validation_period"], p["updates"] + 1, p["validation_period"])
            )
            if (
                [c["update"] for c in sel["candidates"]] != expected
                or sel["seed"] != seed
                or sel["minibatch_seed"] != p["minibatch_seed"]
            ):
                raise ValueError("invalid validation candidate set")
            for candidate in sel["candidates"]:
                step = candidate["update"]
                name = f"checkpoint-{step:05d}.pkl"
                if candidate["checkpoint"] != name:
                    raise ValueError("invalid checkpoint path")
                params = load(directory / name)
                fn = jax.jit(lambda w, xx: jax.nn.softmax(mlp_logits(w, xx), -1))
                pred = np.asarray(fn(params, jnp.asarray((x[va] - mean) / std)))
                equal(
                    pred,
                    load_npz(directory / f"validation-{step:05d}.npz")["probabilities"],
                    "validation replay",
                )
                if float(nll(pred, data["rewards"][va]).mean()) != candidate["nll"]:
                    raise ValueError("validation score mismatch")
            if (
                min(sel["candidates"], key=lambda c: (c["nll"], c["update"]))
                != sel["selected"]
            ):
                raise ValueError("invalid selection")


def generate(output, context):
    output = Path(output)
    teacher, data, previous, _, location = context
    m = read(output / "manifest.json")
    p = m["protocol"]
    ix = indices(data, 2, m["preflight"])
    heads = {
        f"{g}-{s}": Head(output / "heads" / f"{g}-{s}")
        for g in GROUPS
        for s in p["seeds"]
    }
    raw = {"indices": ix}

    def apply(features, group):
        return group_features(group, features, deter=teacher.deter)

    original = np.asarray(teacher.decode(jnp.asarray(data["targets"][ix]))[1])
    raw["real__original"] = original
    for name, head in heads.items():
        group = name.rsplit("-", 1)[0]
        x = group_features(
            group,
            data["targets"][ix],
            data["observations"][ix],
            previous[ix],
            data["actions"][ix],
            teacher.deter,
        )
        raw["real__" + name] = head(x)
    train = indices(data, 0, m["preflight"])
    frequency = np.bincount(
        data["rewards"][train].astype(int).reshape(-1), minlength=3
    ).astype(float)
    frequency /= frequency.sum()
    for name, probs in (("frequency", frequency), ("zero", np.array([1.0, 0.0, 0.0]))):
        raw["real__" + name] = np.broadcast_to(probs, (*original.shape, 3)).copy()
    adapter = FrozenPrior(teacher)
    corrpath = Path(m["correction"]["path"])
    params = load(corrpath / m["correction"]["selected"]["checkpoint"])
    norm = load_npz(corrpath / "normalization.npz")
    latent = {n: h for n, h in heads.items() if n.rsplit("-", 1)[0] in ("h", "z", "hz")}
    digests = {}
    from .parallel_frozen import parameter_digest

    for path in ("gaussian", "corrected", "original_imf"):
        chunks = {n: [] for n in ("original", *latent)}
        digests[path] = []
        fn = jax.jit(
            lambda s, a, k: rollout(
                adapter,
                params,
                norm,
                s,
                a,
                k,
                particles=p["particles"],
                bypass=path == "gaussian",
            )
        )
        for at in range(0, len(ix), 8):
            rows = ix[at : at + 8]
            if path == "original_imf":
                f = teacher.rollout(
                    data["start"][rows],
                    data["actions"][rows],
                    9901 + at,
                    particles=p["particles"],
                    flow_steps=4,
                )
            else:
                f = fn(
                    jnp.asarray(data["start"][rows]),
                    jnp.asarray(data["actions"][rows]),
                    jax.random.fold_in(jax.random.PRNGKey(8801), at),
                )
            finite(f)
            f = np.asarray(f)
            digests[path].append(parameter_digest({"features": f}))
            chunks["original"].append(np.asarray(teacher.decode(jnp.asarray(f))[1]))
            for name, head in latent.items():
                chunks[name].append(head(apply(f, name.rsplit("-", 1)[0])))
        for name, values in chunks.items():
            raw[path + "__" + name] = np.concatenate(values)
    positions = {int(row): j for j, row in enumerate(ix)}
    for name in ("original", *latent):
        shape = (p["posterior_draws"], len(ix), 15) + (
            () if name == "original" else (3,)
        )
        raw["draw__" + name] = np.full(shape, np.nan, np.float32)
    for draw, rows, f in posterior_draws(
        teacher, data, location, ix, p["posterior_draws"]
    ):
        dest = [positions[int(i)] for i in rows]
        raw["draw__original"][draw, dest] = np.asarray(
            teacher.decode(jnp.asarray(f))[1]
        )
        for name, head in latent.items():
            raw["draw__" + name][draw, dest] = head(apply(f, name.rsplit("-", 1)[0]))
    finite(raw)
    adapter.assert_frozen()
    return raw, digests


def report(raw, data, distance, radius, reps):
    ix = raw["indices"]
    y = data["rewards"][ix]
    ep = data["episode"][ix]
    bins = boundary_bins(distance[ix], radius)
    base = raw["real__original"]
    result = {}
    boot = _EpisodeBootstrap(reps)
    for key, value in raw.items():
        if key == "indices":
            continue
        path, name = key.split("__")
        categorical = name != "original"
        if path == "draw":
            pred = value @ np.arange(3) if categorical else value
            epsteps = np.broadcast_to(ep[:, None], y.shape).reshape(-1)
            result[key] = dict(
                mean_individual_draw_mse=boot.stat(
                    ((pred - y) ** 2).mean(0).reshape(-1), epsteps
                ),
                averaged_draw_mse=boot.stat(
                    (pred.mean(0) - y).reshape(-1) ** 2, epsteps
                ),
                prediction_variance=boot.stat(pred.var(0).reshape(-1), epsteps),
                mean_individual_draw_return_mse=boot.stat(
                    ((pred.sum(-1) - y.sum(-1)) ** 2).mean(0), ep
                ),
                averaged_draw_return_mse=boot.stat(
                    (pred.mean(0).sum(-1) - y.sum(-1)) ** 2, ep
                ),
                return_variance=boot.stat(pred.sum(-1).var(0), ep),
            )
            continue
        averaged = value.mean(1) if path != "real" else value
        probabilities = averaged if categorical else None
        pred = averaged @ np.arange(3) if categorical else averaged
        baseline = base if path == "real" else raw[path + "__original"].mean(1)
        result[key] = summary(
            pred,
            y,
            ep,
            probabilities=probabilities,
            boundary=bins,
            baseline=baseline,
            reps=reps,
        )
        result[key]["actions"] = action_metrics(
            pred, y, ep, data["anchor"][ix], data["plan"][ix], reps
        )
    # Average errors over initialization seeds, not ensemble their predictions.
    averages = {}
    epsteps = np.broadcast_to(ep[:, None], y.shape).reshape(-1)
    for group in GROUPS:
        predictions = np.stack(
            [
                v @ np.arange(3)
                for k, v in raw.items()
                if k.startswith("real__" + group + "-")
            ]
        )
        err = ((predictions - y) ** 2).mean(0)
        stat = boot.stat(err.reshape(-1), epsteps)
        delta = boot.stat((err - (base - y) ** 2).reshape(-1), epsteps)
        bm = boot.stat(((base - y) ** 2).reshape(-1), epsteps)["mean"]
        reduction = None if bm == 0 else 1 - stat["mean"] / bm
        averages[group] = dict(
            mse=stat,
            paired_delta=delta,
            relative_reduction=reduction,
            cumulative_reward_mse=boot.stat(
                ((predictions.sum(-1) - y.sum(-1)) ** 2).mean(0), ep
            ),
            material_exploratory_improvement=bool(
                reduction is not None
                and reduction >= 0.1
                and delta["ci95"][1] is not None
                and delta["ci95"][1] < 0
                and len(np.unique(ep[(y > 0).any(1)])) >= 5
            ),
        )
    return dict(
        metrics=result,
        seed_average_errors=averages,
        limitations=PROTOCOL["limitations"],
        additional_simulator_steps=0,
        statistical_unit="episode",
        test_episodes=np.unique(ep).tolist(),
    )


def evaluate(output, context):
    output = Path(output)
    check_fit(output, context)
    raw, digests = generate(output, context)
    _write_npz(output / "predictions.npz", raw)
    _write_json(output / "feature_digests.json", digests)
    m = read(output / "manifest.json")
    r = report(
        raw,
        context[1],
        context[3],
        m["geometry"]["radius"],
        m["protocol"]["bootstrap_reps"],
    )
    _write_json(output / "report.json", r)
    seal(output, "evaluated")
    print("REWARD_DIAGNOSIS_EVALUATED_REPLAY_REQUIRED", flush=True)


def verify(output, context):
    output = Path(output)
    check_seal(output, "evaluated")
    check_fit(output, context)
    raw, digests = generate(output, context)
    old = load_npz(output / "predictions.npz")
    if set(raw) != set(old):
        raise ValueError("prediction inventory differs")
    for name, v in raw.items():
        equal(v, old[name], name + " independent prediction replay")
    if digests != read(output / "feature_digests.json"):
        raise ValueError("rollout feature replay differs")
    m = read(output / "manifest.json")
    r = report(
        raw,
        context[1],
        context[3],
        m["geometry"]["radius"],
        m["protocol"]["bootstrap_reps"],
    )
    if r != read(output / "report.json"):
        raise ValueError("independent metrics differ")
    if context[0].frozen_digest() != m["frozen_digest"]:
        raise ValueError("parent mutated")
    for path, digest in m["raw_bindings"].items():
        if sha(path) != digest:
            raise ValueError("retained evidence changed")
    _write_json(
        output / "verified.json",
        dict(
            exact_replay=True,
            evaluated_sha256=sha(output / "evaluated.json"),
            report_sha256=sha(output / "report.json"),
            finished_unix=time.time(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    print("MATCHED_REWARD_DIAGNOSIS_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "fit", "evaluate", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--correction", type=Path)
    args = parser.parse_args()
    if args.stage in ("preflight", "fit"):
        if args.parent is None or args.correction is None:
            parser.error("parent and correction required")
        context = initialize(
            args.parent, args.correction, args.output, args.stage == "preflight"
        )
        fit(args.output, context)
        if args.stage == "preflight":
            evaluate(args.output, context)
    else:
        context = setup(args.output)
        (evaluate if args.stage == "evaluate" else verify)(args.output, context)


if __name__ == "__main__":
    main()
