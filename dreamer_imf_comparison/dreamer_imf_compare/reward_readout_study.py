"""Bounded offline reward probes on immutable retained trajectory evidence.

This module never constructs an environment or invokes policy/training on the
parent agent. New readouts are separate objects. No test-dependent search.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

from .parallel_collection import _write_json, _write_npz, _atomic_artifact
from .parallel_runner import load_npz

SOURCE = Path(__file__).resolve().parents[2]
PROTOCOL = SOURCE / "dreamer_imf_comparison/reward_readout_protocol.json"
BASE = Path("/work2/ci72buri-dreamer_imf_neurips")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def commit(path):
    if subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("dirty source checkout")
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def equal(actual, expected, label):
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError("exact replay differs: " + label)


def validate_data(data):
    required = (
        "start",
        "targets",
        "actions",
        "observations",
        "rewards",
        "episode",
        "split",
        "mode",
        "anchor",
        "plan",
    )
    for name in required:
        value = np.asarray(data[name])
        if not np.isfinite(value).all() or len(value) != 1024:
            raise ValueError("invalid retained field " + name)
    if (
        data["rewards"].shape != (1024, 15)
        or not np.isin(data["rewards"], [0, 1, 2]).all()
    ):
        raise ValueError("invalid AR2 reward labels")
    if data["targets"].shape != (1024, 15, 2560):
        raise ValueError("unexpected posterior features")
    for eid in np.unique(data["episode"]):
        mask = data["episode"] == eid
        if len(np.unique(data["split"][mask])) != 1 or mask.sum() != 16:
            raise ValueError("episode split leakage or missing branch")
        keys = {
            (int(a), int(p)) for a, p in zip(data["anchor"][mask], data["plan"][mask])
        }
        if keys != {(a, p) for a in (100, 200, 300, 400) for p in range(4)}:
            raise ValueError("incomplete branch grid")
    for split, count in enumerate((40, 12, 12)):
        mask = data["split"] == split
        if len(np.unique(data["episode"][mask])) != count:
            raise ValueError("wrong episode split")
        for mode in (0, 1):
            if (
                len(np.unique(data["episode"][mask & (data["mode"] == mode)]))
                != count // 2
            ):
                raise ValueError("unbalanced collection split")


def validation_score(prediction, truth, scale):
    p, t = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    if p.shape != t.shape or p.ndim != 2 or p.shape[1] != 15:
        raise ValueError("invalid validation horizon")
    if (
        not np.isfinite(p).all()
        or not np.isfinite(t).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("nonfinite validation")
    return float(np.mean(((p.sum(-1) - t.sum(-1)) / scale) ** 2))


def first_minimum(records):
    if not records or any(not np.isfinite(x["score"]) for x in records):
        raise ValueError("invalid selection records")
    return min(range(len(records)), key=lambda i: records[i]["score"])


def group_features(data, group):
    if group == "observation":
        return data["observations"]
    if group == "h":
        return data["targets"][..., :2048]
    if group == "z":
        return data["targets"][..., 2048:]
    if group == "hz":
        return data["targets"]
    raise ValueError("unknown feature group")


def init_mlp(key, dimension):
    import jax
    import jax.numpy as jnp

    keys = jax.random.split(key, 3)
    return tuple(
        {"w": jax.random.normal(k, (a, b)) * np.sqrt(2 / a), "b": jnp.zeros(b)}
        for k, a, b in zip(keys, (dimension, 128, 128), (128, 128, 3))
    )


def mlp_logits(params, x):
    import jax

    for i, layer in enumerate(params):
        x = x @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


def mlp_prediction(params, x):
    import jax
    import jax.numpy as jnp

    return jax.nn.softmax(mlp_logits(params, x), axis=-1) @ jnp.arange(
        3, dtype=jnp.float32
    )


def train_mlp(directory, xtrain, ytrain, xval, yval, scale, seed, protocol):
    import jax
    import optax

    directory.mkdir()
    flat = xtrain.reshape(-1, xtrain.shape[-1])
    mean = flat.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(flat.std(0, dtype=np.float64), 0.01).astype(np.float32)
    _write_npz(directory / "normalization.npz", dict(mean=mean, std=std))
    x = jax.device_put((flat - mean) / std)
    y = jax.device_put(ytrain.reshape(-1).astype(np.int32))
    vx = jax.device_put((xval - mean) / std)
    params = init_mlp(jax.random.PRNGKey(seed), x.shape[-1])
    optimizer = optax.adam(protocol["mlp_lr"])
    state = optimizer.init(params)

    @jax.jit
    def update(p, opt, xx, yy):
        loss, grad = jax.value_and_grad(
            lambda w: optax.softmax_cross_entropy_with_integer_labels(
                mlp_logits(w, xx), yy
            ).mean()
        )(p)
        delta, opt = optimizer.update(grad, opt, p)
        return optax.apply_updates(p, delta), opt, loss, optax.global_norm(grad)

    predict = jax.jit(mlp_prediction)
    rng = np.random.default_rng(seed + 6211)
    records, logs = [], []
    begun = time.monotonic()
    for step in range(1, protocol["mlp_updates"] + 1):
        ix = jax.device_put(rng.integers(len(x), size=protocol["mlp_batch"]))
        params, state, loss, grad = update(params, state, x[ix], y[ix])
        if step == 1 or step % 100 == 0:
            values = np.asarray([loss, grad], float)
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite readout training")
            logs.append(
                dict(update=step, loss=float(values[0]), gradient_norm=float(values[1]))
            )
        if step % protocol["mlp_validation_period"]:
            continue
        pred = np.asarray(predict(params, vx))
        score = validation_score(pred, yval, scale)
        name = f"checkpoint_{step:05d}.pkl"
        _atomic_artifact(
            directory / name,
            lambda stream: pickle.dump(jax.device_get(params), stream, protocol=5),
        )
        _write_npz(directory / f"validation_{step:05d}.npz", dict(prediction=pred))
        records.append(dict(update=step, score=score, checkpoint=name))
        print("REWARD_READOUT_PROGRESS", seed, step, score, flush=True)
    selected = records[first_minimum(records)]
    _write_json(
        directory / "selection.json",
        dict(
            selected=selected,
            validations=records,
            training=logs,
            seconds=time.monotonic() - begun,
        ),
    )
    return dict(kind="mlp", name=f"mlp_{seed}", directory=directory.name, **selected)


class Readout:
    def __init__(self, output, record):
        from .reward_probe_metrics import predict_ridge, predict_affine

        self.record = record
        self.kind = record["kind"]
        self.predict_ridge, self.predict_affine = predict_ridge, predict_affine
        if self.kind == "mlp":
            import jax

            loc = output / record["directory"]
            with (loc / record["checkpoint"]).open("rb") as stream:
                self.params = jax.tree.map(jax.device_put, pickle.load(stream))
            self.norm = load_npz(loc / "normalization.npz")
            self.fn = jax.jit(mlp_prediction)
        else:
            self.fit = read(output / record["artifact"])

    def __call__(self, features):
        if self.kind == "mlp":
            import jax

            return np.asarray(
                self.fn(
                    self.params,
                    jax.device_put(
                        (np.asarray(features) - self.norm["mean"]) / self.norm["std"]
                    ),
                )
            )
        if self.kind == "ridge":
            return self.predict_ridge(self.fit, np.asarray(features))
        return self.predict_affine(self.fit, np.asarray(features))


def authenticate_parent(parent):
    protocol = read(PROTOCOL)
    manifest = read(parent / "manifest.json")
    if (
        manifest["source_commit"] != protocol["parent_commit"]
        or sha(parent / "report.json") != protocol["parent_report_sha256"]
    ):
        raise ValueError("wrong parent evidence")
    source = Path(manifest["source"])
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=f"{source}/dreamer_imf_comparison:{source}/imf_dreamer_jax/src",
    )
    subprocess.run(
        [
            sys.executable,
            str(source / "dreamer_imf_comparison/scripts/inspect_parallel_study.py"),
            "--root",
            str(parent),
            "--level",
            "final",
        ],
        env=env,
        check=True,
    )
    marker = read(parent / "markers/collect.json")
    data_path = Path(marker["payload"]["dataset"])
    return manifest, data_path


def setup_teacher(manifest):
    if str(manifest["upstream"]) not in sys.path:
        sys.path.insert(0, str(manifest["upstream"]))
    import jax
    from embodied.jax import internal

    internal.setup(
        platform="cuda",
        compute_dtype="bfloat16",
        transfer_guard=False,
        compilation_cache=False,
    )
    if not all(d.platform == "gpu" for d in jax.devices()):
        raise ValueError("GPU required for exact parent replay")
    from .parallel_frozen import FrozenModel

    return FrozenModel(manifest["cells"]["imf"], "imf")


def summarize_one(pred, truth, data, indices, scale, baseline):
    from .reward_probe_metrics import summarize

    return summarize(
        pred,
        truth,
        data["episode"][indices],
        data["mode"][indices],
        data["plan"][indices],
        scale,
        baseline=baseline,
        bootstrap_reps=2000,
    )


def frozen_predictions(teacher, data, parent):
    result = np.zeros_like(data["rewards"], dtype=np.float32)
    for split in range(3):
        ix = np.flatnonzero(data["split"] == split)
        result[ix] = np.asarray(teacher.decode(data["targets"][ix])[1])
    retained = load_npz(parent / "evaluation/imf/posterior_heads.npz")
    equal(
        result[retained["indices"]],
        retained["rewards"][:, 0],
        "original posterior rewards",
    )
    return result


def rollout_predictions(teacher, data, parent, chosen):
    import jax
    from .parallel_runner import load_predictor

    ix = np.flatnonzero(data["split"] == 2)
    results = {}
    for name in ("imf1", "imf4", "head0", "head1", "head2"):
        head = name.startswith("head")
        n = int(name[-1])
        predictor = load_predictor(parent / "fit" / str(n), teacher) if head else None
        original_path = parent / (
            f"evaluation/head-{n}/nfe1.npz" if head else f"evaluation/imf/nfe{n}.npz"
        )
        original = load_npz(original_path)
        values, decoded = [], []
        digests = []
        for chunk, start in enumerate(range(0, len(ix), 8)):
            rows = ix[start : start + 8]
            if head:
                features = predictor.particles(
                    data["start"][rows],
                    data["actions"][rows],
                    jax.random.PRNGKey(19900 + n * 100 + chunk),
                    32,
                    1,
                )
            else:
                features = teacher.rollout(
                    data["start"][rows], data["actions"][rows], 67100 + start, 32, n
                )
            old = np.asarray(teacher.decode(features)[1])
            decoded.append(old)
            host = np.asarray(features)
            digests.append(hashlib.sha256(host.tobytes()).hexdigest())
            values.append(chosen(features))
        values, decoded = np.concatenate(values), np.concatenate(decoded)
        equal(decoded, original["rewards"], name + " retained reward replay")
        print("REWARD_READOUT_PARENT_REPLAY_VERIFIED", name, flush=True)
        results[name] = dict(
            prediction=values, original=decoded, feature_digests=digests
        )
    return results


def resampled_posterior(teacher, data, dataset_dir, chosen, draws):
    from .parallel_collection import load_raw_history

    eid = int(np.min(data["episode"][data["split"] == 2]))
    indices = np.flatnonzero(data["episode"] == eid)
    history, obs = load_raw_history(dataset_dir / f"episode-{eid:03d}.npz")
    outputs, original = [], []
    for draw in range(draws):

        def seed(value):
            return int(
                np.random.SeedSequence([int(value), draw, 9127]).generate_state(1)[0]
            )

        carry, _ = teacher.observe(
            teacher.initial(),
            obs[0],
            history["previous_action"],
            seed(history["observe_seeds"][0]),
        )
        features = np.zeros((len(indices), 15, 2560), np.float32)
        for t in range(1, 401):
            carry, _ = teacher.observe(
                carry,
                obs[t],
                history["actions"][t - 1],
                seed(history["observe_seeds"][t]),
            )
            if t not in (100, 200, 300, 400):
                continue
            for i in np.flatnonzero(data["anchor"][indices] == t):
                plan = int(data["plan"][indices[i]])
                branch, raw = load_raw_history(
                    dataset_dir / f"branch-{eid:03d}-{t:03d}-{plan}.npz"
                )
                state = copy.deepcopy(carry)
                for j in range(15):
                    state, feature = teacher.observe(
                        state,
                        raw[j + 1],
                        branch["actions"][j],
                        seed(branch["observe_seeds"][j + 1]),
                    )
                    features[i, j] = feature
        original.append(np.asarray(teacher.decode(features)[1]))
        outputs.append(chosen(features))
        print("REWARD_READOUT_POSTERIOR_DRAW", draw, flush=True)
    return dict(
        indices=indices, original=np.asarray(original), prediction=np.asarray(outputs)
    )


def result_report(output, data, baseline, posterior, rollouts, sampling, scale):
    from .reward_probe_metrics import decomposition

    ix = np.flatnonzero(data["split"] == 2)
    truth = data["rewards"][ix]
    report = dict(
        schema=read(PROTOCOL)["schema"],
        additional_simulator_steps=0,
        source_commit=commit(SOURCE),
        posterior={},
        rollouts={},
        posterior_sampling={},
        limitations=read(PROTOCOL)["limitations"],
    )
    report["posterior"]["frozen"] = summarize_one(
        baseline[ix], truth, data, ix, scale, baseline[ix]
    )
    for name, value in posterior.items():
        report["posterior"][name] = summarize_one(
            value, truth, data, ix, scale, baseline[ix]
        )
    for name, value in rollouts.items():
        old, pred = value["original"].mean(1, dtype=np.float64), value[
            "prediction"
        ].mean(1, dtype=np.float64)
        report["rollouts"][name] = dict(
            original=summarize_one(old, truth, data, ix, scale, old),
            improved=summarize_one(pred, truth, data, ix, scale, old),
            original_decomposition=decomposition(old, baseline[ix], truth, scale),
            readout_decomposition=decomposition(
                pred, posterior["primary"], truth, scale
            ),
        )
    subset_truth = data["rewards"][sampling["indices"]]
    for name in ("original", "prediction"):
        values = sampling[name].astype(np.float64)
        returns = values.sum(-1) / scale
        target = subset_truth.sum(-1) / scale
        report["posterior_sampling"][name] = dict(
            episode=int(data["episode"][sampling["indices"][0]]),
            draws=len(values),
            mean_single_draw_cumulative_mse=float(np.mean((returns - target) ** 2)),
            ensemble_mean_cumulative_mse=float(
                np.mean((returns.mean(0) - target) ** 2)
            ),
            across_draw_return_variance=float(returns.var(0).mean()),
        )
    return report


def run(parent, output):
    from .reward_probe_metrics import (
        fit_ridge,
        predict_ridge,
        fit_affine,
        predict_affine,
    )

    protocol = read(PROTOCOL)
    manifest, data_path = authenticate_parent(parent)
    source_commit = commit(SOURCE)
    output.mkdir(parents=True, exist_ok=False)
    dependencies = {
        str(path): sha(path)
        for path in (
            parent / "report.json",
            parent / "completion.json",
            parent / "manifest.json",
            data_path,
            Path(manifest["budget_path"]),
        )
    }
    _write_json(
        output / "manifest.json",
        dict(
            source=str(SOURCE),
            source_commit=source_commit,
            protocol_sha256=sha(PROTOCOL),
            parent=str(parent),
            dependencies=dependencies,
            started=time.time(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            pid=os.getpid(),
        ),
    )
    _write_json(output / "protocol.json", protocol)
    data = load_npz(data_path)
    validate_data(data)
    teacher = setup_teacher(manifest)
    before = teacher.frozen_digest()
    baseline = frozen_predictions(teacher, data, parent)
    _write_npz(output / "frozen_rewards.npz", dict(prediction=baseline))
    masks = [np.flatnonzero(data["split"] == s) for s in range(3)]
    tr, va, te = masks
    scale = float(np.sqrt(np.mean(data["rewards"][tr].astype(np.float64).sum(-1) ** 2)))
    scale = max(scale, 0.01)
    records = []
    affine = fit_affine(baseline[tr], data["rewards"][tr])
    _write_json(output / "affine.json", affine)
    records.append(
        dict(
            name="affine",
            kind="affine",
            artifact="affine.json",
            score=validation_score(
                predict_affine(affine, baseline[va]), data["rewards"][va], scale
            ),
        )
    )
    for group in protocol["ridge_groups"]:
        features = group_features(data, group)
        candidates = []
        for alpha in protocol["ridge_alpha"]:
            fitted = fit_ridge(features[tr], data["rewards"][tr], alpha)
            artifact = f"ridge_{group}_{alpha}.json"
            _write_json(output / artifact, fitted)
            pred = predict_ridge(fitted, features[va])
            score = validation_score(pred, data["rewards"][va], scale)
            candidates.append(
                dict(
                    name="ridge_" + group,
                    kind="ridge",
                    group=group,
                    alpha=alpha,
                    artifact=artifact,
                    score=score,
                )
            )
        _write_json(
            output / f"ridge_{group}_selection.json",
            dict(candidates=candidates, selected=candidates[first_minimum(candidates)]),
        )
        records.append(candidates[first_minimum(candidates)])
    for seed in protocol["mlp_seeds"]:
        records.append(
            train_mlp(
                output / f"mlp_{seed}",
                data["targets"][tr],
                data["rewards"][tr],
                data["targets"][va],
                data["rewards"][va],
                scale,
                seed,
                protocol,
            )
        )
    eligible = [r for r in records if r["name"] in protocol["primary_candidates"]]
    primary = eligible[first_minimum(eligible)]
    _write_json(
        output / "selection.json",
        dict(
            records=records,
            primary=primary,
            scale=scale,
            training_episodes=np.unique(data["episode"][tr]).tolist(),
            validation_episodes=np.unique(data["episode"][va]).tolist(),
        ),
    )
    posterior = {}
    for record in records:
        readout = Readout(output, record)
        x = (
            baseline[te]
            if record["kind"] == "affine"
            else group_features(data, record.get("group", "hz"))[te]
        )
        posterior[record["name"]] = readout(x)
    posterior["primary"] = posterior[primary["name"]]
    _write_npz(output / "posterior_predictions.npz", dict(indices=te, **posterior))
    chosen = Readout(output, primary)
    rollouts = rollout_predictions(teacher, data, parent, chosen)
    for name, value in rollouts.items():
        _write_npz(
            output / f"rollout_{name}.npz",
            dict(
                indices=te, prediction=value["prediction"], original=value["original"]
            ),
        )
        _write_json(output / f"rollout_{name}_digests.json", value["feature_digests"])
    sampling = resampled_posterior(
        teacher, data, data_path.parent, chosen, protocol["posterior_draws"]
    )
    _write_npz(output / "posterior_sampling.npz", sampling)
    report = result_report(output, data, baseline, posterior, rollouts, sampling, scale)
    report["selected_readout"] = primary
    report["frozen_parameter_digest"] = before
    if before != teacher.frozen_digest():
        raise ValueError("parent mutated")
    for path, digest in dependencies.items():
        if sha(path) != digest:
            raise ValueError("parent artifact changed")
    _write_json(output / "report.json", report)
    _write_json(
        output / "run_completed.json",
        dict(
            source_commit=source_commit,
            report_sha256=sha(output / "report.json"),
            artifacts={
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            ended=time.time(),
        ),
    )
    print("REWARD_READOUT_RUN_COMPLETED", flush=True)


def verify(output):
    from .reward_probe_metrics import (
        fit_ridge,
        predict_ridge,
        fit_affine,
        predict_affine,
    )

    saved = read(output / "manifest.json")
    protocol = read(PROTOCOL)
    if saved["source_commit"] != commit(SOURCE) or saved["protocol_sha256"] != sha(
        PROTOCOL
    ):
        raise ValueError("audit source changed")
    completed = read(output / "run_completed.json")
    if saved["pid"] == os.getpid():
        raise ValueError("verification requires an independent process")
    expected = set(completed["artifacts"]) | {"run_completed.json"}
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("unexpected audit artifacts")
    if completed["source_commit"] != saved["source_commit"] or completed[
        "report_sha256"
    ] != sha(output / "report.json"):
        raise ValueError("completion binding differs")
    for path, digest in completed["artifacts"].items():
        if sha(output / path) != digest:
            raise ValueError("audit artifact changed")
    parent = Path(saved["parent"])
    manifest, data_path = authenticate_parent(parent)
    data = load_npz(data_path)
    validate_data(data)
    teacher = setup_teacher(manifest)
    before = teacher.frozen_digest()
    baseline = frozen_predictions(teacher, data, parent)
    equal(
        baseline,
        load_npz(output / "frozen_rewards.npz")["prediction"],
        "frozen rewards",
    )
    selection = read(output / "selection.json")
    tr, va, te = [np.flatnonzero(data["split"] == s) for s in range(3)]
    scale = max(
        float(np.sqrt(np.mean(data["rewards"][tr].astype(np.float64).sum(-1) ** 2))),
        0.01,
    )
    equal(selection["scale"], scale, "training normalization")
    equal(
        selection["training_episodes"],
        np.unique(data["episode"][tr]),
        "training episodes",
    )
    equal(
        selection["validation_episodes"],
        np.unique(data["episode"][va]),
        "validation episodes",
    )
    names = (
        ["affine"]
        + ["ridge_" + g for g in protocol["ridge_groups"]]
        + [f"mlp_{s}" for s in protocol["mlp_seeds"]]
    )
    if [r["name"] for r in selection["records"]] != names:
        raise ValueError("incomplete readout candidate set")
    for record in selection["records"]:
        if record["kind"] == "affine":
            fitted = fit_affine(baseline[tr], data["rewards"][tr])
            if fitted != read(output / record["artifact"]):
                raise ValueError("affine training replay differs")
            equal(
                validation_score(
                    predict_affine(fitted, baseline[va]), data["rewards"][va], scale
                ),
                record["score"],
                "affine validation",
            )
        elif record["kind"] == "ridge":
            grid = read(output / f"ridge_{record['group']}_selection.json")
            equal(
                [c["alpha"] for c in grid["candidates"]],
                protocol["ridge_alpha"],
                "ridge grid",
            )
            for candidate in grid["candidates"]:
                features = group_features(data, candidate["group"])
                fitted = fit_ridge(
                    features[tr], data["rewards"][tr], candidate["alpha"]
                )
                if fitted != read(output / candidate["artifact"]):
                    raise ValueError("ridge training replay differs")
                pred = predict_ridge(
                    fitted,
                    features[va],
                )
                equal(
                    validation_score(pred, data["rewards"][va], scale),
                    candidate["score"],
                    "ridge validation",
                )
            if grid["candidates"][first_minimum(grid["candidates"])] != record:
                raise ValueError("ridge selection differs")
        elif record["kind"] == "mlp":
            details = read(output / record["directory"] / "selection.json")
            flat = data["targets"][tr].reshape(-1, 2560)
            normalization = load_npz(output / record["directory"] / "normalization.npz")
            equal(
                normalization["mean"],
                flat.mean(0, dtype=np.float64).astype(np.float32),
                "MLP train mean",
            )
            equal(
                normalization["std"],
                np.maximum(flat.std(0, dtype=np.float64), 0.01).astype(np.float32),
                "MLP train std",
            )
            equal(
                [c["update"] for c in details["validations"]],
                list(
                    range(
                        protocol["mlp_validation_period"],
                        protocol["mlp_updates"] + 1,
                        protocol["mlp_validation_period"],
                    )
                ),
                "MLP checkpoint grid",
            )
            for checkpoint in details["validations"]:
                probe = Readout(output, dict(record, **checkpoint))
                pred = probe(data["targets"][va])
                equal(
                    pred,
                    load_npz(
                        output
                        / record["directory"]
                        / f"validation_{checkpoint['update']:05d}.npz"
                    )["prediction"],
                    "MLP validation predictions",
                )
                equal(
                    validation_score(pred, data["rewards"][va], scale),
                    checkpoint["score"],
                    "MLP validation score",
                )
            if (
                details["selected"]
                != details["validations"][first_minimum(details["validations"])]
            ):
                raise ValueError("MLP checkpoint selection differs")
            if record["checkpoint"] != details["selected"]["checkpoint"]:
                raise ValueError("MLP selected checkpoint differs")
            for key in ("score", "update"):
                equal(record[key], details["selected"][key], "MLP selection " + key)
    eligible = [
        r
        for r in selection["records"]
        if r["name"] in read(PROTOCOL)["primary_candidates"]
    ]
    if selection["primary"] != eligible[first_minimum(eligible)]:
        raise ValueError("primary readout selection differs")
    posterior = {}
    retained = load_npz(output / "posterior_predictions.npz")
    equal(retained["indices"], te, "test split")
    for record in selection["records"]:
        x = (
            baseline[te]
            if record["kind"] == "affine"
            else group_features(data, record.get("group", "hz"))[te]
        )
        posterior[record["name"]] = Readout(output, record)(x)
        equal(posterior[record["name"]], retained[record["name"]], "posterior readout")
    posterior["primary"] = posterior[selection["primary"]["name"]]
    equal(posterior["primary"], retained["primary"], "primary posterior")
    chosen = Readout(output, selection["primary"])
    rollouts = rollout_predictions(teacher, data, parent, chosen)
    for name, value in rollouts.items():
        raw = load_npz(output / f"rollout_{name}.npz")
        equal(raw["indices"], te, "rollout test indices")
        for key in ("prediction", "original"):
            equal(value[key], raw[key], "rollout " + name + " " + key)
        if read(output / f"rollout_{name}_digests.json") != value["feature_digests"]:
            raise ValueError("latent rollout replay differs")
    sampling = resampled_posterior(
        teacher, data, data_path.parent, chosen, read(PROTOCOL)["posterior_draws"]
    )
    retained_sampling = load_npz(output / "posterior_sampling.npz")
    for key, value in sampling.items():
        equal(value, retained_sampling[key], "posterior resampling " + key)
    regenerated = result_report(
        output, data, baseline, posterior, rollouts, sampling, scale
    )
    regenerated["selected_readout"] = selection["primary"]
    regenerated["frozen_parameter_digest"] = before
    if regenerated != read(output / "report.json"):
        raise ValueError("independent report differs")
    if before != teacher.frozen_digest():
        raise ValueError("parent mutated during verification")
    for path, digest in saved["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("parent artifact changed")
    _write_json(
        output / "verification.json",
        dict(
            verified=True,
            source_commit=commit(SOURCE),
            report_sha256=sha(output / "report.json"),
            pid=os.getpid(),
            additional_simulator_steps=0,
        ),
    )
    print("REWARD_READOUT_INDEPENDENT_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "verify"))
    parser.add_argument("--parent")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stage == "run":
        run(Path(args.parent), Path(args.output))
    else:
        verify(Path(args.output))
