"""Offline learned-prior correction: immutable fitting and separate replay.

Run and verify are separate processes. There are no simulator calls. The
previously inspected test split is exploratory, not a new confirmation set.
"""

import argparse
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from imf_dreamer_jax import prior_correction as field
from .prior_correction import (
    FrozenPrior,
    corrected,
    normalization,
    rollout,
    validate_data,
)
from .parallel_collection import _atomic_artifact, _write_json, _write_npz
from .parallel_runner import finite, load_npz
from .parallel_metrics import fit_scales, summarize
from .reward_readout_study import (
    authenticate_parent,
    setup_teacher,
    sha,
    read,
    commit,
    SOURCE,
)

PROTOCOL = dict(
    schema=1,
    seed=811,
    updates=2000,
    validation_period=250,
    batch_size=256,
    hidden_dim=256,
    depth=2,
    learning_rate=3e-4,
    weight_decay=1e-4,
    gradient_clip=1.0,
    validation_particles=8,
    test_particles=32,
    horizon=15,
    additional_simulator_steps=0,
    train_only_normalization=True,
    target="stopped real posterior samples",
    source="frozen learned conditional Gaussian",
    initial_correction="identity",
    selection="validation normalized observation MSE plus cumulative reward MSE",
    final_distribution_KL=False,
    statistical_unit="episode; one frozen parent",
    status="exploratory reuse of previously inspected held-out episodes",
)


def dump(path, value):
    _atomic_artifact(
        Path(path), lambda f: pickle.dump(jax.device_get(value), f, protocol=5)
    )


def load(path):
    # Only our hash-authenticated, trusted local checkpoint files may be loaded.
    with Path(path).open("rb") as f:
        return pickle.load(f)


def predictions(
    adapter, params, norm, data, indices, *, bypass=False, steps=1, count=32, seed=8801
):
    run = jax.jit(
        lambda s, a, k: rollout(
            adapter, params, norm, s, a, k, particles=count, bypass=bypass, steps=steps
        )
    )
    obs, rewards = [], []
    for at in range(0, len(indices), 8):
        ix = indices[at : at + 8]
        features = run(
            jnp.asarray(data["start"][ix]),
            jnp.asarray(data["actions"][ix]),
            jax.random.fold_in(jax.random.PRNGKey(seed), at),
        )
        o, r = adapter.teacher.decode(features)
        finite((features, o, r))
        obs.append(np.asarray(o))
        rewards.append(np.asarray(r))
    return dict(observation=np.concatenate(obs), reward=np.concatenate(rewards))


def score(pred, data, indices, obs_scale, return_scale):
    o = np.asarray(pred["observation"], np.float64).mean(1)
    r = np.asarray(pred["reward"], np.float64).mean(1)
    return float(
        np.mean(((o - data["observations"][indices]) / obs_scale) ** 2)
        + np.mean(((r.sum(1) - data["rewards"][indices].sum(1)) / return_scale) ** 2)
    )


def fit(output, adapter, data, protocol=PROTOCOL):
    """Only this separate flow tree is differentiated or passed to AdamW."""
    output = Path(output)
    teacher = adapter.teacher
    validate_data(data, teacher.feature_dim)
    norm = normalization(data, teacher.deter)
    _write_npz(output / "normalization.npz", norm)
    train = np.flatnonzero(data["split"] == 0)
    val = np.flatnonzero(data["split"] == 1)
    obs_scale, return_scale = fit_scales(
        data["observations"][train], data["rewards"][train]
    )
    _write_npz(
        output / "scales.npz", dict(obs_scale=obs_scale, return_scale=return_scale)
    )
    values = data["targets"][train].reshape(-1, teacher.feature_dim)
    h, target = values[:, : teacher.deter], values[:, teacher.deter :]
    means, stds = [], []
    for at in range(0, len(h), 256):
        m, s = adapter.prior(jnp.asarray(h[at : at + 256]))
        means.append(np.asarray(m))
        stds.append(np.asarray(s))
    mean, std = np.concatenate(means), np.concatenate(stds)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("invalid frozen source scale")
    arrays = tuple(
        map(
            jnp.asarray,
            (
                (target - norm["zmean"]) / norm["zstd"],
                (h - norm["hmean"]) / norm["hstd"],
                (mean - norm["zmean"]) / norm["zstd"],
                std / norm["zstd"],
            ),
        )
    )
    params = field.init(
        jax.random.PRNGKey(protocol["seed"]),
        target.shape[-1],
        h.shape[-1],
        protocol["hidden_dim"],
        protocol["depth"],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(protocol["gradient_clip"]),
        optax.adamw(protocol["learning_rate"], weight_decay=protocol["weight_decay"]),
    )
    state = optimizer.init(params)

    @jax.jit
    def update(p, state, x, c, m, s, key):
        def objective(w):
            d = field.loss(w, x, c, m, s, key, return_details=True)
            return d.loss, (d.raw_loss_u.mean(), d.raw_loss_v.mean())

        (value, raw), grad = jax.value_and_grad(objective, has_aux=True)(p)
        delta, state = optimizer.update(grad, state, p)
        return (
            optax.apply_updates(p, delta),
            state,
            (value, *raw, optax.global_norm(grad), optax.global_norm(delta)),
        )

    rng = np.random.default_rng(protocol["seed"])
    records, training = [], []
    for step in range(protocol["updates"] + 1):
        if step:
            ix = rng.integers(len(target), size=protocol["batch_size"])
            params, state, metrics = update(
                params,
                state,
                *(x[ix] for x in arrays),
                jax.random.fold_in(jax.random.PRNGKey(protocol["seed"]), step),
            )
            finite((params, state, metrics))
            if step == 1 or step % 50 == 0:
                training.append(
                    dict(
                        update=step,
                        loss=float(metrics[0]),
                        raw_u=float(metrics[1]),
                        raw_v=float(metrics[2]),
                        gradient_norm=float(metrics[3]),
                        update_norm=float(metrics[4]),
                    )
                )
        if step % protocol["validation_period"]:
            continue
        pred = predictions(
            adapter, params, norm, data, val, count=protocol["validation_particles"]
        )
        name = f"checkpoint-{step:05d}.pkl"
        dump(output / name, params)
        _write_npz(output / f"validation-{step:05d}.npz", pred)
        records.append(
            dict(
                update=step,
                checkpoint=name,
                score=score(pred, data, val, obs_scale, return_scale),
            )
        )
        adapter.assert_frozen()
        print("PRIOR_CORRECTION_VALIDATION", records[-1], flush=True)
    # Includes the zero-initialized identity checkpoint. Ties select earliest.
    chosen = min(records, key=lambda x: (x["score"], x["update"]))
    _write_json(
        output / "selection.json",
        dict(selected=chosen, candidates=records, training=training),
    )
    return load(output / chosen["checkpoint"]), norm


def reports(adapter, params, norm, data, indices, output=None):
    train = data["split"] == 0
    oscale, rscale = fit_scales(data["observations"][train], data["rewards"][train])
    results, raw = {}, {}
    for name, bypass, steps in (
        ("base", True, 1),
        ("corrected", False, 1),
        ("corrected_four", False, 4),
    ):
        pred = predictions(
            adapter, params, norm, data, indices, bypass=bypass, steps=steps
        )
        raw[name] = pred
    # Existing iMF path is context, not the source-prior baseline.
    obs, rew = [], []
    for at in range(0, len(indices), 8):
        ix = indices[at : at + 8]
        f = adapter.teacher.rollout(
            data["start"][ix],
            data["actions"][ix],
            9901 + at,
            particles=32,
            flow_steps=4,
        )
        o, r = adapter.teacher.decode(f)
        obs.append(np.asarray(o))
        rew.append(np.asarray(r))
    raw["original_imf_four"] = dict(
        observation=np.concatenate(obs), reward=np.concatenate(rew)
    )
    o, r = adapter.teacher.decode(jnp.asarray(data["targets"][indices]))
    raw["real_posterior"] = dict(
        observation=np.asarray(o)[:, None], reward=np.asarray(r)[:, None]
    )
    for name, pred in raw.items():
        finite(pred)
        if output is not None:
            _write_npz(Path(output) / f"test-{name}.npz", pred)
        results[name] = summarize(
            pred["observation"],
            pred["reward"],
            data["observations"][indices],
            data["rewards"][indices],
            data["episode"][indices],
            data["anchor"][indices],
            data["plan"][indices],
            oscale,
            rscale,
        )
    return results, raw


def conditional_samples(adapter, params, norm, data, indices):
    """Teacher-forced first transition: final latent distribution, not base KL.

    The target's recurrent h is a deterministic function of the previous belief
    and first action, not of the target observation. Future posterior z is a
    held-out scoring target only. This is distinct from free rollout metrics.
    """
    h = jnp.repeat(
        jnp.asarray(data["targets"][indices, 0, : adapter.teacher.deter]), 32, 0
    )
    mean, std = adapter.prior(h)
    noise = jax.random.normal(jax.random.PRNGKey(9107), mean.shape)
    raw, result = {}, {}
    target = data["targets"][indices, 0, adapter.teacher.deter :]
    for name, bypass in (("base", True), ("corrected", False)):
        samples = np.asarray(
            corrected(params, norm, h, mean, std, noise, bypass=bypass)
        )
        samples = samples.reshape(len(indices), 32, -1)
        raw[name] = samples
        x = samples.astype(np.float64) / norm["zstd"]
        y = target.astype(np.float64) / norm["zstd"]
        distance = lambda a: np.sqrt(np.mean(a * a, axis=-1))
        cross = distance(x - y[:, None]).mean(1)
        # Unbiased within-distribution term, excluding identical sample pairs.
        within = sum(distance(x - x[:, i : i + 1]).sum(1) for i in range(32)) / (
            32 * 31
        )
        energy = cross - 0.5 * within
        per_episode = {
            str(int(e)): float(energy[data["episode"][indices] == e].mean())
            for e in np.unique(data["episode"][indices])
        }
        result[name] = dict(
            energy_score=float(np.mean(list(per_episode.values()))),
            energy_score_by_episode=per_episode,
            interpretation="teacher-forced first-step final-sample energy score; lower is better",
        )
    finite(raw)
    return result, raw


def protocol_for(preflight):
    return (
        dict(
            PROTOCOL,
            updates=1,
            validation_period=1,
            batch_size=8,
            validation_particles=2,
        )
        if preflight
        else dict(PROTOCOL)
    )


def subset(data, preflight):
    if not preflight:
        return data
    ix = np.concatenate([np.flatnonzero(data["split"] == s)[:4] for s in (0, 1, 2)])
    return {k: v[ix] for k, v in data.items()}


def run(parent, output, *, preflight=False):
    source_commit = commit(SOURCE)
    manifest, dataset = authenticate_parent(Path(parent))
    teacher = setup_teacher(manifest)
    adapter = FrozenPrior(teacher)
    data = subset(load_npz(dataset), preflight)
    validate_data(data, teacher.feature_dim)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "manifest.json",
        dict(
            source_commit=source_commit,
            parent=str(Path(parent).resolve()),
            dataset=str(dataset),
            dataset_sha256=sha(dataset),
            frozen_digest=adapter.before,
            protocol=protocol_for(preflight),
            preflight=preflight,
            devices=[str(d) for d in jax.devices()],
        ),
    )
    started = time.monotonic()
    params, norm = fit(output, adapter, data, protocol_for(preflight))
    test = np.flatnonzero(data["split"] == 2)
    report, _ = reports(adapter, params, norm, data, test, output)
    conditional, samples = conditional_samples(adapter, params, norm, data, test)
    _write_npz(output / "conditional-samples.npz", samples)
    adapter.assert_frozen()
    _write_json(
        output / "report.json",
        dict(
            metrics=report,
            conditional=conditional,
            additional_simulator_steps=0,
            elapsed_seconds=time.monotonic() - started,
            claims="exploratory frozen-parent diagnostic",
            correction_parameters=sum(x.size for x in jax.tree.leaves(params)),
            frozen_parent_parameters=teacher.parameter_count,
            final_distribution_KL_computed=False,
        ),
    )
    _write_json(
        output / "created.json",
        dict(
            artifacts={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        ),
    )
    print("PRIOR_CORRECTION_CREATED_REPLAY_REQUIRED", flush=True)


def verify(output):
    output = Path(output)
    for name, digest in read(output / "created.json")["artifacts"].items():
        if Path(name).name != name or sha(output / name) != digest:
            raise ValueError("artifact binding differs")
    m = read(output / "manifest.json")
    if m["source_commit"] != commit(SOURCE) or m["protocol"] != protocol_for(
        m["preflight"]
    ):
        raise ValueError("source/protocol mismatch")
    parent, dataset = authenticate_parent(Path(m["parent"]))
    if str(dataset) != m["dataset"] or sha(dataset) != m["dataset_sha256"]:
        raise ValueError("retained input mismatch")
    teacher = setup_teacher(parent)
    adapter = FrozenPrior(teacher)
    if adapter.before != m["frozen_digest"]:
        raise ValueError("parent mismatch")
    data = subset(load_npz(dataset), m["preflight"])
    validate_data(data, teacher.feature_dim)
    norm = load_npz(output / "normalization.npz")
    for k, v in normalization(data, teacher.deter).items():
        if not np.array_equal(v, norm[k]):
            raise ValueError("train-only normalizer mismatch")
    sel = read(output / "selection.json")
    val = np.flatnonzero(data["split"] == 1)
    oscale, rscale = fit_scales(
        data["observations"][data["split"] == 0], data["rewards"][data["split"] == 0]
    )
    for candidate in sel["candidates"]:
        pred = load_npz(output / f'validation-{candidate["update"]:05d}.npz')
        if score(pred, data, val, oscale, rscale) != candidate["score"]:
            raise ValueError("selection score mismatch")
    if (
        min(sel["candidates"], key=lambda x: (x["score"], x["update"]))
        != sel["selected"]
    ):
        raise ValueError("selection mismatch")
    params = load(output / sel["selected"]["checkpoint"])
    report, raw = reports(
        adapter, params, norm, data, np.flatnonzero(data["split"] == 2)
    )
    for name, pred in raw.items():
        retained = load_npz(output / f"test-{name}.npz")
        for k in pred:
            if not np.array_equal(pred[k], retained[k]):
                raise ValueError(f"independent prediction replay differs: {name}/{k}")
    if report != read(output / "report.json")["metrics"]:
        raise ValueError("independent metrics differ")
    conditional, samples = conditional_samples(
        adapter, params, norm, data, np.flatnonzero(data["split"] == 2)
    )
    retained = load_npz(output / "conditional-samples.npz")
    if conditional != read(output / "report.json")["conditional"] or any(
        not np.array_equal(v, retained[k]) for k, v in samples.items()
    ):
        raise ValueError("conditional distribution replay differs")
    adapter.assert_frozen()
    _write_json(
        output / "verified.json",
        dict(
            created_sha256=sha(output / "created.json"),
            report_sha256=sha(output / "report.json"),
            exact_replay=True,
        ),
    )
    print("PRIOR_CORRECTION_DIAGNOSTIC_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "preflight", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.mode in ("run", "preflight"):
        if args.parent is None:
            parser.error("run requires --parent")
        run(args.parent, args.output, preflight=args.mode == "preflight")
    else:
        verify(args.output)
