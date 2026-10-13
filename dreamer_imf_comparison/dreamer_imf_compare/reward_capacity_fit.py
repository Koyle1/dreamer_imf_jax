"""Isolated capacity-only variation of the matched categorical readout."""

from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .parallel_collection import _write_json, _write_npz
from .parallel_runner import finite, load_npz
from .prior_correction_study import dump, load
from .reward_readout_study import mlp_logits, read, equal
from .reward_diagnosis_metrics import nll


def init(key, dimension, width):
    if dimension < 1 or width < 1:
        raise ValueError("invalid architecture")
    return tuple(
        {"w": jax.random.normal(k, (a, b)) * np.sqrt(2 / a), "b": jnp.zeros(b)}
        for k, a, b in zip(
            jax.random.split(key, 3), (dimension, width, width), (width, width, 3)
        )
    )


def normalize(train):
    flat = np.asarray(train, np.float32).reshape(-1, train.shape[-1])
    return dict(
        mean=flat.mean(0, dtype=np.float64).astype(np.float32),
        std=np.maximum(flat.std(0, dtype=np.float64), 0.01).astype(np.float32),
    )


@jax.jit
def probabilities(params, x):
    return jax.nn.softmax(mlp_logits(params, x), -1)


def predict(params, norm, x):
    result = np.asarray(
        probabilities(params, jnp.asarray((x - norm["mean"]) / norm["std"]))
    )
    finite(result)
    return result


def scores(probs, y):
    return dict(
        nll=float(nll(probs, y).mean()),
        mse=float(np.mean((probs @ np.arange(3) - y) ** 2)),
    )


def select(records):
    if not records or any(not np.isfinite(r["validation"]["nll"]) for r in records):
        raise ValueError("invalid candidate scores")
    return min(records, key=lambda r: (r["validation"]["nll"], r["update"]))


def fit(directory, train_x, train_y, val_x, val_y, width, seed, protocol):
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    norm = normalize(train_x)
    _write_npz(directory / "normalization.npz", norm)
    x = jnp.asarray(
        ((train_x - norm["mean"]) / norm["std"]).reshape(-1, train_x.shape[-1])
    )
    y = jnp.asarray(train_y.reshape(-1).astype(np.int32))
    params = init(jax.random.PRNGKey(seed), train_x.shape[-1], width)
    optimizer = optax.adam(protocol["learning_rate"])
    state = optimizer.init(params)

    @jax.jit
    def update(p, state, xx, yy):
        loss, grad = jax.value_and_grad(
            lambda w: optax.softmax_cross_entropy_with_integer_labels(
                mlp_logits(w, xx), yy
            ).mean()
        )(p)
        delta, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, delta), state, loss, optax.global_norm(grad)

    rng = np.random.default_rng(protocol["minibatch_seed"])
    records, gradients = [], []
    for step in range(1, protocol["updates"] + 1):
        ix = rng.integers(len(x), size=protocol["batch_size"])
        params, state, loss, grad = update(params, state, x[ix], y[ix])
        finite((params, state, loss, grad))
        if step == 1 or step % protocol["validation_period"] == 0:
            gradients.append(
                dict(update=step, minibatch_loss=float(loss), gradient_norm=float(grad))
            )
        if step % protocol["validation_period"]:
            continue
        train = predict(params, norm, train_x)
        val = predict(params, norm, val_x)
        name = f"checkpoint-{step:05d}.pkl"
        dump(directory / name, params)
        _write_npz(
            directory / f"candidate-{step:05d}.npz", dict(train=train, validation=val)
        )
        records.append(
            dict(
                update=step,
                checkpoint=name,
                train=scores(train, train_y),
                validation=scores(val, val_y),
            )
        )
    record = dict(
        selected=select(records),
        candidates=records,
        gradients=gradients,
        width=width,
        depth=2,
        seed=seed,
        minibatch_seed=protocol["minibatch_seed"],
        parameters=sum(v.size for v in jax.tree.leaves(params)),
        fitting_seconds_including_compilation=time.perf_counter() - started,
    )
    _write_json(directory / "selection.json", record)
    print(
        "REWARD_CAPACITY_HEAD_FITTED",
        directory.name,
        record["selected"]["update"],
        flush=True,
    )


def verify_fit(directory, train_x, train_y, val_x, val_y, width, seed, protocol):
    directory = Path(directory)
    norm = load_npz(directory / "normalization.npz")
    expected = normalize(train_x)
    for key in expected:
        equal(norm[key], expected[key], "train-only normalization")
    record = read(directory / "selection.json")
    steps = list(
        range(
            protocol["validation_period"],
            protocol["updates"] + 1,
            protocol["validation_period"],
        )
    )
    count = (train_x.shape[-1] + 1) * width + (width + 1) * width + (width + 1) * 3
    if (
        record["width"],
        record["depth"],
        record["seed"],
        record["minibatch_seed"],
        record["parameters"],
    ) != (width, 2, seed, protocol["minibatch_seed"], count):
        raise ValueError("architecture or sampler mismatch")
    if [r["update"] for r in record["candidates"]] != steps:
        raise ValueError("candidate inventory differs")
    if record["selected"] != select(record["candidates"]):
        raise ValueError("selection is not earliest validation minimum")
    for row in record["candidates"]:
        step = row["update"]
        if row["checkpoint"] != f"checkpoint-{step:05d}.pkl":
            raise ValueError("invalid checkpoint path")
        params = load(directory / row["checkpoint"])
        shapes = [(train_x.shape[-1], width), (width, width), (width, 3)]
        if len(params) != 3 or any(
            p["w"].shape != s or p["b"].shape != (s[1],) for p, s in zip(params, shapes)
        ):
            raise ValueError("checkpoint architecture differs")
        saved = load_npz(directory / f"candidate-{step:05d}.npz")
        for name, x, y in (("train", train_x, train_y), ("validation", val_x, val_y)):
            pred = predict(params, norm, x)
            equal(pred, saved[name], "independent candidate " + name)
            if scores(pred, y) != row[name]:
                raise ValueError("candidate metrics differ")
    return record
