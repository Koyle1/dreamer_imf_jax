"""Frozen-feature readouts with early selection, explicit L2, and PCA/ridge.

Only training and validation arrays enter fitting or selection. Test prediction is
a separate, read-only operation. All output directories and artifacts are new.
"""

import hashlib
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .parallel_collection import _write_json, _write_npz
from .parallel_runner import finite, load_npz
from .prior_correction_study import dump, load
from .reward_capacity_fit import normalize, probabilities
from .reward_readout_study import equal, mlp_logits, read

WIDTH = 128


def candidate_steps(updates=5000):
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("updates must be a nonnegative integer")
    return sorted(
        {
            0,
            updates,
            *range(10, min(updates, 250) + 1, 10),
            *range(500, updates + 1, 250),
        }
    )


def _arrays(train_x, train_y, val_x, val_y, classes=None):
    result = []
    dimension = None
    for x, y in ((train_x, train_y), (val_x, val_y)):
        x, y = np.asarray(x, np.float32), np.asarray(y)
        if (
            x.ndim != 2
            or y.shape != (len(x),)
            or not len(x)
            or x.shape[1] < 1
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("expected finite, nonempty flat features and labels")
        if dimension is not None and x.shape[1] != dimension:
            raise ValueError("training/validation feature dimension differs")
        if classes is not None and not np.isin(y, np.arange(classes)).all():
            raise ValueError("label outside categorical support")
        dimension = x.shape[1]
        result.extend((x, y.astype(np.int32) if classes else y.astype(np.float64)))
    return tuple(result)


def _bindings(arrays):
    result = {}
    for name, value in zip(
        ("train_x", "train_y", "validation_x", "validation_y"), arrays
    ):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
        result[name] = digest.hexdigest()
    return result


def _protocol(protocol):
    updates = protocol.get("updates", 5000)
    candidate_steps(updates)
    batch = protocol.get("batch_size", 256)
    sampler = protocol.get("minibatch_seed", 6211)
    lr = protocol.get("learning_rate", 3e-4)
    if (
        not isinstance(batch, int)
        or isinstance(batch, bool)
        or batch < 1
        or not isinstance(sampler, int)
        or isinstance(sampler, bool)
        or sampler < 0
        or not np.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("invalid training protocol")
    return dict(
        updates=updates,
        batch_size=batch,
        minibatch_seed=sampler,
        learning_rate=float(lr),
    )


def init(key, dimension, classes=3):
    if dimension < 1 or classes not in (2, 3):
        raise ValueError("invalid readout architecture")
    return tuple(
        {"w": jax.random.normal(k, (a, b)) * np.sqrt(2 / a), "b": jnp.zeros(b)}
        for k, a, b in zip(
            jax.random.split(key, 3), (dimension, WIDTH, WIDTH), (WIDTH, WIDTH, classes)
        )
    )


def predict(params, norm, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1:] != norm["mean"].shape or not np.isfinite(x).all():
        raise ValueError("invalid prediction features")
    result = np.asarray(
        probabilities(params, jnp.asarray((x - norm["mean"]) / norm["std"]))
    )
    finite(result)
    return result


def scores(probs, y):
    probs, y = np.asarray(probs), np.asarray(y)
    if (
        probs.ndim != 2
        or y.shape != (len(probs),)
        or not len(y)
        or not np.isfinite(probs).all()
        or (probs < 0).any()
        or not np.allclose(probs.sum(-1), 1, atol=2e-6, rtol=0)
        or not np.isin(y, np.arange(probs.shape[-1])).all()
    ):
        raise ValueError("invalid categorical probabilities or labels")
    label = y.astype(int)
    nll = -np.log(np.maximum(probs[np.arange(len(y)), label].astype(float), 1e-8))
    return dict(
        nll=float(nll.mean()),
        mse=float(np.mean((probs @ np.arange(probs.shape[-1]) - y) ** 2)),
    )


def select(records):
    if (
        not records
        or any(not np.isfinite(r["validation"]["nll"]) for r in records)
        or len({r["update"] for r in records}) != len(records)
    ):
        raise ValueError("invalid candidate scores or duplicate updates")
    return min(records, key=lambda r: (r["validation"]["nll"], r["update"]))


def weight_penalty(params, alpha):
    """Explicit alpha/2 * sum(W**2); biases are never penalized."""
    return alpha * 0.5 * sum(jnp.sum(layer["w"] ** 2) for layer in params)


def fit_cell(
    directory, train_x, train_y, val_x, val_y, seed, alpha, protocol, classes=3
):
    started = time.perf_counter()
    p = _protocol(protocol)
    if (
        classes not in (2, 3)
        or not np.isfinite(alpha)
        or alpha < 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed < 2**32
    ):
        raise ValueError("invalid classes, seed, or L2 coefficient")
    arrays = _arrays(train_x, train_y, val_x, val_y, classes)
    train_x, train_y, val_x, val_y = arrays
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    norm = normalize(train_x)
    _write_npz(directory / "normalization.npz", norm)
    x = jnp.asarray((train_x - norm["mean"]) / norm["std"])
    y = jnp.asarray(train_y)
    params = init(jax.random.PRNGKey(seed), train_x.shape[1], classes)
    optimizer = optax.adam(p["learning_rate"])
    state = optimizer.init(params)

    @jax.jit
    def update(params, state, xx, yy):
        def objective(w):
            data_loss = optax.softmax_cross_entropy_with_integer_labels(
                mlp_logits(w, xx), yy
            ).mean()
            penalty = weight_penalty(w, alpha) if alpha else jnp.zeros(())
            return (data_loss + penalty if alpha else data_loss), (data_loss, penalty)

        (loss, (data_loss, penalty)), grad = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        delta, state = optimizer.update(grad, state, params)
        return (
            optax.apply_updates(params, delta),
            state,
            loss,
            data_loss,
            penalty,
            optax.global_norm(grad),
        )

    steps = candidate_steps(p["updates"])
    records, gradients = [], []

    def retain(step, params):
        train, val = predict(params, norm, train_x), predict(params, norm, val_x)
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

    retain(0, params)
    rng = np.random.default_rng(p["minibatch_seed"])
    for step in range(1, p["updates"] + 1):
        ix = rng.integers(len(x), size=p["batch_size"])
        params, state, loss, data_loss, penalty, grad = update(
            params, state, x[ix], y[ix]
        )
        finite((params, state, loss, data_loss, penalty, grad))
        if step == 1 or step in steps:
            gradients.append(
                dict(
                    update=step,
                    total_loss=float(loss),
                    data_nll=float(data_loss),
                    weight_penalty=float(penalty),
                    gradient_norm=float(grad),
                )
            )
        if step in steps:
            retain(step, params)
    record = dict(
        selected=select(records),
        candidates=records,
        gradients=gradients,
        width=WIDTH,
        depth=2,
        classes=classes,
        seed=int(seed),
        alpha=float(alpha),
        protocol=p,
        parameters=sum(v.size for v in jax.tree.leaves(params)),
        input_sha256=_bindings(arrays),
        fitting_seconds_including_compilation=time.perf_counter() - started,
    )
    _write_json(directory / "selection.json", record)
    print(
        "REWARD_RMT_HEAD_FITTED",
        directory.name,
        record["selected"]["update"],
        flush=True,
    )
    return record


def verify_cell(
    directory, train_x, train_y, val_x, val_y, seed, alpha, protocol, classes=3
):
    arrays = _arrays(train_x, train_y, val_x, val_y, classes)
    train_x, train_y, val_x, val_y = arrays
    directory, p = Path(directory), _protocol(protocol)
    norm = load_npz(directory / "normalization.npz")
    expected = normalize(train_x)
    if set(norm) != set(expected):
        raise ValueError("normalization inventory differs")
    for key in expected:
        equal(norm[key], expected[key], "train-only normalization")
    record = read(directory / "selection.json")
    count = (train_x.shape[1] + 1) * WIDTH + (WIDTH + 1) * WIDTH + (WIDTH + 1) * classes
    fields = dict(
        width=WIDTH,
        depth=2,
        classes=classes,
        seed=seed,
        alpha=float(alpha),
        protocol=p,
        parameters=count,
        input_sha256=_bindings(arrays),
    )
    if any(record.get(key) != value for key, value in fields.items()):
        raise ValueError("architecture, protocol, L2, or input binding differs")
    steps = candidate_steps(p["updates"])
    if [r["update"] for r in record["candidates"]] != steps:
        raise ValueError("candidate inventory differs")
    if record["selected"] != select(record["candidates"]):
        raise ValueError("selection is not earliest validation minimum")
    inventory = {"normalization.npz", "selection.json"} | {
        f"{prefix}-{step:05d}.{suffix}"
        for step in steps
        for prefix, suffix in (("checkpoint", "pkl"), ("candidate", "npz"))
    }
    if {path.name for path in directory.iterdir()} != inventory:
        raise ValueError("readout artifact inventory differs")
    gradients = [s for s in range(1, p["updates"] + 1) if s == 1 or s in steps]
    if [r["update"] for r in record["gradients"]] != gradients:
        raise ValueError("gradient inventory differs")
    finite(record["gradients"])
    for row in record["candidates"]:
        step = row["update"]
        if row["checkpoint"] != f"checkpoint-{step:05d}.pkl":
            raise ValueError("invalid checkpoint path")
        params = load(directory / row["checkpoint"])
        shapes = [(train_x.shape[1], WIDTH), (WIDTH, WIDTH), (WIDTH, classes)]
        if len(params) != 3 or any(
            set(layer) != {"w", "b"}
            or layer["w"].shape != shape
            or layer["b"].shape != (shape[1],)
            for layer, shape in zip(params, shapes)
        ):
            raise ValueError("checkpoint architecture differs")
        finite(params)
        if step == 0:
            for actual, want in zip(
                jax.tree.leaves(params),
                jax.tree.leaves(
                    init(jax.random.PRNGKey(seed), train_x.shape[1], classes)
                ),
            ):
                equal(actual, want, "initialization")
        saved = load_npz(directory / f"candidate-{step:05d}.npz")
        if set(saved) != {"train", "validation"}:
            raise ValueError("candidate prediction inventory differs")
        for name, xx, yy in (("train", train_x, train_y), ("validation", val_x, val_y)):
            prediction = predict(params, norm, xx)
            equal(prediction, saved[name], "independent candidate " + name)
            if scores(prediction, yy) != row[name]:
                raise ValueError("candidate metrics differ")
    return record


def select_family(records):
    """Choose alpha by mean seedwise validation-selected NLL, never test error."""
    if not records:
        raise ValueError("no regularization families")
    families, expected_seeds, seen = [], None, set()
    for key, rows in records.items():
        alpha = float(key)
        if not np.isfinite(alpha) or alpha < 0 or alpha in seen or not rows:
            raise ValueError("invalid or duplicate regularization family")
        seen.add(alpha)
        seeds = sorted(r["seed"] for r in rows)
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate initialization seeds")
        if expected_seeds is not None and seeds != expected_seeds:
            raise ValueError("regularization families have unmatched seeds")
        expected_seeds = seeds
        selections = {}
        for row in rows:
            selected = select(row["candidates"])
            if row["alpha"] != alpha or row["selected"] != selected:
                raise ValueError("family record or selection differs")
            selections[str(row["seed"])] = selected
        families.append(
            dict(
                alpha=alpha,
                selections=selections,
                mean_validation_nll=float(
                    np.mean([r["validation"]["nll"] for r in selections.values()])
                ),
            )
        )
    best = min(families, key=lambda r: (r["mean_validation_nll"], r["alpha"]))
    return dict(best, families=sorted(families, key=lambda r: r["alpha"]))


def _ridge_spec(protocol, dimension):
    ranks = protocol.get("ridge_ranks", [16, 64, 128])
    alphas = protocol.get("ridge_alphas", [10.0**k for k in range(-6, 3)])
    if (
        any(not isinstance(r, int) or isinstance(r, bool) or r < 1 for r in ranks)
        or not alphas
        or any(not np.isfinite(a) or a <= 0 for a in alphas)
    ):
        raise ValueError("invalid ridge candidate grid")
    ranks = sorted({r for r in ranks if r < dimension}) + [dimension]
    return dict(
        ranks=ranks,
        alphas=sorted({float(a) for a in alphas}),
        selection="unclipped validation MSE; ties smaller rank then larger alpha",
    )


def _ridge_transform(train_x, train_y):
    # Float64 centering makes the intercept unpenalized even after float32 input.
    x = np.asarray(train_x, np.float64)
    mean, std = x.mean(0), np.maximum(x.std(0), 0.01)
    z = (x - mean) / std
    target_mean = float(np.mean(train_y))
    covariance = z.T @ z / len(z)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues, vectors = np.maximum(eigenvalues[::-1], 0), vectors[:, ::-1]
    cross = z.T @ (train_y - target_mean) / len(z)
    return dict(
        mean=mean,
        std=std,
        eigenvalues=eigenvalues,
        vectors=vectors,
        cross=cross,
        target_mean=np.asarray(target_mean),
    )


def _ridge_parameters(transform, rank, alpha):
    vectors = transform["vectors"][:, :rank]
    coefficient = vectors @ (
        (vectors.T @ transform["cross"]) / (transform["eigenvalues"][:rank] + alpha)
    )
    return dict(coefficient=coefficient, intercept=transform["target_mean"])


def _ridge_predict(parameters, transform, x):
    # Fitting binds float32 features; inference must use the same conversion.
    x = np.asarray(x, np.float32).astype(np.float64)
    if (
        x.ndim != 2
        or x.shape[1:] != transform["mean"].shape
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid ridge prediction features")
    result = (x - transform["mean"]) / transform["std"] @ parameters[
        "coefficient"
    ] + parameters["intercept"]
    finite(result)
    return result


def _regression_scores(raw, y):
    return dict(
        mse=float(np.mean((raw - y) ** 2)),
        clipped_mse=float(np.mean((np.clip(raw, 0, 2) - y) ** 2)),
    )


def _select_ridge(rows):
    if not rows or any(not np.isfinite(r["validation"]["mse"]) for r in rows):
        raise ValueError("invalid ridge scores")
    return min(rows, key=lambda r: (r["validation"]["mse"], r["rank"], -r["alpha"]))


def fit_ridge(directory, train_x, train_y, val_x, val_y, protocol):
    started = time.perf_counter()
    arrays = _arrays(train_x, train_y, val_x, val_y)
    train_x, train_y, val_x, val_y = arrays
    spec = _ridge_spec(protocol, train_x.shape[1])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    transform = _ridge_transform(train_x, train_y)
    _write_npz(directory / "ridge-transform.npz", transform)
    candidates = []
    for rank in spec["ranks"]:
        for alpha in spec["alphas"]:
            index = len(candidates)
            name = f"ridge-{index:03d}.npz"
            parameters = _ridge_parameters(transform, rank, alpha)
            _write_npz(directory / name, parameters)
            train = _ridge_predict(parameters, transform, train_x)
            validation = _ridge_predict(parameters, transform, val_x)
            _write_npz(
                directory / f"ridge-predictions-{index:03d}.npz",
                dict(
                    train=train,
                    train_clipped=np.clip(train, 0, 2),
                    validation=validation,
                    validation_clipped=np.clip(validation, 0, 2),
                ),
            )
            candidates.append(
                dict(
                    index=index,
                    rank=rank,
                    projection="full" if rank == train_x.shape[1] else "PCA",
                    alpha=alpha,
                    checkpoint=name,
                    train=_regression_scores(train, train_y),
                    validation=_regression_scores(validation, val_y),
                )
            )
    record = dict(
        selected=_select_ridge(candidates),
        candidates=candidates,
        protocol=spec,
        input_sha256=_bindings(arrays),
        prediction_type="continuous expectation; clipped version restricted to [0,2]",
        fitting_seconds_including_decomposition=time.perf_counter() - started,
    )
    _write_json(directory / "ridge-selection.json", record)
    print("REWARD_RMT_RIDGE_FITTED", directory.name, flush=True)
    return record


def predict_ridge(directory, x):
    directory = Path(directory)
    record = read(directory / "ridge-selection.json")
    selected = _select_ridge(record["candidates"])
    if (
        record["selected"] != selected
        or selected["checkpoint"] != f"ridge-{selected['index']:03d}.npz"
    ):
        raise ValueError("invalid ridge selection")
    parameters = load_npz(directory / selected["checkpoint"])
    return _ridge_predict(parameters, load_npz(directory / "ridge-transform.npz"), x)


def verify_ridge(directory, train_x, train_y, val_x, val_y, protocol):
    arrays = _arrays(train_x, train_y, val_x, val_y)
    train_x, train_y, val_x, val_y = arrays
    directory = Path(directory)
    record = read(directory / "ridge-selection.json")
    spec = _ridge_spec(protocol, train_x.shape[1])
    if record["protocol"] != spec or record["input_sha256"] != _bindings(arrays):
        raise ValueError("ridge protocol or input binding differs")
    expected = _ridge_transform(train_x, train_y)
    transform = load_npz(directory / "ridge-transform.npz")
    if set(transform) != set(expected):
        raise ValueError("ridge transform inventory differs")
    for key in expected:
        equal(transform[key], expected[key], "training-only ridge transform " + key)
    grid = [(rank, alpha) for rank in spec["ranks"] for alpha in spec["alphas"]]
    if len(grid) != len(record["candidates"]):
        raise ValueError("ridge candidate inventory differs")
    if record["selected"] != _select_ridge(record["candidates"]):
        raise ValueError("ridge selection differs")
    inventory = {"ridge-selection.json", "ridge-transform.npz"} | {
        f"{prefix}-{index:03d}.npz"
        for index in range(len(grid))
        for prefix in ("ridge", "ridge-predictions")
    }
    if {path.name for path in directory.iterdir()} != inventory:
        raise ValueError("ridge artifact inventory differs")
    for index, (row, (rank, alpha)) in enumerate(zip(record["candidates"], grid)):
        metadata = dict(
            index=index,
            rank=rank,
            alpha=alpha,
            projection="full" if rank == train_x.shape[1] else "PCA",
            checkpoint=f"ridge-{index:03d}.npz",
        )
        if any(row.get(key) != value for key, value in metadata.items()):
            raise ValueError("ridge candidate metadata differs")
        parameters = _ridge_parameters(expected, rank, alpha)
        saved_params = load_npz(directory / row["checkpoint"])
        if set(saved_params) != set(parameters):
            raise ValueError("ridge parameter inventory differs")
        for key in parameters:
            equal(saved_params[key], parameters[key], "independent ridge coefficients")
        saved = load_npz(directory / f"ridge-predictions-{index:03d}.npz")
        if set(saved) != {"train", "train_clipped", "validation", "validation_clipped"}:
            raise ValueError("ridge prediction inventory differs")
        for name, xx, yy in (("train", train_x, train_y), ("validation", val_x, val_y)):
            raw = _ridge_predict(parameters, expected, xx)
            equal(saved[name], raw, "independent ridge " + name)
            equal(
                saved[name + "_clipped"],
                np.clip(raw, 0, 2),
                "independent clipped ridge " + name,
            )
            if _regression_scores(raw, yy) != row[name]:
                raise ValueError("ridge metrics differ")
    return record
