"""Matched categorical probes: no parent arrays enter the optimizer."""

from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .parallel_collection import _write_json, _write_npz
from .parallel_runner import finite, load_npz
from .prior_correction_study import dump, load
from .reward_readout_study import init_mlp, mlp_logits, read
from .reward_diagnosis_metrics import nll


def fit_head(output, train_x, train_y, val_x, val_y, seed, protocol):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    x = np.asarray(train_x, np.float32).reshape(-1, train_x.shape[-1])
    y = np.asarray(train_y).reshape(-1).astype(np.int32)
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(0, dtype=np.float64), 0.01).astype(np.float32)
    _write_npz(output / "normalization.npz", dict(mean=mean, std=std))
    x = jnp.asarray((x - mean) / std)
    y = jnp.asarray(y)
    vx = jnp.asarray((val_x - mean) / std)
    params = init_mlp(jax.random.PRNGKey(seed), x.shape[-1])
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

    predict = jax.jit(lambda p, x: jax.nn.softmax(mlp_logits(p, x), -1))
    # Deliberately identical across groups AND seeds; seed only changes weights.
    rng = np.random.default_rng(protocol["minibatch_seed"])
    records, logs = [], []
    for step in range(1, protocol["updates"] + 1):
        ix = rng.integers(len(x), size=protocol["batch_size"])
        params, state, loss, grad = update(params, state, x[ix], y[ix])
        finite((params, state, loss, grad))
        if step == 1 or step % 250 == 0:
            logs.append(dict(update=step, loss=float(loss), gradient_norm=float(grad)))
        if step % protocol["validation_period"]:
            continue
        probs = np.asarray(predict(params, vx))
        value = float(nll(probs, val_y).mean())
        name = f"checkpoint-{step:05d}.pkl"
        dump(output / name, params)
        _write_npz(output / f"validation-{step:05d}.npz", dict(probabilities=probs))
        records.append(dict(update=step, nll=value, checkpoint=name))
    selected = min(records, key=lambda r: (r["nll"], r["update"]))
    _write_json(
        output / "selection.json",
        dict(
            selected=selected,
            candidates=records,
            training=logs,
            parameters=sum(v.size for v in jax.tree.leaves(params)),
            seed=seed,
            minibatch_seed=protocol["minibatch_seed"],
            fitting_seconds_including_compilation=time.perf_counter() - started,
        ),
    )
    print("REWARD_DIAGNOSIS_HEAD_FITTED", output.name, selected, flush=True)


class Head:
    def __init__(self, directory):
        directory = Path(directory)
        self.selection = read(directory / "selection.json")
        self.params = load(directory / self.selection["selected"]["checkpoint"])
        self.norm = load_npz(directory / "normalization.npz")
        self._predict = jax.jit(
            lambda x: jax.nn.softmax(
                mlp_logits(self.params, (x - self.norm["mean"]) / self.norm["std"]), -1
            )
        )

    def __call__(self, features):
        shape = features.shape[:-1]
        x = np.asarray(features, np.float32).reshape(-1, features.shape[-1])
        # Bound device memory and make create/replay chunk shapes identical.
        p = np.concatenate(
            [
                np.asarray(self._predict(jnp.asarray(x[i : i + 1024])))
                for i in range(0, len(x), 1024)
            ]
        )
        finite(p)
        return p.reshape((*shape, 3))
