"""Matched fresh control-readout fitting and shared offline exploration warmup.

No parent model is instantiated, no parent weights initialize these heads, and
no simulator is used. The existing head is the original upstream architecture,
objective, input casting, and raw expectation decoder in a fresh namespace.
Only train rows enter optimizer updates or normalizers. Validation chooses
checkpoints then initialization seed, with the earliest checkpoint/seed breaking
ties. Test diagnostics are computed only after selection has been persisted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .parallel_collection import _atomic_artifact, _write_json, _write_npz
from .parallel_runner import load_npz
from .reward_readout_study import (
    equal,
    first_minimum,
    sha,
    validate_data,
    validation_score,
)

KINDS = ("existing", "categorical")
MATCH_DEFAULTS = dict(
    seeds=[0, 1, 2],
    updates=5000,
    learning_rate=3e-4,
    batch_size=256,
    validation_period=250,
    minibatch_seed=6211,
)
ENSEMBLE_DEFAULTS = dict(
    members=5,
    updates=5000,
    learning_rate=3e-4,
    batch_size=256,
    seed=27183,
    bootstrap_seed=9191,
    calibration_quantile=0.95,
    calibration_floor=1e-4,
    bonus_clip=10.0,
)
CONTROL_FIXED = ("control_rew/input_mean", "control_rew/input_std")
ENSEMBLE_FIXED = tuple(
    "explore_ensemble/" + key
    for key in ("input_mean", "input_std", "target_mean", "target_std", "bonus_scale")
)


def _read(path):
    return json.loads(Path(path).read_text())


def _dump(path, value):
    _atomic_artifact(Path(path), lambda stream: pickle.dump(value, stream, protocol=5))


def _load(path):
    # Artifacts are local trusted experiment outputs authenticated before use.
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def load_export(match_output, kind):
    """Authenticated raw auxiliary entries for a parent-state merge.

    This does not load/rewrite any parent entry or create optimizer state. The
    runner owns parent restore and must authenticate the enclosing stage marker.
    Full numerical replay is available separately through ``verify_match``.
    """
    if kind not in KINDS:
        raise ValueError("unknown exported control kind")
    output = Path(match_output)
    marker = _read(output / "match_completed.json")
    if marker.get("stage") != "match" or marker.get("additional_simulator_steps") != 0:
        raise ValueError("invalid matched completion marker")
    for name in ("exports.json", "manifest.json"):
        if marker["artifacts"].get(name) != sha(output / name):
            raise ValueError("matched export manifest binding differs")
    exports = _read(output / "exports.json")
    protocol_settings(_read(output / "manifest.json")["protocol"])
    expected = {"control": f"{kind}_control.pkl", "ensemble": "ensemble/ensemble.pkl"}
    records = {"control": exports["control"][kind], "ensemble": exports["ensemble"]}
    payloads = {}
    for name, record in records.items():
        if record["path"] != expected[name]:
            raise ValueError("unexpected matched export path")
        digest = sha(output / record["path"])
        if (
            record["sha256"] != digest
            or marker["artifacts"].get(record["path"]) != digest
        ):
            raise ValueError("matched export parameter binding differs")
        payloads[name] = _load(output / record["path"])
    control, ensemble = payloads["control"], payloads["ensemble"]
    if control["kind"] != kind or ensemble["kind"] != "ensemble":
        raise ValueError("matched export kind differs")
    for payload, prefix in ((control, "control_rew/"), (ensemble, "explore_ensemble/")):
        if not payload["params"] or any(
            not k.startswith(prefix) for k in payload["params"]
        ):
            raise ValueError("export contains non-auxiliary parameter namespace")
        if not all(np.isfinite(v).all() for v in payload["params"].values()):
            raise ValueError("nonfinite exported parameters")
    if ensemble["bootstrap_seed"] != ENSEMBLE_DEFAULTS["bootstrap_seed"]:
        raise ValueError("ensemble bootstrap seed differs")
    return {
        **control["params"],
        **ensemble["params"],
        "explore_bootstrap_seed/value": np.asarray(
            ensemble["bootstrap_seed"], np.int32
        ),
    }


def protocol_settings(protocol):
    """Production budgets are frozen; tiny test fits use lower-level APIs."""
    match = protocol.get("matched_fit", MATCH_DEFAULTS)
    ensemble = protocol.get("ensemble_fit", ENSEMBLE_DEFAULTS)
    if match != MATCH_DEFAULTS or ensemble != ENSEMBLE_DEFAULTS:
        raise ValueError("matched reward/ensemble protocol differs from frozen budgets")
    return dict(match), dict(ensemble)


def split_indices(data):
    validate_data(data)
    if data["observations"].shape != (1024, 15, 6):
        raise ValueError("unexpected observation coordinates")
    if data["actions"].shape != (1024, 15, 2) or data["start"].shape != (1024, 2560):
        raise ValueError("unexpected current belief or action coordinates")
    return tuple(np.flatnonzero(data["split"] == split) for split in range(3))


def train_normalization(data, train):
    """Common posterior-belief and target-observation train-only coordinates."""
    if not len(train) or not np.all(data["split"][train] == 0):
        raise ValueError("normalization requires training rows only")
    norm = {}
    for label, key in (("belief", "targets"), ("observation", "observations")):
        value = np.asarray(data[key][train], np.float64).reshape(
            -1, data[key].shape[-1]
        )
        if not np.isfinite(value).all():
            raise ValueError("nonfinite training coordinates")
        norm[label + "_mean"] = value.mean(0).astype(np.float32)
        norm[label + "_std"] = np.maximum(value.std(0), 0.01).astype(np.float32)
    norm["mean"], norm["std"] = norm["belief_mean"], norm["belief_std"]
    return norm


def minibatch_indices(ntrain, settings):
    if ntrain < 1 or settings["updates"] < 1 or settings["batch_size"] < 1:
        raise ValueError("invalid matched minibatch dimensions")
    return np.random.default_rng(settings["minibatch_seed"]).integers(
        ntrain, size=(settings["updates"], settings["batch_size"]), dtype=np.int32
    )


def array_digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(
        str((value.shape, value.dtype)).encode() + value.tobytes()
    ).hexdigest()


def select_primary(records):
    """Input order is frozen seed order, and every score is validation-only."""
    return {
        kind: rows[first_minimum(rows)]
        for kind in KINDS
        for rows in ([r for r in records if r["kind"] == kind],)
    }


def _head_functions(kind, config):
    import jax.numpy as jnp
    import ninjax as nj
    from embodied.jax import nets
    from .reward_exploration_agent import build_control_head

    head = build_control_head(kind, config)

    def predict(features):
        return head(nets.cast(features), features.ndim - 1).pred().astype(jnp.float32)

    def loss(features, target):
        return head(nets.cast(features), features.ndim - 1).loss(target).mean()

    return nj.pure(predict), nj.pure(loss)


def initialize_control(kind, config, dimension, seed, normalization):
    import jax
    import jax.numpy as jnp

    predict, _ = _head_functions(kind, config)
    params, _ = predict(
        {},
        jnp.zeros((1, dimension), jnp.float32),
        seed=jax.random.PRNGKey(seed),
        create=True,
        modify=True,
    )
    if kind == "categorical":
        for name, key in zip(CONTROL_FIXED, ("mean", "std")):
            if name not in params or params[name].shape != (dimension,):
                raise ValueError("shared control normalizer key/shape mismatch")
            params[name] = jnp.asarray(normalization[key], jnp.float32)
    if not params or any(not k.startswith("control_rew/") for k in params):
        raise ValueError("non-isolated control-head namespace")
    return params


def predictor(kind, config):
    import jax

    pure, _ = _head_functions(kind, config)
    return jax.jit(lambda params, features: pure(params, features, seed=0)[1])


def _optimizer(params, loss, fixed_keys, learning_rate):
    """Fixed coordinate variables never enter Adam (including its moments)."""
    import jax
    import optax

    fixed = {k: v for k, v in params.items() if k in fixed_keys}
    weights = {k: v for k, v in params.items() if k not in fixed_keys}
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(weights)

    @jax.jit
    def update(weights, state, *batch):
        value, grad = jax.value_and_grad(lambda p: loss({**fixed, **p}, *batch))(
            weights
        )
        changes, state = optimizer.update(grad, state, weights)
        weights = optax.apply_updates(weights, changes)
        return weights, state, value, optax.global_norm(grad)

    return weights, fixed, state, update


def fit_candidate(
    directory,
    kind,
    seed,
    config,
    xtrain,
    ytrain,
    xval,
    yval,
    normalization,
    indices,
    scale,
    settings,
):
    """Train a fresh module; intentionally receives no test tensors or parent weights."""
    import jax
    import jax.numpy as jnp

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if indices.shape != (settings["updates"], settings["batch_size"]):
        raise ValueError("wrong shared minibatch schedule")
    flat = np.asarray(xtrain).reshape(-1, xtrain.shape[-1])
    if indices.min() < 0 or indices.max() >= len(flat):
        raise ValueError("minibatch index outside train split")
    x, y = jnp.asarray(flat), jnp.asarray(ytrain.reshape(-1), jnp.float32)
    vx = jnp.asarray(xval)
    params = initialize_control(kind, config, x.shape[-1], seed, normalization)
    _dump(directory / "initial.pkl", jax.device_get(params))
    _, pure_loss = _head_functions(kind, config)
    weights, fixed, state, update = _optimizer(
        params,
        lambda p, xx, yy: pure_loss(p, xx, yy, seed=0)[1],
        CONTROL_FIXED,
        settings["learning_rate"],
    )
    predict = predictor(kind, config)
    logs, records = [], []
    begun = time.monotonic()
    for step, ix in enumerate(indices, 1):
        weights, state, loss, grad = update(weights, state, x[ix], y[ix])
        # Every update must be finite, not merely those retained in the log.
        values = np.asarray([loss, grad], np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("nonfinite matched reward update")
        if step == 1 or step % 100 == 0 or step == settings["updates"]:
            logs.append(
                dict(update=step, loss=float(values[0]), gradient_norm=float(values[1]))
            )
        if step % settings["validation_period"]:
            continue
        params = {**fixed, **weights}
        pred = np.asarray(predict(params, vx))
        name = f"checkpoint_{step:05d}.pkl"
        _dump(directory / name, jax.device_get(params))
        _write_npz(directory / f"validation_{step:05d}.npz", dict(prediction=pred))
        records.append(
            dict(
                update=step, score=validation_score(pred, yval, scale), checkpoint=name
            )
        )
        print(
            "REWARD_EXPLORATION_MATCH_PROGRESS",
            kind,
            seed,
            step,
            records[-1]["score"],
            flush=True,
        )
    selected = records[first_minimum(records)]
    record = dict(kind=kind, seed=seed, directory=directory.name, **selected)
    _write_json(
        directory / "selection.json",
        dict(
            selected=record,
            validations=records,
            training=logs,
            updates_completed=len(indices),
            minibatch_sha256=array_digest(indices),
            seconds=time.monotonic() - begun,
            initialization="fresh independent seed; no parent weights",
        ),
    )
    return record


def export_control(output, record, config, normalization, validation_features):
    """Export raw Ninjax names used by online control, and prove exact replay."""
    import jax

    params = _load(Path(output) / record["directory"] / record["checkpoint"])
    payload = dict(
        kind=record["kind"],
        params=params,
        normalization={k: normalization[k] for k in ("mean", "std")},
        selection=record,
        initialization="matched fresh fit",
        normalization_applied=record["kind"] == "categorical",
    )
    path = Path(output) / (record["kind"] + "_control.pkl")
    _dump(path, payload)
    replay = _load(path)
    actual = np.asarray(
        predictor(record["kind"], config)(
            jax.tree.map(jax.device_put, replay["params"]),
            jax.device_put(validation_features),
        )
    )
    saved = load_npz(
        Path(output) / record["directory"] / f"validation_{record['update']:05d}.npz"
    )
    equal(actual, saved["prediction"], record["kind"] + " exported validation replay")
    return dict(path=path.name, sha256=sha(path), kind=record["kind"], selection=record)


def transition_arrays(data, rows):
    """Branch index j maps the preceding belief and actual action to next obs j."""
    features = np.concatenate(
        [data["start"][rows, None], data["targets"][rows, :-1]], axis=1
    )
    return features, data["actions"][rows], data["observations"][rows]


def episode_bootstrap(episodes, members, seed):
    from .reward_exploration_agent import episode_bootstrap as shared_bootstrap

    episodes = np.asarray(episodes)
    if members != 5:
        raise ValueError("shared ensemble requires five members")
    unique, inverse = np.unique(episodes, return_inverse=True)
    masks = np.asarray(shared_bootstrap(unique, seed), np.float32)
    # Fail closed instead of data-dependent resampling if a member is empty.
    if not masks.any(1).all():
        raise ValueError("empty episode bootstrap member")
    return unique, masks, masks[:, inverse]


def _ensemble_function():
    import ninjax as nj
    from .reward_exploration_agent import build_ensemble

    model = build_ensemble()
    return nj.pure(lambda features, actions: model(features, actions))


def ensemble_predictor():
    import jax

    pure = _ensemble_function()
    return jax.jit(
        lambda params, features, actions: pure(params, features, actions, seed=0)[1]
    )


def disagreement(prediction):
    from .reward_exploration_agent import predicted_mean_disagreement

    value = np.asarray(prediction, np.float64)
    if value.shape[0] != 5 or value.shape[-1] != 6 or not np.isfinite(value).all():
        raise ValueError("invalid normalized ensemble predictions")
    # The calibration uses exactly the same float32 reduction as online bonus.
    return np.asarray(predicted_mean_disagreement(np.asarray(prediction, np.float32)))


def _association(disagreement_values, error):
    from scipy.stats import rankdata

    x, y = np.asarray(disagreement_values).reshape(-1), np.asarray(error).reshape(-1)

    def corr(a, b):
        return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])

    return dict(
        pearson=corr(x, y),
        spearman=corr(rankdata(x), rankdata(y)),
        transitions=len(x),
        mean_error=float(y.mean()),
        mean_disagreement=float(x.mean()),
    )


def ensemble_diagnostics(pred, truth_normalized, control_predictions):
    error = ((np.asarray(pred, np.float64).mean(0) - truth_normalized) ** 2).mean(-1)
    result = dict(actual=_association(disagreement(pred), error), controls={})
    for name, prediction in control_predictions.items():
        control_error = (
            (np.asarray(prediction, np.float64).mean(0) - truth_normalized) ** 2
        ).mean(-1)
        result["controls"][name] = _association(disagreement(prediction), control_error)
    result["interpretation"] = (
        "Controls reuse actual next observations; mismatch errors are not counterfactual accuracy. No tuning or favorable-association requirement."
    )
    return result


def fit_ensemble(directory, data, train, normalization, settings):
    import jax
    import jax.numpy as jnp

    if not np.all(data["split"][train] == 0):
        raise ValueError("ensemble requires training rows only")
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    features, actions, targets = transition_arrays(data, train)
    pure = _ensemble_function()
    params, _ = pure(
        {},
        jnp.zeros((1, features.shape[-1]), jnp.float32),
        jnp.zeros((1, actions.shape[-1]), jnp.float32),
        seed=settings["seed"],
        create=True,
        modify=True,
    )
    values = (
        normalization["belief_mean"],
        normalization["belief_std"],
        normalization["observation_mean"],
        normalization["observation_std"],
        np.float32(1.0),
    )
    for name, value in zip(ENSEMBLE_FIXED, values):
        if name not in params or params[name].shape != np.shape(value):
            raise ValueError("shared ensemble normalizer key/shape mismatch: " + name)
        params[name] = jnp.asarray(value, jnp.float32)
    _dump(directory / "initial.pkl", jax.device_get(params))
    episodes, masks, by_row = episode_bootstrap(
        data["episode"][train], settings["members"], settings["bootstrap_seed"]
    )
    flat_masks = np.repeat(by_row, features.shape[1], axis=1)
    _write_npz(directory / "bootstrap.npz", dict(episodes=episodes, masks=masks))
    sampled = minibatch_indices(
        features.shape[0] * features.shape[1],
        dict(settings, minibatch_seed=settings["seed"] + 1),
    )
    _write_npz(directory / "minibatches.npz", dict(indices=sampled))
    x = jnp.asarray(features.reshape(-1, features.shape[-1]))
    a = jnp.asarray(actions.reshape(-1, actions.shape[-1]))
    y = jnp.asarray(
        (
            (targets - normalization["observation_mean"])
            / normalization["observation_std"]
        ).reshape(-1, targets.shape[-1])
    )
    m = jnp.asarray(flat_masks)

    def loss(p, xx, aa, yy, mm):
        prediction = pure(p, xx, aa, seed=0)[1]
        mse = ((prediction - yy[None]) ** 2).mean(-1)
        return ((mse * mm).sum(-1) / jnp.maximum(mm.sum(-1), 1)).mean()

    weights, fixed, state, update = _optimizer(
        params, loss, ENSEMBLE_FIXED, settings["learning_rate"]
    )
    logs = []
    for step, ix in enumerate(sampled, 1):
        weights, state, value, gradient = update(
            weights, state, x[ix], a[ix], y[ix], m[:, ix]
        )
        measures = np.asarray([value, gradient], np.float64)
        if not np.isfinite(measures).all():
            raise FloatingPointError("nonfinite shared ensemble update")
        if step == 1 or step % 100 == 0 or step == len(sampled):
            logs.append(
                dict(
                    update=step,
                    loss=float(measures[0]),
                    gradient_norm=float(measures[1]),
                )
            )
            print(
                "REWARD_EXPLORATION_ENSEMBLE_PROGRESS",
                step,
                float(measures[0]),
                flush=True,
            )
    params = {**fixed, **weights}
    predict = ensemble_predictor()
    train_prediction = np.asarray(
        predict(params, jnp.asarray(features), jnp.asarray(actions))
    )
    train_disagreement = disagreement(train_prediction)
    scale = max(
        float(np.quantile(train_disagreement, settings["calibration_quantile"])),
        settings["calibration_floor"],
    )
    params["explore_ensemble/bonus_scale"] = jnp.asarray(scale, jnp.float32)
    # Record the actual float32 value consumed online, not an unrepresentable scalar.
    scale = float(np.asarray(params["explore_ensemble/bonus_scale"]))
    _write_npz(
        directory / "train_calibration.npz",
        dict(
            indices=train, prediction=train_prediction, disagreement=train_disagreement
        ),
    )
    payload = dict(
        kind="ensemble",
        params=jax.device_get(params),
        normalization=normalization,
        scale=scale,
        bonus_clip=settings["bonus_clip"],
        bootstrap_seed=settings["bootstrap_seed"],
    )
    _dump(directory / "ensemble.pkl", payload)
    _write_json(
        directory / "training.json",
        dict(
            updates_completed=len(sampled),
            logs=logs,
            bootstrap_unit="retained episode",
            scale=scale,
            calibration_split="train",
            quantile=settings["calibration_quantile"],
        ),
    )
    return payload


def _heldout_ensemble(output, data, rows, payload, split):
    import jax

    features, actions, observations = transition_arrays(data, rows)
    rng = np.random.default_rng(7171 + split)
    permutation = rng.permutation(actions.size // actions.shape[-1])
    shuffled = actions.reshape(-1, actions.shape[-1])[permutation].reshape(
        actions.shape
    )
    random_actions = rng.uniform(-1.0, 1.0, actions.shape).astype(np.float32)
    predict = ensemble_predictor()
    params = jax.tree.map(jax.device_put, payload["params"])
    pred = np.asarray(predict(params, features, actions))
    controls = {
        name: np.asarray(predict(params, features, action))
        for name, action in (
            ("shuffled_actions", shuffled),
            ("uniform_actions", random_actions),
        )
    }
    norm = payload["normalization"]
    target = (observations - norm["observation_mean"]) / norm["observation_std"]
    values = dict(
        indices=rows,
        prediction=pred,
        truth_normalized=target,
        shuffled_permutation=permutation,
        random_actions=random_actions,
        **controls,
    )
    _write_npz(Path(output) / f"heldout_{split}.npz", values)
    return ensemble_diagnostics(pred, target, controls)


def _config(parent_cell):
    import elements
    from ruamel.yaml import YAML

    value = YAML(typ="safe").load((Path(parent_cell) / "config.yaml").read_text())
    return elements.Config(value["agent"])


def _reward_metrics(data, rows, predictions, primary, scale):
    from .reward_probe_metrics import summarize

    result = {}
    for kind in KINDS:
        pred = predictions[kind]
        result[kind] = summarize(
            pred,
            data["rewards"][rows],
            data["episode"][rows],
            data["mode"][rows],
            data["plan"][rows],
            scale,
            baseline=predictions["existing"],
            bootstrap_reps=2000,
        )
    candidates = {}
    for kind in KINDS:
        for seed in MATCH_DEFAULTS["seeds"]:
            name = f"{kind}_{seed}"
            if name not in predictions:
                continue
            pred, truth = (
                np.asarray(predictions[name], np.float64),
                data["rewards"][rows],
            )
            candidates[name] = dict(
                raw_step_mse=float(np.mean((pred - truth) ** 2)),
                raw_step_bias=float(np.mean(pred - truth)),
                normalized_cumulative_15_mse=validation_score(pred, truth, scale),
            )
    return dict(
        selected=primary,
        test=result,
        candidate_test=candidates,
        additional_simulator_steps=0,
        matched_comparison="fresh matched control-readout package; not pure architecture",
        limitation="Historical test exposure makes this exploratory; no test tuning in this run.",
    )


def run_match(output, dataset, parent_cell, protocol):
    """Run in configured upstream/JAX runtime. output is a NEW match directory."""
    settings, ensemble_settings = protocol_settings(protocol)
    output, dataset, parent_cell = Path(output), Path(dataset), Path(parent_cell)
    checkpoint = parent_cell / protocol["parent"]["checkpoint_name"]
    dependencies = {
        str(p.resolve()): sha(p)
        for p in (dataset, checkpoint, parent_cell / "config.yaml")
    }
    if sha(checkpoint) != protocol["parent"]["checkpoint_sha256"]:
        raise ValueError("wrong matched-fit parent checkpoint")
    if protocol.get("dataset_sha256") and sha(dataset) != protocol["dataset_sha256"]:
        raise ValueError("wrong matched-fit retained dataset")
    data = load_npz(dataset)
    train, val, test = split_indices(data)
    normalization = train_normalization(data, train)
    scale = max(
        float(np.sqrt(np.mean(data["rewards"][train].astype(np.float64).sum(-1) ** 2))),
        0.01,
    )
    config = _config(parent_cell)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "manifest.json",
        dict(
            dependencies=dependencies,
            parent_cell=str(parent_cell.resolve()),
            dataset=str(dataset.resolve()),
            protocol=protocol,
            started=time.time(),
            additional_simulator_steps=0,
        ),
    )
    _write_npz(output / "normalization.npz", normalization)
    sampled = minibatch_indices(len(train) * data["rewards"].shape[1], settings)
    _write_npz(output / "minibatches.npz", dict(indices=sampled, train_rows=train))
    records = []
    for kind in KINDS:
        for seed in settings["seeds"]:
            records.append(
                fit_candidate(
                    output / f"{kind}_{seed}",
                    kind,
                    seed,
                    config,
                    data["targets"][train],
                    data["rewards"][train],
                    data["targets"][val],
                    data["rewards"][val],
                    normalization,
                    sampled,
                    scale,
                    settings,
                )
            )
    primary = select_primary(records)
    # This immutable artifact is persisted before accessing held-out predictions.
    _write_json(
        output / "selection.json",
        dict(
            records=records,
            primary=primary,
            scale=scale,
            train_episodes=np.unique(data["episode"][train]).tolist(),
            validation_episodes=np.unique(data["episode"][val]).tolist(),
            selection_rule="minimum validation normalized 15-step cumulative MSE; earliest checkpoint within seed, then earliest seed",
        ),
    )
    exports = {
        kind: export_control(
            output, primary[kind], config, normalization, data["targets"][val]
        )
        for kind in KINDS
    }
    predictions = {}
    for record in records:
        params = _load(output / record["directory"] / record["checkpoint"])
        predictions[record["directory"]] = np.asarray(
            predictor(record["kind"], config)(params, data["targets"][test])
        )
    predictions.update(
        {kind: predictions[primary[kind]["directory"]] for kind in KINDS}
    )
    _write_npz(
        output / "test_predictions.npz",
        dict(indices=test, truth=data["rewards"][test], **predictions),
    )
    ensemble = fit_ensemble(
        output / "ensemble", data, train, normalization, ensemble_settings
    )
    associations = {
        str(split): _heldout_ensemble(output / "ensemble", data, rows, ensemble, split)
        for split, rows in ((1, val), (2, test))
    }
    report = _reward_metrics(data, test, predictions, primary, scale)
    report.update(ensemble_association=associations, ensemble_scale=ensemble["scale"])
    _write_json(output / "report.json", report)
    for path, digest in dependencies.items():
        if sha(path) != digest:
            raise ValueError("frozen parent or retained dataset changed during fitting")
    _write_json(
        output / "exports.json",
        dict(
            control=exports,
            ensemble=dict(
                path="ensemble/ensemble.pkl",
                sha256=sha(output / "ensemble/ensemble.pkl"),
            ),
        ),
    )
    marker = dict(
        schema=1,
        stage="match",
        additional_simulator_steps=0,
        artifacts={
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    _write_json(output / "match_completed.json", marker)
    print("REWARD_EXPLORATION_MATCH_COMPLETED", flush=True)
    return marker


def verify_match(output, dataset=None, parent_cell=None, protocol=None):
    """Independently replay every retained checkpoint and selected raw exports."""

    output = Path(output)
    manifest, marker = _read(output / "manifest.json"), _read(
        output / "match_completed.json"
    )
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual != set(marker["artifacts"]) | {"match_completed.json"}:
        raise ValueError("matched fit artifact set differs")
    for path, digest in marker["artifacts"].items():
        if (
            Path(path).is_absolute()
            or ".." in Path(path).parts
            or sha(output / path) != digest
        ):
            raise ValueError("matched fit artifact changed")
    for path, digest in manifest["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("matched fit input changed")
    if protocol is not None and protocol != manifest["protocol"]:
        raise ValueError("matched fit protocol changed")
    protocol = manifest["protocol"]
    settings, ensemble_settings = protocol_settings(protocol)
    if (
        dataset is not None
        and Path(dataset).resolve() != Path(manifest["dataset"]).resolve()
    ):
        raise ValueError("matched fit dataset differs")
    if (
        parent_cell is not None
        and Path(parent_cell).resolve() != Path(manifest["parent_cell"]).resolve()
    ):
        raise ValueError("matched fit parent differs")
    data = load_npz(manifest["dataset"])
    train, val, test = split_indices(data)
    config = _config(manifest["parent_cell"])
    normalization = train_normalization(data, train)
    saved_norm = load_npz(output / "normalization.npz")
    for key, value in normalization.items():
        equal(value, saved_norm[key], "train-only " + key)
    indices = minibatch_indices(len(train) * 15, settings)
    schedule = load_npz(output / "minibatches.npz")
    equal(indices, schedule["indices"], "matched minibatch schedule")
    equal(train, schedule["train_rows"], "matched train indices")
    selection = _read(output / "selection.json")
    scale = max(
        float(np.sqrt(np.mean(data["rewards"][train].astype(np.float64).sum(-1) ** 2))),
        0.01,
    )
    equal(scale, selection["scale"], "training return scale")
    equal(
        np.unique(data["episode"][train]),
        selection["train_episodes"],
        "training episode selection",
    )
    equal(
        np.unique(data["episode"][val]),
        selection["validation_episodes"],
        "validation episode selection",
    )
    records = []
    rawtest = load_npz(output / "test_predictions.npz")
    equal(rawtest["indices"], test, "test rows")
    equal(rawtest["truth"], data["rewards"][test], "test truth")
    for kind in KINDS:
        predict = predictor(kind, config)
        for seed in settings["seeds"]:
            directory = output / f"{kind}_{seed}"
            saved = _read(directory / "selection.json")
            if saved["updates_completed"] != settings["updates"] or saved[
                "minibatch_sha256"
            ] != array_digest(indices):
                raise ValueError("unmatched fit update count or minibatches")
            initial = initialize_control(kind, config, 2560, seed, normalization)
            stored_initial = _load(directory / "initial.pkl")
            if set(initial) != set(stored_initial):
                raise ValueError("initial parameter keys differ")
            for key, value in initial.items():
                equal(value, stored_initial[key], "fresh initialization " + key)
            expected_updates = list(
                range(
                    settings["validation_period"],
                    settings["updates"] + 1,
                    settings["validation_period"],
                )
            )
            if [r["update"] for r in saved["validations"]] != expected_updates:
                raise ValueError("validation checkpoint grid differs")
            for record in saved["validations"]:
                params = _load(directory / record["checkpoint"])
                if kind == "categorical":
                    for key in CONTROL_FIXED:
                        equal(params[key], initial[key], "fixed categorical normalizer")
                pred = np.asarray(predict(params, data["targets"][val]))
                equal(
                    pred,
                    load_npz(directory / f"validation_{record['update']:05d}.npz")[
                        "prediction"
                    ],
                    "checkpoint validation replay",
                )
                equal(
                    validation_score(pred, data["rewards"][val], scale),
                    record["score"],
                    "checkpoint validation score",
                )
            chosen = saved["validations"][first_minimum(saved["validations"])]
            chosen = dict(kind=kind, seed=seed, directory=directory.name, **chosen)
            if chosen != saved["selected"]:
                raise ValueError(
                    "validation-only earliest checkpoint selection differs"
                )
            records.append(chosen)
            pred = np.asarray(
                predict(_load(directory / chosen["checkpoint"]), data["targets"][test])
            )
            equal(pred, rawtest[directory.name], "candidate test replay")
    if (
        records != selection["records"]
        or select_primary(records) != selection["primary"]
    ):
        raise ValueError("validation-only seed selection differs")
    for kind in KINDS:
        payload = _load(output / (kind + "_control.pkl"))
        chosen = selection["primary"][kind]
        params = _load(output / chosen["directory"] / chosen["checkpoint"])
        if payload["kind"] != kind or set(payload["params"]) != set(params):
            raise ValueError("control export identity differs")
        for key, value in params.items():
            equal(payload["params"][key], value, "selected export raw parameters")
        for key in ("mean", "std"):
            equal(
                payload["normalization"][key], normalization[key], "export normalizer"
            )
        pred = np.asarray(
            predictor(kind, config)(payload["params"], data["targets"][test])
        )
        equal(pred, rawtest[kind], "selected export test replay")
    ensemble = _load(output / "ensemble/ensemble.pkl")
    if (
        ensemble["bootstrap_seed"] != ensemble_settings["bootstrap_seed"]
        or ensemble["bonus_clip"] != ensemble_settings["bonus_clip"]
    ):
        raise ValueError("ensemble export protocol differs")
    for key, value in normalization.items():
        equal(
            ensemble["normalization"][key], value, "ensemble export fixed coordinates"
        )
    for param_key, norm_key in zip(
        ENSEMBLE_FIXED[:-1],
        ("belief_mean", "belief_std", "observation_mean", "observation_std"),
    ):
        equal(
            ensemble["params"][param_key],
            normalization[norm_key],
            "online ensemble fixed coordinates",
        )
    ep, mask, _ = episode_bootstrap(
        data["episode"][train], 5, ensemble_settings["bootstrap_seed"]
    )
    bootstrap = load_npz(output / "ensemble/bootstrap.npz")
    equal(ep, bootstrap["episodes"], "bootstrap episodes")
    equal(mask, bootstrap["masks"], "bootstrap member masks")
    training = _read(output / "ensemble/training.json")
    if training["updates_completed"] != ensemble_settings["updates"]:
        raise ValueError("ensemble update budget differs")
    ensemble_indices = minibatch_indices(
        len(train) * 15,
        dict(ensemble_settings, minibatch_seed=ensemble_settings["seed"] + 1),
    )
    equal(
        ensemble_indices,
        load_npz(output / "ensemble/minibatches.npz")["indices"],
        "ensemble train schedule",
    )
    predict = ensemble_predictor()
    features, actions, _ = transition_arrays(data, train)
    pred = np.asarray(predict(ensemble["params"], features, actions))
    calibration = load_npz(output / "ensemble/train_calibration.npz")
    equal(train, calibration["indices"], "train calibration rows")
    equal(pred, calibration["prediction"], "ensemble train prediction replay")
    values = disagreement(pred)
    equal(values, calibration["disagreement"], "train disagreement")
    scale_bonus = float(np.float32(max(float(np.quantile(values, 0.95)), 1e-4)))
    equal(scale_bonus, ensemble["scale"], "train-only uncertainty scale")
    equal(
        scale_bonus,
        ensemble["params"]["explore_ensemble/bonus_scale"],
        "online uncertainty scale",
    )
    associations = {}
    for split, rows in ((1, val), (2, test)):
        retained = load_npz(output / f"ensemble/heldout_{split}.npz")
        equal(rows, retained["indices"], "ensemble heldout rows")
        features, actions, observations = transition_arrays(data, rows)
        pred = np.asarray(predict(ensemble["params"], features, actions))
        equal(pred, retained["prediction"], "ensemble heldout replay")
        rng = np.random.default_rng(7171 + split)
        perm = rng.permutation(actions.size // actions.shape[-1])
        random_actions = rng.uniform(-1.0, 1.0, actions.shape).astype(np.float32)
        equal(perm, retained["shuffled_permutation"], "action shuffle")
        equal(random_actions, retained["random_actions"], "uniform action control")
        shuffled = actions.reshape(-1, 2)[perm].reshape(actions.shape)
        controls = {}
        for name, action in (
            ("shuffled_actions", shuffled),
            ("uniform_actions", random_actions),
        ):
            controls[name] = np.asarray(predict(ensemble["params"], features, action))
            equal(controls[name], retained[name], "heldout negative control replay")
        target = (observations - normalization["observation_mean"]) / normalization[
            "observation_std"
        ]
        equal(target, retained["truth_normalized"], "heldout target normalization")
        associations[str(split)] = ensemble_diagnostics(pred, target, controls)
    report = _reward_metrics(data, test, rawtest, selection["primary"], scale)
    report.update(ensemble_association=associations, ensemble_scale=scale_bonus)
    if report != _read(output / "report.json"):
        raise ValueError("independently reconstructed matched report differs")
    for path, digest in manifest["dependencies"].items():
        if sha(path) != digest:
            raise ValueError("matched parent/input mutation")
    print("REWARD_EXPLORATION_MATCH_VERIFIED", flush=True)
    return dict(
        verified=True,
        exports=_read(output / "exports.json"),
        additional_simulator_steps=0,
    )
