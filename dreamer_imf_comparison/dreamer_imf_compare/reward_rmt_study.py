"""Isolated RMT/reward-generalization diagnostic. Never steps a simulator.

Scientific stages require an exact clean commit and authenticated retained inputs.
Pure helpers accept small fixtures for local acceptance tests; the CLI does not
expose protocol overrides or bypasses for scientific stages.
"""

import argparse
import copy
import os
from pathlib import Path
import platform
import time

import jax
import numpy as np

from . import reward_capacity as capacity
from . import reward_rmt_fit as fitting
from . import reward_rmt_posterior as posterior
from . import reward_rmt_spectral as spectral
from .parallel_collection import _write_json, _write_npz
from .parallel_runner import load_npz
from .prior_correction_study import load
from .reward_diagnosis import artifacts, check_seal, seal
from .reward_diagnosis_data import GROUPS, validate_endpoint
from .reward_diagnosis_metrics import boundary_bins
from .reward_probe_metrics import _EpisodeBootstrap
from .reward_readout_study import (
    SOURCE,
    authenticate_parent,
    commit,
    equal,
    read,
    setup_teacher,
    sha,
)

DEPENDENCY_COMMIT = "f0b1be7f04e2fb1fcf2ecda58d68d6bed70dcad3"
DEPENDENCY_REPORT = "890b613ffcf49c143b9b760c711e38634f7461625d6fcda0542d55b5d6c31b16"
INPUTS = ("observation", "h", "hz", "hmean")
ENDPOINT_INPUTS = ("observation", "h", "hz")
SPLITS = ("train", "validation", "test")
PROTOCOL = dict(
    schema="reward-rmt-offline-v1",
    groups=list(GROUPS),
    readout_groups=list(INPUTS),
    endpoint_groups=list(ENDPOINT_INPUTS),
    seeds=[0, 1, 2],
    width=128,
    depth=2,
    updates=5000,
    batch_size=256,
    learning_rate=3e-4,
    minibatch_seed=6211,
    alphas=[0.0, 1e-6, 1e-4, 1e-2],
    bootstrap_reps=2000,
    spectral_resamples=16,
    spectral_permutations=100,
    kernel_examples=256,
    spectral_seed=7231,
    kernel_seed=7919,
    kernel_steps=[0, 250, 5000],
    ridge_ranks=[16, 64, 128],
    ridge_alphas=[10.0**i for i in range(-6, 3)],
    normalization_floor=0.01,
    additional_simulator_steps=0,
    selection="per-seed earliest minimum validation NLL; family mean over seeds; lower alpha tie",
    ridge_selection="unclipped validation MSE; smaller rank then larger alpha tie",
    gradpca="max-logit gradient, centered class means; ratio in class subspace; larger is in-subspace",
    gradient_scope="recomputed readout diagnostics; not historical world-model training gradients",
    precision="float32 readouts; float64 covariance/ridge/metrics; deployed frozen posterior precision unchanged",
    posterior="original retained inference seeds; exact h/z/carry parity; frozen parameters",
    reward_label="sum of two native rewards; endpoint occupancy is a separate binary task",
    statistical_unit="episode clusters; average seed errors, never predictions",
    material_deterioration_relative=0.10,
    promotion="at least 10% MSE reduction, paired interval below zero, upper paired NLL and cumulative-error intervals <=10% baseline; >=5 reward-positive test episodes",
    limitations=[
        "Reused test episodes: exploratory, not fresh confirmation",
        "One frozen world-model seed; head initializations are not independent world-model seeds",
        "Pointwise intervals without multiplicity correction",
        "Readout failure does not identify conditional reward variance or missing information",
        "No invented intermediate native observation; endpoint occupancy differs from summed reward",
        "Finite-network empirical kernels, not an infinite-width or fixed-kernel guarantee",
        "Spectral size is not reward relevance; GradPCA is not calibrated epistemic uncertainty",
        "No production repair, new environment interaction or online-return claim",
    ],
)


def protocol(preflight=False):
    p = copy.deepcopy(PROTOCOL)
    if preflight:
        p.update(
            updates=2,
            seeds=[0],
            bootstrap_reps=20,
            spectral_resamples=2,
            spectral_permutations=2,
            kernel_examples=16,
            kernel_steps=[0, 2],
            ridge_alphas=[1e-4, 1.0],
            ridge_ranks=[16],
        )
    return p


def authenticate(path):
    """Authenticate capacity, its diagnosis, raw episodes, dataset and parent."""
    path = Path(path).resolve()
    v, m = read(path / "verified.json"), read(path / "manifest.json")
    if (
        v.get("exact_replay") is not True
        or v.get("report_sha256") != DEPENDENCY_REPORT
        or sha(path / "report.json") != DEPENDENCY_REPORT
        or v.get("evaluated_sha256") != sha(path / "evaluated.json")
        or m.get("source_commit") != DEPENDENCY_COMMIT
        or m.get("preflight") is not False
    ):
        raise ValueError("wrong or unauthenticated capacity dependency")
    check_seal(path, "evaluated")
    data, inputs, geometry, binding = capacity.authenticate(m["dependency"]["path"])
    if binding != m["dependency"]:
        raise ValueError("capacity/diagnosis binding differs")
    binding = dict(
        path=str(path),
        marker_sha256=sha(path / "verified.json"),
        manifest_sha256=sha(path / "manifest.json"),
        report_sha256=DEPENDENCY_REPORT,
        evaluated_sha256=sha(path / "evaluated.json"),
        diagnosis=binding,
    )
    validate_splits(data)
    validate_endpoint(inputs["distance"], data["rewards"], geometry["radius"])
    return data, inputs, geometry, binding


def validate_splits(data):
    episode, split = np.asarray(data["episode"]), np.asarray(data["split"])
    if episode.shape != split.shape or not np.isin(split, [0, 1, 2]).all():
        raise ValueError("invalid split identity")
    for e in np.unique(episode):
        if len(np.unique(split[episode == e])) != 1:
            raise ValueError("episode leaked across splits")
    if any(not np.any(split == s) for s in range(3)):
        raise ValueError("empty episode split")


def selected_rows(data, preflight):
    validate_splits(data)
    if not preflight:
        return np.arange(len(data["episode"]), dtype=np.int64)
    selected = []
    for s in range(3):
        for mode in np.unique(data["mode"]):
            eligible = np.unique(
                data["episode"][(data["split"] == s) & (data["mode"] == mode)]
            )
            selected.extend(eligible[: 2 if s == 0 else 1])
    return np.flatnonzero(np.isin(data["episode"], selected))


def fresh_seal(output, stage):
    """Seal a stage once; both known hashes and final exact inventory are checked."""
    if (Path(output) / f"{stage}.json").exists():
        raise FileExistsError(f"{stage} already sealed")
    seal(output, stage)


def exact_inventory(output, stage, extras=()):
    marker = check_seal(output, stage)
    names = set(artifacts(output))
    expected = set(marker["artifacts"]) | {f"{stage}.json"} | set(extras)
    if names != expected:
        raise ValueError(f"unregistered artifacts: {sorted(names ^ expected)}")


def frozen_teacher(binding):
    dm = read(Path(binding["diagnosis"]["path"]) / "manifest.json")
    pm, dataset = authenticate_parent(Path(dm["parent"]))
    teacher = setup_teacher(pm)
    if teacher.frozen_digest() != binding["diagnosis"]["frozen_digest"]:
        raise ValueError("frozen checkpoint changed")
    return teacher, dataset.parent


def initialize(output, dependency, preflight=False):
    source = commit(SOURCE)
    context = authenticate(dependency)
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise ValueError(
            "scientific stages require a GPU; use unit tests for CPU fixtures"
        )
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
            initialized_unix=time.time(),
            devices=[str(d) for d in jax.devices()],
            jax=jax.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    data = context[0]
    rows = selected_rows(data, preflight)
    teacher, location = frozen_teacher(context[3])
    start = time.monotonic()
    stats = posterior.extract_statistics(teacher, data, location, indices=rows)
    _write_npz(output / "posterior.npz", dict(indices=rows, **stats))
    _write_json(
        output / "preparation.json",
        dict(
            frozen_digest=teacher.frozen_digest(),
            elapsed_seconds=time.monotonic() - start,
            rows=len(rows),
            protocol="fresh read-only statistics, original retained draw",
        ),
    )
    fresh_seal(output, "prepared")
    print("REWARD_RMT_PREPARED", flush=True)


def setup(output):
    m = read(Path(output) / "manifest.json")
    if m["source_commit"] != commit(SOURCE) or m["protocol"] != protocol(
        m["preflight"]
    ):
        raise ValueError("source or fixed protocol changed")
    if m["jax"] != jax.__version__ or m["numpy"] != np.__version__:
        raise ValueError("numeric runtime changed")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise ValueError("scientific stages require the GPU runtime")
    context = authenticate(m["dependency"]["path"])
    if context[3] != m["dependency"] or context[2] != m["geometry"]:
        raise ValueError("dependency identity changed")
    return context


def feature_context(output, context):
    check_seal(output, "prepared")
    data, inputs, _, _ = context
    m = read(Path(output) / "manifest.json")
    stats = load_npz(Path(output) / "posterior.npz")
    rows = selected_rows(data, m["preflight"])
    equal(rows, stats["indices"], "posterior row identities")
    x = {g: capacity.features(data, inputs, g)[rows] for g in GROUPS}
    equal(stats["h"], x["h"], "retained h")
    equal(stats["z"], x["z"], "retained z")
    if stats["mean"].shape != x["z"].shape or stats["std"].shape != x["z"].shape:
        raise ValueError("posterior statistics shape")
    if (
        any(not np.isfinite(v).all() for v in stats.values())
        or (stats["std"] <= 0).any()
    ):
        raise ValueError("invalid posterior statistics")
    x["hmean"] = np.concatenate((stats["h"], stats["mean"]), -1)
    d = {
        k: np.asarray(v)[rows]
        for k, v in data.items()
        if np.ndim(v) and len(v) == len(data["episode"])
    }
    d["distance"] = inputs["distance"][rows]
    d["global_rows"] = rows
    ix = {name: np.flatnonzero(d["split"] == s) for s, name in enumerate(SPLITS)}
    return x, d, ix


def flat(x):
    return np.asarray(x).reshape(-1, x.shape[-1])


def cell_path(output, task, group, alpha, seed):
    return Path(output) / "heads" / f"{task}-{group}-a{float(alpha):g}-s{seed}"


def fit_models(output, context, verify=False):
    x, d, ix = feature_context(output, context)
    p = read(Path(output) / "manifest.json")["protocol"]
    train, val = ix["train"], ix["validation"]
    choose = {}
    fit = fitting.verify_cell if verify else fitting.fit_cell
    for group in p["readout_groups"]:
        records = {}
        for alpha in p["alphas"]:
            records[str(alpha)] = [
                fit(
                    cell_path(output, "reward", group, alpha, seed),
                    flat(x[group][train]),
                    d["rewards"][train].ravel(),
                    flat(x[group][val]),
                    d["rewards"][val].ravel(),
                    seed,
                    alpha,
                    p,
                    classes=3,
                )
                for seed in p["seeds"]
            ]
        choose[group] = fitting.select_family(records)
        choose[group]["unregularized"] = fitting.select_family(
            {str(0.0): records[str(0.0)]}
        )
    endpoint = (d["distance"] <= context[2]["radius"]).astype(np.int32)
    for group in p["endpoint_groups"]:
        alpha = choose[group]["alpha"]
        records = [
            fit(
                cell_path(output, "endpoint", group, alpha, seed),
                flat(x[group][train]),
                endpoint[train].ravel(),
                flat(x[group][val]),
                endpoint[val].ravel(),
                seed,
                alpha,
                p,
                classes=2,
            )
            for seed in p["seeds"]
        ]
        choose["endpoint-" + group] = fitting.select_family({str(alpha): records})
    ridge = fitting.verify_ridge if verify else fitting.fit_ridge
    for group in p["groups"]:
        choose["ridge-" + group] = ridge(
            Path(output) / "ridge" / group,
            flat(x[group][train]),
            d["rewards"][train].ravel(),
            flat(x[group][val]),
            d["rewards"][val].ravel(),
            p,
        )
    if verify:
        if choose != read(Path(output) / "selections.json"):
            raise ValueError("validation-only selections differ")
    else:
        _write_json(Path(output) / "selections.json", choose)
        fresh_seal(output, "fitted")
        print("REWARD_RMT_FITTED", flush=True)
    return choose


def balanced_subset(episodes, count, seed):
    episodes = np.asarray(episodes)
    groups = np.unique(episodes)
    if not len(groups) or count < len(groups) or count > len(episodes):
        raise ValueError("cannot form episode-balanced kernel subset")
    rng = np.random.default_rng(seed)
    quota, remainder = divmod(count, len(groups))
    order = rng.permutation(groups)
    rows = []
    for j, episode in enumerate(order):
        available = np.flatnonzero(episodes == episode)
        n = quota + (j < remainder)
        if n > len(available):
            raise ValueError("insufficient examples in an episode")
        rows.extend(rng.choice(available, n, replace=False))
    return np.sort(np.asarray(rows, dtype=np.int64))


def correlations(score, errors, episode, reps):
    """Descriptive association, with episode-block bootstrap; not OOD evidence."""
    score, errors, episode = [np.asarray(v).ravel() for v in (score, errors, episode)]
    if not (len(score) == len(errors) == len(episode)) or not np.isfinite(score).all():
        raise ValueError("invalid diagnostic association")

    def corr(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    value = corr(score, errors)
    eps = np.unique(episode)
    rng = np.random.default_rng(832)
    samples = []
    for _ in range(reps):
        rows = np.concatenate(
            [np.flatnonzero(episode == e) for e in rng.choice(eps, len(eps))]
        )
        c = corr(score[rows], errors[rows])
        if c is not None:
            samples.append(c)
    return dict(
        pearson=value,
        ci95=list(np.quantile(samples, [0.025, 0.975])) if samples else [None, None],
        episodes=len(eps),
        valid_resamples=len(samples),
        scope="held-out descriptive error association; not calibrated OOD/uncertainty",
    )


def diagnostics(output, context):
    check_seal(output, "fitted")
    x, d, ix = feature_context(output, context)
    p = read(Path(output) / "manifest.json")["protocol"]
    tr = ix["train"]
    y = d["rewards"][tr].ravel()
    episode = np.repeat(d["episode"][tr], d["rewards"].shape[1])
    mode = np.repeat(d["mode"][tr], d["rewards"].shape[1])
    sub = balanced_subset(episode, min(p["kernel_examples"], len(y)), p["kernel_seed"])
    raw = dict(
        kernel_training_flat_indices=sub,
        kernel_global_branch=d["global_rows"][tr][sub // 15],
        kernel_horizon=sub % 15 + 1,
    )
    report = dict(features={}, kernels={}, gradpca={})
    for group in p["groups"]:
        tx = flat(x[group][tr]).astype(np.float64)
        mean, std = tx.mean(0), np.maximum(tx.std(0), p["normalization_floor"])
        raw[group + "__mean"], raw[group + "__std"] = mean, std
        for scale, values in (("raw", tx), ("standardized", (tx - mean) / std)):
            key = group + "__" + scale
            info, arrays = spectral.analyze_features(
                values,
                y,
                episode,
                mode,
                seed=p["spectral_seed"],
                resamples=p["spectral_resamples"],
                permutations=p["spectral_permutations"],
            )
            report["features"][key] = info
            raw.update({key + "__" + k: v for k, v in arrays.items()})
    for group in p["endpoint_groups"]:
        directory = cell_path(output, "reward", group, 0.0, 0)
        norm = load_npz(directory / "normalization.npz")
        tx = flat(x[group][tr])
        standardized = (tx - norm["mean"]) / norm["std"]
        for step in p["kernel_steps"]:
            params = load(directory / f"checkpoint-{step:05d}.pkl")
            key = f"{group}__{step}"
            info, arrays = spectral.kernel_diagnostics(
                params, standardized[sub], y[sub], episode[sub]
            )
            report["kernels"][key] = info
            raw.update({key + "__kernel__" + k: v for k, v in arrays.items()})
        record = read(directory / "selection.json")
        params = load(directory / f'checkpoint-{record["selected"]["update"]:05d}.pkl')
        state = spectral.fit_gradpca(params, standardized, y)
        raw.update(
            {group + "__gradpca_state__" + k: np.asarray(v) for k, v in state.items()}
        )
        report["gradpca"][group] = dict(
            selected_update=record["selected"]["update"],
            fitted_on="train only",
            maximum_centered_class_rank=2,
            variant=p["gradpca"],
            splits={},
            numerical_rank=int(state["numerical_rank"]),
            rank_cap=int(state["rank_cap"]),
            present_classes=state["classes"].tolist(),
            class_counts=state["counts"].tolist(),
            degenerate=bool(state["degenerate"]),
            zero_centered_gradient_convention="score zero; has no OOD interpretation",
        )
        for split in ("validation", "test"):
            rows = ix[split]
            qx = (flat(x[group][rows]) - norm["mean"]) / norm["std"]
            score = spectral.score_gradpca(params, qx, state)
            dist = spectral.nearest_distance(standardized, qx)
            probs = fitting.predict(params, norm, flat(x[group][rows]))
            error = (probs @ np.arange(3) - d["rewards"][rows].ravel()) ** 2
            ep = np.repeat(d["episode"][rows], 15)
            raw[f"{group}__gradpca__{split}"] = score
            raw[f"{group}__distance__{split}"] = dist
            report["gradpca"][group]["splits"][split] = dict(
                gradpca=correlations(-score, error, ep, p["bootstrap_reps"]),
                distance=correlations(dist, error, ep, p["bootstrap_reps"]),
                score_direction="negative in-subspace ratio; larger means less class-subspace alignment",
            )
    return report, raw


def predictions(output, context):
    """Retain the selected, matched dense/coarse baseline and final predictions."""
    check_seal(output, "fitted")
    x, d, ix = feature_context(output, context)
    p = read(Path(output) / "manifest.json")["protocol"]
    selected = read(Path(output) / "selections.json")
    raw = {"indices__" + s: d["global_rows"][rows] for s, rows in ix.items()}
    specs = {}

    def add(name, task, group, seed, directory, step):
        params = load(directory / f"checkpoint-{step:05d}.pkl")
        norm = load_npz(directory / "normalization.npz")
        specs[name] = dict(
            task=task,
            group=group,
            seed=seed,
            update=step,
            directory=str(directory.relative_to(output)),
            categorical=True,
        )
        for split, rows in ix.items():
            prob = fitting.predict(params, norm, flat(x[group][rows]))
            raw[name + "__" + split] = prob.reshape(
                d["rewards"][rows].shape + (prob.shape[-1],)
            )

    for group in p["readout_groups"]:
        family = selected[group]
        for seed in p["seeds"]:
            alpha = family["alpha"]
            directory = cell_path(output, "reward", group, alpha, seed)
            add(
                f"regularized-{group}-s{seed}",
                "reward",
                group,
                seed,
                directory,
                family["selections"][str(seed)]["update"],
            )
            add(
                f"final-{group}-s{seed}", "reward", group, seed, directory, p["updates"]
            )
            directory = cell_path(output, "reward", group, 0.0, seed)
            record = read(directory / "selection.json")
            add(
                f"baseline-{group}-s{seed}",
                "reward",
                group,
                seed,
                directory,
                record["selected"]["update"],
            )
            coarse = [
                c
                for c in record["candidates"]
                if c["update"] >= 250 and c["update"] % 250 == 0
            ]
            if not coarse:  # Tiny preflight only, never a full-run fallback.
                if not read(Path(output) / "manifest.json")["preflight"]:
                    raise ValueError("missing original coarse candidate schedule")
                coarse = [record["candidates"][-1]]
            add(
                f"coarse-{group}-s{seed}",
                "reward",
                group,
                seed,
                directory,
                fitting.select(coarse)["update"],
            )
    for group in p["endpoint_groups"]:
        family = selected["endpoint-" + group]
        for seed in p["seeds"]:
            add(
                f"endpoint-{group}-s{seed}",
                "endpoint",
                group,
                seed,
                cell_path(output, "endpoint", group, family["alpha"], seed),
                family["selections"][str(seed)]["update"],
            )
    for group in p["groups"]:
        for clipped in (False, True):
            name = ("ridge-clipped-" if clipped else "ridge-") + group
            specs[name] = dict(
                task="reward",
                group=group,
                seed=None,
                categorical=False,
                clipped=clipped,
                label="continuous prediction; no NLL/Brier",
            )
            for split, rows in ix.items():
                pred = fitting.predict_ridge(
                    Path(output) / "ridge" / group, flat(x[group][rows])
                )
                if clipped:
                    pred = np.clip(pred, 0, 2)
                raw[name + "__" + split] = pred.reshape(d["rewards"][rows].shape)
    return raw, specs


def error_arrays(value, truth, categorical):
    truth = np.asarray(truth)
    value = np.asarray(value, np.float64)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite prediction")
    if categorical:
        classes = value.shape[-1]
        if (
            value.shape != truth.shape + (classes,)
            or classes not in (2, 3)
            or not np.isin(truth, np.arange(classes)).all()
            or (value < 0).any()
            or not np.allclose(value.sum(-1), 1, rtol=0, atol=2e-6)
        ):
            raise ValueError("invalid categorical output")
        pred = value @ np.arange(classes)
    else:
        if value.shape != truth.shape:
            raise ValueError("invalid continuous prediction")
        pred = value
    result = dict(
        mse=(pred - truth) ** 2,
        bias=pred - truth,
        cumulative_mse=(pred.sum(-1) - truth.sum(-1)) ** 2,
    )
    for h in (1, 3, 5, 10, 15):
        result[f"cumulative_mse_h{h}"] = (pred[:, :h].sum(1) - truth[:, :h].sum(1)) ** 2
        result[f"mse_h{h}"] = (pred[:, h - 1] - truth[:, h - 1]) ** 2
    if categorical:
        result.update(
            nll=-np.log(
                np.maximum(
                    np.take_along_axis(value, truth.astype(int)[..., None], -1)[..., 0],
                    1e-8,
                )
            ),
            brier=((value - np.eye(classes)[truth.astype(int)]) ** 2).sum(-1),
        )
    return result


def metric_stats(errors, episodes, boot):
    return {
        key: boot.stat(
            value.ravel(),
            (
                np.asarray(episodes)
                if value.ndim == 1
                else np.broadcast_to(np.asarray(episodes)[:, None], value.shape)
            ).ravel(),
        )
        for key, value in errors.items()
    }


def categorical_calibration(value, truth):
    result = {}
    for c in range(value.shape[-1]):
        probability, target = value[..., c], truth == c
        bins = np.minimum((probability * 10).astype(int), 9)
        result[str(c)] = [
            dict(
                count=int((bins == b).sum()),
                mean_prediction=(
                    float(probability[bins == b].mean()) if (bins == b).any() else None
                ),
                empirical_frequency=(
                    float(target[bins == b].mean()) if (bins == b).any() else None
                ),
            )
            for b in range(10)
        ]
    return result


def summarize_prediction(value, y, categorical, data, rows, radius, boot):
    errors = error_arrays(value, y, categorical)
    episodes = data["episode"][rows]
    ep = np.broadcast_to(episodes[:, None], y.shape)
    result = dict(
        overall=metric_stats(errors, episodes, boot),
        statistical_unit="episode",
        strata={},
    )
    labels = dict(
        collection_mode=np.broadcast_to(data["mode"][rows, None], y.shape),
        action_plan=np.broadcast_to(data["plan"][rows, None], y.shape),
        reward_class=y,
        endpoint_distance=boundary_bins(data["distance"][rows], radius),
    )
    for kind, lab in labels.items():
        groups = (
            range(4) if kind in ("action_plan", "endpoint_distance") else np.unique(lab)
        )
        result["strata"][kind] = {}
        for k in groups:
            mask = lab == k
            result["strata"][kind][str(k)] = dict(
                count=int(mask.sum()),
                episodes=int(len(np.unique(ep[mask]))),
                **{
                    key: boot.stat(arr[mask], ep[mask])
                    for key, arr in errors.items()
                    if arr.ndim == 2
                },
            )
    if categorical:
        result["calibration"] = categorical_calibration(value, y)
    else:
        result["probability_metrics"] = "not applicable to continuous ridge"
    return result


def promotion(candidate, baseline, episodes, positive_episodes, p):
    """Prospective exploratory screen with explicit 10% safety margins."""
    boot = _EpisodeBootstrap(p["bootstrap_reps"])
    deltas = {k: candidate[k] - baseline[k] for k in candidate if k in baseline}
    stats = metric_stats(deltas, episodes, boot)
    base = metric_stats(baseline, episodes, boot)
    cand = metric_stats(candidate, episodes, boot)
    reduction = (
        1 - cand["mse"]["mean"] / base["mse"]["mean"]
        if base["mse"]["mean"] > 0
        else None
    )
    safety = {}
    for key in ("nll", "cumulative_mse"):
        if key not in stats:
            safety[key] = None
        else:
            upper = stats[key]["ci95"][1]
            safety[key] = (
                upper is not None
                and upper <= p["material_deterioration_relative"] * base[key]["mean"]
            )
    hi = stats["mse"]["ci95"][1]
    return dict(
        relative_mse_reduction=reduction,
        paired_deltas=stats,
        no_material_deterioration=safety,
        positive_test_episodes=int(positive_episodes),
        exploratory_screen_passed=bool(
            reduction is not None
            and reduction >= 0.10
            and hi is not None
            and hi < 0
            and all(v is True for v in safety.values())
            and positive_episodes >= 5
        ),
        interpretation="engineering screen on reused test episodes; no superiority or causal claim",
    )


def make_report(raw, specs, spectral_report, selections, data, inputs, geometry, p):
    boot = _EpisodeBootstrap(p["bootstrap_reps"])
    data = dict(data, distance=inputs["distance"])
    individual, averages, comparisons, gaps = {}, {}, {}, {}

    def y_for(task, rows):
        return (
            (data["distance"][rows] <= geometry["radius"]).astype(int)
            if task == "endpoint"
            else data["rewards"][rows]
        )

    for name, spec in specs.items():
        individual[name] = {}
        for split in SPLITS:
            rows = raw["indices__" + split]
            individual[name][split] = summarize_prediction(
                raw[name + "__" + split],
                y_for(spec["task"], rows),
                spec["categorical"],
                data,
                rows,
                geometry["radius"],
                boot,
            )
        gaps[name] = {
            key: individual[name]["validation"]["overall"][key]["mean"]
            - individual[name]["train"]["overall"][key]["mean"]
            for key in ("mse", "nll", "cumulative_mse")
            if key in individual[name]["train"]["overall"]
        }

    def average_errors(prefix, group, split, task="reward"):
        rows = raw["indices__" + split]
        values = [
            error_arrays(
                raw[f"{prefix}-{group}-s{s}__{split}"], y_for(task, rows), True
            )
            for s in p["seeds"]
        ]
        return {key: np.mean([v[key] for v in values], 0) for key in values[0]}

    test = raw["indices__test"]
    ep = data["episode"][test]
    positive = len(
        np.unique(
            np.broadcast_to(ep[:, None], data["rewards"][test].shape)[
                data["rewards"][test] > 0
            ]
        )
    )
    for group in p["readout_groups"]:
        for prefix in ("regularized", "baseline", "coarse", "final"):
            averages[f"{prefix}-{group}"] = {
                split: metric_stats(
                    average_errors(prefix, group, split),
                    data["episode"][raw["indices__" + split]],
                    boot,
                )
                for split in SPLITS
            }
        for name, candidate, baseline in (
            ("regularization_vs_dense_early", "regularized", "baseline"),
            ("early_vs_coarse", "baseline", "coarse"),
            ("combined_vs_coarse", "regularized", "coarse"),
        ):
            comparisons[group + "__" + name] = promotion(
                average_errors(candidate, group, "test"),
                average_errors(baseline, group, "test"),
                ep,
                positive,
                p,
            )
    for group in p["endpoint_groups"]:
        averages["endpoint-" + group] = {
            split: metric_stats(
                average_errors("endpoint", group, split, "endpoint"),
                data["episode"][raw["indices__" + split]],
                boot,
            )
            for split in SPLITS
        }
    comparisons["posterior_mean_vs_sample"] = promotion(
        average_errors("regularized", "hmean", "test"),
        average_errors("regularized", "hz", "test"),
        ep,
        positive,
        p,
    )
    comparisons["posterior_sample_added_to_h"] = promotion(
        average_errors("regularized", "hz", "test"),
        average_errors("regularized", "h", "test"),
        ep,
        positive,
        p,
    )
    for group in p["endpoint_groups"]:
        baseline = average_errors("baseline", group, "test")
        for prefix in ("ridge", "ridge-clipped"):
            candidate = error_arrays(
                raw[f"{prefix}-{group}__test"], data["rewards"][test], False
            )
            delta = {k: candidate[k] - baseline[k] for k in candidate}
            base_mse = metric_stats(baseline, ep, boot)["mse"]["mean"]
            mse = metric_stats(candidate, ep, boot)["mse"]["mean"]
            comparisons[f"{prefix}-{group}__vs_dense_early"] = dict(
                paired_deltas=metric_stats(delta, ep, boot),
                relative_mse_reduction=1 - mse / base_mse if base_mse > 0 else None,
                probabilistic_screen="not applicable: continuous ridge has no NLL",
                baseline="fresh unregularized validation-selected neural readout; errors averaged over seeds",
            )
    coverage = {}
    for split in SPLITS:
        rows = raw["indices__" + split]
        coverage[split] = dict(
            episodes=len(np.unique(data["episode"][rows])),
            branches=len(rows),
            transitions=data["rewards"][rows].size,
            reward_counts={
                str(c): int((data["rewards"][rows] == c).sum()) for c in range(3)
            },
            endpoint_counts={
                str(c): int((y_for("endpoint", rows) == c).sum()) for c in (0, 1)
            },
        )
    return dict(
        schema=p["schema"],
        protocol=p,
        coverage=coverage,
        selections=selections,
        per_seed=individual,
        seed_averaged_errors=averages,
        validation_minus_train=gaps,
        comparisons=comparisons,
        diagnostics=spectral_report,
        endpoint_cumulative_label="error in number of occupied endpoints, NOT AR2 cumulative reward",
        timing_definition="fit durations include compilation; stage and allocation times are separate",
        scientific_status="exploratory diagnosis, regardless of screen outcome",
    )


def evaluate(output, context):
    output = Path(output)
    exact_inventory(output, "fitted")
    started = time.monotonic()
    info, arrays = diagnostics(output, context)
    _write_json(output / "spectral.json", info)
    _write_npz(output / "spectral.npz", arrays)
    raw, specs = predictions(output, context)
    _write_npz(output / "predictions.npz", raw)
    _write_json(output / "prediction-specs.json", specs)
    p = read(output / "manifest.json")["protocol"]
    report = make_report(
        raw, specs, info, read(output / "selections.json"), *context[:3], p
    )
    _write_json(output / "report.json", report)
    _write_json(
        output / "evaluation-runtime.json",
        dict(elapsed_seconds=time.monotonic() - started),
    )
    fresh_seal(output, "evaluated")
    print("REWARD_RMT_EVALUATED", flush=True)


def equal_arrays(actual, retained, label):
    if set(actual) != set(retained):
        raise ValueError(label + " array inventory differs")
    for key in actual:
        equal(actual[key], retained[key], label + " " + key)


def verify(output, context):
    output = Path(output)
    exact_inventory(output, "evaluated")
    started = time.monotonic()
    retained = load_npz(output / "posterior.npz")
    teacher, location = frozen_teacher(context[3])
    recomputed = posterior.extract_statistics(
        teacher, context[0], location, indices=retained["indices"]
    )
    equal_arrays(
        recomputed,
        {k: v for k, v in retained.items() if k != "indices"},
        "posterior replay",
    )
    if teacher.frozen_digest() != context[3]["diagnosis"]["frozen_digest"]:
        raise ValueError("frozen parameters mutated")
    choose = fit_models(output, context, verify=True)
    info, arrays = diagnostics(output, context)
    if info != read(output / "spectral.json"):
        raise ValueError("spectral diagnostics differ")
    equal_arrays(arrays, load_npz(output / "spectral.npz"), "spectral replay")
    raw, specs = predictions(output, context)
    equal_arrays(raw, load_npz(output / "predictions.npz"), "prediction replay")
    if specs != read(output / "prediction-specs.json"):
        raise ValueError("prediction identities changed")
    p = read(output / "manifest.json")["protocol"]
    report = make_report(raw, specs, info, choose, *context[:3], p)
    if report != read(output / "report.json"):
        raise ValueError("independent metrics differ")
    if authenticate(context[3]["path"])[3] != context[3]:
        raise ValueError("dependency changed during verification")
    exact_inventory(output, "evaluated")
    _write_json(
        output / "verified.json",
        dict(
            exact_replay=True,
            source_commit=commit(SOURCE),
            evaluated_sha256=sha(output / "evaluated.json"),
            report_sha256=sha(output / "report.json"),
            frozen_digest=teacher.frozen_digest(),
            verification_seconds=time.monotonic() - started,
            process_id=os.getpid(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    print("REWARD_RMT_STUDY_VERIFIED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate", "verify"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dependency", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.dependency is None:
            parser.error("prepare requires --dependency")
        initialize(args.output, args.dependency, args.preflight)
    else:
        if args.dependency is not None or args.preflight:
            parser.error(
                "later stages derive dependency/protocol from immutable manifest"
            )
        context = setup(args.output)
        if args.stage == "fit":
            exact_inventory(args.output, "prepared")
            fit_models(args.output, context)
        elif args.stage == "evaluate":
            evaluate(args.output, context)
        else:
            verify(args.output, context)


if __name__ == "__main__":
    main()
