"""Independent NumPy metrics and preregistered three-seed exploratory screen.

Categorical predictive NLL uses the mean particle probability, never the mean
particle NLL. The original head's loss is a separate native-objective score;
it is not recast into three classes and receives no fabricated Brier score.
Reward errors are in raw AR2 reward units. Source policies and environment
groups are equally weighted. The hierarchical bootstrap resamples continuation
seeds first, then paired evaluation-environment groups, never particles/anchors.
Only three top-level seeds exist, so intervals are weak exploratory diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

ARMS, SEEDS, MILESTONES = ("A", "B", "C"), (901, 902, 903), (0, 40000, 80000)
HORIZONS, ANCHORS = (1, 3, 5, 10, 15), (100, 200, 300, 400)
NLL_FLOOR = 1e-12
BOOTSTRAP_REPS = 2000
HEADS = (
    "posterior_original",
    "posterior_student",
    "posterior_teacher",
    "predicted_original",
    "predicted_student",
)


def _array(value, name, shape=None):
    raw = np.asarray(value)
    if (
        raw.dtype.kind not in "biuf"
        or not np.isfinite(raw).all()
        or (shape is not None and raw.shape != shape)
    ):
        raise ValueError("invalid finite array/shape: " + name)
    return raw.astype(np.float64)


def _json(value):
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("metrics must be finite JSON") from exc
    return value


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for name in (
        "truth_reward",
        "truth_observation",
        "action",
        "episode",
        "episode_seed",
        "anchor",
    ):
        raw = np.asarray(arrays[name])
        value = np.ascontiguousarray(
            raw, dtype="<i8" if raw.dtype.kind in "iu" else "<f8"
        )
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _calibration(probabilities, truth):
    probability = probabilities.reshape(-1, 3)
    labels = truth.astype(int).reshape(-1)
    confidence = probability.max(-1)
    correct = probability.argmax(-1) == labels
    edges = np.linspace(0, 1, 11)

    def bins(p, observed):
        index = np.minimum((p * 10).astype(int), 9)
        records, error = [], 0.0
        for i in range(10):
            selected = index == i
            count = int(selected.sum())
            mean = float(p[selected].mean()) if count else None
            frequency = float(observed[selected].mean()) if count else None
            if count:
                error += count * abs(mean - frequency) / len(p)
            records.append(
                dict(
                    lower=float(edges[i]),
                    upper=float(edges[i + 1]),
                    count=count,
                    mean_probability=mean,
                    observed_frequency=frequency,
                )
            )
        return dict(ece=float(error), bins=records)

    return dict(
        top_label=bins(confidence, correct),
        classwise={
            str(label): bins(probability[:, label], labels == label)
            for label in range(3)
        },
    )


def _score_vectors(prediction, truth, probability=None, native=None):
    scores = {
        "reward_mse": (prediction - truth) ** 2,
        "reward_bias": prediction - truth,
    }
    if probability is not None:
        chosen = np.take_along_axis(probability, truth.astype(int)[..., None], -1)[
            ..., 0
        ]
        scores["nll"] = -np.log(np.maximum(chosen, NLL_FLOOR))
        scores["brier"] = ((probability - np.eye(3)[truth.astype(int)]) ** 2).sum(-1)
        cdf = probability.cumsum(-1)
        lower, upper = (np.sum(cdf < quantile, axis=-1) for quantile in (0.05, 0.95))
        scores["interval90_coverage"] = ((truth >= lower) & (truth <= upper)).astype(
            float
        )
        scores["zero_true_class_probability"] = (chosen == 0).astype(float)
    if native is not None:
        scores["native_objective"] = native
    return scores


def _aggregate(scores, prediction, truth, mask, horizon=None):
    # Each invocation sees equal anchors per environment, so ordinary means here
    # equal episode means. Conditional scores only describe observed class rows.
    result = {}
    target = truth[mask] if horizon is None else truth[mask, horizon - 1]
    error = (
        prediction[mask] - truth[mask]
        if horizon is None
        else prediction[mask, horizon - 1] - target
    )
    for name, value in scores.items():
        result[name] = float(
            np.mean(value[mask] if horizon is None else value[mask, horizon - 1])
        )
    for label, selected in (("positive", target > 0), ("zero", target == 0)):
        result[label + "_count"] = int(selected.sum())
        result[label + "_reward_mse"] = (
            float(np.mean(error[selected] ** 2)) if selected.any() else None
        )
    h = truth.shape[1] if horizon is None else horizon
    result["cumulative_reward_mse"] = float(
        np.mean((prediction[mask, :h] - truth[mask, :h]).sum(-1) ** 2)
    )
    return result


def summarize_arrays(arrays):
    """Score one ten-episode source-policy probe and retain matched group scores.

    Returns pooled15 and five horizon metrics, real class/episode coverage,
    categorical calibration, and by_episode keyed by actual environment seed.
    These are source-stratum summaries; paired_report supplies uncertainty and
    cross-source weighting. Original native losses are averaged over particles
    and tagged native_objective, not comparable categorical predictive NLL.
    """
    required = {
        "truth_reward",
        "truth_observation",
        "action",
        "episode",
        "episode_seed",
        "anchor",
        "predicted_observation",
        *HEADS,
        "posterior_student_probabilities",
        "posterior_teacher_probabilities",
        "predicted_student_probabilities",
        "posterior_original_native_loss",
        "predicted_original_native_loss",
    }
    if not required.issubset(arrays):
        raise ValueError("probe fields missing: " + str(sorted(required - set(arrays))))
    if any(
        name.startswith("predicted_teacher") or name.startswith("imagined_teacher")
        for name in arrays
    ):
        raise ValueError("teacher is undefined at imagined future states")
    truth = _array(arrays["truth_reward"], "truth_reward", (40, 15))
    obs = _array(arrays["truth_observation"], "truth_observation", (40, 15, 6))
    _array(arrays["action"], "action", (40, 15, 2))
    episode, anchor = np.asarray(arrays["episode"]), np.asarray(arrays["anchor"])
    ids = np.asarray(arrays["episode_seed"])
    if (
        episode.dtype.kind not in "iu"
        or anchor.dtype.kind not in "iu"
        or ids.dtype.kind not in "iu"
    ):
        raise ValueError("probe identities must be integer arrays")
    if not np.array_equal(episode, np.repeat(np.arange(10), 4)) or not np.array_equal(
        anchor, np.tile(ANCHORS, 10)
    ):
        raise ValueError("probe episode/anchor layout differs")
    if (
        ids.shape != (40,)
        or (ids < 0).any()
        or (ids >= 2**32).any()
        or len(np.unique(ids)) != 10
        or any(len(np.unique(ids[episode == e])) != 1 for e in range(10))
    ):
        raise ValueError("environment identities are missing or aliased")
    if not np.isin(truth, (0, 1, 2)).all():
        raise ValueError("truth reward is not an AR2 class")
    predictions, probabilities, vectors = {}, {}, {}
    for name in HEADS:
        imagined = name.startswith("predicted_")
        shape = (40, 8, 15) if imagined else (40, 15)
        raw = _array(arrays[name], name, shape)
        prediction = raw.mean(1) if imagined else raw
        probability, native = None, None
        if not name.endswith("original"):
            raw_probability = _array(
                arrays[name + "_probabilities"], name + "_probabilities", (*shape, 3)
            )
            if (
                (raw_probability < 0).any()
                or (raw_probability > 1).any()
                or not np.allclose(raw_probability.sum(-1), 1, atol=1e-6, rtol=1e-6)
            ):
                raise ValueError("categorical probabilities are not normalized")
            if not np.allclose(
                raw, raw_probability @ np.arange(3), atol=2e-5, rtol=2e-5
            ):
                raise ValueError(
                    "categorical prediction differs from its probability mean"
                )
            probability = raw_probability.mean(1) if imagined else raw_probability
            probabilities[name] = probability
        else:
            loss = _array(arrays[name + "_native_loss"], name + "_native_loss", shape)
            native = loss.mean(1) if imagined else loss
        predictions[name] = prediction
        vectors[name] = _score_vectors(prediction, truth, probability, native)
    predicted_obs = _array(
        arrays["predicted_observation"], "predicted_observation", (40, 8, 15, 6)
    )
    observation_error = ((predicted_obs.mean(1) - obs) ** 2).mean(-1)
    all_rows = np.ones(40, bool)
    result = dict(
        schema="posterior-reward-probe-v1",
        rows=40,
        particles=8,
        episodes=10,
        truth_digest=_fingerprint(arrays),
        pooled={},
        horizons={},
        by_episode={},
        calibration={},
        conventions=dict(
            reward_units="raw summed AR2 rewards",
            pooled="equal environment and anchor; all15 decisions",
            categorical_nll="negative log of particle-mean true-class probability",
            nll_floor=NLL_FLOOR,
            original_score="particle-mean native head objective; not three-class predictive NLL",
            intervals="central90 categorical intervals; inclusive integer endpoints",
            original_brier=None,
            causal_intervention=False,
        ),
    )
    for name in HEADS:
        result["pooled"][name] = _aggregate(
            vectors[name], predictions[name], truth, all_rows
        )
        if name in probabilities:
            result["calibration"][name] = _calibration(probabilities[name], truth)
    result["pooled"]["observation_mse"] = float(observation_error.mean())
    for h in HORIZONS:
        result["horizons"][str(h)] = {
            name: _aggregate(vectors[name], predictions[name], truth, all_rows, h)
            for name in HEADS
        }
        result["horizons"][str(h)]["observation_mse"] = float(
            observation_error[:, h - 1].mean()
        )
    positive = []
    for e in range(10):
        mask, identity = episode == e, int(ids[4 * e])
        count = int((truth[mask] > 0).sum())
        if count:
            positive.append(identity)
        item = dict(
            episode=e,
            episode_seed=identity,
            positive_count=count,
            class_counts=[int((truth[mask] == label).sum()) for label in range(3)],
            pooled={},
            horizons={},
        )
        for name in HEADS:
            item["pooled"][name] = _aggregate(
                vectors[name], predictions[name], truth, mask
            )
        item["pooled"]["observation_mse"] = float(observation_error[mask].mean())
        for h in HORIZONS:
            item["horizons"][str(h)] = {
                name: _aggregate(vectors[name], predictions[name], truth, mask, h)
                for name in HEADS
            }
            item["horizons"][str(h)]["observation_mse"] = float(
                observation_error[mask, h - 1].mean()
            )
        result["by_episode"][str(identity)] = item
    result["coverage"] = dict(
        class_counts=[int((truth == label).sum()) for label in range(3)],
        class_episode_counts=[
            sum(bool((truth[episode == e] == label).any()) for e in range(10))
            for label in range(3)
        ],
        positive_episodes=len(positive),
        positive_episode_seeds=positive,
        status="adequate" if len(positive) >= 5 else "inconclusive",
        threshold=5,
        unit="unique environment groups within retained anchor windows",
    )
    return _json(result)


def _integer(value, label, allowed=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(label + " must be an integer")
    value = int(value)
    if allowed is not None and value not in allowed:
        raise ValueError(label + " is not preregistered")
    return value


def _hierarchical(groups, *, seed=912731):
    """Paired deltas [continuation_seed, environment_group], retaining n=3."""
    groups = _array(groups, "paired groups", (3, 10))
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, 3, (BOOTSTRAP_REPS, 3))
    environments = rng.integers(0, 10, (BOOTSTRAP_REPS, 3, 10))
    samples = groups[selected[..., None], environments].mean((1, 2))
    return dict(
        mean=float(groups.mean()),
        ci95=[float(x) for x in np.quantile(samples, [0.025, 0.975])],
        by_seed={str(s): float(groups[i].mean()) for i, s in enumerate(SEEDS)},
        independent_seeds=3,
        environment_groups_per_seed=10,
        bootstrap_reps=BOOTSTRAP_REPS,
        interpretation="weak exploratory n=3 paired hierarchical bootstrap; not confirmatory evidence",
    )


def _metric_value(
    summary, environment, scope, metric, horizon=None, head="predicted_student"
):
    try:
        item = summary["by_episode"][str(environment)]
        view = item["pooled"] if scope == "pooled" else item["horizons"][str(horizon)]
        value = view[head][metric]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing per-environment prediction metric") from exc
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError("invalid per-environment error metric")
    return float(value)


def _learning_curve(groups, *, seed=912731):
    """Return scores [milestone,continuation_seed,environment], not same-env paths.

    The registered environment seeds change at each checkpoint. Each bootstrap
    replicate therefore shares its sampled continuation seeds over the curve,
    but samples the paired environmental groups separately at each checkpoint.
    """
    groups = _array(groups, "learning curve groups", (3, 3, 10))
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, 3, (BOOTSTRAP_REPS, 3))
    environments = rng.integers(0, 10, (BOOTSTRAP_REPS, 3, 3, 10))
    sample = groups[
        np.arange(3)[None, :, None, None], selected[:, None, :, None], environments
    ].mean((2, 3))
    original = groups.mean(2).T
    result = {}
    for name, weight in (
        ("checkpoint_mean", np.ones(3) / 3),
        ("normalized_auc", np.array([0.25, 0.5, 0.25])),
        ("auc_return_native_steps", np.array([20000.0, 40000.0, 20000.0])),
    ):
        values, observed = sample @ weight, original @ weight
        result[name] = dict(
            mean=float(observed.mean()),
            ci95=[float(x) for x in np.quantile(values, [0.025, 0.975])],
            by_seed={str(s): float(observed[i]) for i, s in enumerate(SEEDS)},
            independent_seeds=3,
            environment_groups_per_seed_per_milestone=10,
            bootstrap_reps=BOOTSTRAP_REPS,
            interpretation="weak exploratory n=3; paired environment resampling independently at each milestone",
        )
    return result


def paired_report(cell_results, probe_summaries):
    """Report all nine cells and81 matched source-policy blocks, with fixed screen.

    Cells: {arm,seed,evaluations:{milestone:{episodes:[{seed,return}...]}}}; the
    retained runner spelling return_ is also accepted. Probes: records with
    arm,seed,milestone,source_arm,summary (from summarize_arrays). Every target
    checkpoint is scored on all A/B/C source-policy blocks with matching ten
    environment seeds. Duplicate/incomplete/unmatched inputs raise ValueError.

    Final80k screen: C pooled15 imagined-student MSE <=.9 B; student mixture NLL
    and cumulativeH15 MSE <=1.1 B; average each score equally over3 seeds. Every
    seed's paired C-B evaluation return must be positive and their mean >=50.
    Fewer than5 reward-positive environment groups in any seed is inconclusive.
    Confidence intervals describe uncertainty and are not extra screen gates.
    Existing pooled/horizons contrasts compare the student family (A's shadow
    student); deployed_control separately compares A original versus B/C student
    without manufacturing categorical NLL for A. Return curve means and AUCs
    are descriptive, not additional screening criteria.
    """
    if not isinstance(cell_results, (list, tuple)) or not isinstance(
        probe_summaries, (list, tuple)
    ):
        raise ValueError("cells and probes must be complete record lists")
    cells, evaluation = {}, {}
    for cell in cell_results:
        if not isinstance(cell, Mapping) or cell.get("arm") not in ARMS:
            raise ValueError("invalid cell arm")
        arm, seed = cell["arm"], _integer(cell.get("seed"), "continuation seed", SEEDS)
        key = (arm, seed)
        if key in cells:
            raise ValueError("duplicate cell")
        cells[key] = cell
        ev = cell.get("evaluations", {})
        if (
            not isinstance(ev, Mapping)
            or set(map(str, ev)) != {str(m) for m in MILESTONES}
            or len(ev) != 3
        ):
            raise ValueError("missing or duplicate evaluation milestones")
        for milestone in MILESTONES:
            value = ev.get(str(milestone), ev.get(milestone))
            records = value.get("episodes", []) if isinstance(value, Mapping) else []
            if not isinstance(records, (list, tuple)) or len(records) != 10:
                raise ValueError("each evaluation requires ten complete environments")
            outcomes = {}
            for episode in records:
                if not isinstance(episode, Mapping):
                    raise ValueError("invalid evaluation episode record")
                identity = _integer(episode.get("seed"), "environment seed")
                if not 0 <= identity < 2**32:
                    raise ValueError("environment seed is not uint32")
                score = episode.get("return", episode.get("return_"))
                if (
                    isinstance(score, (bool, np.bool_))
                    or not isinstance(score, (int, float, np.integer, np.floating))
                    or not np.isfinite(score)
                    or not 0 <= score <= 1000
                ):
                    raise ValueError("invalid environment return")
                if identity in outcomes or (
                    "return" in episode
                    and "return_" in episode
                    and episode["return"] != episode["return_"]
                ):
                    raise ValueError(
                        "duplicate environment or inconsistent return aliases"
                    )
                outcomes[identity] = float(score)
            evaluation[(arm, seed, milestone)] = outcomes
    if set(cells) != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("exactly nine arm/seed cells required")
    for seed in SEEDS:
        for milestone in MILESTONES:
            if any(
                set(evaluation[(arm, seed, milestone)])
                != set(evaluation[("A", seed, milestone)])
                for arm in ARMS
            ):
                raise ValueError("evaluation environments are not paired across arms")
    probes = {}
    for record in probe_summaries:
        if (
            not isinstance(record, Mapping)
            or record.get("arm") not in ARMS
            or record.get("source_arm") not in ARMS
        ):
            raise ValueError("invalid target/source policy arm")
        key = (
            record["arm"],
            _integer(record.get("seed"), "probe seed", SEEDS),
            _integer(record.get("milestone"), "probe milestone", MILESTONES),
            record["source_arm"],
        )
        if key in probes:
            raise ValueError("duplicate probe source block")
        summary = record.get("summary")
        if (
            not isinstance(summary, Mapping)
            or summary.get("rows") != 40
            or summary.get("particles") != 8
            or summary.get("episodes") != 10
        ):
            raise ValueError("invalid fixed probe design")
        try:
            groups = summary["by_episode"]
            actual = {int(k) for k in groups}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid probe environment groups") from exc
        if (
            not isinstance(groups, Mapping)
            or set(groups) != {str(k) for k in actual}
            or len(groups) != 10
            or actual != set(evaluation[key[:3]])
        ):
            raise ValueError("probe/evaluation environment identities differ")
        for identity, group in groups.items():
            if not isinstance(group, Mapping) or group.get("episode_seed") != int(
                identity
            ):
                raise ValueError("invalid per-environment metadata")
            counts = group.get("class_counts")
            if (
                not isinstance(counts, list)
                or len(counts) != 3
                or any(type(v) is not int or v < 0 for v in counts)
                or sum(counts) != 60
            ):
                raise ValueError("invalid per-environment class coverage")
            positive = group.get("positive_count")
            if type(positive) is not int or positive != sum(counts[1:]):
                raise ValueError("invalid per-environment positive coverage")
        probes[key] = summary
    expected = {
        (a, s, m, source)
        for a in ARMS
        for s in SEEDS
        for m in MILESTONES
        for source in ARMS
    }
    if set(probes) != expected:
        raise ValueError("all81 target/seed/milestone/source-policy probes required")
    for seed in SEEDS:
        for milestone in MILESTONES:
            for source in ARMS:
                reference = probes[("A", seed, milestone, source)]
                if (
                    not isinstance(reference.get("truth_digest"), str)
                    or len(reference["truth_digest"]) != 64
                    or any(
                        c not in "0123456789abcdef" for c in reference["truth_digest"]
                    )
                ):
                    raise ValueError("missing matched truth fingerprint")
                for arm in ARMS:
                    candidate = probes[(arm, seed, milestone, source)]
                    if candidate.get("truth_digest") != reference["truth_digest"]:
                        raise ValueError(
                            "target models did not receive the same source-policy truths"
                        )
                    for identity in reference["by_episode"]:
                        left, right = (
                            reference["by_episode"][identity],
                            candidate["by_episode"][identity],
                        )
                        if (left.get("class_counts"), left.get("positive_count")) != (
                            right.get("class_counts"),
                            right.get("positive_count"),
                        ):
                            raise ValueError(
                                "matched reward coverage differs across target models"
                            )

    def scores(arm, milestone, scope, metric, horizon=None, head="predicted_student"):
        result = []
        for seed in SEEDS:
            identities = sorted(evaluation[(arm, seed, milestone)])
            result.append(
                [
                    np.mean(
                        [
                            _metric_value(
                                probes[(arm, seed, milestone, source)],
                                env,
                                scope,
                                metric,
                                horizon,
                                head,
                            )
                            for source in ARMS
                        ]
                    )
                    for env in identities
                ]
            )
        return np.asarray(result)

    deployed_heads = dict(
        A="predicted_original", B="predicted_student", C="predicted_student"
    )
    contrasts = {}
    for left, right in (("B", "A"), ("C", "B"), ("C", "A")):
        milestones = {}
        for milestone in MILESTONES:
            returns = np.asarray(
                [
                    [
                        evaluation[(left, seed, milestone)][env]
                        - evaluation[(right, seed, milestone)][env]
                        for env in sorted(evaluation[(left, seed, milestone)])
                    ]
                    for seed in SEEDS
                ]
            )
            item = dict(
                returns=_hierarchical(returns),
                pooled={},
                horizons={},
                prediction_family="same-family student diagnostic; A is its shadow student, not its deployed original head",
                deployed_control=dict(
                    heads={left: deployed_heads[left], right: deployed_heads[right]},
                    pooled={},
                    horizons={},
                    categorical_nll=None,
                    categorical_nll_note="this block reports MSE/cumulative only; original native loss is not comparable categorical NLL",
                ),
            )
            for metric in ("reward_mse", "nll", "cumulative_reward_mse"):
                item["pooled"][metric] = _hierarchical(
                    scores(left, milestone, "pooled", metric)
                    - scores(right, milestone, "pooled", metric)
                )
            for h in HORIZONS:
                item["horizons"][str(h)] = {
                    metric: _hierarchical(
                        scores(left, milestone, "horizon", metric, h)
                        - scores(right, milestone, "horizon", metric, h)
                    )
                    for metric in ("reward_mse", "nll", "cumulative_reward_mse")
                }
            for scope, horizon in (
                ("pooled", None),
                *(("horizon", h) for h in HORIZONS),
            ):
                view = {
                    metric: _hierarchical(
                        scores(
                            left,
                            milestone,
                            scope,
                            metric,
                            horizon,
                            deployed_heads[left],
                        )
                        - scores(
                            right,
                            milestone,
                            scope,
                            metric,
                            horizon,
                            deployed_heads[right],
                        )
                    )
                    for metric in ("reward_mse", "cumulative_reward_mse")
                }
                if scope == "pooled":
                    item["deployed_control"]["pooled"] = view
                else:
                    item["deployed_control"]["horizons"][str(horizon)] = view
            milestones[str(milestone)] = item
        contrasts[left + "-" + right] = milestones
    curve_groups = {
        arm: np.asarray(
            [
                [
                    [
                        evaluation[(arm, seed, milestone)][env]
                        for env in sorted(evaluation[(arm, seed, milestone)])
                    ]
                    for seed in SEEDS
                ]
                for milestone in MILESTONES
            ]
        )
        for arm in ARMS
    }
    learning_curves = dict(
        milestones=list(MILESTONES),
        native_step_axis="continuation collection steps; evaluations excluded",
        normalized_auc="trapezoidal AUC /80000; weights .25,.5,.25; reward-return units",
        checkpoint_mean="equal mean over checkpoints0,40000,80000",
        arms={arm: _learning_curve(curve_groups[arm]) for arm in ARMS},
        contrasts={
            left
            + "-"
            + right: _learning_curve(curve_groups[left] - curve_groups[right])
            for left, right in (("B", "A"), ("C", "B"), ("C", "A"))
        },
    )
    coverage = {}
    for seed in SEEDS:
        positive = []
        for env in sorted(evaluation[("A", seed, 80000)]):
            if any(
                probes[("A", seed, 80000, source)]["by_episode"][str(env)][
                    "positive_count"
                ]
                > 0
                for source in ARMS
            ):
                positive.append(env)
        coverage[str(seed)] = dict(
            positive_environment_groups=len(positive),
            positive_environment_seeds=positive,
            required=5,
            adequate=len(positive) >= 5,
        )
    comparisons, reasons = {}, []
    for metric, limit in (
        ("reward_mse", 0.9),
        ("nll", 1.1),
        ("cumulative_reward_mse", 1.1),
    ):
        b, c = (scores(arm, 80000, "pooled", metric) for arm in ("B", "C"))
        baseline, candidate = float(b.mean()), float(c.mean())
        ratio = (
            candidate / baseline if baseline > 0 else (1.0 if candidate == 0 else None)
        )
        # Exact boundary values can move a few ulps during equal-weight means;
        # tolerate only floating-point summation noise, not a scientific margin.
        passed = bool(
            ratio is not None and ratio <= limit + 16 * np.finfo(float).eps * limit
        )
        comparisons[metric] = dict(
            B=baseline,
            C=candidate,
            C_over_B=ratio,
            maximum_ratio=limit,
            passed=passed,
            by_seed={
                str(seed): dict(B=float(b[i].mean()), C=float(c[i].mean()))
                for i, seed in enumerate(SEEDS)
            },
        )
        if not passed:
            reasons.append(metric + " threshold not met")
    returned = contrasts["C-B"]["80000"]["returns"]
    return_passed = (
        all(delta > 0 for delta in returned["by_seed"].values())
        and returned["mean"] >= 50
    )
    if not return_passed:
        reasons.append("C-B return must be positive in every seed and mean>=50")
    sufficient = all(item["adequate"] for item in coverage.values())
    if not sufficient:
        reasons.append(
            "fewer than five reward-positive environment groups in at least one seed"
        )
    passed = sufficient and not reasons
    decision = dict(
        passed=passed,
        status=(
            "inconclusive"
            if not sufficient
            else ("screen_passed" if passed else "screen_not_passed")
        ),
        reasons=reasons,
        milestone=80000,
        predicted_student=comparisons,
        returns=dict(returned, passed=return_passed, required_mean_benefit=50),
        coverage=coverage,
    )
    return _json(
        dict(
            schema="posterior-reward-paired-report-v1",
            decision=decision,
            contrasts=contrasts,
            learning_curves=learning_curves,
            inference=dict(
                independent_continuation_seeds=3,
                paired_environment_groups_per_seed=10,
                source_policy_strata=3,
                bootstrap_reps=BOOTSTRAP_REPS,
                unit="seed then paired environment; all source-policy blocks remain together",
                limitation="n=3 is weak exploratory evidence; one parent world-model seed; no superiority claim",
            ),
            cells=list(cell_results),
            probe_summaries=list(probe_summaries),
        )
    )
