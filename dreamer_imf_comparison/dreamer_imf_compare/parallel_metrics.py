"""Independent NumPy scoring for the frozen parallel-trajectory diagnostic.

Errors use the particle *mean*, not the mean particle error. Observations and
instantaneous rewards are scored at the named horizon; returns sum decisions
1..h. Observation coordinates are divided by train-only population standard
deviations. All reward quantities use the RMS *full training trajectory return*
(no centering), with both scales floored at .01. This convention does not fit
anything on validation/test predictions.

Every point estimate averages rows within episode and then episodes equally.
Percentile intervals resample those episode means; particles, anchors, action
branches, and repeated replay rows never become bootstrap replicates. CRPS is
the exact score of the finite empirical predictive distribution, not a Gaussian
approximation. Intervals are central 90% empirical quantile intervals.

Material discrepancies are declared only when a simultaneous one-sided 95%
episode-bootstrap lower bound exceeds .1**2 in normalized units. Action effects
subtract particle-mean variance from squared error; distribution comparisons
use the unbiased marginal energy statistic (which can legitimately be negative
at finite S). These reduce false failures from independent Monte Carlo draws.
The threshold is a diagnostic convention, not a theorem about physical fidelity.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

HORIZONS = (1, 3, 5, 10, 15)
MIN_POSITIVE_EPISODES = 5
MATERIAL_TOLERANCE = 0.1


def _array(value, name, ndim):
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex-valued metrics are unsupported")
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be nonempty with {ndim} dimensions")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _ids(value, name, n, allowed=None):
    array = np.asarray(value)
    if array.shape != (n,) or array.dtype.kind not in "iu" or (array < 0).any():
        raise ValueError(f"{name} must be {n} nonnegative integer labels")
    if (array > np.iinfo(np.int64).max).any():
        raise ValueError(f"{name} labels exceed the signed 64-bit range")
    if allowed is not None and not np.isin(array, allowed).all():
        raise ValueError(f"{name} has unsupported labels")
    return array


def _scales(obs_scale, return_scale, dimensions):
    obs = _array(obs_scale, "obs_scale", 1)
    try:
        if np.iscomplexobj(return_scale):
            raise ValueError("complex scales are unsupported")
        ret = np.asarray(return_scale, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("return_scale must be one finite positive scalar") from exc
    if obs.shape != (dimensions,) or (obs <= 0).any():
        raise ValueError(
            "obs_scale must have one positive value per observation coordinate"
        )
    if ret.ndim != 0 or not np.isfinite(ret) or float(ret) <= 0:
        raise ValueError("return_scale must be one finite positive scalar")
    return obs, float(ret)


def _finite_json(result):
    # This also catches overflow from finite-but-unrepresentable input arithmetic.
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("metrics produced nonfinite or non-JSON output") from exc
    return result


def fit_scales(train_obs, train_rewards, *, split=None):
    """Return (obs_std[O], full_return_rms), fitted exclusively to supplied train rows.

    Optional split labels are checked to be all zero, providing a fail-closed
    boundary when a caller still holds an unsliced dataset. Population std uses
    every train row/time; return RMS uses each train row's full sum over time.
    Neither function can infer provenance when split labels are not supplied.
    """
    obs = _array(train_obs, "train_obs", 3)
    rewards = _array(train_rewards, "train_rewards", 2)
    if rewards.shape != obs.shape[:2]:
        raise ValueError("train observations and rewards must share N,H")
    if split is not None:
        _ids(split, "split", len(obs), allowed=(0,))
    with np.errstate(over="raise", invalid="raise"):
        try:
            obs_scale = np.maximum(obs.std(axis=(0, 1)), 0.01)
            return_scale = max(float(np.sqrt(np.mean(rewards.sum(1) ** 2))), 0.01)
        except FloatingPointError as exc:
            raise ValueError("training scale computation overflowed") from exc
    return obs_scale, return_scale


def split_counts(episode, split, rewards=None):
    """Validate whole-episode splitting and count rows/episodes per fixed split.

    If rewards are supplied, report positive-, zero-, and negative-return rows
    and episodes separately, as well as episodes with any positive reward step.
    Positive-return, zero-return and negative-return episode sets can overlap
    because one episode contributes several counterfactual action branches.
    """
    raw = np.asarray(episode)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("episode must be nonempty and one-dimensional")
    episodes = _ids(episode, "episode", len(raw))
    labels = _ids(split, "split", len(raw), (0, 1, 2))
    reward_array = None if rewards is None else _array(rewards, "rewards", 2)
    if reward_array is not None and len(reward_array) != len(episodes):
        raise ValueError("rewards and episode lengths differ")
    for value in np.unique(episodes):
        if len(np.unique(labels[episodes == value])) != 1:
            raise ValueError(f"episode {value} crosses dataset splits")
    result = {}
    for label, name in enumerate(("train", "validation", "test")):
        mask = labels == label
        result[name] = {
            "rows": int(mask.sum()),
            "episodes": int(len(np.unique(episodes[mask]))),
        }
        if reward_array is not None:
            result[name].update(_reward_counts(reward_array[mask], episodes[mask]))
    return _finite_json(result)


class _Bootstrap:
    """Shared episode resamples keep related metrics reproducible and inexpensive."""

    def __init__(self, reps, random_seed):
        if (
            isinstance(reps, bool)
            or not isinstance(reps, (int, np.integer))
            or reps < 2
        ):
            raise ValueError("bootstrap_reps must be an integer >=2")
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, (int, np.integer))
            or random_seed < 0
        ):
            raise ValueError("random_seed must be a nonnegative integer")
        self.reps, self.seed, self.indices = int(reps), int(random_seed), {}

    def stat(self, values, episodes, *, family_size=1):
        values = np.asarray(values, dtype=np.float64)
        if (
            values.ndim != 1
            or values.shape != episodes.shape
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "bootstrap requires one finite value and episode label per unit"
            )
        unique, inverse, counts = np.unique(
            episodes, return_inverse=True, return_counts=True
        )
        n = len(unique)
        base = {
            "mean": None,
            "ci95": [None, None],
            "lower_simultaneous95": None,
            "n_episodes": n,
            "n_units": len(values),
            "status": "unavailable",
        }
        if not n:
            return base
        means = np.bincount(inverse, weights=values) / counts
        base["mean"] = float(means.mean())
        base["status"] = "adequate" if n >= 2 else "inconclusive"
        if n < 2:
            return base
        key = tuple(int(x) for x in unique)
        if key not in self.indices:
            self.indices[key] = np.random.default_rng(self.seed).integers(
                0, n, size=(self.reps, n)
            )
        replicates = means[self.indices[key]].mean(1)
        base["ci95"] = [float(x) for x in np.quantile(replicates, [0.025, 0.975])]
        base["lower_simultaneous95"] = float(
            np.quantile(replicates, 0.05 / family_size)
        )
        return base


def _reward_counts(rewards, episodes):
    returns = rewards.sum(axis=1)
    if not np.isfinite(returns).all():
        raise ValueError("reward sums overflowed")
    result = {}
    for name, mask in (
        ("positive", returns > 0),
        ("zero", returns == 0),
        ("negative", returns < 0),
    ):
        result[f"{name}_return_rows"] = int(mask.sum())
        result[f"{name}_return_episodes"] = int(len(np.unique(episodes[mask])))
    result["positive_reward_steps"] = int((rewards > 0).sum())
    result["zero_reward_steps"] = int((rewards == 0).sum())
    result["negative_reward_steps"] = int((rewards < 0).sum())
    result["episodes_with_positive_reward"] = int(
        len(np.unique(episodes[(rewards > 0).any(1)]))
    )
    return result


def _empirical_crps(samples, truth):
    """Exact E|X-y| - .5 E|X-X'| for the finite empirical distribution."""
    score = np.mean(np.abs(samples - truth[:, None]), axis=1) - 0.5 * _within_distance(
        samples
    )
    # An empirical CRPS is nonnegative analytically; remove negative roundoff.
    return np.maximum(score, 0.0)


def _within_distance(samples, *, unbiased=False):
    """Mean absolute pair distance from sorted gaps, stable under translation."""
    n = samples.shape[1]
    gaps = np.diff(np.sort(samples, axis=1), axis=1)
    indices = np.arange(1, n)
    weights = (indices * (n - indices)).reshape((1, n - 1) + (1,) * (samples.ndim - 2))
    denominator = n * (n - 1) if unbiased else n**2
    return 2 * np.sum(gaps * weights, axis=1) / denominator


def _interval_coverage(samples, truth):
    lo, hi = np.quantile(samples, (0.05, 0.95), axis=1)
    return ((truth >= lo) & (truth <= hi)).astype(float)


def _mean_variance(samples):
    if samples.shape[1] == 1:
        return np.zeros(samples.shape[:1] + samples.shape[2:], dtype=float)
    return samples.var(axis=1, ddof=1) / samples.shape[1]


def _fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        dtype = "<i8" if np.asarray(array).dtype.kind in "iu" else "<f8"
        canonical = np.ascontiguousarray(array, dtype=dtype)
        digest.update(dtype.encode())
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _material(stat, tolerance):
    lower = stat["lower_simultaneous95"]
    return bool(stat["n_episodes"] >= 5 and lower is not None and lower > tolerance**2)


def _action_metrics(
    obs_samples, ret_samples, truth_obs, truth_ret, episodes, anchors, plans, bootstrap
):
    # Average duplicate replay rows inside each (episode,anchor,plan), retaining
    # particle index. Repetition does not manufacture more independent particles.
    groups = {}
    for row, key in enumerate(zip(episodes, anchors, plans)):
        groups.setdefault(tuple(int(x) for x in key), []).append(row)
    summaries = {}
    for key, rows in groups.items():
        summaries[key] = (
            obs_samples[rows].mean(0),
            ret_samples[rows].mean(0),
            truth_obs[rows].mean(0),
            float(truth_ret[rows].mean()),
        )
    anchor_keys = sorted(set(key[:2] for key in summaries))
    complete = sum(
        all((*key, plan) in summaries for plan in range(4)) for key in anchor_keys
    )
    effects, all_accuracy, all_episodes = {}, [], []
    for plan_a, plan_b in combinations(range(4), 2):
        units = []
        for episode, anchor in anchor_keys:
            left, right = (episode, anchor, plan_a), (episode, anchor, plan_b)
            if left not in summaries or right not in summaries:
                continue
            oa, ra, ta, ya = summaries[left]
            ob, rb, tb, yb = summaries[right]
            pred_delta = float(rb.mean() - ra.mean())
            true_delta = yb - ya
            obs_error = ob.mean(0) - oa.mean(0) - (tb - ta)
            obs_var = _mean_variance(oa[None])[0] + _mean_variance(ob[None])[0]
            ret_var = float(_mean_variance(ra[None])[0] + _mean_variance(rb[None])[0])
            units.append(
                (
                    episode,
                    pred_delta,
                    true_delta,
                    float(np.mean(obs_error**2)),
                    (pred_delta - true_delta) ** 2,
                    float(np.mean(obs_error**2 - obs_var)),
                    (pred_delta - true_delta) ** 2 - ret_var,
                )
            )
            if true_delta != 0:
                # A predicted tie earns half credit; true ties are unidentifiable.
                accuracy = (
                    0.5
                    if pred_delta == 0
                    else float(np.sign(pred_delta) == np.sign(true_delta))
                )
                all_accuracy.append(accuracy)
                all_episodes.append(episode)
        ep = np.array([x[0] for x in units], dtype=np.int64)
        effect = {"anchor_pairs": len(units)}
        names = (
            "predicted_return_effect",
            "true_return_effect",
            "observation_effect_mse",
            "return_effect_mse",
            "observation_effect_debiased_mse",
            "return_effect_debiased_mse",
        )
        for column, name in enumerate(names, start=1):
            effect[name] = bootstrap.stat(
                np.array([x[column] for x in units]), ep, family_size=60
            )
        effect["material_failure"] = any(
            _material(effect[name], MATERIAL_TOLERANCE) for name in names[-2:]
        )
        effects[f"{plan_a}_to_{plan_b}"] = effect
    return {
        "complete_anchors": complete,
        "total_anchors": len(anchor_keys),
        "coverage_adequate": complete == len(anchor_keys),
        "paired_effects": effects,
        "ranking_pairwise_accuracy": bootstrap.stat(
            np.asarray(all_accuracy), np.asarray(all_episodes)
        ),
        "strict_comparable_pairs": len(all_accuracy),
        "material_failure": any(x["material_failure"] for x in effects.values()),
    }


def summarize(
    pred_obs,
    pred_rewards,
    truth_obs,
    truth_rewards,
    episode,
    anchor,
    plan,
    obs_scale,
    return_scale,
    *,
    bootstrap_reps=2000,
    random_seed=0,
):
    """Return JSON-ready endpoint/return metrics and paired action diagnostics.

    ``pred_obs[N,S,H,O]``, ``pred_rewards[N,S,H]``, and truth arrays ``[N,H,O]`` /
    ``[N,H]`` must have H>=15; action plan labels are 0,1,2,3. Call with held-out
    test rows for test conclusions. ``coverage.adequate`` requires at least five
    distinct episodes containing positive rewards and all four plans per anchor.
    No positive-reward examples is *inconclusive*, never a successful zero-error
    result. Empty conditioned groups use JSON null with status unavailable.
    """
    obs = _array(pred_obs, "pred_obs", 4)
    rewards = _array(pred_rewards, "pred_rewards", 3)
    target_obs = _array(truth_obs, "truth_obs", 3)
    target_rewards = _array(truth_rewards, "truth_rewards", 2)
    n, particles, horizon, dimensions = obs.shape
    if horizon < max(HORIZONS) or rewards.shape != (n, particles, horizon):
        raise ValueError("predictions must share N,S,H with H>=15")
    if target_obs.shape != (n, horizon, dimensions) or target_rewards.shape != (
        n,
        horizon,
    ):
        raise ValueError("truth shapes must match prediction N,H,O")
    episodes, anchors = _ids(episode, "episode", n), _ids(anchor, "anchor", n)
    plans = _ids(plan, "plan", n, (0, 1, 2, 3))
    oscale, rscale = _scales(obs_scale, return_scale, dimensions)
    bootstrap = _Bootstrap(bootstrap_reps, random_seed)
    obs, target_obs = obs / oscale, target_obs / oscale
    rewards, target_rewards = rewards / rscale, target_rewards / rscale
    cumulative, truth_cumulative = rewards.cumsum(2), target_rewards.cumsum(1)
    result = {
        "schema_version": 1,
        "n_rows": n,
        "n_particles": particles,
        "n_episodes": int(len(np.unique(episodes))),
        "horizons": {},
        "data_fingerprint": _fingerprint(
            target_obs, target_rewards, episodes, anchors, plans, oscale, rscale
        ),
        "normalization": {
            "obs_scale": oscale.tolist(),
            "return_scale": rscale,
            "aggregation": "equal episode means; rows averaged within episode",
            "observation_error": "endpoint particle-mean MSE",
            "reward_error": "endpoint and cumulative particle-mean MSE",
            "interval_probability": 0.9,
            "bootstrap_reps": bootstrap.reps,
        },
        "coverage": _reward_counts(target_rewards[:, : max(HORIZONS)], episodes),
    }
    for h in HORIZONS:
        pobs, tobs = obs[:, :, h - 1], target_obs[:, h - 1]
        prew, trew = rewards[:, :, h - 1], target_rewards[:, h - 1]
        pret, tret = cumulative[:, :, h - 1], truth_cumulative[:, h - 1]
        row_values = {
            "observation_mse": ((pobs.mean(1) - tobs) ** 2).mean(-1),
            "reward_mse": (prew.mean(1) - trew) ** 2,
            "cumulative_reward_mse": (pret.mean(1) - tret) ** 2,
            "observation_crps": _empirical_crps(pobs, tobs).mean(-1),
            "reward_crps": _empirical_crps(prew, trew),
            "cumulative_reward_crps": _empirical_crps(pret, tret),
            "observation_interval_coverage": _interval_coverage(pobs, tobs).mean(-1),
            "reward_interval_coverage": _interval_coverage(prew, trew),
            "cumulative_reward_interval_coverage": _interval_coverage(pret, tret),
        }
        metrics = {
            name: bootstrap.stat(value, episodes) for name, value in row_values.items()
        }
        metrics["reward_counts"] = _reward_counts(target_rewards[:, :h], episodes)
        for label, mask in (
            ("positive_return", tret > 0),
            ("zero_return", tret == 0),
            ("negative_return", tret < 0),
        ):
            metrics[label] = {
                name: bootstrap.stat(value[mask], episodes[mask])
                for name, value in row_values.items()
            }
        metrics["actions"] = _action_metrics(
            pobs, pret, tobs, tret, episodes, anchors, plans, bootstrap
        )
        result["horizons"][str(h)] = metrics
    action_coverage = all(
        x["actions"]["coverage_adequate"] for x in result["horizons"].values()
    )
    adequate = (
        result["coverage"]["episodes_with_positive_reward"] >= MIN_POSITIVE_EPISODES
        and action_coverage
    )
    result["coverage"].update(
        {
            "adequate": bool(adequate),
            "status": "adequate" if adequate else "inconclusive",
            "required_positive_episodes": MIN_POSITIVE_EPISODES,
            "action_coverage_adequate": action_coverage,
        }
    )
    result["failures"] = {
        "material_action_failure": any(
            x["actions"]["material_failure"] for x in result["horizons"].values()
        ),
        "material_tolerance": MATERIAL_TOLERANCE,
        "definition": "simultaneous lower95 of particle-noise-debiased action-effect MSE > tolerance squared",
    }
    return _finite_json(result)


def _energy_unbiased(left, right):
    """Unbiased marginal energy discrepancy, averaging coordinates if present.

    Uses independent sample-set distances, never correspondence of particles.
    Sorting handles within-sample terms without an S*S*O allocation; cross terms
    are accumulated over the smaller sample axis to bound working memory.
    """
    if left.shape[1] > right.shape[1]:
        left, right = right, left
    cross = np.zeros(left.shape[:1] + left.shape[2:])
    for sample in range(left.shape[1]):
        cross += np.abs(left[:, sample : sample + 1] - right).mean(1) / left.shape[1]
    result = (
        2 * cross
        - _within_distance(left, unbiased=True)
        - _within_distance(right, unbiased=True)
    )
    return result.mean(-1) if result.ndim == 2 else result


def compare_distributions(
    obs_a,
    rewards_a,
    obs_b,
    rewards_b,
    episode,
    obs_scale,
    return_scale,
    *,
    prefix_horizon=None,
    bootstrap_reps=2000,
    random_seed=0,
    material_tolerance=MATERIAL_TOLERANCE,
):
    """Compare direct/composed rollouts or identical-action-prefix distributions.

    A and B share N,H,O but may have different particle counts, each >=2. Rows
    must be aligned by episode/anchor by the caller. Omit ``prefix_horizon`` for
    the five temporal composition horizons. Set it to the final unchanged-action
    decision for plan2/3 prefix leakage; *every* prefix decision is then checked.
    This tests marginal distributions of observations and cumulative returns,
    not pathwise equality or a claim of joint trajectory-distribution equality.

    The signed unbiased energy discrepancy subtracts finite-particle within-set
    dispersion. Confidence intervals resample whole episodes and simultaneously
    correct all horizons/metrics within this comparison. At least five episodes
    are needed for an evaluable diagnostic; fewer yields inconclusive. Adequate
    episode coverage and no detected failure do not prove equivalence. A particle
    permutation cannot change the score. Large dispersion can limit power.
    """
    a, b = _array(obs_a, "obs_a", 4), _array(obs_b, "obs_b", 4)
    ra, rb = _array(rewards_a, "rewards_a", 3), _array(rewards_b, "rewards_b", 3)
    n, sa, horizon, dimensions = a.shape
    if (
        b.shape[0] != n
        or b.shape[2:] != (horizon, dimensions)
        or min(sa, b.shape[1]) < 2
    ):
        raise ValueError(
            "distribution observations require common N,H,O and >=2 particles"
        )
    if ra.shape != a.shape[:3] or rb.shape != b.shape[:3]:
        raise ValueError("distribution rewards must share observation N,S,H")
    if prefix_horizon is None:
        if horizon < max(HORIZONS):
            raise ValueError("temporal comparison requires H>=15")
        horizons = HORIZONS
    else:
        if (
            isinstance(prefix_horizon, bool)
            or not isinstance(prefix_horizon, (int, np.integer))
            or not 1 <= prefix_horizon <= horizon
        ):
            raise ValueError("prefix_horizon must be an integer within the trajectory")
        horizons = tuple(range(1, int(prefix_horizon) + 1))
    if not np.isfinite(material_tolerance) or material_tolerance <= 0:
        raise ValueError("material_tolerance must be positive and finite")
    episodes = _ids(episode, "episode", n)
    oscale, rscale = _scales(obs_scale, return_scale, dimensions)
    bootstrap = _Bootstrap(bootstrap_reps, random_seed)
    a, b, ra, rb = (
        a / oscale,
        b / oscale,
        (ra / rscale).cumsum(2),
        (rb / rscale).cumsum(2),
    )
    result = {
        "comparison": "temporal" if prefix_horizon is None else "prefix",
        "prefix_horizon": None if prefix_horizon is None else int(prefix_horizon),
        "n_episodes": int(len(np.unique(episodes))),
        "n_rows": n,
        "particles_a": sa,
        "particles_b": b.shape[1],
        "horizons": {},
        "normalization": {"obs_scale": oscale.tolist(), "return_scale": rscale},
        "bootstrap_reps": bootstrap.reps,
        "simultaneous_family_size": 2 * len(horizons),
        "material_tolerance": float(material_tolerance),
        "definition": "unbiased marginal energy discrepancy; simultaneous episode-bootstrap lower95 > tolerance squared",
        "limitation": "marginal discrepancy diagnostic; no detected failure does not prove equivalence or full joint-distribution equality",
    }
    for h in horizons:
        metrics = {
            "observation_energy": bootstrap.stat(
                _energy_unbiased(a[:, :, h - 1], b[:, :, h - 1]),
                episodes,
                family_size=2 * len(horizons),
            ),
            "cumulative_reward_energy": bootstrap.stat(
                _energy_unbiased(ra[:, :, h - 1], rb[:, :, h - 1]),
                episodes,
                family_size=2 * len(horizons),
            ),
        }
        metrics["material_failure"] = any(
            _material(value, material_tolerance) for value in metrics.values()
        )
        result["horizons"][str(h)] = metrics
    result["material_failure"] = any(
        value["material_failure"] for value in result["horizons"].values()
    )
    result["status"] = "adequate" if result["n_episodes"] >= 5 else "inconclusive"
    return _finite_json(result)


def promotion(
    seed_reports: Mapping,
    baseline: Mapping,
    student_latencies: Mapping,
    baseline_latency: Mapping,
    distribution_checks: Mapping,
    *,
    error_ratio_limit=1.1,
    speedup_min=2.0,
):
    """Fail-closed promotion of the NFE1 candidate, independently for seeds0/1/2.

    ``seed_reports``, ``student_latencies``, ``distribution_checks`` are keyed by
    integer or string seeds. Latencies are ``{'batch1': seconds,'batch64':seconds}``
    measured end-to-end on matched hardware, batches and synchronization; this
    function cannot authenticate measurement provenance. Each distribution list
    must include exactly one adequate temporal and one adequate prefix test.
    The caller must supply the NFE1 variant; summaries alone cannot infer NFE.

    Both endpoint observation MSE and cumulative-return MSE must be <=1.1 times
    the baseline at *every* horizon in *every* seed. A zero baseline requires a
    zero student error (never a denominator floor). Both batch speedups must be
    >=2. Missing/duplicate seeds, inadequate coverage, action/temporal failures,
    or unmatched test data cannot be averaged away. Returns finite JSON with
    ``promote``, ``status`` and specific reasons instead of a favorable partial.
    """
    if (
        not np.isfinite(error_ratio_limit)
        or error_ratio_limit < 1
        or not np.isfinite(speedup_min)
        or speedup_min <= 0
    ):
        raise ValueError("promotion thresholds must be finite; ratio>=1 and speedup>0")
    reasons, per_seed = [], {}

    def nested(value, *keys):
        for key in keys:
            value = value.get(key) if isinstance(value, Mapping) else None
        return value

    def finite_real(value):
        return (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
        )

    def nonnegative(value):
        return finite_real(value) and value >= 0

    def adequate_coverage(report):
        count = nested(report, "coverage", "episodes_with_positive_reward")
        episodes = nested(report, "n_episodes")
        return (
            nested(report, "coverage", "adequate") is True
            and nested(report, "coverage", "action_coverage_adequate") is True
            and nonnegative(count)
            and count >= MIN_POSITIVE_EPISODES
            and nonnegative(episodes)
            and episodes >= count
        )

    def canonical(mapping, label):
        if not isinstance(mapping, Mapping):
            reasons.append(f"{label}: expected seed-keyed mapping")
            return {}
        converted = {str(k): v for k, v in mapping.items()}
        if len(converted) != len(mapping) or set(converted) != {"0", "1", "2"}:
            reasons.append(f"{label}: exactly seeds 0,1,2 required")
        return converted

    reports = canonical(seed_reports, "reports")
    latencies = canonical(student_latencies, "latencies")
    comparisons = canonical(distribution_checks, "distribution_checks")
    if not adequate_coverage(baseline):
        reasons.append("baseline reward/action coverage is inadequate or missing")

    for seed in ("0", "1", "2"):
        report = reports.get(seed, {})
        local, ratios, speedups = [], {}, {}
        if not isinstance(report, Mapping):
            report = {}
        if not adequate_coverage(report):
            local.append("inadequate positive-reward/action coverage")
        if nested(report, "failures", "material_action_failure") is not False:
            local.append("material action failure or missing action diagnostics")
        if nested(report, "failures", "material_tolerance") != MATERIAL_TOLERANCE:
            local.append(
                "action materiality threshold differs from the fixed diagnostic"
            )
        if not report.get("data_fingerprint") or report.get(
            "data_fingerprint"
        ) != nested(baseline, "data_fingerprint"):
            local.append("test data or normalization do not match baseline")
        for h in HORIZONS:
            ratios[str(h)] = {}
            action = nested(report, "horizons", str(h), "actions")
            if (
                nested(action, "coverage_adequate") is not True
                or nested(action, "material_failure") is not False
            ):
                local.append(f"h{h}: missing/inadequate/failed action check")
            for metric in ("observation_mse", "cumulative_reward_mse"):
                value = nested(report, "horizons", str(h), metric, "mean")
                reference = nested(baseline, "horizons", str(h), metric, "mean")
                ratio = None
                if not nonnegative(value) or not nonnegative(reference):
                    local.append(f"h{h} {metric}: missing/nonfinite error")
                else:
                    ratio = (
                        1.0
                        if reference == 0 and value == 0
                        else (float(value / reference) if reference > 0 else None)
                    )
                    if value > error_ratio_limit * reference:
                        local.append(f"h{h} {metric}: error exceeds baseline limit")
                    if ratio is not None and not np.isfinite(ratio):
                        ratio = None
                ratios[str(h)][metric] = ratio
        times = latencies.get(seed, {})
        for batch in ("batch1", "batch64"):
            value = times.get(batch) if isinstance(times, Mapping) else None
            reference = (
                baseline_latency.get(batch)
                if isinstance(baseline_latency, Mapping)
                else None
            )
            if (
                not nonnegative(value)
                or value == 0
                or not nonnegative(reference)
                or reference == 0
            ):
                local.append(f"{batch}: missing/nonpositive/nonfinite latency")
                speedups[batch] = None
            else:
                speedups[batch] = float(reference / value)
                if not np.isfinite(speedups[batch]):
                    speedups[batch] = None
                    local.append(f"{batch}: speedup overflow")
                elif speedups[batch] < speedup_min:
                    local.append(f"{batch}: speedup below required minimum")
        checks = comparisons.get(seed, [])
        if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
            checks = []
        kinds = [
            check.get("comparison") for check in checks if isinstance(check, Mapping)
        ]
        if sorted(str(kind) for kind in kinds) != ["prefix", "temporal"]:
            local.append("one temporal and one prefix distribution check required")
        for check in checks:
            if (
                not isinstance(check, Mapping)
                or check.get("status") != "adequate"
                or check.get("material_failure") is not False
            ):
                local.append(
                    "material distribution failure or inconclusive/missing check"
                )
                continue
            count = check.get("n_episodes")
            if not nonnegative(count) or count < MIN_POSITIVE_EPISODES:
                local.append("distribution check has insufficient independent episodes")
            if check.get("material_tolerance") != MATERIAL_TOLERANCE:
                local.append(
                    "distribution materiality threshold differs from the fixed diagnostic"
                )
            kind = check.get("comparison")
            if kind not in ("prefix", "temporal"):
                continue
            prefix = check.get("prefix_horizon")
            if kind == "prefix" and (
                isinstance(prefix, bool)
                or not isinstance(prefix, (int, np.integer))
                or not 1 <= prefix <= max(HORIZONS)
            ):
                local.append("invalid prefix horizon")
                continue
            expected = HORIZONS if kind == "temporal" else range(1, prefix + 1)
            for h in expected:
                if nested(check, "horizons", str(h), "material_failure") is not False:
                    local.append(f"distribution h{h}: missing or failed diagnostic")
                for metric in ("observation_energy", "cumulative_reward_energy"):
                    value = nested(check, "horizons", str(h), metric, "mean")
                    # Unbiased energy can be negative; require finite, not >=0.
                    if not finite_real(value):
                        local.append(f"distribution h{h}: missing/nonfinite {metric}")
        per_seed[seed] = {
            "passed": not local,
            "reasons": local,
            "error_ratios": ratios,
            "speedups": speedups,
        }
        reasons.extend(f"seed{seed}: {reason}" for reason in local)
    coverage_conclusive = adequate_coverage(baseline) and all(
        adequate_coverage(reports.get(seed)) for seed in ("0", "1", "2")
    )
    status = (
        "promote"
        if not reasons
        else ("do_not_promote" if coverage_conclusive else "inconclusive")
    )
    return _finite_json(
        {
            "promote": not reasons,
            "status": status,
            "reasons": reasons,
            "seeds": per_seed,
            "requirements": {
                "error_ratio_limit": float(error_ratio_limit),
                "speedup_min": float(speedup_min),
                "seeds": [0, 1, 2],
                "horizons": list(HORIZONS),
            },
        }
    )
