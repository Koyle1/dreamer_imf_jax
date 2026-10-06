"""NumPy-only, offline reward readouts and episode-cluster diagnostics.

Ridge minimizes ``sum((y - intercept - standardized_x @ coefficient)**2)
+ alpha * sum(coefficient**2)``. Alpha is NOT multiplied by the training count.
Every input-coordinate mean and population standard deviation is fitted on the
supplied training examples only, with a .01 standard-deviation floor. Callers
must supply train-only rows: an array alone cannot prove its split provenance.

Summary errors use the supplied, shared training-return scale, not a newly fit
test scale. Point estimates average rows within episodes, then episodes equally;
95% percentile intervals resample episodes with replacement. Baseline deltas are
paired candidate-minus-baseline errors (negative is better). Conditional class
statistics average matching steps within each episode. These exploratory CIs are
pointwise, not multiple-comparison adjusted and not causal attribution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

HORIZONS = (1, 3, 5, 10, 15)
STD_FLOOR = 0.01


def _array(value, name, *, ndim=None):
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex data are unsupported")
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if (ndim is not None and result.ndim != ndim) or 0 in result.shape:
        raise ValueError(f"{name} has an invalid or empty shape")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value, name):
    result = _array(value, name, ndim=0)
    if isinstance(value, (bool, np.bool_)) or float(result) <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(result)


def _json(value):
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("calculation produced nonfinite or non-JSON output") from exc
    return value


def _features(value):
    result = _array(value, "features")
    if result.ndim < 2:
        raise ValueError("features must have shape [..., features] with examples")
    return result


def fit_ridge(x_train, y_train, alpha):
    """Fit scalar rewards y[...] from x[...,D]; return a JSON-serializable fit.

    Uses the smaller primal/dual regularized Gram system in float64, avoiding a
    large augmented least-squares matrix. The 9600 x 2560 study uses the primal
    2560 x 2560 system. Positive alpha permits collinear and constant features.
    """
    features = _features(x_train)
    target = _array(y_train, "y_train")
    penalty = _positive(alpha, "alpha")
    if target.shape != features.shape[:-1]:
        raise ValueError("y_train must match every feature axis except the last")
    x = features.reshape(-1, features.shape[-1])
    y = target.reshape(-1)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            mean = x.mean(axis=0)
            std = np.maximum(x.std(axis=0), STD_FLOOR)
            normalized = (x - mean) / std
            intercept = float(y.mean())
            centered = y - intercept
            if x.shape[1] <= x.shape[0]:
                gram = normalized.T @ normalized
                gram.flat[:: gram.shape[0] + 1] += penalty
                coefficient = np.linalg.solve(gram, normalized.T @ centered)
                solver = "primal"
            else:
                gram = normalized @ normalized.T
                gram.flat[:: gram.shape[0] + 1] += penalty
                coefficient = normalized.T @ np.linalg.solve(gram, centered)
                solver = "dual"
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("ridge fitting failed numerically") from exc
    return _json(
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "coefficient": coefficient.tolist(),
            "intercept": intercept,
            "alpha": penalty,
            "n_train_examples": len(x),
            "solver": solver,
            "objective": "sum_squared_error + alpha * squared_coefficient_norm",
            "std_floor": STD_FLOOR,
        }
    )


def predict_ridge(fit, x):
    """Apply stored train-only normalization, preserve leading axes, clip [0,2]."""
    features = _features(x)
    if not isinstance(fit, Mapping):
        raise ValueError("ridge fit must be a mapping")
    try:
        mean = _array(fit["mean"], "fit mean", ndim=1)
        std = _array(fit["std"], "fit std", ndim=1)
        coefficient = _array(fit["coefficient"], "fit coefficient", ndim=1)
        intercept = _array(fit["intercept"], "fit intercept", ndim=0)
    except KeyError as exc:
        raise ValueError("ridge fit is missing required parameters") from exc
    if (
        mean.shape != (features.shape[-1],)
        or std.shape != mean.shape
        or (coefficient.shape != mean.shape or (std < STD_FLOOR).any())
    ):
        raise ValueError(
            "ridge parameter dimensions or standard deviations are invalid"
        )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = ((features - mean) / std) @ coefficient + intercept
        except FloatingPointError as exc:
            raise ValueError("ridge prediction overflowed") from exc
    if not np.isfinite(result).all():
        raise ValueError("ridge prediction is nonfinite")
    return np.clip(result, 0.0, 2.0)


def fit_affine(train_prediction, truth):
    """Fit one global affine calibration on training steps, not one per horizon.

    Constant predictions identify only an intercept; slope zero gives the
    training target mean and avoids an arbitrary unidentifiable slope.
    """
    prediction = _array(train_prediction, "train_prediction")
    target = _array(truth, "truth")
    if prediction.ndim < 1 or target.shape != prediction.shape:
        raise ValueError("calibration arrays must have identical non-scalar shapes")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            pmean, tmean = float(prediction.mean()), float(target.mean())
            centered = prediction.reshape(-1) - pmean
            # Rescale before squaring: a tiny but varying training predictor is
            # not a constant predictor merely because its variance underflows.
            spread = float(np.max(np.abs(centered)))
            if spread > 0:
                normalized = centered / spread
                slope = float(normalized @ (target.reshape(-1) - tmean))
                slope = slope / float(normalized @ normalized) / spread
            else:
                slope = 0.0
            intercept = tmean - slope * pmean
        except FloatingPointError as exc:
            raise ValueError("affine fitting overflowed") from exc
    return _json(
        {
            "slope": slope,
            "intercept": intercept,
            "n_train_examples": prediction.size,
            "constant_prediction": spread == 0,
        }
    )


def predict_affine(fit, pred):
    """Apply stored affine calibration without fitting on prediction rows."""
    prediction = _array(pred, "pred")
    if prediction.ndim < 1 or not isinstance(fit, Mapping):
        raise ValueError("affine prediction requires an array and a fit mapping")
    try:
        slope = _array(fit["slope"], "slope", ndim=0)
        intercept = _array(fit["intercept"], "intercept", ndim=0)
    except KeyError as exc:
        raise ValueError("affine fit is missing required parameters") from exc
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = prediction * slope + intercept
        except FloatingPointError as exc:
            raise ValueError("affine prediction overflowed") from exc
    return np.clip(result, 0.0, 2.0)


def _labels(value, name, n, *, text=False, allowed=None):
    result = np.asarray(value)
    if result.shape != (n,):
        raise ValueError(f"{name} must contain one label per row")
    if text and result.dtype.kind in "US":
        if (result == "").any():
            raise ValueError(f"{name} labels must not be empty")
    elif result.dtype.kind not in "iu" or (result < 0).any():
        raise ValueError(f"{name} must contain nonnegative integer labels")
    elif (result > np.iinfo(np.int64).max).any():
        raise ValueError(f"{name} labels exceed signed 64-bit range")
    if allowed is not None and not np.isin(result, allowed).all():
        raise ValueError(f"{name} has unsupported labels")
    return result


class _EpisodeBootstrap:
    def __init__(self, reps):
        if (
            isinstance(reps, (bool, np.bool_))
            or not isinstance(reps, (int, np.integer))
            or reps < 2
        ):
            raise ValueError("bootstrap_reps must be an integer >=2")
        self.reps, self.indices = int(reps), {}

    def stat(self, values, episodes):
        unique, inverse, counts = np.unique(
            episodes, return_inverse=True, return_counts=True
        )
        n = len(unique)
        result = {
            "mean": None,
            "ci95": [None, None],
            "n_episodes": n,
            "n_units": int(len(values)),
            "status": "unavailable",
        }
        if not n:
            return result
        means = np.bincount(inverse, weights=values) / counts
        result.update(
            mean=float(means.mean()), status="adequate" if n >= 2 else "inconclusive"
        )
        if n >= 2:
            key = tuple(int(x) for x in unique)
            if key not in self.indices:
                self.indices[key] = np.random.default_rng(0).integers(
                    0, n, size=(self.reps, n)
                )
            draws = means[self.indices[key]].mean(axis=1)
            result["ci95"] = np.quantile(draws, (0.025, 0.975)).tolist()
        return result


def _reward_arrays(pred, truth):
    prediction, target = _array(pred, "pred", ndim=2), _array(truth, "truth", ndim=2)
    if prediction.shape != target.shape or target.shape[1] < max(HORIZONS):
        raise ValueError("pred and truth must have matching [N,H] shapes with H>=15")
    if not np.isin(target, (0.0, 1.0, 2.0)).all():
        raise ValueError("truth must contain raw aggregate reward classes 0,1,2")
    return prediction, target


def _group(pred, truth, episodes, scale, bootstrap, baseline):
    error = (pred - truth) / scale
    cumulative_error = error.cumsum(axis=1)
    base_error = None if baseline is None else (baseline - truth) / scale
    base_cumulative = None if base_error is None else base_error.cumsum(axis=1)
    result = {
        "n_rows": len(pred),
        "n_episodes": len(np.unique(episodes)),
        "horizons": {},
    }
    for h in HORIZONS:
        values = {
            "reward_mse": error[:, h - 1] ** 2,
            "reward_bias": error[:, h - 1],
            "cumulative_reward_mse": cumulative_error[:, h - 1] ** 2,
            "cumulative_reward_bias": cumulative_error[:, h - 1],
        }
        metrics = {
            key: bootstrap.stat(value, episodes) for key, value in values.items()
        }
        metrics["delta_vs_baseline"] = (
            None
            if baseline is None
            else {
                "reward_mse": bootstrap.stat(
                    values["reward_mse"] - base_error[:, h - 1] ** 2, episodes
                ),
                "cumulative_reward_mse": bootstrap.stat(
                    values["cumulative_reward_mse"] - base_cumulative[:, h - 1] ** 2,
                    episodes,
                ),
            }
        )
        classes = {}
        expanded_episodes = np.broadcast_to(episodes[:, None], truth[:, :h].shape)
        for cls in (0, 1, 2):
            mask = truth[:, :h] == cls
            class_error, class_episodes = error[:, :h][mask], expanded_episodes[mask]
            item = {
                "n_steps": int(mask.sum()),
                "n_rows": int(mask.any(axis=1).sum()),
                "n_episodes": len(np.unique(class_episodes)),
                "reward_mse": bootstrap.stat(class_error**2, class_episodes),
                "reward_bias": bootstrap.stat(class_error, class_episodes),
                "prediction_mean_raw": bootstrap.stat(
                    pred[:, :h][mask], class_episodes
                ),
            }
            item["delta_vs_baseline"] = (
                None
                if baseline is None
                else {
                    "reward_mse": bootstrap.stat(
                        class_error**2 - base_error[:, :h][mask] ** 2, class_episodes
                    )
                }
            )
            classes[str(cls)] = item
        metrics["reward_classes"] = classes
        result["horizons"][str(h)] = metrics
    return result


def summarize(
    pred, truth, episode, mode, plan, return_scale, baseline=None, bootstrap_reps=2000
):
    """Score rewards [N,H], H>=15, overall and by observed mode/all four plans.

    ``reward_mse``/``reward_bias`` are endpoint-step errors at h; cumulative
    metrics sum steps 1..h. Class metrics condition on raw truth over steps 1..h.
    All errors are normalized by return_scale (squared for MSE); prediction means
    explicitly marked raw are not normalized. Missing groups/classes use null
    estimates with zero counts, and a single episode has no confidence interval.
    No positive reward evidence yields an inconclusive coverage status even if
    zero-reward prediction is perfect. Bootstrapping uses fixed random seed 0.
    """
    prediction, target = _reward_arrays(pred, truth)
    n = len(prediction)
    episodes = _labels(episode, "episode", n)
    modes = _labels(mode, "mode", n, text=True)
    plans = _labels(plan, "plan", n, allowed=(0, 1, 2, 3))
    scale = _positive(return_scale, "return_scale")
    bootstrap = _EpisodeBootstrap(bootstrap_reps)
    reference = None if baseline is None else _array(baseline, "baseline", ndim=2)
    if reference is not None and reference.shape != target.shape:
        raise ValueError("baseline shape must exactly match pred and truth")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            result = _group(prediction, target, episodes, scale, bootstrap, reference)
            for key, labels, options in (
                ("by_mode", modes, np.unique(modes)),
                ("by_plan", plans, range(4)),
            ):
                result[key] = {}
                for label in options:
                    mask = labels == label
                    result[key][str(label)] = _group(
                        prediction[mask],
                        target[mask],
                        episodes[mask],
                        scale,
                        bootstrap,
                        None if reference is None else reference[mask],
                    )
        except FloatingPointError as exc:
            raise ValueError("summary arithmetic overflowed") from exc
    positive = (target[:, : max(HORIZONS)] > 0).any(axis=1)
    n_positive = len(np.unique(episodes[positive]))
    result.update(
        schema_version=1,
        normalization={
            "return_scale": scale,
            "aggregation": "equal episode means; units averaged within episode",
            "step_metric": "endpoint h",
            "class_metric": "matching steps in prefix 1..h",
            "delta": "candidate minus baseline MSE; negative is better",
            "bootstrap_unit": "episode",
            "bootstrap_reps": bootstrap.reps,
            "bootstrap_seed": 0,
            "confidence": 0.95,
            "interval": "pointwise percentile",
        },
        coverage={
            "positive_reward_rows": int(positive.sum()),
            "episodes_with_positive_reward": n_positive,
            "reward_class_counts": {
                str(c): int((target[:, : max(HORIZONS)] == c).sum()) for c in (0, 1, 2)
            },
            "status": "adequate" if n_positive >= 5 else "inconclusive",
            "minimum_positive_episodes": 5,
        },
    )
    return _json(result)


def decomposition(pred, posterior, truth, return_scale):
    """Exact row-mean cumulative identity: total = base + extra + cross.

    base = E[(posterior-truth)^2], extra = E[(pred-posterior)^2], and
    cross = 2 E[(posterior-truth)*(pred-posterior)], after cumulative summation
    and shared-scale normalization. The signed cross term can cancel either
    square. These are algebraic diagnostics, NOT independent causal fractions.
    This API has no episode IDs, so its aggregation is explicitly equal rows.
    """
    prediction, target = _reward_arrays(pred, truth)
    post = _array(posterior, "posterior", ndim=2)
    if post.shape != target.shape:
        raise ValueError("posterior must match pred and truth shape")
    scale = _positive(return_scale, "return_scale")
    result = {
        "n_rows": len(prediction),
        "aggregation": "equal rows; no episode weighting",
        "return_scale": scale,
        "identity": "total = base + extra + cross",
        "interpretation": "algebraic signed terms, not causal error fractions",
        "horizons": {},
    }
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            base = ((post - target) / scale).cumsum(axis=1)
            extra = ((prediction - post) / scale).cumsum(axis=1)
            total = ((prediction - target) / scale).cumsum(axis=1)
            for h in HORIZONS:
                b, e, t = base[:, h - 1], extra[:, h - 1], total[:, h - 1]
                terms = {
                    "base": float(np.mean(b**2)),
                    "extra": float(np.mean(e**2)),
                    "cross": float(np.mean(2 * b * e)),
                    "total": float(np.mean(t**2)),
                }
                terms["identity_residual"] = terms["total"] - sum(
                    terms[k] for k in ("base", "extra", "cross")
                )
                result["horizons"][str(h)] = terms
        except FloatingPointError as exc:
            raise ValueError("decomposition arithmetic overflowed") from exc
    return _json(result)
