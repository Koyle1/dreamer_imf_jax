"""Episode-clustered metrics for summed AR2 rewards; no endpoint-label fiction."""

import numpy as np
from .reward_probe_metrics import _EpisodeBootstrap


def boundary_bins(distance, radius):
    d = np.asarray(distance, float)
    if (
        not np.isfinite(d).all()
        or (d < 0).any()
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("invalid geometry")
    r = d / radius
    return np.where(r < 0.8, 0, np.where(r <= 1.0, 1, np.where(r <= 1.2, 2, 3)))


def validate_probabilities(p, y):
    p, y = np.asarray(p, float), np.asarray(y)
    if p.shape != y.shape + (3,) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("invalid categorical predictions")
    if (
        not np.allclose(p.sum(-1), 1, atol=2e-6, rtol=0)
        or not np.isin(y, [0, 1, 2]).all()
    ):
        raise ValueError("invalid probabilities or labels")
    return p, y.astype(int)


def nll(p, y):
    p, y = validate_probabilities(p, y)
    return -np.log(np.maximum(np.take_along_axis(p, y[..., None], -1)[..., 0], 1e-8))


def calibration(p, y):
    """Fixed ten bins for P(R>0), plus classwise calibration; no fitted bins."""
    p, y = validate_probabilities(p, y)
    result = {}
    for name, probability, truth in [
        ("positive", 1 - p[..., 0], y > 0),
        *[(str(k), p[..., k], y == k) for k in range(3)],
    ]:
        bins = np.minimum((probability * 10).astype(int), 9)
        result[name] = [
            dict(
                count=int((bins == b).sum()),
                prediction=(
                    float(probability[bins == b].mean()) if (bins == b).any() else None
                ),
                frequency=float(truth[bins == b].mean()) if (bins == b).any() else None,
            )
            for b in range(10)
        ]
    return result


def summary(
    prediction,
    truth,
    episode,
    *,
    probabilities=None,
    boundary=None,
    baseline=None,
    reps=2000,
):
    pred, y = np.asarray(prediction, float), np.asarray(truth, float)
    if (
        pred.shape != y.shape
        or y.ndim != 2
        or y.shape[1] != 15
        or not np.isfinite(pred).all()
    ):
        raise ValueError("expected finite reward matrices [N,15]")
    if not np.isin(y, [0, 1, 2]).all() or len(episode) != len(y):
        raise ValueError("invalid labels/episodes")
    boot = _EpisodeBootstrap(reps)
    eps = np.broadcast_to(np.asarray(episode)[:, None], y.shape)
    errors = (pred - y) ** 2
    values = dict(mse=errors, bias=pred - y)
    if probabilities is not None:
        p, yi = validate_probabilities(probabilities, y)
        values.update(nll=nll(p, yi), brier=((p - np.eye(3)[yi]) ** 2).sum(-1))
    else:
        p = None

    def group(mask):
        return dict(
            count=int(mask.sum()),
            episodes=int(len(np.unique(eps[mask]))),
            **{k: boot.stat(v[mask], eps[mask]) for k, v in values.items()},
        )

    all_rows = np.ones_like(y, bool)
    result = dict(
        overall=group(all_rows),
        classes={str(k): group(y == k) for k in range(3)},
        probability_metrics=(
            "available"
            if p is not None
            else "unavailable; original head exposes expectation only"
        ),
        positive_episodes=int(len(np.unique(eps[y > 0]))),
        horizons={},
    )
    result["reward_conclusions"] = (
        "adequate" if result["positive_episodes"] >= 5 else "inconclusive"
    )
    for h in (1, 3, 5, 10, 15):
        result["horizons"][str(h)] = dict(
            reward_mse=boot.stat(errors[:, h - 1], episode),
            cumulative_reward_mse=boot.stat(
                (pred[:, :h].sum(1) - y[:, :h].sum(1)) ** 2, episode
            ),
        )
    if p is not None:
        zero = y == 0
        result["false_positive_rate"] = boot.stat(
            ((1 - p[..., 0]) >= 0.5)[zero], eps[zero]
        )
        result["calibration"] = calibration(p, y)
    if boundary is not None:
        if boundary.shape != y.shape or not np.isin(boundary, [0, 1, 2, 3]).all():
            raise ValueError("invalid boundary bins")
        result["endpoint_distance_strata"] = {
            str(k): group(boundary == k) for k in range(4)
        }
    if baseline is not None:
        baseline = np.asarray(baseline, float)
        if baseline.shape != y.shape or not np.isfinite(baseline).all():
            raise ValueError("invalid baseline")
        result["paired_mse_delta"] = boot.stat(
            (errors - (baseline - y) ** 2).reshape(-1), eps.reshape(-1)
        )
        base_mse = boot.stat(((baseline - y) ** 2).reshape(-1), eps.reshape(-1))["mean"]
        reduction = (
            None if base_mse == 0 else 1 - result["overall"]["mse"]["mean"] / base_mse
        )
        result["relative_mse_reduction"] = reduction
        hi = result["paired_mse_delta"]["ci95"][1]
        result["material_exploratory_improvement"] = bool(
            reduction is not None
            and reduction >= 0.1
            and hi is not None
            and hi < 0
            and result["positive_episodes"] >= 5
        )
    return result


def action_metrics(prediction, truth, episode, anchor, plan, reps=2000):
    """Raw paired cumulative effects and strict ranking; one vote per episode."""
    from itertools import combinations

    boot = _EpisodeBootstrap(reps)
    effects, ranking = {}, []
    for a, b in combinations(range(4), 2):
        errors, ep, predicted, actual = [], [], [], []
        for e, t in sorted(set(zip(episode, anchor))):
            rows = {
                int(plan[i]): i for i in np.flatnonzero((episode == e) & (anchor == t))
            }
            if set(rows) != {0, 1, 2, 3}:
                raise ValueError("incomplete matched action plans")
            dp = prediction[rows[b]].sum() - prediction[rows[a]].sum()
            dy = truth[rows[b]].sum() - truth[rows[a]].sum()
            errors.append((dp - dy) ** 2)
            ep.append(e)
            predicted.append(dp)
            actual.append(dy)
            if dy != 0:
                ranking.append((e, float(dp * dy > 0)))
        effects[f"{a}_to_{b}"] = dict(
            mse=boot.stat(errors, np.asarray(ep)),
            predicted=boot.stat(predicted, np.asarray(ep)),
            actual=boot.stat(actual, np.asarray(ep)),
        )
    return dict(
        paired_effects=effects,
        ranking_accuracy=boot.stat(
            np.asarray([x[1] for x in ranking]), np.asarray([x[0] for x in ranking])
        ),
    )
