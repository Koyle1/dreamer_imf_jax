"""Train-only spectral and gradient diagnostics for frozen reward readouts.

No routine fits transforms to held-out data. Feature diagnostics never construct
the full N-by-N Gram matrix; gradient diagnostics never retain an N-by-P Jacobian.
Scores measure geometry, not calibrated uncertainty or proof of OOD membership.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _matrix(x, name="features"):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(f"invalid {name}")
    return x


def _vectors(n, *values):
    result = [np.asarray(v) for v in values]
    if any(v.ndim != 1 or len(v) != n for v in result):
        raise ValueError("row metadata differs")
    if any(v.dtype.kind in "fci" and not np.isfinite(v).all() for v in result):
        raise ValueError("nonfinite row metadata")
    return result


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _rank(eigenvalues):
    values = np.maximum(np.asarray(eigenvalues, np.float64), 0)
    trace = float(values.sum())
    tol = float(values.max(initial=0) * max(1, len(values)) * np.finfo(float).eps)
    p = values[values > 0] / trace if trace else np.zeros(0)
    return dict(
        trace=trace,
        numerical_rank=int(np.sum(values > tol)),
        numerical_rank_tolerance=tol,
        entropy_effective_rank=float(np.exp(-np.sum(p * np.log(p)))) if trace else 0.0,
        participation_ratio=_ratio(trace**2, float(values @ values)) or 0.0,
        stable_rank=_ratio(trace, float(values.max(initial=0))) or 0.0,
        degenerate=not bool(trace),
    )


def _blocks(episode, mode):
    """Preserve episode order and within-episode row order under permutation."""
    ids = np.unique(episode)
    blocks = [np.flatnonzero(episode == e) for e in ids]
    strata = {}
    for k, ix in enumerate(blocks):
        kinds = np.unique(mode[ix])
        if len(kinds) != 1:
            raise ValueError("collection mode varies inside episode")
        strata.setdefault((str(kinds[0]), len(ix)), []).append(k)
    return ids, blocks, list(strata.values())


def _label_stats(x, y, blocks, covariance_frobenius_sq):
    """Centred linear CKA, also excluding all same-episode pairs."""
    yy = y - np.mean(y)
    cross = x.T @ yy
    numerator = float(cross @ cross)
    label_sq = float(yy @ yy) ** 2
    within_num = within_x = within_y = 0.0
    for ix in blocks:
        moment = x[ix].T @ yy[ix]
        within_num += float(moment @ moment)
        for a in range(0, len(ix), 256):
            for b in range(0, len(ix), 256):
                gram = x[ix[a : a + 256]] @ x[ix[b : b + 256]].T
                within_x += float(np.sum(gram**2))
        within_y += float(yy[ix] @ yy[ix]) ** 2
    outside_x = max(0.0, covariance_frobenius_sq - within_x)
    outside_y = max(0.0, label_sq - within_y)
    return dict(
        alignment=_ratio(numerator, np.sqrt(covariance_frobenius_sq * label_sq)),
        cross_episode_alignment=(
            _ratio(numerator - within_num, np.sqrt(outside_x * outside_y))
            if len(blocks) > 1
            else None
        ),
        interpretation="global training centering; cross-episode excludes diagonal episode blocks",
    )


def analyze_features(x, y, episode, mode, *, seed, resamples, permutations):
    """Exact covariance spectrum and train-only episode-level stability controls.

    Bootstrap uncertainty uses the *fixed full-training* PCA basis: it assesses
    stability of variances and reward correlations, not a refitted eigenbasis.
    Label permutations exchange intact equal-length episodes within mode. Their
    p value is descriptive: exchangeability across episodes is an assumption.
    """
    x = _matrix(x)
    y, episode, mode = _vectors(len(x), y, episode, mode)
    y = np.asarray(y, np.float64)
    if (
        resamples < 0
        or permutations < 0
        or int(resamples) != resamples
        or int(permutations) != permutations
    ):
        raise ValueError("invalid resampling counts")
    resamples, permutations = int(resamples), int(permutations)
    ids, blocks, strata = _blocks(episode, mode)
    centered = x - x.mean(0)
    covariance = centered.T @ centered / len(x)
    if not np.isfinite(covariance).all():
        raise ValueError("nonfinite feature covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.arange(len(eigenvalues) - 1, -1, -1)
    eigenvalues, eigenvectors = (
        np.maximum(eigenvalues[order], 0),
        eigenvectors[:, order],
    )
    summary = _rank(eigenvalues)
    yy = y - y.mean()
    variance_y = float(np.mean(yy**2))
    covariance_y = centered.T @ yy / len(x)
    projected_covariance = eigenvectors.T @ covariance_y
    explained = np.zeros_like(eigenvalues)
    valid = eigenvalues > summary["numerical_rank_tolerance"]
    if variance_y > 0:
        explained[valid] = projected_covariance[valid] ** 2 / (
            eigenvalues[valid] * variance_y
        )
    frobenius_sq = float(np.sum(covariance**2) * len(x) ** 2)
    alignment = _label_stats(centered, y, blocks, frobenius_sq)
    arrays = dict(
        eigenvalues=eigenvalues,
        component_reward_cross_covariance=projected_covariance,
        component_linear_reward_r2=explained,
        cumulative_linear_reward_r2=np.cumsum(explained),
        episode_ids=ids,
    )

    # Include low-variance directions so a large-eigenvalue heuristic is not
    # silently treated as a reward-relevance test. Maximum projection width 32.
    positive = np.flatnonzero(valid)
    selected = np.unique(np.concatenate([positive[:16], positive[-16:]]))
    projected = centered @ eigenvectors[:, selected]
    bootstrap_var, bootstrap_corr, bootstrap_cos, draws = [], [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        chosen = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        ix = np.concatenate([blocks[k] for k in chosen])
        zz, target = projected[ix], y[ix]
        zz = zz - zz.mean(0)
        target = target - target.mean()
        vv = np.mean(zz**2, axis=0)
        cov = np.mean(zz * target[:, None], axis=0)
        denom = np.sqrt(vv * np.mean(target**2))
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
        moment = centered[ix].T @ target / len(ix)
        den = np.linalg.norm(moment) * np.linalg.norm(covariance_y)
        bootstrap_cos.append(float(moment @ covariance_y / den) if den else 0.0)
        bootstrap_var.append(vv)
        bootstrap_corr.append(corr)
        draws.append(chosen)
    arrays.update(
        stability_component_indices=selected,
        bootstrap_episode_indices=np.asarray(draws, dtype=np.int64).reshape(
            resamples, len(ids)
        ),
        bootstrap_projected_variance=np.asarray(bootstrap_var).reshape(
            resamples, len(selected)
        ),
        bootstrap_projected_reward_correlation=np.asarray(bootstrap_corr).reshape(
            resamples, len(selected)
        ),
        bootstrap_reward_moment_cosine=np.asarray(bootstrap_cos),
    )

    # Bounded full-data permutation computation: all N-by-P label draws fit in
    # a few MB at the registered P=100, unlike an N-by-N feature Gram matrix.
    permuted = np.empty((len(x), permutations), np.float64)
    mappings = np.empty((permutations, len(ids)), np.int64)
    for p in range(permutations):
        mapping = np.arange(len(ids))
        for stratum in strata:
            mapping[stratum] = rng.permutation(stratum)
        for destination, source in enumerate(mapping):
            permuted[blocks[destination], p] = y[blocks[source]]
        mappings[p] = mapping
    permuted -= y.mean()
    cross = centered.T @ permuted
    numerator = np.sum(cross**2, axis=0)
    denom = np.sqrt(frobenius_sq) * float(yy @ yy)
    permutation_alignment = numerator / denom if denom else np.zeros(permutations)
    arrays.update(
        permutation_episode_indices=mappings,
        permutation_alignment=permutation_alignment,
    )
    exchangeable = any(len(s) > 1 for s in strata)
    tie_tolerance = 64 * np.finfo(float).eps
    pvalue = (
        float(
            (
                1
                + np.sum(
                    permutation_alignment >= alignment["alignment"] - tie_tolerance
                )
            )
            / (1 + permutations)
        )
        if permutations and exchangeable and alignment["alignment"] is not None
        else None
    )
    report = dict(
        rows=len(x),
        features=x.shape[1],
        episodes=len(ids),
        spectrum=summary,
        reward_variance=variance_y,
        linear_reward_r2=float(explained.sum()),
        **alignment,
        episode_block_permutation_p=pvalue,
        permutation_alignment_tie_tolerance=tie_tolerance,
        permutation_exchangeable_blocks=exchangeable,
        bootstrap_basis="fixed full-training PCA; at most 16 highest and 16 lowest nonzero components",
        bootstrap_zero_variance_correlation_convention=0.0,
        resamples=int(resamples),
        permutations=int(permutations),
        seed=int(seed),
        statistical_unit="episode; rows and branches are not independent seeds",
    )
    return report, arrays


def _validate_params(params, x):
    if not params:
        raise ValueError("empty readout")
    width = x.shape[1]
    for layer in params:
        w, b = np.asarray(layer["w"]), np.asarray(layer["b"])
        if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
            raise ValueError("readout architecture differs")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite readout")
        width = w.shape[1]
    if width != 3:
        raise ValueError("diagnostics require three reward logits")


@partial(jax.jit, static_argnames=("kind",))
def _factors(params, x, kind):
    """Exact dense-ReLU Jacobian factors, in the deployed parameter dtype."""
    acts, masks = [x], []
    for layer in params[:-1]:
        pre = acts[-1] @ layer["w"] + layer["b"]
        masks.append(pre > 0)
        acts.append(jax.nn.relu(pre))
    logits = acts[-1] @ params[-1]["w"] + params[-1]["b"]
    if kind == "trace":
        delta = jnp.broadcast_to(jnp.eye(3, dtype=logits.dtype), (len(x), 3, 3))
    else:
        # jnp.max shares the derivative equally across ties (not argmax).
        tied = logits == logits.max(-1, keepdims=True)
        delta = (tied / tied.sum(-1, keepdims=True))[:, None, :]
    deltas = [delta]
    for k in range(len(params) - 2, -1, -1):
        delta = (delta @ params[k + 1]["w"].T) * masks[k][:, None, :]
        deltas.append(delta)
    return tuple(acts), tuple(reversed(deltas))


def _host_factors(params, x, kind):
    acts, deltas = _factors(params, jnp.asarray(x, params[0]["w"].dtype), kind)
    acts, deltas = [np.asarray(a, np.float64) for a in acts], [
        np.asarray(d, np.float64) for d in deltas
    ]
    if any(not np.isfinite(a).all() for a in acts + deltas):
        raise ValueError("nonfinite readout Jacobian factors")
    return acts, deltas


def _gradient_batches(params, x, size=16):
    for start in range(0, len(x), size):
        acts, deltas = _host_factors(params, x[start : start + size], "max")
        pieces = []
        for a, d in zip(acts, deltas):
            delta = d[:, 0]
            pieces.extend(
                [np.einsum("ni,nj->nij", a, delta).reshape(len(a), -1), delta]
            )
        yield start, np.concatenate(pieces, axis=1)


def kernel_diagnostics(params, x, y, episode):
    """Traced 3-logit empirical NTK on a caller-chosen <=256-row subset."""
    x = _matrix(x)
    y, episode = _vectors(len(x), y, episode)
    _validate_params(params, x)
    if len(x) > 256:
        raise ValueError("kernel subset exceeds frozen 256-row cap")
    acts, deltas = _host_factors(params, x, "trace")
    kernel = np.zeros((len(x), len(x)), np.float64)
    for a, d in zip(acts, deltas):
        kernel += (a @ a.T + 1) * np.einsum("ncf,mcf->nm", d, d)
    centered = kernel - kernel.mean(0)[None] - kernel.mean(1)[:, None] + kernel.mean()
    eig = np.maximum(np.linalg.eigvalsh((centered + centered.T) / 2)[::-1], 0)
    target = y.astype(np.float64) - np.mean(y)
    target_kernel = np.outer(target, target)
    outside = episode[:, None] != episode[None]

    def align(mask):
        a, b = centered[mask], target_kernel[mask]
        return _ratio(float(a @ b), float(np.linalg.norm(a) * np.linalg.norm(b)))

    return dict(
        rows=len(x),
        episodes=len(np.unique(episode)),
        classes=3,
        kernel="sum of same-output parameter-Jacobian inner products, all weights and biases",
        spectrum=_rank(eig),
        alignment=align(np.ones_like(outside)),
        cross_episode_alignment=align(outside),
        precision="forward/backward factors use parameter dtype; Gram accumulation float64",
    ), dict(
        kernel=kernel,
        centered_kernel=centered,
        eigenvalues=eig,
        episode=episode,
        labels=y,
    )


def fit_gradpca(params, train_x, train_y):
    """Max-logit variant of GradPCA Algorithm 1, using all readout parameters.

    Returns a serializable numeric state. Larger scores mean greater alignment
    with the class-mean subspace, NOT larger uncertainty. Unlike the paper's
    default sum-of-logits aggregation, this registered experiment uses max.
    """
    x = _matrix(train_x)
    (y,) = _vectors(len(x), train_y)
    _validate_params(params, x)
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("invalid reward classes")
    classes, counts = np.unique(y.astype(np.int64), return_counts=True)
    dimension = sum(np.asarray(l[k]).size for l in params for k in ("w", "b"))
    means = np.zeros((len(classes), dimension), np.float64)
    for start, gradients in _gradient_batches(params, x):
        labels = y[start : start + len(gradients)]
        for k, c in enumerate(classes):
            means[k] += gradients[labels == c].sum(0)
    means /= counts[:, None]
    # Global row-weighted mean as in Algorithm 1. The C centered class means
    # still have rank <= C-1, even with unequal class counts.
    mean = np.average(means, axis=0, weights=counts)
    centered = means - mean
    gram = centered @ centered.T
    eig, vectors = np.linalg.eigh(gram)
    order = np.arange(len(eig) - 1, -1, -1)
    eig, vectors = np.maximum(eig[order], 0), vectors[:, order]
    tol = float(eig.max(initial=0) * max(dimension, len(classes)) * np.finfo(float).eps)
    rank = min(int(np.sum(eig > tol)), len(classes) - 1)
    basis = vectors[:, :rank].T @ centered / np.sqrt(eig[:rank])[:, None]
    return dict(
        mean=mean,
        basis=basis,
        classes=classes,
        counts=counts,
        eigenvalues=eig,
        numerical_rank=np.asarray(rank),
        rank_cap=np.asarray(len(classes) - 1),
        tolerance=np.asarray(tol),
        parameter_dimension=np.asarray(dimension),
        degenerate=np.asarray(rank == 0),
    )


def score_gradpca(params, x, state):
    """In-subspace ratio in [0,1]; exact zero centered gradients score zero.

    The zero convention is finite but has no OOD interpretation; rank-zero fits
    are explicitly labelled degenerate in the fitted state.
    """
    x = _matrix(x)
    _validate_params(params, x)
    mean, basis = np.asarray(state["mean"]), np.asarray(state["basis"])
    dimension = sum(np.asarray(l[k]).size for l in params for k in ("w", "b"))
    if mean.shape != (dimension,) or basis.ndim != 2 or basis.shape[1] != dimension:
        raise ValueError("GradPCA state differs from readout")
    if not np.isfinite(mean).all() or not np.isfinite(basis).all():
        raise ValueError("nonfinite GradPCA state")
    result = np.empty(len(x), np.float64)
    for start, gradients in _gradient_batches(params, x):
        gradients -= mean
        norm = np.linalg.norm(gradients, axis=1)
        projected = np.linalg.norm(gradients @ basis.T, axis=1)
        result[start : start + len(gradients)] = np.divide(
            projected, norm, out=np.zeros_like(norm), where=norm > 0
        )
    return np.clip(result, 0, 1)


def nearest_distance(train_x, x):
    """Nearest training-feature RMS distance; bounded 64-by-512 work buffers."""
    train, x = _matrix(train_x), _matrix(x)
    if train.shape[1] != x.shape[1]:
        raise ValueError("feature dimensions differ")
    result = np.full(len(x), np.inf)
    for start in range(0, len(x), 64):
        query = x[start : start + 64]
        best = np.full(len(query), np.inf)
        nearest = np.zeros(len(query), np.int64)
        for t in range(0, len(train), 512):
            reference = train[t : t + 512]
            distances = np.maximum(
                np.sum(query**2, axis=1)[:, None]
                + np.sum(reference**2, axis=1)[None]
                - 2 * query @ reference.T,
                0,
            )
            local = distances.argmin(1)
            candidate = distances[np.arange(len(query)), local]
            better = candidate < best
            nearest[better] = t + local[better]
            best = np.minimum(best, candidate)
        # Recompute only the selected distances by subtraction. This avoids
        # cancellation making an exact retained training row spuriously distant.
        result[start : start + len(query)] = np.sqrt(
            np.mean((query - train[nearest]) ** 2, axis=1)
        )
    return result
