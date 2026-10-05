"""Frozen-representation trajectory diagnostic: fit, predict and profile."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import socket
import time

import numpy as np

from .parallel_collection import (
    _write_json,
    _write_npz,
    _atomic_artifact,
    load_raw_history,
)
from .parallel_metrics import fit_scales, summarize, compare_distributions


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def finite(tree):
    import jax

    for x in jax.tree.leaves(tree):
        a = np.asarray(jax.device_get(x))
        if a.dtype.kind in "fc" and not np.isfinite(a).all():
            raise FloatingPointError("nonfinite diagnostic data")


def normalized_data(data):
    train = data["split"] == 0
    if not train.any():
        raise ValueError("no training rows")
    values = np.concatenate(
        [data["start"][train, None], data["targets"][train]], axis=1
    )
    mean = values.mean((0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(values.std((0, 1), dtype=np.float64), 0.01).astype(np.float32)
    obs_scale, return_scale = fit_scales(
        data["observations"][train], data["rewards"][train], split=data["split"][train]
    )
    return dict(
        mean=mean, std=std, obs_scale=obs_scale, return_scale=np.asarray(return_scale)
    )


class Predictor:
    """No stateful inference; only frozen decoder and learned field parameters."""

    def __init__(self, teacher, params, normalizer):
        import jax
        from imf_dreamer_jax.parallel_trajectory import sample

        self.teacher, self.params = teacher, params
        self.normalizer = normalizer
        self._sample = jax.jit(sample, static_argnames=("flow_steps",))

    def predict_trajectory(self, start_belief, action_sequence, noise, flow_steps=1):
        import jax.numpy as jnp

        mean, std = map(jnp.asarray, (self.normalizer["mean"], self.normalizer["std"]))
        if start_belief.shape[-1] != self.teacher.feature_dim:
            raise ValueError("belief feature width mismatch")
        result = self._sample(
            self.params,
            (start_belief - mean) / std,
            action_sequence,
            noise,
            flow_steps=flow_steps,
        )
        return result * std + mean

    def particles(self, starts, actions, key, count=32, steps=1, composed=False):
        import jax
        import jax.numpy as jnp

        b, h = actions.shape[:2]
        start = jnp.repeat(jnp.asarray(starts), count, axis=0)
        actions = jnp.repeat(jnp.asarray(actions), count, axis=0)
        noise = jax.random.normal(key, (b * count, h, self.teacher.feature_dim))
        direct = self.predict_trajectory(start, actions, noise, steps)
        if composed:
            # Explicit diagnostic convention, not a variable-horizon claim:
            # take first5 of direct15, restart at5 using remaining10 + zero pad.
            remaining = jnp.concatenate(
                [actions[:, 5:], jnp.zeros_like(actions[:, :5])], axis=1
            )
            noise2 = jax.random.normal(jax.random.fold_in(key, 81), noise.shape)
            second = self.predict_trajectory(direct[:, 4], remaining, noise2, steps)
            direct = jnp.concatenate([direct[:, :5], second[:, :10]], axis=1)
        return direct.reshape(b, count, h, self.teacher.feature_dim)


def predictions(predictor, data, indices, seed, *, count=32, steps=1, composed=False):
    import jax

    obs, rew = [], []
    for chunk_id, start in enumerate(range(0, len(indices), 8)):
        row = indices[start : start + 8]
        features = predictor.particles(
            data["start"][row],
            data["actions"][row],
            jax.random.PRNGKey(seed + chunk_id),
            count,
            steps,
            composed,
        )
        o, r = predictor.teacher.decode(features)
        finite((o, r))
        obs.append(np.asarray(jax.device_get(o)))
        rew.append(np.asarray(jax.device_get(r)))
    return np.concatenate(obs), np.concatenate(rew)


def fit(directory, data, teacher, protocol, seed, *, preflight=False):
    import jax
    import optax
    from imf_dreamer_jax import parallel_trajectory as field

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    before = teacher.frozen_digest()
    norm = normalized_data(data)
    _write_npz(directory / "normalization.npz", norm)
    params = field.init(
        jax.random.PRNGKey(seed),
        teacher.feature_dim,
        teacher.action_dim,
        **protocol["field"],
    )
    _write_json(
        directory / "architecture.json",
        dict(
            parameters=sum(x.size for x in jax.tree.leaves(params)),
            field=protocol["field"],
            feature_dim=teacher.feature_dim,
            velocity_readout=protocol["velocity_readout"],
        ),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(protocol["gradient_clip"]),
        optax.adamw(protocol["learning_rate"], weight_decay=protocol["weight_decay"]),
    )
    opt = optimizer.init(params)

    @jax.jit
    def update(p, opt, target, initial, acts, key):
        def objective(w):
            details = field.loss(w, target, initial, acts, key, return_details=True)
            return details.loss, (details.raw_loss_u.mean(), details.raw_loss_v.mean())

        (value, raw), grad = jax.value_and_grad(objective, has_aux=True)(p)
        delta, opt = optimizer.update(grad, opt, p)
        return (
            optax.apply_updates(p, delta),
            opt,
            (value, *raw, optax.global_norm(grad), optax.global_norm(delta)),
        )

    train = np.flatnonzero(data["split"] == 0)
    validation = (
        train[: min(8, len(train))] if preflight else np.flatnonzero(data["split"] == 1)
    )
    if not len(validation):
        raise ValueError("no disjoint validation data")
    mean, std = norm["mean"], norm["std"]
    target = (data["targets"][train] - mean) / std
    starts = (data["start"][train] - mean) / std
    acts = data["actions"][train]
    target, starts, acts = map(jax.device_put, (target, starts, acts))
    predictor = Predictor(teacher, params, norm)
    rng = np.random.default_rng(seed + 511)
    maximum, period = (
        (4, 4) if preflight else (protocol["updates"], protocol["validation_period"])
    )
    best, selected, records = float("inf"), None, []
    begun = time.monotonic()
    with (directory / "training.jsonl").open("x") as log:
        for step in range(1, maximum + 1):
            rows = jax.device_put(rng.integers(0, len(train), protocol["batch_size"]))
            params, opt, values = update(
                params,
                opt,
                target[rows],
                starts[rows],
                acts[rows],
                jax.random.PRNGKey(seed * 100000 + step),
            )
            if step == 1 or step % 100 == 0 or step == maximum:
                finite(values)
                vals = list(map(float, jax.device_get(values)))
                log.write(
                    json.dumps(
                        dict(
                            update=step,
                            loss=vals[0],
                            raw_u=vals[1],
                            raw_v=vals[2],
                            gradient_norm=vals[3],
                            update_norm=vals[4],
                            elapsed_seconds=time.monotonic() - begun,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                log.flush()
            if step % period:
                continue
            finite((params, opt))
            predictor.params = params
            o, r = predictions(predictor, data, validation, 7139, count=32)
            omse = float(
                np.mean(
                    ((o.mean(1) - data["observations"][validation]) / norm["obs_scale"])
                    ** 2
                )
            )
            rmse = float(
                np.mean(
                    (
                        (r.mean(1).sum(1) - data["rewards"][validation].sum(1))
                        / norm["return_scale"]
                    )
                    ** 2
                )
            )
            score = omse + rmse
            checkpoint = f"checkpoint_{step:05d}.pkl"
            _atomic_artifact(
                directory / checkpoint,
                lambda stream: pickle.dump(
                    jax.device_get(params), stream, protocol=pickle.HIGHEST_PROTOCOL
                ),
            )
            _write_npz(
                directory / f"validation_{step:05d}.npz",
                dict(indices=validation, observations=o, rewards=r),
            )
            records.append(
                dict(
                    update=step,
                    score=score,
                    observation_mse=omse,
                    cumulative_reward_mse=rmse,
                    checkpoint=checkpoint,
                )
            )
            if score < best:
                best, selected = score, checkpoint
            print("PARALLEL_FIT_PROGRESS", seed, step, score, flush=True)
    if teacher.frozen_digest() != before:
        raise RuntimeError("teacher mutated during head fitting")
    result = dict(
        seed=seed,
        updates=maximum,
        selected=selected,
        score=best,
        validations=records,
        seconds=time.monotonic() - begun,
        frozen_digest=before,
        preflight=preflight,
    )
    _write_json(directory / "selection.json", result)
    return result


def load_predictor(directory, teacher):
    directory = Path(directory)
    selection = json.loads((directory / "selection.json").read_text())
    if selection["frozen_digest"] != teacher.frozen_digest():
        raise ValueError("selected head belongs to a different frozen teacher")
    name = selection["selected"]
    if Path(name).name != name:
        raise ValueError("invalid checkpoint selection")
    with (directory / name).open("rb") as stream:
        params = pickle.load(stream)
    import jax

    return Predictor(
        teacher,
        jax.tree.map(jax.device_put, params),
        load_npz(directory / "normalization.npz"),
    )


def summary(data, indices, obs, reward, norm):
    return summarize(
        obs,
        reward,
        data["observations"][indices],
        data["rewards"][indices],
        data["episode"][indices],
        data["anchor"][indices],
        data["plan"][indices],
        norm["obs_scale"],
        float(norm["return_scale"]),
    )


def prefix_rows(data, indices):
    lookup = {
        (int(data["episode"][r]), int(data["anchor"][r]), int(data["plan"][r])): i
        for i, r in enumerate(indices)
    }
    left, right = [], []
    for episode, anchor, plan in sorted(lookup):
        if plan != 2:
            continue
        l, r = lookup[(episode, anchor, 2)], lookup[(episode, anchor, 3)]
        if not np.array_equal(
            data["actions"][indices[l], :5], data["actions"][indices[r], :5]
        ):
            raise ValueError("prefix plans differ before decision five")
        if not np.array_equal(
            data["observations"][indices[l], :5], data["observations"][indices[r], :5]
        ):
            raise ValueError("same real prefix is not identical after restore")
        left.append(l)
        right.append(r)
    return np.asarray(left), np.asarray(right)


def evaluate_head(directory, data, teacher, train_dir, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    before = teacher.frozen_digest()
    indices = np.flatnonzero(data["split"] == 2)
    p = load_predictor(train_dir, teacher)
    report = dict(seed=seed, variants={})
    for steps in (1, 2, 4):
        o, r = predictions(p, data, indices, 19900 + seed * 100, steps=steps)
        _write_npz(
            directory / f"nfe{steps}.npz",
            dict(indices=indices, observations=o, rewards=r),
        )
        report["variants"][str(steps)] = summary(data, indices, o, r, p.normalizer)
        if steps == 1:
            # Independent samples for energy estimators, not common-random-number
            # coupling interpreted as independent observations.
            co, cr = predictions(
                p, data, indices, 81900 + seed * 100, steps=steps, composed=True
            )
            _write_npz(
                directory / "composed.npz",
                dict(indices=indices, observations=co, rewards=cr),
            )
            l, rr = prefix_rows(data, indices)
            report["distribution_checks"] = [
                compare_distributions(
                    o,
                    r,
                    co,
                    cr,
                    data["episode"][indices],
                    p.normalizer["obs_scale"],
                    float(p.normalizer["return_scale"]),
                ),
                compare_distributions(
                    o[l],
                    r[l],
                    o[rr],
                    r[rr],
                    data["episode"][indices[l]],
                    p.normalizer["obs_scale"],
                    float(p.normalizer["return_scale"]),
                    prefix_horizon=5,
                ),
            ]
    if teacher.frozen_digest() != before:
        raise RuntimeError("teacher mutated during evaluation")
    _write_json(directory / "report.json", report)
    return report


def categorical_starts(dataset_dir, data, teacher):
    """Reconstruct from real histories, caching each episode prefix once."""
    import copy

    starts = np.empty_like(data["start"])
    targets = np.empty_like(data["targets"])
    indices = np.flatnonzero(data["split"] == 2)
    for eid in np.unique(data["episode"][indices]):
        base, raw = load_raw_history(Path(dataset_dir) / f"episode-{eid:03d}.npz")
        carry = teacher.initial()
        carry, feature = teacher.observe(
            carry, raw[0], base["previous_action"], int(base["observe_seeds"][0])
        )
        for t in range(1, 401):
            carry, feature = teacher.observe(
                carry, raw[t], base["actions"][t - 1], int(base["observe_seeds"][t])
            )
            if t not in (100, 200, 300, 400):
                continue
            for row in np.flatnonzero((data["episode"] == eid) & (data["anchor"] == t)):
                starts[row] = feature
                plan = int(data["plan"][row])
                branch, obs = load_raw_history(
                    Path(dataset_dir) / f"branch-{eid:03d}-{t:03d}-{plan}.npz"
                )
                state = copy.deepcopy(carry)
                for j in range(15):
                    state, f = teacher.observe(
                        state,
                        obs[j + 1],
                        branch["actions"][j],
                        int(branch["observe_seeds"][j + 1]),
                    )
                    targets[row, j] = f
    return starts[indices], targets[indices]


def evaluate_baseline(
    directory, data, teacher, norm, *, categorical=False, dataset_dir=None
):
    import jax

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    indices = np.flatnonzero(data["split"] == 2)
    start, target = (
        categorical_starts(dataset_dir, data, teacher)
        if categorical
        else (data["start"][indices], data["targets"][indices])
    )
    _write_npz(
        directory / "beliefs.npz", dict(indices=indices, start=start, targets=target)
    )
    reports = {}
    for steps in ((1,) if categorical else (1, 4)):
        oo, rr = [], []
        for i in range(0, len(indices), 8):
            f = teacher.rollout(
                start[i : i + 8],
                data["actions"][indices[i : i + 8]],
                67100 + i,
                32,
                steps,
            )
            o, r = teacher.decode(f)
            finite((o, r))
            oo.append(np.asarray(jax.device_get(o)))
            rr.append(np.asarray(jax.device_get(r)))
        o, r = np.concatenate(oo), np.concatenate(rr)
        _write_npz(
            directory / f"nfe{steps}.npz",
            dict(indices=indices, observations=o, rewards=r),
        )
        reports[str(steps)] = summary(data, indices, o, r, norm)
    o, r = teacher.decode(target)
    o, r = (
        np.asarray(jax.device_get(o))[:, None],
        np.asarray(jax.device_get(r))[:, None],
    )
    _write_npz(
        directory / "posterior_heads.npz",
        dict(indices=indices, observations=o, rewards=r),
    )
    reports["posterior_heads"] = summary(data, indices, o, r, norm)
    if not categorical:
        o = np.repeat(data["initial_observation"][indices, None, None], 15, axis=2)
        r = np.zeros((len(indices), 1, 15), np.float32)
        reports["persistence_zero"] = summary(data, indices, o, r, norm)
    _write_json(directory / "report.json", reports)
    return reports


def profile(teacher, data, mode, seed, steps, train_dir=None):
    """Fresh process per model/NFE; optimizer excluded from device inference."""
    import jax

    p = load_predictor(train_dir, teacher) if mode == "head" else None
    indices = np.flatnonzero(data["split"] == 2)
    reports = {}
    for batch in (1, 64):
        rows = indices[np.arange(batch) % len(indices)]
        starts, acts = map(jax.device_put, (data["start"][rows], data["actions"][rows]))

        def latent():
            return (
                p.particles(starts, acts, jax.random.PRNGKey(831), 32, steps)
                if p
                else teacher.rollout(starts, acts, 831, 32, steps)
            )

        def decoded():
            return teacher.decode(latent())

        value = {}
        for label, fn in (("latent", latent), ("decoded", decoded)):
            t = time.perf_counter()
            jax.block_until_ready(fn())
            cold = time.perf_counter() - t
            for _ in range(3):
                jax.block_until_ready(fn())
            durations = []
            for _ in range(20):
                t = time.perf_counter()
                jax.block_until_ready(fn())
                durations.append(time.perf_counter() - t)
            value[label] = dict(
                median_seconds=float(np.median(durations)),
                p95_seconds=float(np.percentile(durations, 95)),
                cold_call_including_compilation_seconds=cold,
                samples=durations,
            )
        memory = jax.devices()[0].memory_stats() or {}
        value["allocator_memory"] = {
            k: int(v) for k, v in memory.items() if isinstance(v, (int, np.integer))
        }
        reports[f"batch{batch}"] = value
    own = sum(x.size for x in jax.tree.leaves(p.params)) if p else 0
    active = (
        sum(x.size for k, v in p.params.items() if k != "v" for x in jax.tree.leaves(v))
        if p
        else teacher.active_dynamics_parameters
    )
    return dict(
        mode=mode,
        seed=seed,
        flow_steps=steps,
        particles=32,
        horizon=15,
        batches=reports,
        device=str(jax.devices()[0].device_kind),
        platform_version=str(jax.devices()[0].client.platform_version),
        device_identity=dict(
            node=socket.gethostname(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            ids=[int(d.id) for d in jax.devices()],
        ),
        head_parameters=own,
        frozen_agent_parameters=teacher.parameter_count,
        resident_model_parameters=own + teacher.parameter_count,
        resident_parent_state_scalars=teacher.resident_state_scalars,
        resident_array_scalars=own + teacher.resident_array_scalars,
        active_latent_parameters=active,
        active_decoded_parameters=active
        + teacher.parameter_counts["dec"]
        + teacher.parameter_counts["rew"],
        memory_definition="process-local JAX allocator peak includes compile/warmup and earlier batches; not isolated steady-state tensor memory",
        precision=dict(student_flow="float32", upstream_actual=teacher.compute_dtype),
        optimizer_excluded_from_device_inference=True,
        parameter_definition="resident includes encoder,dynamics,decoder,reward,continuation,policy,value,slow-value target and student; other retained non-optimizer arrays reported as state scalars; active counts functionally influential coefficients, not FLOPs",
        compilation_cost_definition="cold synchronized first call includes tracing, compilation and first execution; reported separately from warmed median/p95",
    )
