"""Read-only, common-random-number diagnostics for the continuous Dreamer arms.

These are fixed-buffer model diagnostics, not independent policy evaluations.
Latent targets remain model-dependent; normalized errors are operational gates,
not a calibrated probability of successful control. No optimizer is called.
"""

import hashlib
import json
from functools import partial

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import embodied.jax.nets as nn
from imf_dreamer_jax.imf import imf_field, imf_velocity, sample_imf_steps


def _host(x):
    return np.asarray(jax.device_get(x))


def _digest(batch):
    digest = hashlib.sha256()
    for name, value in sorted(batch.items()):
        value = _host(value)
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _json(tree):
    def convert(x):
        x = _host(x)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite staged diagnostic")
        return x.item() if x.ndim == 0 else x.tolist()

    out = jax.tree.map(convert, tree)
    json.dumps(out, allow_nan=False)
    return out


def _distribution(reference, generated):
    """Conditional ensemble [samples, cases, features] distribution distances."""
    mean_error = jnp.mean((reference.mean(0) - generated.mean(0)) ** 2)
    std_error = jnp.mean((reference.std(0) - generated.std(0)) ** 2)
    # Deterministic shared projections; sorting compares distributions rather
    # than arbitrarily pairing independent posterior and prior draws.
    axes = jax.random.normal(jax.random.PRNGKey(701), (reference.shape[-1], 8))
    axes /= jnp.maximum(jnp.linalg.norm(axes, axis=0, keepdims=True), 1e-8)
    a, b = reference @ axes, generated @ axes
    return dict(
        mean_mse=mean_error,
        std_mse=std_error,
        sliced_wasserstein2=jnp.mean((jnp.sort(a, axis=0) - jnp.sort(b, axis=0)) ** 2),
        posterior_rms=jnp.sqrt(jnp.mean(reference**2)),
        prior_rms=jnp.sqrt(jnp.mean(generated**2)),
    )


def evaluate_batch(model, params, batch, seed, previous=None):
    """Return JSON-safe metrics and an ephemeral drift reference.

    ``model`` is the raw upstream Agent, ``params`` its outer flat parameter
    dictionary, and ``batch`` ordinary [B,T] observations plus stored actions.
    The first observation is treated as a context reset (no unavailable carry).
    Later resets are respected. Repeat calls must use the same batch and seed.
    All returned device arrays are explicitly transferred to host, including
    the reference; no training state, policy RNG, or parameters are changed.
    """
    required = set(model.act_space) | set(model.obs_space)
    if not required.issubset(batch):
        raise ValueError(f"Missing diagnostic fields: {required - set(batch)}")
    shape = batch["is_first"].shape
    if len(shape) != 2 or shape[1] < 2:
        raise ValueError("Diagnostics need [B,T] with T >= 2")
    if any(value.shape[:2] != shape for value in batch.values()):
        raise ValueError("All diagnostic fields must share [B,T]")
    signature = _digest(batch)
    if previous is not None and (
        previous["batch_sha256"] != signature or previous["seed"] != int(seed)
    ):
        raise ValueError("Drift requires the same fixed batch and common noise seed")
    # Explicit device_put is intentional under upstream transfer_guard=disallow.
    data = jax.tree.map(jax.device_put, batch)
    raw, current = _evaluate_arrays(model, params, data, int(seed))
    metrics = _json(raw)
    if not any(row["valid_paths"] > 0 for row in metrics["rollout"]["4"].values()):
        raise ValueError("No uninterrupted diagnostic rollout available")
    longest_valid = max(
        int(h) for h, row in metrics["rollout"]["4"].items() if row["valid_paths"] > 0
    )
    metrics["quality_normalized_rollout_error"] = metrics["rollout"]["4"][
        str(longest_valid)
    ]["normalized_latent_mean_mse"]
    # A zero-path horizon never passes the operational gate.
    if (
        "5" in metrics["rollout"]["1"]
        and metrics["rollout"]["1"]["5"]["valid_paths"] == 0
    ):
        metrics["quality_gate"] = None
    current = jax.tree.map(_host, current)
    metrics["drift"] = None
    if previous is not None:
        metrics["drift"] = {
            k
            + "_mse": float(
                np.mean((v.astype(np.float64) - previous[k].astype(np.float64)) ** 2)
            )
            for k, v in current.items()
        }
        if not all(np.isfinite(v) for v in metrics["drift"].values()):
            raise ValueError("Nonfinite staged drift diagnostic")
    metrics["batch_sha256"] = signature
    metrics["seed"] = int(seed)
    return metrics, dict(current, batch_sha256=signature, seed=int(seed))


@partial(jax.jit, static_argnums=(0, 3))
def _evaluate_arrays(model, params, data, seed):
    """One cached numerical trace; explicit transfers remain in the wrapper."""
    b, t = data["is_first"].shape
    reset = data["is_first"].at[:, 0].set(True)
    acts = {
        key: jnp.concatenate([jnp.zeros_like(data[key][:, :1]), data[key][:, :-1]], 1)
        for key in model.act_space
    }

    def posterior():
        _, _, tokens = model.enc(model.enc.initial(b), data, reset, False)
        _, _, feat = model.dyn.observe(model.dyn.initial(b), tokens, acts, reset, False)
        return tokens, feat

    postfun = nj.pure(posterior)
    # Ninjax 3.6.3 scan requires modify=True for its access-discovery pass;
    # pure() copies the state dictionary and no returned state is installed.
    _, (tokens, feat) = postfun(params, seed=seed)
    if not {"mean", "std"}.issubset(feat):
        raise ValueError("Staged diagnostics require continuous posterior statistics")
    mean = feat["mean"].astype(jnp.float32).reshape(b * t, -1)
    std = feat["std"].astype(jnp.float32).reshape(b * t, -1)
    condition = feat["deter"].astype(jnp.float32).reshape(b * t, -1)
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(jax.random.fold_in(key, 1), (8, *mean.shape))
    reference = mean[None] + std[None] * noise
    is_imf = hasattr(model.dyn, "_flow_params")
    flow = nj.pure(model.dyn._flow_params)(params, seed=seed)[1] if is_imf else None

    def sample(deter, eps, steps):
        if is_imf:
            return sample_imf_steps(flow, deter, noise=eps, steps=steps)
        _, (mu, sigma) = nj.pure(model.dyn._prior_normal)(params, deter, seed=seed)
        return mu.reshape(eps.shape) + sigma.reshape(eps.shape) * eps

    prior = {
        steps: jax.vmap(lambda eps: sample(condition, eps, steps))(noise)
        for steps in (1, 4)
    }
    metrics = {
        "samples_per_condition": 8,
        "distribution": {str(s): _distribution(reference, prior[s]) for s in prior},
        "one_vs_four_sample_mse_common_noise": jnp.mean((prior[1] - prior[4]) ** 2),
    }
    if is_imf:
        target = feat["stoch"].astype(jnp.float32).reshape(mean.shape)
        eps = noise[0]
        bins = {}
        for tv in (0.125, 0.375, 0.625, 0.875):
            tt, rr = jnp.full((b * t, 1), tv), jnp.full((b * t, 1), tv / 2)
            z = (1 - tt) * target + tt * eps
            velocity = imf_velocity(flow, z, condition, tt, tt)
            u, derivative = jax.jvp(
                lambda zz, r, time: imf_field(flow, zz, condition, r, time),
                (z, rr, tt),
                (velocity, jnp.zeros_like(rr), jnp.ones_like(tt)),
            )
            truth = eps - target
            bins[str(tv)] = dict(
                compound_u_mse=jnp.mean((u + (tt - rr) * derivative - truth) ** 2),
                boundary_v_mse=jnp.mean((velocity - truth) ** 2),
                target_velocity_rms=jnp.sqrt(jnp.mean(truth**2)),
            )
        metrics["raw_velocity_time_bins"] = bins

    # All starts supporting the maximum horizon, shared across shorter horizons,
    # with paths crossing a real reset excluded. The
    # recorded action at k causes observation/reward k+1 (upstream convention).
    horizon = min(5, t - 1)
    n = b * (t - horizon)
    width = t - horizon
    starts = {
        k: feat[k][:, :width].reshape((n, *feat[k].shape[2:]))
        for k in ("deter", "stoch")
    }
    core = nj.pure(
        lambda carry, action: model.dyn._core(
            carry["deter"], carry["stoch"], nn.DictConcat(model.act_space, 1)(action)
        )
    )
    reward = nj.pure(lambda f: model.rew(model.feat2tensor(f), 1).pred())
    rollout = {}
    for steps in (1, 4):
        ensemble = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (8, *x.shape)).reshape((8 * n, *x.shape[1:])),
            starts,
        )
        valid = jnp.ones((b, width), bool)
        rows = {}
        for h in range(1, horizon + 1):
            valid &= ~reset[:, h : h + width]
            action = {
                k: data[k][:, h - 1 : h - 1 + width].reshape((n, *data[k].shape[2:]))
                for k in model.act_space
            }
            action = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (8, *x.shape)).reshape(
                    (8 * n, *x.shape[1:])
                ),
                action,
            )
            _, deter = core(params, ensemble, action, seed=seed)
            eps = jax.random.normal(
                jax.random.fold_in(key, 100 * h), (8 * n, mean.shape[-1])
            )
            stoch = nn.cast(
                sample(deter.astype(jnp.float32), eps, steps).reshape(
                    (8 * n, *feat["stoch"].shape[2:])
                )
            )
            ensemble = dict(deter=deter, stoch=stoch)
            rewards = reward(params, ensemble, seed=seed)[1].reshape(8, n)
            predicted = stoch.astype(jnp.float32).reshape(8, n, -1).mean(0)
            expected = feat["mean"][:, h : h + width].astype(jnp.float32).reshape(n, -1)
            variance = (
                feat["std"][:, h : h + width].astype(jnp.float32).reshape(n, -1) ** 2
            )
            mask = valid.reshape(n).astype(jnp.float32)
            count = mask.sum()
            reduce = lambda x: jnp.sum(x * mask) / jnp.maximum(count, 1)
            latent_mse = reduce(jnp.mean((predicted - expected) ** 2, -1))
            scale = reduce(jnp.mean(expected**2 + variance, -1)) + 1e-6
            observed_reward = data["reward"][:, h : h + width].reshape(n)
            reward_mse = reduce((rewards.mean(0) - observed_reward) ** 2)
            reward_mean = reduce(observed_reward)
            reward_scale = jnp.maximum(
                reduce((observed_reward - reward_mean) ** 2), 1.0
            )
            rows[str(h)] = dict(
                valid_paths=count,
                latent_mean_mse=latent_mse,
                normalized_latent_mean_mse=latent_mse / scale,
                latent_denominator=scale,
                reward_mse=reward_mse,
                reward_denominator=reward_scale,
                normalized_reward_mse=reward_mse / reward_scale,
                latent_variance_mse=reduce(
                    jnp.mean(
                        (stoch.astype(jnp.float32).reshape(8, n, -1).var(0) - variance)
                        ** 2,
                        -1,
                    )
                ),
            )
        rollout[str(steps)] = rows
    metrics["rollout"] = rollout
    metrics["quality_normalized_rollout_error"] = rollout["4"][
        str(max(map(int, rollout["4"])))
    ]["normalized_latent_mean_mse"]
    metrics["quality_gate"] = (
        dict(
            normalized_latent_mse_5=rollout["1"]["5"]["normalized_latent_mean_mse"],
            normalized_reward_mse_5=rollout["1"]["5"]["normalized_reward_mse"],
        )
        if "5" in rollout["1"]
        else None
    )

    def losses():
        _, posterior_feat = posterior()
        features = {k: posterior_feat[k] for k in ("deter", "stoch")}
        dyn = model.dyn._dynamics_loss(
            features["deter"], jax.lax.stop_gradient(features["stoch"])
        ).mean()
        _, _, recons = model.dec(model.dec.initial(b), features, reset, False)
        rec = sum(
            dist.loss(
                data[k].astype(jnp.float32) / 255
                if data[k].dtype == jnp.uint8 and len(model.obs_space[k].shape) == 3
                else data[k]
            ).mean()
            * model.scales[k]
            for k, dist in recons.items()
        )
        inp = model.feat2tensor(features)
        # Match actual upstream reward_grad routing, not hypothetical gradients.
        if not model.config.reward_grad:
            inp = jax.lax.stop_gradient(inp)
        rew = model.rew(inp, 2).loss(data["reward"]).mean() * model.scales["rew"]
        return dyn, rec, rew

    lossfun = nj.pure(losses)
    core_keys = [k for k in params if k.startswith("dyn/dyn")]
    if not core_keys:
        raise ValueError("Cannot identify shared recurrent core parameters")
    grads = {}
    gradient_arrays = []
    for index, label in enumerate(
        ("dynamics_unscaled", "observation_scaled", "reward_scaled")
    ):

        def objective(core_params):
            p = dict(params, **core_params)
            return lossfun(p, seed=seed)[1][index]

        value, gradient = jax.value_and_grad(objective)(
            {k: params[k] for k in core_keys}
        )
        gradient_arrays.append(gradient)
        grads[label] = dict(
            loss=value,
            shared_core_l2=jnp.sqrt(sum(jnp.sum(v**2) for v in gradient.values())),
        )
    metrics["gradient_norms"] = grads
    metrics["calibration"] = dict(
        shared_core_dynamics_grad_norm=grads["dynamics_unscaled"]["shared_core_l2"],
        shared_core_model_grad_norm=jnp.sqrt(
            sum(
                jnp.sum((gradient_arrays[1][k] + gradient_arrays[2][k]) ** 2)
                for k in core_keys
            )
        ),
    )
    current = {"tokens": tokens, "mean": mean, "std": std, "sample": feat["stoch"]}
    return metrics, current
