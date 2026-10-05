"""Schedule-only intervention and non-certifying, covered observable controls."""

import numpy as np

from .staged_runner import Schedule


class JointSchedule(Schedule):
    """Change only the freeze mask; retain all other legacy control rules."""

    def controls(self, native):
        return dict(super().controls(native), transition_only=False)


def retained_views(batch):
    """Ordinary prefix and explicitly biased real positive-reward window.

    Never manufacture rewards. Retain the full minibatch separately. A positive
    view is optional, is not a held-out sample and must not be pooled with the
    ordinary prefix to estimate on-policy reward frequency.
    """
    views = {"ordinary_prefix": {k: v[:2, :16].copy() for k, v in batch.items()}}
    hits = np.argwhere(batch["reward"][:, 1:] > 0)
    if len(hits):
        row, time = map(int, hits[0])
        time += 1
        start = max(0, min(time - 5, batch["reward"].shape[1] - 16))
        views["positive_selected"] = {
            k: v[row : row + 1, start : start + 16].copy() for k, v in batch.items()
        }
    return views


def assess(rows):
    """Fail closed on missing coverage or controls; never feed this into training."""
    original, shuffled = rows["model"], rows["shuffled_actions"]
    keys = (
        "valid_paths",
        "positive_paths",
        "zero_paths",
        "observation_mse",
        "persistence_mse",
        "reward_mse",
        "zero_reward_mse",
        "positive_reward_mse",
        "positive_zero_reward_mse",
    )
    values = [r[k] for r in (original, shuffled) for k in keys]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observable control")
    covered = original["positive_paths"] >= 8 and original["zero_paths"] >= 8
    better = (
        original["observation_mse"] < 0.95 * original["persistence_mse"]
        and original["observation_mse"] < 0.95 * shuffled["observation_mse"]
        and original["reward_mse"] < 0.95 * original["zero_reward_mse"]
        and original["positive_reward_mse"]
        < 0.95 * original["positive_zero_reward_mse"]
    )
    return dict(
        status=(
            "insufficient_reward_coverage"
            if not covered
            else "beats_controls_on_this_batch" if better else "fails_negative_controls"
        ),
        useful_control_certified=False,
        influences_training=False,
        note="Descriptive heuristic on correlated replay paths, not a statistical guarantee.",
    )


def evaluate_controls(model, params, batch, seed, steps=1):
    import jax
    from .staged_diagnostics import _json

    rows = _json(
        _observable_arrays(
            model, params, jax.tree.map(jax.device_put, batch), seed, steps
        )
    )
    return dict(
        rows=rows,
        assessment=assess(rows),
        batch_positive_rewards=int(np.count_nonzero(batch["reward"] > 0)),
        batch_rewards=int(batch["reward"].size),
        sampling_steps=steps,
    )


# Keep imports of JAX/upstream out of lightweight schedule/unit tests.
def _observable_arrays(model, params, data, seed, steps=1):
    from functools import partial
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import embodied.jax.nets as nn
    from imf_dreamer_jax.imf import sample_imf_steps

    # Memoize one compiled callable per raw model; no mutable learner state.
    fn = getattr(model, "_joint_observable_probe", None)
    if fn is None:

        @partial(jax.jit, static_argnums=(2, 3))
        def fn(params, data, seed, steps):
            b, t = data["is_first"].shape
            reset = data["is_first"].at[:, 0].set(True)
            actions = {
                k: jnp.concatenate([jnp.zeros_like(data[k][:, :1]), data[k][:, :-1]], 1)
                for k in model.act_space
            }

            def posterior():
                _, _, tokens = model.enc(model.enc.initial(b), data, reset, False)
                return model.dyn.observe(
                    model.dyn.initial(b), tokens, actions, reset, False
                )[2]

            feat = nj.pure(posterior)(params, seed=seed)[1]
            flow = nj.pure(model.dyn._flow_params)(params, seed=seed)[1]
            hmax, samples = 5, 8
            width = t - hmax
            n = b * width
            starts = {
                k: feat[k][:, :width].reshape(n, *feat[k].shape[2:])
                for k in ("deter", "stoch")
            }
            core = nj.pure(
                lambda f, a: model.dyn._core(
                    f["deter"], f["stoch"], nn.DictConcat(model.act_space, 1)(a)
                )
            )
            rew = nj.pure(lambda f: model.rew(model.feat2tensor(f), 1).pred())

            def decode(f):
                predictions = model.dec(
                    model.dec.initial(samples * n),
                    f,
                    jnp.zeros((samples * n,), bool),
                    False,
                    single=True,
                )[2]
                return {k: v.pred() for k, v in predictions.items()}

            decoder = nj.pure(decode)
            result = {}
            for mode in ("model", "shuffled_actions"):
                ensemble = jax.tree.map(
                    lambda x: jnp.tile(x, (samples,) + (1,) * (x.ndim - 1)), starts
                )
                valid = jnp.ones((b, width), bool)
                for h in range(1, hmax + 1):
                    valid &= ~reset[:, h : h + width]
                    act = {
                        k: data[k][:, h - 1 : h - 1 + width].reshape(
                            n, *data[k].shape[2:]
                        )
                        for k in model.act_space
                    }
                    if mode == "shuffled_actions":
                        order = jax.random.permutation(jax.random.PRNGKey(seed + h), n)
                        act = {k: v[order] for k, v in act.items()}
                    act = nn.cast(
                        jax.tree.map(
                            lambda x: jnp.tile(x, (samples,) + (1,) * (x.ndim - 1)), act
                        )
                    )
                    deter = core(params, ensemble, act, seed=seed)[1]
                    eps = jax.random.normal(
                        jax.random.PRNGKey(seed + 100 * h),
                        (samples * n, int(np.prod(feat["stoch"].shape[2:]))),
                    )
                    stoch = sample_imf_steps(
                        flow, deter.astype(jnp.float32), noise=eps, steps=steps
                    )
                    ensemble = dict(
                        deter=deter,
                        stoch=nn.cast(
                            stoch.reshape(samples * n, *feat["stoch"].shape[2:])
                        ),
                    )
                pred = decoder(params, ensemble, seed=seed)[1]
                obs_error, persist_error = [], []
                for k, value in pred.items():
                    expected = (
                        data[k][:, hmax : hmax + width]
                        .reshape(n, -1)
                        .astype(jnp.float32)
                    )
                    initial = data[k][:, :width].reshape(n, -1).astype(jnp.float32)
                    prediction = (
                        value.astype(jnp.float32).reshape(samples, n, -1).mean(0)
                    )
                    # Scale per coordinate using the retained target batch only.
                    scale = jnp.maximum(expected.var(0), 0.01)
                    obs_error.append(((prediction - expected) ** 2 / scale).mean(-1))
                    persist_error.append(((initial - expected) ** 2 / scale).mean(-1))
                reward = rew(params, ensemble, seed=seed)[1].reshape(samples, n).mean(0)
                truth = data["reward"][:, hmax : hmax + width].reshape(n)
                mask = valid.reshape(n)
                pos = mask & (truth > 0)
                reduce = lambda x, m: (x * m).sum() / jnp.maximum(m.sum(), 1)
                result[mode] = dict(
                    valid_paths=mask.sum(),
                    positive_paths=pos.sum(),
                    zero_paths=(mask & (truth == 0)).sum(),
                    observation_mse=reduce(sum(obs_error) / len(obs_error), mask),
                    persistence_mse=reduce(
                        sum(persist_error) / len(persist_error), mask
                    ),
                    reward_mse=reduce((reward - truth) ** 2, mask),
                    zero_reward_mse=reduce(truth**2, mask),
                    positive_reward_mse=reduce((reward - truth) ** 2, pos),
                    positive_zero_reward_mse=reduce(truth**2, pos),
                )
            return result

        # object.__setattr__ avoids registering diagnostic functions as parameters.
        object.__setattr__(model, "_joint_observable_probe", fn)
    return fn(params, data, seed, steps)
