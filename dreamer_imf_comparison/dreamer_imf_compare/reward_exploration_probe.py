"""Recomputed posterior and recorded-action prediction probes, no simulator.

These are not historical training gradients and not causal action-intervention
tests: recorded feedback-policy actions may themselves convey state information.
"""

from pathlib import Path

import numpy as np

from .parallel_collection import _write_npz


def retained_probe(
    agent, directory, milestone, seed, *, verify=False, data_override=None
):
    import jax
    import jax.numpy as jnp
    import ninjax as nj

    directory = Path(directory)
    files = [directory / f"eval-{milestone}-{ep}.npz" for ep in range(5)]
    if data_override is None:
        rows = []
        for path in files:
            with np.load(path, allow_pickle=False) as values:
                rows.append({k: np.array(values[k]) for k in values.files})
        data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    else:
        data = data_override
    m = agent.model
    anchors, particles, horizon = (100, 200, 300, 400), 8, 15
    # Shape (episode, anchor); flattening preserves clustered episode identity.
    indices = np.asarray(anchors, np.int32)
    truth_reward = np.stack(
        [data["reward"][:, t + 1 : t + 16] for t in anchors], 1
    ).reshape(20, 15)
    truth_obs = np.stack(
        [
            np.concatenate(
                [
                    data[k][:, t + 1 : t + 16]
                    for k in ("position", "to_target", "velocity")
                ],
                -1,
            )
            for t in anchors
        ],
        1,
    ).reshape(20, 15, 6)
    action = np.stack([data["action"][:, t : t + 15] for t in anchors], 1).reshape(
        20, 15, 2
    )
    selected_data = {k: data[k] for k in agent.obs_space}
    previous = data["previous_action"]

    def body(obs, previous, plans):
        reset = obs["is_first"]
        _, _, tokens = m.enc(m.enc.initial(5), obs, reset, False)
        _, _, posterior = m.dyn.observe(
            m.dyn.initial(5), tokens, {"action": previous}, reset, False
        )
        posterior = {k: posterior[k] for k in ("deter", "stoch")}
        starts = jax.tree.map(
            lambda x: x[:, jnp.array(anchors)].reshape((20, *x.shape[2:])), posterior
        )
        targetfeat = jax.tree.map(
            lambda x: jnp.stack([x[:, t + 1 : t + 16] for t in anchors], 1).reshape(
                (20, 15, *x.shape[2:])
            ),
            posterior,
        )
        inputs = m.feat2tensor(targetfeat)
        posterior_control = m.control_rew(inputs, 2).pred()
        posterior_original = m.rew(inputs, 2).pred()
        repeated = jax.tree.map(lambda x: jnp.repeat(x, particles, axis=0), starts)
        plans_repeated = jnp.repeat(plans, particles, axis=0)
        _, predicted, _ = m.dyn.imagine(
            repeated, {"action": plans_repeated}, horizon, False
        )
        prediction_input = m.feat2tensor(predicted)
        original = m.rew(prediction_input, 2).pred().reshape(20, particles, horizon)
        control = (
            m.control_rew(prediction_input, 2).pred().reshape(20, particles, horizon)
        )
        _, _, decoded = m.dec(
            m.dec.initial(20 * particles),
            predicted,
            jnp.zeros((20 * particles, horizon), bool),
            False,
        )
        observation = jnp.concatenate(
            [decoded[k].pred() for k in ("position", "to_target", "velocity")], -1
        ).reshape(20, particles, horizon, 6)
        cur_features = jnp.concatenate(
            [m.feat2tensor(starts)[:, None], inputs[:, :-1]], 1
        )
        means = m.ensemble(cur_features, plans)
        target_mean, target_std, _ = m.ensemble.statistics()
        return dict(
            posterior_control=posterior_control,
            posterior_original=posterior_original,
            predicted_control=control,
            predicted_original=original,
            predicted_observation=observation,
            ensemble_means=means,
            observation_mean=target_mean,
            observation_std=target_std,
        )

    pure = nj.pure(body)
    predict = jax.jit(
        lambda params, obs, prev, act: pure(
            params, obs, prev, act, seed=seed + milestone + 8851
        )[1]
    )
    inputs = (
        agent.params,
        jax.device_put(selected_data),
        jax.device_put(previous),
        jax.device_put(action),
    )
    # Pure discarded execution stabilizes compilation/execution role; neither
    # environment nor learner state is advanced by this diagnostic.
    jax.block_until_ready(predict(*inputs))
    arrays = predict(*inputs)
    arrays = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), arrays)
    for value in arrays.values():
        if not np.isfinite(value).all():
            raise ValueError("nonfinite recomputed prediction probe")
    arrays.update(
        truth_reward=truth_reward,
        truth_observation=truth_obs,
        action=action,
        episode=np.repeat(np.arange(5, dtype=np.int32), 4),
        anchor=np.tile(indices, 5),
    )
    path = directory / f"prediction-probe-{milestone}.npz"
    if verify:
        with np.load(path, allow_pickle=False) as retained:
            if set(retained.files) != set(arrays):
                raise ValueError("prediction replay fields differ")
            for key, value in arrays.items():
                if not np.array_equal(value, retained[key]):
                    raise ValueError("prediction replay differs: " + key)
    else:
        _write_npz(path, arrays)
    return path
