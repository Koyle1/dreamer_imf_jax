import json
import pickle
from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamer_imf_compare import online_learning as ol
from imf_dreamer_jax.agent import create_agent
from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.flowmpc import (
    ReBRACConfig,
    init_rebrac_state,
    rebrac_actor_loss,
    rebrac_critics,
)
from imf_dreamer_jax.transition_reward import (
    init_transition_reward_state,
    _aligned_state_action_inputs,
)


def replay(offset=0, terminal=False):
    observations = np.arange(12, dtype=np.float32).reshape(2, 6, 1) + offset
    actions = np.broadcast_to(
        np.arange(6, dtype=np.float32)[None, :, None] / 10, (2, 6, 1)
    ).copy()
    rewards = np.broadcast_to(np.arange(6, dtype=np.float32), (2, 6)).copy()
    cont = np.ones((2, 6), np.float32)
    if terminal:
        cont[:, 3:] = 0
    first = np.zeros((2, 6), bool)
    first[:, 0] = True
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continuations=cont,
        is_first=first,
        train_episode_ids=np.array([0]),
        test_episode_ids=np.array([1]),
    )


def online(offset=100, terminal=False):
    a = replay(offset, terminal)
    return {k: v[:1] for k, v in a.items() if k in ol.FIELDS}


def equal(a, b):
    aa, at = jax.tree_util.tree_flatten(a)
    bb, bt = jax.tree_util.tree_flatten(b)
    return at == bt and all(np.array_equal(x, y) for x, y in zip(aa, bb))


@pytest.fixture(scope="module")
def model():
    cfg = DreamerConfig(
        observation_shape=(1,),
        action_dim=1,
        deterministic_dim=4,
        stochastic_dim=2,
        embedding_dim=4,
        hidden_dim=4,
        imf_trajectory_enabled=True,
        burn_in=1,
        overshooting_scale=0,
    )
    rc = ReBRACConfig(state_dim=1, action_dim=1, hidden_dim=4)
    world = create_agent(cfg, jax.random.PRNGKey(1)).params.world_model
    reward = init_transition_reward_state(
        cfg, jax.random.PRNGKey(2), hidden_dim=4
    ).params
    p = init_rebrac_state(jax.random.PRNGKey(3), rc)._replace(step=jnp.array(4))
    return dict(
        config=cfg,
        rebrac_config=rc,
        world=world,
        reward_world={**world, "reward_transition": reward},
        policies={541: p},
        anchors=np.array([[0.2]], np.float32),
        thresholds={541: 0.1},
    )


def test_causal_transitions_actual_next_action_and_timeout():
    data = ol._dataset(ol._episodes(replay(), offline=True))
    np.testing.assert_array_equal(data.states[:, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(data.next_states[:, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(data.actions[:, 0], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(data.next_actions[:, 0], [0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(data.rewards, [1, 2, 3, 4])
    assert np.max(data.states) < 6  # heldout episode never sampled
    terminal = ol._dataset(ol._episodes(replay(terminal=True), offline=True))
    np.testing.assert_array_equal(terminal.dones, [0, 0, 1])
    assert terminal.next_actions[-1, 0] == 0
    padded_timeout = replay()
    padded_timeout["is_last"] = np.zeros((2, 6), bool)
    padded_timeout["is_last"][:, 3] = True
    assert len(ol._dataset(ol._episodes(padded_timeout, offline=True)).states) == 2


def test_sequence_mixture_alignment_padding_and_fixed_shape():
    old, new = ol._episodes(replay(), offline=True), ol._episodes(
        online(), offline=False
    )
    for seed in range(12):
        b = ol._sequence_batch(old, new, np.random.default_rng(seed), 4, 7, 2)
        assert b["observations"].shape == (4, 9, 1)
        assert np.all(b["observations"][:2] < 6)
        assert np.all(b["observations"][2:] >= 100)
        previous, actions = _aligned_state_action_inputs(
            b["observations"], b["actions"], b["is_first"]
        )
        mask = np.asarray(b["loss_mask"], bool)
        np.testing.assert_array_equal(np.asarray(b["observations"] - previous)[mask], 1)
        np.testing.assert_allclose(
            np.asarray(actions[..., 0])[mask] * 10, np.asarray(b["rewards"])[mask]
        )
        assert not np.any(mask[:, :2])
    for seed in range(8):
        batch = ol._sequence_batch(old, new, np.random.default_rng(seed), 4, 3, 2)
        np.testing.assert_array_equal(np.sum(batch["loss_mask"], axis=1), 3)


def test_reject_eval_overlap_resets_and_empty():
    a = replay()
    a["train_episode_ids"] = np.array([1])
    with pytest.raises(ValueError, match="overlap"):
        ol._episodes(a, offline=True)
    with pytest.raises(ValueError, match="evaluation"):
        ol._episodes({**online(), "training": False}, offline=False)
    a = online()
    a["is_first"][0, 3] = True
    with pytest.raises(ValueError, match="reset"):
        ol._episodes(a, offline=False)
    with pytest.raises(ValueError, match="nonempty"):
        ol._episodes([], offline=False)


def test_policy_batch_is_exactly_mixed_with_fixed_shape():
    old = ol._dataset(ol._episodes(replay(), offline=True))
    for count in (1, 3):
        new = ol._dataset(
            ol._episodes([online(100 + i * 10) for i in range(count)], offline=False)
        )
        batch = ol._policy_batch(old, new, np.random.default_rng(5), 1024)
        assert batch.states.shape == (1024, 1)
        assert np.count_nonzero(np.asarray(batch.states[:, 0]) < 6) == 512
        assert np.count_nonzero(np.asarray(batch.states[:, 0]) >= 100) == 512


def test_init_fresh_optimizers_retained_policy_export_and_pickle(model):
    state = ol.initialize(
        {
            **model,
            "config": asdict(model["config"]),
            "rebrac_config": asdict(model["rebrac_config"]),
        },
        541,
    )
    assert state["world_optimizer"].step == state["reward_optimizer"].step == 0
    assert equal(state["policy"], model["policies"][541])
    restored = pickle.loads(pickle.dumps(state))
    assert equal(restored["world"], model["world"])
    exported = ol.export_model(restored, model, 541)
    assert exported["anchors"] is model["anchors"]
    assert exported["thresholds"] is model["thresholds"]
    assert equal(exported["reward_world"], model["reward_world"])


def test_policy_only_freeze_full_model_updates_and_serializable_resume(model):
    s = ol.initialize(model, 541)
    p, pm = ol.update(
        s,
        replay(),
        online(),
        full_model=False,
        seed=44,
        world_updates=2,
        policy_updates=2,
        batch_size=2,
        sequence_length=2,
    )
    for k in ("world", "reward", "world_optimizer", "reward_optimizer"):
        assert equal(p[k], s[k])
    assert not equal(p["policy"].actor, s["policy"].actor)
    assert not equal(p["policy"].critics, s["policy"].critics)
    assert p["policy"].step == 6 and pm["world_updates"] == 0
    m, mm = ol.update(
        s,
        replay(),
        online(),
        full_model=True,
        seed=44,
        world_updates=1,
        policy_updates=2,
        batch_size=2,
        sequence_length=2,
    )
    assert not equal(m["world"], s["world"])
    assert not equal(m["reward"], s["reward"])
    assert m["world_optimizer"].step == m["reward_optimizer"].step == 1
    for key in ("observation_mean", "observation_std"):
        np.testing.assert_array_equal(m["reward"][key], s["reward"][key])
    json.dumps(mm, allow_nan=False)
    restored = pickle.loads(pickle.dumps(p))
    kwargs = dict(full_model=False, seed=55, world_updates=0, policy_updates=1)
    resumed, _ = ol.update(restored, replay(), online(), **kwargs)
    repeated, _ = ol.update(p, replay(), online(), **kwargs)
    assert equal(resumed["policy"], repeated["policy"])
    assert equal(s["policy"], model["policies"][541])
    assert equal(s["world"], model["world"])
    frozen, fm = ol.update(
        s,
        replay(),
        online(),
        full_model=False,
        seed=5,
        world_updates=0,
        policy_updates=0,
    )
    for k in ("world", "reward", "policy", "world_optimizer", "reward_optimizer"):
        assert equal(frozen[k], s[k])
    assert fm["world_updates"] == fm["policy_updates"] == 0


def test_zero_q_guard_and_ordinary_scale(model):
    p, rc = model["policies"][541], model["rebrac_config"]
    batch = ol._dataset(ol._episodes(replay(), offline=True))
    batch = jax.tree_util.tree_map(jnp.asarray, batch)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, p.critics)
    loss, aux = rebrac_actor_loss(p.actor, zeros, batch, rc)
    assert np.isfinite(loss) and np.isfinite(aux[-1])
    gradients = jax.grad(lambda actor: rebrac_actor_loss(actor, zeros, batch, rc)[0])(
        p.actor
    )
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(gradients))
    _, aux = rebrac_actor_loss(p.actor, p.critics, batch, rc)
    from imf_dreamer_jax.flowmpc import rebrac_actor

    q = jnp.min(
        rebrac_critics(p.critics, batch.states, rebrac_actor(p.actor, batch.states)),
        axis=0,
    )
    np.testing.assert_allclose(aux[-1], 1 / jnp.mean(jnp.abs(q)))
    with pytest.raises(ValueError):
        replace(rc, q_scale_floor=0)
