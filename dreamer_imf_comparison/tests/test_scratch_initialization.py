import copy

import jax
import numpy as np
import pytest

from dreamer_imf_compare import online_learning as ol
from dreamer_imf_compare import scratch_initialization as si
from dreamer_imf_compare.scratch_initialization import initialize_model


def protocol():
    return dict(
        task="dmc_reacher_hard",
        actor_seed=541,
        anchor_count=3,
        reward_hidden_dim=4,
        rebrac_config={"hidden_dim": 4},
        world_model_template=dict(
            observation_shape=[1],
            action_dim=1,
            deterministic_dim=4,
            stochastic_dim=2,
            embedding_dim=4,
            hidden_dim=4,
            imf_trajectory_enabled=True,
        ),
    )


def prefill():
    return dict(
        training=True,
        observations=np.arange(4, dtype=np.float32).reshape(1, 4, 1),
        actions=np.zeros((1, 4, 1), np.float32),
        rewards=np.zeros((1, 4), np.float32),
        continuations=np.ones((1, 4), np.float32),
        is_first=np.array([[True, False, False, False]]),
    )


def identical(a, b):
    return all(
        np.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def test_fresh_random_reproducible_weights_zero_clocks_and_own_statistics():
    p = protocol()
    a = initialize_model(p, 431, [prefill()])
    b = initialize_model(p, 431, [prefill()])
    c = initialize_model(p, 433, [prefill()])
    assert identical(a["world"], b["world"])
    assert not identical(a["world"], c["world"])
    assert identical(a["policies"], b["policies"])
    assert all(
        isinstance(leaf, np.ndarray)
        for leaf in jax.tree_util.tree_leaves(
            (a["world"], a["reward_world"], a["policies"], a["anchors"])
        )
    )
    state = ol.initialize(a, 541)
    assert int(state["policy"].step) == 0
    for opt in (
        state["world_optimizer"],
        state["reward_optimizer"],
        state["policy"].actor_optimizer,
        state["policy"].critic_optimizer,
    ):
        assert all(not np.any(v) for v in jax.tree_util.tree_leaves(opt))
    reward = a["reward_world"]["reward_transition"]
    np.testing.assert_array_equal(reward["observation_mean"], [1.5])
    np.testing.assert_allclose(reward["observation_std"], [np.std(np.arange(4))])
    np.testing.assert_equal(a["anchors"][:, 0], [0, 1, 3])
    shifted = prefill()
    shifted["observations"] += 100
    d = initialize_model(p, 431, [shifted])
    assert identical(a["world"], d["world"])
    assert identical(
        reward["network"], d["reward_world"]["reward_transition"]["network"]
    )
    np.testing.assert_equal(d["anchors"], a["anchors"] + 100)


def test_cpu_initialization_context_is_local_and_applies_to_all_networks(monkeypatch):
    observed = []
    original_default = jax.config.jax_default_device
    for name, key_position in (
        ("init_world_model", 1),
        ("init_transition_reward_state", 1),
        ("init_rebrac_state", 0),
    ):
        original = getattr(si, name)

        def checked(*args, _fn=original, _pos=key_position, **kwargs):
            observed.append(args[_pos].device.platform)
            assert jax.config.jax_default_device.platform == "cpu"
            return _fn(*args, **kwargs)

        monkeypatch.setattr(si, name, checked)
    a = initialize_model(protocol(), 431, [prefill()])
    assert observed == ["cpu", "cpu", "cpu"]
    assert jax.config.jax_default_device == original_default
    b = initialize_model(protocol(), 431, [prefill()])
    assert identical(a["world"], b["world"])
    assert identical(a["policies"], b["policies"])
    assert identical(a["reward_world"], b["reward_world"])


def test_no_data_bootstrap_and_inferred_dimensions():
    p = protocol()
    a = initialize_model(p, 431)
    assert not np.any(a["anchors"])
    np.testing.assert_array_equal(
        a["reward_world"]["reward_transition"]["observation_std"], [1]
    )
    del p["world_model_template"]["observation_shape"]
    del p["world_model_template"]["action_dim"]
    with pytest.raises(ValueError, match="explicit dimensions"):
        initialize_model(p, 431)
    assert initialize_model(p, 431, [prefill()])["config"].observation_shape == (1,)


@pytest.mark.parametrize(
    "change",
    [
        {"training": False},
        {"training": None},
        {"split": "evaluation"},
        {"train_episode_ids": np.array([0])},
        {"test_episode_ids": np.array([0])},
        {"observations": np.full((1, 4, 1), np.nan)},
        {"actions": np.ones((1, 4, 1)) * 2},
    ],
)
def test_reject_ineligible_prefill(change):
    a = prefill()
    a.update(change)
    with pytest.raises(ValueError):
        initialize_model(protocol(), 431, [a])


def test_mismatched_dimensions_fail_closed():
    p = copy.deepcopy(protocol())
    p["world_model_template"]["action_dim"] = 2
    with pytest.raises(ValueError, match="action dimension"):
        initialize_model(p, 431, [prefill()])
    p = protocol()
    p["rebrac_config"]["state_dim"] = 2
    with pytest.raises(ValueError, match="ReBRAC dimensions"):
        initialize_model(p, 431, [prefill()])
