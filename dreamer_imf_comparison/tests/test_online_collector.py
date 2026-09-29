"""Small real JAX controller tests, with deterministic native-step environment."""

import json
import jax
import numpy as np
import pytest

from imf_dreamer_jax.config import DreamerConfig
from imf_dreamer_jax.flowmpc import ReBRACConfig, init_rebrac_state
from imf_dreamer_jax.world_model import init_world_model, init_transition_reward_head
from dreamer_imf_compare import online_collector as oc
from dreamer_imf_compare.dmc import Step


class TinyEnvironment:
    instances = []

    def __init__(self, task, *, seed, action_repeat):
        assert action_repeat == 1
        self.actions, self.closed = [], False
        self.instances.append(self)

    def reset(self):
        return np.asarray([0.1, 0.2, 0.3], np.float32)

    def step(self, action):
        self.actions.append(action.copy())
        n = len(self.actions)
        return Step(
            np.asarray([n, action[0], action[1]], np.float32),
            0.123456789012345 + n,
            1.0,
            n == 2,
        )

    def close(self):
        self.closed = True


@pytest.fixture(scope="module")
def setup():
    cfg = DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=5,
        stochastic_dim=3,
        embedding_dim=4,
        hidden_dim=8,
        prior="imf",
        imf_trajectory_enabled=True,
        reward_loss="mse",
        reward_prediction_horizon=0,
    )
    rc = ReBRACConfig(state_dim=3, action_dim=2, hidden_dim=8)
    world = init_world_model(cfg, jax.random.key(50))
    world["reward_transition"] = init_transition_reward_head(
        cfg, jax.random.key(51), hidden_dim=8
    )
    policy = init_rebrac_state(jax.random.key(52), rc)
    controller = dict(
        horizon=1,
        flowmpc_particles=2,
        flowmpc_step_size=5e-6,
        trust_anchor_mse_sum_budget=0.01,
        trust_current_linf_budget=0.1,
    )
    model = dict(
        reward_world=world, policies={541: policy}, anchors=np.zeros((3, 3), np.float32)
    )
    return oc.make_collector(cfg, rc, controller), model


def test_shifted_real_observations_actions_boundary_and_discarded_warmup(
    setup, monkeypatch
):
    collector, model = setup
    monkeypatch.setattr(oc, "DMCAdapter", TinyEnvironment)
    posterior_actions = []
    original = collector.observe

    def record(world, current, previous, belief, key):
        posterior_actions.append(np.asarray(previous))
        return original(world, current, previous, belief, key)

    monkeypatch.setattr(collector, "observe", record)
    episode, trace, metrics = collector.rollout(
        model, 541, 431, 19, maximum_steps=3, training=True, exploration_std=0.1
    )
    assert episode["observations"].shape == (1, 3, 3)
    assert metrics["native_steps"] == 2 and metrics["native_boundary"]
    assert metrics["discarded_compile_warmup_steps"] == 1
    assert len(TinyEnvironment.instances[-1].actions) == 2
    assert TinyEnvironment.instances[-1].closed
    np.testing.assert_array_equal(posterior_actions[-1][0], trace["actions"][0, 0])
    np.testing.assert_array_equal(episode["actions"][0, 1:], trace["actions"][0])
    np.testing.assert_array_equal(
        episode["observations"][0, -1], [2, *trace["actions"][0, -1]]
    )
    np.testing.assert_array_equal(episode["is_last"], [[False, False, True]])
    np.testing.assert_array_equal(episode["is_first"], [[True, False, False]])
    assert trace["rewards"].dtype == np.float64
    assert trace["rewards"][0, 0] == 1.123456789012345
    assert not np.array_equal(trace["clean_actions"], trace["actions"])
    assert trace["within_reference_budgets"].all()
    json.dumps(metrics, allow_nan=False)


def test_episode_reset_pairing_and_dynamic_parameters_without_recompile(
    setup, monkeypatch
):
    collector, model = setup
    monkeypatch.setattr(oc, "DMCAdapter", TinyEnvironment)
    _, first, _ = collector.rollout(model, 541, 431, 23)
    caches = collector.select._cache_size(), collector.observe._cache_size()
    _, second, metrics = collector.rollout(model, 541, 431, 23)
    np.testing.assert_array_equal(first["actions"], second["actions"])
    assert metrics["discarded_compile_warmup_steps"] == 0
    changed_world = dict(
        model,
        reward_world=jax.tree_util.tree_map(lambda x: x + 0.01, model["reward_world"]),
    )
    _, world_trace, _ = collector.rollout(changed_world, 541, 431, 23)
    assert not np.array_equal(
        first["objective_before"], world_trace["objective_before"]
    )
    changed_critics = model["policies"][541]._replace(
        critics=jax.tree_util.tree_map(
            lambda x: x + 0.01, model["policies"][541].critics
        )
    )
    _, critic_trace, _ = collector.rollout(
        dict(model, policies={541: changed_critics}), 541, 431, 23
    )
    assert not np.array_equal(
        first["objective_before"], critic_trace["objective_before"]
    )
    replacement = init_rebrac_state(
        jax.random.key(72), ReBRACConfig(state_dim=3, action_dim=2, hidden_dim=8)
    )
    changed = dict(
        model,
        policies={541: replacement},
        reward_world=jax.tree_util.tree_map(lambda x: x + 0.01, model["reward_world"]),
    )
    _, third, _ = collector.rollout(changed, 541, 431, 23)
    assert not np.array_equal(first["actions"], third["actions"])
    assert caches == (collector.select._cache_size(), collector.observe._cache_size())
    _, restored, _ = collector.rollout(model, 541, 431, 23)
    np.testing.assert_array_equal(first["actions"], restored["actions"])


def test_eval_exploration_rejected_before_environment_and_truncation(
    setup, monkeypatch
):
    collector, model = setup
    monkeypatch.setattr(oc, "DMCAdapter", TinyEnvironment)
    before = len(TinyEnvironment.instances)
    with pytest.raises(ValueError, match="evaluation exploration"):
        collector.rollout(model, 541, 431, 1, exploration_std=0.1)
    assert before == len(TinyEnvironment.instances)
    episode, trace, metrics = collector.rollout(model, 541, 431, 1, maximum_steps=1)
    assert metrics["truncated"] and not episode["is_last"].any()
    np.testing.assert_array_equal(trace["actions"], trace["clean_actions"])


def test_matches_existing_a1_rollout(setup, monkeypatch):
    from dreamer_imf_compare import actor_gap_roadmap_study as roadmap, dmc

    collector, model = setup
    monkeypatch.setattr(oc, "DMCAdapter", TinyEnvironment)
    monkeypatch.setattr(dmc, "DMCAdapter", TinyEnvironment)
    monkeypatch.setattr(roadmap, "_flowmpc_config", lambda rc: collector.flow)
    policy = model["policies"][541]
    rc = ReBRACConfig(state_dim=3, action_dim=2, hidden_dim=8)
    _, reference, _ = roadmap._run_flowmpc_arm(
        model["reward_world"],
        collector.cfg,
        policy,
        rc,
        world_seed=431,
        actor_seed=541,
        evaluation_seeds=[29],
        maximum_steps=2,
        trust=True,
        persistence="persistent",
        heldout_acceptance_enabled=False,
        anchor_observations=model["anchors"],
        task="dmc_reacher_hard",
    )
    _, actual, _ = collector.rollout(model, 541, 431, 29, maximum_steps=2)
    for name in (
        "actions",
        "objective_before",
        "objective_after",
        "gradient_norm",
        "anchor_drift",
        "current_drift",
        "parameter_delta",
    ):
        np.testing.assert_allclose(actual[name], reference[name], rtol=1e-5, atol=1e-7)


def test_random_prefill_partial_repeat_accounting_without_model(setup, monkeypatch):
    collector, _ = setup

    class PartialRepeat(TinyEnvironment):
        def __init__(self, task, *, seed, action_repeat):
            assert action_repeat == 2
            self.actions, self.closed = [], False
            self.instances.append(self)

        def step(self, action):
            transition = super().step(action)
            return Step(
                transition.observation,
                transition.reward,
                transition.continuation,
                transition.is_last,
                1 if transition.is_last else 2,
            )

    def forbidden(*args):
        raise AssertionError("random prefill must not call model")

    monkeypatch.setattr(oc, "DMCAdapter", PartialRepeat)
    monkeypatch.setattr(collector, "observe", forbidden)
    monkeypatch.setattr(collector, "select", forbidden)
    kwargs = dict(training=True, random_policy=True, action_repeat=2, maximum_steps=9)
    episode, trace, metrics = collector.rollout({}, 541, 431, 91, **kwargs)
    _, repeat, _ = collector.rollout({}, 541, 431, 91, **kwargs)
    _, other, _ = collector.rollout({}, 541, 431, 92, **kwargs)
    np.testing.assert_array_equal(trace["native_steps"], [[2, 1]])
    assert metrics["native_steps"] == 3 and metrics["decision_steps"] == 2
    assert metrics["discarded_compile_warmup_steps"] == 0
    np.testing.assert_array_equal(trace["actions"], trace["clean_actions"])
    np.testing.assert_array_equal(trace["actions"], repeat["actions"])
    assert not np.array_equal(trace["actions"], other["actions"])
    assert (np.abs(trace["actions"]) <= 1).all()
    assert not trace["gradient_norm"].any()
    assert not trace["parameter_delta"].any()
    np.testing.assert_array_equal(episode["actions"][0, 1:], trace["actions"][0])
    assert PartialRepeat.instances[-1].closed


def test_real_dmc_repeat_two_matches_native_reward_and_termination():
    from dreamer_imf_compare.dmc import DMCAdapter

    native = DMCAdapter("dmc_reacher_hard", seed=17, action_repeat=1)
    repeated = DMCAdapter("dmc_reacher_hard", seed=17, action_repeat=2)
    try:
        np.testing.assert_array_equal(native.reset(), repeated.reset())
        native_count = 0
        for decision in range(500):
            action = np.asarray([np.sin(decision), np.cos(decision)], np.float32)
            first, second = native.step(action), native.step(action)
            result = repeated.step(action)
            native_count += result.native_steps
            assert result.native_steps == 2
            assert result.reward == first.reward + second.reward
            assert result.continuation == first.continuation * second.continuation
            assert result.is_last == second.is_last == (decision == 499)
            np.testing.assert_array_equal(result.observation, second.observation)
        assert native_count == 1000
    finally:
        native.close()
        repeated.close()


def test_adapter_stops_partial_repeat_at_native_terminal():
    from types import SimpleNamespace
    from dreamer_imf_compare.dmc import DMCAdapter

    class TerminalEnvironment:
        calls = 0

        def step(self, action):
            self.calls += 1
            assert self.calls == 1, "must not step after terminal boundary"
            return SimpleNamespace(
                observation={"x": np.asarray([2.0])},
                reward=0.75,
                discount=0.0,
                last=lambda: True,
            )

    adapter = object.__new__(DMCAdapter)
    adapter.action_repeat = 2
    adapter.action_shape = (1,)
    adapter._action_spec = SimpleNamespace(minimum=[-1.0], maximum=[1.0])
    adapter._environment = TerminalEnvironment()
    result = adapter.step(np.asarray([0.2], np.float32))
    assert result.native_steps == 1
    assert result.reward == 0.75 and result.continuation == 0.0 and result.is_last
    assert adapter._environment.calls == 1


@pytest.mark.parametrize("repeat", [0, -1, True, 1.5])
def test_invalid_repeat_and_eval_random_are_rejected(setup, repeat):
    collector, model = setup
    with pytest.raises(ValueError, match="action_repeat"):
        collector.rollout(model, 541, 431, 1, action_repeat=repeat)
    with pytest.raises(ValueError, match="only permitted for training"):
        collector.rollout({}, 541, 431, 1, random_policy=True)
    with pytest.raises(ValueError, match="uniform exploration"):
        collector.rollout(
            {}, 541, 431, 1, random_policy=True, training=True, exploration_std=0.1
        )
