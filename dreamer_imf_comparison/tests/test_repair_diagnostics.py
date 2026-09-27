"""Matched-state reward/continuation and immutable checkpoint controls."""

from copy import deepcopy
from dataclasses import asdict
import json
import pickle

import numpy as np
import pytest

from dreamer_imf_compare import repair_diagnostics as diagnostics
from dreamer_imf_compare.dmc import Step


def protocol():
    return dict(
        diagnostics=dict(
            actor_seed=541,
            calibration_environment_seeds=[76001],
            validation_environment_seeds=[76003],
            snapshot_steps=[25, 100, 300, 600],
            maximum_episode_steps=1000,
            horizon=5,
            direction_scales=[0.025, 0.05, 0.1],
            families=["recursive", "direct"],
            reference_modes=["latent", "endpoint"],
        ),
        controller=dict(
            horizon=5,
            action_sequence_particles=64,
            action_sequence_residual_limit=0.1,
            action_sequence_objective_evaluations=10,
        ),
        preflight=dict(
            diagnostic_snapshot_steps=[0],
            diagnostic_environment_seeds=[75901],
            diagnostic_continuation_steps=7,
        ),
        evaluation_environment_seeds=[78001, 78007, 78013],
    )


class ToyEnvironment:
    """The pre-action state and action have different reward coefficients."""

    def __init__(self, end=7):
        self.value = 1.0
        self.time = 0
        self.end = end
        self.restores = 0

    def snapshot(self):
        return self.value, self.time

    def restore(self, snapshot):
        self.value, self.time = snapshot
        self.restores += 1

    def step(self, action):
        reward = 10 * self.value + float(action[0])
        self.value += float(action[0])
        self.time += 1
        last = self.time == self.end
        return Step(np.array([self.value], np.float32), reward, float(not last), last)


def test_fixed_plan_restores_and_aligns_current_state_action_reward():
    env = ToyEnvironment(end=5)
    snapshot = env.snapshot()
    actions = np.array([[0.2], [-0.1]], np.float32)
    out = diagnostics.rollout_fixed_plan(
        env,
        snapshot,
        np.array([1.0]),
        actions,
        lambda obs: np.array([0.3]),
        discount=0.5,
        maximum_steps=5,
    )
    # Rewards are r(s0,a0), r(s1,a1), then the frozen actor from s2.
    expected = np.array([10.2, 11.9, 11.3, 14.3, 17.3])
    np.testing.assert_allclose(out["rewards"], expected, atol=1e-6)
    np.testing.assert_allclose(out["stage_observations"].ravel(), [1, 1.2, 1.1])
    assert out["stage_reward"] == pytest.approx(expected[0] + 0.5 * expected[1])
    assert out["terminal_value"] == pytest.approx(
        0.5**2 * (expected[2] + 0.5 * expected[3] + 0.5**2 * expected[4])
    )
    assert out["objective"] == out["stage_reward"] + out["terminal_value"]
    np.testing.assert_allclose(out["actions"][2:], 0.3)
    assert env.snapshot() == snapshot and env.restores == 2
    assert out["native_episode_end"] and out["length"] == 5
    repeated = diagnostics.rollout_fixed_plan(
        env,
        snapshot,
        np.array([1.0]),
        actions,
        lambda obs: np.array([0.3]),
        discount=0.5,
        maximum_steps=5,
    )
    for name in out:
        np.testing.assert_array_equal(out[name], repeated[name])


def test_restoration_even_when_continuation_actor_fails():
    env = ToyEnvironment()
    snapshot = env.snapshot()
    with pytest.raises(ValueError, match="invalid action"):
        diagnostics.rollout_fixed_plan(
            env,
            snapshot,
            np.array([1.0]),
            np.zeros((1, 1)),
            lambda obs: np.array([np.nan]),
            discount=0.99,
            maximum_steps=5,
        )
    assert env.snapshot() == snapshot and env.restores == 2


def test_termination_padding_and_preflight_truncation_are_distinct():
    env = ToyEnvironment(end=2)
    out = diagnostics.rollout_fixed_plan(
        env,
        env.snapshot(),
        np.array([1.0]),
        np.zeros((5, 1)),
        lambda obs: np.array([0.0]),
        discount=0.99,
        maximum_steps=7,
    )
    assert out["native_episode_end"] and out["stage_length"] == 2
    assert out["terminal_value"] == 0 and out["stage_reward"] == pytest.approx(19.9)
    np.testing.assert_array_equal(out["rewards"][2:], 0)
    env = ToyEnvironment(end=10)
    out = diagnostics.rollout_fixed_plan(
        env,
        env.snapshot(),
        np.array([1.0]),
        np.zeros((5, 1)),
        lambda obs: np.array([0.0]),
        discount=0.99,
        maximum_steps=7,
    )
    assert not out["native_episode_end"] and out["length"] == 7
    assert out["terminal_value"] == pytest.approx(0.99**5 * (10 + 0.99 * 10))


def test_ladder_is_seeded_bounded_and_reports_actual_clipped_changes():
    reference = np.array([[1.0, -1.0], [0.0, 0.0]], np.float32)
    plans, signed, direction = diagnostics.directional_ladder(
        reference, [0.025, 0.05, 0.1], 41
    )
    np.testing.assert_array_equal(
        signed, np.array([-0.025, 0.025, -0.05, 0.05, -0.1, 0.1], np.float32)
    )
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert np.max(np.abs(plans)) <= 1 and np.max(np.abs(plans - reference)) <= 0.1
    np.testing.assert_array_equal(
        plans, np.clip(reference + signed[:, None, None] * direction, -1, 1)
    )
    again = diagnostics.directional_ladder(reference, [0.025, 0.05, 0.1], 41)
    for a, b in zip((plans, signed, direction), again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(plans[0] - reference, -(plans[1] - reference))


def test_protocol_splits_and_parameters_are_fail_closed():
    p = protocol()
    original = deepcopy(p)
    assert diagnostics.resolve_settings(p, preflight=False)["splits"] == {
        "calibration": [76001],
        "validation": [76003],
    }
    assert diagnostics.resolve_settings(p, preflight=True)["splits"] == {
        "preflight": [75901]
    }
    assert p == original
    for mutation in (
        {"validation_environment_seeds": [76001]},
        {"calibration_environment_seeds": [78001]},
        {"calibration_environment_seeds": [76001, 76001]},
        {"snapshot_steps": [25, 25]},
        {"snapshot_steps": [999]},
        {"horizon": 4},
        {"direction_scales": [0.2]},
        {"families": ["recursive"]},
        {"reference_modes": ["wrong"]},
    ):
        bad = deepcopy(p)
        bad["diagnostics"].update(mutation)
        with pytest.raises(ValueError):
            diagnostics.resolve_settings(bad, preflight=False)
    bad = deepcopy(p)
    bad["preflight"]["diagnostic_environment_seeds"] = [76003]
    with pytest.raises(ValueError, match="preflight seeds"):
        diagnostics.resolve_settings(bad, preflight=True)


def test_training_support_excludes_heldout_and_reset_but_includes_last_transition():
    arrays = dict(
        train_episode_ids=np.array([0]),
        test_episode_ids=np.array([1]),
        rewards=np.array([[0, 0, 1, 2, 999], [0, 99, 99, 99, 99]], np.float32),
        continuations=np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], np.float32),
        is_first=np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], bool),
    )
    support = diagnostics.training_support(arrays)
    assert support["transitions"] == 3 and support["positive_rewards"] == 2
    assert support["positive_reward_fraction"] == pytest.approx(2 / 3)
    assert support["episode_return_mean"] == 3


def test_ranking_omits_true_ties_and_counts_predicted_ties_as_disagreement():
    agreement, count = diagnostics._ranking_agreement(
        np.array([1, 1, 2]), np.array([1, 2, 2])
    )
    assert count == 2 and agreement == 0.5
    assert diagnostics._ranking_agreement(np.array([1, 2]), np.array([0, 0])) == (
        None,
        0,
    )


def test_real_dmc_snapshot_rollout_is_exact_and_does_not_change_occupancy():
    pytest.importorskip("dm_control")
    from dreamer_imf_compare.dmc import DMCAdapter

    env = DMCAdapter("dmc_reacher_hard", seed=75901)
    try:
        observation = env.reset()
        snapshot = env.snapshot()
        actions = np.full((5, env.action_dim), 0.1, np.float32)
        policy = lambda obs: np.full(env.action_dim, -0.2, np.float32)
        first = diagnostics.rollout_fixed_plan(
            env, snapshot, observation, actions, policy, discount=0.99, maximum_steps=7
        )
        second = diagnostics.rollout_fixed_plan(
            env, snapshot, observation, actions, policy, discount=0.99, maximum_steps=7
        )
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        step = env.step(np.zeros(env.action_dim, np.float32))
        env.restore(snapshot)
        replay = env.step(np.zeros(env.action_dim, np.float32))
        np.testing.assert_array_equal(step.observation, replay.observation)
        assert step.reward == replay.reward
    finally:
        env.close()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import DreamerConfig, ReBRACConfig, init_rebrac_state
    from imf_dreamer_jax.imf import init_imf
    from imf_dreamer_jax.world_model import (
        init_world_model,
        init_transition_reward_head,
    )
    from dreamer_imf_compare.actor_gap_model_training import endpoint_condition_dim

    directory = tmp_path_factory.mktemp("frozen-diagnostic-checkpoint")
    cfg = DreamerConfig(
        observation_shape=(6,),
        action_dim=2,
        deterministic_dim=4,
        stochastic_dim=2,
        embedding_dim=4,
        hidden_dim=8,
        prior="imf",
    )
    rc = ReBRACConfig(state_dim=6, action_dim=2, hidden_dim=8, discount=0.99)
    world = init_world_model(cfg, jax.random.PRNGKey(1))
    reward_world = dict(
        world,
        reward_transition=init_transition_reward_head(
            cfg, jax.random.PRNGKey(2), hidden_dim=8
        ),
    )
    endpoint = dict(
        params=init_imf(
            jax.random.PRNGKey(3),
            sample_dim=6,
            condition_dim=endpoint_condition_dim(cfg, 5),
            hidden_dim=8,
            depth=1,
        ),
        maximum_chunk_horizon=5,
        observation_mean=jnp.zeros(6),
        observation_std=jnp.ones(6),
    )
    model = dict(
        config=asdict(cfg),
        rebrac_config=asdict(rc),
        world=world,
        reward_world=reward_world,
        policies={541: init_rebrac_state(jax.random.PRNGKey(4), rc)},
        endpoints={"endpoint_h1_imf": endpoint, "endpoint_anystep_imf": endpoint},
        thresholds={541: 0.0},
    )
    with (directory / "checkpoint.pkl").open("wb") as handle:
        pickle.dump(
            jax.tree_util.tree_map(
                lambda x: np.asarray(x) if hasattr(x, "dtype") else x, model
            ),
            handle,
        )
    np.savez(
        directory / "replay.npz",
        observations=np.zeros((2, 8, 6), np.float32),
        actions=np.zeros((2, 8, 2), np.float32),
        rewards=np.zeros((2, 8), np.float32),
        continuations=np.ones((2, 8), np.float32),
        is_first=np.zeros((2, 8), bool),
        train_episode_ids=np.array([0]),
        test_episode_ids=np.array([1]),
    )
    return directory


def test_tiny_frozen_dmc_diagnostic_components_rejection_replay_and_immutability(
    tiny_checkpoint,
):
    pytest.importorskip("dm_control")
    from dreamer_imf_compare import mechanism_replication as replication

    p = protocol()
    p["preflight"]["diagnostic_snapshot_steps"] = [0, 2]
    cell = dict(
        index=0,
        training_index=0,
        task="dmc_reacher_hard",
        world_model_seed=431,
        actor_seed=541,
    )
    before = {
        name: (tiny_checkpoint / name).read_bytes()
        for name in ("checkpoint.pkl", "replay.npz")
    }
    core, trace, timing = diagnostics.evaluate_diagnostics(
        p, cell, tiny_checkpoint, preflight=True
    )
    assert {name: (tiny_checkpoint / name).read_bytes() for name in before} == before
    assert core["preflight"] and set(core["summary"]) == {"preflight"}
    assert "truncated" in core["continuation_semantics"]
    assert core["actor_seed"] == 541 and timing["planner_calls"] == 8
    assert trace["action_sequence"].shape == (72, 5, 2)
    assert all(
        value.dtype.kind in "biuf" and np.isfinite(value).all()
        for value in trace.values()
    )
    assert trace["real_rewards"].shape == (72, 7)
    assert not trace["native_episode_end"].any()
    np.testing.assert_allclose(
        trace["model_stage"] + trace["model_terminal"],
        trace["model_objective"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        trace["real_stage"] + trace["real_terminal"],
        trace["real_objective"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        trace["terminal_transition_error"] + trace["terminal_critic_mc_error"],
        trace["model_terminal"] - trace["real_terminal"],
        atol=1e-12,
    )
    for family in (0, 1):
        for mode in (0, 1):
            for snapshot in (0, 1):
                indices = np.where(
                    (trace["family_id"] == family)
                    & (trace["reference_mode_id"] == mode)
                    & (trace["snapshot_id"] == snapshot)
                )[0]
                ref, proposed, executed = indices[:3]
                assert trace["used_fallback"][executed]
                np.testing.assert_array_equal(
                    trace["action_sequence"][executed], trace["action_sequence"][ref]
                )
                np.testing.assert_array_equal(
                    trace["real_rewards"][executed], trace["real_rewards"][ref]
                )
                np.testing.assert_array_equal(
                    trace["initial_observation"][indices],
                    np.repeat(
                        trace["initial_observation"][ref][None], len(indices), axis=0
                    ),
                )
                assert (
                    trace["plan_kind"][proposed] == 1
                )  # Rejected proposal is retained.
                assert trace["real_objective_gain"][executed] == 0
    json.dumps(core, allow_nan=False)
    settings = diagnostics.resolve_settings(p, preflight=True)
    diagnostics.validate_diagnostic_trace(trace, settings)
    for name in (
        "initial_observation",
        "first_action",
        "real_terminal",
        "model_objective_gain",
    ):
        corrupted = dict(trace)
        corrupted[name] = trace[name].copy()
        corrupted[name][0] += 0.01
        with pytest.raises(ValueError):
            diagnostics.validate_diagnostic_trace(corrupted, settings)
    # Equal gains against two different references do not imply equal levels.
    shifted = dict(trace)
    mode_mask = trace["reference_mode_id"] == 1
    for name in ("real_objective", "real_stage"):
        shifted[name] = trace[name].copy()
        shifted[name][mode_mask] += 1000
    shifted_summary = diagnostics.summarize_trace(shifted)["preflight"]["recursive"][
        "endpoint"
    ]
    unshifted = core["summary"]["preflight"]["recursive"]["endpoint"]
    assert shifted_summary["real_executed_objective_mean"] == pytest.approx(
        unshifted["real_executed_objective_mean"] + 1000
    )
    assert (
        shifted_summary["real_executed_objective_gain_mean"]
        == unshifted["real_executed_objective_gain_mean"]
    )
    again, again_trace, _ = diagnostics.evaluate_diagnostics(
        p, cell, tiny_checkpoint, preflight=True
    )
    assert core == again
    replication.exact_trace(trace, again_trace)
    # A separate clean actor rollout is an independent occupancy oracle.
    import jax
    import jax.numpy as jnp
    from imf_dreamer_jax import rebrac_actor
    from dreamer_imf_compare.dmc import DMCAdapter

    actor = replication.load_checkpoint(tiny_checkpoint)["policies"][541].actor
    policy = jax.jit(lambda obs: rebrac_actor(actor, obs))
    env = DMCAdapter(cell["task"], seed=75901)
    try:
        observation = env.reset()
        for step in range(7):
            np.testing.assert_array_equal(
                observation, trace["baseline_observations"][0, step]
            )
            action = np.asarray(policy(jnp.asarray(observation[None]))[0])
            np.testing.assert_array_equal(action, trace["baseline_actions"][0, step])
            transition = env.step(action)
            assert transition.reward == trace["baseline_rewards"][0, step]
            observation = transition.observation
    finally:
        env.close()


def test_checkpoint_digest_has_value_semantics(tiny_checkpoint):
    from dreamer_imf_compare.mechanism_replication import load_checkpoint

    one, two = load_checkpoint(tiny_checkpoint), load_checkpoint(tiny_checkpoint)
    assert one["config"] is not two["config"]
    assert diagnostics._model_digest(one) == diagnostics._model_digest(two)
    two["thresholds"][541] = 0.1
    assert diagnostics._model_digest(one) != diagnostics._model_digest(two)


def test_diagnostic_detects_frozen_model_mutation(tiny_checkpoint, monkeypatch):
    pytest.importorskip("dm_control")
    from dreamer_imf_compare import repaired_controllers

    original = repaired_controllers.make_endpoint_planner

    def mutating_factory(model, *args, **kwargs):
        planner = original(model, *args, **kwargs)
        model["thresholds"][541] += 0.001
        return planner

    monkeypatch.setattr(repaired_controllers, "make_endpoint_planner", mutating_factory)
    with pytest.raises(RuntimeError, match="mutated the frozen model"):
        diagnostics.evaluate_diagnostics(
            protocol(),
            dict(task="dmc_reacher_hard", world_model_seed=431),
            tiny_checkpoint,
            preflight=True,
        )


def test_full_diagnostic_refuses_truncated_continuations(tiny_checkpoint):
    pytest.importorskip("dm_control")
    p = protocol()
    p["diagnostics"].update(snapshot_steps=[0], maximum_episode_steps=7)
    with pytest.raises(ValueError, match="truncated before native episode end"):
        diagnostics.evaluate_diagnostics(
            p,
            dict(task="dmc_reacher_hard", world_model_seed=431),
            tiny_checkpoint,
            preflight=False,
        )
