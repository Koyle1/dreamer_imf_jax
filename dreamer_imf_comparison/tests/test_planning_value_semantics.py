"""Executable semantics of the existing critic, scorer, and real continuation.

Tiny deterministic components expose the production composition without fitting
parameters, loading checkpoints, launching DMC, or changing a controller study.
"""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamer_imf_compare import repaired_controllers as repaired
from dreamer_imf_compare.dmc import Step
from dreamer_imf_compare.flowmpc_actor_study import _build_rebrac_dataset
from dreamer_imf_compare.repair_diagnostics import rollout_fixed_plan
from imf_dreamer_jax.flowmpc import (
    ReBRACConfig,
    ReBRACDataset,
    init_rebrac_actor,
    init_rebrac_critics,
    rebrac_critic_loss,
)
from imf_dreamer_jax.robust_flowmpc import feasible_first_order

CONTROLLER = dict(
    horizon=5,
    action_sequence_particles=64,
    action_sequence_objective_evaluations=10,
    action_sequence_residual_limit=0.1,
)


def _constant_network(network, output):
    result = jax.tree_util.tree_map(jnp.zeros_like, network)
    result["output"]["bias"] = jnp.asarray(output, jnp.float32)
    return result


@pytest.fixture(scope="module")
def loss_inputs():
    config = ReBRACConfig(
        state_dim=1, action_dim=2, hidden_dim=2, discount=0.5, critic_bc_coefficient=2.0
    )
    actor = _constant_network(
        init_rebrac_actor(jax.random.key(71), config), np.arctanh([0.8, -0.8])
    )
    critics = init_rebrac_critics(jax.random.key(72), config)
    targets = {
        "members": tuple(
            _constant_network(member, [value])
            for member, value in zip(critics["members"], [10.0, 14.0], strict=True)
        )
    }
    batch = ReBRACDataset(
        states=jnp.zeros((1, 1)),
        actions=jnp.zeros((1, 2)),
        rewards=jnp.array([3.0]),
        next_states=jnp.ones((1, 1)),
        next_actions=jnp.array([[0.2, -0.4]]),
        dones=jnp.zeros(1),
    )
    return config, actor, critics, targets, batch


@pytest.mark.parametrize(
    "noise, penalty, expected",
    [
        ([100.0, -100.0], 1.0, 7.0),  # noise clips to +/- .5; actions to +/- 1
        ([1.0, -1.0], 1.0, 7.0),  # scale .2 is applied before action clipping
        ([0.0, 0.0], 0.52, 7.48),
        ([-1.0, 1.0], 0.20, 7.80),
        ([-100.0, 100.0], 0.02, 7.98),
    ],
)
def test_actual_loss_uses_smoothed_actions_sum_penalty_and_minimum(
    loss_inputs, noise, penalty, expected
):
    config, actor, critics, targets, batch = loss_inputs
    loss, (_, target, measured_penalty) = rebrac_critic_loss(
        critics, targets, actor, batch, jnp.array([noise]), config
    )
    assert np.isfinite(float(loss))
    assert float(measured_penalty) == pytest.approx(penalty, abs=2e-7)
    assert float(target) == pytest.approx(expected, abs=1e-6)
    _, (_, raw_target, _) = rebrac_critic_loss(
        critics,
        targets,
        actor,
        batch,
        jnp.array([noise]),
        replace(config, critic_bc_coefficient=0),
    )
    assert float(raw_target) == 8.0
    # Changing only the larger critic cannot change the clipped-double target.
    changed = {
        "members": (
            targets["members"][0],
            _constant_network(targets["members"][1], [100.0]),
        )
    }
    _, (_, same, _) = rebrac_critic_loss(
        critics, changed, actor, batch, jnp.array([noise]), config
    )
    assert float(same) == float(target)


def test_actual_loss_masks_bootstrap_at_true_terminal_only(loss_inputs):
    config, actor, critics, targets, batch = loss_inputs
    _, (_, target, penalty) = rebrac_critic_loss(
        critics,
        targets,
        actor,
        batch._replace(dones=jnp.ones(1)),
        jnp.array([[100.0, -100.0]]),
        config,
    )
    assert float(target) == 3.0
    assert float(penalty) == pytest.approx(1.0)


def test_replay_builder_retains_behavior_successor_and_omits_final_timeout():
    arrays = dict(
        train_episode_ids=np.array([0]),
        observations=np.array([[[0], [1], [2], [3]], [[90], [91], [92], [93]]]),
        actions=np.array([[[0], [0.1], [0.2], [0.3]], [[0], [0.9], [0.9], [0.9]]]),
        rewards=np.array([[0, 1, 2, 3], [0, 9, 9, 9]]),
        continuations=np.ones((2, 4)),
    )
    batch = _build_rebrac_dataset(arrays)
    np.testing.assert_array_equal(batch.states[:, 0], [0, 1])
    np.testing.assert_allclose(batch.next_actions[:, 0], [0.2, 0.3])
    np.testing.assert_array_equal(batch.rewards, [1, 2])
    np.testing.assert_array_equal(batch.dones, [0, 0])
    terminal = _build_rebrac_dataset(
        dict(arrays, continuations=np.array([[1, 0, 0, 0], [1, 1, 1, 1]]))
    )
    np.testing.assert_array_equal(terminal.states[:, 0], [0])
    np.testing.assert_array_equal(terminal.dones, [1])


class NativeTimeoutEnvironment:
    """State-integrator with a native time limit and nonzero final discount."""

    def __init__(self, remaining, value=0.0):
        self.remaining, self.value = remaining, value

    def snapshot(self):
        return self.remaining, self.value

    def restore(self, snapshot):
        self.remaining, self.value = snapshot

    def step(self, action):
        reward = self.value
        self.value += float(action[0])
        self.remaining -= 1
        return Step(
            np.array([self.value], np.float32), reward, 1.0, self.remaining == 0
        )


def _real_plan(sequence, remaining=8, value=0.0):
    env = NativeTimeoutEnvironment(remaining, value)
    snapshot = env.snapshot()
    result = rollout_fixed_plan(
        env,
        snapshot,
        np.array([value]),
        sequence,
        lambda observation: np.zeros(1, np.float32),
        discount=0.99,
        maximum_steps=remaining,
    )
    assert env.snapshot() == snapshot
    return result


def test_actual_mc_stops_at_native_timeout_despite_nonzero_discount():
    sequence = np.zeros((5, 1), np.float32)
    short, long = (_real_plan(sequence, remaining=n, value=1) for n in (5, 8))
    assert short["native_episode_end"] and long["native_episode_end"]
    assert np.all(long["continuations"] == 1)
    assert short["terminal_value"] == 0
    assert long["terminal_value"] == pytest.approx(sum(0.99**t for t in (5, 6, 7)))
    assert short["stage_reward"] == long["stage_reward"]
    # Identical observations and actor, different time remaining: different V.
    assert short["objective"] < long["objective"]


def _toy_model_and_components(monkeypatch, *, offset=0.0, error_slope=0.0):
    # Only components are analytical stand-ins. Composition, discounting,
    # feasibility, and ranking are the production functions under test.
    cfg = SimpleNamespace(observation_shape=(1,), observation_dim=1, action_dim=1)
    rc = ReBRACConfig(state_dim=1, action_dim=1)
    model = dict(
        config=cfg,
        rebrac_config=rc,
        reward_world={"reward_transition": None},
        policies={541: SimpleNamespace(actor=None, critics=None)},
        thresholds={541: 0.1},
        endpoints={"endpoint_anystep_imf": {"maximum_chunk_horizon": 5}},
    )
    monkeypatch.setattr(
        repaired,
        "_endpoint_predictor",
        lambda checkpoint, cfg: lambda start, actions, horizon, noise: start
        + jnp.sum(actions, axis=0)
        + noise,
    )
    monkeypatch.setattr(
        repaired, "rebrac_actor", lambda actor, obs: jnp.zeros_like(obs)
    )
    monkeypatch.setattr(
        repaired,
        "predict_state_action_reward_from_observation",
        lambda head, obs, actions, cfg: obs[:, 0],
    )

    def critics(params, obs, actions):
        # Three rewards remain after the H=5 plan in the eight-step toy episode.
        q = obs[:, 0] * (1 + 0.99 + 0.99**2 + error_slope) + offset
        return jnp.stack((q, q + 2))

    monkeypatch.setattr(repaired, "rebrac_critics", critics)
    return model


def _scores(model):
    scorer = repaired.make_endpoint_scorer(
        model, 541, direct_any_step=True, controller=CONTROLLER
    )
    sequences = np.zeros((2, 5, 1), np.float32)
    sequences[1, 0, 0] = 0.1
    scores = [
        scorer(jnp.zeros((1, 1)), jnp.asarray(sequence), jnp.zeros((64, 5, 1)))
        for sequence in sequences
    ]
    stacked = {
        key: jnp.stack([row[key] for row in scores])
        for key in ("objective", "feasible", "violation")
    }
    return (
        sequences,
        np.array(stacked["objective"]),
        np.array(feasible_first_order(stacked)),
    )


def test_actual_scorer_and_ranker_cancel_common_value_offset(monkeypatch):
    model = _toy_model_and_components(monkeypatch)
    sequences, original, order = _scores(model)
    real = np.array([_real_plan(sequence)["objective"] for sequence in sequences])
    np.testing.assert_allclose(original, real, atol=2e-7)
    assert order[0] == 1
    shifted = _toy_model_and_components(monkeypatch, offset=-16)
    _, values, shifted_order = _scores(shifted)
    np.testing.assert_allclose(values - original, -(0.99**5) * 16, atol=1e-5)
    np.testing.assert_allclose(values[1] - values[0], real[1] - real[0], atol=1e-5)
    np.testing.assert_array_equal(order, shifted_order)


def test_actual_scorer_reverses_ranking_under_candidate_dependent_error(monkeypatch):
    model = _toy_model_and_components(monkeypatch, error_slope=-10)
    sequences, values, order = _scores(model)
    real = np.array([_real_plan(sequence)["objective"] for sequence in sequences])
    assert real[1] > real[0] and order[0] == 0
    errors = values - real
    true_gap = real[1] - real[0]
    assert errors[0] - errors[1] > true_gap
    # Pairwise range bound is invariant to any common offset and tighter than
    # twice the uncentered absolute error when the offset is large.
    regret = real.max() - real[order[0]]
    assert regret <= np.ptp(errors) + 1e-6
    assert regret <= 2 * np.max(np.abs(errors - np.mean(errors))) + 1e-6


def test_actual_scorer_averages_particle_minima_not_minimum_particle_means(monkeypatch):
    model = _toy_model_and_components(monkeypatch)
    monkeypatch.setattr(
        repaired,
        "rebrac_critics",
        lambda params, obs, actions: jnp.stack((obs[:, 0], -obs[:, 0])),
    )
    scorer = repaired.make_endpoint_scorer(
        model, 541, direct_any_step=True, controller=CONTROLLER
    )
    noise = jnp.broadcast_to(jnp.array([-1.0, 1.0] * 32)[:, None, None], (64, 5, 1))
    result = scorer(jnp.zeros((1, 1)), jnp.zeros((5, 1)), noise)
    # Each critic has particle mean zero, while every particle minimum is -1.
    assert float(result["terminal_q_raw"]) == -1.0
    assert float(result["terminal_value"]) == pytest.approx(-(0.99**5), abs=1e-7)


def test_stochastic_endpoint_reference_can_be_infeasible_and_falls_back(monkeypatch):
    model = _toy_model_and_components(monkeypatch)
    monkeypatch.setattr(repaired, "rebrac_actor", lambda actor, obs: jnp.tanh(obs))
    noise = jnp.broadcast_to(jnp.array([-1.0, 1.0] * 32)[:, None, None], (64, 5, 1))
    planner = repaired.make_endpoint_planner(
        model,
        541,
        direct_any_step=True,
        controller=CONTROLLER,
        reference_mode="endpoint",
    )
    result = planner(None, jnp.zeros((1, 1)), noise, jnp.zeros((2, 4, 5, 1)))
    # Zero-noise reference uses actor(0)=0 at every step, while particle drift
    # measures mean abs(tanh(+/-1)), not abs(mean(tanh(+/-1)))=0.
    assert float(result["reference_behavior_distance"]) == pytest.approx(
        np.tanh(1), abs=1e-6
    )
    assert not bool(result["reference_feasible"])
    assert bool(result["used_fallback"])
    assert not bool(result["executed_feasible"])
    np.testing.assert_array_equal(
        result["executed_sequence"], result["reference_sequence"]
    )
    np.testing.assert_array_equal(
        result["executed_objective"], result["reference_objective"]
    )
