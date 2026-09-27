from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dreamer_imf_compare import repaired_controllers as repaired
from dreamer_imf_compare.actor_gap_model_training import endpoint_condition_dim
from imf_dreamer_jax import (
    DreamerConfig,
    ReBRACConfig,
    initial_state,
    init_rebrac_state,
)
from imf_dreamer_jax.flowmpc import FlowMPCConfig, flowmpc_objective
from imf_dreamer_jax.imf import init_imf
from imf_dreamer_jax import robust_flowmpc as robust
from imf_dreamer_jax.world_model import init_transition_reward_head, init_world_model

CONTROLLER = {
    "horizon": 5,
    "action_sequence_particles": 64,
    "action_sequence_objective_evaluations": 10,
    "action_sequence_residual_limit": 0.1,
}


@pytest.fixture(scope="module")
def model():
    cfg = DreamerConfig(
        observation_shape=(3,),
        action_dim=1,
        deterministic_dim=4,
        stochastic_dim=2,
        embedding_dim=4,
        hidden_dim=4,
        prior="imf",
        imf_trajectory_enabled=True,
    )
    rc = ReBRACConfig(state_dim=3, action_dim=1, hidden_dim=4)
    world = init_world_model(cfg, jax.random.PRNGKey(1))
    world["reward_transition"] = init_transition_reward_head(
        cfg,
        jax.random.PRNGKey(2),
        observation_mean=jnp.zeros(3),
        observation_std=jnp.ones(3),
        hidden_dim=4,
    )
    checkpoint = {
        "maximum_chunk_horizon": 5,
        "observation_mean": jnp.zeros(3),
        "observation_std": jnp.ones(3),
        "params": init_imf(
            jax.random.PRNGKey(3),
            sample_dim=3,
            condition_dim=endpoint_condition_dim(cfg, 5),
            hidden_dim=4,
            depth=1,
        ),
    }
    return {
        "config": cfg,
        "rebrac_config": rc,
        "reward_world": world,
        "policies": {7: init_rebrac_state(jax.random.PRNGKey(4), rc)},
        "thresholds": {7: 2.0},
        "endpoints": {
            "endpoint_h1_imf": checkpoint,
            "endpoint_anystep_imf": checkpoint,
        },
    }


def arrays(model):
    return (
        initial_state(model["config"], 1),
        jnp.zeros((1, 3)),
        jax.random.normal(jax.random.PRNGKey(5), (64, 5, 3)),
        jax.random.normal(jax.random.PRNGKey(6), (2, 4, 5, 1)),
    )


def tree_arrays(tree):
    return [np.array(x, copy=True) for x in jax.tree_util.tree_leaves(tree)]


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("reference_mode", ["latent", "endpoint"])
def test_real_model_planner_scores_executed_sequence_exactly(
    model, direct, reference_mode
):
    frozen = (model["reward_world"], model["policies"], model["endpoints"])
    before = tree_arrays(frozen)
    planner = repaired.make_endpoint_planner(
        model,
        7,
        direct_any_step=direct,
        controller=CONTROLLER,
        reference_mode=reference_mode,
    )
    inputs = arrays(model)
    result = planner(*inputs)
    replay = planner(*inputs)
    for left, right in zip(
        jax.tree_util.tree_leaves(result),
        jax.tree_util.tree_leaves(replay),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
        assert np.isfinite(left).all()
    assert int(result["evaluations"]) == 10
    assert not bool(result["used_fallback"])
    assert (
        np.max(np.abs(result["action_sequence"] - result["reference_sequence"]))
        <= 0.10000003
    )
    assert np.max(np.abs(result["action_sequence"])) <= 1.0
    np.testing.assert_array_equal(result["actions"], result["executed_sequence"][0])
    scorer = repaired.make_endpoint_scorer(
        model, 7, direct_any_step=direct, controller=CONTROLLER
    )
    measured = scorer(inputs[1], result["executed_sequence"], inputs[2])
    for key, value in measured.items():
        np.testing.assert_allclose(
            result[f"executed_{key}"], value, rtol=2e-6, atol=2e-6
        )
    np.testing.assert_allclose(
        result["executed_objective"],
        result["executed_stage_reward"] + result["executed_terminal_value"],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        result["executed_terminal_value"],
        model["rebrac_config"].discount ** 5 * result["executed_terminal_q_raw"],
        rtol=2e-6,
    )
    for left, right in zip(before, tree_arrays(frozen), strict=True):
        np.testing.assert_array_equal(left, right)


def test_precision_safe_order_and_stable_exact_ties():
    # A million-point sentinel maps these distinct distances to the same float.
    np.testing.assert_array_equal(
        np.float32(-1e6) - np.float32(0.70), np.float32(-1e6) - np.float32(0.71)
    )
    scores = {
        "objective": jnp.array([1e12, -10.0, -10.0, 1e15, -20.0]),
        "violation": jnp.array([0.71, 0.0, 0.0, 0.70, 0.0]),
        "feasible": jnp.array([False, True, True, False, True]),
    }
    np.testing.assert_array_equal(robust.feasible_first_order(scores), [1, 2, 4, 3, 0])
    nonfinite = {
        "objective": jnp.array([jnp.nan, 0.0, jnp.inf]),
        "violation": jnp.array([0.0, 0.7, 0.0]),
        "feasible": jnp.array([True, False, True]),
    }
    assert int(robust.feasible_first_order(nonfinite)[0]) == 1


def test_cem_improves_infeasible_violation_without_sentinel_or_extra_scores():
    domain = robust.make_action_sequence_domain(
        jnp.zeros((5, 1)), robust.ActionSequenceDomainConfig(5, 1, 0.1)
    )

    evaluations = []

    def score(actions):
        x = actions[0, 0]
        jax.debug.callback(lambda value: evaluations.append(float(value)), x)
        return {
            "objective": -1e5 * x,
            "violation": 0.71 - x,
            "feasible": jnp.asarray(False),
        }

    bank = jnp.zeros((2, 4, 5, 1)).at[:, 1, 0, 0].set(0.1).at[:, 2, 0, 0].set(0.05)
    result = jax.jit(
        lambda bank: robust.feasible_first_cem_action_sequence_search(
            score,
            domain,
            robust.ActionSequenceProposal(jnp.zeros((5, 1))),
            bank,
            robust.CEMActionSequenceConfig(
                iterations=2,
                population=4,
                elite_count=2,
                initial_std=0.1,
                minimum_std=0.01,
                maximum_std=0.1,
            ),
        )
    )(bank)
    assert float(result["final_metrics"]["violation"]) < float(
        result["initial_metrics"]["violation"]
    )
    assert float(result["final_metrics"]["objective"]) < float(
        result["initial_metrics"]["objective"]
    )
    assert int(result["evaluations"]) == 10
    jax.effects_barrier()
    assert len(evaluations) == 10


@pytest.mark.parametrize("direct", [False, True])
def test_infeasible_reference_falls_back_entire_sequence_with_reference_metrics(
    model, monkeypatch, direct
):
    local = dict(model, thresholds={7: 0.1})
    monkeypatch.setattr(
        repaired,
        "rebrac_actor",
        lambda actor, observation: jnp.tanh(observation[:, :1]),
    )
    monkeypatch.setattr(repaired, "decode", lambda *args: jnp.zeros((1, 3)))
    monkeypatch.setattr(
        repaired,
        "sample_endpoint_observation",
        lambda params, start, *args, **kwargs: jnp.ones_like(start),
    )
    monkeypatch.setattr(
        repaired,
        "predict_state_action_reward_from_observation",
        lambda head, observation, action, cfg: 1.0 + action[:, 0],
    )
    monkeypatch.setattr(
        repaired,
        "rebrac_critics",
        lambda critics, observation, action: jnp.zeros((2, observation.shape[0])),
    )
    planner = repaired.make_endpoint_planner(
        local, 7, direct_any_step=direct, controller=CONTROLLER
    )
    belief, current, noise, _ = arrays(model)
    bank = jnp.ones((2, 4, 5, 1))
    result = planner(belief, current, noise, bank)
    assert bool(result["used_fallback"])
    assert not bool(result["reference_feasible"])
    assert not bool(result["executed_feasible"])
    assert float(result["proposal_violation"]) < float(result["reference_violation"])
    assert float(result["proposal_objective"]) > float(result["executed_objective"])
    np.testing.assert_array_equal(
        result["executed_sequence"], result["reference_sequence"]
    )
    np.testing.assert_allclose(
        result["executed_stage_reward"], sum(0.99**t for t in range(5)), rtol=2e-7
    )
    np.testing.assert_array_equal(result["executed_reward_mean"], np.ones(5))
    np.testing.assert_array_equal(result["executed_reward_second_moment"], np.ones(5))
    for key, value in result["reference"].items():
        np.testing.assert_array_equal(result["executed"][key], value)
    # The optional intervention uses the endpoint model for its reference.
    endpoint_plan = repaired.make_endpoint_planner(
        local,
        7,
        direct_any_step=direct,
        controller=CONTROLLER,
        reference_mode="endpoint",
    )
    alternative = endpoint_plan(belief, current, noise, bank)
    assert bool(alternative["reference_feasible"])
    np.testing.assert_allclose(
        alternative["reference_sequence"][1:], np.tanh(1.0), atol=1e-7
    )


def test_controller_protocol_and_shape_changes_fail_closed(model):
    for key, value in [
        ("horizon", 4),
        ("action_sequence_particles", 4),
        ("action_sequence_objective_evaluations", 12),
        ("action_sequence_residual_limit", 0.2),
    ]:
        with pytest.raises(ValueError, match="unsupported endpoint controller"):
            repaired.make_endpoint_planner(
                model,
                7,
                direct_any_step=True,
                controller=dict(CONTROLLER, **{key: value}),
            )
    with pytest.raises(ValueError, match="reference_mode"):
        repaired.make_endpoint_planner(
            model,
            7,
            direct_any_step=True,
            controller=CONTROLLER,
            reference_mode="other",
        )
    planner = repaired.make_endpoint_planner(
        model, 7, direct_any_step=True, controller=CONTROLLER
    )
    belief, current, noise, bank = arrays(model)
    with pytest.raises(ValueError, match="noise bank"):
        planner(belief, current, noise[:4], bank)
    with pytest.raises(ValueError, match="standard_normal_proposals"):
        planner(belief, current, noise, bank[:1])


def _constant_actor(actor, value):
    result = jax.tree_util.tree_map(jnp.zeros_like, actor)
    result["output"]["bias"] = jnp.full_like(result["output"]["bias"], value)
    return result


def test_compiled_acceptance_and_final_executed_heldout_metrics(model, monkeypatch):
    cfg, rc = model["config"], model["rebrac_config"]
    policy, world = model["policies"][7], model["reward_world"]
    belief, current, _, _ = arrays(model)
    flow = FlowMPCConfig(horizon=5, particles=4, discount=rc.discount)
    noise = jax.random.normal(jax.random.PRNGKey(8), (4, 5, cfg.stochastic_dim))
    reference = _constant_actor(policy.actor, 0.0)
    start = _constant_actor(policy.actor, 0.5)
    proposal = _constant_actor(policy.actor, 1.0)
    acceptance_config = robust.HeldOutAcceptanceConfig(
        mode="held_out_noise", minimum_improvement=1e6
    )
    args = (
        proposal,
        start,
        reference,
        policy.critics,
        world,
        belief,
        current,
        noise,
        cfg,
        rc,
        flow,
        acceptance_config,
    )
    eager = robust.evaluate_flowmpc_held_out_acceptance(*args)
    traced = []
    original = robust.flowmpc_objective

    def counter(*args, **kwargs):
        traced.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(robust, "flowmpc_objective", counter)
    robust.jit_evaluate_flowmpc_held_out_acceptance.clear_cache()
    compiled = robust.jit_evaluate_flowmpc_held_out_acceptance(*args)
    jax.block_until_ready(compiled)
    assert len(traced) == 2
    for _ in range(3):
        jax.block_until_ready(robust.jit_evaluate_flowmpc_held_out_acceptance(*args))
    assert len(traced) == 2
    np.testing.assert_allclose(
        compiled.improvement, eager.improvement, rtol=2e-5, atol=2e-6
    )
    assert not bool(compiled.accepted)
    rejected = robust.apply_held_out_noise_fallback(
        proposal, start, reference, compiled, acceptance_config
    )
    final = robust.jit_backtrack_actor_proposal(
        reference,
        start,
        rejected,
        jnp.zeros((3, 3)),
        current,
        robust.ActionTrustRegionConfig(),
    )
    assert bool(final.used_reference_fallback)
    metrics = robust.jit_evaluate_executed_flowmpc_held_out_metrics(
        proposal,
        start,
        reference,
        final.actor,
        policy.critics,
        world,
        belief,
        current,
        noise,
        cfg,
        rc,
        flow,
    )
    for name, actor in [
        ("proposal", proposal),
        ("adaptation_start", start),
        ("reference", reference),
        ("executed", final.actor),
    ]:
        expected = flowmpc_objective(
            actor, policy.critics, world, belief, current, noise, cfg, rc, flow
        ).objective
        np.testing.assert_allclose(
            metrics[f"heldout_{name}_objective"], expected, rtol=2e-5, atol=2e-6
        )
    np.testing.assert_array_equal(
        metrics["heldout_executed_objective"], metrics["heldout_reference_objective"]
    )
    np.testing.assert_array_equal(
        metrics["heldout_executed_improvement"],
        metrics["heldout_executed_objective"]
        - metrics["heldout_adaptation_start_objective"],
    )


def test_live_a3_selects_with_jit_and_scores_after_final_projection():
    from dreamer_imf_compare import actor_gap_roadmap_study as roadmap

    source = inspect.getsource(roadmap._run_flowmpc_arm)
    assert "@jax.jit\n    def select_actor(" in source
    assert "acceptance = jit_evaluate_flowmpc_held_out_acceptance(" in source
    assert source.index("executed_constraint =") < source.index(
        "jit_evaluate_executed_flowmpc_held_out_metrics(", source.index("telemetry = {")
    )


def test_real_a3_controller_replays_and_reports_final_heldout_gain(model, monkeypatch):
    from dreamer_imf_compare import actor_gap_roadmap_study as roadmap
    from dreamer_imf_compare import dmc

    class TinyEnvironment:
        def __init__(self, *args, **kwargs):
            self.step_index = 0

        def reset(self):
            return np.zeros(3, np.float32)

        def step(self, action):
            self.step_index += 1
            return SimpleNamespace(
                observation=np.full(3, 0.1 * self.step_index, np.float32),
                reward=float(action[0]),
                continuation=1.0,
                is_last=self.step_index == 2,
            )

        def close(self):
            pass

    monkeypatch.setattr(dmc, "DMCAdapter", TinyEnvironment)
    monkeypatch.setattr(
        roadmap,
        "_flowmpc_config",
        lambda rc: FlowMPCConfig(horizon=5, particles=4, discount=rc.discount),
    )

    def run():
        return roadmap._run_flowmpc_arm(
            model["reward_world"],
            model["config"],
            model["policies"][7],
            model["rebrac_config"],
            world_seed=1,
            actor_seed=7,
            evaluation_seeds=[9],
            maximum_steps=2,
            trust=True,
            persistence="persistent",
            heldout_acceptance_enabled=True,
            anchor_observations=np.zeros((3, 3), np.float32),
            task="tiny",
        )

    returns, trace, timing = run()
    replay_returns, replay, _ = run()
    np.testing.assert_array_equal(returns, replay_returns)
    for key, value in trace.items():
        np.testing.assert_array_equal(value, replay[key])
        assert np.isfinite(value).all()
    np.testing.assert_allclose(
        trace["heldout_executed_improvement"],
        trace["heldout_executed_objective"]
        - trace["heldout_adaptation_start_objective"],
        atol=2e-8,
    )
    np.testing.assert_allclose(
        trace["heldout_executed_reference_improvement"],
        trace["heldout_executed_objective"] - trace["heldout_reference_objective"],
        atol=2e-8,
    )
    assert timing["timed_steps"] == 2
    assert np.asarray(trace["within_reference_budgets"]).all()
