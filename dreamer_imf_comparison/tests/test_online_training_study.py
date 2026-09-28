"""Synthetic regression oracles for the bounded online runner."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from dreamer_imf_compare import online_training_study as study


@pytest.fixture
def protocol():
    return study.m.read(study.PROTOCOL)


def records(protocol):
    result = []
    for cell in study.cells(protocol, "evaluation"):
        i = protocol["world_model_seeds"].index(cell["world_model_seed"])
        base = 10.0 * (i + 1)
        gain = {"F": 0, "P": [2, -1, 5][i], "M": [4, 3, 2][i]}[cell["arm"]]
        value = base + gain * cell["steps"] / protocol["training_steps"]
        result.append(
            dict(
                cell=cell,
                episode_returns=[value + x for x in [-2, -1, 0, 1, 2]],
                evaluation_seeds=protocol["evaluation_seeds"],
            )
        )
    return result


def test_cell_matrix_and_training_pair_indices(protocol):
    assert protocol["execution"]["time_limits"] == {
        "preflight": "01:00:00",
        "training": "24:00:00",
        "evaluation": "02:00:00",
    }
    assert len(study.cells(protocol, "preflight")) == 1
    training = study.cells(protocol, "training")
    evaluation = study.cells(protocol, "evaluation")
    assert len(training) == 6 and len(evaluation) == 33
    assert [c["index"] for c in evaluation] == list(range(33))
    assert {c["arm"] for c in training} == {"P", "M"}
    for c in evaluation:
        if c["arm"] == "F":
            assert c["steps"] == 0
        else:
            t = training[study.training_index(c)]
            assert (t["world_model_seed"], t["arm"]) == (
                c["world_model_seed"],
                c["arm"],
            )
    with pytest.raises(ValueError, match="unknown stage"):
        study.cells(protocol, "not-a-stage")


def test_paired_world_seed_contrasts_and_learning_area(protocol):
    data = records(protocol)
    summary = study.summarize(data[::-1], protocol)
    assert summary["contrasts"]["P_minus_F"]["world_seed_deltas"] == [2, -1, 5]
    assert summary["contrasts"]["M_minus_F"]["world_seed_deltas"] == [4, 3, 2]
    assert summary["contrasts"]["M_minus_P"]["world_seed_deltas"] == [2, 4, -3]
    assert summary["contrasts"]["P_minus_F"]["mean"] == 2
    assert summary["contrasts"]["P_minus_F"]["favorable_fraction"] == 2 / 3
    assert "three paired" in summary["statistical_unit"]
    np.testing.assert_allclose(
        summary["normalized_learning_curve_area"]["P"], [11, 19.5, 32.5]
    )
    for arm in protocol["arms"]:
        assert summary["curves"][arm]["0"] == [10, 20, 30]
    for bad in (data[:-1], data[:-1] + [data[0]], data + [data[0]]):
        with pytest.raises(ValueError, match="matrix"):
            study.summarize(bad, protocol)


def test_training_seeds_are_reproducible_disjoint_and_arm_paired(protocol, monkeypatch):
    values = [
        study.training_seed(protocol, w, ep)
        for w in protocol["world_model_seeds"]
        for ep in range(1, 101)
    ]
    assert len(set(values)) == 300
    assert not set(values) & set(
        protocol["evaluation_seeds"] + [protocol["preflight_seed"]]
    )
    assert values[0] == study.training_seed(protocol, 431, 1)
    for collision in protocol["evaluation_seeds"] + [protocol["preflight_seed"]]:
        monkeypatch.setattr(study.b, "derive_seed", lambda *args: collision)
        with pytest.raises(ValueError, match="collision"):
            study.training_seed(protocol, 431, 1)


def test_seal_binding_inventory_digest_and_exclusive_writes(tmp_path):
    value = dict(source_commit="abc", protocol={"steps": 2})
    context = dict(stage="test")
    study.m.publish(tmp_path / "result.json", {"value": 1})
    mark = study.seal(tmp_path, value, context)
    assert study.verify_directory(tmp_path, value, context) == mark
    with pytest.raises(FileExistsError):
        study.seal(tmp_path, value, context)
    with pytest.raises(ValueError, match="binding"):
        study.verify_directory(tmp_path, dict(value, source_commit="other"), context)
    with pytest.raises(ValueError, match="binding"):
        study.verify_directory(tmp_path, value, {"stage": "other"})
    study.m.publish(tmp_path / "extra.json", {"extra": True})
    with pytest.raises(ValueError, match="inventory"):
        study.verify_directory(tmp_path, value, context)
    (tmp_path / "extra.json").unlink()
    (tmp_path / "result.json").write_text('{"value":2}')
    with pytest.raises(ValueError, match="digest"):
        study.verify_directory(tmp_path, value, context)


def test_parameter_digests_do_not_confuse_world_change_with_reward_update():
    policy = SimpleNamespace(
        actor={"x": np.array([1.0])}, critics={"x": np.array([2.0])}
    )
    model = dict(
        world={"x": np.array([3.0])},
        reward_world={
            "x": np.array([3.0]),
            "reward_transition": {"x": np.array([4.0])},
        },
        policies={541: policy},
    )
    baseline = study.parameter_digests(model, 541)
    changed = deepcopy(model)
    changed["world"]["x"] += 1
    changed["reward_world"]["x"] += 1
    after = study.parameter_digests(changed, 541)
    assert after["world"] != baseline["world"]
    assert after["reward"] == baseline["reward"]
    changed["reward_world"]["reward_transition"]["x"] += 1
    assert study.parameter_digests(changed, 541)["reward"] != baseline["reward"]


def test_parameter_change_scopes():
    before = dict(world="w", reward="r", actor="a", critic="c")
    study.validate_changes(before, before, "F")
    policy = dict(before, actor="new-a", critic="new-c")
    study.validate_changes(before, policy, "P")
    full = dict(policy, world="new-w", reward="new-r")
    study.validate_changes(before, full, "M")
    for after, arm in [
        (policy, "F"),
        (before, "P"),
        (full, "P"),
        (policy, "M"),
        (dict(full, reward="r"), "M"),
        (dict(policy, critic="c"), "P"),
    ]:
        with pytest.raises(ValueError):
            study.validate_changes(before, after, arm)


def episode():
    return dict(
        observations=np.zeros((1, 3, 3), np.float32),
        actions=np.zeros((1, 3, 2), np.float32),
        rewards=np.array([[0, 0.2, 0.3]]),
        continuations=np.ones((1, 3)),
        is_first=np.array([[True, False, False]]),
        is_last=np.array([[False, False, True]]),
    )


@pytest.mark.parametrize(
    "field,index,value",
    [
        ("observations", (0, 1, 0), np.nan),
        ("actions", (0, 0, 0), 0.1),
        ("actions", (0, 1, 0), 1.1),
        ("rewards", (0, 0), 1),
        ("continuations", (0, 1), -0.1),
        ("is_first", (0, 1), True),
        ("is_first", (0, 0), False),
        ("is_last", (0, 1), True),
        ("is_last", (0, 2), False),
    ],
)
def test_episode_rejects_invalid_alignment_boundaries_support(field, index, value):
    good = episode()
    study.validate_episode(good, 2, require_native_end=True)
    bad = deepcopy(good)
    bad[field][index] = value
    with pytest.raises(ValueError):
        study.validate_episode(bad, 2, require_native_end=True)


def test_episode_truncation_and_float64_rewards(tmp_path):
    ep = episode()
    ep["is_last"][0, -1] = False
    ep["rewards"][0, -1] = 0.123456789012345
    study.validate_episode(ep, 2, require_native_end=False)
    study.write_npz(tmp_path / "episode.npz", ep)
    loaded = study.b.load_npz(tmp_path / "episode.npz")
    assert loaded["rewards"].dtype == np.float64
    assert loaded["rewards"][0, -1] == 0.123456789012345
    with pytest.raises(FileExistsError):
        study.write_npz(tmp_path / "episode.npz", ep)
    with pytest.raises(ValueError, match="length"):
        study.validate_episode(ep, 3, require_native_end=False)


def trace_for(ep, seed):
    trace = {
        k: ep[k][:, 1:].copy()
        for k in ("actions", "rewards", "continuations", "is_last")
    }
    trace.update(
        observations=ep["observations"][:, :-1].copy(),
        lengths=np.array([2]),
        evaluation_seeds=np.array([seed]),
    )
    return trace


@pytest.mark.parametrize(
    "field", ["actions", "rewards", "observations", "lengths", "evaluation_seeds"]
)
def test_trace_alignment_rejects_valid_shaped_but_wrong_evidence(field):
    ep = episode()
    trace = trace_for(ep, 123)
    study.validate_alignment(ep, trace, 123)
    trace[field].flat[0] += 1
    with pytest.raises(ValueError):
        study.validate_alignment(ep, trace, 123)


def tiny_model():
    import jax
    from imf_dreamer_jax.config import DreamerConfig
    from imf_dreamer_jax.flowmpc import ReBRACConfig, init_rebrac_state

    cfg = DreamerConfig(
        observation_shape=(3,),
        action_dim=2,
        deterministic_dim=4,
        stochastic_dim=2,
        embedding_dim=4,
        hidden_dim=8,
    )
    rc = ReBRACConfig(state_dim=3, action_dim=2, hidden_dim=8)
    return dict(
        config=cfg,
        rebrac_config=rc,
        world={"x": np.array([3.0])},
        reward_world={
            "x": np.array([3.0]),
            "reward_transition": {"x": np.array([4.0])},
        },
        policies={541: init_rebrac_state(jax.random.key(3), rc)},
    )


def test_finite_learner_accepts_config_dataclasses_but_rejects_nonfinite_parameters():
    from dreamer_imf_compare.online_learning import initialize

    state = initialize(tiny_model(), 541)
    assert study.finite_learner(state)
    state["world"] = {"x": np.array([np.nan])}
    assert not study.finite_learner(state)


@pytest.mark.parametrize(
    "corruption", [None, "clock", "snapshot", "trace", "ancestry", "budget", "parent"]
)
def test_epoch_semantics_rechecked_after_valid_artifact_seal(
    tmp_path, protocol, corruption
):
    import jax
    from dreamer_imf_compare.online_learning import initialize, export_model

    protocol.update(
        episode_steps=2,
        training_steps=2,
        evaluation_steps=[0, 2],
        policy_updates_per_episode=1,
    )
    value = dict(protocol=protocol, source_commit="test-source")
    cell = study.cells(protocol, "training")[0]
    model = tiny_model()
    state = initialize(model, 541)
    policy = state["policy"]
    state["policy"] = policy._replace(
        actor=jax.tree_util.tree_map(lambda x: x + 0.01, policy.actor),
        critics=jax.tree_util.tree_map(lambda x: x + 0.01, policy.critics),
        step=policy.step + 1,
        actor_optimizer=policy.actor_optimizer._replace(
            step=policy.actor_optimizer.step + 1
        ),
        critic_optimizer=policy.critic_optimizer._replace(
            step=policy.critic_optimizer.step + 1
        ),
    )
    current = export_model(state, model, 541)
    ep = episode()
    seed = study.training_seed(protocol, 431, 1)
    trace = trace_for(ep, seed)
    result = dict(
        context=study.epoch_context(cell, 1),
        parent_sha256="parent",
        steps=2,
        environment_seed=seed,
        before=study.parameter_digests(model, 541),
        after=study.parameter_digests(current, 541),
        return_=0.5,
        update_metrics=dict(world_updates=0, reward_updates=0, policy_updates=1),
    )
    if corruption == "clock":
        state["world_optimizer"] = state["world_optimizer"]._replace(step=np.array(1))
    elif corruption == "snapshot":
        current = dict(current, unexpected="not-exported")
    elif corruption == "trace":
        trace["actions"][0, 0, 0] = 0.1
    elif corruption == "ancestry":
        result["before"]["actor"] = "unrelated"
    elif corruption == "budget":
        result["update_metrics"]["policy_updates"] = 2
    elif corruption == "parent":
        result["parent_sha256"] = "unrelated"
    directory = study.epoch_dir(tmp_path, cell, 1)
    directory.mkdir(parents=True)
    study.write_pickle(directory / "learner.pkl", state)
    study.write_pickle(directory / "model.pkl", current)
    study.write_npz(directory / "episode.npz", ep)
    study.write_npz(directory / "trace.npz", trace)
    study.m.publish(directory / "result.json", result)
    study.seal(directory, value, study.epoch_context(cell, 1))
    if corruption is None:
        study.verify_epoch(tmp_path, value, cell, 1, "parent", base_model=model)
    else:
        with pytest.raises(ValueError):
            study.verify_epoch(tmp_path, value, cell, 1, "parent", base_model=model)
