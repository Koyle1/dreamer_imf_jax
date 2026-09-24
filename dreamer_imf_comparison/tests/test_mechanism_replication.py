"""Positive controls and fail-closed replication contract checks."""

from copy import deepcopy
import inspect
import json
from unittest.mock import patch

import numpy as np
import pytest

from dreamer_imf_compare import mechanism_replication as m
from dreamer_imf_compare.mechanism_replication_analysis import analyze


def protocol():
    return m.read(m.PROTOCOL)


def test_matrix_complete_and_paired():
    p = protocol()
    assert [len(m.cells(p, s)) for s in m.STAGES] == [3, 9, 72]
    rows = m.cells(p, "evaluation")
    assert (
        len(
            {
                (r["task"], r["world_model_seed"], r["actor_seed"], r["arm"])
                for r in rows
            }
        )
        == 72
    )
    for row in rows:
        dependency = m.cells(p, "training")[row["training_index"]]
        assert all(row[k] == dependency[k] for k in ("task", "world_model_seed"))
    assert not set(p["evaluation_environment_seeds"]) & set(
        p["preflight"]["evaluation_seeds"]
    )


def test_trace_zero_tolerance():
    a = {"actions": np.array([1], np.float32)}
    m.exact_trace(a, a)
    with pytest.raises(ValueError):
        m.exact_trace(a, {"actions": np.nextafter(a["actions"], np.float32(2))})
    with pytest.raises(ValueError):
        m.exact_trace(a, {"actions": a["actions"].astype(np.float64)})
    with pytest.raises(ValueError):
        m.exact_trace(a, {})
    assert not m.finite({"loss": np.nan})


def test_exclusive_publication(tmp_path):
    p = tmp_path / "result.json"
    m.publish(p, {"value": 1})
    with pytest.raises(FileExistsError):
        m.publish(p, {"value": 2})
    assert m.read(p) == {"value": 1}


def test_scheduler_completeness_and_failure():
    good = "12_0|COMPLETED|0:0|1|gres/gpu=1\n12_1|COMPLETED|0:0|1|gres/gpu=1\n"
    with patch.object(m.subprocess, "check_output", return_value=good):
        assert len(m.accounting("12", 2)["records"]) == 2
    for bad in (
        good.replace("12_1", "12_2"),
        good.replace("0:0", "1:0"),
        good.replace("COMPLETED", "RUNNING"),
    ):
        with patch.object(m.subprocess, "check_output", return_value=bad):
            with pytest.raises(RuntimeError):
                m.accounting("12", 2)


def test_analysis_pairing_and_negative_controls():
    p = protocol()
    p["uncertainty"]["resamples"] = 100
    rows = [
        dict(
            c,
            evaluation_environment_seeds=p["evaluation_environment_seeds"],
            episode_returns=[10.0 if c["arm"].startswith(("A3", "K0")) else 0.0] * 5,
        )
        for c in m.cells(p, "evaluation")
    ]
    result = analyze(rows, p)
    assert all(c["primary_effect"] == 10 for c in result["contrasts"].values())
    assert all(
        c["bootstrap_interval"] == [10, 10] for c in result["contrasts"].values()
    )
    for bad in (
        rows[:-1],
        rows + [rows[0]],
        [dict(rows[0], episode_returns=[1])] + rows[1:],
        [dict(rows[0], episode_returns=[float("nan")] * 5)] + rows[1:],
        [dict(rows[0], evaluation_environment_seeds=[1] * 5)] + rows[1:],
    ):
        with pytest.raises(ValueError):
            analyze(bad, p)


def test_historical_default_unchanged_and_task_parameterized():
    from dreamer_imf_compare import actor_gap_roadmap_study as roadmap

    for fn in (roadmap._run_flowmpc_arm, roadmap._run_endpoint_action_sequence_arm):
        assert inspect.signature(fn).parameters["task"].default == "dmc_reacher_easy"
        body = inspect.getsource(fn).split(") ->", 1)[1]
        assert "DMCAdapter(task," in body
        assert "TASK," not in body


def test_artifact_digest_tamper(tmp_path):
    manifest = {"source_commit": "abc"}
    m.publish(tmp_path / "result.json", {"value": 1})
    with patch.object(m, "manifest", return_value=manifest):
        m.marker(tmp_path, tmp_path, {"index": 0}, ["result.json"])
        m.verify_files(tmp_path, tmp_path)
        # Test fixture corruption, not production evidence.
        (tmp_path / "result.json").write_text(json.dumps({"value": 2}))
        with pytest.raises(ValueError):
            m.verify_files(tmp_path, tmp_path)


@pytest.mark.parametrize(
    "task,shape,action",
    [
        ("dmc_reacher_hard", 6, 2),
        ("dmc_cartpole_swingup", 5, 1),
        ("dmc_finger_spin", 9, 2),
    ],
)
def test_task_collection_and_heldout_partition(task, shape, action):
    settings = dict(
        protocol()["data"], episodes=4, steps_per_episode=32, train_episodes=3
    )
    data = m.collect(task, 431, settings)
    again = m.collect(task, 431, settings)
    assert data["observations"].shape == (4, 33, shape)
    assert data["actions"].shape == (4, 33, action)
    assert set(data["train_episode_ids"]) == {0, 1, 2}
    assert set(data["test_episode_ids"]) == {3}
    m.exact_trace(data, again)


def test_training_numerical_smoke(tmp_path, monkeypatch):
    """Small CPU integration; full-sized every-task GPU gates remain mandatory."""
    p = deepcopy(protocol())
    for k in ("deterministic_dim", "embedding_dim", "hidden_dim", "stochastic_dim"):
        p["world_model_template"][k] = 8
    p["data"]["batch_size"] = 2
    monkeypatch.setattr(m, "runtime", lambda: {"test_only": "cpu"})
    result = m.train(p, m.cells(p, "preflight")[0], tmp_path / "train", preflight=True)
    assert result["reward_optimizer_updates"] == 2
    assert result["policies"][541]["state_updates"] == 4
    assert all(e["optimizer_final_step"] == 2 for e in result["endpoints"].values())
    model = m.load_checkpoint(tmp_path / "train")
    m.verify_training(p, tmp_path / "train", preflight=True)
    assert model["config"].observation_dim == 6
    from dreamer_imf_compare import actor_gap_roadmap_study as roadmap

    monkeypatch.setattr(roadmap, "FLOWMPC_PARTICLES", 4)
    monkeypatch.setattr(roadmap, "ACTION_SEQUENCE_PARTICLES", 4)
    p["preflight"]["evaluation_steps"] = 2
    for arm in p["arms"]:
        core, trace, _ = m.evaluate(
            p,
            dict(
                task="dmc_reacher_hard", world_model_seed=431, actor_seed=541, arm=arm
            ),
            tmp_path / "train",
            preflight=True,
        )
        assert len(core["episode_returns"]) == 1
        assert m.finite(trace)
