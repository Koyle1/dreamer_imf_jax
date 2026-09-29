"""Budget, fresh-data provenance and artifact-verifier regression oracles."""

from copy import deepcopy
import json

import numpy as np
import pytest

from dreamer_imf_compare import scratch_online_study as s


@pytest.fixture
def p():
    return s.m.read(s.PROTOCOL)


@pytest.fixture
def tiny(p):
    p["world_model_template"].update(
        deterministic_dim=5, stochastic_dim=3, embedding_dim=4, hidden_dim=8, burn_in=2
    )
    p["rebrac_config"]["hidden_dim"] = 8
    p["reward_hidden_dim"] = 8
    p["anchor_count"] = 3
    p["controller"].update(horizon=1, flowmpc_particles=2)
    p["preflight_decisions"] = 8
    p["sequence_length"] = 4
    p["batch_size"] = 2
    return p


def test_budget_grid_and_seed_isolation(p):
    s.validate_protocol(p)
    assert len(s.cells(p, "training")) == 3
    assert len(s.cells(p, "evaluation")) == 15
    assert len(s.cells(p, "preflight")) == 1
    assert p["training_native_steps"] // p["action_repeat"] == 250000
    assert 5000 + 495 * 1000 == p["training_native_steps"]
    assert 5 * 250 + 495 * 250 == 125000
    for c in s.cells(p, "evaluation"):
        assert (
            s.cells(p, "training")[c["training_index"]]["world_model_seed"]
            == c["world_model_seed"]
        )
    with pytest.raises(ValueError):
        s.cells(p, "bad")


@pytest.mark.parametrize(
    "key,value",
    [
        ("action_repeat", 1),
        ("training_native_steps", 100000),
        ("prefill_native_steps", 0),
        ("evaluation_native_steps", [500000]),
        ("world_model_seeds", [431, 431, 439]),
    ],
)
def test_protocol_changes_rejected(p, key, value):
    p[key] = value
    with pytest.raises(ValueError):
        s.validate_protocol(p)


def records(p):
    return [
        dict(
            cell=c,
            evaluation_seeds=p["evaluation_seeds"],
            episode_returns=[c["training_index"] * 10 + c["native_steps"] / 100000] * 5,
        )
        for c in s.cells(p, "evaluation")
    ]


def test_published_reference_is_descriptive_and_all_seeds_reported(p):
    r = s.summarize(records(p)[::-1], p)
    assert r["final_seed_returns"] == [5, 15, 25]
    assert r["final_mean"] == 15
    assert r["final_mean_minus_published"] == 15 - 938
    assert r["final_minus_published_by_seed"] == [5 - 938, 15 - 938, 25 - 938]
    assert r["change_100k_to_500k"] == [4, 4, 4]
    assert r["fraction_seeds_above_published"] == 0
    assert "independently initialized" in r["statistical_unit"]
    for bad in [records(p)[:-1], records(p) + records(p)[:1]]:
        with pytest.raises(ValueError):
            s.summarize(bad, p)
    bad = records(p)
    bad[0]["episode_returns"][0] = float("nan")
    with pytest.raises(ValueError):
        s.summarize(bad, p)


def test_register_fresh_no_dependency_and_manifest_tamper(tmp_path, p, monkeypatch):
    monkeypatch.setattr(s.m, "clean_commit", lambda: "exact")
    root = tmp_path / "study"
    s.register(root)
    assert s.manifest(root)["dependencies"] == []
    with pytest.raises(FileExistsError):
        s.register(root)
    path = root / "manifest.json"
    v = json.loads(path.read_text())
    v["dependencies"] = ["old"]
    path.write_text(json.dumps(v))
    with pytest.raises(ValueError, match="dependency"):
        s.manifest(root)


def test_bootstrap_reconstructs_untrained_parameters_and_counts_prefill(tmp_path, tiny):
    value = dict(protocol=tiny, source_commit="test", dependencies=[])
    cell = s.cells(tiny, "training")[0]
    model, episodes, parent = s.bootstrap(tmp_path, value, cell)
    assert len(episodes) == 5 and episodes[0]["rewards"].shape == (1, 501)
    assert len(parent) == 64
    model2, episodes2, parent2 = s.verify_bootstrap(tmp_path, value, cell)
    assert s.u.snapshot_digest(model) == s.u.snapshot_digest(model2)
    assert parent2 == parent
    np.testing.assert_array_equal(episodes[0]["actions"], episodes2[0]["actions"])
    from dreamer_imf_compare.online_learning import initialize

    s.clocks(initialize(model, tiny["actor_seed"]), 0)
    with pytest.raises(ValueError, match="clock"):
        s.clocks(initialize(model, tiny["actor_seed"]), 1)
    out = tmp_path / "training/000/bootstrap"
    data = s.b.load_npz(out / "episodes.npz")
    data["actions"][0, 1, 0] += 0.01
    with (out / "episodes.npz").open("wb") as f:
        np.savez_compressed(f, **data)
    with pytest.raises(ValueError, match="digest"):
        s.verify_bootstrap(tmp_path, value, cell)


def test_native_count_corruption_rejected(tmp_path, tiny):
    from dreamer_imf_compare.online_collector import make_collector

    model = s.fresh_model(tiny, 431)
    c = make_collector(model["config"], model["rebrac_config"], tiny["controller"])
    ep, tr, _ = c.rollout(
        {"task": tiny["task"]},
        541,
        431,
        29,
        maximum_steps=3,
        action_repeat=2,
        training=True,
        random_policy=True,
    )
    s.validate_collection(ep, tr, 29, 3, 6, complete=False)
    bad = deepcopy(tr)
    bad["native_steps"][0, 0] = 1
    with pytest.raises(ValueError, match="budget"):
        s.validate_collection(ep, bad, 29, 3, 6, complete=False)
    bad = deepcopy(tr)
    bad["native_steps"] = bad["native_steps"].astype(float)
    with pytest.raises(ValueError, match="type"):
        s.validate_collection(ep, bad, 29, 3, 6, complete=False)


def test_real_small_preflight_learning_and_verifier(tmp_path, tiny, monkeypatch):
    value = dict(protocol=tiny, source_commit="test", dependencies=[])
    monkeypatch.setattr(s, "manifest", lambda root: value)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache))

    def local_readers(root, value, stage, index):
        # CPU integration checks artifacts; actual process/cache replay is a GPU gate.
        out = root / stage / f"{index:03d}"
        core, arr, metrics = s.compute_evaluation(root, value, stage, index)
        s.m.publish(out / "result.json", core)
        s.u.write_npz(out / "trace.npz", arr)
        for i, mode in enumerate(s.MODES):
            s.m.publish(
                out / f"{mode}.json",
                dict(
                    pid=i + 1,
                    cache="same",
                    core_sha256=s.m.digest(core),
                    metrics=metrics,
                    wall_seconds=0,
                ),
            )
            s.m.publish(
                out / f"{mode}-seal.json",
                dict(
                    cache="same", receipt_sha256=s.u.file_digest(out / f"{mode}.json")
                ),
            )
        s.verify_readers(root, value, stage, index)
        s.u.seal(out, value, s.context(stage, s.cells(tiny, stage)[index]))

    monkeypatch.setattr(s, "readers", local_readers)
    s.preflight(tmp_path)
    s.verify_preflight(tmp_path, value)
    out = tmp_path / "preflight/000"
    assert s.m.read(out / "inputs.json")["metrics"]["offline_fraction"] == 0
    r = s.m.read(out / "replay.json")
    r["pid"] = 1
    (out / "replay.json").write_text(json.dumps(r))
    (out / "replay-seal.json").write_text(
        json.dumps(
            dict(cache="same", receipt_sha256=s.u.file_digest(out / "replay.json"))
        )
    )
    with pytest.raises(ValueError, match="process"):
        s.verify_readers(tmp_path, value, "preflight", 0)


@pytest.mark.parametrize(
    "corruption", [None, "offline", "clock", "budget", "ancestry", "snapshot"]
)
def test_epoch_authentication_rejects_semantic_corruption(tmp_path, tiny, corruption):
    from dreamer_imf_compare.online_learning import initialize, update, export_model

    tiny["updates_per_episode"] = 1
    tiny["evaluation_native_steps"] = [5000]
    value = dict(protocol=tiny, source_commit="test", dependencies=[])
    cell = s.cells(tiny, "training")[0]
    base, episodes, parent = s.bootstrap(tmp_path, value, cell)
    state, metrics = update(
        initialize(base, 541),
        None,
        episodes,
        full_model=True,
        seed=31,
        world_updates=5,
        policy_updates=5,
        batch_size=2,
        sequence_length=4,
    )
    model = export_model(state, base, 541)
    if corruption == "clock":
        state["world_optimizer"] = state["world_optimizer"]._replace(step=0)
    out = s.epoch_dir(tmp_path, cell, 5)
    out.mkdir()
    s.u.write_pickle(out / "learner.pkl", state)
    s.u.write_pickle(out / "model.pkl", base if corruption == "snapshot" else model)
    if corruption == "offline":
        metrics["offline_fraction"] = 0.5
    result = dict(
        cell=cell,
        episode=5,
        native_steps=4999 if corruption == "budget" else 5000,
        decision_steps=2500,
        parent_sha256="wrong" if corruption == "ancestry" else parent,
        before=s.u.parameter_digests(base, 541),
        after=s.u.parameter_digests(model, 541),
        update_metrics=metrics,
    )
    s.m.publish(out / "result.json", result)
    s.u.seal(out, value, s.context("epoch", cell, 5))
    if corruption is None:
        s.verify_epoch(tmp_path, value, cell, 5, parent, base)
    else:
        with pytest.raises(ValueError):
            s.verify_epoch(tmp_path, value, cell, 5, parent, base)


def test_launch_duplicate_intent_is_never_resubmitted(tmp_path, p, monkeypatch):
    value = dict(protocol=p, source_commit="test", dependencies=[])
    monkeypatch.setattr(s, "manifest", lambda root: value)
    (tmp_path / "submissions").mkdir()
    s.m.publish(tmp_path / "submissions/preflight.intent.json", {"uncertain": True})
    monkeypatch.setattr(
        s.subprocess, "check_output", lambda *a, **k: pytest.fail("resubmitted")
    )
    with pytest.raises(FileExistsError, match="never duplicate"):
        s.launch(tmp_path, "preflight")
