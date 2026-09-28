"""Extraction and publication controls for the bounded trace-only pilot."""

from copy import deepcopy

import numpy as np
import pytest

from dreamer_imf_compare.planning_value_pilot import (
    extract_training,
    publish,
    read_json,
    validate_output_snapshot,
)


def trace_fixture():
    # Two calibration plans, an executed duplicate, and a held-out plan.
    horizon, maximum, count = 2, 8, 5
    rewards = np.zeros((count, maximum))
    rewards[:, 3] = 0.5
    c = np.ones_like(rewards)
    state = np.arange(count * 3 * 2, dtype=np.float32).reshape(count, 3, 2)
    state[1] = state[0]
    actions = np.zeros((count, horizon, 1), np.float32)
    actions[2:] = 0.1
    return dict(
        baseline_split_id=np.array([0, 1]),
        baseline_length=np.array([8, 8]),
        baseline_native_episode_end=np.array([True, True]),
        baseline_rewards=np.array([[0.1] * 8, [0.9] * 8]),
        baseline_continuations=np.ones((2, 8)),
        baseline_observations=np.arange(36, dtype=np.float32).reshape(2, 9, 2),
        baseline_environment_seed=np.array([76001, 76003]),
        split_id=np.array([0, 0, 0, 0, 1]),
        reference_mode_id=np.zeros(count, int),
        plan_kind=np.array([0, 0, 1, 2, 1]),
        snapshot_id=np.zeros(count, int),
        action_sequence=actions,
        real_stage_observations=state,
        real_terminal=np.full(count, 0.5 * 0.99**3),
        real_length=np.full(count, 7),
        stage_mask=np.ones((count, 2), bool),
        real_continuations=c,
        real_rewards=rewards,
        episode_step=np.ones(count, int),
    )


def test_extract_deduplicates_across_families_and_never_fits_validation():
    trace = trace_fixture()
    a = extract_training(trace, horizon=2, maximum=8)
    changed = deepcopy(trace)
    changed["baseline_observations"][1] = np.nan
    changed["baseline_rewards"][1] = np.nan
    changed["real_stage_observations"][4] = np.nan
    changed["real_terminal"][4] = np.nan
    b = extract_training(changed, horizon=2, maximum=8)
    for key in ("observations", "remaining", "returns", "weights"):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["support"] == b["support"]
    assert a["support"]["unique_candidate_endpoints"] == 2
    assert a["endpoint_trace_indices"] == [0, 2]
    assert a["observations"].shape == (10, 2)
    np.testing.assert_array_equal(
        a["observations"][-2:], trace["real_stage_observations"][[0, 2], 2]
    )
    np.testing.assert_array_equal(a["remaining"][-2:], [5, 5])
    np.testing.assert_allclose(a["returns"][-2:], 0.5 * 0.99)
    assert a["weights"][:8].sum() == 0.5
    assert a["weights"][8:].sum() == 0.5


@pytest.mark.parametrize("kind", ["duplicate", "arithmetic", "continuation", "timeout"])
def test_extract_rejects_invalid_mc_evidence(kind):
    trace = trace_fixture()
    if kind == "duplicate":
        trace["real_terminal"][1] += 0.01
    elif kind == "arithmetic":
        trace["real_terminal"] += 0.01
    elif kind == "continuation":
        trace["real_continuations"][0, 0] = 0
    else:
        trace["baseline_native_episode_end"][0] = False
    with pytest.raises(ValueError):
        extract_training(trace, horizon=2, maximum=8)


def test_publish_is_exclusive_and_binds_every_file(tmp_path):
    import hashlib

    out = tmp_path / "fresh"
    publish(out, {"a": 1}, [{"weight": 2}], [1.5], {"commit": "fake"})
    manifest = read_json(out / "manifest.json")
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        publish(out, {}, [], [], {})
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    validate_output_snapshot(out, manifest, before["manifest.json"])
    (out / "runtime.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_output_snapshot(out, manifest, before["manifest.json"])
    (out / "runtime.json").write_bytes(before["runtime.json"])
    (out / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        validate_output_snapshot(out, manifest, before["manifest.json"])


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'])
def test_json_rejects_ambiguous_or_nonfinite_evidence(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_json(path)
