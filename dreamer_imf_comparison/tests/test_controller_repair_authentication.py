"""Independent adversarial controls for the new repair artifact authenticator.

Every mutation starts from the orchestrator's own passing fixture, then rehashes
all affected files/receipts/seals. Rejection must therefore be semantic, not just
a stale checksum. No production evidence or orchestration source is changed.
"""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from dreamer_imf_compare import controller_repair_study as s
from dreamer_imf_comparison.tests.test_controller_repair_study import (
    fixture,
    save_fixture,
)


def rebind(out, mark, *, core=None, trace=None):
    if core is None:
        core = s.m.read(out / "result.json")
    if trace is not None:
        np.savez(out / "trace.npz", **trace)
        core["trace_sha256"] = s.b.array_sha256(trace)
    save_fixture(out / "result.json", core)
    for role in s.ROLES:
        path = out / f"{role}-receipt.json"
        receipt = s.m.read(path)
        receipt.update(core_sha256=s.m.digest(core), trace_sha256=core["trace_sha256"])
        save_fixture(path, receipt)
        seal = s.m.read(out / f"{role}-seal.json")
        seal["receipt_sha256"] = s.b.file_sha256(path)
        save_fixture(out / f"{role}-seal.json", seal)
    mark["files"] = {name: s.b.file_sha256(out / name) for name in s.FILES}
    save_fixture(out / "verified.json", mark)


@pytest.mark.parametrize("returns", [[False, False, False], ["0", "0", "0"]])
def test_rehashed_boolean_and_string_episode_returns_fail(tmp_path, returns):
    value, out, mark = fixture(tmp_path)
    with patch.object(s, "manifest", return_value=value):
        s.verify_cell(tmp_path, "evaluation", 0)
        core = s.m.read(out / "result.json")
        core["episode_returns"] = returns
        rebind(out, mark, core=core)
        with pytest.raises(ValueError):
            s.verify_cell(tmp_path, "evaluation", 0)


@pytest.mark.parametrize(
    "lengths",
    [np.array([1.5] * 3, np.float32), np.ones(3, bool), np.ones(3, np.float64)],
)
def test_rehashed_noninteger_length_dtype_fails(tmp_path, lengths):
    value, out, mark = fixture(tmp_path)
    with patch.object(s, "manifest", return_value=value):
        s.verify_cell(tmp_path, "evaluation", 0)
        trace = s.b.load_npz(out / "trace.npz")
        trace["lengths"] = lengths
        rebind(out, mark, trace=trace)
        with pytest.raises(ValueError):
            s.verify_cell(tmp_path, "evaluation", 0)


@pytest.mark.parametrize("change", ["extra", "missing", "boolean_index"])
def test_rehashed_result_schema_and_identity_are_exact(tmp_path, change):
    value, out, mark = fixture(tmp_path)
    with patch.object(s, "manifest", return_value=value):
        s.verify_cell(tmp_path, "evaluation", 0)
        core = s.m.read(out / "result.json")
        if change == "extra":
            core["unregistered_field"] = "must not pass"
        elif change == "missing":
            del core["telemetry"]
        else:
            core["index"] = False
        rebind(out, mark, core=core)
        with pytest.raises(ValueError):
            s.verify_cell(tmp_path, "evaluation", 0)


@pytest.mark.parametrize("role", s.ROLES)
@pytest.mark.parametrize("change", ["extra", "missing", "boolean_wall_seconds"])
def test_rehashed_receipt_schema_is_exact(tmp_path, role, change):
    value, out, mark = fixture(tmp_path)
    with patch.object(s, "manifest", return_value=value):
        s.verify_cell(tmp_path, "evaluation", 0)
        path = out / f"{role}-receipt.json"
        receipt = s.m.read(path)
        if change == "extra":
            receipt["unregistered_field"] = 1
        elif change == "missing":
            del receipt["timing"]
        else:
            receipt["wall_seconds"] = True
        save_fixture(path, receipt)
        rebind(out, mark)
        with pytest.raises(ValueError):
            s.verify_cell(tmp_path, "evaluation", 0)


@pytest.mark.parametrize("mode", ["unknown", True, "endpoint"])
def test_result_reference_modes_must_match_selection(tmp_path, mode):
    value, out, mark = fixture(tmp_path)
    with patch.object(s, "manifest", return_value=value):
        s.verify_cell(tmp_path, "evaluation", 0)
        core = s.m.read(out / "result.json")
        core["reference_modes"]["recursive"] = mode
        rebind(out, mark, core=core)
        with pytest.raises(ValueError):
            s.verify_cell(tmp_path, "evaluation", 0)


def test_unknown_reference_mode_cannot_be_rehashed_into_selection_and_result(tmp_path):
    value, out, mark = fixture(tmp_path)
    with patch.object(s, "manifest", return_value=value):
        s.verify_cell(tmp_path, "evaluation", 0)
        core = s.m.read(out / "result.json")
        core["reference_modes"]["recursive"] = "unknown"
        selected = s.m.read(tmp_path / "selection.json")
        selected["reference_modes"] = core["reference_modes"]
        save_fixture(tmp_path / "selection.json", selected)
        rebind(out, mark, core=core)
        with pytest.raises(ValueError):
            s.verify_cell(tmp_path, "evaluation", 0)


@pytest.mark.parametrize(
    "change", ["digest", "empty_files", "missing_cell", "wrong_cell"]
)
def test_manifest_dependency_index_rebound_to_pinned_dependency(tmp_path, change):
    p = s.m.read(s.PROTOCOL)
    entry = dict(
        cell=dict(
            index=0, task=p["tasks"][0], world_model_seed=p["world_model_seeds"][0]
        ),
        marker_sha256="a" * 64,
        files={name: "b" * 64 for name in s.TRAIN_FILES},
    )
    expected = {
        str(c["index"]): dict(
            deepcopy(entry),
            cell={name: c[name] for name in ("index", "task", "world_model_seed")},
        )
        for c in s.cells(p, "diagnostics")
    }
    value = dict(
        schema="imf-controller-repair-v1",
        source_commit="f" * 40,
        source_root=str(s.SOURCE),
        protocol=p,
        protocol_sha256=s.m.digest(p),
        dependency_root=str(tmp_path / "dependency"),
        dependency_index=deepcopy(expected),
        created_unix=1.0,
    )
    s.m.publish(tmp_path / "manifest.json", value)
    with patch.object(
        s.m, "clean_commit", return_value=value["source_commit"]
    ), patch.object(s, "dependency_index", return_value=expected):
        assert s.manifest(tmp_path) == value
        if change == "digest":
            value["dependency_index"]["0"]["files"]["checkpoint.pkl"] = "c" * 64
        elif change == "empty_files":
            value["dependency_index"]["0"]["files"] = {}
        elif change == "missing_cell":
            value["dependency_index"] = {}
        else:
            value["dependency_index"]["0"]["cell"]["world_model_seed"] = 999
        save_fixture(tmp_path / "manifest.json", value)
        with pytest.raises(ValueError):
            s.manifest(tmp_path)
