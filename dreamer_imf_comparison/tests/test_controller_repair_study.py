"""Independent orchestration/selection negative controls; no cluster mutation."""

from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest

from dreamer_imf_compare import controller_repair_study as s
from dreamer_imf_comparison.tests import test_repair_diagnostics as diagnostic_tests

tiny_checkpoint = diagnostic_tests.tiny_checkpoint


def protocol():
    return s.m.read(s.PROTOCOL)


def test_matrix_and_disjoint_protocol():
    p = protocol()
    s.validate_protocol(p)
    assert [len(s.cells(p, stage)) for stage in s.STAGES] == [3, 9, 90]
    assert [c["training_index"] for c in s.cells(p, "preflight")] == [0, 3, 6]
    assert len(s.variants(p)) == 8
    for c in s.cells(p, "evaluation"):
        d = s.cells(p, "diagnostics")[c["training_index"]]
        assert (c["task"], c["world_model_seed"]) == (d["task"], d["world_model_seed"])
    p["controller"]["horizon"] = 9
    with pytest.raises(ValueError):
        s.validate_protocol(p)


def diagnostics(p, *, improvement=2.0):
    result = []
    for cell in s.cells(p, "diagnostics"):
        summary = {}
        for split in ("calibration", "validation"):
            summary[split] = {
                family: {
                    "latent": dict(
                        reference_feasible_fraction=0.1,
                        real_executed_objective_mean=10.0,
                        real_executed_objective_gain_mean=100.0,
                    ),
                    "endpoint": dict(
                        reference_feasible_fraction=0.9,
                        real_executed_objective_mean=10.0 + improvement,
                        real_executed_objective_gain_mean=-100.0,
                    ),
                }
                for family in ("recursive", "direct")
            }
        result.append(dict(cell, summary=summary))
    return result


def test_selection_uses_matched_executed_levels_not_different_references():
    p = protocol()
    rows = diagnostics(p)
    choice = s.choose_reference_modes(rows, p)
    assert choice["reference_modes"] == dict(recursive="endpoint", direct="endpoint")
    assert choice["evidence"]["recursive"]["validation_real_gain_improvement"] == 2.0
    for gain in (0.0, -2.0):
        assert set(
            s.choose_reference_modes(diagnostics(p, improvement=gain), p)[
                "reference_modes"
            ].values()
        ) == {"latent"}
    rows = diagnostics(p)
    for r in rows[3:]:
        for f in ("recursive", "direct"):
            r["summary"]["validation"][f]["endpoint"][
                "real_executed_objective_mean"
            ] = 9.0
    assert set(s.choose_reference_modes(rows, p)["reference_modes"].values()) == {
        "latent"
    }
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            s.choose_reference_modes(bad, p)


def fixture(tmp_path):
    p = protocol()
    value = dict(
        source_commit="source",
        protocol=p,
        dependency_index={"0": dict(files={"checkpoint.pkl": "c" * 64})},
    )
    cell = s.cells(p, "evaluation")[0]
    out = s.cell_output(tmp_path, "evaluation", 0)
    out.mkdir(parents=True)
    trace = dict(
        actions=np.zeros((3, 1000, 2), np.float32),
        observations=np.zeros((3, 1000, 6), np.float32),
        rewards=np.zeros((3, 1000)),
        continuations=np.ones((3, 1000)),
        is_last=np.zeros((3, 1000), bool),
        lengths=np.ones(3, np.int32),
    )
    np.savez(out / "trace.npz", **trace)
    core = dict(
        cell,
        source_commit="source",
        manifest_sha256=s.m.digest(value),
        checkpoint_sha256="c" * 64,
        trace_sha256=s.b.array_sha256(trace),
        evaluation_environment_seeds=p["evaluation_environment_seeds"],
        episode_returns=[0.0, 0.0, 0.0],
        realized_controller_config=p["controller"],
        reference_modes=dict(recursive="latent", direct="latent"),
        action_saturation_fraction=0.0,
        telemetry={},
        endpoint_objective_metrics_applicable=False,
    )
    s.m.publish(out / "result.json", core)
    runtime = dict(
        python="3.12",
        jax="0.8",
        jaxlib="0.8",
        numpy="2",
        mujoco="3",
        dm_control="1",
        devices=["NVIDIA L40S"],
        x64=False,
    )
    for i, role in enumerate(s.ROLES):
        receipt = dict(
            mode=role,
            pid=i + 1,
            runtime=runtime,
            manifest_sha256=s.m.digest(value),
            source_commit="source",
            checkpoint_sha256="c" * 64,
            core_sha256=s.m.digest(core),
            trace_sha256=core["trace_sha256"],
            cache_sha256="f" * 64,
            wall_seconds=1.0,
            timing={},
        )
        s.m.publish(out / f"{role}-receipt.json", receipt)
        s.m.publish(
            out / f"{role}-seal.json",
            dict(
                cache_sha256="f" * 64,
                receipt_sha256=s.b.file_sha256(out / f"{role}-receipt.json"),
            ),
        )
    mark = dict(
        cell=cell,
        stage="evaluation",
        variant=None,
        source_commit="source",
        manifest_sha256=s.m.digest(value),
        strict_bitwise_replay=True,
        cache_sha256="f" * 64,
        files={n: s.b.file_sha256(out / n) for n in s.FILES},
    )
    s.m.publish(out / "verified.json", mark)
    s.m.publish(
        tmp_path / "selection.json",
        {"reference_modes": dict(recursive="latent", direct="latent")},
    )
    return value, out, mark


def save_fixture(path, obj):
    path.write_text(json.dumps(obj))


def test_genuine_artifacts_and_fail_closed_mutations(tmp_path):
    value, out, mark = fixture(tmp_path)
    with patch.object(s, "manifest", return_value=value):
        s.verify_cell(tmp_path, "evaluation", 0)
        for bad in (
            {},
            dict(mark, files={}),
            dict(mark, strict_bitwise_replay=False),
            dict(mark, strict_bitwise_replay=1),
            dict(mark, cell={}),
            dict(mark, cache_sha256="wrong"),
        ):
            save_fixture(out / "verified.json", bad)
            with pytest.raises((ValueError, KeyError)):
                s.verify_cell(tmp_path, "evaluation", 0)
        save_fixture(out / "verified.json", mark)
        core = s.m.read(out / "result.json")
        original_bytes = (out / "result.json").read_bytes()
        for key, value_ in (
            ("checkpoint_sha256", "wrong"),
            ("trace_sha256", "wrong"),
            ("evaluation_environment_seeds", [1, 2, 3]),
            ("realized_controller_config", {}),
        ):
            changed = dict(core, **{key: value_})
            save_fixture(out / "result.json", changed)
            remap = deepcopy(mark)
            remap["files"]["result.json"] = s.b.file_sha256(out / "result.json")
            save_fixture(out / "verified.json", remap)
            with pytest.raises(ValueError):
                s.verify_cell(tmp_path, "evaluation", 0)
        (out / "result.json").write_bytes(original_bytes)
        save_fixture(out / "verified.json", mark)
        s.verify_cell(tmp_path, "evaluation", 0)


def test_duplicate_launch_fails_before_scheduler(tmp_path):
    p = protocol()
    value = dict(source_commit="test", protocol=p)
    s.m.publish(tmp_path / "submissions/preflight.intent.json", {"uncertain": True})
    with patch.object(s, "manifest", return_value=value), patch.object(
        s.subprocess, "check_output"
    ) as submit:
        with pytest.raises(FileExistsError):
            s.launch(tmp_path, "preflight")
        submit.assert_not_called()


def test_selection_not_launched_before_diagnostics(tmp_path):
    value = dict(protocol=protocol())
    with patch.object(s, "manifest", return_value=value), patch.object(
        s, "verify_stage", side_effect=RuntimeError("not terminal")
    ), patch.object(s.subprocess, "check_output") as submit:
        with pytest.raises(RuntimeError):
            s.launch(tmp_path, "evaluation")
        submit.assert_not_called()


@pytest.mark.parametrize(
    "arm", ["B0_frozen_rebrac", "K0_repaired_recursive", "K1_repaired_direct"]
)
@pytest.mark.parametrize("reference_mode", ["latent", "endpoint"])
def test_actual_repaired_evaluation_path_replays(tiny_checkpoint, arm, reference_mode):
    p = protocol()
    p["preflight"]["evaluation_steps"] = 2
    cell = dict(
        index=0,
        training_index=0,
        task="dmc_reacher_hard",
        world_model_seed=431,
        actor_seed=541,
        arm=arm,
    )
    modes = dict(recursive=reference_mode, direct=reference_mode)
    core, trace, timing = s.evaluate(p, cell, tiny_checkpoint, modes, preflight=True)
    again, again_trace, _ = s.evaluate(p, cell, tiny_checkpoint, modes, preflight=True)
    assert core == again
    s.m.exact_trace(trace, again_trace)
    assert trace["actions"].shape == (1, 2, 2)
    assert timing["timed_steps"] == 2
    if arm.startswith("K"):
        np.testing.assert_array_equal(trace["objective_evaluations"], 10)
        # Zero-threshold fixture forces fallback; stage and Q are actual reference scores.
        assert trace["used_fallback"].all()
        np.testing.assert_array_equal(
            trace["executed_objective"], trace["reference_objective"]
        )
    else:
        assert core["endpoint_objective_metrics_applicable"] is False
