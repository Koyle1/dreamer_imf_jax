"""Fail-closed positive controls and independently rehashed negative mutations.

The fixture is intentionally tiny and does no model fitting. Training numerical
validity is covered by the separate CPU smoke test; these tests exercise the
real artifact, whole-stage and final verifiers, not an imitation of their logic.
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from dreamer_imf_compare import mechanism_replication as m


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=True))


def runtime():
    return dict(
        python="3.12.3",
        jax="0.8.1",
        jaxlib="0.8.1",
        numpy="2.5.3",
        mujoco="3.13.0",
        dm_control="1.0.46",
        devices=["NVIDIA L40S"],
        x64=False,
    )


def training_result(cell, preflight):
    return dict(
        cell=cell,
        preflight=preflight,
        runtime=runtime(),
        source_world_sha256="a" * 64,
        config={},
        budgets={},
        reward_optimizer_updates=2,
        normalization_train_targets=2,
        reward_test={},
        policies={},
        endpoints={},
        replay_sha256="b" * 64,
        schedule_sha256={},
        wall_seconds=1.0,
    )


class Study:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.p = deepcopy(m.read(m.PROTOCOL))
        self.p.update(
            tasks=self.p["tasks"][:1],
            world_model_seeds=[431],
            nested_actor_seeds=[541],
            evaluation_environment_seeds=[65001],
            evaluation_episodes_per_cell=1,
            maximum_environment_steps=2,
        )
        self.p["preflight"].update(evaluation_steps=2)
        self.p["uncertainty"]["resamples"] = 10
        self.m = dict(source_commit="f" * 40, protocol=self.p)
        monkeypatch.setattr(m, "manifest", lambda root: self.m)
        monkeypatch.setattr(m, "accounting", self.accounting)
        # These fixtures bind checkpoint bytes; they do not assert training
        # optimizer semantics. The production semantic verifier is not modified.
        monkeypatch.setattr(
            m,
            "verify_training",
            lambda protocol, directory, **kw: m.read(directory / "result.json"),
        )
        for stage in m.STAGES:
            submission = dict(
                job_id=str(100 + m.STAGES.index(stage)),
                manifest_sha256=m.digest(self.m),
                stage=stage,
                count=len(m.cells(self.p, stage)),
                script_sha256="c" * 64,
            )
            m.publish(root / "submissions" / f"{stage}.json", submission)
        train = m.cell_dir(root, "training", 0)
        train.mkdir(parents=True)
        for name in m.TRAINING_FILES - {"result.json"}:
            (train / name).write_bytes(name.encode())
        m.publish(
            train / "result.json",
            training_result(m.cells(self.p, "training")[0], False),
        )
        m.marker(root, train, m.cells(self.p, "training")[0], m.TRAINING_FILES)
        pf = m.cell_dir(root, "preflight", 0)
        pf_training = pf / "training"
        pf_training.mkdir(parents=True)
        (pf_training / "checkpoint.pkl").write_bytes(b"preflight checkpoint")
        m.publish(
            pf_training / "result.json",
            training_result(m.cells(self.p, "preflight")[0], True),
        )
        m.publish(
            pf / "result.json",
            dict(
                cell=m.cells(self.p, "preflight")[0], runtime=runtime(), status="passed"
            ),
        )
        m.marker(root, pf, m.cells(self.p, "preflight")[0], ["result.json"])
        for i, arm in enumerate(self.p["arms"]):
            cell = dict(m.cells(self.p, "preflight")[0], actor_seed=541, arm=arm)
            self.evaluation(pf / f"arm-{i}", cell, pf_training, preflight=True)
        for cell in m.cells(self.p, "evaluation"):
            self.evaluation(m.cell_dir(root, "evaluation", cell["index"]), cell, train)

    def accounting(self, job, count):
        rows = [
            [f"{job}_{i}", "COMPLETED", "0:0", "1", "gres/gpu=1"] for i in range(count)
        ]
        return dict(records=rows, raw="".join("|".join(row) + "\n" for row in rows))

    def evaluation(self, directory, cell, train, *, preflight=False):
        directory.mkdir(parents=True)
        seeds = (
            self.p["preflight"]["evaluation_seeds"]
            if preflight
            else self.p["evaluation_environment_seeds"]
        )
        trace = dict(
            actions=np.zeros((1, 2, 1), np.float32),
            observations=np.zeros((1, 2, 2), np.float32),
            rewards=np.ones((1, 2), np.float32),
            continuations=np.ones((1, 2), np.float32),
            is_last=np.array([[False, True]]),
            lengths=np.array([2], np.int32),
            evaluation_seeds=np.array(seeds, np.uint32),
        )
        np.savez_compressed(directory / "trace.npz", **trace)
        result = dict(
            cell,
            evaluation_environment_seeds=seeds,
            episode_returns=[2.0],
            action_saturation_fraction=0.0,
            telemetry={},
            coverage={},
            trace_sha256=m.b.array_sha256(trace),
            checkpoint_sha256=m.b.file_sha256(train / "checkpoint.pkl"),
            realized_controller_config=m.validate_controller_protocol(
                self.p["controller"]
            ),
        )
        m.publish(directory / "result.json", result)
        for i, mode in enumerate(m.REPLAY_MODES):
            receipt = dict(
                mode=mode,
                pid=1000 + i,
                runtime=runtime(),
                core_sha256=m.digest(result),
                trace_sha256=result["trace_sha256"],
                cache_sha256="d" * 64,
                wall_seconds=2.0,
                timing=dict(
                    discarded_compile_warmup_steps=1.0,
                    mean_milliseconds_per_step=500.0,
                    timed_steps=2.0,
                    total_timed_seconds=1.0,
                ),
            )
            m.publish(directory / f"{mode}-receipt.json", receipt)
            m.publish(
                directory / f"{mode}-seal.json",
                dict(
                    cache_sha256="d" * 64,
                    receipt_sha256=m.b.file_sha256(directory / f"{mode}-receipt.json"),
                ),
            )
        m.marker(
            self.root,
            directory,
            result,
            m.EVALUATION_FILES,
            strict_bitwise_replay=True,
            cache_sha256="d" * 64,
        )

    @property
    def evaluation_dir(self):
        return m.cell_dir(self.root, "evaluation", 0)

    def rebind(self, directory, *, result=False):
        mark = m.read(directory / "verified.json")
        if result:
            core = m.read(directory / "result.json")
            mark["cell"] = core
            for mode in m.REPLAY_MODES:
                path = directory / f"{mode}-receipt.json"
                receipt = m.read(path)
                receipt.update(
                    core_sha256=m.digest(core), trace_sha256=core["trace_sha256"]
                )
                write_json(path, receipt)
        for mode in m.REPLAY_MODES:
            seal = directory / f"{mode}-seal.json"
            if seal.exists():
                value = m.read(seal)
                value["receipt_sha256"] = m.b.file_sha256(
                    directory / f"{mode}-receipt.json"
                )
                write_json(seal, value)
        mark["files"] = {
            name: m.b.file_sha256(directory / name) for name in mark["files"]
        }
        write_json(directory / "verified.json", mark)


@pytest.fixture
def study(tmp_path, monkeypatch):
    return Study(tmp_path, monkeypatch)


def test_minimal_complete_study_passes_cell_stage_and_final(study):
    for stage in m.STAGES:
        m.stage_verify(study.root, stage)
    m.finalize(study.root)
    m.finalize(study.root)
    m.verify_files(study.root, study.root)
    assert m.read(study.root / "report.json")["cells"] == 4


@pytest.mark.parametrize("flag", [None, False, 0, 1, "true"])
@pytest.mark.parametrize("stage", ["evaluation", "preflight"])
def test_strict_replay_cannot_be_opted_out(study, flag, stage):
    directory = (
        study.evaluation_dir
        if stage == "evaluation"
        else m.cell_dir(study.root, "preflight", 0) / "arm-0"
    )
    mark = m.read(directory / "verified.json")
    if flag is None:
        del mark["strict_bitwise_replay"]
    else:
        mark["strict_bitwise_replay"] = flag
    write_json(directory / "verified.json", mark)
    with pytest.raises(ValueError):
        m.verify_files(study.root, directory)
    with pytest.raises(ValueError):
        m.stage_verify(study.root, stage)
    with pytest.raises(ValueError):
        m.finalize(study.root)


@pytest.mark.parametrize("filename", sorted(m.EVALUATION_FILES) + ["all", "extra"])
def test_exact_evaluation_bound_file_set(study, filename):
    directory = study.evaluation_dir
    mark = m.read(directory / "verified.json")
    if filename == "all":
        mark["files"] = {}
    elif filename == "extra":
        mark["files"]["unknown.json"] = "0" * 64
    else:
        del mark["files"][filename]
    write_json(directory / "verified.json", mark)
    with pytest.raises(ValueError):
        m.stage_verify(study.root, "evaluation")


@pytest.mark.parametrize("filename", sorted(m.EVALUATION_FILES))
def test_missing_or_unbound_artifact_fails(study, filename):
    (study.evaluation_dir / filename).unlink()
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "other"),
        ("index", 1),
        ("index", False),
        ("actor_seed", 999),
        ("world_model_seed", 433),
        ("training_index", 1),
        ("arm", "unknown"),
        ("checkpoint_sha256", "e" * 64),
        ("evaluation_environment_seeds", [75001]),
        ("episode_returns", [True]),
    ],
)
def test_rehashed_result_identity_and_checkpoint_substitution_fails(
    study, field, value
):
    path = study.evaluation_dir / "result.json"
    result = m.read(path)
    result[field] = value
    write_json(path, result)
    study.rebind(study.evaluation_dir, result=True)
    with pytest.raises(ValueError):
        m.stage_verify(study.root, "evaluation")
    with pytest.raises(ValueError):
        m.finalize(study.root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_commit", "wrong"),
        ("manifest_sha256", "e" * 64),
        ("cell", {}),
        ("unexpected", 1),
    ],
)
def test_marker_identity_schema_fails(study, field, value):
    path = study.evaluation_dir / "verified.json"
    mark = m.read(path)
    mark[field] = value
    write_json(path, mark)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize("mode", m.REPLAY_MODES)
@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "wrong"),
        ("pid", True),
        ("pid", -1),
        ("pid", 1001),
        ("cache_sha256", "e" * 64),
        ("wall_seconds", float("inf")),
        ("runtime", dict(runtime(), numpy="wrong")),
        ("unexpected", 1),
    ],
)
def test_rehashed_receipt_mutations_fail(study, mode, field, value):
    if field == "pid" and value == 1001 and mode == "create":
        value = 1000
    path = study.evaluation_dir / f"{mode}-receipt.json"
    receipt = m.read(path)
    receipt[field] = value
    write_json(path, receipt)
    study.rebind(study.evaluation_dir)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize("mode", ["create", "replay"])
@pytest.mark.parametrize("field", ["core_sha256", "trace_sha256"])
def test_rehashed_reader_binding_fails(study, mode, field):
    path = study.evaluation_dir / f"{mode}-receipt.json"
    receipt = m.read(path)
    receipt[field] = "e" * 64
    write_json(path, receipt)
    study.rebind(study.evaluation_dir)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize("mode", m.REPLAY_MODES)
@pytest.mark.parametrize("field", ["cache_sha256", "receipt_sha256", "unexpected"])
def test_rehashed_seal_mutations_fail(study, mode, field):
    path = study.evaluation_dir / f"{mode}-seal.json"
    seal = m.read(path)
    seal[field] = "e" * 64
    write_json(path, seal)
    mark_path = study.evaluation_dir / "verified.json"
    mark = m.read(mark_path)
    mark["files"][path.name] = m.b.file_sha256(path)
    write_json(mark_path, mark)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize("mutation", ["nan", "digest", "seeds", "lengths", "missing"])
def test_rehashed_retained_trace_mutations_fail(study, mutation):
    path = study.evaluation_dir / "trace.npz"
    trace = m.b.load_npz(path)
    if mutation in ("nan", "digest"):
        trace["actions"][0, 0, 0] = np.nan if mutation == "nan" else 0.5
    elif mutation == "seeds":
        trace["evaluation_seeds"][0] = 2
    elif mutation == "lengths":
        trace["lengths"][0] = 0
    else:
        del trace["rewards"]
    np.savez_compressed(path, **trace)
    if mutation != "digest":
        result = m.read(study.evaluation_dir / "result.json")
        result["trace_sha256"] = m.b.array_sha256(trace)
        write_json(study.evaluation_dir / "result.json", result)
        study.rebind(study.evaluation_dir, result=True)
    else:
        study.rebind(study.evaluation_dir)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize("stage", m.STAGES)
@pytest.mark.parametrize(
    "field", ["stage", "manifest_sha256", "accounting", "cell_markers", "extra"]
)
def test_existing_stage_aggregate_schema_and_content_fail(study, stage, field):
    m.stage_verify(study.root, stage)
    path = study.root / "verified" / f"{stage}.json"
    record = m.read(path)
    record[field] = {} if field in ("accounting", "cell_markers") else "wrong"
    write_json(path, record)
    with pytest.raises(ValueError):
        m.stage_verify(study.root, stage)


@pytest.mark.parametrize(
    "field,value",
    [
        ("stage", "wrong"),
        ("manifest_sha256", "e" * 64),
        ("count", 0),
        ("count", True),
        ("job_id", "bad"),
        ("script_sha256", "bad"),
        ("extra", 1),
    ],
)
def test_submission_schema_and_binding_fail(study, field, value):
    path = study.root / "submissions/evaluation.json"
    receipt = m.read(path)
    receipt[field] = value
    write_json(path, receipt)
    with pytest.raises(ValueError):
        m.stage_verify(study.root, "evaluation")


@pytest.mark.parametrize(
    "field", ["source_commit", "cell", "files", "manifest_sha256", "extra"]
)
def test_final_marker_cannot_weaken_schema(study, field):
    m.finalize(study.root)
    path = study.root / "verified.json"
    mark = m.read(path)
    mark[field] = {} if field in ("cell", "files") else "wrong"
    write_json(path, mark)
    with pytest.raises(ValueError):
        m.finalize(study.root)


def test_source_authentication_is_not_relaxed(monkeypatch, tmp_path):
    protocol = m.read(m.PROTOCOL)
    m.publish(
        tmp_path / "manifest.json",
        dict(
            source_commit="old", protocol=protocol, protocol_sha256=m.digest(protocol)
        ),
    )
    monkeypatch.setattr(m, "clean_commit", lambda: "new")
    with pytest.raises(ValueError, match="source/protocol"):
        m.manifest(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "changed", "bool", "nan"])
def test_controller_protocol_rejected_before_checkpoint_load(monkeypatch, mutation):
    p = deepcopy(m.read(m.PROTOCOL))
    if mutation == "missing":
        del p["controller"]["horizon"]
    elif mutation == "unknown":
        p["controller"]["unknown"] = 1
    else:
        p["controller"]["horizon"] = {"changed": 6, "bool": True, "nan": float("nan")}[
            mutation
        ]
    monkeypatch.setattr(
        m,
        "load_checkpoint",
        lambda _: pytest.fail("checkpoint loaded before protocol validation"),
    )
    with pytest.raises(ValueError):
        m.evaluate(p, m.cells(p, "evaluation")[0], "unused")


def test_controller_values_are_actual_runner_constants(monkeypatch):
    from dreamer_imf_compare import actor_gap_roadmap_study as roadmap

    p = m.read(m.PROTOCOL)["controller"]
    assert m.validate_controller_protocol(p) == p
    monkeypatch.setattr(roadmap, "FLOWMPC_PARTICLES", 4)
    with pytest.raises(ValueError, match="flowmpc_particles"):
        m.validate_controller_protocol(p)
    assert (
        m.validate_controller_protocol(dict(p, flowmpc_particles=4))[
            "flowmpc_particles"
        ]
        == 4
    )


def test_new_results_require_realized_config(study):
    path = study.evaluation_dir / "result.json"
    core = m.read(path)
    del core["realized_controller_config"]
    write_json(path, core)
    study.rebind(study.evaluation_dir, result=True)
    with pytest.raises(ValueError, match="field set"):
        m.verify_files(study.root, study.evaluation_dir)
    # Historical schema compatibility is explicit and does not bypass manifest
    # authentication: here only the pure result-schema validator is exercised.
    m.validate_evaluation_result(
        core, study.p, m.cells(study.p, "evaluation")[0], legacy=True
    )


def test_nonfinite_result_cannot_hide_behind_finite_receipts(study):
    path = study.evaluation_dir / "result.json"
    core = m.read(path)
    core["episode_returns"][0] = float("nan")
    write_json(path, core)
    study.rebind(study.evaluation_dir)
    with pytest.raises(ValueError, match="nonfinite"):
        m.verify_files(study.root, study.evaluation_dir)


def test_actual_checkpoint_replacement_cannot_be_rehashed_into_evaluation(study):
    directory = m.cell_dir(study.root, "training", 0)
    (directory / "checkpoint.pkl").write_bytes(b"different task checkpoint")
    study.rebind(directory)
    with pytest.raises(ValueError, match="checkpoint binding"):
        m.stage_verify(study.root, "evaluation")


@pytest.mark.parametrize("stage", ["training", "evaluation"])
def test_extra_physical_artifact_is_not_silently_ignored(study, stage):
    directory = m.cell_dir(study.root, stage, 0)
    (directory / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="directory set"):
        m.stage_verify(study.root, stage)


@pytest.mark.parametrize("stage", ["training", "preflight"])
def test_training_and_preflight_result_identity_bound(study, stage):
    directory = m.cell_dir(study.root, stage, 0)
    result = m.read(directory / "result.json")
    result["cell"]["world_model_seed"] = 433
    write_json(directory / "result.json", result)
    study.rebind(directory)
    with pytest.raises(ValueError, match="cell|identity"):
        m.stage_verify(study.root, stage)


def test_all_three_runtime_receipts_must_match_preflight(study):
    for role in m.REPLAY_MODES:
        path = study.evaluation_dir / f"{role}-receipt.json"
        receipt = m.read(path)
        receipt["runtime"]["jax"] = "wrong-but-consistent-across-all-three"
        write_json(path, receipt)
    study.rebind(study.evaluation_dir)
    with pytest.raises(ValueError, match="authenticated preflight"):
        m.stage_verify(study.root, "evaluation")


@pytest.mark.parametrize("mutation", ["nan", "missing", "negative", "zero_steps"])
def test_receipt_timing_schema_and_finiteness(study, mutation):
    path = study.evaluation_dir / "create-receipt.json"
    receipt = m.read(path)
    if mutation == "missing":
        del receipt["timing"]["total_timed_seconds"]
    else:
        receipt["timing"]["timed_steps"] = {
            "nan": float("nan"),
            "negative": -1,
            "zero_steps": 0,
        }[mutation]
    write_json(path, receipt)
    study.rebind(study.evaluation_dir)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize("mutation", ["missing", "changed", "bool", "unknown"])
def test_realized_controller_config_cannot_lie(study, mutation):
    path = study.evaluation_dir / "result.json"
    core = m.read(path)
    if mutation == "missing":
        del core["realized_controller_config"]["horizon"]
    else:
        key = "unknown" if mutation == "unknown" else "heldout_minimum_improvement"
        core["realized_controller_config"][key] = False if mutation == "bool" else 7
    write_json(path, core)
    study.rebind(study.evaluation_dir, result=True)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize(
    "mutation", ["analysis", "allocation", "runtime", "stages", "extra", "records"]
)
def test_rehashed_final_report_schema_and_reconstruction(study, mutation):
    m.finalize(study.root)
    path = study.root / "report.json"
    report = m.read(path)
    if mutation == "analysis":
        report["contrasts"]["heldout_acceptance"]["primary_effect"] += 1
    elif mutation == "allocation":
        report["runtime"]["allocation_gpu_seconds_by_stage"]["preflight"] = True
    elif mutation == "runtime":
        report["runtime"]["evaluation_by_cell"]["0"]["role_wall_seconds"] = {}
    elif mutation == "stages":
        del report["stages"]["preflight"]
    elif mutation == "extra":
        report["unknown"] = 1
    else:
        report["records"][0]["checkpoint_sha256"] = "e" * 64
    write_json(path, report)
    study.rebind(study.root)
    with pytest.raises(ValueError):
        m.finalize(study.root)
    with pytest.raises(ValueError):
        m.verify_files(study.root, study.root)


def test_duplicate_json_fields_fail_closed(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"strict_bitwise_replay":false,"strict_bitwise_replay":true}')
    with pytest.raises(ValueError, match="duplicate"):
        m.read(path)


def test_new_evaluation_serializes_actual_controller_config(study, monkeypatch):
    from types import SimpleNamespace
    from dreamer_imf_compare import actor_gap_roadmap_study as roadmap
    from dreamer_imf_compare import actor_gap_diagnostics as diagnostics

    model = dict(
        config=None,
        rebrac_config=None,
        reward_world={},
        policies={541: None},
        anchors=[],
    )
    trace = m.b.load_npz(study.evaluation_dir / "trace.npz")
    monkeypatch.setattr(m, "load_checkpoint", lambda _: model)
    monkeypatch.setattr(roadmap, "_run_flowmpc_arm", lambda *a, **k: ([2.0], trace, {}))
    monkeypatch.setattr(m.b, "load_npz", lambda _: {})
    monkeypatch.setattr(
        roadmap,
        "_training_transitions",
        lambda _: (np.zeros((2, 2)), np.zeros((2, 1)), None, None, None),
    )
    monkeypatch.setattr(diagnostics, "fit_coverage_calibration", lambda *a, **k: None)
    monkeypatch.setattr(
        diagnostics,
        "score_observation_action_coverage",
        lambda *a, **k: SimpleNamespace(to_dict=lambda: {}),
    )
    core, _, _ = m.evaluate(
        study.p,
        m.cells(study.p, "evaluation")[0],
        m.cell_dir(study.root, "training", 0),
    )
    assert core["realized_controller_config"] == m.validate_controller_protocol(
        study.p["controller"]
    )


@pytest.mark.parametrize(
    "field,value", [("episode_returns", [3.0]), ("action_saturation_fraction", 0.5)]
)
def test_return_and_action_statistics_bound_to_retained_trace(study, field, value):
    path = study.evaluation_dir / "result.json"
    core = m.read(path)
    core[field] = value
    write_json(path, core)
    study.rebind(study.evaluation_dir, result=True)
    with pytest.raises(ValueError, match="retained"):
        m.verify_files(study.root, study.evaluation_dir)


def test_new_float64_reward_trace_and_legacy_float32_both_validate(study):
    path = study.evaluation_dir / "trace.npz"
    trace = m.b.load_npz(path)
    trace["rewards"] = trace["rewards"].astype(np.float64)
    np.savez_compressed(path, **trace)
    result_path = study.evaluation_dir / "result.json"
    core = m.read(result_path)
    core["trace_sha256"] = m.b.array_sha256(trace)
    write_json(result_path, core)
    study.rebind(study.evaluation_dir, result=True)
    m.verify_files(study.root, study.evaluation_dir)


@pytest.mark.parametrize(
    "stage,filename",
    [("training", name) for name in sorted(m.TRAINING_FILES)]
    + [("preflight", "result.json")],
)
def test_every_nonevaluation_stage_has_exact_file_schema(study, stage, filename):
    path = m.cell_dir(study.root, stage, 0) / "verified.json"
    mark = m.read(path)
    del mark["files"][filename]
    write_json(path, mark)
    with pytest.raises(ValueError, match="field set"):
        m.stage_verify(study.root, stage)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "no_gpu", "raw_mismatch", "noninteger_time", "missing"]
)
def test_accounting_exact_cell_status_and_gpu_binding(study, mutation):
    value = study.accounting("102", 4)
    if mutation == "duplicate":
        value["records"][1] = value["records"][0]
    elif mutation == "no_gpu":
        value["records"][0][4] = "cpu=8"
    elif mutation == "raw_mismatch":
        value["raw"] = ""
    elif mutation == "noninteger_time":
        value["records"][0][3] = "NaN"
    else:
        value["records"].pop()
    with pytest.raises(ValueError):
        m.validate_accounting(value, "102", 4)
