"""Independent analytic controls and tamper tests for stored-trace reanalysis."""

import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from dreamer_imf_compare import decision_ranking_analysis as ranking

ARTIFACTS = Path(__file__).resolve().parents[2] / ".unlazy/controller-repair/artifacts"


@pytest.fixture(scope="module")
def study():
    if not ARTIFACTS.is_dir():
        pytest.skip(
            "frozen controller-repair evidence not present; analytic tests still run"
        )
    return ranking.authenticate_artifacts(ARTIFACTS)


@pytest.fixture(scope="module")
def report(study):
    return ranking.analyze_artifacts(ARTIFACTS)


def metric(scores, real, feasible=None, *, policy="unrestricted", name="test"):
    if feasible is None:
        feasible = np.ones(len(real), bool)
    return ranking.analyze_candidates({name: scores}, real, feasible)[policy]["scores"][
        name
    ]


def test_common_offset_cancels_exactly():
    real = np.array([2.0, 7.0, 5.0, -1.0])
    shifted = real + 1000
    assert np.mean(abs(shifted - real)) == 1000
    result = metric(shifted, real)
    assert result["gain_mae"] == 0
    assert result["informative_pair_agreement"] == 1
    assert result["chosen_index"] == 1
    assert result["chosen_real_gain"] == 5
    assert result["oracle_regret"] == 0


def test_action_dependent_error_reverses_order():
    result = metric([100.0, 99.0, 98.0], [0.0, 1.0, 2.0])
    assert result["gain_mae"] == 3
    assert result["gain_mae_including_reference"] == 2
    assert result["informative_pair_count"] == 3
    assert result["informative_pair_agreement"] == 0
    assert result["chosen_index"] == 0
    assert result["oracle_regret"] == 2


def test_model_ties_are_not_correct_informative_rankings():
    result = metric([7.0, 7.0, 7.0], [0.0, 2.0, 1.0])
    assert result["model_tie_pair_count"] == 3
    assert result["model_tie_informative_pair_count"] == 3
    assert result["pair_agreement_count"] == 0
    assert result["chosen_index"] == 0
    assert result["exact_argmax_indices"] == [0, 1, 2]
    assert result["exact_tie_real_gain_min"] == 0
    assert result["exact_tie_real_gain_max"] == 2
    assert result["exact_tie_regret_min"] == 0
    assert result["exact_tie_regret_max"] == 2


def test_real_ties_are_explicit_uninformative_pairs():
    result = metric([1.0, 2.0, 2.0], [4.0, 4.0, 4.0])
    assert result["informative_pair_count"] == 0
    assert result["informative_pair_agreement"] is None
    assert result["real_tie_pair_count"] == 3
    assert result["model_tie_pair_count"] == 1
    assert result["both_tie_pair_count"] == 1
    assert result["oracle_regret"] == 0


def test_near_tie_does_not_change_exact_first_argmax():
    result = metric([1.0, 1.0 + 5e-9, 0.0], [0.0, 3.0, 2.0])
    assert result["chosen_index"] == 1
    assert result["exact_argmax_indices"] == [1]
    assert result["near_argmax_indices"] == [0, 1]
    assert result["exact_tie_regret_max"] == 0
    assert result["near_tie_regret_max"] == 3
    assert result["model_tie_informative_pair_count"] == 1


def test_empty_feasible_decision_has_no_oracle_regret_or_fake_safe_fallback():
    result = ranking.analyze_candidates(
        {"test": [2, 3]}, [1, 8], np.array([False, False])
    )
    constrained = result["fixed_feasible"]
    assert constrained["empty"] and constrained["admissible_indices"] == []
    for key in (
        "chosen_index",
        "chosen_real_gain",
        "oracle_real_gain",
        "oracle_regret",
        "gain_mae",
        "informative_pair_agreement",
    ):
        assert constrained["scores"]["test"][key] is None
    assert constrained["scores"]["test"]["exact_argmax_indices"] == []
    fallback = result["reference_fallback"]
    assert fallback["required_by_empty_fixed_feasible_set"]
    assert not fallback["model_feasible"] and not fallback["is_feasible_decision"]
    assert fallback["real_gain"] == 0
    assert fallback["unrestricted_oracle_regret"] == 7
    assert result["unrestricted"]["scores"]["test"]["chosen_real_gain"] == 7


def test_mask_is_fixed_and_reference_is_not_forced_into_feasible_set():
    result = ranking.analyze_candidates(
        {"bad": [100, 3, 2], "good": [100, 2, 3]},
        [7, 6, 9],
        np.array([False, True, True]),
    )
    constrained = result["fixed_feasible"]
    assert constrained["admissible_indices"] == [1, 2]
    assert not constrained["reference_is_admissible"]
    assert constrained["scores"]["bad"]["chosen_index"] == 1
    assert constrained["scores"]["bad"]["chosen_real_gain"] == -1
    assert constrained["scores"]["bad"]["oracle_regret"] == 3
    assert constrained["scores"]["good"]["chosen_index"] == 2
    assert constrained["scores"]["good"]["oracle_regret"] == 0
    # Both gains continue to subtract the original (infeasible) reference.
    assert constrained["scores"]["bad"]["gain_mae"] == 98


def test_single_reference_has_no_nonreference_mae_or_informative_pairs():
    result = metric([5.0], [9.0])
    assert result["gain_mae"] is None
    assert result["gain_mae_including_reference"] == 0
    assert result["informative_pair_agreement"] is None
    assert result["chosen_index"] == 0 and result["oracle_regret"] == 0


def test_nonzero_reference_and_first_admissible_tie():
    result = ranking.analyze_candidates(
        {"test": [10, 10, 10]},
        [4, 6, 8],
        np.array([False, True, True]),
        reference_index=1,
    )
    chosen = result["fixed_feasible"]["scores"]["test"]
    assert chosen["chosen_index"] == 1
    assert chosen["gain_mae"] == 2
    assert chosen["chosen_real_gain"] == 0
    assert chosen["exact_argmax_indices"] == [1, 2]
    assert chosen["exact_tie_real_gain_max"] == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_and_returns_fail_closed(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        metric([0, bad], [0, 1])
    with pytest.raises(ValueError, match="nonfinite"):
        metric([0, 1], [0, bad])


@pytest.mark.parametrize("feasible", [[0, 1], [True], [[True, True]], [True, None]])
def test_invalid_feasibility_mask_rejected(feasible):
    with pytest.raises(ValueError, match="Boolean"):
        metric([1, 2], [1, 2], feasible)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference_index": -1},
        {"reference_index": 2},
        {"reference_index": True},
        {"tie_atol": -1},
        {"tie_atol": np.inf},
    ],
)
def test_invalid_reference_or_tolerance_rejected(kwargs):
    with pytest.raises(ValueError):
        ranking.analyze_candidates({"test": [1, 2]}, [1, 2], np.ones(2, bool), **kwargs)


def test_independent_pair_and_regret_example():
    real, model, mask = [3, 2, 7, 5], [6, 6, 8, 3], np.array([True, False, True, True])
    result = metric(model, real, mask, policy="fixed_feasible")
    assert result["pair_count"] == 3
    assert result["pair_agreement_count"] == 2
    assert result["informative_pair_agreement"] == pytest.approx(2 / 3)
    assert result["chosen_real_gain"] == 4 and result["oracle_regret"] == 0
    assert result["gain_mae"] == 3.5


@pytest.mark.parametrize(
    "encoded", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}']
)
def test_strict_json_rejects_duplicate_or_nonfinite(encoded):
    with pytest.raises(ValueError):
        ranking._json(encoded)


def test_named_array_digest_binds_order_shape_dtype_values():
    arrays = {"a": np.array([1, 2], dtype=np.int64), "b": np.array([3.0])}
    baseline = ranking.array_digest(arrays)
    assert baseline == ranking.array_digest(dict(reversed(list(arrays.items()))))
    for value in (
        np.array([1, 3], np.int64),
        np.array([1, 2], np.int32),
        np.array([[1, 2]], np.int64),
    ):
        assert ranking.array_digest(dict(arrays, a=value)) != baseline


def test_wrong_source_and_manifest_bindings_rejected():
    valid = {
        "source_commit": ranking.SOURCE_COMMIT,
        "manifest_sha256": ranking.MANIFEST_SHA256,
    }
    ranking._bound(valid, "control")
    for key in valid:
        with pytest.raises(ValueError, match="binding differs"):
            ranking._bound(dict(valid, **{key: "0" * len(valid[key])}), "mutant")


def test_frozen_evidence_authenticates_without_production_jax(study):
    assert len(study.records) == len(study.traces) == 9
    assert {r["world_model_seed"] for r in study.records} == {431, 433, 439}
    assert len({r["task"] for r in study.records}) == 3
    assert all(not a.flags.writeable for t in study.traces for a in t.values())
    code = "import sys; from dreamer_imf_compare.decision_ranking_analysis import authenticate_artifacts; authenticate_artifacts(sys.argv[1]); assert 'jax' not in sys.modules; print('CPU_ONLY_OK')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(ARTIFACTS)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CPU_ONLY_OK"


@pytest.mark.parametrize(
    "relative",
    [
        "report.json",
        "manifest.json",
        "selection.json",
        "verified.json",
        "diagnostics/000/verified.json",
        "diagnostics/000/trace.npz",
        "diagnostics/004/create-receipt.json",
        "diagnostics/008/replay-seal.json",
    ],
)
def test_artifact_tampering_rejected_after_positive_control(study, tmp_path, relative):
    # Copy, never edit scientific inputs. The module fixture is the positive control.
    root = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, root)
    target = root / relative
    if relative.endswith(".npz"):
        target.write_bytes(target.read_bytes() + b"tamper")
    else:
        value = json.loads(target.read_text())
        value["tampered"] = True
        target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="mismatch|differs|bindings"):
        ranking.authenticate_artifacts(root)


def test_missing_authenticated_input_rejected(study, tmp_path):
    root = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, root)
    (root / "diagnostics/008/trace.npz").unlink()
    with pytest.raises(FileNotFoundError):
        ranking.authenticate_artifacts(root)


@pytest.mark.parametrize(
    "key,edit",
    [
        ("model_objective", lambda a: a.__setitem__(0, np.nan)),
        ("real_terminal_critic", lambda a: a.__setitem__(0, a[0] + 1)),
        ("snapshot_id", lambda a: a.__setitem__(0, a[0] + 1)),
        ("plan_kind", lambda a: a.__setitem__(2, 3)),
        ("action_sequence", lambda a: a.__setitem__((2, 0, 0), 0.234567)),
        ("real_rewards", lambda a: a.__setitem__((0, 0), a[0, 0] + 1)),
    ],
)
def test_semantic_trace_defects_rejected(study, key, edit):
    trace = dict(study.traces[0])
    trace[key] = trace[key].copy()
    edit(trace[key])
    with pytest.raises(ValueError):
        ranking.validate_trace(trace, study.records[0])


def test_report_preserves_all_groups_and_selected_primary(report):
    assert len(report["cases"]) == 288
    assert len(report["world_model_summaries"]) == 72
    assert len(report["task_summaries"]) == 24
    assert len(report["group_summaries"]) == 8
    assert len(report["primary_validation"]) == 2
    assert {
        (r["split"], r["family"], r["reference_mode"])
        for r in report["primary_validation"]
    } == {("validation", f, "latent") for f in ranking.FAMILIES}
    assert all(
        r["snapshot_count"] == 4 and r["world_model_count"] == 1
        for r in report["world_model_summaries"]
    )
    assert all(
        r["snapshot_count"] == 12 and r["world_model_count"] == 3
        for r in report["task_summaries"]
    )
    assert all(
        r["snapshot_count"] == 36 and r["world_model_count"] == 9
        for r in report["group_summaries"]
    )
    json.dumps(report, allow_nan=False)


def test_all_case_scores_and_duplicate_exclusion_recomputed(study, report):
    for case in report["cases"]:
        t = study.traces[case["diagnostic_index"]]
        ids = case["candidate_trace_indices"]
        assert len(ids) == 8 and np.count_nonzero(t["plan_kind"][ids] == 3) == 6
        assert not np.any(t["plan_kind"][ids] == 2)
        assert case["observed_execution"]["excluded_trace_index"] not in ids
        np.testing.assert_array_equal(
            case["scores"]["model"], t["model_objective"][ids]
        )
        np.testing.assert_allclose(
            case["scores"]["real_reward"],
            t["real_stage"][ids] + t["model_terminal"][ids],
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            case["scores"]["real_endpoint_critic"],
            t["real_stage"][ids] + t["real_terminal_critic"][ids],
            rtol=1e-12,
            atol=1e-12,
        )
        for policy in ranking.POLICIES:
            domain = (
                np.arange(8)
                if policy == "unrestricted"
                else np.flatnonzero(t["model_feasible"][ids])
            )
            for score in ranking.SCORES:
                outcome = case["decisions"][policy]["scores"][score]
                if not len(domain):
                    assert outcome["oracle_regret"] is None
                    continue
                prediction, real = (
                    np.array(case["scores"][score]),
                    t["real_objective"][ids],
                )
                chosen = domain[np.argmax(prediction[domain])]
                assert outcome["chosen_index"] == chosen
                assert outcome["oracle_regret"] == max(real[domain]) - real[chosen]
                informative = [
                    (a, b)
                    for a, b in itertools.combinations(domain, 2)
                    if abs(real[a] - real[b]) > 1e-8
                ]
                correct = sum(
                    abs(prediction[a] - prediction[b]) > 1e-8
                    and np.sign(real[a] - real[b])
                    == np.sign(prediction[a] - prediction[b])
                    for a, b in informative
                )
                assert outcome["informative_pair_count"] == len(informative)
                assert outcome["pair_agreement_count"] == correct
                assert (
                    case["decisions"][policy]["scores"]["oracle"]["oracle_regret"] == 0
                )


def test_hierarchical_task_means_match_model_means(report):
    for task_row in report["task_summaries"]:
        worlds = [
            r
            for r in report["world_model_summaries"]
            if all(
                r[k] == task_row[k]
                for k in ("split", "family", "reference_mode", "task")
            )
        ]
        for score in ranking.SCORES:
            actual = task_row["decisions"]["unrestricted"][score]["means"][
                "oracle_regret"
            ]
            expected = np.mean(
                [
                    r["decisions"]["unrestricted"][score]["means"]["oracle_regret"]
                    for r in worlds
                ]
            )
            assert actual == expected
    for row in report["group_summaries"]:
        tasks = [
            r
            for r in report["task_summaries"]
            if all(r[k] == row[k] for k in ("split", "family", "reference_mode"))
        ]
        for score in ranking.SCORES:
            assert row["decisions"]["unrestricted"][score]["means"][
                "oracle_regret"
            ] == np.mean(
                [
                    r["decisions"]["unrestricted"][score]["means"]["oracle_regret"]
                    for r in tasks
                ]
            )


def test_conditional_summary_denominators_are_explicit(report):
    for level, unit, children in (
        ("world_model_summaries", "snapshot", 4),
        ("task_summaries", "world_model", 3),
        ("group_summaries", "task", 3),
    ):
        for row in report[level]:
            assert row["averaging_unit"] == unit
            for policy in ranking.POLICIES:
                for score in ranking.SCORES:
                    metrics = row["decisions"][policy][score]
                    for key, value in metrics["means"].items():
                        contributors = metrics["mean_contributor_counts"][key]
                        assert 0 <= contributors <= children
                        assert (value is None) == (contributors == 0)
            assert (
                row["decisions"]["unrestricted"]["model"]["mean_contributor_counts"][
                    "oracle_regret"
                ]
                == children
            )


def test_cli_json_token_and_exclusive_external_output(study, tmp_path, capsys):
    before = {
        str(p.relative_to(ARTIFACTS)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARTIFACTS.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "ranking.json"
    assert ranking.main([str(ARTIFACTS), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.strip() == ranking.SUCCESS_TOKEN
    assert (
        json.loads(output.read_text())["schema"]
        == "controller-repair-decision-ranking-v1"
    )
    saved = output.read_bytes()
    assert ranking.main([str(ARTIFACTS), "--output", str(output)]) == 1
    captured = capsys.readouterr()
    assert ranking.SUCCESS_TOKEN not in captured.out + captured.err
    assert output.read_bytes() == saved
    assert (
        ranking.main(
            [str(ARTIFACTS), "--output", str(ARTIFACTS / "forbidden-derived.json")]
        )
        == 1
    )
    assert not (ARTIFACTS / "forbidden-derived.json").exists()
    captured = capsys.readouterr()
    assert ranking.SUCCESS_TOKEN not in captured.out + captured.err
    after = {
        str(p.relative_to(ARTIFACTS)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ARTIFACTS.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_cli_stdout_parseable_and_failure_has_no_success_token(study, tmp_path, capsys):
    assert ranking.main([str(ARTIFACTS)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["authentication"]["diagnostic_count"] == 9
    assert captured.err.strip() == ranking.SUCCESS_TOKEN
    assert ranking.main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert ranking.SUCCESS_TOKEN not in captured.out + captured.err


def test_output_symlink_into_inputs_rejected(study, tmp_path, capsys):
    link = tmp_path / "input-link"
    link.symlink_to(ARTIFACTS, target_is_directory=True)
    assert ranking.main([str(ARTIFACTS), "--output", str(link / "new.json")]) == 1
    captured = capsys.readouterr()
    assert "outside the input" in captured.err
    assert ranking.SUCCESS_TOKEN not in captured.err
