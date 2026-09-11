#!/usr/bin/env python3
"""Render a deterministic NeurIPS report from authenticated evidence.

Existing verifiers rederive raw scientific artifacts. This final layer binds seven
decisive inputs, extracts registered fields, recomputes the claim state, and requires
the Markdown report to match exactly. Synthetic bundles test both decision branches
but can never become empirical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
for _path in (WORKSPACE, PROJECT, WORKSPACE / "imf_dreamer_jax" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA = "trajectory-imf-neurips-report-evidence-v1"
DEFAULT_SCAFFOLD = PROJECT / "NEURIPS_READINESS_REPORT.md"
SELF_TEST_SUCCESS = "TRAJECTORY_IMF_REPORT_SELF_TEST_VERIFIED"
SCAFFOLD_SUCCESS = "TRAJECTORY_IMF_REPORT_SCAFFOLD_VERIFIED"
BUNDLE_SUCCESS = "TRAJECTORY_IMF_REPORT_BUNDLE_VERIFIED"
BINDING_ORDER = ("source", "protocol", "matrix", "analysis", "control", "pixel", "theory")
TRACK_ORDER = ("equal_updates", "equal_compiler_flops")
METRIC_ORDER = ("normalized_free_running_rollout_error_auc", "real_environment_normalized_return_iqm")
ARM_ORDER = ("shortcut_forcing", "trajectory_imf")
CONTROL_FAMILIES = {"shortcut_forcing", "ordinary_imf", "gaussian_rssm",
                    "temporal_increment_imf", "trajectory_endpoint_certificate", "trajectory_imf"}
PRIMARY_NFE = {"shortcut_forcing": 4, "trajectory_imf": 1}
SYNTHETIC_PIXEL_TASKS = ("dmc_walker_walk", "dmc_cheetah_run", "dmc_finger_spin")
SYNTHETIC_PIXEL_SEEDS = (19, 29, 39, 49, 59)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.@+:-]+$")

LIMITATIONS = (
    "The conditional-to-rollout theorem is proved only under its stated endpoint-calibration, coverage, and generated-context assumptions; those assumptions are not established for the trained model.",
    "The empirical gate compares trajectory iMF with shortcut forcing under the frozen protocol; this is not a Dreamer 4 reproduction or a Dreamer 4 performance comparison.",
    "The controls are secondary and descriptive; they cannot access, alter, rescue, or replace the frozen four-interval primary gate.",
    "The hard pixel benchmark is secondary and cannot substitute for the primary state-space gate.",
    "The study does not establish a broad video-generation result.",
    "Aleatoric sampling is not epistemic uncertainty, and robustness to planner exploitation remains unresolved.",
)
FORBIDDEN_DREAMER4 = ("we reproduce dreamer 4", "we reproduced dreamer 4", "reproduces dreamer 4",
                      "matches dreamer 4", "matched dreamer 4", "outperforms dreamer 4", "beats dreamer 4")
FORBIDDEN_VIDEO = ("general video generator", "general-purpose video generator", "stable video generation",
                   "solves video generation", "video-generation system", "video generation system",
                   "generalizes to video generation")
FORBIDDEN_POSITIVE = ("outperforms shortcut forcing", "is superior to shortcut forcing",
                      "beats shortcut forcing", "improves over shortcut forcing",
                      "superiority is established", "superiority claim is supported")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VerificationError(f"non-canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _without_digest(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def file_sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing or linked bound file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(
                               VerificationError(f"non-finite JSON constant: {item}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _map(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _seq(value: Any, label: str) -> Sequence[Any]:
    require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{label} must be an array")
    return value


def _num(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _int(value: Any, label: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _bool(value: Any, label: str) -> bool:
    require(type(value) is bool, f"{label} must be boolean")
    return value


def _name(value: Any, label: str) -> str:
    require(isinstance(value, str) and SAFE_NAME.fullmatch(value) is not None, f"unsafe {label}")
    return value


def _digest(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"invalid {label}")
    return value


def _close(observed: float, expected: float, label: str) -> None:
    require(math.isclose(observed, expected, rel_tol=1e-11, abs_tol=1e-12), f"{label} does not rederive")


def _at(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        require(isinstance(current, Mapping) and key in current,
                f"required evidence field is missing: {'.'.join(path)}")
        current = current[key]
    return current


def _iqm(values: Sequence[float]) -> float:
    ordered = sorted(_num(value, "IQM value") for value in values)
    require(bool(ordered), "IQM is empty")
    lo, hi, total = len(ordered) * .25, len(ordered) * .75, 0.0
    for index, value in enumerate(ordered):
        total += max(0., min(index + 1., hi) - max(float(index), lo)) * value
    return total / (hi - lo)


def _validate_identity(documents: Mapping[str, Mapping[str, Any]]) -> None:
    source, protocol, matrix, analysis = (documents[name] for name in BINDING_ORDER[:4])
    source_sha = _digest(source.get("source_sha256"), "source semantic digest")
    protocol_sha = _digest(matrix.get("protocol_sha256"), "protocol semantic digest")
    matrix_sha = _digest(matrix.get("matrix_sha256"), "matrix semantic digest")
    analysis_sha = _digest(analysis.get("analysis_sha256"), "analysis semantic digest")
    require(source_sha == object_sha256(_without_digest(source, "source_sha256")), "source digest does not rederive")
    require(protocol_sha == object_sha256(protocol), "protocol digest does not rederive")
    require(matrix_sha == object_sha256(_without_digest(matrix, "matrix_sha256")), "matrix digest does not rederive")
    require(analysis_sha == object_sha256(_without_digest(analysis, "analysis_sha256")), "analysis digest does not rederive")
    require(matrix.get("source_sha256") == source_sha and analysis.get("source_sha256") == source_sha,
            "source identity mismatch")
    require(analysis.get("protocol_sha256") == protocol_sha, "protocol identity mismatch")
    require(analysis.get("matrix_sha256") == matrix_sha, "matrix identity mismatch")
    require(matrix.get("profile") == analysis.get("profile") and
            matrix.get("evidence_class") == analysis.get("evidence_class"), "matrix/analysis profile mismatch")


def _core_evidence(core: Mapping[str, Any]) -> dict[str, Any]:
    require(core.get("status") == "complete", "core analysis is incomplete")
    statistical, practical, actor, heterogeneity = [], [], [], []
    satisfied = 0
    for track in TRACK_ORDER:
        metrics = _map(_at(core, "tracks", track), f"{track} metrics")
        require(set(metrics) == set(METRIC_ORDER), f"{track} metric family drifted")
        for metric in METRIC_ORDER:
            row, effect = metrics[metric], _at(core, "practical_significance", "tracks", track, metric)
            arms = _map(_at(row, "arm_iqm"), "arm IQMs")
            require(set(arms) == set(ARM_ORDER), "primary arm family drifted")
            shortcut, trajectory = _num(arms[ARM_ORDER[0]], "shortcut IQM"), _num(arms[ARM_ORDER[1]], "trajectory IQM")
            contrast = _num(_at(row, "contrast"), "contrast")
            _close(contrast, shortcut - trajectory if metric == METRIC_ORDER[0] else trajectory - shortcut,
                   "primary contrast")
            interval = _map(_at(row, "interval"), "adjusted interval")
            require(set(interval) == {"lower", "upper"}, "interval keys drifted")
            lower, upper = _num(interval["lower"], "interval lower"), _num(interval["upper"], "interval upper")
            require(lower <= upper, "interval is reversed")
            passed = lower > 0.; satisfied += int(passed)
            output = {"track": track, "metric": metric, "shortcut_iqm": shortcut,
                      "trajectory_iqm": trajectory, "contrast": contrast, "interval_lower": lower,
                      "interval_upper": upper, "independent_units": _int(_at(row, "independent_units"), "units", 1),
                      "passed": passed}
            statistical.append(output)
            if metric == METRIC_ORDER[1]:
                actor.append({key: output[key] for key in ("track", "shortcut_iqm", "trajectory_iqm",
                                                           "contrast", "interval_lower", "interval_upper")})
            effect_contrast = _num(_at(effect, "absolute_iqm_contrast"), "practical contrast")
            threshold = _num(_at(effect, "minimum_absolute_iqm_contrast"), "practical threshold")
            _close(effect_contrast, contrast, "practical/statistical contrast")
            effect_pass = effect_contrast >= threshold
            effect_output = {"track": track, "metric": metric, "contrast": effect_contrast,
                             "minimum_absolute_contrast": threshold}
            if metric == METRIC_ORDER[0]:
                relative = _num(_at(effect, "relative_iqm_reduction"), "relative effect")
                relative_threshold = _num(_at(effect, "minimum_relative_iqm_reduction"), "relative threshold")
                require(abs(shortcut) > 1e-15, "relative effect has zero comparator")
                _close(relative, contrast / shortcut, "relative practical effect")
                effect_pass = effect_pass and relative >= relative_threshold
                effect_output.update(relative_reduction=relative, minimum_relative_reduction=relative_threshold)
            require(_bool(_at(effect, "passed"), "practical pass") == effect_pass,
                    "practical pass does not rederive")
            effect_output["passed"] = effect_pass; practical.append(effect_output)
        tasks = _map(_at(core, "practical_significance", "per_task", track), "per-task evidence")
        require(bool(tasks), f"{track} per-task heterogeneity is empty")
        for task in sorted(tasks):
            task_metrics = _map(tasks[task], "per-task metrics"); _name(task, "task")
            require(set(task_metrics) == set(METRIC_ORDER), "per-task metric family drifted")
            for metric in METRIC_ORDER:
                row = task_metrics[metric]
                require(bool(_seq(_at(row, "paired_seed_contrasts"), "paired seed contrasts")),
                        "paired seed contrasts are empty")
                heterogeneity.append({"track": track, "task": task, "metric": metric,
                    "paired_seed_contrast_iqm": _num(_at(row, "paired_seed_contrast_iqm"), "task IQM"),
                    "paired_seed_contrast_mean": _num(_at(row, "paired_seed_contrast_mean"), "task mean"),
                    "paired_seed_contrast_standard_deviation": _num(
                        _at(row, "paired_seed_contrast_standard_deviation"), "task SD"),
                    "favorable_seed_fraction": _num(_at(row, "favorable_seed_fraction"), "seed fraction")})

    eligible = core.get("profile") == "confirmatory" and core.get("evidence_class") == "confirmatory"
    statistical_pass = eligible and satisfied == 4
    stored = _map(_at(core, "superiority"), "stored superiority")
    require(_int(_at(stored, "required_interval_count"), "required intervals") == 4 and
            _int(_at(stored, "satisfied_interval_count"), "satisfied intervals") == satisfied,
            "stored interval counts do not rederive")
    require(_bool(_at(stored, "passed"), "statistical gate") == statistical_pass,
            "stored statistical gate does not rederive")
    practical_pass = all(row["passed"] for row in practical)
    practical_root = _at(core, "practical_significance")
    require(_bool(_at(practical_root, "all_thresholds_met"), "practical aggregate") == practical_pass and
            _bool(_at(practical_root, "statistical_superiority_passed"), "statistical dependency") == statistical_pass and
            _bool(_at(practical_root, "practically_meaningful_superiority_claim_allowed"), "claim permission")
            == (statistical_pass and practical_pass), "aggregate practical gate does not rederive")

    calibration = []
    horizons = [_int(value, "rollout horizon", 1) for value in _seq(_at(core, "rollout_horizons"), "horizons")]
    require(horizons == sorted(set(horizons)) and horizons, "rollout horizons drifted")
    primary_nfe = _map(_at(core, "primary_nfe"), "primary NFE")
    require(set(primary_nfe) == set(ARM_ORDER), "primary-NFE arm family drifted")
    for track in TRACK_ORDER:
        for arm in ARM_ORDER:
            nfe = _int(primary_nfe[arm], "primary NFE", 1)
            primary = _at(core, "secondary_rollout", "tracks", track, arm, str(nfe))
            auc, rows = _num(_at(primary, "normalized_rollout_error_auc_iqm"), "rollout AUC"), _at(primary, "per_horizon")
            require(set(rows) == {str(value) for value in horizons}, "calibration horizon family drifted")
            for horizon in horizons:
                row = rows[str(horizon)]
                coverage = _num(_at(row, "central_90_percent_interval_coverage_iqm"), "coverage")
                error = _num(_at(row, "central_90_percent_interval_calibration_absolute_error_iqm"), "calibration error")
                width = _num(_at(row, "central_90_percent_interval_width_iqm"), "interval width")
                require(0 <= coverage <= 1 and error >= 0 and width >= 0, "calibration domain error")
                _close(error, abs(coverage - .9), "calibration error")
                calibration.append({"track": track, "arm": arm, "nfe": nfe, "horizon": horizon,
                    "rollout_auc_iqm": auc, "central_90_coverage_iqm": coverage,
                    "central_90_calibration_absolute_error_iqm": error,
                    "central_90_interval_width_iqm": width})

    compute = []
    for arm in ARM_ORDER:
        resource = _at(core, "compute_resources", "arms", arm)
        inference = _at(resource, "inference")
        require(set(inference) == {"1", "2", "4"}, "inference NFE frontier drifted")
        world = _num(_at(resource, "compiler_flops", "world_forward_and_backward_train_mean"), "world train FLOPs")
        actor_flops = _num(_at(resource, "compiler_flops", "actor_forward_and_backward_train_mean"), "actor train FLOPs")
        parameters = _seq(_at(resource, "parameters_unique_values", "world_model_active"), "parameter counts")
        require(len(parameters) == 1, "active parameter count is not unique")
        for nfe in (1, 2, 4):
            row = inference[str(nfe)]
            require(_int(_at(row, "structural_nfe_per_transition"), "structural NFE", 1) == nfe,
                    "structural NFE does not match frontier")
            compute.append({"arm": arm, "nfe": nfe, "is_primary": nfe == primary_nfe[arm],
                "compiler_forward_flops_per_transition": _num(
                    _at(row, "compiler_forward_flops_per_transition_mean"), "inference FLOPs"),
                "world_train_flops_per_update": world, "actor_train_flops_per_update": actor_flops,
                "active_world_parameters": _int(parameters[0], "active parameters", 1)})
    confidence = _num(_at(core, "bootstrap", "interval_confidence"), "interval confidence")
    if core.get("profile") == "confirmatory":
        _close(confidence, .9875, "confirmatory adjusted confidence")
    return {"profile": _name(core.get("profile"), "core profile"),
            "evidence_class": _name(core.get("evidence_class"), "core evidence class"),
            "adjusted_interval_confidence": confidence, "statistical_gate_passed": statistical_pass,
            "practical_gate_passed": practical_pass, "statistical_intervals": statistical,
            "practical_effects": practical, "actor_returns": actor,
            "per_task_heterogeneity": heterogeneity, "calibration": calibration, "compute_nfe": compute}


def _control_evidence(control: Mapping[str, Any]) -> dict[str, Any]:
    require(control.get("status") == "complete", "controls analysis is incomplete")
    require(control.get("claim_eligible") is False and control.get("primary_gate_accessed") is False and
            control.get("inference_role") == "descriptive_only" and control.get("superiority_decision") is None and
            control.get("multiplicity_adjusted_decision") is None, "controls accessed or decided the primary gate")
    digest = _digest(control.get("analysis_sha256"), "control semantic digest")
    require(digest == object_sha256(_without_digest(control, "analysis_sha256")),
            "control semantic digest does not rederive")
    rows = []
    summaries = _at(control, "summaries")
    require(set(summaries) == set(TRACK_ORDER), "control track family drifted")
    for track in TRACK_ORDER:
        families = set()
        for run_id in sorted(summaries[track]):
            row, family = summaries[track][run_id], _name(_at(summaries[track][run_id], "family"), "control family")
            families.add(family)
            rows.append({"track": track, "run_id": _name(run_id, "control run"), "family": family,
                         "rollout_auc_iqm": _num(_at(row, "rollout_auc_iqm"), "control rollout AUC"),
                         "actor_return_iqm": _num(_at(row, "actor_return_iqm"), "control actor return"),
                         "independent_units": _int(_at(row, "unit_count"), "control units", 1)})
        require(CONTROL_FAMILIES <= families,
                f"required control families are missing from {track}: "
                f"{sorted(CONTROL_FAMILIES - families)}")
    return {"profile": _name(control.get("profile"), "control profile"),
            "evidence_class": _name(control.get("evidence_class"), "control evidence class"),
            "decision_role": "secondary_descriptive_only", "rows": rows}


def _pixel_evidence(pixel: Mapping[str, Any], mode: str) -> dict[str, Any]:
    require(pixel.get("schema_version") in {"trajectory-imf-pixel-aggregate-v1", "synthetic-pixel-aggregate-v1"},
            "pixel aggregate schema is invalid")
    require(pixel.get("status") == "complete" and pixel.get("claim_eligible_for_primary_gate") is False and
            pixel.get("primary_gate_substitution") is False, "pixel claim boundary is invalid")
    digest = _digest(pixel.get("analysis_sha256"), "pixel semantic digest")
    require(digest == object_sha256(_without_digest(pixel, "analysis_sha256")),
            "pixel semantic digest does not rederive")
    if mode == "synthetic_validation":
        horizons = [_int(value, "pixel horizon", 1) for value in _at(pixel, "horizons")]
        expected_tasks, expected_seeds = list(SYNTHETIC_PIXEL_TASKS), list(SYNTHETIC_PIXEL_SEEDS)
    else:
        hard = _at(read_json(PROJECT / "pixel_benchmark_protocol.json"), "profiles", "hard")
        horizons = [_int(value, "pixel horizon", 1) for value in _at(hard, "horizons")]
        expected_tasks = [_name(value, "registered pixel task") for value in _at(hard, "tasks")]
        expected_seeds = [_int(value, "registered pixel seed") for value in _at(hard, "seeds")]
    require(horizons == sorted(set(horizons)) and horizons[-1] >= 60,
            "hard pixel aggregate does not reach the registered long horizon")
    tracks = _at(pixel, "tracks")
    require(set(tracks) == set(TRACK_ORDER), "pixel aggregate must contain both compute tracks")
    rows, contrasts, identity = [], [], None
    for track in TRACK_ORDER:
        item = tracks[track]
        tasks = [_name(value, "pixel task") for value in _at(item, "task_order")]
        seeds = [_int(value, "pixel seed") for value in _at(item, "seed_order")]
        require(tasks == expected_tasks and seeds == expected_seeds,
                f"{track} pixel task/seed identity differs from the registered matrix")
        require(identity in (None, (tasks, seeds)), "pixel tracks use different task/seed matrices")
        identity = (tasks, seeds)
        matrix = [[_num(value, "pixel contrast") for value in row]
                  for row in _at(item, "task_by_seed_primary_delta")]
        require(len(matrix) == len(tasks) and
                all(len(row) == len(seeds) for row in matrix), "pixel task/seed matrix is incomplete")
        contrast = _num(_at(item, "primary_visual_delta_shortcut_minus_trajectory"), "pixel contrast")
        _close(contrast, _iqm([value for row in matrix for value in row]), "pixel primary contrast")
        interval = [_num(value, "pixel interval") for value in _at(item, "descriptive_hierarchical_bootstrap_interval")]
        require(len(interval) == 2 and interval[0] <= interval[1] and
                item.get("positive_favors_trajectory_imf") is True, "pixel interval/direction is invalid")
        contrasts.append({"track": track, "contrast": contrast, "interval_lower": interval[0],
                          "interval_upper": interval[1], "task_count": len(tasks), "seed_count": len(seeds)})
        frontier = _at(item, "frontier_iqm")
        require(set(frontier) == set(ARM_ORDER), "pixel arm family drifted")
        for arm in ARM_ORDER:
            require(set(frontier[arm]) == {"1", "2", "4"}, "pixel NFE frontier drifted")
            for nfe in (1, 2, 4):
                rows.append({"track": track, "arm": arm, "nfe": nfe,
                             "is_primary": nfe == PRIMARY_NFE[arm], "horizons": horizons,
                             "normalized_visual_mse_auc_iqm": _num(frontier[arm][str(nfe)], "pixel IQM")})
    return {"profile": "hard", "claim_role": "secondary_hard_visual_only",
            "primary_contrasts": contrasts, "rows": rows}


def _theory_evidence(path: Path, mode: str) -> dict[str, Any]:
    if mode == "synthetic_validation":
        expected = {"schema_version": "synthetic-theory-v1", "conditional_rollout_theorem_proved": True,
                    "trained_model_assumptions_established": False, "fixed_policy_only": True,
                    "numerical_controls_passed": True}
        require(read_json(path) == expected, "synthetic theory changed its conditional claim boundary")
        numeric_sha = object_sha256({"synthetic": "passed"})
    else:
        from scripts.verify_trajectory_imf_theory import run_numerics, verify_document
        verify_document(path, PROJECT / "matched_objective_protocol.json",
                        WORKSPACE / "imf_dreamer_jax" / "src" / "imf_dreamer_jax" / "trajectory.py")
        numeric_sha = object_sha256(run_numerics())
    return {"status": "proved_under_stated_assumptions",
            "trained_model_assumptions_established": False, "fixed_policy_only": True,
            "numerical_controls_sha256": numeric_sha}


def derive_evidence(bindings: Mapping[str, Mapping[str, str]],
                    documents: Mapping[str, Mapping[str, Any]], mode: str) -> dict[str, Any]:
    require(mode in {"authenticated_runs", "synthetic_validation"}, "invalid evidence mode")
    _validate_identity(documents)
    core, controls = _core_evidence(documents["analysis"]), _control_evidence(documents["control"])
    pixel = _pixel_evidence(documents["pixel"], mode)
    gate_math = core["statistical_gate_passed"] and core["practical_gate_passed"]
    profile_ready = (documents["matrix"].get("claim_eligible") is True and
                     core["profile"] == "confirmatory" and core["evidence_class"] == "confirmatory" and
                     controls["profile"] == "confirmatory" and
                     controls["evidence_class"] == "confirmatory_secondary_descriptive")
    registered_profiles_complete = mode == "authenticated_runs" and profile_ready
    registered_gate_passed = registered_profiles_complete and gate_math
    state = (("synthetic_positive_control" if gate_math else "synthetic_valid_negative_control")
             if mode == "synthetic_validation" else
             ("nonfinal_incomplete_evidence" if not registered_profiles_complete else
              ("registered_claim_supported" if registered_gate_passed else "registered_claim_not_supported")))
    semantic = {"source": documents["source"]["source_sha256"], "protocol": documents["matrix"]["protocol_sha256"],
                "matrix": documents["analysis"]["matrix_sha256"], "analysis": documents["analysis"]["analysis_sha256"],
                "control": documents["control"]["analysis_sha256"], "pixel": documents["pixel"]["analysis_sha256"],
                "theory": bindings["theory"]["sha256"]}
    digests = {name: {"file_sha256": _digest(bindings[name]["sha256"], f"{name} file digest"),
                      "semantic_sha256": _digest(semantic[name], f"{name} semantic digest")}
               for name in BINDING_ORDER}
    return {"decision": {"claim_state": state, "evidence_mode": mode, "profile": core["profile"],
                          "evidence_class": core["evidence_class"],
                          "registered_profiles_complete": registered_profiles_complete,
                          "statistical_gate_passed": core["statistical_gate_passed"],
                          "practical_gate_passed": core["practical_gate_passed"],
                          "gate_math_passed": gate_math,
                          "registered_gate_passed": registered_gate_passed,
                          "empirical_claim_allowed": registered_gate_passed,
                          "adjusted_interval_confidence": core["adjusted_interval_confidence"]},
            "digests": digests, "statistical_intervals": core["statistical_intervals"],
            "practical_effects": core["practical_effects"], "actor_returns": core["actor_returns"],
            "per_task_heterogeneity": core["per_task_heterogeneity"], "calibration": core["calibration"],
            "compute_nfe": core["compute_nfe"], "controls": controls, "pixel": pixel,
            "theory": _theory_evidence(Path(bindings["theory"]["path"]), mode),
            "limitations": list(LIMITATIONS)}


def _paths(bundle: Mapping[str, Any]) -> dict[str, Path]:
    bindings = _map(bundle.get("bindings"), "bindings")
    require(set(bindings) == set(BINDING_ORDER), "bundle binding set is not exact")
    output = {}
    for name in BINDING_ORDER:
        binding = _map(bindings[name], f"{name} binding")
        require(set(binding) == {"path", "sha256"}, f"{name} binding keys drifted")
        path = Path(binding["path"])
        require(path.is_absolute() and path.resolve() == path, f"{name} path is not canonical")
        _digest(binding["sha256"], f"{name} binding digest")
        output[name] = path
    return output


def _layout(bundle: Mapping[str, Any]) -> None:
    paths, roots, mode = _paths(bundle), _map(bundle.get("roots"), "roots"), bundle.get("evidence_mode")
    if mode == "synthetic_validation":
        require(set(roots) == {"fixture"}, "synthetic root set is invalid")
        root = Path(roots["fixture"])
        require(root.is_absolute() and root.resolve() == root and all(path.parent == root for path in paths.values()),
                "synthetic binding escaped its fixture root")
        return
    require(mode == "authenticated_runs" and set(roots) == {"core", "controls", "pixel"},
            "authenticated root set is invalid")
    core, controls, pixel = (Path(roots[name]) for name in ("core", "controls", "pixel"))
    require(all(path.is_absolute() and path.resolve() == path for path in (core, controls, pixel)),
            "authenticated roots are not canonical")
    expected = {"source": core / "source_manifest.json", "protocol": core / "frozen_protocol.json",
                "matrix": core / "matrix.json", "analysis": core / "analysis.json",
                "control": controls / "analysis.json", "pixel": pixel,
                "theory": PROJECT / "TRAJECTORY_IMF_THEORY.md"}
    require(paths == expected, "authenticated bindings do not match the fixed layout")


def _documents(bundle: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    output = {}
    for name, path in _paths(bundle).items():
        require(file_sha256(path) == bundle["bindings"][name]["sha256"], f"{name} digest drift")
        output[name] = {} if name == "theory" and bundle["evidence_mode"] == "authenticated_runs" else read_json(path)
    return output


def _authenticate(bundle: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]]) -> None:
    if bundle["evidence_mode"] != "authenticated_runs":
        return
    from dreamer_imf_compare.matched_objective_benchmark import verify_output_root
    from dreamer_imf_compare.neurips_controls import verify_controls_output
    from dreamer_imf_compare.pixel_benchmark import verify_aggregate
    roots = {name: Path(value) for name, value in bundle["roots"].items()}
    core, controls = verify_output_root(roots["core"], workspace=WORKSPACE), verify_controls_output(
        roots["controls"], workspace=WORKSPACE)
    require(core["analysis_sha256"] == documents["analysis"]["analysis_sha256"] and
            controls["analysis_sha256"] == documents["control"]["analysis_sha256"],
            "upstream verifier identity mismatch")
    pixel_inputs = [row["path"] for row in _seq(documents["pixel"].get("inputs"), "pixel inputs")]
    verified_pixel = verify_aggregate(roots["pixel"], pixel_inputs)
    require(canonical_bytes(verified_pixel) == canonical_bytes(documents["pixel"]),
            "pixel aggregate returned by its verifier differs from the bound aggregate")
    control_matrix = read_json(roots["controls"] / "matrix.json")
    require(control_matrix.get("parent_source_sha256") == documents["source"].get("source_sha256") and
            control_matrix.get("parent_protocol_sha256") == documents["matrix"].get("protocol_sha256"),
            "controls do not share the bound source/protocol")


def validate_report_language(text: str, state: str) -> None:
    lower = text.lower()
    for phrase in FORBIDDEN_DREAMER4:
        require(phrase not in lower, f"unsupported Dreamer 4 claim: {phrase}")
    for phrase in FORBIDDEN_VIDEO:
        require(phrase not in lower, f"unsupported video-generation claim: {phrase}")
    if state != "registered_claim_supported":
        for phrase in FORBIDDEN_POSITIVE:
            require(phrase not in lower, f"unsupported positive comparison wording: {phrase}")


def _fmt(value: float) -> str:
    return format(value, ".9g")


def _table(lines: list[str], title: str, headings: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    lines += ["", title, "", "| " + " | ".join(headings) + " |",
              "|" + "|".join("---" for _ in headings) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]


def render_report(bundle: Mapping[str, Any]) -> str:
    evidence, lines = bundle["evidence"], ["# Trajectory-iMF NeurIPS evidence report", ""]
    state = evidence["decision"]["claim_state"]
    statuses = {"registered_claim_supported": "REGISTERED GATES PASSED — trajectory iMF versus shortcut forcing only",
                "registered_claim_not_supported": "REGISTERED CLAIM NOT SUPPORTED",
                "nonfinal_incomplete_evidence": "NONFINAL — confirmatory evidence incomplete",
                "synthetic_positive_control": "SYNTHETIC POSITIVE CONTROL — NOT EMPIRICAL",
                "synthetic_valid_negative_control": "SYNTHETIC VALID-NEGATIVE CONTROL — NOT EMPIRICAL"}
    require(state in statuses, "unknown claim state")
    lines += [f"Status: **{statuses[state]}**", "", f"Evidence bundle SHA-256: `{bundle['bundle_sha256']}`.", ""]
    if state.startswith("synthetic_"):
        lines += ["This is an executable reporting control; its synthetic numbers support no empirical comparison.", ""]
    elif state == "registered_claim_supported":
        lines += ["All frozen statistical and practical gates pass within the registered comparison.", ""]
    elif state == "registered_claim_not_supported":
        lines += ["The registered comparison does not support the trajectory-iMF superiority claim.", ""]
    else:
        lines += ["No empirical conclusion is permitted from this incomplete evidence set.", ""]

    _table(lines, "## Authenticated bindings", ("Artifact", "File SHA-256", "Semantic SHA-256"),
           [(f"`{name}`", f"`{evidence['digests'][name]['file_sha256']}`",
             f"`{evidence['digests'][name]['semantic_sha256']}`") for name in BINDING_ORDER])
    practical = {(row["track"], row["metric"]): row for row in evidence["practical_effects"]}
    _table(lines, "## Statistical and practical gates",
           ("Track", "Estimand", "Shortcut IQM", "Trajectory IQM", "Contrast", "Adjusted interval", "Statistical", "Practical"),
           [(f"`{r['track']}`", f"`{r['metric']}`", _fmt(r["shortcut_iqm"]), _fmt(r["trajectory_iqm"]),
             _fmt(r["contrast"]), f"[{_fmt(r['interval_lower'])}, {_fmt(r['interval_upper'])}]",
             "PASS" if r["passed"] else "FAIL", "PASS" if practical[(r["track"], r["metric"])]["passed"] else "FAIL")
            for r in evidence["statistical_intervals"]])
    _table(lines, "### Practical-effect magnitudes",
           ("Track", "Estimand", "Contrast / absolute threshold", "Relative reduction / threshold"),
           [(f"`{r['track']}`", f"`{r['metric']}`", f"{_fmt(r['contrast'])} / {_fmt(r['minimum_absolute_contrast'])}",
             f"{_fmt(r['relative_reduction'])} / {_fmt(r['minimum_relative_reduction'])}" if "relative_reduction" in r else "n/a")
            for r in evidence["practical_effects"]])
    _table(lines, "## Actor returns", ("Track", "Shortcut IQM", "Trajectory IQM", "Contrast", "Interval"),
           [(f"`{r['track']}`", _fmt(r["shortcut_iqm"]), _fmt(r["trajectory_iqm"]), _fmt(r["contrast"]),
             f"[{_fmt(r['interval_lower'])}, {_fmt(r['interval_upper'])}]") for r in evidence["actor_returns"]])
    _table(lines, "## Per-task heterogeneity", ("Track", "Task", "Estimand", "Task IQM", "Mean", "SD", "Favorable seeds"),
           [(f"`{r['track']}`", f"`{r['task']}`", f"`{r['metric']}`", _fmt(r["paired_seed_contrast_iqm"]),
             _fmt(r["paired_seed_contrast_mean"]), _fmt(r["paired_seed_contrast_standard_deviation"]),
             _fmt(r["favorable_seed_fraction"])) for r in evidence["per_task_heterogeneity"]])
    _table(lines, "## Calibration", ("Track", "Arm", "NFE", "Horizon", "AUC", "Coverage", "Error", "Width"),
           [(f"`{r['track']}`", f"`{r['arm']}`", str(r["nfe"]), str(r["horizon"]), _fmt(r["rollout_auc_iqm"]),
             _fmt(r["central_90_coverage_iqm"]), _fmt(r["central_90_calibration_absolute_error_iqm"]),
             _fmt(r["central_90_interval_width_iqm"])) for r in evidence["calibration"]])
    _table(lines, "## Compute and NFE",
           ("Arm", "NFE", "Primary", "Forward FLOPs", "World train FLOPs", "Actor train FLOPs", "Parameters"),
           [(f"`{r['arm']}`", str(r["nfe"]), "yes" if r["is_primary"] else "no",
             _fmt(r["compiler_forward_flops_per_transition"]), _fmt(r["world_train_flops_per_update"]),
             _fmt(r["actor_train_flops_per_update"]), str(r["active_world_parameters"])) for r in evidence["compute_nfe"]])
    _table(lines, "## Strong controls (descriptive only)",
           ("Track", "Run", "Family", "Rollout AUC", "Actor return", "Units"),
           [(f"`{r['track']}`", f"`{r['run_id']}`", f"`{r['family']}`", _fmt(r["rollout_auc_iqm"]),
             _fmt(r["actor_return_iqm"]), str(r["independent_units"])) for r in evidence["controls"]["rows"]])
    _table(lines, "## Hard pixel aggregate (secondary)",
           ("Track", "Primary favorable contrast", "Descriptive interval", "Tasks", "Seeds"),
           [(f"`{r['track']}`", _fmt(r["contrast"]), f"[{_fmt(r['interval_lower'])}, {_fmt(r['interval_upper'])}]",
             str(r["task_count"]), str(r["seed_count"])) for r in evidence["pixel"]["primary_contrasts"]])
    _table(lines, "### Pixel NFE frontier", ("Track", "Arm", "NFE", "Primary", "Horizons", "Visual MSE AUC IQM"),
           [(f"`{r['track']}`", f"`{r['arm']}`", str(r["nfe"]), "yes" if r["is_primary"] else "no",
             ", ".join(map(str, r["horizons"])), _fmt(r["normalized_visual_mse_auc_iqm"]))
            for r in evidence["pixel"]["rows"]])
    theory = evidence["theory"]
    lines += ["", "## Mathematics", "", f"Theorem: `{theory['status']}`; trained-model assumptions established: "
              f"`{str(theory['trained_model_assumptions_established']).lower()}`; fixed-policy only: "
              f"`{str(theory['fixed_policy_only']).lower()}`; numerical SHA-256: `{theory['numerical_controls_sha256']}`.",
              "", "## Limitations", ""]
    require(evidence["limitations"] == list(LIMITATIONS), "limitations are incomplete")
    lines += [f"- {value}" for value in evidence["limitations"]] + [""]
    output = "\n".join(lines); validate_report_language(output, state)
    return output


def _atomic(path: Path, data: bytes) -> None:
    require(not path.is_symlink(), f"refusing linked output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def create_bundle(binding_paths: Mapping[str, Path], roots: Mapping[str, Path], mode: str) -> dict[str, Any]:
    require(set(binding_paths) == set(BINDING_ORDER), "creation binding set is not exact")
    bindings = {name: {"path": str(binding_paths[name].resolve()), "sha256": file_sha256(binding_paths[name].resolve())}
                for name in BINDING_ORDER}
    body = {"schema_version": SCHEMA, "status": "complete", "evidence_mode": mode,
            "roots": {name: str(path.resolve()) for name, path in roots.items()}, "bindings": bindings, "evidence": {}}
    placeholder = {**body, "bundle_sha256": "0" * 64}
    _layout(placeholder); documents = _documents(placeholder); _authenticate(placeholder, documents)
    body["evidence"] = derive_evidence(bindings, documents, mode)
    bundle = {**body, "bundle_sha256": object_sha256(body)}; render_report(bundle)
    return bundle


def verify_bundle(bundle_path: Path, report_path: Path) -> dict[str, Any]:
    require(bundle_path.is_file() and not bundle_path.is_symlink() and report_path.is_file() and
            not report_path.is_symlink(), "bundle/report is missing or linked")
    bundle = read_json(bundle_path)
    require(set(bundle) == {"schema_version", "status", "evidence_mode", "roots", "bindings", "evidence",
                            "bundle_sha256"} and bundle.get("schema_version") == SCHEMA and
            bundle.get("status") == "complete", "bundle schema/status is invalid")
    require(bundle.get("bundle_sha256") == object_sha256(_without_digest(bundle, "bundle_sha256")),
            "bundle digest does not rederive")
    _layout(bundle); before = {name: file_sha256(path) for name, path in _paths(bundle).items()}
    documents = _documents(bundle); _authenticate(bundle, documents)
    evidence = derive_evidence(bundle["bindings"], documents, bundle["evidence_mode"])
    require(canonical_bytes(evidence) == canonical_bytes(bundle["evidence"]), "stored report evidence does not rederive")
    require(before == {name: file_sha256(path) for name, path in _paths(bundle).items()},
            "bound inputs changed during verification")
    observed = report_path.read_text(encoding="utf-8"); validate_report_language(
        observed, evidence["decision"]["claim_state"])
    require(observed == render_report({**bundle, "evidence": evidence}), "human report does not exactly rederive")
    return evidence


def render_scaffold() -> str:
    return """# Trajectory-iMF NeurIPS readiness report

Status: **NONFINAL — awaiting authenticated confirmatory evidence**

This checked-in document is a pre-result scaffold. It contains no empirical superiority claim and must not be cited as a completed result. The final report is generated only by `scripts/verify_neurips_report.py` from a verified evidence bundle.

## Evidence required before a final report

- A finalized confirmatory core analysis with all four multiplicity-adjusted intervals, the registered practical-effect thresholds, actor returns, per-task heterogeneity, calibration, and matched compute/NFE evidence.
- Finalized descriptive strong-control evidence covering shortcut forcing, ordinary iMF, Gaussian RSSM, temporal-increment iMF, endpoint-certificate, and trajectory-iMF arms.
- An authenticated hard pixel rollout result across the registered NFE frontier.
- The verified conditional-to-rollout theorem note and its numerical controls.
- Exact SHA-256 bindings for source, protocol, matrix, analysis, controls, pixel evidence, and theory.

## Claim boundary

The intended primary comparison is trajectory iMF versus shortcut forcing under the frozen protocol. This is not a Dreamer 4 reproduction or a Dreamer 4 performance comparison. The pixel study is secondary, not a replacement for the primary gate, and the project does not establish a broad video-generation result. The theorem remains conditional, aleatoric sampling is not epistemic uncertainty, and planner exploitation remains unresolved.
"""


def verify_scaffold(path: Path = DEFAULT_SCAFFOLD) -> None:
    require(path.is_file() and not path.is_symlink() and path.read_text(encoding="utf-8") == render_scaffold(),
            "checked-in scaffold is not the exact nonfinal template")
    validate_report_language(render_scaffold(), "nonfinal_incomplete_evidence")


def _synthetic_documents(positive: bool) -> dict[str, Any]:
    source_body = {"schema_version": "synthetic-source-v1", "files": ["model.py"]}
    source = {**source_body, "source_sha256": object_sha256(source_body)}
    protocol = {"schema_version": "synthetic-protocol-v1", "status": "frozen_before_outcomes"}
    matrix_body = {"schema_version": "synthetic-matrix-v1", "profile": "confirmatory",
                   "evidence_class": "confirmatory", "claim_eligible": True,
                   "source_sha256": source["source_sha256"], "protocol_sha256": object_sha256(protocol)}
    matrix = {**matrix_body, "matrix_sha256": object_sha256(matrix_body)}
    tracks, practical, per_task = {}, {}, {}
    for track_index, track in enumerate(TRACK_ORDER):
        rollout, actor = ((1., .8), (.4, .5)) if positive else ((1., 1.1), (.4, .35))
        tracks[track], practical[track], per_task[track] = {}, {}, {}
        for metric, arms in zip(METRIC_ORDER, (rollout, actor)):
            contrast = arms[0] - arms[1] if metric == METRIC_ORDER[0] else arms[1] - arms[0]
            tracks[track][metric] = {"arm_iqm": dict(zip(ARM_ORDER, arms)), "contrast": contrast,
                                     "interval": {"lower": .02 if positive else -.2,
                                                  "upper": .3 if positive else -.01},
                                     "independent_units": 20}
            practical[track][metric] = {"absolute_iqm_contrast": contrast,
                                        "minimum_absolute_iqm_contrast": .05, "passed": positive}
            if metric == METRIC_ORDER[0]:
                practical[track][metric].update(relative_iqm_reduction=contrast / arms[0],
                                                 minimum_relative_iqm_reduction=.1)
        for task_index, task in enumerate(("dmc_cartpole_swingup", "dmc_walker_walk")):
            per_task[track][task] = {}
            for metric_index, metric in enumerate(METRIC_ORDER):
                sign = 1 if positive else -1
                values = [sign * (.1 + .01 * track_index + .01 * task_index + .01 * metric_index),
                          sign * (.12 + .01 * track_index + .01 * task_index + .01 * metric_index)]
                per_task[track][task][metric] = {"paired_seed_contrasts": values,
                    "paired_seed_contrast_iqm": _iqm(values), "paired_seed_contrast_mean": sum(values) / 2,
                    "paired_seed_contrast_standard_deviation": abs(values[1] - values[0]) / 2,
                    "favorable_seed_fraction": sum(value > 0 for value in values) / 2}
    horizons, secondary = [1, 5, 15, 30], {"tracks": {}}
    for track in TRACK_ORDER:
        secondary["tracks"][track] = {}
        for arm in ARM_ORDER:
            coverage = .88 if arm == ARM_ORDER[1] else .84
            secondary["tracks"][track][arm] = {str(nfe): {
                "normalized_rollout_error_auc_iqm": .8 if arm == ARM_ORDER[1] else 1.,
                "per_horizon": {str(h): {"central_90_percent_interval_coverage_iqm": coverage,
                    "central_90_percent_interval_calibration_absolute_error_iqm": abs(coverage - .9),
                    "central_90_percent_interval_width_iqm": .2 + h / 100} for h in horizons}}
                for nfe in (1, 2, 4)}
    compute = {arm: {"compiler_flops": {"world_forward_and_backward_train_mean": 1e6 + i * 1e5,
                                         "actor_forward_and_backward_train_mean": 5e5 + i * 5e4},
                     "parameters_unique_values": {"world_model_active": [100000 + i * 1000]},
                     "inference": {str(nfe): {"structural_nfe_per_transition": nfe,
                                               "compiler_forward_flops_per_transition_mean": 1e4 + nfe * 1e3}
                                   for nfe in (1, 2, 4)}} for i, arm in enumerate(ARM_ORDER)}
    analysis_body = {"schema_version": "matched-objective-analysis-v1", "status": "complete",
        "profile": "confirmatory", "evidence_class": "confirmatory",
        "source_sha256": source["source_sha256"], "protocol_sha256": matrix["protocol_sha256"],
        "matrix_sha256": matrix["matrix_sha256"], "bootstrap": {"interval_confidence": .9875},
        "superiority": {"required_interval_count": 4, "satisfied_interval_count": 4 if positive else 0,
                        "passed": positive}, "tracks": tracks,
        "practical_significance": {"tracks": practical, "all_thresholds_met": positive,
            "statistical_superiority_passed": positive,
            "practically_meaningful_superiority_claim_allowed": positive, "per_task": per_task},
        "primary_nfe": PRIMARY_NFE, "rollout_horizons": horizons, "secondary_rollout": secondary,
        "compute_resources": {"arms": compute}}
    analysis = {**analysis_body, "analysis_sha256": object_sha256(analysis_body)}
    run_specs = (("shortcut_forcing", "shortcut_forcing"), ("ordinary_imf@fixed", "ordinary_imf"),
                 ("gaussian_rssm@fixed", "gaussian_rssm"),
                 ("temporal_increment_imf@fixed", "temporal_increment_imf"),
                 ("trajectory_endpoint_certificate", "trajectory_endpoint_certificate"),
                 ("trajectory_imf", "trajectory_imf"))
    summaries = {track: {run: {"family": family, "rollout_auc_iqm": 1.2 - i * .05,
                                "actor_return_iqm": .3 + i * .02, "unit_count": 20}
                         for i, (run, family) in enumerate(run_specs)} for track in TRACK_ORDER}
    control_body = {"schema_version": "trajectory-imf-neurips-controls-analysis-v1", "status": "complete",
                    "profile": "confirmatory", "evidence_class": "confirmatory_secondary_descriptive",
                    "claim_eligible": False, "primary_gate_accessed": False,
                    "inference_role": "descriptive_only", "superiority_decision": None,
                    "multiplicity_adjusted_decision": None, "summaries": summaries}
    control = {**control_body, "analysis_sha256": object_sha256(control_body)}
    pixel_tracks = {}
    for track in TRACK_ORDER:
        delta = [[.2 + .01 * row + .001 * col for col in range(5)] for row in range(3)]
        pixel_tracks[track] = {"primary_visual_delta_shortcut_minus_trajectory": _iqm(sum(delta, [])),
            "positive_favors_trajectory_imf": True, "descriptive_hierarchical_bootstrap_interval": [.1, .3],
            "task_by_seed_primary_delta": delta,
            "task_order": ["dmc_walker_walk", "dmc_cheetah_run", "dmc_finger_spin"],
            "seed_order": [19, 29, 39, 49, 59],
            "frontier_iqm": {arm: {str(nfe): .9 + i * .1 - nfe * .01 for nfe in (1, 2, 4)}
                             for i, arm in enumerate(ARM_ORDER)}}
    pixel_body = {"schema_version": "synthetic-pixel-aggregate-v1", "status": "complete",
                  "claim_eligible_for_primary_gate": False, "primary_gate_substitution": False,
                  "horizons": [1, 2, 4, 8, 15, 30, 60], "tracks": pixel_tracks}
    pixel = {**pixel_body, "analysis_sha256": object_sha256(pixel_body)}
    theory = {"schema_version": "synthetic-theory-v1", "conditional_rollout_theorem_proved": True,
              "trained_model_assumptions_established": False, "fixed_policy_only": True,
              "numerical_controls_passed": True}
    return {"source": source, "protocol": protocol, "matrix": matrix, "analysis": analysis,
            "control": control, "pixel": pixel, "theory": theory}


def create_synthetic_bundle(root: Path, positive: bool) -> tuple[Path, Path]:
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / f"{name}.json" for name in BINDING_ORDER}
    for name, value in _synthetic_documents(positive).items():
        write_json_atomic(paths[name], value)
    bundle = create_bundle(paths, {"fixture": root}, "synthetic_validation")
    bundle_path, report_path = root / "evidence_bundle.json", root / "REPORT.md"
    write_json_atomic(bundle_path, bundle); _atomic(report_path, render_report(bundle).encode())
    verify_bundle(bundle_path, report_path)
    return bundle_path, report_path


def build_authenticated_bundle(core: Path, controls: Path, pixel_aggregate: Path,
                               bundle_path: Path, report_path: Path) -> dict[str, Any]:
    require(not bundle_path.is_symlink() and not report_path.is_symlink(),
            "refusing linked bundle/report output")
    require(not core.is_symlink() and not controls.is_symlink() and not pixel_aggregate.is_symlink(),
            "refusing linked authenticated input")
    core, controls, pixel_aggregate = core.resolve(), controls.resolve(), pixel_aggregate.resolve()
    paths = {"source": core / "source_manifest.json", "protocol": core / "frozen_protocol.json",
             "matrix": core / "matrix.json", "analysis": core / "analysis.json",
             "control": controls / "analysis.json", "pixel": pixel_aggregate,
             "theory": PROJECT / "TRAJECTORY_IMF_THEORY.md"}
    bundle = create_bundle(paths, {"core": core, "controls": controls, "pixel": pixel_aggregate},
                           "authenticated_runs")
    write_json_atomic(bundle_path, bundle); _atomic(report_path, render_report(bundle).encode())
    return verify_bundle(bundle_path, report_path)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="trajectory-imf-report-") as temporary:
        positive = verify_bundle(*create_synthetic_bundle(Path(temporary) / "positive", True))
        negative = verify_bundle(*create_synthetic_bundle(Path(temporary) / "negative", False))
        require(positive["decision"]["claim_state"] == "synthetic_positive_control" and
                negative["decision"]["claim_state"] == "synthetic_valid_negative_control" and
                positive["decision"]["gate_math_passed"] is True and
                positive["decision"]["registered_profiles_complete"] is False and
                positive["decision"]["registered_gate_passed"] is False and
                positive["decision"]["empirical_claim_allowed"] is False and
                negative["decision"]["registered_gate_passed"] is False,
                "synthetic decision controls were relabeled")
        for key in ("statistical_intervals", "practical_effects", "actor_returns", "per_task_heterogeneity",
                    "calibration", "compute_nfe", "controls", "pixel", "theory", "limitations"):
            require(bool(positive[key]), f"self-test omitted {key}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("self-test", "scaffold", "build", "verify"):
        modes.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--core-root", type=Path); parser.add_argument("--controls-root", type=Path)
    parser.add_argument("--pixel-aggregate", type=Path); parser.add_argument("--bundle", type=Path)
    parser.add_argument("--report", type=Path); args = parser.parse_args(argv)
    try:
        if args.self_test:
            run_self_test(); print(SELF_TEST_SUCCESS)
        elif args.scaffold:
            verify_scaffold(); print(SCAFFOLD_SUCCESS)
        elif args.build:
            require(all((args.core_root, args.controls_root, args.pixel_aggregate, args.bundle, args.report)),
                    "--build requires core, controls, pixel aggregate, bundle, and report paths")
            print(json.dumps(build_authenticated_bundle(args.core_root, args.controls_root, args.pixel_aggregate,
                  args.bundle, args.report)["decision"], indent=2, sort_keys=True)); print(BUNDLE_SUCCESS)
        else:
            require(args.bundle is not None and args.report is not None, "--verify requires --bundle and --report")
            print(json.dumps(verify_bundle(args.bundle, args.report)["decision"],
                             indent=2, sort_keys=True)); print(BUNDLE_SUCCESS)
    except (OSError, KeyError, TypeError, ValueError, VerificationError) as error:
        print(f"TRAJECTORY_IMF_REPORT_VERIFICATION_FAILED: {error}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
