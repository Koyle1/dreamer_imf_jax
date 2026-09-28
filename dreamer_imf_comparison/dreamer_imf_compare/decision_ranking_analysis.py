"""CPU-only reanalysis of the nine frozen 7d44604 candidate diagnostics.

Run with ``python -m dreamer_imf_compare.decision_ranking_analysis ARTIFACTS``.
No simulator, checkpoint deserialization, JAX, or production runner is used.
Authentication pins the original evidence, not the current checkout's commit.
The oracle is the stored finite native-episode frozen-actor continuation; these
are retrospective fixed-plan decisions, not new controller evaluations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

SOURCE_COMMIT = "7d44604d2fe55970c87e6d980bf208cceae71143"
REPORT_SHA256 = "69ed0431a8fb2d37577a9053874bcb16a81ae93d20720867844c4b575fb90141"
MANIFEST_SHA256 = "b5979d7f248a68af1eee3489d96fd94e73b864537f1b6639228d7596f6335e0c"
SELECTION_SHA256 = "e5b82d7328381e17edaec22bb58123361e078ecc22fd2cb78324d4b07dc11b18"
DIAGNOSTIC_MARKERS = (
    "0de44fb4404f4778adc64959d01896821f9f1d1b1a5c1ac2a0cb1d359e1672d8",
    "12dec21d74ef93380e7dfd057c0824e3cce6c8e5c04cec6ca58f31989b8f5a79",
    "095a372c9e731ada5a0b303a1ca404fb30995161e1c71d1124bee0dc5c103adf",
    "51b52c71e40fc54d22ecfcff89e12ac97c85578a3494101a66b38b571c0573a4",
    "e3fe328c6da66801de0fa51664b0231aaa7f89ba08ac10fb47e24e3555ec3c3b",
    "bd6ac9f5bf925ce93bb5c484e48affa80b815c8229b4e49f7ace3b536afd5c2a",
    "ae28e000561a0a00dcfc48144008d02025f2ce2739fd739921c1bff23d2705df",
    "eab3c9090fc6354097022a78d93475d93af11252b1ee9ff07d061ab75ec2ca81",
    "9bba0b763898c052aa8cd47a1be69d82025b2f4a750139f030e583ba67fb00e4",
)
SPLITS = ("calibration", "validation")
FAMILIES = ("recursive", "direct")
REFERENCE_MODES = ("latent", "endpoint")
SCORES = ("model", "real_reward", "real_endpoint_critic", "oracle")
POLICIES = ("unrestricted", "fixed_feasible")
TIE_ATOL = 1e-8
SUCCESS_TOKEN = "DECISION_RANKING_ANALYSIS_VERIFIED"
_FILES = {"result.json", "trace.npz"} | {
    f"{phase}-{kind}.json"
    for phase in ("primer", "create", "replay")
    for kind in ("receipt", "seal")
}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return _sha(_canonical(value))


def _json(data: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in out, f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def bad_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    value = json.loads(data, object_pairs_hook=unique, parse_constant=bad_constant)
    # Also rejects overflowed JSON numbers such as 1e999.
    _canonical(value)
    return value


def _bound(value: Mapping[str, Any], label: str) -> None:
    _require(
        value.get("source_commit") == SOURCE_COMMIT, f"{label}: source binding differs"
    )
    _require(
        value.get("manifest_sha256") == MANIFEST_SHA256,
        f"{label}: manifest binding differs",
    )


def _pinned_file(root: Path, relative: str, expected: str) -> bytes:
    path = root / relative
    _require(
        path.resolve().is_relative_to(root.resolve()),
        f"input escapes artifact directory: {relative}",
    )
    data = path.read_bytes()
    _require(_sha(data) == expected, f"SHA256 mismatch: {relative}")
    return data


def array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Reproduce the original named-array digest without importing its runner."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        descriptor = _canonical([name, value.dtype.str, list(value.shape)])
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _close(
    actual: Any, expected: Any, label: str, *, atol: float = 1e-12, rtol: float = 1e-12
) -> None:
    _require(
        np.allclose(actual, expected, atol=atol, rtol=rtol),
        f"trace arithmetic differs: {label}",
    )


def validate_trace(trace: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> None:
    """Independently check the grouping, duplicate plan and score decomposition."""
    settings = record["realized_settings"]
    horizon, maximum = settings["horizon"], settings["maximum_episode_steps"]
    count = len(trace["plan_kind"])
    for key, array in trace.items():
        _require(
            array.dtype.kind in "biuf" and np.isfinite(array).all(),
            f"nonfinite/nonnumeric trace: {key}",
        )
        _require(array.ndim > 0, f"scalar trace field: {key}")
        _require(
            array.shape[0] == (2 if key.startswith("baseline_") else count),
            f"trace length differs: {key}",
        )
    for key in ("model_feasible", "used_fallback", "native_episode_end"):
        _require(
            trace[key].dtype.kind == "b" and trace[key].shape == (count,),
            f"invalid Boolean mask: {key}",
        )
    for key in (
        "split_id",
        "family_id",
        "reference_mode_id",
        "snapshot_id",
        "environment_seed",
        "episode_step",
        "plan_kind",
    ):
        _require(
            trace[key].dtype.kind in "iu" and trace[key].shape == (count,),
            f"invalid integer identity: {key}",
        )
    _require(
        trace["action_sequence"].shape[:2] == (count, horizon),
        "invalid action sequence shape",
    )
    _require(
        trace["real_rewards"].shape
        == trace["real_continuations"].shape
        == (count, maximum),
        "invalid continuation shape",
    )
    _require(
        trace["native_episode_end"].all(),
        "continuations do not reach native episode end",
    )
    _require(
        np.all(np.abs(trace["action_sequence"]) <= 1), "unbounded candidate action"
    )
    _require(
        np.array_equal(trace["first_action"], trace["action_sequence"][:, 0]),
        "first-action mismatch",
    )
    for source in ("model", "real"):
        _close(
            trace[f"{source}_objective"],
            trace[f"{source}_stage"] + trace[f"{source}_terminal"],
            f"{source} objective",
            atol=2e-6,
            rtol=1e-6,
        )
    _close(
        trace["real_terminal_critic"],
        trace["real_terminal"] + trace["terminal_critic_mc_error"],
        "real-terminal critic",
    )
    _close(
        trace["terminal_transition_error"],
        trace["model_terminal"] - trace["real_terminal_critic"],
        "terminal transition",
    )
    weights = np.broadcast_to(
        record["discount"] ** np.arange(maximum), trace["real_rewards"].shape
    ).copy()
    weights[:, 1:] *= np.cumprod(trace["real_continuations"][:, :-1], axis=1)
    _close(
        trace["real_stage"],
        np.sum(weights[:, :horizon] * trace["real_rewards"][:, :horizon], axis=1),
        "real stage",
    )
    _close(
        trace["real_terminal"],
        np.sum(weights[:, horizon:] * trace["real_rewards"][:, horizon:], axis=1),
        "real continuation",
    )
    cursor, sid, baseline = 0, 0, 0
    signed_scales = np.asarray(
        [sign * scale for scale in settings["direction_scales"] for sign in (-1, 1)],
        np.float32,
    )
    for split, seeds in settings["splits"].items():
        for seed in seeds:
            _require(
                trace["baseline_split_id"][baseline] == SPLITS.index(split),
                "baseline split mismatch",
            )
            _require(
                trace["baseline_environment_seed"][baseline] == seed,
                "baseline seed mismatch",
            )
            for step in settings["snapshot_steps"]:
                for family, mode in itertools.product(FAMILIES, REFERENCE_MODES):
                    ids = np.arange(cursor, cursor + 9)
                    _require(cursor + 9 <= count, "missing candidate group")
                    for key, value in (
                        ("split_id", SPLITS.index(split)),
                        ("environment_seed", seed),
                        ("episode_step", step),
                        ("snapshot_id", sid),
                        ("family_id", FAMILIES.index(family)),
                        ("reference_mode_id", REFERENCE_MODES.index(mode)),
                    ):
                        _require(
                            np.all(trace[key][ids] == value),
                            f"group identity mismatch: {key}",
                        )
                    _require(
                        np.array_equal(
                            trace["plan_kind"][ids], [0, 1, 2, 3, 3, 3, 3, 3, 3]
                        ),
                        "candidate kind/order mismatch",
                    )
                    _require(
                        np.array_equal(
                            trace["direction_scale"][ids],
                            np.r_[np.zeros(3), signed_scales],
                        ),
                        "direction ladder mismatch",
                    )
                    start = trace["baseline_observations"][baseline, step]
                    _require(
                        np.all(trace["initial_observation"][ids] == start),
                        "unmatched actor-occupancy snapshot",
                    )
                    ref, proposal, executed = ids[:3]
                    duplicate = ref if trace["used_fallback"][executed] else proposal
                    _require(
                        np.array_equal(
                            trace["action_sequence"][executed],
                            trace["action_sequence"][duplicate],
                        ),
                        "executed plan is not reference/proposal duplicate",
                    )
                    _require(
                        trace["real_objective"][executed]
                        == trace["real_objective"][duplicate],
                        "duplicate plan real objective mismatch",
                    )
                    for source, component in itertools.product(
                        ("model", "real"), ("stage", "terminal", "objective")
                    ):
                        values = trace[f"{source}_{component}"][ids]
                        _require(
                            np.array_equal(
                                trace[f"{source}_{component}_gain"][ids],
                                values - values[0],
                            ),
                            "stored gain mismatch",
                        )
                    delta = (
                        trace["action_sequence"][ids] - trace["action_sequence"][ref]
                    )
                    _require(
                        np.array_equal(trace["actual_delta"][ids], delta),
                        "finite action change mismatch",
                    )
                    expected = np.clip(
                        trace["action_sequence"][ref]
                        + signed_scales[:, None, None]
                        * trace["common_direction"][ids[3:]],
                        -1,
                        1,
                    )
                    _require(
                        np.array_equal(expected, trace["action_sequence"][ids[3:]]),
                        "directional plan mismatch",
                    )
                    cursor += 9
                sid += 1
            baseline += 1
    _require(cursor == count and baseline == 2, "extra diagnostic groups")


@dataclass(frozen=True)
class AuthenticatedStudy:
    manifest: Mapping[str, Any]
    selection: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    traces: tuple[Mapping[str, np.ndarray], ...]


def authenticate_artifacts(artifact_dir: str | Path) -> AuthenticatedStudy:
    """Authenticate all original bindings before returning any analytic input."""
    root = Path(artifact_dir)
    report = _json(_pinned_file(root, "report.json", REPORT_SHA256))
    selection = _json(_pinned_file(root, "selection.json", SELECTION_SHA256))
    manifest = _json((root / "manifest.json").read_bytes())
    _require(_digest(manifest) == MANIFEST_SHA256, "manifest canonical SHA256 mismatch")
    _require(manifest["source_commit"] == SOURCE_COMMIT, "manifest source mismatch")
    _require(
        _digest(manifest["protocol"]) == manifest["protocol_sha256"],
        "protocol digest mismatch",
    )
    verified = _json((root / "verified.json").read_bytes())
    _require(
        verified
        == dict(
            source_commit=SOURCE_COMMIT,
            manifest_sha256=MANIFEST_SHA256,
            report_sha256=REPORT_SHA256,
            selection_sha256=SELECTION_SHA256,
        ),
        "top-level verified bindings differ",
    )
    _bound(report, "report")
    _bound(selection, "selection")
    _require(report["selection"] == selection, "report selection mismatch")
    _require(
        selection["diagnostic_markers"]
        == {f"{i}:diagnostics": digest for i, digest in enumerate(DIAGNOSTIC_MARKERS)},
        "diagnostic marker set differs",
    )
    _require(
        selection["reference_modes"] == {family: "latent" for family in FAMILIES},
        "selected reference modes differ",
    )
    protocol = manifest["protocol"]
    expected_settings = dict(protocol["diagnostics"])
    expected_settings["splits"] = {
        s: expected_settings[f"{s}_environment_seeds"] for s in SPLITS
    }
    records, traces = [], []
    for index, marker_hash in enumerate(DIAGNOSTIC_MARKERS):
        prefix = f"diagnostics/{index:03}"
        marker = _json(_pinned_file(root, f"{prefix}/verified.json", marker_hash))
        _bound(marker, f"diagnostic {index} marker")
        _require(
            marker["stage"] == "diagnostics"
            and marker["variant"] is None
            and marker["strict_bitwise_replay"] is True,
            "invalid diagnostic verification marker",
        )
        _require(set(marker["files"]) == _FILES, "diagnostic file set differs")
        blobs = {
            name: _pinned_file(root, f"{prefix}/{name}", digest)
            for name, digest in marker["files"].items()
        }
        record = _json(blobs["result.json"])
        _bound(record, f"diagnostic {index} result")
        _require(
            record["index"] == index and record["preflight"] is False,
            "diagnostic index/preflight mismatch",
        )
        _require(
            record["schema"] == "controller-repair-matched-diagnostics-v1",
            "diagnostic schema mismatch",
        )
        _require(
            all(record[key] == value for key, value in marker["cell"].items()),
            "diagnostic cell mismatch",
        )
        _require(
            record["realized_settings"] == expected_settings
            and record["realized_controller"] == protocol["controller"],
            "realized settings differ",
        )
        _require(record["discount"] == 0.99, "discount mismatch")
        _require(
            record == report["diagnostics"][index],
            "diagnostic differs from pinned report",
        )
        dependency = manifest["dependency_index"][str(record["training_index"])][
            "files"
        ]
        _require(
            record["checkpoint_sha256"] == dependency["checkpoint.pkl"]
            and record["replay_file_sha256"] == dependency["replay.npz"],
            "checkpoint/replay dependency mismatch",
        )
        receipts = {}
        for phase in ("primer", "create", "replay"):
            receipt, seal = _json(blobs[f"{phase}-receipt.json"]), _json(
                blobs[f"{phase}-seal.json"]
            )
            _bound(receipt, f"diagnostic {index} {phase} receipt")
            _require(receipt["mode"] == phase, "receipt phase mismatch")
            _require(
                seal["receipt_sha256"] == _sha(blobs[f"{phase}-receipt.json"]),
                "receipt seal mismatch",
            )
            _require(
                seal["cache_sha256"]
                == receipt["cache_sha256"]
                == marker["cache_sha256"],
                "cache binding mismatch",
            )
            _require(
                receipt["checkpoint_sha256"] == record["checkpoint_sha256"],
                "receipt checkpoint mismatch",
            )
            if phase != "primer":
                _require(
                    receipt["core_sha256"] == _digest(record)
                    and receipt["trace_sha256"] == record["trace_sha256"],
                    "receipt result/trace mismatch",
                )
            receipts[phase] = receipt
        _require(
            receipts["create"]["runtime"] == receipts["replay"]["runtime"],
            "replay runtime mismatch",
        )
        with np.load(io.BytesIO(blobs["trace.npz"]), allow_pickle=False) as loaded:
            _require(len(loaded.files) == len(set(loaded.files)), "duplicate NPZ field")
            trace = {key: loaded[key] for key in loaded.files}
        _require(
            array_digest(trace) == record["trace_sha256"], "named-array digest mismatch"
        )
        validate_trace(trace, record)
        for array in trace.values():
            array.flags.writeable = False
        records.append(record)
        traces.append(trace)
    _require(
        {(r["task"], r["world_model_seed"]) for r in records}
        == set(itertools.product(protocol["tasks"], protocol["world_model_seeds"])),
        "task/world-model coverage differs",
    )
    return AuthenticatedStudy(manifest, selection, tuple(records), tuple(traces))


def _finite_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    _require(
        array.ndim == 1
        and len(array) > 0
        and array.dtype.kind in "biuf"
        and np.isfinite(array).all(),
        f"invalid/nonfinite vector: {label}",
    )
    return array.astype(np.float64)


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def analyze_candidates(
    scores: Mapping[str, Any],
    real_values: Any,
    feasible: Any,
    *,
    reference_index: int = 0,
    tie_atol: float = TIE_ATOL,
) -> dict[str, Any]:
    """Compare scores on the same candidate set and frozen feasibility mask.

    The argmax is exact (NumPy first-index order); tolerance is used only for
    informative pair signs and separately labelled near-tie sensitivity.
    Reference-relative errors exclude the reference's tautological zero by
    default, including when that reference is outside the feasible set.
    """
    real = _finite_vector(real_values, "real values")
    _require(
        type(reference_index) is int and 0 <= reference_index < len(real),
        "invalid reference index",
    )
    _require(np.isfinite(tie_atol) and tie_atol >= 0, "invalid tie tolerance")
    mask = np.asarray(feasible)
    _require(
        mask.dtype.kind == "b" and mask.shape == real.shape,
        "feasibility must be a matching Boolean vector",
    )
    predictions = {name: _finite_vector(value, name) for name, value in scores.items()}
    _require(
        bool(predictions)
        and all(value.shape == real.shape for value in predictions.values()),
        "score shapes differ",
    )
    real_gain = real - real[reference_index]
    result: dict[str, Any] = {}
    for policy, admissible in (
        ("unrestricted", np.ones(len(real), bool)),
        ("fixed_feasible", mask),
    ):
        ids = np.flatnonzero(admissible)
        nonref = ids[ids != reference_index]
        pairs = list(itertools.combinations(ids.tolist(), 2))
        left = np.array([a for a, _ in pairs], dtype=int)
        right = np.array([b for _, b in pairs], dtype=int)
        real_diff = real[left] - real[right]
        informative = np.abs(real_diff) > tie_atol
        oracle_value = float(np.max(real[ids])) if len(ids) else None
        by_score = {}
        for name, predicted in predictions.items():
            gain_error = (predicted - predicted[reference_index]) - real_gain
            predicted_diff = predicted[left] - predicted[right]
            model_ties = np.abs(predicted_diff) <= tie_atol
            agreements = (
                informative
                & ~model_ties
                & (np.sign(predicted_diff) == np.sign(real_diff))
            )
            metric: dict[str, Any] = {
                "candidate_count": int(len(ids)),
                "nonreference_count": int(len(nonref)),
                "gain_error_absolute_sum": float(np.sum(np.abs(gain_error[nonref]))),
                "gain_mae": (
                    float(np.mean(np.abs(gain_error[nonref]))) if len(nonref) else None
                ),
                "gain_mae_including_reference": (
                    float(np.mean(np.abs(gain_error[ids]))) if len(ids) else None
                ),
                "pair_count": len(pairs),
                "informative_pair_count": int(informative.sum()),
                "pair_agreement_count": int(agreements.sum()),
                "informative_pair_agreement": (
                    float(agreements.sum() / informative.sum())
                    if informative.any()
                    else None
                ),
                "real_tie_pair_count": int((~informative).sum()),
                "model_tie_pair_count": int(model_ties.sum()),
                "model_tie_informative_pair_count": int(
                    (model_ties & informative).sum()
                ),
                "both_tie_pair_count": int((model_ties & ~informative).sum()),
                "chosen_index": None,
                "chosen_real_gain": None,
                "oracle_real_gain": None,
                "oracle_regret": None,
                "exact_argmax_indices": [],
                "near_argmax_indices": [],
                "exact_tie_real_gain_min": None,
                "exact_tie_real_gain_max": None,
                "exact_tie_regret_min": None,
                "exact_tie_regret_max": None,
                "near_tie_real_gain_min": None,
                "near_tie_real_gain_max": None,
                "near_tie_regret_min": None,
                "near_tie_regret_max": None,
            }
            if len(ids):
                chosen = int(ids[np.argmax(predicted[ids])])
                best = predicted[chosen]
                exact = ids[predicted[ids] == best]
                near = ids[(best - predicted[ids]) <= tie_atol]
                metric.update(
                    chosen_index=chosen,
                    chosen_real_gain=float(real_gain[chosen]),
                    oracle_real_gain=oracle_value - float(real[reference_index]),
                    oracle_regret=oracle_value - float(real[chosen]),
                    exact_argmax_indices=exact.tolist(),
                    near_argmax_indices=near.tolist(),
                )
                for label, tied in (("exact", exact), ("near", near)):
                    metric[f"{label}_tie_real_gain_min"] = float(
                        np.min(real_gain[tied])
                    )
                    metric[f"{label}_tie_real_gain_max"] = float(
                        np.max(real_gain[tied])
                    )
                    metric[f"{label}_tie_regret_min"] = oracle_value - float(
                        np.max(real[tied])
                    )
                    metric[f"{label}_tie_regret_max"] = oracle_value - float(
                        np.min(real[tied])
                    )
            by_score[name] = metric
        result[policy] = {
            "admissible_indices": ids.tolist(),
            "empty": not bool(len(ids)),
            "reference_is_admissible": bool(admissible[reference_index]),
            "reference_model_feasible": bool(mask[reference_index]),
            "scores": by_score,
        }
    result["reference_fallback"] = {
        "required_by_empty_fixed_feasible_set": not bool(mask.any()),
        "candidate_index": reference_index,
        "real_gain": 0.0,
        "model_feasible": bool(mask[reference_index]),
        "is_feasible_decision": bool(mask[reference_index]),
        "unrestricted_oracle_regret": float(np.max(real) - real[reference_index]),
        "interpretation": "Reference return is a diagnostic fallback, never a certificate of feasibility.",
    }
    return result


def _case(
    record: Mapping[str, Any], trace: Mapping[str, np.ndarray], ids: np.ndarray
) -> dict[str, Any]:
    full_ids = ids
    ids = ids[trace["plan_kind"][ids] != 2]
    _require(
        len(ids) == 8 and trace["plan_kind"][ids[0]] == 0,
        "expected eight nonexecuted candidates",
    )
    first = int(ids[0])
    real = trace["real_objective"][ids]
    scores = {
        "model": trace["model_objective"][ids],
        "real_reward": trace["real_stage"][ids] + trace["model_terminal"][ids],
        "real_endpoint_critic": trace["real_stage"][ids]
        + trace["real_terminal"][ids]
        + trace["terminal_critic_mc_error"][ids],
        "oracle": real,
    }
    components = {
        "reward": trace["model_stage"][ids] - trace["real_stage"][ids],
        "terminal_transition": trace["terminal_transition_error"][ids],
        "terminal_critic": trace["terminal_critic_mc_error"][ids],
    }
    executed = int(full_ids[trace["plan_kind"][full_ids] == 2][0])
    return {
        "split": SPLITS[int(trace["split_id"][first])],
        "family": FAMILIES[int(trace["family_id"][first])],
        "reference_mode": REFERENCE_MODES[int(trace["reference_mode_id"][first])],
        "task": record["task"],
        "world_model_seed": record["world_model_seed"],
        "actor_seed": record["actor_seed"],
        "diagnostic_index": record["index"],
        "snapshot_id": int(trace["snapshot_id"][first]),
        "environment_seed": int(trace["environment_seed"][first]),
        "episode_step": int(trace["episode_step"][first]),
        "candidate_trace_indices": ids.tolist(),
        "candidate_kinds": ["reference", "proposal"] + ["directional"] * 6,
        "direction_scales": trace["direction_scale"][ids].tolist(),
        "model_feasible": trace["model_feasible"][ids].tolist(),
        "scores": {name: values.tolist() for name, values in scores.items()},
        "real_gains": (real - real[0]).tolist(),
        "component_errors": {
            name: {
                "level_mae": float(np.mean(np.abs(values))),
                "reference_error": float(values[0]),
                "gain_mae": float(np.mean(np.abs(values[1:] - values[0]))),
                "candidate_errors": values.tolist(),
            }
            for name, values in components.items()
        },
        "observed_execution": {
            "excluded_trace_index": executed,
            "used_fallback": bool(trace["used_fallback"][executed]),
            "model_feasible": bool(trace["model_feasible"][executed]),
            "real_gain": float(trace["real_objective"][executed] - real[0]),
        },
        "decisions": analyze_candidates(scores, real, trace["model_feasible"][ids]),
    }


_COUNT_METRICS = (
    "candidate_count",
    "nonreference_count",
    "pair_count",
    "informative_pair_count",
    "pair_agreement_count",
    "real_tie_pair_count",
    "model_tie_pair_count",
    "model_tie_informative_pair_count",
    "both_tie_pair_count",
)
_MEAN_METRICS = (
    "gain_mae",
    "gain_mae_including_reference",
    "informative_pair_agreement",
    "chosen_real_gain",
    "oracle_real_gain",
    "oracle_regret",
    "exact_tie_real_gain_min",
    "exact_tie_real_gain_max",
    "exact_tie_regret_min",
    "exact_tie_regret_max",
    "near_tie_real_gain_min",
    "near_tie_real_gain_max",
    "near_tie_regret_min",
    "near_tie_regret_max",
)


def _summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "snapshot_count": len(cases),
        "world_model_count": 1,
        "averaging_unit": "snapshot",
        "reference_feasible_count": sum(c["model_feasible"][0] for c in cases),
        "empty_feasible_count": sum(
            c["decisions"]["fixed_feasible"]["empty"] for c in cases
        ),
        "observed_fallback_count": sum(
            c["observed_execution"]["used_fallback"] for c in cases
        ),
        "observed_infeasible_execution_count": sum(
            not c["observed_execution"]["model_feasible"] for c in cases
        ),
        "empty_feasible_infeasible_reference_count": sum(
            c["decisions"]["fixed_feasible"]["empty"] and not c["model_feasible"][0]
            for c in cases
        ),
        "decisions": {},
        "component_errors": {},
    }
    for policy in POLICIES:
        summary["decisions"][policy] = {}
        for score in SCORES:
            rows = [c["decisions"][policy]["scores"][score] for c in cases]
            counts = {key: sum(r[key] for r in rows) for key in _COUNT_METRICS}
            counts.update(
                nonempty_decision_count=sum(
                    r["chosen_index"] is not None for r in rows
                ),
                exact_argmax_tied_count=sum(
                    len(r["exact_argmax_indices"]) > 1 for r in rows
                ),
                near_argmax_tied_count=sum(
                    len(r["near_argmax_indices"]) > 1 for r in rows
                ),
            )
            summary["decisions"][policy][score] = {
                "counts": counts,
                "means": {key: _mean([r[key] for r in rows]) for key in _MEAN_METRICS},
                "mean_contributor_counts": {
                    key: sum(r[key] is not None for r in rows) for key in _MEAN_METRICS
                },
                "pooled_informative_pair_agreement": (
                    counts["pair_agreement_count"] / counts["informative_pair_count"]
                    if counts["informative_pair_count"]
                    else None
                ),
            }
    for component in ("reward", "terminal_transition", "terminal_critic"):
        summary["component_errors"][component] = {
            key: _mean([c["component_errors"][component][key] for c in cases])
            for key in ("level_mae", "reference_error", "gain_mae")
        }
    return summary


def _macro_summaries(
    children: Sequence[Mapping[str, Any]], averaging_unit: str
) -> dict[str, Any]:
    """Equal-weight child means; counts/pair ratios remain descriptive totals."""
    out: dict[str, Any] = {
        key: sum(c[key] for c in children)
        for key in children[0]
        if key.endswith("_count")
    }
    out["averaging_unit"] = averaging_unit
    out["decisions"], out["component_errors"] = {}, {}
    for policy in POLICIES:
        out["decisions"][policy] = {}
        for score in SCORES:
            rows = [c["decisions"][policy][score] for c in children]
            counts = {
                key: sum(r["counts"][key] for r in rows) for key in rows[0]["counts"]
            }
            out["decisions"][policy][score] = {
                "counts": counts,
                "means": {
                    key: _mean([r["means"][key] for r in rows]) for key in _MEAN_METRICS
                },
                "mean_contributor_counts": {
                    key: sum(r["means"][key] is not None for r in rows)
                    for key in _MEAN_METRICS
                },
                "pooled_informative_pair_agreement": (
                    counts["pair_agreement_count"] / counts["informative_pair_count"]
                    if counts["informative_pair_count"]
                    else None
                ),
            }
    for component, metrics in children[0]["component_errors"].items():
        out["component_errors"][component] = {
            key: _mean([c["component_errors"][component][key] for c in children])
            for key in metrics
        }
    return out


def analyze_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    study = authenticate_artifacts(artifact_dir)
    cases = []
    for record, trace in zip(study.records, study.traces):
        for split, family, mode, sid in itertools.product(
            range(2), range(2), range(2), np.unique(trace["snapshot_id"])
        ):
            ids = np.flatnonzero(
                (trace["split_id"] == split)
                & (trace["family_id"] == family)
                & (trace["reference_mode_id"] == mode)
                & (trace["snapshot_id"] == sid)
            )
            if len(ids):
                cases.append(_case(record, trace, ids))
    world_rows, task_rows, aggregate_rows = [], [], []
    tasks = study.manifest["protocol"]["tasks"]
    worlds = study.manifest["protocol"]["world_model_seeds"]
    for split, family, mode in itertools.product(SPLITS, FAMILIES, REFERENCE_MODES):
        identity = dict(split=split, family=family, reference_mode=mode)
        current_tasks = []
        for task in tasks:
            current_worlds = []
            for world in worlds:
                selected = [
                    c
                    for c in cases
                    if all(c[k] == v for k, v in identity.items())
                    and c["task"] == task
                    and c["world_model_seed"] == world
                ]
                _require(
                    len(selected) == 4,
                    "expected four nested snapshots per world/split/family/mode",
                )
                row = dict(
                    identity,
                    task=task,
                    world_model_seed=world,
                    **_summarize_cases(selected),
                )
                world_rows.append(row)
                current_worlds.append(row)
            row = dict(
                identity, task=task, **_macro_summaries(current_worlds, "world_model")
            )
            task_rows.append(row)
            current_tasks.append(row)
        aggregate_rows.append(
            dict(
                identity,
                task_count=len(tasks),
                **_macro_summaries(current_tasks, "task"),
            )
        )
    _require(len(cases) == 288, "expected 288 matched candidate groups")
    return {
        "schema": "controller-repair-decision-ranking-v1",
        "authentication": {
            "source_commit": SOURCE_COMMIT,
            "report_sha256": REPORT_SHA256,
            "manifest_canonical_sha256": MANIFEST_SHA256,
            "selection_sha256": SELECTION_SHA256,
            "diagnostic_marker_sha256": list(DIAGNOSTIC_MARKERS),
            "diagnostic_count": len(study.records),
        },
        "definitions": {
            "scores": {
                "model": "model_objective",
                "real_reward": "real_stage + model_terminal",
                "real_endpoint_critic": "real_stage + real_terminal + terminal_critic_mc_error = real_stage + real_terminal_critic",
                "oracle": "real_objective = real_stage + real_terminal",
            },
            "candidate_order": "original trace order: reference, proposal, six signed directions; executed duplicate excluded; coincident remaining plans retained as registered candidate labels",
            "gain_mae": "mean absolute error of candidate-minus-reference scores versus real gains; excludes reference zero; feasible analysis includes only admissible nonreference candidates but always uses original reference",
            "pair_agreement": "pairs with abs(real difference) > tie_atol; model ties count as disagreement; no real-informative pairs produces null",
            "tie_atol": TIE_ATOL,
            "argmax": "exact maximum with deterministic first candidate index; exact and tolerance near-tie ranges both reported",
            "oracle_regret": "maximum stored real objective in same admissible set minus chosen real objective",
            "fixed_feasible": "original model_feasible mask held fixed across all four scores; empty decisions and regret are null; fallback reference is reported separately even when infeasible",
            "aggregation": "snapshot means within each frozen world model; equal world-model means within fixed task; equal task means across tasks; null conditional metrics omit children without observations; mean_contributor_counts reports denominators in the row's averaging_unit; counts and pooled pair agreement are descriptive, not independent sample sizes",
            "statistical_unit": "three trained world-model seeds nested within each of three fixed tasks; actor, split, snapshot, candidate and family/mode reuse do not add independent models; no confidence interval or confirmatory claim",
            "scope": "retrospective candidate-set score substitutions with fixed five-action open-loop plans then finite native-episode frozen-ReBRAC continuation; no replanning or new controller performance claim",
            "primary": "validation / selected latent reference mode; endpoint and calibration are labelled sensitivity analyses",
        },
        "selection": dict(study.selection["reference_modes"]),
        "primary_validation": [
            row
            for row in aggregate_rows
            if row["split"] == "validation"
            and row["reference_mode"]
            == study.selection["reference_modes"][row["family"]]
        ],
        "group_summaries": aggregate_rows,
        "task_summaries": task_rows,
        "world_model_summaries": world_rows,
        "cases": cases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="write a new JSON file outside the input tree; existing files are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            _require(
                not args.output.resolve().is_relative_to(args.artifact_dir.resolve()),
                "output must be outside the input artifact tree",
            )
            _require(
                not args.output.exists() and not args.output.is_symlink(),
                "output already exists",
            )
        analysis = analyze_artifacts(args.artifact_dir)
        serialized = (
            json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        # Keep stdout parseable JSON; token appears only after successful output.
        print(SUCCESS_TOKEN, file=sys.stderr)
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(f"decision-ranking analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
