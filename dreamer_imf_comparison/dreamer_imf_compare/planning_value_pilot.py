"""Fixed, trace-only finite-return head pilot; never executes a controller.

Create from a clean commit, then independently refit with ``verify``. Both
commands authenticate the immutable input study. Output must be a fresh path
outside that input tree; verification never rewrites scientific evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from .decision_ranking_analysis import (
    FAMILIES,
    MANIFEST_SHA256,
    REPORT_SHA256,
    SOURCE_COMMIT,
    analyze_candidates,
    array_digest,
    authenticate_artifacts,
)
from .planning_value_head import HeadSettings, fit_head, mc_suffix_returns, predict_head

SCHEMA = "finite-planning-value-trace-pilot-v1"
PROTOCOL_PATH = "docs/planning_value_pilot_protocol.md"
SCORES = ("model", "real_endpoint_critic", "planning_head", "time_only", "oracle")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> Any:
    return parse_json(path.read_bytes())


def parse_json(data: bytes) -> Any:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    value = json.loads(data, object_pairs_hook=unique)
    canonical(value)  # reject all NaN/Inf, including exponent overflow
    return value


def source_identity() -> dict[str, str]:
    import jax

    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("scientific fitting requires a clean exact source checkout")
    if git("branch", "--show-current") != "main":
        raise ValueError("this pilot must run from the exact committed main branch")
    return {
        "commit": git("rev-parse", "HEAD"),
        "protocol_sha256": hashlib.sha256(
            (root / PROTOCOL_PATH).read_bytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "machine": platform.machine(),
    }


def extract_training(
    trace: Mapping[str, np.ndarray],
    *,
    horizon: int = 5,
    maximum: int = 1000,
    gamma: float = 0.99,
) -> dict[str, Any]:
    """Read ONLY calibration rows. Never label fixed-plan prefixes as V_pi."""
    baseline_ids = np.flatnonzero(trace["baseline_split_id"] == 0)
    if len(baseline_ids) != 1:
        raise ValueError("pilot requires one calibration root episode")
    bi = int(baseline_ids[0])
    length = int(trace["baseline_length"][bi])
    if length != maximum or not bool(trace["baseline_native_episode_end"][bi]):
        raise ValueError("baseline must reach the registered native time limit")
    rewards = trace["baseline_rewards"][bi, :length]
    continuation = trace["baseline_continuations"][bi, :length]
    suffix = mc_suffix_returns(rewards, continuation, gamma=gamma)
    observations = trace["baseline_observations"][bi, :length]
    remaining = np.arange(length, 0, -1, dtype=np.int32)
    mask = (
        (trace["split_id"] == 0)
        & (trace["reference_mode_id"] == 0)
        & (trace["plan_kind"] != 2)
    )
    ids = np.flatnonzero(mask)
    seen: dict[tuple[int, bytes], int] = {}
    unique_ids = []
    for i in ids:
        key = (int(trace["snapshot_id"][i]), trace["action_sequence"][i].tobytes())
        if key in seen:
            old = seen[key]
            if (
                not np.array_equal(
                    trace["real_stage_observations"][i],
                    trace["real_stage_observations"][old],
                )
                or trace["real_terminal"][i] != trace["real_terminal"][old]
            ):
                raise ValueError("duplicate fixed plan has different retained target")
            continue
        seen[key] = int(i)
        unique_ids.append(int(i))
    ids = np.asarray(unique_ids, dtype=int)
    if not len(ids) or not np.all(trace["real_length"][ids] > horizon):
        raise ValueError("missing or prematurely terminated training endpoint")
    if not np.all(trace["real_length"][ids] == maximum - trace["episode_step"][ids]):
        raise ValueError("endpoint continuation disagrees with native remaining time")
    if not np.all(trace["stage_mask"][ids, :horizon]):
        raise ValueError("incomplete fixed-plan prefix")
    prefix_cont = np.prod(trace["real_continuations"][ids, :horizon], axis=1)
    if not np.all(prefix_cont == 1):
        raise ValueError("this frozen pilot requires surviving native prefixes")
    endpoint_targets = trace["real_terminal"][ids] / gamma**horizon
    # Independently recompute every suffix label from rewards, not the stored sum.
    for i, target in zip(ids, endpoint_targets):
        n = int(trace["real_length"][i])
        actual = mc_suffix_returns(
            trace["real_rewards"][i, :n],
            trace["real_continuations"][i, :n],
            gamma=gamma,
        )[horizon]
        if not np.isclose(actual, target, atol=1e-11, rtol=1e-12):
            raise ValueError("endpoint MC label arithmetic differs")
    endpoint_remaining = maximum - trace["episode_step"][ids] - horizon
    endpoint_observations = trace["real_stage_observations"][ids, horizon]
    arrays = {
        "observations": np.concatenate((observations, endpoint_observations)),
        "remaining": np.concatenate((remaining, endpoint_remaining)).astype(np.int32),
        "returns": np.concatenate((suffix[:-1], endpoint_targets)),
        "weights": np.r_[
            np.full(length, 0.5 / length), np.full(len(ids), 0.5 / len(ids))
        ],
    }
    aliases = {}
    for obs, rem, target in zip(
        arrays["observations"], arrays["remaining"], arrays["returns"]
    ):
        aliases.setdefault((obs.tobytes(), int(rem)), []).append(float(target))
    repeated = [values for values in aliases.values() if len(values) > 1]
    return dict(
        arrays,
        baseline_suffix=suffix,
        endpoint_trace_indices=ids.tolist(),
        support={
            "environment_seed": int(trace["baseline_environment_seed"][bi]),
            "root_episodes": 1,
            "baseline_rows": length,
            "candidate_endpoint_rows_before_dedup": int(mask.sum()),
            "unique_candidate_endpoints": len(ids),
            "baseline_positive_rewards": int(np.count_nonzero(rewards > 0)),
            "baseline_positive_targets": int(np.count_nonzero(suffix[:-1] > 0)),
            "endpoint_positive_targets": int(np.count_nonzero(endpoint_targets > 0)),
            "baseline_group_weight": 0.5,
            "endpoint_group_weight": 0.5,
            "repeated_input_groups": len(repeated),
            "maximum_repeated_input_target_span": max(
                (max(v) - min(v) for v in repeated), default=0.0
            ),
            "data_sha256": array_digest(arrays),
        },
    )


def mean(values):
    present = [float(v) for v in values if v is not None]
    return float(np.mean(present)) if present else None


def summarize_cases(cases: list[dict]) -> dict:
    result = {
        "snapshots": len(cases),
        "empty_feasible_sets": sum(
            c["decisions"]["fixed_feasible"]["empty"] for c in cases
        ),
        "scores": {},
    }
    for name in SCORES:
        by_policy = {}
        for policy in ("unrestricted", "fixed_feasible"):
            rows = [c["decisions"][policy]["scores"][name] for c in cases]
            by_policy[policy] = {
                key: mean([r[key] for r in rows])
                for key in (
                    "gain_mae",
                    "oracle_regret",
                    "chosen_real_gain",
                    "informative_pair_agreement",
                )
            }
            by_policy[policy].update(
                informative_pairs=sum(r["informative_pair_count"] for r in rows),
                agreement_pairs=sum(r["pair_agreement_count"] for r in rows),
                nonempty_decisions=sum(r["chosen_index"] is not None for r in rows),
            )
        result["scores"][name] = by_policy
    return result


def evaluate_head(
    trace, head, data, *, split_id: int, horizon=5, maximum=1000, gamma=0.99
):
    mask = (
        (trace["split_id"] == split_id)
        & (trace["reference_mode_id"] == 0)
        & (trace["plan_kind"] != 2)
    )
    ids = np.flatnonzero(mask)
    if len(ids) != 64:
        raise ValueError("expected exactly 64 selected-mode candidate rows per split")
    if not np.all(trace["real_length"][ids] == maximum - trace["episode_step"][ids]):
        raise ValueError("evaluation continuation disagrees with remaining time")
    remaining = maximum - trace["episode_step"][ids] - horizon
    if not np.all(trace["real_continuations"][ids, :horizon] == 1):
        raise ValueError("unsupported terminated prefix")
    prediction = predict_head(
        head, trace["real_stage_observations"][ids, horizon], remaining
    ).astype(np.float64)
    true = trace["real_terminal"][ids] / gamma**horizon
    critic = trace["real_terminal_critic"][ids] / gamma**horizon
    time_only = data["baseline_suffix"][maximum - remaining]
    endpoints = {}
    for name, values in (
        ("planning_head", prediction),
        ("real_endpoint_critic", critic),
        ("time_only", time_only),
    ):
        error = values - true
        endpoints[name] = dict(
            mae=float(np.mean(abs(error))),
            mse=float(np.mean(error**2)),
            positive_target_count=int(np.count_nonzero(true > 0)),
            positive_target_mae=mean(abs(error[true > 0])),
            positive_target_mse=mean(error[true > 0] ** 2),
        )
    cap = (1 - gamma**remaining) / (1 - gamma)
    endpoints["planning_head"].update(
        zero_clipped_fraction=float(np.mean(prediction == 0)),
        upper_bound_fraction=float(np.mean(prediction >= cap - 1e-5)),
    )
    cases = []
    for family in range(2):
        for snapshot in sorted(set(trace["snapshot_id"][ids])):
            positions = np.flatnonzero(
                (trace["family_id"][ids] == family)
                & (trace["snapshot_id"][ids] == snapshot)
            )
            ix = ids[positions]
            if len(ix) != 8 or trace["plan_kind"][ix[0]] != 0:
                raise ValueError("invalid eight-candidate reference ordering")
            scores = dict(
                model=trace["model_objective"][ix],
                real_endpoint_critic=trace["real_stage"][ix]
                + trace["real_terminal_critic"][ix],
                planning_head=trace["real_stage"][ix]
                + gamma**horizon * prediction[positions],
                time_only=trace["real_stage"][ix]
                + gamma**horizon * time_only[positions],
                oracle=trace["real_objective"][ix],
            )
            cases.append(
                dict(
                    family=FAMILIES[family],
                    snapshot_id=int(snapshot),
                    episode_step=int(trace["episode_step"][ix[0]]),
                    environment_seed=int(trace["environment_seed"][ix[0]]),
                    candidate_trace_indices=ix.tolist(),
                    predicted_endpoint_values=prediction[positions].tolist(),
                    scores={k: v.tolist() for k, v in scores.items()},
                    decisions=analyze_candidates(
                        scores, scores["oracle"], trace["model_feasible"][ix]
                    ),
                )
            )
    baseline_ids = np.flatnonzero(trace["baseline_split_id"] == split_id)
    if len(baseline_ids) != 1:
        raise ValueError("expected one evaluation root episode")
    bi = int(baseline_ids[0])
    return dict(
        support=dict(
            environment_seed=int(trace["baseline_environment_seed"][bi]),
            root_episodes=1,
            candidate_rows=len(ids),
            positive_endpoint_targets=int(np.count_nonzero(true > 0)),
            baseline_positive_rewards=int(
                np.count_nonzero(trace["baseline_rewards"][bi] > 0)
            ),
        ),
        endpoint_errors=endpoints,
        cases=cases,
        families={
            family: summarize_cases([c for c in cases if c["family"] == family])
            for family in FAMILIES
        },
    )


def aggregate(cells: list[dict]) -> list[dict]:
    rows = []
    for task in dict.fromkeys(c["task"] for c in cells):
        selected = [c for c in cells if c["task"] == task]
        if len(selected) != 3:
            raise ValueError("expected three world models per task")
        for family in FAMILIES:
            item = dict(
                task=task,
                family=family,
                world_models=3,
                scores={},
                paired_world_deltas=[],
            )
            for score in SCORES:
                item["scores"][score] = {
                    metric: mean(
                        [
                            c["validation"]["families"][family]["scores"][score][
                                "unrestricted"
                            ][metric]
                            for c in selected
                        ]
                    )
                    for metric in (
                        "gain_mae",
                        "oracle_regret",
                        "chosen_real_gain",
                        "informative_pair_agreement",
                    )
                }
            for c in selected:
                s = c["validation"]["families"][family]["scores"]
                item["paired_world_deltas"].append(
                    dict(
                        world_model_seed=c["world_model_seed"],
                        gain_mae_head_minus_critic=s["planning_head"]["unrestricted"][
                            "gain_mae"
                        ]
                        - s["real_endpoint_critic"]["unrestricted"]["gain_mae"],
                        regret_head_minus_critic=s["planning_head"]["unrestricted"][
                            "oracle_regret"
                        ]
                        - s["real_endpoint_critic"]["unrestricted"]["oracle_regret"],
                    )
                )
            g = [r["gain_mae_head_minus_critic"] for r in item["paired_world_deltas"]]
            r = [r["regret_head_minus_critic"] for r in item["paired_world_deltas"]]
            item.update(
                gain_mae_improving_worlds=sum(v < 0 for v in g),
                regret_improving_worlds=sum(v < 0 for v in r),
                mean_gain_mae_delta=mean(g),
                mean_regret_delta=mean(r),
                promising_ranking_diagnostic=mean(g) < 0
                and mean(r) < 0
                and sum(v < 0 for v in g) >= 2
                and sum(v < 0 for v in r) >= 2,
                empty_feasible_sets=sum(
                    c["validation"]["families"][family]["empty_feasible_sets"]
                    for c in selected
                ),
            )
            rows.append(item)
    return rows


def compute(study):
    settings = HeadSettings()
    cells, heads, runtimes = [], [], []
    for record, trace in zip(study.records, study.traces):
        if record["actor_seed"] != 541:
            raise ValueError("unexpected frozen actor")
        data = extract_training(trace)
        if data["support"]["environment_seed"] != 76001:
            raise ValueError("unexpected training seed")
        validation_seeds = set(
            trace["environment_seed"][trace["split_id"] == 1].tolist()
        )
        if validation_seeds != {76003}:
            raise ValueError("unexpected or overlapping validation seed")
        start = time.perf_counter()
        head = fit_head(
            data["observations"],
            data["remaining"],
            data["returns"],
            sample_weights=data["weights"],
            seed=88000 + record["index"],
            settings=settings,
        )
        train_prediction = predict_head(head, data["observations"], data["remaining"])
        cell = {
            k: record[k]
            for k in (
                "index",
                "task",
                "world_model_seed",
                "actor_seed",
                "checkpoint_sha256",
                "trace_sha256",
            )
        }
        cell.update(
            training_support=data["support"],
            head_sha256=digest(head),
            training_weighted_mse=float(
                np.sum(data["weights"] * (train_prediction - data["returns"]) ** 2)
            ),
            calibration=evaluate_head(trace, head, data, split_id=0),
            validation=evaluate_head(trace, head, data, split_id=1),
        )
        cells.append(cell)
        heads.append(head)
        runtimes.append(time.perf_counter() - start)
        print(f"PLANNING_VALUE_HEAD_FITTED index={record['index']}", flush=True)
    return (
        dict(
            schema=SCHEMA,
            dependency_source_commit=SOURCE_COMMIT,
            dependency_report_sha256=REPORT_SHA256,
            dependency_manifest_sha256=MANIFEST_SHA256,
            settings=asdict(settings),
            number_of_heads=len(heads),
            train_environment_seed=76001,
            validation_environment_seed=76003,
            statistical_unit="three frozen world models per fixed task; one root episode per split/checkpoint; no independent suffix/candidate observations",
            interpretation="exploratory real-endpoint score test on reused diagnostics, not blind confirmation or controller performance",
            cells=cells,
            task_family_summaries=aggregate(cells),
        ),
        heads,
        runtimes,
    )


def publish(output: Path, report: dict, heads: list, runtimes: list, identity: dict):
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    values = {
        "report.json": dict(report, source=identity),
        "runtime.json": dict(
            cpu_backend_wall_seconds_per_head=runtimes,
            total_cpu_backend_wall_seconds=sum(runtimes),
        ),
        **{f"head-{i:03}.json": head for i, head in enumerate(heads)},
    }
    for name, value in values.items():
        data = canonical(value) + b"\n"
        with (output / name).open("xb") as handle:
            handle.write(data)
        files[name] = hashlib.sha256(data).hexdigest()
    with (output / "manifest.json").open("xb") as handle:
        handle.write(
            canonical(dict(schema=SCHEMA, source=identity, files=files)) + b"\n"
        )


def validate_output_snapshot(
    output: Path, manifest: dict, manifest_bytes: bytes
) -> None:
    if (output / "manifest.json").read_bytes() != manifest_bytes:
        raise ValueError("output manifest changed during verification")
    for name, sha in manifest["files"].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"output hash mismatch: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.artifacts.resolve()):
        raise ValueError("output cannot be inside immutable inputs")
    identity = source_identity()
    study = authenticate_artifacts(args.artifacts)
    if args.mode == "create":
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists; no overwrite or retry")
        report, heads, runtimes = compute(study)
        # Reauthentication after fitting catches concurrent input changes.
        authenticate_artifacts(args.artifacts)
        if source_identity() != identity:
            raise ValueError("source changed during fitting")
        publish(args.output, report, heads, runtimes, identity)
        print("PLANNING_VALUE_PILOT_CREATED")
    else:
        manifest_bytes = (args.output / "manifest.json").read_bytes()
        manifest = parse_json(manifest_bytes)
        expected_files = {"report.json", "runtime.json"} | {
            f"head-{i:03}.json" for i in range(9)
        }
        if (
            manifest["schema"] != SCHEMA
            or manifest["source"] != identity
            or set(manifest["files"]) != expected_files
        ):
            raise ValueError("output manifest/source/file set differs")
        validate_output_snapshot(args.output, manifest, manifest_bytes)
        report, heads, _ = compute(study)
        if canonical(dict(report, source=identity)) != canonical(
            read_json(args.output / "report.json")
        ):
            raise ValueError("independent scientific-report replay differs")
        for i, head in enumerate(heads):
            if canonical(head) != canonical(
                read_json(args.output / f"head-{i:03}.json")
            ):
                raise ValueError(f"independent head replay differs: {i}")
        authenticate_artifacts(args.artifacts)
        if source_identity() != identity:
            raise ValueError("source changed during verification")
        validate_output_snapshot(args.output, manifest, manifest_bytes)
        marker = dict(
            source=identity,
            strict_cpu_refit=True,
            heads=9,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            report_sha256=manifest["files"]["report.json"],
        )
        path = args.output / "verified.json"
        if path.exists():
            if read_json(path) != marker:
                raise ValueError("existing strict marker differs")
        else:
            with path.open("xb") as handle:
                handle.write(canonical(marker) + b"\n")
        print("PLANNING_VALUE_PILOT_STRICTLY_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
