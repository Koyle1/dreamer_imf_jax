"""Independent NumPy accounting and outcome reconstruction for the pilot.

Does not import the agent, runner, loss code, or production metric functions.
Reported intrinsic rewards are never added to environment returns.
"""

from pathlib import Path
import pickle
import re
import subprocess

import numpy as np

from .reward_exploration_protocol import (
    authenticate,
    cell_for_index,
    cell_directory,
    collection_policy,
    eval_seeds,
    NativeStepBudget,
    read,
    sha,
    write_json_exclusive,
    write_marker,
    require_marker,
)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as f:
        result = {k: np.array(f[k]) for k in f.files}
    for key, value in result.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError("nonfinite retained field " + key)
    return result


def reward_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, np.float64), np.asarray(
        prediction, np.float64
    )
    if truth.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid reward prediction")
    result = dict(
        count=int(truth.size),
        mse=float(np.mean((prediction - truth) ** 2)),
        bias=float(np.mean(prediction - truth)),
        positive_count=int((truth > 0).sum()),
    )
    for name, mask in (("positive", truth > 0), ("zero", truth == 0)):
        result[name] = dict(
            count=int(mask.sum()),
            mse=(
                float(np.mean((prediction[mask] - truth[mask]) ** 2))
                if mask.any()
                else None
            ),
        )
    # Nonoverlapping within-episode windows: no crossing reset boundaries.
    n = len(truth) // 15
    if n:
        delta = (prediction[: 15 * n] - truth[: 15 * n]).reshape(n, 15).sum(-1)
        result["cumulative15_mse_raw"] = float(np.mean(delta**2))
        result["cumulative15_windows"] = n
    return result


def probe_metrics(data, scale):
    if data["truth_reward"].shape != (20, 15) or data["truth_observation"].shape != (
        20,
        15,
        6,
    ):
        raise ValueError("invalid probe target shape")
    if not np.array_equal(
        data["episode"], np.repeat(np.arange(5), 4)
    ) or not np.array_equal(data["anchor"], np.tile([100, 200, 300, 400], 5)):
        raise ValueError("probe clustering/anchor identity")
    result = {
        "recomputed": True,
        "causal_intervention": False,
        "particles": 8,
        "episodes": 5,
        "horizons": {},
    }
    truth = data["truth_reward"].astype(np.float64)
    for h in (1, 3, 5, 10, 15):
        row = {}
        for key in (
            "posterior_control",
            "posterior_original",
            "predicted_control",
            "predicted_original",
        ):
            p = data[key].astype(np.float64)
            if key.startswith("predicted_"):
                if p.shape != (20, 8, 15):
                    raise ValueError("invalid imagined reward shape")
                p = p.mean(1)
            if p.shape != (20, 15):
                raise ValueError("invalid posterior reward shape")
            cumulative = (p[:, :h] - truth[:, :h]).sum(-1)
            row[key] = dict(
                step_mse=float(np.mean((p[:, h - 1] - truth[:, h - 1]) ** 2)),
                cumulative_mse_raw=float(np.mean(cumulative**2)),
                cumulative_mse_normalized=float(np.mean((cumulative / scale) ** 2)),
            )
        obs = data["predicted_observation"].astype(np.float64)
        if obs.shape != (20, 8, 15, 6):
            raise ValueError("invalid imagined observation shape")
        row["observation_mse"] = float(
            np.mean(
                (obs[:, :, h - 1].mean(1) - data["truth_observation"][:, h - 1]) ** 2
            )
        )
        result["horizons"][str(h)] = row
    means = data["ensemble_means"].astype(np.float64)
    if means.shape != (5, 20, 15, 6):
        raise ValueError("invalid ensemble mean shape")
    target = (data["truth_observation"] - data["observation_mean"]) / data[
        "observation_std"
    ]
    error = np.mean((means.mean(0) - target) ** 2, -1).reshape(-1)
    uncertainty = np.mean(np.var(means, axis=0), -1).reshape(-1)
    correlation = (
        float(np.corrcoef(error, uncertainty)[0, 1])
        if error.std() > 0 and uncertainty.std() > 0
        else None
    )
    result["ensemble"] = dict(
        mean_next_observation_mse_normalized=float(error.mean()),
        mean_disagreement=float(uncertainty.mean()),
        error_disagreement_correlation=correlation,
    )
    positive_episodes = sum(
        bool((truth[data["episode"] == ep] > 0).any()) for ep in range(5)
    )
    result.update(
        positive_reward_episodes=positive_episodes,
        reward_sensitive_conclusion=(
            "inconclusive: fewer than five positive episodes"
            if positive_episodes < 5
            else "exploratory coverage threshold met"
        ),
    )
    return result


def verify_episode(data, expected_native=1000):
    n = expected_native // 2
    if data["reward"].shape != (n + 1,) or not np.isin(data["reward"], [0, 1, 2]).all():
        raise ValueError("evaluation reward/length semantics differ")
    if (
        not (data["is_first"][0] and data["is_last"][-1])
        or data["is_first"][1:].any()
        or data["is_last"][:-1].any()
    ):
        raise ValueError("evaluation reset/end boundaries invalid")
    if data["reward"][0] != 0 or np.any(np.abs(data["action"]) > 1):
        raise ValueError("invalid reset reward or unclipped action")
    if not np.array_equal(data["previous_action"][1:], data["action"][:-1]):
        raise ValueError("executed action/reward indexing mismatch")
    result = {
        "return": float(data["reward"].sum(dtype=np.float64)),
        "native_steps": 2 * n,
        "positive_decisions": int((data["reward"][1:] > 0).sum()),
        "action_saturation": float(np.mean(np.abs(data["action"][:-1]) >= 0.99)),
    }
    for label, key in (
        ("control", "log/control_reward"),
        ("original", "log/original_reward"),
    ):
        result[label] = reward_metrics(data["reward"][1:], data[key][1:])
    for key in (
        "log/disagreement",
        "log/action_entropy",
        "log/feature_rms",
        "log/feature_zscore_rms",
    ):
        result[key[4:] + "_mean"] = float(data[key][1:].mean())
    return result


def verify_raw_transitions(directory, cell):
    files = sorted((directory / "transitions").glob("chunk-*.npz"))
    if not files:
        raise ValueError("missing complete retained collection")
    chunks = [load_arrays(path) for path in files]
    keys = set(chunks[0])
    if any(set(c) != keys for c in chunks):
        raise ValueError("collection chunk schema differs")
    data = {k: np.concatenate([c[k] for c in chunks]) for k in keys}
    if not np.isin(data["reward"], [0, 1, 2]).all() or np.any(
        np.abs(data["action"]) > 1
    ):
        raise ValueError("collection label/action bounds invalid")
    steps = resets = 0
    counts = dict(task=0, random=0, explore=0)
    positives = 0
    for worker in range(16):
        ix = np.flatnonzero(data["worker"] == worker)
        if not len(ix):
            raise ValueError("missing worker")
        d = {k: v[ix] for k, v in data.items()}
        if not d["is_first"][0] or not d["is_last"][-1]:
            raise ValueError("unclosed collection worker")
        expected_delta = np.where(d["is_first"], 0, 2)
        if not np.array_equal(d["native_delta"], expected_delta):
            raise ValueError("reset counted as simulator transition")
        if int(expected_delta.sum()) != 5000:
            raise ValueError("worker native steps mismatch")
        decision_index = 0
        previous = np.zeros(2, np.float32)
        previous_episode = -1
        for i in range(len(ix)):
            if d["is_first"][i]:
                previous = np.zeros(2, np.float32)
                previous_episode += 1
                if d["reward"][i] != 0:
                    raise ValueError("nonzero reset reward")
            if d["episode_id"][i] != worker * 100000 + previous_episode:
                raise ValueError("unstable bootstrap episode identity")
            if not np.array_equal(d["previous_action"][i], previous):
                raise ValueError("stored incoming executed action differs")
            mode = collection_policy(cell["arm"], cell["seed"], decision_index)
            if d["collection_mode"][i] != {"task": 0, "random": 1, "explore": 2}[mode]:
                raise ValueError("exploration block schedule mismatch")
            if not d["is_last"][i]:
                counts[mode] += 1
                if mode == "random":
                    rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [cell["seed"], 759, worker, decision_index]
                        )
                    )
                    expected = rng.uniform(-1, 1, 2).astype(np.float32)
                else:
                    expected = np.clip(d["proposed_action"][i], -1, 1)
                if not np.array_equal(expected, d["action"][i]):
                    raise ValueError(
                        "actual collection action differs from registered intervention"
                    )
                decision_index += 1
            previous = d["action"][i]
        steps += int(expected_delta.sum())
        resets += int(d["is_first"].sum())
        positives += int((d["reward"] > 0).sum())
    expected_counts = dict(task=40000, random=0, explore=0)
    if cell["arm"] in ("C", "D"):
        expected_counts["task"] = 32000
        expected_counts["random" if cell["arm"] == "C" else "explore"] = 8000
    if counts != expected_counts or steps != 80000:
        raise ValueError("unmatched exploration fraction or total budget")
    return (
        dict(
            native_steps=steps,
            reset_rows=resets,
            rows=len(data["reward"]),
            collection_modes=counts,
            positive_decisions=positives,
        ),
        files,
    )


def cell_summary(output, index):
    """Recompute all evidence without trusting aggregate result values."""
    cell = cell_for_index(index)
    directory = cell_directory(output, index)
    result = read(directory / "result.json")
    if any(result.get(k) != v for k, v in cell.items()):
        raise ValueError("wrong cell identity")
    collection, files = verify_raw_transitions(directory, cell)
    with NativeStepBudget(directory, index) as budget:
        snapshot = budget.snapshot()
    if snapshot != result["budget"] or snapshot["total_native_steps"] != 90000:
        raise ValueError("budget chain or successful-path expenditure differs")
    if (
        result["training_decisions"] != 40000
        or result["native_collection_steps"] != 80000
        or result["replay_rows"] != collection["rows"]
    ):
        raise ValueError("reported collection counters differ")
    if (
        result["reset_rows"] != collection["reset_rows"]
        or result["collection_modes"] != collection["collection_modes"]
    ):
        raise ValueError("reported reset/intervention counters differ")
    if result["learner_updates"] != (40000 - result["first_update_at"]) // 2:
        raise ValueError("unequal train ratio or incorrect learner clock")
    evaluations = {}
    scale = read(Path(output) / "match" / "selection.json")["scale"]
    for milestone in (40000, 80000):
        record = read(directory / f"evaluation-{milestone}.json")
        episodes = []
        expected_seeds = eval_seeds(cell["seed"], milestone)
        if len(record["episodes"]) != 5:
            raise ValueError("evaluation episode count")
        for ep, row in enumerate(record["episodes"]):
            path = directory / f"eval-{milestone}-{ep}.npz"
            if (
                row["seed"] != expected_seeds[ep]
                or row["path"] != path.name
                or row["sha256"] != sha(path)
            ):
                raise ValueError("raw evaluation identity/hash mismatch")
            metrics = verify_episode(load_arrays(path))
            if (
                row["return_"] != metrics["return"]
                or row["native_steps"] != metrics["native_steps"]
            ):
                raise ValueError("reported environment return mismatch")
            episodes.append(metrics)
            files.append(path)
        mean = float(np.mean([x["return"] for x in episodes]))
        if mean != record["mean_return"]:
            raise ValueError("reported mean return mismatch")
        evaluations[str(milestone)] = dict(mean_return=mean, episodes=episodes)
        files.append(directory / f"evaluation-{milestone}.json")
        probe_path = directory / f"prediction-probe-{milestone}.npz"
        if record["recomputed_probe"] != {
            "path": probe_path.name,
            "sha256": sha(probe_path),
            "native_steps": 0,
        }:
            raise ValueError("recomputed probe artifact binding")
        probe = load_arrays(probe_path)
        # Independently recover action/target indexing from raw held-out episodes.
        truth_rows = [
            load_arrays(directory / f"eval-{milestone}-{ep}.npz") for ep in range(5)
        ]
        expected_reward = np.stack(
            [
                r["reward"][t + 1 : t + 16]
                for r in truth_rows
                for t in (100, 200, 300, 400)
            ]
        )
        expected_action = np.stack(
            [r["action"][t : t + 15] for r in truth_rows for t in (100, 200, 300, 400)]
        )
        expected_obs = np.stack(
            [
                np.concatenate(
                    [
                        r[k][t + 1 : t + 16]
                        for k in ("position", "to_target", "velocity")
                    ],
                    -1,
                )
                for r in truth_rows
                for t in (100, 200, 300, 400)
            ]
        )
        for key, expected in (
            ("truth_reward", expected_reward),
            ("truth_observation", expected_obs),
            ("action", expected_action),
        ):
            if not np.array_equal(probe[key], expected):
                raise ValueError("probe indexing mismatch: " + key)
        evaluations[str(milestone)]["prediction_probe"] = probe_metrics(probe, scale)
        files.append(probe_path)
    initial = read(directory / "initialization.json")
    parent_updates = initial["binding"]["counters"]["updates"]
    for ckpt in result["checkpoints"]:
        path = directory / ckpt["path"]
        if sha(path) != ckpt["sha256"]:
            raise ValueError("checkpoint hash mismatch")
        with path.open("rb") as stream:
            saved = pickle.load(stream)
        for value in saved["params"].values():
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError("nonfinite checkpoint state")
        expected = (ckpt["native_steps"] // 2 - result["first_update_at"]) // 2
        if saved["counters"]["updates"] - parent_updates != expected:
            raise ValueError("checkpoint update clock mismatch")
        files.append(path)
    if sorted(c["native_steps"] for c in result["checkpoints"]) != [40000, 80000]:
        raise ValueError("missing checkpoint")
    files.extend(
        [
            directory / "result.json",
            directory / "initialization.json",
            directory / "config.yaml",
        ]
    )
    replay_path = directory / "prediction_replay.json"
    if read(replay_path) != {
        "exact": True,
        "milestones": [40000, 80000],
        "native_steps": 0,
    }:
        raise ValueError("prediction replay evidence missing")
    files.append(replay_path)
    return (
        dict(
            **cell,
            collection=collection,
            evaluations=evaluations,
            learner_updates=result["learner_updates"],
            successful_path_seconds=result["wall_seconds"],
            update_timings=result["update_timings"],
            continuation_not_independent_world_seed=True,
            parameter_and_runtime_evidence=result["runtime"],
        ),
        files,
    )


def verify_cell(output, index):
    authenticate(output)
    require_marker(output, "preflight")
    summary, files = cell_summary(output, index)
    path = cell_directory(output, index) / "verified_summary.json"
    write_json_exclusive(path, summary)
    write_marker(
        output, f"cell-{index:02d}", files + [path], details={"native_steps": 90000}
    )
    print("REWARD_EXPLORATION_CELL_VERIFIED", index, flush=True)


def verify_preflight(output):
    authenticate(output)
    require_marker(output, "match")
    files = []
    for arm in ("A", "D"):
        directory = Path(output) / "preflight" / arm
        result = read(directory / "result.json")
        if (
            result["updates"] != 2
            or result["native_steps"] != 0
            or result["full_batch_shape"] != [16, 65]
        ):
            raise ValueError("invalid full-size no-interaction preflight")
        for group in ("enc", "dyn", "rew", "control_rew", "explore_pol", "explore_val"):
            if result["parameter_deltas"][group]["changed"] <= 0:
                raise ValueError("required parameter group did not learn")
        if not all(np.isfinite(v) for v in result["metrics"].values()):
            raise ValueError("nonfinite GPU preflight metrics")
        with (directory / "checkpoint.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        for value in saved["params"].values():
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError("nonfinite preflight checkpoint")
        probe = directory / "prediction-probe-40000.npz"
        with np.load(probe, allow_pickle=False) as arrays:
            if not arrays.files or any(
                not np.isfinite(arrays[key]).all() for key in arrays.files
            ):
                raise ValueError("invalid preflight prediction probe")
        files.extend(
            [
                directory / "result.json",
                directory / "checkpoint.pkl",
                directory / "config.yaml",
                probe,
            ]
        )
    write_marker(
        output, "preflight", files, details={"native_steps": 0, "full_size_updates": 4}
    )
    print("REWARD_EXPLORATION_PREFLIGHT_VERIFIED", flush=True)


def allocation_runtime(output):
    records = []
    for stage in ("match", "preflight", "cells"):
        receipt = read(Path(output) / "submissions" / f"{stage}.json")
        job = receipt["job"]
        if not re.fullmatch(r"[1-9][0-9]*", job):
            raise ValueError("invalid scheduler job")
        text = subprocess.check_output(
            [
                "sacct",
                "-X",
                "--array",
                "-j",
                job,
                "--noheader",
                "--parsable2",
                "--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES",
            ],
            text=True,
        )
        lines = [line.strip().split("|") for line in text.splitlines() if line.strip()]
        expected = {job} if stage != "cells" else {f"{job}_{i}" for i in range(12)}
        if (
            len(lines) != len(expected)
            or {r[0] for r in lines} != expected
            or any(r[1:3] != ["COMPLETED", "0:0"] for r in lines)
        ):
            raise ValueError(
                "scientific-stage scheduler completion did not authenticate"
            )
        records.extend(
            dict(
                stage=stage,
                job=row[0],
                state=row[1],
                exit_code=row[2],
                elapsed_seconds=int(row[3]),
                allocation=row[4],
            )
            for row in lines
        )
    return dict(
        records=records,
        allocated_wall_seconds=sum(r["elapsed_seconds"] for r in records),
        exclusions="CPU handoffs and this report/finalization allocation; no retries authorized",
    )


def build_report(output):
    cells = [cell_summary(output, i)[0] for i in range(12)]
    updates = {c["learner_updates"] for c in cells}
    if len(updates) != 1:
        raise ValueError("arms have unequal learner-update budgets")
    contrasts = {}
    for lhs, rhs in (("B", "A"), ("C", "B"), ("D", "C"), ("D", "B")):
        by_seed = []
        for seed in (701, 702, 703):
            a = next(c for c in cells if c["arm"] == lhs and c["seed"] == seed)
            b = next(c for c in cells if c["arm"] == rhs and c["seed"] == seed)
            by_seed.append(
                dict(
                    seed=seed,
                    delta=a["evaluations"]["80000"]["mean_return"]
                    - b["evaluations"]["80000"]["mean_return"],
                )
            )
        deltas = [r["delta"] for r in by_seed]
        contrasts[lhs + "-" + rhs] = dict(
            paired_continuation_seeds=by_seed,
            mean_delta=float(np.mean(deltas)),
            favorable_fraction=float(np.mean(np.asarray(deltas) > 0)),
        )
    return dict(
        cells=cells,
        contrasts=contrasts,
        total_native_steps=12 * 90000,
        successful_cell_path_seconds=sum(c["successful_path_seconds"] for c in cells),
        slurm_runtime=allocation_runtime(output),
        scientific_status="exploratory; one parent world-model seed; three paired continuation seeds",
        limitations=[
            "Original training replay unavailable; identical replay restart in all arms.",
            "Matched-data readout test episodes were examined previously.",
            "Ensemble disagreement is a proxy, not guaranteed epistemic uncertainty.",
            "Auxiliary ensemble/exploration learner compute is included in every arm.",
            "Posterior reward error is not a measure of imagined rollout accuracy.",
            "No claim of architectural superiority or generalization across tasks.",
        ],
    )


def finalize(output):
    authenticate(output)
    for i in range(12):
        require_marker(output, f"cell-{i:02d}")
    write_json_exclusive(Path(output) / "report.json", build_report(output))
    print("REWARD_EXPLORATION_REPORT_CREATED", flush=True)


def verify_final(output):
    authenticate(output)
    for i in range(12):
        require_marker(output, f"cell-{i:02d}")
    path = Path(output) / "report.json"
    if read(path) != build_report(output):
        raise ValueError("independently reconstructed report differs")
    write_marker(
        output,
        "finalize",
        [path],
        details={
            "native_steps": 1080000,
            "world_model_seeds": 1,
            "continuation_seeds": 3,
        },
    )
    print("REWARD_EXPLORATION_STUDY_FINAL_VERIFIED", flush=True)
