"""Independent NumPy accounting and metric reconstruction, never agent.train."""

from pathlib import Path
import pickle
import re
import subprocess

import numpy as np

from .posterior_reward_protocol import (
    authenticate,
    read,
    sha,
    write_json_exclusive as write,
    cell_for_index,
    cell_directory,
    eval_seeds,
    NativeStepBudget,
    write_marker,
    require_marker,
)
from .reward_exploration_verify import load_arrays, reward_metrics

MILESTONES = (0, 40000, 80000)


def optimizer_clocks(saved):
    result = {}
    for name, key in (
        ("original", "opt/step/value"),
        ("auxiliary", "reward_aux_opt/step/value"),
    ):
        value = np.asarray(saved["params"][key])
        if (
            value.shape != ()
            or not np.isfinite(value)
            or float(value) != int(value)
            or int(value) < 0
        ):
            raise ValueError("invalid optimizer clock: " + key)
        result[name] = int(value)
    return result


def verify_episode(data):
    n = 500
    if data["reward"].shape != (n + 1,) or not np.isin(data["reward"], [0, 1, 2]).all():
        raise ValueError("evaluation reward/length invalid")
    if (
        not data["is_first"][0]
        or data["is_first"][1:].any()
        or not data["is_last"][-1]
        or data["is_last"][:-1].any()
        or data["reward"][0] != 0
    ):
        raise ValueError("evaluation reset/terminal boundary invalid")
    if np.any(np.abs(data["action"]) > 1) or not np.array_equal(
        data["previous_action"][1:], data["action"][:-1]
    ):
        raise ValueError("executed action/reward indexing mismatch")
    result = dict(
        return_=float(data["reward"].sum(dtype=np.float64)),
        native_steps=1000,
        positive_decisions=int((data["reward"][1:] > 0).sum()),
        action_saturation=float(np.mean(np.abs(data["action"][:-1]) >= 0.99)),
        reward_errors={
            name: reward_metrics(
                data["reward"][1:], data["log/" + name + "_reward"][1:]
            )
            for name in ("original", "student", "teacher", "control")
        },
    )
    result["return"] = result.pop("return_")
    result["logged_diagnostics"] = {
        k[4:]: float(v[1:].mean())
        for k, v in data.items()
        if k.startswith("log/") and v.ndim == 1
    }
    return result


def verify_collection(directory):
    files = sorted((Path(directory) / "transitions").glob("chunk-*.npz"))
    if not files:
        raise ValueError("missing raw collection")
    chunks = [load_arrays(p) for p in files]
    keys = set(chunks[0])
    if any(set(c) != keys for c in chunks):
        raise ValueError("raw collection schema changed")
    data = {k: np.concatenate([x[k] for x in chunks]) for k in keys}
    if (
        not np.isin(data["reward"], [0, 1, 2]).all()
        or np.any(np.abs(data["action"]) > 1)
        or np.any(data["collection_mode"] != 0)
        or set(np.unique(data["worker"])) != set(range(16))
    ):
        raise ValueError("collection label/action/intervention invalid")
    resets = positives = decisions = steps = 0
    for worker in range(16):
        mask = data["worker"] == worker
        d = {k: v[mask] for k, v in data.items()}
        if not len(d["reward"]) or not d["is_first"][0] or not d["is_last"][-1]:
            raise ValueError("unclosed collection worker")
        expected = np.where(d["is_first"], 0, 2)
        if not np.array_equal(expected, d["native_delta"]) or expected.sum() != 5000:
            raise ValueError("native/reset accounting invalid")
        previous = np.zeros(2, np.float32)
        episode = -1
        for i in range(len(expected)):
            if d["is_first"][i]:
                if i and not d["is_last"][i - 1]:
                    raise ValueError("reset before terminal")
                episode += 1
                previous = np.zeros(2, np.float32)
                if d["reward"][i] != 0:
                    raise ValueError("reset carries reward")
            elif i and d["is_last"][i - 1]:
                raise ValueError("terminal without reset")
            if d["episode_id"][i] != worker * 100000 + episode:
                raise ValueError("episode identity mismatch")
            if not np.array_equal(d["previous_action"][i], previous):
                raise ValueError("incoming action misaligned")
            if not d["is_last"][i]:
                if not np.array_equal(
                    d["action"][i], np.clip(d["proposed_action"][i], -1, 1)
                ):
                    raise ValueError("unregistered action replacement")
                decisions += 1
            elif np.any(d["action"][i]):
                raise ValueError("terminal action must be zero")
            previous = d["action"][i]
        steps += int(expected.sum())
        resets += int(d["is_first"].sum())
        positives += int((d["reward"] > 0).sum())
    if steps != 80000 or decisions != 40000:
        raise ValueError("collection budget mismatch")
    return (
        dict(
            native_steps=steps,
            decisions=decisions,
            rows=len(data["reward"]),
            reset_rows=resets,
            positive_decisions=positives,
        ),
        files,
    )


def cell_summary(output, index):
    cell = cell_for_index(index)
    directory = cell_directory(output, index)
    result = read(directory / "result.json")
    if any(result.get(k) != v for k, v in cell.items()):
        raise ValueError("cell identity mismatch")
    collection, files = verify_collection(directory)
    with NativeStepBudget(directory, index) as budget:
        snapshot = budget.snapshot()
    if snapshot != result["budget"] or snapshot["total_native_steps"] != 110000:
        raise ValueError("native budget mismatch")
    if (
        result["native_collection_steps"] != 80000
        or result["training_decisions"] != 40000
        or result["replay_rows"] != collection["rows"]
        or result["reset_rows"] != collection["reset_rows"]
        or result["collection_modes"] != dict(task=40000, random=0, explore=0)
    ):
        raise ValueError("collection result counters differ")
    first = result["first_update_at"]
    if type(first) is not int or not 0 <= first < 40000:
        raise ValueError("invalid replay learner start")
    if result["learner_updates"] != (40000 - first) // 2:
        raise ValueError("learner update budget mismatch")
    initial = read(directory / "initialization.json")
    evaluations = {}
    for milestone in MILESTONES:
        record = read(directory / f"evaluation-{milestone}.json")
        if len(record["episodes"]) != 10 or record["milestone"] != milestone:
            raise ValueError("evaluation count/milestone mismatch")
        expected = eval_seeds(cell["seed"], milestone)
        episodes = []
        for ep, row in enumerate(record["episodes"]):
            path = directory / f"eval-{milestone}-{ep}.npz"
            if (
                row["seed"] != expected[ep]
                or row["path"] != path.name
                or row["sha256"] != sha(path)
            ):
                raise ValueError("evaluation binding mismatch")
            metric = verify_episode(load_arrays(path))
            if row["return_"] != metric["return"] or row["native_steps"] != 1000:
                raise ValueError("evaluation return/cost mismatch")
            episodes.append(dict(seed=expected[ep], **metric))
            files.append(path)
        mean = float(np.mean([v["return"] for v in episodes]))
        if mean != record["mean_return"]:
            raise ValueError("aggregate return mismatch")
        evaluations[str(milestone)] = dict(
            mean_return=mean,
            episodes=episodes,
            policy_latency_seconds=record["policy_latency_seconds"],
        )
        files.append(directory / f"evaluation-{milestone}.json")
    if result["evaluations"] != [
        read(directory / f"evaluation-{m}.json") for m in MILESTONES
    ]:
        raise ValueError("retained evaluation records differ")
    if [v["native_steps"] for v in result["checkpoints"]] != list(MILESTONES):
        raise ValueError("checkpoint grid mismatch")
    base_norms, initial_clocks = None, None
    for record in result["checkpoints"]:
        milestone = record["native_steps"]
        path = directory / f"checkpoint-{milestone}.pkl"
        if record["path"] != path.name or record["sha256"] != sha(path):
            raise ValueError("checkpoint binding changed")
        with path.open("rb") as stream:
            saved = pickle.load(stream)
        for value in saved["params"].values():
            if not np.isfinite(np.asarray(value)).all():
                raise ValueError("nonfinite checkpoint")
        expected = max(0, (milestone // 2 - first) // 2)
        clocks = optimizer_clocks(saved)
        if initial_clocks is None:
            initial_clocks = clocks
            if clocks["auxiliary"] != 0:
                raise ValueError("auxiliary Adam state was not freshly initialized")
        if any(clocks[k] - initial_clocks[k] != expected for k in clocks):
            raise ValueError("actual original/auxiliary optimizer clocks disagree")
        if (
            saved["counters"]["updates"] - initial["binding"]["counters"]["updates"]
            != expected
        ):
            raise ValueError("checkpoint optimizer clock mismatch")
        norms = {
            k: v
            for k, v in saved["params"].items()
            if k.startswith(("reward_teacher/", "reward_student/"))
            and k.rsplit("/", 1)[-1] in ("input_mean", "input_std")
        }
        if len(norms) != 4:
            raise ValueError("normalization state missing")
        if base_norms is None:
            base_norms = norms
            from .parallel_frozen import parameter_digest

            binding = initial["binding"]
            if (
                parameter_digest(
                    {k: saved["params"][k] for k in binding["original_keys"]}
                )
                != binding["parent_parameter_digest"]
            ):
                raise ValueError("initial parent optimizer/parameters changed")
            if (
                parameter_digest(
                    {k: saved["params"][k] for k in binding["imported_auxiliary_keys"]}
                )
                != binding["auxiliary_parameter_digest"]
            ):
                raise ValueError("initial auxiliary import changed")
        elif any(not np.array_equal(norms[k], v) for k, v in base_norms.items()):
            raise ValueError("fixed normalization changed")
        files.append(path)
    files.extend(
        [
            directory / "result.json",
            directory / "initialization.json",
            directory / "config.yaml",
            directory / "started.json",
            directory / "budget-genesis.json",
            directory / "native-budget.jsonl",
        ]
    )
    files.extend(sorted(directory.glob("progress-*.json")))
    files.extend(sorted(directory.glob("batch-*.npz")))
    return (
        dict(
            **cell,
            collection=collection,
            evaluations=evaluations,
            learner_updates=result["learner_updates"],
            runtime=result["runtime"],
            update_timings=result["update_timings"],
            compilation=result["compilation"],
            successful_path_seconds=result["wall_seconds"],
            initialization=initial["binding"],
        ),
        files,
    )


def verify_cell(output, index):
    authenticate(output)
    require_marker(output, "preflight")
    summary, files = cell_summary(output, index)
    path = cell_directory(output, index) / "verified_summary.json"
    write(path, summary)
    write_marker(
        output, f"cell-{index}", files + [path], details=dict(native_steps=110000)
    )
    print("POSTERIOR_REWARD_CELL_VERIFIED", index, flush=True)


def verify_preflight(output):
    authenticate(output)
    files, bindings = [], []
    for arm in ("A", "B", "C"):
        directory = Path(output) / "preflight" / arm
        result, replay = read(directory / "result.json"), read(
            directory / "replayed.json"
        )
        if (
            result["arm"] != arm
            or result["native_steps"] != 0
            or result["updates"] != 2
            or result["full_batch_shape"] != [16, 65]
            or not replay["exact"]
            or replay["native_steps"] != 0
            or replay["process_id"] == result["process_id"]
            or replay["checkpoint_sha256"] != result["checkpoint_sha256"]
            or replay["checkpoint_sha256"] != sha(directory / "checkpoint.pkl")
            or replay["probe_sha256"] != sha(directory / "probe.npz")
            or replay["parameter_digest"] != result["parameter_digest"]
        ):
            raise ValueError("preflight did not authenticate")
        bindings.append(result["parent_binding"])
        before, after = (
            result["optimizer_clocks_before"],
            result["optimizer_clocks_after"],
        )
        if (
            set(before) != {"original", "auxiliary"}
            or set(after) != set(before)
            or before["auxiliary"] != 0
            or any(after[k] - before[k] != 2 for k in before)
        ):
            raise ValueError("preflight optimizer clock mismatch")
        files.extend(sorted(p for p in directory.iterdir() if p.is_file()))
    if any(b != bindings[0] for b in bindings[1:]):
        raise ValueError("paired initial state differs across reward routes")
    write_marker(
        output, "preflight", files, details=dict(native_steps=0, arms=["A", "B", "C"])
    )
    print("POSTERIOR_REWARD_PREFLIGHT_VERIFIED", flush=True)


def probe_records(output):
    from .posterior_reward_metrics import summarize_arrays

    records, files = [], []
    for index in range(9):
        require_marker(output, f"cell-{index}", authenticate_source=False)
        target = cell_for_index(index)
        directory = Path(output) / "probes" / target["cell_id"]
        created, replayed = read(directory / "created.json"), read(
            directory / "replayed.json"
        )
        if (
            created["records"] != replayed["records"]
            or not replayed["exact"]
            or created["native_steps"] != 0
            or replayed["native_steps"] != 0
            or created["process_id"] == replayed["process_id"]
            or len(created["records"]) != 9
        ):
            raise ValueError("independent probe replay not authenticated")
        seen = set()
        for rec in created["records"]:
            milestone, source_arm = rec["milestone"], rec["source_arm"]
            identity = (milestone, source_arm)
            if (
                identity in seen
                or milestone not in MILESTONES
                or source_arm not in ("A", "B", "C")
            ):
                raise ValueError("duplicate/unregistered probe")
            seen.add(identity)
            path = directory / f"m{milestone}-source{source_arm}.npz"
            if (
                rec["path"] != path.name
                or rec["sha256"] != sha(path)
                or rec["seed"] != target["seed"]
                or rec["target_arm"] != target["arm"]
                or rec["checkpoint_sha256"]
                != sha(cell_directory(output, index) / f"checkpoint-{milestone}.pkl")
            ):
                raise ValueError("probe/checkpoint binding changed")
            source_index = ("A", "B", "C").index(source_arm) * 3 + index % 3
            source_dir = cell_directory(output, source_index)
            rows = []
            if len(rec["inputs"]) != 10:
                raise ValueError("probe input count")
            for ep, item in enumerate(rec["inputs"]):
                inp = source_dir / f"eval-{milestone}-{ep}.npz"
                if item != dict(path=str(inp), sha256=sha(inp)):
                    raise ValueError("probe input binding changed")
                rows.append(load_arrays(inp))
            arrays = load_arrays(path)
            expected_seed = int(
                np.random.SeedSequence(
                    [20261014, target["seed"], milestone, source_index // 3, 332]
                ).generate_state(1)[0]
            )
            if (
                rec["prediction_seed"] != expected_seed
                or int(arrays["probe_seed"]) != expected_seed
                or not np.array_equal(
                    arrays["episode_seed"],
                    np.repeat(eval_seeds(target["seed"], milestone), 4),
                )
            ):
                raise ValueError("probe RNG/environment grouping mismatch")
            truth = np.stack(
                [
                    row["reward"][t + 1 : t + 16]
                    for row in rows
                    for t in (100, 200, 300, 400)
                ]
            )
            action = np.stack(
                [
                    row["action"][t : t + 15]
                    for row in rows
                    for t in (100, 200, 300, 400)
                ]
            )
            obs = np.stack(
                [
                    np.concatenate(
                        [
                            row[k][t + 1 : t + 16]
                            for k in ("position", "to_target", "velocity")
                        ],
                        -1,
                    )
                    for row in rows
                    for t in (100, 200, 300, 400)
                ]
            )
            for key, value in (
                ("truth_reward", truth),
                ("action", action),
                ("truth_observation", obs),
            ):
                if not np.array_equal(arrays[key], value):
                    raise ValueError("probe action/reward indexing differs: " + key)
            records.append(
                dict(
                    arm=target["arm"],
                    seed=target["seed"],
                    milestone=milestone,
                    source_arm=source_arm,
                    summary=summarize_arrays(arrays),
                )
            )
            files.append(path)
        files.extend([directory / "created.json", directory / "replayed.json"])
    return records, files


def verify_probes(output):
    authenticate(output)
    records, files = probe_records(output)
    path = Path(output) / "probes" / "summary.json"
    write(path, records)
    write_marker(
        output, "probes", files + [path], details=dict(native_steps=0, probe_count=81)
    )
    print("POSTERIOR_REWARD_PROBES_VERIFIED", flush=True)


def allocation_runtime(output):
    records = []
    for stage in ("preflight", "cells", "probes"):
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
        lines = [row.strip().split("|") for row in text.splitlines() if row.strip()]
        expected = {f"{job}_{i}" for i in range(9)} if stage == "cells" else {job}
        if (
            len(lines) != len(expected)
            or {v[0] for v in lines} != expected
            or any(v[1:3] != ["COMPLETED", "0:0"] for v in lines)
        ):
            raise ValueError("scheduler scientific-stage completion failed")
        records.extend(
            dict(
                stage=stage,
                job=v[0],
                state=v[1],
                exit_code=v[2],
                elapsed_seconds=int(v[3]),
                allocation=v[4],
            )
            for v in lines
        )
    return dict(
        records=records,
        allocated_wall_seconds=sum(v["elapsed_seconds"] for v in records),
        exclusions="CPU handoffs and current finalizer; terminal finalizer attested by afterok handoff",
    )


def build_report(output):
    from .posterior_reward_metrics import paired_report

    manifest, protocol = authenticate(output)
    require_marker(output, "probes")
    cells = [cell_summary(output, i)[0] for i in range(9)]
    for seed in (901, 902, 903):
        pair = [c["initialization"] for c in cells if c["seed"] == seed]
        if any(p != pair[0] for p in pair[1:]):
            raise ValueError("unmatched continuation initializations")
    probes, _ = probe_records(output)
    if probes != read(Path(output) / "probes" / "summary.json"):
        raise ValueError("independent probe metrics differ")
    probe_runtime = {}
    for index in range(9):
        directory = Path(output) / "probes" / cell_for_index(index)["cell_id"]
        probe_runtime[cell_for_index(index)["cell_id"]] = {
            role: {
                k: read(directory / f"{role}.json")[k]
                for k in ("seconds", "compilation")
            }
            for role in ("created", "replayed")
        }
    return dict(
        schema=protocol["schema"],
        source_commit=manifest["source_commit"],
        protocol=protocol,
        cells=cells,
        analysis=paired_report(cells, probes),
        allocation_runtime=allocation_runtime(output),
        probe_runtime=probe_runtime,
        total_successful_native_steps=990000,
        limitations=[
            "one pretrained world-model seed; three paired continuations, not independent world-model seeds",
            "posterior teacher privileged information is training-only",
            "recorded feedback actions are not causal open-loop interventions",
            "offline initialization uses previously inspected exploratory data",
            "pure prediction replay is exact; asynchronous training is not claimed bitwise reproducible",
            "original native reward loss is not directly comparable to three-class categorical NLL",
            "engineering screen is not statistical proof or permission for another experiment",
        ],
    )


def finalize(output):
    report = build_report(output)
    write(Path(output) / "report.json", report)
    print("POSTERIOR_REWARD_REPORT_CREATED", flush=True)


def verify_final(output):
    if build_report(output) != read(Path(output) / "report.json"):
        raise ValueError("independent final report mismatch")
    write_marker(
        output,
        "finalize",
        [Path(output) / "report.json"],
        details=dict(native_steps=990000, scientific_result_may_be_negative=True),
    )
    print("POSTERIOR_REWARD_ONLINE_FINAL_VERIFIED", flush=True)
