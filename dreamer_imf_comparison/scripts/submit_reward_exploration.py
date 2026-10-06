"""Fail-closed, write-once submission for the frozen continuation pilot.

An ambiguous sbatch response retains its intent and failure receipt. This tool
never guesses whether a job ran, never retries a cell, and never resets a budget.
CPU handoffs can wait for predecessor completion, but scientific stages are
submitted only after completed 0:0 accounting and authenticated result markers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

from dreamer_imf_compare import reward_exploration_protocol as protocol

NEXT = {"match": "preflight", "preflight": "cells", "cells": "finalize"}
PREVIOUS = {
    "match": None,
    "preflight": "match",
    "cells": "preflight",
    "finalize": "cells",
}


def terminal(job, count=1):
    if (
        not isinstance(job, str)
        or not re.fullmatch(r"[1-9][0-9]*", job)
        or count not in (1, 12)
    ):
        raise ValueError("unregistered scheduler identity or array size")
    # squeue is authoritative for still-active allocations; sacct may lag.
    try:
        active = subprocess.check_output(
            ["squeue", "--noheader", "--array", "--jobs", job, "--format=%i|%T"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as exc:
        if "Invalid job id specified" not in (exc.stderr or ""):
            raise
        active = ""
    if active:
        raise RuntimeError("predecessor is still pending/running: " + active)
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "--array",
            "-j",
            job,
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode",
        ],
        text=True,
    )
    rows = [line.strip().split("|")[:3] for line in output.splitlines() if line.strip()]
    expected = {job} if count == 1 else {f"{job}_{index}" for index in range(12)}
    if (
        len(rows) != count
        or any(len(row) != 3 for row in rows)
        or {row[0] for row in rows} != expected
        or any(state != "COMPLETED" or code != "0:0" for _, state, code in rows)
    ):
        raise RuntimeError(
            "predecessor accounting not exactly COMPLETED 0:0: " + output
        )
    return dict(
        job=job,
        expected_rows=count,
        raw=output,
        raw_sha256=protocol.object_sha256(output),
        records=rows,
    )


def _receipt(root, stage):
    value = protocol.read(root / "submissions" / f"{stage}.json")
    intent = root / "submissions" / f"{stage}-intent.json"
    manifest = protocol.read(root / "manifest.json")
    if (
        value.get("stage") != stage
        or value.get("status") != "submitted"
        or value.get("manifest_sha256") != manifest["manifest_sha256"]
        or value.get("intent_sha256") != protocol.sha(intent)
    ):
        raise ValueError("invalid predecessor submission receipt")
    intent_value = protocol.read(intent)
    if (
        intent_value["manifest_sha256"] != manifest["manifest_sha256"]
        or intent_value["stage"] != stage
        or intent_value["command"] != value["command"]
    ):
        raise ValueError("submission intent and receipt do not agree")
    return value


def _launch(root, stage, command, prerequisite=None):
    manifest = protocol.read(root / "manifest.json")
    intent = root / "submissions" / f"{stage}-intent.json"
    receipt = root / "submissions" / f"{stage}.json"
    # link-based exclusive publication is the cross-process submission lock.
    if receipt.exists():
        raise FileExistsError("submission receipt exists; no automatic retry")
    protocol.write_json_exclusive(
        intent,
        dict(
            stage=stage,
            manifest_sha256=manifest["manifest_sha256"],
            source_commit=manifest["source_commit"],
            command=command,
            prerequisite=prerequisite,
        ),
    )
    record = dict(
        stage=stage,
        manifest_sha256=manifest["manifest_sha256"],
        intent_sha256=protocol.sha(intent),
        command=command,
    )
    try:
        response = subprocess.check_output(command, text=True).strip()
        if not re.fullmatch(r"[1-9][0-9]*", response):
            raise RuntimeError("ambiguous or nonlocal sbatch job identity: " + response)
    except BaseException as exc:
        protocol.write_json_exclusive(
            receipt,
            dict(
                record,
                status="ambiguous_submission",
                exception_type=type(exc).__name__,
                message=str(exc),
            ),
        )
        raise
    protocol.write_json_exclusive(
        receipt, dict(record, status="submitted", job=response)
    )
    print("REWARD_EXPLORATION_SUBMITTED", stage, response, flush=True)
    return response


def _args(manifest, root, stage):
    source = Path(manifest["source"])
    return [
        str(source / "dreamer_imf_comparison/scripts/reward_exploration_job.sh"),
        str(source),
        str(root),
        stage,
        manifest["inputs"]["parent_cell"],
        manifest["inputs"]["dataset"]["path"],
    ]


def submit(output, stage, continue_chain=False):
    root = Path(output).resolve()
    if stage not in PREVIOUS:
        raise ValueError("unregistered scientific stage")
    manifest, _ = protocol.authenticate(root)
    for name in (f"{stage}-intent.json", f"{stage}.json"):
        if (root / "submissions" / name).exists():
            raise FileExistsError(
                "existing attempt; never duplicate valid, pending, running or failed jobs"
            )
    markers = (
        [f"cell-{index:02d}" for index in range(12)] if stage == "cells" else [stage]
    )
    if any((root / "markers" / f"{name}.json").exists() for name in markers):
        raise FileExistsError("stage already has completion evidence; never overwrite")
    if stage == "cells":
        for index in range(12):
            with protocol.NativeStepBudget(
                protocol.cell_directory(root, index), index
            ) as budget:
                if budget.snapshot()["records"] != 1:
                    raise ValueError(
                        "cell already attempted; preserve all charges and evidence"
                    )
    previous = PREVIOUS[stage]
    prerequisite = None
    if previous:
        predecessor = _receipt(root, previous)
        accounting = terminal(predecessor["job"], 12 if previous == "cells" else 1)
        names = (
            [f"cell-{index:02d}" for index in range(12)]
            if previous == "cells"
            else [previous]
        )
        evidence = {
            name: protocol.require_marker(root, name, authenticate_source=False)[
                "marker_sha256"
            ]
            for name in names
        }
        prerequisite = dict(
            stage=previous,
            accounting=accounting,
            markers=evidence,
            receipt_sha256=protocol.sha(root / "submissions" / f"{previous}.json"),
        )
    time_limit = "08:00:00" if stage == "cells" else "02:00:00"
    command = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=gpu-a30",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        f"--time={time_limit}",
        f"--job-name=reward-explore-{stage}",
        f"--output={root}/logs/{stage}-%A_%a.log",
    ]
    if stage == "cells":
        command.append("--array=0-11%4")
    command += _args(manifest, root, "cell" if stage == "cells" else stage)
    job = _launch(root, stage, command, prerequisite)
    if continue_chain and stage in NEXT:
        handoff(root, stage, job)
    return job


def handoff(output, stage, job):
    root = Path(output).resolve()
    if stage not in NEXT or not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("invalid handoff identity")
    manifest, _ = protocol.authenticate(root)
    receipt = _receipt(root, stage)
    if receipt["job"] != job:
        raise ValueError("handoff predecessor receipt mismatch")
    command = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=cpu-zen3",
        "--cpus-per-task=2",
        "--mem=4G",
        "--time=00:30:00",
        f"--dependency=afterany:{job}",
        "--kill-on-invalid-dep=yes",
        f"--job-name=reward-explore-gate-{stage}",
        f"--output={root}/logs/gate-{stage}-%j.log",
    ]
    command += _args(manifest, root, f"advance-{stage}")
    return _launch(
        root,
        f"handoff-{stage}",
        command,
        dict(
            stage=stage,
            job=job,
            receipt_sha256=protocol.sha(root / "submissions" / f"{stage}.json"),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=(*PREVIOUS, *(f"advance-{stage}" for stage in NEXT))
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--continue-chain", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage.startswith("advance-"):
        submit(arguments.output, NEXT[arguments.stage[8:]], continue_chain=True)
    else:
        submit(arguments.output, arguments.stage, arguments.continue_chain)


if __name__ == "__main__":
    main()
