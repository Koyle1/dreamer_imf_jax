"""Canonical immutable scheduler chain for the posterior-reward online experiment.

Scientific work is submitted only after exact COMPLETED 0:0 accounting and
authenticated predecessor markers. CPU afterok handoffs also attest final job
completion. Existing intent, failure, receipt or result always prevents retry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

from dreamer_imf_compare import posterior_reward_protocol as protocol

NEXT = {"preflight": "cells", "cells": "probes", "probes": "finalize"}
PREVIOUS = {
    "preflight": None,
    "cells": "preflight",
    "probes": "cells",
    "finalize": "probes",
}


def terminal(job, count=1):
    if (
        not isinstance(job, str)
        or not re.fullmatch(r"[1-9][0-9]*", job)
        or count not in (1, 9)
    ):
        raise ValueError("unregistered scheduler identity or array size")
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
        raise RuntimeError("predecessor is still active: " + active)
    raw = subprocess.check_output(
        [
            "sacct",
            "-X",
            "--array",
            "-j",
            job,
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode,ElapsedRaw,Start,End,AllocTRES",
        ],
        text=True,
    )
    rows = [
        line.strip().rstrip("|").split("|") for line in raw.splitlines() if line.strip()
    ]
    expected = {job} if count == 1 else {f"{job}_{i}" for i in range(count)}
    if (
        len(rows) != count
        or any(len(row) != 7 for row in rows)
        or {row[0] for row in rows} != expected
        or any(row[1:3] != ["COMPLETED", "0:0"] for row in rows)
        or any(
            not row[3].isdigit()
            or row[4] in ("", "Unknown")
            or row[5] in ("", "Unknown")
            for row in rows
        )
    ):
        raise RuntimeError("accounting is not exactly terminal COMPLETED 0:0: " + raw)
    return dict(
        job=job,
        expected_rows=count,
        raw=raw,
        raw_sha256=protocol.object_sha256(raw),
        records=rows,
    )


def _receipt(root, stage):
    manifest = protocol.read(root / "manifest.json")
    protocol.submission_reference(root, stage, manifest)
    return protocol.read(root / "submissions" / f"{stage}.json")


def _launch(root, stage, command, prerequisite=None):
    manifest = protocol.read(root / "manifest.json")
    intent = root / "submissions" / f"{stage}-intent.json"
    receipt = root / "submissions" / f"{stage}.json"
    if receipt.exists():
        raise FileExistsError("submission receipt exists; never retry automatically")
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
            raise RuntimeError("ambiguous sbatch response: " + response)
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
    print("POSTERIOR_REWARD_SUBMITTED", stage, response, flush=True)
    return response


def _markers(stage):
    return [f"cell-{i}" for i in range(9)] if stage == "cells" else [stage]


def attest(output, stage):
    """Persist completed scheduler allocation accounting separately from markers."""
    root = Path(output).resolve()
    if stage not in PREVIOUS:
        raise ValueError("unregistered attestation stage")
    manifest, _ = protocol.authenticate(root)
    receipt = _receipt(root, stage)
    accounting = terminal(receipt["job"], 9 if stage == "cells" else 1)
    evidence = {
        name: protocol.require_marker(root, name, authenticate_source=False)[
            "marker_sha256"
        ]
        for name in _markers(stage)
    }
    value = dict(
        stage=stage,
        manifest_sha256=manifest["manifest_sha256"],
        source_commit=manifest["source_commit"],
        accounting=accounting,
        markers=evidence,
        receipt_sha256=protocol.sha(root / "submissions" / f"{stage}.json"),
    )
    value["attestation_sha256"] = protocol.object_sha256(value)
    path = root / "attestations" / f"{stage}.json"
    if path.exists():
        # Reinspection may reuse exactly the same terminal accounting; it never
        # creates another job and will fail closed if scheduler evidence changed.
        if protocol.read(path) != value:
            raise ValueError("immutable terminal scheduler attestation changed")
    else:
        protocol.write_json_exclusive(path, value)
    print("POSTERIOR_REWARD_SCHEDULER_ATTESTED", stage, flush=True)
    return value


def _args(manifest, root, stage):
    return [
        str(
            Path(manifest["source"])
            / "dreamer_imf_comparison/scripts/posterior_reward_job.sh"
        ),
        manifest["source"],
        str(root),
        stage,
        manifest["upstream"]["path"],
    ]


def submit(output, stage, continue_chain=False):
    root = Path(output).resolve()
    if stage not in PREVIOUS:
        raise ValueError("unregistered scientific stage")
    manifest, _ = protocol.authenticate(root)
    for name in (f"{stage}-intent.json", f"{stage}.json"):
        if (root / "submissions" / name).exists():
            raise FileExistsError(
                "existing attempt; never duplicate pending, running, valid or failed work"
            )
    if any((root / "markers" / f"{name}.json").exists() for name in _markers(stage)):
        raise FileExistsError("stage already has completion evidence")
    if stage == "cells":
        for i in range(9):
            with protocol.NativeStepBudget(
                protocol.cell_directory(root, i), i
            ) as budget:
                if budget.snapshot()["records"] != 1:
                    raise ValueError(
                        "cell has prior native-step charges or resets; no restart"
                    )
    previous = PREVIOUS[stage]
    prerequisite = attest(root, previous) if previous else None
    command = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=gpu-a30",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=" + ("08:00:00" if stage in ("cells", "probes") else "02:00:00"),
        f"--job-name=posterior-reward-{stage}",
        f"--output={root}/logs/{stage}-%A_%a.log",
    ]
    if stage == "cells":
        command.append("--array=0-8%3")
    command += _args(manifest, root, "cell" if stage == "cells" else stage)
    job = _launch(root, stage, command, prerequisite)
    if continue_chain:
        handoff(root, stage, job)
    return job


def handoff(output, stage, job):
    root = Path(output).resolve()
    if stage not in PREVIOUS or not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("invalid handoff identity")
    manifest, _ = protocol.authenticate(root)
    if _receipt(root, stage)["job"] != job:
        raise ValueError("handoff job differs from predecessor")
    command = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=cpu-zen3",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:30:00",
        f"--dependency=afterok:{job}",
        "--kill-on-invalid-dep=yes",
        f"--job-name=posterior-reward-gate-{stage}",
        f"--output={root}/logs/gate-{stage}-%j.log",
    ]
    command += _args(manifest, root, "advance-" + stage)
    return _launch(
        root,
        "handoff-" + stage,
        command,
        dict(
            stage=stage,
            job=job,
            receipt_sha256=protocol.sha(root / "submissions" / f"{stage}.json"),
        ),
    )


def advance(output, stage):
    if stage not in PREVIOUS:
        raise ValueError("unregistered handoff stage")
    attest(output, stage)
    if stage in NEXT:
        return submit(output, NEXT[stage], continue_chain=True)
    # Scientific final verification precedes the batch exit; this distinct CPU
    # process closes the remaining completion/exit-status gap after finalization.
    print("POSTERIOR_REWARD_STUDY_TERMINAL_VERIFIED", flush=True)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("init", *PREVIOUS, *("advance-" + s for s in PREVIOUS))
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--continue-chain", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--parent-cell")
    parser.add_argument("--dataset")
    parser.add_argument("--rmt-full")
    parser.add_argument("--upstream")
    parser.add_argument("--bundle")
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    if args.stage == "init":
        if not all(
            (
                args.source,
                args.parent_cell,
                args.dataset,
                args.rmt_full,
                args.bundle,
                args.bundle_sha256,
                args.expected_commit,
            )
        ):
            parser.error(
                "init requires source, parent-cell, dataset, rmt-full, bundle, bundle-sha256, expected-commit"
            )
        if (
            protocol.clean_commit(args.source) != args.expected_commit
            or protocol.sha(args.bundle) != args.bundle_sha256
        ):
            raise ValueError("source commit or deployment bundle digest differs")
        value = protocol.initialize_manifest(
            args.output,
            args.source,
            args.parent_cell,
            args.dataset,
            args.rmt_full,
            upstream=args.upstream,
            bundle=args.bundle,
        )
        print("POSTERIOR_REWARD_REGISTERED", value["manifest_sha256"], flush=True)
    elif args.stage.startswith("advance-"):
        advance(args.output, args.stage[8:])
    else:
        submit(args.output, args.stage, args.continue_chain)


if __name__ == "__main__":
    main()
