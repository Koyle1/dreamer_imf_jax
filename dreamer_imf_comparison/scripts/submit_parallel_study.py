"""Canonical stage submitter: authenticate prior evidence and terminal Slurm state."""

import argparse
from pathlib import Path
import subprocess
from dreamer_imf_compare.parallel_study import authenticate, read, require, SOURCE
from dreamer_imf_compare.parallel_collection import _write_json


def terminal(job, count):
    if not str(job).isdigit() or count not in (1, 3):
        raise ValueError("unregistered scheduler identity/count")
    output = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-j",
            str(job),
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode",
        ],
        text=True,
    )
    rows = [line.split("|")[:3] for line in output.splitlines() if line.strip()]
    expected = {str(job)} if count == 1 else {f"{job}_{index}" for index in range(3)}
    if (
        len(rows) != count
        or {row[0] for row in rows} != expected
        or any(state != "COMPLETED" or code != "0:0" for _, state, code in rows)
    ):
        raise RuntimeError("predecessor not completed 0:0: " + output)
    return rows


def submit(root, stage, continue_chain=False):
    root = Path(root)
    m, p = authenticate(root)
    intent = root / "submissions" / f"{stage}-intent.json"
    receipt = root / "submissions" / f"{stage}.json"
    if intent.exists() or receipt.exists():
        raise FileExistsError(
            "existing submission evidence; inspect Slurm, never duplicate"
        )
    previous = {
        "preflight": None,
        "collect": "preflight",
        "fit": "collect",
        "evaluate": "fit",
        "verify": "evaluate",
    }[stage]
    attest = []
    if previous:
        last = read(root / "submissions" / f"{previous}.json")
        attest = terminal(last["job"], 3 if previous == "fit" else 1)
        for mark in (
            [f"fit-{s}" for s in range(3)] if previous == "fit" else [previous]
        ):
            require(root, mark)
    _write_json(
        intent,
        dict(
            stage=stage,
            source_commit=m["source_commit"],
            predecessor=previous,
            accounting=attest,
        ),
    )
    hours = {
        "preflight": "01:00:00",
        "collect": "04:00:00",
        "fit": "06:00:00",
        "evaluate": "04:00:00",
        "verify": "04:00:00",
    }[stage]
    command = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=gpu-a30",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        f"--time={hours}",
        f"--job-name=parallel-imf-{stage}",
        f"--output={root}/logs/{stage}-%A_%a.log",
    ]
    if stage == "fit":
        command += ["--array=0-2%3"]
    command += [
        str(SOURCE / "dreamer_imf_comparison/scripts/parallel_job.sh"),
        str(SOURCE),
        str(root),
        stage,
    ]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("unrecognized submission response")
    _write_json(receipt, dict(stage=stage, job=job, command=command))
    print("PARALLEL_STAGE_SUBMITTED", stage, job, flush=True)
    if continue_chain:
        handoff(root, stage, job)
    return job


def handoff(root, stage, job):
    """CPU-only after-any gate, not an eagerly submitted scientific stage."""
    root = Path(root)
    intent = root / "submissions" / f"handoff-{stage}-intent.json"
    receipt = root / "submissions" / f"handoff-{stage}.json"
    if intent.exists() or receipt.exists():
        raise FileExistsError("handoff already registered; never resubmit blindly")
    _write_json(intent, dict(stage=stage, predecessor_job=job))
    command = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=cpu-zen3",
        "--cpus-per-task=2",
        "--mem=4G",
        "--time=00:20:00",
        f"--dependency=afterany:{job}",
        f"--job-name=parallel-gate-{stage}",
        f"--output={root}/logs/gate-{stage}-%j.log",
        str(SOURCE / "dreamer_imf_comparison/scripts/parallel_handoff.sh"),
        str(SOURCE),
        str(root),
        stage,
    ]
    identity = subprocess.check_output(command, text=True).strip().split(";")[0]
    if not identity.isdigit():
        raise RuntimeError("unrecognized handoff response")
    _write_json(
        receipt, dict(stage=stage, job=identity, predecessor_job=job, command=command)
    )
    print("PARALLEL_HANDOFF_REGISTERED", stage, identity, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=("preflight", "collect", "fit", "evaluate", "verify")
    )
    p.add_argument("--root", required=True)
    p.add_argument("--continue-chain", action="store_true")
    a = p.parse_args()
    submit(a.root, a.stage, a.continue_chain)
