"""After-any stage gate: failures stop, authenticated successes advance once."""

import argparse
from pathlib import Path
import subprocess

from dreamer_imf_compare.parallel_collection import _write_json
from dreamer_imf_compare.parallel_study import authenticate, read, require, sha
from submit_parallel_study import submit, terminal


def advance(root, stage):
    root = Path(root)
    manifest, _ = authenticate(root)
    receipt = read(root / "submissions" / f"{stage}.json")
    accounting = terminal(receipt["job"], 3 if stage == "fit" else 1)
    for name in ([f"fit-{seed}" for seed in range(3)] if stage == "fit" else [stage]):
        require(root, name)
    _write_json(
        root / "submissions" / f"{stage}-completed.json",
        dict(stage=stage, job=receipt["job"], accounting=accounting),
    )
    if stage == "verify":
        stages = {
            name: read(root / "submissions" / f"{name}-completed.json")
            for name in ("preflight", "collect", "fit", "evaluate", "verify")
        }
        resources = {}
        for name, evidence in stages.items():
            for mark in (
                [f"fit-{seed}" for seed in range(3)] if name == "fit" else [name]
            ):
                require(root, mark)
            job = read(root / "submissions" / f"{name}.json")["job"]
            measured = terminal(job, 3 if name == "fit" else 1)
            if evidence != {"stage": name, "job": job, "accounting": measured}:
                raise ValueError("terminal completion attestation differs")
            resources[name] = subprocess.check_output(
                [
                    "sacct",
                    "-X",
                    "-j",
                    job,
                    "--noheader",
                    "--parsable2",
                    "--format=JobID,State,ExitCode,Start,End,ElapsedRaw,AllocTRES,NodeList",
                ],
                text=True,
            )
        _write_json(
            root / "completion.json",
            dict(
                source_commit=manifest["source_commit"],
                manifest_sha256=sha(root / "manifest.json"),
                report_sha256=sha(root / "report.json"),
                stages=stages,
                slurm_allocation_evidence=resources,
                completed=True,
                scientific_success_not_implied=True,
            ),
        )
        print("PARALLEL_TRAJECTORY_SCHEDULER_COMPLETION_VERIFIED", flush=True)
    else:
        following = {
            "preflight": "collect",
            "collect": "fit",
            "fit": "evaluate",
            "evaluate": "verify",
        }[stage]
        submit(root, following, continue_chain=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--completed-stage",
        choices=("preflight", "collect", "fit", "evaluate", "verify"),
        required=True,
    )
    args = parser.parse_args()
    advance(args.root, args.completed_stage)
