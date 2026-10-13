"""Exclusive bounded capacity-sweep submission; never auto-retry ambiguity."""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def submit(source, output, dependency, expected_commit, bundle, bundle_sha256):
    source, output, dependency = map(
        lambda p: Path(p).resolve(), (source, output, dependency)
    )
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != expected_commit
    ):
        raise ValueError("wrong source commit")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("dirty source")
    if hashlib.sha256(Path(bundle).read_bytes()).hexdigest() != bundle_sha256:
        raise ValueError("wrong bundle digest")
    marker = json.loads((dependency / "verified.json").read_text())
    if marker.get("exact_replay") is not True:
        raise ValueError("unverified dependency")
    output.mkdir(parents=True, exist_ok=False)
    cmd = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=gpu-a30",
        "--gres=gpu:1",
        "--cpus-per-task=4",
        "--mem=64G",
        "--time=02:00:00",
        "--job-name=reward-capacity-" + expected_commit[:7],
        "--output=" + str(output / "job-%j.log"),
        str(source / "dreamer_imf_comparison/scripts/reward_capacity_job.sh"),
        str(source),
        str(output),
        str(dependency),
        expected_commit,
    ]
    write(
        output / "intent.json",
        dict(
            command=cmd,
            source_commit=expected_commit,
            bundle=str(Path(bundle).resolve()),
            bundle_sha256=bundle_sha256,
            created_unix=time.time(),
        ),
    )
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        job = result.stdout.strip()
        if not re.fullmatch(r"[1-9][0-9]*", job):
            raise RuntimeError("ambiguous scheduler response: " + job)
        write(
            output / "submission.json",
            dict(
                job=job,
                stdout=result.stdout,
                stderr=result.stderr,
                submitted_unix=time.time(),
            ),
        )
        print(job, flush=True)
    except BaseException as exc:
        write(
            output / "submission-uncertain.json",
            dict(error=repr(exc), automatic_retry_allowed=False),
        )
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in (
        "source",
        "output",
        "dependency",
        "expected-commit",
        "bundle",
        "bundle-sha256",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    submit(
        a.source, a.output, a.dependency, a.expected_commit, a.bundle, a.bundle_sha256
    )
