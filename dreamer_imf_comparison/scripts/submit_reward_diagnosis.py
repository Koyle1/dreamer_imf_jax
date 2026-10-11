"""Single immutable launch intent: an ambiguous scheduler response is not retried."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)


def submit(source, output, parent, correction, expected_commit, bundle, bundle_sha256):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        != expected_commit
    ):
        raise ValueError("source commit differs")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("dirty source")
    if hashlib.sha256(Path(bundle).read_bytes()).hexdigest() != bundle_sha256:
        raise ValueError("bundle hash differs")
    if (
        not (Path(parent) / "report.json").is_file()
        or not (Path(correction) / "verified.json").is_file()
    ):
        raise ValueError("required dependencies missing")
    # This reservation is exclusive, including when a job already terminated.
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
        "--job-name=reward-diagnosis-" + expected_commit[:7],
        "--output=" + str(output / "job-%j.log"),
        str(source / "dreamer_imf_comparison/scripts/reward_diagnosis_job.sh"),
        str(source),
        str(output),
        str(Path(parent).resolve()),
        str(Path(correction).resolve()),
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
        completed = subprocess.run(cmd, text=True, capture_output=True, check=True)
        response = completed.stdout.strip()
        if not re.fullmatch(r"[1-9][0-9]*", response):
            raise RuntimeError("ambiguous submission response: " + response)
        write(
            output / "submission.json",
            dict(
                job=response,
                stdout=completed.stdout,
                stderr=completed.stderr,
                submitted_unix=time.time(),
            ),
        )
        print(response, flush=True)
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
        "parent",
        "correction",
        "expected-commit",
        "bundle",
        "bundle-sha256",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    submit(
        a.source,
        a.output,
        a.parent,
        a.correction,
        a.expected_commit,
        a.bundle,
        a.bundle_sha256,
    )
