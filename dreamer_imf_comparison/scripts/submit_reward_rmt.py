"""Exclusive offline RMT diagnostic submission; ambiguity never auto-retries."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

CAPACITY_COMMIT = "f0b1be7f04e2fb1fcf2ecda58d68d6bed70dcad3"
CAPACITY_REPORT = "890b613ffcf49c143b9b760c711e38634f7461625d6fcda0542d55b5d6c31b16"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def validate_dependency(path):
    v, m = read(path / "verified.json"), read(path / "manifest.json")
    if (
        v.get("exact_replay") is not True
        or v.get("report_sha256") != CAPACITY_REPORT
        or digest(path / "report.json") != CAPACITY_REPORT
        or v.get("evaluated_sha256") != digest(path / "evaluated.json")
        or m.get("source_commit") != CAPACITY_COMMIT
        or m.get("preflight") is not False
    ):
        raise ValueError("wrong or unauthenticated capacity dependency")
    sealed = read(path / "evaluated.json")
    if sealed.get("stage") != "evaluated" or not sealed.get("artifacts"):
        raise ValueError("invalid capacity seal")
    for name, value in sealed["artifacts"].items():
        artifact = path / name
        if (
            not artifact.resolve().is_relative_to(path.resolve())
            or digest(artifact) != value
        ):
            raise ValueError("dependency artifact changed")


def submit(source, output, dependency, expected_commit, bundle, bundle_sha256):
    source, output, dependency, bundle = [
        Path(p).resolve() for p in (source, output, dependency, bundle)
    ]
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit) or not re.fullmatch(
        r"[0-9a-f]{64}", bundle_sha256
    ):
        raise ValueError("full source and bundle identities required")
    if output.is_relative_to(source) or output.is_relative_to(dependency):
        raise ValueError("output must be outside immutable source and dependency")
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
    if digest(bundle) != bundle_sha256:
        raise ValueError("wrong bundle digest")
    heads = subprocess.check_output(
        ["git", "bundle", "list-heads", str(bundle)], text=True
    ).splitlines()
    if expected_commit not in {line.split()[0] for line in heads}:
        raise ValueError("bundle does not bind expected commit")
    validate_dependency(dependency)
    output.mkdir(parents=True, exist_ok=False)
    cmd = [
        "sbatch",
        "--parsable",
        "--account=dep_inin_dat",
        "--partition=gpu-a30",
        "--gres=gpu:1",
        "--cpus-per-task=4",
        "--mem=64G",
        "--time=04:00:00",
        "--job-name=reward-rmt-" + expected_commit[:7],
        "--output=" + str(output / "job-%j.log"),
        str(source / "dreamer_imf_comparison/scripts/reward_rmt_job.sh"),
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
            bundle=str(bundle),
            bundle_sha256=bundle_sha256,
            dependency=str(dependency),
            dependency_marker_sha256=digest(dependency / "verified.json"),
            created_unix=time.time(),
            additional_simulator_steps=0,
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
        return job
    except BaseException as exc:
        write(
            output / "submission-uncertain.json",
            dict(error=repr(exc), automatic_retry_allowed=False),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "output",
        "dependency",
        "expected-commit",
        "bundle",
        "bundle-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    a = parser.parse_args()
    submit(
        a.source, a.output, a.dependency, a.expected_commit, a.bundle, a.bundle_sha256
    )
